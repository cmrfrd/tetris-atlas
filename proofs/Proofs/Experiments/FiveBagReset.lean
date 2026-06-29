import Proofs.Safety.SafeSet
import Proofs.Invariants.GameplayExtra
import Proofs.Invariants.ColumnCount
import Proofs.Invariants.Holes
import Proofs.Model.PieceGeometry
import Proofs.Safety.SafeIterateFinite

/-!
# Experiment: five-bag reset

This file explores the following sufficient condition for solving canonical
Tetris:

* play five complete 7-bags, hence 35 tetrominoes;
* clear exactly 14 lines during those 35 moves;
* survive every intermediate state; and
* require the placements to be chosen online by a `Solver`.

The cell equation is

```
35 * 4 = 14 * 10 = 140.
```

Consequently, a valid 35-move trace with 14 line clears has zero occupied
cells. Bag legality independently implies that the bag is full again after
35 draws, so the terminal state is exactly `GameState.init`.

The main results define finite backward winning layers, extract one consistent
online solver from a depth-35 winning proof, and derive `TetrisSolvable`.
Concretely, `tetrisSolvable_of_init_mem_winningFinset` reduces the remaining
work to proving that the root belongs to a computed depth-35 winning layer.
That finite membership fact remains the experimental search/proof obligation.
-/

namespace Tetris
namespace Experiments
namespace FiveBagReset

open Board Placement

/-- Number of complete bags in the reset experiment. -/
def bagsPerBlock : ℕ := 5

/-- Number of pieces in one canonical bag. -/
def piecesPerBag : ℕ := 7

/-- Number of placements in one reset block. -/
def blockLength : ℕ := bagsPerBlock * piecesPerBag

/-- Required number of line clears in one reset block. -/
def targetLines : ℕ := 14

@[simp] theorem bagsPerBlock_eq : bagsPerBlock = 5 := rfl

@[simp] theorem piecesPerBag_eq : piecesPerBag = 7 := rfl

@[simp] theorem blockLength_eq : blockLength = 35 := rfl

@[simp] theorem targetLines_eq : targetLines = 14 := rfl

/-- The arithmetic behind the experiment: five bags contain exactly as many
cells as fourteen standard-width rows. -/
theorem block_cell_balance :
    4 * blockLength = GameConfig.standard.cols * targetLines := by
  norm_num [blockLength, bagsPerBlock, piecesPerBag, targetLines]

/-- Before the refill draw, the current bag loses exactly one element per
legal draw. Keeping the index symbolic avoids expensive numeral reduction of
`bagAt`. -/
theorem bagAt_card_eq_seven_sub {s : ℕ → Piece} (hl : LegalSequence s)
    (n : ℕ) (hn : n ≤ 6) :
    (bagAt Bag.full s n).card = 7 - n := by
  induction n with
  | zero =>
      rw [bagAt_zero, Bag.full_card]
  | succ n ih =>
      have hn_le_five : n ≤ 5 := by omega
      have ihcard : (bagAt Bag.full s n).card = 7 - n :=
        ih (by omega)
      have hge : 2 ≤ (bagAt Bag.full s n).card := by
        rw [ihcard]
        omega
      rw [bagAt_succ, Bag.card_draw_of_ge_two (hl n) hge, ihcard]
      omega

/-- A legal draw from a singleton bag triggers the canonical refill. -/
theorem bagAt_succ_eq_full_of_card_eq_one
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ)
    (hcard : (bagAt Bag.full s n).card = 1) :
    bagAt Bag.full s (n + 1) = Bag.full := by
  apply (Bag.card_eq_seven_iff_eq_full _).mp
  rw [bagAt_succ]
  exact Bag.card_draw_of_singleton (hl n) hcard

/-- Seven legal draws from a full bag return the bag to full. -/
theorem bagAt_seven_eq_full {s : ℕ → Piece} (hl : LegalSequence s) :
    bagAt Bag.full s 7 = Bag.full := by
  have c6 : (bagAt Bag.full s 6).card = 1 := by
    simpa using bagAt_card_eq_seven_sub hl 6 (by omega)
  simpa using bagAt_succ_eq_full_of_card_eq_one hl 6 c6

/-- Every complete group of seven legal draws ends at a full-bag boundary. -/
theorem bagAt_mul_seven_eq_full {s : ℕ → Piece} (hl : LegalSequence s)
    (bags : ℕ) :
    bagAt Bag.full s (7 * bags) = Bag.full := by
  induction bags with
  | zero => rfl
  | succ bags ih =>
      rw [Nat.mul_succ, bagAt_add, ih]
      apply bagAt_seven_eq_full
      have hsuffix := hl.suffix (7 * bags)
      rw [ih] at hsuffix
      exact hsuffix

/-- In particular, every legal five-bag prefix returns the bag to full. -/
theorem bagAt_blockLength_eq_full {s : ℕ → Piece} (hl : LegalSequence s) :
    bagAt Bag.full s blockLength = Bag.full := by
  have h := bagAt_mul_seven_eq_full hl bagsPerBlock
  norm_num [blockLength, bagsPerBlock, piecesPerBag] at h ⊢
  exact h

/-! ## Exact-depth reachability and phase recovery -/

/-- Reachability indexed by the exact number of pieces played. This is the
depth-aware version of `Reachable`, used to recover the five-bag phase from
the state itself. -/
inductive ReachableAt (cfg : GameConfig) : ℕ → GameState → Prop
  | init : ReachableAt cfg 0 GameState.init
  | step {n : ℕ} {g : GameState} {p : Piece} {pl : Placement} :
      ReachableAt cfg n g →
      ¬ g.lost cfg →
      p ∈ g.bag →
      ({ pl with piece := p } : Placement).Valid cfg →
      ReachableAt cfg (n + 1) (adversarialStep cfg g p pl)

namespace ReachableAt

/-- Exact-depth reachability implies ordinary reachability. -/
theorem reachable {cfg : GameConfig} {n : ℕ} {g : GameState}
    (h : ReachableAt cfg n g) : Reachable cfg g := by
  induction h with
  | init => exact Reachable.init
  | @step n g p pl _ hnl hp hvalid ih =>
      exact ih.adversarialStep hnl hp hvalid

/-- Exact-depth reachable states have well-formed boards. -/
theorem wf {cfg : GameConfig} {n : ℕ} {g : GameState}
    (h : ReachableAt cfg n g) : Board.WF cfg g.board :=
  h.reachable.wf

/-- Exact cell-count equation at a fixed depth: after `n` pieces, the current
cell count plus the cells removed by line clears equals `4 * n`. -/
theorem exists_count_eq {cfg : GameConfig} {n : ℕ} {g : GameState}
    (h : ReachableAt cfg n g) :
    ∃ k : ℕ, g.board.count + cfg.cols * k = 4 * n := by
  induction h with
  | init =>
      refine ⟨0, ?_⟩
      rw [GameState.init_board]
      change (∅ : Board).card + cfg.cols * 0 = 0
      simp
  | @step n g p pl hprev _hnl _hp hvalid ih =>
      obtain ⟨k, hk⟩ := ih
      let pl' : Placement := { pl with piece := p }
      have hstep := applyStep_count cfg g.board pl' hprev.wf hvalid
      refine ⟨k + Board.linesCleared cfg (pl'.place g.board), ?_⟩
      rw [adversarialStep_board]
      change (Placement.applyStep cfg g.board pl').count +
          cfg.cols * (k + Board.linesCleared cfg (pl'.place g.board)) =
        4 * (n + 1)
      rw [Nat.mul_add]
      omega

/-- Standard-board count residue at an exact depth. -/
theorem standard_count_mod_ten {n : ℕ} {g : GameState}
    (h : ReachableAt GameConfig.standard n g) :
    g.board.count % 10 = (4 * n) % 10 := by
  obtain ⟨k, hk⟩ := h.exists_count_eq
  have hmod := congrArg (fun x : ℕ => x % 10) hk
  norm_num [GameConfig.standard] at hmod
  simpa [Nat.add_mul_mod_self_left] using hmod

/-- Exact-depth bag cardinality. The phase inside the 7-bag is visible from
`g.bag.card`: `7` at a bag boundary, then `6, ..., 1`. -/
theorem bag_card_eq {cfg : GameConfig} {n : ℕ} {g : GameState}
    (h : ReachableAt cfg n g) : g.bag.card = 7 - n % 7 := by
  induction h with
  | init =>
      rw [GameState.init_bag, Bag.full_card]
  | @step n g p pl _ _hnl hp _hvalid ih =>
      rw [adversarialStep_bag]
      by_cases hphase : n % 7 = 6
      · have hcard_one : g.bag.card = 1 := by
          rw [ih]
          omega
        rw [Bag.card_draw_of_singleton hp hcard_one]
        omega
      · have hcard_ge_two : 2 ≤ g.bag.card := by
          have hlt : n % 7 < 7 := Nat.mod_lt n (by decide)
          rw [ih]
          omega
        rw [Bag.card_draw_of_ge_two hp hcard_ge_two]
        omega

/-- Equal exact-depth states have equal bag phases. -/
theorem mod_seven_eq_of_eq_state
    {m n : ℕ} {g : GameState}
    (hm : ReachableAt GameConfig.standard m g)
    (hn : ReachableAt GameConfig.standard n g) :
    m % 7 = n % 7 := by
  have hmc := hm.bag_card_eq
  have hnc := hn.bag_card_eq
  have hmlt : m % 7 < 7 := Nat.mod_lt m (by decide)
  have hnlt : n % 7 < 7 := Nat.mod_lt n (by decide)
  omega

/-- Equal exact-depth states have equal five-phase residues, derived from
the board-cell count modulo 10. -/
theorem mod_five_eq_of_eq_state
    {m n : ℕ} {g : GameState}
    (hm : ReachableAt GameConfig.standard m g)
    (hn : ReachableAt GameConfig.standard n g) :
    m % 5 = n % 5 := by
  have hmc := hm.standard_count_mod_ten
  have hnc := hn.standard_count_mod_ten
  omega

/-- The two visible residues, modulo 5 and modulo 7, determine any depth below
35. This is the small CRT fact needed by phase recovery. -/
theorem eq_of_mod_five_and_mod_seven_of_lt_block
    {m n : ℕ} (hm : m < blockLength) (hn : n < blockLength)
    (h5 : m % 5 = n % 5) (h7 : m % 7 = n % 7) : m = n := by
  norm_num [blockLength, bagsPerBlock, piecesPerBag] at hm hn
  omega

/-- Same state below the first five-bag boundary implies same depth. This is
the theorem that lets a layer-indexed strategy collapse to an online
state-based solver. -/
theorem depth_unique_lt_block
    {m n : ℕ} {g : GameState}
    (hmr : ReachableAt GameConfig.standard m g)
    (hnr : ReachableAt GameConfig.standard n g)
    (hm : m < blockLength) (hn : n < blockLength) : m = n :=
  eq_of_mod_five_and_mod_seven_of_lt_block hm hn
    (mod_five_eq_of_eq_state hmr hnr)
    (mod_seven_eq_of_eq_state hmr hnr)

end ReachableAt

/-! ## Backward winning layers -/

/-- `Winning n g` means that, starting from `g`, an online player can survive
the next `n` adversarial legal draws and finish exactly at the root state.

The quantifier order is important: the adversary chooses `p` first, then the
player chooses a valid placement for that announced piece. -/
def Winning : ℕ → GameState → Prop
  | 0, g => g = GameState.init
  | n + 1, g =>
      ¬ g.lost GameConfig.standard ∧
        ∀ p ∈ g.bag,
          ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
            Winning n (adversarialStep GameConfig.standard g p pl)

@[simp] theorem winning_zero_iff {g : GameState} :
    Winning 0 g ↔ g = GameState.init := by
  rfl

@[simp] theorem winning_succ_iff {n : ℕ} {g : GameState} :
    Winning (n + 1) g ↔
      ¬ g.lost GameConfig.standard ∧
        ∀ p ∈ g.bag,
          ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
            Winning n (adversarialStep GameConfig.standard g p pl) := by
  rfl

/-- A positive-horizon winning state has not already topped out. -/
theorem Winning.not_lost {n : ℕ} {g : GameState}
    (h : Winning (n + 1) g) :
    ¬ g.lost GameConfig.standard :=
  h.1

/-- Every legal next piece has a valid response leading to the preceding
winning layer. -/
theorem Winning.exists_step {n : ℕ} {g : GameState}
    (h : Winning (n + 1) g) {p : Piece} (hp : p ∈ g.bag) :
    ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
      Winning n (adversarialStep GameConfig.standard g p pl) :=
  h.2 p hp

/-- One-step robust predecessor inside a finite source universe. -/
noncomputable def resetPre
    (source target : Finset GameState) : Finset GameState :=
  source.filter fun g =>
    ¬ g.lost GameConfig.standard ∧
      ∀ p ∈ g.bag,
        ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
          adversarialStep GameConfig.standard g p pl ∈ target

@[simp] theorem mem_resetPre_iff
    {source target : Finset GameState} {g : GameState} :
    g ∈ resetPre source target ↔
      g ∈ source ∧
        ¬ g.lost GameConfig.standard ∧
          ∀ p ∈ g.bag,
            ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
              adversarialStep GameConfig.standard g p pl ∈ target := by
  classical
  simp [resetPre]

/-- Backward expansion never introduces states outside its source universe. -/
theorem resetPre_subset_source (source target : Finset GameState) :
    resetPre source target ⊆ source := by
  intro g hg
  exact (mem_resetPre_iff.mp hg).1

/-- Enlarging the target layer can only enlarge its robust predecessor. -/
theorem resetPre_mono_target {source a b : Finset GameState}
    (hab : a ⊆ b) :
    resetPre source a ⊆ resetPre source b := by
  intro g hg
  rw [mem_resetPre_iff] at hg ⊢
  refine ⟨hg.1, hg.2.1, ?_⟩
  intro p hp
  obtain ⟨pl, hpl, hnext⟩ := hg.2.2 p hp
  exact ⟨pl, hpl, hab hnext⟩

/-- The semantic winning predicate restricted to states in a finite universe.
The terminal root need not belong to `U`; every positive-depth state does. -/
def WinningIn (U : Finset GameState) : ℕ → GameState → Prop
  | 0, g => g = GameState.init
  | n + 1, g =>
      g ∈ U ∧
        ¬ g.lost GameConfig.standard ∧
          ∀ p ∈ g.bag,
            ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
              WinningIn U n (adversarialStep GameConfig.standard g p pl)

/-- Executable backward winning layers over a finite universe. -/
noncomputable def winningFinset (U : Finset GameState) : ℕ → Finset GameState
  | 0 => {GameState.init}
  | n + 1 => resetPre U (winningFinset U n)

/-- The finite backward computation exactly implements `WinningIn`. -/
theorem mem_winningFinset_iff
    (U : Finset GameState) (n : ℕ) (g : GameState) :
    g ∈ winningFinset U n ↔ WinningIn U n g := by
  induction n generalizing g with
  | zero =>
      classical
      simp [winningFinset, WinningIn]
  | succ n ih =>
      rw [winningFinset, mem_resetPre_iff]
      simp only [WinningIn]
      constructor
      · rintro ⟨hgU, hnl, hstep⟩
        refine ⟨hgU, hnl, ?_⟩
        intro p hp
        obtain ⟨pl, hpl, hnext⟩ := hstep p hp
        exact ⟨pl, hpl, (ih _).mp hnext⟩
      · rintro ⟨hgU, hnl, hstep⟩
        refine ⟨hgU, hnl, ?_⟩
        intro p hp
        obtain ⟨pl, hpl, hnext⟩ := hstep p hp
        exact ⟨pl, hpl, (ih _).mpr hnext⟩

/-- Restricting the search to a finite universe is sound: every computed
member is genuinely winning in the unrestricted adversarial game. -/
theorem WinningIn.winning {U : Finset GameState} {n : ℕ} {g : GameState}
    (h : WinningIn U n g) : Winning n g := by
  induction n generalizing g with
  | zero =>
      exact h
  | succ n ih =>
      refine ⟨h.2.1, ?_⟩
      intro p hp
      obtain ⟨pl, hpl, hnext⟩ := h.2.2 p hp
      exact ⟨pl, hpl, ih hnext⟩

/-- A successful finite backward computation discharges the mathematical
five-bag game obligation at the root. -/
theorem winning_of_init_mem_winningFinset
    {U : Finset GameState}
    (h : GameState.init ∈ winningFinset U blockLength) :
    Winning blockLength GameState.init :=
  (mem_winningFinset_iff U blockLength GameState.init).mp h |>.winning

/-! ## Extracting one online solver -/

/-- A move is globally eligible for the extracted solver when it is justified
by some reachable winning layer strictly inside the five-bag block. Exact-depth
uniqueness will show that any such witness has the current trace depth. -/
def GoodMove (g : GameState) (p : Piece) (pl : Placement) : Prop :=
  ∃ d : ℕ,
    d < blockLength ∧
      ReachableAt GameConfig.standard d g ∧
        Winning (blockLength - d) g ∧
          p ∈ g.bag ∧
            pl ∈ Placement.allValidFor GameConfig.standard p ∧
              Winning (blockLength - (d + 1))
                (adversarialStep GameConfig.standard g p pl)

/-- The state-based solver chooses one eligible backward-winning move whenever
one exists, and uses the always-valid canonical placement elsewhere. -/
noncomputable def winningSolver : Solver GameConfig.standard := by
  classical
  exact fun g p =>
    if h : ∃ pl : Placement, GoodMove g p pl then
      Classical.choose h
    else
      trivialSolver GameConfig.standard g p

/-- The choice made by `winningSolver` satisfies `GoodMove` whenever an
eligible move exists. -/
theorem winningSolver_goodMove {g : GameState} {p : Piece}
    (h : ∃ pl : Placement, GoodMove g p pl) :
    GoodMove g p (winningSolver g p) := by
  rw [winningSolver, dif_pos h]
  exact Classical.choose_spec h

/-- Every output of the extracted solver is a valid placement for the
announced piece, including states outside the certified winning layers. -/
theorem winningSolver_mem_allValidFor (g : GameState) (p : Piece) :
    winningSolver g p ∈ Placement.allValidFor GameConfig.standard p := by
  by_cases h : ∃ pl : Placement, GoodMove g p pl
  · exact (winningSolver_goodMove h).choose_spec.2.2.2.2.1
  · rw [winningSolver, dif_neg h]
    exact Placement.mk_zero_mem_allValidFor_standard p

/-- The extracted state-based strategy satisfies the global `ValidSolver`
contract. -/
theorem winningSolver_valid :
    ValidSolver GameConfig.standard winningSolver := by
  intro g p _hp
  exact (Placement.mem_allValidFor GameConfig.standard p _).mp
    (winningSolver_mem_allValidFor g p)

/-- Arithmetic decomposition of the positive remaining horizon. -/
theorem block_sub_eq_succ {d : ℕ} (hd : d < blockLength) :
    blockLength - d = (blockLength - (d + 1)) + 1 := by
  norm_num [blockLength, bagsPerBlock, piecesPerBag] at hd ⊢
  omega

/-- A reachable winning state with positive remaining horizon has an eligible
move for every legal next piece. -/
theorem exists_goodMove
    {d : ℕ} {g : GameState} {p : Piece}
    (hd : d < blockLength)
    (hreach : ReachableAt GameConfig.standard d g)
    (hwin : Winning (blockLength - d) g)
    (hp : p ∈ g.bag) :
    ∃ pl : Placement, GoodMove g p pl := by
  have hwinSucc :
      Winning ((blockLength - (d + 1)) + 1) g := by
    rw [← block_sub_eq_succ hd]
    exact hwin
  obtain ⟨pl, hpl, hnext⟩ := hwinSucc.exists_step hp
  exact ⟨pl, d, hd, hreach, hwin, hp, hpl, hnext⟩

/-- At an exact reachable depth, the state-based choice advances to exactly
the expected previous winning layer. A choice justified at any other depth is
ruled out by `ReachableAt.depth_unique_lt_block`. -/
theorem winningSolver_step_winning
    {d : ℕ} {g : GameState} {p : Piece}
    (hd : d < blockLength)
    (hreach : ReachableAt GameConfig.standard d g)
    (hwin : Winning (blockLength - d) g)
    (hp : p ∈ g.bag) :
    Winning (blockLength - (d + 1))
      (adversarialStep GameConfig.standard g p (winningSolver g p)) := by
  have hexists := exists_goodMove hd hreach hwin hp
  obtain ⟨d', hd', hreach', _hwin', _hp', _hpl', hnext⟩ :=
    winningSolver_goodMove hexists
  have hdepth : d' = d :=
    ReachableAt.depth_unique_lt_block hreach' hreach hd' hd
  subst d'
  exact hnext

/-- Along every legal sequence, the extracted strategy remains in the
corresponding exact-depth reachable and backward-winning layers. -/
theorem winningSolver_trace_reachable_and_winning
    (hroot : Winning blockLength GameState.init)
    {s : ℕ → Piece} (hl : LegalSequence s)
    (n : ℕ) (hn : n ≤ blockLength) :
    ReachableAt GameConfig.standard n
        (adversarialTrace GameConfig.standard winningSolver s GameState.init n) ∧
      Winning (blockLength - n)
        (adversarialTrace GameConfig.standard winningSolver s GameState.init n) := by
  induction n with
  | zero =>
      exact ⟨ReachableAt.init, hroot⟩
  | succ n ih =>
      have hnlt : n < blockLength := by omega
      have hprev := ih (by omega)
      let g :=
        adversarialTrace GameConfig.standard winningSolver s GameState.init n
      let p := s n
      have hp : p ∈ g.bag := by
        dsimp [g, p]
        rw [adversarialTrace_bag]
        exact hl n
      have hremaining :
          Winning ((blockLength - (n + 1)) + 1) g := by
        rw [← block_sub_eq_succ hnlt]
        exact hprev.2
      have hmem := winningSolver_mem_allValidFor g p
      have hplacement :=
        (Placement.mem_allValidFor GameConfig.standard p _).mp hmem
      constructor
      · rw [adversarialTrace_succ]
        apply ReachableAt.step hprev.1 hremaining.not_lost hp
        rw [Placement.eta_of_piece_eq hplacement.1]
        exact hplacement.2
      · rw [adversarialTrace_succ]
        exact winningSolver_step_winning hnlt hprev.1 hprev.2 hp

/-- Every legal five-bag trace of the extracted strategy ends at root. -/
theorem winningSolver_resets
    (hroot : Winning blockLength GameState.init)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    adversarialTrace GameConfig.standard winningSolver s GameState.init
        blockLength =
      GameState.init := by
  have h :=
    (winningSolver_trace_reachable_and_winning hroot hl blockLength le_rfl).2
  exact winning_zero_iff.mp (by simpa using h)

/-- Every state through and including the end of the certified block is
non-lost. -/
theorem winningSolver_survives_block
    (hroot : Winning blockLength GameState.init)
    {s : ℕ → Piece} (hl : LegalSequence s)
    (n : ℕ) (hn : n ≤ blockLength) :
    ¬ (adversarialTrace GameConfig.standard winningSolver s GameState.init n).lost
      GameConfig.standard := by
  by_cases hnend : n = blockLength
  · subst n
    rw [winningSolver_resets hroot hl]
    exact GameState.init_not_lost GameConfig.standard
  · have hnlt : n < blockLength := by omega
    have hwin :=
      (winningSolver_trace_reachable_and_winning hroot hl n hn).2
    rw [block_sub_eq_succ hnlt] at hwin
    exact hwin.not_lost

/-! ## The induced finite closed cycle -/

/-- States occurring in some reachable winning phase strictly before the
five-bag reset boundary. -/
def WinningCycleState (g : GameState) : Prop :=
  ∃ d : ℕ,
    d < blockLength ∧
      ReachableAt GameConfig.standard d g ∧
        Winning (blockLength - d) g

/-- Every winning phase state is non-lost. -/
theorem WinningCycleState.not_lost {g : GameState}
    (h : WinningCycleState g) :
    ¬ g.lost GameConfig.standard := by
  obtain ⟨d, hd, _hreach, hwin⟩ := h
  rw [block_sub_eq_succ hd] at hwin
  exact hwin.not_lost

/-- Finite exact-depth reachable layers up to any requested depth. -/
noncomputable def reachableLayer : ℕ → Finset GameState
  | 0 => {GameState.init}
  | n + 1 =>
      (reachableLayer n).biUnion fun g =>
        if g.lost GameConfig.standard then
          ∅
        else
          g.bag.biUnion fun p =>
            (Placement.allValidFor GameConfig.standard p).image
              (adversarialStep GameConfig.standard g p)

/-- Exact-depth reachability is included in the finite generated layer. -/
theorem ReachableAt.mem_reachableLayer
    {n : ℕ} {g : GameState}
    (h : ReachableAt GameConfig.standard n g) :
    g ∈ reachableLayer n := by
  induction h with
  | init =>
      simp [reachableLayer]
  | @step n g p pl hprev hnl hp hvalid ih =>
      classical
      rw [reachableLayer, Finset.mem_biUnion]
      refine ⟨g, ih, ?_⟩
      rw [if_neg hnl, Finset.mem_biUnion]
      let pl' : Placement := { pl with piece := p }
      refine ⟨p, hp, ?_⟩
      rw [Finset.mem_image]
      refine ⟨pl', ?_, ?_⟩
      · exact (Placement.mem_allValidFor GameConfig.standard p pl').mpr
          ⟨rfl, hvalid⟩
      · rfl

/-- All generated exact-depth states before the reset boundary. -/
noncomputable def reachableBeforeBlock : Finset GameState := by
  classical
  exact (Finset.range blockLength).biUnion reachableLayer

/-- Finite materialization of all reachable winning phase states. -/
noncomputable def winningCycleStates : Finset GameState := by
  classical
  exact reachableBeforeBlock.filter WinningCycleState

@[simp] theorem mem_winningCycleStates_iff {g : GameState} :
    g ∈ winningCycleStates ↔ WinningCycleState g := by
  classical
  rw [winningCycleStates, Finset.mem_filter]
  constructor
  · exact fun h => h.2
  · intro h
    obtain ⟨d, hd, hreach, hwin⟩ := h
    refine ⟨?_, ⟨d, hd, hreach, hwin⟩⟩
    rw [reachableBeforeBlock, Finset.mem_biUnion]
    exact ⟨d, Finset.mem_range.mpr hd, hreach.mem_reachableLayer⟩

/-- A winning root belongs to the induced cycle at phase zero. -/
theorem init_mem_winningCycleStates
    (hroot : Winning blockLength GameState.init) :
    GameState.init ∈ winningCycleStates := by
  rw [mem_winningCycleStates_iff]
  exact ⟨0, by norm_num [blockLength, bagsPerBlock, piecesPerBag],
    ReachableAt.init, hroot⟩

/-- One extracted-solver step advances a winning phase, with the final phase
wrapping back to the root phase. -/
theorem WinningCycleState.step
    (hroot : Winning blockLength GameState.init)
    {g : GameState} (hg : WinningCycleState g)
    {p : Piece} (hp : p ∈ g.bag) :
    WinningCycleState
      (adversarialStep GameConfig.standard g p (winningSolver g p)) := by
  have hnl := WinningCycleState.not_lost hg
  obtain ⟨d, hd, hreach, hwin⟩ := hg
  have hnextwin := winningSolver_step_winning hd hreach hwin hp
  have hplacement :=
    (Placement.mem_allValidFor GameConfig.standard p _).mp
      (winningSolver_mem_allValidFor g p)
  have hnextreach :
      ReachableAt GameConfig.standard (d + 1)
        (adversarialStep GameConfig.standard g p (winningSolver g p)) := by
    apply ReachableAt.step hreach hnl hp
    rw [Placement.eta_of_piece_eq hplacement.1]
    exact hplacement.2
  by_cases hinside : d + 1 < blockLength
  · exact ⟨d + 1, hinside, hnextreach, hnextwin⟩
  · have hboundary : d + 1 = blockLength := by omega
    have hreset :
        adversarialStep GameConfig.standard g p (winningSolver g p) =
          GameState.init := by
      rw [hboundary] at hnextwin
      exact winning_zero_iff.mp (by simpa using hnextwin)
    rw [hreset]
    exact ⟨0, by norm_num [blockLength, bagsPerBlock, piecesPerBag],
      ReachableAt.init, hroot⟩

/-- Lines cleared by move `n` of an adversarial trace. The placement is
piece-normalized exactly as it is by `adversarialStep`. -/
def linesClearedAt (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece)
    (g₀ : GameState) (n : ℕ) : ℕ :=
  let g := adversarialTrace cfg σ s g₀ n
  let pl : Placement := { σ g (s n) with piece := s n }
  Board.linesCleared cfg (pl.place g.board)

/-- Cumulative line clears in the first `n` moves of an adversarial trace. -/
def totalLinesCleared (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece)
    (g₀ : GameState) : ℕ → ℕ
  | 0 => 0
  | n + 1 => totalLinesCleared cfg σ s g₀ n + linesClearedAt cfg σ s g₀ n

@[simp] theorem totalLinesCleared_zero (cfg : GameConfig) (σ : Solver cfg)
    (s : ℕ → Piece) (g₀ : GameState) :
    totalLinesCleared cfg σ s g₀ 0 = 0 := rfl

theorem totalLinesCleared_succ (cfg : GameConfig) (σ : Solver cfg)
    (s : ℕ → Piece) (g₀ : GameState) (n : ℕ) :
    totalLinesCleared cfg σ s g₀ (n + 1) =
      totalLinesCleared cfg σ s g₀ n + linesClearedAt cfg σ s g₀ n := rfl

/-- Exact cell conservation along a legal trace of a valid solver. This is
the trace-indexed form of `Placement.applyStep_count`. -/
theorem adversarialTrace_wf_and_count
    {cfg : GameConfig} {σ : Solver cfg} (hvalid : ValidSolver cfg σ)
    {s : ℕ → Piece} {g₀ : GameState}
    (hl : LegalSequenceFrom g₀.bag s) (hwf : Board.WF cfg g₀.board)
    (n : ℕ) :
    Board.WF cfg (adversarialTrace cfg σ s g₀ n).board ∧
      (adversarialTrace cfg σ s g₀ n).board.count +
          cfg.cols * totalLinesCleared cfg σ s g₀ n =
        g₀.board.count + 4 * n := by
  induction n with
  | zero =>
      exact ⟨hwf, by simp⟩
  | succ n ih =>
      let g := adversarialTrace cfg σ s g₀ n
      let pl : Placement := { σ g (s n) with piece := s n }
      have hp : s n ∈ g.bag := by
        dsimp [g]
        rw [adversarialTrace_bag_from]
        exact hl n
      have hsolver := hvalid g (s n) hp
      have heta : pl = σ g (s n) := by
        exact Placement.eta_of_piece_eq hsolver.1
      have hplValid : pl.Valid cfg := by
        rw [heta]
        exact hsolver.2
      have hstep :=
        applyStep_count cfg g.board pl ih.1 hplValid
      constructor
      · rw [adversarialTrace_succ, adversarialStep_board]
        change Board.WF cfg (Placement.applyStep cfg g.board pl)
        exact Placement.applyStep_wf ih.1 hplValid
      · rw [adversarialTrace_succ, adversarialStep_board,
            totalLinesCleared_succ]
        change (Placement.applyStep cfg g.board pl).count +
            cfg.cols *
              (totalLinesCleared cfg σ s g₀ n +
                Board.linesCleared cfg (pl.place g.board)) =
          g₀.board.count + 4 * (n + 1)
        calc
          (Placement.applyStep cfg g.board pl).count +
              cfg.cols *
                (totalLinesCleared cfg σ s g₀ n +
                  Board.linesCleared cfg (pl.place g.board))
              =
            ((Placement.applyStep cfg g.board pl).count +
                cfg.cols * Board.linesCleared cfg (pl.place g.board)) +
              cfg.cols * totalLinesCleared cfg σ s g₀ n := by
                rw [Nat.mul_add]
                omega
          _ = (g.board.count + 4) +
              cfg.cols * totalLinesCleared cfg σ s g₀ n := by
                rw [hstep]
          _ = (g.board.count +
                cfg.cols * totalLinesCleared cfg σ s g₀ n) + 4 := by
                omega
          _ = (g₀.board.count + 4 * n) + 4 := by
                rw [ih.2]
          _ = g₀.board.count + 4 * (n + 1) := by
                omega

/-- Cell-count specialization for traces starting at the root state. -/
theorem adversarialTrace_count_from_init
    {cfg : GameConfig} {σ : Solver cfg} (hvalid : ValidSolver cfg σ)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace cfg σ s GameState.init n).board.count +
        cfg.cols * totalLinesCleared cfg σ s GameState.init n =
      4 * n := by
  have hl' : LegalSequenceFrom GameState.init.bag s := by
    change LegalSequence s
    exact hl
  have h := (adversarialTrace_wf_and_count hvalid hl'
    (GameState.init_board_wf cfg) n).2
  simpa using h

/-- Fourteen clears in a valid five-bag standard trace force the terminal
board to be empty. -/
theorem board_count_zero_of_fourteen_lines
    {σ : Solver GameConfig.standard} (hvalid : ValidSolver GameConfig.standard σ)
    {s : ℕ → Piece} (hl : LegalSequence s)
    (hclears :
      totalLinesCleared GameConfig.standard σ s GameState.init blockLength =
        targetLines) :
    (adversarialTrace GameConfig.standard σ s GameState.init blockLength).board.count =
      0 := by
  have hcount := adversarialTrace_count_from_init hvalid hl blockLength
  rw [hclears] at hcount
  norm_num [blockLength, bagsPerBlock, piecesPerBag, targetLines] at hcount ⊢
  omega

/-- The exact five-bag reset lemma: valid play plus fourteen clears returns
both the board and bag to the root state. -/
theorem trace_eq_init_of_fourteen_lines
    {σ : Solver GameConfig.standard} (hvalid : ValidSolver GameConfig.standard σ)
    {s : ℕ → Piece} (hl : LegalSequence s)
    (hclears :
      totalLinesCleared GameConfig.standard σ s GameState.init blockLength =
        targetLines) :
    adversarialTrace GameConfig.standard σ s GameState.init blockLength =
      GameState.init := by
  apply (GameState.eq_init_iff_count_zero_and_bag_full _).mpr
  refine ⟨board_count_zero_of_fourteen_lines hvalid hl hclears, ?_⟩
  rw [adversarialTrace_bag]
  exact bagAt_blockLength_eq_full hl

/-- Conversely, a valid five-bag trace that returns to root must have cleared
exactly fourteen lines. -/
theorem totalLinesCleared_eq_target_of_trace_eq_init
    {σ : Solver GameConfig.standard} (hvalid : ValidSolver GameConfig.standard σ)
    {s : ℕ → Piece} (hl : LegalSequence s)
    (hreset :
      adversarialTrace GameConfig.standard σ s GameState.init blockLength =
        GameState.init) :
    totalLinesCleared GameConfig.standard σ s GameState.init blockLength =
      targetLines := by
  have hcount := adversarialTrace_count_from_init hvalid hl blockLength
  rw [hreset] at hcount
  rw [GameState.init_board] at hcount
  change (∅ : Board).card +
      GameConfig.standard.cols *
        totalLinesCleared GameConfig.standard σ s GameState.init blockLength =
      4 * blockLength at hcount
  simp at hcount
  norm_num [blockLength, bagsPerBlock, piecesPerBag, targetLines,
    GameConfig.standard] at hcount ⊢
  omega

/-- For valid five-bag play, the arithmetic certificate and root-reset
certificate are equivalent. -/
theorem trace_eq_init_iff_totalLinesCleared_eq_target
    {σ : Solver GameConfig.standard} (hvalid : ValidSolver GameConfig.standard σ)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    adversarialTrace GameConfig.standard σ s GameState.init blockLength =
        GameState.init ↔
      totalLinesCleared GameConfig.standard σ s GameState.init blockLength =
        targetLines := by
  constructor
  · exact totalLinesCleared_eq_target_of_trace_eq_init hvalid hl
  · exact trace_eq_init_of_fourteen_lines hvalid hl

/-- A bounded, adversarial five-bag reset certificate.

`survives_block` checks every intermediate state in the 35-move block.
`clears_target` checks the arithmetic reset condition at the endpoint.
Both quantify over every legal sequence, while `σ` sees only the current
state and announced piece, so the certificate is online rather than
clairvoyant. -/
structure Certificate (σ : Solver GameConfig.standard) : Prop where
  valid : ValidSolver GameConfig.standard σ
  survives_block :
    ∀ s : ℕ → Piece, LegalSequence s →
      ∀ n ≤ blockLength,
        ¬ (adversarialTrace GameConfig.standard σ s GameState.init n).lost
          GameConfig.standard
  clears_target :
    ∀ s : ℕ → Piece, LegalSequence s →
      totalLinesCleared GameConfig.standard σ s GameState.init blockLength =
        targetLines

/-- A certified solver returns to root after every legal five-bag block. -/
theorem Certificate.resets (h : Certificate σ)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    adversarialTrace GameConfig.standard σ s GameState.init blockLength =
      GameState.init :=
  trace_eq_init_of_fourteen_lines h.valid hl (h.clears_target s hl)

/-- A root winning proof for the 35-move backward game yields the concrete
five-bag reset certificate carried by `winningSolver`. -/
theorem winningSolver_certificate_of_winning
    (hroot : Winning blockLength GameState.init) :
    Certificate winningSolver where
  valid := winningSolver_valid
  survives_block := by
    intro s hl n hn
    exact winningSolver_survives_block hroot hl n hn
  clears_target := by
    intro s hl
    exact totalLinesCleared_eq_target_of_trace_eq_init winningSolver_valid hl
      (winningSolver_resets hroot hl)

/-- The finite backward game predicate is sufficient for the existing
certificate theorem. -/
theorem exists_fiveBagResetCertificate_of_winning
    (hroot : Winning blockLength GameState.init) :
    ∃ σ : Solver GameConfig.standard, Certificate σ :=
  ⟨winningSolver, winningSolver_certificate_of_winning hroot⟩

/-- If the root is in the depth-35 finite winning layer, the five-bag reset
certificate exists. This is the concrete finite-search target. -/
theorem exists_fiveBagResetCertificate_of_init_mem_winningFinset
    {U : Finset GameState}
    (h : GameState.init ∈ winningFinset U blockLength) :
    ∃ σ : Solver GameConfig.standard, Certificate σ :=
  exists_fiveBagResetCertificate_of_winning
    (winning_of_init_mem_winningFinset h)

/-- Reset recurrence at every five-bag boundary. -/
theorem Certificate.resets_mul_blockLength (h : Certificate σ)
    {s : ℕ → Piece} (hl : LegalSequence s) (blocks : ℕ) :
    adversarialTrace GameConfig.standard σ s GameState.init
        (blockLength * blocks) =
      GameState.init := by
  induction blocks with
  | zero => rfl
  | succ blocks ih =>
      have hbag :
          bagAt Bag.full s (blockLength * blocks) = Bag.full := by
        rw [← adversarialTrace_bag GameConfig.standard σ s, ih]
        rfl
      have hsuffix :
          LegalSequence (fun k => s (blockLength * blocks + k)) := by
        have hs := hl.suffix (blockLength * blocks)
        rw [hbag] at hs
        exact hs
      rw [Nat.mul_succ, adversarialTrace_add, ih]
      exact h.resets hsuffix

/-- A five-bag reset certificate can be iterated forever. -/
theorem Certificate.solvesTetris (h : Certificate σ) :
    SolvesTetris GameConfig.standard σ := by
  intro s hl n
  let blocks := n / blockLength
  let remainder := n % blockLength
  have hlength : 0 < blockLength := by norm_num [blockLength, bagsPerBlock, piecesPerBag]
  have hsplit : n = blockLength * blocks + remainder := by
    dsimp [blocks, remainder]
    omega
  have hboundary := h.resets_mul_blockLength hl blocks
  have hbag :
      bagAt Bag.full s (blockLength * blocks) = Bag.full := by
    rw [← adversarialTrace_bag GameConfig.standard σ s, hboundary]
    rfl
  let suffix : ℕ → Piece := fun k => s (blockLength * blocks + k)
  have hsuffix : LegalSequence suffix := by
    have hs := hl.suffix (blockLength * blocks)
    rw [hbag] at hs
    exact hs
  have hremainder : remainder ≤ blockLength := by
    have := Nat.mod_lt n hlength
    omega
  rw [hsplit, adversarialTrace_add, hboundary]
  exact h.survives_block suffix hsuffix remainder hremainder

/-- The certificate also satisfies the validity-strengthened solver contract
used by the safe-set machinery. -/
theorem Certificate.solvesTetrisValid (h : Certificate σ) :
    SolvesTetrisValid GameConfig.standard σ :=
  ⟨h.valid, h.solvesTetris⟩

/-- Safe-set formulation of the experimental result for a fixed certificate. -/
theorem Certificate.init_mem_safe (h : Certificate σ) :
    GameState.init ∈ safe GameConfig.standard :=
  init_safe_of_solvesTetrisValid h.solvesTetrisValid

/-- Existence of a certificate proves the validity-strengthened form of
canonical Tetris solvability. -/
theorem tetrisSolvableValid_of_exists_fiveBagResetCertificate
    (hexists :
      ∃ σ : Solver GameConfig.standard, Certificate σ) :
    TetrisSolvableValid := by
  obtain ⟨σ, hσ⟩ := hexists
  exact ⟨σ, hσ.solvesTetrisValid⟩

/-- The experimental theorem approach: a bounded five-bag reset certificate
is sufficient to solve canonical Tetris for infinite horizons. -/
theorem tetrisSolvable_of_exists_fiveBagResetCertificate
    (hexists :
      ∃ σ : Solver GameConfig.standard, Certificate σ) :
    TetrisSolvable := by
  obtain ⟨σ, hσ⟩ := hexists
  exact ⟨σ, hσ.solvesTetris⟩

/-- Direct bridge from the 35-step backward game to validity-strengthened
canonical Tetris solvability. -/
theorem tetrisSolvableValid_of_winning
    (hroot : Winning blockLength GameState.init) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_exists_fiveBagResetCertificate
    (exists_fiveBagResetCertificate_of_winning hroot)

/-- Direct bridge from the 35-step backward game to canonical Tetris
solvability. -/
theorem tetrisSolvable_of_winning
    (hroot : Winning blockLength GameState.init) :
    TetrisSolvable :=
  tetrisSolvable_of_exists_fiveBagResetCertificate
    (exists_fiveBagResetCertificate_of_winning hroot)

/-- Direct finite-search target: if the root is found in the computed
depth-35 winning layer, canonical Tetris is solvable. -/
theorem tetrisSolvable_of_init_mem_winningFinset
    {U : Finset GameState}
    (h : GameState.init ∈ winningFinset U blockLength) :
    TetrisSolvable :=
  tetrisSolvable_of_winning (winning_of_init_mem_winningFinset h)

/-- Validity-strengthened form of the finite-search bridge. -/
theorem tetrisSolvableValid_of_init_mem_winningFinset
    {U : Finset GameState}
    (h : GameState.init ∈ winningFinset U blockLength) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_winning (winning_of_init_mem_winningFinset h)

/-! ## Item 1 — Bidirectional reachable-layer characterisation -/

/-- The reverse direction of `ReachableAt.mem_reachableLayer`: every member
of the finite layer is exactly-depth reachable. By induction on `n`,
unfolding the `biUnion / if / image` structure of `reachableLayer (n+1)`
and applying `ReachableAt.step` with the placement-validity witness
extracted from `Placement.mem_allValidFor`. -/
theorem reachableAt_of_mem_reachableLayer :
    ∀ {n : ℕ} {g : GameState},
      g ∈ reachableLayer n → ReachableAt GameConfig.standard n g := by
  intro n
  induction n with
  | zero =>
      intro g hg
      classical
      rw [reachableLayer, Finset.mem_singleton] at hg
      subst hg
      exact ReachableAt.init
  | succ n ih =>
      intro g hg
      classical
      rw [reachableLayer, Finset.mem_biUnion] at hg
      obtain ⟨g', hg', hbody⟩ := hg
      by_cases hlost : g'.lost GameConfig.standard
      · rw [if_pos hlost] at hbody
        exact (Finset.notMem_empty g hbody).elim
      · rw [if_neg hlost, Finset.mem_biUnion] at hbody
        obtain ⟨p, hp, himg⟩ := hbody
        rw [Finset.mem_image] at himg
        obtain ⟨pl, hpl, hstep⟩ := himg
        have hplValid :=
          (Placement.mem_allValidFor GameConfig.standard p pl).mp hpl
        subst hstep
        have hreach' := ih hg'
        apply ReachableAt.step hreach' hlost hp
        rw [Placement.eta_of_piece_eq hplValid.1]
        exact hplValid.2

/-- **Item 1**: `g ∈ reachableLayer n ↔ ReachableAt standard n g`. The finite
generator contains exactly the exact-depth reachable states — no missing
states, no spurious states. -/
theorem mem_reachableLayer_iff {n : ℕ} {g : GameState} :
    g ∈ reachableLayer n ↔ ReachableAt GameConfig.standard n g :=
  ⟨reachableAt_of_mem_reachableLayer, ReachableAt.mem_reachableLayer⟩

/-! ## Item 2 — Complete finite universe -/

/-- **Item 2**: the pre-block finite universe is exactly the union of all
exact-depth reachable states with depth `< blockLength`. Direct via
`Finset.mem_biUnion` + Item 1. -/
theorem mem_reachableBeforeBlock_iff {g : GameState} :
    g ∈ reachableBeforeBlock ↔
      ∃ n < blockLength, ReachableAt GameConfig.standard n g := by
  classical
  rw [reachableBeforeBlock, Finset.mem_biUnion]
  constructor
  · rintro ⟨n, hn, hg⟩
    exact ⟨n, Finset.mem_range.mp hn, mem_reachableLayer_iff.mp hg⟩
  · rintro ⟨n, hn, hg⟩
    exact ⟨n, Finset.mem_range.mpr hn, mem_reachableLayer_iff.mpr hg⟩

/-! ## Item 3 — Completeness of finite backward search -/

/-- Auxiliary: from a reachable state at depth `d`, with `d + n = blockLength`
and `Winning n g`, we get `WinningIn reachableBeforeBlock n g`. The depth
`d` is generalized so it can shrink (grow by one) in the inductive successor
case. -/
theorem winningIn_reachableBeforeBlock_of_winning :
    ∀ {n : ℕ} {d : ℕ} {g : GameState},
      d + n = blockLength →
      ReachableAt GameConfig.standard d g →
      Winning n g →
      WinningIn reachableBeforeBlock n g := by
  intro n
  induction n with
  | zero =>
      intro _ _ _ _ hwin
      exact hwin
  | succ m ih =>
      intro d g hsum hreach hwin
      have hdlt : d < blockLength := by omega
      refine ⟨?_, hwin.1, ?_⟩
      · classical
        rw [reachableBeforeBlock, Finset.mem_biUnion]
        exact ⟨d, Finset.mem_range.mpr hdlt, hreach.mem_reachableLayer⟩
      · intro p hp
        obtain ⟨pl, hpl, hnext⟩ := hwin.2 p hp
        refine ⟨pl, hpl, ?_⟩
        have hplValid :=
          (Placement.mem_allValidFor GameConfig.standard p pl).mp hpl
        have hreach' :
            ReachableAt GameConfig.standard (d + 1)
              (adversarialStep GameConfig.standard g p pl) := by
          apply ReachableAt.step hreach hwin.1 hp
          rw [Placement.eta_of_piece_eq hplValid.1]
          exact hplValid.2
        exact ih (by omega) hreach' hnext

/-- **Item 3 (root specialization)**: a positive root-winning proof is
exactly a finite-universe winning proof for the complete pre-block universe.
Together with the existing `WinningIn.winning` (soundness), this gives the
two-sided characterisation `Winning blockLength init ↔ WinningIn U blockLength
init` for `U = reachableBeforeBlock`. -/
theorem winningIn_reachableBeforeBlock_blockLength_iff :
    WinningIn reachableBeforeBlock blockLength GameState.init ↔
      Winning blockLength GameState.init := by
  refine ⟨WinningIn.winning, fun hwin => ?_⟩
  apply winningIn_reachableBeforeBlock_of_winning (d := 0) _ ReachableAt.init hwin
  simp [blockLength, bagsPerBlock, piecesPerBag]

/-- **Item 3 finite form**: the root is in the depth-`blockLength` finite
backward layer iff it is mathematically winning. Combined with
`tetrisSolvable_of_init_mem_winningFinset`, this is the cleanest
mechanical-search obligation. -/
theorem init_mem_winningFinset_reachableBeforeBlock_iff :
    GameState.init ∈ winningFinset reachableBeforeBlock blockLength ↔
      Winning blockLength GameState.init := by
  rw [mem_winningFinset_iff,
      winningIn_reachableBeforeBlock_blockLength_iff]

/-! ## Item 4 — Executable boolean checker -/

/-- Executable boolean version of `Winning`: returns `true` iff an online
player can survive the next `n` adversarial legal draws and finish exactly
at `GameState.init`. The body uses `decide` on the propositional form,
which is well-typed because:

* `g = GameState.init` is decidable via `DecidableEq GameState` (`Game.lean`).
* `g.lost cfg` is decidable (instance in `Game.lean`).
* `∀ p ∈ g.bag, _` is decidable via `Finset.decidableBAll`.
* `∃ pl ∈ allValidFor cfg p, _` is decidable via `Finset.decidableBEx`.
* `winningBool n x = true` is `Bool` equality, always decidable.

The recursive structure terminates because `n` decreases. -/
def winningBool : ℕ → GameState → Bool
  | 0, g => decide (g = GameState.init)
  | n + 1, g =>
      decide
        (¬ g.lost GameConfig.standard ∧
          ∀ p ∈ g.bag,
            ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
              winningBool n
                  (adversarialStep GameConfig.standard g p pl) = true)

@[simp] theorem winningBool_zero (g : GameState) :
    winningBool 0 g = decide (g = GameState.init) := rfl

@[simp] theorem winningBool_succ (n : ℕ) (g : GameState) :
    winningBool (n + 1) g =
      decide
        (¬ g.lost GameConfig.standard ∧
          ∀ p ∈ g.bag,
            ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
              winningBool n
                  (adversarialStep GameConfig.standard g p pl) = true) := rfl

/-- **Item 4**: the boolean checker reflects the propositional `Winning`.
Induction on `n`; both sides unfold to identical propositional statements
modulo the inner `winningBool n ... = true ↔ Winning n ...` rewrite given
by the inductive hypothesis. -/
theorem winningBool_eq_true_iff :
    ∀ (n : ℕ) (g : GameState),
      winningBool n g = true ↔ Winning n g := by
  intro n
  induction n with
  | zero =>
      intro g
      simp [winningBool, Winning]
  | succ n ih =>
      intro g
      rw [winningBool_succ, decide_eq_true_eq, Winning]
      refine and_congr Iff.rfl ?_
      refine forall_congr' fun p => ?_
      refine forall_congr' fun _ => ?_
      refine exists_congr fun pl => ?_
      refine and_congr Iff.rfl ?_
      exact ih _

/-- **Decidability**: `Winning n g` is decidable, via reflection through
`winningBool`. Enables `decide` / `native_decide` proofs at the root. -/
instance Winning.decidable (n : ℕ) (g : GameState) : Decidable (Winning n g) :=
  decidable_of_iff _ (winningBool_eq_true_iff n g)

/-- **The headline mechanical-search bridge**: if `winningBool blockLength
init` evaluates to `true` (provable by `decide` or `native_decide`), then
canonical Tetris is solvable. -/
theorem tetrisSolvable_of_winningBool_blockLength_init
    (h : winningBool blockLength GameState.init = true) :
    TetrisSolvable :=
  tetrisSolvable_of_winning ((winningBool_eq_true_iff _ _).mp h)

/-! ## Decomposition framework — split the 35-piece game into 5 single-bag games

This section provides the *meta-theorem* that breaks the monolithic
`Winning 35 init` obligation into 5 separate single-bag obligations
`P_i → P_{i+1}`, where `P_0 = P_5 = {init}` and `P_1, P_2, P_3, P_4` are
freely chosen intermediate phase boards. This shape lets us attack the
35-piece reset by independently verifying the 5 single-bag transitions
(7-piece adversarial games), each of which is tractable by case analysis
on the 7! = 5040 bag orderings (and much less after symmetry quotient). -/

/-- Generalised winning predicate: `WinningTo T n g` says an online player
can survive `n` adversarial legal draws and finish in some state in `T`.
Specialises to `Winning` at `T = {GameState.init}`. -/
def WinningTo (T : Finset GameState) : ℕ → GameState → Prop
  | 0, g => g ∈ T
  | n + 1, g =>
      ¬ g.lost GameConfig.standard ∧
        ∀ p ∈ g.bag,
          ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
            WinningTo T n (adversarialStep GameConfig.standard g p pl)

@[simp] theorem winningTo_zero_iff (T : Finset GameState) (g : GameState) :
    WinningTo T 0 g ↔ g ∈ T := Iff.rfl

@[simp] theorem winningTo_succ_iff (T : Finset GameState) (n : ℕ) (g : GameState) :
    WinningTo T (n + 1) g ↔
      ¬ g.lost GameConfig.standard ∧
        ∀ p ∈ g.bag,
          ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
            WinningTo T n (adversarialStep GameConfig.standard g p pl) := Iff.rfl

/-- `Winning n g` is exactly `WinningTo {init} n g`. Bridges the existing
predicate to the generalised one for use in decomposition. -/
theorem winning_iff_winningTo_init (n : ℕ) (g : GameState) :
    Winning n g ↔ WinningTo {GameState.init} n g := by
  induction n generalizing g with
  | zero =>
      simp [Winning, WinningTo, Finset.mem_singleton]
  | succ n ih =>
      simp only [Winning, WinningTo]
      refine and_congr Iff.rfl ?_
      refine forall_congr' fun _ => ?_
      refine forall_congr' fun _ => ?_
      refine exists_congr fun _ => ?_
      refine and_congr Iff.rfl ?_
      exact ih _

/-- **The composition lemma**: if `g` is winning into intermediate target
`T₁` in `m` moves, and every state in `T₁` is winning into final target
`T₂` in `n` moves, then `g` is winning into `T₂` in `m + n` moves.

This is the structural backbone of the decomposition — it lets us *chain*
single-bag obligations to reconstruct the full 35-piece obligation. -/
theorem WinningTo.compose {T₁ T₂ : Finset GameState} :
    ∀ {m n : ℕ} {g : GameState},
      WinningTo T₁ m g →
      (∀ g' ∈ T₁, WinningTo T₂ n g') →
      WinningTo T₂ (m + n) g := by
  intro m
  induction m with
  | zero =>
      intro n g h₁ h₂
      simp only [Nat.zero_add]
      exact h₂ g h₁
  | succ m ih =>
      intro n g h₁ h₂
      rw [show m + 1 + n = (m + n) + 1 from by omega]
      refine ⟨h₁.1, ?_⟩
      intro p hp
      obtain ⟨pl, hpl, hnext⟩ := h₁.2 p hp
      refine ⟨pl, hpl, ?_⟩
      exact ih hnext h₂

/-- A *bag-winning* relation: every state in `P_in` can be steered into
`P_out` by an online player against an adversarial 7-piece bag. This is
the per-bag obligation produced by the decomposition. -/
def BagWinning (P_in P_out : Finset GameState) : Prop :=
  ∀ g ∈ P_in, WinningTo P_out piecesPerBag g

/-- **A bag-closed set survives arbitrarily many bags.** If `P` is
self-recurrent (`BagWinning P P`), then from any `g ∈ P` the online player can be
steered to remain in `P` for `k` whole bags, i.e. `WinningTo P (k * piecesPerBag)
g` for every `k`. Proof: induction on `k`, base `WinningTo P 0 g ↔ g ∈ P`, step
by `WinningTo.compose` with the self-recurrence premise. This is the explicit
unbounded-horizon survival statement underlying every self-recurrent certificate:
a single bag-closure obligation yields safety against an adversary of any depth,
which is the heart of the Atlas-existence claim. -/
theorem BagWinning.self_winningTo_iterate {P : Finset GameState}
    (hself : BagWinning P P) :
    ∀ (k : ℕ) (g : GameState), g ∈ P → WinningTo P (k * piecesPerBag) g := by
  intro k
  induction k with
  | zero =>
      intro g hg
      simpa using hg
  | succ k ih =>
      intro g hg
      have hstep : WinningTo P (k * piecesPerBag + piecesPerBag) g :=
        (ih g hg).compose hself
      have he : k * piecesPerBag + piecesPerBag = (k + 1) * piecesPerBag := by ring
      rwa [he] at hstep

/-- **The phase decomposition meta-theorem**: 5 separate single-bag wins
compose into the full 35-piece reset.

To prove `Winning 35 GameState.init`, choose any 4 intermediate phase
boards `P_1, P_2, P_3, P_4 : Finset GameState` and discharge 5 single-bag
obligations:

* `{init} → P_1`  (first bag)
* `P_1 → P_2`     (second bag)
* `P_2 → P_3`     (third bag)
* `P_3 → P_4`     (fourth bag)
* `P_4 → {init}`  (fifth bag — must return to the start)

Each `BagWinning` premise is a 7-piece online game, which is tractable
by case analysis on the (≤ 7! = 5040) bag orderings. -/
theorem Winning_35_of_phase_decomposition
    (P₁ P₂ P₃ P₄ : Finset GameState)
    (hbag1 : BagWinning {GameState.init} P₁)
    (hbag2 : BagWinning P₁ P₂)
    (hbag3 : BagWinning P₂ P₃)
    (hbag4 : BagWinning P₃ P₄)
    (hbag5 : BagWinning P₄ {GameState.init}) :
    Winning blockLength GameState.init := by
  rw [winning_iff_winningTo_init]
  have hinit_mem : GameState.init ∈ ({GameState.init} : Finset GameState) :=
    Finset.mem_singleton.mpr rfl
  -- Step 1: init is WinningTo P₁ in 7 moves.
  have h1 : WinningTo P₁ piecesPerBag GameState.init :=
    hbag1 GameState.init hinit_mem
  -- Step 2: compose with bag 2 (P₁ → P₂), obtaining WinningTo P₂ in 14 moves.
  have h2 : WinningTo P₂ (piecesPerBag + piecesPerBag) GameState.init :=
    h1.compose hbag2
  -- Step 3: compose with bag 3 (P₂ → P₃), 21 moves.
  have h3 :
      WinningTo P₃ ((piecesPerBag + piecesPerBag) + piecesPerBag)
        GameState.init :=
    h2.compose hbag3
  -- Step 4: compose with bag 4 (P₃ → P₄), 28 moves.
  have h4 :
      WinningTo P₄
        (((piecesPerBag + piecesPerBag) + piecesPerBag) + piecesPerBag)
        GameState.init :=
    h3.compose hbag4
  -- Step 5: compose with bag 5 (P₄ → {init}), 35 moves.
  have h5 :
      WinningTo {GameState.init}
        ((((piecesPerBag + piecesPerBag) + piecesPerBag) + piecesPerBag)
            + piecesPerBag)
        GameState.init :=
    h4.compose hbag5
  -- Arithmetic: 7+7+7+7+7 = 35 = blockLength.
  have hsum :
      (((piecesPerBag + piecesPerBag) + piecesPerBag) + piecesPerBag)
          + piecesPerBag = blockLength := by
    norm_num [blockLength, bagsPerBlock, piecesPerBag]
  rw [hsum] at h5
  exact h5

/-- A self-loop specialisation of the decomposition: if a single
intermediate phase `Q` works as `{init} → Q → ⋯ → Q → {init}` with the
same step repeated 5 times, the recursion collapses to two obligations.

(Useful when looking for a *unique* canonical mid-bag residue board that
the strategy returns to between consecutive bags.) -/
theorem Winning_35_of_uniform_phase
    (Q : Finset GameState)
    (hopen : BagWinning {GameState.init} Q)
    (hloop : BagWinning Q Q)
    (hclose : BagWinning Q {GameState.init}) :
    Winning blockLength GameState.init :=
  Winning_35_of_phase_decomposition Q Q Q Q hopen hloop hloop hloop hclose

/-! ## Decidable BagWinning -/

/-- Executable boolean version of `WinningTo`. Mirrors `winningBool` but
parameterised over the target set `T`. Uses `decide` on the inner
proposition (same Decidable infrastructure: `DecidableEq GameState`,
`Decidable (g.lost cfg)`, `Finset.decidableBAll/BEx`). -/
def winningToBool (T : Finset GameState) : ℕ → GameState → Bool
  | 0, g => decide (g ∈ T)
  | n + 1, g =>
      decide
        (¬ g.lost GameConfig.standard ∧
          ∀ p ∈ g.bag,
            ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
              winningToBool T n
                  (adversarialStep GameConfig.standard g p pl) = true)

@[simp] theorem winningToBool_zero (T : Finset GameState) (g : GameState) :
    winningToBool T 0 g = decide (g ∈ T) := rfl

@[simp] theorem winningToBool_succ (T : Finset GameState) (n : ℕ) (g : GameState) :
    winningToBool T (n + 1) g =
      decide
        (¬ g.lost GameConfig.standard ∧
          ∀ p ∈ g.bag,
            ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
              winningToBool T n
                  (adversarialStep GameConfig.standard g p pl) = true) := rfl

/-- `winningToBool` reflects `WinningTo`: an executable check soundly
identifies the propositional generalised winning predicate. -/
theorem winningToBool_eq_true_iff
    (T : Finset GameState) :
    ∀ (n : ℕ) (g : GameState),
      winningToBool T n g = true ↔ WinningTo T n g := by
  intro n
  induction n with
  | zero =>
      intro g
      simp [winningToBool, WinningTo]
  | succ n ih =>
      intro g
      rw [winningToBool_succ, decide_eq_true_eq, WinningTo]
      refine and_congr Iff.rfl ?_
      refine forall_congr' fun p => ?_
      refine forall_congr' fun _ => ?_
      refine exists_congr fun pl => ?_
      refine and_congr Iff.rfl ?_
      exact ih _

/-- Decidability for the generalised winning predicate, via reflection
through `winningToBool`. -/
instance WinningTo.decidable (T : Finset GameState) (n : ℕ) (g : GameState) :
    Decidable (WinningTo T n g) :=
  decidable_of_iff _ (winningToBool_eq_true_iff T n g)

/-- Boolean version of `BagWinning`: every state in the source set passes
the executable `winningToBool` check at depth 7. -/
def bagWinningBool (P_in P_out : Finset GameState) : Bool :=
  decide (∀ g ∈ P_in, winningToBool P_out piecesPerBag g = true)

/-- Reflection: `bagWinningBool` is sound and complete for `BagWinning`. -/
theorem bagWinningBool_eq_true_iff (P_in P_out : Finset GameState) :
    bagWinningBool P_in P_out = true ↔ BagWinning P_in P_out := by
  simp only [bagWinningBool, decide_eq_true_eq, BagWinning]
  refine ⟨fun h g hg => (winningToBool_eq_true_iff _ _ _).mp (h g hg),
          fun h g hg => (winningToBool_eq_true_iff _ _ _).mpr (h g hg)⟩

/-- Decidability for `BagWinning`. -/
instance BagWinning.decidable (P_in P_out : Finset GameState) :
    Decidable (BagWinning P_in P_out) :=
  decidable_of_iff _ (bagWinningBool_eq_true_iff P_in P_out)

/-! ## Decidable version of the decomposition meta-theorem -/

/-- **Decidable version of the decomposition theorem**: 5 boolean checks
on bag-winning suffice to prove `Winning blockLength init`. Each check is
a finite Finset enumeration; combined with a manageable choice of phase
boards `P₁..P₄`, the whole obligation reduces to one `decide` call. -/
theorem Winning_35_of_bagWinningBool
    (P₁ P₂ P₃ P₄ : Finset GameState)
    (hbag1 : bagWinningBool {GameState.init} P₁ = true)
    (hbag2 : bagWinningBool P₁ P₂ = true)
    (hbag3 : bagWinningBool P₂ P₃ = true)
    (hbag4 : bagWinningBool P₃ P₄ = true)
    (hbag5 : bagWinningBool P₄ {GameState.init} = true) :
    Winning blockLength GameState.init :=
  Winning_35_of_phase_decomposition P₁ P₂ P₃ P₄
    ((bagWinningBool_eq_true_iff _ _).mp hbag1)
    ((bagWinningBool_eq_true_iff _ _).mp hbag2)
    ((bagWinningBool_eq_true_iff _ _).mp hbag3)
    ((bagWinningBool_eq_true_iff _ _).mp hbag4)
    ((bagWinningBool_eq_true_iff _ _).mp hbag5)

/-- Headline: combine the decomposition with the existing
`tetrisSolvable_of_winning` bridge to land on `TetrisSolvable`
directly from 5 boolean bag-winning checks. -/
theorem tetrisSolvable_of_bagWinningBool
    (P₁ P₂ P₃ P₄ : Finset GameState)
    (hbag1 : bagWinningBool {GameState.init} P₁ = true)
    (hbag2 : bagWinningBool P₁ P₂ = true)
    (hbag3 : bagWinningBool P₂ P₃ = true)
    (hbag4 : bagWinningBool P₃ P₄ = true)
    (hbag5 : bagWinningBool P₄ {GameState.init} = true) :
    TetrisSolvable :=
  tetrisSolvable_of_winning
    (Winning_35_of_bagWinningBool P₁ P₂ P₃ P₄ hbag1 hbag2 hbag3 hbag4 hbag5)

/-- **Validity-strengthened headline.** The `TetrisSolvableValid` companion of
`tetrisSolvable_of_bagWinningBool`: the same five decidable bag-winning checks
land directly on the project's ultimate goal `TetrisSolvableValid` (a *valid*
solver that never tops out), not merely `TetrisSolvable`. Routes the decidable
5-bag decomposition through `tetrisSolvableValid_of_winning` instead of
`tetrisSolvable_of_winning`; the constructed winning solver is reset-certified,
hence valid. This is the headline a `native_decide` search for phase boards
`P₁..P₄` should target to discharge the affirmative atlas claim. -/
theorem tetrisSolvableValid_of_bagWinningBool
    (P₁ P₂ P₃ P₄ : Finset GameState)
    (hbag1 : bagWinningBool {GameState.init} P₁ = true)
    (hbag2 : bagWinningBool P₁ P₂ = true)
    (hbag3 : bagWinningBool P₂ P₃ = true)
    (hbag4 : bagWinningBool P₃ P₄ = true)
    (hbag5 : bagWinningBool P₄ {GameState.init} = true) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_winning
    (Winning_35_of_bagWinningBool P₁ P₂ P₃ P₄ hbag1 hbag2 hbag3 hbag4 hbag5)

/-! ## Phase boundary invariants — what any candidate `P_i` must satisfy

A *bag boundary* is a depth `7 * i` where the entire current bag has been
drawn and refilled. The 5-bag reset block has 6 such boundaries (at depths
`0, 7, 14, 21, 28, 35`). At every boundary the bag is exactly `Bag.full`;
the board cell count is forced into a known residue class modulo 10.

These lemmas characterise the phase boards `P_1..P_4` that the decomposition
theorem requires: they must consist of states with full bag and the
appropriate count residue. -/

/-- The bag is `Bag.full` at every bag boundary `7 * i`. Direct via
`ReachableAt.bag_card_eq` + `Bag.card_eq_seven_iff_eq_full`. -/
theorem bag_eq_full_at_bag_boundary
    {cfg : GameConfig} {i : ℕ} {g : GameState}
    (h : ReachableAt cfg (piecesPerBag * i) g) :
    g.bag = Bag.full := by
  have hcard := h.bag_card_eq
  apply (Bag.card_eq_seven_iff_eq_full _).mp
  have hmod : (piecesPerBag * i) % 7 = 0 := by
    change (7 * i) % 7 = 0
    exact Nat.mul_mod_right 7 i
  rw [hmod] at hcard
  omega

/-- Standard-board count residue at any bag boundary: `count mod 10 = (28*i) mod 10`. -/
theorem standard_count_mod_ten_at_bag_boundary
    {i : ℕ} {g : GameState}
    (h : ReachableAt GameConfig.standard (piecesPerBag * i) g) :
    g.board.count % 10 = (28 * i) % 10 := by
  have hmod := h.standard_count_mod_ten
  have heq : 4 * (piecesPerBag * i) = 28 * i := by
    change 4 * (7 * i) = 28 * i; ring
  rw [heq] at hmod
  exact hmod

/-! ### Concrete count residues at each of the 6 phase boundaries

In the 5-bag block, depths split into 6 boundaries:

| Boundary | Depth | `count mod 10` |
|----------|-------|----------------|
| start    |  0    | 0              |
| bag 1 end|  7    | 8              |
| bag 2 end| 14    | 6              |
| bag 3 end| 21    | 4              |
| bag 4 end| 28    | 2              |
| bag 5 end| 35    | 0              |

Each candidate phase board `P_i` must consist of states with full bag and
the corresponding count residue (and of course non-lost / WF). -/

/-- Count = 0 at depth 0 (the root). -/
theorem count_at_phase_0 {g : GameState}
    (h : ReachableAt GameConfig.standard 0 g) :
    g.board.count = 0 := by
  cases h
  rw [GameState.init_board]
  rfl

/-- Count mod 10 = 8 at depth 7 (end of bag 1). -/
theorem count_mod_ten_at_phase_1 {g : GameState}
    (h : ReachableAt GameConfig.standard 7 g) :
    g.board.count % 10 = 8 := by
  have := h.standard_count_mod_ten
  omega

/-- Count mod 10 = 6 at depth 14 (end of bag 2). -/
theorem count_mod_ten_at_phase_2 {g : GameState}
    (h : ReachableAt GameConfig.standard 14 g) :
    g.board.count % 10 = 6 := by
  have := h.standard_count_mod_ten
  omega

/-- Count mod 10 = 4 at depth 21 (end of bag 3). -/
theorem count_mod_ten_at_phase_3 {g : GameState}
    (h : ReachableAt GameConfig.standard 21 g) :
    g.board.count % 10 = 4 := by
  have := h.standard_count_mod_ten
  omega

/-- Count mod 10 = 2 at depth 28 (end of bag 4). -/
theorem count_mod_ten_at_phase_4 {g : GameState}
    (h : ReachableAt GameConfig.standard 28 g) :
    g.board.count % 10 = 2 := by
  have := h.standard_count_mod_ten
  omega

/-- Count mod 10 = 0 at depth 35 (end of bag 5, i.e., reset). -/
theorem count_mod_ten_at_phase_5 {g : GameState}
    (h : ReachableAt GameConfig.standard 35 g) :
    g.board.count % 10 = 0 := by
  have := h.standard_count_mod_ten
  omega

/-- **The bag-boundary count residue returns after exactly five bags.**
One full bag drops 28 cells onto the board (7 pieces times 4 cells), and `28 ≡ 8 (mod 10)`,
so the standard-board count residue advances by 8 at every bag boundary. Five bags advance it
by `5 * 28 = 140 ≡ 0 (mod 10)`, so the residue at boundary `i + 5` equals the residue at
boundary `i`. This is the modular reason the reset block spans exactly five bags
(`bagsPerBlock = 5`) and clears exactly `targetLines = 14` rows: only after five bags does the
cell budget close back onto a multiple of ten, the necessary condition for the board to return
to a previously seen residue (and ultimately to `init`). Both states are taken at genuine bag
boundaries (`piecesPerBag * i`), where the bag is full and the residue is forced. -/
theorem count_mod_ten_returns_after_five_bags {i : ℕ} {g g' : GameState}
    (h : ReachableAt GameConfig.standard (piecesPerBag * i) g)
    (h' : ReachableAt GameConfig.standard (piecesPerBag * (i + 5)) g') :
    g'.board.count % 10 = g.board.count % 10 := by
  rw [standard_count_mod_ten_at_bag_boundary h,
      standard_count_mod_ten_at_bag_boundary h']
  omega

/-- **Five bags is the minimal residue period.** The companion lower bound to
`count_mod_ten_returns_after_five_bags`: for any positive `k < 5`, the standard-board count
residue at boundary `i + k` differs from the residue at boundary `i`, because `28 * k ≡ 8 * k`
runs through `8, 6, 4, 2 (mod 10)` for `k = 1, 2, 3, 4`, none of which is `0`. Together with the
period-five return this pins the bag-boundary residue period at *exactly* five — no shorter reset
block can return the board to its starting residue class, so a closed atlas cycle through `init`
must contain at least five bag boundaries. -/
theorem count_mod_ten_ne_before_five_bags {i k : ℕ} {g g' : GameState}
    (hk0 : 0 < k) (hk5 : k < 5)
    (h : ReachableAt GameConfig.standard (piecesPerBag * i) g)
    (h' : ReachableAt GameConfig.standard (piecesPerBag * (i + k)) g') :
    g'.board.count % 10 ≠ g.board.count % 10 := by
  rw [standard_count_mod_ten_at_bag_boundary h,
      standard_count_mod_ten_at_bag_boundary h']
  interval_cases k <;> omega

/-- The bag is full at depth 7. -/
theorem bag_full_at_phase_1 {g : GameState}
    (h : ReachableAt GameConfig.standard 7 g) :
    g.bag = Bag.full :=
  bag_eq_full_at_bag_boundary
    (show ReachableAt GameConfig.standard (piecesPerBag * 1) g from h)

/-- The bag is full at depth 14. -/
theorem bag_full_at_phase_2 {g : GameState}
    (h : ReachableAt GameConfig.standard 14 g) :
    g.bag = Bag.full :=
  bag_eq_full_at_bag_boundary
    (show ReachableAt GameConfig.standard (piecesPerBag * 2) g from h)

/-- The bag is full at depth 21. -/
theorem bag_full_at_phase_3 {g : GameState}
    (h : ReachableAt GameConfig.standard 21 g) :
    g.bag = Bag.full :=
  bag_eq_full_at_bag_boundary
    (show ReachableAt GameConfig.standard (piecesPerBag * 3) g from h)

/-- The bag is full at depth 28. -/
theorem bag_full_at_phase_4 {g : GameState}
    (h : ReachableAt GameConfig.standard 28 g) :
    g.bag = Bag.full :=
  bag_eq_full_at_bag_boundary
    (show ReachableAt GameConfig.standard (piecesPerBag * 4) g from h)

/-- The bag is full at depth 35 (reset boundary). -/
theorem bag_full_at_phase_5 {g : GameState}
    (h : ReachableAt GameConfig.standard 35 g) :
    g.bag = Bag.full :=
  bag_eq_full_at_bag_boundary
    (show ReachableAt GameConfig.standard (piecesPerBag * 5) g from h)

/-- **The strongest end-of-block characterisation**: at depth 35 with full
bag and count 0, the state is exactly `init`. This is the bridge between
the arithmetic conditions at the final boundary and the reset itself. -/
theorem eq_init_at_phase_5_of_count_zero
    {g : GameState}
    (h : ReachableAt GameConfig.standard 35 g)
    (hcount : g.board.count = 0) :
    g = GameState.init :=
  (GameState.eq_init_iff_count_zero_and_bag_full g).mpr
    ⟨hcount, bag_full_at_phase_5 h⟩

/-! ### Constraints on phase-board cell counts

Standard count at depth `n` is `4n - 10k` for some `k`, hence in
`{4n mod 10, 4n mod 10 + 10, ..., 4n}`. For each phase boundary we list
the (small) candidate counts. -/

/-- Count at depth 7 is in `{8, 18, 28}` — at most 2 lines cleared. -/
theorem count_at_phase_1_in {g : GameState}
    (h : ReachableAt GameConfig.standard 7 g) :
    g.board.count = 8 ∨ g.board.count = 18 ∨ g.board.count = 28 := by
  obtain ⟨k, hk⟩ := h.exists_count_eq
  change g.board.count + 10 * k = 28 at hk
  have hmod : g.board.count % 10 = 8 := count_mod_ten_at_phase_1 h
  omega

/-- Count at depth 14 is in `{6, 16, 26, 36, 46, 56}` — at most 5 lines cleared. -/
theorem count_at_phase_2_le_56 {g : GameState}
    (h : ReachableAt GameConfig.standard 14 g) :
    g.board.count ≤ 56 := by
  obtain ⟨k, hk⟩ := h.exists_count_eq
  change g.board.count + 10 * k = 56 at hk
  omega

/-- Count at depth 21 ≤ 84. -/
theorem count_at_phase_3_le_84 {g : GameState}
    (h : ReachableAt GameConfig.standard 21 g) :
    g.board.count ≤ 84 := by
  obtain ⟨k, hk⟩ := h.exists_count_eq
  change g.board.count + 10 * k = 84 at hk
  omega

/-- Count at depth 28 ≤ 112. -/
theorem count_at_phase_4_le_112 {g : GameState}
    (h : ReachableAt GameConfig.standard 28 g) :
    g.board.count ≤ 112 := by
  obtain ⟨k, hk⟩ := h.exists_count_eq
  change g.board.count + 10 * k = 112 at hk
  omega

/-- Count at depth 35 ≤ 140. -/
theorem count_at_phase_5_le_140 {g : GameState}
    (h : ReachableAt GameConfig.standard 35 g) :
    g.board.count ≤ 140 := by
  obtain ⟨k, hk⟩ := h.exists_count_eq
  change g.board.count + 10 * k = 140 at hk
  omega

/-! ### Putting it together: phase-board constraints

The decomposition theorem `Winning_35_of_phase_decomposition` takes 4
intermediate phase boards `P_1..P_4` as parameters. Any *useful* choice
must consist of states satisfying:

* `g ∈ P_i ⇒ ReachableAt GameConfig.standard (7*i) g`
* `g ∈ P_i ⇒ g.bag = Bag.full`
* `g ∈ P_i ⇒ g.board.count % 10 = (28 * i) % 10`
* `g ∈ P_i ⇒ ¬ g.lost GameConfig.standard`
* `g ∈ P_i ⇒ Board.WF GameConfig.standard g.board`

The lemmas above mechanise each of these characterisations. Combined with
the `BagWinning` boolean checker, the user can now construct candidate
phase boards by *enumerating count-constrained finite sets* rather than
searching the full state space. -/

/-- The full phase characterisation predicate: a state is a valid candidate
for the phase-`i` board if all four invariants hold simultaneously. -/
def PhaseStateOK (i : ℕ) (g : GameState) : Prop :=
  ReachableAt GameConfig.standard (piecesPerBag * i) g ∧
  g.bag = Bag.full ∧
  g.board.count % 10 = (28 * i) % 10 ∧
  ¬ g.lost GameConfig.standard ∧
  Board.WF GameConfig.standard g.board

/-- Every phase-i reachable state satisfies `PhaseStateOK i`. The phase
boards `P_i` may safely be restricted to subsets of `{g | PhaseStateOK i g}`. -/
theorem phaseStateOK_of_reachableAt
    {i : ℕ} {g : GameState}
    (h : ReachableAt GameConfig.standard (piecesPerBag * i) g)
    (hnl : ¬ g.lost GameConfig.standard) :
    PhaseStateOK i g := by
  refine ⟨h, ?_, ?_, hnl, ?_⟩
  · exact bag_eq_full_at_bag_boundary h
  · exact standard_count_mod_ten_at_bag_boundary h
  · exact h.wf

/-! ## Sanity checks and decidability verification

These trivial examples confirm the framework's `decide`/reflection
machinery works end-to-end at the smallest scale. They run instantly. -/

/-- Trivial base case: `Winning 0 init` is `init = init`. -/
example : Winning 0 GameState.init := by decide

/-- Boolean side agrees at depth 0. -/
example : winningBool 0 GameState.init = true := rfl

/-- `WinningTo {init} 0 init` is `init ∈ {init}`. -/
example : WinningTo {GameState.init} 0 GameState.init := by decide

/-- Boolean side agrees for the generalised target predicate. -/
example : winningToBool {GameState.init} 0 GameState.init = true := rfl

/-- Sanity check: composition with `m = n = 0` is the identity on
`WinningTo {init} 0` after rewriting `0 + 0 = 0`. -/
example {g : GameState} (h : WinningTo {GameState.init} 0 g) :
    WinningTo {GameState.init} (0 + 0) g :=
  h.compose (T₁ := {GameState.init}) (T₂ := {GameState.init})
    (fun _ hg' => hg')

/-! ## Phase-board helpers — constructing candidate `P_i` boards concretely

Practical helpers for writing candidate phase boards inline as Lean terms.
A typical phase-1 candidate is a state with 8 specific cells and a full bag;
the helpers below let you spell out such states without ceremony. -/

/-- Construct a `GameState` from an explicit list of filled cells and a full
bag. Useful for writing phase-board candidates inline. -/
def mkStateFullBag (cells : List Coord) : GameState :=
  ⟨cells.toFinset, Bag.full⟩

@[simp] theorem mkStateFullBag_board (cells : List Coord) :
    (mkStateFullBag cells).board = cells.toFinset := rfl

@[simp] theorem mkStateFullBag_bag (cells : List Coord) :
    (mkStateFullBag cells).bag = Bag.full := rfl

/-- A candidate phase-1 state with 8 cells in the bottom row's right half.
Concretely: cells (2, 0), (3, 0), …, (9, 0). Bag is full.

Cell count = 8 = (28 * 1) mod 10 ✓
Bag = Bag.full ✓
WF: all columns < 10 ✓
Not lost: all rows < 20 ✓

This is one of the simplest candidate Q boards for the `Winning_35_of_uniform_phase`
specialisation, satisfying the structural requirements without claim to
actual reachability. -/
def candidatePhase1_bottomRight8 : GameState :=
  mkStateFullBag [(2, 0), (3, 0), (4, 0), (5, 0),
                  (6, 0), (7, 0), (8, 0), (9, 0)]

/-- The candidate has the expected bag. -/
example : candidatePhase1_bottomRight8.bag = Bag.full := rfl

/-- The candidate has the expected cell count. -/
example : candidatePhase1_bottomRight8.board.count = 8 := by
  rw [candidatePhase1_bottomRight8, mkStateFullBag_board]
  decide

/-! ## Per-piece decomposition: split each 7-piece bag into 7 single-step obligations

The most search-intensive part of `BagWinning` is the depth-7 AND/OR tree
(∀ pieces × ∃ placements × 7 deep). Decomposing further into 7 one-step
obligations with explicit intermediate boards `Q₁..Q₆` shrinks each search
to depth 1, at the cost of choosing 6 intermediate sets. -/

/-- One-step bag-winning: `BagStep P_in P_out` means every state in `P_in`
can take exactly one valid adversarial step into `P_out`. -/
def BagStep (P_in P_out : Finset GameState) : Prop :=
  ∀ g ∈ P_in, WinningTo P_out 1 g

/-- A `BagStep` is exactly `BagWinningN` at depth 1. -/
theorem bagStep_iff_bagWinning_one (P_in P_out : Finset GameState) :
    BagStep P_in P_out ↔ ∀ g ∈ P_in, WinningTo P_out 1 g := Iff.rfl

/-- Generalised n-step bag-winning predicate. -/
def BagWinningN (P_in P_out : Finset GameState) (n : ℕ) : Prop :=
  ∀ g ∈ P_in, WinningTo P_out n g

@[simp] theorem bagWinningN_zero (P_in P_out : Finset GameState) :
    BagWinningN P_in P_out 0 ↔ ∀ g ∈ P_in, g ∈ P_out := by
  rw [BagWinningN]
  rfl

theorem BagWinning_eq_bagWinningN_seven (P_in P_out : Finset GameState) :
    BagWinning P_in P_out ↔ BagWinningN P_in P_out piecesPerBag := Iff.rfl

/-- **Generic composition of bag-winning at arbitrary depths**: chains two
`BagWinningN` obligations through an intermediate set. -/
theorem BagWinningN.compose {P_a P_b P_c : Finset GameState} {m n : ℕ}
    (h₁ : BagWinningN P_a P_b m) (h₂ : BagWinningN P_b P_c n) :
    BagWinningN P_a P_c (m + n) := by
  intro g hg
  exact (h₁ g hg).compose h₂

/-- **Single-step composition specialisation**: chains a single `BagStep`
with an `n`-step continuation, producing an `(n+1)`-step obligation. -/
theorem BagStep.cons {P_a P_b P_c : Finset GameState} {n : ℕ}
    (h₁ : BagStep P_a P_b) (h₂ : BagWinningN P_b P_c n) :
    BagWinningN P_a P_c (n + 1) := by
  rw [show n + 1 = 1 + n from by omega]
  intro g hg
  exact (h₁ g hg).compose h₂

/-- **7-step bag decomposition**: a single `BagWinning P_in P_out` obligation
splits into 7 chained `BagStep` obligations through 6 intermediate boards
`Q₁..Q₆`. Each `BagStep` is a depth-1 obligation (∀ piece × ∃ placement),
which is dramatically smaller than the depth-7 nested AND/OR. -/
theorem BagWinning_of_seven_BagSteps
    (P_in P_out : Finset GameState)
    (Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ : Finset GameState)
    (h1 : BagStep P_in Q₁)
    (h2 : BagStep Q₁ Q₂)
    (h3 : BagStep Q₂ Q₃)
    (h4 : BagStep Q₃ Q₄)
    (h5 : BagStep Q₄ Q₅)
    (h6 : BagStep Q₅ Q₆)
    (h7 : BagStep Q₆ P_out) :
    BagWinning P_in P_out := by
  rw [BagWinning_eq_bagWinningN_seven]
  -- Compose 7 BagSteps: h1 ; h2 ; h3 ; h4 ; h5 ; h6 ; h7
  -- Total depth: 1 + 1 + 1 + 1 + 1 + 1 + 1 = 7 = piecesPerBag
  have c7 : BagWinningN Q₆ P_out 1 := h7
  have c67 : BagWinningN Q₅ P_out 2 := h6.cons c7
  have c567 : BagWinningN Q₄ P_out 3 := h5.cons c67
  have c4567 : BagWinningN Q₃ P_out 4 := h4.cons c567
  have c34567 : BagWinningN Q₂ P_out 5 := h3.cons c4567
  have c234567 : BagWinningN Q₁ P_out 6 := h2.cons c34567
  have c1234567 : BagWinningN P_in P_out 7 := h1.cons c234567
  exact c1234567

/-! ### Decidable BagStep -/

/-- Boolean version of `BagStep`. -/
def bagStepBool (P_in P_out : Finset GameState) : Bool :=
  decide (∀ g ∈ P_in, winningToBool P_out 1 g = true)

/-- Reflection for `BagStep`. -/
theorem bagStepBool_eq_true_iff (P_in P_out : Finset GameState) :
    bagStepBool P_in P_out = true ↔ BagStep P_in P_out := by
  simp only [bagStepBool, decide_eq_true_eq, BagStep]
  refine ⟨fun h g hg => (winningToBool_eq_true_iff _ _ _).mp (h g hg),
          fun h g hg => (winningToBool_eq_true_iff _ _ _).mpr (h g hg)⟩

instance BagStep.decidable (P_in P_out : Finset GameState) :
    Decidable (BagStep P_in P_out) :=
  decidable_of_iff _ (bagStepBool_eq_true_iff P_in P_out)

/-- **Boolean form of the 7-step decomposition**: 7 single-step boolean
checks suffice to prove a single `BagWinning` obligation. -/
theorem BagWinning_of_seven_bagStepBool
    (P_in P_out : Finset GameState)
    (Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ : Finset GameState)
    (h1 : bagStepBool P_in Q₁ = true)
    (h2 : bagStepBool Q₁ Q₂ = true)
    (h3 : bagStepBool Q₂ Q₃ = true)
    (h4 : bagStepBool Q₃ Q₄ = true)
    (h5 : bagStepBool Q₄ Q₅ = true)
    (h6 : bagStepBool Q₅ Q₆ = true)
    (h7 : bagStepBool Q₆ P_out = true) :
    BagWinning P_in P_out :=
  BagWinning_of_seven_BagSteps P_in P_out Q₁ Q₂ Q₃ Q₄ Q₅ Q₆
    ((bagStepBool_eq_true_iff _ _).mp h1)
    ((bagStepBool_eq_true_iff _ _).mp h2)
    ((bagStepBool_eq_true_iff _ _).mp h3)
    ((bagStepBool_eq_true_iff _ _).mp h4)
    ((bagStepBool_eq_true_iff _ _).mp h5)
    ((bagStepBool_eq_true_iff _ _).mp h6)
    ((bagStepBool_eq_true_iff _ _).mp h7)

/-- **Maximal decomposition**: combine the 5-bag decomposition (Tier 1) with
the 7-step per-bag decomposition. The full `Winning 35 init` obligation
becomes `5 × 7 = 35` independent `bagStepBool` checks through 34 intermediate
phase sets `Qᵢⱼ` (5 bags × 6 intermediate-piece boards per bag, with the
4 mid-bag boundaries `P₁..P₄` reused across bag boundaries). -/
theorem Winning_35_of_thirtyFive_bagStepBool
    (P₁ P₂ P₃ P₄ : Finset GameState)
    (Q11 Q12 Q13 Q14 Q15 Q16 : Finset GameState)
    (Q21 Q22 Q23 Q24 Q25 Q26 : Finset GameState)
    (Q31 Q32 Q33 Q34 Q35 Q36 : Finset GameState)
    (Q41 Q42 Q43 Q44 Q45 Q46 : Finset GameState)
    (Q51 Q52 Q53 Q54 Q55 Q56 : Finset GameState)
    -- Bag 1: {init} → Q11 → Q12 → ... → Q16 → P₁
    (s11 : bagStepBool {GameState.init} Q11 = true)
    (s12 : bagStepBool Q11 Q12 = true)
    (s13 : bagStepBool Q12 Q13 = true)
    (s14 : bagStepBool Q13 Q14 = true)
    (s15 : bagStepBool Q14 Q15 = true)
    (s16 : bagStepBool Q15 Q16 = true)
    (s17 : bagStepBool Q16 P₁ = true)
    -- Bag 2: P₁ → Q21 → ... → Q26 → P₂
    (s21 : bagStepBool P₁ Q21 = true)
    (s22 : bagStepBool Q21 Q22 = true)
    (s23 : bagStepBool Q22 Q23 = true)
    (s24 : bagStepBool Q23 Q24 = true)
    (s25 : bagStepBool Q24 Q25 = true)
    (s26 : bagStepBool Q25 Q26 = true)
    (s27 : bagStepBool Q26 P₂ = true)
    -- Bag 3: P₂ → Q31 → ... → Q36 → P₃
    (s31 : bagStepBool P₂ Q31 = true)
    (s32 : bagStepBool Q31 Q32 = true)
    (s33 : bagStepBool Q32 Q33 = true)
    (s34 : bagStepBool Q33 Q34 = true)
    (s35 : bagStepBool Q34 Q35 = true)
    (s36 : bagStepBool Q35 Q36 = true)
    (s37 : bagStepBool Q36 P₃ = true)
    -- Bag 4: P₃ → Q41 → ... → Q46 → P₄
    (s41 : bagStepBool P₃ Q41 = true)
    (s42 : bagStepBool Q41 Q42 = true)
    (s43 : bagStepBool Q42 Q43 = true)
    (s44 : bagStepBool Q43 Q44 = true)
    (s45 : bagStepBool Q44 Q45 = true)
    (s46 : bagStepBool Q45 Q46 = true)
    (s47 : bagStepBool Q46 P₄ = true)
    -- Bag 5: P₄ → Q51 → ... → Q56 → {init}
    (s51 : bagStepBool P₄ Q51 = true)
    (s52 : bagStepBool Q51 Q52 = true)
    (s53 : bagStepBool Q52 Q53 = true)
    (s54 : bagStepBool Q53 Q54 = true)
    (s55 : bagStepBool Q54 Q55 = true)
    (s56 : bagStepBool Q55 Q56 = true)
    (s57 : bagStepBool Q56 {GameState.init} = true) :
    Winning blockLength GameState.init := by
  apply Winning_35_of_bagWinningBool P₁ P₂ P₃ P₄
  · rw [bagWinningBool_eq_true_iff]
    exact BagWinning_of_seven_bagStepBool _ _ Q11 Q12 Q13 Q14 Q15 Q16
            s11 s12 s13 s14 s15 s16 s17
  · rw [bagWinningBool_eq_true_iff]
    exact BagWinning_of_seven_bagStepBool _ _ Q21 Q22 Q23 Q24 Q25 Q26
            s21 s22 s23 s24 s25 s26 s27
  · rw [bagWinningBool_eq_true_iff]
    exact BagWinning_of_seven_bagStepBool _ _ Q31 Q32 Q33 Q34 Q35 Q36
            s31 s32 s33 s34 s35 s36 s37
  · rw [bagWinningBool_eq_true_iff]
    exact BagWinning_of_seven_bagStepBool _ _ Q41 Q42 Q43 Q44 Q45 Q46
            s41 s42 s43 s44 s45 s46 s47
  · rw [bagWinningBool_eq_true_iff]
    exact BagWinning_of_seven_bagStepBool _ _ Q51 Q52 Q53 Q54 Q55 Q56
            s51 s52 s53 s54 s55 s56 s57

/-! ## Concrete sanity-check: `¬ Winning 1 init`

A direct proof that the framework correctly identifies impossibility:
after exactly one step from `init`, the resulting state cannot be `init`
(the bag has shrunk from 7 to 6 pieces). This validates that the
`Winning` predicate has the right semantics and that the framework
detects unreachable targets correctly. -/

/-- After any adversarial step from `init`, the bag has 6 pieces, not 7. -/
theorem adversarialStep_init_bag_card
    (p : Piece) (pl : Placement) :
    (adversarialStep GameConfig.standard GameState.init p pl).bag.card = 6 := by
  rw [adversarialStep_bag, GameState.init_bag]
  exact Bag.card_draw_of_ge_two
    (by exact Bag.mem_full p)
    (by rw [Bag.full_card]; omega)

/-- `Winning 1 init` is impossible: after 1 step we cannot be back at `init`
(bag has shrunk). The single-step reset is arithmetically ruled out. -/
theorem not_winning_one_init :
    ¬ Winning 1 GameState.init := by
  intro h
  obtain ⟨_, hstep⟩ := h
  have hp : Piece.O ∈ GameState.init.bag :=
    GameState.init_bag.symm ▸ Bag.mem_full Piece.O
  obtain ⟨pl, _, hwin⟩ := hstep Piece.O hp
  rw [winning_zero_iff] at hwin
  -- hwin : adversarialStep ... = init
  -- LHS bag card = 6; init bag card = 7; contradiction.
  have hcard : (adversarialStep GameConfig.standard GameState.init Piece.O pl).bag.card = 6 :=
    adversarialStep_init_bag_card Piece.O pl
  rw [hwin, GameState.init_bag_card] at hcard
  omega

/-- The boolean checker agrees: `winningBool 1 init = false`. -/
theorem winningBool_one_init_eq_false :
    winningBool 1 GameState.init = false := by
  by_contra h
  -- h : winningBool 1 init ≠ false, so it's true
  have htrue : winningBool 1 GameState.init = true := by
    cases hr : winningBool 1 GameState.init
    · exact absurd hr h
    · rfl
  have hwin : Winning 1 GameState.init :=
    (winningBool_eq_true_iff _ _).mp htrue
  exact not_winning_one_init hwin

/-- And, equivalently, the same statement via the `BagWinning` lens:
`{init}` cannot be its own immediate successor in any single step. -/
theorem not_bagStep_init_singleton :
    ¬ BagStep {GameState.init} {GameState.init} := by
  intro h
  have hwin := h GameState.init (Finset.mem_singleton.mpr rfl)
  -- hwin : WinningTo {init} 1 init
  -- Convert to Winning 1 init via bridge.
  have : Winning 1 GameState.init :=
    (winning_iff_winningTo_init 1 GameState.init).mpr hwin
  exact not_winning_one_init this

/-! ### Generalised impossibility: any `Winning n init` for `n ∉ multiples-of-35`

For `n` not divisible by 35, `Winning n init` is impossible. Two
arguments combine:

* **Cell-count divisibility**: returning to `init` requires
  `4n - 10·linesCleared = 0`, hence `n` divisible by 5.
* **Bag refill divisibility**: returning to `init` requires the bag back
  to `Bag.full`, which only happens at `n` divisible by 7.

The intersection is `n` divisible by lcm(5,7) = 35. So only multiples
of 35 are arithmetically *possible* depths for a full reset.

This is a powerful constraint — it tells us that the 5-bag reset
(`Winning 35 init`) is the *minimal* candidate; no shorter reset is
even arithmetically feasible. -/

/-- For `Winning n init` to hold, we need a trace returning to init in `n`
moves. Such a trace must satisfy both arithmetic constraints, so `n` is
divisible by 5 (cell-count) and 7 (bag-refill).

This lemma extracts the bag-refill direction: if there's a trace from init
back to init in `n` moves, then `n` is a multiple of 7. -/
theorem dvd_seven_of_winning_init_aux
    {n : ℕ} (σ : Solver GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    (htr : adversarialTrace GameConfig.standard σ s GameState.init n = GameState.init) :
    7 ∣ n := by
  -- The trace's bag at step n equals init.bag = Bag.full.
  -- Also, trace bag at step n equals bagAt Bag.full s n.
  -- So bagAt Bag.full s n = Bag.full, hence (bagAt _ s n).card = 7.
  -- By bag-card-cycle: card = 7 - n % 7. Setting = 7 gives n % 7 = 0.
  have hbag :
      (adversarialTrace GameConfig.standard σ s GameState.init n).bag = Bag.full := by
    rw [htr, GameState.init_bag]
  rw [adversarialTrace_bag] at hbag
  -- hbag : bagAt Bag.full s n = Bag.full
  -- The bag's card cycles: card (bagAt full s n) only depends on n mod 7.
  -- For card = 7, we need n mod 7 = 0.
  have hcard : (bagAt Bag.full s n).card = 7 := by rw [hbag]; exact Bag.full_card
  -- We use the existing helper `bagAt_card_eq_seven_sub` for n ≤ 6.
  -- For larger n, induct via bag-cycle.
  -- Alternative: directly use the bag-card induction.
  -- The cleanest path: use the existing bagAt_mul_seven_eq_full for the converse
  -- and inductive argument for forward.
  -- We argue by induction on n via mod 7.
  -- Use mod-7 case split.
  have : n % 7 = 0 := by
    -- The bag at any depth has card = 7 - (n % 7). Apply at n.
    have hgen : ∀ k : ℕ, (bagAt Bag.full s k).card = 7 - k % 7 := by
      intro k
      -- proven by induction on k, similar to bagAt_card_eq_seven_sub
      induction k with
      | zero =>
          rw [bagAt_zero, Bag.full_card]
      | succ k ih =>
          have hk7 : k % 7 < 7 := Nat.mod_lt k (by decide)
          by_cases hk6 : k % 7 = 6
          · -- Bag will refill: card goes from 1 to 7.
            have hkcard : (bagAt Bag.full s k).card = 1 := by rw [ih]; omega
            rw [bagAt_succ, Bag.card_draw_of_singleton (hl k) hkcard]
            have : (k + 1) % 7 = 0 := by omega
            rw [this]
          · have hge2 : 2 ≤ (bagAt Bag.full s k).card := by rw [ih]; omega
            rw [bagAt_succ, Bag.card_draw_of_ge_two (hl k) hge2, ih]
            have hpre : (k + 1) % 7 = k % 7 + 1 := by omega
            rw [hpre]
            omega
    have happ := hgen n
    rw [hcard] at happ
    omega
  exact Nat.dvd_of_mod_eq_zero this

/-! The headline consequence — `Winning n init → 7 ∣ n` — follows from
`dvd_seven_of_winning_init_aux` once we have a `Winning`-to-trace extraction
helper. The current `winningSolver` machinery (defined for `blockLength = 35`)
gives this for n = 35; a parametric version would generalise. -/

/-! ### Count + bag bridge for singleton-target `WinningTo`

The key lemma: if `WinningTo {g_end} n g₀`, then along *any* legal sequence
from `g₀.bag` the cell count and bag end up matching `g_end`'s. This packages
the structural consequence of WinningTo without needing to extract a Solver. -/

/-- **Count + bag bridge**: from `WinningTo {g_end} n g₀`, the endpoint
`g_end`'s cell count is `g₀.count + 4n - 10·k` for some line-clear count `k`,
and `g_end.bag` equals the deterministic bag state `bagAt g₀.bag s n` for
*any* legal sequence `s` from `g₀.bag`. -/
theorem WinningTo.singleton_count_bag_bridge :
    ∀ {n : ℕ} {g₀ g_end : GameState},
      WinningTo {g_end} n g₀ →
      Board.WF GameConfig.standard g₀.board →
      ∀ (s : ℕ → Piece), LegalSequenceFrom g₀.bag s →
      (∃ k : ℕ, g_end.board.count + 10 * k = g₀.board.count + 4 * n) ∧
      g_end.bag = bagAt g₀.bag s n := by
  intro n
  induction n with
  | zero =>
      intro g₀ g_end h _hwf s _hl
      rw [winningTo_zero_iff, Finset.mem_singleton] at h
      subst h
      refine ⟨⟨0, by ring⟩, ?_⟩
      rw [bagAt_zero]
  | succ n ih =>
      intro g₀ g_end h hwf s hl
      rw [winningTo_succ_iff] at h
      obtain ⟨_, hstep⟩ := h
      have hp : s 0 ∈ g₀.bag := by
        have hcd := hl 0
        rw [bagAt_zero, Bag.canDraw_iff_mem] at hcd
        exact hcd
      obtain ⟨pl₀, hpl₀, hnext⟩ := hstep (s 0) hp
      have hpl₀_spec := (Placement.mem_allValidFor _ _ _).mp hpl₀
      have hpl₀_piece : pl₀.piece = s 0 := hpl₀_spec.1
      have hpl₀_valid : pl₀.Valid GameConfig.standard := hpl₀_spec.2
      have heta : ({ pl₀ with piece := s 0 } : Placement) = pl₀ :=
        Placement.eta_of_piece_eq hpl₀_piece
      -- next state's board
      have hbag_next :
          (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag
            = g₀.bag.draw (s 0) := rfl
      have hboard_next :
          (adversarialStep GameConfig.standard g₀ (s 0) pl₀).board
            = Placement.applyStep GameConfig.standard g₀.board pl₀ := by
        rw [adversarialStep_board, heta]
      have hwf' :
          Board.WF GameConfig.standard
            (adversarialStep GameConfig.standard g₀ (s 0) pl₀).board := by
        rw [hboard_next]
        exact Placement.applyStep_wf hwf hpl₀_valid
      -- Suffix legality
      have hsuffix :
          LegalSequenceFrom (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag
            (fun k => s (k + 1)) := by
        have hsuf := hl.suffix 1
        have hbag1_eq : bagAt g₀.bag s 1
            = (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag := by
          rw [bagAt_succ, bagAt_zero, hbag_next]
        rw [hbag1_eq] at hsuf
        convert hsuf using 1
        ext k; ring_nf
      -- Apply IH
      obtain ⟨⟨k', hcount'⟩, hbag_eq⟩ := ih hnext hwf' _ hsuffix
      -- Count equation from applyStep_count
      have hstep_count :
          (adversarialStep GameConfig.standard g₀ (s 0) pl₀).board.count
            + 10 * Board.linesCleared GameConfig.standard (pl₀.place g₀.board)
            = g₀.board.count + 4 := by
        rw [hboard_next]
        have h1 := applyStep_count GameConfig.standard g₀.board pl₀ hwf hpl₀_valid
        change (Placement.applyStep GameConfig.standard g₀.board pl₀).count
                + 10 * Board.linesCleared GameConfig.standard (pl₀.place g₀.board)
              = g₀.board.count + 4
        exact h1
      refine ⟨?_, ?_⟩
      · refine ⟨k' + Board.linesCleared GameConfig.standard (pl₀.place g₀.board), ?_⟩
        -- From IH: g_end.count + 10*k' = (step ...).board.count + 4*n
        -- From hstep_count: (step ...).board.count + 10*lines = g₀.count + 4
        -- Want: g_end.count + 10*(k' + lines) = g₀.count + 4*(n+1)
        have hmul : 10 * (k' + Board.linesCleared GameConfig.standard (pl₀.place g₀.board))
                  = 10 * k' + 10 * Board.linesCleared GameConfig.standard (pl₀.place g₀.board) :=
          Nat.mul_add 10 k' _
        have hexp : 4 * (n + 1) = 4 * n + 4 := by ring
        rw [hmul, hexp]
        omega
      · -- Combine bag equations
        rw [hbag_eq]
        have h1 : (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag
                = bagAt g₀.bag s 1 := rfl
        rw [h1]
        have h2 := bagAt_add g₀.bag s 1 n
        have h3 : bagAt g₀.bag s (n + 1) = bagAt g₀.bag s (1 + n) := by
          congr 1; ring
        rw [h3, h2]
        congr 1
        ext k; ring_nf

/-- **5 ∣ n** whenever `Winning n init` holds: cell-count divisibility.
At the end of any winning trace from init to init, count is 0, hence
`0 + 10·k = 0 + 4n`, so `5k = 2n`, so `5 ∣ 2n`, so `5 ∣ n`. -/
theorem dvd_five_of_winning_init {n : ℕ}
    (h : Winning n GameState.init) : 5 ∣ n := by
  have hwt : WinningTo {GameState.init} n GameState.init :=
    (winning_iff_winningTo_init n GameState.init).mp h
  obtain ⟨s, hl⟩ := exists_legalSequence
  have hwf : Board.WF GameConfig.standard GameState.init.board := by
    rw [GameState.init_board]; exact Board.empty_wf _
  obtain ⟨⟨k, hk⟩, _⟩ :=
    WinningTo.singleton_count_bag_bridge hwt hwf s hl
  -- hk : init.count + 10*k = init.count + 4*n
  -- init.count = 0, so 10*k = 4*n, so 5*k = 2*n, so 5 ∣ 2*n, so 5 ∣ n
  rw [GameState.init_board_count] at hk
  -- hk : 0 + 10*k = 0 + 4*n, i.e., 10*k = 4*n
  -- Hence 5 ∣ n via (5 ∣ 2*n) ∧ Nat.Coprime 5 2
  have h2n : 10 * k = 4 * n := by omega
  have : 5 ∣ 2 * n := ⟨k, by omega⟩
  exact (Nat.Coprime.dvd_of_dvd_mul_left (by decide : Nat.Coprime 5 2) this)

/-- The bag-card residue formula, factored from `dvd_seven_of_winning_init_aux`. -/
theorem bagAt_card_eq_residue
    {s : ℕ → Piece} (hl : LegalSequence s) (k : ℕ) :
    (bagAt Bag.full s k).card = 7 - k % 7 := by
  induction k with
  | zero =>
      rw [bagAt_zero, Bag.full_card]
  | succ k ih =>
      have hk7 : k % 7 < 7 := Nat.mod_lt k (by decide)
      by_cases hk6 : k % 7 = 6
      · have hkcard : (bagAt Bag.full s k).card = 1 := by rw [ih]; omega
        rw [bagAt_succ, Bag.card_draw_of_singleton (hl k) hkcard]
        have : (k + 1) % 7 = 0 := by omega
        rw [this]
      · have hge2 : 2 ≤ (bagAt Bag.full s k).card := by rw [ih]; omega
        rw [bagAt_succ, Bag.card_draw_of_ge_two (hl k) hge2, ih]
        have hpre : (k + 1) % 7 = k % 7 + 1 := by omega
        rw [hpre]
        omega

/-- **7 ∣ n** whenever `Winning n init` holds: bag-refill divisibility.
At the end of any winning trace from init to init, the bag is `Bag.full`,
which only happens at depths `n ≡ 0 mod 7`. -/
theorem dvd_seven_of_winning_init {n : ℕ}
    (h : Winning n GameState.init) : 7 ∣ n := by
  have hwt : WinningTo {GameState.init} n GameState.init :=
    (winning_iff_winningTo_init n GameState.init).mp h
  obtain ⟨s, hl⟩ := exists_legalSequence
  have hwf : Board.WF GameConfig.standard GameState.init.board := by
    rw [GameState.init_board]; exact Board.empty_wf _
  obtain ⟨_, hbag⟩ :=
    WinningTo.singleton_count_bag_bridge hwt hwf s hl
  -- hbag : init.bag = bagAt init.bag s n
  -- init.bag = Bag.full, so bagAt Bag.full s n = Bag.full
  rw [GameState.init_bag] at hbag
  -- (bagAt Bag.full s n).card = 7 (since bagAt = Bag.full)
  have hcard : (bagAt Bag.full s n).card = 7 := by
    rw [← hbag]; exact Bag.full_card
  -- Combined with bagAt_card_eq_residue: 7 = 7 - n % 7, so n % 7 = 0
  have := bagAt_card_eq_residue hl n
  rw [hcard] at this
  exact Nat.dvd_of_mod_eq_zero (by omega)

/-- **35 ∣ n** combining the two divisibility constraints. The smallest
candidate reset depth is `35`, exactly the 5-bag block — confirming the
experiment targets the *minimal* arithmetically feasible reset. -/
theorem dvd_thirtyfive_of_winning_init {n : ℕ}
    (h : Winning n GameState.init) : 35 ∣ n := by
  have h5 := dvd_five_of_winning_init h
  have h7 := dvd_seven_of_winning_init h
  -- lcm(5,7) = 35; gcd(5,7) = 1, so coprime, so 5*7 = 35 ∣ n
  exact Nat.Coprime.mul_dvd_of_dvd_of_dvd (by decide : Nat.Coprime 5 7) h5 h7

/-- **Direct consequence**: `Winning 1 init`, `Winning 7 init`,
`Winning 14 init`, ..., `Winning 34 init` are all impossible. The only
positive `n < 35` for which `Winning n init` could conceivably hold is
none — every multiple of 35 less than 35 is just 0.

So `Winning 0 init` (which is trivially `init = init`) and
`Winning 35 init` are the only candidates `≤ 35`. -/
theorem not_winning_lt_35_init {n : ℕ} (hn : 0 < n) (hlt : n < 35)
    (h : Winning n GameState.init) : False := by
  have h35 := dvd_thirtyfive_of_winning_init h
  -- 35 ∣ n, n > 0 ⟹ n ≥ 35, contradicting n < 35
  have := Nat.le_of_dvd hn h35
  omega

/-! ### Set-form count + bag bridge

Generalises `WinningTo.singleton_count_bag_bridge` to arbitrary target
sets `T`. For any `WinningTo T n g₀`, the actual endpoint reached by a
chosen strategy is *some* element of `T` — and that element's count and
bag are determined by the count + bag invariants. -/

/-- **Set-form count + bag bridge**: from `WinningTo T n g₀` and a legal
sequence, some element `g_end ∈ T` is the endpoint satisfying the
count + bag invariants. -/
theorem WinningTo.set_count_bag_bridge :
    ∀ {n : ℕ} {T : Finset GameState} {g₀ : GameState},
      WinningTo T n g₀ →
      Board.WF GameConfig.standard g₀.board →
      ∀ (s : ℕ → Piece), LegalSequenceFrom g₀.bag s →
      ∃ g_end ∈ T,
        (∃ k : ℕ, g_end.board.count + 10 * k = g₀.board.count + 4 * n) ∧
        g_end.bag = bagAt g₀.bag s n := by
  intro n
  induction n with
  | zero =>
      intro T g₀ h _hwf s _hl
      refine ⟨g₀, h, ⟨0, by ring⟩, ?_⟩
      rw [bagAt_zero]
  | succ n ih =>
      intro T g₀ h hwf s hl
      rw [winningTo_succ_iff] at h
      obtain ⟨_, hstep⟩ := h
      have hp : s 0 ∈ g₀.bag := by
        have hcd := hl 0
        rw [bagAt_zero, Bag.canDraw_iff_mem] at hcd
        exact hcd
      obtain ⟨pl₀, hpl₀, hnext⟩ := hstep (s 0) hp
      have hpl₀_spec := (Placement.mem_allValidFor _ _ _).mp hpl₀
      have hpl₀_piece : pl₀.piece = s 0 := hpl₀_spec.1
      have hpl₀_valid : pl₀.Valid GameConfig.standard := hpl₀_spec.2
      have heta : ({ pl₀ with piece := s 0 } : Placement) = pl₀ :=
        Placement.eta_of_piece_eq hpl₀_piece
      have hbag_next :
          (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag
            = g₀.bag.draw (s 0) := rfl
      have hboard_next :
          (adversarialStep GameConfig.standard g₀ (s 0) pl₀).board
            = Placement.applyStep GameConfig.standard g₀.board pl₀ := by
        rw [adversarialStep_board, heta]
      have hwf' :
          Board.WF GameConfig.standard
            (adversarialStep GameConfig.standard g₀ (s 0) pl₀).board := by
        rw [hboard_next]
        exact Placement.applyStep_wf hwf hpl₀_valid
      have hsuffix :
          LegalSequenceFrom (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag
            (fun k => s (k + 1)) := by
        have hsuf := hl.suffix 1
        have hbag1_eq : bagAt g₀.bag s 1
            = (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag := by
          rw [bagAt_succ, bagAt_zero, hbag_next]
        rw [hbag1_eq] at hsuf
        convert hsuf using 1
        ext k; ring_nf
      obtain ⟨g_end, hg_end_mem, ⟨k', hcount'⟩, hbag_eq⟩ :=
        ih hnext hwf' _ hsuffix
      have hstep_count :
          (adversarialStep GameConfig.standard g₀ (s 0) pl₀).board.count
            + 10 * Board.linesCleared GameConfig.standard (pl₀.place g₀.board)
            = g₀.board.count + 4 := by
        rw [hboard_next]
        exact applyStep_count GameConfig.standard g₀.board pl₀ hwf hpl₀_valid
      refine ⟨g_end, hg_end_mem, ?_, ?_⟩
      · refine ⟨k' + Board.linesCleared GameConfig.standard (pl₀.place g₀.board), ?_⟩
        have hmul : 10 * (k' + Board.linesCleared GameConfig.standard (pl₀.place g₀.board))
                  = 10 * k' + 10 * Board.linesCleared GameConfig.standard (pl₀.place g₀.board) :=
          Nat.mul_add 10 k' _
        have hexp : 4 * (n + 1) = 4 * n + 4 := by ring
        rw [hmul, hexp]
        omega
      · rw [hbag_eq]
        have h1 : (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag
                = bagAt g₀.bag s 1 := rfl
        rw [h1]
        have h2 := bagAt_add g₀.bag s 1 n
        have h3 : bagAt g₀.bag s (n + 1) = bagAt g₀.bag s (1 + n) := by
          congr 1; ring
        rw [h3, h2]
        congr 1
        ext k; ring_nf

/-! ### Necessary conditions on each phase board `P_i`

Direct corollaries of `set_count_bag_bridge`. -/

/-- **`P_1` is nonempty and contains a state with count ≡ 8 (mod 10)
and full bag.** Any candidate `P_1` admitting `BagWinning {init} P_1`
must intersect the set of (count residue 8, full bag) states. -/
theorem BagWinning_init_implies_P1_has_residue
    (P_1 : Finset GameState)
    (h : BagWinning {GameState.init} P_1) :
    ∃ g ∈ P_1, g.board.count % 10 = 8 ∧ g.bag = Bag.full := by
  have hwt : WinningTo P_1 piecesPerBag GameState.init :=
    h GameState.init (Finset.mem_singleton.mpr rfl)
  obtain ⟨s, hl⟩ := exists_legalSequence
  -- Convert LegalSequence to LegalSequenceFrom GameState.init.bag
  have hl' : LegalSequenceFrom GameState.init.bag s := by
    rw [GameState.init_bag]; exact hl
  have hwf : Board.WF GameConfig.standard GameState.init.board := by
    rw [GameState.init_board]; exact Board.empty_wf _
  obtain ⟨g_end, hmem, ⟨k, hcount⟩, hbag⟩ :=
    WinningTo.set_count_bag_bridge hwt hwf s hl'
  refine ⟨g_end, hmem, ?_, ?_⟩
  · -- count: g_end.count + 10k = 0 + 4*7 = 28 ⟹ count ≡ 28 ≡ 8 (mod 10)
    rw [GameState.init_board_count, piecesPerBag_eq] at hcount
    omega
  · -- bag: g_end.bag = bagAt init.bag s 7 = bagAt Bag.full s 7 = Bag.full
    rw [hbag, GameState.init_bag, piecesPerBag_eq]
    exact bagAt_seven_eq_full hl

/-- **Closing-leg residue**: for any candidate `P_4` admitting
`BagWinning P_4 {init}`, every element of `P_4` that is well-formed and
has a full bag (i.e., every strategy-reachable element) must have count
residue 2 (mod 10).

The full-bag hypothesis is automatic for strategy-reachable states
because `bagAt_blockLength_eq_full` forces the bag to be full after
every multiple of 7 draws from `Bag.full`. -/
theorem BagWinning_close_init_implies_P4_residue
    (P_4 : Finset GameState)
    (h : BagWinning P_4 {GameState.init})
    (g : GameState) (hg : g ∈ P_4)
    (hwf : Board.WF GameConfig.standard g.board)
    (hbag : g.bag = Bag.full) :
    g.board.count % 10 = 2 := by
  have hwt : WinningTo {GameState.init} piecesPerBag g := h g hg
  -- Construct a legal sequence from g.bag = Bag.full
  obtain ⟨s, hl⟩ : ∃ s : ℕ → Piece, LegalSequenceFrom g.bag s := by
    rw [hbag]
    exact exists_legalSequenceFrom Bag.full Bag.full_nonempty
  obtain ⟨g_end, hmem, ⟨k, hcount⟩, _hbag_end⟩ :=
    WinningTo.set_count_bag_bridge hwt hwf s hl
  -- g_end ∈ {init}, so g_end = init, so init.count + 10k = g.count + 4·7
  rw [Finset.mem_singleton] at hmem
  rw [hmem, GameState.init_board_count, piecesPerBag_eq] at hcount
  -- 0 + 10k = g.count + 28, so g.count = 10k - 28, residue 2
  omega

/-! ### Full bridge with WF preservation

For iterating across multiple bags, we need WF and bag at the endpoint
of one bag to feed into the next. Bundle all properties together. -/

/-- **Full bridge**: extends `set_count_bag_bridge` with the additional
conclusion that the endpoint is well-formed. -/
theorem WinningTo.set_full_bridge :
    ∀ {n : ℕ} {T : Finset GameState} {g₀ : GameState},
      WinningTo T n g₀ →
      Board.WF GameConfig.standard g₀.board →
      ∀ (s : ℕ → Piece), LegalSequenceFrom g₀.bag s →
      ∃ g_end ∈ T,
        Board.WF GameConfig.standard g_end.board ∧
        (∃ k : ℕ, g_end.board.count + 10 * k = g₀.board.count + 4 * n) ∧
        g_end.bag = bagAt g₀.bag s n := by
  intro n
  induction n with
  | zero =>
      intro T g₀ h hwf s _hl
      refine ⟨g₀, h, hwf, ⟨0, by ring⟩, ?_⟩
      rw [bagAt_zero]
  | succ n ih =>
      intro T g₀ h hwf s hl
      rw [winningTo_succ_iff] at h
      obtain ⟨_, hstep⟩ := h
      have hp : s 0 ∈ g₀.bag := by
        have hcd := hl 0
        rw [bagAt_zero, Bag.canDraw_iff_mem] at hcd
        exact hcd
      obtain ⟨pl₀, hpl₀, hnext⟩ := hstep (s 0) hp
      have hpl₀_spec := (Placement.mem_allValidFor _ _ _).mp hpl₀
      have hpl₀_piece : pl₀.piece = s 0 := hpl₀_spec.1
      have hpl₀_valid : pl₀.Valid GameConfig.standard := hpl₀_spec.2
      have heta : ({ pl₀ with piece := s 0 } : Placement) = pl₀ :=
        Placement.eta_of_piece_eq hpl₀_piece
      have hbag_next :
          (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag
            = g₀.bag.draw (s 0) := rfl
      have hboard_next :
          (adversarialStep GameConfig.standard g₀ (s 0) pl₀).board
            = Placement.applyStep GameConfig.standard g₀.board pl₀ := by
        rw [adversarialStep_board, heta]
      have hwf' :
          Board.WF GameConfig.standard
            (adversarialStep GameConfig.standard g₀ (s 0) pl₀).board := by
        rw [hboard_next]
        exact Placement.applyStep_wf hwf hpl₀_valid
      have hsuffix :
          LegalSequenceFrom (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag
            (fun k => s (k + 1)) := by
        have hsuf := hl.suffix 1
        have hbag1_eq : bagAt g₀.bag s 1
            = (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag := by
          rw [bagAt_succ, bagAt_zero, hbag_next]
        rw [hbag1_eq] at hsuf
        convert hsuf using 1
        ext k; ring_nf
      obtain ⟨g_end, hg_end_mem, hwf_end, ⟨k', hcount'⟩, hbag_eq⟩ :=
        ih hnext hwf' _ hsuffix
      have hstep_count :
          (adversarialStep GameConfig.standard g₀ (s 0) pl₀).board.count
            + 10 * Board.linesCleared GameConfig.standard (pl₀.place g₀.board)
            = g₀.board.count + 4 := by
        rw [hboard_next]
        exact applyStep_count GameConfig.standard g₀.board pl₀ hwf hpl₀_valid
      refine ⟨g_end, hg_end_mem, hwf_end, ?_, ?_⟩
      · refine ⟨k' + Board.linesCleared GameConfig.standard (pl₀.place g₀.board), ?_⟩
        have hmul : 10 * (k' + Board.linesCleared GameConfig.standard (pl₀.place g₀.board))
                  = 10 * k' + 10 * Board.linesCleared GameConfig.standard (pl₀.place g₀.board) :=
          Nat.mul_add 10 k' _
        have hexp : 4 * (n + 1) = 4 * n + 4 := by ring
        rw [hmul, hexp]
        omega
      · rw [hbag_eq]
        have h1 : (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag
                = bagAt g₀.bag s 1 := rfl
        rw [h1]
        have h2 := bagAt_add g₀.bag s 1 n
        have h3 : bagAt g₀.bag s (n + 1) = bagAt g₀.bag s (1 + n) := by
          congr 1; ring
        rw [h3, h2]
        congr 1
        ext k; ring_nf

/-! ### Uniform-`Q` residue-step lemma -/

/-- **General step residue progression**: for any `BagWinning P_in P_out`,
applying a single bag of pieces to a WF state `g ∈ P_in` with full bag
and count residue `r` lands at some WF state `g' ∈ P_out` with full bag
and count residue `(r + 8) mod 10`. -/
theorem BagWinning_step_residue
    (P_in P_out : Finset GameState) (h : BagWinning P_in P_out)
    (g : GameState) (hg : g ∈ P_in)
    (hwf : Board.WF GameConfig.standard g.board)
    (hbag : g.bag = Bag.full) :
    ∃ g' ∈ P_out,
      Board.WF GameConfig.standard g'.board ∧
      g'.bag = Bag.full ∧
      g'.board.count % 10 = (g.board.count + 8) % 10 := by
  have hwt : WinningTo P_out piecesPerBag g := h g hg
  obtain ⟨s, hl⟩ : ∃ s : ℕ → Piece, LegalSequenceFrom g.bag s := by
    rw [hbag]
    exact exists_legalSequenceFrom Bag.full Bag.full_nonempty
  obtain ⟨g', hmem, hwf', ⟨k, hcount⟩, hbag_eq⟩ :=
    WinningTo.set_full_bridge hwt hwf s hl
  refine ⟨g', hmem, hwf', ?_, ?_⟩
  · -- g'.bag = bagAt g.bag s 7 = bagAt Bag.full s 7 = Bag.full
    rw [hbag_eq, hbag, piecesPerBag_eq]
    have hl' : LegalSequence s := by
      have := hl
      rw [hbag] at this
      exact this
    exact bagAt_seven_eq_full hl'
  · -- count: g'.count + 10k = g.count + 4·7 = g.count + 28
    rw [piecesPerBag_eq] at hcount
    omega

/-- **Self-step residue progression** (specialization of the general step
to `P_in = P_out = Q`). -/
theorem BagWinning_self_residue_step
    (Q : Finset GameState) (h : BagWinning Q Q)
    (g : GameState) (hg : g ∈ Q)
    (hwf : Board.WF GameConfig.standard g.board)
    (hbag : g.bag = Bag.full) :
    ∃ g' ∈ Q,
      Board.WF GameConfig.standard g'.board ∧
      g'.bag = Bag.full ∧
      g'.board.count % 10 = (g.board.count + 8) % 10 :=
  BagWinning_step_residue Q Q h g hg hwf hbag

/-- **Initial step into uniform Q from init**: from `BagWinning {init} Q`,
derive existence of `g₁ ∈ Q` with WF, full bag, count residue 8. -/
theorem BagWinning_init_implies_Q_has_initial
    (Q : Finset GameState)
    (h : BagWinning {GameState.init} Q) :
    ∃ g ∈ Q,
      Board.WF GameConfig.standard g.board ∧
      g.bag = Bag.full ∧
      g.board.count % 10 = 8 := by
  have hwt : WinningTo Q piecesPerBag GameState.init :=
    h GameState.init (Finset.mem_singleton.mpr rfl)
  obtain ⟨s, hl⟩ := exists_legalSequence
  have hl' : LegalSequenceFrom GameState.init.bag s := by
    rw [GameState.init_bag]; exact hl
  have hwf : Board.WF GameConfig.standard GameState.init.board := by
    rw [GameState.init_board]; exact Board.empty_wf _
  obtain ⟨g, hmem, hwf_end, ⟨k, hcount⟩, hbag_eq⟩ :=
    WinningTo.set_full_bridge hwt hwf s hl'
  refine ⟨g, hmem, hwf_end, ?_, ?_⟩
  · rw [hbag_eq, GameState.init_bag, piecesPerBag_eq]
    exact bagAt_seven_eq_full hl
  · rw [GameState.init_board_count, piecesPerBag_eq] at hcount
    omega

/-- **Four-residue obligation**: in a uniform-phase decomposition,
`Q` must contain states with all four residues `{8, 6, 4, 2}` (mod 10),
hence **at least 4 distinct elements**. -/
theorem uniform_Q_requires_four_residues
    (Q : Finset GameState)
    (h1 : BagWinning {GameState.init} Q)
    (h2 : BagWinning Q Q) :
    ∃ g8 g6 g4 g2 : GameState,
      g8 ∈ Q ∧ g8.board.count % 10 = 8 ∧
      g6 ∈ Q ∧ g6.board.count % 10 = 6 ∧
      g4 ∈ Q ∧ g4.board.count % 10 = 4 ∧
      g2 ∈ Q ∧ g2.board.count % 10 = 2 := by
  -- Step 1: from init, land at g8 ∈ Q with residue 8
  obtain ⟨g8, hg8_mem, hg8_wf, hg8_bag, hg8_res⟩ :=
    BagWinning_init_implies_Q_has_initial Q h1
  -- Step 2: from g8 (residue 8), land at g6 ∈ Q with residue 6
  obtain ⟨g6, hg6_mem, hg6_wf, hg6_bag, hg6_res⟩ :=
    BagWinning_self_residue_step Q h2 g8 hg8_mem hg8_wf hg8_bag
  have hg6_res' : g6.board.count % 10 = 6 := by omega
  -- Step 3: from g6, land at g4 ∈ Q with residue 4
  obtain ⟨g4, hg4_mem, hg4_wf, hg4_bag, hg4_res⟩ :=
    BagWinning_self_residue_step Q h2 g6 hg6_mem hg6_wf hg6_bag
  have hg4_res' : g4.board.count % 10 = 4 := by omega
  -- Step 4: from g4, land at g2 ∈ Q with residue 2
  obtain ⟨g2, hg2_mem, hg2_wf, hg2_bag, hg2_res⟩ :=
    BagWinning_self_residue_step Q h2 g4 hg4_mem hg4_wf hg4_bag
  have hg2_res' : g2.board.count % 10 = 2 := by omega
  exact ⟨g8, g6, g4, g2, hg8_mem, hg8_res, hg6_mem, hg6_res',
         hg4_mem, hg4_res', hg2_mem, hg2_res'⟩

/-- **Five-residue obligation**: in a uniform-phase decomposition,
`Q` must contain states with all five residues `{8, 6, 4, 2, 0}` (mod 10),
hence **at least 5 distinct elements**.

This strengthens `uniform_Q_requires_four_residues`: the fourth self-step
(`g2` at residue 2 → residue `(2+8)%10 = 0`) forces a residue-0 state into
`Q` as well. Equivalently, the `+8 mod 10` self-progression cycles through
the full coset `8 → 6 → 4 → 2 → 0 → 8`, so a closed uniform `Q` must witness
every one of the five residues, not merely the first four. -/
theorem uniform_Q_requires_five_residues
    (Q : Finset GameState)
    (h1 : BagWinning {GameState.init} Q)
    (h2 : BagWinning Q Q) :
    ∃ g8 g6 g4 g2 g0 : GameState,
      g8 ∈ Q ∧ g8.board.count % 10 = 8 ∧
      g6 ∈ Q ∧ g6.board.count % 10 = 6 ∧
      g4 ∈ Q ∧ g4.board.count % 10 = 4 ∧
      g2 ∈ Q ∧ g2.board.count % 10 = 2 ∧
      g0 ∈ Q ∧ g0.board.count % 10 = 0 := by
  -- Step 1: from init, land at g8 ∈ Q with residue 8
  obtain ⟨g8, hg8_mem, hg8_wf, hg8_bag, hg8_res⟩ :=
    BagWinning_init_implies_Q_has_initial Q h1
  -- Step 2: from g8 (residue 8), land at g6 ∈ Q with residue 6
  obtain ⟨g6, hg6_mem, hg6_wf, hg6_bag, hg6_res⟩ :=
    BagWinning_self_residue_step Q h2 g8 hg8_mem hg8_wf hg8_bag
  have hg6_res' : g6.board.count % 10 = 6 := by omega
  -- Step 3: from g6, land at g4 ∈ Q with residue 4
  obtain ⟨g4, hg4_mem, hg4_wf, hg4_bag, hg4_res⟩ :=
    BagWinning_self_residue_step Q h2 g6 hg6_mem hg6_wf hg6_bag
  have hg4_res' : g4.board.count % 10 = 4 := by omega
  -- Step 4: from g4, land at g2 ∈ Q with residue 2
  obtain ⟨g2, hg2_mem, hg2_wf, hg2_bag, hg2_res⟩ :=
    BagWinning_self_residue_step Q h2 g4 hg4_mem hg4_wf hg4_bag
  have hg2_res' : g2.board.count % 10 = 2 := by omega
  -- Step 5: from g2 (residue 2), land at g0 ∈ Q with residue 0
  obtain ⟨g0, hg0_mem, hg0_wf, hg0_bag, hg0_res⟩ :=
    BagWinning_self_residue_step Q h2 g2 hg2_mem hg2_wf hg2_bag
  have hg0_res' : g0.board.count % 10 = 0 := by omega
  exact ⟨g8, g6, g4, g2, g0, hg8_mem, hg8_res, hg6_mem, hg6_res',
         hg4_mem, hg4_res', hg2_mem, hg2_res', hg0_mem, hg0_res'⟩

/-- **|Q| ≥ 4 for any uniform-phase decomposition**: the four states from
`uniform_Q_requires_four_residues` are distinct (since their count
residues differ), so `Q.card ≥ 4`.

This **rules out the simplest possible decomposition shape**: a uniform
phase set `Q` cannot be a singleton, a pair, or even a triple. -/
theorem uniform_Q_card_ge_four
    (Q : Finset GameState)
    (h1 : BagWinning {GameState.init} Q)
    (h2 : BagWinning Q Q) :
    4 ≤ Q.card := by
  obtain ⟨g8, g6, g4, g2,
          hg8_mem, hg8_res, hg6_mem, hg6_res, hg4_mem, hg4_res, hg2_mem, hg2_res⟩ :=
    uniform_Q_requires_four_residues Q h1 h2
  -- Construct the 4-element subset {g8, g6, g4, g2} ⊆ Q
  have hne_86 : g8 ≠ g6 := by
    intro he; rw [he] at hg8_res; rw [hg6_res] at hg8_res; omega
  have hne_84 : g8 ≠ g4 := by
    intro he; rw [he] at hg8_res; rw [hg4_res] at hg8_res; omega
  have hne_82 : g8 ≠ g2 := by
    intro he; rw [he] at hg8_res; rw [hg2_res] at hg8_res; omega
  have hne_64 : g6 ≠ g4 := by
    intro he; rw [he] at hg6_res; rw [hg4_res] at hg6_res; omega
  have hne_62 : g6 ≠ g2 := by
    intro he; rw [he] at hg6_res; rw [hg2_res] at hg6_res; omega
  have hne_42 : g4 ≠ g2 := by
    intro he; rw [he] at hg4_res; rw [hg2_res] at hg4_res; omega
  -- Build the subset as a Finset
  have hsub : ({g8, g6, g4, g2} : Finset GameState) ⊆ Q := by
    intro x hx
    simp only [Finset.mem_insert, Finset.mem_singleton] at hx
    rcases hx with rfl | rfl | rfl | rfl
    · exact hg8_mem
    · exact hg6_mem
    · exact hg4_mem
    · exact hg2_mem
  have hcard4 : ({g8, g6, g4, g2} : Finset GameState).card = 4 := by
    show (insert g8 (insert g6 (insert g4 ({g2} : Finset GameState)))).card = 4
    rw [Finset.card_insert_of_notMem, Finset.card_insert_of_notMem,
        Finset.card_insert_of_notMem, Finset.card_singleton]
    · simp [hne_42]
    · simp [hne_64, hne_62]
    · simp [hne_86, hne_84, hne_82]
  calc 4 = ({g8, g6, g4, g2} : Finset GameState).card := hcard4.symm
    _ ≤ Q.card := Finset.card_le_card hsub

/-- **|Q| ≥ 5 for any uniform-phase decomposition**: the five states from
`uniform_Q_requires_five_residues` are pairwise distinct (their count
residues `{8, 6, 4, 2, 0}` differ mod 10), so `Q.card ≥ 5`.

This strengthens `uniform_Q_card_ge_four`: a uniform phase set `Q` cannot
have only four elements either — the `+8 mod 10` self-loop forces a fifth,
residue-`0` state. So the **smallest conceivable uniform `Q` has ≥ 5
states**, and in particular `quadruple`-shaped uniform decompositions are
ruled out alongside singleton/pair/triple. -/
theorem uniform_Q_card_ge_five
    (Q : Finset GameState)
    (h1 : BagWinning {GameState.init} Q)
    (h2 : BagWinning Q Q) :
    5 ≤ Q.card := by
  obtain ⟨g8, g6, g4, g2, g0,
          hg8_mem, hg8_res, hg6_mem, hg6_res, hg4_mem, hg4_res,
          hg2_mem, hg2_res, hg0_mem, hg0_res⟩ :=
    uniform_Q_requires_five_residues Q h1 h2
  -- 10 pairwise distinctness facts from the differing residues
  have hne_86 : g8 ≠ g6 := by
    intro he; rw [he] at hg8_res; rw [hg6_res] at hg8_res; omega
  have hne_84 : g8 ≠ g4 := by
    intro he; rw [he] at hg8_res; rw [hg4_res] at hg8_res; omega
  have hne_82 : g8 ≠ g2 := by
    intro he; rw [he] at hg8_res; rw [hg2_res] at hg8_res; omega
  have hne_80 : g8 ≠ g0 := by
    intro he; rw [he] at hg8_res; rw [hg0_res] at hg8_res; omega
  have hne_64 : g6 ≠ g4 := by
    intro he; rw [he] at hg6_res; rw [hg4_res] at hg6_res; omega
  have hne_62 : g6 ≠ g2 := by
    intro he; rw [he] at hg6_res; rw [hg2_res] at hg6_res; omega
  have hne_60 : g6 ≠ g0 := by
    intro he; rw [he] at hg6_res; rw [hg0_res] at hg6_res; omega
  have hne_42 : g4 ≠ g2 := by
    intro he; rw [he] at hg4_res; rw [hg2_res] at hg4_res; omega
  have hne_40 : g4 ≠ g0 := by
    intro he; rw [he] at hg4_res; rw [hg0_res] at hg4_res; omega
  have hne_20 : g2 ≠ g0 := by
    intro he; rw [he] at hg2_res; rw [hg0_res] at hg2_res; omega
  -- Build the 5-element subset {g8, g6, g4, g2, g0} ⊆ Q
  have hsub : ({g8, g6, g4, g2, g0} : Finset GameState) ⊆ Q := by
    intro x hx
    simp only [Finset.mem_insert, Finset.mem_singleton] at hx
    rcases hx with rfl | rfl | rfl | rfl | rfl
    · exact hg8_mem
    · exact hg6_mem
    · exact hg4_mem
    · exact hg2_mem
    · exact hg0_mem
  have hcard5 : ({g8, g6, g4, g2, g0} : Finset GameState).card = 5 := by
    show (insert g8 (insert g6 (insert g4
        (insert g2 ({g0} : Finset GameState))))).card = 5
    rw [Finset.card_insert_of_notMem, Finset.card_insert_of_notMem,
        Finset.card_insert_of_notMem, Finset.card_insert_of_notMem,
        Finset.card_singleton]
    · simp [hne_20]
    · simp [hne_42, hne_40]
    · simp [hne_64, hne_62, hne_60]
    · simp [hne_86, hne_84, hne_82, hne_80]
  calc 5 = ({g8, g6, g4, g2, g0} : Finset GameState).card := hcard5.symm
    _ ≤ Q.card := Finset.card_le_card hsub

/-- **Self-recurrent sets are large.** Any `T` closed under one bag of
adversarial play (`BagWinning T T`) that contains even a *single*
well-formed, full-bag state `g₀` must contain **at least five** states.
Iterating the `+8 mod 10` self-step four times from `g₀` produces a chain
`g₀ → g₁ → g₂ → g₃ → g₄`, all in `T`, with cumulative count residues
`g₀ + {0, 8, 6, 4, 2} (mod 10)` — five distinct values — so the five states
are pairwise distinct.

This is the self-loop analogue of `uniform_Q_card_ge_five`, but needs only
the loop hypothesis (no open/close phases). It therefore bounds *every*
bag-closed family — including the self-recurrent certificates `T` that
`tetrisSolvableValid_of_selfBagWinningBool` consumes: such a `T`, once it
contains any full-bag reachable state, can never be a singleton/pair/triple/
quadruple. -/
theorem BagWinning_self_card_ge_five
    (T : Finset GameState)
    (hself : BagWinning T T)
    {g₀ : GameState} (hg₀ : g₀ ∈ T)
    (hwf : Board.WF GameConfig.standard g₀.board)
    (hbag : g₀.bag = Bag.full) :
    5 ≤ T.card := by
  -- four self-steps: g₀ → g1 → g2 → g3 → g4, all in T, full bag preserved
  obtain ⟨g1, hg1_mem, hg1_wf, hg1_bag, hg1_res⟩ :=
    BagWinning_self_residue_step T hself g₀ hg₀ hwf hbag
  obtain ⟨g2, hg2_mem, hg2_wf, hg2_bag, hg2_res⟩ :=
    BagWinning_self_residue_step T hself g1 hg1_mem hg1_wf hg1_bag
  obtain ⟨g3, hg3_mem, hg3_wf, hg3_bag, hg3_res⟩ :=
    BagWinning_self_residue_step T hself g2 hg2_mem hg2_wf hg2_bag
  obtain ⟨g4, hg4_mem, hg4_wf, hg4_bag, hg4_res⟩ :=
    BagWinning_self_residue_step T hself g3 hg3_mem hg3_wf hg3_bag
  -- pairwise distinctness from the distinct cumulative residue offsets
  have hne_01 : g₀ ≠ g1 := by
    intro he
    have hc : g₀.board.count = g1.board.count := by rw [he]
    omega
  have hne_02 : g₀ ≠ g2 := by
    intro he
    have hc : g₀.board.count = g2.board.count := by rw [he]
    omega
  have hne_03 : g₀ ≠ g3 := by
    intro he
    have hc : g₀.board.count = g3.board.count := by rw [he]
    omega
  have hne_04 : g₀ ≠ g4 := by
    intro he
    have hc : g₀.board.count = g4.board.count := by rw [he]
    omega
  have hne_12 : g1 ≠ g2 := by
    intro he
    have hc : g1.board.count = g2.board.count := by rw [he]
    omega
  have hne_13 : g1 ≠ g3 := by
    intro he
    have hc : g1.board.count = g3.board.count := by rw [he]
    omega
  have hne_14 : g1 ≠ g4 := by
    intro he
    have hc : g1.board.count = g4.board.count := by rw [he]
    omega
  have hne_23 : g2 ≠ g3 := by
    intro he
    have hc : g2.board.count = g3.board.count := by rw [he]
    omega
  have hne_24 : g2 ≠ g4 := by
    intro he
    have hc : g2.board.count = g4.board.count := by rw [he]
    omega
  have hne_34 : g3 ≠ g4 := by
    intro he
    have hc : g3.board.count = g4.board.count := by rw [he]
    omega
  -- the five states form a 5-element subset of T
  have hsub : ({g₀, g1, g2, g3, g4} : Finset GameState) ⊆ T := by
    intro x hx
    simp only [Finset.mem_insert, Finset.mem_singleton] at hx
    rcases hx with rfl | rfl | rfl | rfl | rfl
    · exact hg₀
    · exact hg1_mem
    · exact hg2_mem
    · exact hg3_mem
    · exact hg4_mem
  have hcard5 : ({g₀, g1, g2, g3, g4} : Finset GameState).card = 5 := by
    show (insert g₀ (insert g1 (insert g2
        (insert g3 ({g4} : Finset GameState))))).card = 5
    rw [Finset.card_insert_of_notMem, Finset.card_insert_of_notMem,
        Finset.card_insert_of_notMem, Finset.card_insert_of_notMem,
        Finset.card_singleton]
    · simp [hne_34]
    · simp [hne_23, hne_24]
    · simp [hne_12, hne_13, hne_14]
    · simp [hne_01, hne_02, hne_03, hne_04]
  calc 5 = ({g₀, g1, g2, g3, g4} : Finset GameState).card := hcard5.symm
    _ ≤ T.card := Finset.card_le_card hsub

/-- **Reachable self-recurrent certificates have ≥5 states — unconditionally.**
If `init` reaches `T` in exactly `m` full bags (`WinningTo T (7*m) init`) and
`T` is bag-closed (`BagWinning T T`), then `5 ≤ T.card`.

This removes the manual full-bag hypothesis of `BagWinning_self_card_ge_five`:
the bag-boundary endpoint of a `7*m`-step win from `init` automatically has a
**full bag** (`bagAt Bag.full s (7*m) = Bag.full`) and a well-formed board, so
it seeds the residue cycle. This is the form that bites on the *live* route —
the self-recurrent `T` that `tetrisSolvableValid_of_selfBagWinningBool`
consumes (reached at a bag boundary) can never be smaller than five states. -/
theorem reachable_selfRecurrent_card_ge_five
    (T : Finset GameState) {m : ℕ}
    (hreach : WinningTo T (7 * m) GameState.init)
    (hself : BagWinning T T) :
    5 ≤ T.card := by
  obtain ⟨s, hl⟩ := exists_legalSequence
  have hl' : LegalSequenceFrom GameState.init.bag s := by
    rw [GameState.init_bag]; exact hl
  have hwf : Board.WF GameConfig.standard GameState.init.board := by
    rw [GameState.init_board]; exact Board.empty_wf _
  obtain ⟨g, hmem, hwf_end, _, hbag_eq⟩ :=
    WinningTo.set_full_bridge hreach hwf s hl'
  have hbag : g.bag = Bag.full := by
    rw [hbag_eq, GameState.init_bag]; exact bagAt_mul_seven_eq_full hl m
  exact BagWinning_self_card_ge_five T hself hmem hwf_end hbag

/-- **Self-loop residue period 5** — the "five-bag reset" at the residue
level. From a full-bag, well-formed `g₀ ∈ T` with `BagWinning T T`, five
self-steps reach a state `g₅ ∈ T` whose count residue *equals* `g₀`'s. The
`+8 mod 10` step has order 5 (`5 · 8 = 40 ≡ 0 mod 10`), so the residue cycle
`r → r+8 → r+6 → r+4 → r+2 → r` closes after exactly five bags — mirroring
`bagsPerBlock = 5` at the residue level, now for an *arbitrary* bag-closed
`T` rather than a return to `init`. -/
theorem BagWinning_self_residue_returns
    (T : Finset GameState)
    (hself : BagWinning T T)
    {g₀ : GameState} (hg₀ : g₀ ∈ T)
    (hwf : Board.WF GameConfig.standard g₀.board)
    (hbag : g₀.bag = Bag.full) :
    ∃ g₅ ∈ T, Board.WF GameConfig.standard g₅.board ∧ g₅.bag = Bag.full ∧
      g₅.board.count % 10 = g₀.board.count % 10 := by
  obtain ⟨g1, hg1_mem, hg1_wf, hg1_bag, hg1_res⟩ :=
    BagWinning_self_residue_step T hself g₀ hg₀ hwf hbag
  obtain ⟨g2, hg2_mem, hg2_wf, hg2_bag, hg2_res⟩ :=
    BagWinning_self_residue_step T hself g1 hg1_mem hg1_wf hg1_bag
  obtain ⟨g3, hg3_mem, hg3_wf, hg3_bag, hg3_res⟩ :=
    BagWinning_self_residue_step T hself g2 hg2_mem hg2_wf hg2_bag
  obtain ⟨g4, hg4_mem, hg4_wf, hg4_bag, hg4_res⟩ :=
    BagWinning_self_residue_step T hself g3 hg3_mem hg3_wf hg3_bag
  obtain ⟨g5, hg5_mem, hg5_wf, hg5_bag, hg5_res⟩ :=
    BagWinning_self_residue_step T hself g4 hg4_mem hg4_wf hg4_bag
  exact ⟨g5, hg5_mem, hg5_wf, hg5_bag, by omega⟩

/-- **Self-recurrent residue cycle**: from any full-bag WF anchor `g₀ ∈ T` in a
self-recurrent table, the four forced successor states tile the count-residue
offsets `{+8, +6, +4, +2}` (mod 10) relative to `g₀`. Together with `g₀` itself
(offset `+0`) these cover the entire even-residue parity class, the structural
companion to `BagWinning_self_card_ge_five`. -/
theorem BagWinning_self_residue_cycle
    (T : Finset GameState)
    (hself : BagWinning T T)
    {g₀ : GameState} (hg₀ : g₀ ∈ T)
    (hwf : Board.WF GameConfig.standard g₀.board)
    (hbag : g₀.bag = Bag.full) :
    ∃ g1 ∈ T, ∃ g2 ∈ T, ∃ g3 ∈ T, ∃ g4 ∈ T,
      g1.board.count % 10 = (g₀.board.count + 8) % 10 ∧
      g2.board.count % 10 = (g₀.board.count + 6) % 10 ∧
      g3.board.count % 10 = (g₀.board.count + 4) % 10 ∧
      g4.board.count % 10 = (g₀.board.count + 2) % 10 := by
  obtain ⟨g1, hg1_mem, hg1_wf, hg1_bag, hg1_res⟩ :=
    BagWinning_self_residue_step T hself g₀ hg₀ hwf hbag
  obtain ⟨g2, hg2_mem, hg2_wf, hg2_bag, hg2_res⟩ :=
    BagWinning_self_residue_step T hself g1 hg1_mem hg1_wf hg1_bag
  obtain ⟨g3, hg3_mem, hg3_wf, hg3_bag, hg3_res⟩ :=
    BagWinning_self_residue_step T hself g2 hg2_mem hg2_wf hg2_bag
  obtain ⟨g4, hg4_mem, hg4_wf, hg4_bag, hg4_res⟩ :=
    BagWinning_self_residue_step T hself g3 hg3_mem hg3_wf hg3_bag
  exact ⟨g1, hg1_mem, g2, hg2_mem, g3, hg3_mem, g4, hg4_mem,
    by omega, by omega, by omega, by omega⟩

/-- **Reachable self-recurrent table covers the even-residue class
unconditionally**: any `T` forced-reached from `init` at a bag boundary `7*m`
and closed under one bag (`BagWinning T T`) contains a full-bag WF anchor `g0`
together with four successors realizing residue offsets `{+8,+6,+4,+2}` (mod 10).
This is the unconditional capstone of the residue-structure subprogram: it
fuses the init-reachability bridge (`WinningTo.set_full_bridge`) with the
self-recurrent residue cycle (`BagWinning_self_residue_cycle`), needing no
externally-supplied anchor. -/
theorem reachable_selfRecurrent_covers_even_residues
    (T : Finset GameState) {m : ℕ}
    (hreach : WinningTo T (7 * m) GameState.init)
    (hself : BagWinning T T) :
    ∃ g0 ∈ T, ∃ g1 ∈ T, ∃ g2 ∈ T, ∃ g3 ∈ T, ∃ g4 ∈ T,
      g1.board.count % 10 = (g0.board.count + 8) % 10 ∧
      g2.board.count % 10 = (g0.board.count + 6) % 10 ∧
      g3.board.count % 10 = (g0.board.count + 4) % 10 ∧
      g4.board.count % 10 = (g0.board.count + 2) % 10 := by
  obtain ⟨s, hl⟩ := exists_legalSequence
  have hl' : LegalSequenceFrom GameState.init.bag s := by
    rw [GameState.init_bag]; exact hl
  have hwf : Board.WF GameConfig.standard GameState.init.board := by
    rw [GameState.init_board]; exact Board.empty_wf _
  obtain ⟨g, hmem, hwf_end, _, hbag_eq⟩ := WinningTo.set_full_bridge hreach hwf s hl'
  have hbag : g.bag = Bag.full := by
    rw [hbag_eq, GameState.init_bag]; exact bagAt_mul_seven_eq_full hl m
  obtain ⟨g1, hg1, g2, hg2, g3, hg3, g4, hg4, h1, h2, h3, h4⟩ :=
    BagWinning_self_residue_cycle T hself hmem hwf_end hbag
  exact ⟨g, hmem, g1, hg1, g2, hg2, g3, hg3, g4, hg4, h1, h2, h3, h4⟩

/-! ### Non-uniform residue chain

For the general (non-uniform) decomposition `init → P_1 → P_2 → P_3 → P_4 → init`,
each `P_i` must contain a state with the correct count residue. -/

/-- **Non-uniform residue chain**: every candidate non-uniform phase
decomposition forces each `P_i` to contain a WF, full-bag state with
the specified count residue. -/
theorem phase_decomposition_residue_chain
    (P_1 P_2 P_3 P_4 : Finset GameState)
    (h12 : BagWinning {GameState.init} P_1)
    (h23 : BagWinning P_1 P_2)
    (h34 : BagWinning P_2 P_3)
    (h45 : BagWinning P_3 P_4) :
    (∃ g ∈ P_1, Board.WF GameConfig.standard g.board ∧
                g.bag = Bag.full ∧ g.board.count % 10 = 8) ∧
    (∃ g ∈ P_2, Board.WF GameConfig.standard g.board ∧
                g.bag = Bag.full ∧ g.board.count % 10 = 6) ∧
    (∃ g ∈ P_3, Board.WF GameConfig.standard g.board ∧
                g.bag = Bag.full ∧ g.board.count % 10 = 4) ∧
    (∃ g ∈ P_4, Board.WF GameConfig.standard g.board ∧
                g.bag = Bag.full ∧ g.board.count % 10 = 2) := by
  -- Step 1: from init, land at g1 ∈ P_1 with residue 8
  obtain ⟨g1, hg1_mem, hg1_wf, hg1_bag, hg1_res⟩ :=
    BagWinning_init_implies_Q_has_initial P_1 h12
  -- Step 2: from g1, land at g2 ∈ P_2 with residue 6
  obtain ⟨g2, hg2_mem, hg2_wf, hg2_bag, hg2_res⟩ :=
    BagWinning_step_residue P_1 P_2 h23 g1 hg1_mem hg1_wf hg1_bag
  have hg2_res' : g2.board.count % 10 = 6 := by omega
  -- Step 3: from g2, land at g3 ∈ P_3 with residue 4
  obtain ⟨g3, hg3_mem, hg3_wf, hg3_bag, hg3_res⟩ :=
    BagWinning_step_residue P_2 P_3 h34 g2 hg2_mem hg2_wf hg2_bag
  have hg3_res' : g3.board.count % 10 = 4 := by omega
  -- Step 4: from g3, land at g4 ∈ P_4 with residue 2
  obtain ⟨g4, hg4_mem, hg4_wf, hg4_bag, hg4_res⟩ :=
    BagWinning_step_residue P_3 P_4 h45 g3 hg3_mem hg3_wf hg3_bag
  have hg4_res' : g4.board.count % 10 = 2 := by omega
  exact ⟨⟨g1, hg1_mem, hg1_wf, hg1_bag, hg1_res⟩,
         ⟨g2, hg2_mem, hg2_wf, hg2_bag, hg2_res'⟩,
         ⟨g3, hg3_mem, hg3_wf, hg3_bag, hg3_res'⟩,
         ⟨g4, hg4_mem, hg4_wf, hg4_bag, hg4_res'⟩⟩

/-! ### Checkpoint count menus

The bridge gives more than residues: the count at the `i`-th checkpoint
is *at most* `28·i` (placements deliver `4·7·i` cells and clears only
subtract). Combined with the residue, each checkpoint has a small finite
menu of possible counts. -/

/-- **General checkpoint bound**: if `init` wins into `T` in exactly
`7·bags` pieces, then `T` contains a WF, full-bag state with count
≡ `28·bags (mod 10)` and count ≤ `28·bags`. -/
theorem WinningTo.checkpoint_count_bound
    (T : Finset GameState) (bags : ℕ)
    (h : WinningTo T (7 * bags) GameState.init) :
    ∃ g ∈ T, Board.WF GameConfig.standard g.board ∧
      g.bag = Bag.full ∧
      g.board.count % 10 = (28 * bags) % 10 ∧
      g.board.count ≤ 28 * bags := by
  obtain ⟨s, hl⟩ := exists_legalSequence
  have hl' : LegalSequenceFrom GameState.init.bag s := by
    rw [GameState.init_bag]; exact hl
  have hwf : Board.WF GameConfig.standard GameState.init.board := by
    rw [GameState.init_board]; exact Board.empty_wf _
  obtain ⟨g, hmem, hwf_end, ⟨k, hcount⟩, hbag_eq⟩ :=
    WinningTo.set_full_bridge h hwf s hl'
  refine ⟨g, hmem, hwf_end, ?_, ?_, ?_⟩
  · rw [hbag_eq, GameState.init_bag]
    exact bagAt_mul_seven_eq_full hl bags
  · rw [GameState.init_board_count] at hcount
    omega
  · rw [GameState.init_board_count] at hcount
    omega

/-- **`P_1` count menu**: after the first bag, the board count is exactly
8, 18, or 28 — three possible values only (28 cells delivered, 0/1/2
lines cleared). -/
theorem P1_count_menu
    (P_1 : Finset GameState)
    (h : BagWinning {GameState.init} P_1) :
    ∃ g ∈ P_1, Board.WF GameConfig.standard g.board ∧
      g.bag = Bag.full ∧
      (g.board.count = 8 ∨ g.board.count = 18 ∨ g.board.count = 28) := by
  have hwt : WinningTo P_1 (7 * 1) GameState.init := by
    have := h GameState.init (Finset.mem_singleton.mpr rfl)
    rwa [show (7 : ℕ) * 1 = piecesPerBag from rfl]
  obtain ⟨g, hmem, hwf, hbag, hres, hle⟩ :=
    WinningTo.checkpoint_count_bound P_1 1 hwt
  exact ⟨g, hmem, hwf, hbag, by omega⟩

/-- **`P_2` count bound**: after two bags, count ≡ 6 (mod 10) and ≤ 56
(menu {6, 16, 26, 36, 46, 56}). -/
theorem P2_count_menu
    (P_1 P_2 : Finset GameState)
    (h12 : BagWinning {GameState.init} P_1)
    (h23 : BagWinning P_1 P_2) :
    ∃ g ∈ P_2, Board.WF GameConfig.standard g.board ∧
      g.bag = Bag.full ∧
      g.board.count % 10 = 6 ∧ g.board.count ≤ 56 := by
  have hwt1 : WinningTo P_1 piecesPerBag GameState.init :=
    h12 GameState.init (Finset.mem_singleton.mpr rfl)
  have hwt2 : WinningTo P_2 (7 * 2) GameState.init := by
    have := hwt1.compose h23
    rwa [show piecesPerBag + piecesPerBag = 7 * 2 from rfl] at this
  obtain ⟨g, hmem, hwf, hbag, hres, hle⟩ :=
    WinningTo.checkpoint_count_bound P_2 2 hwt2
  exact ⟨g, hmem, hwf, hbag, by omega, by omega⟩

/-- **`P_3` count bound**: after three bags, count ≡ 4 (mod 10) and ≤ 84. -/
theorem P3_count_menu
    (P_1 P_2 P_3 : Finset GameState)
    (h12 : BagWinning {GameState.init} P_1)
    (h23 : BagWinning P_1 P_2)
    (h34 : BagWinning P_2 P_3) :
    ∃ g ∈ P_3, Board.WF GameConfig.standard g.board ∧
      g.bag = Bag.full ∧
      g.board.count % 10 = 4 ∧ g.board.count ≤ 84 := by
  have hwt1 : WinningTo P_1 piecesPerBag GameState.init :=
    h12 GameState.init (Finset.mem_singleton.mpr rfl)
  have hwt2 : WinningTo P_2 (piecesPerBag + piecesPerBag) GameState.init :=
    hwt1.compose h23
  have hwt3 : WinningTo P_3 (7 * 3) GameState.init := by
    have := hwt2.compose h34
    rwa [show piecesPerBag + piecesPerBag + piecesPerBag = 7 * 3 from rfl] at this
  obtain ⟨g, hmem, hwf, hbag, hres, hle⟩ :=
    WinningTo.checkpoint_count_bound P_3 3 hwt3
  exact ⟨g, hmem, hwf, hbag, by omega, by omega⟩

/-- **`P_4` count bound**: after four bags, count ≡ 2 (mod 10) and ≤ 112. -/
theorem P4_count_menu
    (P_1 P_2 P_3 P_4 : Finset GameState)
    (h12 : BagWinning {GameState.init} P_1)
    (h23 : BagWinning P_1 P_2)
    (h34 : BagWinning P_2 P_3)
    (h45 : BagWinning P_3 P_4) :
    ∃ g ∈ P_4, Board.WF GameConfig.standard g.board ∧
      g.bag = Bag.full ∧
      g.board.count % 10 = 2 ∧ g.board.count ≤ 112 := by
  have hwt1 : WinningTo P_1 piecesPerBag GameState.init :=
    h12 GameState.init (Finset.mem_singleton.mpr rfl)
  have hwt2 : WinningTo P_2 (piecesPerBag + piecesPerBag) GameState.init :=
    hwt1.compose h23
  have hwt3 : WinningTo P_3 (piecesPerBag + piecesPerBag + piecesPerBag)
      GameState.init :=
    hwt2.compose h34
  have hwt4 : WinningTo P_4 (7 * 4) GameState.init := by
    have := hwt3.compose h45
    rwa [show piecesPerBag + piecesPerBag + piecesPerBag + piecesPerBag
          = 7 * 4 from rfl] at this
  obtain ⟨g, hmem, hwf, hbag, hres, hle⟩ :=
    WinningTo.checkpoint_count_bound P_4 4 hwt4
  exact ⟨g, hmem, hwf, hbag, by omega, by omega⟩

/-! ### Concrete impossibility corollaries

Direct applications of the residue constraints to rule out trivial
candidate decompositions. -/

/-- **No 1-bag reset**: cannot win and return to `init` after just 7 pieces. -/
theorem not_winning_one_bag : ¬ Winning piecesPerBag GameState.init :=
  fun h => not_winning_lt_35_init (by decide) (by decide) h

/-- **No 2-bag reset**: cannot win after 14 pieces. -/
theorem not_winning_two_bags : ¬ Winning (2 * piecesPerBag) GameState.init :=
  fun h => not_winning_lt_35_init (by decide) (by decide) h

/-- **No 3-bag reset**: cannot win after 21 pieces. -/
theorem not_winning_three_bags : ¬ Winning (3 * piecesPerBag) GameState.init :=
  fun h => not_winning_lt_35_init (by decide) (by decide) h

/-- **No 4-bag reset**: cannot win after 28 pieces. -/
theorem not_winning_four_bags : ¬ Winning (4 * piecesPerBag) GameState.init :=
  fun h => not_winning_lt_35_init (by decide) (by decide) h

/-- **The trivial uniform-phase decomposition with `Q = {init}` is
impossible**: `init` itself cannot serve as the uniform phase set,
because its count residue 0 ≠ 8. -/
theorem trivial_uniform_Q_impossible :
    ¬ BagWinning {GameState.init} {GameState.init} := by
  intro h
  obtain ⟨g, hg_mem, _hwf, _hbag, hres⟩ :=
    BagWinning_init_implies_Q_has_initial {GameState.init} h
  rw [Finset.mem_singleton] at hg_mem
  rw [hg_mem, GameState.init_board_count] at hres
  norm_num at hres

/-- **Singleton candidates with any single residue class are impossible
for uniform Q**: a singleton `Q = {g}` cannot serve as uniform phase
because it can hold at most one count residue, but four are required. -/
theorem singleton_uniform_Q_impossible (g : GameState) :
    ¬ (BagWinning {GameState.init} {g} ∧ BagWinning {g} {g}) := by
  rintro ⟨h1, h2⟩
  have hcard : 4 ≤ ({g} : Finset GameState).card :=
    uniform_Q_card_ge_four {g} h1 h2
  simp at hcard

/-- **Pair candidates are impossible for uniform Q**: `|Q| = 2` cannot
hold all four required residue classes. -/
theorem pair_uniform_Q_impossible (g₁ g₂ : GameState) :
    ¬ (BagWinning {GameState.init} {g₁, g₂} ∧ BagWinning {g₁, g₂} {g₁, g₂}) := by
  rintro ⟨h1, h2⟩
  have hcard : 4 ≤ ({g₁, g₂} : Finset GameState).card :=
    uniform_Q_card_ge_four {g₁, g₂} h1 h2
  have : ({g₁, g₂} : Finset GameState).card ≤ 2 := by
    rw [show ({g₁, g₂} : Finset GameState) = insert g₁ {g₂} from rfl]
    calc (insert g₁ ({g₂} : Finset GameState)).card
        ≤ ({g₂} : Finset GameState).card + 1 := Finset.card_insert_le _ _
      _ = 2 := by simp
  omega

/-- **Triple candidates are impossible for uniform Q**: `|Q| = 3` cannot
hold all four required residue classes. -/
theorem triple_uniform_Q_impossible (g₁ g₂ g₃ : GameState) :
    ¬ (BagWinning {GameState.init} {g₁, g₂, g₃} ∧
       BagWinning {g₁, g₂, g₃} {g₁, g₂, g₃}) := by
  rintro ⟨h1, h2⟩
  have hcard : 4 ≤ ({g₁, g₂, g₃} : Finset GameState).card :=
    uniform_Q_card_ge_four {g₁, g₂, g₃} h1 h2
  have : ({g₁, g₂, g₃} : Finset GameState).card ≤ 3 := by
    rw [show ({g₁, g₂, g₃} : Finset GameState) = insert g₁ (insert g₂ {g₃}) from rfl]
    calc (insert g₁ (insert g₂ ({g₃} : Finset GameState))).card
        ≤ (insert g₂ ({g₃} : Finset GameState)).card + 1 := Finset.card_insert_le _ _
      _ ≤ ({g₃} : Finset GameState).card + 1 + 1 := by
        gcongr; exact Finset.card_insert_le _ _
      _ = 3 := by simp
  omega

/-- **Quadruple candidates are impossible for uniform Q**: `|Q| = 4` cannot
hold all five required residue classes `{8, 6, 4, 2, 0}`. Combined with the
trivial/singleton/pair/triple cases, this shows **every uniform phase set
must have at least five states** — the residue-0 self-loop closure forces
the fifth. -/
theorem quadruple_uniform_Q_impossible (g₁ g₂ g₃ g₄ : GameState) :
    ¬ (BagWinning {GameState.init} {g₁, g₂, g₃, g₄} ∧
       BagWinning {g₁, g₂, g₃, g₄} {g₁, g₂, g₃, g₄}) := by
  rintro ⟨h1, h2⟩
  have hcard : 5 ≤ ({g₁, g₂, g₃, g₄} : Finset GameState).card :=
    uniform_Q_card_ge_five {g₁, g₂, g₃, g₄} h1 h2
  have : ({g₁, g₂, g₃, g₄} : Finset GameState).card ≤ 4 := by
    rw [show ({g₁, g₂, g₃, g₄} : Finset GameState)
        = insert g₁ (insert g₂ (insert g₃ {g₄})) from rfl]
    calc (insert g₁ (insert g₂ (insert g₃ ({g₄} : Finset GameState)))).card
        ≤ (insert g₂ (insert g₃ ({g₄} : Finset GameState))).card + 1 :=
          Finset.card_insert_le _ _
      _ ≤ (insert g₃ ({g₄} : Finset GameState)).card + 1 + 1 := by
        gcongr; exact Finset.card_insert_le _ _
      _ ≤ ({g₄} : Finset GameState).card + 1 + 1 + 1 := by
        gcongr; exact Finset.card_insert_le _ _
      _ = 4 := by simp
  omega

/-- **The uniform-phase route is impossible — for *every* `Q`.** No single
set can serve as the uniform phase of a 35-reset. `BagWinning {init} Q`
forces a **residue-8** state into `Q` (init advances `+8 mod 10` in one
bag), while `BagWinning Q {init}` demands *every* `Q`-state close to `init`
in one bag — which requires **residue 2** (the unique `r` with
`(r + 8) % 10 = 0`). Since `8 ≠ 2 (mod 10)`, the open and close phases are
mutually exclusive on the shared residue-8 witness.

This **kills the entire uniform decomposition** (`Winning_35_of_uniform_phase`)
unconditionally, strengthening the `|Q| ≤ 4` cardinality ladder: it is not
that `Q` must be large, but that *no* `Q` works at all. Any genuine 35-reset
must therefore use a **non-uniform** chain `P₁ → P₂ → P₃ → P₄` with distinct
phase sets. -/
theorem uniform_open_close_impossible
    (Q : Finset GameState)
    (hopen : BagWinning {GameState.init} Q)
    (hclose : BagWinning Q {GameState.init}) :
    False := by
  -- hopen forces a residue-8 state g8 ∈ Q (full bag, WF board)
  obtain ⟨g8, hg8_mem, hg8_wf, hg8_bag, hg8_res⟩ :=
    BagWinning_init_implies_Q_has_initial Q hopen
  -- hclose makes g8 close to init in one bag, landing at residue (8+8)%10 = 6
  obtain ⟨g', hg'_mem, _, _, hg'_res⟩ :=
    BagWinning_step_residue Q {GameState.init} hclose g8 hg8_mem hg8_wf hg8_bag
  -- but the only member of {init} has count residue 0
  rw [Finset.mem_singleton] at hg'_mem
  rw [hg'_mem, GameState.init_board_count] at hg'_res
  -- hg'_res : 0 % 10 = (g8.count + 8) % 10 ; hg8_res : g8.count % 10 = 8 ⟹ 0 = 6
  omega

/-! ### Per-column budget

`Proofs.Invariants.ColumnCount` refines the cell-count law to each column:
a placement adds its per-column profile and every line clear removes
exactly **one** cell from every valid column. Threading that law through
the winning induction gives a *vector* bridge: all ten column equations
share the **same** line-clear counter `k`. Consequences:

* a 35-reset must deliver **exactly 14 cells to every column** (the
  10 × 14 budget from the file header, now per-column rather than total);
* any state that can still reach `init` in `n` moves obeys
  `10 · colCount j ≤ count + 4n` — closing phases cannot contain tall
  towers, no matter where they sit. -/

/-- **Column bridge**: vector form of `set_count_bag_bridge`. Some endpoint
`g_end ∈ T` satisfies, for every column `j < 10`, the equation
`g_end.colCount j + k = g₀.colCount j + c j` with a **shared** line-clear
counter `k`, where the per-column deliveries `c` total `4·n` cells. -/
theorem WinningTo.set_column_bridge :
    ∀ {n : ℕ} {T : Finset GameState} {g₀ : GameState},
      WinningTo T n g₀ →
      Board.WF GameConfig.standard g₀.board →
      ∀ (s : ℕ → Piece), LegalSequenceFrom g₀.bag s →
      ∃ g_end ∈ T, ∃ k : ℕ, ∃ c : ℕ → ℕ,
        (∀ j, j < 10 →
          g_end.board.colCount j + k = g₀.board.colCount j + c j) ∧
        (∑ j ∈ Finset.range 10, c j = 4 * n) ∧
        g_end.bag = bagAt g₀.bag s n ∧
        Board.WF GameConfig.standard g_end.board := by
  intro n
  induction n with
  | zero =>
      intro T g₀ h hwf s _hl
      refine ⟨g₀, h, 0, fun _ => 0, fun j _ => rfl, by simp, ?_, hwf⟩
      rw [bagAt_zero]
  | succ n ih =>
      intro T g₀ h hwf s hl
      rw [winningTo_succ_iff] at h
      obtain ⟨_, hstep⟩ := h
      have hp : s 0 ∈ g₀.bag := by
        have hcd := hl 0
        rw [bagAt_zero, Bag.canDraw_iff_mem] at hcd
        exact hcd
      obtain ⟨pl₀, hpl₀, hnext⟩ := hstep (s 0) hp
      have hpl₀_spec := (Placement.mem_allValidFor _ _ _).mp hpl₀
      have hpl₀_piece : pl₀.piece = s 0 := hpl₀_spec.1
      have hpl₀_valid : pl₀.Valid GameConfig.standard := hpl₀_spec.2
      have heta : ({ pl₀ with piece := s 0 } : Placement) = pl₀ :=
        Placement.eta_of_piece_eq hpl₀_piece
      have hbag_next :
          (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag
            = g₀.bag.draw (s 0) := rfl
      have hboard_next :
          (adversarialStep GameConfig.standard g₀ (s 0) pl₀).board
            = Placement.applyStep GameConfig.standard g₀.board pl₀ := by
        rw [adversarialStep_board, heta]
      have hwf' :
          Board.WF GameConfig.standard
            (adversarialStep GameConfig.standard g₀ (s 0) pl₀).board := by
        rw [hboard_next]
        exact Placement.applyStep_wf hwf hpl₀_valid
      have hsuffix :
          LegalSequenceFrom (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag
            (fun k => s (k + 1)) := by
        have hsuf := hl.suffix 1
        have hbag1_eq : bagAt g₀.bag s 1
            = (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag := by
          rw [bagAt_succ, bagAt_zero, hbag_next]
        rw [hbag1_eq] at hsuf
        convert hsuf using 1
        ext k; ring_nf
      obtain ⟨g_end, hg_end_mem, k', c', hcol', hsum', hbag_eq, hwf_end⟩ :=
        ih hnext hwf' _ hsuffix
      -- the per-column step equation, shared clears counter
      have hstep_col : ∀ j, j < 10 →
          (adversarialStep GameConfig.standard g₀ (s 0) pl₀).board.colCount j
            + Board.linesCleared GameConfig.standard (pl₀.place g₀.board)
          = g₀.board.colCount j + pl₀.colProfile j := by
        intro j hj
        rw [hboard_next]
        exact applyStep_colCount GameConfig.standard g₀.board pl₀ hj
      have hsum4 : ∑ j ∈ Finset.range 10, pl₀.colProfile j = 4 :=
        Placement.sum_colProfile hpl₀_valid
      refine ⟨g_end, hg_end_mem,
        k' + Board.linesCleared GameConfig.standard (pl₀.place g₀.board),
        fun j => c' j + pl₀.colProfile j, ?_, ?_, ?_, hwf_end⟩
      · intro j hj
        have h1 := hcol' j hj
        have h2 := hstep_col j hj
        exact (by omega : g_end.board.colCount j
            + (k' + Board.linesCleared GameConfig.standard (pl₀.place g₀.board))
          = g₀.board.colCount j + (c' j + pl₀.colProfile j))
      · have hgoal : ∑ j ∈ Finset.range 10, (c' j + pl₀.colProfile j)
            = 4 * (n + 1) := by
          rw [Finset.sum_add_distrib, hsum', hsum4]
          ring
        exact hgoal
      · rw [hbag_eq]
        have h1 : (adversarialStep GameConfig.standard g₀ (s 0) pl₀).bag
                = bagAt g₀.bag s 1 := rfl
        rw [h1]
        have h2 := bagAt_add g₀.bag s 1 n
        have h3 : bagAt g₀.bag s (n + 1) = bagAt g₀.bag s (1 + n) := by
          congr 1; ring
        rw [h3, h2]
        congr 1
        ext k; ring_nf

/-- **Exact column budget for a 35-reset**: along any legal sequence, a
winning trace clears exactly **14 lines** and delivers exactly **14 cells
to every column** — the forced solution of the ten column equations
`0 + k = 0 + c j` under `∑ c = 140`. -/
theorem winning_init_exact_column_budget
    (h : Winning blockLength GameState.init)
    (s : ℕ → Piece) (hl : LegalSequence s) :
    ∃ k : ℕ, ∃ c : ℕ → ℕ,
      k = 14 ∧ (∀ j, j < 10 → c j = 14) ∧
      ∑ j ∈ Finset.range 10, c j = 4 * blockLength := by
  have hwt : WinningTo {GameState.init} blockLength GameState.init :=
    (winning_iff_winningTo_init blockLength GameState.init).mp h
  have hl' : LegalSequenceFrom GameState.init.bag s := by
    rw [GameState.init_bag]; exact hl
  have hwf : Board.WF GameConfig.standard GameState.init.board := by
    rw [GameState.init_board]; exact Board.empty_wf _
  obtain ⟨g_end, hmem, k, c, hcol, hsum, _, _⟩ :=
    WinningTo.set_column_bridge hwt hwf s hl'
  rw [Finset.mem_singleton] at hmem
  subst hmem
  have hinit_col : ∀ i, GameState.init.board.colCount i = 0 := by
    intro i
    rw [GameState.init_board]
    exact Board.colCount_empty i
  -- every column equation collapses to `c j = k`
  have hck : ∀ j, j < 10 → c j = k := by
    intro j hj
    have h1 := hcol j hj
    have h0 := hinit_col j
    omega
  -- the delivery sum then forces `10 k = 140`
  have hsum_k : ∑ j ∈ Finset.range 10, c j = 10 * k := by
    rw [Finset.sum_congr rfl (fun j hj => hck j (Finset.mem_range.1 hj)),
      Finset.sum_const, Finset.card_range, smul_eq_mul]
  have hk14 : k = 14 := by
    rw [blockLength_eq] at hsum
    omega
  exact ⟨k, c, hk14, fun j hj => by rw [hck j hj, hk14], hsum⟩

/-- **Closing column bound**: a state that can still reach `init` in `n`
moves satisfies `10 · colCount j ≤ count + 4n` in every column `j < 10`.
Each of the at-most-`(count + 4n)/10` line clears removes only one cell
per column, and `init` has empty columns — so tall towers cannot be
disposed of. Not derivable from the total-count bridge. -/
theorem closing_column_bound {n : ℕ} {g : GameState}
    (h : WinningTo {GameState.init} n g)
    (hwf : Board.WF GameConfig.standard g.board)
    (hbag : g.bag.Nonempty)
    {j : ℕ} (hj : j < 10) :
    10 * g.board.colCount j ≤ g.board.count + 4 * n := by
  obtain ⟨s, hl⟩ := exists_legalSequenceFrom g.bag hbag
  obtain ⟨g_end, hmem, k, c, hcol, hsum, _hbag, _hwf_end⟩ :=
    WinningTo.set_column_bridge h hwf s hl
  rw [Finset.mem_singleton] at hmem
  subst hmem
  have hinit_col : ∀ i, GameState.init.board.colCount i = 0 := by
    intro i
    rw [GameState.init_board]
    exact Board.colCount_empty i
  -- sum the ten column equations to recover `10·k = count + 4n`
  have hsummed : ∑ i ∈ Finset.range 10, (GameState.init.board.colCount i + k)
      = ∑ i ∈ Finset.range 10, (g.board.colCount i + c i) :=
    Finset.sum_congr rfl (fun i hi => hcol i (Finset.mem_range.1 hi))
  have hsum_g : ∑ i ∈ Finset.range 10, g.board.colCount i = g.board.count :=
    Board.sum_colCount hwf
  have hzero : ∑ i ∈ Finset.range 10, GameState.init.board.colCount i = 0 :=
    Finset.sum_eq_zero (fun i _ => hinit_col i)
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, hsum_g, hsum, hzero,
    Finset.sum_const, Finset.card_range, smul_eq_mul] at hsummed
  -- hsummed : 0 + 10 * k = g.count + 4 * n
  have hk := hcol j hj
  have hjcol := hinit_col j
  omega

/-- **Closing-phase column bound**: every member of a candidate `P_4`
admitting `BagWinning P_4 {init}` (with WF board and nonempty bag) has
`10 · colCount j ≤ count + 28` in every column. E.g. a closing state with
count 2 has no column taller than 3; with count 12, none taller than 4. -/
theorem BagWinning_close_column_bound
    (P_4 : Finset GameState)
    (h : BagWinning P_4 {GameState.init})
    {g : GameState} (hg : g ∈ P_4)
    (hwf : Board.WF GameConfig.standard g.board)
    (hbag : g.bag.Nonempty)
    {j : ℕ} (hj : j < 10) :
    10 * g.board.colCount j ≤ g.board.count + 28 := by
  have h' := closing_column_bound (h g hg) hwf hbag hj
  rw [piecesPerBag_eq] at h'
  omega

/-- **Full `P_4` witness profile**: any 5-phase decomposition yields a
`P_4` witness that simultaneously satisfies the count menu (≡ 2 mod 10,
≤ 112) *and* the per-column closing bound. The search for `P_4` candidates
can be restricted to boards with every column at most `(count + 28)/10`
tall — e.g. at most 14 cells per column even at the maximal count 112. -/
theorem P4_witness_column_bound
    (P_1 P_2 P_3 P_4 : Finset GameState)
    (h12 : BagWinning {GameState.init} P_1)
    (h23 : BagWinning P_1 P_2)
    (h34 : BagWinning P_2 P_3)
    (h45 : BagWinning P_3 P_4)
    (h5 : BagWinning P_4 {GameState.init}) :
    ∃ g ∈ P_4, Board.WF GameConfig.standard g.board ∧
      g.bag = Bag.full ∧
      g.board.count % 10 = 2 ∧ g.board.count ≤ 112 ∧
      ∀ j, j < 10 → 10 * g.board.colCount j ≤ g.board.count + 28 := by
  obtain ⟨g, hmem, hwf, hbag, hres, hle⟩ :=
    P4_count_menu P_1 P_2 P_3 P_4 h12 h23 h34 h45
  refine ⟨g, hmem, hwf, hbag, hres, hle, ?_⟩
  intro j hj
  have hne : g.bag.Nonempty := by rw [hbag]; exact Bag.full_nonempty
  exact BagWinning_close_column_bound P_4 h5 hmem hwf hne hj

/-- **`P_1` member column bound**: *every* member of a phase-1 candidate
(not just the bridge witness) can still reach `init` in 28 moves, so
`10 · colCount j ≤ count + 112` in every column. -/
theorem P1_member_column_bound
    (P_1 P_2 P_3 P_4 : Finset GameState)
    (h23 : BagWinning P_1 P_2)
    (h34 : BagWinning P_2 P_3)
    (h45 : BagWinning P_3 P_4)
    (h5 : BagWinning P_4 {GameState.init})
    {g : GameState} (hg : g ∈ P_1)
    (hwf : Board.WF GameConfig.standard g.board)
    (hbag : g.bag.Nonempty)
    {j : ℕ} (hj : j < 10) :
    10 * g.board.colCount j ≤ g.board.count + 112 := by
  have hw : WinningTo {GameState.init}
      (piecesPerBag + piecesPerBag + piecesPerBag + piecesPerBag) g :=
    (((h23 g hg).compose h34).compose h45).compose h5
  have h' := closing_column_bound hw hwf hbag hj
  rw [piecesPerBag_eq] at h'
  omega

/-- **`P_2` member column bound**: every member of a phase-2 candidate can
still reach `init` in 21 moves, so `10 · colCount j ≤ count + 84`. -/
theorem P2_member_column_bound
    (P_2 P_3 P_4 : Finset GameState)
    (h34 : BagWinning P_2 P_3)
    (h45 : BagWinning P_3 P_4)
    (h5 : BagWinning P_4 {GameState.init})
    {g : GameState} (hg : g ∈ P_2)
    (hwf : Board.WF GameConfig.standard g.board)
    (hbag : g.bag.Nonempty)
    {j : ℕ} (hj : j < 10) :
    10 * g.board.colCount j ≤ g.board.count + 84 := by
  have hw : WinningTo {GameState.init}
      (piecesPerBag + piecesPerBag + piecesPerBag) g :=
    ((h34 g hg).compose h45).compose h5
  have h' := closing_column_bound hw hwf hbag hj
  rw [piecesPerBag_eq] at h'
  omega

/-- **`P_3` member column bound**: every member of a phase-3 candidate can
still reach `init` in 14 moves, so `10 · colCount j ≤ count + 56`. -/
theorem P3_member_column_bound
    (P_3 P_4 : Finset GameState)
    (h45 : BagWinning P_3 P_4)
    (h5 : BagWinning P_4 {GameState.init})
    {g : GameState} (hg : g ∈ P_3)
    (hwf : Board.WF GameConfig.standard g.board)
    (hbag : g.bag.Nonempty)
    {j : ℕ} (hj : j < 10) :
    10 * g.board.colCount j ≤ g.board.count + 56 := by
  have hw : WinningTo {GameState.init} (piecesPerBag + piecesPerBag) g :=
    (h45 g hg).compose h5
  have h' := closing_column_bound hw hwf hbag hj
  rw [piecesPerBag_eq] at h'
  omega

/-! ## Invariant transport and the pruned decomposition

`WinningTo` target sets can be *shrunk for free* along any step-indexed
invariant: if `I` is preserved by every adversarial step (counting down
the remaining draws) and forces `pred` on arrival, then a strategy into
`T` is already a strategy into `T.filter pred`.

Instantiating `I` with the count/bag/well-formedness facts collected above
turns all the necessary conditions on the phase boards `P₁..P₄` into a
**WLOG principle** (`phase_decomposition_pruned`): *any* decomposition can
be replaced by one whose phase boards consist exclusively of states
satisfying every known obstruction — well-formed board, full bag, count
residue and budget, and the per-column closing budget. This formally
prunes the search space for the `native_decide` programme below. -/

/-- **Invariant transport.** If the step-indexed invariant `I` is preserved
by every adversarial step (`I (k+1)` before implies `I k` after) and, on
arrival, membership in `T` plus `I 0` forces `pred`, then any strategy
winning into `T` also wins into the *filtered* target `T.filter pred`. -/
theorem WinningTo.filter_target {T : Finset GameState}
    (I : ℕ → GameState → Prop) (pred : GameState → Prop) [DecidablePred pred]
    (hstep : ∀ (k : ℕ) (g : GameState) (p : Piece) (pl : Placement),
      p ∈ g.bag → pl ∈ Placement.allValidFor GameConfig.standard p →
      I (k + 1) g → I k (adversarialStep GameConfig.standard g p pl))
    (hfinal : ∀ g ∈ T, I 0 g → pred g) :
    ∀ {n : ℕ} {g : GameState}, WinningTo T n g → I n g →
      WinningTo (T.filter pred) n g := by
  intro n
  induction n with
  | zero =>
      intro g h hI
      rw [winningTo_zero_iff] at h ⊢
      exact Finset.mem_filter.mpr ⟨h, hfinal g h hI⟩
  | succ n ih =>
      intro g h hI
      rw [winningTo_succ_iff] at h ⊢
      obtain ⟨hnl, hall⟩ := h
      refine ⟨hnl, ?_⟩
      intro p hp
      obtain ⟨pl, hpl, hnext⟩ := hall p hp
      exact ⟨pl, hpl, ih hnext (hstep n g p pl hp hpl hI)⟩

/-- Bag cardinality dictated by the 7-bag draw cycle, indexed by the number
of draws *remaining* to the next bag boundary: full (7) at multiples of 7,
otherwise exactly `k % 7` pieces left in the current bag. -/
def cycleBagCard (k : ℕ) : ℕ := if k % 7 = 0 then 7 else k % 7

/-- One draw advances the cycle: a bag of `cycleBagCard (k+1)` pieces
holding `p` yields, after drawing `p`, exactly `cycleBagCard k` pieces.
The refill fires exactly at the bag boundary `k ≡ 0 (mod 7)`. -/
theorem cycleBagCard_step {bag : Bag} {p : Piece} (hp : p ∈ bag) (k : ℕ)
    (hcard : bag.card = cycleBagCard (k + 1)) :
    (bag.draw p).card = cycleBagCard k := by
  unfold cycleBagCard at hcard ⊢
  by_cases h7 : k % 7 = 0
  · -- boundary: the bag held exactly one piece; the draw refills to 7
    have hk1 : (k + 1) % 7 = 1 := by omega
    rw [if_neg (by omega : ¬(k + 1) % 7 = 0), hk1] at hcard
    rw [if_pos h7]
    exact Bag.card_draw_of_singleton hp hcard
  · -- interior: a plain erase, the cardinality drops by one
    have hge : 2 ≤ bag.card := by
      by_cases hb : (k + 1) % 7 = 0
      · rw [if_pos hb] at hcard; omega
      · rw [if_neg hb] at hcard; omega
    rw [if_neg h7, Bag.card_draw_of_ge_two hp hge]
    by_cases hb : (k + 1) % 7 = 0
    · rw [if_pos hb] at hcard
      omega
    · rw [if_neg hb] at hcard
      omega

/-- The transported invariant for one leg of the reset: with `k` adversarial
draws remaining, the board is well-formed, the bag sits at its forced cycle
cardinality, and the cell count is pinned to residue `r` and budget `B`
*after accounting for the `4·k` cells still to be delivered*. -/
def LegInvariant (r B : ℕ) (k : ℕ) (g : GameState) : Prop :=
  Board.WF GameConfig.standard g.board ∧
  g.bag.card = cycleBagCard k ∧
  (g.board.count + 4 * k) % 10 = r ∧
  g.board.count + 4 * k ≤ B

/-- `LegInvariant` is preserved by every adversarial step: the board stays
well-formed, the bag follows the draw cycle, and `count + 4·k` keeps both
its residue (the `−10·clears` vanishes mod 10) and its budget (count grows
by at most 4 per draw). -/
theorem LegInvariant_step (r B : ℕ) (k : ℕ) (g : GameState) (p : Piece)
    (pl : Placement) (hp : p ∈ g.bag)
    (hpl : pl ∈ Placement.allValidFor GameConfig.standard p)
    (hI : LegInvariant r B (k + 1) g) :
    LegInvariant r B k (adversarialStep GameConfig.standard g p pl) := by
  unfold LegInvariant at hI ⊢
  obtain ⟨hwf, hbag, hres, hbound⟩ := hI
  have hpl_spec := (Placement.mem_allValidFor _ _ _).mp hpl
  have heta : ({ pl with piece := p } : Placement) = pl :=
    Placement.eta_of_piece_eq hpl_spec.1
  have hboard : (adversarialStep GameConfig.standard g p pl).board
      = Placement.applyStep GameConfig.standard g.board pl := by
    rw [adversarialStep_board, heta]
  have hcount : (Placement.applyStep GameConfig.standard g.board pl).count
      + 10 * Board.linesCleared GameConfig.standard (pl.place g.board)
      = g.board.count + 4 :=
    applyStep_count GameConfig.standard g.board pl hwf hpl_spec.2
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hboard]
    exact Placement.applyStep_wf hwf hpl_spec.2
  · rw [adversarialStep_bag]
    exact cycleBagCard_step hp k hbag
  · rw [hboard]
    omega
  · rw [hboard]
    omega

/-- `init` launches the first leg: empty board, full bag, 28 cells due at
residue 8 within budget 28. -/
theorem LegInvariant_init : LegInvariant 8 28 7 GameState.init := by
  unfold LegInvariant
  refine ⟨?_, ?_, by simp, by simp⟩
  · rw [GameState.init_board]
    exact Board.empty_wf _
  · simp [cycleBagCard]

/-- A phase state (well-formed, full bag, count residue `r₀`, count ≤ `B₀`)
launches the next leg with residue `(r₀ + 28) % 10` and budget `B₀ + 28`. -/
theorem LegInvariant_launch {g : GameState} {r₀ B₀ : ℕ}
    (hwf : Board.WF GameConfig.standard g.board)
    (hbag : g.bag = Bag.full)
    (hres : g.board.count % 10 = r₀)
    (hB : g.board.count ≤ B₀) :
    LegInvariant ((r₀ + 28) % 10) (B₀ + 28) 7 g := by
  unfold LegInvariant
  refine ⟨hwf, ?_, by omega, by omega⟩
  simp [hbag, cycleBagCard]

/-- On arrival (`k = 0`) the leg invariant delivers the phase facts: a
well-formed board, a refilled full bag, and the pinned count residue and
budget. -/
theorem LegInvariant_arrive {r B : ℕ} {g : GameState}
    (h : LegInvariant r B 0 g) :
    Board.WF GameConfig.standard g.board ∧ g.bag = Bag.full ∧
      g.board.count % 10 = r ∧ g.board.count ≤ B := by
  unfold LegInvariant at h
  obtain ⟨hwf, hbag, hres, hB⟩ := h
  refine ⟨hwf, ?_, by omega, by omega⟩
  exact (Bag.card_eq_seven_iff_eq_full _).mp (by simp [hbag, cycleBagCard])

/-- Every obstruction this file derives for a state of phase `i`
(`i ∈ {1,2,3,4}`, i.e. `7·i` pieces into the 35-piece reset): well-formed
board, refilled full bag, count pinned to residue and budget `28·i`, every
column within the closing budget `10·colCount ≤ count + 28·(5−i)`, and all
cells strictly inside the 10×20 field (rows `< 20` forced by `¬ lost` at
the next leg's first step). The in-field clause makes each obstruction set
**finite**: a phase board lives inside the playfield with a full bag. -/
def PhaseObstruction (i : ℕ) (g : GameState) : Prop :=
  Board.WF GameConfig.standard g.board ∧
  g.bag = Bag.full ∧
  g.board.count % 10 = 28 * i % 10 ∧
  g.board.count ≤ 28 * i ∧
  (∀ j, j < 10 → 10 * g.board.colCount j ≤ g.board.count + 28 * (5 - i)) ∧
  ∀ p ∈ g.board, p.2 < 20

/-- Arrival at phase 1 satisfies every phase-1 obstruction. -/
theorem phase1_arrival (P₁ P₂ P₃ P₄ : Finset GameState)
    (h2 : BagWinning P₁ P₂) (h3 : BagWinning P₂ P₃)
    (h4 : BagWinning P₃ P₄) (h5 : BagWinning P₄ {GameState.init}) :
    ∀ g ∈ P₁, LegInvariant 8 28 0 g → PhaseObstruction 1 g := by
  intro g hg hI
  obtain ⟨hwf, hbag, hres, hB⟩ := LegInvariant_arrive hI
  have hne : g.bag.Nonempty := by rw [hbag]; exact Bag.full_nonempty
  unfold PhaseObstruction
  refine ⟨hwf, hbag, by omega, by omega, ?_, ?_⟩
  · intro j hj
    have h' := P1_member_column_bound P₁ P₂ P₃ P₄ h2 h3 h4 h5 hg hwf hne hj
    omega
  · exact (GameState.not_lost_iff_forall_row_lt GameConfig.standard g).mp
      ((winningTo_succ_iff _ _ _).mp (h2 g hg)).1

/-- Arrival at phase 2 satisfies every phase-2 obstruction. -/
theorem phase2_arrival (P₂ P₃ P₄ : Finset GameState)
    (h3 : BagWinning P₂ P₃) (h4 : BagWinning P₃ P₄)
    (h5 : BagWinning P₄ {GameState.init}) :
    ∀ g ∈ P₂, LegInvariant 6 56 0 g → PhaseObstruction 2 g := by
  intro g hg hI
  obtain ⟨hwf, hbag, hres, hB⟩ := LegInvariant_arrive hI
  have hne : g.bag.Nonempty := by rw [hbag]; exact Bag.full_nonempty
  unfold PhaseObstruction
  refine ⟨hwf, hbag, by omega, by omega, ?_, ?_⟩
  · intro j hj
    have h' := P2_member_column_bound P₂ P₃ P₄ h3 h4 h5 hg hwf hne hj
    omega
  · exact (GameState.not_lost_iff_forall_row_lt GameConfig.standard g).mp
      ((winningTo_succ_iff _ _ _).mp (h3 g hg)).1

/-- Arrival at phase 3 satisfies every phase-3 obstruction. -/
theorem phase3_arrival (P₃ P₄ : Finset GameState)
    (h4 : BagWinning P₃ P₄) (h5 : BagWinning P₄ {GameState.init}) :
    ∀ g ∈ P₃, LegInvariant 4 84 0 g → PhaseObstruction 3 g := by
  intro g hg hI
  obtain ⟨hwf, hbag, hres, hB⟩ := LegInvariant_arrive hI
  have hne : g.bag.Nonempty := by rw [hbag]; exact Bag.full_nonempty
  unfold PhaseObstruction
  refine ⟨hwf, hbag, by omega, by omega, ?_, ?_⟩
  · intro j hj
    have h' := P3_member_column_bound P₃ P₄ h4 h5 hg hwf hne hj
    omega
  · exact (GameState.not_lost_iff_forall_row_lt GameConfig.standard g).mp
      ((winningTo_succ_iff _ _ _).mp (h4 g hg)).1

/-- Arrival at phase 4 satisfies every phase-4 obstruction. -/
theorem phase4_arrival (P₄ : Finset GameState)
    (h5 : BagWinning P₄ {GameState.init}) :
    ∀ g ∈ P₄, LegInvariant 2 112 0 g → PhaseObstruction 4 g := by
  intro g hg hI
  obtain ⟨hwf, hbag, hres, hB⟩ := LegInvariant_arrive hI
  have hne : g.bag.Nonempty := by rw [hbag]; exact Bag.full_nonempty
  unfold PhaseObstruction
  refine ⟨hwf, hbag, by omega, by omega, ?_, ?_⟩
  · intro j hj
    have h' := BagWinning_close_column_bound P₄ h5 hg hwf hne hj
    omega
  · exact (GameState.not_lost_iff_forall_row_lt GameConfig.standard g).mp
      ((winningTo_succ_iff _ _ _).mp (h5 g hg)).1

/-- **The pruned decomposition (WLOG principle).** Any 5-phase decomposition
of the 35-piece reset can be replaced by one whose phase boards contain
*only* states satisfying every derived obstruction: well-formed board, full
bag, count residue/budget `28·i`, all ten per-column closing budgets, and
in-field rows (`< 20`).

Consequence for the search programme: when hunting for candidate phase
boards `P₁..P₄` (e.g. by `native_decide` over explicit Finsets), it is
*no loss of generality* to enumerate only obstruction-satisfying states —
e.g. phase-1 boards with exactly 28, 18, or 8 cells spread so that no
column exceeds `(count + 112)/10` cells. -/
theorem phase_decomposition_pruned
    (P₁ P₂ P₃ P₄ : Finset GameState)
    (h1 : BagWinning {GameState.init} P₁) (h2 : BagWinning P₁ P₂)
    (h3 : BagWinning P₂ P₃) (h4 : BagWinning P₃ P₄)
    (h5 : BagWinning P₄ {GameState.init}) :
    ∃ Q₁ Q₂ Q₃ Q₄ : Finset GameState,
      Q₁ ⊆ P₁ ∧ Q₂ ⊆ P₂ ∧ Q₃ ⊆ P₃ ∧ Q₄ ⊆ P₄ ∧
      (∀ g ∈ Q₁, PhaseObstruction 1 g) ∧ (∀ g ∈ Q₂, PhaseObstruction 2 g) ∧
      (∀ g ∈ Q₃, PhaseObstruction 3 g) ∧ (∀ g ∈ Q₄, PhaseObstruction 4 g) ∧
      BagWinning {GameState.init} Q₁ ∧ BagWinning Q₁ Q₂ ∧
      BagWinning Q₂ Q₃ ∧ BagWinning Q₃ Q₄ ∧ BagWinning Q₄ {GameState.init} := by
  classical
  refine ⟨P₁.filter (PhaseObstruction 1), P₂.filter (PhaseObstruction 2),
    P₃.filter (PhaseObstruction 3), P₄.filter (PhaseObstruction 4),
    Finset.filter_subset _ _, Finset.filter_subset _ _,
    Finset.filter_subset _ _, Finset.filter_subset _ _,
    fun g hg => (Finset.mem_filter.mp hg).2,
    fun g hg => (Finset.mem_filter.mp hg).2,
    fun g hg => (Finset.mem_filter.mp hg).2,
    fun g hg => (Finset.mem_filter.mp hg).2,
    ?_, ?_, ?_, ?_, ?_⟩
  · -- {init} → pruned P₁
    intro g hg
    rw [Finset.mem_singleton] at hg
    subst hg
    exact WinningTo.filter_target (LegInvariant 8 28) (PhaseObstruction 1)
      (LegInvariant_step 8 28) (phase1_arrival P₁ P₂ P₃ P₄ h2 h3 h4 h5)
      (h1 GameState.init (Finset.mem_singleton.mpr rfl)) LegInvariant_init
  · -- pruned P₁ → pruned P₂
    intro g hg
    have hmem := Finset.mem_filter.mp hg
    have hpred := hmem.2
    unfold PhaseObstruction at hpred
    have hlaunch : LegInvariant 6 56 7 g :=
      LegInvariant_launch hpred.1 hpred.2.1 hpred.2.2.1 hpred.2.2.2.1
    exact WinningTo.filter_target (LegInvariant 6 56) (PhaseObstruction 2)
      (LegInvariant_step 6 56) (phase2_arrival P₂ P₃ P₄ h3 h4 h5)
      (h2 g hmem.1) hlaunch
  · -- pruned P₂ → pruned P₃
    intro g hg
    have hmem := Finset.mem_filter.mp hg
    have hpred := hmem.2
    unfold PhaseObstruction at hpred
    have hlaunch : LegInvariant 4 84 7 g :=
      LegInvariant_launch hpred.1 hpred.2.1 hpred.2.2.1 hpred.2.2.2.1
    exact WinningTo.filter_target (LegInvariant 4 84) (PhaseObstruction 3)
      (LegInvariant_step 4 84) (phase3_arrival P₃ P₄ h4 h5)
      (h3 g hmem.1) hlaunch
  · -- pruned P₃ → pruned P₄
    intro g hg
    have hmem := Finset.mem_filter.mp hg
    have hpred := hmem.2
    unfold PhaseObstruction at hpred
    have hlaunch : LegInvariant 2 112 7 g :=
      LegInvariant_launch hpred.1 hpred.2.1 hpred.2.2.1 hpred.2.2.2.1
    exact WinningTo.filter_target (LegInvariant 2 112) (PhaseObstruction 4)
      (LegInvariant_step 2 112) (phase4_arrival P₄ h5)
      (h4 g hmem.1) hlaunch
  · -- pruned P₄ → {init}: shrinking the source is free
    intro g hg
    exact h5 g (Finset.mem_filter.mp hg).1

/-! ## Completeness of the phase method

`Winning_35_of_phase_decomposition` shows 5 single-bag obligations suffice.
The converse also holds: *any* winning 35-reset induces a decomposition,
obtained by splitting the alternating strategy tree at each bag boundary
and collecting the (finitely many) states the strategy can occupy there.
Hence the phase search is a **complete** method — `Winning 35 init` holds
iff some decomposition exists — and by `phase_decomposition_pruned` the
search may further be limited to obstruction-satisfying phase boards. -/

/-- Enlarging the target preserves winning. -/
theorem WinningTo.mono {T₁ T₂ : Finset GameState} (hsub : T₁ ⊆ T₂) :
    ∀ {n : ℕ} {g : GameState}, WinningTo T₁ n g → WinningTo T₂ n g := by
  intro n
  induction n with
  | zero =>
      intro g h
      rw [winningTo_zero_iff] at h ⊢
      exact hsub h
  | succ n ih =>
      intro g h
      rw [winningTo_succ_iff] at h ⊢
      obtain ⟨hnl, hall⟩ := h
      refine ⟨hnl, ?_⟩
      intro p hp
      obtain ⟨pl, hpl, hnext⟩ := hall p hp
      exact ⟨pl, hpl, ih hnext⟩

/-- **Target monotonicity for `BagWinning`**: enlarging the destination phase
set preserves bag-winning. If every source state can be steered into `P_out`
across the bag, the same strategy lands in any superset `P_out'`. This is the
relaxation move a phase-board search uses to *widen* an intermediate target so
the next bag's source obligation becomes easier. Immediate from
`WinningTo.mono`. -/
theorem BagWinning.mono_target {P_in P_out P_out' : Finset GameState}
    (hsub : P_out ⊆ P_out') (h : BagWinning P_in P_out) :
    BagWinning P_in P_out' := by
  intro g hg
  exact WinningTo.mono hsub (h g hg)

/-- **Source antitonicity for `BagWinning`**: shrinking the source phase set
preserves bag-winning. If every state of `P_in` is bag-winning into `P_out`,
then so is every state of any subset `P_in'` — there are simply fewer source
obligations to discharge. This is the tightening move dual to
`BagWinning.mono_target`. -/
theorem BagWinning.mono_source {P_in P_in' P_out : Finset GameState}
    (hsub : P_in' ⊆ P_in) (h : BagWinning P_in P_out) :
    BagWinning P_in' P_out := by
  intro g hg
  exact h g (hsub hg)

/-- **Source union for `BagWinning`**: if every state of `P` and every state of
`Q` can be steered into the common target `R` across one bag, then so can every
state of `P ∪ Q`. This is the assembly primitive for building a self-recurrent
family from phase pieces: combined with `BagWinning.mono_target`, a collection
of phases each bag-winning into the *union* of all phases yields
`BagWinning (⋃ phases) (⋃ phases)` — exactly the self-recurrent certificate
`tetrisSolvableValid_of_selfBagWinning` consumes. -/
theorem BagWinning.union_source {P Q R : Finset GameState}
    (hP : BagWinning P R) (hQ : BagWinning Q R) :
    BagWinning (P ∪ Q) R := by
  intro g hg
  rw [Finset.mem_union] at hg
  rcases hg with h | h
  · exact hP g h
  · exact hQ g h

/-- **Bag-closed sets are closed under binary union.** If `P` and `Q` are each
self-recurrent (`BagWinning P P`, `BagWinning Q Q`), their union `P ∪ Q` is too:
each set bag-wins into the larger union by `BagWinning.mono_target`, and the two
source obligations fold by `BagWinning.union_source`. So the self-recurrent
(bag-closed) Finsets form a join-semilattice under `∪` — two partial closed
certificates combine into one, the binary core of the n-ary assembly used by
`BagWinning_five_cyclic_union_self`. -/
theorem BagWinning.union_self {P Q : Finset GameState}
    (hP : BagWinning P P) (hQ : BagWinning Q Q) :
    BagWinning (P ∪ Q) (P ∪ Q) := by
  have hsubP : P ⊆ P ∪ Q := by intro x hx; simp [hx]
  have hsubQ : Q ⊆ P ∪ Q := by intro x hx; simp [hx]
  exact (hP.mono_target hsubP).union_source (hQ.mono_target hsubQ)

/-- **Five cyclic phases assemble into a self-recurrent union.** Given five
phase sets each bag-winning into the next (cyclically, `P₄ → P₀`), their union
`U = P₀ ∪ P₁ ∪ P₂ ∪ P₃ ∪ P₄` is bag-closed: `BagWinning U U`. Each phase
bag-wins into `U` by `BagWinning.mono_target` (its individual target is a subset
of `U`), and the five source obligations fold together by
`BagWinning.union_source`. This is the modular *assembly* step — it produces the
self-recurrent certificate `tetrisSolvableValid_of_selfBagWinning` consumes,
cleanly separated from the solvability conclusion drawn in
`tetrisSolvableValid_of_five_cyclic_phases`. -/
theorem BagWinning_five_cyclic_union_self
    {P₀ P₁ P₂ P₃ P₄ : Finset GameState}
    (h01 : BagWinning P₀ P₁) (h12 : BagWinning P₁ P₂)
    (h23 : BagWinning P₂ P₃) (h34 : BagWinning P₃ P₄)
    (h40 : BagWinning P₄ P₀) :
    BagWinning (P₀ ∪ P₁ ∪ P₂ ∪ P₃ ∪ P₄) (P₀ ∪ P₁ ∪ P₂ ∪ P₃ ∪ P₄) := by
  have hsub₀ : P₀ ⊆ P₀ ∪ P₁ ∪ P₂ ∪ P₃ ∪ P₄ := by intro x hx; simp [hx]
  have hsub₁ : P₁ ⊆ P₀ ∪ P₁ ∪ P₂ ∪ P₃ ∪ P₄ := by intro x hx; simp [hx]
  have hsub₂ : P₂ ⊆ P₀ ∪ P₁ ∪ P₂ ∪ P₃ ∪ P₄ := by intro x hx; simp [hx]
  have hsub₃ : P₃ ⊆ P₀ ∪ P₁ ∪ P₂ ∪ P₃ ∪ P₄ := by intro x hx; simp [hx]
  have hsub₄ : P₄ ⊆ P₀ ∪ P₁ ∪ P₂ ∪ P₃ ∪ P₄ := by intro x hx; simp [hx]
  exact (((((h01.mono_target hsub₁).union_source (h12.mono_target hsub₂)).union_source
    (h23.mono_target hsub₃)).union_source (h34.mono_target hsub₄)).union_source
    (h40.mono_target hsub₀))

/-- **Strategy-tree splitting**: a win in `m + n` moves factors through an
intermediate Finset `mid` — the states the strategy can occupy after `m`
moves — with a win into `mid` in `m` moves and, from every state of `mid`,
a win into the original target in `n` moves. Converse of `compose`. -/
theorem WinningTo.split {T : Finset GameState} :
    ∀ (m : ℕ) {n : ℕ} {g : GameState}, WinningTo T (m + n) g →
      ∃ mid : Finset GameState, WinningTo mid m g ∧
        ∀ g' ∈ mid, WinningTo T n g' := by
  intro m
  induction m with
  | zero =>
      intro n g h
      rw [Nat.zero_add] at h
      refine ⟨{g}, ?_, ?_⟩
      · rw [winningTo_zero_iff]
        exact Finset.mem_singleton_self g
      · intro g' hg'
        rw [Finset.mem_singleton] at hg'
        subst hg'
        exact h
  | succ m ih =>
      intro n g h
      rw [show m + 1 + n = (m + n) + 1 from by omega, winningTo_succ_iff] at h
      obtain ⟨hnl, hall⟩ := h
      have hchoice : ∀ p ∈ g.bag,
          ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
            ∃ mid : Finset GameState,
              WinningTo mid m (adversarialStep GameConfig.standard g p pl) ∧
                ∀ g' ∈ mid, WinningTo T n g' := by
        intro p hp
        obtain ⟨pl, hpl, hnext⟩ := hall p hp
        obtain ⟨mid, hmid, hrest⟩ := ih hnext
        exact ⟨pl, hpl, mid, hmid, hrest⟩
      classical
      choose pl hpl mid hmid hrest using hchoice
      refine ⟨g.bag.attach.biUnion (fun q => mid q.1 q.2), ?_, ?_⟩
      · rw [winningTo_succ_iff]
        refine ⟨hnl, ?_⟩
        intro p hp
        refine ⟨pl p hp, hpl p hp, ?_⟩
        exact WinningTo.mono
          (Finset.subset_biUnion_of_mem _ (Finset.mem_attach _ ⟨p, hp⟩))
          (hmid p hp)
      · intro g' hg'
        obtain ⟨q, -, hq⟩ := Finset.mem_biUnion.mp hg'
        exact hrest q.1 q.2 g' hq

/-- Peel one bag off a member-wide obligation: if every state of `P` can
reach `T` in `piecesPerBag + n` moves, some `P'` one bag downstream of `P`
has every state reaching `T` in `n` moves. -/
theorem BagWinning.peel {T P : Finset GameState} {n : ℕ}
    (h : ∀ g ∈ P, WinningTo T (piecesPerBag + n) g) :
    ∃ P' : Finset GameState, BagWinning P P' ∧ ∀ g ∈ P', WinningTo T n g := by
  classical
  have hchoice : ∀ g ∈ P, ∃ mid : Finset GameState,
      WinningTo mid piecesPerBag g ∧ ∀ g' ∈ mid, WinningTo T n g' :=
    fun g hg => WinningTo.split piecesPerBag (h g hg)
  choose mid hmid hrest using hchoice
  refine ⟨P.attach.biUnion (fun q => mid q.1 q.2), ?_, ?_⟩
  · intro g hg
    exact WinningTo.mono
      (Finset.subset_biUnion_of_mem _ (Finset.mem_attach _ ⟨g, hg⟩))
      (hmid g hg)
  · intro g' hg'
    obtain ⟨q, -, hq⟩ := Finset.mem_biUnion.mp hg'
    exact hrest q.1 q.2 g' hq

/-- **Completeness of the 5-phase method**: the 35-piece reset is winnable
*iff* a 5-phase decomposition exists. The forward direction splits the
winning strategy tree at each bag boundary; the reverse is
`Winning_35_of_phase_decomposition`. Combined with
`phase_decomposition_pruned`, deciding `Winning 35 init` reduces exactly
to searching for obstruction-satisfying phase boards. -/
theorem winning_init_iff_phase_decomposition :
    Winning blockLength GameState.init ↔
      ∃ P₁ P₂ P₃ P₄ : Finset GameState,
        BagWinning {GameState.init} P₁ ∧ BagWinning P₁ P₂ ∧
        BagWinning P₂ P₃ ∧ BagWinning P₃ P₄ ∧
        BagWinning P₄ {GameState.init} := by
  constructor
  · intro h
    rw [winning_iff_winningTo_init] at h
    have h0 : ∀ g ∈ ({GameState.init} : Finset GameState),
        WinningTo {GameState.init} (piecesPerBag + 28) g := by
      intro g hg
      rw [Finset.mem_singleton] at hg
      subst hg
      exact h
    obtain ⟨P₁, hb1, h1⟩ := BagWinning.peel h0
    obtain ⟨P₂, hb2, h2⟩ := BagWinning.peel (n := 21) h1
    obtain ⟨P₃, hb3, h3⟩ := BagWinning.peel (n := 14) h2
    obtain ⟨P₄, hb4, h4⟩ := BagWinning.peel (n := piecesPerBag) h3
    exact ⟨P₁, P₂, P₃, P₄, hb1, hb2, hb3, hb4, h4⟩
  · rintro ⟨P₁, P₂, P₃, P₄, hb1, hb2, hb3, hb4, hb5⟩
    exact Winning_35_of_phase_decomposition P₁ P₂ P₃ P₄ hb1 hb2 hb3 hb4 hb5

/-- **The decision problem, fully pruned**: `Winning 35 init` holds *iff* a
decomposition by **obstruction-satisfying** phase boards exists. Combines
the completeness converse with the pruning theorem: the forward direction
splits a winning strategy at the bag boundaries and then filters each
intermediate set by its phase obstruction (no loss of generality); the
reverse direction forgets the obstructions and composes. The search for a
35-reset is therefore *exactly* the search for `P₁..P₄` whose members have
the right cell-count residue, count budget `28·i`, full bag, and bounded
columns. -/
theorem winning_init_iff_pruned_phase_decomposition :
    Winning blockLength GameState.init ↔
      ∃ P₁ P₂ P₃ P₄ : Finset GameState,
        (∀ g ∈ P₁, PhaseObstruction 1 g) ∧ (∀ g ∈ P₂, PhaseObstruction 2 g) ∧
        (∀ g ∈ P₃, PhaseObstruction 3 g) ∧ (∀ g ∈ P₄, PhaseObstruction 4 g) ∧
        BagWinning {GameState.init} P₁ ∧ BagWinning P₁ P₂ ∧
        BagWinning P₂ P₃ ∧ BagWinning P₃ P₄ ∧
        BagWinning P₄ {GameState.init} := by
  constructor
  · intro h
    obtain ⟨P₁, P₂, P₃, P₄, hb1, hb2, hb3, hb4, hb5⟩ :=
      winning_init_iff_phase_decomposition.mp h
    obtain ⟨Q₁, Q₂, Q₃, Q₄, -, -, -, -, ho1, ho2, ho3, ho4,
      hq1, hq2, hq3, hq4, hq5⟩ :=
      phase_decomposition_pruned P₁ P₂ P₃ P₄ hb1 hb2 hb3 hb4 hb5
    exact ⟨Q₁, Q₂, Q₃, Q₄, ho1, ho2, ho3, ho4, hq1, hq2, hq3, hq4, hq5⟩
  · rintro ⟨P₁, P₂, P₃, P₄, -, -, -, -, hb1, hb2, hb3, hb4, hb5⟩
    exact Winning_35_of_phase_decomposition P₁ P₂ P₃ P₄ hb1 hb2 hb3 hb4 hb5

/-! ## The phase search space is finite

A phase board satisfies `PhaseObstruction`, hence is well-formed (columns
`< 10`), in-field (rows `< 20`), and carries the full bag. Such states form
an *explicit finite set*: boards drawn from the playfield's powerset, paired
with `Bag.full`. Combined with the pruned iff, deciding the 35-piece reset
is — in principle — a search over subsets of one concrete `Finset`. -/

/-- The finite universe of candidate phase states: every board inside the
10×20 playfield, paired with the full bag. -/
def phaseSpace : Finset GameState :=
  ((Finset.range 10 ×ˢ Finset.range 20).powerset).image
    (fun b => (⟨b, Bag.full⟩ : GameState))

/-- Any state satisfying a phase obstruction lies in `phaseSpace`. -/
theorem mem_phaseSpace_of_phaseObstruction {i : ℕ} {g : GameState}
    (h : PhaseObstruction i g) : g ∈ phaseSpace := by
  unfold PhaseObstruction at h
  obtain ⟨hwf, hbag, -, -, -, hfield⟩ := h
  unfold phaseSpace
  rw [Finset.mem_image]
  refine ⟨g.board, ?_, ?_⟩
  · rw [Finset.mem_powerset]
    intro p hp
    exact Finset.mem_product.mpr
      ⟨Finset.mem_range.mpr (hwf p hp), Finset.mem_range.mpr (hfield p hp)⟩
  · rw [← hbag]

/-- **The 35-piece reset is a finite search problem.** `Winning 35 init`
holds iff a 5-phase decomposition exists whose phase boards are subsets of
the explicit finite set `phaseSpace`. Forward: prune to obstruction-
satisfying boards and embed them in `phaseSpace`; reverse: forget the
bounds and compose the five legs. -/
theorem winning_init_iff_decomposition_in_phaseSpace :
    Winning blockLength GameState.init ↔
      ∃ P₁ P₂ P₃ P₄ : Finset GameState,
        P₁ ⊆ phaseSpace ∧ P₂ ⊆ phaseSpace ∧
        P₃ ⊆ phaseSpace ∧ P₄ ⊆ phaseSpace ∧
        BagWinning {GameState.init} P₁ ∧ BagWinning P₁ P₂ ∧
        BagWinning P₂ P₃ ∧ BagWinning P₃ P₄ ∧
        BagWinning P₄ {GameState.init} := by
  constructor
  · intro h
    obtain ⟨P₁, P₂, P₃, P₄, ho1, ho2, ho3, ho4, hb1, hb2, hb3, hb4, hb5⟩ :=
      winning_init_iff_pruned_phase_decomposition.mp h
    exact ⟨P₁, P₂, P₃, P₄,
      fun g hg => mem_phaseSpace_of_phaseObstruction (ho1 g hg),
      fun g hg => mem_phaseSpace_of_phaseObstruction (ho2 g hg),
      fun g hg => mem_phaseSpace_of_phaseObstruction (ho3 g hg),
      fun g hg => mem_phaseSpace_of_phaseObstruction (ho4 g hg),
      hb1, hb2, hb3, hb4, hb5⟩
  · rintro ⟨P₁, P₂, P₃, P₄, -, -, -, -, hb1, hb2, hb3, hb4, hb5⟩
    exact Winning_35_of_phase_decomposition P₁ P₂ P₃ P₄ hb1 hb2 hb3 hb4 hb5

/-! ## Quantitative width of the decomposition

`WinningTo.split` collects the states a strategy can be forced into as a
`biUnion` over the adversary's piece choices. Along a bag-aligned leg the
bag sizes are `7, 6, …, 1` (`cycleBagCard`), so the branching gives at most
`7! = 5040` reachable states per leg — one per bag ordering. The phase
boards of the completeness theorem can therefore be taken with
`|Pᵢ| ≤ 5040^i`, sharpening the `|phaseSpace| ≤ 2^200`-style bound on the
search width to one polynomial in the per-bag orderings. -/

/-- Maximum number of distinct states an adversary can force after the last
`m` draws of a leg: the product `cycleBagCard m ⋯ cycleBagCard 1` of the
bag sizes along the way. For a full leg, `legWidth 7 = 7! = 5040`. -/
def legWidth : ℕ → ℕ
  | 0 => 1
  | m + 1 => cycleBagCard (m + 1) * legWidth m

@[simp] theorem legWidth_seven : legWidth 7 = 5040 := by decide

/-- **Width-bounded strategy splitting.** Along a bag-aligned leg (bag size
pinned to `cycleBagCard`), the intermediate set of `WinningTo.split` can be
taken of size at most `legWidth m`, and every collected state arrives with
a freshly refilled bag. -/
theorem WinningTo.split_card {T : Finset GameState} :
    ∀ (m : ℕ) {n : ℕ} {g : GameState}, g.bag.card = cycleBagCard m →
      WinningTo T (m + n) g →
      ∃ mid : Finset GameState, mid.card ≤ legWidth m ∧
        (∀ g' ∈ mid, g'.bag = Bag.full) ∧
        WinningTo mid m g ∧ ∀ g' ∈ mid, WinningTo T n g' := by
  intro m
  induction m with
  | zero =>
      intro n g hbag h
      rw [Nat.zero_add] at h
      refine ⟨{g}, by simp [legWidth], ?_, ?_, ?_⟩
      · intro g' hg'
        rw [Finset.mem_singleton] at hg'
        subst hg'
        exact (Bag.card_eq_seven_iff_eq_full _).mp
          (by simpa [cycleBagCard] using hbag)
      · rw [winningTo_zero_iff]
        exact Finset.mem_singleton_self g
      · intro g' hg'
        rw [Finset.mem_singleton] at hg'
        subst hg'
        exact h
  | succ m ih =>
      intro n g hbag h
      rw [show m + 1 + n = (m + n) + 1 from by omega, winningTo_succ_iff] at h
      obtain ⟨hnl, hall⟩ := h
      have hchoice : ∀ p ∈ g.bag,
          ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
            ∃ mid : Finset GameState, mid.card ≤ legWidth m ∧
              (∀ g' ∈ mid, g'.bag = Bag.full) ∧
              WinningTo mid m (adversarialStep GameConfig.standard g p pl) ∧
                ∀ g' ∈ mid, WinningTo T n g' := by
        intro p hp
        obtain ⟨pl, hpl, hnext⟩ := hall p hp
        have hbag' : (adversarialStep GameConfig.standard g p pl).bag.card
            = cycleBagCard m := by
          rw [adversarialStep_bag]
          exact cycleBagCard_step hp m hbag
        obtain ⟨mid, hcard, hfull, hmid, hrest⟩ := ih hbag' hnext
        exact ⟨pl, hpl, mid, hcard, hfull, hmid, hrest⟩
      classical
      choose pl hpl mid hcard hfull hmid hrest using hchoice
      refine ⟨g.bag.attach.biUnion (fun q => mid q.1 q.2), ?_, ?_, ?_, ?_⟩
      · calc (g.bag.attach.biUnion (fun q => mid q.1 q.2)).card
            ≤ g.bag.attach.card * legWidth m :=
              Finset.card_biUnion_le_card_mul _ _ _ (fun q _ => hcard q.1 q.2)
          _ = cycleBagCard (m + 1) * legWidth m := by
              rw [Finset.card_attach, hbag]
          _ = legWidth (m + 1) := rfl
      · intro g' hg'
        obtain ⟨q, -, hq⟩ := Finset.mem_biUnion.mp hg'
        exact hfull q.1 q.2 g' hq
      · rw [winningTo_succ_iff]
        refine ⟨hnl, ?_⟩
        intro p hp
        refine ⟨pl p hp, hpl p hp, ?_⟩
        exact WinningTo.mono
          (Finset.subset_biUnion_of_mem _ (Finset.mem_attach _ ⟨p, hp⟩))
          (hmid p hp)
      · intro g' hg'
        obtain ⟨q, -, hq⟩ := Finset.mem_biUnion.mp hg'
        exact hrest q.1 q.2 g' hq

/-- Width-bounded bag peeling: from a set `P` of full-bag states all winning
in `7 + n` moves, the next phase set can be taken of size at most
`|P| · 5040`, with every member's bag refilled. -/
theorem BagWinning.peel_card {T P : Finset GameState} {n : ℕ}
    (hfull : ∀ g ∈ P, g.bag = Bag.full)
    (h : ∀ g ∈ P, WinningTo T (piecesPerBag + n) g) :
    ∃ P' : Finset GameState, P'.card ≤ P.card * 5040 ∧
      (∀ g ∈ P', g.bag = Bag.full) ∧
      BagWinning P P' ∧ ∀ g ∈ P', WinningTo T n g := by
  classical
  have hchoice : ∀ g ∈ P, ∃ mid : Finset GameState,
      mid.card ≤ legWidth 7 ∧
      (∀ g' ∈ mid, g'.bag = Bag.full) ∧
      WinningTo mid 7 g ∧ ∀ g' ∈ mid, WinningTo T n g' := by
    intro g hg
    have hbag : g.bag.card = cycleBagCard 7 := by
      simp [hfull g hg, cycleBagCard]
    exact WinningTo.split_card 7 hbag (h g hg)
  choose mid hcard hfullmid hmid hrest using hchoice
  refine ⟨P.attach.biUnion (fun q => mid q.1 q.2), ?_, ?_, ?_, ?_⟩
  · calc (P.attach.biUnion (fun q => mid q.1 q.2)).card
        ≤ P.attach.card * legWidth 7 :=
          Finset.card_biUnion_le_card_mul _ _ _ (fun q _ => hcard q.1 q.2)
      _ = P.card * 5040 := by rw [Finset.card_attach, legWidth_seven]
  · intro g hg
    obtain ⟨q, -, hq⟩ := Finset.mem_biUnion.mp hg
    exact hfullmid q.1 q.2 g hq
  · intro g hg
    exact WinningTo.mono
      (Finset.subset_biUnion_of_mem _ (Finset.mem_attach _ ⟨g, hg⟩))
      (hmid g hg)
  · intro g hg
    obtain ⟨q, -, hq⟩ := Finset.mem_biUnion.mp hg
    exact hrest q.1 q.2 g hq

/-- **Width-bounded completeness**: the 35-piece reset is winnable iff a
decomposition with explicitly size-bounded phase boards exists,
`|Pᵢ| ≤ 5040^i` — the adversary's `7!` bag orderings per leg bound how many
states a fixed strategy can be forced into. -/
theorem winning_init_iff_bounded_decomposition :
    Winning blockLength GameState.init ↔
      ∃ P₁ P₂ P₃ P₄ : Finset GameState,
        P₁.card ≤ 5040 ∧ P₂.card ≤ 5040 ^ 2 ∧
        P₃.card ≤ 5040 ^ 3 ∧ P₄.card ≤ 5040 ^ 4 ∧
        BagWinning {GameState.init} P₁ ∧ BagWinning P₁ P₂ ∧
        BagWinning P₂ P₃ ∧ BagWinning P₃ P₄ ∧
        BagWinning P₄ {GameState.init} := by
  constructor
  · intro h
    rw [winning_iff_winningTo_init] at h
    have h0 : ∀ g ∈ ({GameState.init} : Finset GameState),
        WinningTo {GameState.init} (piecesPerBag + 28) g := by
      intro g hg
      rw [Finset.mem_singleton] at hg
      subst hg
      exact h
    have hfull0 : ∀ g ∈ ({GameState.init} : Finset GameState),
        g.bag = Bag.full := by
      intro g hg
      rw [Finset.mem_singleton] at hg
      subst hg
      exact GameState.init_bag
    obtain ⟨P₁, hc1, hf1, hb1, h1⟩ := BagWinning.peel_card hfull0 h0
    obtain ⟨P₂, hc2, hf2, hb2, h2⟩ := BagWinning.peel_card (n := 21) hf1 h1
    obtain ⟨P₃, hc3, hf3, hb3, h3⟩ := BagWinning.peel_card (n := 14) hf2 h2
    obtain ⟨P₄, hc4, hf4, hb4, h4⟩ :=
      BagWinning.peel_card (n := piecesPerBag) hf3 h3
    have hc1' : P₁.card ≤ 5040 := by simpa using hc1
    have hpow2 : (5040 : ℕ) ^ 2 = 25401600 := by norm_num
    have hpow3 : (5040 : ℕ) ^ 3 = 128024064000 := by norm_num
    have hpow4 : (5040 : ℕ) ^ 4 = 645241282560000 := by norm_num
    exact ⟨P₁, P₂, P₃, P₄, hc1', by omega, by omega, by omega,
      hb1, hb2, hb3, hb4, h4⟩
  · rintro ⟨P₁, P₂, P₃, P₄, -, -, -, -, hb1, hb2, hb3, hb4, hb5⟩
    exact Winning_35_of_phase_decomposition P₁ P₂ P₃ P₄ hb1 hb2 hb3 hb4 hb5

/-- **The obstruction programme in one statement.** The 35-piece reset is
winnable iff there is a 5-phase decomposition whose phase boards
simultaneously (a) satisfy every derived obstruction — well-formed, full
bag, count residue/budget `28·i`, column budgets, in-field rows, hence lie
in the finite `phaseSpace` — and (b) have explicitly bounded size
`|Pᵢ| ≤ 5040^i`. Forward: peel the winning strategy bag by bag with width
tracking, then prune each phase board to its obstruction (filtering only
shrinks the card bounds). Reverse: forget all bounds and compose the legs. -/
theorem winning_init_iff_obstructed_bounded_decomposition :
    Winning blockLength GameState.init ↔
      ∃ P₁ P₂ P₃ P₄ : Finset GameState,
        (∀ g ∈ P₁, PhaseObstruction 1 g) ∧ (∀ g ∈ P₂, PhaseObstruction 2 g) ∧
        (∀ g ∈ P₃, PhaseObstruction 3 g) ∧ (∀ g ∈ P₄, PhaseObstruction 4 g) ∧
        P₁.card ≤ 5040 ∧ P₂.card ≤ 5040 ^ 2 ∧
        P₃.card ≤ 5040 ^ 3 ∧ P₄.card ≤ 5040 ^ 4 ∧
        BagWinning {GameState.init} P₁ ∧ BagWinning P₁ P₂ ∧
        BagWinning P₂ P₃ ∧ BagWinning P₃ P₄ ∧
        BagWinning P₄ {GameState.init} := by
  constructor
  · intro h
    obtain ⟨P₁, P₂, P₃, P₄, hc1, hc2, hc3, hc4, hb1, hb2, hb3, hb4, hb5⟩ :=
      winning_init_iff_bounded_decomposition.mp h
    obtain ⟨Q₁, Q₂, Q₃, Q₄, hs1, hs2, hs3, hs4, ho1, ho2, ho3, ho4,
      hq1, hq2, hq3, hq4, hq5⟩ :=
      phase_decomposition_pruned P₁ P₂ P₃ P₄ hb1 hb2 hb3 hb4 hb5
    exact ⟨Q₁, Q₂, Q₃, Q₄, ho1, ho2, ho3, ho4,
      (Finset.card_le_card hs1).trans hc1,
      (Finset.card_le_card hs2).trans hc2,
      (Finset.card_le_card hs3).trans hc3,
      (Finset.card_le_card hs4).trans hc4,
      hq1, hq2, hq3, hq4, hq5⟩
  · rintro ⟨P₁, P₂, P₃, P₄, -, -, -, -, -, -, -, -, hb1, hb2, hb3, hb4, hb5⟩
    exact Winning_35_of_phase_decomposition P₁ P₂ P₃ P₄ hb1 hb2 hb3 hb4 hb5

/-! ## Strategy certificates: linear-time verification of search output

`bagWinningBool` decides a leg by enumerating *every* placement at every
level — about `(7·17)·(6·17)⋯(1·17) ≈ 5040·17⁷` positions per source
state, far beyond `native_decide`. But the OR-branching over placements
exists only because the checker must *search*. An external search engine
(the Rust side) already knows the winning placement for every adversary
draw; if it emits that strategy as a **certificate tree**, checking
collapses to the AND-branching alone: at most `7! = 5040` paths per
source state, linear in the certificate.

`Cert` stores one chosen placement per piece per level; `checkCert`
replays the adversarial game along the tree; `WinningTo.of_checkCert`
proves any accepted certificate sound. The 35-reset then needs only five
certificate families (`Winning_35_of_certs`). Completeness is *not*
needed: a search-produced witness only ever has to be checked. -/

/-- A strategy certificate: at each level, the player's chosen placement
for every piece the adversary may draw, with one child certificate per
piece. Only pieces in the current bag are ever consulted. -/
inductive Cert : Type where
  | leaf : Cert
  | node (choose : Piece → Placement) (child : Piece → Cert) : Cert

/-- Replay a certificate: at depth `0` check target membership; otherwise
check the state is alive and, for every piece in the bag, that the chosen
placement is valid and the child certificate accepts the successor.
Cost is linear in the certificate — no search over placements. -/
def checkCert (T : Finset GameState) : ℕ → GameState → Cert → Bool
  | 0, g, _ => decide (g ∈ T)
  | _ + 1, _, .leaf => false
  | n + 1, g, .node choose child =>
      decide (¬ g.lost GameConfig.standard) &&
        decide (∀ p ∈ g.bag,
          choose p ∈ Placement.allValidFor GameConfig.standard p ∧
            checkCert T n (adversarialStep GameConfig.standard g p (choose p))
              (child p) = true)

/-- **Certificate soundness**: an accepted certificate proves the win. -/
theorem WinningTo.of_checkCert {T : Finset GameState} :
    ∀ {n : ℕ} {g : GameState} {c : Cert},
      checkCert T n g c = true → WinningTo T n g := by
  intro n
  induction n with
  | zero =>
      intro g c h
      rw [winningTo_zero_iff]
      cases c <;> simpa [checkCert] using h
  | succ n ih =>
      intro g c h
      cases c with
      | leaf => simp [checkCert] at h
      | node choose child =>
          simp only [checkCert, Bool.and_eq_true, decide_eq_true_eq] at h
          obtain ⟨hnl, hall⟩ := h
          rw [winningTo_succ_iff]
          exact ⟨hnl, fun p hp =>
            ⟨choose p, (hall p hp).1, ih (hall p hp).2⟩⟩

/-- Certify a whole `BagWinning` leg from one certificate per source
state. -/
theorem BagWinning.of_certs {P P' : Finset GameState}
    (cert : GameState → Cert)
    (h : ∀ g ∈ P, checkCert P' piecesPerBag g (cert g) = true) :
    BagWinning P P' :=
  fun g hg => WinningTo.of_checkCert (h g hg)

/-- **Certified 35-reset**: five certificate families — one per leg, one
certificate per source state — discharge the whole question. Each
`checkCert` call is linear in its certificate (≤ `7!` paths), so all five
hypotheses are realistic `decide`/`native_decide` obligations once a
search engine has produced the certificates. -/
theorem Winning_35_of_certs
    (P₁ P₂ P₃ P₄ : Finset GameState)
    (c₁ c₂ c₃ c₄ c₅ : GameState → Cert)
    (h1 : ∀ g ∈ ({GameState.init} : Finset GameState),
      checkCert P₁ piecesPerBag g (c₁ g) = true)
    (h2 : ∀ g ∈ P₁, checkCert P₂ piecesPerBag g (c₂ g) = true)
    (h3 : ∀ g ∈ P₂, checkCert P₃ piecesPerBag g (c₃ g) = true)
    (h4 : ∀ g ∈ P₃, checkCert P₄ piecesPerBag g (c₄ g) = true)
    (h5 : ∀ g ∈ P₄, checkCert {GameState.init} piecesPerBag g (c₅ g) = true) :
    Winning blockLength GameState.init :=
  Winning_35_of_phase_decomposition P₁ P₂ P₃ P₄
    (BagWinning.of_certs c₁ h1) (BagWinning.of_certs c₂ h2)
    (BagWinning.of_certs c₃ h3) (BagWinning.of_certs c₄ h4)
    (BagWinning.of_certs c₅ h5)

/-! ### Policy-table certificates

A `Cert` tree duplicates work: a board reached along two different draw orders
gets two identical subtrees. A **policy table** `π : GameState → Piece →
Placement` fixes this — it is a pure lookup table (exactly the Atlas artifact
the project is after), states revisited along different paths are shared
automatically, and `π` only needs to be *correct on the states the checker
actually visits*; everywhere else it can return a junk default. The Rust
search can emit it as a flat `(board, bag, piece) → placement` match table.

`checkPolicy` walks the same AND-tree as `checkCert` but reads its move from
`π` instead of a tree node, so checking stays linear in visited states with
no OR-enumeration over placements. -/

/-- Check a policy table: from `g`, surviving `n` adversarial draws into `T`,
playing `π g p` whenever the adversary hands us `p`. -/
def checkPolicy (T : Finset GameState) (π : GameState → Piece → Placement) :
    ℕ → GameState → Bool
  | 0, g => decide (g ∈ T)
  | n + 1, g =>
      decide (¬ g.lost GameConfig.standard) &&
        decide (∀ p ∈ g.bag,
          π g p ∈ Placement.allValidFor GameConfig.standard p ∧
            checkPolicy T π n
              (adversarialStep GameConfig.standard g p (π g p)) = true)

/-- Soundness: a passing policy check is a winning strategy. -/
theorem WinningTo.of_checkPolicy {T : Finset GameState}
    (π : GameState → Piece → Placement) :
    ∀ {n : ℕ} {g : GameState}, checkPolicy T π n g = true → WinningTo T n g := by
  intro n
  induction n with
  | zero =>
      intro g h
      rw [winningTo_zero_iff]
      simpa [checkPolicy] using h
  | succ n ih =>
      intro g h
      simp only [checkPolicy, Bool.and_eq_true, decide_eq_true_eq] at h
      obtain ⟨hnl, hall⟩ := h
      rw [winningTo_succ_iff]
      exact ⟨hnl, fun p hp => ⟨π g p, (hall p hp).1, ih (hall p hp).2⟩⟩

/-- A policy table that passes on every state of `P_in` certifies a leg. -/
theorem BagWinning.of_policy {P P' : Finset GameState}
    (π : GameState → Piece → Placement)
    (h : ∀ g ∈ P, checkPolicy P' π piecesPerBag g = true) :
    BagWinning P P' :=
  fun g hg => WinningTo.of_checkPolicy π (h g hg)

/-- **Five policy tables prove the 35-piece reset.** Each `πᵢ` is a flat
lookup table for one leg; the search side emits the tables, Lean only
re-checks them. -/
theorem Winning_35_of_policies
    (P₁ P₂ P₃ P₄ : Finset GameState)
    (π₁ π₂ π₃ π₄ π₅ : GameState → Piece → Placement)
    (h1 : checkPolicy P₁ π₁ piecesPerBag GameState.init = true)
    (h2 : ∀ g ∈ P₁, checkPolicy P₂ π₂ piecesPerBag g = true)
    (h3 : ∀ g ∈ P₂, checkPolicy P₃ π₃ piecesPerBag g = true)
    (h4 : ∀ g ∈ P₃, checkPolicy P₄ π₄ piecesPerBag g = true)
    (h5 : ∀ g ∈ P₄, checkPolicy {GameState.init} π₅ piecesPerBag g = true) :
    Winning blockLength GameState.init :=
  Winning_35_of_phase_decomposition P₁ P₂ P₃ P₄
    (fun g hg => by
      rw [Finset.mem_singleton] at hg
      subst hg
      exact WinningTo.of_checkPolicy π₁ h1)
    (BagWinning.of_policy π₂ h2) (BagWinning.of_policy π₃ h3)
    (BagWinning.of_policy π₄ h4) (BagWinning.of_policy π₅ h5)

/-- Uniform-`Q` form: one candidate set, three policy tables
(enter the loop, stay in the loop, exit to the empty board). -/
theorem Winning_35_of_uniform_policy (Q : Finset GameState)
    (πo πl πc : GameState → Piece → Placement)
    (hopen : checkPolicy Q πo piecesPerBag GameState.init = true)
    (hloop : ∀ g ∈ Q, checkPolicy Q πl piecesPerBag g = true)
    (hclose : ∀ g ∈ Q, checkPolicy {GameState.init} πc piecesPerBag g = true) :
    Winning blockLength GameState.init :=
  Winning_35_of_uniform_phase Q
    (fun g hg => by
      rw [Finset.mem_singleton] at hg
      subst hg
      exact WinningTo.of_checkPolicy πo hopen)
    (BagWinning.of_policy πl hloop) (BagWinning.of_policy πc hclose)

/-! ### A specific algorithm provably solving the 5-bag reset

Two routes to the final theorem, both reducing to boolean evaluation:

1. **Full search** (Rust side): emit explicit phase sets `P₁..P₄` and per-leg
   policy tables, check with `Winning_35_of_policies`.
2. **A specific algorithm**: implement the policies as concrete Lean-computable
   functions. Then the phase sets need not be guessed at all — they are the
   algorithm's own reachable **frontiers**, computed by `policyFrontier`.
   `algorithmSolves` bundles the entire obligation into one `Bool`, so a
   single `decide`/`native_decide` on a concrete algorithm proves
   `Winning blockLength GameState.init`.

Worst case a frontier multiplies by `7! = 5040` per leg, but an algorithm
*designed to funnel* (different draw orders converging to the same boards)
keeps frontiers small; the Rust side can measure frontier sizes cheaply
before attempting the Lean check. -/

/-- All states the adversary can reach in `n` draws when we answer piece `p`
at state `g` with `π g p`. -/
def policyFrontier (π : GameState → Piece → Placement) :
    ℕ → GameState → Finset GameState
  | 0, g => {g}
  | n + 1, g =>
      g.bag.biUnion fun p =>
        policyFrontier π n (adversarialStep GameConfig.standard g p (π g p))

/-- Phase-1 boards of an algorithm: the frontier of leg 1 from `init`. -/
def algoPhase1 (π₁ : GameState → Piece → Placement) : Finset GameState :=
  policyFrontier π₁ piecesPerBag GameState.init

/-- Phase-2 boards: leg-2 frontiers from every phase-1 board. -/
def algoPhase2 (π₁ π₂ : GameState → Piece → Placement) : Finset GameState :=
  (algoPhase1 π₁).biUnion (policyFrontier π₂ piecesPerBag)

/-- Phase-3 boards: leg-3 frontiers from every phase-2 board. -/
def algoPhase3 (π₁ π₂ π₃ : GameState → Piece → Placement) : Finset GameState :=
  (algoPhase2 π₁ π₂).biUnion (policyFrontier π₃ piecesPerBag)

/-- Phase-4 boards: leg-4 frontiers from every phase-3 board. -/
def algoPhase4 (π₁ π₂ π₃ π₄ : GameState → Piece → Placement) :
    Finset GameState :=
  (algoPhase3 π₁ π₂ π₃).biUnion (policyFrontier π₄ piecesPerBag)

/-- The whole 35-piece-reset obligation for a concrete algorithm, as one
boolean: each leg's policy survives with valid placements and lands in the
computed frontier, and leg 5 funnels every phase-4 board back to `init`. -/
def algorithmSolves (π₁ π₂ π₃ π₄ π₅ : GameState → Piece → Placement) : Bool :=
  checkPolicy (algoPhase1 π₁) π₁ piecesPerBag GameState.init &&
    (decide (∀ g ∈ algoPhase1 π₁,
        checkPolicy (algoPhase2 π₁ π₂) π₂ piecesPerBag g = true) &&
      (decide (∀ g ∈ algoPhase2 π₁ π₂,
          checkPolicy (algoPhase3 π₁ π₂ π₃) π₃ piecesPerBag g = true) &&
        (decide (∀ g ∈ algoPhase3 π₁ π₂ π₃,
            checkPolicy (algoPhase4 π₁ π₂ π₃ π₄) π₄ piecesPerBag g = true) &&
          decide (∀ g ∈ algoPhase4 π₁ π₂ π₃ π₄,
            checkPolicy {GameState.init} π₅ piecesPerBag g = true))))

/-- **A specific algorithm provably solves the 5-bag reset**: if the bundled
boolean evaluates to `true`, the algorithm is a constructive proof of
`Winning blockLength GameState.init`. -/
theorem Winning_35_of_algorithm
    (π₁ π₂ π₃ π₄ π₅ : GameState → Piece → Placement)
    (h : algorithmSolves π₁ π₂ π₃ π₄ π₅ = true) :
    Winning blockLength GameState.init := by
  unfold algorithmSolves at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  exact Winning_35_of_policies _ _ _ _ _ _ _ _ _ h1 h2 h3 h4 h5

/-! ### Executability smoke test

The proof layer above never *runs* the checker, so we keep one cheap
`native_decide` example as a regression test that the whole pipeline —
lost-detection, bag iteration, `allValidFor` membership, `adversarialStep`,
frontier membership — actually compiles and evaluates. A policy targeting
column 10 is out of bounds on the standard board, so the depth-1 check
fails at the very first validity test.

Measured costs (2026-06-11, M-series laptop): this depth-1 check is
instant; a full `algorithmSolves` on the same failing policy ≈ 100 s
(eval-dominated, not compile-dominated — one 13700-node leg tree plus the
5040-leaf frontier dedup; the Boolean `&&` short-circuits legs 2–5).
Strict evaluation forces each leg's frontier and full policy tree, so
certification cost scales as `Σ legs |Pᵢ₋₁| · 13700` node visits plus
frontier dedups — **cheap exactly when the algorithm funnels**. The Rust
search should therefore optimize frontier size first. A full
`algorithmSolves` example is deliberately not committed: it would triple
this file's rebuild time. -/

set_option linter.style.nativeDecide false in
example :
    checkPolicy {GameState.init} (fun _ p => ⟨p, 0, 10⟩) 1 GameState.init
      = false := by
  native_decide

/-! ### Policy tables: the search handoff format

The search side emits each leg's policy as a finite table of
`(state, piece) ↦ placement` rows — literally the Atlas rows for that leg.
`tablePolicy` turns such a table into the function the checkers consume.
States missing from the table get a default placement; that is harmless
because `checkPolicy` only queries the policy on states the leg actually
reaches, and certification fails if a reached state's row is missing or
wrong. A *verified* search artifact is therefore exactly: five tables
plus one `native_decide` on `algorithmSolves`. -/

/-- Interpret a finite lookup table as a policy (default: rotation 0,
column 0 for states the table omits). -/
def tablePolicy (rows : List ((GameState × Piece) × Placement)) :
    GameState → Piece → Placement := fun g p =>
  match rows.lookup (g, p) with
  | some pl => pl
  | none => ⟨p, 0, 0⟩

/-- **The artifact form of the result**: five leg tables passing the
bundled checker are a machine-checked proof of the 5-bag reset. -/
theorem Winning_35_of_tables
    (t₁ t₂ t₃ t₄ t₅ : List ((GameState × Piece) × Placement))
    (h : algorithmSolves (tablePolicy t₁) (tablePolicy t₂) (tablePolicy t₃)
      (tablePolicy t₄) (tablePolicy t₅) = true) :
    Winning blockLength GameState.init :=
  Winning_35_of_algorithm _ _ _ _ _ h

set_option linter.style.nativeDecide false in
example :
    checkPolicy (policyFrontier (tablePolicy []) 1 GameState.init)
      (tablePolicy []) 1 GameState.init = true := by
  native_decide

/-! ### The algorithm generates the Atlas

A passing algorithm doesn't just *prove* the reset — it **generates the
Atlas**. The 35-piece cycle returns exactly to `init`, so play under the
algorithm is periodic: the states visited during one cycle are *all* the
states that ever occur in infinite play. `policyStates` enumerates one leg's
tree (interior states included, not just the frontier), `algoAtlas` the whole
cycle's; `algoAtlas_card_le` bounds the Atlas by
`13700 · (1 + |P₁| + |P₂| + |P₃| + |P₄|)` entries, where `13700 =
1 + 7·(1 + 6·(1 + 5·(1 + 4·(1 + 3·(1 + 2·(1 + 1))))))` is one leg tree.
A funneling algorithm (frontiers near singletons) gives an Atlas of
~10⁴–10⁵ states for provably infinite Tetris. -/

/-- All states visited (root included) within `n` draws under policy `π`. -/
def policyStates (π : GameState → Piece → Placement) :
    ℕ → GameState → Finset GameState
  | 0, g => {g}
  | n + 1, g =>
      insert g (g.bag.biUnion fun p =>
        policyStates π n (adversarialStep GameConfig.standard g p (π g p)))

/-- Node bound for one leg tree: branching follows the bag cycle
`7, 6, 5, 4, 3, 2, 1`. -/
def nodeBound : ℕ → ℕ
  | 0 => 1
  | m + 1 => 1 + cycleBagCard (m + 1) * nodeBound m

@[simp] theorem nodeBound_seven : nodeBound 7 = 13700 := by decide

/-- One leg's tree holds at most `nodeBound m` states when the bag sits at
its cycle cardinality. -/
theorem policyStates_card_le (π : GameState → Piece → Placement) :
    ∀ (m : ℕ) {g : GameState}, g.bag.card = cycleBagCard m →
      (policyStates π m g).card ≤ nodeBound m := by
  intro m
  induction m with
  | zero =>
      intro g _
      simp [policyStates, nodeBound]
  | succ m ih =>
      intro g hbag
      have hcard : (policyStates π (m + 1) g).card ≤
          (g.bag.biUnion fun p => policyStates π m
            (adversarialStep GameConfig.standard g p (π g p))).card + 1 := by
        simp only [policyStates]
        exact Finset.card_insert_le _ _
      have hb := Finset.card_biUnion_le_card_mul g.bag
        (fun p => policyStates π m
          (adversarialStep GameConfig.standard g p (π g p)))
        (nodeBound m)
        (fun p hp => ih (by
          rw [adversarialStep_bag]
          exact cycleBagCard_step hp m hbag))
      have hnb : nodeBound (m + 1) = 1 + cycleBagCard (m + 1) * nodeBound m :=
        rfl
      rw [hbag] at hb
      omega

/-- Every state on a leg frontier carries a refilled full bag. -/
theorem policyFrontier_bag (π : GameState → Piece → Placement) :
    ∀ (m : ℕ) {g : GameState}, g.bag.card = cycleBagCard m →
      ∀ g' ∈ policyFrontier π m g, g'.bag = Bag.full := by
  intro m
  induction m with
  | zero =>
      intro g hbag g' hg'
      simp only [policyFrontier, Finset.mem_singleton] at hg'
      subst hg'
      exact (Bag.card_eq_seven_iff_eq_full _).mp
        (by simpa [cycleBagCard] using hbag)
  | succ m ih =>
      intro g hbag g' hg'
      simp only [policyFrontier, Finset.mem_biUnion] at hg'
      obtain ⟨p, hp, hg''⟩ := hg'
      exact ih (by rw [adversarialStep_bag]; exact cycleBagCard_step hp m hbag)
        g' hg''

/-- Phase-1 roots all carry a full bag. -/
theorem algoPhase1_bag (π₁ : GameState → Piece → Placement) :
    ∀ g ∈ algoPhase1 π₁, g.bag = Bag.full :=
  policyFrontier_bag π₁ piecesPerBag
    (by simp [piecesPerBag_eq, cycleBagCard])

/-- Frontiers launched from full-bag roots land on full-bag states. -/
theorem policyFrontier_biUnion_bag {P : Finset GameState}
    (π : GameState → Piece → Placement)
    (hP : ∀ g ∈ P, g.bag = Bag.full) :
    ∀ g ∈ P.biUnion (policyFrontier π piecesPerBag), g.bag = Bag.full := by
  intro g hg
  rw [Finset.mem_biUnion] at hg
  obtain ⟨r, hr, hg'⟩ := hg
  refine policyFrontier_bag π piecesPerBag ?_ g hg'
  have hfull := hP r hr
  simp [hfull, cycleBagCard]

/-- Phase-2 roots all carry a full bag. -/
theorem algoPhase2_bag (π₁ π₂ : GameState → Piece → Placement) :
    ∀ g ∈ algoPhase2 π₁ π₂, g.bag = Bag.full := by
  unfold algoPhase2
  exact policyFrontier_biUnion_bag π₂ (algoPhase1_bag π₁)

/-- Phase-3 roots all carry a full bag. -/
theorem algoPhase3_bag (π₁ π₂ π₃ : GameState → Piece → Placement) :
    ∀ g ∈ algoPhase3 π₁ π₂ π₃, g.bag = Bag.full := by
  unfold algoPhase3
  exact policyFrontier_biUnion_bag π₃ (algoPhase2_bag π₁ π₂)

/-- Phase-4 roots all carry a full bag. -/
theorem algoPhase4_bag (π₁ π₂ π₃ π₄ : GameState → Piece → Placement) :
    ∀ g ∈ algoPhase4 π₁ π₂ π₃ π₄, g.bag = Bag.full := by
  unfold algoPhase4
  exact policyFrontier_biUnion_bag π₄ (algoPhase3_bag π₁ π₂ π₃)

/-- One leg's trees over a set of full-bag roots: at most `13700` states
per root. -/
theorem policyStates_biUnion_card_le {P : Finset GameState}
    (π : GameState → Piece → Placement)
    (hP : ∀ g ∈ P, g.bag = Bag.full) :
    (P.biUnion (policyStates π piecesPerBag)).card ≤ P.card * 13700 := by
  refine Finset.card_biUnion_le_card_mul _ _ _ (fun r hr => ?_)
  have hfull := hP r hr
  have h7 : r.bag.card = cycleBagCard 7 := by simp [hfull, cycleBagCard]
  have := policyStates_card_le π 7 h7
  simpa using this

/-- The **Atlas generated by an algorithm**: every state the adversary can
ever force during the infinite repetition of the 35-piece cycle — the five
legs' trees, rooted at `init` and at the computed phase frontiers. -/
def algoAtlas (π₁ π₂ π₃ π₄ π₅ : GameState → Piece → Placement) :
    Finset GameState :=
  policyStates π₁ piecesPerBag GameState.init ∪
    ((algoPhase1 π₁).biUnion (policyStates π₂ piecesPerBag) ∪
      ((algoPhase2 π₁ π₂).biUnion (policyStates π₃ piecesPerBag) ∪
        ((algoPhase3 π₁ π₂ π₃).biUnion (policyStates π₄ piecesPerBag) ∪
          (algoPhase4 π₁ π₂ π₃ π₄).biUnion
            (policyStates π₅ piecesPerBag))))

/-- **Atlas size bound**: the algorithm's Atlas holds at most
`13700 · (1 + |P₁| + |P₂| + |P₃| + |P₄|)` states. -/
theorem algoAtlas_card_le (π₁ π₂ π₃ π₄ π₅ : GameState → Piece → Placement) :
    (algoAtlas π₁ π₂ π₃ π₄ π₅).card ≤
      13700 * (1 + (algoPhase1 π₁).card + (algoPhase2 π₁ π₂).card +
        (algoPhase3 π₁ π₂ π₃).card + (algoPhase4 π₁ π₂ π₃ π₄).card) := by
  have h0 : (policyStates π₁ piecesPerBag GameState.init).card ≤ 13700 := by
    have := policyStates_card_le π₁ 7 (g := GameState.init)
      (by simp [cycleBagCard])
    simpa using this
  have h1 := policyStates_biUnion_card_le π₂ (algoPhase1_bag π₁)
  have h2 := policyStates_biUnion_card_le π₃ (algoPhase2_bag π₁ π₂)
  have h3 := policyStates_biUnion_card_le π₄ (algoPhase3_bag π₁ π₂ π₃)
  have h4 := policyStates_biUnion_card_le π₅ (algoPhase4_bag π₁ π₂ π₃ π₄)
  have hu4 := Finset.card_union_le
    ((algoPhase3 π₁ π₂ π₃).biUnion (policyStates π₄ piecesPerBag))
    ((algoPhase4 π₁ π₂ π₃ π₄).biUnion (policyStates π₅ piecesPerBag))
  have hu3 := Finset.card_union_le
    ((algoPhase2 π₁ π₂).biUnion (policyStates π₃ piecesPerBag))
    ((algoPhase3 π₁ π₂ π₃).biUnion (policyStates π₄ piecesPerBag) ∪
      (algoPhase4 π₁ π₂ π₃ π₄).biUnion (policyStates π₅ piecesPerBag))
  have hu2 := Finset.card_union_le
    ((algoPhase1 π₁).biUnion (policyStates π₂ piecesPerBag))
    ((algoPhase2 π₁ π₂).biUnion (policyStates π₃ piecesPerBag) ∪
      ((algoPhase3 π₁ π₂ π₃).biUnion (policyStates π₄ piecesPerBag) ∪
        (algoPhase4 π₁ π₂ π₃ π₄).biUnion (policyStates π₅ piecesPerBag)))
  have hu1 := Finset.card_union_le
    (policyStates π₁ piecesPerBag GameState.init)
    ((algoPhase1 π₁).biUnion (policyStates π₂ piecesPerBag) ∪
      ((algoPhase2 π₁ π₂).biUnion (policyStates π₃ piecesPerBag) ∪
        ((algoPhase3 π₁ π₂ π₃).biUnion (policyStates π₄ piecesPerBag) ∪
          (algoPhase4 π₁ π₂ π₃ π₄).biUnion (policyStates π₅ piecesPerBag))))
  unfold algoAtlas
  omega

/-- Every root is in its own leg tree. -/
theorem mem_policyStates_self (π : GameState → Piece → Placement)
    (n : ℕ) (g : GameState) : g ∈ policyStates π n g := by
  cases n with
  | zero => simp [policyStates]
  | succ n => simp [policyStates]

/-- A leg's frontier is contained in its full tree. -/
theorem policyFrontier_subset_policyStates
    (π : GameState → Piece → Placement) :
    ∀ (n : ℕ) (g : GameState), policyFrontier π n g ⊆ policyStates π n g := by
  intro n
  induction n with
  | zero =>
      intro g
      simp only [policyFrontier, policyStates]
      exact Finset.Subset.refl _
  | succ n ih =>
      intro g g' hg'
      simp only [policyFrontier, Finset.mem_biUnion] at hg'
      obtain ⟨p, hp, hg'⟩ := hg'
      simp only [policyStates, Finset.mem_insert, Finset.mem_biUnion]
      exact Or.inr ⟨p, hp, ih _ hg'⟩

/-- A set of roots is contained in the union of its leg trees. -/
theorem subset_biUnion_policyStates {P : Finset GameState}
    (π : GameState → Piece → Placement) :
    P ⊆ P.biUnion (policyStates π piecesPerBag) := by
  intro g hg
  rw [Finset.mem_biUnion]
  exact ⟨g, hg, mem_policyStates_self π piecesPerBag g⟩

/-! **Atlas coverage**: the Atlas contains the initial state and all four
phase frontiers (alongside every interior state of every leg tree, by
construction). With `algoAtlas_card_le` this completes the artifact story:
a passing algorithm's Atlas is a *small, structurally complete* table. -/

/-- The whole leg-1 tree (rooted at `init`) is in the Atlas. -/
theorem policyStates_init_subset_algoAtlas
    (π₁ π₂ π₃ π₄ π₅ : GameState → Piece → Placement) :
    policyStates π₁ piecesPerBag GameState.init ⊆ algoAtlas π₁ π₂ π₃ π₄ π₅ := by
  unfold algoAtlas
  exact Finset.subset_union_left

/-- Singleton-subset form of `mem_policyStates_self`. Stated with a
*variable* root and depth: at concrete `n`/`g` a `∈ policyStates …` type
whnf-forces the set value (5040 stuck branches), while here the recursion
is stuck on `n` immediately. -/
theorem singleton_subset_policyStates (π : GameState → Piece → Placement)
    (n : ℕ) (g : GameState) : {g} ⊆ policyStates π n g :=
  Finset.singleton_subset_iff.mpr (mem_policyStates_self π n g)

/-- The Atlas covers the initial state. Stated as a singleton inclusion:
the membership form `GameState.init ∈ algoAtlas …` is unprovable in
practice — Finset membership whnf-forces the set value via `.val`, which
descends the 5040-branch leg tree; subset statements stop at the binder. -/
theorem init_singleton_subset_algoAtlas
    (π₁ π₂ π₃ π₄ π₅ : GameState → Piece → Placement) :
    {GameState.init} ⊆ algoAtlas π₁ π₂ π₃ π₄ π₅ :=
  Finset.Subset.trans
    (singleton_subset_policyStates π₁ piecesPerBag GameState.init)
    (policyStates_init_subset_algoAtlas π₁ π₂ π₃ π₄ π₅)

/-- The Atlas covers all phase-1 boards (as roots of leg-2 trees). -/
theorem algoPhase1_subset_algoAtlas
    (π₁ π₂ π₃ π₄ π₅ : GameState → Piece → Placement) :
    algoPhase1 π₁ ⊆ algoAtlas π₁ π₂ π₃ π₄ π₅ := by
  unfold algoAtlas
  exact Finset.Subset.trans (subset_biUnion_policyStates π₂)
    (Finset.Subset.trans Finset.subset_union_left Finset.subset_union_right)

/-- The Atlas covers all phase-2 boards (as roots of leg-3 trees). -/
theorem algoPhase2_subset_algoAtlas
    (π₁ π₂ π₃ π₄ π₅ : GameState → Piece → Placement) :
    algoPhase2 π₁ π₂ ⊆ algoAtlas π₁ π₂ π₃ π₄ π₅ := by
  unfold algoAtlas
  exact Finset.Subset.trans (subset_biUnion_policyStates π₃)
    (Finset.Subset.trans Finset.subset_union_left
      (Finset.Subset.trans Finset.subset_union_right
        Finset.subset_union_right))

/-- The Atlas covers all phase-3 boards (as roots of leg-4 trees). -/
theorem algoPhase3_subset_algoAtlas
    (π₁ π₂ π₃ π₄ π₅ : GameState → Piece → Placement) :
    algoPhase3 π₁ π₂ π₃ ⊆ algoAtlas π₁ π₂ π₃ π₄ π₅ := by
  unfold algoAtlas
  exact Finset.Subset.trans (subset_biUnion_policyStates π₄)
    (Finset.Subset.trans Finset.subset_union_left
      (Finset.Subset.trans Finset.subset_union_right
        (Finset.Subset.trans Finset.subset_union_right
          Finset.subset_union_right)))

/-- The Atlas covers all phase-4 boards (as roots of leg-5 trees). -/
theorem algoPhase4_subset_algoAtlas
    (π₁ π₂ π₃ π₄ π₅ : GameState → Piece → Placement) :
    algoPhase4 π₁ π₂ π₃ π₄ ⊆ algoAtlas π₁ π₂ π₃ π₄ π₅ := by
  unfold algoAtlas
  exact Finset.Subset.trans (subset_biUnion_policyStates π₅)
    (Finset.Subset.trans Finset.subset_union_right
      (Finset.Subset.trans Finset.subset_union_right
        (Finset.Subset.trans Finset.subset_union_right
          Finset.subset_union_right)))

/-! ### Search-space constraints from bag-cycle arithmetic

Necessary conditions on candidate decompositions, for free, before any
search: every leg *ends* on a refilled bag, so each phase set must actually
**contain** a full-bag state — nonemptiness for the **general** (non-uniform)
decomposition, complementing the WLOG pruning of `PhaseObstruction` and the
uniform-shape bound `uniform_Q_card_ge_four`. -/

/-- Along a leg the bag empties and refills on schedule, so a winning
strategy always **reaches the target at a full-bag state**. -/
theorem WinningTo.hits_full {T : Finset GameState} :
    ∀ (m : ℕ) {g : GameState}, g.bag.card = cycleBagCard m →
      WinningTo T m g → ∃ g' ∈ T, g'.bag = Bag.full := by
  intro m
  induction m with
  | zero =>
      intro g hbag h
      rw [winningTo_zero_iff] at h
      exact ⟨g, h, (Bag.card_eq_seven_iff_eq_full _).mp
        (by simpa [cycleBagCard] using hbag)⟩
  | succ m ih =>
      intro g hbag h
      rw [winningTo_succ_iff] at h
      have hne : g.bag.Nonempty := by
        rw [← Finset.card_pos, hbag]
        unfold cycleBagCard
        by_cases h7 : (m + 1) % 7 = 0
        · rw [if_pos h7]; omega
        · rw [if_neg h7]; omega
      obtain ⟨p, hp⟩ := hne
      obtain ⟨pl, hpl, hnext⟩ := h.2 p hp
      exact ih (by rw [adversarialStep_bag]; exact cycleBagCard_step hp m hbag)
        hnext

/-- **Every phase set of a 5-leg decomposition contains a full-bag state** —
in particular all four are nonempty. -/
theorem phase_sets_hit_full
    (P₁ P₂ P₃ P₄ : Finset GameState)
    (h1 : BagWinning {GameState.init} P₁) (h2 : BagWinning P₁ P₂)
    (h3 : BagWinning P₂ P₃) (h4 : BagWinning P₃ P₄) :
    (∃ g ∈ P₁, g.bag = Bag.full) ∧ (∃ g ∈ P₂, g.bag = Bag.full) ∧
      (∃ g ∈ P₃, g.bag = Bag.full) ∧ (∃ g ∈ P₄, g.bag = Bag.full) := by
  have hfull : ∀ {g : GameState}, g.bag = Bag.full →
      g.bag.card = cycleBagCard 7 := by
    intro g hg
    simp [hg, cycleBagCard]
  obtain ⟨g₁, hg₁P, hg₁f⟩ := WinningTo.hits_full 7 (hfull GameState.init_bag)
    (h1 _ (Finset.mem_singleton_self _))
  obtain ⟨g₂, hg₂P, hg₂f⟩ := WinningTo.hits_full 7 (hfull hg₁f) (h2 g₁ hg₁P)
  obtain ⟨g₃, hg₃P, hg₃f⟩ := WinningTo.hits_full 7 (hfull hg₂f) (h3 g₂ hg₂P)
  obtain ⟨g₄, hg₄P, hg₄f⟩ := WinningTo.hits_full 7 (hfull hg₃f) (h4 g₃ hg₃P)
  exact ⟨⟨g₁, hg₁P, hg₁f⟩, ⟨g₂, hg₂P, hg₂f⟩, ⟨g₃, hg₃P, hg₃f⟩,
    ⟨g₄, hg₄P, hg₄f⟩⟩

/-! ## Strategic next-step interface

Once you commit to specific candidate phase boards `P₁..P₄`, the obligation
reduces to 5 boolean checks of `bagWinningBool`:

```lean
example : Winning blockLength GameState.init :=
  Winning_35_of_bagWinningBool
    {candidatePhase1_bottomRight8}    -- P₁
    {candidatePhase2}                  -- P₂
    {candidatePhase3}                  -- P₃
    {candidatePhase4}                  -- P₄
    (by native_decide)  -- bag 1: {init} → P₁
    (by native_decide)  -- bag 2: P₁ → P₂
    (by native_decide)  -- bag 3: P₂ → P₃
    (by native_decide)  -- bag 4: P₃ → P₄
    (by native_decide)  -- bag 5: P₄ → {init}
```

Each `native_decide` call independently enumerates the bag-piece × placement
space. Realistically these will likely **fail** (return `false`) for the
canonical 10×20 board with hard-drop-only — but if any succeeds, that
specific transition has been certified, isolating where the obstruction lies.

The `Winning_35_of_uniform_phase` specialisation accepts a single `Q` if
the same residue board works at every mid-bag boundary (requires only 3
distinct `bagWinningBool` checks instead of 5). -/

/-! ## Impossibility of the empty-board reset

The remaining obligation of this experiment was the finite search for
`Winning blockLength GameState.init`. This section settles it **negatively**,
and for *every* positive horizon, not just `blockLength`:

* `applyStep_S_ne_empty` / `applyStep_Z_ne_empty`: under hard-drop gravity an
  S or Z piece can never finish a perfect clear. Each S/Z orientation has a
  cell `(a, e)` whose row crosses a column `j` where the drop profile has a
  bottom cell `(j, 0)` but a hole at `(j, e)`. A perfect clear forces row
  `dropOffset + e` to be full at column `col + j`; a stack cell there would
  have pushed the drop higher (`dropOffset ≥ colHeight`), and a piece cell
  there is impossible because the drop-profile translation is injective and
  `(j, e)` is not in the profile.
* `not_winning_of_S_reserved`: the adversary keeps an `S` in reserve for the
  final draw. Any state whose bag card matches the legal-draw schedule and
  still contains `S` near the end cannot win: the very last placement would
  have to perfect-clear with an `S` — impossible.
* `Winning.bag_card_schedule`: along *any* winning trace the bag card is
  forced backward from the terminal full bag, so `Winning m init` already
  implies `m ≡ 0 [MOD 7]` (`mod_seven_of_winning_init`).
* `not_winning_init`: combining the two, **no positive horizon `m` admits
  `Winning m GameState.init`**. In particular the 35-piece five-bag reset
  fails (`not_winning_blockLength_init`), every whole-bag horizon fails
  (`not_winning_bags_init`), and no 5-phase decomposition exists
  (`no_phase_decomposition`).

Scope: this refutes the *empty-board reset strategy* — the sufficient
condition explored by this file (returning to exactly `GameState.init`
after each block against an adversarial bag). It does **not** refute
`TetrisSolvable`: survivable play may still exist via strategies that never
return to the empty board, e.g. loops through a nonempty residue board. -/

/-- If clearing lines wipes the board completely, every filled cell sat in a
full row: a surviving (non-full-row) cell would survive into `clearLines`. -/
theorem isFull_of_clearLines_eq_empty {cfg : GameConfig} {b : Board}
    (h : Board.clearLines cfg b = ∅) :
    ∀ q ∈ b, Board.isFull cfg b q.2 := by
  intro q hq
  by_contra hnf
  have hmem : (q.1, q.2 - Board.clearedBelow cfg b q.2) ∈ Board.clearLines cfg b := by
    unfold Board.clearLines
    exact Finset.mem_image_of_mem _ (Finset.mem_filter.mpr ⟨hq, hnf⟩)
  rw [h] at hmem
  exact Finset.notMem_empty _ hmem

/-- Pair-literal form of `le_sup_sub`: instantiating at `(j, e)` with `H`
opaque reduces the pair projections once, so downstream unifications never
see them. (Unifying `(j, e).1` against `j` next to `colHeight` makes the
elaborator unfold `colHeight`'s filter/decide internals — a known blow-up.) -/
theorem le_sup_sub_pair (s : Finset Coord) (H : ℕ → ℕ) (col : ℕ) {j e : ℕ}
    (h : (j, e) ∈ s) :
    H (col + j) - e ≤ s.sup (fun c => H (col + c.1) - c.2) :=
  Placement.le_sup_sub s H col h

/-- Bottom-cell drop bound in `dropOffset` form, projection-free. -/
theorem le_dropOffset_pair (b : Board) (pl : Placement) {j e : ℕ}
    (h : (j, e) ∈ pl.shapeUp) :
    b.colHeight (pl.col + j) - e ≤ pl.dropOffset b := by
  rw [Placement.dropOffset_eq_sup]
  exact le_sup_sub_pair pl.shapeUp b.colHeight pl.col h

/-- **Overhang obstruction.** A placement whose drop profile contains a cell
`(a, e)` and a bottom cell `(j, 0)` but has a hole at `(j, e)` can never
finish a perfect clear: row `dropOffset + e` would have to be full at column
`col + j`, yet a stack cell there contradicts the drop offset and a piece
cell there contradicts the hole. -/
theorem applyStep_ne_empty_of_overhang {cfg : GameConfig} {b : Board}
    {pl : Placement} (hv : pl.Valid cfg) {a j e : ℕ}
    (ha : (a, e) ∈ pl.shapeUp) (hj : (j, 0) ∈ pl.shapeUp)
    (hhole : (j, e) ∉ pl.shapeUp) :
    Placement.applyStep cfg b pl ≠ ∅ := by
  intro hempty
  have hempty' : Board.clearLines cfg (pl.place b) = ∅ := hempty
  have hfull := isFull_of_clearLines_eq_empty hempty'
  -- The high cell of the dropped piece lies in the merged board.
  have hd : (pl.col + a, pl.dropOffset b + e) ∈ pl.dropped b := by
    rw [Placement.dropped_eq_image]
    exact Finset.mem_image.mpr ⟨(a, e), ha, rfl⟩
  have hhigh := Placement.dropped_subset_place b pl hd
  -- Its row is full, so column `col + j` is filled at that row too.
  have hjlt : pl.col + j < cfg.cols := hv (j, 0) hj
  have hcell : (pl.col + j, pl.dropOffset b + e) ∈ pl.place b :=
    hfull _ hhigh (pl.col + j) (Finset.mem_range.mpr hjlt)
  rw [Placement.place_eq_union_dropped] at hcell
  rcases Finset.mem_union.mp hcell with hstack | hpiece
  · -- Stack cell: the column height would exceed the drop offset.
    have hlt : pl.dropOffset b + e < b.colHeight (pl.col + j) :=
      Board.lt_colHeight hstack
    have hge := le_dropOffset_pair b pl hj
    generalize b.colHeight (pl.col + j) = H at hge hlt
    generalize pl.dropOffset b = D at hge hlt
    omega
  · -- Piece cell: the injective translation would force `(j, e) ∈ shapeUp`.
    rw [Placement.dropped_eq_image] at hpiece
    obtain ⟨⟨x, y⟩, hxy, heq⟩ := Finset.mem_image.mp hpiece
    simp only [Prod.mk.injEq] at heq
    obtain ⟨h1, h2⟩ := heq
    have hx : x = j := by omega
    have hy : y = e := by omega
    subst hx
    subst hy
    exact hhole hxy

/-- Every S orientation has an overhang in its drop profile: a high cell
`(a, e)`, a bottom cell `(j, 0)`, and a hole at `(j, e)`. (Stated per
rotation with closed goals so `decide` applies.) -/
theorem s_overhang_witness (r : Rotation) :
    ∃ a j e : ℕ, (a, e) ∈ Piece.S.shapeUp r ∧ (j, 0) ∈ Piece.S.shapeUp r ∧
      (j, e) ∉ Piece.S.shapeUp r := by
  fin_cases r
  · exact ⟨1, 0, 1, by decide, by decide, by decide⟩
  · exact ⟨0, 1, 2, by decide, by decide, by decide⟩
  · exact ⟨1, 0, 1, by decide, by decide, by decide⟩
  · exact ⟨0, 1, 2, by decide, by decide, by decide⟩

/-- Every Z orientation has an overhang (mirror of `s_overhang_witness`). -/
theorem z_overhang_witness (r : Rotation) :
    ∃ a j e : ℕ, (a, e) ∈ Piece.Z.shapeUp r ∧ (j, 0) ∈ Piece.Z.shapeUp r ∧
      (j, e) ∉ Piece.Z.shapeUp r := by
  fin_cases r
  · exact ⟨0, 2, 1, by decide, by decide, by decide⟩
  · exact ⟨1, 0, 2, by decide, by decide, by decide⟩
  · exact ⟨0, 2, 1, by decide, by decide, by decide⟩
  · exact ⟨1, 0, 2, by decide, by decide, by decide⟩

/-- **An S piece can never finish a perfect clear** under hard-drop gravity:
every S orientation has an overhang cell above a hole in its own profile. -/
theorem applyStep_S_ne_empty {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hS : pl.piece = Piece.S) :
    Placement.applyStep cfg b pl ≠ ∅ := by
  obtain ⟨a, j, e, ha, hj, hhole⟩ := s_overhang_witness pl.rot
  have hsu : pl.shapeUp = Piece.S.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hS]
  refine applyStep_ne_empty_of_overhang hv (a := a) (j := j) (e := e) ?_ ?_ ?_
  · rw [hsu]; exact ha
  · rw [hsu]; exact hj
  · rw [hsu]; exact hhole

/-- **A Z piece can never finish a perfect clear** (mirror of
`applyStep_S_ne_empty`). -/
theorem applyStep_Z_ne_empty {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hZ : pl.piece = Piece.Z) :
    Placement.applyStep cfg b pl ≠ ∅ := by
  obtain ⟨a, j, e, ha, hj, hhole⟩ := z_overhang_witness pl.rot
  have hsu : pl.shapeUp = Piece.Z.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hZ]
  refine applyStep_ne_empty_of_overhang hv (a := a) (j := j) (e := e) ?_ ?_ ?_
  · rw [hsu]; exact ha
  · rw [hsu]; exact hj
  · rw [hsu]; exact hhole

/-- **The "save an S for last" adversary strategy.** No state on the legal
bag-card schedule (`card = (m - 1) % 7 + 1`) that still holds an `S` whenever
the horizon fits inside the current bag (`m ≤ 7`) can win: the adversary
plays non-`S` pieces until the very last draw, which must then perfect-clear
with an `S` — impossible by `applyStep_S_ne_empty`. -/
theorem not_winning_of_S_reserved :
    ∀ m : ℕ, 1 ≤ m → ∀ g : GameState,
      g.bag.card = (m - 1) % 7 + 1 →
      (m ≤ 7 → Piece.S ∈ g.bag) →
      ¬ Winning m g := by
  intro m
  induction m with
  | zero => exact fun h => absurd h (by omega)
  | succ m ih =>
    intro _ g hcard hS hwin
    rw [winning_succ_iff] at hwin
    rcases Nat.eq_zero_or_pos m with rfl | hmpos
    · -- Final draw: the adversary demands the reserved `S`, which would have
      -- to perfect-clear the board to reach `GameState.init`.
      have hSmem : Piece.S ∈ g.bag := hS (by omega)
      obtain ⟨pl, hpl, hnext⟩ := hwin.2 Piece.S hSmem
      rw [winning_zero_iff] at hnext
      rw [Placement.mem_allValidFor] at hpl
      have hboard := congrArg GameState.board hnext
      rw [adversarialStep_board, Placement.eta_of_piece_eq hpl.1,
        GameState.init_board_eq_emptyset] at hboard
      exact applyStep_S_ne_empty hpl.2 hpl.1 hboard
    · have hcard' : g.bag.card = m % 7 + 1 := by simpa using hcard
      by_cases h7 : m % 7 = 0
      · -- Refill step: the singleton bag's draw refills to full (∋ S).
        have h1 : g.bag.card = 1 := by omega
        obtain ⟨p, hp⟩ := Finset.card_pos.mp (show 0 < g.bag.card by omega)
        obtain ⟨pl, -, hnext⟩ := hwin.2 p hp
        have hfull : g.bag.draw p = Bag.full :=
          (Bag.card_eq_seven_iff_eq_full _).mp (Bag.card_draw_of_singleton hp h1)
        refine ih hmpos _ ?_ ?_ hnext
        · rw [adversarialStep_bag, hfull, Bag.full_card]
          omega
        · intro _
          rw [adversarialStep_bag, hfull]
          exact Bag.mem_full Piece.S
      · have hge2 : 2 ≤ g.bag.card := by omega
        by_cases hle : m + 1 ≤ 7
        · -- Near the end: draw any piece other than the reserved `S`.
          have hSmem : Piece.S ∈ g.bag := hS hle
          obtain ⟨p, hp, hpne⟩ :=
            Finset.exists_mem_ne (show 1 < g.bag.card by omega) Piece.S
          obtain ⟨pl, -, hnext⟩ := hwin.2 p hp
          refine ih hmpos _ ?_ ?_ hnext
          · rw [adversarialStep_bag, Bag.card_draw_of_ge_two hp hge2, hcard']
            omega
          · intro _
            rw [adversarialStep_bag, Bag.draw_eq_erase_of_card_ge_two hp hge2]
            exact Finset.mem_erase.mpr ⟨hpne.symm, hSmem⟩
        · -- Far from the end: any draw keeps the schedule; the S side
          -- condition is vacuous (`m ≥ 8` here since `m % 7 ≠ 0`).
          obtain ⟨p, hp⟩ := Finset.card_pos.mp (show 0 < g.bag.card by omega)
          obtain ⟨pl, -, hnext⟩ := hwin.2 p hp
          refine ih hmpos _ ?_ ?_ hnext
          · rw [adversarialStep_bag, Bag.card_draw_of_ge_two hp hge2, hcard']
            omega
          · intro hm7
            exact absurd hm7 (by omega)

/-- **The bag-card schedule is forced.** Along any winning trace the bag card
is determined backward from the terminal full bag: a state with a nonempty
bag that wins in `m ≥ 1` moves has `card = (m - 1) % 7 + 1`. (The
nonemptiness hypothesis rules out the vacuously-winning empty-bag states,
which never occur on legal traces.) -/
theorem Winning.bag_card_schedule :
    ∀ m : ℕ, 1 ≤ m → ∀ g : GameState, g.bag.Nonempty → Winning m g →
      g.bag.card = (m - 1) % 7 + 1 := by
  intro m
  induction m with
  | zero => exact fun h => absurd h (by omega)
  | succ m ih =>
    intro _ g hne hwin
    rw [winning_succ_iff] at hwin
    obtain ⟨p, hp⟩ := hne
    obtain ⟨pl, -, hnext⟩ := hwin.2 p hp
    rcases Nat.eq_zero_or_pos m with rfl | hmpos
    · -- Last move: the post-draw bag is `init`'s full bag, so the draw was a
      -- refill from a singleton.
      rw [winning_zero_iff] at hnext
      have hbag := congrArg GameState.bag hnext
      rw [adversarialStep_bag, GameState.init_bag] at hbag
      have hcard7 : (g.bag.draw p).card = 7 := by rw [hbag, Bag.full_card]
      rcases Nat.lt_or_ge g.bag.card 2 with hlt | hge2
      · have hpos : 0 < g.bag.card := Finset.card_pos.mpr ⟨p, hp⟩
        omega
      · rw [Bag.card_draw_of_ge_two hp hge2] at hcard7
        have := Bag.card_le_seven g.bag
        omega
    · -- Interior move: pull the schedule back through the draw.
      have hnext_ne : (adversarialStep GameConfig.standard g p pl).bag.Nonempty := by
        rw [adversarialStep_bag]
        exact Bag.draw_nonempty g.bag p
      have hnc := ih hmpos _ hnext_ne hnext
      rw [adversarialStep_bag] at hnc
      rcases Nat.lt_or_ge g.bag.card 2 with hlt | hge2
      · have hpos : 0 < g.bag.card := Finset.card_pos.mpr ⟨p, hp⟩
        have h1 : g.bag.card = 1 := by omega
        rw [Bag.card_draw_of_singleton hp h1] at hnc
        omega
      · rw [Bag.card_draw_of_ge_two hp hge2] at hnc
        have hle7 := Bag.card_le_seven g.bag
        omega

/-- A winning horizon from the initial state must be a whole number of bags:
the full 7-piece bag pins `(m - 1) % 7 + 1 = 7`. -/
theorem mod_seven_of_winning_init {m : ℕ} (hm : 1 ≤ m)
    (h : Winning m GameState.init) : m % 7 = 0 := by
  have hcard :=
    Winning.bag_card_schedule m hm GameState.init GameState.init_bag_nonempty h
  rw [GameState.init_bag_card] at hcard
  omega

/-- The empty-board reset fails at every whole-bag horizon: the initial bag
is full (so it contains `S`) and matches the bag-card schedule, hence the
"save an S for last" adversary wins. -/
theorem not_winning_init_of_mod_seven {m : ℕ} (hm : 1 ≤ m) (h7 : m % 7 = 0) :
    ¬ Winning m GameState.init := by
  refine not_winning_of_S_reserved m hm GameState.init ?_ ?_
  · rw [GameState.init_bag_card]
    omega
  · intro _
    rw [GameState.init_bag]
    exact Bag.mem_full Piece.S

/-- **Total settlement of the empty-board reset: it is impossible at every
positive horizon.** Either `m` is not a multiple of 7 — then the forced
bag-card schedule (`mod_seven_of_winning_init`) already refutes it — or it
is, and the adversary saves an `S` for the final perfect clear, which no S
placement can complete (`not_winning_init_of_mod_seven`). -/
theorem not_winning_init (m : ℕ) (hm : 1 ≤ m) :
    ¬ Winning m GameState.init :=
  fun h => not_winning_init_of_mod_seven hm (mod_seven_of_winning_init hm h) h

/-- The 35-piece five-bag reset — the headline question of this experiment —
is unwinnable. -/
theorem not_winning_blockLength_init : ¬ Winning blockLength GameState.init :=
  not_winning_init blockLength (by decide)

/-- No whole number of bags admits an empty-board reset. -/
theorem not_winning_bags_init (k : ℕ) (hk : 1 ≤ k) :
    ¬ Winning (7 * k) GameState.init :=
  not_winning_init_of_mod_seven (by omega) (by omega)

/-- **No 5-phase decomposition exists**: the search space that
`winning_init_iff_phase_decomposition` reduced `Winning 35 init` to is
empty. The phase-decomposition program of this file is thereby closed. -/
theorem no_phase_decomposition :
    ¬ ∃ P₁ P₂ P₃ P₄ : Finset GameState,
        BagWinning {GameState.init} P₁ ∧ BagWinning P₁ P₂ ∧
        BagWinning P₂ P₃ ∧ BagWinning P₃ P₄ ∧
        BagWinning P₄ {GameState.init} :=
  fun h => not_winning_blockLength_init (winning_init_iff_phase_decomposition.mpr h)



/-! ## M3 composition — anchored loops at a nonempty reset board

The empty-board reset is impossible (`not_winning_init`), but the loop
machinery survives at any *nonempty* anchor. This section proves the
composition: a **recurrent family** `T` (every state in `T` is forced back
into `T` in ≥ 1 steps) is contained in the `safe` gfp, and any state with a
forced reach into `T` is safe. Specialised to `T = {g₀}` and
`g = GameState.init`, this is exactly milestone M3:

`WinningTo {g₀} j init`  +  `WinningTo {g₀} n g₀ (n ≥ 1)`  ⇒  `TetrisSolvable`. -/

/-- **Coinduction into the safe set.** If every state of `T` is forced back
into `T` in at least one step (a *recurrent family*), then any state with a
forced trajectory into `T` is in `safe GameConfig.standard`.

The post-fixpoint witness is `S = {g | ∃ k, WinningTo T k g}`: a state of
`S` at depth `k+1` steps into `S` at depth `k`; a state of `S` at depth `0`
lies in `T`, and recurrence restarts it at depth ≥ 1. -/
theorem mem_safe_of_winningTo_recurrent {T : Finset GameState}
    (hrec : ∀ t ∈ T, ∃ k, 1 ≤ k ∧ WinningTo T k t)
    {j : ℕ} {g : GameState} (hreach : WinningTo T j g) :
    g ∈ safe GameConfig.standard := by
  suffices hsub : {g : GameState | ∃ k, WinningTo T k g} ⊆
      safe GameConfig.standard from hsub ⟨j, hreach⟩
  have hpost : {g : GameState | ∃ k, WinningTo T k g} ≤
      safeOp GameConfig.standard {g : GameState | ∃ k, WinningTo T k g} := by
    rintro g ⟨k, hwin⟩
    obtain ⟨m, hwin⟩ : ∃ m, WinningTo T (m + 1) g := by
      cases k with
      | zero =>
          obtain ⟨k', hk1, hw'⟩ := hrec g hwin
          exact ⟨k' - 1, by rwa [Nat.sub_add_cancel hk1]⟩
      | succ m => exact ⟨m, hwin⟩
    refine ⟨hwin.1, ?_⟩
    intro p hp
    obtain ⟨pl, hpl, hnext⟩ := hwin.2 p hp
    obtain ⟨hpiece, hvalid⟩ := (Placement.mem_allValidFor _ _ _).mp hpl
    exact ⟨pl, hpiece, hvalid, m, hnext⟩
  exact OrderHom.le_gfp (safeOp GameConfig.standard) hpost

/-- A recurrent family is itself contained in the safe set. -/
theorem subset_safe_of_winningTo_recurrent {T : Finset GameState}
    (hrec : ∀ t ∈ T, ∃ k, 1 ≤ k ∧ WinningTo T k t) :
    ∀ t ∈ T, t ∈ safe GameConfig.standard := fun t ht =>
  mem_safe_of_winningTo_recurrent hrec (winningTo_zero_iff T t |>.mpr ht)

/-- **Recurrent family + forced reach from init ⇒ Tetris is solvable.** -/
theorem tetrisSolvable_of_winningTo_recurrent {T : Finset GameState}
    (hrec : ∀ t ∈ T, ∃ k, 1 ≤ k ∧ WinningTo T k t)
    {j : ℕ} (hreach : WinningTo T j GameState.init) :
    TetrisSolvable :=
  safe_extract (mem_safe_of_winningTo_recurrent hrec hreach)

/-- **Validity-strengthened recurrent-family certificate.** The
`TetrisSolvableValid` companion of `tetrisSolvable_of_winningTo_recurrent`: a
recurrent family `T` plus a forced reach from `init` yields not merely
`TetrisSolvable` but the project's ultimate goal `TetrisSolvableValid` (a
*valid* solver that provably never tops out). Routes the same
`mem_safe_of_winningTo_recurrent` safety witness through
`tetrisSolvableValid_iff_init_safe` (which packages `safeSolver`'s validity via
`safeSolver_validSolver`, sound because `GameConfig.standard.cols = 10 ≥ 4`)
instead of the plain `safe_extract`. This is the M3-shaped headline: any
bag-aligned closed cycle reachable from the empty board discharges the
affirmative atlas claim at full validity strength. -/
theorem tetrisSolvableValid_of_winningTo_recurrent {T : Finset GameState}
    (hrec : ∀ t ∈ T, ∃ k, 1 ≤ k ∧ WinningTo T k t)
    {j : ℕ} (hreach : WinningTo T j GameState.init) :
    TetrisSolvableValid :=
  tetrisSolvableValid_iff_init_safe.mpr (mem_safe_of_winningTo_recurrent hrec hreach)

/-- **A bag-closed set lies entirely inside `safe`.** If `T` is self-recurrent
(`BagWinning T T`), then every `t ∈ T` is in `safe GameConfig.standard`. Each
state bag-wins back into `T` in `piecesPerBag = 7 ≥ 1` steps, which is exactly the
recurrence premise of `subset_safe_of_winningTo_recurrent`. This is the direct
bridge from the single-bag `BagWinning` algebra to the canonical greatest-fixpoint
safe region: a bag-closed certificate is a *safe* certificate, so (combined with a
reach into `T`) `init ∈ safe` and hence `TetrisSolvableValid` follow. -/
theorem BagWinning.subset_safe {T : Finset GameState} (hself : BagWinning T T) :
    ∀ t ∈ T, t ∈ safe GameConfig.standard :=
  subset_safe_of_winningTo_recurrent
    (fun t ht => ⟨piecesPerBag, by decide, hself t ht⟩)

/-- **The M3 anchor theorem.** A single anchor state `g₀` with

* a forced reach `init → g₀` in `j` steps, and
* a forced loop `g₀ → g₀` in `n ≥ 1` steps

yields `TetrisSolvable`. This is the post-impossibility-pivot search target:
find a nonempty board `B₀` (full bag) such that both `WinningTo {⟨B₀,full⟩} j init`
and `WinningTo {⟨B₀,full⟩} n ⟨B₀,full⟩` hold. -/
theorem tetrisSolvable_of_anchor {g₀ : GameState} {j n : ℕ} (hn : 1 ≤ n)
    (hreach : WinningTo {g₀} j GameState.init)
    (hloop : WinningTo {g₀} n g₀) :
    TetrisSolvable := by
  refine tetrisSolvable_of_winningTo_recurrent ?_ hreach
  intro t ht
  rw [Finset.mem_singleton] at ht
  subst ht
  exact ⟨n, hn, hloop⟩

/-- **Executable M3 anchor certificate.** Both obligations checked by the
boolean reflection `winningToBool` — the shape a Rust-discovered certificate
will be verified in (`decide` / `native_decide` at explicit `g₀`, `j`, `n`). -/
theorem tetrisSolvable_of_anchorBool {g₀ : GameState} {j n : ℕ} (hn : 1 ≤ n)
    (hreach : winningToBool {g₀} j GameState.init = true)
    (hloop : winningToBool {g₀} n g₀ = true) :
    TetrisSolvable :=
  tetrisSolvable_of_anchor hn
    ((winningToBool_eq_true_iff _ _ _).mp hreach)
    ((winningToBool_eq_true_iff _ _ _).mp hloop)

/-- **Validity-strengthened M3 anchor theorem.** The `TetrisSolvableValid`
companion of `tetrisSolvable_of_anchor`: a single anchor state `g₀` with a
forced reach `init → g₀` and a forced self-loop `g₀ → g₀` (in `n ≥ 1` steps)
yields the project's ultimate goal `TetrisSolvableValid`, not merely
`TetrisSolvable`. Identical to the non-valid form except it routes the
recurrence through `tetrisSolvableValid_of_winningTo_recurrent`. This is the
headline the M3 singleton search should target: a nonempty full-bag board `B₀`
with both obligations holding gives a valid never-topping-out solver. -/
theorem tetrisSolvableValid_of_anchor {g₀ : GameState} {j n : ℕ} (hn : 1 ≤ n)
    (hreach : WinningTo {g₀} j GameState.init)
    (hloop : WinningTo {g₀} n g₀) :
    TetrisSolvableValid := by
  refine tetrisSolvableValid_of_winningTo_recurrent ?_ hreach
  intro t ht
  rw [Finset.mem_singleton] at ht
  subst ht
  exact ⟨n, hn, hloop⟩

/-- **Validity-strengthened executable M3 anchor certificate.** The
`TetrisSolvableValid` companion of `tetrisSolvable_of_anchorBool`: both anchor
obligations are `native_decide`-able boolean checks (`winningToBool` at explicit
`g₀`, `j`, `n`) and discharge the project's ultimate goal `TetrisSolvableValid`
directly. This is the exact verification shape a Rust-discovered singleton anchor
is checked in, now delivering a valid never-topping-out solver with no further
proof work. -/
theorem tetrisSolvableValid_of_anchorBool {g₀ : GameState} {j n : ℕ} (hn : 1 ≤ n)
    (hreach : winningToBool {g₀} j GameState.init = true)
    (hloop : winningToBool {g₀} n g₀ = true) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_anchor hn
    ((winningToBool_eq_true_iff _ _ _).mp hreach)
    ((winningToBool_eq_true_iff _ _ _).mp hloop)

/-- Anchored loops decompose bag-by-bag: chained `BagWinning` obligations
through intermediate phase sets give the loop premise of
`tetrisSolvable_of_anchor` without ever materialising the depth-`7k` tree. -/
theorem anchor_loop_of_bagWinning_chain {g₀ : GameState}
    (P₁ P₂ P₃ P₄ : Finset GameState)
    (hbag1 : BagWinning {g₀} P₁)
    (hbag2 : BagWinning P₁ P₂)
    (hbag3 : BagWinning P₂ P₃)
    (hbag4 : BagWinning P₃ P₄)
    (hbag5 : BagWinning P₄ {g₀}) :
    WinningTo {g₀} (5 * piecesPerBag) g₀ := by
  have h1 : WinningTo P₁ piecesPerBag g₀ :=
    hbag1 g₀ (Finset.mem_singleton.mpr rfl)
  have h12 : WinningTo P₂ (piecesPerBag + piecesPerBag) g₀ := h1.compose hbag2
  have h123 : WinningTo P₃ (piecesPerBag + piecesPerBag + piecesPerBag) g₀ :=
    h12.compose hbag3
  have h1234 : WinningTo P₄
      (piecesPerBag + piecesPerBag + piecesPerBag + piecesPerBag) g₀ :=
    h123.compose hbag4
  have h12345 : WinningTo {g₀}
      (piecesPerBag + piecesPerBag + piecesPerBag + piecesPerBag + piecesPerBag)
      g₀ := h1234.compose hbag5
  have : piecesPerBag + piecesPerBag + piecesPerBag + piecesPerBag +
      piecesPerBag = 5 * piecesPerBag := by omega
  rwa [this] at h12345

/-- **Reach-side bag-by-bag chain.** The init→T counterpart of
`anchor_loop_of_bagWinning_chain`: five consecutive single-bag forcings starting
from the singleton initial state compose into a `WinningTo P₅` over five bags.
Unlike the loop lemma the final phase `P₅` is arbitrary (it need not return to
the start), so this packages the *reachability bridge* from `init` into any
bag-closed target reached after a fixed five-bag block. -/
theorem reach_via_bagWinning_chain
    (P₁ P₂ P₃ P₄ P₅ : Finset GameState)
    (hbag1 : BagWinning {GameState.init} P₁)
    (hbag2 : BagWinning P₁ P₂)
    (hbag3 : BagWinning P₂ P₃)
    (hbag4 : BagWinning P₃ P₄)
    (hbag5 : BagWinning P₄ P₅) :
    WinningTo P₅ (5 * piecesPerBag) GameState.init := by
  have h1 : WinningTo P₁ piecesPerBag GameState.init :=
    hbag1 GameState.init (Finset.mem_singleton.mpr rfl)
  have h12 : WinningTo P₂ (piecesPerBag + piecesPerBag) GameState.init :=
    h1.compose hbag2
  have h123 : WinningTo P₃ (piecesPerBag + piecesPerBag + piecesPerBag)
      GameState.init := h12.compose hbag3
  have h1234 : WinningTo P₄
      (piecesPerBag + piecesPerBag + piecesPerBag + piecesPerBag)
      GameState.init := h123.compose hbag4
  have h12345 : WinningTo P₅
      (piecesPerBag + piecesPerBag + piecesPerBag + piecesPerBag + piecesPerBag)
      GameState.init := h1234.compose hbag5
  have : piecesPerBag + piecesPerBag + piecesPerBag + piecesPerBag +
      piecesPerBag = 5 * piecesPerBag := by omega
  rwa [this] at h12345

/-- **End-to-end ten-bag chain certificate.** Combines the reach-side chain
(`reach_via_bagWinning_chain`: `init` into the anchor singleton across five bags)
with the loop-side chain (`anchor_loop_of_bagWinning_chain`: the anchor back to
itself across five bags) and routes both through `tetrisSolvableValid_of_anchor`.
Ten raw single-bag `BagWinning` obligations — no residue reasoning, no explicit
depth-`7k` tree — discharge the project's ultimate goal `TetrisSolvableValid`.
This is the shape a Rust-discovered five-phase reach + five-phase loop is checked
in, delivering a valid never-topping-out solver directly. -/
theorem tetrisSolvableValid_of_bagWinning_reach_loop_chain {g₀ : GameState}
    (Q₁ Q₂ Q₃ Q₄ P₁ P₂ P₃ P₄ : Finset GameState)
    (hr1 : BagWinning {GameState.init} Q₁)
    (hr2 : BagWinning Q₁ Q₂)
    (hr3 : BagWinning Q₂ Q₃)
    (hr4 : BagWinning Q₃ Q₄)
    (hr5 : BagWinning Q₄ {g₀})
    (hl1 : BagWinning {g₀} P₁)
    (hl2 : BagWinning P₁ P₂)
    (hl3 : BagWinning P₂ P₃)
    (hl4 : BagWinning P₃ P₄)
    (hl5 : BagWinning P₄ {g₀}) :
    TetrisSolvableValid := by
  have hreach : WinningTo {g₀} (5 * piecesPerBag) GameState.init :=
    reach_via_bagWinning_chain Q₁ Q₂ Q₃ Q₄ {g₀} hr1 hr2 hr3 hr4 hr5
  have hloop : WinningTo {g₀} (5 * piecesPerBag) g₀ :=
    anchor_loop_of_bagWinning_chain P₁ P₂ P₃ P₄ hl1 hl2 hl3 hl4 hl5
  exact tetrisSolvableValid_of_anchor (by decide) hreach hloop

/-- **Boolean/`native_decide` form of the end-to-end ten-bag certificate.** Each
of the ten single-bag obligations is a decidable `bagWinningBool` check at
explicit phase sets; reflecting all ten through `bagWinningBool_eq_true_iff` feeds
`tetrisSolvableValid_of_bagWinning_reach_loop_chain`. A Rust-discovered five-phase
reach + five-phase loop is thus checked entirely by `native_decide`-able boolean
equalities, discharging `TetrisSolvableValid` with no further proof work. -/
theorem tetrisSolvableValid_of_bagWinning_reach_loop_chainBool {g₀ : GameState}
    (Q₁ Q₂ Q₃ Q₄ P₁ P₂ P₃ P₄ : Finset GameState)
    (hr1 : bagWinningBool {GameState.init} Q₁ = true)
    (hr2 : bagWinningBool Q₁ Q₂ = true)
    (hr3 : bagWinningBool Q₂ Q₃ = true)
    (hr4 : bagWinningBool Q₃ Q₄ = true)
    (hr5 : bagWinningBool Q₄ {g₀} = true)
    (hl1 : bagWinningBool {g₀} P₁ = true)
    (hl2 : bagWinningBool P₁ P₂ = true)
    (hl3 : bagWinningBool P₂ P₃ = true)
    (hl4 : bagWinningBool P₃ P₄ = true)
    (hl5 : bagWinningBool P₄ {g₀} = true) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_bagWinning_reach_loop_chain Q₁ Q₂ Q₃ Q₄ P₁ P₂ P₃ P₄
    ((bagWinningBool_eq_true_iff _ _).mp hr1)
    ((bagWinningBool_eq_true_iff _ _).mp hr2)
    ((bagWinningBool_eq_true_iff _ _).mp hr3)
    ((bagWinningBool_eq_true_iff _ _).mp hr4)
    ((bagWinningBool_eq_true_iff _ _).mp hr5)
    ((bagWinningBool_eq_true_iff _ _).mp hl1)
    ((bagWinningBool_eq_true_iff _ _).mp hl2)
    ((bagWinningBool_eq_true_iff _ _).mp hl3)
    ((bagWinningBool_eq_true_iff _ _).mp hl4)
    ((bagWinningBool_eq_true_iff _ _).mp hl5)


/-- **The five-phase cyclic family theorem.** Five phase sets
`P₀, …, P₄`, each forcing into the next within a single bag (cyclically,
`P₄` back into `P₀`), plus a forced reach from `init` into `P₀`, yield
`TetrisSolvable`.

Unlike the singleton anchor, no trajectory ever has to *funnel* to one
exact board — each bag only has to land somewhere in the next phase set.
Since cell-count residues mod 10 advance by 8 per bag, period 5 is the
universal shape: any bag-aligned recurrent family refines into five
residue classes, so this theorem subsumes loops of any length `5k` bags. -/
theorem tetrisSolvable_of_five_cyclic_phases
    {P₀ P₁ P₂ P₃ P₄ : Finset GameState} {j : ℕ}
    (h01 : BagWinning P₀ P₁) (h12 : BagWinning P₁ P₂)
    (h23 : BagWinning P₂ P₃) (h34 : BagWinning P₃ P₄)
    (h40 : BagWinning P₄ P₀)
    (hreach : WinningTo P₀ j GameState.init) :
    TetrisSolvable := by
  classical
  set T : Finset GameState := P₀ ∪ P₁ ∪ P₂ ∪ P₃ ∪ P₄ with hT
  have hsub₀ : P₀ ⊆ T := by intro x hx; simp [hT, hx]
  have hsub₁ : P₁ ⊆ T := by intro x hx; simp [hT, hx]
  have hsub₂ : P₂ ⊆ T := by intro x hx; simp [hT, hx]
  have hsub₃ : P₃ ⊆ T := by intro x hx; simp [hT, hx]
  have hsub₄ : P₄ ⊆ T := by intro x hx; simp [hT, hx]
  refine tetrisSolvable_of_winningTo_recurrent (T := T) ?_ (hreach.mono hsub₀)
  intro t ht
  have h7 : 1 ≤ piecesPerBag := by decide
  have hcases : t ∈ P₀ ∨ t ∈ P₁ ∨ t ∈ P₂ ∨ t ∈ P₃ ∨ t ∈ P₄ := by
    simpa [hT, Finset.mem_union, or_assoc] using ht
  rcases hcases with h | h | h | h | h
  · exact ⟨piecesPerBag, h7, (h01 t h).mono hsub₁⟩
  · exact ⟨piecesPerBag, h7, (h12 t h).mono hsub₂⟩
  · exact ⟨piecesPerBag, h7, (h23 t h).mono hsub₃⟩
  · exact ⟨piecesPerBag, h7, (h34 t h).mono hsub₄⟩
  · exact ⟨piecesPerBag, h7, (h40 t h).mono hsub₀⟩

/-- **Executable five-phase cyclic certificate.** All six obligations are
boolean checks on explicit finite data — the exact shape a Rust-discovered
certificate is verified in:

* five `bagWinningBool` cyclic-hop checks (depth-7 trees over each phase), and
* one `winningToBool` reach check from `init` (depth `j`).

The reach obligation can itself be split bag-by-bag with `bagWinningBool`
chains through intermediate sets and composed via `WinningTo.compose`. -/
theorem tetrisSolvable_of_five_cyclic_phasesBool
    {P₀ P₁ P₂ P₃ P₄ : Finset GameState} {j : ℕ}
    (h01 : bagWinningBool P₀ P₁ = true) (h12 : bagWinningBool P₁ P₂ = true)
    (h23 : bagWinningBool P₂ P₃ = true) (h34 : bagWinningBool P₃ P₄ = true)
    (h40 : bagWinningBool P₄ P₀ = true)
    (hreach : winningToBool P₀ j GameState.init = true) :
    TetrisSolvable :=
  tetrisSolvable_of_five_cyclic_phases
    ((bagWinningBool_eq_true_iff _ _).mp h01)
    ((bagWinningBool_eq_true_iff _ _).mp h12)
    ((bagWinningBool_eq_true_iff _ _).mp h23)
    ((bagWinningBool_eq_true_iff _ _).mp h34)
    ((bagWinningBool_eq_true_iff _ _).mp h40)
    ((winningToBool_eq_true_iff _ _ _).mp hreach)

/-- **Validity-strengthened five-phase cyclic family theorem.** The
`TetrisSolvableValid` companion of `tetrisSolvable_of_five_cyclic_phases`: five
phase sets cyclically bag-winning (`P₀→P₁→…→P₄→P₀`) plus a forced reach from
`init` into `P₀` yield the project's ultimate goal `TetrisSolvableValid`, not
merely `TetrisSolvable`. Identical to the non-valid form except it routes the
recurrence through `tetrisSolvableValid_of_winningTo_recurrent`. Period 5 is the
universal shape the count-residue argument predicts (residues advance `+8` mod
10 per bag, order 5), so this theorem delivers a valid never-topping-out solver
for any bag-aligned recurrent family of any length `5k` bags. -/
theorem tetrisSolvableValid_of_five_cyclic_phases
    {P₀ P₁ P₂ P₃ P₄ : Finset GameState} {j : ℕ}
    (h01 : BagWinning P₀ P₁) (h12 : BagWinning P₁ P₂)
    (h23 : BagWinning P₂ P₃) (h34 : BagWinning P₃ P₄)
    (h40 : BagWinning P₄ P₀)
    (hreach : WinningTo P₀ j GameState.init) :
    TetrisSolvableValid := by
  classical
  set T : Finset GameState := P₀ ∪ P₁ ∪ P₂ ∪ P₃ ∪ P₄ with hT
  have hsub₀ : P₀ ⊆ T := by intro x hx; simp [hT, hx]
  have hsub₁ : P₁ ⊆ T := by intro x hx; simp [hT, hx]
  have hsub₂ : P₂ ⊆ T := by intro x hx; simp [hT, hx]
  have hsub₃ : P₃ ⊆ T := by intro x hx; simp [hT, hx]
  have hsub₄ : P₄ ⊆ T := by intro x hx; simp [hT, hx]
  refine tetrisSolvableValid_of_winningTo_recurrent (T := T) ?_ (hreach.mono hsub₀)
  intro t ht
  have h7 : 1 ≤ piecesPerBag := by decide
  have hcases : t ∈ P₀ ∨ t ∈ P₁ ∨ t ∈ P₂ ∨ t ∈ P₃ ∨ t ∈ P₄ := by
    simpa [hT, Finset.mem_union, or_assoc] using ht
  rcases hcases with h | h | h | h | h
  · exact ⟨piecesPerBag, h7, (h01 t h).mono hsub₁⟩
  · exact ⟨piecesPerBag, h7, (h12 t h).mono hsub₂⟩
  · exact ⟨piecesPerBag, h7, (h23 t h).mono hsub₃⟩
  · exact ⟨piecesPerBag, h7, (h34 t h).mono hsub₄⟩
  · exact ⟨piecesPerBag, h7, (h40 t h).mono hsub₀⟩

/-- **Validity-strengthened executable five-phase cyclic certificate.** The
`TetrisSolvableValid` companion of `tetrisSolvable_of_five_cyclic_phasesBool`:
all six obligations are `native_decide`-able boolean checks on explicit finite
data — five `bagWinningBool` cyclic-hop checks plus one `winningToBool` reach
from `init` — and discharge the project's ultimate goal `TetrisSolvableValid`
directly. This is the exact verification shape a Rust-discovered five-phase
period-5 certificate is checked in, now delivering a valid never-topping-out
solver with no further proof work. -/
theorem tetrisSolvableValid_of_five_cyclic_phasesBool
    {P₀ P₁ P₂ P₃ P₄ : Finset GameState} {j : ℕ}
    (h01 : bagWinningBool P₀ P₁ = true) (h12 : bagWinningBool P₁ P₂ = true)
    (h23 : bagWinningBool P₂ P₃ = true) (h34 : bagWinningBool P₃ P₄ = true)
    (h40 : bagWinningBool P₄ P₀ = true)
    (hreach : winningToBool P₀ j GameState.init = true) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_five_cyclic_phases
    ((bagWinningBool_eq_true_iff _ _).mp h01)
    ((bagWinningBool_eq_true_iff _ _).mp h12)
    ((bagWinningBool_eq_true_iff _ _).mp h23)
    ((bagWinningBool_eq_true_iff _ _).mp h34)
    ((bagWinningBool_eq_true_iff _ _).mp h40)
    ((winningToBool_eq_true_iff _ _ _).mp hreach)

/-- **Self-recurrent family certificate.** `BagWinning T T` says every state
of `T` forces the next full bag back into `T` — i.e. `T` is closed under
one-bag adversarial play. Together with a forced reach from `init`, this is
`TetrisSolvable`. This is the exact shape a bag-granularity GFP search
emits: the converged fixpoint `T` *is* the certificate. -/
theorem tetrisSolvable_of_selfBagWinning {T : Finset GameState} {j : ℕ}
    (hself : BagWinning T T) (hreach : WinningTo T j GameState.init) :
    TetrisSolvable :=
  tetrisSolvable_of_winningTo_recurrent
    (fun t ht => ⟨piecesPerBag, by decide, hself t ht⟩) hreach

/-- **Validity-strengthened self-recurrent family certificate.** The
`TetrisSolvableValid` companion of `tetrisSolvable_of_selfBagWinning`: a
bag-closed family `T` (every state forces the next full bag back into `T`) plus
a forced reach from `init` yields the project's ultimate goal
`TetrisSolvableValid`, not merely `TetrisSolvable`. Identical to the non-valid
form except it routes through `tetrisSolvableValid_of_winningTo_recurrent`. This
is the exact shape a bag-granularity GFP search emits — the converged fixpoint
`T` *is* the certificate — now delivering a valid never-topping-out solver. -/
theorem tetrisSolvableValid_of_selfBagWinning {T : Finset GameState} {j : ℕ}
    (hself : BagWinning T T) (hreach : WinningTo T j GameState.init) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_winningTo_recurrent
    (fun t ht => ⟨piecesPerBag, by decide, hself t ht⟩) hreach

/-- Executable form of the self-recurrent family certificate: two boolean
checks on explicit finite data. -/
theorem tetrisSolvable_of_selfBagWinningBool {T : Finset GameState} {j : ℕ}
    (hself : bagWinningBool T T = true)
    (hreach : winningToBool T j GameState.init = true) :
    TetrisSolvable :=
  tetrisSolvable_of_selfBagWinning
    ((bagWinningBool_eq_true_iff _ _).mp hself)
    ((winningToBool_eq_true_iff _ _ _).mp hreach)

/-- **Validity-strengthened executable self-recurrent certificate.** The
`TetrisSolvableValid` companion of `tetrisSolvable_of_selfBagWinningBool`: two
`native_decide`-able boolean checks on explicit finite data — `T` is bag-closed
(`bagWinningBool T T`) and `init` forcibly reaches `T` (`winningToBool T j init`)
— discharge the project's ultimate goal `TetrisSolvableValid` directly. This is
the headline a converged bag-granularity GFP search should target: supply the
fixpoint Finset `T` plus a reach depth `j`, evaluate both Bools by reflection,
and obtain a valid never-topping-out solver with no further proof work. -/
theorem tetrisSolvableValid_of_selfBagWinningBool {T : Finset GameState} {j : ℕ}
    (hself : bagWinningBool T T = true)
    (hreach : winningToBool T j GameState.init = true) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_selfBagWinning
    ((bagWinningBool_eq_true_iff _ _).mp hself)
    ((winningToBool_eq_true_iff _ _ _).mp hreach)

/-- **Policy-table self-recurrence certificate.** One table `π` passing the
leg check on every member of `T` (into `T` itself) makes `T` self-recurrent;
with a forced reach from `init` this is `TetrisSolvable`. Both hypotheses are
decidable on explicit data — `native_decide` shape, linear in the table
(≤ 5040·7 steps per member, no 17⁷ search). -/
theorem tetrisSolvable_of_selfPolicy {T : Finset GameState} {j : ℕ}
    (π : GameState → Piece → Placement)
    (hself : ∀ g ∈ T, checkPolicy T π piecesPerBag g = true)
    (hreach : WinningTo T j GameState.init) :
    TetrisSolvable :=
  tetrisSolvable_of_selfBagWinning (BagWinning.of_policy π hself) hreach

/-- Membership form: if the empty-board state itself survives in `T`, the
reach obligation is trivial. -/
theorem tetrisSolvable_of_selfPolicy_mem {T : Finset GameState}
    (π : GameState → Piece → Placement)
    (hself : ∀ g ∈ T, checkPolicy T π piecesPerBag g = true)
    (hinit : GameState.init ∈ T) :
    TetrisSolvable :=
  tetrisSolvable_of_selfPolicy π hself ((winningTo_zero_iff T _).mpr hinit)

/-- One-bag-reach form: a second table `πr` certifying `init → T` in a
single bag. Longer reaches compose leg tables via `WinningTo.compose`. -/
theorem tetrisSolvable_of_selfPolicy_reach {T : Finset GameState}
    (π πr : GameState → Piece → Placement)
    (hself : ∀ g ∈ T, checkPolicy T π piecesPerBag g = true)
    (hreach : checkPolicy T πr piecesPerBag GameState.init = true) :
    TetrisSolvable :=
  tetrisSolvable_of_selfPolicy π hself (WinningTo.of_checkPolicy πr hreach)


/-! ## Indexed certificates — scan-free family checking

`checkPolicy`/`checkCert` test target membership at every depth-7 leaf with
`decide (g ∈ T)` — a linear scan of `T` per leaf, quadratic in family size
across a whole self-recurrent family. An **indexed certificate** instead
names, at each leaf, the array index of the state the trajectory lands on:
the leaf check is a single state equality, and the family check is linear in
the total certificate size. Certificates are shipped as flat ℕ streams and
decoded structurally — the decoder needs no correctness proof, since
`checkICert` validates whatever it produces. -/

/-- Indexed strategy certificate: nodes carry the player's placement per
piece; leaves name the index (in the certified array) of the landing state. -/
inductive ICert : Type where
  | leaf (i : ℕ) : ICert
  | node (choose : Piece → Placement) (child : Piece → ICert) : ICert

/-- Replay an indexed certificate against the array `A`: at depth `0` the
state must equal `A[i]` for the leaf's index `i`; otherwise the state is
alive and every bag piece's chosen placement is valid with an accepting
child. Cost is linear in the certificate — no search, no membership scan. -/
def checkICert (A : Array GameState) : ℕ → GameState → ICert → Bool
  | 0, g, .leaf i =>
      match A[i]? with
      | some t => decide (g = t)
      | none => false
  | 0, _, .node _ _ => false
  | _ + 1, _, .leaf _ => false
  | n + 1, g, .node choose child =>
      decide (¬ g.lost GameConfig.standard) &&
        decide (∀ p ∈ g.bag,
          choose p ∈ Placement.allValidFor GameConfig.standard p ∧
            checkICert A n (adversarialStep GameConfig.standard g p (choose p))
              (child p) = true)

/-- **Indexed-certificate soundness**: acceptance proves the forced win into
the array's underlying state set. -/
theorem WinningTo.of_checkICert {A : Array GameState} :
    ∀ {n : ℕ} {g : GameState} {c : ICert},
      checkICert A n g c = true → WinningTo A.toList.toFinset n g := by
  intro n
  induction n with
  | zero =>
      intro g c h
      rw [winningTo_zero_iff, List.mem_toFinset]
      cases c with
      | node choose child => simp [checkICert] at h
      | leaf i =>
          simp only [checkICert] at h
          cases hA : A[i]? with
          | none => rw [hA] at h; simp at h
          | some t =>
              rw [hA, decide_eq_true_eq] at h
              subst h
              rw [Array.getElem?_eq_some_iff] at hA
              obtain ⟨hi, rfl⟩ := hA
              rw [← Array.getElem_toList hi]
              exact List.getElem_mem _
  | succ n ih =>
      intro g c h
      cases c with
      | leaf i => simp [checkICert] at h
      | node choose child =>
          simp only [checkICert, Bool.and_eq_true, decide_eq_true_eq] at h
          obtain ⟨hnl, hall⟩ := h
          rw [winningTo_succ_iff]
          exact ⟨hnl, fun p hp =>
            ⟨choose p, (hall p hp).1, ih (hall p hp).2⟩⟩

/-- Decode one certificate tree from a flat ℕ stream (preorder):
`0 :: i :: rest` is a leaf naming array index `i`;
`1 :: c₀ :: ⋯ :: c₆ :: rest` is a node carrying one placement code per piece
(in `O,I,S,Z,T,L,J` order, `code = rot·10 + col`) followed by the seven child
trees. Malformed input decodes to junk leaves — soundness is enforced by
`checkICert`, never by the decoder. -/
def decodeICert : ℕ → List ℕ → ICert × List ℕ
  | 0, rest => (.leaf 0, rest)
  | _ + 1, 0 :: i :: rest => (.leaf i, rest)
  | fuel + 1, 1 :: cO :: cI :: cS :: cZ :: cT :: cL :: cJ :: rest =>
      let (chO, rest) := decodeICert fuel rest
      let (chI, rest) := decodeICert fuel rest
      let (chS, rest) := decodeICert fuel rest
      let (chZ, rest) := decodeICert fuel rest
      let (chT, rest) := decodeICert fuel rest
      let (chL, rest) := decodeICert fuel rest
      let (chJ, rest) := decodeICert fuel rest
      let code : Piece → ℕ := fun p => match p with
        | .O => cO | .I => cI | .S => cS | .Z => cZ
        | .T => cT | .L => cL | .J => cJ
      (.node
        (fun p => ⟨p, ⟨code p / 10 % 4, Nat.mod_lt _ (by omega)⟩, code p % 10⟩)
        (fun p => match p with
          | .O => chO | .I => chI | .S => chS | .Z => chZ
          | .T => chT | .L => chL | .J => chJ),
       rest)
  | _ + 1, rest => (.leaf 0, rest)

/-- Check a whole self-recurrent family: state `i` of `A` must pass its
decoded certificate back into `A` itself. -/
def checkICerts (A : Array GameState) (certs : Array (List ℕ)) : Bool :=
  decide (A.size = certs.size) &&
    (List.range A.size).all (fun i =>
      match A[i]?, certs[i]? with
      | some g, some s => checkICert A piecesPerBag g (decodeICert s.length s).1
      | _, _ => false)

/-- A passing indexed-certificate family is self-recurrent:
`BagWinning T T` for `T` the array's state set. -/
theorem bagWinning_of_checkICerts {A : Array GameState}
    {certs : Array (List ℕ)} (h : checkICerts A certs = true) :
    BagWinning A.toList.toFinset A.toList.toFinset := by
  intro g hg
  rw [List.mem_toFinset] at hg
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.mp hg
  simp only [checkICerts, Bool.and_eq_true, List.all_eq_true, List.mem_range,
    decide_eq_true_eq] at h
  obtain ⟨hsz, hall⟩ := h
  have hiA : i < A.size := by simpa using hi
  have hcheck := hall i hiA
  have hAi : A[i]? = some A[i] := by
    rw [Array.getElem?_eq_some_iff]; exact ⟨hiA, rfl⟩
  have hci : i < certs.size := hsz ▸ hiA
  have hCi : certs[i]? = some certs[i] := by
    rw [Array.getElem?_eq_some_iff]; exact ⟨hci, rfl⟩
  rw [hAi, hCi] at hcheck
  have : A.toList[i] = A[i] := Array.getElem_toList hiA
  rw [this]
  exact WinningTo.of_checkICert hcheck

/-- **Indexed-certificate self-recurrence + forced reach ⇒ Tetris solvable.** -/
theorem tetrisSolvable_of_checkICerts {A : Array GameState}
    {certs : Array (List ℕ)} {j : ℕ}
    (h : checkICerts A certs = true)
    (hreach : WinningTo A.toList.toFinset j GameState.init) :
    TetrisSolvable :=
  tetrisSolvable_of_selfBagWinning (bagWinning_of_checkICerts h) hreach

/-- Membership form: the empty-board state is itself in the family. -/
theorem tetrisSolvable_of_checkICerts_mem {A : Array GameState}
    {certs : Array (List ℕ)}
    (h : checkICerts A certs = true)
    (hinit : GameState.init ∈ A.toList) :
    TetrisSolvable :=
  tetrisSolvable_of_checkICerts h
    ((winningTo_zero_iff _ _).mpr (List.mem_toFinset.mpr hinit))

/-- One-bag-reach form: a single extra certificate forcing `init` into the
family. Longer reaches compose via `WinningTo.compose`. -/
theorem tetrisSolvable_of_checkICerts_reach {A : Array GameState}
    {certs : Array (List ℕ)} {s : List ℕ}
    (h : checkICerts A certs = true)
    (hreach : checkICert A piecesPerBag GameState.init
      (decodeICert s.length s).1 = true) :
    TetrisSolvable :=
  tetrisSolvable_of_checkICerts h (WinningTo.of_checkICert hreach)

/-! ## Pooled indexed certificates — DAG-shared streams

A full `ICert` tree per family member is ~173k naturals (8,660 interior
nodes, 5,040 real leaves, 46,921 dummy leaves): too large to ship as text.
But the strategy DAG is far smaller — the same `(board, bag)` node recurs
across orderings and across members. The pooled decoder adds one opcode:
`2 :: j` reuses the `j`-th previously completed node, so the stream is
DAG-sized while `checkICert` still walks the (virtual) tree. The decoder
remains proof-free: malformed input decodes to junk and the checker rejects
it. Streams ship as space-separated digit strings (`natsOfString`) so the
Lean elaborator sees one literal, not a million-element list. -/

/-- Decode one certificate tree with back-references: `0 :: i` is a leaf
naming core index `i`; `2 :: j` reuses pool node `j`; `1 :: cO..cJ` opens a
node (placement code per piece, `code = rot·10 + col`, children in
`O,I,S,Z,T,L,J` order), and the completed node is appended to the pool. -/
def decodePICert : ℕ → List ℕ → Array ICert → ICert × List ℕ × Array ICert
  | 0, rest, pool => (.leaf 0, rest, pool)
  | _ + 1, 0 :: i :: rest, pool => (.leaf i, rest, pool)
  | _ + 1, 2 :: j :: rest, pool => (pool[j]?.getD (.leaf 0), rest, pool)
  | fuel + 1, 1 :: cO :: cI :: cS :: cZ :: cT :: cL :: cJ :: rest, pool =>
      let (chO, rest, pool) := decodePICert fuel rest pool
      let (chI, rest, pool) := decodePICert fuel rest pool
      let (chS, rest, pool) := decodePICert fuel rest pool
      let (chZ, rest, pool) := decodePICert fuel rest pool
      let (chT, rest, pool) := decodePICert fuel rest pool
      let (chL, rest, pool) := decodePICert fuel rest pool
      let (chJ, rest, pool) := decodePICert fuel rest pool
      let code : Piece → ℕ := fun p => match p with
        | .O => cO | .I => cI | .S => cS | .Z => cZ
        | .T => cT | .L => cL | .J => cJ
      let node : ICert := .node
        (fun p => ⟨p, ⟨code p / 10 % 4, Nat.mod_lt _ (by omega)⟩, code p % 10⟩)
        (fun p => match p with
          | .O => chO | .I => chI | .S => chS | .Z => chZ
          | .T => chT | .L => chL | .J => chJ)
      (node, rest, pool.push node)
  | _ + 1, rest, pool => (.leaf 0, rest, pool)

/-- Decode `n` certificate roots off one shared stream, threading the pool. -/
def decodePICerts : ℕ → ℕ → List ℕ → Array ICert → Array ICert → Array ICert
  | 0, _, _, _, roots => roots
  | n + 1, fuel, s, pool, roots =>
      let (c, rest, pool) := decodePICert fuel s pool
      decodePICerts n fuel rest pool (roots.push c)

/-- Family check against pre-decoded roots: member `i` of `A` must pass its
root certificate back into `A` itself. -/
def checkPICertRoots (A : Array GameState) (roots : Array ICert) : Bool :=
  decide (A.size = roots.size) &&
    (List.range A.size).all (fun i =>
      match A[i]?, roots[i]? with
      | some g, some c => checkICert A piecesPerBag g c
      | _, _ => false)

/-- Whole-family check from one pooled stream. -/
def checkPICerts (A : Array GameState) (s : List ℕ) : Bool :=
  checkPICertRoots A (decodePICerts A.size s.length s #[] #[])

theorem bagWinning_of_checkPICertRoots {A : Array GameState}
    {roots : Array ICert} (h : checkPICertRoots A roots = true) :
    BagWinning A.toList.toFinset A.toList.toFinset := by
  intro g hg
  rw [List.mem_toFinset] at hg
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.mp hg
  simp only [checkPICertRoots, Bool.and_eq_true, List.all_eq_true,
    List.mem_range, decide_eq_true_eq] at h
  obtain ⟨hsz, hall⟩ := h
  have hiA : i < A.size := by simpa using hi
  have hcheck := hall i hiA
  have hAi : A[i]? = some A[i] := by
    rw [Array.getElem?_eq_some_iff]; exact ⟨hiA, rfl⟩
  have hri : i < roots.size := hsz ▸ hiA
  have hRi : roots[i]? = some roots[i] := by
    rw [Array.getElem?_eq_some_iff]; exact ⟨hri, rfl⟩
  rw [hAi, hRi] at hcheck
  have : A.toList[i] = A[i] := Array.getElem_toList hiA
  rw [this]
  exact WinningTo.of_checkICert hcheck

/-- A passing pooled-certificate stream makes the array's state set
self-recurrent. -/
theorem bagWinning_of_checkPICerts {A : Array GameState} {s : List ℕ}
    (h : checkPICerts A s = true) :
    BagWinning A.toList.toFinset A.toList.toFinset :=
  bagWinning_of_checkPICertRoots h

/-- **Pooled-certificate self-recurrence + forced reach ⇒ Tetris solvable.** -/
theorem tetrisSolvable_of_checkPICerts {A : Array GameState} {s : List ℕ}
    {j : ℕ} (h : checkPICerts A s = true)
    (hreach : WinningTo A.toList.toFinset j GameState.init) :
    TetrisSolvable :=
  tetrisSolvable_of_selfBagWinning (bagWinning_of_checkPICerts h) hreach

/-- Membership form: the empty-board state is itself in the family. -/
theorem tetrisSolvable_of_checkPICerts_mem {A : Array GameState} {s : List ℕ}
    (h : checkPICerts A s = true)
    (hinit : GameState.init ∈ A.toList) :
    TetrisSolvable :=
  tetrisSolvable_of_checkPICerts h
    ((winningTo_zero_iff _ _).mpr (List.mem_toFinset.mpr hinit))

/-- One-bag-reach form: an extra pooled stream forcing `init` into the
family. -/
theorem tetrisSolvable_of_checkPICerts_reach {A : Array GameState}
    {s r : List ℕ} (h : checkPICerts A s = true)
    (hreach : checkICert A piecesPerBag GameState.init
      (decodePICert r.length r #[]).1 = true) :
    TetrisSolvable :=
  tetrisSolvable_of_checkPICerts h (WinningTo.of_checkICert hreach)

/-- Parse a space-separated ℕ stream. Proof-free: garbage parses to garbage
and the checker rejects it. -/
def natsOfString (s : String) : List ℕ :=
  (s.splitOn " ").filterMap String.toNat?

/-- Decode `k` coordinates (flattened pairs) off a ℕ stream. -/
def decodeCoords : ℕ → List ℕ → List Coord × List ℕ
  | 0, rest => ([], rest)
  | k + 1, c :: r :: rest =>
      let (cs, rest) := decodeCoords k rest
      ((c, r) :: cs, rest)
  | _ + 1, rest => ([], rest)

/-- Decode `n` full-bag states (`cellCount :: c r ...` each) off a ℕ
stream. Proof-free: the family this produces is whatever the certificates
then certify. -/
def decodeStates : ℕ → List ℕ → Array GameState → Array GameState
  | 0, _, acc => acc
  | n + 1, k :: rest, acc =>
      let (cs, rest) := decodeCoords k rest
      decodeStates n rest (acc.push ⟨cs.toFinset, Bag.full⟩)
  | _ + 1, [], acc => acc

/-! ## Closed-table certificates — the depth-1 atlas form

A *closed table* is the M4 artifact in executable form: an array of states
`A` (boards with explicit bags, so mid-bag states are first-class) and, for
each state and each bag-drawable piece, one placement that is valid and
lands directly on another member of `A`. Depth-1 closure makes every member
recurrent (`WinningTo A 1`), so the recurrent-family coinduction
(`tetrisSolvable_of_winningTo_recurrent`) applies with no strategy trees at
all: the certificate IS the atlas, and checking it is seven placement
replays per state instead of a 5040-ordering tree walk. Certificates reuse
the `ICert` machinery at depth 1 — a root node whose children are all
leaves — so soundness is inherited verbatim from `WinningTo.of_checkICert`. -/

/-- Piece bit position in the canonical `O,I,S,Z,T,L,J` emission order
(matches the Rust mirror's bag bitmask, `FULL_BAG = 0x7F`). -/
def pieceIdx : Piece → ℕ
  | .O => 0 | .I => 1 | .S => 2 | .Z => 3 | .T => 4 | .L => 5 | .J => 6

/-- Decode a 7-bit bag mask: bit `pieceIdx p` set ⇔ `p` is in the bag. -/
def bagOfMask (m : ℕ) : Bag :=
  Finset.univ.filter fun p => m / 2 ^ pieceIdx p % 2 = 1

/-- Decode `n` states with explicit bag masks
(`bagMask :: cellCount :: c r ...` each) off a ℕ stream. Proof-free: the
family this produces is whatever the closed-table check then certifies. -/
def decodeStatesBag : ℕ → List ℕ → Array GameState → Array GameState
  | 0, _, acc => acc
  | n + 1, m :: k :: rest, acc =>
      let (cs, rest) := decodeCoords k rest
      decodeStatesBag n rest (acc.push ⟨cs.toFinset, bagOfMask m⟩)
  | _ + 1, _, acc => acc

/-- Depth-1 family check against pre-decoded roots: member `i` must replay
one placement per bag piece, each landing directly on a member of `A`. -/
def checkClosedRoots (A : Array GameState) (roots : Array ICert) : Bool :=
  decide (A.size = roots.size) &&
    (List.range A.size).all (fun i =>
      match A[i]?, roots[i]? with
      | some g, some c => checkICert A 1 g c
      | _, _ => false)

/-- Whole closed-table check from one pooled stream (each root decodes as
`1 :: cO..cJ :: (0 :: i)·7` — a node with seven leaf children). -/
def checkClosedStream (A : Array GameState) (s : List ℕ) : Bool :=
  checkClosedRoots A (decodePICerts A.size s.length s #[] #[])

/-- A passing closed table makes every member recurrent in one step. -/
theorem winningTo_one_of_checkClosedRoots {A : Array GameState}
    {roots : Array ICert} (h : checkClosedRoots A roots = true) :
    ∀ g ∈ A.toList.toFinset, WinningTo A.toList.toFinset 1 g := by
  intro g hg
  rw [List.mem_toFinset] at hg
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.mp hg
  simp only [checkClosedRoots, Bool.and_eq_true, List.all_eq_true,
    List.mem_range, decide_eq_true_eq] at h
  obtain ⟨hsz, hall⟩ := h
  have hiA : i < A.size := by simpa using hi
  have hcheck := hall i hiA
  have hAi : A[i]? = some A[i] := by
    rw [Array.getElem?_eq_some_iff]; exact ⟨hiA, rfl⟩
  have hri : i < roots.size := hsz ▸ hiA
  have hRi : roots[i]? = some roots[i] := by
    rw [Array.getElem?_eq_some_iff]; exact ⟨hri, rfl⟩
  rw [hAi, hRi] at hcheck
  have : A.toList[i] = A[i] := Array.getElem_toList hiA
  rw [this]
  exact WinningTo.of_checkICert hcheck

/-- Stream form of `winningTo_one_of_checkClosedRoots`. -/
theorem winningTo_one_of_checkClosedStream {A : Array GameState}
    {s : List ℕ} (h : checkClosedStream A s = true) :
    ∀ g ∈ A.toList.toFinset, WinningTo A.toList.toFinset 1 g :=
  winningTo_one_of_checkClosedRoots h

/-- **Closed table + `init` membership ⇒ Tetris solvable.** The M4-shaped
entry point: the array is the atlas, the stream carries one placement per
(state, bag piece), and the whole obligation is two `native_decide`s. -/
theorem tetrisSolvable_of_checkClosedStream_mem {A : Array GameState}
    {s : List ℕ} (h : checkClosedStream A s = true)
    (hinit : GameState.init ∈ A.toList) :
    TetrisSolvable :=
  tetrisSolvable_of_winningTo_recurrent
    (fun t ht => ⟨1, le_refl 1, winningTo_one_of_checkClosedStream h t ht⟩)
    ((winningTo_zero_iff _ _).mpr (List.mem_toFinset.mpr hinit))

/-- **Closed table + forced reach ⇒ Tetris solvable.** The table need not
contain `init`: an unreachable closed table (an M2 cycle set) plus one plain
`ICert` forcing `init` into it — reach intermediates are unconstrained —
still yields the full result. -/
theorem tetrisSolvable_of_checkClosedStream_reach {A : Array GameState}
    {s : List ℕ} {n : ℕ} {r : List ℕ}
    (h : checkClosedStream A s = true)
    (hreach : checkICert A n GameState.init (decodeICert r.length r).1 = true) :
    TetrisSolvable :=
  tetrisSolvable_of_winningTo_recurrent
    (fun t ht => ⟨1, le_refl 1, winningTo_one_of_checkClosedStream h t ht⟩)
    (WinningTo.of_checkICert hreach)

set_option linter.style.nativeDecide false in
/-- Executability smoke: a malformed (leaf-rooted) singleton table is
rejected — depth-1 replay demands a node root. -/
example : checkClosedStream #[GameState.init] [0, 0] = false := by
  native_decide

/-! ## Hole accounting — buried cells obstruct holes-0 closed tables

A *hole* is an empty cell strictly below the topmost filled cell of its
column. Holes are created by overhanging piece placements (S/Z most
prominently) and can be eliminated only by line clears — a row strictly
above a hole filling completely. The `colHoles`/`holes` definitions and
their general algebra now live in `Proofs.Invariants.Holes`; this subsection
exhibits per-rotation "hole-creating column" witnesses for S and Z, and
shows that **dropping an S (resp. Z) onto the empty board creates at least
one hole**. The empirical (∅,full)-died-at-sweep-1
result for every (rows, holes=0) closed-GFP rung is the concrete payoff:
no closed table that contains `init` can lie entirely inside the
holes-0 universe. -/

/-- Every S orientation has a *hole-creating column*: a relative column `j`
with no piece cell at relative row 0 but a piece cell at some row `h ≥ 1`.
When the piece is hard-dropped onto a flat surface the column ends up with
piece body above an empty row 0 — a genuine hole. -/
theorem s_hole_witness (r : Rotation) :
    ∃ j h : ℕ, 0 < h ∧ (j, h) ∈ Piece.S.shapeUp r ∧ (j, 0) ∉ Piece.S.shapeUp r := by
  fin_cases r
  · exact ⟨2, 1, by omega, by decide, by decide⟩
  · exact ⟨0, 1, by omega, by decide, by decide⟩
  · exact ⟨2, 1, by omega, by decide, by decide⟩
  · exact ⟨0, 1, by omega, by decide, by decide⟩

/-- Z analog of `s_hole_witness` — Z's mirror shape also leaves a
hole-creating column on a flat surface. -/
theorem z_hole_witness (r : Rotation) :
    ∃ j h : ℕ, 0 < h ∧ (j, h) ∈ Piece.Z.shapeUp r ∧ (j, 0) ∉ Piece.Z.shapeUp r := by
  fin_cases r
  · exact ⟨0, 1, by omega, by decide, by decide⟩
  · exact ⟨1, 1, by omega, by decide, by decide⟩
  · exact ⟨0, 1, by omega, by decide, by decide⟩
  · exact ⟨1, 1, by omega, by decide, by decide⟩

/-- On an empty board the hard-drop offset collapses to `0` — every
column's height is `0` and nat subtraction by the cell's row contributes
`0` to the supremum. -/
@[simp] theorem dropOffset_empty (pl : Placement) : pl.dropOffset ∅ = 0 := by
  unfold Placement.dropOffset
  simp [Board.colHeight_empty]

/-- `place ∅` only adds the piece body: there is no stack to merge with. -/
theorem place_empty_eq_dropped (pl : Placement) : pl.place ∅ = pl.dropped ∅ := by
  unfold Placement.place
  simp

/-- On the empty board, a cell belongs to the placed board iff it is
the absolute coordinate of some `shapeUp` cell (no drop offset). -/
theorem mem_place_empty_iff (pl : Placement) (c : Coord) :
    c ∈ pl.place ∅ ↔ ∃ x ∈ pl.shapeUp, (pl.col + x.1, x.2) = c := by
  rw [place_empty_eq_dropped, Placement.dropped_eq_image, dropOffset_empty]
  simp [Finset.mem_image]

/-- **Hole creation from S on an empty board.** Every S placement onto
the empty board leaves the piece body above an empty row 0 in the
hole-creating column — at least one buried cell. The empirical
(rows, holes=0) closed-GFP rungs all die at sweep 1 because the
adversary can draw S from `(∅, full)` and this theorem rules out a
holes-0 successor. -/
theorem one_le_holes_S_place_empty {cfg : GameConfig} {pl : Placement}
    (hv : pl.Valid cfg) (hS : pl.piece = Piece.S) :
    1 ≤ holes cfg (pl.place ∅) := by
  classical
  obtain ⟨j, h, hpos, hmem, hzero⟩ := s_hole_witness pl.rot
  have hshape : pl.shapeUp = Piece.S.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hS]
  have hmem' : (j, h) ∈ pl.shapeUp := hshape ▸ hmem
  have hzero' : (j, 0) ∉ pl.shapeUp := hshape ▸ hzero
  -- The piece cell `(j, h)` lands at absolute coordinate `(pl.col + j, h)`.
  have hcellMem : (pl.col + j, h) ∈ pl.place ∅ := by
    rw [mem_place_empty_iff]
    exact ⟨(j, h), hmem', by simp⟩
  -- And `(pl.col + j, 0)` is NOT in the placed board.
  have hcellNotMem : (pl.col + j, 0) ∉ pl.place ∅ := by
    rw [mem_place_empty_iff]
    rintro ⟨⟨c, r⟩, hcellShape, heq⟩
    simp only [Prod.mk.injEq, add_right_inj] at heq
    obtain ⟨hc, hr⟩ := heq
    subst hc; subst hr
    exact hzero' hcellShape
  -- Hence the column has at least one hole.
  have hcol : 1 ≤ colHoles (pl.place ∅) (pl.col + j) :=
    one_le_colHoles_of_zero_notMem_of_mem_pos hpos hcellMem hcellNotMem
  -- And the column lies within the board's column range, so the sum bound applies.
  have hrange : pl.col + j < cfg.cols := hv (j, h) hmem'
  unfold holes
  calc 1 ≤ colHoles (pl.place ∅) (pl.col + j) := hcol
    _ ≤ ∑ j' ∈ Finset.range cfg.cols, colHoles (pl.place ∅) j' :=
        Finset.single_le_sum (f := fun j' => colHoles (pl.place ∅) j')
          (fun _ _ => Nat.zero_le _)
          (Finset.mem_range.mpr hrange)

/-- Z analog of `one_le_holes_S_place_empty`. -/
theorem one_le_holes_Z_place_empty {cfg : GameConfig} {pl : Placement}
    (hv : pl.Valid cfg) (hZ : pl.piece = Piece.Z) :
    1 ≤ holes cfg (pl.place ∅) := by
  classical
  obtain ⟨j, h, hpos, hmem, hzero⟩ := z_hole_witness pl.rot
  have hshape : pl.shapeUp = Piece.Z.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hZ]
  have hmem' : (j, h) ∈ pl.shapeUp := hshape ▸ hmem
  have hzero' : (j, 0) ∉ pl.shapeUp := hshape ▸ hzero
  have hcellMem : (pl.col + j, h) ∈ pl.place ∅ := by
    rw [mem_place_empty_iff]
    exact ⟨(j, h), hmem', by simp⟩
  have hcellNotMem : (pl.col + j, 0) ∉ pl.place ∅ := by
    rw [mem_place_empty_iff]
    rintro ⟨⟨c, r⟩, hcellShape, heq⟩
    simp only [Prod.mk.injEq, add_right_inj] at heq
    obtain ⟨hc, hr⟩ := heq
    subst hc; subst hr
    exact hzero' hcellShape
  have hcol : 1 ≤ colHoles (pl.place ∅) (pl.col + j) :=
    one_le_colHoles_of_zero_notMem_of_mem_pos hpos hcellMem hcellNotMem
  have hrange : pl.col + j < cfg.cols := hv (j, h) hmem'
  unfold holes
  calc 1 ≤ colHoles (pl.place ∅) (pl.col + j) := hcol
    _ ≤ ∑ j' ∈ Finset.range cfg.cols, colHoles (pl.place ∅) j' :=
        Finset.single_le_sum (f := fun j' => colHoles (pl.place ∅) j')
          (fun _ _ => Nat.zero_le _)
          (Finset.mem_range.mpr hrange)

/-- Placing a piece on `∅` produces a 4-cell board — far below the
column count of any reasonable config (`5 ≤ cfg.cols`), so no row can
be full and `clearLines` reduces to the identity. Hence `applyStep`
collapses to `place` on the empty board. -/
theorem applyStep_empty_eq_place_empty {cfg : GameConfig} (hcols : 5 ≤ cfg.cols)
    (pl : Placement) : pl.applyStep cfg ∅ = pl.place ∅ := by
  unfold Placement.applyStep
  apply Board.clearLines_eq_self_of_count_lt
  have : (pl.place (∅ : Board)).count = 4 := by
    rw [Placement.count_place]; rfl
  rw [this]; omega

/-- **`applyStep` form: S on `∅` lands with at least one hole.** Combines
`one_le_holes_S_place_empty` with `applyStep_empty_eq_place_empty` — line
clearing is a no-op on the 4-cell post-placement board. -/
theorem one_le_holes_S_applyStep_empty {cfg : GameConfig}
    (hcols : 5 ≤ cfg.cols) {pl : Placement}
    (hv : pl.Valid cfg) (hS : pl.piece = Piece.S) :
    1 ≤ holes cfg (pl.applyStep cfg ∅) := by
  rw [applyStep_empty_eq_place_empty hcols]
  exact one_le_holes_S_place_empty hv hS

/-- Z analog: `applyStep` form. -/
theorem one_le_holes_Z_applyStep_empty {cfg : GameConfig}
    (hcols : 5 ≤ cfg.cols) {pl : Placement}
    (hv : pl.Valid cfg) (hZ : pl.piece = Piece.Z) :
    1 ≤ holes cfg (pl.applyStep cfg ∅) := by
  rw [applyStep_empty_eq_place_empty hcols]
  exact one_le_holes_Z_place_empty hv hZ

/-! ## Lyapunov closure principle

The greatest-fixpoint definition of `safe` already gives a generic
post-fixpoint inclusion: any set closed under `safeOp` embeds into `safe`
(`SafeSet.safe_greatest`). This subsection repackages that principle in
forms that plug a *Lyapunov function* `f : GameState → ℕ` and a *bound*
`K` directly into a `TetrisSolvable` proof. Concretely:

- `safe_of_post_fixpoint`: destructured post-fixpoint inclusion.
- `safe_of_Lyapunov_function`: post-fixpoint of `{g | f g ≤ K}`.
- `tetrisSolvable_of_Lyapunov`: TetrisSolvable from any Lyapunov pair
  containing `init`.

Constructing the WITNESSING Lyapunov function is the open research
question (no concrete `(f, K)` has yet been formally verified). Standard
candidates include `colHeight.max + holes` (refuted at small bounds by
the closed-GFP search), `card`, or hybrid surface metrics. The cell-count
modular argument rules out small `K` for `card` (no return to low-cell
states without `5 ∣ piece-count`). -/

/-- **Set form.** Any set `S` whose every member is non-lost and has a
valid `S`-landing for each bag piece is contained in `safe`. (Direct
restatement of `safe_greatest` with the post-fixpoint condition
destructured.) -/
theorem safe_of_post_fixpoint {cfg : GameConfig} {S : Set GameState}
    (h : ∀ g ∈ S, ¬ g.lost cfg ∧ ∀ p ∈ g.bag,
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
        adversarialStep cfg g p pl ∈ S) :
    S ⊆ safe cfg :=
  safe_greatest S (fun g hg => h g hg)

/-- **Function form: Lyapunov closure.** For any `f : GameState → ℕ`,
the sublevel set `{g | f g ≤ K}` is contained in `safe` whenever:
(a) every state in the sublevel set is non-lost, and (b) every state in
the sublevel set has, for every bag piece, a valid placement keeping `f`
inside the bound. This is the canonical "Lyapunov function on a state
graph" pattern. -/
theorem safe_of_Lyapunov_function {cfg : GameConfig}
    (f : GameState → ℕ) (K : ℕ)
    (hnl : ∀ g, f g ≤ K → ¬ g.lost cfg)
    (hstep : ∀ g, f g ≤ K → ∀ p ∈ g.bag, ∃ pl : Placement,
      pl.piece = p ∧ pl.Valid cfg ∧
        f (adversarialStep cfg g p pl) ≤ K) :
    {g | f g ≤ K} ⊆ safe cfg := by
  apply safe_of_post_fixpoint
  intro g hg
  exact ⟨hnl g hg, fun p hp =>
    let ⟨pl, hpiece, hv, hf⟩ := hstep g hg p hp
    ⟨pl, hpiece, hv, hf⟩⟩

/-- **TetrisSolvable from a Lyapunov function.** Any Lyapunov pair
`(f, K)` containing `init` proves the headline result. Plugging a
concrete Lyapunov function (height+holes, surface roughness, …) into
this theorem reduces TetrisSolvable to a per-state placement-existence
obligation — decidable if `f` is and the sublevel set is finite. -/
theorem tetrisSolvable_of_Lyapunov (f : GameState → ℕ) (K : ℕ)
    (hinit : f GameState.init ≤ K)
    (hnl : ∀ g, f g ≤ K → ¬ g.lost GameConfig.standard)
    (hstep : ∀ g, f g ≤ K → ∀ p ∈ g.bag, ∃ pl : Placement,
      pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        f (adversarialStep GameConfig.standard g p pl) ≤ K) :
    TetrisSolvable :=
  safe_extract
    (safe_of_Lyapunov_function f K hnl hstep hinit)

/-- **Validity-strengthened Lyapunov certificate.** The `TetrisSolvableValid`
companion of `tetrisSolvable_of_Lyapunov`: a Lyapunov pair `(f, K)` containing
`init` proves not just solvability but the never-topping-out (`Valid`) headline,
since `safe_of_Lyapunov_function` places `init` in `safe GameConfig.standard` and
`tetrisSolvableValid_iff_init_safe` identifies that membership with the valid
result. Any concrete bounded potential (height+holes, surface roughness, …) with
a per-state placement-existence obligation now discharges the project's ultimate
goal directly — the canonical existence route to the Atlas. -/
theorem tetrisSolvableValid_of_Lyapunov (f : GameState → ℕ) (K : ℕ)
    (hinit : f GameState.init ≤ K)
    (hnl : ∀ g, f g ≤ K → ¬ g.lost GameConfig.standard)
    (hstep : ∀ g, f g ≤ K → ∀ p ∈ g.bag, ∃ pl : Placement,
      pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        f (adversarialStep GameConfig.standard g p pl) ≤ K) :
    TetrisSolvableValid :=
  tetrisSolvableValid_iff_init_safe.mpr
    (safe_of_Lyapunov_function f K hnl hstep hinit)

/-- **Finset form.** A finite candidate set `T` whose every member is
non-lost and has, for each bag piece, a valid placement landing in `T`,
is contained in `safe`. Decidable when membership and lostness are
decidable — the depth-1 closure check `winningTo_one_of_checkClosedRoots`
is the array-flavored mirror. -/
theorem safe_of_finset_post_fixpoint {cfg : GameConfig} {T : Finset GameState}
    (h : ∀ g ∈ T, ¬ g.lost cfg ∧ ∀ p ∈ g.bag,
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
        adversarialStep cfg g p pl ∈ T) :
    (T : Set GameState) ⊆ safe cfg :=
  safe_of_post_fixpoint (S := (T : Set GameState)) (by
    intro g hg
    exact h g (by simpa using hg))

/-- TetrisSolvable from a finite Lyapunov post-fixpoint containing `init`. -/
theorem tetrisSolvable_of_finset_post_fixpoint {T : Finset GameState}
    (hinit : GameState.init ∈ T)
    (h : ∀ g ∈ T, ¬ g.lost GameConfig.standard ∧ ∀ p ∈ g.bag,
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        adversarialStep GameConfig.standard g p pl ∈ T) :
    TetrisSolvable :=
  safe_extract
    (safe_of_finset_post_fixpoint h (by simpa using hinit))

/-- **Validity-strengthened finite post-fixpoint certificate.** The
`TetrisSolvableValid` companion of `tetrisSolvable_of_finset_post_fixpoint`: a
finite candidate set `T` containing `init`, every member non-lost with a valid
`T`-landing for each bag piece, places `init` in `safe GameConfig.standard`
(`safe_of_finset_post_fixpoint`), which `tetrisSolvableValid_iff_init_safe`
identifies with the never-topping-out headline. This is the exact obligation a
Rust-discovered finite closed table is checked against, now delivering the
project's ultimate goal directly — decidable when membership and lostness are. -/
theorem tetrisSolvableValid_of_finset_post_fixpoint {T : Finset GameState}
    (hinit : GameState.init ∈ T)
    (h : ∀ g ∈ T, ¬ g.lost GameConfig.standard ∧ ∀ p ∈ g.bag,
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        adversarialStep GameConfig.standard g p pl ∈ T) :
    TetrisSolvableValid :=
  tetrisSolvableValid_iff_init_safe.mpr
    (safe_of_finset_post_fixpoint h (by simpa using hinit))

/-- **Structural impossibility: holes-0 closed tables cannot contain `init`.**
A closed-table cert that includes `GameState.init` must, when the
adversary draws `S` from `(∅, full)`, supply a successor in the table.
But every valid `S` placement on the empty board produces a board with
≥ 1 hole — so the successor cannot have `holes = 0`. Hence no closed
table whose every member has `holes = 0` can contain `init`.

This is the structural explanation of the empirical
(∅,full)-died-at-sweep-1 collapses observed at every `(rows≤K, holes=0)`
closed-GFP rung, lifted from search to proof. -/
theorem no_holes_zero_closed_table_contains_init {A : Array GameState}
    {s : List ℕ} (hcheck : checkClosedStream A s = true)
    (hinit : GameState.init ∈ A.toList)
    (hzero : ∀ g ∈ A.toList, holes GameConfig.standard g.board = 0) :
    False := by
  classical
  have hwin1 : WinningTo A.toList.toFinset 1 GameState.init :=
    winningTo_one_of_checkClosedStream hcheck _ (List.mem_toFinset.mpr hinit)
  rw [winningTo_succ_iff] at hwin1
  obtain ⟨_, hbag⟩ := hwin1
  have hS : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  obtain ⟨pl, hplmem, hsuccIn⟩ := hbag Piece.S hS
  rw [winningTo_zero_iff, List.mem_toFinset] at hsuccIn
  rw [Placement.mem_allValidFor] at hplmem
  obtain ⟨hpiece, hval⟩ := hplmem
  -- `adversarialStep` reaches a successor whose board is `applyStep ∅` of a
  -- placement whose piece is forced to be `S`; since `pl.piece = Piece.S`
  -- already, the with-update is a no-op.
  have hpl_eq : ({pl with piece := Piece.S} : Placement) = pl := by
    rcases pl with ⟨pp, pr, pc⟩
    cases hpiece
    rfl
  have hsucc_board :
      (adversarialStep GameConfig.standard GameState.init Piece.S pl).board =
        pl.applyStep GameConfig.standard ∅ := by
    rw [adversarialStep_board, GameState.init_board_eq_emptyset, hpl_eq]
  -- S placement on ∅ gives ≥ 1 hole — contradicting the holes-0 hypothesis.
  have hholes_succ :
      1 ≤ holes GameConfig.standard
          (adversarialStep GameConfig.standard GameState.init Piece.S pl).board := by
    rw [hsucc_board]
    exact one_le_holes_S_applyStep_empty (by simp) hval hpiece
  have hholes_zero :
      holes GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S pl).board = 0 :=
    hzero _ hsuccIn
  omega

/-! ## Lean-native atlas constructor

Mirror of the Rust closed-GFP algorithm in Lean: iteratively filter a
finite universe `U : Finset GameState`, keeping only states whose every
bag piece has a valid placement landing inside the universe. The
fixpoint is the largest closed subset of `U`. If it contains `init`,
TetrisSolvable follows.

Three results give this a usable executable form:

1. **`closeStep`** — one filter pass: `U.filter (state has every-bag-piece
   landing inside U)`.
2. **`iterate`** — repeated application; terminates via cardinality
   well-foundedness.
3. **`atlas`** — the fixpoint, defined as `iterate U (U.card + 1)` and
   proved to be a fixed point of `closeStep` (pigeonhole on cardinality).
4. **`atlas_subset_safe`** — soundness against the abstract `safe` set.
5. **`tetrisSolvable_of_init_mem_atlas`** — direct route to the
   headline result.

The construction is decidable: for small `U`, `decide` (and on the Rust
side, `native_decide`) verifies init membership and returns the verdict.
The Lean atlas is the "ground truth" — Rust empirical search and Lean
formal proof agree by construction. -/

/-- One closed-table filter step: keep only states that are non-lost AND
have, for every bag piece, some valid placement landing inside `U`. -/
def closeStep (cfg : GameConfig) (U : Finset GameState) : Finset GameState :=
  U.filter (fun g =>
    decide (¬ g.lost cfg) &&
    decide (∀ p ∈ g.bag, ∃ pl ∈ Placement.allValidFor cfg p,
      adversarialStep cfg g p pl ∈ U))

theorem closeStep_subset (cfg : GameConfig) (U : Finset GameState) :
    closeStep cfg U ⊆ U := Finset.filter_subset _ _

theorem mem_closeStep_iff (cfg : GameConfig) (U : Finset GameState) (g : GameState) :
    g ∈ closeStep cfg U ↔
      g ∈ U ∧ ¬ g.lost cfg ∧
      ∀ p ∈ g.bag, ∃ pl ∈ Placement.allValidFor cfg p,
        adversarialStep cfg g p pl ∈ U := by
  unfold closeStep
  rw [Finset.mem_filter, Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq]

/-- Iterate `closeStep` `n` times starting from `U`. -/
def iterate (cfg : GameConfig) (U : Finset GameState) : ℕ → Finset GameState
  | 0 => U
  | n + 1 => closeStep cfg (iterate cfg U n)

theorem iterate_subset (cfg : GameConfig) (U : Finset GameState) (n : ℕ) :
    iterate cfg U (n + 1) ⊆ iterate cfg U n := by
  change closeStep cfg (iterate cfg U n) ⊆ iterate cfg U n
  exact closeStep_subset cfg _

/-- Once `closeStep` is a no-op at iteration `k`, all later iterations equal
the iteration `k` result. -/
theorem iterate_stable_from (cfg : GameConfig) (U : Finset GameState) {k n : ℕ}
    (h : closeStep cfg (iterate cfg U k) = iterate cfg U k) (hk : k ≤ n) :
    iterate cfg U n = iterate cfg U k := by
  induction n, hk using Nat.le_induction with
  | base => rfl
  | succ m _ ih => rw [iterate, ih, h]

/-- Pigeonhole termination: `closeStep` must hit a fixed point within
`U.card + 1` iterations. If not, the cardinality strictly decreases at
every step, falling below zero — impossible. -/
theorem closeStep_iterate_card_eq (cfg : GameConfig) (U : Finset GameState) :
    closeStep cfg (iterate cfg U U.card) = iterate cfg U U.card := by
  classical
  by_contra hne
  -- If closeStep is not a no-op at U.card, prove that every step up to
  -- U.card is strict — otherwise stability propagates and contradicts hne.
  have hstrict : ∀ j ≤ U.card,
      closeStep cfg (iterate cfg U j) ≠ iterate cfg U j := by
    intro j hj hstable
    have heq : iterate cfg U U.card = iterate cfg U j :=
      iterate_stable_from cfg U hstable hj
    have hcs : closeStep cfg (iterate cfg U U.card) = iterate cfg U U.card := by
      rw [heq]; exact hstable
    exact hne hcs
  -- Cardinality decreases by ≥ 1 for the first `U.card` strict steps.
  have hcards : ∀ i ≤ U.card, (iterate cfg U i).card + i ≤ U.card := by
    intro i hi
    induction i with
    | zero => simp [iterate]
    | succ n ih =>
      have hn : n ≤ U.card := by omega
      have ihn := ih hn
      have hne_n : closeStep cfg (iterate cfg U n) ≠ iterate cfg U n :=
        hstrict n hn
      have hsub : iterate cfg U (n+1) ⊆ iterate cfg U n := iterate_subset _ _ _
      have hne_iter : iterate cfg U (n+1) ≠ iterate cfg U n := by
        change closeStep cfg (iterate cfg U n) ≠ iterate cfg U n
        exact hne_n
      have hssub : iterate cfg U (n+1) ⊂ iterate cfg U n :=
        ⟨hsub, fun h => hne_iter (Finset.Subset.antisymm hsub h)⟩
      have hlt : (iterate cfg U (n+1)).card < (iterate cfg U n).card :=
        Finset.card_lt_card hssub
      omega
  have hcap : (iterate cfg U U.card).card + U.card ≤ U.card :=
    hcards U.card (le_refl _)
  have hempty_card : (iterate cfg U U.card).card = 0 := by omega
  have hempty : iterate cfg U U.card = ∅ :=
    Finset.card_eq_zero.mp hempty_card
  have hclose_empty : closeStep cfg (∅ : Finset GameState) = ∅ := by
    unfold closeStep
    exact Finset.filter_empty _
  rw [hempty, hclose_empty] at hne
  exact hne rfl

/-- **The atlas**: the largest closed subset of `U`. Defined as
`iterate cfg U U.card`; by pigeonhole this is a fixed point of `closeStep`. -/
def atlas (cfg : GameConfig) (U : Finset GameState) : Finset GameState :=
  iterate cfg U U.card

theorem atlas_subset (cfg : GameConfig) (U : Finset GameState) : atlas cfg U ⊆ U := by
  unfold atlas
  induction U.card with
  | zero => intro g h; exact h
  | succ n ih =>
    intro g hg
    have : iterate cfg U (n + 1) ⊆ iterate cfg U n := iterate_subset _ _ _
    exact ih (this hg)

theorem closeStep_atlas (cfg : GameConfig) (U : Finset GameState) :
    closeStep cfg (atlas cfg U) = atlas cfg U :=
  closeStep_iterate_card_eq cfg U

/-- **Atlas soundness:** every state in the fixpoint is `safe`. -/
theorem atlas_subset_safe (cfg : GameConfig) (U : Finset GameState) :
    (atlas cfg U : Set GameState) ⊆ safe cfg := by
  apply safe_of_finset_post_fixpoint
  intro g hg
  have hclose : g ∈ closeStep cfg (atlas cfg U) := by
    rw [closeStep_atlas]; exact hg
  rw [mem_closeStep_iff] at hclose
  obtain ⟨_, hnl, hstep⟩ := hclose
  refine ⟨hnl, fun p hp => ?_⟩
  obtain ⟨pl, hpl, hsucc⟩ := hstep p hp
  rw [Placement.mem_allValidFor] at hpl
  exact ⟨pl, hpl.1, hpl.2, hsucc⟩

/-- **TetrisSolvable from init membership in any computed atlas.** Pick
any `U : Finset GameState`; if `init ∈ atlas standard U`, the game is
solvable. The atlas computation is decidable, so this reduces
TetrisSolvable to one `decide` (or `native_decide`) check per choice of
`U`. -/
theorem tetrisSolvable_of_init_mem_atlas (U : Finset GameState)
    (h : GameState.init ∈ atlas GameConfig.standard U) :
    TetrisSolvable :=
  safe_extract (atlas_subset_safe GameConfig.standard U (by simpa using h))

/-! ### Atlas structural theorems — maximality, monotonicity, composition -/

/-- **Maximality.** Any "closed-in-itself" subset `T` of `U` is contained in
`atlas U`. The atlas is the LARGEST closed subset of `U`: it captures every
closed witness simultaneously. Useful for soundness without computation —
just exhibit a closed `T` and conclude `T ⊆ atlas U`. -/
theorem atlas_maximal (cfg : GameConfig) (U T : Finset GameState)
    (hsub : T ⊆ U)
    (hclosed : ∀ g ∈ T, ¬ g.lost cfg ∧ ∀ p ∈ g.bag,
      ∃ pl ∈ Placement.allValidFor cfg p,
        adversarialStep cfg g p pl ∈ T) :
    T ⊆ atlas cfg U := by
  change T ⊆ iterate cfg U U.card
  -- Every iterate is a superset of T: by induction on n.
  suffices hall : ∀ n, T ⊆ iterate cfg U n from hall U.card
  intro n
  induction n with
  | zero => exact hsub
  | succ k ih =>
    intro g hgT
    have hgi : g ∈ iterate cfg U k := ih hgT
    change g ∈ closeStep cfg (iterate cfg U k)
    rw [mem_closeStep_iff]
    refine ⟨hgi, (hclosed g hgT).1, ?_⟩
    intro p hp
    obtain ⟨pl, hpl, hsucc⟩ := (hclosed g hgT).2 p hp
    exact ⟨pl, hpl, ih hsucc⟩

/-- **Monotonicity.** A larger universe gives at least as large an atlas.
Enlarging `U` can only add (never remove) closed witnesses inside it. -/
theorem atlas_monotone {cfg : GameConfig} {U₁ U₂ : Finset GameState}
    (h : U₁ ⊆ U₂) :
    atlas cfg U₁ ⊆ atlas cfg U₂ := by
  apply atlas_maximal cfg U₂ (atlas cfg U₁)
    ((atlas_subset cfg U₁).trans h)
  intro g hg
  have hclose : g ∈ closeStep cfg (atlas cfg U₁) := by
    rw [closeStep_atlas]; exact hg
  rw [mem_closeStep_iff] at hclose
  exact ⟨hclose.2.1, hclose.2.2⟩

/-- **Membership characterisation.** A state `g` is in `atlas cfg U` iff
some "closed witness" `T ⊆ U` containing `g` exists. The atlas equals the
union of all closed-in-themselves subsets of `U`. -/
theorem mem_atlas_iff_exists_closed_witness (cfg : GameConfig)
    (U : Finset GameState) (g : GameState) :
    g ∈ atlas cfg U ↔
      ∃ T : Finset GameState, T ⊆ U ∧ g ∈ T ∧
        ∀ g' ∈ T, ¬ g'.lost cfg ∧ ∀ p ∈ g'.bag,
          ∃ pl ∈ Placement.allValidFor cfg p,
            adversarialStep cfg g' p pl ∈ T := by
  refine ⟨?_, ?_⟩
  · intro hg
    refine ⟨atlas cfg U, atlas_subset cfg U, hg, ?_⟩
    intro g' hg'
    have hclose : g' ∈ closeStep cfg (atlas cfg U) := by
      rw [closeStep_atlas]; exact hg'
    rw [mem_closeStep_iff] at hclose
    exact ⟨hclose.2.1, hclose.2.2⟩
  · rintro ⟨T, hTU, hgT, hclosed⟩
    exact atlas_maximal cfg U T hTU hclosed hgT

/-- **Union containment.** The union of atlases of two universes is
contained in the atlas of their union. (Reverse inclusion may fail —
unioning universes can produce NEW closed subsets that didn't exist in
either alone.) Used for compositional construction: assemble a global
atlas from independently-verified local pieces. -/
theorem atlas_union_supset (cfg : GameConfig) (U₁ U₂ : Finset GameState) :
    atlas cfg U₁ ∪ atlas cfg U₂ ⊆ atlas cfg (U₁ ∪ U₂) := by
  intro g hg
  rcases Finset.mem_union.mp hg with h1 | h2
  · exact atlas_monotone Finset.subset_union_left h1
  · exact atlas_monotone Finset.subset_union_right h2

/-- **Idempotence on closed sets.** If `T` is already closed-in-itself,
then `atlas cfg T = T`. The atlas is a fixed point of "close + restrict"
on the universe of closed sets. -/
theorem atlas_eq_of_closed (cfg : GameConfig) (T : Finset GameState)
    (hclosed : ∀ g ∈ T, ¬ g.lost cfg ∧ ∀ p ∈ g.bag,
      ∃ pl ∈ Placement.allValidFor cfg p,
        adversarialStep cfg g p pl ∈ T) :
    atlas cfg T = T := by
  apply Finset.Subset.antisymm
  · exact atlas_subset cfg T
  · exact atlas_maximal cfg T T Finset.Subset.rfl hclosed

/-- **TetrisSolvable from any closed witness.** The headline theorem in
its most concise form: a finite closed set containing `init` proves the
game solvable. Bridges to `atlas` for decidable verification — the
witness may be the atlas itself or any closed subset thereof. -/
theorem tetrisSolvable_of_closed_witness_containing_init
    (T : Finset GameState) (hinit : GameState.init ∈ T)
    (hclosed : ∀ g ∈ T, ¬ g.lost GameConfig.standard ∧ ∀ p ∈ g.bag,
      ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
        adversarialStep GameConfig.standard g p pl ∈ T) :
    TetrisSolvable :=
  tetrisSolvable_of_init_mem_atlas T (by
    rw [atlas_eq_of_closed GameConfig.standard T hclosed]; exact hinit)

/-! ### Connecting cert systems to the atlas

The closed-table cert (`checkClosedStream`) and the atlas constructor are
two views of the same fixed-point. This subsection ties them together:
a passing cert IS its own atlas, and vice versa any closed witness gives
a `tetrisSolvable` route via the atlas. -/

/-- **A passing `checkClosedStream` cert is closed in itself.** Every
member of the cert array `A` has, for each bag piece, a valid placement
landing on another member. -/
theorem checkClosedStream_implies_closed {A : Array GameState}
    {s : List ℕ} (h : checkClosedStream A s = true)
    (g : GameState) (hg : g ∈ A.toList.toFinset) :
    ¬ g.lost GameConfig.standard ∧ ∀ p ∈ g.bag,
      ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
        adversarialStep GameConfig.standard g p pl ∈ A.toList.toFinset := by
  have hwin := winningTo_one_of_checkClosedStream h g hg
  rw [winningTo_succ_iff] at hwin
  obtain ⟨hnl, hstep⟩ := hwin
  refine ⟨hnl, fun p hp => ?_⟩
  obtain ⟨pl, hpl, hsucc⟩ := hstep p hp
  rw [winningTo_zero_iff] at hsucc
  exact ⟨pl, hpl, hsucc⟩

/-- **Cert array is its own atlas.** A passing cert is a fixed point of
`closeStep`: the atlas of the cert array equals the cert array. -/
theorem atlas_array_eq_of_checkClosedStream {A : Array GameState}
    {s : List ℕ} (h : checkClosedStream A s = true) :
    atlas GameConfig.standard A.toList.toFinset = A.toList.toFinset :=
  atlas_eq_of_closed GameConfig.standard A.toList.toFinset
    (checkClosedStream_implies_closed h)

/-- **Closure-check predicate** — directly decidable. `T` is closed-in-itself
iff every member is non-lost AND has, for each bag piece, a valid landing
inside `T`. Use with `decide`/`native_decide` to verify candidate closed
sets without computing the atlas. -/
def IsClosedSet (cfg : GameConfig) (T : Finset GameState) : Prop :=
  ∀ g ∈ T, ¬ g.lost cfg ∧ ∀ p ∈ g.bag,
    ∃ pl ∈ Placement.allValidFor cfg p,
      adversarialStep cfg g p pl ∈ T

instance (cfg : GameConfig) (T : Finset GameState) : Decidable (IsClosedSet cfg T) := by
  unfold IsClosedSet
  exact Finset.decidableDforallFinset

/-- **TetrisSolvable from a `decide`-checkable witness.** A finite Finset
`T` with `IsClosedSet T` (which `decide`/`native_decide` can verify on
small `T`) and `init ∈ T` gives `TetrisSolvable` in one shot. This is the
tightest form for native_decide-based proofs. -/
theorem tetrisSolvable_of_IsClosedSet (T : Finset GameState)
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T) :
    TetrisSolvable :=
  tetrisSolvable_of_closed_witness_containing_init T hinit hclosed

/-- **Refutation tool.** If `atlas U` is empty, no closed witness exists
inside `U` — neither for `TetrisSolvable` nor for any other purpose.
Formalises the empirical (∅,full)-died-at-sweep-K collapses observed in
every (rows, holes) Rust closed-GFP rung tested so far: those universes
provably admit no closed subset. -/
theorem no_closed_witness_of_atlas_empty (cfg : GameConfig) (U : Finset GameState)
    (hempty : atlas cfg U = ∅)
    (T : Finset GameState) (hsub : T ⊆ U) (hne : T.Nonempty)
    (hclosed : IsClosedSet cfg T) :
    False := by
  have hTatlas : T ⊆ atlas cfg U := atlas_maximal cfg U T hsub hclosed
  obtain ⟨g, hg⟩ := hne
  have hga : g ∈ atlas cfg U := hTatlas hg
  rw [hempty] at hga
  exact Finset.notMem_empty g hga

/-! ### Concentrated refutation toolkit

These theorems sharpen the empty-atlas refutation in two directions:

- **From `init ∉ atlas U`** (one membership check, not full emptiness)
  to "no init-containing closed witness in `U`". Strictly stronger than
  the empty-atlas tool: many atlases have non-init members but no init
  member.

- **From a *single offending piece* + *universe predicate*** to "no
  init-containing closed witness in `U`". A parametric S/Z-style
  argument: pick a piece `p ∈ init.bag` and a property `Q` holding on
  `U`; if every `p`-placement on `∅` violates `Q`, no closed init-atlas
  lives in `U`. Plugs directly into `one_le_holes_S_applyStep_empty`
  (holes-violation) or any other piece-vs-board invariant.

Both routes have computable forms — `decide`/`native_decide` checks the
side conditions on the user's choice of universe. -/

/-- The atlas of any universe is itself closed. -/
theorem atlas_self_IsClosedSet (cfg : GameConfig) (U : Finset GameState) :
    IsClosedSet cfg (atlas cfg U) := by
  intro g hg
  have hclose : g ∈ closeStep cfg (atlas cfg U) := by
    rw [closeStep_atlas]; exact hg
  rw [mem_closeStep_iff] at hclose
  exact ⟨hclose.2.1, hclose.2.2⟩

/-! ### Atlas-constructor convergence: fixed point & greatest closed subset

These characterise the *program* `atlas := iterate U U.card` itself:

- **Stationarity** — once the constructor reaches `atlas U`, every further
  `closeStep` is a no-op (`iterate (atlas U) n = atlas U`).
- **Idempotence** — running the whole constructor on its own output
  reproduces it (`atlas (atlas U) = atlas U`): a genuine fixed point.
- **Greatest closed subset** — `atlas U` is closed and contains every
  closed subset of `U`; `atlas_unique_greatest_closed` shows it is THE
  unique such set, so the constructor's output is canonical. -/

/-- **Stationarity at the atlas.** Every `closeStep` applied to
`atlas cfg U` returns it unchanged: `iterate cfg (atlas cfg U) n =
atlas cfg U` for all `n`. Once converged, the constructor never moves. -/
theorem iterate_atlas_eq (cfg : GameConfig) (U : Finset GameState) (n : ℕ) :
    iterate cfg (atlas cfg U) n = atlas cfg U := by
  induction n with
  | zero => rfl
  | succ k ih =>
    show closeStep cfg (iterate cfg (atlas cfg U) k) = atlas cfg U
    rw [ih, closeStep_atlas]

/-- **Constructor idempotence.** Running the atlas constructor on its own
output reproduces that output: `atlas cfg (atlas cfg U) = atlas cfg U`.
The atlas is a genuine fixed point of the whole construction, not merely
of a single `closeStep`. -/
theorem atlas_idempotent (cfg : GameConfig) (U : Finset GameState) :
    atlas cfg (atlas cfg U) = atlas cfg U := by
  show iterate cfg (atlas cfg U) (atlas cfg U).card = atlas cfg U
  exact iterate_atlas_eq cfg U _

/-- **Greatest closed subset.** `atlas cfg U` is a closed subset of `U`
that contains every closed subset of `U`. It is the greatest element
(under `⊆`) of the lattice of closed-in-`U` Finsets. -/
theorem atlas_greatest_closed (cfg : GameConfig) (U : Finset GameState) :
    atlas cfg U ⊆ U ∧ IsClosedSet cfg (atlas cfg U) ∧
      ∀ T : Finset GameState, T ⊆ U → IsClosedSet cfg T → T ⊆ atlas cfg U :=
  ⟨atlas_subset cfg U, atlas_self_IsClosedSet cfg U,
   fun T hTU hT => atlas_maximal cfg U T hTU hT⟩

/-- **Uniqueness of the greatest closed subset.** Any `S ⊆ U` that is
closed and contains every closed subset of `U` equals `atlas cfg U`.
With `atlas_greatest_closed`, this pins down the atlas as THE unique
greatest closed subset: the constructor's output is canonical, independent
of how it is computed. -/
theorem atlas_unique_greatest_closed (cfg : GameConfig) (U S : Finset GameState)
    (hSU : S ⊆ U) (hScl : IsClosedSet cfg S)
    (hSmax : ∀ T : Finset GameState, T ⊆ U → IsClosedSet cfg T → T ⊆ S) :
    S = atlas cfg U := by
  apply Finset.Subset.antisymm
  · exact atlas_maximal cfg U S hSU hScl
  · exact hSmax (atlas cfg U) (atlas_subset cfg U) (atlas_self_IsClosedSet cfg U)

/-! ### The solver's move set — the program never gets stuck

`atlasMoves cfg U g p` is the (computable) set of placements of piece `p`
at state `g` that keep the game inside `atlas cfg U`. The headline
`atlasMoves_nonempty` says: at every atlas state, for every bag piece, at
least one such move exists — the atlas strategy provably never tops out.
`adversarialStep_mem_atlas_of_mem_atlasMoves` shows the property is
self-perpetuating: each chosen move lands in the atlas, where a move set
is again nonempty. Together these are the program-level read of
`atlas_self_IsClosedSet`. -/

/-- Placements of `p` at `g` that land back in `atlas cfg U`. Computable
(a `Finset.filter`), so a solver can enumerate its legal continuations. -/
def atlasMoves (cfg : GameConfig) (U : Finset GameState) (g : GameState)
    (p : Piece) : Finset Placement :=
  (Placement.allValidFor cfg p).filter
    (fun pl => adversarialStep cfg g p pl ∈ atlas cfg U)

theorem mem_atlasMoves_iff (cfg : GameConfig) (U : Finset GameState)
    (g : GameState) (p : Piece) (pl : Placement) :
    pl ∈ atlasMoves cfg U g p ↔
      pl ∈ Placement.allValidFor cfg p ∧
        adversarialStep cfg g p pl ∈ atlas cfg U := by
  unfold atlasMoves
  rw [Finset.mem_filter]

/-- Every solver move is a genuinely valid placement of `p`. -/
theorem atlasMoves_subset_allValidFor (cfg : GameConfig) (U : Finset GameState)
    (g : GameState) (p : Piece) :
    atlasMoves cfg U g p ⊆ Placement.allValidFor cfg p :=
  Finset.filter_subset _ _

/-- **The program never gets stuck.** At every atlas state `g`, for every
piece `p` the adversary may draw from `g.bag`, the solver has at least one
legal placement keeping the game inside the atlas. Direct program-level
restatement of closedness (`atlas_self_IsClosedSet`). -/
theorem atlasMoves_nonempty (cfg : GameConfig) (U : Finset GameState)
    {g : GameState} (hg : g ∈ atlas cfg U) {p : Piece} (hp : p ∈ g.bag) :
    (atlasMoves cfg U g p).Nonempty := by
  obtain ⟨_, hstep⟩ := atlas_self_IsClosedSet cfg U g hg
  obtain ⟨pl, hpl, hsucc⟩ := hstep p hp
  exact ⟨pl, (mem_atlasMoves_iff cfg U g p pl).mpr ⟨hpl, hsucc⟩⟩

/-- **Invariant preservation (self-perpetuating non-stuck play).** Playing
any move from `atlasMoves` lands back in the atlas, where the move set is
again nonempty. Atlas play continues forever: combine with
`atlasMoves_nonempty` to step indefinitely. -/
theorem adversarialStep_mem_atlas_of_mem_atlasMoves (cfg : GameConfig)
    (U : Finset GameState) {g : GameState} {p : Piece} {pl : Placement}
    (h : pl ∈ atlasMoves cfg U g p) :
    adversarialStep cfg g p pl ∈ atlas cfg U :=
  ((mem_atlasMoves_iff cfg U g p pl).mp h).2

/-- **Atlas states are never lost.** A program-level safety statement: no
state the solver can occupy is a topped-out (lost) state. -/
theorem not_lost_of_mem_atlas (cfg : GameConfig) (U : Finset GameState)
    {g : GameState} (hg : g ∈ atlas cfg U) : ¬ g.lost cfg :=
  (atlas_self_IsClosedSet cfg U g hg).1

/-! ### Executable strategy: the `safeSolver` runs forever from any atlas state

The previous cluster shows `atlasMoves` is never stuck on the atlas. Here we
go further and connect the atlas to the *materialised executable solver*
`safeSolver` (from `SafeSet.lean`). Since `atlas cfg U ⊆ safe cfg`
(`atlas_subset_safe`), and `safeSolver` keeps any safe state safe forever, the
solver survives indefinitely **starting from any atlas state** — not just from
`init`. This is the "program that plays infinitely" turned into a concrete
strategy with a proof that it never tops out. -/

/-- **`safeSolver` trace from a safe start stays safe.** The safe-half of the
`init`-specialised `adversarialTrace_reachable_and_safe`, generalised to an
arbitrary safe starting state `g₀` and a sequence legal from `g₀.bag`.
Induction: base is `hg₀`; the successor uses `safeSolver_spec` on the
inductive safe state, with bag-membership of `t k` from
`adversarialTrace_bag_from`. -/
theorem safeSolver_adversarialTrace_mem_safe_from {cfg : GameConfig}
    {g₀ : GameState} (hg₀ : g₀ ∈ safe cfg)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    ∀ n, adversarialTrace cfg (safeSolver cfg) t g₀ n ∈ safe cfg := by
  intro n
  induction n with
  | zero => exact hg₀
  | succ k ih =>
    have hpb : t k ∈ (adversarialTrace cfg (safeSolver cfg) t g₀ k).bag := by
      rw [adversarialTrace_bag_from]; exact hl k
    rw [adversarialTrace_succ]
    exact (safeSolver_spec ih hpb).2.2

/-- **`safeSolver` solves Tetris from any safe state.** Strengthens
`safeSolver_solvesTetris` (which covered only `init ∈ safe`) to every safe
state. Direct from `safeSolver_adversarialTrace_mem_safe_from` + `safe_not_lost`. -/
theorem safeSolver_solvesTetrisFrom_of_mem_safe {cfg : GameConfig}
    {g₀ : GameState} (hg₀ : g₀ ∈ safe cfg) :
    SolvesTetrisFrom cfg g₀ (safeSolver cfg) :=
  fun _ hl n => safe_not_lost (safeSolver_adversarialTrace_mem_safe_from hg₀ hl n)

/-- **The atlas-to-executable-solver bridge.** From *any* state `g` in the
constructed `atlas cfg U`, the concrete `safeSolver` survives forever against
every bag-legal sequence from `g.bag`. Combines `atlas_subset_safe` (atlas ⊆
safe) with `safeSolver_solvesTetrisFrom_of_mem_safe`. This is the program-level
payoff: the Lean-native atlas constructor yields not just an abstract closed
set but a runnable strategy with an infinite-survival guarantee from each of
its states. -/
theorem safeSolver_solvesTetrisFrom_of_mem_atlas (cfg : GameConfig)
    (U : Finset GameState) {g : GameState} (hg : g ∈ atlas cfg U) :
    SolvesTetrisFrom cfg g (safeSolver cfg) :=
  safeSolver_solvesTetrisFrom_of_mem_safe (atlas_subset_safe cfg U (by simpa using hg))

/-- **`init`-specialisation: explicit solver solving Tetris.** When `init` is in
the atlas, `safeSolver` concretely `SolvesTetris` (init-based, no-loss against
all legal sequences). Sharper than `tetrisSolvable_of_init_mem_atlas` (which
only asserts *existence* of a solver): this names the solver. -/
theorem safeSolver_solvesTetris_of_init_mem_atlas (U : Finset GameState)
    (hU : GameState.init ∈ atlas GameConfig.standard U) :
    SolvesTetris GameConfig.standard (safeSolver GameConfig.standard) :=
  (solvesTetrisFrom_init_iff_solvesTetris GameConfig.standard _).mp
    (safeSolver_solvesTetrisFrom_of_mem_atlas GameConfig.standard U hU)

/-- **Membership characterization, sharp form.** `init` is in `atlas cfg U`
iff some closed witness `T ⊆ U` containing `init` exists. The atlas is
the maximal init-containing closed subset of `U` (if any). -/
theorem init_mem_atlas_iff_exists_closed_init_witness (cfg : GameConfig)
    (U : Finset GameState) :
    GameState.init ∈ atlas cfg U ↔
      ∃ T : Finset GameState, T ⊆ U ∧ GameState.init ∈ T ∧ IsClosedSet cfg T := by
  refine ⟨?_, ?_⟩
  · intro h
    exact ⟨atlas cfg U, atlas_subset cfg U, h, atlas_self_IsClosedSet cfg U⟩
  · rintro ⟨T, hTU, hinitT, hclosed⟩
    exact atlas_maximal cfg U T hTU hclosed hinitT

/-- **Sharper than empty-atlas refutation.** If `init ∉ atlas cfg U`,
no init-containing closed witness in `U` exists — a one-membership-check
refutation. `atlas U` need NOT be empty; many universes have closed
"unreachable" subsets without admitting `init` in the closure. -/
theorem no_init_closed_witness_of_init_not_mem_atlas (cfg : GameConfig)
    (U : Finset GameState) (hinit : GameState.init ∉ atlas cfg U)
    (T : Finset GameState) (hsub : T ⊆ U) (hinitT : GameState.init ∈ T)
    (hclosed : IsClosedSet cfg T) :
    False := by
  exact hinit (atlas_maximal cfg U T hsub hclosed hinitT)

/-- **Parametric piece-escape refutation.** Pick any piece `p ∈ init.bag`.
If every valid placement of `p` from `init` lands outside `U`, then no
closed init-atlas in `U` can respond to that piece — refutation.

This is the S-hole argument in abstract form: the "piece" picks an
adversary draw, the "escape" condition packages a structural reason
(creates a hole, exceeds height, breaks an invariant) for the strategy
to fail. -/
theorem no_init_closed_atlas_of_piece_escapes
    {cfg : GameConfig} {U : Finset GameState}
    (p : Piece) (hp : p ∈ GameState.init.bag)
    (hescape : ∀ pl ∈ Placement.allValidFor cfg p,
      adversarialStep cfg GameState.init p pl ∉ U) :
    ∀ T : Finset GameState, T ⊆ U → GameState.init ∈ T →
      ¬ IsClosedSet cfg T := by
  intro T hTU hinitT hclosed
  obtain ⟨_, hstep⟩ := hclosed GameState.init hinitT
  obtain ⟨pl, hpl, hsucc⟩ := hstep p hp
  exact hescape pl hpl (hTU hsucc)

/-- **Parametric predicate-violation refutation.** A finer-grained
restatement of the piece-escape rule: if every state in the universe `U`
satisfies a property `Q`, and *no* valid `p`-placement from `init` lands
on a `Q`-state, then no closed init-atlas in `U` can fit. The `Q` slot
takes any structural invariant — holes, height, cell count, mod-residue,
etc. -/
theorem no_init_closed_atlas_of_predicate_violated
    {cfg : GameConfig} {U : Finset GameState}
    {p : Piece} (hp : p ∈ GameState.init.bag)
    {Q : GameState → Prop}
    (hUQ : ∀ g ∈ U, Q g)
    (hpQ : ∀ pl ∈ Placement.allValidFor cfg p,
      ¬ Q (adversarialStep cfg GameState.init p pl)) :
    ∀ T : Finset GameState, T ⊆ U → GameState.init ∈ T →
      ¬ IsClosedSet cfg T := by
  refine no_init_closed_atlas_of_piece_escapes p hp ?_
  intro pl hpl hsucc
  exact hpQ pl hpl (hUQ _ hsucc)

/-- **Cardinality lower bound.** Any closed atlas containing `init` has
at least 8 distinct states — `init` itself plus one successor per piece
in the bag (7 pieces, 7 distinct post-draw bag states). Refutes any
candidate atlas of size ≤ 7 by computation. -/
theorem atlas_card_ge_eight_of_init_mem (cfg : GameConfig)
    {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet cfg T) :
    8 ≤ T.card := by
  classical
  -- For each piece p, pick a successor — its bag is `init.bag.draw p`,
  -- which is distinct from init.bag and from each other p's draw.
  have hstep_all : ∀ p : Piece, ∃ succ : GameState,
      succ ∈ T ∧ succ.bag = GameState.init.bag.draw p := by
    intro p
    have hpb : p ∈ GameState.init.bag := by
      rw [GameState.init_bag]; exact Bag.mem_full p
    obtain ⟨_, hstep⟩ := hclosed GameState.init hinit
    obtain ⟨pl, _, hsucc⟩ := hstep p hpb
    refine ⟨adversarialStep cfg GameState.init p pl, hsucc, ?_⟩
    rw [adversarialStep_bag]
  -- Choose one successor per piece via `Classical.choice`.
  choose f hf using hstep_all
  -- The map `Option Piece → GameState`: `none ↦ init`, `some p ↦ f p`.
  let m : Option Piece → GameState := fun o => o.elim GameState.init f
  -- All 8 values land in `T`.
  have hm_mem : ∀ o, m o ∈ T := by
    intro o
    cases o with
    | none => exact hinit
    | some p => exact (hf p).1
  -- And the map is injective: `init` has full bag (7), each `f p` has
  -- bag of size 6 with `p` removed, so they're all distinct.
  have hm_inj : Function.Injective m := by
    intro a b hab
    cases a with
    | none =>
      cases b with
      | none => rfl
      | some pb =>
        exfalso
        have hbag : GameState.init.bag = (f pb).bag := congrArg GameState.bag hab
        rw [(hf pb).2, GameState.init_bag] at hbag
        have hcard : Bag.full.card = (Bag.full.draw pb).card := congrArg _ hbag
        rw [Bag.full_card, Bag.draw_full_card] at hcard
        omega
    | some pa =>
      cases b with
      | none =>
        exfalso
        have hbag : (f pa).bag = GameState.init.bag := congrArg GameState.bag hab
        rw [(hf pa).2, GameState.init_bag] at hbag
        have hcard : (Bag.full.draw pa).card = Bag.full.card := congrArg _ hbag
        rw [Bag.draw_full_card, Bag.full_card] at hcard
        omega
      | some pb =>
        congr 1
        have hbag : (f pa).bag = (f pb).bag := congrArg GameState.bag hab
        rw [(hf pa).2, (hf pb).2, GameState.init_bag] at hbag
        -- Bag.full.draw = Bag.full.erase, so erase pa = erase pb on full ⇒ pa = pb.
        rw [Bag.draw_full_eq_erase, Bag.draw_full_eq_erase] at hbag
        by_contra hne
        have hpa_in : pa ∈ Bag.full := Bag.mem_full pa
        have : pa ∉ Bag.full.erase pa := Finset.notMem_erase pa Bag.full
        rw [hbag] at this
        exact this (Finset.mem_erase.mpr ⟨hne, hpa_in⟩)
  -- Hence `Finset.univ.image m ⊆ T` with cardinality ≥ 8.
  have h_univ_card : (Finset.univ : Finset (Option Piece)).card = 8 := by
    rw [Finset.card_univ]
    decide
  have h_image_card : ((Finset.univ : Finset (Option Piece)).image m).card = 8 := by
    rw [Finset.card_image_of_injective _ hm_inj, h_univ_card]
  have h_image_subset : (Finset.univ : Finset (Option Piece)).image m ⊆ T := by
    intro x hx
    rw [Finset.mem_image] at hx
    obtain ⟨o, _, rfl⟩ := hx
    exact hm_mem o
  calc 8 = ((Finset.univ : Finset (Option Piece)).image m).card := h_image_card.symm
    _ ≤ T.card := Finset.card_le_card h_image_subset

/-! ### Concrete refutation corollaries

Applying the parametric refutation tools to specific structural invariants
yields ready-to-use impossibility theorems. -/

/-- **Holes-0 atlas impossibility (atlas form).** No closed init-atlas
lives in a universe of holes-0 boards. The S-piece argument
(`one_le_holes_S_applyStep_empty`) supplies the `hpQ` side condition. The
companion theorem in cert form is `no_holes_zero_closed_table_contains_init`. -/
theorem no_init_closed_atlas_in_holes_zero_universe
    {U : Finset GameState}
    (hU : ∀ g ∈ U, holes GameConfig.standard g.board = 0)
    (T : Finset GameState) (hTU : T ⊆ U) (hinitT : GameState.init ∈ T) :
    ¬ IsClosedSet GameConfig.standard T := by
  have hSb : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  refine no_init_closed_atlas_of_predicate_violated (p := Piece.S) hSb
    (Q := fun g => holes GameConfig.standard g.board = 0) hU ?_ T hTU hinitT
  intro pl hpl
  rw [Placement.mem_allValidFor] at hpl
  obtain ⟨hpiece, hvalid⟩ := hpl
  -- Show the successor board has ≥ 1 hole — contradicting `Q = (holes = 0)`.
  have hpl_eq : ({pl with piece := Piece.S} : Placement) = pl := by
    rcases pl with ⟨pp, pr, pc⟩
    cases hpiece
    rfl
  have hsucc_board :
      (adversarialStep GameConfig.standard GameState.init Piece.S pl).board =
        pl.applyStep GameConfig.standard ∅ := by
    rw [adversarialStep_board, GameState.init_board_eq_emptyset, hpl_eq]
  have hone : 1 ≤ holes GameConfig.standard
      (adversarialStep GameConfig.standard GameState.init Piece.S pl).board := by
    rw [hsucc_board]
    exact one_le_holes_S_applyStep_empty (by simp) hvalid hpiece
  omega

/-- **Z-piece variant.** Same as above with Z replacing S. Either of S, Z
suffices since both create holes on the empty board. -/
theorem no_init_closed_atlas_in_holes_zero_universe_via_Z
    {U : Finset GameState}
    (hU : ∀ g ∈ U, holes GameConfig.standard g.board = 0)
    (T : Finset GameState) (hTU : T ⊆ U) (hinitT : GameState.init ∈ T) :
    ¬ IsClosedSet GameConfig.standard T := by
  have hZb : Piece.Z ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.Z
  refine no_init_closed_atlas_of_predicate_violated (p := Piece.Z) hZb
    (Q := fun g => holes GameConfig.standard g.board = 0) hU ?_ T hTU hinitT
  intro pl hpl
  rw [Placement.mem_allValidFor] at hpl
  obtain ⟨hpiece, hvalid⟩ := hpl
  have hpl_eq : ({pl with piece := Piece.Z} : Placement) = pl := by
    rcases pl with ⟨pp, pr, pc⟩
    cases hpiece
    rfl
  have hsucc_board :
      (adversarialStep GameConfig.standard GameState.init Piece.Z pl).board =
        pl.applyStep GameConfig.standard ∅ := by
    rw [adversarialStep_board, GameState.init_board_eq_emptyset, hpl_eq]
  have hone : 1 ≤ holes GameConfig.standard
      (adversarialStep GameConfig.standard GameState.init Piece.Z pl).board := by
    rw [hsucc_board]
    exact one_le_holes_Z_applyStep_empty (by simp) hvalid hpiece
  omega

/-- **Atlas-form impossibility from universe predicates.** Wrapping the
piece-based refutation in a higher-level "TetrisSolvable refutation": if
`init ∈ atlas standard U` BUT every member of `U` has a `Q` violated by
some `init`-bag piece, contradiction. This is the headline form for
plugging in any structural invariant. -/
theorem not_init_mem_atlas_of_init_predicate_violated
    {U : Finset GameState} {p : Piece} (hp : p ∈ GameState.init.bag)
    {Q : GameState → Prop} (hUQ : ∀ g ∈ U, Q g)
    (hpQ : ∀ pl ∈ Placement.allValidFor GameConfig.standard p,
      ¬ Q (adversarialStep GameConfig.standard GameState.init p pl)) :
    GameState.init ∉ atlas GameConfig.standard U := by
  intro hinit
  rcases (init_mem_atlas_iff_exists_closed_init_witness _ _).mp hinit with
    ⟨T, hTU, hinitT, hclosed⟩
  exact no_init_closed_atlas_of_predicate_violated hp hUQ hpQ T hTU hinitT hclosed

/-! ### Composed refutation: TetrisSolvable from any universe refutation -/

/-- **From a universe-level refutation to a TetrisSolvable obstruction.**
If we can establish, for every `U`, that some structural obstruction
violates a closure precondition (e.g., some piece always escapes `U`),
then atlases over those universes don't admit `init`. Used inductively
with widening universe predicates. -/
theorem not_tetrisSolvable_via_atlas_iff
    (H : ∀ U : Finset GameState, GameState.init ∉ atlas GameConfig.standard U) :
    ¬ ∃ U : Finset GameState, GameState.init ∈ atlas GameConfig.standard U := by
  rintro ⟨U, hU⟩
  exact H U hU

/-- **A constructive iff:** TetrisSolvable iff some finite universe admits
`init` in its atlas. Forward direction: from any safe state we can carve
out the orbit (existence — would need finiteness of the orbit, not always
true). Backward direction: trivial via `tetrisSolvable_of_init_mem_atlas`.
We record only the easy direction here and the iff as a target. -/
theorem tetrisSolvable_of_exists_init_atlas
    (h : ∃ U : Finset GameState, GameState.init ∈ atlas GameConfig.standard U) :
    TetrisSolvable :=
  let ⟨U, hU⟩ := h
  tetrisSolvable_of_init_mem_atlas U hU

/-! ### Construction-side composability

These theorems support BUILDING closed atlases incrementally — combining
verified small pieces into larger ones. Together with the maximality
theorem, they let us assemble a global atlas from local subgraphs. -/

/-- **Empty set is vacuously closed.** Useful as a base case for inductive
constructions. -/
theorem IsClosedSet_empty (cfg : GameConfig) :
    IsClosedSet cfg (∅ : Finset GameState) := by
  intro g hg
  exact absurd hg (Finset.notMem_empty g)

/-- **Union of closed sets is closed.** Compositional construction: verify
small closed pieces independently, take their union, get a bigger closed
witness. -/
theorem IsClosedSet_union {cfg : GameConfig} {T₁ T₂ : Finset GameState}
    (h₁ : IsClosedSet cfg T₁) (h₂ : IsClosedSet cfg T₂) :
    IsClosedSet cfg (T₁ ∪ T₂) := by
  intro g hg
  rcases Finset.mem_union.mp hg with hg₁ | hg₂
  · obtain ⟨hnl, hstep⟩ := h₁ g hg₁
    refine ⟨hnl, fun p hp => ?_⟩
    obtain ⟨pl, hpl, hsucc⟩ := hstep p hp
    exact ⟨pl, hpl, Finset.mem_union.mpr (Or.inl hsucc)⟩
  · obtain ⟨hnl, hstep⟩ := h₂ g hg₂
    refine ⟨hnl, fun p hp => ?_⟩
    obtain ⟨pl, hpl, hsucc⟩ := hstep p hp
    exact ⟨pl, hpl, Finset.mem_union.mpr (Or.inr hsucc)⟩

/-- **TetrisSolvable from assembled witnesses.** A union of closed sets
containing `init` proves TetrisSolvable, supplying the cleanest
compositional bridge: each piece can be small enough to verify with
`decide` independently. -/
theorem tetrisSolvable_of_union_of_closed
    {T₁ T₂ : Finset GameState}
    (h₁ : IsClosedSet GameConfig.standard T₁)
    (h₂ : IsClosedSet GameConfig.standard T₂)
    (hinit : GameState.init ∈ T₁ ∪ T₂) :
    TetrisSolvable :=
  tetrisSolvable_of_IsClosedSet (T₁ ∪ T₂) hinit (IsClosedSet_union h₁ h₂)

/-! ### Depth-N forced refutation — the formal sweep-K death theorem

These theorems formalize the Rust empirical "(∅,full) dies at sweep K"
observations. The closed-GFP iteration `iterate cfg U n` removes states
that can't survive `n` adversarial draws. If `init` is removed at any
finite depth `n`, no closed init-atlas in `U` exists — and this fact is
DECIDABLE (`decide` / `native_decide`) on small universes for small `n`.

This decouples refutation from atlas computation: instead of computing
the full fixpoint, you verify a single iteration. -/

/-- **Atlas is contained in every iterate.** Either `n ≤ U.card` (chain
of subsets) or `n > U.card` (stability). In both cases atlas ⊆ iterate. -/
theorem atlas_subset_iterate (cfg : GameConfig) (U : Finset GameState) (n : ℕ) :
    atlas cfg U ⊆ iterate cfg U n := by
  classical
  by_cases hcase : n ≤ U.card
  · -- Descending chain: iterate U U.card ⊆ ... ⊆ iterate U n.
    have hdesc : ∀ m, n ≤ m → iterate cfg U m ⊆ iterate cfg U n := by
      intro m hm
      induction m, hm using Nat.le_induction with
      | base => exact Finset.Subset.refl _
      | succ k _ ih => exact (iterate_subset _ _ _).trans ih
    exact hdesc U.card hcase
  · push Not at hcase
    -- Stability at U.card propagates to iterate U n.
    have heq : iterate cfg U n = iterate cfg U U.card :=
      iterate_stable_from cfg U (closeStep_iterate_card_eq cfg U) (le_of_lt hcase)
    change iterate cfg U U.card ⊆ iterate cfg U n
    rw [heq]

/-- **Depth-N refutation tool.** If `init ∉ iterate cfg U n` at ANY
depth `n`, no closed init-atlas in `U` exists. The CRITICAL practical
form: `decide` (or `native_decide`) on small universes for small `n` can
discharge `init ∉ iterate U n` in seconds, refuting any closed witness
without computing the full atlas. -/
theorem no_init_closed_witness_of_init_not_in_iterate
    {cfg : GameConfig} {U : Finset GameState} {n : ℕ}
    (h : GameState.init ∉ iterate cfg U n)
    (T : Finset GameState) (hsub : T ⊆ U) (hinitT : GameState.init ∈ T)
    (hclosed : IsClosedSet cfg T) :
    False := by
  have hTatlas : T ⊆ atlas cfg U := atlas_maximal cfg U T hsub hclosed
  have hAt : atlas cfg U ⊆ iterate cfg U n := atlas_subset_iterate cfg U n
  exact h (hAt (hTatlas hinitT))

/-- **Existential form of the depth-N refutation.** If there EXISTS some
depth at which `init` is filtered out of `iterate U`, no closed
init-atlas in `U` exists. The Rust closed-GFP records this exact event
("(∅,full) died at sweep K"); this theorem is the formal counterpart. -/
theorem no_init_closed_witness_of_exists_init_not_in_iterate
    {cfg : GameConfig} {U : Finset GameState}
    (h : ∃ n, GameState.init ∉ iterate cfg U n)
    (T : Finset GameState) (hsub : T ⊆ U) (hinitT : GameState.init ∈ T)
    (hclosed : IsClosedSet cfg T) :
    False := by
  obtain ⟨n, hn⟩ := h
  exact no_init_closed_witness_of_init_not_in_iterate hn T hsub hinitT hclosed

/-- **Sweep-N refutation, contrapositive form.** If at depth `n` the
init-removal is established, then `init ∉ atlas U` — the atlas membership
question is settled negatively at finite depth. -/
theorem init_not_mem_atlas_of_init_not_in_iterate
    {cfg : GameConfig} {U : Finset GameState} {n : ℕ}
    (h : GameState.init ∉ iterate cfg U n) :
    GameState.init ∉ atlas cfg U :=
  fun hatlas => h (atlas_subset_iterate cfg U n hatlas)

/-! ### Closed-init-witness API — consolidated TetrisSolvable proof obligation

A `ClosedInitWitness` bundles everything needed for a TetrisSolvable proof
into a single structure: the candidate state set, the init membership
proof, and the closure proof. Users construct one and immediately invoke
`tetrisSolvable` to close the goal. -/

/-- Bundled witness for TetrisSolvable via the atlas framework: a finite
state set containing `init` and closed under standard-config gameplay. -/
structure ClosedInitWitness where
  states : Finset GameState
  init_mem : GameState.init ∈ states
  is_closed : IsClosedSet GameConfig.standard states

/-- Extracting `TetrisSolvable` from a `ClosedInitWitness`. The ONE-CALL
proof obligation: produce a witness, get the headline result. -/
theorem ClosedInitWitness.tetrisSolvable (w : ClosedInitWitness) : TetrisSolvable :=
  tetrisSolvable_of_IsClosedSet w.states w.init_mem w.is_closed

/-- Refutation API — bundle the impossibility obligation. Constructing a
`ClosedInitRefutation` for `U` establishes that no closed init-atlas
exists inside `U`. The constructor takes any depth `n` and a proof
`init ∉ iterate U n` — small `n` makes this cheap to verify by `decide`
or `native_decide`. -/
structure ClosedInitRefutation (cfg : GameConfig) (U : Finset GameState) where
  depth : ℕ
  init_not_in_iterate : GameState.init ∉ iterate cfg U depth

/-- The refutation does what it promises: no closed init-witness in `U`. -/
theorem ClosedInitRefutation.no_witness {cfg : GameConfig} {U : Finset GameState}
    (r : ClosedInitRefutation cfg U)
    (T : Finset GameState) (hsub : T ⊆ U) (hinitT : GameState.init ∈ T)
    (hclosed : IsClosedSet cfg T) :
    False :=
  no_init_closed_witness_of_init_not_in_iterate r.init_not_in_iterate T hsub hinitT hclosed

/-- Refutation from non-membership in the atlas itself. -/
def ClosedInitRefutation.fromAtlas {cfg : GameConfig} (U : Finset GameState)
    (h : GameState.init ∉ atlas cfg U) : ClosedInitRefutation cfg U :=
  { depth := U.card, init_not_in_iterate := h }

/-! ### Cardinality-based refutation + concrete smoke tests

A direct corollary of `atlas_card_ge_eight_of_init_mem`: any candidate
closed init-witness with fewer than 8 states is automatically refuted.
This rules out a large class of trivial candidates in one shot. -/

/-- **Cardinality refutation.** Any candidate `T` containing `init` with
`|T| < 8` cannot be closed. The simplest hand-verifiable refutation —
no structural argument or empirical search required. -/
theorem not_IsClosedSet_of_init_mem_card_lt_eight
    {T : Finset GameState} (hinit : GameState.init ∈ T) (hcard : T.card < 8) :
    ¬ IsClosedSet GameConfig.standard T := fun h => by
  have := atlas_card_ge_eight_of_init_mem GameConfig.standard hinit h
  omega

/-- **Smoke test: singleton-`init` is not closed.** A direct corollary of
the cardinality refutation — `{init}` has 1 < 8 elements. Matches the
∅-reset impossibility (`not_winning_init`) but routed through the atlas
framework. -/
theorem not_IsClosedSet_singleton_init :
    ¬ IsClosedSet GameConfig.standard {GameState.init} :=
  not_IsClosedSet_of_init_mem_card_lt_eight
    (Finset.mem_singleton.mpr rfl)
    (by rw [Finset.card_singleton]; omega)

/-- **Smoke test: any ≤ 7-state init-containing candidate fails.** A
direct instantiation of the cardinality bound. Quickly excludes a huge
swath of small candidates without atlas computation. -/
theorem not_IsClosedSet_of_init_mem_pair {a : GameState} (ha : a ≠ GameState.init) :
    ¬ IsClosedSet GameConfig.standard ({GameState.init, a} : Finset GameState) := by
  apply not_IsClosedSet_of_init_mem_card_lt_eight
  · exact Finset.mem_insert_self _ _
  · rw [Finset.card_insert_of_notMem (by
          simp only [Finset.mem_singleton]; exact Ne.symm ha),
        Finset.card_singleton]
    omega

/-! ### Universe-membership refutation API

When the universe `U` itself does not contain `init` (a basic sanity
check), the refutation is one line. The atlas of `init`-free universes
trivially doesn't contain `init`. -/

/-- If `init ∉ U`, the atlas of `U` is also `init`-free. -/
theorem init_not_mem_atlas_of_init_not_mem_universe
    {cfg : GameConfig} {U : Finset GameState} (h : GameState.init ∉ U) :
    GameState.init ∉ atlas cfg U :=
  fun ha => h (atlas_subset cfg U ha)

/-- Refutation when the universe is `init`-free. -/
theorem no_init_closed_witness_of_init_not_mem_universe
    {U : Finset GameState} (h : GameState.init ∉ U)
    (T : Finset GameState) (hsub : T ⊆ U) (hinitT : GameState.init ∈ T) :
    False := h (hsub hinitT)

/-! ### Sharp cardinality bound — 127 = 2^7 − 1 states minimum

Every closed init-atlas T must contain, for each of the 127 non-empty bag
subsets B ⊆ {O,I,S,Z,T,L,J}, at least one state g ∈ T with g.bag = B.
Reason: the adversarial bag-orbit from init touches every subset by
piece-by-piece draws, and closure forces a T-witness at each step.
Distinct bags ⇒ distinct states, hence |T| ≥ 127. -/

/-- **Bag-orbit coverage.** For every non-empty bag `B`, any closed
init-atlas contains a state with that bag. Proof by induction on
`(Bag.full \ B).card` — pick a piece `p` outside `B`, build `B' = insert p B`,
get a witness for `B'` by IH, then take its `p`-successor (whose bag is
`B'.erase p = B`). -/
theorem exists_state_with_bag_of_closed_init_atlas
    {cfg : GameConfig} {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet cfg T) :
    ∀ B : Bag, B.Nonempty → ∃ g ∈ T, g.bag = B := by
  classical
  have aux : ∀ n B, B.Nonempty → (Bag.full \ B).card ≤ n → ∃ g ∈ T, g.bag = B := by
    intro n
    induction n with
    | zero =>
      intro B hne hle
      have h0 : (Bag.full \ B).card = 0 := Nat.le_zero.mp hle
      have hempty : Bag.full \ B = ∅ := Finset.card_eq_zero.mp h0
      have hfull : B = Bag.full := by
        apply Finset.Subset.antisymm
        · intro p _; exact Bag.mem_full p
        · intro p hp
          by_contra hpB
          exact (Finset.notMem_empty p) (hempty ▸ Finset.mem_sdiff.mpr ⟨hp, hpB⟩)
      refine ⟨GameState.init, hinit, ?_⟩
      rw [GameState.init_bag, hfull]
    | succ k ih =>
      intro B hne hle
      by_cases hzero : (Bag.full \ B).card = 0
      · exact ih B hne (by omega)
      · have hpos : 0 < (Bag.full \ B).card := Nat.pos_of_ne_zero hzero
        obtain ⟨p, hp⟩ := Finset.card_pos.mp hpos
        have hpfull : p ∈ Bag.full := (Finset.mem_sdiff.mp hp).1
        have hpnB : p ∉ B := (Finset.mem_sdiff.mp hp).2
        let B' : Bag := insert p B
        have hB'_ne : B'.Nonempty := ⟨p, Finset.mem_insert_self p B⟩
        have hdiff_eq : Bag.full \ B' = (Bag.full \ B).erase p := by
          ext q
          simp only [B', Finset.mem_sdiff, Finset.mem_erase, Finset.mem_insert]
          tauto
        have hB'_diff_card : (Bag.full \ B').card ≤ k := by
          rw [hdiff_eq, Finset.card_erase_of_mem hp]
          omega
        obtain ⟨g', hg'T, hg'B⟩ := ih B' hB'_ne hB'_diff_card
        have hp_in_g' : p ∈ g'.bag := by
          rw [hg'B]; exact Finset.mem_insert_self p B
        obtain ⟨_, hstep⟩ := hclosed g' hg'T
        obtain ⟨pl, _, hsucc⟩ := hstep p hp_in_g'
        refine ⟨adversarialStep cfg g' p pl, hsucc, ?_⟩
        rw [adversarialStep_bag, hg'B]
        show B'.draw p = B
        unfold Bag.draw
        have herase : (insert p B : Bag).erase p = B := Finset.erase_insert hpnB
        change (if (insert p B : Bag).erase p = ∅ then Bag.full else (insert p B : Bag).erase p) = B
        rw [herase, if_neg hne.ne_empty]
  exact fun B hne => aux 7 B hne (by
    have : (Bag.full \ B).card ≤ Bag.full.card := Finset.card_le_card Finset.sdiff_subset
    rw [Bag.full_card] at this
    exact this)

/-- **Sharp cardinality bound: |T| ≥ 127.** Any closed init-atlas contains
at least 127 distinct states — one for each non-empty bag subset
(2^7 − 1 = 127). Refutes by computation every candidate of size < 127. -/
theorem closed_init_atlas_card_ge_127 {cfg : GameConfig} {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet cfg T) :
    127 ≤ T.card := by
  classical
  -- The 127 non-empty subsets of `Bag.full`.
  set NE : Finset Bag := Bag.full.powerset.erase ∅
  have hNE_card : NE.card = 127 := by
    have hmem : (∅ : Bag) ∈ Bag.full.powerset := Finset.empty_mem_powerset _
    rw [show NE = Bag.full.powerset.erase ∅ from rfl,
        Finset.card_erase_of_mem hmem, Finset.card_powerset, Bag.full_card]
    decide
  have hNE_to_Ne : ∀ B ∈ NE, B.Nonempty := by
    intro B hB
    rw [show NE = Bag.full.powerset.erase ∅ from rfl,
        Finset.mem_erase, Finset.mem_powerset] at hB
    exact Finset.nonempty_iff_ne_empty.mpr hB.1
  -- Map NE.attach to T via Classical.choose
  let g : { B : Bag // B ∈ NE } → GameState := fun ⟨B, hB⟩ =>
    Classical.choose (exists_state_with_bag_of_closed_init_atlas hinit hclosed B (hNE_to_Ne B hB))
  have hgT : ∀ x : { B : Bag // B ∈ NE }, g x ∈ T := fun ⟨B, hB⟩ =>
    (Classical.choose_spec (exists_state_with_bag_of_closed_init_atlas hinit hclosed B (hNE_to_Ne B hB))).1
  have hgb : ∀ x : { B : Bag // B ∈ NE }, (g x).bag = x.val := fun ⟨B, hB⟩ =>
    (Classical.choose_spec (exists_state_with_bag_of_closed_init_atlas hinit hclosed B (hNE_to_Ne B hB))).2
  -- Image of NE.attach under g.
  let img : Finset GameState := NE.attach.image g
  have himg_sub : img ⊆ T := by
    intro y hy
    rw [Finset.mem_image] at hy
    obtain ⟨x, _, rfl⟩ := hy
    exact hgT x
  have hg_inj : Function.Injective g := by
    rintro ⟨B1, hB1⟩ ⟨B2, hB2⟩ heq
    apply Subtype.ext
    have hbag : (g ⟨B1, hB1⟩).bag = (g ⟨B2, hB2⟩).bag := congrArg _ heq
    rw [hgb ⟨B1, hB1⟩, hgb ⟨B2, hB2⟩] at hbag
    exact hbag
  have himg_card : img.card = NE.card := by
    change (NE.attach.image g).card = NE.card
    rw [Finset.card_image_of_injective _ hg_inj, Finset.card_attach]
  calc 127 = NE.card := hNE_card.symm
    _ = img.card := himg_card.symm
    _ ≤ T.card := Finset.card_le_card himg_sub

/-- **Cardinality refutation, sharp form.** Any candidate `T` containing
`init` with `|T| < 127` cannot be closed. Refutes ALL atlas candidates of
size < 127 in one shot — including everything previously refuted by the
≥ 8 bound and much more (eg. anything up to 126 states). -/
theorem not_IsClosedSet_of_init_mem_card_lt_127
    {T : Finset GameState} (hinit : GameState.init ∈ T) (hcard : T.card < 127) :
    ¬ IsClosedSet GameConfig.standard T := fun h => by
  have := closed_init_atlas_card_ge_127 hinit h
  omega

/-! ### Per-bag-size structural breakdown

The 127 states from `closed_init_atlas_card_ge_127` are not evenly
distributed — they cluster by bag size. For each `k ∈ {1, …, 7}`, any
closed init-atlas contains at least `C(7, k)` states whose bag has
exactly `k` pieces. The breakdown:

| k | C(7,k) | meaning                       |
|---|--------|-------------------------------|
| 1 | 7      | singleton-bag states          |
| 2 | 21     | 2-element bag states          |
| 3 | 35     | 3-element bag states          |
| 4 | 35     | 4-element bag states          |
| 5 | 21     | 5-element bag states          |
| 6 | 7      | 6-element bag states          |
| 7 | 1      | full-bag states (≥ init)      |

Sum: 7 + 21 + 35 + 35 + 21 + 7 + 1 = 127. Every closed init-atlas has
this exact lower bound structure — a strong invariant. -/

/-- **Per-bag-size lower bound.** For every `k ∈ {1, …, 7}`, any closed
init-atlas contains at least `Nat.choose 7 k` states whose bag has size
`k` exactly. Proves the cardinality distribution above. -/
theorem closed_init_atlas_card_by_bag_size
    {cfg : GameConfig} {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet cfg T)
    (k : ℕ) (hk : 1 ≤ k) (hk7 : k ≤ 7) :
    Nat.choose 7 k ≤ (T.filter (fun g => g.bag.card = k)).card := by
  classical
  -- The bags of size k.
  set NEk : Finset Bag := Bag.full.powersetCard k
  have hNEk_card : NEk.card = Nat.choose 7 k := by
    rw [show NEk = Bag.full.powersetCard k from rfl,
        Finset.card_powersetCard, Bag.full_card]
  have hNEk_to_Ne : ∀ B ∈ NEk, B.Nonempty := by
    intro B hB
    rw [show NEk = Bag.full.powersetCard k from rfl,
        Finset.mem_powersetCard] at hB
    exact Finset.card_pos.mp (hB.2 ▸ hk)
  -- Map NEk.attach to filtered T via the bag-coverage theorem.
  let g : { B : Bag // B ∈ NEk } → GameState := fun ⟨B, hB⟩ =>
    Classical.choose
      (exists_state_with_bag_of_closed_init_atlas hinit hclosed B (hNEk_to_Ne B hB))
  have hgT : ∀ x : { B : Bag // B ∈ NEk }, g x ∈ T := fun ⟨B, hB⟩ =>
    (Classical.choose_spec
      (exists_state_with_bag_of_closed_init_atlas hinit hclosed B (hNEk_to_Ne B hB))).1
  have hgb : ∀ x : { B : Bag // B ∈ NEk }, (g x).bag = x.val := fun ⟨B, hB⟩ =>
    (Classical.choose_spec
      (exists_state_with_bag_of_closed_init_atlas hinit hclosed B (hNEk_to_Ne B hB))).2
  have hgbcard : ∀ x : { B : Bag // B ∈ NEk }, (g x).bag.card = k := by
    intro ⟨B, hB⟩
    rw [hgb ⟨B, hB⟩]
    rw [show NEk = Bag.full.powersetCard k from rfl, Finset.mem_powersetCard] at hB
    exact hB.2
  -- Image lands in `T.filter (fun g => g.bag.card = k)`.
  have himg_sub : NEk.attach.image g ⊆ T.filter (fun g => g.bag.card = k) := by
    intro y hy
    rw [Finset.mem_image] at hy
    obtain ⟨x, _, rfl⟩ := hy
    rw [Finset.mem_filter]
    exact ⟨hgT x, hgbcard x⟩
  have hg_inj : Function.Injective g := by
    rintro ⟨B1, hB1⟩ ⟨B2, hB2⟩ heq
    apply Subtype.ext
    have hbag : (g ⟨B1, hB1⟩).bag = (g ⟨B2, hB2⟩).bag := congrArg _ heq
    rw [hgb ⟨B1, hB1⟩, hgb ⟨B2, hB2⟩] at hbag
    exact hbag
  have himg_card : (NEk.attach.image g).card = NEk.card := by
    rw [Finset.card_image_of_injective _ hg_inj, Finset.card_attach]
  calc Nat.choose 7 k = NEk.card := hNEk_card.symm
    _ = (NEk.attach.image g).card := himg_card.symm
    _ ≤ (T.filter (fun g => g.bag.card = k)).card := Finset.card_le_card himg_sub

/-! ### Cell-count residue distribution — the 5 phases revealed

Every closed init-atlas contains states at all 5 residue classes mod 10:
{0, 4, 8, 2, 6}. Reason: from `init` (residue 0), depth-n forced reach
yields a state with cell-count residue = (4n) mod 10, which cycles
through all 5 even residues as n goes from 0 to 4 (since gcd(4, 10) = 2).
Closure ensures each such state is in T.

This is the structural footprint of the M3 5-cyclic phases inside any
closed init-atlas. -/

/-- **Forced-depth reach inside a closed init-atlas.** At every depth `n`,
some `ReachableAt`-witness of T-membership exists. Built by induction on
`n` using closure to pick a piece-placement at each step. -/
theorem exists_reachableAt_in_closed_init_atlas {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T) :
    ∀ n : ℕ, ∃ g, ReachableAt GameConfig.standard n g ∧ g ∈ T := by
  intro n
  induction n with
  | zero =>
    exact ⟨GameState.init, ReachableAt.init, hinit⟩
  | succ k ih =>
    obtain ⟨g, hreach, hgT⟩ := ih
    obtain ⟨hnl, hstep⟩ := hclosed g hgT
    have hbag_ne : g.bag.Nonempty := by
      have hcard := ReachableAt.bag_card_eq hreach
      have h7 : k % 7 < 7 := Nat.mod_lt k (by decide)
      apply Finset.card_pos.mp
      rw [hcard]
      omega
    obtain ⟨p, hp⟩ := hbag_ne
    obtain ⟨pl, hpl, hsucc⟩ := hstep p hp
    rw [Placement.mem_allValidFor] at hpl
    obtain ⟨hpiece, hvalid⟩ := hpl
    have hpl_eq : ({pl with piece := p} : Placement) = pl := by
      rcases pl with ⟨pp, pr, pc⟩
      cases hpiece
      rfl
    have hvalid' : ({pl with piece := p} : Placement).Valid GameConfig.standard := by
      rw [hpl_eq]; exact hvalid
    exact ⟨adversarialStep GameConfig.standard g p pl,
           ReachableAt.step hreach hnl hp hvalid', hsucc⟩

/-- **Residue at depth `n`.** Closed init-atlases contain a state whose
board cell count has residue `(4n) mod 10`. Direct application of
`standard_count_mod_ten` to a depth-`n` `ReachableAt`-witness. -/
theorem exists_state_at_residue_in_closed_init_atlas {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T)
    (n : ℕ) :
    ∃ g ∈ T, g.board.count % 10 = (4 * n) % 10 := by
  obtain ⟨g, hreach, hgT⟩ := exists_reachableAt_in_closed_init_atlas hinit hclosed n
  exact ⟨g, hgT, ReachableAt.standard_count_mod_ten hreach⟩

/-- **All 5 residue classes covered.** Closed init-atlases contain states
at each of `{0, 4, 8, 2, 6}` — every even residue mod 10. -/
theorem closed_init_atlas_covers_all_residues {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T) :
    (∃ g ∈ T, g.board.count % 10 = 0) ∧
    (∃ g ∈ T, g.board.count % 10 = 4) ∧
    (∃ g ∈ T, g.board.count % 10 = 8) ∧
    (∃ g ∈ T, g.board.count % 10 = 2) ∧
    (∃ g ∈ T, g.board.count % 10 = 6) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · have := exists_state_at_residue_in_closed_init_atlas hinit hclosed 0
    simpa using this
  · have := exists_state_at_residue_in_closed_init_atlas hinit hclosed 1
    simpa using this
  · have := exists_state_at_residue_in_closed_init_atlas hinit hclosed 2
    simpa using this
  · have := exists_state_at_residue_in_closed_init_atlas hinit hclosed 3
    simpa using this
  · have := exists_state_at_residue_in_closed_init_atlas hinit hclosed 4
    simpa using this

/-! ### Sharper: multiple full-bag states

The per-bag-size bound at `k = 7` is `C(7, 7) = 1`. But the depth-7
forced reach yields a SECOND full-bag state (with residue 8 ≠ 0,
distinct from init). So the count is actually ≥ 2 — and in general
≥ ⌈n / 7⌉ + 1 for any forced-reach depth `7n`. -/

/-- **At least two distinct full-bag states.** Any closed init-atlas
contains both `init` (residue 0) and the depth-7 forced-reach state
(residue 8). Strictly sharper than `C(7, 7) = 1` from the bag-size
breakdown. -/
theorem closed_init_atlas_has_two_full_bag_states
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    2 ≤ (T.filter (fun g => g.bag = Bag.full)).card := by
  classical
  obtain ⟨g7, hreach7, hg7T⟩ := exists_reachableAt_in_closed_init_atlas hinit hclosed 7
  have hg7_card : g7.bag.card = 7 := by
    have := ReachableAt.bag_card_eq hreach7
    norm_num at this
    exact this
  have hg7_bag : g7.bag = Bag.full := by
    have hsub : g7.bag ⊆ Bag.full := fun p _ => Bag.mem_full p
    exact Finset.eq_of_subset_of_card_le hsub (by rw [hg7_card, Bag.full_card])
  have hg7_res : g7.board.count % 10 = 8 := by
    have := ReachableAt.standard_count_mod_ten hreach7
    rw [this]
  have hinit_res : GameState.init.board.count % 10 = 0 := by
    rw [GameState.init_board_count]
  have hne : GameState.init ≠ g7 := by
    intro heq
    rw [heq, hg7_res] at hinit_res
    omega
  have hinit_in : GameState.init ∈ T.filter (fun g => g.bag = Bag.full) :=
    Finset.mem_filter.mpr ⟨hinit, GameState.init_bag⟩
  have hg7_in : g7 ∈ T.filter (fun g => g.bag = Bag.full) :=
    Finset.mem_filter.mpr ⟨hg7T, hg7_bag⟩
  have hsub : ({GameState.init, g7} : Finset GameState) ⊆
      T.filter (fun g => g.bag = Bag.full) := by
    intro x hx
    rcases Finset.mem_insert.mp hx with rfl | hxs
    · exact hinit_in
    · rw [Finset.mem_singleton] at hxs; rw [hxs]; exact hg7_in
  have hpaircard : ({GameState.init, g7} : Finset GameState).card = 2 := by
    rw [Finset.card_insert_of_notMem
        (by rw [Finset.mem_singleton]; exact hne), Finset.card_singleton]
  calc 2 = ({GameState.init, g7} : Finset GameState).card := hpaircard.symm
    _ ≤ _ := Finset.card_le_card hsub

/-- **Depth-`7n` full-bag witness.** At every depth `7n`, the
forced-reach witness has a FULL bag. Combined with the residue theorem,
this gives full-bag states with cell-count residue `(28n) mod 10`. -/
theorem exists_full_bag_at_depth_seven_mul
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) (n : ℕ) :
    ∃ g ∈ T, ReachableAt GameConfig.standard (7 * n) g ∧
      g.bag = Bag.full ∧ g.board.count % 10 = (28 * n) % 10 := by
  obtain ⟨g, hreach, hgT⟩ :=
    exists_reachableAt_in_closed_init_atlas hinit hclosed (7 * n)
  refine ⟨g, hgT, hreach, ?_, ?_⟩
  · have hcard := ReachableAt.bag_card_eq hreach
    have h7n : (7 * n) % 7 = 0 := by omega
    rw [h7n] at hcard
    have hsub : g.bag ⊆ Bag.full := fun p _ => Bag.mem_full p
    exact Finset.eq_of_subset_of_card_le hsub (by rw [hcard, Bag.full_card])
  · have := ReachableAt.standard_count_mod_ten hreach
    rw [this]; ring_nf

/-! ### Bag-orbit with ReachableAt — controlling both bag and residue

The bag-orbit theorem `exists_state_with_bag_of_closed_init_atlas` gives a
witness for every non-empty bag. The version below ALSO tracks the
`ReachableAt` depth — specifically, the witness for bag `B` lives at
depth `7 - B.card`, which by `standard_count_mod_ten` fixes the
cell-count residue to `(4 * (7 - B.card)) % 10`. This lets us control
BOTH bag and residue together. -/

/-- **Strengthened bag-orbit: bag + depth witness.** For each non-empty
bag `B`, the closed init-atlas contains a state with that bag at depth
`7 - B.card`. Direct corollary: the state's cell-count residue is
`(4 * (7 - B.card)) % 10`. -/
theorem exists_state_with_bag_and_reachable_at
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    ∀ B : Bag, B.Nonempty →
      ∃ g ∈ T, g.bag = B ∧ ReachableAt GameConfig.standard (7 - B.card) g := by
  classical
  have aux : ∀ n B, B.Nonempty → (Bag.full \ B).card ≤ n →
      ∃ g ∈ T, g.bag = B ∧ ReachableAt GameConfig.standard (7 - B.card) g := by
    intro n
    induction n with
    | zero =>
      intro B hne hle
      have h0 : (Bag.full \ B).card = 0 := Nat.le_zero.mp hle
      have hempty : Bag.full \ B = ∅ := Finset.card_eq_zero.mp h0
      have hfull : B = Bag.full := by
        apply Finset.Subset.antisymm
        · intro p _; exact Bag.mem_full p
        · intro p hp
          by_contra hpB
          exact (Finset.notMem_empty p) (hempty ▸ Finset.mem_sdiff.mpr ⟨hp, hpB⟩)
      refine ⟨GameState.init, hinit, ?_, ?_⟩
      · rw [GameState.init_bag, hfull]
      · have : 7 - B.card = 0 := by rw [hfull, Bag.full_card]
        rw [this]
        exact ReachableAt.init
    | succ k ih =>
      intro B hne hle
      by_cases hzero : (Bag.full \ B).card = 0
      · exact ih B hne (by omega)
      · have hpos : 0 < (Bag.full \ B).card := Nat.pos_of_ne_zero hzero
        obtain ⟨p, hp⟩ := Finset.card_pos.mp hpos
        have hpfull : p ∈ Bag.full := (Finset.mem_sdiff.mp hp).1
        have hpnB : p ∉ B := (Finset.mem_sdiff.mp hp).2
        let B' : Bag := insert p B
        have hB'_ne : B'.Nonempty := ⟨p, Finset.mem_insert_self p B⟩
        have hdiff_eq : Bag.full \ B' = (Bag.full \ B).erase p := by
          ext q
          simp only [B', Finset.mem_sdiff, Finset.mem_erase, Finset.mem_insert]
          tauto
        have hB'_diff_card : (Bag.full \ B').card ≤ k := by
          rw [hdiff_eq, Finset.card_erase_of_mem hp]; omega
        obtain ⟨g', hg'T, hg'B, hg'reach⟩ := ih B' hB'_ne hB'_diff_card
        have hp_in_g' : p ∈ g'.bag := by
          rw [hg'B]; exact Finset.mem_insert_self p B
        obtain ⟨hnl, hstep⟩ := hclosed g' hg'T
        obtain ⟨pl, hpl, hsucc⟩ := hstep p hp_in_g'
        rw [Placement.mem_allValidFor] at hpl
        obtain ⟨hpiece, hvalid⟩ := hpl
        have hpl_eq : ({pl with piece := p} : Placement) = pl := by
          rcases pl with ⟨pp, pr, pc⟩
          cases hpiece
          rfl
        have hvalid' : ({pl with piece := p} : Placement).Valid GameConfig.standard := by
          rw [hpl_eq]; exact hvalid
        refine ⟨adversarialStep GameConfig.standard g' p pl, hsucc, ?_, ?_⟩
        · rw [adversarialStep_bag, hg'B]
          show B'.draw p = B
          unfold Bag.draw
          have herase : (insert p B : Bag).erase p = B := Finset.erase_insert hpnB
          change (if (insert p B : Bag).erase p = ∅ then Bag.full
                  else (insert p B : Bag).erase p) = B
          rw [herase, if_neg hne.ne_empty]
        · -- Depth tracking: 7 - B.card = (7 - B'.card) + 1
          have hB'_card : B'.card = B.card + 1 := by
            rw [Finset.card_insert_of_notMem hpnB]
          have hdepth_eq : 7 - B.card = (7 - B'.card) + 1 := by
            have : B.card ≤ 6 := by
              by_contra hgt
              push Not at hgt
              have hsub : B ⊆ Bag.full := fun p _ => Bag.mem_full p
              have hcardle : B.card ≤ Bag.full.card := Finset.card_le_card hsub
              rw [Bag.full_card] at hcardle
              have hcardeq : B.card = 7 := by omega
              have hBfull : B = Bag.full := Finset.eq_of_subset_of_card_le hsub
                (by rw [hcardeq, Bag.full_card])
              rw [hBfull] at hpnB
              exact hpnB hpfull
            omega
          rw [hdepth_eq]
          exact ReachableAt.step hg'reach hnl hp_in_g' hvalid'
  intro B hne
  exact aux 7 B hne (by
    have : (Bag.full \ B).card ≤ Bag.full.card := Finset.card_le_card Finset.sdiff_subset
    rw [Bag.full_card] at this
    exact this)

/-- **Bag + residue corollary.** For each non-empty bag `B`, the closed
init-atlas contains a state with that bag AND cell-count residue
`(28 - 4 * B.card) % 10`. Combines `exists_state_with_bag_and_reachable_at`
with `standard_count_mod_ten`. -/
theorem exists_state_with_bag_and_residue
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    ∀ B : Bag, B.Nonempty →
      ∃ g ∈ T, g.bag = B ∧ g.board.count % 10 = (4 * (7 - B.card)) % 10 := by
  intro B hne
  obtain ⟨g, hgT, hgB, hgreach⟩ :=
    exists_state_with_bag_and_reachable_at hinit hclosed B hne
  exact ⟨g, hgT, hgB, ReachableAt.standard_count_mod_ten hgreach⟩

/-- **Parametric bag-orbit from any depth-7m full-bag start.** Generalizes
`exists_state_with_bag_and_reachable_at` to start from ANY full-bag witness
at some `start_depth`. Specializing to depth-`7m` full-bag witnesses (from
`exists_full_bag_at_depth_seven_mul`) gives the 5 cycles needed for the
sharp |T_{size=k}| ≥ 5 × C(7, k) bound. -/
theorem exists_state_with_bag_and_reachable_at_from
    {T : Finset GameState}
    (hclosed : IsClosedSet GameConfig.standard T)
    (start : GameState) (hstart_T : start ∈ T) (start_depth : ℕ)
    (hstart_reach : ReachableAt GameConfig.standard start_depth start)
    (hstart_bag : start.bag = Bag.full) :
    ∀ B : Bag, B.Nonempty →
      ∃ g ∈ T, g.bag = B ∧
        ReachableAt GameConfig.standard (start_depth + (7 - B.card)) g := by
  classical
  have aux : ∀ n B, B.Nonempty → (Bag.full \ B).card ≤ n →
      ∃ g ∈ T, g.bag = B ∧
        ReachableAt GameConfig.standard (start_depth + (7 - B.card)) g := by
    intro n
    induction n with
    | zero =>
      intro B hne hle
      have h0 : (Bag.full \ B).card = 0 := Nat.le_zero.mp hle
      have hempty : Bag.full \ B = ∅ := Finset.card_eq_zero.mp h0
      have hfull : B = Bag.full := by
        apply Finset.Subset.antisymm
        · intro p _; exact Bag.mem_full p
        · intro p hp
          by_contra hpB
          exact (Finset.notMem_empty p) (hempty ▸ Finset.mem_sdiff.mpr ⟨hp, hpB⟩)
      refine ⟨start, hstart_T, ?_, ?_⟩
      · rw [hstart_bag, hfull]
      · have : 7 - B.card = 0 := by rw [hfull, Bag.full_card]
        rw [this, Nat.add_zero]
        exact hstart_reach
    | succ k ih =>
      intro B hne hle
      by_cases hzero : (Bag.full \ B).card = 0
      · exact ih B hne (by omega)
      · have hpos : 0 < (Bag.full \ B).card := Nat.pos_of_ne_zero hzero
        obtain ⟨p, hp⟩ := Finset.card_pos.mp hpos
        have hpfull : p ∈ Bag.full := (Finset.mem_sdiff.mp hp).1
        have hpnB : p ∉ B := (Finset.mem_sdiff.mp hp).2
        let B' : Bag := insert p B
        have hB'_ne : B'.Nonempty := ⟨p, Finset.mem_insert_self p B⟩
        have hdiff_eq : Bag.full \ B' = (Bag.full \ B).erase p := by
          ext q
          simp only [B', Finset.mem_sdiff, Finset.mem_erase, Finset.mem_insert]
          tauto
        have hB'_diff_card : (Bag.full \ B').card ≤ k := by
          rw [hdiff_eq, Finset.card_erase_of_mem hp]; omega
        obtain ⟨g', hg'T, hg'B, hg'reach⟩ := ih B' hB'_ne hB'_diff_card
        have hp_in_g' : p ∈ g'.bag := by
          rw [hg'B]; exact Finset.mem_insert_self p B
        obtain ⟨hnl, hstep⟩ := hclosed g' hg'T
        obtain ⟨pl, hpl, hsucc⟩ := hstep p hp_in_g'
        rw [Placement.mem_allValidFor] at hpl
        obtain ⟨hpiece, hvalid⟩ := hpl
        have hpl_eq : ({pl with piece := p} : Placement) = pl := by
          rcases pl with ⟨pp, pr, pc⟩
          cases hpiece
          rfl
        have hvalid' : ({pl with piece := p} : Placement).Valid GameConfig.standard := by
          rw [hpl_eq]; exact hvalid
        refine ⟨adversarialStep GameConfig.standard g' p pl, hsucc, ?_, ?_⟩
        · rw [adversarialStep_bag, hg'B]
          show B'.draw p = B
          unfold Bag.draw
          have herase : (insert p B : Bag).erase p = B := Finset.erase_insert hpnB
          change (if (insert p B : Bag).erase p = ∅ then Bag.full
                  else (insert p B : Bag).erase p) = B
          rw [herase, if_neg hne.ne_empty]
        · have hB'_card : B'.card = B.card + 1 := by
            rw [Finset.card_insert_of_notMem hpnB]
          have hdepth_eq : start_depth + (7 - B.card) =
              (start_depth + (7 - B'.card)) + 1 := by
            have : B.card ≤ 6 := by
              by_contra hgt
              push Not at hgt
              have hsub : B ⊆ Bag.full := fun p _ => Bag.mem_full p
              have hcardle : B.card ≤ Bag.full.card := Finset.card_le_card hsub
              rw [Bag.full_card] at hcardle
              have hcardeq : B.card = 7 := by omega
              have hBfull : B = Bag.full :=
                Finset.eq_of_subset_of_card_le hsub (by rw [hcardeq, Bag.full_card])
              rw [hBfull] at hpnB
              exact hpnB hpfull
            omega
          rw [hdepth_eq]
          exact ReachableAt.step hg'reach hnl hp_in_g' hvalid'
  intro B hne
  exact aux 7 B hne (by
    have : (Bag.full \ B).card ≤ Bag.full.card := Finset.card_le_card Finset.sdiff_subset
    rw [Bag.full_card] at this
    exact this)

/-- **Bag + residue at any cycle m.** For each `m ∈ ℕ` and non-empty
bag `B`, the closed init-atlas contains a state with that bag AND
cell-count residue `(4 * (7m + 7 - B.card)) % 10`. -/
theorem exists_state_with_bag_and_residue_at_cycle
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) (m : ℕ) :
    ∀ B : Bag, B.Nonempty →
      ∃ g ∈ T, g.bag = B ∧
        g.board.count % 10 = (4 * (7 * m + 7 - B.card)) % 10 := by
  intro B hne
  obtain ⟨start, hstart_T, hstart_reach, hstart_bag, _⟩ :=
    exists_full_bag_at_depth_seven_mul hinit hclosed m
  obtain ⟨g, hgT, hgB, hgreach⟩ :=
    exists_state_with_bag_and_reachable_at_from hclosed start hstart_T (7*m)
      hstart_reach hstart_bag B hne
  refine ⟨g, hgT, hgB, ?_⟩
  have hres := ReachableAt.standard_count_mod_ten hgreach
  rw [hres]
  have hB7 : B.card ≤ 7 := by
    have hsub : B ⊆ Bag.full := fun p _ => Bag.mem_full p
    have := Finset.card_le_card hsub
    rw [Bag.full_card] at this
    exact this
  congr 1
  omega

set_option maxHeartbeats 4000000 in
/-- **Sharp per-bag-size bound: |T_{size=k}| ≥ 5 × C(7, k).** Combines
bag-orbit (one witness per bag) with the 5-cycle residue structure to
get 5 distinct states per bag — one for each residue class mod 10.
Sums to |T| ≥ 5 × 127 = 635 across all bag sizes. -/
theorem closed_init_atlas_card_by_bag_size_sharp
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) (k : ℕ) (hk : 1 ≤ k) (hk7 : k ≤ 7) :
    5 * Nat.choose 7 k ≤ (T.filter (fun g => g.bag.card = k)).card := by
  classical
  set NEk : Finset Bag := Bag.full.powersetCard k
  have hNEk_card : NEk.card = Nat.choose 7 k := by
    rw [show NEk = Bag.full.powersetCard k from rfl,
        Finset.card_powersetCard, Bag.full_card]
  have hNEk_to_Ne : ∀ B ∈ NEk, B.Nonempty := by
    intro B hB
    rw [show NEk = Bag.full.powersetCard k from rfl, Finset.mem_powersetCard] at hB
    exact Finset.card_pos.mp (hB.2 ▸ hk)
  have hNEk_card_eq : ∀ B ∈ NEk, B.card = k := by
    intro B hB
    rw [show NEk = Bag.full.powersetCard k from rfl, Finset.mem_powersetCard] at hB
    exact hB.2
  let g : { B : Bag // B ∈ NEk } × Fin 5 → GameState := fun ⟨⟨B, hB⟩, m⟩ =>
    Classical.choose
      (exists_state_with_bag_and_residue_at_cycle hinit hclosed m.val B
        (hNEk_to_Ne B hB))
  have hgT : ∀ x, g x ∈ T := fun ⟨⟨B, hB⟩, m⟩ =>
    (Classical.choose_spec
      (exists_state_with_bag_and_residue_at_cycle hinit hclosed m.val B
        (hNEk_to_Ne B hB))).1
  have hgb : ∀ x, (g x).bag = x.1.val := fun ⟨⟨B, hB⟩, m⟩ =>
    (Classical.choose_spec
      (exists_state_with_bag_and_residue_at_cycle hinit hclosed m.val B
        (hNEk_to_Ne B hB))).2.1
  have hgr : ∀ x, (g x).board.count % 10 =
      (4 * (7 * x.2.val + 7 - x.1.val.card)) % 10 := fun ⟨⟨B, hB⟩, m⟩ =>
    (Classical.choose_spec
      (exists_state_with_bag_and_residue_at_cycle hinit hclosed m.val B
        (hNEk_to_Ne B hB))).2.2
  have himg_sub : (Finset.univ.image g) ⊆ T.filter (fun g => g.bag.card = k) := by
    intro y hy
    rw [Finset.mem_image] at hy
    obtain ⟨x, _, rfl⟩ := hy
    rw [Finset.mem_filter]
    refine ⟨hgT x, ?_⟩
    rw [hgb x]
    exact hNEk_card_eq x.1.val x.1.property
  have hg_inj : Function.Injective g := by
    rintro ⟨⟨B1, hB1⟩, m1⟩ ⟨⟨B2, hB2⟩, m2⟩ heq
    have hbag : (g ⟨⟨B1, hB1⟩, m1⟩).bag = (g ⟨⟨B2, hB2⟩, m2⟩).bag := by rw [heq]
    rw [hgb ⟨⟨B1, hB1⟩, m1⟩, hgb ⟨⟨B2, hB2⟩, m2⟩] at hbag
    have hres : (g ⟨⟨B1, hB1⟩, m1⟩).board.count % 10 =
                (g ⟨⟨B2, hB2⟩, m2⟩).board.count % 10 := by rw [heq]
    rw [hgr ⟨⟨B1, hB1⟩, m1⟩, hgr ⟨⟨B2, hB2⟩, m2⟩] at hres
    have hk1 : B1.card = k := hNEk_card_eq B1 hB1
    have hk2 : B2.card = k := hNEk_card_eq B2 hB2
    rw [hk1, hk2] at hres
    have hmeq : m1.val = m2.val := by
      have hi1 : m1.val < 5 := m1.isLt
      have hi2 : m2.val < 5 := m2.isLt
      interval_cases k <;> interval_cases m1.val <;> interval_cases m2.val <;>
        first | rfl | omega
    exact Prod.ext (Subtype.ext hbag) (Fin.ext hmeq)
  have himg_card : (Finset.univ.image g).card = 5 * Nat.choose 7 k := by
    rw [Finset.card_image_of_injective _ hg_inj, Finset.card_univ]
    rw [Fintype.card_prod, Fintype.card_coe, hNEk_card, Fintype.card_fin]
    ring
  calc 5 * Nat.choose 7 k = (Finset.univ.image g).card := himg_card.symm
    _ ≤ _ := Finset.card_le_card himg_sub

/-- **|T| ≥ 635.** Sharp total cardinality bound: any closed init-atlas
has at least 635 distinct states. Sums the per-bag-size sharp bound
`5 × C(7, k)` for `k ∈ {1, …, 7}`. Reason: `5 × ∑ₖ C(7, k) = 5 × 127 =
635`. -/
theorem closed_init_atlas_card_ge_635
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    635 ≤ T.card := by
  classical
  have hdisj : ∀ k₁ k₂ : ℕ, k₁ ≠ k₂ →
      Disjoint (T.filter (fun g => g.bag.card = k₁))
               (T.filter (fun g => g.bag.card = k₂)) := by
    intro k₁ k₂ hne
    rw [Finset.disjoint_filter]
    intro g _ hg1 hg2
    rw [hg1] at hg2
    exact hne hg2
  set Ks : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
  have hbiUnion_sub : Ks.biUnion (fun k => T.filter (fun g => g.bag.card = k)) ⊆ T := by
    intro x hx
    rw [Finset.mem_biUnion] at hx
    obtain ⟨_, _, hx⟩ := hx
    exact (Finset.mem_filter.mp hx).1
  have hsum_eq :
      (Ks.biUnion (fun k => T.filter (fun g => g.bag.card = k))).card =
        ∑ k ∈ Ks, (T.filter (fun g => g.bag.card = k)).card := by
    rw [Finset.card_biUnion (fun k1 _ k2 _ hne => hdisj k1 k2 hne)]
  have h1 := closed_init_atlas_card_by_bag_size_sharp hinit hclosed 1
    (by omega) (by omega)
  have h2 := closed_init_atlas_card_by_bag_size_sharp hinit hclosed 2
    (by omega) (by omega)
  have h3 := closed_init_atlas_card_by_bag_size_sharp hinit hclosed 3
    (by omega) (by omega)
  have h4 := closed_init_atlas_card_by_bag_size_sharp hinit hclosed 4
    (by omega) (by omega)
  have h5 := closed_init_atlas_card_by_bag_size_sharp hinit hclosed 5
    (by omega) (by omega)
  have h6 := closed_init_atlas_card_by_bag_size_sharp hinit hclosed 6
    (by omega) (by omega)
  have h7 := closed_init_atlas_card_by_bag_size_sharp hinit hclosed 7
    (by omega) (by omega)
  rw [show Nat.choose 7 1 = 7 from rfl] at h1
  rw [show Nat.choose 7 2 = 21 from rfl] at h2
  rw [show Nat.choose 7 3 = 35 from rfl] at h3
  rw [show Nat.choose 7 4 = 35 from rfl] at h4
  rw [show Nat.choose 7 5 = 21 from rfl] at h5
  rw [show Nat.choose 7 6 = 7 from rfl] at h6
  rw [show Nat.choose 7 7 = 1 from rfl] at h7
  have hsumK : ∑ k ∈ Ks, (T.filter (fun g => g.bag.card = k)).card =
      (T.filter (fun g => g.bag.card = 1)).card +
      (T.filter (fun g => g.bag.card = 2)).card +
      (T.filter (fun g => g.bag.card = 3)).card +
      (T.filter (fun g => g.bag.card = 4)).card +
      (T.filter (fun g => g.bag.card = 5)).card +
      (T.filter (fun g => g.bag.card = 6)).card +
      (T.filter (fun g => g.bag.card = 7)).card := by
    rw [show Ks = ({1, 2, 3, 4, 5, 6, 7} : Finset ℕ) from rfl]
    simp [Finset.sum_insert, Finset.mem_insert, Finset.mem_singleton]
    ring
  have hTcard : (Ks.biUnion (fun k => T.filter (fun g => g.bag.card = k))).card ≤
      T.card := Finset.card_le_card hbiUnion_sub
  rw [hsum_eq, hsumK] at hTcard
  omega

/-- **Cardinality refutation, sharp form (635).** Any candidate `T`
containing `init` with `|T| < 635` cannot be closed. Refutes ALL atlas
candidates of size < 635 in one shot — five times the previous threshold
of 127. -/
theorem not_IsClosedSet_of_init_mem_card_lt_635
    {T : Finset GameState} (hinit : GameState.init ∈ T) (hcard : T.card < 635) :
    ¬ IsClosedSet GameConfig.standard T := fun h => by
  have := closed_init_atlas_card_ge_635 hinit h
  omega

/-! ### Specific cell-count witnesses

At early depths, the cell count is FIXED (not just constrained modulo 10):
no line clears can happen with fewer than 10 cells. -/

/-- **Cell count = 4.** Every closed init-atlas contains a state with
exactly 4 cells (the depth-1 forced reach). -/
theorem exists_state_with_count_eq_four
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, g.board.count = 4 := by
  obtain ⟨g, hreach, hgT⟩ := exists_reachableAt_in_closed_init_atlas hinit hclosed 1
  refine ⟨g, hgT, ?_⟩
  obtain ⟨k, hk⟩ := ReachableAt.exists_count_eq hreach
  simp [GameConfig.standard] at hk
  omega

/-- **Cell count = 8.** Every closed init-atlas contains a state with
exactly 8 cells (the depth-2 forced reach). -/
theorem exists_state_with_count_eq_eight
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, g.board.count = 8 := by
  obtain ⟨g, hreach, hgT⟩ := exists_reachableAt_in_closed_init_atlas hinit hclosed 2
  refine ⟨g, hgT, ?_⟩
  obtain ⟨k, hk⟩ := ReachableAt.exists_count_eq hreach
  simp [GameConfig.standard] at hk
  omega

/-- **Cell-count diversity ≥ 3.** Closed init-atlases contain states at
cell counts 0, 4, AND 8 simultaneously (three distinct values: init at 0,
depth-1 at 4, depth-2 at 8). -/
theorem closed_init_atlas_has_three_cell_counts
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    (∃ g ∈ T, g.board.count = 0) ∧
    (∃ g ∈ T, g.board.count = 4) ∧
    (∃ g ∈ T, g.board.count = 8) := by
  refine ⟨⟨GameState.init, hinit, ?_⟩, ?_, ?_⟩
  · rw [GameState.init_board_count]
  · exact exists_state_with_count_eq_four hinit hclosed
  · exact exists_state_with_count_eq_eight hinit hclosed

/-- **Distinct cell counts ≥ 5.** Closed init-atlases contain at least 5
distinct cell-count values — one per residue class mod 10. Stronger than
"5 residues covered" because we show distinct VALUES, not just residues. -/
theorem closed_init_atlas_distinct_cell_counts_ge_five
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    5 ≤ (T.image (fun g => g.board.count)).card := by
  classical
  obtain ⟨⟨g0, hg0T, hg0r⟩, ⟨g4, hg4T, hg4r⟩, ⟨g8, hg8T, hg8r⟩,
         ⟨g2, hg2T, hg2r⟩, ⟨g6, hg6T, hg6r⟩⟩ :=
    closed_init_atlas_covers_all_residues hinit hclosed
  -- Map Fin 5 → ℕ via the 5 cell counts.
  let cnts : Fin 5 → ℕ := fun i =>
    match i.val with
    | 0 => g0.board.count
    | 1 => g4.board.count
    | 2 => g8.board.count
    | 3 => g2.board.count
    | _ => g6.board.count
  have hcnts_res : ∀ i : Fin 5, cnts i % 10 =
      (match i.val with | 0 => 0 | 1 => 4 | 2 => 8 | 3 => 2 | _ => 6) := by
    intro i
    fin_cases i <;> simp only [cnts] <;>
      first | exact hg0r | exact hg4r | exact hg8r | exact hg2r | exact hg6r
  have hcnts_inj : Function.Injective cnts := by
    intro i j hij
    have hres : cnts i % 10 = cnts j % 10 := by rw [hij]
    rw [hcnts_res i, hcnts_res j] at hres
    apply Fin.ext
    fin_cases i <;> fin_cases j <;> simp_all <;> omega
  have hcnts_in : ∀ i : Fin 5, cnts i ∈ T.image (fun g => g.board.count) := by
    intro i
    fin_cases i <;> simp only [cnts]
    · exact Finset.mem_image_of_mem _ hg0T
    · exact Finset.mem_image_of_mem _ hg4T
    · exact Finset.mem_image_of_mem _ hg8T
    · exact Finset.mem_image_of_mem _ hg2T
    · exact Finset.mem_image_of_mem _ hg6T
  have hsub : Finset.univ.image cnts ⊆ T.image (fun g => g.board.count) := by
    intro x hx
    rw [Finset.mem_image] at hx
    obtain ⟨i, _, rfl⟩ := hx
    exact hcnts_in i
  have hcard : (Finset.univ.image cnts).card = 5 := by
    rw [Finset.card_image_of_injective _ hcnts_inj, Finset.card_univ,
        Fintype.card_fin]
  calc 5 = (Finset.univ.image cnts).card := hcard.symm
    _ ≤ _ := Finset.card_le_card hsub

/-- **Reachable-state cell counts are even.** Every `ReachableAt`-state
has even cell count: `board.count = 4 * depth - cols * line_clears` is a
combination of even quantities. -/
theorem reachableAt_count_even {n : ℕ} {g : GameState}
    (h : ReachableAt GameConfig.standard n g) :
    Even g.board.count := by
  obtain ⟨k, hk⟩ := ReachableAt.exists_count_eq h
  simp [GameConfig.standard] at hk
  -- hk : g.board.count + 10 * k = 4 * n
  -- g.board.count = 4*n - 10*k. Both summands even.
  use 2 * n - 5 * k
  omega

/-- **Reachable witnesses in T have even cell counts.** Every depth-`n`
forced reach witness in T has an even cell count. -/
theorem reachableAt_witness_in_T_count_even
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) (n : ℕ) :
    ∃ g ∈ T, Even g.board.count := by
  obtain ⟨g, hreach, hgT⟩ := exists_reachableAt_in_closed_init_atlas hinit hclosed n
  exact ⟨g, hgT, reachableAt_count_even hreach⟩

/-- **At least 5 distinct full-bag states.** Closed init-atlases contain
full-bag witnesses at depths `0, 7, 14, 21, 28` with residues `0, 8, 6,
4, 2` respectively. Five distinct residues ⇒ five distinct states.
Strictly sharper than the ≥ 2 bound. -/
theorem closed_init_atlas_has_five_full_bag_states
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    5 ≤ (T.filter (fun g => g.bag = Bag.full)).card := by
  classical
  let g : Fin 5 → GameState := fun n =>
    Classical.choose (exists_full_bag_at_depth_seven_mul hinit hclosed n.val)
  have hgspec : ∀ i : Fin 5, g i ∈ T ∧ (g i).bag = Bag.full ∧
      (g i).board.count % 10 = (28 * i.val) % 10 := by
    intro i
    have := Classical.choose_spec (exists_full_bag_at_depth_seven_mul hinit hclosed i.val)
    exact ⟨this.1, this.2.2.1, this.2.2.2⟩
  have hgT : ∀ i : Fin 5, g i ∈ T.filter (fun x => x.bag = Bag.full) := fun i =>
    Finset.mem_filter.mpr ⟨(hgspec i).1, (hgspec i).2.1⟩
  have hg_inj : Function.Injective g := by
    intro i j hij
    have hres_eq : (g i).board.count % 10 = (g j).board.count % 10 := by rw [hij]
    rw [(hgspec i).2.2, (hgspec j).2.2] at hres_eq
    apply Fin.ext
    fin_cases i <;> fin_cases j <;> simp_all <;> omega
  have hcard : (Finset.univ.image g).card = 5 := by
    rw [Finset.card_image_of_injective _ hg_inj, Finset.card_univ, Fintype.card_fin]
  have hsub : Finset.univ.image g ⊆ T.filter (fun x => x.bag = Bag.full) := by
    intro x hx
    rw [Finset.mem_image] at hx
    obtain ⟨i, _, rfl⟩ := hx
    exact hgT i
  calc 5 = (Finset.univ.image g).card := hcard.symm
    _ ≤ _ := Finset.card_le_card hsub

/-! ### Phases of boards — the period-35 phase function

Reachable states cycle through a finite set of *phases*. Two independent
clocks run in lockstep:

- the **bag clock** (period 7): `g.bag.card = 7 - n % 7` at depth `n`
  (`ReachableAt.bag_card_eq`);
- the **count clock** (period 5): `g.board.count % 10 = (4 * n) % 10`
  (`ReachableAt.standard_count_mod_ten`), cycling through `0,4,8,2,6`.

Their least common period is `lcm 7 5 = 35`: every reachable state's
`(bag.card, count % 10)` signature is determined by `n % 35`. This section
makes the phase structure first-class — the natural clock of the solving
program is 35 pieces long. -/

/-- Count phase at depth `n`: the cell-count residue mod 10. -/
def boardPhase (n : ℕ) : ℕ := (4 * n) % 10

/-- Bag phase at depth `n`: pieces remaining in the current bag. -/
def bagPhase (n : ℕ) : ℕ := 7 - n % 7

/-- Joint phase signature at depth `n`. -/
def jointPhase (n : ℕ) : ℕ × ℕ := (bagPhase n, boardPhase n)

/-- Count phase has period 5. -/
@[simp] theorem boardPhase_add_five (n : ℕ) :
    boardPhase (n + 5) = boardPhase n := by
  unfold boardPhase; omega

/-- Count phase depends only on `n % 5`. -/
theorem boardPhase_eq_mod_five (n : ℕ) : boardPhase n = boardPhase (n % 5) := by
  unfold boardPhase; omega

/-- Bag phase has period 7. -/
@[simp] theorem bagPhase_add_seven (n : ℕ) :
    bagPhase (n + 7) = bagPhase n := by
  unfold bagPhase; omega

/-- Bag phase depends only on `n % 7`. -/
theorem bagPhase_eq_mod_seven (n : ℕ) : bagPhase n = bagPhase (n % 7) := by
  unfold bagPhase; omega

/-- Joint phase has period 35 = lcm 7 5 — the natural clock of the orbit. -/
theorem jointPhase_add_35 (n : ℕ) : jointPhase (n + 35) = jointPhase n := by
  unfold jointPhase bagPhase boardPhase
  rw [Prod.mk.injEq]
  omega

/-- The count phase is always one of the 5 even residues `{0,4,8,2,6}`. -/
theorem boardPhase_mem_five (n : ℕ) :
    boardPhase n ∈ ({0, 4, 8, 2, 6} : Finset ℕ) := by
  unfold boardPhase
  simp only [Finset.mem_insert, Finset.mem_singleton]
  omega

/-- Exactly 5 distinct count phases occur over one period. -/
theorem boardPhase_image_range_five :
    (Finset.range 5).image boardPhase = ({0, 4, 8, 2, 6} : Finset ℕ) := by
  unfold boardPhase; decide

/-- The "home phase" `(7, 0)` — full bag, zero count residue (the init-like
phase) — recurs exactly at multiples of 35. CRT: bag resets every 7, count
every 5, both together every 35. -/
theorem jointPhase_eq_home_iff (n : ℕ) :
    jointPhase n = (7, 0) ↔ n % 35 = 0 := by
  unfold jointPhase bagPhase boardPhase
  rw [Prod.mk.injEq]
  omega

/-- **Phase of a reachable state.** A depth-`n` reachable state's
`(bag.card, count % 10)` signature equals `jointPhase n`: the phase is
fully determined by the depth. -/
theorem reachableAt_jointPhase {n : ℕ} {g : GameState}
    (h : ReachableAt GameConfig.standard n g) :
    (g.bag.card, g.board.count % 10) = jointPhase n := by
  unfold jointPhase bagPhase boardPhase
  rw [h.bag_card_eq, h.standard_count_mod_ten]

/-- **Every closed init-atlas realizes the depth-`n` phase.** For each `n`,
some `g ∈ T` carries the phase signature `jointPhase n`. Combined with
`jointPhase_add_35`, the atlas realizes all 35 distinct joint phases — a
phase portrait of the orbit inside every closed init-atlas. -/
theorem closed_init_atlas_realizes_jointPhase
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) (n : ℕ) :
    ∃ g ∈ T, (g.bag.card, g.board.count % 10) = jointPhase n := by
  obtain ⟨g, hreach, hgT⟩ := exists_reachableAt_in_closed_init_atlas hinit hclosed n
  exact ⟨g, hgT, reachableAt_jointPhase hreach⟩

/-- The 35 joint phases over one period are all distinct: `jointPhase` is
injective on `range 35`. The minimal period is exactly 35, not a proper
divisor — the bag clock (7) and count clock (5) are coprime, so the orbit
genuinely cycles through all 35 combined phases (CRT, discharged by `omega`). -/
theorem jointPhase_injOn_range_35 :
    Set.InjOn jointPhase (Finset.range 35) := by
  intro i hi j hj hij
  simp only [Finset.coe_range, Set.mem_Iio] at hi hj
  unfold jointPhase bagPhase boardPhase at hij
  rw [Prod.mk.injEq] at hij
  omega

/-- Exactly 35 distinct joint phases occur over one period. -/
theorem jointPhase_image_range_35_card :
    ((Finset.range 35).image jointPhase).card = 35 := by
  rw [Finset.card_image_of_injOn jointPhase_injOn_range_35, Finset.card_range]

/-- **Phase portrait: every closed init-atlas exhibits all 35 phases.** The
image of `T` under the phase map `(bag.card, count % 10)` has at least 35
elements — the atlas realizes the full joint-phase cycle. A structural
footprint independent of (and finer-grained than) the `|T| ≥ 635`
cardinality bound: it counts distinct PHASES, not states. -/
theorem closed_init_atlas_phase_image_ge_35
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    35 ≤ (T.image (fun g => (g.bag.card, g.board.count % 10))).card := by
  have hsub : (Finset.range 35).image jointPhase ⊆
      T.image (fun g => (g.bag.card, g.board.count % 10)) := by
    intro x hx
    rw [Finset.mem_image] at hx
    obtain ⟨n, _, rfl⟩ := hx
    obtain ⟨g, hgT, hgphase⟩ := closed_init_atlas_realizes_jointPhase hinit hclosed n
    rw [Finset.mem_image]
    exact ⟨g, hgT, hgphase⟩
  calc 35 = ((Finset.range 35).image jointPhase).card :=
        jointPhase_image_range_35_card.symm
    _ ≤ _ := Finset.card_le_card hsub

/-! ### Phase-space completeness: the orbit is exactly the 7×5 grid

The joint phase `(bag.card, count % 10)` lives on the product grid
`{1,…,7} × {0,4,8,2,6}` — 7 bag sizes times 5 even count-residues = 35 cells.
We show the natural orbit `jointPhase` is *exactly* this grid: every depth's
phase lies on it (`jointPhase_mem_grid`), and conversely every grid cell is
realized at some depth `< 35` (`exists_eq_jointPhase_of_mem_grid`). This is the
"phases of boards" theme made sharp — the phase space is fully mapped, finite,
and every phase is *discoverable* within one period. -/

/-- `jointPhase` depends only on `n % 35` (period 35, structurally). Bag clock
factors through `% 7` and count clock through `% 5`, both divisors of 35. -/
theorem jointPhase_eq_mod_35 (n : ℕ) : jointPhase n = jointPhase (n % 35) := by
  unfold jointPhase bagPhase boardPhase
  rw [Prod.mk.injEq]
  omega

/-- Every depth's joint phase is one of the 35 phases over one period. -/
theorem jointPhase_mem_image_range_35 (n : ℕ) :
    jointPhase n ∈ (Finset.range 35).image jointPhase := by
  rw [Finset.mem_image]
  exact ⟨n % 35, Finset.mem_range.mpr (by omega), (jointPhase_eq_mod_35 n).symm⟩

/-- **The orbit equals the full product grid.** Over one period, `jointPhase`
hits exactly the 35 cells `{1,…,7} × {0,4,8,2,6}` — bag size crossed with even
count-residue. (`decide` evaluates both finite sides.) -/
theorem jointPhase_image_range_35_eq_grid :
    (Finset.range 35).image jointPhase
      = (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ) := by
  unfold jointPhase bagPhase boardPhase
  decide

/-- **Every phase lies on the grid.** For all `n`, `(bag.card, count%10)`'s
orbit value sits in `{1,…,7} × {0,4,8,2,6}`. -/
theorem jointPhase_mem_grid (n : ℕ) :
    jointPhase n ∈ (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ) := by
  rw [← jointPhase_image_range_35_eq_grid]
  exact jointPhase_mem_image_range_35 n

/-- **Phase discoverability: every grid cell is realized.** For any bag size
`k ∈ {1,…,7}` and even residue `r ∈ {0,4,8,2,6}`, some depth `n < 35` has
`jointPhase n = (k, r)`. Together with `jointPhase_mem_grid`, the phase orbit
is *exactly* the 7×5 grid — nothing missing, nothing extra. -/
theorem exists_eq_jointPhase_of_mem_grid {k r : ℕ}
    (hk : k ∈ Finset.Icc 1 7) (hr : r ∈ ({0, 4, 8, 2, 6} : Finset ℕ)) :
    ∃ n < 35, jointPhase n = (k, r) := by
  have hmem : (k, r) ∈ (Finset.range 35).image jointPhase := by
    rw [jointPhase_image_range_35_eq_grid]
    exact Finset.mem_product.mpr ⟨hk, hr⟩
  rw [Finset.mem_image] at hmem
  obtain ⟨n, hn, hnp⟩ := hmem
  exact ⟨n, Finset.mem_range.mp hn, hnp⟩

/-! ### The phase automaton and the reachable-state phase constraint

The phase advances deterministically each piece: a 35-state clock. And every
*reachable* state — hence every state a solver can legally occupy — has its
phase pinned onto the 7×5 grid. This bounds where play can ever go in
phase-space. -/

/-- The deterministic phase-transition map: one piece advances the clock.
Bag size counts down `7→6→…→1→7`; count residue advances `+4 mod 10`. -/
def phaseStep (p : ℕ × ℕ) : ℕ × ℕ :=
  (if p.1 = 1 then 7 else p.1 - 1, (p.2 + 4) % 10)

/-- **The phase automaton.** `jointPhase (n+1) = phaseStep (jointPhase n)`: the
orbit is a deterministic 35-state cycle driven by `phaseStep`. The "program"
always knows its next phase from its current one — no board inspection needed. -/
theorem jointPhase_succ (n : ℕ) :
    jointPhase (n + 1) = phaseStep (jointPhase n) := by
  have hbag : bagPhase (n + 1) = if bagPhase n = 1 then 7 else bagPhase n - 1 := by
    unfold bagPhase
    by_cases h : n % 7 = 6
    · rw [if_pos (by omega)]; omega
    · rw [if_neg (by omega)]; omega
  have hboard : boardPhase (n + 1) = (boardPhase n + 4) % 10 := by
    unfold boardPhase; omega
  unfold jointPhase phaseStep
  rw [hbag, hboard]

/-- **Every reachable state's phase lies on the 7×5 grid.** A depth-`n`
reachable state has `(bag.card, count%10) ∈ {1,…,7} × {0,4,8,2,6}`. Combines
`reachableAt_jointPhase` (phase determined by depth) with `jointPhase_mem_grid`
(orbit ⊆ grid). Pins where reachable play lives in phase-space. -/
theorem reachableAt_phase_mem_grid {n : ℕ} {g : GameState}
    (h : ReachableAt GameConfig.standard n g) :
    (g.bag.card, g.board.count % 10)
      ∈ (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ) := by
  rw [reachableAt_jointPhase h]
  exact jointPhase_mem_grid n

/-- **Reachable count-residue is one of the five even classes.** Direct
corollary: any reachable state has `count % 10 ∈ {0,4,8,2,6}`. -/
theorem reachableAt_count_mod_mem {n : ℕ} {g : GameState}
    (h : ReachableAt GameConfig.standard n g) :
    g.board.count % 10 ∈ ({0, 4, 8, 2, 6} : Finset ℕ) := by
  have hmem := reachableAt_phase_mem_grid h
  rw [Finset.mem_product] at hmem
  exact hmem.2

/-- **Reachable bag size is between 1 and 7.** Companion corollary: any
reachable state has `bag.card ∈ {1,…,7}`. -/
theorem reachableAt_bag_card_mem {n : ℕ} {g : GameState}
    (h : ReachableAt GameConfig.standard n g) :
    g.bag.card ∈ Finset.Icc 1 7 := by
  have hmem := reachableAt_phase_mem_grid h
  rw [Finset.mem_product] at hmem
  exact hmem.1

/-! ### The phase automaton is closed: 35-state machine, orbit of `(7,0)`

Closing the automaton: the grid is `phaseStep`-invariant, and `jointPhase n` is
literally the `n`-th iterate of `phaseStep` from the home phase `(7,0)`. So the
entire phase behaviour is a self-contained finite-state machine on 35 states —
no infinite phase-space to search. -/

/-- Projection of `phaseStep` (first component), definitional. -/
@[simp] theorem phaseStep_fst (p : ℕ × ℕ) :
    (phaseStep p).1 = if p.1 = 1 then 7 else p.1 - 1 := rfl

/-- Projection of `phaseStep` (second component), definitional. -/
@[simp] theorem phaseStep_snd (p : ℕ × ℕ) :
    (phaseStep p).2 = (p.2 + 4) % 10 := rfl

/-- **The grid is `phaseStep`-invariant.** Applying the phase transition to any
grid cell yields a grid cell: bag size stays in `{1,…,7}` (count-down with
wraparound), count residue stays in `{0,4,8,2,6}` (`+4 mod 10` permutes the
even residues). The 35-cell grid is a closed invariant set of the automaton. -/
theorem phaseStep_mem_grid {p : ℕ × ℕ}
    (hp : p ∈ (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ)) :
    phaseStep p ∈ (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ) := by
  rw [Finset.mem_product] at hp
  obtain ⟨h1, h2⟩ := hp
  rw [Finset.mem_Icc] at h1
  rw [Finset.mem_insert, Finset.mem_insert, Finset.mem_insert, Finset.mem_insert,
      Finset.mem_singleton] at h2
  apply Finset.mem_product.mpr
  refine ⟨?_, ?_⟩
  · rw [Finset.mem_Icc, phaseStep_fst]
    by_cases hc : p.1 = 1
    · rw [if_pos hc]; omega
    · rw [if_neg hc]; omega
  · rw [Finset.mem_insert, Finset.mem_insert, Finset.mem_insert, Finset.mem_insert,
        Finset.mem_singleton, phaseStep_snd]
    omega

/-- **The home phase.** `jointPhase 0 = (7, 0)` — full bag, zero count residue:
the phase of `init`. The automaton's start state. -/
theorem jointPhase_zero : jointPhase 0 = (7, 0) := rfl

/-- **Closed-form: the phase is the iterated automaton from home.**
`jointPhase n = phaseStep^[n] (7, 0)`. The depth-`n` phase is exactly the
`n`-fold application of the transition map to the start state — the orbit of
`(7,0)` under `phaseStep`. With `phaseStep_mem_grid`, this orbit never leaves
the 35-cell grid. -/
theorem jointPhase_eq_iterate (n : ℕ) :
    jointPhase n = (phaseStep^[n]) (7, 0) := by
  induction n with
  | zero => rfl
  | succ k ih =>
    rw [jointPhase_succ, ih, Function.iterate_succ', Function.comp_apply]

/-! ### The automaton is an order-35 permutation; the home phase recurs forever

`phaseStep` returns to start after exactly 35 steps on every grid cell, and is
injective there — so it is a permutation of the 35-cell grid, a single 35-cycle.
Consequently the home phase `(7,0)` (full bag, the init phase) recurs at every
multiple of 35: in any infinite play, a fresh full bag with zero count-residue
appears infinitely often. -/

/-- **Cycle length 35.** Every grid cell returns to itself after 35 phase
steps. Each grid cell is `jointPhase k` for some `k < 35`; then
`phaseStep^[35] (jointPhase k) = jointPhase (35 + k) = jointPhase k` via
`jointPhase_add_35`. -/
theorem phaseStep_iterate_35_of_mem_grid {p : ℕ × ℕ}
    (hp : p ∈ (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ)) :
    phaseStep^[35] p = p := by
  rw [← jointPhase_image_range_35_eq_grid, Finset.mem_image] at hp
  obtain ⟨k, _, rfl⟩ := hp
  have key : jointPhase (35 + k) = phaseStep^[35] (jointPhase k) := by
    rw [jointPhase_eq_iterate (35 + k), Function.iterate_add_apply,
        ← jointPhase_eq_iterate k]
  rw [← key, Nat.add_comm, jointPhase_add_35]

/-- **The automaton is injective on the grid.** `phaseStep^[34]` is a left
inverse (since `phaseStep^[35] = id` on the grid), so `phaseStep` is injective
there — hence a permutation of the 35 cells, i.e. a single 35-cycle. -/
theorem phaseStep_injOn_grid :
    Set.InjOn phaseStep ↑((Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ)) := by
  have key : ∀ {p : ℕ × ℕ},
      p ∈ (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ) →
      phaseStep^[34] (phaseStep p) = p := by
    intro p hp
    rw [← Function.iterate_succ_apply]
    exact phaseStep_iterate_35_of_mem_grid hp
  intro a ha b hb hab
  rw [Finset.mem_coe] at ha hb
  rw [← key ha, hab, key hb]

/-- **The home phase recurs infinitely.** For every `N`, some depth `n ≥ N` has
`jointPhase n = (7, 0)` — full bag with zero count-residue (the init phase)
returns at every multiple of 35. A "phases of infinite play" property: no
matter how far the game runs, the home phase keeps coming back. -/
theorem jointPhase_home_recurs (N : ℕ) : ∃ n ≥ N, jointPhase n = (7, 0) := by
  refine ⟨35 * (N + 1), by omega, ?_⟩
  rw [jointPhase_eq_home_iff]
  omega

/-! ### The solving program's trajectory is phase-confined

The previous phase results characterised *reachable* states abstractly. Here we
pin them onto the live solver: every state the `safeSolver` visits along an
adversarial trace from `init` is `ReachableAt` at exactly that depth, so its
phase signature is precisely `jointPhase n`. The program that plays forever never
leaves the 35-cell phase grid, and its phase at step `n` is a closed-form
function of the move counter. -/

/-- **The `safeSolver` trace from `init` is exact-depth reachable (and safe).**
Depth-aware upgrade of `adversarialTrace_reachable_and_safe`: the depth-`n` trace
state is `ReachableAt cfg n`, not merely `Reachable`. The induction maintains the
safety invariant (needed to invoke `safeSolver_spec`), and—because `ReachableAt`
is phrased directly via `adversarialStep`—the step is immediate, with no
`adversarialStep_eq_step` detour. -/
theorem safeSolver_adversarialTrace_reachableAt {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg)
    (s : ℕ → Piece) (hl : LegalSequence s) :
    ∀ n, ReachableAt cfg n (adversarialTrace cfg (safeSolver cfg) s GameState.init n)
         ∧ adversarialTrace cfg (safeSolver cfg) s GameState.init n ∈ safe cfg := by
  intro n
  induction n with
  | zero => exact ⟨ReachableAt.init, h⟩
  | succ k ih =>
    obtain ⟨ihr, ihs⟩ := ih
    set g := adversarialTrace cfg (safeSolver cfg) s GameState.init k with hg
    have hpb : s k ∈ g.bag := by
      have hbag : g.bag = bagAt Bag.full s k := adversarialTrace_bag cfg _ s k
      rw [hbag]; exact hl k
    obtain ⟨hpiece, hvalid, hstepsafe⟩ := safeSolver_spec ihs hpb
    have hvalid' : ({safeSolver cfg g (s k) with piece := s k} : Placement).Valid cfg := by
      rw [Placement.eta_of_piece_eq hpiece]; exact hvalid
    refine ⟨?_, ?_⟩
    · rw [adversarialTrace_succ, ← hg]
      exact ReachableAt.step ihr (safe_not_lost ihs) hpb hvalid'
    · rw [adversarialTrace_succ, ← hg]; exact hstepsafe

/-- **The solver trace's phase is exactly `jointPhase n`.** Specialising the
exact-depth reachability to `standard` and reading off `reachableAt_jointPhase`:
at depth `n`, the `safeSolver` trace state has signature
`(bag.card, count % 10) = jointPhase n`. The phase of the live solving program is
a closed-form function of the move counter. -/
theorem safeSolver_trace_jointPhase
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).bag.card,
     (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board.count % 10) = jointPhase n :=
  reachableAt_jointPhase (safeSolver_adversarialTrace_reachableAt h s hl n).1

/-- **The solving program never leaves the 35-cell phase grid.** Combining the
trace-phase formula with `jointPhase_mem_grid`: every state the `safeSolver`
visits has phase in `Icc 1 7 ×ˢ {0,4,8,2,6}`. The program's entire infinite
trajectory is phase-confined to 35 cells — there is no unbounded phase space to
search. -/
theorem safeSolver_trace_phase_mem_grid
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).bag.card,
     (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board.count % 10)
      ∈ (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ) := by
  rw [safeSolver_trace_jointPhase h s hl n]; exact jointPhase_mem_grid n

/-- The solver trace's live bag always holds between 1 and 7 pieces. -/
theorem safeSolver_trace_bag_card_mem
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).bag.card ∈ Finset.Icc 1 7 := by
  have hmem := safeSolver_trace_phase_mem_grid h s hl n
  rw [Finset.mem_product] at hmem; exact hmem.1

/-- The solver trace's board count residue mod 10 stays in {0,4,8,2,6}. -/
theorem safeSolver_trace_count_mod_mem
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board.count % 10 ∈ ({0, 4, 8, 2, 6} : Finset ℕ) := by
  have hmem := safeSolver_trace_phase_mem_grid h s hl n
  rw [Finset.mem_product] at hmem; exact hmem.2

/-! ### The live solver resets to a fresh full bag forever

The phase-home recurrence (`jointPhase_home_recurs`), transported along the actual
`safeSolver` trajectory, becomes a concrete statement about infinite play: no
matter how long the program runs, the home phase `(7, 0)` — a *full seven-piece
bag* atop a board whose cell count is a multiple of the width — recurs at infinitely
many depths. This is the file's namesake **five-bag-reset** property realised at the
program level: the solver keeps returning to fresh full bags forever. -/

/-- **The solver's home phase recurs infinitely.** For every horizon `N`, some
later depth `n ≥ N` along the `safeSolver` trace has phase signature exactly
`(7, 0)`. Transports `jointPhase_home_recurs` through `safeSolver_trace_jointPhase`. -/
theorem safeSolver_trace_home_phase_recurs
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (N : ℕ) :
    ∃ n ≥ N,
      ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).bag.card,
       (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).board.count % 10) = (7, 0) := by
  obtain ⟨n, hn, hp⟩ := jointPhase_home_recurs N
  exact ⟨n, hn, by rw [safeSolver_trace_jointPhase h s hl n]; exact hp⟩

/-- **A fresh full bag recurs forever.** Along any `safeSolver` play from `init`,
the bag returns to the complete seven-piece `Bag.full` at infinitely many depths.
The home-phase `bag.card = 7` is upgraded to `bag = Bag.full` via
`Bag.card_eq_seven_iff_eq_full`. This is the five-bag-reset phenomenon proved for
the live solving program: the randomizer's bag periodically refills, forever. -/
theorem safeSolver_trace_full_bag_recurs
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (N : ℕ) :
    ∃ n ≥ N,
      (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).bag = Bag.full := by
  obtain ⟨n, hn, hp⟩ := safeSolver_trace_home_phase_recurs h s hl N
  rw [Prod.mk.injEq] at hp
  exact ⟨n, hn, (Bag.card_eq_seven_iff_eq_full _).mp hp.1⟩

/-- **The board count returns to a width-multiple forever.** Companion projection:
along any `safeSolver` play, `board.count % 10 = 0` at infinitely many depths — the
board periodically holds an exact multiple of `10` cells. -/
theorem safeSolver_trace_count_mod_ten_zero_recurs
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (N : ℕ) :
    ∃ n ≥ N,
      (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).board.count % 10 = 0 := by
  obtain ⟨n, hn, hp⟩ := safeSolver_trace_home_phase_recurs h s hl N
  rw [Prod.mk.injEq] at hp
  exact ⟨n, hn, hp.2⟩

/-! ### Exact bag clock and board-count bound of the live solver

The recurrence results above pin down *that* the bag refills; these pin down
*exactly when*. The `safeSolver` trace inherits the depth-indexed bag clock
`bag.card = 7 - n % 7` from exact-depth reachability, so the bag is full at
precisely the multiples of `7` — sharper than the every-`35` home-phase
recurrence. The board count is simultaneously bounded by `4 n` (four cells per
piece, before any line clears only ever remove cells). -/

/-- **The live solver's bag clock.** At trace depth `n` the bag holds exactly
`7 - n % 7` pieces: `7` at each bag boundary, counting down to `1`. Inherited
from `ReachableAt.bag_card_eq` through `safeSolver_adversarialTrace_reachableAt`. -/
theorem safeSolver_trace_bag_card_eq
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).bag.card = 7 - n % 7 :=
  (safeSolver_adversarialTrace_reachableAt h s hl n).1.bag_card_eq

/-- **The bag is full at exactly the multiples of 7.** Exact characterisation of
the five-bag-reset boundary on the live trace: `bag = Bag.full ↔ 7 ∣ n`. Strictly
sharper than `safeSolver_trace_full_bag_recurs` (which only exhibited *some* later
full-bag depth). Via `Bag.card_eq_seven_iff_eq_full` + the bag clock + `omega`. -/
theorem safeSolver_trace_bag_full_iff
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).bag = Bag.full ↔ 7 ∣ n := by
  rw [← Bag.card_eq_seven_iff_eq_full, safeSolver_trace_bag_card_eq h s hl n]
  omega

/-- **The live solver's board never holds more than `4 n` cells.** At depth `n`
the board count is at most `4 * n` — each of the `n` placed pieces contributes
four cells, and line clears only ever remove cells. From the exact count equation
`count + cols * k = 4 n` (`ReachableAt.exists_count_eq`), so `count ≤ 4 n`. -/
theorem safeSolver_trace_count_le
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board.count ≤ 4 * n := by
  obtain ⟨k, hk⟩ := (safeSolver_adversarialTrace_reachableAt h s hl n).1.exists_count_eq
  exact Nat.le.intro hk

/-! ### The live solver clears unboundedly many lines

Two facts collide to force unbounded line-clearing. (a) Every non-lost reachable
board has at most `cols * rows` cells (it fits in the playfield) — so the live
solver's board count is capped at `200`. (b) The exact count equation
`count + cols * k = 4 n` says the cumulative number of cleared lines is exactly
`k`. With `count ≤ 200` but `4 n → ∞`, the cleared-line count `k` must diverge:
playing forever necessarily clears arbitrarily many lines. -/

/-- **Non-lost reachable boards fit the playfield.** Re-proved locally (the
`SafeIterateFinite` version is not in scope here): a `Reachable`, non-`lost` board
has every cell inside `[0,cols) × [0,rows)`, so its cell count is `≤ cols * rows`. -/
theorem reachable_not_lost_count_le {cfg : GameConfig} {g : GameState}
    (hr : Reachable cfg g) (hnl : ¬ g.lost cfg) :
    g.board.count ≤ cfg.cols * cfg.rows := by
  have hsub : g.board ⊆ Finset.range cfg.cols ×ˢ Finset.range cfg.rows := by
    intro p hp
    rw [Finset.mem_product, Finset.mem_range, Finset.mem_range]
    exact ⟨hr.wf p hp, Nat.lt_of_not_le (fun hc => hnl ⟨p, hp, hc⟩)⟩
  have hcard := Finset.card_le_card hsub
  rw [Finset.card_product, Finset.card_range, Finset.card_range] at hcard
  exact hcard

/-- **The live solver's board never exceeds playfield capacity.** Its count is
`≤ cols * rows = 200`. From `reachable_not_lost_count_le` applied to the trace,
which is reachable (`safeSolver_adversarialTrace_reachableAt`) and never lost
(safe ⇒ not lost). -/
theorem safeSolver_trace_count_le_capacity
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board.count
      ≤ GameConfig.standard.cols * GameConfig.standard.rows :=
  reachable_not_lost_count_le
    (safeSolver_adversarialTrace_reachableAt h s hl n).1.reachable
    (safe_not_lost (safeSolver_adversarialTrace_reachableAt h s hl n).2)

/-- **The live solver clears unboundedly many lines.** For every target `L`, there
is a depth `n` whose exact count equation `count + cols * k = 4 n` has cleared-line
count `k ≥ L`. Pick `n = 50 + 3 L`: then `4 n = 200 + 12 L`, and since
`count ≤ 200`, the cleared lines satisfy `10 k ≥ 12 L`, hence `k ≥ L`. Infinite
survival is impossible without clearing arbitrarily many lines. -/
theorem safeSolver_trace_clears_unbounded
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (L : ℕ) :
    ∃ n k, (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board.count + GameConfig.standard.cols * k = 4 * n ∧ L ≤ k := by
  obtain ⟨k, hk⟩ :=
    (safeSolver_adversarialTrace_reachableAt h s hl (50 + 3 * L)).1.exists_count_eq
  refine ⟨50 + 3 * L, k, hk, ?_⟩
  have hcap := safeSolver_trace_count_le_capacity h s hl (50 + 3 * L)
  have hcols : GameConfig.standard.cols = 10 := rfl
  have hrows : GameConfig.standard.rows = 20 := rfl
  rw [hcols] at hk
  rw [hcols, hrows] at hcap
  omega

/-! ### The live solver never overflows the field — column-height bound

Survival is geometric: a board is `lost` exactly when some cell sits at row
`≥ rows`. The live `safeSolver` board is never lost, so *every* cell is strictly
below `rows`, and hence *every column's height* is at most `rows = 20`. The
program keeps all 10 columns within the 20-row field forever. -/

/-- **Not-lost means every cell is in-field.** Unpacking `¬ g.lost cfg`
(= `¬ ∃ p ∈ board, rows ≤ p.2`): every occupied cell has row `< cfg.rows`. -/
theorem not_lost_in_field {cfg : GameConfig} {g : GameState}
    (hnl : ¬ g.lost cfg) : ∀ p ∈ g.board, p.2 < cfg.rows :=
  fun p hp => Nat.lt_of_not_le (fun hc => hnl ⟨p, hp, hc⟩)

/-- **The live solver keeps every column within the field.** At every depth and in
every column, the `safeSolver` board height is `≤ cfg.rows`. From
`Board.colHeight_le_rows_of_in_field` applied to the not-lost trace. The geometric
heart of infinite survival: no column ever climbs out of the playfield. -/
theorem safeSolver_trace_colHeight_le
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) (j : ℕ) :
    Board.colHeight (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board j ≤ GameConfig.standard.rows :=
  Board.colHeight_le_rows_of_in_field
    (not_lost_in_field
      (safe_not_lost (safeSolver_adversarialTrace_reachableAt h s hl n).2)) j

/-- **Concrete 20-row bound.** Every column of the live solver's board stays at
height `≤ 20`. The `standard` specialisation of `safeSolver_trace_colHeight_le`. -/
theorem safeSolver_trace_colHeight_le_twenty
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) (j : ℕ) :
    Board.colHeight (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board j ≤ 20 := by
  have hb := safeSolver_trace_colHeight_le h s hl n j
  rwa [GameConfig.standard_rows] at hb

/-- **Layer phase invariance (discoverability).** Every state discovered at
depth `n` carries the depth-`n` joint phase `(bag.card, count % 10) = jointPhase n`.
The phase is a *layer invariant*: it is constant across the entire reachable layer,
determined purely by the discovery depth. Immediate from
`reachableAt_jointPhase` via `reachableAt_of_mem_reachableLayer`. -/
theorem reachableLayer_jointPhase {n : ℕ} {g : GameState}
    (hg : g ∈ reachableLayer n) :
    (g.bag.card, g.board.count % 10) = jointPhase n :=
  reachableAt_jointPhase (reachableAt_of_mem_reachableLayer hg)

/-- **Joint-phase equality is exactly congruence mod 35.** `jointPhase n = jointPhase m`
iff `n ≡ m (mod 35)`. The bag clock (period 7) and count clock (period 5) are
coprime, so by CRT the joint orbit has minimal period `35` and `jointPhase`
separates residues mod 35. (omega decides the underlying Presburger formula.) -/
theorem jointPhase_eq_iff_mod_35 (n m : ℕ) :
    jointPhase n = jointPhase m ↔ n % 35 = m % 35 := by
  unfold jointPhase bagPhase boardPhase
  rw [Prod.mk.injEq]
  omega

/-- **Phase-distinct layers are disjoint.** If two depths carry different joint
phases then their reachable layers share no state: any common state would
simultaneously exhibit both phases. A discoverability bound — a board cannot be
discovered at two depths whose phase signatures differ. -/
theorem reachableLayer_disjoint_of_jointPhase_ne {n m : ℕ}
    (hne : jointPhase n ≠ jointPhase m) :
    Disjoint (reachableLayer n) (reachableLayer m) := by
  rw [Finset.disjoint_left]
  intro g hgn hgm
  exact hne (by rw [← reachableLayer_jointPhase hgn, reachableLayer_jointPhase hgm])

/-- **Sharp recurrence law: layers disjoint unless depths agree mod 35.** A state
can only be re-discovered at depths congruent mod 35 — `reachableLayer n` and
`reachableLayer m` are disjoint whenever `n % 35 ≠ m % 35`. Combines
`jointPhase_eq_iff_mod_35` with phase-distinct disjointness. In particular any
35 consecutive layers carry the 35 distinct phases, so a board recurs (if at all)
only on the 35-periodic sublattice of depths. -/
theorem reachableLayer_disjoint_of_mod_35_ne {n m : ℕ}
    (hne : n % 35 ≠ m % 35) :
    Disjoint (reachableLayer n) (reachableLayer m) :=
  reachableLayer_disjoint_of_jointPhase_ne
    (fun h => hne ((jointPhase_eq_iff_mod_35 n m).mp h))

/-- **Discovery depth is well-defined mod 35 (positive recurrence law).** If a
state is discovered at depths `n` and `m`, then `n ≡ m (mod 35)`. The positive
form of `reachableLayer_disjoint_of_mod_35_ne`: a board's set of discovery depths
lies in a single residue class mod 35. -/
theorem reachableLayer_depth_mod_35_eq {n m : ℕ} {g : GameState}
    (hn : g ∈ reachableLayer n) (hm : g ∈ reachableLayer m) :
    n % 35 = m % 35 := by
  by_contra hne
  exact absurd hm (Finset.disjoint_left.mp (reachableLayer_disjoint_of_mod_35_ne hne) hn)

/-- **Home phase recurs exactly at multiples of 35.** `jointPhase n = (7, 0)`
(full bag, count ≡ 0 mod 10) iff `35 ∣ n`. Divisibility restatement of
`jointPhase_eq_home_iff`. -/
theorem jointPhase_eq_home_iff_dvd (n : ℕ) :
    jointPhase n = (7, 0) ↔ 35 ∣ n := by
  rw [jointPhase_eq_home_iff]; omega

/-- **Bag-refill layer law.** Every state discovered at depth `n` has a full bag
iff `7 ∣ n`. A whole-atlas analog of `safeSolver_trace_bag_full_iff`: bag-refill
is a property of the discovery depth, holding uniformly across the layer. From
`ReachableAt.bag_card_eq` + `Bag.card_eq_seven_iff_eq_full`. -/
theorem reachableLayer_bag_full_iff {n : ℕ} {g : GameState}
    (hg : g ∈ reachableLayer n) :
    g.bag = Bag.full ↔ 7 ∣ n := by
  rw [← Bag.card_eq_seven_iff_eq_full,
      (reachableAt_of_mem_reachableLayer hg).bag_card_eq]
  omega

/-- **Home-state characterization.** At every depth divisible by 35, every
discovered state has a full bag and board count ≡ 0 mod 10 — the canonical
"reset" signature `(7, 0)` realized concretely on the board and bag. Combines
`reachableLayer_jointPhase`, `jointPhase_eq_home_iff_dvd`, and
`Bag.card_eq_seven_iff_eq_full`. -/
theorem reachableLayer_home_state {n : ℕ} {g : GameState}
    (hg : g ∈ reachableLayer n) (hdvd : 35 ∣ n) :
    g.bag = Bag.full ∧ g.board.count % 10 = 0 := by
  have hph : (g.bag.card, g.board.count % 10) = jointPhase n := reachableLayer_jointPhase hg
  rw [(jointPhase_eq_home_iff_dvd n).mpr hdvd, Prod.mk.injEq] at hph
  exact ⟨(Bag.card_eq_seven_iff_eq_full _).mp hph.1, hph.2⟩

/-- **The drawn piece is always legally available to the solver.** At every step
the piece `s n` belongs to the live solver's current bag. Solver-independent:
follows from `adversarialTrace_bag` (bag dynamics depend only on the sequence)
plus the legality of `s`. -/
theorem safeSolver_trace_drawn_piece_mem_bag
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    s n ∈ (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).bag := by
  rw [adversarialTrace_bag]; exact hl n

/-- **Safety invariant (headline).** Every state the live solver visits is safe.
Named projection of `safeSolver_adversarialTrace_reachableAt`. -/
theorem safeSolver_trace_mem_safe
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n ∈ safe GameConfig.standard :=
  (safeSolver_adversarialTrace_reachableAt h s hl n).2

/-- **The solver never tops out (headline).** At every depth the live solver's
state is not lost — the program plays forever. `safe_not_lost` along the trace. -/
theorem safeSolver_trace_not_lost
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    ¬ (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).lost GameConfig.standard :=
  safe_not_lost (safeSolver_trace_mem_safe h s hl n)

/-- **Per-step legality + safety contract of the program.** At every step the
live solver outputs a placement that (i) places exactly the drawn piece `s n`,
(ii) is a `Valid` placement, and (iii) lands the successor state back in `safe`.
The complete one-step guarantee of the solving program, packaged along its own
trace from `safeSolver_spec`. -/
theorem safeSolver_trace_step_spec
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    (safeSolver GameConfig.standard
        (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n) (s n)).piece = s n
    ∧ (safeSolver GameConfig.standard
        (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n) (s n)).Valid GameConfig.standard
    ∧ adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init (n + 1) ∈ safe GameConfig.standard := by
  obtain ⟨hpiece, hvalid, hstep⟩ :=
    safeSolver_spec (safeSolver_trace_mem_safe h s hl n)
      (safeSolver_trace_drawn_piece_mem_bag s hl n)
  refine ⟨hpiece, hvalid, ?_⟩
  rw [adversarialTrace_succ]
  exact hstep

/-- **Not-lost ⇒ bounded column heights.** Any non-lost state has every column
height `≤ cfg.rows`. The pure geometric core extracted from
`Board.colHeight_le_rows_of_in_field ∘ not_lost_in_field`. -/
theorem not_lost_colHeight_le {cfg : GameConfig} {g : GameState}
    (hnl : ¬ g.lost cfg) (j : ℕ) :
    Board.colHeight g.board j ≤ cfg.rows :=
  Board.colHeight_le_rows_of_in_field (not_lost_in_field hnl) j

/-- **Every safe state is field-confined.** Each column of a safe state's board
stays at height `≤ cfg.rows`. Since the atlas is a subset of `safe`, this is an
atlas-wide geometric invariant. -/
theorem safe_colHeight_le {cfg : GameConfig} {g : GameState}
    (hs : g ∈ safe cfg) (j : ℕ) :
    Board.colHeight g.board j ≤ cfg.rows :=
  not_lost_colHeight_le (safe_not_lost hs) j

/-- **Closed atlas tables are field-confined (geometric invariant).** Every state
of any closed set `T` keeps all columns within the playfield: `colHeight ≤
cfg.rows`. `IsClosedSet` bundles `¬ g.lost` for each member, which forces the
height bound directly. -/
theorem closedSet_colHeight_le {cfg : GameConfig} {T : Finset GameState}
    (hT : IsClosedSet cfg T) {g : GameState} (hg : g ∈ T) (j : ℕ) :
    Board.colHeight g.board j ≤ cfg.rows :=
  not_lost_colHeight_le (hT g hg).1 j

/-- **Closed atlas boards live inside the playfield.** Every filled cell of every
state in a closed set has row index `< cfg.rows`. The cell-level form of
`closedSet_colHeight_le`. -/
theorem closedSet_board_in_field {cfg : GameConfig} {T : Finset GameState}
    (hT : IsClosedSet cfg T) {g : GameState} (hg : g ∈ T) :
    ∀ p ∈ g.board, p.2 < cfg.rows :=
  not_lost_in_field (hT g hg).1

/-- **Concrete 20-row bound for closed atlas tables.** The `standard`
specialisation: every column of every state in a closed set stays `≤ 20`. -/
theorem closedSet_colHeight_le_twenty {T : Finset GameState}
    (hT : IsClosedSet GameConfig.standard T) {g : GameState} (hg : g ∈ T) (j : ℕ) :
    Board.colHeight g.board j ≤ 20 := by
  have hb := closedSet_colHeight_le hT hg j
  rwa [GameConfig.standard_rows] at hb

/-- **Quantitative line clearing (atlas-wide).** For any reachable, non-lost state
at depth `n`, the number of cleared lines `k` satisfies `count + 10·k = 4·n`
exactly, and is pinned to `(2n-100)/5 ≤ k ≤ 2n/5`. The lower bound comes from the
200-cell capacity (`reachable_not_lost_count_le`), the upper from `count ≥ 0`. So
survival *forces* line clears to grow linearly with depth: at least `(2n-100)/5`
lines cleared by step `n`. -/
theorem reachableAt_lines_cleared_bounds {n : ℕ} {g : GameState}
    (hg : ReachableAt GameConfig.standard n g) (hnl : ¬ g.lost GameConfig.standard) :
    ∃ k, g.board.count + 10 * k = 4 * n ∧ 2 * n ≤ 5 * k + 100 ∧ 5 * k ≤ 2 * n := by
  obtain ⟨k, hk⟩ := hg.exists_count_eq
  have hcols : GameConfig.standard.cols = 10 := rfl
  have hrows : GameConfig.standard.rows = 20 := rfl
  rw [hcols] at hk
  have hcap := reachable_not_lost_count_le hg.reachable hnl
  rw [hcols, hrows] at hcap
  exact ⟨k, hk, by omega, by omega⟩

/-- **Quantitative line clearing for the live solver.** Along the `safeSolver`
trace the cleared-line count `k` obeys `count + 10·k = 4·n` and
`(2n-100)/5 ≤ k ≤ 2n/5`. Corollary of `reachableAt_lines_cleared_bounds` on the
trace (reachable + never lost). The program clears lines at an asymptotic rate of
`2/5` line per piece — the exact rate needed to keep `4` cells/piece bounded. -/
theorem safeSolver_trace_lines_cleared_bounds
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    ∃ k, (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board.count + 10 * k = 4 * n
      ∧ 2 * n ≤ 5 * k + 100 ∧ 5 * k ≤ 2 * n :=
  reachableAt_lines_cleared_bounds
    (safeSolver_adversarialTrace_reachableAt h s hl n).1
    (safe_not_lost (safeSolver_adversarialTrace_reachableAt h s hl n).2)

/-- **Even occupancy is a layer invariant.** Every state in `reachableLayer n`
has even board count. Layer form of `reachableAt_count_even` (the parity invariant
`count = 2·(2n − 5k)`: placement adds 4, clearing removes 10, both even). -/
theorem reachableLayer_count_even {n : ℕ} {g : GameState}
    (hg : g ∈ reachableLayer n) : Even g.board.count :=
  reachableAt_count_even (reachableAt_of_mem_reachableLayer hg)

/-- **The live solver's board always has even occupancy.** Solver form of the
parity invariant along the `safeSolver` trace. -/
theorem safeSolver_trace_count_even
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    Even (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board.count :=
  reachableAt_count_even (safeSolver_adversarialTrace_reachableAt h s hl n).1

/-- **Bag clock is a layer invariant.** Every state discovered at depth `n` has
exactly `7 − n%7` pieces left in its bag. `ReachableAt.bag_card_eq` read off the
layer. -/
theorem reachableLayer_bag_card {n : ℕ} {g : GameState}
    (hg : g ∈ reachableLayer n) : g.bag.card = 7 - n % 7 :=
  (reachableAt_of_mem_reachableLayer hg).bag_card_eq

/-- **Count clock is a layer invariant.** Every state discovered at depth `n` has
board count `≡ 4·n (mod 10)`. `ReachableAt.standard_count_mod_ten` read off the
layer. -/
theorem reachableLayer_count_mod_ten {n : ℕ} {g : GameState}
    (hg : g ∈ reachableLayer n) : g.board.count % 10 = (4 * n) % 10 :=
  (reachableAt_of_mem_reachableLayer hg).standard_count_mod_ten

/-- **Safe states reach every depth.** If the empty board is safe, then for every
`n` there is a safe state exactly-`n`-reachable from `init`. Induction using the
nonempty bag (card `7 − n%7 ≥ 1`) to pick a piece and `safeSolver_spec` to take a
safe-preserving valid step. No global legal sequence needed — pieces are chosen
existentially per step. -/
theorem exists_reachableAt_safe (h : GameState.init ∈ safe GameConfig.standard) :
    ∀ n, ∃ g, ReachableAt GameConfig.standard n g ∧ g ∈ safe GameConfig.standard := by
  intro n
  induction n with
  | zero => exact ⟨GameState.init, ReachableAt.init, h⟩
  | succ k ih =>
    obtain ⟨g, hr, hs⟩ := ih
    have hcard : 0 < g.bag.card := by rw [hr.bag_card_eq]; omega
    obtain ⟨p, hp⟩ := Finset.card_pos.mp hcard
    obtain ⟨hpiece, hvalid, hstep⟩ := safeSolver_spec hs hp
    have hvalid' : ({safeSolver GameConfig.standard g p with piece := p} : Placement).Valid
        GameConfig.standard := by
      rw [Placement.eta_of_piece_eq hpiece]; exact hvalid
    exact ⟨adversarialStep GameConfig.standard g p (safeSolver GameConfig.standard g p),
      ReachableAt.step hr (safe_not_lost hs) hp hvalid', hstep⟩

/-- **Infinite-depth discoverability.** If the empty board is safe, every reachable
layer is nonempty: the game tree has states at every depth `n`. The discrete
analog of "the program plays forever" — survival means there is always *something*
to discover one step deeper. -/
theorem reachableLayer_nonempty (h : GameState.init ∈ safe GameConfig.standard)
    (n : ℕ) : (reachableLayer n).Nonempty := by
  obtain ⟨g, hr, _⟩ := exists_reachableAt_safe h n
  exact ⟨g, hr.mem_reachableLayer⟩

/-- **Each depth realizes its joint phase.** When `init` is safe, for every `n`
there is a reachable state at depth `n` carrying the phase signature
`jointPhase n`. Combines `exists_reachableAt_safe` with `reachableAt_jointPhase`. -/
theorem exists_reachable_of_jointPhase (h : GameState.init ∈ safe GameConfig.standard)
    (n : ℕ) :
    ∃ g, ReachableAt GameConfig.standard n g
      ∧ (g.bag.card, g.board.count % 10) = jointPhase n := by
  obtain ⟨g, hr, _⟩ := exists_reachableAt_safe h n
  exact ⟨g, hr, reachableAt_jointPhase hr⟩

/-- **The reachable state space realizes every joint phase.** When `init` is safe,
every cell `φ` of the phase grid `{1,…,7} × {0,4,8,2,6}` is the phase signature of
some reachable state. The orbit `jointPhase` surjects onto the grid
(`jointPhase_image_range_35_eq_grid`), and each depth is reachable
(`exists_reachable_of_jointPhase`). A complete phase-portrait of the reachable
space: all 35 phases occur. -/
theorem reachable_realizes_all_phases (h : GameState.init ∈ safe GameConfig.standard)
    {φ : ℕ × ℕ} (hφ : φ ∈ (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ)) :
    ∃ n g, ReachableAt GameConfig.standard n g
      ∧ (g.bag.card, g.board.count % 10) = φ := by
  rw [← jointPhase_image_range_35_eq_grid, Finset.mem_image] at hφ
  obtain ⟨n, _, hn⟩ := hφ
  obtain ⟨g, hr, hp⟩ := exists_reachable_of_jointPhase h n
  exact ⟨n, g, hr, by rw [hp]; exact hn⟩

/-! ### The reachable phase set is *exactly* the 7×5 grid

Iteration 21's `reachable_realizes_all_phases` gives the `⊇` direction (every grid
cell is realized) and the pre-existing `reachableAt_phase_mem_grid` gives `⊆`
(every reachable state's phase lies on the grid). We fuse them into a single
Set-level equality and read off its finiteness and exact cardinality (35). This
makes the "phases of boards" theme sharp at the level of the whole reachable
space: the set of phase signatures `{(bag.card, count%10)}` of all reachable
states is a finite 35-element grid, no more, no less — when `init` is safe. -/

/-- The phase grid `{1,…,7} × {0,4,8,2,6}` has exactly 35 cells. -/
theorem phaseGrid_card :
    ((Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ)).card = 35 := by
  decide

/-- **The reachable phase set equals the grid.** When `init` is safe, the set of
phase signatures `(bag.card, count%10)` ranging over *all* reachable states is
exactly the coerced finite grid `↑({1,…,7} ×ˢ {0,4,8,2,6})`. The `⊆` inclusion is
`reachableAt_phase_mem_grid` (every reachable phase lies on the grid); the `⊇`
inclusion is `reachable_realizes_all_phases` (every grid cell is realized). A
crisp, closed-form description of where reachable play lives in phase-space. -/
theorem reachablePhases_eq_grid (h : GameState.init ∈ safe GameConfig.standard) :
    {φ : ℕ × ℕ | ∃ n g, ReachableAt GameConfig.standard n g
        ∧ (g.bag.card, g.board.count % 10) = φ}
      = ↑((Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ)) := by
  ext φ
  rw [Set.mem_setOf_eq, Finset.mem_coe]
  constructor
  · rintro ⟨n, g, hr, rfl⟩
    exact reachableAt_phase_mem_grid hr
  · intro hφ
    exact reachable_realizes_all_phases h hφ

/-- **The reachable phase set is finite.** Immediate from
`reachablePhases_eq_grid`: it equals a coerced `Finset`. -/
theorem reachablePhases_finite (h : GameState.init ∈ safe GameConfig.standard) :
    {φ : ℕ × ℕ | ∃ n g, ReachableAt GameConfig.standard n g
        ∧ (g.bag.card, g.board.count % 10) = φ}.Finite := by
  rw [reachablePhases_eq_grid h]
  exact ((Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ)).finite_toSet

/-- **Exactly 35 reachable phases.** The reachable phase set has `ncard = 35`:
bag size (7 values) crossed with even count-residue (5 values). The whole infinite
reachable orbit projects onto just 35 phase cells. -/
theorem reachablePhases_ncard_eq_35 (h : GameState.init ∈ safe GameConfig.standard) :
    {φ : ℕ × ℕ | ∃ n g, ReachableAt GameConfig.standard n g
        ∧ (g.bag.card, g.board.count % 10) = φ}.ncard = 35 := by
  rw [reachablePhases_eq_grid h, Set.ncard_coe_finset, phaseGrid_card]

/-! ### Every phase recurs infinitely — total discoverability along the live play

`jointPhase_home_recurs` gave infinite recurrence of the *single* home phase
`(7,0)`. We generalise to *every* grid cell: each of the 35 phases recurs at
arbitrarily large depth (`jointPhase_recurs`), then transport this onto the live
`safeSolver` trajectory (`safeSolver_trace_phase_recurs`). The capstone
`safeSolver_trace_visits_phase_iff` characterises *exactly* which phases the
solving program ever occupies: a phase is visited iff it lies on the 7×5 grid.
The program's infinite play is a dense tour of all 35 phases, forever. -/

/-- **Every grid phase recurs infinitely.** For any cell `φ` of the 7×5 grid and
any horizon `N`, some depth `n ≥ N` has `jointPhase n = φ`. Generalises
`jointPhase_home_recurs` (the `φ = (7,0)` case) to the whole grid: pick the
period representative `m < 35` with `jointPhase m = φ`
(`jointPhase_image_range_35_eq_grid`), then shift by `35*(N+1)` and reduce mod 35
(`jointPhase_eq_mod_35`). -/
theorem jointPhase_recurs {φ : ℕ × ℕ}
    (hφ : φ ∈ (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ)) (N : ℕ) :
    ∃ n ≥ N, jointPhase n = φ := by
  rw [← jointPhase_image_range_35_eq_grid, Finset.mem_image] at hφ
  obtain ⟨m, hm, hmp⟩ := hφ
  rw [Finset.mem_range] at hm
  refine ⟨m + 35 * (N + 1), by omega, ?_⟩
  have hmod : (m + 35 * (N + 1)) % 35 = m := by omega
  rw [jointPhase_eq_mod_35, hmod, hmp]

/-- **The live solver revisits every phase forever.** Along any `safeSolver` play
from `init`, each of the 35 grid phases `φ` recurs at infinitely many depths
(`∃ n ≥ N`). Transports `jointPhase_recurs` through `safeSolver_trace_jointPhase`.
Strictly generalises `safeSolver_trace_home_phase_recurs` (only the home phase) —
the program's trajectory is recurrent on the *entire* phase grid. -/
theorem safeSolver_trace_phase_recurs
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) {φ : ℕ × ℕ}
    (hφ : φ ∈ (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ)) (N : ℕ) :
    ∃ n ≥ N,
      ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).bag.card,
       (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).board.count % 10) = φ := by
  obtain ⟨n, hn, hp⟩ := jointPhase_recurs hφ N
  exact ⟨n, hn, by rw [safeSolver_trace_jointPhase h s hl n]; exact hp⟩

/-- **Exactly the grid phases are visited.** A phase `φ` is occupied by the
`safeSolver` trace at some depth *iff* `φ` lies on the 7×5 grid. `→` is
phase-confinement (`safeSolver_trace_phase_mem_grid`); `←` is the `N = 0` case of
`safeSolver_trace_phase_recurs`. The program's reachable phase footprint is
*precisely* the 35-cell grid — no phase outside it, every phase inside it. -/
theorem safeSolver_trace_visits_phase_iff
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (φ : ℕ × ℕ) :
    (∃ n,
        ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init n).bag.card,
         (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init n).board.count % 10) = φ)
      ↔ φ ∈ (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ) := by
  constructor
  · rintro ⟨n, rfl⟩
    exact safeSolver_trace_phase_mem_grid h s hl n
  · intro hφ
    obtain ⟨n, _, hp⟩ := safeSolver_trace_phase_recurs h s hl hφ 0
    exact ⟨n, hp⟩

/-! ### The program itself: a finite-branching policy of precomputed legal moves

Theme 3 — the solving program *as a program*. At every one of its infinitely many
steps the `safeSolver` commits to a single placement drawn from the finite
precomputed candidate set `Placement.allValidFor standard (s n)` — the very
enumeration a search would scan. That set is always nonempty and has at most 40
elements, so the program is a well-defined policy whose per-step branching is
bounded by 40. The capstone `safeSolver_trace_policy_step` packages the full
operational contract of one program step: it picks a precomputed legal move for
the drawn piece, that move drives the successor trace state, and the successor is
safe — the inductive heartbeat of infinite play. -/

/-- **The program's move is always a precomputed legal placement.** At trace depth
`n`, the `safeSolver`'s chosen placement for the drawn piece `s n` lies in the
finite enumeration `Placement.allValidFor standard (s n)`. Connects the per-state
`safeSolver_mem_allValidFor` to the live trajectory via
`safeSolver_trace_drawn_piece_mem_bag`. (Independent of safety — only legality of
the sequence is needed.) -/
theorem safeSolver_trace_move_mem_allValidFor
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    safeSolver GameConfig.standard
        (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n) (s n)
      ∈ Placement.allValidFor GameConfig.standard (s n) :=
  safeSolver_mem_allValidFor (by decide)
    (safeSolver_trace_drawn_piece_mem_bag s hl n)

/-- **Per-step branching is at most 40.** The candidate set the program selects
from at step `n` has `card ≤ 40` (= `4 × 10`, four cells of horizontal freedom per
column). The infinite play is a walk through a policy whose every node has out-degree
bounded by 40. -/
theorem safeSolver_trace_candidates_card_le (s : ℕ → Piece) (n : ℕ) :
    (Placement.allValidFor GameConfig.standard (s n)).card ≤ 40 :=
  Placement.standard_allValidFor_card_le (s n)

/-- **There is always a legal move.** The candidate set at every step is nonempty —
the program never gets stuck for lack of a placement. -/
theorem safeSolver_trace_candidates_nonempty (s : ℕ → Piece) (n : ℕ) :
    (Placement.allValidFor GameConfig.standard (s n)).Nonempty :=
  Placement.standard_allValidFor_nonempty (s n)

/-- **The operational contract of one program step.** At every depth `n`, the
`safeSolver` (i) commits to a placement in the precomputed legal set
`allValidFor standard (s n)`, (ii) that placement announces the drawn piece `s n`,
(iii) applying it advances the trace to `trace (n+1)`, and (iv) the successor is
safe. The four conjuncts are the inductive heartbeat of infinite survival, packaged
as a single per-step guarantee about the program. -/
theorem safeSolver_trace_policy_step
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    safeSolver GameConfig.standard
        (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n) (s n)
        ∈ Placement.allValidFor GameConfig.standard (s n)
      ∧ (safeSolver GameConfig.standard
          (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init n) (s n)).piece = s n
      ∧ adversarialStep GameConfig.standard
          (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init n) (s n)
          (safeSolver GameConfig.standard
            (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
              GameState.init n) (s n))
          = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
              GameState.init (n + 1)
      ∧ adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init (n + 1) ∈ safe GameConfig.standard :=
  ⟨safeSolver_trace_move_mem_allValidFor s hl n,
   safeSolver_piece _ _ _,
   (adversarialTrace_succ _ _ _ _ n).symm,
   safeSolver_trace_mem_safe h s hl (n + 1)⟩

/-! ### The phase automaton is a permutation of the 35-cell grid

We already have the three ingredients — `phaseStep_mem_grid` (the grid maps into
itself), `phaseStep_injOn_grid` (injective there), and
`phaseStep_iterate_35_of_mem_grid` (order 35) — but never assembled them into the
clean statement that `phaseStep` is a *bijection* of the grid onto itself. Here we
do: surjectivity comes for free from injectivity on a finite set (the explicit
inverse is `phaseStep^[34]`), giving `Set.BijOn`, and hence the image equality
`phaseStep '' grid = grid`. The 35 phases of every closed init-atlas are permuted
by a single bijective clock — a structural footprint of the atlas's phase space. -/

/-- **`phaseStep` maps the grid into itself (Set form).** `Set.MapsTo` repackaging
of `phaseStep_mem_grid`, needed to iterate the map within `Set.BijOn`. -/
theorem phaseStep_mapsTo_grid :
    Set.MapsTo phaseStep ↑((Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ))
      ↑((Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ)) := by
  intro φ hφ
  rw [Finset.mem_coe] at hφ ⊢
  exact phaseStep_mem_grid hφ

/-- **The phase automaton is a bijection of the grid.** `Set.BijOn phaseStep grid
grid`: `phaseStep` maps the 35-cell grid onto itself injectively. Surjectivity is
witnessed constructively by the inverse `phaseStep^[34]` (since
`phaseStep^[35] = id` on the grid). The automaton is a genuine permutation — a
single 35-cycle through all phases. -/
theorem phaseStep_bijOn_grid :
    Set.BijOn phaseStep ↑((Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ))
      ↑((Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ)) := by
  refine ⟨phaseStep_mapsTo_grid, phaseStep_injOn_grid, ?_⟩
  intro φ hφ
  refine ⟨phaseStep^[34] φ, phaseStep_mapsTo_grid.iterate 34 hφ, ?_⟩
  calc phaseStep (phaseStep^[34] φ)
      = phaseStep^[34 + 1] φ := (Function.iterate_succ_apply' phaseStep 34 φ).symm
    _ = φ := phaseStep_iterate_35_of_mem_grid (Finset.mem_coe.mp hφ)

/-- **The automaton permutes the grid (image equality).** `phaseStep '' grid =
grid`: the forward image of the 35-cell grid under the phase transition is the grid
itself. Immediate from `phaseStep_bijOn_grid`. The phase space is exactly invariant —
not merely forward-closed, but bijectively shuffled. -/
theorem phaseStep_image_grid :
    phaseStep '' ↑((Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ))
      = ↑((Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ)) :=
  phaseStep_bijOn_grid.image_eq

/-! ### The orbit is an exact 35-cycle — minimal period and full range

`jointPhase` is periodic with period 35 (`jointPhase_add_35`); here we show 35 is
*minimal* — no smaller positive shift fixes the orbit (`jointPhase_minimal_period`) —
and that the orbit's full range over all of `ℕ` is exactly the 35-cell grid
(`jointPhase_range_eq_grid`). Transported to the live play, the program hits the
home/reset signature `(7,0)` at *precisely* the multiples of 35
(`safeSolver_trace_home_phase_iff`), never sooner: the five-bag-reset clock has
exact period 35. -/

/-- **The orbit's range is exactly the grid.** `Set.range jointPhase = grid`: as `n`
ranges over all of `ℕ`, the phase `jointPhase n` takes exactly the 35 grid values —
nothing outside, everything inside. The whole-`ℕ` strengthening of
`jointPhase_image_range_35_eq_grid` (which only covered one period). -/
theorem jointPhase_range_eq_grid :
    Set.range jointPhase = ↑((Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ)) := by
  ext φ
  rw [Set.mem_range, Finset.mem_coe]
  constructor
  · rintro ⟨n, rfl⟩
    exact jointPhase_mem_grid n
  · intro hφ
    rw [← jointPhase_image_range_35_eq_grid, Finset.mem_image] at hφ
    obtain ⟨n, _, hn⟩ := hφ
    exact ⟨n, hn⟩

/-- **35 is the minimal period.** No positive shift `T < 35` is a period of
`jointPhase`: there is always a depth (in fact `n = 0`) where `jointPhase (n + T) ≠
jointPhase n`. Hence the orbit genuinely cycles through all 35 phases before
returning — the bag clock (7) and count clock (5) being coprime forces the full
35-cycle, not a shorter one. -/
theorem jointPhase_minimal_period (T : ℕ) (hT0 : 0 < T) (hT35 : T < 35) :
    ∃ n, jointPhase (n + T) ≠ jointPhase n := by
  refine ⟨0, ?_⟩
  simp only [Nat.zero_add, ne_eq, jointPhase_eq_iff_mod_35]
  omega

/-- **The live solver hits the home/reset signature exactly at multiples of 35.**
Along any `safeSolver` play from `init`, the full home phase `(7,0)` — a complete
bag atop a count divisible by 10 — occurs at depth `n` iff `35 ∣ n`. Transports
`jointPhase_eq_home_iff_dvd` through `safeSolver_trace_jointPhase`. Sharpens the
existential `safeSolver_trace_home_phase_recurs` to an exact arithmetic
characterization, and the joint signature is finer than the bag-only
`safeSolver_trace_bag_full_iff` (`bag = full ↔ 7 ∣ n`). -/
theorem safeSolver_trace_home_phase_iff
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).bag.card,
     (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board.count % 10) = (7, 0) ↔ 35 ∣ n := by
  rw [safeSolver_trace_jointPhase h s hl n, jointPhase_eq_home_iff_dvd]

/-! ### Memorylessness / stationarity of the solving program

The `safeSolver` is a *stationary* (memoryless, Markov) policy: it reads only
the current `GameState` and emits a move, with no dependence on history. The
following cluster turns that into hard theorems about the live trace. The
backbone is the compositional law `adversarialTrace_add`: the tail of a play
from step `n` is *literally* a fresh play restarted from the state `trace n`
under the time-shifted piece sequence `fun k => s (n + k)`. Combined with the
"safe start ⇒ safe forever" guarantee, this shows every state the program
visits is itself a self-contained, infinitely-survivable starting position —
the program needs no memory of how it got there. -/

/-- **The shifted suffix sequence is legal from the visited state's bag.**
The tail piece-stream `fun k => s (n + k)` is a `LegalSequenceFrom` the bag of
`trace n`, because bag evolution depends only on the starting bag and the draws
(`adversarialTrace_bag`) and legality is suffix-closed
(`LegalSequenceFrom.suffix`). This is the legality half of the restart. -/
theorem safeSolver_trace_suffix_legal (s : ℕ → Piece) (hl : LegalSequence s)
    (n : ℕ) :
    LegalSequenceFrom
      (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).bag (fun k => s (n + k)) := by
  rw [adversarialTrace_bag]
  exact LegalSequenceFrom.suffix hl n

/-- **Markov restart decomposition of the live trace.** The state at step
`n + m` of the play equals the state after `m` steps of a *fresh* play started
from `trace n` under the time-shifted sequence `fun k => s (n + k)`. A direct
specialisation of `adversarialTrace_add` to `safeSolver`/`init`. This is the
formal statement that the program is memoryless: its future depends only on the
present state and the remaining inputs, not on the path taken to reach it. -/
theorem safeSolver_trace_tail_eq (s : ℕ → Piece) (n m : ℕ) :
    adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init (n + m) =
      adversarialTrace GameConfig.standard (safeSolver GameConfig.standard)
        (fun k => s (n + k))
        (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n) m :=
  adversarialTrace_add GameConfig.standard (safeSolver GameConfig.standard) s
    GameState.init n m

/-- **Tail safety via the restart, not the global trace.** Every state from
step `n` onward is safe — re-derived *through* the Markov decomposition: rewrite
the tail as a fresh play from the safe state `trace n` (which is safe by
`safeSolver_trace_mem_safe`) under the legal shifted sequence, then invoke the
"safe start ⇒ safe forever" guarantee `safeSolver_adversarialTrace_mem_safe_from`.
Demonstrates that safety propagates compositionally along restarts, independent
of the `init`-rooted argument. -/
theorem safeSolver_trace_tail_mem_safe
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n m : ℕ) :
    adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init (n + m) ∈ safe GameConfig.standard := by
  rw [safeSolver_trace_tail_eq]
  exact safeSolver_adversarialTrace_mem_safe_from
    (safeSolver_trace_mem_safe h s hl n) (safeSolver_trace_suffix_legal s hl n) m

/-- **Stationarity: every visited state is an infinite-survival start.** From
any state `trace n` the program visits along a safe play, `safeSolver` itself
`SolvesTetrisFrom` that state — it survives forever against every bag-legal
continuation. The policy carries no memory: reaching `trace n` via the original
play is interchangeable with being handed `trace n` cold. Direct from
`safeSolver_solvesTetrisFrom_of_mem_safe` on `safeSolver_trace_mem_safe`. -/
theorem safeSolver_trace_solvesTetrisFrom_state
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    SolvesTetrisFrom GameConfig.standard
      (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n)
      (safeSolver GameConfig.standard) :=
  safeSolver_solvesTetrisFrom_of_mem_safe (safeSolver_trace_mem_safe h s hl n)

/-! ### Closed cycles under a periodic adversary (M2)

The closed-atlas pigeonhole lemmas (`Atlas.IsClosedOn.toSolver_adversarialTrace_*`)
give a single *state* recurrence `trace i = trace j`, but the period `j - i` is
uncontrolled (≤ `S.card`) and need not match any structure in the piece input —
so it does not by itself yield an *honest* loop (the same state can evolve
differently under different future draws). Here we synchronise the recurrence
with a **periodic input**: if the adversary's piece sequence is `P`-periodic, we
pigeonhole over *multiples of `P`*, forcing the recurrence period to be a
multiple of `P`. Then the input genuinely repeats, and the trace is periodic
forever — a concrete closed cycle in the surviving subgraph. -/

/-- **Periodic trace ⇒ exact recurrence on the period lattice.** If the trace
obeys the one-period law `trace (base+k) = trace (base+d+k)` for every `k`, then
it returns *exactly* to `trace base` at every lattice point `base + n*d`.
Induction on `n`, each step shifting by `d` via the law at `k = m*d`.
Solver-agnostic. -/
theorem adversarialTrace_iterate_of_periodic
    (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece) (g0 : GameState)
    {base d : ℕ}
    (hcyc : ∀ k, adversarialTrace cfg σ s g0 (base + k)
      = adversarialTrace cfg σ s g0 (base + d + k))
    (n : ℕ) :
    adversarialTrace cfg σ s g0 (base + n * d)
      = adversarialTrace cfg σ s g0 base := by
  induction n with
  | zero => simp
  | succ m ih =>
    have key : adversarialTrace cfg σ s g0 (base + (m + 1) * d)
        = adversarialTrace cfg σ s g0 (base + m * d) := by
      have e : base + (m + 1) * d = base + d + m * d := by ring
      rw [e, ← hcyc (m * d)]
    rw [key]; exact ih

/-- **Closed cycle under a periodic adversary.** If `A` is a closed atlas on the
finite set `S`, the start `g₀ ∈ S`, and the piece sequence `t` is globally
`P`-periodic (`t (k+P) = t k`, `P ≥ 1`), then the materialised-solver trace is
genuinely periodic: there exist a base index and a period `d ≥ 1` with
`trace (base+k) = trace (base+d+k)` for all `k`, and the cycle state
`trace base ∈ S`. Proof: pigeonhole the map `k ↦ trace (k*P)` (values in the
finite `S`) over `0..S.card` to get `a*P, b*P` with equal trace; `P`-periodicity
of `t` makes the input `(b-a)*P`-periodic past `a*P`, so
`adversarialTrace_periodic_of_periodic_suffix` lifts the lone state recurrence to
full periodicity. Restricting to multiples of `P` is exactly what synchronises
the period with the input cycle. -/
theorem closedAtlas_periodic_input_closedCycle
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, t (k + P) = t k) :
    ∃ base d : ℕ, 1 ≤ d ∧
      (∀ k, adversarialTrace cfg A.toSolver t g₀ (base + k)
        = adversarialTrace cfg A.toSolver t g₀ (base + d + k)) ∧
      adversarialTrace cfg A.toSolver t g₀ base ∈ S := by
  have hshift : ∀ n k, t (n * P + k) = t k := by
    intro n k
    induction n with
    | zero => simp
    | succ m ih =>
      have hstep : (m + 1) * P + k = (m * P + k) + P := by ring
      rw [hstep, hper, ih]
  have key : ∀ x y : ℕ, x < y →
      adversarialTrace cfg A.toSolver t g₀ (x * P)
        = adversarialTrace cfg A.toSolver t g₀ (y * P) →
      ∃ base d : ℕ, 1 ≤ d ∧
        (∀ k, adversarialTrace cfg A.toSolver t g₀ (base + k)
          = adversarialTrace cfg A.toSolver t g₀ (base + d + k)) ∧
        adversarialTrace cfg A.toSolver t g₀ base ∈ S := by
    intro x y hxy hxyeq
    have harith : x * P + (y - x) * P = y * P := by
      have hxy' : x + (y - x) = y := by omega
      rw [← add_mul, hxy']
    refine ⟨x * P, (y - x) * P, ?_, ?_,
      h.toSolver_adversarialTrace_mem hg₀ hl (x * P)⟩
    · have hpos : (y - x) * P ≠ 0 := Nat.mul_ne_zero (by omega) (by omega)
      omega
    · intro k
      apply adversarialTrace_periodic_of_periodic_suffix
      · rw [harith]; exact hxyeq
      · intro j; rw [harith, hshift x j, hshift y j]
  have hmap : ∀ k ∈ Finset.range (S.card + 1),
      adversarialTrace cfg A.toSolver t g₀ (k * P) ∈ S :=
    fun k _ => h.toSolver_adversarialTrace_mem hg₀ hl (k * P)
  have hcard : S.card < (Finset.range (S.card + 1)).card := by
    rw [Finset.card_range]; omega
  obtain ⟨a, _, b, _, hne, heq⟩ :=
    Finset.exists_ne_map_eq_of_card_lt_of_maps_to hcard hmap
  rcases lt_or_gt_of_ne hne with hab | hab
  · exact key a b hab heq
  · exact key b a hab heq.symm

/-- **Headline: a state in `S` revisited forever under a periodic adversary.**
Combines `closedAtlas_periodic_input_closedCycle` with
`adversarialTrace_iterate_of_periodic`: under a closed atlas and a `P`-periodic
input there are a base and a period `d ≥ 1` so the trace returns *exactly* to the
state `trace base ∈ S` at every multiple `base + n*d`. A concrete closed loop in
the surviving subgraph — the M2 "closed cycle ⇒ infinite play" milestone,
realised against any periodic piece sequence. -/
theorem closedAtlas_periodic_input_recurrent_state
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, t (k + P) = t k) :
    ∃ base d : ℕ, 1 ≤ d ∧
      (∀ n, adversarialTrace cfg A.toSolver t g₀ (base + n * d)
        = adversarialTrace cfg A.toSolver t g₀ base) ∧
      adversarialTrace cfg A.toSolver t g₀ base ∈ S := by
  obtain ⟨base, d, hd, hcyc, hmem⟩ :=
    closedAtlas_periodic_input_closedCycle h hg₀ hl hP hper
  exact ⟨base, d, hd,
    fun n => adversarialTrace_iterate_of_periodic cfg A.toSolver t g₀ hcyc n, hmem⟩

/-! ### The limit cycle is a finite explicit set

A periodic trace doesn't just recur at lattice points `base + n*d`; the *entire*
tail `{trace (base + n) | n}` is confined to the `d` explicit states
`{trace (base + j) | j < d}`. This turns the closed cycle into a concrete finite
object — a `Finset` of at most `d` states inside `S` that the play never leaves.
The "winning strategy uses bounded memory" characterisation of the loop. -/

/-- **Shift-by-`d`-multiple invariance.** Generalises
`adversarialTrace_iterate_of_periodic` (the `r = 0` case): from the one-period
law, the trace is invariant under adding any multiple `d*q` to an arbitrary
offset `r`. Induction on `q`, peeling one `d` per step via `hcyc (r + d*m)` and a
`ring` re-association of the index. -/
theorem adversarialTrace_add_mul_period
    (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece) (g0 : GameState)
    {base d : ℕ}
    (hcyc : ∀ k, adversarialTrace cfg σ s g0 (base + k)
      = adversarialTrace cfg σ s g0 (base + d + k))
    (r q : ℕ) :
    adversarialTrace cfg σ s g0 (base + (r + d * q))
      = adversarialTrace cfg σ s g0 (base + r) := by
  induction q with
  | zero => simp
  | succ m ih =>
    rw [show base + (r + d * (m + 1)) = base + d + (r + d * m) from by ring,
        ← hcyc (r + d * m)]
    exact ih

/-- **Reduce the index modulo the period.** Every tail value equals the value at
the residue `n % d`. Write `n = n % d + d * (n / d)` (`Nat.mod_add_div`) and apply
`adversarialTrace_add_mul_period`. Collapses the infinite tail onto the finite
window `[0, d)`. -/
theorem adversarialTrace_mod_period
    (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece) (g0 : GameState)
    {base d : ℕ}
    (hcyc : ∀ k, adversarialTrace cfg σ s g0 (base + k)
      = adversarialTrace cfg σ s g0 (base + d + k))
    (n : ℕ) :
    adversarialTrace cfg σ s g0 (base + n)
      = adversarialTrace cfg σ s g0 (base + n % d) := by
  have e : base + n = base + (n % d + d * (n / d)) := by rw [Nat.mod_add_div]
  rw [e]
  exact adversarialTrace_add_mul_period cfg σ s g0 hcyc (n % d) (n / d)

/-- **Tail confined to the `d` cycle states.** With period `d ≥ 1`, every tail
state `trace (base + n)` lies in the finite image
`{trace (base + j) | j < d}`. Direct from `adversarialTrace_mod_period` +
`Nat.mod_lt`. The play, after `base`, visits at most `d` distinct states. -/
theorem adversarialTrace_tail_mem_cycle_image
    (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece) (g0 : GameState)
    {base d : ℕ} (hd : 1 ≤ d)
    (hcyc : ∀ k, adversarialTrace cfg σ s g0 (base + k)
      = adversarialTrace cfg σ s g0 (base + d + k))
    (n : ℕ) :
    adversarialTrace cfg σ s g0 (base + n)
      ∈ (Finset.range d).image (fun j => adversarialTrace cfg σ s g0 (base + j)) := by
  rw [adversarialTrace_mod_period cfg σ s g0 hcyc n]
  apply Finset.mem_image_of_mem
  rw [Finset.mem_range]
  exact Nat.mod_lt n hd

/-- **Closed atlas + periodic adversary ⇒ the play is confined to a finite cycle
inside `S`.** Packages the closed cycle as a concrete `Finset C` with
`C.Nonempty`, `C ⊆ S`, and `∀ n, trace (base + n) ∈ C`. From `base` onward the
materialised solver never leaves the ≤ `d` states of `C` — an explicit, finite,
`S`-internal limit cycle. The bounded-memory form of the M2 milestone. -/
theorem closedAtlas_periodic_input_tail_confined
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, t (k + P) = t k) :
    ∃ (base : ℕ) (C : Finset GameState), C.Nonempty ∧ C ⊆ S ∧
      ∀ n, adversarialTrace cfg A.toSolver t g₀ (base + n) ∈ C := by
  obtain ⟨base, d, hd, hcyc, _⟩ :=
    closedAtlas_periodic_input_closedCycle h hg₀ hl hP hper
  refine ⟨base,
    (Finset.range d).image (fun j => adversarialTrace cfg A.toSolver t g₀ (base + j)),
    ?_, ?_, ?_⟩
  · refine ⟨adversarialTrace cfg A.toSolver t g₀ (base + 0), ?_⟩
    rw [Finset.mem_image]
    exact ⟨0, Finset.mem_range.mpr (by omega), rfl⟩
  · intro g hg
    rw [Finset.mem_image] at hg
    obtain ⟨j, _, rfl⟩ := hg
    exact h.toSolver_adversarialTrace_mem hg₀ hl (base + j)
  · intro n
    exact adversarialTrace_tail_mem_cycle_image cfg A.toSolver t g₀ hd hcyc n

/-- **Infinitely-recurrent cycle state.** Under a closed atlas and a `P`-periodic
adversary, there is a single board state `g ∈ S` that the adversarial play
returns to *infinitely often*: for every horizon `N` there is a later time
`n ≥ N` with `trace (base + n) = g`. This is the canonical ω-recurrence form of
the limit cycle — the witness is `g = trace base`, revisited exactly at every
multiple `base + N*d` of the period. -/
theorem closedAtlas_periodic_input_infinitely_recurrent
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, t (k + P) = t k) :
    ∃ (base : ℕ) (g : GameState), g ∈ S ∧
      ∀ N, ∃ n, N ≤ n ∧ adversarialTrace cfg A.toSolver t g₀ (base + n) = g := by
  obtain ⟨base, d, hd, hcyc, hmem⟩ :=
    closedAtlas_periodic_input_closedCycle h hg₀ hl hP hper
  refine ⟨base, adversarialTrace cfg A.toSolver t g₀ base, hmem, ?_⟩
  intro N
  refine ⟨N * d, ?_, ?_⟩
  · have h1 : N * 1 ≤ N * d := Nat.mul_le_mul (le_refl N) hd
    simpa using h1
  · exact adversarialTrace_iterate_of_periodic cfg A.toSolver t g₀ hcyc N

/-- **Limit-cycle cardinality bound.** The finite confining set `C` of the tail
play (from `closedAtlas_periodic_input_tail_confined`) is not just a subset of
`S` but has `C.card ≤ S.card`: the limit cycle visits at most `|S|` distinct
board states. A concrete finiteness budget for the eventual behaviour. -/
theorem closedAtlas_periodic_input_tail_confined_card
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, t (k + P) = t k) :
    ∃ (base : ℕ) (C : Finset GameState), C.Nonempty ∧ C ⊆ S ∧ C.card ≤ S.card ∧
      ∀ n, adversarialTrace cfg A.toSolver t g₀ (base + n) ∈ C := by
  obtain ⟨base, C, hne, hsub, htail⟩ :=
    closedAtlas_periodic_input_tail_confined h hg₀ hl hP hper
  exact ⟨base, C, hne, hsub, Finset.card_le_card hsub, htail⟩

/-- **Phase-determinism of the limit cycle.** Once a one-period recurrence
`hcyc` holds at `base`, the tail state is a function of the *phase* `n mod d`
alone: any two offsets with the same residue mod `d` give the same board. This
is the precise "the eventual board depends only on which phase of the cycle you
are in" statement. -/
theorem adversarialTrace_eq_of_mod_eq
    (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece) (g0 : GameState)
    {base d : ℕ}
    (hcyc : ∀ k, adversarialTrace cfg σ s g0 (base + k)
      = adversarialTrace cfg σ s g0 (base + d + k))
    (m n : ℕ) (hmn : m % d = n % d) :
    adversarialTrace cfg σ s g0 (base + m)
      = adversarialTrace cfg σ s g0 (base + n) := by
  rw [adversarialTrace_mod_period cfg σ s g0 hcyc m,
    adversarialTrace_mod_period cfg σ s g0 hcyc n, hmn]

/-- **Ultimate periodicity of the play.** Under a closed atlas and a `P`-periodic
adversary, the adversarial board sequence is *eventually periodic*: there is a
threshold `base` and a period `d ≥ 1` such that `trace (n + d) = trace n` for
every `n ≥ base`. This is the canonical dynamical-systems characterization — the
infinite play settles into a strict period-`d` orbit. -/
theorem closedAtlas_periodic_input_ultimately_periodic
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, t (k + P) = t k) :
    ∃ (base d : ℕ), 1 ≤ d ∧
      ∀ n, base ≤ n →
        adversarialTrace cfg A.toSolver t g₀ (n + d)
          = adversarialTrace cfg A.toSolver t g₀ n := by
  obtain ⟨base, d, hd, hcyc, _⟩ :=
    closedAtlas_periodic_input_closedCycle h hg₀ hl hP hper
  refine ⟨base, d, hd, ?_⟩
  intro n hn
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
  rw [show base + k + d = base + d + k from by ring]
  exact (hcyc k).symm

/-- **Limit cycle fits in its period.** Exposing the period `d`, the confining
set `C` of tail states has `C.card ≤ d`: the orbit visits at most `d` distinct
boards. Tighter than the `C.card ≤ S.card` budget — `d ≤ S.card` is the
pigeonhole gap, so this is the genuine orbit length bound. -/
theorem closedAtlas_periodic_input_tail_confined_period
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, t (k + P) = t k) :
    ∃ (base d : ℕ) (C : Finset GameState), 1 ≤ d ∧ C.Nonempty ∧ C ⊆ S ∧
      C.card ≤ d ∧ ∀ n, adversarialTrace cfg A.toSolver t g₀ (base + n) ∈ C := by
  obtain ⟨base, d, hd, hcyc, _⟩ :=
    closedAtlas_periodic_input_closedCycle h hg₀ hl hP hper
  refine ⟨base, d,
    (Finset.range d).image (fun j => adversarialTrace cfg A.toSolver t g₀ (base + j)),
    hd, ?_, ?_, ?_, ?_⟩
  · refine ⟨adversarialTrace cfg A.toSolver t g₀ (base + 0), ?_⟩
    rw [Finset.mem_image]
    exact ⟨0, Finset.mem_range.mpr (by omega), rfl⟩
  · intro g hg
    rw [Finset.mem_image] at hg
    obtain ⟨j, _, rfl⟩ := hg
    exact h.toSolver_adversarialTrace_mem hg₀ hl (base + j)
  · calc ((Finset.range d).image
            (fun j => adversarialTrace cfg A.toSolver t g₀ (base + j))).card
        ≤ (Finset.range d).card := Finset.card_image_le
      _ = d := Finset.card_range d
  · intro n
    exact adversarialTrace_tail_mem_cycle_image cfg A.toSolver t g₀ hd hcyc n

/-- **Every cycle state is ω-recurrent.** Strengthens
`closedAtlas_periodic_input_infinitely_recurrent` from a single state to the
whole limit set: *every* board `g ∈ C` is revisited infinitely often. The board
at phase `j` recurs at all times `base + (j + N*d)`, because
`(j + N*d) mod d = j mod d`. So `C` is exactly the set of ω-limit points of the
play. -/
theorem closedAtlas_periodic_input_all_cycle_states_recurrent
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, t (k + P) = t k) :
    ∃ (base : ℕ) (C : Finset GameState), C.Nonempty ∧ C ⊆ S ∧
      (∀ n, adversarialTrace cfg A.toSolver t g₀ (base + n) ∈ C) ∧
      (∀ g ∈ C, ∀ N, ∃ n, N ≤ n ∧
        adversarialTrace cfg A.toSolver t g₀ (base + n) = g) := by
  obtain ⟨base, d, hd, hcyc, _⟩ :=
    closedAtlas_periodic_input_closedCycle h hg₀ hl hP hper
  refine ⟨base,
    (Finset.range d).image (fun j => adversarialTrace cfg A.toSolver t g₀ (base + j)),
    ?_, ?_, ?_, ?_⟩
  · refine ⟨adversarialTrace cfg A.toSolver t g₀ (base + 0), ?_⟩
    rw [Finset.mem_image]
    exact ⟨0, Finset.mem_range.mpr (by omega), rfl⟩
  · intro g hg
    rw [Finset.mem_image] at hg
    obtain ⟨j, _, rfl⟩ := hg
    exact h.toSolver_adversarialTrace_mem hg₀ hl (base + j)
  · intro n
    exact adversarialTrace_tail_mem_cycle_image cfg A.toSolver t g₀ hd hcyc n
  · intro g hg N
    rw [Finset.mem_image] at hg
    obtain ⟨j, _, rfl⟩ := hg
    refine ⟨j + N * d, ?_, ?_⟩
    · have h1 : N * 1 ≤ N * d := Nat.mul_le_mul (le_refl N) hd
      have h2 : N ≤ N * d := by simpa using h1
      omega
    · exact adversarialTrace_eq_of_mod_eq cfg A.toSolver t g₀ hcyc (j + N * d) j
        (by rw [Nat.add_mul_mod_self_right])

/-- **Periodicity-free recurrence in every window.** *Without any assumption on
the adversary* (no periodic input), a closed-atlas play recurs within every
length-`S.card` window: for every start time `m` there is a base time
`base ≥ m`, a period `1 ≤ period ≤ S.card`, with `trace base = trace (base +
period)` and `trace base ∈ S`. Proof: apply the Atlas base-period pigeonhole to
the *shifted* play from the visited state `trace m` under the suffix input
`fun k => t (m+k)` (legal via `LegalSequenceFrom.suffix` + `adversarialTrace_bag_from`),
then translate the shifted recurrence back through `adversarialTrace_add`. This
is the honest "the play is recurrent" statement that needs no input cycle. -/
theorem closedAtlas_recurrence_in_window
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (m : ℕ) :
    ∃ base period : ℕ, m ≤ base ∧ 1 ≤ period ∧ period ≤ S.card ∧
      adversarialTrace cfg A.toSolver t g₀ base
        = adversarialTrace cfg A.toSolver t g₀ (base + period) ∧
      adversarialTrace cfg A.toSolver t g₀ base ∈ S := by
  have hgm : adversarialTrace cfg A.toSolver t g₀ m ∈ S :=
    h.toSolver_adversarialTrace_mem hg₀ hl m
  have hsuffix : LegalSequenceFrom (adversarialTrace cfg A.toSolver t g₀ m).bag
      (fun k => t (m + k)) := by
    rw [adversarialTrace_bag_from]
    exact hl.suffix m
  obtain ⟨b, p, hp1, hple, heq⟩ :=
    h.toSolver_adversarialTrace_pigeonhole_base_period hgm hsuffix
  refine ⟨m + b, p, by omega, hp1, hple, ?_, ?_⟩
  · rw [show m + b + p = m + (b + p) from by ring,
      adversarialTrace_add cfg A.toSolver t g₀ m b,
      adversarialTrace_add cfg A.toSolver t g₀ m (b + p)]
    exact heq
  · exact h.toSolver_adversarialTrace_mem hg₀ hl (m + b)

/-- **Infinitely-often state repetition (periodicity-free).** A direct corollary:
for *every* start time `m`, the closed-atlas play repeats some state strictly
later — `∃ i j, m ≤ i < j ∧ trace i = trace j`. So the play can never become
eventually injective; it keeps returning to previously-seen boards forever,
with no assumption on the piece sequence. -/
theorem closedAtlas_repeat_after
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (m : ℕ) :
    ∃ i j : ℕ, m ≤ i ∧ i < j ∧
      adversarialTrace cfg A.toSolver t g₀ i
        = adversarialTrace cfg A.toSolver t g₀ j := by
  obtain ⟨base, period, hm, hp1, _, heq, _⟩ :=
    closedAtlas_recurrence_in_window h hg₀ hl m
  exact ⟨base, base + period, hm, by omega, heq⟩

/-- **The program's phase signature is a perfect mod-35 odometer.** Along any
`safeSolver` play from `init`, the observable phase `(bag.card, count % 10)` at
two depths `n` and `m` coincides *iff* `n ≡ m [MOD 35]`. The infinite trajectory
carries a faithful ℤ/35 clock readable from the live state alone — no two
in-cycle phases collide except by congruence. Transports
`jointPhase_eq_iff_mod_35` through `safeSolver_trace_jointPhase`. -/
theorem safeSolver_trace_phase_eq_iff_mod_35
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n m : ℕ) :
    (((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).bag.card,
       (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).board.count % 10)
      = ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init m).bag.card,
         (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init m).board.count % 10))
      ↔ n % 35 = m % 35 := by
  rw [safeSolver_trace_jointPhase h s hl n, safeSolver_trace_jointPhase h s hl m,
    jointPhase_eq_iff_mod_35]

/-- **The phase signature has period 35 along the live play.** Immediate corollary:
the program's observable phase at depth `n + 35` equals that at depth `n`, for
every `n`. The five-bag-reset clock returns to the same `(bag.card, count % 10)`
reading every 35 steps. -/
theorem safeSolver_trace_phase_periodic_35
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init (n + 35)).bag.card,
     (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init (n + 35)).board.count % 10)
      = ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init n).bag.card,
         (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init n).board.count % 10) := by
  rw [safeSolver_trace_phase_eq_iff_mod_35 h s hl (n + 35) n]
  omega

/-- **No proper iterate fixes the home phase.** For every `0 < k < 35`,
`phaseStep^[k] (7,0) ≠ (7,0)`: the automaton does not return the init phase to
itself before 35 steps. Since `phaseStep^[k] (7,0) = jointPhase k` and `(7,0) =
jointPhase 0`, this is exactly `k % 35 ≠ 0`. The lower-bound half of "order 35". -/
theorem phaseStep_iterate_ne_of_lt_35 {k : ℕ} (hk0 : 0 < k) (hk35 : k < 35) :
    phaseStep^[k] (7, 0) ≠ (7, 0) := by
  rw [← jointPhase_eq_iterate k, ← jointPhase_zero, ne_eq, jointPhase_eq_iff_mod_35]
  omega

/-- **`phaseStep` has exact order 35 on the grid.** For every `0 < k < 35` there
is a grid cell (namely the home phase `(7,0)`) that `phaseStep^[k]` does **not**
fix — so no proper iterate is the identity on the grid. Combined with
`phaseStep_iterate_35_of_mem_grid` (order divides 35) and `phaseStep_bijOn_grid`
(it is a permutation), this pins the automaton as a permutation of *exact* order
35: a single 35-cycle through all phases, minimal. -/
theorem phaseStep_minimal_order {k : ℕ} (hk0 : 0 < k) (hk35 : k < 35) :
    ∃ p ∈ (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ), phaseStep^[k] p ≠ p :=
  ⟨(7, 0), by decide, phaseStep_iterate_ne_of_lt_35 hk0 hk35⟩

/-- **A single ω-recurrent state — with no periodicity assumption.** For *any*
adversary, a closed-atlas play has one board `g ∈ S` that it returns to
infinitely often: `∀ N, ∃ n ≥ N, trace n = g`. This is the genuine ω-limit-point
statement, strictly stronger than the window recurrence of
`closedAtlas_recurrence_in_window` (which could hand back a different state each
window). Proof by infinite pigeonhole: the fibers `{n | trace n = g}` over `g ∈ S`
cover all of `ℕ` (every `trace n` lies in `S`); were each fiber finite, their
finite union would make `ℕ` finite — contradiction. So some fiber is unbounded. -/
theorem closedAtlas_omega_recurrent_state
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    ∃ g ∈ S, ∀ N, ∃ n, N ≤ n ∧ adversarialTrace cfg A.toSolver t g₀ n = g := by
  by_contra hcon
  push_neg at hcon
  have hcover : (Set.univ : Set ℕ) ⊆
      ⋃ g ∈ (S : Set GameState),
        {n | adversarialTrace cfg A.toSolver t g₀ n = g} := by
    intro n _
    exact Set.mem_biUnion
      (Finset.mem_coe.mpr (h.toSolver_adversarialTrace_mem hg₀ hl n)) rfl
  have hfin : (⋃ g ∈ (S : Set GameState),
      {n | adversarialTrace cfg A.toSolver t g₀ n = g}).Finite := by
    refine S.finite_toSet.biUnion ?_
    intro g hg
    obtain ⟨N, hN⟩ := hcon g (Finset.mem_coe.mp hg)
    refine Set.Finite.subset (Set.finite_Iio N) ?_
    intro n hn
    rw [Set.mem_setOf_eq] at hn
    rw [Set.mem_Iio]
    by_contra hle
    push_neg at hle
    exact hN n hle hn
  exact Set.infinite_univ (hfin.subset hcover)

/-- Strengthening of `closedAtlas_omega_recurrent_state`: under any closed atlas
and any legal adversary, there is a state `g ∈ S` whose set of visit-times is
genuinely infinite (hit at infinitely many distinct steps, not merely
unboundedly often). -/
theorem closedAtlas_omega_recurrent_state_infinite_times
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    ∃ g ∈ S, {n | adversarialTrace cfg A.toSolver t g₀ n = g}.Infinite := by
  obtain ⟨g, hgS, hrec⟩ := closedAtlas_omega_recurrent_state h hg₀ hl
  refine ⟨g, hgS, ?_⟩
  intro hfin
  obtain ⟨ub, hub⟩ := hfin.bddAbove
  obtain ⟨n, hn, hng⟩ := hrec (ub + 1)
  have hnub : n ≤ ub := hub hng
  omega

/-- The adversarial trace of a materialized closed atlas never escapes `S`:
its entire range is contained in the closed table. This is the "the solver
stays inside its safe table forever" invariant, with no periodicity assumption
on the adversary. -/
theorem closedAtlas_trace_range_subset
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    Set.range (fun n => adversarialTrace cfg A.toSolver t g₀ n) ⊆ (S : Set GameState) := by
  rintro g ⟨n, rfl⟩
  exact Finset.mem_coe.mpr (h.toSolver_adversarialTrace_mem hg₀ hl n)

/-- Consequently the range of the adversarial trace is a finite set (bounded by
the finite closed table `S`), even though the play itself is infinite. -/
theorem closedAtlas_trace_range_finite
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    (Set.range (fun n => adversarialTrace cfg A.toSolver t g₀ n)).Finite :=
  S.finite_toSet.subset (closedAtlas_trace_range_subset h hg₀ hl)

/-- Quantitative footprint bound: the (finite) set of distinct states the
adversarial trace ever visits has cardinality at most `S.card`. The infinite
play touches no more states than the closed table holds. -/
theorem closedAtlas_trace_range_card_le
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    (closedAtlas_trace_range_finite h hg₀ hl).toFinset.card ≤ S.card := by
  apply Finset.card_le_card
  intro g hg
  rw [Set.Finite.mem_toFinset] at hg
  exact Finset.mem_coe.mp (closedAtlas_trace_range_subset h hg₀ hl hg)

/-- The footprint is nonempty (it contains at least the start state), so the
distinct-state count is sandwiched: `1 ≤ footprint.card ≤ S.card`. -/
theorem closedAtlas_trace_range_card_pos
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    1 ≤ (closedAtlas_trace_range_finite h hg₀ hl).toFinset.card := by
  have hne : (closedAtlas_trace_range_finite h hg₀ hl).toFinset.Nonempty := by
    rw [Set.Finite.toFinset_nonempty]
    exact Set.range_nonempty _
  exact hne.card_pos

/-- Strictly-increasing enumeration of visit times: the ω-recurrent state `g`
(from `closedAtlas_omega_recurrent_state_infinite_times`) is visited along a
strictly monotone infinite time sequence `τ` with `k ≤ τ k`. This packages the
infinite visit-set into an actual schedule of return times. -/
theorem closedAtlas_omega_recurrent_visit_seq
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    ∃ g ∈ S, ∃ τ : ℕ → ℕ, StrictMono τ ∧ (∀ k, k ≤ τ k) ∧
      ∀ k, adversarialTrace cfg A.toSolver t g₀ (τ k) = g := by
  obtain ⟨g, hgS, hinf⟩ := closedAtlas_omega_recurrent_state_infinite_times h hg₀ hl
  refine ⟨g, hgS, Nat.nth (fun n => adversarialTrace cfg A.toSolver t g₀ n = g),
    Nat.nth_strictMono hinf, (Nat.nth_strictMono hinf).id_le, ?_⟩
  intro k
  exact Nat.nth_mem_of_infinite hinf k

/-- Infinitely many ordered return-pairs: beyond any horizon `N` there are two
times `N ≤ i < j` both landing on the recurrent state `g`. Each such pair is a
`g → … → g` loop of the adversarial play starting past `N`. -/
theorem closedAtlas_omega_recurrent_infinitely_many_repeats
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    ∃ g ∈ S, ∀ N, ∃ i j, N ≤ i ∧ i < j ∧
      adversarialTrace cfg A.toSolver t g₀ i = g ∧
      adversarialTrace cfg A.toSolver t g₀ j = g := by
  obtain ⟨g, hgS, τ, hmono, hid, hmem⟩ := closedAtlas_omega_recurrent_visit_seq h hg₀ hl
  refine ⟨g, hgS, fun N => ?_⟩
  exact ⟨τ N, τ (N + 1), hid N, hmono (Nat.lt_succ_self N), hmem N, hmem (N + 1)⟩

/-- **Bag-boundary characterization.** The bag phase is "full" (`= 7`, a fresh
7-bag) exactly at the multiples of 7. -/
theorem bagPhase_eq_seven_iff (n : ℕ) : bagPhase n = 7 ↔ 7 ∣ n := by
  unfold bagPhase
  omega

/-- **Board-reset characterization.** The board phase returns to `0` (cell count
≡ 0 mod 10) exactly at the multiples of 5 — five tetromino placements add 20
cells, a multiple of 10. -/
theorem boardPhase_eq_zero_iff (n : ℕ) : boardPhase n = 0 ↔ 5 ∣ n := by
  unfold boardPhase
  omega

/-- **Odometer reset.** The joint phase returns to its origin `(7, 0)` exactly at
the multiples of 35 = lcm(7, 5): the bag and the board-count realign only every
35 placements. -/
theorem jointPhase_eq_origin_iff (n : ℕ) : jointPhase n = (7, 0) ↔ 35 ∣ n := by
  unfold jointPhase bagPhase boardPhase
  rw [Prod.mk.injEq]
  omega

/-- **Solver-level board reset.** Lifting `boardPhase_eq_zero_iff` through
`safeSolver_trace_jointPhase`: the live solving program's board-cell count is
≡ 0 mod 10 exactly at move counts divisible by 5. -/
theorem safeSolver_trace_board_reset_iff
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board.count % 10 = 0 ↔ 5 ∣ n := by
  have hp := safeSolver_trace_jointPhase h s hl n
  unfold jointPhase at hp
  rw [Prod.mk.injEq] at hp
  rw [hp.2]
  exact boardPhase_eq_zero_iff n

/-- **Solver-level odometer reset.** Combining the bag-boundary and board-reset
lifts: the live solving program is simultaneously at a fresh full bag AND a
board-count ≡ 0 mod 10 exactly at move counts divisible by 35 = lcm(7, 5). -/
theorem safeSolver_trace_phase_origin_iff
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).bag.card = 7 ∧
     (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board.count % 10 = 0) ↔ 35 ∣ n := by
  have hp := safeSolver_trace_jointPhase h s hl n
  unfold jointPhase at hp
  rw [Prod.mk.injEq] at hp
  rw [hp.1, hp.2, bagPhase_eq_seven_iff, boardPhase_eq_zero_iff]
  omega

/-- **No phase repeats within any 35-move window.** `jointPhase` is injective on
every length-35 interval `[m, m+35)`, not just the first one: in any 35
consecutive placements the solver visits 35 distinct phases. Generalises
`jointPhase_injOn_range_35` (the `m = 0` case) to an arbitrary start. -/
theorem jointPhase_injOn_window (m : ℕ) :
    Set.InjOn jointPhase (Set.Ico m (m + 35)) := by
  intro i hi j hj hij
  simp only [Set.mem_Ico] at hi hj
  rw [jointPhase_eq_iff_mod_35] at hij
  omega

/-- **Solver phase discoverability (image form).** Over its first 35 moves the
live `safeSolver` exhibits exactly the 7×5 phase grid `{1,…,7} × {0,4,8,2,6}` of
observable `(bag.card, count % 10)` pairs — nothing missing, nothing extra. -/
theorem safeSolver_trace_phase_image_eq_grid
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) :
    (Finset.range 35).image (fun n =>
      ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).bag.card,
       (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).board.count % 10))
      = (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ) := by
  have hfun : (fun n =>
      ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).bag.card,
       (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).board.count % 10)) = jointPhase := by
    funext n
    exact safeSolver_trace_jointPhase h s hl n
  rw [hfun, jointPhase_image_range_35_eq_grid]

/-- **Solver phase discoverability (pointwise form).** Every grid cell `(k, r)`
with `k ∈ {1,…,7}`, `r ∈ {0,4,8,2,6}` is realized by the live solver at some move
`n < 35`: the program reaches every observable phase within one period. -/
theorem safeSolver_trace_discovers_phase
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {k r : ℕ} (hk : k ∈ Finset.Icc 1 7) (hr : r ∈ ({0, 4, 8, 2, 6} : Finset ℕ)) :
    ∃ n < 35,
      ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).bag.card,
       (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).board.count % 10) = (k, r) := by
  obtain ⟨n, hn, hnp⟩ := exists_eq_jointPhase_of_mem_grid hk hr
  refine ⟨n, hn, ?_⟩
  rw [safeSolver_trace_jointPhase h s hl n]
  exact hnp

/-- **Fully-windowed state repetition.** Strengthens `closedAtlas_repeat_after`:
not only does the closed-atlas play repeat some state strictly after every start
time `m`, but *both* repeat indices land inside the bounded window
`[m, m + |S|]`. The pigeonhole fires within `S.card` steps of any checkpoint, so
the play revisits a board at least once per `|S|`-length window — a uniform
recurrence rate, with no assumption on the piece sequence. -/
theorem closedAtlas_repeat_in_window
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (m : ℕ) :
    ∃ i j : ℕ, m ≤ i ∧ i < j ∧ j ≤ m + S.card ∧
      adversarialTrace cfg A.toSolver t g₀ i
        = adversarialTrace cfg A.toSolver t g₀ j := by
  have hgm : adversarialTrace cfg A.toSolver t g₀ m ∈ S :=
    h.toSolver_adversarialTrace_mem hg₀ hl m
  have hsuffix : LegalSequenceFrom (adversarialTrace cfg A.toSolver t g₀ m).bag
      (fun k => t (m + k)) := by
    rw [adversarialTrace_bag_from]
    exact hl.suffix m
  obtain ⟨i, j, hij, hj, heq⟩ :=
    h.toSolver_adversarialTrace_pigeonhole_lt hgm hsuffix
  refine ⟨m + i, m + j, by omega, by omega, by omega, ?_⟩
  rw [adversarialTrace_add cfg A.toSolver t g₀ m i,
    adversarialTrace_add cfg A.toSolver t g₀ m j]
  exact heq

/-- **No injective window.** Equivalent contrapositive of
`closedAtlas_repeat_in_window`: the trace map is never injective on any window
`[m, m + |S|]`. Because that closed interval has `|S| + 1` integer points but the
play takes values in the `|S|`-element set `S`, the pigeonhole forbids injectivity
— a clean finite-to-finite collision statement for the closed-atlas solver. -/
theorem closedAtlas_not_injOn_window
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (m : ℕ) :
    ¬ Set.InjOn (fun n => adversarialTrace cfg A.toSolver t g₀ n)
        (Set.Icc m (m + S.card)) := by
  intro hinj
  obtain ⟨i, j, hmi, hij, hjm, heq⟩ := closedAtlas_repeat_in_window h hg₀ hl m
  have hi : i ∈ Set.Icc m (m + S.card) := Set.mem_Icc.mpr ⟨hmi, by omega⟩
  have hj : j ∈ Set.Icc m (m + S.card) := Set.mem_Icc.mpr ⟨by omega, hjm⟩
  have : i = j := hinj hi hj heq
  omega

/-- **Bounded recurrence gap.** Sharpens `closedAtlas_repeat_in_window` to a
statement about the *gap* between collisions: from every checkpoint `m`, the
closed-atlas play repeats some state with a gap `j - i ≤ |S|`. So the recurrence
rate is uniform — never more than `|S|` steps elapse without the play closing a
loop on a state it has already seen since `m`. No assumption on the piece
sequence. -/
theorem closedAtlas_repeat_gap_le_card
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (m : ℕ) :
    ∃ i j : ℕ, m ≤ i ∧ i < j ∧ j - i ≤ S.card ∧
      adversarialTrace cfg A.toSolver t g₀ i
        = adversarialTrace cfg A.toSolver t g₀ j := by
  obtain ⟨i, j, hmi, hij, hjm, heq⟩ := closedAtlas_repeat_in_window h hg₀ hl m
  exact ⟨i, j, hmi, hij, by omega, heq⟩

/-- **Window compression bound (quantitative pigeonhole core).** Over any window
of `|S| + 1` consecutive depths `Set.Icc m (m + |S|)`, the closed-atlas play
visits at most `|S|` distinct states — because every trace value lies in the
`|S|`-element set `S`. Since the window has one more integer point than the
codomain admits distinct values, this *is* the pigeonhole underlying
`closedAtlas_not_injOn_window`: more inputs than possible outputs forces a
collision. -/
theorem closedAtlas_trace_image_window_card_le
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (m : ℕ) :
    ((Finset.Icc m (m + S.card)).image
        (fun n => adversarialTrace cfg A.toSolver t g₀ n)).card ≤ S.card := by
  apply Finset.card_le_card
  intro g hg
  rw [Finset.mem_image] at hg
  obtain ⟨n, _, rfl⟩ := hg
  exact h.toSolver_adversarialTrace_mem hg₀ hl n

/-- **The solving program survives forever and keeps coming home.** The headline
operational reading of a closed atlas: against *any* legal piece sequence, the
materialised solver `A.toSolver` (a) never tops out at any depth, and (b) there is
a state `g ∈ S` it revisits infinitely often. This packages the per-depth
non-loss invariant (`toSolver_adversarialTrace_not_lost`) together with the
ω-recurrence (`closedAtlas_omega_recurrent_state_infinite_times`) into the single
"plays forever, returning endlessly" statement that M2 closed cycles are meant to
deliver. -/
theorem closedAtlas_survives_and_recurs
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    ∃ g ∈ S,
      (∀ n, ¬ (adversarialTrace cfg A.toSolver t g₀ n).lost cfg) ∧
      {n | adversarialTrace cfg A.toSolver t g₀ n = g}.Infinite := by
  obtain ⟨g, hgS, hinf⟩ := closedAtlas_omega_recurrent_state_infinite_times h hg₀ hl
  exact ⟨g, hgS, fun n => h.toSolver_adversarialTrace_not_lost hg₀ hl n, hinf⟩

/-- **Infinite non-losing play inside a finite board orbit.** The solver's
infinite trajectory never loses, yet the *set* of distinct boards it ever touches
is finite — bounded by the closed table `S`. This is the precise sense in which a
closed atlas "solves" Tetris by a finite cycle: an unbounded survival play whose
state footprint is a finite orbit. Combines `toSolver_adversarialTrace_not_lost`
with `closedAtlas_trace_range_finite`. -/
theorem closedAtlas_survives_in_finite_orbit
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    (∀ n, ¬ (adversarialTrace cfg A.toSolver t g₀ n).lost cfg) ∧
      (Set.range (fun n => adversarialTrace cfg A.toSolver t g₀ n)).Finite :=
  ⟨fun n => h.toSolver_adversarialTrace_not_lost hg₀ hl n,
   closedAtlas_trace_range_finite h hg₀ hl⟩

/-- **The bag refills at infinitely many depths (`Set.Infinite` form).** Upgrades
the cofinal `safeSolver_trace_full_bag_recurs` (`∀ N, ∃ n ≥ N, …`) to the standard
order-theoretic statement: the depth-set on which the live solver's bag equals
`Bag.full` is genuinely infinite. Proof by the `bddAbove`-contradiction dual of
iter-37: a finite set of naturals is bounded above, but the cofinal recurrence
produces a witness past any bound. -/
theorem safeSolver_trace_full_bag_infinite
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) :
    {n | (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).bag = Bag.full}.Infinite := by
  intro hfin
  obtain ⟨ub, hub⟩ := hfin.bddAbove
  obtain ⟨n, hn, hp⟩ := safeSolver_trace_full_bag_recurs h s hl (ub + 1)
  have hnub : n ≤ ub := hub hp
  omega

/-- **The board count returns to a width-multiple at infinitely many depths
(`Set.Infinite` form).** Companion upgrade of
`safeSolver_trace_count_mod_ten_zero_recurs`: the depth-set on which the board
holds an exact multiple of `10` cells is infinite. Same `bddAbove`-contradiction
schema. -/
theorem safeSolver_trace_count_mod_ten_zero_infinite
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) :
    {n | (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board.count % 10 = 0}.Infinite := by
  intro hfin
  obtain ⟨ub, hub⟩ := hfin.bddAbove
  obtain ⟨n, hn, hp⟩ := safeSolver_trace_count_mod_ten_zero_recurs h s hl (ub + 1)
  have hnub : n ≤ ub := hub hp
  omega

/-- **Every grid phase recurs at an infinite depth-set (`Set.Infinite` form).**
Generic upgrade of `safeSolver_trace_phase_recurs`: for *any* cell `φ` of the 7×5
grid, the depth-set on which the live solver's phase signature equals `φ` is
infinite. Subsumes both iter-46 projections (`bag = full`, `count % 10 = 0`) and
the home phase as special cases. Same `bddAbove`-contradiction schema. -/
theorem safeSolver_trace_phase_infinite
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) {φ : ℕ × ℕ}
    (hφ : φ ∈ (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ)) :
    {n | ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).bag.card,
       (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board.count % 10) = φ}.Infinite := by
  intro hfin
  obtain ⟨ub, hub⟩ := hfin.bddAbove
  obtain ⟨n, hn, hp⟩ := safeSolver_trace_phase_recurs h s hl hφ (ub + 1)
  have hnub : n ≤ ub := hub hp
  omega

/-- **A phase is visited infinitely often iff it is a grid cell.** Sharpens
`safeSolver_trace_visits_phase_iff` (visited *at all* ↔ grid) to visited
*infinitely often* ↔ grid. Because the phase space is finite and the play
infinite, occurring once already forces occurring forever — the forward direction
extracts one witness via `Set.Infinite.nonempty` and feeds it to
`safeSolver_trace_visits_phase_iff`; the reverse is `safeSolver_trace_phase_infinite`.
-/
theorem safeSolver_trace_phase_infinite_iff
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (φ : ℕ × ℕ) :
    {n | ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).bag.card,
       (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board.count % 10) = φ}.Infinite
      ↔ φ ∈ (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ) := by
  constructor
  · intro hinf
    obtain ⟨n, hn⟩ := hinf.nonempty
    exact (safeSolver_trace_visits_phase_iff h s hl φ).mp ⟨n, hn⟩
  · intro hφ
    exact safeSolver_trace_phase_infinite h s hl hφ

/-- **The live solver's bag is always nonempty and never over-full.** At every
depth `n` the bag holds between `1` and `7` pieces: `1 ≤ bag.card ≤ 7`. The lower
bound says the program never faces an empty bag (there is always a piece to be
drawn next); the upper bound caps the bag at a full seven-piece reservoir. A clean
operational invariant, read off the exact bag clock `bag.card = 7 - n % 7`. -/
theorem safeSolver_trace_bag_card_bounds
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    1 ≤ (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).bag.card
      ∧ (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).bag.card ≤ 7 := by
  rw [safeSolver_trace_bag_card_eq h s hl n]
  omega

/-- **Bag bounds are a whole-atlas layer invariant.** Every state discovered at
depth `n` — not just those on one solver trace — has `1 ≤ bag.card ≤ 7`. The atlas
analog of `safeSolver_trace_bag_card_bounds`, read off the layer bag clock
`reachableLayer_bag_card`. Confirms no reachable state in the atlas ever stalls on
an empty bag. -/
theorem reachableLayer_bag_card_bounds {n : ℕ} {g : GameState}
    (hg : g ∈ reachableLayer n) :
    1 ≤ g.bag.card ∧ g.bag.card ≤ 7 := by
  rw [reachableLayer_bag_card hg]
  omega

/-- **The program draws from a FINITE repertoire of moves.** Although `Placement`
itself is infinite (its `col` field is an unbounded `ℕ`), the set of placements the
`safeSolver` ever actually plays — over the entire infinite trajectory, across all
depths `n` — is finite. It is contained in the finite union, over the 7 pieces, of
the precomputed candidate sets `Placement.allValidFor standard p`. So the infinite
play is a walk through a *finite* policy alphabet: the program is, in effect, a
finite-state controller. -/
theorem safeSolver_trace_moves_finite (s : ℕ → Piece) (hl : LegalSequence s) :
    (Set.range (fun n => safeSolver GameConfig.standard
        (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n) (s n))).Finite := by
  apply Set.Finite.subset
    (Finset.univ.biUnion
      (fun p : Piece => Placement.allValidFor GameConfig.standard p)).finite_toSet
  rintro pl ⟨n, rfl⟩
  exact Finset.mem_coe.mpr (Finset.mem_biUnion.mpr
    ⟨s n, Finset.mem_univ (s n), safeSolver_trace_move_mem_allValidFor s hl n⟩)

/-- **The move repertoire has at most 280 distinct placements.** Quantitative form
of `safeSolver_trace_moves_finite`: the total move alphabet is bounded by
`7 × 40 = 280` (seven pieces, each contributing at most 40 candidate placements via
`Placement.standard_allValidFor_card_le`). A hard, piece-count-times-branching
ceiling on the size of the policy table the infinite-survival program needs. -/
theorem safeSolver_trace_moves_card_le (s : ℕ → Piece) (hl : LegalSequence s) :
    (safeSolver_trace_moves_finite s hl).toFinset.card ≤ 280 := by
  have hsub : (safeSolver_trace_moves_finite s hl).toFinset ⊆
      Finset.univ.biUnion
        (fun p : Piece => Placement.allValidFor GameConfig.standard p) := by
    intro pl hpl
    rw [Set.Finite.mem_toFinset, Set.mem_range] at hpl
    obtain ⟨n, rfl⟩ := hpl
    exact Finset.mem_biUnion.mpr
      ⟨s n, Finset.mem_univ (s n), safeSolver_trace_move_mem_allValidFor s hl n⟩
  refine le_trans (Finset.card_le_card hsub) ?_
  refine le_trans Finset.card_biUnion_le ?_
  refine le_trans (Finset.sum_le_sum
    (fun p _ => Placement.standard_allValidFor_card_le p)) ?_
  simp

/-- **`jointPhase` is surjective onto the grid over EVERY length-35 window.** Not
just `[0,35)` (`jointPhase_image_range_35_eq_grid`): for any start `m`, the image
of `[m, m+35)` under `jointPhase` is exactly the full 7×5 grid. Proof: the image
is ⊆ grid (`jointPhase_mem_grid`) and has card 35 (35 inputs, injective on the
window via `jointPhase_injOn_window`), and grid also has card 35, so subset +
equal-card forces equality. The phase clock is genuinely periodic-surjective:
every window sees every phase. -/
theorem jointPhase_image_Ico_eq_grid (m : ℕ) :
    (Finset.Ico m (m + 35)).image jointPhase
      = (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ) := by
  have hinj : Set.InjOn jointPhase ↑(Finset.Ico m (m + 35)) := by
    rw [Finset.coe_Ico]; exact jointPhase_injOn_window m
  have himg : ((Finset.Ico m (m + 35)).image jointPhase).card = 35 := by
    rw [Finset.card_image_of_injOn hinj, Nat.card_Ico]
    omega
  have hgrid : ((Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ)).card = 35 := by
    rw [← jointPhase_image_range_35_eq_grid]; exact jointPhase_image_range_35_card
  refine Finset.eq_of_subset_of_card_le ?_ ?_
  · intro φ hφ
    rw [Finset.mem_image] at hφ
    obtain ⟨n, _, rfl⟩ := hφ
    exact jointPhase_mem_grid n
  · exact le_of_eq (hgrid.trans himg.symm)

/-- **Solver phase discoverability is window-uniform (image form).** Generalises
`safeSolver_trace_phase_image_eq_grid` from the first window to ANY start `m`:
across any 35 consecutive moves `[m, m+35)` the live `safeSolver` exhibits exactly
the full 7×5 phase grid. No matter how late an observer starts watching, one
period's worth of play reveals every observable `(bag.card, count % 10)` phase. -/
theorem safeSolver_trace_phase_image_window_eq_grid
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (m : ℕ) :
    (Finset.Ico m (m + 35)).image (fun n =>
      ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).bag.card,
       (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).board.count % 10))
      = (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ) := by
  have hfun : (fun n =>
      ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).bag.card,
       (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).board.count % 10)) = jointPhase := by
    funext n
    exact safeSolver_trace_jointPhase h s hl n
  rw [hfun, jointPhase_image_Ico_eq_grid]

/-- **Solver phase discoverability is window-uniform (pointwise form).** Every
grid cell `(k, r)` is realized by the live solver at some move inside any window
`[m, m+35)`. Generalises `safeSolver_trace_discovers_phase` (the `m = 0` case):
from any checkpoint, the program reaches every observable phase within the next
35 placements — a uniform recurrence rate for phase discovery. -/
theorem safeSolver_trace_discovers_phase_window
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (m : ℕ)
    {k r : ℕ} (hk : k ∈ Finset.Icc 1 7) (hr : r ∈ ({0, 4, 8, 2, 6} : Finset ℕ)) :
    ∃ n, m ≤ n ∧ n < m + 35 ∧
      ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).bag.card,
       (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).board.count % 10) = (k, r) := by
  have hmem : (k, r) ∈ (Finset.Ico m (m + 35)).image (fun n =>
      ((adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).bag.card,
       (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init n).board.count % 10)) := by
    rw [safeSolver_trace_phase_image_window_eq_grid h s hl m]
    exact Finset.mk_mem_product hk hr
  rw [Finset.mem_image] at hmem
  obtain ⟨n, hn, hnp⟩ := hmem
  rw [Finset.mem_Ico] at hn
  exact ⟨n, hn.1, hn.2, hnp⟩

/-- **Any 35 consecutive layers are pairwise disjoint.** Over any window of depths
`[m, m+35)` the reachable layers share no state: a board discovered at one depth in
the window is discovered at no other depth in the window. Windowed form of the
per-pair `reachableLayer_disjoint_of_mod_35_ne` (two depths in a length-35 interval
have distinct residues mod 35, so their phases — hence their layers — differ).
The atlas's per-period discovery structure is a genuine partition. -/
theorem reachableLayer_pairwiseDisjoint_Ico (m : ℕ) :
    (↑(Finset.Ico m (m + 35)) : Set ℕ).PairwiseDisjoint reachableLayer := by
  intro n hn k hk hnk
  simp only [Finset.coe_Ico, Set.mem_Ico] at hn hk
  exact reachableLayer_disjoint_of_mod_35_ne (by omega)

/-- **Exact per-period atlas count: no double-counting across a window.** Because
35 consecutive layers are pairwise disjoint
(`reachableLayer_pairwiseDisjoint_Ico`), the number of distinct states discovered
across the window `[m, m+35)` is *exactly* the sum of the individual layer sizes —
the union's cardinality splits with no overlap. The headline counting identity for
atlas discoverability over one phase period. -/
theorem reachableLayer_biUnion_Ico_card (m : ℕ) :
    ((Finset.Ico m (m + 35)).biUnion reachableLayer).card
      = ∑ n ∈ Finset.Ico m (m + 35), (reachableLayer n).card :=
  Finset.card_biUnion (reachableLayer_pairwiseDisjoint_Ico m)

/-- **The first 35 layers (one full reset block) are pairwise disjoint.** The
`Finset.range 35` form of `reachableLayer_pairwiseDisjoint_Ico` — used directly for
the materialized atlas `reachableBeforeBlock`, which is the `biUnion` of these 35
layers. -/
theorem reachableLayer_pairwiseDisjoint_range_35 :
    (↑(Finset.range 35) : Set ℕ).PairwiseDisjoint reachableLayer := by
  intro n hn k hk hnk
  simp only [Finset.coe_range, Set.mem_Iio] at hn hk
  exact reachableLayer_disjoint_of_mod_35_ne (by omega)

/-- **The materialized pre-reset atlas has exactly ∑ layer-sizes states.** The
concrete finite atlas `reachableBeforeBlock` (= union of the 35 exact-depth layers
before the reset boundary, used to build the winning-cycle set) has cardinality
*exactly* the sum of the individual layer cardinalities — no state is discovered at
two different depths within the block. Connects the per-period partition
(`reachableLayer_pairwiseDisjoint_range_35`) to the concrete object the M2/M3
construction materializes. -/
theorem reachableBeforeBlock_card_eq_sum :
    reachableBeforeBlock.card
      = ∑ n ∈ Finset.range 35, (reachableLayer n).card := by
  unfold reachableBeforeBlock
  rw [blockLength_eq]
  exact Finset.card_biUnion reachableLayer_pairwiseDisjoint_range_35

/-- **The materialized atlas holds at least 35 distinct states.** When `init` is
safe, the pre-reset atlas `reachableBeforeBlock` has cardinality `≥ 35` — one for
each of the 35 nonempty, pairwise-disjoint layers in the block. Concrete lower
bound matching the 35-cell phase grid: the atlas is at least as large as the number
of observable phases it must realize. -/
theorem reachableBeforeBlock_card_ge_35
    (h : GameState.init ∈ safe GameConfig.standard) :
    35 ≤ reachableBeforeBlock.card := by
  rw [reachableBeforeBlock_card_eq_sum]
  calc 35 = ∑ _n ∈ Finset.range 35, 1 := by simp
    _ ≤ ∑ n ∈ Finset.range 35, (reachableLayer n).card :=
        Finset.sum_le_sum (fun n _ => (reachableLayer_nonempty h n).card_pos)

/-- **Phase discoverability in the materialized atlas (pointwise).** When `init` is
safe, every cell `(k, r)` of the 7×5 phase grid `{1,…,7} × {0,4,8,2,6}` is realized
by some concrete state in the finite atlas `reachableBeforeBlock` — its phase
signature `(bag.card, count % 10)` hits `(k, r)`. Combined with
`reachableAt_phase_mem_grid` (every atlas state lies on the grid), the atlas's phase
projection is *exactly* the grid: nothing missing, nothing extra. We state it
pointwise (`∃ g ∈ …`) rather than as an image equality precisely because membership
in `reachableBeforeBlock.image _` forces the noncomputable atlas to whnf, whereas the
existential keeps the atlas folded behind a binder. Sharpens
`reachableBeforeBlock_card_ge_35`: the ≥35 distinct states cover all 35 phases. -/
theorem reachableBeforeBlock_realizes_grid_cell
    (h : GameState.init ∈ safe GameConfig.standard)
    {k r : ℕ} (hk : k ∈ Finset.Icc 1 7) (hr : r ∈ ({0, 4, 8, 2, 6} : Finset ℕ)) :
    ∃ g ∈ reachableBeforeBlock, (g.bag.card, g.board.count % 10) = (k, r) := by
  obtain ⟨n, hn, hnφ⟩ := exists_eq_jointPhase_of_mem_grid hk hr
  obtain ⟨g, hgn⟩ := reachableLayer_nonempty h n
  refine ⟨g, ?_, ?_⟩
  · rw [reachableBeforeBlock, Finset.mem_biUnion, blockLength_eq]
    exact ⟨n, Finset.mem_range.mpr hn, hgn⟩
  · rw [reachableLayer_jointPhase hgn]; exact hnφ

/-! ### The live solver's board ranges over only finitely many configurations

The count bound (`safeSolver_trace_count_le_capacity`) and column-height bound
(`safeSolver_trace_colHeight_le_twenty`) are *cardinality/geometry* facts. Here we
push to the **cell-set** level: every trace board, being reachable (well-formed:
columns `< cols`) and non-lost (rows `< rows`), is a subset of the fixed finite grid
`range cols ×ˢ range rows`. Hence the solver's board takes values in the (finite)
powerset of that grid: the actual program confines its board to one of finitely
many configurations forever, no matter the adversarial piece stream. This is the
concrete-program finiteness that underwrites pigeonhole recurrence of the *real*
`safeSolver` play (not just abstract closed-table play). -/

/-- **The live solver's board fits the playfield grid (cell-level).** Every cell of
every trace board lies in `[0,cols) × [0,rows)`: columns are bounded by
well-formedness of reachable states (`Reachable.wf`), rows by non-lostness
(`not_lost_in_field`). Strengthens the count bound `≤ cols*rows` to an actual subset
relation `board ⊆ range cols ×ˢ range rows`. -/
theorem safeSolver_trace_board_subset_grid
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board
      ⊆ Finset.range GameConfig.standard.cols ×ˢ Finset.range GameConfig.standard.rows := by
  intro p hp
  rw [Finset.mem_product, Finset.mem_range, Finset.mem_range]
  have hr := (safeSolver_adversarialTrace_reachableAt h s hl n).1.reachable
  have hnl := safe_not_lost (safeSolver_adversarialTrace_reachableAt h s hl n).2
  exact ⟨hr.wf p hp, Nat.lt_of_not_le (fun hc => hnl ⟨p, hp, hc⟩)⟩

/-- **The live solver visits only finitely many board configurations.** The range of
`n ↦ (trace n).board` is a finite set: each board is a subset of the fixed grid
`range cols ×ˢ range rows`, so it lies in that grid's (finite) powerset. The actual
program's board, over its entire infinite play against any legal stream, takes only
finitely many distinct values — the geometric core of recurrence for the real
solver. -/
theorem safeSolver_trace_boards_finite
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) :
    (Set.range (fun n => (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n).board)).Finite := by
  apply Set.Finite.subset
    (Finset.range GameConfig.standard.cols ×ˢ
      Finset.range GameConfig.standard.rows).powerset.finite_toSet
  rintro b ⟨n, rfl⟩
  exact Finset.mem_coe.mpr (Finset.mem_powerset.mpr (safeSolver_trace_board_subset_grid h s hl n))

/-- **Generic ω-recurrence from a finite range.** If a sequence `f : ℕ → α` takes only
finitely many values, then some value `a` is attained infinitely often: for every
threshold `N` there is an index `n ≥ N` with `f n = a`. Pigeonhole over the
finite range: if no value recurred past its own threshold, `ℕ` would be a finite
union of bounded fibers, contradicting `Set.infinite_univ`. -/
theorem exists_omega_recurrent_of_range_finite {α : Type*} {f : ℕ → α}
    (hfin : (Set.range f).Finite) :
    ∃ a, ∀ N, ∃ n, N ≤ n ∧ f n = a := by
  by_contra hcon
  push_neg at hcon
  have hcover : (Set.univ : Set ℕ) ⊆ ⋃ a ∈ Set.range f, {n | f n = a} := by
    intro n _
    exact Set.mem_biUnion (Set.mem_range_self n) rfl
  have hfin2 : (⋃ a ∈ Set.range f, {n | f n = a}).Finite := by
    refine hfin.biUnion ?_
    intro a _
    obtain ⟨N, hN⟩ := hcon a
    refine Set.Finite.subset (Set.finite_Iio N) ?_
    intro n hn
    rw [Set.mem_setOf_eq] at hn
    rw [Set.mem_Iio]
    by_contra hle
    push_neg at hle
    exact hN n hle hn
  exact Set.infinite_univ (hfin2.subset hcover)

/-- **The live solver visits only finitely many game states.** Lifting
`safeSolver_trace_boards_finite` through `GameState = {board, bag}`: the state's image
under `g ↦ (g.board, g.bag)` lands in `(range of trace boards) ×ˢ univ`, a finite
product (boards finite above; `Bag = Finset Piece` is a `Fintype`). Since that pairing
is injective by structure eta, the state range itself is finite. -/
theorem safeSolver_trace_states_finite
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) :
    (Set.range (fun n => adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n)).Finite := by
  refine Set.Finite.of_finite_image (f := fun g : GameState => (g.board, g.bag)) ?_ ?_
  · refine Set.Finite.subset
      ((safeSolver_trace_boards_finite h s hl).prod (Set.finite_univ)) ?_
    rintro _ ⟨g, ⟨n, rfl⟩, rfl⟩
    exact ⟨Set.mem_range_self n, Set.mem_univ _⟩
  · intro a _ b _ hab
    have hb : a.board = b.board := congrArg Prod.fst hab
    have hg : a.bag = b.bag := congrArg Prod.snd hab
    calc a = (⟨a.board, a.bag⟩ : GameState) := rfl
      _ = ⟨b.board, b.bag⟩ := by rw [hb, hg]
      _ = b := rfl

/-- **The real solver's play is ω-recurrent.** Combining the state-range finiteness with
the generic pigeonhole: along the actual `safeSolver` trace against any legal stream,
some game state `g` recurs infinitely often — for every horizon `N` there is a later
step `n ≥ N` revisiting `g`. This is the concrete recurrence theorem for the live
program, the operational analog of the abstract `closedAtlas_omega_recurrent_state`. -/
theorem safeSolver_trace_omega_recurrent_state
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) :
    ∃ g, ∀ N, ∃ n, N ≤ n ∧ adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init n = g :=
  exists_omega_recurrent_of_range_finite (safeSolver_trace_states_finite h s hl)

/-- **ω-recurrence yields arbitrarily-late repeats.** If some value of `f` recurs past
every threshold, then above ANY horizon `N` there are two distinct indices `i < j`
(both `≥ N`) with `f i = f j`. Take the recurring value at `N` (index `i ≥ N`) and
again past `i` (index `j > i`); both equal `a`, hence each other. The state repeats
do not stop — they happen unboundedly far out. -/
theorem exists_lt_eq_above_of_omega_recurrent {α : Type*} {f : ℕ → α}
    (hrec : ∃ a, ∀ N, ∃ n, N ≤ n ∧ f n = a) (N : ℕ) :
    ∃ i j, N ≤ i ∧ i < j ∧ f i = f j := by
  obtain ⟨a, ha⟩ := hrec
  obtain ⟨i, hiN, hi⟩ := ha N
  obtain ⟨j, hj, hjeq⟩ := ha (i + 1)
  exact ⟨i, j, hiN, Nat.lt_of_lt_of_le (Nat.lt_succ_self i) hj, by rw [hi, hjeq]⟩

/-- **The live solver repeats a state unboundedly often.** Specializing the generic
arbitrarily-late-repeat lemma to the real `safeSolver` trace: past every horizon `N`
there exist distinct steps `N ≤ i < j` whose game states coincide. The actual program,
against any legal stream, keeps returning to previously-seen states forever — the
state-graph footprint of its play has cycles arbitrarily far along. -/
theorem safeSolver_trace_state_repeats_unbounded
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) (N : ℕ) :
    ∃ i j, N ≤ i ∧ i < j ∧
      adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s GameState.init i
        = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s GameState.init j :=
  exists_lt_eq_above_of_omega_recurrent (safeSolver_trace_omega_recurrent_state h s hl) N

/-- **The recurring state's board never tops out.** The state `g` that the live solver
revisits infinitely often (from `safeSolver_trace_omega_recurrent_state`) has its board
confined to the fixed grid `range cols ×ˢ range rows`: `g = trace n` for some `n`, whose
board is grid-confined by `safeSolver_trace_board_subset_grid`. So the solver returns
forever to a bounded, alive board — the geometric heart of an M2-style closed cycle for
the actual program. -/
theorem safeSolver_trace_recurrent_state_board_subset_grid
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) :
    ∃ g, (∀ N, ∃ n, N ≤ n ∧
        adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s GameState.init n = g)
      ∧ g.board ⊆ Finset.range GameConfig.standard.cols ×ˢ Finset.range GameConfig.standard.rows := by
  obtain ⟨g, hg⟩ := safeSolver_trace_omega_recurrent_state h s hl
  refine ⟨g, hg, ?_⟩
  obtain ⟨n, _, hn⟩ := hg 0
  rw [← hn]
  exact safeSolver_trace_board_subset_grid h s hl n

/-- **The recurrence target is a fully-bounded, alive, safe state.** Packaging every
established per-step invariant onto the infinitely-recurring state `g` (via the
`omega_recurrent_state` ▸ per-step-bound recipe, instantiating at one witness depth):
`g` is in `safe`, never lost, has `count ≤ cols*rows`, and every column height `≤ 20`.
The live solver returns forever to a state that is simultaneously safe, alive, and
geometrically bounded — the complete invariant profile of an M2-style recurrent core. -/
theorem safeSolver_trace_recurrent_state_invariant
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) :
    ∃ g, (∀ N, ∃ n, N ≤ n ∧
        adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s GameState.init n = g)
      ∧ g ∈ safe GameConfig.standard
      ∧ ¬ g.lost GameConfig.standard
      ∧ g.board.count ≤ GameConfig.standard.cols * GameConfig.standard.rows
      ∧ ∀ j, Board.colHeight g.board j ≤ 20 := by
  obtain ⟨g, hg⟩ := safeSolver_trace_omega_recurrent_state h s hl
  obtain ⟨n, _, hn⟩ := hg 0
  refine ⟨g, hg, ?_, ?_, ?_, ?_⟩
  · rw [← hn]; exact safeSolver_trace_mem_safe h s hl n
  · rw [← hn]; exact safeSolver_trace_not_lost h s hl n
  · rw [← hn]; exact safeSolver_trace_count_le_capacity h s hl n
  · intro j; rw [← hn]; exact safeSolver_trace_colHeight_le_twenty h s hl n j

/-- **A genuine alive-bounded state repeat.** There are two distinct steps `i < j` whose
game states coincide, and that shared state is neither lost nor over-filled
(`count ≤ cols*rows`). Combining `safeSolver_trace_state_repeats_unbounded` (at horizon
`0`) with the per-step alive/count bounds at the earlier index: the live solver's play
literally re-enters an identical, bounded, non-topped-out configuration — the closed
state-loop seed for an M2 cycle of the real program. -/
theorem safeSolver_trace_bounded_state_repeat
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) :
    ∃ i j, i < j
      ∧ adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s GameState.init i
          = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s GameState.init j
      ∧ ¬ (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init i).lost GameConfig.standard
      ∧ (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init i).board.count ≤ GameConfig.standard.cols * GameConfig.standard.rows := by
  obtain ⟨i, j, _, hij, heq⟩ := safeSolver_trace_state_repeats_unbounded h s hl 0
  exact ⟨i, j, hij, heq, safeSolver_trace_not_lost h s hl i,
    safeSolver_trace_count_le_capacity h s hl i⟩

/-- **The real solver enters a genuine closed cycle under a periodic adversary.** When the
piece stream `s` is `P`-periodic (`P ≥ 1`, `s (k+P) = s k`), the live `safeSolver` play is
honestly periodic: there are a base index and a period `d ≥ 1` with `trace (base+k) =
trace (base+d+k)` for all `k`. Unlike the bare pigeonhole repeat (iter 57), the period here
is synchronised with the INPUT — pigeonhole runs over *multiples of `P`* (via the iter-55/56
finite-range + recurrence lemmas applied to `m ↦ trace (m*P)`), forcing `d` a multiple of
`P`, so the input genuinely repeats and `adversarialTrace_periodic_of_periodic_suffix` lifts
the lone state recurrence to full periodicity. The concrete `safeSolver` analog of
`closedAtlas_periodic_input_closedCycle`. -/
theorem safeSolver_periodic_input_closedCycle
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d ∧
      (∀ k, adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init (base + k)
        = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init (base + d + k)) := by
  have hshift : ∀ n k, s (n * P + k) = s k := by
    intro n k
    induction n with
    | zero => simp
    | succ m ih =>
      have hstep : (m + 1) * P + k = (m * P + k) + P := by ring
      rw [hstep, hper, ih]
  have hfin : (Set.range (fun m => adversarialTrace GameConfig.standard
      (safeSolver GameConfig.standard) s GameState.init (m * P))).Finite :=
    (safeSolver_trace_states_finite h s hl).subset (by rintro _ ⟨m, rfl⟩; exact ⟨m * P, rfl⟩)
  obtain ⟨a, b, _, hab, heq⟩ :=
    exists_lt_eq_above_of_omega_recurrent (exists_omega_recurrent_of_range_finite hfin) 0
  have harith : a * P + (b - a) * P = b * P := by
    have hxy' : a + (b - a) = b := by omega
    rw [← add_mul, hxy']
  refine ⟨a * P, (b - a) * P, ?_, ?_⟩
  · have hpos : (b - a) * P ≠ 0 := Nat.mul_ne_zero (by omega) (by omega)
    omega
  · intro k
    apply adversarialTrace_periodic_of_periodic_suffix
    · rw [harith]; exact heq
    · intro j; rw [harith, hshift a j, hshift b j]

/-- **Headline (M2 for the real solver): a safe state revisited forever under a periodic
adversary.** Combining `safeSolver_periodic_input_closedCycle` with
`adversarialTrace_iterate_of_periodic`: against any `P`-periodic stream the live `safeSolver`
returns *exactly* to the state `trace base` at every lattice point `base + n*d` (`d ≥ 1`), and
that state is in `safe`. A concrete closed cycle in the surviving subgraph for the ACTUAL
program — the M2 "closed cycle ⇒ infinite play" milestone realised by the safe-set solver. -/
theorem safeSolver_periodic_input_recurrent_state
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d ∧
      (∀ n, adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init (base + n * d)
        = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init base)
      ∧ adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init base ∈ safe GameConfig.standard := by
  obtain ⟨base, d, hd, hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  exact ⟨base, d, hd,
    fun n => adversarialTrace_iterate_of_periodic GameConfig.standard
      (safeSolver GameConfig.standard) s GameState.init hcyc n,
    safeSolver_trace_mem_safe h s hl base⟩

/-- **The real solver's limit cycle is a finite explicit set.** Under a `P`-periodic adversary
the live `safeSolver` play, after `base`, is confined to at most `d` distinct states
`{trace (base+j) | j < d}` (via `adversarialTrace_tail_mem_cycle_image`). The winning strategy
for the actual program uses bounded memory: a concrete `Finset` of ≤ `d` states the play never
leaves. -/
theorem safeSolver_periodic_input_tail_finite
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d ∧
      ∀ n, adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init (base + n)
        ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) s GameState.init (base + j)) := by
  obtain ⟨base, d, hd, hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  exact ⟨base, d, hd, fun n => adversarialTrace_tail_mem_cycle_image GameConfig.standard
    (safeSolver GameConfig.standard) s GameState.init hd hcyc n⟩

/-- **M3 (reachable closed cycle) for the real solver under a periodic adversary.** Strengthens
the periodic cycle to a reachability statement: the cycle base `trace base` is `ReachableAt`
depth `base` from `init` (the prefix `[0, base]` of the play IS the reach witness), is in
`safe`, and is genuinely `d`-periodic. So a real game starting from the empty board enters a
provably infinite loop — the M3 "reachable cycle" milestone, realised by the safe-set solver
against any periodic stream. -/
theorem safeSolver_periodic_input_reachable_closed_cycle
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ReachableAt GameConfig.standard base
          (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init base)
      ∧ adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
          GameState.init base ∈ safe GameConfig.standard
      ∧ (∀ k, adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init (base + k)
          = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init (base + d + k)) := by
  obtain ⟨base, d, hd, hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  exact ⟨base, d, hd, (safeSolver_adversarialTrace_reachableAt h s hl base).1,
    (safeSolver_adversarialTrace_reachableAt h s hl base).2, hcyc⟩

/-- **The limit cycle is an explicit finite set of `safe` states the play never leaves.** Under
a `P`-periodic adversary there is a base, a period `d ≥ 1`, and a concrete `Finset`
`C = {trace (base+j) | j < d}` such that (i) every tail state `trace (base+n) ∈ C`, and (ii)
every `g ∈ C` is in `safe`. The M2 closed-cycle proof artifact made fully concrete for the real
program: a bounded set of guaranteed-survivable states, closed under the periodic play. -/
theorem safeSolver_periodic_input_cycle_subset_safe
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ (∀ n, adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init (base + n)
          ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
              (safeSolver GameConfig.standard) s GameState.init (base + j)))
      ∧ (∀ g ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
              (safeSolver GameConfig.standard) s GameState.init (base + j)),
          g ∈ safe GameConfig.standard) := by
  obtain ⟨base, d, hd, hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  refine ⟨base, d, hd,
    fun n => adversarialTrace_tail_mem_cycle_image GameConfig.standard
      (safeSolver GameConfig.standard) s GameState.init hd hcyc n, ?_⟩
  intro g hg
  rw [Finset.mem_image] at hg
  obtain ⟨j, _, rfl⟩ := hg
  exact safeSolver_trace_mem_safe h s hl (base + j)

/-- **A concrete residue-indexed piece map.** Sends `0,…,6` to the seven distinct
tetrominoes (`O I S Z T L J`) and collapses everything `≥ 7` onto `J`; intended to be
called only on `n % 7`, so it is `7`-periodic by construction. Building block for a fixed,
deterministic 7-bag stream witnessing that the periodic-cycle theorems are not vacuous. -/
def pieceOfResidue : ℕ → Piece
  | 0 => Piece.O | 1 => Piece.I | 2 => Piece.S | 3 => Piece.Z
  | 4 => Piece.T | 5 => Piece.L | _ => Piece.J

/-- **The canonical 7-periodic legal stream.** `sevenBagCycle n := pieceOfResidue (n % 7)`
plays the fixed permutation `O I S Z T L J` and repeats it forever — exactly one full bag,
drawn in the same order each cycle. The concrete adversary that makes every
`safeSolver_periodic_input_*` theorem apply to real 7-bag play. -/
def sevenBagCycle (n : ℕ) : Piece := pieceOfResidue (n % 7)

/-- `sevenBagCycle` has period `7`: `sevenBagCycle (k + 7) = sevenBagCycle k`. -/
theorem sevenBagCycle_period (k : ℕ) : sevenBagCycle (k + 7) = sevenBagCycle k := by
  unfold sevenBagCycle; congr 1; omega

/-- **The fixed 7-bag stream is bag-legal.** Each of the seven pieces is drawn exactly once
per bag in the order `O I S Z T L J`, so the draw is always available in the current bag and
the bag refills to full every seventh step. This discharges the `LegalSequence` hypothesis of
every periodic-input theorem with a *concrete witness*, proving they are not vacuous: genuine
7-bag play (one fixed permutation, repeated) satisfies them. -/
theorem sevenBagCycle_legal : LegalSequence sevenBagCycle := by
  have hbagperiod : ∀ m, bagAt Bag.full sevenBagCycle (7 + m)
      = bagAt Bag.full sevenBagCycle m := by
    intro m
    rw [bagAt_add Bag.full sevenBagCycle 7 m]
    have h7 : bagAt Bag.full sevenBagCycle 7 = Bag.full := by decide
    have hsh : (fun k => sevenBagCycle (7 + k)) = sevenBagCycle := by
      funext k; rw [Nat.add_comm 7 k]; exact sevenBagCycle_period k
    rw [h7, hsh]
  have hbagres : ∀ r q, bagAt Bag.full sevenBagCycle (r + 7 * q)
      = bagAt Bag.full sevenBagCycle r := by
    intro r q
    induction q with
    | zero => simp
    | succ p ih =>
      rw [show r + 7 * (p + 1) = 7 + (r + 7 * p) from by ring, hbagperiod, ih]
  have hcheck : ∀ r, r < 7 → sevenBagCycle r ∈ bagAt Bag.full sevenBagCycle r := by decide
  intro n
  show sevenBagCycle n ∈ bagAt Bag.full sevenBagCycle n
  have e1 : bagAt Bag.full sevenBagCycle n = bagAt Bag.full sevenBagCycle (n % 7) := by
    conv_lhs => rw [← Nat.mod_add_div n 7]
    exact hbagres (n % 7) (n / 7)
  have e2 : sevenBagCycle n = sevenBagCycle (n % 7) := by
    unfold sevenBagCycle; congr 1; omega
  rw [e1, e2]
  exact hcheck (n % 7) (Nat.mod_lt n (by decide))

/-- **Headline non-vacuity: the real solver has a reachable closed cycle under genuine 7-bag
play.** Instantiating `safeSolver_periodic_input_reachable_closed_cycle` at the concrete legal
stream `sevenBagCycle`, *assuming only* `init ∈ safe`, yields a base index, a period `d ≥ 1`,
a `ReachableAt` proof that the cycle entry is reached from the empty board, that entry being
`safe`, and the exact replay identity `trace (base+k) = trace (base+d+k)`. This is the M3
"reachable provably-infinite loop" statement for a *specific deterministic game*, no longer
parameterised over an abstract periodic adversary. -/
theorem safeSolver_sevenBagCycle_reachable_closed_cycle
    (h : GameState.init ∈ safe GameConfig.standard) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ReachableAt GameConfig.standard base
          (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard)
            sevenBagCycle GameState.init base)
      ∧ adversarialTrace GameConfig.standard (safeSolver GameConfig.standard)
          sevenBagCycle GameState.init base ∈ safe GameConfig.standard
      ∧ (∀ k, adversarialTrace GameConfig.standard (safeSolver GameConfig.standard)
            sevenBagCycle GameState.init (base + k)
          = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard)
            sevenBagCycle GameState.init (base + d + k)) :=
  safeSolver_periodic_input_reachable_closed_cycle (P := 7) h sevenBagCycle
    sevenBagCycle_legal (by decide) sevenBagCycle_period

/-- **The reachable, safe, invariant limit set — packaged as one statement.** Under any
`P`-periodic legal adversary there exist a base index and period `d ≥ 1` with an explicit
`Finset` `C = {trace (base+j) | j < d}` such that simultaneously: (i) the cycle entry
`trace base` is `ReachableAt … base` from the empty board, (ii) `trace base ∈ C`, (iii) every
tail state `trace (base+n) ∈ C`, and (iv) every `g ∈ C` is `safe`. This fuses the M3
reachability result with the M2 closed-invariant-set result into a single artifact: a finite,
reachable, `safe`-internal set the play provably enters and never leaves. -/
theorem safeSolver_periodic_input_reachable_invariant_set
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ReachableAt GameConfig.standard base
          (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init base)
      ∧ adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init base
          ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
              (safeSolver GameConfig.standard) s GameState.init (base + j))
      ∧ (∀ n, adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init (base + n)
          ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
              (safeSolver GameConfig.standard) s GameState.init (base + j)))
      ∧ (∀ g ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
              (safeSolver GameConfig.standard) s GameState.init (base + j)),
          g ∈ safe GameConfig.standard) := by
  obtain ⟨base, d, hd, hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  refine ⟨base, d, hd, (safeSolver_adversarialTrace_reachableAt h s hl base).1, ?_,
    fun n => adversarialTrace_tail_mem_cycle_image GameConfig.standard
      (safeSolver GameConfig.standard) s GameState.init hd hcyc n, ?_⟩
  · rw [Finset.mem_image]
    exact ⟨0, Finset.mem_range.mpr (by omega), by simp only [Nat.add_zero]⟩
  · intro g hg
    rw [Finset.mem_image] at hg
    obtain ⟨j, _, rfl⟩ := hg
    exact safeSolver_trace_mem_safe h s hl (base + j)

/-- **Concrete M2+M3 for genuine 7-bag play.** The fully-instantiated reachable invariant set
for the deterministic stream `sevenBagCycle`: assuming only `init ∈ safe`, the real solver's
play enters an explicit finite `Finset` `C` of `safe` states that is reachable from the empty
board and never escaped. No abstract adversary — a specific repeated bag permutation. -/
theorem safeSolver_sevenBagCycle_reachable_invariant_set
    (h : GameState.init ∈ safe GameConfig.standard) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ReachableAt GameConfig.standard base
          (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard)
            sevenBagCycle GameState.init base)
      ∧ adversarialTrace GameConfig.standard (safeSolver GameConfig.standard)
            sevenBagCycle GameState.init base
          ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
              (safeSolver GameConfig.standard) sevenBagCycle GameState.init (base + j))
      ∧ (∀ n, adversarialTrace GameConfig.standard (safeSolver GameConfig.standard)
            sevenBagCycle GameState.init (base + n)
          ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
              (safeSolver GameConfig.standard) sevenBagCycle GameState.init (base + j)))
      ∧ (∀ g ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
              (safeSolver GameConfig.standard) sevenBagCycle GameState.init (base + j)),
          g ∈ safe GameConfig.standard) :=
  safeSolver_periodic_input_reachable_invariant_set (P := 7) h sevenBagCycle
    sevenBagCycle_legal (by decide) sevenBagCycle_period

/-- **The limit set is a small, fully-characterised phase of boards.** Under any periodic legal
adversary the play's invariant set materialises as a concrete `Finset C` with: `C` nonempty,
`C.card ≤ d` (at most `d` distinct states — a hard bound on the cycle's memory footprint), the
play's tail confined to `C`, and *every* `g ∈ C` simultaneously `safe`, not-lost (alive),
capacity-bounded (`count ≤ cols*rows = 200`), and height-bounded (every column `≤ 20`). This is
the "phase" of the atlas the program settles into: a finite catalogue of guaranteed-survivable,
physically-valid boards — the discoverable steady-state region of play. -/
theorem safeSolver_periodic_input_invariant_set_bounded
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, ∃ C : Finset GameState,
      1 ≤ d
      ∧ C = (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
              (safeSolver GameConfig.standard) s GameState.init (base + j))
      ∧ C.Nonempty
      ∧ C.card ≤ d
      ∧ (∀ n, adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init (base + n) ∈ C)
      ∧ (∀ g ∈ C, g ∈ safe GameConfig.standard
            ∧ ¬ g.lost GameConfig.standard
            ∧ g.board.count ≤ GameConfig.standard.cols * GameConfig.standard.rows
            ∧ ∀ c, Board.colHeight g.board c ≤ 20) := by
  obtain ⟨base, d, hd, hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  refine ⟨base, d, (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
      (safeSolver GameConfig.standard) s GameState.init (base + j)), hd, rfl, ?_, ?_,
    fun n => adversarialTrace_tail_mem_cycle_image GameConfig.standard
      (safeSolver GameConfig.standard) s GameState.init hd hcyc n, ?_⟩
  · have h0 : (0 : ℕ) ∈ Finset.range d := Finset.mem_range.mpr (by omega)
    exact ⟨_, Finset.mem_image_of_mem _ h0⟩
  · exact le_trans Finset.card_image_le (le_of_eq (Finset.card_range d))
  · intro g hg
    rw [Finset.mem_image] at hg
    obtain ⟨j, _, rfl⟩ := hg
    exact ⟨safeSolver_trace_mem_safe h s hl (base + j),
      safeSolver_trace_not_lost h s hl (base + j),
      safeSolver_trace_count_le_capacity h s hl (base + j),
      fun c => safeSolver_trace_colHeight_le_twenty h s hl (base + j) c⟩

/-- **Concrete bounded phase for genuine 7-bag play.** The fully-characterised limit set
instantiated at `sevenBagCycle`: assuming only `init ∈ safe`, the real solver settles into an
explicit `Finset` of at most `d` safe, alive, ≤200-cell, ≤20-tall boards it never leaves. -/
theorem safeSolver_sevenBagCycle_invariant_set_bounded
    (h : GameState.init ∈ safe GameConfig.standard) :
    ∃ base d : ℕ, ∃ C : Finset GameState,
      1 ≤ d
      ∧ C = (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
              (safeSolver GameConfig.standard) sevenBagCycle GameState.init (base + j))
      ∧ C.Nonempty
      ∧ C.card ≤ d
      ∧ (∀ n, adversarialTrace GameConfig.standard (safeSolver GameConfig.standard)
            sevenBagCycle GameState.init (base + n) ∈ C)
      ∧ (∀ g ∈ C, g ∈ safe GameConfig.standard
            ∧ ¬ g.lost GameConfig.standard
            ∧ g.board.count ≤ GameConfig.standard.cols * GameConfig.standard.rows
            ∧ ∀ c, Board.colHeight g.board c ≤ 20) :=
  safeSolver_periodic_input_invariant_set_bounded (P := 7) h sevenBagCycle
    sevenBagCycle_legal (by decide) sevenBagCycle_period

/-- **State recurrence forces depth-congruence mod 35.** If the live solver returns to the
*exact same state* at depths `i` and `j` (`trace i = trace j`), then `i ≡ j [MOD 35]`. The
phase odometer `safeSolver_trace_phase_eq_iff_mod_35` reads `(bag.card, count % 10)` from the
state alone, so identical states share a phase and hence congruent depths. Strengthens the
phase-equality law to genuine state-equality — the fundamental recurrence constraint on any
loop the program can form. -/
theorem safeSolver_trace_eq_imp_mod_35
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s) {i j : ℕ}
    (heq : adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init i
      = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init j) :
    i % 35 = j % 35 := by
  apply (safeSolver_trace_phase_eq_iff_mod_35 h s hl i j).mp
  rw [heq]

/-- **Every closed cycle of the program has period divisible by 35.** Under any periodic legal
adversary the limit cycle `trace (base+k) = trace (base+d+k)` has `35 ∣ d`. The construction
only guarantees `P ∣ d`; the joint bag/count phase clock (period `35 = 5·7`, by CRT) refines
this: the exact return `trace base = trace (base+d)` forces `base ≡ base+d [MOD 35]`, i.e.
`35 ∣ d`. Hence (with `1 ≤ d`) the minimal recurrence length is at least 35 — no shorter loop
in `safe` play is possible. -/
theorem safeSolver_periodic_input_period_dvd_35
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d ∧ 35 ∣ d
      ∧ (∀ k, adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init (base + k)
          = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init (base + d + k)) := by
  obtain ⟨base, d, hd, hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  refine ⟨base, d, hd, ?_, hcyc⟩
  have hret : adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init base
      = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init (base + d) := by
    have h0 := hcyc 0
    simp only [Nat.add_zero] at h0
    exact h0
  have hmod := safeSolver_trace_eq_imp_mod_35 h s hl hret
  omega

/-- **The 7-bag cycle has period divisible by 35.** Concrete instantiation: the deterministic
`sevenBagCycle` play (assuming `init ∈ safe`) loops with a period that is a multiple of 35 —
at least one full `5·7` phase revolution. -/
theorem safeSolver_sevenBagCycle_period_dvd_35
    (h : GameState.init ∈ safe GameConfig.standard) :
    ∃ base d : ℕ, 1 ≤ d ∧ 35 ∣ d
      ∧ (∀ k, adversarialTrace GameConfig.standard (safeSolver GameConfig.standard)
            sevenBagCycle GameState.init (base + k)
          = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard)
            sevenBagCycle GameState.init (base + d + k)) :=
  safeSolver_periodic_input_period_dvd_35 (P := 7) h sevenBagCycle sevenBagCycle_legal
    (by decide) sevenBagCycle_period

/-- **The limit set has between 35 and `d` states.** Sharp two-sided cardinality bound on the
program's invariant cycle `C = {trace (base+j) | j < d}` under any periodic legal adversary:
`35 ≤ C.card ≤ d`. The upper bound is `card_image_le`; the lower bound is the phase odometer —
the 35 states `trace (base+0), …, trace (base+34)` carry the 35 *distinct* joint phases
(`safeSolver_trace_eq_imp_mod_35`: equal states ⇒ congruent depths, impossible for distinct
residues `< 35`), so they are pairwise distinct and embed into `C` (since `35 ≤ d`). The
solver's steady-state phase therefore visits at least a full 35-state revolution — a hard floor
on the discoverable cycle's size. -/
theorem safeSolver_periodic_input_cycle_card_ge_35
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 35 ≤ d ∧ 35 ∣ d
      ∧ 35 ≤ ((Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) s GameState.init (base + j))).card
      ∧ ((Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) s GameState.init (base + j))).card ≤ d := by
  obtain ⟨base, d, hd, hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  have hret : adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init base
      = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init (base + d) := by
    have h0 := hcyc 0
    simp only [Nat.add_zero] at h0
    exact h0
  have h35d : 35 ∣ d := by
    have hmod := safeSolver_trace_eq_imp_mod_35 h s hl hret
    omega
  have hge : 35 ≤ d := Nat.le_of_dvd (by omega) h35d
  refine ⟨base, d, hge, h35d, ?_, ?_⟩
  · have hinj : Set.InjOn (fun j => adversarialTrace GameConfig.standard
        (safeSolver GameConfig.standard) s GameState.init (base + j))
        (↑(Finset.range 35) : Set ℕ) := by
      intro a ha b hb hfab
      simp only [Finset.coe_range, Set.mem_Iio] at ha hb
      have hmod := safeSolver_trace_eq_imp_mod_35 h s hl hfab
      omega
    have hcard35 : ((Finset.range 35).image (fun j => adversarialTrace GameConfig.standard
        (safeSolver GameConfig.standard) s GameState.init (base + j))).card = 35 := by
      rw [Finset.card_image_of_injOn hinj, Finset.card_range]
    calc 35 = ((Finset.range 35).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) s GameState.init (base + j))).card := hcard35.symm
      _ ≤ ((Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) s GameState.init (base + j))).card :=
          Finset.card_le_card (Finset.image_subset_image
            (by intro x hx; rw [Finset.mem_range] at hx ⊢; omega))
  · exact le_trans Finset.card_image_le (le_of_eq (Finset.card_range d))

/-- **Concrete 35-to-`d` cycle-size bound for 7-bag play.** The limit set of the deterministic
`sevenBagCycle` play (assuming `init ∈ safe`) contains between 35 and `d` distinct states. -/
theorem safeSolver_sevenBagCycle_cycle_card_ge_35
    (h : GameState.init ∈ safe GameConfig.standard) :
    ∃ base d : ℕ, 35 ≤ d ∧ 35 ∣ d
      ∧ 35 ≤ ((Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) sevenBagCycle GameState.init (base + j))).card
      ∧ ((Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) sevenBagCycle GameState.init (base + j))).card ≤ d :=
  safeSolver_periodic_input_cycle_card_ge_35 (P := 7) h sevenBagCycle sevenBagCycle_legal
    (by decide) sevenBagCycle_period

/-- **The limit cycle passes through a "home" reset state.** Under any periodic legal adversary,
the limit set `C = {trace(base+j) | j < d}` of `safeSolver` play (assuming `init ∈ safe`) contains
a state whose bag is full (`Bag.full`) and whose board cell-count is `≡ 0 (mod 10)` — the joint
origin phase `(7,0)`. Reason: `35 ∣ d` (so `d ≥ 35`), hence the window `[base, base+d)` straddles
a depth divisible by 35, and `jointPhase_eq_origin_iff` says exactly those depths carry the home
phase. This realizes one concrete value of the period-35 phase clock inside the cycle. -/
theorem safeSolver_periodic_input_cycle_contains_home
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ∃ g ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) s GameState.init (base + j)),
          g.bag = Bag.full ∧ g.board.count % 10 = 0 := by
  obtain ⟨base, d, hd, hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  have hret : adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init base
      = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init (base + d) := by
    have h0 := hcyc 0
    simp only [Nat.add_zero] at h0
    exact h0
  have h35d : 35 ∣ d := by
    have hmod := safeSolver_trace_eq_imp_mod_35 h s hl hret
    omega
  have hged : 35 ≤ d := Nat.le_of_dvd (by omega) h35d
  obtain ⟨j, hjlt, hjd35⟩ : ∃ j, j < 35 ∧ 35 ∣ (base + j) :=
    ⟨(35 - base % 35) % 35, by omega, by omega⟩
  have hjltd : j < d := by omega
  have hph := safeSolver_trace_jointPhase h s hl (base + j)
  rw [(jointPhase_eq_origin_iff (base + j)).mpr hjd35, Prod.mk.injEq] at hph
  refine ⟨base, d, hd, adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
      GameState.init (base + j), ?_, (Bag.card_eq_seven_iff_eq_full _).mp hph.1, hph.2⟩
  rw [Finset.mem_image]
  exact ⟨j, Finset.mem_range.mpr hjltd, rfl⟩

/-- **Concrete home-state realization for 7-bag play.** The limit cycle of the deterministic
`sevenBagCycle` play (assuming `init ∈ safe`) contains a full-bag, count-`≡0 (mod 10)` state. -/
theorem safeSolver_sevenBagCycle_cycle_contains_home
    (h : GameState.init ∈ safe GameConfig.standard) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ∃ g ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) sevenBagCycle GameState.init (base + j)),
          g.bag = Bag.full ∧ g.board.count % 10 = 0 :=
  safeSolver_periodic_input_cycle_contains_home (P := 7) h sevenBagCycle sevenBagCycle_legal
    (by decide) sevenBagCycle_period

/-- **The limit cycle realizes every joint phase (phase surjectivity).** Under any periodic legal
adversary, for *every* target depth `r` there is a cycle state `g ∈ C = {trace(base+j) | j < d}`
whose observed phase `(g.bag.card, g.board.count % 10)` equals `jointPhase r`. Since `jointPhase`
has period 35 and the cycle window has length `d ≥ 35` (from `35 ∣ d`), a single period fits inside
`[base, base+d)`, covering all 35 residue classes. This strengthens the `≥ 35 distinct states`
bound (`safeSolver_periodic_input_cycle_card_ge_35`) to an explicit phase-labelling: the cycle hits
each of the 35 joint-phase classes. The home state `(7,0)` is the `35 ∣ r` instance. -/
theorem safeSolver_periodic_input_cycle_realizes_all_phases
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ∀ r : ℕ, ∃ g ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) s GameState.init (base + j)),
          (g.bag.card, g.board.count % 10) = jointPhase r := by
  obtain ⟨base, d, hd, hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  have hret : adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init base
      = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
        GameState.init (base + d) := by
    have h0 := hcyc 0
    simp only [Nat.add_zero] at h0
    exact h0
  have h35d : 35 ∣ d := by
    have hmod := safeSolver_trace_eq_imp_mod_35 h s hl hret
    omega
  have hged : 35 ≤ d := Nat.le_of_dvd (by omega) h35d
  refine ⟨base, d, hd, ?_⟩
  intro r
  obtain ⟨j, hjlt, hjmod⟩ : ∃ j, j < 35 ∧ (base + j) % 35 = r % 35 :=
    ⟨(r + 35 - base % 35) % 35, by omega, by omega⟩
  have hjltd : j < d := by omega
  refine ⟨adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
      GameState.init (base + j), ?_, ?_⟩
  · rw [Finset.mem_image]
    exact ⟨j, Finset.mem_range.mpr hjltd, rfl⟩
  · rw [safeSolver_trace_jointPhase h s hl (base + j)]
    exact (jointPhase_eq_iff_mod_35 (base + j) r).mpr hjmod

/-- **Concrete phase surjectivity for 7-bag play.** The deterministic `sevenBagCycle` limit cycle
(assuming `init ∈ safe`) realizes every joint phase `jointPhase r`. -/
theorem safeSolver_sevenBagCycle_cycle_realizes_all_phases
    (h : GameState.init ∈ safe GameConfig.standard) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ∀ r : ℕ, ∃ g ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) sevenBagCycle GameState.init (base + j)),
          (g.bag.card, g.board.count % 10) = jointPhase r :=
  safeSolver_periodic_input_cycle_realizes_all_phases (P := 7) h sevenBagCycle sevenBagCycle_legal
    (by decide) sevenBagCycle_period

/-- **The limit cycle's phase image is *exactly* the 7×5 grid.** Projecting the cycle states
`C = {trace(base+j) | j < d}` through the phase map `g ↦ (g.bag.card, g.board.count % 10)` yields
precisely the 35-cell grid `{1,…,7} × {0,4,8,2,6}` — no phase missing, none outside it. The `⊆`
inclusion is `safeSolver_trace_phase_mem_grid` (phase confinement); the `⊇` inclusion is the
cycle phase-surjectivity `safeSolver_periodic_input_cycle_realizes_all_phases` composed with the
grid surjectivity `jointPhase_image_range_35_eq_grid`. This is the finite-cycle analogue of the
whole-orbit `reachablePhases_eq_grid`: the *bounded* limit set already exhausts phase-space. -/
theorem safeSolver_periodic_input_cycle_phase_image_eq_grid
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ (((Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) s GameState.init (base + j))).image
          (fun g => (g.bag.card, g.board.count % 10)))
        = (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ) := by
  obtain ⟨base, d, hd, hreal⟩ :=
    safeSolver_periodic_input_cycle_realizes_all_phases h s hl hP hper
  refine ⟨base, d, hd, ?_⟩
  ext φ
  constructor
  · intro hmem
    rw [Finset.mem_image] at hmem
    obtain ⟨g, hgC, rfl⟩ := hmem
    rw [Finset.mem_image] at hgC
    obtain ⟨j, _, rfl⟩ := hgC
    exact safeSolver_trace_phase_mem_grid h s hl (base + j)
  · intro hφ
    rw [← jointPhase_image_range_35_eq_grid, Finset.mem_image] at hφ
    obtain ⟨m, _, hmp⟩ := hφ
    obtain ⟨g, hgC, hgph⟩ := hreal m
    rw [Finset.mem_image]
    exact ⟨g, hgC, by rw [hgph, hmp]⟩

/-- **The limit cycle realizes exactly 35 phases.** Cardinality corollary of
`safeSolver_periodic_input_cycle_phase_image_eq_grid`: the cycle's phase image has card `35`. -/
theorem safeSolver_periodic_input_cycle_phase_image_card
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ (((Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) s GameState.init (base + j))).image
          (fun g => (g.bag.card, g.board.count % 10))).card = 35 := by
  obtain ⟨base, d, hd, heq⟩ :=
    safeSolver_periodic_input_cycle_phase_image_eq_grid h s hl hP hper
  exact ⟨base, d, hd, by rw [heq, phaseGrid_card]⟩

/-- **Concrete phase-image = grid for 7-bag play.** The deterministic `sevenBagCycle` limit cycle
(assuming `init ∈ safe`) has phase image exactly the 35-cell grid. -/
theorem safeSolver_sevenBagCycle_cycle_phase_image_eq_grid
    (h : GameState.init ∈ safe GameConfig.standard) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ (((Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) sevenBagCycle GameState.init (base + j))).image
          (fun g => (g.bag.card, g.board.count % 10)))
        = (Finset.Icc 1 7) ×ˢ ({0, 4, 8, 2, 6} : Finset ℕ) :=
  safeSolver_periodic_input_cycle_phase_image_eq_grid (P := 7) h sevenBagCycle sevenBagCycle_legal
    (by decide) sevenBagCycle_period

/-- **Every state of the limit cycle is itself an infinite-survival start.** Under any periodic
legal adversary, each state `g` in the finite cycle `C = {trace(base+j) | j < d}` satisfies
`SolvesTetrisFrom standard g safeSolver` — handed `g` cold, the program survives forever against
*every* bag-legal continuation, not merely the periodic one that generated the cycle. This is the
cycle-localized payoff of stationarity (`safeSolver_trace_solvesTetrisFrom_state`): the limit cycle
is a finite set of mutually-winning positions, a bounded-memory winning certificate. -/
theorem safeSolver_periodic_input_cycle_states_solvesTetrisFrom
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ∀ g ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) s GameState.init (base + j)),
          SolvesTetrisFrom GameConfig.standard g (safeSolver GameConfig.standard) := by
  obtain ⟨base, d, hd, _hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  refine ⟨base, d, hd, ?_⟩
  intro g hg
  rw [Finset.mem_image] at hg
  obtain ⟨j, _, rfl⟩ := hg
  exact safeSolver_trace_solvesTetrisFrom_state h s hl (base + j)

/-- **Finite winning certificate for the limit cycle.** Strengthens the bounded invariant set
(`safeSolver_periodic_input_invariant_set_bounded`) with the solve guarantee: every cycle state `g`
is simultaneously (i) `SolvesTetrisFrom`-winning, (ii) in `safe`, and (iii) not lost. A complete,
self-contained certificate that the limit cycle is a finite set of safe, alive, infinitely-solvable
positions. -/
theorem safeSolver_periodic_input_cycle_winning_certificate
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ∀ g ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) s GameState.init (base + j)),
          SolvesTetrisFrom GameConfig.standard g (safeSolver GameConfig.standard)
            ∧ g ∈ safe GameConfig.standard
            ∧ ¬ g.lost GameConfig.standard := by
  obtain ⟨base, d, hd, _hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  refine ⟨base, d, hd, ?_⟩
  intro g hg
  rw [Finset.mem_image] at hg
  obtain ⟨j, _, rfl⟩ := hg
  exact ⟨safeSolver_trace_solvesTetrisFrom_state h s hl (base + j),
    safeSolver_trace_mem_safe h s hl (base + j),
    safeSolver_trace_not_lost h s hl (base + j)⟩

/-- **Concrete winning cycle certificate for 7-bag play.** Every state of the deterministic
`sevenBagCycle` limit cycle (assuming `init ∈ safe`) is a `SolvesTetrisFrom`-winning start. -/
theorem safeSolver_sevenBagCycle_cycle_states_solvesTetrisFrom
    (h : GameState.init ∈ safe GameConfig.standard) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ∀ g ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
            (safeSolver GameConfig.standard) sevenBagCycle GameState.init (base + j)),
          SolvesTetrisFrom GameConfig.standard g (safeSolver GameConfig.standard) :=
  safeSolver_periodic_input_cycle_states_solvesTetrisFrom (P := 7) h sevenBagCycle
    sevenBagCycle_legal (by decide) sevenBagCycle_period

/-- **The successor of every window state stays in the cycle.** Under any periodic legal adversary,
the trajectory successor `trace(base+(n+1))` of each cycle index lies in `C = {trace(base+j) | j<d}`.
Immediate from `adversarialTrace_tail_mem_cycle_image` at `n+1`: the tail never leaves `C`. -/
theorem safeSolver_periodic_input_cycle_successor_mem
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ∀ n, adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
            GameState.init (base + (n + 1))
          ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
              (safeSolver GameConfig.standard) s GameState.init (base + j)) := by
  obtain ⟨base, d, hd, hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  exact ⟨base, d, hd, fun n => adversarialTrace_tail_mem_cycle_image GameConfig.standard
    (safeSolver GameConfig.standard) s GameState.init hd hcyc (n + 1)⟩

/-- **The cycle is forward-closed under the program's own transition.** Under any periodic legal
adversary, applying the `safeSolver`'s chosen move at any window state `trace(base+n)` for the
piece `s(base+n)` lands the successor back in `C = {trace(base+j) | j<d}`. Since
`adversarialTrace_succ` makes the program step *definitionally* the trajectory successor, this is
the operational reading of `safeSolver_periodic_input_cycle_successor_mem`: `C` is a closed
transition graph — a finite automaton the program walks forever without escaping. -/
theorem safeSolver_periodic_input_cycle_step_closed
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ∀ n, adversarialStep GameConfig.standard
            (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
              GameState.init (base + n))
            (s (base + n))
            (safeSolver GameConfig.standard
              (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
                GameState.init (base + n))
              (s (base + n)))
          ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
              (safeSolver GameConfig.standard) s GameState.init (base + j)) := by
  obtain ⟨base, d, hd, hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  refine ⟨base, d, hd, ?_⟩
  intro n
  rw [← adversarialTrace_succ GameConfig.standard (safeSolver GameConfig.standard) s
      GameState.init (base + n)]
  exact adversarialTrace_tail_mem_cycle_image GameConfig.standard (safeSolver GameConfig.standard)
    s GameState.init hd hcyc (n + 1)

/-- **Concrete forward-closed cycle for 7-bag play.** The deterministic `sevenBagCycle` limit cycle
(assuming `init ∈ safe`) is closed under the program's own one-step transition. -/
theorem safeSolver_sevenBagCycle_cycle_step_closed
    (h : GameState.init ∈ safe GameConfig.standard) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ∀ n, adversarialStep GameConfig.standard
            (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) sevenBagCycle
              GameState.init (base + n))
            (sevenBagCycle (base + n))
            (safeSolver GameConfig.standard
              (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) sevenBagCycle
                GameState.init (base + n))
              (sevenBagCycle (base + n)))
          ∈ (Finset.range d).image (fun j => adversarialTrace GameConfig.standard
              (safeSolver GameConfig.standard) sevenBagCycle GameState.init (base + j)) :=
  safeSolver_periodic_input_cycle_step_closed (P := 7) h sevenBagCycle sevenBagCycle_legal
    (by decide) sevenBagCycle_period

/-- **The limit cycle is a cyclic `d`-state automaton — the board is a function of the index mod
`d`.** Under any periodic legal adversary, every tail state collapses onto one of the `d` window
states by its index residue: `trace(base+n) = trace(base + n % d)`. Packaged form of
`adversarialTrace_mod_period` with the cycle's own `base`/`d`. The eventual play is a walk over a
finite `d`-element state set addressed by `n mod d`. -/
theorem safeSolver_periodic_input_cycle_index_reduction
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ∀ n, adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
              GameState.init (base + n)
          = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
              GameState.init (base + n % d) := by
  obtain ⟨base, d, hd, hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  exact ⟨base, d, hd, fun n => adversarialTrace_mod_period GameConfig.standard
    (safeSolver GameConfig.standard) s GameState.init hcyc n⟩

/-- **The cycle's transition is exactly the successor map `j ↦ (j+1) mod d`.** Under any periodic
legal adversary, applying the `safeSolver`'s move at window index `j` advances to the window state
at index `(j+1) mod d`: the program's one-step transition on the limit cycle is the canonical
`d`-cycle rotation. Proof: `adversarialTrace_succ` makes the program step the trajectory successor
`trace(base+(j+1))` (definitionally), and `adversarialTrace_mod_period` folds that index back into
`[0,d)`. Together with `safeSolver_periodic_input_cycle_index_reduction` this exhibits the limit
cycle as a genuine finite cyclic automaton on `d` states. -/
theorem safeSolver_periodic_input_cycle_automaton_step
    (h : GameState.init ∈ safe GameConfig.standard)
    (s : ℕ → Piece) (hl : LegalSequence s)
    {P : ℕ} (hP : 1 ≤ P) (hper : ∀ k, s (k + P) = s k) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ∀ j, adversarialStep GameConfig.standard
            (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
              GameState.init (base + j))
            (s (base + j))
            (safeSolver GameConfig.standard
              (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
                GameState.init (base + j))
              (s (base + j)))
          = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) s
              GameState.init (base + (j + 1) % d) := by
  obtain ⟨base, d, hd, hcyc⟩ := safeSolver_periodic_input_closedCycle h s hl hP hper
  refine ⟨base, d, hd, ?_⟩
  intro j
  rw [← adversarialTrace_succ GameConfig.standard (safeSolver GameConfig.standard) s
      GameState.init (base + j)]
  exact adversarialTrace_mod_period GameConfig.standard (safeSolver GameConfig.standard) s
    GameState.init hcyc (j + 1)

/-- **Concrete cyclic automaton for 7-bag play.** The deterministic `sevenBagCycle` limit cycle
(assuming `init ∈ safe`) transitions by `j ↦ (j+1) mod d`. -/
theorem safeSolver_sevenBagCycle_cycle_automaton_step
    (h : GameState.init ∈ safe GameConfig.standard) :
    ∃ base d : ℕ, 1 ≤ d
      ∧ ∀ j, adversarialStep GameConfig.standard
            (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) sevenBagCycle
              GameState.init (base + j))
            (sevenBagCycle (base + j))
            (safeSolver GameConfig.standard
              (adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) sevenBagCycle
                GameState.init (base + j))
              (sevenBagCycle (base + j)))
          = adversarialTrace GameConfig.standard (safeSolver GameConfig.standard) sevenBagCycle
              GameState.init (base + (j + 1) % d) :=
  safeSolver_periodic_input_cycle_automaton_step (P := 7) h sevenBagCycle sevenBagCycle_legal
    (by decide) sevenBagCycle_period

/-- **Holes are bounded across any closed atlas.** Every state of a closed `standard` table has at
most `200` buried cells: `holes standard g.board ≤ 200`. Chains the board envelope
`holes ≤ ∑ colHeight` with the closed-set column bound `colHeight ≤ 20` over all `10` columns. So
holes — like `count` and `colHeight` — live in a finite range on the atlas, a prerequisite for any
holes-based Lyapunov sublevel argument. -/
theorem closedSet_holes_le_two_hundred {T : Finset GameState}
    (hT : IsClosedSet GameConfig.standard T) {g : GameState} (hg : g ∈ T) :
    holes GameConfig.standard g.board ≤ 200 := by
  have hcols : GameConfig.standard.cols = 10 := rfl
  calc holes GameConfig.standard g.board
      ≤ ∑ j ∈ Finset.range GameConfig.standard.cols, Board.colHeight g.board j :=
        holes_le_sum_colHeight GameConfig.standard g.board
    _ ≤ ∑ j ∈ Finset.range GameConfig.standard.cols, 20 :=
        Finset.sum_le_sum (fun j _ => closedSet_colHeight_le_twenty hT hg j)
    _ = 200 := by rw [hcols]; simp

/-- **Per-column filled-cell count = fiber cardinality.** The buried/visible rows of column `j`,
`(b.colRows j).card`, equal the number of board cells lying in that column,
`(b.filter (·.1 = j)).card`. The second-projection `(·.2)` is injective on a fixed-column fiber
(two cells sharing column `j` and the same row are equal), so the image taken by `colRows` loses no
elements. Bridges the row-set view (`colRows`) and the cell-count view (`filter`). -/
theorem colRows_card_eq_filter_card (b : Board) (j : ℕ) :
    (b.colRows j).card = (b.filter (·.1 = j)).card := by
  unfold Board.colRows
  apply Finset.card_image_of_injOn
  intro x hx y hy hxy
  simp only [Finset.mem_coe, Finset.mem_filter] at hx hy
  ext
  · rw [hx.2, hy.2]
  · exact hxy

/-- **Column fibers partition the board: `∑ⱼ (colRows j).card = count`.** For a column-well-formed
board (every cell in a column `< cfg.cols`), summing the per-column filled-cell counts over the
playfield recovers the total cell count `b.count`. Immediate from `card_eq_sum_card_fiberwise` along
the column projection `(·.1)`, rewritten through `colRows_card_eq_filter_card`. -/
theorem sum_colRows_card_eq_count {cfg : GameConfig} {b : Board}
    (hWF : Board.ColWF cfg.cols b) :
    (∑ j ∈ Finset.range cfg.cols, (b.colRows j).card) = b.count := by
  unfold Board.count
  rw [Finset.card_eq_sum_card_fiberwise
        (f := fun p => p.1) (s := b) (t := Finset.range cfg.cols)
        (fun x hx => Finset.mem_range.mpr (hWF x hx))]
  exact Finset.sum_congr rfl (fun j _ => colRows_card_eq_filter_card b j)

/-- **Global volume conservation law: `count + holes = ∑ colHeight`.** For a column-well-formed
board, the total filled cells plus the total buried holes equals the total stack volume (sum of
column heights). The capstone of the holes decomposition: every cell under a column top is either
filled (counted in `count`) or buried (counted in `holes`), with none double-counted. Combines
`sum_colRows_card_add_holes` (iter-73) with the fiber partition `sum_colRows_card_eq_count`. With
`closedSet_holes_le_two_hundred` and the count bounds this pins all three atlas quantities together. -/
theorem count_add_holes_eq_sum_colHeight {cfg : GameConfig} {b : Board}
    (hWF : Board.ColWF cfg.cols b) :
    b.count + holes cfg b = ∑ j ∈ Finset.range cfg.cols, b.colHeight j := by
  have h := sum_colRows_card_add_holes cfg b
  rw [sum_colRows_card_eq_count hWF] at h
  exact h

/-- **Holes as the height/count deficit: `holes = (∑ colHeight) − count`.** Solving the conservation
law for `holes` (Nat subtraction is exact here since `count ≤ ∑ colHeight`). The buried-cell total is
precisely the gap between the stack volume bounded by the column tops and the cells actually filled. -/
theorem holes_eq_sum_colHeight_sub_count {cfg : GameConfig} {b : Board}
    (hWF : Board.ColWF cfg.cols b) :
    holes cfg b = (∑ j ∈ Finset.range cfg.cols, b.colHeight j) - b.count := by
  have h := count_add_holes_eq_sum_colHeight hWF
  omega

/-- **Filled cells never exceed the stack volume: `count ≤ ∑ colHeight`.** A direct corollary of the
conservation law — the column-height envelope dominates the cell count, with the slack being exactly
the holes. -/
theorem count_le_sum_colHeight {cfg : GameConfig} {b : Board}
    (hWF : Board.ColWF cfg.cols b) :
    b.count ≤ ∑ j ∈ Finset.range cfg.cols, b.colHeight j := by
  have h := count_add_holes_eq_sum_colHeight hWF
  omega

/-- **Conservation law specialised to the atlas.** Every state reachable at any depth satisfies
`count + holes = ∑ colHeight` with no well-formedness side condition — reachable boards are
column-well-formed (`ReachableAt.wf`), so the law holds automatically across the entire reachable
state graph. The atlas-level form of `count_add_holes_eq_sum_colHeight`. -/
theorem reachableAt_count_add_holes_eq_sum_colHeight {cfg : GameConfig} {n : ℕ} {g : GameState}
    (h : ReachableAt cfg n g) :
    g.board.count + holes cfg g.board
      = ∑ j ∈ Finset.range cfg.cols, g.board.colHeight j :=
  count_add_holes_eq_sum_colHeight (Board.colWF_iff_wf.mpr h.wf)

/-- **Row membership in a column ⇔ cell membership.** `r` is a filled row of column `j` exactly when
the cell `(j, r)` is on the board. The clean characterisation of `colRows` that bypasses the
image/filter encoding. -/
theorem mem_colRows_iff (b : Board) (j r : ℕ) :
    r ∈ b.colRows j ↔ (j, r) ∈ b := by
  unfold Board.colRows
  constructor
  · intro h
    rcases Finset.mem_image.mp h with ⟨⟨c, r'⟩, hmem', heq⟩
    rcases Finset.mem_filter.mp hmem' with ⟨hcell, hcol⟩
    simp only at hcol heq
    subst hcol; subst heq
    exact hcell
  · intro h
    exact Finset.mem_image.mpr ⟨(j, r), Finset.mem_filter.mpr ⟨h, rfl⟩, rfl⟩

/-! ### Definitive atlas existence: the M4 reduction

The headline existence question — does a tetris atlas (a finite lookup table,
closed under adversarial play, containing the start state `init`) *exist*? — is
proved here to be **exactly equivalent** to `init ∈ safe`, i.e. to infinite
survivability. Three results:

* `isClosedSet_atlas` — the atlas constructor always outputs a closed table;
* `init_mem_safe_iff_init_mem_atlas_inFieldStates` — the *computational* M4
  reduction: `init` is safe iff it lies in the atlas of the canonical finite
  universe `inFieldStates`, a single decidable test;
* `exists_closed_atlas_iff_init_mem_safe` — the *abstract* existence Iff:
  a closed atlas containing `init` exists **iff** `init` is safe.

Together they collapse the open M4 existence problem to one finite atlas
computation, and certify that "the atlas exists" and "tetris is infinitely
survivable" are one and the same proposition. -/

/-- **The atlas constructor always produces a closed lookup table.** Every state
in `atlas cfg U` is non-lost and, for each piece the adversary may draw, has a
valid placement landing back inside `atlas cfg U`. This is the defining
correctness property of the atlas-generating program: its output is a fixed
point of `closeStep`, hence closed under play. -/
theorem isClosedSet_atlas (cfg : GameConfig) (U : Finset GameState) :
    IsClosedSet cfg (atlas cfg U) := by
  intro g hg
  have hclose : g ∈ closeStep cfg (atlas cfg U) := by
    rw [closeStep_atlas]; exact hg
  rw [mem_closeStep_iff] at hclose
  exact ⟨hclose.2.1, hclose.2.2⟩

/-- **Computational M4 reduction.** `init` is `safe` (infinite survival is
possible) **iff** `init` lies in the atlas of the canonical universe
`inFieldStates cfg`. The forward direction builds the closed witness
`{g ∈ inFieldStates | g ∈ safe ∧ Reachable g}` (closed because a safe state
has, per piece, a safe successor, which is again reachable and hence in-field);
the reverse is atlas soundness. Since `atlas` and Finset membership are
decidable, this reduces the open existence question to ONE finite computation:
build `atlas cfg (inFieldStates cfg)` and test `init ∈ ·`. -/
theorem init_mem_safe_iff_init_mem_atlas_inFieldStates (cfg : GameConfig) :
    GameState.init ∈ safe cfg ↔
      GameState.init ∈ atlas cfg (inFieldStates cfg) := by
  constructor
  · intro hinit
    classical
    rw [mem_atlas_iff_exists_closed_witness]
    refine ⟨(inFieldStates cfg).filter (fun g => g ∈ safe cfg ∧ Reachable cfg g),
            Finset.filter_subset _ _,
            Finset.mem_filter.mpr ⟨init_mem_inFieldStates cfg, hinit, Reachable.init⟩, ?_⟩
    intro g' hg'
    obtain ⟨_, hsafe, hreach⟩ := Finset.mem_filter.mp hg'
    refine ⟨safe_not_lost hsafe, fun p hp => ?_⟩
    obtain ⟨_, hforall⟩ := mem_safe_iff_allValidFor.mp hsafe
    obtain ⟨pl, hpl, hstepSafe⟩ := hforall p hp
    obtain ⟨hpiece, hvalid⟩ := (Placement.mem_allValidFor cfg p pl).mp hpl
    have hvalid' : ({pl with piece := p} : Placement).Valid cfg := by
      have heq : ({pl with piece := p} : Placement) = pl := by rw [← hpiece]
      rw [heq]; exact hvalid
    have hreach' : Reachable cfg (adversarialStep cfg g' p pl) :=
      hreach.adversarialStep (safe_not_lost hsafe) hp hvalid'
    exact ⟨pl, hpl, Finset.mem_filter.mpr
      ⟨reachable_safe_mem_inFieldStates hreach' hstepSafe, hstepSafe, hreach'⟩⟩
  · intro hinit
    exact atlas_subset_safe cfg (inFieldStates cfg) (Finset.mem_coe.mpr hinit)

/-- **Definitive atlas existence (abstract form).** A finite closed atlas
containing `init` EXISTS **iff** `init` is `safe`. Forward: any closed table is
its own atlas (`atlas_eq_of_closed`), and the atlas is sound (`atlas_subset_safe`).
Reverse: when `init` is safe, the canonical atlas `atlas cfg (inFieldStates cfg)`
is such a table (closed by `isClosedSet_atlas`, contains `init` by the
computational reduction). This certifies that "the tetris atlas exists" and
"tetris is infinitely survivable from `init`" are the very same proposition. -/
theorem exists_closed_atlas_iff_init_mem_safe (cfg : GameConfig) :
    (∃ T : Finset GameState, GameState.init ∈ T ∧ IsClosedSet cfg T) ↔
      GameState.init ∈ safe cfg := by
  constructor
  · rintro ⟨T, hinit, hclosed⟩
    apply atlas_subset_safe cfg T
    rw [Finset.mem_coe, atlas_eq_of_closed cfg T hclosed]
    exact hinit
  · intro hinit
    exact ⟨atlas cfg (inFieldStates cfg),
           (init_mem_safe_iff_init_mem_atlas_inFieldStates cfg).mp hinit,
           isClosedSet_atlas cfg (inFieldStates cfg)⟩

/-! ### Lifting atlas existence to the headline `TetrisSolvable` proposition

The previous block reduced atlas existence to `init ∈ safe`. The
`SafeSet`-level iffs (`tetrisSolvableValidFor_iff_init_safe`,
`tetrisSolvableValid_iff_init_safe`) further identify `init ∈ safe` with the
existence of a surviving *solver program*. Composing, we obtain the project's
central proposition — Tetris is (valid-)solvable — phrased directly in terms of
the atlas: a finite closed lookup table containing `init` exists **iff** a solver
that survives forever exists, and both reduce to one atlas-membership test. -/

/-- **Closed atlas exists ⟺ a surviving solver exists (parametric).** For any
board wide enough to hold a piece (`4 ≤ cfg.cols`), the existence of a finite
closed atlas containing `init` is equivalent to `TetrisSolvableValidFor cfg`
(there is a valid solver surviving every legal 7-bag sequence). Pure composition
of `exists_closed_atlas_iff_init_mem_safe` with the headline `SafeSet` iff. -/
theorem exists_closed_atlas_iff_tetrisSolvableValidFor {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) :
    (∃ T : Finset GameState, GameState.init ∈ T ∧ IsClosedSet cfg T) ↔
      TetrisSolvableValidFor cfg := by
  rw [exists_closed_atlas_iff_init_mem_safe cfg,
      tetrisSolvableValidFor_iff_init_safe hcols]

/-- **Closed atlas exists ⟺ canonical Tetris is solvable.** The standard-config
(`10×20`) specialization: a finite closed atlas containing `init` exists **iff**
`TetrisSolvableValid` — the project's central conjecture that a valid solver
survives forever on the canonical board. This is the Atlas-as-proof-artifact
statement at the top level: building the atlas IS solving Tetris. -/
theorem exists_closed_atlas_iff_tetrisSolvableValid :
    (∃ T : Finset GameState,
        GameState.init ∈ T ∧ IsClosedSet GameConfig.standard T) ↔
      TetrisSolvableValid := by
  rw [exists_closed_atlas_iff_init_mem_safe GameConfig.standard,
      tetrisSolvableValid_iff_init_safe]

/-- **Canonical Tetris is solvable ⟺ `init` is in the atlas of the finite
universe.** The fully computational top-level reduction: `TetrisSolvableValid`
holds **iff** `init ∈ atlas standard (inFieldStates standard)`, a single
decidable Finset-membership test on the atlas built over the canonical state
universe. Solving Tetris reduces to one atlas lookup. -/
theorem tetrisSolvableValid_iff_init_mem_atlas_inFieldStates :
    TetrisSolvableValid ↔
      GameState.init ∈
        atlas GameConfig.standard (inFieldStates GameConfig.standard) := by
  rw [tetrisSolvableValid_iff_init_safe,
      init_mem_safe_iff_init_mem_atlas_inFieldStates]

/-! ### Discoverability: the atlas test is COMPUTABLE

`inFieldStates` is a plain (computable) `def`, so the canonical universe — and
hence `atlas cfg (inFieldStates cfg)` and membership in it — are computable
`Finset` objects. Transporting the iter-80/81 iffs along
`decidable_of_iff` therefore yields **computable** decision procedures for
`init ∈ safe` and for `TetrisSolvableValid`. This is strictly stronger than the
existing `decidable_init_safe` / `decidable_tetrisSolvableValid`
(`SafeIterateFinite`), which are `noncomputable` because they route through the
`noncomputable` `safeReachableCycle`. The atlas existence question is not merely
decidable in principle — it is *discoverable by computation* (build the atlas of
the finite universe, test `init ∈ ·`), even though the universe is infeasibly
large to enumerate in practice. -/

/-- **`init ∈ safe` is COMPUTABLY decidable**, via the atlas constructor over the
finite universe `inFieldStates`. No stability witness and no `noncomputable`:
`atlas cfg (inFieldStates cfg)` is an honest computable `Finset`, and membership
is `Finset.decidableMem`. The `safe`-membership proposition (otherwise built from
a `noncomputable` greatest fixpoint) is reduced to a concrete finite computation. -/
def decidableInitMemSafeViaAtlas (cfg : GameConfig) :
    Decidable (GameState.init ∈ safe cfg) :=
  decidable_of_iff _ (init_mem_safe_iff_init_mem_atlas_inFieldStates cfg).symm

/-- **`TetrisSolvableValid` is COMPUTABLY decidable.** Composes the computable
atlas decision procedure with the headline `tetrisSolvableValid_iff_init_safe`:
whether the canonical game admits a forever-surviving valid solver is settled by
one computable atlas-membership test. -/
def decidableTetrisSolvableValid : Decidable TetrisSolvableValid :=
  letI := decidableInitMemSafeViaAtlas GameConfig.standard
  decidable_of_iff _ tetrisSolvableValid_iff_init_safe.symm

/-- **Atlas refutation: an empty canonical atlas certifies UNSOLVABILITY.** If
the atlas of the finite universe comes back empty, then `init ∉ atlas`, so by the
top-level reduction the canonical game is not (valid-)solvable. This is the
formal counterpart of the project's empirical refutation route — every
`(rows, holes)` closed-GFP rung whose fixpoint collapses to `∅` thereby *proves*
no surviving solver exists for that universe. -/
theorem not_tetrisSolvableValid_of_atlas_inFieldStates_empty
    (h : atlas GameConfig.standard (inFieldStates GameConfig.standard) = ∅) :
    ¬ TetrisSolvableValid := by
  rw [tetrisSolvableValid_iff_init_mem_atlas_inFieldStates, h]
  exact Finset.notMem_empty _

/-! ### Existence verdict: a Tetris atlas, if it exists, is necessarily holey

The empirical/structural evidence leans toward NON-existence. These two
theorems pin down a sharp, GLOBAL necessary structural condition that any
existing atlas must satisfy — sharpening the obstruction surface that an
existence proof would have to thread, and giving the contrapositive a
hole-free atlas would refute solvability. Unlike
`no_init_closed_atlas_in_holes_zero_universe` (which restricts to a
holes-0 *universe*), this is unconditional on the universe: it shows the
forced first S-draw injects a holey board into ANY closed init-set. -/

/-- **Every Tetris atlas contains a holey board.** Any closed set `T`
containing `init` must contain a board with ≥ 1 buried hole — concretely
the S-successor of `init`. Since `init.board = ∅` is flat and the S-piece
unavoidably buries a cell on a flat surface, the adversary's very first
S-draw forces a hole into the atlas. Rules out the "clean atlas"
hypothesis globally: no Tetris atlas can consist solely of hole-free
boards. -/
theorem exists_holey_board_of_init_closed
    {T : Finset GameState}
    (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, 1 ≤ holes GameConfig.standard g.board := by
  have hSb : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  obtain ⟨_, hstep⟩ := hclosed GameState.init hinit
  obtain ⟨pl, hpl, hsucc⟩ := hstep Piece.S hSb
  rw [Placement.mem_allValidFor] at hpl
  obtain ⟨hpiece, hvalid⟩ := hpl
  refine ⟨adversarialStep GameConfig.standard GameState.init Piece.S pl, hsucc, ?_⟩
  have hpl_eq : ({pl with piece := Piece.S} : Placement) = pl := by
    rcases pl with ⟨pp, pr, pc⟩
    cases hpiece
    rfl
  have hsucc_board :
      (adversarialStep GameConfig.standard GameState.init Piece.S pl).board =
        pl.applyStep GameConfig.standard ∅ := by
    rw [adversarialStep_board, GameState.init_board_eq_emptyset, hpl_eq]
  rw [hsucc_board]
  exact one_le_holes_S_applyStep_empty (by simp) hvalid hpiece

/-- **Existence ⇒ a closed witness with a holey board.** Restated on the
M4 existence predicate itself: IF the Tetris atlas exists
(`TetrisSolvableValid`), THEN there is a closed init-containing witness `T`
that necessarily contains a board with ≥ 1 hole. Contrapositive: any
candidate winning strategy whose every reachable board is hole-free is
*refuted*. Bridges the structural hole obstruction directly to the
existence question via the iter-81 abstract-witness reduction
(`exists_closed_atlas_iff_tetrisSolvableValid`); routing through the
abstract witness avoids elaborating the astronomical canonical universe. -/
theorem holey_board_of_tetrisSolvableValid
    (h : TetrisSolvableValid) :
    ∃ T : Finset GameState, GameState.init ∈ T ∧
      IsClosedSet GameConfig.standard T ∧
      ∃ g ∈ T, 1 ≤ holes GameConfig.standard g.board := by
  obtain ⟨T, hinitT, hclosed⟩ := exists_closed_atlas_iff_tetrisSolvableValid.mpr h
  exact ⟨T, hinitT, hclosed, exists_holey_board_of_init_closed hinitT hclosed⟩

/-- **Every Tetris atlas contains TWO distinct holey boards at depth 1.**
Sharpening of `exists_holey_board_of_init_closed`: the adversary's first
draw can be S *or* Z, and BOTH the S-successor and the Z-successor of
`init` lie in any closed init-set, each carrying ≥ 1 buried hole. They are
distinct because their bags differ (`full.erase S ≠ full.erase Z`). So an
atlas is not merely "not hole-free" — it already holds two different holey
states one step from the start, evidence that holes proliferate rather
than stay isolated (tightening the holes≤1-impossibility intuition). -/
theorem exists_two_distinct_holey_boards_of_init_closed
    {T : Finset GameState}
    (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g₁ ∈ T, ∃ g₂ ∈ T, g₁ ≠ g₂ ∧
      1 ≤ holes GameConfig.standard g₁.board ∧
      1 ≤ holes GameConfig.standard g₂.board := by
  have hSb : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  have hZb : Piece.Z ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.Z
  obtain ⟨_, hstep⟩ := hclosed GameState.init hinit
  obtain ⟨plS, hplS, hsuccS⟩ := hstep Piece.S hSb
  obtain ⟨plZ, hplZ, hsuccZ⟩ := hstep Piece.Z hZb
  rw [Placement.mem_allValidFor] at hplS hplZ
  obtain ⟨hpieceS, hvalidS⟩ := hplS
  obtain ⟨hpieceZ, hvalidZ⟩ := hplZ
  have holeS : 1 ≤ holes GameConfig.standard
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).board := by
    have hpl_eq : ({plS with piece := Piece.S} : Placement) = plS := by
      rcases plS with ⟨pp, pr, pc⟩; cases hpieceS; rfl
    have hb : (adversarialStep GameConfig.standard GameState.init Piece.S plS).board =
        plS.applyStep GameConfig.standard ∅ := by
      rw [adversarialStep_board, GameState.init_board_eq_emptyset, hpl_eq]
    rw [hb]; exact one_le_holes_S_applyStep_empty (by simp) hvalidS hpieceS
  have holeZ : 1 ≤ holes GameConfig.standard
      (adversarialStep GameConfig.standard GameState.init Piece.Z plZ).board := by
    have hpl_eq : ({plZ with piece := Piece.Z} : Placement) = plZ := by
      rcases plZ with ⟨pp, pr, pc⟩; cases hpieceZ; rfl
    have hb : (adversarialStep GameConfig.standard GameState.init Piece.Z plZ).board =
        plZ.applyStep GameConfig.standard ∅ := by
      rw [adversarialStep_board, GameState.init_board_eq_emptyset, hpl_eq]
    rw [hb]; exact one_le_holes_Z_applyStep_empty (by simp) hvalidZ hpieceZ
  have hne : (adversarialStep GameConfig.standard GameState.init Piece.S plS) ≠
      (adversarialStep GameConfig.standard GameState.init Piece.Z plZ) := by
    intro heq
    have hbag : (adversarialStep GameConfig.standard GameState.init Piece.S plS).bag =
        (adversarialStep GameConfig.standard GameState.init Piece.Z plZ).bag :=
      congrArg GameState.bag heq
    rw [adversarialStep_bag, adversarialStep_bag, GameState.init_bag,
        Bag.draw_full_eq_erase, Bag.draw_full_eq_erase] at hbag
    have hSin : Piece.S ∈ Bag.full.erase Piece.Z :=
      Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.S⟩
    rw [← hbag] at hSin
    exact (Finset.notMem_erase Piece.S Bag.full) hSin
  exact ⟨_, hsuccS, _, hsuccZ, hne, holeS, holeZ⟩

/-- **Existence ⇒ a closed witness with two distinct holey boards.**
Headline bridge of `exists_two_distinct_holey_boards_of_init_closed` to the
M4 existence predicate, routed through the iter-81 abstract-witness iff. If
the Tetris atlas exists, any closing witness already contains two different
holey boards. Contrapositive: a strategy whose reachable set has at most
one holey board (in particular a hole-free one) is refuted. -/
theorem two_holey_boards_of_tetrisSolvableValid
    (h : TetrisSolvableValid) :
    ∃ T : Finset GameState, GameState.init ∈ T ∧
      IsClosedSet GameConfig.standard T ∧
      ∃ g₁ ∈ T, ∃ g₂ ∈ T, g₁ ≠ g₂ ∧
        1 ≤ holes GameConfig.standard g₁.board ∧
        1 ≤ holes GameConfig.standard g₂.board := by
  obtain ⟨T, hinitT, hclosed⟩ := exists_closed_atlas_iff_tetrisSolvableValid.mpr h
  exact ⟨T, hinitT, hclosed,
    exists_two_distinct_holey_boards_of_init_closed hinitT hclosed⟩

/-- **Total hole-count of any Tetris atlas is ≥ 2.** Aggregating
`exists_two_distinct_holey_boards_of_init_closed`: the two distinct
depth-1 holey successors (S- and Z-successors of `init`) each contribute
≥ 1 to the hole-sum over `T`, so any closed init-set carries at least two
buried cells in total. A quantitative "an atlas cannot be (near-)clean"
invariant: it lower-bounds the global structural debt the strategy must
forever manage. -/
theorem two_le_sum_holes_of_init_closed
    {T : Finset GameState}
    (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    2 ≤ ∑ g ∈ T, holes GameConfig.standard g.board := by
  obtain ⟨g₁, hg₁, g₂, hg₂, hne, h₁, h₂⟩ :=
    exists_two_distinct_holey_boards_of_init_closed hinit hclosed
  have hsub : ({g₁, g₂} : Finset GameState) ⊆ T := by
    intro x hx
    rcases Finset.mem_insert.mp hx with h | h
    · subst h; exact hg₁
    · rw [Finset.mem_singleton] at h; subst h; exact hg₂
  have hpair :
      (∑ g ∈ ({g₁, g₂} : Finset GameState), holes GameConfig.standard g.board)
        = holes GameConfig.standard g₁.board + holes GameConfig.standard g₂.board :=
    Finset.sum_pair hne
  have hle :
      (∑ g ∈ ({g₁, g₂} : Finset GameState), holes GameConfig.standard g.board)
        ≤ ∑ g ∈ T, holes GameConfig.standard g.board :=
    Finset.sum_le_sum_of_subset hsub
  omega

/-- **Existence ⇒ a closed witness with hole-sum ≥ 2.** Headline bridge of
`two_le_sum_holes_of_init_closed` to the M4 existence predicate, routed
through the iter-81 abstract-witness iff. Contrapositive: any candidate
winning strategy whose reachable set has total hole-count ≤ 1 is refuted. -/
theorem two_le_sum_holes_of_tetrisSolvableValid
    (h : TetrisSolvableValid) :
    ∃ T : Finset GameState, GameState.init ∈ T ∧
      IsClosedSet GameConfig.standard T ∧
      2 ≤ ∑ g ∈ T, holes GameConfig.standard g.board := by
  obtain ⟨T, hinitT, hclosed⟩ := exists_closed_atlas_iff_tetrisSolvableValid.mpr h
  exact ⟨T, hinitT, hclosed, two_le_sum_holes_of_init_closed hinitT hclosed⟩

/-! ### Opening-move dynamics ("phase 0")

A different angle from the hole corollaries: the EXACT behaviour of the
atlas/solver on its very first move. Whatever piece the adversary draws,
the opening response is *pure placement* — no line can clear, because four
cells cannot fill a width-10 row. These are the reusable building blocks
for any concrete depth-2 reasoning (e.g. analysing the S-successor board
to push the holes obstruction past depth 1). -/

/-- **Opening move is pure placement.** For any adversary piece `p` and any
valid placement `pl`, the first successor of `init` has board exactly
`pl.place ∅` — the dropped piece, with NO line clear. Follows from
`init.board = ∅` and `applyStep_empty_eq_place_empty` (count 4 < cols ⇒
`clearLines` is the identity). -/
theorem succ_init_board_eq_place_empty {p : Piece}
    {pl : Placement} (hpl : pl ∈ Placement.allValidFor GameConfig.standard p) :
    (adversarialStep GameConfig.standard GameState.init p pl).board = pl.place ∅ := by
  rw [Placement.mem_allValidFor] at hpl
  obtain ⟨hpiece, _⟩ := hpl
  have hpl_eq : ({pl with piece := p} : Placement) = pl := by
    rcases pl with ⟨pp, pr, pc⟩; cases hpiece; rfl
  rw [adversarialStep_board, GameState.init_board_eq_emptyset, hpl_eq,
      applyStep_empty_eq_place_empty (cfg := GameConfig.standard) (by simp)]

/-- **Opening cell-count is exactly 4.** The first successor of `init`
carries exactly the four cells of the dropped piece: no clears, no holes
removed. A quantitative "phase 0" invariant — the game opens in pure
accumulation (count `4 < 10`), so the first line clear is necessarily
deferred past move 1. -/
theorem count_succ_init_eq_four {p : Piece}
    {pl : Placement} (hpl : pl ∈ Placement.allValidFor GameConfig.standard p) :
    (adversarialStep GameConfig.standard GameState.init p pl).board.count = 4 := by
  rw [succ_init_board_eq_place_empty hpl, Placement.count_place]
  rfl

/-! ### Stack-volume lower bound (synthesis: count + holes + conservation)

A new quantity for the atlas, distinct from the holes corollaries and the
opening-move counts: a *lower* bound on total stack volume `∑ⱼ colHeight`.
Prior atlas height facts were all *upper* bounds (`colHeight ≤ 20`,
`holes ≤ 200`). Here three previously separate results combine through the
conservation law `count + holes = ∑ colHeight`:
* `count_succ_init_eq_four` — the S-successor of `init` has exactly 4 cells;
* `one_le_holes_S_applyStep_empty` — it also buries ≥ 1 hole;
* `reachableAt_count_add_holes_eq_sum_colHeight` — volume = cells + holes.
So that one reachable atlas state already stacks volume `≥ 4 + 1 = 5`. -/

/-- **Every Tetris atlas contains a board of stack-volume ≥ 5.** For any closed
`standard` table `T` containing `init`, some `g ∈ T` has
`∑ⱼ colHeight g.board ≥ 5`. The S-successor of `init` is reachable at depth 1,
carries exactly 4 filled cells (`count_succ_init_eq_four`) AND at least 1 buried
hole (`one_le_holes_S_applyStep_empty`), so by the volume conservation law its
column-height total is `count + holes ≥ 5`. The first *lower* bound on the
stack volume of an atlas state — complementing the `≤ 200` envelope and pinning
the atlas to a nondegenerate height band. -/
theorem exists_sum_colHeight_ge_five_of_init_closed
    {T : Finset GameState}
    (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, 5 ≤ ∑ j ∈ Finset.range GameConfig.standard.cols, g.board.colHeight j := by
  have hSb : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  obtain ⟨hinit_nl, hstep⟩ := hclosed GameState.init hinit
  obtain ⟨pl, hpl, hsucc⟩ := hstep Piece.S hSb
  obtain ⟨hpiece, hvalid⟩ := (Placement.mem_allValidFor GameConfig.standard Piece.S pl).mp hpl
  refine ⟨adversarialStep GameConfig.standard GameState.init Piece.S pl, hsucc, ?_⟩
  have hpl_eq : ({pl with piece := Piece.S} : Placement) = pl := by
    rcases pl with ⟨pp, pr, pc⟩; cases hpiece; rfl
  have hvalid' : ({pl with piece := Piece.S} : Placement).Valid GameConfig.standard := by
    rw [hpl_eq]; exact hvalid
  have hreach : ReachableAt GameConfig.standard 1
      (adversarialStep GameConfig.standard GameState.init Piece.S pl) :=
    ReachableAt.step ReachableAt.init hinit_nl hSb hvalid'
  have hcount : (adversarialStep GameConfig.standard GameState.init Piece.S pl).board.count = 4 :=
    count_succ_init_eq_four hpl
  have hb : (adversarialStep GameConfig.standard GameState.init Piece.S pl).board =
      pl.applyStep GameConfig.standard ∅ := by
    rw [adversarialStep_board, GameState.init_board_eq_emptyset, hpl_eq]
  have hholes : 1 ≤ holes GameConfig.standard
      (adversarialStep GameConfig.standard GameState.init Piece.S pl).board := by
    rw [hb]; exact one_le_holes_S_applyStep_empty (by simp) hvalid hpiece
  have hcons := reachableAt_count_add_holes_eq_sum_colHeight hreach
  omega

/-- **Existence ⇒ a closed witness with stack-volume ≥ 5.** Headline bridge of
`exists_sum_colHeight_ge_five_of_init_closed` to the M4 existence predicate,
routed through the iter-81 abstract-witness iff (so the astronomical canonical
universe is never elaborated). If the Tetris atlas exists, any closing witness
contains a board whose columns total height ≥ 5. Contrapositive: a candidate
strategy whose every reachable board has stack volume ≤ 4 is refuted. -/
theorem sum_colHeight_ge_five_of_tetrisSolvableValid
    (h : TetrisSolvableValid) :
    ∃ T : Finset GameState, GameState.init ∈ T ∧
      IsClosedSet GameConfig.standard T ∧
      ∃ g ∈ T, 5 ≤ ∑ j ∈ Finset.range GameConfig.standard.cols, g.board.colHeight j := by
  obtain ⟨T, hinitT, hclosed⟩ := exists_closed_atlas_iff_tetrisSolvableValid.mpr h
  exact ⟨T, hinitT, hclosed, exists_sum_colHeight_ge_five_of_init_closed hinitT hclosed⟩

/-! ### Depth-2 opening dynamics: the S-then-Z board has exactly 8 cells

Pushing past the depth-1 successor that powered iters 83–87. The adversary may
draw S, then (from the S-successor, whose bag is `full.erase S`) draw Z. That
two-move board still holds only `4 + 4 = 8 < 10` cells, so — exactly as at
depth 1 — *no line can have cleared yet* (`clearLines_eq_self_of_count_lt`). The
key enabler is `Placement.count_place : (pl.place b).count = b.count + 4`, which
holds for ANY board (the dropped piece is always disjoint from `b`), so it
chains across moves without a validity side condition. This is the count
backbone for a future single-board `holes ≥ 2` argument (S-hole + Z-hole on the
same depth-2 board), the route to killing `holes ≤ 1` atlases. -/

/-- **Every Tetris atlas contains a depth-2 board with exactly 8 cells.** For any
closed `standard` table `T ∋ init`, the S-successor of `init` lies in `T`
(closedness), and from it the Z-successor also lies in `T` (Z is still in the
bag `full.erase S`). That two-move board has count `= 8`: each placement adds
exactly 4 cells (`count_place`), and 8 < 10 means no row filled, so line-clearing
is the identity. The depth-2 analogue of `count_succ_init_eq_four`, proving the
first line clear is deferred past move 2. -/
theorem exists_depth_two_count_eight_of_init_closed
    {T : Finset GameState}
    (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, g.board.count = 8 := by
  have hSb : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  obtain ⟨_, hstepInit⟩ := hclosed GameState.init hinit
  obtain ⟨plS, hplS, hsuccS⟩ := hstepInit Piece.S hSb
  have hcountS :
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).board.count = 4 :=
    count_succ_init_eq_four hplS
  have hZb : Piece.Z ∈
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).bag := by
    rw [adversarialStep_bag, GameState.init_bag, Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.Z⟩
  obtain ⟨_, hstepS⟩ := hclosed _ hsuccS
  obtain ⟨plZ, hplZ, hsuccZ⟩ := hstepS Piece.Z hZb
  refine ⟨adversarialStep GameConfig.standard
      (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ, hsuccZ, ?_⟩
  have hpre : (({plZ with piece := Piece.Z} : Placement).place
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).board).count = 8 := by
    rw [Placement.count_place, hcountS]
  have hclear : Board.clearLines GameConfig.standard
      (({plZ with piece := Piece.Z} : Placement).place
        (adversarialStep GameConfig.standard GameState.init Piece.S plS).board)
      = ({plZ with piece := Piece.Z} : Placement).place
        (adversarialStep GameConfig.standard GameState.init Piece.S plS).board := by
    apply Board.clearLines_eq_self_of_count_lt
    rw [hpre]; decide
  rw [adversarialStep_board, Placement.applyStep_eq_clearLines_place, hclear]
  exact hpre

/-- **Existence ⇒ a closed witness with a depth-2 8-cell board.** Headline bridge
of `exists_depth_two_count_eight_of_init_closed` to `TetrisSolvableValid`, routed
through the iter-81 abstract-witness iff (the astronomical universe is never
elaborated). Any atlas necessarily reaches a state with exactly 8 filled cells
two moves in — accumulation with no clears through move 2. -/
theorem depth_two_count_eight_of_tetrisSolvableValid
    (h : TetrisSolvableValid) :
    ∃ T : Finset GameState, GameState.init ∈ T ∧
      IsClosedSet GameConfig.standard T ∧
      ∃ g ∈ T, g.board.count = 8 := by
  obtain ⟨T, hinitT, hclosed⟩ := exists_closed_atlas_iff_tetrisSolvableValid.mpr h
  exact ⟨T, hinitT, hclosed, exists_depth_two_count_eight_of_init_closed hinitT hclosed⟩

/-- **Depth-2 bag-card ladder (phase structure of `safe`).** Any closed atlas
containing `init` reaches, two adversarial moves in (draw `S` then `Z`), a state
whose bag has exactly `5` pieces remaining. The 7-bag refills only when emptied,
so the card descends `7 → 6 → 5` across the first two draws with no refill: this
pins the depth-2 phase. Proof: `card_draw_of_ge_two` twice off `Bag.full`
(`card 7 → 6` then `6 → 5`), routed through the abstract closed witness so the
giant `inFieldStates` term is never elaborated. -/
theorem exists_depth_two_bagcard_five_of_init_closed
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, g.bag.card = 5 := by
  have hSb : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  obtain ⟨_, hstepInit⟩ := hclosed GameState.init hinit
  obtain ⟨plS, _, hsuccS⟩ := hstepInit Piece.S hSb
  have hZb : Piece.Z ∈
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).bag := by
    rw [adversarialStep_bag, GameState.init_bag, Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.Z⟩
  obtain ⟨_, hstepS⟩ := hclosed _ hsuccS
  obtain ⟨plZ, _, hsuccZ⟩ := hstepS Piece.Z hZb
  refine ⟨adversarialStep GameConfig.standard
      (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ,
      hsuccZ, ?_⟩
  have hZin : Piece.Z ∈ Bag.full.draw Piece.S := by
    rw [Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.Z⟩
  have h2 : 2 ≤ (Bag.full.draw Piece.S).card := by rw [Bag.draw_full_card]; omega
  simp only [adversarialStep_bag, GameState.init_bag]
  rw [Bag.card_draw_of_ge_two hZin h2, Bag.draw_full_card]

/-- `TetrisSolvableValid` form of the depth-2 bag-card ladder: any valid atlas
necessarily contains a state two moves deep whose bag holds exactly 5 pieces. -/
theorem depth_two_bagcard_five_of_tetrisSolvableValid
    (h : TetrisSolvableValid) :
    ∃ T : Finset GameState, GameState.init ∈ T ∧
      IsClosedSet GameConfig.standard T ∧
      ∃ g ∈ T, g.bag.card = 5 := by
  obtain ⟨T, hinitT, hclosed⟩ := exists_closed_atlas_iff_tetrisSolvableValid.mpr h
  exact ⟨T, hinitT, hclosed, exists_depth_two_bagcard_five_of_init_closed hinitT hclosed⟩

/-- **Depth-3 bag-card ladder.** Any closed atlas containing `init` reaches, three
adversarial moves in (draw `S`, then `Z`, then `T`), a state whose bag has exactly
`4` pieces. The 7-bag refills only on emptying, so the card descends `7 → 6 → 5 → 4`
over the first three draws with no refill — one rung deeper than the depth-2
`bag.card = 5` invariant. Proof: peel three draws off `Bag.full` with
`card_draw_of_ge_two`, routed through the abstract closed witness so the giant
`inFieldStates` term is never elaborated. The third adversary piece `T` is a
concrete survivor of `(Bag.full.draw S).draw Z = (Bag.full.erase S).erase Z`. -/
theorem exists_depth_three_bagcard_four_of_init_closed
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, g.bag.card = 4 := by
  have hSb : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  obtain ⟨_, hstepInit⟩ := hclosed GameState.init hinit
  obtain ⟨plS, _, hsuccS⟩ := hstepInit Piece.S hSb
  have hZb : Piece.Z ∈
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).bag := by
    rw [adversarialStep_bag, GameState.init_bag, Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.Z⟩
  obtain ⟨_, hstepS⟩ := hclosed _ hsuccS
  obtain ⟨plZ, _, hsuccZ⟩ := hstepS Piece.Z hZb
  have hZin : Piece.Z ∈ Bag.full.draw Piece.S := by
    rw [Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.Z⟩
  have h2 : 2 ≤ (Bag.full.draw Piece.S).card := by rw [Bag.draw_full_card]; omega
  have hTin : Piece.T ∈ (Bag.full.draw Piece.S).draw Piece.Z := by
    rw [Bag.draw_eq_erase_of_card_ge_two hZin h2, Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr
      ⟨by decide, Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.T⟩⟩
  have hTb : Piece.T ∈
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).bag := by
    rw [adversarialStep_bag, adversarialStep_bag, GameState.init_bag]
    exact hTin
  obtain ⟨_, hstepZ⟩ := hclosed _ hsuccZ
  obtain ⟨plT, _, hsuccT⟩ := hstepZ Piece.T hTb
  refine ⟨adversarialStep GameConfig.standard
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ)
      Piece.T plT, hsuccT, ?_⟩
  have h3 : 2 ≤ ((Bag.full.draw Piece.S).draw Piece.Z).card := by
    rw [Bag.card_draw_of_ge_two hZin h2, Bag.draw_full_card]; omega
  simp only [adversarialStep_bag, GameState.init_bag]
  rw [Bag.card_draw_of_ge_two hTin h3, Bag.card_draw_of_ge_two hZin h2, Bag.draw_full_card]

/-- `TetrisSolvableValid` form of the depth-3 bag-card ladder: any valid atlas
necessarily contains a state three moves deep whose bag holds exactly 4 pieces. -/
theorem depth_three_bagcard_four_of_tetrisSolvableValid
    (h : TetrisSolvableValid) :
    ∃ T : Finset GameState, GameState.init ∈ T ∧
      IsClosedSet GameConfig.standard T ∧
      ∃ g ∈ T, g.bag.card = 4 := by
  obtain ⟨T, hinitT, hclosed⟩ := exists_closed_atlas_iff_tetrisSolvableValid.mpr h
  exact ⟨T, hinitT, hclosed, exists_depth_three_bagcard_four_of_init_closed hinitT hclosed⟩

/-- An `adversarialStep` preserves board well-formedness (the column bound
`∀ p ∈ b, p.1 < cfg.cols`) for a valid placement. Chains `adversarialStep_board`
into `Placement.applyStep_wf`. -/
theorem adversarialStep_board_wf {cfg : GameConfig} {g : GameState} {p : Piece}
    {pl : Placement} (hg : Board.WF cfg g.board)
    (hv : ({pl with piece := p} : Placement).Valid cfg) :
    Board.WF cfg (adversarialStep cfg g p pl).board := by
  rw [adversarialStep_board]
  exact Placement.applyStep_wf hg hv

/-- **Depth-3 board-count dichotomy (first board-structure verdict past the
no-clear band).** Any closed atlas containing `init` reaches, three adversarial
moves in (draw `S`, `Z`, `T`), a board whose cell count is *exactly* `2` or `12`.
At depth 3 the placed board carries `12` cells (`8 + 4`, by `count_place`), the
first depth at which a line *could* fill (`12 ≥ 10 = cols`). The conservation
`clearLines_count_add` gives `12 = postCount + 10·k` with `k = #fullRows`; since
`postCount ≥ 0` forces `k ≤ 1`, the post-clear count is `12` (no clear) or `2`
(one line). Unlike the pure bag-card ladder this constrains the *board*. Routed
through the abstract closed witness; WF tracked via `adversarialStep_board_wf`. -/
theorem exists_depth_three_count_two_or_twelve_of_init_closed
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, g.board.count = 2 ∨ g.board.count = 12 := by
  -- move 1: draw S, extract a valid S-placement
  have hSb : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  obtain ⟨_, hstepInit⟩ := hclosed GameState.init hinit
  obtain ⟨plS, hplS, hsuccS⟩ := hstepInit Piece.S hSb
  obtain ⟨hpieceS, hvalidS⟩ :=
    (Placement.mem_allValidFor GameConfig.standard Piece.S plS).mp hplS
  have hvS' : ({plS with piece := Piece.S} : Placement).Valid GameConfig.standard := by
    have : ({plS with piece := Piece.S} : Placement) = plS := by
      rcases plS with ⟨pp, pr, pc⟩; cases hpieceS; rfl
    rw [this]; exact hvalidS
  -- move 2: draw Z, extract a valid Z-placement
  have hZb : Piece.Z ∈
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).bag := by
    rw [adversarialStep_bag, GameState.init_bag, Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.Z⟩
  obtain ⟨_, hstepS⟩ := hclosed _ hsuccS
  obtain ⟨plZ, hplZ, hsuccZ⟩ := hstepS Piece.Z hZb
  obtain ⟨hpieceZ, hvalidZ⟩ :=
    (Placement.mem_allValidFor GameConfig.standard Piece.Z plZ).mp hplZ
  have hvZ' : ({plZ with piece := Piece.Z} : Placement).Valid GameConfig.standard := by
    have : ({plZ with piece := Piece.Z} : Placement) = plZ := by
      rcases plZ with ⟨pp, pr, pc⟩; cases hpieceZ; rfl
    rw [this]; exact hvalidZ
  -- move 3: draw T, extract a valid T-placement
  have hZin : Piece.Z ∈ Bag.full.draw Piece.S := by
    rw [Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.Z⟩
  have h2 : 2 ≤ (Bag.full.draw Piece.S).card := by rw [Bag.draw_full_card]; omega
  have hTin : Piece.T ∈ (Bag.full.draw Piece.S).draw Piece.Z := by
    rw [Bag.draw_eq_erase_of_card_ge_two hZin h2, Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr
      ⟨by decide, Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.T⟩⟩
  have hTb : Piece.T ∈
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).bag := by
    rw [adversarialStep_bag, adversarialStep_bag, GameState.init_bag]; exact hTin
  obtain ⟨_, hstepZ⟩ := hclosed _ hsuccZ
  obtain ⟨plT, hplT, hsuccT⟩ := hstepZ Piece.T hTb
  obtain ⟨hpieceT, hvalidT⟩ :=
    (Placement.mem_allValidFor GameConfig.standard Piece.T plT).mp hplT
  have hvT' : ({plT with piece := Piece.T} : Placement).Valid GameConfig.standard := by
    have : ({plT with piece := Piece.T} : Placement) = plT := by
      rcases plT with ⟨pp, pr, pc⟩; cases hpieceT; rfl
    rw [this]; exact hvalidT
  refine ⟨adversarialStep GameConfig.standard
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ)
      Piece.T plT, hsuccT, ?_⟩
  -- WF chain: ∅ → depth-1 → depth-2 boards
  have hWF0 : Board.WF GameConfig.standard GameState.init.board := by
    rw [GameState.init_board_eq_emptyset]
    intro q hq; exact absurd hq (Finset.notMem_empty q)
  have hWF1 : Board.WF GameConfig.standard
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).board :=
    adversarialStep_board_wf hWF0 hvS'
  have hWF2 : Board.WF GameConfig.standard
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board :=
    adversarialStep_board_wf hWF1 hvZ'
  -- depth-2 board count = 8 (4 from S-successor + 4, no clear at 8 < 10)
  have hb1count :
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).board.count = 4 :=
    count_succ_init_eq_four hplS
  have hplaceZcount :
      (({plZ with piece := Piece.Z} : Placement).place
        (adversarialStep GameConfig.standard GameState.init Piece.S plS).board).count = 8 := by
    rw [Placement.count_place, hb1count]
  have hb2_eq :
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board
      = ({plZ with piece := Piece.Z} : Placement).place
        (adversarialStep GameConfig.standard GameState.init Piece.S plS).board := by
    rw [adversarialStep_board, Placement.applyStep_eq_clearLines_place]
    apply Board.clearLines_eq_self_of_count_lt
    rw [hplaceZcount]; decide
  have hb2count :
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board.count = 8 := by
    rw [hb2_eq]; exact hplaceZcount
  -- depth-3 placed board (pre-clear) count = 12, and is WF
  have hplaced3count :
      (({plT with piece := Piece.T} : Placement).place
        (adversarialStep GameConfig.standard
          (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board).count
        = 12 := by
    rw [Placement.count_place, hb2count]
  have hWFplaced3 : Board.WF GameConfig.standard
      (({plT with piece := Piece.T} : Placement).place
        (adversarialStep GameConfig.standard
          (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board) :=
    Board.WF_place hWF2 hvT'
  -- conservation: 12 = postCount + 10·#fullRows ⇒ postCount ∈ {2, 12}
  have hadd := Board.clearLines_count_add hWFplaced3
  rw [hplaced3count] at hadd
  have hcols : GameConfig.standard.cols = 10 := by decide
  rw [hcols] at hadd
  rw [adversarialStep_board, Placement.applyStep_eq_clearLines_place]
  omega

/-- `TetrisSolvableValid` form of the depth-3 board-count dichotomy: any valid
atlas necessarily contains a state three moves deep whose board count is `2` or
`12`. -/
theorem depth_three_count_two_or_twelve_of_tetrisSolvableValid
    (h : TetrisSolvableValid) :
    ∃ T : Finset GameState, GameState.init ∈ T ∧
      IsClosedSet GameConfig.standard T ∧
      ∃ g ∈ T, g.board.count = 2 ∨ g.board.count = 12 := by
  obtain ⟨T, hinitT, hclosed⟩ := exists_closed_atlas_iff_tetrisSolvableValid.mpr h
  exact ⟨T, hinitT, hclosed,
    exists_depth_three_count_two_or_twelve_of_init_closed hinitT hclosed⟩

/-- **Drop-resting geometry.** Every hard-dropped cell lands at or above the
pre-drop column height of its own column. A piece rests on the surface and never
sinks below the existing stack: if `c ∈ pl.dropped b` then `b.colHeight c.1 ≤ c.2`.
This is the general-board analogue of `dropped_disjoint`, reading the supremum
drop offset as a per-cell lower bound rather than a disjointness fact. -/
theorem colHeight_le_of_mem_dropped (b : Board) (pl : Placement) {c : Coord}
    (h : c ∈ pl.dropped b) : b.colHeight c.1 ≤ c.2 := by
  rw [Placement.dropped_eq_image] at h
  obtain ⟨cell, hcell, rfl⟩ := Finset.mem_image.1 h
  show b.colHeight (pl.col + cell.1) ≤ pl.dropOffset b + cell.2
  have hle : b.colHeight (pl.col + cell.1) - cell.2 ≤ pl.dropOffset b :=
    Placement.le_sup_sub pl.shapeUp b.colHeight pl.col hcell
  omega

/-- A **newly filled** cell of a placement — one present in `pl.place b` but not
already in `b` — rests at or above its column's pre-placement height. This lifts
drop-resting from the bare dropped piece to the merged board, giving the entry
point for forcing holes on a general (non-empty) board: any covering cell of the
placed board that was not already there sits no lower than the prior surface. -/
theorem colHeight_le_of_mem_place_not_mem (b : Board) (pl : Placement) {c : Coord}
    (hmem : c ∈ pl.place b) (hnew : c ∉ b) : b.colHeight c.1 ≤ c.2 := by
  rw [Placement.place_eq_union_dropped, Finset.mem_union] at hmem
  rcases hmem with hb | hd
  · exact absurd hb hnew
  · exact colHeight_le_of_mem_dropped b pl hd

/-- **Clear-count volume bound.** A single `clearLines` pass removes `cfg.cols`
cells per cleared row, and the post-clear board still has a non-negative count,
so the number of simultaneously cleared rows is bounded by the cell budget:
`cfg.cols * #(fullRows) ≤ count`. In particular on `standard` (10 cols) a board
of count `c` clears at most `⌊c/10⌋` lines. -/
theorem cols_mul_fullRows_card_le_count {cfg : GameConfig} {b : Board}
    (hb : Board.WF cfg b) :
    cfg.cols * (Board.fullRows cfg b).card ≤ b.count := by
  have hadd := Board.clearLines_count_add hb
  omega

/-- **Per-move count law.** One full move — hard-drop then line clear — adds four
cells and removes a multiple of `cfg.cols`. For a well-formed board and a valid
placement, `(applyStep).count + cfg.cols * #(fullRows of the placed board)
= count + 4`. This is the uniform accounting that *subsumes* every depth-`n`
count dichotomy: the placed (pre-clear) board has `count + 4` cells and each
cleared line subtracts exactly `cfg.cols`, so the post-move count is
`count + 4 - cfg.cols * k` for `k = #fullRows`. The depth-3 `{2,12}` and the
prospective depth-4 `{6,16}` results are both instances with `count + 4 ∈ {12,16}`
and `k ∈ {0,1}` forced by `cols_mul_fullRows_card_le_count`. -/
theorem count_applyStep_add_fullRows {cfg : GameConfig} {b : Board} {pl : Placement}
    (hb : Board.WF cfg b) (hv : pl.Valid cfg) :
    (pl.applyStep cfg b).count
        + cfg.cols * (Board.fullRows cfg (pl.place b)).card
      = b.count + 4 := by
  have hwf : Board.WF cfg (pl.place b) := Board.WF_place hb hv
  have hadd := Board.clearLines_count_add hwf
  rw [Placement.count_place] at hadd
  rw [Placement.applyStep_eq_clearLines_place]
  omega

/-- **Per-move count upper bound.** A full move adds at most four cells: line
clearing never increases the count and the hard-drop adds exactly four, so
`(applyStep).count ≤ count + 4` for ANY board and placement (no well-formedness
or validity needed). -/
theorem count_applyStep_le (cfg : GameConfig) (b : Board) (pl : Placement) :
    (pl.applyStep cfg b).count ≤ b.count + 4 := by
  rw [Placement.applyStep_eq_clearLines_place]
  calc (Board.clearLines cfg (pl.place b)).count
      ≤ (pl.place b).count := Board.count_clearLines_le cfg (pl.place b)
    _ = b.count + 4 := Placement.count_place b pl

/-- **At-most-one-clear move dichotomy.** When the placed board still fits under
two full rows (`count + 4 < 2 * cols`), a single move clears at most one line, so
the post-move count is *either* `count + 4` (no clear) *or* `count + 4 - cols`
(exactly one clear). This eliminates the multiplicative `cols * k` term of the
per-move count law, turning every depth-`n` count constraint into a two-way case
split (depth-3 `{2,12}`, depth-4 `{6,16}`, …). -/
theorem count_applyStep_eq_or {cfg : GameConfig} {b : Board} {pl : Placement}
    (hb : Board.WF cfg b) (hv : pl.Valid cfg)
    (hsmall : b.count + 4 < 2 * cfg.cols) :
    (pl.applyStep cfg b).count = b.count + 4 ∨
      (pl.applyStep cfg b).count + cfg.cols = b.count + 4 := by
  have hlaw := count_applyStep_add_fullRows hb hv
  generalize hk : (Board.fullRows cfg (pl.place b)).card = k at hlaw
  have hk2 : k < 2 := by
    by_contra hge
    push_neg at hge
    have hmono : cfg.cols * 2 ≤ cfg.cols * k := by gcongr
    omega
  interval_cases k
  · left; omega
  · right; omega

/-- **Per-move count lower bound.** Under the same small-count hypothesis a single
move drops the count by at most `cols - 4`: `count + 4 - cols ≤ (applyStep).count`.
Together with `count_applyStep_le` this pins the per-move count change to the
envelope `[count + 4 - cols, count + 4]` — the count can neither jump by more than
four nor crash by more than `cols - 4` in one move. -/
theorem count_applyStep_ge {cfg : GameConfig} {b : Board} {pl : Placement}
    (hb : Board.WF cfg b) (hv : pl.Valid cfg)
    (hsmall : b.count + 4 < 2 * cfg.cols) :
    b.count + 4 - cfg.cols ≤ (pl.applyStep cfg b).count := by
  rcases count_applyStep_eq_or hb hv hsmall with h | h <;> omega

/-- **Depth-4 board-count dichotomy.** One adversarial move past depth 3 — draw `O`
(any of the remaining `{O,I,L,J}` works) — and the four-move board carries *exactly*
`6` or `16` cells. The depth-3 board has count `2` or `12`
(`exists_depth_three_count_two_or_twelve_of_init_closed`, re-derived here on the
concrete successor so the count is a usable hypothesis rather than hidden under an
`∃`); the per-move count law (`count_applyStep_add_fullRows`) adds `4` and subtracts
`10·k`, with `k ≤ 1` forced because `count + 4 ∈ {6,16} < 20 = 2·cols`. Case-splitting
the depth-3 dichotomy: `2 → 6` (no clear possible, `6 < 10`) and `12 → 16` (no clear)
or `6` (one clear). A pure board-structure constraint four moves deep, the next rung
of the count ladder; routed through the abstract closed witness, WF via
`adversarialStep_board_wf`. -/
theorem exists_depth_four_count_six_or_sixteen_of_init_closed
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, g.board.count = 6 ∨ g.board.count = 16 := by
  -- move 1: draw S, extract a valid S-placement
  have hSb : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  obtain ⟨_, hstepInit⟩ := hclosed GameState.init hinit
  obtain ⟨plS, hplS, hsuccS⟩ := hstepInit Piece.S hSb
  obtain ⟨hpieceS, hvalidS⟩ :=
    (Placement.mem_allValidFor GameConfig.standard Piece.S plS).mp hplS
  have hvS' : ({plS with piece := Piece.S} : Placement).Valid GameConfig.standard := by
    have : ({plS with piece := Piece.S} : Placement) = plS := by
      rcases plS with ⟨pp, pr, pc⟩; cases hpieceS; rfl
    rw [this]; exact hvalidS
  -- move 2: draw Z, extract a valid Z-placement
  have hZb : Piece.Z ∈
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).bag := by
    rw [adversarialStep_bag, GameState.init_bag, Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.Z⟩
  obtain ⟨_, hstepS⟩ := hclosed _ hsuccS
  obtain ⟨plZ, hplZ, hsuccZ⟩ := hstepS Piece.Z hZb
  obtain ⟨hpieceZ, hvalidZ⟩ :=
    (Placement.mem_allValidFor GameConfig.standard Piece.Z plZ).mp hplZ
  have hvZ' : ({plZ with piece := Piece.Z} : Placement).Valid GameConfig.standard := by
    have : ({plZ with piece := Piece.Z} : Placement) = plZ := by
      rcases plZ with ⟨pp, pr, pc⟩; cases hpieceZ; rfl
    rw [this]; exact hvalidZ
  -- move 3: draw T, extract a valid T-placement
  have hZin : Piece.Z ∈ Bag.full.draw Piece.S := by
    rw [Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.Z⟩
  have h2 : 2 ≤ (Bag.full.draw Piece.S).card := by rw [Bag.draw_full_card]; omega
  have hTin : Piece.T ∈ (Bag.full.draw Piece.S).draw Piece.Z := by
    rw [Bag.draw_eq_erase_of_card_ge_two hZin h2, Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr
      ⟨by decide, Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.T⟩⟩
  have hTb : Piece.T ∈
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).bag := by
    rw [adversarialStep_bag, adversarialStep_bag, GameState.init_bag]; exact hTin
  obtain ⟨_, hstepZ⟩ := hclosed _ hsuccZ
  obtain ⟨plT, hplT, hsuccT⟩ := hstepZ Piece.T hTb
  obtain ⟨hpieceT, hvalidT⟩ :=
    (Placement.mem_allValidFor GameConfig.standard Piece.T plT).mp hplT
  have hvT' : ({plT with piece := Piece.T} : Placement).Valid GameConfig.standard := by
    have : ({plT with piece := Piece.T} : Placement) = plT := by
      rcases plT with ⟨pp, pr, pc⟩; cases hpieceT; rfl
    rw [this]; exact hvalidT
  -- move 4: draw O, extract a valid O-placement
  have h2Z : 2 ≤ ((Bag.full.draw Piece.S).draw Piece.Z).card := by
    rw [Bag.card_draw_of_ge_two hZin h2, Bag.draw_full_card]; omega
  have hOin : Piece.O ∈ ((Bag.full.draw Piece.S).draw Piece.Z).draw Piece.T := by
    rw [Bag.draw_eq_erase_of_card_ge_two hTin h2Z,
      Bag.draw_eq_erase_of_card_ge_two hZin h2, Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr
      ⟨by decide, Finset.mem_erase.mpr
        ⟨by decide, Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.O⟩⟩⟩
  have hOb : Piece.O ∈
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard
          (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ)
        Piece.T plT).bag := by
    rw [adversarialStep_bag, adversarialStep_bag, adversarialStep_bag, GameState.init_bag]
    exact hOin
  obtain ⟨_, hstepT⟩ := hclosed _ hsuccT
  obtain ⟨plO, hplO, hsuccO⟩ := hstepT Piece.O hOb
  obtain ⟨hpieceO, hvalidO⟩ :=
    (Placement.mem_allValidFor GameConfig.standard Piece.O plO).mp hplO
  have hvO' : ({plO with piece := Piece.O} : Placement).Valid GameConfig.standard := by
    have : ({plO with piece := Piece.O} : Placement) = plO := by
      rcases plO with ⟨pp, pr, pc⟩; cases hpieceO; rfl
    rw [this]; exact hvalidO
  refine ⟨adversarialStep GameConfig.standard
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard
          (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ)
        Piece.T plT) Piece.O plO, hsuccO, ?_⟩
  -- WF chain: ∅ → depth-1 → depth-2 → depth-3 boards
  have hWF0 : Board.WF GameConfig.standard GameState.init.board := by
    rw [GameState.init_board_eq_emptyset]
    intro q hq; exact absurd hq (Finset.notMem_empty q)
  have hWF1 : Board.WF GameConfig.standard
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).board :=
    adversarialStep_board_wf hWF0 hvS'
  have hWF2 : Board.WF GameConfig.standard
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board :=
    adversarialStep_board_wf hWF1 hvZ'
  have hWF3 : Board.WF GameConfig.standard
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard
          (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ)
        Piece.T plT).board :=
    adversarialStep_board_wf hWF2 hvT'
  -- depth-2 board count = 8 (4 from S-successor + 4, no clear at 8 < 10)
  have hb1count :
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).board.count = 4 :=
    count_succ_init_eq_four hplS
  have hplaceZcount :
      (({plZ with piece := Piece.Z} : Placement).place
        (adversarialStep GameConfig.standard GameState.init Piece.S plS).board).count = 8 := by
    rw [Placement.count_place, hb1count]
  have hb2_eq :
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board
      = ({plZ with piece := Piece.Z} : Placement).place
        (adversarialStep GameConfig.standard GameState.init Piece.S plS).board := by
    rw [adversarialStep_board, Placement.applyStep_eq_clearLines_place]
    apply Board.clearLines_eq_self_of_count_lt
    rw [hplaceZcount]; decide
  have hb2count :
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board.count = 8 := by
    rw [hb2_eq]; exact hplaceZcount
  -- depth-3 placed board (pre-clear) count = 12, dichotomy via conservation
  have hplaced3count :
      (({plT with piece := Piece.T} : Placement).place
        (adversarialStep GameConfig.standard
          (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board).count
        = 12 := by
    rw [Placement.count_place, hb2count]
  have hWFplaced3 : Board.WF GameConfig.standard
      (({plT with piece := Piece.T} : Placement).place
        (adversarialStep GameConfig.standard
          (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board) :=
    Board.WF_place hWF2 hvT'
  have hb3 :
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard
          (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ)
        Piece.T plT).board.count = 2 ∨
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard
          (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ)
        Piece.T plT).board.count = 12 := by
    have hadd := Board.clearLines_count_add hWFplaced3
    rw [hplaced3count] at hadd
    have hcols : GameConfig.standard.cols = 10 := by decide
    rw [hcols] at hadd
    rw [adversarialStep_board, Placement.applyStep_eq_clearLines_place]
    omega
  -- move 4: per-move count law at the concrete depth-3 state ⇒ count ∈ {6,16}
  have hlaw4 := count_applyStep_add_fullRows hWF3 hvO'
  rw [adversarialStep_board]
  have hcols : GameConfig.standard.cols = 10 := by decide
  rw [hcols] at hlaw4
  rcases hb3 with h3 | h3 <;> rw [h3] at hlaw4 <;> omega

/-- S, Z, T pieces each span at most 3 distinct columns in every rotation. -/
theorem shapeUp_colImage_card_le_three {p : Piece}
    (hp : p = Piece.S ∨ p = Piece.Z ∨ p = Piece.T) (r : Rotation) :
    ((p.shapeUp r).image Prod.fst).card ≤ 3 := by
  rcases hp with h | h | h <;> subst h <;> fin_cases r <;> decide

/-- A dropped S/Z/T placement occupies at most 3 distinct columns on any board. -/
theorem dropped_colImage_card_le_three (pl : Placement) (b : Board)
    (hp : pl.piece = Piece.S ∨ pl.piece = Piece.Z ∨ pl.piece = Piece.T) :
    ((pl.dropped b).image Prod.fst).card ≤ 3 := by
  have h1 : (pl.dropped b).image Prod.fst
      = (pl.shapeUp.image Prod.fst).image (fun x => pl.col + x) := by
    unfold Placement.dropped Placement.cellsAt
    simp only [Finset.image_image, Function.comp_def]
  calc ((pl.dropped b).image Prod.fst).card
      = ((pl.shapeUp.image Prod.fst).image (fun x => pl.col + x)).card := by rw [h1]
    _ ≤ (pl.shapeUp.image Prod.fst).card := Finset.card_image_le
    _ ≤ 3 := by unfold Placement.shapeUp; exact shapeUp_colImage_card_le_three hp pl.rot

/-- If a board occupies fewer than `cfg.cols` distinct columns, it has no full rows. -/
theorem fullRows_eq_empty_of_colImage_card_lt {cfg : GameConfig} {b : Board}
    (h : (b.image Prod.fst).card < cfg.cols) : Board.fullRows cfg b = ∅ := by
  rcases Finset.eq_empty_or_nonempty (Board.fullRows cfg b) with he | hne
  · exact he
  · exfalso
    obtain ⟨r, hr⟩ := hne
    have hfull : Board.isFull cfg b r := Board.isFull_of_mem_fullRows hr
    have hsub : Finset.range cfg.cols ⊆ b.image Prod.fst := by
      intro c hc
      rw [Finset.mem_range] at hc
      have hmem : (c, r) ∈ b := Board.mem_of_isFull cfg hc hfull
      exact Finset.mem_image.mpr ⟨(c, r), hmem, rfl⟩
    have hcard : cfg.cols ≤ (b.image Prod.fst).card := by
      have hle := Finset.card_le_card hsub
      rwa [Finset.card_range] at hle
    omega

/-- Placing an S/Z/T piece adds at most 3 newly occupied columns. -/
theorem colImage_place_card_le_add_three (b : Board) (pl : Placement)
    (hp : pl.piece = Piece.S ∨ pl.piece = Piece.Z ∨ pl.piece = Piece.T) :
    ((pl.place b).image Prod.fst).card ≤ (b.image Prod.fst).card + 3 := by
  unfold Placement.place
  rw [Finset.image_union]
  have h3 := dropped_colImage_card_le_three pl b hp
  have hu := Finset.card_union_le (b.image Prod.fst) ((pl.dropped b).image Prod.fst)
  omega

/-- **Depth-3 count collapse: `{2,12} → {12}`.** The count=2 branch (a line clear at
move 3) is geometrically impossible: S, Z, T each occupy ≤3 distinct columns, so the
board after init→S→Z→T touches ≤9 < 10 columns and can have no full row. Hence no
clear happens at depth 3 and the board count is exactly 12. This sharpens
`exists_depth_three_count_two_or_twelve_of_init_closed`, eliminating the `count=2`
disjunct. -/
theorem exists_depth_three_count_twelve_of_init_closed
    {T : Finset GameState} (hinit : GameState.init ∈ T)
    (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, g.board.count = 12 := by
  -- move 1: draw S, extract a valid S-placement
  have hSb : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  obtain ⟨_, hstepInit⟩ := hclosed GameState.init hinit
  obtain ⟨plS, hplS, hsuccS⟩ := hstepInit Piece.S hSb
  obtain ⟨hpieceS, hvalidS⟩ :=
    (Placement.mem_allValidFor GameConfig.standard Piece.S plS).mp hplS
  have hvS' : ({plS with piece := Piece.S} : Placement).Valid GameConfig.standard := by
    have : ({plS with piece := Piece.S} : Placement) = plS := by
      rcases plS with ⟨pp, pr, pc⟩; cases hpieceS; rfl
    rw [this]; exact hvalidS
  -- move 2: draw Z, extract a valid Z-placement
  have hZb : Piece.Z ∈
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).bag := by
    rw [adversarialStep_bag, GameState.init_bag, Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.Z⟩
  obtain ⟨_, hstepS⟩ := hclosed _ hsuccS
  obtain ⟨plZ, hplZ, hsuccZ⟩ := hstepS Piece.Z hZb
  obtain ⟨hpieceZ, hvalidZ⟩ :=
    (Placement.mem_allValidFor GameConfig.standard Piece.Z plZ).mp hplZ
  have hvZ' : ({plZ with piece := Piece.Z} : Placement).Valid GameConfig.standard := by
    have : ({plZ with piece := Piece.Z} : Placement) = plZ := by
      rcases plZ with ⟨pp, pr, pc⟩; cases hpieceZ; rfl
    rw [this]; exact hvalidZ
  -- move 3: draw T, extract a valid T-placement
  have hZin : Piece.Z ∈ Bag.full.draw Piece.S := by
    rw [Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.Z⟩
  have h2 : 2 ≤ (Bag.full.draw Piece.S).card := by rw [Bag.draw_full_card]; omega
  have hTin : Piece.T ∈ (Bag.full.draw Piece.S).draw Piece.Z := by
    rw [Bag.draw_eq_erase_of_card_ge_two hZin h2, Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr
      ⟨by decide, Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.T⟩⟩
  have hTb : Piece.T ∈
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).bag := by
    rw [adversarialStep_bag, adversarialStep_bag, GameState.init_bag]; exact hTin
  obtain ⟨_, hstepZ⟩ := hclosed _ hsuccZ
  obtain ⟨plT, hplT, hsuccT⟩ := hstepZ Piece.T hTb
  obtain ⟨hpieceT, hvalidT⟩ :=
    (Placement.mem_allValidFor GameConfig.standard Piece.T plT).mp hplT
  have hvT' : ({plT with piece := Piece.T} : Placement).Valid GameConfig.standard := by
    have : ({plT with piece := Piece.T} : Placement) = plT := by
      rcases plT with ⟨pp, pr, pc⟩; cases hpieceT; rfl
    rw [this]; exact hvalidT
  refine ⟨adversarialStep GameConfig.standard
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ)
      Piece.T plT, hsuccT, ?_⟩
  -- WF chain: ∅ → depth-1 → depth-2 boards
  have hWF0 : Board.WF GameConfig.standard GameState.init.board := by
    rw [GameState.init_board_eq_emptyset]
    intro q hq; exact absurd hq (Finset.notMem_empty q)
  have hWF1 : Board.WF GameConfig.standard
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).board :=
    adversarialStep_board_wf hWF0 hvS'
  have hWF2 : Board.WF GameConfig.standard
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board :=
    adversarialStep_board_wf hWF1 hvZ'
  -- depth-2 board count = 8 (no clear at 8 < 10)
  have hb1count :
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).board.count = 4 :=
    count_succ_init_eq_four hplS
  have hplaceZcount :
      (({plZ with piece := Piece.Z} : Placement).place
        (adversarialStep GameConfig.standard GameState.init Piece.S plS).board).count = 8 := by
    rw [Placement.count_place, hb1count]
  have hb2_eq :
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board
      = ({plZ with piece := Piece.Z} : Placement).place
        (adversarialStep GameConfig.standard GameState.init Piece.S plS).board := by
    rw [adversarialStep_board, Placement.applyStep_eq_clearLines_place]
    apply Board.clearLines_eq_self_of_count_lt
    rw [hplaceZcount]; decide
  -- depth-3 placed board (pre-clear) count = 12, and is WF
  have hplaced3count :
      (({plT with piece := Piece.T} : Placement).place
        (adversarialStep GameConfig.standard
          (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board).count
        = 12 := by
    rw [Placement.count_place, hb2_eq, hplaceZcount]
  have hWFplaced3 : Board.WF GameConfig.standard
      (({plT with piece := Piece.T} : Placement).place
        (adversarialStep GameConfig.standard
          (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board) :=
    Board.WF_place hWF2 hvT'
  -- column-image bounds: ∅(0) → depth-1(≤3) → depth-2(≤6) → depth-3 placed(≤9)
  have hb1_eq :
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).board
      = ({plS with piece := Piece.S} : Placement).place GameState.init.board := by
    rw [adversarialStep_board, Placement.applyStep_eq_clearLines_place]
    apply Board.clearLines_eq_self_of_count_lt
    rw [Placement.count_place, GameState.init_board_eq_emptyset]; decide
  have hcol1 :
      ((adversarialStep GameConfig.standard GameState.init Piece.S plS).board.image
        Prod.fst).card ≤ 3 := by
    rw [hb1_eq]
    have h := colImage_place_card_le_add_three GameState.init.board
      ({plS with piece := Piece.S} : Placement) (Or.inl rfl)
    have hz : (GameState.init.board.image Prod.fst).card = 0 := by
      rw [GameState.init_board_eq_emptyset, Finset.image_empty, Finset.card_empty]
    omega
  have hcol2 :
      ((adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board.image
        Prod.fst).card ≤ 6 := by
    rw [hb2_eq]
    have h := colImage_place_card_le_add_three
      (adversarialStep GameConfig.standard GameState.init Piece.S plS).board
      ({plZ with piece := Piece.Z} : Placement) (Or.inr (Or.inl rfl))
    omega
  have hcol3 :
      ((({plT with piece := Piece.T} : Placement).place
        (adversarialStep GameConfig.standard
          (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board).image
        Prod.fst).card ≤ 9 := by
    have h := colImage_place_card_le_add_three
      (adversarialStep GameConfig.standard
        (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board
      ({plT with piece := Piece.T} : Placement) (Or.inr (Or.inr rfl))
    omega
  -- ≤9 < 10 columns ⇒ no full rows ⇒ no clear at depth 3 ⇒ count = 12
  have hnofull :
      Board.fullRows GameConfig.standard
        (({plT with piece := Piece.T} : Placement).place
          (adversarialStep GameConfig.standard
            (adversarialStep GameConfig.standard GameState.init Piece.S plS) Piece.Z plZ).board) = ∅ := by
    apply fullRows_eq_empty_of_colImage_card_lt
    have hcols : GameConfig.standard.cols = 10 := by decide
    rw [hcols]; omega
  have hadd := Board.clearLines_count_add hWFplaced3
  rw [hplaced3count, hnofull, Finset.card_empty] at hadd
  rw [adversarialStep_board, Placement.applyStep_eq_clearLines_place]
  omega

/-- The filled rows of a column lie within `range (colHeight)`: every filled
cell sits strictly below its column's height. -/
theorem colRows_subset_range_colHeight (b : Board) (j : ℕ) :
    b.colRows j ⊆ Finset.range (b.colHeight j) := by
  intro r hr
  rw [Finset.mem_range]
  have h : r + 1 ≤ b.colHeight j := Finset.le_sup (f := (· + 1)) hr
  omega

/-- `colHoles` as a set-difference cardinality: the empty rows strictly below
the column top. This reformulation turns the `Nat`-subtraction definition into a
genuine `Finset` so hole counts can be compared by subset reasoning. -/
theorem colHoles_eq_sdiff_card (b : Board) (j : ℕ) :
    colHoles b j = ((Finset.range (b.colHeight j)) \ b.colRows j).card := by
  unfold colHoles
  rw [Finset.card_sdiff_of_subset (colRows_subset_range_colHeight b j), Finset.card_range]

/-- **Per-column hole monotonicity under a hard drop.** Dropping a piece never
decreases the buried-cell count of any column. The dropped cells rest at or
above the prior surface (`colHeight_le_of_mem_dropped`), so they never fill a row
that was empty *below* the old top — the empty-below-top set only grows. -/
theorem colHoles_le_colHoles_place (b : Board) (pl : Placement) (j : ℕ) :
    colHoles b j ≤ colHoles (pl.place b) j := by
  rw [colHoles_eq_sdiff_card, colHoles_eq_sdiff_card]
  apply Finset.card_le_card
  intro r hr
  rw [Finset.mem_sdiff, Finset.mem_range] at hr ⊢
  obtain ⟨hlt, hnotin⟩ := hr
  refine ⟨?_, ?_⟩
  · have hsub : b ⊆ pl.place b := by
      rw [Placement.place_eq_union_dropped]; exact Finset.subset_union_left
    exact lt_of_lt_of_le hlt (Board.colHeight_le_of_subset hsub j)
  · intro hrin
    rw [Placement.place_eq_union_dropped, Board.colRows_union, Finset.mem_union] at hrin
    rcases hrin with hb | hd
    · exact hnotin hb
    · have hjr : (j, r) ∈ pl.dropped b := by
        rcases Finset.mem_image.mp hd with ⟨⟨c, r'⟩, hmem', heq⟩
        rcases Finset.mem_filter.mp hmem' with ⟨hcell, hcol⟩
        simp only at hcol heq
        subst hcol
        subst heq
        exact hcell
      have hc : b.colHeight j ≤ r := colHeight_le_of_mem_dropped b pl hjr
      omega

/-- **Total hole monotonicity under a hard drop.** Summing the per-column bound:
a pure placement (no line clear) never decreases the total hole count. This is
the keystone for early-phase hole lower bounds — combined with the depth-3
no-clear result and the S/Z hole-creation lemmas, holes accumulate
monotonically across the opening moves. -/
theorem holes_le_holes_place (cfg : GameConfig) (b : Board) (pl : Placement) :
    holes cfg b ≤ holes cfg (pl.place b) := by
  unfold holes
  exact Finset.sum_le_sum (fun j _ => colHoles_le_colHoles_place b pl j)

/-- **Hole monotonicity lifted to `applyStep` in the no-clear regime.** When the
placed board has no full row, `applyStep` is just the hard drop (`clearLines` is
the identity), so the per-drop hole bound `holes_le_holes_place` carries over. The
no-clear hypothesis is essential: a line clear *removes* a covering row and can
**un-bury** holes, breaking monotonicity. This is the bridge that connects iter-98's
`place`-level result to the adversarial walk (whose steps are `applyStep`). -/
theorem holes_le_holes_applyStep_of_no_fullRows {cfg : GameConfig} (b : Board)
    (pl : Placement) (hno : Board.fullRows cfg (pl.place b) = ∅) :
    holes cfg b ≤ holes cfg (pl.applyStep cfg b) := by
  have hself : pl.applyStep cfg b = pl.place b := by
    rw [Placement.applyStep_eq_clearLines_place]
    exact Board.clearLines_eq_self_of_no_fullRows cfg hno
  rw [hself]
  exact holes_le_holes_place cfg b pl

/-- **Count-driven corollary.** If the placed board stays under `cfg.cols` cells it
cannot have a full row, so holes are non-decreasing across the `applyStep`. This is
the form that applies directly in the opening (depths 1–2: place-count 4, 8 < 10). -/
theorem holes_le_holes_applyStep_of_count_lt {cfg : GameConfig} (b : Board)
    (pl : Placement) (h : (pl.place b).count < cfg.cols) :
    holes cfg b ≤ holes cfg (pl.applyStep cfg b) :=
  holes_le_holes_applyStep_of_no_fullRows b pl
    (Board.fullRows_eq_empty_of_count_lt cfg h)

/-- **Adversarial-walk form.** A single adversary move with no resulting line clear
never decreases the hole count. This is the directly chainable form for the opening
sequence: with the iter-97 no-clear facts in hand, holes accumulate monotonically
along `init → S → Z → …`, so any hole forced early (e.g. the S-on-∅ hole) persists. -/
theorem holes_le_holes_adversarialStep_of_no_fullRows {cfg : GameConfig}
    (g : GameState) (p : Piece) (pl : Placement)
    (hno : Board.fullRows cfg (({pl with piece := p} : Placement).place g.board) = ∅) :
    holes cfg g.board ≤ holes cfg (adversarialStep cfg g p pl).board := by
  rw [adversarialStep_board]
  exact holes_le_holes_applyStep_of_no_fullRows g.board ({pl with piece := p}) hno

/-- The empty-below-top set of a column only grows under a hard drop. This is the
inclusion underlying the per-column hole monotonicity (iter-98); extracted as a
standalone lemma so the **strict** increase below can reuse it as the base subset. -/
theorem sdiff_range_colRows_subset_place (b : Board) (pl : Placement) (j : ℕ) :
    (Finset.range (b.colHeight j)) \ b.colRows j ⊆
      (Finset.range ((pl.place b).colHeight j)) \ (pl.place b).colRows j := by
  intro r hr
  rw [Finset.mem_sdiff, Finset.mem_range] at hr ⊢
  obtain ⟨hlt, hnotin⟩ := hr
  refine ⟨?_, ?_⟩
  · have hsub : b ⊆ pl.place b := by
      rw [Placement.place_eq_union_dropped]; exact Finset.subset_union_left
    exact lt_of_lt_of_le hlt (Board.colHeight_le_of_subset hsub j)
  · intro hrin
    rw [Placement.place_eq_union_dropped, Board.colRows_union, Finset.mem_union] at hrin
    rcases hrin with hb | hd
    · exact hnotin hb
    · have hjr : (j, r) ∈ pl.dropped b := by
        rcases Finset.mem_image.mp hd with ⟨⟨c, r'⟩, hmem', heq⟩
        rcases Finset.mem_filter.mp hmem' with ⟨hcell, hcol⟩
        simp only at hcol heq
        subst hcol
        subst heq
        exact hcell
      have hc : b.colHeight j ≤ r := colHeight_le_of_mem_dropped b pl hjr
      omega

/-- **Strict per-column hole creation.** If a hard drop leaves a row `r₀` empty in
column `j` that sits **at or above the old column top** but **below the new top**
(`b.colHeight j ≤ r₀ < (pl.place b).colHeight j`, with `(j, r₀)` unfilled), then the
column's buried-cell count strictly increases. The new empty-below-top row `r₀`
witnesses the strict inclusion of the empty-below-top sets. This is the engine for
*forcing* an extra hole — the piece raised the surface past `r₀` but never filled it. -/
theorem colHoles_lt_colHoles_place_of_buried {b : Board} {pl : Placement} {j r₀ : ℕ}
    (hge : b.colHeight j ≤ r₀) (hlt : r₀ < (pl.place b).colHeight j)
    (hempty : (j, r₀) ∉ pl.place b) :
    colHoles b j < colHoles (pl.place b) j := by
  rw [colHoles_eq_sdiff_card, colHoles_eq_sdiff_card]
  apply Finset.card_lt_card
  rw [Finset.ssubset_iff_of_subset (sdiff_range_colRows_subset_place b pl j)]
  refine ⟨r₀, ?_, ?_⟩
  · rw [Finset.mem_sdiff, Finset.mem_range]
    refine ⟨hlt, ?_⟩
    intro hr0in
    apply hempty
    rcases Finset.mem_image.mp hr0in with ⟨⟨c, r'⟩, hmem', heq⟩
    rcases Finset.mem_filter.mp hmem' with ⟨hcell, hcol⟩
    simp only at hcol heq
    subst hcol
    subst heq
    exact hcell
  · rw [Finset.mem_sdiff, Finset.mem_range]
    rintro ⟨hr0lt, -⟩
    omega

/-- **Strict total hole creation.** Summing: if any in-range column `j < cfg.cols`
gains a fresh buried row under a hard drop, the total hole count strictly increases.
This is the quantitative companion to `holes_le_holes_place` — the lever that turns
"a piece buries a cell" into "holes go up by at least one". -/
theorem holes_lt_holes_place_of_buried {cfg : GameConfig} {b : Board} {pl : Placement}
    {j r₀ : ℕ} (hj : j < cfg.cols) (hge : b.colHeight j ≤ r₀)
    (hlt : r₀ < (pl.place b).colHeight j) (hempty : (j, r₀) ∉ pl.place b) :
    holes cfg b < holes cfg (pl.place b) := by
  unfold holes
  apply Finset.sum_lt_sum
  · exact fun i _ => colHoles_le_colHoles_place b pl i
  · exact ⟨j, Finset.mem_range.mpr hj, colHoles_lt_colHoles_place_of_buried hge hlt hempty⟩

/-! ### Board max-height monotonicity (height-based Lyapunov infrastructure)

`maxHeight` itself — its empty value, the per-column lower bound
`colHeight_le_maxHeight`, and the `count ≤ cols * maxHeight` volume bounds — now
lives in `Proofs.Combinatorics.BoardCount` as `Board.maxHeight` (reachable here as bare
`maxHeight` via `open Board`). This section keeps the *game-step* height-monotonicity
lemmas (hard drop / no-clear / adversarial step) — the height analogues of the
`holes_le_holes_place` chain feeding the `f := maxHeight + holes` Lyapunov argument. -/

/-- **Height monotonicity under a hard drop.** A hard drop only adds cells, so no
column shrinks and the board max-height cannot decrease. The height companion of
`holes_le_holes_place`. -/
theorem maxHeight_le_maxHeight_place {cfg : GameConfig} (b : Board) (pl : Placement) :
    maxHeight cfg b ≤ maxHeight cfg (pl.place b) := by
  have hsub : b ⊆ pl.place b := by
    rw [Placement.place_eq_union_dropped]; exact Finset.subset_union_left
  unfold maxHeight
  apply Finset.sup_le
  intro j hj
  exact le_trans (Board.colHeight_le_of_subset hsub j) (Finset.le_sup hj)

/-- **Height monotonicity across a no-clear step.** When the placed board has no
full rows, `applyStep` is just the hard drop, so the max-height cannot decrease.
The height companion of `holes_le_holes_applyStep_of_no_fullRows` (iter-99). -/
theorem maxHeight_le_maxHeight_applyStep_of_no_fullRows {cfg : GameConfig} (b : Board)
    (pl : Placement) (hno : Board.fullRows cfg (pl.place b) = ∅) :
    maxHeight cfg b ≤ maxHeight cfg (pl.applyStep cfg b) := by
  have hself : pl.applyStep cfg b = pl.place b := by
    rw [Placement.applyStep_eq_clearLines_place]
    exact Board.clearLines_eq_self_of_no_fullRows cfg hno
  rw [hself]
  exact maxHeight_le_maxHeight_place b pl

/-- Adversarial-step form of no-clear height monotonicity: feeds the height
half of a combined `maxHeight + holes` Lyapunov walk along the bag-forced path. -/
theorem maxHeight_le_maxHeight_adversarialStep_of_no_fullRows {cfg : GameConfig}
    (g : GameState) (p : Piece) (pl : Placement)
    (hno : Board.fullRows cfg (({pl with piece := p} : Placement).place g.board) = ∅) :
    maxHeight cfg g.board ≤ maxHeight cfg (adversarialStep cfg g p pl).board := by
  rw [adversarialStep_board]
  exact maxHeight_le_maxHeight_applyStep_of_no_fullRows g.board ({pl with piece := p}) hno

/-! ### Top-out from volume (the FALSE-route terminal step, iter-103)

`lost cfg g := Board.isLost cfg g.board`, and `¬ g.lost cfg ↔ ∀ p ∈ g.board,
p.2 < cfg.rows` (`GameState.not_lost_iff_forall_row_lt`). So an in-field board has
`maxHeight ≤ rows` (every filled row `< rows` ⇒ every `colHeight = maxRow + 1 ≤
rows`), and contrapositively `rows < maxHeight ⇒ lost`. Chaining the iter-102
volume bound gives the keystone: **a well-formed board holding more than
`cols · rows` cells is a lost state** — the playfield's `cols · rows = 200`
(standard) cell ceiling, formalized. With `count = 4d` along a no-clear opening,
`d ≥ 51` no-clear pieces force a top-out. -/

/-- **Top-out from height.** If the board rises strictly above the field
(`cfg.rows < maxHeight`), the state is lost. Contrapositive of the height cap. -/
theorem lost_of_rows_lt_maxHeight {cfg : GameConfig} {g : GameState}
    (h : cfg.rows < maxHeight cfg g.board) : g.lost cfg := by
  by_contra hcon
  rw [GameState.not_lost_iff_forall_row_lt] at hcon
  have hle := maxHeight_le_rows_of_forall_row_lt hcon
  omega

/-- **Top-out from volume (keystone).** A well-formed board holding more than
`cfg.cols · cfg.rows` cells is lost: the playfield holds at most `cols · rows`
in-field cells, so exceeding that capacity forces a cell above the ceiling.
Combine with the no-clear count ladder (`count = 4d`) to force a top-out. -/
theorem lost_of_cols_mul_rows_lt_count {cfg : GameConfig} {g : GameState}
    (hwf : Board.WF cfg g.board) (h : cfg.cols * cfg.rows < g.board.count) :
    g.lost cfg :=
  lost_of_rows_lt_maxHeight (lt_maxHeight_of_cols_mul_lt_count hwf h)

/-! ### Holes block line clears

A buried empty cell (a hole) makes its row impossible to complete, so that row can
never be a member of `fullRows` and is never cleared while it stays buried. This is
the structural bridge between the hole machinery (the FALSE-route obstruction) and
the clear-count machinery (`count_applyStep_add_fullRows` and friends): holes carve
out permanently non-clearable rows, which is what lets the cell count outrun the
line clears toward the volume top-out (`lost_of_cols_mul_rows_lt_count`). -/

/-- A column with at least one hole exposes a concrete buried empty cell: some row
`r < colHeight j` with `(j, r) ∉ b`. Extracted from the set-difference recast
`colHoles = |range (colHeight j) \ colRows j|` (`colHoles_eq_sdiff_card`). -/
theorem exists_notMem_of_colHoles_pos {b : Board} {j : ℕ}
    (hpos : 0 < colHoles b j) :
    ∃ r, r < b.colHeight j ∧ (j, r) ∉ b := by
  rw [colHoles_eq_sdiff_card] at hpos
  obtain ⟨r, hr⟩ := Finset.card_pos.mp hpos
  rw [Finset.mem_sdiff, Finset.mem_range] at hr
  refine ⟨r, hr.1, ?_⟩
  intro hmem
  apply hr.2
  unfold Board.colRows
  exact Finset.mem_image.mpr ⟨(j, r), Finset.mem_filter.mpr ⟨hmem, rfl⟩, rfl⟩

/-- **Holes block clears.** A column with a hole (`0 < colHoles b j`, `j < cfg.cols`)
carries a buried row `r < colHeight j` that is *not* a full row, hence never cleared
while buried. The per-column witness that accumulated holes pin down non-clearable
rows beneath the surface — the mechanism by which clears cannot keep the count
bounded once holes build up. -/
theorem exists_lt_colHeight_notMem_fullRows_of_colHoles_pos {cfg : GameConfig}
    {b : Board} {j : ℕ} (hj : j < cfg.cols) (hpos : 0 < colHoles b j) :
    ∃ r, r < b.colHeight j ∧ r ∉ Board.fullRows cfg b := by
  obtain ⟨r, hlt, hnotmem⟩ := exists_notMem_of_colHoles_pos hpos
  exact ⟨r, hlt, notMem_fullRows_of_notMem hj hnotmem⟩

/-! ### Clears per step are throttled by the stack height (and strictly by holes)

A full row holds a cell in every in-range column, so its row index lies below the
stack height: `fullRows ⊆ range (maxHeight)`, hence at most `maxHeight` rows clear
in a single move. Sharpening with the holes bridge above: if any column carries a
hole, the buried non-clearable row is *inside* `range (maxHeight)` yet outside
`fullRows`, so strictly fewer than `maxHeight` rows clear. This is the aggregate
form of "holes block clears" — it caps the `cfg.cols * #fullRows` subtraction in the
per-move count law (`count_applyStep_add_fullRows`) by the height, and shaves it
further for every hole, the lever that keeps the count climbing toward the volume
top-out (`lost_of_cols_mul_rows_lt_count`). -/

/-- **Holes strictly throttle clears.** If some in-range column carries a hole, the
buried non-clearable row lies in `range (maxHeight)` but outside `fullRows`, so the
number of full rows is *strictly* below the height: `#fullRows < maxHeight`. -/
theorem fullRows_card_lt_maxHeight_of_colHoles_pos {cfg : GameConfig} {b : Board}
    {j : ℕ} (hcols : 0 < cfg.cols) (hj : j < cfg.cols) (hpos : 0 < colHoles b j) :
    (Board.fullRows cfg b).card < maxHeight cfg b := by
  obtain ⟨r₀, hr₀lt, hr₀notmem⟩ :=
    exists_lt_colHeight_notMem_fullRows_of_colHoles_pos hj hpos
  have hr₀max : r₀ < maxHeight cfg b :=
    lt_of_lt_of_le hr₀lt (colHeight_le_maxHeight hj)
  have hsub : Board.fullRows cfg b ⊆ (Finset.range (maxHeight cfg b)).erase r₀ := by
    intro r hr
    rw [Finset.mem_erase]
    refine ⟨?_, fullRows_subset_range_maxHeight hcols hr⟩
    intro heq
    exact hr₀notmem (heq ▸ hr)
  calc (Board.fullRows cfg b).card
      ≤ ((Finset.range (maxHeight cfg b)).erase r₀).card := Finset.card_le_card hsub
    _ = maxHeight cfg b - 1 := by
        rw [Finset.card_erase_of_mem (Finset.mem_range.mpr hr₀max), Finset.card_range]
    _ < maxHeight cfg b := by
        have : 0 < maxHeight cfg b := lt_of_le_of_lt (Nat.zero_le r₀) hr₀max
        omega

/-! ### Terminal top-out on a non-clearing move from a nearly-full board (iter-106)

The count machinery and the volume top-out join here. The per-move count law
(`count_applyStep_add_fullRows`) collapses, on a *no-clear* step
(`fullRows (place) = ∅`), to the exact recurrence `count' = count + 4`: a move that
completes no rows simply deposits its four cells. Chaining this with the volume
keystone (`lost_of_cols_mul_rows_lt_count`, a `> cols·rows` board is lost) yields the
terminal lever of the FALSE narrative: once the stack has climbed to within four
cells of the `cols·rows` ceiling (`≥ 197` on the standard field), *any* placement
that fails to clear a line tops the board out. Holes are precisely what force such
non-clearing moves (`fullRows_card_lt_maxHeight_of_colHoles_pos`,
`notMem_fullRows_of_notMem`), so an accumulating-holes board is driven into this
terminal régime. -/

/-- **No-clear count law (exact).** A placement completing no rows deposits exactly
its four cells: the clear step is the identity, so `count' = count + 4`. -/
theorem count_applyStep_eq_of_no_fullRows {cfg : GameConfig} {b : Board} {pl : Placement}
    (hno : Board.fullRows cfg (pl.place b) = ∅) :
    (pl.applyStep cfg b).count = b.count + 4 := by
  show (Board.clearLines cfg (pl.place b)).count = b.count + 4
  rw [Board.count_clearLines_eq_self_of_no_fullRows cfg hno, Placement.count_place]

/-- Adversarial form of the no-clear count law. -/
theorem count_adversarialStep_eq_of_no_fullRows {cfg : GameConfig}
    (g : GameState) (p : Piece) (pl : Placement)
    (hno : Board.fullRows cfg (({pl with piece := p} : Placement).place g.board) = ∅) :
    (adversarialStep cfg g p pl).board.count = g.board.count + 4 := by
  rw [adversarialStep_board]
  exact count_applyStep_eq_of_no_fullRows hno

/-- **Terminal top-out.** From a well-formed board within four cells of the
`cfg.cols · cfg.rows` ceiling, a valid placement that clears no line lands a lost
state: the no-clear law pushes the count strictly past the volume capacity. -/
theorem lost_adversarialStep_of_no_fullRows_of_count_lt {cfg : GameConfig}
    (g : GameState) (p : Piece) (pl : Placement)
    (hwf : Board.WF cfg g.board)
    (hv : ({pl with piece := p} : Placement).Valid cfg)
    (hno : Board.fullRows cfg (({pl with piece := p} : Placement).place g.board) = ∅)
    (hcount : cfg.cols * cfg.rows < g.board.count + 4) :
    (adversarialStep cfg g p pl).lost cfg := by
  have hwf' : Board.WF cfg (adversarialStep cfg g p pl).board := by
    rw [adversarialStep_board]
    show Board.WF cfg (Board.clearLines cfg (({pl with piece := p} : Placement).place g.board))
    exact Board.WF_clearLines (Board.WF_place hwf hv)
  have hcnt : (adversarialStep cfg g p pl).board.count = g.board.count + 4 :=
    count_adversarialStep_eq_of_no_fullRows g p pl hno
  exact lost_of_cols_mul_rows_lt_count hwf' (by rw [hcnt]; exact hcount)

/-- **Terminal top-out on the standard field.** A non-clearing valid move from any
board holding `≥ 197` cells tops out: `197 + 4 = 201 > 200 = 10 · 20`. The concrete
FALSE-route endgame — a hole-throttled stack that reaches `197` cells has no escape
on a move that completes no line. -/
theorem lost_adversarialStep_of_no_fullRows_of_count_ge
    (g : GameState) (p : Piece) (pl : Placement)
    (hwf : Board.WF GameConfig.standard g.board)
    (hv : ({pl with piece := p} : Placement).Valid GameConfig.standard)
    (hno : Board.fullRows GameConfig.standard
      (({pl with piece := p} : Placement).place g.board) = ∅)
    (hcount : 197 ≤ g.board.count) :
    (adversarialStep GameConfig.standard g p pl).lost GameConfig.standard := by
  apply lost_adversarialStep_of_no_fullRows_of_count_lt g p pl hwf hv hno
  rw [GameConfig.standard_cols_mul_rows]
  omega

/-! ### Holes top-out: the buried-cell dual of the volume keystone (iter-107)

The conservation law `count + holes = ∑ colHeight` (`count_add_holes_eq_sum_colHeight`,
iter-73) pairs with the `maxHeight` sup bound to expose the *true* survival budget.
Each column height is `≤ maxHeight`, so `∑ colHeight ≤ cols · maxHeight`, hence
`count + holes ≤ cols · maxHeight`: **filled cells and buried holes together fit
inside the `cols × maxHeight` envelope**. On a non-lost board `maxHeight ≤ rows`, so
`count + holes ≤ cols · rows = 200` — every hole permanently consumes one of the 200
playfield cells that could otherwise hold the stack. Contrapositively, once
`count + holes` exceeds `cols · rows` the board is lost. This is the FALSE-route
keystone on the *holes* axis, dual to `lost_of_cols_mul_rows_lt_count` (iter-103,
the count axis): the S/Z hole-forcing (`holes_lt_holes_place_of_buried`,
`one_le_holes_S_place_empty`) drives `holes` up, and that growth is now a direct
top-out lever, not merely a clear-throttle. -/

/-- The column-height total is dominated by `cols · maxHeight`: every one of the
`cols` columns is no taller than the board max-height. -/
theorem sum_colHeight_le_cols_mul_maxHeight {cfg : GameConfig} (b : Board) :
    (∑ j ∈ Finset.range cfg.cols, b.colHeight j) ≤ cfg.cols * maxHeight cfg b := by
  calc (∑ j ∈ Finset.range cfg.cols, b.colHeight j)
      ≤ ∑ _j ∈ Finset.range cfg.cols, maxHeight cfg b :=
        Finset.sum_le_sum (fun j hj => colHeight_le_maxHeight (Finset.mem_range.mp hj))
    _ = cfg.cols * maxHeight cfg b := by
        rw [Finset.sum_const, Finset.card_range, smul_eq_mul]

/-- **Survival budget.** Filled cells plus buried holes fit inside the
`cols × maxHeight` envelope: `count + holes ≤ cols · maxHeight`. The conservation
law `count + holes = ∑ colHeight` capped by the height sup. -/
theorem count_add_holes_le_cols_mul_maxHeight {cfg : GameConfig} {b : Board}
    (hWF : Board.ColWF cfg.cols b) :
    b.count + holes cfg b ≤ cfg.cols * maxHeight cfg b := by
  rw [count_add_holes_eq_sum_colHeight hWF]
  exact sum_colHeight_le_cols_mul_maxHeight b

/-- **Top-out from the combined budget (keystone).** A well-formed board whose cells
*and* holes together exceed `cfg.cols · cfg.rows` is lost: the survival budget caps
`count + holes` at `cols · maxHeight`, so overflowing `cols · rows` forces
`rows < maxHeight`. The sharp FALSE-route resource bound — every buried hole counts
against the `200`-cell ceiling exactly as a filled cell does. -/
theorem lost_of_cols_mul_rows_lt_count_add_holes {cfg : GameConfig} {g : GameState}
    (hWF : Board.WF cfg g.board)
    (h : cfg.cols * cfg.rows < g.board.count + holes cfg g.board) :
    g.lost cfg := by
  have hle : g.board.count + holes cfg g.board ≤ cfg.cols * maxHeight cfg g.board :=
    count_add_holes_le_cols_mul_maxHeight (Board.colWF_iff_wf.mpr hWF)
  have hlt : cfg.cols * cfg.rows < cfg.cols * maxHeight cfg g.board := lt_of_lt_of_le h hle
  have hrows : cfg.rows < maxHeight cfg g.board := by
    by_contra hcon
    push_neg at hcon
    have hmul : cfg.cols * maxHeight cfg g.board ≤ cfg.cols * cfg.rows :=
      Nat.mul_le_mul (le_refl cfg.cols) hcon
    omega
  exact lost_of_rows_lt_maxHeight hrows

/-- **Top-out from holes alone (the iter-103 dual).** A well-formed board carrying
more than `cfg.cols · cfg.rows` buried holes is lost — holes alone can exhaust the
playfield's cell ceiling. Corollary of the combined budget via `holes ≤ count + holes`. -/
theorem lost_of_cols_mul_rows_lt_holes {cfg : GameConfig} {g : GameState}
    (hWF : Board.WF cfg g.board)
    (h : cfg.cols * cfg.rows < holes cfg g.board) :
    g.lost cfg :=
  lost_of_cols_mul_rows_lt_count_add_holes hWF
    (lt_of_lt_of_le h (Nat.le_add_left _ _))

/-! ### The count dichotomy: clears are the *only* brake on cell growth

Iteration 106 pinned the no-clear branch (`count' = count + 4`). These lemmas pin the
complementary branch: a *clearing* move (at least one full row in the placed board)
*strictly* lowers the count — by at least `cols - 4` cells in general, and by at least
`6` on the standard `10`-wide field. Packaged as a dichotomy this says the adversary's
only way to keep the count from marching `+4` toward the `200`-cell ceiling is to force
a line clear, and every standard-field clear pays a net `≥ 6` cells. A verdict-relevant
brake law: between the `+4` growth (iter-106) and the volume top-out (iter-103/106),
survival *requires* a steady supply of line clears. -/

/-- **A clearing move strictly lowers the count.** If the placed board has at least one
full row, then on any field at least `5` wide the post-clear count drops below the
starting count: the hard-drop's `+4` cannot offset removing a full `cols ≥ 5` row. -/
theorem count_applyStep_lt_of_fullRows_pos {cfg : GameConfig} {b : Board} {pl : Placement}
    (hb : Board.WF cfg b) (hv : pl.Valid cfg) (hcols : 5 ≤ cfg.cols)
    (hk : 1 ≤ (Board.fullRows cfg (pl.place b)).card) :
    (pl.applyStep cfg b).count < b.count := by
  have hlaw := count_applyStep_add_fullRows hb hv
  have hge : cfg.cols * 1 ≤ cfg.cols * (Board.fullRows cfg (pl.place b)).card :=
    Nat.mul_le_mul (le_refl cfg.cols) hk
  rw [Nat.mul_one] at hge
  omega

/-- **Standard-field clear cost.** On the canonical `10`-wide field a clearing move drops
the count by at least `6`: `count' + 6 ≤ count`. Sharpens
`count_applyStep_lt_of_fullRows_pos` with the exact `cols = 10` deficit. -/
theorem count_applyStep_add_six_le_of_fullRows_pos {b : Board} {pl : Placement}
    (hb : Board.WF GameConfig.standard b) (hv : pl.Valid GameConfig.standard)
    (hk : 1 ≤ (Board.fullRows GameConfig.standard (pl.place b)).card) :
    (pl.applyStep GameConfig.standard b).count + 6 ≤ b.count := by
  have hlaw := count_applyStep_add_fullRows hb hv
  have hge : GameConfig.standard.cols * 1 ≤
      GameConfig.standard.cols * (Board.fullRows GameConfig.standard (pl.place b)).card :=
    Nat.mul_le_mul (le_refl GameConfig.standard.cols) hk
  rw [Nat.mul_one] at hge
  have hc : GameConfig.standard.cols = 10 := GameConfig.standard_cols
  omega

/-- **The per-move count dichotomy (standard field).** Every move either grows the count
by exactly `4` (no line cleared, iter-106) or shrinks it by at least `6` (a line cleared).
There is no middle ground: the count never merely drifts by a small amount. -/
theorem count_applyStep_standard_dichotomy {b : Board} {pl : Placement}
    (hb : Board.WF GameConfig.standard b) (hv : pl.Valid GameConfig.standard) :
    (pl.applyStep GameConfig.standard b).count = b.count + 4
      ∨ (pl.applyStep GameConfig.standard b).count + 6 ≤ b.count := by
  by_cases hk : (Board.fullRows GameConfig.standard (pl.place b)).card = 0
  · exact Or.inl (count_applyStep_eq_of_no_fullRows (Finset.card_eq_zero.mp hk))
  · exact Or.inr
      (count_applyStep_add_six_le_of_fullRows_pos hb hv (Nat.one_le_iff_ne_zero.mpr hk))

/-! ### The stack-volume monovariant: `count + holes` rises on every no-clear step

The conservation law `count + holes = ∑ colHeight` (iter-73) reads the combined
quantity `Φ := count + holes` as the *total stack volume* — every filled cell and
every buried hole counts once toward the `cols · rows = 200` ceiling (iter-107's
survival budget caps `Φ ≤ cols · maxHeight`, and `Φ > 200 ⇒ lost`). Here we pin its
*dynamics*: a no-clear adversarial step raises `Φ` by at least `4` — the hard-drop's
four fresh cells land (count `+4`, iter-106) while holes never fall on a no-clear
step (iter-99). So `Φ` is a strict anti-Lyapunov monovariant on the no-clear
sub-dynamics: bounded above by `200` on non-lost states yet forced up `≥ 4` each
non-clearing move, it admits at most `⌊200/4⌋ = 50` consecutive no-clear steps before
top-out. The defender's *only* escape from this volume ratchet is to clear lines —
which the hole-throttle lemmas (iter-104/105) cap and, when holes are present, forbid
on the buried rows. -/

/-- **Stack-volume ratchet (no-clear step).** A valid move clearing no line raises the
combined cell-plus-hole volume `count + holes` by at least `4`: the four dropped cells
are retained and holes cannot decrease without a clear. The quantitative monovariant
underlying the no-clear-run length bound. -/
theorem count_add_holes_add_four_le_adversarialStep_of_no_fullRows {cfg : GameConfig}
    (g : GameState) (p : Piece) (pl : Placement)
    (hno : Board.fullRows cfg (({pl with piece := p} : Placement).place g.board) = ∅) :
    g.board.count + holes cfg g.board + 4
      ≤ (adversarialStep cfg g p pl).board.count
        + holes cfg (adversarialStep cfg g p pl).board := by
  have hcount := count_adversarialStep_eq_of_no_fullRows g p pl hno
  have hholes := holes_le_holes_adversarialStep_of_no_fullRows g p pl hno
  omega

/-- **Terminal top-out on the combined volume (standard field).** A non-clearing valid
move from any board whose cells *and* holes already total `≥ 197` tops out:
`197 + 4 = 201 > 200 = 10 · 20`. The combined-volume dual of
`lost_adversarialStep_of_no_fullRows_of_count_ge` — buried holes count against the
ceiling exactly as filled cells do, so a hole-laden stack tops out sooner. -/
theorem lost_adversarialStep_of_no_fullRows_of_count_add_holes_ge
    (g : GameState) (p : Piece) (pl : Placement)
    (hwf : Board.WF GameConfig.standard g.board)
    (hv : ({pl with piece := p} : Placement).Valid GameConfig.standard)
    (hno : Board.fullRows GameConfig.standard
      (({pl with piece := p} : Placement).place g.board) = ∅)
    (hge : 197 ≤ g.board.count + holes GameConfig.standard g.board) :
    (adversarialStep GameConfig.standard g p pl).lost GameConfig.standard := by
  have hwf' : Board.WF GameConfig.standard
      (adversarialStep GameConfig.standard g p pl).board := by
    rw [adversarialStep_board]
    show Board.WF GameConfig.standard
      (Board.clearLines GameConfig.standard (({pl with piece := p} : Placement).place g.board))
    exact Board.WF_clearLines (Board.WF_place hwf hv)
  have hquant := count_add_holes_add_four_le_adversarialStep_of_no_fullRows g p pl hno
  apply lost_of_cols_mul_rows_lt_count_add_holes hwf'
  rw [GameConfig.standard_cols_mul_rows]
  omega

/-! ### Per-move brake capacity: a single move sheds at most `cols · maxHeight` cells

The exact count ledger `count' + cols · |fullRows(place)| = count + 4` (backbone) says
the only way a move *lowers* the count is by clearing rows, each clear shedding `cols`
cells. Iteration 105 caps the clears at `|fullRows(place)| ≤ maxHeight(place)`. Composing
the two bounds the *per-move count reduction* with no small-count hypothesis: a move can
drop the count by at most `cols · maxHeight(place) − 4`. The structural reading — the
defender's only brake against the `+4` cell inflow is line clearing, and its capacity is
metered by the current stack height. A short stack simply cannot shed many cells: with
`maxHeight(place) ≤ h` a single move sheds at most `cols · h − 4` (e.g. `h = 1 ⇒ ≤ 6`).
This is the height-metered companion to the no-clear law (iter-106) and the count
dichotomy (iter-108). -/

/-- **Per-move count lower bound (general).** With no restriction on the board count, a
valid move cannot push the count below `count + 4 − cols · maxHeight(place)`: equivalently
`count + 4 ≤ count' + cols · maxHeight(place)`. The single-move shed is height-metered —
clearing `≤ maxHeight(place)` rows removes `≤ cols · maxHeight(place)` cells against the
`+4` hard-drop inflow. -/
theorem count_add_four_le_count_applyStep_add_cols_mul_maxHeight {cfg : GameConfig}
    {b : Board} {pl : Placement} (hb : Board.WF cfg b) (hv : pl.Valid cfg)
    (hcols : 0 < cfg.cols) :
    b.count + 4
      ≤ (pl.applyStep cfg b).count + cfg.cols * maxHeight cfg (pl.place b) := by
  have hlaw := count_applyStep_add_fullRows hb hv
  have hk := fullRows_card_le_maxHeight (cfg := cfg) (b := pl.place b) hcols
  have hmul : cfg.cols * (Board.fullRows cfg (pl.place b)).card
      ≤ cfg.cols * maxHeight cfg (pl.place b) :=
    Nat.mul_le_mul (le_refl cfg.cols) hk
  omega

/-- **Per-move count lower bound (adversarial form).** The `adversarialStep` packaging of
`count_add_four_le_count_applyStep_add_cols_mul_maxHeight`, ready for trajectory reasoning:
the adversary's piece lands four cells and the defender can shed at most
`cols · maxHeight` against them. -/
theorem count_add_four_le_count_adversarialStep_add_cols_mul_maxHeight {cfg : GameConfig}
    (g : GameState) (p : Piece) (pl : Placement)
    (hwf : Board.WF cfg g.board) (hv : ({pl with piece := p} : Placement).Valid cfg)
    (hcols : 0 < cfg.cols) :
    g.board.count + 4
      ≤ (adversarialStep cfg g p pl).board.count
        + cfg.cols * maxHeight cfg (({pl with piece := p} : Placement).place g.board) := by
  rw [adversarialStep_board]
  exact count_add_four_le_count_applyStep_add_cols_mul_maxHeight hwf hv hcols

/-! ### Trajectory-level forced clears (count aggregation)

The per-step count laws aggregate along a whole `ReachableAt` trajectory through the
conservation identity `exists_count_eq` (`count + cols · k = 4 · n`, where `k` is the
total number of rows cleared across all `n` pieces). Pairing that with the volume
top-out (`lost_of_cols_mul_rows_lt_count`: a surviving board holds at most
`cols · rows = 200` cells) converts depth `n` into a *lower bound on total clears*:
surviving `n` pieces forces `4 · n ≤ 200 + 10 · k`. This is the first brick that
constrains the trajectory as a whole rather than a single step — the defender's line
clears must keep pace with the relentless `+4`-per-piece inflow, and a clear-free run
cannot reach depth 51. -/

/-- **Survival forces clears.** A standard-board state reached at depth `n` and not yet
lost has, across its trajectory, cleared `k` rows with `count + 10 · k = 4 · n`
(conservation) and `4 · n ≤ 200 + 10 · k` (from the survival budget `count ≤ 200`). The
second inequality is the forced-clear lower bound `k ≥ (4 · n − 200) / 10`. -/
theorem survival_forces_clears {n : ℕ} {g : GameState}
    (h : ReachableAt GameConfig.standard n g)
    (hnl : ¬ g.lost GameConfig.standard) :
    ∃ k : ℕ, g.board.count + 10 * k = 4 * n ∧ 4 * n ≤ 200 + 10 * k := by
  obtain ⟨k, hk⟩ := h.exists_count_eq
  have hcols : GameConfig.standard.cols = 10 := rfl
  rw [hcols] at hk
  have hcount_le : g.board.count ≤ 200 := by
    by_contra hcon
    push_neg at hcon
    have hprod : GameConfig.standard.cols * GameConfig.standard.rows = 200 := rfl
    exact hnl (lost_of_cols_mul_rows_lt_count h.wf (by rw [hprod]; exact hcon))
  exact ⟨k, hk, by omega⟩

/-- **Clear-free survival ceiling.** If no row has ever been cleared along the trajectory
— so the board count accounts for every piece, `count = 4 · n` — then the depth is capped
at `n ≤ 50`: a clear-free game tops out by piece 51, since 204 cells cannot fit under the
200-cell ceiling. -/
theorem depth_le_fifty_of_count_eq_four_mul {n : ℕ} {g : GameState}
    (h : ReachableAt GameConfig.standard n g)
    (hnl : ¬ g.lost GameConfig.standard)
    (hcount : g.board.count = 4 * n) :
    n ≤ 50 := by
  obtain ⟨k, hcounteq, hle⟩ := survival_forces_clears h hnl
  omega

/-! ### Clears-per-step bound (clear-rate ceiling)

The forced-clear *lower* bound (`survival_forces_clears`: depth `n` demands
`k ≥ (4·n − 200)/10` cleared rows) needs an opposing *upper* bound on how fast the
defender can clear. The structural ceiling: a single placement drops only four cells, so
it can newly complete at most four rows. Any row full after the placement but not full
before must contain one of the four dropped cells, so the newly-completed rows inject
into the rows of `place b \ b` (a four-cell set). This caps the per-step clear rate
independently of board height — the first *upper* bound on clears, opposing the forced
lower bound. -/

/-- **Clears-per-step bound (general).** A single placement completes at most four new
rows beyond those already full: `|fullRows (place b)| ≤ |fullRows b| + 4`. Each newly full
row must contain one of the four freshly dropped cells, so the new full rows inject into
the (four-element) row image of `place b \ b`. -/
theorem fullRows_place_card_le {cfg : GameConfig} (b : Board) (pl : Placement)
    (hcols : 0 < cfg.cols) :
    (Board.fullRows cfg (pl.place b)).card ≤ (Board.fullRows cfg b).card + 4 := by
  have hbsub : b ⊆ pl.place b := by
    unfold Placement.place; exact Finset.subset_union_left
  have hsub : Board.fullRows cfg (pl.place b) \ Board.fullRows cfg b
      ⊆ (pl.place b \ b).image (·.2) := by
    intro r hr
    rw [Finset.mem_sdiff] at hr
    obtain ⟨hrin, hrout⟩ := hr
    have hfull : Board.isFull cfg (pl.place b) r := Board.isFull_of_mem_fullRows hrin
    have hnotfull : ¬ Board.isFull cfg b r := by
      intro hfb
      apply hrout
      rw [Board.mem_fullRows_iff]
      exact ⟨Finset.mem_image.mpr ⟨(0, r), Board.mem_of_isFull cfg hcols hfb, rfl⟩, hfb⟩
    obtain ⟨c, hcr, hcnotin⟩ := (Board.not_isFull_iff_exists_missing cfg b r).mp hnotfull
    have hcin : (c, r) ∈ pl.place b :=
      Board.mem_of_isFull cfg (Finset.mem_range.mp hcr) hfull
    exact Finset.mem_image.mpr ⟨(c, r), Finset.mem_sdiff.mpr ⟨hcin, hcnotin⟩, rfl⟩
  have hcard4 : (pl.place b \ b).card = 4 := by
    have hcp := Placement.count_place b pl
    unfold Board.count at hcp
    have hinter : b ∩ pl.place b = b := Finset.inter_eq_left.mpr hbsub
    rw [Finset.card_sdiff, hinter]
    omega
  have h1 : (Board.fullRows cfg (pl.place b) \ Board.fullRows cfg b).card
      ≤ (pl.place b \ b).card :=
    le_trans (Finset.card_le_card hsub) Finset.card_image_le
  have h2 : (Board.fullRows cfg (pl.place b)).card
      ≤ (Board.fullRows cfg (pl.place b) \ Board.fullRows cfg b).card
        + (Board.fullRows cfg b).card := by
    refine le_trans (Finset.card_le_card ?_) (Finset.card_union_le _ _)
    intro x hx
    rw [Finset.mem_union, Finset.mem_sdiff]
    by_cases hxb : x ∈ Board.fullRows cfg b
    · exact Or.inr hxb
    · exact Or.inl ⟨hx, hxb⟩
  omega

/-- **Clears-per-step bound on a settled board.** When `b` carries no full rows — the
invariant held by every board already passed through `clearLines` — a single move clears
at most four rows: `|fullRows (place b)| ≤ 4`. -/
theorem fullRows_place_card_le_four_of_no_fullRows {cfg : GameConfig} (b : Board)
    (pl : Placement) (hcols : 0 < cfg.cols) (hb : Board.fullRows cfg b = ∅) :
    (Board.fullRows cfg (pl.place b)).card ≤ 4 := by
  have h := fullRows_place_card_le b pl hcols
  rw [hb] at h
  simpa using h

/-! ### Iteration 113 — no-full-rows invariant ⇒ constant per-step clear cap

`clearLines` always lands on a board with **no full rows** (`clearLines_no_full`,
packaged here as `fullRows = ∅`). Hence the board entering any non-initial step
already has no full rows, so by the iter-112 cap each piece completes at most `4`
rows. That turns the height-dependent per-step drain bound (iter-110) into a
*constant* one: a single placement sheds at most `cfg.cols * 4` cells, regardless
of how tall the stack is. This pins the defender's maximal line-clear drain rate
to a constant — a key brick for the FALSE-route trajectory aggregate. -/

/-- **No-full-rows invariant.** After any line clear the resulting board has no
full rows at all: gravity is injective on surviving rows, so it never recombines
two partial rows into a full one. Packages the pointwise `clearLines_no_full`
as a `fullRows = ∅` statement. -/
theorem fullRows_clearLines_eq_empty {cfg : GameConfig} (b : Board)
    (hcol : 0 < cfg.cols) :
    Board.fullRows cfg (Board.clearLines cfg b) = ∅ := by
  apply Finset.eq_empty_of_forall_notMem
  intro r hr
  exact Board.clearLines_no_full b hcol r (Board.isFull_of_mem_fullRows hr)

/-- The board produced by *any* piece placement (`applyStep` ends in a clear) has
no full rows. -/
theorem fullRows_applyStep_eq_empty {cfg : GameConfig} (b : Board) (pl : Placement)
    (hcol : 0 < cfg.cols) :
    Board.fullRows cfg (pl.applyStep cfg b) = ∅ := by
  rw [Placement.applyStep_eq_clearLines_place]
  exact fullRows_clearLines_eq_empty (pl.place b) hcol

/-- **Per-step clear cap along a trajectory.** Once a board has been produced by a
clear (`pl.applyStep`), the *next* placement completes at most `4` rows — the four
freshly-dropped cells are the only cells that can turn a previously non-full row
full. Deploys `fullRows_place_card_le_four_of_no_fullRows` on the no-full-rows
board `pl.applyStep cfg b`. -/
theorem fullRows_place_applyStep_card_le_four {cfg : GameConfig} (b : Board)
    (pl pl' : Placement) (hcol : 0 < cfg.cols) :
    (Board.fullRows cfg (pl'.place (pl.applyStep cfg b))).card ≤ 4 :=
  fullRows_place_card_le_four_of_no_fullRows (pl.applyStep cfg b) pl' hcol
    (fullRows_applyStep_eq_empty b pl hcol)

/-- **Constant per-step count lower bound.** On a board with no full rows a single
placement sheds at most `cfg.cols * 4` cells: `count + 4 ≤ count' + cfg.cols * 4`.
This sharpens the height-dependent
`count_add_four_le_count_applyStep_add_cols_mul_maxHeight` (iter-110) to a
*constant* drain cap independent of stack height, because the clear cap bounds the
number of cleared rows by `4` rather than by `maxHeight`. -/
theorem count_add_four_le_count_applyStep_add_cols_mul_four_of_no_fullRows
    {cfg : GameConfig} (b : Board) (pl : Placement) (hcol : 0 < cfg.cols)
    (hwf : Board.WF cfg b) (hv : pl.Valid cfg) (hb : Board.fullRows cfg b = ∅) :
    b.count + 4 ≤ (pl.applyStep cfg b).count + cfg.cols * 4 := by
  have hcp := Placement.count_place b pl
  have hlaw := Board.clearLines_count_add (Board.WF_place hwf hv)
  have hcap := fullRows_place_card_le_four_of_no_fullRows b pl hcol hb
  have hmul : cfg.cols * (Board.fullRows cfg (pl.place b)).card ≤ cfg.cols * 4 :=
    Nat.mul_le_mul (le_refl cfg.cols) hcap
  rw [Placement.applyStep_eq_clearLines_place]
  omega

/-! ### Iteration 114 — the forward survival budget for non-lost / reachable states

The keystone `lost_of_cols_mul_rows_lt_count_add_holes` (iter-107) is stated
contrapositively: `count + holes > cols · rows ⇒ lost`. Trajectory aggregation,
however, threads the *non-lost* hypothesis forward and needs the budget in positive
form: **every non-lost board satisfies `count + holes ≤ cols · rows`**. We package
that here and specialise it to reachable states — where well-formedness comes for
free from `ReachableAt.wf` — giving the directly chainable invariant
`count + holes ≤ 200` that holds at every surviving step of a standard game. (The
no-clear Φ ratchet `Φ' ≥ Φ + 4` and its `Φ ≥ 197 ⇒ lost` corollary are already in
place, iters 107/113; this is their forward companion.) We also fill the missing
*placement* form of the no-clear ratchet (only the adversarial form existed). -/

/-- **Forward survival budget.** A well-formed, non-lost board keeps its combined
cell-plus-hole volume within the playfield: `count + holes ≤ cols · rows`. The
positive form of the iter-107 keystone — the invariant a surviving trajectory
carries forward, rather than the loss it triggers when violated. -/
theorem count_add_holes_le_cols_mul_rows_of_not_lost {cfg : GameConfig} {g : GameState}
    (hwf : Board.WF cfg g.board) (hnl : ¬ g.lost cfg) :
    g.board.count + holes cfg g.board ≤ cfg.cols * cfg.rows := by
  by_contra hcon
  push_neg at hcon
  exact hnl (lost_of_cols_mul_rows_lt_count_add_holes hwf hcon)

/-- **Chainable trajectory budget.** Every non-lost reachable standard state obeys
`count + holes ≤ 200`; well-formedness is discharged automatically by
`ReachableAt.wf`. The ready-to-use invariant for induction over a surviving game: at
every depth the filled cells and buried holes together fit the `10 × 20` field. -/
theorem count_add_holes_le_two_hundred_of_reachableAt {n : ℕ} {g : GameState}
    (h : ReachableAt GameConfig.standard n g) (hnl : ¬ g.lost GameConfig.standard) :
    g.board.count + holes GameConfig.standard g.board ≤ 200 := by
  have hbudget := count_add_holes_le_cols_mul_rows_of_not_lost h.wf hnl
  rwa [GameConfig.standard_cols_mul_rows] at hbudget

/-- **Placement form of the no-clear Φ ratchet.** A line-clear-free placement raises
the survival budget `count + holes` by at least `4`:
`count' + holes' ≥ count + holes + 4`. The placement-level companion to the
adversarial `count_add_holes_add_four_le_adversarialStep_of_no_fullRows` — the
no-clear count law (`count_applyStep_eq_of_no_fullRows`, iter-106) plus hole
monotonicity (`holes_le_holes_applyStep_of_no_fullRows`, iter-98). -/
theorem count_add_holes_applyStep_ge_of_no_fullRows {cfg : GameConfig} (b : Board)
    (pl : Placement) (hno : Board.fullRows cfg (pl.place b) = ∅) :
    b.count + holes cfg b + 4
      ≤ (pl.applyStep cfg b).count + holes cfg (pl.applyStep cfg b) := by
  have hc := count_applyStep_eq_of_no_fullRows (cfg := cfg) (b := b) (pl := pl) hno
  have hh := holes_le_holes_applyStep_of_no_fullRows b pl hno
  omega

/-! ### Iteration 115 — the no-full-rows invariant, deployed along every trajectory

Iter-113 proved the *settled-board* fact `fullRows (clearLines b) = ∅` and its
`applyStep` form `fullRows (pl.applyStep b) = ∅`. The board of every `ReachableAt`
state is produced by exactly such an operation — either the empty init board, or one
`adversarialStep` (defeq `clearLines ∘ place`) — so the no-full-rows property holds at
*every* surviving depth: a game never carries an uncompleted full row between steps.
This is the trajectory deployment TOP TARGET (a) asks for. It lets the iter-112
per-placement clear cap `≤ 4` fire on *any* reachable board, with no side hypotheses,
giving the chainable per-step clear upper bound that pairs against the iter-111
forced-clear lower bound `k ≥ (4n − 200)/10`. -/

/-- **No-full-rows invariant along reachable trajectories.** Every exact-depth
reachable board has *no* full rows. The property is not inductive — it holds because
the last operation producing the board was either the empty init (vacuously no full
rows) or a `clearLines` (which removes every full row, iter-113). So a surviving game
never carries an uncompleted full row between steps. -/
theorem fullRows_eq_empty_of_reachableAt {cfg : GameConfig} {n : ℕ} {g : GameState}
    (hcol : 0 < cfg.cols) (h : ReachableAt cfg n g) :
    Board.fullRows cfg g.board = ∅ := by
  cases h with
  | init =>
      rw [GameState.init_board_eq_emptyset]
      exact Board.fullRows_empty cfg
  | @step n g p pl _hprev _hnl _hp _hvalid =>
      rw [adversarialStep_board]
      exact fullRows_applyStep_eq_empty g.board ({pl with piece := p}) hcol

/-- **Per-placement clear cap along trajectories.** From any reachable board, a single
placement completes at most `4` rows: `|fullRows (pl.place g.board)| ≤ 4`. The iter-112
cap `fullRows_place_card_le_four_of_no_fullRows`, with its `fullRows = ∅` side
condition discharged automatically by the trajectory invariant above. This is the
chainable per-step *upper* bound on clears that must be reconciled with the iter-111
forced-clear *lower* bound `k ≥ (4n − 200)/10` for any survival verdict. -/
theorem fullRows_place_card_le_four_of_reachableAt {cfg : GameConfig} {n : ℕ}
    {g : GameState} (pl : Placement) (hcol : 0 < cfg.cols)
    (h : ReachableAt cfg n g) :
    (Board.fullRows cfg (pl.place g.board)).card ≤ 4 :=
  fullRows_place_card_le_four_of_no_fullRows g.board pl hcol
    (fullRows_eq_empty_of_reachableAt hcol h)

/-! ### Iteration 116 — line clears do not raise total stack height (downward half of Φ)

The survival potential `Φ = ∑ colHeight = count + holes` (conservation law, iter-73) has a
proven *upward* half: a no-clear step raises it by `≥ 4` (`count_add_holes_applyStep_ge_of_no_fullRows`,
iter-114). The missing *downward* half is how `Φ` moves under a line clear. Summing the
per-column fact `colHeight_clearLines_le` (gravity after a clear lowers each column, never
raises it) gives the aggregate statement: `clearLines` is `∑ colHeight`-non-increasing, and
hence so is the clear phase of any `applyStep`. This pins the only direction `Φ` can fall —
the clear branch — isolating exactly where a divergence argument must control the drop. -/

/-- **Clears never raise the total stack height.** `∑ colHeight` is non-increasing under
`clearLines`: the per-column bound `colHeight_clearLines_le` (gravity lowers each column,
never raises it) summed over all columns. The downward companion to the no-clear
`Φ' ≥ Φ + 4` rise (iter-114). -/
theorem sum_colHeight_clearLines_le {cfg : GameConfig} (b : Board) :
    ∑ j ∈ Finset.range cfg.cols, (Board.clearLines cfg b).colHeight j
      ≤ ∑ j ∈ Finset.range cfg.cols, b.colHeight j :=
  Finset.sum_le_sum (fun j _ => colHeight_clearLines_le cfg b j)

/-- **A full step's height is capped by its post-placement height.** Because
`applyStep = clearLines ∘ place`, the aggregate stack height after a complete move never
exceeds the height immediately after the placement (before clears): the clear phase can
only lower `∑ colHeight`. The companion to the per-step clear-count cap — clears shed cells
*and* never lift the stack. -/
theorem sum_colHeight_applyStep_le_place {cfg : GameConfig} (b : Board) (pl : Placement) :
    ∑ j ∈ Finset.range cfg.cols, (pl.applyStep cfg b).colHeight j
      ≤ ∑ j ∈ Finset.range cfg.cols, (pl.place b).colHeight j := by
  rw [Placement.applyStep_eq_clearLines_place]
  exact sum_colHeight_clearLines_le (pl.place b)

/-! ### Iteration 117 — hole creation escapes the empty board

`one_le_holes_S_place_empty` / `_Z_` only fire from `∅`. The empty board's role there is
narrow: it merely guarantees the witness column's *bottom* cell `(pl.col+j, 0)` is unfilled.
The hard-drop offset is otherwise irrelevant to the buried-cell argument — the witness body
cell lands at `(pl.col+j, dropOffset b + h)` with `h ≥ 1`, so its row is positive for *any*
board, and `(pl.col+j, 0)` is never a *dropped* cell (that would force `(j,0) ∈ shapeUp`,
contradicting the witness). So the empty-board hypothesis weakens to **"row 0 is unfilled"**
(`∀ c, (c,0) ∉ b`): on any such board an S (or Z) placement still buries a cell, for an
arbitrary hard-drop offset. The first hole-creation lemma that escapes `∅`, and the technical
template — handling `dropOffset b` symbolically — for pushing hole lower bounds onto the
non-empty stacks a depth-growing bound requires. -/

/-- **S buries a cell on any empty-bottom-row board.** If row 0 of `b` is entirely empty,
placing an S leaves the hole-creating column's body above an empty bottom cell — at least one
buried cell. Generalises `one_le_holes_S_place_empty` (the `b = ∅` case) to arbitrary stacks
with an unfilled bottom row, for any hard-drop offset. -/
theorem one_le_holes_S_place_of_bottom_row_empty {cfg : GameConfig} {b : Board}
    {pl : Placement} (hv : pl.Valid cfg) (hS : pl.piece = Piece.S)
    (hbot : ∀ c : ℕ, (c, 0) ∉ b) :
    1 ≤ holes cfg (pl.place b) := by
  classical
  obtain ⟨j, h, hpos, hmem, hzero⟩ := s_hole_witness pl.rot
  have hshape : pl.shapeUp = Piece.S.shapeUp pl.rot := by
    unfold Placement.shapeUp; rw [hS]
  have hmem' : (j, h) ∈ pl.shapeUp := hshape ▸ hmem
  have hzero' : (j, 0) ∉ pl.shapeUp := hshape ▸ hzero
  have hcellMem : (pl.col + j, pl.dropOffset b + h) ∈ pl.place b := by
    rw [Placement.place_eq_union_dropped, Finset.mem_union]
    right
    rw [Placement.dropped_eq_image]
    exact Finset.mem_image.mpr ⟨(j, h), hmem', rfl⟩
  have hposH : 0 < pl.dropOffset b + h := by omega
  have hcellNotMem : (pl.col + j, 0) ∉ pl.place b := by
    rw [Placement.place_eq_union_dropped, Finset.mem_union]
    rintro (hb | hd)
    · exact hbot _ hb
    · rw [Placement.dropped_eq_image, Finset.mem_image] at hd
      obtain ⟨⟨c, r⟩, hcellShape, heq⟩ := hd
      simp only [Prod.mk.injEq, add_right_inj] at heq
      obtain ⟨hc, hr⟩ := heq
      have hr0 : r = 0 := by omega
      subst hc; subst hr0
      exact hzero' hcellShape
  have hcol : 1 ≤ colHoles (pl.place b) (pl.col + j) :=
    one_le_colHoles_of_zero_notMem_of_mem_pos hposH hcellMem hcellNotMem
  have hrange : pl.col + j < cfg.cols := hv (j, h) hmem'
  unfold holes
  calc 1 ≤ colHoles (pl.place b) (pl.col + j) := hcol
    _ ≤ ∑ j' ∈ Finset.range cfg.cols, colHoles (pl.place b) j' :=
        Finset.single_le_sum (f := fun j' => colHoles (pl.place b) j')
          (fun _ _ => Nat.zero_le _) (Finset.mem_range.mpr hrange)

/-- **Z buries a cell on any empty-bottom-row board.** The Z mirror of
`one_le_holes_S_place_of_bottom_row_empty`; generalises `one_le_holes_Z_place_empty`. -/
theorem one_le_holes_Z_place_of_bottom_row_empty {cfg : GameConfig} {b : Board}
    {pl : Placement} (hv : pl.Valid cfg) (hZ : pl.piece = Piece.Z)
    (hbot : ∀ c : ℕ, (c, 0) ∉ b) :
    1 ≤ holes cfg (pl.place b) := by
  classical
  obtain ⟨j, h, hpos, hmem, hzero⟩ := z_hole_witness pl.rot
  have hshape : pl.shapeUp = Piece.Z.shapeUp pl.rot := by
    unfold Placement.shapeUp; rw [hZ]
  have hmem' : (j, h) ∈ pl.shapeUp := hshape ▸ hmem
  have hzero' : (j, 0) ∉ pl.shapeUp := hshape ▸ hzero
  have hcellMem : (pl.col + j, pl.dropOffset b + h) ∈ pl.place b := by
    rw [Placement.place_eq_union_dropped, Finset.mem_union]
    right
    rw [Placement.dropped_eq_image]
    exact Finset.mem_image.mpr ⟨(j, h), hmem', rfl⟩
  have hposH : 0 < pl.dropOffset b + h := by omega
  have hcellNotMem : (pl.col + j, 0) ∉ pl.place b := by
    rw [Placement.place_eq_union_dropped, Finset.mem_union]
    rintro (hb | hd)
    · exact hbot _ hb
    · rw [Placement.dropped_eq_image, Finset.mem_image] at hd
      obtain ⟨⟨c, r⟩, hcellShape, heq⟩ := hd
      simp only [Prod.mk.injEq, add_right_inj] at heq
      obtain ⟨hc, hr⟩ := heq
      have hr0 : r = 0 := by omega
      subst hc; subst hr0
      exact hzero' hcellShape
  have hcol : 1 ≤ colHoles (pl.place b) (pl.col + j) :=
    one_le_colHoles_of_zero_notMem_of_mem_pos hposH hcellMem hcellNotMem
  have hrange : pl.col + j < cfg.cols := hv (j, h) hmem'
  unfold holes
  calc 1 ≤ colHoles (pl.place b) (pl.col + j) := hcol
    _ ≤ ∑ j' ∈ Finset.range cfg.cols, colHoles (pl.place b) j' :=
        Finset.single_le_sum (f := fun j' => colHoles (pl.place b) j')
          (fun _ _ => Nat.zero_le _) (Finset.mem_range.mpr hrange)

/-- **Every closed init-atlas contains a board with a buried cell.** From `init`
the adversary may draw `S`; closedness forces a valid `S`-placement whose successor
stays in `T`, and that successor's board is `pl.applyStep ∅`, which carries at least
one hole (`one_le_holes_S_applyStep_empty`). So no closed atlas containing `init`
is hole-free — the first atlas-level brick of Brzustowski hole accumulation, lifting
`one_le_holes_S_place_empty` from a single drop to a structural property of the
*whole* closed witness. -/
theorem exists_holes_pos_of_init_mem_closed {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, 1 ≤ holes GameConfig.standard g.board := by
  have hSb : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  obtain ⟨_, hstep⟩ := hclosed GameState.init hinit
  obtain ⟨pl, hpl, hsucc⟩ := hstep Piece.S hSb
  rw [Placement.mem_allValidFor] at hpl
  obtain ⟨hpiece, hvalid⟩ := hpl
  refine ⟨adversarialStep GameConfig.standard GameState.init Piece.S pl, hsucc, ?_⟩
  have hpl_eq : ({pl with piece := Piece.S} : Placement) = pl := by
    rcases pl with ⟨pp, pr, pc⟩
    cases hpiece
    rfl
  have hsucc_board :
      (adversarialStep GameConfig.standard GameState.init Piece.S pl).board =
        pl.applyStep GameConfig.standard ∅ := by
    rw [adversarialStep_board, GameState.init_board_eq_emptyset, hpl_eq]
  rw [hsucc_board]
  exact one_le_holes_S_applyStep_empty (by simp) hvalid hpiece

/-- **The singleton `{init}` is not a closed atlas.** Were it closed it would have
to contain a board with a buried cell (`exists_holes_pos_of_init_mem_closed`); but
its only member is `init`, whose board is `∅` with zero holes. The most trivial
TRUE-route witness is therefore impossible — any closed atlas needs ≥ 2 states. -/
theorem not_isClosedSet_singleton_init :
    ¬ IsClosedSet GameConfig.standard {GameState.init} := by
  intro hclosed
  obtain ⟨g, hg, hpos⟩ :=
    exists_holes_pos_of_init_mem_closed (Finset.mem_singleton_self _) hclosed
  rw [Finset.mem_singleton] at hg
  subst hg
  rw [GameState.init_board_eq_emptyset, holes_empty] at hpos
  omega

/-- **Sharper atlas hole brick: the forced holey member has exactly four cells.**
Strengthens `exists_holes_pos_of_init_mem_closed` by pinning the witness board's
cell count. The `S`-successor of `init` is `pl.place ∅` (with only four cells no row
can clear, so `applyStep` collapses to `place`), giving `count = 4` **and**
`holes ≥ 1`. So every closed init-atlas contains a state with both quantities known
exactly enough to lower-bound the survival potential `Φ = count + holes`. -/
theorem exists_mem_closed_count_four_holes_pos {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, g.board.count = 4 ∧ 1 ≤ holes GameConfig.standard g.board := by
  have h5 : (5 : ℕ) ≤ GameConfig.standard.cols := by decide
  have hSb : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  obtain ⟨_, hstep⟩ := hclosed GameState.init hinit
  obtain ⟨pl, hpl, hsucc⟩ := hstep Piece.S hSb
  rw [Placement.mem_allValidFor] at hpl
  obtain ⟨hpiece, hvalid⟩ := hpl
  have hpl_eq : ({pl with piece := Piece.S} : Placement) = pl := by
    rcases pl with ⟨pp, pr, pc⟩
    cases hpiece
    rfl
  have hsucc_board :
      (adversarialStep GameConfig.standard GameState.init Piece.S pl).board =
        pl.applyStep GameConfig.standard ∅ := by
    rw [adversarialStep_board, GameState.init_board_eq_emptyset, hpl_eq]
  refine ⟨adversarialStep GameConfig.standard GameState.init Piece.S pl, hsucc, ?_, ?_⟩
  · rw [hsucc_board, applyStep_empty_eq_place_empty h5, Placement.count_place]; rfl
  · rw [hsucc_board]
    exact one_le_holes_S_applyStep_empty h5 hvalid hpiece

/-- **Survival-potential floor on a closed witness.** Combining the pinned count and
the buried cell: every closed init-atlas contains a state whose `Φ = count + holes`
is at least `5`. A first nontrivial lower bound on the potential a closed witness
must carry, sitting beneath the budget ceiling `count_add_holes_le_two_hundred_of_reachableAt`. -/
theorem exists_mem_closed_five_le_count_add_holes {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, 5 ≤ g.board.count + holes GameConfig.standard g.board := by
  obtain ⟨g, hg, hc, hh⟩ := exists_mem_closed_count_four_holes_pos hinit hclosed
  exact ⟨g, hg, by omega⟩

/-- **General buried-cell certificate.** A column with an empty cell `(j, re)` sitting
strictly below a filled cell `(j, rf)` has at least one hole. Generalises
`one_le_colHoles_of_zero_notMem_of_mem_pos` from the empty cell being row `0` to *any*
row below a filled one — the form needed to certify holes created by an overhang at the
piece's resting offset rather than at the board floor. -/
theorem one_le_colHoles_of_notMem_of_mem_gt {b : Board} {j re rf : ℕ}
    (hlt : re < rf) (hf : (j, rf) ∈ b) (he : (j, re) ∉ b) :
    1 ≤ colHoles b j := by
  classical
  rw [colHoles_eq_sdiff_card]
  refine Finset.card_pos.mpr ⟨re, ?_⟩
  rw [Finset.mem_sdiff, Finset.mem_range]
  refine ⟨?_, ?_⟩
  · have hrf : rf < b.colHeight j := Board.lt_colHeight hf
    omega
  · intro hin
    rcases Finset.mem_image.mp hin with ⟨⟨c, r'⟩, hmem', heq⟩
    rcases Finset.mem_filter.mp hmem' with ⟨hcell, hcol⟩
    simp only at hcol heq
    subst hcol
    subst heq
    exact he hcell

/-- **Overhang hole creation — the localised iter-117 lemma.** Suppose the piece has a
*notch column* `j`: a body cell `(j, h)` with `h > 0` but nothing at relative row `0`
(`s_hole_witness` / `z_hole_witness` supply exactly this for S and Z). If that column's
surface in `b` sits at or below the piece's resting offset
(`b.colHeight (pl.col + j) ≤ pl.dropOffset b` — i.e. the piece rests on a *taller*
neighbour and overhangs this column), then placing the piece buries a cell, so
`1 ≤ holes`. Unlike `one_le_holes_S_place_of_bottom_row_empty` this needs only a
*single-column* surface condition, so it survives on tall, partially-filled boards —
the gateway to a depth-growing holes bound. -/
theorem one_le_holes_place_of_overhang {cfg : GameConfig} {b : Board} {pl : Placement}
    {j h : ℕ} (hv : pl.Valid cfg) (hpos : 0 < h)
    (hmem : (j, h) ∈ pl.shapeUp) (hzero : (j, 0) ∉ pl.shapeUp)
    (hover : b.colHeight (pl.col + j) ≤ pl.dropOffset b) :
    1 ≤ holes cfg (pl.place b) := by
  classical
  have hbodyMem : (pl.col + j, pl.dropOffset b + h) ∈ pl.place b := by
    rw [Placement.place_eq_union_dropped, Finset.mem_union]
    right
    rw [Placement.dropped_eq_image]
    exact Finset.mem_image.mpr ⟨(j, h), hmem, rfl⟩
  have hbaseNotMem : (pl.col + j, pl.dropOffset b) ∉ pl.place b := by
    rw [Placement.place_eq_union_dropped, Finset.mem_union]
    rintro (hb | hd)
    · have := Board.lt_colHeight hb
      omega
    · rw [Placement.dropped_eq_image, Finset.mem_image] at hd
      obtain ⟨⟨c, r⟩, hcellShape, heq⟩ := hd
      simp only [Prod.mk.injEq, add_right_inj] at heq
      obtain ⟨hc, hr⟩ := heq
      have hr0 : r = 0 := by omega
      subst hc
      subst hr0
      exact hzero hcellShape
  have hcol : 1 ≤ colHoles (pl.place b) (pl.col + j) :=
    one_le_colHoles_of_notMem_of_mem_gt (by omega) hbodyMem hbaseNotMem
  have hrange : pl.col + j < cfg.cols := hv (j, h) hmem
  unfold holes
  calc 1 ≤ colHoles (pl.place b) (pl.col + j) := hcol
    _ ≤ ∑ j' ∈ Finset.range cfg.cols, colHoles (pl.place b) j' :=
        Finset.single_le_sum (f := fun j' => colHoles (pl.place b) j')
          (fun _ _ => Nat.zero_le _) (Finset.mem_range.mpr hrange)

/-- **S overhang corollary — the per-piece hole engine.** Couples `s_hole_witness`
to `one_le_holes_place_of_overhang`: every S placement exposes a concrete *notch column*
`j` (a body cell `(j, h)` with `h > 0` but nothing at `(j, 0)`) such that **if** the
board's surface directly under that column sits at or below the resting offset
(`b.colHeight (pl.col + j) ≤ pl.dropOffset b`), the placement buries a cell and
`1 ≤ holes cfg (pl.place b)`. Discharges the abstract overhang side-condition from the
S shape itself — the caller now only inspects *one* column's surface profile, not the
global empty-row-0 hypothesis of `one_le_holes_S_place_of_bottom_row_empty`. -/
theorem one_le_holes_S_place_of_overhang {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hS : pl.piece = Piece.S) :
    ∃ j h : ℕ, 0 < h ∧ (j, h) ∈ pl.shapeUp ∧ (j, 0) ∉ pl.shapeUp ∧
      (b.colHeight (pl.col + j) ≤ pl.dropOffset b → 1 ≤ holes cfg (pl.place b)) := by
  classical
  obtain ⟨j, h, hpos, hmem, hzero⟩ := s_hole_witness pl.rot
  have hshape : pl.shapeUp = Piece.S.shapeUp pl.rot := by
    unfold Placement.shapeUp; rw [hS]
  have hmem' : (j, h) ∈ pl.shapeUp := hshape ▸ hmem
  have hzero' : (j, 0) ∉ pl.shapeUp := hshape ▸ hzero
  exact ⟨j, h, hpos, hmem', hzero',
    fun hover => one_le_holes_place_of_overhang hv hpos hmem' hzero' hover⟩

/-- Z analog of `one_le_holes_S_place_of_overhang`. Every Z placement likewise exposes a
notch column whose overhang forces a buried cell. -/
theorem one_le_holes_Z_place_of_overhang {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hZ : pl.piece = Piece.Z) :
    ∃ j h : ℕ, 0 < h ∧ (j, h) ∈ pl.shapeUp ∧ (j, 0) ∉ pl.shapeUp ∧
      (b.colHeight (pl.col + j) ≤ pl.dropOffset b → 1 ≤ holes cfg (pl.place b)) := by
  classical
  obtain ⟨j, h, hpos, hmem, hzero⟩ := z_hole_witness pl.rot
  have hshape : pl.shapeUp = Piece.Z.shapeUp pl.rot := by
    unfold Placement.shapeUp; rw [hZ]
  have hmem' : (j, h) ∈ pl.shapeUp := hshape ▸ hmem
  have hzero' : (j, 0) ∉ pl.shapeUp := hshape ▸ hzero
  exact ⟨j, h, hpos, hmem', hzero',
    fun hover => one_le_holes_place_of_overhang hv hpos hmem' hzero' hover⟩

/-- **S overhang corollary lifted to `applyStep` (no-clear branch).** When the placed
board has no full row (`clearLines` is the identity, so `applyStep = place`), the
per-piece overhang hole survives the line-clear phase. Every S placement exposes a notch
column `j` such that *if* its surface is overhung **and** the placement clears no line,
the resulting `applyStep` board carries a buried cell. This is the adversarial-walk form:
`adversarialStep` runs through `applyStep`, so this — not the bare `place` form
`one_le_holes_S_place_of_overhang` — is the version that chains along a trajectory. -/
theorem one_le_holes_S_applyStep_of_overhang {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hS : pl.piece = Piece.S) :
    ∃ j h : ℕ, 0 < h ∧ (j, h) ∈ pl.shapeUp ∧ (j, 0) ∉ pl.shapeUp ∧
      (b.colHeight (pl.col + j) ≤ pl.dropOffset b →
        Board.fullRows cfg (pl.place b) = ∅ →
        1 ≤ holes cfg (pl.applyStep cfg b)) := by
  classical
  obtain ⟨j, h, hpos, hmem, hzero, hcond⟩ := one_le_holes_S_place_of_overhang (b := b) hv hS
  refine ⟨j, h, hpos, hmem, hzero, fun hover hno => ?_⟩
  have hself : pl.applyStep cfg b = pl.place b := by
    rw [Placement.applyStep_eq_clearLines_place]
    exact Board.clearLines_eq_self_of_no_fullRows cfg hno
  rw [hself]
  exact hcond hover

/-- Z analog of `one_le_holes_S_applyStep_of_overhang`. -/
theorem one_le_holes_Z_applyStep_of_overhang {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hZ : pl.piece = Piece.Z) :
    ∃ j h : ℕ, 0 < h ∧ (j, h) ∈ pl.shapeUp ∧ (j, 0) ∉ pl.shapeUp ∧
      (b.colHeight (pl.col + j) ≤ pl.dropOffset b →
        Board.fullRows cfg (pl.place b) = ∅ →
        1 ≤ holes cfg (pl.applyStep cfg b)) := by
  classical
  obtain ⟨j, h, hpos, hmem, hzero, hcond⟩ := one_le_holes_Z_place_of_overhang (b := b) hv hZ
  refine ⟨j, h, hpos, hmem, hzero, fun hover hno => ?_⟩
  have hself : pl.applyStep cfg b = pl.place b := by
    rw [Placement.applyStep_eq_clearLines_place]
    exact Board.clearLines_eq_self_of_no_fullRows cfg hno
  rw [hself]
  exact hcond hover

/-- **Checkerboard 2-coloring — even cells.** Cells whose `col + row` is even (one colour
class of the standard chessboard coloring). The motivation (Brzustowski 1992) is that a
coloring functional surviving line clears is the natural candidate for a depth-growing net
potential, since `colHoles` provably does *not* persist across clears. -/
def evenCells (b : Board) : ℕ := (b.filter (fun c => (c.1 + c.2) % 2 = 0)).card

/-- **Checkerboard 2-coloring — odd cells.** The complementary colour class:
cells whose `col + row` is odd. -/
def oddCells (b : Board) : ℕ := (b.filter (fun c => (c.1 + c.2) % 2 = 1)).card

/-- The two colour classes partition the board: `evenCells + oddCells = count`. This is the
bridge tying the coloring functional to the conserved `count = 4n − cols·k` axis. -/
theorem evenCells_add_oddCells (b : Board) :
    evenCells b + oddCells b = b.card := by
  classical
  have hodd : oddCells b
      = (b.filter (fun c : Coord => ¬ ((c.1 + c.2) % 2 = 0))).card := by
    unfold oddCells
    congr 1
    apply Finset.filter_congr
    intro c _
    omega
  unfold evenCells
  rw [hodd]
  exact Finset.filter_card_add_filter_neg_card_eq_card _

/-- **S is checkerboard-balanced.** In *every* rotation the S tetromino's body covers
exactly two even and two odd cells. Combined with the fact that a uniform translation
preserves a 2–2 split and a full row removes 5 even + 5 odd, this is the seed for a
clear-invariant coloring potential restricted to the S/Z sub-dynamics. Decidable per
rotation. -/
theorem s_shape_parity_balanced (r : Rotation) :
    evenCells (Piece.S.shapeUp r) = 2 ∧ oddCells (Piece.S.shapeUp r) = 2 := by
  unfold evenCells oddCells
  fin_cases r <;> exact ⟨by decide, by decide⟩

/-- Z analog of `s_shape_parity_balanced`: the Z tetromino is likewise checkerboard-balanced
(2 even, 2 odd) in every rotation. -/
theorem z_shape_parity_balanced (r : Rotation) :
    evenCells (Piece.Z.shapeUp r) = 2 ∧ oddCells (Piece.Z.shapeUp r) = 2 := by
  unfold evenCells oddCells
  fin_cases r <;> exact ⟨by decide, by decide⟩

/-- **Signed checkerboard potential.** `+1` on even cells, `−1` on odd cells, summed over the
board (values in `ℤ`). A full line removes 5 even + 5 odd cells, contributing `+5 − 5 = 0`, so
`colorSum` is *clear-invariant* — the property `colHoles` lacks (iter-119). Signed companion of
`evenCells`/`oddCells`. -/
def colorSum (b : Board) : ℤ := ∑ c ∈ b, if (c.1 + c.2) % 2 = 0 then (1 : ℤ) else -1

/-- The signed potential is the difference of the two colour-class counts. Bridges `colorSum`
to `evenCells`/`oddCells` (hence, via `evenCells_add_oddCells`, to the conserved `count`). -/
theorem colorSum_eq (b : Board) :
    colorSum b = (evenCells b : ℤ) - (oddCells b : ℤ) := by
  classical
  unfold colorSum evenCells oddCells
  have key : ∀ c : Coord,
      (if (c.1 + c.2) % 2 = 0 then (1 : ℤ) else -1)
        = (if (c.1 + c.2) % 2 = 0 then (1 : ℤ) else 0)
          - (if (c.1 + c.2) % 2 = 1 then (1 : ℤ) else 0) := by
    intro c
    by_cases h : (c.1 + c.2) % 2 = 0
    · have h1 : ¬ (c.1 + c.2) % 2 = 1 := by omega
      rw [if_pos h, if_pos h, if_neg h1]; ring
    · have h1 : (c.1 + c.2) % 2 = 1 := by omega
      rw [if_neg h, if_neg h, if_pos h1]; ring
  rw [Finset.sum_congr rfl (fun c _ => key c), Finset.sum_sub_distrib,
      Finset.sum_boole, Finset.sum_boole]

/-- **S is colour-neutral.** Every rotation of the S tetromino has signed potential `0`
(2 even − 2 odd). Follows from `s_shape_parity_balanced` via the bridge. -/
theorem colorSum_S_shapeUp_zero (r : Rotation) :
    colorSum (Piece.S.shapeUp r) = 0 := by
  rw [colorSum_eq]
  obtain ⟨he, ho⟩ := s_shape_parity_balanced r
  rw [he, ho]; norm_num

/-- Z analog: every rotation of the Z tetromino is colour-neutral. -/
theorem colorSum_Z_shapeUp_zero (r : Rotation) :
    colorSum (Piece.Z.shapeUp r) = 0 := by
  rw [colorSum_eq]
  obtain ⟨he, ho⟩ := z_shape_parity_balanced r
  rw [he, ho]; norm_num

/-- The signed colour potential is additive over disjoint unions. -/
theorem colorSum_union_disjoint {s t : Board} (h : Disjoint s t) :
    colorSum (s ∪ t) = colorSum s + colorSum t := by
  unfold colorSum
  rw [Finset.sum_union h]

/-- **Translation invariance of colour-neutrality.** A pure translation of a
colour-balanced cell set stays balanced: if `colorSum pl.shapeUp = 0` then the
copy translated to absolute columns at any vertical offset `d` (i.e. `cellsAt d`)
also has `colorSum = 0`. The parity shift `(pl.col + d) % 2` either preserves
every cell's colour (global factor `+1`) or flips all of them (factor `-1`);
either way `0` maps to `0`. -/
theorem colorSum_cellsAt_of_zero (pl : Placement) (d : ℕ)
    (h : colorSum pl.shapeUp = 0) : colorSum (pl.cellsAt d) = 0 := by
  classical
  have factor : ∀ cell ∈ pl.shapeUp,
      (if (pl.col + cell.1 + (d + cell.2)) % 2 = 0 then (1 : ℤ) else -1)
        = (if (pl.col + d) % 2 = 0 then (1 : ℤ) else -1)
          * (if (cell.1 + cell.2) % 2 = 0 then (1 : ℤ) else -1) := by
    intro cell _
    by_cases h1 : (pl.col + d) % 2 = 0
    · by_cases h2 : (cell.1 + cell.2) % 2 = 0
      · rw [if_pos (show (pl.col + cell.1 + (d + cell.2)) % 2 = 0 by omega),
            if_pos h1, if_pos h2]; ring
      · rw [if_neg (show ¬ (pl.col + cell.1 + (d + cell.2)) % 2 = 0 by omega),
            if_pos h1, if_neg h2]; ring
    · by_cases h2 : (cell.1 + cell.2) % 2 = 0
      · rw [if_neg (show ¬ (pl.col + cell.1 + (d + cell.2)) % 2 = 0 by omega),
            if_neg h1, if_pos h2]; ring
      · rw [if_pos (show (pl.col + cell.1 + (d + cell.2)) % 2 = 0 by omega),
            if_neg h1, if_neg h2]; ring
  unfold colorSum Placement.cellsAt
  rw [Finset.sum_image (fun x _ y _ hxy => Placement.cellsAt_injective pl d hxy)]
  show (∑ cell ∈ pl.shapeUp,
      (if (pl.col + cell.1 + (d + cell.2)) % 2 = 0 then (1 : ℤ) else -1)) = 0
  have step : (∑ cell ∈ pl.shapeUp,
      (if (pl.col + cell.1 + (d + cell.2)) % 2 = 0 then (1 : ℤ) else -1))
      = (if (pl.col + d) % 2 = 0 then (1 : ℤ) else -1) * colorSum pl.shapeUp := by
    unfold colorSum
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl factor
  rw [step, h, mul_zero]

/-- The hard-dropped copy of an `S` piece onto any board is colour-neutral. -/
theorem colorSum_dropped_S (b : Board) {pl : Placement} (hS : pl.piece = Piece.S) :
    colorSum (pl.dropped b) = 0 := by
  unfold Placement.dropped
  refine colorSum_cellsAt_of_zero pl (pl.dropOffset b) ?_
  unfold Placement.shapeUp
  rw [hS]
  exact colorSum_S_shapeUp_zero pl.rot

/-- The hard-dropped copy of a `Z` piece onto any board is colour-neutral. -/
theorem colorSum_dropped_Z (b : Board) {pl : Placement} (hZ : pl.piece = Piece.Z) :
    colorSum (pl.dropped b) = 0 := by
  unfold Placement.dropped
  refine colorSum_cellsAt_of_zero pl (pl.dropOffset b) ?_
  unfold Placement.shapeUp
  rw [hZ]
  exact colorSum_Z_shapeUp_zero pl.rot

/-- **S-drop colour invariance.** Hard-dropping an `S` piece onto any board leaves
the signed checkerboard potential `colorSum` unchanged (before any line clears).
This makes `colorSum` a genuine conserved quantity along S-only play. -/
theorem colorSum_place_S (b : Board) {pl : Placement} (hS : pl.piece = Piece.S) :
    colorSum (pl.place b) = colorSum b := by
  rw [Placement.place_eq_union_dropped,
      colorSum_union_disjoint (pl.dropped_disjoint b).symm,
      colorSum_dropped_S b hS, add_zero]

/-- **Z-drop colour invariance.** Hard-dropping a `Z` piece onto any board leaves
`colorSum` unchanged (before any line clears). -/
theorem colorSum_place_Z (b : Board) {pl : Placement} (hZ : pl.piece = Piece.Z) :
    colorSum (pl.place b) = colorSum b := by
  rw [Placement.place_eq_union_dropped,
      colorSum_union_disjoint (pl.dropped_disjoint b).symm,
      colorSum_dropped_Z b hZ, add_zero]

/-- Alternating ±1 sum over an even range vanishes: `∑_{j<2m} (-1)^j = 0`. -/
theorem sum_alt_eq_zero_two_mul (m : ℕ) :
    (∑ j ∈ Finset.range (2 * m), (-1 : ℤ) ^ j) = 0 := by
  induction m with
  | zero => simp
  | succ k ih =>
    have h2k : (-1 : ℤ) ^ (2 * k) = 1 := by rw [pow_mul]; norm_num
    rw [Nat.mul_succ, Finset.sum_range_succ, Finset.sum_range_succ, ih, pow_succ, h2k]
    ring

/-- Column-parity signed potential: alternate the sign of each column's cell
count by column index. Unlike row-checkerboard `colorSum`, this is invariant
under line clears on an even-width board (gravity keeps each cell in its column,
and a removed full row contributes `∑_{j<cols}(-1)^j = 0`). -/
def colSign (cfg : GameConfig) (b : Board) : ℤ :=
  ∑ j ∈ Finset.range cfg.cols, (-1 : ℤ) ^ j * (b.colCount j : ℤ)

@[simp] theorem colSign_empty (cfg : GameConfig) : colSign cfg (∅ : Board) = 0 := by
  unfold colSign; simp [Board.colCount_empty]

/-- `colSign` is preserved by a line clear whenever the alternating column-sign
sum vanishes (the per-cleared-row contribution cancels). -/
theorem colSign_clearLines_of_alt_zero (cfg : GameConfig) (b : Board)
    (halt : (∑ j ∈ Finset.range cfg.cols, (-1 : ℤ) ^ j) = 0) :
    colSign cfg (Board.clearLines cfg b) = colSign cfg b := by
  unfold colSign
  have hstep : ∀ j ∈ Finset.range cfg.cols,
      (-1 : ℤ) ^ j * ((Board.clearLines cfg b).colCount j : ℤ)
        = (-1 : ℤ) ^ j * (b.colCount j : ℤ)
          - ((Board.fullRows cfg b).card : ℤ) * (-1 : ℤ) ^ j := by
    intro j hj
    have hjlt := Finset.mem_range.mp hj
    have hnat := Board.colCount_clearLines_add cfg b hjlt
    have hcast : ((Board.clearLines cfg b).colCount j : ℤ)
        = (b.colCount j : ℤ) - ((Board.fullRows cfg b).card : ℤ) := by
      have : (b.colCount j : ℤ)
          = ((Board.clearLines cfg b).colCount j : ℤ) + ((Board.fullRows cfg b).card : ℤ) := by
        exact_mod_cast congrArg (Nat.cast : ℕ → ℤ) hnat
      omega
    rw [hcast]; ring
  rw [Finset.sum_congr rfl hstep, Finset.sum_sub_distrib, ← Finset.mul_sum, halt,
      mul_zero, sub_zero]

/-- On an even-width board, `colSign` is invariant under line clears. -/
theorem colSign_clearLines (cfg : GameConfig) (b : Board) (hcols : Even cfg.cols) :
    colSign cfg (Board.clearLines cfg b) = colSign cfg b := by
  apply colSign_clearLines_of_alt_zero
  obtain ⟨m, hm⟩ := hcols
  have hc : cfg.cols = 2 * m := by omega
  rw [hc]; exact sum_alt_eq_zero_two_mul m

/-- **Column-parity 2-coloring — even columns.** Cells in an even-indexed column.
Companion to `evenCells` but keyed on the column only, so (unlike the checkerboard)
it is preserved by line-clear gravity (which never moves a cell sideways). -/
def evenCols (b : Board) : ℕ := (b.filter (fun c => c.1 % 2 = 0)).card

/-- **Column-parity 2-coloring — odd columns.** -/
def oddCols (b : Board) : ℕ := (b.filter (fun c => c.1 % 2 = 1)).card

/-- The two column-parity classes partition the board. -/
theorem evenCols_add_oddCols (b : Board) :
    evenCols b + oddCols b = b.card := by
  classical
  have hodd : oddCols b
      = (b.filter (fun c : Coord => ¬ (c.1 % 2 = 0))).card := by
    unfold oddCols
    congr 1
    apply Finset.filter_congr
    intro c _
    omega
  unfold evenCols
  rw [hodd]
  exact Finset.filter_card_add_filter_neg_card_eq_card _

/-- **S is column-parity-balanced.** In every rotation the S tetromino covers exactly
two even-column and two odd-column cells (shapeUp only shifts rows, so column parity is
the shape's). Decidable per rotation. -/
theorem s_shape_colparity_balanced (r : Rotation) :
    evenCols (Piece.S.shapeUp r) = 2 ∧ oddCols (Piece.S.shapeUp r) = 2 := by
  unfold evenCols oddCols
  fin_cases r <;> exact ⟨by decide, by decide⟩

/-- Z analog: the Z tetromino is column-parity-balanced (2 even-col, 2 odd-col) every rotation. -/
theorem z_shape_colparity_balanced (r : Rotation) :
    evenCols (Piece.Z.shapeUp r) = 2 ∧ oddCols (Piece.Z.shapeUp r) = 2 := by
  unfold evenCols oddCols
  fin_cases r <;> exact ⟨by decide, by decide⟩

/-- **Signed column-parity potential** (cell-level twin of `colSign`): `+1` on even columns,
`−1` on odd columns. Unlike the checkerboard `colorSum`, this depends only on a cell's column,
so it is preserved by S/Z hard drops (proved here) and — being column-only — also by line-clear
gravity. The clean conserved quantity for the S/Z sub-dynamics at the cell level. -/
def colColorSum (b : Board) : ℤ := ∑ c ∈ b, if c.1 % 2 = 0 then (1 : ℤ) else -1

/-- `colColorSum` is the difference of the two column-parity class counts. -/
theorem colColorSum_eq (b : Board) :
    colColorSum b = (evenCols b : ℤ) - (oddCols b : ℤ) := by
  classical
  unfold colColorSum evenCols oddCols
  have key : ∀ c : Coord,
      (if c.1 % 2 = 0 then (1 : ℤ) else -1)
        = (if c.1 % 2 = 0 then (1 : ℤ) else 0)
          - (if c.1 % 2 = 1 then (1 : ℤ) else 0) := by
    intro c
    by_cases h : c.1 % 2 = 0
    · have h1 : ¬ c.1 % 2 = 1 := by omega
      rw [if_pos h, if_pos h, if_neg h1]; ring
    · have h1 : c.1 % 2 = 1 := by omega
      rw [if_neg h, if_neg h, if_pos h1]; ring
  rw [Finset.sum_congr rfl (fun c _ => key c), Finset.sum_sub_distrib,
      Finset.sum_boole, Finset.sum_boole]

/-- Every rotation of `S` has column-parity potential `0` (2 even − 2 odd columns). -/
theorem colColorSum_S_shapeUp_zero (r : Rotation) :
    colColorSum (Piece.S.shapeUp r) = 0 := by
  rw [colColorSum_eq]
  obtain ⟨he, ho⟩ := s_shape_colparity_balanced r
  rw [he, ho]; norm_num

/-- Z analog: every rotation of `Z` has column-parity potential `0`. -/
theorem colColorSum_Z_shapeUp_zero (r : Rotation) :
    colColorSum (Piece.Z.shapeUp r) = 0 := by
  rw [colColorSum_eq]
  obtain ⟨he, ho⟩ := z_shape_colparity_balanced r
  rw [he, ho]; norm_num

/-- The signed column-parity potential is additive over disjoint unions. -/
theorem colColorSum_union_disjoint {s t : Board} (h : Disjoint s t) :
    colColorSum (s ∪ t) = colColorSum s + colColorSum t := by
  unfold colColorSum
  rw [Finset.sum_union h]

/-- **Column-balance is translation-invariant.** Hard-dropping translates a shape by
`(pl.col, d)`; the vertical part `d` never changes a cell's column, and the horizontal
shift either preserves all column parities (`pl.col` even) or flips them all (`pl.col`
odd). So a column-balanced shape stays balanced after translation. -/
theorem colColorSum_cellsAt_of_zero (pl : Placement) (d : ℕ)
    (h : colColorSum pl.shapeUp = 0) : colColorSum (pl.cellsAt d) = 0 := by
  classical
  have factor : ∀ cell ∈ pl.shapeUp,
      (if (pl.col + cell.1) % 2 = 0 then (1 : ℤ) else -1)
        = (if pl.col % 2 = 0 then (1 : ℤ) else -1)
          * (if cell.1 % 2 = 0 then (1 : ℤ) else -1) := by
    intro cell _
    by_cases h1 : pl.col % 2 = 0
    · by_cases h2 : cell.1 % 2 = 0
      · rw [if_pos (show (pl.col + cell.1) % 2 = 0 by omega), if_pos h1, if_pos h2]; ring
      · rw [if_neg (show ¬ (pl.col + cell.1) % 2 = 0 by omega), if_pos h1, if_neg h2]; ring
    · by_cases h2 : cell.1 % 2 = 0
      · rw [if_neg (show ¬ (pl.col + cell.1) % 2 = 0 by omega), if_neg h1, if_pos h2]; ring
      · rw [if_pos (show (pl.col + cell.1) % 2 = 0 by omega), if_neg h1, if_neg h2]; ring
  unfold colColorSum Placement.cellsAt
  rw [Finset.sum_image (fun x _ y _ hxy => Placement.cellsAt_injective pl d hxy)]
  show (∑ cell ∈ pl.shapeUp, (if (pl.col + cell.1) % 2 = 0 then (1 : ℤ) else -1)) = 0
  have step : (∑ cell ∈ pl.shapeUp, (if (pl.col + cell.1) % 2 = 0 then (1 : ℤ) else -1))
      = (if pl.col % 2 = 0 then (1 : ℤ) else -1) * colColorSum pl.shapeUp := by
    unfold colColorSum
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl factor
  rw [step, h, mul_zero]

/-- The hard-dropped copy of an `S` piece onto any board has column-parity potential `0`. -/
theorem colColorSum_dropped_S (b : Board) {pl : Placement} (hS : pl.piece = Piece.S) :
    colColorSum (pl.dropped b) = 0 := by
  unfold Placement.dropped
  refine colColorSum_cellsAt_of_zero pl (pl.dropOffset b) ?_
  unfold Placement.shapeUp
  rw [hS]
  exact colColorSum_S_shapeUp_zero pl.rot

/-- The hard-dropped copy of a `Z` piece onto any board has column-parity potential `0`. -/
theorem colColorSum_dropped_Z (b : Board) {pl : Placement} (hZ : pl.piece = Piece.Z) :
    colColorSum (pl.dropped b) = 0 := by
  unfold Placement.dropped
  refine colColorSum_cellsAt_of_zero pl (pl.dropOffset b) ?_
  unfold Placement.shapeUp
  rw [hZ]
  exact colColorSum_Z_shapeUp_zero pl.rot

/-- **S-drop column-parity invariance.** Hard-dropping an `S` piece onto any board leaves
the signed column-parity potential unchanged. -/
theorem colColorSum_place_S (b : Board) {pl : Placement} (hS : pl.piece = Piece.S) :
    colColorSum (pl.place b) = colColorSum b := by
  rw [Placement.place_eq_union_dropped,
      colColorSum_union_disjoint (pl.dropped_disjoint b).symm,
      colColorSum_dropped_S b hS, add_zero]

/-- **Z-drop column-parity invariance.** -/
theorem colColorSum_place_Z (b : Board) {pl : Placement} (hZ : pl.piece = Piece.Z) :
    colColorSum (pl.place b) = colColorSum b := by
  rw [Placement.place_eq_union_dropped,
      colColorSum_union_disjoint (pl.dropped_disjoint b).symm,
      colColorSum_dropped_Z b hZ, add_zero]

/-- **Bridge: the cell-level and colCount-level column-parity potentials agree** on any
well-formed board. `colColorSum` sums `±1` per cell by column parity; `colSign` sums
`(-1)^j · colCount j` per column. On a `WF` board every cell sits in a column `< cols`, so
fibering the cell sum over its column collapses it to the colCount sum (`(-1)^j = ±1` by
parity of `j`). This unifies the two invariance results: cell-level S/Z place-invariance
(`colColorSum_place_S/Z`) and colCount-level clear-invariance (`colSign_clearLines`). -/
theorem colColorSum_eq_colSign {cfg : GameConfig} {b : Board} (hb : Board.WF cfg b) :
    colColorSum b = colSign cfg b := by
  classical
  have hpow : ∀ j : ℕ, (-1 : ℤ) ^ j = if j % 2 = 0 then (1 : ℤ) else -1 := by
    intro j
    rcases Nat.even_or_odd j with he | ho
    · rw [Even.neg_one_pow he, if_pos (Nat.even_iff.mp he)]
    · rw [Odd.neg_one_pow ho, if_neg (by have := Nat.odd_iff.mp ho; omega)]
  have hfib := Finset.sum_fiberwise_of_maps_to
      (s := b) (t := Finset.range cfg.cols) (g := fun c : Coord => c.1)
      (f := fun c : Coord => if c.1 % 2 = 0 then (1 : ℤ) else -1)
      (fun c hc => Finset.mem_range.2 (hb c hc))
  unfold colColorSum colSign
  rw [← hfib]
  apply Finset.sum_congr rfl
  intro j hj
  have hconst : ∀ c ∈ b.filter (fun c : Coord => c.1 = j),
      (if c.1 % 2 = 0 then (1 : ℤ) else -1) = (if j % 2 = 0 then (1 : ℤ) else -1) := by
    intro c hc; rw [(Finset.mem_filter.mp hc).2]
  rw [Finset.sum_congr rfl hconst, Finset.sum_const, nsmul_eq_mul, hpow j]
  unfold Board.colCount
  ring

/-- **`colColorSum` is clear-invariant** on well-formed even-width boards. Routed through the
bridge: `colColorSum = colSign` (which is clear-invariant by `colSign_clearLines`). The column
view is what survives gravity — clearing keeps each cell in its column. -/
theorem colColorSum_clearLines {cfg : GameConfig} {b : Board}
    (hb : Board.WF cfg b) (hcols : Even cfg.cols) :
    colColorSum (Board.clearLines cfg b) = colColorSum b := by
  rw [colColorSum_eq_colSign (Board.clearLines_wf hb), colSign_clearLines cfg b hcols,
      ← colColorSum_eq_colSign hb]

/-- **Full adversarial-step invariance for `S`.** A complete move (`applyStep` = hard-drop then
line-clear) of an `S` piece on a well-formed even-width board leaves `colColorSum` unchanged:
the place half is `colColorSum_place_S`, the clear half is `colColorSum_clearLines`. So
`colColorSum` is a genuine conserved quantity of the S/Z sub-dynamics — confirming the coloring
lever is *conserved, not growing*, hence cannot by itself force top-out. -/
theorem colColorSum_applyStep_S {cfg : GameConfig} {b : Board} {pl : Placement}
    (hb : Board.WF cfg b) (hv : pl.Valid cfg) (hcols : Even cfg.cols)
    (hS : pl.piece = Piece.S) :
    colColorSum (pl.applyStep cfg b) = colColorSum b := by
  rw [Placement.applyStep_eq_clearLines_place,
      colColorSum_clearLines (Board.WF_place hb hv) hcols, colColorSum_place_S b hS]

/-- **Full adversarial-step invariance for `Z`.** -/
theorem colColorSum_applyStep_Z {cfg : GameConfig} {b : Board} {pl : Placement}
    (hb : Board.WF cfg b) (hv : pl.Valid cfg) (hcols : Even cfg.cols)
    (hZ : pl.piece = Piece.Z) :
    colColorSum (pl.applyStep cfg b) = colColorSum b := by
  rw [Placement.applyStep_eq_clearLines_place,
      colColorSum_clearLines (Board.WF_place hb hv) hcols, colColorSum_place_Z b hZ]

/-! ### The tuck dichotomy — isolating the unique hole-free survival mode

The hole-floor program shows an `S`/`Z` placement buries a cell when the notch
column is flat (`one_le_holes_S_place_of_bottom_row_empty`) or overhangs
(`one_le_holes_S_place_of_overhang`). The only remaining escape is a **tuck**:
the piece's notch column rests with its surface *strictly above* the resting
offset, so the body cell at `dropOffset + h` lands on pre-existing structure and
no gap opens below it. The next two theorems make this precise as a dichotomy —
*every* valid `S`/`Z` placement either buries a cell or tucks — and the following
two restate the tuck side as the directly-usable necessary condition on any
hole-free placement. This is the structural gateway to a tuck-budget argument:
a closed atlas surviving the forced `S`/`Z` draws can avoid holes only by
spending pre-existing surface height (`dropOffset < colHeight (pl.col + j)`),
which is a finite resource. -/

/-- **Tuck dichotomy for `S`.** Every valid `S` placement either buries a cell
(`1 ≤ holes`) or tucks its notch column under pre-existing structure: there is a
notch column `pl.col + j` (body at row `h > 0`, empty at row `0`) whose board
surface strictly exceeds the resting offset. The contrapositive of the overhang
witness `one_le_holes_S_place_of_overhang`, with the `≤`/`<` trichotomy resolved. -/
theorem holes_pos_or_tuck_S {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hS : pl.piece = Piece.S) :
    1 ≤ holes cfg (pl.place b) ∨
      ∃ j h : ℕ, 0 < h ∧ (j, h) ∈ pl.shapeUp ∧ (j, 0) ∉ pl.shapeUp ∧
        pl.dropOffset b < b.colHeight (pl.col + j) := by
  obtain ⟨j, h, hpos, hmem, hzero, himpl⟩ := one_le_holes_S_place_of_overhang hv hS
  rcases Nat.lt_or_ge (pl.dropOffset b) (b.colHeight (pl.col + j)) with hlt | hge
  · exact Or.inr ⟨j, h, hpos, hmem, hzero, hlt⟩
  · exact Or.inl (himpl hge)

/-- **Tuck dichotomy for `Z`.** Z analog of `holes_pos_or_tuck_S`. -/
theorem holes_pos_or_tuck_Z {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hZ : pl.piece = Piece.Z) :
    1 ≤ holes cfg (pl.place b) ∨
      ∃ j h : ℕ, 0 < h ∧ (j, h) ∈ pl.shapeUp ∧ (j, 0) ∉ pl.shapeUp ∧
        pl.dropOffset b < b.colHeight (pl.col + j) := by
  obtain ⟨j, h, hpos, hmem, hzero, himpl⟩ := one_le_holes_Z_place_of_overhang hv hZ
  rcases Nat.lt_or_ge (pl.dropOffset b) (b.colHeight (pl.col + j)) with hlt | hge
  · exact Or.inr ⟨j, h, hpos, hmem, hzero, hlt⟩
  · exact Or.inl (himpl hge)

/-- **Hole-free `S` placement must tuck.** The directly-usable necessary
condition: if an `S` placement opens no hole, its notch column rests strictly
below pre-existing surface (`dropOffset < colHeight (pl.col + j)`). So avoiding a
hole *consumes* surface height — the resource a tuck-budget argument bounds. -/
theorem tuck_of_holes_zero_S {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hS : pl.piece = Piece.S)
    (hno : holes cfg (pl.place b) = 0) :
    ∃ j h : ℕ, 0 < h ∧ (j, h) ∈ pl.shapeUp ∧ (j, 0) ∉ pl.shapeUp ∧
      pl.dropOffset b < b.colHeight (pl.col + j) := by
  rcases holes_pos_or_tuck_S (b := b) hv hS with hpos | htuck
  · exfalso; omega
  · exact htuck

/-- **Hole-free `Z` placement must tuck.** Z analog of `tuck_of_holes_zero_S`. -/
theorem tuck_of_holes_zero_Z {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hZ : pl.piece = Piece.Z)
    (hno : holes cfg (pl.place b) = 0) :
    ∃ j h : ℕ, 0 < h ∧ (j, h) ∈ pl.shapeUp ∧ (j, 0) ∉ pl.shapeUp ∧
      pl.dropOffset b < b.colHeight (pl.col + j) := by
  rcases holes_pos_or_tuck_Z (b := b) hv hZ with hpos | htuck
  · exfalso; omega
  · exact htuck

/-! ### A tuck rests on a pre-existing board cell

The tuck dichotomy says a hole-free `S`/`Z` placement must rest its notch column
on a surface strictly above the drop offset. We now sharpen *which* cell it rests
on: the board cell directly beneath the notch — `(pl.col + j, dropOffset b) ∈ b`.
This is the form a counting/budget argument needs, since it injects each
hole-free tuck into a concrete pre-existing board cell at the resting plane. -/

/-- **Gapless column ⇒ every sub-height cell is filled.** If column `j` has no
buried cell (`colHoles b j = 0`) then it is gravity-packed: every row below the
column height is occupied. From `colHoles_eq_zero_iff` the filled rows have the
same cardinality as `range (colHeight j)` and embed into it, so the two coincide. -/
theorem mem_of_colHoles_zero {b : Board} {j r : ℕ}
    (hz : colHoles b j = 0) (hr : r < b.colHeight j) : (j, r) ∈ b := by
  have hcard : (b.colRows j).card = b.colHeight j := (colHoles_eq_zero_iff b j).mp hz
  have hsub : b.colRows j ⊆ Finset.range (b.colHeight j) := colRows_subset_range_colHeight b j
  have heq : b.colRows j = Finset.range (b.colHeight j) := by
    apply Finset.eq_of_subset_of_card_le hsub
    rw [Finset.card_range]; omega
  rw [← mem_colRows_iff, heq, Finset.mem_range]
  exact hr

/-- **A hole-free `S` placement rests on a board cell.** Sharpens
`tuck_of_holes_zero_S`: the notch column `pl.col + j` (body at `h > 0`, empty at
row `0`) has the board cell `(pl.col + j, pl.dropOffset b)` already filled. Since
the piece does not occupy that cell (it has no `(j, 0)`) and `pl.place b` is
gravity-packed there (hole-free), the base cell comes from `b` itself — each tuck
is supported by, and injects into, a pre-existing board cell. -/
theorem tuck_rests_on_cell_S {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hS : pl.piece = Piece.S)
    (hno : holes cfg (pl.place b) = 0) :
    ∃ j h : ℕ, 0 < h ∧ (j, h) ∈ pl.shapeUp ∧ (j, 0) ∉ pl.shapeUp ∧
      (pl.col + j, pl.dropOffset b) ∈ b := by
  obtain ⟨j, h, hpos, hmem, hzero, _⟩ := tuck_of_holes_zero_S hv hS hno
  refine ⟨j, h, hpos, hmem, hzero, ?_⟩
  have hrange : pl.col + j < cfg.cols := hv (j, h) hmem
  have hbodyMem : (pl.col + j, pl.dropOffset b + h) ∈ pl.place b := by
    rw [Placement.place_eq_union_dropped, Finset.mem_union]
    right
    rw [Placement.dropped_eq_image]
    exact Finset.mem_image.mpr ⟨(j, h), hmem, rfl⟩
  have hlt : pl.dropOffset b < (pl.place b).colHeight (pl.col + j) := by
    have := Board.lt_colHeight hbodyMem; omega
  have hcolz : colHoles (pl.place b) (pl.col + j) = 0 :=
    (holes_eq_zero_iff cfg (pl.place b)).mp hno (pl.col + j) (Finset.mem_range.mpr hrange)
  have hbase : (pl.col + j, pl.dropOffset b) ∈ pl.place b := mem_of_colHoles_zero hcolz hlt
  rw [Placement.place_eq_union_dropped, Finset.mem_union] at hbase
  rcases hbase with hb | hd
  · exact hb
  · exfalso
    rw [Placement.dropped_eq_image] at hd
    obtain ⟨⟨x, y⟩, hxy, heq⟩ := Finset.mem_image.mp hd
    simp only [Prod.mk.injEq] at heq
    obtain ⟨hx, hy⟩ := heq
    have hx' : x = j := by omega
    have hy' : y = 0 := by omega
    subst hx'; subst hy'
    exact hzero hxy

/-- **A hole-free `Z` placement rests on a board cell.** Z analog of
`tuck_rests_on_cell_S`. -/
theorem tuck_rests_on_cell_Z {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hZ : pl.piece = Piece.Z)
    (hno : holes cfg (pl.place b) = 0) :
    ∃ j h : ℕ, 0 < h ∧ (j, h) ∈ pl.shapeUp ∧ (j, 0) ∉ pl.shapeUp ∧
      (pl.col + j, pl.dropOffset b) ∈ b := by
  obtain ⟨j, h, hpos, hmem, hzero, _⟩ := tuck_of_holes_zero_Z hv hZ hno
  refine ⟨j, h, hpos, hmem, hzero, ?_⟩
  have hrange : pl.col + j < cfg.cols := hv (j, h) hmem
  have hbodyMem : (pl.col + j, pl.dropOffset b + h) ∈ pl.place b := by
    rw [Placement.place_eq_union_dropped, Finset.mem_union]
    right
    rw [Placement.dropped_eq_image]
    exact Finset.mem_image.mpr ⟨(j, h), hmem, rfl⟩
  have hlt : pl.dropOffset b < (pl.place b).colHeight (pl.col + j) := by
    have := Board.lt_colHeight hbodyMem; omega
  have hcolz : colHoles (pl.place b) (pl.col + j) = 0 :=
    (holes_eq_zero_iff cfg (pl.place b)).mp hno (pl.col + j) (Finset.mem_range.mpr hrange)
  have hbase : (pl.col + j, pl.dropOffset b) ∈ pl.place b := mem_of_colHoles_zero hcolz hlt
  rw [Placement.place_eq_union_dropped, Finset.mem_union] at hbase
  rcases hbase with hb | hd
  · exact hb
  · exfalso
    rw [Placement.dropped_eq_image] at hd
    obtain ⟨⟨x, y⟩, hxy, heq⟩ := Finset.mem_image.mp hd
    simp only [Prod.mk.injEq] at heq
    obtain ⟨hx, hy⟩ := heq
    have hx' : x = j := by omega
    have hy' : y = 0 := by omega
    subst hx'; subst hy'
    exact hzero hxy

/-- **Depth-2 hole floor on a closed witness.** Pushing the depth-1 floor
`exists_mem_closed_count_four_holes_pos` one adversarial draw deeper. In any closed
init-atlas the adversary plays `init →S→ g1 →Z→ g2`: `g1` has `count = 4` and at least
one hole (S on the empty board), and since `count(g1) = 4 < cols = 10` the subsequent
Z-placement clears no line, so holes are monotone (`holes_le_holes_place`) and
`g2` carries `count = 8` while retaining `≥ 1` hole. This doubles the constant count
floor (4 → 8) and is the natural ceiling of the no-clear monotone argument: `count = 4n`
stays below `cols` only for `n ≤ 2`. -/
theorem exists_mem_closed_count_eight_holes_pos {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, g.board.count = 8 ∧ 1 ≤ holes GameConfig.standard g.board := by
  have h5 : (5 : ℕ) ≤ GameConfig.standard.cols := by decide
  -- First draw: S on the empty initial board.
  have hSb : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  obtain ⟨_, hstep0⟩ := hclosed GameState.init hinit
  obtain ⟨plS, hplS, hsucc1⟩ := hstep0 Piece.S hSb
  rw [Placement.mem_allValidFor] at hplS
  obtain ⟨hpieceS, hvalidS⟩ := hplS
  set g1 := adversarialStep GameConfig.standard GameState.init Piece.S plS with hg1def
  have hplS_eq : ({plS with piece := Piece.S} : Placement) = plS := by
    rcases plS with ⟨pp, pr, pc⟩; cases hpieceS; rfl
  have hg1board : g1.board = plS.applyStep GameConfig.standard ∅ := by
    rw [hg1def, adversarialStep_board, GameState.init_board_eq_emptyset, hplS_eq]
  have hg1count : g1.board.count = 4 := by
    rw [hg1board, applyStep_empty_eq_place_empty h5, Placement.count_place]; rfl
  have hg1holes : 1 ≤ holes GameConfig.standard g1.board := by
    rw [hg1board]; exact one_le_holes_S_applyStep_empty h5 hvalidS hpieceS
  -- Second draw: Z is still in the bag (only S has been removed).
  have hZb : Piece.Z ∈ g1.bag := by
    rw [hg1def, adversarialStep_bag, GameState.init_bag, Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.Z⟩
  obtain ⟨_, hstep1⟩ := hclosed g1 hsucc1
  obtain ⟨plZ, hplZ, hsucc2⟩ := hstep1 Piece.Z hZb
  rw [Placement.mem_allValidFor] at hplZ
  obtain ⟨hpieceZ, hvalidZ⟩ := hplZ
  set g2 := adversarialStep GameConfig.standard g1 Piece.Z plZ with hg2def
  have hplZ_eq : ({plZ with piece := Piece.Z} : Placement) = plZ := by
    rcases plZ with ⟨pp, pr, pc⟩; cases hpieceZ; rfl
  have hg2board0 : g2.board = plZ.applyStep GameConfig.standard g1.board := by
    rw [hg2def, adversarialStep_board, hplZ_eq]
  -- count 4 < 10 ⇒ the Z placement clears no line ⇒ board = plZ.place g1.board.
  have hplaceCount : (plZ.place g1.board).count < GameConfig.standard.cols := by
    rw [Placement.count_place, hg1count]; decide
  have hg2board : g2.board = plZ.place g1.board := by
    rw [hg2board0, Placement.applyStep_eq_clearLines_place,
        Board.clearLines_eq_self_of_count_lt GameConfig.standard hplaceCount]
  refine ⟨g2, hsucc2, ?_, ?_⟩
  · rw [hg2board, Placement.count_place, hg1count]
  · rw [hg2board]; exact le_trans hg1holes (holes_le_holes_place _ _ _)

/-- **A hole-free `S` tuck strictly raises the column height at the notch column.**
Refining `tuck_rests_on_cell_S`: in a hole-free placement the notch column `pl.col+j`
carries the piece body cell `(pl.col+j, dropOffset+h)` (with `h > 0`), so the placed
board's height there is `≥ dropOffset+h+1`, while the *original* board's height there is
`≤ dropOffset+h` (`le_dropOffset_pair` at the body cell `(j,h)`). Hence the tuck strictly
*increases* `colHeight` at that column — tucking is not free, it spends bounded vertical
room (`colHeight ≤ rows` on a live board). This is the per-tuck seed of a height-Lyapunov:
forced repeated hole-free tucks pump a column upward toward top-out. -/
theorem colHeight_lt_of_tuck_S {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hS : pl.piece = Piece.S)
    (hno : holes cfg (pl.place b) = 0) :
    ∃ j : ℕ, b.colHeight (pl.col + j) < (pl.place b).colHeight (pl.col + j) := by
  obtain ⟨j, h, hpos, hmem, hzero, hrest⟩ := tuck_rests_on_cell_S hv hS hno
  refine ⟨j, ?_⟩
  have hub : b.colHeight (pl.col + j) ≤ pl.dropOffset b + h := by
    have hle := le_dropOffset_pair b pl hmem
    omega
  have hbodyMem : (pl.col + j, pl.dropOffset b + h) ∈ pl.place b := by
    rw [Placement.place_eq_union_dropped, Finset.mem_union]
    right
    rw [Placement.dropped_eq_image]
    exact Finset.mem_image.mpr ⟨(j, h), hmem, rfl⟩
  have hlb : pl.dropOffset b + h < (pl.place b).colHeight (pl.col + j) :=
    Board.lt_colHeight hbodyMem
  omega

/-- **A hole-free `Z` tuck strictly raises the column height at the notch column.**
Z analog of `colHeight_lt_of_tuck_S`. -/
theorem colHeight_lt_of_tuck_Z {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hZ : pl.piece = Piece.Z)
    (hno : holes cfg (pl.place b) = 0) :
    ∃ j : ℕ, b.colHeight (pl.col + j) < (pl.place b).colHeight (pl.col + j) := by
  obtain ⟨j, h, hpos, hmem, hzero, hrest⟩ := tuck_rests_on_cell_Z hv hZ hno
  refine ⟨j, ?_⟩
  have hub : b.colHeight (pl.col + j) ≤ pl.dropOffset b + h := by
    have hle := le_dropOffset_pair b pl hmem
    omega
  have hbodyMem : (pl.col + j, pl.dropOffset b + h) ∈ pl.place b := by
    rw [Placement.place_eq_union_dropped, Finset.mem_union]
    right
    rw [Placement.dropped_eq_image]
    exact Finset.mem_image.mpr ⟨(j, h), hmem, rfl⟩
  have hlb : pl.dropOffset b + h < (pl.place b).colHeight (pl.col + j) :=
    Board.lt_colHeight hbodyMem
  omega

/-- **Lift `colHeight_lt_of_tuck_S` through the full adversarial step on a no-clear board.**
When the placed board has fewer cells than the column count (`(pl.place b).count < cfg.cols`),
no row can be full, so `clearLines` is the identity and `applyStep = place`. Hence a hole-free
`S` tuck strictly raises `colHeight` at the notch column even after the *complete* step
(place + clear), not just after `place`. This carries the per-tuck height-growth fact onto the
actual `applyStep`/`adversarialStep` transition used in `IsClosedSet`, mirroring how the
depth-2 hole floor is depth-aware through `applyStep`. -/
theorem colHeight_lt_of_tuck_applyStep_S {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hS : pl.piece = Piece.S)
    (hcount : (pl.place b).count < cfg.cols)
    (hno : holes cfg (pl.place b) = 0) :
    ∃ j : ℕ, b.colHeight (pl.col + j) < (pl.applyStep cfg b).colHeight (pl.col + j) := by
  have happly : pl.applyStep cfg b = pl.place b := by
    rw [Placement.applyStep_eq_clearLines_place,
        Board.clearLines_eq_self_of_count_lt cfg hcount]
  rw [happly]
  exact colHeight_lt_of_tuck_S hv hS hno

/-- Z analog of `colHeight_lt_of_tuck_applyStep_S`. -/
theorem colHeight_lt_of_tuck_applyStep_Z {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hZ : pl.piece = Piece.Z)
    (hcount : (pl.place b).count < cfg.cols)
    (hno : holes cfg (pl.place b) = 0) :
    ∃ j : ℕ, b.colHeight (pl.col + j) < (pl.applyStep cfg b).colHeight (pl.col + j) := by
  have happly : pl.applyStep cfg b = pl.place b := by
    rw [Placement.applyStep_eq_clearLines_place,
        Board.clearLines_eq_self_of_count_lt cfg hcount]
  rw [happly]
  exact colHeight_lt_of_tuck_Z hv hZ hno

/-- **Depth-2 `count + holes` floor for any init-containing closed set.** Sharpens
`exists_mem_closed_five_le_count_add_holes` (depth-1, `≥ 5`) to `≥ 9`: forcing the
adversarial pair `S` then `Z` from `init` (the full no-clear window, since the placed
`count = 8 < cols = 10`) lands on a state with `count = 8` AND at least one hole
(`exists_mem_closed_count_eight_holes_pos`, iter-131), so `count + holes ≥ 9`. This is the
tightest unconditional `Φ = count + holes` floor the no-clear forcing can reach (the window
shuts at `count = 4n < cols = 10`, i.e. `n ≤ 2`), pinning the FALSE-budget gap to `[9, 200]`. -/
theorem exists_mem_closed_nine_le_count_add_holes {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, 9 ≤ g.board.count + holes GameConfig.standard g.board := by
  obtain ⟨g, hg, hc, hh⟩ := exists_mem_closed_count_eight_holes_pos hinit hclosed
  exact ⟨g, hg, by omega⟩

/-- **Bag-variety hole avoidance: `O` has a hole-free continuation on the empty board.**
The exact counterweight to `one_le_holes_S_place_empty` / `one_le_holes_Z_place_empty`:
whereas `S` and `Z` *force* `≥ 1` buried cell from `init`, the player handed `O` can place it
flat — the `O` drop profile `{(0,0),(0,1),(1,0),(1,1)}` is fully *supported* (every occupied
column has a row-`0` cell) — landing on `holes = 0`. Since every 7-bag contains exactly one `O`,
the adversary cannot force a hole on every step, so the no-clear depth-floor (which compounds
forced `S`/`Z` holes) cannot be extended through an `O` draw. This is the structural seed of the
TRUE verdict (7-bag survivability): the player always has a hole-free option for at least one
bag member. Witness: `O` at column `0`, rotation `0`; membership and `holes = 0` by `decide`. -/
theorem exists_holes_zero_O_place_empty :
    ∃ pl ∈ Placement.allValidFor GameConfig.standard Piece.O,
      holes GameConfig.standard (pl.place ∅) = 0 := by
  refine ⟨⟨Piece.O, 0, 0⟩, ?_, ?_⟩
  · exact (Placement.mem_allValidFor GameConfig.standard Piece.O _).mpr ⟨rfl, by decide⟩
  · decide

/-- **Bag-variety hole avoidance, `I` piece.** Companion to `exists_holes_zero_O_place_empty`:
the player handed `I` on the empty board can lay it flat (rotation `0`, drop profile
`{(0,0),(1,0),(2,0),(3,0)}` — a single fully supported row) landing on `holes = 0`. Together
with the `O` companion, *two* of the seven bag members admit hole-free play directly on `init`;
only `S` and `Z` force a buried cell (`one_le_holes_S_place_empty` / `one_le_holes_Z_place_empty`).
So the adversary forcing a hole at depth 1 must specifically draw `S` or `Z`, not merely "some
piece". Witness: `I` at column `0`, rotation `0`; membership and `holes = 0` by `decide`. -/
theorem exists_holes_zero_I_place_empty :
    ∃ pl ∈ Placement.allValidFor GameConfig.standard Piece.I,
      holes GameConfig.standard (pl.place ∅) = 0 := by
  refine ⟨⟨Piece.I, 0, 0⟩, ?_, ?_⟩
  · exact (Placement.mem_allValidFor GameConfig.standard Piece.I _).mpr ⟨rfl, by decide⟩
  · decide

/-- **Clear-collapse: a single line-clear can lower a column's height by far more than the
number of rows it removes.** Concretely, there is a config, board, and column where
`(clearLines b).colHeight j + (fullRows b).card < b.colHeight j` — i.e. the height drop
caused by clearing *strictly exceeds* `(fullRows).card`, the count of cleared rows.

Witness: a `1`-wide, `4`-tall board with cells `{(0,1),(0,3)}`. Both occupied rows are full
(width 1), so `clearLines` empties the column: `colHeight 0` falls `4 → 0`, a drop of `4`,
while only `2` rows were full. Hence `0 + 2 < 4`.

**Why this matters for the verdict.** It formally refutes the natural FALSE-route hypothesis
"a clear lowers `colHeight j` by at most `(fullRows).card`" — the bound any *summed* height
potential `Φ = ∑ colHeight` would need to be clear-robust (so that `Φ` net-drifts downward
across the full adversarial place+clear step and eventually forces a top-out). A tall sparse
column collapses by its *entire* height when its few occupied rows happen to be full, so no
summed-`colHeight`/`Φ` Lyapunov function is clear-robust. This prunes the height-Lyapunov
FALSE sub-route, narrowing the open verdict toward the remaining live engines (the
count+holes budget floor and the TRUE-route closed-Finset witness). -/
theorem exists_clearLines_colHeight_drop_gt_fullRows :
    ∃ (cfg : GameConfig) (b : Board) (j : ℕ),
      (Board.clearLines cfg b).colHeight j + (Board.fullRows cfg b).card < b.colHeight j := by
  refine ⟨⟨1, 4, by decide, by decide⟩, {(0, 1), (0, 3)}, 0, by decide⟩

/-- **Clears can remove holes: a single line-clear can strictly lower the total hole count.**
There is a config and board with `holes (clearLines b) < holes b` — clearing a full row
*reduces* the buried-cell budget rather than leaving it fixed.

Witness: a `2`-wide, `4`-tall board `{(0,1),(1,1),(0,3)}`. Row `1` is full (both columns
occupied) and is cleared; the lone cell `(0,3)` drops to `(0,2)`. Hole accounting:
before, column `0` has cells at rows `{1,3}` (holes at rows `0,2`, so `colHoles = 2`) and
column `1` has a cell at row `1` (hole at row `0`, `colHoles = 1`), total `holes = 3`;
after, the board is `{(0,2)}` — column `0` holes `0,1` give `colHoles = 2`, column `1`
empty — total `holes = 2`. Hence `2 < 3`.

**Why this matters for the verdict.** Together with `exists_clearLines_colHeight_drop_gt_fullRows`
(clears collapse *height* non-monotonically), this shows the *holes* potential is likewise
non-monotone under the full place+clear step: placement can bury cells (`S`/`Z` force `+1`
hole, `one_le_holes_S/Z_place_empty`) but a clear can *remove* holes. So the count+holes
budget does **not** monotonically climb toward the `200` ceiling — the only surviving FALSE
engine must quantitatively out-race hole *removal* by clears, not merely exhibit hole
*creation*. This is the formal statement of the genuine open crux ("can clears remove holes
as fast as `S`/`Z` create them?"): the answer is *clears do remove holes*, so a naive
monotone-budget FALSE argument is blocked. It tilts the open verdict toward TRUE without
closing it. -/
theorem exists_holes_clearLines_lt :
    ∃ (cfg : GameConfig) (b : Board),
      holes cfg (Board.clearLines cfg b) < holes cfg b := by
  refine ⟨⟨2, 4, by decide, by decide⟩, {(0, 1), (1, 1), (0, 3)}, by decide⟩

/-- **Hole-removal rate is unbounded by rows cleared: one line-clear can remove strictly
more holes than the number of rows it clears.** There is a config and board with
`holes (clearLines b) + (fullRows b).card < holes b` — i.e. the *drop* in total holes
strictly exceeds `(fullRows).card`, so a single cleared row eliminates *several* holes.

Witness: a `2`-wide, `4`-tall board `{(0,0),(0,3),(1,3)}`. Only row `3` is full (both
columns), so `(fullRows).card = 1`; clearing it removes `(0,3)` and `(1,3)` and drops the
lone survivor `(0,0)` nowhere (it is below the cut). Hole accounting: before, column `0`
spans rows `{0,3}` with holes at `1,2` (`colHoles = 2`) and column `1` has a single cell at
row `3` with holes at `0,1,2` (`colHoles = 3`), total `holes = 5`; after, the board is
`{(0,0)}` with `holes = 0`. Hence `0 + 1 < 5` — one clear removed *five* holes.

**Why this matters for the verdict.** This strengthens `exists_holes_clearLines_lt`
(iter-138, clears remove *some* holes) to a *rate* statement: the per-clear hole removal is
**not** bounded by `(fullRows).card`. A tall sparse column whose lone supporting cell sits in
the cleared row collapses entirely, vaporizing every hole beneath it. So the natural FALSE
rate hypothesis "a clear removes at most `(fullRows).card` holes" is false — clears are an
arbitrarily strong hole sink. Any surviving FALSE (budget-divergence) engine must therefore
show `S`/`Z` *out-create* an unboundedly-effective hole remover, not merely beat a `1`-per-row
removal rate. Combined with the height-collapse witness (`exists_clearLines_colHeight_drop_gt_fullRows`)
this removes the last cheap monotone-potential FALSE route; the verdict stays TRUE-leaning,
open. -/
theorem exists_holes_clearLines_add_fullRows_card_lt :
    ∃ (cfg : GameConfig) (b : Board),
      holes cfg (Board.clearLines cfg b) + (Board.fullRows cfg b).card < holes cfg b := by
  refine ⟨⟨2, 4, by decide, by decide⟩, {(0, 0), (0, 3), (1, 3)}, by decide⟩

/-- **Per-column cell-count after a clear: every valid column loses exactly `(fullRows).card`
cells.** For `j < cfg.cols`, `(b.colRows j).card = ((clearLines b).colRows j).card + (fullRows b).card`.

This is the *row-set* (`colRows`) restatement of the cell-count law `colCount_clearLines_add`,
obtained because `(colRows j).card = (b.filter (·.1 = j)).card = colCount b j`
(`colRows_card_eq_filter_card`, the `(·.2)` projection being injective on a fixed-column fiber).
Each of the `k = (fullRows).card` cleared rows is full, hence contributes exactly one cell to
column `j`, so clearing them deletes exactly `k` cells from every valid column.

**Role toward the verdict (KEYSTONE step a).** The open FALSE engine needs the total-hole budget
`holes = ∑ⱼ (colHeight j − (colRows j).card)` to be clear-robust. This lemma pins the *subtracted*
term's behavior exactly: the `(colRows j).card` summand drops by precisely `k` per column under a
clear. The companion fact still needed for the full monotone-removal keystone
`holes (clearLines b) ≤ holes b` is that `colHeight j` drops by *at least* `k` per column (a clear
lowers each column's top by ≥ the number of full rows beneath it); combined with this card lemma it
yields per-column `colHoles` non-increase. Proving the card half exactly (no inequality slack) is the
load-bearing arithmetic; the height half is a pure order argument on the surviving top cell. -/
theorem colRows_card_clearLines_add (cfg : GameConfig) (b : Board) {j : ℕ}
    (hj : j < cfg.cols) :
    (b.colRows j).card
      = ((Board.clearLines cfg b).colRows j).card + (Board.fullRows cfg b).card := by
  rw [colRows_card_eq_filter_card b j,
      colRows_card_eq_filter_card (Board.clearLines cfg b) j]
  exact Board.colCount_clearLines_add cfg b hj

/-- **Height half of the monotone-removal keystone: a clear lowers each valid column's height by
at least `(fullRows).card`.** For `j < cfg.cols`,
`(clearLines b).colHeight j + (fullRows b).card ≤ b.colHeight j`.

Strengthens `colHeight_clearLines_le` (the non-strict bound in `Theorems/GameplayExtra.lean`) with the
exact `k = (fullRows).card` slack. Proof: every cleared (full) row `r` has a cell in column `j`
(`j < cols`), so all `k` cleared rows lie strictly below the column top; gravity sends a surviving
non-full cell at row `s` to `s − clearedBelow s`, and the full rows in `(s, colHeight)` number at most
`colHeight − s − 1` (they embed in `Finset.Ioo s (colHeight)`), forcing
`clearedBelow s ≥ k − (colHeight − 1 − s)`. Hence every surviving cell lands at height
`≤ colHeight − k`, and `k ≤ colHeight` (the `k` full rows embed in `range (colHeight)`). -/
theorem colHeight_clearLines_add_le (cfg : GameConfig) (b : Board) {j : ℕ} (hj : j < cfg.cols) :
    (Board.clearLines cfg b).colHeight j + (Board.fullRows cfg b).card ≤ b.colHeight j := by
  have hk_le : (Board.fullRows cfg b).card ≤ b.colHeight j := by
    have hsub : Board.fullRows cfg b ⊆ Finset.range (b.colHeight j) := by
      intro r hr
      simp only [Board.fullRows, Finset.mem_filter] at hr
      have hjr : (j, r) ∈ b := hr.2 j (Finset.mem_range.mpr hj)
      exact Finset.mem_range.mpr (Board.lt_colHeight hjr)
    calc (Board.fullRows cfg b).card
        ≤ (Finset.range (b.colHeight j)).card := Finset.card_le_card hsub
      _ = b.colHeight j := Finset.card_range _
  have key : ∀ y, (j, y) ∈ Board.clearLines cfg b →
      y + 1 + (Board.fullRows cfg b).card ≤ b.colHeight j := by
    intro y hy
    simp only [Board.clearLines, Finset.mem_image, Finset.mem_filter] at hy
    obtain ⟨p, ⟨hpb, hpf⟩, hpe⟩ := hy
    rw [Prod.mk.injEq] at hpe
    obtain ⟨hp1, hp2⟩ := hpe
    have hjs : (j, p.2) ∈ b := by rw [← hp1]; exact hpb
    have hs_lt : p.2 < b.colHeight j := Board.lt_colHeight hjs
    have hcb_le_row : Board.clearedBelow cfg b p.2 ≤ p.2 := Board.clearedBelow_le_row cfg b p.2
    have hpart : Board.clearedBelow cfg b p.2
        + ((Board.fullRows cfg b).filter (fun r => ¬ r < p.2)).card
        = (Board.fullRows cfg b).card := by
      rw [show Board.clearedBelow cfg b p.2
            = ((Board.fullRows cfg b).filter (· < p.2)).card from rfl]
      exact Finset.filter_card_add_filter_neg_card_eq_card _
    have hsub : (Board.fullRows cfg b).filter (fun r => ¬ r < p.2)
        ⊆ Finset.Ioo p.2 (b.colHeight j) := by
      intro r hr
      simp only [Finset.mem_filter, not_lt] at hr
      obtain ⟨hrfull, hrge⟩ := hr
      have hisfull : Board.isFull cfg b r := by
        simp only [Board.fullRows, Finset.mem_filter] at hrfull
        exact hrfull.2
      have hne : r ≠ p.2 := by
        intro h; rw [h] at hisfull; exact hpf hisfull
      have hr_gt : p.2 < r := lt_of_le_of_ne hrge (Ne.symm hne)
      have hjr : (j, r) ∈ b := hisfull j (Finset.mem_range.mpr hj)
      exact Finset.mem_Ioo.mpr ⟨hr_gt, Board.lt_colHeight hjr⟩
    have hcard_le : ((Board.fullRows cfg b).filter (fun r => ¬ r < p.2)).card
        ≤ b.colHeight j - p.2 - 1 := by
      calc ((Board.fullRows cfg b).filter (fun r => ¬ r < p.2)).card
          ≤ (Finset.Ioo p.2 (b.colHeight j)).card := Finset.card_le_card hsub
        _ = b.colHeight j - p.2 - 1 := Nat.card_Ioo _ _
    omega
  have hsup_le : (Board.clearLines cfg b).colHeight j
      ≤ b.colHeight j - (Board.fullRows cfg b).card := by
    change (Board.colRows (Board.clearLines cfg b) j).sup (· + 1)
        ≤ b.colHeight j - (Board.fullRows cfg b).card
    apply Finset.sup_le
    intro x hx
    simp only [Board.colRows, Finset.mem_image, Finset.mem_filter] at hx
    obtain ⟨cell, ⟨hcl, hcj⟩, hcx⟩ := hx
    have hjx : (j, x) ∈ Board.clearLines cfg b := by
      have hce : cell = (j, x) := Prod.ext_iff.mpr ⟨hcj, hcx⟩
      rw [← hce]; exact hcl
    have := key x hjx
    omega
  omega

/-- **Per-column hole non-increase under a clear.** For `j < cfg.cols`,
`colHoles (clearLines b) j ≤ colHoles b j`. Immediate from the two halves: the card term drops by
exactly `k` (`colRows_card_clearLines_add`) while the height term drops by at least `k`
(`colHeight_clearLines_add_le`), so `colHoles = colHeight − card` cannot rise. -/
theorem colHoles_clearLines_le (cfg : GameConfig) (b : Board) {j : ℕ} (hj : j < cfg.cols) :
    colHoles (Board.clearLines cfg b) j ≤ colHoles b j := by
  have hA := colRows_card_clearLines_add cfg b hj
  have hB := colHeight_clearLines_add_le cfg b hj
  simp only [colHoles]
  omega

/-- **THE MONOTONE-REMOVAL KEYSTONE: line clears never increase the total hole count.**
`holes (clearLines b) ≤ holes b`, with no well-formedness hypothesis. Summing the per-column
non-increase `colHoles_clearLines_le` over `range cfg.cols` via `Finset.sum_le_sum`.

**Verdict significance.** This is the lemma the FALSE (budget-divergence) route needed and could not
have: it proves the *only* hole source is placement (a clear is a pure hole sink). Combined with
`count_add_holes_le_two_hundred_of_reachableAt` (count+holes ≤ 200 on reachable boards) and the
per-step placement hole-creation bounds, the total-hole budget is now framed as a contest between
bounded creation (≤ a few per placed piece, capped by 7-bag S/Z frequency) and unbounded removal
(iter-139). The naive "holes climb monotonically to the `200` ceiling" FALSE argument is now formally
dead at the source: holes can only climb via placements, and every clear claws them back. The verdict
stays TRUE-leaning; a FALSE proof would require a *quantitative* creation-beats-removal rate that the
7-bag cap appears to forbid. -/
theorem holes_clearLines_le (cfg : GameConfig) (b : Board) :
    holes cfg (Board.clearLines cfg b) ≤ holes cfg b := by
  unfold holes
  apply Finset.sum_le_sum
  intro j hj
  exact colHoles_clearLines_le cfg b (Finset.mem_range.mp hj)

/-- **Clear-side per-step hole bound.** `applyStep` is a hard drop followed by a
line clear, and the monotone-removal keystone (`holes_clearLines_le`) says the
clear never adds holes. Hence the full step's hole count never exceeds the
*placed* board's hole count — unconditionally, with no no-clear hypothesis. This
is the general companion to `holes_le_holes_applyStep_of_no_fullRows` (which only
bounds the no-clear regime): together with the place-side creation bound it pins
the hole arithmetic of a single adversarial move. -/
theorem holes_applyStep_le_holes_place (cfg : GameConfig) (b : Board) (pl : Placement) :
    holes cfg (pl.applyStep cfg b) ≤ holes cfg (pl.place b) := by
  rw [Placement.applyStep_eq_clearLines_place]
  exact holes_clearLines_le cfg (pl.place b)

/-- **Adversarial-walk form of the clear-side hole bound.** The directly chainable
version along `init → S → Z → …`: one adversary move's resulting hole count never
exceeds the hole count of the board after the placement (before the clear). -/
theorem holes_adversarialStep_le_holes_place (cfg : GameConfig) (g : GameState)
    (p : Piece) (pl : Placement) :
    holes cfg (adversarialStep cfg g p pl).board
      ≤ holes cfg (({pl with piece := p} : Placement).place g.board) := by
  rw [adversarialStep_board]
  exact holes_applyStep_le_holes_place cfg g.board ({pl with piece := p})

/-- **Bag-variety hole avoidance, `T` piece.** Like `O`/`I`, the player handed `T` on the
empty board has a hole-free placement (the flat `⊥`-orientation lays three cells on row `0`
with one supported bump). Existence is decidable over the finite valid-placement set. -/
theorem exists_holes_zero_T_place_empty :
    ∃ pl ∈ Placement.allValidFor GameConfig.standard Piece.T,
      holes GameConfig.standard (pl.place ∅) = 0 := by decide

/-- **Bag-variety hole avoidance, `L` piece.** Companion to the `O`/`I`/`T` lemmas: `L` has a
hole-free placement on the empty board (a flat orientation, fully supported on row `0`). -/
theorem exists_holes_zero_L_place_empty :
    ∃ pl ∈ Placement.allValidFor GameConfig.standard Piece.L,
      holes GameConfig.standard (pl.place ∅) = 0 := by decide

/-- **Bag-variety hole avoidance, `J` piece.** Completes the non-`S`/`Z` catalogue: `J` too
has a hole-free placement on the empty board. -/
theorem exists_holes_zero_J_place_empty :
    ∃ pl ∈ Placement.allValidFor GameConfig.standard Piece.J,
      holes GameConfig.standard (pl.place ∅) = 0 := by decide

/-- **Sharp player-side existence on `init`: exactly `S` and `Z` force a hole.** Every piece
other than `S`/`Z` admits a hole-free placement on the empty board. Paired with
`one_le_holes_S_place_empty` / `one_le_holes_Z_place_empty` (which force `≥ 1` buried cell for
`S`/`Z`), this is the exact depth-1 dichotomy: *five of seven* bag members can be played with
`holes = 0`, and only the two skew pieces are forced to bury a cell. Because every 7-bag delivers
each non-skew piece, the adversary cannot force a hole on consecutive steps — the structural
engine behind the TRUE-leaning verdict (the player always retains a hole-free option except when
specifically handed `S` or `Z`, whose lone forced hole a later clear can reclaim). -/
theorem exists_holes_zero_place_empty_of_ne_S_Z {p : Piece}
    (hS : p ≠ Piece.S) (hZ : p ≠ Piece.Z) :
    ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
      holes GameConfig.standard (pl.place ∅) = 0 := by
  cases p with
  | O => exact exists_holes_zero_O_place_empty
  | I => exact exists_holes_zero_I_place_empty
  | S => exact absurd rfl hS
  | Z => exact absurd rfl hZ
  | T => exact exists_holes_zero_T_place_empty
  | L => exact exists_holes_zero_L_place_empty
  | J => exact exists_holes_zero_J_place_empty

/-- **A universal hole-free surface: the single-cell board `{(2,0)}` admits a hole-free
placement of *every* piece.** On the empty board only `O,I,T,L,J` can be played hole-free
(`exists_holes_zero_place_empty_of_ne_S_Z`); the two skew pieces `S,Z` are forced to bury a
cell on any flat surface. A single block at column `2` breaks the flatness into a step, and
that *one* notch is enough for **both** `S` and `Z` to tuck in hole-free (each skew piece can
exploit the up/down step from the appropriate side), while the remaining flat columns keep
`O,I,T,L,J` hole-free. Verified by `decide` over the finite valid-placement set for each of
the seven pieces.

**Why this matters for the verdict.** It refutes the intuition that the skew pieces are an
*unavoidable* hole liability: there is a concrete, tiny surface on which the player has a
hole-free response to whatever the adversary draws. This is the surface-shape seed of the
TRUE verdict — survival does not require ever taking a hole, provided the player maintains a
notched (non-flat) surface. The remaining gap to a closed atlas is *regeneration*: showing
such a surface (or a bounded family of them) recurs after the hole-free placements and any
clears. -/
theorem holeFree_all_pieces_singleton :
    ∀ p : Piece, ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
      holes GameConfig.standard (pl.place ({(2, 0)} : Board)) = 0 := by decide

/-- **Existence form of the universal hole-free surface.** There is a board on which every
piece has a hole-free placement — the existential wrapper around `holeFree_all_pieces_singleton`
(witness `{(2,0)}`). Instantiated at `S`/`Z` it certifies that the skew pieces admit hole-free
play on a suitable surface, the exact converse of `one_le_holes_S/Z_place_empty`. -/
theorem exists_universal_holeFree_surface :
    ∃ b : Board, ∀ p : Piece, ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
      holes GameConfig.standard (pl.place b) = 0 :=
  ⟨{(2, 0)}, holeFree_all_pieces_singleton⟩

/-- **The universal hole-free surface is a position-independent family: *every* bottom-row
single-cell notch `{(c,0)}`, `c < cols`, admits a hole-free placement of all seven pieces.**
Iteration 144 established this for the specific notch at column `2`; an `#eval` sweep showed
the property is invariant under the notch column, so it is proved here for the whole range
`c ∈ {0,…,9}` by `decide`. Floating cells (`{(c,1)}`) and height-`2` columns (`{(c,0),(c,1)}`)
are *not* universal — what matters is a single bottom-supported step, anywhere across the row.

**Why this matters for the verdict.** A line clear shifts surviving cells straight down and can
relocate the surface step to a different column; this family shows the universal-hole-free
property survives any such horizontal relocation of the notch. It is the *invariant family* a
closed atlas would have to live inside: the set of bounded notched surfaces, closed under the
adversary's draw because each member has a hole-free response. The remaining gap is purely
quantitative closure — by the cell-count parity `4·(placements) = 10·(rows cleared)`, no
*single*-step cycle exists (a placement adds `4`, a clear removes `10·k`), so regeneration of a
family member requires a multi-piece run (minimally `5` placements clearing `2` rows). -/
theorem holeFree_all_pieces_bottom_notch :
    ∀ c ∈ Finset.range GameConfig.standard.cols, ∀ p : Piece,
      ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
        holes GameConfig.standard (pl.place ({(c, 0)} : Board)) = 0 := by decide

/-- **Per-column usable form of the bottom-row notch family.** For any column `c < cols`, the
notch `{(c,0)}` is a universal hole-free surface. Extracted from
`holeFree_all_pieces_bottom_notch` by range membership for direct downstream instantiation. -/
theorem holeFree_bottom_notch_of_lt {c : ℕ} (hc : c < GameConfig.standard.cols) :
    ∀ p : Piece, ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
      holes GameConfig.standard (pl.place ({(c, 0)} : Board)) = 0 :=
  holeFree_all_pieces_bottom_notch c (Finset.mem_range.mpr hc)

/-- **The universal hole-free surface is NOT one-step closed: the adversary breaks it with `S`.**
On the universal surface `{(2,0)}` (iter 144: all seven pieces have a hole-free placement),
consider the adversary drawing `S`. There are exactly four hole-free `S` placements, and an
`#eval` sweep showed that **none** of them lands on a board that is again universal-hole-free —
every hole-free `S` response leaves a successor on which *some* piece has no hole-free placement.
This theorem certifies that fact by `decide`: for every hole-free `S` placement `pl` on `{(2,0)}`,
there is a witnessing piece `p` (the adversary's *next* draw) such that **every** valid placement
of `p` on the successor `pl.place {(2,0)}` creates a hole.

**Why this matters for the verdict.** The cheapest TRUE route is the strategy "stay on a
universal-hole-free surface forever" — every member of the notch family
(`holeFree_all_pieces_bottom_notch`) answers every draw without a hole. This theorem kills that
route: the family is *not* invariant under one adversarial step. From `{(2,0)}` the adversary
plays `S` and the player is forced to either (i) accept a hole now, or (ii) leave the
universal-surface family (a successor where a future draw can force a hole). The
universal-hole-free property alone is therefore not a closed certificate — consistent with the
Rust GFP search collapsing to the empty safe set. A TRUE verdict, if it holds, must live in a
*larger* invariant that tolerates transient holes / non-universal surfaces and relies on line
clears (the 7-bag bound caps `S`,`Z` at one per window) to recover. -/
theorem univHoleFree_surface_not_closed_under_S :
    ∀ pl ∈ Placement.allValidFor GameConfig.standard Piece.S,
      holes GameConfig.standard (pl.place ({(2, 0)} : Board)) = 0 →
        ∃ p : Piece, ∀ pl' ∈ Placement.allValidFor GameConfig.standard p,
          holes GameConfig.standard (pl'.place (pl.place ({(2, 0)} : Board))) ≠ 0 := by decide

/-- **Depth-2 forced hole in the no-clear regime: `S` then a forcing piece breaks the universal
surface without any line clear available to repair it.** Strengthens
`univHoleFree_surface_not_closed_under_S`. Start from the universal notch `{(2,0)}` (count `1`).
For every hole-free `S` placement `pl`:

* the successor `pl.place {(2,0)}` has count `5 < cols = 10` (a tetromino adds exactly four cells
  with no overlap, and `{(2,0)}` had one) — so **no row can be full**, no line clears; and
* there is a forcing piece `p` such that **every** valid placement `pl'` of `p` leaves a board that
  is *still* sub-full (count `9 < 10`, again no clears) yet carries at least one hole.

So across the entire two-ply window `{(2,0)} --S--> · --p--> ·` the cell count never reaches `cols`,
hence `clearLines` is the identity at every step (`clearLines_eq_self_of_count_lt`) and the forced
hole **cannot be cleared away** — it is a permanent `+1` to `holes`.

**Why this matters for the verdict.** A forced hole is only dangerous if it cannot be erased; line
clears are the player's sole hole-removal tool. This theorem shows the universal surface fails after
two adversarial draws *in the regime where clears are unavailable*, so the hole sticks. It isolates
the genuine FALSE mechanism — accumulation of unclearable holes — from the confound of line clears,
and is the depth-2 base case for a forced-hole chain. The remaining gap to a full FALSE verdict is to
show the adversary can re-enter such a no-clear forcing configuration indefinitely (the `7`-bag
renews one `S` and one `Z` per window), driving `count + holes` past the `200` budget of
`count_add_holes_le_two_hundred_of_reachableAt`. -/
theorem forced_hole_depth_two_no_clear_from_S :
    ∀ pl ∈ Placement.allValidFor GameConfig.standard Piece.S,
      holes GameConfig.standard (pl.place ({(2, 0)} : Board)) = 0 →
        (pl.place ({(2, 0)} : Board)).count < GameConfig.standard.cols ∧
        ∃ p : Piece, ∀ pl' ∈ Placement.allValidFor GameConfig.standard p,
          (pl'.place (pl.place ({(2, 0)} : Board))).count < GameConfig.standard.cols ∧
          1 ≤ holes GameConfig.standard (pl'.place (pl.place ({(2, 0)} : Board))) := by decide

/-- **The depth-2 forced hole, stated for the real game step `Placement.applyStep`.** Iteration 147
phrased the forced hole with bare `place`; this lifts it to the actual adversarial transition
(`applyStep` = `clearLines ∘ place`). The lift is *structural, not brute force*: on every hole-free
`S` response the forcing successor is sub-full (count `9 < cols`), so `clearLines` is the identity
there (`clearLines_eq_self_of_count_lt`) and `applyStep` collapses to `place`. Hence: for every
hole-free `S` placement `pl` on `{(2,0)}`, the adversary has a draw `p` such that **every**
placement `pl'` of `p`, run through the *full* step `applyStep`, leaves at least one hole.

We deliberately avoid a clear-active `decide` here. An `#eval` sweep confirms the stronger depth-3
fact — across `{(2,0)} --S--> · --p2--> · --p3--> ·` (where ply 3 reaches count `13 ≥ cols`, so a
line clear *can* fire) the adversary still keeps a hole through the clear, i.e. the dual "player
drives holes back to `0`" is `false`. But that statement, reduced through `clearLines` inside five
nested quantifiers, is infeasible for the *kernel* `decide` (it timed out at `4·10⁶` heartbeats),
and `native_decide` is barred from base-axiom theorems. The depth-3 stickiness across a clear is
therefore deferred to a *structural* proof (a hole's burying cell is never in a full row, so a
clear cannot un-bury it), tracked as the next keystone. This depth-2 `applyStep` form is the part
provable cheaply and is the real-game base case for that chain. -/
theorem forced_hole_depth_two_applyStep_from_S :
    ∀ pl ∈ Placement.allValidFor GameConfig.standard Piece.S,
      holes GameConfig.standard (pl.place ({(2, 0)} : Board)) = 0 →
        ∃ p : Piece, ∀ pl' ∈ Placement.allValidFor GameConfig.standard p,
          1 ≤ holes GameConfig.standard
            (Placement.applyStep GameConfig.standard (pl.place ({(2, 0)} : Board)) pl') := by
  intro pl hpl hhf
  obtain ⟨_hc1, p, hp⟩ := forced_hole_depth_two_no_clear_from_S pl hpl hhf
  refine ⟨p, fun pl' hpl' => ?_⟩
  obtain ⟨hcount', hholes⟩ := hp pl' hpl'
  rw [Placement.applyStep_eq_clearLines_place,
    Board.clearLines_eq_self_of_count_lt GameConfig.standard hcount']
  exact hholes

set_option maxRecDepth 4000 in
set_option maxHeartbeats 1000000 in
/-- **The S obstruction generalizes to the entire left half-board.** Iteration 146 proved that on
the single notch `{(2,0)}` the universal-hole-free surface is *not* one-step closed under `S`: every
hole-free `S` placement lands on a board where some adversary draw `p` admits *no* hole-free reply.
An `#eval` sweep of all ten bottom notches shows this is not special to column 2 — it holds for
**every** notch in the left half `c ∈ {0,1,2,3}` (and, by the S/Z reflection below, fails for `c ≥ 4`,
where the player can stay universal one more step). All boards here are sub-full (count `1 → 5 → 9 <
cols`), so this is a pure no-clear `decide`. -/
theorem univHoleFree_left_notches_not_closed_under_S :
    ∀ c ∈ Finset.range 4,
      ∀ pl ∈ Placement.allValidFor GameConfig.standard Piece.S,
        holes GameConfig.standard (pl.place ({(c, 0)} : Board)) = 0 →
          ∃ p : Piece, ∀ pl' ∈ Placement.allValidFor GameConfig.standard p,
            holes GameConfig.standard (pl'.place (pl.place ({(c, 0)} : Board))) ≠ 0 := by decide

set_option maxRecDepth 4000 in
set_option maxHeartbeats 1000000 in
/-- **The mirror obstruction: `Z` breaks the universal-hole-free surface on the right half-board.**
The `S`/`Z` reflection swaps left and right: where `S` kills the left notches `{0,1,2,3}`, `Z` kills
the right notches `{6,7,8,9}`. For every such notch, every hole-free `Z` placement lands on a board
where some adversary draw admits no hole-free reply. (Columns `4,5` are broken by neither single
piece — the player survives them universally for one more step.) Pure no-clear `decide`: every board
is sub-full (count `≤ 9`). -/
theorem univHoleFree_right_notches_not_closed_under_Z :
    ∀ c ∈ Finset.Icc 6 9,
      ∀ pl ∈ Placement.allValidFor GameConfig.standard Piece.Z,
        holes GameConfig.standard (pl.place ({(c, 0)} : Board)) = 0 →
          ∃ p : Piece, ∀ pl' ∈ Placement.allValidFor GameConfig.standard p,
            holes GameConfig.standard (pl'.place (pl.place ({(c, 0)} : Board))) ≠ 0 := by decide

set_option maxRecDepth 4000 in
set_option maxHeartbeats 1000000 in
/-- **The depth-2 forced hole, generalized to the entire left half-board (no-clear form).**
Iteration 147 proved that on the single notch `{(2,0)}` every hole-free `S` placement admits an
adversary draw `p` forcing a hole — and that this hole is *unclearable* because the whole 2-ply
window stays sub-full (`count 1 → 5 → 9 < cols`, so `clearLines` is the identity throughout). This
generalizes that to **every** left notch `c ∈ {0,1,2,3}` (the columns where `S` breaks universality,
per iteration 149's `#eval` sweep). For each such notch and each hole-free `S` placement, the
adversary forces a board on which *every* reply has `count < cols` and at least one hole. Pure
no-clear `decide`. -/
theorem forced_hole_depth_two_no_clear_left_S :
    ∀ c ∈ Finset.range 4,
      ∀ pl ∈ Placement.allValidFor GameConfig.standard Piece.S,
        holes GameConfig.standard (pl.place ({(c, 0)} : Board)) = 0 →
          ∃ p : Piece, ∀ pl' ∈ Placement.allValidFor GameConfig.standard p,
            (pl'.place (pl.place ({(c, 0)} : Board))).count < GameConfig.standard.cols ∧
            1 ≤ holes GameConfig.standard (pl'.place (pl.place ({(c, 0)} : Board))) := by decide

/-- **The left-half depth-2 forced hole, lifted to the real game step `Placement.applyStep`.**
The half-board analogue of `forced_hole_depth_two_applyStep_from_S` (which covered only `{(2,0)}`).
The lift is *structural, not brute force*: every forcing successor is sub-full (`count 9 < cols`), so
`clearLines` is the identity there and `applyStep` collapses to `place`. Hence on every left notch
`c ∈ {0,1,2,3}`, for every hole-free `S` placement, the adversary has a draw `p` such that **every**
placement of `p`, run through the *full* adversarial step `applyStep`, leaves at least one hole. -/
theorem forced_hole_depth_two_applyStep_left_S :
    ∀ c ∈ Finset.range 4,
      ∀ pl ∈ Placement.allValidFor GameConfig.standard Piece.S,
        holes GameConfig.standard (pl.place ({(c, 0)} : Board)) = 0 →
          ∃ p : Piece, ∀ pl' ∈ Placement.allValidFor GameConfig.standard p,
            1 ≤ holes GameConfig.standard
              (Placement.applyStep GameConfig.standard (pl.place ({(c, 0)} : Board)) pl') := by
  intro c hc pl hpl hhf
  obtain ⟨p, hp⟩ := forced_hole_depth_two_no_clear_left_S c hc pl hpl hhf
  refine ⟨p, fun pl' hpl' => ?_⟩
  obtain ⟨hcount', hholes⟩ := hp pl' hpl'
  rw [Placement.applyStep_eq_clearLines_place,
    Board.clearLines_eq_self_of_count_lt GameConfig.standard hcount']
  exact hholes

set_option maxRecDepth 4000 in
set_option maxHeartbeats 1000000 in
/-- **The mirror of `forced_hole_depth_two_no_clear_left_S`: the depth-2 forced hole on the right
half-board, under `Z` (no-clear form).** By the `S`/`Z` reflection, `Z` breaks universality exactly
on the right notches `c ∈ {6,7,8,9}` (iteration 149's `#eval` sweep). For each such notch and each
hole-free `Z` placement, the adversary forces a board on which *every* reply has `count < cols` and
at least one hole. The whole 2-ply window stays sub-full (`count 1 → 5 → 9 < cols`), so this is a
pure no-clear `decide`. -/
theorem forced_hole_depth_two_no_clear_right_Z :
    ∀ c ∈ Finset.Icc 6 9,
      ∀ pl ∈ Placement.allValidFor GameConfig.standard Piece.Z,
        holes GameConfig.standard (pl.place ({(c, 0)} : Board)) = 0 →
          ∃ p : Piece, ∀ pl' ∈ Placement.allValidFor GameConfig.standard p,
            (pl'.place (pl.place ({(c, 0)} : Board))).count < GameConfig.standard.cols ∧
            1 ≤ holes GameConfig.standard (pl'.place (pl.place ({(c, 0)} : Board))) := by decide

/-- **The right-half depth-2 forced hole, lifted to the real game step `Placement.applyStep`.**
The `Z`/right-half mirror of `forced_hole_depth_two_applyStep_left_S`. Same structural lift: every
forcing successor is sub-full (`count 9 < cols`), so `clearLines` is the identity and `applyStep`
collapses to `place`. Together with the left-half theorem this completes the symmetric forced-hole
map: on **every** single-notch surface in the broken region (`{0,1,2,3}` under `S`, `{6,7,8,9}` under
`Z`), the real adversarial step `applyStep` forces an unclearable hole within two draws. -/
theorem forced_hole_depth_two_applyStep_right_Z :
    ∀ c ∈ Finset.Icc 6 9,
      ∀ pl ∈ Placement.allValidFor GameConfig.standard Piece.Z,
        holes GameConfig.standard (pl.place ({(c, 0)} : Board)) = 0 →
          ∃ p : Piece, ∀ pl' ∈ Placement.allValidFor GameConfig.standard p,
            1 ≤ holes GameConfig.standard
              (Placement.applyStep GameConfig.standard (pl.place ({(c, 0)} : Board)) pl') := by
  intro c hc pl hpl hhf
  obtain ⟨p, hp⟩ := forced_hole_depth_two_no_clear_right_Z c hc pl hpl hhf
  refine ⟨p, fun pl' hpl' => ?_⟩
  obtain ⟨hcount', hholes⟩ := hp pl' hpl'
  rw [Placement.applyStep_eq_clearLines_place,
    Board.clearLines_eq_self_of_count_lt GameConfig.standard hcount']
  exact hholes

set_option maxRecDepth 4000 in
set_option maxHeartbeats 1000000 in
/-- **Empty-board hole-forcing dichotomy: exactly `S` and `Z` force a hole on `init`.**
The unified "non-forcing" half, generalising `exists_holes_zero_O_place_empty` and
`exists_holes_zero_I_place_empty` to *all five* non-`S`/`Z` pieces at once: for every piece
`p ∉ {S, Z}` the player has a hole-free placement on the empty board. Combined with the forcing
half (`one_le_holes_S_place_empty` / `one_le_holes_Z_place_empty`, which prove every `S`/`Z`
placement on `∅` buries `≥ 1` cell) this is the *complete* depth-1 forcing map: the adversary
can force a hole at the very first move **iff** it draws `S` or `Z`.

**Why this matters for the verdict.** It is the structural seed of the TRUE route. In a 7-bag the
adversary draws each piece exactly once, so at most `2` of every `7` draws (`S`, `Z`) can force a
fresh hole from a flat surface; the other `5` admit hole-free play. Any FALSE (budget-divergence)
engine must therefore manufacture unbounded holes using only the `2`-per-bag `S`/`Z` quota,
against a player who plays the remaining `5` flat and (per `exists_holes_clearLines_lt`) can clear
holes away. The `decide` is hole-free throughout (placements on `∅`, `count = 4 < cols`, so no
line ever clears), keeping the proof in the base-axiom regime. -/
theorem exists_holes_zero_place_empty_of_ne_SZ (p : Piece)
    (hS : p ≠ Piece.S) (hZ : p ≠ Piece.Z) :
    ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
      holes GameConfig.standard (pl.place ∅) = 0 := by
  cases p with
  | S => exact absurd rfl hS
  | Z => exact absurd rfl hZ
  | O => exact exists_holes_zero_O_place_empty
  | I => exact exists_holes_zero_I_place_empty
  | T => decide
  | L => decide
  | J => decide

/-- **Per-row placement bound for S: an S placement adds at most 2 cells to any row.**
Sharper than the generic `rowCount_place_le`'s `+4`: an S tetromino is two stacked dominoes, so
its drop profile contributes `≤ 2` to every row (`shapeUp_S_shift_row_le_two`). This is the
per-row engine for the depth-3 no-clear ladder — two S/Z draws from `∅` keep every row at `≤ 4`,
under the `cols = 10` clear threshold even after a generic third placement (`+4 → ≤ 8`). -/
theorem rowCount_place_le_two_S (b : Board) (pl : Placement)
    (hp : pl.piece = Piece.S) (r : ℕ) :
    rowCount (pl.place b) r ≤ rowCount b r + 2 := by
  unfold rowCount
  rw [Placement.place_eq_union_dropped, Finset.filter_union]
  refine le_trans (Finset.card_union_le _ _) ?_
  have hd : ((pl.dropped b).filter (fun p => p.2 = r)).card ≤ 2 := by
    rw [Placement.dropped_eq_image]
    have hfi : (pl.shapeUp.image
          (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2))).filter (fun p => p.2 = r)
        = (pl.shapeUp.filter (fun c => pl.dropOffset b + c.2 = r)).image
          (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2)) := by
      ext p
      simp only [Finset.mem_filter, Finset.mem_image]
      constructor
      · rintro ⟨⟨c, hc, rfl⟩, hP⟩; exact ⟨c, ⟨hc, hP⟩, rfl⟩
      · rintro ⟨c, ⟨hc, hP⟩, rfl⟩; exact ⟨⟨c, hc, rfl⟩, hP⟩
    rw [hfi]
    refine le_trans Finset.card_image_le ?_
    unfold Placement.shapeUp
    rw [hp]
    exact Piece.shapeUp_S_shift_row_le_two pl.rot (pl.dropOffset b) r
  omega

/-- **Per-row placement bound for Z: a Z placement adds at most 2 cells to any row.** -/
theorem rowCount_place_le_two_Z (b : Board) (pl : Placement)
    (hp : pl.piece = Piece.Z) (r : ℕ) :
    rowCount (pl.place b) r ≤ rowCount b r + 2 := by
  unfold rowCount
  rw [Placement.place_eq_union_dropped, Finset.filter_union]
  refine le_trans (Finset.card_union_le _ _) ?_
  have hd : ((pl.dropped b).filter (fun p => p.2 = r)).card ≤ 2 := by
    rw [Placement.dropped_eq_image]
    have hfi : (pl.shapeUp.image
          (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2))).filter (fun p => p.2 = r)
        = (pl.shapeUp.filter (fun c => pl.dropOffset b + c.2 = r)).image
          (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2)) := by
      ext p
      simp only [Finset.mem_filter, Finset.mem_image]
      constructor
      · rintro ⟨⟨c, hc, rfl⟩, hP⟩; exact ⟨c, ⟨hc, hP⟩, rfl⟩
      · rintro ⟨c, ⟨hc, hP⟩, rfl⟩; exact ⟨⟨c, hc, rfl⟩, hP⟩
    rw [hfi]
    refine le_trans Finset.card_image_le ?_
    unfold Placement.shapeUp
    rw [hp]
    exact Piece.shapeUp_Z_shift_row_le_two pl.rot (pl.dropOffset b) r
  omega

/-- **Depth-3 hole floor: count = 12 with a surviving hole.** Pushes
`exists_mem_closed_count_eight_holes_pos` (depth-2, `count = 8`) one adversarial draw past the
*total*-count no-clear wall (which shuts at `count = 4n < cols = 10`, i.e. `n ≤ 2`). The adversary
plays `init →S→ g1 →Z→ g2 →(any)→ g3`. The third placement lands on `count = 12 ≥ cols`, so the
total-count gate `clearLines_eq_self_of_count_lt` no longer applies — but the **per-row** gate does:
`S` and `Z` each add `≤ 2` cells to any row (`rowCount_place_le_two_S/_Z`), so from `∅` every row of
`g2.board` holds `≤ 4`; a generic third placement adds `≤ 4` more (`rowCount_place_le`), keeping every
row `≤ 8 < 10 = cols`. Hence no row is full (`fullRows_eq_empty_of_forall_rowCount_lt`), the clear is
the identity, and holes stay monotone (`holes_le_holes_place`). The depth-2 hole survives to `g3`,
which carries `count = 12` and `≥ 1` hole. This is the first ladder rung **above** the total-count
window — the per-row infrastructure is exactly what unlocks it. -/
theorem exists_mem_closed_count_twelve_holes_pos {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, g.board.count = 12 ∧ 1 ≤ holes GameConfig.standard g.board := by
  have h5 : (5 : ℕ) ≤ GameConfig.standard.cols := by decide
  have hcols : GameConfig.standard.cols = 10 := by decide
  -- First draw: S on the empty initial board.
  have hSb : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  obtain ⟨_, hstep0⟩ := hclosed GameState.init hinit
  obtain ⟨plS, hplS, hsucc1⟩ := hstep0 Piece.S hSb
  rw [Placement.mem_allValidFor] at hplS
  obtain ⟨hpieceS, hvalidS⟩ := hplS
  set g1 := adversarialStep GameConfig.standard GameState.init Piece.S plS with hg1def
  have hplS_eq : ({plS with piece := Piece.S} : Placement) = plS := by
    rcases plS with ⟨pp, pr, pc⟩; cases hpieceS; rfl
  have hg1board : g1.board = plS.place ∅ := by
    rw [hg1def, adversarialStep_board, GameState.init_board_eq_emptyset, hplS_eq,
        applyStep_empty_eq_place_empty h5]
  have hg1count : g1.board.count = 4 := by
    rw [hg1board, Placement.count_place]; rfl
  have hg1holes : 1 ≤ holes GameConfig.standard g1.board := by
    rw [hg1board]; exact one_le_holes_S_place_empty hvalidS hpieceS
  have hg1row : ∀ r, rowCount g1.board r ≤ 2 := by
    intro r
    rw [hg1board]
    have hh := rowCount_place_le_two_S ∅ plS hpieceS r
    rw [rowCount_empty] at hh
    omega
  -- Second draw: Z is still in the bag (only S removed). count 4 < 10 ⇒ no clear.
  have hZb : Piece.Z ∈ g1.bag := by
    rw [hg1def, adversarialStep_bag, GameState.init_bag, Bag.draw_full_eq_erase]
    exact Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.Z⟩
  obtain ⟨_, hstep1⟩ := hclosed g1 hsucc1
  obtain ⟨plZ, hplZ, hsucc2⟩ := hstep1 Piece.Z hZb
  rw [Placement.mem_allValidFor] at hplZ
  obtain ⟨hpieceZ, hvalidZ⟩ := hplZ
  set g2 := adversarialStep GameConfig.standard g1 Piece.Z plZ with hg2def
  have hplZ_eq : ({plZ with piece := Piece.Z} : Placement) = plZ := by
    rcases plZ with ⟨pp, pr, pc⟩; cases hpieceZ; rfl
  have hg2board0 : g2.board = plZ.applyStep GameConfig.standard g1.board := by
    rw [hg2def, adversarialStep_board, hplZ_eq]
  have hplaceCount2 : (plZ.place g1.board).count < GameConfig.standard.cols := by
    rw [Placement.count_place, hg1count]; decide
  have hg2board : g2.board = plZ.place g1.board := by
    rw [hg2board0, Placement.applyStep_eq_clearLines_place,
        Board.clearLines_eq_self_of_count_lt GameConfig.standard hplaceCount2]
  have hg2count : g2.board.count = 8 := by
    rw [hg2board, Placement.count_place, hg1count]
  have hg2holes : 1 ≤ holes GameConfig.standard g2.board := by
    rw [hg2board]; exact le_trans hg1holes (holes_le_holes_place _ _ _)
  have hg2row : ∀ r, rowCount g2.board r ≤ 4 := by
    intro r
    rw [hg2board]
    have hz := rowCount_place_le_two_Z g1.board plZ hpieceZ r
    have hs := hg1row r
    omega
  -- Third draw: any remaining piece (bag is nonempty). count 8 + 4 = 12 ≥ cols, so the
  -- total-count gate is gone — but the per-row bound (≤ 8 < 10) blocks the clear.
  have hne2 : g2.bag.Nonempty := by
    rw [hg2def, adversarialStep_bag]; exact Bag.draw_nonempty _ _
  obtain ⟨p3, hp3b⟩ := hne2
  obtain ⟨_, hstep2⟩ := hclosed g2 hsucc2
  obtain ⟨pl3, hpl3, hsucc3⟩ := hstep2 p3 hp3b
  rw [Placement.mem_allValidFor] at hpl3
  obtain ⟨hpiece3, hvalid3⟩ := hpl3
  set g3 := adversarialStep GameConfig.standard g2 p3 pl3 with hg3def
  have hpl3_eq : ({pl3 with piece := p3} : Placement) = pl3 := by
    rcases pl3 with ⟨pp, pr, pc⟩; cases hpiece3; rfl
  have hg3board0 : g3.board = pl3.applyStep GameConfig.standard g2.board := by
    rw [hg3def, adversarialStep_board, hpl3_eq]
  have hnofull : Board.fullRows GameConfig.standard (pl3.place g2.board) = ∅ := by
    apply fullRows_eq_empty_of_forall_rowCount_lt
    intro r
    have h1 := rowCount_place_le g2.board pl3 r
    have h2 := hg2row r
    omega
  have hg3board : g3.board = pl3.place g2.board := by
    rw [hg3board0, Placement.applyStep_eq_clearLines_place,
        Board.clearLines_eq_self_of_no_fullRows GameConfig.standard hnofull]
  refine ⟨g3, hsucc3, ?_, ?_⟩
  · rw [hg3board, Placement.count_place, hg2count]
  · rw [hg3board]; exact le_trans hg2holes (holes_le_holes_place _ _ _)

/-- **Depth-3 `count + holes` floor: `≥ 13`.** Sharpens `exists_mem_closed_nine_le_count_add_holes`
(depth-2, `≥ 9`) by one no-clear draw past the total-count window
(`exists_mem_closed_count_twelve_holes_pos`: `count = 12 ∧ holes ≥ 1`). Tightens the FALSE-budget
gap from `[9, 200]` to `[13, 200]` — the first floor improvement obtained from the per-row
(rather than total-count) no-clear gate. -/
theorem exists_mem_closed_thirteen_le_count_add_holes {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, 13 ≤ g.board.count + holes GameConfig.standard g.board := by
  obtain ⟨g, hg, hc, hh⟩ := exists_mem_closed_count_twelve_holes_pos hinit hclosed
  exact ⟨g, hg, by omega⟩

/-- **Per-row placement bound for O: an O placement adds at most 2 cells to any row.**
Sharper than the generic `rowCount_place_le`'s `+4`. Completes the `≤ 2` per-row family
`{O, S, Z}` — the only three pieces with a 2-wide drop profile. Combined with a `≤ 3` bound for
`T`/`L`/`J`, this unlocks a depth-4 no-clear ladder rung (`S,Z` give `≤ 4`, a `≤ 3` piece `≤ 7`,
an O `≤ 9 < cols = 10`, so `count = 16` with no clear). -/
theorem rowCount_place_le_two_O (b : Board) (pl : Placement)
    (hp : pl.piece = Piece.O) (r : ℕ) :
    rowCount (pl.place b) r ≤ rowCount b r + 2 := by
  unfold rowCount
  rw [Placement.place_eq_union_dropped, Finset.filter_union]
  refine le_trans (Finset.card_union_le _ _) ?_
  have hd : ((pl.dropped b).filter (fun p => p.2 = r)).card ≤ 2 := by
    rw [Placement.dropped_eq_image]
    have hfi : (pl.shapeUp.image
          (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2))).filter (fun p => p.2 = r)
        = (pl.shapeUp.filter (fun c => pl.dropOffset b + c.2 = r)).image
          (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2)) := by
      ext p
      simp only [Finset.mem_filter, Finset.mem_image]
      constructor
      · rintro ⟨⟨c, hc, rfl⟩, hP⟩; exact ⟨c, ⟨hc, hP⟩, rfl⟩
      · rintro ⟨c, ⟨hc, hP⟩, rfl⟩; exact ⟨⟨c, hc, rfl⟩, hP⟩
    rw [hfi]
    refine le_trans Finset.card_image_le ?_
    unfold Placement.shapeUp
    rw [hp]
    exact Piece.shapeUp_O_shift_row_le_two pl.rot (pl.dropOffset b) r
  omega

/-- **Per-row placement bound for T: a T placement adds at most 3 cells to any row.**
Between the `≤ 2` family `{O, S, Z}` and the `I` piece's `≤ 4`. Together with `rowCount_place_le_two_O`
this powers the depth-4 no-clear ladder: from `∅`, the sequence `S, Z` (each `≤ 2`) reaches rows
`≤ 4`, a `T` (`≤ 3`) reaches `≤ 7`, then an `O` (`≤ 2`) reaches `≤ 9 < cols = 10`, so all four
placements clear no line and `count = 16` with the S-hole intact. -/
theorem rowCount_place_le_three_T (b : Board) (pl : Placement)
    (hp : pl.piece = Piece.T) (r : ℕ) :
    rowCount (pl.place b) r ≤ rowCount b r + 3 := by
  unfold rowCount
  rw [Placement.place_eq_union_dropped, Finset.filter_union]
  refine le_trans (Finset.card_union_le _ _) ?_
  have hd : ((pl.dropped b).filter (fun p => p.2 = r)).card ≤ 3 := by
    rw [Placement.dropped_eq_image]
    have hfi : (pl.shapeUp.image
          (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2))).filter (fun p => p.2 = r)
        = (pl.shapeUp.filter (fun c => pl.dropOffset b + c.2 = r)).image
          (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2)) := by
      ext p
      simp only [Finset.mem_filter, Finset.mem_image]
      constructor
      · rintro ⟨⟨c, hc, rfl⟩, hP⟩; exact ⟨c, ⟨hc, hP⟩, rfl⟩
      · rintro ⟨c, ⟨hc, hP⟩, rfl⟩; exact ⟨⟨c, hc, rfl⟩, hP⟩
    rw [hfi]
    refine le_trans Finset.card_image_le ?_
    unfold Placement.shapeUp
    rw [hp]
    exact Piece.shapeUp_T_shift_row_le_three pl.rot (pl.dropOffset b) r
  omega

/-- **Depth-4 hole floor: `count = 16` with a surviving hole.** Extends
`exists_mem_closed_count_twelve_holes_pos` by one more no-clear draw, exploiting the *mixed*
per-row footprints `S, Z = 2`, `T = 3`, `O = 2` — four distinct pieces that all coexist in a
single 7-bag. The adversary front-loads `S → Z → T → O`; the resulting rows stay at
`≤ 2, 4, 7, 9`, every one `< cols = 10`, so `fullRows = ∅` at all four steps and no line clears
(`fullRows_eq_empty_of_forall_rowCount_lt`). The `S`-hole therefore persists
(`holes_le_holes_place`) while `count` climbs `4 → 8 → 12 → 16`. This is the per-row gate's reach
past the depth-3 rung: it needs the `T` footprint bound that the total-count wall cannot supply. -/
theorem exists_mem_closed_count_sixteen_holes_pos {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, g.board.count = 16 ∧ 1 ≤ holes GameConfig.standard g.board := by
  have h5 : (5 : ℕ) ≤ GameConfig.standard.cols := by decide
  have hcols : GameConfig.standard.cols = 10 := by decide
  -- First draw: S on the empty initial board.
  have hSb : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full Piece.S
  obtain ⟨_, hstep0⟩ := hclosed GameState.init hinit
  obtain ⟨plS, hplS, hsucc1⟩ := hstep0 Piece.S hSb
  rw [Placement.mem_allValidFor] at hplS
  obtain ⟨hpieceS, hvalidS⟩ := hplS
  set g1 := adversarialStep GameConfig.standard GameState.init Piece.S plS with hg1def
  have hplS_eq : ({plS with piece := Piece.S} : Placement) = plS := by
    rcases plS with ⟨pp, pr, pc⟩; cases hpieceS; rfl
  have hg1board : g1.board = plS.place ∅ := by
    rw [hg1def, adversarialStep_board, GameState.init_board_eq_emptyset, hplS_eq,
        applyStep_empty_eq_place_empty h5]
  have hg1count : g1.board.count = 4 := by
    rw [hg1board, Placement.count_place]; rfl
  have hg1holes : 1 ≤ holes GameConfig.standard g1.board := by
    rw [hg1board]; exact one_le_holes_S_place_empty hvalidS hpieceS
  have hg1row : ∀ r, rowCount g1.board r ≤ 2 := by
    intro r
    rw [hg1board]
    have hh := rowCount_place_le_two_S ∅ plS hpieceS r
    rw [rowCount_empty] at hh
    omega
  -- Bag thread: g1.bag = full.erase S.
  have hg1bag : g1.bag = Bag.full.erase Piece.S := by
    rw [hg1def, adversarialStep_bag, GameState.init_bag, Bag.draw_full_eq_erase]
  have hZin : Piece.Z ∈ Bag.full.erase Piece.S :=
    Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.Z⟩
  -- Second draw: Z. count 8 < 10 ⇒ no clear (total-count gate still available).
  have hZb : Piece.Z ∈ g1.bag := by rw [hg1bag]; exact hZin
  obtain ⟨_, hstep1⟩ := hclosed g1 hsucc1
  obtain ⟨plZ, hplZ, hsucc2⟩ := hstep1 Piece.Z hZb
  rw [Placement.mem_allValidFor] at hplZ
  obtain ⟨hpieceZ, hvalidZ⟩ := hplZ
  set g2 := adversarialStep GameConfig.standard g1 Piece.Z plZ with hg2def
  have hplZ_eq : ({plZ with piece := Piece.Z} : Placement) = plZ := by
    rcases plZ with ⟨pp, pr, pc⟩; cases hpieceZ; rfl
  have hg2board0 : g2.board = plZ.applyStep GameConfig.standard g1.board := by
    rw [hg2def, adversarialStep_board, hplZ_eq]
  have hplaceCount2 : (plZ.place g1.board).count < GameConfig.standard.cols := by
    rw [Placement.count_place, hg1count]; decide
  have hg2board : g2.board = plZ.place g1.board := by
    rw [hg2board0, Placement.applyStep_eq_clearLines_place,
        Board.clearLines_eq_self_of_count_lt GameConfig.standard hplaceCount2]
  have hg2count : g2.board.count = 8 := by
    rw [hg2board, Placement.count_place, hg1count]
  have hg2holes : 1 ≤ holes GameConfig.standard g2.board := by
    rw [hg2board]; exact le_trans hg1holes (holes_le_holes_place _ _ _)
  have hg2row : ∀ r, rowCount g2.board r ≤ 4 := by
    intro r
    rw [hg2board]
    have hz := rowCount_place_le_two_Z g1.board plZ hpieceZ r
    have hs := hg1row r
    omega
  -- Bag thread: g2.bag = (full.erase S).erase Z; T still present.
  have hg2bag : g2.bag = (Bag.full.erase Piece.S).erase Piece.Z := by
    rw [hg2def, adversarialStep_bag, hg1bag,
        Bag.draw_eq_erase_of_card_ge_two hZin (by rw [Bag.full_erase_card]; decide)]
  have hTin : Piece.T ∈ (Bag.full.erase Piece.S).erase Piece.Z :=
    Finset.mem_erase.mpr ⟨by decide, Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.T⟩⟩
  have hTb : Piece.T ∈ g2.bag := by rw [hg2bag]; exact hTin
  -- Third draw: T (footprint ≤ 3). count = 12 ≥ cols, but per-row ≤ 7 < 10 blocks the clear.
  obtain ⟨_, hstep2⟩ := hclosed g2 hsucc2
  obtain ⟨pl3, hpl3, hsucc3⟩ := hstep2 Piece.T hTb
  rw [Placement.mem_allValidFor] at hpl3
  obtain ⟨hpiece3, hvalid3⟩ := hpl3
  set g3 := adversarialStep GameConfig.standard g2 Piece.T pl3 with hg3def
  have hpl3_eq : ({pl3 with piece := Piece.T} : Placement) = pl3 := by
    rcases pl3 with ⟨pp, pr, pc⟩; cases hpiece3; rfl
  have hg3board0 : g3.board = pl3.applyStep GameConfig.standard g2.board := by
    rw [hg3def, adversarialStep_board, hpl3_eq]
  have hnofull3 : Board.fullRows GameConfig.standard (pl3.place g2.board) = ∅ := by
    apply fullRows_eq_empty_of_forall_rowCount_lt
    intro r
    have h1 := rowCount_place_le_three_T g2.board pl3 hpiece3 r
    have h2 := hg2row r
    omega
  have hg3board : g3.board = pl3.place g2.board := by
    rw [hg3board0, Placement.applyStep_eq_clearLines_place,
        Board.clearLines_eq_self_of_no_fullRows GameConfig.standard hnofull3]
  have hg3count : g3.board.count = 12 := by
    rw [hg3board, Placement.count_place, hg2count]
  have hg3holes : 1 ≤ holes GameConfig.standard g3.board := by
    rw [hg3board]; exact le_trans hg2holes (holes_le_holes_place _ _ _)
  have hg3row : ∀ r, rowCount g3.board r ≤ 7 := by
    intro r
    rw [hg3board]
    have ht := rowCount_place_le_three_T g2.board pl3 hpiece3 r
    have hs := hg2row r
    omega
  -- Bag thread: g3.bag = ((full.erase S).erase Z).erase T; O still present.
  have hOb : Piece.O ∈ g3.bag := by
    rw [hg3def, adversarialStep_bag, hg2bag,
        Bag.draw_eq_erase_of_card_ge_two hTin
          (by rw [Finset.card_erase_of_mem hZin, Bag.full_erase_card]; decide)]
    exact Finset.mem_erase.mpr ⟨by decide,
      Finset.mem_erase.mpr ⟨by decide, Finset.mem_erase.mpr ⟨by decide, Bag.mem_full Piece.O⟩⟩⟩
  -- Fourth draw: O (footprint ≤ 2). count = 16 ≥ cols, per-row ≤ 9 < 10 blocks the clear.
  obtain ⟨_, hstep3⟩ := hclosed g3 hsucc3
  obtain ⟨pl4, hpl4, hsucc4⟩ := hstep3 Piece.O hOb
  rw [Placement.mem_allValidFor] at hpl4
  obtain ⟨hpiece4, hvalid4⟩ := hpl4
  set g4 := adversarialStep GameConfig.standard g3 Piece.O pl4 with hg4def
  have hpl4_eq : ({pl4 with piece := Piece.O} : Placement) = pl4 := by
    rcases pl4 with ⟨pp, pr, pc⟩; cases hpiece4; rfl
  have hg4board0 : g4.board = pl4.applyStep GameConfig.standard g3.board := by
    rw [hg4def, adversarialStep_board, hpl4_eq]
  have hnofull4 : Board.fullRows GameConfig.standard (pl4.place g3.board) = ∅ := by
    apply fullRows_eq_empty_of_forall_rowCount_lt
    intro r
    have h1 := rowCount_place_le_two_O g3.board pl4 hpiece4 r
    have h2 := hg3row r
    omega
  have hg4board : g4.board = pl4.place g3.board := by
    rw [hg4board0, Placement.applyStep_eq_clearLines_place,
        Board.clearLines_eq_self_of_no_fullRows GameConfig.standard hnofull4]
  refine ⟨g4, hsucc4, ?_, ?_⟩
  · rw [hg4board, Placement.count_place, hg3count]
  · rw [hg4board]; exact le_trans hg3holes (holes_le_holes_place _ _ _)

/-- **Depth-4 `count + holes` floor: `≥ 17`.** Sharpens `exists_mem_closed_thirteen_le_count_add_holes`
(depth-3, `≥ 13`) by one more no-clear draw (`exists_mem_closed_count_sixteen_holes_pos`:
`count = 16 ∧ holes ≥ 1`). Tightens the FALSE-budget gap from `[13, 200]` to `[17, 200]`. -/
theorem exists_mem_closed_seventeen_le_count_add_holes {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, 17 ≤ g.board.count + holes GameConfig.standard g.board := by
  obtain ⟨g, hg, hc, hh⟩ := exists_mem_closed_count_sixteen_holes_pos hinit hclosed
  exact ⟨g, hg, by omega⟩

/-! ### Trajectory-aggregate forced clears at the closed-set level

The per-trajectory `survival_forces_clears` (`4·n ≤ 200 + 10·k`, iter-114) lifts to any
init-containing closed atlas exactly as the residue corollaries did
(`exists_state_at_residue_in_closed_init_atlas`): take the depth-`n` reachable witness in
`T` (`exists_reachableAt_in_closed_init_atlas`) and read its non-lost hypothesis straight
off closure (`hclosed g hg`). The atlas therefore cannot survive by stacking — by depth
`51` it has already cleared at least one row, and its forced clears grow without bound with
depth. This is the closed-set face of the FALSE-route trajectory aggregate: a surviving
atlas is committed to completing full rows forever, the very rows the `S`/`Z` forced-hole
flood obstructs. -/

/-- **Atlas forced-clear relation.** Any init-containing closed atlas contains, at every
depth `n`, a reachable state whose trajectory cleared `k` rows with `count + 10·k = 4·n`
and `4·n ≤ 200 + 10·k` (so `k ≥ (4·n − 200)/10`). The closed-set lift of
`survival_forces_clears`: the forced clears of a surviving atlas grow without bound. -/
theorem exists_mem_closed_clears_ge {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T) (n : ℕ) :
    ∃ g ∈ T, ∃ k : ℕ, g.board.count + 10 * k = 4 * n ∧ 4 * n ≤ 200 + 10 * k := by
  obtain ⟨g, hreach, hgT⟩ := exists_reachableAt_in_closed_init_atlas hinit hclosed n
  obtain ⟨hnl, _⟩ := hclosed g hgT
  obtain ⟨k, hk, hle⟩ := survival_forces_clears hreach hnl
  exact ⟨g, hgT, k, hk, hle⟩

/-- **No stalling atlas: every init-atlas clears a line.** At depth `51` the
`+4`-per-piece inflow (`204` cells) cannot fit under the `200`-cell ceiling, so any
init-containing closed atlas contains a reachable state whose trajectory has cleared at
least one row (`k ≥ 1`). A pure-stacking survival strategy is impossible — the defender is
forced to complete full rows, which is exactly where the `S`/`Z` forced-hole obstruction
will press. -/
theorem exists_mem_closed_clears_pos {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T) :
    ∃ g ∈ T, ∃ k : ℕ, g.board.count + 10 * k = 4 * 51 ∧ 1 ≤ k := by
  obtain ⟨g, hgT, k, hk, hle⟩ := exists_mem_closed_clears_ge hinit hclosed 51
  exact ⟨g, hgT, k, hk, by omega⟩

/-- **Unbounded forced clears: a surviving atlas clears arbitrarily many lines.** For every
target `K`, any init-containing closed atlas contains a reachable state (at depth `50 + 3·K`)
whose trajectory has already cleared at least `K` rows. Sharpens `exists_mem_closed_clears_pos`
(`k ≥ 1`) to `k ≥ K` for all `K`: the forced clears have no finite ceiling. At depth
`n = 50 + 3·K` the inflow is `4·n = 200 + 12·K`, so `200 + 12·K ≤ 200 + 10·k` forces
`k ≥ ⌈12·K/10⌉ ≥ K`.

**Verdict significance.** This is the closed-set face of `survival_forces_clears` pushed to its
asymptotic conclusion: a surviving atlas is committed to completing *unboundedly many* full
rows over its lifetime — it cannot eventually stop clearing. Every one of those full rows must
be assembled in a board the adversary is simultaneously seeding with `S`/`Z` holes (one of each
per 7-bag), and a hole keeps its own row from ever being full until exposed. So the FALSE route
reduces to a *geometric* race — infinitely many full-row completions versus the adversary's
hole placement — rather than any aggregate count/Φ ceiling (those are pinned at `200` on both
sides and cannot yield a strict contradiction). -/
theorem exists_mem_closed_clears_ge_arb {T : Finset GameState}
    (hinit : GameState.init ∈ T) (hclosed : IsClosedSet GameConfig.standard T) (K : ℕ) :
    ∃ g ∈ T, ∃ k : ℕ, g.board.count + 10 * k = 4 * (50 + 3 * K) ∧ K ≤ k := by
  obtain ⟨g, hgT, k, hk, hle⟩ := exists_mem_closed_clears_ge hinit hclosed (50 + 3 * K)
  exact ⟨g, hgT, k, hk, by omega⟩

/-- **Per-row placement bound for L: an L placement adds at most 3 cells to any row.**
Mirror of `rowCount_place_le_three_T`; together with `rowCount_place_le_three_J` it closes the
`≤ 3` footprint family `{T, L, J}`, leaving only the `I` piece able to deposit 4 cells into a
single row. -/
theorem rowCount_place_le_three_L (b : Board) (pl : Placement)
    (hp : pl.piece = Piece.L) (r : ℕ) :
    rowCount (pl.place b) r ≤ rowCount b r + 3 := by
  unfold rowCount
  rw [Placement.place_eq_union_dropped, Finset.filter_union]
  refine le_trans (Finset.card_union_le _ _) ?_
  have hd : ((pl.dropped b).filter (fun p => p.2 = r)).card ≤ 3 := by
    rw [Placement.dropped_eq_image]
    have hfi : (pl.shapeUp.image
          (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2))).filter (fun p => p.2 = r)
        = (pl.shapeUp.filter (fun c => pl.dropOffset b + c.2 = r)).image
          (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2)) := by
      ext p
      simp only [Finset.mem_filter, Finset.mem_image]
      constructor
      · rintro ⟨⟨c, hc, rfl⟩, hP⟩; exact ⟨c, ⟨hc, hP⟩, rfl⟩
      · rintro ⟨c, ⟨hc, hP⟩, rfl⟩; exact ⟨⟨c, hc, rfl⟩, hP⟩
    rw [hfi]
    refine le_trans Finset.card_image_le ?_
    unfold Placement.shapeUp
    rw [hp]
    exact Piece.shapeUp_L_shift_row_le_three pl.rot (pl.dropOffset b) r
  omega

/-- **Per-row placement bound for J: a J placement adds at most 3 cells to any row.**
The mirror of `rowCount_place_le_three_L`, completing the `≤ 3` footprint family `{T, L, J}`. -/
theorem rowCount_place_le_three_J (b : Board) (pl : Placement)
    (hp : pl.piece = Piece.J) (r : ℕ) :
    rowCount (pl.place b) r ≤ rowCount b r + 3 := by
  unfold rowCount
  rw [Placement.place_eq_union_dropped, Finset.filter_union]
  refine le_trans (Finset.card_union_le _ _) ?_
  have hd : ((pl.dropped b).filter (fun p => p.2 = r)).card ≤ 3 := by
    rw [Placement.dropped_eq_image]
    have hfi : (pl.shapeUp.image
          (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2))).filter (fun p => p.2 = r)
        = (pl.shapeUp.filter (fun c => pl.dropOffset b + c.2 = r)).image
          (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2)) := by
      ext p
      simp only [Finset.mem_filter, Finset.mem_image]
      constructor
      · rintro ⟨⟨c, hc, rfl⟩, hP⟩; exact ⟨c, ⟨hc, hP⟩, rfl⟩
      · rintro ⟨c, ⟨hc, hP⟩, rfl⟩; exact ⟨⟨c, hc, rfl⟩, hP⟩
    rw [hfi]
    refine le_trans Finset.card_image_le ?_
    unfold Placement.shapeUp
    rw [hp]
    exact Piece.shapeUp_J_shift_row_le_three pl.rot (pl.dropOffset b) r
  omega

/-- **Only the `I` piece can fast-fill a row.** Every non-`I` tetromino deposits at most `3`
cells into any single row — `O, S, Z ≤ 2` and `T, L, J ≤ 3` — while `I` alone lays a flat 4-bar.
Uniform capstone over the per-row footprint table: a row can leap from `cols − 4` cells to full
in a single placement *only* when the adversary hands the player the `I` piece. Every other draw
adds at most `3`, so completing a row that was `≥ 2` cells short requires an `I`. This pins the
clear-side of the geometric race: the adversary controls when a 4-fill is even possible. -/
theorem rowCount_place_le_three_of_ne_I (b : Board) (pl : Placement)
    (hne : pl.piece ≠ Piece.I) (r : ℕ) :
    rowCount (pl.place b) r ≤ rowCount b r + 3 := by
  cases hpiece : pl.piece with
  | O => have := rowCount_place_le_two_O b pl hpiece r; omega
  | I => exact absurd hpiece hne
  | S => have := rowCount_place_le_two_S b pl hpiece r; omega
  | Z => have := rowCount_place_le_two_Z b pl hpiece r; omega
  | T => exact rowCount_place_le_three_T b pl hpiece r
  | L => exact rowCount_place_le_three_L b pl hpiece r
  | J => exact rowCount_place_le_three_J b pl hpiece r

/-- **A full row holds at least `cols` cells.** The injection `c ↦ (c, r)` embeds `range cols`
into row `r`'s cell-fiber, so `cfg.cols ≤ rowCount b r`. Standalone extraction of the embedding
buried inside `not_isFull_of_rowCount_lt`; the forward (rather than contrapositive) direction. -/
theorem cols_le_rowCount_of_isFull (cfg : GameConfig) (b : Board) (r : ℕ)
    (hfull : Board.isFull cfg b r) : cfg.cols ≤ rowCount b r := by
  have hsub : (Finset.range cfg.cols).image (fun c => ((c, r) : ℕ × ℕ))
      ⊆ b.filter (fun p => p.2 = r) := by
    intro p hp
    rw [Finset.mem_image] at hp
    obtain ⟨c, hc, rfl⟩ := hp
    rw [Finset.mem_range] at hc
    rw [Finset.mem_filter]
    exact ⟨Board.mem_of_isFull cfg hc hfull, rfl⟩
  have hinj : Function.Injective (fun c => ((c, r) : ℕ × ℕ)) := by
    intro a b hab; simpa using hab
  unfold rowCount
  have h1 : ((Finset.range cfg.cols).image (fun c => ((c, r) : ℕ × ℕ))).card
      ≤ (b.filter (fun p => p.2 = r)).card := Finset.card_le_card hsub
  rwa [Finset.card_image_of_injective _ hinj, Finset.card_range] at h1

/-- **Row-completion threshold (non-`I`).** If a non-`I` placement completes row `r`
(`r ∈ fullRows (pl.place b)`), then row `r` already held `≥ cols − 3` cells *before* the move:
`cfg.cols ≤ rowCount b r + 3`. Chains the full-row floor `cols ≤ rowCount (place)` against the
capstone `rowCount (place) ≤ rowCount b + 3`. The clear-side of the geometric race made
quantitative — a non-`I` piece can only finish a row that was already nearly full. -/
theorem cols_le_rowCount_add_three_of_mem_fullRows_place_of_ne_I
    (cfg : GameConfig) (b : Board) (pl : Placement)
    (hne : pl.piece ≠ Piece.I) {r : ℕ} (hr : r ∈ Board.fullRows cfg (pl.place b)) :
    cfg.cols ≤ rowCount b r + 3 := by
  have hfull : Board.isFull cfg (pl.place b) r := Board.isFull_of_mem_fullRows hr
  have h1 : cfg.cols ≤ rowCount (pl.place b) r :=
    cols_le_rowCount_of_isFull cfg (pl.place b) r hfull
  have h2 : rowCount (pl.place b) r ≤ rowCount b r + 3 :=
    rowCount_place_le_three_of_ne_I b pl hne r
  omega

/-- **Standard-field row-completion threshold (`≥ 7`).** On the `10`-wide standard board, a non-`I`
placement that clears a row requires that row to have *already* held `≥ 7` of its `10` cells.
Because each `7`-bag delivers the `I` piece exactly once, at most one placement per bag can finish
a row from `< 7` cells — the adversary throttles "deep" row completions to one per bag. -/
theorem seven_le_rowCount_of_mem_fullRows_place_of_ne_I
    (b : Board) (pl : Placement) (hne : pl.piece ≠ Piece.I) {r : ℕ}
    (hr : r ∈ Board.fullRows GameConfig.standard (pl.place b)) :
    7 ≤ rowCount b r := by
  have h := cols_le_rowCount_add_three_of_mem_fullRows_place_of_ne_I GameConfig.standard b pl hne hr
  have hc : GameConfig.standard.cols = 10 := by decide
  omega

/-- **Wide no-clear gate for non-`I` placements.** A non-`I` piece adds at most `3` cells to any
single row (`rowCount_place_le_three_of_ne_I`), so if every row of `b` satisfies
`rowCount b r + 3 < cfg.cols` then placing a non-`I` piece on `b` creates *no* full row. This is the
non-`I` analogue of `fullRows_eq_empty_of_forall_rowCount_lt`, but stated for the post-placement
board in terms of the *pre-placement* row counts. -/
theorem fullRows_place_eq_empty_of_forall_rowCount_add_three_lt_of_ne_I
    (cfg : GameConfig) (b : Board) (pl : Placement) (hne : pl.piece ≠ Piece.I)
    (h : ∀ r, rowCount b r + 3 < cfg.cols) :
    Board.fullRows cfg (pl.place b) = ∅ := by
  rcases Finset.eq_empty_or_nonempty (Board.fullRows cfg (pl.place b)) with he | hne'
  · exact he
  · exfalso
    obtain ⟨r, hr⟩ := hne'
    have h1 := cols_le_rowCount_add_three_of_mem_fullRows_place_of_ne_I cfg b pl hne hr
    have h2 := h r
    omega

/-- **Standard-field `≤ 6` no-clear gate for non-`I` placements.** On the `10`-wide standard board, a
non-`I` placement clears no line whenever every row already holds `≤ 6` cells. This strictly widens
the generic gate (`≤ 5`, from the `+4` footprint of `fullRows_eq_empty_of_forall_rowCount_lt`): only
the once-per-bag `I` piece can clear from a row that started at `7`. -/
theorem fullRows_place_eq_empty_of_forall_rowCount_le_six_of_ne_I
    (b : Board) (pl : Placement) (hne : pl.piece ≠ Piece.I)
    (h : ∀ r, rowCount b r ≤ 6) :
    Board.fullRows GameConfig.standard (pl.place b) = ∅ := by
  apply fullRows_place_eq_empty_of_forall_rowCount_add_three_lt_of_ne_I GameConfig.standard b pl hne
  intro r
  have hc : GameConfig.standard.cols = 10 := by decide
  have := h r
  omega

/-- **`I`-out-of-bag forced no-clear gate.** A `7`-bag delivers the `I` piece exactly once per
cycle; `Bag` is a `Finset` of distinct pieces, so once `I` has been drawn it is absent
(`Piece.I ∉ bag`) until the bag refills. While `I` is absent every deliverable piece `p ∈ bag` is a
non-`I` piece, so on a *flat* board (every row `≤ 6` cells) the adversary cannot force a line clear no
matter which piece it hands the player — by `fullRows_place_eq_empty_of_forall_rowCount_le_six_of_ne_I`
the placement creates no full row. This ties the row-completion threshold to the bag's position in
its `7`-cycle: the post-`I` segment of a bag is a guaranteed no-clear window over flat boards. -/
theorem fullRows_place_eq_empty_of_notMem_bag_I_of_forall_rowCount_le_six
    (b : Board) (bag : Bag) (hI : Piece.I ∉ bag)
    (pl : Placement) (hpl : pl.piece ∈ bag)
    (hrows : ∀ r, rowCount b r ≤ 6) :
    Board.fullRows GameConfig.standard (pl.place b) = ∅ := by
  have hne : pl.piece ≠ Piece.I := fun h => hI (h ▸ hpl)
  exact fullRows_place_eq_empty_of_forall_rowCount_le_six_of_ne_I b pl hne hrows

/-- **Adversarial step performs no clear in the `I`-out-of-bag flat regime.** When the `I` piece is
absent from `g.bag` and `g.board` is flat (every row `≤ 6`), the adversary's forced piece `p ∈ g.bag`
is non-`I`, so the resulting board is *exactly* the placement with no line clear:
`(adversarialStep ... g p pl).board = ({pl with piece := p}).place g.board`. Composed with
`count_place` this says the cell count rises by `+4` with no relief — the player is locked into pure
stacking for the post-`I` portion of every bag spent on a flat board. A FALSE-leaning sharpening
(conditional on flatness), not a verdict flip. -/
theorem adversarialStep_board_eq_place_of_notMem_bag_I_of_forall_rowCount_le_six
    (g : GameState) (p : Piece) (hp : p ∈ g.bag) (hI : Piece.I ∉ g.bag)
    (pl : Placement) (hrows : ∀ r, rowCount g.board r ≤ 6) :
    (adversarialStep GameConfig.standard g p pl).board
      = ({pl with piece := p} : Placement).place g.board := by
  rw [adversarialStep_board, Placement.applyStep_eq_clearLines_place]
  apply Board.clearLines_eq_self_of_no_fullRows
  exact fullRows_place_eq_empty_of_notMem_bag_I_of_forall_rowCount_le_six
    g.board g.bag hI ({pl with piece := p}) hp hrows

/-! ## Per-bag piece-count: each piece is drawn exactly once per 7-bag cycle

The `7`-bag randomizer draws all seven distinct pieces before refilling, so within one bag
(`bagAt Bag.full s n` for `n ∈ [0,6]`, before the refill at step `7`) no piece repeats. In
particular the `I` piece — the only one that can complete a row from fewer than `7` filled cells
(`seven_le_rowCount_of_mem_fullRows_place_of_ne_I`) — is delivered at most once per bag. This caps
the adversary's "deep" row completions to one per bag, the per-bag I-count the floor/ceiling race
needs. -/

/-- **Within the first bag a legal draw simply erases.** For `n ≤ 5` the bag still has
`7 - n ≥ 2` pieces, so the draw does not refill and `bagAt Bag.full s (n+1)` is exactly the erase. -/
theorem bagAt_succ_eq_erase_of_le_five {s : ℕ → Piece} (hl : LegalSequence s)
    (n : ℕ) (hn : n ≤ 5) :
    bagAt Bag.full s (n + 1) = (bagAt Bag.full s n).erase (s n) := by
  rw [bagAt_succ]
  apply Bag.draw_eq_erase_of_card_ge_two (hl n)
  rw [bagAt_card_eq_seven_sub hl n (by omega)]
  omega

/-- **The first bag is antitone** (shrinks by erase) across its seven draws: for `i + d ≤ 6`,
`bagAt Bag.full s (i + d) ⊆ bagAt Bag.full s i`. No refill happens before step `7`. -/
theorem bagAt_antitone_first_bag {s : ℕ → Piece} (hl : LegalSequence s) :
    ∀ (i d : ℕ), i + d ≤ 6 → bagAt Bag.full s (i + d) ⊆ bagAt Bag.full s i := by
  intro i d
  induction d with
  | zero => intro _; exact Finset.Subset.refl _
  | succ d ih =>
      intro h
      have hstep : bagAt Bag.full s (i + d + 1) ⊆ bagAt Bag.full s (i + d) := by
        rw [bagAt_succ_eq_erase_of_le_five hl (i + d) (by omega)]
        exact Finset.erase_subset _ _
      have hgoal : bagAt Bag.full s (i + (d + 1)) ⊆ bagAt Bag.full s (i + d) := by
        rw [show i + (d + 1) = i + d + 1 from rfl]; exact hstep
      exact hgoal.trans (ih (by omega))

/-- **Pairwise distinctness of the seven draws of one bag.** For `i < j ≤ 6` the draws `s i` and
`s j` differ: `s i` is erased from the bag at step `i+1` and never returns before the refill, while
`s j` is still drawable at step `j`. -/
theorem draw_ne_of_lt_first_bag {s : ℕ → Piece} (hl : LegalSequence s)
    {i j : ℕ} (hij : i < j) (hj : j ≤ 6) : s i ≠ s j := by
  intro heq
  have hni : s i ∉ bagAt Bag.full s (i + 1) := by
    rw [bagAt_succ_eq_erase_of_le_five hl i (by omega)]
    exact Finset.notMem_erase _ _
  have hsub : bagAt Bag.full s j ⊆ bagAt Bag.full s (i + 1) := by
    have hje : (i + 1) + (j - (i + 1)) = j := by omega
    rw [← hje]
    exact bagAt_antitone_first_bag hl (i + 1) (j - (i + 1)) (by omega)
  have hmj : s j ∈ bagAt Bag.full s j := hl j
  have hsi : s i ∈ bagAt Bag.full s j := by rw [heq]; exact hmj
  exact hni (hsub hsi)

/-- **Per-bag `I`-count: at most one of the first seven legal draws is the `I` piece.** Immediate
from `draw_ne_of_lt_first_bag` (the seven draws are pairwise distinct). Combined with
`seven_le_rowCount_of_mem_fullRows_place_of_ne_I` (a non-`I` placement can clear a row only when that
row already held `≥ 7` cells), this bounds the number of "deep" row completions — clears of a row
that was below `7` — by at most one per bag. -/
theorem atMostOne_I_first_bag {s : ℕ → Piece} (hl : LegalSequence s) :
    ((Finset.range 7).filter (fun n => s n = Piece.I)).card ≤ 1 := by
  rw [Finset.card_le_one]
  intro a ha b hb
  simp only [Finset.mem_filter, Finset.mem_range] at ha hb
  obtain ⟨ha7, haI⟩ := ha
  obtain ⟨hb7, hbI⟩ := hb
  by_contra hne
  rcases Nat.lt_or_ge a b with hlt | hge
  · exact draw_ne_of_lt_first_bag hl hlt (by omega) (haI.trans hbI.symm)
  · have hba : b < a := by omega
    exact draw_ne_of_lt_first_bag hl hba (by omega) (hbI.trans haI.symm)

/-- **Global per-bag `I`-count.** Lifts `atMostOne_I_first_bag` from the first bag to *every* bag:
within the seven draws of the `m`-th bag (indices `7·m … 7·m+6`) at most one is the `I` piece. The
suffix `fun k => s (7·m + k)` starting at a full-bag boundary (`bagAt_mul_seven_eq_full`) is itself a
`LegalSequence` (via `LegalSequenceFrom.suffix`), so the first-bag bound applies verbatim. -/
theorem atMostOne_I_bag {s : ℕ → Piece} (hl : LegalSequence s) (m : ℕ) :
    ((Finset.range 7).filter (fun k => s (7 * m + k) = Piece.I)).card ≤ 1 := by
  have hsuf := hl.suffix (7 * m)
  rw [bagAt_mul_seven_eq_full hl m] at hsuf
  exact atMostOne_I_first_bag hsuf

/-- **Global per-bag `I`-count over absolute indices.** The same bound stated over the absolute
index window `Ico (7·m) (7·m+7)` rather than the shifted `range 7`: at most one index in the `m`-th
bag draws the `I` piece. Reindexes `atMostOne_I_bag` through the injective shift `k ↦ 7·m + k`. -/
theorem atMostOne_I_Ico {s : ℕ → Piece} (hl : LegalSequence s) (m : ℕ) :
    ((Finset.Ico (7 * m) (7 * m + 7)).filter (fun k => s k = Piece.I)).card ≤ 1 := by
  have himg : (Finset.Ico (7 * m) (7 * m + 7)).filter (fun k => s k = Piece.I)
      = ((Finset.range 7).filter (fun k => s (7 * m + k) = Piece.I)).image (fun k => 7 * m + k) := by
    ext x
    simp only [Finset.mem_filter, Finset.mem_image, Finset.mem_Ico, Finset.mem_range]
    constructor
    · rintro ⟨⟨hlo, hhi⟩, hI⟩
      exact ⟨x - 7 * m, ⟨by omega, by rw [Nat.add_sub_cancel' hlo]; exact hI⟩, by omega⟩
    · rintro ⟨k, ⟨hk, hI⟩, rfl⟩
      exact ⟨⟨by omega, by omega⟩, hI⟩
  rw [himg, Finset.card_image_of_injective _ (by intro a b h; simpa using h)]
  exact atMostOne_I_bag hl m

/-- **Cumulative `I`-count over `n` bags.** Summing the global per-bag bound: among the first `7·n`
legal draws (the first `n` complete bags) at most `n` are the `I` piece. Inducts on `n`, splitting
`range (7·(n+1)) = range (7·n) ∪ Ico (7·n) (7·n+7)`; the prefix is bounded by the induction
hypothesis (`≤ n`) and the new bag block by `atMostOne_I_Ico` (`≤ 1`). Bounds the *total* number of
"deep" row completions (clears of a row that held `< 7` cells) over any `n`-bag horizon by `n`. -/
theorem atMostOne_I_cumulative {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    ((Finset.range (7 * n)).filter (fun k => s k = Piece.I)).card ≤ n := by
  induction n with
  | zero => simp
  | succ n ih =>
      have hsplit : Finset.range (7 * (n + 1))
          = Finset.range (7 * n) ∪ Finset.Ico (7 * n) (7 * n + 7) := by
        ext x
        simp only [Finset.mem_range, Finset.mem_union, Finset.mem_Ico]
        omega
      rw [hsplit, Finset.filter_union]
      refine le_trans (Finset.card_union_le _ _) ?_
      have h2 := atMostOne_I_Ico hl n
      omega

/-- **The first bag's seven draws cover every piece.** The seven draws `s 0 … s 6` are pairwise
distinct (`draw_ne_of_lt_first_bag`), hence `s` is injective on `range 7`; an injective map from a
7-element set into the 7-element `Piece` type is onto, so the image is all of `univ`. -/
theorem image_first_bag_eq_univ {s : ℕ → Piece} (hl : LegalSequence s) :
    (Finset.range 7).image s = Finset.univ := by
  have hinj : Set.InjOn s (Finset.range 7) := by
    intro a ha b hb hab
    simp only [Finset.mem_coe, Finset.mem_range] at ha hb
    by_contra hne
    rcases Nat.lt_or_ge a b with hlt | hge
    · exact draw_ne_of_lt_first_bag hl hlt (by omega) hab
    · exact draw_ne_of_lt_first_bag hl (by omega) (by omega) hab.symm
  apply Finset.eq_univ_of_card
  rw [Finset.card_image_of_injOn hinj, Finset.card_range]
  decide

/-- **At least one of the first seven draws is the `I` piece.** Immediate from
`image_first_bag_eq_univ`: `I ∈ univ` means some draw `s k = I`. -/
theorem one_le_I_first_bag {s : ℕ → Piece} (hl : LegalSequence s) :
    1 ≤ ((Finset.range 7).filter (fun k => s k = Piece.I)).card := by
  apply Finset.card_pos.mpr
  have hmem : Piece.I ∈ (Finset.range 7).image s := by
    rw [image_first_bag_eq_univ hl]; exact Finset.mem_univ _
  obtain ⟨k, hk, hkI⟩ := Finset.mem_image.mp hmem
  exact ⟨k, Finset.mem_filter.mpr ⟨hk, hkI⟩⟩

/-- **Exactly one of the first seven legal draws is the `I` piece.** Combines the upper bound
`atMostOne_I_first_bag` (`≤ 1`) with the lower bound `one_le_I_first_bag` (`≥ 1`). The `7`-bag
delivers `I` precisely once per cycle. -/
theorem exactlyOne_I_first_bag {s : ℕ → Piece} (hl : LegalSequence s) :
    ((Finset.range 7).filter (fun k => s k = Piece.I)).card = 1 :=
  le_antisymm (atMostOne_I_first_bag hl) (one_le_I_first_bag hl)

/-- **Every deep completion is an `I`-placement.** Contrapositive of
`seven_le_rowCount_of_mem_fullRows_place_of_ne_I`: if placing `pl` on `b` fills row `r` (`r ∈ fullRows
(pl.place b)`) and that row held *fewer than `7`* cells beforehand (`rowCount b r < 7`), the placed
piece must be `I` — the only piece that can add four cells to one row. This is the first link between
the abstract per-bag `I`-count and the board trajectory: a "deep" row completion forces an `I`. -/
theorem piece_eq_I_of_mem_fullRows_place_of_rowCount_lt_seven
    (b : Board) (pl : Placement) {r : ℕ}
    (hr : r ∈ Board.fullRows GameConfig.standard (pl.place b))
    (hlt : rowCount b r < 7) :
    pl.piece = Piece.I := by
  by_contra hne
  have h := seven_le_rowCount_of_mem_fullRows_place_of_ne_I b pl hne hr
  omega

/-- **Exactly one `I` in every bag.** Lifts `exactlyOne_I_first_bag` to bag `m`: the suffix
`fun k => s (7·m + k)` starts at a full-bag boundary (`bagAt_mul_seven_eq_full`), hence is itself a
`LegalSequence`, so its first bag draws `I` exactly once. -/
theorem exactlyOne_I_bag {s : ℕ → Piece} (hl : LegalSequence s) (m : ℕ) :
    ((Finset.range 7).filter (fun k => s (7 * m + k) = Piece.I)).card = 1 := by
  have hsuf := hl.suffix (7 * m)
  rw [bagAt_mul_seven_eq_full hl m] at hsuf
  exact exactlyOne_I_first_bag hsuf

/-- **Step-level deep-completion bridge.** Lifts `piece_eq_I_of_mem_fullRows_place_of_rowCount_lt_seven`
to the game step: if drawing a *non-`I`* piece `p` and placing it via `pl` fills at least one row of
`g.board` (`fullRows` of the post-placement board is nonempty), then `g.board` already contained a
row with `≥ 7` cells. A non-`I` move that clears a line presupposes a pre-built near-full row. -/
theorem exists_seven_le_rowCount_of_clears_of_ne_I
    (g : GameState) (p : Piece) (pl : Placement) (hp : p ≠ Piece.I)
    (hclear : (Board.fullRows GameConfig.standard
        (({pl with piece := p} : Placement).place g.board)).Nonempty) :
    ∃ r, 7 ≤ rowCount g.board r := by
  obtain ⟨r, hr⟩ := hclear
  exact ⟨r, seven_le_rowCount_of_mem_fullRows_place_of_ne_I g.board
    ({pl with piece := p}) hp hr⟩

/-- Same bridge phrased via the line-clear count: a non-`I` placement that clears `≥ 1` line
(`0 < linesCleared`) requires a pre-existing `≥ 7`-cell row. -/
theorem exists_seven_le_rowCount_of_linesCleared_pos_of_ne_I
    (g : GameState) (p : Piece) (pl : Placement) (hp : p ≠ Piece.I)
    (hclear : 0 < Board.linesCleared GameConfig.standard
        (({pl with piece := p} : Placement).place g.board)) :
    ∃ r, 7 ≤ rowCount g.board r :=
  exists_seven_le_rowCount_of_clears_of_ne_I g p pl hp (Finset.card_pos.mp hclear)

/-- **Cell count rises by exactly `+4` in the `I`-out-of-bag flat regime.** The count companion to
`adversarialStep_board_eq_place_of_notMem_bag_I_of_forall_rowCount_le_six` (its docstring anticipates
this): when `I` is absent from `g.bag` and `g.board` is flat (every row `≤ 6`), the forced non-`I`
placement clears nothing, so the board gains exactly the four cells of the piece with no clearing
relief. Composing the no-clear board identity with `count_place` (`+4`), the player is locked into a
strict `+4` stacking step for the post-`I` portion of every bag spent on a flat board. -/
theorem count_adversarialStep_eq_of_notMem_bag_I_of_forall_rowCount_le_six
    (g : GameState) (p : Piece) (hp : p ∈ g.bag) (hI : Piece.I ∉ g.bag)
    (pl : Placement) (hrows : ∀ r, rowCount g.board r ≤ 6) :
    (adversarialStep GameConfig.standard g p pl).board.count = g.board.count + 4 := by
  rw [adversarialStep_board_eq_place_of_notMem_bag_I_of_forall_rowCount_le_six g p hp hI pl hrows]
  exact Placement.count_place g.board ({pl with piece := p})

/-- **`count + holes` strictly climbs (`≥ +4`) in the `I`-out-of-bag flat regime.** Combines the exact
`+4` count gain (`count_adversarialStep_eq_of_notMem_bag_I_of_forall_rowCount_le_six`) with hole
monotonicity across a pure placement (`holes_le_holes_place`: a hard drop can only bury, never fill,
holes). Since the step performs no clear, no hole-removal relief applies, so the conserved budget
quantity `count + holes` — the one pinned `≤ 200` on reachable boards by
`count_add_holes_le_two_hundred_of_reachableAt` — rises by at least `4`. This is the per-step engine of
the FALSE budget-divergence route, valid for as long as the board stays flat with `I` out of bag. -/
theorem count_add_holes_adversarialStep_ge_of_notMem_bag_I_of_forall_rowCount_le_six
    (g : GameState) (p : Piece) (hp : p ∈ g.bag) (hI : Piece.I ∉ g.bag)
    (pl : Placement) (hrows : ∀ r, rowCount g.board r ≤ 6) :
    g.board.count + holes GameConfig.standard g.board + 4
      ≤ (adversarialStep GameConfig.standard g p pl).board.count
        + holes GameConfig.standard (adversarialStep GameConfig.standard g p pl).board := by
  have hcount :=
    count_adversarialStep_eq_of_notMem_bag_I_of_forall_rowCount_le_six g p hp hI pl hrows
  have hholes : holes GameConfig.standard g.board
      ≤ holes GameConfig.standard (adversarialStep GameConfig.standard g p pl).board := by
    rw [adversarialStep_board_eq_place_of_notMem_bag_I_of_forall_rowCount_le_six g p hp hI pl hrows]
    exact holes_le_holes_place GameConfig.standard g.board ({pl with piece := p})
  omega

/-- **Step-level cell-count conservation law.** The general adversarial-step form of `applyStep_count`:
for any drawn piece `p` placed by a *valid* `pl` on a well-formed board, the post-step cell count plus
`cols ·` (lines cleared this step) equals the pre-step count `+ 4`. Unwinding: each placement deposits
four cells (`count_place`) and every cleared line removes exactly `cols` of them. This is the exact
count recurrence the budget/clear-rate route is built on — there is no slack and no hypothesis beyond
well-formedness and placement validity (both hold along any `ReachableAt` trajectory whose placements
come from `allValidFor`). -/
theorem count_add_cols_mul_linesCleared_adversarialStep_eq
    (g : GameState) (p : Piece) (pl : Placement)
    (hb : Board.WF GameConfig.standard g.board)
    (hv : ({pl with piece := p} : Placement).Valid GameConfig.standard) :
    (adversarialStep GameConfig.standard g p pl).board.count
      + GameConfig.standard.cols * Board.linesCleared GameConfig.standard
          (({pl with piece := p} : Placement).place g.board)
      = g.board.count + 4 := by
  rw [adversarialStep_board]
  exact applyStep_count GameConfig.standard g.board ({pl with piece := p}) hb hv

/-- **Per-step count rises by at most `+4`.** Immediate from the conservation law
`count_add_cols_mul_linesCleared_adversarialStep_eq`: the cleared-line term `cols · linesCleared` is
non-negative, so the post-step count can exceed the pre-step count by no more than the four cells the
piece deposits — and falls below it whenever a line clears. The upper companion to the no-clear `= +4`
gain (`count_adversarialStep_eq_of_notMem_bag_I_of_forall_rowCount_le_six`); together with the budget
`count + holes ≤ 200` this caps how fast a reachable board's cell count can grow. -/
theorem count_adversarialStep_le_add_four
    (g : GameState) (p : Piece) (pl : Placement)
    (hb : Board.WF GameConfig.standard g.board)
    (hv : ({pl with piece := p} : Placement).Valid GameConfig.standard) :
    (adversarialStep GameConfig.standard g p pl).board.count ≤ g.board.count + 4 := by
  have h := count_add_cols_mul_linesCleared_adversarialStep_eq g p pl hb hv
  omega

/-- **Row-fiber sum bound.** For any finite set `S` of rows, the filled cells across those rows sum to
no more than the whole board: `∑_{r ∈ S} rowCount b r ≤ count b`. The row cells partition into disjoint
fibers (`card_eq_sum_card_fiberwise`), so summing any sub-collection of rows undercounts the board. The
per-row analogue of the columnwise `sum_colCount`, but with no well-formedness hypothesis and over an
arbitrary row set. -/
theorem sum_rowCount_le (b : Board) (S : Finset ℕ) :
    ∑ r ∈ S, rowCount b r ≤ b.count := by
  have hfib : (b.filter (fun p => p.2 ∈ S)).card
      = ∑ r ∈ S, ((b.filter (fun p => p.2 ∈ S)).filter (fun p => p.2 = r)).card :=
    Finset.card_eq_sum_card_fiberwise (fun p hp => (Finset.mem_filter.mp hp).2)
  have heq : ∀ r ∈ S, ((b.filter (fun p => p.2 ∈ S)).filter (fun p => p.2 = r)).card
      = rowCount b r := by
    intro r hr
    unfold rowCount
    congr 1
    ext p
    simp only [Finset.mem_filter]
    constructor
    · rintro ⟨⟨hpb, _⟩, hpr⟩; exact ⟨hpb, hpr⟩
    · rintro ⟨hpb, hpr⟩; exact ⟨⟨hpb, hpr ▸ hr⟩, hpr⟩
  rw [Finset.sum_congr rfl heq] at hfib
  rw [← hfib]
  unfold Board.count
  exact Finset.card_filter_le b _

/-- **Near-full rows are scarce: `7 · |{near-full rows}| ≤ count`.** Any set `S` of rows each holding
`≥ 7` cells collectively consumes `≥ 7 · |S|` cells, bounded by the board total (`sum_rowCount_le`).
Hence at most `count / 7` rows can sit at `≥ 7` cells simultaneously — at most `⌊200/7⌋ = 28` on a
reachable board (budget `count ≤ 200`). **Verdict significance.** This is a direct geometric brake on
the `≥ 7`-prebuild escape hatch — the route by which the player clears lines *without* the `I` piece
(non-`I` pieces add `≤ 3` to a row, so a non-`I` clear needs the row pre-stacked to `≥ 7`), the very
mechanism that makes the per-bag `I`-count unable to throttle clears. The player simply cannot keep
arbitrarily many near-full rows in flight. FALSE-side infra: it caps the inventory of shallow-clearable
rows, not (yet) the clear *rate*. -/
theorem seven_mul_card_le_count_of_forall_rowCount_ge_seven (b : Board) (S : Finset ℕ)
    (hS : ∀ r ∈ S, 7 ≤ rowCount b r) :
    7 * S.card ≤ b.count := by
  have h7 : ∑ _r ∈ S, (7 : ℕ) = 7 * S.card := by
    rw [Finset.sum_const, smul_eq_mul]; ring
  calc 7 * S.card = ∑ _r ∈ S, (7 : ℕ) := h7.symm
    _ ≤ ∑ r ∈ S, rowCount b r := Finset.sum_le_sum hS
    _ ≤ b.count := sum_rowCount_le b S

/-- **Row-fiber partition of the cell count.** When every occupied cell sits in a row `< R`, the total
count is exactly the sum of the per-row counts over `range R` — the row analogue of `sum_colCount`
(which partitions by column under `WF`). The hypothesis `∀ p ∈ b, p.2 < R` is the row bound supplied
by `not_lost_in_field` (`¬ lost ⇒ every cell has row `< cfg.rows``). -/
theorem sum_rowCount_range_eq_count (b : Board) (R : ℕ)
    (hb : ∀ p ∈ b, p.2 < R) :
    ∑ r ∈ Finset.range R, rowCount b r = b.count := by
  unfold rowCount Board.count
  exact (Finset.card_eq_sum_card_fiberwise (fun p hp => Finset.mem_range.2 (hb p hp))).symm

/-- **Flat self-cap: a flat, not-lost board holds `≤ 6 · rows` cells.** If every row carries `≤ 6`
cells (`flat`) and the board is not lost (every cell row `< cfg.rows`, via `not_lost_in_field`), then
`count ≤ 6 · cfg.rows` (`= 120` on `standard`). **Verdict significance.** This *sharpens* the generic
reachable budget `count ≤ cols·rows = 200` (`reachable_not_lost_count_le`) down to `120` *whenever the
board stays flat*. Paired with iter-174's `seven_mul_card_le_count_of_forall_rowCount_ge_seven`, it
frames the entire count budget around `≥ 7` rows: to push count from the flat ceiling `120` toward the
hard ceiling `200`, the player **must** open `≥ 7`-filled rows. This is FALSE-side infra/sharpening, not
a verdict flip — the flat window alone cannot breach the budget, but the player is free to leave it by
prebuilding `≥ 7` rows (the unthrottled escape hatch). It pins down *where* the count budget lives. -/
theorem count_le_six_mul_rows_of_forall_rowCount_le_six_of_not_lost
    {cfg : GameConfig} {g : GameState} (hnl : ¬ g.lost cfg)
    (hflat : ∀ r, rowCount g.board r ≤ 6) :
    g.board.count ≤ 6 * cfg.rows := by
  rw [← sum_rowCount_range_eq_count g.board cfg.rows (not_lost_in_field hnl)]
  calc ∑ r ∈ Finset.range cfg.rows, rowCount g.board r
      ≤ ∑ _r ∈ Finset.range cfg.rows, 6 := Finset.sum_le_sum (fun r _ => hflat r)
    _ = 6 * cfg.rows := by rw [Finset.sum_const, Finset.card_range, smul_eq_mul]; ring

/-- **A row holds at most `cols` cells** on a well-formed board: distinct cells in row `r` carry
distinct column indices (a cell is `(col, r)`), so the column projection injects row `r`'s cell-fiber
into `range cfg.cols`. The per-row ceiling complementing `colCount_le_count`; a *full* row hits this
bound with equality (`rowCount = cols`). -/
theorem rowCount_le_cols {cfg : GameConfig} {b : Board} (hb : Board.WF cfg b) (r : ℕ) :
    rowCount b r ≤ cfg.cols := by
  unfold rowCount
  have hinj : Set.InjOn (fun p : ℕ × ℕ => p.1) ↑(b.filter (fun p => p.2 = r)) := by
    intro p hp q hq hpq
    simp only [Finset.mem_coe, Finset.mem_filter] at hp hq
    exact Prod.ext hpq (hp.2.trans hq.2.symm)
  calc (b.filter (fun p => p.2 = r)).card
      = ((b.filter (fun p => p.2 = r)).image (fun p => p.1)).card :=
        (Finset.card_image_of_injOn hinj).symm
    _ ≤ (Finset.range cfg.cols).card := by
        apply Finset.card_le_card
        intro x hx
        simp only [Finset.mem_image, Finset.mem_filter] at hx
        obtain ⟨p, ⟨hpb, _⟩, rfl⟩ := hx
        exact Finset.mem_range.2 (hb p hpb)
    _ = cfg.cols := Finset.card_range cfg.cols

/-- **Flat-ceiling dichotomy: exceeding `6 · rows` forces an open `≥ 7` row.** Contrapositive of the
flat self-cap (`count_le_six_mul_rows_of_forall_rowCount_le_six_of_not_lost`): on a not-lost board, if
the count climbs past the flat ceiling `6 · cfg.rows` (`= 120` on `standard`), some row already holds
`≥ 7` cells. **Verdict significance.** This is the usable existential form of "to reach the upper count
budget the player *must* open near-full rows" — the precise junction where iter-175's flat cap
(`≤ 120`) hands off to iter-174's near-full-row inventory cap (`≤ 28`). Still FALSE-side framing, not a
flip: it pins *that* `≥ 7` rows must exist at high count, not a bound on how fast the player can clear
them. -/
theorem exists_rowCount_ge_seven_of_not_lost_of_six_mul_rows_lt_count
    {cfg : GameConfig} {g : GameState} (hnl : ¬ g.lost cfg)
    (hcount : 6 * cfg.rows < g.board.count) :
    ∃ r, 7 ≤ rowCount g.board r := by
  by_contra h
  push_neg at h
  have hcap := count_le_six_mul_rows_of_forall_rowCount_le_six_of_not_lost hnl
    (fun r => by have := h r; omega)
  omega

/-- **Refined high-row lower bound.** On a well-formed, in-field `standard` board, partition the rows
into "low" (`rowCount ≤ 6`) and "high" (`rowCount ≥ 7`). Each low row contributes `≤ 6`; each high row
contributes `≤ cols = 10` (`rowCount_le_cols`); so pointwise `rowCount r ≤ 6 + 4·[high r]`. Summing
over all `rows = 20` rows (via `sum_rowCount_range_eq_count`) gives
`count ≤ 4·|{high rows}| + 6·rows`. Rearranged: `|{high rows}| ≥ (count − 6·rows)/4`. **Verdict
significance.** This is the *lower* companion to iter-174's inventory cap `|{high rows}| ≤ count/7`. On a
budget board (`count ≤ 200`, `rows = 20`) the two PIN the near-full-row population to
`[(count−120)/4, count/7]` — e.g. at `count = 200`, between `20` and `28` rows must sit at `≥ 7` cells
*simultaneously*. Still FALSE-side: it bounds the standing near-full inventory, not the clear *rate*. -/
theorem count_le_four_mul_high_add_six_mul_rows {b : Board}
    (hwf : Board.WF GameConfig.standard b)
    (hfield : ∀ p ∈ b, p.2 < GameConfig.standard.rows) :
    b.count ≤ 4 * ((Finset.range GameConfig.standard.rows).filter
        (fun r => 7 ≤ rowCount b r)).card + 6 * GameConfig.standard.rows := by
  classical
  have hcols : GameConfig.standard.cols = 10 := by decide
  have hpt : ∀ r ∈ Finset.range GameConfig.standard.rows,
      rowCount b r ≤ 6 + 4 * (if 7 ≤ rowCount b r then 1 else 0) := by
    intro r _
    by_cases h : 7 ≤ rowCount b r
    · have hle := rowCount_le_cols hwf r
      rw [hcols] at hle
      simp only [h, if_true]
      omega
    · simp only [h, if_false]
      omega
  have hbound : b.count
      ≤ ∑ r ∈ Finset.range GameConfig.standard.rows,
          (6 + 4 * (if 7 ≤ rowCount b r then 1 else 0)) := by
    rw [← sum_rowCount_range_eq_count b GameConfig.standard.rows hfield]
    exact Finset.sum_le_sum hpt
  have h1 : ∑ _r ∈ Finset.range GameConfig.standard.rows, (6 : ℕ)
      = 6 * GameConfig.standard.rows := by
    rw [Finset.sum_const, Finset.card_range, smul_eq_mul]; ring
  have h2 : ∑ r ∈ Finset.range GameConfig.standard.rows,
        4 * (if 7 ≤ rowCount b r then 1 else 0)
      = 4 * ((Finset.range GameConfig.standard.rows).filter (fun r => 7 ≤ rowCount b r)).card := by
    rw [← Finset.mul_sum, ← Finset.card_filter]
  have heq : ∑ r ∈ Finset.range GameConfig.standard.rows,
        (6 + 4 * (if 7 ≤ rowCount b r then 1 else 0))
      = 6 * GameConfig.standard.rows
        + 4 * ((Finset.range GameConfig.standard.rows).filter (fun r => 7 ≤ rowCount b r)).card := by
    rw [Finset.sum_add_distrib, h1, h2]
  omega

/-- **Two-sided pin on the near-full-row inventory of any reachable, not-lost board.**
Bundles iter-177's refined lower bound (`count ≤ 4·|high| + 6·rows`) with iter-174's
inventory cap (`7·|high| ≤ count`), where `high := {r ∈ range rows : rowCount ≥ 7}`.
On `standard` at the worst case `count = 200`, `rows = 20`, this forces
`|high| ∈ [20, 28]`: at least `(200 − 120)/4 = 20` and at most `200/7 < 29` rows hold
≥7 cells simultaneously. Reachability supplies `WF` (via `ReachableAt.wf`) and
not-lost supplies the row-field bound (via `not_lost_in_field`).

FALSE-side framing: this pins the board *state* (how many near-full rows must coexist),
not the line-clear *rate*. The verdict stays open; carryover of near-full rows across
bags is the genuine obstacle a rate bound would have to defeat. -/
theorem high_rows_pinned_of_reachableAt {n : ℕ} {g : GameState}
    (h : ReachableAt GameConfig.standard n g) (hnl : ¬ g.lost GameConfig.standard) :
    g.board.count ≤ 4 * ((Finset.range GameConfig.standard.rows).filter
        (fun r => 7 ≤ rowCount g.board r)).card + 6 * GameConfig.standard.rows
      ∧ 7 * ((Finset.range GameConfig.standard.rows).filter
        (fun r => 7 ≤ rowCount g.board r)).card ≤ g.board.count := by
  refine ⟨?_, ?_⟩
  · exact count_le_four_mul_high_add_six_mul_rows h.wf (not_lost_in_field hnl)
  · exact seven_mul_card_le_count_of_forall_rowCount_ge_seven g.board _
      (fun r hr => (Finset.mem_filter.mp hr).2)

/-- **Cell-budget backbone for the per-bag rate route.** Applying a list of `pls`
placements in sequence — with *no* intervening line clears — adds exactly
`4 * pls.length` cells to the board. Each individual placement contributes exactly
4 (`Placement.count_place`, unconditional); folding over the list accumulates the
budget. This is the unconditional foundation the per-bag production bound rests on:
a bag is seven placements, hence a 28-cell budget. -/
theorem count_foldl_place (b : Board) (pls : List Placement) :
    (pls.foldl (fun acc pl => pl.place acc) b).count = b.count + 4 * pls.length := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    simp only [List.foldl_cons, List.length_cons]
    rw [ih (pl.place b), Placement.count_place]
    omega

/-- **One-bag cell budget.** Seven no-clear placements (a full 7-bag, pre-clear)
add exactly 28 cells. The concrete instance of `count_foldl_place` at `length = 7`;
the numerator of the per-bag `floor(28/7) = 4` near-full-row production cap. -/
theorem count_foldl_place_bag (b : Board) (pls : List Placement)
    (h7 : pls.length = 7) :
    (pls.foldl (fun acc pl => pl.place acc) b).count = b.count + 28 := by
  rw [count_foldl_place]; omega

/-- **Per-row placement monotonicity (the lower companion to `rowCount_place_le`).**
A placement never *removes* cells from any row: `place` is union-with-`dropped`, so the
row-`r` fiber of `b` injects into that of `pl.place b`. The "deposited cells ≥ 0" half of
the per-row creation-cost accounting — together with `rowCount_place_le` it brackets a
single placement's per-row delta in `[0, 4]`. -/
theorem rowCount_place_ge (b : Board) (pl : Placement) (r : ℕ) :
    rowCount b r ≤ rowCount (pl.place b) r := by
  unfold rowCount
  apply Finset.card_le_card
  intro p hp
  rw [Finset.mem_filter] at hp ⊢
  refine ⟨?_, hp.2⟩
  rw [Placement.place_eq_union_dropped, Finset.mem_union]
  exact Or.inl hp.1

/-- **Per-row sequence monotonicity.** Across any list of no-clear placements a row's count
never drops below its starting value (fold of `rowCount_place_ge`). This is the structural
core of the "creation cost" claim: with no clears, a row that ends at `rowCount ≥ 7` from an
empty start must have had `≥ 7` cells deposited into it over the sequence. -/
theorem rowCount_foldl_place_ge (b : Board) (pls : List Placement) (r : ℕ) :
    rowCount b r ≤ rowCount (pls.foldl (fun acc pl => pl.place acc) b) r := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    simp only [List.foldl_cons]
    have h1 := rowCount_place_ge b pl r
    have h2 := ih (pl.place b)
    omega

/-- **Per-row sequence envelope (upper).** The per-row analogue of `count_foldl_place`: a row's
count grows by at most `4 * pls.length` over a no-clear placement list (fold of `rowCount_place_le`).
Brackets, with `rowCount_foldl_place_ge`, any row across a sequence to
`[rowCount b r, rowCount b r + 4*length]`. -/
theorem rowCount_foldl_place_le (b : Board) (pls : List Placement) (r : ℕ) :
    rowCount (pls.foldl (fun acc pl => pl.place acc) b) r ≤ rowCount b r + 4 * pls.length := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    simp only [List.foldl_cons, List.length_cons]
    have h1 := ih (pl.place b)
    have h2 := rowCount_place_le b pl r
    omega

/-- **Hole monotonicity across a no-clear placement sequence.** Holes never decrease over a
list of placements applied without intervening clears (fold of `holes_le_holes_place`). The
hole-side companion to `count_foldl_place` / `rowCount_foldl_place_ge`, completing the
count/row/hole fold trio. This is the *creation* side of the hole-rate question: across a
clear-free run holes only climb. By itself this is FALSE-side infra, NOT a flip — the
matching *removal* side (`holes_clearLines_le`) shows a single clear can claw back an
unbounded number of holes, so monotone creation over a no-clear stretch yields no per-bag
creation-beats-removal contradiction (same cross-bag/cheap-removal blocker as the count- and
row-production routes). -/
theorem holes_le_holes_foldl_place (cfg : GameConfig) (b : Board) (pls : List Placement) :
    holes cfg b ≤ holes cfg (pls.foldl (fun acc pl => pl.place acc) b) := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    simp only [List.foldl_cons]
    exact le_trans (holes_le_holes_place cfg b pl) (ih (pl.place b))

/-! ### The FALSE-side adversarial monovariant principle (iter-182)

The file already carries the TRUE-side Lyapunov route (`tetrisSolvable_of_Lyapunov`,
`safe_of_Lyapunov_function`): a bounded-*sublevel* invariant set containing `init`
proves solvability. Its logical *dual* — the principle a FALSE proof must use — was
missing. We supply it here as a finite pigeonhole argument, with no `decide` of any
kind.

The principle: an **adversarial monovariant** is a `f : GameState -> Nat` such that
from every non-lost state the adversary has a piece all of whose in-set landings
*strictly raise* `f`. No finite set can be closed under such an `f`: take the
`f`-maximal member (finite, nonempty), and the adversary forces a strictly-greater-`f`
successor still inside the set — impossible.

Why this CRYSTALLIZES the impasse rather than settling it. To flip the verdict FALSE
one must exhibit such an `f` for the canonical game. The natural candidate `f = holes`
satisfies the *creation* half (`one_le_holes_S_applyStep_empty`: S buries a cell on a
flat surface) but FAILS the universally-quantified hypothesis: the player is not forced
to raise holes, because a line clear claws holes back (`holes_clearLines_le`). So the
"for ALL valid landings `f` strictly increases" clause is exactly the empirically-false
obligation — the same cheap-removal / cross-bag blocker that killed the count-, row- and
hole-rate routes, now named as a single precise missing hypothesis. -/

/-- **No finite set is closed under an adversarial monovariant.** If for every non-lost
`g in T` the adversary owns a piece `p in g.bag` whose every valid landing that stays in
`T` strictly increases `f`, then `T` cannot be closed. Pure finite pigeonhole: the
`f`-maximal element of `T` would be forced to step to a strictly-greater-`f` member of
`T`. No `decide`, no clears reasoned about — entirely structural. -/
theorem not_isClosedSet_of_adversarial_monovariant
    (cfg : GameConfig) (T : Finset GameState) (hne : T.Nonempty)
    (f : GameState → ℕ)
    (hmono : ∀ g ∈ T, ¬ g.lost cfg →
      ∃ p ∈ g.bag, ∀ pl ∈ Placement.allValidFor cfg p,
        adversarialStep cfg g p pl ∈ T →
          f g < f (adversarialStep cfg g p pl)) :
    ¬ IsClosedSet cfg T := by
  intro hclosed
  obtain ⟨gmax, hgmaxT, hgf⟩ :=
    Finset.mem_image.mp (Finset.max'_mem (T.image f) (hne.image f))
  have hmax : ∀ b ∈ T, f b ≤ f gmax := by
    intro b hb
    rw [hgf]
    exact Finset.le_max' _ _ (Finset.mem_image.mpr ⟨b, hb, rfl⟩)
  obtain ⟨hnl, hstep⟩ := hclosed gmax hgmaxT
  obtain ⟨p, hp, hpf⟩ := hmono gmax hgmaxT hnl
  obtain ⟨pl, hpl, hsucc⟩ := hstep p hp
  have hlt : f gmax < f (adversarialStep cfg gmax p pl) := hpf pl hpl hsucc
  have hle : f (adversarialStep cfg gmax p pl) ≤ f gmax := hmax _ hsucc
  omega

/-- **Verdict-level form: a GLOBAL adversarial monovariant refutes solvability.** If some
`f : GameState -> Nat` rises under a forced adversary draw from *every* non-lost state
(unconditionally on any set), then no closed `init`-set exists, so `TetrisSolvableValid`
is FALSE. This is the exact shape a FALSE proof of the existence verdict must take: not a
per-bag rate bound (those are cross-bag-defeated) but a single state-local function the
adversary can always push up no matter how the player responds — including by clearing.
That last "including by clearing" is precisely why no such `f` has been (or, the evidence
suggests, can be) instantiated here under the loop's axiom constraints. -/
theorem not_tetrisSolvableValid_of_global_adversarial_monovariant
    (f : GameState → ℕ)
    (hmono : ∀ g : GameState, ¬ g.lost GameConfig.standard →
      ∃ p ∈ g.bag, ∀ pl ∈ Placement.allValidFor GameConfig.standard p,
        f g < f (adversarialStep GameConfig.standard g p pl)) :
    ¬ TetrisSolvableValid := by
  intro h
  obtain ⟨T, hinitT, hclosed⟩ := exists_closed_atlas_iff_tetrisSolvableValid.mpr h
  refine not_isClosedSet_of_adversarial_monovariant GameConfig.standard T
    ⟨_, hinitT⟩ f ?_ hclosed
  intro g hgT hnl
  obtain ⟨p, hp, hpf⟩ := hmono g hnl
  exact ⟨p, hp, fun pl hpl _ => hpf pl hpl⟩

/-- **The hole-rate FALSE route, formally closed.** Specialising the global monovariant
principle to `f g = holes g.board` pins the EXACT hypothesis a holes-based FALSE proof
would require: from every non-lost state the adversary owns a draw whose *every* valid
landing strictly raises the hole count. Everything the file has built on the creation
side (`holes_lt_holes_place_of_buried`: a buried hard drop adds a hole;
`holes_le_holes_foldl_place`: holes climb monotonically over a clear-free run) establishes
the `<` only for placements that do NOT clear. The hypothesis below additionally demands
it for the landings that DO clear — and `holes_clearLines_le` shows a clear can only
*lower* holes (`colHoles` does not even persist across clears, iter-119). So the player
escapes the monovariant by completing a line, and this hypothesis is exactly the
empirically-false obligation. This is the formal capstone of the hole-rate route: it is
not a fresh obstruction but the precise, named reason that route cannot flip the verdict —
the same cheap-removal blocker that closed the count- and row-production routes, now a
single quantified clause rather than scattered rate dead-ends. INFRA, not a flip. -/
theorem not_tetrisSolvableValid_of_holes_adversarial_monovariant
    (hmono : ∀ g : GameState, ¬ g.lost GameConfig.standard →
      ∃ p ∈ g.bag, ∀ pl ∈ Placement.allValidFor GameConfig.standard p,
        holes GameConfig.standard g.board
          < holes GameConfig.standard
              (adversarialStep GameConfig.standard g p pl).board) :
    ¬ TetrisSolvableValid :=
  not_tetrisSolvableValid_of_global_adversarial_monovariant
    (fun g => holes GameConfig.standard g.board) hmono

/-- **The combined-volume (Φ = count + holes) FALSE route, pinned to the no-clear forcing
question.** Specialising the global monovariant principle to `Φ g = count g.board + holes g.board`
reduces the entire FALSE verdict to one combinatorial obligation: *from every non-lost state the
adversary can name a bag piece whose every valid landing clears no line* (`fullRows (place) = ∅`).
Under that hypothesis the Φ-ratchet `count_add_holes_add_four_le_adversarialStep_of_no_fullRows`
fires on each forced move — a non-clearing hard drop lands four fresh cells and may bury but never
sheds, so `Φ` climbs by `≥ 4` every step — making `Φ` a genuine adversarial monovariant and
`init ∉ safe` by `not_isClosedSet_of_adversarial_monovariant`.

This is strictly the better packaging of the hole route than
`not_tetrisSolvableValid_of_holes_adversarial_monovariant`: that demands holes rise even on
*clearing* landings (empirically false — the player clears to dodge, `holes_clearLines_le`); this
asks only that the adversary can avoid *offering* a clearable piece, shifting the burden from a
quantity the player controls (holes) to a draw the adversary controls (which piece). The remaining
open crux is exactly this forcing hypothesis — the S/Z-flood mechanism the empirical GFP exhibits,
where a 1-wide well leaves every S/Z landing non-clearing — now a single named clause rather than a
rate dead-end. INFRA: names the precise gap on the FALSE side; does not by itself flip the verdict. -/
theorem not_tetrisSolvableValid_of_adversary_forces_no_clear
    (hforce : ∀ g : GameState, ¬ g.lost GameConfig.standard →
      ∃ p ∈ g.bag, ∀ pl ∈ Placement.allValidFor GameConfig.standard p,
        Board.fullRows GameConfig.standard
          (({pl with piece := p} : Placement).place g.board) = ∅) :
    ¬ TetrisSolvableValid := by
  apply not_tetrisSolvableValid_of_global_adversarial_monovariant
    (fun g => g.board.count + holes GameConfig.standard g.board)
  intro g hnl
  obtain ⟨p, hp, hno⟩ := hforce g hnl
  refine ⟨p, hp, fun pl hpl => ?_⟩
  have hratchet :=
    count_add_holes_add_four_le_adversarialStep_of_no_fullRows g p pl (hno pl hpl)
  show g.board.count + holes GameConfig.standard g.board
      < (adversarialStep GameConfig.standard g p pl).board.count
        + holes GameConfig.standard (adversarialStep GameConfig.standard g p pl).board
  omega

/-! ### Empty columns block every clear (a well-geometry brick toward `hforce`)

`not_tetrisSolvableValid_of_adversary_forces_no_clear` (above) reduced the FALSE verdict to
`hforce`: from every non-lost state the adversary owns a bag piece whose *every* valid landing
clears no line. The empirical mechanism is a 1-wide well that the offered S/Z cannot fill — a
column that stays empty makes every row that needs it un-completable. These lemmas formalise the
column-empty half of that mechanism: a board column with no cells admits no full row, and a
placement that misses such a column leaves it empty, hence clears nothing. They feed the
Φ-ratchet's `fullRows (place) = ∅` hypothesis directly. PARTIAL: this covers landings that avoid
the well; the remaining `hforce` obligation — that the well-*filling* S/Z landings also fail to
complete a row — is the deferred S/Z surface-geometry case. -/

/-- A column with zero cells (`colCount b j = 0`) contains no cell at any row: `(j, r) ∉ b`. The
membership form of column-emptiness, the bridge into `notMem_fullRows_of_notMem`. -/
theorem notMem_of_colCount_zero {b : Board} {j r : ℕ}
    (h : b.colCount j = 0) : (j, r) ∉ b := by
  intro hmem
  have hpos : 0 < b.colCount j := by
    unfold Board.colCount
    exact Finset.card_pos.mpr ⟨(j, r), Finset.mem_filter.mpr ⟨hmem, rfl⟩⟩
  omega

/-- **An empty in-field column blocks every clear.** If column `j < cfg.cols` holds no cells then
no row is full (each would need a cell in column `j`), so `fullRows = ∅`: the board-level core of
the well-blocks-clears mechanism behind `hforce`. -/
theorem fullRows_eq_empty_of_colCount_zero {cfg : GameConfig} {b : Board} {j : ℕ}
    (hj : j < cfg.cols) (h : b.colCount j = 0) :
    Board.fullRows cfg b = ∅ := by
  apply Finset.eq_empty_of_forall_notMem
  intro r hr
  exact notMem_fullRows_of_notMem hj (notMem_of_colCount_zero h) hr

/-- **A landing that misses an empty column clears nothing.** If column `j < cfg.cols` is empty in
`b` and the placement contributes no cell to it (`colProfile = 0`), then column `j` is still empty
after the hard drop (`colCount_place`), so `fullRows (pl.place b) = ∅` — the placement clears no
line. This is exactly the `fullRows (place) = ∅` premise the Φ-ratchet
(`count_add_holes_add_four_le_adversarialStep_of_no_fullRows`) consumes, supplied for every
well-avoiding landing. -/
theorem fullRows_place_eq_empty_of_colCount_zero_of_colProfile_zero
    {cfg : GameConfig} {b : Board} {pl : Placement} {j : ℕ}
    (hj : j < cfg.cols) (hb : b.colCount j = 0) (hmiss : pl.colProfile j = 0) :
    Board.fullRows cfg (pl.place b) = ∅ := by
  apply fullRows_eq_empty_of_colCount_zero hj
  rw [Placement.colCount_place, hb, hmiss]

/-- **A finite-depth well pins every cleared row above the well floor.** If column
`j < cfg.cols` is empty at every row below `h` (the real adversary structure — a column
empty *below* a height, not necessarily empty everywhere), then no full row can sit below
`h`: a full row needs `(j, r) ∈ b`, but every such `r < h` is missing. Hence all full rows
lie at height `≥ h`. -/
theorem le_of_mem_fullRows_of_column_empty_below {cfg : GameConfig} {b : Board} {j h : ℕ}
    (hj : j < cfg.cols) (hempty : ∀ r, r < h → (j, r) ∉ b) :
    ∀ r ∈ Board.fullRows cfg b, h ≤ r := by
  intro r hr
  by_contra hlt
  push_neg at hlt
  exact notMem_fullRows_of_notMem hj (hempty r hlt) hr

/-- **A finite-depth well caps per-step clears by its exposed band.** Combining the well
floor (`le_of_mem_fullRows_of_column_empty_below`) with the surface ceiling
(`lt_maxHeight_of_mem_fullRows`), the full rows live in `Ico h (maxHeight b)`, so at most
`maxHeight b - h` rows can clear in one move. This is the **quantitative** "holes/well throttle
clears" brick: a depth-`h` well shrinks the clearable band from `maxHeight` down to
`maxHeight - h`, driving the clear rate below the survival requirement as the well deepens. -/
theorem fullRows_card_le_maxHeight_sub_of_column_empty_below {cfg : GameConfig} {b : Board}
    {j h : ℕ} (hcols : 0 < cfg.cols) (hj : j < cfg.cols)
    (hempty : ∀ r, r < h → (j, r) ∉ b) :
    (Board.fullRows cfg b).card ≤ maxHeight cfg b - h := by
  have hsub : Board.fullRows cfg b ⊆ Finset.Ico h (maxHeight cfg b) := by
    intro r hr
    rw [Finset.mem_Ico]
    exact ⟨le_of_mem_fullRows_of_column_empty_below hj hempty r hr,
           lt_maxHeight_of_mem_fullRows hcols hr⟩
  calc (Board.fullRows cfg b).card
      ≤ (Finset.Ico h (maxHeight cfg b)).card := Finset.card_le_card hsub
    _ = maxHeight cfg b - h := Nat.card_Ico h (maxHeight cfg b)

/-- **Every in-field column heights-bounds the clears.** A full row `r` needs a cell `(j, r)`
in *every* in-field column `j < cfg.cols`, so `r < colHeight j` for each such `j`. The
per-column refinement of `lt_maxHeight_of_mem_fullRows` (which only exploited column 0). -/
theorem lt_colHeight_of_mem_fullRows {cfg : GameConfig} {b : Board} {j r : ℕ}
    (hj : j < cfg.cols) (hr : r ∈ Board.fullRows cfg b) : r < b.colHeight j := by
  unfold Board.fullRows at hr
  rw [Finset.mem_filter] at hr
  have hfull := hr.2
  unfold Board.isFull at hfull
  have hmem : (j, r) ∈ b := hfull j (Finset.mem_range.mpr hj)
  exact Board.lt_colHeight hmem

/-- The full rows live inside `range (colHeight j)` for every in-field column `j`. -/
theorem fullRows_subset_range_colHeight {cfg : GameConfig} {b : Board} {j : ℕ}
    (hj : j < cfg.cols) : Board.fullRows cfg b ⊆ Finset.range (b.colHeight j) := by
  intro r hr
  exact Finset.mem_range.mpr (lt_colHeight_of_mem_fullRows hj hr)

/-- **Clears ≤ the well height.** At most `colHeight j` rows are full at once, for *every*
column `j < cfg.cols` — so a single move clears at most the height of the *shortest* column. A
1-wide well of height `h` caps per-step clears at `h`: an empty well (`colHeight = 0`) clears
nothing, a shallow well throttles clears to its depth. The sharp per-column form of the
well-throttle (strictly below `fullRows_card_le_maxHeight`, which uses the *tallest* column),
and the cleanest quantitative bridge into the `survival_forces_clears` clear-rate requirement. -/
theorem fullRows_card_le_colHeight {cfg : GameConfig} {b : Board} {j : ℕ}
    (hj : j < cfg.cols) : (Board.fullRows cfg b).card ≤ b.colHeight j := by
  calc (Board.fullRows cfg b).card
      ≤ (Finset.range (b.colHeight j)).card :=
        Finset.card_le_card (fullRows_subset_range_colHeight hj)
    _ = b.colHeight j := Finset.card_range _

/-- **Clears ≤ the well's cell count.** Each full row `r` contributes a *distinct* cell `(j, r)`
to column `j` (a full row fills every column), so the full rows inject into column `j`'s cells:
`#fullRows ≤ colCount j` for every in-field column. Strictly sharper than
`fullRows_card_le_colHeight` (since `colCount j ≤ colHeight j`) and than the aggregate
`cols_mul_fullRows_card_le_count` (`#fullRows ≤ count / cols`) in the well regime: a 1-wide well
holding few cells throttles per-step clears to exactly that cell count, regardless of empty space
below its surface — and `colCount` plugs straight into the `count = ∑ colCount` conservation law. -/
theorem fullRows_card_le_colCount {cfg : GameConfig} {b : Board} {j : ℕ}
    (hj : j < cfg.cols) : (Board.fullRows cfg b).card ≤ b.colCount j := by
  have hmap : ∀ r ∈ Board.fullRows cfg b, (j, r) ∈ b.filter (fun p => p.1 = j) := by
    intro r hr
    rw [Finset.mem_filter]
    refine ⟨?_, rfl⟩
    unfold Board.fullRows at hr
    rw [Finset.mem_filter] at hr
    have hfull := hr.2
    unfold Board.isFull at hfull
    exact hfull j (Finset.mem_range.mpr hj)
  have hinj : Set.InjOn (fun r => (j, r)) (Board.fullRows cfg b : Set ℕ) := by
    intro r1 _ r2 _ heq
    simpa using heq
  unfold Board.colCount
  exact Finset.card_le_card_of_injOn _ hmap hinj

/-- **A well-missing landing clears no more than the well already holds.** If the placement
contributes nothing to column `j` (`colProfile j = 0`), then by `colCount_place` the placed
board's column `j` still holds exactly `b.colCount j` cells, so the post-placement clears are
capped by the well's *pre-placement* cell count. This lifts the iter-185
`colCount = 0 ⟹ no clear` blocker from an *empty* well to a *shallow* one: a depth-`d` 1-wide
well caps any well-missing piece's clears at `d`, the quantitative throttle hforce needs on the
well-avoiding S/Z landings. -/
theorem fullRows_place_card_le_colCount_of_colProfile_zero {cfg : GameConfig} {b : Board}
    {pl : Placement} {j : ℕ} (hj : j < cfg.cols) (hmiss : pl.colProfile j = 0) :
    (Board.fullRows cfg (pl.place b)).card ≤ b.colCount j := by
  have h := fullRows_card_le_colCount (cfg := cfg) (b := pl.place b) (j := j) hj
  rw [Placement.colCount_place, hmiss, Nat.add_zero] at h
  exact h

/-- **A persisting empty well turns every move into a pure `+4` deposit (the count-climb engine).**
This is the first *with-clear* foldl ledger: every prior fold (`count_foldl_place`, the `rowCount`
and `holes` folds) was clear-free by *fiat* (`pl.place`); here the moves are genuine
`Placement.applyStep` (clear-capable), yet under the hypothesis that a single in-field column
`j` starts empty (`b.colCount j = 0`) and *every* placement in the list misses it
(`colProfile j = 0`), the column stays empty forever (`colCount_place`), so by
`fullRows_place_eq_empty_of_colCount_zero_of_colProfile_zero` no move clears a line and
`applyStep` collapses to `place` (`clearLines_eq_self_of_no_fullRows`). Hence the total count
climbs by exactly `4` per move with *zero* clears — `count = b.count + 4 * length`.

This is loss-RELEVANT but INFRA, not a flip: it is the count-climb engine that drives a
persisting empty well toward the `count + holes ≤ cols*rows` overflow (a 10×20 board holds only
200 cells, so an empty well can persist at most `(200 - count)/4` moves before the board would
overflow — see the finite-persistence corollary). But the hypothesis is *player-violable*: the
player can fill the well at will (a well-filling piece has `colProfile j > 0`), enabling a clear
and escaping the ratchet. Forcing the well to persist would require the adversary to *withhold*
the well-filling piece, which the 7-bag forbids (every piece is offered once per bag). So this
brick constrains the cooperative-player scenario but does not by itself decide the verdict. -/
theorem count_foldl_applyStep_eq_of_forall_colProfile_zero {cfg : GameConfig}
    {j : ℕ} (hj : j < cfg.cols) (b : Board) (pls : List Placement)
    (hb : b.colCount j = 0)
    (hmiss : ∀ pl ∈ pls, pl.colProfile j = 0) :
    (pls.foldl (fun acc pl => pl.applyStep cfg acc) b).count = b.count + 4 * pls.length := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    simp only [List.foldl_cons, List.length_cons]
    have hpl : pl.colProfile j = 0 := hmiss pl List.mem_cons_self
    have hnoclear : Board.fullRows cfg (pl.place b) = ∅ :=
      fullRows_place_eq_empty_of_colCount_zero_of_colProfile_zero hj hb hpl
    have happly : pl.applyStep cfg b = pl.place b := by
      rw [Placement.applyStep_eq_clearLines_place,
          Board.clearLines_eq_self_of_no_fullRows cfg hnoclear]
    have hplace0 : (pl.place b).colCount j = 0 := by
      rw [Placement.colCount_place, hb, hpl]
    rw [happly, ih (pl.place b) hplace0
        (fun pl' hpl' => hmiss pl' (List.mem_cons_of_mem pl hpl')), Placement.count_place]
    omega

/-- **With-clear count upper envelope over a placement list (the general-position bracket).**
The clears-allowed companion to `count_foldl_applyStep_eq_of_forall_colProfile_zero`: over *any*
list of genuine `applyStep` moves (clears permitted, no hypotheses at all), the count is at most
the clear-free ceiling `b.count + 4 * length`. Folds the hypothesis-free per-move bound
`count_applyStep_le` (`(applyStep).count ≤ count + 4`, since clearing only removes cells). Together
with the iter-189 equality (which pins count *to* the ceiling exactly when no line ever clears)
this **brackets** any bag's count: the deficit `b.count + 4*length - (foldl applyStep b).count`
equals `cols` times the total lines cleared over the run — the amortized ledger's measuring stick.

HONEST STATUS of the amortized-net route this bracket serves (honesty clause, route assessment --
NOT halting the loop): the FALSE verdict via "count nets UP per 7-bag" needs total bag-clears
`< 2.8` lines (so `28 - 10k > 0`). But the 7-bag GUARANTEES the player one I per bag, and a single
I dropped into a maintained 4-deep 1-wide well clears a **Tetris (4 lines)** in ONE move -- so if
the player can keep a Tetris-ready well, clears run at `>= 4/bag = 0.57/piece > 0.4` and count nets
DOWN `28 - 40 = -12` cells/bag: the player SURVIVES. Hence the amortized-net-UP route is BLOCKED
exactly when the player can maintain a Tetris well -- the same guaranteed-I wall that sank literal
hforce (iter-188). The FALSE verdict therefore hinges on the un-formalized combinatorial core: the
adversary's S/Z-flood + no-HOLD draw-timing PREVENTS a maintained Tetris well. The throttle ladder
(iters 185-189) bounds clears *by* well depth but cannot bound well depth *from above* against an
optimizing player. Verdict STILL OPEN as a base-axiom artifact, FALSE-leaning empirically. -/
theorem count_foldl_applyStep_le (cfg : GameConfig) (b : Board) (pls : List Placement) :
    (pls.foldl (fun acc pl => pl.applyStep cfg acc) b).count ≤ b.count + 4 * pls.length := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    simp only [List.foldl_cons, List.length_cons]
    have h1 := count_applyStep_le cfg b pl
    have h2 := ih (pl.applyStep cfg b)
    omega

end FiveBagReset
end Experiments
end Tetris
