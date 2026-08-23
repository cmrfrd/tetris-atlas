import Mathlib
import Proofs.Invariants.StateSpace
import Proofs.Safety.HorizonCompactness

/-!
# The Atlas may be taken finite

`solvable_iff_exists_invariant` reduced existence to one closed invariant — a
*set* of states, possibly infinite. This file sharpens the witness one more
notch: **the invariant may always be taken finite**, and with an explicit size
bound.

The forward direction runs any solving solver and collects its reachable cone.
A `SolvesTetrisValid` solver keeps every reachable board well-formed and
in-field, and in-field boards form a finite type (`InFieldBoard`, `2^200`
members at standard size) while bags number `128` — so the cone embeds in a
finite type and is finite (`solverReachable_finite`). The cone contains `init`,
avoids loss, and is closed under the solver's own replies: a finite closed
invariant (`solvable_iff_exists_finite_invariant`).

Counting the embedding (`solvable_implies_bounded_atlas`): the invariant can be
taken of size at most `2^200 · 128 = 2^207`. So the M4 artifact is not just
set-shaped but **finite-table-shaped, with an a-priori size bound**, whenever it
exists at all — "build the Atlas" and "prove Tetris solvable" are the same task
even at the level of witness cardinality.
-/

namespace Tetris

/-- Every state in a valid solving solver's reachable cone embeds into
`InFieldBoard × Bag`. -/
theorem solverReachable_subset_image {cfg : GameConfig} {σ : Solver cfg}
    (hσ : SolvesTetrisValid cfg σ) :
    {g : GameState | solverReachable σ g}
      ⊆ (fun q : InFieldBoard cfg × Bag => GameState.mk q.1.val q.2) ''
          Set.univ := by
  intro g hg
  have hwf := solverReachable_wf_of_solvesTetrisValid hσ hg
  have hnl := solverReachable_not_lost_of_solvesTetrisValid hσ hg
  have hif : ∀ p ∈ g.board, p.2 < cfg.rows :=
    (GameState.not_lost_iff_forall_row_lt cfg g).mp hnl
  exact ⟨(⟨g.board, hwf, hif⟩, g.bag), Set.mem_univ _, by cases g; rfl⟩

/-- **The reachable cone of a valid solving solver is finite.** Its boards are
well-formed and in-field — a finite type — and its bags are finite. -/
theorem solverReachable_finite {cfg : GameConfig} {σ : Solver cfg}
    (hσ : SolvesTetrisValid cfg σ) :
    {g : GameState | solverReachable σ g}.Finite :=
  (Set.finite_univ.image _).subset (solverReachable_subset_image hσ)

/-- **The Atlas may be taken finite.** Solvability is equivalent to the
existence of a *finite* closed invariant: the set-shaped witness of
`solvable_iff_exists_invariant` compresses to a `Finset` with no loss of
strength. -/
theorem solvable_iff_exists_finite_invariant :
    TetrisSolvableValid ↔
      ∃ S : Finset GameState,
        GameState.init ∈ S ∧
        (∀ g ∈ S, ¬ g.lost GameConfig.standard) ∧
        ∀ g ∈ S, ∀ p, p ∈ g.bag →
          ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
            adversarialStep GameConfig.standard g p pl ∈ S := by
  constructor
  · rintro ⟨σ, hσ⟩
    have hfin := solverReachable_finite hσ
    refine ⟨hfin.toFinset, ?_, ?_, ?_⟩
    · rw [Set.Finite.mem_toFinset]
      exact solverReachable.init
    · intro g hg
      rw [Set.Finite.mem_toFinset] at hg
      exact solverReachable_not_lost_of_solvesTetrisValid hσ hg
    · intro g hg p hp
      rw [Set.Finite.mem_toFinset] at hg
      obtain ⟨hpiece, hvalid⟩ := hσ.1 g p hp
      refine ⟨σ g p, hpiece, hvalid, ?_⟩
      rw [Set.Finite.mem_toFinset]
      exact solverReachable.step p hg hp
  · rintro ⟨S, hinit, hnl, hstep⟩
    refine tetrisSolvableValid_of_invariant (↑S : Set GameState)
      (Finset.mem_coe.mpr hinit)
      (fun g hg => hnl g (Finset.mem_coe.mp hg))
      (fun g hg p hp => ?_)
    obtain ⟨pl, h1, h2, h3⟩ := hstep g (Finset.mem_coe.mp hg) p hp
    exact ⟨pl, h1, h2, Finset.mem_coe.mpr h3⟩

set_option maxRecDepth 8000 in
/-- **An a-priori size bound on the Atlas.** If Tetris is solvable at all, it
is solvable by a closed table of at most `2^207` entries — `2^200` in-field
boards times `128` bags. The bound is astronomical but *finite and explicit*:
the existence problem and the table-construction problem have the same witness
up to cardinality. -/
theorem solvable_implies_bounded_atlas (h : TetrisSolvableValid) :
    ∃ S : Finset GameState,
      S.card ≤ 2 ^ 207 ∧
      GameState.init ∈ S ∧
      (∀ g ∈ S, ¬ g.lost GameConfig.standard) ∧
      ∀ g ∈ S, ∀ p, p ∈ g.bag →
        ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
          adversarialStep GameConfig.standard g p pl ∈ S := by
  obtain ⟨σ, hσ⟩ := h
  have hfin := solverReachable_finite hσ
  have hget : ∀ g : GameState, g ∈ hfin.toFinset → solverReachable σ g := by
    intro g hg
    rwa [Set.Finite.mem_toFinset] at hg
  have hcard : hfin.toFinset.card ≤ 2 ^ 207 := by
    let f : ↥hfin.toFinset → InFieldBoard GameConfig.standard × Bag := fun g =>
      (⟨g.val.board,
        solverReachable_wf_of_solvesTetrisValid hσ (hget g.val g.prop),
        (GameState.not_lost_iff_forall_row_lt _ _).mp
          (solverReachable_not_lost_of_solvesTetrisValid hσ (hget g.val g.prop))⟩,
       g.val.bag)
    have hinj : Function.Injective f := by
      rintro ⟨⟨ab, abag⟩, ha⟩ ⟨⟨bb, bbag⟩, hb⟩ hab
      have hboard : ab = bb := congrArg (fun q => (Prod.fst q).val) hab
      have hbag : abag = bbag := congrArg Prod.snd hab
      subst hboard
      subst hbag
      rfl
    have hle := Fintype.card_le_of_injective f hinj
    rw [Fintype.card_coe, Fintype.card_prod, InFieldBoard.standard_fintype_card,
      Bag.fintype_card] at hle
    calc hfin.toFinset.card ≤ 2 ^ 200 * 128 := hle
      _ = 2 ^ 207 := by norm_num [← pow_add]
  refine ⟨hfin.toFinset, hcard, ?_, ?_, ?_⟩
  · rw [Set.Finite.mem_toFinset]
    exact solverReachable.init
  · intro g hg
    rw [Set.Finite.mem_toFinset] at hg
    exact solverReachable_not_lost_of_solvesTetrisValid hσ hg
  · intro g hg p hp
    rw [Set.Finite.mem_toFinset] at hg
    obtain ⟨hpiece, hvalid⟩ := hσ.1 g p hp
    refine ⟨σ g p, hpiece, hvalid, ?_⟩
    rw [Set.Finite.mem_toFinset]
    exact solverReachable.step p hg hp

end Tetris
