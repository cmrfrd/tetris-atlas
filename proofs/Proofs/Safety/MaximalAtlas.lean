import Mathlib
import Proofs.Safety.SafeSet

/-!
# Is the global solver unique?

Suppose we write down the "global solver" as a lookup table
`(board, bag, piece) ↦ placements`, ask that it be closed (every prescribed move
lands back inside the table), and ask that it contain *every* placement that
could belong to such a table. Is that object unique, and is every other solver a
part of it?

The answer splits cleanly in two, and the split is the whole content:

* **As a relation, yes — uniquely so.** The states such a table can cover are
  exactly `safe cfg`, the greatest fixed point of `safeOp`
  (`safe_isGreatest`, `safe_unique`). Every closed atlas lives inside it
  (`isClosedOn_subset_safe`), and every move any closed atlas prescribes is one
  of the moves the maximal table already lists
  (`atlas_choice_mem_safeMoves`). There is exactly one maximal table:
  **`safeMoves`**, which at each `(g, p)` lists *all* placements answering `p`
  that land back in `safe`.

* **As a function, no — wildly non-unique.** A *solver* must pick one placement
  per `(state, piece)`. Any pick from `safeMoves` works
  (`selection_survives`), and wherever `safeMoves` offers two options both picks
  are valid (`selection_update_mem`). So solvers are **selections** of the one
  maximal relation, not subsets of one canonical function, and there are
  generally exponentially many of them.

So "the Atlas" is well-defined and unique — provided you read it as the maximal
*relation* (state → piece → set of safe answers). The moment you collapse it to
a single prescribed move per situation you are choosing among many equally
correct Atlases, and nothing in the theory prefers one.

A practical consequence: the object worth computing is `safeMoves`, not a
particular solver. Storing all safe answers keeps every downstream tie-break
available (shortest cycle, lowest occupancy, most robust to piece order), and
costs nothing in correctness.
-/

namespace Tetris
namespace MaximalAtlas

/-! ## The safe set is the unique maximal closed set -/

/-- **`safe` is the greatest closed set of states.** It is closed under the
safe-move operator, and every closed set lies inside it. -/
theorem safe_isGreatest (cfg : GameConfig) :
    IsGreatest {S : Set GameState | S ⊆ safeOp cfg S} (safe cfg) :=
  ⟨le_of_eq (safe_eq cfg).symm, fun S hS => safe_greatest S hS⟩

/-- **And it is the only one.** Any fixed point of the safe-move operator that
also dominates every closed set *is* `safe`. Maximality pins the table
uniquely: there is no second, different "largest closed solver". -/
theorem safe_unique {cfg : GameConfig} {S : Set GameState}
    (hfix : safeOp cfg S = S)
    (hmax : ∀ T : Set GameState, T ⊆ safeOp cfg T → T ⊆ S) :
    S = safe cfg :=
  Set.Subset.antisymm
    (safe_greatest S (le_of_eq hfix.symm))
    (hmax (safe cfg) (le_of_eq (safe_eq cfg).symm))

/-- **Every closed atlas lives inside `safe`.** Whatever set of states your
lookup table certifies, those states were already safe. -/
theorem isClosedOn_subset_safe {cfg : GameConfig} {A : Atlas cfg}
    {S : Finset GameState} (h : A.IsClosedOn cfg S) :
    (↑S : Set GameState) ⊆ safe cfg := by
  refine safe_greatest _ ?_
  intro g hg
  have hgS : g ∈ S := Finset.mem_coe.mp hg
  refine ⟨h.not_lost g hgS, ?_⟩
  intro p hp
  obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp (h.total g hgS p hp)
  obtain ⟨hpiece, hvalid⟩ := h.valid g hgS p hp pl hpl
  exact ⟨pl, hpiece, hvalid, Finset.mem_coe.mpr (h.closed g hgS p hp pl hpl)⟩

/-! ## The maximal table -/

/-- **The maximal atlas.** At state `g` facing piece `p`, all placements that
answer `p`, are in bounds, and land back in the safe set. This is the unique
largest lookup table: no closed table can offer a move outside it, and it offers
every move any closed table could. -/
def safeMoves (cfg : GameConfig) (g : GameState) (p : Piece) : Set Placement :=
  { pl | pl.piece = p ∧ pl.Valid cfg ∧ adversarialStep cfg g p pl ∈ safe cfg }

/-- The maximal table is never empty where it matters: a safe state has an
answer to every piece its bag can deal. -/
theorem safeMoves_nonempty {cfg : GameConfig} {g : GameState} (hg : g ∈ safe cfg)
    {p : Piece} (hp : p ∈ g.bag) :
    (safeMoves cfg g p).Nonempty := by
  obtain ⟨pl, hpiece, hvalid, hstep⟩ := safe_forall_step hg p hp
  exact ⟨pl, hpiece, hvalid, hstep⟩

/-- **Every closed atlas is pointwise contained in the maximal one.** Any move
your table prescribes at a state it certifies is already listed by `safeMoves`.
This is the precise sense in which all solvers are "part of" one global
solver. -/
theorem atlas_choice_mem_safeMoves {cfg : GameConfig} {A : Atlas cfg}
    {S : Finset GameState} (h : A.IsClosedOn cfg S) {g : GameState} (hg : g ∈ S)
    {p : Piece} (hp : p ∈ g.bag) {pl : Placement} (hA : A g p = some pl) :
    pl ∈ safeMoves cfg g p := by
  obtain ⟨hpiece, hvalid⟩ := h.valid g hg p hp pl hA
  refine ⟨hpiece, hvalid, ?_⟩
  exact isClosedOn_subset_safe h (Finset.mem_coe.mpr (h.closed g hg p hp pl hA))

/-! ## Solvers are selections, and there are many -/

/-- **Any selection from the maximal table keeps you safe.** A solver that always
picks *some* listed move never leaves `safe`. Nothing distinguishes one pick
from another. -/
theorem selection_survives {cfg : GameConfig} {σ : Solver cfg}
    (hσ : ∀ g ∈ safe cfg, ∀ p ∈ g.bag, σ g p ∈ safeMoves cfg g p)
    {g : GameState} (hg : g ∈ safe cfg) {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (σ g p) ∈ safe cfg :=
  (hσ g hg p hp).2.2

/-- **Solvers are not unique.** Wherever the maximal table lists an alternative,
overwriting a selection at that one point yields another equally valid
selection. So a `Solver` is a *choice function* over `safeMoves`, and any place
the table offers two moves doubles the number of correct solvers. -/
theorem selection_update_mem {cfg : GameConfig} {σ : Solver cfg}
    (hσ : ∀ g ∈ safe cfg, ∀ p ∈ g.bag, σ g p ∈ safeMoves cfg g p)
    {g₀ : GameState} {p₀ : Piece} {pl' : Placement}
    (hpl' : pl' ∈ safeMoves cfg g₀ p₀) :
    ∀ g ∈ safe cfg, ∀ p ∈ g.bag,
      (fun (g : GameState) (p : Piece) =>
        if g = g₀ ∧ p = p₀ then pl' else σ g p) g p ∈ safeMoves cfg g p := by
  intro g hg p hp
  by_cases hcase : g = g₀ ∧ p = p₀
  · simp only [if_pos hcase]
    obtain ⟨hg0, hp0⟩ := hcase
    subst hg0
    subst hp0
    exact hpl'
  · simp only [if_neg hcase]
    exact hσ g hg p hp

/-- Every listed move is in the finite enumeration `allValidFor`: the maximal
table's fibers live inside a computable finite set. -/
theorem safeMoves_subset_allValidFor (cfg : GameConfig) (g : GameState)
    (p : Piece) :
    safeMoves cfg g p ⊆ ↑(Placement.allValidFor cfg p) := by
  rintro pl ⟨hpiece, hvalid, -⟩
  exact Finset.mem_coe.mpr ((Placement.mem_allValidFor cfg p pl).mpr ⟨hpiece, hvalid⟩)

/-- **The maximal table has finite fibers.** At each state and piece the set of
safe answers is finite — the Atlas-as-relation is a finitely-branching object
even before any state-space truncation. -/
theorem safeMoves_finite (cfg : GameConfig) (g : GameState) (p : Piece) :
    (safeMoves cfg g p).Finite :=
  Set.Finite.subset (Placement.allValidFor cfg p).finite_toSet
    (safeMoves_subset_allValidFor cfg g p)

end MaximalAtlas
end Tetris
