import Mathlib
import Proofs.Survival.Lasso

/-!
# The cooperative lasso witness (T1)

**The first concrete infinite-play certificate in this repository.** An
explicit 35-placement loop — 5 bags, each drawing all 7 pieces in a chosen
order — that takes `GameState.init` (empty board, full bag) back to
`GameState.init` exactly, hole-free, never lost. Certified from scratch by
`checkTable` (`Proofs/Survival/Lasso.lean`) via `native_decide`, then
converted into a `ClosedCycle` and the headline theorem

  `cooperative_tetris_survivable :
     ∃ π, SurvivesForever GameConfig.standard π GameState.init`

This is the *cooperative* game: the policy chooses both the draw order within
each bag and the placement (`Policy`/`trace` semantics — the piece played is
`(π g).piece`, constrained to the bag by `legal_draw`). It is **not**
`TetrisSolvable` (which quantifies over all announcement orders); it is the
M2-shape witness that de-vacuizes `ClosedCycle`, `closed_cycle_survives`, and
`SurvivesForever` on the canonical 10×20 config.

Loop arithmetic: 35 pieces = 5 bags place 140 cells and clear exactly 14
lines (`4·35 = 10·14`), the minimum possible loop length (cell conservation
forces `L ≡ 0 mod 5`, bag renewal forces `L ≡ 0 mod 7`).

The placement list was found by `scripts/find_cooperative_lasso.py` (a
one-off hole-free skyline DFS mirroring this model); the script carries no
trust — everything is re-checked here by the kernel-validated `native_decide`
evaluation of `checkTable`.
-/

namespace Tetris

open Piece

/-- The 35-placement loop: `init → init` over 5 bags. Found by
`scripts/find_cooperative_lasso.py`'s with-holes bitboard beam (transient
holes are essential — a hole-free/flush 5-bag perfect clear does not exist);
re-certified here from scratch by `checkTable` + `native_decide`. Each block of
7 is a bag permutation; the board returns to empty and the bag to full at
placement 35. -/
def lassoPlacements : List Placement :=
  [ ⟨.T, 2, 3⟩, ⟨.L, 0, 7⟩, ⟨.S, 1, 5⟩, ⟨.Z, 0, 6⟩, ⟨.I, 0, 0⟩, ⟨.O, 0, 8⟩, ⟨.J, 0, 0⟩,
    ⟨.J, 0, 2⟩, ⟨.S, 1, 0⟩, ⟨.I, 0, 4⟩, ⟨.T, 3, 3⟩, ⟨.O, 0, 1⟩, ⟨.L, 0, 7⟩, ⟨.Z, 0, 4⟩,
    ⟨.T, 2, 6⟩, ⟨.L, 2, 0⟩, ⟨.Z, 0, 7⟩, ⟨.O, 0, 5⟩, ⟨.J, 3, 3⟩, ⟨.I, 0, 0⟩, ⟨.S, 0, 2⟩,
    ⟨.J, 2, 7⟩, ⟨.O, 0, 5⟩, ⟨.S, 0, 0⟩, ⟨.L, 0, 7⟩, ⟨.T, 3, 0⟩, ⟨.I, 0, 4⟩, ⟨.Z, 0, 1⟩,
    ⟨.Z, 1, 8⟩, ⟨.S, 0, 6⟩, ⟨.J, 0, 3⟩, ⟨.T, 2, 4⟩, ⟨.L, 2, 2⟩, ⟨.O, 0, 0⟩, ⟨.I, 0, 6⟩ ]

/-- The lasso table: each visited state paired with the placement played. -/
def lassoWitnessTable : List (GameState × Placement) :=
  lassoTable GameConfig.standard GameState.init lassoPlacements

/-- The loop passes the full closure check: every visited state plays a
bag-legal, in-bounds placement, is non-lost, and steps back into the table. -/
theorem lassoWitnessTable_check :
    checkTable GameConfig.standard lassoWitnessTable = true := by
  native_decide

/-- `init` heads its own lasso table. -/
theorem init_mem_lassoWitnessTable :
    GameState.init ∈ lassoWitnessTable.map Prod.fst := by
  native_decide

/-- **The cooperative lasso, materialised**: a `ClosedCycle` on the canonical
config through `GameState.init` — the first concrete inhabitant of the
M2/M3 artifact. -/
def cooperativeLasso : ClosedCycle GameConfig.standard :=
  ClosedCycle.ofTable GameConfig.standard lassoWitnessTable
    lassoWitnessTable_check

/-- `init` is a state of the cooperative lasso. -/
theorem init_mem_cooperativeLasso :
    GameState.init ∈ cooperativeLasso.states :=
  List.mem_toFinset.mpr init_mem_lassoWitnessTable

/-- **T1 — cooperative Tetris is survivable.** There is a policy (choosing
both draw order and placement) that never tops out from the initial state of
the canonical 10×20 game. Proof by construction: the explicit 35-piece loop. -/
theorem cooperative_tetris_survivable :
    ∃ π : Policy GameConfig.standard,
      SurvivesForever GameConfig.standard π GameState.init :=
  cooperativeLasso.exists_survivesForever_of_init_mem init_mem_cooperativeLasso

/-- Closed cycles through `init` exist on the canonical config (M2-shape
existence, cooperative branching). -/
theorem exists_init_closed_cycle :
    ∃ C : ClosedCycle GameConfig.standard, GameState.init ∈ C.states :=
  ⟨cooperativeLasso, init_mem_cooperativeLasso⟩

end Tetris
