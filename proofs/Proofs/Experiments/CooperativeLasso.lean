import Mathlib
import Proofs.Survival.Lasso

/-!
# The cooperative lasso witness (T1)

**The first concrete infinite-play certificate in this repository — for the
COOPERATIVE game only.** An explicit 35-placement loop (5 bags, each drawing
all 7 pieces in a chosen order) that takes `GameState.init` (empty board, full
bag) back to `GameState.init` exactly, never lost. Certified from scratch by
`checkTable` (`Proofs/Survival/Lasso.lean`) via `native_decide`, then converted
into a `ClosedCycle` and the headline theorem

  `cooperative_selfDealt_survives_forever :
     ∃ π, SurvivesForever GameConfig.standard π GameState.init`

⚠ **This is NOT "Tetris is survivable".** In this *cooperative* / self-dealing
game the policy `Policy = GameState → Placement` chooses BOTH which piece to
draw next AND where to place it: the piece played is `(π g).piece`, and `step`
draws exactly that piece. So the "player" also deals itself the pieces — there
is one self-directed play, and `SurvivesForever` says only that this one play
never tops out. Real Tetris is `TetrisSolvable` (`Proofs/Safety/Adversarial`):
`∃ σ : Solver, SolvesTetris σ`, where `Solver = GameState → Piece → Placement`
is *handed* each announced piece and must survive **every** legal 7-bag
sequence (∀-announcement). That is the project's open central conjecture, and
nothing here bears on it. This theorem's only job is to *de-vacuize*
`ClosedCycle`, `closed_cycle_survives`, and `SurvivesForever` on the canonical
10×20 config — one concrete inhabitant of the M2/M3 cooperative artifact.

Loop arithmetic: 35 pieces = 5 bags place 140 cells and clear exactly 14
lines (`4·35 = 10·14`), the minimum possible loop length (cell conservation
forces `L ≡ 0 mod 5`, bag renewal forces `L ≡ 0 mod 7`).

The placement list was found by `scripts/find_cooperative_lasso.py` (a one-off
with-holes bitboard beam mirroring this model — transient holes are essential:
a hole-free/flush 5-bag perfect clear does not exist). The script carries no
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

/-- **T1 — cooperative (self-dealing) infinite play from `init`.** There is a
policy that never tops out from `init` on the canonical 10×20 board **when the
policy also chooses the piece order** (`Policy = GameState → Placement`; the
piece played is `(π g).piece`, which `step` draws). Proof by construction: the
explicit 35-piece `init → init` loop.

⚠ This is the *cooperative* claim, NOT `TetrisSolvable`. It does **not** say a
player can survive an adversary that announces the pieces — that is the open
`∃ σ : Solver, SolvesTetris σ` (surviving *every* legal 7-bag sequence). The
name is deliberately not `..._tetris_survivable`; see the file header. -/
theorem cooperative_selfDealt_survives_forever :
    ∃ π : Policy GameConfig.standard,
      SurvivesForever GameConfig.standard π GameState.init :=
  cooperativeLasso.exists_survivesForever_of_init_mem init_mem_cooperativeLasso

/-- Closed cycles through `init` exist on the canonical config — the M2-shape
*cooperative* existence (the cycle's `policy` self-deals its pieces). This is
the ∃-form behind `cooperative_selfDealt_survives_forever`; it is **not** the
adversarial closed *atlas* (`Atlas.IsClosedOn`) that `TetrisSolvable` needs. -/
theorem exists_init_closed_cycle :
    ∃ C : ClosedCycle GameConfig.standard, GameState.init ∈ C.states :=
  ⟨cooperativeLasso, init_mem_cooperativeLasso⟩

end Tetris
