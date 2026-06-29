import Proofs.Safety.SolverProperties

/-!
# The solver as a function — properties of `(board, bag, piece) → placement`

The previous experiment (`SolverProperties`) characterized a solving program by *equivalences*:
existence ⟺ `init ∈ safe` ⟺ a finite cycle ⟺ an atlas. This file asks a different question:
forget the equivalences and study the **function itself**.

A `Solver cfg` is literally a map `GameState → Piece → Placement`, i.e. `(board, bag, piece) ↦
placement`. What does such a function *output*? How constrained is its range? Is it *compressible* —
does it carry more information than it needs, factor through a coarser statistic, or collapse a huge
input space onto a tiny output space? These are properties of the function, not of what it is
equivalent to.

The recurring theme: a valid solver is heavily **over-determined as data and under-determined as a
strategy** — outputs live in a fixed finite set and reduce to two numbers `(rot, col)`, the range
is bounded independently of the (astronomical) domain, and yet *which* function you pick is free
within the safe-closure constraint. The "real" content is a relation, and the function is one
arbitrary uniformization of it.
-/

namespace Tetris

variable {cfg : GameConfig} {σ : Solver cfg}

/-! ## Part 1 — What the function outputs (the shape of an output) -/

/-- **The output announces the input piece — the function does not choose it.** For a valid solver
`(σ g p).piece = p`: the `piece` field of the output is pinned to the input. The placement type
has three fields `(piece, rot, col)`, but one of them is forced — the function's only genuine output
is `(rot, col)`. -/
theorem solver_output_announces_piece (hv : ValidSolver cfg σ) {g : GameState}
    {p : Piece} (hp : p ∈ g.bag) :
    (σ g p).piece = p :=
  (hv g p hp).1

/-- **Every output is a valid, in-field placement.** The function never returns a placement that
puts cells outside the board; its outputs are always `Valid`. -/
theorem solver_output_valid (hv : ValidSolver cfg σ) {g : GameState}
    {p : Piece} (hp : p ∈ g.bag) :
    (σ g p).Valid cfg :=
  (hv g p hp).2

/-- **The output lands in the finite per-piece action set.** `σ g p ∈ Placement.allValidFor cfg p` —
the (finite) `Finset` of all valid placements of piece `p`. So for each drawn piece the function
selects from a *fixed, finite menu*; its range, per piece, is a `Finset`. -/
theorem solver_output_in_action_set (hv : ValidSolver cfg σ) {g : GameState}
    {p : Piece} (hp : p ∈ g.bag) :
    σ g p ∈ Placement.allValidFor cfg p :=
  (Placement.mem_allValidFor cfg p _).mpr ⟨solver_output_announces_piece hv hp,
    solver_output_valid hv hp⟩

/-- **An output is reconstructed from the forced piece plus two numbers.** `σ g p = ⟨p, (σ g p).rot,
(σ g p).col⟩`: the entire output is determined by the input piece together with the chosen rotation
and column. The information the function actually produces is the pair `(rot, col)` — the `piece`
field is redundant data, recoverable from the input. -/
theorem solver_output_eq_mk (hv : ValidSolver cfg σ) {g : GameState}
    {p : Piece} (hp : p ∈ g.bag) :
    σ g p = ⟨p, (σ g p).rot, (σ g p).col⟩ := by
  show Placement.mk (σ g p).piece (σ g p).rot (σ g p).col = ⟨p, (σ g p).rot, (σ g p).col⟩
  rw [Placement.mk.injEq]
  exact ⟨(hv g p hp).1, rfl, rfl⟩

end Tetris
