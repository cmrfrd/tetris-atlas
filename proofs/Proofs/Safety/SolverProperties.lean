import Proofs.Safety.SafeIterateFinite
import Proofs.Invariants.HeightControl

/-!
# Properties of a Tetris-solving program — the solver-characterization experiment

CONJECTURE (`TetrisSolvable`): there exists a `Solver` that survives forever against every legal
7-bag sequence. This file is a proof-experiment asking the *dual* of "is it solvable?": **if a
solving program exists, what must it be and do?** It assembles, as theorems, the necessary and
sufficient properties of any such program, on the safe-set / atlas framework (`Adversarial`,
`SafeSet`, `SafeIterateFinite`) and the difficulty results (`HeightControl`).

The very type `Solver cfg := GameState → Piece → Placement` already answers the first structural
question: the program sees only the CURRENT state and the CURRENT piece — it commits to a placement
with **no lookahead** of future pieces, yet must survive the worst-case sequence. Existence, the
invariant it maintains, the bound it holds, what it must *do* (clear), whether it can be a finite
table, and whether it must cycle, follow.
-/

namespace Tetris

variable {cfg : GameConfig} {σ : Solver cfg}

/-! ## Q1. When does a solving program exist? -/

/-- **A solving program exists iff `init ∈ safe`.** The existence of any valid solver is equivalent
to the single membership `GameState.init ∈ safe cfg` — the greatest fixed point of the one-step
safety operator. So the whole conjecture reduces to one (in-principle decidable) question, and the
*program* exists exactly when the *state* `init` lies in the maximal controlled-invariant set. -/
theorem solver_exists_iff_init_safe (hcols : 4 ≤ cfg.cols) :
    (∃ σ : Solver cfg, SolvesTetrisValid cfg σ) ↔ GameState.init ∈ safe cfg :=
  (init_safe_iff_exists_solvesTetrisValid hcols).symm

/-! ## Q2. What region of the state space does a solving program stay in? -/

/-- **A solving program never leaves `safe`.** Every state on its trace, against every legal
sequence, lies in `safe cfg`. The program is a controller that renders `safe` invariant — it can
visit only states from which it can still survive. -/
theorem solver_trace_mem_safe (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg σ s GameState.init n ∈ safe cfg :=
  solvesTetrisValid_trace_mem_safe h hl n

/-! ## Q3. What does a solving program GUARANTEE about the board? -/

/-- **A solving program holds `maxHeight ≤ rows` forever.** At every step, against every legal
sequence, a solver that never loses keeps the max column height within the ceiling (the survival
metric of `HeightControl`). So "solving" is operationally *"hold the single scalar `maxHeight ≤
rows` for all time"* — the program is a controller keeping one quantity under a bound against an
adversary, armed only with current-piece information and the line-clear primitive. -/
theorem solver_maintains_maxHeight (h : SolvesTetris cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    Board.maxHeight cfg (adversarialTrace cfg σ s GameState.init n).board ≤ cfg.rows :=
  Board.maxHeight_le_rows_of_not_isLost cfg (h s hl n)

/-! ## Q4. Can a solving program ever be stuck (no legal response)? -/

/-- **A solving program is never stuck.** In every state it reaches, every piece the adversary can
draw has a *valid* placement available (because the state is `safe`, and safe states have a response
to each bag piece). So the controller is total — it can never be forced into "no move"; the only
failure mode it must avoid is a move that *leaves* `safe`. -/
theorem solver_never_stuck (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) {p : Piece}
    (hp : p ∈ (adversarialTrace cfg σ s GameState.init n).bag) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg :=
  safe_has_valid_for_each_bag_piece (solver_trace_mem_safe h hl n) hp

/-! ## Q5. Is the board the program maintains bounded? -/

/-- **A solving program keeps the board inside the field: `count ≤ cols·rows`.** Every reachable
state is well-formed (`reachable_WF`) and not lost, so by `count_le_capacity_of_not_isLost` it holds
at most `cols·rows` cells. The program never lets the stack exceed the field's capacity — combined
with `solver_maintains_maxHeight`, its entire reachable set lives in the finite, bounded box of
in-field boards. -/
theorem solver_count_le_capacity (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace cfg σ s GameState.init n).board.count ≤ cfg.cols * cfg.rows := by
  have hwf : Board.WF cfg (adversarialTrace cfg σ s GameState.init n).board :=
    reachable_WF (solverReachable_implies_reachable_of_solves h.1 h.2
      (adversarialTrace_solverReachable σ hl n))
  exact Board.count_le_capacity_of_not_isLost hwf (h.2 s hl n)

/-! ## Q6. How much machinery does a solving program need — memory? randomness? -/

/-- **A single fixed, deterministic, memoryless program suffices.** The very type
`Solver cfg := GameState → Piece → Placement` already grants the program *no* memory of past moves,
*no* lookahead of future pieces, and *no* randomness — it maps the current `(state, piece)` to a
placement. This theorem shows that is enough: whenever the game is solvable, the *canonical*
`safeSolver` (one fixed such function) survives every legal sequence. Solving needs no state
machine, no history, no coin flips — just a pure lookup keyed on the present. -/
theorem canonical_memoryless_solver (hcols : 4 ≤ cfg.cols) (h : GameState.init ∈ safe cfg) :
    SolvesTetrisValid cfg (safeSolver cfg) :=
  init_safe_implies_solvesTetrisValid hcols h

/-- **If any solver works, the canonical one works.** Existence of *some* clever program is
equivalent to one specific, explicitly-constructed program (`safeSolver`) winning. So there is no
gap between "a survivor exists" and "this particular memoryless survivor wins" — no cleverness is
needed beyond membership in `safe`. -/
theorem any_solver_implies_canonical (hcols : 4 ≤ cfg.cols)
    (hex : ∃ σ : Solver cfg, SolvesTetrisValid cfg σ) :
    SolvesTetrisValid cfg (safeSolver cfg) :=
  init_safe_implies_solvesTetrisValid hcols
    ((init_safe_iff_exists_solvesTetrisValid hcols).mpr hex)

/-! ## Q7. Can the solving program be a finite object (a table / cycle)? -/

/-- **A solving program exists iff a finite closed cycle through `init` exists.** The program is not
merely *some* infinite-horizon strategy — it is equivalent to a concrete, finite `AdversarialClosed
CycleWF`: a finite set of states, closed under every adversary draw, containing `init`. This is the
M2/M3/M4 artifact: "solving Tetris" = "exhibit a finite closed cycle from the empty board." -/
theorem solver_exists_iff_init_cycle (hcols : 4 ≤ cfg.cols) :
    (∃ σ : Solver cfg, SolvesTetrisValid cfg σ) ↔
      ∃ C : AdversarialClosedCycleWF cfg,
        GameState.init ∈ C.toAdversarialClosedCycle.states :=
  (solver_exists_iff_init_safe hcols).trans
    (init_safe_iff_exists_init_adversarialClosedCycleWF cfg)

/-- **The program's table is finite and bounded.** Whenever a solver exists, its realizing cycle —
the finite lookup table the program is — has between `28` and `|inFieldStates cfg|` states. So the
Atlas is a genuinely finite object with an a-priori size envelope: not just abstractly finite, but
bounded by the number of in-field game states, the moment solvability is known. -/
theorem solver_table_size_bounded (hcols : 4 ≤ cfg.cols)
    (hex : ∃ σ : Solver cfg, SolvesTetrisValid cfg σ) :
    ∃ C : AdversarialClosedCycleWF cfg,
      GameState.init ∈ C.toAdversarialClosedCycle.states ∧
      28 ≤ C.toAdversarialClosedCycle.states.card ∧
      C.toAdversarialClosedCycle.states.card ≤ (inFieldStates cfg).card :=
  tetrisSolvableValidFor_gives_cycle_card_envelope hcols hex

/-! ## Q8. When is a solving program impossible — and is existence all-or-nothing? -/

/-- **From an unsafe start, no program survives.** If `init ∉ safe cfg`, then *no* valid solver
wins: the impossibility is not "we failed to find one" but a theorem. Death is forced against the
worst-case sequence regardless of how clever the program is. -/
theorem no_solver_from_unsafe_init (hcols : 4 ≤ cfg.cols) (h : GameState.init ∉ safe cfg) :
    ¬ ∃ σ : Solver cfg, SolvesTetrisValid cfg σ :=
  fun hex => h ((solver_exists_iff_init_safe hcols).mp hex)

/-- **Solving is all-or-nothing.** Either the one canonical program `safeSolver` already wins, or
*every* program loses. There is no middle ground where survival is possible but only via some exotic
strategy: the maximal safe set decides it, and when survival is possible the explicit memoryless
lookup achieves it. -/
theorem solver_dichotomy (hcols : 4 ≤ cfg.cols) :
    SolvesTetrisValid cfg (safeSolver cfg) ∨ (∀ σ : Solver cfg, ¬ SolvesTetrisValid cfg σ) := by
  by_cases h : GameState.init ∈ safe cfg
  · exact Or.inl (canonical_memoryless_solver hcols h)
  · exact Or.inr fun σ hσ => h ((solver_exists_iff_init_safe hcols).mp ⟨σ, hσ⟩)

/-! ## Q9. How is a candidate program verified — must we unroll forever? -/

/-- **A one-step certificate suffices (coinduction).** Any set `S` that is *locally* controlled-
invariant — every `g ∈ S` is not lost and, for every drawable piece, has a valid placement landing
back in `S` — is automatically a winning region (`S ⊆ safe`). The infinite-horizon guarantee follows
from a single-step closure check; no unbounded lookahead is needed to certify the program. -/
theorem solver_region_local_certificate (S : Set GameState)
    (hS : ∀ g ∈ S, ¬ g.lost cfg ∧
      ∀ p, p ∈ g.bag → ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
        adversarialStep cfg g p pl ∈ S) :
    S ⊆ safe cfg :=
  safe_greatest S hS

/-- **The Atlas verification recipe.** To prove a solving program exists it suffices to exhibit a
set `S` containing `init` that is closed under one adversarial step. This is exactly the M4 proof
obligation: build a finite closed `S ∋ init`, check local closure, and survival-forever is implied
— turning an infinite property into a finite, mechanically checkable one. -/
theorem solver_exists_of_local_certificate (hcols : 4 ≤ cfg.cols) (S : Set GameState)
    (hinit : GameState.init ∈ S)
    (hS : ∀ g ∈ S, ¬ g.lost cfg ∧
      ∀ p, p ∈ g.bag → ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
        adversarialStep cfg g p pl ∈ S) :
    ∃ σ : Solver cfg, SolvesTetrisValid cfg σ :=
  (solver_exists_iff_init_safe hcols).mpr (solver_region_local_certificate S hS hinit)

/-! ## Q10. What must a solving program physically DO — does it have to clear lines? -/

/-- **Near capacity, the program is forced to clear a line.** If the board is within 4 cells of the
field capacity `cols·rows` and the program's chosen (valid, piece-matching) move keeps the state
safe, then that move *must* clear at least one line. Survival is not passive stacking: at the brink,
the only safe moves are clearing moves — the program must use the line-clear primitive. -/
theorem safe_near_capacity_must_clear {g : GameState} (hwf : Board.WF cfg g.board)
    {p : Piece} {pl : Placement} (hpl : pl.piece = p) (hv : pl.Valid cfg)
    (hnear : cfg.cols * cfg.rows < g.board.count + 4)
    (hsafe' : adversarialStep cfg g p pl ∈ safe cfg) :
    0 < Board.linesCleared cfg (pl.place g.board) := by
  refine Board.must_clear_near_capacity hwf hv hnear ?_
  have hnl : ¬ (adversarialStep cfg g p pl).lost cfg := safe_not_lost hsafe'
  simpa [GameState.lost, adversarialStep, Placement.eta_of_piece_eq hpl] using hnl

/-! ## Q11. How fast can danger arrive — can the program be caught off guard? -/

/-- **Danger accrues at most 4 height per move.** A single placed piece raises the max column
height by at most 4. So the program is never surprised: from `maxHeight = m` it has a margin of
`rows - m` that the adversary can erode only 4 at a time — survival is a slow-moving control
problem, never a one-step ambush. -/
theorem safe_step_maxHeight_le_add_four {g : GameState} {p : Piece} {pl : Placement}
    (hpl : pl.piece = p) (hv : pl.Valid cfg) :
    Board.maxHeight cfg (adversarialStep cfg g p pl).board
      ≤ Board.maxHeight cfg g.board + 4 := by
  have h := Board.maxHeight_applyStep_le_add_four (cfg := cfg) g.board hv
  simpa [adversarialStep, Placement.eta_of_piece_eq hpl] using h

end Tetris
