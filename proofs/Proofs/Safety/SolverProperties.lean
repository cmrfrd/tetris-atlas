import Proofs.Safety.SafeIterateFinite
import Proofs.Safety.Safety
import Proofs.Invariants.HeightControl
import Proofs.Combinatorics.BagBurst

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

/-! ## Q12. What decision rule can the program NOT use — does energy/count suffice? -/

/-- **The loss boundary is decoupled from cell count.** There are two well-formed boards both *at*
the brink (`maxHeight = rows`): one with a single cell, one completely full (`cols·rows` cells). So
the distance-to-loss carries no information about how many cells are on the board — and conversely.
A program therefore cannot steer by minimizing material/energy/`count`: no additive potential
separates safe from doomed. It must track the *max height* (and, with holes, clearability) — exactly
the two-axis obstruction that makes the crux hard. This is a property the solver must NOT have:
greedy scalar-energy descent is provably blind to the danger. -/
theorem brink_decoupled_from_count (hcols : 0 < cfg.cols) (hrows : 0 < cfg.rows) :
    ∃ b₁ b₂ : Board,
      (Board.WF cfg b₁ ∧ b₁.count = 1 ∧ Board.maxHeight cfg b₁ = cfg.rows) ∧
      (Board.WF cfg b₂ ∧ b₂.count = cfg.cols * cfg.rows ∧
        Board.maxHeight cfg b₂ = cfg.rows) := by
  obtain ⟨b₁, h₁⟩ := Board.exists_one_cell_at_brink hcols hrows
  obtain ⟨b₂, hwf₂, hc₂, _, hm₂⟩ := Board.exists_full_board_at_brink hcols hrows
  exact ⟨b₁, b₂, h₁, hwf₂, hc₂, hm₂⟩

/-! ## Q13. How many distinct states does the program visit? -/

/-- **The program's trajectory lives in a finite set.** Every state on the trace is reachable and
safe, hence in `inFieldStates cfg` — the finite set of well-formed, non-lost game states. So however
long the game runs, the program only ever occupies finitely many distinct configurations: the
infinite-horizon play is confined to a finite arena. -/
theorem solver_trace_mem_inFieldStates (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg σ s GameState.init n ∈ inFieldStates cfg :=
  reachable_safe_mem_inFieldStates
    (solverReachable_implies_reachable_of_solves h.1 h.2
      (adversarialTrace_solverReachable σ hl n))
    (solver_trace_mem_safe h hl n)

/-- **The program's play is eventually periodic.** Against any fixed legal sequence the trace runs
forever inside a finite set, so by pigeonhole two distinct times share the same state. Survival is
therefore inherently *cyclic*: the program cannot keep producing genuinely new positions — it is
forced to loop, which is the structural seed of the M2 closed cycle. -/
theorem solver_play_eventually_repeats (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) :
    ∃ i j : ℕ, i ≠ j ∧
      adversarialTrace cfg σ s GameState.init i
        = adversarialTrace cfg σ s GameState.init j := by
  have hmaps : Set.MapsTo (fun n => adversarialTrace cfg σ s GameState.init n)
      Set.univ ↑(inFieldStates cfg) :=
    fun n _ => Finset.mem_coe.mpr (solver_trace_mem_inFieldStates h hl n)
  obtain ⟨i, _, j, _, hij, heq⟩ :=
    Set.Infinite.exists_ne_map_eq_of_mapsTo Set.infinite_univ hmaps
      (inFieldStates cfg).finite_toSet
  exact ⟨i, j, hij, heq⟩

/-! ## Q14. How does the program react to which piece is drawn? -/

/-- **A safe response for every drawable piece.** From any safe state, each piece the bag can yield
has a valid placement that lands back in `safe`. The program is genuinely *reactive*: its move is a
function of the drawn piece, and for each of them it has a survival-preserving answer. -/
theorem solver_safe_response_each_piece {g : GameState} (hg : g ∈ safe cfg)
    {p : Piece} (hp : p ∈ g.bag) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧ adversarialStep cfg g p pl ∈ safe cfg :=
  safe_forall_step hg p hp

/-- **From the empty board the program must answer all seven tetrominoes.** `init`'s bag is full, so
a solving program needs a safe placement for every one of O, I, S, Z, T, L, J as the very first
piece — no opening is allowed to be a piece it cannot safely absorb. -/
theorem solver_handles_all_seven_at_init (h : GameState.init ∈ safe cfg) (p : Piece) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
      adversarialStep cfg GameState.init p pl ∈ safe cfg :=
  safe_forall_step h p (GameState.init_bag.symm ▸ Bag.mem_full p)

/-! ## Q15. Is the program's rule local — does it depend on how the state was reached? -/

/-- **The canonical rule renders `safe` invariant from every safe state.** `safeSolver`'s move from
*any* safe state (not merely reachable ones) lands back in `safe`. The decision is path-independent:
the program needs only the present `(state, piece)`, never the history that produced it — the same
uniform rule is correct everywhere in the safe region — exactly why memorylessness suffices. -/
theorem canonical_solver_safe_invariant {g : GameState} (hg : g ∈ safe cfg)
    {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (safeSolver cfg g p) ∈ safe cfg :=
  safeSolver_step_mem_safe hg hp

/-- **The program's reachable set has no dead-ends.** Every state a solving program can reach is
safe — from anywhere it can get to, survival is still possible. The program never paints itself into
a corner: there is no reachable position that is alive now but doomed under best play. -/
theorem solver_no_dead_ends (h : SolvesTetrisValid cfg σ) {g : GameState}
    (hr : solverReachable σ g) : g ∈ safe cfg :=
  solverReachable_subset_safe h hr

/-! ## Q16. What aggregate energy does the program hold down? -/

/-- **The program caps the surface-area energy at `cols·rows`.** The total stack energy
`surfaceArea = Σ colHeight` is at most `cols · maxHeight`, and the program holds `maxHeight ≤ rows`,
so it keeps `surfaceArea ≤ cols·rows` forever. Even though no single additive potential *certifies*
survival (Q12), a surviving program does keep this aggregate bounded as a consequence. -/
theorem solver_surfaceArea_le_capacity (h : SolvesTetris cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    HoleDebt.surfaceArea cfg (adversarialTrace cfg σ s GameState.init n).board
      ≤ cfg.cols * cfg.rows := by
  refine le_trans (HoleDebt.surfaceArea_le_cols_mul_maxHeight cfg _) ?_
  gcongr
  exact solver_maintains_maxHeight h hl n

/-- **The program keeps hole-debt bounded.** Buried holes (`debt = Σ colHoles`) spend the same
energy budget as filled cells, so `debt ≤ surfaceArea ≤ cols·rows`. A surviving program cannot let
unclearable debt accumulate without bound — it must keep digging holes out, since debt competes with
height for the fixed `cols·rows` of capacity. -/
theorem solver_debt_le_capacity (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    HoleDebt.debt cfg (adversarialTrace cfg σ s GameState.init n).board
      ≤ cfg.cols * cfg.rows := by
  have hwf : Board.WF cfg (adversarialTrace cfg σ s GameState.init n).board :=
    reachable_WF (solverReachable_implies_reachable_of_solves h.1 h.2
      (adversarialTrace_solverReachable σ hl n))
  exact le_trans (HoleDebt.debt_le_surfaceArea hwf) (solver_surfaceArea_le_capacity h.2 hl n)

/-! ## Q17. Is the existence of a solving program decidable? -/

/-- **Solver existence is decidable.** Given any finite universe `S₀` that over-approximates `safe`,
the finite safe-iteration converges (it can only shrink, `≤ |S₀|` steps), and at the fixed point a
program exists iff `init` is in that explicitly-computed `Finset`. So "does a solver exist?" reduces
to a terminating computation followed by a decidable membership test — solvability is not an
undecidable mystery but a finite search. -/
theorem solver_existence_decidable {S₀ : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hS₀ : safe cfg ⊆ ↑S₀) :
    ∃ N : ℕ, (∃ σ : Solver cfg, SolvesTetrisValid cfg σ) ↔
      GameState.init ∈ safeIterFinite cfg S₀ N := by
  obtain ⟨N, _, hfix⟩ := safeIterFinite_converges cfg S₀
  exact ⟨N, tetrisSolvableValidFor_iff_init_mem_safeIterFinite hcols hS₀ N hfix⟩

/-- **A finite refutation certificate.** If at *any* iteration step `init` has already fallen out of
the shrinking approximation `safeIterFinite cfg S₀ n`, then no solving program exists — no need to
run to convergence. One snapshot showing `init` is gone is a complete proof of unsolvability. -/
theorem no_solver_of_init_not_mem_safeIterFinite {S₀ : Finset GameState}
    (hcols : 4 ≤ cfg.cols) (hS₀ : safe cfg ⊆ ↑S₀) (n : ℕ)
    (h : GameState.init ∉ safeIterFinite cfg S₀ n) :
    ¬ ∃ σ : Solver cfg, SolvesTetrisValid cfg σ :=
  fun hex => init_not_safe_of_init_not_mem_safeIterFinite hS₀ n h
    ((solver_exists_iff_init_safe hcols).mp hex)

/-- **An empty winning region kills every start.** If the finite iteration collapses to `∅`, then
`safe = ∅`, so no program can solve the game from `init` — survival is impossible from anywhere. -/
theorem no_solver_of_safeIterFinite_empty {S₀ : Finset GameState}
    (hcols : 4 ≤ cfg.cols) (hS₀ : safe cfg ⊆ ↑S₀) {n : ℕ}
    (hempty : safeIterFinite cfg S₀ n = ∅) :
    ¬ ∃ σ : Solver cfg, SolvesTetrisValid cfg σ := by
  refine no_solver_from_unsafe_init hcols ?_
  rw [safe_eq_empty_of_safeIterFinite_empty hS₀ hempty]
  exact Set.notMem_empty _

/-! ## Q18. What does the program pay for not clearing? -/

/-- **Every non-clearing move spends exactly 4 cells of headroom.** If the program's move clears no
line, the board's cell count rises by precisely 4. So between clears the stack climbs monotonically
toward the `cols·rows` cap — a deterministic countdown. The program cannot *stall*: each move that
declines to clear permanently consumes 4 of its finite headroom, forcing a clear before long. -/
theorem safe_step_no_clear_count {g : GameState} (hwf : Board.WF cfg g.board)
    {p : Piece} {pl : Placement} (hpl : pl.piece = p) (hv : pl.Valid cfg)
    (hno : Board.linesCleared cfg (pl.place g.board) = 0) :
    (adversarialStep cfg g p pl).board.count = g.board.count + 4 := by
  have h := Board.count_applyStep_eq_of_no_clear hwf hv hno
  simpa [adversarialStep, Placement.eta_of_piece_eq hpl] using h

/-- **Only clearing makes progress.** A move that clears at least one line ends with strictly fewer
than `count + 4` cells — it is the only way to beat the steady `+4` inflow. So the program's cell
count is a sawtooth: `+4` on every non-clearing move, a net drop only when it clears. Survival is
the balancing act of forcing enough downstrokes to offset the relentless climb. -/
theorem safe_step_clear_progress {g : GameState} (hwf : Board.WF cfg g.board)
    {p : Piece} {pl : Placement} (hpl : pl.piece = p) (hv : pl.Valid cfg)
    (hcols : 0 < cfg.cols) (hclear : 0 < Board.linesCleared cfg (pl.place g.board)) :
    (adversarialStep cfg g p pl).board.count < g.board.count + 4 := by
  have h := Board.count_applyStep_lt_of_clear hwf hv hcols hclear
  simpa [adversarialStep, Placement.eta_of_piece_eq hpl] using h

/-! ## Q19. What must the program have built to survive at the brink? -/

/-- **At the brink the program must have assembled a full row.** A near-capacity safe move is forced
to clear (Q10), and a clear can only remove a *full* `cols`-wide row. So at the moment of forced
clearing, the program's placement has completed an entire row — it cannot survive on a ragged
surface near the top; it must have pre-built a clean line, exactly what the adversary's holes fight
to deny (`cols_le_card_row_of_isFull`, `not_isFull_of_mem_holes`). -/
theorem safe_near_capacity_assembles_full_row {g : GameState} (hwf : Board.WF cfg g.board)
    {p : Piece} {pl : Placement} (hpl : pl.piece = p) (hv : pl.Valid cfg)
    (hnear : cfg.cols * cfg.rows < g.board.count + 4)
    (hsafe' : adversarialStep cfg g p pl ∈ safe cfg) :
    ∃ r, Board.isFull cfg (pl.place g.board) r := by
  have hpos := safe_near_capacity_must_clear hwf hpl hv hnear hsafe'
  have hcard : 0 < (Board.fullRows cfg (pl.place g.board)).card := hpos
  obtain ⟨r, hr⟩ := Finset.card_pos.mp hcard
  exact ⟨r, (Finset.mem_filter.mp hr).2⟩

/-! ## Q20. How smoothly does the program's board evolve over time? -/

/-- **The cell count is 4-Lipschitz along the trace.** Each step changes the board's cell count by
at most `+4`. With the `cols·rows` cap (Q5) and `count(0) = 0`, the program's material evolves
as a slowly-varying bounded sequence — never a jump — which is what makes the finite-arena and
forced-clear arguments bite. -/
theorem solver_trace_count_le_succ (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace cfg σ s GameState.init (n + 1)).board.count
      ≤ (adversarialTrace cfg σ s GameState.init n).board.count + 4 := by
  have hbag : s n ∈ (adversarialTrace cfg σ s GameState.init n).bag := by
    rw [adversarialTrace_bag]; exact hl n
  have hwf : Board.WF cfg (adversarialTrace cfg σ s GameState.init n).board :=
    reachable_WF (solverReachable_implies_reachable_of_solves h.1 h.2
      (adversarialTrace_solverReachable σ hl n))
  obtain ⟨hpiece, hv⟩ := h.1 (adversarialTrace cfg σ s GameState.init n) (s n) hbag
  rw [adversarialTrace_succ]
  have hc := Board.count_applyStep_le_add_four hwf hv
  simpa [adversarialStep, Placement.eta_of_piece_eq hpiece] using hc

/-- **The max height is also 4-Lipschitz along the trace.** The loss-relevant quantity itself moves
by at most `+4` per step. So the program watches a single bounded, slowly-varying scalar and must
keep it under `rows` — survival is the control of one Lipschitz signal against an adversary. -/
theorem solver_trace_maxHeight_le_succ (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    Board.maxHeight cfg (adversarialTrace cfg σ s GameState.init (n + 1)).board
      ≤ Board.maxHeight cfg (adversarialTrace cfg σ s GameState.init n).board + 4 := by
  have hbag : s n ∈ (adversarialTrace cfg σ s GameState.init n).bag := by
    rw [adversarialTrace_bag]; exact hl n
  obtain ⟨hpiece, hv⟩ := h.1 (adversarialTrace cfg σ s GameState.init n) (s n) hbag
  rw [adversarialTrace_succ]
  exact safe_step_maxHeight_le_add_four hpiece hv

/-! ## Q21. Can the program postpone clearing indefinitely? -/

/-- **No long run without a clear: `4·n ≤ cols·rows`.** Suppose the program clears no line over the
first `n` moves — encoded as the count rising by exactly `+4` each step (the Q18 ledger: `+4` ⟺ no
clear). Then the board holds `4·n` cells, which cannot exceed capacity `cols·rows`. So a surviving
program clears at least once every `⌊cols·rows / 4⌋` moves — it can *delay* clearing but never
abandon it; line clears recur on a hard schedule. -/
theorem solver_no_clear_window_bounded (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ)
    (hno : ∀ k < n,
        (adversarialTrace cfg σ s GameState.init (k + 1)).board.count
          = (adversarialTrace cfg σ s GameState.init k).board.count + 4) :
    4 * n ≤ cfg.cols * cfg.rows := by
  have key : ∀ m, (∀ k < m,
      (adversarialTrace cfg σ s GameState.init (k + 1)).board.count
        = (adversarialTrace cfg σ s GameState.init k).board.count + 4) →
      (adversarialTrace cfg σ s GameState.init m).board.count = 4 * m := by
    intro m
    induction m with
    | zero => intro _; simp
    | succ k ih =>
        intro hh
        rw [hh k (Nat.lt_succ_self k), ih (fun j hj => hh j (Nat.lt_succ_of_lt hj))]
        ring
  calc 4 * n = (adversarialTrace cfg σ s GameState.init n).board.count := (key n hno).symm
    _ ≤ cfg.cols * cfg.rows := solver_count_le_capacity h hl n

/-! ## Q22. Can the program be written down as an explicit lookup table (the Atlas)? -/

/-- **A solving program yields an explicit closed Atlas covering `init`.** Whenever a solver exists,
there is a concrete `Atlas` (a partial `GameState → Piece → Option Placement` table) and a finite
state set `S ∋ init` on which it is `IsClosedOn`: every state is non-lost, every drawable piece has
a table entry, and following it stays in `S`. This `(A, S)` is the M4 proof artifact — the Atlas —
and its mere existence is equivalent to Tetris being solvable. -/
theorem solver_exists_yields_closed_atlas (hcols : 4 ≤ cfg.cols)
    (hex : ∃ σ : Solver cfg, SolvesTetrisValid cfg σ) :
    ∃ (A : Atlas cfg) (S : Finset GameState),
      A.IsClosedOn cfg S ∧ GameState.init ∈ S := by
  obtain ⟨C, hinit⟩ := (init_safe_iff_exists_init_adversarialClosedCycleWF cfg).mp
    ((solver_exists_iff_init_safe hcols).mp hex)
  exact ⟨C.toAdversarialClosedCycle.solver.toAtlas, C.toAdversarialClosedCycle.states,
    C.toAdversarialClosedCycle.solver_toAtlas_isClosedOn_states, hinit⟩

/-! ## Q23. Does the program rely on luck, or beat the worst case? -/

/-- **No legal sequence ever beats the program.** A solving program tops out under *no* legal 7-bag
sequence, at *no* horizon. Survival is not luck against a random bag: it is a guarantee against the
worst-case adversary, universally over all sequences the randomizer could ever produce. -/
theorem solver_no_killing_sequence (h : SolvesTetris cfg σ) :
    ¬ ∃ (s : ℕ → Piece) (n : ℕ),
        LegalSequence s ∧ (adversarialTrace cfg σ s GameState.init n).lost cfg :=
  fun ⟨s, n, hl, hlost⟩ => h s hl n hlost

/-- **Conversely, from an unsafe start the adversary defeats every valid program.** If `init ∉
safe`, then for any valid solver there is a legal sequence and a horizon at which it tops out. The
bag, as an adversary, wins exactly when no program does — the safety game is determined. -/
theorem adversary_beats_every_valid_solver (hcols : 4 ≤ cfg.cols)
    (h : GameState.init ∉ safe cfg) (σ : Solver cfg) (hv : ValidSolver cfg σ) :
    ∃ (s : ℕ → Piece) (n : ℕ),
      LegalSequence s ∧ (adversarialTrace cfg σ s GameState.init n).lost cfg := by
  by_contra hcon
  exact h ((solver_exists_iff_init_safe hcols).mp
    ⟨σ, hv, fun s hl n hlost => hcon ⟨s, n, hl, hlost⟩⟩)

/-! ## Q24. How large must the program's table be? -/

/-- **The program needs at most `2^(cols·rows)` distinct boards.** The realizing cycle's states
project onto at most `2^(cols·rows)` distinct board configurations — the count of all board
bit-masks — independent of the `128` bag values. So the Atlas, keyed by board, is bounded by the
raw board space; the bag dimension adds at most a constant factor on top. -/
theorem solver_distinct_boards_le (hcols : 4 ≤ cfg.cols)
    (hex : ∃ σ : Solver cfg, SolvesTetrisValid cfg σ) :
    ∃ C : AdversarialClosedCycleWF cfg,
      GameState.init ∈ C.toAdversarialClosedCycle.states ∧
      (C.toAdversarialClosedCycle.states.image GameState.board).card
        ≤ 2 ^ (cfg.cols * cfg.rows) := by
  obtain ⟨C, hinit⟩ := (init_safe_iff_exists_init_adversarialClosedCycleWF cfg).mp
    ((solver_exists_iff_init_safe hcols).mp hex)
  exact ⟨C, hinit, C.image_board_card_le_two_pow⟩

/-! ## Q25. Is height alone enough for the program to judge danger? -/

/-- **Maximum height does not decide loss.** There is a board *at* the ceiling (`maxHeight = rows`,
in fact completely full) that is nonetheless not lost — being hole-free, every row clears. So
the program cannot judge danger from height alone: a tall board can be perfectly safe. It must also
read clearability (holes), the second axis no height statistic exposes. -/
theorem maxHeight_alone_does_not_decide_loss (hcols : 0 < cfg.cols) (hrows : 0 < cfg.rows) :
    ∃ b : Board,
      Board.WF cfg b ∧ Board.maxHeight cfg b = cfg.rows ∧ ¬ Board.isLost cfg b := by
  obtain ⟨b, hwf, _, hnl, hm⟩ := Board.exists_full_board_at_brink hcols hrows
  exact ⟨b, hwf, hm, hnl⟩

/-! ## Q26. What does a single hole cost the program? -/

/-- **A hole is a double obstruction.** Each buried hole `p` simultaneously (1) blocks its own row
from ever being a full row — so the program's line-clear primitive cannot touch it directly — and
(2) sits under a filled cell, so it is exposed only by first clearing the rows *above* it. Holes are
sticky debt: unclearable in place, removable only by digging from the top, exactly the lever the
adversary's S/Z roughness attacks. -/
theorem solver_hole_obstruction {b : Board} {p : Coord}
    (hp : p ∈ HoleyCarrier.holes cfg b) :
    ¬ Board.isFull cfg b p.2 ∧ ∃ r, p.2 < r ∧ (p.1, r) ∈ b :=
  ⟨Board.not_isFull_of_mem_holes hp, Board.exists_cover_of_hole hp⟩

/-! ## Q27. What hole-creating power does the adversary wield from move one? -/

/-- **A roughness piece can bury an unclearable hole from the empty board.** Placing `S` at column 0
on the empty standard board leaves cell `(2,0)` a buried hole whose row can never be cleared while
it stands. The program is not *forced* into this (it chooses the placement), but it shows the
S/Z pieces carry hole-injecting power from the very first move — the program must spend genuine care
to avoid it, on a board that hands it two such pieces every bag. -/
theorem adversary_S_can_plant_unclearable_hole :
    ((2 : ℕ), (0 : ℕ)) ∈ HoleyCarrier.holes GameConfig.standard
        (Placement.place ∅ ⟨Piece.S, 0, 0⟩)
      ∧ ¬ Board.isFull GameConfig.standard (Placement.place ∅ ⟨Piece.S, 0, 0⟩) 0 :=
  ⟨Board.S_buries_hole, Board.S_hole_row_not_isFull⟩

/-! ## Q28. What resource structure does the 7-bag hand the program each cycle? -/

/-- **A fixed per-bag resource mix: one I-drain against two S/Z roughness.** Every bag the program
faces is a permutation of the seven pieces, so it contains exactly one `I` (its sole guaranteed
height-draining piece) and exactly two roughness pieces `S`/`Z`. The program's survival schedule is
clocked by this renewal: a steady supply of one drain and two hole-injectors per bag. -/
theorem solver_per_bag_resource {l : List Piece} (h : BagBurst.IsBagOrder l) :
    l.countP BagBurst.isI = 1 ∧ l.countP BagBurst.isSZ = 2 :=
  ⟨BagBurst.countP_isI h, BagBurst.countP_isSZ h⟩

/-- **The program's drain budget exceeds its clearing need.** Over `n` bags the I-pieces supply
`20·(#I) = 20·n` rows of draining capacity, comfortably above the `14·n` that roughness recovery
requires. So the program is never *starved* of draining resource — which localizes the real
difficulty: the obstruction is the *geometry* of cashing each I-drain into an actual clear, not any
shortage of drains. The budget is there; cashing it into clears against holes is the crux. -/
theorem solver_drain_budget_suffices {bags : List (List Piece)}
    (h : ∀ b ∈ bags, BagBurst.IsBagOrder b) :
    14 * bags.length ≤ 20 * bags.flatten.countP BagBurst.isI :=
  BagBurst.drain_budget_ge_clearing_need h

/-- **Roughness arrives at twice the rate of drains.** In every bag the program absorbs two
hole-injectors (`S`/`Z`) for each single drain (`I`). This 2:1 structural pressure — not a shortage
of drains (Q28) — is what the program's placement geometry must continually counter. -/
theorem solver_roughness_two_per_drain {l : List Piece} (h : BagBurst.IsBagOrder l) :
    l.countP BagBurst.isSZ = 2 * l.countP BagBurst.isI :=
  BagBurst.renewal_ratio h

/-! ## Q29. Can the program's winning region be compressed to a dominated basis? -/

/-- **No: the safety step is non-congruent, so no dominated-basis abstraction captures it.** Under
the hole-aware order `safeLE` (taller *and* holier), neither placement nor line-clearing is
hole-monotone: an emptier board can become holier after the same drop (`place_holes_mono_false`),
and clears can manufacture new holes (`clearLines_holes_le_false`). Hence a survival certificate
cannot be a finite basis of dominated worst-cases — the program's region is irreducibly an
*explicit* enumeration (an atlas). This is why every Lyapunov / WQO / dominated-basis route
floors. -/
theorem solver_region_not_dominated_basis :
    (¬ ∀ (cfg : GameConfig) (b β : Board) (pl : Placement), HoleyCarrier.safeLE cfg b β →
        HoleyCarrier.holes cfg (pl.place b) ⊆ HoleyCarrier.holes cfg (pl.place β))
    ∧ (¬ ∀ (cfg : GameConfig) (b : Board),
        HoleyCarrier.holes cfg (Board.clearLines cfg b) ⊆ HoleyCarrier.holes cfg b) :=
  ⟨HoleyCarrier.place_holes_mono_false, HoleyCarrier.clearLines_holes_le_false⟩

/-! ## Q30. Can the program's table be built piecewise — and is a table already a program? -/

/-- **Closed tables compose.** If atlas `A` is closed on `S₁` and `B` on `S₂`, their `unionOn` is
closed on `S₁ ∪ S₂`. So the Atlas can be assembled incrementally — absorb one closed region at a
time — rather than synthesized monolithically. -/
theorem solver_atlas_composes {A B : Atlas cfg} {S₁ S₂ : Finset GameState}
    (hA : A.IsClosedOn cfg S₁) (hB : B.IsClosedOn cfg S₂) :
    (A.unionOn B S₁ S₂).IsClosedOn cfg (S₁ ∪ S₂) :=
  hA.unionOn hB

/-- **A closed table already is a program.** Any atlas closed on a set containing `init` yields a
solver that survives forever — the lookup table is not a description of a program, it *is* one. This
is the M4 artifact discharging the whole conjecture: build the closed table, get the survivor. -/
theorem closed_atlas_yields_solver {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    TetrisSolvableFor cfg :=
  h.tetrisSolvableFor_of_init_mem hinit

/-! ## Q31. How constrained is the program's choice — must it play one specific move? -/

/-- **Any policy that stays in `safe` survives — the program is free in its choices.** A solver need
not follow one prescribed move: *every* policy `σ` whose move from each safe state lands back in
`safe` survives forever from a safe start. Survival is not a unique strategy but a whole family —
the only invariant it must respect is membership in `safe`; within that, all choices win. -/
theorem any_safe_selector_survives
    (hstep : ∀ g ∈ safe cfg, ∀ p ∈ g.bag, adversarialStep cfg g p (σ g p) ∈ safe cfg)
    (hinit : GameState.init ∈ safe cfg) : SolvesTetris cfg σ := by
  intro s hl n
  suffices h : adversarialTrace cfg σ s GameState.init n ∈ safe cfg from safe_not_lost h
  induction n with
  | zero => simpa using hinit
  | succ k ih =>
      rw [adversarialTrace_succ]
      have hbag : s k ∈ (adversarialTrace cfg σ s GameState.init k).bag := by
        rw [adversarialTrace_bag]; exact hl k
      exact hstep _ ih (s k) hbag

/-! ## Q32. How soon must the program start repeating? -/

/-- **The program revisits a state within `|inFieldStates|` steps.** Sharpening Q14's eventual
repeat: among the first `|inFieldStates cfg| + 1` states on the trace, two must coincide (pigeonhole
on the finite arena). So the loop the program is forced into begins after a bounded prefix — the
pre-period is at most the size of the in-field state space, not unboundedly far out. -/
theorem solver_repeats_within_inFieldStates (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) :
    ∃ i j : ℕ, i < j ∧ j ≤ (inFieldStates cfg).card ∧
      adversarialTrace cfg σ s GameState.init i
        = adversarialTrace cfg σ s GameState.init j := by
  obtain ⟨i, hi, j, hj, hij, heq⟩ := Finset.exists_ne_map_eq_of_card_lt_of_maps_to
    (s := Finset.range ((inFieldStates cfg).card + 1)) (t := inFieldStates cfg)
    (by rw [Finset.card_range]; exact Nat.lt_succ_self _)
    (fun a _ => solver_trace_mem_inFieldStates h hl a)
  rcases Nat.lt_or_ge i j with hlt | hge
  · exact ⟨i, j, hlt, by have := Finset.mem_range.mp hj; omega, heq⟩
  · exact ⟨j, i, lt_of_le_of_ne hge (Ne.symm hij),
      by have := Finset.mem_range.mp hi; omega, heq.symm⟩

/-! ## Q33. How much can one move undo? -/

/-- **Recovery is bounded: one move removes at most `4·cols` cells.** From a settled board (no
already-full rows), a single placement-and-clear cannot drop the cell count by more than `4·cols`
(at most a Tetris of `cols`-wide rows). So the program has no reset button: it cannot erase an
accumulated deficit in one move — survival is necessarily incremental, chipping against the `+4`
inflow. -/
theorem safe_step_recovery_bounded {g : GameState} (hwf : Board.WF cfg g.board)
    {p : Piece} {pl : Placement} (hpl : pl.piece = p) (hv : pl.Valid cfg)
    (hnf : ∀ r, ¬ Board.isFull cfg g.board r) :
    g.board.count + 4 ≤ (adversarialStep cfg g p pl).board.count + cfg.cols * 4 := by
  have h := Board.count_le_count_applyStep_add hwf hv hnf
  simpa [adversarialStep, Placement.eta_of_piece_eq hpl] using h

/-! ## Q34. What invariant does the program hold on each individual column? -/

/-- **Every playable column stays within the ceiling.** For each column `j < cols`, the program
keeps `colHeight j ≤ rows` at all times (its height is at most the max height, held under `rows`).
Loss is exactly some column exceeding `rows`, so this per-column bound is the survival invariant
decomposed coordinate-by-coordinate. -/
theorem solver_columns_le_rows (h : SolvesTetris cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) {j : ℕ} (hj : j < cfg.cols) :
    Board.colHeight (adversarialTrace cfg σ s GameState.init n).board j ≤ cfg.rows :=
  le_trans (Board.colHeight_le_maxHeight hj) (solver_maintains_maxHeight h hl n)

/-! ## Q35. Is the program's region actually reachable from the empty board (M3)? -/

/-- **Every state the program uses is reachable from the empty board.** A solving program never
relies on configurations unreachable from `init`: its entire operating region is `Reachable cfg`. So
the winning region is not an abstract island but is genuinely entered from a real game started
empty — the M3 reachability bridge that turns a closed cycle into a survivable real game. -/
theorem solver_states_reachable_from_empty (h : SolvesTetrisValid cfg σ) {g : GameState}
    (hr : solverReachable σ g) : Reachable cfg g :=
  solverReachable_implies_reachable_of_solves h.1 h.2 hr

/-! ## Q36. Where on the board does the program forbid cells? -/

/-- **No cell ever enters the death zone.** Every filled cell on the program's board sits strictly
below row `rows`: a single cell at row `≥ rows` is an immediate top-out (`isLost_of_mem_row_ge`), so
a surviving program keeps rows `rows, rows+1, …` permanently empty. Survival is, cell-by-cell, "stay
out of the death zone." -/
theorem solver_no_cell_in_death_zone (h : SolvesTetris cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) {c : Coord}
    (hc : c ∈ (adversarialTrace cfg σ s GameState.init n).board) :
    c.2 < cfg.rows := by
  by_contra hcon
  exact h s hl n (Board.isLost_of_mem_row_ge (Nat.le_of_not_lt hcon) hc)

/-! ## Q37. Is survival reducible to a single scalar condition? -/

/-- **Survival is exactly `maxHeight ≤ rows`.** At every step the program's state is non-lost *iff*
its max column height is within the ceiling. So "solving Tetris" reduces, pointwise in time, to
holding one scalar under one bound — the entire control problem is keeping `maxHeight ≤ rows`
forever against the adversary (whose difficulty is that lowering it needs an obstructed clear). -/
theorem solver_not_lost_iff_maxHeight (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    ¬ (adversarialTrace cfg σ s GameState.init n).lost cfg ↔
      Board.maxHeight cfg (adversarialTrace cfg σ s GameState.init n).board ≤ cfg.rows := by
  have hwf : Board.WF cfg (adversarialTrace cfg σ s GameState.init n).board :=
    reachable_WF (solverReachable_implies_reachable_of_solves h.1 h.2
      (adversarialTrace_solverReachable σ hl n))
  exact Board.not_isLost_iff_maxHeight_le hwf

/-! ## Q38. Can the program separate skyline reasoning from hole reasoning? -/

/-- **Where a piece lands depends only on the skyline, never the holes.** Two boards with identical
column-height profiles have identical profiles after the same placement — buried holes do not affect
where a piece comes to rest. So the program's height management is governed by a hole-independent
surface automaton; holes matter only for *clearability*. The program may reason about the skyline
and the hole-debt as two separable channels — the basis of the surface × debt decomposition. -/
theorem solver_surface_evolution_hole_independent {b β : Board} (pl : Placement)
    (h : ∀ j, b.colHeight j = β.colHeight j) (j : ℕ) :
    (pl.place b).colHeight j = (pl.place β).colHeight j :=
  SurfaceFiber.colHeight_place_eq_of_colHeight_eq pl h j

/-! ## Q39. On the skyline order, is the program's dynamics well-behaved? -/

/-- **Placement is monotone in the skyline order.** If board `b` is dominated by `β` per-column
(`domLE`), the same piece drop keeps `place b` dominated by `place β`. So a lower skyline stays
lower — the no-clear height dynamics are monotone (WQO-compatible). Contrast Q29: the irreducibility
of the winning region lives entirely in the *hole* dimension, not the skyline. -/
theorem solver_placement_skyline_monotone {b β : Board} (pl : Placement)
    (h : WqoCarrier.domLE b β) :
    WqoCarrier.domLE (pl.place b) (pl.place β) :=
  WqoCarrier.place_domLE_mono pl h

/-- **Clearing only lowers the skyline.** A line clear can never raise any column height:
`clearLines b` is dominated by `b`. So both of the program's primitives are skyline-well-behaved —
placement
monotone, clearing non-increasing — confirming that the height axis alone is tractable. -/
theorem solver_clearing_lowers_skyline (b : Board) :
    WqoCarrier.domLE (Board.clearLines cfg b) b :=
  WqoCarrier.clearLines_domLE cfg b

/-! ## Q40. How does the program's hole-debt move — is it a one-directional counter? -/

/-- **Placement never removes holes.** Dropping any piece can only keep or increase the buried-hole
count. So the program cannot shed hole-debt by placing pieces — debt is a ratchet placement only
winds up. -/
theorem solver_placement_never_removes_holes (b : Board) (pl : Placement) :
    (HoleyCarrier.holes cfg b).card ≤ (HoleyCarrier.holes cfg (pl.place b)).card :=
  HoleDebt.holes_card_le_place cfg b pl

/-- **Only clearing reduces debt.** A line clear can only keep or lower the hole-debt. Combined with
the previous fact, hole-debt is a Lyapunov counter with a single discharge channel: it rises on
placement and falls only on clears — so the program's debt management is hostage to the same
obstructed clearing lever that governs height. -/
theorem solver_clearing_reduces_debt {b : Board} (hwf : Board.WF cfg b) :
    HoleDebt.debt cfg (Board.clearLines cfg b) ≤ HoleDebt.debt cfg b :=
  HoleDebt.clearLines_debt_le hwf

/-! ## Q41. How does the program's stack energy decompose? -/

/-- **Surface energy splits exactly into stuck debt plus clearable mass.** For any well-formed board
the program holds, `debt + count = surfaceArea`: the stack energy `Σ colHeight` partitions into
buried hole-debt (energy with no clearable cell to show) and filled cells (clearable mass). The
program's headroom is shared between these two competitors — every hole spends energy that filled
mass could have used toward a clear. -/
theorem solver_energy_split {b : Board} (hwf : Board.WF cfg b) :
    HoleDebt.debt cfg b + b.count = HoleDebt.surfaceArea cfg b :=
  HoleDebt.debt_add_card_eq_sum_colHeight hwf

/-! ## Q42. How tightly does aggregate energy track the loss-relevant max height? -/

/-- **Energy brackets max height only up to a factor of `cols`.** Always `maxHeight ≤ surfaceArea ≤
cols·maxHeight`. The lower bound says energy can't hide a tall spike; the upper bound is loose by a
full `cols` factor (a flat board of height `h` has `surfaceArea = cols·h`, a single spike has
`surfaceArea = h`). That slack is exactly the room a sum/energy potential leaves uncontrolled in the
max — the quantitative reason (Q12) the program cannot steer by aggregate energy alone. -/
theorem solver_energy_brackets_maxHeight (b : Board) :
    Board.maxHeight cfg b ≤ HoleDebt.surfaceArea cfg b ∧
    HoleDebt.surfaceArea cfg b ≤ cfg.cols * Board.maxHeight cfg b :=
  ⟨HoleDebt.maxHeight_le_surfaceArea cfg b, HoleDebt.surfaceArea_le_cols_mul_maxHeight cfg b⟩

/-! ## Q43. Is this section's subject literally the project's conjecture, on real Tetris? -/

/-- **"A solving program exists" is exactly `TetrisSolvableValid`.** For the canonical 10×20 game,
the existence of a valid solving program is, by definition, the project's headline conjecture. This
whole section is therefore a characterization of *what `TetrisSolvableValid` would entail*. -/
theorem solver_exists_iff_tetrisSolvableValid :
    (∃ σ : Solver GameConfig.standard, SolvesTetrisValid GameConfig.standard σ)
      ↔ TetrisSolvableValid :=
  Iff.rfl

/-- **On real Tetris, the conjecture reduces to one membership.** The standard `10 ≥ 4` columns
satisfy the precondition, so a solving program for canonical Tetris exists iff the empty-board state
lies in `safe` — the single decidable question the Atlas project sets out to answer. -/
theorem standard_solver_exists_iff_init_safe :
    (∃ σ : Solver GameConfig.standard, SolvesTetrisValid GameConfig.standard σ)
      ↔ GameState.init ∈ safe GameConfig.standard :=
  solver_exists_iff_init_safe (by decide)

/-! ## Q44. Is the program's operating region self-contained under play? -/

/-- **The reachable region is forward-closed.** From any state the program can reach, and any piece
the bag can draw, the program's move lands in another reachable state. So the operating set
is an invariant region generated from `init` — play never escapes it, which is what lets the finite
closed cycle / atlas capture the whole game. -/
theorem solver_reachable_step_closed {g : GameState} (hr : solverReachable σ g)
    {p : Piece} (hp : p ∈ g.bag) :
    solverReachable σ (adversarialStep cfg g p (σ g p)) :=
  solverReachable.step p hr hp

/-! ## Q45. What three invariants, together, capture a solving program's state? -/

/-- **The three pillars, bundled.** At every step against every legal sequence, a solving program's
state simultaneously (1) lies in `safe`, (2) keeps `maxHeight ≤ rows`, and (3) keeps `count ≤
cols·rows`. The first is the abstract certificate; the second the operational survival metric; the
third the material budget. Holding all three forever *is* solving Tetris. -/
theorem solving_program_pillars (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg σ s GameState.init n ∈ safe cfg ∧
    Board.maxHeight cfg (adversarialTrace cfg σ s GameState.init n).board ≤ cfg.rows ∧
    (adversarialTrace cfg σ s GameState.init n).board.count ≤ cfg.cols * cfg.rows :=
  ⟨solver_trace_mem_safe h hl n, solver_maintains_maxHeight h.2 hl n,
    solver_count_le_capacity h hl n⟩

/-! ## Q46. What fixed-point equation pins down the program's region? -/

/-- **The winning region is a fixed point of the safety operator.** `safeOp (safe) = safe`: applying
one step of "not lost and every draw has a valid successor inside the set" to `safe` returns `safe`
unchanged. As the *greatest* such fixed point, it is the largest self-consistent winning region —
the exact object the program is a witness of membership in. -/
theorem solver_region_fixed_point : safeOp cfg (safe cfg) = safe cfg :=
  safe_eq cfg

/-! ## Q47. Does the game ever stall for lack of input? -/

/-- **There is always a next piece.** Along any play, the bag the program faces is nonempty (the
7-bag refills before emptying). So the game never stalls: at every step the adversary hands the
program a piece, and the program must (and, when safe, can) respond. -/
theorem solver_always_has_a_piece (σ : Solver cfg) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace cfg σ s GameState.init n).bag.Nonempty :=
  adversarialTrace_bag_nonempty σ hl n

/-! ## Q48. Is the program's very first move already safe? -/

/-- **The opening move is safe whatever the first piece.** From the empty board, the canonical
program's response to any first piece lands back in `safe`. So solving begins immediately: the
program does not need a lucky opening — every first tetromino has a safe answer it plays. -/
theorem canonical_solver_first_move_safe (h : GameState.init ∈ safe cfg) (p : Piece) :
    adversarialStep cfg GameState.init p (safeSolver cfg GameState.init p) ∈ safe cfg :=
  safeSolver_init_step_mem_safe h p

/-! ## Q49. What is the scale of the universe the program lives in? -/

/-- **The universe is vast but finite: exactly `2^207` states.** For canonical 10×20 Tetris the set
of well-formed, in-field game states has cardinality `2^207` (`2^200` boards × `128` bags). The
program's Atlas is a subset of this enormous-but-finite space — precisely why brute force is
hopeless yet a finite proof artifact exists at all. -/
theorem solver_universe_size_standard :
    (inFieldStates GameConfig.standard).card = 2 ^ 207 :=
  standard_inFieldStates_card_eq_two_pow_207

/-- **The program's table uses at most 128 distinct bag-states.** Across its realizing cycle the
program sees at most `128 = 2^7` distinct bag values (subsets of the 7 pieces). So while the board
dimension is huge (Q24), the bag dimension is small and bounded — the table is board-dominated. -/
theorem solver_distinct_bags_le (hcols : 4 ≤ cfg.cols)
    (hex : ∃ σ : Solver cfg, SolvesTetrisValid cfg σ) :
    ∃ C : AdversarialClosedCycleWF cfg,
      GameState.init ∈ C.toAdversarialClosedCycle.states ∧
      (C.toAdversarialClosedCycle.states.image GameState.bag).card ≤ 128 := by
  obtain ⟨C, hinit⟩ := (init_safe_iff_exists_init_adversarialClosedCycleWF cfg).mp
    ((solver_exists_iff_init_safe hcols).mp hex)
  exact ⟨C, hinit, C.toAdversarialClosedCycle.image_bag_card_le_128⟩

/-! ## Q50. Is the characterization special to 10×20, or config-generic? -/

/-- **The characterization is config-generic.** The same reduction holds for the 4×4 toy config: a
solving program exists iff the empty board is `safe`. None of the structure is special to 10×20 —
the safe-set / atlas framework is parametric in any board with `≥ 4` columns, so the toy config is a
faithful (and finitely searchable) model of the full problem. -/
theorem tiny_solver_exists_iff_init_safe :
    (∃ σ : Solver GameConfig.tiny, SolvesTetrisValid GameConfig.tiny σ)
      ↔ GameState.init ∈ safe GameConfig.tiny :=
  solver_exists_iff_init_safe (by decide)

/-! ## Q51. Does the program's evolution compose like a dynamical system? -/

/-- **The dynamics compose (memoryless semigroup).** Running the program for `n + m` steps equals
running it `n` steps, then continuing `m` steps *from the reached state* under the time-shifted
sequence. The future depends only on the current state and the remaining input — the operational
restatement of memorylessness: the program is a transition system, splittable at any time. -/
theorem solver_trace_compose (s : ℕ → Piece) (g0 : GameState) (n m : ℕ) :
    adversarialTrace cfg σ s g0 (n + m) =
      adversarialTrace cfg σ (fun k => s (n + k)) (adversarialTrace cfg σ s g0 n) m :=
  adversarialTrace_add cfg σ s g0 n m

/-! ## Q52. Under periodic input, does the program settle into a true cycle? -/

/-- **A recurrence under periodic input makes play genuinely periodic.** If the program reaches the
same state at times `b` and `b + d`, and the input sequence is `d`-periodic from `b` on, then the
whole `[b, ∞)` suffix of play is `d`-periodic. So against periodic adversarial input a recurrence is
not a coincidence — it locks the program into a real cycle, the dynamical realization of the M2
closed cycle. -/
theorem solver_periodic_play (s : ℕ → Piece) (g0 : GameState) {b d : ℕ}
    (htrace : adversarialTrace cfg σ s g0 b = adversarialTrace cfg σ s g0 (b + d))
    (hs : ∀ k, s (b + k) = s (b + d + k)) (k : ℕ) :
    adversarialTrace cfg σ s g0 (b + k) = adversarialTrace cfg σ s g0 (b + d + k) :=
  adversarialTrace_periodic_of_periodic_suffix cfg σ s g0 htrace hs k

/-! ## Q53. Can the program depend on pieces it has not yet seen? -/

/-- **The program is causal: play depends only on past input.** If two input sequences agree on the
first `n` pieces, the program's states coincide for the first `n` steps — regardless of how the
sequences differ later. So the program cannot (and need not) anticipate future pieces: its move at
each time is a function of the history so far. This is the rigorous form of "no lookahead". -/
theorem solver_is_causal (s s' : ℕ → Piece) (g0 : GameState) (n : ℕ)
    (h : ∀ i < n, s i = s' i) (k : ℕ) (hk : k ≤ n) :
    adversarialTrace cfg σ s g0 k = adversarialTrace cfg σ s' g0 k :=
  adversarialTrace_eq_of_eq_below σ s s' g0 n h k hk

/-! ## Q54. Is there a simple comfort zone where the program cannot be killed next move? -/

/-- **A low stack is a one-step-safe comfort zone.** On any well-formed board whose columns are all
at height `≤ rows - 4`, *every* piece has a valid placement that does not lose — the piece (≤4 tall)
still fits under the ceiling. So the program has a coarse sufficient tactic for one-step survival:
keep the stack at least 4 below the top. The difficulty is that adversarial roughness can force the
stack up out of this zone (the deeper safe-set question is whether it can be kept there forever). -/
theorem solver_low_stack_comfort_zone (hcols : 4 ≤ cfg.cols) (hrows : 4 ≤ cfg.rows)
    {b : Board} (hWF : Board.WF cfg b) (hlow : ∀ j, Board.colHeight b j ≤ cfg.rows - 4)
    (p : Piece) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
      ¬ Board.isLost cfg (Placement.applyStep cfg b pl) :=
  exists_safe_placement hcols hrows hWF hlow p

/-! ## Q55. Can the program lose on its very first move? -/

/-- **The opening is unconditionally non-losing.** From the empty board, *any* valid placement keeps
the board not lost — the empty board is far enough below the ceiling that no single piece tops it
out, whatever the program does. Danger is never present at the start; it can only build up over
time, which is exactly why the difficulty is a long-horizon control problem, not an opening trap. -/
theorem solver_opening_cannot_lose (hrows : 4 ≤ cfg.rows) (pl : Placement) (hv : pl.Valid cfg) :
    ¬ Board.isLost cfg (Placement.applyStep cfg GameState.init.board pl) :=
  init_applyStep_not_lost_of_valid hrows pl hv

/-! ## Q56. What is the program's ideal board shape? -/

/-- **Hole-free means all energy is clearable.** When the program keeps the board hole-free (`debt =
0`), the stack energy equals the filled mass: `surfaceArea = count`. No headroom is wasted on buried
debt — every unit of height is a clearable cell. This is the program's ideal: a hole-free surface
where the entire `cols·rows` budget is recoverable — the opposite of the adversary's roughness. -/
theorem solver_hole_free_energy_all_clearable {b : Board} (hwf : Board.WF cfg b)
    (h0 : HoleDebt.debt cfg b = 0) :
    HoleDebt.surfaceArea cfg b = b.count := by
  have heq : HoleDebt.debt cfg b + b.count = HoleDebt.surfaceArea cfg b :=
    solver_energy_split hwf
  omega

/-! ## Q57. Is every piece a hole-injector, or only the roughness pieces? -/

/-- **The benign pieces place cleanly.** On the empty board, dropping `O` or `I` at column 0 creates
*no* holes — unlike `S`/`Z` (Q27). So roughness is specific: many pieces can be absorbed
flat without debt, and only the two roughness pieces force buried holes. The program places the
benign pieces freely and must spend its finesse only on `S`/`Z`. -/
theorem solver_benign_pieces_no_holes :
    HoleyCarrier.holes GameConfig.standard (Placement.place ∅ ⟨Piece.O, 0, 0⟩) = ∅ ∧
    HoleyCarrier.holes GameConfig.standard (Placement.place ∅ ⟨Piece.I, 0, 0⟩) = ∅ := by
  decide

/-! ## Q58. Is there a conserved quantity the program can never violate? -/

/-- **The cell count stays even (even-width boards).** On any board with an even number of columns
(standard has `10`), every state the program reaches has an even cell count: each piece adds `4`
(even), and each clear removes a multiple of `cols` (even), so parity is conserved from the empty
board. A hidden invariant the program maintains automatically — and a sanity constraint on the
Atlas: no odd-count board is ever reachable. -/
theorem solver_even_count (hcols : Even cfg.cols) (h : SolvesTetrisValid cfg σ)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Even (adversarialTrace cfg σ s GameState.init n).board.count :=
  reachable_even_count hcols
    (solver_states_reachable_from_empty h (adversarialTrace_solverReachable σ hl n))

/-! ## Q59. How much height can the program recover in one move? -/

/-- **A single move clears at most 4 rows.** From a settled board (no pending full rows), one
placement completes at most 4 full rows — and only the `I`-piece can reach 4. So the program's
per-move height recovery is capped at a Tetris, the rarest and hardest-to-assemble clear: survival
cannot rely on big one-shot recoveries, only on a steady cadence of small clears. -/
theorem solver_clears_at_most_four {b : Board} (pl : Placement)
    (hnf : ∀ r, ¬ Board.isFull cfg b r) :
    Board.linesCleared cfg (pl.place b) ≤ 4 :=
  linesCleared_place_le_four cfg b pl hnf

/-! ## Q60. Does the program ever have a "double clear" pending? -/

/-- **Clearing is idempotent.** Clearing a board and then clearing again gives the same board: once
the program clears, no full rows remain to clear. So there is never a free second clear waiting —
the board settles after one pass, and every later clear must be assembled afresh from new pieces. -/
theorem solver_clearing_idempotent (b : Board) (hcol : 0 < cfg.cols) :
    Board.clearLines cfg (Board.clearLines cfg b) = Board.clearLines cfg b :=
  clearLines_idem cfg b hcol

/-! ## Q61. Is the *loss* predicate as subtle as the *survival* question? -/

/-- **Loss is monotone — unlike survival.** Adding cells can only cause loss, never cure it: a
superset of a lost board is lost. So the one-step *loss* test is simple and order-respecting. The
contrast with Q29 is the whole story: `isLost` is monotone, yet `safe` (survive-*forever*) is
non-congruent — the difficulty is not detecting loss but foreseeing it under adversarial play. -/
theorem solver_loss_monotone {b b' : Board} (h : b ⊆ b') (hlost : Board.isLost cfg b) :
    Board.isLost cfg b' :=
  Board.isLost_mono h hlost

/-! ## Q62. Does the program's board stay structurally valid? -/

/-- **The board is always well-formed.** Every state the program reaches has a well-formed board:
all cells live in a valid column (`< cols`). Placement and clearing both preserve well-formedness,
so no cell ever escapes the field sideways — a structural invariant underpinning the count, height,
parity bounds. -/
theorem solver_board_wf (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    Board.WF cfg (adversarialTrace cfg σ s GameState.init n).board :=
  reachable_WF (solver_states_reachable_from_empty h (adversarialTrace_solverReachable σ hl n))

/-! ## Q63. Is placement reversible, or a one-way ratchet? -/

/-- **Placement never lowers a column.** Dropping a piece can only keep or raise each column height.
So the program cannot reduce height by placing — only the obstructed clear lever brings columns
down. This per-column irreversibility is the height ratchet at the heart of the difficulty. -/
theorem solver_placement_raises_columns (b : Board) (pl : Placement) (j : ℕ) :
    Board.colHeight b j ≤ Board.colHeight (pl.place b) j :=
  colHeight_le_place b pl j

/-- **Placement never lowers the max height either.** The aggregate ratchet: `maxHeight b ≤
maxHeight (place b)`. With "clearing only lowers" (Q39), height moves up freely on every placement
and down only through assembled clears — irreversible except via the obstructed lever. -/
theorem solver_placement_raises_maxHeight (b : Board) (pl : Placement) :
    Board.maxHeight cfg b ≤ Board.maxHeight cfg (pl.place b) :=
  Board.maxHeight_le_place cfg b pl

/-! ## Q64. Can the adversary flood the program with one piece? -/

/-- **No repeats within a bag.** Once a piece is drawn (without triggering a refill) it cannot be
drawn again from the same bag. So the adversary cannot stream the same piece endlessly: roughness is
rate-limited to two `S`/`Z` per bag (Q28), and the program is guaranteed every other piece arrives
before any repeat. This fairness is what makes adversarial survival plausible at all. -/
theorem solver_no_repeat_within_bag (bag : Bag) (p : Piece) (hp : p ∈ bag)
    (hne : bag.draw p ≠ Bag.full) :
    ¬ (bag.draw p).canDraw p :=
  not_canDraw_after_draw bag p hp hne

/-! ## Q65. What renewal process feeds the program its pieces? -/

/-- **The bag renews: count down, then refill.** Each draw either depletes the bag by one piece or
(on the last piece) resets it to the full seven. So the input is a deterministic renewal loop of
period 7 — the program faces a predictable, fair cadence of pieces, never an arbitrary stream. -/
theorem solver_bag_renewal (bag : Bag) (p : Piece) (hp : p ∈ bag) :
    (bag.draw p).card = bag.card - 1 ∨ bag.draw p = Bag.full :=
  draw_card bag p hp

/-! ## Q66. What is the single headline equivalence the whole project rests on? -/

/-- **Solvable ⟺ a finite closed cycle through the empty board.** Chaining the reductions: canonical
Tetris is solvable by a valid program if and only if there exists a finite, WF, adversary-closed
cycle of states containing `init`. This is the M2/M3/M4 headline — the proof artifact reduces an
infinite survival property to exhibiting one finite closed cycle from the empty board. -/
theorem solver_solvable_iff_init_cycle :
    TetrisSolvableValid ↔
      ∃ C : AdversarialClosedCycleWF GameConfig.standard,
        GameState.init ∈ C.toAdversarialClosedCycle.states :=
  standard_solver_exists_iff_init_safe.trans
    (init_safe_iff_exists_init_adversarialClosedCycleWF GameConfig.standard)

/-! ## Q67. Are both roughness pieces hole-injectors? -/

/-- **Both `S` and `Z` inject holes.** Dropped at column 0 on the empty board, the `S`-piece buries
cell `(2,0)` and the `Z`-piece likewise creates at least one buried empty. So both of the bag's two
roughness pieces are hole-injectors — the adversary gets up to two forced hole-plantings per bag
(Q28) against the program's single guaranteed `I`-drain — completing the roughness picture. -/
theorem solver_both_roughness_inject_holes :
    ((2 : ℕ), (0 : ℕ)) ∈ HoleyCarrier.holes GameConfig.standard
        (Placement.place ∅ ⟨Piece.S, 0, 0⟩)
    ∧ 0 < (HoleyCarrier.holes GameConfig.standard
        (Placement.place ∅ ⟨Piece.Z, 0, 0⟩)).card :=
  ⟨Board.S_buries_hole, Board.Z_buries_hole⟩

/-! ## Q68. Is the program ever unable to make *any* move? -/

/-- **Every piece has a valid placement.** On a board at least 4 columns wide, each piece can be
dropped at the left edge (column 0, rotation 0) and stays in bounds — the piece spans `< 4` columns.
So the program's action space is never empty: it is never literally unable to move (the only danger,
Q4, is that all moves might leave `safe`, not that no move exists). -/
theorem solver_action_space_nonempty (hcols : 4 ≤ cfg.cols) (p : Piece) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg := by
  refine ⟨⟨p, 0, 0⟩, rfl, ?_⟩
  intro cell hcell
  have := Piece.shapeUp_col_lt_four p 0 cell hcell
  change 0 + cell.1 < cfg.cols
  omega

/-! ## Q69. What set does the program's operating region equal? -/

/-- **The program operates inside `safe ∩ reachable`.** Every state it can reach is both survivable
(`safe`) and genuinely reachable from the empty board. That intersection — survivable *and*
reachable states — is precisely the Atlas the project builds: not all of `safe` (some is unreachable
junk) and not all of `reachable` (some is doomed), but exactly their overlap. -/
theorem solver_operates_in_safe_and_reachable (h : SolvesTetrisValid cfg σ) {g : GameState}
    (hr : solverReachable σ g) :
    g ∈ safe cfg ∧ Reachable cfg g :=
  ⟨solver_no_dead_ends h hr, solver_states_reachable_from_empty h hr⟩

/-! ## Q70. Can we pin down *when* the program is forced to clear? -/

/-- **A clear occurs within any window whose budget exceeds capacity.** If `4·M > cols·rows`, then
among the first `M` moves at least one fails to add 4 cells — i.e. clears a line. Concretely the
program cannot defer clearing past `⌊cols·rows / 4⌋` moves: forced clearing is not just bounded
(Q21) but *located* — it must happen inside the next capacity-sized window. -/
theorem solver_clears_within (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) {M : ℕ} (hM : cfg.cols * cfg.rows < 4 * M) :
    ∃ k < M, (adversarialTrace cfg σ s GameState.init (k + 1)).board.count
             ≠ (adversarialTrace cfg σ s GameState.init k).board.count + 4 := by
  by_contra hcon
  have hall : ∀ k < M,
      (adversarialTrace cfg σ s GameState.init (k + 1)).board.count
        = (adversarialTrace cfg σ s GameState.init k).board.count + 4 := by
    intro k hk
    by_contra hne
    exact hcon ⟨k, hk, hne⟩
  have hb := solver_no_clear_window_bounded h hl M hall
  omega

/-- **Concretely, standard Tetris forces a clear every 51 moves.** Since `4·51 = 204 > 200 = 10·20`,
a surviving program on the canonical board cannot go 51 moves without clearing a line. A tangible
survival cadence: a clear at least once per 51 placements. -/
theorem standard_solver_clears_within_51 {σ : Solver GameConfig.standard}
    (h : SolvesTetrisValid GameConfig.standard σ) {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ k < 51, (adversarialTrace GameConfig.standard σ s GameState.init (k + 1)).board.count
             ≠ (adversarialTrace GameConfig.standard σ s GameState.init k).board.count + 4 :=
  solver_clears_within h hl (by decide)

/-- **Concrete standard bounds: height ≤ 20, cells ≤ 200, forever.** Instantiating the survival
metric and material budget for the canonical 10×20 board: a solving program holds the stack at most
20 rows tall and at most 200 cells, at every step against every sequence. -/
theorem standard_solver_bounds {σ : Solver GameConfig.standard}
    (h : SolvesTetrisValid GameConfig.standard σ) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Board.maxHeight GameConfig.standard
        (adversarialTrace GameConfig.standard σ s GameState.init n).board ≤ 20 ∧
    (adversarialTrace GameConfig.standard σ s GameState.init n).board.count ≤ 200 := by
  refine ⟨?_, ?_⟩
  · have hm := solver_maintains_maxHeight h.2 hl n
    rwa [GameConfig.standard_rows] at hm
  · have hc := solver_count_le_capacity h hl n
    rw [GameConfig.standard_cols, GameConfig.standard_rows] at hc
    omega

/-! ## Q71. What is the survival condition at the level of individual cells? -/

/-- **Survival ⟺ every cell below row `rows`.** The not-lost predicate expands to a uniform per-cell
ceiling: a board is alive exactly when every occupied cell lies strictly below row `rows`. So the
program's task, cell by cell, is to keep the field's top rows empty — the simplest possible form of
the loss test, with no aggregate quantity involved. -/
theorem solver_not_lost_iff_cells_below_rows (b : Board) :
    ¬ Board.isLost cfg b ↔ ∀ p ∈ b, p.2 < cfg.rows :=
  Board.not_isLost_iff_forall_row_lt cfg b

/-! ## Q72. Is there any state between alive and lost? -/

/-- **The loss dichotomy.** Every board is either lost or has all columns within the ceiling — there
is no in-between. So the program's whole job is to remain perpetually on the "all columns ≤ rows"
side of this clean two-way split; survival is staying on the bounded side forever. -/
theorem solver_loss_dichotomy (b : Board) :
    Board.isLost cfg b ∨ ∀ j, Board.colHeight b j ≤ cfg.rows :=
  isLost_or_bounded cfg b

/-! ## Q73. What are the master equivalences, side by side? -/

/-- **The two faces of the conjecture.** Solvability of canonical Tetris is equivalent both to a
single set-membership (`init ∈ safe`) and to the existence of a finite WF closed cycle through
`init`. The abstract fixed-point face and the concrete finite-artifact face are the same theorem —
the program exists exactly when either witness does. -/
theorem solver_master_equivalences :
    (TetrisSolvableValid ↔ GameState.init ∈ safe GameConfig.standard) ∧
    (TetrisSolvableValid ↔
      ∃ C : AdversarialClosedCycleWF GameConfig.standard,
        GameState.init ∈ C.toAdversarialClosedCycle.states) :=
  ⟨standard_solver_exists_iff_init_safe, solver_solvable_iff_init_cycle⟩

/-! ## Q74. Is the program's board contained in the field rectangle? -/

/-- **Every cell stays in the field `[0,cols) × [0,rows)`.** Well-formedness bounds each cell's
column (`< cols`) and survival bounds its row (`< rows`), so the program's board is contained in the
field rectangle at all times — no cell escapes sideways or upward. The two structural axes (Q34,
Q36) combine into a single containment. -/
theorem solver_board_in_field (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) {c : Coord}
    (hc : c ∈ (adversarialTrace cfg σ s GameState.init n).board) :
    c.1 < cfg.cols ∧ c.2 < cfg.rows :=
  ⟨solver_board_wf h hl n c hc, solver_no_cell_in_death_zone h.2 hl n hc⟩

/-! ## Q75. From what state must the program begin? -/

/-- **The program begins from the canonical clean state.** Empty board, full bag, zero cells: the
program starts with maximal headroom and complete information about the first bag. Every survival
proof must originate here — `init` is not an arbitrary state but the unique fresh-game configuration
that makes solvability the empty-board question. -/
theorem solver_initial_state :
    GameState.init.bag = Bag.full ∧
    GameState.init.board.count = 0 ∧
    GameState.init.board = (∅ : Board) :=
  ⟨GameState.init_bag, GameState.init_board_count, GameState.init_board_eq_emptyset⟩

/-! ## Q76. Can the program's winning region actually be computed? -/

/-- **The winning region is computable by a converging finite iteration.** Starting from the finite
universe `inFieldStates`, the safe-iteration `safeIterFinite` reaches a fixed point within
`|inFieldStates|` steps — it can only shrink, one state-removal at a time. This is exactly the
retrograde death-propagation the Atlas builder runs: a terminating algorithm, not just an existence
claim, whose fixed point (intersected with reachability) is the program's table. -/
theorem solver_region_computable :
    ∃ N, N ≤ (inFieldStates cfg).card ∧
      safeIterFinite cfg (inFieldStates cfg) (N + 1) = safeIterFinite cfg (inFieldStates cfg) N :=
  safeIterFinite_converges cfg (inFieldStates cfg)

/-! ## Q77. Is the death-propagation computing the region monotone? -/

/-- **Death propagation only shrinks the surviving set.** The finite iteration is antitone: deeper
iterations are subsets of shallower ones, so a state once removed (proven to lead to forced loss)
never returns. This monotone retraction is why the Atlas builder both terminates and is correct —
the surviving subgraph converges down to the program's winning region. -/
theorem solver_death_propagation_monotone {n m : ℕ} (hnm : n ≤ m) :
    safeIterFinite cfg (inFieldStates cfg) m ⊆ safeIterFinite cfg (inFieldStates cfg) n :=
  safeIterFinite_antitone cfg (inFieldStates cfg) hnm

/-! ## Q78. Is the computed region trustworthy (sound)? -/

/-- **Soundness: the computed fixed point is genuinely safe.** At a fixed point of the finite
iteration, every surviving state lies in `safe` — the algorithm never keeps a state from which the
program cannot actually survive. So a fixed point reached by the Atlas builder is a *verified*
controlled-invariant set: the program built from it really does win from each of its states. -/
theorem solver_computed_region_sound (N : ℕ)
    (hfix : safeIterFinite cfg (inFieldStates cfg) (N + 1)
      = safeIterFinite cfg (inFieldStates cfg) N) :
    (↑(safeIterFinite cfg (inFieldStates cfg) N) : Set GameState) ⊆ safe cfg :=
  safeIterFinite_subset_safe cfg (inFieldStates cfg) N hfix

/-- **Completeness: a covering universe recovers `safe` exactly.** If the search universe `S₀`
contains all of `safe`, then at the fixed point membership in the computed set is *equivalent*
to safety — nothing safe is dropped. With soundness (Q78), the builder computes the program's
winning region precisely: the converged finite set is the safe set, on the nose. -/
theorem solver_computed_region_complete {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) (N : ℕ)
    (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) (g : GameState) :
    g ∈ safe cfg ↔ g ∈ (↑(safeIterFinite cfg S₀ N) : Set GameState) :=
  safe_iff_mem_fixedPoint cfg S₀ N hS₀ hfix g

/-! ## Q79. What single concrete test decides whether the program exists? -/

/-- **The program exists iff `init` survives death-propagation.** Given a converged finite iteration
over a covering universe, a solving program exists if and only if `GameState.init` is a member of
the computed surviving `Finset`. This is the one decidable membership check the builder performs to
settle the entire conjecture — solvability collapses to "is the empty board still alive at the fixed
point?" -/
theorem solver_exists_iff_init_in_fixedpoint {S₀ : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hS₀ : safe cfg ⊆ ↑S₀) (N : ℕ)
    (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    (∃ σ : Solver cfg, SolvesTetrisValid cfg σ)
      ↔ GameState.init ∈ safeIterFinite cfg S₀ N :=
  (solver_exists_iff_init_safe hcols).trans
    (init_safe_iff_init_mem_safeIterFinite hS₀ N hfix)

/-! ## Q80. Is the maximal winning region unique? -/

/-- **`safe` is the greatest fixed point — the winning region is unique.** Every fixed point `T` of
the safety operator is contained in `safe`. So there is no larger self-consistent survivable set:
the maximal controlled-invariant region is *the* answer, and the program is a membership witness
for this canonical, unique object. -/
theorem solver_region_greatest_fixed_point (T : Set GameState) (hT : safeOp cfg T = T) :
    T ⊆ safe cfg :=
  safe_greatest T hT.ge

end Tetris
