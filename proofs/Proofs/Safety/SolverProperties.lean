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

end Tetris
