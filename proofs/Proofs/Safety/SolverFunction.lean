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

/-! ## Part 2 — The genuine degrees of freedom and the bounded range -/

/-- **The output is determined by `(rot, col)`.** Two outputs of a valid solver for the same piece
that agree on rotation and column are *equal* — the piece field, being forced, adds nothing. So the
function's real codomain is the two-number space `(rotation, column)`, not the full placement. -/
theorem solver_eq_of_rotcol (hv : ValidSolver cfg σ) {g₁ g₂ : GameState} {p : Piece}
    (hp₁ : p ∈ g₁.bag) (hp₂ : p ∈ g₂.bag)
    (hr : (σ g₁ p).rot = (σ g₂ p).rot) (hc : (σ g₁ p).col = (σ g₂ p).col) :
    σ g₁ p = σ g₂ p := by
  show Placement.mk (σ g₁ p).piece (σ g₁ p).rot (σ g₁ p).col
      = Placement.mk (σ g₂ p).piece (σ g₂ p).rot (σ g₂ p).col
  rw [Placement.mk.injEq]
  exact ⟨(hv g₁ p hp₁).1.trans (hv g₂ p hp₂).1.symm, hr, hc⟩

/-- **The chosen column is in range: `col < cols`.** Every output column is a real board column. So
the column coordinate of the output is bounded by `cols`, and (with the ≤4 rotations) the per-piece
output ranges over at most `4·cols` values — a tiny, fixed space. -/
theorem solver_col_lt_cols (hv : ValidSolver cfg σ) {g : GameState}
    {p : Piece} (hp : p ∈ g.bag) :
    (σ g p).col < cfg.cols := by
  have h := solver_output_in_action_set hv hp
  unfold Placement.allValidFor at h
  rw [Finset.mem_filter, Finset.mem_image] at h
  obtain ⟨⟨rc, hrc, hrceq⟩, _⟩ := h
  rw [Finset.mem_product] at hrc
  have hcol : (σ g p).col = rc.1 := by rw [← hrceq]
  rw [hcol]
  exact Finset.mem_range.mp hrc.1

/-- **The whole range lives in one fixed finite set.** Every output of a valid solver — over all
states and all drawable pieces — belongs to `⋃ₚ allValidFor cfg p`, a single `Finset`. The function
takes only finitely many distinct values, bounded by a quantity that does not depend on the (vast)
domain. This is the first compressibility fact: the output space is small and fixed. -/
theorem solver_output_in_total_action_set (hv : ValidSolver cfg σ) {g : GameState}
    {p : Piece} (hp : p ∈ g.bag) :
    σ g p ∈ (Finset.univ : Finset Piece).biUnion (Placement.allValidFor cfg) :=
  Finset.mem_biUnion.mpr ⟨p, Finset.mem_univ p, solver_output_in_action_set hv hp⟩

/-! ## Part 3 — Compressibility: finite range, finite active domain -/

/-- **The function's range is finite.** The set of *all* placements a valid solver ever returns
(over every state and every drawable piece) is a finite set — contained in the fixed finite menu
`⋃ₚ allValidFor cfg p`. However astronomically large the input space, the function emits only
finitely many distinct outputs: as a relation it is a *finite-image* map, hence compressible on the
output side to a bounded table of placements. -/
theorem solver_range_finite (hv : ValidSolver cfg σ) :
    {pl : Placement | ∃ g p, p ∈ g.bag ∧ σ g p = pl}.Finite := by
  apply Set.Finite.subset
    ((Finset.univ : Finset Piece).biUnion (Placement.allValidFor cfg)).finite_toSet
  rintro pl ⟨g, p, hp, rfl⟩
  exact Finset.mem_coe.mpr (solver_output_in_total_action_set hv hp)

/-- **The active domain is finite.** A solving program is only ever *queried* on states it can
reach, and every such state lies in the finite `inFieldStates cfg`. So the survival-relevant part of
the function is a map on `inFieldStates × Piece` — a finite lookup table — even though its type
domain (all `GameState`) is infinite. Together with finite range, the solver compresses to a finite
object: this is exactly the Atlas. -/
theorem solver_active_domain_finite (h : SolvesTetrisValid cfg σ) {g : GameState}
    (hr : solverReachable σ g) :
    g ∈ inFieldStates cfg :=
  reachable_safe_mem_inFieldStates
    (solver_states_reachable_from_empty h hr) (solver_no_dead_ends h hr)

/-! ## Part 4 — Under-determination: the function is a choice from a relation -/

/-- **At each safe state the function chooses from a nonempty set of good moves.** For a safe `g`
and drawable `p`, the placements that are valid *and* keep the state safe form a nonempty set; a
solving function simply *picks one*. So the survival-essential object is the relation `goodMove g p
pl := (valid ∧ leads back to safe)`, and the solver is one **uniformization** (choice function). -/
theorem solver_selects_from_good_moves {g : GameState} (hg : g ∈ safe cfg)
    {p : Piece} (hp : p ∈ g.bag) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧ adversarialStep cfg g p pl ∈ safe cfg :=
  safe_forall_step hg p hp

/-- **Survival does not determine the function (non-uniqueness).** Two *different* solvers `σ₁`,
`σ₂` — which may disagree on the actual placement at every state — *both* survive forever, provided
each keeps every safe state's move inside `safe`. So the function carries strictly more information
than its job requires: the survival-relevant content is the shared safe-closure relation, and
everything beyond that is free. This is compressibility on the *strategy* side — the function is an
arbitrary section of a coarser relation. -/
theorem solvers_agree_on_survival {σ₁ σ₂ : Solver cfg}
    (h₁ : ∀ g ∈ safe cfg, ∀ p ∈ g.bag, adversarialStep cfg g p (σ₁ g p) ∈ safe cfg)
    (h₂ : ∀ g ∈ safe cfg, ∀ p ∈ g.bag, adversarialStep cfg g p (σ₂ g p) ∈ safe cfg)
    (hinit : GameState.init ∈ safe cfg) :
    SolvesTetris cfg σ₁ ∧ SolvesTetris cfg σ₂ :=
  ⟨any_safe_selector_survives h₁ hinit, any_safe_selector_survives h₂ hinit⟩

/-! ## Part 5 — The function genuinely collapses its input space -/

/-- **Pigeonhole: the function must give two states the same move.** Fix a piece `p`. If a set `T`
of states all drawing `p` is larger than the action menu `allValidFor cfg p`, then two distinct
states in `T` receive the *same* placement. So the solver, viewed as `state ↦ placement` for a fixed
piece, cannot be injective once the input set outgrows the (small, fixed) output set — it provably
*loses information*, compressing many distinct boards onto one response. -/
theorem solver_per_piece_noninjective (hv : ValidSolver cfg σ) {p : Piece}
    (T : Finset GameState) (hT : ∀ g ∈ T, p ∈ g.bag)
    (hcard : (Placement.allValidFor cfg p).card < T.card) :
    ∃ g₁ ∈ T, ∃ g₂ ∈ T, g₁ ≠ g₂ ∧ σ g₁ p = σ g₂ p :=
  Finset.exists_ne_map_eq_of_card_lt_of_maps_to hcard
    (fun g hg => solver_output_in_action_set hv (hT g hg))

/-! ## Part 6 — What the function's output *does*, and a closing portrait -/

/-- **The output's skyline-effect ignores buried holes.** If two boards share a surface and the
function returns the same placement on both, the resulting surfaces coincide. So whatever the
function reads to *choose* a move, the move's effect on the skyline is a function of the surface and
the placement alone — the holes beneath the board play no role in where the piece comes to rest. -/
theorem solver_skyline_effect_factors_through_surface {g₁ g₂ : GameState} {p : Piece}
    (hsurf : ∀ j, g₁.board.colHeight j = g₂.board.colHeight j)
    (hpl : σ g₁ p = σ g₂ p) (j : ℕ) :
    ((σ g₁ p).place g₁.board).colHeight j = ((σ g₂ p).place g₂.board).colHeight j := by
  rw [hpl]
  exact SurfaceFiber.colHeight_place_eq_of_colHeight_eq (σ g₂ p) hsurf j

/-- **Portrait of a single output (Parts 1–2 assembled).** Every value a valid solver returns is, at
once: a placement that *announces the input piece*, is *in-bounds/valid*, sits in a *real column*
`< cols`, and belongs to the *fixed finite menu* `⋃ₚ allValidFor cfg p`. Four constraints that
together pin an output down to a tiny `(rotation, column)` choice from a finite set — the function's
output is highly structured and highly compressed. -/
theorem solver_output_portrait (hv : ValidSolver cfg σ) {g : GameState}
    {p : Piece} (hp : p ∈ g.bag) :
    (σ g p).piece = p ∧
    (σ g p).Valid cfg ∧
    (σ g p).col < cfg.cols ∧
    σ g p ∈ (Finset.univ : Finset Piece).biUnion (Placement.allValidFor cfg) :=
  ⟨solver_output_announces_piece hv hp, solver_output_valid hv hp,
   solver_col_lt_cols hv hp, solver_output_in_total_action_set hv hp⟩

/-! ## Part 7 — Quantitative smallness of the output space -/

/-- **The per-piece action menu has at most `4·cols` entries.** `Rotation = Fin 4`, so the valid
placements of a piece form a `Finset` of size `≤ 4·cols`. The function's choice for each piece is
from a tiny, config-explicit set — not the unbounded placement type. -/
theorem card_allValidFor_le (cfg : GameConfig) (p : Piece) :
    (Placement.allValidFor cfg p).card ≤ cfg.cols * 4 := by
  refine le_trans (Finset.card_filter_le _ _) (le_trans Finset.card_image_le ?_)
  simp [Finset.card_product, Finset.card_range, Finset.card_univ, Fintype.card_fin]

/-- **The whole output menu has at most `|Piece|·4·cols` entries.** Summing the per-piece bound over
the seven pieces, the entire range a valid solver can ever produce sits in a `Finset` of size
`≤ |Piece|·(4·cols)`. A fixed, config-explicit ceiling on the function's distinct outputs. -/
theorem card_total_action_set_le (cfg : GameConfig) :
    ((Finset.univ : Finset Piece).biUnion (Placement.allValidFor cfg)).card
      ≤ Fintype.card Piece * (cfg.cols * 4) :=
  calc ((Finset.univ : Finset Piece).biUnion (Placement.allValidFor cfg)).card
      ≤ ∑ p ∈ (Finset.univ : Finset Piece), (Placement.allValidFor cfg p).card :=
        Finset.card_biUnion_le
    _ ≤ ∑ _p ∈ (Finset.univ : Finset Piece), cfg.cols * 4 :=
        Finset.sum_le_sum (fun p _ => card_allValidFor_le cfg p)
    _ = Fintype.card Piece * (cfg.cols * 4) := by
        rw [Finset.sum_const, Finset.card_univ, smul_eq_mul]

/-- **On standard Tetris the function emits at most 280 distinct placements.** With `|Piece| = 7`
and `cols = 10`, the whole output menu has `≤ 7·40 = 280` entries. The input space has `2^207`
states, so as a map it compresses an astronomical domain onto a range of fewer than 300 values. -/
theorem card_total_action_set_standard_le :
    ((Finset.univ : Finset Piece).biUnion
      (Placement.allValidFor GameConfig.standard)).card ≤ 280 := by
  have h := card_total_action_set_le GameConfig.standard
  have hp : Fintype.card Piece = 7 := by decide
  rw [hp, GameConfig.standard_cols] at h
  omega

/-! ## Part 8 — What applying the output does to the board -/

/-- **The output always places exactly four cells.** Applying any output of the function to a board
adds precisely 4 filled cells (a tetromino) before clears: `count (place) = count + 4`. The function
never produces a "partial" move — every output deposits a full 4-cell piece. -/
theorem solver_output_places_four (b : Board) (g : GameState) (p : Piece) :
    ((σ g p).place b).count = b.count + 4 :=
  Placement.count_place b (σ g p)

/-- **The output's dropped set is exactly four cells.** The cells the function's move newly fills
form a set of size 4 — independent of board and choice. The output denotes a 4-cell shape. -/
theorem solver_output_dropped_card (b : Board) (g : GameState) (p : Piece) :
    ((σ g p).dropped b).card = 4 :=
  Placement.card_dropped b (σ g p)

/-- **The move preserves well-formedness.** Applying a valid solver's output to a well-formed board
yields a well-formed board — every resulting cell is still in a real column. The output respects the
field width on every board, because its validity (`col + cell < cols`) is board-independent. -/
theorem solver_move_preserves_wf (hv : ValidSolver cfg σ) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) {b : Board} (hWF : Board.WF cfg b) :
    Board.WF cfg ((σ g p).place b) :=
  Board.WF_place hWF (solver_output_valid hv hp)

/-! ## Part 9 — The canonical function `safeSolver` as a compressed object -/

/-- **The canonical function is constant outside the winning region.** Wherever the state is unsafe
or the piece is not drawable, `safeSolver` returns the fixed default `⟨p, 0, 0⟩`. All of the
function's genuine content lives on `safe ∩ (p ∈ bag)`; on the rest of its infinite domain it is a
single constant per piece — maximally compressed. -/
theorem safeSolver_trivial_outside {g : GameState} {p : Piece}
    (h : ¬ (g ∈ safe cfg ∧ p ∈ g.bag)) :
    safeSolver cfg g p = ⟨p, 0, 0⟩ :=
  safeSolver_eq_trivial_of_not_safe_and_in_bag h

/-- **The canonical function always announces the piece — unconditionally.** Even outside the safe
region, `(safeSolver g p).piece = p`. The piece field is pinned for the canonical solver with no
hypothesis at all. -/
theorem safeSolver_always_announces_piece (g : GameState) (p : Piece) :
    (safeSolver cfg g p).piece = p :=
  safeSolver_piece cfg g p

/-- **The canonical function only emits valid placements.** `safeSolver` is a `ValidSolver`: every
output is in-bounds, on both the safe branch and the default branch. -/
theorem safeSolver_is_valid (hcols : 4 ≤ cfg.cols) : ValidSolver cfg (safeSolver cfg) :=
  safeSolver_validSolver hcols

/-- **On safe states the canonical function's choice preserves survival.** From a safe `g` with
drawable `p`, the step under `safeSolver`'s output lands back in `safe`. The function's *content*
(its non-default values) is exactly a survival-preserving selection on the winning region. -/
theorem safeSolver_choice_stays_safe {g : GameState} (hg : g ∈ safe cfg)
    {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (safeSolver cfg g p) ∈ safe cfg :=
  safeSolver_step_mem_safe hg hp

/-! ## Part 10 — The function is determined only by where it is queried -/

/-- **The trajectory depends only on the function's values at visited states.** If two solvers agree
on every state actually reached along `σ₁`'s play (at the drawn pieces), their entire traces
coincide. So the function's values off the trajectory are irrelevant — the survival behavior is
determined by a (finite, reachable) restriction, and everything else is free data. This is the sharp
form of compressibility: only the *visited* part of the table matters. -/
theorem solver_trace_determined_by_visited (s : ℕ → Piece) {σ₁ σ₂ : Solver cfg} :
    ∀ n, (∀ k < n, σ₁ (adversarialTrace cfg σ₁ s GameState.init k) (s k)
                 = σ₂ (adversarialTrace cfg σ₁ s GameState.init k) (s k)) →
      adversarialTrace cfg σ₁ s GameState.init n
        = adversarialTrace cfg σ₂ s GameState.init n := by
  intro n
  induction n with
  | zero => intro _; simp
  | succ k ih =>
      intro h
      have ihk := ih (fun j hj => h j (Nat.lt_succ_of_lt hj))
      rw [adversarialTrace_succ, adversarialTrace_succ, ← ihk, h k (Nat.lt_succ_self k)]

/-! ## Part 11 — Image bounds: the function compresses regardless of input size -/

/-- **Fixed-piece image lands in the action set.** For a fixed piece, the outputs over any set of
states (all drawing that piece) form a subset of `allValidFor cfg p`. -/
theorem solver_image_per_piece_subset (hv : ValidSolver cfg σ) {p : Piece}
    (T : Finset GameState) (hT : ∀ g ∈ T, p ∈ g.bag) :
    T.image (fun g => σ g p) ⊆ Placement.allValidFor cfg p := by
  intro pl hpl
  rw [Finset.mem_image] at hpl
  obtain ⟨g, hg, rfl⟩ := hpl
  exact solver_output_in_action_set hv (hT g hg)

/-- **Fixed-piece image has at most `4·cols` values — for *any* input size.** However many states
draw a given piece, the function maps them onto at most `4·cols` distinct placements. The output
count is capped by the action set, independent of the (arbitrarily large) input set. -/
theorem solver_image_per_piece_card_le (hv : ValidSolver cfg σ) {p : Piece}
    (T : Finset GameState) (hT : ∀ g ∈ T, p ∈ g.bag) :
    (T.image (fun g => σ g p)).card ≤ cfg.cols * 4 :=
  le_trans (Finset.card_le_card (solver_image_per_piece_subset hv T hT))
    (card_allValidFor_le cfg p)

/-- **The output menus partition by piece.** For distinct pieces the action sets are disjoint: no
placement is valid for two different pieces (the `piece` field discriminates perfectly). So the
function's output range splits cleanly into seven independent per-piece blocks, and the piece of an
output recovers which block — the output carries its own routing. -/
theorem allValidFor_disjoint_of_ne {p p' : Piece} (h : p ≠ p') :
    Disjoint (Placement.allValidFor cfg p) (Placement.allValidFor cfg p') := by
  rw [Finset.disjoint_left]
  intro pl hpl hpl'
  rw [Placement.mem_allValidFor] at hpl hpl'
  exact h (hpl.1.symm.trans hpl'.1)

/-! ## Part 12 — The function induces a deterministic dynamical system -/

/-- The state-transition the function induces for a fixed drawn piece: place the function's choice
and draw `p`. The solver, fixed against one piece, *is* a self-map `GameState → GameState`. -/
def solverStep (cfg : GameConfig) (σ : Solver cfg) (p : Piece) (g : GameState) : GameState :=
  adversarialStep cfg g p (σ g p)

/-- **The play is the orbit of the induced self-map.** Each step of the trace is one application of
`solverStep` for the drawn piece: `trace (n+1) = solverStep (s n) (trace n)`. So the function is a
discrete dynamical system and the game is its orbit from `init` — survival is non-escape of that
orbit from the non-lost region. -/
theorem solver_trace_eq_solverStep (s : ℕ → Piece) (n : ℕ) :
    adversarialTrace cfg σ s GameState.init (n + 1)
      = solverStep cfg σ (s n) (adversarialTrace cfg σ s GameState.init n) := by
  rw [adversarialTrace_succ]; rfl

/-- **For the canonical solver, the winning region is invariant under the induced map.** The map
`solverStep` of `safeSolver`, for any drawable piece, sends `safe` into `safe`. So `safe` is an
invariant set of the induced dynamical system — an orbit started inside it can never leave. -/
theorem safeSolver_solverStep_preserves_safe {g : GameState} (hg : g ∈ safe cfg)
    {p : Piece} (hp : p ∈ g.bag) :
    solverStep cfg (safeSolver cfg) p g ∈ safe cfg :=
  safeSolver_step_mem_safe hg hp

/-! ## Part 13 — The realized moves over a whole play -/

/-- **Every move actually made comes from the finite menu.** Along any legal play, the placement the
function returns at step `n` is in `allValidFor cfg (s n)`. So the realized output sequence —
however long the game runs — lives entirely in the fixed finite menu. -/
theorem solver_play_outputs_in_menu (hv : ValidSolver cfg σ) (s : ℕ → Piece)
    (hl : LegalSequence s) (n : ℕ) :
    σ (adversarialTrace cfg σ s GameState.init n) (s n)
      ∈ Placement.allValidFor cfg (s n) := by
  have hbag : s n ∈ (adversarialTrace cfg σ s GameState.init n).bag := by
    rw [adversarialTrace_bag]; exact hl n
  exact solver_output_in_action_set hv hbag

/-- **The realized-move sequence has finite range.** The infinite sequence of placements produced
along a play takes only finitely many distinct values — its range sits in the finite menu
`⋃ₚ allValidFor cfg p`. The function's whole history of moves is a finite-alphabet sequence. -/
theorem solver_realized_outputs_finite (hv : ValidSolver cfg σ) (s : ℕ → Piece)
    (hl : LegalSequence s) :
    (Set.range fun n => σ (adversarialTrace cfg σ s GameState.init n) (s n)).Finite := by
  apply Set.Finite.subset
    ((Finset.univ : Finset Piece).biUnion (Placement.allValidFor cfg)).finite_toSet
  rintro pl ⟨n, rfl⟩
  exact Finset.mem_coe.mpr (Finset.mem_biUnion.mpr
    ⟨s n, Finset.mem_univ _, solver_play_outputs_in_menu hv s hl n⟩)

/-! ## Part 14 — The function as a finite table -/

/-- **The relevant table has a finite, exact domain.** A solving function only matters on the finite
product `inFieldStates × Piece`, whose size is exactly `|inFieldStates|·|Piece|`. So the function
compresses to a finite lookup with an explicitly counted input domain. -/
theorem solver_table_domain_card :
    ((inFieldStates cfg) ×ˢ (Finset.univ : Finset Piece)).card
      = (inFieldStates cfg).card * Fintype.card Piece := by
  rw [Finset.card_product, Finset.card_univ]

/-! ## Part 15 — Geometric shape of what the output places -/

/-- **The move is purely additive before clears.** Placing the output is monotone in the board: the
original board is a subset of the placed board (`b ⊆ place b`). The output only *adds* its four
cells; the only way cells leave is the subsequent clear phase. -/
theorem solver_move_superset (b : Board) (g : GameState) (p : Piece) :
    b ⊆ (σ g p).place b := by
  rw [Placement.place_eq_union_dropped]
  exact Finset.subset_union_left

/-- **The output's four new cells are disjoint from the board.** The dropped tetromino never
overlaps existing cells — it rests on top, adding four genuinely new cells. -/
theorem solver_dropped_disjoint (b : Board) (g : GameState) (p : Piece) :
    Disjoint b ((σ g p).dropped b) :=
  (Placement.dropped_disjoint b (σ g p)).symm

/-- **Each state's response table has at most `|bag|` entries.** The function's outputs for a state,
over its drawable pieces, number at most the bag size (`≤ 7`) — a tiny per-state slice. -/
theorem solver_response_table_card_le (g : GameState) :
    (g.bag.image (fun p => σ g p)).card ≤ g.bag.card :=
  Finset.card_image_le

/-- **The output rotation is one of four.** `rot : Fin 4`, so the rotation component is always `< 4`
— with the column `< cols`, the genuine output is a point in `Fin 4 × range cols`. -/
theorem solver_rot_lt_four (g : GameState) (p : Piece) : ((σ g p).rot : ℕ) < 4 :=
  (σ g p).rot.isLt

/-- **The function is only ever queried at `p ∈ g.bag`.** Along any legal play the drawn piece lies
in the current bag, so the function is consulted only on live `(state, piece)` pairs; its values at
off-bag pieces are dead data, never reached. -/
theorem solver_queried_in_bag (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    s n ∈ (adversarialTrace cfg σ s GameState.init n).bag := by
  rw [adversarialTrace_bag]; exact hl n

/-- **The decision is a point in the `Rotation × range cols` grid.** The genuine output `(rot, col)`
of a valid solver is a point of the finite `4·cols` grid — the choice is one grid coordinate. -/
theorem solver_output_in_grid (hv : ValidSolver cfg σ) {g : GameState}
    {p : Piece} (hp : p ∈ g.bag) :
    ((σ g p).rot, (σ g p).col)
      ∈ (Finset.univ : Finset Rotation) ×ˢ Finset.range cfg.cols := by
  rw [Finset.mem_product]
  exact ⟨Finset.mem_univ _, Finset.mem_range.mpr (solver_col_lt_cols hv hp)⟩

/-! ## Part 16 — Semantic guarantees of the output -/

/-- **The function's opening move never loses.** From the empty board, applying any valid output for
any first piece keeps the board not lost. -/
theorem solver_opening_move_safe (hv : ValidSolver cfg σ) (hrows : 4 ≤ cfg.rows)
    {p : Piece} (hp : p ∈ GameState.init.bag) :
    ¬ Board.isLost cfg ((σ GameState.init p).applyStep cfg GameState.init.board) :=
  solver_opening_cannot_lose hrows (σ GameState.init p) (solver_output_valid hv hp)

/-- **On a low stack the output never loses.** If the board's columns are all `≤ rows - 4`, applying
any valid output keeps it not lost — the output is safe in the comfort zone. -/
theorem solver_lowstack_move_safe (hv : ValidSolver cfg σ) (hrows : 4 ≤ cfg.rows)
    {g : GameState} {p : Piece} (hp : p ∈ g.bag) {b : Board} (hWF : Board.WF cfg b)
    (hlow : ∀ j, Board.colHeight b j ≤ cfg.rows - 4) :
    ¬ Board.isLost cfg ((σ g p).applyStep cfg b) :=
  Tetris.low_stack_safe hrows hWF hlow (σ g p) (solver_output_valid hv hp)

/-- **The induced step's board is the output applied to the current board.** For a valid solver,
`(solverStep p g).board = (σ g p).applyStep g.board` — the dynamics are "apply the output". -/
theorem solver_next_board (hv : ValidSolver cfg σ) {g : GameState}
    {p : Piece} (hp : p ∈ g.bag) :
    (solverStep cfg σ p g).board = (σ g p).applyStep cfg g.board := by
  simp [solverStep, adversarialStep, Placement.eta_of_piece_eq (hv g p hp).1]

/-- **The induced step advances the bag by drawing the piece.** `(solverStep p g).bag = g.bag.draw
p` — the bag component evolves by the renewal draw, independent of the function's choice. -/
theorem solver_next_bag (g : GameState) (p : Piece) :
    (solverStep cfg σ p g).bag = g.bag.draw p :=
  rfl

/-- **The bag-evolution is choice-independent.** Any two solvers advance the bag identically — the
bag component of the dynamics ignores the function. Only the board evolution carries the choice. -/
theorem solver_bag_evolution_independent (σ₁ σ₂ : Solver cfg) (g : GameState) (p : Piece) :
    (solverStep cfg σ₁ p g).bag = (solverStep cfg σ₂ p g).bag :=
  rfl

/-- **The canonical choice is a definite witness on the winning region.** On `safe ∩ bag`,
`safeSolver g p` is the `Classical.choose` of `safe_step` — a fixed (noncomputable) selection. -/
theorem safeSolver_choice_eq_choose {g : GameState} (hg : g ∈ safe cfg)
    {p : Piece} (hp : p ∈ g.bag) :
    safeSolver cfg g p = Classical.choose (safe_step hg hp) :=
  safeSolver_eq_choose_of_safe_and_in_bag hg hp

/-- **The canonical opening choice keeps `init` safe.** If `init` is safe, the step under
`safeSolver`'s output for any first piece lands back in `safe`. -/
theorem safeSolver_init_choice_safe (h : GameState.init ∈ safe cfg) (p : Piece) :
    adversarialStep cfg GameState.init p (safeSolver cfg GameState.init p) ∈ safe cfg :=
  safeSolver_init_step_mem_safe h p

end Tetris
