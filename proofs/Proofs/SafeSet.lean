import Mathlib
import Proofs.Adversarial

/-!
# The safe set, and the reduction of `TetrisSolvable` to a single membership

The **safe set** `safe cfg ⊆ GameState` is the set of states from which a
surviving strategy exists against any legal piece sequence. It is defined as
the greatest fixed point (Knaster–Tarski) of the monotone operator

```
F(S) := { g | ¬ g.lost ∧ ∀ p ∈ g.bag, ∃ valid pl with pl.piece = p,
              adversarialStep g p pl ∈ S }
```

on the complete lattice `Set GameState`.

The headline theorem `safe_extract` shows

```
GameState.init ∈ safe StandardTetris  →  TetrisSolvable
```

This **reduces the open existential `TetrisSolvable`** (∃ strategy beating any
bag sequence) to a single membership claim in a fixed-point characterized set.
Every subsequent research path — symmetry reduction, concrete cycle
construction, mechanical search — then targets that single membership.
-/

namespace Tetris

/-- The "exists-safe-move" operator on sets of game states. A state lies in
`safeOp cfg S` iff it is not lost and, for every piece in its bag, there is a
valid placement landing inside `S`. -/
def safeOp (cfg : GameConfig) : Set GameState →o Set GameState where
  toFun S :=
    { g | ¬ g.lost cfg ∧
          ∀ p, p ∈ g.bag →
            ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
              adversarialStep cfg g p pl ∈ S }
  monotone' := by
    intro S T hST g hg
    refine ⟨hg.1, ?_⟩
    intro p hp
    obtain ⟨pl, hpiece, hv, hs⟩ := hg.2 p hp
    exact ⟨pl, hpiece, hv, hST hs⟩

/-- **The safe set**: the largest set of game states from which a surviving
strategy exists. Defined as the greatest fixed point of `safeOp`. -/
def safe (cfg : GameConfig) : Set GameState := OrderHom.gfp (safeOp cfg)

/-- The safe set is a fixed point of the safe-move operator. -/
theorem safe_eq (cfg : GameConfig) : safeOp cfg (safe cfg) = safe cfg :=
  OrderHom.map_gfp _

/-- Membership iff for the safe set as a fixed point. -/
theorem mem_safe_iff_mem_safeOp (cfg : GameConfig) (g : GameState) :
    g ∈ safe cfg ↔ g ∈ safeOp cfg (safe cfg) := by
  rw [safe_eq]

/-- Membership unfolding: a state is safe iff it is not lost and every legal
next piece has a valid placement leading to another safe state. -/
theorem mem_safe_iff (cfg : GameConfig) (g : GameState) :
    g ∈ safe cfg ↔
      ¬ g.lost cfg ∧
        ∀ p, p ∈ g.bag →
          ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
            adversarialStep cfg g p pl ∈ safe cfg := by
  conv_lhs => rw [← safe_eq cfg]
  rfl

/-- Constructor: if `g` is not lost and each draw has a safe successor, `g` is safe. -/
theorem mem_safe_of_not_lost_and_forall_step {cfg : GameConfig} {g : GameState}
    (hnl : ¬ g.lost cfg)
    (hstep : ∀ p, p ∈ g.bag →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
        adversarialStep cfg g p pl ∈ safe cfg) :
    g ∈ safe cfg :=
  (mem_safe_iff cfg g).mpr ⟨hnl, hstep⟩

/-- Eliminator: a safe state has a safe successor for every piece in its bag. -/
theorem safe_forall_step {cfg : GameConfig} {g : GameState} (h : g ∈ safe cfg) :
    ∀ p, p ∈ g.bag →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
        adversarialStep cfg g p pl ∈ safe cfg :=
  ((mem_safe_iff cfg g).mp h).2

/-- For a safe state, every drawable piece has a safe successor (`canDraw` form). -/
theorem safe_exists_step_of_canDraw {cfg : GameConfig} {g : GameState}
    (hsafe : g ∈ safe cfg) {p : Piece} (hp : g.bag.canDraw p) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
      adversarialStep cfg g p pl ∈ safe cfg :=
  safe_forall_step hsafe p hp

/-- A safe state with a nonempty bag has at least one safe successor. -/
theorem safe_exists_step_of_bag_nonempty {cfg : GameConfig} {g : GameState}
    (hsafe : g ∈ safe cfg) (hne : g.bag.Nonempty) :
    ∃ p : Piece, ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
      adversarialStep cfg g p pl ∈ safe cfg := by
  obtain ⟨p, hp⟩ := hne
  exact ⟨p, safe_forall_step hsafe p hp⟩

/-- Safe states are not lost. -/
theorem safe_not_lost {cfg : GameConfig} {g : GameState} (h : g ∈ safe cfg) :
    ¬ g.lost cfg :=
  ((mem_safe_iff cfg g).mp h).1

/-- Set form: safe states are not lost. -/
theorem safe_subset_not_lost (cfg : GameConfig) :
    safe cfg ⊆ {g | ¬ g.lost cfg} := fun _ h => safe_not_lost h

theorem safe_subset_univ (cfg : GameConfig) : safe cfg ⊆ Set.univ :=
  Set.subset_univ _

/-- `safe cfg` is disjoint from the lost-state set. -/
theorem safe_disjoint_lost (cfg : GameConfig) :
    Disjoint (safe cfg) { g | g.lost cfg } := by
  rw [Set.disjoint_left]
  exact fun _ h hlost => safe_not_lost h hlost

/-- Symmetric form: the lost-state set is disjoint from `safe cfg`. -/
theorem lost_disjoint_safe (cfg : GameConfig) :
    Disjoint { g | g.lost cfg } (safe cfg) :=
  (safe_disjoint_lost cfg).symm

/-- **Contrapositive — lost ⇒ not safe**. Lost states cannot belong to the
safe set. Atlas: instant pruning of lost candidate states. -/
theorem not_safe_of_lost {cfg : GameConfig} {g : GameState} (h : g.lost cfg) :
    g ∉ safe cfg := fun hsafe => safe_not_lost hsafe h


/-- Specialised safe-membership characterization for states with full bag
(e.g., `init`): the quantifier collapses since every piece is in the bag. -/
theorem mem_safe_iff_of_bag_full {cfg : GameConfig} {g : GameState}
    (hbag : g.bag = Bag.full) :
    g ∈ safe cfg ↔
      ¬ g.lost cfg ∧
        ∀ p : Piece, ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
          adversarialStep cfg g p pl ∈ safe cfg := by
  rw [mem_safe_iff]
  refine ⟨fun ⟨h1, h2⟩ => ⟨h1, fun p => h2 p (hbag ▸ Bag.mem_full p)⟩,
          fun ⟨h1, h2⟩ => ⟨h1, fun p _ => h2 p⟩⟩

/-- Specialised safe-membership for states with a singleton bag `{p}`. -/
theorem mem_safe_iff_of_bag_singleton {cfg : GameConfig} {g : GameState}
    {p : Piece} (hbag : g.bag = {p}) :
    g ∈ safe cfg ↔
      ¬ g.lost cfg ∧
        ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
          adversarialStep cfg g p pl ∈ safe cfg := by
  rw [mem_safe_iff]
  refine ⟨fun ⟨h1, h2⟩ => ⟨h1,
            h2 p (hbag.symm ▸ Finset.mem_singleton.mpr rfl)⟩,
          fun ⟨h1, h2⟩ => ⟨h1, ?_⟩⟩
  intro q hq
  rw [hbag, Finset.mem_singleton] at hq
  rw [hq]
  exact h2

/-- **`init`-specific safe characterization**: init is safe iff every piece
has a valid placement landing in safe. Drops the `¬ g.lost` part since
init is vacuously not lost. -/
theorem init_mem_safe_iff (cfg : GameConfig) :
    GameState.init ∈ safe cfg ↔
      ∀ p : Piece, ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
        adversarialStep cfg GameState.init p pl ∈ safe cfg := by
  rw [mem_safe_iff_of_bag_full GameState.init_bag]
  exact ⟨fun h => h.2, fun h => ⟨GameState.init_not_lost cfg, h⟩⟩

/-- **`init`-safe characterization via the finite `allValidFor` set**: a
*decidable*-friendly reformulation that searches `Placement.allValidFor`
(a finite Finset) instead of the infinite `Placement` universe. -/
theorem init_mem_safe_iff_allValidFor (cfg : GameConfig) :
    GameState.init ∈ safe cfg ↔
      ∀ p : Piece, ∃ pl ∈ Placement.allValidFor cfg p,
        adversarialStep cfg GameState.init p pl ∈ safe cfg := by
  rw [init_mem_safe_iff]
  apply forall_congr'
  intro p
  constructor
  · rintro ⟨pl, hpiece, hvalid, hsafe⟩
    exact ⟨pl, (Placement.mem_allValidFor _ _ _).mpr ⟨hpiece, hvalid⟩, hsafe⟩
  · rintro ⟨pl, hpl, hsafe⟩
    obtain ⟨hpiece, hvalid⟩ := (Placement.mem_allValidFor _ _ _).mp hpl
    exact ⟨pl, hpiece, hvalid, hsafe⟩

/-- **General safe characterization via `allValidFor`**: any state's
safety is decidable-friendly when expressed over `allValidFor`. -/
theorem mem_safe_iff_allValidFor {cfg : GameConfig} {g : GameState} :
    g ∈ safe cfg ↔
      ¬ g.lost cfg ∧
        ∀ p, p ∈ g.bag → ∃ pl ∈ Placement.allValidFor cfg p,
          adversarialStep cfg g p pl ∈ safe cfg := by
  rw [mem_safe_iff]
  refine ⟨fun ⟨h1, h2⟩ => ⟨h1, fun p hp => ?_⟩,
          fun ⟨h1, h2⟩ => ⟨h1, fun p hp => ?_⟩⟩
  · obtain ⟨pl, hpiece, hvalid, hsafe⟩ := h2 p hp
    exact ⟨pl, (Placement.mem_allValidFor _ _ _).mpr ⟨hpiece, hvalid⟩, hsafe⟩
  · obtain ⟨pl, hpl, hsafe⟩ := h2 p hp
    obtain ⟨hpiece, hvalid⟩ := (Placement.mem_allValidFor _ _ _).mp hpl
    exact ⟨pl, hpiece, hvalid, hsafe⟩


/-- From a safe state, every piece in the bag has a valid placement that lands
in another safe state. -/
theorem safe_step {cfg : GameConfig} {g : GameState} (h : g ∈ safe cfg)
    {p : Piece} (hp : p ∈ g.bag) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
      adversarialStep cfg g p pl ∈ safe cfg :=
  ((mem_safe_iff cfg g).mp h).2 p hp

/-- **Coinduction / greatest-postfix principle.** Any set closed under safe
moves is contained in the safe set. The natural tool for proving safe-set
membership: exhibit a closed set `S` containing the state of interest. -/
theorem safe_greatest {cfg : GameConfig} (S : Set GameState)
    (hS : S ⊆ safeOp cfg S) : S ⊆ safe cfg :=
  OrderHom.le_gfp _ hS

/-- **No valid placement for a drawable piece ⇒ not safe**. If some piece
in the bag cannot be played in-bounds, the state is doomed (no surviving
strategy responds to that piece). -/
theorem not_safe_of_no_valid_placement {cfg : GameConfig} {g : GameState}
    {p : Piece} (hp : p ∈ g.bag)
    (h : ¬ ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg) :
    g ∉ safe cfg := fun hsafe => by
  obtain ⟨pl, hpiece, hvalid, _⟩ := safe_step hsafe hp
  exact h ⟨pl, hpiece, hvalid⟩

/-- **Safe-set property**: every safe state admits a valid placement for
each drawable piece. -/
theorem safe_has_valid_for_each_bag_piece {cfg : GameConfig} {g : GameState}
    (hg : g ∈ safe cfg) {p : Piece} (hp : p ∈ g.bag) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg :=
  let ⟨pl, hpiece, hvalid, _⟩ := safe_step hg hp
  ⟨pl, hpiece, hvalid⟩

/-! ## Extracting a solver from safe-set membership -/

/-- A solver that, at each step, picks a placement keeping the game inside the
safe set (when possible) — chosen via `Classical.choose`. Outside the safe set
or when the piece is not in the bag, falls back to a trivial placement. -/
noncomputable def safeSolver (cfg : GameConfig) : Solver cfg := fun g p =>
  letI : Decidable (g ∈ safe cfg ∧ p ∈ g.bag) := Classical.propDecidable _
  if h : g ∈ safe cfg ∧ p ∈ g.bag then
    Classical.choose (safe_step h.1 h.2)
  else
    ⟨p, 0, 0⟩

theorem safeSolver_spec {cfg : GameConfig} {g : GameState} (hg : g ∈ safe cfg)
    {p : Piece} (hp : p ∈ g.bag) :
    (safeSolver cfg g p).piece = p ∧
    (safeSolver cfg g p).Valid cfg ∧
    adversarialStep cfg g p (safeSolver cfg g p) ∈ safe cfg := by
  unfold safeSolver
  rw [dif_pos ⟨hg, hp⟩]
  exact Classical.choose_spec (safe_step hg hp)

/-- **Solver property**: the `safeSolver`-step from any safe state lands
in a safe state. Direct projection from `safeSolver_spec.2.2`. -/
theorem safeSolver_step_mem_safe {cfg : GameConfig} {g : GameState}
    (hsafe : g ∈ safe cfg) {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (safeSolver cfg g p) ∈ safe cfg :=
  (safeSolver_spec hsafe hp).2.2

/-- **Solver characterization (fallback branch)**: when not in safe ∧ in
bag, `safeSolver` returns the trivial placement `⟨p, 0, 0⟩`. -/
theorem safeSolver_eq_trivial_of_not_safe_and_in_bag {cfg : GameConfig}
    {g : GameState} {p : Piece}
    (h : ¬ (g ∈ safe cfg ∧ p ∈ g.bag)) :
    safeSolver cfg g p = ⟨p, 0, 0⟩ := by
  unfold safeSolver
  rw [dif_neg h]

/-- **Solver characterization (safe branch)**: when in safe ∧ in bag,
`safeSolver` returns the placement chosen by `Classical.choose` on
`safe_step`. -/
theorem safeSolver_eq_choose_of_safe_and_in_bag {cfg : GameConfig}
    {g : GameState} {p : Piece}
    (hsafe : g ∈ safe cfg) (hp : p ∈ g.bag) :
    safeSolver cfg g p = Classical.choose (safe_step hsafe hp) := by
  unfold safeSolver
  rw [dif_pos ⟨hsafe, hp⟩]

/-- **safeSolver from init lands in safe** (when init is safe). One-line
restate of `safeSolver_spec` for init's full bag. -/
theorem safeSolver_init_step_mem_safe {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) (p : Piece) :
    adversarialStep cfg GameState.init p (safeSolver cfg GameState.init p)
      ∈ safe cfg :=
  (safeSolver_spec h (GameState.init_bag.symm ▸ Bag.mem_full p)).2.2

/-- `safeSolver` always returns the announced piece (no safety hypothesis
needed): in the safe branch via `safe_step`'s spec, in the fallback via
the trivial `⟨p, 0, 0⟩`. -/
@[simp] theorem safeSolver_piece (cfg : GameConfig) (g : GameState) (p : Piece) :
    (safeSolver cfg g p).piece = p := by
  unfold safeSolver
  split_ifs with h
  · exact (Classical.choose_spec (safe_step h.1 h.2)).1
  · rfl



/-! ## The reduction: solvability ⇐ initial safety -/

/-- **The headline reduction, parametric form.** If `init` is in `safe cfg`,
then `TetrisSolvableFor cfg`. -/
theorem safe_extract_for {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) : TetrisSolvableFor cfg := by
  refine ⟨safeSolver cfg, ?_⟩
  intro s hlegal n
  suffices htrace :
      adversarialTrace cfg (safeSolver _) s GameState.init n ∈ safe cfg from
    safe_not_lost htrace
  induction n with
  | zero => exact h
  | succ k ih =>
      rw [adversarialTrace_succ]
      have hbag : s k ∈
          (adversarialTrace cfg (safeSolver _) s GameState.init k).bag := by
        have hl := hlegal k
        rw [adversarialTrace_bag]
        exact hl
      exact (safeSolver_spec ih hbag).2.2

/-- The original `safe_extract` specialised to `GameConfig.standard`. -/
theorem safe_extract
    (h : GameState.init ∈ safe GameConfig.standard) : TetrisSolvable :=
  safe_extract_for h

/-- `safeSolver` is a `ValidSolver` when `cfg.cols ≥ 4`. In the safe branch
the placement comes from `safeSolver_spec`; in the fallback branch the
trivial `⟨p, 0, 0⟩` is in-bounds because every piece's `shapeUp` has
cells with column `< 4`. -/
theorem safeSolver_validSolver {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) :
    ValidSolver cfg (safeSolver cfg) := by
  intro g p _hp
  unfold safeSolver
  by_cases h : g ∈ safe cfg ∧ p ∈ g.bag
  · rw [dif_pos h]
    have hspec := Classical.choose_spec (safe_step h.1 h.2)
    exact ⟨hspec.1, hspec.2.1⟩
  · rw [dif_neg h]
    refine ⟨rfl, ?_⟩
    intro cell hcell
    have hc4 := Piece.shapeUp_col_lt_four p 0 cell hcell
    change 0 + cell.1 < cfg.cols
    omega

/-- **Solver bounded-domain property**: `safeSolver`'s choice for a
drawable piece always lies in the finite `allValidFor` set. -/
theorem safeSolver_mem_allValidFor {cfg : GameConfig} {g : GameState}
    {p : Piece} (hcols : 4 ≤ cfg.cols) (hp : p ∈ g.bag) :
    safeSolver cfg g p ∈ Placement.allValidFor cfg p := by
  rw [Placement.mem_allValidFor]
  exact ⟨safeSolver_piece cfg g p,
         (safeSolver_validSolver hcols g p hp).2⟩

/-- Helper: substituting in the same piece is a no-op. -/
theorem Placement.eta_of_piece_eq {pl : Placement} {p : Piece}
    (h : pl.piece = p) :
    ({pl with piece := p} : Placement) = pl := by
  rcases pl with ⟨pc, r, c⟩
  cases h
  rfl

/-- `adversarialStep` with `safeSolver` collapses to the single-state `step`
(the `{with piece := p}` substitution is a no-op since safeSolver
already returns the announced piece). -/
theorem adversarialStep_safeSolver (cfg : GameConfig) (g : GameState)
    (p : Piece) :
    adversarialStep cfg g p (safeSolver cfg g p) =
      g.step cfg (safeSolver cfg g p) := by
  rw [adversarialStep_eq_step,
      Placement.eta_of_piece_eq (safeSolver_piece cfg g p)]

/-- The first `safeSolver`-step from `init` is reachable. -/
theorem safeSolver_init_step_reachable {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (p : Piece) :
    Reachable cfg
      (adversarialStep cfg GameState.init p (safeSolver cfg GameState.init p)) := by
  rw [adversarialStep_safeSolver]
  apply Reachable.step Reachable.init
  · rw [safeSolver_piece]
    exact GameState.init_bag.symm ▸ Bag.mem_full p
  · exact (safeSolver_validSolver hcols GameState.init p
      (GameState.init_bag.symm ▸ Bag.mem_full p)).2
  · exact GameState.init_not_lost cfg

/-- **The trace from `init` via `safeSolver` is reachable.** A safe state
played through `safeSolver` against any legal sequence yields exclusively
`Reachable`-from-init states. The induction also maintains the
"`trace n ∈ safe cfg`" invariant — a strengthened induction hypothesis
needed because `safeSolver_spec` needs the current state to be safe. -/
theorem adversarialTrace_reachable_and_safe {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg)
    (s : ℕ → Piece) (hl : LegalSequence s) :
    ∀ n, Reachable cfg (adversarialTrace cfg (safeSolver cfg) s GameState.init n) ∧
         adversarialTrace cfg (safeSolver cfg) s GameState.init n ∈ safe cfg := by
  intro n
  induction n with
  | zero => exact ⟨Reachable.init, h⟩
  | succ k ih =>
    obtain ⟨ihr, ihs⟩ := ih
    set g := adversarialTrace cfg (safeSolver cfg) s GameState.init k with hg
    have hpb : s k ∈ g.bag := by
      have hbag : g.bag = bagAt Bag.full s k := adversarialTrace_bag cfg _ s k
      rw [hbag]; exact hl k
    obtain ⟨hpiece, hvalid, hstepsafe⟩ := safeSolver_spec ihs hpb
    have heta : ({safeSolver cfg g (s k) with piece := s k} : Placement)
        = safeSolver cfg g (s k) := Placement.eta_of_piece_eq hpiece
    refine ⟨?_, hstepsafe⟩
    rw [adversarialTrace_succ, ← hg, adversarialStep_eq_step, heta]
    refine Reachable.step ihr ?_ hvalid (safe_not_lost ihs)
    rw [hpiece]; exact hpb

/-- Direct corollary: every trace state is `Reachable`. -/
theorem adversarialTrace_reachable {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    Reachable cfg (adversarialTrace cfg (safeSolver cfg) s GameState.init n) :=
  (adversarialTrace_reachable_and_safe h s hl n).1

/-- **Every trace state via `safeSolver` (from a safe `init`) is also in
`safe`.** Companion to `adversarialTrace_reachable`. -/
theorem adversarialTrace_mem_safe {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg (safeSolver cfg) s GameState.init n ∈ safe cfg :=
  (adversarialTrace_reachable_and_safe h s hl n).2

/-- `safeSolver` `SolvesTetris` whenever `init ∈ safe`. (Inlining of the
core argument of `safe_extract_for`.) -/
theorem safeSolver_solvesTetris {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    SolvesTetris cfg (safeSolver cfg) := fun s hl n =>
  safe_not_lost (adversarialTrace_reachable_and_safe h s hl n).2

/-- **The forward direction with valid placements**: if `init ∈ safe cfg`
and `cfg.cols ≥ 4`, then `safeSolver cfg` witnesses `SolvesTetrisValid`. -/
theorem init_safe_implies_solvesTetrisValid {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (h : GameState.init ∈ safe cfg) :
    SolvesTetrisValid cfg (safeSolver cfg) :=
  ⟨safeSolver_validSolver hcols, safeSolver_solvesTetris h⟩

/-- **Every solver-reachable state is safe (when σ is a `ValidSolver`
that solves).** Extracts the Knaster-Tarski argument of
`init_safe_of_solvesTetrisValid` as a standalone — useful for any
solver-reachable state, not just `init`. -/
theorem solverReachable_subset_safe {cfg : GameConfig} {σ : Solver cfg}
    (h : SolvesTetrisValid cfg σ) {g : GameState} (hr : solverReachable σ g) :
    g ∈ safe cfg := by
  obtain ⟨hvalid, hsolves⟩ := h
  apply safe_greatest { g | solverReachable σ g } _ hr
  intro g' hg'
  refine ⟨solverReachable_not_lost hsolves hg', ?_⟩
  intro p hp
  refine ⟨{σ g' p with piece := p}, rfl, ?_, ?_⟩
  · have hvp := hvalid g' p hp
    rw [Placement.eta_of_piece_eq hvp.1]
    exact hvp.2
  · show adversarialStep cfg g' p {σ g' p with piece := p} ∈
      { g | solverReachable σ g }
    rw [Placement.eta_of_piece_eq (hvalid g' p hp).1]
    exact solverReachable.step p hg' hp

/-- **The harder direction**: if a `ValidSolver` solves Tetris, then `init`
is in `safe`. Now a one-line corollary of Tick 226's
`solverReachable_subset_safe` applied to `solverReachable.init`. -/
theorem init_safe_of_solvesTetrisValid {cfg : GameConfig} {σ : Solver cfg}
    (h : SolvesTetrisValid cfg σ) : GameState.init ∈ safe cfg :=
  solverReachable_subset_safe h solverReachable.init

/-- **Solving-solver's init-step is safe**: any `SolvesTetrisValid` σ has
its first init-step landing in `safe cfg`. Direct from
`solverReachable_subset_safe` + `solverReachable_init_step`. -/
theorem solvesTetrisValid_init_step_mem_safe {cfg : GameConfig}
    {σ : Solver cfg} (h : SolvesTetrisValid cfg σ) (p : Piece) :
    adversarialStep cfg GameState.init p (σ GameState.init p) ∈ safe cfg :=
  solverReachable_subset_safe h (solverReachable_init_step cfg σ p)

/-- **Solving-solver's full trace is in safe**: every trace state via a
`SolvesTetrisValid` σ from init lies in `safe cfg`. Generalises Tick
240 to n-step. -/
theorem solvesTetrisValid_trace_mem_safe {cfg : GameConfig} {σ : Solver cfg}
    (h : SolvesTetrisValid cfg σ) {s : ℕ → Piece} (hl : LegalSequence s)
    (n : ℕ) :
    adversarialTrace cfg σ s GameState.init n ∈ safe cfg :=
  solverReachable_subset_safe h (adversarialTrace_solverReachable σ hl n)

/-- **Solver-reachable states are `Reachable`** when the solver is valid
and solves. Each step's placement is `Valid` (from `ValidSolver`) and the
source is not lost (from `SolvesTetris`), satisfying `Reachable.step`. -/
theorem solverReachable_implies_reachable_of_solves {cfg : GameConfig}
    {σ : Solver cfg} (hv : ValidSolver cfg σ) (hs : SolvesTetris cfg σ)
    {g : GameState} (h : solverReachable σ g) : Reachable cfg g := by
  induction h with
  | init => exact Reachable.init
  | @step g p hr _hp ih =>
    rw [adversarialStep_eq_step, Placement.eta_of_piece_eq (hv g p _hp).1]
    refine Reachable.step ih ?_ (hv g p _hp).2 (solverReachable_not_lost hs hr)
    rw [(hv g p _hp).1]; exact _hp

/-- **The full bidirectional equivalence** for `SolvesTetrisValid`. -/
theorem init_safe_iff_exists_solvesTetrisValid {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) :
    GameState.init ∈ safe cfg ↔ ∃ σ : Solver cfg, SolvesTetrisValid cfg σ := by
  constructor
  · intro h
    exact ⟨safeSolver cfg, init_safe_implies_solvesTetrisValid hcols h⟩
  · rintro ⟨_, hσ⟩
    exact init_safe_of_solvesTetrisValid hσ

/-- **Headline iff for parametric `TetrisSolvableValidFor`**. -/
theorem tetrisSolvableValidFor_iff_init_safe {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) :
    TetrisSolvableValidFor cfg ↔ GameState.init ∈ safe cfg :=
  (init_safe_iff_exists_solvesTetrisValid hcols).symm

/-- Dot accessor: solvability implies `init ∈ safe cfg`. -/
theorem TetrisSolvableValidFor.init_safe {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (h : TetrisSolvableValidFor cfg) :
    GameState.init ∈ safe cfg :=
  (tetrisSolvableValidFor_iff_init_safe hcols).mp h

/-- Standard-config specialization. -/
theorem TetrisSolvableValid.init_safe (h : TetrisSolvableValid) :
    GameState.init ∈ safe GameConfig.standard :=
  TetrisSolvableValidFor.init_safe (by decide) h

/-- Tiny-config specialization. -/
theorem TetrisSolvableValidFor.tiny_init_safe
    (h : TetrisSolvableValidFor GameConfig.tiny) :
    GameState.init ∈ safe GameConfig.tiny :=
  TetrisSolvableValidFor.init_safe (by decide) h

/-- **Headline iff for `TetrisSolvableValid`** (the standard 10×20 config). -/
theorem tetrisSolvableValid_iff_init_safe :
    TetrisSolvableValid ↔ GameState.init ∈ safe GameConfig.standard :=
  tetrisSolvableValidFor_iff_init_safe (by decide)

/-! ### Existence from a closed invariant (search-free sufficient condition)

The model-independent reduction of solvability to *exhibiting one invariant*:
no atlas enumeration, no reflected search artifact. A set `S` is a **closed
invariant** when it contains `init`, holds no lost state, and from every state
in `S` each drawable bag piece has a valid placement landing back in `S`. The
coinduction principle `safe_greatest` turns any such `S` into a proof that
`init ∈ safe`, hence `TetrisSolvableValid`. These theorems package that
composition with the `safeOp` membership pre-unfolded, so an invariant-closure
argument (e.g. by piece geometry) plugs straight into `hnotlost`/`hstep`. They
are the exact spec any witness — symbolic invariant or potential bound — must
meet. -/

/-- **Safe-membership from a closed invariant.** If `S` contains `init`, holds
no lost state, and is closed under adversarial steps (every bag piece has a
valid placement landing back in `S`), then `init ∈ safe`. The init-level
specialisation of `safe_greatest` with the `safeOp` membership spelled out into
its two explicit obligations. -/
theorem init_mem_safe_of_invariant {cfg : GameConfig}
    (S : Set GameState) (hinit : GameState.init ∈ S)
    (hnotlost : ∀ g ∈ S, ¬ g.lost cfg)
    (hstep : ∀ g ∈ S, ∀ p, p ∈ g.bag →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
        adversarialStep cfg g p pl ∈ S) :
    GameState.init ∈ safe cfg :=
  safe_greatest S (fun g hg => ⟨hnotlost g hg, hstep g hg⟩) hinit

/-- **Solvability from a closed invariant (parametric).** For any wide-enough
board, a closed invariant containing `init` certifies `TetrisSolvableValidFor`.
The hard content of "Tetris is solvable" is reduced to exhibiting one closed
invariant — no enumeration. -/
theorem tetrisSolvableValidFor_of_invariant {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols)
    (S : Set GameState) (hinit : GameState.init ∈ S)
    (hnotlost : ∀ g ∈ S, ¬ g.lost cfg)
    (hstep : ∀ g ∈ S, ∀ p, p ∈ g.bag →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
        adversarialStep cfg g p pl ∈ S) :
    TetrisSolvableValidFor cfg :=
  (tetrisSolvableValidFor_iff_init_safe hcols).mpr
    (init_mem_safe_of_invariant S hinit hnotlost hstep)

/-- **Solvability from a closed invariant (standard 10×20).** The headline
search-free existence theorem: any closed invariant containing `init` proves
`TetrisSolvableValid`. -/
theorem tetrisSolvableValid_of_invariant
    (S : Set GameState) (hinit : GameState.init ∈ S)
    (hnotlost : ∀ g ∈ S, ¬ g.lost GameConfig.standard)
    (hstep : ∀ g ∈ S, ∀ p, p ∈ g.bag →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        adversarialStep GameConfig.standard g p pl ∈ S) :
    TetrisSolvableValid :=
  tetrisSolvableValid_iff_init_safe.mpr
    (init_mem_safe_of_invariant S hinit hnotlost hstep)

/-- **Solvability from a height-bounded closed invariant.** A convenience
specialisation of `tetrisSolvableValid_of_invariant` that trades the abstract
`¬ lost` obligation for a concrete per-column height bound: if every state in
`S` keeps all column heights at most `cfg.rows`, the loss obligation is
discharged automatically (via `Board.not_isLost_of_colHeight_le`). A symbolic
invariant phrased as a surface-height envelope then only needs to supply
`hinit`, the height bound, and the closure step `hstep` — the loss obligation
comes for free. -/
theorem tetrisSolvableValid_of_height_bounded_invariant
    (S : Set GameState) (hinit : GameState.init ∈ S)
    (hheight : ∀ g ∈ S, ∀ j, Board.colHeight g.board j ≤ GameConfig.standard.rows)
    (hstep : ∀ g ∈ S, ∀ p, p ∈ g.bag →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        adversarialStep GameConfig.standard g p pl ∈ S) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_invariant S hinit
    (fun g hg => (GameState.not_lost_iff_not_board_isLost _ g).mpr
      (Board.not_isLost_of_colHeight_le (hheight g hg)))
    hstep

/-- Config-specific iff: `standard`. -/
theorem standard_init_safe_iff_exists_solvesTetrisValid :
    GameState.init ∈ safe GameConfig.standard ↔
    ∃ σ : Solver GameConfig.standard, SolvesTetrisValid GameConfig.standard σ :=
  init_safe_iff_exists_solvesTetrisValid (by decide)

/-- Config-specific iff: `tiny`. -/
theorem tiny_init_safe_iff_exists_solvesTetrisValid :
    GameState.init ∈ safe GameConfig.tiny ↔
    ∃ σ : Solver GameConfig.tiny, SolvesTetrisValid GameConfig.tiny σ :=
  init_safe_iff_exists_solvesTetrisValid (by decide)

/-- **The safe-and-reachable invariant is preserved by `safeSolver` steps.**
For atlas construction: walking from any reachable safe state via the
canonical `safeSolver` keeps us in the reachable-safe subset. -/
theorem reachable_safe_safeSolver_step {cfg : GameConfig}
    {g : GameState} (hr : Reachable cfg g) (hs : g ∈ safe cfg)
    {p : Piece} (hp : p ∈ g.bag) :
    Reachable cfg (adversarialStep cfg g p (safeSolver cfg g p)) ∧
      adversarialStep cfg g p (safeSolver cfg g p) ∈ safe cfg := by
  obtain ⟨hpiece, hvalid, hstep_safe⟩ := safeSolver_spec hs hp
  refine ⟨?_, hstep_safe⟩
  rw [adversarialStep_eq_step, Placement.eta_of_piece_eq hpiece]
  refine Reachable.step hr ?_ hvalid (safe_not_lost hs)
  rw [hpiece]; exact hp

/-- Config-specific specialisation: `standard`. -/
theorem init_safe_implies_solvesTetrisValid_standard
    (h : GameState.init ∈ safe GameConfig.standard) :
    SolvesTetrisValid GameConfig.standard (safeSolver GameConfig.standard) :=
  init_safe_implies_solvesTetrisValid (by decide) h

/-- Config-specific specialisation: `tiny`. -/
theorem init_safe_implies_solvesTetrisValid_tiny
    (h : GameState.init ∈ safe GameConfig.tiny) :
    SolvesTetrisValid GameConfig.tiny (safeSolver GameConfig.tiny) :=
  init_safe_implies_solvesTetrisValid (by decide) h

/-! ## Adversarial closed cycles — the concrete tool for proving `init ∈ safe` -/

/-- An **adversarial closed cycle**: a finite set of game states with a solver
that, for every piece in the current bag, returns a valid non-losing placement
keeping us inside the set. Any such cycle witnesses safety of its states
(`subset_safe`), reducing M2 to *exhibiting* such a structure containing
`init`. -/
structure AdversarialClosedCycle (cfg : GameConfig) where
  /-- The set of states comprising the cycle. -/
  states : Finset GameState
  /-- The cycle's solver. -/
  solver : Solver cfg
  /-- No state in the cycle is lost. -/
  not_lost : ∀ s ∈ states, ¬ s.lost cfg
  /-- For every piece in the bag, the chosen placement is valid and plays that piece. -/
  valid : ∀ s ∈ states, ∀ p, p ∈ s.bag →
    (solver s p).piece = p ∧ (solver s p).Valid cfg
  /-- The cycle is closed under any legal piece + the solver's move. -/
  closed : ∀ s ∈ states, ∀ p, p ∈ s.bag →
    adversarialStep cfg s p (solver s p) ∈ states

namespace AdversarialClosedCycle

/-- Piece-half of `valid`. -/
theorem solver_piece {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {g : GameState} (hg : g ∈ C.states) {p : Piece} (hp : p ∈ g.bag) :
    (C.solver g p).piece = p :=
  (C.valid g hg p hp).1

/-- Bundled `valid`: the solver's placement is valid AND plays the given piece. -/
theorem solver_valid_and_piece {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {g : GameState} (hg : g ∈ C.states) {p : Piece} (hp : p ∈ g.bag) :
    (C.solver g p).piece = p ∧ (C.solver g p).Valid cfg :=
  C.valid g hg p hp

/-- Mem-form of `closed`: post-step is in cycle. -/
theorem step_mem_states {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {g : GameState} (hg : g ∈ C.states) {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (C.solver g p) ∈ C.states :=
  C.closed g hg p hp

/-- Post-step is non-lost (combines `step_mem_states` with `not_lost`). -/
theorem step_not_lost {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {g : GameState} (hg : g ∈ C.states) {p : Piece} (hp : p ∈ g.bag) :
    ¬ (adversarialStep cfg g p (C.solver g p)).lost cfg :=
  C.not_lost _ (C.step_mem_states hg hp)

/-- Bundled `step_mem_states + step_not_lost`. -/
theorem step_mem_and_not_lost {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {g : GameState} (hg : g ∈ C.states) {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (C.solver g p) ∈ C.states ∧
      ¬ (adversarialStep cfg g p (C.solver g p)).lost cfg :=
  ⟨C.step_mem_states hg hp, C.step_not_lost hg hp⟩

/-- Membership in `C.states` is closed under all-pieces step. -/
theorem step_mem_states_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) (p : Piece) :
    adversarialStep cfg GameState.init p (C.solver GameState.init p) ∈ C.states :=
  C.step_mem_states h (GameState.init_bag.symm ▸ Bag.mem_full p)

/-- Init-step is non-lost for any piece. -/
theorem step_not_lost_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) (p : Piece) :
    ¬ (adversarialStep cfg GameState.init p (C.solver GameState.init p)).lost cfg :=
  C.not_lost _ (C.step_mem_states_of_init h p)

/-- Solver at init plays the requested piece (any piece is in init's bag). -/
theorem solver_init_piece {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) (p : Piece) :
    (C.solver GameState.init p).piece = p :=
  C.solver_piece h (GameState.init_bag.symm ▸ Bag.mem_full p)

/-- Solver at init plays a valid placement. -/
theorem solver_init_valid {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) (p : Piece) :
    (C.solver GameState.init p).Valid cfg :=
  (C.valid _ h _ (GameState.init_bag.symm ▸ Bag.mem_full p)).2

/-- Bundled init-solver: plays the requested piece AND is valid. -/
theorem solver_init_piece_and_valid {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) (p : Piece) :
    (C.solver GameState.init p).piece = p ∧
      (C.solver GameState.init p).Valid cfg :=
  ⟨C.solver_init_piece h p, C.solver_init_valid h p⟩

/-- Init-step is in cycle AND non-lost (bundled). -/
theorem step_mem_and_not_lost_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) (p : Piece) :
    adversarialStep cfg GameState.init p (C.solver GameState.init p) ∈ C.states ∧
      ¬ (adversarialStep cfg GameState.init p (C.solver GameState.init p)).lost cfg :=
  ⟨C.step_mem_states_of_init h p, C.step_not_lost_of_init h p⟩

/-- **Trace stays in the cycle**: any adversarial trace of `C.solver` from
`init`, along a legal piece sequence, never leaves `C.states`. -/
theorem adversarialTrace_mem_states_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg C.solver s GameState.init n ∈ C.states := by
  induction n with
  | zero => exact h
  | succ k ih =>
    rw [adversarialTrace_succ]
    refine C.step_mem_states ih ?_
    have := hl.canDraw k
    rw [Bag.canDraw_iff_mem, ← adversarialTrace_bag] at this
    exact this

/-- **Trace stays in the cycle from any cycle state.** Generalises
`adversarialTrace_mem_states_of_init`: starting from any `g0 ∈ C.states` and
any sequence `t` that is `LegalSequenceFrom g0.bag`, the trace under
`C.solver` stays in `C.states` for all steps. Proof: induction on `n`,
identical to the init version but uses `adversarialTrace_bag_from` (Tick
1575) to identify the trace's bag with `bagAt g0.bag t k`. The final
ingredient of the M3 splice toolkit. -/
theorem adversarialTrace_mem_states_from_mem {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g0 : GameState} (hg0 : g0 ∈ C.states)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g0.bag t) (n : ℕ) :
    adversarialTrace cfg C.solver t g0 n ∈ C.states := by
  induction n with
  | zero => exact hg0
  | succ k ih =>
    rw [adversarialTrace_succ]
    refine C.step_mem_states ih ?_
    have := hl k
    rw [Bag.canDraw_iff_mem, ← adversarialTrace_bag_from] at this
    exact this

/-- **Adversarial trace is never lost**: corollary of
`adversarialTrace_mem_states_of_init` + `C.not_lost`. -/
theorem adversarialTrace_not_lost_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    ¬ (adversarialTrace cfg C.solver s GameState.init n).lost cfg :=
  C.not_lost _ (C.adversarialTrace_mem_states_of_init h hl n)

/-- **Adversarial trace from init under a cycle solver is `Reachable`.** A
foundational structural theorem: combining the cycle invariants
(`adversarialTrace_mem_states_of_init` + `C.not_lost` + `C.solver_valid_and_piece`)
with the reachability lift `Reachable.adversarialStep` (Tick 1558), every
state visited by an adversarial trace of `C.solver` from `init`, along any
legal piece sequence, is reachable from `init`. Composed with
`Reachable.mem_inFieldStates_of_not_lost` (Tick 1557) this embeds the entire
adversarial-trace cone of an init-containing cycle into `inFieldStates cfg`. -/
theorem adversarialTrace_reachable_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Reachable cfg (adversarialTrace cfg C.solver s GameState.init n) := by
  induction n with
  | zero => exact Reachable.init
  | succ k ih =>
    rw [adversarialTrace_succ]
    set g := adversarialTrace cfg C.solver s GameState.init k with hgdef
    have hmem : g ∈ C.states := C.adversarialTrace_mem_states_of_init h hl k
    have hnl : ¬ g.lost cfg := C.not_lost _ hmem
    have hbag : s k ∈ g.bag := by
      have := hl.canDraw k
      rw [Bag.canDraw_iff_mem, ← adversarialTrace_bag] at this
      exact this
    obtain ⟨hpiece, hval⟩ := C.solver_valid_and_piece hmem hbag
    have heta : ({ C.solver g (s k) with piece := s k } : Placement)
        = C.solver g (s k) := by
      rcases hpl : C.solver g (s k) with ⟨pc, r, c⟩
      rw [hpl] at hpiece
      cases hpiece
      rfl
    refine ih.adversarialStep hnl hbag ?_
    rw [heta]
    exact hval

/-- **The cycle's solver solves Tetris**: if `init ∈ C.states`, then
`C.solver` is a `SolvesTetris` witness — the unfolded form of
`init_solves_for` cycle-wise. -/
theorem solvesTetris_solver_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    SolvesTetris cfg C.solver :=
  fun _ hl n => C.adversarialTrace_not_lost_of_init h hl n

/-- **Cycle's solver `SolvesTetrisFrom` every cycle state**: for any
`g₀ ∈ C.states`, `SolvesTetrisFrom cfg g₀ C.solver`. Cycle-side
parallel to M4's Tick 1667's `M4Artifact.toSolver_solvesTetrisFrom`.
Direct via `not_lost` applied to `adversarialTrace_mem_states_from_mem`. -/
theorem solvesTetrisFrom_of_mem {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g₀ : GameState}
    (hg₀ : g₀ ∈ C.states) :
    SolvesTetrisFrom cfg g₀ C.solver :=
  fun _ hl n =>
    C.not_lost _ (C.adversarialTrace_mem_states_from_mem hg₀ hl n)

/-- **Cycle's trace from any state is never lost**: parallel to the
existing `_of_init` version (Tick 1773 area's `adversarialTrace_not_lost_of_init`)
but starting from any `g₀ ∈ C.states`. Direct via `not_lost` applied to
`adversarialTrace_mem_states_from_mem`. -/
theorem adversarialTrace_not_lost_from_mem {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g₀ : GameState}
    (hg₀ : g₀ ∈ C.states) {t : ℕ → Piece}
    (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    ¬ (adversarialTrace cfg C.solver t g₀ n).lost cfg :=
  C.not_lost _ (C.adversarialTrace_mem_states_from_mem hg₀ hl n)


/-- **Cycle ⇒ existence of a solver** (any-config version). -/
theorem exists_solvesTetris_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    ∃ σ : Solver cfg, SolvesTetris cfg σ :=
  ⟨C.solver, C.solvesTetris_solver_of_init h⟩

/-- **Cycle ⇒ closed Atlas.** A cycle's solver lifted into the `Atlas`
language (via `Solver.toAtlas`) is **closed** on the cycle's state set
(`Atlas.IsClosedOn`). This is the structural identity at the heart of
`AGENTS.md`'s milestone M2/M3/M4 vision: a closed atlas IS a cycle, and
vice versa. Every cycle is a proof-by-construction artifact in the Atlas
language. -/
theorem solver_toAtlas_isClosedOn_states {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) :
    C.solver.toAtlas.IsClosedOn cfg C.states where
  not_lost := C.not_lost
  total := fun _ _ _ _ => by simp [Solver.toAtlas]
  valid := fun g hg p hp pl hpl => by
    simp only [Solver.toAtlas, Option.some.injEq] at hpl
    rw [← hpl]
    exact C.valid g hg p hp
  closed := fun g hg p hp pl hpl => by
    simp only [Solver.toAtlas, Option.some.injEq] at hpl
    rw [← hpl]
    exact C.closed g hg p hp

/-- **Cycle's solver-atlas dom cardinality equals states cardinality**:
`(C.solver.toAtlas.dom C.states).card = C.states.card`. Direct via
Tick 1601 (`solver_toAtlas_isClosedOn_states`) + Tick 1960's
atlas-level `dom_card_eq`. Cycle-side parallel of M4 Tick 1959 /
atlas Tick 1960. -/
theorem solver_toAtlas_dom_states_card_eq {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) :
    (C.solver.toAtlas.dom C.states).card = C.states.card :=
  C.solver_toAtlas_isClosedOn_states.dom_card_eq

/-- **Cycle's canon solver-atlas dom cardinality equals states
cardinality**: `((C.solver.toAtlas.canon C.states).dom C.states).card
= C.states.card`. Direct via Tick 1601 + Tick 1962
(`canon_dom_card_eq`). Canon-form parallel of Tick 1961. -/
theorem solver_toAtlas_canon_dom_states_card_eq {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) :
    ((C.solver.toAtlas.canon C.states).dom C.states).card = C.states.card :=
  C.solver_toAtlas_isClosedOn_states.canon_dom_card_eq

/-- **Cycle's restricted solver-atlas dom cardinality equals states
cardinality**: `((C.solver.toAtlas.restrict C.states).dom C.states).card
= C.states.card`. Restrict-form parallel of Ticks 1961, 1963. -/
theorem solver_toAtlas_restrict_dom_states_card_eq {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) :
    ((C.solver.toAtlas.restrict C.states).dom C.states).card = C.states.card :=
  C.solver_toAtlas_isClosedOn_states.restrict_dom_card_eq

/-- **The canonical form of a cycle's solver-lifted atlas is closed on the
cycle's state set.** Direct cross-link of the cycle theory (`AdversarialClosedCycle`)
to the canonical atlas algebra (`Atlas.canon`) — composes
`solver_toAtlas_isClosedOn_states` (above) with `Atlas.IsClosedOn.canon_iff`
(Tick 1615). Establishes that every cycle has a **canonical proof artifact**:
its solver lifted to an atlas, then canonicalised at its state set, is
still a closed atlas. This is the M4-form of the cycle: the smallest
representation of the cycle's certificate that retains all the
gameplay-relevant information. -/
theorem solver_toAtlas_canon_isClosedOn_states {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) :
    (C.solver.toAtlas.canon C.states).IsClosedOn cfg C.states :=
  (Atlas.IsClosedOn.canon_iff _ _).mpr C.solver_toAtlas_isClosedOn_states

/-- **First step of incremental M4 construction**: starting from the empty
atlas (which certifies `∅` by `Atlas.IsClosedOn.empty_set`) and unioning
with a cycle's solver-lifted atlas yields a closed atlas on the cycle's
state set. The atomic "absorb one cycle" operation for growing the M4
atlas from zero — produces an atlas certified on `∅ ∪ C.states = C.states`
via `Atlas.IsClosedOn.unionOn` (Tick 1604). -/
theorem empty_unionOn_solver_toAtlas_isClosedOn {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) :
    ((Atlas.empty cfg).unionOn C.solver.toAtlas ∅ C.states).IsClosedOn cfg
      C.states := by
  have h := Atlas.IsClosedOn.unionOn
    (Atlas.IsClosedOn.empty_set (Atlas.empty cfg))
    C.solver_toAtlas_isClosedOn_states
  rw [Finset.empty_union] at h
  exact h

/-- **Two-cycle M4 construction**: the `unionOn` of two cycles'
solver-lifted atlases is closed on the union of their state sets.
The "absorb two cycles" atomic operation — building blocks for the
full M4 atlas via iterated `unionOn`. -/
theorem solver_toAtlas_unionOn_isClosedOn {cfg : GameConfig}
    (C₁ C₂ : AdversarialClosedCycle cfg) :
    ((C₁.solver.toAtlas).unionOn C₂.solver.toAtlas
      C₁.states C₂.states).IsClosedOn cfg (C₁.states ∪ C₂.states) :=
  Atlas.IsClosedOn.unionOn
    C₁.solver_toAtlas_isClosedOn_states
    C₂.solver_toAtlas_isClosedOn_states

/-- **Two-cycle M4 construction (canonical form)**: the `unionOn` of two
cycles' **canonical** atlases (each canonicalised at its own state set)
is closed on the union of their state sets. The minimal-representation
form of the two-cycle absorption — preferred for actual M4 storage as
each piece is in canonical form. -/
theorem solver_toAtlas_canon_unionOn_canon_isClosedOn {cfg : GameConfig}
    (C₁ C₂ : AdversarialClosedCycle cfg) :
    ((C₁.solver.toAtlas.canon C₁.states).unionOn
      (C₂.solver.toAtlas.canon C₂.states)
      C₁.states C₂.states).IsClosedOn cfg (C₁.states ∪ C₂.states) :=
  Atlas.IsClosedOn.unionOn
    C₁.solver_toAtlas_canon_isClosedOn_states
    C₂.solver_toAtlas_canon_isClosedOn_states

/-- **N-cycle M4 construction (fold form)**: left-fold of `unionOn` over a
list of cycles, starting from a base atlas `A` certified on a state set
`S`. Each fold step absorbs one cycle into the accumulator, growing the
certified state set by the cycle's state set. -/
def buildFromCycles {cfg : GameConfig}
    (A : Atlas cfg) (S : Finset GameState)
    (cycles : List (AdversarialClosedCycle cfg)) :
    Atlas cfg × Finset GameState :=
  cycles.foldl
    (fun acc C =>
      (acc.1.unionOn C.solver.toAtlas acc.2 C.states, acc.2 ∪ C.states))
    (A, S)

/-- **N-cycle M4 construction is closed**: the fold produces an atlas
closed on the accumulator's final state set, provided the seed atlas is
closed on its seed state set. By induction on the cycle list — each step
applies `Atlas.IsClosedOn.unionOn` to absorb the next cycle. -/
theorem buildFromCycles_isClosedOn {cfg : GameConfig}
    (A : Atlas cfg) (S : Finset GameState) (hA : A.IsClosedOn cfg S)
    (cycles : List (AdversarialClosedCycle cfg)) :
    (buildFromCycles A S cycles).1.IsClosedOn cfg
      (buildFromCycles A S cycles).2 := by
  induction cycles generalizing A S with
  | nil => exact hA
  | cons C Cs ih =>
    exact ih (A.unionOn C.solver.toAtlas S C.states) (S ∪ C.states)
      (Atlas.IsClosedOn.unionOn hA C.solver_toAtlas_isClosedOn_states)

/-- **N-cycle M4 construction from empty**: starting from `Atlas.empty cfg`
and absorbing all cycles in a list yields an atlas closed on the union of
all cycle state sets (via repeated `unionOn`). The complete inductive
M4 construction primitive. -/
theorem buildFromCycles_empty_isClosedOn {cfg : GameConfig}
    (cycles : List (AdversarialClosedCycle cfg)) :
    (buildFromCycles (Atlas.empty cfg) ∅ cycles).1.IsClosedOn cfg
      (buildFromCycles (Atlas.empty cfg) ∅ cycles).2 :=
  buildFromCycles_isClosedOn _ _ (Atlas.IsClosedOn.empty_set _) cycles

/-- **Closed form for `buildFromCycles`'s state set**: the second component
of the fold is a Finset left-fold accumulating each cycle's states. -/
theorem buildFromCycles_states {cfg : GameConfig}
    (A : Atlas cfg) (S : Finset GameState)
    (cycles : List (AdversarialClosedCycle cfg)) :
    (buildFromCycles A S cycles).2 =
      cycles.foldl (fun acc C => acc ∪ C.states) S := by
  induction cycles generalizing A S with
  | nil => rfl
  | cons C Cs ih => exact ih _ _

/-- **Closed form for `buildFromCycles`'s state set from empty**: the M4
state set is exactly `cycles.foldl (· ∪ ·.states) ∅` — the iterated
union of all cycle state sets. -/
theorem buildFromCycles_empty_states {cfg : GameConfig}
    (cycles : List (AdversarialClosedCycle cfg)) :
    (buildFromCycles (Atlas.empty cfg) ∅ cycles).2 =
      cycles.foldl (fun acc C => acc ∪ C.states) ∅ :=
  buildFromCycles_states _ _ cycles

/-- **Foldl-union monotonicity**: if `g ∈ S` (the seed), then `g` is in
the foldl-union result. Each accumulation step only grows the set via
union, so seed elements survive to the end. -/
theorem foldl_union_mono {cfg : GameConfig} (g : GameState)
    (cycles : List (AdversarialClosedCycle cfg))
    (S : Finset GameState) (h : g ∈ S) :
    g ∈ cycles.foldl (fun acc C => acc ∪ C.states) S := by
  induction cycles generalizing S with
  | nil => exact h
  | cons C Cs ih =>
    exact ih (S ∪ C.states) (Finset.mem_union_left _ h)

/-- **A state in any cycle of the list is in the foldl-union result.**
For each `C ∈ cycles` and `g ∈ C.states`, the foldl-union accumulator
contains `g` at the end (no matter the seed). -/
theorem mem_foldl_union_of_mem_list {cfg : GameConfig}
    (g : GameState) {cycles : List (AdversarialClosedCycle cfg)}
    {C : AdversarialClosedCycle cfg} (hC : C ∈ cycles) (hg : g ∈ C.states)
    (S : Finset GameState) :
    g ∈ cycles.foldl (fun acc C => acc ∪ C.states) S := by
  induction cycles generalizing S with
  | nil => simp at hC
  | cons C' Cs ih =>
    rw [List.mem_cons] at hC
    rcases hC with hCeq | hC_in_Cs
    · subst hCeq
      exact foldl_union_mono g Cs (S ∪ C.states)
        (Finset.mem_union_right S hg)
    · exact ih hC_in_Cs (S ∪ C'.states)

/-- **Init membership in M4 atlas state set**: if any cycle in the list
contains `init`, then `init ∈ (buildFromCycles ... cycles).2`. The
operational link from M3 (init-cycle existence) to M4 (init in the M4
atlas's certified state set). -/
theorem init_mem_buildFromCycles_states_of_exists_init_cycle {cfg : GameConfig}
    {cycles : List (AdversarialClosedCycle cfg)}
    (h : ∃ C ∈ cycles, GameState.init ∈ C.states) :
    GameState.init ∈ (buildFromCycles (Atlas.empty cfg) ∅ cycles).2 := by
  rw [buildFromCycles_empty_states]
  obtain ⟨C, hCmem, hCinit⟩ := h
  exact mem_foldl_union_of_mem_list GameState.init hCmem hCinit ∅

/-- **Bundled M3→M4 artifact**: given any cycle list containing an
init-cycle witness, produce the M4 artifact — the constructed atlas
together with proofs that `init` is in its certified state set AND the
atlas is closed on that state set. The single-theorem package of the
M3→M4 transition: input is a list of cycle candidates with at least one
through `init`, output is the M4 atlas + completeness witness. -/
theorem buildFromCycles_m4_artifact {cfg : GameConfig}
    {cycles : List (AdversarialClosedCycle cfg)}
    (h : ∃ C ∈ cycles, GameState.init ∈ C.states) :
    GameState.init ∈ (buildFromCycles (Atlas.empty cfg) ∅ cycles).2 ∧
    (buildFromCycles (Atlas.empty cfg) ∅ cycles).1.IsClosedOn cfg
      (buildFromCycles (Atlas.empty cfg) ∅ cycles).2 :=
  ⟨init_mem_buildFromCycles_states_of_exists_init_cycle h,
   buildFromCycles_empty_isClosedOn cycles⟩

/-- **Single-cycle M4 artifact**: a single init-cycle directly yields an
M4 artifact. The minimal-input form: one cycle witness `C` with
`init ∈ C.states` suffices to produce the certified atlas via the
singleton cycle list `[C]`. -/
theorem singleton_m4_artifact {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    GameState.init ∈ (buildFromCycles (Atlas.empty cfg) ∅ [C]).2 ∧
    (buildFromCycles (Atlas.empty cfg) ∅ [C]).1.IsClosedOn cfg
      (buildFromCycles (Atlas.empty cfg) ∅ [C]).2 :=
  buildFromCycles_m4_artifact ⟨C, List.mem_singleton.mpr rfl, hinit⟩

/-- **`buildFromCycles` is associative under list concatenation**: building
from `cycles1 ++ cycles2` equals building from `cycles1` then continuing
with `cycles2` from the result. The fundamental "build in chunks"
property — M4 atlas construction can be split across multiple invocations
and chained without loss. -/
theorem buildFromCycles_append {cfg : GameConfig}
    (A : Atlas cfg) (S : Finset GameState)
    (cycles1 cycles2 : List (AdversarialClosedCycle cfg)) :
    buildFromCycles A S (cycles1 ++ cycles2) =
      buildFromCycles
        (buildFromCycles A S cycles1).1
        (buildFromCycles A S cycles1).2
        cycles2 := by
  induction cycles1 generalizing A S with
  | nil => rfl
  | cons C Cs ih => exact ih _ _

/-- **Canonical-form M4 construction**: like `buildFromCycles` but uses
each cycle's *canonical* form (`solver.toAtlas.canon C.states`) instead
of the raw `solver.toAtlas`. Produces a more compact atlas where each
cycle contributes only its minimal certificate (the active-query
restriction). -/
def buildFromCyclesCanon {cfg : GameConfig}
    (A : Atlas cfg) (S : Finset GameState)
    (cycles : List (AdversarialClosedCycle cfg)) :
    Atlas cfg × Finset GameState :=
  cycles.foldl
    (fun acc C =>
      (acc.1.unionOn (C.solver.toAtlas.canon C.states) acc.2 C.states,
       acc.2 ∪ C.states))
    (A, S)

/-- **Canonical-form M4 construction is closed**: analog of
`buildFromCycles_isClosedOn` for the canon variant, using
`solver_toAtlas_canon_isClosedOn_states` (Tick 1621) instead of
`solver_toAtlas_isClosedOn_states` (Tick 1601). -/
theorem buildFromCyclesCanon_isClosedOn {cfg : GameConfig}
    (A : Atlas cfg) (S : Finset GameState) (hA : A.IsClosedOn cfg S)
    (cycles : List (AdversarialClosedCycle cfg)) :
    (buildFromCyclesCanon A S cycles).1.IsClosedOn cfg
      (buildFromCyclesCanon A S cycles).2 := by
  induction cycles generalizing A S with
  | nil => exact hA
  | cons C Cs ih =>
    exact ih (A.unionOn (C.solver.toAtlas.canon C.states) S C.states)
      (S ∪ C.states)
      (Atlas.IsClosedOn.unionOn hA C.solver_toAtlas_canon_isClosedOn_states)

/-- **Canonical-form state set**: identical to the raw form's state set
(the second component of the fold doesn't depend on the atlas content).
Captures that canon vs raw differs only in the atlas content. -/
theorem buildFromCyclesCanon_states {cfg : GameConfig}
    (A : Atlas cfg) (S : Finset GameState)
    (cycles : List (AdversarialClosedCycle cfg)) :
    (buildFromCyclesCanon A S cycles).2 =
      cycles.foldl (fun acc C => acc ∪ C.states) S := by
  induction cycles generalizing A S with
  | nil => rfl
  | cons C Cs ih => exact ih _ _

/-- **Canon variant: init membership**. If any cycle in the list contains
`init`, then `init ∈ (buildFromCyclesCanon ... cycles).2`. The canon
variant's state set equals the raw variant's, so init membership lifts
via the same `mem_foldl_union_of_mem_list` lemma. -/
theorem init_mem_buildFromCyclesCanon_states_of_exists_init_cycle
    {cfg : GameConfig}
    {cycles : List (AdversarialClosedCycle cfg)}
    (h : ∃ C ∈ cycles, GameState.init ∈ C.states) :
    GameState.init ∈
      (buildFromCyclesCanon (Atlas.empty cfg) ∅ cycles).2 := by
  rw [buildFromCyclesCanon_states]
  obtain ⟨C, hCmem, hCinit⟩ := h
  exact mem_foldl_union_of_mem_list GameState.init hCmem hCinit ∅

/-- **Canon variant: bundled M3→M4 artifact**. The canonical-form M4
construction yields both init membership and closure in a single bundle. -/
theorem buildFromCyclesCanon_m4_artifact {cfg : GameConfig}
    {cycles : List (AdversarialClosedCycle cfg)}
    (h : ∃ C ∈ cycles, GameState.init ∈ C.states) :
    GameState.init ∈ (buildFromCyclesCanon (Atlas.empty cfg) ∅ cycles).2 ∧
    (buildFromCyclesCanon (Atlas.empty cfg) ∅ cycles).1.IsClosedOn cfg
      (buildFromCyclesCanon (Atlas.empty cfg) ∅ cycles).2 :=
  ⟨init_mem_buildFromCyclesCanon_states_of_exists_init_cycle h,
   buildFromCyclesCanon_isClosedOn _ _
     (Atlas.IsClosedOn.empty_set _) cycles⟩

/-- **Canon variant: singleton M4 artifact**. The minimal-input form for
canon construction — one cycle `C` with `init ∈ C.states` directly yields
the canonical M4 artifact via singleton list. -/
theorem singleton_canon_m4_artifact {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    GameState.init ∈ (buildFromCyclesCanon (Atlas.empty cfg) ∅ [C]).2 ∧
    (buildFromCyclesCanon (Atlas.empty cfg) ∅ [C]).1.IsClosedOn cfg
      (buildFromCyclesCanon (Atlas.empty cfg) ∅ [C]).2 :=
  buildFromCyclesCanon_m4_artifact ⟨C, List.mem_singleton.mpr rfl, hinit⟩

/-- **Canon variant associativity**: `buildFromCyclesCanon` is associative
under list concatenation. The canonical-form M4 construction can be split
into chunks and re-combined without loss. Same proof structure as
`buildFromCycles_append` (Tick 1631). -/
theorem buildFromCyclesCanon_append {cfg : GameConfig}
    (A : Atlas cfg) (S : Finset GameState)
    (cycles1 cycles2 : List (AdversarialClosedCycle cfg)) :
    buildFromCyclesCanon A S (cycles1 ++ cycles2) =
      buildFromCyclesCanon
        (buildFromCyclesCanon A S cycles1).1
        (buildFromCyclesCanon A S cycles1).2
        cycles2 := by
  induction cycles1 generalizing A S with
  | nil => rfl
  | cons C Cs ih => exact ih _ _

/-- **Canon and raw variants have the same state set**:
`(buildFromCyclesCanon A S cycles).2 = (buildFromCycles A S cycles).2`.
Both fold the same union-accumulation function over the same cycle list,
producing identical state sets — only the atlas content differs. -/
theorem buildFromCyclesCanon_states_eq_buildFromCycles_states
    {cfg : GameConfig} (A : Atlas cfg) (S : Finset GameState)
    (cycles : List (AdversarialClosedCycle cfg)) :
    (buildFromCyclesCanon A S cycles).2 = (buildFromCycles A S cycles).2 := by
  rw [buildFromCyclesCanon_states, buildFromCycles_states]

/-- **Singleton canon-form atlas characterisation**:
`(buildFromCyclesCanon (Atlas.empty cfg) ∅ [C]).1 = C.solver.toAtlas.canon C.states`.
The canon M4 construction for a single cycle is exactly the cycle's
canonical atlas (no extra wrapping). Proof: unfold `buildFromCyclesCanon`,
apply `unionOn_empty_left` to reduce to `restrict`, then `restrict_canon`
collapses to just the canon. -/
theorem singleton_canon_atlas {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) :
    (buildFromCyclesCanon (Atlas.empty cfg) ∅ [C]).1 =
      C.solver.toAtlas.canon C.states := by
  change (Atlas.empty cfg).unionOn (C.solver.toAtlas.canon C.states)
        ∅ C.states = C.solver.toAtlas.canon C.states
  rw [Atlas.unionOn_empty_left, Atlas.restrict_canon]

/-- **Singleton raw-form atlas characterisation**:
`(buildFromCycles (Atlas.empty cfg) ∅ [C]).1 = C.solver.toAtlas.restrict C.states`.
The raw M4 construction for a single cycle is the cycle's solver-lifted
atlas, restricted to the cycle's state set. -/
theorem singleton_raw_atlas {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) :
    (buildFromCycles (Atlas.empty cfg) ∅ [C]).1 =
      C.solver.toAtlas.restrict C.states := by
  change (Atlas.empty cfg).unionOn C.solver.toAtlas ∅ C.states =
        C.solver.toAtlas.restrict C.states
  rw [Atlas.unionOn_empty_left]

/-- **Canon of raw build equals canon build from canonicalised seed**: the
canonicalisation of the raw M4 construction's atlas equals the canon
variant's atlas starting from `A.canon S` instead of `A`. By induction on
`cycles` with `canon_unionOn` (Tick 1618) distribution at each step. -/
theorem canon_buildFromCycles_eq_buildFromCyclesCanon_canon {cfg : GameConfig}
    (A : Atlas cfg) (S : Finset GameState)
    (cycles : List (AdversarialClosedCycle cfg)) :
    ((buildFromCycles A S cycles).1).canon
        ((buildFromCycles A S cycles).2) =
      (buildFromCyclesCanon (A.canon S) S cycles).1 := by
  induction cycles generalizing A S with
  | nil => rfl
  | cons C Cs ih =>
    have h := ih (A.unionOn C.solver.toAtlas S C.states) (S ∪ C.states)
    rw [Atlas.canon_unionOn] at h
    exact h

/-- **Empty-seed corollary**: starting from `Atlas.empty cfg` and `∅`,
canonicalising the raw M4 atlas equals the canon variant's atlas directly.
The cleanest form for M4 implementations starting from scratch. -/
theorem canon_buildFromCycles_empty_eq_buildFromCyclesCanon_empty
    {cfg : GameConfig}
    (cycles : List (AdversarialClosedCycle cfg)) :
    ((buildFromCycles (Atlas.empty cfg) ∅ cycles).1).canon
        ((buildFromCycles (Atlas.empty cfg) ∅ cycles).2) =
      (buildFromCyclesCanon (Atlas.empty cfg) ∅ cycles).1 := by
  rw [canon_buildFromCycles_eq_buildFromCyclesCanon_canon, Atlas.empty_canon]

/-- **Canon-form M4 idempotence**: applying `canon` to an already-canonical
M4 atlas at its own state set is a no-op. The canon variant is, well,
canonical — applying canonicalisation again doesn't change it. Proof:
substitute the canon atlas via Tick 1638, use Tick 1635 to align state
sets, then `canon_canon` (Tick 1616) collapses the double canon. -/
theorem buildFromCyclesCanon_empty_canon_eq_self {cfg : GameConfig}
    (cycles : List (AdversarialClosedCycle cfg)) :
    ((buildFromCyclesCanon (Atlas.empty cfg) ∅ cycles).1).canon
        ((buildFromCyclesCanon (Atlas.empty cfg) ∅ cycles).2) =
      (buildFromCyclesCanon (Atlas.empty cfg) ∅ cycles).1 := by
  conv_lhs => rw [← canon_buildFromCycles_empty_eq_buildFromCyclesCanon_empty]
  rw [buildFromCyclesCanon_states_eq_buildFromCycles_states, Atlas.canon_canon]
  exact canon_buildFromCycles_empty_eq_buildFromCyclesCanon_empty cycles

/-- **Cycle certificate-equivalence**: two adversarial closed cycles are
*equivalent* if they have the same state set and their solvers agree on
every active query `(g, p)` with `g ∈ states, p ∈ g.bag`. Mirrors
`M4Artifact.Equiv` (Tick 1657): only the active queries matter — what
the solver returns on inactive queries (out of cycle, or pieces not in
bag) is irrelevant to the cycle invariants `not_lost`, `valid`,
`closed`. Captures the precise notion of "carrying the same cycle
certificate" — two cycles can have wildly different out-of-bag/out-of-
states behavior yet be equivalent. -/
def Equiv {cfg : GameConfig}
    (C₁ C₂ : AdversarialClosedCycle cfg) : Prop :=
  C₁.states = C₂.states ∧
  ∀ g ∈ C₁.states, ∀ p ∈ g.bag, C₁.solver g p = C₂.solver g p

/-- **Cycle Equiv is reflexive**: every cycle is equivalent to itself. -/
theorem Equiv.refl {cfg : GameConfig} (C : AdversarialClosedCycle cfg) :
    C.Equiv C :=
  ⟨rfl, fun _ _ _ _ => rfl⟩

/-- **Cycle Equiv is symmetric**: the equivalence relation is preserved
under reversal. -/
theorem Equiv.symm {cfg : GameConfig}
    {C₁ C₂ : AdversarialClosedCycle cfg} (h : C₁.Equiv C₂) :
    C₂.Equiv C₁ := by
  refine ⟨h.1.symm, fun g hg p hp => ?_⟩
  exact (h.2 g (h.1.symm ▸ hg) p hp).symm

/-- **Cycle Equiv is transitive**: certificate-equivalence chains
through intermediate cycles. Together with `refl` and `symm`, confirms
`AdversarialClosedCycle.Equiv` is an equivalence relation. -/
theorem Equiv.trans {cfg : GameConfig}
    {C₁ C₂ C₃ : AdversarialClosedCycle cfg}
    (h₁₂ : C₁.Equiv C₂) (h₂₃ : C₂.Equiv C₃) :
    C₁.Equiv C₃ := by
  refine ⟨h₁₂.1.trans h₂₃.1, fun g hg p hp => ?_⟩
  exact (h₁₂.2 g hg p hp).trans (h₂₃.2 g (h₁₂.1 ▸ hg) p hp)

/-- **`AdversarialClosedCycle` Setoid instance**: registers
`AdversarialClosedCycle.Equiv` as the canonical equivalence relation on
cycles. Enables `Quotient (AdversarialClosedCycle.setoid cfg)` — the
type of cycle-equivalence classes. Mirrors `M4Artifact.setoid` at the
cycle level. -/
instance setoid (cfg : GameConfig) :
    Setoid (AdversarialClosedCycle cfg) where
  r := Equiv
  iseqv := ⟨Equiv.refl, Equiv.symm, Equiv.trans⟩

/-- **Cycle Equiv lifts to canon-atlas equality**: equivalent cycles
`C₁ ≈ C₂` produce **identical** canonicalised solver atlases:
`C₁.solver.toAtlas.canon C₁.states = C₂.solver.toAtlas.canon C₂.states`.
Stronger than Tick 1678 / Tick 1679 (which give only M4 Equiv) — this
is a direct atlas-level equality. Does not require `init ∈ Cᵢ.states`,
so applies to arbitrary cycle equivalences. Pointwise proof:
- *Active* `(g, p) ∈ C₁.states × g.bag` (also active for `C₂` via
  `heq.1`): both sides reduce to `Cᵢ.solver.toAtlas g p = some (Cᵢ.solver
  g p)`, equal by `heq.2`.
- *Inactive*: both sides yield `none` via `canon_apply_of_inactive`,
  with the C₁/C₂ inactivity equivalence following from `heq.1`. -/
theorem solver_toAtlas_canon_eq_of_equiv {cfg : GameConfig}
    {C₁ C₂ : AdversarialClosedCycle cfg} (heq : C₁.Equiv C₂) :
    C₁.solver.toAtlas.canon C₁.states =
      C₂.solver.toAtlas.canon C₂.states := by
  funext g p
  by_cases h : g ∈ C₁.states ∧ p ∈ g.bag
  · have h₂ : g ∈ C₂.states ∧ p ∈ g.bag := ⟨heq.1 ▸ h.1, h.2⟩
    rw [Atlas.canon_apply_of_active _ h.1 h.2,
        Atlas.canon_apply_of_active _ h₂.1 h₂.2]
    unfold Solver.toAtlas
    rw [heq.2 g h.1 p h.2]
  · have h₂ : ¬ (g ∈ C₂.states ∧ p ∈ g.bag) := by
      intro ⟨h₂_g, h₂_p⟩
      exact h ⟨heq.1.symm ▸ h₂_g, h₂_p⟩
    rw [Atlas.canon_apply_of_inactive _ h,
        Atlas.canon_apply_of_inactive _ h₂]

/-- **Cycle's canonical atlas `none` ↔ inactive**: for any cycle C,
`(C.solver.toAtlas.canon C.states) g p = none ↔ ¬(g ∈ C.states ∧ p
∈ g.bag)`. Cycle-side parallel to Tick 1761 (`M4Artifact.canon_atlas_eq_none_iff_inactive`).

Proof: forward — at active, `canon_apply_of_active` reduces canon
to `Solver.toAtlas`, which is always `some _`, contradicting `none`.
Reverse — at inactive, `canon_apply_of_inactive` directly. -/
theorem solver_toAtlas_canon_eq_none_iff_inactive {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g : GameState} {p : Piece} :
    (C.solver.toAtlas.canon C.states) g p = none ↔
      ¬(g ∈ C.states ∧ p ∈ g.bag) := by
  refine ⟨fun h hac => ?_, fun hi => Atlas.canon_apply_of_inactive _ hi⟩
  obtain ⟨hg, hp⟩ := hac
  rw [Atlas.canon_apply_of_active _ hg hp] at h
  unfold Solver.toAtlas at h
  exact Option.some_ne_none _ h

/-- **Cycle's canonical atlas `isSome` ↔ active**: contrapositive of
Tick 1763. The cycle's canonical atlas is `some _` exactly at active
queries. Derived via `Option.isSome_iff_ne_none` + Tick 1763's
characterization + `not_not`. -/
theorem solver_toAtlas_canon_isSome_iff_active {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g : GameState} {p : Piece} :
    ((C.solver.toAtlas.canon C.states) g p).isSome ↔
      g ∈ C.states ∧ p ∈ g.bag := by
  rw [Option.isSome_iff_ne_none, ne_eq,
      solver_toAtlas_canon_eq_none_iff_inactive, not_not]

/-- **Cycle Equiv from canon-atlas equality**: reverse direction of Tick
1680. If two cycles have equal state sets AND equal canon solver
atlases, then the cycles are equivalent. The state-set hypothesis is
necessary — canon atlas equality alone does not imply states equality
(states can differ at states with empty bags, where canon is `none`
either way). Together with Tick 1680, gives the full
canon-data ↔ cycle Equiv equivalence. Parallel to Tick 1662's
`M4Artifact.equiv_of_canon_atlas_eq`. -/
theorem equiv_of_canon_atlas_eq {cfg : GameConfig}
    {C₁ C₂ : AdversarialClosedCycle cfg}
    (hstates : C₁.states = C₂.states)
    (hatlas : C₁.solver.toAtlas.canon C₁.states =
              C₂.solver.toAtlas.canon C₂.states) :
    C₁.Equiv C₂ := by
  refine ⟨hstates, fun g hg p hp => ?_⟩
  have h₁ : (C₁.solver.toAtlas.canon C₁.states) g p =
            C₁.solver.toAtlas g p :=
    Atlas.canon_apply_of_active _ hg hp
  have hg₂ : g ∈ C₂.states := hstates ▸ hg
  have h₂ : (C₂.solver.toAtlas.canon C₂.states) g p =
            C₂.solver.toAtlas g p :=
    Atlas.canon_apply_of_active _ hg₂ hp
  have hsome : C₁.solver.toAtlas g p = C₂.solver.toAtlas g p := by
    rw [← h₁, ← h₂, hatlas]
  exact Option.some.inj hsome

/-- **Cycle Equiv ↔ canon-atlas equality** (full biconditional). Two
cycles are equivalent iff their state sets and canon solver atlases
both match. The complete canonical-form characterisation of cycle
equivalence — parallel to Tick 1663's
`M4Artifact.equiv_iff_canon_atlas_eq`. -/
theorem equiv_iff_canon_atlas_eq {cfg : GameConfig}
    (C₁ C₂ : AdversarialClosedCycle cfg) :
    C₁.Equiv C₂ ↔
      C₁.states = C₂.states ∧
      C₁.solver.toAtlas.canon C₁.states =
        C₂.solver.toAtlas.canon C₂.states :=
  ⟨fun h => ⟨h.1, solver_toAtlas_canon_eq_of_equiv h⟩,
   fun ⟨hs, ha⟩ => equiv_of_canon_atlas_eq hs ha⟩

/-- **Cycles are determined by states + solver**: `C = C' ↔ C.states =
C'.states ∧ C.solver = C'.solver`. Parallel to Tick 1752's
`M4Artifact.eq_iff_states_eq_and_atlas_eq`. The Prop fields
(`not_lost`, `valid`, `closed`) are irrelevant via proof irrelevance;
equality of the two data fields suffices for full cycle struct
equality. -/
theorem eq_iff_states_eq_and_solver_eq {cfg : GameConfig}
    {C C' : AdversarialClosedCycle cfg} :
    C = C' ↔ C.states = C'.states ∧ C.solver = C'.solver := by
  refine ⟨fun h => ?_, fun ⟨hs, hσ⟩ => ?_⟩
  · subst h; exact ⟨rfl, rfl⟩
  · obtain ⟨s, σ, nl, v, cl⟩ := C
    obtain ⟨s', σ', nl', v', cl'⟩ := C'
    subst hs
    subst hσ
    rfl

/-- An **M4 artifact**: the bundled certified-survival proof object. A
closed atlas on a state set that contains `init`. This is the M4 final
form per `AGENTS.md`: a lookup table (atlas) that lets any reachable game
state look up its move and stay in the certified domain forever. -/
structure M4Artifact (cfg : GameConfig) where
  atlas : Atlas cfg
  states : Finset GameState
  init_mem : GameState.init ∈ states
  closed : atlas.IsClosedOn cfg states

/-- **Raw-form M4 artifact constructor** from a cycle list with init-cycle
witness. -/
def buildFromCycles_M4Artifact {cfg : GameConfig}
    (cycles : List (AdversarialClosedCycle cfg))
    (h : ∃ C ∈ cycles, GameState.init ∈ C.states) : M4Artifact cfg where
  atlas := (buildFromCycles (Atlas.empty cfg) ∅ cycles).1
  states := (buildFromCycles (Atlas.empty cfg) ∅ cycles).2
  init_mem := init_mem_buildFromCycles_states_of_exists_init_cycle h
  closed := buildFromCycles_empty_isClosedOn cycles

/-- **Canon-form M4 artifact constructor** from a cycle list with init-cycle
witness. The canon variant produces a minimal-storage artifact. -/
def buildFromCyclesCanon_M4Artifact {cfg : GameConfig}
    (cycles : List (AdversarialClosedCycle cfg))
    (h : ∃ C ∈ cycles, GameState.init ∈ C.states) : M4Artifact cfg where
  atlas := (buildFromCyclesCanon (Atlas.empty cfg) ∅ cycles).1
  states := (buildFromCyclesCanon (Atlas.empty cfg) ∅ cycles).2
  init_mem := init_mem_buildFromCyclesCanon_states_of_exists_init_cycle h
  closed := buildFromCyclesCanon_isClosedOn _ _ (Atlas.IsClosedOn.empty_set _) cycles

/-- **M4 artifact existence from cycle existence**: any single init-cycle
witness suffices to produce an `M4Artifact cfg`. The cycle is wrapped in
a singleton list and passed to the raw-form constructor. -/
theorem nonempty_M4Artifact_of_exists_init_cycle {cfg : GameConfig}
    (h : ∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states) :
    Nonempty (M4Artifact cfg) := by
  obtain ⟨C, hinit⟩ := h
  exact ⟨buildFromCycles_M4Artifact [C] ⟨C, List.mem_singleton.mpr rfl, hinit⟩⟩

/-- **Direct cycle → M4Artifact (raw form)**: from any init-cycle, directly
construct the corresponding M4 artifact whose atlas is the cycle's
solver-lifted atlas. Cleaner than the `buildFromCycles_M4Artifact [C]`
form when only one cycle is involved. -/
def M4Artifact.fromCycle {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    M4Artifact cfg where
  atlas := C.solver.toAtlas
  states := C.states
  init_mem := hinit
  closed := C.solver_toAtlas_isClosedOn_states

/-- **`fromCycle` state-set projection**: `(fromCycle C h).states = C.states`.
Direct `rfl` from the definition. -/
@[simp] theorem M4Artifact.fromCycle_states {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    (M4Artifact.fromCycle C h).states = C.states := rfl

/-- **`fromCycle` atlas projection**: `(fromCycle C h).atlas = C.solver.toAtlas`.
Direct `rfl` from the definition. -/
@[simp] theorem M4Artifact.fromCycle_atlas {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    (M4Artifact.fromCycle C h).atlas = C.solver.toAtlas := rfl

/-- **Direct cycle → M4Artifact (canonical form)**: same as `fromCycle` but
uses the canon atlas form, yielding a minimal-storage M4 artifact
directly. -/
def M4Artifact.fromCycleCanon {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    M4Artifact cfg where
  atlas := C.solver.toAtlas.canon C.states
  states := C.states
  init_mem := hinit
  closed := C.solver_toAtlas_canon_isClosedOn_states

/-- **`fromCycleCanon` state-set projection**: `(fromCycleCanon C h).states
= C.states`. Direct `rfl` from the definition. -/
@[simp] theorem M4Artifact.fromCycleCanon_states {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    (M4Artifact.fromCycleCanon C h).states = C.states := rfl

/-- **`fromCycleCanon` atlas projection**: `(fromCycleCanon C h).atlas =
C.solver.toAtlas.canon C.states`. Direct `rfl` from the definition. -/
@[simp] theorem M4Artifact.fromCycleCanon_atlas {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    (M4Artifact.fromCycleCanon C h).atlas =
      C.solver.toAtlas.canon C.states := rfl

/-- **Direct closed-atlas → M4Artifact**: bundles an `IsClosedOn`
witness containing `init` into an M4 artifact. The most direct
construction — no cycle/solver wrapping needed. Useful when the
atlas is already in M4-relevant form. Named `ofIsClosedOn` to avoid
clashing with `AdversarialClosedCycle.ofClosedAtlas`. -/
def M4Artifact.ofIsClosedOn {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    M4Artifact cfg where
  atlas := A
  states := S
  init_mem := hinit
  closed := h

/-- **`ofIsClosedOn` preserves `states`**: simp accessor lemma. -/
@[simp] theorem M4Artifact.ofIsClosedOn_states {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    (M4Artifact.ofIsClosedOn h hinit).states = S := rfl

/-- **`ofIsClosedOn` preserves `atlas`**: simp accessor lemma. -/
@[simp] theorem M4Artifact.ofIsClosedOn_atlas {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    (M4Artifact.ofIsClosedOn h hinit).atlas = A := rfl

/-- **`ofIsClosedOn` atlas toSolver passthrough**:
`(M4Artifact.ofIsClosedOn h hinit).atlas.toSolver = A.toSolver`.
Simp-style lemma chaining the atlas accessor (Tick 1860) with
`Atlas.toSolver` — useful in materialised-solver reasoning where
the constructor's atlas needs to be exposed at the solver level. -/
@[simp] theorem M4Artifact.ofIsClosedOn_atlas_toSolver {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    (M4Artifact.ofIsClosedOn h hinit).atlas.toSolver = A.toSolver := rfl

/-- **`fromCycle` agrees with `ofIsClosedOn` on cycle-derived
inputs**: when the input atlas comes from a cycle's solver-lifted
form, the two M4 constructors produce identical artifacts. Direct
via `rfl` — both constructors fill all four fields with the same
values (atlas = `C.solver.toAtlas`, states = `C.states`, init_mem =
hinit, closed = `C.solver_toAtlas_isClosedOn_states`). -/
theorem M4Artifact.fromCycle_eq_ofIsClosedOn {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    M4Artifact.fromCycle C h =
      M4Artifact.ofIsClosedOn C.solver_toAtlas_isClosedOn_states h := rfl

/-- **`fromCycleCanon` agrees with `ofIsClosedOn` on canon-form
cycle-derived inputs**: parallel to Tick 1861 for the canonical
form. Direct via `rfl` — both constructors fill all four fields
identically (atlas = `C.solver.toAtlas.canon C.states`, states =
`C.states`, init_mem = hinit, closed =
`C.solver_toAtlas_canon_isClosedOn_states`). -/
theorem M4Artifact.fromCycleCanon_eq_ofIsClosedOn {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    M4Artifact.fromCycleCanon C h =
      M4Artifact.ofIsClosedOn C.solver_toAtlas_canon_isClosedOn_states h := rfl


/-- **`init` lives in the M4 artifact's atlas domain**: every M4 artifact's
atlas is defined for the `init` state (across its bag). Direct corollary
of Tick 1611's `IsClosedOn.subset_dom` applied to the artifact's `init_mem`
and `closed` fields. -/
theorem init_mem_dom_of_M4Artifact {cfg : GameConfig}
    (M : M4Artifact cfg) : GameState.init ∈ M.atlas.dom M.states :=
  M.closed.subset_dom M.init_mem

/-- **States subset of atlas domain**: every state in the M4 artifact's
state set is also in the atlas's domain at that state set. Generalises
Tick 1649 to all states (not just `init`). Direct from Tick 1611's
`IsClosedOn.subset_dom`. -/
theorem states_subset_dom_of_M4Artifact {cfg : GameConfig}
    (M : M4Artifact cfg) : M.states ⊆ M.atlas.dom M.states :=
  M.closed.subset_dom

/-- **M4 atlas is total at `init`**: the M4 artifact's atlas is defined
for every piece in `init.bag` (i.e., every standard tetromino). Direct
from the `total` clause of `IsClosedOn` at `init`. -/
theorem isTotalAt_init_of_M4Artifact {cfg : GameConfig}
    (M : M4Artifact cfg) : M.atlas.IsTotalAt GameState.init :=
  M.closed.total GameState.init M.init_mem

/-- **M4 atlas trace stays in the certified state set**: under any legal
piece sequence, the trace generated by the M4 artifact's materialised
solver stays within the certified state set. By induction on the step
count, using the closure clause of `IsClosedOn` at each step. -/
theorem M4_atlas_trace_mem_states {cfg : GameConfig} (M : M4Artifact cfg)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg M.atlas.toSolver s GameState.init n ∈ M.states := by
  induction n with
  | zero => exact M.init_mem
  | succ k ih =>
    rw [adversarialTrace_succ]
    set g := adversarialTrace cfg M.atlas.toSolver s GameState.init k with hg_def
    have hg : g ∈ M.states := ih
    have hp : s k ∈ g.bag := by
      have := hl.canDraw k
      rw [Bag.canDraw_iff_mem,
          ← adversarialTrace_bag cfg M.atlas.toSolver s k] at this
      exact this
    have htot := M.closed.total g hg (s k) hp
    obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp htot
    rw [Atlas.toSolver_apply_of_some hpl]
    exact M.closed.closed g hg (s k) hp pl hpl

/-- **M4 atlas's solver `SolvesTetris cfg`**: the M4 artifact's
materialised solver is provably non-losing under any legal piece
sequence. The atlas-as-proof-artifact direct consequence — given an
M4 artifact, the canonical solver derived from it actually wins
Tetris from `init`. -/
theorem solvesTetris_of_M4Artifact {cfg : GameConfig} (M : M4Artifact cfg) :
    SolvesTetris cfg M.atlas.toSolver := by
  intro s hl n
  exact M.closed.not_lost _ (M4_atlas_trace_mem_states M hl n)

/-- **The atlas-as-proof-artifact theorem (explicit form)**: from any M4
artifact, *directly* extract its materialised solver as a
`TetrisSolvableFor cfg` witness. Unlike Tick 1644 (which routes through
cycle existence), this is a *direct* extraction — the solver IS the
atlas materialised. The atlas literally is the proof. -/
theorem tetrisSolvableFor_of_M4Artifact {cfg : GameConfig}
    (M : M4Artifact cfg) : TetrisSolvableFor cfg :=
  ⟨M.atlas.toSolver, solvesTetris_of_M4Artifact M⟩

/-- **M4 artifact union** — combine two M4 artifacts into one whose
atlas is the priority-dispatch union and state set is the Finset union.
The closure transfers via `Atlas.IsClosedOn.unionOn` (Tick 1604); init
membership transfers via `Finset.mem_union_left` from `M₁.init_mem`. -/
def M4Artifact.union {cfg : GameConfig} (M₁ M₂ : M4Artifact cfg) :
    M4Artifact cfg where
  atlas := M₁.atlas.unionOn M₂.atlas M₁.states M₂.states
  states := M₁.states ∪ M₂.states
  init_mem := Finset.mem_union_left _ M₁.init_mem
  closed := Atlas.IsClosedOn.unionOn M₁.closed M₂.closed

/-- **`M4Artifact.union` state-set projection**: `(M₁.union M₂).states =
M₁.states ∪ M₂.states`. Direct `rfl` from the definition. Useful as a
`@[simp]` lemma in proofs that need to rewrite the union's state set. -/
@[simp] theorem M4Artifact.union_states {cfg : GameConfig}
    (M₁ M₂ : M4Artifact cfg) :
    (M₁.union M₂).states = M₁.states ∪ M₂.states := rfl

/-- **`M4Artifact.union` states cardinality upper bound**: by
`Finset.card_union_le`, `(M₁.union M₂).states.card ≤ M₁.states.card
+ M₂.states.card`. Useful upper-bound storage estimate for M4 union
constructions. Direct via `M4Artifact.union_states` (Tick 1934) +
`Finset.card_union_le`. -/
theorem M4Artifact.union_states_card_le {cfg : GameConfig}
    (M₁ M₂ : M4Artifact cfg) :
    (M₁.union M₂).states.card ≤ M₁.states.card + M₂.states.card := by
  rw [M4Artifact.union_states]
  exact Finset.card_union_le _ _

/-- **`M4Artifact.union` states cardinality equality for disjoint
states**: when `M₁.states` and `M₂.states` are disjoint,
`(M₁.union M₂).states.card = M₁.states.card + M₂.states.card`.
Tight version of Tick 1966 — bound is achieved exactly when no
overlap. Direct via `M4Artifact.union_states` (Tick 1934) +
`Finset.card_union_of_disjoint`. -/
theorem M4Artifact.union_states_card_of_disjoint {cfg : GameConfig}
    (M₁ M₂ : M4Artifact cfg) (hdisj : Disjoint M₁.states M₂.states) :
    (M₁.union M₂).states.card = M₁.states.card + M₂.states.card := by
  rw [M4Artifact.union_states]
  exact Finset.card_union_of_disjoint hdisj

/-- **`M4Artifact.union` self-union states cardinality**: `(M.union
M).states.card = M.states.card`. Direct via `union_states` (Tick
1934) + `Finset.union_self`. Useful for proofs that need self-union
M4 storage equality. -/
theorem M4Artifact.union_self_states_card_eq {cfg : GameConfig}
    (M : M4Artifact cfg) :
    (M.union M).states.card = M.states.card := by
  rw [M4Artifact.union_states, Finset.union_self]

/-- **`M4Artifact.union` atlas projection**: `(M₁.union M₂).atlas =
M₁.atlas.unionOn M₂.atlas M₁.states M₂.states`. Direct `rfl` from
the definition. -/
@[simp] theorem M4Artifact.union_atlas {cfg : GameConfig}
    (M₁ M₂ : M4Artifact cfg) :
    (M₁.union M₂).atlas =
      M₁.atlas.unionOn M₂.atlas M₁.states M₂.states := rfl

/-- **M4 artifact canonicalisation**: same state set + init, atlas
canonicalised at the state set (stripped of inactive-query values).
The closure preserves via Tick 1615's `Atlas.IsClosedOn.canon_iff`.
Produces the minimal-storage form of an M4 artifact without changing
the certified content. -/
def M4Artifact.canon {cfg : GameConfig} (M : M4Artifact cfg) :
    M4Artifact cfg where
  atlas := M.atlas.canon M.states
  states := M.states
  init_mem := M.init_mem
  closed := (Atlas.IsClosedOn.canon_iff M.atlas M.states).mpr M.closed

/-- **`M4Artifact.canon` state-set projection**: `M.canon.states = M.states`.
Direct `rfl` from the definition. -/
@[simp] theorem M4Artifact.canon_states {cfg : GameConfig} (M : M4Artifact cfg) :
    M.canon.states = M.states := rfl

/-- **`M4Artifact.canon` atlas projection**: `M.canon.atlas = M.atlas.canon M.states`.
Direct `rfl` from the definition. -/
@[simp] theorem M4Artifact.canon_atlas {cfg : GameConfig} (M : M4Artifact cfg) :
    M.canon.atlas = M.atlas.canon M.states := rfl

/-- **`M4Artifact.union.canon` states cardinality upper bound**:
`(M.union M').canon.states.card ≤ M.states.card + M'.states.card`.
Canonicalisation preserves states (`canon_states` is rfl), so this
is definitionally Tick 1966's bound for the underlying union.
Useful for canonical-form storage estimates of M4 unions. -/
theorem M4Artifact.union_canon_states_card_le {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    (M.union M').canon.states.card ≤ M.states.card + M'.states.card :=
  M.union_states_card_le M'

/-- **`M4Artifact.union.canon` states cardinality equality for
disjoint states**: tight version of Tick 1969. Equality holds when
states are disjoint. -/
theorem M4Artifact.union_canon_states_card_of_disjoint {cfg : GameConfig}
    (M M' : M4Artifact cfg) (hdisj : Disjoint M.states M'.states) :
    (M.union M').canon.states.card = M.states.card + M'.states.card :=
  M.union_states_card_of_disjoint M' hdisj

/-- **M4 artifact canon idempotence (atlas-level)**: applying `canon`
twice to an M4 artifact yields the same atlas as one application.
Lifts Tick 1616's `Atlas.canon_canon` to the artifact level. -/
theorem M4Artifact.canon_canon_atlas {cfg : GameConfig} (M : M4Artifact cfg) :
    (M.canon.canon).atlas = M.canon.atlas :=
  Atlas.canon_canon _ _

/-- **M4 artifact canon idempotence (state-set-level)**: the state set is
preserved through double canonicalisation. -/
theorem M4Artifact.canon_canon_states {cfg : GameConfig} (M : M4Artifact cfg) :
    (M.canon.canon).states = M.canon.states := rfl

/-- **`(ofIsClosedOn h hinit).canon = ofIsClosedOn (canon_iff.mpr
h) hinit`**: canonicalising an `ofIsClosedOn`-constructed M4
artifact equals constructing one from the canon-closure of the
atlas. Direct via `rfl` — both constructors land on identical
fields (atlas = `A.canon S`, states = `S`, init_mem = hinit,
closed = `(canon_iff A S).mpr h`). -/
theorem M4Artifact.ofIsClosedOn_canon {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    (M4Artifact.ofIsClosedOn h hinit).canon =
      M4Artifact.ofIsClosedOn ((Atlas.IsClosedOn.canon_iff A S).mpr h) hinit
    := rfl

/-- **`(ofIsClosedOn h₁ hinit₁).union (ofIsClosedOn h₂ hinit₂) =
ofIsClosedOn (IsClosedOn.unionOn h₁ h₂) (mem_union_left hinit₁)`**:
the union of two `ofIsClosedOn`-constructed artifacts equals an
`ofIsClosedOn` built from the union-closure. Direct via `rfl` —
both constructors fill identical fields. -/
theorem M4Artifact.ofIsClosedOn_union_ofIsClosedOn {cfg : GameConfig}
    {A₁ A₂ : Atlas cfg} {S₁ S₂ : Finset GameState}
    (h₁ : A₁.IsClosedOn cfg S₁) (hinit₁ : GameState.init ∈ S₁)
    (h₂ : A₂.IsClosedOn cfg S₂) (hinit₂ : GameState.init ∈ S₂) :
    (M4Artifact.ofIsClosedOn h₁ hinit₁).union
      (M4Artifact.ofIsClosedOn h₂ hinit₂) =
      M4Artifact.ofIsClosedOn (Atlas.IsClosedOn.unionOn h₁ h₂)
        (Finset.mem_union_left _ hinit₁) := by
  rfl

/-- **M4 artifact certificate equivalence**: two artifacts are equivalent
if they have the same state set and their atlases agree on every active
query `(g, p)` with `g ∈ states, p ∈ g.bag`. Captures the precise notion
of "carrying the same certificate" — only the active queries matter for
M4 purposes. -/
def M4Artifact.Equiv {cfg : GameConfig} (M₁ M₂ : M4Artifact cfg) : Prop :=
  M₁.states = M₂.states ∧
  ∀ g ∈ M₁.states, ∀ p ∈ g.bag, M₁.atlas g p = M₂.atlas g p

/-- **`M4Artifact.Equiv` is reflexive**: every artifact is equivalent to
itself. -/
theorem M4Artifact.Equiv.refl {cfg : GameConfig} (M : M4Artifact cfg) :
    M.Equiv M :=
  ⟨rfl, fun _ _ _ _ => rfl⟩

/-- **Canon is certificate-equivalent to the original**: applying `canon`
to an M4 artifact preserves the certificate-equivalence class. Same
state set; atlases agree on active queries (by Tick 1615's
`canon_apply_of_active`). -/
theorem M4Artifact.canon_equiv {cfg : GameConfig} (M : M4Artifact cfg) :
    M.canon.Equiv M :=
  ⟨rfl, fun _ hg _ hp => Atlas.canon_apply_of_active M.atlas hg hp⟩

/-- **`M4Artifact.Equiv` is symmetric**: the equivalence relation is
preserved under reversal. -/
theorem M4Artifact.Equiv.symm {cfg : GameConfig} {M₁ M₂ : M4Artifact cfg}
    (h : M₁.Equiv M₂) : M₂.Equiv M₁ := by
  refine ⟨h.1.symm, fun g hg p hp => ?_⟩
  exact (h.2 g (h.1.symm ▸ hg) p hp).symm

/-- **`M4Artifact.Equiv` is transitive**: certificate-equivalence chains
through intermediate artifacts. Together with `refl` (Tick 1658) and
`symm`, this confirms `M4Artifact.Equiv` is an equivalence relation. -/
theorem M4Artifact.Equiv.trans {cfg : GameConfig}
    {M₁ M₂ M₃ : M4Artifact cfg} (h₁₂ : M₁.Equiv M₂) (h₂₃ : M₂.Equiv M₃) :
    M₁.Equiv M₃ := by
  refine ⟨h₁₂.1.trans h₂₃.1, fun g hg p hp => ?_⟩
  exact (h₁₂.2 g hg p hp).trans (h₂₃.2 g (h₁₂.1 ▸ hg) p hp)

/-- **`M4Artifact` Setoid instance**: registers `M4Artifact.Equiv` as the
canonical equivalence relation on M4 artifacts. Enables
`Quotient (M4Artifact.setoid cfg)` — the type of M4-equivalence classes. -/
instance M4Artifact.setoid (cfg : GameConfig) : Setoid (M4Artifact cfg) where
  r := M4Artifact.Equiv
  iseqv := ⟨M4Artifact.Equiv.refl,
            M4Artifact.Equiv.symm,
            M4Artifact.Equiv.trans⟩

/-- **Canon is the canonical representative (atlas)**: equivalent M4
artifacts have identical canon atlases. The substantive part of the
canonical-form principle — `canon` collapses each equivalence class to
a single representative atlas. -/
theorem M4Artifact.canon_atlas_eq_of_equiv {cfg : GameConfig}
    {M₁ M₂ : M4Artifact cfg} (h : M₁.Equiv M₂) :
    M₁.canon.atlas = M₂.canon.atlas := by
  change M₁.atlas.canon M₁.states = M₂.atlas.canon M₂.states
  funext g p
  by_cases hga : g ∈ M₁.states ∧ p ∈ g.bag
  · obtain ⟨hg, hp⟩ := hga
    have hg₂ : g ∈ M₂.states := h.1 ▸ hg
    rw [Atlas.canon_apply_of_active _ hg hp,
        Atlas.canon_apply_of_active _ hg₂ hp]
    exact h.2 g hg p hp
  · have hga₂ : ¬ (g ∈ M₂.states ∧ p ∈ g.bag) := fun ⟨hg₂, hp⟩ =>
      hga ⟨h.1.symm ▸ hg₂, hp⟩
    rw [Atlas.canon_apply_of_inactive _ hga,
        Atlas.canon_apply_of_inactive _ hga₂]

/-- **Canon is the canonical representative (states)**: equivalent M4
artifacts have identical canon state sets. Trivially from the Equiv's
state-set equality. -/
theorem M4Artifact.canon_states_eq_of_equiv {cfg : GameConfig}
    {M₁ M₂ : M4Artifact cfg} (h : M₁.Equiv M₂) :
    M₁.canon.states = M₂.canon.states := h.1

/-- **Equivalence from canon atlas + state set equality**: the reverse
direction of Tick 1661 — if two M4 artifacts have equal state sets and
equal canon atlases, then they are certificate-equivalent. Together with
Tick 1661, gives the full canon-data ↔ Equiv equivalence. -/
theorem M4Artifact.equiv_of_canon_atlas_eq {cfg : GameConfig}
    {M₁ M₂ : M4Artifact cfg}
    (hstates : M₁.states = M₂.states)
    (hatlas : M₁.canon.atlas = M₂.canon.atlas) :
    M₁.Equiv M₂ := by
  refine ⟨hstates, fun g hg p hp => ?_⟩
  have h1 : M₁.canon.atlas g p = M₁.atlas g p :=
    Atlas.canon_apply_of_active M₁.atlas hg hp
  have hg₂ : g ∈ M₂.states := hstates ▸ hg
  have h2 : M₂.canon.atlas g p = M₂.atlas g p :=
    Atlas.canon_apply_of_active M₂.atlas hg₂ hp
  rw [← h1, hatlas, h2]

/-- **Full Iff**: `M₁.Equiv M₂` iff their state sets are equal AND their
canon atlases coincide. The complete canonical-form characterisation of
M4 artifact equivalence. -/
theorem M4Artifact.equiv_iff_canon_atlas_eq {cfg : GameConfig}
    (M₁ M₂ : M4Artifact cfg) :
    M₁.Equiv M₂ ↔
      M₁.states = M₂.states ∧ M₁.canon.atlas = M₂.canon.atlas :=
  ⟨fun h => ⟨h.1, canon_atlas_eq_of_equiv h⟩,
   fun ⟨hs, ha⟩ => equiv_of_canon_atlas_eq hs ha⟩

/-- **`M4Artifact.union` is a congruence on Equiv**: if `M₁ ≈ M₁'` and
`M₂ ≈ M₂'`, then `M₁.union M₂ ≈ M₁'.union M₂'`. Shows `union` is
well-defined on M4 equivalence classes — enables `Quotient.lift`-style
construction of union on `Quotient (M4Artifact.setoid cfg)`. -/
theorem M4Artifact.union_equiv {cfg : GameConfig}
    {M₁ M₁' M₂ M₂' : M4Artifact cfg}
    (h₁ : M₁.Equiv M₁') (h₂ : M₂.Equiv M₂') :
    (M₁.union M₂).Equiv (M₁'.union M₂') := by
  refine ⟨?_, ?_⟩
  · change M₁.states ∪ M₂.states = M₁'.states ∪ M₂'.states
    rw [h₁.1, h₂.1]
  · intro g hg p hp
    change (M₁.atlas.unionOn M₂.atlas M₁.states M₂.states) g p =
           (M₁'.atlas.unionOn M₂'.atlas M₁'.states M₂'.states) g p
    by_cases hg1 : g ∈ M₁.states
    · have hg1' : g ∈ M₁'.states := h₁.1 ▸ hg1
      rw [Atlas.unionOn_apply_of_mem_left M₁.atlas M₂.atlas hg1 p,
          Atlas.unionOn_apply_of_mem_left M₁'.atlas M₂'.atlas hg1' p]
      exact h₁.2 g hg1 p hp
    · have hg_union : g ∈ M₁.states ∪ M₂.states := hg
      have hg2 : g ∈ M₂.states := by
        rw [Finset.mem_union] at hg_union
        exact hg_union.resolve_left hg1
      have hg1' : g ∉ M₁'.states := fun h => hg1 (h₁.1.symm ▸ h)
      have hg2' : g ∈ M₂'.states := h₂.1 ▸ hg2
      rw [Atlas.unionOn_apply_of_mem_right M₁.atlas M₂.atlas hg1 hg2 p,
          Atlas.unionOn_apply_of_mem_right M₁'.atlas M₂'.atlas hg1' hg2' p]
      exact h₂.2 g hg2 p hp

/-- **M4 union idempotence**: `(M.union M).Equiv M`. The first
algebraic identity for `M4Artifact.union`. Proof: state-set clause via
`Finset.union_self`; atlas clause via `Atlas.unionOn_self` reducing
the union-on-self to a `restrict (M.states ∪ M.states)`, which agrees
with `M.atlas` at any `g ∈ M.states ∪ M.states` (= `hg` directly). -/
theorem M4Artifact.union_self_equiv {cfg : GameConfig}
    (M : M4Artifact cfg) : (M.union M).Equiv M := by
  refine ⟨Finset.union_self _, fun g hg p _ => ?_⟩
  change (M.atlas.unionOn M.atlas M.states M.states) g p = M.atlas g p
  rw [Atlas.unionOn_self]
  exact Atlas.restrict_apply_of_mem _ hg p

/-- **M4 union associativity** (modulo Equiv):
`((M₁.union M₂).union M₃).Equiv (M₁.union (M₂.union M₃))`.

Combined with Tick 1686's idempotence and Tick 1657's congruence,
elevates `M4Artifact.union` to a **join-semilattice operation modulo
Equiv** (commutativity requires intersection-agreement and is a
separate result).

Proof: state-set clause via `Finset.union_assoc`. Atlas clause via
4-way case analysis on `g`'s membership in `M₁.states`, `M₂.states`,
`M₃.states`:
- `g ∈ M₁`: both sides reduce to `M₁.atlas g p` via leftmost
  unionOn left-rules.
- `g ∉ M₁, g ∈ M₂`: both reduce to `M₂.atlas g p` (LHS via inner
  left-then-outer left; RHS via outer right then inner left).
- `g ∉ M₁, g ∉ M₂, g ∈ M₃`: both reduce to `M₃.atlas g p` (LHS via
  outer right; RHS via outer right then inner right).
- All negative: contradiction with `hg : g ∈ ((M₁.union M₂).union
  M₃).states = (M₁.states ∪ M₂.states) ∪ M₃.states`. -/
theorem M4Artifact.union_assoc {cfg : GameConfig}
    (M₁ M₂ M₃ : M4Artifact cfg) :
    ((M₁.union M₂).union M₃).Equiv (M₁.union (M₂.union M₃)) := by
  refine ⟨?_, ?_⟩
  · change (M₁.states ∪ M₂.states) ∪ M₃.states =
           M₁.states ∪ (M₂.states ∪ M₃.states)
    exact Finset.union_assoc _ _ _
  · intro g hg p _
    change ((M₁.atlas.unionOn M₂.atlas M₁.states M₂.states).unionOn
              M₃.atlas (M₁.states ∪ M₂.states) M₃.states) g p =
           (M₁.atlas.unionOn
              (M₂.atlas.unionOn M₃.atlas M₂.states M₃.states)
              M₁.states (M₂.states ∪ M₃.states)) g p
    by_cases h1 : g ∈ M₁.states
    · have hg12 : g ∈ M₁.states ∪ M₂.states :=
        Finset.mem_union_left _ h1
      rw [Atlas.unionOn_apply_of_mem_left _ M₃.atlas hg12 p,
          Atlas.unionOn_apply_of_mem_left M₁.atlas M₂.atlas h1 p,
          Atlas.unionOn_apply_of_mem_left M₁.atlas _ h1 p]
    · by_cases h2 : g ∈ M₂.states
      · have hg12 : g ∈ M₁.states ∪ M₂.states :=
          Finset.mem_union_right _ h2
        have hg23 : g ∈ M₂.states ∪ M₃.states :=
          Finset.mem_union_left _ h2
        rw [Atlas.unionOn_apply_of_mem_left _ M₃.atlas hg12 p,
            Atlas.unionOn_apply_of_mem_right M₁.atlas M₂.atlas h1 h2 p,
            Atlas.unionOn_apply_of_mem_right M₁.atlas _ h1 hg23 p,
            Atlas.unionOn_apply_of_mem_left M₂.atlas M₃.atlas h2 p]
      · by_cases h3 : g ∈ M₃.states
        · have hg12 : g ∉ M₁.states ∪ M₂.states := by
            rw [Finset.mem_union]; tauto
          have hg23 : g ∈ M₂.states ∪ M₃.states :=
            Finset.mem_union_right _ h3
          rw [Atlas.unionOn_apply_of_mem_right _ M₃.atlas hg12 h3 p,
              Atlas.unionOn_apply_of_mem_right M₁.atlas _ h1 hg23 p,
              Atlas.unionOn_apply_of_mem_right M₂.atlas M₃.atlas h2 h3 p]
        · exfalso
          have hg' : g ∈ (M₁.states ∪ M₂.states) ∪ M₃.states := hg
          rw [Finset.mem_union, Finset.mem_union] at hg'
          tauto

/-- **M4 union commutativity (disjoint case)**:
`(M₁.union M₂).Equiv (M₂.union M₁)` when
`M₁.states ∩ M₂.states = ∅`. The simplest commutativity case — no
agreement on intersection needed because the intersection is empty.
Combined with Ticks 1686 (idempotence) and 1687 (associativity),
completes M4Artifact.union as a commutative join-semilattice operation
on **pairwise-disjoint** M4 artifacts. Proof: state-set via
`Finset.union_comm`; atlas clause via `Atlas.unionOn_comm_of_disjoint`
(line 841 in Adversarial.lean) reducing to strict atlas equality. -/
theorem M4Artifact.union_comm_of_disjoint {cfg : GameConfig}
    (M₁ M₂ : M4Artifact cfg)
    (hdisj : M₁.states ∩ M₂.states = ∅) :
    (M₁.union M₂).Equiv (M₂.union M₁) := by
  refine ⟨Finset.union_comm _ _, fun g _ p _ => ?_⟩
  change (M₁.atlas.unionOn M₂.atlas M₁.states M₂.states) g p =
         (M₂.atlas.unionOn M₁.atlas M₂.states M₁.states) g p
  rw [Atlas.unionOn_comm_of_disjoint M₁.atlas M₂.atlas hdisj]

/-- **M4 union commutativity (agreement case)**:
`(M₁.union M₂).Equiv (M₂.union M₁)` when the two atlases agree on
their state-set intersection (`∀ g ∈ M₁.states ∩ M₂.states, ∀ p,
M₁.atlas g p = M₂.atlas g p`). The most general commutativity result
for M4 union — subsumes Tick 1688's disjoint case (when intersection
is empty, the universal is vacuous).

Proof: state-set via `Finset.union_comm`; atlas clause via
`Atlas.unionOn_comm_of_agree` (line 825 in Adversarial.lean) reducing
to strict atlas equality. Lifts the atlas-level agreement
commutativity directly to M4 level. -/
theorem M4Artifact.union_comm_of_agree {cfg : GameConfig}
    (M₁ M₂ : M4Artifact cfg)
    (hagree : ∀ g ∈ M₁.states ∩ M₂.states, ∀ p,
              M₁.atlas g p = M₂.atlas g p) :
    (M₁.union M₂).Equiv (M₂.union M₁) := by
  refine ⟨Finset.union_comm _ _, fun g _ p _ => ?_⟩
  change (M₁.atlas.unionOn M₂.atlas M₁.states M₂.states) g p =
         (M₂.atlas.unionOn M₁.atlas M₂.states M₁.states) g p
  rw [Atlas.unionOn_comm_of_agree M₁.atlas M₂.atlas
        M₁.states M₂.states hagree]

/-- **Canon distributes over M4 union**: `(M.union M').canon.Equiv
(M.canon.union M'.canon)`. The canonicalisation of a union equals the
union of canonicalisations. Connects M4 canon algebra (Ticks 1659-
1670) with M4 union algebra (Ticks 1686-1689).

Proof: state-set clause is `rfl` (both = `M.states ∪ M'.states`,
since `M.canon.states = M.states`). Atlas clause: `change` exposes
the canonical/union structures, then `Atlas.canon_unionOn` (line 1181
in Adversarial.lean,
`(A.unionOn B S₁ S₂).canon (S₁ ∪ S₂) = (A.canon S₁).unionOn (B.canon
S₂) S₁ S₂`) gives strict atlas equality directly.

**Algebra-theoretic significance**: shows `canon` is a
**semilattice homomorphism** — distributes over the union operation.
Combined with `canon` being idempotent (Tick 1659), this confirms
`canon` is a quotient/projection of the M4 semilattice that respects
the algebraic structure. -/
theorem M4Artifact.union_canon_equiv {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    (M.union M').canon.Equiv (M.canon.union M'.canon) := by
  refine ⟨rfl, fun g _ p _ => ?_⟩
  change ((M.atlas.unionOn M'.atlas M.states M'.states).canon
            (M.states ∪ M'.states)) g p =
         ((M.atlas.canon M.states).unionOn
            (M'.atlas.canon M'.states) M.states M'.states) g p
  rw [Atlas.canon_unionOn]

/-- **Strict atlas equality version of Tick 1690**:
`(M.union M').canon.atlas = (M.canon.union M'.canon).atlas`.
Stronger than the Equiv version — the canonicalised union's atlas
equals the union of canonicalised atlases **as functions**, not just
modulo Equiv. Direct application of `Atlas.canon_unionOn` (line 1181
in Adversarial.lean) after a `change` to expose the structure.

**Algebraic significance**: confirms `canon` is a **strict atlas-level
homomorphism** over `unionOn`. The Equiv-level Tick 1690 already
established `canon` respects M4 union; this strengthens to atlas
identity. -/
theorem M4Artifact.union_canon_atlas_eq {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    (M.union M').canon.atlas = (M.canon.union M'.canon).atlas := by
  change (M.atlas.unionOn M'.atlas M.states M'.states).canon
           (M.states ∪ M'.states) =
         (M.atlas.canon M.states).unionOn
           (M'.atlas.canon M'.states) M.states M'.states
  exact Atlas.canon_unionOn _ _ _ _

/-- **Full strict equality strengthening of Tick 1690/1691**:
`(M.union M').canon = M.canon.union M'.canon` (M4 artifact equality,
not just Equiv or atlas-level). Strengthens the atlas-level
identity (Tick 1691) to the full structure — atlas equal by
`union_canon_atlas_eq`; states equal trivially (both `M.states ∪
M'.states`); the Prop fields (init_mem, closed) collapse via proof
irrelevance on Lean 4's `Prop`-typed structure components. -/
theorem M4Artifact.union_canon_eq {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    (M.union M').canon = M.canon.union M'.canon := by
  obtain ⟨A1, S1, i1, c1⟩ := M
  obtain ⟨A2, S2, i2, c2⟩ := M'
  unfold M4Artifact.canon M4Artifact.union
  congr 1
  exact Atlas.canon_unionOn _ _ _ _


/-- **Canon idempotence on canon-union (atlas level)**:
`(M.canon.union M'.canon).canon.atlas = (M.canon.union M'.canon).atlas`.
Confirms the canon-union form is a **fixed point** of further
canonicalisation at the atlas level — further `canon` is a no-op.

Proof: `change` exposes the structure. The LHS is the canon of a
unionOn of canonical atlases; by `Atlas.canon_unionOn` this becomes
unionOn of `canon ∘ canon` of each component, and two applications of
`Atlas.canon_canon` collapse those to single `canon`s — exactly the
RHS.

**Algebraic significance**: the "union of canons" form is **already
in canonical shape** — once you canonicalise the components and union
them, no further canonicalisation is needed. This makes
`canon-union` a closed operation on canonical-form atlases. -/
theorem M4Artifact.canon_union_canon_atlas_self {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    (M.canon.union M'.canon).canon.atlas =
      (M.canon.union M'.canon).atlas := by
  change ((M.atlas.canon M.states).unionOn
            (M'.atlas.canon M'.states) M.states M'.states).canon
          (M.states ∪ M'.states) =
         (M.atlas.canon M.states).unionOn
            (M'.atlas.canon M'.states) M.states M'.states
  rw [Atlas.canon_unionOn, Atlas.canon_canon, Atlas.canon_canon]

/-- **M4 left atlas-projection under canon**: canonicalising the
union's atlas at the left operand's state set collapses to
`M₁.canon.atlas`. Lifts Tick 1788's `Atlas.canon_unionOn_left` to
the M4 artifact level. No disjointness hypothesis required — the
left operand is always projection-clean under priority dispatch. -/
theorem M4Artifact.union_atlas_canon_left {cfg : GameConfig}
    (M₁ M₂ : M4Artifact cfg) :
    (M₁.union M₂).atlas.canon M₁.states = M₁.canon.atlas := by
  change (M₁.atlas.unionOn M₂.atlas M₁.states M₂.states).canon M₁.states =
         M₁.atlas.canon M₁.states
  exact Atlas.canon_unionOn_left M₁.atlas M₂.atlas M₁.states M₂.states

/-- **M4 right atlas-projection under canon, under disjointness**:
when the operand state-sets are disjoint, canonicalising the
union's atlas at the right operand's state set collapses to
`M₂.canon.atlas`. Lifts Tick 1789's
`Atlas.canon_unionOn_right_of_disjoint` to the M4 artifact level.
Disjointness is essential — see Tick 1789 for the geometric
explanation. -/
theorem M4Artifact.union_atlas_canon_right_of_disjoint {cfg : GameConfig}
    (M₁ M₂ : M4Artifact cfg) (hdisj : Disjoint M₁.states M₂.states) :
    (M₁.union M₂).atlas.canon M₂.states = M₂.canon.atlas := by
  change (M₁.atlas.unionOn M₂.atlas M₁.states M₂.states).canon M₂.states =
         M₂.atlas.canon M₂.states
  exact Atlas.canon_unionOn_right_of_disjoint
    M₁.atlas M₂.atlas M₁.states M₂.states hdisj

/-- **Canonical-form M4 artifact predicate**: an M4 artifact `M` is
*canonical* if its atlas equals its own canonicalisation at its state
set. Captures the minimal-storage representative property — the atlas
returns `none` at all inactive queries (`g ∉ states ∨ p ∉ g.bag`). -/
def M4Artifact.IsCanonical {cfg : GameConfig} (M : M4Artifact cfg) : Prop :=
  M.atlas = M.atlas.canon M.states

/-- **`M.canon` is always canonical**: applying `canon` to any M4
artifact produces a canonical-form artifact. Direct from
`Atlas.canon_canon` (idempotence of `canon` at the atlas level). -/
theorem M4Artifact.canon_isCanonical {cfg : GameConfig}
    (M : M4Artifact cfg) : M.canon.IsCanonical := by
  change M.canon.atlas = M.canon.atlas.canon M.canon.states
  exact (Atlas.canon_canon M.atlas M.states).symm

/-- **`ofIsClosedOn` canonicality biconditional**: an `ofIsClosedOn`-
constructed artifact is canonical iff its input atlas is already
in canonical form. Direct via `Iff.rfl` — `IsCanonical` definition
unfolds to the atlas-equality form. -/
theorem M4Artifact.ofIsClosedOn_isCanonical_iff {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    (M4Artifact.ofIsClosedOn h hinit).IsCanonical ↔ A = A.canon S :=
  Iff.rfl

/-- **`ofIsClosedOn` Equiv biconditional**: two `ofIsClosedOn`-
constructed artifacts are Equiv iff their state sets agree AND
their atlases agree on every active query (`g ∈ S₁, p ∈ g.bag`).
Direct via `Iff.rfl` — `M4Artifact.Equiv` definition unfolds to
exactly this form when both M4s come from `ofIsClosedOn`. -/
theorem M4Artifact.ofIsClosedOn_equiv_iff {cfg : GameConfig}
    {A₁ A₂ : Atlas cfg} {S₁ S₂ : Finset GameState}
    (h₁ : A₁.IsClosedOn cfg S₁) (hinit₁ : GameState.init ∈ S₁)
    (h₂ : A₂.IsClosedOn cfg S₂) (hinit₂ : GameState.init ∈ S₂) :
    (M4Artifact.ofIsClosedOn h₁ hinit₁).Equiv
        (M4Artifact.ofIsClosedOn h₂ hinit₂) ↔
      S₁ = S₂ ∧ ∀ g ∈ S₁, ∀ p ∈ g.bag, A₁ g p = A₂ g p :=
  Iff.rfl


/-- **Canonical form is closed under M4 union**: if both `M` and `M'`
are canonical, then `M.union M'` is canonical. Operationally critical
— this means **canonical-form storage is preserved by union** without
needing a re-canonicalisation pass.

Proof: `change` exposes the union's atlas. The goal becomes
`M.atlas.unionOn M'.atlas M.states M'.states =
 (M.atlas.unionOn M'.atlas M.states M'.states).canon (M.states ∪ M'.states)`.
`Atlas.canon_unionOn` rewrites the RHS to
`(M.atlas.canon M.states).unionOn (M'.atlas.canon M'.states) M.states M'.states`.
Then `← hM` and `← hM'` substitute back to `M.atlas` and `M'.atlas`,
giving syntactic equality with the LHS. -/
theorem M4Artifact.isCanonical_union_of_isCanonical {cfg : GameConfig}
    {M M' : M4Artifact cfg} (hM : M.IsCanonical) (hM' : M'.IsCanonical) :
    (M.union M').IsCanonical := by
  change M.atlas.unionOn M'.atlas M.states M'.states =
         (M.atlas.unionOn M'.atlas M.states M'.states).canon
           (M.states ∪ M'.states)
  rw [Atlas.canon_unionOn, ← hM, ← hM']

/-- **`canon` is the identity on canonical artifacts (atlas level)**:
if `M.IsCanonical`, then `M.canon.atlas = M.atlas`. Direct corollary
of the `IsCanonical` definition's symmetric form (since
`M.canon.atlas := M.atlas.canon M.states` by definition). -/
theorem M4Artifact.canon_atlas_eq_of_isCanonical {cfg : GameConfig}
    {M : M4Artifact cfg} (hM : M.IsCanonical) :
    M.canon.atlas = M.atlas := hM.symm

/-- **Canonical uniqueness (atlas level)**: if two M4 artifacts are
Equiv and BOTH canonical, their atlases are STRICTLY EQUAL as
functions. The canonical form is a **categorical normal form** —
every Equiv class has AT MOST ONE canonical representative atlas.
Combined with Tick 1693's `canon_isCanonical` (every M has at least
one canonical representative: `M.canon`), this establishes
**canonical existence + uniqueness** — the canonical form is the
canonical representative.

Proof: rewrite both sides to their canon forms via
`canon_atlas_eq_of_isCanonical`, then apply Tick 1661's
`canon_atlas_eq_of_equiv` to collapse them. -/
theorem M4Artifact.canon_atlas_unique_of_equiv {cfg : GameConfig}
    {M M' : M4Artifact cfg} (h : M.Equiv M')
    (hM : M.IsCanonical) (hM' : M'.IsCanonical) :
    M.atlas = M'.atlas := by
  rw [← canon_atlas_eq_of_isCanonical hM,
      ← canon_atlas_eq_of_isCanonical hM',
      canon_atlas_eq_of_equiv h]

/-- **The canonical N-cycle build produces a canonical M4 artifact**:
for any list of cycles with an init-containing witness,
`(buildFromCyclesCanon_M4Artifact cycles h).IsCanonical`. Direct
corollary of `buildFromCyclesCanon_empty_canon_eq_self` — the
existing atlas-level theorem stating `canon (build) = build` reverses
to exactly the `IsCanonical` predicate when applied to the M4
artifact wrapping. The canonical builder formally produces canonical
artifacts. -/
theorem M4Artifact.buildFromCyclesCanon_M4Artifact_isCanonical
    {cfg : GameConfig}
    (cycles : List (AdversarialClosedCycle cfg))
    (h : ∃ C ∈ cycles, GameState.init ∈ C.states) :
    (buildFromCyclesCanon_M4Artifact cycles h).IsCanonical :=
  (buildFromCyclesCanon_empty_canon_eq_self cycles).symm

/-- **`fromCycleCanon` produces canonical M4 artifacts**: for any
init-containing cycle `C`, `(fromCycleCanon C h).IsCanonical`. The
single-cycle canonical constructor is also a canonical-form
producer. Proof: `(fromCycleCanon C h).atlas =
C.solver.toAtlas.canon C.states` by definition; canonicality
unfolds to `canon C.states = canon C.states .canon C.states`, which
is `Atlas.canon_canon.symm`. -/
theorem M4Artifact.fromCycleCanon_isCanonical {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    (M4Artifact.fromCycleCanon C h).IsCanonical :=
  (Atlas.canon_canon _ _).symm

/-- **Canonical Equiv-equivalents are STRUCTURALLY EQUAL**: if two M4
artifacts are M4-Equiv and both canonical, they are EQUAL as M4
artifacts (not just Equiv). Strictly stronger than Tick 1694's
atlas-only uniqueness — also handles the state set field and the
Prop fields (init_mem, closed) via Lean 4's definitional proof
irrelevance.

This establishes **canonical M4 artifacts modulo Equiv form a
discrete category** — each Equiv class has EXACTLY ONE canonical
element. The canonical representative is unique not just up to atlas
content, but as a full M4 artifact.

Proof: extract atlas and states equalities via Tick 1694 + `h.1`;
destructure both artifacts via `obtain`; `subst` collapses the data
fields; `rfl` closes the goal (Prop fields equal by definitional
proof irrelevance). -/
theorem M4Artifact.eq_of_equiv_of_isCanonical_both {cfg : GameConfig}
    {M M' : M4Artifact cfg} (h : M.Equiv M')
    (hM : M.IsCanonical) (hM' : M'.IsCanonical) :
    M = M' := by
  have hatlas : M.atlas = M'.atlas :=
    canon_atlas_unique_of_equiv h hM hM'
  have hstates : M.states = M'.states := h.1
  obtain ⟨a, s, i, c⟩ := M
  obtain ⟨a', s', i', c'⟩ := M'
  subst hatlas
  subst hstates
  rfl

/-- **`M.canon` is the UNIQUE canonical representative**: for any M
and any M' that is Equiv to M AND canonical, `M' = M.canon` (full
structural equality). Combines Tick 1702's discrete-category uniqueness
with Tick 1657's `canon_equiv` (M.canon ≈ M) and Tick 1693's
`canon_isCanonical` (M.canon is canonical).

This characterises `M.canon` as the **unique projection** from the
M4 Equiv class of M to the canonical sub-category. -/
theorem M4Artifact.canon_unique {cfg : GameConfig}
    {M M' : M4Artifact cfg} (heq : M.Equiv M')
    (hM' : M'.IsCanonical) : M' = M.canon :=
  (eq_of_equiv_of_isCanonical_both
    (M.canon_equiv.trans heq) M.canon_isCanonical hM').symm

/-- **Full structural canon idempotence**: `M.canon.canon = M.canon`
as M4 artifacts (not just at the atlas/states levels). Strictly
stronger than Tick 1659's `canon_canon_atlas` (which gives only
atlas equality). Direct corollary of Tick 1703's `canon_unique` —
since `M.canon.canon` is canonical (Tick 1693) and Equiv to M (via
two applications of `canon_equiv`), it must equal `M.canon`. -/
theorem M4Artifact.canon_canon_eq {cfg : GameConfig}
    (M : M4Artifact cfg) : M.canon.canon = M.canon := by
  have heq : M.Equiv M.canon.canon :=
    M.canon_equiv.symm.trans M.canon.canon_equiv.symm
  exact canon_unique heq (canon_isCanonical M.canon)

/-- **`canon` is constant on Equiv classes (full structural)**: if
`M ≈ M'`, then `M.canon = M'.canon` as M4 artifacts (full struct
equality, not just atlas equality from Tick 1661). The `canon`
projection is a **well-defined function on Equiv classes**. Direct
corollary of Tick 1702: both `M.canon` and `M'.canon` are canonical
and (via transitivity) Equiv to each other, so they must be equal.

This is the **strict version** of `canon` respecting Equiv:
Equiv-related inputs produce identical canonical outputs. -/
theorem M4Artifact.canon_eq_of_equiv {cfg : GameConfig}
    {M M' : M4Artifact cfg} (h : M.Equiv M') :
    M.canon = M'.canon := by
  have heq : M.canon.Equiv M'.canon :=
    M.canon_equiv.trans (h.trans M'.canon_equiv.symm)
  exact eq_of_equiv_of_isCanonical_both heq
    (canon_isCanonical M) (canon_isCanonical M')

/-- **Canonical artifacts are equal to their canon**: if
`M.IsCanonical`, then `M = M.canon` as M4 artifacts. Direct
consequence of Tick 1703's universal property of canon with
`M' := M` and reflexivity. -/
theorem M4Artifact.canon_eq_self_of_isCanonical {cfg : GameConfig}
    {M : M4Artifact cfg} (hM : M.IsCanonical) :
    M = M.canon :=
  canon_unique (M4Artifact.Equiv.refl M) hM

/-- **`M.canon = M ↔ M.IsCanonical`** (biconditional). The strictest
characterization of canonicality at the M4 artifact level: M is
canonical iff applying `canon` is the identity. Forward: `M = M.canon`
+ Tick 1693's `canon_isCanonical` gives `M.IsCanonical`. Reverse:
Tick 1706's `canon_eq_self_of_isCanonical` (symm) gives `M.canon = M`. -/
theorem M4Artifact.canon_eq_iff_isCanonical {cfg : GameConfig}
    {M : M4Artifact cfg} :
    M.canon = M ↔ M.IsCanonical := by
  refine ⟨fun h => ?_, fun hM => (canon_eq_self_of_isCanonical hM).symm⟩
  rw [← h]
  exact canon_isCanonical M

/-- **`M' = M.canon ↔ M.Equiv M' ∧ M'.IsCanonical`** (biconditional).
Characterizes when M' IS the canonical form of M: precisely when M'
is canonical AND Equiv to M. Forward: `M' = M.canon` gives both
M.Equiv M.canon (via Tick 1657 `canon_equiv.symm`) and
M.canon.IsCanonical (Tick 1693). Reverse: Tick 1703's `canon_unique`
directly gives `M' = M.canon` from the two hypotheses. -/
theorem M4Artifact.eq_canon_iff_equiv_and_isCanonical {cfg : GameConfig}
    {M M' : M4Artifact cfg} :
    M' = M.canon ↔ M.Equiv M' ∧ M'.IsCanonical := by
  refine ⟨fun h => ?_, fun ⟨heq, hM'⟩ => canon_unique heq hM'⟩
  subst h
  exact ⟨M.canon_equiv.symm, canon_isCanonical M⟩

/-- **Unique canonical representative**: `∃! M', M.Equiv M' ∧
M'.IsCanonical`. Existence + uniqueness combined: each Equiv class
has EXACTLY ONE canonical M4 artifact, namely `M.canon`. Direct from
Tick 1657 (canon_equiv), Tick 1693 (canon_isCanonical), and Tick
1703 (canon_unique). -/
theorem M4Artifact.exists_unique_canonical_equiv {cfg : GameConfig}
    (M : M4Artifact cfg) :
    ∃! M' : M4Artifact cfg, M.Equiv M' ∧ M'.IsCanonical :=
  ⟨M.canon, ⟨M.canon_equiv.symm, canon_isCanonical M⟩,
   fun _ ⟨heq, hM'⟩ => canon_unique heq hM'⟩

/-- **Canon equality implies struct equality for canonical pair**:
if both M and M' are canonical AND their canons are equal, then
M = M'. Direct chain via Tick 1706's `canon_eq_self_of_isCanonical`
applied to both. -/
theorem M4Artifact.eq_of_canon_eq_of_isCanonical_both {cfg : GameConfig}
    {M M' : M4Artifact cfg} (hM : M.IsCanonical) (hM' : M'.IsCanonical)
    (h : M.canon = M'.canon) : M = M' :=
  (canon_eq_self_of_isCanonical hM).trans
    (h.trans (canon_eq_self_of_isCanonical hM').symm)

/-- **`M.canon.atlas g p = none ↔ inactive`**: the canonical atlas
returns `none` exactly at inactive queries (g ∉ M.states OR p ∉ g.bag).
At active queries, the closure totality guarantees `some _`. -/
theorem M4Artifact.canon_atlas_eq_none_iff_inactive {cfg : GameConfig}
    {M : M4Artifact cfg} {g : GameState} {p : Piece} :
    M.canon.atlas g p = none ↔ ¬(g ∈ M.states ∧ p ∈ g.bag) := by
  refine ⟨fun h hac => ?_, fun hi => ?_⟩
  · obtain ⟨hg, hp⟩ := hac
    change M.atlas.canon M.states g p = none at h
    rw [Atlas.canon_apply_of_active _ hg hp] at h
    have h_total := M.closed.total g hg p hp
    rw [Option.isSome_iff_ne_none] at h_total
    exact h_total h
  · change M.atlas.canon M.states g p = none
    exact Atlas.canon_apply_of_inactive _ hi

/-- **`M.canon.atlas g p` is `some _` iff active**: contrapositive
of Tick 1761. The canon atlas returns `some _` exactly at active
queries. Derived via `Option.isSome_iff_ne_none` + iff negation. -/
theorem M4Artifact.canon_atlas_isSome_iff_active {cfg : GameConfig}
    {M : M4Artifact cfg} {g : GameState} {p : Piece} :
    (M.canon.atlas g p).isSome ↔ g ∈ M.states ∧ p ∈ g.bag := by
  rw [Option.isSome_iff_ne_none, ne_eq,
      canon_atlas_eq_none_iff_inactive, not_not]

/-- **Canonical M4 artifact: `M.atlas g p = none ↔ inactive`**:
specialization of `canon_atlas_eq_none_iff_inactive` to canonical
M4 artifacts (where `M.atlas = M.atlas.canon M.states`). Direct
via the `IsCanonical` definition + the canon-form theorem. -/
theorem M4Artifact.atlas_eq_none_iff_inactive_of_isCanonical
    {cfg : GameConfig} {M : M4Artifact cfg} (hM : M.IsCanonical)
    {g : GameState} {p : Piece} :
    M.atlas g p = none ↔ ¬(g ∈ M.states ∧ p ∈ g.bag) := by
  have h1 : M.atlas g p = M.canon.atlas g p := by
    change M.atlas g p = (M.atlas.canon M.states) g p
    rw [← hM]
  rw [h1]
  exact M.canon_atlas_eq_none_iff_inactive

/-- **Canonical M4 artifact: `M.atlas g p` is `some _` iff active**:
contrapositive form of Tick 1956. The canonical M4 atlas returns
`some _` exactly at active queries. -/
theorem M4Artifact.atlas_isSome_iff_active_of_isCanonical
    {cfg : GameConfig} {M : M4Artifact cfg} (hM : M.IsCanonical)
    {g : GameState} {p : Piece} :
    (M.atlas g p).isSome ↔ g ∈ M.states ∧ p ∈ g.bag := by
  rw [Option.isSome_iff_ne_none, ne_eq,
      atlas_eq_none_iff_inactive_of_isCanonical hM, not_not]


/-- **`M.canon.atlas g p = M.atlas g p` at active queries**: useful
named M4 lemma directly applying `Atlas.canon_apply_of_active`. -/
theorem M4Artifact.canon_atlas_eq_at_active {cfg : GameConfig}
    {M : M4Artifact cfg} {g : GameState} (hg : g ∈ M.states)
    {p : Piece} (hp : p ∈ g.bag) :
    M.canon.atlas g p = M.atlas g p :=
  Atlas.canon_apply_of_active _ hg hp

/-- **`M.canon.atlas.toSolver g p = M.atlas.toSolver g p` at active
queries**: the materialised solver agrees at active queries between
the original M4 atlas and its canonicalised form. Direct application
of Tick 1716's `Atlas.canon_toSolver_apply_of_active`. -/
theorem M4Artifact.canon_atlas_toSolver_eq_at_active {cfg : GameConfig}
    {M : M4Artifact cfg} {g : GameState} (hg : g ∈ M.states)
    {p : Piece} (hp : p ∈ g.bag) :
    M.canon.atlas.toSolver g p = M.atlas.toSolver g p :=
  Atlas.canon_toSolver_apply_of_active hg hp


/-- **`IsCanonical` is characterised by being in canon's image**:
`M.IsCanonical ↔ ∃ M', M = M'.canon`. The canonical artifacts are
**exactly** the image of the `canon` projection.

Forward: take `M' := M`; canonicality gives `M = M.canon` via Tick
1706's `canon_eq_self_of_isCanonical`. Reverse: substitute `M :=
M'.canon` and apply Tick 1693's `canon_isCanonical`. -/
theorem M4Artifact.isCanonical_iff_in_canon_image {cfg : GameConfig}
    {M : M4Artifact cfg} :
    M.IsCanonical ↔ ∃ M' : M4Artifact cfg, M = M'.canon := by
  refine ⟨fun hM => ⟨M, canon_eq_self_of_isCanonical hM⟩, ?_⟩
  rintro ⟨M', rfl⟩
  exact canon_isCanonical M'

/-- **For canonical artifacts, Equiv = structural equality**: if both
`M` and `M'` are canonical, then `M.Equiv M' ↔ M = M'`. Strict
biconditional showing Equiv and `=` coincide on the canonical
subset. Forward: Tick 1702's discrete-category theorem. Reverse:
trivial (equal artifacts are reflexively Equiv). -/
theorem M4Artifact.equiv_iff_eq_of_isCanonical_both {cfg : GameConfig}
    {M M' : M4Artifact cfg} (hM : M.IsCanonical) (hM' : M'.IsCanonical) :
    M.Equiv M' ↔ M = M' := by
  refine ⟨fun h => eq_of_equiv_of_isCanonical_both h hM hM', ?_⟩
  rintro rfl
  exact M4Artifact.Equiv.refl M

/-- **M4 Equiv ↔ canon equality**: `M.Equiv M' ↔ M.canon = M'.canon`.
Reduces the Equiv relation to strict equality of canonical forms.
The **operationally most useful** characterisation of M4 Equiv —
decidable by comparing canon atlases for equality (when atlas
equality is decidable on the relevant Finset).

Forward: Tick 1705's `canon_eq_of_equiv`. Reverse: chain
`M ≈ M.canon` (via `canon_equiv.symm`), then `M.canon.Equiv M'` (from
hypothesis after rewriting M.canon to M'.canon and using
`M'.canon_equiv`). -/
theorem M4Artifact.equiv_iff_canon_eq {cfg : GameConfig}
    {M M' : M4Artifact cfg} :
    M.Equiv M' ↔ M.canon = M'.canon := by
  refine ⟨canon_eq_of_equiv, fun h => ?_⟩
  have h₁ : M.canon.Equiv M' := by rw [h]; exact M'.canon_equiv
  exact M.canon_equiv.symm.trans h₁

/-- **`IsCanonical` characterised pointwise**: `M.IsCanonical ↔
∀ g p, ¬(g ∈ M.states ∧ p ∈ g.bag) → M.atlas g p = none`. The
practical/computational characterisation — canonical means the
atlas returns `none` at every inactive query (out of states OR not in
bag). Strictly equivalent to the definitional form `M.atlas =
M.atlas.canon M.states` but operationally more direct.

Forward: substitute `M.atlas` via the canonicality hypothesis, apply
`Atlas.canon_apply_of_inactive`. Reverse: pointwise equality via
`funext`, case-split on active/inactive — active uses
`canon_apply_of_active`, inactive uses `canon_apply_of_inactive`
combined with the hypothesis. -/
theorem M4Artifact.isCanonical_iff_none_at_inactive {cfg : GameConfig}
    {M : M4Artifact cfg} :
    M.IsCanonical ↔
      ∀ g p, ¬(g ∈ M.states ∧ p ∈ g.bag) → M.atlas g p = none := by
  refine ⟨fun hM g p hi => ?_, fun h => ?_⟩
  · have : M.atlas g p = (M.atlas.canon M.states) g p := by rw [← hM]
    rw [this, Atlas.canon_apply_of_inactive _ hi]
  · funext g p
    by_cases hac : g ∈ M.states ∧ p ∈ g.bag
    · rw [Atlas.canon_apply_of_active _ hac.1 hac.2]
    · rw [Atlas.canon_apply_of_inactive _ hac, h g p hac]

/-- **Canon-of-union equals union-of-canons** as full M4 artifacts:
`(M.union M').canon = M.canon.union M'.canon`. Strictly stronger
than Tick 1690's Equiv-level identity and Tick 1691's atlas-level
identity. The full struct equality follows from Tick 1702's
discrete-category property: both sides are canonical (Tick 1693)
and Equiv (Tick 1690). -/
theorem M4Artifact.canon_union_eq_canon_union {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    (M.union M').canon = M.canon.union M'.canon :=
  eq_of_equiv_of_isCanonical_both
    (union_canon_equiv M M')
    (canon_isCanonical _)
    (isCanonical_union_of_isCanonical
      (canon_isCanonical M) (canon_isCanonical M'))

/-- **Canon-canon of union equals canon-union**:
`(M.union M').canon.canon = M.canon.union M'.canon` as full M4
artifacts. Derived chain identity: apply Tick 1704's
`canon_canon_eq` then Tick 1742's `canon_union_eq_canon_union`. -/
theorem M4Artifact.canon_canon_union_eq_canon_union {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    (M.union M').canon.canon = M.canon.union M'.canon :=
  (canon_canon_eq (M.union M')).trans (canon_union_eq_canon_union M M')

/-- **M4 union is strictly associative on canonical artifacts**:
when M₁, M₂, M₃ are all canonical, `(M₁.union M₂).union M₃ = M₁.union
(M₂.union M₃)` as full M4 artifacts. Lifts Tick 1687's Equiv-level
associativity to strict structural equality using Tick 1702's
discrete-category property. -/
theorem M4Artifact.union_assoc_eq_of_all_canonical {cfg : GameConfig}
    {M₁ M₂ M₃ : M4Artifact cfg}
    (h₁ : M₁.IsCanonical) (h₂ : M₂.IsCanonical) (h₃ : M₃.IsCanonical) :
    (M₁.union M₂).union M₃ = M₁.union (M₂.union M₃) :=
  eq_of_equiv_of_isCanonical_both
    (union_assoc M₁ M₂ M₃)
    (isCanonical_union_of_isCanonical
      (isCanonical_union_of_isCanonical h₁ h₂) h₃)
    (isCanonical_union_of_isCanonical h₁
      (isCanonical_union_of_isCanonical h₂ h₃))

/-- **M4 union is strictly commutative on canonical artifacts with
atlas agreement**: when M₁, M₂ are both canonical AND their atlases
agree on the intersection, `M₁.union M₂ = M₂.union M₁` as full M4
artifacts. Lifts Tick 1689's Equiv-level commutativity (with
agreement) to strict structural equality using Tick 1702's
discrete-category property. -/
theorem M4Artifact.union_comm_eq_of_canonical_and_agree {cfg : GameConfig}
    {M₁ M₂ : M4Artifact cfg} (h₁ : M₁.IsCanonical) (h₂ : M₂.IsCanonical)
    (hagree : ∀ g ∈ M₁.states ∩ M₂.states, ∀ p,
              M₁.atlas g p = M₂.atlas g p) :
    M₁.union M₂ = M₂.union M₁ :=
  eq_of_equiv_of_isCanonical_both
    (union_comm_of_agree M₁ M₂ hagree)
    (isCanonical_union_of_isCanonical h₁ h₂)
    (isCanonical_union_of_isCanonical h₂ h₁)

/-- **M4 union is strictly commutative on canonical artifacts with
disjoint states**: when M₁, M₂ are both canonical AND
`M₁.states ∩ M₂.states = ∅`, `M₁.union M₂ = M₂.union M₁`.
Disjoint case (no agreement needed). Parallel to Tick 1744 but with
disjointness condition. -/
theorem M4Artifact.union_comm_eq_of_canonical_and_disjoint
    {cfg : GameConfig} {M₁ M₂ : M4Artifact cfg}
    (h₁ : M₁.IsCanonical) (h₂ : M₂.IsCanonical)
    (hdisj : M₁.states ∩ M₂.states = ∅) :
    M₁.union M₂ = M₂.union M₁ :=
  eq_of_equiv_of_isCanonical_both
    (union_comm_of_disjoint M₁ M₂ hdisj)
    (isCanonical_union_of_isCanonical h₁ h₂)
    (isCanonical_union_of_isCanonical h₂ h₁)

/-- **Canon of canon-union is the canon-union itself** (strict M4
struct equality): `(M.canon.union M'.canon).canon = M.canon.union
M'.canon`. The canon-union form is a fixed point of `canon` at the
full M4 artifact level. Direct from Tick 1706's `canon_eq_self_of_isCanonical`
applied with Tick 1693's `isCanonical_union_of_isCanonical` of two
`canon_isCanonical` witnesses. -/
theorem M4Artifact.canon_union_canon_canon_eq {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    (M.canon.union M'.canon).canon = M.canon.union M'.canon :=
  (canon_eq_self_of_isCanonical
    (isCanonical_union_of_isCanonical
      (canon_isCanonical M) (canon_isCanonical M'))).symm

/-- **Strict equality for canonical union-self**: `(M.union M).canon
= M.canon`. Lifts Tick 1686's Equiv-level `union_self_equiv` to
strict M4-artifact equality at the canonical projection.
Composes `(M.union M).canon ≈ M.union M ≈ M ≈ M.canon` (via
`canon_equiv`, `union_self_equiv`, `canon_equiv.symm`), then uses
`eq_of_equiv_of_isCanonical_both` (Tick 1702) — both sides are
canonical via `canon_isCanonical`. -/
theorem M4Artifact.union_self_canon_eq {cfg : GameConfig}
    (M : M4Artifact cfg) :
    (M.union M).canon = M.canon :=
  eq_of_equiv_of_isCanonical_both
    (((M.union M).canon_equiv.trans M.union_self_equiv).trans M.canon_equiv.symm)
    (M.union M).canon_isCanonical
    M.canon_isCanonical

/-- **Strict equality for canonical union associativity**:
`((M₁.union M₂).union M₃).canon = (M₁.union (M₂.union M₃)).canon`.
Lifts Tick 1687's Equiv-level `union_assoc` to strict M4-artifact
equality at the canonical projection. **No hypothesis on
canonicality of M₁, M₂, M₃** — unlike Tick 1727's
`union_assoc_eq_of_all_canonical` which requires all three
canonical. Apply canon to both sides first; uses
`eq_of_equiv_of_isCanonical_both` (Tick 1702) with the chain
`((M₁.union M₂).union M₃).canon ≈ (M₁.union M₂).union M₃ ≈
M₁.union (M₂.union M₃) ≈ (M₁.union (M₂.union M₃)).canon`. -/
theorem M4Artifact.union_assoc_canon_eq {cfg : GameConfig}
    (M₁ M₂ M₃ : M4Artifact cfg) :
    ((M₁.union M₂).union M₃).canon = (M₁.union (M₂.union M₃)).canon :=
  eq_of_equiv_of_isCanonical_both
    ((((M₁.union M₂).union M₃).canon_equiv.trans
        (union_assoc M₁ M₂ M₃)).trans
      (M₁.union (M₂.union M₃)).canon_equiv.symm)
    ((M₁.union M₂).union M₃).canon_isCanonical
    (M₁.union (M₂.union M₃)).canon_isCanonical

/-- **Strict equality for canonical union commutativity under
disjointness**: `(M₁.union M₂).canon = (M₂.union M₁).canon` when
`M₁.states ∩ M₂.states = ∅`. Lifts Tick 1688's Equiv-level
`union_comm_of_disjoint` to strict M4-artifact equality at the
canonical projection. **No canonicality hypothesis on M₁, M₂** —
parallel pattern to Tick 1795. Apply canon to both sides first;
uses `eq_of_equiv_of_isCanonical_both`. -/
theorem M4Artifact.union_comm_canon_eq_of_disjoint {cfg : GameConfig}
    (M₁ M₂ : M4Artifact cfg)
    (hdisj : M₁.states ∩ M₂.states = ∅) :
    (M₁.union M₂).canon = (M₂.union M₁).canon :=
  eq_of_equiv_of_isCanonical_both
    (((M₁.union M₂).canon_equiv.trans
        (union_comm_of_disjoint M₁ M₂ hdisj)).trans
      (M₂.union M₁).canon_equiv.symm)
    (M₁.union M₂).canon_isCanonical
    (M₂.union M₁).canon_isCanonical

/-- **Strict equality for canonical union commutativity under atlas
agreement**: `(M₁.union M₂).canon = (M₂.union M₁).canon` when
M₁ and M₂'s atlases agree on the intersection of state sets.
Lifts Tick 1689's Equiv-level `union_comm_of_agree` to strict
M4-artifact equality at the canonical projection. **No canonicality
hypothesis on M₁, M₂** — parallel pattern to Ticks 1795-1796.
Companion to Tick 1796 for the overlap-with-agreement case. -/
theorem M4Artifact.union_comm_canon_eq_of_agree {cfg : GameConfig}
    (M₁ M₂ : M4Artifact cfg)
    (hagree : ∀ g ∈ M₁.states ∩ M₂.states, ∀ p,
              M₁.atlas g p = M₂.atlas g p) :
    (M₁.union M₂).canon = (M₂.union M₁).canon :=
  eq_of_equiv_of_isCanonical_both
    (((M₁.union M₂).canon_equiv.trans
        (union_comm_of_agree M₁ M₂ hagree)).trans
      (M₂.union M₁).canon_equiv.symm)
    (M₁.union M₂).canon_isCanonical
    (M₂.union M₁).canon_isCanonical

/-- **Canonicalising the left operand doesn't change union under
canon**: `(M.canon.union M').canon = (M.union M').canon`. The
canon-projected union is invariant under canonicalising operands
before unioning. Direct via Tick 1705's `canon_eq_of_equiv`:
`M.canon ≈ M` (canon_equiv), so `M.canon.union M' ≈ M.union M'`
(union_equiv with refl on M'); applying canon to both sides
preserves the Equiv-equality at strict M4-equality. -/
theorem M4Artifact.canon_union_canon_eq {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    (M.canon.union M').canon = (M.union M').canon :=
  canon_eq_of_equiv (union_equiv M.canon_equiv (M4Artifact.Equiv.refl M'))

/-- **Canonicalising the right operand doesn't change union under
canon**: `(M.union M'.canon).canon = (M.union M').canon`. Symmetric
companion to Tick 1798. Direct via `canon_eq_of_equiv` applied to
`union_equiv (refl M) M'.canon_equiv`. -/
theorem M4Artifact.union_canon_canon_eq {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    (M.union M'.canon).canon = (M.union M').canon :=
  canon_eq_of_equiv (union_equiv (M4Artifact.Equiv.refl M) M'.canon_equiv)

/-- **Canonicalising both operands doesn't change union under
canon**: `(M.canon.union M'.canon).canon = (M.union M').canon`.
Composed form of Ticks 1798 + 1799 — pre-canonicalising **both**
operands of a canon-projected union doesn't change the result.
One-line proof via chained `rw` of the two operand-canon invariance
lemmas. -/
theorem M4Artifact.canon_left_right_union_canon_eq {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    (M.canon.union M'.canon).canon = (M.union M').canon := by
  rw [canon_union_canon_eq, union_canon_canon_eq]

/-- **Canon of `fromCycle` union equals union of `fromCycleCanon`**:
`((fromCycle C₁ h₁).union (fromCycle C₂ h₂)).canon = (fromCycleCanon
C₁ h₁).union (fromCycleCanon C₂ h₂)`. Bridges the raw-form
union-of-fromCycles to the canonical-form union-of-fromCycleCanons
under a single canon projection. Composes Tick 1793 (canon
distributes over union as strict equality) with Tick 1792
(`(fromCycle).canon = fromCycleCanon`). -/
theorem M4Artifact.fromCycle_union_fromCycle_canon_eq {cfg : GameConfig}
    (C₁ C₂ : AdversarialClosedCycle cfg)
    (h₁ : GameState.init ∈ C₁.states) (h₂ : GameState.init ∈ C₂.states) :
    ((fromCycle C₁ h₁).union (fromCycle C₂ h₂)).canon =
      (fromCycleCanon C₁ h₁).union (fromCycleCanon C₂ h₂) :=
  union_canon_eq _ _

/-- **`fromCycleCanon` is a fixed point of `canon`**: `(fromCycleCanon
C h).canon = fromCycleCanon C h`. The canonical-form M4 constructor
produces canonical artifacts (Tick 1675 `fromCycleCanon_isCanonical`),
and canonical artifacts are fixed points of `canon` (Tick 1707
biconditional). Direct one-line composition. -/
theorem M4Artifact.fromCycleCanon_canon_eq {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    (fromCycleCanon C hinit).canon = fromCycleCanon C hinit :=
  canon_eq_iff_isCanonical.mpr (fromCycleCanon_isCanonical C hinit)

/-- **Canon of 3-cycle right-associated `fromCycle` union equals
union of `fromCycleCanon`**: extension of Tick 1801 to a right-
associated 3-cycle nested union. Applies `union_canon_eq` twice
(once at outer, once at inner) and relies on Tick 1792's defeq
`(fromCycle C h).canon = fromCycleCanon C h`. -/
theorem M4Artifact.fromCycle_union_3_canon_eq {cfg : GameConfig}
    (C₁ C₂ C₃ : AdversarialClosedCycle cfg)
    (h₁ : GameState.init ∈ C₁.states)
    (h₂ : GameState.init ∈ C₂.states)
    (h₃ : GameState.init ∈ C₃.states) :
    ((fromCycle C₁ h₁).union
        ((fromCycle C₂ h₂).union (fromCycle C₃ h₃))).canon =
      (fromCycleCanon C₁ h₁).union
        ((fromCycleCanon C₂ h₂).union (fromCycleCanon C₃ h₃)) := by
  rw [union_canon_eq, union_canon_eq]
  rfl

/-- **M4Artifact extensionality on data fields**: two M4 artifacts
are strictly equal if their `atlas` and `states` fields agree. The
`init_mem` and `closed` Prop-valued fields collapse via proof
irrelevance on Lean 4's `Prop`-typed structure components — no
explicit hypotheses needed on them. Useful as a clean abstraction
of the proof-irrelevance argument used in many strict-equality
proofs. -/
theorem M4Artifact.ext_of_atlas_states_eq {cfg : GameConfig}
    {M M' : M4Artifact cfg}
    (hatlas : M.atlas = M'.atlas)
    (hstates : M.states = M'.states) :
    M = M' := by
  obtain ⟨A, S, i, c⟩ := M
  obtain ⟨A', S', i', c'⟩ := M'
  subst hatlas
  subst hstates
  rfl

/-- **M4Artifact equality biconditional**: two M4 artifacts are
strictly equal iff their `atlas` and `states` fields agree.
Forward via `congrArg` on the field projections; reverse via
Tick 1804's `ext_of_atlas_states_eq`. Useful as a normal form
for M4-equality goals. -/
theorem M4Artifact.eq_iff {cfg : GameConfig}
    {M M' : M4Artifact cfg} :
    M = M' ↔ M.atlas = M'.atlas ∧ M.states = M'.states :=
  ⟨fun h => ⟨congrArg M4Artifact.atlas h, congrArg M4Artifact.states h⟩,
   fun ⟨h1, h2⟩ => ext_of_atlas_states_eq h1 h2⟩

/-- **M4Artifact inequality biconditional**: complement of Tick
1805's `eq_iff` — two artifacts differ iff their atlas OR their
states differ. Direct via `not_and_or` on the negation of `eq_iff`.
-/
theorem M4Artifact.ne_iff {cfg : GameConfig}
    {M M' : M4Artifact cfg} :
    M ≠ M' ↔ M.atlas ≠ M'.atlas ∨ M.states ≠ M'.states := by
  rw [Ne, eq_iff, not_and_or]

/-- **AdversarialClosedCycle extensionality on data fields**: two
cycles are strictly equal if their `states` and `solver` fields
agree. The `not_lost`, `valid`, and `closed` Prop-valued fields
collapse via proof irrelevance. Cycle-side parallel to Tick 1804's
`M4Artifact.ext_of_atlas_states_eq`. -/
theorem ext_of_states_solver_eq {cfg : GameConfig}
    {C₁ C₂ : AdversarialClosedCycle cfg}
    (hstates : C₁.states = C₂.states)
    (hsolver : C₁.solver = C₂.solver) :
    C₁ = C₂ := by
  obtain ⟨S₁, σ₁, n₁, v₁, c₁⟩ := C₁
  obtain ⟨S₂, σ₂, n₂, v₂, c₂⟩ := C₂
  subst hstates
  subst hsolver
  rfl

/-- **AdversarialClosedCycle equality biconditional**: two cycles
are strictly equal iff their `states` and `solver` fields agree.
Cycle-side parallel to Tick 1805's `M4Artifact.eq_iff`. Forward
via `congrArg` on field projections; reverse via Tick 1807's
`ext_of_states_solver_eq`. -/
theorem eq_iff {cfg : GameConfig}
    {C₁ C₂ : AdversarialClosedCycle cfg} :
    C₁ = C₂ ↔ C₁.states = C₂.states ∧ C₁.solver = C₂.solver :=
  ⟨fun h => ⟨congrArg AdversarialClosedCycle.states h,
              congrArg AdversarialClosedCycle.solver h⟩,
   fun ⟨h1, h2⟩ => ext_of_states_solver_eq h1 h2⟩

/-- **AdversarialClosedCycle inequality biconditional**: complement
of Tick 1808's `eq_iff` — two cycles differ iff their states OR
their solver differ. Cycle-side parallel to Tick 1806's
`M4Artifact.ne_iff`. Direct via `not_and_or` on the negation of
`eq_iff`. -/
theorem ne_iff {cfg : GameConfig}
    {C₁ C₂ : AdversarialClosedCycle cfg} :
    C₁ ≠ C₂ ↔ C₁.states ≠ C₂.states ∨ C₁.solver ≠ C₂.solver := by
  rw [Ne, eq_iff, not_and_or]

/-- **M4 atlas survives Solver round-trip under canon**:
`M.atlas.toSolver.toAtlas.canon M.states = M.canon.atlas`. The M4
artifact's atlas, materialised as a solver and then re-embedded
back to atlas form, equals the M4's canonical atlas (i.e.,
`M.atlas.canon M.states` by definition). Direct one-liner lifting
Tick 1815's `Atlas.toSolver_toAtlas_canon_eq_canon_of_closed`
using M's built-in `closed` field. -/
theorem M4Artifact.atlas_toSolver_toAtlas_canon_eq_canon_atlas
    {cfg : GameConfig} (M : M4Artifact cfg) :
    M.atlas.toSolver.toAtlas.canon M.states = M.canon.atlas :=
  Atlas.toSolver_toAtlas_canon_eq_canon_of_closed M.closed


/-- **M4 artifacts are determined by states + atlas**: `M = M' ↔
M.states = M'.states ∧ M.atlas = M'.atlas`. The Prop fields
(`init_mem`, `closed`) are irrelevant by proof irrelevance; equality
of data fields suffices for full M4 struct equality. -/
theorem M4Artifact.eq_iff_states_eq_and_atlas_eq {cfg : GameConfig}
    {M M' : M4Artifact cfg} :
    M = M' ↔ M.states = M'.states ∧ M.atlas = M'.atlas := by
  refine ⟨fun h => ?_, fun ⟨hs, ha⟩ => ?_⟩
  · subst h; exact ⟨rfl, rfl⟩
  · obtain ⟨a, s, i, c⟩ := M
    obtain ⟨a', s', i', c'⟩ := M'
    subst hs
    subst ha
    rfl

/-- **`ofIsClosedOn` equality biconditional**: two `ofIsClosedOn`-
constructed artifacts are strictly equal iff their state sets AND
atlases agree. Direct via the M4 eq biconditional + simp lemmas
(Ticks 1859/1860) reducing accessor projections. -/
theorem M4Artifact.ofIsClosedOn_eq_iff {cfg : GameConfig}
    {A₁ A₂ : Atlas cfg} {S₁ S₂ : Finset GameState}
    (h₁ : A₁.IsClosedOn cfg S₁) (hinit₁ : GameState.init ∈ S₁)
    (h₂ : A₂.IsClosedOn cfg S₂) (hinit₂ : GameState.init ∈ S₂) :
    M4Artifact.ofIsClosedOn h₁ hinit₁ = M4Artifact.ofIsClosedOn h₂ hinit₂ ↔
      S₁ = S₂ ∧ A₁ = A₂ :=
  M4Artifact.eq_iff_states_eq_and_atlas_eq

/-- **`M4Artifact.canon` is a congruence on Equiv**: if `M₁ ≈ M₂`, then
`M₁.canon ≈ M₂.canon`. Direct corollary of `canon_atlas_eq_of_equiv`
(Tick 1661): equivalent artifacts have identical canon atlases, so their
canons are trivially Equiv. -/
theorem M4Artifact.canon_equiv_canon_of_equiv {cfg : GameConfig}
    {M₁ M₂ : M4Artifact cfg} (h : M₁.Equiv M₂) :
    M₁.canon.Equiv M₂.canon := by
  refine ⟨h.1, fun g _ p _ => ?_⟩
  rw [canon_atlas_eq_of_equiv h]

/-- **Canon is idempotent at the Equiv level**: `M.canon.canon ≈ M.canon`.
Lifts Tick 1657's `canon_canon_atlas` field-level idempotence to the
Equiv level — applying `canon` twice yields an artifact in the same
equivalence class as one application. -/
theorem M4Artifact.canon_canon_equiv {cfg : GameConfig} (M : M4Artifact cfg) :
    M.canon.canon.Equiv M.canon := by
  refine ⟨rfl, fun g _ p _ => ?_⟩
  rw [canon_canon_atlas M]

/-- **`M.canon.canon ≈ M`** — direct corollary chaining the canon
idempotence with the canon-equiv property. Useful for chained
canon reasoning. -/
theorem M4Artifact.canon_canon_equiv_orig {cfg : GameConfig}
    (M : M4Artifact cfg) :
    M.canon.canon.Equiv M :=
  (canon_canon_equiv M).trans M.canon_equiv

/-- **M4 artifact states are nonempty**: `M.states.Nonempty`. Trivial
since `init ∈ M.states` by `M.init_mem`. Useful named lemma. -/
theorem M4Artifact.states_nonempty {cfg : GameConfig}
    (M : M4Artifact cfg) : M.states.Nonempty :=
  ⟨GameState.init, M.init_mem⟩

/-- **M4 artifact has positive state cardinality**: `0 < M.states.card`.
Direct via `Finset.card_pos` + `M.states_nonempty`. M4-level parallel
to cycle's `init_states_card_pos` (line ~3651) — but unconditional
since M4Artifact always has `init ∈ states`. -/
theorem M4Artifact.states_card_pos {cfg : GameConfig}
    (M : M4Artifact cfg) : 0 < M.states.card :=
  Finset.card_pos.mpr M.states_nonempty

/-- **M4 artifact has at least one state**: `1 ≤ M.states.card`.
Direct via Tick 1937 (`states_card_pos`). M4-level parallel to
cycle's `init_states_card_ge_one`. -/
theorem M4Artifact.states_card_ge_one {cfg : GameConfig}
    (M : M4Artifact cfg) : 1 ≤ M.states.card :=
  M.states_card_pos

/-- **M4 artifact has nonzero state cardinality**: `M.states.card
≠ 0`. Direct via Tick 1937 (`states_card_pos`) and `Nat.ne_of_gt`.
M4-level parallel to cycle's `init_states_card_ne_zero`. -/
theorem M4Artifact.states_card_ne_zero {cfg : GameConfig}
    (M : M4Artifact cfg) : M.states.card ≠ 0 :=
  Nat.ne_of_gt M.states_card_pos

/-- **`fromCycle` and `fromCycleCanon` are certificate-equivalent**: the
two single-cycle M4 artifacts (raw vs canon) agree on every active
query and have identical state sets. The raw form's atlas is
`C.solver.toAtlas`; the canon form's is `C.solver.toAtlas.canon C.states`.
On active queries, the latter reduces to the former via Tick 1615's
`canon_apply_of_active`. -/
theorem M4Artifact.fromCycle_equiv_fromCycleCanon {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    (fromCycle C hinit).Equiv (fromCycleCanon C hinit) :=
  ⟨rfl, fun _ hg _ hp =>
    (Atlas.canon_apply_of_active C.solver.toAtlas hg hp).symm⟩

/-- **`fromCycle` respects cycle Equiv** (functoriality): equivalent
cycles `C₁ ≈ C₂` produce equivalent M4 artifacts `fromCycle C₁ h₁ ≈
fromCycle C₂ h₂`. Connects the cycle-equivalence algebra (Tick 1677)
to the M4-artifact-equivalence algebra (Tick 1657). The state-set
clause comes directly from the cycle Equiv; the atlas-agreement clause
unfolds `Solver.toAtlas` and applies the cycle Equiv's solver agreement
at every active query. -/
theorem M4Artifact.fromCycle_equiv_of_cycle_equiv {cfg : GameConfig}
    {C₁ C₂ : AdversarialClosedCycle cfg} (heq : C₁.Equiv C₂)
    {h₁ : GameState.init ∈ C₁.states}
    {h₂ : GameState.init ∈ C₂.states} :
    (fromCycle C₁ h₁).Equiv (fromCycle C₂ h₂) := by
  refine ⟨heq.1, fun g hg p hp => ?_⟩
  change C₁.solver.toAtlas g p = C₂.solver.toAtlas g p
  unfold Solver.toAtlas
  rw [heq.2 g hg p hp]

/-- **`fromCycleCanon` respects cycle Equiv** (canonical-form
functoriality): equivalent cycles `C₁ ≈ C₂` produce equivalent
canonical M4 artifacts `fromCycleCanon C₁ h₁ ≈ fromCycleCanon C₂ h₂`.
Parallel to Tick 1678's `fromCycle_equiv_of_cycle_equiv`. Proof:
state-set clause from `heq.1`; atlas clause reduces canon to underlying
solver via `canon_apply_of_active` on both sides, then `Solver.toAtlas`
unfolds and `heq.2` provides the active-query agreement. -/
theorem M4Artifact.fromCycleCanon_equiv_of_cycle_equiv {cfg : GameConfig}
    {C₁ C₂ : AdversarialClosedCycle cfg} (heq : C₁.Equiv C₂)
    {h₁ : GameState.init ∈ C₁.states}
    {h₂ : GameState.init ∈ C₂.states} :
    (fromCycleCanon C₁ h₁).Equiv (fromCycleCanon C₂ h₂) := by
  refine ⟨heq.1, fun g hg p hp => ?_⟩
  have hg₁ : g ∈ C₁.states := hg
  have hg₂ : g ∈ C₂.states := heq.1 ▸ hg₁
  change (C₁.solver.toAtlas.canon C₁.states) g p =
         (C₂.solver.toAtlas.canon C₂.states) g p
  rw [Atlas.canon_apply_of_active _ hg₁ hp,
      Atlas.canon_apply_of_active _ hg₂ hp]
  unfold Solver.toAtlas
  rw [heq.2 g hg₁ p hp]

/-- **`fromCycle` reflects cycle Equiv** (reverse of Tick 1678).
If `(fromCycle C₁ h₁).Equiv (fromCycle C₂ h₂)`, then `C₁.Equiv C₂`.
Together with Tick 1678's `fromCycle_equiv_of_cycle_equiv` (forward
direction), establishes **`fromCycle` as a fully faithful embedding**
of cycle Equiv into M4 Equiv — two cycles are equivalent iff their
`fromCycle` M4 artifacts are equivalent.

Proof: state-set clause from `h.1` directly. Solver clause: extract
the M4 atlas agreement at the active query `(g, p)`; both sides
unfold to `some (Cᵢ.solver g p)` via `Solver.toAtlas` definition;
apply `Option.some.inj` to extract solver equality. -/
theorem cycle_equiv_of_fromCycle_equiv {cfg : GameConfig}
    {C₁ C₂ : AdversarialClosedCycle cfg}
    {h₁ : GameState.init ∈ C₁.states}
    {h₂ : GameState.init ∈ C₂.states}
    (h : (M4Artifact.fromCycle C₁ h₁).Equiv
         (M4Artifact.fromCycle C₂ h₂)) :
    C₁.Equiv C₂ := by
  refine ⟨h.1, fun g hg p hp => ?_⟩
  have hg' : g ∈ (M4Artifact.fromCycle C₁ h₁).states := hg
  have hsome : (M4Artifact.fromCycle C₁ h₁).atlas g p =
               (M4Artifact.fromCycle C₂ h₂).atlas g p :=
    h.2 g hg' p hp
  change some (C₁.solver g p) = some (C₂.solver g p) at hsome
  exact Option.some.inj hsome

/-- **`fromCycleCanon` reflects cycle Equiv** (reverse of Tick 1679).
Canonical-form parallel to Tick 1695. If `(fromCycleCanon C₁ h₁).Equiv
(fromCycleCanon C₂ h₂)`, then `C₁.Equiv C₂`. Together with Tick 1679,
establishes **`fromCycleCanon` is also a fully faithful embedding** of
cycle Equiv into M4 Equiv.

Proof: state-set clause from `h.1`. Solver clause: the M4 atlas
agreement at `(g, p)` reduces to canon-of-solver-toAtlas equality;
`Atlas.canon_apply_of_active` strips both canon wrappers (since
`g ∈ Cᵢ.states, p ∈ g.bag`), exposing `some (Cᵢ.solver g p)`;
`Option.some.inj` extracts solver equality. -/
theorem cycle_equiv_of_fromCycleCanon_equiv {cfg : GameConfig}
    {C₁ C₂ : AdversarialClosedCycle cfg}
    {h₁ : GameState.init ∈ C₁.states}
    {h₂ : GameState.init ∈ C₂.states}
    (h : (M4Artifact.fromCycleCanon C₁ h₁).Equiv
         (M4Artifact.fromCycleCanon C₂ h₂)) :
    C₁.Equiv C₂ := by
  have hsts : C₁.states = C₂.states := h.1
  refine ⟨hsts, fun g hg p hp => ?_⟩
  have hg₁ : g ∈ C₁.states := hg
  have hg₂ : g ∈ C₂.states := hsts ▸ hg₁
  have hsome : (M4Artifact.fromCycleCanon C₁ h₁).atlas g p =
               (M4Artifact.fromCycleCanon C₂ h₂).atlas g p :=
    h.2 g hg p hp
  change (C₁.solver.toAtlas.canon C₁.states) g p =
         (C₂.solver.toAtlas.canon C₂.states) g p at hsome
  rw [Atlas.canon_apply_of_active _ hg₁ hp,
      Atlas.canon_apply_of_active _ hg₂ hp] at hsome
  change some (C₁.solver g p) = some (C₂.solver g p) at hsome
  exact Option.some.inj hsome

/-- **Biconditional: cycle Equiv ↔ `fromCycle` M4 Equiv**.
Crystallises the "fully faithful embedding" property (Tick 1678
forward + Tick 1695 reverse) into a single iff for ergonomic use. -/
theorem cycle_equiv_iff_fromCycle_equiv {cfg : GameConfig}
    {C₁ C₂ : AdversarialClosedCycle cfg}
    {h₁ : GameState.init ∈ C₁.states}
    {h₂ : GameState.init ∈ C₂.states} :
    C₁.Equiv C₂ ↔
      (M4Artifact.fromCycle C₁ h₁).Equiv (M4Artifact.fromCycle C₂ h₂) :=
  ⟨fun heq => M4Artifact.fromCycle_equiv_of_cycle_equiv heq,
   cycle_equiv_of_fromCycle_equiv⟩

/-- **Biconditional: cycle Equiv ↔ `fromCycleCanon` M4 Equiv**.
Canon-form parallel. Crystallises the "canonical-form fully faithful
embedding" property (Tick 1679 forward + Tick 1696 reverse) into a
single iff. -/
theorem cycle_equiv_iff_fromCycleCanon_equiv {cfg : GameConfig}
    {C₁ C₂ : AdversarialClosedCycle cfg}
    {h₁ : GameState.init ∈ C₁.states}
    {h₂ : GameState.init ∈ C₂.states} :
    C₁.Equiv C₂ ↔
      (M4Artifact.fromCycleCanon C₁ h₁).Equiv
        (M4Artifact.fromCycleCanon C₂ h₂) :=
  ⟨fun heq => M4Artifact.fromCycleCanon_equiv_of_cycle_equiv heq,
   cycle_equiv_of_fromCycleCanon_equiv⟩

/-- **`fromCycle` is injective on cycles**: if `fromCycle C₁ h₁ =
fromCycle C₂ h₂` as M4 artifacts, then `C₁ = C₂` as cycles. Uses
Tick 1752's M4 struct equality + Tick 1714's
`Solver.toAtlas_toSolver` round-trip + Tick 1753's cycle struct
equality to extract `C₁ = C₂` from the M4 equality.

This **strict injectivity** strengthens Tick 1695's cycle-Equiv
reflection — when `fromCycle` outputs are equal (not just Equiv),
inputs are equal. -/
theorem M4Artifact.fromCycle_injective {cfg : GameConfig}
    {C₁ C₂ : AdversarialClosedCycle cfg}
    {h₁ : GameState.init ∈ C₁.states}
    {h₂ : GameState.init ∈ C₂.states}
    (h : M4Artifact.fromCycle C₁ h₁ = M4Artifact.fromCycle C₂ h₂) :
    C₁ = C₂ := by
  rw [eq_iff_states_eq_and_solver_eq]
  obtain ⟨hs, hatlas⟩ := M4Artifact.eq_iff_states_eq_and_atlas_eq.mp h
  have hs' : C₁.states = C₂.states := hs
  have hatlas' : C₁.solver.toAtlas = C₂.solver.toAtlas := hatlas
  refine ⟨hs', ?_⟩
  have h1 : (C₁.solver.toAtlas).toSolver = C₁.solver :=
    Solver.toAtlas_toSolver C₁.solver
  have h2 : (C₂.solver.toAtlas).toSolver = C₂.solver :=
    Solver.toAtlas_toSolver C₂.solver
  rw [← h1, ← h2, hatlas']

/-- **`fromCycleCanon` equality implies cycle Equiv** (NOT strict
injection): unlike Tick 1754's `fromCycle_injective`, `fromCycleCanon`
discards inactive-query solver behavior via canon — so strict
equality of `fromCycleCanon C₁ = fromCycleCanon C₂` only recovers
cycle Equiv, not strict cycle equality. Direct via Tick 1696
applied to the (M4-strict ⇒ M4-Equiv) coercion. -/
theorem fromCycleCanon_eq_implies_cycle_equiv {cfg : GameConfig}
    {C₁ C₂ : AdversarialClosedCycle cfg}
    {h₁ : GameState.init ∈ C₁.states}
    {h₂ : GameState.init ∈ C₂.states}
    (h : M4Artifact.fromCycleCanon C₁ h₁ =
         M4Artifact.fromCycleCanon C₂ h₂) :
    C₁.Equiv C₂ := by
  have heq : (M4Artifact.fromCycleCanon C₁ h₁).Equiv
             (M4Artifact.fromCycleCanon C₂ h₂) := by
    rw [h]; exact M4Artifact.Equiv.refl _
  exact cycle_equiv_of_fromCycleCanon_equiv heq

/-- **`fromCycle C` is certificate-equivalent to `buildFromCycles_M4Artifact
[C]`**: the direct cycle-to-artifact constructor and the singleton-list
constructor produce equivalent M4 artifacts. By Tick 1636's
`singleton_raw_atlas`, the buildFromCycles atlas reduces to
`C.solver.toAtlas.restrict C.states` which agrees with
`C.solver.toAtlas` on active queries via `restrict_apply_of_mem`. -/
theorem M4Artifact.fromCycle_equiv_buildFromCycles_singleton
    {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    (fromCycle C hinit).Equiv
      (buildFromCycles_M4Artifact [C]
        ⟨C, List.mem_singleton.mpr rfl, hinit⟩) := by
  refine ⟨?_, ?_⟩
  · change C.states = ∅ ∪ C.states
    exact (Finset.empty_union _).symm
  · intro g hg p _hp
    change C.solver.toAtlas g p =
           (buildFromCycles (Atlas.empty cfg) ∅ [C]).1 g p
    rw [singleton_raw_atlas]
    exact (Atlas.restrict_apply_of_mem _ hg p).symm

/-- **`fromCycleCanon C ≈ buildFromCyclesCanon_M4Artifact [C]`**: the
canonical-form variant of Tick 1671. Both construction styles produce
certificate-equivalent canon M4 artifacts. By Tick 1636's
`singleton_canon_atlas`, the buildFromCyclesCanon atlas is exactly the
cycle's canon atlas. -/
theorem M4Artifact.fromCycleCanon_equiv_buildFromCyclesCanon_singleton
    {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    (fromCycleCanon C hinit).Equiv
      (buildFromCyclesCanon_M4Artifact [C]
        ⟨C, List.mem_singleton.mpr rfl, hinit⟩) := by
  refine ⟨?_, ?_⟩
  · change C.states = ∅ ∪ C.states
    exact (Finset.empty_union _).symm
  · intro g _hg p _hp
    change (C.solver.toAtlas.canon C.states) g p =
           (buildFromCyclesCanon (Atlas.empty cfg) ∅ [C]).1 g p
    rw [singleton_canon_atlas]

/-- **`(fromCycle C h).canon ≈ fromCycleCanon C h`**: applying `canon` to a
`fromCycle` artifact produces the same data as `fromCycleCanon` directly.
Both have atlas `C.solver.toAtlas.canon C.states` and state set `C.states`
— structurally identical, trivially Equiv. Identifies the canon
projection on `fromCycle` with the direct canonical constructor. -/
theorem M4Artifact.fromCycle_canon_equiv_fromCycleCanon {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    (fromCycle C hinit).canon.Equiv (fromCycleCanon C hinit) :=
  ⟨rfl, fun _ _ _ _ => rfl⟩

/-- **`(fromCycle C h).canon = fromCycleCanon C h`** (strict equality).
Strengthens the Equiv form (Tick 1672) to full M4 artifact equality:
both have the same atlas, states, init_mem, and closed fields. The
two `closed` proofs collapse by proof irrelevance on `IsClosedOn`'s
`Prop`-valued structure — both are proofs of
`(C.solver.toAtlas.canon C.states).IsClosedOn cfg C.states`. -/
theorem M4Artifact.fromCycle_canon_eq_fromCycleCanon {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    (fromCycle C hinit).canon = fromCycleCanon C hinit := rfl

/-- **M4 artifact union covers its left input**: `M.states ⊆ (M.union M').states`.
The union artifact certifies at least all states of the left input.
Trivial via `Finset.subset_union_left`. -/
theorem M4Artifact.states_subset_union_left {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    M.states ⊆ (M.union M').states :=
  Finset.subset_union_left

/-- **M4 artifact union covers its right input**: `M'.states ⊆ (M.union M').states`.
Right-side analogue of `states_subset_union_left`. -/
theorem M4Artifact.states_subset_union_right {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    M'.states ⊆ (M.union M').states :=
  Finset.subset_union_right

/-- **M4 atlas's dom equals its certified state set**: `M.atlas.dom
M.states = M.states`. Combines Tick 1609's `dom_subset` (`A.dom S ⊆ S`)
with Tick 1650's `states_subset_dom_of_M4Artifact` (`M.states ⊆
M.atlas.dom M.states`). The certified state set is exactly the atlas's
canonical domain — no spurious "phantom" coverage. -/
theorem M4Artifact.atlas_dom_states_eq {cfg : GameConfig}
    (M : M4Artifact cfg) : M.atlas.dom M.states = M.states :=
  Finset.Subset.antisymm (M.atlas.dom_subset M.states)
    (states_subset_dom_of_M4Artifact M)

/-- **M4 atlas's dom cardinality equals its states cardinality**:
`(M.atlas.dom M.states).card = M.states.card`. Direct corollary
of Tick 1651 (`atlas_dom_states_eq`). Useful for cardinality
arithmetic over the atlas's certified domain. -/
theorem M4Artifact.atlas_dom_states_card_eq {cfg : GameConfig}
    (M : M4Artifact cfg) : (M.atlas.dom M.states).card = M.states.card := by
  rw [M.atlas_dom_states_eq]

/-- **M4 atlas's restricted dom cardinality equals states
cardinality**: `((M.atlas.restrict M.states).dom M.states).card =
M.states.card`. M4-level lift of Tick 1964. -/
theorem M4Artifact.atlas_restrict_dom_states_card_eq {cfg : GameConfig}
    (M : M4Artifact cfg) :
    ((M.atlas.restrict M.states).dom M.states).card = M.states.card :=
  M.closed.restrict_dom_card_eq

/-- **M4 atlas trace stays in states from any starting state**: generalises
Tick 1652 to start from any state in `M.states`, not just `init`. Under
a legal piece sequence from the starting state's bag, the trace under
the M4 artifact's materialised solver stays in `M.states`. -/
theorem M4Artifact.toSolver_trace_mem_states_from_mem {cfg : GameConfig}
    (M : M4Artifact cfg) {g₀ : GameState} (hg₀ : g₀ ∈ M.states)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    adversarialTrace cfg M.atlas.toSolver t g₀ n ∈ M.states := by
  induction n with
  | zero => exact hg₀
  | succ k ih =>
    rw [adversarialTrace_succ]
    set g := adversarialTrace cfg M.atlas.toSolver t g₀ k with hg_def
    have hg : g ∈ M.states := ih
    have hp : t k ∈ g.bag := by
      have := hl k
      rw [Bag.canDraw_iff_mem,
          ← adversarialTrace_bag_from cfg M.atlas.toSolver t g₀ k] at this
      exact this
    have htot := M.closed.total g hg (t k) hp
    obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp htot
    rw [Atlas.toSolver_apply_of_some hpl]
    exact M.closed.closed g hg (t k) hp pl hpl

/-- **M4 atlas trace is never lost from any starting state**: composes
Tick 1666's `toSolver_trace_mem_states_from_mem` with the `not_lost`
clause of `IsClosedOn`. The M4 artifact's materialised solver is
non-losing from any certified starting state, not just `init`. -/
theorem M4Artifact.toSolver_trace_not_lost_from_mem {cfg : GameConfig}
    (M : M4Artifact cfg) {g₀ : GameState} (hg₀ : g₀ ∈ M.states)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    ¬ (adversarialTrace cfg M.atlas.toSolver t g₀ n).lost cfg :=
  M.closed.not_lost _ (toSolver_trace_mem_states_from_mem M hg₀ hl n)

/-- **M4 atlas's solver `SolvesTetrisFrom`** every certified state: for
any `g₀ ∈ M.states`, the M4 artifact's materialised solver satisfies
`SolvesTetrisFrom cfg g₀`. Direct corollary of Tick 1666's
`toSolver_trace_not_lost_from_mem`. -/
theorem M4Artifact.toSolver_solvesTetrisFrom {cfg : GameConfig}
    (M : M4Artifact cfg) {g₀ : GameState} (hg₀ : g₀ ∈ M.states) :
    SolvesTetrisFrom cfg g₀ M.atlas.toSolver :=
  fun _ hl n => toSolver_trace_not_lost_from_mem M hg₀ hl n

end AdversarialClosedCycle

/-- **Closed atlas ⇒ cycle (the converse).** Given any `Atlas.IsClosedOn cfg
A S` certificate, extract an `AdversarialClosedCycle` with `states = S` by
choosing a placement at each `(g, p)` query via `Classical.choose` on
`hclosed.total`. The resulting solver is the Atlas's *materialisation* — for
each in-domain query, it returns whatever the `Option`-typed atlas would
return; out of domain it defaults to an arbitrary placement. This closes the
algebraic equivalence `AdversarialClosedCycle ↔ closed Atlas` started in
Tick 1601's forward direction. -/
noncomputable def AdversarialClosedCycle.ofClosedAtlas {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState}
    (hclosed : A.IsClosedOn cfg S) : AdversarialClosedCycle cfg where
  states := S
  solver := fun g p => (A g p).getD ⟨p, ⟨0, by norm_num⟩, 0⟩
  not_lost := hclosed.not_lost
  valid := fun g hg p hp => by
    have htot := hclosed.total g hg p hp
    obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp htot
    rw [hpl, Option.getD_some]
    exact hclosed.valid g hg p hp pl hpl
  closed := fun g hg p hp => by
    have htot := hclosed.total g hg p hp
    obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp htot
    rw [hpl, Option.getD_some]
    exact hclosed.closed g hg p hp pl hpl

/-- The cycle extracted from a closed atlas has `states = S`. -/
@[simp] theorem AdversarialClosedCycle.ofClosedAtlas_states {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState}
    (hclosed : A.IsClosedOn cfg S) :
    (AdversarialClosedCycle.ofClosedAtlas hclosed).states = S := rfl

/-- **`ofClosedAtlas` preserves the canonical atlas exactly**:
`(ofClosedAtlas h).solver.toAtlas.canon S = A.canon S`. The cycle
extracted from a closed atlas, when re-lifted to an atlas and
canonicalised at `S`, recovers the original closed atlas's
canonicalisation.

Proof: pointwise via `funext` + case-split on active/inactive:
- **Active**: by closure totality, `A g p = some pl` for some `pl`;
  unfold `Solver.toAtlas` exposes `some ((A g p).getD _)`,
  `Option.getD_some` collapses to `A g p`.
- **Inactive**: both sides reduce to `none` via
  `Atlas.canon_apply_of_inactive`. -/
theorem AdversarialClosedCycle.ofClosedAtlas_solver_toAtlas_canon
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) :
    (AdversarialClosedCycle.ofClosedAtlas h).solver.toAtlas.canon S =
      A.canon S := by
  funext g p
  by_cases hac : g ∈ S ∧ p ∈ g.bag
  · obtain ⟨hg, hp⟩ := hac
    rw [Atlas.canon_apply_of_active _ hg hp,
        Atlas.canon_apply_of_active _ hg hp]
    obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp (h.total g hg p hp)
    unfold Solver.toAtlas
    change some ((A g p).getD _) = A g p
    rw [hpl, Option.getD_some]
  · rw [Atlas.canon_apply_of_inactive _ hac,
        Atlas.canon_apply_of_inactive _ hac]

/-- **Direct strict atlas equality**: `(fromCycleCanon (ofClosedAtlas
M.closed) M.init_mem).atlas = M.canon.atlas`. The atlas of the
canonical reconstruction equals M.canon's atlas — strict function
equality at the M4 artifact's atlas level.

Tighter and more direct than Tick 1685b (which used Equiv-routing).
Direct from Tick 1717 with `S := M.states`: by definition of
`fromCycleCanon`, the LHS atlas is
`(ofClosedAtlas M.closed).solver.toAtlas.canon (ofClosedAtlas
M.closed).states = (ofClosedAtlas M.closed).solver.toAtlas.canon
M.states`, which by Tick 1717 equals `M.atlas.canon M.states =
M.canon.atlas`. -/
theorem AdversarialClosedCycle.M4Artifact.fromCycleCanon_ofClosedAtlas_atlas_eq_canon
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    (AdversarialClosedCycle.M4Artifact.fromCycleCanon
      (AdversarialClosedCycle.ofClosedAtlas M.closed)
      M.init_mem).atlas = M.canon.atlas := by
  change (AdversarialClosedCycle.ofClosedAtlas M.closed).solver.toAtlas.canon
           M.states = M.atlas.canon M.states
  exact AdversarialClosedCycle.ofClosedAtlas_solver_toAtlas_canon M.closed

/-- **`ofClosedAtlas` respects atlas active-query agreement** (reverse-
direction functoriality). If two closed atlases on the same state set
agree on every active query `(g ∈ S, p ∈ g.bag)`, then the cycles
extracted by `ofClosedAtlas` are cycle-equivalent (Tick 1677). This is
the M4-to-cycle dual of Tick 1678/1679 (which proved `fromCycle` and
`fromCycleCanon` respect cycle Equiv). Closes the loop: cycle ↔ atlas
constructors both descend to equivalence-class level. Proof: state-set
clause is `rfl` (both cycles have states `S`); solver clause unfolds
`ofClosedAtlas.solver` to `(Aᵢ g p).getD _` and the agreement
`A₁ g p = A₂ g p` yields equal `Option.getD` results. -/
theorem AdversarialClosedCycle.ofClosedAtlas_equiv_of_atlas_agree
    {cfg : GameConfig} {A₁ A₂ : Atlas cfg} {S : Finset GameState}
    (h₁ : A₁.IsClosedOn cfg S) (h₂ : A₂.IsClosedOn cfg S)
    (hagree : ∀ g ∈ S, ∀ p ∈ g.bag, A₁ g p = A₂ g p) :
    (AdversarialClosedCycle.ofClosedAtlas h₁).Equiv
      (AdversarialClosedCycle.ofClosedAtlas h₂) := by
  refine ⟨rfl, fun g hg p hp => ?_⟩
  have hg' : g ∈ S := hg
  change (A₁ g p).getD _ = (A₂ g p).getD _
  rw [hagree g hg' p hp]

/-- **`ofClosedAtlas` reflects atlas active-query agreement** (reverse
of Tick 1682). If `(ofClosedAtlas h₁).Equiv (ofClosedAtlas h₂)` for
two closed atlases on the same state set `S`, then `A₁` and `A₂`
agree on every active query (`g ∈ S, p ∈ g.bag`). Together with Tick
1682 (forward), establishes **`ofClosedAtlas` is a fully faithful
functor** on atlas-agreement classes — parallel to Ticks 1695/1696
for `fromCycle`/`fromCycleCanon`.

Proof: extract `pl₁, pl₂` via `Option.isSome_iff_exists` from the
closed atlases' totality; use the cycle Equiv's solver agreement
(`heq.2 g hg p hp`) which after `change` exposes
`(A₁ g p).getD _ = (A₂ g p).getD _`; rewrite via `hpl₁, hpl₂,
Option.getD_some` to get `pl₁ = pl₂`; finally combine via
`congrArg some` to derive `some pl₁ = some pl₂`, i.e.,
`A₁ g p = A₂ g p` after the symmetric rewrites. -/
theorem AdversarialClosedCycle.atlas_agree_of_ofClosedAtlas_equiv
    {cfg : GameConfig} {A₁ A₂ : Atlas cfg} {S : Finset GameState}
    (h₁ : A₁.IsClosedOn cfg S) (h₂ : A₂.IsClosedOn cfg S)
    (heq : (AdversarialClosedCycle.ofClosedAtlas h₁).Equiv
           (AdversarialClosedCycle.ofClosedAtlas h₂)) :
    ∀ g ∈ S, ∀ p ∈ g.bag, A₁ g p = A₂ g p := by
  intros g hg p hp
  obtain ⟨pl₁, hpl₁⟩ := Option.isSome_iff_exists.mp (h₁.total g hg p hp)
  obtain ⟨pl₂, hpl₂⟩ := Option.isSome_iff_exists.mp (h₂.total g hg p hp)
  have hsolver : (AdversarialClosedCycle.ofClosedAtlas h₁).solver g p =
                 (AdversarialClosedCycle.ofClosedAtlas h₂).solver g p :=
    heq.2 g hg p hp
  change (A₁ g p).getD _ = (A₂ g p).getD _ at hsolver
  rw [hpl₁, hpl₂, Option.getD_some, Option.getD_some] at hsolver
  rw [hpl₁, hpl₂, hsolver]

/-- **Biconditional: `ofClosedAtlas` Equiv ↔ atlas active-query
agreement**. Crystallises the fully faithful M4→cycle functoriality
(Tick 1682 forward + Tick 1698 reverse) into a single iff. Closed
atlases on a shared state set produce Equiv cycles iff they agree on
every active query. -/
theorem AdversarialClosedCycle.ofClosedAtlas_equiv_iff_atlas_agree
    {cfg : GameConfig} {A₁ A₂ : Atlas cfg} {S : Finset GameState}
    (h₁ : A₁.IsClosedOn cfg S) (h₂ : A₂.IsClosedOn cfg S) :
    (AdversarialClosedCycle.ofClosedAtlas h₁).Equiv
      (AdversarialClosedCycle.ofClosedAtlas h₂) ↔
    ∀ g ∈ S, ∀ p ∈ g.bag, A₁ g p = A₂ g p :=
  ⟨AdversarialClosedCycle.atlas_agree_of_ofClosedAtlas_equiv h₁ h₂,
   AdversarialClosedCycle.ofClosedAtlas_equiv_of_atlas_agree h₁ h₂⟩

/-- **`fromCycle ∘ ofClosedAtlas ≈ ofIsClosedOn`**: building an M4
artifact via cycle extraction + cycle-to-M4 conversion is Equiv to
directly constructing via `ofIsClosedOn`. Atlases differ on inactive
queries (`A.toSolver.toAtlas` returns `some <default>` for `none`-
valued queries, while `A` returns `none`), so the relationship is
Equiv (active-query agreement), not strict equality. Direct via
`toSolver_toAtlas_apply_of_some` + `h.total` on active queries. -/
theorem AdversarialClosedCycle.M4Artifact.fromCycle_ofClosedAtlas_equiv_ofIsClosedOn
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    (AdversarialClosedCycle.M4Artifact.fromCycle
      (AdversarialClosedCycle.ofClosedAtlas h) hinit).Equiv
      (AdversarialClosedCycle.M4Artifact.ofIsClosedOn h hinit) := by
  refine ⟨rfl, fun g hg p hp => ?_⟩
  change A.toSolver.toAtlas g p = A g p
  obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp (h.total g hg p hp)
  rw [Atlas.toSolver_toAtlas_apply_of_some hpl, hpl]

/-- **Canon-projected strict equality of the two M4 routes**:
`(fromCycle (ofClosedAtlas h) hinit).canon = (ofIsClosedOn h
hinit).canon`. Strengthens Tick 1869's Equiv to strict M4-artifact
equality at the canonical projection — at canon level, both routes
land on the same artifact. One-liner via `canon_eq_of_equiv` (Tick
1705) applied to Tick 1869. -/
theorem AdversarialClosedCycle.M4Artifact.fromCycle_ofClosedAtlas_canon_eq_ofIsClosedOn_canon
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    (AdversarialClosedCycle.M4Artifact.fromCycle
      (AdversarialClosedCycle.ofClosedAtlas h) hinit).canon =
      (AdversarialClosedCycle.M4Artifact.ofIsClosedOn h hinit).canon :=
  M4Artifact.canon_eq_of_equiv
    (AdversarialClosedCycle.M4Artifact.fromCycle_ofClosedAtlas_equiv_ofIsClosedOn h hinit)

/-- **`fromCycleCanon ∘ ofClosedAtlas = ofIsClosedOn.canon`**:
the canon-form cycle-route to M4 equals the canon-projected direct
route. Direct via Tick 1792's `(fromCycle C h).canon =
fromCycleCanon C h` (rfl) applied to Tick 1870. -/
theorem AdversarialClosedCycle.M4Artifact.fromCycleCanon_ofClosedAtlas_eq_ofIsClosedOn_canon
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    AdversarialClosedCycle.M4Artifact.fromCycleCanon
      (AdversarialClosedCycle.ofClosedAtlas h) hinit =
      (AdversarialClosedCycle.M4Artifact.ofIsClosedOn h hinit).canon :=
  AdversarialClosedCycle.M4Artifact.fromCycle_ofClosedAtlas_canon_eq_ofIsClosedOn_canon h hinit

/-- **`fromCycleCanon ∘ ofClosedAtlas = ofIsClosedOn` (strict
equality) for canonical input atlases**: when `A = A.canon S`,
the canon-route equals the direct route without an outer canon.
Composes Tick 1871 with `canon_eq_self_of_isCanonical` applied
through Tick 1865's iff. -/
theorem AdversarialClosedCycle.M4Artifact.fromCycleCanon_ofClosedAtlas_eq_ofIsClosedOn_of_canonical
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    (hcanon : A = A.canon S) :
    AdversarialClosedCycle.M4Artifact.fromCycleCanon
      (AdversarialClosedCycle.ofClosedAtlas h) hinit =
      AdversarialClosedCycle.M4Artifact.ofIsClosedOn h hinit := by
  rw [AdversarialClosedCycle.M4Artifact.fromCycleCanon_ofClosedAtlas_eq_ofIsClosedOn_canon]
  exact (M4Artifact.canon_eq_self_of_isCanonical
    ((AdversarialClosedCycle.M4Artifact.ofIsClosedOn_isCanonical_iff h hinit).mpr hcanon)).symm

/-- **Cycles from canon and original M4 are Equiv**: for any M4
artifact `M`, the cycles extracted via `ofClosedAtlas` from `M.canon`
and `M` are cycle-Equiv: `ofClosedAtlas M.canon.closed ≈ ofClosedAtlas
M.closed`.

A key bridge between M4 canon algebra and cycle algebra:
canonicalising M before extracting a cycle gives the same cycle-Equiv
class as extracting directly. Direct via Tick 1682's
`ofClosedAtlas_equiv_of_atlas_agree` applied with the active-query
agreement `(M.atlas.canon M.states) g p = M.atlas g p` from
`Atlas.canon_apply_of_active`. -/
theorem AdversarialClosedCycle.M4Artifact.ofClosedAtlas_canon_equiv_ofClosedAtlas
    {cfg : GameConfig}
    (M : AdversarialClosedCycle.M4Artifact cfg) :
    (AdversarialClosedCycle.ofClosedAtlas M.canon.closed).Equiv
      (AdversarialClosedCycle.ofClosedAtlas M.closed) :=
  AdversarialClosedCycle.ofClosedAtlas_equiv_of_atlas_agree
    M.canon.closed M.closed
    (fun _ hg _ hp => Atlas.canon_apply_of_active M.atlas hg hp)

/-- **Round-trip identity: `ofClosedAtlas ∘ fromCycle ≈ id`**. Starting
from a cycle `C` and constructing the M4 artifact via `fromCycle`,
then extracting a cycle via `ofClosedAtlas` of the artifact's closure
certificate, recovers a cycle Equiv to the original `C`. The simplest
round-trip in the cycle ↔ M4-artifact ↔ closed-atlas correspondence.

Proof: both Equiv clauses reduce to `rfl` — the round-trip is
**definitionally the identity** on `states` and `solver` at every query
(not just active queries, since `Solver.toAtlas` produces `some _` for
every input). Specifically, the reduction chain
`(ofClosedAtlas (fromCycle C h).closed).solver g p
   = ((fromCycle C h).atlas g p).getD _
   = (C.solver.toAtlas g p).getD _
   = (some (C.solver g p)).getD _
   = C.solver g p`
collapses by structure projection + β + `Option.getD` pattern match. -/
theorem AdversarialClosedCycle.ofClosedAtlas_fromCycle_equiv
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) :
    (AdversarialClosedCycle.ofClosedAtlas
       (AdversarialClosedCycle.M4Artifact.fromCycle C h).closed).Equiv C :=
  ⟨rfl, fun _ _ _ _ => rfl⟩

/-- **Round-trip identity: `ofClosedAtlas ∘ fromCycleCanon ≈ id`**.
Canonical-form parallel to Tick 1683. Starting from cycle `C`,
constructing the canonical M4 artifact via `fromCycleCanon`, then
extracting a cycle via `ofClosedAtlas` of its closure certificate,
recovers a cycle Equiv to the original.

Proof: unlike Tick 1683's pure `rfl`, the canon variant requires
unfolding `canon` at active queries. The reduction chain
`(ofClosedAtlas (fromCycleCanon C h).closed).solver g p
   = ((C.solver.toAtlas.canon C.states) g p).getD _
   = (C.solver.toAtlas g p).getD _   -- canon_apply_of_active
   = (some (C.solver g p)).getD _     -- defn of Solver.toAtlas
   = C.solver g p                     -- Option.getD pattern match`
applies `Atlas.canon_apply_of_active` for the canon-stripping step;
the rest collapses by `rfl`. State-set clause is `rfl` (both = `C.states`). -/
theorem AdversarialClosedCycle.ofClosedAtlas_fromCycleCanon_equiv
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) :
    (AdversarialClosedCycle.ofClosedAtlas
       (AdversarialClosedCycle.M4Artifact.fromCycleCanon C h).closed).Equiv C := by
  refine ⟨rfl, fun g hg p hp => ?_⟩
  have hg' : g ∈ C.states := hg
  change ((C.solver.toAtlas.canon C.states) g p).getD _ = C.solver g p
  rw [Atlas.canon_apply_of_active _ hg' hp]
  unfold Solver.toAtlas
  rw [Option.getD_some]

/-- **Atlas ↔ cycle existence equivalence**: an `init`-containing closed
atlas exists iff an `init`-containing `AdversarialClosedCycle` exists. The
algebraic bridge — `AGENTS.md`'s Atlas vocabulary and the project's
cycle/safe-set vocabulary are interchangeable for the purpose of certifying
infinite play through `init`. -/
theorem exists_init_AdversarialClosedCycle_iff_exists_init_closed_atlas
    {cfg : GameConfig} :
    (∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states) ↔
      (∃ (S : Finset GameState) (A : Atlas cfg),
        A.IsClosedOn cfg S ∧ GameState.init ∈ S) := by
  constructor
  · rintro ⟨C, hinit⟩
    exact ⟨C.states, C.solver.toAtlas,
           C.solver_toAtlas_isClosedOn_states, hinit⟩
  · rintro ⟨S, A, hclosed, hinit⟩
    refine ⟨AdversarialClosedCycle.ofClosedAtlas hclosed, ?_⟩
    rw [AdversarialClosedCycle.ofClosedAtlas_states]
    exact hinit

/-- **Equivalence**: M4 artifact existence ↔ init-cycle existence. The
cycle ⇒ artifact direction is `AdversarialClosedCycle.nonempty_M4Artifact_of_exists_init_cycle`;
the reverse direction extracts a cycle via `AdversarialClosedCycle.ofClosedAtlas`
(Tick 1602) from the artifact's `closed` field. This is the **headline
M4-existence theorem**: existence of an init-containing cycle is
precisely existence of an M4 artifact (the AGENTS.md final form). -/
theorem nonempty_M4Artifact_iff_exists_init_cycle (cfg : GameConfig) :
    Nonempty (AdversarialClosedCycle.M4Artifact cfg) ↔
      ∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states := by
  refine ⟨?_, AdversarialClosedCycle.nonempty_M4Artifact_of_exists_init_cycle⟩
  rintro ⟨⟨A, S, hinit, hclosed⟩⟩
  refine ⟨AdversarialClosedCycle.ofClosedAtlas hclosed, ?_⟩
  rw [AdversarialClosedCycle.ofClosedAtlas_states]
  exact hinit

/-- **Every M4 artifact is Equiv to `fromCycle` of its `ofClosedAtlas`
cycle.** Closes the loop: every M4 artifact can be realised as a
single-cycle `fromCycle` artifact (modulo Equiv). The cycle is extracted
from the atlas via `AdversarialClosedCycle.ofClosedAtlas` (Tick 1602). -/
theorem exists_cycle_fromCycle_equiv_M4Artifact {cfg : GameConfig}
    (M : AdversarialClosedCycle.M4Artifact cfg) :
    ∃ (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states),
      M.Equiv (AdversarialClosedCycle.M4Artifact.fromCycle C hinit) := by
  refine ⟨AdversarialClosedCycle.ofClosedAtlas M.closed,
          M.init_mem, rfl, ?_⟩
  intro g hg p hp
  have htot := M.closed.total g hg p hp
  obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp htot
  change M.atlas g p = some ((M.atlas g p).getD _)
  rw [hpl, Option.getD_some]

/-- **Direct round-trip Equiv** (non-existential form of Tick 1673):
for any M4 artifact `M`, `M` is M4-Equiv to the artifact built by
`fromCycle` applied to `ofClosedAtlas M.closed` with `M.init_mem`. Same
proof as `exists_cycle_fromCycle_equiv_M4Artifact` (which uses these
exact witnesses) but targets the concrete cycle/init directly,
enabling direct application of canon-level theorems without unpacking
the existential. -/
theorem AdversarialClosedCycle.M4Artifact.equiv_fromCycle_ofClosedAtlas
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    M.Equiv (AdversarialClosedCycle.M4Artifact.fromCycle
      (AdversarialClosedCycle.ofClosedAtlas M.closed) M.init_mem) := by
  refine ⟨rfl, ?_⟩
  intro g hg p hp
  have htot := M.closed.total g hg p hp
  obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp htot
  change M.atlas g p = some ((M.atlas g p).getD _)
  rw [hpl, Option.getD_some]

/-- **Canon-strict round-trip equality**: the canon atlas is a true
invariant of the M4 → cycle → M4 round-trip. Strictly stronger than
the Equiv-level Tick 1685a — the round-trip preserves the canonical
representative *exactly*. Direct application of Tick 1661's
`canon_atlas_eq_of_equiv` to Tick 1685a's Equiv. -/
theorem AdversarialClosedCycle.M4Artifact.canon_atlas_eq_fromCycle_ofClosedAtlas_canon_atlas
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    M.canon.atlas = (AdversarialClosedCycle.M4Artifact.fromCycle
      (AdversarialClosedCycle.ofClosedAtlas M.closed) M.init_mem).canon.atlas :=
  AdversarialClosedCycle.M4Artifact.canon_atlas_eq_of_equiv
    M.equiv_fromCycle_ofClosedAtlas

/-- **`M.canon ≈ fromCycleCanon (ofClosedAtlas M.closed) M.init_mem`**:
the canonical M4 artifact is Equiv to the canonical-form M4 artifact
built from M's extracted cycle. Bridges M.canon with the
`fromCycleCanon` constructor through the M4 → cycle → M4 round-trip
chain.

Proof chain (three composed equivalences):
1. `M.equiv_fromCycle_ofClosedAtlas` (Tick 1685a): `M ≈ fromCycle
   (ofClosedAtlas M.closed) M.init_mem`.
2. `canon_equiv_canon_of_equiv` (Tick 1672): lift to canon level —
   `M.canon ≈ (fromCycle ...).canon`.
3. `fromCycle_canon_equiv_fromCycleCanon` (Tick 1674): the canon
   of `fromCycle` is Equiv to `fromCycleCanon` —
   `(fromCycle ...).canon ≈ fromCycleCanon ...`.

Transitivity gives the conclusion. -/
theorem AdversarialClosedCycle.M4Artifact.canon_equiv_fromCycleCanon_ofClosedAtlas
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    M.canon.Equiv (AdversarialClosedCycle.M4Artifact.fromCycleCanon
      (AdversarialClosedCycle.ofClosedAtlas M.closed) M.init_mem) :=
  (AdversarialClosedCycle.M4Artifact.canon_equiv_canon_of_equiv
    M.equiv_fromCycle_ofClosedAtlas).trans
    (AdversarialClosedCycle.M4Artifact.fromCycle_canon_equiv_fromCycleCanon
      (AdversarialClosedCycle.ofClosedAtlas M.closed) M.init_mem)

/-- **Strict structural canon round-trip equality**:
`M.canon = fromCycleCanon (ofClosedAtlas M.closed) M.init_mem` as
**M4 artifacts** (full struct equality, not just Equiv from Tick 1711).

Both sides are canonical (Tick 1693 `canon_isCanonical` and Tick
1701 `fromCycleCanon_isCanonical`) and Equiv (Tick 1711). By Tick
1702's discrete-category theorem `eq_of_equiv_of_isCanonical_both`,
they are literally equal.

This is the **maximum-strength** characterisation: M.canon and the
canonical reconstruction via cycle extraction are bit-identical M4
artifacts. -/
theorem AdversarialClosedCycle.M4Artifact.canon_eq_fromCycleCanon_ofClosedAtlas
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    M.canon = AdversarialClosedCycle.M4Artifact.fromCycleCanon
      (AdversarialClosedCycle.ofClosedAtlas M.closed) M.init_mem :=
  AdversarialClosedCycle.M4Artifact.eq_of_equiv_of_isCanonical_both
    M.canon_equiv_fromCycleCanon_ofClosedAtlas
    (AdversarialClosedCycle.M4Artifact.canon_isCanonical M)
    (AdversarialClosedCycle.M4Artifact.fromCycleCanon_isCanonical
      (AdversarialClosedCycle.ofClosedAtlas M.closed) M.init_mem)

/-- **Canonical M4 = its cycle reconstruction**: if `M.IsCanonical`,
then `M = fromCycleCanon (ofClosedAtlas M.closed) M.init_mem` as M4
artifacts. Combines Tick 1706's `canon_eq_self_of_isCanonical`
(canonical M equals M.canon) with Tick 1712's
`canon_eq_fromCycleCanon_ofClosedAtlas` (M.canon equals the cycle
reconstruction). For canonical M4 artifacts, M IS literally its own
cycle reconstruction — no canonicalisation needed. -/
theorem AdversarialClosedCycle.M4Artifact.eq_fromCycleCanon_ofClosedAtlas_of_isCanonical
    {cfg : GameConfig} {M : AdversarialClosedCycle.M4Artifact cfg}
    (hM : M.IsCanonical) :
    M = AdversarialClosedCycle.M4Artifact.fromCycleCanon
      (AdversarialClosedCycle.ofClosedAtlas M.closed) M.init_mem :=
  (AdversarialClosedCycle.M4Artifact.canon_eq_self_of_isCanonical hM).trans
    M.canon_eq_fromCycleCanon_ofClosedAtlas

/-- **Strict-equality form of `M.canon` as `fromCycle` re-embedding
under canon**: `M.canon = (fromCycle (ofClosedAtlas M.closed)
M.init_mem).canon`. Direct via defeq: the existing
`canon_eq_fromCycleCanon_ofClosedAtlas` gives `M.canon =
fromCycleCanon ...`, and Tick 1792's
`fromCycle_canon_eq_fromCycleCanon` is `rfl`, so the two RHS forms
are definitionally interchangeable. -/
theorem AdversarialClosedCycle.M4Artifact.canon_eq_fromCycle_ofClosedAtlas_canon
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    M.canon =
      (AdversarialClosedCycle.M4Artifact.fromCycle
        (AdversarialClosedCycle.ofClosedAtlas M.closed) M.init_mem).canon :=
  M.canon_eq_fromCycleCanon_ofClosedAtlas

/-- **Top-level any-config bridge**: existence of an init-containing
adversarial closed cycle implies existence of a `SolvesTetris` solver
(for any `cfg`). -/
theorem exists_solvesTetris_of_exists_init_adversarialClosedCycle
    {cfg : GameConfig}
    (h : ∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states) :
    ∃ σ : Solver cfg, SolvesTetris cfg σ :=
  let ⟨C, hC⟩ := h
  C.exists_solvesTetris_of_init hC

/-- **The standard-config M3 statement**: existence of an init-containing
adversarial closed cycle on `GameConfig.standard` implies `TetrisSolvable`. -/
theorem tetrisSolvable_of_exists_init_adversarialClosedCycle
    (h : ∃ C : AdversarialClosedCycle GameConfig.standard,
        GameState.init ∈ C.states) :
    TetrisSolvable :=
  exists_solvesTetris_of_exists_init_adversarialClosedCycle h

/-- **The standard-config M4 statement**: existence of an M4 artifact for
`GameConfig.standard` implies `TetrisSolvable`. Composes
`nonempty_M4Artifact_iff_exists_init_cycle` (Tick 1641) with
`tetrisSolvable_of_exists_init_adversarialClosedCycle` (above). The
single-theorem M4→TetrisSolvable bridge for the standard config. -/
theorem tetrisSolvable_of_nonempty_M4Artifact_standard
    (h : Nonempty (AdversarialClosedCycle.M4Artifact GameConfig.standard)) :
    TetrisSolvable :=
  tetrisSolvable_of_exists_init_adversarialClosedCycle
    ((nonempty_M4Artifact_iff_exists_init_cycle GameConfig.standard).mp h)

/-- **Iff form**: `TetrisSolvable` is equivalent to the existence of an
init-containing adversarial closed cycle, in the
`TetrisSolvable ↔ ∃ σ, SolvesTetris cfg σ` definitional sense. Forward
direction is definitional; reverse direction is Tick 1310. -/
theorem tetrisSolvable_iff_exists_solvesTetris :
    TetrisSolvable ↔
      ∃ σ : Solver GameConfig.standard, SolvesTetris GameConfig.standard σ :=
  Iff.rfl

/-- Definitional unfolding for `TetrisSolvableFor` (any `cfg`). -/
theorem tetrisSolvableFor_iff_exists_solvesTetris (cfg : GameConfig) :
    TetrisSolvableFor cfg ↔ ∃ σ : Solver cfg, SolvesTetris cfg σ :=
  Iff.rfl

/-- **`TetrisSolvableFor` via a cycle (any `cfg`):** existence of an
init-containing adversarial closed cycle on `cfg` ⇒ `TetrisSolvableFor cfg`. -/
theorem tetrisSolvableFor_of_exists_init_adversarialClosedCycle
    {cfg : GameConfig}
    (h : ∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states) :
    TetrisSolvableFor cfg :=
  exists_solvesTetris_of_exists_init_adversarialClosedCycle h

/-- **Generic-config M4 → `TetrisSolvableFor`** bridge: for any
`cfg`, existence of an M4 artifact implies `TetrisSolvableFor cfg`.
Composes Tick 1641's `nonempty_M4Artifact_iff_exists_init_cycle` with
the generic cycle-existence bridge above. -/
theorem tetrisSolvableFor_of_nonempty_M4Artifact {cfg : GameConfig}
    (h : Nonempty (AdversarialClosedCycle.M4Artifact cfg)) :
    TetrisSolvableFor cfg :=
  tetrisSolvableFor_of_exists_init_adversarialClosedCycle
    ((nonempty_M4Artifact_iff_exists_init_cycle cfg).mp h)

/-- **Tiny-config specialisation**: existence of an init-containing
adversarial closed cycle on `GameConfig.tiny` ⇒ `TetrisSolvableFor tiny`. -/
theorem tetrisSolvableFor_tiny_of_exists_init_adversarialClosedCycle
    (h : ∃ C : AdversarialClosedCycle GameConfig.tiny,
        GameState.init ∈ C.states) :
    TetrisSolvableFor GameConfig.tiny :=
  tetrisSolvableFor_of_exists_init_adversarialClosedCycle h

/-- **Tiny-config M4 → `TetrisSolvableFor tiny`** bridge: existence of an
M4 artifact on `GameConfig.tiny` implies `TetrisSolvableFor tiny`.
Composes Tick 1641's `nonempty_M4Artifact_iff_exists_init_cycle` with
the cycle-existence specialisation above. -/
theorem tetrisSolvableFor_tiny_of_nonempty_M4Artifact
    (h : Nonempty (AdversarialClosedCycle.M4Artifact GameConfig.tiny)) :
    TetrisSolvableFor GameConfig.tiny :=
  tetrisSolvableFor_tiny_of_exists_init_adversarialClosedCycle
    ((nonempty_M4Artifact_iff_exists_init_cycle GameConfig.tiny).mp h)

/-- **Standard-config M4 → TetrisSolvable (direct)**: from a standard-config
M4 artifact, directly extract a `TetrisSolvable` witness via atlas
materialisation. The **AGENTS.md headline statement** in direct form. -/
theorem tetrisSolvable_of_M4Artifact_standard
    (M : AdversarialClosedCycle.M4Artifact GameConfig.standard) :
    TetrisSolvable :=
  AdversarialClosedCycle.tetrisSolvableFor_of_M4Artifact M

/-- **Tiny-config M4 → `TetrisSolvableFor tiny` (direct)**: tiny-config
specialisation of Tick 1653 — direct extraction without going through
the cycle bridge. -/
theorem tetrisSolvableFor_tiny_of_M4Artifact
    (M : AdversarialClosedCycle.M4Artifact GameConfig.tiny) :
    TetrisSolvableFor GameConfig.tiny :=
  AdversarialClosedCycle.tetrisSolvableFor_of_M4Artifact M

/-- **Contrapositive of Tick 1313**: if `cfg` is not solvable, then no
adversarial closed cycle on `cfg` contains `init`. -/
theorem forall_init_not_mem_states_of_not_tetrisSolvableFor
    {cfg : GameConfig} (h : ¬ TetrisSolvableFor cfg)
    (C : AdversarialClosedCycle cfg) :
    GameState.init ∉ C.states :=
  fun hmem => h (C.exists_solvesTetris_of_init hmem)

/-- **Negation form of the cycle bridge**: if `cfg` is not solvable,
no init-containing adversarial closed cycle exists. -/
theorem not_exists_init_adversarialClosedCycle_of_not_tetrisSolvableFor
    {cfg : GameConfig} (h : ¬ TetrisSolvableFor cfg) :
    ¬ ∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states :=
  fun ⟨C, hC⟩ => forall_init_not_mem_states_of_not_tetrisSolvableFor h C hC

/-- **Standard-config contrapositive**: if Tetris is unsolvable, no
init-containing `AdversarialClosedCycle GameConfig.standard` exists. -/
theorem not_exists_init_adversarialClosedCycle_of_not_tetrisSolvable
    (h : ¬ TetrisSolvable) :
    ¬ ∃ C : AdversarialClosedCycle GameConfig.standard,
        GameState.init ∈ C.states :=
  not_exists_init_adversarialClosedCycle_of_not_tetrisSolvableFor h

namespace AdversarialClosedCycle

/-- Init-step is `solverReachable` (because `solverReachable σ` is closed
under adversarial steps of `init`, which has every piece available). -/
theorem step_solverReachable_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (p : Piece) :
    solverReachable C.solver
      (adversarialStep cfg GameState.init p (C.solver GameState.init p)) :=
  solverReachable.step p (solverReachable.init)
    (GameState.init_bag.symm ▸ Bag.mem_full p)

/-- **Adversarial trace from `init` is `solverReachable` under the
cycle solver.** Independent of the cycle structure — purely
`solverReachable` closure under `adversarialStep`. -/
theorem adversarialTrace_solverReachable_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    solverReachable C.solver
      (adversarialTrace cfg C.solver s GameState.init n) := by
  induction n with
  | zero => exact solverReachable.init
  | succ k ih =>
    rw [adversarialTrace_succ]
    refine solverReachable.step (s k) ih ?_
    have := hl.canDraw k
    rw [Bag.canDraw_iff_mem, ← adversarialTrace_bag] at this
    exact this

/-- **Cycle's `solverReachable` set stays in cycle states**: for
any init-containing cycle, every `solverReachable C.solver g`
lies in `C.states`. Direct induction on `solverReachable` using
`C.closed` at the step. Cycle-level parallel to Tick 1895
(Atlas-level). -/
theorem solverReachable_mem_states {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states)
    {g : GameState} (hg : solverReachable C.solver g) :
    g ∈ C.states := by
  induction hg with
  | init => exact hinit
  | step p _ hp ih => exact C.closed _ ih _ hp

/-- **Cycle's `solverReachable` set is never lost**: every
`solverReachable C.solver g` is non-lost. Direct via Tick 1903 +
`C.not_lost`. Cycle-level parallel to Tick 1898. -/
theorem solverReachable_not_lost {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states)
    {g : GameState} (hg : solverReachable C.solver g) :
    ¬ g.lost cfg :=
  C.not_lost _ (C.solverReachable_mem_states hinit hg)


end AdversarialClosedCycle

namespace AdversarialClosedCycle

/-- A cycle containing `init` has nonempty states. -/
theorem init_states_nonempty {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    C.states.Nonempty :=
  ⟨GameState.init, h⟩

/-- A cycle containing `init` has `states.card ≥ 1`. -/
theorem init_states_card_pos {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    0 < C.states.card :=
  Finset.card_pos.mpr (C.init_states_nonempty h)

/-- A cycle containing `init` has `states ≠ ∅`. -/
theorem init_states_ne_empty {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    C.states ≠ ∅ :=
  (C.init_states_nonempty h).ne_empty

/-- A cycle containing `init` has `states.card ≥ 1`. -/
theorem init_states_card_ge_one {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    1 ≤ C.states.card :=
  C.init_states_card_pos h

/-- A cycle containing `init` has `states.card ≠ 0`. -/
theorem init_states_card_ne_zero {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    C.states.card ≠ 0 :=
  Nat.ne_of_gt (C.init_states_card_pos h)


/-- Valid-half of `valid`. -/
theorem solver_placement_valid {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg)
    {g : GameState} (hg : g ∈ C.states) {p : Piece} (hp : p ∈ g.bag) :
    (C.solver g p).Valid cfg :=
  (C.valid g hg p hp).2

/-- The solver's choice always lies in `Placement.allValidFor`. -/
theorem solver_mem_allValidFor {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg)
    {g : GameState} (hg : g ∈ C.states) {p : Piece} (hp : p ∈ g.bag) :
    C.solver g p ∈ Placement.allValidFor cfg p :=
  (Placement.mem_allValidFor cfg p _).mpr (C.valid g hg p hp)

/-- `GameState.step` form of `closed`. -/
theorem step_mem_step {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {g : GameState} (hg : g ∈ C.states) {p : Piece} (hp : p ∈ g.bag) :
    g.step cfg (C.solver g p) ∈ C.states := by
  have hsm := C.closed g hg p hp
  rw [adversarialStep_eq_step, Placement.eta_of_piece_eq
        (C.solver_piece hg hp)] at hsm
  exact hsm

/-- The cycle's chosen successor at `(s, p)`: apply the solver's placement. -/
def next {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (s : GameState) (p : Piece) : GameState :=
  adversarialStep cfg s p (C.solver s p)

/-- The successor lies in the cycle whenever the source does and the piece is
in the bag (re-export of `closed` via `next`). -/
theorem next_mem {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {s : GameState} (hs : s ∈ C.states) {p : Piece} (hp : p ∈ s.bag) :
    C.next s p ∈ C.states := C.closed s hs p hp

/-- `next` agrees with `GameState.step` whenever the solver's piece is `p`. -/
theorem next_eq_step {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {s : GameState} (hs : s ∈ C.states) {p : Piece} (hp : p ∈ s.bag) :
    C.next s p = s.step cfg (C.solver s p) := by
  unfold next
  rw [adversarialStep_eq_step,
      Placement.eta_of_piece_eq (C.solver_piece hs hp)]

/-- The successor under `next` is not lost. -/
theorem next_not_lost {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {s : GameState} (hs : s ∈ C.states) {p : Piece} (hp : p ∈ s.bag) :
    ¬ (C.next s p).lost cfg :=
  C.not_lost _ (C.next_mem hs hp)

/-- Bag of the successor: drops the played piece. -/
@[simp] theorem next_bag {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (s : GameState) (p : Piece) :
    (C.next s p).bag = s.bag.draw p := rfl

/-- `C.next s` is injective on `s.bag` (the bag-projection is `Bag.draw p`,
injective on `s.bag` by `Bag.draw_injOn`). -/
theorem next_injOn {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (s : GameState) : Set.InjOn (C.next s) ↑s.bag := by
  intro p hp q hq hpq
  have hbag : (C.next s p).bag = (C.next s q).bag := congrArg GameState.bag hpq
  rw [C.next_bag, C.next_bag] at hbag
  exact Bag.draw_injOn s.bag hp hq hbag

/-- The *layer* of one-step successors of `s` under the cycle's solver. -/
def layer {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (s : GameState) : Finset GameState :=
  s.bag.image (C.next s)

/-- The layer sits inside the cycle whenever the source does. -/
theorem layer_subset {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {s : GameState} (hs : s ∈ C.states) :
    C.layer s ⊆ C.states := by
  intro x hx
  rcases Finset.mem_image.mp hx with ⟨p, hp, rfl⟩
  exact C.next_mem hs hp

/-- Cardinality of the layer matches the source's bag. -/
theorem layer_card {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (s : GameState) : (C.layer s).card = s.bag.card :=
  Finset.card_image_of_injOn (C.next_injOn s)

/-- Layer members carry the post-draw bag. -/
theorem mem_layer_bag {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {s x : GameState} (hx : x ∈ C.layer s) :
    ∃ p ∈ s.bag, x.bag = s.bag.draw p := by
  rcases Finset.mem_image.mp hx with ⟨p, hp, rfl⟩
  exact ⟨p, hp, by simp⟩

/-- **Layer bag-card invariant.** When the source bag has at least 2 pieces,
every member of the layer has bag.card exactly `s.bag.card - 1` (uniform
because the non-emptying branch fires for every choice of `p`). -/
theorem layer_bag_card {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {s : GameState} (h : 2 ≤ s.bag.card) {x : GameState} (hx : x ∈ C.layer s) :
    x.bag.card = s.bag.card - 1 := by
  obtain ⟨p, hp, hbag⟩ := C.mem_layer_bag hx
  rw [hbag]
  exact Bag.card_draw_of_ge_two hp h

/-- **Every state of an adversarial closed cycle is safe.** This is the
working tool: to prove `init ∈ safe`, construct a cycle whose `states`
contains `init`. -/
theorem subset_safe {cfg : GameConfig} (C : AdversarialClosedCycle cfg) :
    (↑C.states : Set GameState) ⊆ safe cfg := by
  apply safe_greatest
  intro g hg
  rw [Finset.mem_coe] at hg
  refine ⟨C.not_lost g hg, ?_⟩
  intro p hp
  have hv := C.valid g hg p hp
  refine ⟨C.solver g p, hv.1, hv.2, ?_⟩
  rw [Finset.mem_coe]
  exact C.closed g hg p hp

/-- Pointwise: every state in an adversarial closed cycle is safe. -/
theorem mem_safe {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {g : GameState} (h : g ∈ C.states) : g ∈ safe cfg :=
  C.subset_safe (Finset.mem_coe.mpr h)

/-- If an adversarial closed cycle contains `init`, then `init` is
safe. The headline application of `mem_safe`. -/
theorem init_safe_of_init_mem {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    GameState.init ∈ safe cfg :=
  C.mem_safe h

/-- **Existence of a safe state in any nonempty cycle.** From `mem_safe`
+ a nonemptiness witness. -/
theorem exists_mem_safe {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (hne : C.states.Nonempty) :
    ∃ g, g ∈ safe cfg := by
  obtain ⟨g, hg⟩ := hne
  exact ⟨g, C.mem_safe hg⟩

/-- **Cycle's `solverReachable` set is safe**: every
`solverReachable C.solver g` lies in `safe cfg`. Direct via
Tick 1903 + `mem_safe`. Cycle-level parallel to Tick 1897. -/
theorem solverReachable_mem_safe {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states)
    {g : GameState} (hg : solverReachable C.solver g) :
    g ∈ safe cfg :=
  C.mem_safe (C.solverReachable_mem_states hinit hg)

/-- **Cycle's `solverReachable` set is finite**: bounded by the
finite cycle state set `C.states`. Direct via Tick 1903 +
`C.states.finite_toSet.subset`. Cycle-level parallel to Tick 1896. -/
theorem solverReachable_finite {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    {g | solverReachable C.solver g}.Finite :=
  C.states.finite_toSet.subset
    (fun _ hg => C.solverReachable_mem_states hinit hg)

/-- **Cycle's `solverReachable` set cardinality bound**: the ncard
of `{g | solverReachable C.solver g}` is bounded by `C.states.card`.
Direct via `Set.ncard_le_ncard` + Tick 1903's subset fact, with
`Set.ncard_coe_finset` to convert `↑C.states`'s ncard to
`C.states.card`. Pattern mirrors
`solverReachable_ncard_le_inFieldStates_card` (SafeIterateFinite). -/
theorem solverReachable_ncard_le_states_card {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    {g | solverReachable C.solver g}.ncard ≤ C.states.card := by
  rw [← Set.ncard_coe_finset]
  exact Set.ncard_le_ncard
    (fun _ hg => C.solverReachable_mem_states hinit hg)
    C.states.finite_toSet

/-- **Cycle's `solverReachable` set Finset cardinality bound**:
the Finset cardinality of `(solverReachable_finite).toFinset` is
bounded by `C.states.card`. Finset-form companion to Tick 1947's
ncard version. Useful when downstream proofs need a Finset (not
just a Set) cardinality bound. -/
theorem solverReachable_toFinset_card_le_states_card {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    (C.solverReachable_finite hinit).toFinset.card ≤ C.states.card := by
  apply Finset.card_le_card
  intro g hg
  rw [Set.Finite.mem_toFinset] at hg
  exact C.solverReachable_mem_states hinit hg

/-- **Cycle's `solverReachable` Set ⊆ `↑C.states`**: Set-form
of Tick 1903 (`solverReachable_mem_states`). Useful when Set-level
subset reasoning is needed (e.g., for `Set.ncard_le_ncard`,
`Set.image_subset`, or other Set lemmas). -/
theorem solverReachable_subset_states {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    {g | solverReachable C.solver g} ⊆ ↑C.states :=
  fun _ hg => Finset.mem_coe.mpr (C.solverReachable_mem_states hinit hg)

/-- **Cycle's `solverReachable` Set ⊆ `safe cfg`**: Set-form of
Tick 1905 (`solverReachable_mem_safe`). Distinct from the top-level
pointwise `solverReachable_subset_safe` (~line 450, despite the
name) — this is the proper Set-form. Useful for Set algebra chains. -/
theorem solverReachable_set_subset_safe {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    {g | solverReachable C.solver g} ⊆ safe cfg :=
  fun _ hg => C.solverReachable_mem_safe hinit hg

/-- **Bundled per-element cycle `solverReachable` invariants**:
3-conjunct bundle of mem-states (Tick 1903), not-lost (Tick 1904),
and mem-safe (Tick 1905). Per-element parallel to
`adversarialTrace_invariants_from_mem`. -/
theorem solverReachable_invariants {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states)
    {g : GameState} (hg : solverReachable C.solver g) :
    g ∈ C.states ∧ ¬ g.lost cfg ∧ g ∈ safe cfg :=
  ⟨C.solverReachable_mem_states hinit hg,
   C.solverReachable_not_lost hinit hg,
   C.solverReachable_mem_safe hinit hg⟩

/-- **4-conjunct cycle `solverReachable` invariants**: extends
`solverReachable_invariants` with `g.bag.Nonempty` (free for any
`solverReachable`-state via `solverReachable_bag_nonempty`).
Useful for downstream proofs that need to draw a piece. -/
theorem solverReachable_invariants_with_bag {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states)
    {g : GameState} (hg : solverReachable C.solver g) :
    g ∈ C.states ∧ ¬ g.lost cfg ∧ g ∈ safe cfg ∧ g.bag.Nonempty :=
  ⟨C.solverReachable_mem_states hinit hg,
   C.solverReachable_not_lost hinit hg,
   C.solverReachable_mem_safe hinit hg,
   solverReachable_bag_nonempty hg⟩

/-- **5-conjunct cycle `solverReachable` invariants with solver
validity**: extends `solverReachable_invariants_with_bag` with the
universally-quantified `C.valid` fact for every piece in the bag:
`∀ p ∈ g.bag, (C.solver g p).piece = p ∧ (C.solver g p).Valid cfg`.
This is the complete "step-ready" bundle — given solverReachable,
you can pick any piece, get a valid placement, know it plays that
piece, and the result is non-losing & safe & remains in the cycle. -/
theorem solverReachable_invariants_full {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states)
    {g : GameState} (hg : solverReachable C.solver g) :
    g ∈ C.states ∧ ¬ g.lost cfg ∧ g ∈ safe cfg ∧ g.bag.Nonempty ∧
      ∀ p ∈ g.bag, (C.solver g p).piece = p ∧ (C.solver g p).Valid cfg :=
  ⟨C.solverReachable_mem_states hinit hg,
   C.solverReachable_not_lost hinit hg,
   C.solverReachable_mem_safe hinit hg,
   solverReachable_bag_nonempty hg,
   fun p hp => C.valid g (C.solverReachable_mem_states hinit hg) p hp⟩

/-- **Cycle step-closure of the full invariant bundle**: from a
`solverReachable` state `g` and any piece `p ∈ g.bag`, the
next state `adversarialStep cfg g p (C.solver g p)` also satisfies
the full 5-conjunct invariant bundle. Proof composes the
inductive `step` constructor of `solverReachable` with Tick 1909. -/
theorem solverReachable_step_invariants_full {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states)
    {g : GameState} (hg : solverReachable C.solver g)
    {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (C.solver g p) ∈ C.states ∧
    ¬ (adversarialStep cfg g p (C.solver g p)).lost cfg ∧
    adversarialStep cfg g p (C.solver g p) ∈ safe cfg ∧
    (adversarialStep cfg g p (C.solver g p)).bag.Nonempty ∧
    ∀ q ∈ (adversarialStep cfg g p (C.solver g p)).bag,
      (C.solver (adversarialStep cfg g p (C.solver g p)) q).piece = q ∧
      (C.solver (adversarialStep cfg g p (C.solver g p)) q).Valid cfg :=
  C.solverReachable_invariants_full hinit (solverReachable.step p hg hp)

/-- **Cycle's adversarial trace from a `solverReachable` start
retains the full 5-conjunct invariant bundle at every step**.
Composes `adversarialTrace_solverReachable_from` (Tick 1911) with
`solverReachable_invariants_full` (Tick 1909). The iterated form
of `solverReachable_step_invariants_full` (Tick 1910). -/
theorem adversarialTrace_full_invariants_from_solverReachable
    {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states)
    {g₀ : GameState} (hg₀ : solverReachable C.solver g₀)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    adversarialTrace cfg C.solver t g₀ n ∈ C.states ∧
    ¬ (adversarialTrace cfg C.solver t g₀ n).lost cfg ∧
    adversarialTrace cfg C.solver t g₀ n ∈ safe cfg ∧
    (adversarialTrace cfg C.solver t g₀ n).bag.Nonempty ∧
    ∀ p ∈ (adversarialTrace cfg C.solver t g₀ n).bag,
      (C.solver (adversarialTrace cfg C.solver t g₀ n) p).piece = p ∧
      (C.solver (adversarialTrace cfg C.solver t g₀ n) p).Valid cfg :=
  C.solverReachable_invariants_full hinit
    (adversarialTrace_solverReachable_from hg₀ hl n)

/-- **Init specialisation of `adversarialTrace_full_invariants_from_solverReachable`**
(universal-piece form): start at `GameState.init` (solverReachable
trivially) with a full `LegalSequence`. Distinct from the existing
`adversarialTrace_full_invariants_of_init` (line ~4459) which is the
*sequenced* form (piece = `s n`); this is the *universal* form
(`∀ p ∈ bag`), which is the natural shape for `solverReachable`-driven
proofs that don't fix a specific piece. Eliminates the
`solverReachable g₀` and `LegalSequenceFrom g₀.bag t` hypotheses
(the former is `solverReachable.init`; the latter follows
definitionally from `init.bag = Bag.full`). -/
theorem adversarialTrace_full_invariants_univ_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states)
    {t : ℕ → Piece} (hl : LegalSequence t) (n : ℕ) :
    adversarialTrace cfg C.solver t GameState.init n ∈ C.states ∧
    ¬ (adversarialTrace cfg C.solver t GameState.init n).lost cfg ∧
    adversarialTrace cfg C.solver t GameState.init n ∈ safe cfg ∧
    (adversarialTrace cfg C.solver t GameState.init n).bag.Nonempty ∧
    ∀ p ∈ (adversarialTrace cfg C.solver t GameState.init n).bag,
      (C.solver (adversarialTrace cfg C.solver t GameState.init n) p).piece = p ∧
      (C.solver (adversarialTrace cfg C.solver t GameState.init n) p).Valid cfg :=
  C.adversarialTrace_full_invariants_from_solverReachable hinit
    solverReachable.init hl n

/-- **Init-specialised universal full bundle + `Reachable`**:
6-conjunct form combining Tick 1913's universal 5-conjunct bundle
with `Reachable cfg trace` from `adversarialTrace_reachable_of_init`
(line ~723). Stronger than either alone — sits between the existing
sequenced `_of_init` (piece-specific facts + `Reachable`) and the
universal `_univ_of_init` (universal-piece facts, no `Reachable`). -/
theorem adversarialTrace_full_invariants_univ_with_reachable_of_init
    {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states)
    {t : ℕ → Piece} (hl : LegalSequence t) (n : ℕ) :
    adversarialTrace cfg C.solver t GameState.init n ∈ C.states ∧
    ¬ (adversarialTrace cfg C.solver t GameState.init n).lost cfg ∧
    adversarialTrace cfg C.solver t GameState.init n ∈ safe cfg ∧
    (adversarialTrace cfg C.solver t GameState.init n).bag.Nonempty ∧
    (∀ p ∈ (adversarialTrace cfg C.solver t GameState.init n).bag,
      (C.solver (adversarialTrace cfg C.solver t GameState.init n) p).piece = p ∧
      (C.solver (adversarialTrace cfg C.solver t GameState.init n) p).Valid cfg) ∧
    Reachable cfg (adversarialTrace cfg C.solver t GameState.init n) :=
  let h5 := C.adversarialTrace_full_invariants_univ_of_init hinit hl n
  ⟨h5.1, h5.2.1, h5.2.2.1, h5.2.2.2.1, h5.2.2.2.2,
   C.adversarialTrace_reachable_of_init hinit hl n⟩

/-- **Cycle-side `solverReachable` ⇒ `Reachable`**: for any
init-containing cycle, every `solverReachable C.solver g` lifts to
`Reachable cfg g`. Induction on `solverReachable` using `C.valid`
at each step (mirrors the proof shape of
`adversarialTrace_reachable_of_init`, line ~723). -/
theorem solverReachable_reachable {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states)
    {g : GameState} (hg : solverReachable C.solver g) :
    Reachable cfg g := by
  induction hg with
  | init => exact Reachable.init
  | @step g' p hsr hp ih =>
    have hmem : g' ∈ C.states := C.solverReachable_mem_states hinit hsr
    have hnl : ¬ g'.lost cfg := C.not_lost _ hmem
    obtain ⟨hpiece, hval⟩ := C.solver_valid_and_piece hmem hp
    have heta : ({ C.solver g' p with piece := p } : Placement)
        = C.solver g' p := by
      rcases hpl : C.solver g' p with ⟨pc, r, c⟩
      rw [hpl] at hpiece
      cases hpiece
      rfl
    refine ih.adversarialStep hnl hp ?_
    rw [heta]
    exact hval

/-- **Cycle's `solverReachable` Set ⊆ `{g | Reachable cfg g}`**:
Set-form of Tick 1915 (`solverReachable_reachable`). Bridges the
cycle-driven reachability to the broader `Reachable cfg` predicate. -/
theorem solverReachable_subset_Reachable {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    {g | solverReachable C.solver g} ⊆ {g | Reachable cfg g} :=
  fun _ hg => C.solverReachable_reachable hinit hg

/-- **Cycle's `solverReachable` Set ⊆ joint intersection
`states ∩ safe cfg ∩ {g | Reachable cfg g}`**: combines Ticks 1951,
1952, 1953 into a single intersection inclusion. The ultimate Set-form
fact for cycle-driven reachable states. -/
theorem solverReachable_subset_inter {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states) :
    {g | solverReachable C.solver g} ⊆
      ↑C.states ∩ safe cfg ∩ {g | Reachable cfg g} := by
  intro g hg
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · exact C.solverReachable_subset_states hinit hg
  · exact C.solverReachable_set_subset_safe hinit hg
  · exact C.solverReachable_subset_Reachable hinit hg

/-- **6-conjunct cycle `solverReachable` invariants with `Reachable`**:
extends `solverReachable_invariants_full` (Tick 1909) with `Reachable
cfg g` (Tick 1915). The maximal *per-element* cycle invariant bundle
— includes everything from per-state safety + draw-readiness + per-piece
validity + Reachable. Useful for proofs that need to convert from
cycle/solverReachable reasoning to the broader `Reachable` machinery. -/
theorem solverReachable_invariants_full_with_reachable {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states)
    {g : GameState} (hg : solverReachable C.solver g) :
    g ∈ C.states ∧ ¬ g.lost cfg ∧ g ∈ safe cfg ∧ g.bag.Nonempty ∧
      (∀ p ∈ g.bag, (C.solver g p).piece = p ∧ (C.solver g p).Valid cfg) ∧
      Reachable cfg g :=
  let h5 := C.solverReachable_invariants_full hinit hg
  ⟨h5.1, h5.2.1, h5.2.2.1, h5.2.2.2.1, h5.2.2.2.2,
   C.solverReachable_reachable hinit hg⟩

/-- **Cycle trace from `solverReachable` start retains the
6-conjunct ultimate bundle**: lifts Tick 1916's per-element
`solverReachable_invariants_full_with_reachable` to traces from any
solverReachable start. Equivalent to Tick 1914 (the init-specialised
form) but for arbitrary solverReachable `g₀`. Proof: apply Tick 1916
to the trace's solverReachability (Tick 1911). -/
theorem adversarialTrace_full_invariants_with_reachable_from_solverReachable
    {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hinit : GameState.init ∈ C.states)
    {g₀ : GameState} (hg₀ : solverReachable C.solver g₀)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    adversarialTrace cfg C.solver t g₀ n ∈ C.states ∧
    ¬ (adversarialTrace cfg C.solver t g₀ n).lost cfg ∧
    adversarialTrace cfg C.solver t g₀ n ∈ safe cfg ∧
    (adversarialTrace cfg C.solver t g₀ n).bag.Nonempty ∧
    (∀ p ∈ (adversarialTrace cfg C.solver t g₀ n).bag,
      (C.solver (adversarialTrace cfg C.solver t g₀ n) p).piece = p ∧
      (C.solver (adversarialTrace cfg C.solver t g₀ n) p).Valid cfg) ∧
    Reachable cfg (adversarialTrace cfg C.solver t g₀ n) :=
  C.solverReachable_invariants_full_with_reachable hinit
    (adversarialTrace_solverReachable_from hg₀ hl n)

/-- **Cycle's trace from any state is safe**: parallel to the
existing `_of_init` version of safe-trace, generalizing the
init-case. Composes `adversarialTrace_mem_states_from_mem` with
`mem_safe`. -/
theorem adversarialTrace_mem_safe_from_mem {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g₀ : GameState}
    (hg₀ : g₀ ∈ C.states) {t : ℕ → Piece}
    (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    adversarialTrace cfg C.solver t g₀ n ∈ safe cfg :=
  C.mem_safe (C.adversarialTrace_mem_states_from_mem hg₀ hl n)

/-- **Bundled cycle trace invariants from any state**: 3-conjunct
bundle of mem-states, not-lost, and mem-safe. Cycle-side parallel
to M4 bundle theorems. -/
theorem adversarialTrace_invariants_from_mem {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g₀ : GameState}
    (hg₀ : g₀ ∈ C.states) {t : ℕ → Piece}
    (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    adversarialTrace cfg C.solver t g₀ n ∈ C.states ∧
    ¬ (adversarialTrace cfg C.solver t g₀ n).lost cfg ∧
    adversarialTrace cfg C.solver t g₀ n ∈ safe cfg :=
  ⟨C.adversarialTrace_mem_states_from_mem hg₀ hl n,
   C.adversarialTrace_not_lost_from_mem hg₀ hl n,
   C.adversarialTrace_mem_safe_from_mem hg₀ hl n⟩

/-- **Per-step solver placement is valid and plays `t n`, from any
cycle state.** Trace-level parallel to `solver_valid_and_piece`,
generalized from init to any `g₀ ∈ C.states`. Combines
`adversarialTrace_mem_states_from_mem` (the trace state is in the
cycle) with `valid` field (the placement is valid + correct piece
on cycle-state bag). The bag-membership of `t n` is via
`adversarialTrace_bag_from`. -/
theorem adversarialTrace_solver_valid_and_piece_from_mem
    {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g₀ : GameState}
    (hg₀ : g₀ ∈ C.states) {t : ℕ → Piece}
    (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    (C.solver (adversarialTrace cfg C.solver t g₀ n) (t n)).piece = t n ∧
    (C.solver (adversarialTrace cfg C.solver t g₀ n) (t n)).Valid cfg := by
  have hg := C.adversarialTrace_mem_states_from_mem hg₀ hl n
  have hbag : t n ∈ (adversarialTrace cfg C.solver t g₀ n).bag := by
    rw [adversarialTrace_bag_from]
    exact hl n
  exact C.valid _ hg (t n) hbag

/-- **Full 5-conjunct cycle trace invariants from any state**:
combines the 3-conjunct trace invariants (mem-states, not-lost,
mem-safe) with the 2-conjunct per-step solver facts (piece, valid).
Drops the `reachable` conjunct from the init version since
reachability requires the additional hypothesis that `g₀` is
reachable from `init`. Cycle-side from-mem counterpart to
`adversarialTrace_full_invariants_of_init` (sans `Reachable`). -/
theorem adversarialTrace_full_invariants_from_mem {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g₀ : GameState}
    (hg₀ : g₀ ∈ C.states) {t : ℕ → Piece}
    (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    adversarialTrace cfg C.solver t g₀ n ∈ C.states ∧
    ¬ (adversarialTrace cfg C.solver t g₀ n).lost cfg ∧
    adversarialTrace cfg C.solver t g₀ n ∈ safe cfg ∧
    (C.solver (adversarialTrace cfg C.solver t g₀ n) (t n)).piece = t n ∧
    (C.solver (adversarialTrace cfg C.solver t g₀ n) (t n)).Valid cfg := by
  have h3 := C.adversarialTrace_invariants_from_mem hg₀ hl n
  have h2 := C.adversarialTrace_solver_valid_and_piece_from_mem hg₀ hl n
  exact ⟨h3.1, h3.2.1, h3.2.2, h2.1, h2.2⟩

/-- **`Set.range` form for `C.states`, from any cycle state**: the
full image of the adversarial trace from `g₀ ∈ C.states` along a
legal sequence is contained in the cycle's state set. From-mem
parallel to `adversarialTrace_range_subset_states_of_init`. -/
theorem adversarialTrace_range_subset_states_from_mem
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {g₀ : GameState} (hg₀ : g₀ ∈ C.states)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    Set.range (adversarialTrace cfg C.solver t g₀) ⊆ C.states :=
  Set.range_subset_iff.mpr (C.adversarialTrace_mem_states_from_mem hg₀ hl)

/-- **`Set.range` form for `safe cfg`, from any cycle state**: the
full image of the adversarial trace from `g₀ ∈ C.states` along a
legal sequence is contained in `safe cfg`. From-mem parallel to
`adversarialTrace_range_subset_safe_of_init`. -/
theorem adversarialTrace_range_subset_safe_from_mem
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {g₀ : GameState} (hg₀ : g₀ ∈ C.states)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    Set.range (adversarialTrace cfg C.solver t g₀) ⊆ safe cfg :=
  Set.range_subset_iff.mpr (C.adversarialTrace_mem_safe_from_mem hg₀ hl)

/-- **Adversarial trace range is finite, from any cycle state**:
the trace image is finite (contained in the Finset `C.states`).
From-mem parallel to `adversarialTrace_range_finite_of_init`. -/
theorem adversarialTrace_range_finite_from_mem {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g₀ : GameState}
    (hg₀ : g₀ ∈ C.states) {t : ℕ → Piece}
    (hl : LegalSequenceFrom g₀.bag t) :
    (Set.range (adversarialTrace cfg C.solver t g₀)).Finite :=
  (C.states.finite_toSet).subset
    (C.adversarialTrace_range_subset_states_from_mem hg₀ hl)

/-- **Pigeonhole on `adversarialTrace` from any cycle state**:
indices in `0..C.states.card` cannot be all distinct under the
trace map (since the trace lives in `C.states`), so two indices
share a trace value. From-mem parallel to
`adversarialTrace_pigeonhole_of_init`. -/
theorem adversarialTrace_pigeonhole_from_mem {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g₀ : GameState}
    (hg₀ : g₀ ∈ C.states) {t : ℕ → Piece}
    (hl : LegalSequenceFrom g₀.bag t) :
    ∃ i j : ℕ, i ≤ C.states.card ∧ j ≤ C.states.card ∧ i ≠ j ∧
      adversarialTrace cfg C.solver t g₀ i =
        adversarialTrace cfg C.solver t g₀ j := by
  set N := C.states.card
  have hmap : ∀ k ∈ Finset.range (N + 1),
      adversarialTrace cfg C.solver t g₀ k ∈ C.states :=
    fun k _ => C.adversarialTrace_mem_states_from_mem hg₀ hl k
  have hcard : N < (Finset.range (N + 1)).card := by
    rw [Finset.card_range]; omega
  obtain ⟨i, hi, j, hj, hne, heq⟩ :=
    Finset.exists_ne_map_eq_of_card_lt_of_maps_to hcard hmap
  rw [Finset.mem_range] at hi hj
  exact ⟨i, j, by omega, by omega, hne, heq⟩

/-- **Ordered pigeonhole on `adversarialTrace`, from any cycle
state**: same as Tick 1782 but with `i < j`. From-mem parallel to
`adversarialTrace_pigeonhole_lt_of_init`. -/
theorem adversarialTrace_pigeonhole_lt_from_mem {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g₀ : GameState}
    (hg₀ : g₀ ∈ C.states) {t : ℕ → Piece}
    (hl : LegalSequenceFrom g₀.bag t) :
    ∃ i j : ℕ, i < j ∧ j ≤ C.states.card ∧
      adversarialTrace cfg C.solver t g₀ i =
        adversarialTrace cfg C.solver t g₀ j := by
  obtain ⟨i, j, hi, hj, hne, heq⟩ := C.adversarialTrace_pigeonhole_from_mem hg₀ hl
  rcases lt_or_gt_of_ne hne with hlt | hgt
  · exact ⟨i, j, hlt, hj, heq⟩
  · exact ⟨j, i, hgt, hi, heq.symm⟩

/-- **Ordered pigeonhole with state membership, from any cycle
state**: same as Tick 1783 but the common trace state lies in
`C.states`. From-mem parallel to
`adversarialTrace_pigeonhole_lt_in_states_of_init`. -/
theorem adversarialTrace_pigeonhole_lt_in_states_from_mem {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g₀ : GameState}
    (hg₀ : g₀ ∈ C.states) {t : ℕ → Piece}
    (hl : LegalSequenceFrom g₀.bag t) :
    ∃ i j : ℕ, i < j ∧ j ≤ C.states.card ∧
      adversarialTrace cfg C.solver t g₀ i =
        adversarialTrace cfg C.solver t g₀ j ∧
      adversarialTrace cfg C.solver t g₀ i ∈ C.states := by
  obtain ⟨i, j, hij, hj, heq⟩ := C.adversarialTrace_pigeonhole_lt_from_mem hg₀ hl
  exact ⟨i, j, hij, hj, heq, C.adversarialTrace_mem_states_from_mem hg₀ hl i⟩

/-- **Base + period form of pigeonhole, from any cycle state**:
from `i < j` with shared trace value, repackage as
`∃ base, period ≥ 1, trace base = trace (base + period)`.
The period is at most `C.states.card`. From-mem parallel to
`adversarialTrace_pigeonhole_base_period_of_init`. -/
theorem adversarialTrace_pigeonhole_base_period_from_mem {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g₀ : GameState}
    (hg₀ : g₀ ∈ C.states) {t : ℕ → Piece}
    (hl : LegalSequenceFrom g₀.bag t) :
    ∃ base period : ℕ, 1 ≤ period ∧ period ≤ C.states.card ∧
      adversarialTrace cfg C.solver t g₀ base =
        adversarialTrace cfg C.solver t g₀ (base + period) := by
  obtain ⟨i, j, hij, hj, heq⟩ := C.adversarialTrace_pigeonhole_lt_from_mem hg₀ hl
  refine ⟨i, j - i, ?_, ?_, ?_⟩
  · omega
  · omega
  · rw [Nat.add_sub_cancel' (le_of_lt hij)]; exact heq

/-- **Base + period + `base < C.states.card` form, from any cycle
state**: extends Tick 1785 with a bound on the base index.
From-mem parallel to
`adversarialTrace_pigeonhole_base_period_lt_of_init`. -/
theorem adversarialTrace_pigeonhole_base_period_lt_from_mem {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g₀ : GameState}
    (hg₀ : g₀ ∈ C.states) {t : ℕ → Piece}
    (hl : LegalSequenceFrom g₀.bag t) :
    ∃ base period : ℕ, base < C.states.card ∧
      1 ≤ period ∧ period ≤ C.states.card ∧
      adversarialTrace cfg C.solver t g₀ base =
        adversarialTrace cfg C.solver t g₀ (base + period) := by
  obtain ⟨i, j, hij, hj, heq⟩ := C.adversarialTrace_pigeonhole_lt_from_mem hg₀ hl
  refine ⟨i, j - i, ?_, ?_, ?_, ?_⟩
  · omega
  · omega
  · omega
  · rw [Nat.add_sub_cancel' (le_of_lt hij)]; exact heq

/-- **Base + period + base-state-in-cycle, from any cycle state**:
extends Tick 1786 by witnessing that the base state lies in
`C.states`. From-mem parallel to
`adversarialTrace_pigeonhole_base_period_in_states_of_init`. -/
theorem adversarialTrace_pigeonhole_base_period_in_states_from_mem
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {g₀ : GameState} (hg₀ : g₀ ∈ C.states)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    ∃ base period : ℕ, base < C.states.card ∧
      1 ≤ period ∧ period ≤ C.states.card ∧
      adversarialTrace cfg C.solver t g₀ base =
        adversarialTrace cfg C.solver t g₀ (base + period) ∧
      adversarialTrace cfg C.solver t g₀ base ∈ C.states := by
  obtain ⟨base, period, hbase, hpos, hperiod, heq⟩ :=
    C.adversarialTrace_pigeonhole_base_period_lt_from_mem hg₀ hl
  exact ⟨base, period, hbase, hpos, hperiod, heq,
    C.adversarialTrace_mem_states_from_mem hg₀ hl base⟩

/-- **`Set.Nonempty` form**: any nonempty cycle witnesses
`(safe cfg).Nonempty`. -/
theorem safe_nonempty {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (hne : C.states.Nonempty) :
    (safe cfg).Nonempty :=
  C.exists_mem_safe hne

/-- **Contrapositive of Tick 1321**: if `safe cfg = ∅`, every adversarial
closed cycle is empty. -/
theorem states_eq_empty_of_safe_eq_empty {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hsafe : safe cfg = ∅) :
    C.states = ∅ := by
  by_contra hne
  rcases Finset.nonempty_iff_ne_empty.mpr hne with ⟨g, hg⟩
  have : g ∈ safe cfg := C.mem_safe hg
  rw [hsafe] at this
  exact this.elim

/-- **Contrapositive of `init_safe_of_init_mem`**: if `init` is not safe,
no adversarial closed cycle contains `init`. -/
theorem init_not_mem_states_of_init_not_safe {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∉ safe cfg) :
    GameState.init ∉ C.states :=
  fun hmem => h (C.init_safe_of_init_mem hmem)

/-- **General contrapositive of `mem_safe`**: any state not in `safe cfg`
cannot be a member of any cycle. -/
theorem not_mem_states_of_not_mem_safe {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g : GameState} (h : g ∉ safe cfg) :
    g ∉ C.states :=
  fun hmem => h (C.mem_safe hmem)

/-- **Every state of an M4 artifact is safe**: parallel to
`AdversarialClosedCycle.subset_safe` at the M4 level. Direct via
`ofClosedAtlas M.closed` — the cycle extracted from M's closure has
the same states, and the cycle theorem applies. -/
theorem M4Artifact.subset_safe {cfg : GameConfig} (M : M4Artifact cfg) :
    (↑M.states : Set GameState) ⊆ safe cfg :=
  (ofClosedAtlas M.closed).subset_safe

/-- **Init is safe for any M4 artifact**: corollary of
`M4Artifact.subset_safe` + `M.init_mem`. -/
theorem M4Artifact.init_safe {cfg : GameConfig} (M : M4Artifact cfg) :
    GameState.init ∈ safe cfg :=
  M.subset_safe (Finset.mem_coe.mpr M.init_mem)

/-- **M4 artifact's adversarial trace is reachable from init**:
for any M4 artifact M and any legal piece sequence, every state in
the adversarial trace under `M.atlas.toSolver` is reachable from init.
Direct via the cycle's
`adversarialTrace_reachable_of_init` applied to `ofClosedAtlas
M.closed`. The cycle's solver IS `M.atlas.toSolver` (by definition
unfolding through `ofClosedAtlas`). -/
theorem M4Artifact.toSolver_trace_reachable_of_init {cfg : GameConfig}
    (M : M4Artifact cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Reachable cfg (adversarialTrace cfg M.atlas.toSolver s
      GameState.init n) :=
  (ofClosedAtlas M.closed).adversarialTrace_reachable_of_init
    M.init_mem hl n

/-- **M4 artifact's materialised-solver `solverReachable` set
stays in M.states**: every `solverReachable M.atlas.toSolver g`
lies in `M.states`. M4-level lift of Tick 1895 via M.closed and
M.init_mem. -/
theorem M4Artifact.toSolver_solverReachable_mem {cfg : GameConfig}
    (M : M4Artifact cfg) {g : GameState}
    (hg : solverReachable M.atlas.toSolver g) :
    g ∈ M.states :=
  M.closed.toSolver_solverReachable_mem M.init_mem hg

/-- **M4 artifact's materialised-solver `solverReachable` states
are non-lost**: every `solverReachable M.atlas.toSolver g` is
non-lost. M4-level lift of Tick 1898. -/
theorem M4Artifact.toSolver_solverReachable_not_lost {cfg : GameConfig}
    (M : M4Artifact cfg) {g : GameState}
    (hg : solverReachable M.atlas.toSolver g) :
    ¬ g.lost cfg :=
  M.closed.toSolver_solverReachable_not_lost M.init_mem hg

/-- **M4 artifact's materialised-solver `solverReachable` set
is finite**: bounded by the finite `M.states`. M4-level lift
of Tick 1896. -/
theorem M4Artifact.toSolver_solverReachable_finite {cfg : GameConfig}
    (M : M4Artifact cfg) :
    {g | solverReachable M.atlas.toSolver g}.Finite :=
  M.closed.toSolver_solverReachable_finite M.init_mem


/-- **M4 artifact's adversarial trace stays in `safe cfg`**: composition
of `toSolver_trace_mem_states_from_mem` (trace stays in M.states) with
`M.subset_safe` (M.states ⊆ safe). For any M4 artifact M, the trace
under `M.atlas.toSolver` from init lies entirely in `safe cfg`. -/
theorem M4Artifact.toSolver_trace_safe_of_init {cfg : GameConfig}
    (M : M4Artifact cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg M.atlas.toSolver s GameState.init n ∈ safe cfg := by
  apply M.subset_safe
  rw [Finset.mem_coe]
  exact M.toSolver_trace_mem_states_from_mem M.init_mem hl n

/-- **Bundled trace invariants for M4 artifacts**: for any M4
artifact M and legal piece sequence, the trace under `M.atlas.toSolver`
from init satisfies ALL four semantic invariants simultaneously:
mem-states, reachable, safe, not-lost. A single-statement
"infinite play certificate". -/
theorem M4Artifact.toSolver_trace_invariants_of_init {cfg : GameConfig}
    (M : M4Artifact cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg M.atlas.toSolver s GameState.init n ∈ M.states ∧
    Reachable cfg
      (adversarialTrace cfg M.atlas.toSolver s GameState.init n) ∧
    adversarialTrace cfg M.atlas.toSolver s GameState.init n ∈ safe cfg ∧
    ¬ (adversarialTrace cfg M.atlas.toSolver s GameState.init n).lost cfg :=
  ⟨M.toSolver_trace_mem_states_from_mem M.init_mem hl n,
   M.toSolver_trace_reachable_of_init hl n,
   M.toSolver_trace_safe_of_init hl n,
   M.toSolver_trace_not_lost_from_mem M.init_mem hl n⟩

/-- **M4 atlas's solver plays the right piece and is valid at
certified states**: for `g ∈ M.states, p ∈ g.bag`,
`(M.atlas.toSolver g p).piece = p ∧ (M.atlas.toSolver g p).Valid cfg`.
The restricted-ValidSolver property at M's certified states.

Proof: M's closure totality gives `M.atlas g p = some pl` for some
pl; `Atlas.toSolver_apply_of_some` rewrites M.atlas.toSolver g p to
pl; M's closure validity gives the `piece` and `Valid` properties.

This is the M4 analog of `AdversarialClosedCycle.solver_valid_and_piece`
— note that `ValidSolver` doesn't hold for M.atlas.toSolver in
general (only at certified states), since at inactive queries the
default placement may not be valid for arbitrary configs. -/
theorem M4Artifact.toSolver_piece_and_valid_of_mem {cfg : GameConfig}
    (M : M4Artifact cfg) {g : GameState} (hg : g ∈ M.states)
    {p : Piece} (hp : p ∈ g.bag) :
    (M.atlas.toSolver g p).piece = p ∧
      (M.atlas.toSolver g p).Valid cfg := by
  obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp (M.closed.total g hg p hp)
  rw [Atlas.toSolver_apply_of_some hpl]
  exact M.closed.valid g hg p hp pl hpl

/-- **Single adversarial step preserves M.states**: for `g ∈ M.states`
and `p ∈ g.bag`, the result of one adversarial step under
`M.atlas.toSolver` stays in `M.states`. Direct from M's closure
constraint after materialising the atlas. -/
theorem M4Artifact.toSolver_adversarialStep_mem_states {cfg : GameConfig}
    (M : M4Artifact cfg) {g : GameState} (hg : g ∈ M.states)
    {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (M.atlas.toSolver g p) ∈ M.states := by
  obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp (M.closed.total g hg p hp)
  rw [Atlas.toSolver_apply_of_some hpl]
  exact M.closed.closed g hg p hp pl hpl

/-- **Single adversarial step is not lost**: the post-step state is
non-lost. Composition of Tick 1726 (`toSolver_adversarialStep_mem_states`)
with M's closure `not_lost` field. -/
theorem M4Artifact.toSolver_adversarialStep_not_lost {cfg : GameConfig}
    (M : M4Artifact cfg) {g : GameState} (hg : g ∈ M.states)
    {p : Piece} (hp : p ∈ g.bag) :
    ¬ (adversarialStep cfg g p (M.atlas.toSolver g p)).lost cfg :=
  M.closed.not_lost _ (M.toSolver_adversarialStep_mem_states hg hp)

/-- **Single adversarial step lands in `safe cfg`**: the post-step
state is safe. Composition of Tick 1726 with Tick 1719's
`M4Artifact.subset_safe`. -/
theorem M4Artifact.toSolver_adversarialStep_safe {cfg : GameConfig}
    (M : M4Artifact cfg) {g : GameState} (hg : g ∈ M.states)
    {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (M.atlas.toSolver g p) ∈ safe cfg :=
  M.subset_safe
    (Finset.mem_coe.mpr (M.toSolver_adversarialStep_mem_states hg hp))

/-- **Bundled single-step invariants**: for `g ∈ M.states, p ∈ g.bag`,
one adversarial step under `M.atlas.toSolver` satisfies all three
step-level invariants simultaneously: mem-states, not-lost, safe.
Parallel to Tick 1722's bundled trace invariants but at the single-
step granularity. -/
theorem M4Artifact.toSolver_adversarialStep_invariants {cfg : GameConfig}
    (M : M4Artifact cfg) {g : GameState} (hg : g ∈ M.states)
    {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (M.atlas.toSolver g p) ∈ M.states ∧
    ¬ (adversarialStep cfg g p (M.atlas.toSolver g p)).lost cfg ∧
    adversarialStep cfg g p (M.atlas.toSolver g p) ∈ safe cfg :=
  ⟨M.toSolver_adversarialStep_mem_states hg hp,
   M.toSolver_adversarialStep_not_lost hg hp,
   M.toSolver_adversarialStep_safe hg hp⟩

/-- **Full single-step invariants (5 conjuncts)**: extends Tick 1728's
3-conjunct step bundle with the structural validity invariants
(piece + valid). Parallel to Tick 1725's full 6-conjunct trace
bundle, at the single-step granularity. -/
theorem M4Artifact.toSolver_adversarialStep_full_invariants {cfg : GameConfig}
    (M : M4Artifact cfg) {g : GameState} (hg : g ∈ M.states)
    {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (M.atlas.toSolver g p) ∈ M.states ∧
    ¬ (adversarialStep cfg g p (M.atlas.toSolver g p)).lost cfg ∧
    adversarialStep cfg g p (M.atlas.toSolver g p) ∈ safe cfg ∧
    (M.atlas.toSolver g p).piece = p ∧
    (M.atlas.toSolver g p).Valid cfg :=
  ⟨M.toSolver_adversarialStep_mem_states hg hp,
   M.toSolver_adversarialStep_not_lost hg hp,
   M.toSolver_adversarialStep_safe hg hp,
   (M.toSolver_piece_and_valid_of_mem hg hp).1,
   (M.toSolver_piece_and_valid_of_mem hg hp).2⟩

/-- **M4 atlas's solver gives valid piece response at every trace
state**: for any M4 artifact M, legal piece sequence, and step n,
the solver at the trace state for the drawn piece `s n` plays piece
`s n` and is valid. Combines Tick 1723's `toSolver_piece_and_valid_of_mem`
with Tick 1722's trace-stays-in-states and bag identification via
`adversarialTrace_bag` + `LegalSequence.canDraw`. -/
theorem M4Artifact.toSolver_trace_piece_and_valid {cfg : GameConfig}
    (M : M4Artifact cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    (M.atlas.toSolver
      (adversarialTrace cfg M.atlas.toSolver s GameState.init n)
        (s n)).piece = s n ∧
    (M.atlas.toSolver
      (adversarialTrace cfg M.atlas.toSolver s GameState.init n)
        (s n)).Valid cfg := by
  have hg : adversarialTrace cfg M.atlas.toSolver s GameState.init n
              ∈ M.states :=
    M.toSolver_trace_mem_states_from_mem M.init_mem hl n
  have hp : s n ∈
      (adversarialTrace cfg M.atlas.toSolver s GameState.init n).bag := by
    have := hl.canDraw n
    rw [Bag.canDraw_iff_mem, ← adversarialTrace_bag] at this
    exact this
  exact M.toSolver_piece_and_valid_of_mem hg hp

/-- **Full M4 trace invariants (6 conjuncts)**: maximally bundled
infinite play + structural correctness certificate. For any M4
artifact M and legal piece sequence, the trace under `M.atlas.toSolver`
satisfies ALL semantic AND structural invariants. -/
theorem M4Artifact.toSolver_trace_full_invariants_of_init
    {cfg : GameConfig} (M : M4Artifact cfg)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg M.atlas.toSolver s GameState.init n
      ∈ M.states ∧
    Reachable cfg
      (adversarialTrace cfg M.atlas.toSolver s GameState.init n) ∧
    adversarialTrace cfg M.atlas.toSolver s GameState.init n
      ∈ safe cfg ∧
    ¬ (adversarialTrace cfg M.atlas.toSolver s GameState.init n).lost
      cfg ∧
    (M.atlas.toSolver
      (adversarialTrace cfg M.atlas.toSolver s GameState.init n)
        (s n)).piece = s n ∧
    (M.atlas.toSolver
      (adversarialTrace cfg M.atlas.toSolver s GameState.init n)
        (s n)).Valid cfg :=
  ⟨M.toSolver_trace_mem_states_from_mem M.init_mem hl n,
   M.toSolver_trace_reachable_of_init hl n,
   M.toSolver_trace_safe_of_init hl n,
   M.toSolver_trace_not_lost_from_mem M.init_mem hl n,
   (M.toSolver_trace_piece_and_valid hl n).1,
   (M.toSolver_trace_piece_and_valid hl n).2⟩

/-- **`M.canon` trace equals `M` trace**: for any M4 artifact M and
legal piece sequence, the adversarial trace under `M.canon.atlas.toSolver`
equals the trace under `M.atlas.toSolver` at every step. Proof: by
induction on n — at each step, the trace state is in M.states (Tick
1722), so the solver values agree by Tick 1766 (canon agrees with
original at active queries). -/
theorem M4Artifact.adversarialTrace_canon_eq_self {cfg : GameConfig}
    (M : M4Artifact cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg M.canon.atlas.toSolver s GameState.init n =
      adversarialTrace cfg M.atlas.toSolver s GameState.init n := by
  induction n with
  | zero => rfl
  | succ k ih =>
    rw [adversarialTrace_succ, adversarialTrace_succ, ih]
    have hg : adversarialTrace cfg M.atlas.toSolver s GameState.init k
              ∈ M.states :=
      M.toSolver_trace_mem_states_from_mem M.init_mem hl k
    have hp : s k ∈
        (adversarialTrace cfg M.atlas.toSolver s GameState.init k).bag := by
      have := hl.canDraw k
      rw [Bag.canDraw_iff_mem, ← adversarialTrace_bag] at this
      exact this
    rw [M.canon_atlas_toSolver_eq_at_active hg hp]

/-- **Adversarial trace agrees for Equiv M4 artifacts**: if `M.Equiv
M'`, then their adversarial traces under their respective materialised
solvers are bit-identical at every step. Operational consequence of
the canonical equivalence: equivalent M4 artifacts produce identical
gameplay traces. Chain proof: M trace = M.canon trace (Tick 1769
symm) = M'.canon trace (canon_eq_of_equiv, Tick 1705) = M' trace
(Tick 1769). -/
theorem M4Artifact.adversarialTrace_eq_of_equiv {cfg : GameConfig}
    {M M' : M4Artifact cfg} (h : M.Equiv M')
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg M.atlas.toSolver s GameState.init n =
      adversarialTrace cfg M'.atlas.toSolver s GameState.init n := by
  have heq : M.canon = M'.canon := canon_eq_of_equiv h
  rw [← M.adversarialTrace_canon_eq_self hl n, heq,
      M'.adversarialTrace_canon_eq_self hl n]

/-- **`M.canon` trace equals `M` trace from any state in M.states**:
generalization of Tick 1769 to start from any `g₀ ∈ M.states` (not
just init). Same proof pattern but uses `toSolver_trace_mem_states_from_mem`
+ `adversarialTrace_bag_from` + `LegalSequenceFrom.canDraw`. -/
theorem M4Artifact.adversarialTrace_canon_eq_self_from_mem
    {cfg : GameConfig} (M : M4Artifact cfg)
    {g₀ : GameState} (hg₀ : g₀ ∈ M.states)
    {s : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag s) (n : ℕ) :
    adversarialTrace cfg M.canon.atlas.toSolver s g₀ n =
      adversarialTrace cfg M.atlas.toSolver s g₀ n := by
  induction n with
  | zero => rfl
  | succ k ih =>
    rw [adversarialTrace_succ, adversarialTrace_succ, ih]
    have hg : adversarialTrace cfg M.atlas.toSolver s g₀ k ∈ M.states :=
      M.toSolver_trace_mem_states_from_mem hg₀ hl k
    have hp : s k ∈
        (adversarialTrace cfg M.atlas.toSolver s g₀ k).bag := by
      have := hl.canDraw k
      rw [Bag.canDraw_iff_mem, ← adversarialTrace_bag_from] at this
      exact this
    rw [M.canon_atlas_toSolver_eq_at_active hg hp]

/-- **Canon's trace stays in M.states**: useful named consequence of
Tick 1769 + Tick 1722. The trace under M.canon's materialised solver
from init stays in M.states. -/
theorem M4Artifact.canon_toSolver_trace_mem_states {cfg : GameConfig}
    (M : M4Artifact cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg M.canon.atlas.toSolver s GameState.init n
      ∈ M.states := by
  rw [M.adversarialTrace_canon_eq_self hl n]
  exact M.toSolver_trace_mem_states_from_mem M.init_mem hl n

/-- **Filtering cycle states by `safe` is the identity** — every cycle
state is already safe. -/
theorem states_filter_mem_safe {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) [DecidablePred (· ∈ safe cfg)] :
    C.states.filter (· ∈ safe cfg) = C.states :=
  Finset.filter_eq_self.mpr (fun _ h => C.mem_safe h)

/-- **Filtering cycle states by non-membership of `safe` is empty** —
no cycle state lies outside the safe set. -/
theorem states_filter_not_mem_safe {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) [DecidablePred (· ∈ safe cfg)] :
    C.states.filter (· ∉ safe cfg) = ∅ :=
  Finset.filter_eq_empty_iff.mpr (fun _ h hns => hns (C.mem_safe h))

/-- **Step from a cycle state lands in `safe`.** Composes
`step_mem_states` with `mem_safe`. -/
theorem step_mem_safe {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {g : GameState} (hg : g ∈ C.states) {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (C.solver g p) ∈ safe cfg :=
  C.mem_safe (C.step_mem_states hg hp)

/-- **Init-step lands in `safe`.** Init-specialisation of `step_mem_safe`. -/
theorem step_mem_safe_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) (p : Piece) :
    adversarialStep cfg GameState.init p (C.solver GameState.init p) ∈ safe cfg :=
  C.mem_safe (C.step_mem_states_of_init h p)

/-- **Cycle single-step full invariants (5 conjuncts)**: for any
`g ∈ C.states, p ∈ g.bag`, the cycle's single-step result satisfies
all five invariants simultaneously: mem-states, not-lost, safe,
piece, valid. Parallel to Tick 1729's M4 single-step full bundle. -/
theorem step_full_invariants {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {g : GameState} (hg : g ∈ C.states)
    {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (C.solver g p) ∈ C.states ∧
    ¬ (adversarialStep cfg g p (C.solver g p)).lost cfg ∧
    adversarialStep cfg g p (C.solver g p) ∈ safe cfg ∧
    (C.solver g p).piece = p ∧
    (C.solver g p).Valid cfg :=
  ⟨C.step_mem_states hg hp,
   C.step_not_lost hg hp,
   C.step_mem_safe hg hp,
   C.solver_piece hg hp,
   (C.valid g hg p hp).2⟩

/-- **Cycle trace full invariants (6 conjuncts)**: for any
init-containing cycle and legal piece sequence, the trace under
`C.solver` from init satisfies all six invariants. Parallel to Tick
1725's M4 trace full bundle. -/
theorem adversarialTrace_full_invariants_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg C.solver s GameState.init n ∈ C.states ∧
    Reachable cfg (adversarialTrace cfg C.solver s GameState.init n) ∧
    adversarialTrace cfg C.solver s GameState.init n ∈ safe cfg ∧
    ¬ (adversarialTrace cfg C.solver s GameState.init n).lost cfg ∧
    (C.solver
      (adversarialTrace cfg C.solver s GameState.init n) (s n)).piece
      = s n ∧
    (C.solver
      (adversarialTrace cfg C.solver s GameState.init n) (s n)).Valid
      cfg := by
  have hg : adversarialTrace cfg C.solver s GameState.init n ∈ C.states :=
    C.adversarialTrace_mem_states_of_init h hl n
  have hbag : s n ∈
      (adversarialTrace cfg C.solver s GameState.init n).bag := by
    have := hl.canDraw n
    rw [Bag.canDraw_iff_mem, ← adversarialTrace_bag] at this
    exact this
  have hvp := C.valid _ hg (s n) hbag
  exact ⟨hg,
    C.adversarialTrace_reachable_of_init h hl n,
    C.mem_safe hg,
    C.adversarialTrace_not_lost_of_init h hl n,
    hvp.1, hvp.2⟩

/-- **Adversarial trace from `init` under the cycle solver lies in
`safe cfg`.** Composes `adversarialTrace_mem_states_of_init` with
`mem_safe`. -/
theorem adversarialTrace_mem_safe_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg C.solver s GameState.init n ∈ safe cfg :=
  C.mem_safe (C.adversarialTrace_mem_states_of_init h hl n)

/-- **`Set.range` form**: the full image of the adversarial trace from
`init` along a legal sequence is contained in `safe cfg`. -/
theorem adversarialTrace_range_subset_safe_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    Set.range (adversarialTrace cfg C.solver s GameState.init) ⊆ safe cfg :=
  Set.range_subset_iff.mpr (C.adversarialTrace_mem_safe_of_init h hl)

/-- **`Set.range` form for `C.states`**: the trace image is contained in
the cycle's state set. -/
theorem adversarialTrace_range_subset_states_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    Set.range (adversarialTrace cfg C.solver s GameState.init) ⊆ C.states :=
  Set.range_subset_iff.mpr (C.adversarialTrace_mem_states_of_init h hl)

/-- **`Set.range` form for `solverReachable C.solver`**: the trace image
is contained in `solverReachable`. -/
theorem adversarialTrace_range_subset_solverReachable_of_init
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    Set.range (adversarialTrace cfg C.solver s GameState.init) ⊆
      {g | solverReachable C.solver g} :=
  Set.range_subset_iff.mpr (fun n => C.adversarialTrace_solverReachable_of_init hl n)

/-- **Adversarial trace range is finite.** Contained in the Finset
`C.states`, hence finite. -/
theorem adversarialTrace_range_finite_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    (Set.range (adversarialTrace cfg C.solver s GameState.init)).Finite :=
  (C.states.finite_toSet).subset (C.adversarialTrace_range_subset_states_of_init h hl)

/-- **Pigeonhole on `adversarialTrace` through an init-containing
cycle**: indices in `0..C.states.card` cannot be all distinct under
the trace map (since the trace lives in `C.states`), so two indices
share a trace value. -/
theorem adversarialTrace_pigeonhole_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ i j : ℕ, i ≤ C.states.card ∧ j ≤ C.states.card ∧ i ≠ j ∧
      adversarialTrace cfg C.solver s GameState.init i =
        adversarialTrace cfg C.solver s GameState.init j := by
  set N := C.states.card
  have hmap : ∀ k ∈ Finset.range (N + 1),
      adversarialTrace cfg C.solver s GameState.init k ∈ C.states :=
    fun k _ => C.adversarialTrace_mem_states_of_init h hl k
  have hcard : N < (Finset.range (N + 1)).card := by
    rw [Finset.card_range]; omega
  obtain ⟨i, hi, j, hj, hne, heq⟩ :=
    Finset.exists_ne_map_eq_of_card_lt_of_maps_to hcard hmap
  rw [Finset.mem_range] at hi hj
  exact ⟨i, j, by omega, by omega, hne, heq⟩

/-- **Ordered pigeonhole**: same as Tick 1347 but with `i < j`. -/
theorem adversarialTrace_pigeonhole_lt_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ i j : ℕ, i < j ∧ j ≤ C.states.card ∧
      adversarialTrace cfg C.solver s GameState.init i =
        adversarialTrace cfg C.solver s GameState.init j := by
  obtain ⟨i, j, hi, hj, hne, heq⟩ := C.adversarialTrace_pigeonhole_of_init h hl
  rcases lt_or_gt_of_ne hne with hlt | hgt
  · exact ⟨i, j, hlt, hj, heq⟩
  · exact ⟨j, i, hgt, hi, heq.symm⟩

/-- **Ordered pigeonhole with state membership**: same as Tick 1348
but the common trace state lies in `C.states`. -/
theorem adversarialTrace_pigeonhole_lt_in_states_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ i j : ℕ, i < j ∧ j ≤ C.states.card ∧
      adversarialTrace cfg C.solver s GameState.init i =
        adversarialTrace cfg C.solver s GameState.init j ∧
      adversarialTrace cfg C.solver s GameState.init i ∈ C.states := by
  obtain ⟨i, j, hij, hj, heq⟩ := C.adversarialTrace_pigeonhole_lt_of_init h hl
  exact ⟨i, j, hij, hj, heq, C.adversarialTrace_mem_states_of_init h hl i⟩

/-- **Base+period form of pigeonhole**: from `i < j` with shared trace
value, repackage as `∃ base, period ≥ 1, trace base = trace (base + period)`.
The period is at most `C.states.card`. -/
theorem adversarialTrace_pigeonhole_base_period_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ base period : ℕ, 1 ≤ period ∧ period ≤ C.states.card ∧
      adversarialTrace cfg C.solver s GameState.init base =
        adversarialTrace cfg C.solver s GameState.init (base + period) := by
  obtain ⟨i, j, hij, hj, heq⟩ := C.adversarialTrace_pigeonhole_lt_of_init h hl
  refine ⟨i, j - i, ?_, ?_, ?_⟩
  · omega
  · omega
  · rw [Nat.add_sub_cancel' (le_of_lt hij)]; exact heq

/-- **Base+period+`base < C.states.card` form**: extends Tick 1351 with
a bound on the base index. -/
theorem adversarialTrace_pigeonhole_base_period_lt_of_init {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ base period : ℕ, base < C.states.card ∧
      1 ≤ period ∧ period ≤ C.states.card ∧
      adversarialTrace cfg C.solver s GameState.init base =
        adversarialTrace cfg C.solver s GameState.init (base + period) := by
  obtain ⟨i, j, hij, hj, heq⟩ := C.adversarialTrace_pigeonhole_lt_of_init h hl
  refine ⟨i, j - i, ?_, ?_, ?_, ?_⟩
  · omega
  · omega
  · omega
  · rw [Nat.add_sub_cancel' (le_of_lt hij)]; exact heq

/-- **Base+period+base-state-in-cycle**: extends Tick 1352 by witnessing
that the base state lies in `C.states`. -/
theorem adversarialTrace_pigeonhole_base_period_in_states_of_init
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ base period : ℕ, base < C.states.card ∧
      1 ≤ period ∧ period ≤ C.states.card ∧
      adversarialTrace cfg C.solver s GameState.init base =
        adversarialTrace cfg C.solver s GameState.init (base + period) ∧
      adversarialTrace cfg C.solver s GameState.init base ∈ C.states := by
  obtain ⟨base, period, hbase, hpos, hperiod, heq⟩ :=
    C.adversarialTrace_pigeonhole_base_period_lt_of_init h hl
  exact ⟨base, period, hbase, hpos, hperiod, heq,
    C.adversarialTrace_mem_states_of_init h hl base⟩

/-- Parametric form: an adversarial closed cycle on any `cfg` containing
`init` proves `TetrisSolvableFor cfg`. -/
theorem init_solves_for {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) : TetrisSolvableFor cfg :=
  safe_extract_for (C.subset_safe (Finset.mem_coe.mpr h))

/-- An adversarial closed cycle containing `init` proves Tetris is solvable.
This is the **complete chain**: build a cycle ⇒ safety ⇒ solver. -/
theorem init_solves (C : AdversarialClosedCycle GameConfig.standard)
    (h : GameState.init ∈ C.states) : TetrisSolvable :=
  C.init_solves_for h

/-- **Single-state cycles are impossible.** Any cycle containing a state with
a non-empty bag has at least two states — because drawing the chosen piece
necessarily changes the bag (`Bag.draw_ne_self`), so the next state differs. -/
theorem states_card_ge_two {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {s : GameState} (hs : s ∈ C.states) {p : Piece} (hp : p ∈ s.bag) :
    2 ≤ C.states.card := by
  set s' := adversarialStep cfg s p (C.solver s p)
  have hs' : s' ∈ C.states := C.closed s hs p hp
  have hs_ne : s ≠ s' := by
    intro heq
    have hbag : s.bag = s'.bag := congrArg GameState.bag heq
    -- s'.bag = s.bag.draw p by def, and draw_ne_self gives ≠
    exact Bag.draw_ne_self hp hbag.symm
  have hlt : 1 < C.states.card := Finset.one_lt_card.mpr ⟨s, hs, s', hs', hs_ne⟩
  omega

/-- **Bag-card lower bound.** Any cycle containing `s` has at least
`s.bag.card` states. Proof: the map `p ↦ adversarialStep s p (solver s p)`
sends `s.bag` injectively into `C.states` (injectivity from `Bag.draw_injOn`
on the bag-projection of the step). -/
theorem states_card_ge_bag_card {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {s : GameState} (hs : s ∈ C.states) :
    s.bag.card ≤ C.states.card := by
  let f : Piece → GameState := fun p => adversarialStep cfg s p (C.solver s p)
  have hsubset : s.bag.image f ⊆ C.states := by
    intro x hx
    rcases Finset.mem_image.mp hx with ⟨p, hp, rfl⟩
    exact C.closed s hs p hp
  have hinj : Set.InjOn f ↑s.bag := by
    intro p hp q hq hpq
    have hbag : (f p).bag = (f q).bag := congrArg GameState.bag hpq
    -- (f p).bag = s.bag.draw p by def of adversarialStep
    change s.bag.draw p = s.bag.draw q at hbag
    exact Bag.draw_injOn s.bag hp hq hbag
  have hcard : (s.bag.image f).card = s.bag.card := Finset.card_image_of_injOn hinj
  calc s.bag.card
      = (s.bag.image f).card := hcard.symm
    _ ≤ C.states.card := Finset.card_le_card hsubset

/-- A cycle containing `init` has `states.card ≠ 1` — `init.bag` is
non-empty so cycle size ≥ 2. -/
theorem init_states_card_ne_one {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    C.states.card ≠ 1 := by
  have h2 : 2 ≤ C.states.card :=
    C.states_card_ge_two h (GameState.init_bag.symm ▸ Bag.mem_full Piece.O)
  omega

/-- A cycle containing `init` has `states.card ≥ 2`. -/
theorem init_states_card_ge_two {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    2 ≤ C.states.card :=
  C.states_card_ge_two h (GameState.init_bag.symm ▸ Bag.mem_full Piece.O)


/-- **Strict bag-card lower bound.** A cycle containing `s` (with a non-empty
bag) has at least `s.bag.card + 1` states. Proof: extend
`states_card_ge_bag_card` by observing `s` itself is not in the image
`s.bag.image f` — for any `p ∈ s.bag`, `(f p).bag = s.bag.draw p ≠ s.bag`
by `Bag.draw_ne_self`. -/
theorem states_card_gt_bag_card {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {s : GameState} (hs : s ∈ C.states) (hne : s.bag.Nonempty) :
    s.bag.card + 1 ≤ C.states.card := by
  let f : Piece → GameState := fun p => adversarialStep cfg s p (C.solver s p)
  have hsubset : s.bag.image f ⊆ C.states := by
    intro x hx
    rcases Finset.mem_image.mp hx with ⟨p, hp, rfl⟩
    exact C.closed s hs p hp
  have hinj : Set.InjOn f ↑s.bag := by
    intro p hp q hq hpq
    have hbag : (f p).bag = (f q).bag := congrArg GameState.bag hpq
    change s.bag.draw p = s.bag.draw q at hbag
    exact Bag.draw_injOn s.bag hp hq hbag
  have hcard : (s.bag.image f).card = s.bag.card := Finset.card_image_of_injOn hinj
  have hs_notin : s ∉ s.bag.image f := by
    intro hmem
    rcases Finset.mem_image.mp hmem with ⟨p, hp, hfp⟩
    -- s = f p ⇒ s.bag = (f p).bag = s.bag.draw p, contradicting draw_ne_self.
    have hbag : s.bag = (f p).bag := congrArg GameState.bag hfp.symm
    change s.bag = s.bag.draw p at hbag
    exact Bag.draw_ne_self hp hbag.symm
  -- Insert s into the image; cards add, all sit inside C.states.
  have hsubset' : insert s (s.bag.image f) ⊆ C.states := by
    intro x hx
    rcases Finset.mem_insert.mp hx with rfl | hx
    · exact hs
    · exact hsubset hx
  have hcard' : (insert s (s.bag.image f)).card = s.bag.card + 1 := by
    rw [Finset.card_insert_of_notMem hs_notin, hcard]
  calc s.bag.card + 1
      = (insert s (s.bag.image f)).card := hcard'.symm
    _ ≤ C.states.card := Finset.card_le_card hsubset'

/-- A cycle containing `init` has `states.card ≥ 8` (= init.bag.card + 1). -/
theorem init_states_card_ge_eight_via_bag {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    8 ≤ C.states.card := by
  have hne : GameState.init.bag.Nonempty :=
    GameState.init_bag.symm ▸ Bag.full_nonempty
  have h8 : GameState.init.bag.card + 1 ≤ C.states.card :=
    C.states_card_gt_bag_card h hne
  rw [GameState.init_bag_card] at h8
  exact h8

/-- Weakenings of `init_states_card_ge_eight_via_bag` for `3..7`. -/
theorem init_states_card_ge_three {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    3 ≤ C.states.card :=
  Nat.le_trans (by decide : (3 : ℕ) ≤ 8) (C.init_states_card_ge_eight_via_bag h)

theorem init_states_card_ge_four {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    4 ≤ C.states.card :=
  Nat.le_trans (by decide : (4 : ℕ) ≤ 8) (C.init_states_card_ge_eight_via_bag h)

theorem init_states_card_ge_five {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    5 ≤ C.states.card :=
  Nat.le_trans (by decide : (5 : ℕ) ≤ 8) (C.init_states_card_ge_eight_via_bag h)

theorem init_states_card_ge_six {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    6 ≤ C.states.card :=
  Nat.le_trans (by decide : (6 : ℕ) ≤ 8) (C.init_states_card_ge_eight_via_bag h)

/-- `states.card ≠ k` for `k ∈ {2, ..., 7}` via the `≥ 8` bound. -/
theorem init_states_card_ne_two {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    C.states.card ≠ 2 := by
  have := C.init_states_card_ge_eight_via_bag h; omega

theorem init_states_card_ne_three {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    C.states.card ≠ 3 := by
  have := C.init_states_card_ge_eight_via_bag h; omega

theorem init_states_card_ne_seven {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    C.states.card ≠ 7 := by
  have := C.init_states_card_ge_eight_via_bag h; omega

theorem init_states_card_ne_four {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    C.states.card ≠ 4 := by
  have := C.init_states_card_ge_eight_via_bag h; omega

theorem init_states_card_ne_five {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    C.states.card ≠ 5 := by
  have := C.init_states_card_ge_eight_via_bag h; omega

theorem init_states_card_ne_six {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    C.states.card ≠ 6 := by
  have := C.init_states_card_ge_eight_via_bag h; omega

/-- **Depth-2 init bound: 14.** Any cycle containing `GameState.init` has
at least 14 states. Three-layer partition by `bag.card`:

* depth-0: `{init}` — 1 state, bag.card = 7.
* depth-1: `f p := adversarialStep cfg init p (solver init p)` for `p ∈ Bag.full`
  — 7 states (injectivity from `Bag.draw_injOn`), bag.card = 6.
* depth-2: pick `s' := f Piece.O`; `g q := adversarialStep cfg s' q (solver s' q)`
  for `q ∈ s'.bag` — 6 states, bag.card = 5.

All three layers are pairwise disjoint by bag.card, each layer lives in
`C.states` (`C.closed` twice), totalling 1 + 7 + 6 = 14. -/
theorem init_states_card_ge_fourteen {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    14 ≤ C.states.card := by
  let f : Piece → GameState := fun p =>
    adversarialStep cfg GameState.init p (C.solver GameState.init p)
  let s' : GameState := f Piece.O
  let g : Piece → GameState := fun q =>
    adversarialStep cfg s' q (C.solver s' q)
  have hs' : s' ∈ C.states := C.closed _ h Piece.O (Bag.mem_full _)
  have hs'_card : s'.bag.card = 6 := init_step_bag_card cfg Piece.O _
  -- Layer 1: image f
  have hf_inj : Set.InjOn f ↑Bag.full := by
    intro p _ q _ hpq
    have hbag : (f p).bag = (f q).bag := congrArg GameState.bag hpq
    change Bag.full.draw p = Bag.full.draw q at hbag
    exact Bag.draw_injOn Bag.full (Bag.mem_full _) (Bag.mem_full _) hbag
  have hf_subset : Bag.full.image f ⊆ C.states := by
    intro x hx
    rcases Finset.mem_image.mp hx with ⟨p, hp, rfl⟩
    exact C.closed _ h p hp
  have hf_card : (Bag.full.image f).card = 7 := by
    rw [Finset.card_image_of_injOn hf_inj]
    decide
  have hf_bag_six : ∀ x ∈ Bag.full.image f, x.bag.card = 6 := by
    intro x hx
    rcases Finset.mem_image.mp hx with ⟨p, _, rfl⟩
    exact init_step_bag_card cfg p _
  -- Layer 2: image g
  have hg_inj : Set.InjOn g ↑s'.bag := by
    intro p hp q hq hpq
    have hbag : (g p).bag = (g q).bag := congrArg GameState.bag hpq
    change s'.bag.draw p = s'.bag.draw q at hbag
    exact Bag.draw_injOn s'.bag hp hq hbag
  have hg_subset : s'.bag.image g ⊆ C.states := by
    intro x hx
    rcases Finset.mem_image.mp hx with ⟨q, hq, rfl⟩
    exact C.closed _ hs' q hq
  have hg_card : (s'.bag.image g).card = 6 := by
    rw [Finset.card_image_of_injOn hg_inj, hs'_card]
  have hg_bag_five : ∀ x ∈ s'.bag.image g, x.bag.card = 5 := by
    intro x hx
    rcases Finset.mem_image.mp hx with ⟨q, hq, rfl⟩
    change (s'.bag.draw q).card = 5
    have h2 : 2 ≤ s'.bag.card := by rw [hs'_card]; decide
    rw [Bag.card_draw_of_ge_two hq h2, hs'_card]
  -- Layer 0: init
  have hinit_bag : GameState.init.bag.card = 7 := by
    change Bag.full.card = 7
    decide
  -- Disjointness via bag.card
  have h_init_notin_f : GameState.init ∉ Bag.full.image f := by
    intro hmem
    have h6 := hf_bag_six _ hmem
    rw [hinit_bag] at h6
    omega
  have h_init_notin_g : GameState.init ∉ s'.bag.image g := by
    intro hmem
    have h5 := hg_bag_five _ hmem
    rw [hinit_bag] at h5
    omega
  have h_f_disj_g : Disjoint (Bag.full.image f) (s'.bag.image g) := by
    rw [Finset.disjoint_left]
    intro x hxf hxg
    have h6 := hf_bag_six _ hxf
    have h5 := hg_bag_five _ hxg
    omega
  -- Assemble
  set S := Bag.full.image f ∪ s'.bag.image g with hS_def
  have hS_card : S.card = 13 := by
    rw [hS_def, Finset.card_union_of_disjoint h_f_disj_g, hf_card, hg_card]
  have hinit_notin_S : GameState.init ∉ S := by
    intro hmem
    rw [hS_def] at hmem
    rcases Finset.mem_union.mp hmem with hl | hr
    · exact h_init_notin_f hl
    · exact h_init_notin_g hr
  have hS_sub : insert GameState.init S ⊆ C.states := by
    intro x hx
    rcases Finset.mem_insert.mp hx with rfl | hx
    · exact h
    · rw [hS_def] at hx
      rcases Finset.mem_union.mp hx with hl | hr
      · exact hf_subset hl
      · exact hg_subset hr
  have hcard : (insert GameState.init S).card = 14 := by
    rw [Finset.card_insert_of_notMem hinit_notin_S, hS_card]
  calc 14
      = (insert GameState.init S).card := hcard.symm
    _ ≤ C.states.card := Finset.card_le_card hS_sub

/-- **Depth-3 init bound: 19.** Same partition strategy as
`init_states_card_ge_fourteen`, extended by one more layer. Using
the `C.next`/`C.layer` helpers (Tick 57) makes the proof much cleaner. -/
theorem init_states_card_ge_nineteen {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    19 ≤ C.states.card := by
  -- Pick s' ∈ Layer 1 (any piece — use Piece.O)
  let s' : GameState := C.next GameState.init Piece.O
  have hs' : s' ∈ C.states := C.next_mem h (Bag.mem_full _)
  have hs'_card : s'.bag.card = 6 := init_step_bag_card cfg Piece.O _
  -- Pick s'' ∈ Layer 2 (any q ∈ s'.bag — exists by nonemptiness)
  have hs'_ne : s'.bag.Nonempty := by
    rw [← Finset.card_pos, hs'_card]; decide
  obtain ⟨q, hq⟩ := hs'_ne
  let s'' : GameState := C.next s' q
  have hs'' : s'' ∈ C.states := C.next_mem hs' hq
  have hs''_card : s''.bag.card = 5 := by
    change (s'.bag.draw q).card = 5
    rw [Bag.card_draw_of_ge_two hq (by rw [hs'_card]; decide), hs'_card]
  -- Bag-card facts
  have h_init_bag : GameState.init.bag.card = 7 := by
    change Bag.full.card = 7; decide
  have h_L1_bag : ∀ x ∈ C.layer GameState.init, x.bag.card = 6 := fun x hx => by
    have := C.layer_bag_card (s := GameState.init) (by rw [h_init_bag]; decide) hx
    rw [h_init_bag] at this; omega
  have h_L2_bag : ∀ x ∈ C.layer s', x.bag.card = 5 := fun x hx => by
    have := C.layer_bag_card (s := s') (by rw [hs'_card]; decide) hx
    rw [hs'_card] at this; omega
  have h_L3_bag : ∀ x ∈ C.layer s'', x.bag.card = 4 := fun x hx => by
    have := C.layer_bag_card (s := s'') (by rw [hs''_card]; decide) hx
    rw [hs''_card] at this; omega
  -- Layer sizes
  have h_L1_card : (C.layer GameState.init).card = 7 := by
    rw [C.layer_card]; exact h_init_bag
  have h_L2_card : (C.layer s').card = 6 := by
    rw [C.layer_card]; exact hs'_card
  have h_L3_card : (C.layer s'').card = 5 := by
    rw [C.layer_card]; exact hs''_card
  -- init disjoint from layers
  have h_init_notin_L1 : GameState.init ∉ C.layer GameState.init := fun hmem => by
    have := h_L1_bag _ hmem; rw [h_init_bag] at this; omega
  have h_init_notin_L2 : GameState.init ∉ C.layer s' := fun hmem => by
    have := h_L2_bag _ hmem; rw [h_init_bag] at this; omega
  have h_init_notin_L3 : GameState.init ∉ C.layer s'' := fun hmem => by
    have := h_L3_bag _ hmem; rw [h_init_bag] at this; omega
  -- Layer pairwise disjointness via bag.card
  have h_L1_L2 : Disjoint (C.layer GameState.init) (C.layer s') := by
    rw [Finset.disjoint_left]; intro x h1 h2
    have h6 := h_L1_bag _ h1; have h5 := h_L2_bag _ h2; omega
  have h_L1_L3 : Disjoint (C.layer GameState.init) (C.layer s'') := by
    rw [Finset.disjoint_left]; intro x h1 h3
    have h6 := h_L1_bag _ h1; have h4 := h_L3_bag _ h3; omega
  have h_L2_L3 : Disjoint (C.layer s') (C.layer s'') := by
    rw [Finset.disjoint_left]; intro x h2 h3
    have h5 := h_L2_bag _ h2; have h4 := h_L3_bag _ h3; omega
  have h_12_disj_3 : Disjoint (C.layer GameState.init ∪ C.layer s') (C.layer s'') := by
    rw [Finset.disjoint_union_left]; exact ⟨h_L1_L3, h_L2_L3⟩
  -- Assemble
  set S := C.layer GameState.init ∪ C.layer s' ∪ C.layer s'' with hS_def
  have hS_card : S.card = 18 := by
    rw [hS_def, Finset.card_union_of_disjoint h_12_disj_3,
        Finset.card_union_of_disjoint h_L1_L2, h_L1_card, h_L2_card, h_L3_card]
  have h_init_notin_S : GameState.init ∉ S := by
    intro hmem
    rw [hS_def] at hmem
    rcases Finset.mem_union.mp hmem with h12 | h3
    · rcases Finset.mem_union.mp h12 with h1 | h2
      · exact h_init_notin_L1 h1
      · exact h_init_notin_L2 h2
    · exact h_init_notin_L3 h3
  have h_sub : insert GameState.init S ⊆ C.states := by
    intro x hx
    rcases Finset.mem_insert.mp hx with rfl | hx
    · exact h
    · rw [hS_def] at hx
      rcases Finset.mem_union.mp hx with h12 | h3
      · rcases Finset.mem_union.mp h12 with h1 | h2
        · exact C.layer_subset h h1
        · exact C.layer_subset hs' h2
      · exact C.layer_subset hs'' h3
  calc 19
      = (insert GameState.init S).card := by
          rw [Finset.card_insert_of_notMem h_init_notin_S, hS_card]
    _ ≤ C.states.card := Finset.card_le_card h_sub

/-- **Chain existence by bag-card.** For any cycle containing `init`, and any
depth `k ≤ 6`, there exists a cycle state with bag.card = `7 - k`. Proof by
induction on `k`: base = `init` (bag.card = 7); step extracts an arbitrary
piece from the predecessor's bag (which is `≥ 2` because we're at depth `≤ 6`)
and uses `C.next_mem` + `Bag.card_draw_of_ge_two`. Foundation for future
inductive cycle-size bounds. -/
theorem exists_descent_chain {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    ∀ k, k ≤ 6 → ∃ s ∈ C.states, s.bag.card = 7 - k := by
  intro k hk
  induction k with
  | zero =>
    refine ⟨GameState.init, h, ?_⟩
    change Bag.full.card = 7 - 0
    decide
  | succ k ih =>
    obtain ⟨s, hs, hcard⟩ := ih (by omega)
    have h2 : 2 ≤ s.bag.card := by rw [hcard]; omega
    have hne : s.bag.Nonempty := by
      rw [← Finset.card_pos, hcard]; omega
    obtain ⟨p, hp⟩ := hne
    refine ⟨C.next s p, C.next_mem hs hp, ?_⟩
    change (s.bag.draw p).card = 7 - (k + 1)
    rw [Bag.card_draw_of_ge_two hp h2, hcard]
    omega

/-- **Indexed chain witness.** For each depth `k : Fin 7`, pick a cycle state
with `bag.card = 7 - k.val`. The seven chain states have distinct bag-cards,
so they are pairwise distinct as `GameState`s — yielding `7 ≤ C.states.card`
(weaker than the depth-2 bound but valid for any cycle containing `init`). -/
noncomputable def chainState {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) (k : Fin 7) : GameState :=
  Classical.choose (C.exists_descent_chain h k.val (Nat.lt_succ_iff.mp k.isLt))

theorem chainState_mem {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) (k : Fin 7) :
    C.chainState h k ∈ C.states :=
  (Classical.choose_spec (C.exists_descent_chain h k.val (Nat.lt_succ_iff.mp k.isLt))).1

theorem chainState_bag_card {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) (k : Fin 7) :
    (C.chainState h k).bag.card = 7 - k.val :=
  (Classical.choose_spec (C.exists_descent_chain h k.val (Nat.lt_succ_iff.mp k.isLt))).2

/-- The chain function is injective: distinct depths give distinct states
(different bag-cards). -/
theorem chainState_injective {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) : Function.Injective (C.chainState h) := by
  intro k₁ k₂ hk
  have hbag : (C.chainState h k₁).bag.card = (C.chainState h k₂).bag.card :=
    congrArg _ (congrArg GameState.bag hk)
  rw [C.chainState_bag_card, C.chainState_bag_card] at hbag
  -- 7 - k₁.val = 7 - k₂.val with k₁,k₂ < 7 ⇒ k₁ = k₂
  have h1 := k₁.isLt
  have h2 := k₂.isLt
  apply Fin.ext
  omega

/-- **Universal cycle bound: 7.** Any cycle containing `init` has at least 7
states (one per bag-card depth 7..1). Weaker than `init_states_card_ge_fourteen`,
but proven via the general indexed chain — pattern reusable for future bounds. -/
theorem init_states_card_ge_seven {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    7 ≤ C.states.card := by
  have hsub : Finset.univ.image (C.chainState h) ⊆ C.states := by
    intro x hx
    rcases Finset.mem_image.mp hx with ⟨k, _, rfl⟩
    exact C.chainState_mem h k
  have hcard : (Finset.univ.image (C.chainState h)).card = 7 := by
    rw [Finset.card_image_of_injective _ (C.chainState_injective h)]
    decide
  calc 7 = _ := hcard.symm
    _ ≤ _ := Finset.card_le_card hsub

/-- **Indexed layer at chain-depth `k`.** Layer of one-step successors from
the chain witness at depth `k`. Indexed by `Fin 6`: depths 0..5 (where the
source's bag has ≥ 2 pieces, so the draw is non-emptying and layer
bag-cards stay uniform). -/
noncomputable def chainLayer {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) (k : Fin 6) : Finset GameState :=
  C.layer (C.chainState h ⟨k.val, by omega⟩)

theorem chainLayer_subset {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) (k : Fin 6) :
    C.chainLayer h k ⊆ C.states :=
  C.layer_subset (C.chainState_mem h _)

theorem chainLayer_card {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) (k : Fin 6) :
    (C.chainLayer h k).card = 7 - k.val := by
  unfold chainLayer
  rw [C.layer_card, C.chainState_bag_card]

theorem chainLayer_bag_card {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) (k : Fin 6)
    {x : GameState} (hx : x ∈ C.chainLayer h k) :
    x.bag.card = 6 - k.val := by
  have hk : k.val < 6 := k.isLt
  have hsrc_card : (C.chainState h ⟨k.val, by omega⟩).bag.card = 7 - k.val := by
    rw [C.chainState_bag_card]
  have h2 : 2 ≤ (C.chainState h ⟨k.val, by omega⟩).bag.card := by
    rw [hsrc_card]; omega
  unfold chainLayer at hx
  have h3 := C.layer_bag_card h2 hx
  rw [hsrc_card] at h3
  omega

/-- `init` is never in a `chainLayer` (bag.card = 7 vs ≤ 6). -/
theorem init_notMem_chainLayer {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) (k : Fin 6) :
    GameState.init ∉ C.chainLayer h k := by
  intro hmem
  have hbag := C.chainLayer_bag_card h k hmem
  have hinit : GameState.init.bag.card = 7 := by change Bag.full.card = 7; decide
  rw [hinit] at hbag
  have hk : k.val < 6 := k.isLt
  omega

/-- Distinct chain depths give disjoint layers (uniform bag.card distinguishes). -/
theorem chainLayer_disjoint {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) {k₁ k₂ : Fin 6} (hne : k₁ ≠ k₂) :
    Disjoint (C.chainLayer h k₁) (C.chainLayer h k₂) := by
  rw [Finset.disjoint_left]
  intro x hx1 hx2
  have h1 := C.chainLayer_bag_card h k₁ hx1
  have h2 := C.chainLayer_bag_card h k₂ hx2
  have hk1 : k₁.val < 6 := k₁.isLt
  have hk2 : k₂.val < 6 := k₂.isLt
  have hv : k₁.val = k₂.val := by omega
  exact hne (Fin.ext hv)

/-- **Depth-6 chain bound: 28.** Any cycle containing `init` has at least 28
states. Assembled from `{init}` (size 1) and the six `chainLayer`s (sizes
7 + 6 + 5 + 4 + 3 + 2 = 27), all pairwise disjoint by bag.card. -/
theorem init_states_card_ge_twenty_eight {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    28 ≤ C.states.card := by
  -- The pairwise-disjoint biUnion of layers
  have hpair : (SetLike.coe (Finset.univ : Finset (Fin 6))).PairwiseDisjoint (C.chainLayer h) := by
    intro k₁ _ k₂ _ hne
    exact C.chainLayer_disjoint h hne
  let BU := (Finset.univ : Finset (Fin 6)).biUnion (C.chainLayer h)
  -- BU.card = 27
  have hBU_card : BU.card = 27 := by
    change ((Finset.univ : Finset (Fin 6)).biUnion (C.chainLayer h)).card = 27
    rw [Finset.card_biUnion hpair,
        Finset.sum_congr rfl (fun k _ => C.chainLayer_card h k)]
    decide
  -- init ∉ BU
  have hinit_notin : GameState.init ∉ BU := by
    intro hmem
    rcases Finset.mem_biUnion.mp hmem with ⟨k, _, hk⟩
    exact C.init_notMem_chainLayer h k hk
  -- Subset
  have hsub : insert GameState.init BU ⊆ C.states := by
    intro x hx
    rcases Finset.mem_insert.mp hx with rfl | hx
    · exact h
    · rcases Finset.mem_biUnion.mp hx with ⟨k, _, hk⟩
      exact C.chainLayer_subset h k hk
  -- Card
  have hS_card : (insert GameState.init BU).card = 28 := by
    rw [Finset.card_insert_of_notMem hinit_notin, hBU_card]
  calc 28 = _ := hS_card.symm
    _ ≤ _ := Finset.card_le_card hsub

end AdversarialClosedCycle

/-- **Atlas-level depth-6 chain bound**: any closed atlas on `S`
containing `init` has `28 ≤ S.card`. Lifts the cycle's
`init_states_card_ge_twenty_eight` (line 4752) to the Atlas level
via `ofClosedAtlas` (cycle extraction preserves states
definitionally). -/
theorem Atlas.IsClosedOn.init_states_card_ge_twenty_eight {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) :
    28 ≤ S.card :=
  (AdversarialClosedCycle.ofClosedAtlas h).init_states_card_ge_twenty_eight hinit

namespace AdversarialClosedCycle

/-- **Re-indexed chain existence.** For any `n ∈ {1, …, 7}`, there's a cycle
state with bag.card = `n`. Direct corollary of `exists_descent_chain` with
`k = 7 - n`. -/
theorem exists_state_with_bag_card {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states)
    (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 7) :
    ∃ s ∈ C.states, s.bag.card = n := by
  have hk : 7 - n ≤ 6 := by omega
  obtain ⟨s, hs, hcard⟩ := C.exists_descent_chain h (7 - n) hk
  refine ⟨s, hs, ?_⟩
  rw [hcard]
  omega

/-- The bag *does* empty (and refill) at some point along a cycle: there
exists a state with bag.card = 1 (the singleton-bag state immediately
preceding a refill draw). -/
theorem exists_state_with_singleton_bag {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    ∃ s ∈ C.states, s.bag.card = 1 :=
  C.exists_state_with_bag_card h 1 (by decide) (by decide)

end AdversarialClosedCycle

/-- **Atlas-level bag-card existence**: for any closed atlas
containing init, every bag size `n ∈ {1, …, 7}` is realized at
some state of `S`. Lifts the cycle-level `exists_state_with_bag_card`
(line 4802) via `ofClosedAtlas` (cycle extraction preserves states
definitionally). -/
theorem Atlas.IsClosedOn.exists_state_with_bag_card {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 7) :
    ∃ s ∈ S, s.bag.card = n :=
  (AdversarialClosedCycle.ofClosedAtlas h).exists_state_with_bag_card
    hinit n h1 h2

/-- **Atlas-level singleton-bag existence**: any closed atlas
containing init contains a state with `bag.card = 1` (the
immediate-pre-refill state). Specialisation of Tick 1837 at
`n = 1`. -/
theorem Atlas.IsClosedOn.exists_state_with_singleton_bag {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) :
    ∃ s ∈ S, s.bag.card = 1 :=
  h.exists_state_with_bag_card hinit 1 (by decide) (by decide)

/-- **Atlas-level subset-of-safe**: every state in a closed-atlas
state set is safe. Lifts the cycle-level `subset_safe` via
`ofClosedAtlas` — the extracted cycle has the same states
definitionally. -/
theorem Atlas.IsClosedOn.subset_safe {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S) :
    (↑S : Set GameState) ⊆ safe cfg :=
  (AdversarialClosedCycle.ofClosedAtlas h).subset_safe

/-- **Atlas-level pointwise mem-safe**: every state in a closed-
atlas state set is individually safe. Pointwise form of Tick 1839's
`subset_safe`. -/
theorem Atlas.IsClosedOn.mem_safe {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    {g : GameState} (hg : g ∈ S) : g ∈ safe cfg :=
  h.subset_safe (Finset.mem_coe.mpr hg)

/-- **Atlas-level init safety**: if a closed atlas contains `init`,
then `init ∈ safe cfg`. Specialisation of Tick 1840's `mem_safe`
at `g = init`. Atlas-level parallel to cycle's
`init_safe_of_init_mem`. -/
theorem Atlas.IsClosedOn.init_safe_of_init_mem {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) :
    GameState.init ∈ safe cfg :=
  h.mem_safe hinit

/-- **Atlas-level safe-set nonemptiness witness**: a nonempty
closed-atlas state set witnesses `(safe cfg).Nonempty`. One-liner
from Tick 1840 (`mem_safe`) — any state in S is safe. -/
theorem Atlas.IsClosedOn.safe_nonempty {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hne : S.Nonempty) :
    (safe cfg).Nonempty :=
  ⟨hne.choose, h.mem_safe hne.choose_spec⟩

/-- **Atlas-level canon `none` ↔ inactive (for closed atlases)**:
for any closed atlas `(h : A.IsClosedOn cfg S)`, `A.canon S g p =
none ↔ ¬(g ∈ S ∧ p ∈ g.bag)`. Atlas-side parallel of cycle's
`solver_toAtlas_canon_eq_none_iff_inactive` (line ~1260) and M4's
`canon_atlas_eq_none_iff_inactive` (line ~2242). Uses closure
totality at active queries to rule out `none`. -/
theorem Atlas.IsClosedOn.canon_eq_none_iff_inactive
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g : GameState} {p : Piece} :
    (A.canon S) g p = none ↔ ¬(g ∈ S ∧ p ∈ g.bag) := by
  refine ⟨fun hn hac => ?_, fun hi => Atlas.canon_apply_of_inactive _ hi⟩
  obtain ⟨hg, hp⟩ := hac
  rw [Atlas.canon_apply_of_active _ hg hp] at hn
  have htotal := h.total g hg p hp
  rw [Option.isSome_iff_ne_none] at htotal
  exact htotal hn

/-- **Atlas-level canon `isSome` ↔ active (for closed atlases)**:
contrapositive of `canon_eq_none_iff_inactive`. -/
theorem Atlas.IsClosedOn.canon_isSome_iff_active
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g : GameState} {p : Piece} :
    ((A.canon S) g p).isSome ↔ g ∈ S ∧ p ∈ g.bag := by
  rw [Option.isSome_iff_ne_none, ne_eq,
      h.canon_eq_none_iff_inactive, not_not]

/-- **Atlas-level trace range ⊆ safe (from-mem)**: from any
`g₀ ∈ S` and legal sequence, the materialised-solver trace range
is contained in `safe cfg`. Generalises Tick 1848 from `g₀ = init`
to arbitrary `g₀ ∈ S`. Composes Tick 1824's range-subset (trace
⊆ S) with Tick 1839's `subset_safe` (S ⊆ safe). -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_range_subset_safe_from_mem
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    Set.range (adversarialTrace cfg A.toSolver t g₀) ⊆ safe cfg :=
  fun _ hg =>
    h.subset_safe (h.toSolver_adversarialTrace_range_subset hg₀ hl hg)

/-- **Atlas-level per-step trace safety (from-mem)**: pointwise
form of Tick 1889. At each step `n`, the materialised-solver
trace from `g₀ ∈ S` lies in `safe cfg`. Composes Tick 1821's
trace-mem with Tick 1840's `mem_safe`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_safe_from_mem
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    adversarialTrace cfg A.toSolver t g₀ n ∈ safe cfg :=
  h.mem_safe (h.toSolver_adversarialTrace_mem hg₀ hl n)

/-- **Atlas-level per-step trace safety (from-init)**: pointwise
init-specialisation of Tick 1890. Useful for proofs that start
from init and need each step's safety. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_safe_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg A.toSolver s GameState.init n ∈ safe cfg :=
  h.toSolver_adversarialTrace_safe_from_mem hinit hl n

/-- **Atlas-level `solverReachable` subset of safe**: every
solverReachable state under the materialised solver is in `safe
cfg` for init-containing closed atlases. One-liner composing Tick
1895 (solverReachable ⊆ S) with Tick 1840 (S ⊆ safe). -/
theorem Atlas.IsClosedOn.toSolver_solverReachable_subset_safe
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {g : GameState} (hg : solverReachable A.toSolver g) :
    g ∈ safe cfg :=
  h.mem_safe (h.toSolver_solverReachable_mem hinit hg)

/-- **Atlas-level `solverReachable` ⇒ `Reachable`**: every
`solverReachable A.toSolver g` is `Reachable cfg g` for an
init-containing closed atlas. Goes through the extracted cycle:
since `(ofClosedAtlas h).solver = A.toSolver` definitionally and
`(ofClosedAtlas h).states = S`, the cycle-side
`solverReachable_reachable` (Tick 1915) applies directly. Atlas-side
parallel to the cycle-side Tick 1915. -/
theorem Atlas.IsClosedOn.toSolver_solverReachable_reachable
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {g : GameState} (hg : solverReachable A.toSolver g) :
    Reachable cfg g :=
  (AdversarialClosedCycle.ofClosedAtlas h).solverReachable_reachable hinit hg

/-- **Atlas-level 5-conjunct bundled `solverReachable` invariants
with `Reachable`**: combines mem-S (Tick 1895), not-lost (Tick 1898),
subset-safe (Tick 1897 atlas-form), bag-nonempty (solver-agnostic),
and Reachable (Tick 1918) into one destructure. Atlas-side parallel
of the M4-side `toSolver_solverReachable_invariants_with_reachable`
(Tick 1920) and the cycle-side
`solverReachable_invariants_full_with_reachable` (Tick 1916,
modulo per-piece valid). -/
theorem Atlas.IsClosedOn.toSolver_solverReachable_invariants_with_reachable
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {g : GameState} (hg : solverReachable A.toSolver g) :
    g ∈ S ∧ ¬ g.lost cfg ∧ g ∈ safe cfg ∧ g.bag.Nonempty ∧
      Reachable cfg g :=
  ⟨h.toSolver_solverReachable_mem hinit hg,
   h.toSolver_solverReachable_not_lost hinit hg,
   h.toSolver_solverReachable_subset_safe hinit hg,
   solverReachable_bag_nonempty hg,
   h.toSolver_solverReachable_reachable hinit hg⟩

/-- **Atlas-level one-step closure of the 5-conjunct invariant
bundle**: from a `solverReachable` state `g` and any piece `p ∈
g.bag`, the next state `adversarialStep cfg g p (A.toSolver g p)`
also satisfies the 5-conjunct bundle (∈ S ∧ ¬lost ∧ ∈ safe ∧
bag-nonempty ∧ Reachable). Atlas-side parallel to cycle's Tick
1910 (`solverReachable_step_invariants_full`). Proof: one-liner —
apply Tick 1921 to `solverReachable.step p hg hp`. -/
theorem Atlas.IsClosedOn.toSolver_solverReachable_step_invariants
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {g : GameState} (hg : solverReachable A.toSolver g)
    {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (A.toSolver g p) ∈ S ∧
    ¬ (adversarialStep cfg g p (A.toSolver g p)).lost cfg ∧
    adversarialStep cfg g p (A.toSolver g p) ∈ safe cfg ∧
    (adversarialStep cfg g p (A.toSolver g p)).bag.Nonempty ∧
    Reachable cfg (adversarialStep cfg g p (A.toSolver g p)) :=
  h.toSolver_solverReachable_invariants_with_reachable hinit
    (solverReachable.step p hg hp)

/-- **Atlas-level 6-conjunct one-step closure with per-piece
validity**: extends Tick 1930 with `∀ q ∈ next.bag, (A.toSolver
next q).piece = q ∧ (A.toSolver next q).Valid cfg`. Proof: bundle
Tick 1930 with Atlas-level `toSolver_valid_and_piece` applied at
the next state (whose `∈ S` membership comes from Tick 1930's
first conjunct). -/
theorem Atlas.IsClosedOn.toSolver_solverReachable_step_full_invariants
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {g : GameState} (hg : solverReachable A.toSolver g)
    {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (A.toSolver g p) ∈ S ∧
    ¬ (adversarialStep cfg g p (A.toSolver g p)).lost cfg ∧
    adversarialStep cfg g p (A.toSolver g p) ∈ safe cfg ∧
    (adversarialStep cfg g p (A.toSolver g p)).bag.Nonempty ∧
    Reachable cfg (adversarialStep cfg g p (A.toSolver g p)) ∧
    ∀ q ∈ (adversarialStep cfg g p (A.toSolver g p)).bag,
      (A.toSolver (adversarialStep cfg g p (A.toSolver g p)) q).piece = q ∧
      (A.toSolver (adversarialStep cfg g p (A.toSolver g p)) q).Valid cfg :=
  let h5 := h.toSolver_solverReachable_step_invariants hinit hg hp
  ⟨h5.1, h5.2.1, h5.2.2.1, h5.2.2.2.1, h5.2.2.2.2,
   fun _ hq => h.toSolver_valid_and_piece h5.1 hq⟩

/-- **M4 artifact's materialised-solver `solverReachable` states
are safe**: every `solverReachable M.atlas.toSolver g` lies in
`safe cfg`. M4-level lift of Tick 1897. -/
theorem AdversarialClosedCycle.M4Artifact.toSolver_solverReachable_subset_safe
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg)
    {g : GameState} (hg : solverReachable M.atlas.toSolver g) :
    g ∈ safe cfg :=
  M.closed.toSolver_solverReachable_subset_safe M.init_mem hg

/-- **M4 artifact's materialised-solver `solverReachable` ⇒
`Reachable`**: every `solverReachable M.atlas.toSolver g` is
`Reachable cfg g`. M4-level lift of Tick 1918 (atlas-side
`toSolver_solverReachable_reachable`). -/
theorem AdversarialClosedCycle.M4Artifact.toSolver_solverReachable_reachable
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg)
    {g : GameState} (hg : solverReachable M.atlas.toSolver g) :
    Reachable cfg g :=
  M.closed.toSolver_solverReachable_reachable M.init_mem hg

/-- **M4Artifact-level 5-conjunct bundled `solverReachable`
invariants with `Reachable`**: combines mem-states (Tick 1899),
not-lost (Tick 1900), subset-safe (Tick 1901), bag-nonempty (general
solver-agnostic), and Reachable (Tick 1919) into a single
destructure. M4-level parallel of the cycle-side
`solverReachable_invariants_full_with_reachable` (Tick 1916)
modulo the per-piece `(C.solver g p).Valid` conjunct (which has no
direct M4-level analog because `M.atlas.toSolver`'s validity needs
to route through `M.closed.valid`). -/
theorem AdversarialClosedCycle.M4Artifact.toSolver_solverReachable_invariants_with_reachable
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg)
    {g : GameState} (hg : solverReachable M.atlas.toSolver g) :
    g ∈ M.states ∧ ¬ g.lost cfg ∧ g ∈ safe cfg ∧ g.bag.Nonempty ∧
      Reachable cfg g :=
  ⟨M.toSolver_solverReachable_mem hg,
   M.toSolver_solverReachable_not_lost hg,
   M.toSolver_solverReachable_subset_safe hg,
   solverReachable_bag_nonempty hg,
   M.toSolver_solverReachable_reachable hg⟩

/-- **M4-level one-step closure of the 5-conjunct invariant
bundle**: M4-level lift of Tick 1930 (atlas form). From a
`solverReachable` state `g` and any piece `p ∈ g.bag`, the next
state satisfies the 5-conjunct bundle. Proof: one-liner via Tick
1930 + `M.init_mem`. -/
theorem AdversarialClosedCycle.M4Artifact.toSolver_solverReachable_step_invariants
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg)
    {g : GameState} (hg : solverReachable M.atlas.toSolver g)
    {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (M.atlas.toSolver g p) ∈ M.states ∧
    ¬ (adversarialStep cfg g p (M.atlas.toSolver g p)).lost cfg ∧
    adversarialStep cfg g p (M.atlas.toSolver g p) ∈ safe cfg ∧
    (adversarialStep cfg g p (M.atlas.toSolver g p)).bag.Nonempty ∧
    Reachable cfg (adversarialStep cfg g p (M.atlas.toSolver g p)) :=
  M.closed.toSolver_solverReachable_step_invariants M.init_mem hg hp

/-- **M4-level 6-conjunct one-step closure with per-piece
validity**: M4-level lift of Tick 1932 (atlas form). Extends
Tick 1931's 5-conjunct with `∀ q ∈ next.bag, (M.atlas.toSolver
next q).piece = q ∧ ... Valid cfg`. Proof: one-liner via Tick
1932 + `M.init_mem`. -/
theorem AdversarialClosedCycle.M4Artifact.toSolver_solverReachable_step_full_invariants
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg)
    {g : GameState} (hg : solverReachable M.atlas.toSolver g)
    {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (M.atlas.toSolver g p) ∈ M.states ∧
    ¬ (adversarialStep cfg g p (M.atlas.toSolver g p)).lost cfg ∧
    adversarialStep cfg g p (M.atlas.toSolver g p) ∈ safe cfg ∧
    (adversarialStep cfg g p (M.atlas.toSolver g p)).bag.Nonempty ∧
    Reachable cfg (adversarialStep cfg g p (M.atlas.toSolver g p)) ∧
    ∀ q ∈ (adversarialStep cfg g p (M.atlas.toSolver g p)).bag,
      (M.atlas.toSolver (adversarialStep cfg g p (M.atlas.toSolver g p)) q).piece = q ∧
      (M.atlas.toSolver (adversarialStep cfg g p (M.atlas.toSolver g p)) q).Valid cfg :=
  M.closed.toSolver_solverReachable_step_full_invariants M.init_mem hg hp

/-- **Atlas-level single-state impossibility**: any closed atlas
on `S` containing a state with non-empty bag must have `2 ≤ S.card`.
Lifts the cycle-level `states_card_ge_two` via `ofClosedAtlas`. -/
theorem Atlas.IsClosedOn.states_card_ge_two {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    {s : GameState} (hs : s ∈ S) {p : Piece} (hp : p ∈ s.bag) :
    2 ≤ S.card :=
  (AdversarialClosedCycle.ofClosedAtlas h).states_card_ge_two hs hp

/-- **Atlas-level init cardinality lower bound**: any closed atlas
containing `init` has `2 ≤ S.card`. Specialisation of Tick 1842
at `s = init, p = Piece.O` (using `init.bag = Bag.full`). -/
theorem Atlas.IsClosedOn.init_states_card_ge_two {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) :
    2 ≤ S.card :=
  h.states_card_ge_two hinit (GameState.init_bag.symm ▸ Bag.mem_full Piece.O)

/-- **M4 artifact has at least two states**: `2 ≤ M.states.card`.
Unconditional M4-level lift of `Atlas.IsClosedOn.init_states_card_ge_two`
(no hypothesis needed since `M.init_mem` is structural). -/
theorem AdversarialClosedCycle.M4Artifact.states_card_ge_two {cfg : GameConfig}
    (M : AdversarialClosedCycle.M4Artifact cfg) : 2 ≤ M.states.card :=
  M.closed.init_states_card_ge_two M.init_mem

/-- **Atlas-level bag-card lower bound**: any closed atlas
containing a state `s` has `s.bag.card ≤ S.card`. Lifts the
cycle-level `states_card_ge_bag_card` (line ~4810) via
`ofClosedAtlas`. -/
theorem Atlas.IsClosedOn.states_card_ge_bag_card {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    {s : GameState} (hs : s ∈ S) :
    s.bag.card ≤ S.card :=
  (AdversarialClosedCycle.ofClosedAtlas h).states_card_ge_bag_card hs

/-- **M4-level bag-card lower bound**: `(M : M4Artifact cfg) →
s ∈ M.states → s.bag.card ≤ M.states.card`. M4-level lift of
the atlas-form. -/
theorem AdversarialClosedCycle.M4Artifact.states_card_ge_bag_card
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg)
    {s : GameState} (hs : s ∈ M.states) :
    s.bag.card ≤ M.states.card :=
  M.closed.states_card_ge_bag_card hs

/-- **Atlas-level strict bag-card lower bound**: any closed atlas
containing `s` with nonempty bag has `s.bag.card + 1 ≤ S.card`.
Lifts the cycle-level `states_card_gt_bag_card` (line ~4850) via
`ofClosedAtlas`. Strict tightening of Tick 1940's atlas form. -/
theorem Atlas.IsClosedOn.states_card_gt_bag_card {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    {s : GameState} (hs : s ∈ S) (hne : s.bag.Nonempty) :
    s.bag.card + 1 ≤ S.card :=
  (AdversarialClosedCycle.ofClosedAtlas h).states_card_gt_bag_card hs hne

/-- **M4-level strict bag-card lower bound**: `(M : M4Artifact cfg)
→ s ∈ M.states → s.bag.Nonempty → s.bag.card + 1 ≤ M.states.card`.
M4-level lift of the atlas form. -/
theorem AdversarialClosedCycle.M4Artifact.states_card_gt_bag_card
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg)
    {s : GameState} (hs : s ∈ M.states) (hne : s.bag.Nonempty) :
    s.bag.card + 1 ≤ M.states.card :=
  M.closed.states_card_gt_bag_card hs hne

/-- **M4 artifact has at least 8 states**: `8 ≤ M.states.card`.
M4-level init specialization of `states_card_gt_bag_card` at
`s = init` (using `init.bag = Bag.full`, `init.bag.card = 7`).
Parallel to cycle's `init_states_card_ge_eight_via_bag` (line ~4884). -/
theorem AdversarialClosedCycle.M4Artifact.states_card_ge_eight
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    8 ≤ M.states.card := by
  have h := M.states_card_gt_bag_card M.init_mem
    (GameState.init_bag.symm ▸ Bag.full_nonempty)
  rw [GameState.init_bag_card] at h
  exact h

/-- **Atlas-level init cardinality lower bound (≥ 8)**: any closed
atlas containing `init` has `8 ≤ S.card`. Lifts the cycle-level
`init_states_card_ge_eight_via_bag` (line ~4884) via `ofClosedAtlas`.
Atlas-side parallel to the M4-side `states_card_ge_eight` (Tick 1941). -/
theorem Atlas.IsClosedOn.init_states_card_ge_eight {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) : 8 ≤ S.card :=
  (AdversarialClosedCycle.ofClosedAtlas h).init_states_card_ge_eight_via_bag
    hinit

/-- **Atlas-level cardinality weakenings**: any closed atlas
containing `init` has `S.card ≠ k` for `k ∈ {2,...,7}`. Six
parallel theorems, each one-liner via `omega` on Tick 1943's
`≥ 8` bound. Parallel to cycle's `init_states_card_ne_*` family
(lines ~4915-4945) and M4's Tick 1942 family. -/
theorem Atlas.IsClosedOn.init_states_card_ne_two {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) : S.card ≠ 2 := by
  have := h.init_states_card_ge_eight hinit; omega

theorem Atlas.IsClosedOn.init_states_card_ne_three {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) : S.card ≠ 3 := by
  have := h.init_states_card_ge_eight hinit; omega

theorem Atlas.IsClosedOn.init_states_card_ne_four {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) : S.card ≠ 4 := by
  have := h.init_states_card_ge_eight hinit; omega

theorem Atlas.IsClosedOn.init_states_card_ne_five {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) : S.card ≠ 5 := by
  have := h.init_states_card_ge_eight hinit; omega

theorem Atlas.IsClosedOn.init_states_card_ne_six {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) : S.card ≠ 6 := by
  have := h.init_states_card_ge_eight hinit; omega

theorem Atlas.IsClosedOn.init_states_card_ne_seven {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) : S.card ≠ 7 := by
  have := h.init_states_card_ge_eight hinit; omega

/-- **Atlas-level cardinality lower-bound weakenings**: any closed
atlas containing `init` has `k ≤ S.card` for `k ∈ {3,...,7}`. Five
parallel theorems, each one-liner via `Nat.le_trans` on Tick 1943's
`≥ 8` bound. Parallel to cycle's `init_states_card_ge_{three..seven}`
family (lines ~4895+). Useful when only a specific weak bound is
needed (avoid pulling in the full `≥ 8` fact). -/
theorem Atlas.IsClosedOn.init_states_card_ge_three {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) : 3 ≤ S.card :=
  Nat.le_trans (by decide : (3 : ℕ) ≤ 8) (h.init_states_card_ge_eight hinit)

theorem Atlas.IsClosedOn.init_states_card_ge_four {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) : 4 ≤ S.card :=
  Nat.le_trans (by decide : (4 : ℕ) ≤ 8) (h.init_states_card_ge_eight hinit)

theorem Atlas.IsClosedOn.init_states_card_ge_five {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) : 5 ≤ S.card :=
  Nat.le_trans (by decide : (5 : ℕ) ≤ 8) (h.init_states_card_ge_eight hinit)

theorem Atlas.IsClosedOn.init_states_card_ge_six {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) : 6 ≤ S.card :=
  Nat.le_trans (by decide : (6 : ℕ) ≤ 8) (h.init_states_card_ge_eight hinit)

theorem Atlas.IsClosedOn.init_states_card_ge_seven {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) : 7 ≤ S.card :=
  Nat.le_trans (by decide : (7 : ℕ) ≤ 8) (h.init_states_card_ge_eight hinit)

/-- **M4 artifact states.card ≠ k for k ∈ {2,...,7}**: weakenings of
`states_card_ge_eight` (Tick 1941) via `omega`. Parallel to cycle's
`init_states_card_ne_*` family (lines ~4915-4945). Unconditional at
M4 level (no init hypothesis needed). -/
theorem AdversarialClosedCycle.M4Artifact.states_card_ne_two
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    M.states.card ≠ 2 := by
  have := M.states_card_ge_eight; omega

theorem AdversarialClosedCycle.M4Artifact.states_card_ne_three
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    M.states.card ≠ 3 := by
  have := M.states_card_ge_eight; omega

theorem AdversarialClosedCycle.M4Artifact.states_card_ne_four
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    M.states.card ≠ 4 := by
  have := M.states_card_ge_eight; omega

theorem AdversarialClosedCycle.M4Artifact.states_card_ne_five
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    M.states.card ≠ 5 := by
  have := M.states_card_ge_eight; omega

theorem AdversarialClosedCycle.M4Artifact.states_card_ne_six
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    M.states.card ≠ 6 := by
  have := M.states_card_ge_eight; omega

theorem AdversarialClosedCycle.M4Artifact.states_card_ne_seven
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    M.states.card ≠ 7 := by
  have := M.states_card_ge_eight; omega

/-- **M4 cardinality lower-bound weakenings `≥ {3,4,5,6,7}`**:
five theorems each `k ≤ M.states.card` for `k ∈ {3,...,7}`. Each
is a one-liner via `Nat.le_trans` on Tick 1941's `≥ 8` bound.
M4-side parallel to atlas's Tick 1945 and cycle's existing family. -/
theorem AdversarialClosedCycle.M4Artifact.states_card_ge_three
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    3 ≤ M.states.card :=
  Nat.le_trans (by decide : (3 : ℕ) ≤ 8) M.states_card_ge_eight

theorem AdversarialClosedCycle.M4Artifact.states_card_ge_four
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    4 ≤ M.states.card :=
  Nat.le_trans (by decide : (4 : ℕ) ≤ 8) M.states_card_ge_eight

theorem AdversarialClosedCycle.M4Artifact.states_card_ge_five
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    5 ≤ M.states.card :=
  Nat.le_trans (by decide : (5 : ℕ) ≤ 8) M.states_card_ge_eight

theorem AdversarialClosedCycle.M4Artifact.states_card_ge_six
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    6 ≤ M.states.card :=
  Nat.le_trans (by decide : (6 : ℕ) ≤ 8) M.states_card_ge_eight

theorem AdversarialClosedCycle.M4Artifact.states_card_ge_seven
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    7 ≤ M.states.card :=
  Nat.le_trans (by decide : (7 : ℕ) ≤ 8) M.states_card_ge_eight

/-- **Atlas-level `solverReachable` set ncard bound**: for any
init-containing closed atlas, the ncard of `{g | solverReachable
A.toSolver g}` is bounded by `S.card`. Lifts Tick 1947 via
`ofClosedAtlas` (whose solver is `A.toSolver` and states is `S`,
both definitionally). -/
theorem Atlas.IsClosedOn.toSolver_solverReachable_ncard_le_states_card
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    {g | solverReachable A.toSolver g}.ncard ≤ S.card :=
  (AdversarialClosedCycle.ofClosedAtlas h).solverReachable_ncard_le_states_card
    hinit

/-- **M4-level `solverReachable` set ncard bound**: M4-level lift
of the atlas form. Unconditional (no init hypothesis needed since
`M.init_mem` is structural). -/
theorem AdversarialClosedCycle.M4Artifact.toSolver_solverReachable_ncard_le_states_card
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    {g | solverReachable M.atlas.toSolver g}.ncard ≤ M.states.card :=
  M.closed.toSolver_solverReachable_ncard_le_states_card M.init_mem

/-- **Atlas-level `solverReachable` set Finset cardinality bound**:
the Finset cardinality of `(toSolver_solverReachable_finite).toFinset`
is bounded by `S.card`. Atlas-side parallel to Tick 1949
(cycle-level Finset cardinality bound). -/
theorem Atlas.IsClosedOn.toSolver_solverReachable_toFinset_card_le_states_card
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    (h.toSolver_solverReachable_finite hinit).toFinset.card ≤ S.card := by
  apply Finset.card_le_card
  intro g hg
  rw [Set.Finite.mem_toFinset] at hg
  exact h.toSolver_solverReachable_mem hinit hg

/-- **M4-level `solverReachable` set Finset cardinality bound**:
M4-level lift of the atlas form. Unconditional. -/
theorem AdversarialClosedCycle.M4Artifact.toSolver_solverReachable_toFinset_card_le_states_card
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    M.toSolver_solverReachable_finite.toFinset.card ≤ M.states.card :=
  M.closed.toSolver_solverReachable_toFinset_card_le_states_card M.init_mem

/-- **Atlas-level `solverReachable` Set ⊆ `↑S`**: Set-form of the
atlas's `toSolver_solverReachable_mem`. Atlas-side parallel of cycle
Tick 1951. -/
theorem Atlas.IsClosedOn.toSolver_solverReachable_subset_states
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    {g | solverReachable A.toSolver g} ⊆ ↑S :=
  fun _ hg => Finset.mem_coe.mpr (h.toSolver_solverReachable_mem hinit hg)

/-- **M4-level `solverReachable` Set ⊆ `↑M.states`**: M4-level lift
of the atlas form. -/
theorem AdversarialClosedCycle.M4Artifact.toSolver_solverReachable_subset_states
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    {g | solverReachable M.atlas.toSolver g} ⊆ ↑M.states :=
  M.closed.toSolver_solverReachable_subset_states M.init_mem

/-- **Atlas-level `solverReachable` Set ⊆ `safe cfg`**: Set-form
of the atlas's pointwise `toSolver_solverReachable_subset_safe`
(line ~5468). Atlas-side parallel of cycle Tick 1952. -/
theorem Atlas.IsClosedOn.toSolver_solverReachable_set_subset_safe
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    {g | solverReachable A.toSolver g} ⊆ safe cfg :=
  fun _ hg => h.toSolver_solverReachable_subset_safe hinit hg

/-- **M4-level `solverReachable` Set ⊆ `safe cfg`**: M4-level lift
of the atlas form. -/
theorem AdversarialClosedCycle.M4Artifact.toSolver_solverReachable_set_subset_safe
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    {g | solverReachable M.atlas.toSolver g} ⊆ safe cfg :=
  M.closed.toSolver_solverReachable_set_subset_safe M.init_mem

/-- **Atlas-level `solverReachable` Set ⊆ `{g | Reachable cfg g}`**:
Set-form of Tick 1918 (`toSolver_solverReachable_reachable`). -/
theorem Atlas.IsClosedOn.toSolver_solverReachable_subset_Reachable
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    {g | solverReachable A.toSolver g} ⊆ {g | Reachable cfg g} :=
  fun _ hg => h.toSolver_solverReachable_reachable hinit hg

/-- **M4-level `solverReachable` Set ⊆ `{g | Reachable cfg g}`**:
Set-form of Tick 1919. M4-level lift; no hypothesis needed. -/
theorem AdversarialClosedCycle.M4Artifact.toSolver_solverReachable_subset_Reachable
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    {g | solverReachable M.atlas.toSolver g} ⊆ {g | Reachable cfg g} :=
  fun _ hg => M.toSolver_solverReachable_reachable hg

/-- **Atlas-level joint intersection Set-form**: atlas-side parallel
to cycle Tick 1954. Combines `⊆ S`, `⊆ safe cfg`, and `⊆ Reachable`
into a single intersection inclusion. -/
theorem Atlas.IsClosedOn.toSolver_solverReachable_subset_inter
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    {g | solverReachable A.toSolver g} ⊆
      ↑S ∩ safe cfg ∩ {g | Reachable cfg g} := by
  intro g hg
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · exact h.toSolver_solverReachable_subset_states hinit hg
  · exact h.toSolver_solverReachable_set_subset_safe hinit hg
  · exact h.toSolver_solverReachable_subset_Reachable hinit hg

/-- **M4-level joint intersection Set-form**: M4-level lift of the
atlas form. Unconditional. -/
theorem AdversarialClosedCycle.M4Artifact.toSolver_solverReachable_subset_inter
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg) :
    {g | solverReachable M.atlas.toSolver g} ⊆
      ↑M.states ∩ safe cfg ∩ {g | Reachable cfg g} :=
  M.closed.toSolver_solverReachable_subset_inter M.init_mem

/-- **Atlas-level trace reachable from init**: every state in the
materialised-solver trace from init is reachable from init. Lifts
the cycle-level `adversarialTrace_reachable_of_init` via
`ofClosedAtlas` — the cycle's solver is `A.toSolver` definitionally,
so the trace under either solver is the same. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_reachable_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Reachable cfg (adversarialTrace cfg A.toSolver s GameState.init n) :=
  (AdversarialClosedCycle.ofClosedAtlas h).adversarialTrace_reachable_of_init
    hinit hl n

/-- **Atlas-level trace solverReachable from init**: every state
in the materialised-solver trace from init is `solverReachable`
under the materialised solver. Note: no `init ∈ S` hypothesis
needed — purely a `solverReachable` closure fact under
`adversarialStep`. Lifts cycle's
`adversarialTrace_solverReachable_of_init`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_solverReachable_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {s : ℕ → Piece} (hl : LegalSequence s)
    (n : ℕ) :
    solverReachable A.toSolver
      (adversarialTrace cfg A.toSolver s GameState.init n) :=
  (AdversarialClosedCycle.ofClosedAtlas h).adversarialTrace_solverReachable_of_init
    hl n

/-- **Atlas-level trace 5-conjunct bundled invariants with
`Reachable` from init**: trace-of-init parallel to Tick 1921.
For any init-containing closed atlas and `LegalSequence`, every
step of the materialised-solver trace satisfies the 5-conjunct
bundle (mem-S ∧ not-lost ∧ safe ∧ bag-nonempty ∧ Reachable).
Proof: apply Tick 1921 to the trace's solverReachability
(`toSolver_adversarialTrace_solverReachable_of_init`). -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_invariants_with_reachable_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg A.toSolver s GameState.init n ∈ S ∧
    ¬ (adversarialTrace cfg A.toSolver s GameState.init n).lost cfg ∧
    adversarialTrace cfg A.toSolver s GameState.init n ∈ safe cfg ∧
    (adversarialTrace cfg A.toSolver s GameState.init n).bag.Nonempty ∧
    Reachable cfg (adversarialTrace cfg A.toSolver s GameState.init n) :=
  h.toSolver_solverReachable_invariants_with_reachable hinit
    (h.toSolver_adversarialTrace_solverReachable_of_init hl n)

/-- **Atlas-level trace 5-conjunct bundled invariants with
`Reachable` from a `solverReachable` start**: generalizes Tick 1922
from `g₀ = init` to any solverReachable g₀ start. Atlas-side
parallel to cycle's Tick 1917
(`adversarialTrace_full_invariants_with_reachable_from_solverReachable`).
Proof: one-liner — apply Tick 1921 to
`adversarialTrace_solverReachable_from` (Tick 1911). -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_invariants_with_reachable_from_solverReachable
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {g₀ : GameState} (hg₀ : solverReachable A.toSolver g₀)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    adversarialTrace cfg A.toSolver t g₀ n ∈ S ∧
    ¬ (adversarialTrace cfg A.toSolver t g₀ n).lost cfg ∧
    adversarialTrace cfg A.toSolver t g₀ n ∈ safe cfg ∧
    (adversarialTrace cfg A.toSolver t g₀ n).bag.Nonempty ∧
    Reachable cfg (adversarialTrace cfg A.toSolver t g₀ n) :=
  h.toSolver_solverReachable_invariants_with_reachable hinit
    (adversarialTrace_solverReachable_from hg₀ hl n)

/-- **Atlas-level trace 6-conjunct full invariants with `Reachable`
and per-piece validity from a `solverReachable` start**: strongest
atlas-level trace-from-solverReachable bundle. Extends Tick 1924
with `∀ p ∈ trace_state.bag, (A.toSolver state p).piece = p ∧
(A.toSolver state p).Valid cfg` via
`Atlas.IsClosedOn.toSolver_valid_and_piece`. Atlas-side parallel of
the cycle-side Tick 1917 (which has per-piece-valid via `C.valid`). -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_full_invariants_from_solverReachable
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {g₀ : GameState} (hg₀ : solverReachable A.toSolver g₀)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    adversarialTrace cfg A.toSolver t g₀ n ∈ S ∧
    ¬ (adversarialTrace cfg A.toSolver t g₀ n).lost cfg ∧
    adversarialTrace cfg A.toSolver t g₀ n ∈ safe cfg ∧
    (adversarialTrace cfg A.toSolver t g₀ n).bag.Nonempty ∧
    Reachable cfg (adversarialTrace cfg A.toSolver t g₀ n) ∧
    ∀ p ∈ (adversarialTrace cfg A.toSolver t g₀ n).bag,
      (A.toSolver (adversarialTrace cfg A.toSolver t g₀ n) p).piece = p ∧
      (A.toSolver (adversarialTrace cfg A.toSolver t g₀ n) p).Valid cfg :=
  let h5 := h.toSolver_adversarialTrace_invariants_with_reachable_from_solverReachable
              hinit hg₀ hl n
  ⟨h5.1, h5.2.1, h5.2.2.1, h5.2.2.2.1, h5.2.2.2.2,
   fun _ hp => h.toSolver_valid_and_piece h5.1 hp⟩

/-- **Init specialisation of Tick 1926**: 6-conjunct atlas-level
full trace invariants from init. Start at `GameState.init`
(solverReachable trivially via `solverReachable.init`) with full
`LegalSequence`. Coerces `LegalSequence s` to `LegalSequenceFrom
init.bag s` definitionally via `init.bag = Bag.full`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_full_invariants_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg A.toSolver s GameState.init n ∈ S ∧
    ¬ (adversarialTrace cfg A.toSolver s GameState.init n).lost cfg ∧
    adversarialTrace cfg A.toSolver s GameState.init n ∈ safe cfg ∧
    (adversarialTrace cfg A.toSolver s GameState.init n).bag.Nonempty ∧
    Reachable cfg (adversarialTrace cfg A.toSolver s GameState.init n) ∧
    ∀ p ∈ (adversarialTrace cfg A.toSolver s GameState.init n).bag,
      (A.toSolver (adversarialTrace cfg A.toSolver s GameState.init n) p).piece = p ∧
      (A.toSolver (adversarialTrace cfg A.toSolver s GameState.init n) p).Valid cfg :=
  h.toSolver_adversarialTrace_full_invariants_from_solverReachable
    hinit solverReachable.init hl n

/-- **M4-level trace 5-conjunct bundled invariants with `Reachable`
from init**: M4-level lift of Tick 1922 (atlas form). For any M4
artifact and `LegalSequence`, every step of the materialised-solver
trace satisfies (∈ M.states ∧ ¬lost ∧ ∈ safe ∧ bag-nonempty ∧
Reachable). Extends the existing 4-conjunct
`M4Artifact.toSolver_trace_invariants_of_init` (line ~4274) with
`bag.Nonempty`. Proof: one-liner via Tick 1922 applied to `M.closed
+ M.init_mem`. -/
theorem AdversarialClosedCycle.M4Artifact.toSolver_trace_invariants_with_bag_of_init
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg M.atlas.toSolver s GameState.init n ∈ M.states ∧
    ¬ (adversarialTrace cfg M.atlas.toSolver s GameState.init n).lost cfg ∧
    adversarialTrace cfg M.atlas.toSolver s GameState.init n ∈ safe cfg ∧
    (adversarialTrace cfg M.atlas.toSolver s GameState.init n).bag.Nonempty ∧
    Reachable cfg (adversarialTrace cfg M.atlas.toSolver s GameState.init n) :=
  M.closed.toSolver_adversarialTrace_invariants_with_reachable_of_init
    M.init_mem hl n

namespace AdversarialClosedCycle.M4Artifact

/-- **M4-level trace 5-conjunct bundled invariants with `Reachable`
from a `solverReachable` start**: M4-level lift of Tick 1924 (atlas
form). Generalizes Tick 1923 from init to any solverReachable g₀
start. Proof: one-liner via Tick 1924 + `M.init_mem`. -/
theorem toSolver_trace_invariants_with_reachable_from_solverReachable
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg)
    {g₀ : GameState} (hg₀ : solverReachable M.atlas.toSolver g₀)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    adversarialTrace cfg M.atlas.toSolver t g₀ n ∈ M.states ∧
    ¬ (adversarialTrace cfg M.atlas.toSolver t g₀ n).lost cfg ∧
    adversarialTrace cfg M.atlas.toSolver t g₀ n ∈ safe cfg ∧
    (adversarialTrace cfg M.atlas.toSolver t g₀ n).bag.Nonempty ∧
    Reachable cfg (adversarialTrace cfg M.atlas.toSolver t g₀ n) :=
  M.closed.toSolver_adversarialTrace_invariants_with_reachable_from_solverReachable
    M.init_mem hg₀ hl n

end AdversarialClosedCycle.M4Artifact

/-- **M4-level trace 6-conjunct full invariants with `Reachable`
and per-piece validity from `solverReachable` start**: M4-level
lift of Tick 1926 (atlas form). Strongest M4-level trace
from-solverReachable bundle, adding `∀ p ∈ trace_state.bag,
(M.atlas.toSolver state p).piece = p ∧ ... Valid cfg` to Tick
1925's 5-conjunct. Proof: one-liner via Tick 1926. -/
theorem AdversarialClosedCycle.M4Artifact.toSolver_trace_full_invariants_from_solverReachable
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg)
    {g₀ : GameState} (hg₀ : solverReachable M.atlas.toSolver g₀)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    adversarialTrace cfg M.atlas.toSolver t g₀ n ∈ M.states ∧
    ¬ (adversarialTrace cfg M.atlas.toSolver t g₀ n).lost cfg ∧
    adversarialTrace cfg M.atlas.toSolver t g₀ n ∈ safe cfg ∧
    (adversarialTrace cfg M.atlas.toSolver t g₀ n).bag.Nonempty ∧
    Reachable cfg (adversarialTrace cfg M.atlas.toSolver t g₀ n) ∧
    ∀ p ∈ (adversarialTrace cfg M.atlas.toSolver t g₀ n).bag,
      (M.atlas.toSolver (adversarialTrace cfg M.atlas.toSolver t g₀ n) p).piece = p ∧
      (M.atlas.toSolver (adversarialTrace cfg M.atlas.toSolver t g₀ n) p).Valid cfg :=
  M.closed.toSolver_adversarialTrace_full_invariants_from_solverReachable
    M.init_mem hg₀ hl n

/-- **M4-level init specialisation of the 6-conjunct trace bundle
(universal-piece form)**: M4-level lift of Tick 1928 (atlas form).
Specialises Tick 1927 to `g₀ = init` (using `solverReachable.init`
and `init.bag = Bag.full`). Distinct from the existing
`M4Artifact.toSolver_trace_full_invariants_of_init` (line ~4400)
which is the *sequenced* form (piece = `s n`); this is the
*universal* form (`∀ p ∈ bag`). -/
theorem AdversarialClosedCycle.M4Artifact.toSolver_trace_full_invariants_univ_of_init
    {cfg : GameConfig} (M : AdversarialClosedCycle.M4Artifact cfg)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg M.atlas.toSolver s GameState.init n ∈ M.states ∧
    ¬ (adversarialTrace cfg M.atlas.toSolver s GameState.init n).lost cfg ∧
    adversarialTrace cfg M.atlas.toSolver s GameState.init n ∈ safe cfg ∧
    (adversarialTrace cfg M.atlas.toSolver s GameState.init n).bag.Nonempty ∧
    Reachable cfg (adversarialTrace cfg M.atlas.toSolver s GameState.init n) ∧
    ∀ p ∈ (adversarialTrace cfg M.atlas.toSolver s GameState.init n).bag,
      (M.atlas.toSolver
        (adversarialTrace cfg M.atlas.toSolver s GameState.init n) p).piece = p ∧
      (M.atlas.toSolver
        (adversarialTrace cfg M.atlas.toSolver s GameState.init n) p).Valid cfg :=
  M.closed.toSolver_adversarialTrace_full_invariants_of_init
    M.init_mem hl n

/-- **Atlas-level init-step solverReachable**: a single adversarial
step from init under the materialised solver lands in a
`solverReachable` state. Lifts cycle's `step_solverReachable_of_init`. -/
theorem Atlas.IsClosedOn.toSolver_step_solverReachable_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (p : Piece) :
    solverReachable A.toSolver
      (adversarialStep cfg GameState.init p (A.toSolver GameState.init p)) :=
  (AdversarialClosedCycle.ofClosedAtlas h).step_solverReachable_of_init p

/-- **Atlas-level trace range ⊆ solverReachable**: the full image
of the materialised-solver trace from init is contained in the
solverReachable set. `Set.range_subset_iff` form of Tick 1845.
Atlas-level parallel to cycle's
`adversarialTrace_range_subset_solverReachable_of_init`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_range_subset_solverReachable_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {s : ℕ → Piece} (hl : LegalSequence s) :
    Set.range (adversarialTrace cfg A.toSolver s GameState.init) ⊆
      {g | solverReachable A.toSolver g} :=
  Set.range_subset_iff.mpr
    (fun n => h.toSolver_adversarialTrace_solverReachable_of_init hl n)

/-- **Atlas-level trace range ⊆ safe**: the full image of the
materialised-solver trace from init is contained in `safe cfg`.
Lifts cycle's `adversarialTrace_range_subset_safe_of_init`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_range_subset_safe_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    Set.range (adversarialTrace cfg A.toSolver s GameState.init) ⊆ safe cfg :=
  (AdversarialClosedCycle.ofClosedAtlas h).adversarialTrace_range_subset_safe_of_init
    hinit hl

/-- **Atlas-level trace range ⊆ S**: the full image of the
materialised-solver trace from init is contained in S. Lifts
cycle's `adversarialTrace_range_subset_states_of_init`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_range_subset_states_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    Set.range (adversarialTrace cfg A.toSolver s GameState.init) ⊆ S :=
  (AdversarialClosedCycle.ofClosedAtlas h).adversarialTrace_range_subset_states_of_init
    hinit hl

/-- **Atlas-level trace range is finite**: the trace image from
init is finite (contained in the finite Finset S). Direct via
`S.finite_toSet.subset` applied to Tick 1849. Atlas-level parallel
to cycle's `adversarialTrace_range_finite_of_init`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_range_finite_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    (Set.range (adversarialTrace cfg A.toSolver s GameState.init)).Finite :=
  S.finite_toSet.subset
    (h.toSolver_adversarialTrace_range_subset_states_of_init hinit hl)

namespace AdversarialClosedCycle

/-- **Bundled obligations.** The four predicate-level requirements that turn
a `(Finset GameState × Solver cfg)` pair into an `AdversarialClosedCycle`.
Bundled as a single decidable `Prop` so cycle candidates can be machine
checked via `decide` (Roadmap sub-task 1.3). -/
def Obligations (cfg : GameConfig) (states : Finset GameState)
    (solver : Solver cfg) : Prop :=
  (∀ s ∈ states, ¬ s.lost cfg) ∧
  (∀ s ∈ states, ∀ p, p ∈ s.bag →
    (solver s p).piece = p ∧ (solver s p).Valid cfg) ∧
  (∀ s ∈ states, ∀ p, p ∈ s.bag →
    adversarialStep cfg s p (solver s p) ∈ states)

/-- Dot accessor: "not lost" clause of `Obligations`. -/
theorem Obligations.not_lost {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : Obligations cfg states solver)
    {g : GameState} (hg : g ∈ states) : ¬ g.lost cfg :=
  h.1 g hg

/-- Dot accessor: "solver valid" clause of `Obligations`. -/
theorem Obligations.solver_valid {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : Obligations cfg states solver)
    {g : GameState} (hg : g ∈ states) {p : Piece} (hp : p ∈ g.bag) :
    (solver g p).piece = p ∧ (solver g p).Valid cfg :=
  h.2.1 g hg p hp

/-- Dot accessor: "step closed" clause of `Obligations`. -/
theorem Obligations.step_mem {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : Obligations cfg states solver)
    {g : GameState} (hg : g ∈ states) {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (solver g p) ∈ states :=
  h.2.2 g hg p hp

/-- Piece half of `Obligations.solver_valid`. -/
theorem Obligations.solver_piece {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : Obligations cfg states solver)
    {g : GameState} (hg : g ∈ states) {p : Piece} (hp : p ∈ g.bag) :
    (solver g p).piece = p := (h.solver_valid hg hp).1

/-- Valid half of `Obligations.solver_valid`. -/
theorem Obligations.solver_placement_valid {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : Obligations cfg states solver)
    {g : GameState} (hg : g ∈ states) {p : Piece} (hp : p ∈ g.bag) :
    (solver g p).Valid cfg := (h.solver_valid hg hp).2

/-- Membership in `Placement.allValidFor` form. -/
theorem Obligations.solver_mem_allValidFor {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : Obligations cfg states solver)
    {g : GameState} (hg : g ∈ states) {p : Piece} (hp : p ∈ g.bag) :
    solver g p ∈ Placement.allValidFor cfg p :=
  (Placement.mem_allValidFor cfg p _).mpr (h.solver_valid hg hp)

/-- `GameState.step` form of `Obligations.step_mem` (using
`Placement.eta_of_piece_eq` to absorb the piece substitution). -/
theorem Obligations.step_mem_step {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : Obligations cfg states solver)
    {g : GameState} (hg : g ∈ states) {p : Piece} (hp : p ∈ g.bag) :
    g.step cfg (solver g p) ∈ states := by
  have hsm := h.step_mem hg hp
  rw [adversarialStep_eq_step, Placement.eta_of_piece_eq
        (h.solver_piece hg hp)] at hsm
  exact hsm


instance (cfg : GameConfig) (states : Finset GameState) (solver : Solver cfg) :
    Decidable (Obligations cfg states solver) := by
  unfold Obligations
  infer_instance

/-- Smart constructor from the bundled obligations. -/
def mkChecked {cfg : GameConfig} (states : Finset GameState) (solver : Solver cfg)
    (h : Obligations cfg states solver) : AdversarialClosedCycle cfg where
  states := states
  solver := solver
  not_lost := h.1
  valid := h.2.1
  closed := h.2.2

@[simp] theorem mkChecked_states {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : Obligations cfg states solver) :
    (mkChecked states solver h).states = states := rfl

@[simp] theorem mkChecked_solver {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : Obligations cfg states solver) :
    (mkChecked states solver h).solver = solver := rfl

/-- The reverse direction: any existing cycle satisfies the bundled
obligations. Together with `mkChecked`, this gives a tight equivalence
between cycles and `(states, solver, Obligations.proof)` triples. -/
theorem obligations_of {cfg : GameConfig} (C : AdversarialClosedCycle cfg) :
    Obligations cfg C.states C.solver :=
  ⟨C.not_lost, C.valid, C.closed⟩

/-- **Cycle-through-init existence is decidable** for any given `(states,
solver)` candidate: combine `Obligations` decidability with a check that
`init ∈ states`. Concretely: to prove `TetrisSolvableFor cfg` it suffices
to exhibit a `(states, solver)` candidate satisfying `Obligations` *and*
containing `init`, both decidable propositions. -/
theorem solvable_of_checked_cycle_through_init {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : Obligations cfg states solver) (hinit : GameState.init ∈ states) :
    TetrisSolvableFor cfg :=
  (mkChecked states solver h).init_solves_for (by
    show GameState.init ∈ (mkChecked states solver h).states
    rw [mkChecked_states]
    exact hinit)

/-- **Combined decidable predicate**: a candidate is an "init-cycle" iff it
satisfies all obligations *and* contains `init`. This is the single
decidable proposition whose discovery proves `TetrisSolvableFor cfg`. -/
def IsInitCycle (cfg : GameConfig) (states : Finset GameState)
    (solver : Solver cfg) : Prop :=
  Obligations cfg states solver ∧ GameState.init ∈ states

instance (cfg : GameConfig) (states : Finset GameState) (solver : Solver cfg) :
    Decidable (IsInitCycle cfg states solver) := by
  unfold IsInitCycle
  infer_instance

/-- The headline reduction in its tightest form: a single decidable
proposition implies `TetrisSolvableFor cfg`. -/
theorem solvable_of_isInitCycle {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : IsInitCycle cfg states solver) :
    TetrisSolvableFor cfg :=
  solvable_of_checked_cycle_through_init h.1 h.2

/-- **The vacuous obligations**: the empty `states` set trivially satisfies
all obligations (all clauses are vacuously true). Useful as a base case
when building up cycle candidates. -/
theorem obligations_empty {cfg : GameConfig} (solver : Solver cfg) :
    Obligations cfg (∅ : Finset GameState) solver :=
  ⟨fun _ h => absurd h (Finset.notMem_empty _),
   fun _ h => absurd h (Finset.notMem_empty _),
   fun _ h => absurd h (Finset.notMem_empty _)⟩

/-- **Round-trip identity**: `mkChecked` is a left inverse of
`obligations_of` (up to definitional equality on the underlying data).
Establishes that the cycle structure and the `(states, solver, Obligations)`
encoding carry exactly the same information. -/
theorem mkChecked_obligations_of {cfg : GameConfig} (C : AdversarialClosedCycle cfg) :
    mkChecked C.states C.solver (obligations_of C) = C := by
  cases C; rfl

/-- **`IsInitCycle` round-trip**: every cycle through `init` corresponds to
a witness of `IsInitCycle`. -/
theorem isInitCycle_of {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) :
    IsInitCycle cfg C.states C.solver :=
  ⟨obligations_of C, h⟩

/-- The empty `states` set never contains `init`, so it's not an init-cycle. -/
theorem not_isInitCycle_empty (cfg : GameConfig) (solver : Solver cfg) :
    ¬ IsInitCycle cfg ∅ solver := fun ⟨_, hinit⟩ =>
  (Finset.notMem_empty _) hinit

/-- An `IsInitCycle` witness implies `states.card ≥ 28` — the depth-6
chain lower bound (Tick 64). -/
theorem isInitCycle_card_ge_28 {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : IsInitCycle cfg states solver) :
    28 ≤ states.card :=
  (mkChecked states solver h.1).init_states_card_ge_twenty_eight h.2

/-- Dot accessor: `obligations` half of an `IsInitCycle` witness. -/
theorem IsInitCycle.obligations {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : IsInitCycle cfg states solver) :
    Obligations cfg states solver := h.1

/-- Dot accessor: `init ∈ states` half of an `IsInitCycle` witness. -/
theorem IsInitCycle.init_mem {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : IsInitCycle cfg states solver) :
    GameState.init ∈ states := h.2

/-- Dot accessor for the 28-lower-bound (Tick 64). -/
theorem IsInitCycle.card_ge_28 {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : IsInitCycle cfg states solver) :
    28 ≤ states.card := isInitCycle_card_ge_28 h

/-- Dot accessor: an `IsInitCycle` witness proves
`TetrisSolvableFor cfg`. Dot form of `solvable_of_isInitCycle`. -/
theorem IsInitCycle.solvable {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : IsInitCycle cfg states solver) :
    TetrisSolvableFor cfg := solvable_of_isInitCycle h

/-- **Init membership is preserved by `init`-cycle witnesses**: a
direct re-exposure of the `init_mem` field as `0 < card`. -/
theorem IsInitCycle.card_pos {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : IsInitCycle cfg states solver) :
    0 < states.card :=
  Finset.card_pos.mpr ⟨GameState.init, h.init_mem⟩

/-- An `IsInitCycle` witness yields `states.Nonempty`. -/
theorem IsInitCycle.nonempty {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : IsInitCycle cfg states solver) :
    states.Nonempty := ⟨GameState.init, h.init_mem⟩


/-- Every state satisfying `Obligations` is safe. Lifts the
`AdversarialClosedCycle.subset_safe` invariant through
`mkChecked`. -/
theorem Obligations.subset_safe {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : Obligations cfg states solver) :
    (states : Set GameState) ⊆ safe cfg :=
  fun _ hg => (mkChecked states solver h).subset_safe
    (Finset.mem_coe.mp hg)

/-- Pointwise form. -/
theorem Obligations.mem_safe {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : Obligations cfg states solver)
    {g : GameState} (hg : g ∈ states) : g ∈ safe cfg :=
  h.subset_safe (Finset.mem_coe.mpr hg)

/-- Dot accessor: promote `Obligations` to a full
`AdversarialClosedCycle`. Equivalent to `mkChecked`. -/
def Obligations.toAdversarialClosedCycle {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : Obligations cfg states solver) : AdversarialClosedCycle cfg :=
  mkChecked states solver h

/-- Promote `Obligations` plus `init ∈ states` to an
`IsInitCycle` witness. -/
theorem Obligations.toIsInitCycle {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : Obligations cfg states solver) (hinit : GameState.init ∈ states) :
    IsInitCycle cfg states solver := ⟨h, hinit⟩

/-- One-shot reduction: `Obligations` + `init ∈ states` +
`4 ≤ cols` implies `TetrisSolvableValidFor cfg`. -/
theorem Obligations.tetrisSolvableValidFor {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (hcols : 4 ≤ cfg.cols) (h : Obligations cfg states solver)
    (hinit : GameState.init ∈ states) : TetrisSolvableValidFor cfg :=
  (tetrisSolvableValidFor_iff_init_safe hcols).mpr (h.mem_safe hinit)

/-- Standard-config specialization. -/
theorem Obligations.tetrisSolvableValid_standard
    {states : Finset GameState} {solver : Solver GameConfig.standard}
    (h : Obligations GameConfig.standard states solver)
    (hinit : GameState.init ∈ states) : TetrisSolvableValid :=
  h.tetrisSolvableValidFor (by decide) hinit

/-- Tiny-config specialization. -/
theorem Obligations.tetrisSolvableValidFor_tiny
    {states : Finset GameState} {solver : Solver GameConfig.tiny}
    (h : Obligations GameConfig.tiny states solver)
    (hinit : GameState.init ∈ states) :
    TetrisSolvableValidFor GameConfig.tiny :=
  h.tetrisSolvableValidFor (by decide) hinit

/-- Permissive form: `Obligations` + `init ∈ states` already gives
the permissive `TetrisSolvableFor cfg` (no `4 ≤ cols` hypothesis
needed, since this only uses the underlying cycle machinery). -/
theorem Obligations.solvable {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : Obligations cfg states solver)
    (hinit : GameState.init ∈ states) : TetrisSolvableFor cfg :=
  h.toAdversarialClosedCycle.init_solves_for hinit

/-- `Obligations` + `init ∈ states` implies `28 ≤ states.card`
(the depth-6 chain lower bound). -/
theorem Obligations.init_states_card_ge_28 {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : Obligations cfg states solver)
    (hinit : GameState.init ∈ states) : 28 ≤ states.card :=
  h.toAdversarialClosedCycle.init_states_card_ge_twenty_eight hinit

/-- `Obligations` + `init ∈ states` implies `0 < states.card`. -/
theorem Obligations.init_states_card_pos {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (_h : Obligations cfg states solver)
    (hinit : GameState.init ∈ states) : 0 < states.card :=
  Finset.card_pos.mpr ⟨GameState.init, hinit⟩

/-- `Obligations` + `init ∈ states` implies `states.Nonempty`. -/
theorem Obligations.init_states_nonempty {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (_h : Obligations cfg states solver)
    (hinit : GameState.init ∈ states) : states.Nonempty :=
  ⟨GameState.init, hinit⟩

/-- `Obligations` + `init ∈ states` implies `init ∈ safe cfg`.
The atomic ingredient of the `tetrisSolvableValidFor` reduction. -/
theorem Obligations.init_safe {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : Obligations cfg states solver)
    (hinit : GameState.init ∈ states) : GameState.init ∈ safe cfg :=
  h.mem_safe hinit

@[simp] theorem Obligations.toAdversarialClosedCycle_states
    {cfg : GameConfig} {states : Finset GameState} {solver : Solver cfg}
    (h : Obligations cfg states solver) :
    h.toAdversarialClosedCycle.states = states := rfl

@[simp] theorem Obligations.toAdversarialClosedCycle_solver
    {cfg : GameConfig} {states : Finset GameState} {solver : Solver cfg}
    (h : Obligations cfg states solver) :
    h.toAdversarialClosedCycle.solver = solver := rfl

/-- **Every IsInitCycle state is safe**: any candidate state in a cycle
through init is provably in the safe set. -/
theorem isInitCycle_states_subset_safe {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    (states : Set GameState) ⊆ safe cfg :=
  h.obligations.subset_safe

/-- Dot form of `isInitCycle_states_subset_safe`. -/
theorem IsInitCycle.subset_safe {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    (states : Set GameState) ⊆ safe cfg :=
  isInitCycle_states_subset_safe h

/-- Init-state version: `init ∈ safe cfg` from any `IsInitCycle`. -/
theorem IsInitCycle.init_safe {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    GameState.init ∈ safe cfg :=
  h.subset_safe (Finset.mem_coe.mpr h.init_mem)

/-- **Contrapositive of `IsInitCycle.init_safe`**: if `init ∉ safe
cfg`, then no `IsInitCycle` witness exists for that config. -/
theorem not_isInitCycle_of_init_not_safe {cfg : GameConfig}
    (h : GameState.init ∉ safe cfg) (states : Finset GameState)
    (solver : Solver cfg) : ¬ IsInitCycle cfg states solver :=
  fun hc => h hc.init_safe

/-- An `IsInitCycle` witness implies `TetrisSolvableValidFor cfg`
(when `4 ≤ cfg.cols`). Stronger than `IsInitCycle.solvable`,
which only produces `TetrisSolvableFor`. -/
theorem IsInitCycle.tetrisSolvableValidFor {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (hcols : 4 ≤ cfg.cols) (h : IsInitCycle cfg states solver) :
    TetrisSolvableValidFor cfg :=
  (tetrisSolvableValidFor_iff_init_safe hcols).mpr h.init_safe

/-- Standard-config specialization of
`IsInitCycle.tetrisSolvableValidFor`. -/
theorem IsInitCycle.tetrisSolvableValid_standard
    {states : Finset GameState} {solver : Solver GameConfig.standard}
    (h : IsInitCycle GameConfig.standard states solver) :
    TetrisSolvableValid :=
  h.tetrisSolvableValidFor (by decide)

/-- Tiny-config specialization of
`IsInitCycle.tetrisSolvableValidFor`. -/
theorem IsInitCycle.tetrisSolvableValidFor_tiny
    {states : Finset GameState} {solver : Solver GameConfig.tiny}
    (h : IsInitCycle GameConfig.tiny states solver) :
    TetrisSolvableValidFor GameConfig.tiny :=
  h.tetrisSolvableValidFor (by decide)

/-- Promote an `IsInitCycle` witness to a full
`AdversarialClosedCycle`. The underlying `states` and `solver` are
preserved (by `mkChecked_*` lemmas). -/
def IsInitCycle.toAdversarialClosedCycle {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    AdversarialClosedCycle cfg :=
  mkChecked states solver h.obligations

@[simp] theorem IsInitCycle.toAdversarialClosedCycle_states
    {cfg : GameConfig} {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    h.toAdversarialClosedCycle.states = states := rfl

@[simp] theorem IsInitCycle.toAdversarialClosedCycle_solver
    {cfg : GameConfig} {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    h.toAdversarialClosedCycle.solver = solver := rfl

/-- Pointwise version of `IsInitCycle.subset_safe`: each cycle
state individually is safe. -/
theorem IsInitCycle.mem_safe {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver)
    {g : GameState} (hg : g ∈ states) : g ∈ safe cfg :=
  h.subset_safe (Finset.mem_coe.mpr hg)

/-- Pointwise: every state in an `IsInitCycle` is not lost. -/
theorem IsInitCycle.not_lost {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver)
    {g : GameState} (hg : g ∈ states) : ¬ g.lost cfg :=
  h.obligations.1 g hg

/-- Pointwise: the solver's choice on each cycle state is valid. -/
theorem IsInitCycle.solver_valid {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver)
    {g : GameState} (hg : g ∈ states) {p : Piece} (hp : p ∈ g.bag) :
    (solver g p).piece = p ∧ (solver g p).Valid cfg :=
  h.obligations.2.1 g hg p hp

/-- Pointwise: the cycle Finset is closed under the solver's step. -/
theorem IsInitCycle.step_mem {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver)
    {g : GameState} (hg : g ∈ states) {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (solver g p) ∈ states :=
  h.obligations.2.2 g hg p hp

/-- Piece-half of `solver_valid`. -/
theorem IsInitCycle.solver_piece {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver)
    {g : GameState} (hg : g ∈ states) {p : Piece} (hp : p ∈ g.bag) :
    (solver g p).piece = p := (h.solver_valid hg hp).1

/-- Valid-half of `solver_valid`. -/
theorem IsInitCycle.solver_placement_valid {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver)
    {g : GameState} (hg : g ∈ states) {p : Piece} (hp : p ∈ g.bag) :
    (solver g p).Valid cfg := (h.solver_valid hg hp).2

/-- **Init is not lost in an `IsInitCycle`.** Init-specialisation
of `IsInitCycle.not_lost`. Useful for confirming the starting
state is alive before any moves. -/
theorem IsInitCycle.init_not_lost {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    ¬ GameState.init.lost cfg :=
  h.not_lost h.init_mem

/-- **Init's solver step lands in `states`.** Init-specialisation
of `IsInitCycle.step_mem`, using `init.bag = full`. -/
theorem IsInitCycle.init_step_mem {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) (p : Piece) :
    adversarialStep cfg GameState.init p (solver GameState.init p) ∈ states :=
  h.step_mem h.init_mem (by simp [GameState.init, Bag.mem_full])

/-- Init-specialisation of `solver_piece`. -/
theorem IsInitCycle.init_solver_piece {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) (p : Piece) :
    (solver GameState.init p).piece = p :=
  h.solver_piece h.init_mem (by simp [GameState.init, Bag.mem_full])

/-- Init-specialisation of `solver_placement_valid`. -/
theorem IsInitCycle.init_solver_placement_valid {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) (p : Piece) :
    (solver GameState.init p).Valid cfg :=
  h.solver_placement_valid h.init_mem (by simp [GameState.init, Bag.mem_full])

/-- Pointwise: for any cycle state and any bag piece, the solver's
choice lies in the search Finset. -/
theorem IsInitCycle.solver_mem_allValidFor {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver)
    {g : GameState} (hg : g ∈ states) {p : Piece} (hp : p ∈ g.bag) :
    solver g p ∈ Placement.allValidFor cfg p :=
  (Placement.mem_allValidFor cfg p _).mpr (h.solver_valid hg hp)

/-- Init-instance `mem_allValidFor`: the solver's choice at init
lies in the search Finset. -/
theorem IsInitCycle.init_solver_mem_allValidFor {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) (p : Piece) :
    solver GameState.init p ∈ Placement.allValidFor cfg p :=
  h.solver_mem_allValidFor h.init_mem
    (by simp [GameState.init, Bag.mem_full])

/-- Pointwise: the solver step factored as a `GameState.step`
(after substituting the announced piece into the placement). The
substitution is a no-op for `IsInitCycle` because
`(solver g p).piece = p`. -/
theorem IsInitCycle.step_mem_step {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver)
    {g : GameState} (hg : g ∈ states) {p : Piece} (hp : p ∈ g.bag) :
    g.step cfg (solver g p) ∈ states := by
  have hsm := h.step_mem hg hp
  rw [adversarialStep_eq_step, Placement.eta_of_piece_eq
        (h.solver_piece hg hp)] at hsm
  exact hsm

/-- Init-specialisation of `step_mem_step`: from init, the
`GameState.step` lands in `states` for any piece. -/
theorem IsInitCycle.init_step_mem_step {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) (p : Piece) :
    GameState.init.step cfg (solver GameState.init p) ∈ states :=
  h.step_mem_step h.init_mem (by simp [GameState.init, Bag.mem_full])

/-- An `IsInitCycle` witness implies `states ≠ ∅`. Dual to
`not_isInitCycle_empty`. -/
theorem IsInitCycle.states_ne_empty {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) : states ≠ ∅ :=
  h.nonempty.ne_empty

/-- An `IsInitCycle` witness has positive card. -/
theorem IsInitCycle.states_card_pos {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) : 0 < states.card :=
  Finset.card_pos.mpr h.nonempty

/-- `≥ 1` form of `states_card_pos`. -/
theorem IsInitCycle.states_card_ge_one {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) : 1 ≤ states.card :=
  h.states_card_pos

/-- `≠ 0` form of `states_card_pos`. -/
theorem IsInitCycle.states_card_ne_zero {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) : states.card ≠ 0 :=
  Nat.ne_of_gt h.states_card_pos

/-- `IsInitCycle` ⇒ `28 ≤ states.card` (the depth-6 chain lower bound). -/
theorem IsInitCycle.states_card_ge_28 {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) : 28 ≤ states.card :=
  h.1.init_states_card_ge_28 h.2

/-- `IsInitCycle` ⇒ `8 ≤ states.card` (weakening of `_ge_28`). -/
theorem IsInitCycle.states_card_ge_eight {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) : 8 ≤ states.card :=
  Nat.le_trans (by decide : (8 : ℕ) ≤ 28) h.states_card_ge_28

/-- `IsInitCycle` ⇒ `14 ≤ states.card` (weakening of `_ge_28`). -/
theorem IsInitCycle.states_card_ge_fourteen {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) : 14 ≤ states.card :=
  Nat.le_trans (by decide : (14 : ℕ) ≤ 28) h.states_card_ge_28

/-- `IsInitCycle` ⇒ `states.card > 27`. -/
theorem IsInitCycle.states_card_gt_twenty_seven {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) : 27 < states.card := by
  have := h.states_card_ge_28; omega

/-- `IsInitCycle` ⇒ `7 ≤ states.card` (weakening of `_ge_28`). -/
theorem IsInitCycle.states_card_ge_seven {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) : 7 ≤ states.card :=
  Nat.le_trans (by decide : (7 : ℕ) ≤ 28) h.states_card_ge_28

/-- `IsInitCycle` ⇒ `2 ≤ states.card`. -/
theorem IsInitCycle.states_card_ge_two {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) : 2 ≤ states.card :=
  Nat.le_trans (by decide : (2 : ℕ) ≤ 28) h.states_card_ge_28

/-- `IsInitCycle` ⇒ there exists a state with full bag in `states`. -/
theorem IsInitCycle.exists_bag_full {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) : ∃ g ∈ states, g.bag = Bag.full :=
  ⟨GameState.init, h.init_mem, GameState.init_bag⟩

/-- `IsInitCycle` ⇒ there exists a state in `states` with board count 0. -/
theorem IsInitCycle.exists_board_count_zero {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    ∃ g ∈ states, g.board.count = 0 :=
  ⟨GameState.init, h.init_mem, GameState.init_board_count⟩

/-- `IsInitCycle` ⇒ there exists a state in `states` with `count ≤ k`
for any `k`. Trivially via init (count 0). -/
theorem IsInitCycle.exists_board_count_le {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) (k : ℕ) :
    ∃ g ∈ states, g.board.count ≤ k :=
  ⟨GameState.init, h.init_mem, by rw [GameState.init_board_count]; exact Nat.zero_le _⟩

/-- `IsInitCycle` ⇒ there exists a state in `states` with `count % n = 0`
for any `n`. Trivially via init (count = 0). -/
theorem IsInitCycle.exists_board_count_mod_eq_zero {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) (n : ℕ) :
    ∃ g ∈ states, g.board.count % n = 0 :=
  ⟨GameState.init, h.init_mem, by rw [GameState.init_board_count]; exact Nat.zero_mod _⟩

/-- `IsInitCycle` ⇒ there exists a state in `states` with empty board. -/
theorem IsInitCycle.exists_board_empty {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    ∃ g ∈ states, g.board = Board.empty :=
  ⟨GameState.init, h.init_mem, GameState.init_board⟩

/-- `IsInitCycle` ⇒ there exists a state in `states` with `bag.card = 7`. -/
theorem IsInitCycle.exists_bag_card_eq_seven {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    ∃ g ∈ states, g.bag.card = 7 :=
  ⟨GameState.init, h.init_mem, GameState.init_bag_card⟩

/-- `IsInitCycle` ⇒ there exists a state in `states` with simultaneously
empty board and full bag (i.e. `init` itself). -/
theorem IsInitCycle.exists_board_empty_and_bag_full {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    ∃ g ∈ states, g.board = Board.empty ∧ g.bag = Bag.full :=
  ⟨GameState.init, h.init_mem, GameState.init_board, GameState.init_bag⟩

/-- `IsInitCycle` ⇒ there exists a state in `states` equal to `init`. -/
theorem IsInitCycle.exists_eq_init {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    ∃ g ∈ states, g = GameState.init :=
  ⟨GameState.init, h.init_mem, rfl⟩

/-- `IsInitCycle` ⇒ `init ∈ states ∩ safe`. -/
theorem IsInitCycle.init_mem_inter_safe {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    GameState.init ∈ states ∧ GameState.init ∈ safe cfg :=
  ⟨h.init_mem, h.init_safe⟩

/-- `IsInitCycle` ⇒ `init ∈ states` and `init` is not lost. -/
theorem IsInitCycle.init_mem_and_not_lost {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    GameState.init ∈ states ∧ ¬ GameState.init.lost cfg :=
  ⟨h.init_mem, h.init_not_lost⟩

/-- `IsInitCycle` ⇒ there exists `g ∈ states` that is also in `safe cfg`. -/
theorem IsInitCycle.exists_mem_safe {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    ∃ g ∈ states, g ∈ safe cfg :=
  ⟨GameState.init, h.init_mem, h.init_safe⟩

/-- `IsInitCycle` ⇒ there exists `g ∈ states` that is not lost. -/
theorem IsInitCycle.exists_not_lost {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    ∃ g ∈ states, ¬ g.lost cfg :=
  ⟨GameState.init, h.init_mem, h.init_not_lost⟩

/-- `IsInitCycle` ⇒ `states.card ≠ 1` (since `≥ 2`). -/
theorem IsInitCycle.states_card_ne_one {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) : states.card ≠ 1 := by
  have h2 : 2 ≤ states.card := h.states_card_ge_two
  omega

/-- **`IsInitCycle` full pointwise invariants**: at any cycle
state, the state is `¬ lost` and safe. (Solver-side invariants
require also `hp : p ∈ g.bag`, so are separate.) -/
theorem IsInitCycle.full_invariants {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) {g : GameState}
    (hg : g ∈ states) : ¬ g.lost cfg ∧ g ∈ safe cfg :=
  ⟨h.not_lost hg, h.mem_safe hg⟩

/-- **`IsInitCycle` per-piece full invariants**: at any cycle
state `g` and bag piece `p`, the solver plays the right piece,
the placement is valid, and the successor is in `states`. -/
theorem IsInitCycle.full_solver_invariants {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) {g : GameState}
    (hg : g ∈ states) {p : Piece} (hp : p ∈ g.bag) :
    (solver g p).piece = p ∧ (solver g p).Valid cfg ∧
    adversarialStep cfg g p (solver g p) ∈ states :=
  ⟨h.solver_piece hg hp, h.solver_placement_valid hg hp,
   h.step_mem hg hp⟩

/-- **Concrete non-cycle witness**: the singleton `{init}` is never an
init-cycle, regardless of `solver`. Direct contradiction with
`Bag.draw_ne_self`: closure would require the post-draw state to equal
`init`, but the bag necessarily changes. This is the smallest possible
test case for the `IsInitCycle` decidable predicate. -/
theorem not_isInitCycle_singleton (cfg : GameConfig) (solver : Solver cfg) :
    ¬ IsInitCycle cfg {GameState.init} solver := by
  intro ⟨hobl, _⟩
  have hO : Piece.O ∈ GameState.init.bag := Bag.mem_full _
  have hclosed := hobl.2.2 GameState.init (Finset.mem_singleton.mpr rfl) Piece.O hO
  have heq : adversarialStep cfg GameState.init Piece.O
      (solver GameState.init Piece.O) = GameState.init :=
    Finset.mem_singleton.mp hclosed
  have hbag : GameState.init.bag.draw Piece.O = GameState.init.bag := by
    have := congrArg GameState.bag heq
    rw [adversarialStep_bag] at this
    exact this
  exact Bag.draw_ne_self hO hbag

/-- **Direct contraposition of `init_states_card_ge_twenty_eight`**: any
`(states, solver)` pair with `states.card < 28` is *not* an init-cycle.
Packaged on the decidable predicate so it can be used for early pruning
in a search loop. -/
theorem not_isInitCycle_of_card_lt_28 {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : states.card < 28) :
    ¬ IsInitCycle cfg states solver := by
  intro hcyc
  have hbound :=
    (mkChecked states solver hcyc.1).init_states_card_ge_twenty_eight hcyc.2
  rw [mkChecked_states] at hbound
  omega

/-- Singleton cardinalities can't form an init-cycle (contrapositive of
`IsInitCycle.states_card_ne_one`). -/
theorem not_isInitCycle_of_card_eq_one {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : states.card = 1) :
    ¬ IsInitCycle cfg states solver :=
  fun hcyc => hcyc.states_card_ne_one h

/-- Empty `states` can't form an init-cycle (contrapositive of
`IsInitCycle.states_card_ne_zero`). -/
theorem not_isInitCycle_of_card_eq_zero {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : states.card = 0) :
    ¬ IsInitCycle cfg states solver :=
  fun hcyc => hcyc.states_card_ne_zero h

/-- States with card `≤ 27` can't form an init-cycle. -/
theorem not_isInitCycle_of_card_le_27 {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : states.card ≤ 27) :
    ¬ IsInitCycle cfg states solver :=
  not_isInitCycle_of_card_lt_28 (by omega)

/-- States with card exactly 7 can't form an init-cycle. -/
theorem not_isInitCycle_of_card_eq_seven {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : states.card = 7) :
    ¬ IsInitCycle cfg states solver :=
  not_isInitCycle_of_card_le_27 (by omega)

/-- States with card exactly 28 can be an init-cycle (boundary case);
states with card `< 28` cannot. -/
theorem not_isInitCycle_of_card_eq_two {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : states.card = 2) :
    ¬ IsInitCycle cfg states solver :=
  not_isInitCycle_of_card_le_27 (by omega)

/-- States with card in `[1, 27]` can't form an init-cycle. -/
theorem not_isInitCycle_of_card_in_one_to_27 {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (_h1 : 1 ≤ states.card) (h2 : states.card ≤ 27) :
    ¬ IsInitCycle cfg states solver :=
  not_isInitCycle_of_card_le_27 h2

/-- `IsInitCycle` ⇒ there exists a state in `states` other than `init`
(card ≥ 28 ≥ 2 guarantees a second state via `Finset.one_lt_card`). -/
theorem IsInitCycle.exists_ne_init {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : IsInitCycle cfg states solver) :
    ∃ g ∈ states, g ≠ GameState.init := by
  have h2 : 2 ≤ states.card := h.states_card_ge_two
  obtain ⟨a, ha, b, hb, hab⟩ := Finset.one_lt_card.mp (by omega : 1 < states.card)
  by_cases ha_init : a = GameState.init
  · exact ⟨b, hb, fun hb_init => hab (ha_init.trans hb_init.symm)⟩
  · exact ⟨a, ha, ha_init⟩

/-- If `init ∉ states`, no `IsInitCycle` witness exists. -/
theorem not_isInitCycle_of_init_notMem {cfg : GameConfig}
    {states : Finset GameState} {solver : Solver cfg}
    (h : GameState.init ∉ states) :
    ¬ IsInitCycle cfg states solver :=
  fun hcyc => h hcyc.2

/-- **Cycle states are in-field upward**: every cycle state has every
filled cell strictly below `cfg.rows`. Direct unwinding of `not_lost` from
the cycle structure. The first step toward a `C.states ⊆ inFieldStates`
upper bound. -/
theorem cycle_state_in_field {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {s : GameState} (hs : s ∈ C.states) : ∀ p ∈ s.board, p.2 < cfg.rows := by
  intro p hp
  have hnl : ¬ s.lost cfg := C.not_lost s hs
  exact Nat.lt_of_not_le (fun h => hnl ⟨p, hp, h⟩)

/-- **Init corollary.** Any cycle containing `GameState.init` has at least 8
states (the 7-bag forces at least one transition state per piece, plus `init`
itself). -/
theorem init_states_card_ge_eight {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) :
    8 ≤ C.states.card := by
  have hne : (GameState.init.bag).Nonempty := Bag.full_nonempty
  have hfull : (GameState.init.bag).card = 7 := by
    change Bag.full.card = 7
    decide
  have := C.states_card_gt_bag_card h hne
  rw [hfull] at this
  exact this

/-- **`M4Artifact.union` states cardinality monotonicity (left)**:
`M.states.card ≤ (M.union M').states.card`. Lower bound complement
to Tick 1966's upper bound. Direct via `union_states` + `Finset.card_le_card`
on `subset_union_left`. -/
theorem M4Artifact.states_card_le_union_states_card_left {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    M.states.card ≤ (M.union M').states.card := by
  rw [M4Artifact.union_states]
  exact Finset.card_le_card Finset.subset_union_left

/-- **`M4Artifact.union` states cardinality monotonicity (right)**:
right-side counterpart of `states_card_le_union_states_card_left`. -/
theorem M4Artifact.states_card_le_union_states_card_right {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    M'.states.card ≤ (M.union M').states.card := by
  rw [M4Artifact.union_states]
  exact Finset.card_le_card Finset.subset_union_right

/-- **M4 union states.card max lower bound**:
`max M.states.card M'.states.card ≤ (M.union M').states.card`.
Combines Tick 1971's two monotonicity lemmas via `max_le`. -/
theorem M4Artifact.max_states_card_le_union_states_card {cfg : GameConfig}
    (M M' : M4Artifact cfg) :
    max M.states.card M'.states.card ≤ (M.union M').states.card :=
  max_le (M.states_card_le_union_states_card_left M')
         (M.states_card_le_union_states_card_right M')

/-- **`M4Artifact.fromCycle` states cardinality ≥ 8**: `(fromCycle C
h).states.card ≥ 8`. Direct via `fromCycle_states` (Tick 1936) +
cycle's local `init_states_card_ge_eight` (end of file). Useful
concrete bound for cycle-derived M4 artifacts. -/
theorem M4Artifact.fromCycle_states_card_ge_eight {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    8 ≤ (M4Artifact.fromCycle C h).states.card := by
  rw [M4Artifact.fromCycle_states]
  exact C.init_states_card_ge_eight h

/-- **`M4Artifact.fromCycleCanon` states cardinality ≥ 8**:
canonical-form parallel of Tick 1973. Direct via
`fromCycleCanon_states` (Tick 1936) +
cycle's `init_states_card_ge_eight`. -/
theorem M4Artifact.fromCycleCanon_states_card_ge_eight {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (h : GameState.init ∈ C.states) :
    8 ≤ (M4Artifact.fromCycleCanon C h).states.card := by
  rw [M4Artifact.fromCycleCanon_states]
  exact C.init_states_card_ge_eight h

/-- **`M4Artifact.ofIsClosedOn` states cardinality ≥ 8**: direct
M4-constructor parallel of atlas-level `init_states_card_ge_eight`
(Tick 1943). Closes the `≥ 8` toolkit at all three M4 constructors:
fromCycle, fromCycleCanon, ofIsClosedOn. -/
theorem M4Artifact.ofIsClosedOn_states_card_ge_eight {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    8 ≤ (M4Artifact.ofIsClosedOn h hinit).states.card := by
  rw [M4Artifact.ofIsClosedOn_states]
  exact h.init_states_card_ge_eight hinit

/-- **M4 atlas dom is nonempty**: `(M.atlas.dom M.states).Nonempty`.
Direct via `atlas_dom_states_eq` (Tick 1651) + `states_nonempty`.
Useful named accessor for proofs needing dom nonemptiness. -/
theorem M4Artifact.atlas_dom_states_nonempty {cfg : GameConfig}
    (M : M4Artifact cfg) : (M.atlas.dom M.states).Nonempty := by
  rw [M.atlas_dom_states_eq]
  exact M.states_nonempty

end AdversarialClosedCycle

/-- **Atlas-level dom nonempty for closed atlases**:
`(h : A.IsClosedOn cfg S) → S.Nonempty → (A.dom S).Nonempty`.
Direct via `dom_eq` (Adversarial line ~1449). Atlas-side parallel
of M4-level Tick 1976. -/
theorem Atlas.IsClosedOn.dom_nonempty {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hne : S.Nonempty) : (A.dom S).Nonempty := by
  rw [h.dom_eq]
  exact hne

/-- **Atlas dom cardinality ≥ 8 for init-containing closed atlases**:
`(h : A.IsClosedOn cfg S) → init ∈ S → 8 ≤ (A.dom S).card`. Direct
via `dom_card_eq` (Tick 1960) + `init_states_card_ge_eight` (Tick
1943). Atlas-side parallel of M4-level Tick 1979. -/
theorem Atlas.IsClosedOn.dom_card_ge_eight {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) : 8 ≤ (A.dom S).card := by
  rw [h.dom_card_eq]
  exact h.init_states_card_ge_eight hinit

/-- **Atlas canon-form dom card ≥ 8 for init-containing closed atlases**:
parallel of Tick 1980 in canonical form. -/
theorem Atlas.IsClosedOn.canon_dom_card_ge_eight {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) : 8 ≤ ((A.canon S).dom S).card := by
  rw [h.canon_dom_card_eq]
  exact h.init_states_card_ge_eight hinit

/-- **Atlas restrict-form dom card ≥ 8 for init-containing closed atlases**:
parallel of Tick 1980 in restricted form. -/
theorem Atlas.IsClosedOn.restrict_dom_card_ge_eight {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) : 8 ≤ ((A.restrict S).dom S).card := by
  rw [h.restrict_dom_card_eq]
  exact h.init_states_card_ge_eight hinit

namespace AdversarialClosedCycle

/-- **Cycle solver-atlas dom states nonempty**:
`(C.solver.toAtlas.dom C.states).Nonempty` for any cycle with
nonempty states. Cycle-side parallel of Ticks 1976/1977. -/
theorem solver_toAtlas_dom_states_nonempty {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) (hne : C.states.Nonempty) :
    (C.solver.toAtlas.dom C.states).Nonempty :=
  C.solver_toAtlas_isClosedOn_states.dom_nonempty hne

/-- **M4 atlas dom states cardinality ≥ 8**: `8 ≤ (M.atlas.dom
M.states).card`. Direct via `atlas_dom_states_card_eq` (Tick 1959)
+ `states_card_ge_eight` (Tick 1941). Useful concrete bound for M4
effective domain size. -/
theorem M4Artifact.atlas_dom_states_card_ge_eight {cfg : GameConfig}
    (M : M4Artifact cfg) : 8 ≤ (M.atlas.dom M.states).card := by
  rw [M.atlas_dom_states_card_eq]
  exact M.states_card_ge_eight

/-- **Cycle solver-atlas dom cardinality ≥ 8 for init-containing
cycles**: `(C : AdversarialClosedCycle cfg) → init ∈ C.states →
8 ≤ (C.solver.toAtlas.dom C.states).card`. Cycle-side parallel of
Ticks 1979 + 1980. -/
theorem solver_toAtlas_dom_states_card_ge_eight
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (hinit : GameState.init ∈ C.states) :
    8 ≤ (C.solver.toAtlas.dom C.states).card :=
  C.solver_toAtlas_isClosedOn_states.dom_card_ge_eight hinit

/-- **Cycle canon solver-atlas dom card ≥ 8**: canon-form parallel
of Tick 1981 via Tick 1982. -/
theorem solver_toAtlas_canon_dom_states_card_ge_eight
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (hinit : GameState.init ∈ C.states) :
    8 ≤ ((C.solver.toAtlas.canon C.states).dom C.states).card :=
  C.solver_toAtlas_isClosedOn_states.canon_dom_card_ge_eight hinit

/-- **M4 canon-form atlas dom card ≥ 8**: canon-form parallel of
Tick 1979 via Tick 1982. -/
theorem M4Artifact.atlas_canon_dom_states_card_ge_eight {cfg : GameConfig}
    (M : M4Artifact cfg) :
    8 ≤ ((M.atlas.canon M.states).dom M.states).card :=
  M.closed.canon_dom_card_ge_eight M.init_mem

/-- **Cycle restrict-form solver-atlas dom card ≥ 8**: restrict-form
parallel of Tick 1981 via Tick 1982. -/
theorem solver_toAtlas_restrict_dom_states_card_ge_eight
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (hinit : GameState.init ∈ C.states) :
    8 ≤ ((C.solver.toAtlas.restrict C.states).dom C.states).card :=
  C.solver_toAtlas_isClosedOn_states.restrict_dom_card_ge_eight hinit

/-- **M4 restrict-form atlas dom card ≥ 8**: restrict-form parallel
of Tick 1979 via Tick 1982. -/
theorem M4Artifact.atlas_restrict_dom_states_card_ge_eight {cfg : GameConfig}
    (M : M4Artifact cfg) :
    8 ≤ ((M.atlas.restrict M.states).dom M.states).card :=
  M.closed.restrict_dom_card_ge_eight M.init_mem

/-- **M4 IsCanonical iff atlas = canon.atlas**: `M.IsCanonical ↔
M.atlas = M.canon.atlas`. Direct `Iff.rfl` because
`M.canon.atlas = M.atlas.canon M.states` (canon_atlas, @[simp])
and `IsCanonical` is defined as `M.atlas = M.atlas.canon M.states`.
Useful named accessor for canonicality biconditional via `M.canon.atlas`. -/
theorem M4Artifact.isCanonical_iff_atlas_eq_canon_atlas {cfg : GameConfig}
    (M : M4Artifact cfg) :
    M.IsCanonical ↔ M.atlas = M.canon.atlas := Iff.rfl

/-- **`(M.union M').canon = M.union M'` when both are canonical**:
combines `isCanonical_union_of_isCanonical` with
`canon_eq_self_of_isCanonical`. Union of two canonical M4 artifacts
is canon-stable (canon doesn't change it). -/
theorem M4Artifact.union_canon_eq_self_of_isCanonical_isCanonical
    {cfg : GameConfig} {M M' : M4Artifact cfg}
    (hM : M.IsCanonical) (hM' : M'.IsCanonical) :
    (M.union M').canon = M.union M' :=
  (canon_eq_self_of_isCanonical
    (isCanonical_union_of_isCanonical hM hM')).symm

end AdversarialClosedCycle

end Tetris
