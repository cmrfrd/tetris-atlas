# Roadmap: proving Tetris solvable

## The target

```
TetrisSolvable := ∃ σ : Solver StandardTetris, SolvesTetris StandardTetris σ
```

By `safe_extract`, this reduces to **one** membership claim:

> `GameState.init ∈ safe GameConfig.standard`

`safe` is the greatest fixed point of `safeOp` (defined in `Proofs/Safety/SafeSet.lean`).

> For the module/layer structure and the curated theorem spine, see `LIBRARY.md`.
> Build the green standard library with `lake build`; the research routes with
> `lake build ProofsExperiments`.

The question is **mathematically open**. Project memory (see
`~/.claude/projects/.../memory/project_7bag_atlas_findings.md`) shows the
team's computational GFP search converged to ∅ under various restrictions —
the unrestricted 10×20 case is unresolved.

This roadmap is the parallel attack plan: three independent routes whose
infrastructure pays off regardless of whether the conjecture turns out to be
provable.

---

## Route 1 — Concrete construction (`AdversarialClosedCycle`)

Build a `Finset GameState` and a `Solver` satisfying the four obligations of
`AdversarialClosedCycle`; apply `init_solves` for `TetrisSolvable`.

**Hard for full 10×20 7-bag** (open). Workable on degenerate configs.

Sub-tasks:
- 1.1 ✅ Parameterize `TetrisSolvable`/`SafeSolves` over arbitrary `GameConfig`. (Done early; `TetrisSolvableFor cfg`.)
- 1.2 ⏳ Pick a deliberately solvable degenerate config — `GameConfig.tiny` (4×4) defined (Tick 100); no concrete cycle yet.
- 1.3 ✅ `AdversarialClosedCycle.Obligations` (decidable bundle, Ticks 66-68);
  `IsInitCycle` (decidable cycle-through-init predicate);
  `mkChecked` smart constructor; `solvable_of_isInitCycle` reduction.
  Plus WF + Reachable strengthened versions (Ticks 76, 88).
- 1.4 ❌ Concrete cycle for degenerate config. (Still open.)
- 1.5 ❌ Cycle on `StandardTetris`. (Open.)

**Major framework results** (Ticks 47-131):
- Cycle structure tower: `Reachable → WF → vanilla`, each with decidable obligations.
- Lower bound: `28 ≤ |C.states|` for any init-cycle (Tick 64).
- Upper bound: `|C.states| ≤ |inFieldStates cfg|` for WF cycles (Tick 76).
- Full bidirectional equivalence:
  `init ∈ safe ↔ TetrisSolvableValidFor cfg ↔ ∃ ReachableClosedCycle through init`
  (Ticks 98, 124, 128).
- Legal-sequence existence (Tick 111) + extension (Tick 120).
- `safeSolver` is a `ValidSolver` for `cfg.cols ≥ 4` (Tick 104).
- All theorems axiom-clean (only `propext`, `Classical.choice`, `Quot.sound`).

---

## Route 2 — Computable safe-set iteration (decidability)

Replace the non-computable `safe`/`safeIterate` (defined over `Set GameState`)
with a computable form on `Finset (InFieldBoard × Bag)`, with convergence in
≤ |universe| steps. Reduces `init ∈ safe` to a decidable proposition (even if
intractable in practice). This is the closest match to the team's prior
empirical GFP work.

Sub-tasks:
- 2.1 `DecidableEq GameState`; `DecidableEq InFieldBoard`. (Should follow from
  `DecidableEq` on `Board = Finset Coord`, already available.)
- 2.2 `Decidable` instance for `Placement.Valid cfg pl` (it's a bounded ∀).
- 2.3 `Decidable` instance for `Board.isLost`, `Board.isFull` (already partial).
- 2.4 Define `allValidPlacements cfg : Piece → Finset Placement` (rotations ×
  in-bounds columns).
- 2.5 Computable `Fintype` instance for `InFieldBoard cfg` (replacing the
  current `noncomputable` via `Fintype.ofInjective`).
- 2.6 `F_finite cfg : Finset GameState → Finset GameState` (the computable
  monotone operator).
- 2.7 `safeIterFinite cfg : ℕ → Finset GameState` and convergence: there
  exists `N ≤ |universe|` such that `safeIterFinite cfg N = F_finite (safeIterFinite cfg N)`.
- 2.8 Soundness: `↑(safeIterFinite cfg N) ⊆ safe cfg`.
- 2.9 Completeness on in-field states: `g ∈ safe cfg ∧ g ∈ in-field → g ∈ safeIterFinite cfg N`.
- 2.10 `Decidable (GameState.init ∈ safe GameConfig.standard)`.

---

## Route 3 — Symmetry-quotient (REMOVED)

Reflection / horizontal symmetry was explored as a state-space-quotient
optimisation (2× reduction). Decision: **not needed for a final proof of
`TetrisSolvable`** — symmetry only shrinks the search space, doesn't change
whether the proof exists. Removed in favour of focusing on the existence
question itself.

Historical sub-tasks (kept only as a record of what was/wasn't done):

- 3.5 `Board.dropped_reflect`: dropping the reflected placement on the reflected
  board gives the reflection of dropping the original.
- 3.6 `Board.applyStep_reflect`: full step commutes with reflection.
- 3.7 `safe_reflect_iff`: `g ∈ safe cfg ↔ GameState.reflect cfg g ∈ safe cfg`.

(All of Route 3 was removed in favour of focusing on Routes 1 and 2.)

---

## Files

| Module | Contents |
|---|---|
| `Proofs.Safety.Adversarial` | `Solver`, `LegalSequence`, `SolvesTetris`, `TetrisSolvable` |
| `Proofs.Safety.SafeSet` | `safeOp`, `safe`, `safe_extract`, `AdversarialClosedCycle` |
| `Proofs.Safety.SafeIterate` | abstract `safeIterate` from top + chain lemmas |
| `Proofs.Safety.SafeIterateFinite` | computable `safeIterFinite`, convergence, soundness, completeness, `decideSafeFromUniverse`, `inFieldStates` |
| `Proofs.Survival.Survival` | `Policy`, `trace`, `SurvivesForever`, `safe_invariant`, `ClosedCycle` |
| `Proofs.Safety.Safety` | universal safety / pigeonhole obstructions |
| `Proofs.Invariants.StateSpace` | `InFieldBoard` finite type (computable Fintype) |
| ... | (game model, gameplay, etc.) |

---

## Ground rules

1. **Only base axioms**: `propext`, `Classical.choice`, `Quot.sound`. No
   custom axioms. Verify with `#print axioms ThmName`.
2. **Minimize `sorry`**. Permitted only as explicit scaffolding for an
   in-progress proof, with a comment indicating *why* and what's planned.
   Track count in `PROGRESS.md`.
3. **Document progress**. After every working session, update `PROGRESS.md`
   with what changed, current sorry count, blockers, and the next step.
4. **Build must pass**. If broken, fix the build before doing anything else.

---

## Recommended order

Route 2 (algorithmic safe set) is substantively complete — its only remaining
gap is the universe-containment hypothesis (`safe cfg ⊆ ↑S₀`), blocked by
`safe` not implying WF. Route 3 is removed (symmetry was a constant-factor
optimisation, not a path to the proof).

The remaining productive directions are:

1. **Route 1 — concrete witness.** Construct an explicit `AdversarialClosedCycle`
   (even on a degenerate config) and discharge via `init_solves`. This is the
   honest path to a positive proof, and produces falsifiable structure that the
   project's empirical work can target.

2. **Address the Route 2 universe blocker** — either redefine `safe` to require
   WF (and re-verify the chain), or define a relaxed `inFieldStates` that
   doesn't require WF (and prove finiteness via a different bound).

3. **Strengthen the abstract framework** — e.g. add `Reachable → WF` style
   invariants to better connect `safe` (defined adversarially) with the game's
   actual reachable subset.
