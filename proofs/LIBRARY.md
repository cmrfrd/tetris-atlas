# Tetris Proofs — Standard Library Guide

This document is the curated map of the Lean formalization: the canonical
theorem **spine**, the **build/axiom-hygiene** structure, and the **roadmap**
for turning research experiments into reusable library modules.

It complements `README.md` (build instructions) and `ROADMAP.md` (the proof
strategy). Audit basis: a 749-declaration theorem-by-theorem inventory of all 35
project files (2026-06-28).

---

## TL;DR

- The **conceptual goal is reduced, with proof**, to one decidable membership:
  `GameState.init ∈ safe GameConfig.standard`, characterized three equivalent
  ways — greatest-fixed-point safe set / reachable closed cycle / nonempty
  closed **Atlas**. *"Building the Atlas = solving Tetris"* is a proven `Iff`.
  What is open is the **witness**, not the framework.
- The repo has **no custom axioms** and the green library is **`sorry`-free and
  `native_decide`-free** (verified — see below).
- Research lives in a **separate lake target** (`ProofsExperiments`) so its
  `native_decide` / large scaffolds never taint the standard library.

---

## 1. Build structure & axiom hygiene

Two `lean_lib` targets (`lakefile.toml`):

| Target | Root | Contents | Axiom status |
|---|---|---|---|
| **`Proofs`** (default) | `Proofs.lean` | game model (`Model/`), `Combinatorics/`, `Invariants/`, safe-set/Atlas/solvability spine (`Safety/`, `Survival/`), promoted results | base axioms only: `propext, Classical.choice, Quot.sound`; no `sorry`, no `native_decide` |
| `ProofsExperiments` (manual) | `ProofsExperiments.lean` | active + floored research routes | may use `native_decide` (→ `Lean.ofReduceBool`); still `sorry`-free |

```sh
lake build                  # green standard library (fast; native_decide-free)
lake build ProofsExperiments  # research routes (foreground only)
```

**Why the split.** Lake's default glob builds only modules transitively imported
from the lib root. Previously `Proofs.lean` rooted `FiveBagReset` and
`OnlineReservoir`, both of which use `native_decide`, so the *default build was
not base-axiom clean*. Moving the experiment imports into `ProofsExperiments.lean`
restores a clean spine. The `sorry` scaffold `Archive/AbstractSafe.lean` is imported
by **neither** target (record only; its proofs are the realization crux #66/#72).

**Verification (run after any spine change).**
```sh
# 1. green hygiene gate — fails if any sorry/native_decide leaks into the green tree
#    (scans Proofs/ minus Experiments/ and Archive/). Wire into CI / pre-commit.
scripts/check-green-clean.sh
lake build                  # green build must succeed
# 2. axiom gate — must print exactly [propext, Classical.choice, Quot.sound]
cat > axiom_gate_check.lean <<'EOF'
import Proofs
open Tetris
#print axioms safe_extract
#print axioms safe_greatest
#print axioms safe_invariant
#print axioms closed_cycle_survives
#print axioms tetrisSolvableValidFor_iff_init_mem_safeIterFinite
EOF
lake env lean axiom_gate_check.lean && rm axiom_gate_check.lean
```
Last run (2026-06-28): all spine theorems `[propext, Classical.choice, Quot.sound]`;
green target 8271 jobs clean; `ProofsExperiments` 8281 jobs.

> **Build rules (project convention):** Lean/lake builds are **foreground only**;
> never SIGTERM an in-flight `lake` build; re-run `lake build` after any theorem
> move to confirm it still compiles.

---

## 2. The curated spine (critical theorems)

The standard library cannot omit these, organized bottom-up. (Trivial restatement
families — the `Config` decide-facts, `bag_card_*` ladder, per-index
`trace_one/two/three` — are intentionally excluded.) All names are in
`namespace Tetris` (sometimes nested, e.g. `Tetris.Board`, `Tetris.Placement`).

### Layer 0 — Game model (`Config, Piece, Board, Placement, Bag, Game`)
| Theorem | Statement |
|---|---|
| `Board.not_isLost_of_colHeight_le` | every col height ≤ rows ⇒ not lost — **the survival discharge** |
| `Board.not_isLost_iff_forall_row_lt`, `Board.clearLines_wf` | loss↔in-field bridge; clears preserve WF |
| `Placement.count_place`, `Placement.cellsAt_injective`, `Placement.dropped_disjoint` | a placement adds exactly 4 cells; injectivity; drop has no overlap |
| `Bag.draw`, `Bag.draw_ne_self`, `Bag.draw_injOn`, `Bag.fintype_card` | the 7-bag mechanic; finiteness (128 bags) |
| `Reachable`, `Reachable.eq_init_or_exists_step` | reachability + the predecessor case-split |
| `Placement.Valid`, `allValidFor`, `mem_allValidFor`, `GameState.step_image_card`, `init_step_image_card` | finite/decidable move set; branching = `|bag|` |

### Layer 1 — Geometry & counting (`PieceGeometry, BoardCount, ColumnCount, StepInvariants, GameplayExtra`)
| Theorem | Statement |
|---|---|
| `Piece.shape_card`/`shapeUp_card`; `shapeUp_S_row_card_le_two`, `shapeUp_T_row_card_le_three` | every piece = 4 cells; per-row footprint (only I makes a 4-bar) |
| `Board.applyStep_colCount` | per-column conservation of a full move, **no WF hypothesis** |
| `Board.colCount_clearLines_add`, `clearLines_card`, `clearLines_count_add` | each clear removes 1 cell/col; gravity relocates-not-merges; `cleared = cols·k` |
| `Board.not_isLost_iff_forall_colHeight_le`; `isLost_union`, `not_isLost_of_subset`, `not_isLost_clearLines` | column-form safety equivalence + the safety lattice |
| `applyStep_count` | `count' + cols·linesCleared = count + 4` |
| `Board.clearLines_no_full`, `dropped_resting`, `not_canDraw_after_draw` | line-clear correctness; hard-drop maximality; 7-bag no-immediate-repeat |
| `colHeight_clearLines_le`, `linesCleared_place_le_four` | clears never raise height; per-drop ≤ 4 lines |
| `reachable_WF`, `reachable_no_full`, `reachable_bag_nonempty` | reachable invariants |

### Layer 2 — State space & local safety (`StateSpace, Safety, Holes`)
| Theorem | Statement |
|---|---|
| `InFieldBoard` + `instFintype` + `fintype_card` (`2^(cols·rows)`) | **finiteness** — makes Atlas/GFP enumeration well-defined |
| `low_stack_safe`, `exists_safe_placement`, `reachable_low_stack_exists_reachable_step` | universal local safety: "you can always play one more step" |
| `isLost_of_count_gt` | over-capacity ⇒ lost (pigeonhole) |
| `Board.colRows_card_add_colHoles`, `holes_eq_zero_iff`, `holes_pos_iff` | filled+holes=height; hole-free characterization |

### Layer 3 — Safe-set, Atlas & solvability (`Adversarial, SafeSet, SafeIterate, SafeIterateFinite`) — the reduction spine
| Theorem | Statement |
|---|---|
| `Atlas`, `Atlas.IsClosedOn`, `TetrisSolvable`, `tetrisSolvable_of_exists_init_closed_atlas`, `Atlas.IsClosedOn.unionOn` | the **Atlas** as a formal object; init-containing closed atlas ⇒ solvable; atlases compose |
| `safeOp`, `safe`, `safe_eq`, `mem_safe_iff`, **`safe_greatest`** | the GFP + **its coinduction principle** (primary membership tool) |
| **`safe_extract`** | `init ∈ safe StandardTetris → TetrisSolvable` — **THE headline reduction** |
| `tetrisSolvableValid_iff_init_safe`, `init_safe_iff_exists_solvesTetrisValid`, `init_mem_safe_iff_allValidFor` | full bidirectional equivalence + decidable init-safety bridge |
| **`init_mem_safe_of_invariant`**, `tetrisSolvableValid_of_invariant`, `_of_height_bounded_invariant` | one closed invariant ⇒ solvable (most reusable, search-free) |
| `AdversarialClosedCycle` + `.subset_safe` + `.init_solves`; `solvable_of_isInitCycle` | the working M2 tool + machine-checkable cycle endpoint |
| `F_finite`, `safeIterFinite`, `safeIterFinite_converges`; `safeIterFinite_subset_safe` (sound) + `safe_subset_safeIterFinite` (complete); `decideSafeFromUniverse` | computable GFP iteration, two-sided-correct |
| **`tetrisSolvableValidFor_iff_init_mem_safeIterFinite`**; the M2/M3/M4 iffs; `IsSolvabilityCertificate` + `solvable_of_certificate` | master decidable characterization — "building the Atlas = solving Tetris" |
| `LegalSequenceFrom.splice`, `adversarialTrace_*`, the pigeonhole cycle-existence lemmas | the survival + cycle-existence engine under the bridge |

### Layer 4 — Survival vocabulary (`Survival`)
| Theorem | Statement |
|---|---|
| `Policy`, `trace`, `SurvivesForever` | the survival abstraction |
| **`safe_invariant`** (+ `SurvivesForever_iff_exists_invariant`) | preserved loss-avoiding invariant ⇒ survives (universal template) |
| `ClosedCycle` + `closed_cycle_survives`; `exists_survivesForever_of_exists_init_cycle` | **local-to-global**: finite closed set+policy ⇒ infinite play (M2/M3 backbone) |

---

## 3. Promotion catalog (bubbling results out of `Experiments/`)

Promotion is **value-driven, not convenience-driven**: a result earns the green library
if it's a foundational fact about the dynamics any survival proof must respect, or is
demonstrably used by the spine. Cleanliness alone does not qualify. **DONE** = promoted;
**TODO** = pending; **DEMOTED** = moved back to research (not yet earned).

| Result | From | Target module | Status |
|---|---|---|---|
| Bag renewal: `countP_isSZ`, `countP_isI`, `renewal_ratio`, `drain_budget_ge_clearing_need` | `BagBurst` | `Combinatorics/BagBurst` | **DONE** (clean, Piece-only) |
| Hole-debt Lyapunov: `debt`, `debt_add_card_eq_sum_colHeight`, `holes_card_eq_debt`, `holes_card_le_place` (↑ place), `clearLines_debt_le` (↓ clear) | `HoleDebt` | `Invariants/HoleDebt` | **DONE** (split: SafeSet import pruned) |
| Surface-fiber: `colHeight_place_eq_of_colHeight_eq` (keystone) + refutation `place_holes_mono_within_hole_fiber_false` | `SurfaceFiber` | `Invariants/SurfaceFiber` | **DONE** (split) |
| WQO keystone: `domLE`, `place_domLE_mono`, `clearLines_domLE`, `domLE_trans` | `WqoCarrier` | `Invariants/Wqo` | **DONE** (primitives green; reduction `tetrisSolvableValid_of_wqo` stays in `Experiments/WqoCarrier`) |
| Hole-monotone FALSE (refutations): `place_holes_mono_false`, `clearLines_holes_le_false`; `safeLE`, `HoleyBoard` | `HoleyCarrier` | `Invariants/HoleyCarrier` | **DONE** (primitives green; reduction `tetrisSolvableValid_of_holey_wqo_basis` stays in `Experiments/HoleyCarrier`) |
| Piece-charge parity: `checkerCharge_classification` (only T charged) | `PieceCharge` | (was `Invariants/Charge`) | **DEMOTED** → `Experiments/PieceCharge` — clean/classic but **not used by any survival argument**; re-promote when it is |
| Roughness budget (Board-native): cert `not_isLost_of_holes_add_fullRows_add_roughness_le`; refutation `clearLines_uniform_shift_false` | `HoleyTopical` | `Invariants/RoughnessBudget` | TODO — cut its `SurfaceInvariant` import first; commit |
| Topical/max-plus: `dropMap_topical`, `dropMap_maxplus`, `eigen_cycle_survives_iff`, `oscDist_nonexpansive`, `contraction_bounded_roughness` | `TopicalTetris` | `Structure/Topical` | TODO — decouple `native_decide` evals (`fiveO_eigen`) |
| Energy game: `headroom`, `capacity_conservation` (master identity), reduction `tetrisSolvableValid_of_maxHeight_invariant`, `survival_forces_clears` | `EnergyGame` | `Survival/EnergyGame` | TODO — decouple `native_decide` witnesses |
| Lean-native Atlas constructor + M4 iffs + impossibility theorems (`closeStep`, `atlas_greatest_closed`, `exists_closed_atlas_iff_tetrisSolvableValid`, `safe_of_Lyapunov_function`, `not_winning_init`, `applyStep_S/Z_ne_empty`, `no_holes_zero_closed_table_contains_init`) | `FiveBagReset` | `Safety/Atlas` | TODO — split (promote this half; keep concrete cycle results) |
| Online-controller adapter: `OnlineControlMachine` + `.solves` | `OnlineControlMachine` | `Survival/OnlineControlMachine` | TODO — repoint import off `SurfaceInvariant`; commit |
| Skyline API + strategy reduction: `skyline` lemmas, `maxColHeight_applyStep_le`, `ReachableUnder`, `tetrisSolvableValid_of_strategy` | `SurfaceInvariant` | `Structure/Skyline` + `Survival/SurfaceStrategy` | TODO — extract from the 30k-line carrier zoo |

**Refutations are first-class library content.** `place_holes_mono_false`,
`clearLines_holes_le_false`, `clearLines_uniform_shift_false`,
`applyStep_S/Z_ne_empty`, `no_holes_zero_closed_table_contains_init` are *proven
theorems* that stop the team re-attempting dead routes — keep them green.

> **The split pattern (done for Wqo/Holey/SurfaceFiber/HoleDebt).** These files mixed
> `Invariants`-layer primitives with `Safety`-layer reductions, glued by a (mostly
> vestigial) `import Proofs.SafeSet`. The split puts the primitives/refutations in green
> `Invariants/` (SafeSet import removed) and leaves the route-specific
> `tetrisSolvableValid_of_*` reductions in `Experiments/` (importing the green primitives).
> `EnergyGame` was repointed to the green `Invariants` modules. Remaining `HoleyTopical`/
> `EnergyGame`/`FiveBagReset`/`SurfaceInvariant` promotions follow the same recipe.

---

## 4. Reorganization tree (now the actual layout)

```
Proofs.lean                  -- curated GREEN re-export root (native_decide-free)
Proofs/
  Model/         Config Piece Board Placement Bag Game                         ✓
  Combinatorics/ PieceGeometry BoardCount ColumnCount  BagBurst                ✓
  Invariants/    StepInvariants Gameplay GameplayExtra Holes StateSpace
                 Wqo HoleyCarrier SurfaceFiber HoleDebt                        ✓
  Survival/      Survival                                                      ✓
  Safety/        Safety Adversarial SafeSet SafeIterate SafeIterateFinite      ✓
  Archive/       AbstractSafe                                                ✓ (built by neither lib)
ProofsExperiments.lean       -- separate lib (route reductions, native_decide, Scratch/*)
Proofs/Experiments/          -- WqoCarrier/HoleyCarrier reductions, EnergyGame, PieceCharge,
                                carrier zoo (SurfaceInvariant, FiveBagReset), Scratch/*
scripts/check-green-clean.sh -- the green hygiene gate (no sorry/native_decide)
```
✓ = in place and building. `Theorems/` is **dissolved**; the root holds only `Proofs.lean`
(+ `ProofsExperiments.lean`). Not-yet-created (future): `Invariants/RoughnessBudget`,
`Safety/Atlas`, `Structure/{Skyline,Topical}`, `Api.lean` — these arrive with the deferred
promotions (§3). `Gameplay`+`GameplayExtra` are two files for now (merge optional).
The `Invariants/{Wqo,HoleyCarrier,SurfaceFiber,HoleDebt}` are the green primitive halves;
their `tetrisSolvableValid_of_*` reductions remain in `Experiments/`.

**Dependency layering (strict, bottom-up; verified against actual imports):**
```
Mathlib → Model → Combinatorics → Invariants → Survival → Safety → Structure → Api
```
Note `Survival` is **below** `Safety` (because `Adversarial` imports
`Proofs.Survival`); only the reduction modules that cite `safe_extract`
(`OnlineControlMachine`, `EnergyGame`, `SurfaceStrategy`) sit above `Safety`.

**Migration sequence (each step ends in a foreground green rebuild):**
1. ✅ Baseline axiom gate on the spine (`#print axioms`).
2. ✅ Split lakefile into `Proofs` + `ProofsExperiments`.
3. ✅ De-taint green root (drop the 4 experiment imports; add `Theorems.Holes`).
4. ✅ Promote the clean Piece-only `BagBurst`; later demoted `PieceCharge` (unearned).
5. ✅ Split `Wqo`/`HoleyCarrier` (primitives→`Invariants`, reductions→`Experiments`);
   relocate `SurfaceFiber`/`HoleDebt` to `Invariants`; repoint `EnergyGame`.
6. ✅ Dissolved `Theorems/`; moved all core files into `Model/Combinatorics/Invariants/
   Survival/Safety` (namespaces unchanged ⇒ theorem names stable; only import paths changed).
7. ⏳ Promote the remaining split-required results (`HoleyTopical`→`RoughnessBudget`,
   `EnergyGame` core, `FiveBagReset`→`Atlas`, `SurfaceInvariant`→`Skyline`) — but only when
   foundationally useful: the `EnergyGame capacity_conservation` / `SurfaceInvariant skyline`
   pair were evaluated and **deferred** (redundant / not-yet-load-bearing) on 2026-06-28.
8. ✅ Green hygiene gate `scripts/check-green-clean.sh` (fails on sorry/native_decide in
   the green tree); archived the `sorry` scaffold to `Archive/AbstractSafe.lean`.
9. ⏳ (Optional, deferred) merge `Gameplay`+`GameplayExtra`; write `Api.lean` façade;
   wire `scripts/check-green-clean.sh` into actual CI (no `.github/` workflows exist yet).

---

## 5. Archive vs keep-active, and the open crux

**Archive (floored — record only, built by neither lib):**
`AbstractSafe.lean` (3 real `sorry`s = crux #66/#72); the `FiveBagReset`
phase-decomposition program (`winning_init_iff_phase_decomposition` — *provably
empty* by its own `no_phase_decomposition`/`not_winning_init`); the
`SurfaceInvariant` carrier zoo (`isFlatFrontBandAt_*`, `reservoir*Surface_*` —
fixed-move-ordering lemmas that floor at all-orders accounting). **Keep** the
concrete reachable-cycle results `safeSolver_sevenBagCycle_reachable_closed_cycle`,
`sevenBagCycle_legal` (genuine M3) and all refutations.

**Keep-active (live research):** `OnlineReservoir` phase graph (open field
`PhaseGraphCompletion.frontier_step`); `SurfaceStrategy` (`ReachableUnder` +
`tetrisSolvableValid_of_strategy` — the structurally-correct positive route);
`EnergyGame`; the `Wqo/HoleDebt/RoughnessBudget/Topical` characterization routes.

### The single open crux (every route reduces to this)

> **There is no per-piece height regulator.** A non-I fill (e.g. flat O) raises
> height +2 with no clear (`oStep_strictly_raises`). So any bounded closed
> carrier must amortize that rise against the once-per-bag I-piece well-drain —
> and the unproven content is the *geometry of cashing that drain*: the
> un-instantiated `ReservoirGeometryCert.I_regulator_geometry` ≡
> `PhaseGraphCompletion.frontier_step` ≡ SurfaceInvariant crux #66/#72 ≡ the
> sorried realization half of `AbstractSafe.abs_simulation`. Four lenses, one
> obligation.

Two proven results pin its character:
- **Budget is sufficient, geometry is the obstruction:**
  `BagBurst.drain_budget_ge_clearing_need` (`14·bags ≤ 20·#I`).
- **A dominated-basis order cannot close it:** `HoleyCarrier.place_holes_mono_false`
  + `clearLines_holes_le_false` (holes non-congruent under place and clear).

⇒ Discharge it by **either** an explicit enumerated hole-aware atlas
(`HoleyCarrier.HoleyBoard` fed through the green `Safety/Atlas` decision
procedure) **or** a constructive online I-drain schedule (instantiating
`OnlineControlMachine`/`frontier_step`). Reframing relocates the crux; nothing
has dissolved it.
