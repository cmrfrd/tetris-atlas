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
restores a clean spine. The repo is **`sorry`-free everywhere**: route B
(`Experiments/AbstractSafe.lean`) is a sorry-free conditional reduction (solvability ⟸
{per-(class,piece) realization (crux #66/#72), `α init ∈ A`}), not a scaffold.

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
| `Piece.colBot/colTop` + **`sStep_exclusive`/`zStep_exclusive`**, `flatPair_receivers`, `twoStep_left_only_L/right_only_J`, `tops_*` (`Invariants/SlotAlgebra`) | the flush-landing calculus: ±1 steps admit ONLY vertical S/Z/T; flat pairs exactly {O, L r1, J r3}; ±2 steps exclusively L/J and both FLATTEN — the cross-piece currency, all by `decide` |
| `place_horizS_step`/`place_horizZ_step`, `place_notchT/L/J`, `place_stepT_toZ/toS`, `place_flatT/L/J_lane`, `place_horizI_lane`, `place_pairL/J`, `place_fillL/J` (`Invariants/LaneCalculus`) | the **lane calculus**: every remaining local flush transition at arbitrary profiles (3-wide step lanes for horizontal S/Z incl. the roving-step flattening move, the three notch fills, the T alternator, local flat-3/flat-4 landings, the L/J pair economy) — with the window mechanisms, every slot-algebra landing shape now has its skyline transition law |
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
| `ShiftCertificate` + **`tetrisSolvableValid_of_shiftCertificate`** (`Safety/ShiftCertificate`); T1/T2 transport (`Invariants/BandShift`) | the translation quotient: `DebtCertificate` closure at base-0 band representatives — witnessed transitions (`place_debtBoard_bandLift`) and the generic drain (`drain_debtBoard_bandLift`) transport to every admissible base; base is a designer scalar predicate |
| `BandScheduleCert` + **`tetrisSolvableValid_of_bandSchedule`** (`Safety/BandSchedule`); debt-carry wrapper + bag-1 pack (`Invariants/BandMechanisms`) | the steady-state reduction: anchored S/Z/O closure discharged from the reproduction mechanisms + `place_debtBoard_of_flush`; open remainder = T/L/J/I schedule + bootstrap wiring + `okB` rate bookkeeping |
| `PlinthCert` + **`tetrisSolvableValid_of_plinthCert`** (`Safety/PlinthCert`); plinth transport T1′/T2′ + well plug (`Invariants/PlinthShift`) | the CORRECTED inhabitation target (findings D1/D2: v1 certs sound but uninhabitable — base never rises; row-0 hole blocks re-anchoring): immortal floor, mid-row drain, `ReanchorsTo` membership; open remainder = T/L/J/I schedule + boot tree + rate bookkeeping |
| **`safe_isGreatest`** / `safe_unique`, `isClosedOn_subset_safe`, **`safeMoves`** + `atlas_choice_mem_safeMoves`, `selection_survives` / `selection_update_mem` (`Safety/MaximalAtlas`) | **the global solver is unique as a RELATION, not as a function**: `safe` is the one greatest closed state set, `safeMoves` the one maximal table; every closed atlas is pointwise inside it, and solvers are its (many) selections |
| `sumCountAdv`, `clearingStepsAdv`, `clearAdv_step_le`, **`adversary_standing_inventory_floor`** | the occupancy floor made ADVERSARIAL: time-averaged inventory ≥ 2.4 cells against every piece order — the adversary can neither starve nor excuse the solver's stock |
| `window_same_piece_card_le_two` (BagCadence), `adversary_tetris_steps_subset`, **`adversary_two_tetris_per_seven`** | counted cadence: ≤ 2 of any piece per 7-window (Finset form) ⇒ **at most two tetrises in any seven adversarial placements** — pure cadence, no board reasoning |
| `sizeCountAdv_eq_card_filter`, `sizeCountAdv_four_le_I_card` | the counter↔filter bridge: the recursive size counter IS the windowed cardinality, and adversarial tetris count ≤ the I's dealt |
| `adversary_window_tetris_le_six`, `adversarialTrace_board_no_full_of_pos`, **`adversarialClosedCycle_period_tetris_le_five`** / **`closedCycle_period_tetris_le_five`** | tetris budget on cycle periods: **≤ 5 per period, matching the I supply exactly**; ≤ 6 per adversarial 35-window |
| `adversary_fullRows_card_le_four`, `adversary_three_clear_ILJ`, `mix_identity_adv`, **`adversary_mix_law`** | the clear-size mix transported adversarially: `10·(a₁+2a₂+3a₃+4a₄) + count = 4n` against every piece order; triples read `s n ∈ {I,L,J}` off the clear log |
| `adversary_six_le_count_of_clearing`, `adversarialTrace_board_no_full`, **`adversary_tetris_step_I`**, `adversary_card_empty_times_le` | adversarial mirrors: 6 banked cells before every clear; a 4-clear at step n READS `s n = I` off the clear log; board empty ≤ 1/5 of the time — all against every piece order |
| `clearedAdv`, `clearedAdv_ledger`, **`adversary_cannot_force_gt`**, `adversary_forces_ge`, **`adversary_forces_rate`** (`Safety/AdversarialRate`) | **no adversary can force the rate above 2.8** (the ceiling is a conservation law, piece choice is powerless against it), and every adversary pins any survivor at exactly 2.8 for free — the adversary's only lever is survival itself |
| `exists_draw_within_card`, `bagAt_add_card_eq_full`, **`every_piece_within_thirteen`** / `exists_I_within_thirteen` (`Safety/BagCadence`) | the 7-bag is **13-syndetic**: every piece in every 13-draw window; **max I-drought = 12 placements** — with `tetris_requires_I`, the worst case a tetris well must survive |
| `bagAt_card_of_full`, `not_full_of_full_close`, `exists_refill_between`, **`same_piece_three_apart`** | refills are exactly 7 apart; same-piece draws straddle a refill ⇒ **≤ 2 of any piece per 7-window** — tetris bursts come in pairs, never triples |
| `init_mem_safe_of_block_invariant` (generic n), **`init_mem_safe_of_five_bag_invariant`** / `tetrisSolvableValid_of_five_bag_invariant` | block coinduction at the cycle quantum: the invariant only has to describe boards at **five-bag boundaries** (35-step re-closure), matching the 35-quantum |
| **`iInter_safeIterate_subset_safe`** / **`safe_eq_iInter_safeIterate`** (`Safety/HorizonCompactness`) | **horizon compactness (König)**: the safe set = ⋂ of the finite-horizon iterates, unconditionally — a placement succeeding at cofinally many depths succeeds at all (finite move set + pigeonhole) |
| `solvable_iff_forall_horizon`, `solvable_or_finite_refutation`, **`solvable_iff_exists_invariant`** | solver exists ⟺ every finite horizon is winnable; either solvable or a finite kill certificate exists; and **the irreducible core: solvable ⟺ a closed invariant exists** — the witness is equivalent to the theorem, but compressible to any describable closed predicate |
| `solverReachable_finite`, **`solvable_iff_exists_finite_invariant`**, **`solvable_implies_bounded_atlas`** (`Safety/FiniteInvariant`) | **the Atlas may be taken FINITE, with an a-priori bound**: solvable ⟺ ∃ finite closed invariant, of size ≤ `2^207` (`2^200` in-field boards × 128 bags) |
| `applyStep_row_lt`, **`mem_safeIterate_of_headroom`**, **`init_mem_safeIterate_five`** (`Safety/HeadroomIterate`) | headroom-graded safety: `4k` rows of clearance ⇒ `safeIterate k` with NO strategy; init's 20 rows ⇒ **no kill certificate of depth ≤ 5** — tight (5 vertical pieces reach height 20) |
| `shapeUp_row_lt_two_rot_zero`, `mem_safeIterate_of_flat_headroom`, **`init_mem_safeIterate_ten`** | flat play doubles the horizon: rotation 0 of EVERY piece is ≤ 2 tall ⇒ **no kill certificate of depth ≤ 10**; rung 11 is the first requiring clears or spreading |
| `applyStep_group_bound`, `mem_safeIterate_of_two_group`, **`init_mem_safeIterate_twenty`** | the two-group flat schedule (cols 0–3 / 4–7, play the shorter): 40 rows of shared budget ⇒ **no kill certificate of depth ≤ 20** — the first certified rung with a decision |
| `narrowRot` + `shapeUp_narrowRot_bounds` (every piece fits 2×4), `applyStep_group_bound'` (parametric w×h), **`mem_safeIterate_of_three_group`**, **`init_mem_safeIterate_twentyfive`** | the three-group static-quota schedule (10+10+5 over cols 0–3/4–7/8–9): **no kill certificate of depth ≤ 25** — the disjoint-group ceiling |
| `clear_free_le_fifty`, `first_clear_by_fiftyone`, `adversary_first_clear_by_fiftyone` | **clear-free survival ends at placement 50** (mass: 4n ≤ 200); any surviving play — adversarial included — has cleared a row by placement 51; certificates past depth 50 must clear |
| `GameConfig.flat` (10×1), `flat_O_step_lost`, `init_not_safe_flat`, **`not_tetrisSolvableValidFor_flat`** (`Safety/CountingBarrier`) | **first machine-checked unsolvability instance** — and the counting barrier: every ledger theorem holds at 10×1 verbatim, yet 10×1 is unsolvable ⇒ no config-generic counting argument can prove the standard game solvable; the missing ingredient is geometric |
| `trace_eq_clears` / `trace_eq_thirtyfive_clears_fourteen`, `adversarialTrace_eq_clears` | cycle windows balance EXACTLY: `10·Δcleared = 4·Δn` between equal states ⇒ **every 5-bag cycle period clears exactly 14 rows**, cooperative and adversarial |
| `init_closedCycle_card_ge_thirtyfive`, `init_adversarialClosedCycle_card_ge_thirtyfive`; `adversary_tetris_filter_subset` / `adversary_tetris_card_le_I_card` | init-seeded M2/M3 cycles hold ≥ 35 states; tetris steps embed in I steps over ANY index set (window tetris count ≤ window I count) |
| `adversarial_clears_bracket`, **`adversarial_mass_band`**, `clearedAdv_mono` | the every-horizon 14-row clearing bracket and the [−140, +136]-cell occupancy band survive adversarial piece choice under a periodic loop witness |
| `adversarial_multi_period_mix`, `adversarial_multi_period_tetris_le` | against a periodic stream: mix weight-sum exactly `14j` and at most `3j` tetrises over `j` periods — the telescoped row budget survives adversarial piece choice |
| **`adversarialTrace_periodic`**, `adversarialTrace_period_multiples`, `adversarial_multi_period_clears` | with a 35-periodic piece stream, an adversarial 35-return iterates (determinism) and clears exactly `14j` rows per `j` periods — the missing periodic-stream hypothesis identified in batch 33, supplied |
| `trace_window_counts_ge_five` | the finest diversity observable: any FIVE consecutive placements already show five distinct occupancy values (the mod-10 residue steps through its whole cycle) |
| **`exists_survivor_iff_exists_nonempty_cycle`**, `ClosedCycle.globalize` (+`globalPolicy`) | the cooperative equivalence closed BOTH ways: a globally-valid trace-legal survivor exists ⟺ a nonempty well-formed closed cycle exists; `globalize` patches an on-set policy to global validity without touching the cycle |
| **`exists_closedCycle_of_survives`**, `closedCycleOfReturn`, `survivesForever_exists_return_from` | every surviving policy HIDES a closed cycle: the M2 artifact is extracted from any valid surviving trace (general-seed pigeonhole + the return-orbit constructor); with `closed_cycle_survives`, survival and cycle certificates are the same phenomenon |
| **`survivesForever_iff_bounded_evidence`** | survival is a bounded formula: the Π⁰₁ statement `SurvivesForever` is equivalent to one whose every quantifier is bounded by `2^207` — no logical strength beyond finite verification |
| **`survivesForever_return_within`**, `card_infield_times_bag` | quantitative: a surviving valid trace revisits a state within `2^207` steps (the live state space is exactly `2^200 · 2^7`), with the revisit a positive multiple of 35 — deciding cooperative survival is a finite computation |
| `survivesForever_iff_live_return_from`, `survivesForever_of_trace_return_from` | the survival characterization at EVERY well-formed seed (seed-general form of the init theorem) |
| **`survivesForever_iff_live_return`**, `survivesForever_of_trace_return`, `survivesForever_exists_return` | THE CHARACTERIZATION: cooperative survival ⟺ a finite live prefix ending in a state revisit — infinite play is equivalent to finite checkable evidence (pigeonhole on the in-field states for necessity) |
| **`survivesForever_of_perfect_clear_pair`** | survival from finite evidence: liveness on `[0, n₂)` + two aligned perfect clears ⟹ `SurvivesForever` — every hypothesis checkable by running the policy `n₂` steps |
| **`perfectClearCycle`** (+`mem_`, `_card_ge`) | the M2 constructor: a live PC-to-PC segment at matching bag phase packaged as a genuine `ClosedCycle` (≥ 35 states) — one checkable trace segment yields the certificate |
| **`survives_forever_of_perfect_clear_pair`**, `perfect_clear_pair_return`, `perfect_clear_spacing` | the perfect-clear route to M2: two aligned perfect clears ARE the same state (loop closed), and a live PC-to-PC segment at matching bag phase proves infinite play; PCs sit multiples of 5 apart |
| **`count_step_eq`**, `clear_requires_mass`, `tetris_requires_thirtysix` | per-step conservation: `count(m+1) + 10·size = count(m) + 4` — occupancy moves by `+4/−6/−16/−26/−36`; a `k`-clear needs `10k−4` standing cells, a tetris needs 36 |
| `trace_three_tetrises_span` | three tetrises span ≥ 7 placements on any legal trace — tetris bursts come at most in pairs (I-draws straddle two refills) |
| **`no_cycle_of_only_tetris_clears`**, `no_cycle_of_only_triple_clears`, `no_cycle_of_I_strictly_tetris` | solver-design impossibilities: a policy clearing only tetrises (or only triples) admits NO 35-return, and a policy playing I only for tetrises can never cycle — every loop needs idle I's and mixed clears |
| `cycle_bag_periodic`, `cycle_clear_size_periodic`, **`cycle_window_sizeCount_shift`** | the observable loop: bag, per-step clear size, and every windowed clearing statistic repeat verbatim one period later |
| `adversary_period_idle_I_ge_two` | the idle-I law is adversary-proof: even choosing the order, at least two of each period's five I's do lesser work — the row budget is order-independent |
| **`period_idle_I_ge_two`** | at least two of every period's five I's are idle: the row budget admits ≤ 3 tetrises but the balance deals exactly 5 I's — the I piece cannot serve as a pure tetris tool on any cycle |
| **`closedCycle_boards_ge_five`**, `closedCycle_bags_ge_seven`, `adversarialClosedCycle_diversity`, `trace_window_boards/bags_ge_*_from` | the M2 artifacts' diversity: every closed cycle (cooperative or adversarial) spans ≥ 5 boards and ≥ 7 bag states; diversity floors generalized to arbitrary well-formed seeds |
| `adversary_survivor_window_clears`, `adversary_survivor_window_events`, `adversarialTrace_count_lt` | the survivor's windowed clearing bracket (`4w ± 200` tenths) and event-rate bracket `[~10%, ~40%]` survive adversarial piece choice |
| `survivor_window_clears_ceiling`, `survivor_window_events_ceiling` | the matching ceilings: `10·Δcleared ≤ 4w + 200` and events ≤ `(4w+200)/10` — with the floors, the live event rate is pinned to `[~10%, ~40%]` at every scale |
| **`survivor_window_events_floor`**, `survivor_window_clears_floor` | the survivor's floors: any window of a live trace clears ≥ `(4w−200)/10` rows and clears on ≥ `(4w−200)/40` of its placements — clearing events cannot be rarer than ~1 in 10, sustainably |
| `adversary_tetris_frequency_cap` | the tetris frequency cap `⌊Δn/7⌋+2` survives adversarial piece choice — the four-clear reads I off the announced stream and the stream obeys the frequency law |
| **`big_clear_frequency_cap`** | triples + tetrises ≤ `3⌊Δn/7⌋ + 6` in any window: every 3+-clear rides an I/L/J and those run at `3/7 + O(1)` — disjoint fibers, union bound, three frequency laws |
| `iCount_frequency_law`, `tetris_frequency_cap` | the frequency law on the counters: `ΔiCount ∈ [⌊(Δn−7)/7⌋, ⌊Δn/7⌋+2]` and at most `⌊Δn/7⌋+2` tetrises at every horizon — both supersede the 35-block brackets |
| **`window_frequency_law`**, `prefix_le_one` | THE TRUE FREQUENCY LAW: in ANY window from ANY start, every piece appears within `[⌊(w−7)/7⌋, ⌊w/7⌋+2]` — frequency 1/7 with O(1) error at every scale, superseding the `[4/35, 6/35]` sandwich for large windows; pre-refill stretches repeat no piece |
| **`heavy_feed_requires_tall`**, `shape_col_fiber_not_big_of_SZO`, `shape_col_fiber_le_three_of_ne_I` | heavy column feeds require a TALL piece: 3+ cells into one column ⇒ I, L, J or T (vertical T carries a 3-column too — decide caught the I/L/J-only conjecture as FALSE); S, Z, O never feed past two |
| **`clear_count_le_shape_rows`**, `clears_le_two_of_O`, `clears_le_three_of_ne_I`, `shape_rows_le_three_of_ne_I`, `shape_rows_le_two_of_O` | the per-piece clear-cap ladder: clears ≤ the piece's row span; O clears ≤ 2, every non-I ≤ 3, four ⇒ I — completes the graded companion of `tetris_requires_I` |
| **`survivor_orbit_card_le`** | the orbit ceiling: a surviving orbit holds ≤ 2^207 states (embeds into in-field boards × bags) — with the 35-floor, every solution's minimal atlas is pinned to 35 ≤ \|orbit\| ≤ 2^207 |
| **`adversarial_holes_mono_of_dry`** | the drought hole-monotone is adversary-proof: clear-free adversarial windows never lower the hole count |
| **`place_hole_columns_le_three`**, `colHoles_place_eq_of_unfed` | hole damage is NARROW: at most three columns per move can gain holes (≤ 4 touched, one always flush) — as well as exactly priced |
| **`exists_flush_cell`**, `shape_has_bottom_cell` | every placement lands flush somewhere: some cell rests exactly on its column's stack top (or floor) — with the flush law, every move has at least one hole-free fed column; gap damage is never total |
| **`colHoles_place_eq_of_flush`** | flush landings are hole-neutral: one cell dropped exactly onto the old stack top makes the whole column's landing gap zero (no-burrow makes the flush cell the fiber bottom automatically) |
| **`colHoles_place_eq`** | the EXACT hole-genesis formula: a fed column's hole count grows by precisely its landing gap (fiber bottom − old height) — every placement's hole bill computed to the cell |
| **`tetris_no_new_holes`** | a tetris creates NO holes: the vertical I lands with zero gap, the merge leaves every column's hole count unchanged, and the clear can only lower it — the debt-free harvest |
| **`tetris_skyline_mass`** | a tetris shaves 36 rows of skyline: nine columns lose four each, the well loses nothing — the skyline-mass bill matching the 36-cell mass bill |
| **`tetris_step_skyline`** | the complete before/after skyline of the tetris in one theorem: the well's height exactly preserved AND every other column ≥ 4 lower, well identified across both clauses |
| **`tetris_well_height_preserved`** | the well's height survives the tetris UNCHANGED: I in, four rows out, the stack beneath untouched — a tetris is a pure harvest of the other nine columns, invisible in the well's own skyline |
| **`clear_step_unfed_colHeight_le`** | unfed columns sink by k in HEIGHT too: the skyline drain matching the cell drain — clearing planes the board down wherever the piece didn't build |
| **`holes_clearLines_le`**, `colHoles_clearLines_le`, `colHeight_clearLines_add_le`, `colRows_card_eq_colCount` | clearing never INCREASES holes: a k-clear drops every valid column's height by ≥ k while removing exactly k of its cells — the falling half of the hole-debt Lyapunov; with holes_place_ge the full structure is certified: place ↑, clear ↓ |
| **`trace_holes_ledger_cap`**, `trace_holes_add_count_le` | the debt priced by the ledger: at any live step holes + 4m ≤ 200 + 10·cleared — a slow-clearing game is forced nearly hole-free; hole-heavy boards are debt against future clears, exactly priced |
| **`holes_add_count_le_two_hundred`** | debt plus mass fits the board: holes + cells ≤ 200 on any in-field well-formed board — hole debt is capped by the free volume the mass leaves behind |
| **`trace_holes_mono_of_dry`**, `holes_step_ge_of_no_clear` | the hole debt is monotone along dry play: over any clear-free trace window total holes never decrease — debt accumulated in a drought is owed until a clear |
| **`holes_place_ge`**, `colHoles_place_ge` | placing never repairs a hole, COUNTED: every column's hole count is non-decreasing through the merge — the quantitative hole-debt monotone (only clearing lowers the debt) |
| **`block_contains`**, `first_block_contains` | stream-level block coverage: from any full-bag instant each piece appears within seven draws — the source of the trace-level opening-seven facts |
| **`trace_first_block_all_pieces`** | every piece — the skew pair included — is faced within the first block: the S/Z pressure starts before move eight in every game |
| **`trace_first_block_pieces_card`** | the opening seven are all different: any legal game's first block plays each tetromino exactly once |
| **`safe_wf_ncard_le`** | the well-formed safe set fits in 2^207 states: `safe` is infinite only through junk (ill-formed boards + empty-bag degeneracy) — the Atlas's true domain is astronomically finite, not infinite |
| **`safeMoves_union_ncard_le`**, `card_allValid_biUnion` | one node of the Atlas-as-relation never stores more than the alphabet: ≤ 240 safe answers per state over all pieces (the union of per-piece enumerations IS the 240 alphabet, kernel-counted) |
| **`safeMoves_ncard_le`**, `card_allValidFor_standard` | the maximal table is ≤ 36 wide at every node (kernel-counted enumeration: O has 36 valid placements, all others 34) — the Atlas-as-relation branches within a hard three-dozen bound |
| **`tetris_of_well`** | the anatomy is SUFFICIENT — the constructive tetris: rows [h, h+3] complete except column c₀ at height h + vertical I into c₀ ⇒ fullRows = exactly those four rows. With tetris_anatomy: the four-clear happens iff the well is presented and the I takes it |
| **`tetris_anatomy`** | THE CAPSTONE: at any four-clear, the finisher is the I and ONE well column carries every law at once — drop at its height, window = its four rows, full feed to it and nothing else, window rows = everything-but-it, all other columns ≥ 4 above it |
| **`bag_singleton_forced`** | the forced draw: a one-piece bag IS the singleton of the next draw — each block's seventh letter is determined by its first six (the adversary's per-block freedom is 6!, not 7!) |
| **`bagAt_card_clock`** | the bag-card clock from ANY seed: |bag| + n is constant mod 7 along every legal stream (draws tick down one, refills wind up seven) — generalizes the full-start clock |
| **`bagAt_refill_schedule`** | one bag reading fixes the whole refill calendar: bag of size c now ⇒ full at now + c and every 7 thereafter — the future block structure from a single observation |
| **`lost_not_absorbing`** | model documentation, kernel witness: a lost state can RETURN to life (overflow piled into full rows clears away) — why SurvivesForever quantifies over all times instead of a final verdict |
| **`hole_persists_step`** | a clear-free move carries every hole forward AS a hole (still empty, still covered) — the step form of the hole-debt monotone |
| **`hole_blocks_row`** | a hole freezes its row out of the clearing economy: an uncloseable row while the cover stands — only clears above can release it |
| **`hole_persists_place`**, `hole_never_filled_by_drop` | holes are PERMANENT until a clear: no piece cell ever lands below its column's height, so a covered empty cell stays empty across every merge — the board-level core of hole-debt (debt falls only on clears) |
| **`grounded_rotation_iff_not_skew`** | the flat-ground classification as one kernel IFF: a piece has an always-grounded rotation on the empty board ⟺ it is not S and not Z |
| **`I_grounded_on_empty`**, `O_grounded_on_empty` | the flat-bottomed pieces never bury a cell on virgin ground, in any rotation and column — of the seven, only the skew pair is forced to make holes on a flat floor |
| **`S_creates_hole_on_empty`**, `Z_creates_hole_on_empty` | the skew pieces are hole factories: dropped flat even on VIRGIN ground, S and Z bury a cell — the geometric seed of the S/Z pressure every survival argument must absorb (kernel witnesses) |
| **`dropped_fiber_contiguous`**, `shape_col_fiber_contiguous('), ` | the dropped piece is SOLID in every column: no placement sandwiches a gap between its own cells — holes are born only in the space UNDER the piece, never inside it |
| **`no_low_pair_five_high`** | the availability dichotomy: a board with no safe adjacent pair has ≥ 5 columns at height ≥ 17 — either headroom_move_exists applies somewhere or the board is already half towers |
| **`headroom_move_exists`**, `exists_narrow_rotation` | two low neighbours guarantee a headroom move for EVERY piece (each has a ≤2-wide rotation) — the availability half of the headroom reduction: keep one low two-column window and you never run out of safe moves |
| **`adversarial_survives_of_headroom`** | headroom beats every adversary: a solver that always drops ≥ 4 below the ceiling never loses against any stream — to solve Tetris it suffices to always HAVE a headroom move |
| **`lost_step_requires_high_column`**, `step_live_of_headroom` | death requires a high column: every top-out is a drop into a column within 4 rows of the ceiling — never an accident of a low board; the complete diagnosis of the loss event |
| **`survivesForever_of_headroom`** | perpetual headroom is perpetual survival: a policy that always drops onto columns ≥ 4 below the ceiling NEVER loses — the M1 goal reduced to maintaining one O(4)-checkable invariant |
| **`applyStep_safe_of_low_skyline`**, `clearLines_in_field` | the FULL move is safe under four rows of headroom (clearing only moves cells down) — a per-move safety certificate checkable in O(4) height reads |
| **`place_safe_of_low_skyline`**, `place_in_field_of_low_drop`, `dropOffset_le_of_heights` | four rows of headroom make a move top-out-free: if every touched column stands ≥ 4 below the ceiling, the merge stays in the field — the exact safety margin a solver must hold |
| **`dropOffset_mono`**, `colHeight_mono` | pieces land higher on fuller boards: heights and the drop offset ARE monotone in the board — the contrast pair to clears_not_monotone |
| **`clears_not_monotone`** | more material can mean FEWER clears (kernel witness): one extra cell lifts the landing above the row it would have completed — "a fuller board clears at least as much" is dead on arrival |
| **`clears_not_surface_determined`** | the complement, kernel-witnessed: identical skylines + identical landed cells can clear DIFFERENT rows (solid 2-stack vs holey twin) — the piece reads the surface, the clears read the holes; surface-only abstractions must lose the clear ledger |
| **`dropped_eq_of_colHeight_eq`**, `dropOffset_eq_of_colHeight_eq` | placement is SURFACE-DETERMINED: boards agreeing on the piece's column heights land the move identically — holes and everything below the skyline are invisible to the falling piece |
| **`place_unfed_colHeight_eq`** | placing leaves unfed columns' heights untouched — with the exact fed formula, the placement's skyline update is fully determined column by column |
| **`place_fed_colHeight_eq`** | the landing is EXACT: a fed column's post-move height = dropOffset + top piece cell + 1 — the complete height formula for every column the piece touches |
| **`dropped_above_own_column`** | hard drops never burrow: every dropped cell lands at or above its own column's stack top — the fundamental no-interleaving property of the hard-drop model |
| **`cleared_window_band`** | the per-window pinch: any live-endpoint window clears within one boardful of the exact 0.4/move line (4w − 200 ≤ 10·Δ ≤ 4w + 200), at every position and scale |
| **`tetris_window_cap`** | the burst allowance quantified: from any live moment a w-window holds ≤ 5 + w/10 tetrises — the 200-cell bank buys at most five beyond the steady one-in-ten rate |
| **`tetris_train_law`** | tetris trains are bank-limited: 40·(tetrises in window) ≤ count(start) + 4w — bursts are as long as the bank is deep and no longer |
| **`tetris_pair_mass_law`** | two four-clears at m < m' obey 76 ≤ count(m) + 4·gap: a tetris from a lean 36-cell board pushes the next ≥ 10 moves out — only a rich board double-fires |
| **`trace_tetris_relief`**, `adversarialTrace_succ_colHeight_le` | the relief law on live traces (a four-clear step shows two columns ≥ 4 apart) and the ≤4-per-step skyline cap on adversarial traces — growth is a property of the move, not of who chose it |
| **`trace_succ_colHeight_le`**, `applyStep_colHeight_le` | the skyline climbs ≤ 4 per step along any trace (clearing only lowers) — height spikes are rate-limited by one piece's geometry |
| **`place_colHeight_le`** | one move lifts the skyline by at most four: after placing, every column's height is within 4 of SOME pre-move column — a piece can bridge from a tall stack but cannot levitate above what stood |
| **`place_empty_low`**, `dropOffset_empty` | the opening geometry: on the empty board every piece falls to the floor (dropOffset = 0) and the first piece lies entirely in the bottom four rows |
| **`fed_column_height_le`**, `fed_column_height_le_three` | fed columns sit at or below the landing: no supporting stack pokes past where its cell rests (≤ dropOffset + cell-row ≤ dropOffset + 3) — the dual splitting of the board at every clear |
| **`clear_untouched_column_height_ge`** | unfed columns overtop the landing site by k: a k-clear's ≥ 6 untouched columns rise ≥ dropOffset + k — sharpens the +1 law; at k = 4 recovers the tetris well-depth geometry |
| **`adversarial_first_period_card`**, `adversarialTrace_ne_of_step_mod_ne` | the 35-state floor is adversary-proof: off-phase adversarial states are distinct and the 35 opening states never repeat, whatever the stream |
| **`no_tetris_on_flat`**, `tetris_relief_ge_four` | flat boards cannot tetris: every four-clear needs a column pair differing by ≥ 4 in height — flat-stacking strategies structurally forfeit the tetris; the skyline must be broken before it can be harvested |
| **`exists_tall_column`**, `colCount_le_colHeight` | mass forces height: a column holds at most colHeight cells, so some column of any well-formed board reaches ≥ count/10 — D cells cannot lie flatter than D/10 |
| **`perfect_clear_exact_rate`**, `perfect_clear_ge_five` | a perfect clear settles the ledger exactly (10·cleared = 4n, zero slack) and cannot happen before move five |
| **`perfect_clear_step_mod_five`**, `init_revisit_thirtyfive_dvd` | perfect clears keep the beat: the board is empty only at steps ≡ 0 mod 5, and a return to the exact initial state (empty + full bag) happens only at multiples of 35 — the reset is locked to the five-bag grid |
| **`survivor_orbit_eq`**, `survivor_orbit_card_le_return` | the orbit EQUALS the pre-return prefix image, and the first return time bounds the atlas: loop by step n₂ ⇒ at most n₂ states ever |
| **`return_states_determined`**, `return_tail_orbit_subset` | the rho shape at ANY period: after a return of positive gap p (always a multiple of 35), every later state is its phase-state in the first p-window — every survivor is transient prefix + finite wheel |
| **`cycle_tail_orbit_subset`** | the tail lives in the period's image: under a 35-return the loop is closed as a SET, not merely recurrent — membership packaging of the wheel |
| **`cycle_period_states_card`**, `cycle_states_determined` | the eventual loop has EXACTLY 35 states: the period's states are pairwise distinct (clocks disagree) and every tail state is one of them — under a 35-return the orbit = transient prefix + a 35-state loop (a general survivor loops on a positive multiple of 35) |
| **`cycle_actions_determined`**, `cycle_placement_periodic` | a cycle's play is a 35-letter word: past the loop point every move is one of the first period's 35 placements — the infinite game compresses to one word over the 240-letter alphabet |
| **`adversarial_cleared_pinch`**, `adversarial_first_clear_by_fifty_one` | the pinch and the move-51 first-clear deadline hold against every stream — per-step generalization of the bag-aligned adversarial floor |
| **`first_clear_by_fifty_one`**, `live_clear_events_floor` | the first clear falls in moves 3–51, both ends certified; and 4m ≤ 40·events + 200 — at least one clearing moment per ten moves past fifty |
| **`cleared_pinch`**, `live_clear_floor` | the clearing pinch at every live step: (4m − 200)/10 ≤ cleared ≤ 4m/10 — a 20-row window around the exact 0.4/move line at every horizon, not just bag boundaries |
| **`cleared_le_two_fifths`**, `clear_events_le_two_fifths` | the universal clearing speed limit: 5·cleared ≤ 2m and 5·events ≤ 2m at every horizon of every game — the 2.8-rows/bag cycle rate is a ceiling that binds from move one |
| **`tetris_count_le_tenth`**, `triple_count_le` | lifetime rate caps from move one (not just cycles): tetrises ≤ m/10 and triples ≤ 2m/15 at every horizon — each clear size's cell bill against the 4/move income |
| **`tetris_dry_opening`**, `earliest_tetris_needs_dry_opening` | a tetris taxes the whole opening: 10·cleared(m) + 36 ≤ 4m at any step-m four-clear — a tetris at steps 9–11 demands a perfectly clear-free game before it |
| **`adversarial_earliest_tetris_step`**, `adversarial_earliest_clear_law`, `adversarial_cleared_two_eq_zero` | the opening schedule is adversary-proof: 10k ≤ 4n + 4, no clears in the first two moves, no tetris before step nine — whatever the stream deals |
| **`earliest_tetris_step`**, `earliest_clear_law` | the opening clear schedule: 10k ≤ 4m + 4 at every step (singles from move 2, doubles 4, triples 6) and no game tetrises before step nine — the tenth placement is the earliest possible tetris |
| **`cleared_two_eq_zero`**, `rowCount_le_count` | no clears in the opening two moves: a cleared row needs six prior cells but the board holds only four after one placement — the earliest possible clear is the THIRD placement |
| **`cleared_one_eq_zero`**, `no_clear_on_empty` | the first move never clears: four cells cannot complete a ten-cell row from nothing — the base case of every clearing-rate induction, pinned |
| **`clear_rows_in_drop_window`**, `clear_untouched_column_height` | cfg-general k-clear geometry: every cleared row lies in [dropOffset, dropOffset+3], and every unfed column rises strictly above the drop offset — the stack the piece lands beside must reach through what it completes |
| **`tetris_well_height_cap`** | a tetris is a MID-BOARD event: on an in-field board the well sits at height ≤ 16 — the other nine columns must fit their four extra rows under the ceiling; no last-gasp tetrises off a full stack |
| **`tetris_well_depth`** | the well outruns the skyline by FOUR: at a tetris every other column is ≥ 4 rows taller than the well — the steepest possible local relief, forced at every four-clear |
| **`tetris_window_at_well_height`**, `I_shape_vertical_eq` | the window is the well-stack's crown: dropOffset = colHeight c₀ and the four clearing rows are exactly [colHeight c₀, +3] — the tetris geometry fully pinned to the board |
| **`tetris_window_base`**, `shape_rows_eq_of_card_four` | the tetris window sits exactly at the drop offset: fullRows = [dropOffset, dropOffset+3] — the clearing window starts precisely where the vertical I comes to rest on the well stack |
| **`tetris_rows_pre_shape`** | the tetris pre-board is DETERMINED on its window: before a four-clear, cell (c, r) of the four clearing rows is present iff c ≠ c₀ — the anatomy is unique up to the well column |
| **`card_valid_placements_O`**, `card_valid_placements_of_ne_O` | the 240 splits as 36 + 6×34: the O — the piece that can never clear more than two rows — is ironically the MOST placeable (2-wide footprint, 9 columns × 4 rotations) |
| **`card_valid_placements`**, `valid_iff_mem_enum` | the action alphabet has exactly 240 letters: kernel-counted valid placements on the standard board (piece × rotation × column), with a faithfulness lemma showing the range-10 enumeration misses nothing |
| **`survivor_orbit_card_ge`**, `trace_first_period_card`, `trace_ne_of_step_mod_ne` | every solution carries ≥ 35 states: off-phase states are provably DISTINCT (their clocks disagree), the 35 opening states never repeat, and a surviving orbit is squeezed between 35 and finite |
| **`survivor_finite_invariant`**, `survivesForever_of_invariant` | finite-invariant extraction, both directions: every cooperative survivor yields a FINITE step-closed lost-free set containing init, and any such set certifies survival — the finite invariant IS the survival certificate |
| **`survivor_orbit_finite`**, `visited_state_early` | a survivor's orbit is FINITE: once the trace returns, the pre-return prefix exhausts every state it will ever visit — the minimal atlas of any solution is a finite object |
| **`survivesForever_of_policy_agree`**, `trace_eq_of_policy_agree` | policy locality: the trace never consults the policy off its own orbit, and a surviving policy can be replaced by ANY policy agreeing on the visited states — the Atlas need only record the orbit; off-orbit values are free |
| **`adversarial_state_reveals_step_mod_thirtyfive`**, `adversarial_board_count_mod_ten`, `adversarial_board_count_even`, `adversarial_bag_card` | the clocks are adversary-proof: even-mass pruning, the mod-10 mass clock, the mod-7 bag clock and their CRT to n mod 35 hold in every adversarial game — across different solvers and different legal streams |
| **`trace_state_reveals_step_mod_thirtyfive`**, `trace_bag_card` | the state wears the cycle clock: bag size (mod-7 clock) + board mass (mod-5 clock) CRT to the step index mod 35 across any two policies — a state revisit can only happen a multiple of 35 steps later, the cycle quantum read off the state itself |
| **`trace_board_count_even`**, `trace_board_count_mod_ten`, `trace_board_count_determines_step_mod_five` | the mass clock: reachable boards have EVEN mass (odd-count boards unreachable — half of all configurations pruned from the Atlas for free); count ≡ 4n mod 10, so board mass reveals the step index mod 5 across ALL policies |
| **`bagAt_card`**, `bagAt_eq_full_iff` | the bag runs on the block clock: along any legal stream the bag holds exactly 7 − n%7 pieces, and is full precisely when 7 ∣ n — refill instants are observable from the bag alone |
| **`clear_step_column_drain`**, `placement_touched_columns_le_four`, `placement_untouched_columns_ge_six` | the general-k drain: at a k-clear every unfed column drops by exactly k (so held ≥ k), a placement touches ≤ 4 columns, hence ≥ 6 of the 10 columns pay the full bill on every clearing move |
| **`tetris_step_column_flow`** | the tetris column flow: a four-clear is a NO-OP on its well column (+4 feed, −4 clears) and a pure four-cell drain on each of the other nine — which therefore each held ≥ 4 cells going in |
| **`cycle_non_O_clear_floor`**, `cycle_O_step_clears_le`, `cleared_window_sum` | tall pieces must clear: per period the O's five appearances complete ≤ 10 rows, so ≥ 4 of the 14 cleared rows fall at non-O steps — square-only clearing cannot meet the rate law |
| **`cycle_heavy_feed_total_cap`** | global heavy-feed cap: per period ≤ 20 placements deliver 3+ cells into any single column — each plays a tall piece (I/L/J/T) and the bag deals five of each; supply-side companion of the tall-drop cap |
| **`cycle_tall_drop_total_cap`** | the GLOBAL tall-drop cap: per period at most 5 placements pour a full four-cell feed into any column at all — each plays the I and the bag deals exactly five; sharper than summing the ten per-column caps (30) |
| **`trace_tetris_feeds_single_column`**, `trace_clears_le_two_of_O`, `trace_clears_le_three_of_ne_I` | the feed law and the clear-cap ladder transferred to live traces: past the seed, a four-clear pours its whole feed into one column; O steps clear ≤ 2, non-I steps ≤ 3 |
| **`clear_free_column_feed_le`**, `clear_free_column_feed_eq` | clear-free intake is pure stacking (delivery = column growth exactly) and capped at 20 per column while alive — the height ceiling becomes an intake ceiling; summing columns recovers the 50-placement clear-free horizon |
| **`starving_column_caps_clears`**, `cycle_column_starvation_le` | column starvation laws: a window where column j receives nothing caps the WHOLE game's clears at j's starting holdings; on a cycle no column can starve for 35 placements |
| **`adversary_column_window_bracket`**, `adversary_starving_column_caps_clears`, `adversary_column_starvation_le` | starvation is adversary-proof: per-column 14⌊w/35⌋..+14 window bracket, a starving column caps the game's clears at its holdings, and no column goes unfed for 35 placements — regardless of who picks the pieces |
| **`adversary_column_load_exact`**, `adversary_tall_drop_column_cap`, `colDeliveredAdv_window`, `colDeliveredAdv_mono` | the exact column law is ADVERSARY-PROOF: against any 35-periodic stream a returning solver delivers exactly 14k cells per column per k periods, and pours a full four-cell feed into any one column at most 3 times per period |
| **`cycle_tetris_well_cap`**, `well_feed_four` | tetris-well rationing: per cycle period at most 3 tetrises may sink their well into any one fixed column (a fourth needs 16 of the column's exact 14-cell budget) — tetris wells must ROTATE across the board |
| **`tetris_feeds_single_column`** | the tetris feeds ONE column: at a four-clear the finishing I pours all four cells into the well column (colProfile = 4) and delivers 0 to the other nine — every tetris spends its whole feed budget on a single column |
| **`full_feed_requires_I`**, `shape_col_fiber_not_four_of_ne_I` | a full-column feed pins the I: only the I piece can pour all four of its cells into a single column |
| **`tetris_piece_vanishes`** | the I vanishes: every cell of a tetris's finishing piece lies in a cleared row — the vertical I is consumed whole, leaving no trace of itself |
| **`clearing_total_gaps_bracket`**, `rowCount_le_cols`, `isFull_of_rowCount_eq_cols` | one move closes between `k` and 4 gaps in total; rows hold ≤ `cols` cells and a `cols`-count row IS full |
| **`clearing_gaps_in_four_box`** | one move's entire clearing action is confined to a tetromino-sized 4×4 window: every pair of gaps closed differs by ≤ 3 in both coordinates |
| **`cleared_row_gaps_within_four_cols`** | gaps are horizontally local: any two cells missing from a completed row lie within three columns — the horizontal dual of the vertical span law |
| `trace_tetris_well`, `trace_cleared_row_pre_ge` | the anatomy theorems packaged at trace level: past the seed, the standing hypotheses (WF, validity, no-full) come for free |
| **`cleared_row_pre_ge`** | the unified per-row floor: each row of a `k`-clear held ≥ `5+k` cells beforehand (k=1: ≥6 … k=4: ≥9) — the other completed rows each claim a piece cell |
| **`tetris_gaps_share_column`**, `gap_filled_by_piece`, `I_four_rows_single_col` | THE WELL IS STRAIGHT: at a four-clear, the four one-cell gaps all sit in the same column — the vertical I's column. A tetris demands a clean 1-wide 4-deep well; nothing else can be true of the pre-board |
| **`tetris_row_missing_unique`**, `row_missing_unique` | the tetris WELL: each of a four-clear's rows misses exactly one column before the piece — four one-cell gaps awaiting the vertical I |
| **`tetris_rows_pre_nine`**, `rowCount_place_eq` | the tetris anatomy completes: each of a four-clear's rows held EXACTLY NINE cells before the vertical I supplied its tenth — one piece cell per row, forced by counting |
| **`four_clear_piece_rows_card`** | a tetris's finisher stands VERTICAL: the completing piece's four cells occupy four distinct rows — with `tetris_requires_I`, the finisher is an I in vertical orientation |
| **`four_clear_rows_eq_Icc`** | a tetris clears four CONSECUTIVE rows `[r₀, r₀+3]` — four distinct rows within a span of three have no other shape |
| **`fullRows_place_span_le_three`**, `mem_fullRows_place_has_piece_cell` | clears are vertically LOCAL: any two rows completed by one placement lie within 3 of each other — every completed row touches the piece, and a piece spans ≤ 4 rows |
| **`cleared_rows_pre_mass`**, `rowCount_of_isFull`, `sum_row_added_le_four` | the localized clear mass: the rows a `k`-clear completes held ≥ `10k − 4` cells BETWEEN THEM before the piece — the clearing mass must stand in the cleared rows themselves (localizes the 36-cell tetris floor) |
| **`cleared_row_was_six_tenths`**, `cleared_row_pre_count_ge` | clears must be prepared, never improvised: every cleared row held ≥ 6 of its 10 cells before the finishing piece (`cols − 4` generally) |
| **`cycle_column_profile_cap`** | the general rationing law: per period, placements delivering ≥ `p` cells to one column number ≤ `⌊14/p⌋` — heavy feeders of any column are rationed by its exact budget |
| **`cycle_tall_drop_column_cap`**, `colDelivered_window` | per period at most THREE placements pour their full four cells into any one column — a column's 14-cell budget can't absorb a fourth vertical I; window intake = sum of profiles |
| `sum_colDelivered` | consistency: the ten column ledgers sum to `4n` — the column brackets are a decomposition of mass conservation, not extra information |
| `adversary_column_load_bracket`, `colDeliveredAdv` (+`_ledger`) | the load-distribution law is adversary-proof: whoever picks the pieces, every column receives `cleared` to `cleared + 20` cells |
| **`cycle_column_window_bracket`**, `colDelivered_mono` | the per-column frequency law: every window of a cycle delivers `14⌊w/35⌋` to `14⌊w/35⌋+14` cells to every column — each column's intake runs at exactly 0.4/placement with ≤ one period of slack, at every position and scale |
| **`cycle_column_load_exact`**, `column_pair_balance` | on a cycle EVERY column receives exactly `14k` cells per `k` periods (the 140-cell total splits as 10 × 14, column by column); any two columns' deliveries differ by ≤ 20 on live traces |
| **`column_load_bracket`**, `colDelivered_ledger`, `colDelivered`, `colCount_le_rows` | THE LOAD-DISTRIBUTION LAW: on a live trace every column has received between `cleared` and `cleared + 20` cells — the clearing duty is billed to all ten columns equally, up to one board-height of slack (per-column ledger: held + lost-to-clears = received) |
| **`tetrisSolvableValid_implies_canonical_evidence_bounded`**, `not_tetrisSolvableValid_of_no_canonical_evidence` | solvability puts a certificate inside an explicit bounded search space — and its absence would REFUTE the mission: a finite (astronomical) computation decides this necessary condition |
| `adversarial_survives_return_within`, **`canonical_evidence_bounded`** | the mission's necessary condition is a BOUNDED search: a surviving valid solver revisits a state within `2^207` steps of the canonical game, with live prefix |
| **`tetrisSolvableValid_implies_canonical_evidence`**, `canonicalStream` (+`_legal`, `_periodic`) | a finite NECESSARY condition for the mission: if Tetris is solvable, some solver exhibits a live state-revisit against the canonical `IOSZTLJ`-repeated stream |
| **`adversarial_survives_iff_return`**, `adversarial_survives_exists_return`, `stream_periodic_iterate` | surviving a periodic adversary ⟺ finite evidence: against a legal 35-periodic stream, forever-liveness is equivalent to a live prefix ending in a revisit (pigeonhole + the quantum supplies the periodicity at the separation) |
| `adversarialTrace_periodic_T`, `adversarial_survives_of_return_T` (+`_tail_period_multiples_T`) | the general-period forms: determinism, iteration and finite-evidence survival for any `T`-periodic stream and `T`-return |
| **`adversarial_survives_of_return`**, `adversarial_perfect_clear_pair_return`, `adversarialTrace_tail_period_multiples` | adversarial survival from finite evidence: against a periodic stream, liveness on `[0, n+35)` plus a 35-return proves the trace lives forever; aligned PCs close the adversarial loop too |
| **`tetrisSolvable_iff_pattern_game`** | THE MISSION STATEMENT in combinatorial normal form: Tetris is solvable iff some solver survives every infinite sequence of bag permutations |
| **`solvesTetris_iff_forall_patterns`** | solving Tetris = surviving every permutation schedule: the adversary's strategy space reduced to its combinatorial core `(7!)^ℕ` |
| **`legalSequenceFrom_iff_exists_pattern_seq`** | the 7-bag process fully characterized: a stream is legal iff it is the concatenation of a sequence of injective 7-slot block patterns |
| **`pattern_seq_legal`**, `card_block_patterns` | the general stream constructor (any sequence of injective block patterns concatenates to a legal stream) and the branching factor: each block admits exactly `7! = 5040` orderings |
| **`exists_periodic_legal_stream`**, `periodic_stream_legal`, `sevenPattern` | periodic legal streams exist: any injective 7-slot pattern repeated forever is legal, and the canonical `I O S Z T L J` stream is 35-periodic — the loop-witness shape the adversarial multi-period theory consumes |
| **`legalSequenceFrom_iff_block_injective`**, `bagAt_eq_sdiff_of_block_injective` | LEGALITY IS BLOCK-INJECTIVITY: a stream is a legal 7-bag sequence iff it never repeats a piece within an aligned block — building legal witnesses (periodic streams for cycle certificates!) becomes trivial |
| `adversarialTrace_bag_eq_sdiff`, **`adversary_piece_available_iff`** | the adversary's entire freedom is a no-repeat-within-block rule: a piece is announceable iff not announced in the last `n mod 7` steps |
| `trace_bag_eq_sdiff`, **`trace_piece_available_iff`** | on cooperative traces: the bag = full ∖ pieces played this block, and a piece is drawable iff not played in the last `n mod 7` moves — bag membership fully explicit |
| **`bagAt_eq_sdiff`**, `bag_full_of_card_seven` | the bag content law from the start: on a full-bag stream, `bag(n) = full ∖ {draws since 7⌊n/7⌋}` — the bag state is a pure function of the current block's prefix; card 7 ⇒ full |
| `block_draws_injective`, **`block_image_eq_univ`** | within one bag block no piece repeats, and the block's image is EVERY piece — the permutation statement in its cleanest set form |
| **`refill_bag_sdiff`** | the bag is the complement of the block prefix: for `k ≤ 6` draws past a refill, `bag = full ∖ drawn` — the full bag CONTENT (not just size) is determined by the draw history |
| `refill_multiblock_balanced`, **`refill_window_bounds`** | aligned exactness: `7k` draws from a refill deal each piece exactly `k` times, and ANY aligned window is exact to ±1 (`[⌊w/7⌋, ⌊w/7⌋+1]`) — alignment removes the frequency slack |
| **`refill_block_balanced`** | a full-bag block is a permutation: from a refill, the next seven draws deal each piece EXACTLY once — the 7-bag's defining property recovered from legality alone |
| **`trace_window_grid_unique`** | the window grid is a BIJECTION: every (bag level, mass phase) cell is realised at exactly one position of any 35-window (∃! form: CRT existence + observable-phase uniqueness) |
| **`trace_window_phase_determined`**, `adversarialTrace_window_phase_determined` | the phase is observable: bag fill level + occupancy mod 10 jointly reconstruct a state's position within any 35-window (CRT of the clocks), in both settings |
| **`isClosedOn_grid_inhabited`** | the CRT grid: a closed Atlas set inhabits every cell of the 7 × 5 (bag level × mass phase) grid — the 35-floor is exactly the Chinese-remainder product of the two clocks |
| `closedCycle_stratum_ge_five`, `adversarialClosedCycle_stratum_ge_five` | the stratified floor on both M2 artifacts: ≥ 5 states at every bag fill level |
| `isClosedOn_count_stratum_ge_seven` | the dual stratification: ≥ 7 states at every mass phase the trajectory carries — the 7 × 5 grid decomposed from the other axis |
| **`isClosedOn_stratum_ge_five`** | the stratified floor: a closed Atlas set holds ≥ 5 states at EVERY bag fill level 1–7 — the 35-state floor decomposed into its seven bag strata of five |
| **`isClosedOn_boards_ge_five`**, `isClosedOn_bags_ge_seven`, `init_closed_atlas_diversity` | the M4 witness diversity: any closed Atlas set spans ≥ 5 distinct boards and ≥ 7 distinct bag states — a full bag-clock cycle lives inside every certificate |
| `adversarialTrace_window_boards_ge_five`, `adversarialTrace_window_bags_ge_seven` | the diversity floors are adversary-proof: both clocks tick whoever picks the pieces |
| **`trace_window_boards_ge_five`**, `trace_window_bags_ge_seven` | any 35-window shows ≥ 5 distinct boards (mass clock, residues mod 5) and ≥ 7 distinct bag states (bag clock, tight) — the state diversity of every five-bag stretch |
| **`periodMixes_card`**, `period_mix_mem_polytope` | the period-mix polytope has exactly 47 points (kernel-decided) and every cycle period's clear-size vector is one of them |
| `adversarial_window_mix_stationary`, `adversarial_window_tetris_le_three_stationary`, `adversarial_window_events_stationary` | adversarial homogeneity: every 35-window is a full period under a periodic stream — mix exactly 14, ≤ 3 tetrises, 4–14 events, whoever picks the pieces |
| **`cycle_window_piece_balanced_stationary`**, `cycle_window_mix_stationary`, `cycle_window_tetris_le_three_stationary`, `cycle_window_events_stationary` | statistical homogeneity: EVERY 35-window of a cycle is a full period — each piece exactly 5×, mix exactly 14, ≤ 3 tetrises, 4–14 clear events, from any starting point |
| **`cycle_window_clears_exact`**, `cycle_dry_spell_le_thirtyfour`, + adversarial mirrors | every 35-window of a cycle clears EXACTLY 14 rows from any point — dry spells last at most 34 placements (halves the pre-anchor 68 bound), both settings |
| `adversarialTrace_tail_periodic`, `adversarial_clears_bracket_stationary`, `adversarial_mass_diameter_sharp` | the adversarial mirrors of the anchor upgrade: sharp stationary bracket and `+136/−140` diameter under a periodic stream |
| **`cycle_clears_bracket_stationary`**, `cycle_tetris_density_stationary`, `cycle_mass_diameter_sharp` | tail periodicity makes EVERY point an anchor: the sharp anchored laws hold verbatim from any `m₀ ≥ n` — clears `[14⌊w/35⌋, +14]`, tetrises `≤ 3⌊w/35⌋+3`, occupancy `+136/−140` between any two horizons |
| `trace_tail_periodic`, `cycle_count_periodic`, **`cycle_piece_stream_periodic`** | one return makes the whole tail 35-periodic; a cooperative cycle deals itself a 35-periodic piece stream — exactly the `hper` hypothesis the adversarial theory requires: the two theories meet |
| `cycle_mass_diameter`, `adversarial_mass_diameter` | any two states on a cycle differ by ≤ 276 cells — the whole cycle lives in a 28-row occupancy corridor, in both settings |
| **`cycle_clears_stationary_bracket`** | shift-invariant: at every position and window length on a cycle, `14⌊(w−34)/35⌋ ≤ Δcleared ≤ 14⌊w/35⌋ + 28` — the 2.8/bag law holds in every window, not just from the entry point |
| `cycle_height_floor` | a heavy cycle keeps a tall column forever: boundary count > `140 + 10H` forces some column above `H` at every horizon (mass band × the volume bound) |
| **`init_closedCycle_reachable_minimal_orbit`** | the M3 bridge: a cycle through `init` contains a tight quantised sub-cycle every state of which is REACHED from the empty board by an explicit trace index |
| **`closedCycle_contains_minimal_orbit`**, `orbitCycleP`, `cycle_orbit_subset_period` | capstone: EVERY closed cycle contains a tight sub-cycle whose state count is exactly the trace's minimal period — positive, a multiple of 35, at most the ambient size, and every state trace-reachable from the seed (Nat.find minimality + generalized orbit construction) |
| `closedCycle_exists_return`, **`closedCycle_exists_period`** | pigeonhole: every closed cycle's trace returns within `card` steps, and the return period is a positive multiple of 35 bounded by `card` — every M2 artifact carries its own quantised loop |
| **`orbitCycle`** (+`_subset`, `_card`) | constructive: a five-bag return inside any `ClosedCycle` carves out a genuine 35-state sub-`ClosedCycle` — every field inherited, closure via the orbit window; every cycle with a minimal return CONTAINS the minimal certificate |
| **`cycle_orbit_subset`**, `trace_window_image_card_thirtyfive` | the forward orbit of a cycle is its first 35 states, and any 35 consecutive trace states are pairwise distinct — a minimal five-bag cycle visits EXACTLY 35 states |
| **`cycle_piece_bracket`**, `cycle_iCount_bracket` | on a cycle every piece's count over any horizon lies in `[5⌊Δ/35⌋, 5⌊Δ/35⌋+5]` — frequency exactly 1/7 with ≤ one period's error; the I-counter sharpened to slope 1/7 |
| **`no_pure_tetris_period`**, `no_pure_triple_period`, `period_mix_no_small_clears`, + adversarial mirrors | divisibility obstructions: `4∤14` and `3∤14` — no cycle clears exclusively via tetrises or exclusively via triples; the unique small-clear-free period mix is exactly 2 triples + 2 tetrises |
| `adversarial_dry_spell_le`, `adversarial_window_clears_fourteen` | the 68-placement dry-spell bound survives adversarial piece choice under a periodic loop witness |
| **`cycle_dry_spell_le`**, `cycle_window_clears_fourteen` | dry spells on a cycle last at most 68 placements — every 69-window contains a full aligned period and clears ≥ 14 rows; far tighter than the 50-placement capacity horizon and valid forever |
| **`period_clear_events_bounds`**, `multi_period_clear_events_bounds`, `adversary_period_clear_events_bounds` | silence dominates: only 4–14 of a period's 35 placements clear anything (≥ 21 silent), in both settings; `2·events ≥ 7j` over `j` periods |
| `mix_window_identity`, `cycle_size_density` | the mix identity over ANY window (`Δweights = Δcleared`, no cycle) and every-horizon caps on cycles: triples ≤ `4⌊Δ/35⌋+4`, doubles ≤ `7⌊Δ/35⌋+7`, singles ≤ `14⌊Δ/35⌋+14` |
| **`tetris_le_I_window`**, `tetris_bracket_any` | the windowed tetris–I embedding (each 4-clear step IS an I step, over any window past the seed) and the cycle-free tetris bracket `≤ 6⌊Δn/35⌋ + 6` at every horizon |
| **`window_bounds_any_length`**, `iCount_bracket_any` | window law at every length: `[4⌊w/35⌋, 6⌊w/35⌋+6]` of every piece per `w`-window — piece frequency sandwiched in `[4/35, 6/35]` at all scales; the I-counter bracket at every (unaligned) horizon |
| **`window_multiblock_bounds`**, `iCount_window_bounds` | any `35q`-window holds between `4q` and `6q` of every piece (pure cadence, no cycle); the I-counter bracket at every aligned horizon |
| `trace_period_piece_balanced`, `trace_multi_period_piece_balanced`, **`cycle_iCount_linear`** | balance at policy-trace level (single + multi period) and the I-counter linear law: `ΔiCount = 5j` over `j` periods on a periodic legally-drawn trace |
| **`closedCycle_multi_period_piece_balanced`**, `card_filter_range_add` | over `j` cycle periods each piece is played exactly `5j` times (periodicity re-arms the balance each lap); generic filtered-range splitting lemma |
| **`cycle_mass_band`**, `cycle_mass_periodic` | on a cycle the board occupancy is trapped in a 14-row band of its boundary value at every horizon (−140/+136 cells); exactly periodic at boundaries |
| **`cycle_clears_bracket`** | on a cycle every horizon's cleared count stays within 14 rows of the linear 2.8/bag law (sharper than the general 20-row deviation budget) |
| `cycle_tetris_density` | at most `3⌊Δn/35⌋ + 3` tetrises at every horizon — asymptotic tetris density ≤ 3/35 per placement |
| `multi_period_clears`, `multi_period_mix_fourteen` | the linear laws on cycles: `j` periods clear exactly `14j` rows with mix weight-sum exactly `14j` |
| **`multi_period_tetris_le`**, `multi_period_triples_le`, `multi_period_doubles_le`, `multi_period_singles_le` | ≤ `3j` tetrises / `4j` triples / `7j` doubles / `14j` singles over `j` periods — the tetris and triple caps telescope per period, strictly sharper than the aggregate mix |
| `sizeCount_eq_card_filter`, `iCount_eq_card_filter`, `sizeCount_window` | cooperative counter↔filter bridges + the window-difference form: cumulative counters and windowed caps are interchangeable |
| **`period_tetris_le_three`** (+ triples ≤ 4, doubles ≤ 7, singles ≤ 14; adversarial mirrors) | **at most THREE tetrises per cycle period** — 14 rows cannot absorb a fourth (`4·4 > 14`), sharper than the five-I supply |
| `sizeCount_mono` / `sizeCountAdv_mono`, **`period_mix_fourteen`** / **`adversary_period_mix_fourteen`** | the period mix: `Δa₁ + 2Δa₂ + 3Δa₃ + 4Δa₄ = 14` per cycle period, both settings — with 5 I's per period the per-period mixes form a small explicit polytope |
| `trace_board_no_full_of_pos`, `trace_tetris_step_I`, **`trace_window_tetris_le_six`** / `closedCycle_window_tetris_le_six` | ≤ 6 tetrises per 35 placements along any legal trace (from any seed), incl. the M2 trajectory |
| `trace_window_piece_bounds` / **`closedCycle_window_piece_bounds`** | the [4,6] window law on policy traces and the M2 artifact's own trajectory |
| `trace_bag_eq_bagAt` / `legalSequence_of_trace_draws`, **`closedCycle_period_piece_balanced`**, **`adversarialClosedCycle_period_piece_balanced`** | the balance theorem lands on the M2 artifacts: **every cycle period plays each piece exactly 5 times** — 5 T's (charge input), 5 I's (tetris cap); inside a cycle the adversary's freedom is only the ORDER of a fixed multiset |
| `isClosedOn_trace_forced_valid`, `isClosedOn_thirtyfive_dvd`, **`isClosedOn_card_ge_thirtyfive`** / **`init_closed_atlas_card_ge_thirtyfive`** | the quantum on the M4 object: a closed Atlas covering a real state — in particular any init-containing one — holds **≥ 35 states**. With the 2^207 bound: **the Atlas's size is pinned to [35, 2^207] by counting alone** |
| **`exists_legalSequenceFrom`** (BagCadence); **`closedCycle_card_ge_thirtyfive`**, **`adversarialClosedCycle_card_ge_thirtyfive`** | legal sequences exist from every nonempty bag (greedy stream) ⇒ **every closed cycle, cooperative or adversarial, holds ≥ 35 states** — the counting size floor on the M2 artifact |
| `bagAt_card_countdown`, `no_refill_no_repeat`, **`window_thirtyfive_le_six`** | the matching upper bound: ≤ 6 of any piece per 35-window (≤ six blocks touched, blocks never repeat a piece) — **every 35-window's piece counts lie in [4, 6]**, exactly 5 on cycle periods |
| `exists_block_hit`, `window_thirtyfive_ge_four`; `trace_exists_I_within_thirteen` / `closedCycle_exists_I_within_thirteen` | any 35-window holds ≥ 4 of every piece (no cycle hypothesis); the I-cadence lifted to policy traces and the M2 artifact |
| `mem_bagAt_succ_of_ne` / `mem_bagAt_of_not_drawn` (survival), `bagAt_full_iterate`, **`window_thirtyfive_balanced`** | **the balance theorem**: a 35-draw window with equal bag states deals EVERY piece exactly 5 times — head+tail jointly cover each piece once, four full blocks once each, and 7×5 = 35 leaves no slack. Delivers "#T = 5k per cycle period" for the charge theory |
| `mem_safe_of_bag_empty`; `safeMoves_subset_allValidFor` / **`safeMoves_finite`**; `every_piece_infinitely_often`; `clear_free_le_capacity`; `init_mem_safeIterate_tiny_two` | API round-out: the empty-bag degeneracy of `safe` documented; the maximal table has FINITE fibers; every piece recurs infinitely often (ω-syndeticity); clear-free ≤ cols·rows/4 config-generic; the ladder runs on `tiny` too |
| `LegalSequenceFrom.splice`, `adversarialTrace_*`, the pigeonhole cycle-existence lemmas | the survival + cycle-existence engine under the bridge |

### Layer 4 — Survival vocabulary (`Survival`)
| Theorem | Statement |
|---|---|
| `Policy`, `trace`, `SurvivesForever` | the survival abstraction |
| **`safe_invariant`** (+ `SurvivesForever_iff_exists_invariant`) | preserved loss-avoiding invariant ⇒ survives (universal template) |
| `ClosedCycle` + `closed_cycle_survives`; `exists_survivesForever_of_exists_init_cycle` | **local-to-global**: finite closed set+policy ⇒ infinite play (M2/M3 backbone) |
| `ClearRate.cleared`, `mass_ledger`, `init_ledger` | mass conservation along a trace: board mass + `cols`·cleared = `4`·placements — the deficit **is** the board |
| **`ClearRate.bags_sandwich`**, `lost_of_clear_deficit`, `not_survivesForever_of_rate_lt`, **`survival_forces_clear_rate`** | the **2.8 rows/bag law**: `2.8m − 20 ≤ cleared ≤ 2.8m`; sub-`2.8` clearing tops out, super-`2.8` is impossible, immortal play converges to exactly `28/10` |
| `ClearRate.play_bag_sandwich`; `average_clears_bounds`, `expected_clears_bounds` | the same law without a policy (every trajectory, adversarial included) and under any finite average / probability distribution |
| `ClearRate.deficit`, **`deficit_eq_mass`**, `window_bags_ge/le` (`ClearDeviation`) | the deficit **is** the board mass ⇒ over ANY window of `w` bags the clears land within 20 rows of `2.8w` |
| **`lost_of_sustained_shortfall`** / `sustained_shortfall_window_le`; `lost_of_dry_spell`; `bagClears_le_twentytwo` | death horizon: shortfall `β` lasts ≤ `20/β` bags; **no eight dry bags**; per-bag clears ∈ `{0..22}` (⇒ no marginal-variance bound) |
| `centered`, `centered_nonpos`, `abs_centered_le`, `abs_centered_sub_le`, `centered_div_sqrt_tendsto_zero`, `centered_sq_div_tendsto_zero` | never ahead, never 20 behind, max drawdown 20; the `√m`-scaled deviation and the long-run-variance estimator both vanish |
| **`covariance_sum_le`**, `variance_zero_of_bounded_partial_sums`, `variance_zero_of_nonneg_covariance`, `survival_forces_indep_variance_zero` | the whole covariance matrix sums to ≤ 400 ⇒ independent — indeed merely non-negatively-correlated — per-bag clearing with any spread is fatal |
| `count_mod_ten`, `five_dvd_of_count_eq`, `exists_count_eq_le`, `exists_recurrent_count` (`ClearRecurrence`) | occupancy ≡ `4n` mod 10 (a clock); equal occupancy ⇒ the window balanced EXACTLY and `5 ∣ Δn`; such a window occurs every ≤201 placements and some occupancy recurs forever |
| `bag_card_trace` (bag = mod-7 clock); **`thirtyfive_dvd_of_trace_eq`**, `thirtyfive_le_of_trace_eq` | **every closed cycle has length divisible by 35 placements = 5 bags** — an arithmetic lower bound on any M2 certificate, no geometry used |
| `sumCount`, `six_le_count_of_clearing`, **`standing_inventory_floor`**, `card_empty_times_le` (`StandingInventory`) | the occupancy FLOOR: time-averaged board mass ≥ **2.4 cells**; every clearing moment sits on ≥ 6 banked cells; the board is occupied ≥ **80% of the time** — empty-board play is impossible |
| `count_even`, `count_mod_ten_ne`, `five_dvd_of_count_eq_zero`, `clear_step_le`, `thirtysix_le_count_of_tetris` | cell-count arithmetic: always even, residue exactly uniform over `{0,2,4,6,8}`, empty only at `5 ∣ n`, a tetris needs 36 cells banked |
| `clearingSteps`, `fullRows_card_le_four`, **`clearingSteps_le` / `le_clearingSteps`** | the fraction of line-clearing pieces is trapped in `[1/10, 2/5]` (tetris-only vs singles-only). The *level* of the count is NOT determined by counting — that needs geometry |
| `sizeCount`, `iCount`, **`mix_identity`** / **`mix_law`** (`ClearMix`) | `a₁+2a₂+3a₃+4a₄ = cleared`, hence `10·(…) + occupancy = 4·pieces` — **one equation in four unknowns ⇒ the clear-size mix is free (3 d.o.f.)** |
| `iljCount`, `sizeCount_big_le_iljCount` | triples + tetrises ≤ #{I,L,J} placements (3/7 supply); timing constraint, does not bind the mix |
| `tetris_only_count_ge`, `singles_only_count_ge`; `sizeCount_four_le_iCount`, `ten_mul_sizeCount_four_le` | the corners: tetris-only must tetris ≥`0.7m−5` bags (70%, lifetime slack 5); singles-only clears on 40% of pieces. Only side-constraint: `a₄ ≤ #I` (≤1 tetris/bag), and it does not bind |
| `ClearableRow`, `maxClears`, **`fullRows_card_le_maxClears`**, `maxClears_le_two`, **`three_clear_requires_I_L_or_J`** | hard-drop ceiling per shape: SPAN is necessary not sufficient — S/Z/T span 3 rows but clear only **2**. Max clears: I 4, L/J 3, S/Z/T/O 2 |
| **`tetris_requires_I`** (+ `four_rows_only_I`, `dropped_rows_card`, `tetris_requires_I_trace`) | every cleared row must contain a cell of the drop ⇒ a four-row clear spans four rows ⇒ **only I can tetris** ⇒ ≤1 tetris per bag ⇒ tetris-only play needs a 70% I-conversion rate |
| `cleared_le_mul_clearingSteps`, `fullRows_card_le_of_count_le`, **`le_clearingSteps_of_max_clear`** | the tightness/frequency trade-off: max clear `K` ⇒ ≥`4/(10K)` of pieces must clear; an occupancy ceiling caps `K`. Tight board XOR rare clears |
| **`dry_runway_le`** / `lost_of_runway_overrun`, **`window_clears_ge_of_count`** | solver-facing design laws: exact dry runway `(200−count)/4` placements, and the sound pruning obligation "clear ≥ `(4w+count−200)/10` rows in the next `w`" |
| **`phase_mod_thirtyfive_of_trace_eq`** | `(board, bag)` determines the piece count mod 35 ⇒ **cycle search need only compare states 35k apart** |
| **`recovery_deadline`** (`ClearDeviation`), `card_mul_variance_le_of_nonneg_covariance` | *when* a solver must correct: no negative correlation for `L` bags ⇒ `L ≤ 400/σ²` — the debt must enter the policy on that timescale |
| `mass_ledger_of_trace`, `bag_card_trace_from`, `thirtyfive_dvd_of_trace_eq_from`; **`closedCycle_thirtyfive_dvd` / `_le`** | the 5-bag quantum from ANY start, applied to the M2 artifact: a `ClosedCycle` period is `35k` placements and never shorter than 35 |
| `exists_count_eq_le_of_step_five` | sharpened recurrence gap: an exact-balance window within **105** placements (was 201), via the mod-10 clock |
| `sum_grid_gap_eq`, **`stationary_covariance_budget`** / **`survival_stationary_lag_budget`** | the FULL lag profile: under stationarity `L·γ(0) + 2·Σ (L−l)·γ(l) ≤ 400` at every horizon — the spectral density at zero vanishes at rate `O(1/L)` |
| `offDiag_covariance_sum_le`, `exists_neg_covariance_of_horizon`, **`exists_correcting_pair`** | per-lag form: once `L > 400/σ²`, a NAMED pair of bags inside the window is negatively correlated — correction is scheduled, not asymptotic |
| **`adversarialClosedCycle_thirtyfive_dvd` / `_le`** (`Safety/CycleQuantum`) | the 5-bag quantum on the ADVERSARIAL M2 artifact (the `TetrisSolvable` side): both clocks survive the adversary, who picks which piece but not how many cells |

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
                 Wqo HoleyCarrier SurfaceFiber HoleDebt BandShift
                 BandMechanisms PlinthShift SlotAlgebra LaneCalculus           ✓
  Survival/      Survival Lasso ClearRate ClearDeviation ClearRecurrence
                 ClearMix StandingInventory                                    ✓
  Safety/        Safety Adversarial SafeSet SafeIterate SafeIterateFinite
                 SkylineInvariant ShiftCertificate BandSchedule PlinthCert
                 CycleQuantum MaximalAtlas AdversarialRate CountingBarrier
                 BagCadence FiniteInvariant HeadroomIterate
                 HorizonCompactness                                            ✓
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
   the green tree); rewrote route B (`Experiments/AbstractSafe.lean`) from a 3-sorry
   scaffold into a sorry-free conditional reduction → repo is now `sorry`-free everywhere.
9. ⏳ (Optional, deferred) merge `Gameplay`+`GameplayExtra`; write `Api.lean` façade;
   wire `scripts/check-green-clean.sh` into actual CI (no `.github/` workflows exist yet).

---

## 5. Archive vs keep-active, and the open crux

**Floored — record only (in `Experiments/`, sorry-free):**
`AbstractSafe.lean` is route B as a sorry-free conditional reduction (the open content
is now explicit hypotheses, not `sorry`); the `FiveBagReset`
phase-decomposition program (`winning_init_iff_phase_decomposition` — *provably
empty* by its own `no_phase_decomposition`/`not_winning_init`); the
`SurfaceInvariant` carrier zoo (`isFlatFrontBandAt_*`, `reservoir*Surface_*` —
fixed-move-ordering lemmas that floor at all-orders accounting). **Keep** the
concrete reachable-cycle results `safeSolver_sevenBagCycle_reachable_closed_cycle`,
`sevenBagCycle_legal` (genuine M3) and all refutations.

**Keep-active (live research):** `FlushZoneGame` (in-kernel flush-zone verdicts, ZoneGame pattern — the schedule design-space cartography; + scheduled game `flushDeadP`, drain-invisibility `normZ_shift`, I-pool verdicts: the ≤6-col zone paradigm is exhausted, all dead and drain-robust); `OnlineReservoir` phase graph (open field
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
> `realization` hypothesis of `AbstractSafe.tetrisSolvableValid_of_realization`.
> Four lenses, one obligation.

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
