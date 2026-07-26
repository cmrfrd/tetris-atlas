# Slot Algebra + Flush-Zone Verdicts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The redirect answer as green theorems (`Proofs/Invariants/SlotAlgebra.lean`) and the zone-game design space decided in-kernel (`Proofs/Experiments/FlushZoneGame.lean`, `native_decide` verdicts olj3/olj4/szt4/olj5/szt6/oljt5/oljsz6).

**Architecture:** SlotAlgebra defines per-column bottom/top profiles over `Piece.shapeUp` and proves the flush-landing exclusivity theorems by `decide` (F1: only vertical S/Z/T land on ±1 steps; O/L/J run the flat/±2 economy). FlushZoneGame mirrors the accepted `ZoneGame.lean` pattern: computable flush-only heights game, memoized AND-OR DFS with a bag-refill fuel horizon, verdict instances committed with their actual results. Spec: `docs/superpowers/specs/2026-07-12-slot-algebra-verdicts-design.md`.

**Tech Stack:** Lean 4 + mathlib (pinned), lake; `Std.HashMap` (already used by ZoneGame).

## Global Constraints

- Green target rules apply to SlotAlgebra only: no `sorry`, no `native_decide`, base axioms (plain `decide` on shape data is the established green pattern). FlushZoneGame lives in `ProofsExperiments` where `native_decide` is allowed; still no `sorry`.
- `lake` builds foreground only from `proofs/`; never SIGTERM; rebuild after every edit. `lake build ProofsExperiments` for the experiments target.
- Commits: one per task, staging only `proofs/`. Messages `proofs(slot-algebra): …` / `proofs(flush-zone): …`.
- Verdict instances are committed with their ACTUAL results (`= true` or `= false` as observed); expectations in the spec are hypotheses. Use `#eval` transiently during development to observe results (never committed); the committed artifact is the `native_decide` theorem. Record wall-times in PROGRESS.md.
- No instance above 6 columns (the zone-scale line; band-scale deciders are banned).
- Key verified facts:
  - `Rotation = Fin 4` (`Model/Piece.lean:25`); `Piece.shape` table at `Model/Piece.lean:40–60`; `shapeUp` flips via `maxT - cell.2`.
  - `Finset.min/max : Finset α → WithBot α`; use `.unbot' 0`.
  - ZoneGame's engine (`Experiments/ZoneGame.lean:60–190`): memoized DFS threading `Std.HashMap` through folds, fuel-structural recursion, verdicts `theorem … = true := by native_decide`. **Read it before Task 2 and mirror its recursion/termination pattern.**
  - The 2-wide shapeUp bottom/top table (derived from `Piece.shape`, to be re-verified by the consistency `decide`s):
    O r0 `[(0,1),(0,1)]`; I r0 `[(0,0)]×4`, I r1 `[(0,3)]`;
    S r0 `[(0,0),(0,1),(1,1)]`, S r1 `[(1,2),(0,1)]`;
    Z r0 `[(1,1),(0,1),(0,0)]`, Z r1 `[(0,1),(1,2)]`;
    T r0 `[(1,1),(0,1),(1,1)]`, T r1 `[(1,1),(0,2)]`, T r2 `[(0,0),(0,1),(0,0)]`, T r3 `[(0,2),(1,1)]`;
    L r0 `[(0,0),(0,0),(0,1)]`, L r1 `[(0,2),(0,0)]`, L r2 `[(0,1),(1,1),(1,1)]`, L r3 `[(2,2),(0,2)]`;
    J r0 `[(0,1),(0,0),(0,0)]`, J r1 `[(0,2),(2,2)]`, J r2 `[(1,1),(1,1),(0,1)]`, J r3 `[(0,0),(0,2)]`.
    (S/Z/I rotations 2/3 repeat 0/1; O all rotations equal.)

---

### Task 1: SlotAlgebra — the landing calculus and the exclusivity theorems

**Files:**
- Create: `proofs/Proofs/Invariants/SlotAlgebra.lean`
- Modify: `proofs/Proofs.lean` (import after `import Proofs.Invariants.PlinthShift`)

**Interfaces:**
- Consumes: `Piece.shapeUp` (`Model/Piece.lean`).
- Produces: `Piece.colBot`, `Piece.colTop` (`Piece → Rotation → ℕ → ℕ`), theorems `sStep_exclusive`, `zStep_exclusive`, `flatPair_receivers`, `twoStep_left_only_L`, `twoStep_right_only_J`, and the tops-value lemmas.

- [ ] **Step 1: Write the module**

```lean
import Mathlib
import Proofs.Model.Piece

/-!
# The slot algebra: flush-landing profiles and their exclusivity theorems

`colBot p r i` / `colTop p r i` — the lowest/highest occupied row of column
`i` in `p`'s drop profile. A rotation seats flush on a surface segment iff
the segment's height differences match the bottom profile; the landing
raises column `i` to `off + colTop + 1`. The exclusivity theorems decide,
once and for all, which pieces can consume which local surface shapes:

- a standing ±1 step admits ONLY vertical S/Z and vertical T (`sStep_exclusive`,
  `zStep_exclusive`) — so schedule designs that redirect O/L/J onto step
  zones are impossible (finding F1, 2026-07-12);
- 2-wide flat pairs admit exactly O, L (rot 1), J (rot 3)
  (`flatPair_receivers`);
- ±2 steps are the exclusive L/J currency (`twoStep_left_only_L`,
  `twoStep_right_only_J`), and both landings FLATTEN the step into a flat
  pair (the tops lemmas) — the O/L/J flat/±2-step economy.
-/

namespace Tetris
namespace Piece

/-- Lowest occupied row of column `i` in the drop profile (0 if empty). -/
def colBot (p : Piece) (r : Rotation) (i : ℕ) : ℕ :=
  (((p.shapeUp r).filter (fun c => c.1 = i)).image (fun c => c.2)).min.unbot' 0

/-- Highest occupied row of column `i` in the drop profile (0 if empty). -/
def colTop (p : Piece) (r : Rotation) (i : ℕ) : ℕ :=
  (((p.shapeUp r).filter (fun c => c.1 = i)).image (fun c => c.2)).max.unbot' 0

/-- A rotation is 2-wide when it occupies exactly columns 0 and 1. -/
def TwoWide (p : Piece) (r : Rotation) : Prop :=
  (∀ cell ∈ p.shapeUp r, cell.1 < 2) ∧ (∃ cell ∈ p.shapeUp r, cell.1 = 1)

instance (p : Piece) (r : Rotation) : Decidable (TwoWide p r) := by
  unfold TwoWide; infer_instance

/-- **A standing S-step admits only S and T.** Any 2-wide rotation whose
bottom profile is `(1, 0)` belongs to S (vertical) or T (rot 1). -/
theorem sStep_exclusive : ∀ (p : Piece) (r : Rotation), TwoWide p r →
    colBot p r 0 = colBot p r 1 + 1 → p = Piece.S ∨ p = Piece.T := by
  decide

/-- **A standing Z-step admits only Z and T** (mirror). -/
theorem zStep_exclusive : ∀ (p : Piece) (r : Rotation), TwoWide p r →
    colBot p r 1 = colBot p r 0 + 1 → p = Piece.Z ∨ p = Piece.T := by
  decide

/-- **A 2-wide flat pair admits only O, L, J.** -/
theorem flatPair_receivers : ∀ (p : Piece) (r : Rotation), TwoWide p r →
    colBot p r 0 = colBot p r 1 → p = Piece.O ∨ p = Piece.L ∨ p = Piece.J := by
  decide

/-- **A left-high ±2 step is exclusively L's** (rot 3). -/
theorem twoStep_left_only_L : ∀ (p : Piece) (r : Rotation), TwoWide p r →
    colBot p r 0 = colBot p r 1 + 2 → p = Piece.L := by
  decide

/-- **A right-high ±2 step is exclusively J's** (rot 1). -/
theorem twoStep_right_only_J : ∀ (p : Piece) (r : Rotation), TwoWide p r →
    colBot p r 1 = colBot p r 0 + 2 → p = Piece.J := by
  decide

/-! ## Tops: what each 2-wide landing leaves behind -/

/-- Vertical S preserves the S-step: tops `(2, 1)`. -/
theorem tops_vertS : colTop Piece.S 1 0 = 2 ∧ colTop Piece.S 1 1 = 1 := by
  decide

/-- Vertical Z preserves the Z-step: tops `(1, 2)`. -/
theorem tops_vertZ : colTop Piece.Z 1 0 = 1 ∧ colTop Piece.Z 1 1 = 2 := by
  decide

/-- T rot 1 consumes an S-step and leaves a Z-step: tops `(1, 2)`. -/
theorem tops_T1 : colTop Piece.T 1 0 = 1 ∧ colTop Piece.T 1 1 = 2 := by
  decide

/-- T rot 3 consumes a Z-step and leaves an S-step: tops `(2, 1)`. -/
theorem tops_T3 : colTop Piece.T 3 0 = 2 ∧ colTop Piece.T 3 1 = 1 := by
  decide

/-- O preserves the flat pair: tops `(1, 1)`. -/
theorem tops_O : colTop Piece.O 0 0 = 1 ∧ colTop Piece.O 0 1 = 1 := by
  decide

/-- L rot 1 on a flat pair leaves a left-high ±2 step: tops `(2, 0)`. -/
theorem tops_L1 : colTop Piece.L 1 0 = 2 ∧ colTop Piece.L 1 1 = 0 := by
  decide

/-- **L rot 3 consumes a left-high ±2 step and FLATTENS it**: tops `(2, 2)`. -/
theorem tops_L3 : colTop Piece.L 3 0 = 2 ∧ colTop Piece.L 3 1 = 2 := by
  decide

/-- J rot 3 on a flat pair leaves a right-high ±2 step: tops `(0, 2)`. -/
theorem tops_J3 : colTop Piece.J 3 0 = 0 ∧ colTop Piece.J 3 1 = 2 := by
  decide

/-- **J rot 1 consumes a right-high ±2 step and FLATTENS it**: tops `(2, 2)`. -/
theorem tops_J1 : colTop Piece.J 1 0 = 2 ∧ colTop Piece.J 1 1 = 2 := by
  decide

end Piece
end Tetris
```

Notes: if a `decide` over `∀ p r` is slow, split per piece (`∀ r, TwoWide
Piece.O r → …` etc., 7 lemmas each) — but the domain is 28 pairs of tiny
Finsets; it should be instant. If a tops lemma's value disagrees with the
table, trust `decide` (fix the theorem statement to the actual value and
re-derive the downstream design note; the shape table in the Global
Constraints is hand-derived).

- [ ] **Step 2: Wire the import** — in `proofs/Proofs.lean` after
`import Proofs.Invariants.PlinthShift`:

```lean
import Proofs.Invariants.SlotAlgebra
```

- [ ] **Step 3: Build** — `lake build`, expect success.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Invariants/SlotAlgebra.lean proofs/Proofs.lean
git commit -m "proofs(slot-algebra): the flush-landing calculus — step exclusivity (only S/Z/T), the O/L/J flat and ±2-step economy"
```

---

### Task 2: FlushZoneGame — the verdict engine

**Files:**
- Create: `proofs/Proofs/Experiments/FlushZoneGame.lean`
- Modify: `proofs/ProofsExperiments.lean` (add `import Proofs.Experiments.FlushZoneGame` after the ZoneGame import if present, else with the other experiment imports)

**Interfaces:**
- Consumes: `Piece` (`Model/Piece.lean`), `Piece.colBot/colTop` (Task 1, for consistency checks); the ZoneGame recursion pattern (`Experiments/ZoneGame.lean:60–190` — READ FIRST).
- Produces: `FlushZone.flushDead (w spread : ℕ) (bag : List Piece) (bags : ℕ) : Bool` and the shape table `FlushZone.shapesOf : Piece → List FShape` (Tasks 3–4 state verdicts against `flushDead`).

- [ ] **Step 1: Write the module**

```lean
import Mathlib
import Proofs.Invariants.SlotAlgebra

/-!
# The flush zone game — in-kernel verdicts on schedule design candidates

A zone of `w` columns receives its per-bag piece multiset in adversarial
order and must answer each piece with a FLUSH placement (hole-free seat:
the surface segment matches the rotation's bottom profile exactly) keeping
the normalized spread within a cap. No clears (the drain is global, not
zonal); states are normalized by subtracting the minimum height — the
base-shift quotient the plinth transport justifies.

`flushDead w spread bag bags = true` is an UNCONDITIONAL deadness fact:
no strategy survives even `bags` bags under all orders within the spread
cap. `= false` (alive at the horizon) is evidence only — closure must be
certified separately. Mirrors the accepted `ZoneGame.lean` verdict pattern
(memoized AND-OR DFS, `native_decide` instances at zone scale ≤ 6 columns).
-/

namespace Tetris
namespace FlushZone

/-- A rotation as per-column `(bottom, top)` row offsets; width = length. -/
abbrev FShape := List (ℕ × ℕ)

/-- Heights state (relative). -/
abbrev ZS := List ℕ

/-- The 2-to-4-wide drop profiles per piece (from `Piece.shape`, checked
against `Piece.colBot/colTop` below). -/
def shapesOf : Piece → List FShape
  | .O => [[(0,1),(0,1)]]
  | .I => [[(0,0),(0,0),(0,0),(0,0)], [(0,3)]]
  | .S => [[(0,0),(0,1),(1,1)], [(1,2),(0,1)]]
  | .Z => [[(1,1),(0,1),(0,0)], [(0,1),(1,2)]]
  | .T => [[(1,1),(0,1),(1,1)], [(1,1),(0,2)], [(0,0),(0,1),(0,0)], [(0,2),(1,1)]]
  | .L => [[(0,0),(0,0),(0,1)], [(0,2),(0,0)], [(0,1),(1,1),(1,1)], [(2,2),(0,2)]]
  | .J => [[(0,1),(0,0),(0,0)], [(0,2),(2,2)], [(1,1),(1,1),(0,1)], [(0,0),(0,2)]]

def pidx : Piece → ℕ
  | .O => 0 | .I => 1 | .S => 2 | .Z => 3 | .T => 4 | .L => 5 | .J => 6

def insSorted (n : ℕ) : List ℕ → List ℕ
  | [] => [n]
  | m :: ms => if n ≤ m then n :: m :: ms else m :: insSorted n ms

/-- Canonical multiset key for the remaining bag pieces. -/
def remKey (ps : List Piece) : List ℕ :=
  ps.foldl (fun acc p => insSorted (pidx p) acc) []

def minZ (h : ZS) : ℕ := h.foldl Nat.min (h.headD 0)

def normZ (h : ZS) : ZS := let m := minZ h; h.map (· - m)

def spreadZ (h : ZS) : ℕ := h.foldl Nat.max 0 - minZ h

def fitsAux (h : ZS) : ℕ → ℕ → FShape → Bool
  | _, _, [] => true
  | c, off, (b, _) :: rest =>
      h.getD c 0 == off + b && fitsAux h (c + 1) off rest

/-- Flush seat: in-bounds, and the segment matches the bottoms exactly. -/
def fits (h : ZS) (c : ℕ) (s : FShape) : Bool :=
  match s with
  | [] => false
  | (b0, _) :: _ =>
      decide (b0 ≤ h.getD c 0) && decide (c + s.length ≤ h.length) &&
        fitsAux h c (h.getD c 0 - b0) s

def applyAux : ZS → ℕ → ℕ → FShape → ZS
  | h, _, _, [] => h
  | h, c, off, (_, t) :: rest => applyAux (h.set c (off + t + 1)) (c + 1) off rest

def applyAt (h : ZS) (c : ℕ) (s : FShape) : ZS :=
  match s with
  | [] => h
  | (b0, _) :: _ => applyAux h c (h.getD c 0 - b0) s

/-- All flush responses to piece `p` keeping normalized spread ≤ cap. -/
def movesZ (spread : ℕ) (h : ZS) (p : Piece) : List ZS :=
  (shapesOf p).flatMap fun s =>
    (List.range h.length).filterMap fun c =>
      if fits h c s then
        let h' := applyAt h c s
        if spreadZ h' ≤ spread then some (normZ h') else none
      else none

abbrev Memo := Std.HashMap (ℕ × ZS × List ℕ) Bool

/-- Memoized AND-OR survival: adversary picks any remaining piece, player
picks any flush response; empty bag refills (consuming one fuel). Fuel 0 =
horizon reached alive. -/
def surv (spread : ℕ) (bag : List Piece) :
    (fuel : ℕ) → ZS → List Piece → Memo → Bool × Memo
  | 0, _, _, memo => (true, memo)
  | fuel + 1, h, [], memo => surv spread bag fuel h bag memo
  | fuel + 1, h, ps, memo =>
      let key := (fuel + 1, h, remKey ps)
      match memo.get? key with
      | some b => (b, memo)
      | none =>
          let res := (List.range ps.length).foldl
            (fun (st : Bool × Memo) i =>
              if !st.1 then st
              else
                let p := ps.getD i Piece.O
                let orRes := (movesZ spread h p).foldl
                  (fun (st2 : Bool × Memo) h' =>
                    if st2.1 then st2
                    else
                      let (b, m) := surv spread bag fuel h' (ps.eraseIdx i) st2.2
                      (b, m))
                  (false, st.2)
                (orRes.1, orRes.2))
            (true, memo)
          (res.1, res.2.insert key res.1)

/-- The verdict function: TRUE = no strategy survives `bags` bags under all
orders within the spread cap (unconditional deadness). -/
def flushDead (w spread : ℕ) (bag : List Piece) (bags : ℕ) : Bool :=
  !(surv spread bag (bags * (bag.length + 1)) (List.replicate w 0) bag
    (∅ : Memo)).1

/-! ## Shape-table consistency: `shapesOf` matches the green calculus. -/

/-- Every `shapesOf` entry agrees with `Piece.colBot`/`Piece.colTop` on some
rotation of the piece (bots and tops columnwise). -/
theorem shapesOf_consistent : ∀ p : Piece, (shapesOf p).all (fun s =>
    (List.range 4).any (fun rv =>
      s.length = ((p.shapeUp ⟨rv % 4, by omega⟩).image (fun c => c.1)).card ∧
      (List.range s.length).all (fun i =>
        (s.getD i (0, 0)).1 = Piece.colBot p ⟨rv % 4, by omega⟩ i ∧
        (s.getD i (0, 0)).2 = Piece.colTop p ⟨rv % 4, by omega⟩ i))) = true := by
  decide

end FlushZone
end Tetris
```

Adjustment notes:
1. **Termination of `surv`**: recursive calls sit inside `foldl` lambdas
   with structurally smaller fuel. If the equation compiler rejects it,
   mirror ZoneGame's exact recursion shape (`ZoneGame.lean:140–160` threads
   the memo through `List.foldl` the same way — copy its `termination_by`
   /structure verbatim). Fallback: replace the two folds with explicit
   fuel-indexed helper recursions in a `mutual` block,
   `termination_by (fuel, list length)`.
2. `shapesOf_consistent`'s Bool/Prop mixing: if the `∧`s inside `List.all`
   resist elaboration, restate with `decide (… = …)` per conjunct or split
   into 7 per-piece lemmas. The `⟨rv % 4, by omega⟩` Fin construction may
   simplify to `(rv : Fin 4)` coercion; use whichever elaborates. This
   lemma is a plain `decide` (green-legal check, but it lives here to keep
   SlotAlgebra free of the game).
3. `Std.HashMap` empty: `(∅ : Memo)` or `Std.HashMap.emptyWithCapacity`/
   `.empty` — match ZoneGame's spelling.

- [ ] **Step 2: Wire the import** — in `proofs/ProofsExperiments.lean`, add
with the other experiment imports:

```lean
import Proofs.Experiments.FlushZoneGame
```

- [ ] **Step 3: Build** — `lake build ProofsExperiments`, expect success.
Also `lake build` (green target must be unaffected).

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Experiments/FlushZoneGame.lean proofs/ProofsExperiments.lean
git commit -m "proofs(flush-zone): the flush-only zone game — memoized AND-OR verdict engine, shape table checked against the slot algebra"
```

---

### Task 3: verdict batch 1 — olj3, olj4, szt4

**Files:**
- Modify: `proofs/Proofs/Experiments/FlushZoneGame.lean` (append verdicts section)

**Interfaces:**
- Consumes: `flushDead` (Task 2).
- Produces: `olj3_verdict`, `olj4_verdict`, `szt4_verdict` theorems.

- [ ] **Step 1: Observe results transiently**

Add temporarily at the file end and build `ProofsExperiments` (foreground;
note wall-times):

```lean
#eval FlushZone.flushDead 3 6 [Piece.O, Piece.L, Piece.J] 4
#eval FlushZone.flushDead 4 6 [Piece.O, Piece.L, Piece.J] 4
#eval FlushZone.flushDead 4 6 [Piece.S, Piece.Z, Piece.T] 4
```

Record the three Booleans. Remove the `#eval` lines.

- [ ] **Step 2: State the verdicts with the observed values**

```lean
/-! ## Verdicts, batch 1 (results as observed; caps in the names' docstrings) -/

/-- The isolated 3-column {O,L,J} zone, spread ≤ 6, horizon 4 bags. -/
theorem olj3_verdict :
    flushDead 3 6 [Piece.O, Piece.L, Piece.J] 4 = true := by
  native_decide

/-- The 4-column {O,L,J} zone, spread ≤ 6, horizon 4 bags. -/
theorem olj4_verdict :
    flushDead 4 6 [Piece.O, Piece.L, Piece.J] 4 = true := by
  native_decide

/-- The 4-column {S,Z,T} pool, spread ≤ 6, horizon 4 bags. -/
theorem szt4_verdict :
    flushDead 4 6 [Piece.S, Piece.Z, Piece.T] 4 = true := by
  native_decide
```

**Replace each `= true` with the observed value** — if an instance is
ALIVE, the theorem is `… = false` and its docstring says "alive at the
horizon (evidence, not closure)". The names stay `*_verdict` either way.

- [ ] **Step 3: Build** — `lake build ProofsExperiments`, expect success.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Experiments/FlushZoneGame.lean
git commit -m "proofs(flush-zone): verdict batch 1 — olj3/olj4/szt4 decided in-kernel"
```

---

### Task 4: verdict batch 2 — olj5, szt6, oljt5, oljsz6

**Files:**
- Modify: `proofs/Proofs/Experiments/FlushZoneGame.lean` (append)

Same procedure as Task 3 with:

```lean
#eval FlushZone.flushDead 5 6 [Piece.O, Piece.L, Piece.J] 3
#eval FlushZone.flushDead 6 6 [Piece.S, Piece.Z, Piece.T] 3
#eval FlushZone.flushDead 5 6 [Piece.O, Piece.L, Piece.J, Piece.T] 3
#eval FlushZone.flushDead 6 5 [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z] 2
```

- [ ] **Step 1: Observe (one `#eval` at a time for the larger two; if an
`#eval` exceeds ~10 minutes foreground, abandon that instance, shrink
(`spread`−1 or `bags`−1) and re-observe; note the final parameters).**

- [ ] **Step 2: State `olj5_verdict`, `szt6_verdict`, `oljt5_verdict`,
`oljsz6_verdict` with observed values and the final parameters, mirroring
batch 1's format.**

- [ ] **Step 3: Build** — `lake build ProofsExperiments`, expect success
(note native_decide wall-times per instance).

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Experiments/FlushZoneGame.lean
git commit -m "proofs(flush-zone): verdict batch 2 — olj5/szt6/oljt5/oljsz6 decided in-kernel"
```

---

### Task 5: gates, findings, docs

**Files:**
- Modify: `proofs/PROGRESS.md`, `proofs/LIBRARY.md`

- [ ] **Step 1: Gates** — `scripts/check-green-clean.sh` (expect OK — the
green tree gained only SlotAlgebra, `decide`-only); axiom gate in the
scratchpad on `Piece.sStep_exclusive`, `Piece.flatPair_receivers`,
`Piece.twoStep_left_only_L` (expect exactly
`[propext, Classical.choice, Quot.sound]` or fewer); `lake build` and
`lake build ProofsExperiments` both green.

- [ ] **Step 2: PROGRESS.md** — insert under `## Last tick` (fill the
verdict table with actual results and wall-times):

```markdown
Tick (manual, 2026-07-12b) — **the slot algebra + flush-zone verdicts**.
F1 as GREEN THEOREMS (`Invariants/SlotAlgebra.lean`, all by `decide` over the
shape table): a standing ±1 step admits ONLY vertical S/Z/T
(`sStep_exclusive`/`zStep_exclusive`) — redirecting O/L/J onto step zones is
IMPOSSIBLE, killing the D4 rate fix; flat pairs admit exactly {O, L rot 1,
J rot 3}; ±2 steps are exclusively L (rot 3) / J (rot 1) and both landings
FLATTEN (`tops_L3`/`tops_J1`) — the O/L/J flat/±2-step economy is the only
cross-piece currency. `Experiments/FlushZoneGame.lean` (ZoneGame-pattern
memoized AND-OR, flush-only, shape table `decide`-checked against the green
calculus): verdicts —
| instance | params | verdict | wall |
|---|---|---|---|
| olj3 | w3 K6 B4 | <result> | <t> |
| olj4 | w4 K6 B4 | <result> | <t> |
| szt4 | w4 K6 B4 | <result> | <t> |
| olj5 | w5 K6 B3 | <result> | <t> |
| szt6 | w6 K6 B3 | <result> | <t> |
| oljt5 | w5 K6 B3 | <result> | <t> |
| oljsz6 | w6 K5 B2 | <result> | <t> |
Schedule implication: <one paragraph from the actual results — either the
zone split that survives, or the impossibility composition against the
9-column budget>.
```

- [ ] **Step 3: LIBRARY.md** — Layer-1 table row for SlotAlgebra
(`colBot/colTop + the five exclusivity theorems + tops lemmas — the flush
landing calculus`); §4 tree: `Invariants/` + `SlotAlgebra`; §5 keep-active
list: add FlushZoneGame with one line (in-kernel flush-zone verdicts,
ZoneGame pattern).

- [ ] **Step 4: Final builds** — `lake build` + `lake build
ProofsExperiments`, expect success.

- [ ] **Step 5: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/PROGRESS.md proofs/LIBRARY.md
git commit -m "proofs(flush-zone): findings F1-F3 + the verdict table recorded"
```

---

## Self-review notes (already applied)

- Spec coverage: Part A → Task 1; Part B core → Task 2; instances → Tasks
  3–4 (the spec's instance table, with the shrink-on-timeout rule); Part C
  → Task 5.
- Type consistency: `flushDead (w spread : ℕ) (bag : List Piece)
  (bags : ℕ) : Bool` used identically in Tasks 2–4; `FShape`/`ZS`/`Memo`
  defined once; `Piece.colBot/colTop` names match Task 1.
- The 1-wide loophole (vertical I vacuously matching 2-wide bottoms via
  empty-column defaults) is closed by `TwoWide`'s occupies-column-1
  conjunct.
- Known uncertainties flagged in-task: `surv` termination through folds
  (mirror ZoneGame; mutual fallback), `shapesOf_consistent` elaboration
  (per-piece split fallback), HashMap empty spelling, verdict results
  unknown until observed (by design).
```
