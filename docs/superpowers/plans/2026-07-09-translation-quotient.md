# Translation Quotient (BandShift + ShiftCertificate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two new green Lean modules that let `DebtCertificate` closure be proven once at base-0 band representatives: `Proofs/Invariants/BandShift.lean` (well-anchored lift + transport lemmas T1/T2 + lifted flat witnesses) and `Proofs/Safety/ShiftCertificate.lean` (`ShiftCertificate` structure + `tetrisSolvableValid_of_shiftCertificate`).

**Architecture:** Transition transport — T1 transports *witnessed* `place`-equalities on `debtBoard`s from the base-0 representative to every band-lift `c`; T2 is the generic vertical-I well drain (`4 ≤ c`: pattern and rep-hole unchanged, base −4). `ShiftCertificate` packages a relative family `Q` + designer base predicate `okBase`, and reduces into `DebtCertificate` via a `toDebtCertificate` constructor. Spec: `docs/superpowers/specs/2026-07-09-translation-quotient-design.md`.

**Tech Stack:** Lean 4 + mathlib (pinned via `proofs/lean-toolchain`), lake. All work in `namespace Tetris` (board content in `namespace Board`).

## Global Constraints

- Green target: **no `sorry`, no `native_decide`, no new axioms** — every new theorem must depend on exactly `[propext, Classical.choice, Quot.sound]` (some may use fewer).
- `lake` builds are **foreground only**; never SIGTERM an in-flight build; re-run `lake build` after ANY edit/move before proceeding. Working dir for all lake commands: `/Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas/proofs`.
- Commits: one commit per task, staging **only files under `proofs/`** (never `Cargo*`, `docs/`, `.claude/`). Message style: `proofs(band-shift): <what>` / `proofs(shift-certificate): <what>`.
- Lean docstrings close at the first `-/` — never write `word-/word` inside a `/-- ... -/` comment.
- Line width ≈ 100 chars, match surrounding style (docstring per theorem, `by` proofs, `omega`-heavy arithmetic).
- Key existing identifiers (verify with grep before use; all already green):
  - `Board.debtBoard cfg h ho` (`Proofs/Invariants/HoledSkyline.lean:566`), simp lemmas `debtBoard_none`/`debtBoard_some`, `colHeight_debtBoard (hcov : ∀ x, ho = some x → x.2 + 1 < h x.1)` (:578)
  - `Board.skyline`, `mem_skyline'` (`Skyline.lean:45`), `colHeight_skyline`, `clearLines_skyline (hm) (hm0)` (:149)
  - `mem_holedSkyline` (`HoledSkyline.lean:38`), `clearLines_holedSkyline_of_le (hxcols) (hm) (hm0) (hxm)` (:244)
  - `Placement.dropOffset_eq_sup` (`Model/Placement.lean:144`), `dropped_eq_image` (:149), `place_eq_union_dropped` (:136), `applyStep_eq_clearLines_place` (:140)
  - `Piece.shapeUp_exists_bottom : ∀ p r, ∃ c ∈ p.shapeUp r, c.2 = 0` (`Confluence.lean:191`)
  - `shapeUp_vertI'`, `shapeUp_horizS`, `shapeUp_horizZ` (used at `HoledSkyline.lean:803,615,652`)
  - flat witnesses: `place_O_flat` (:730), `place_vertI_flat` (:798), `place_flatT` (:870), `place_flatL` (:940), `place_flatJ` (:1010), `place_horizS_flat_eq_holedSkyline` (:604), `place_horizZ_flat_eq_holedSkyline` (:641)
  - `DebtCertificate` + `tetrisSolvableValid_of_debtCertificate` (`Safety/SkylineInvariant.lean:126,147`)
  - `WellFree`, `AvoidsWell`, `drainPl`, `drainPl_valid`, `applyStep_eq_place_of_wellFree` (`Safety/SeamBridge.lean:37,40,92,94,76`)
- The "test" cycle for Lean tasks is the build: an unproven/ill-stated theorem fails `lake build`; a task is done only when the full green build passes. Transient `sorry` while iterating is fine but must never be committed.

---

### Task 1: BandShift skeleton — definitions + membership algebra

**Files:**
- Create: `proofs/Proofs/Invariants/BandShift.lean`
- Modify: `proofs/Proofs.lean` (add import after line 34, `import Proofs.Invariants.HoledSkyline`)

**Interfaces:**
- Consumes: `Board.debtBoard`, `mem_skyline'`, `mem_holedSkyline` (all via `import Proofs.Invariants.HoledSkyline`).
- Produces (used by Tasks 2–6): `Board.bandLift (w c : ℕ) (h : ℕ → ℕ) : ℕ → ℕ`, `Tetris.holeLift (c : ℕ) : Option Coord → Option Coord`, `Board.mem_debtBoard`, `Board.bandLift_well`, `Board.bandLift_ne`, `Board.bandLift_zero`, `holeLift_none`, `holeLift_some`.

- [ ] **Step 1: Write the module**

```lean
import Proofs.Invariants.HoledSkyline

/-!
# Well-anchored band lifts: the translation-quotient transport layer

The translation symmetry of the debt-1 board algebra is *well-anchored*: one
well column is pinned at height `0` (which blocks every line clear) and the
remaining band columns shift up by `c`. `bandLift` lifts a base-0 profile to
band base `c`; `holeLift` rides the (optional) buried cell along. The
transport theorems (`place_debtBoard_bandLift`, `drain_debtBoard_bandLift`)
let a `DebtCertificate`-style closure obligation be proven once at the
representative and reused at every base — see
`Proofs/Safety/ShiftCertificate.lean` for the certificate that packages this.
-/

namespace Tetris

/-- Transport an optional buried cell up by `c` rows. -/
def holeLift (c : ℕ) : Option Coord → Option Coord :=
  Option.map (fun x => (x.1, x.2 + c))

@[simp] theorem holeLift_none (c : ℕ) : holeLift c none = none := rfl

@[simp] theorem holeLift_some (c : ℕ) (x : Coord) :
    holeLift c (some x) = some (x.1, x.2 + c) := rfl

namespace Board

/-- The well-anchored band lift: column `w` pinned at `0`, every other
column raised by `c`. `bandLift w 0 h = h` exactly when `h w = 0`. -/
def bandLift (w c : ℕ) (h : ℕ → ℕ) : ℕ → ℕ :=
  fun j => if j = w then 0 else h j + c

@[simp] theorem bandLift_well (w c : ℕ) (h : ℕ → ℕ) : bandLift w c h w = 0 :=
  if_pos rfl

theorem bandLift_ne (w c : ℕ) (h : ℕ → ℕ) {j : ℕ} (hj : j ≠ w) :
    bandLift w c h j = h j + c := if_neg hj

theorem bandLift_zero {w : ℕ} {h : ℕ → ℕ} (hw : h w = 0) :
    bandLift w 0 h = h := by
  funext j
  by_cases hj : j = w
  · subst hj; simp [bandLift, hw]
  · simp [bandLift, hj]

/-- Uniform membership for debt-≤1 boards, covering both `Option` cases. -/
theorem mem_debtBoard {cfg : GameConfig} {h : ℕ → ℕ} {ho : Option Coord}
    (p : Coord) :
    p ∈ debtBoard cfg h ho
      ↔ (∀ x, ho = some x → p ≠ x) ∧ p.1 < cfg.cols ∧ p.2 < h p.1 := by
  cases ho with
  | none => simp [debtBoard_none, mem_skyline']
  | some x =>
      rw [debtBoard_some, mem_holedSkyline]
      constructor
      · rintro ⟨hne, hc, hr⟩
        exact ⟨fun y hy => by cases hy; exact hne, hc, hr⟩
      · rintro ⟨hne, hc, hr⟩
        exact ⟨hne x rfl, hc, hr⟩

end Board
end Tetris
```

- [ ] **Step 2: Wire the import**

In `proofs/Proofs.lean`, after `import Proofs.Invariants.HoledSkyline` (line 34), add:

```lean
import Proofs.Invariants.BandShift
```

- [ ] **Step 3: Build**

Run (from `proofs/`): `lake build`
Expected: `Build completed successfully` (≈8290 jobs, only pre-existing lint warnings). If a proof step fails, fix within this task — do not proceed red.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Invariants/BandShift.lean proofs/Proofs.lean
git commit -m "proofs(band-shift): well-anchored band lift and hole transport — definitions and membership algebra"
```

---

### Task 2: drop-offset shift lemma

**Files:**
- Modify: `proofs/Proofs/Invariants/BandShift.lean` (append inside `namespace Board`, before `end Board`)

**Interfaces:**
- Consumes: Task 1 defs; `Placement.dropOffset_eq_sup`, `colHeight_debtBoard`, `Piece.shapeUp_exists_bottom`.
- Produces (used by Tasks 3–4): `Board.sup_sub_add_shift`, `Board.dropOffset_debtBoard_bandLift` with the exact signature below.

- [ ] **Step 1: Write the sup-arithmetic helper and the shift lemma**

```lean
/-- Truncated-subtraction sup shift: raising every value by `c` raises the
sup of `value − row` by exactly `c`, provided some cell sits at row `0`. -/
theorem sup_sub_add_shift (s : Finset Coord) (f : Coord → ℕ) (c : ℕ)
    (hbot : ∃ cell ∈ s, cell.2 = 0) :
    s.sup (fun cell => f cell + c - cell.2)
      = s.sup (fun cell => f cell - cell.2) + c := by
  obtain ⟨c₀, hc₀, hc₀0⟩ := hbot
  refine le_antisymm (Finset.sup_le fun cell hcell => ?_) ?_
  · have := Finset.le_sup (f := fun cell => f cell - cell.2) hcell
    omega
  · rcases Finset.exists_mem_eq_sup s ⟨c₀, hc₀⟩ (fun cell => f cell - cell.2)
      with ⟨cm, hcm, heq⟩
    rw [heq]
    by_cases hz : f cm - cm.2 = 0
    · rw [hz]
      have := Finset.le_sup (f := fun cell => f cell + c - cell.2) hc₀
      omega
    · have := Finset.le_sup (f := fun cell => f cell + c - cell.2) hcm
      omega

/-- **Shift-equivariance of the hard drop on debt-1 boards.** For a
band placement (in-bounds, avoiding the well) the drop offset at band
base `c` is the base-0 offset plus `c`; the strictly covered hole is
invisible on both boards. -/
theorem dropOffset_debtBoard_bandLift {cfg : GameConfig} {w c : ℕ}
    {ρ : ℕ → ℕ} {ho : Option Coord} {pl : Placement}
    (hcols : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 < cfg.cols)
    (havoid : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 ≠ w)
    (hho : ∀ x, ho = some x → x.1 ≠ w ∧ x.2 + 1 < ρ x.1) :
    pl.dropOffset (debtBoard cfg (bandLift w c ρ) (holeLift c ho))
      = pl.dropOffset (debtBoard cfg ρ ho) + c := by
  have hcov : ∀ x, ho = some x → x.2 + 1 < ρ x.1 := fun x hx => (hho x hx).2
  have hcovL : ∀ x, holeLift c ho = some x → x.2 + 1 < bandLift w c ρ x.1 := by
    rintro x hx
    cases ho with
    | none => simp at hx
    | some x₀ =>
        rw [holeLift_some, Option.some.injEq] at hx
        subst hx
        rw [bandLift_ne w c ρ (hho x₀ rfl).1]
        have := (hho x₀ rfl).2
        omega
  rw [Placement.dropOffset_eq_sup, Placement.dropOffset_eq_sup]
  have hL : ∀ cell ∈ pl.shapeUp,
      (debtBoard cfg (bandLift w c ρ) (holeLift c ho)).colHeight (pl.col + cell.1)
        = ρ (pl.col + cell.1) + c := by
    intro cell hcell
    rw [colHeight_debtBoard hcovL, if_pos (hcols cell hcell),
      bandLift_ne w c ρ (havoid cell hcell)]
  have hR : ∀ cell ∈ pl.shapeUp,
      (debtBoard cfg ρ ho).colHeight (pl.col + cell.1) = ρ (pl.col + cell.1) := by
    intro cell hcell
    rw [colHeight_debtBoard hcov, if_pos (hcols cell hcell)]
  rw [Finset.sup_congr rfl fun cell hcell => by rw [hL cell hcell],
    Finset.sup_congr rfl fun cell hcell => by rw [hR cell hcell]]
  exact sup_sub_add_shift pl.shapeUp (fun cell => ρ (pl.col + cell.1)) c
    (Piece.shapeUp_exists_bottom pl.piece pl.rot)
```

Notes for the implementer:
- `pl.shapeUp` unfolds to `pl.piece.shapeUp pl.rot` (`Model/Placement.lean:31`), so `Piece.shapeUp_exists_bottom pl.piece pl.rot` has the right type; if elaboration balks, insert `show ∃ cell ∈ pl.piece.shapeUp pl.rot, cell.2 = 0` or unfold `Placement.shapeUp`.
- If `Finset.exists_mem_eq_sup` is not the available name, alternatives in mathlib: `Finset.exists_mem_eq_sup` for `ℕ` (a `SemilatticeSup` with `OrderBot`) exists as `Finset.exists_mem_eq_sup s hs f`; otherwise use `Finset.sup_mem_of_nonempty` or induct with `Finset.sup'` and `Finset.sup'_eq_sup`.
- The two `Finset.sup_congr … by rw` rewrites can also be done with `refine Finset.sup_congr rfl fun cell hcell => ?_` blocks if the inline form misparses.

- [ ] **Step 2: Build**

Run: `lake build`
Expected: `Build completed successfully`.

- [ ] **Step 3: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Invariants/BandShift.lean
git commit -m "proofs(band-shift): shift-equivariance of the hard-drop offset on debt-1 boards"
```

---

### Task 3: T1 — clear-free transition transport

**Files:**
- Modify: `proofs/Proofs/Invariants/BandShift.lean` (append inside `namespace Board`)

**Interfaces:**
- Consumes: Tasks 1–2 (`mem_debtBoard`, `dropOffset_debtBoard_bandLift`); `Placement.place_eq_union_dropped`, `Placement.dropped_eq_image`.
- Produces (used by Tasks 5–6): `Board.place_debtBoard_bandLift` with the exact signature below.

- [ ] **Step 1: Write T1**

```lean
/-- **T1 — clear-free transport.** A witnessed placement transition between
debt-1 boards at the base-0 representative transports verbatim to every
band base `c`: the landing cells shift up by `c`, the band bottom slab
(`rows < c`) is filled on both sides, and the hole rides at `+c`. -/
theorem place_debtBoard_bandLift {cfg : GameConfig} {w c : ℕ}
    {ρ ρ' : ℕ → ℕ} {ho ho' : Option Coord} {pl : Placement}
    (hcols : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 < cfg.cols)
    (havoid : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 ≠ w)
    (hho : ∀ x, ho = some x → x.1 ≠ w ∧ x.2 + 1 < ρ x.1)
    (hho' : ∀ x, ho' = some x → x.1 ≠ w)
    (hrep : pl.place (debtBoard cfg ρ ho) = debtBoard cfg ρ' ho') :
    pl.place (debtBoard cfg (bandLift w c ρ) (holeLift c ho))
      = debtBoard cfg (bandLift w c ρ') (holeLift c ho') := by
  have hD := dropOffset_debtBoard_bandLift (cfg := cfg) (w := w) (c := c)
    (ρ := ρ) (ho := ho) (pl := pl) hcols havoid hho
  have hpt := fun p => Finset.ext_iff.mp hrep p
  ext ⟨a, r⟩
  rw [Placement.place_eq_union_dropped, Finset.mem_union, mem_debtBoard,
    Placement.dropped_eq_image, Finset.mem_image, hD]
  -- Pointwise rep instance at the un-shifted row.
  have hrep_ar := hpt (a, r - c)
  rw [Placement.place_eq_union_dropped, Finset.mem_union, mem_debtBoard,
    Placement.dropped_eq_image, Finset.mem_image, mem_debtBoard] at hrep_ar
  rw [mem_debtBoard]
  by_cases haw : a = w
  · -- Column w: empty on both lifted sides (bandLift pins it at 0, the
    -- placement avoids it, and the hole is in the band).
    subst haw
    constructor
    · rintro (⟨-, -, hlt⟩ | ⟨cell, hcell, hc⟩)
      · rw [bandLift_well] at hlt; omega
      · exact absurd (congrArg Prod.fst hc) (by simpa using havoid cell hcell)
    · rintro ⟨-, -, hlt⟩
      rw [bandLift_well] at hlt; omega
  · by_cases hrc : r < c
    · -- Bottom slab: filled on both sides for in-range band columns; the
      -- dropped cells and both lifted holes live at rows ≥ c.
      constructor
      · rintro (⟨-, hac, -⟩ | ⟨cell, hcell, hc⟩)
        · refine ⟨?_, hac, ?_⟩
          · rintro x hx
            cases ho' with
            | none => simp at hx
            | some x₀ =>
                rw [holeLift_some, Option.some.injEq] at hx
                subst hx
                intro hcontra
                have : r = x₀.2 + c := congrArg Prod.snd hcontra
                omega
          · rw [bandLift_ne w c ρ' haw]; omega
        · have : r = pl.dropOffset (debtBoard cfg ρ ho) + c + cell.2 :=
            (congrArg Prod.snd hc).symm
          omega
      · rintro ⟨-, hac, -⟩
        refine Or.inl ⟨?_, hac, ?_⟩
        · rintro x hx
          cases ho with
          | none => simp at hx
          | some x₀ =>
              rw [holeLift_some, Option.some.injEq] at hx
              subst hx
              intro hcontra
              have : r = x₀.2 + c := congrArg Prod.snd hcontra
              omega
        · rw [bandLift_ne w c ρ haw]; omega
    · -- Band, above the slab: transfer through the representative at row r − c.
      push_neg at hrc
      constructor
      · rintro (⟨hhole, hac, hlt⟩ | ⟨cell, hcell, hc⟩)
        · rw [bandLift_ne w c ρ haw] at hlt
          have hL : ((a, r - c) : Coord) ∈ debtBoard cfg ρ ho ∨ _ :=
            Or.inl (by
              rw [mem_debtBoard]
              refine ⟨?_, hac, by omega⟩
              rintro x hx rfl
              exact hhole (x.1, x.2 + c) (by rw [hx, holeLift_some])
                (by simp; omega))
          sorry_replaced_by_transfer
        · sorry_replaced_by_transfer
      · sorry_replaced_by_transfer
```

**IMPORTANT — the block above is the shape, not the final script.** The three
`sorry_replaced_by_transfer` markers are where the mechanical transfer goes;
write them out as follows (this is the actual work of the task, expect
~60–120 further lines):

1. Rewrite `hrep_ar` so both of its sides are propositional (it now reads
   `(hole-cond ∧ a < cols ∧ r − c < ρ a) ∨ (∃ cell ∈ shapeUp, (col+cell.1, D+cell.2) = (a, r−c)) ↔ (hole'-cond ∧ a < cols ∧ r − c < ρ' a)`).
2. Forward direction, base case: from `⟨hhole, hac, hlt⟩` (lift-side base
   membership at `(a, r)`), build the rep-side base membership at
   `(a, r − c)` — hole condition transfers because `holeLift` shifts rows by
   exactly `c` and `r = c + (r − c)` (`omega`); height because
   `r < ρ a + c ↔ r − c < ρ a` given `c ≤ r` (`omega`). Feed it through
   `hrep_ar.mp`, get the rep-RHS triple, and rebuild the lift-RHS triple the
   same way in reverse (using `bandLift_ne w c ρ' haw` and, for the hole,
   `hho'` is NOT needed — only row arithmetic is).
3. Forward direction, dropped case: from `⟨cell, hcell, hc⟩` with
   `hc : (pl.col + cell.1, pl.dropOffset (debtBoard cfg ρ ho) + c + cell.2) = (a, r)`,
   produce the rep dropped cell
   `(pl.col + cell.1, pl.dropOffset (debtBoard cfg ρ ho) + cell.2) = (a, r − c)`
   (`Prod.ext` + `omega`), feed through `hrep_ar.mp` (as `Or.inr ⟨cell, hcell, _⟩`),
   rebuild lift-RHS as in 2.
4. Backward direction: from lift-RHS at `(a, r)` build rep-RHS at
   `(a, r − c)` (same two omega translations), apply `hrep_ar.mpr`, then case
   on the resulting disjunction: base membership lifts by adding `c` back;
   a dropped cell `(pl.col + cell.1, D + cell.2) = (a, r − c)` lifts to
   `(pl.col + cell.1, D + c + cell.2) = (a, r)` using `c ≤ r` (`omega`).

General tactics: keep every branch's arithmetic in `omega`; destructure
`Prod` equalities with `Prod.mk.injEq`/`Prod.ext_iff` and `simp only`; when a
hole condition `∀ x, holeLift c ho = some x → (a, r) ≠ x` needs to become
`∀ x, ho = some x → (a, r − c) ≠ x`, do `cases ho` first (`holeLift_none` /
`holeLift_some` normalize).

Fallback if the single proof balloons past ~250 lines: split into
`place_skyline_bandLift` (the `ho = none, ho' = none` case) and the general
lemma, sharing the slab/dropped sub-lemmas
(`mem_cellsAt_shift : (a, r) ∈ pl.cellsAt (d + c) ↔ c ≤ r ∧ (a, r - c) ∈ pl.cellsAt d`
is a good standalone helper — prove it first if you take this route).

- [ ] **Step 2: Build**

Run: `lake build`
Expected: `Build completed successfully`, zero `sorry` (grep the file: `grep -n sorry proofs/Proofs/Invariants/BandShift.lean` → no hits).

- [ ] **Step 3: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Invariants/BandShift.lean
git commit -m "proofs(band-shift): T1 clear-free transport — witnessed debt-1 place transitions lift to every band base"
```

---

### Task 4: T2 — the generic drain

**Files:**
- Modify: `proofs/Proofs/Invariants/BandShift.lean` (append inside `namespace Board`)

**Interfaces:**
- Consumes: Tasks 1–2; `shapeUp_vertI'`, `clearLines_skyline`, `clearLines_holedSkyline_of_le`, `Placement.applyStep_eq_clearLines_place`.
- Produces (used by Tasks 5–6): `Board.bandDrain (w : ℕ) : Placement`, `Board.bandDrain_shapeUp`, `Board.drain_debtBoard_bandLift` with the signature below. (`bandDrain w` is definitionally `Safety/SeamBridge.drainPl w`; the bridge is Task 6's `bandDrain_eq_drainPl`.)

- [ ] **Step 1: Write the drain placement and its landing computation**

```lean
/-- The drain placement: vertical I in column `w`. Mirrors
`Safety/SeamBridge.drainPl`, which lives above this layer; the definitional
bridge is stated beside `ShiftCertificate`. -/
def bandDrain (w : ℕ) : Placement := ⟨Piece.I, 1, w⟩

theorem bandDrain_shapeUp (w : ℕ) :
    (bandDrain w).shapeUp = {((0 : ℕ), (0 : ℕ)), (0, 1), (0, 2), (0, 3)} :=
  shapeUp_vertI' w 1 (by decide)

/-- The drain drops to the floor of the (empty) well and fills rows 0–3. -/
theorem bandDrain_place {cfg : GameConfig} {w c : ℕ} {ρ : ℕ → ℕ}
    {ho : Option Coord}
    (hho : ∀ x, ho = some x → x.1 ≠ w ∧ x.2 + 1 < ρ x.1) :
    (bandDrain w).place (debtBoard cfg (bandLift w c ρ) (holeLift c ho))
      = debtBoard cfg (fun j => if j = w then 4 else ρ j + c) (holeLift c ho) := by
  -- (i) dropOffset = 0: every sup term is colHeight at column w = 0.
  -- (ii) dropped = {(w,0),(w,1),(w,2),(w,3)}.
  -- (iii) ext with mem_debtBoard; column w gains rows 0–3, everything else
  --       (band, slab, hole) is untouched; omega closes each branch.
  sorry_replace_with_proof
```

Proof recipe for `bandDrain_place`:
1. `hcovL` as in Task 2 (lifted hole is covered) — factor it or inline it.
2. `hD : (bandDrain w).dropOffset (debtBoard cfg (bandLift w c ρ) (holeLift c ho)) = 0`:
   `rw [Placement.dropOffset_eq_sup, bandDrain_shapeUp]`, then
   `simp only [Finset.sup_insert, Finset.sup_singleton]`; each term is
   `colHeight … (w + 0) - k`; rewrite `colHeight_debtBoard hcovL`; the
   `if` reduces via `bandLift_well` when `w < cfg.cols`, and to `0`
   otherwise — either way `omega` finishes. NOTE: `w < cfg.cols` is NOT a
   hypothesis here and is not needed: if `w ≥ cols` the colHeight is `0`
   anyway. Handle with `by_cases hw : w < cfg.cols` inside the `hD` proof
   if the `if` does not discharge uniformly.
3. `hdr` : `dropped = {(w,0),(w,1),(w,2),(w,3)}` via `Placement.dropped_eq_image,
   bandDrain_shapeUp, hD`, `simp only [Finset.image_insert, Finset.image_singleton]`,
   `norm_num` — exactly the `place_vertI_flat` pattern (`HoledSkyline.lean:798`).
4. `rw [Placement.place_eq_union_dropped, hdr]`; `ext ⟨a, r⟩`;
   `simp only [Finset.mem_union, mem_debtBoard, Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]`;
   `by_cases haw : a = w` (`bandLift_well`/`bandLift_ne` + hole-row `omega`
   as in T1's column-w branch) — each branch closes with `omega` after
   normalizing the hole condition by `cases ho`.

- [ ] **Step 2: Write T2**

```lean
/-- **T2 — the generic drain.** At any band base `c ≥ 4`, dropping the
vertical I into the well clears exactly rows 0–3: the pattern and the
rep-hole are unchanged and the base drops by 4. -/
theorem drain_debtBoard_bandLift {cfg : GameConfig} {w c : ℕ} {ρ : ℕ → ℕ}
    {ho : Option Coord} (hw : w < cfg.cols) (hc : 4 ≤ c)
    (hho : ∀ x, ho = some x → x.1 ≠ w ∧ x.2 + 1 < ρ x.1) :
    (bandDrain w).applyStep cfg (debtBoard cfg (bandLift w c ρ) (holeLift c ho))
      = debtBoard cfg (bandLift w (c - 4) ρ) (holeLift (c - 4) ho) := by
  rw [Placement.applyStep_eq_clearLines_place, bandDrain_place hho]
  have hm : ∀ j < cfg.cols, 4 ≤ (fun j => if j = w then 4 else ρ j + c) j := by
    intro j hj
    by_cases hjw : j = w <;> simp [hjw] <;> omega
  have hm0 : ∃ j < cfg.cols, (fun j => if j = w then 4 else ρ j + c) j = 4 :=
    ⟨w, hw, by simp⟩
  cases ho with
  | none =>
      rw [holeLift_none, debtBoard_none, debtBoard_none,
        clearLines_skyline hm hm0]
      congr 1
      funext j
      by_cases hjw : j = w <;> simp [bandLift, hjw] <;> omega
  | some x =>
      rw [holeLift_some, debtBoard_some, debtBoard_some,
        clearLines_holedSkyline_of_le
          (by simpa using (hho x rfl).1.lt_or_lt.elim (fun _ => ?) (fun _ => ?))
          hm hm0 (by simp; omega)]
      sorry_replace_with_congr
```

The `some` branch above sketches the call; write it cleanly as:
1. `hxcols : x.1 < cfg.cols` — NOT derivable from `hho` (which only gives
   `x.1 ≠ w`); **add it to `hho`**: change the hypothesis (in BOTH
   `bandDrain_place` and here, and back-propagate to Task 2/3 if you kept a
   shared shape) to
   `hho : ∀ x, ho = some x → x.1 ≠ w ∧ x.1 < cfg.cols ∧ x.2 + 1 < ρ x.1`.
   (The certificate's `cover` field supplies all three conjuncts — see
   Task 6.)
2. Apply `clearLines_holedSkyline_of_le (hxcols := hx1) hm hm0 (hxm := by omega)`
   — `hxm : 4 ≤ x.2 + c` from `hc : 4 ≤ c`.
3. Close with `congr 1`: profile `funext j` + `by_cases hjw : j = w` +
   `simp [bandLift, hjw]` + `omega` (uses `4 ≤ c` for
   `ρ j + c − 4 = ρ j + (c − 4)`), and hole
   `(x.1, x.2 + c - 4) = (x.1, x.2 + (c - 4))` by `Prod.ext rfl (by omega)`.

**Decision locked by this task:** the `hho` hypothesis shape for Tasks 2–4 is
the 3-conjunct form `x.1 ≠ w ∧ x.1 < cfg.cols ∧ x.2 + 1 < ρ x.1`. Use it
uniformly from Task 2 onward (Task 2/3 only use conjuncts 1 and 3; carrying
the second costs nothing and keeps one shape).

- [ ] **Step 3: Build**

Run: `lake build`
Expected: `Build completed successfully`, no `sorry` hits in the file.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Invariants/BandShift.lean
git commit -m "proofs(band-shift): T2 generic drain — vertical I into the well at base ≥ 4 shifts the band down 4, pattern and hole unchanged"
```

---

### Task 5: lifted flat witnesses (acceptance test)

**Files:**
- Modify: `proofs/Proofs/Invariants/BandShift.lean` (final section)

**Interfaces:**
- Consumes: T1/T2 (Tasks 3–4); the seven flat place-equalities
  (`HoledSkyline.lean:604–1063`, all `base`-parameterized — instantiate at `base := 0`); `shapeUp_horizS`, `shapeUp_horizZ`, `shapeUp_vertI'` and the analogous `shapeUp_*` facts used inside `place_O_flat`/`place_flatT`/`place_flatL`/`place_flatJ` (read each theorem's proof header for the exact per-piece `shapeUp` lemma name and cell set — they follow one pattern).
- Produces: `Board.place_bandFlat_{O,I,T,L,J,S,Z}_lift`, `Board.drain_bandFlat_lift` — directly reusable by any future certificate inhabitant.

- [ ] **Step 1: Write the S witness first (hole-creating — the hardest case)**

```lean
/-! ## The seven flat responses, at every band base

Validation that the transport layer bites on real content: the base-0 flat
witnesses of `Proofs/Invariants/HoledSkyline.lean` lift to the well-anchored
flat at arbitrary band base `c`. `S`/`Z` exercise the hole-creating case. -/

theorem place_bandFlat_S_lift (cfg : GameConfig) (w c col : ℕ)
    (hcol : col + 2 < cfg.cols)
    (h0 : col ≠ w) (h1 : col + 1 ≠ w) (h2 : col + 2 ≠ w) :
    ({ piece := Piece.S, rot := 0, col := col } : Placement).place
        (debtBoard cfg (bandLift w c (fun _ => 0)) none)
      = debtBoard cfg
          (bandLift w c (fun j => if j = col then 0 + 1
            else if j = col + 1 then 0 + 2
            else if j = col + 2 then 0 + 2 else 0))
          (holeLift c (some (col + 2, 0))) := by
  refine place_debtBoard_bandLift ?hcols ?havoid ?hho ?hho' ?hrep
  case hcols =>
    intro cell hcell
    rw [show ({ piece := Piece.S, rot := 0, col := col } : Placement).shapeUp
        = {((0 : ℕ), (0 : ℕ)), (1, 0), (1, 1), (2, 1)}
      from shapeUp_horizS col 0 (by decide)] at hcell
    fin_cases hcell <;> omega
  case havoid =>
    intro cell hcell
    rw [show ({ piece := Piece.S, rot := 0, col := col } : Placement).shapeUp
        = {((0 : ℕ), (0 : ℕ)), (1, 0), (1, 1), (2, 1)}
      from shapeUp_horizS col 0 (by decide)] at hcell
    fin_cases hcell <;> simpa using by omega
  case hho => intro x hx; exact absurd hx (by simp)
  case hho' =>
    intro x hx
    rw [Option.some.injEq] at hx
    subst hx
    refine ⟨h2, hcol, ?_⟩
    simp
  case hrep =>
    rw [debtBoard_none, debtBoard_some]
    exact place_horizS_flat_eq_holedSkyline cfg 0 col hcol
```

Adjustments the implementer should expect:
- T1's `hho'` (as finalized in Task 4) may have 2 or 3 conjuncts — match it.
- The `fin_cases hcell <;> …` closers depend on how the shapeUp set
  elaborates; `simp only [Finset.mem_insert, Finset.mem_singleton] at hcell`
  then `rcases hcell` + per-case `omega`/`exact hi` is the robust fallback.
- The hole-cover conjunct is `0 + 1 < (if col + 2 = col then … else if … then 2 else …)`:
  `simp [h2, show col + 2 ≠ col by omega, show col + 2 ≠ col + 1 by omega]` then `omega`.
- If T1's final hypothesis list dropped `hho'` entirely (Task 3 note says only
  row arithmetic is needed on the primed side), delete that case here.

- [ ] **Step 2: Write Z (mirror of S), then O, I, T, L, J (flush, `ho' = none`), then the drain instance**

Each flush witness has the same skeleton with: the piece's `Placement` literal
and rep-equality from `HoledSkyline.lean` at `base := 0`
(`place_O_flat cfg 0 col hcol`, `place_vertI_flat cfg 0 col hcol`,
`place_flatT cfg 0 col hcol`, `place_flatL cfg 0 col hcol`,
`place_flatJ cfg 0 col hcol` — rewrite both sides with `debtBoard_none`),
the piece's column-avoidance hypotheses (`h0 : col ≠ w` … up to the piece's
width: O needs `h0,h1`; I(vertical) needs `h0` only; T/L/J need `h0,h1,h2`;
S/Z need `h0,h1,h2`), `hho := fun x hx => absurd hx (by simp)`, and
`hho' := fun x hx => absurd hx (by simp)` (flush results carry `none`).
The lifted profile is `bandLift w c (<the base-0 RHS profile copied verbatim
from the source theorem>)` — read each source statement and copy its profile
with `base := 0` substituted.

The drain instance:

```lean
theorem drain_bandFlat_lift (cfg : GameConfig) (w c : ℕ)
    (hw : w < cfg.cols) (hc : 4 ≤ c) :
    (bandDrain w).applyStep cfg
        (debtBoard cfg (bandLift w c (fun _ => 0)) none)
      = debtBoard cfg (bandLift w (c - 4) (fun _ => 0)) none :=
  drain_debtBoard_bandLift hw hc (fun x hx => absurd hx (by simp))
```

- [ ] **Step 3: Build**

Run: `lake build`
Expected: `Build completed successfully`.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Invariants/BandShift.lean
git commit -m "proofs(band-shift): the seven flat responses lift to every band base — transport layer validated on real content"
```

---

### Task 6: ShiftCertificate + the reduction

**Files:**
- Create: `proofs/Proofs/Safety/ShiftCertificate.lean`
- Modify: `proofs/Proofs.lean` (add import after line 49, `import Proofs.Safety.SkylineInvariant`)

**Interfaces:**
- Consumes: `DebtCertificate`, `tetrisSolvableValid_of_debtCertificate`
  (`Safety/SkylineInvariant.lean`); `AvoidsWell`, `drainPl`, `drainPl_valid`
  (`Safety/SeamBridge.lean`); T1/T2/`bandDrain`/`bandLift_zero`/`mem_debtBoard`
  (BandShift); `Placement.Valid` (`Model/Game.lean:22` — it is exactly
  `∀ cell ∈ pl.shapeUp, pl.col + cell.1 < cfg.cols`; verify before writing,
  and if it differs, extract `hcols` from it accordingly).
- Produces: `Tetris.ShiftCertificate`, `ShiftCertificate.toDebtCertificate`,
  `Tetris.tetrisSolvableValid_of_shiftCertificate`, `Board.bandDrain_eq_drainPl`.

- [ ] **Step 1: Write the module**

```lean
import Proofs.Safety.SkylineInvariant
import Proofs.Safety.SeamBridge
import Proofs.Invariants.BandShift

/-!
# The translation-quotient certificate

`ShiftCertificate` is `DebtCertificate` quotiented by the well-anchored band
translation: states are base-0 representatives `(bag, pattern, hole)` plus a
designer base predicate `okBase`; every board-level closure obligation is
stated at the representative (no base in the `place` equality) and
transported to all bases by `Proofs/Invariants/BandShift.lean`. The drain
case is fully generic — the inhabitant only chooses *when* to drain
(`4 ≤ c` under `okBase`), never *how*.
-/

namespace Tetris

open Board

/-- `bandDrain` is `SeamBridge`'s drain placement (they live in different
layers; the definition is duplicated, the equality is definitional). -/
theorem Board.bandDrain_eq_drainPl (w : ℕ) : bandDrain w = drainPl w := rfl

/-- A debt board over a well-anchored profile keeps its well column empty. -/
theorem Board.wellFree_debtBoard {cfg : GameConfig} {h : ℕ → ℕ}
    {ho : Option Coord} {w : ℕ} (hanch : h w = 0)
    (hho : ∀ x, ho = some x → x.1 ≠ w) :
    WellFree w (debtBoard cfg h ho) := by
  intro r hmem
  rw [mem_debtBoard] at hmem
  obtain ⟨-, -, hlt⟩ := hmem
  simp only at hlt
  omega

/-- **The translation-quotient certificate.** A bag-indexed family of base-0
band representatives (profiles anchored at an empty well, debt ≤ 1) with a
designer base predicate, closed under one response per pending piece: either
a well-avoiding placement proven at the representative, or the generic
well drain. Inhabiting this proves Tetris solvable
(`tetrisSolvableValid_of_shiftCertificate`); it collapses the
`DebtCertificate` witness space from absolute boards to relative patterns. -/
structure ShiftCertificate where
  /-- The well column. -/
  well : ℕ
  hwell : well < GameConfig.standard.cols
  /-- The relative family: pending bag, base-0 profile, optional hole. -/
  Q : Bag → (ℕ → ℕ) → Option Coord → Prop
  /-- The designer base predicate: which band bases each state admits. -/
  okBase : Bag → (ℕ → ℕ) → Option Coord → ℕ → Prop
  /-- The empty board at a fresh bag, at base 0. -/
  init : Q Bag.full (fun _ => 0) none
  initBase : okBase Bag.full (fun _ => 0) none 0
  /-- Representatives keep the well empty. -/
  anchored : ∀ T ρ ho, Q T ρ ho → ρ well = 0
  /-- Holes live in the band, in-field, strictly covered. -/
  cover : ∀ T ρ x, Q T ρ (some x) →
    x.1 ≠ well ∧ x.1 < GameConfig.standard.cols ∧ x.2 + 1 < ρ x.1
  /-- Admissible bases respect the ceiling. -/
  height : ∀ T ρ ho c, Q T ρ ho → okBase T ρ ho c →
    ∀ j < GameConfig.standard.cols,
      Board.bandLift well c ρ j ≤ GameConfig.standard.rows
  /-- Closure: every pending piece has a response — a well-avoiding
  placement proven at the representative, or the generic drain. -/
  step : ∀ T ρ ho c p, Q T ρ ho → okBase T ρ ho c → p ∈ T →
    (∃ pl ρ' ho', pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        AvoidsWell well pl ∧
        pl.place (Board.debtBoard GameConfig.standard ρ ho)
          = Board.debtBoard GameConfig.standard ρ' ho' ∧
        Q (T.draw p) ρ' ho' ∧ okBase (T.draw p) ρ' ho' c)
    ∨ (p = Piece.I ∧ 4 ≤ c ∧ Q (T.draw p) ρ ho ∧ okBase (T.draw p) ρ ho (c - 4))

namespace ShiftCertificate

/-- The absolute family realized by a shift certificate: every band lift of
every representative at an admissible base. -/
def toFamily (C : ShiftCertificate) :
    Bag → (ℕ → ℕ) → Option Coord → Prop := fun T h ho' =>
  ∃ ρ ho c, C.Q T ρ ho ∧ C.okBase T ρ ho c ∧
    h = Board.bandLift C.well c ρ ∧ ho' = holeLift c ho

/-- **The quotient reduction**: a shift certificate yields a debt certificate. -/
def toDebtCertificate (C : ShiftCertificate) : DebtCertificate where
  P := C.toFamily
  init := ⟨fun _ => 0, none, 0, C.init, C.initBase, by
    funext j
    by_cases hj : j = C.well <;> simp [Board.bandLift, hj], rfl⟩
  cover := by
    rintro T h x ⟨ρ, ho, c, hQ, hok, rfl, hx⟩
    cases ho with
    | none => simp at hx
    | some x₀ =>
        rw [holeLift_some, Option.some.injEq] at hx
        subst hx
        obtain ⟨hxw, hxc, hxcov⟩ := C.cover T ρ x₀ hQ
        refine ⟨hxc, ?_⟩
        rw [Board.bandLift_ne C.well c ρ hxw]
        omega
  height := by
    rintro T h ho' ⟨ρ, ho, c, hQ, hok, rfl, rfl⟩ j hj
    exact C.height T ρ ho c hQ hok j hj
  step := by
    rintro T h ho' p ⟨ρ, ho, c, hQ, hok, rfl, rfl⟩ hp
    have hhoW : ∀ x, ho = some x → x.1 ≠ C.well ∧ x.1 < GameConfig.standard.cols
        ∧ x.2 + 1 < ρ x.1 :=
      fun x hx => C.cover T ρ x (hx ▸ hQ)
    rcases C.step T ρ ho c p hQ hok hp with
      ⟨pl, ρ', ho'', hpiece, hvalid, havoidW, hrepEq, hQ', hok'⟩ | ⟨hpI, hc4, hQ', hok'⟩
    · -- Placement case: transport by T1, then applyStep = place (well open).
      refine ⟨pl, Board.bandLift C.well c ρ', holeLift c ho'', hpiece, hvalid, ?_, ?_⟩
      · have hlift := Board.place_debtBoard_bandLift
          (cfg := GameConfig.standard) (w := C.well) (c := c)
          (fun cell hcell => hvalid cell hcell) havoidW hhoW
          (fun x hx => (C.cover (T.draw p) ρ' x (hx ▸ hQ')).1) hrepEq
        rw [Placement.applyStep_eq_clearLines_place, hlift,
          Board.clearLines_eq_self_of_no_fullRows GameConfig.standard
            (fullRows_eq_empty_of_wellFree C.hwell
              (Board.wellFree_debtBoard
                (by simp [Board.bandLift_well])
                (fun x hx => ?_)))]
        -- lifted result hole is off the well: cases ho''; C.cover on hQ'.
        sorry_replace_with_hole_col_arg
      · exact ⟨ρ', ho'', c, hQ', hok', rfl, rfl⟩
    · -- Drain case: T2 verbatim.
      subst hpI
      refine ⟨bandDrain C.well, Board.bandLift C.well (c - 4) ρ,
        holeLift (c - 4) ho, rfl, ?_, ?_, ?_⟩
      · rw [Board.bandDrain_eq_drainPl]
        exact drainPl_valid C.hwell
      · exact Board.drain_debtBoard_bandLift C.hwell hc4 hhoW
      · exact ⟨ρ, ho, c - 4, hQ', hok', rfl, rfl⟩

end ShiftCertificate

/-- Inhabiting the quotient certificate proves Tetris solvable. -/
theorem tetrisSolvableValid_of_shiftCertificate (C : ShiftCertificate) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_debtCertificate C.toDebtCertificate

end Tetris
```

Implementation notes:
- The `sorry_replace_with_hole_col_arg` hole: prove
  `∀ x, holeLift c ho'' = some x → x.1 ≠ C.well` by `cases ho''` +
  `holeLift_some` + `(C.cover (T.draw p) ρ' x₀ hQ').1` — 4 lines. Restructure
  the `refine` if inline placement fights the elaborator: bind
  `hWF : WellFree C.well (debtBoard … (bandLift …) (holeLift c ho''))` as a
  standalone `have` before the `rw`.
- `Board.clearLines_eq_self_of_no_fullRows` — verify exact name/signature
  (used at `SeamBridge.lean:81`); it takes the `fullRows … = ∅` proof.
- `hvalid cell hcell` as `hcols`: this assumes `Placement.Valid` is literally
  the bounded-∀ (`Model/Game.lean:22`). Check first; adapt if it is a
  structure or has extra conjuncts.
- `DebtCertificate.step`'s existential order is `(pl, h', ho')` with fields
  `pl.piece = p`, `pl.Valid`, the `applyStep` equality, then membership —
  match `SkylineInvariant.lean:138–144` exactly when assembling the anonymous
  constructor; adjust the `refine` tuple shape to the real field order.
- `toDebtCertificate.init`'s profile equality: goal is
  `(fun _ => 0) = Board.bandLift C.well 0 (fun _ => 0)` possibly reversed —
  orient to match the field (`h = bandLift …` per `toFamily`); `bandLift_zero`
  (Task 1) with `hw : (fun _ => 0) C.well = 0 := rfl` is the alternative
  one-liner.

- [ ] **Step 2: Wire the import**

In `proofs/Proofs.lean`, after `import Proofs.Safety.SkylineInvariant`
(line 49), add:

```lean
import Proofs.Safety.ShiftCertificate
```

- [ ] **Step 3: Build**

Run: `lake build`
Expected: `Build completed successfully`, no `sorry` in the new file.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Safety/ShiftCertificate.lean proofs/Proofs.lean
git commit -m "proofs(shift-certificate): the translation quotient — DebtCertificate closure at base-0 representatives, base handled by a designer predicate"
```

---

### Task 7: gates, docs, final verification

**Files:**
- Modify: `proofs/PROGRESS.md` (insert an entry under `## Last tick`, above the current one)
- Modify: `proofs/LIBRARY.md` (two one-line additions, see below)

**Interfaces:**
- Consumes: everything above.
- Produces: verified hygiene + updated docs.

- [ ] **Step 1: Green hygiene gate**

Run (from `proofs/`): `scripts/check-green-clean.sh`
Expected: exit 0 (no `sorry`/`native_decide` in the green tree).

- [ ] **Step 2: Axiom gate**

Write to the session scratchpad (NOT the repo):

```lean
import Proofs
open Tetris
#print axioms tetrisSolvableValid_of_shiftCertificate
#print axioms Board.place_debtBoard_bandLift
#print axioms Board.drain_debtBoard_bandLift
```

Run: `lake env lean <scratchpad>/axiom_gate_shift.lean`
Expected: every line prints exactly
`depends on axioms: [propext, Classical.choice, Quot.sound]` (a subset is
also acceptable).

- [ ] **Step 3: Experiments target still builds**

Run: `lake build ProofsExperiments` (foreground)
Expected: `Build completed successfully`.

- [ ] **Step 4: Update PROGRESS.md and LIBRARY.md**

`PROGRESS.md` — insert directly under the `## Last tick` heading (keep the
old entry below it):

```markdown
Tick (manual, 2026-07-09) — **the translation quotient**: `Invariants/BandShift.lean`
(well-anchored band lift; T1 `place_debtBoard_bandLift` transports witnessed debt-1
place transitions to every band base; T2 `drain_debtBoard_bandLift` — the generic
well drain at base ≥ 4, pattern and hole unchanged; the seven flat witnesses lifted
as validation) + `Safety/ShiftCertificate.lean` (`ShiftCertificate` = `DebtCertificate`
quotiented by band translation, designer `okBase` predicate;
`tetrisSolvableValid_of_shiftCertificate` via `toDebtCertificate`). Board-level closure
obligations are now stated once at base-0 representatives; base handling is scalar
arithmetic. Build green, axiom gate clean.
```

`LIBRARY.md` — in the §2 Layer-3 table, add a row after the `DebtCertificate`
mention (or after the `safe_extract` row if none):

```markdown
| `ShiftCertificate` + `tetrisSolvableValid_of_shiftCertificate`; BandShift T1/T2 | the translation quotient: DebtCertificate closure at base-0 band representatives; witnessed transitions and the generic drain transport to every base |
```

and in the §4 tree, extend the `Invariants/` line with `BandShift` and the
`Safety/` line with `ShiftCertificate`.

- [ ] **Step 5: Final full build**

Run: `lake build`
Expected: `Build completed successfully`.

- [ ] **Step 6: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/PROGRESS.md proofs/LIBRARY.md
git commit -m "proofs(shift-certificate): progress + library map entries for the translation quotient"
```

---

## Self-review notes (already applied)

- Spec coverage: bandLift/holeLift (T1), T2 drain, mem_debtBoard helper,
  ShiftCertificate + okBase + reduction via toDebtCertificate, seven lifted
  witnesses + drain instance, gates and doc updates — all spec sections have
  a task. The spec's `AvoidsCol` local predicate was simplified away: T1/T2
  take the avoidance hypothesis inline; the certificate uses SeamBridge's
  `AvoidsWell` (same layer) and the reduction unfolds it — no duplicate
  definition needed.
- Type consistency: `hho` is the 3-conjunct form everywhere from Task 2 on
  (Task 4 locks it); T1's primed-side hypothesis is only the hole-column
  conjunct (`x.1 ≠ w`) — Task 5 and Task 6 match that shape.
- Known uncertainty, flagged in-task: exact mathlib name
  `Finset.exists_mem_eq_sup` (Task 2 fallback given), `Placement.Valid`'s
  literal form (Task 6 note), `clearLines_eq_self_of_no_fullRows` signature
  (Task 6 note), and elaboration details of `fin_cases` on shapeUp sets
  (Task 5 fallback given).
