# Band Schedule (BandMechanisms + BandScheduleCert) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two new green Lean modules: `Proofs/Invariants/BandMechanisms.lean` (the debt-carry wrapper, the horizontal-I flat witness, the bag-1 S/Z mutual-enabling pack) and `Proofs/Safety/BandSchedule.lean` (`BandState`, `bump2`, `BandScheduleCert`, `toShiftCertificate`, `tetrisSolvableValid_of_bandSchedule`) — reducing the open 7-piece closure to a 4-piece (T/L/J/I) schedule plus bag-1 wiring.

**Architecture:** One master lemma (`place_debtBoard_of_flush`) upgrades every flush skyline transition to debt-1 boards. `BandScheduleCert` packages a designer family over `BandState` (profile + hole + `Option` window anchors); the anchored S/Z/O cases are discharged by us via the reproduction mechanisms + wrapper; T/L/J/I and un-anchored cases remain designer obligations. Reduction lands in `ShiftCertificate`. Spec: `docs/superpowers/specs/2026-07-10-band-schedule-design.md`.

**Tech Stack:** Lean 4 + mathlib (pinned via `proofs/lean-toolchain`), lake. `namespace Tetris` (board content in `Tetris.Board`).

## Global Constraints

- Green target: **no `sorry`, no `native_decide`, no new axioms** — every new theorem exactly `[propext, Classical.choice, Quot.sound]` or fewer.
- `lake` builds **foreground only**, from `/Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas/proofs`; never SIGTERM an in-flight build; rebuild after every edit before proceeding.
- Commits: one per task, staging **only `proofs/`**. Messages: `proofs(band-mechanisms): …` / `proofs(band-schedule): …`.
- Lean docstrings close at the first `-/`; never `word-/word` inside them. Line width ≈ 100.
- Key existing identifiers (all verified green):
  - `Board.debtBoard`, `debtBoard_none`/`debtBoard_some` (simp), `Board.holedSkyline = (skyline …).erase x` (definitional), `place_holedSkyline (pl) (hxcols : x.1 < cfg.cols) (hcov : x.2 + 1 < h x.1) : pl.place (holedSkyline cfg h x) = (pl.place (skyline cfg h)).erase x` (`HoledSkyline.lean:132`)
  - `place_vertS_skyline (hc : c < cfg.cols) (hc1 : c + 1 < cfg.cols) (hstep : h c = h (c+1) + 1) : Placement.place (skyline cfg h) ⟨.S, 1, c⟩ = skyline cfg (Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2))` (`Skyline.lean:439`); `place_vertZ_skyline` mirror with `hstep : h (c+1) = h c + 1` and the SAME output profile (`Skyline.lean:474`); `place_O_pair (hc) (hc1) (hpair : h c = h (c+1)) : … = skyline cfg (Function.update (Function.update h c (h c + 2)) (c + 1) (h c + 2))` (`HoledSkyline.lean:1310`-ish)
  - `colHeight_le_place (b) (pl) (j) : b.colHeight j ≤ (pl.place b).colHeight j` (`GameplayExtra.lean:102`), `colHeight_skyline (hj : j < cfg.cols)`
  - `shapeUp_vertS (c)` (`Skyline.lean:423`), `shapeUp_vertZ (c)` (:430), `shapeUp_O (c) (r)` (:707), `shapeUp_horizS/Z (c) (r) (hr)` (:518/:607) — read each statement before citing; the vertS cells are `{(0,1),(0,2),(1,0),(1,1)}`-shaped, vertZ `{(0,0),(0,1),(1,1),(1,2)}`-shaped (verify from the file)
  - `place_horizS_flat_eq_holedSkyline (cfg) (base col) (hcol : col + 2 < cfg.cols)` (`HoledSkyline.lean:604`), `place_horizZ_flat_eq_holedSkyline` (:641)
  - `ShiftCertificate` fields `well hwell Q okBase init initBase anchored cover height step` (`Safety/ShiftCertificate.lean`) — `step`'s exact disjunction shape is quoted in Task 5
  - `AvoidsWell`, namespace `Tetris.Seam` (open it); `Placement.Valid cfg pl = ∀ cell ∈ pl.shapeUp, pl.col + cell.1 < cfg.cols`
- The Lean "test cycle" is the build; transient `sorry` never committed.

---

### Task 1: BandMechanisms — the master debt-carry wrapper

**Files:**
- Create: `proofs/Proofs/Invariants/BandMechanisms.lean`
- Modify: `proofs/Proofs.lean` (add `import Proofs.Invariants.BandMechanisms` after the `import Proofs.Invariants.BandShift` line)

**Interfaces:**
- Consumes: `place_holedSkyline`, `colHeight_le_place`, `colHeight_skyline`, `debtBoard_none/some`.
- Produces (Tasks 3, 5): `Board.place_debtBoard_of_flush` and `Board.le_of_flush` with the exact signatures below.

- [ ] **Step 1: Write the module**

```lean
import Proofs.Invariants.HoledSkyline

/-!
# Band mechanisms: the debt-carry wrapper and the bag-1 bootstrap pack

`place_debtBoard_of_flush` upgrades every flush skyline transition to debt-1
boards: the strictly covered hole is disjoint from the drop path, so the
placement commutes with the erasure. Combined with the proven reproduction
mechanisms (`place_vertS_skyline`, `place_vertZ_skyline`, `place_O_pair`)
this makes the permanent bootstrap hole ride through all flush play for
free. The rest of the file is the bag-1 pack: the horizontal-I flat
response, and the S/Z mutual-enabling edges (the flat-S residue contains a
Z-window and vice versa, so the second roughness piece of bag 1 seats
flush — exactly one hole is ever forced).
-/

namespace Tetris
namespace Board

/-- Heights never drop across a flush transition (read off the flush
equality column-wise). -/
theorem le_of_flush {cfg : GameConfig} {ρ ρ' : ℕ → ℕ} {pl : Placement}
    (hflush : pl.place (skyline cfg ρ) = skyline cfg ρ')
    {j : ℕ} (hj : j < cfg.cols) : ρ j ≤ ρ' j := by
  have h := colHeight_le_place (skyline cfg ρ) pl j
  rwa [hflush, colHeight_skyline hj, colHeight_skyline hj] at h

/-- **The debt-carry wrapper.** A flush skyline transition holds verbatim on
the debt-1 board: the strictly covered hole is untouched by the drop and
survives on both sides. Upgrades every flush mechanism to debt-1. -/
theorem place_debtBoard_of_flush {cfg : GameConfig} {ρ ρ' : ℕ → ℕ}
    {ho : Option Coord} {pl : Placement}
    (hflush : pl.place (skyline cfg ρ) = skyline cfg ρ')
    (hho : ∀ x, ho = some x → x.1 < cfg.cols ∧ x.2 + 1 < ρ x.1) :
    pl.place (debtBoard cfg ρ ho) = debtBoard cfg ρ' ho := by
  cases ho with
  | none => simpa using hflush
  | some x =>
      obtain ⟨hxc, hxcov⟩ := hho x rfl
      rw [debtBoard_some, debtBoard_some,
        place_holedSkyline pl hxc hxcov, hflush]
      rfl

end Board
end Tetris
```

Note: the final `rfl` closes `(skyline cfg ρ').erase x = holedSkyline cfg ρ' x`
(definitional — `holedSkyline` IS the erase). If `rfl` fails, use
`unfold holedSkyline` first.

- [ ] **Step 2: Wire the import** — in `proofs/Proofs.lean`, directly after
`import Proofs.Invariants.BandShift`, add:

```lean
import Proofs.Invariants.BandMechanisms
```

- [ ] **Step 3: Build**

Run: `lake build`
Expected: `Build completed successfully` (only pre-existing lint warnings).

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Invariants/BandMechanisms.lean proofs/Proofs.lean
git commit -m "proofs(band-mechanisms): the debt-carry wrapper — flush skyline transitions hold verbatim on debt-1 boards"
```

---

### Task 2: the horizontal-I flat witness

**Files:**
- Modify: `proofs/Proofs/Invariants/BandMechanisms.lean` (append inside `namespace Board`)

**Interfaces:**
- Consumes: `Placement.dropOffset_eq_sup`, `dropped_eq_image`, `place_eq_union_dropped`, `colHeight_skyline`, `mem_skyline'`.
- Produces: `Board.shapeUp_horizI`, `Board.place_horizI_flat` — the bag-1 I response at base 0.

- [ ] **Step 1: Write the shape lemma and the witness**

```lean
theorem shapeUp_horizI (c : ℕ) :
    ({ piece := Piece.I, rot := 0, col := c } : Placement).shapeUp
      = {((0 : ℕ), (0 : ℕ)), (1, 0), (2, 0), (3, 0)} := by
  show Piece.shapeUp Piece.I 0 = _
  decide

/-- **Flush placement of the horizontal I on a flat surface** — four columns,
+1 each. This is the bag-1 I response at base 0, where the well drain
(`4 ≤ c`) is unavailable. -/
theorem place_horizI_flat (cfg : GameConfig) (base col : ℕ)
    (hcol : col + 3 < cfg.cols) :
    ({ piece := Piece.I, rot := 0, col := col } : Placement).place
        (skyline cfg (fun _ => base))
      = skyline cfg (fun j =>
          if j = col ∨ j = col + 1 ∨ j = col + 2 ∨ j = col + 3
          then base + 1 else base) := by
  have hsh := shapeUp_horizI col
  have hc0 : col + 0 < cfg.cols := by omega
  have hc1 : col + 1 < cfg.cols := by omega
  have hc2 : col + 2 < cfg.cols := by omega
  have hd : ({ piece := Piece.I, rot := 0, col := col } : Placement).dropOffset
      (skyline cfg (fun _ => base)) = base := by
    rw [Placement.dropOffset_eq_sup, hsh]
    simp only [Finset.sup_insert, Finset.sup_singleton,
      colHeight_skyline hc0, colHeight_skyline hc1, colHeight_skyline hc2,
      colHeight_skyline hcol]
    omega
  have hdr : ({ piece := Piece.I, rot := 0, col := col } : Placement).dropped
      (skyline cfg (fun _ => base))
      = {(col, base), (col + 1, base), (col + 2, base), (col + 3, base)} := by
    rw [Placement.dropped_eq_image, hsh, hd]
    simp only [Finset.image_insert, Finset.image_singleton]
    norm_num
  rw [Placement.place_eq_union_dropped, hdr]
  ext ⟨a, b⟩
  simp only [Finset.mem_union, mem_skyline', Finset.mem_insert,
    Finset.mem_singleton, Prod.mk.injEq]
  split_ifs <;> omega
```

Fallback if the `decide` in `shapeUp_horizI` fails: the I rot-0 cell set in
`Proofs/Model/Piece.lean` differs from `{(0,0),(1,0),(2,0),(3,0)}` — read
`Piece.shapeUp` there, correct the literal (and the `hdr` set / profile
accordingly; the flat witnesses `place_O_flat`/`place_flatT` show the exact
per-piece pattern to mirror). The sup in `hd` may present `colHeight … (col + 0)`
etc.; if the `colHeight_skyline` rewrites miss, insert `Nat.add_zero` in the
simp set.

- [ ] **Step 2: Build** — `lake build`, expect success.

- [ ] **Step 3: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Invariants/BandMechanisms.lean
git commit -m "proofs(band-mechanisms): horizontal-I flat witness — the bag-1 I response at base 0"
```

---

### Task 3: the bag-1 bootstrap pack (S/Z mutual enabling)

**Files:**
- Modify: `proofs/Proofs/Invariants/BandMechanisms.lean` (append inside `namespace Board`)

**Interfaces:**
- Consumes: Task 1's `place_debtBoard_of_flush`; `place_horizS_flat_eq_holedSkyline`, `place_horizZ_flat_eq_holedSkyline`, `place_vertZ_skyline`, `place_vertS_skyline`.
- Produces: `Board.sBootProfile`, `Board.zBootProfile`, `Board.sBoot`, `Board.zBoot`, `Board.zWindow_of_sBoot`, `Board.sWindow_of_zBoot`, `Board.place_vertZ_afterS`, `Board.place_vertS_afterZ`.

- [ ] **Step 1: Write the residue profiles and the first-piece edges**

```lean
/-! ## The bag-1 bootstrap pack

From flat, S-or-Z-first forces the single permanent hole
(`no_holefree_closed_invariant`); the residue of each contains the OTHER
roughness piece's window, so the second roughness piece seats flush. These
are the forced edges of any bag-1 tree; the wiring into a steady family is
the `BandScheduleCert` inhabitant's obligation. -/

/-- The flat-S residue: `(1, 2, 2)` at `col..col+2`, hole at `(col+2, 0)`. -/
def sBootProfile (col : ℕ) : ℕ → ℕ := fun j =>
  if j = col then 1 else if j = col + 1 then 2 else if j = col + 2 then 2 else 0

/-- The flat-Z residue: `(2, 2, 1)` at `col..col+2`, hole at `(col, 0)`. -/
def zBootProfile (col : ℕ) : ℕ → ℕ := fun j =>
  if j = col then 2 else if j = col + 1 then 2 else if j = col + 2 then 1 else 0

/-- S-first from flat, in `debtBoard` form. -/
theorem sBoot (cfg : GameConfig) (col : ℕ) (hcol : col + 2 < cfg.cols) :
    ({ piece := Piece.S, rot := 0, col := col } : Placement).place
        (debtBoard cfg (fun _ => 0) none)
      = debtBoard cfg (sBootProfile col) (some (col + 2, 0)) := by
  simpa [sBootProfile] using place_horizS_flat_eq_holedSkyline cfg 0 col hcol

/-- Z-first from flat, in `debtBoard` form. -/
theorem zBoot (cfg : GameConfig) (col : ℕ) (hcol : col + 2 < cfg.cols) :
    ({ piece := Piece.Z, rot := 0, col := col } : Placement).place
        (debtBoard cfg (fun _ => 0) none)
      = debtBoard cfg (zBootProfile col) (some (col, 0)) := by
  simpa [zBootProfile] using place_horizZ_flat_eq_holedSkyline cfg 0 col hcol

/-- **Mutual enabling, S side**: the flat-S residue carries a Z-window at
`col`. -/
theorem zWindow_of_sBoot (col : ℕ) :
    sBootProfile col (col + 1) = sBootProfile col col + 1 := by
  simp [sBootProfile]

/-- **Mutual enabling, Z side**: the flat-Z residue carries an S-window at
`col + 1`. -/
theorem sWindow_of_zBoot (col : ℕ) :
    zBootProfile col (col + 1) = zBootProfile col (col + 2) + 1 := by
  simp [zBootProfile]
```

Note: the `simp [sBootProfile]` closers must reduce the if-chains at the
window columns; if `simp` leaves goals, use
`simp only [sBootProfile]; split_ifs <;> omega` (with
`show col + 1 ≠ col by omega`-style side facts as needed).

- [ ] **Step 2: Write the second-piece flush edges**

```lean
/-- **The forced second edge, S-then-Z**: vertical Z seats flush on the
flat-S residue's Z-window; the fresh hole rides. Exactly one hole is ever
forced by the roughness pair. -/
theorem place_vertZ_afterS (cfg : GameConfig) (col : ℕ)
    (hcol : col + 2 < cfg.cols) :
    ({ piece := Piece.Z, rot := 1, col := col } : Placement).place
        (debtBoard cfg (sBootProfile col) (some (col + 2, 0)))
      = debtBoard cfg
          (Function.update (Function.update (sBootProfile col) col
            (sBootProfile col col + 2)) (col + 1)
            (sBootProfile col (col + 1) + 2))
          (some (col + 2, 0)) :=
  place_debtBoard_of_flush
    (place_vertZ_skyline (by omega) (by omega) (zWindow_of_sBoot col))
    (fun x hx => by
      cases hx
      exact ⟨hcol, by simp [sBootProfile]⟩)

/-- **The forced second edge, Z-then-S** (mirror): vertical S seats flush on
the flat-Z residue's S-window at `col + 1`. -/
theorem place_vertS_afterZ (cfg : GameConfig) (col : ℕ)
    (hcol : col + 2 < cfg.cols) :
    ({ piece := Piece.S, rot := 1, col := col + 1 } : Placement).place
        (debtBoard cfg (zBootProfile col) (some (col, 0)))
      = debtBoard cfg
          (Function.update (Function.update (zBootProfile col) (col + 1)
            (zBootProfile col (col + 1) + 2)) (col + 1 + 1)
            (zBootProfile col (col + 1 + 1) + 2))
          (some (col, 0)) :=
  place_debtBoard_of_flush
    (place_vertS_skyline (by omega) (by omega) (sWindow_of_zBoot col))
    (fun x hx => by
      cases hx
      exact ⟨by omega, by simp [zBootProfile]⟩)
```

Notes: (i) the hole-cover side goals are `0 + 1 < sBootProfile col (col+2)`
(= 2) and `0 + 1 < zBootProfile col col` (= 2) — `simp [sBootProfile]` with
arithmetic side-conditions; use `split_ifs <;> omega` fallback. (ii) The
`place_vertS_skyline` instantiation is at anchor `c := col + 1`, so its
`hstep` argument is `sWindow_of_zBoot` verbatim and the output profile
mentions `col + 1 + 1` — keep that spelling (do NOT normalize to `col + 2`)
so the statement matches the lemma output definitionally. (iii)
`cases hx` on `hx : some (col+2, 0) = some x` substitutes `x`; if `cases`
balks use `rw [Option.some.injEq] at hx; subst hx`.

- [ ] **Step 3: Build** — `lake build`, expect success.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Invariants/BandMechanisms.lean
git commit -m "proofs(band-mechanisms): the bag-1 bootstrap pack — S/Z mutual enabling, one hole ever forced"
```

---

### Task 4: BandState + BandScheduleCert

**Files:**
- Create: `proofs/Proofs/Safety/BandSchedule.lean`
- Modify: `proofs/Proofs.lean` (add `import Proofs.Safety.BandSchedule` after `import Proofs.Safety.ShiftCertificate`)

**Interfaces:**
- Consumes: `ShiftCertificate` (for Task 5), `AvoidsWell` (`Tetris.Seam`), `Board.debtBoard`, `Board.bandLift`.
- Produces: `Tetris.BandState` (fields `ρ ho cS cZ cO`), `BandState.bump2 : BandState → ℕ → BandState`, `Tetris.BandScheduleCert` with the exact fields below.

- [ ] **Step 1: Write the module (structure only, no reduction yet)**

```lean
import Proofs.Safety.ShiftCertificate
import Proofs.Invariants.BandMechanisms

/-!
# The band-schedule certificate

`BandScheduleCert` is the isolated remainder of the Tetris-solvability
construction after the translation quotient: a designer family over
`BandState` (base-0 profile + debt hole + `Option` window anchors) whose
anchored S/Z/O closure is discharged HERE, unconditionally, from the
reproduction mechanisms and the debt-carry wrapper. The open content is
exactly: the T/L/J responses, the I disjunction (drain guard or band
response), the un-anchored S/Z/O cases (the bag-1 bootstrap wiring — see
the pack in `Proofs/Invariants/BandMechanisms.lean`), and the `okB` rate
bookkeeping. Inhabiting this structure proves Tetris solvable
(`tetrisSolvableValid_of_bandSchedule`).
-/

namespace Tetris

open Board Seam

/-- A steady-state band representative: base-0 profile, optional debt hole,
and optional window anchors (unset during the bag-1 bootstrap). -/
structure BandState where
  ρ  : ℕ → ℕ
  ho : Option Coord
  cS : Option ℕ
  cZ : Option ℕ
  cO : Option ℕ

namespace BandState

/-- The forced flush successor at a 2-column window: both columns rise 2.
Matches the output profiles of `place_vertS_skyline`, `place_vertZ_skyline`
and (after rewriting by the pair equality) `place_O_pair` verbatim. -/
def bump2 (σ : BandState) (c : ℕ) : BandState :=
  { σ with ρ := Function.update (Function.update σ.ρ c (σ.ρ c + 2))
      (c + 1) (σ.ρ (c + 1) + 2) }

@[simp] theorem bump2_ho (σ : BandState) (c : ℕ) : (σ.bump2 c).ho = σ.ho := rfl

end BandState

/-- The flat start state: empty board, no debt, no anchors. -/
def BandState.start : BandState := ⟨fun _ => 0, none, none, none, none⟩

/-- **The band-schedule certificate** — the isolated open remainder.
Anchored S/Z/O responses are forced (`bump2`) and their board content is
proven by the library; the inhabitant supplies the schedule: T/L/J, the I
case, the bootstrap (`none`-anchor) cases, and the rate bookkeeping. -/
structure BandScheduleCert where
  well : ℕ
  hwell : well < GameConfig.standard.cols
  /-- The designer family. -/
  Inv : Bag → BandState → Prop
  /-- The designer base predicate (admissible band bases per state). -/
  okB : Bag → BandState → ℕ → Prop
  init : Inv Bag.full BandState.start
  initBase : okB Bag.full BandState.start 0
  /-- Window shapes when anchored. -/
  winS : ∀ T σ c, Inv T σ → σ.cS = some c → σ.ρ c = σ.ρ (c + 1) + 1
  winZ : ∀ T σ c, Inv T σ → σ.cZ = some c → σ.ρ (c + 1) = σ.ρ c + 1
  winO : ∀ T σ c, Inv T σ → σ.cO = some c → σ.ρ c = σ.ρ (c + 1)
  /-- Anchored windows sit in the band, off the well. -/
  winColsS : ∀ T σ c, Inv T σ → σ.cS = some c →
    c + 1 < GameConfig.standard.cols ∧ c ≠ well ∧ c + 1 ≠ well
  winColsZ : ∀ T σ c, Inv T σ → σ.cZ = some c →
    c + 1 < GameConfig.standard.cols ∧ c ≠ well ∧ c + 1 ≠ well
  winColsO : ∀ T σ c, Inv T σ → σ.cO = some c →
    c + 1 < GameConfig.standard.cols ∧ c ≠ well ∧ c + 1 ≠ well
  /-- The well column is empty at every representative. -/
  anchored : ∀ T σ, Inv T σ → σ.ρ well = 0
  /-- The debt hole is in-band, in-field, strictly covered. -/
  cover : ∀ T σ x, Inv T σ → σ.ho = some x →
    x.1 ≠ well ∧ x.1 < GameConfig.standard.cols ∧ x.2 + 1 < σ.ρ x.1
  /-- Admissible bases respect the ceiling. -/
  height : ∀ T σ b, Inv T σ → okB T σ b →
    ∀ j < GameConfig.standard.cols,
      Board.bandLift well b σ.ρ j ≤ GameConfig.standard.rows
  /-- Bookkeeping: the forced successors stay in the family. -/
  invS : ∀ T σ c, Inv T σ → σ.cS = some c → Piece.S ∈ T →
    Inv (T.draw Piece.S) (σ.bump2 c) ∧
    ∀ b, okB T σ b → okB (T.draw Piece.S) (σ.bump2 c) b
  invZ : ∀ T σ c, Inv T σ → σ.cZ = some c → Piece.Z ∈ T →
    Inv (T.draw Piece.Z) (σ.bump2 c) ∧
    ∀ b, okB T σ b → okB (T.draw Piece.Z) (σ.bump2 c) b
  invO : ∀ T σ c, Inv T σ → σ.cO = some c → Piece.O ∈ T →
    Inv (T.draw Piece.O) (σ.bump2 c) ∧
    ∀ b, okB T σ b → okB (T.draw Piece.O) (σ.bump2 c) b
  /-- The open schedule content: T. -/
  stepT : ∀ T σ b, Inv T σ → okB T σ b → Piece.T ∈ T →
    ∃ pl σ', pl.piece = Piece.T ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
        = Board.debtBoard GameConfig.standard σ'.ρ σ'.ho ∧
      Inv (T.draw Piece.T) σ' ∧ okB (T.draw Piece.T) σ' b
  /-- The open schedule content: L. -/
  stepL : ∀ T σ b, Inv T σ → okB T σ b → Piece.L ∈ T →
    ∃ pl σ', pl.piece = Piece.L ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
        = Board.debtBoard GameConfig.standard σ'.ρ σ'.ho ∧
      Inv (T.draw Piece.L) σ' ∧ okB (T.draw Piece.L) σ' b
  /-- The open schedule content: J. -/
  stepJ : ∀ T σ b, Inv T σ → okB T σ b → Piece.J ∈ T →
    ∃ pl σ', pl.piece = Piece.J ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
        = Board.debtBoard GameConfig.standard σ'.ρ σ'.ho ∧
      Inv (T.draw Piece.J) σ' ∧ okB (T.draw Piece.J) σ' b
  /-- The open schedule content: I — drain when the base allows, else a
  band response (the bag-1 case). -/
  stepI : ∀ T σ b, Inv T σ → okB T σ b → Piece.I ∈ T →
    (4 ≤ b ∧ Inv (T.draw Piece.I) σ ∧ okB (T.draw Piece.I) σ (b - 4))
    ∨ (∃ pl σ', pl.piece = Piece.I ∧ pl.Valid GameConfig.standard ∧
        AvoidsWell well pl ∧
        pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
          = Board.debtBoard GameConfig.standard σ'.ρ σ'.ho ∧
        Inv (T.draw Piece.I) σ' ∧ okB (T.draw Piece.I) σ' b)
  /-- Bootstrap: S with no anchor (the bag-1 wiring; see the pack). -/
  stepSBoot : ∀ T σ b, Inv T σ → okB T σ b → Piece.S ∈ T → σ.cS = none →
    ∃ pl σ', pl.piece = Piece.S ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
        = Board.debtBoard GameConfig.standard σ'.ρ σ'.ho ∧
      Inv (T.draw Piece.S) σ' ∧ okB (T.draw Piece.S) σ' b
  /-- Bootstrap: Z with no anchor. -/
  stepZBoot : ∀ T σ b, Inv T σ → okB T σ b → Piece.Z ∈ T → σ.cZ = none →
    ∃ pl σ', pl.piece = Piece.Z ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
        = Board.debtBoard GameConfig.standard σ'.ρ σ'.ho ∧
      Inv (T.draw Piece.Z) σ' ∧ okB (T.draw Piece.Z) σ' b
  /-- Bootstrap: O with no anchor. -/
  stepOBoot : ∀ T σ b, Inv T σ → okB T σ b → Piece.O ∈ T → σ.cO = none →
    ∃ pl σ', pl.piece = Piece.O ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
        = Board.debtBoard GameConfig.standard σ'.ρ σ'.ho ∧
      Inv (T.draw Piece.O) σ' ∧ okB (T.draw Piece.O) σ' b

end Tetris
```

- [ ] **Step 2: Wire the import** — in `proofs/Proofs.lean`, after
`import Proofs.Safety.ShiftCertificate`, add:

```lean
import Proofs.Safety.BandSchedule
```

- [ ] **Step 3: Build** — `lake build`, expect success.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Safety/BandSchedule.lean proofs/Proofs.lean
git commit -m "proofs(band-schedule): BandState and the band-schedule certificate — the isolated 4-piece remainder, typed"
```

---

### Task 5: the reduction — anchored S/Z/O discharged, headline theorem

**Files:**
- Modify: `proofs/Proofs/Safety/BandSchedule.lean` (append before `end Tetris`)

**Interfaces:**
- Consumes: Task 4's structures; `ShiftCertificate` (fields `well hwell Q okBase init initBase anchored cover height step`); `place_vertS_skyline`, `place_vertZ_skyline`, `place_O_pair`, `place_debtBoard_of_flush`, `shapeUp_vertS`, `shapeUp_vertZ`, `shapeUp_O`.
- Produces: `BandScheduleCert.toShiftCertificate`, `Tetris.tetrisSolvableValid_of_bandSchedule`.

`ShiftCertificate.step`'s target shape (quote, for reference while writing
the case analysis):

```
(∃ pl ρ' ho', pl.piece = p ∧ pl.Valid GameConfig.standard ∧
    AvoidsWell well pl ∧
    pl.place (Board.debtBoard GameConfig.standard ρ ho)
      = Board.debtBoard GameConfig.standard ρ' ho' ∧
    Q (T.draw p) ρ' ho' ∧ okBase (T.draw p) ρ' ho' c)
∨ (p = Piece.I ∧ 4 ≤ c ∧ Q (T.draw p) ρ ho ∧ okBase (T.draw p) ρ ho (c - 4))
```

Also note `ShiftCertificate` has an `anchored` field
(`∀ T ρ ho, Q T ρ ho → ρ well = 0`) — map it from `C.anchored`.

- [ ] **Step 1: Write a shared discharge helper for the anchored window cases**

To avoid triple duplication, factor the common core:

```lean
namespace BandScheduleCert

variable (C : BandScheduleCert)

/-- The three anchored window responses share one discharge: a flush
2-column mechanism at the anchor + the debt-carry wrapper. -/
private theorem window_response {T : Bag} {σ : BandState} {c : ℕ} {p : Piece}
    (pl : Placement) (hpl : pl.piece = p)
    (hcols : c + 1 < GameConfig.standard.cols)
    (hcw : c ≠ C.well) (hc1w : c + 1 ≠ C.well)
    (hshape : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 = c ∨ pl.col + cell.1 = c + 1)
    (hInv : C.Inv T σ)
    (hflush : pl.place (skyline GameConfig.standard σ.ρ)
      = skyline GameConfig.standard (σ.bump2 c).ρ) :
    pl.Valid GameConfig.standard ∧ AvoidsWell C.well pl ∧
      pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
        = Board.debtBoard GameConfig.standard (σ.bump2 c).ρ (σ.bump2 c).ho := by
  refine ⟨?_, ?_, ?_⟩
  · intro cell hcell
    rcases hshape cell hcell with h | h <;> omega
  · intro cell hcell
    rcases hshape cell hcell with h | h <;> simp [h, hcw, hc1w]
  · rw [BandState.bump2_ho]
    exact place_debtBoard_of_flush hflush
      (fun x hx => ⟨(C.cover T σ x hInv hx).2.1, (C.cover T σ x hInv hx).2.2⟩)

end BandScheduleCert
```

Notes: `hshape` for each concrete piece comes from its `shapeUp` lemma
(`shapeUp_vertS c`, `shapeUp_vertZ c`, `shapeUp_O c 0`) by `fin_cases` +
`omega` at the call sites — read those lemmas' exact cell sets first; the
vertical pieces have `pl.col = c` and cells with `.1 ∈ {0, 1}`. The
`AvoidsWell` branch: `simp [h, hcw, hc1w]` may need
`exact h ▸ hcw`-style closers instead — adjust at build time. If the
`private` helper fights the structure namespace, make it a plain lemma.

- [ ] **Step 2: Write `toShiftCertificate`**

```lean
namespace BandScheduleCert

/-- **The reduction**: a band schedule yields a shift certificate. The
anchored S/Z/O cases are discharged here — the inhabitant never proves a
board equality for them. -/
def toShiftCertificate (C : BandScheduleCert) : ShiftCertificate where
  well := C.well
  hwell := C.hwell
  Q := fun T ρ ho => ∃ σ : BandState, C.Inv T σ ∧ σ.ρ = ρ ∧ σ.ho = ho
  okBase := fun T ρ ho b =>
    ∃ σ : BandState, C.Inv T σ ∧ σ.ρ = ρ ∧ σ.ho = ho ∧ C.okB T σ b
  init := ⟨BandState.start, C.init, rfl, rfl⟩
  initBase := ⟨BandState.start, C.init, rfl, rfl, C.initBase⟩
  anchored := by
    rintro T ρ ho ⟨σ, hInv, rfl, -⟩
    exact C.anchored T σ hInv
  cover := by
    rintro T ρ x ⟨σ, hInv, rfl, hho⟩
    exact C.cover T σ x hInv hho
  height := by
    rintro T ρ ho b hQ ⟨σ, hInv, rfl, -, hokB⟩ j hj
    exact C.height T σ b hInv hokB j hj
  step := by
    rintro T ρ ho b p hQ ⟨σ, hInv, rfl, rfl, hokB⟩ hp
    cases p with
    | S =>
        cases hcS : σ.cS with
        | none =>
            obtain ⟨pl, σ', h1, h2, h3, h4, h5, h6⟩ :=
              C.stepSBoot T σ b hInv hokB hp hcS
            exact Or.inl ⟨pl, σ'.ρ, σ'.ho, h1, h2, h3, h4,
              ⟨σ', h5, rfl, rfl⟩, ⟨σ', h5, rfl, rfl, h6⟩⟩
        | some c =>
            obtain ⟨hcols, hcw, hc1w⟩ := C.winColsS T σ c hInv hcS
            obtain ⟨hInv', hok'⟩ := C.invS T σ c hInv hcS hp
            have hwin := C.winS T σ c hInv hcS
            have hflush : Placement.place
                (skyline GameConfig.standard σ.ρ)
                { piece := Piece.S, rot := 1, col := c }
                = skyline GameConfig.standard (σ.bump2 c).ρ :=
              place_vertS_skyline (by omega) hcols hwin
            obtain ⟨hval, havd, hplace⟩ := C.window_response
              ({ piece := Piece.S, rot := 1, col := c }) rfl hcols hcw hc1w
              (by
                intro cell hcell
                rw [shapeUp_vertS c] at hcell
                fin_cases hcell <;> simp <;> omega) hInv hflush
            exact Or.inl ⟨_, (σ.bump2 c).ρ, (σ.bump2 c).ho, rfl, hval, havd,
              hplace, ⟨σ.bump2 c, hInv', rfl, rfl⟩,
              ⟨σ.bump2 c, hInv', rfl, rfl, hok' b hokB⟩⟩
    | Z =>
        cases hcZ : σ.cZ with
        | none =>
            obtain ⟨pl, σ', h1, h2, h3, h4, h5, h6⟩ :=
              C.stepZBoot T σ b hInv hokB hp hcZ
            exact Or.inl ⟨pl, σ'.ρ, σ'.ho, h1, h2, h3, h4,
              ⟨σ', h5, rfl, rfl⟩, ⟨σ', h5, rfl, rfl, h6⟩⟩
        | some c =>
            obtain ⟨hcols, hcw, hc1w⟩ := C.winColsZ T σ c hInv hcZ
            obtain ⟨hInv', hok'⟩ := C.invZ T σ c hInv hcZ hp
            have hwin := C.winZ T σ c hInv hcZ
            have hflush : Placement.place
                (skyline GameConfig.standard σ.ρ)
                { piece := Piece.Z, rot := 1, col := c }
                = skyline GameConfig.standard (σ.bump2 c).ρ :=
              place_vertZ_skyline (by omega) hcols hwin
            obtain ⟨hval, havd, hplace⟩ := C.window_response
              ({ piece := Piece.Z, rot := 1, col := c }) rfl hcols hcw hc1w
              (by
                intro cell hcell
                rw [shapeUp_vertZ c] at hcell
                fin_cases hcell <;> simp <;> omega) hInv hflush
            exact Or.inl ⟨_, (σ.bump2 c).ρ, (σ.bump2 c).ho, rfl, hval, havd,
              hplace, ⟨σ.bump2 c, hInv', rfl, rfl⟩,
              ⟨σ.bump2 c, hInv', rfl, rfl, hok' b hokB⟩⟩
    | O =>
        cases hcO : σ.cO with
        | none =>
            obtain ⟨pl, σ', h1, h2, h3, h4, h5, h6⟩ :=
              C.stepOBoot T σ b hInv hokB hp hcO
            exact Or.inl ⟨pl, σ'.ρ, σ'.ho, h1, h2, h3, h4,
              ⟨σ', h5, rfl, rfl⟩, ⟨σ', h5, rfl, rfl, h6⟩⟩
        | some c =>
            obtain ⟨hcols, hcw, hc1w⟩ := C.winColsO T σ c hInv hcO
            obtain ⟨hInv', hok'⟩ := C.invO T σ c hInv hcO hp
            have hpair := C.winO T σ c hInv hcO
            have hflush : Placement.place
                (skyline GameConfig.standard σ.ρ)
                { piece := Piece.O, rot := 0, col := c }
                = skyline GameConfig.standard (σ.bump2 c).ρ := by
              have h := place_O_pair (by omega) hcols hpair
              rw [h]
              unfold BandState.bump2
              congr 1
              rw [hpair]
            obtain ⟨hval, havd, hplace⟩ := C.window_response
              ({ piece := Piece.O, rot := 0, col := c }) rfl hcols hcw hc1w
              (by
                intro cell hcell
                rw [shapeUp_O c 0] at hcell
                fin_cases hcell <;> simp <;> omega) hInv hflush
            exact Or.inl ⟨_, (σ.bump2 c).ρ, (σ.bump2 c).ho, rfl, hval, havd,
              hplace, ⟨σ.bump2 c, hInv', rfl, rfl⟩,
              ⟨σ.bump2 c, hInv', rfl, rfl, hok' b hokB⟩⟩
    | T =>
        obtain ⟨pl, σ', h1, h2, h3, h4, h5, h6⟩ := C.stepT T σ b hInv hokB hp
        exact Or.inl ⟨pl, σ'.ρ, σ'.ho, h1, h2, h3, h4,
          ⟨σ', h5, rfl, rfl⟩, ⟨σ', h5, rfl, rfl, h6⟩⟩
    | L =>
        obtain ⟨pl, σ', h1, h2, h3, h4, h5, h6⟩ := C.stepL T σ b hInv hokB hp
        exact Or.inl ⟨pl, σ'.ρ, σ'.ho, h1, h2, h3, h4,
          ⟨σ', h5, rfl, rfl⟩, ⟨σ', h5, rfl, rfl, h6⟩⟩
    | J =>
        obtain ⟨pl, σ', h1, h2, h3, h4, h5, h6⟩ := C.stepJ T σ b hInv hokB hp
        exact Or.inl ⟨pl, σ'.ρ, σ'.ho, h1, h2, h3, h4,
          ⟨σ', h5, rfl, rfl⟩, ⟨σ', h5, rfl, rfl, h6⟩⟩
    | I =>
        rcases C.stepI T σ b hInv hokB hp with ⟨hb4, hInv', hok'⟩ |
          ⟨pl, σ', h1, h2, h3, h4, h5, h6⟩
        · exact Or.inr ⟨rfl, hb4, ⟨σ, hInv', rfl, rfl⟩,
            ⟨σ, hInv', rfl, rfl, hok'⟩⟩
        · exact Or.inl ⟨pl, σ'.ρ, σ'.ho, h1, h2, h3, h4,
            ⟨σ', h5, rfl, rfl⟩, ⟨σ', h5, rfl, rfl, h6⟩⟩

end BandScheduleCert

/-- **Inhabiting the band schedule proves Tetris solvable.** The remaining
open content of the whole development is the schedule: T/L/J, the I
disjunction, the bootstrap wiring, and the rate bookkeeping. -/
theorem tetrisSolvableValid_of_bandSchedule (C : BandScheduleCert) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_shiftCertificate C.toShiftCertificate
```

Adjustment notes for build-time friction:
1. `place_vertS_skyline`'s output profile is
   `Function.update (Function.update σ.ρ c (σ.ρ c + 2)) (c+1) (σ.ρ (c+1) + 2)`
   — definitionally `(σ.bump2 c).ρ`; if the `exact` needs help, insert
   `show … = skyline GameConfig.standard (σ.bump2 c).ρ` via
   `unfold BandState.bump2` on the goal.
2. The O flush: `place_O_pair`'s output has `σ.ρ c + 2` in the outer update;
   `congr 1; rw [hpair]` converts — if `congr 1` splits wrongly, use
   `funext`-free profile equality:
   `have : Function.update (Function.update σ.ρ c (σ.ρ c + 2)) (c+1) (σ.ρ c + 2) = (σ.bump2 c).ρ := by unfold BandState.bump2; rw [hpair]`
   then `rw [h, this]`. (`rw [hpair]` rewrites `σ.ρ c` under the update —
   it will ALSO rewrite the inner `σ.ρ c + 2`; if that breaks the match, use
   `conv` to target only the outer value, or `simp only [hpair]` with
   occurrence control; worst case `funext j; by_cases` + `Function.update`
   simp lemmas + `omega`.)
3. `window_response` is declared with `variable (C : …)` — call as
   `C.window_response …`; if dot-notation fails, call
   `BandScheduleCert.window_response C …`.
4. `fin_cases hcell <;> simp <;> omega` mirrors the Task-5 pattern from the
   translation-quotient plan; same fallbacks apply.
5. `cases hcS : σ.cS` inside the structure-field tactic proof: fine; the
   `hcS` equation is needed by `winColsS`/`invS`.

- [ ] **Step 3: Build** — `lake build`, expect success.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Safety/BandSchedule.lean
git commit -m "proofs(band-schedule): the reduction — anchored S/Z/O closure discharged, TetrisSolvable from a 4-piece band schedule"
```

---

### Task 6: gates, docs, final verification

**Files:**
- Modify: `proofs/PROGRESS.md` (new entry under `## Last tick`, above the translation-quotient entry)
- Modify: `proofs/LIBRARY.md` (Layer-3 table row + §4 tree)

- [ ] **Step 1: Gates**

Run: `scripts/check-green-clean.sh` — expect `green hygiene OK`.
Write to the scratchpad (NOT the repo) `axiom_gate_band.lean`:

```lean
import Proofs
open Tetris
#print axioms tetrisSolvableValid_of_bandSchedule
#print axioms Board.place_debtBoard_of_flush
#print axioms Board.place_horizI_flat
#print axioms Board.place_vertZ_afterS
```

Run: `lake env lean <scratchpad>/axiom_gate_band.lean` — every line exactly
`[propext, Classical.choice, Quot.sound]` (or a subset).
Run: `lake build ProofsExperiments` — expect success.

- [ ] **Step 2: PROGRESS.md** — insert under `## Last tick`:

```markdown
Tick (manual, 2026-07-11) — **the band-schedule reduction**:
`Invariants/BandMechanisms.lean` (debt-carry wrapper `place_debtBoard_of_flush` —
every flush skyline transition holds verbatim on debt-1 boards; `place_horizI_flat`
— the bag-1 I response; the bootstrap pack `sBoot`/`zBoot` + mutual-enabling
windows + `place_vertZ_afterS`/`place_vertS_afterZ` — the flat-S residue carries a
Z-window and vice versa, so exactly one hole is ever forced) +
`Safety/BandSchedule.lean` (`BandState` with Option window anchors, forced `bump2`
successors, `BandScheduleCert`; `toShiftCertificate` discharges the anchored S/Z/O
closure unconditionally; `tetrisSolvableValid_of_bandSchedule`). The open content
of the whole development is now: T/L/J responses, the I disjunction, the
bag-1 wiring, and okB rate bookkeeping — a 4-piece schedule. Both targets green,
axiom gate clean.
```

- [ ] **Step 3: LIBRARY.md** — in the Layer-3 table, after the
`ShiftCertificate` row, add:

```markdown
| `BandScheduleCert` + **`tetrisSolvableValid_of_bandSchedule`** (`Safety/BandSchedule`); debt-carry wrapper + bag-1 pack (`Invariants/BandMechanisms`) | the steady-state reduction: anchored S/Z/O closure discharged from the reproduction mechanisms; open remainder = T/L/J/I schedule + bootstrap wiring + rate bookkeeping |
```

and extend §4's tree: `Invariants/` line with `BandMechanisms`, `Safety/`
line with `BandSchedule`.

- [ ] **Step 4: Final build** — `lake build`, expect success.

- [ ] **Step 5: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/PROGRESS.md proofs/LIBRARY.md
git commit -m "proofs(band-schedule): progress + library map entries for the band-schedule reduction"
```

---

## Self-review notes (already applied)

- Spec coverage: Part A (wrapper Task 1, horizI Task 2, pack Task 3),
  Part B (Task 4), Part C (Task 5), acceptance/gates/docs (Task 6). The
  spec's `le_of_flush` helper is in Task 1. No toy inhabitant, per spec.
- Type consistency: `BandState.bump2 σ c`, `BandState.start`,
  `C.window_response`, field names `winS/winColsS/invS/stepT/stepSBoot`
  used consistently across Tasks 4–5; `b` is the base variable everywhere
  (never `c`, which is the anchor).
- Known uncertainties flagged in-task: I rot-0 cell literal (Task 2
  fallback), `shapeUp_vertS`/`shapeUp_vertZ` exact cell sets (read before
  Task 5), the O-profile rewrite under `Function.update` (Task 5 note 2),
  `place_O_pair`'s exact location/hypothesis order.
