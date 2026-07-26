# Plinth Foundation (PlinthShift + PlinthCert) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The corrected, inhabitable certificate foundation: `Proofs/Invariants/PlinthShift.lean` (floor-1 lift, T1′ transport, T2′ mid-row drain, well-plug witness) and `Proofs/Safety/PlinthCert.lean` (re-anchoring `Mem`, two-regime `PlinthCert`, direct reduction to `DebtCertificate`), plus the D1/D2 findings recorded.

**Architecture:** The plinth regime pins the well at height 1 and the entombed hole at `(hx, 0)`; the active band rides `c+1` above the floor in coordinates the hole never touches, so re-anchoring `(ρ,c) ↦ (ρ−d, c+d)` is unobstructed (fixes D2), and every step obligation concludes in membership-up-to-re-anchor `PlinthMem` (fixes D1). Rep obligations stay bare-skyline so all existing mechanisms reuse. Spec: `docs/superpowers/specs/2026-07-12-plinth-cert-design.md`.

**Tech Stack:** Lean 4 + mathlib (pinned), lake. `namespace Tetris` / `Tetris.Board`.

## Global Constraints

- Green target: no `sorry`, no `native_decide`, no new axioms — every new theorem exactly `[propext, Classical.choice, Quot.sound]` or fewer.
- `lake` builds foreground only from `proofs/`; never SIGTERM; rebuild after every edit.
- Commits: one per task, staging only `proofs/`. Messages `proofs(plinth-shift): …` / `proofs(plinth-cert): …`.
- Docstrings: never `word-/word`; ~100-char lines.
- Key verified identifiers:
  - `Board.clearLines cfg b = (b.filter (fun p => ¬ isFull cfg b p.2)).image (fun p => (p.1, p.2 - clearedBelow cfg b p.2))`; `clearedBelow cfg b r = ((fullRows cfg b).filter (· < r)).card`; `fullRows cfg b = (b.image (·.2)).filter (isFull cfg b ·)`; `isFull cfg b r = ∀ c ∈ Finset.range cfg.cols, (c, r) ∈ b` (`Model/Board.lean:77–95`)
  - `Seam.drain_applyStep` proof pattern (`SeamBridge.lean:202`): characterize `isFull` post-place, rewrite the filter, `Finset.image_congr`
  - From `Invariants/BandShift.lean`: `mem_debtBoard`, `sup_sub_add_shift`, `mem_cellsAt`, `mem_cellsAt_add`, `bandDrain`, `bandDrain_shapeUp`, `bandDrain_col`
  - From `Invariants/BandMechanisms.lean`: `le_of_flush`; mechanisms `place_vertS_skyline`/`place_vertZ_skyline` (`Skyline.lean:439/474`), `place_O_pair` (`HoledSkyline.lean:1310`), `shapeUp_vertS/vertZ` (`Skyline.lean:423/430`), `shapeUp_O`
  - `Piece.shape J 3 = {(0,2),(1,0),(1,1),(1,2)}` so `shapeUp J 3 = {(0,0),(1,0),(1,1),(1,2)}` (col 0: row 0 only; col 1: rows 0–2)
  - `DebtCertificate` (`Safety/SkylineInvariant.lean:126`): fields `P init cover height step`; `step`'s response is `applyStep`-shaped
  - `BandState`/`BandState.bump2`/`bump2_ho` (`Safety/BandSchedule.lean`); `Seam.AvoidsWell`; `clearLines_eq_self_of_no_fullRows (cfg) (h : fullRows cfg b = ∅)` (`BoardCount.lean:822`)
  - `colHeight_debtBoard (hcov : ∀ x, ho = some x → x.2 + 1 < h x.1)` (`HoledSkyline.lean:578`), `colHeight_skyline (hj)`
- The Lean test cycle is the build; transient `sorry` never committed.

---

### Task 1: PlinthShift basics — lift, re-anchoring, no-full-rows, well plug

**Files:**
- Create: `proofs/Proofs/Invariants/PlinthShift.lean`
- Modify: `proofs/Proofs.lean` (import after `import Proofs.Invariants.BandMechanisms`)

**Interfaces:**
- Consumes: `mem_debtBoard`, `mem_skyline'`, flat-witness proof pattern.
- Produces (Tasks 2–5): `Board.plinthLift (w c : ℕ) (ρ : ℕ → ℕ) : ℕ → ℕ`, `plinthLift_well`, `plinthLift_ne`, `Board.ReanchorsTo`, `Board.plinthLift_congr_reanchor`, `Board.fullRows_plinth_eq_empty`, `Board.shapeUp_wellPlugJ`, `Board.place_wellPlug_flat`.

- [ ] **Step 1: Write the module**

```lean
import Proofs.Invariants.BandShift
import Proofs.Invariants.BandMechanisms

/-!
# The plinth: floor-1 transport for the entombed-hole regime

Findings D1/D2 (see PROGRESS.md, 2026-07-12): the `bandLift` certificates
cannot be inhabited — the base never rises, and the forced row-0 bootstrap
hole blocks re-anchoring. The plinth regime fixes both: one bag-1 placement
(`place_wellPlug_flat`) plugs the well's row 0; row 0 becomes a permanent
floor of nine cells plus the entombed hole, which keeps row 0 from ever
being full — the floor is immortal. The well operates at height 1, drains
fill and clear rows 1–4 (`drain_debtBoard_plinthLift`), and the active band
rides `c + 1` above the floor in coordinates the hole never touches, so
re-anchoring (`ReanchorsTo`) is unobstructed.
-/

namespace Tetris
namespace Board

/-- The plinth lift: well pinned at height 1 (the plug), band riding `c + 1`
above the floor. -/
def plinthLift (w c : ℕ) (ρ : ℕ → ℕ) : ℕ → ℕ :=
  fun j => if j = w then 1 else ρ j + c + 1

@[simp] theorem plinthLift_well (w c : ℕ) (ρ : ℕ → ℕ) :
    plinthLift w c ρ w = 1 := if_pos rfl

theorem plinthLift_ne (w c : ℕ) (ρ : ℕ → ℕ) {j : ℕ} (hj : j ≠ w) :
    plinthLift w c ρ j = ρ j + c + 1 := if_neg hj

/-- Re-anchoring: the same absolute band at a different base split. The
board-level no-op that lets the base rise as the pattern grows (fix for
finding D1). -/
def ReanchorsTo (well : ℕ) (ρ : ℕ → ℕ) (c : ℕ) (ρ' : ℕ → ℕ) (c' : ℕ) : Prop :=
  (∀ j, j ≠ well → ρ j + c = ρ' j + c') ∧ ρ' well = 0

/-- Re-anchored splits denote the same plinth profile. -/
theorem plinthLift_congr_reanchor {well : ℕ} {ρ ρ' : ℕ → ℕ} {c c' : ℕ}
    (h : ReanchorsTo well ρ c ρ' c') :
    plinthLift well c ρ = plinthLift well c' ρ' := by
  funext j
  by_cases hj : j = w  -- NOTE: use the actual binder name `well`
  · subst hj; simp [plinthLift]
  · rw [plinthLift_ne well c ρ hj, plinthLift_ne well c' ρ' hj]
    have := h.1 j hj
    omega

/-- **The floor is immortal.** A plinth board has no full rows: the well
column stops at height 1 (blocking every row ≥ 1) and the entombed hole
blocks row 0. -/
theorem fullRows_plinth_eq_empty {cfg : GameConfig} {w hx c : ℕ}
    {ρ : ℕ → ℕ} (hw : w < cfg.cols) (hxw : hx ≠ w) :
    fullRows cfg (debtBoard cfg (plinthLift w c ρ) (some (hx, 0))) = ∅ := by
  rw [Finset.eq_empty_iff_forall_notMem]
  intro r hr
  unfold fullRows at hr
  rw [Finset.mem_filter] at hr
  have hfull := hr.2
  by_cases hr0 : r = 0
  · subst hr0
    have := hfull hx (Finset.mem_range.mpr (by omega))
    -- impossible: (hx, 0) is the hole
    rw [mem_debtBoard] at this
    exact (this.1 (hx, 0) rfl) rfl
  · have := hfull w (Finset.mem_range.mpr hw)
    rw [mem_debtBoard] at this
    have hlt := this.2.2
    simp only [plinthLift_well] at hlt
    omega

/-- The well-plug shape: `J` rot 3 — one cell in its left column at row 0,
three cells in the right column rows 0–2. -/
theorem shapeUp_wellPlugJ (c : ℕ) :
    ({ piece := Piece.J, rot := 3, col := c } : Placement).shapeUp
      = {((0 : ℕ), (0 : ℕ)), (1, 0), (1, 1), (1, 2)} := by
  show Piece.shapeUp Piece.J 3 = _
  decide

/-- **The well plug**: `J` rot 3 straddling the well boundary on flat ground
— exactly one cell lands at `(w, base)` (the plug), three seat flush on the
neighbor. No hole is created. The bag-1 entombment move (finding D2). -/
theorem place_wellPlug_flat (cfg : GameConfig) (base w : ℕ)
    (hw1 : w + 1 < cfg.cols) :
    ({ piece := Piece.J, rot := 3, col := w } : Placement).place
        (skyline cfg (fun _ => base))
      = skyline cfg (fun j =>
          if j = w then base + 1 else if j = w + 1 then base + 3 else base) := by
  have hsh := shapeUp_wellPlugJ w
  have hc0 : w + 0 < cfg.cols := by omega
  have hd : ({ piece := Piece.J, rot := 3, col := w } : Placement).dropOffset
      (skyline cfg (fun _ => base)) = base := by
    rw [Placement.dropOffset_eq_sup, hsh]
    simp only [Finset.sup_insert, Finset.sup_singleton,
      colHeight_skyline hc0, colHeight_skyline hw1]
    omega
  have hdr : ({ piece := Piece.J, rot := 3, col := w } : Placement).dropped
      (skyline cfg (fun _ => base))
      = {(w, base), (w + 1, base), (w + 1, base + 1), (w + 1, base + 2)} := by
    rw [Placement.dropped_eq_image, hsh, hd]
    simp only [Finset.image_insert, Finset.image_singleton]
    norm_num
  rw [Placement.place_eq_union_dropped, hdr]
  ext ⟨a, b⟩
  simp only [Finset.mem_union, mem_skyline', Finset.mem_insert,
    Finset.mem_singleton, Prod.mk.injEq]
  split_ifs <;> omega

end Board
end Tetris
```

Fix the noted binder in `plinthLift_congr_reanchor` (`by_cases hj : j = well`).
If `decide` in `shapeUp_wellPlugJ` fails, the rot-3 literal differs — read
`Piece.shape` (`Model/Piece.lean:40`) and correct set + `hdr` + profile.

- [ ] **Step 2: Wire the import** — in `proofs/Proofs.lean` after
`import Proofs.Invariants.BandMechanisms`:

```lean
import Proofs.Invariants.PlinthShift
```

- [ ] **Step 3: Build** — `lake build`, expect success.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Invariants/PlinthShift.lean proofs/Proofs.lean
git commit -m "proofs(plinth-shift): the plinth lift, re-anchoring, the immortal floor, and the well plug"
```

---

### Task 2: T1′ — plinth place transport

**Files:**
- Modify: `proofs/Proofs/Invariants/PlinthShift.lean` (append inside `namespace Board`)

**Interfaces:**
- Consumes: Task 1; `sup_sub_add_shift`, `mem_debtBoard`, `mem_cellsAt`, `mem_cellsAt_add`, `colHeight_debtBoard`, `colHeight_skyline`, `Placement.shapeUp_exists_bottom` (via `Placement.` in Confluence, already imported through BandShift).
- Produces (Tasks 3, 5): `Board.dropOffset_plinthLift`, `Board.place_debtBoard_plinthLift` with the signatures below.

- [ ] **Step 1: Write the drop-offset shift**

```lean
/-- Hard-drop shift onto the plinth: the offset is the bare-skyline offset
plus `c + 1` (the entombed hole is strictly covered, hence invisible). -/
theorem dropOffset_plinthLift {cfg : GameConfig} {w hx c : ℕ}
    {ρ : ℕ → ℕ} {pl : Placement}
    (hcols : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 < cfg.cols)
    (havoid : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 ≠ w)
    (hhx : hx ≠ w ∧ hx < cfg.cols ∧ 1 ≤ ρ hx) :
    pl.dropOffset (debtBoard cfg (plinthLift w c ρ) (some (hx, 0)))
      = pl.dropOffset (skyline cfg ρ) + (c + 1) := by
  have hcovL : ∀ x, (some ((hx : ℕ), (0 : ℕ)) : Option Coord) = some x →
      x.2 + 1 < plinthLift w c ρ x.1 := by
    rintro x hx'
    cases hx'
    rw [plinthLift_ne w c ρ hhx.1]
    omega
  have hL : pl.dropOffset (debtBoard cfg (plinthLift w c ρ) (some (hx, 0)))
      = pl.shapeUp.sup (fun cell => ρ (pl.col + cell.1) + (c + 1) - cell.2) := by
    rw [Placement.dropOffset_eq_sup]
    refine Finset.sup_congr rfl fun cell hcell => ?_
    rw [colHeight_debtBoard hcovL, if_pos (hcols cell hcell),
      plinthLift_ne w c ρ (havoid cell hcell)]
    ring_nf
  have hR : pl.dropOffset (skyline cfg ρ)
      = pl.shapeUp.sup (fun cell => ρ (pl.col + cell.1) - cell.2) := by
    rw [Placement.dropOffset_eq_sup]
    refine Finset.sup_congr rfl fun cell hcell => ?_
    rw [colHeight_skyline (hcols cell hcell)]
  rw [hL, hR]
  exact sup_sub_add_shift pl.shapeUp (fun cell => ρ (pl.col + cell.1)) (c + 1)
    (Placement.shapeUp_exists_bottom pl.piece pl.rot)
```

Note: the `ring_nf` normalizes `ρ j + c + 1 - cell.2` vs
`ρ j + (c + 1) - cell.2`; if it misfires use
`show ρ (pl.col + cell.1) + c + 1 - cell.2 = _` reshaping or state `hL`'s
sup with `+ c + 1` and instantiate `sup_sub_add_shift` accordingly (it is
associativity-transparent to `omega` inside that lemma; the cleanest route
is to write BOTH sup functions as `… + (c + 1) - cell.2` and avoid `ring_nf`
by stating `plinthLift_ne` output as `ρ j + c + 1` then
`rw [show ρ (pl.col + cell.1) + c + 1 = ρ (pl.col + cell.1) + (c + 1) from by omega]`).

- [ ] **Step 2: Write T1′**

```lean
/-- **T1′ — plinth transport.** A bare flush skyline transition holds on the
plinth board at every base: landing cells ride `c + 1` above the floor, the
well column is `{(w, 0)}` on both sides, and the entombed hole never moves.
Rep obligations are bare-skyline, so every existing mechanism applies. -/
theorem place_debtBoard_plinthLift {cfg : GameConfig} {w hx c : ℕ}
    {ρ ρ' : ℕ → ℕ} {pl : Placement}
    (hcols : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 < cfg.cols)
    (havoid : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 ≠ w)
    (hhx : hx ≠ w ∧ hx < cfg.cols ∧ 1 ≤ ρ hx)
    (hrep : pl.place (skyline cfg ρ) = skyline cfg ρ') :
    pl.place (debtBoard cfg (plinthLift w c ρ) (some (hx, 0)))
      = debtBoard cfg (plinthLift w c ρ') (some (hx, 0)) := by
  have hD := dropOffset_plinthLift (cfg := cfg) (w := w) (hx := hx) (c := c)
    (ρ := ρ) (pl := pl) hcols havoid hhx
  have hpt := fun p => Finset.ext_iff.mp hrep p
  ext ⟨a, r⟩
  simp only [Placement.place_eq_union_dropped, Finset.mem_union,
    Placement.dropped_eq_cellsAt, hD, mem_cellsAt_add, mem_debtBoard]
  have hrep_ar := hpt (a, r - (c + 1))
  simp only [Placement.place_eq_union_dropped, Finset.mem_union,
    Placement.dropped_eq_cellsAt, mem_skyline'] at hrep_ar
  by_cases haw : a = w
  · -- Column w: exactly {(w, 0)} on both sides.
    subst haw
    constructor
    · rintro (⟨hne, hac, hlt⟩ | ⟨-, hmem⟩)
      · rw [plinthLift_well] at hlt
        exact ⟨hne, hac, by rw [plinthLift_well]; omega⟩
      · rcases mem_cellsAt.mp hmem with ⟨cell, hcell, h1, -⟩
        exact absurd h1 (havoid cell hcell)
    · rintro ⟨hne, hac, hlt⟩
      rw [plinthLift_well] at hlt
      exact Or.inl ⟨hne, hac, by rw [plinthLift_well]; omega⟩
  · by_cases hrc : r < c + 1
    · -- The slab (rows ≤ c): filled on both sides except the hole point.
      by_cases hhole : a = hx ∧ r = 0
      · obtain ⟨rfl, rfl⟩ := hhole
        constructor
        · rintro (⟨hne, -, -⟩ | ⟨-, hmem⟩)
          · exact absurd rfl (hne (hx, 0) rfl)
          · rcases mem_cellsAt.mp hmem with ⟨cell, hcell, h1, h2⟩
            -- dropped cells sit at rows ≥ dropOffset ≥ 0 + (c+1) > 0? No:
            -- rows are D + (c+1) + cell.2 ≥ c + 1 ≥ 1 > 0; contradiction with r = 0
            omega
        · rintro ⟨hne, -, -⟩
          exact absurd rfl (hne (hx, 0) rfl)
      · constructor
        · rintro (⟨-, hac, -⟩ | ⟨hcr, -⟩)
          · refine Or.inl ⟨?_, hac, ?_⟩
            · rintro x hx' hcontra
              cases hx'
              apply hhole
              obtain ⟨h1, h2⟩ := Prod.mk.injEq .. ▸ hcontra
              exact ⟨h1, h2⟩
            · rw [plinthLift_ne w c ρ' haw]; omega
          · omega
        · rintro ⟨-, hac, -⟩
          refine Or.inl ⟨?_, hac, ?_⟩
          · rintro x hx' hcontra
            cases hx'
            apply hhole
            obtain ⟨h1, h2⟩ := Prod.mk.injEq .. ▸ hcontra
            exact ⟨h1, h2⟩
          · rw [plinthLift_ne w c ρ haw]; omega
    · -- Above the slab: transfer through the bare representative.
      push_neg at hrc
      have hne_auto : ∀ (side : ℕ → ℕ),
          (∀ x, (some ((hx : ℕ), (0 : ℕ)) : Option Coord) = some x →
            ((a : ℕ), (r : ℕ)) ≠ x) := by
        intro _ x hx' hcontra
        cases hx'
        have h2 := congrArg Prod.snd hcontra
        simp only at h2
        omega
      rw [plinthLift_ne w c ρ haw, plinthLift_ne w c ρ' haw]
      constructor
      · rintro (⟨-, h2, h3⟩ | ⟨-, hmem⟩)
        · have hout := hrep_ar.mp (Or.inl ⟨h2, by omega⟩)
          exact ⟨hne_auto ρ', hout.1, by omega⟩
        · have hout := hrep_ar.mp (Or.inr hmem)
          exact ⟨hne_auto ρ', hout.1, by omega⟩
      · rintro ⟨-, h2, h3⟩
        rcases hrep_ar.mpr ⟨h2, by omega⟩ with ⟨g2, g3⟩ | hmem
        · exact Or.inl ⟨hne_auto ρ, g2, by omega⟩
        · exact Or.inr ⟨hrc, hmem⟩
```

**This block is the shape, not a verified script** — expect the usual
frictions and fix at build time:
- The rep side (`hrep_ar`) is about BARE skylines (`mem_skyline'`), so its
  membership triples are 2-conjunct (`a < cols ∧ row < ρ a`) — the
  destructuring above reflects that; adjust arity if elaboration differs.
- `Prod.mk.injEq .. ▸ hcontra` may need the explicit
  `rw [Prod.mk.injEq] at hcontra` form, or replace the whole hole-≠ argument
  with `intro hcontra; cases hcontra; …` plus `omega` on components.
- The `hne_auto` helper as written takes a dummy argument to be usable on
  both sides; if it fights elaboration, inline the 5-line argument twice.
- The dropped-cell row bound in the hole-point case: `mem_cellsAt` gives
  `d + cell.2 = r` with `d = D + (c+1)`; `omega` closes from `r = 0`.

- [ ] **Step 3: Build** — `lake build`, expect success, no `sorry` in file.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Invariants/PlinthShift.lean
git commit -m "proofs(plinth-shift): T1' — bare flush transitions transport onto the plinth at every base"
```

---

### Task 3: T2′ — the mid-row drain

**Files:**
- Modify: `proofs/Proofs/Invariants/PlinthShift.lean` (append inside `namespace Board`)

**Interfaces:**
- Consumes: Task 1–2; `bandDrain`, `bandDrain_shapeUp`, `bandDrain_col`; raw `clearLines`/`isFull`/`fullRows`/`clearedBelow` defs; the `Seam.drain_applyStep` proof pattern.
- Produces (Task 5): `Board.drain_debtBoard_plinthLift` with the signature below.

- [ ] **Step 1: Write the landing computation**

```lean
/-- The drain lands on the plug: offset 1, filling well rows 1–4. -/
theorem bandDrain_place_plinth {cfg : GameConfig} {w hx c : ℕ} {ρ : ℕ → ℕ}
    (hw : w < cfg.cols) (hhx : hx ≠ w ∧ hx < cfg.cols ∧ 1 ≤ ρ hx) :
    (bandDrain w).place (debtBoard cfg (plinthLift w c ρ) (some (hx, 0)))
      = debtBoard cfg (fun j => if j = w then 5 else ρ j + c + 1)
          (some (hx, 0)) := by
  -- (i) colHeight at w = 1 (colHeight_debtBoard + plinthLift_well)
  -- (ii) dropOffset = 1: sup of (1 - k) for k = 0..3 = 1
  -- (iii) dropped = {(w,1),(w,2),(w,3),(w,4)}
  -- (iv) ext with mem_debtBoard: column w gains rows 1–4 (row 0 was the plug,
  --      heights 1 → 5); everything else and the hole untouched; omega.
  sorry_replace_with_proof
```

Recipe (mirror `bandDrain_place` in `BandShift.lean` exactly, shifted by 1):
`hcovL` as in Task 2; `hcolw : colHeight … w = 1` via
`colHeight_debtBoard hcovL, if_pos hw, plinthLift_well`;
`hD : dropOffset = 1` via `dropOffset_eq_sup, bandDrain_shapeUp`,
`simp only [Finset.sup_insert, Finset.sup_singleton, Nat.add_zero, bandDrain_col, hcolw]`
then close `1 ⊔ (1-1) ⊔ … = 1` with `Nat.sub_self`-friendly simp or `omega`-
free `simp [Nat.sub_succ]`; worst case `decide`-free explicit `max` lemmas.
`hdr : dropped = {(w,1),(w,2),(w,3),(w,4)}` via `dropped_eq_image, hD` +
`simp only [Finset.image_insert, Finset.image_singleton]` + `norm_num`.
Final ext: `by_cases haw : a = w` with `plinthLift_well/plinthLift_ne` +
hole normalization (`cases`-free: hole ≠ column w) + `omega`.

- [ ] **Step 2: Write T2′**

```lean
/-- **T2′ — the plinth drain.** At `4 ≤ c` the vertical I fills well rows
1–4; rows 1–4 are full while row 0 never is (the entombed hole): exactly
rows 1–4 clear, the band drops to `c − 4`, plinth and hole intact. -/
theorem drain_debtBoard_plinthLift {cfg : GameConfig} {w hx c : ℕ}
    {ρ : ℕ → ℕ} (hw : w < cfg.cols) (hc : 4 ≤ c)
    (hhx : hx ≠ w ∧ hx < cfg.cols ∧ 1 ≤ ρ hx) :
    (bandDrain w).applyStep cfg
        (debtBoard cfg (plinthLift w c ρ) (some (hx, 0)))
      = debtBoard cfg (plinthLift w (c - 4) ρ) (some (hx, 0)) := by
  rw [Placement.applyStep_eq_clearLines_place, bandDrain_place_plinth hw hhx]
  -- Set B := the pre-clear board (the debtBoard with well at 5).
  -- 1. hfull_iff : isFull cfg B r ↔ 1 ≤ r ∧ r ≤ 4
  -- 2. Unfold clearLines; rewrite the filter and compute clearedBelow
  --    pointwise (0 for r = 0; 4 for r ≥ 5); survivors are row 0 and rows ≥ 5.
  -- 3. ext ⟨a, s⟩; mem_image/mem_filter/mem_debtBoard; omega per case.
  sorry_replace_with_proof
```

Detailed recipe (this is the session's main proof; ~150–250 lines):
1. Abbreviate `B := debtBoard cfg (fun j => if j = w then 5 else ρ j + c + 1) (some (hx, 0))`
   (a `set … with hB` or just repeat the term).
2. `hmem : ∀ a r, ((a, r) : Coord) ∈ B ↔ ((a, r) ≠ (hx, 0)) ∧ a < cfg.cols ∧
   r < (if a = w then 5 else ρ a + c + 1))` — instance of `mem_debtBoard`
   (with the `∀ x, some … = some x → …` hole condition collapsed via
   `forall_eq'`-style simp; a small `have` doing
   `simp only [mem_debtBoard, Option.some.injEq, forall_eq']` is enough).
3. `hfull_iff : ∀ r, isFull cfg B r ↔ 1 ≤ r ∧ r ≤ 4`:
   - forward: `r = 0` → instantiate at `hx` → the hole contradiction;
     `r ≥ 5` → instantiate at `w` → `5 ≤ r < 5` contradiction (well height 5).
   - backward: for `1 ≤ r ≤ 4` and any `col < cols`: membership by `hmem` +
     `omega` (band: `r ≤ 4 ≤ c < ρ a + c + 1`; well: `r < 5`; hole: `r ≠ 0`).
4. `hfullRows : fullRows cfg B = {1, 2, 3, 4}`:
   ext + `Finset.mem_filter`/`mem_image` + `hfull_iff`; the image-side
   witness for membership is `(w, r) ∈ B` (well column has rows 0–4).
5. `hcb0 : clearedBelow cfg B 0 = 0` and
   `hcb5 : ∀ r, 5 ≤ r → clearedBelow cfg B r = 4`:
   `unfold clearedBelow; rw [hfullRows]` then
   `Finset.filter` on the literal `{1,2,3,4}`: for `r = 0` the filter is
   empty (`Finset.filter_false_of_mem` + `decide`-free omega); for `r ≥ 5`
   the filter keeps everything (`Finset.filter_true_of_mem`), `card = 4`
   (`rfl` or `simp`).
6. Unfold `clearLines`; `ext ⟨a, s⟩`;
   `simp only [Finset.mem_image, Finset.mem_filter]` on the left and
   `hmem`-style membership on the right. Forward: a surviving pre-image
   `(a₀, r₀)` has `¬ isFull … r₀` → (`hfull_iff`) `r₀ = 0 ∨ 5 ≤ r₀`;
   case `r₀ = 0`: maps to `(a₀, 0)` (via `hcb0`), and `(a₀, 0) ∈ B` gives the
   target membership at row 0 (`plinthLift` band/well both ≥ 1; hole
   excluded by the carried `≠`); case `r₀ ≥ 5`: maps to `(a₀, r₀ − 4)` (via
   `hcb5`), heights: `r₀ < ρ a₀ + c + 1 ↔ r₀ − 4 < ρ a₀ + (c−4) + 1`
   (`omega`, uses `4 ≤ c`); well: `r₀ < 5` contradiction so no well cells
   arrive above row 0. Backward: target `(a, s)`: `s = 0` → pre-image
   `(a, 0)`; `s ≥ 1` → pre-image `(a, s + 4)` (heights by `omega`; the
   `¬ isFull` side conditions from `hfull_iff` + `omega`).
7. Close with the hole bookkeeping carried through each case (the target
   hole condition `(a, s) ≠ (hx, 0)` ↔ pre-image condition, `omega` on rows).

If step 6's single `ext` balloons, factor
`hfilter : B.filter (fun p => ¬ isFull cfg B p.2) = B.filter (fun p => p.2 = 0 ∨ 5 ≤ p.2)`
first (mirroring `Seam.drain_applyStep`'s `hfilter`), then do the image
analysis on the cleaner set.

- [ ] **Step 3: Build** — `lake build`, expect success, no `sorry`.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Invariants/PlinthShift.lean
git commit -m "proofs(plinth-shift): T2' — the mid-row drain; rows 1-4 clear, the floor and hole are immortal"
```

---

### Task 4: PlinthCert — the two-regime certificate

**Files:**
- Create: `proofs/Proofs/Safety/PlinthCert.lean`
- Modify: `proofs/Proofs.lean` (import after `import Proofs.Safety.BandSchedule`)

**Interfaces:**
- Consumes: `BandState`, `BandState.bump2` (`Safety/BandSchedule.lean`); `DebtCertificate` (`Safety/SkylineInvariant.lean`); `plinthLift`, `ReanchorsTo` (Task 1); `Seam.AvoidsWell`.
- Produces (Task 5): `Tetris.PlinthMem`, `Tetris.PlinthCert` with the exact fields below.

- [ ] **Step 1: Write the module (structures only)**

```lean
import Proofs.Safety.BandSchedule
import Proofs.Invariants.PlinthShift

/-!
# The plinth certificate — the corrected inhabitation target

Two regimes: `Boot` (raw bag-1 boards, absolute level) and the plinth
steady state (`Inv`/`okB` over `BandState` representatives riding above the
immortal floor). Every steady obligation concludes in membership up to
re-anchoring (`PlinthMem`) — the D1 fix — and the entombed hole lives
outside the pattern coordinates — the D2 fix. Reduces directly to
`DebtCertificate` (plinth boards have well height 1, outside
`ShiftCertificate`'s image).
-/

namespace Tetris

open Board Seam

/-- Membership up to re-anchoring: the successor may re-split its absolute
band into (pattern, base) any equivalent way. Hole slot and window anchors
survive (heights shift uniformly; columns do not move). -/
def PlinthMem (well : ℕ) (Inv : Bag → BandState → Prop)
    (okB : Bag → BandState → ℕ → Prop)
    (T : Bag) (σ : BandState) (c : ℕ) : Prop :=
  ∃ σ' c', ReanchorsTo well σ.ρ c σ'.ρ c' ∧ σ'.ho = σ.ho ∧
    σ'.cS = σ.cS ∧ σ'.cZ = σ.cZ ∧ σ'.cO = σ.cO ∧ Inv T σ' ∧ okB T σ' c'

/-- **The plinth certificate.** Boot: raw bag-1 family from the empty board
through the well plug and the forced hole into the plinth. Steady: the
band-schedule obligations over the immortal floor, with re-anchoring. -/
structure PlinthCert where
  well : ℕ
  hwell : well < GameConfig.standard.cols
  /-- The entombed hole's column. -/
  hx : ℕ
  hhx : hx ≠ well ∧ hx < GameConfig.standard.cols
  /-- The bag-1 (pre-plinth) family, at the absolute level. -/
  Boot : Bag → (ℕ → ℕ) → Option Coord → Prop
  bootInit : Boot Bag.full (fun _ => 0) none
  bootCover : ∀ T h x, Boot T h (some x) →
    x.1 < GameConfig.standard.cols ∧ x.2 + 1 < h x.1
  bootHeight : ∀ T h ho, Boot T h ho →
    ∀ j < GameConfig.standard.cols, h j ≤ GameConfig.standard.rows
  /-- The steady family and its base predicate. -/
  Inv : Bag → BandState → Prop
  okB : Bag → BandState → ℕ → Prop
  /-- Steady states carry no extra hole (the debt slot is the entombed hole). -/
  hoNone : ∀ T σ, Inv T σ → σ.ho = none
  winS : ∀ T σ c, Inv T σ → σ.cS = some c → σ.ρ c = σ.ρ (c + 1) + 1
  winZ : ∀ T σ c, Inv T σ → σ.cZ = some c → σ.ρ (c + 1) = σ.ρ c + 1
  winO : ∀ T σ c, Inv T σ → σ.cO = some c → σ.ρ c = σ.ρ (c + 1)
  winColsS : ∀ T σ c, Inv T σ → σ.cS = some c →
    c + 1 < GameConfig.standard.cols ∧ c ≠ well ∧ c + 1 ≠ well
  winColsZ : ∀ T σ c, Inv T σ → σ.cZ = some c →
    c + 1 < GameConfig.standard.cols ∧ c ≠ well ∧ c + 1 ≠ well
  winColsO : ∀ T σ c, Inv T σ → σ.cO = some c →
    c + 1 < GameConfig.standard.cols ∧ c ≠ well ∧ c + 1 ≠ well
  anchored : ∀ T σ, Inv T σ → σ.ρ well = 0
  /-- The hole column stays loaded (keeps the entombed hole covered). -/
  holeLoaded : ∀ T σ, Inv T σ → 1 ≤ σ.ρ hx
  height : ∀ T σ b, Inv T σ → okB T σ b →
    ∀ j < GameConfig.standard.cols,
      Board.plinthLift well b σ.ρ j ≤ GameConfig.standard.rows
  invS : ∀ T σ c b, Inv T σ → okB T σ b → σ.cS = some c → Piece.S ∈ T →
    PlinthMem well Inv okB (T.draw Piece.S) (σ.bump2 c) b
  invZ : ∀ T σ c b, Inv T σ → okB T σ b → σ.cZ = some c → Piece.Z ∈ T →
    PlinthMem well Inv okB (T.draw Piece.Z) (σ.bump2 c) b
  invO : ∀ T σ c b, Inv T σ → okB T σ b → σ.cO = some c → Piece.O ∈ T →
    PlinthMem well Inv okB (T.draw Piece.O) (σ.bump2 c) b
  stepT : ∀ T σ b, Inv T σ → okB T σ b → Piece.T ∈ T →
    ∃ pl σ', pl.piece = Piece.T ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.skyline GameConfig.standard σ.ρ)
        = Board.skyline GameConfig.standard σ'.ρ ∧ σ'.ho = none ∧
      PlinthMem well Inv okB (T.draw Piece.T) σ' b
  stepL : ∀ T σ b, Inv T σ → okB T σ b → Piece.L ∈ T →
    ∃ pl σ', pl.piece = Piece.L ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.skyline GameConfig.standard σ.ρ)
        = Board.skyline GameConfig.standard σ'.ρ ∧ σ'.ho = none ∧
      PlinthMem well Inv okB (T.draw Piece.L) σ' b
  stepJ : ∀ T σ b, Inv T σ → okB T σ b → Piece.J ∈ T →
    ∃ pl σ', pl.piece = Piece.J ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.skyline GameConfig.standard σ.ρ)
        = Board.skyline GameConfig.standard σ'.ρ ∧ σ'.ho = none ∧
      PlinthMem well Inv okB (T.draw Piece.J) σ' b
  stepI : ∀ T σ b, Inv T σ → okB T σ b → Piece.I ∈ T →
    (4 ≤ b ∧ PlinthMem well Inv okB (T.draw Piece.I) σ (b - 4))
    ∨ (∃ pl σ', pl.piece = Piece.I ∧ pl.Valid GameConfig.standard ∧
        AvoidsWell well pl ∧
        pl.place (Board.skyline GameConfig.standard σ.ρ)
          = Board.skyline GameConfig.standard σ'.ρ ∧ σ'.ho = none ∧
        PlinthMem well Inv okB (T.draw Piece.I) σ' b)
  stepSBoot : ∀ T σ b, Inv T σ → okB T σ b → Piece.S ∈ T → σ.cS = none →
    ∃ pl σ', pl.piece = Piece.S ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.skyline GameConfig.standard σ.ρ)
        = Board.skyline GameConfig.standard σ'.ρ ∧ σ'.ho = none ∧
      PlinthMem well Inv okB (T.draw Piece.S) σ' b
  stepZBoot : ∀ T σ b, Inv T σ → okB T σ b → Piece.Z ∈ T → σ.cZ = none →
    ∃ pl σ', pl.piece = Piece.Z ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.skyline GameConfig.standard σ.ρ)
        = Board.skyline GameConfig.standard σ'.ρ ∧ σ'.ho = none ∧
      PlinthMem well Inv okB (T.draw Piece.Z) σ' b
  stepOBoot : ∀ T σ b, Inv T σ → okB T σ b → Piece.O ∈ T → σ.cO = none →
    ∃ pl σ', pl.piece = Piece.O ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.skyline GameConfig.standard σ.ρ)
        = Board.skyline GameConfig.standard σ'.ρ ∧ σ'.ho = none ∧
      PlinthMem well Inv okB (T.draw Piece.O) σ' b
  /-- Boot closure: every pending piece has a full-move response staying in
  Boot or handing off into the plinth. -/
  bootStep : ∀ T h ho p, Boot T h ho → p ∈ T →
    ∃ pl h' ho', pl.piece = p ∧ pl.Valid GameConfig.standard ∧
      Placement.applyStep GameConfig.standard
        (Board.debtBoard GameConfig.standard h ho) pl
        = Board.debtBoard GameConfig.standard h' ho' ∧
      (Boot (T.draw p) h' ho' ∨
        ∃ σ c, Inv (T.draw p) σ ∧ okB (T.draw p) σ c ∧
          h' = Board.plinthLift well c σ.ρ ∧ ho' = some (hx, 0))

end Tetris
```

Note `σ'.ho = none` appears in each designer step (their successor carries
no extra hole) — consistent with `hoNone`.

- [ ] **Step 2: Wire the import** — in `proofs/Proofs.lean` after
`import Proofs.Safety.BandSchedule`:

```lean
import Proofs.Safety.PlinthCert
```

- [ ] **Step 3: Build** — `lake build`, expect success.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Safety/PlinthCert.lean proofs/Proofs.lean
git commit -m "proofs(plinth-cert): the two-regime certificate with re-anchoring — the corrected inhabitation target, typed"
```

---

### Task 5: the reduction — direct to DebtCertificate

**Files:**
- Modify: `proofs/Proofs/Safety/PlinthCert.lean` (append before `end Tetris`)

**Interfaces:**
- Consumes: everything above; `place_vertS_skyline`, `place_vertZ_skyline`, `place_O_pair`, `shapeUp_vertS/vertZ/O`; `tetrisSolvableValid_of_debtCertificate`; `clearLines_eq_self_of_no_fullRows`; `fullRows_plinth_eq_empty`; T1′/T2′; `plinthLift_congr_reanchor`.
- Produces: `PlinthCert.toDebtCertificate`, `Tetris.tetrisSolvableValid_of_plinthCert`.

- [ ] **Step 1: Write the shared plinth-step discharge helper**

```lean
namespace PlinthCert

/-- Any plinth response (bare flush rep + `PlinthMem` successor) discharges
one `DebtCertificate.step` case: transport by T1′, no clears (the floor is
immortal), then pack the re-anchored member. -/
theorem plinth_response (C : PlinthCert) {T : Bag} {σ σ' : BandState}
    {b : ℕ} {p : Piece} (pl : Placement)
    (hInv : C.Inv T σ)
    (hpiece : pl.piece = p) (hval : pl.Valid GameConfig.standard)
    (havd : AvoidsWell C.well pl)
    (hrep : pl.place (Board.skyline GameConfig.standard σ.ρ)
      = Board.skyline GameConfig.standard σ'.ρ)
    (hMem : PlinthMem C.well C.Inv C.okB (T.draw p) σ' b) :
    ∃ h' ho', pl.piece = p ∧ pl.Valid GameConfig.standard ∧
      Placement.applyStep GameConfig.standard
        (Board.debtBoard GameConfig.standard
          (Board.plinthLift C.well b σ.ρ) (some (C.hx, 0))) pl
        = Board.debtBoard GameConfig.standard h' ho' ∧
      (∃ σ'' c, C.Inv (T.draw p) σ'' ∧ C.okB (T.draw p) σ'' c ∧
        h' = Board.plinthLift C.well c σ''.ρ ∧ ho' = some (C.hx, 0)) := by
  obtain ⟨σ'', c'', hre, -, -, -, -, hInv'', hok''⟩ := hMem
  have hhx3 : C.hx ≠ C.well ∧ C.hx < GameConfig.standard.cols ∧ 1 ≤ σ.ρ C.hx :=
    ⟨C.hhx.1, C.hhx.2, C.holeLoaded T σ hInv⟩
  have hlift := Board.place_debtBoard_plinthLift (c := b)
    (fun cell hcell => hval cell hcell) havd hhx3 hrep
  refine ⟨Board.plinthLift C.well c'' σ''.ρ, some (C.hx, 0), hpiece, hval, ?_,
    σ'', c'', hInv'', hok'', rfl, rfl⟩
  rw [Placement.applyStep_eq_clearLines_place, hlift,
    Board.clearLines_eq_self_of_no_fullRows GameConfig.standard
      (Board.fullRows_plinth_eq_empty C.hwell C.hhx.1),
    Board.plinthLift_congr_reanchor hre]

end PlinthCert
```

Adjustment notes: `PlinthMem`'s re-anchor is stated on `σ'.ρ` (the rep
successor) at base `b` — `plinthLift_congr_reanchor hre :
plinthLift C.well b σ'.ρ = plinthLift C.well c'' σ''.ρ` rewrites the
transported profile; the `rw` order may need the `congr` equation applied
via a separate `have` + `rw`.

- [ ] **Step 2: Write `toDebtCertificate` + the headline**

```lean
namespace PlinthCert

/-- **The reduction**: a plinth certificate yields a debt certificate. -/
def toDebtCertificate (C : PlinthCert) : DebtCertificate where
  P := fun T h ho =>
    C.Boot T h ho ∨
    ∃ σ c, C.Inv T σ ∧ C.okB T σ c ∧
      h = Board.plinthLift C.well c σ.ρ ∧ ho = some (C.hx, 0)
  init := Or.inl C.bootInit
  cover := by
    rintro T h x (hB | ⟨σ, c, hInv, -, rfl, hx'⟩)
    · exact C.bootCover T h x hB
    · rw [Option.some.injEq] at hx'
      subst hx'
      refine ⟨C.hhx.2, ?_⟩
      rw [Board.plinthLift_ne C.well c σ.ρ C.hhx.1]
      have := C.holeLoaded T σ hInv
      omega
  height := by
    rintro T h ho (hB | ⟨σ, c, hInv, hok, rfl, -⟩) j hj
    · exact C.bootHeight T h ho hB j hj
    · exact C.height T σ c hInv hok j hj
  step := by
    rintro T h ho p hP hp
    rcases hP with hB | ⟨σ, b, hInv, hokB, rfl, rfl⟩
    · -- Boot regime: designer's step, successor packed in either arm.
      obtain ⟨pl, h', ho', h1, h2, h3, h4⟩ := C.bootStep T h ho p hB hp
      refine ⟨pl, h', ho', h1, h2, h3, ?_⟩
      rcases h4 with hB' | ⟨σ, c, hInv, hok, rfl, rfl⟩
      · exact Or.inl hB'
      · exact Or.inr ⟨σ, c, hInv, hok, rfl, rfl⟩
    · -- Plinth regime.
      cases p with
      | S =>
          cases hcS : σ.cS with
          | none =>
              obtain ⟨pl, σ', h1, h2, h3, h4, -, h6⟩ :=
                C.stepSBoot T σ b hInv hokB hp hcS
              obtain ⟨h', ho', g1, g2, g3, g4⟩ :=
                C.plinth_response pl hInv h1 h2 h3 h4 h6
              exact ⟨pl, h', ho', g1, g2, g3, Or.inr g4⟩
          | some cA =>
              have hwin := C.winS T σ cA hInv hcS
              obtain ⟨hcols, hcw, hc1w⟩ := C.winColsS T σ cA hInv hcS
              have hMem := C.invS T σ cA b hInv hokB hcS hp
              have hrep : Placement.place
                  (Board.skyline GameConfig.standard σ.ρ)
                  { piece := Piece.S, rot := 1, col := cA }
                  = Board.skyline GameConfig.standard (σ.bump2 cA).ρ :=
                Board.place_vertS_skyline (by omega) hcols hwin
              obtain ⟨h', ho', g1, g2, g3, g4⟩ :=
                C.plinth_response ({ piece := Piece.S, rot := 1, col := cA })
                  hInv rfl
                  (by
                    intro cell hcell
                    rw [Board.shapeUp_vertS cA] at hcell
                    fin_cases hcell <;> simp <;> omega)
                  (by
                    intro cell hcell
                    rw [Board.shapeUp_vertS cA] at hcell
                    fin_cases hcell <;> simp <;> omega)
                  hrep hMem
              exact ⟨_, h', ho', g1, g2, g3, Or.inr g4⟩
      | Z => -- mirror of S with winZ/winColsZ/invZ/stepZBoot/place_vertZ_skyline/shapeUp_vertZ
          sorry_replace_with_mirror
      | O => -- mirror with the winO pair-rewrite for place_O_pair (see BandSchedule's O case)
          sorry_replace_with_mirror
      | T =>
          obtain ⟨pl, σ', h1, h2, h3, h4, -, h6⟩ := C.stepT T σ b hInv hokB hp
          obtain ⟨h', ho', g1, g2, g3, g4⟩ :=
            C.plinth_response pl hInv h1 h2 h3 h4 h6
          exact ⟨pl, h', ho', g1, g2, g3, Or.inr g4⟩
      | L => -- same as T with stepL
          sorry_replace_with_mirror
      | J => -- same as T with stepJ
          sorry_replace_with_mirror
      | I =>
          rcases C.stepI T σ b hInv hokB hp with ⟨hb4, hMem⟩ |
            ⟨pl, σ', h1, h2, h3, h4, -, h6⟩
          · -- Drain: T2′ + re-anchored packing.
            obtain ⟨σ'', c'', hre, -, -, -, -, hInv'', hok''⟩ := hMem
            refine ⟨Board.bandDrain C.well,
              Board.plinthLift C.well c'' σ''.ρ, some (C.hx, 0), rfl,
              ?_, ?_, Or.inr ⟨σ'', c'', hInv'', hok'', rfl, rfl⟩⟩
            · rw [Board.bandDrain_eq_drainPl]
              exact drainPl_valid C.hwell
            · rw [show Board.plinthLift C.well c'' σ''.ρ
                  = Board.plinthLift C.well (b - 4) σ.ρ from
                  (Board.plinthLift_congr_reanchor hre).symm]
              exact Board.drain_debtBoard_plinthLift C.hwell hb4
                ⟨C.hhx.1, C.hhx.2, C.holeLoaded T σ hInv⟩
          · obtain ⟨h', ho', g1, g2, g3, g4⟩ :=
              C.plinth_response pl hInv h1 h2 h3 h4 h6
            exact ⟨pl, h', ho', g1, g2, g3, Or.inr g4⟩

end PlinthCert

/-- **Inhabiting the plinth certificate proves Tetris solvable.** The open
content: the T/L/J/I schedule with re-anchored bookkeeping, the unanchored
S/Z/O cases, and the bag-1 boot tree through the well plug. -/
theorem tetrisSolvableValid_of_plinthCert (C : PlinthCert) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_debtCertificate C.toDebtCertificate
```

Write the three `sorry_replace_with_mirror` cases out fully (Z and O mirror
S with their own fields — the O case needs the `place_O_pair` profile
rewrite exactly as in `BandSchedule.toShiftCertificate`'s O case: build
`hprof : Function.update (Function.update σ.ρ cA (σ.ρ cA + 2)) (cA + 1) (σ.ρ cA + 2)
= (σ.bump2 cA).ρ` via `unfold BandState.bump2; rw [hpair]`; L and J
duplicate the T case with `stepL`/`stepJ`). Watch two spots:
1. `DebtCertificate.step`'s tuple order is `⟨pl, h', ho', piece-eq, valid,
   applyStep-eq, P-membership⟩` — no AvoidsWell in the target (it is only a
   hypothesis for our transport).
2. The S-case's final `exact ⟨_, h', ho', …⟩` — the placeholder `_` is the
   placement literal; name it explicitly if elaboration stalls.

- [ ] **Step 3: Build** — `lake build`, expect success, no `sorry`.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Safety/PlinthCert.lean
git commit -m "proofs(plinth-cert): the reduction — anchored S/Z/O discharged over the plinth, TetrisSolvable from the corrected certificate"
```

---

### Task 6: gates, findings, docs

**Files:**
- Modify: `proofs/PROGRESS.md`, `proofs/LIBRARY.md`

- [ ] **Step 1: Gates** — `scripts/check-green-clean.sh` (expect OK); axiom
gate in the scratchpad on `tetrisSolvableValid_of_plinthCert`,
`Board.place_debtBoard_plinthLift`, `Board.drain_debtBoard_plinthLift`,
`Board.place_wellPlug_flat` (expect exactly
`[propext, Classical.choice, Quot.sound]`); `lake build ProofsExperiments`.

- [ ] **Step 2: PROGRESS.md** — insert under `## Last tick`:

```markdown
Tick (manual, 2026-07-12) — **FINDINGS D1/D2 + the plinth foundation**.
D1: `ShiftCertificate`/`BandScheduleCert` (2026-07-09/10) are SOUND but
UNINHABITABLE — every placement obligation propagates the base unchanged, the
drain lowers it, `initBase` starts at 0: the drain guard `4 ≤ b` is unreachable,
no clears fire, and `height_floor` forces the `height` obligation to fail.
Missing operation: re-anchoring `(ρ, b) ↦ (ρ − d, b + d)`. D2: even with
re-anchoring, the forced row-0 bootstrap hole (any first S/Z on flat) blocks all
re-anchors (`holeLift` row arithmetic) — physically, a row-0 hole means rows 0–3
are never simultaneously full. FIX (the plinth): one bag-1 J-rot-3 placement
plugs the well's row 0 (`place_wellPlug_flat`); row 0 becomes a permanent floor
whose hole keeps it from ever clearing (`fullRows_plinth_eq_empty` — the floor
is immortal); drains fill and clear rows 1–4 (`drain_debtBoard_plinthLift`,
a MID-ROW clear proven from the raw `clearLines`); the band rides `c + 1` above
in coordinates the hole never touches, so re-anchoring (`ReanchorsTo`) is free.
New: `Invariants/PlinthShift.lean` (lift, T1′ `place_debtBoard_plinthLift` from
BARE skyline reps — all mechanisms reuse, T2′, the plug) +
`Safety/PlinthCert.lean` (`PlinthMem` re-anchored membership, two-regime
`PlinthCert`, direct `toDebtCertificate`, `tetrisSolvableValid_of_plinthCert`).
The v1 certificates stay green as sound theorems; PlinthCert is the
inhabitation target. Schedule roadmap (10-bag/7-drain/3-band-I forced; zone
layout well+S+Z+T+OLJ = 10; the redirect flush-compatibility puzzle) in
`docs/superpowers/specs/2026-07-12-plinth-cert-design.md`.
```

- [ ] **Step 3: LIBRARY.md** — after the `BandScheduleCert` row add:

```markdown
| `PlinthCert` + **`tetrisSolvableValid_of_plinthCert`** (`Safety/PlinthCert`); plinth transport T1′/T2′ + well plug (`Invariants/PlinthShift`) | the CORRECTED inhabitation target (findings D1/D2: v1 certs sound but uninhabitable — base never rises; row-0 hole blocks re-anchoring): immortal floor, mid-row drain, `ReanchorsTo` membership; open remainder = T/L/J/I schedule + boot tree + rate bookkeeping |
```

and extend §4's tree (`Invariants/` + `PlinthShift`, `Safety/` + `PlinthCert`).

- [ ] **Step 4: Final build** — `lake build`, expect success.

- [ ] **Step 5: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/PROGRESS.md proofs/LIBRARY.md
git commit -m "proofs(plinth-cert): findings D1/D2 recorded + progress and library map entries"
```

---

## Self-review notes (already applied)

- Spec coverage: Part A → Tasks 1–3; Part B → Tasks 4–5; findings/docs →
  Task 6; Part C (schedule roadmap) lives in the spec by design.
- Type consistency: `PlinthMem well Inv okB T σ c`; `ReanchorsTo well ρ c ρ' c'`;
  `plinthLift w c ρ`; designer steps produce BARE skyline equalities +
  `σ'.ho = none` + `PlinthMem`; `plinth_response` consumes exactly that
  shape; the drain packs via `plinthLift_congr_reanchor`.
- Known uncertainties flagged in-task: `shapeUp_wellPlugJ` literal (Task 1),
  `ring_nf`/associativity in `dropOffset_plinthLift` (Task 2), the T1′
  slab/hole-point case juggling (Task 2), T2′'s filter/image bookkeeping
  (Task 3 recipe with the `hfilter` fallback), tuple orders in Task 5.
```
