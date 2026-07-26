# The plinth foundation + the 10-bag schedule roadmap — design

**Date**: 2026-07-12
**Status**: approved (brainstorming session)
**Target**: green `Proofs` lake target, base-axiom-clean, no `sorry`, no `native_decide`.
**Builds on**: `ShiftCertificate` (2026-07-09), `BandScheduleCert` (2026-07-10).

## Findings that drive this design

**D1 — the committed certificates cannot be inhabited: the base never rises.**
In `ShiftCertificate.step` / `BandScheduleCert.step*`, every placement
obligation propagates `okBase`/`okB` at the SAME base; the drain lowers it by
4; `initBase` starts at 0. Along any certified chain the base is stuck at 0,
the drain guard `4 ≤ b` is unreachable, no clears ever fire, and by
`height_floor` (mass floor 2.8 rows/bag) the `height` obligation must
eventually fail. The reductions are sound; the structures are vacuous for
inhabitation. Missing operation: **re-anchoring** `(ρ, b) ↦ (ρ − d, b + d)`
when the pattern floats `d` above base — a board-level no-op the obligations
must permit.

**D2 — the bootstrap hole at absolute row 0 blocks re-anchoring forever.**
Any first roughness piece on flat buries its gap at row 0
(`place_horizS_flat_eq_holedSkyline` at base 0 etc.). Re-anchoring by `d`
requires the `holeLift`-tracked rep hole row ≥ `d`; row 0 admits no lift.
Physically: with a hole at row 0, rows 0–3 are never simultaneously full, so
4-row drains can never fire. **Fix: the plinth.** During bag 1, place one
piece straddling the well boundary — `Piece.J` rot 3 has `shapeUp`
`{(0,0),(1,0),(1,1),(1,2)}`: exactly one cell at the well's row 0, three
cells flush on the neighbor, no new hole (verified against
`Piece.shape`). Row 0 then becomes a permanent floor (9 cells + the entombed
hole), the well operates at height 1, drains fill and clear rows 1–4, and
**row 0 is immortal because the hole keeps it from ever being full**. The
active band lives above the plinth in coordinates the hole never touches, so
re-anchoring is unobstructed.

**D3 — the rate law forces a 10-bag schedule.** Periodicity of a
well-drained band requires `24P + 4k = 36(P − k)` (P bags, k band-Is), i.e.
`k = 3P/10`: minimal period **10 bags, 7 drains, 3 band-Is, +28 rows per
column per period** — independently matching `BagGrowth`'s theorems
(`perfect_rectangle_bag_period_even`, the 10-bag flat-to-flat minimum).

**D4 — an exact-fit zone layout.** well(1) + S-window(2) + Z-window(2) +
T-alternator(2) + OLJ-zone(3) = **10 columns**. Mechanisms (verified from
the shape table):
- T rot 1 seats flush on an S-step and leaves a Z-step; T rot 3 the mirror —
  the T-zone alternates orientation each bag.
- {O, L, J} tile a 3×4 box (T-parity-legal: zero T pieces), so the OLJ-zone
  cycles flat → partial → flat+4.
- Rates per 10 bags: every column needs +28. S/Z/T zones: 40 cells from
  their own piece, deficit 16 each; OLJ: 120 from O+L+J, surplus 36;
  band-Is contribute 12. Deficits 48 = 36 + 12 exactly: each of S/Z/T
  receives 3 redirected O/L/J placements + 1 band vertical I per period.
- **The open redirect puzzle** (next session's design target): O seats only
  on flat pairs; L/J bottom profiles are `(0,0)`, `(2,0)`, `(0,2)` — never
  ±1 — so redirects onto the step-shaped zones need mediating surface
  shapes (or the zones must pass through flat/±2 configurations on
  schedule). This puzzle, with the exact numbers above, is the remaining
  crux content.

## Decisions

1. **Scope**: corrected foundation (PlinthShift + PlinthCert) + the schedule
   roadmap in this spec. No inhabitant attempt this session; no toy family.
2. `ShiftCertificate`/`BandScheduleCert`/BandShift/BandMechanisms stay green
   (sound theorems; the wrapper, flat witnesses, and bootstrap pack are
   consumed below). PROGRESS.md records D1/D2 and marks them superseded for
   inhabitation by `PlinthCert`.
3. `PlinthCert` reduces **directly to `DebtCertificate`** — plinth boards
   have well height 1 and are outside `ShiftCertificate`'s image (its
   `bandLift` pins the well at 0).

## Part A — `Proofs/Invariants/PlinthShift.lean` (new, green)

```lean
def plinthLift (w c : ℕ) (ρ : ℕ → ℕ) : ℕ → ℕ :=
  fun j => if j = w then 1 else ρ j + c + 1
```

The plinth board: `debtBoard cfg (plinthLift w c ρ) (some (hx, 0))`,
`hx ≠ w`, `hx < cols`, cover from `1 ≤ ρ hx` (family invariant: the hole
column stays loaded at rep level).

Contents:
- Basics: `plinthLift_well`, `plinthLift_ne`, cover helper.
- **T1′** `place_debtBoard_plinthLift`: hypotheses `hcols`, `havoid`
  (inline, as in T1), `hhx : hx ≠ w ∧ hx < cfg.cols ∧ 1 ≤ ρ hx`, and the
  BARE rep equality `pl.place (skyline cfg ρ) = skyline cfg ρ'`; conclusion
  `pl.place (debtBoard cfg (plinthLift w c ρ) (some (hx, 0)))
     = debtBoard cfg (plinthLift w c ρ') (some (hx, 0))`.
  Proof: T1's structure at shift `c + 1`; the well column is `{(w, 0)}` on
  both sides (height 1, hole ≠ well); the hole is fixed and both sides
  exclude it by the membership predicate. Reuses `sup_sub_add_shift`,
  `mem_debtBoard`, `mem_cellsAt`/`mem_cellsAt_add` from
  `Invariants/BandShift.lean` (import it). Note the rep obligations are
  bare-skyline — every existing mechanism (`place_vertS_skyline`,
  `place_O_pair`, the flat witnesses) applies unchanged.
- **T2′** `drain_debtBoard_plinthLift`: for `4 ≤ c` (and the `hhx` facts),
  `(bandDrain w).applyStep cfg (debtBoard cfg (plinthLift w c ρ) (some (hx, 0)))
     = debtBoard cfg (plinthLift w (c - 4) ρ) (some (hx, 0))`.
  The I lands at dropOffset 1 (well height 1), fills rows 1–4; full rows are
  exactly {1,2,3,4} (row 0 is not full — the hole; rows ≥ 5 miss the well).
  This is a MID-ROW clear: the bottom-contiguous clearing laws do not apply.
  Prove from the raw `Board.clearLines` definition (read it first), modeled
  on `Seam.drain_applyStep` (`SeamBridge.lean:202`) which computes a
  filter/image clear directly: characterize `isFull` rows as
  `r ∈ {1,2,3,4}`, then the image arithmetic: row 0 keeps (0 full rows
  below), rows ≥ 5 shift down 4. This is the session's main new proof
  (~150–250 lines).
- **`place_wellPlug_flat`**: `Piece.J` rot 3 at col `w` on
  `skyline (fun _ => base)` (needs `w + 1 < cols`): flush, profile becomes
  `w ↦ base + 1, w+1 ↦ base + 3` (others `base`). Mirrors the flat-witness
  pattern; `shapeUp` via `decide`.

## Part B — `Proofs/Safety/PlinthCert.lean` (new, green)

Reuses `BandState` (obligation `hoNone : Inv T σ → σ.ho = none` — the debt
slot is structurally spent on the entombed hole) and `BandState.bump2`.

```lean
def ReanchorsTo (well : ℕ) (ρ : ℕ → ℕ) (c : ℕ) (ρ' : ℕ → ℕ) (c' : ℕ) : Prop :=
  (∀ j, j ≠ well → ρ j + c = ρ' j + c') ∧ ρ' well = 0
```

and the membership-up-to-re-anchor
`Mem (C) (T) (σ : BandState) (c : ℕ) : Prop :=
  ∃ σ' c', ReanchorsTo well σ.ρ c σ'.ρ c' ∧ σ'.ho = σ.ho ∧
    σ'.cS = σ.cS ∧ σ'.cZ = σ.cZ ∧ σ'.cO = σ.cO ∧ Inv T σ' ∧ okB T σ' c'`
(anchor positions survive re-anchoring — heights shift uniformly, columns
don't move; window shapes are height differences, invariant under the
shift).

`PlinthCert` fields:
- Constants: `well hx : ℕ`, `hwell : well < cols`, `hhx : hx ≠ well ∧ hx < cols`.
- Plinth family: `Inv : Bag → BandState → Prop`, `okB : Bag → BandState → ℕ → Prop`.
- Boot family: `Boot : Bag → (ℕ → ℕ) → Option Coord → Prop` (raw absolute
  boards; bag-1 scale) with `bootInit : Boot Bag.full (fun _ => 0) none`,
  `bootCover`/`bootHeight` (the DebtCertificate-shaped side conditions), and
  `bootStep`: every pending piece has a response landing in `Boot` **or** in
  the plinth image (the handoff: `∃ σ c, Inv/okB ∧ h' = plinthLift well c σ.ρ
  ∧ ho' = some (hx, 0)`).
- Plinth static obligations: `hoNone`, `winS/winZ/winO`, `winColsS/Z/O`
  (in-band, off-well — landing on the `hx` column is allowed), `anchored :
  σ.ρ well = 0`, `holeLoaded : 1 ≤ σ.ρ hx`, `height : okB → plinthLift well
  c σ.ρ j ≤ rows`.
- Bookkeeping with re-anchor: `invS/invZ/invO : … → Mem (T.draw _) (σ.bump2 c) b`.
- Schedule content: `stepT/stepL/stepJ` (flush bare-skyline rep equality +
  `Mem` successor), `stepI` (`(4 ≤ b ∧ Mem (T.draw I) σ (b − 4))` ∨ band
  response), `stepSBoot/stepZBoot/stepOBoot` (unanchored cases).

Reduction: `PlinthCert.toDebtCertificate` with
`P := Boot ∪ { (plinthLift well c σ.ρ, some (hx, 0)) | Inv T σ ∧ okB T σ c }`.
Step proof: boot states → designer's `bootStep` (packing either regime);
plinth states → anchored S/Z/O discharged by us (mechanisms + T1′ + `Mem`
unpacking: `ReanchorsTo` gives function-level equality of the lifted
profiles, so `plinthLift well c σ''.ρ = plinthLift well c' σ'''.ρ` by
`funext` + `omega` — packing the re-anchored member), T/L/J/I/unanchored →
designer, drain → T2′. Headline:

```lean
theorem tetrisSolvableValid_of_plinthCert (C : PlinthCert) : TetrisSolvableValid
```

## Part C — the schedule roadmap (not Lean this session)

Recorded in this spec for the inhabitant build-out, in order:
1. **The redirect puzzle** (D4): design mediating shapes or scheduled
   flat/±2 pass-throughs so 3 O/L/J placements + 1 band-I per period land
   on each of S/Z/T zones. Constraints: piece bottoms available are
   flat×many, `(1,0)/(0,1)` (S/Z/T verticals), `(2,0)/(0,2)` (L/J
   verticals), `(0,0,1)/(1,0,0)` (S/Z horizontals), `(0,1)`-with-3-cells
   (T verticals), flat-3 (T/L/J flats), flat-4 (I flat).
2. OLJ-zone closure: the 3-column {O,L,J} state machine, all within-bag
   orders (~15 transitions).
3. T-alternator closure (2 states) and S/Z window persistence (already
   proven mechanisms).
4. The 10-bag phase clock: `Inv` as (phase × zone states), `okB` as the
   phase-determined base interval; drains at the 7 scheduled phases.
5. The bag-1 boot tree: from empty through the plug + the S/Z hole into
   phase 0 (the pack's edges + `place_wellPlug_flat` + `place_horizI_flat`).

## Acceptance

- Green `lake build` + `lake build ProofsExperiments`; hygiene gate; axiom
  gate: `tetrisSolvableValid_of_plinthCert`, `place_debtBoard_plinthLift`,
  `drain_debtBoard_plinthLift`, `place_wellPlug_flat` all exactly
  `[propext, Classical.choice, Quot.sound]`.
- PROGRESS.md records D1/D2 explicitly (the v1 certificates remain sound;
  superseded for inhabitation); LIBRARY.md row for PlinthCert.
- No toy inhabitant.

## Process and hygiene

As before: foreground builds; imports wired in `Proofs.lean`; one commit per
logical unit staging only `proofs/`; this spec not committed.

## Risks

1. T2′ (mid-row clear) is the effort concentration; the raw `clearLines`
   definition must be read first and the row-shift arithmetic done by hand.
   Fallback: prove first for `ρ` with a spread bound if the general form
   balloons; the certificate then carries that bound.
2. The two-regime union in `toDebtCertificate` doubles the case analysis;
   keep `Boot` obligations DebtCertificate-shaped so boot cases are pure
   plumbing.
3. `Mem`-unpacking in our S/Z/O discharge adds a re-anchor rewrite step;
   the `ReanchorsTo` function-level equality is designed to make that a
   `funext`+`omega` lemma (`plinthLift_congr_reanchor`).
