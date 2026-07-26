# Steady-state pattern family + bag-1 bootstrap (BandScheduleCert) — design

**Date**: 2026-07-10
**Status**: approved (brainstorming session)
**Target**: green `Proofs` lake target, base-axiom-clean, no `sorry`, no `native_decide`.
**Builds on**: `ShiftCertificate` (`docs/superpowers/specs/2026-07-09-translation-quotient-design.md`).

## Problem and scope decision

The remaining path to `TetrisSolvable` is inhabiting `ShiftCertificate`
(closure of a relative pattern family under all seven pieces per bag). A full
inhabitant requires designing and verifying the complete multi-bag periodic
schedule — the open crux itself. This session's approved scope is the
**conditional reduction + bootstrap**: discharge everything that is provable
today, and isolate the remainder as one precisely-typed structure.

Facts established during exploration that pin the design:

1. **Debt is permanent under the well discipline.** The only debt-repaying
   clear (`clearLines_holedSkyline_exposed`) requires the hole's column at
   the global row-minimum `m > 0`; the open well pins the minimum at 0. The
   bootstrap hole (forced by S-or-Z-first from flat,
   `no_holefree_closed_invariant`) is therefore carried forever — legal,
   since `DebtCertificate`/`ShiftCertificate` budget debt ≤ 1. All
   post-bootstrap responses must be flush.
2. **S and Z are mutually enabling from flat.** The flat-S residue `(1,2,2)`
   contains a Z-window (`ρ (c+1) = ρ c + 1` at its left edge); the flat-Z
   residue `(2,2,1)` contains an S-window. The second roughness piece of
   bag 1 seats flush on the first's residue: exactly one hole is ever forced.
3. **Dedicated zones are rate-dead** (existing ZoneView/ZoneGame theorems;
   2-column windows absorb 2 cells/bag/col vs the band's required 8/3). Any
   inhabitant must migrate pieces across zones on a multi-bag schedule, and
   by `no_orderFree_bag_plan` some within-bag order-branching is
   irreducible. Hence the schedule (T/L/J/I placement policy + anchors
   migration + rate bookkeeping) is the honest open remainder — the
   certificate must leave it free, not bake in a dead zone design.
4. **The flush mechanisms are already proven at general profiles**:
   `place_vertS_step_reproduces`, `place_vertZ_step_reproduces`,
   `place_O_pair`, `vertI_drain`, locality
   (`colHeight_step_preserved_of_disjoint`, `well_preserved`).

## Decisions (from the brainstorming session)

1. **Scope**: conditional reduction + bootstrap pack (no full-schedule
   attempt, no toy inhabitant — a demo family that cannot close would be
   fake validation).
2. **Interface**: concrete `BandState` record + designer `Inv` predicate.
   S/Z/O successors are definitional; their closure is discharged by us.
   The inhabitant keeps T/L/J/I and the bootstrap wiring.
3. **Bootstrap encoding**: window anchors are `Option ℕ` — `none` during
   bootstrap, so the flat init state is a legal family member and the
   un-anchored S/Z/O cases are designer obligations (assisted by the pack).

## Part A — `Proofs/Invariants/BandMechanisms.lean` (new, green)

**The master debt-carry wrapper** (subsumes per-piece debt versions):

```lean
theorem place_debtBoard_of_flush {cfg : GameConfig} {ρ ρ' : ℕ → ℕ}
    {ho : Option Coord} {pl : Placement}
    (hflush : pl.place (skyline cfg ρ) = skyline cfg ρ')
    (hho : ∀ x, ho = some x → x.1 < cfg.cols ∧ x.2 + 1 < ρ x.1) :
    pl.place (debtBoard cfg ρ ho) = debtBoard cfg ρ' ho
```

Proof route: `ho = none` is `hflush` itself; `ho = some x` via the existing
`place_holedSkyline` (place-on-holed = erase-of-place-on-skyline, under
cover) and cover of `ρ'` from height-monotonicity of `place`
(`ρ' x.1 ≥ ρ x.1` read off `hflush` and `colHeight_skyline`).

**The bag-1 pack** (all unconditional):

- `place_horizI_flat` (new): the horizontal I seats flush on 4 flat columns,
  raising each by 1 — the bag-1 I response at base 0 (drain unavailable).
  Proved like the other flat witnesses (`shapeUp` computation + `ext`), or
  via `place_flush_skyline` with width 4.
- `zWindow_of_sBoot` / `sWindow_of_zBoot`: restatements of the flat-S/flat-Z
  residues exposing the created window (pure arithmetic on the residue
  profile — the mutual-enabling facts, kept as named theorems so the
  inhabitant's bag-1 tree can cite them).
- `place_vertZ_afterS` / `place_vertS_afterZ`: the forced second-piece
  edges — vertical Z seated flush on the S-residue profile at its Z-window
  (via the general `place_vertZ_skyline` + `place_debtBoard_of_flush` to
  carry the fresh hole), and the mirror. Stated as full `debtBoard`
  transition equalities from the post-bootstrap state.

## Part B — `Proofs/Safety/BandSchedule.lean` (new, green)

```lean
structure BandState where
  ρ  : ℕ → ℕ
  ho : Option Coord
  cS cZ cO : Option ℕ
```

Definitional successors (both window columns +2):

```lean
def BandState.bump2 (σ : BandState) (c : ℕ) : BandState :=
  { σ with ρ := Function.update (Function.update σ.ρ c (σ.ρ c + 2))
      (c + 1) (σ.ρ (c + 1) + 2) }
-- succS σ c := σ.bump2 c  (used when σ.cS = some c; likewise Z at cZ, O at cO)
```

`BandScheduleCert` fields (`cfg = GameConfig.standard` throughout; `cols`,
`rows` its dimensions):

Designer data:
- `well : ℕ`, `hwell : well < cols`
- `Inv : Bag → BandState → Prop`
- `okB : Bag → BandState → ℕ → Prop`
- `init : Inv Bag.full ⟨fun _ => 0, none, none, none, none⟩`
- `initBase : okB Bag.full ⟨fun _ => 0, none, none, none, none⟩ 0`

Static obligations on `Inv` (consumed by our discharge):
- `winS : Inv T σ → σ.cS = some c → σ.ρ c = σ.ρ (c + 1) + 1`
- `winZ : Inv T σ → σ.cZ = some c → σ.ρ (c + 1) = σ.ρ c + 1`
- `winO : Inv T σ → σ.cO = some c → σ.ρ c = σ.ρ (c + 1)`
- `winCols`: each set anchor `c` has `c + 1 < cols`, `c ≠ well`,
  `c + 1 ≠ well`
- `anchored : Inv T σ → σ.ρ well = 0`
- `cover : Inv T σ → σ.ho = some x →
     x.1 ≠ well ∧ x.1 < cols ∧ x.2 + 1 < σ.ρ x.1`
- `height : Inv T σ → okB T σ c → ∀ j < cols, bandLift well c σ.ρ j ≤ rows`

(Pairwise window disjointness is NOT a certificate field: the forced
successor only touches its own window's two columns, and preservation of the
*other* windows is part of the designer's `invS/invZ/invO` bookkeeping —
they will need disjointness inside their own `Inv` to prove it, but the
certificate does not impose a fixed disjointness shape.)

Bookkeeping obligations (pure `Function.update` arithmetic, no board
reasoning):
- `invS : Inv T σ → σ.cS = some c → Piece.S ∈ T →
     Inv (T.draw .S) (σ.bump2 c) ∧
     ∀ b, okB T σ b → okB (T.draw .S) (σ.bump2 c) b`
- `invZ`, `invO` mirrors at `cZ`, `cO`.

The open schedule content (exactly the crux):
- `stepT : Inv T σ → okB T σ b → Piece.T ∈ T →
     ∃ pl σ', pl.piece = .T ∧ pl.Valid cfg ∧ AvoidsWell well pl ∧
       pl.place (debtBoard cfg σ.ρ σ.ho) = debtBoard cfg σ'.ρ σ'.ho ∧
       Inv (T.draw .T) σ' ∧ okB (T.draw .T) σ' b`
- `stepL`, `stepJ`: same shape.
- `stepI : Inv T σ → okB T σ b → Piece.I ∈ T →
     (4 ≤ b ∧ Inv (T.draw .I) σ ∧ okB (T.draw .I) σ (b - 4))
     ∨ (∃ pl σ', … same shape with piece = .I)`
- `stepS_boot : Inv T σ → okB T σ b → Piece.S ∈ T → σ.cS = none →
     ∃ pl σ', … same shape with piece = .S` — and `stepZ_boot`,
  `stepO_boot` mirrors. (The un-anchored cases: the bag-1 tree wiring,
  assisted by the Part A pack.)

## Part C — the reduction (in `BandSchedule.lean`)

```lean
def BandScheduleCert.toShiftCertificate (C : BandScheduleCert) : ShiftCertificate
theorem tetrisSolvableValid_of_bandSchedule (C : BandScheduleCert) :
    TetrisSolvableValid
```

Instantiation: `Q T ρ ho := ∃ σ, C.Inv T σ ∧ σ.ρ = ρ ∧ σ.ho = ho`;
`okBase T ρ ho b := ∃ σ, C.Inv T σ ∧ σ.ρ = ρ ∧ σ.ho = ho ∧ C.okB T σ b`.
The `step` proof uses the `okBase` witness state (so the two existentials
stay coherent; the `Q` witness is only used for `cover`, whose conclusion
depends only on the shared `(ρ, ho)`).

Case discharge in `step`:
- **S with `cS = some c` (ours)**: response = vertical S at `c`
  (`⟨.S, 1, c⟩`). Board equality: `place_vertS_step_reproduces` (or directly
  `place_vertS_skyline`) at `winS` + `place_debtBoard_of_flush` with `cover`.
  `Valid`/`AvoidsWell` from `winCols`. Family membership: `invS`.
- **Z / O with set anchors (ours)**: mirrors via `place_vertZ_skyline` /
  `place_O_pair`.
- **T / L / J (theirs)**: `stepT/L/J` verbatim.
- **I (theirs)**: `stepI` — drain disjunct maps to `ShiftCertificate`'s
  drain case (`4 ≤ b`, same `(ρ, ho)`, base − 4); band disjunct to the
  placement case.
- **S/Z/O with `none` anchors (theirs)**: `stepS_boot` etc.
- `init`/`initBase`/`cover`/`height` map directly.

Then `tetrisSolvableValid_of_bandSchedule :=
tetrisSolvableValid_of_shiftCertificate ∘ toShiftCertificate`.

**What this reduction honestly buys**: the seven-piece adaptive closure
shrinks to a four-piece schedule (T/L/J/I) plus bag-1 wiring; all
board-level reasoning (skyline calculus, debt transport, drop geometry,
translation) is finished — the remaining object is a finite combinatorial
schedule whose obligations are profile arithmetic and family bookkeeping.

## Acceptance

- Green `lake build` + `lake build ProofsExperiments`, hygiene gate, axiom
  gate: `tetrisSolvableValid_of_bandSchedule`, `place_debtBoard_of_flush`,
  `place_horizI_flat`, `place_vertZ_afterS` all exactly
  `[propext, Classical.choice, Quot.sound]`.
- The bootstrap pack proven as concrete theorems (exercises the wrapper
  end-to-end, including the hole-creating → flush edge).
- No toy inhabitant. PROGRESS/LIBRARY entries state the reduction precisely.

## Process and hygiene

Same as the translation-quotient work: foreground `lake` builds only; both
new modules imported from `Proofs.lean`; one commit per logical unit staging
only `proofs/`; docstrings never contain `word-/word`; this spec is not
committed (branch convention).

## Risks

1. `place_holedSkyline`'s exact hypotheses may differ from assumed (read
   before Part A; if it demands the placement misses the hole explicitly,
   derive that from strict cover + `dropped` sitting at/above the surface).
2. The `place_vertS_skyline` / `place_vertZ_skyline` / `place_O_pair`
   statements produce `Function.update` profiles; `bump2` must match them
   definitionally or via a one-line `funext`.
3. Scope estimate ~500–700 lines across two files; if the bootstrap
   second-piece edges balloon, they may land as statements-with-full-proofs
   in a follow-up commit while the wrapper + certificate + reduction land
   first (each commit independently green).
