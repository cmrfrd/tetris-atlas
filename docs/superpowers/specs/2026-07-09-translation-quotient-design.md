# Translation quotient for the Tetris safe-set witness — design

**Date**: 2026-07-09
**Status**: approved (brainstorming session)
**Target**: green `Proofs` lake target, base-axiom-clean, no `sorry`, no `native_decide`.

## Problem

`TetrisSolvable` is reduced (with proof) to inhabiting `DebtCertificate`
(`Proofs/Safety/SkylineInvariant.lean:126`): a bag-indexed family
`P : Bag → (ℕ → ℕ) → Option Coord → Prop` over absolute surface profiles with
≤ 1 buried hole, closed under one full move per pending piece. Every prior
enumeration attempt walls because the same *relative* surface pattern at each
absolute base height counts as a distinct state.

The placement dynamics are translation-equivariant; the obstruction to
quotienting is that (a) the ceiling `h ≤ rows` and the drain guard are
absolute, and (b) a *global* uniform shift `skyline (h + c)` creates `c` full
bottom rows — unreachable junk that `clearLines` would instantly remove.

**Key insight**: the correct symmetry is *well-anchored*. Keep one well column
at height 0 (which blocks all line clears — the `SeamBridge` discipline) and
shift only the band columns by `c`. Under that shift:

- band placements transport exactly (`dropOffset_skyline_sub` applies with
  `m := c`; its side condition holds automatically on band columns),
- the vertical-I drain into the well clears exactly 4 rows at every base
  `c ≥ 4`, leaving pattern and rep-hole unchanged and base `c − 4`,
- buried holes ride at `+c` and stay strictly covered.

So the quotient state is **band pattern × debt ≤ 1 × bag**, and base height
becomes a scalar handled by interval arithmetic instead of board enumeration.

Debt-1 support is mandatory, not optional: `no_holefree_closed_invariant`
proves any family containing `init` must answer S-from-flat with a buried
hole, so a hole-free-only certificate would be provably uninhabitable.

## Decisions (from the brainstorming session)

1. **Target interface**: reduce into `DebtCertificate` (the established
   squeeze point), not `SeamCert`. Reuses the proven sufficiency chain.
2. **Base handling**: a designer-supplied predicate
   `okBase : Bag → pattern → hole → ℕ → Prop` with closure obligations;
   no hard-coded drain schedule.
3. **Transport shape**: transition transport — transport *witnessed*
   `place`-equalities from the base-0 representative to every band-lift,
   plus one generic drain lemma. No new dynamics function, no raw-board
   shift operator.
4. **Acceptance test**: lift the seven flat witnesses
   (`exists_debtBoard_step_flat_{O,I,T,L,J,S,Z}`) to arbitrary base via the
   transport lemmas, plus one drain instance; green build + axiom gate.

## Part A — transport lemmas (`Proofs/Invariants/BandShift.lean`, new, green)

Definitions:

```lean
def Board.bandLift (w c : ℕ) (h : ℕ → ℕ) : ℕ → ℕ :=
  fun j => if j = w then 0 else h j + c

def holeLift (c : ℕ) : Option Coord → Option Coord :=
  Option.map (fun x => (x.1, x.2 + c))
```

plus a local column-avoidance predicate (`AvoidsCol w pl`: no cell of
`pl.shapeUp` lands in column `w`). `Safety/SeamBridge.lean`'s `AvoidsWell`
cannot be imported here (layering: `Invariants` sits below `Safety`); an
`iff` bridge lemma is added at the Safety layer instead.

**T1 — clear-free transport.** Hypotheses: `AvoidsCol w pl`, piece columns
in-bounds (via `pl.Valid` or explicit), `ρ w = 0`, `ρ' w = 0`, holes (when
`some`) in-band, not in the well column, strictly covered. Statement:

```
pl.place (debtBoard ρ ho) = debtBoard ρ' ho'
  → pl.place (debtBoard (bandLift w c ρ) (holeLift c ho))
      = debtBoard (bandLift w c ρ') (holeLift c ho')
```

Proof route: drop-offset shift via `dropOffset_skyline_sub`
(`Invariants/Confluence.lean:201`) extended through the hole by
`colHeight_holedSkyline` (hole invisible to drops); then `Finset.ext`
relating shifted landing cells, the band bottom slab (`r < c`), and the
hole. If `dropOffset` lacks a board-generic "depends only on piece-column
heights" congruence lemma, add it here. This is the main effort
concentration (~150–250 lines).

**T2 — generic drain.** The drain placement (vertical I in column `w`) is
defined locally (`Invariants` cannot import `Safety/SeamBridge.lean`'s
`drainPl`; an equality bridge is added at the Safety layer alongside the
`AvoidsCol`/`AvoidsWell` one). Hypotheses: `4 ≤ c`, `ρ w = 0`, hole (when
`some`) in-band strictly covered, `w < cols`. Statement:

```
(drainPl w).applyStep cfg (debtBoard (bandLift w c ρ) (holeLift c ho))
  = debtBoard (bandLift w (c-4) ρ) (holeLift (c-4) ho)
```

Proof route: the pre-clear board is the `holedSkyline` of the profile
`(w ↦ 4, band ↦ ρ + c)`; assemble from `place_vertI_flat` (already
base-generic), the clearing laws `clearLines_holedSkyline_of_le`/`_of_lt`,
and hole-ride arithmetic (`hole row ≥ c ≥ 4` sits above the cleared band).
Payoff: pattern and rep-hole unchanged; only the base drops by 4.

## Part B — the certificate (`Proofs/Safety/ShiftCertificate.lean`, new, green)

```lean
structure ShiftCertificate where
  well     : ℕ
  hwell    : well < GameConfig.standard.cols
  Q        : Bag → (ℕ → ℕ) → Option Coord → Prop      -- base-0 representatives
  okBase   : Bag → (ℕ → ℕ) → Option Coord → ℕ → Prop  -- designer base predicate
  init     : Q Bag.full (fun _ => 0) none
  initBase : okBase Bag.full (fun _ => 0) none 0
  anchored : ∀ T ρ ho, Q T ρ ho → ρ well = 0
  cover    : ∀ T ρ x, Q T ρ (some x) →
               x.1 < GameConfig.standard.cols ∧ x.1 ≠ well ∧ x.2 + 1 < ρ x.1
  height   : ∀ T ρ ho c, Q T ρ ho → okBase T ρ ho c →
               ∀ j < GameConfig.standard.cols,
                 Board.bandLift well c ρ j ≤ GameConfig.standard.rows
  step     : ∀ T ρ ho c p, Q T ρ ho → okBase T ρ ho c → p ∈ T →
    (∃ pl ρ' ho', pl.piece = p ∧ pl.Valid GameConfig.standard ∧
       AvoidsCol well pl ∧
       pl.place (Board.debtBoard GameConfig.standard ρ ho)
         = Board.debtBoard GameConfig.standard ρ' ho' ∧   -- at the rep, no c
       Q (T.draw p) ρ' ho' ∧ okBase (T.draw p) ρ' ho' c)
    ∨ (p = Piece.I ∧ 4 ≤ c ∧ Q (T.draw p) ρ ho ∧ okBase (T.draw p) ρ ho (c - 4))
```

**Reduction theorem**:

```lean
theorem tetrisSolvableValid_of_shiftCertificate
    (C : ShiftCertificate) : TetrisSolvableValid
```

proved by instantiating `DebtCertificate` with

```
P T h ho' := ∃ ρ ho c, C.Q T ρ ho ∧ C.okBase T ρ ho c ∧
             h = Board.bandLift C.well c ρ ∧ ho' = holeLift c ho
```

Field discharge: `init` at `(ρ = 0, c = 0)`; `cover`/`height` by lift
arithmetic from `C.cover`/`C.height`; `step` placement case via T1 plus
"well open ⇒ no full rows ⇒ `applyStep = place`"
(`fullRows_eq_empty_of_wellFree` + `clearLines_id_of_no_full`); drain case
via T2 plus `valid_vertI`.

**What this buys**: every board-level closure obligation is stated at the
base-0 representative (no `c` in the `place` equality) — proven once per
pattern transition. Base appears only in `okBase` propagation (scalar,
`omega`-dischargeable) and the drain guard `4 ≤ c`. Responses may still
depend on `c` when the designer wants (the `∃` sits inside the `∀ c`).

**Non-normalization note**: patterns are *not* required to have band-min 0.
Soundness needs no canonical decomposition — the designer picks the
`(ρ, c)` split. Drains at pattern-min ≥ 4 with low base are expressed by
re-anchoring (`ρ − 4` at `c + 4`).

## Part C — validation (final section of `BandShift.lean`, green)

Apply T1 to the existing rep-level equalities behind
`exists_debtBoard_step_flat_{O,I,T,L,J,S,Z}`
(`Invariants/HoledSkyline.lean:787–1109`, column-parameterized) with columns
constrained to the band, producing `…_lift` theorems on the well-anchored
flat at arbitrary base `c`. This exercises T1 end-to-end including the
hole-creating S/Z case (`holeLift (some …)`); add one T2 instance on the
band-flat. These lifted witnesses are directly reusable by the future
certificate inhabitant.

## Process and hygiene

- Foreground `lake build` only; never SIGTERM an in-flight build; rebuild
  after any theorem move.
- Imports: add the two new modules to `Proofs.lean` (green root).
- Gates: `scripts/check-green-clean.sh`; axiom gate
  (`#print axioms tetrisSolvableValid_of_shiftCertificate` must print exactly
  `[propext, Classical.choice, Quot.sound]`).
- Commits: one commit per logical unit, staging **only `proofs/`** (branch
  convention; this doc itself is therefore not committed).
- PROGRESS.md entry on completion.

## Risks

1. T1's `Finset.ext` bookkeeping (bottom slab + hole cases) is the effort
   concentration; budget the majority of proof time there.
2. A missing `dropOffset` board-congruence lemma may need to be added first.
3. Scope estimate: ~600–900 new lines across two files. If T1 balloons,
   fall back to proving T1 first for `ho = none`/`ho' = none` (skyline case)
   and layering hole cases as separate lemmas.
