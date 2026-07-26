# I-in-pools + schedules: the drain-faithful zone verdicts — design

**Date**: 2026-07-13
**Status**: approved (brainstorming session)
**Target**: `ProofsExperiments` only (additive changes to
`Proofs/Experiments/FlushZoneGame.lean`); docs.

## Finding driving this design

**The periodic drain is provably invisible to the relative zone game.** The
drain subtracts 4 from every band column exactly (base ≥ 4, no truncation)
and `FlushZoneGame` states are min-normalized: `normZ (h.map (· − 4)) =
normZ h`. Consequences: (1) the ten committed DEAD verdicts (2026-07-12b)
are **drain-robust** — they hold under every drain schedule; (2) the
faithful "add the drain" upgrade is to model what the drain actually
changes: the **band-I** (3 of 10 I's per the rate law enter the band; I is
the only 1-column piece and the only 4-flat) and the **honest spread
budget** (drains pin the base, freeing ~12–14 rows of pattern spread above
the plinth, vs the conservative 6–8 caps used so far).

## Deliverables

**A. Engine (additive; committed verdicts untouched):**
1. `normZ_shift : ∀ (h : ZS) (k : ℕ), (∀ x ∈ h, k ≤ x) →
   normZ (h.map (· - k)) = normZ h` — the invisibility fact, proven by list
   induction (generalized `foldl Nat.min` accumulator lemma first). A
   docstring note records that batch-1/2 verdicts are therefore
   drain-robust.
2. The scheduled game: a parallel mutual family `survP`/`survAndP`/`survOrP`
   with a phase index (refill draws `sched.getD (phase % sched.length) []`,
   phase increments mod length, memo key gains the phase), and
   `flushDeadP (w spread : ℕ) (sched : List (List Piece)) (bags : ℕ) : Bool`.
   The existing `surv`/`flushDead` definitions stay byte-identical.

**B. Verdicts** (observe by `#eval` from the scratchpad, state with actual
values, `native_decide`; shrink caps on > 10 min):
- `olji4`, `olji5`: {O,L,J,I} at widths 4, 5 — caps 6 then 10.
- `szti4`, `szti6`: {S,Z,T,I} at widths 4, 6 — caps 6 then 10.
- `all7`: all seven pieces at width 6 — caps 8 then 12 (as feasible).
- `mixed10`: the rate-faithful slice — `sched` of 10 bags, all-six pieces
  every bag, I added in bags 0, 3, 6; width 6, cap 10, horizon ≥ 10 bags
  (as feasible; shrink horizon first, then cap).
Every decided instance commits with its actual verdict, alive or dead;
alive-at-horizon is labeled as evidence, not closure.

**C. Docs:** PROGRESS tick extending the verdict table + the
drain-robustness note; LIBRARY keep-active line updated.

## Acceptance

`lake build` (green unaffected) + `lake build ProofsExperiments` green;
hygiene gate clean; wall-times recorded; no instance above 6 columns.

## Process

As established: foreground builds; one commit per logical unit staging only
`proofs/`; spec not committed.
