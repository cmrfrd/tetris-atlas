# Max-Plus Certificate for Infinite Tetris via Convex CEGIS

Date: 2026-06-26
Status: approved (design), implementation not started

## Goal

Build a **sound proof route** for `TetrisSolvableValid` (infinite survival under
canonical 7-bag rules) by *convex optimization*: search for a max-plus
piecewise-linear (PWL) certificate of a controlled-invariant region that contains
the empty board. If the convex program is feasible the certificate **is** the
proof; if it is infeasible, convex duality returns a Farkas/measure certificate
that no certificate of that class exists — a sharper negative result than prior
enumeration floors.

This is a deliberately different attack from the prior portfolio (compression,
feature-bands, refined-order WQO, discrete potential hunt, disk atlas), all of
which floored at the I-drain crux #66/#72 (`project_portfolio_exhaustion`). It
reuses, rather than discards, the existing sorry-free structure.

## Why this is not "just another reframing"

The honest baseline: every prior route relocates the I-drain crux, never
dissolves it. Two things are genuinely new here:

1. **Richer, searched function class.** The discrete potential hunt collapsed to
   `Φ = Σheights` because it only searched simple monotone potentials. The
   certificate `V(h) = osc(h − v)` over a free eigen-surface `v ∈ ℝ^W` is a
   `W`-dimensional family of conditional, geometric certificates the hunt never
   touched. It is exactly the eigen-surface `v` that `TopicalTetris`'s
   `eigen_global_roughness` *promises exists* but that greedy roughness-homing
   could not *find*. We **search** for `v` by LP instead of homing to it.
2. **Failure is a theorem.** If CEGIS converges to infeasible, LP duality yields a
   minimal finite set of states + adversary responses admitting no max-plus `V`
   of radius `≤ R` — strictly sharper than "search exploded `>4e6`", and likely
   pinpointing the I-drain geometry as a small witness.

## High-level decisions (from the user)

1. **Aim:** a sound convex *certificate* — the optimum is the proof artifact (not
   a solver, lens, or relaxation as the primary goal).
2. **Certificate class:** tropical / max-plus PWL (a max of affine forms),
   anchored on `V(h) = osc(h − v)`. Chosen because the dynamics are max-plus
   linear, so `V∘step` stays PWL; it is convex; and it is the eigen-surface object
   already proven to exist.
3. **∃-player handling:** CEGIS alternation — convex LP certificate search ⇄
   greedy policy improvement ⇄ Rust counterexample oracle.
4. **End artifact:** a Rust prototype (CEGIS+LP) that finds `(v, R, H_max, π)`,
   then a sorry-free Lean theorem that checks controlled-invariance as a
   decidable obligation. Staged so Rust fails fast before any Lean investment.

## The certificate (math)

Reduced state = surface height vector `h ∈ ℤ^W` (`W = 10`). By `SurfaceFiber`,
drops are hole-blind (depend only on `h`). Debt `d` is carried as a scalar
side-channel — the `HoleDebt` Lyapunov counter, budget `K = 1` from `EnergyGame`
iter10 — not (in the first cut) a certificate coordinate.

The certificate is a **convex polytope**, described entirely by max-plus PWL
constraints with parameters `(v ∈ ℝ^W, R, H_max)`:

```
S  =  { h ∈ ℝ^W :  0 ≤ h_j ≤ H_max  (all j),   osc(h − v) ≤ R }
```

with `osc(x) = max_j x_j − min_j x_j` (the Hilbert projective diameter). Two
facts make this the right object:

- **Clear-invariance is free.** Clears subtract a constant vector `c·1`, and
  `osc(h − c·1 − v) = osc(h − v)`. `V(h) = osc(h − v)` literally cannot see a
  clear — this is `clearMap_shapeKey`, used as the load-bearing reason `S` is
  well-defined under the full place-then-clear step.
- **The loss coupling is explicit.** `osc` is translation-blind, so it cannot
  bound the absolute top; the `max_j h_j ≤ H_max` face (also convex PWL) sits in
  `S` for that. After a clear `maxHeight = osc` (`maxHt_clearMap`), so "keep
  `osc ≤ R` and clear enough to hold `max ≤ H_max`" is exactly the
  clearing-equilibrium / I-drain content — now a face of a convex set.

**Proof obligation (controlled-invariance):**

```
∀ h ∈ S ∩ ℤ^W,  ∀ piece p,  ∃ placement a :  step(h, p, a) ∈ S
```

**Why the certificate search is an LP.** The drop is max-plus affine,
`h ↦ A_{p,a} ⊗ h` (`dropMap_maxplus`, `A_jk = top_j + 1 − bot_k` on the
footprint). In CEGIS we hold a *finite sample* `Σ` of states and a *fixed policy*
`π`. For sampled `h`, piece `p`, choice `a = π(h,p)`, the post-state
`h' = A ⊗ h` is a **constant** computed by the Rust engine. The obligation
`h' ∈ S` is `osc(h' − v) ≤ R ∧ max(h') ≤ H_max`, **convex in the parameters**
`(v, R, H_max)`, and linearizes to an honest LP: introduce `M, m` with
`M ≥ h'_j − v_j ≥ m` for all `j` and require `M − m ≤ R`. The non-convex `∃a`
and `∀h` live entirely outside the LP — in the policy and the verifier.

## The CEGIS loop

```
seed policy π (edge-well + adaptive I-drain), sample Σ ← {empty}
repeat:
  (v,R,H_max) ← solve LP:  minimize R  s.t.  step(h,p,π(h,p)) ∈ S  ∀h∈Σ, ∀p
  π           ← greedy argmin_a V(step(h,p,a))        # policy improvement
  cex         ← Rust verifier: closure of S under (∀p ∃a∈π); first out-of-S state
  if no cex:                      return CERTIFIED (v,R,H_max,π)   # survival proof
  if LP infeasible after improve: return REFUTED + Farkas dual
  Σ ← Σ ∪ {cex}
```

The verifier *is* the existing `tetris_policy` / `tetris_preview` closure engine —
it already does in-band leak detection over the 7-bag AND-OR frontier.

## Components

| Unit | Responsibility | Depends on |
|---|---|---|
| `tetris_maxplus_cert` (new research bin) | Drive CEGIS; own sample `Σ`, policy `π` | `tetris-game`, `tetris-search` |
| LP layer | Solve / re-solve the certificate LP | `good_lp` + `clarabel` backend (pure-Rust; also does SDP, keeping the SOS escalation open) |
| Drop-matrix module | Build `A_{p,a}` (tropical matrix per piece×placement) | `tetris-game` placement tables |
| Verifier / oracle | Closure of `S`, return counterexample | reuse `tetris_policy` closure |
| `MaxPlusCert.lean` (new) | Import `(v,R,H_max,π)`; check invariance as a *decidable* obligation — no clears in the kernel `decide` (clears via structural `clearLines_*` lemmas), then wire `tetrisSolvableValid_of_maxHeight_invariant` | `TopicalTetris`, `EnergyGame`, `HoleDebt` |

Binary location: `crates/tetris-playground/src/bin/research/tetris_maxplus_cert/`,
matching the existing research-bin convention (`tetris_policy`, `tetris_preview`,
`tetris_tropical`, `tetris_eigen`, `tetris_carrier_probe`, `tetris_energy`).

Lean: avoid clears in any multi-quantifier kernel `decide`
(`feedback_kernel_decide_no_clears`). The check is a no-clear `place` surface-map
closure over the finite integer points of `S` (decidable / small), with clears
handled structurally; then `tetrisSolvableValid_of_maxHeight_invariant`
(`EnergyGame` iter4) discharges init + safety.

## Milestones & eval specs

Per `CLAUDE.md`, every run reports: `N` / seed-set definition, the certified
`(v, R, H_max)`, `|Σ|` at convergence, CEGIS iteration count, wall-time and
pieces/sec, hardware + `rustc -Vv`, and — on failure — the Farkas/leak
certificate.

- **M0** — `W = 10`, fixed seed-set of bag orders, single-controller (no
  ∃-adversary). Expect feasible, `R ≤ 7` (fixed orders survive at `osc ≤ 7`).
  **Gate**: if M0 fails, the bug is the pipeline, not Tetris — stop and fix.
- **M1** — `W ∈ {4,5,6}`, switching adversary, `K = 1`. Feasible ⟹ first real
  survival certificate; infeasible ⟹ minimal refutation at small width
  (informative either way).
- **M2** — `W = 10`, switching adversary. The real target.
- **M3** — Lean import: `#print axioms` clean (modulo `native_decide`),
  sorry-free.

## Risk analysis & rollback

- The crux may be irreducible: all prior routes floor at the I-drain. This route
  does not claim to dissolve it; it claims a richer searched class and a
  dual-certificate payoff on failure (see "Why this is not just another
  reframing").
- **Decision gates:** M0 fails ⇒ fix pipeline. M1 floors ⇒ choose between (a)
  enriching the class to full SOS (the `clarabel` SDP path is already wired), or
  (b) banking the dual certificate as the sharpened negative and stopping. No
  open-ended grind.
- **First-cut scope limits (deliberate, may be revisited):** `V` anchored on
  `osc(h − v)` rather than a general max-of-affine `V`; debt carried as a side
  scalar rather than a certificate coordinate; `good_lp` + `clarabel` stack. The
  hole/debt coupling is where the residual I-drain crux still lives, attacked
  here with a new convex tool.

## Out of scope (this spec)

- Full SOS / moment-hierarchy escalation (kept open via `clarabel`, not built).
- The occupation-measure / game-LP route (rejected as primary; heaviest).
- Any change to the canonical engine in `tetris-game`.

## Related prior work (memory)

`project_topical_tetris` (max-plus / eigen-surface, the spectral characterization),
`project_energy_game` (`tetrisSolvableValid_of_maxHeight_invariant` finish-line),
`project_hole_debt` (debt Lyapunov counter), `project_surface_fiber` (drops
hole-blind), `project_portfolio_exhaustion` (the I-drain crux localization),
`feedback_kernel_decide_no_clears`, `feedback_commit_per_iter_proofs_only`.
