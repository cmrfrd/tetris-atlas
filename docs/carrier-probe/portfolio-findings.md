# Portfolio Findings — counterexample-guided invariant refinement (the overnight `/goal`)

Goal: prove `TetrisSolvableValid` by counterexample-guided invariant refinement in a
make-or-break-gated portfolio. Metric = **closure fraction** of a candidate invariant `I`;
each iteration must raise it, resolve a make-or-break, or remove a named Lean sorry.
Terminals: WIN / EXHAUSTION (this file, with numbers) / STALL.

Probe binary: `crates/tetris-playground/src/bin/research/tetris_carrier_probe`.
Background: the deep-player carrier explodes past 5×10⁵ exact canonical surfaces
(`deepcarrier`), so enumeration is out; every prior route (SurfaceInvariant, AbstractSafe,
WQO, Lyapunov, box bands) converges to the same crux — *prove a structured survival
surface is closed under all 7 pieces in all orders on 10 columns*. The primary approach
attacks this with two representations of `I`: (1) compressed representative surfaces, (2)
feature-region clauses (declarative bands). This file records each iteration's number.

---

## PRIMARY · representation 1 — compressed representative surfaces — **FLOORED**

**Iteration 1 (`compress` mode).** Ran the deep-player AND-OR bag-boundary closure keyed by
the *exact* canonical board (faithful expansion), tracking |I| under 8 compression schemes
simultaneously, to budget 6×10⁵ exact surfaces (max_height 11 = full carrier diversity).

Make-or-break: **does any scheme stay ≤10⁵ distinct signatures (certifiable) while remaining
congruent (sound)?**

| scheme | distinct @600k | ratio | growth | congruence (hole-spread≥2 / multi-group) |
|---|---|---|---|---|
| exact-limbs | 600,000 | 1.0× | linear | — (baseline) |
| heights (sound) | 422,802 | 1.4× | linear | — |
| relheights | 365,998 | 1.6× | linear | — |
| relh-clamp4 | 292,029 | 2.1× | linear | **5.4%** (sound-ish) but **166k > 10⁵** |
| adjdiff-clamp4 | 361,613 | 1.7× | linear | — |
| **heights/2** | **47,731** | **12.6×** | sublinear | **31.0%** non-congruent, max-group 1185 |
| relh4+holes/2 | 308,399 | 1.9× | linear | — |

**Result: FLOORED.** No scheme is simultaneously ≤10⁵ AND congruent. The only sub-10⁵
scheme (`heights/2`, 47,731) discards all hole information + half the vertical resolution:
31.0% of its multi-board groups mix boards differing by ≥2 holes (up to 1185 exact boards
per signature) → its preimage admits undrainable junk → unsound. The information-preserving
schemes (heights and finer) all sit at 3–4×10⁵ and grow linearly → uncertifiable. This is
the WQO fine⟺coarse dichotomy reproduced for enumeration: *small enough ⇒ drops the holes
that decide drainability; sound enough ⇒ too big*. Compression does not escape the size wall.

---

## PRIMARY · representation 2 — feature-region clauses (declarative bands)

**Iteration 2 (`closurefrac` mode).** New measurement the prior band work never took: the
**raw closure fraction** of a band (fraction of in-band surfaces with a safe in-band move
for *every* piece) + counterexample characterization — distinguishing "almost closed, few
concentrated counterexamples" from "diffusely broken." Band (h12, holes2, r6), 20,634
sampled in-band surfaces:

- **closure fraction = 65.8%** (13,582/20,634 fully closed) → **diffusely broken**.
- failures by piece: **I=3320** (dominant), Z=2506, S=1908, O=1683, T/L/J small.
- failing surfaces: avg_height **3.32** (low), avg_holes 1.13, avg_roughness **5.47** (at
  the r6 cap). So the **roughness cap binds**, on low-but-rough surfaces; the line piece `I`
  is the hardest to place keeping roughness ≤ 6.

The counterexamples are NOT a few concentrated cases — they are a third of the band, spread
across pieces (I-heavy) and bound by the roughness cap. This points to the
closure-fraction-vs-band-tightness tradeoff, measured next.

**Iteration 3 (`closuresweep` mode).** Sampled 200,000 surfaces once (under h14/holes4/r12),
evaluated raw closure fraction over a 4×5 ladder (holes ∈ {1..4} × roughness ∈ {4,6,8,10,12}):

| holes \ r | r4 | r6 | r8 | r10 | r12 |
|---|---|---|---|---|---|
| holes1 | 6.6% | 26.3% | 49.1% | 72.9% | 87.4% |
| holes2 | 20.1% | 54.9% | 74.7% | 88.4% | 95.5% |
| holes3 | 39.1% | 78.3% | 89.6% | 95.4% | 98.3% |
| holes4 | 52.8% | 90.2% | 96.4% | 98.4% | **99.45%** |

(in-band counts: r4 ≈ 7k–13k surfaces; r12 ≈ 161k–200k.)

**Result: FLOORED — smooth tradeoff, no knee.** Closure fraction is a smooth increasing
function of band looseness. To reach ≥99% needs the loosest bands (holes4, r10–r12), which
hold 173k–200k *sampled* surfaces — and the *true* band (including unreached junk) is >10⁷
for r ≥ 6 (prior `optimal`/Lyapunov runs). Even at 99.45% it is not 100% over clean samples:
the player needs transient h13–14 / holes-5 slack, so no fixed tight band is invariant. The
small certifiable bands (r4, 7k–13k surfaces) sit at 6–53% closure. **No band is both
≥99%-closed and ≤10⁵-certifiable.** Representation 2 reproduces the same wall.

### PRIMARY approach verdict: FLOORED (both representations)

Compressed surfaces (rep 1) and feature-region bands (rep 2) both hit the identical wall:
the representation must keep hole/overhang detail to be sound (exclude undrainable junk), but
that detail makes it both huge (>10⁵–10⁷) and non-monotone. Closure fraction never reaches a
certifiable-and-closed point. Counterexample-guided refinement does not help because the
counterexamples are *diffuse*, not concentrated — they are the band's intrinsic need for
transient slack, not a removable junk set. Moving to the portfolio.

---

## P1 — refined-order WQO — **FLOORED**

Make-or-break: is any *fine* domination order (one that excludes undrainable junk) exactly
monotone under no-clear placement, so it gives a clean WQO keystone? Tested via `hmono` (the
truncation-built dominated pairs satisfy height, per-col-hole, and subset orders at once).

Two runs — the second one matters:
- **junk β** (random column-bits, 19M checks): height 0%, height+holes 2.28%, total-holes
  0.116%, **overhang-depth 0.009%**, subset 81.4%. overhang-depth looked like an almost-clean
  keystone.
- **carrier β** (β drawn from 642 real deep-play surfaces, roughness≤6/holes≤3, 313M checks):
  height **0%**, height+holes **8.61%**, total-holes **3.33%**, **overhang-depth 3.20%**,
  subset 30.6%.

**Result: FLOORED.** The overhang-depth near-miss was a junk artifact — random boards are so
holey that placements rarely reorder relative buriedness; on the *actual carrier regime* where
the keystone must hold, every fine order is broken 3–9%. Only coarse height-domination is
monotone (0%), and it is too coarse (admits non-drainable junk → no closed carrier, per the
WQO `wqo-findings.md`). No fine order threads the needle. The proven WqoCarrier lemmas
(`place_domLE_mono`, `clearLines_domLE`, `tetrisSolvableValid_of_wqo`) remain valid but cannot
be re-pointed at a fine order — confirming the monotone⟺coarse dichotomy a third time.

---

## P2 — amortized potential (Lyapunov) — **FLOORED (to the crux)**

Make-or-break: a simple potential Φ with (a) Φ ≥ height, (b) provable per-bag drift ≤ 0 vs
worst-order. Empirical leg supported (the deep player keeps Φ bounded — height ≤ 11, Φ ≤ 27
vs the S/Z-first burst, plateaued); verifiable leg is the wall.

`hunt` (8-candidate family, local-window, player minimizes ΔP): **no candidate is both
monotone and clear-invariant.** Every potential S/Z push up, some cleaner pulls back down
(rough: −8…−14; rough+wells: −10…−22). The *only* forced-monotone quantity is aggregate
height (+4/piece), and it is clear-variant (a line clear removes 10). So the Lyapunov
potential collapses to Φ = Σheights, with per-bag drift = +28 − 10·(lines cleared) ≤ 0 **iff
the player forces ≥ 2.8 lines/bag vs the worst order** — exactly the per-piece placement
geometry crux. And the empirical validation route is closed too: prior `optimal`-mode
Lyapunov bands (Φ = height + a·rough + b·holes ≤ cap, all (a,b), caps 10–16) **every config
exploded past 10⁷ states** — a bounded-Φ board set in the survival regime is uncertifiably
large. Both legs floor: empirical enumeration explodes >10⁷; symbolic drift = crux #66/#72.

## P3 — disk-backed atlas with compression — **FLOORED (certification size)**

Make-or-break: does a *closed* carrier fit the night's compute AND the native_decide ceiling
(~10⁵)? Settled by the `deepcarrier` + `compress` data above, now with a definitive overnight
figure: a budget-4×10⁶ `deepcarrier depth2` run **exceeded 4,000,000 canonical boundary
surfaces in 2018 s, with max_height climbing 11→15 — still growing in BOTH size and height**,
no convergence. So the lower bound is >4×10⁶ (not merely >5×10⁵), and the carrier is provably
non-convergent at certifiable scale. A disk-backed store
could hold more, but (a) closure (death-prop) needs the *full* carrier, est. 10⁶–10⁷, and
(b) Lean certification (the M2 bridge `tetrisSolvable_of_init_mem_atlas` / native_decide)
cannot ingest >10⁵ states. Compression cannot rescue it: no signature scheme is both ≤10⁵ and
congruent (heights/2 = 47,731 but 31% hole-non-congruent; relh-clamp4 sound-ish but 166,411
and growing). So even a successful overnight enumeration produces an object too large to
certify. Disk scale relocates the wall to the certifier, doesn't remove it.

---

## OVERALL: portfolio EXHAUSTED — every route floors to the same crux

| route | representation | make-or-break number | verdict |
|---|---|---|---|
| PRIMARY rep 1 | compressed surfaces | sound schemes ≥2.9×10⁵; only heights/2 ≤10⁵ but 31% non-congruent | FLOORED |
| PRIMARY rep 2 | feature-region bands | smooth closure-frac tradeoff, no knee; ≥99% only at ≥173k-surface bands | FLOORED |
| P1 | refined-order WQO | fine orders 3.2–8.6% non-monotone on carrier-β; only coarse height 0% | FLOORED |
| P2 | amortized potential | no monotone clear-invariant Φ; Φ=Σheights drift needs ≥2.8 clears/bag (crux); bands >10⁷ | FLOORED |
| P3 | disk-backed atlas | carrier >5×10⁵→10⁶–10⁷; no congruent ≤10⁵ compression; uncertifiable | FLOORED |

Every route converges to the single irreducible crux #66/#72: **prove a structured survival
surface family is closed under all 7 pieces in all orders on 10 columns.** Reframing relocates
the wall (to monotonicity, to closure-fraction, to congruence, to certification size), never
dissolves it. The survival carrier is a genuinely complex object — neither enumerable (>5×10⁵)
nor cleanly characterizable (no monotone fine order, no bounded simple potential, no congruent
compression). This is a **documented negative result (EXHAUSTION)** for the counterexample-
guided-refinement portfolio: the empirical/order/compression shortcuts are closed; a proof of
`TetrisSolvableValid` must be **symbolic** and attack the per-piece geometry crux directly
(the SurfaceInvariant / OnlineReservoir Lean route), with no enumeration or order shortcut.

---

## Where the WIN lives — the crux, localized (the session's forward-pointing result)

The exhaustion is not just negative: it pins the open content to one named Lean obligation.
`Proofs/Experiments/OnlineReservoir.lean` proves

  `onlineReservoir_solves_tetris (G : ReservoirGeometryCert) : TetrisSolvableValid`

— i.e. `TetrisSolvableValid` holds **conditionally on a `ReservoirGeometryCert`** (the file is
0-sorry because `G` is an explicit hypothesis, not a `sorry`). The WIN = construct `G`. Its two
fields are the entire remaining gap:

- `nonI_slot_geometry` — every non-`I` piece has a valid placement preserving the reservoir
  shape + support ledger;
- **`I_regulator_geometry`** — the `I` piece has a valid drain placement preserving the shape.

**The portfolio data points at `I_regulator_geometry` as the binding one.** Across every band
in the closure-fraction sweep, the dominant forced-out-of-band piece was **`I`** (e.g. 3320
failures at h12/holes2/r6, the top of every row). And the simple fixed-well reservoir
(`Strat::Well`) empirically explodes/drifts to height 20 (prior `full`-mode run, boundary set
5M+). Together: a *fixed* `I`-placement does not close the reservoir; the gap is an **adaptive,
drain-aware `I`-move** (matching the standing OnlineReservoir note "rich tail's fixed placements
top out, needs a drain-aware I-move").

Why this object and not the others: the reservoir carrier is **strategy-local** — only the
boards the reservoir solver generates, a far smaller set than the >5×10⁵ all-surfaces carrier
that the enumeration/compression routes choke on. That is exactly why the symbolic reservoir
route can win where enumeration cannot: it never has to represent the whole carrier, only prove
the shape is preserved piece-by-piece. The irreducible work is the `I`-drain geometry.

**Recommended next effort (a separate focused goal, not this portfolio):** construct
`ReservoirGeometryCert`, starting with `I_regulator_geometry` and the adaptive drain — the one
place a real Lean advance toward `TetrisSolvableValid` remains. Empirical shortcuts are closed.
