# Hole-debt × surface-WQO route — make-or-break findings

Route: prove `TetrisSolvableValid` by separating the two concerns no single order could
unify — **surface** by height-domination (WQO; `place_domLE_mono` proven monotone in
`WqoCarrier.lean`) and **drainability** by hole-**debt** as a *scalar* Lyapunov counter
(`debt = ΣcolHeight − card = |holes|`, proven in `HoleDebt.lean`: rises only on placement,
falls only on clears; `SurfaceFiber.lean`: placement is hole-independent). The separation
sidesteps the non-monotonicity that kills `safeLE` (height + hole-*subset*): the surface
lift uses the proven monotone keystone, debt rides along as a scalar cap.

Lean foundations (`HoleDebt`, `SurfaceFiber`, `HoleyCarrier`) are all **sorry-free**. The
remaining work is the bounded-debt × surface-WQO `Carrier` reduction. The make-or-break
the route is gated on: **is there a debt bound `D` and a SMALL height-domination basis such
that the carrier `{b : surface(b) ≼ β ∧ debt(b) ≤ D}` closes?**

## Iteration-1 finding: the literal "next target" is FALSE

`HoleDebt`'s stated next target `debt(place b pl) ≤ debt b + 3` is **false as a
∀-placement statement.** An I-piece dropped flat over heights `[0,5,0,0]` rests at row 5 and
buries ~15 cells (ΣΔheight 19 − 4 cells = +15 debt). So debt growth is bounded only when the
*player chooses* the placement (∃-placement) — which is the closure/crux, not a clean
per-piece bound. (This is the same I-piece signal as the portfolio's closure-fraction data.)

## Make-or-break A (`debttraj`): deep player vs structural-worst — PASS

`debttraj depth4 beam8 bags3000` (deep player vs S/Z-first, the structural-worst order):
- **max debt = 1** (0 most bags; 179/3000 boundaries at debt 1), **max height = 4**,
- **surface antichain = 8**, **0 topouts** over 3000 bags.

So vs the structural attack a bounded-debt (D=1) × tiny-basis (8 surfaces) carrier is real.
But that's one order — the Lean obligation is closure vs *all* orders.

## Make-or-break B (`debtcarrier`): debt≤1 AND-OR closure vs ALL orders

Added a hole cap to the deep-player bag-boundary AND-OR closure (`deepcarrier` + `HOLE_CAP`),
plus WQO-antichain tracking. `debtcarrier holes1 depth2 budget1e6`:
- the debt≤1 carrier is survival-**feasible** vs all orders (the player keeps finding debt≤1
  moves — no `has_loss` collapse), BUT the full surface set **EXPLODES > 1,000,000**, max
  height drifting to **15**;
- the **WQO antichain = 6144 (peaked 8703)** — it rose then *fell back* (tall surfaces
  dominating lower ones), a **saturation** signal: the downward-closed carrier's basis is
  bounded ~6–8k even though the full set explodes (≈160× compression).

Reading: debt≤1 doesn't collapse (good) but doesn't bound height/surface-count vs the
adaptive adversary at depth-2 (height→15 signals depth-2 suboptimality inflating the basis
with tall maximal surfaces). The antichain ~6–8k is borderline for `native_decide` — much
smaller than >10⁶, but not the tens-scale basis seen vs S/Z-first.

## Make-or-break C (`debtcarrier depth4 beam6`): strong player — borderline→FLOOR

`debtcarrier holes1 depth4 beam6 budget3e5` (2118 s): full surface set still **EXCEEDS
300,000**, **WQO antichain = 3630 (still rising — max == current, not saturated)**, max
height **14**. A stronger player shrinks the basis (6144 → 3630) but it stays in the
**thousands**, does NOT saturate to ≤10³, and height drifts to 14 *regardless of depth* —
so the depth-2 drift was not pure suboptimality; the all-orders carrier genuinely needs
height ~14 and a few-thousand-surface basis even with debt≤1.

## Verdict: HONEST FLOOR (best route yet, but does not cross the certification wall)

The hole-debt × surface-WQO separation is the **cleverest and most successful** route tried:
it sidesteps the non-monotonicity that refuted `safeLE` (surface lift via the proven monotone
keystone), debt≤1 is survival-feasible vs all orders (no collapse), and it yields the
**smallest all-orders carrier basis found anywhere** — a WQO antichain of ~3.6k (depth-4) to
~6k (depth-2, saturating), a real ~150–300× compression of the >10⁶ full surface set.

But it does not reach certifiable size, for two compounding reasons:
1. **Basis size.** The antichain is in the low **thousands** (3.6k–6k), well above the ≤~10³
   threshold for a clean `native_decide`, and it does not saturate smaller even with a strong
   player (height drifts to 14 vs the adaptive adversary at every depth tried).
2. **The closure check is O(N²) in the basis size N.** Verifying closure means: for each
   basis surface × piece × placement, the successor must be dominated by *some* basis element
   — an `∃ β' ∈ basis, domLE` scan of the whole basis. So the `native_decide` cost is
   `N × 7 × ~34 × N` = O(N²). At N ≈ 6k that is ≈ **8.6×10⁹** domination checks (~10¹¹
   primitive `colHeight` comparisons) — infeasible. A clean `native_decide` needs N ≲ few
   hundred (few-hundred² × 238 ≈ 10⁷); the all-orders basis is ~10–30× too large.
3. **The closure is also clear-heavy.** Bounding debt *requires line clears* (debt rises on
   every placement, falls only on clears), so the check cannot use WqoCarrier's no-clear
   `place` trick (which bounds the surface but not debt) — `applyStep`/`clearLines` sit inside
   the decide, compounding (2).

So the route relocates the wall (to certifying a clear-heavy ~few-k-surface debt-aware basis
closure) without crossing it — the **same convergence/certification wall** every enumeration
route hits, reached here at the smallest basis yet. The separation is genuinely the right
*structural* idea (and the Lean foundations are sorry-free), but the all-orders basis is
irreducibly a few thousand clear-coupled surfaces, not the tens/hundreds a clean certificate
needs.

**Reusable residue:** `debttraj` / `debtcarrier` (+ `HOLE_CAP`, WQO-antichain tracking) probe
modes; the sorry-free `HoleDebt`/`SurfaceFiber`/`HoleyCarrier` Lean foundations; the precise
basis-size numbers (3.6k–6k). The strongest candidate for a *future* certified attempt if
`native_decide` can be made to handle a clear-active few-thousand-case closure (e.g. by proving
the debt step structurally rather than by decide — the open Lean-engineering question).
