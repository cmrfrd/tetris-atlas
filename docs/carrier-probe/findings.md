# Carrier-Probe Findings — Search-Grounded Closure Route (Phase 0)

**Status: T2 HONEST FLOOR reached.** Three distinct strategies probed; none reached R1
(a nonempty closed carrier of certifiable size ≲1e5). This is a bounded negative
result — a SUCCESS terminal state of the goal loop, not a failure to grind past.

**Date:** 2026-06-23
**Binary:** `crates/tetris-playground/src/bin/research/tetris_carrier_probe/main.rs`
**Engine:** `tetris-game` (10×20, exact `[u32;10]` boards, 7-bag).

---

## Goal recap

Find — by search — a concrete finite set `S` of game states that is (a) nonempty,
(b) closed under all adversarial 7-bag play given a fixed strategy, (c) contains
`init`, and (d) small enough to certify in Lean by `native_decide` (target |S| ≲ 1e5).
Then certify it, composing with the proven M2 bridge to get `TetrisSolvableValid`.
The point of search-first: confirm such an `S` EXISTS before proving anything (both
prior Lean attempts stalled by trying to prove existence of an unconfirmed object).

## What was built

`tetris_carrier_probe` with three modes, all on exact boards under ALL adversary
orders (no lossy abstraction — whatever it finds is directly `native_decide`-able):

- **full**: forward BFS of the exact `(board, bag)` reachable set + backward death
  propagation.
- **boundary**: per-bag macro-BFS — nodes are full-bag "reset surface" boards; one
  edge folds a whole bag of adversarial play (intra-bag transients folded away). This
  is the state space the M2 invariant `tetrisSolvableValid_of_bag_indexed_invariant`
  actually certifies. Optional **admissibility band** (height/roughness/hole/well
  caps): treats band-escape as a death, so the surviving core is a controlled-invariant
  closed set, small by construction when caps are tight.
- **optimal**: AND-OR fixpoint — adversary picks the worst piece (AND), player picks
  ANY admissible placement (OR). Computes the TRUE band-restricted safe set, answering
  "does a closed band exist for ANY strategy?", not just a hand-crafted greedy one.
  (Subsumes every deterministic strategy.) Transient holes allowed mid-bag; full band
  enforced at bag boundaries.

## Strategies probed and regimes (the ladder's R0/R1 evidence)

### Strategy 1 — `flatten` (greedy: minimize holes, height, roughness)
- **Free `(board,bag)` BFS:** truncated at 2M states, 1.13M distinct boards, max
  height 13, **0 deaths**. Height does NOT drift → strategy controls height — but the
  exact reachable set is too large to certify (and BFS had not converged).
- **`boundary`, holes=0 bands:** converge fast but **empty core**:
  - `h6 r2`: 4 boards → core EMPTY
  - `h8 r4`: 944 boards → core EMPTY
  - `h12 r8`: 70,465 boards → core EMPTY
  In every case the core is empty *from `init`* (the adversary forces a boundary hole
  in bag 1). Holes-allowed bands did not converge at feasible budget.

### Strategy 2 — `well` (reserve a well column; flatten the rest; burn the I in the well)
- **`boundary` free:** explodes (5M+ boards, height climbs to 20).
- **holes=0 bands:** same empty-core outcome as `flatten` (greedy can't maintain a
  hole-free boundary surface under adversarial S/Z).

### Strategy 3 — `optimal` (AND-OR; player plays optimally — subsumes all strategies)
- `wellempty h6 holes0 r4`: **2,839,962 states, ALL unsafe, init UNSAFE** (converged).
- `wellempty h8 holes0 r4`: identical (r4 binds before h8) — **init UNSAFE** (converged).
- `wellempty h10 holes0 r6`: 8M states, **EXPLODED** (inconclusive).
- `h8 holes0 r6` (no well): 8M states, **EXPLODED** (inconclusive).

This reproduces the project's known h≤6 empty-safe-set result (machinery validated)
and shows it is a real obstruction, not greedy myopia: even the optimal player cannot
survive the enumerable (r≤4) band.

## The characterization (why R1 is unreachable by this route)

A single, sharp tension explains every regime:

> **Survival needs roughness; roughness explodes the state space.**

- The surface that survives adversarial S/Z (per the SurfaceInvariant analysis) is a
  *structured* surface — reserved well plus S- and Z-valleys/pocket — which has
  non-trivial roughness.
- Bands tight enough to ENUMERATE and CERTIFY (roughness ≲4) cannot host those
  valleys, so even the optimal player has **no surviving core** (init unsafe).
- Bands loose enough to host the valleys (roughness ≥6) **explode past 8M states** —
  beyond both enumeration (memory) and `native_decide`'s ~1e5 Finset-literal capacity.

So no configuration yields a set that is simultaneously **nonempty, closed, and
≲1e5**. The greedy holes=0 bands are small but empty; the optimal r4 band is converged
but empty; loose bands are potentially-nonempty but uncertifiably large. **R1 is
unreachable by the search-grounded-closure route as probed.**

## Honest caveats (what this is NOT)

- This is **not** a proof that Tetris is unsolvable, nor that no closed carrier exists.
  The loose (roughness ≥6) bands are *inconclusive* (exploded, not proven empty); a
  surviving core could exist there — it would simply be too large to certify by
  `native_decide`.
- The optimal-mode intra-bag band applies roughness/well caps to transient states too
  (holes only at boundary). That is conservative: an "unsafe" verdict is strong
  evidence within the band but not a proof for looser bands.
- A nonempty closed carrier almost certainly exists in principle (7-bag defeats
  Burgiel's S/Z adversary); the finding is that **this search route cannot exhibit one
  at certifiable size** — the survival-necessary structure is exactly what exceeds the
  certification budget.

## Conclusion → routes to T2 / next directions

The search-grounded-closure route empirically reconfirms, and now precisely
characterizes, the structural obstruction (crux #66/#72): the carrier the proof needs
lives in a roughness regime that is too large to enumerate or `native_decide`. The
certification-by-enumeration idea does not escape the wall; it relocates it to the
`native_decide` size budget.

Genuinely different directions that remain open (each its own future Phase 0):
1. **Compress before certifying** — quotient the loose-band carrier by reflection
   symmetry (provable bijection, ~2×) and bag-phase canonicalization; only then is a
   `native_decide` plausible. Requires the carrier to compress below ~1e5, unknown.
2. **Amortized potential (Lyapunov), not a set** — prove a per-bag drift bound
   `Φ` (height + structured-roughness penalty) with the one guaranteed I-clear paying
   for ≤6 fillers. Sidesteps enumeration entirely **in the PROOF**; but see the
   addendum below — its empirical leg hits the same wall.
3. **A smarter structured strategy** whose reachable boundary set is provably a small
   surface family (e.g. always rebuild to one of K canonical valley surfaces). If such
   a strategy's boundary core is ≲1e5 and closed, the route revives.

---

## Addendum (2026-06-24): the Lyapunov potential's EMPIRICAL leg also hits the wall

Direction #2 was pursued. A weighted potential `Φ = height + a·roughness + b·holes`
was added to the probe as an admissibility band (`pa<a> pb<b> pc<cap>`), replacing the
box caps. The hypothesis: a potential expresses a *trade-off* the box bands can't
("roughness is OK if height is low"), so its sublevel set might contain a closed core
where box bands collapse-or-explode.

**Result — swept `(a,b) ∈ {(1,1),(1,2),(1,3),(2,2)}`, caps 10–16, well on/off, budget
10M, optimal player: EVERY config EXPLODED past 10M admissible states.** Even `Φ ≤ 10`.

**Why:** a bounded-Φ board set in the survival regime still contains >10⁷ distinct
boards, because roughness up to ~10 (which any survival-permitting cap must allow)
admits combinatorially many surface shapes — and the optimal-player fixpoint must
enumerate the intra-bag `(board, partial-bag)` tree over all of them. The potential
*shape* doesn't shrink this; the size wall is intrinsic to the survival roughness
regime, not to the band geometry.

**Conclusion — the empirical/enumerative leg is now EXHAUSTED from every angle:**
free reachable BFS, box-band boundary closure, optimal-player box bands, and now
optimal-player potential bands ALL hit the same >10⁷ size wall in the survival regime.
Enumeration cannot exhibit OR validate a survival carrier — period. The Lyapunov
direction therefore does **not** offer an empirical shortcut; its only escape is the
**symbolic drift proof**, which requires per-piece placement existence with Φ-control
on a parametric surface family — i.e. it reduces to the same realization crux #66/#72
as SurfaceInvariant and AbstractSafe.

**The unifying finding of all this work:** every distinct-looking route —
hand-crafted carrier (SurfaceInvariant), sound finite abstraction (AbstractSafe),
search-grounded closure (this probe, box + potential bands), and amortized Lyapunov —
**converges to one irreducible crux**: proving a structured survival surface is closed
under all 7 pieces in all adversary orders on 10 columns. Search can't exhibit it (the
survival-roughness state space is >10⁷); abstraction can't make it congruent; hand-proof
stalls on the per-piece geometry. This crux IS the open research content of "solve
Tetris," and it is not dissolved by reframing — only relocated.

---

## Addendum 2 (2026-06-24): the drift / strategy-realization arc

Pursued the amortized-Lyapunov direction empirically via NEW probe modes (`drift`,
`cycle`, `longrun`, `minimax`), testing whether a concrete STRATEGY survives the
worst-order adversary (which would furnish the carrier + empirical Lyapunov region).

**Structural positives (strategy-independent, real):**
- `drift` mode: per-piece local Φ-drift (Φ = Σ heights = cells+holes) is BOUNDED by
  surface roughness — worst-case forced holes ≤ D (well depth): O≤4,S/Z≤3,L/J≤2 at D=4.
  No unbounded trap. Confirmed across W=7/8/9, D=3/4/6.
- Surfaces hosting all 7 pieces hole-free EXIST and are abundant (≈125k of 390k windows
  at W=8), e.g. `[0,1,0,…]`. Individual-hostability is satisfiable (NOT simultaneous
  closure — placing one piece consumes its site).

**The decisive negative (strategy realization):**
- `cycle` mode: from a flat floor f∈[4,14], the well-reserving greedy clears 4 lines and
  ΔΦ=−10 for ALL 5040 orders — BUT this was a flat-start MIRAGE: per bag cells −12 yet
  holes +2, and `longrun` showed those holes ACCUMULATE.
- `longrun`/`minimax`: NO tractable player survives the worst-order adversary. Top-out
  bag: Φ-greedy 5, holes-greedy 4, Lee-weights depth-1 **7**, depth-2 minimax **21**,
  depth-4 minimax **30** — deeper lookahead only DELAYS (sublinear), holes always climb
  to ~23. Full one-bag exact minimax EXPLODES (40M memo / 4.8 GB on bag 1).
- vs RANDOM order, the SAME Lee player survives 1573 bags (11k pc); the project's beam
  survives millions. **So the entire difficulty is the adversarial ORDERING, not Tetris.**

**Conclusion:** worst-order adversarial 7-bag survival is empirically intractable —
no player up to depth-4 lookahead survives 30 bags, and the trend does not converge to
survival; the optimal player dies in every tractable band (r≤4) and is uncomputable in
the survival regime (r≥6, >10⁷). This is the SAME crux from the strategy side, and it
sharpens the open strategic question: **is worst-order adversarial 7-bag even
survivable?** RANDOM 7-bag (real Tetris) demonstrably is. The all-orders proof goal
targets the adversarial case, which may be intractable or false; the achievable,
real-Tetris target is random-bag empirical infinite play (M1).
