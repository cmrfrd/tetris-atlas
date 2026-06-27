# tetris_phase_atlas — progress log

A phase-layered, 5-bag, perfect-clear atlas. The mission: build an Atlas that
proves Tetris is survivable forever under canonical 7-bag rules by closing the
`empty → 5 bags → empty` cycle (and its multi-cycle generalizations) against an
adversarial bag order.

This file is the running journal: what was tried, what the numbers said, what to
try next. Newest entries at the top.

---

## The core idea

Every tetromino is 4 cells. A perfect clear (board back to empty) needs the
running cell count to be a multiple of 10, i.e. pieces placed `≡ 0 (mod 5)`. Bag
boundaries are multiples of 7. They align only at `lcm(5,7) = 35 = 5 bags`. So
after exactly 5 bags you have both a *fresh bag* and the geometric possibility of
an *empty board* (14 lines cleared). The 5-bag empty→empty loop is the minimal
cycle whose repetition proves infinite play.

Because the bag refills to full at every boundary, a bag-boundary state is
identified by the **board alone**. So we track six phase sets of boards:

```
P0 = {empty}  →  P1  →  P2  →  P3  →  P4  →  P5 = {empty}
```

`Pk` = boards that, with the remaining `5-k` bags, can be driven back to empty.

### The soundness rule (the whole game)

A board `b ∈ Pk` is only a valid "solved" merge target if it returns to empty for
**every** upcoming adversarial bag order — i.e. `Pk` is *adversarially closed*.
A trajectory found cooperatively (one specific bag order) only proves `b` good for
*that* order. Therefore:

- **Cooperative discovery** (v1) finds *candidate* phase boards and measures the
  per-tuple perfect-clear rate. It does NOT prove anything on its own, and it
  does NOT count cross-tuple "merges" as solves (they would be unsound).
- **Adversarial closure** (v2) is where the proof lives: certify, layer by layer,
  that each candidate board survives all bag orders into the next phase. The
  within-supercycle graph is a DAG in the step index (step 0→35), so closure is
  an *exact backward pass*, no fixed-point iteration needed inside one cycle.

---

## Log

### 2026-06-27 — v2: exact adversarial certification (`certify`)

**What:** `certify --bag-cycles N --max-height H` runs the exact per-piece online
AND-OR game over an N-bag cycle: `good(board, step, bag)` = AND over the
adversary's possible next pieces, OR over the player's hard-drop placements,
terminal = empty at step `N*7`. Memoized on `(board, step, bag)`; pruned by PC
feasibility and the height cap. **Holes are allowed** during intermediates (only
height is capped) — the prior hole-free solvers can't even contain the empty
board (a known Lean result: an S-piece forces a transient hole).

**Results (bag_cycles=5):**

| max_height | status        | root_cov | nodes  | time   |
|-----------:|---------------|---------:|-------:|-------:|
| 3          | NOT-WINNING   | 0/7      | 19.5M  | 6.1s   |
| 4          | INCONCLUSIVE  | 0/7      | 150M+  | 190s   | (hit node budget; ~7.5 GB memo)

**Findings:**
1. The strict 5-bag empty reset is **NOT-WINNING and exact at height ≤ 3** —
   the adversary defeats *every* opening piece (0/7 coverage).
2. **Why this is expected (impossibility argument):** if even one full 35-piece
   bag sequence admits *no* perfect clear, the per-piece adversary can realize
   exactly that sequence, so the player cannot force empty. v1 found ~36% of
   sequences with no beam-PC, i.e. strong evidence such no-PC sequences exist.
   So the strict 5-bag empty reset is (almost surely) impossible at full height,
   and provably impossible at height ≤ 3. **The empty-every-5-bags target is a
   dead end** — exactly the case the user anticipated for "the other 40%".
3. **The carrier wall, reproduced:** at height ≥ 4 the per-piece search explodes
   (150M+ `(board,step,bag)` states, multi-GB memo) — consistent with every
   prior route flooring at >1e5–5e5 carrier boards. Naive enumeration of the
   adversarial cycle does not scale past height 3.

**Consequence for direction:** stop chasing return-to-*empty*. Survival only
needs the board to stay *bounded* under a policy that answers every bag order —
a closed safe set that need never be empty again after the start. That is v3.

### 2026-06-27 — v1: cooperative discovery (beam + candidate carrier)

**What:** `search_pc` beam-searches one fixed 5-bag sequence for any perfect
clear (depth 35, ranked by `height_mse_distance_from_empty`, pruned by the PC
feasibility filter `(cells + 4·remaining) % 10 == 0`). Verified PCs deposit their
four interior boundary boards into the candidate carrier `P1..P4`. Parallel over
tuples with rayon. Metrics → `artifacts/output/tetris_phase_atlas/`.

**Results (start=0, 500 tuples, max_height=20):**

| beam_width | pc_rate | carrier [P1,P2,P3,P4] | rate (tuples/s) |
|-----------:|--------:|-----------------------|----------------:|
| 512        | 0.0%    | [0,0,0,0]             | 243             |
| 2048       | 2.8%    | [12,6,1,1]            | 45              |
| 8192       | 63.8%   | [144,11,1,1]          | 9               |

**Findings:**
1. The cooperative PC rate is **~64%** at beam 8192 — consistent with the
   "~60% of 5-bag configs can reach empty" hypothesis. This is a beam *lower
   bound* on the true rate; a beam cannot prove non-existence of a PC.
2. **The carrier funnels hard:** P1 is wide (144) but P3=P4=1. Late-cycle PC
   trajectories converge to (nearly) a single board. Whether that is a genuine
   structural funnel or a determinism artifact of the greedy distance heuristic
   is open — to be checked in v2 with exact enumeration.
3. PC rate is *very* sensitive to beam width (0% → 64%). The `height_mse`
   ranking is a weak PC proxy; a better PC-oriented heuristic should reach high
   rate at far lower width (big speedup). TODO.

**Correctness:** every deposited board comes from a replay-verified PC (empty
final board, exactly 14 lines). 7 unit tests cover permutation enumeration,
base-5040 ordinal mapping, PC arithmetic, a hand-built 5-O-piece PC, carrier
deposit semantics, and "no reported PC fails replay".

**Next:**
- v2: adversarial AND-OR closure over the carrier. Layered exact backward pass.
  Per-bag-permutation model first (matches the 5040 enumeration; full within-bag
  lookahead), then tighten to per-piece online.
- Better PC heuristic (cells-first / hole-aware) to raise rate at low width.
- An exact per-tuple PC oracle (full search, not beam) to pin the true PC rate
  and settle whether ~36% genuinely have no 5-bag PC.
