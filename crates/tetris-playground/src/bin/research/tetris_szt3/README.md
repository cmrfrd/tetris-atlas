# Small-Variant Exact AND-OR Solver (S/Z/T×3cols, S/Z/T/J×5cols)

**Level:** L3 for the variants (adversarial, exact per ceiling)
**Status:** szt3 DEAD ≤ 15 rows in both modes; sztj5 coop-ALIVE (even hole-free at 20
rows) but adversarially DEAD at every decided ceiling (≤ 8 rows; kill depth grows
~1 bag per row of ceiling), ≥ 9 open

## The variant

3 columns, bag = {S, Z, T} with the 7-bag rule at bag size 3 (bag refills when
empty, the drawer can only produce pieces still in the bag). The variant is a
real `TetrisGameConfig` implementation (`SztCols3<ROWS>`): the engine's
`apply_piece_placement`, `clear_filled_rows`, `is_lost` and `heights` are fully
generic over `C::COLS`/`C::ROWS`/`C::PIECE_SET`; the only 10-column artifact in
`tetris-game` is the precomputed placement table, filtered here by piece width
(S: 3 placements, Z: 3, T: 6).

Cell arithmetic: one bag = 12 cells = exactly 4 full rows, so bounded-forever
play must sustain exactly 4 clears per bag on average.

## Strategy

Exact forward reachability from `(empty, full bag)` over all
(piece, placement) transitions, then retrograde death propagation (the
greatest fixed point). One graph, two solves:

- **adv** — adversary draws the piece (AND), player places (OR).
- **coop** — player draws AND places (OR over both). Cooperative death means no
  infinite play exists at that ceiling at all: any bounded-forever run must
  revisit a `(board, bag)` state, so coop-DEAD ⇔ no reachable cycle fits under
  the ceiling.

ALIVE verdicts are re-verified by replaying every alive-state obligation
through the engine from scratch. ALIVE is monotone in ROWS; DEAD is not, so
the ceiling is swept. On budget blowup the frontier is treated as dead (ALIVE
stays sound; DEAD becomes EXPLODED/inconclusive).

## Results (2026-07-08, M-series MacBook 32 GB, single-threaded)

| ROWS | states | adversarial | cooperative |
|---|---|---|---|
| 4 | 120 | DEAD (depth 3) | DEAD (depth 6) |
| 5 | 450 | DEAD (depth 3) | DEAD (depth 9) |
| 6 | 1,546 | DEAD (depth 4) | DEAD (depth 10) |
| 7 | 5,400 | DEAD (depth 5) | DEAD (depth 12) |
| 8 | 18,064 | DEAD (depth 7) | DEAD (depth 15) |
| 10 | 202,116 | DEAD (depth 9) | DEAD (depth 18) |
| 12 | 2,262,340 | DEAD (depth 11) | DEAD (depth 22) |
| 13 | 7,576,316 | DEAD (depth 13) | DEAD (depth 24) |
| 14 | 25,318,608 | DEAD (depth 14) | DEAD (depth 27) |
| 15 | 84,720,946 | DEAD (depth 15) | DEAD (depth 28) |

Every solve is exact (no budget hit) and **zero** states survive — the whole
reachable set is dead, not just the initial state. State count grows ×3.3 per
row (R=16 ≈ 280M states exceeds this machine); death depth grows only
linearly (~1 ply/row adv, ~2 plies/row coop): headroom is burned at a constant
rate and buys no equilibrium.

**Verdict:** the S/Z/T 3-column game is not playable forever at any ceiling up
to 15 rows — even cooperatively, i.e. no reachable `(board, bag)` cycle exists
below height 16. The canonical 20-row ceiling is out of direct enumeration
reach, but the mechanism is unambiguous (forced-loss trace: alternating
vertical S/Z buries a hole per bag faster than T can flatten; holes accumulate
monotonically while the stack climbs). Turning "no cycle at any height" into a
theorem (hole-accumulation invariant per bag) is the natural Lean follow-up.

## Results — S/Z/T/J on 5 columns (`--variant sztj5`, 2026-07-08)

Adding J (a flattening repair piece) and two columns changes the game class:
16 cells/bag = 3.2 clears/bag demanded.

**Cooperative (player draws): ALIVE.** Already at ROWS=4 (carrier 955,010,
re-verified). Inside the hole-free band the carrier extends to the canonical
ceiling: ROWS=20 coop-ALIVE with 295,532 states, holes = 0 throughout — the
variant is fundamentally playable when the player controls the draw.

**Adversarial: DEAD at every ceiling decided (ROWS ≤ 8), forced-climb pattern.**

| probe | result |
|---|---|
| ROWS=4 exact (gfp) | DEAD (depth 7) |
| ROWS=5, 60M states, frontier-optimistic | **DEAD sound** (depth 10; kill inside explored region) |
| ROWS=6 kill-dfs | **DEAD** — forced kill within 14 plies (0.13s, 237K states) |
| ROWS=7 kill-dfs | **DEAD** — forced kill within 19 plies (5.8s, 8.7M states) |
| ROWS=8 kill-dfs | **DEAD** — forced kill within 23 plies (267s, 218M nodes) |
| ROWS=10 kill-dfs, 300M budget | **no kill within 24 plies** (315M nodes, 18 min; budget at depth 25) — kill-depth law predicts death at ~32, needing ~3–10B states |
| ROWS=20 kill-dfs, unrestricted | **no kill within 24 plies** (354M nodes, budget at depth 25) — the canonical-board player provably survives 6 full bags of worst-case assault |
| holes≤0 band, any ROWS | degenerate: S from flush ground must bury a hole (depth-0 death) |
| holes≤K band (K=1,2,3), ROWS≤12–20 | band-DEAD, kill depth ≈ 5 + 4K |
| holes≤4/5/6 band, ROWS=6 | band-DEAD, depth saturates 12–13 (ceiling binds, not the band) |
| `--adversary greedy` (1-ply minimax script), ROWS=6 | player survives (carrier 27,829) |
| `--adversary sz-first`, ROWS=6 | player survives (carrier 7,663,194) |

Reading: the adversary wins every ceiling decided, and the kill depth grows
linearly — ≈ 4.5 plies (about one 4-piece bag) per extra row of ceiling
(14/19/23 at 6/7/8). That is a forced-climb rate: each bag delivers 3.2 rows of
cells and the adaptive adversary holds the player's effective clearing below
that, netting ≈ +1 row per bag. No fixed script realizes the forcing (both
deterministic adversaries lose — adaptivity is essential), and against
hole-capped players the same rate shows up as ≈ 1 forced hole per bag
(kill depth 5 + 4K). Extrapolation: canonical ROWS=20 dies in ≈ 77 plies
(~19 bags); exact verification is out of enumeration reach (kill-dfs cost grows
~30× per row), so ROWS ≥ 9 is formally open — but every decided ceiling is DEAD
with a stable mechanism, and the ALIVE direction was refuted for every
band/script probe available. Sample R=6 kill line: S then Z spike holes 1→8,
T/J land on debris, by ply 5 Z has no legal placement under the ceiling.

The `kill-dfs` engine that settled 6–8: iterative-deepening AND-OR DFS over the
adversary's forcing subtree, with two path-independent memos (minimal proven
kill depth; maximal proven kill-free depth — both depth-indexed, so no
graph-history-interaction handling). Worst-piece-first ordering by a greedy
eval; a budget abort blocks kill proofs but never fakes them.

The `parallel` engine is a generic port of `atlas/tetris_atlas_inmemory.rs` —
the continuous fully-parallel AND-OR engine (interleaved expansion + monotone
death propagation on all workers, DashMap interning with publish-before-id,
SkipMap mass-ordered frontier, in-flight quiescence, root-death early exit,
conservative finalize at budget). Validated by a 15-test suite: hand-provable
configs (I-only/1-col ALIVE carrier=1 even at 2 rows — clears precede the loss
check; O-only/3-cols DEAD everywhere — no clear is geometrically possible; O+I
2-cols mixed), GFP root-verdict + alive-set-soundness agreement across ceilings
and worker counts, determinism across 12 runs x {1,2,8} workers, budget-finalize
conservatism, and a corruption test that the independent carrier verifier must
reject. At scale it reproduces the GFP's exact band verdict (sztj5 R=6 holes<=2:
monotone root-DEAD in 1.2s on 8 workers, early-exiting at 1.69M of 1.83M
states), and at R=5 unrestricted it correctly reports INCONCLUSIVE when its
root death arrives only via the pessimistic finalize.

The `guided` engine (anytime best-first AND-OR): AND-nodes interned by
(board, bag), one OR-node per bag piece — the (board, bag, piece) triple is
materialized once ever, transpositions merged, nothing re-searched. Exact death
backprop through reverse edges; scarcity-first heap (children of OR-nodes with
few surviving replies lead, badness breaks ties); final pessimistic sweep can
certify a sound ALIVE, and a drained heap yields the exact GFP. Finding
(sztj5): the guided global heap banks proven-dead states at ~1M/s (R=20: 38.6M
dead in 45s) but does NOT compose root proofs — at R=6 it banked 29M deaths
without finishing the refutation kill-dfs completed in 237K states. Global
best-first lacks path discipline; depth-first ∀-completion is what closes
proofs. The engines are complementary: kill-dfs proves, guided banks cache /
detects ALIVE. A principled upgrade would be df-pn (proof-number thresholds).

## Soundness levers

- `--max-holes K`: player-side band — ALIVE sound for the unrestricted game,
  DEAD band-relative.
- `--adversary sz-first|greedy`: scripted adversary — DEAD sound for the full
  game, ALIVE a necessary-condition pass only.
- Budget explosions solve both frontier polarities (pessimistic → ALIVE sound,
  optimistic → DEAD sound); only disagreement is inconclusive.

## Usage

```sh
cargo run --release -p tetris-playground --bin tetris_szt3
cargo run --release -p tetris-playground --bin tetris_szt3 -- --rows 13,14,15 --budget 150000000
cargo run --release -p tetris-playground --bin tetris_szt3 -- --rows 8 --trace
cargo run --release -p tetris-playground --bin tetris_szt3 -- --variant sztj5 --rows 4,5,6
cargo run --release -p tetris-playground --bin tetris_szt3 -- --variant sztj5 --max-holes 2 --rows 6,8,10
cargo run --release -p tetris-playground --bin tetris_szt3 -- --variant sztj5 --adversary greedy --rows 6
```

Unit tests (`cargo test -p tetris-playground --bin tetris_szt3`) include an
O-only 2-column config that the solver must prove ALIVE (a tiny self-loop
carrier), and a cross-check that variant placement application matches the
first 3 columns of the canonical 10-column board.
