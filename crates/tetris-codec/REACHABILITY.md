> Historical analysis of the pre-exclusion carrier. The default carrier has since been refined;
> see [EXCLUSIONS.md](EXCLUSIONS.md) for current counts and exclusions.

# Reachability investigation

This study uses the uncommitted `feat/board-codec` worktree on 2026-10-30.
Parent commit: `8297faee182b6d6ee53a2d853d620c7bff8b73dc` (dirty worktree).
The current codec describes admissible patterns; this investigation does not
change its membership, IDs, or counting tables.

## Model

The engine is `tetris-game`. Every move uses `apply_piece_placement`, including
its hard drop and line clearing. The height cap applies **after every move and
line clear**. A placement may temporarily exceed the cap before clearing rows.
All legal seven-bag orders are allowed; reachability asks whether at least one
legal piece/action history exists from empty.

The hole-history question is kept explicit:

- **Strict:** every post-move board must stay within the hole cap and codec.
- **Current-board-only:** histories may contain boards with more holes, while
  the target board satisfies the codec's hole limit. Height stays capped throughout.

These sets differ. Neither is the set of all boards reachable in unrestricted
20-row Tetris. A reachable board may still be a losing state.

## Complete checks on tiny configurations

The reference BFS explores `(board, remaining_bag)` from `(empty, full_bag)`.
It expands every available piece and every placement that fits the width,
refills the bag after its last piece, and runs to queue exhaustion. It does not
use the codec's structural admission filter to construct the graph. A separate
projection compares reached boards with the carrier. The state-bound assertion
was not hit; these runs are complete, not depth-limited sampling.

| Columns / cap rows / holes | Admitted boards | Reachable with strict hole history | Reachable with target-only hole limit |
|---|---:|---:|---:|
| 10 / 1 / 0 | 511 | 8 | 8 |
| 4 / 2 / 0 | 19 | 16 | 16 |
| 4 / 3 / 0 | 43 | 43 | 43 |
| 4 / 3 / 1 | 179 | 147 | 159 |

The 10-column one-row result is also directly explained: only a horizontal I
fits on the first move, and a second I cannot appear in that bag. Two pieces
provide only eight cells, too few to clear a ten-cell row. Thus the only boards
are empty and the seven horizontal-I positions.

The narrow carrier counts were checked against the actual codec. Every graph
run had zero reachable target boards outside the stated carrier. Additional
bag-agnostic runs appear in `results.txt`; for `(4,3,1)` with target-only holes,
ignoring bags admits 161 boards, versus 159 with legal bags.

## Why incoming hits do not establish reachability

For four columns and a two-row cap, the board with bottom-first rows `[5,5]` is
admitted but absent from the complete reachable set. Dropping a horizontal I
above it fills a new four-cell row, clears that row, and returns exactly the
same board. This self-loop was checked with the engine (`cycle-results.txt`).

Consequently, sampling arbitrary codec parents can mark an impossible entry
as hit. Such a hit establishes an incoming edge, not a path from empty.
The default experiment tracks incoming hits separately from seed witnesses.

## Default codec sampling

Configuration: `BoardCodec<StandardTetris,4,2>`, 393,685,466 admitted board IDs.
All fixtures are generated using this source snapshot's ID ordering; the files
are not a stable wire format or a promise to preserve those IDs.

The large run contains:

- 1,000,000 uniform parent samples, 998,739 distinct parent IDs.
- Every one of the engine's 162 current placements per sampled parent:
  **162,000,000 canonical transitions**. This count is derived from
  `TetrisPiecePlacement::NUM_PLACEMENTS`, not a hardcoded action-space assumption.
- 9,994,552 transitions returned to the codec, producing **9,815,611 distinct
  incoming-hit IDs**. These parents are not assumed reachable and the experiment
  is bag-agnostic.
- 200,000 strict episodes and 200,000 relaxed-hole episodes from empty, each
  capped at 128 placements. Both modes use legal seven-bag orders.
- 1,141,114 strict placements and 1,447,513 relaxed-hole placements.
- 564,822 distinct strict-run board IDs and 133,022 relaxed-run IDs, with
  58,372 overlapping: **639,472 distinct witnessed reachable board IDs** in union.
- The relaxed run made 214,783 excursions outside the hole cap and returned to
  the codec 17,446 times. Being seen only in this run does not prove that a board
  requires an excursion; a different strict path might exist.

The strict run chooses randomly among allowed moves for the drawn piece. The
relaxed run allows more holes and usually chooses a minimum-hole move while
outside the codec, with random exploration mixed in. Episode stops are outcomes
of these sampled paths, not proofs about all possible policies or infinite survival.

A held-out set of **100,000 uniform target IDs**, plus 216 known-negative
controls, was reassessed:

| Status among uniform targets | Count |
|---|---:|
| Incoming hit from sampled codec parents | 2,546 |
| Actual legal-play witness from empty | 165 |
| Impossible by the last-move argument below | 296 |
| Neither a seed witness nor an impossibility proof | 99,539 |

Incoming hits overlap the other observations and are not a reachability class.
Notably, **158 of the 165 witnessed reachable targets had no incoming hit in the
million-parent experiment**. Zero sampled hits is therefore insufficient to
reject an entry, even at this scale.

There are 226 saved action-sequence witnesses. Each was independently replayed
from empty within the analysis program, checking piece availability, bag refills,
height, strict-mode membership where applicable, final codec ID, and line count.
Every positively witnessed uniform target has a saved path. The sampler seeds
are research-sampler seeds, not `TetrisGameRng` seeds; the recorded piece/placement
indices and bag masks specify the actual replay. No fixed bag order was assumed.

Incoming coverage is 9,815,611 / 393,685,466 carrier entries; witnessed coverage
is 639,472 / 393,685,466 carrier entries. **The fraction of the true reachable
set covered remains unknown**, because its size is not established.

## A sound additional last-move test

The following argument applies when **width is greater than four**, history
height is capped at R, and the target reaches height R.

If the last move cleared a row, every cleared row must have been below the old
height cap: a tetromino has only four cells and cannot fill a wider empty row
above that cap. Surviving old cells therefore end below the target's top row.
The target's top-row cells must all come from the last tetromino.

Each horizontal row of a standard tetromino is contiguous. This property was
checked for all engine placements. Therefore:

> If a height-at-cap target has multiple separated occupied top-row runs, its
> last move cannot have cleared any rows.

For those targets we can enumerate **every no-clear predecessor**: remove each
possible translated tetromino footprint, then replay its actual hard drop with
the engine. A predecessor in a real history cannot contain an empty horizontal
row beneath occupied cells. This is an invariant: a hard-dropped connected piece
extends the existing nonempty row interval without a gap, and clearing full rows
compresses that interval without creating an empty row inside it.

If no candidate predecessor satisfies this necessary invariant and replays to
the target, the target is impossible under the height-capped model. This proof
does not require a past hole limit or bag restriction.

The study applied this complete no-clear predecessor test only where the
multiple-top-run condition rules out every clearing predecessor. It produced
**296 distinct proofs among the uniform targets**, in addition to the
**216 known-negative controls: 512 distinct proved-impossible default IDs**.
No conclusion was drawn from absence of sampled hits alone.

The width guard matters: at width four, an I can clear a new row above the old
cap while leaving the old board unchanged, as the self-loop example demonstrates.

### A simple default-codec example

Top row first, `#` occupied:

```text
#.#.......
#.........
#.........
#.#.......
```

Bottom-first row masks are `[5,1,1,5]`; native column limbs are
`[15,0,9,0,0,0,0,0,0,0]`. The current codec accepts this board: height four,
six occupied cells, two holes, no full/internal empty rows, and just one
unsupported top run of length one.

Its separated top cells rule out a clearing last move. Its only removable
whole tetromino is the vertical I in the left column. Removing it leaves two
cells separated by empty rows, an impossible predecessor. The engine replay
checks confirm this enumeration.

The 216-board family moves that vertical I and the other column at least two
columns apart, and places the other column's lower cell at row 0, 1, or 2.
There are 72 ordered nonadjacent column pairs times three lower-cell positions.
All 216 are accepted by the codec and pass the impossibility check. They were
injected as controls; the 296 uniform-target proofs are additional entries.

## Stronger predecessor checks to pursue

For a cap-height target and k cleared rows at width >4, the top k final rows
contain only cells of the new piece. At least k of its four cells were consumed
by the cleared rows, so those top k rows contain at most 4-k occupied cells.
This sharply restricts the clearing cases. In particular, k>=3 is impossible
when ending at the height cap; k=2 requires the vertical-I geometry. These
necessary conditions can make a complete clearing-predecessor enumerator cheaper.

For the default four-row cap, restored full rows can occur only in the first
four pre-clear row positions: at most 16 subsets before mass/shape pruning.
For each candidate, reconstruct the pre-clear board, remove a candidate piece,
and replay the engine. Include all viable predecessor bag states.

Having a predecessor still does not prove reachability: disconnected cycles
can support one another without a path from empty. The practical next stage is
to prioritize unresolved sampled targets, enumerate their full predecessors,
and connect backward searches to a seed-anchored forward frontier. Exhausted
searches can establish impossibility; budget-limited searches remain unknown.
An exact reachable-only codec ultimately needs a proven characterization or
complete seed-reachability construction, followed by counting/ranking over that
set. Merely rejecting sampled misses would be unsound and would not produce a
correct smaller dense ID space.

## Reproduction and scope

Programs, raw data, source/binary hashes, and run metadata are under
`artifacts/output/codec-reachability-analysis/`. Primary outputs are:

- `results.txt`: complete tiny-configuration BFS results.
- `cycle-results.txt`: actual codec count checks and unreachable self-loop.
- `large/sample-results.txt`: large sampling totals.
- `large/sampled-targets.csv`: all target observations and predecessor-test results.
- `large/certified-impossible.csv`: 512 entries, conditional on the stated height
  model and mathematical last-move argument.
- `large/unknown-unhit.csv`: candidates with neither an incoming hit nor a seed
  witness nor a proof. Its name intentionally does not assert impossibility.
- `large/witnesses.csv`: replayed action-sequence witnesses.
- `run-spec.json`: fixed seeds, source hashes, configuration, compiler, and host.

```sh
cargo build --release --offline \
  --manifest-path artifacts/output/codec-reachability-analysis/Cargo.toml
artifacts/output/codec-reachability-analysis/target/release/codec-reachability-analysis
artifacts/output/codec-reachability-analysis/target/release/cycle
artifacts/output/codec-reachability-analysis/target/release/sample \
  artifacts/output/codec-reachability-analysis/large 1000000 200000 100000
```

The large study used about 148 MB of temporary bitmap arrays and completed in
about 22 seconds; resource logs are retained. These timings describe a research
run, not a baseline/candidate speed comparison. Physics, bag legality, and
certificate assertions completed successfully; all target-positive witnesses
were saved and replayed. The tiny BFS carrier sizes were checked against the
actual codec. No production Rust source changed, so codec tests and benchmarks
were not rerun for this documentation/research-only work. All source snapshots
and data correspond to the recorded dirty worktree and pinned nightly.
