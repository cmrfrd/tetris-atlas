# Last-move exclusions

This necessary reachability rule is now part of the parameter-derived codec language. The
examples and complete sweep below concern **10 columns, height 4, and at most two holes**;
other supported configurations derive their own language automatically. This is not the exact
seed-reachable set.

## Analysis of the saved findings

The saved `codec-playground-tool/after-profile/findings.jsonl` run contains 4,642 distinct
impossibility findings, across 980 top-two-row pairs. They have 8–20 occupied cells. All have at
least one hole; 366 have one and 4,276 have two. They are not simply odd-cell boards or unsupported
top runs: the previous codec already checked cell-count residue and its weaker top support rule.

Representative raw columns and their boards (top row first):

```
[0,1,13,0,1,11,0,0,0,0]        [3,11,11,1,3,0,3,15,3,1]
..#..#....                      .##....#..
..#.......                      .......#..
.....#....                      ###.#.###.
.##.##....                      #####.####
```

The first board has no tetromino subset that can be the final legal hard drop. In the second,
removing the right vertical I would leave an empty internal row. That cannot be the predecessor
of a history from the empty board. The exhaustive engine replay checks every alternative,
including pieces that do not touch the top row.

The simpler six-cell family illustrates the same obstruction:

```
#.#.......
#.........
#.........
#.#.......
```

Only the left vertical I could have been placed last. Removing it leaves two cells separated
by empty rows. There are 216 reflected/translated examples of this family in the default width.

## Why the exclusion is sound

Assume every board *after placement and line clearing* has height at most four. The engine may
transiently place cells higher before a clear; that is allowed in the argument.

1. At width ten, four new cells cannot fill a row that was entirely empty above the previous cap.
   Therefore a cleared row must lie below the previous cap.
2. If the last move cleared any rows, all surviving old cells end below the final top row at the
   cap. Any occupied cells in that top row must belong to the last tetromino.
3. Each standard tetromino has a contiguous run in each nonempty row. Multiple separated top-row
   runs at the cap therefore imply that the last move cleared **zero** rows.
4. Enumerate every translated tetromino subset of the target. Removing it determines a possible
   predecessor uniquely. Reject predecessors with an empty internal row, then replay the engine's
   hard drop and require zero clears and exact equality with the target.
5. If no removal succeeds, no last move exists. The target is unreachable from empty within this
   height-capped history, regardless of earlier hole excursions or bag state.

No reachable board has an empty internal row: a connected tetromino must land on the floor or
existing cells, so it cannot cross an entirely empty horizontal barrier; deleting full rows
preserves this property. The predecessor check deliberately does **not** require the codec's hole
budget, parity rule, or a reachable bag. It allows a larger predecessor set, making exclusion
conservative. Having a predecessor is not sufficient to prove reachability.

The width-greater-than-four premise matters. At width four, a horizontal I above an impossible
board can clear itself and return to the same board. The rule must not be transferred blindly to
that configuration. See [REACHABILITY.md](REACHABILITY.md) for the original counterexample.

The separate top-support condition needs width at least four. At width four, a full row above
the previous cap can only be a horizontal I: clearing it consumes the entire piece and leaves
the old board unchanged. At width three, a full L/J row can clear above the cap while leaving
a new unsupported cell beside an existing unsupported top run. A legal seven-bag replay reaches
columns `[2,1,2]` at height two, so applying the support condition there was incorrect. The codec
now omits it below width four; its owning module retains the complete replay as a regression test.

The top-support argument is inductive over the entire capped history, rather than a claim about
arbitrary boards. Empty starts with no top runs. After a zero-clear move, an old unsupported run
can remain, but a new top run cannot become unsupported: every multi-row standard tetromino has
an occupied cell immediately below its top row, while a horizontal I must land on support.
Merging a new supported run with an old run makes the combined run supported as well. After a
move that clears rows below the cap, all surviving old cells move below the cap, so only one
contiguous piece row can remain at the cap. An unsupported such row has at most three cells:
a four-cell row is the entire horizontal I, which needs support unless it clears itself. At
width four an above-cap I clear leaves the previous board unchanged. These cases preserve the
condition without assuming that earlier boards obey the target hole budget.

Consequently, a board reached after an earlier post-clear height excursion above the codec cap
is outside this proof's history restriction, even if its current height fits. Such a history is
not a counterexample to the capped codec. Earlier hole excursions are allowed and are exercised
explicitly by the independent forward tests in `src/board.rs`.

## Built into the codec language

The current implementation has no excluded-ID file, rank-correction layer, source-binding
constant, or manual regeneration command. `src/decision.rs` constructs a reduced decision diagram
from the constraints during const evaluation. It combines the original occupancy/row/hole/support
rules with the last-move rule above. Each state is a board, independent of how many possible
last pieces could produce it; overlapping predecessor alternatives are counted once.

The diagram records allowed continuations of a board prefix. Equivalent continuations share a
node. A false branch has zero completions, so it receives no ID. Encoding follows the board's
bits and adds the sizes of earlier branches; decoding selects branches using those same sizes.
The resulting IDs are dense without a separate subtraction/indexing step.

For the default carrier, the original 393,685,466 candidates become **392,431,346** states.
All 1,254,120 previously proved exclusions remain absent. An independent complete engine sweep
checked every original candidate and round-tripped every admitted native board; see
[PERFORMANCE.md](PERFORMANCE.md) for the recorded validation and timings.

The same construction applies to other dimensions automatically. For example, six columns,
three rows and two holes now gives **26,365** states, instead of 28,492. No per-configuration
registration, generated asset, or build script is needed. The width-greater-than-four proof is
not applied at widths 1–4. It is applied at height caps through 28, where the engine's 32-bit
columns can hold a transient four-cell piece before clearing. Higher history caps are rejected
by the codec constructor: spawn overlap can invalidate the original occupancy invariant as well.
StandardTetris's 20-row playable height lies within the supported range.

This remains a carrier defined by necessary conditions. A last-move witness need not itself
have a reachable predecessor or compatible bag history. Retained boards therefore are not all
proved reachable, and no survival/closure claim follows from encoding them.

## Independent forward regression checks

The owning `src/board.rs` tests include two expensive checks, run explicitly with:

```sh
cargo test --release -p tetris-codec false_exclusions -- --ignored --nocapture --test-threads=1
```

The first computes complete forward closure from empty for eight small configurations, allowing
every piece order. This is a superset of seven-bag histories, with no search depth limit. The
second follows fixed-seed legal seven-bag histories for five larger configurations and checks
every in-cap successor of each drawn piece. Neither exploration filters parents by codec admission
or hole budget. Every target within the hole budget must encode and round-trip. Failure output
includes the placement indices needed for replay. Sampling is evidence, not a proof for all large
configurations; complete closure establishes the property only for its enumerated configurations.

Fast default tests separately check the engine shape premises, the twenty-row native-board
boundary, and a four-column clear above the codec cap that must leave the board admitted.

## Ten-column one-hole arithmetic specialization

For caps 4–6, the public codec now applies the stronger closed-family invariant described and
proved in [ARITHMETIC.md](ARITHMETIC.md). Its finite scalar recurrence counts valid suffixes
without constructing a graph. The rules above continue to govern other configurations.
