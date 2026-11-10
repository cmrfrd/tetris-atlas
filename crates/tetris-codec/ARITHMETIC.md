# Arithmetic ranking

`BoardCodec<C, R, 1>` automatically uses arithmetic ranking when `C::COLS == 10` and
`R` is four through six. No benchmark feature flag or build script is needed. The arithmetic
kernel stays private; the permanent benchmark suite uses the public board and row APIs.

The arithmetic language matches the fully refined one-hole proof, including its additional
exclusions:

| Height cap | Previous graph codec | Current arithmetic codec |
|---:|---:|---:|
| 4 | 58,527,051 | 58,149,719 |
| 5 | 463,654,889 | 459,948,915 |
| 6 | 2,573,076,285 | 2,551,894,523 |

IDs are dense, with a new order: zero-hole boards first, then hole-column height, hole row, hole
column, and the remaining column heights. Numeric datasets need reindexing before switching.
These are necessary-condition carriers, not exact reachable-state sets or survival proofs.
Every post-clear board in a considered history must respect the cap; earlier hole excursions
are allowed.

## Structural counts

A solid column is determined by its height. For an allowed interval `[a,b]`, let `N=b-a+1`
and let `D` be the number of even heights minus the number of odd heights. The number of
`n`-column words with occupied-cell parity `p` is:

```
(N^n + (-1)^p D^n) / 2
```

The implementation reads `N^n` from a shared 352-byte compile-time power table
(`N <= 7`, `n <= 10`). Since `D` is only -1, 0, or 1, its power reduces to parity
operations, including the empty-word case. This avoids two runtime exponentiation loops
without storing board IDs or changing their order. Queries still allocate nothing.

Subtract words with every height positive to exclude full rows. Solid columns cannot create an
empty internal row.

With one hole at row `y` in a column of height `h`, all other columns are solid. No empty internal
row requires another column of height greater than `y`. No full row requires another column of
height zero, except that a bottom-row hole also permits minimum other height one. These are
minimum/maximum conditions, counted with four inclusion-exclusion terms of the formula above.
The hole column contributes `h-1` occupied cells to parity.

## A closed impossible family

Within that structural carrier, the refined proof removes exactly this family:

- The hole column reaches the cap `R`, with `R-4 <= y <= R-2`.
- Every other column is solid, with height at most `y` or exactly `R`.
- At least one other column reaches `R`.
- All cap-height columns, including the hole column, are nonadjacent.

The last move must clear zero rows: separated top runs at the cap cannot be left by a clearing
move in a ten-column capped history. A predecessor cannot remove the upper fragment of the hole
column: it has at most three cells, with no neighboring cells high enough to complete a piece.
A piece touching another isolated cap column must lower it to at most `y`; otherwise its shape
would require a neighboring cell above the allowed lower columns. Removing the last other cap
column would leave row `y` empty below the surviving hole-column fragment. A move entirely among
lower columns only decreases their heights. Thus every possible predecessor satisfying the
necessary row invariants remains in the same family.

Every backward zero-clear move removes four cells. The family excludes empty, so no finite
history from empty can reach it. This argument does not require a cap on earlier holes or a
compatible bag; it already allows every piece order.

For example, columns `[11,0,15,0,3,3,0,0,1,0]` have a legal final O placement on columns four and
five. Removing that O leaves `[11,0,15,0,0,0,0,0,1,0]`, which is still impossible. Having an immediate
last-move witness therefore does not establish reachability. A module-local regression replays
this engine placement and checks both exclusions.

## Count that family with arithmetic

Treat a lower solid column as a symbol of weight `A`, and another cap column as a symbol of
weight `U`. Adjacent cap symbols are forbidden. Their weighted word counts obey:

```
F(0) = 1
F(1) = A + U
F(n) = A * F(n-1) + A * U * F(n-2)
```

The fixed hole column forces neighboring columns to be lower. Its left and right intervals
factor into these counts. Require another cap column by subtracting the all-lower case; require
a zero column by subtracting the zero-free alphabet. Evaluate once with all cell weights `+1`
and once with `-1` to select occupied-cell parity. Differentiating the recurrence with respect
to `U` counts a distinguished cap symbol and sums all hole positions without building a table.

Runtime recurrence queries use a shared 616-byte constant array of 154 signed counts for
suffix lengths 0–10, lower-symbol weights -1–5 and cap weights -1/+1. A zero cap weight reduces
to a power lookup, including signed parity and the empty word. Const evaluation checks that
every stored count fits `i32`; queries widen to `i64` for inclusion-exclusion arithmetic.

This yields 474,176 exclusions at cap four, 4,037,830 at cap five, and 21,740,294 at cap six.
The arithmetic sets match the recursively refined decision diagrams, including all members,
not just their counts.

## Rank and unrank

Each hole-metadata bucket has an arithmetic count. Const evaluation builds cumulative offsets
for every bucket, with a final sentinel equal to the board count. The three arrays contain
61, 101 and 151 `u32` entries, totaling 1,252 bytes. Ranking computes the metadata index directly;
unranking finds the last offset at or below the ID with binary partitioning. Repeated offsets
skip empty buckets. Suffix ranks still use arithmetic completion counts. Prefixes need only
height bounds, parity, and a few flags for the impossible family.

Another 112 bytes describe every packed column at caps 4–6. Each byte holds height and
hole-row-plus-one, or a rejection marker for multiple holes. This replaces repeated height,
population-count and hole-position calculations. The tables contain counts and bit descriptions;
there are no stored counts per board or per decision node.

Once a prefix has met its row conditions and escaped the impossible family, the remaining
height word uses ordinary radix arithmetic with a parity bit. The zero-hole sector also admits
a direct radix formula: start with the unrestricted even-parity rank, then subtract the count
of all-positive height words preceding its first zero column.

The arithmetic kernel and public codec are zero-sized; the codec type supplies its shared tables.
The three groups of new constant arrays total 1,980 bytes, shared across the supported caps, in addition
to the existing 352-byte power table. The board count is a const expression. Queries allocate
nothing and use no generated assets, excluded-ID files, memoization, or retained lookup-node
arrays. Temporary values are scalar prefix/count state and ten-column input/output arrays.

## Verification and measurement

Module-local tests check totals, invalid inputs, random rank/unrank, exact suffix counts against
independent engine enumeration, differentiated metadata counts, and exhaustive small radix words.
The retained experiment under `artifacts/output/codec-10-column-sweep-20261103/` additionally checks:

- Every one of the 26,252,300 family members against the refined reference diagram.
- 52,287 actual engine predecessor transitions sampled across that family, all staying in it.
- 196,608 generated boards for reference-membership agreement, with the public native/row APIs
  checked on the same boards after integration.
- 24,576 legal-history fixtures for arithmetic encode/decode agreement.
- 3,000,000 uniformly sampled IDs for public native/row round trips and independent reference
  membership, covering selection across the full dense ID ranges.

An additional ignored test explores fixed-seed legal seven-bag histories through the public
codecs at all three caps, retaining parents above the target hole budget. The original one-step family was also independently checked with exhaustive engine predecessor
replay for all 987,232 members before deriving the stronger invariant.

The permanent Criterion suite in `tetris-benches` covers public native-board and row-mask
encoding and decoding for all three arithmetic caps. Fixed raw-board witnesses are prepared and
checked outside timing, without selecting inputs through codec IDs. No external decision tables,
variant environment variables or special benchmark features are required. See [README.md](README.md)
for the named-baseline commands. Older packed-kernel comparisons below are historical experiments;
they use different workloads and must not be compared directly with the permanent suite.

Four- and five-hole arithmetic ranking remains unresolved. Their counts and additional bounded
proof reductions are recorded in the experiment, but no weaker substitute has replaced their
existing exclusion rules.

The final matched packed-kernel comparison used two balanced trials per variant on native M1 Pro.
Compared with the previous graph language, cap-six encoding/decoding improved about 29%/26%.
Caps four/five regressed about 14%/4% for encoding and 23%/15% for decoding (roughly 4–25 ns).
These are kernel controls, not complete engine-workload speedups. Arithmetic removes all retained
node payload for these configurations. Other configurations still use their existing graphs.
