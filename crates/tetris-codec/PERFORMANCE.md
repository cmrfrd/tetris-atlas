# Permanent public-API benchmark suite

The current suite is always registered and uses no codec research or large-case feature flags.
It covers ten-column height/hole caps 1/0, 4/1, 5/1, 6/1, 4/2, 2/4 and 20/0, plus six-column 3/2.
Every case checks inverse operations and measures native/row encoding and decoding over 256
codec-independent legal-play witnesses. The existing 4/2 shared raw rows are retained. Other
fixtures use a fixed xorshift seed and canonical engine/bag histories, with blocked draws starting
fresh histories. Admission changes cannot alter the sampled inputs. Printed fixture hashes and
state counts let before/after runs verify identical workloads. Setup and allocation are outside
measurement. All cases inherit standard Criterion settings, including command-line overrides.
The initial short timing experiment used 30 samples, a 0.2-second warmup and one-second
measurement; its host was noisy, so it does not establish performance neutrality.

A ten-column 4/4 case was rejected for the permanent suite after more than ten minutes of cold
const compilation; the 2/4 case covers the higher-hole graph path with bounded construction.
Renamed benchmark cases and changed fixture definitions require fresh named baselines. The prior
one-row and packed-kernel numbers below describe their original workloads, not this suite.
The `never-inline` feature forwards to the codec for cargo-show-asm inspection; it is not used
for normal performance measurements. See [README.md](README.md) for the commands.

# Integrated arithmetic caches and zero-sized handles

`BoardCodec` is zero-sized and obtains tables from its type's associated constant. The ten-column
height-4–6 one-hole path uses bounded const arrays: 616 bytes of signed recurrence counts,
1,252 bytes of cumulative metadata offsets and 112 bytes of column descriptions. These 1,980
bytes are shared across the three caps, alongside the existing 352-byte power table. They store
counts and bit descriptions, rather than boards or excluded IDs. Queries allocate nothing.

Metadata ranking uses a direct arithmetic index. Selection binary-partitions cumulative offsets,
with a final count sentinel and repeated offsets for empty buckets. Signed word-count queries
use the recurrence array, with a closed form for zero cap weights. Module-local tests enumerate
marker words and column patterns independently and check every metadata bucket boundary.
The numeric IDs, admission rules and permanent benchmark fixtures are preserved.

The selected combination is Z+R+M+C in the retained independent experiment
`artifacts/output/codec-combinations-20261110T014016Z/`. Its `SUMMARY.md` separates repeated
primary measurements from adjacent retests and records the power-reuse tradeoff. Integration
measurements and workspace validation are recorded separately from that experiment.

The integrated branch was rebuilt and measured against the preserved original executable on
native Apple M1 Pro with nightly-2025-11-23, battery power and Low Power Mode off. A fresh named
baseline preceded editing. After all builds and a 60-second cooldown, all 32 permanent cases ran
in A–B–B–A order, with 60 samples, 0.4 s warmup and 2 s measurement. Results below are geometric
means of two matched batch-median ratios across the native and row APIs, expressed as time
reductions. Mean-change confidence intervals are reported separately in the retained report.

| Ten-column height cap, one hole | Encode time reduction | Decode time reduction |
|---:|---:|---:|
| 4 | 12.32–13.61% | 3.85–5.00% |
| 5 | 20.54–21.12% | 3.92–5.13% |
| 6 | 13.61–16.79% | 0.23–0.62% |

Every encoding mean-change interval excludes zero in both paired trials. Some decoding intervals
include zero, especially the nearly neutral cap-six row case. The default 4/2 graph row encoder
uses 19.08% less time. Criterion reports no regressions across the 32 cases; the largest positive
matched-median change is 0.56% in six-column native encoding. Original-executable control drift
ranges from -2.44% to +1.39%; small changes remain sensitive to background load and scheduling.

A separate A–B–B–A replay of four public operations on 4,096 fixed carrier boards per height
uses 38.92%, 31.55% and 29.49% less time at caps four, five and six. All output digests agree
and hot-loop allocation counters remain zero. This corpus includes boards not proven reachable.
These measurements do not establish a gameplay, survival or individual-move p95 improvement.

Normal optimized ARM64 assembly has 1,778 instruction / 147 conditional-branch sites in shared
`select`, versus 3,534 / 291 before. Rank changes from 3,411 / 381 to 3,206 / 398: static branch
sites do not uniformly decrease. Column extraction is unrolled, signed counts use table loads,
and several choices use conditional-select/negation instructions. The cap-four and cap-six
`bad_count` entry stack frames shrink from 144 to 32 bytes. The benchmark code section shrinks
by 28,368 bytes; net read-only constants grow by 1,336 bytes. These are static whole-executable
measurements, separate from logical array bytes, heap/RSS and executed instruction counts.

Formatting, strict workspace all-target Clippy, all 694 workspace tests/doctests, focused engine
and search tests, all 65 RocksDB partitioned tests, and the three ignored codec correctness
audits pass. Seven tests are ignored by the default workspace run; the codec audits are run
separately. Differential checks cover 600,000 IDs, 600,000 structured boards and 600,000 raw
boards plus boundaries against the original. Both normal and `never-inline` builds are checked;
eight separate sampling profiles succeed. No hardware counters were collected.

The dirty source snapshots are based on `f82b1a9ec16e2bdea27dc073833a19e16ec2ff35`.
Exact commands, fixtures, source/binary hashes, samples, intervals, control drift, replay/RSS,
assembly, profiles and validation logs are retained locally under
`artifacts/output/codec-integrated-20261110T033902Z/`. See `REPORT.md` and `integration.patch`.

# Arithmetic counting leaf optimization (2026-11-09)

The height-4–6 one-hole arithmetic path now uses a shared 352-byte const power table and
parity arithmetic in `words`, replacing its two runtime exponentiation loops. Counts, board
admission and numeric ID order are preserved. There is no new allocation or per-codec storage.
The `never-inline` ARM64 inspection body changes from 55 to 42 instruction sites and from
14 to four branch sites, including safe bounds checks. These are static assembly counts,
not measured dynamic instruction counts. In the normal benchmark executable, the code section
shrinks by 2,336 bytes; page-aligned mapped segment sizes are unchanged.

The permanent public-API suite was measured on Apple M1 Pro ARM64, pinned nightly-2025-11-23,
AC power, Low Power Mode off. Baseline and candidate are dirty snapshots based on
`f82b1a9ec16e2bdea27dc073833a19e16ec2ff35`; the baseline executable is preserved. All eight
configurations (32 cases) ran before/after; the three changed configurations additionally ran
in reversed order. Each trial used 40 Criterion samples, 0.3 s warmup and 1.5 s measurement.
Averages of the two batch medians yield these reductions across row/native APIs:

| Height cap, one hole, ten columns | Encode time reduction | Decode time reduction |
|---:|---:|---:|
| 4 | 29.45–30.15% | 34.61–38.39% |
| 5 | 32.44–33.48% | 38.99–41.02% |
| 6 | 31.43–34.32% | 35.92–39.70% |

A final Criterion comparison against the latest old-binary baseline reports improvement for
all 12 affected cases (`p < 0.05`). The other 20 cases range from -3.93% to +0.79% median time
change in the full-suite comparison. Background load was present; these controls support no
large observed regression, not exact neutrality. These are codec fixture timings, with setup
outside measurement; no end-to-end gameplay speedup or per-move p95 claim is made.

The parity-only and power-table-only changes were separately measured before combining them.
An independent parity DP covers every supported interval and suffix length, including empty
intervals and zero-length words. Differential checks compare 100,000 sampled IDs per height
against the preserved arithmetic implementation, plus fixture and invalid-input checks.

Sources, executables, fixture hashes, full per-case median/mean confidence intervals, named
Criterion samples, normal/inspection assembly and verification logs are retained locally in
`artifacts/output/codec-leaf-perf-20261109T220516Z/`. The experiment branch is
`experiment/codec-leaf-perf-20261109t220516z`. See its `REPORT.md` for exact commands and scope.

# Exact one-row reachability specialization (2026-11-09)

Full-seven-bag codecs with 8–12 columns and a one-row whole-history height cap now use exact
arithmetic seed reachability. For ten columns the language shrinks from 134 boards to eight,
packed IDs from eight bits to three, native IDs from `u16` to `u8`, and retained lookup-node
payload from 426 bytes to zero. See [REACHABLE.md](REACHABLE.md) for the proof and scope.

The fixed-fixture old/new/new/old comparison used the same eight raw reachable boards repeated
into 256-board batches. IDs were prepared outside timing. Each binary had two trials, each
with 30 Criterion samples, 0.2 s warmup and 1 s measurement. On ARM64 Apple M1 Pro with the pinned
nightly, AC power and Low Power Mode off, averages of the two trial means were:

| One-row operation | Before us / 256 | After us / 256 | Latency change |
|---|---:|---:|---:|
| Native encode | 5.1692 | 0.6901 | -86.65% |
| Native decode | 4.1770 | 0.7736 | -81.48% |
| Row-mask encode | 4.3830 | 0.2174 | -95.04% |
| Row-mask decode | 4.1490 | 0.1412 | -96.60% |

All four operations significantly improved against the pre-edit named baseline in both candidate
trials. The unchanged height-4–6 one-hole packed kernels ranged from -0.66% to +0.76% aggregate
latency change. Small significant shifts occur in individual comparisons; these measurements
support absence of a large repeatable regression in those fixtures, not exactly unchanged cost.
No larger-cap native-wrapper or end-to-end gameplay speedup is claimed. This is hot-cache kernel
timing and excludes setup allocations.

Evidence is in `artifacts/output/codec-reachability-refine-20261109/REPORT.md`, with source archives,
exact executable hashes, fixed input hashes, per-trial samples/95% confidence intervals/dispersion,
commands and host snapshots. Base HEAD is `f82b1a9` plus uncommitted changes; all nine verification
checks passed, including 689 workspace tests and 65 RocksDB tests. Height caps 4–6 still encode
conservative supersets; this refinement does not make them exact reachable sets.

# Current arithmetic specialization

Ten-column, one-hole codecs at height caps 4–6 now rank algebraically, with zero retained node
payload. The balanced packed-kernel comparison shows encoding/decoding changes of approximately
+14%/+23% at cap 4, +4%/+15% at cap 5, and -29%/-26% at cap 6. Positive means slower.
The choice prioritizes algebraic ranking and removes 26,960 / 53,096 / 69,104 bytes of prior node
payload respectively. All still use `u32` IDs. These are matched packed-table kernel controls,
not complete native-API or gameplay throughput measurements.

See [ARITHMETIC.md](ARITHMETIC.md) for formulas, scope, proof and verification, and the local
`artifacts/output/codec-10-column-sweep-20261103/REPORT.md` for raw-run provenance and limitations.
The remaining measurements below are historical and describe the general graph backend, which
continues to serve other configurations.

# Rule-derived codec experiment

## Source and scope

Measured candidate: uncommitted `feat/codec-rule-ranking`, based on
`8297faee182b6d6ee53a2d853d620c7bff8b73dc`, in `tetris-atlas-codec-rules`.
At measurement time the previous unfiltered and excluded-index worktrees were preserved.
They were subsequently consolidated into this worktree; verified source archives, working/index
patches, and a branch-history bundle are retained under
`artifacts/worktree-backups/codec-consolidation-20261101T004157Z/`.
The two superseded codec worktrees and branches were removed. The implementation was then
consolidated for commit on `feat/codec-rule-ranking`; the measurements below retain their original
dirty-state provenance. Main and unrelated experiment worktrees were not changed.
Evidence and exact executables are under `artifacts/output/codec-rule-bdd/`; the benchmark
snapshot is under `artifacts/benchmarks/8297faee-codec-rule-ranking-dirty/`.

The candidate replaces candidate counting plus excluded-ID correction with one parameter-derived
reduced binary decision diagram. No excluded-board list is read, embedded, or generated.
This historical construction used an engine mask accessor; the current codec instead owns
normalized symbolic footprints checked against public engine placement, without engine edits. Runtime ranking and selection
traverse the same diagram, so impossible branches consume no numeric IDs.

Both the previous index codec and this candidate admit 392,431,346 default boards: 29 packed
bits, `u32` native IDs, and checked `u64` entry points. ID order changes; existing numeric-ID
datasets must be rebuilt or reindexed from raw boards. There is no historical decoder.
These rules are necessary conditions, not exact seed reachability or a survival certificate.

## Correctness evidence

`audit/` preserves an independent release driver and the original unfiltered codec. It enumerated
all **393,685,466** original default candidates. For every candidate it compared new admission
with exhaustive zero-clear last-move removal followed by actual engine replay. It found:

- **1,254,120** engine-proved rejections, matching the previous analysis exactly.
- **392,431,346** admissions, each checked through native encode and decode.
- **3,885,312** additional checks of row API/ID agreement.
- No admission, rejection, or round-trip disagreement.

`full-audit.json` and `full-audit-progress.log` record the run. Its 90.38 seconds overlapped
compilation and is validation wall time, not performance evidence.

The sampling tool independently replayed all **4,642** saved impossibility findings and
**1,179** saved witness paths against the current engine. The engine-source fingerprint changed
because the piece-mask accessor became const/public; the ordinary verifier refuses that mismatch.
An explicit `verify --recheck-engine` replays raw facts after source review. Resume never bypasses
fingerprints, model guards, or workload guards.

A fixed-seed 11-batch run sampled 110,000 parents (17,820,000 placements), 110,000 candidate boards,
and 22,000 legal episodes. It recorded 89 witness paths and zero additional proved exclusions.
Zero findings do not prove that the remaining boards are reachable. A separate ten-batch run
resumed for one batch with a different worker count produced byte-identical checkpoint,
findings, and watched-target CSV. Resume of a previous **same-count but different-order** dataset
was refused without changing its checkpoint, findings, or CSV. See `sampler-validation.json`.

## Retained storage and construction

| Default implementation | Retained record payload |
|---|---:|
| Original base counting tables | 384,424 B |
| Previous base plus excluded-ID index | 2,940,680 B |
| First rule diagram, unpacked records | 318,264 B |
| Current packed rule diagram | **106,088 B** |

The current diagram has 13,261 records, including terminals, at eight bytes each. This is
**96.39% less record payload** than the index codec and **72.40% less** than the original base
counter. These are compiled lookup bytes, not a reduction in encoded board width or a measurement
of process RSS. Small metadata headers are excluded. These historical measurements used
pointer-sized handles; current handles are zero-sized as described above. Lookups allocate no
heap storage and need only scalar traversal state plus their input/output boards.
There is no per-query export buffer or cache initialization.

Narrow child/count fields are selected losslessly. Even-width occupancy residue means no live
variable can be skipped, so those configurations omit levels and their traversal bookkeeping.
Odd widths keep levels and handle skipped variables explicitly. Native boards of at most 64 cells
use a packed bitstream; wider boards use the same language through column-bit access.

Construction is deliberately bounded and runs in const evaluation. Fresh default consumer builds
in this experiment took about 2–3 minutes; separate targets can each evaluate the diagram.
Compile-time memory was not measured. Large hole budgets can be substantially more expensive;
the historical six-hole benchmark was omitted from the permanent bounded suite. Counts must fit `u64`,
and symbolic workspace/node limits can reject large parameter combinations. Lookup performance
numbers below concern the default configuration, not every possible parameter set.

Prototype ablations preserved in the artifact directory found that column order needed 13,261
final nodes versus 317,316 with row order. Periodic collection reduced temporary symbolic work.
A rejected care-set construction produced the same counts but more work. Those prototype timings
are design evidence, not comparisons of production codec lookup speed.

## Timing protocol

Host: native arm64 Apple M1 Pro, 32 GiB RAM, macOS 26.6.2 (25G83), pinned
`nightly-2025-11-23`, rustc `1.93.0-nightly (94b49fd99)`, LLVM 21.1.5. Measurements use AC power;
each Criterion run records power settings, load, binary SHA-256, and its exact command.
Background activity is not controlled, so repeated runs and the size of the effect matter.
No build, audit, or profiler from this experiment runs alongside the final timing series.

The stable group `board_codec/standard_4_2_reachable_shared` uses the same 256 raw reachable boards
in both saved executables. IDs are prepared outside timing; changed ordering cannot change the
raw-board workload. Four kernels measure native/row encode/decode without allocation. Each run
uses 40 samples, one-second warm-up, and two-second measurement per kernel. Runs alternate
baseline/candidate/candidate/baseline. Reports retain Criterion distributions and confidence
intervals; repeated-run dispersion below describes host variability, not a new significance test.
The previous and candidate full benchmark binaries include different optional groups, so their
whole-file size difference is not attributed to this change.

### Fixed reachable-board kernels

Two trials per implementation, 40 Criterion samples per kernel per trial. Entries are
mean ± sample SD across the two trial means, in **ns/board** (with N=2 the mean and median agree).

| Kernel | Previous index | Packed rules |
|---|---:|---:|
| Native encode | 929 ± 28 | **66.7 ± 1.8** |
| Native decode | 1,102 ± 7 | **98.0 ± 0.5** |
| Row encode | 1,424 ± 712 | **80.8 ± 7.7** |
| Row decode | 1,092 ± 4 | **99.4 ± 0.3** |

Native encoding reduced time by **92.8%** and decoding by **91.1%** on these fixtures. The first
baseline row-encoding result was noisy (1,927 ns versus 920 ns on its second run); do not use
its 94.3% median reduction as a precise general claim. An unrelated `cargo test --workspace`
was observed around the start of this series and had exited by its end. Our builds had finished.
Power/load/process metadata are retained, and the earlier before/generic series remains as
additional evidence rather than being pooled with this final comparison.

A separate final Criterion confirmation compared against `before-d` with `--baseline before-d`.
All four kernels were reported improved at p<0.05 (Criterion printed p=0.00). Time-change intervals
were native encode **[-92.816%, -92.374%]**, native decode **[-90.974%, -90.629%]**, row encode
**[-92.271%, -91.513%]**, and row decode **[-91.261%, -90.567%]**. These are the within-run
Criterion intervals; they do not eliminate host/core-placement variability. See
`final-significance.log` and the preserved Criterion samples.

### Traversal ablation

The ignored release test compares the same packed nodes and 256 seeded boards, with or without
per-node levels. Eight ABBA-ordered trials perform 1,048,576 calls each; all rank results and
checksums agree. Generic traversal averaged **84.91 ns/call** (median 84.92, SD 0.23, N=4),
versus **55.25 ns/call** (median 55.22, SD 0.36, N=4) when the residue invariant removes level
bookkeeping: **34.9% less traversal time**. This isolates that optimization; do not add its
percentage to the whole-kernel reductions. Packing separately changes default record storage
from 318,264 to 106,088 bytes. `traversal-ablation.log` retains all trials.

### Bounded engine and codec replay

One release executable contains both implementations and the same current engine. It replays
256 fixed legal-history parents with bag-allowed actions, resetting the board before every
placement. Preparation and allocation are outside timing. Every resulting board's admission and
successful native `board → u64 → board` round trip are checked before timing; timed kernels hash
raw decoded columns with a rolling addition/rotation checksum, so different ID order cannot
change the expected output. Eight ABBA-ordered trials perform 128 replays each.

Every trial produced **2,735,744 placements**, **532,608 admitted round trips**,
and **10,880 cleared lines**, with identical nonzero checksum `6148914691236517204`.
Previous-index wall time averaged **1.2977 s** (median 1.2886, SD 0.0218, N=4),
versus **0.1905 s** (median 0.1823, SD 0.0167, N=4) for the rule codec.
Median throughput was **2.12 versus 15.00 million placements/s**:
**85.9% less wall time (7.07× throughput)** for this bounded workload.
This includes engine placement, filtering, codec errors/successes, decoding, and checksum work.
It measures neither arbitrary board distributions nor long-term survival.

`placement/src/main.rs`, `placement/fixtures.rs`, the fixture provenance, exact saved executable,
`placement-replay.jsonl`, `/usr/bin/time -l` output, and environment/SHA metadata are retained.
The earlier XOR-checksum driver is preserved separately but is not used for these final results.
No build or profiler from this experiment overlapped this final replay. Process/power/load
snapshots permit inspection of other host activity. The same process retains both codecs, so
its peak RSS is not an isolated comparison of codec storage. No allocation/counter profiler or
whole-program code-size attribution was performed.


## Checks

Passed:

- `cargo test -p tetris-codec`: ten unit tests, one ignored release ablation; three doctests.
- `cargo test -p tetris-game`: 33 unit tests; 32 doctests, one ignored doctest.
- `cargo test -p tetris-playground --bin tetris_codec_reachability`: 13 tests.
- Focused Clippy with `-D warnings` for codec all targets, game all targets, sampler binary,
  and benchmark library. Const-evaluation progress diagnostics can still be emitted by rustc.
- `cargo fmt --all -- --check` and `git diff --check`.

Tests cover exhaustive small carriers, odd/even widths, independent cell/engine admission,
wide boards exceeding 64 cells, native and row inverses, exact mirror distinction, invalid high
bits/IDs, packed-storage boundaries, the 216 known impossible examples, and seeded legal play.
Tests live in their owning modules; the expensive timing ablation is ignored by default.
The full workspace/backend suite and optional six-hole benchmark were not run for this unmerged
experiment. The complete default sweep and focused checks are recorded above.

## Correctness audit follow-up (2026-11-01)

A separate audit verified every one of the 393,685,466 original default candidates, all
392,431,346 committed-codec native round trips, and all 380,842,218 prototype round trips.
It independently replayed the 1,254,120 original exclusions and 11,589,128 additional prototype
exclusions with the engine, and retained all 1,179 saved witness paths (4,632 in-carrier prefixes).

Independent forward search also exposed two domain errors that reference-carrier comparisons
had not caught: three-column clears above the cap can violate the top-support condition, and
31-row engine spawn overlap can invalidate the occupancy residue. The codec now applies top
support only at widths >=4 and supports history caps through 28. Regression tests retain legal
seven-bag sequences for both cases, plus exhaustive forward exploration of the three-column,
two-row board/bag graph. These corrections leave default ten-column IDs and lookup data unchanged.

After both corrections, workspace tests passed (412 unit/doctest cases; three ignored), workspace
all-target Clippy passed with `-D warnings`, and the 65 RocksDB partitioned tests passed.
Eight bounded board/bag reachability graphs and five complete small raw-board universes were
also checked independently. A preserved-binary placement replay reproduced 7.11x median
throughput with identical outcomes. This confirms the earlier bounded workload result; it is
not a general search/survival speed claim or a production benchmark of the stronger prototype.

Full commands, source/binary fingerprints, failures found, scopes, and logs are under
`artifacts/output/codec-validation-20261101/`. The stronger predecessor-elimination rule remains
research only, with the same 380,842,218-state result; it has not replaced the compiled codec.
