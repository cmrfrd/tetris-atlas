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
Compile-time construction uses the engine's const piece masks. Runtime ranking and selection
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
of process RSS. Small metadata headers are excluded. Each codec value is one pointer; lookups
allocate no heap storage and need only scalar traversal state plus their input/output boards.
There is no per-query export buffer or cache initialization.

Narrow child/count fields are selected losslessly. Even-width occupancy residue means no live
variable can be skipped, so those configurations omit levels and their traversal bookkeeping.
Odd widths keep levels and handle skipped variables explicitly. Native boards of at most 64 cells
use a packed bitstream; wider boards use the same language through column-bit access.

Construction is deliberately bounded and runs in const evaluation. Fresh default consumer builds
in this experiment took about 2–3 minutes; separate targets can each evaluate the diagram.
Compile-time memory was not measured. Large hole budgets can be substantially more expensive;
the existing six-hole benchmark is opt-in with `large-codec-benchmarks`. Counts must fit `u64`,
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
