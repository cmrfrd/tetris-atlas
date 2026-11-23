# BoardSet performance and merge review

> Archived November 23, 2026. Active development is closed; see the
> [README archive summary](README.md)
> for the preserved branch, scope, validation and resumption steps. The results
> below are historical and were not rerun for the archival push.

The selected implementation combines compact hash-directory lookup, an immutable
compressed arena, adaptive width metadata, half-byte payloads, short headers for
regular widths, and on-demand sorted iteration. The choice follows the project's
memory priority. It does not claim to beat `HashSet` on every operation.

## Further cleanup (November 11)

Documentation now clarifies encoder stages and literal-selector limits, and a
focused test checks byte-header choices at one, two and three exceptions.
Splitting `append` into helpers was tried with and without explicit inlining.
Alternating old/new process checks still measured an 8.3% median slowdown for
four-column lookups (four processes per arm), so both extractions were rejected.
The production implementation remains unchanged. Measurements and rejected source
snapshots are retained under `artifacts/cleanup-20261111/`.

## Human readability review

Five November 10 review passes separated the arena codec and its inline tests
from the public set API, named format selectors and directory bit fields, added
exhaustive width-boundary and clear/refill tests, and documented the implementation
map. `lib.rs` decreased from 2,115 to 788 lines; total source grew with comments
and tests. The simple public API, encoded records and capacity policy are unchanged.
Fixed-seed fixture fingerprints now catch accidental benchmark input changes.

Every checkpoint was checked against a frozen baseline across 12 operation cases
and eight generic-width lookup cases. A final repeat and repeated old-executable
controls found no material regression in those selected workloads. One checkpoint
growth result of +8.6% did not repeat with the identical executable (-1.8%). The
old executable itself shifted by up to about 5%, so small differences are not
presented as causal speedups. The linked benchmark executable's text section grew
by 20,312 bytes (about 0.6%); this is a whole-suite code-size measurement, not the
crate's isolated instruction footprint.

All eight allocation-counter cells matched the prior portable implementation for
retained requested bytes, build peak, iteration scratch and clear retention.
Formatting, strict workspace Clippy, workspace tests, the release membership
sweep, private rustdocs and benchmark smoke checks passed. At that checkpoint the
crate had 33 unit tests, four normal integration tests and four doctests. Full source
checkpoints, timing intervals, unchanged controls and validation logs are retained
under `artifacts/human-review-20261110/`. These checks do not replace the broader
main-integration gate below; no rebase or merge was part of the review.

## Portable arena initialization

The selected arena now initializes its full `Vec<u8>` capacity only when growing,
retains initialized bytes after `clear()`, and tracks logical records separately.
Appends overwrite record bytes without resizing or zeroing each record. This is
safe Rust with no new fields, dependencies, public methods, or platform backend.

The November 9 rerun compared the exact selected source against the preceding
per-record initialization using frozen executables, fixed fixtures, and balanced
orders: six processes per arm per cell, three fresh rounds per process, twelve
fixture/capacity cells (216 timing processes across three arms). On native M1 Pro,
pinned nightly-2025-11-23, median insertion time fell 5.2–13.1% with reserved
capacity and 2.3–6.8% when growing. Ten of twelve exploratory paired bootstrap
intervals excluded zero; random growing and shared-prefix reserved remained
uncertain. Successful lookup medians mostly stayed within 3%. Duplicate insertion
on the low-board 100k reserved fixture regressed about 3%; this is an insertion
and reuse tradeoff, not an across-operation speedup.

Requested retained heap matched exactly in all eight memory cases, and the object
remains 80 bytes. Initializing headroom earlier can affect resident pages;
requested-heap equality does not guarantee equal RSS. Iteration scratch is
unchanged and remains separate from retained storage. The mapped arena prototype
offered no compelling overall advantage and is not included.

Unchanged standard `HashSet` insertion controls moved -1.3% and +1.0%; its
clear/reuse controls varied more. The host stayed on AC, but intermittent builds
required waiting and rejecting observed overlapping blocks. The machine was not
isolated, intervals are exploratory and not adjusted for multiple comparisons,
and these are collection measurements, not end-to-end Atlas throughput.

The complete report, raw rounds, paired intervals, memory probes, fixture/binary
hashes, and environment logs remain under
`artifacts/allocator-quiet-20261109/`. These results compare initialization
strategies within the current codec; the historical codec comparison and broader
merge gate below remain distinct.

November 9 promotion validation passed formatting, strict workspace all-targets Clippy,
workspace tests, 29 crate unit tests, four normal integration tests, four doctests,
and the ignored release membership sweep. The promoted implementation matches
the measured portable source byte-for-byte. Logs and the rollback snapshot are
under `artifacts/portable-promotion-20261109/`. Optional database/GPU feature checks
were not rerun because those backends are unchanged. No merge or push was made.

## Historical evidence

These measurements predate the November rebases. They compare the selected
small-record codec with the preceding width-codec candidate, on Apple M1 Pro,
macOS 26.6.2, pinned nightly-2025-11-23, with dirty source snapshots identified by
hashes in the archived raw artifacts. They are not fresh measurements of main
`e33a40a` or of this consolidated checkout.

Requested retained heap, reserved populations; MiB means 2^20 bytes. Shared input
buffers, allocator bookkeeping, internal reallocation overlap and physical RSS
are excluded. The set object was 80 bytes, separate from retained heap.

| Fixture | Distinct boards | Previous MiB | Selected MiB | Additional saving | Std HashSet MiB |
|---|---:|---:|---:|---:|---:|
| shallow1m | 1,000,000 | 11.534 | 10.305 | 10.66% | 82.000 |
| deep1m | 981,666 | 24.208 | 22.534 | 6.92% | 82.000 |
| fresh-low1m | 989,846 | 17.571 | 15.815 | 10.00% | 82.000 |
| fresh-high1m | 926,797 | 23.451 | 22.419 | 4.40% | 82.000 |
| random1m | 1,000,000 | 45.022 | 43.826 | 2.66% | 82.000 |
| shared64k | 65,536 | 1.361 | 1.306 | 3.98% | 5.125 |

On these fixed fixtures the selected set uses about 47–87% less requested retained
heap than the standard hash set. This is distribution-dependent, not a guaranteed
compression ratio. Growing populations saved another 2.57–9.46% relative to the
preceding codec; some capacity-boundary cases saved only 1.24–1.30%.

The additional compression increases CPU work. Reserved insertion medians were
22–52% slower and successful lookups approximately 1–38% slower than the preceding
codec. A smaller Criterion engine fixture regressed insertion by about 63%.
Sorted export also costs temporary memory and decoding/sorting time: the reserved
million-board cases used about 2.8–3.8 MiB of iterator scratch on top of retained
heap. Multiple live iterators have independent scratch allocations.

Bounded confirmation used A/B/B/A/B/A/A/B ordering, four processes per arm and two
fresh mutation rounds per process. Raw outputs preserve means, medians, dispersion,
wall times, fixture hashes, cardinality/fingerprint oracles, environment data,
frozen executables and sources. Criterion controls used 20 samples, 200 ms warmup,
one-second measurement and old/new/new/old processes. Profiles ran separately.
These collection timings do not establish solver throughput or survival.

Rejected experiments, including directory growth changes, alternate probes,
shared width-selection scans and forced decoder inlining, are preserved in the
archive. They are not additional production backends or public tuning switches.

## Consolidation and merge gate

Operation benchmarks use the workspace runner described in README.md. Historical
research reports and raw runtime outputs remain outside the review diff.

Before merging, run new named baselines on the consolidated main/toolchain/engine
fixtures for this set, the preceding codec recovered from the archive, standard
`HashSet`, and the fast hash control. Preserve identical fixtures and reset each
mutation batch; separate allocation counting and profiling from timing. Reconfirm
retained/build peak/export scratch at representative populations and directory
boundaries. The memory benefit must justify the measured insertion/lookup cost
for the intended Atlas workload. No fresh timing claim is made by consolidation.

The recovery archive is currently located at
`../tetris-atlas/artifacts/board-set-consolidation-20261109/` relative to the retained
worktree. Its README describes the verified Git bundle, complete source snapshots,
patches, raw results and how to restore an experiment.

## Historical consolidation validation

On the consolidated `feat/board-set` checkout based on main `e33a40a`, formatting,
strict workspace all-targets Clippy, workspace tests, the 640,000-operation ignored
release sweep, and the integrated `board_set_` Criterion smoke cases passed.
Smoke mode verifies registration and correctness; it provides no statistical
performance comparison. Optional backend feature checks were not run because
those backends were unchanged. No merge or push was performed.
