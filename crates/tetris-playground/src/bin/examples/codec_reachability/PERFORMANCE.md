# Reachability sampler implementation and optimization

Measured 2026-10-30 on Apple M1 Pro, 32 GiB, native arm64 macOS 26.6.2,
`nightly-2025-11-23` (rustc 1.93.0-nightly, LLVM 21.1.5), release profile.
Both implementations are uncommitted on `feat/board-codec`, based on
`8297faee182b6d6ee53a2d853d620c7bff8b73dc`. No production codec implementation changed in this pass.

The first working serial implementation was preserved before optimization. Raw artifacts are in
`artifacts/output/codec-playground-tool/`: exact binaries and Rust source snapshots in `before/`
and `after/`, commands/environment in each trial JSON, `/usr/bin/time -l` logs, `measure.py`,
`statistics.json`, `validate.py`, validation results, and separate Samply profiles. These are
gitignored local artifacts. Source snapshots and executable hashes identify dirty variants:

- Before SHA-256: `1dfa9c1cf27c9874e1e8f08d9f79e6d6573f315134215dc019553621e3b6b4a9`.
- After SHA-256: `437a808bdf757f514d792e8b34ff5878553e9736c4eb08592517e726786de333`.

## Workload and results

Each trial starts a fresh dataset: 20 batches, 10,000 parents, 1,000 episodes per play mode,
10,000 candidates, 10,000 watched targets per batch, horizon 128, witness budget four per mode,
seed `0xb04dc0de`, controls disabled, periodic checkpoints disabled, quiet output. A final durable
checkpoint and watched-target export are included. Build time is excluded. Example invocation:

```sh
/usr/bin/time -l artifacts/output/codec-playground-tool/after/tool run \
  --output artifacts/output/codec-playground-tool/fresh-trial --batches 20 \
  --parents 10000 --episodes 1000 --candidates 10000 --watch 10000 \
  --witnesses 4 --checkpoint-seconds 0 --quiet --threads 4
```

External monotonic wall seconds, including process startup and final checkpoint:

| Implementation | N | Mean ± sample SD | Median | Range | Maximum RSS |
|---|---:|---:|---:|---:|---:|
| Serial baseline | 4 | 4.481 ± 0.044 | 4.463 | 4.451–4.547 | 196.1 MiB |
| Optimized, one worker | 3 | 4.613 ± 0.217 | 4.493 | 4.482–4.862 | 196.3 MiB |
| Optimized, two workers | 1 | 2.769 | 2.769 | single exploratory run | 196.2 MiB |
| Optimized, four workers | 5 | 1.899 ± 0.009 | 1.896 | 1.892–1.915 | 196.4 MiB |
| Optimized, eight workers | 4 | 1.455 ± 0.012 | 1.450 | 1.447–1.472 | 196.6 MiB |

Four-worker median wall time is **57.5% lower (2.35× throughput)** than baseline; eight-worker
time is **67.5% lower (3.08× throughput)**. These are descriptive bounded-workload comparisons,
not Criterion significance estimates. One-worker performance did not improve; the benefit is
parallel execution with a small memory increase and more CPU resources. Four workers remain the
automatic cap to limit contention; `--threads 8` was faster on this machine.

Every trial performs 32.4 million parent placements, 200,000 candidate checks, 20,000 strict and
20,000 relaxed episodes. Four workers achieve 17.1 million parent placements per wall second;
eight achieve 22.3 million. These rates include play, candidate checking, and persistence, rather
than measuring only the placement kernel. The example four-worker run used 4.49 user seconds
despite 1.89 seconds elapsed; this is a latency/throughput improvement, not reduced total CPU work.

Machine was on battery at approximately 27%, Low Power Mode off, observed load averages roughly
3.6–4.3. Runs were sequential with no simultaneous builds or profilers. Baseline runs were repeated
among optimized trials; individual commands and power snapshots are retained. Core scheduling,
background load, battery/thermal conditions, and filesystem cache remain limitations. No claim is
made for another machine, every codec configuration, or long-run saturated behavior. Checkpoints
and the journal grow independently of a fixed trial; disk cost matters in extended runs.

## What changed and why

The initial Samply profile (separate from timings) contained 8,350 codec-stack samples. Encoding
accounted for about 62% self samples, decoding 14%, and reference predecessor checking only 0.6%.
That directed optimization toward parent expansion:

- Generate the identical parent-ID stream before parallel execution, then expand chunks of 64
  parents with Rayon. Share four atomic bitmaps and reduce counters locally. No per-worker
  universe copies or concurrent journal writes.
- Group placement lists by piece for legal-play episodes, avoiding repeated filtering of all
  placements at every move. Action order and RNG decisions remain identical.
- Retain the straightforward engine-based predecessor checker. A speculative packed-bit proof
  implementation would add correctness risk to a small part of observed runtime.

Thread scaling above is the parallelism ablation. Placement grouping was not independently timed,
so no speedup is attributed to it separately. There was no assembly-level codec change, instruction
count improvement claim, or modification to the encoding contract.

## Correctness and discovery results

All timed runs have identical logical checkpoint headers, bitmap payloads, findings journals,
watched-board CSVs, and totals relative to the serial baseline. All optimized checkpoints are also
byte-identical across worker counts 1, 2, 4, and 8. Preserved output includes:

- 2,001,465 admitted incoming transitions, reaching 1,993,635 distinct IDs.
- Strict: 114,040 pieces, 4,521 lines, 77,667 distinct IDs.
- Relaxed: 144,824 pieces, 6,680 lines, 21,141 distinct IDs; 21,546 excursions and 1,822 returns.
- 660 distinct impossibility findings from uniform candidates, and 166 saved legal witnesses.

The optimized verifier replayed the baseline's 660 findings and 166 witnesses. Separate bounded
functional runs gave byte-identical checkpoints/journals/CSVs for uninterrupted execution,
resume, and forced-kill recovery. The uncommitted journal tail was preserved before deterministic
replay. Ctrl-C shutdown and subsequent resume succeeded. Trying to resume the baseline dataset
with changed sampler code was refused without changing its checkpoint or journal.

A separate optimized eight-worker Samply run completed 145 batches and produced 4,642 impossibility
findings and 1,179 witnesses, all successfully reverified afterward. Its instrumented elapsed time
is excluded from the timing table. Raw samples and `atos` symbol resolution against the preserved
binary remain in `after/`; encoding and decoding still occupy the prominent hot addresses.

Twelve inline unit tests pass, covering raw-board proof/witness checks, bag legality, legal sampled
paths, watched-target witness retention, concurrent exact bitmap marking, checkpoint corruption,
source/workload mismatch, journal recovery, locking, and deterministic RNG. Focused strict Clippy
also passes. Sampling absence is never promoted to impossibility, and no finding automatically
removes a codec state.

Final checks also passed `cargo fmt --all -- --check`, `git diff --check`, and `cargo test -p
tetris-codec` (25 tests, three doctests, one ignored research sweep). The compiler emits the
existing long-running const-evaluation progress warnings. Full workspace tests/Clippy and optional
backend checks were not run in this example-only pass; nothing was merged.
