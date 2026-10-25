# AGENTS.md — Tetris Atlas

This is the shared contributor and agent guide. `CLAUDE.md` points here; keep one source of truth.

## Mission and proof standard

Build an **Atlas**: a lookup policy that proves infinite survival under this repository's canonical
Tetris engine, for every legal 7-bag piece sequence. The ideal Atlas covers all reachable states;
a validated policy-closed subgraph containing the empty-board seed is sufficient to prove survival
from a new game. Finite survival, heuristic scores, and partial exploration are evidence, not proof.

Milestones: **M0** strong finite play; **M1** no observed deaths (empirical); **M2** a certified
closed policy set; **M3** that set reachable from the empty board; **M4** the global Atlas or a
sufficient closed policy graph from the seed.

The online adversarial quantifier order is **for every possible next piece, there exists a move**.
For a state `(board, remaining_bag)`, every remaining piece is an AND obligation and its placements
are OR alternatives. After the last piece, refill the bag with all seven pieces. A state is dead
when any possible piece has no surviving move. Unknown/unexpanded states are not certified live.
A lookup can include the observed piece in its key or store one move per possible piece.

A cycle or strongly connected component alone is insufficient: certify that every possible draw
has a selected legal move whose successor stays in the winning set. Recompute transitions with
the engine, validate closure independently, and verify reachability from the seed. A fixed piece
order or a script chosen after seeing an entire bag proves only that restricted/lookahead model.
Tighter height/hole carriers restrict the search; a closed winning policy within one still proves
survival under the canonical game. Failure within a carrier does not prove the full game losing.

## Ground truth and ownership

`crates/tetris-game` is authoritative. `StandardTetris` is 10 columns × 20 playable rows, with a
column-major `[u32; 10]` board, seven standard tetrominoes (O, I, S, Z, T, L, J), and deterministic
seeded 7-bag draws. Actions are precomputed rotation/column hard-drop placements, not timed input
sequences or a separate rotation/kick system. Derive action counts from
`TetrisPiecePlacement::NUM_PLACEMENTS`; do not hardcode a stale count. Placement drops and merges
the piece, clears complete rows, then checks loss (`height > ROWS`); a lost result is normalized
to the engine's terminal board marker. All solvers, learned models, and verifiers must agree.

| Location | Responsibility |
|---|---|
| `crates/tetris-game` | Board, configuration, pieces, bags, placement, line clearing |
| `crates/tetris-utils` | Bit operations, fixed-capacity containers, shared low-level utilities |
| `crates/proc-macros` | Conditional inlining |
| `crates/tetris-search` | Beam/multi-beam search, sequence planner, parallel state exploration |
| `crates/tetris-ml` | Candle models, optimizers, tensors, data generation, checkpoints, TUI |
| `crates/tetris-playground` | Explicit Cargo binaries: examples, atlas, research, training, tools |
| `crates/tetris-benches` | Criterion benchmarks and pprof profiling |
| `crates/tensorboard` | TensorBoard event writer and checked-in generated protobuf bindings |
| `scripts/`, `.docker/`, `.devcontainer/` | Utilities and development/container configuration |

Read the relevant source and manifests before editing. Binary paths and required features live
in `crates/tetris-playground/Cargo.toml`; detailed research descriptions live beside their source.
Do not assume old `docs/`, proof sources, archived experiments, or runtime artifacts still exist.
Do not revive removed files or branches incidentally. Treat research-status statements as historical
unless supported by the current run artifacts.

## Implementation workflow

- Propose a minimal change set, preserve unrelated work, and add or adjust meaningful tests.
- Use the pinned nightly in `rust-toolchain.toml`; unstable features include `generic_const_exprs`.
  Follow `rustfmt.toml` (edition 2024, width 100, four spaces, Unix newlines).
- Prefer `anyhow::Result` for fallible operations. Workspace Clippy denies `panic!`, `todo!`,
  `unimplemented!`, and `unreachable!`. Other lint settings, including `unwrap`/`expect`, come from
  workspace `Cargo.toml`; do not weaken them to hide new failures.
- Preserve const-generic, bitwise, fixed-capacity, and unrolled hot paths where justified.
  Large fixed-capacity planners must not be materialized on thread stacks; audit unsafe allocation
  assumptions against the actual field layouts and test equivalence to ordinary initialization.
- Algorithm changes need fixed-seed evaluations, ablations, and a rollback plan. Do not claim a
  search or performance improvement from code inspection alone.
- Keep persisted state encodings, policy semantics, and resume guards consistent. Schema changes
  require explicit compatibility handling or a fresh database. Never mistake a drained work queue,
  optimistic liveness, Bloom-filter hint, or progress ratio for a closure certificate.

Run these before merging, plus checks for the features/targets touched:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
# Focused checks:
cargo test -p tetris-game
cargo test -p tetris-search
cargo test -p tetris-playground --features rocksdb-backend --bin tetris_atlas_partitioned
```

Use fixed seeds in tests; keep unit tests fast and put expensive randomized/research sweeps behind
`#[ignore]`. Default workspace checks skip feature-gated binaries. Report pre-existing failures
separately from regressions, and state exactly what passed, failed, or was not run.

## Features and representative commands

Playground defaults enable no optional backends. Relevant flags:
`ml` (training/CLI), `tui` (interactive atlas), `rocksdb-backend`, `sqlite-backend`, and `smt`.
`candle-metal`/`candle-cuda` select GPU backends; include `ml` for playground training targets.
Choose exactly one explicit `dtype-f16`/`dtype-bf16`/`dtype-f32`/`dtype-f64` override when needed.
`never-inline` supports ASM analysis. Check native backend prerequisites on the target platform;
do not blindly enable every GPU/backend feature together.

```sh
cargo run --release -p tetris-playground --bin tetris_demo_single
cargo run --release -p tetris-playground --bin tetris_demo_multi
cargo run --release -p tetris-playground --bin tetris_demo_adaptive_multi

# Create output directories before runs; use --help to inspect current options.
mkdir -p artifacts/databases/inmemory artifacts/databases/rocksdb
cargo run --release -p tetris-playground --features tui \
  --bin tetris_atlas_inmemory -- --path artifacts/databases/inmemory create
cargo run --release -p tetris-playground --features rocksdb-backend,tui \
  --bin tetris_atlas_rocksdb -- --db-path artifacts/databases/rocksdb create
cargo run --release -p tetris-playground --features rocksdb-backend \
  --bin tetris_atlas_partitioned -- --db-path artifacts/databases/partitioned \
  create --writers 4 --ram-mib 4096

# Metal on macOS; substitute candle-cuda on a configured CUDA host.
RAYON_STACK_SIZE_MB=64 cargo run --release -p tetris-playground --features ml,candle-metal \
  --bin tetris_train_beam_supervised -- train --run-name my-run \
  --logdir artifacts/logdir --checkpoint-dir artifacts/checkpoints
cargo run --release -p tetris-playground --features ml --bin tetris_train_genetic
```

For training inference, pass a checkpoint **directory** via `inference --checkpoint <dir>`.
Training resume accepts a checkpoint ULID via `train ... --resume <ULID>`.
Search's global Rayon pool accepts `RAYON_NUM_THREADS` and `RAYON_STACK_SIZE_MB` (default 64 MiB
stack); individual binaries may create separate pools. Verify container feature flags and output
paths against current manifests before running `.docker/` builds.

## Evaluation and artifacts

Write new runtime outputs under gitignored `artifacts/`, creating directories as needed:
`checkpoints/<run>/`, `logdir/<run>/`, `databases/`, `data/`, `output/`, and `benchmarks/<sha>/`.
Do not commit generated databases, models, logs, or Criterion reports unless explicitly requested.
Some utility scripts and archives are ignored by current rules; check `git status`/`.gitignore`
when adding reproducibility tooling rather than assuming a file is tracked.

Every result needs a run specification, parameters, logs, seed set, and commit hash (with any dirty
worktree noted). Record N, mean/median and dispersion, wall time, throughput, hardware/OS, and
replay inputs (seed + bag state and action sequence or policy/model). Track pieces, lines, score,
height, holes, and mean/p95 move latency. Atlas results also need state/frontier counts, expansion
rate, cycle/closed-set size, and coverage with its denominator; preserve unknown coverage as unknown.
Keep historical measurements distinct from current verification.

## Benchmarks and profiling

Use `crates/tetris-benches` with fixed seeds/fixtures and unchanged benchmark names. Avoid unbounded
demos as benchmarks. For performance-sensitive changes, save a named baseline **before** editing:

```sh
cargo bench -p tetris-benches --bench benches -- --save-baseline before-change
# After the change, same hardware/toolchain/features and benchmark configuration:
cargo bench -p tetris-benches --bench benches -- --baseline before-change
# Profiling mode (separate from statistical timing):
cargo bench -p tetris-benches --bench benches -- --profile-time 5
```

For historical comparisons, save a baseline named for the ref/commit. Switch refs only with a clean
worktree or an explicitly authorized workflow; otherwise use a same-checkout comparison. Keep
compiler, feature flags, fixture data, sample settings, and load comparable. Explain any changed
benchmark definitions or toolchain; do not attribute those effects solely to the code change.

Active reports are in `target/criterion/` (`report/index.html`). For each prepared commit, try to
snapshot that directory under `artifacts/benchmarks/<commit-sha>/`, alongside `commit.txt`,
`git-status.txt`, `rustc -Vv`, hardware/OS, exact commands, and benchmark logs. Restore a saved
Criterion directory to compare on a fresh checkout. If benchmarks are skipped, explain why.
Report baseline/candidate SHAs (or dirty state), Criterion percentage deltas and significance,
plus noise sources such as background load or thermals. A faster heuristic still needs survival
and correctness evaluation.

Use `cargo-show-asm` (`cargo asm`) rather than the unmaintained `cargo-asm`. The low-level
`rshift_slice_from_mask_u32` function now lives in `tetris-utils`; use that crate for ASM or
`cargo expand` inspection, with `never-inline` when needed.

### macOS tools and measurement discipline

| Question | Tool |
|---|---|
| Did a component get faster? | Criterion with named baselines and fixed fixtures |
| Did real work get faster? | Bounded release workloads, alternating A/B or ABBA, identical outputs |
| Which threads, functions, and inlined lines are hot? | [Samply](https://github.com/mstange/samply), with its call tree and timeline |
| Is time going into CPU work, scheduling, or allocation? | Xcode Instruments: Time Profiler, System Trace, Allocations |
| Did instruction/cycle cost change? | Instruments CPU Counters, where supported; normalize by pieces or expansions |
| What are wall time and peak memory? | `/usr/bin/time -l` on a bounded release command |
| Where is a running process stuck? | `/usr/bin/sample PID 10 -file <output>`; `vmmap -summary PID` for memory regions |
| Did generated code change? | `cargo-show-asm`; `atos` for sampled addresses with matching symbols/load address |

Build first, then measure the executable directly; do not time compilation. Release and bench
profiles retain debug info in `Cargo.toml`, enabling symbolized stacks. Preserve the measured binary
and symbols. For example, use the benchmark executable path printed by Cargo (the hash changes):

```sh
cargo bench -p tetris-benches --bench benches --no-run
mkdir -p artifacts/output/profiles
ATLAS_BENCH_BIN="target/release/deps/benches-<hash>"
samply record --save-only -o artifacts/output/profiles/scorer.json.gz -- \
  "$ATLAS_BENCH_BIN" --bench height_mse_board_score \
  --sample-size 10 --warm-up-time 1 --measurement-time 10
samply load artifacts/output/profiles/scorer.json.gz

# Full Xcode is required; list templates before selecting one.
xcrun xctrace list templates
xcrun xctrace record --template 'Time Profiler' --time-limit 15s \
  --output artifacts/output/profiles/scorer.trace --launch -- \
  "$ATLAS_BENCH_BIN" --bench height_mse_board_score \
  --sample-size 10 --warm-up-time 1 --measurement-time 10
```

These are profiling runs: their timings include instrumentation and are not Criterion evidence
of a speedup. Run Samply, Instruments, and the existing pprof `--profile-time` mode separately.
Use a fixed-size production workload for end-to-end confirmation; validate outcomes as well as
time. The related `tetris-optim-rs` project's counter/A-B helper scripts are not installed here.

Before timing, check `uptime`/`top`, `pmset -g batt`, and `pmset -g`; use consistent power and
Low Power Mode settings. Avoid simultaneous builds, IDE indexing, training, other agents' workloads,
or profilers; allow the machine to cool after a large build. Record background load and thermal
limitations instead of attributing every change to the code. Apple Silicon performance/efficiency
core placement matters; counters do not remove scheduling, thermal, or core-mix effects.

If unchanged components shift together, repeat the **old executable** under current conditions
and compare old-before, old-now, and new. Confirm suspicious changes with alternating A/B or ABBA
runs and report medians, dispersion, and matching outputs. Keep native arm64 measurements separate
from Rosetta/x86-64 and other machines. Record tool versions and keep profiles/traces under
`artifacts/`; report unavailable counters explicitly. Do not clear baselines during a comparison.
