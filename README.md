# Tetris Atlas

A research project aiming to **solve Tetris** — including techniques such as agents that play at superhuman level for arbitrarily long horizons through search, learning, and state-space exploration.

## Quick start / Demos

Runnable binaries are in `tetris-playground`. Write checkpoints, datasets, databases, and other
runtime output under the gitignored `artifacts/` directory.

```sh
# Run tests
cargo test --workspace

# Watch a beam search agent play
cargo run --release -p tetris-playground --bin tetris_demo_single

# Multi-beam voting
cargo run --release -p tetris-playground --bin tetris_demo_multi
```

## Crates

```
crates/
├── tetris-game        Core engine: board, pieces, bag, placement, line clearing
├── tetris-utils       Bit operations, loop unrolling, and fixed-capacity containers
├── tetris-search      Beam/multi-beam search, sequence planning, parallel exploration
├── tetris-ml          Candle-based ML: NN modules, optimizers, checkpointing
├── tetris-playground  All runnable binaries (demos, training, atlas builders, experiments)
├── tetris-benches     Criterion + pprof benchmarks
├── tensorboard        TensorBoard event-file writer
└── tetris-macros        Read-only field accessors and conditional inlining
```

## Training

Training binaries live in `tetris-playground` and use `tetris-ml` with Candle.
Enable `ml`; add `candle-metal` on a configured macOS host or `candle-cuda` for CUDA.

```sh
# Beam-supervised learning (learns from beam search trajectories)
RAYON_STACK_SIZE_MB=64 cargo run --release -p tetris-playground \
  --features ml,candle-metal \
  --bin tetris_train_beam_supervised -- train \
  --run-name my-run \
  --logdir ./artifacts/logdir \
  --checkpoint-dir ./artifacts/checkpoints

# Resume an existing run: replace <run-ulid> with the ULID printed by that run.
cargo run --release -p tetris-playground --features ml,candle-metal \
  --bin tetris_train_beam_supervised -- train \
  --run-name my-run --logdir ./artifacts/logdir --checkpoint-dir ./artifacts/checkpoints \
  --resume "<run-ulid>"

# Run inference
cargo run --release -p tetris-playground --features ml,candle-metal \
  --bin tetris_train_beam_supervised -- inference \
  --checkpoint "./artifacts/checkpoints/my-run/<run-ulid>"
```

Checkpoints and logs are grouped as `<base>/<run-name>/<run-ulid>/`. Inference accepts that
checkpoint directory, including its policy configuration. For CPU training, use `--features ml`.

## Benchmarking

```sh
# Run all benchmarks
cargo bench -p tetris-benches

# Check all benchmark cases without collecting timings
cargo bench -p tetris-benches -- --test

# Run benchmarks whose names contain bag
cargo bench -p tetris-benches -- bag

# Generate flamegraphs (5 second profiling)
cargo bench -p tetris-benches -- engine/apply_fixed --profile-time 5
```

Declare benchmark callbacks with `tetris_bench!` to register them automatically. New files also
need a `mod` declaration in `src/benches/mod.rs`. Unused private functions fail compilation.

In `crates/tetris-benches`, suites, their local helpers/tests, and shared `fixtures.rs` live
in `src/benches/` with ordinary Rust modules. `src/registry.rs` owns registration;
`src/criterion.rs` configures Criterion and profiling. Workspace tests discover the suite tests
through `lib.rs`, and Criterion calls the same library to run every registered benchmark.

Open the overall HTML report with `open target/criterion/report/index.html`.

Criterion HTML reports are written to `target/criterion/`. Throughput counts input operations:
placed pieces for game replays, searches for search cases, parent states for expansion, and candidate
attempts for heaps. Kernel outputs are preallocated; input copies remain included where required.
`engine/copy_reset_fixed` measures restoring fixed inputs and clearing them using reused storage.
It replaces the allocated-batch `engine/reset_fixed`; their absolute timings are not comparable.
`piece/sample_fixed` and `placement/sample_fixed` measure random sampling with fixed-seed
`SmallRng` and `StdRng`; RNG initialization is outside timing. Seven-bag draws have separate bag benchmarks.
Placement boundary cases cover zero through four clears, near-top survival, topout, and clearing before loss detection. Renamed workloads require
fresh baselines.

For timing comparisons, use matching settings and named baselines on an otherwise idle machine:

```sh
cargo bench -p tetris-benches -- --save-baseline before-change \
  --sample-size 50 --warm-up-time 2 --measurement-time 5
# After the change, use the same command with --baseline before-change.
```

Repeat suspicious results and inspect dispersion. Compare only unchanged workloads; renamed or
repaired definitions need new baselines. Profiling and smoke runs are separate from timing evidence.

The search suite includes the demo's width-256/depth-8 beam configuration. The gameplay case
runs seeds 1–4 for 16 moves each, including search, applying the chosen move, line clearing, and
next-piece drawing. Search storage is reused, and loss or missing actions fail fixture validation
before timing. Throughput is completed moves (64 per iteration). Its two larger deterministic
checks run in release mode:

```sh
cargo test --release -p tetris-benches --lib -- --ignored
```

The Atlas interning microbenchmark uses a deterministic standard-library hash map; it does not
model the production concurrent interner or frontier.

For the actual in-memory Atlas graph, frontier, and death propagation, run the existing bounded
production replay alongside Criterion:

```sh
cargo test --release -p tetris-playground --features tui --bin tetris_atlas_inmemory \
  tests::long_expansion_replay -- --ignored --exact --nocapture --test-threads 1
```

This single-worker replay starts from the empty board and full bag, performs 2,048 expansions,
and verifies state/edge/death counts plus the complete graph checksum. It prints expansion time;
checksum computation and graph destruction are outside that interval.


See [benchmarking and profiling guidance](AGENTS.md#benchmarks-and-profiling) for named baselines,
reproducible comparisons, and macOS tools including Samply, Instruments, `sample`, and memory checks.

## ASM analysis

Install [cargo-show-asm](https://lib.rs/crates/cargo-show-asm) (replaces the unmaintained `cargo-asm`):

```sh
cargo install cargo-show-asm
cargo asm -p tetris-benches --bench benches --features tetris-game/never-inline \
  "clear_and_collapse_rows" 0 --rust
```

## Macro expansion

```sh
cargo expand -p tetris-game --lib
```

Requires nightly and `cargo-expand` (`cargo install cargo-expand`).

## License

[MIT](LICENSE) — Alexander Comerford
