# Tetris Atlas

A research project aiming to **solve Tetris** — including techniques such as agents that play at superhuman level for arbitrarily long horizons through search, learning, and state-space exploration.

## Quick start / Demos

Runnable binaries are in the `tetris-playground` crate. All artifacts (model checkpoints, datasets, etc.) are generates from these binaries.

```sh
# Run tests
cargo test --workspace

# Watch a beam search agent play
cargo run --release -p tetris-playground --bin tetris_demo_single

# Multi-beam voting (stronger, slower)
cargo run --release -p tetris-playground --bin tetris_demo_multi
```

## Crates

```
crates/
├── tetris-game        Core engine: board, pieces, bag, placement, line clearing
├── tetris-search      Tetris DAG search, Beam search, multi-beam voting, and more
├── tetris-ml          Candle-based ML: NN modules, optimizers, checkpointing
├── tetris-playground  All runnable binaries (demos, training, atlas builders, experiments)
├── tetris-benches     Criterion + pprof benchmarks
├── tensorboard        TensorBoard event-file writer
└── proc-macros        Conditional inlining and loop unrolling macros
```

## Training

Included in this project (specifically in the `tetris-ml` crate) are ML training binaries use Candle with optional CUDA or Metal acceleration.

```sh
# Beam-supervised learning (learns from beam search trajectories)
RAYON_STACK_SIZE_MB=64 cargo run --release -p tetris-playground \
  --features candle-metal \
  --bin tetris_train_beam_supervised -- train \
  --run-name my-run \
  --logdir ./artifacts/logdir \
  --checkpoint-dir ./artifacts/checkpoints

# Resume from checkpoint
cargo run --release -p tetris-playground --features candle-metal \
  --bin tetris_train_beam_supervised -- train \
  --run-name my-run --logdir ./artifacts/logdir --checkpoint-dir ./artifacts/checkpoints --resume

# Run inference
cargo run --release -p tetris-playground --features candle-metal \
  --bin tetris_train_beam_supervised -- inference \
  --checkpoint ./artifacts/checkpoints/my-run/latest.safetensors
```

Use `--features candle-cuda` instead of `candle-metal` for NVIDIA GPUs.

## Benchmarking

```sh
# Run all benchmarks
cargo bench -p tetris-benches

# Generate flamegraphs (5 second profiling)
cargo bench -p tetris-benches -- --profile-time 5
```

Criterion HTML reports are written to `target/criterion/`.

## ASM analysis

Install [cargo-show-asm](https://lib.rs/crates/cargo-show-asm) (replaces the unmaintained `cargo-asm`):

```sh
cargo install cargo-show-asm
cargo asm -p tetris-game --features never-inline --dev --lib \
  "tetris_game::utils::rshift_slice_from_mask_u32" 0 --rust
```

## Macro expansion

```sh
cargo expand -p tetris-game --lib utils::rshift_slice_from_mask_u32
```

Requires nightly and `cargo-expand` (`cargo install cargo-expand`).

## License

[MIT](LICENSE) — Alexander Comerford
