# Tetris Atlas


## testing

```shell
cargo test
```

## Expand to see generated code

```shell
cargo expand --lib utils::asdf
```

## Benchmarking

All benchmarks are located in the `tetris-benches` crate.

```shell
# Run all benchmarks with criterion reports
cargo bench -p tetris-benches

# Generate flamegraph (5 second profiling)
cargo bench -p tetris-benches -- --profile-time 5
```

## ASM analysis

```shell
## example analyzing a specific function
cargo asm --features never-inline --dev --lib "tetris_atlas::utils::rshift_slice_from_mask_u32" 0 --rust
```

## Training

```
cargo run 
```