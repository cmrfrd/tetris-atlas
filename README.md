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

```shell
cargo bench --bench benches -- --profile-time 5 && cargo bench --bench benches
```

## ASM analysis

```shell
## example analyzing a specific function
cargo asm --features never-inline --dev --lib "tetris_atlas::utils::rshift_slice_from_mask_u32" 0 --rust
```