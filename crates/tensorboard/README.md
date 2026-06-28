# tensorboard

Minimal TensorBoard event-file writer (scalars, histograms, images) used by the
ML training binaries.

## Protobuf bindings

The protobuf bindings live in `src/tensorboard_generated/` and are **checked in**.
A normal build just compiles those files — it does **not** pull in the
`prost-build` toolchain or hit the network. This keeps clean builds fast.

### When to regenerate

Only when the protobuf schema needs to change, e.g.:

- bumping the TensorBoard version (`TAG` in `../../scripts/download_tensorboard_protos.sh`)
- adding a `.proto` message/field the writer needs

A normal code change to this crate does **not** require regenerating.

### How to regenerate

```sh
# Pull fresh .proto sources + run prost codegen into src/tensorboard_generated/
cargo build -p tensorboard --features regen-protos

# Review and commit the regenerated bindings
git add crates/tensorboard/src/tensorboard_generated/
```

The `regen-protos` feature is the only thing that activates `prost-build` and the
proto download script (`scripts/download_tensorboard_protos.sh`). Without it,
`build.rs` errors if the checked-in bindings are missing rather than silently
regenerating.
