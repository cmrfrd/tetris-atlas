# AGENTS.md — Tetris Atlas

This is a **research project**. The ultimate mission is to **"solve tetris"** — to build a complete lookup table (the **Atlas**) for the canonical game of Tetris such that for any reachable game state, the correct move to play forever can be found by a simple table lookup.

A solved game means **provably infinite play**: a proof-by-construction that there exists a strategy under the canonical rules which never tops out, regardless of the piece sequence drawn from the 7-bag randomizer.

---

## Project framing

### The final goal: a total lookup table

The ideal end state of this project is a **global Atlas** — a mapping from every reachable `(board, bag_state, current_piece)` to the placement that keeps the game alive indefinitely. If such a table exists and is complete, then playing Tetris reduces to:

1. Observe the current game state.
2. Look up the state in the Atlas.
3. Play the prescribed move.
4. Repeat forever.

This constitutes a **proof by construction** that Tetris (under the canonical 7-bag rules on a 10×20 board) is survivable for infinite horizons. The Atlas itself is the proof artifact.

### Why this is hard

The raw state space of Tetris is enormous. A 10×20 binary board alone has up to 2^200 configurations (though the vast majority are unreachable). The 7-bag randomizer adds combinatorial branching on every piece draw. Building a total lookup table by brute force is infeasible — the project must find structure to exploit:

- **Closed cycles**: subsets of states that form loops — once you enter the cycle, you stay in it forever. Finding even one such cycle proves infinite play for the states it covers.
- **DAG search with death propagation**: expand the reachable state graph, propagate "death" backward from losing states, and identify the surviving subgraph.
- **Symmetry and compression**: many board states are strategically equivalent; the effective state space may be far smaller than the raw space.

### Milestone hierarchy

The milestones below are ordered from most achievable to the final goal. Each earlier milestone provides tools, evidence, or infrastructure for the next.

**M0 — Strong finite play (subgoal)**
Agents that survive for very long horizons (millions of pieces) across many seeds via beam search, multi-beam voting, and heuristic tuning. This is the current baseline and provides the search infrastructure everything else builds on.

**M1 — Empirically infinite play (subgoal)**
Agents (search-based, learned, or hybrid) that have never been observed to top out in any test run. Not a proof, but strong empirical evidence that infinite play is reachable.

**M2 — Closed cycles found (subgoal)**
Discovery of at least one closed cycle in the state graph — a set of states and a policy that loops through them indefinitely, clearing lines as needed to keep the board bounded. This is a local proof of infinite play for those states.

**M3 — Reachable closed cycles (subgoal)**
A closed cycle from M2 that is also reachable from the initial empty-board state via a known action sequence. This proves that a real game, starting from an empty board, can enter a provably infinite loop.

**M4 — The Atlas (final goal)**
A complete or practically complete lookup table covering all reachable states (or a sufficient surviving subgraph), such that any game played from the empty board using the Atlas as its policy never tops out. The Atlas is the proof artifact.

### Evaluation and reproducibility

All results must report:
- Number of runs (N) and seed set definition
- Aggregation statistic (mean/median) and dispersion (stdev/quantiles)
- Runtime cost (wall time, pieces/sec, hardware info)
- For atlas work: state count, cycle length, coverage fraction, expansion rate

Every claim must be backed by a run spec, logs, seed set, and commit hash.

### Canonical environment

The ground-truth Tetris engine lives in `crates/tetris-game`:
- Board: 10 columns × 20 rows, column-major `[u32; 10]` layout
- Pieces: 7 standard tetrominoes (O, I, S, Z, T, L, J)
- Randomizer: 7-bag system with deterministic seeded RNG
- Actions: 117 precomputed `TetrisPiecePlacement` values (piece × rotation × column)
- Loss: board height exceeds 20 rows

All agents, search algorithms, atlas builders, and ML models depend on this implementation.

---

## Repository structure

```
tetris-atlas/
├── crates/
│   ├── tetris-game/        # Core engine: board, pieces, bag, placement, line clearing
│   ├── tetris-search/      # Beam search, multi-beam voting, parallel BFS explorer
│   ├── tetris-ml/          # Candle-based ML: modules, optimizers, tensors, data gen, TUI
│   ├── tetris-playground/  # All runnable binaries: demos, training, atlas, experiments
│   ├── tetris-benches/     # Criterion + pprof benchmarks
│   ├── tensorboard/        # TensorBoard event-file writer
│   └── proc-macros/        # Conditional inlining and loop unrolling macros
├── scripts/                # Plotting and data download scripts (Python)
├── .docker/                # Dockerfiles for CPU and GPU builds
├── artifacts/              # Runtime outputs (checkpoints, logdir, databases, data)
│   ├── checkpoints/
│   ├── logdir/
│   ├── databases/
│   └── data/
├── rust-toolchain.toml     # Pinned to nightly
└── Cargo.toml              # Virtual workspace
```

### Crate dependency graph

```
tetris-game ← proc-macros
     ↑
tetris-search ← proc-macros
     ↑
tetris-ml ← candle-core, candle-nn
     ↑
tetris-playground ← tensorboard, clap, rocksdb, heed, feoxdb, ratatui, ...
```

---

## Agent roles

Agents are humans, automated scripts, or LLM-based contributors. Each must produce artifacts that are reviewable and reproducible.

| Role | Owns | Key crates |
|---|---|---|
| **Research Director** | Research direction, objective definitions, milestone acceptance | — |
| **Search Engineer** | Beam search, multi-beam voting, tiered heuristics, pruning | `tetris-search` |
| **Atlas Engineer** | State-space expansion, persistent lookup tables, coverage metrics | `tetris-playground` (atlas bins) |
| **ML Engineer** | Learned models, training loops, dataset generation, hybrid agents | `tetris-ml`, `tetris-playground` (train bins) |
| **Engine Engineer** | Core game logic correctness, performance, new features | `tetris-game`, `proc-macros` |
| **Benchmarking Engineer** | Performance evidence, flamegraphs, regression detection | `tetris-benches` |
| **Infra Engineer** | CI, Docker builds, Fly.io deploys, toolchain stability | `.docker/`, `fly.*.toml` |

---

## Coding conventions

### Rust toolchain
- **Nightly required** — pinned via `rust-toolchain.toml`
- Uses `#![feature(generic_const_exprs)]` and edition 2024 in newer crates
- `rustfmt.toml`: edition 2021, 100 char max width, 4-space tabs, Unix newlines

### Style
- Clippy is configured in the workspace `Cargo.toml` with many pedantic lints relaxed
- `unwrap_used` and `expect_used` are warnings; `panic!`, `todo!`, `unimplemented!`, `unreachable!` are **denied**
- Prefer `anyhow::Result` for fallible operations
- Performance-critical code uses const generics, bitwise operations, stack allocation (`HeaplessVec`), and compile-time unrolling (`repeat_idx_unroll!`)

### Feature flags
- `never-inline` — disables inlining for ASM/benchmark analysis
- `candle-cuda` — enables CUDA backend for ML
- `candle-metal` — enables Metal backend for ML (macOS)
- `dtype-f16` / `dtype-bf16` / `dtype-f32` / `dtype-f64` — float precision selection

---

## Testing

Run all workspace tests:
```sh
cargo test --workspace
```

Run a single crate:
```sh
cargo test -p tetris-game
cargo test -p tetris-search
```

Guidelines:
- Use fixed seeds for deterministic tests in search/ML components
- Keep unit tests fast; put longer randomized tests behind `#[ignore]`
- All tests must pass before merging

### Formatting and linting
```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

---

## Benchmarking

Benchmarks live in `tetris-benches` using Criterion + pprof:

```sh
cargo bench -p tetris-benches

# Profiling mode (reduced noise, generate flamegraphs)
cargo bench -p tetris-benches -- --profile-time 5
```

Criterion HTML reports go to `target/criterion/`. Flamegraphs are emitted via the pprof integration.

### ASM analysis

The `cargo-asm` crate is unmaintained — use `cargo-show-asm` instead:
```sh
cargo install cargo-show-asm
cargo asm -p tetris-game --features never-inline --dev --lib "tetris_game::utils::rshift_slice_from_mask_u32" 0 --rust
```

### Macro expansion
```sh
cargo expand -p tetris-game --lib utils::rshift_slice_from_mask_u32
```

---

## Running demos and experiments

### Search demos

```sh
# Single beam search
cargo run --release -p tetris-playground --bin tetris_demo_single

# Multi-beam voting
cargo run --release -p tetris-playground --bin tetris_demo_multi

# Tiered multi-beam (adaptive by board difficulty)
cargo run --release -p tetris-playground --bin tetris_demo_multi_tiered
```

### Atlas builders

```sh
# In-memory atlas (fastest, checkpoints to tetris_atlas_inmemory.bin)
cargo run --release -p tetris-playground --bin tetris_atlas_inmemory -- create
cargo run --release -p tetris-playground --bin tetris_atlas_inmemory -- explore

# RocksDB atlas
cargo run --release -p tetris-playground --bin tetris_atlas_rocksdb -- -d ./artifacts/databases/rocksdb create
cargo run --release -p tetris-playground --bin tetris_atlas_rocksdb -- -d ./artifacts/databases/rocksdb explore

# A* cycle search
cargo run --release -p tetris-playground --bin tetris_atlas_astar -- --max-states 2000000 --top-k 3
```

### ML training

```sh
# Beam-supervised learning (GPU)
RAYON_STACK_SIZE_MB=64 cargo run --release -p tetris-playground --features candle-metal \
  --bin tetris_train_beam_supervised -- train \
  --run-name my-run --logdir ./artifacts/logdir --checkpoint-dir ./artifacts/checkpoints

# Policy gradients
cargo run --release -p tetris-playground --bin tetris_train_policy_gradients -- \
  --run-name pg-v1 --logdir ./artifacts/logdir --checkpoint-dir ./artifacts/checkpoints

# Genetic algorithm over heuristic weights
cargo run --release -p tetris-playground --bin tetris_train_genetic

# DQN
cargo run --release -p tetris-playground --bin tetris_train_dqn -- \
  --run-name dqn-v1 --logdir ./artifacts/logdir --checkpoint-dir ./artifacts/checkpoints

# Inference from a checkpoint
cargo run --release -p tetris-playground --features candle-metal \
  --bin tetris_train_beam_supervised -- inference \
  --checkpoint ./artifacts/checkpoints/my-run/model.safetensors
```

### Threading controls

Search and exploration use Rayon. Configure via environment variables:
```sh
RAYON_NUM_THREADS=16 RAYON_STACK_SIZE_MB=128 cargo run --release -p tetris-playground --bin tetris_demo_multi
```

Default stack size is 64MB.

---

## Deployment

### Docker

```sh
# CPU build (in-memory atlas)
docker build -f .docker/Dockerfile.cpu.run -t tetris-atlas-cpu .

# GPU build (beam-supervised training, requires CUDA)
docker build -f .docker/Dockerfile.gpu.run -t tetris-atlas-gpu .
```

---

## Artifacts and reproducibility

### Artifact layout

All runtime outputs go under `artifacts/` (gitignored):
- `artifacts/checkpoints/<run-name>/` — model safetensors
- `artifacts/logdir/<run-name>/` — TensorBoard event files
- `artifacts/databases/` — atlas DB files (RocksDB, LMDB, etc.)
- `artifacts/data/` — generated datasets
- `artifacts/output/` — CSV metrics, plots

### Determinism

Every evaluation should be replayable from:
- Engine seed(s) + bag state
- Action sequence OR policy model + commit hash

### Metrics to track

Per-run and aggregate:
- `pieces_placed`, `lines_cleared`, `score`
- `max_height`, `avg_height`, `holes`
- `time_per_move_ms` (mean/p95), `pieces_per_sec`
- Atlas-specific: `lookup_size`, `frontier_size`, `expansion_rate` (boards/sec)

---

## Research directions

### Approaches by milestone

#### M0 — Strong finite play

| Approach | Status | Binary |
|---|---|---|
| Single beam search | Working | `tetris_demo_single` |
| Multi-beam voting | Working | `tetris_demo_multi` |
| Tiered multi-beam | Working | `tetris_demo_multi_tiered` |
| Genetic heuristic tuning | Working | `tetris_train_genetic` |
| Beam-supervised learning | Active | `tetris_train_beam_supervised` |
| Policy gradients | Experimental | `tetris_train_policy_gradients` |
| DQN | Experimental | `tetris_train_dqn` |
| Evolution strategies | Experimental | `tetris_train_evolution` |
| Value function | Experimental | `tetris_train_value_function` |
| Transition models | Experimental | `tetris_train_transition`, `_transformer` |

#### M1 — Empirically infinite play

- Scaling beam search width/depth until no observed deaths
- Hybrid agents: learned policy for fast moves, search for hard positions
- Imitation from search: generate large datasets from multi-beam, train fast policy network

#### M2/M3 — Closed cycles and reachability

| Approach | Status | Binary |
|---|---|---|
| A* cycle search | Working | `tetris_atlas_astar` |
| In-memory atlas (BFS expansion) | Working | `tetris_atlas_inmemory` |
| In-memory closed subgraph | Working | `tetris_atlas_inmemory_closed` |
| RocksDB/LMDB/FeoxDB atlas | Working | `tetris_atlas_rocksdb`, `_lmdb`, `_feoxdb` |
| Empty-board cycle analysis | Working | `tetris_empty_cycles` |

Key techniques:
- **DAG expansion with backward death propagation**: expand the reachable state graph from a root, identify dead-end states, propagate death backward, and extract the surviving subgraph.
- **Cycle detection**: find strongly connected components in the surviving subgraph — any SCC is a closed cycle proving infinite play for its member states.
- **Reachability bridging**: given a closed cycle, search for a path from the empty board to any cycle entry state.

#### M4 — The Atlas (final goal)

- Grow the surviving subgraph until it covers all reachable states (or a sufficient fraction)
- Compress the Atlas via symmetry exploitation, state equivalence classes, or learned hash functions
- Persist the Atlas in a queryable format (RocksDB, custom on-disk structure, or distributed store)
- Validate completeness: for every state in the Atlas, every possible piece draw leads to another Atlas state

---

## Prompt for automated agents

You are an engineering+research agent working on Tetris Atlas. The final goal is to build a complete lookup table (the Atlas) that proves Tetris is survivable for infinite horizons under canonical 7-bag rules on a 10×20 board. Subgoals include strong finite play via search/ML, empirically infinite agents, and finding closed cycles in the state graph. Always: (1) read relevant code before changing it; (2) propose a minimal change set; (3) add/adjust tests; (4) run `cargo fmt`, `cargo clippy`, `cargo test`; (5) report metrics with seeds, commit, and parameters. Never claim improvements without an eval run over a fixed seed set. When modifying algorithms, include ablations and a rollback plan.
