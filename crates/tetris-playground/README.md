# Tetris Playground

Runnable demos, Atlas builders, training tools, and research experiments. Cargo targets and required
features are declared in [Cargo.toml](Cargo.toml). New runtime output belongs under `artifacts/`.

## Evaluation framework

The proof target is infinite survival from the empty board under the canonical 10×20 engine for
**every legal 7-bag sequence**. At each state, the required order is **for every possible next
piece, there exists a surviving placement**. A cycle alone is insufficient: selected moves must
cover every legal draw and remain inside a verified policy-closed set reachable from the seed.

Use the shared [mission and milestones](../../AGENTS.md#mission-and-proof-standard). Finite play,
sampled rollouts, and heuristic search failures do not establish an infinite-play theorem or an
impossibility result.

### How experimental restrictions affect a result

| Restriction | Meaning |
|---|---|
| Fixed or fully known piece sequence | A witness applies to that sequence/lookahead model. |
| Reduced board width or piece set | A result applies to the modified game. |
| Tighter height, hole, or roughness carrier | A verified closed policy inside the carrier can prove canonical survival; failure inside it does not prove the full game losing. |
| Empty-board reset after a fixed number of bags | A verified online reset strategy can repeat indefinitely; it is a stricter target than general survival. |
| Sampled or abstract transitions | Require concrete transition replay and closure verification before certification. |

### Cell-count arithmetic

- Seven tetrominoes add 28 cells per complete bag.
- A standard line clear removes 10 cells.
- Returning to the same cell count after `N` complete bags requires `28N mod 10 = 0`.
  Thus `N` must be a multiple of five; this is necessary, not sufficient.
- Five bags add 140 cells, requiring 14 clears to return to the same cell count.
- There are `7! = 5040` permutations of one complete bag.

## Research binaries

| Binary | Purpose | Required features | Documentation |
|---|---|---|---|
| `tetris_7bag_atlas` | Adversarial AND-OR retrograde elimination in a bounded carrier | None | [README](src/bin/research/tetris_7bag_atlas/README.md) |
| `tetris_bag_cycle_solver` | Adversarial finite-horizon empty-reset or low-clean target search | None | [README](src/bin/research/tetris_bag_cycle_solver/README.md) |
| `tetris_success_set_solver` | Closed winning-set search with proof-graph and engine replay checks | None | [README](src/bin/research/tetris_success_set_solver/README.md) |
| `tetris_nbag_solver` | Empty-return search for a fixed seeded sequence | None | [README](src/bin/research/tetris_nbag_solver/README.md) |
| `tetris_five_bag_chain` | MCTS tree reuse across related known five-bag sequences | `tui` | [README](src/bin/research/tetris_five_bag_chain/README.md) |
| `tetris_atlas_narrow` | Atlas exploration for the four-column O/L/J variant | `tui` | [README](src/bin/research/tetris_atlas_narrow/README.md) |
| `tetris_invariant_synthesis` | Sampled abstract safety-game synthesis and falsification | None | [README](src/bin/research/tetris_invariant_synthesis/README.md) |
| `tetris_decision_forest` | Boolean decision-forest evolution and search-teacher distillation | None | [README](src/bin/research/tetris_decision_forest/README.md) |

Research results are specific to a run, engine version, configuration, and opponent model. Preserve
those details with replayable witnesses or certificates. An unsuccessful beam/MCTS run does not
show that its sequence is impossible.

## Demos and tools

| Binary | Purpose | Required features |
|---|---|---|
| `tetris_demo_single` | Single beam search | None |
| `tetris_demo_multi` | Multi-beam voting | None |
| `tetris_demo_adaptive_multi` | Adaptive multi-beam search | None |
| `tetris_five_bag_empty` | Five-bag empty-return experiment | None |
| `tetris_five_bag_bench` | Five-bag experiment timing | None |
| `tetris_state_trajectory` | State-trajectory visualization | None |
| `tetris_cli` | Interactive ML-backed game CLI | `ml` |

## Atlas builders

| Binary | Purpose | Required features |
|---|---|---|
| `tetris_atlas_inmemory` | In-memory graph/frontier with checkpointing and death propagation | `tui` |
| `tetris_atlas_rocksdb` | RocksDB-backed Atlas | `rocksdb-backend,tui` |
| `tetris_atlas_partitioned` | Partitioned RocksDB-backed Atlas | `rocksdb-backend` |
| `tetris_atlas_bagstate` | Bag-state-aware RocksDB exploration | `rocksdb-backend` |
| `tetris_atlas_puct` | PUCT-guided Atlas exploration | None |

Graph expansion and persisted lookup entries require independent closure and seed-reachability
checks before they establish a survival proof.

## Training

| Binary | Purpose | Required features |
|---|---|---|
| `tetris_train_beam_supervised` | Candle model trained from beam-search trajectories | `ml` |
| `tetris_train_genetic` | Genetic search over heuristic weights | `ml` |

Add `candle-metal` or `candle-cuda` when using that configured accelerator. See the root
[training commands](../../README.md#training) for checkpoint-directory and resume syntax.
