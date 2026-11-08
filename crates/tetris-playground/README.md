# Tetris Playground — Research Registry

All runnable binaries for the Tetris Atlas project. Each binary represents a
distinct approach toward proving infinite play under canonical 7-bag Tetris.

## Evaluation Framework

### What counts as "solved"?

A solution must prove that **there exists a placement strategy that survives
forever** under the canonical rules (10-wide board, 7-bag randomizer, standard
pieces). The strength of a result depends on what constraints it assumes:

| Level | Description | Proof strength |
|-------|-------------|---------------|
| **L0** | Empirical survival (never observed to die) | No proof |
| **L1** | Infinite play under **added constraints** (fixed bag order, bounded height, reduced board) | Partial proof (constrained game) |
| **L2** | Infinite play under **canonical rules** for a **subset** of reachable states | Local proof (cycle found) |
| **L3** | Infinite play under **canonical rules** from the **empty board** | Full proof |

### Constraint taxonomy

Each approach may relax or strengthen the problem along these axes:

| Axis | Canonical | Relaxed | Strengthened |
|------|-----------|---------|-------------|
| **Piece sequence** | 7-bag randomizer (adversarial ordering within bag) | Fixed/known sequence | — |
| **Board width** | 10 | Narrower (e.g., 4-wide) | — |
| **Height bound** | 20 (loss at overflow) | Tighter (e.g., h≤6) | Unbounded |
| **Intermediate constraints** | None | Hole/roughness limits during play | — |
| **Opponent model** | Adversarial (worst-case bag ordering) | Non-adversarial (known ordering) | — |
| **Cycle target** | Any reachable state | Empty board | Non-empty checkpoint |
| **Cycle length** | Any | Fixed N bags (must be multiple of 5 for empty) | — |

### Key arithmetic

- 7 pieces per bag = 28 cells placed
- Line clear removes exactly 10 cells
- N bags can return to same cell count iff `28N mod 10 = 0` → N must be multiple of 5
- 5 bags: 140 cells, 14 lines needed (2.8 lines/bag average)
- 5040 possible bag permutations per bag (7!)

---

## Research Binaries

### Adversarial solvers (worst-case piece ordering)

#### `tetris_7bag_atlas`
- **Strategy:** AND-OR retrograde elimination over (board, bag_remaining) states
- **Opponent:** Full adversarial — enemy picks any remaining bag piece
- **Constraints:** Configurable h/holes/roughness/count bounds
- **Status:** All tested configurations → 0% winning. Adversarial admissibility constraints help the adversary too much.
- **README:** [Yes](src/bin/research/tetris_7bag_atlas/README.md)

#### `tetris_bag_cycle_solver`
- **Strategy:** Memoized AND-OR depth-bounded search proving player can force empty-board reset in N cycles against adversary
- **Opponent:** Adversarial bag ordering
- **Constraints:** N bags (multiple of 5), configurable h/holes/roughness/count
- **Status:** Tested at various bounds; no winning configuration found yet
- **README:** [Yes](src/bin/research/tetris_bag_cycle_solver/README.md)

#### `tetris_success_set_solver`
- **Strategy:** Exact in-memory retrograde solver; state=(board, bag_mask), target=empty board, AND-OR with witness extraction
- **Opponent:** Adversarial
- **Constraints:** Configurable h/holes/cells/roughness/height_spread
- **Status:** In development
- **README:** [Yes](src/bin/research/tetris_success_set_solver/README.md)

### Non-adversarial / known-sequence solvers

#### `tetris_nbag_solver`
- **Strategy:** Searches for placement sequence over N seeded random bags returning to empty
- **Opponent:** Non-adversarial (known sequence from seed)
- **Constraints:** N bags (multiple of 5 for empty target), seeded RNG
- **Status:** Empirical search tool, not exhaustive
- **README:** [Yes](src/bin/research/tetris_nbag_solver/README.md)

### SMT / synthesis approaches

#### `tetris_invariant_synthesis`
- **Strategy:** Fixed-point elimination on sampled concrete states + refinement rounds
- **Opponent:** Non-adversarial (falsification via rollouts)
- **Constraints:** Bucketed state space (height/holes/roughness)
- **Status:** Experimental
- **README:** [Yes](src/bin/research/tetris_invariant_synthesis/README.md)

---

## Non-Research Binaries

### Examples (demos)

| Binary | Description |
|--------|-------------|
| `tetris_demo_single` | Single beam search playing Tetris |
| `tetris_demo_multi` | Multi-beam voting |
| `tetris_demo_adaptive_multi` | Adaptive multi-beam with dynamic width |

### Atlas builders

| Binary | Description |
|--------|-------------|
| `tetris_atlas_inmemory` | In-memory BFS state-space expansion with checkpointing |
| `tetris_atlas_rocksdb` | RocksDB-backed persistent atlas |

### Training

| Binary | Description |
|--------|-------------|
| `tetris_train_beam_supervised` | Beam-supervised neural net training (Candle) |
| `tetris_train_genetic` | Genetic algorithm over heuristic weight vectors |

---

## Progress Summary

| Approach | Level | Key result |
|----------|-------|------------|
| All exhaustive adversarial solvers | — | 0% winning at all tested bounds |
| SMT/synthesis | — | No invariants found yet |
| Beam search demos | L0 | Millions of pieces without death (empirical) |

### What's proven


### What's NOT proven

- No adversarial solver has found a winning state at any height bound
- No 5-step chain cycle has been closed (cover diverges at step 3)
- No inductive invariant has been synthesized
- The canonical game (adversarial 7-bag on 10×20) remains unproven
