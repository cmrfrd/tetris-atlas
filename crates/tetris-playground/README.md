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

#### `tetris_cycle_atlas`
- **Strategy:** GFP over boards at bag-refill boundaries; safe if all adversarial piece sequences return to safe set
- **Opponent:** Adversarial within each bag
- **Constraints:** Configurable h/holes/roughness/count, two-tier admissibility envelopes
- **Status:** Converges to empty safe set on all tested profiles
- **README:** [Yes](src/bin/research/tetris_cycle_atlas/README.md)

#### `tetris_bag_cycle_solver`
- **Strategy:** Memoized AND-OR depth-bounded search proving player can force empty-board reset in N cycles against adversary
- **Opponent:** Adversarial bag ordering
- **Constraints:** N bags (multiple of 5), configurable h/holes/roughness/count
- **Status:** Tested at various bounds; no winning configuration found yet
- **README:** [Yes](src/bin/research/tetris_bag_cycle_solver/README.md)

#### `tetris_mcgs`
- **Strategy:** Monte Carlo Graph Search with AND-OR semantics. Builds a graph (not tree) of unique `(board, bag_state)` nodes. AND nodes = adversary picks piece from bag. OR nodes = player picks placement. Cycle detection at bag-refill boundaries.
- **Opponent:** Adversarial (AND nodes enumerate all piece choices via `bag.iter_next_states()`)
- **Constraints:** Configurable h/holes/roughness, hard node cap prevents OOM
- **Status:** New — MCGS implementation
- **README:** [Yes](src/bin/research/tetris_mcgs/README.md)

#### `tetris_success_set_solver`
- **Strategy:** Exact in-memory retrograde solver; state=(board, bag_mask), target=empty board, AND-OR with witness extraction
- **Opponent:** Adversarial
- **Constraints:** Configurable h/holes/cells/roughness/height_spread
- **Status:** In development
- **README:** [Yes](src/bin/research/tetris_success_set_solver/README.md)

### Non-adversarial / known-sequence solvers

#### `tetris_bag_checkpoint`
- **Strategy:** Multiple approaches for 5-bag cycle closure with known permutation ordering:
  - `discover`: Find checkpoint boards reachable by all 5040 perms from empty
  - `chain-targeted`: Build 5-step residue-class chain with greedy set cover
  - `gfp`/`gfp2`/`lfp`: Fixed-point pruning/expansion (all converge to empty)
  - `five-bag-cycle`: Beam search / MCTS / split-mode for specific 5-bag sequences
  - `prove`: Exhaustive DFS for impossibility proofs
- **Opponent:** Non-adversarial (player knows full bag permutation)
- **Constraints:** Configurable h/holes, all 5040 perms must be covered
- **Status:**
  - Chain approach: 2→14→83→1000+ cover boards, diverges at step 3
  - MCTS (pacing scorer, 10M rollouts): **71/100 random 5-bag sequences return to empty**
  - ~29% of sequences appear structurally unable to clear exactly 14 lines
  - 100M rollouts on hardest case: still fails (always 10 residual cells)
- **Best command:**
  ```sh
  cargo run --release -p tetris-playground --bin tetris_bag_checkpoint -- \
    five-bag-cycle --mcts --rollouts 10000000 --temperature 1.0 --trials 100 --seed 42
  ```
- **README:** [Yes](src/bin/research/tetris_bag_checkpoint/README.md)

#### `tetris_fixed_bag_atlas`
- **Strategy:** Solves deterministic piece sequence (fixed 7-piece permutation repeating forever) via BFS + retrograde GFP
- **Opponent:** None — piece sequence is fully deterministic
- **Constraints:** Fixed cycle order (e.g., O,I,S,Z,T,L,J repeating), configurable h/holes/roughness/count
- **Status:** **~25-27% of boards are winning** on most tested cycle orders. This is a solved constrained game (L1).
- **README:** [Yes](src/bin/research/tetris_fixed_bag_atlas/README.md)

#### `tetris_nbag_solver`
- **Strategy:** Searches for placement sequence over N seeded random bags returning to empty
- **Opponent:** Non-adversarial (known sequence from seed)
- **Constraints:** N bags (multiple of 5 for empty target), seeded RNG
- **Status:** Empirical search tool, not exhaustive
- **README:** [Yes](src/bin/research/tetris_nbag_solver/README.md)

### SMT / synthesis approaches

#### `tetris_invariant_smt`
- **Strategy:** Finds per-bag-state inductive invariants P_B(board) via SMT/CEGIS such that all 448 bag-state transitions remain safe
- **Opponent:** Adversarial (all transitions must be safe)
- **Constraints:** Bitvector-encoded board, per-bag-state invariant looseness
- **Status:** In development
- **README:** [Yes](src/bin/research/tetris_invariant_smt/README.md)

#### `tetris_program_synthesis`
- **Strategy:** Searches for compact bitvector VM programs via Z3 SMT + CEGIS counterexample loop
- **Opponent:** Adversarial (CEGIS verifies against counterexamples)
- **Constraints:** Fixed VM instruction set, configurable h/holes/roughness invariant
- **Status:** No programs found that maintain invariants across all transitions
- **README:** [Yes](src/bin/research/tetris_program_synthesis/README.md)

#### `tetris_invariant_synthesis`
- **Strategy:** Fixed-point elimination on sampled concrete states + refinement rounds
- **Opponent:** Non-adversarial (falsification via rollouts)
- **Constraints:** Bucketed state space (height/holes/roughness)
- **Status:** Experimental
- **README:** [Yes](src/bin/research/tetris_invariant_synthesis/README.md)

### Heuristic / empirical play

#### `tetris_safe_set`
- **Strategy:** Allocation-free best-first backtracker proving infinite play via bounded recursive recovery scripts
- **Opponent:** Non-adversarial (forced bag sequence)
- **Constraints:** h≤4, recovery depth ≤2 bags
- **Status:** Experimental
- **README:** [Yes](src/bin/research/tetris_safe_set/README.md)

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
| Fixed-bag atlas | **L1** | **SOLVED** — 25-27% winning states for fixed piece cycles |
| Bag checkpoint (chain) | — | Chain diverges at step 3 (cover set explosion) |
| Bag checkpoint (MCTS) | L0 | 71% of random 5-bag sequences return to empty |
| MCGS cycle finder | — | New — graph-based AND-OR cycle search |
| All exhaustive adversarial solvers | — | 0% winning at all tested bounds |
| SMT/synthesis | — | No invariants found yet |
| Beam search demos | L0 | Millions of pieces without death (empirical) |

### What's proven

- **Fixed-bag atlas (L1):** For any fixed repeating piece order, ~25% of low-height boards have a winning strategy. This proves infinite play under the constraint of deterministic piece sequence.
- **5-bag cycle feasibility:** ~71% of random 5-bag sequences can return to empty board. ~29% structurally cannot (exactly 13 line clears, never 14).

### What's NOT proven

- No adversarial solver has found a winning state at any height bound
- No 5-step chain cycle has been closed (cover diverges at step 3)
- No inductive invariant has been synthesized
- The canonical game (adversarial 7-bag on 10×20) remains unproven
