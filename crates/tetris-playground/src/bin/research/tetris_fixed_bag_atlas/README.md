# Fixed-Bag-Order Atlas Solver

**Level:** L1 (solved under added constraints)
**Status:** **SOLVED** — ~25-27% winning states, empty board is winning

## Strategy

Solves a simplified Tetris variant where the bag ordering is fixed: the same
7-piece permutation repeats every cycle (e.g., always O,I,S,Z,T,L,J). This
makes the piece sequence fully deterministic, reducing state to
`(board, position_in_cycle)`.

1. **Forward BFS**: Expand all reachable states from the empty board
2. **Retrograde elimination (GFP)**: A state dies when every placement leads
   to a dead successor. Propagate backward until stable.
3. **Verification**: Confirm atlas self-consistency

## Opponent model

**None.** The piece sequence is fully deterministic (fixed permutation repeating).
The player has complete information.

## Constraints

- 10-wide board (canonical)
- Fixed cycle order (e.g., O,I,S,Z,T,L,J repeating forever)
- Configurable `--max-height`, `--max-holes`, `--max-roughness`, `--max-count`
- State spaces: ~25M states at h≤4/holes≤2/rough≤8

## Results

With `--max-height 4 --max-holes 2 --max-roughness 8`:
- Most cycle orderings yield **~25-27% winning fraction**
- The empty board is typically **winning** (solvable forever)
- Solving takes ~30s on a single core
- Atlas is verified self-consistent

This constitutes an **L1 proof**: infinite play is proven for the constrained
game with deterministic piece sequence.

## Usage

```sh
# Build the atlas
cargo run --release -p tetris-playground --bin tetris_fixed_bag_atlas -- build \
  --cycle O,I,S,Z,T,L,J --max-height 4 --max-holes 2 --max-roughness 8 \
  --atlas-path atlas.bin

# Play from the atlas (infinite loop)
cargo run --release -p tetris-playground --bin tetris_fixed_bag_atlas -- play \
  --atlas-path atlas.bin
```

## Gap to canonical game

The canonical 7-bag randomizer produces 5040 possible orderings per bag. This
solver only handles one fixed ordering. Bridging this gap requires handling all
possible orderings — which is the adversarial problem addressed by
`tetris_7bag_atlas` and `tetris_bag_cycle_solver`.
