# N-Bag Empty-Return Solver

**Level:** L0 (empirical search)
**Status:** Search tool — finds placement sequences returning to empty for specific seeds

## Strategy

Given N bags drawn from a seeded RNG, search for a placement sequence that
returns the board to empty after all N×7 pieces are placed. Uses configurable
search width to explore placement options.

## Opponent model

**Non-adversarial.** The piece sequence is fully determined by the seed. The
player knows the full sequence in advance.

## Constraints

- N bags (must be multiple of 5 for empty-board return due to cell-count arithmetic)
- Seeded deterministic RNG for piece sequence
- Configurable search width
- Target: empty board (0 cells)

## Key arithmetic

Each bag places 28 cells. Line clears remove 10 cells each. For N bags to
return to empty: `28N` must be divisible by 10, so N must be a multiple of 5.

- 5 bags → 14 lines needed (2.8/bag)
- 10 bags → 28 lines needed (2.8/bag)

## Usage

```sh
cargo run --release -p tetris-playground --bin tetris_nbag_solver -- \
  --num-bags 5 --seed 42
```
