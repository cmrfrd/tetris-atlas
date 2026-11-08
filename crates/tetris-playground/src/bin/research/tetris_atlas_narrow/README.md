# tetris_atlas_narrow

Reduced-complexity atlas builder for research.

## Constraints

- **4 columns × 20 rows** — implemented as a dedicated `TetrisGameConfig`
- **3 pieces** — L, J, and O
- **3-piece bag** — refills with only the allowed pieces
- Expansion also applies the current source's height ≤ 10 and total-holes ≤ 10 board filter.

## Why

The full 10-col, 7-piece state space is enormous. This variant shrinks it
dramatically while preserving the core combinatorial structure (rotations,
line clears, stacking decisions). If we can build a complete atlas for the
narrow game, it can inform techniques for the full game. A narrow-game policy proves only
the modified rules unless a separate argument connects it to the canonical game.

## Running

```sh
# Build the atlas
mkdir -p artifacts/databases/narrow
cargo run --release -p tetris-playground --features tui --bin tetris_atlas_narrow -- \
  --path artifacts/databases/narrow create

# Explore interactively
cargo run --release -p tetris-playground --features tui --bin tetris_atlas_narrow -- \
  --path artifacts/databases/narrow explore
```

## State space estimate

- Board: 4 columns × 20 rows = 2^80 raw states (far fewer reachable)
- At most three next-piece choices, depending on the remaining bag
- Bag states: 2^3 - 1 = 7
- These are raw representation counts, not measured reachable or certified-winning state counts.
