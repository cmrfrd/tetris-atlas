# tetris_atlas_narrow

Reduced-complexity atlas builder for research.

## Constraints

- **4 columns × 20 rows** — implemented as a dedicated `TetrisGameConfig`
- **3 pieces** — L, J, and O
- **3-piece bag** — refills with only the allowed pieces

## Why

The full 10-col, 7-piece state space is enormous. This variant shrinks it
dramatically while preserving the core combinatorial structure (rotations,
line clears, stacking decisions). If we can build a complete atlas for the
narrow game, the techniques transfer to the full game.

## Running

```sh
# Build the atlas
cargo run --release -p tetris-playground --bin tetris_atlas_narrow -- create

# Explore interactively
cargo run --release -p tetris-playground --bin tetris_atlas_narrow -- explore
```

## State space estimate

- Board: 4 columns × 20 rows = 2^80 raw states (far fewer reachable)
- Pieces per bag position: 3
- Bag states: 2^3 - 1 = 7
- Much more tractable than the full 2^200 × 127 space
