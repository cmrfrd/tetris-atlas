# `tetris_success_set_solver`

This directory is an isolated implementation plan and scaffold for an exact in-memory solver for the capped recovery game:

- state = `(board, bag_mask)`
- target = `empty board`
- rule = for every next piece allowed by the current bag, there exists a legal placement that stays inside the success set

## Why this design fits this repo

The repo already has the core primitives the solver should reuse directly:

- `tetris_game::TetrisBoard`
  Exact board occupancy, line clearing, height, loss detection, and hashing.
- `tetris_game::TetrisPieceBagState`
  Exact 7-bit bag-mask representation with `contains`, `remove`, `fill`, and `iter_next_states`.
- `tetris_game::TetrisPiecePlacement`
  Canonical final placements, globally indexed by `u8`, with `all_from_piece(piece)`.
- `TetrisBoard::apply_piece_placement`
  Exact transition operator for `(board, placement) -> (board', lines_cleared, is_lost)`.

That means this solver should not introduce an alternate board encoding or alternate bag logic in the first milestone. The first exact implementation should use `TetrisBoard` as the canonical board key and only add dense interning and compact graph storage around it.

## Key repo-specific decisions

### 1. Use `TetrisBoard` directly as the interned board key

The engine already stores boards as `[u32; 10]`, derives `Hash + Eq + Ord + Copy`, and line clears are applied immediately inside `apply_piece_placement`. There is no extra canonicalization step needed beyond rejecting boards above the configured height cap.

### 2. Use `TetrisPieceBagState` directly as the bag mask

The engine’s bag-state type already matches the exact solver’s needs. The correct next-bag transition is:

- remove current piece
- if bag becomes empty, refill to full mask

That behavior already exists in `iter_next_states`.

### 3. Use `TetrisPiecePlacement::all_from_piece(piece)` for action enumeration

This repo models legal actions as canonical final placements for the current piece, not path-dependent input sequences. The solver should follow that exact convention.

### 4. Avoid `TetrisGame` during graph construction

`TetrisGame` is useful for full gameplay, but it shuffles a bag stream and advances RNG state. The success-set solver only needs:

- current board
- current bag mask
- current concrete piece
- exact board transition under one final placement

So graph construction should work from `TetrisBoard`, `TetrisPieceBagState`, and `TetrisPiecePlacement` directly.

## Proposed module split

- `main.rs`
  CLI entrypoint and top-level orchestration.
- `config.rs`
  Solver configuration, CLI-to-config translation, and milestone presets.
- `state.rs`
  Dense IDs, bag helpers, packed placement aliases, and state-key utilities.
- `universe.rs`
  Board/state interning plus forward-reachable capped-universe expansion.
- `graph.rs`
  Flat edge storage and per-state piece-range indexing.
- `retrograde.rs`
  Fixed-point success-set solve plus exact depth recurrence.
- `policy.rs`
  Witness extraction and invariant verification helpers.
- `verifier.rs`
  Independent replay verification for proof results.

## Milestone plan

### M1: exact correctness-first solver

- Build the forward-reachable capped universe from all empty-board nonzero bag masks.
- Store transitions in flat vectors.
- Solve with a simple full-rescan fixed point.
- Store one best witness action per `(state, piece)`.
- Print exact counts and depth stats.

### M2: graph and solve optimization

- Replace nested vectors with tighter range-based indexing if needed.
- Add reverse predecessor tracking for dependency-driven retrograde updates.
- Add packed win flags and optional successor dedup-by-state.

### M3: analysis and export

- Query `is_winning(board, FULL_BAG)`.
- Query `is_winning_for_all_bags(board)`.
- Print depth histograms and board-height histograms.
- Add optional artifact export for later visualization or atlas bridging work.

## Recommended first implementation order

1. Implement `state.rs` helpers and tests.
2. Implement `universe.rs` around `HashMap<TetrisBoard, BoardId>` and `HashMap<StateKey, StateId>`.
3. Enumerate transitions with `TetrisPiecePlacement::all_from_piece`.
4. Filter successors with `!is_lost && board.height() <= height_cap`.
5. Implement the fixed-point scan in `retrograde.rs`.
6. Add post-solve verification that every stored witness points to a winning successor and satisfies the exact depth recurrence.

## Important invariants to preserve

- Empty board is winning for every nonzero bag mask.
- A state with a piece branch that has no in-cap non-losing placement is losing.
- `depth[state] = 1 + max_piece min_winning_action depth[succ]`.
- Witness placement indices should be stored as `u8` because repo placements already fit in `u8`.
