# `tetris_bag_cycle_solver`

This binary is a proof-search experiment for a strict bag-cycle reset strategy.

The target theorem is:

```text
Starting from the empty board and a full 7-bag, the player can force a return
to the empty board after N complete bags, no matter which remaining piece the
adversary draws at each step.
```

If this is true for some `N`, then infinite play follows by repeating the same
reset strategy forever. This is a small, concrete route to M2/M3-style proof
work: instead of proving a huge global Atlas immediately, prove one closed loop
whose loop state is exactly `(empty_board, full_bag)`.

## Current Approach

The solver runs a memoized depth-bounded AND-OR search:

```text
state = (board_id, bag_mask, cycles_remaining)

AND node:
  the adversary chooses any piece still in the current bag
  every such piece branch must be winnable

OR node:
  the player chooses one legal final placement for that piece
  at least one placement must lead to a winnable child state

terminal target:
  after the last piece of the last bag cycle, the board must be empty
```

The code uses the canonical engine primitives directly:

- `TetrisBoard<10, 20>` for exact occupancy, gravity, line clears, loss detection, and hashing.
- `TetrisPieceBagState` for the 7-bit remaining-bag mask and next-piece enumeration.
- `TetrisPiecePlacement::<10>::all_from_piece(piece)` for legal final-placement enumeration.
- `TetrisBoard::apply_piece_placement` for concrete board transitions.

Boards are interned in `BoardCache`, and search results are memoized with a typed
`MemoKey`:

```text
(board_id, bag_mask, cycles_remaining)
```

## Why `--bag-cycles` Must Be a Multiple of 5

Each tetromino adds 4 cells. One complete bag adds 28 cells. Returning to the
empty board after `N` bags requires clearing exactly:

```text
28N / 10 = 14N / 5
```

complete lines. That value must be an integer, so `N` must be a multiple of 5.

## Search Pruning

The current implementation has two pruning layers.

Cell-count feasibility:

```text
(current_filled_cells + 4 * remaining_pieces) % 10 == 0
```

If this fails, the board cannot be empty at the horizon because line clears remove
10 cells at a time.

Board admissibility:

```text
height(board)    <= --max-height
holes(board)     <= --max-holes
roughness(board) <= --max-roughness
count(board)     <= --max-count
```

These caps keep the state space bounded. They are proof assumptions for the run:
a verified positive result in `empty` target mode certifies a reset strategy within that
region. A negative result applies only to this horizon and carrier.

## Running

Smoke check that the binary builds:

```sh
cargo check -p tetris-playground --bin tetris_bag_cycle_solver
```

Strict exploratory runs:

```sh
cargo run --release -p tetris-playground --bin tetris_bag_cycle_solver -- \
  --bag-cycles 5 --max-height 2 --max-holes 0

cargo run --release -p tetris-playground --bin tetris_bag_cycle_solver -- \
  --bag-cycles 5 --max-height 3 --max-holes 0

cargo run --release -p tetris-playground --bin tetris_bag_cycle_solver -- \
  --bag-cycles 5 --max-height 4 --max-holes 0
```

The output currently reports:

```text
winning
boards_interned
memo_entries
nodes_explored
memo_hits
memo_true
memo_false
cell_prunes
target_hits
depth_reached
lost_prunes
admissibility_prunes
transition_cache
policy_entries
root_status
root_coverage
verification
verified_empty_paths
time
```

In `empty` mode, `target_hits` is the number of terminal target witnesses selected
during search. On a winning run, `verified_empty_paths` is the replay-verified
count of distinct terminal witness branches in the final policy that reach the
empty board. The solver does not enumerate every possible player path; it stores
one witness placement per proven obligation.

The periodic DFS progress line also reports partial completion signals:

```text
depth=34/35
root=[O:+ I:* S:. Z:. T:. L:. J:.]
root_done=1/7
target_hits=123
memo_true=...
memo_false=...
branch_win=...
placements_scanned=...
cache_hits=...
cache_misses=...
```

These are lower-bound progress diagnostics, not a percent-complete proof. DFS
does not know the total state space before it searches it.

Useful implementation flags:

```sh
# Default: memoized DFS with root diagnostics and JSON summary output.
cargo run --release -p tetris-playground --bin tetris_bag_cycle_solver -- \
  --algorithm dfs --target-mode empty --bag-cycles 5 --max-height 4 --max-holes 0

# Optional finite-universe bottom-up solver. This is exact for the generated
# capped universe but can grow much faster than DFS.
cargo run --release -p tetris-playground --bin tetris_bag_cycle_solver -- \
  --algorithm retrograde --target-mode empty --bag-cycles 5 --max-height 4 --max-holes 0

# Relaxed terminal target. This is useful for measuring viability, but by itself
# is not yet a closed-set infinite-play proof.
cargo run --release -p tetris-playground --bin tetris_bag_cycle_solver -- \
  --target-mode low-clean --bag-cycles 1 \
  --max-height 4 --max-holes 0 \
  --closed-set-max-height 4 --closed-set-max-holes 0
```

## Result meaning

In `--target-mode empty`, a positive result with clean witness verification establishes an
online reset strategy at the configured horizon and caps. Repeating a verified reset from the
empty/full-bag root gives an infinite-survival policy. The stored witnesses must cover every
legal adversarial draw; a single successful path is insufficient.

In `--target-mode low-clean`, success establishes the configured finite-horizon terminal
predicate. It does not show that the terminal set is closed under another bag.

A completed negative solve excludes a strategy only for the selected horizon, carrier, and
target. It does not exclude all canonical survival policies. Search/progress coverage is
specific to its reported denominator and is not a proof-completion percentage.

## Implementation scope

The default DFS memoizes states and caches board/piece transitions. A bottom-up implementation
is available with `--algorithm retrograde`. Positive results retain an in-memory witness policy
and run concrete replay verification; exported summaries include verification results.

DFS evaluates root piece branches for diagnostics by default; `--no-diagnostics` disables it. Independent policy
export/replay and closure across a general low-clean terminal set remain separate from merely
finding a finite-horizon candidate. Use `--help` for current diagnostic and summary options.
