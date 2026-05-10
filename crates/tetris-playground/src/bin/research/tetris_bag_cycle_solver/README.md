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

- `TetrisBoard` for exact occupancy, gravity, line clears, loss detection, and hashing.
- `TetrisPieceBagState` for the 7-bit remaining-bag mask and next-piece enumeration.
- `TetrisPiecePlacement::all_from_piece(piece)` for legal final-placement enumeration.
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
a `winning=YES` result proves a reset strategy that never leaves the configured
admissible region before the final empty reset.

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

## Result Meaning

`winning=YES` means the solver found a strategy that forces an empty-board reset
within the configured number of bag cycles under the configured admissibility
caps. Chaining that reset gives an infinite-play proof for the loop state.

`winning=NO` does not prove the reset theorem false. It only means no strategy
was found within the current horizon, board caps, search ordering, and strict
empty-reset target.

## Current Limitations

- The target is very strict. Returning to the empty board is much narrower than
  returning to any board in a closed safe set.
- The solver reports a single root boolean, so failed runs do not yet say whether
  the approach is 5%, 40%, or 95% of the way through the current proof ladder.
- Full witness export to a standalone Atlas format is not implemented yet, but
  successful runs now keep an in-memory witness policy and verify it before
  reporting a clean proof.
- The DFS root solve still short-circuits on the first failed adversary branch, which is efficient
  for proving `NO` under current ordering but weak for diagnosis.
- Root diagnostics now evaluate every root piece branch, but deeper failed-frontier
  diagnostics are still limited.
- Transitions are cached as `(board, piece) -> successors`, but the DFS is still
  single-threaded.
- A bottom-up retrograde implementation exists behind `--algorithm retrograde`,
  but dependency-driven reverse updates and parallel expansion are not implemented.

## Incremental Viability System

This experiment should be treated as a measurable system, not a one-shot
`YES/NO` gamble. The goal is to move a fixed score upward through repeated
changes:

```text
33% -> 40% -> 70% -> 99% -> 100%
```

Use a versioned proof ladder. Each scenario is a fixed tuple:

```text
(bag_cycles, max_height, max_holes, max_roughness, max_count)
```

Start with a small ladder and only add harder scenarios when the previous ones
are stable:

```text
L0: cargo check and unit tests pass
L1: 5 bags, height 2, holes 0
L2: 5 bags, height 3, holes 0
L3: 5 bags, height 4, holes 0
L4: 5 bags, height 4, holes 1
L5: 10 bags, height 4, holes 0
L6: relaxed closed-set reset instead of empty reset
```

For every run, write a JSON summary under `artifacts/output/` with:

```text
commit hash
command and config
hardware and rustc -Vv
winning boolean
elapsed seconds
nodes/sec
boards interned
memo entries
cell/admissibility/death prune counts
root piece coverage
adversary obligation coverage
first failing state and piece
hardest frontier boards
```

The main score should not be only "how many scenarios passed." Failed scenarios
need partial progress metrics:

```text
root_piece_coverage =
  pieces from the full bag with at least one proven placement / 7

adversary_obligation_coverage =
  proven (state, next_piece) obligations / total diagnosed obligations

frontier_repair_rate =
  failed obligations that become proven after the latest change / old failures
```

That gives us an honest improvement loop: each code or heuristic change should
show which obligations became solvable, which stayed blocked, and whether the
search got faster or only searched more.

## Top 3 Improvements

### 1. Add witness extraction, verification, and progress diagnostics

The highest-priority improvement is observability. The solver should store the
winning placement for every proven OR obligation and export a witness policy.
Then an independent verifier should replay that policy over the full adversarial
tree and confirm:

```text
for every reachable witness state
for every legal next piece in the bag
the stored placement is legal
the successor is either the final empty board or another verified witness state
```

For failed runs, add a diagnostic mode that does not stop at the first failed
AND branch. It should count branch coverage and emit the smallest or most common
frontier blockers. This is what turns the project into an incremental system:
we can measure that a change improved a scenario from, for example, 33% to 40%
even when it still does not prove the root.

### 2. Replace recursive DFS with a cached transition universe plus retrograde solve

The current DFS repeatedly computes placements and learns only the states it
happens to visit under its current ordering. For the finite reset theorem, a
stronger structure is available:

```text
Win[0] = { empty board at horizon }
Win[t + 1] contains state S iff
  for every next piece p in bag(S)
  there exists a placement a
  such that successor(S, p, a) is in Win[t]
```

Build or lazily cache board-piece successors once, deduplicate successor boards,
and solve this recurrence bottom-up or with dependency-driven reverse updates.
This gives better reuse, easier parallelism, exact coverage counts by depth, and
natural witness extraction.

### 3. Generalize from empty reset to a closed bag-boundary safe set

Empty reset is elegant but probably too restrictive. A more viable proof target
is a closed set of low, clean bag-boundary boards:

```text
S is a safe set iff for every board in S and every full-bag adversarial sequence,
the player can return to some board in S without leaving the admissible region.
```

This keeps the proof-by-construction shape while giving the solver more room.
The system can start with `S = { empty }`, then grow `S` with candidate boards
found by search, beam policies, or success-set solving. The incremental metric
becomes the size and closure rate of `S`, not just whether one empty reset exists.

## Suggested Next Implementation Order

1. Export the verified witness policy to a compact artifact that can be replayed
   outside the process.
2. Extend diagnostics below the root: persist failed frontier obligations and
   track repair rate between runs.
3. Add dependency-driven reverse updates to the retrograde solver so it avoids
   full finite-universe scans on larger caps.
4. Parallelize transition expansion and, where deterministic, branch evaluation.
5. Turn `low-clean` from a relaxed terminal predicate into a true closed-set
   fixed-point solver.
