# Tetris Cycle Atlas

`tetris_cycle_atlas` is a research solver for finding a closed safe set of
Tetris boards at 7-bag refill boundaries. It tries to reduce the adversarial
Tetris state space from `(board, bag_state, current_piece)` to just `board` by
only asking whether a board can return to the safe set after one or more
complete bag cycles.

The intended milestone is M2/M3-style evidence:

- M2: find a non-empty closed set of safe boards.
- M3: prove the empty board can reach that closed set, or that the empty board
  itself is in the set.

If the empty board is in the final safe set, the result is a constructive
infinite-play certificate for the bounded model represented by the solver
constraints.

## Core Idea

At a bag boundary, the bag contains all 7 tetrominoes. The adversary chooses the
order of pieces within the bag. The player sees each piece and chooses a legal
placement. A board is considered safe if:

1. For every possible next piece remaining in the bag,
2. there exists at least one player placement,
3. such that this remains true until `bag_cycles` complete bags have been
   consumed,
4. and the resulting board is back in the safe set.

This is an AND-OR game:

- AND nodes are adversary piece choices. Every piece choice must be survivable.
- OR nodes are player placement choices. At least one placement must work.

The outer algorithm is a greatest fixed point (GFP). It starts by assuming every
discovered board is safe, removes boards that fail the AND-OR check, and repeats
until no more boards are removed.

## State Sets

The solver uses two board admissibility envelopes.

`safe_set`
: Boards allowed to appear at cycle boundaries. These are the boards that can
  remain in the final atlas.

`intermediate`
: Boards allowed during the pieces inside a bag cycle. This can be looser than
  the safe set, allowing temporary messier or taller boards as long as the board
  returns to the safe set at the boundary.

The split is the main reason this approach is interesting: it asks for clean
cycle-boundary states without requiring every intermediate placement to also be
cycle-boundary-clean.

## Build Phases

### 1. Discovery

`BoardUniverse::discover` does a BFS from the empty board. It tries every piece
and every placement and interns boards that satisfy `safe_set_admissibility`.

The discovered universe is the candidate safe set for GFP. If `max_boards` is
hit, discovery is truncated and the result is not complete.

### 2. GFP

`run_gfp` initializes:

```text
safe_set = all discovered boards
```

For each board in the current safe set, it runs `inner_check` from a full bag
for `bag_cycles`. Boards that fail are removed. The process repeats until an
iteration removes zero boards.

The inner check memoizes `(board_id, bag_state, cycles_remaining)`.

Intermediate boards are inserted into a registry shared across the GFP run.
Those intermediate boards can be used during search, but only original universe
boards can be accepted as terminal safe-set boards.

### 3. Verification

If the GFP result is non-empty, verification re-runs the AND-OR check with fresh
memos over the final safe set. This is intended to catch implementation mistakes
or stale memo assumptions.

### 4. Atlas Save

The saved atlas contains:

- all discovered universe boards,
- the final safe board IDs,
- the `bag_cycles` horizon used to build the atlas.

During play, the atlas does not store a full policy. Instead, online play
re-solves the AND-OR choice for the current bag cycle using the saved safe set.

## Running

Fast diagnostic run:

```sh
target/release/tetris_cycle_atlas build \
  --bag-cycles 3 \
  --max-height 4 \
  --max-holes 1 \
  --max-roughness 5 \
  --max-count 24 \
  --inter-max-height 4 \
  --inter-max-holes 1 \
  --inter-max-roughness 6 \
  --inter-max-count 26 \
  --max-boards 250000 \
  --max-registry-boards 1000000 \
  --no-verify \
  --no-summary
```

Looser diagnostic run:

```sh
target/release/tetris_cycle_atlas build \
  --bag-cycles 1 \
  --max-height 4 \
  --max-holes 2 \
  --max-roughness 8 \
  --max-count 40 \
  --inter-max-height 5 \
  --inter-max-holes 3 \
  --inter-max-roughness 10 \
  --inter-max-count 45 \
  --max-boards 2000000 \
  --max-registry-boards 5000000 \
  --no-verify \
  --no-summary
```

The second profile has been observed to produce transient `surviving=2` during
iteration 1, but those candidates were removed in iteration 2 and the registry
cap was hit. That is not a certificate.

## Reading GFP Logs

Example log fields:

```text
[gfp] iter 1 progress: 576/2000000 checked current_board=1240314 surviving=2 removed=574 ...
```

`checked`
: Number of root safe-set boards checked in the current GFP iteration.

`surviving`
: Number of checked root boards that satisfy the full adversarial AND-OR test
  against the current safe set. This is not a count of promising partial paths.

`removed`
: Number of checked root boards scheduled for removal in this GFP iteration.

`depth`
: Deepest searched ply over the configured horizon. For `bag_cycles=3`, full
  depth is `21/21`.

`registry`
: Number of universe plus intermediate boards interned during GFP.

`cap_hits` / `reg_prunes`
: Number of times the intermediate registry cap prevented inserting a board.
  If this is non-zero, the run is conservative and incomplete.

`terminals`
: Successful terminal checks over total terminal checks. A terminal success
  means the search reached a universe board currently in the safe set at the
  configured cycle boundary.

`memo_true` / `memo_false`
: Memoized inner game states proven winning or losing against the current
  GFP safe set.

## Current Empirical Status

So far, this approach has not produced a stable non-empty final safe set under
the tested low-height profiles.

Observed behavior:

- Tight profiles finish quickly but converge to an empty safe set.
- Looser profiles can produce transient survivors in iteration 1.
- Those transient survivors have not remained closed after later GFP
  iterations.
- Wider intermediate envelopes often hit `max_registry_boards`, making empty
  results inconclusive.

The important distinction is:

```text
surviving > 0 during iteration 1
```

is only evidence that some boards can survive against the optimistic initial
candidate set.

```text
final safe_set > 0 with cap_hits = 0 and verification passed
```

is the first meaningful success condition.

## Why This Approach Is Struggling

The cycle-boundary abstraction is elegant, but it appears too brittle with the
current invariant shape.

Likely blockers:

1. The safe-set envelope may be too clean.
   Low height, low roughness, low holes, and low count make terminal acceptance
   difficult after adversarial bags.

2. Loosening intermediates explodes the registry.
   The player needs temporary messy boards, but allowing them creates millions
   of intermediate states quickly.

3. Board-only cycle boundaries may discard important structure.
   The same board can have very different strategic value depending on the
   bag state, recent cycle path, or whether specific cleanup pieces remain.

4. The GFP is all-or-nothing.
   A board must satisfy every adversarial piece order. Partial progress is not
   retained unless it closes under the final safe set.

## Lessons For The Next Approach

The next approach should preserve more actionable structure while still trying
to avoid the full raw state space.

Promising directions:

1. Track `(board, bag_state)` for a smaller targeted region.
   This is larger than board-only, but avoids forcing all reasoning to happen
   only at bag boundaries.

2. Search for small constructive cycles directly.
   Instead of GFP over millions of boards, synthesize a small policy loop and
   then prove closure.

3. Use a staged invariant.
   Separate states into roles such as setup, burn, cleanup, and return, instead
   of requiring one uniform safe-set envelope.

4. Add guided candidate generation.
   Use beam search or heuristic policies to find boards that empirically return
   to clean states, then attempt to certify only that candidate region.

## Success Criteria

A run is worth treating as evidence only if it reports:

- final `safe_set > 0`,
- `cap_hits = 0`,
- verification passed,
- full command line and constraints,
- commit hash,
- runtime and hardware notes.

For the strongest result:

- `empty_in_safe = true`, or
- a separate reachability bridge from the empty board into the safe set.

