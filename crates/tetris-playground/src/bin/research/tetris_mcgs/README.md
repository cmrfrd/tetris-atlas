# Monte Carlo Graph Search (MCGS) Cycle Finder

**Level:** Attempting L2/L3 (adversarial cycle discovery)
**Status:** New implementation

## Strategy

Explores the Tetris game graph using AND-OR Monte Carlo Graph Search to
discover closed cycles proving infinite play.

### Graph vs Tree

Standard MCTS builds a **tree** — identical states reached via different paths
create separate nodes (37M+ nodes → OOM). MCGS builds a **graph** — one node
per unique `(board, bag_state)` with edges that can form cycles:

```
MCTS (tree):                    MCGS (graph):
     root                            root
    / | \                           / | \
   A  B  C  ← duplicates          A  B  C
  /|  |  |\                        \ | /
 D E  F  G H ← more dupes          D   E  ← shared nodes
                                    |  ↗
                                    F-'    ← cycle!
```

Children are materialized lazily. Expansion scores legal placements and stores
their successor `StateKey`, but it does not allocate the child node until the
sampler actually selects that placement. This keeps the node arena closer to
visited graph structure instead of all widened successor possibilities.

### Game graph structure

Each node is identified by `NodeKey`:

- **AND nodes** `(board, bag)`: adversary picks which piece to draw.
  Uses `bag.iter_next_states()` for children (auto-refills bag when empty).
- **OR nodes** `(board, bag, piece)`: player picks a placement.
  Uses `TetrisPiecePlacement::all_from_piece(piece)` for children.

### Cycle detection

At bag refill boundaries, when an OR placement leads to `(board, full_bag)`
that already exists in the graph with visits > threshold and positive value,
the search detects a **cycle close** (reward 1.0).

The sampler also tracks a cheap closedness proxy over the current pool/core. For
each full-bag pool board, there are seven required adversarial first-piece
responses. A response is counted covered when sampling has found at least one
path starting with that piece that returns to another pool board. This is not a
certificate, but it gives a single progress scalar for whether the sampled core
is becoming less open.

Candidate verification runs in the background. The sampler snapshots the current
cycle candidates, launches the verifier, and continues MCGS iterations while the
5040-permutation check runs. If another verification interval arrives while a
previous verifier is still active, the new launch is skipped rather than
blocking sampling.

### PUCT selection

- **AND nodes** (adversary): UCB with negated Q — explores the *hardest*
  pieces (lowest value for player)
- **OR nodes** (player): UCB over value plus recurrence minus novelty — explores
  placements that keep returning to known useful graph regions instead of only
  maximizing survival depth
- Priors: uniform at AND level, softmax over board heuristic at OR level

### Hard-dead propagation

MCGS now has two feedback channels:

- **Soft value backup:** sampled paths update visit counts and value estimates.
- **Hard death backup:** impossible or out-of-envelope responses are marked dead
  and propagated through predecessor edges.

A placement edge is hard-dead if it directly loses, violates the configured
admissibility envelope (`--max-height`, `--max-holes`, `--max-roughness`), or
leads to a state already proven dead. A piece group is hard-dead when all known
responses for that adversarial piece are dead and no pending responses remain.
Because the adversary can choose that piece, the whole state is then hard-dead,
and predecessor edges pointing at it are blocked.

This keeps sampling pressure on recurrent, reusable structure:

```text
value(state) ~= min_piece max_placement(
  value(next)
  + recurrence_reward(next)
  - novelty_cost(next)
  - hard_death_penalty
)
```

The "kernel" is not imposed as a budget. It is the high-value recurrent core
that emerges from the graph after dead edges are blocked and novel one-off
successors remain costly.

## Opponent model

**Adversarial** — AND nodes model worst-case piece selection from the 7-bag.

## Constraints

- Uses canonical game engine types (`TetrisBoard`, `TetrisPieceBagState`)
- Exact state deduplication (no hash collisions)
- Configurable `--max-height`, `--max-holes`, `--max-roughness`
- Hard `--max-nodes` cap prevents OOM (default 5M)
- `--max-bags` controls search depth in bag cycles
- `--db` persists node metadata to SQLite and `--db-flush-every` controls
  periodic flushes; edge groups are recomputed on reload

## Usage

```sh
# Default: 500K iterations, h≤6, holes≤3, 3 bags deep
cargo run --release -p tetris-playground --bin tetris_mcgs

# Wider exploration
cargo run --release -p tetris-playground --bin tetris_mcgs -- \
  --max-height 6 --max-holes 3 --max-roughness 10 \
  --iterations 1000000 --max-bags 5 --max-nodes 10000000

# SQLite-backed persistence with periodic flushes
cargo run --release -p tetris-playground --bin tetris_mcgs -- \
  --db artifacts/databases/mcgs.sqlite --db-flush-every 50000

# More exploration vs exploitation
cargo run --release -p tetris-playground --bin tetris_mcgs -- \
  --exploration 2.0 --prior-temperature 3.0

# Stronger recurrent-core pressure
cargo run --release -p tetris-playground --bin tetris_mcgs -- \
  --recurrence-weight 0.4 --novelty-cost 0.2

# Log candidate verification progress every 250 permutation checks
cargo run --release -p tetris-playground --bin tetris_mcgs -- \
  --verify-every 500000 --verify-log-every 250
```

## Output

- **graph_nodes**: unique states explored (should grow sublinearly vs iterations)
- **root_value**: average value at root (positive = promising)
- **cycle_closes**: number of times a cycle was detected
- **closure/wclosure**: unweighted and start-weighted first-piece response
  coverage for the current pool/core
- **covered/open/core**: covered adversarial first-piece responses, missing
  responses, and active core boards used in the closure proxy
- **dead_states/dead_edges/dead_piece_groups**: hard-pruned graph structure
- **condition_violated/lost_placements**: placements blocked by the configured
  admissibility envelope or immediate top-out
- **verification progress**: when candidate verification runs, `--verify-log-every`
  prints global permutation-check progress and per-candidate completion lines;
  verification is asynchronous, so regular sampler logs continue while it runs
- **cycle candidate boards**: boards that appeared at cycle-close points
- **top full-bag AND nodes**: most-visited states at bag boundaries
