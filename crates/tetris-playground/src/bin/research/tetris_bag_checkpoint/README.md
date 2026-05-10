# Bag Checkpoint Explorer

Starting from a board (default: empty), enumerate all 5040 permutations of the
7 standard tetrominoes (one complete bag). For each permutation, find all boards
reachable by placing the 7 pieces in that order, subject to height/hole bounds.
Then find **checkpoint** boards that every possible bag ordering can reach.

If a non-empty set of such boards exists, any such board is a candidate for a
**bag-cycle closing proof**: if the same property holds starting from each
checkpoint (every bag ordering can return to a checkpoint), then infinite play
follows by induction.

## Cell-count arithmetic

Each bag places 7 pieces = 28 cells. Line clears remove 10 cells each.
After one bag: `cells = start + 28 - 10k` where `k` is lines cleared.
This means `cells mod 10 = (start + 8) mod 10`, so each bag shifts the
residue class by +8. A full cycle back to the same residue requires 5 bags
(since 5 x 8 = 40 = 0 mod 10), visiting residue classes in order:
`R0 -> R8 -> R6 -> R4 -> R2 -> R0`.

Over 5 bags, total cells placed = 140, so total lines cleared must be exactly
14 for the cell count to return to its starting value.

## Approach: 5-step chain cycle

Build a chain of board sets S0 -> S1 -> S2 -> S3 -> S4 -> S0 where each step
verifies: "from every board in Sk and every bag permutation (5040), there
exists a valid placement path reaching some board in Sk+1."

```
S0(c=18) --bag1--> S1(c=26) --bag2--> S2(c={14,24,34})
    ^                                       |
    |           bag5                        bag3
    |                                       v
S4(c={0,10,20,30}) <--bag4-- S3(c={2,12,22,32})
```

### Key tool: `chain-targeted`

The `chain-targeted` subcommand performs one step of the chain:
1. **Phase 1**: For each (source, perm) pair, DFS to find all reachable boards
   at target cell counts. Counts hits per target board.
2. **Phase 1.5** (optional, `--verify-forward-cells`): Tests each top-K
   candidate as a SOURCE for the NEXT chain step. Filters out boards that
   can't reach the next-next cell count for all 5040 perms. Prevents
   "dead-end" boards from entering the cover.
3. **Phase 2**: Re-runs DFS for filtered top-K, tracking per-source PermBitsets.
4. **Greedy set cover**: Picks minimum boards covering all (source, perm) pairs.

### Target filtering flags

- `--max-target-holes N`: Only consider target boards with holes <= N
- `--max-target-height N`: Only consider target boards with height <= N
- `--max-target-rough N`: Only consider target boards with roughness <= N

These prevent the greedy cover from picking boards that are easy to reach but
hard to navigate FROM in subsequent chain steps.

### Forward navigability (`--verify-forward-cells`)

The chain's core problem: greedy cover picks boards optimized for reachability,
not for downstream navigability. Each step's cover becomes the next step's
sources, and "dead-end" boards cause amplification:

```
Step 0: 1 -> 2 boards
Step 1: 2 -> 15 boards
Step 2: 15 -> 67 boards
Step 3: 67 -> 861+ boards (FAILS, 9.3% uncovered)
```

The `--verify-forward-cells` flag adds Phase 1.5: before greedy cover, each
candidate target board is tested as a source for the NEXT step. Only boards
that pass (all 5040 perms can reach the next-next cell count) enter the cover.

## Experimental results

### Step 0: empty -> S0 (discover)

Using `discover --max-height 5 --max-holes 2 --top-k 5000`:

- 10.8M unique boards discovered across all 5040 perms
- **2-board minimum cover at c=18**, both h=3, holes=1
- Board 1: hash=6581569587854041553, rough=12 (covers 3504 perms)
- Board 2: hash=-9017919037081470051, rough=14 (covers 1536 new perms)

With h<=4/holes<=1: 5-board cover needed. h<=5/holes<=2 is strictly better.

### Step 1: S0(c=18) -> S1(c=26)

Using `chain-targeted --max-height 6 --max-holes 3 --target-cells 26
--max-target-holes 1 --max-target-rough 8`:

- zero_reach = 0 (all pairs reachable)
- **14-board complete cover**, all c=26, h=4, holes<=1, rough<=8
- All boards pass forward navigability toward c={4,14,24,34}

### Step 2: S1(c=26) -> S2(c={14,24,34})

Using `chain-targeted --max-height 6 --max-holes 3 --target-cells 4,14,24,34
--max-target-holes 2 --max-target-rough 10 --verify-forward-cells 2,12,22,32
--top-k 5000` from 14 step-1 boards:

- zero_reach = 0
- Phase 1.5: 3830/5000 forward-navigable (77%)
- Main cover: 82 boards, 2 uncovered pairs from 1 source
- Phase 3 recovery: 1 board added (the 2 pairs were recoverable)
- **83-board COMPLETE cover** (5 c=14, 42 c=24, 18 c=34, no c=4)
- Cover board stats: h=2-5, holes=0-2, rough=1-10

### Step 3: S2 -> S3 (c={2,12,22,32}) — DIVERGES

Using 83 step-2 boards as sources, targeting c={2,12,22,32}
with `--verify-forward-cells 0,10,20,30 --top-k 5000`:

- 418,320 total pairs (83 sources x 5040 perms)
- zero_reach = 10 (0.002%)
- Phase 1.5: 4404/5000 forward-navigable (88%)
- Main cover: 919 boards, 23,100 uncovered pairs
- Phase 3 recovery: 361K candidate boards, 1000+ recovery boards (stopped)
- **INCOMPLETE — cover diverges exponentially**

### Chain divergence analysis

The cover sizes grow exponentially through the chain:

```
Step 0: 1 source  -> 2 boards   (empty -> S0)
Step 1: 2 sources -> 14 boards  (S0 -> S1)
Step 2: 14 sources -> 83 boards (S1 -> S2)
Step 3: 83 sources -> 1000+ boards (S2 -> S3, INCOMPLETE)
```

Root cause: even with forward-navigability filtering, the greedy set cover
must handle tail-end (source, perm) pairs that are only coverable by rare,
specialized boards. These boards don't share coverage with other pairs,
preventing the cover from being small. The forward filter helps (88% vs 77%
pass rates at later steps) but cannot overcome the combinatorial explosion.

This means the chain approach with h≤6/holes≤3 constraints cannot close a
5-step cycle — step 3 alone needs 1000+ boards, making steps 4 and 5
infeasible.

### Prior approaches (all converge to empty)

#### Adversarial 7-bag GFP
- Full adversarial solver: for EVERY piece adversary picks, player must survive
- Tested h=4-6 with various hole/roughness combos
- **Every BFS gives 0% winning** — admissibility constraints help adversary

#### Non-adversarial single-tier GFP
- Player knows full permutation, iterative board pruning
- h<=2/holes<=0 safe, h<=4/holes<=2 inter: 19967->6950->...->0 (6 iters)
- **Death spiral**: each pruning round shrinks target set, causing more removals

#### Two-tier GFP (inner safe + outer buffer)
- Inner h<=3/holes<=0, outer h<=4/holes<=0, inter h<=4/holes<=1:
  478K->42K->13K->5K->1K->54->0 (inner), outer dies first at iter 5
- **Outer death cascades into inner collapse**

#### LFP expansion + GFP pruning
- Start from empty, grow set by adding boards for failing perms
- h<=3/holes<=1 safe, h<=5/holes<=2 inter: expanded to 5000, GFP: 5000->1031->60->0
- **All LFP+GFP attempts converge to empty**

### Cell-count cycle analysis

- Minimum cycle = 5 bags (need 28K = 10L for integers K,L; K=5, L=14)
- Cell count mod 10 progression: c, c+8, c+6, c+4, c+2, c+0
- Safe set MUST contain boards at multiple cell counts for transitions
- 1-bag closure doesn't exist for any tested bounded-height configuration

## Subcommands

### `discover`

Find boards reachable by all 5040 permutations from the empty board.

```sh
cargo run --release -p tetris-playground --bin tetris_bag_checkpoint -- \
  discover --max-height 5 --max-holes 2 --top-k 5000 --db /tmp/discover.db
```

### `chain-targeted`

One step of the 5-step chain with navigability filtering.

```sh
# Step 1: c=18 -> c=26, with forward check toward step 2
cargo run --release -p tetris-playground --bin tetris_bag_checkpoint -- \
  chain-targeted --max-height 6 --max-holes 3 \
  --source-hashes=6581569587854041553,-9017919037081470051 \
  --target-cells 26 \
  --max-target-holes 1 --max-target-rough 8 \
  --verify-forward-cells 4,14,24,34 \
  --top-k 200 --db /tmp/discover.db
```

### `analyze` / `analyze-targeted`

Per-cell-count coverage analysis from a given starting board.

### `chain`

Build the residue-class chain automatically (older, less flexible).

### `cycle`

Verify that a set of checkpoints forms a closed cycle.

### `gfp` / `gfp2`

Greatest-fixed-point pruning (single-tier and two-tier). All experiments
converge to empty safe set — see "Prior approaches" above.

### `lfp`

Least-fixed-point expansion + GFP pruning. Also converges to empty.

### `hubs`

Hub discovery via beam play + verification. Finds frequently-visited boards
but they don't survive GFP pruning.

### `five-bag-cycle`

Five-bag feasibility experiments are split into subcommands so the argument
surface matches the experiment being run:

```sh
# Beam-search random 5-bag sequences for return-to-empty witnesses.
cargo run --release -p tetris-playground --bin tetris_bag_checkpoint -- \
  five-bag-cycle beam --trials 1000 --beam-width 5000 --max-height 10

# PUCT tree search with a typed terminal target.
cargo run --release -p tetris-playground --bin tetris_bag_checkpoint -- \
  five-bag-cycle mcts --mcts-policy puct --mcts-iterations 50000 \
  --mcts-terminal-mode empty-or-bust --mcts-restarts 4 \
  --mcts-prior-noise 1.0 --mcts-progress-every 10000 \
  --target empty --trials 100

# Legacy independent rollout sampler, kept as a baseline.
cargo run --release -p tetris-playground --bin tetris_bag_checkpoint -- \
  five-bag-cycle mcts --mcts-policy rollout --rollouts 100000 \
  --rollout-temperatures 0.3,0.7,1.0,2.0 --trials 100

# Exact checks for a fixed permutation sequence.
cargo run --release -p tetris-playground --bin tetris_bag_checkpoint -- \
  five-bag-cycle exact --bags 3111,2091,1752,3716,936
cargo run --release -p tetris-playground --bin tetris_bag_checkpoint -- \
  five-bag-cycle prove --bags 3111,2091,1752,3716,936
```

`mcts` is a witness-search tool, not a proof artifact. Any candidate path or
checkpoint set found here still needs exact replay and closure verification.

## Algorithm

1. Generate all 7! = 5040 permutations of the 7 pieces.
2. For each (source_board, permutation), DFS with pruning (height, holes, loss)
   to collect every board reachable after placing all 7 pieces.
3. (Optional) Forward-navigability filter: test each candidate as a source for
   the next chain step, removing boards that fail any permutation.
4. Build per-board `PermBitset` (tracking which of the 5040 perms reach it).
5. Greedy set cover to find the minimum set of boards covering all perms.
6. For chaining, repeat from each cover board as source.
