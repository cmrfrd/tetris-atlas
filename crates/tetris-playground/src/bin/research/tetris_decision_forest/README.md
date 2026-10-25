# tetris_decision_forest

An evolvable **decision forest** that plays Tetris. Each tree routes a binary
encoding of the game — **board + bag state + current piece** (214 bits) —
through **boolean operations** (`And`/`Or`/`Not`/`Xor`) to a leaf that names a
**placement**; the forest plays the placement with the most tree **votes**.
Trees that ever output an invalid placement are culled. A genetic-programming
loop grows the forest up a survival curriculum toward playing **1,000,000
pieces** in a single game.

- **What it is / why**: see [`DESIGN.md`](./DESIGN.md).

All code for this system lives in this one directory. Fitness is pure
**survival** (pieces placed before topping out).

## Layout

```
tetris_decision_forest/
├── main.rs       # CLI entry (clap subcommands) + module declarations
├── encode.rs     # NUM_INPUTS = 214; encode_game — board + bag + current piece, all binary
├── tree.rs       # BoolExpr (boolean-op conditions) + DecisionTree (placement leaves)
├── forest.rs     # DecisionForest: the vote (game bits → placement) + JSON save/load
├── policy.rs     # choose_placement (encode + vote) + play_game
├── evolve.rs     # genetic operators (tree/boolean mutation, subtree crossover, selection)
├── train.rs      # survival fitness + invalid-tree autokill + generational loop + checkpoints
├── config.rs     # TrainConfig / EvolveConfig (deep/wide/many + cull defaults)
├── README.md     # this file
└── DESIGN.md     # system design
```

The board portion of the input is the engine's own `board.to_cell_array`; the
bag and current-piece bits are read from `game.bag` / `game.current_piece`.

## Build

```sh
cargo build --release -p tetris-playground --bin tetris_decision_forest
```

## Quick start

Train (writes a champion forest + a CSV log under `artifacts/`):

```sh
cargo run --release -p tetris-playground --bin tetris_decision_forest -- train \
  --run-name df-run1 \
  --population 64 \
  --generations 2000 \
  --init-trees 64 \
  --out-dir ./artifacts/decision_forest
```

Watch one game played by the saved champion on a fixed seed:

```sh
cargo run --release -p tetris-playground --bin tetris_decision_forest -- play \
  --forest ./artifacts/decision_forest/df-run1_champion.json \
  --seed 42
```

Evaluate a forest across a seed set (mean / min / max survival):

```sh
cargo run --release -p tetris-playground --bin tetris_decision_forest -- eval \
  --forest ./artifacts/decision_forest/df-run1_champion.json \
  --seeds 32 \
  --max-pieces 100000
```

Stress-test toward the goal — run the champion at the 1,000,000-piece cap:

```sh
cargo run --release -p tetris-playground --bin tetris_decision_forest -- bench \
  --forest ./artifacts/decision_forest/df-run1_champion.json \
  --seeds 16
```

Inspect a forest's structure (tree count, node counts, depths):

```sh
cargo run --release -p tetris-playground --bin tetris_decision_forest -- show \
  --forest ./artifacts/decision_forest/df-run1_champion.json
```

## Subcommands

| command | purpose |
|---------|---------|
| `train` | Evolve a population of forests up the piece-cap curriculum (fitness = survival); autokill invalid-output trees; checkpoint the champion + CSV stats. |
| `distill` | AlphaZero-style policy improvement: label states with a root-PUCT teacher (the forest's votes are the search prior, forest-policy rollouts the value), then evolve forests toward agreement with the teacher. Output resumes into `train`. |
| `play`  | Play one game with a saved forest on a given seed; print the outcome. |
| `eval`  | Play a saved (or random) forest across N seeds; print mean/min/max survival and lines. |
| `bench` | Run a saved forest at the 1,000,000-piece cap across N seeds — the goal check. |
| `show`  | Print structural stats of a saved forest. |

Run any subcommand with `--help` for its full flag list.

## Key training flags

| flag | default | meaning |
|------|---------|---------|
| `--run-name` | `df-run` | prefix for output files |
| `--population` | `64` | forests per generation |
| `--generations` | `1000` | number of generations |
| `--elite` | `8` | top forests copied unchanged each generation |
| `--seeds` | `8` | eval seeds per generation (fitness = mean survival over these) |
| `--seed-block` | `20` | freeze the eval-seed set for this many generations before rotating — a stationary objective within each block (per-generation rotation makes champion acceptance a race against seed noise), no overfitting across blocks |
| `--tournament-k` | `5` | tournament size for parent selection |
| `--init-trees` | `64` | decision trees per forest at initialization (many) |
| `--init-tree-depth` | `6` | depth of randomly *generated* trees — kept small; trees grow via evolution (random generation is a supercritical branching process, so big init depths would blow up tree size) |
| `--max-tree-depth` | `24` | structural cap on decision-tree depth: grow mutations stop at this depth (deep) |
| `--max-cond-depth` | `6` | structural cap on boolean-condition depth per branch (wide). GROWTH cap only: random generation always emits conditions of depth ≤ 4 (`COND_GEN_DEPTH_CAP`) — condition generation is a supercritical branching process, so generating at e.g. depth 16 makes ~100-node random formulas that bloat clone/eval cost without being selected for; depth is earned via grow mutations |
| `--max-trees` | `256` | cap on trees per forest (many) |
| `--min-trees` | `32` | regrowth floor: after the autokill, offspring are topped back up to at least this many trees with fresh piece-gated (100 %-valid) trees (`0` disables) |
| `--min-valid-rate` | `1.0` | autokill: cull trees whose valid-output rate over the per-generation sample game falls below this (`1.0` = cull any tree that outputs an invalid placement during the sample; `0` disables). A forest is never emptied — the most-valid tree survives |
| `--log-every` | `1` | print a progress line every N generations (default: every generation) |
| `--snapshot-every` | *(off)* | save a `<run>_gen_<N>.json` champion snapshot every N generations; omit to disable (the write pauses training briefly) |
| `--start-cap` | `200` | initial per-game piece cap (curriculum start) |
| `--seed` | `42` | master RNG seed for the run |
| `--out-dir` | `./artifacts/decision_forest` | where champion JSON + CSV go |
| `--resume` | *(none)* | resume from a champion JSON, seeding the population |

The curriculum escalates the piece cap `200 → 1k → 10k → 100k → 1M`
automatically once the champion survives the current cap on every eval seed.

### On `--min-valid-rate` and piece-gated generation

The default `1.0` culls any tree that outputs a placement invalid for the
current piece during the per-generation sample game (`cull_sample_pieces`
decisions). A forest is never emptied: if every tree falls below the threshold,
the single most-valid tree survives.

For the aggressive default to be survivable, every randomly *generated* tree
(population init, tree-add mutation, regrowth) is **piece-gated**: its root
chain branches on the 7 current-piece one-hot bits, and the subtree under each
gate votes only placements *of that piece* — 100 % valid by construction (the
engine's legal set is exactly the current piece's placement table), while the
board logic under the gates stays fully random. Leaf-retargeting mutations
also stay within the leaf's piece 80 % of the time. Mutations can still break
a gate — the cull then removes that tree — and `--min-trees` regrows the
forest with fresh gated trees, so under cull-at-1.0 the forest stays
many-treed instead of ratcheting down to a single survivor. (Without gated
generation, cull-at-1.0 demonstrably collapses every forest to 1 tree within
~1 generation and evolution stalls.)

## Anti-stagnation machinery

Early versions plateaued hard (frozen champion, #trees pinned at the
`--min-trees` floor). Five mechanisms now keep the search moving; all preserve
survival-only fitness, cull-at-1.0, and forest-level voting:

- **Same-seed champion gate** — each generation the incumbent champion is
  re-scored on the *current* seed set and replaced only if the generation's
  best beats it on those identical seeds. (Comparing against a historical
  high-water mark set on different seeds froze the champion ~3.4 σ of seed
  noise above its true skill for thousands of generations.)
- **Seed blocks** (`--seed-block`) — a stationary objective to climb within
  each block.
- **Whole-tree crossover** — recombination swaps *whole trees* between
  parents, never subtrees (a subtree graft crossed piece gates 6/7 of the
  time, manufacturing invalid votes that the autokill punished by destroying
  the evolved recipient tree).
- **Duplicate-and-diverge tree-add** — tree-add usually copies an evolved
  tree and mutates the copy (near-neutral gene duplication, the fixation path
  above the `--min-trees` floor) instead of injecting a random tree.
- **Size-scaled mutation + silent regrowth** — mutation count grows with
  forest size (one node change in a 32-tree plurality is almost always
  behaviorally silent), and regrowth injects mostly-abstaining
  (vote-neutral) specialist trees rather than plurality-randomizing voters.

## PUCT teacher distillation (`distill`)

The AlphaZero improve→project cycle with the forest in the network's role,
entirely at TRAINING time — the played artifact stays a pure state→move
boolean circuit (no search at inference, so the proof story and the fast
`play`/`eval` path are untouched):

1. **Improve**: sample states by playing the prior forest; at each state run
   a root-PUCT search — the forest's vote tally (mixed 3:1 with uniform) is
   the prior `P(a)`, rollouts following the forest policy give survival
   values, and `Q + c·P·√N/(1+N)` allocates the simulation budget (after a
   round-robin first pass so every action carries a real Q). Each simulation
   reseeds the game RNG **and reshuffles the remaining bag order** — board,
   bag membership, and current piece are preserved (what the student's bits
   expose); the hidden piece order is resampled, so labels leak no hidden
   information. The most-visited placement is the teacher's label.
2. **Project**: evolve forests (same GP machinery, same cull/regrowth) whose
   fitness is **agreement** with the teacher's labels — a deterministic,
   noise-free objective.
3. **Iterate**: feed the distilled champion back into survival training.

```sh
# one improvement round on the current survival champion
cargo run --release -p tetris-playground --bin tetris_decision_forest -- distill \
  --run-name df-run1 --prior ./artifacts/decision_forest/df-run1_champion.json \
  --states 1024 --sims 256 --out-dir ./artifacts/decision_forest

# then continue survival training from the improved policy
cargo run --release -p tetris-playground --bin tetris_decision_forest -- train \
  --run-name df-run1b --resume ./artifacts/decision_forest/df-run1_distilled.json \
  --out-dir ./artifacts/decision_forest
```

Key flags: `--states` (dataset size), `--sims` (PUCT budget per state),
`--rollout-depth` (value horizon), `--c-puct` (exploration), `--game-seeds`
(sampling games), plus the usual population/generations/elite knobs. The gen-0
`agreement` line tells you how much improvement the search found over the
prior (e.g. 0.46 = the teacher disagrees with the forest on half the states —
plenty to learn).

## Outputs (under `--out-dir`)

- `<run-name>_champion.json` — best forest so far (overwritten whenever a
  generation's best beats the incumbent on the same seed set, and once more
  at shutdown).
- `<run-name>_gen_<NNNNNNNN>.json` — periodic snapshots, only with
  `--snapshot-every N` (off by default: the synchronous write pauses training).
- `<run-name>_log.csv` — per-generation: generation, cap, best/mean survival,
  champion min/mean/max pieces, mean lines, tree/node counts. A progress line
  also prints to stdout every `--log-every` generations (default: every one).

`eval` and `bench` report throughput (`pieces/sec`) alongside survival.

## Totality and abstention

Individual **trees** may abstain (`Abstain` leaf — "no recommendation"), which
is what lets trees specialize: vote only on boards their conditions recognize.
Abstention is never an invalid output, so the autokill does not punish silence.
The **forest**, however, always answers: `vote` returns no placement only for an
empty legal set, which cannot occur in a live game — if every tree abstains (or
votes only illegal ids), the lowest-index legal placement is played. There is no
reachable "no answer" state.

## Determinism

A game is fully determined by `(seed, forest)`: vote ties break by lowest
placement index and all randomness in training flows through explicitly-seeded
`SmallRng`. Re-running `play`/`eval`/`bench` on the same inputs reproduces
results exactly.

## Tests

```sh
cargo test -p tetris-playground --bin tetris_decision_forest
```
