# Tetris Decision Forest — System Design

A deterministic policy over board, bag-membership, and current-piece bits. Evolution searches for
forests that survive longer; optional PUCT distillation improves the training labels. Played
policies perform tree evaluation and voting, without search at inference.

Usage and CLI flags are in [README.md](README.md). The repository's
[proof standard](../../../../../../AGENTS.md#mission-and-proof-standard) distinguishes finite play
from certified infinite survival.

## 1. Policy inputs and outputs

`encode_game` produces 214 binary inputs for standard 10×20 Tetris:

| Indices | Meaning |
|---|---|
| `0..200` | Playable board occupancy in `Major::Col` order: `col * 20 + row` |
| `200..207` | Remaining-bag membership, in O/I/S/Z/T/L/J order |
| `207..214` | Current-piece one-hot encoding in the same order |

The encoder reads the board through `to_cell_array`, and the private bag/current piece through
`bag()` and `current_piece()`. The policy sees bag membership, not the packed future draw order
or RNG state. Bag information narrows possible future draws; it does not reveal their order.

A tree evaluates Boolean conditions over these bits and reaches either a placement-ID leaf or an
abstention. A forest tallies votes for fitting placements of the current piece. Ties, including an
all-zero legal tally, choose the lowest placement ID. An empty legal set produces no answer; the
standard engine's per-piece catalogs are nonempty.

A fitting placement may still top out. Voting guarantees piece/width validity, not survival.

## 2. Representation and format

`BoolExpr` supports input-bit reads, constants, `Not`, `And`, `Or`, and `Xor`.
`DecisionTree` contains placement leaves, `Abstain`, and conditional branches.
`DecisionForest` stores the versioned collection of trees as JSON.

The genome uses boxed trees. `CompiledForest` flattens conditions and decision nodes into
contiguous arenas for gameplay and fitness evaluation, with equivalence tests against the boxed
reference. This changes evaluation representation, not the policy's votes or checkpoint format.

`FOREST_FORMAT_VERSION` guards saved forests. Changes to the input layout, node encoding, or
placement IDs require explicit format compatibility handling or a version bump. Placement IDs
come from `TetrisPiecePlacement<10>`, with symmetric rotations sharing canonical IDs.

## 3. Generation, mutation, and validity

Fresh trees are piece-gated: current-piece tests route to subtrees whose placement leaves belong
to that piece. Conditions under the gates remain board-dependent. Abstention allows a tree to
specialize without contributing an invalid vote.

Random generation starts small. Growth respects configured decision-tree and condition-depth
caps. Evolution uses leaf/condition mutation, grow/prune operations, tree addition/removal,
and whole-tree crossover. Validity-aware mutation uses the governing piece when selecting new
leaves and avoids targeting the piece-gate spine where possible.

The culling sample measures valid-output rates over a finite game. Trees below `min_valid_rate`
are removed, retaining at least one tree. At the default threshold of `1.0`, any sampled invalid
output lowers that tree below the threshold, but culling is not exhaustive validation of the
policy over all reachable states. Regrowth restores the configured floor using gated specialist
trees, often with abstaining leaves.

The forest always filters votes through the current legal placement catalog, including when an
individual tree produces an invalid ID.

## 4. Survival training

Each generation:

1. Evaluate forests on a shared seed set up to the current piece cap. Fitness is mean survival.
2. Keep elites and select parents for reproduction.
3. Apply genetic operators, cull sampled invalid voters, and regrow offspring as configured.
4. Re-score the incumbent champion on the same seeds before accepting a replacement.
5. Write champion checkpoints and CSV statistics.

Evaluation seeds remain fixed within `seed_block` generations, then rotate. This keeps the
comparison objective stationary within a block; held-out seeds remain necessary to measure
generalization. Mutation effort scales with forest size, and tree addition can duplicate and
mutate an existing tree rather than always generating a fresh voter.

The curriculum raises the piece cap from 200 through 1,000, 10,000, 100,000, and 1,000,000 when
the champion reaches the current cap on every evaluation seed. Reaching a cap is a finite
empirical result, not an infinite-play milestone by itself.

## 5. PUCT distillation

`distill` creates a teacher-labeled dataset using root-PUCT search:

- The prior mixes forest votes over legal placements with a uniform component.
- Rollouts following the forest policy estimate finite-horizon survival.
- Simulations resample hidden RNG and remaining-bag order while preserving the visible board,
  bag membership, and current piece.
- The most-visited root action becomes the label.

Forest evolution then maximizes agreement with the fixed labels. A distilled forest can resume
ordinary survival training. Teacher agreement is a training objective; improved survival needs
separate evaluation, and certification still needs concrete closure checks.

## 6. Modules

| File | Responsibility |
|---|---|
| `main.rs` | CLI for train, distill, play, eval, bench, and show |
| `encode.rs` | Fixed binary state encoding |
| `tree.rs` | Boolean expressions, decision trees, generation, validity |
| `forest.rs` | Genome, legal-vote selection, JSON format |
| `compiled.rs` | Arena-based evaluation of the same policy |
| `policy.rs` | Placement selection and bounded gameplay |
| `evolve.rs` | Selection, crossover, mutation, regrowth |
| `train.rs` | Survival evaluation, culling, curriculum, checkpoints |
| `puct.rs` | Search teacher |
| `distill.rs` | Teacher dataset and agreement training |
| `config.rs` | Training and evolution settings |

## 7. Certification requirements

A trained forest is a candidate policy. To certify canonical infinite survival:

- Replay selected placements with the authoritative engine.
- Check every possible next piece in each remaining bag.
- Verify that every selected move is non-losing and stays in the certified set.
- Establish that the empty-board/full-bag seed is included or reaches that set under the policy.

Finite survival and an SCC alone are insufficient. No checked-in Lean development or Lean
translation is assumed by this implementation.

Cell arithmetic provides a useful necessary check: a complete bag adds 28 cells and each standard
line removes 10, so bounded infinite survival requires a long-run rate of 2.8 lines per bag
(0.4 lines per piece). Returning to the same cell count at a bag boundary requires a multiple of
five bags; this does not establish that a particular reset sequence is feasible.
