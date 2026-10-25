# Tetris Decision Forest — System Design

> A decision **forest** whose trees are boolean-operation logic over a binary
> encoding of the game (board + bag state + current piece) and whose leaves are
> **placements**; each tree votes a move and the forest plays the winner. Trees
> that ever output an invalid placement are culled. An evolutionary loop grows
> the forest up a survival curriculum toward playing **1,000,000 pieces** in a
> single game.

## 1. Why this exists

This project's north star is to **prove Tetris survivable forever** (see the
repo `CLAUDE.md` and the `proofs/` Lean development). A proof by construction
needs a concrete policy: a function `state -> move` that provably never tops
out. The cleanest such object is a **decision forest of boolean operations over
the game's bits**: each tree routes the binary game encoding through
`And`/`Or`/`Not`/`Xor` tests to a leaf that names a placement; the forest plays
the placement with the most tree votes. A decision forest over game bits is a
plain Boolean function of the state — exactly the shape that transcribes into a
Lean `Solver` for the eventual `TetrisSolvable` proof, with no feature extractor
to justify.

So this binary has two jobs:

1. **Empirically** find a forest that survives a very long time (milestone M1),
   by evolution against the real 7-bag engine.
2. Produce a policy in a form that is **verifiable** later — total,
   deterministic, expressed as Boolean logic over the raw game bits.

This document outlines the system. Usage lives in `README.md` next to this file.

## 2. The policy: game bits → tree votes → placement

The policy is a **classifier forest**: it maps the current game state directly
to a move, with no board scoring and no lookahead. Each turn:

1. Encode the game to a fixed **214-bit** binary vector (see below), via
   `encode::encode_game`.
2. **Each tree** routes the bits through its boolean tests to a leaf holding a
   **placement id** — the tree's **vote**.
3. The forest **tallies** all votes and — restricted to the placements that are
   *legal* for the current piece — plays the one with the most. Ties break by
   lowest placement index; if no legal placement received a vote, the
   lowest-index legal placement is played (a legal move is always chosen).

```
       game bits (214: board + bag + piece)
                     │
        ┌────────────┴────────────┐
        │  each tree routes to a   │
        │  leaf → placement id     │   (its vote)
        └────────────┬────────────┘
                     │ tally, restrict to legal
                     ▼
          most-voted legal placement
```

**No built-in loss avoidance.** The vote ignores what each placement *leads to*,
so the forest can vote a topping-out placement even when a safe one exists.
Survival fitness (§4) is what teaches the trees to avoid losing votes.

### The input: board + bag + current piece, all binary (`NUM_INPUTS = 214`)

Every tree reads the same 214 binary inputs, assembled by `encode_game`:

| bits      | meaning                                                                    |
|-----------|---------------------------------------------------------------------------|
| `0..200`  | **board** occupancy — the engine's `to_cell_array(Major::Col)`, `col*20+row` |
| `200..207`| **bag remaining** — `bag.contains(piece)` for each of the 7 pieces          |
| `207..214`| **current piece** — one-hot, `current_piece == piece`                       |

The board part is the engine's own cell array (no hand-rolled reader). Giving
the trees the **current piece** is what lets them output a *valid* placement
(they can branch on which piece they must place), and giving them the **bag
state** lets them anticipate the pieces still coming this bag — genuine
lookahead in the input rather than in a search. A `BoolExpr::Cell(i)` tests
input bit `i`; the layout is frozen (guarded by `FOREST_FORMAT_VERSION`).

## 3. The forest: boolean-op decision trees that vote placements

Each **tree** has two node kinds. Internal `Branch` nodes route on a boolean
condition; `Leaf` nodes name a placement. The condition at each branch is a
`BoolExpr` — **every node of which is a boolean operation** over the input bits.

```rust
enum BoolExpr {                        // a boolean operation over the 214 bits
    Cell(usize),                       //   is input bit `i` set?
    Const(bool),
    Not(Box<BoolExpr>),                //   ¬a
    And(Box<BoolExpr>, Box<BoolExpr>), //   a ∧ b
    Or (Box<BoolExpr>, Box<BoolExpr>), //   a ∨ b
    Xor(Box<BoolExpr>, Box<BoolExpr>), //   a ⊕ b
}

enum DecisionTree {
    Leaf(u8),                          // a placement id (the tree's vote)
    Abstain,                           // "no recommendation" (the tree stays silent)
    Branch { cond: BoolExpr, yes: Box<DecisionTree>, no: Box<DecisionTree> },
}

struct DecisionForest { trees: Vec<DecisionTree> }
```

- **Every branch node is a boolean operation** — its `cond` is a formula of
  `And`/`Or`/`Not`/`Xor` over input-bit literals. No thresholds, no arithmetic.
- **A leaf is a placement — or an abstention.** `tree.decide(bits)` walks the
  tests to a leaf and returns `Some(placement id)` (a vote) or `None`
  (`Abstain` — "no recommendation"). Abstention is the specialization
  mechanism: a tree can vote only on boards its conditions recognize and stay
  silent everywhere else. Silence is never an invalid output, so the autokill
  does not punish it.
- **The forest votes — and always answers.** `forest.vote(bits, legal)` tallies
  the non-abstaining trees and returns the legal placement with the most votes.
  Totality: `vote` yields no placement only for an empty legal set, which
  cannot occur in a live game; if every tree abstains or votes illegally, the
  lowest-index legal placement plays. No reachable "no answer" state exists.
- Everything is `serde`-serializable → forests checkpoint to JSON and reload.

Trees may be **deep** (structural cap `max_tree_depth`, default 24) with
**sizeable** conditions (default cond depth 6), and there may be **many** of
them (default 64, up to 256): the capacity to encode a piece-conditional policy
lives in that depth and breadth. Trees are *initialized small*
(`init_tree_depth`, default 6) and **grow via evolution** up to the cap — random
tree generation is a supercritical branching process (expected size ~1.5^depth),
so depth is earned through grow mutations rather than granted at birth. The
cap is a real invariant: grow mutations stop once a tree reaches
`max_tree_depth`, condition-grow respects `max_cond_depth`, and crossover
exchanges whole trees (never subtrees) so it cannot deepen anything — deep is
allowed, unbounded bloat is not (all three are covered by tests).

**Generation is piece-gated.** Every randomly generated tree (population init,
tree-add mutation, regrowth) has a fixed skeleton: the root chain tests the 7
current-piece one-hot bits, and the random board-conditioned subtree under the
gate for piece `i` votes only placements *of piece `i`*
(`random_piece_gated_tree`). Since the engine's legal set is exactly the
current piece's placement table, such a tree is **100 % valid by
construction** — it survives the invalid-output autokill (below) while staying
board-sensitive. Piece-blind uniform leaves are legal on only ~1/7 of
decisions, so without the gate skeleton, cull-at-1.0 kills essentially every
random tree (observed: forests collapse to 1 tree in a generation and
evolution stalls). Mutations may later break a gate; the cull polices that,
and leaf retargets stay within the leaf's piece 80 % of the time to keep gated
paths valid.

### Validity enforcement and the invalid-tree autokill

A tree's output is **valid** for the current piece iff its placement id is legal
(`DecisionTree::output_is_valid`). Two mechanisms:

- **Hard (in `vote`):** only legal placements can win the tally, so an invalid
  placement is **never played** — regardless of what the trees vote.
- **Training-time autokill (`cull_invalid_trees`):** each generation, every
  tree's valid-output rate is measured over a short sample game, and trees below
  `min_valid_rate` are dropped. The **default is `1.0`** — *any* tree that ever
  outputs an invalid placement is culled — never emptying a forest.
- **Regrowth (`regrow_forest`):** the counterweight to the cull. After the
  autokill, offspring are topped back up to at least `min_trees` (default 32)
  with SILENT-SPECIALIST gated trees — mostly-abstaining, so the injection is
  near-neutral for the plurality instead of randomizing it. Broken trees die,
  valid ones replace them — the forest stays a genuine many-tree accumulation
  instead of ratcheting down to a single survivor (tree count only ever
  decreased before this: culled and removed trees had no reliable replacement
  path).

The autokill is only survivable because generation is piece-gated (§3): a
gated tree is 100 % valid by construction, so the aggressive default has a
population of real passers rather than an empty bar. `min_valid_rate` remains
tunable if a softer regime is ever wanted.

## 4. Training: genetic programming with survival fitness and a curriculum

We evolve a **population of forests**. The genome is a whole forest; the genetic
operators act on decision-tree and boolean-condition structure.

**Loop (one generation):**

1. **Evaluate** every forest in parallel (`rayon`): play games from the
   current seed set up to the current piece cap. **Fitness is pure survival**
   — the mean pieces placed before topping out. Seeds are FROZEN for
   `seed_block` generations (default 20) then rotated: within a block
   selection climbs a stationary objective; across blocks nothing can overfit
   one sequence set.
2. **Elitism**: copy the top `elite` forests unchanged.
3. **Reproduce**: tournament-select parents, apply **crossover** then
   **mutation**, then **autokill** invalid-output trees from each offspring
   and **regrow** it back up to the `min_trees` floor with silent-specialist
   gated trees.
4. **Checkpoint (same-seed champion gate)**: re-score the incumbent champion
   on THIS generation's seeds; replace it only if the generation's best beats
   it on the identical seed set. (A historical high-water bar across rotating
   seed sets is a winner's-curse ratchet: it freezes ~3.4 σ above true skill
   and can install lucky-but-worse genomes.) Write the champion JSON on
   replacement; append a CSV row.

**Genetic operators (`evolve.rs`):** leaf mutation (retarget a placement id —
80 % of the time within the same piece, preserving gated-path validity),
grow (leaf → branch), prune (branch → leaf), condition mutation (LOCAL tweak
by default — retarget a `Cell`, flip a `Const`, unwrap a `Not`, rotate a
binary gate — with wholesale condition regrow as the rare macro jump), tree
add (usually DUPLICATE-AND-DIVERGE: copy an evolved tree and mutate the copy
— near-neutral gene duplication is the fixation path above the `min_trees`
floor; sometimes a fresh gated tree) / remove, and WHOLE-TREE crossover (for
each shared slot a fair coin picks parent a's or b's entire tree — subtree
grafts crossed piece gates 6/7 of the time, manufacturing invalid votes the
autokill then punished by destroying the evolved recipient). The number of
node mutations per child scales with forest size (one per 8 trees): a single
node change in one tree of a 32-tree plurality is almost always behaviorally
silent, and offspring that are exact clones give selection nothing to grade.

**Mutation is validity-aware.** Node targeting redraws away from the
piece-gate spine (a gate hit is near-certain tree death: prune collapses the
chain, a `Cell` reroll breaks the gate 213/214 of the time), and every
mutation receives the target's **governing piece** (`governing_piece` — the
innermost gate whose `yes` side contains the node), so leaf retargets,
Abstain→vote revivals, and grown leaves emit placements that are legal exactly
where the node runs. Chains of mutations therefore preserve a gated tree's
100 %-validity (tested) — before this, ~a quarter of structural mutations
invalidated the tree, so the autokill destroyed ~one evolved tree per child
per generation and the mutation budget mostly fed cull churn and silent
regrowth (an abstain ratchet toward the constant fallback policy).

### The curriculum — the road to 1,000,000 pieces

The eval **piece cap escalates** once the champion survives the current cap on
every seed:

```
cap: 200 → 1_000 → 10_000 → 100_000 → 1_000_000
```

Reaching the top rung — a champion that plays a full 1M-piece game on every
seed — is the binary's success condition. A separate `bench` subcommand
stress-tests a saved champion at the 1M cap across a larger seed set.

### The PUCT teacher (`puct.rs` + `distill.rs`) — training-time search

Survival fitness alone gives a weak gradient once games reach the clearing
wall. The `distill` subcommand adds an AlphaZero-style improvement operator
**without touching the artifact's purity** (search exists only at training
time; the played policy remains a pure boolean function of the 214 bits):

- **Prior**: the forest's vote tally over legal placements, mixed 3:1 with
  uniform (a flat Laplace floor would swamp the typically-small vote mass).
- **Value**: rollouts that follow the forest policy; value = survival
  fraction of the horizon (consistent with the training objective).
- **Selection**: `Q(a) + c_puct · P(a) · √N/(1+N(a))` allocates a per-state
  simulation budget after a round-robin first pass (every action gets one
  real Q before allocation starts); each simulation copies the root
  (`TetrisGame` is `Copy`), reseeds the game RNG **and reshuffles the
  remaining bag order** — the bag stores an already-shuffled queue, so a
  reseed alone would replay the root's true hidden order into every
  simulation. Board, bag membership, and current piece are preserved; the
  label never leaks information the policy's inputs do not expose.
- **Label**: the most-visited action; **distillation fitness = agreement**
  with the labels — deterministic and noise-free, so champion updates are
  strict improvements.

The distilled champion is a regular forest JSON that `train --resume`
accepts: train → distill → train is one policy-improvement round.

## 5. Module map (all under this directory)

| file          | responsibility                                                              |
|---------------|-----------------------------------------------------------------------------|
| `main.rs`     | clap CLI (`train`/`play`/`eval`/`bench`/`show`) + module declarations       |
| `encode.rs`   | `NUM_INPUTS = 214`; `encode_game` — board + bag + current piece, all binary  |
| `tree.rs`     | `BoolExpr` (boolean-op conditions) + `DecisionTree` (branches → placement leaves, `output_is_valid`) |
| `forest.rs`   | `DecisionForest`: the vote (bits → placement), random gen, JSON              |
| `policy.rs`   | `choose_placement` (encode + vote); full-game `play_game`                   |
| `evolve.rs`   | genetic operators: tree/boolean mutation, whole-tree crossover, selection   |
| `train.rs`    | survival `fitness`, `cull_invalid_trees`, the loop, curriculum, checkpoints  |
| `puct.rs`     | root-PUCT teacher: forest-prior + forest-rollout policy improvement          |
| `distill.rs`  | teacher-labeled dataset + agreement-fitness distillation loop                |
| `config.rs`   | `TrainConfig` / `EvolveConfig` (deep/wide/many + cull defaults)             |
| `README.md`   | usage / quick start                                                         |
| `DESIGN.md`   | this document                                                               |

Tree/vote **logic** (`encode`, `tree`, `forest`, `policy`) is separate from
**training/improvement** (`evolve`, `train`, `config`): the policy half is what
a proof would consume; the training half is scaffolding to discover it.

## 6. Design guardrails from the proof work

Hard facts any survivable policy must obey (`proofs/`), used as sanity checks:

- **Clearing rate ≥ 2.8 rows/bag** forever (`BagGrowth.lean`); a champion's
  lines/piece should approach 0.1.
- **Order adaptivity is mandatory** (`BagPlan.lean`): no fixed per-piece plan
  survives. The forest reacts to the board *and* (via the input) the current
  piece and bag, so it can be genuinely order- and lookahead-adaptive.
- **Transient holes are unavoidable** (`SurfaceLanguage.lean`): survival, not
  hole-freeness, is the objective.
- **No perfect 5-bag reset** (`BagGrowth.lean` T-parity): a "reset" rhythm must
  be even in bags (10 bags / 28 rows).

## 7. The proof hook (future work, out of scope here)

A 1M-surviving champion is a **candidate M4 Atlas policy**. Its forest is a
decision forest of Boolean operations over **raw game bits** — a `Bool`-driven
function of the state — so it transcribes directly into a Lean `Solver` with no
feature-extractor to verify (`SeamCert`/`safe_extract` turn a closed policy into
`TetrisSolvable`; `SurfaceLanguage.lean`'s `ScanAut` is a decision procedure over
exactly this kind of bit reader). Verification would then prove the forest's
reachable state set closed under the safe-step operator. This binary produces
the artifact; the proof is a separate effort.

## 8. Non-goals

- Not a beam search or lookahead *search* engine **at play time** (a pure
  bits→vote policy; lookahead lives only in the bag bits of the input).
  Search exists solely as the training-time PUCT teacher (§4), which never
  runs at inference.
- Not an ML/gradient system (evolution over discrete tree structure; no autodiff).
- Not a UI/visualizer (text metrics only).
- Not a proof (produces the policy artifact; proving it is downstream).
