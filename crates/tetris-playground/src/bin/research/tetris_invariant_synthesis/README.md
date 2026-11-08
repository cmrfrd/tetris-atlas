# Controlled-Invariant Synthesis (MVP)

Experimental approximation of an adversarial safety game.

## Strategy

Builds an abstract safety game from sampled concrete states and solves for a
controlled invariant set via fixed-point elimination. This approximates the
game-theoretic condition: for every piece draw, there exists a placement that
keeps the state safe.

1. **Sampling**: BFS from empty board, collecting concrete `(board, bag)` states
2. **Abstraction**: Bucket states by height/holes/roughness
3. **Fixed-point elimination**: Remove abstract states that can't survive all
   piece draws (for-all quantifier over pieces, exists quantifier over placements)
4. **Refinement**: Tighten buckets on failure, re-sample, repeat
5. **Falsification**: Rollout-based testing of the surviving invariant

## Opponent model

The fixed-point step uses adversarial piece obligations over a sampled abstract model.
Rollout falsification samples continuations and does not exhaust the adversary's choices.
Optimistic representative selection or ignored unknown successors can leave spurious candidates;
concrete closure, transition replay, and empty-seed reachability remain separate obligations.

## Constraints

- Configurable bucket sizes for height, holes, roughness
- Configurable sampling depth and state count
- One-shot or incremental round-based modes
- Representative quantifier: exists (any rep survives) or for-all

## Usage

```sh
cargo run --release -p tetris-playground --bin tetris_invariant_synthesis -- \
  --mode incremental --max-rounds 20 --max-concrete-states 150000 \
  --policy-map-csv artifacts/output/tetris_invariant_policy_map.csv \
  --artifact-dir artifacts/output/invariant-synthesis
```
