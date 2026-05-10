# Controlled-Invariant Synthesis (MVP)

**Level:** Attempting L2 (non-adversarial approximation)
**Status:** Experimental

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

**Non-adversarial** (falsification via rollouts). The fixed-point is computed
over sampled states, not an exact state space.

## Constraints

- Configurable bucket sizes for height, holes, roughness
- Configurable sampling depth and state count
- One-shot or incremental round-based modes
- Representative quantifier: exists (any rep survives) or for-all

## Usage

```sh
cargo run --release -p tetris-playground --bin tetris_invariant_synthesis -- \
  --mode incremental --max-rounds 20 --max-concrete-states 150000
```
