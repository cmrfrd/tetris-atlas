# tetris_five_bag_chain

Chain-solve 5-bag sequences by reusing the MCTS tree across related sequences.

## Idea

Instead of solving each 5-bag sequence from scratch, solve one sequence then
**swap the last bag** and reuse the existing tree. Since the first 4 bags
(28 pieces) are unchanged, their board transitions can be reused. Search values may still
depend on the changed final bag, so reuse is a heuristic rather than an independent proof.

## Algorithm

1. Generate a random 5-bag sequence `[B0, B1, B2, B3, B4]`
2. Run PUCT/MCTS until solved (empty board after 35 pieces) or the per-sequence timeout
   (60 seconds by default)
3. On solve: swap bag 4 with a new random bag -> `[B0, B1, B2, B3, B5]`
   - Invalidate all nodes at step >= 28
   - Keep tree structure and visit stats for steps 0-27
4. Resume MCTS on the pruned tree
5. Repeat until 6-minute global deadline

## Intended benefit

Reusing prefix transitions can reduce repeated work. The final seven moves are 20% of the
sequence, not necessarily 20% of the search tree or runtime. Measure reuse against a cold
start on identical sequences before claiming a speedup. These are known-sequence witnesses;
they do not establish an online strategy against adversarial draws.

## Running

```sh
cargo run --release -p tetris-playground --features tui --bin tetris_five_bag_chain -- \
  run --timeout 60 --global-timeout 360
```

## Output

Per-sequence: `chain=N solved=true/false time=Xs bags=[...] nodes=N edges=N`

Final: `solved=N/M wall=360.0s solve_rate=X.X%`
