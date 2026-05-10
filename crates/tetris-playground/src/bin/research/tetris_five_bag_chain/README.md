# tetris_five_bag_chain

Chain-solve 5-bag sequences by reusing the MCTS tree across related sequences.

## Idea

Instead of solving each 5-bag sequence from scratch, solve one sequence then
**swap the last bag** and reuse the existing tree. Since the first 4 bags
(28 pieces) are unchanged, all tree knowledge for steps 0-27 carries over.
Only the last 7 steps need re-exploration.

## Algorithm

1. Generate a random 5-bag sequence `[B0, B1, B2, B3, B4]`
2. Run PUCT/MCTS until solved (empty board after 35 pieces) or 10s timeout
3. On solve: swap bag 4 with a new random bag -> `[B0, B1, B2, B3, B5]`
   - Invalidate all nodes at step >= 28
   - Keep tree structure and visit stats for steps 0-27
4. Resume MCTS on the pruned tree
5. Repeat until 6-minute global deadline

## Why this helps

- The tree at steps 0-27 is already well-explored with accurate Q-values
- Only ~20% of the tree (bag 5 territory) needs re-exploration
- Each subsequent solve should be dramatically faster than a cold start
- Enables solving many more sequences in the same wall-clock budget

## Running

```sh
cargo run --release -p tetris-playground --bin tetris_five_bag_chain
```

## Output

Per-sequence: `chain=N solved=true/false time=Xs bags=[...] nodes=N edges=N`

Final: `solved=N/M wall=360.0s solve_rate=X.X%`
