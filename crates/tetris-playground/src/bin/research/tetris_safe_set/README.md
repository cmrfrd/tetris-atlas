# Minimal Safe-Set Certifier

**Level:** Attempting L2 (non-adversarial, bounded recovery)
**Status:** Experimental

## Strategy

Allocation-free best-first backtracker that attempts to prove infinite play via
bounded recursive recovery scripts.

Starting from one board (typically empty), for each forced bag permutation:
1. **Base case**: Find a direct 7-piece placement script back into the safe set
2. **Recovery**: If no direct script exists, choose one fixed script and then
   require every next bag to certify recursively at one smaller remaining depth

The hot search core is intentionally allocation-free (fixed-capacity arena,
heap, and candidate buffer) for maximum throughput.

## Opponent model

**Non-adversarial** — each forced bag is a specific known permutation. All 5040
permutations must be certified, but the player knows which one is coming.

## Constraints

- 10-wide board (canonical)
- Safe height cap: h≤4
- Recovery depth: 2 bags (RECOVERY_BAG_DEPTH=2)
- Planner width: 10 (best-first search width)
- Fixed-capacity data structures (no heap allocation in inner loop)

## Usage

```sh
cargo run --release -p tetris-playground --bin tetris_safe_set
```
