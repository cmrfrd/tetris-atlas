# 7-Bag Adversarial Atlas Solver

**Level:** Attempting L2/L3 (adversarial)
**Status:** All tested configurations → 0% winning

## Strategy

AND-OR retrograde elimination over `(board, bag_remaining)` states. The
adversary picks the worst-case piece from the remaining bag; the player must
survive all choices. States are eliminated backward from death states until a
fixed point is reached.

## Opponent model

**Full adversarial.** At each step the adversary selects which of the remaining
bag pieces to present. The player must have a winning placement for every
possible adversary choice. This is the strongest (hardest) opponent model.

## Constraints

- 10-wide board (canonical)
- Configurable `--max-height`, `--max-holes`, `--max-roughness`, `--max-count`
- State: `(board, bag_remaining)` where bag tracks which of 7 pieces remain
- Retrograde GFP: start with all states as "winning", eliminate those with no
  surviving move against some adversary choice

## Results

Every tested height/hole/roughness configuration produces **0% winning states**.
The adversarial admissibility constraints help the adversary — they can always
force the player into a losing position within the bounded state space.

## Usage

```sh
cargo run --release -p tetris-playground --bin tetris_7bag_atlas -- \
  --max-height 4 --max-holes 1 --max-roughness 8
```
