# Bag-Aware Inductive Invariant Checker (SMT/CEGIS)

**Level:** Attempting L3 (adversarial, canonical rules)
**Status:** In development — no invariants found yet

## Strategy

Uses Z3 SMT solver with CEGIS (counterexample-guided inductive synthesis) to
find per-bag-state inductive invariants `P_B(board)` such that all 448
bag-state transitions remain safe under adversarial piece selection.

### Key insight

Standard 1-step inductiveness checks one invariant P for all pieces. But the
7-bag constrains the adversary: they can only give pieces FROM the current bag.
By tracking bag state, the invariant can be:
- **Looser** when many pieces remain (player has options)
- **Tighter** when few remain (less adversary freedom)
- **Specific** when 1 piece left (no adversary choice at all)

If invariants are found for all 448 bag-state transitions, this constitutes a
**full proof of infinite play** under the canonical 7-bag randomizer.

## Opponent model

**Adversarial** — the adversary selects the worst-case piece from the remaining
bag. All 448 transitions (128 bag states × piece choices) must be proven safe.

## Constraints

- Configurable board dimensions (default 10×20)
- Bitvector-encoded board state for SMT
- Per-bag-state invariant parameters (height, holes, roughness bounds)
- Loosened invariants when bag has more pieces remaining

## Usage

```sh
RUST_LOG=info cargo run --release -p tetris-playground --bin tetris_invariant_smt
```
