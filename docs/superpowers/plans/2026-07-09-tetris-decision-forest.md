# Tetris Decision Forest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new Rust research binary `tetris_decision_forest` that plays Tetris with an evolvable **decision forest** — each tree is a decision tree whose branch conditions are boolean operations (`And`/`Or`/`Not`/`Xor`) over a 214-bit binary encoding of the game (board + bag state + current piece) and whose leaves are **placements**; the forest plays the placement with the most tree votes. Trees that ever output an invalid placement are culled during training. A genetic-programming loop grows the forest up a survival curriculum toward surviving 1,000,000 pieces in a single game.

**Architecture:** Every tree sees the same 214 binary inputs — 200 board-occupancy bits (from the engine's `to_cell_array`), 7 bag-remaining bits, and 7 one-hot current-piece bits. A tree routes those bits through boolean-operation tests to a leaf placement id (its vote). The forest tallies votes and plays the legal placement with the most — one forest pass per turn, no lookahead. A tree's output is *valid* iff its placement is legal for the current piece (which the tree can see); the vote only lets legal placements win, and training culls trees whose valid-output rate is below `min_valid_rate` (default 1.0 = cull any invalid-outputting tree). Fitness is pure survival. See the companion `DESIGN.md` and `README.md`.

**Tech Stack:** Rust (edition 2024, nightly), `tetris-game` engine, `clap` (derive), `rayon` (parallel fitness), `rand` 0.9 (`SmallRng`), `serde` + `serde_json` (forest checkpoints), `ctrlc` (graceful stop), `anyhow`.

## Global Constraints

- **Leave everything unstaged — do NOT stage or commit.** Every task edits the working tree only; there are no `git add` / `git commit` / branch steps. The entire system stays as unstaged changes for the user to review and commit themselves.
- **Location:** ALL new code lives under `crates/tetris-playground/src/bin/research/tetris_decision_forest/`. The only file touched outside it is `crates/tetris-playground/Cargo.toml` (one `[[bin]]` entry).
- **Bin naming:** directory name == binary name == `tetris_decision_forest`.
- **Crate preamble:** `main.rs` (the bin crate root) starts with the research-bin standard:
  ```rust
  #![feature(generic_const_exprs)]
  #![allow(incomplete_features)]
  ```
  Module files do not repeat crate attributes.
- **The trees' input is 214 binary bits, assembled via the engine:** `NUM_INPUTS = NUM_CELLS(200) + NUM_PIECES(7) + NUM_PIECES(7) = 214`. Layout: bits `0..200` = board occupancy from `board.to_cell_array(Major::Col)` verbatim (index `col*ROWS+row`); bits `200..207` = bag-remaining (`bag.contains(piece)` for each of the 7 pieces); bits `207..214` = current-piece one-hot (`current_piece == piece`). Assembled by `encode::encode_game`. All inputs are `0`/`1`; a `BoolExpr::Cell(i)` tests input bit `i` for `i` in `0..214`. Do not hand-roll a board reader — the board portion is the engine's `to_cell_array`.
- **Every internal node is a boolean operation:** a tree's branch condition is a `BoolExpr` — `Cell(i)` / `Const(b)` leaves and `Not`/`And`/`Or`/`Xor` gates only. Tree leaves are placement ids. No arithmetic, no thresholds, no scores anywhere.
- **The forest outputs a placement by vote:** each tree emits one placement id per input; the forest plays the *legal* placement with the most votes (ties → lowest placement index; if no legal placement got a vote, the lowest-index legal placement). Invalid outputs (pids not legal for the current piece) cannot win, so an invalid placement is never played. No board scoring, no lookahead — a topping-out placement can win, and survival fitness teaches the trees to avoid it.
- **Validity is enforced and invalid-output trees are autokilled:** a tree's output is *valid* iff its placement id is legal for the current piece (`DecisionTree::output_is_valid`). The vote already prevents an invalid placement from being played. During training, `train::cull_invalid_trees` removes every tree whose valid-output rate over a short sample game is below `min_valid_rate` — **default 1.0, i.e. any tree that ever outputs an invalid placement is culled** — never emptying a forest. Because trees see the current piece (7 one-hot bits), a fully piece-conditional tree can be 100 % valid, so the aggressive default has a real target to evolve toward; it is demanding, so `--min-valid-rate` lets it be relaxed.
- **Deep, wide, many:** decision trees may be deep (default max depth 24) with sizeable boolean conditions (default cond depth 6), and there may be many of them (default 64 initial, up to 256). All limits are config fields exposed as CLI flags.
- **Fitness is survival:** a forest's fitness is the mean number of pieces placed before topping out (capped by the curriculum). Nothing else enters the fitness scalar.
- **Determinism:** a game is reproducible from `(seed, forest)`; all randomness flows through explicitly-seeded `rand::rngs::SmallRng`; vote ties break by lowest placement index.
- **Run commands:** `cargo run --release -p tetris-playground --bin tetris_decision_forest -- <subcommand>`.
- **Test command:** `cargo test -p tetris-playground --bin tetris_decision_forest`.
- **Engine import note:** standard config is `tetris_game::config::StandardTetris`; cells come from `board.to_cell_array(tetris_game::tetris::Major::Col)`; `game.bag.contains(piece)`, `game.current_piece`, `TetrisPiece::all()`, and placement ids `TetrisPiecePlacement::{index, from_index, NUM_PLACEMENTS}`. If any path errors at first build, resolve it against `crates/tetris-game/src/tetris.rs`. Task 1 Step 3 and Task 2 Step 2 surface these first.
- **No engine changes:** everything is computed from the existing `tetris-game` public API.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `crates/tetris-playground/Cargo.toml` (modify) | Add one `[[bin]]` entry. |
| `.../tetris_decision_forest/main.rs` | clap CLI + `mod` declarations; dispatch to subcommands. |
| `.../tetris_decision_forest/encode.rs` | `NUM_CELLS`/`NUM_PIECES`/`NUM_INPUTS`; `encode_game` — the 214-bit input (board+bag+piece). |
| `.../tetris_decision_forest/tree.rs` | `BoolExpr` (boolean-op conditions over the 214 bits) + `DecisionTree` (boolean branches, placement leaves, `output_is_valid`). |
| `.../tetris_decision_forest/forest.rs` | `DecisionForest`: the vote (inputs → placement), random gen, JSON save/load. |
| `.../tetris_decision_forest/policy.rs` | `GameOutcome`, `choose_placement` (encode + vote), full-game `play_game`. |
| `.../tetris_decision_forest/evolve.rs` | genetic operators: tree/boolean mutation, subtree crossover, selection. |
| `.../tetris_decision_forest/config.rs` | `TrainConfig`, `EvolveConfig` (deep/wide/many + cull defaults). |
| `.../tetris_decision_forest/train.rs` | survival `fitness`, `cull_invalid_trees`, the generational loop, curriculum, checkpoints. |
| `.../tetris_decision_forest/README.md` | Usage (already written). |
| `.../tetris_decision_forest/DESIGN.md` | System design (already written). |

`README.md` and `DESIGN.md` already exist in the directory. This plan builds the code around them, then Task 9 syncs the README's CLI section.

Tests live inline as `#[cfg(test)] mod tests { ... }` at the bottom of each module file and run via `cargo test -p tetris-playground --bin tetris_decision_forest`.

---

### Task 1: Scaffold — the bin compiles and plays random moves

Stand up the binary end-to-end with a trivial random policy, so every engine import is validated before any real logic is written.

**Files:**
- Modify: `crates/tetris-playground/Cargo.toml`
- Create: `crates/tetris-playground/src/bin/research/tetris_decision_forest/main.rs`

**Interfaces:**
- Consumes: `tetris_game::tetris::{TetrisGame, IsLost}`, `tetris_game::config::StandardTetris`.
- Produces: a runnable `tetris_decision_forest` binary whose `main` currently plays one random game (replaced by the real CLI in Task 8).

- [ ] **Step 1: Add the `[[bin]]` entry to Cargo.toml**

In `crates/tetris-playground/Cargo.toml`, near the other research bins (search `name = "tetris_carrier_probe"`), add:
```toml
[[bin]]
name = "tetris_decision_forest"
path = "src/bin/research/tetris_decision_forest/main.rs"
```
No new dependencies are needed — `clap`, `rayon`, `rand` (with `small_rng`), `serde`, `serde_json`, `ctrlc`, and `anyhow` are already declared in this crate.

- [ ] **Step 2: Write a minimal `main.rs` that plays a random game**

Create `crates/tetris-playground/src/bin/research/tetris_decision_forest/main.rs`:
```rust
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tetris_game::config::StandardTetris;
use tetris_game::tetris::{IsLost, TetrisGame};

fn main() {
    let seed = 42u64;
    let mut game = TetrisGame::<StandardTetris>::new_with_seed(seed);
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut pieces = 0u64;

    loop {
        if game.board.is_lost() {
            break;
        }
        let placements = game.current_placements();
        if placements.is_empty() {
            break;
        }
        let idx = rng.random_range(0..placements.len());
        let placement = placements[idx];
        let result = game.apply_placement(placement);
        pieces += 1;
        if result.is_lost == IsLost::LOST {
            break;
        }
    }

    println!(
        "random policy: survived {pieces} pieces, max height {}",
        game.board.height()
    );
}
```

- [ ] **Step 3: Build and run to verify it compiles and plays**

Run:
```sh
cargo run --release -p tetris-playground --bin tetris_decision_forest
```
Expected: compiles, prints e.g. `random policy: survived 23 pieces, max height 21`. If `tetris_game::config::StandardTetris` errors, switch to `tetris_game::StandardTetris` everywhere.

- [ ] **Step 4: Leave changes unstaged**

Do not `git add` or `git commit`. Confirm:
```sh
git status --short
```
Expected: `main.rs`, `README.md`, `DESIGN.md`, and modified `Cargo.toml` appear, all unstaged.

---

### Task 2: The 214-bit game encoding (`encode.rs`)

Assemble the trees' input: board occupancy + bag state + current piece, all binary.

**Files:**
- Create: `crates/tetris-playground/src/bin/research/tetris_decision_forest/encode.rs`
- Modify: `main.rs` (add `mod encode;`)

**Interfaces:**
- Consumes: `tetris_game::tetris::{TetrisGame, Major, TetrisPiece}`, `tetris_game::config::StandardTetris`.
- Produces:
  - `pub const NUM_CELLS: usize = 200;`
  - `pub const NUM_PIECES: usize = 7;`
  - `pub const NUM_INPUTS: usize = NUM_CELLS + NUM_PIECES + NUM_PIECES;` (214)
  - `pub fn encode_game(game: &TetrisGame<StandardTetris>) -> [u8; NUM_INPUTS];`

- [ ] **Step 1: Write the failing test**

Create `encode.rs`:
```rust
use tetris_game::config::StandardTetris;
use tetris_game::tetris::{Major, TetrisGame, TetrisPiece};

// (implementation added in Step 3)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_game_encoding_has_expected_shape() {
        let game = TetrisGame::<StandardTetris>::new_with_seed(1);
        let bits = encode_game(&game);
        assert_eq!(bits.len(), NUM_INPUTS);
        // Fresh board is empty.
        assert!(bits[..NUM_CELLS].iter().all(|&b| b == 0), "empty board");
        // Exactly one current-piece bit is set (one-hot).
        let piece_bits: u32 = bits[NUM_CELLS + NUM_PIECES..].iter().map(|&b| u32::from(b)).sum();
        assert_eq!(piece_bits, 1, "exactly one current piece");
        // The bag has at least one remaining piece at game start.
        let bag_bits: u32 = bits[NUM_CELLS..NUM_CELLS + NUM_PIECES].iter().map(|&b| u32::from(b)).sum();
        assert!(bag_bits >= 1, "bag not empty at start");
        // All inputs are 0/1.
        assert!(bits.iter().all(|&b| b <= 1));
    }

    #[test]
    fn placing_a_piece_marks_board_bits() {
        let mut game = TetrisGame::<StandardTetris>::new_with_seed(1);
        let placement = game.current_placements()[0];
        game.apply_placement(placement);
        let bits = encode_game(&game);
        let board_on: u32 = bits[..NUM_CELLS].iter().map(|&b| u32::from(b)).sum();
        assert_eq!(board_on, 4, "one tetromino fills exactly 4 board bits");
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```sh
cargo test -p tetris-playground --bin tetris_decision_forest encode::tests
```
Expected: FAIL to compile — `encode_game`, `NUM_INPUTS` not found. (If `board.to_cell_array(Major::Col)`, `game.bag.contains`, `game.current_piece`, or `TetrisPiece::all()` error, check `crates/tetris-game/src/tetris.rs` and adjust.)

- [ ] **Step 3: Write the implementation**

Add above the test module in `encode.rs`:
```rust
/// Board occupancy bits (10 × 20).
pub const NUM_CELLS: usize = 200;
/// Number of distinct tetrominoes.
pub const NUM_PIECES: usize = 7;
/// Total binary inputs each tree sees: board + bag-remaining + current-piece.
pub const NUM_INPUTS: usize = NUM_CELLS + NUM_PIECES + NUM_PIECES;

/// Encode the full game as a fixed binary vector the trees read:
/// - `0..200`   board occupancy (`to_cell_array(Major::Col)`, index `col*20+row`),
/// - `200..207` bag-remaining (`bag.contains(piece)` for each of the 7 pieces),
/// - `207..214` current-piece one-hot (`current_piece == piece`).
pub fn encode_game(game: &TetrisGame<StandardTetris>) -> [u8; NUM_INPUTS] {
    let mut bits = [0u8; NUM_INPUTS];

    // Board occupancy, straight from the engine (BOARD_SIZE == 200 for standard).
    let cells = game.board.to_cell_array(Major::Col);
    bits[..NUM_CELLS].copy_from_slice(&cells[..NUM_CELLS]);

    // Bag-remaining and current-piece one-hot, one bit per piece.
    for (i, piece) in TetrisPiece::all().iter().enumerate() {
        if game.bag.contains(*piece) {
            bits[NUM_CELLS + i] = 1;
        }
        if game.current_piece == *piece {
            bits[NUM_CELLS + NUM_PIECES + i] = 1;
        }
    }

    bits
}
```

- [ ] **Step 4: Declare the module in main.rs**

In `main.rs`, add below the crate attributes and above `use`:
```rust
mod encode;
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```sh
cargo test -p tetris-playground --bin tetris_decision_forest encode::tests
```
Expected: both tests PASS.

- [ ] **Step 6: Leave changes unstaged** (no `git add`/`git commit`).

---

### Task 3: Boolean conditions & the decision tree (`tree.rs`)

The boolean-operation condition (`BoolExpr`) over the 214 input bits, and the decision tree (`DecisionTree`) that routes inputs to a leaf placement.

**Files:**
- Create: `crates/tetris-playground/src/bin/research/tetris_decision_forest/tree.rs`
- Modify: `main.rs` (add `mod tree;`)

**Interfaces:**
- Consumes: `encode::NUM_INPUTS`, `rand::rngs::SmallRng`, `tetris_game::tetris::TetrisPiecePlacement`.
- Produces:
  - `pub enum BoolExpr { Cell(usize), Const(bool), Not(Box<BoolExpr>), And(Box<BoolExpr>, Box<BoolExpr>), Or(Box<BoolExpr>, Box<BoolExpr>), Xor(Box<BoolExpr>, Box<BoolExpr>) }` (derives `Clone, Debug, Serialize, Deserialize`); `pub fn random_boolexpr_leaf(&mut SmallRng) -> BoolExpr`.
  - `impl BoolExpr`: `eval(&self, &[u8]) -> bool`, `random(&mut SmallRng, max_depth: u32) -> Self`, `count_nodes(&self) -> usize`, `depth(&self) -> u32`, `node_at_mut(&mut self, usize) -> Option<&mut BoolExpr>`.
  - `pub enum DecisionTree { Leaf(u8), Branch { cond: BoolExpr, yes: Box<DecisionTree>, no: Box<DecisionTree> } }` (derives `Clone, Debug, Serialize, Deserialize`); `pub fn random_pid(&mut SmallRng) -> u8`.
  - `impl DecisionTree`: `decide(&self, &[u8]) -> u8`, `random(&mut SmallRng, max_depth: u32, cond_depth: u32) -> Self`, `count_nodes(&self) -> usize`, `depth(&self) -> u32`, `node_at_mut(&mut self, usize) -> Option<&mut DecisionTree>`, `output_is_valid(&self, &[u8], &[TetrisPiecePlacement]) -> bool`.

- [ ] **Step 1: Write the failing test**

Create `tree.rs`:
```rust
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use tetris_game::tetris::TetrisPiecePlacement;

use crate::encode::NUM_INPUTS;

// (implementation added in Step 3)

#[cfg(test)]
mod tests {
    use super::*;

    fn zeros() -> [u8; NUM_INPUTS] {
        [0u8; NUM_INPUTS]
    }

    #[test]
    fn boolexpr_gates_evaluate() {
        let mut c = zeros();
        c[0] = 1; // bit 0 true, bit 1 false
        let and = BoolExpr::And(Box::new(BoolExpr::Cell(0)), Box::new(BoolExpr::Cell(1)));
        let or = BoolExpr::Or(Box::new(BoolExpr::Cell(0)), Box::new(BoolExpr::Cell(1)));
        let xor = BoolExpr::Xor(Box::new(BoolExpr::Cell(0)), Box::new(BoolExpr::Cell(1)));
        let not = BoolExpr::Not(Box::new(BoolExpr::Cell(1)));
        assert!(!and.eval(&c));
        assert!(or.eval(&c));
        assert!(xor.eval(&c));
        assert!(not.eval(&c));
        assert_eq!(and.count_nodes(), 3);
        assert_eq!(and.depth(), 1);
    }

    #[test]
    fn decision_tree_routes_to_leaf_placement() {
        // if bit 0 set -> Leaf(5) else Leaf(9)
        let tree = DecisionTree::Branch {
            cond: BoolExpr::Cell(0),
            yes: Box::new(DecisionTree::Leaf(5)),
            no: Box::new(DecisionTree::Leaf(9)),
        };
        assert_eq!(tree.decide(&zeros()), 9);
        let mut on = zeros();
        on[0] = 1;
        assert_eq!(tree.decide(&on), 5);
        assert_eq!(tree.count_nodes(), 3);
        assert_eq!(tree.depth(), 1);
    }

    #[test]
    fn random_tree_is_bounded_and_decides_a_valid_pid() {
        let n = TetrisPiecePlacement::NUM_PLACEMENTS as u8;
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..50 {
            let t = DecisionTree::random(&mut rng, 6, 4);
            assert!(t.depth() <= 6, "depth {}", t.depth());
            assert!(t.decide(&zeros()) < n, "pid out of range");
        }
    }

    #[test]
    fn output_validity_matches_legal_pids() {
        let cells = zeros();
        let tree = DecisionTree::Leaf(0);
        let legal_has0 = [
            TetrisPiecePlacement::from_index(0),
            TetrisPiecePlacement::from_index(1),
        ];
        assert!(tree.output_is_valid(&cells, &legal_has0)); // pid 0 in legal
        let legal_no0 = [
            TetrisPiecePlacement::from_index(3),
            TetrisPiecePlacement::from_index(5),
        ];
        assert!(!tree.output_is_valid(&cells, &legal_no0)); // pid 0 not in legal
    }

    #[test]
    fn node_at_mut_addresses_every_node() {
        let mut tree = DecisionTree::Branch {
            cond: BoolExpr::Cell(0),
            yes: Box::new(DecisionTree::Leaf(1)),
            no: Box::new(DecisionTree::Leaf(2)),
        };
        let n = tree.count_nodes();
        assert_eq!(n, 3);
        for i in 0..n {
            assert!(tree.node_at_mut(i).is_some());
        }
        assert!(tree.node_at_mut(n).is_none());

        let mut e = BoolExpr::And(Box::new(BoolExpr::Cell(0)), Box::new(BoolExpr::Const(true)));
        let m = e.count_nodes();
        for i in 0..m {
            assert!(e.node_at_mut(i).is_some());
        }
        assert!(e.node_at_mut(m).is_none());
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```sh
cargo test -p tetris-playground --bin tetris_decision_forest tree::tests
```
Expected: FAIL to compile — `BoolExpr`, `DecisionTree` not found. (If `TetrisPiecePlacement::NUM_PLACEMENTS` errors, use `TetrisPiecePlacement::ALL_PLACEMENTS.len()` in `random_pid` and the test.)

- [ ] **Step 3: Write the implementation**

Add above the test module in `tree.rs`:
```rust
/// A boolean operation over the 214 input bits — the condition at a decision
/// node. Leaves are an input bit (`Cell`) or a constant (`Const`); internal
/// nodes are boolean gates. Evaluates to a single `bool`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BoolExpr {
    Cell(usize),
    Const(bool),
    Not(Box<BoolExpr>),
    And(Box<BoolExpr>, Box<BoolExpr>),
    Or(Box<BoolExpr>, Box<BoolExpr>),
    Xor(Box<BoolExpr>, Box<BoolExpr>),
}

/// A random `BoolExpr` leaf: usually an input bit, occasionally a constant.
pub fn random_boolexpr_leaf(rng: &mut SmallRng) -> BoolExpr {
    if rng.random_bool(0.9) {
        BoolExpr::Cell(rng.random_range(0..NUM_INPUTS))
    } else {
        BoolExpr::Const(rng.random_bool(0.5))
    }
}

impl BoolExpr {
    /// Evaluate on the input bits (out-of-range index reads as `false`).
    pub fn eval(&self, bits: &[u8]) -> bool {
        match self {
            BoolExpr::Cell(i) => bits.get(*i).is_some_and(|&v| v != 0),
            BoolExpr::Const(b) => *b,
            BoolExpr::Not(a) => !a.eval(bits),
            BoolExpr::And(a, b) => a.eval(bits) && b.eval(bits),
            BoolExpr::Or(a, b) => a.eval(bits) || b.eval(bits),
            BoolExpr::Xor(a, b) => a.eval(bits) ^ b.eval(bits),
        }
    }

    pub fn random(rng: &mut SmallRng, max_depth: u32) -> Self {
        if max_depth == 0 || rng.random_bool(0.3) {
            return random_boolexpr_leaf(rng);
        }
        match rng.random_range(0..4) {
            0 => BoolExpr::Not(Box::new(BoolExpr::random(rng, max_depth - 1))),
            1 => BoolExpr::And(
                Box::new(BoolExpr::random(rng, max_depth - 1)),
                Box::new(BoolExpr::random(rng, max_depth - 1)),
            ),
            2 => BoolExpr::Or(
                Box::new(BoolExpr::random(rng, max_depth - 1)),
                Box::new(BoolExpr::random(rng, max_depth - 1)),
            ),
            _ => BoolExpr::Xor(
                Box::new(BoolExpr::random(rng, max_depth - 1)),
                Box::new(BoolExpr::random(rng, max_depth - 1)),
            ),
        }
    }

    pub fn count_nodes(&self) -> usize {
        match self {
            BoolExpr::Cell(_) | BoolExpr::Const(_) => 1,
            BoolExpr::Not(a) => 1 + a.count_nodes(),
            BoolExpr::And(a, b) | BoolExpr::Or(a, b) | BoolExpr::Xor(a, b) => {
                1 + a.count_nodes() + b.count_nodes()
            }
        }
    }

    pub fn depth(&self) -> u32 {
        match self {
            BoolExpr::Cell(_) | BoolExpr::Const(_) => 0,
            BoolExpr::Not(a) => 1 + a.depth(),
            BoolExpr::And(a, b) | BoolExpr::Or(a, b) | BoolExpr::Xor(a, b) => {
                1 + a.depth().max(b.depth())
            }
        }
    }

    pub fn node_at_mut(&mut self, target: usize) -> Option<&mut BoolExpr> {
        let mut cursor = 0usize;
        self.node_at_mut_inner(target, &mut cursor)
    }

    fn node_at_mut_inner(&mut self, target: usize, cursor: &mut usize) -> Option<&mut BoolExpr> {
        if *cursor == target {
            return Some(self);
        }
        *cursor += 1;
        match self {
            BoolExpr::Cell(_) | BoolExpr::Const(_) => None,
            BoolExpr::Not(a) => a.node_at_mut_inner(target, cursor),
            BoolExpr::And(a, b) | BoolExpr::Or(a, b) | BoolExpr::Xor(a, b) => {
                if let Some(found) = a.node_at_mut_inner(target, cursor) {
                    return Some(found);
                }
                b.node_at_mut_inner(target, cursor)
            }
        }
    }
}

/// A random valid placement id (index into `TetrisPiecePlacement::from_index`).
pub fn random_pid(rng: &mut SmallRng) -> u8 {
    rng.random_range(0..TetrisPiecePlacement::NUM_PLACEMENTS) as u8
}

/// A decision tree: internal `Branch` nodes route on a boolean condition; leaves
/// hold a placement id. `decide` routes the input bits to a leaf placement —
/// the tree's vote.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DecisionTree {
    Leaf(u8),
    Branch {
        cond: BoolExpr,
        yes: Box<DecisionTree>,
        no: Box<DecisionTree>,
    },
}

impl DecisionTree {
    /// Route `bits` to a leaf and return its placement id. Iterative so deep
    /// trees do not risk stack overflow.
    pub fn decide(&self, bits: &[u8]) -> u8 {
        let mut node = self;
        loop {
            match node {
                DecisionTree::Leaf(p) => return *p,
                DecisionTree::Branch { cond, yes, no } => {
                    node = if cond.eval(bits) { yes } else { no };
                }
            }
        }
    }

    pub fn random(rng: &mut SmallRng, max_depth: u32, cond_depth: u32) -> Self {
        if max_depth == 0 || rng.random_bool(0.25) {
            return DecisionTree::Leaf(random_pid(rng));
        }
        DecisionTree::Branch {
            cond: BoolExpr::random(rng, cond_depth),
            yes: Box::new(DecisionTree::random(rng, max_depth - 1, cond_depth)),
            no: Box::new(DecisionTree::random(rng, max_depth - 1, cond_depth)),
        }
    }

    pub fn count_nodes(&self) -> usize {
        match self {
            DecisionTree::Leaf(_) => 1,
            DecisionTree::Branch { yes, no, .. } => 1 + yes.count_nodes() + no.count_nodes(),
        }
    }

    pub fn depth(&self) -> u32 {
        match self {
            DecisionTree::Leaf(_) => 0,
            DecisionTree::Branch { yes, no, .. } => 1 + yes.depth().max(no.depth()),
        }
    }

    pub fn node_at_mut(&mut self, target: usize) -> Option<&mut DecisionTree> {
        let mut cursor = 0usize;
        self.node_at_mut_inner(target, &mut cursor)
    }

    fn node_at_mut_inner(
        &mut self,
        target: usize,
        cursor: &mut usize,
    ) -> Option<&mut DecisionTree> {
        if *cursor == target {
            return Some(self);
        }
        *cursor += 1;
        match self {
            DecisionTree::Leaf(_) => None,
            DecisionTree::Branch { yes, no, .. } => {
                if let Some(found) = yes.node_at_mut_inner(target, cursor) {
                    return Some(found);
                }
                no.node_at_mut_inner(target, cursor)
            }
        }
    }

    /// Whether this tree's output is a VALID placement for the current piece:
    /// its leaf placement id is among the legal placements. Used by the vote
    /// (implicitly) and by the training-time autokill.
    pub fn output_is_valid(&self, bits: &[u8], legal: &[TetrisPiecePlacement]) -> bool {
        let pid = self.decide(bits);
        legal.iter().any(|p| p.index() == pid)
    }
}
```

- [ ] **Step 4: Declare the module in main.rs**

Add `mod tree;` to `main.rs`.

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```sh
cargo test -p tetris-playground --bin tetris_decision_forest tree::tests
```
Expected: all five tests PASS.

- [ ] **Step 6: Leave changes unstaged** (no `git add`/`git commit`).

---

### Task 4: The voting forest (`forest.rs`)

The forest tallies each tree's placement vote and returns the legal placement with the most votes.

**Files:**
- Create: `crates/tetris-playground/src/bin/research/tetris_decision_forest/forest.rs`
- Modify: `main.rs` (add `mod forest;`)

**Interfaces:**
- Consumes: `tree::DecisionTree`, `encode::NUM_INPUTS`, `tetris_game::tetris::TetrisPiecePlacement`, `rand::rngs::SmallRng`, `anyhow::Result`.
- Produces:
  - `pub const FOREST_FORMAT_VERSION: u32 = 1;`
  - `pub struct DecisionForest { pub version: u32, pub trees: Vec<DecisionTree> }` (derives `Clone, Debug, Serialize, Deserialize`).
  - `impl DecisionForest`: `vote(&self, bits: &[u8], legal: &[TetrisPiecePlacement]) -> Option<TetrisPiecePlacement>`, `random(&mut SmallRng, num_trees: usize, max_depth: u32, cond_depth: u32) -> Self`, `total_nodes(&self) -> usize`, `save_json(&self, &std::path::Path) -> anyhow::Result<()>`, `load_json(&std::path::Path) -> anyhow::Result<Self>`.

- [ ] **Step 1: Write the failing test**

Create `forest.rs`:
```rust
use std::path::Path;

use anyhow::Result;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use tetris_game::tetris::TetrisPiecePlacement;

use crate::encode::NUM_INPUTS;
use crate::tree::DecisionTree;

// (implementation added in Step 3)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vote_picks_the_legal_placement_with_the_most_votes() {
        let forest = DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![
                DecisionTree::Leaf(0),
                DecisionTree::Leaf(0),
                DecisionTree::Leaf(1),
            ],
        };
        let bits = [0u8; NUM_INPUTS];
        let legal = [
            TetrisPiecePlacement::from_index(0),
            TetrisPiecePlacement::from_index(1),
        ];
        assert_eq!(
            forest.vote(&bits, &legal).unwrap(),
            TetrisPiecePlacement::from_index(0)
        );
    }

    #[test]
    fn invalid_votes_cannot_win_and_fallback_is_lowest_index() {
        // The only tree votes pid 0, which is NOT in the legal set.
        let forest = DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![DecisionTree::Leaf(0)],
        };
        let bits = [0u8; NUM_INPUTS];
        let legal = [
            TetrisPiecePlacement::from_index(5),
            TetrisPiecePlacement::from_index(3),
        ];
        // Invalid vote discarded -> no legal placement got a vote -> lowest index (3).
        assert_eq!(
            forest.vote(&bits, &legal).unwrap(),
            TetrisPiecePlacement::from_index(3)
        );
    }

    #[test]
    fn vote_returns_none_when_no_legal_moves() {
        let forest = DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![DecisionTree::Leaf(0)],
        };
        assert!(forest.vote(&[0u8; NUM_INPUTS], &[]).is_none());
    }

    #[test]
    fn vote_always_returns_a_legal_placement() {
        let mut rng = SmallRng::seed_from_u64(2);
        let forest = DecisionForest::random(&mut rng, 8, 6, 3);
        let bits = [0u8; NUM_INPUTS];
        let legal: Vec<_> = (0u8..9).map(TetrisPiecePlacement::from_index).collect();
        let chosen = forest.vote(&bits, &legal).unwrap();
        assert!(legal.contains(&chosen));
    }

    #[test]
    fn random_forest_and_json_round_trip() {
        let mut rng = SmallRng::seed_from_u64(9);
        let forest = DecisionForest::random(&mut rng, 8, 5, 3);
        assert_eq!(forest.trees.len(), 8);
        assert!(forest.total_nodes() >= 8);
        let path = std::env::temp_dir().join("df_forest_roundtrip.json");
        forest.save_json(&path).unwrap();
        let loaded = DecisionForest::load_json(&path).unwrap();
        let bits = [0u8; NUM_INPUTS];
        let legal = [TetrisPiecePlacement::from_index(0), TetrisPiecePlacement::from_index(1)];
        assert_eq!(forest.vote(&bits, &legal), loaded.vote(&bits, &legal));
        let _ = std::fs::remove_file(&path);
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```sh
cargo test -p tetris-playground --bin tetris_decision_forest forest::tests
```
Expected: FAIL to compile — `DecisionForest` not found.

- [ ] **Step 3: Write the implementation**

Add above the test module in `forest.rs`:
```rust
/// On-disk format version. Bump when the input encoding, node encoding, or
/// placement-id scheme changes so old checkpoints are rejected.
pub const FOREST_FORMAT_VERSION: u32 = 1;

/// A decision forest: each tree votes one placement id per input; the forest
/// plays the legal placement with the most votes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionForest {
    pub version: u32,
    pub trees: Vec<DecisionTree>,
}

impl DecisionForest {
    /// Tally every tree's placement vote and return the legal placement with the
    /// most votes. A vote for a pid that is not legal for the current piece
    /// contributes to that pid's tally but can never win (only legal placements
    /// are considered), so an invalid placement is never played. Ties (including
    /// the all-zero-legal-vote case) break by lowest placement index. `None`
    /// only when `legal` is empty.
    pub fn vote(
        &self,
        bits: &[u8],
        legal: &[TetrisPiecePlacement],
    ) -> Option<TetrisPiecePlacement> {
        if legal.is_empty() {
            return None;
        }
        // Placement ids are `u8` (< NUM_PLACEMENTS ≤ 162); a 256-wide tally
        // covers the whole range without a const dependency.
        let mut votes = [0u32; 256];
        for tree in &self.trees {
            votes[tree.decide(bits) as usize] += 1;
        }
        legal
            .iter()
            .copied()
            .max_by_key(|p| (votes[p.index() as usize], std::cmp::Reverse(p.index())))
    }

    /// A random forest of `num_trees` decision trees.
    pub fn random(
        rng: &mut SmallRng,
        num_trees: usize,
        max_depth: u32,
        cond_depth: u32,
    ) -> Self {
        let trees = (0..num_trees)
            .map(|_| DecisionTree::random(rng, max_depth, cond_depth))
            .collect();
        DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees,
        }
    }

    /// Total decision-node count across all trees (a complexity measure).
    pub fn total_nodes(&self) -> usize {
        self.trees.iter().map(|t| t.count_nodes()).sum()
    }

    pub fn save_json(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    pub fn load_json(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let forest: DecisionForest = serde_json::from_reader(file)?;
        anyhow::ensure!(
            forest.version == FOREST_FORMAT_VERSION,
            "forest format version {} != expected {}",
            forest.version,
            FOREST_FORMAT_VERSION
        );
        Ok(forest)
    }
}
```

- [ ] **Step 4: Declare the module in main.rs**

Add `mod forest;` to `main.rs`.

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```sh
cargo test -p tetris-playground --bin tetris_decision_forest forest::tests
```
Expected: all five tests PASS.

- [ ] **Step 6: Leave changes unstaged** (no `git add`/`git commit`).

---

### Task 5: Policy & playout (`policy.rs`)

Encode the game, delegate to the forest vote, and play full games.

**Files:**
- Create: `crates/tetris-playground/src/bin/research/tetris_decision_forest/policy.rs`
- Modify: `main.rs` (add `mod policy;`)

**Interfaces:**
- Consumes: `forest::DecisionForest`, `encode::encode_game`, `tetris_game::tetris::{TetrisGame, TetrisPiecePlacement, IsLost}`, `tetris_game::config::StandardTetris`.
- Produces:
  - `pub struct GameOutcome { pub pieces_placed: u64, pub lines_cleared: u64, pub max_height: u32, pub alive: bool }` (derives `Clone, Copy, Debug`).
  - `pub fn choose_placement(&DecisionForest, &TetrisGame<StandardTetris>) -> Option<TetrisPiecePlacement>`.
  - `pub fn play_game(&DecisionForest, seed: u64, max_pieces: u64) -> GameOutcome`.

- [ ] **Step 1: Write the failing test**

Create `policy.rs`:
```rust
use tetris_game::config::StandardTetris;
use tetris_game::tetris::{IsLost, TetrisGame, TetrisPiecePlacement};

use crate::encode::encode_game;
use crate::forest::DecisionForest;

// (implementation added in Step 3)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forest::FOREST_FORMAT_VERSION;
    use crate::tree::DecisionTree;

    fn always_pid0() -> DecisionForest {
        DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![DecisionTree::Leaf(0)],
        }
    }

    #[test]
    fn choose_placement_returns_a_legal_move() {
        let game = TetrisGame::<StandardTetris>::new_with_seed(5);
        let chosen = choose_placement(&always_pid0(), &game).unwrap();
        assert!(game.current_placements().contains(&chosen));
    }

    #[test]
    fn play_game_is_deterministic_and_makes_progress() {
        let forest = always_pid0();
        let a = play_game(&forest, 123, 500);
        let b = play_game(&forest, 123, 500);
        assert_eq!(a.pieces_placed, b.pieces_placed);
        assert_eq!(a.lines_cleared, b.lines_cleared);
        assert!(a.pieces_placed > 0);
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```sh
cargo test -p tetris-playground --bin tetris_decision_forest policy::tests
```
Expected: FAIL to compile — `choose_placement`, `play_game`, `GameOutcome` not found.

- [ ] **Step 3: Write the implementation**

Add above the test module in `policy.rs`:
```rust
/// The result of a full game playout.
#[derive(Clone, Copy, Debug)]
pub struct GameOutcome {
    pub pieces_placed: u64,
    pub lines_cleared: u64,
    pub max_height: u32,
    pub alive: bool,
}

/// The forest's move for the current game: encode the game to 214 bits and let
/// the forest vote among the legal placements. No lookahead — survival fitness
/// teaches the trees to avoid losing votes.
pub fn choose_placement(
    forest: &DecisionForest,
    game: &TetrisGame<StandardTetris>,
) -> Option<TetrisPiecePlacement> {
    let bits = encode_game(game);
    forest.vote(&bits, game.current_placements())
}

/// Play a full game with `forest` from `seed`, stopping at loss or after
/// `max_pieces` placements. Deterministic given `(seed, forest)`.
pub fn play_game(forest: &DecisionForest, seed: u64, max_pieces: u64) -> GameOutcome {
    let mut game = TetrisGame::<StandardTetris>::new_with_seed(seed);
    let mut pieces_placed = 0u64;
    let mut lines_cleared = 0u64;
    let mut max_height = 0u32;

    while pieces_placed < max_pieces {
        if game.board.is_lost() {
            break;
        }
        let Some(placement) = choose_placement(forest, &game) else {
            break;
        };
        let result = game.apply_placement(placement);
        pieces_placed += 1;
        lines_cleared += u64::from(result.lines_cleared);
        max_height = max_height.max(game.board.height());
        if result.is_lost == IsLost::LOST {
            return GameOutcome {
                pieces_placed,
                lines_cleared,
                max_height,
                alive: false,
            };
        }
    }

    GameOutcome {
        pieces_placed,
        lines_cleared,
        max_height,
        alive: !game.board.is_lost(),
    }
}
```

- [ ] **Step 4: Declare the module in main.rs**

Add `mod policy;` to `main.rs`.

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```sh
cargo test -p tetris-playground --bin tetris_decision_forest policy::tests
```
Expected: both tests PASS.

- [ ] **Step 6: Leave changes unstaged** (no `git add`/`git commit`).

---

### Task 6: Config + genetic operators (`config.rs`, `evolve.rs`)

Hyperparameters (deep/wide/many + aggressive cull), and the mutation / crossover / selection operators.

**Files:**
- Create: `crates/tetris-playground/src/bin/research/tetris_decision_forest/config.rs`
- Create: `crates/tetris-playground/src/bin/research/tetris_decision_forest/evolve.rs`
- Modify: `main.rs` (add `mod config;` and `mod evolve;`)

**Interfaces:**
- `config.rs` produces:
  - `pub struct EvolveConfig { pub max_tree_depth: u32, pub max_cond_depth: u32, pub leaf_mutation_prob: f64, pub cond_mutation_prob: f64, pub grow_prob: f64, pub prune_prob: f64, pub tree_add_prob: f64, pub tree_remove_prob: f64, pub max_trees: usize, pub min_valid_rate: f64, pub cull_sample_pieces: u64 }` (derives `Clone, Copy, Debug`) with `impl Default`.
  - `pub struct TrainConfig { pub population: usize, pub generations: usize, pub elite: usize, pub tournament_k: usize, pub seeds: usize, pub start_cap: u64, pub init_trees: usize, pub evolve: EvolveConfig }` (derives `Clone, Debug`) with `impl Default`.
- `evolve.rs` produces:
  - `pub fn mutate_forest(&DecisionForest, &mut SmallRng, &EvolveConfig) -> DecisionForest`.
  - `pub fn crossover(&DecisionForest, &DecisionForest, &mut SmallRng) -> DecisionForest`.
  - `pub fn tournament_select<'a>(scored: &'a [(f32, DecisionForest)], rng: &mut SmallRng, k: usize) -> &'a DecisionForest`.

- [ ] **Step 1: Write `config.rs`**

Create `config.rs`:
```rust
/// Genetic-operator tuning. Defaults favor deep, sizeable, and numerous trees,
/// and cull any tree that ever outputs an invalid placement (`min_valid_rate` 1.0).
#[derive(Clone, Copy, Debug)]
pub struct EvolveConfig {
    pub max_tree_depth: u32,
    pub max_cond_depth: u32,
    pub leaf_mutation_prob: f64,
    pub cond_mutation_prob: f64,
    pub grow_prob: f64,
    pub prune_prob: f64,
    pub tree_add_prob: f64,
    pub tree_remove_prob: f64,
    pub max_trees: usize,
    /// Autokill threshold: each generation, trees whose valid-output rate over a
    /// sample game is below this are culled. Default `1.0` = cull any tree that
    /// ever outputs an invalid placement. `0.0` disables culling.
    pub min_valid_rate: f64,
    /// Pieces per sample game used to measure per-tree validity for the cull.
    pub cull_sample_pieces: u64,
}

impl Default for EvolveConfig {
    fn default() -> Self {
        EvolveConfig {
            max_tree_depth: 24,
            max_cond_depth: 6,
            leaf_mutation_prob: 0.25,
            cond_mutation_prob: 0.25,
            grow_prob: 0.15,
            prune_prob: 0.10,
            tree_add_prob: 0.05,
            tree_remove_prob: 0.03,
            max_trees: 256,
            min_valid_rate: 1.0,
            cull_sample_pieces: 300,
        }
    }
}

/// Training-loop tuning.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub population: usize,
    pub generations: usize,
    pub elite: usize,
    pub tournament_k: usize,
    pub seeds: usize,
    pub start_cap: u64,
    pub init_trees: usize,
    pub evolve: EvolveConfig,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            population: 64,
            generations: 1000,
            elite: 8,
            tournament_k: 5,
            seeds: 8,
            start_cap: 200,
            init_trees: 64,
            evolve: EvolveConfig::default(),
        }
    }
}
```

- [ ] **Step 2: Write the failing test for `evolve.rs`**

Create `evolve.rs`:
```rust
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::config::EvolveConfig;
use crate::encode::NUM_INPUTS;
use crate::forest::{DecisionForest, FOREST_FORMAT_VERSION};
use crate::tree::{random_boolexpr_leaf, random_pid, BoolExpr, DecisionTree};

// (implementation added in Step 4)

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_game::tetris::TetrisPiecePlacement;

    fn sample_forest(rng: &mut SmallRng) -> DecisionForest {
        DecisionForest::random(rng, 16, 8, 4)
    }

    #[test]
    fn mutation_keeps_forest_valid_and_bounded() {
        let cfg = EvolveConfig::default();
        let n = TetrisPiecePlacement::NUM_PLACEMENTS as u8;
        let mut rng = SmallRng::seed_from_u64(11);
        let parent = sample_forest(&mut rng);
        for _ in 0..100 {
            let child = mutate_forest(&parent, &mut rng, &cfg);
            assert!(!child.trees.is_empty());
            assert!(child.trees.len() <= cfg.max_trees);
            for t in &child.trees {
                assert!(t.decide(&[0u8; NUM_INPUTS]) < n, "pid out of range");
            }
        }
    }

    #[test]
    fn crossover_produces_valid_forest() {
        let mut rng = SmallRng::seed_from_u64(21);
        let a = sample_forest(&mut rng);
        let b = sample_forest(&mut rng);
        let child = crossover(&a, &b, &mut rng);
        assert!(!child.trees.is_empty());
        assert_eq!(child.version, FOREST_FORMAT_VERSION);
        let n = TetrisPiecePlacement::NUM_PLACEMENTS as u8;
        for t in &child.trees {
            assert!(t.decide(&[0u8; NUM_INPUTS]) < n);
        }
    }

    #[test]
    fn tournament_picks_the_best_of_its_sample() {
        let mut rng = SmallRng::seed_from_u64(31);
        let scored: Vec<(f32, DecisionForest)> = (0..8)
            .map(|i| (i as f32, DecisionForest::random(&mut rng, 2, 2, 2)))
            .collect();
        let mut rng2 = SmallRng::seed_from_u64(1);
        let winner = tournament_select(&scored, &mut rng2, scored.len());
        let bits = [0u8; NUM_INPUTS];
        let legal = [
            TetrisPiecePlacement::from_index(0),
            TetrisPiecePlacement::from_index(1),
        ];
        assert_eq!(winner.vote(&bits, &legal), scored.last().unwrap().1.vote(&bits, &legal));
    }
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```sh
cargo test -p tetris-playground --bin tetris_decision_forest evolve::tests
```
Expected: FAIL to compile — `mutate_forest`, `crossover`, `tournament_select` not found.

- [ ] **Step 4: Write the implementation**

Add above the test module in `evolve.rs`:
```rust
/// Return a mutated copy of `parent`: tree add/remove, plus one structural
/// mutation on a random decision node of a random tree (mutate a leaf's
/// placement id, mutate/regrow a branch's boolean condition, grow a leaf into a
/// branch, or prune a branch to a leaf). Always leaves ≥ 1 tree and ≤ `max_trees`.
pub fn mutate_forest(
    parent: &DecisionForest,
    rng: &mut SmallRng,
    cfg: &EvolveConfig,
) -> DecisionForest {
    let mut child = parent.clone();

    if child.trees.len() < cfg.max_trees && rng.random_bool(cfg.tree_add_prob) {
        child
            .trees
            .push(DecisionTree::random(rng, cfg.max_tree_depth, cfg.max_cond_depth));
    }
    if child.trees.len() > 1 && rng.random_bool(cfg.tree_remove_prob) {
        let idx = rng.random_range(0..child.trees.len());
        child.trees.remove(idx);
    }

    if !child.trees.is_empty() {
        let tidx = rng.random_range(0..child.trees.len());
        let n = child.trees[tidx].count_nodes();
        let target = rng.random_range(0..n);
        if let Some(node) = child.trees[tidx].node_at_mut(target) {
            mutate_tree_node(node, rng, cfg);
        }
    }

    child.version = FOREST_FORMAT_VERSION;
    child
}

fn mutate_tree_node(node: &mut DecisionTree, rng: &mut SmallRng, cfg: &EvolveConfig) {
    match node {
        DecisionTree::Leaf(pid) => {
            if rng.random_bool(cfg.grow_prob) {
                *node = DecisionTree::Branch {
                    cond: BoolExpr::random(rng, cfg.max_cond_depth.min(3).max(1)),
                    yes: Box::new(DecisionTree::Leaf(random_pid(rng))),
                    no: Box::new(DecisionTree::Leaf(*pid)),
                };
            } else {
                let _ = cfg.leaf_mutation_prob; // reserved knob (kept referenced)
                *pid = random_pid(rng);
            }
        }
        DecisionTree::Branch { cond, yes, .. } => {
            if rng.random_bool(cfg.prune_prob) {
                *node = DecisionTree::Leaf(first_leaf_pid(yes));
            } else if rng.random_bool(cfg.cond_mutation_prob) {
                mutate_cond(cond, rng, cfg);
            } else {
                *cond = BoolExpr::random(rng, cfg.max_cond_depth);
            }
        }
    }
}

/// Placement id of the first leaf reached by always taking `yes`.
fn first_leaf_pid(tree: &DecisionTree) -> u8 {
    let mut node = tree;
    loop {
        match node {
            DecisionTree::Leaf(p) => return *p,
            DecisionTree::Branch { yes, .. } => node = yes,
        }
    }
}

/// Mutate a boolean condition in place.
fn mutate_cond(cond: &mut BoolExpr, rng: &mut SmallRng, cfg: &EvolveConfig) {
    let n = cond.count_nodes();
    let target = rng.random_range(0..n);
    let Some(node) = cond.node_at_mut(target) else {
        return;
    };
    let is_gate = !matches!(node, BoolExpr::Cell(_) | BoolExpr::Const(_));

    if is_gate && rng.random_bool(cfg.prune_prob) {
        *node = random_boolexpr_leaf(rng);
        return;
    }
    if !is_gate && rng.random_bool(cfg.grow_prob) {
        *node = BoolExpr::random(rng, 2);
        return;
    }
    match node {
        BoolExpr::Cell(i) => {
            *i = rng.random_range(0..crate::encode::NUM_INPUTS);
        }
        BoolExpr::Const(b) => {
            *b = !*b;
        }
        BoolExpr::Not(_) => {
            let taken = std::mem::replace(node, BoolExpr::Const(false));
            if let BoolExpr::Not(inner) = taken {
                *node = *inner; // double-negation elimination
            } else {
                *node = taken;
            }
        }
        _ => {
            let taken = std::mem::replace(node, BoolExpr::Const(false));
            *node = match taken {
                BoolExpr::And(a, b) => BoolExpr::Or(a, b),
                BoolExpr::Or(a, b) => BoolExpr::Xor(a, b),
                BoolExpr::Xor(a, b) => BoolExpr::And(a, b),
                other => other,
            };
        }
    }
}

/// Subtree crossover: copy `a`, take a uniformly-random decision subtree from a
/// random tree of `b`, and graft it over a uniformly-random node of a random
/// tree of the copy. Falls back to `a` when either parent is empty.
pub fn crossover(a: &DecisionForest, b: &DecisionForest, rng: &mut SmallRng) -> DecisionForest {
    let mut child = a.clone();
    if child.trees.is_empty() || b.trees.is_empty() {
        return child;
    }
    let b_tree = &b.trees[rng.random_range(0..b.trees.len())];
    let b_n = b_tree.count_nodes();
    let mut donor = b_tree.clone();
    let donor_subtree = donor
        .node_at_mut(rng.random_range(0..b_n))
        .map(|n| n.clone())
        .unwrap_or_else(|| DecisionTree::Leaf(random_pid(rng)));

    let a_idx = rng.random_range(0..child.trees.len());
    let a_n = child.trees[a_idx].count_nodes();
    if let Some(slot) = child.trees[a_idx].node_at_mut(rng.random_range(0..a_n)) {
        *slot = donor_subtree;
    }
    child.version = FOREST_FORMAT_VERSION;
    child
}

/// Tournament selection: sample `k` competitors and return the best.
pub fn tournament_select<'a>(
    scored: &'a [(f32, DecisionForest)],
    rng: &mut SmallRng,
    k: usize,
) -> &'a DecisionForest {
    debug_assert!(!scored.is_empty());
    let k = k.max(1).min(scored.len());
    let mut best_idx = rng.random_range(0..scored.len());
    for _ in 1..k {
        let idx = rng.random_range(0..scored.len());
        if scored[idx].0 > scored[best_idx].0 {
            best_idx = idx;
        }
    }
    &scored[best_idx].1
}
```

- [ ] **Step 5: Declare the modules in main.rs**

Add `mod config;` and `mod evolve;` to `main.rs`.

- [ ] **Step 6: Run the tests to verify they pass**

Run:
```sh
cargo test -p tetris-playground --bin tetris_decision_forest evolve::tests
```
Expected: all three tests PASS.

- [ ] **Step 7: Leave changes unstaged** (no `git add`/`git commit`).

---

### Task 7: Fitness (survival), autokill & the training loop (`train.rs`)

Survival fitness, the invalid-tree autokill, the generational loop with the escalating piece-cap curriculum, checkpointing, and a CSV log.

**Files:**
- Create: `crates/tetris-playground/src/bin/research/tetris_decision_forest/train.rs`
- Modify: `main.rs` (add `mod train;`)

**Interfaces:**
- Consumes: `config::{EvolveConfig, TrainConfig}`, `encode::encode_game`, `forest::DecisionForest`, `policy::play_game`, `evolve::{crossover, mutate_forest, tournament_select}`, `tetris_game::{config::StandardTetris, tetris::{IsLost, TetrisGame}}`, `rayon::prelude::*`, `rand::rngs::SmallRng`, `anyhow::Result`.
- Produces:
  - `pub struct FitnessResult { pub mean_pieces: f32, pub min_pieces: u64, pub max_pieces: u64, pub mean_lines: f32 }` (derives `Clone, Copy, Debug`).
  - `pub fn fitness(&DecisionForest, seeds: &[u64], cap: u64) -> FitnessResult`.
  - `pub fn tree_valid_counts(&DecisionForest, seed: u64, sample_pieces: u64) -> (Vec<u64>, u64)`.
  - `pub fn cull_invalid_trees(&DecisionForest, seeds: &[u64], cfg: &EvolveConfig) -> DecisionForest`.
  - `pub struct TrainArtifacts { pub out_dir: std::path::PathBuf, pub run_name: String }`.
  - `pub fn train(cfg: &TrainConfig, artifacts: &TrainArtifacts, seed: u64, resume: Option<DecisionForest>, stop: &std::sync::atomic::AtomicBool) -> anyhow::Result<DecisionForest>`.

- [ ] **Step 1: Write the failing test**

Create `train.rs`:
```rust
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use tetris_game::config::StandardTetris;
use tetris_game::tetris::{IsLost, TetrisGame};

use crate::config::{EvolveConfig, TrainConfig};
use crate::encode::encode_game;
use crate::evolve::{crossover, mutate_forest, tournament_select};
use crate::forest::DecisionForest;
use crate::policy::play_game;

// (implementation added in Step 3)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitness_reports_survival() {
        let mut rng = SmallRng::seed_from_u64(2);
        let forest = DecisionForest::random(&mut rng, 8, 6, 3);
        let seeds = [1u64, 2, 3];
        let r = fitness(&forest, &seeds, 300);
        assert!(r.min_pieces <= r.max_pieces);
        assert!(r.mean_pieces >= r.min_pieces as f32 - 0.001);
        assert!(r.mean_pieces <= r.max_pieces as f32 + 0.001);
    }

    #[test]
    fn cull_never_empties_and_only_removes() {
        let mut rng = SmallRng::seed_from_u64(4);
        let forest = DecisionForest::random(&mut rng, 12, 6, 3);
        let mut cfg = EvolveConfig::default(); // min_valid_rate 1.0 (aggressive)
        cfg.cull_sample_pieces = 120;
        let culled = cull_invalid_trees(&forest, &[1u64, 2], &cfg);
        assert!(!culled.trees.is_empty(), "cull emptied the forest");
        assert!(culled.trees.len() <= forest.trees.len());
    }

    #[test]
    fn cull_disabled_is_identity() {
        let mut rng = SmallRng::seed_from_u64(6);
        let forest = DecisionForest::random(&mut rng, 8, 6, 3);
        let mut cfg = EvolveConfig::default();
        cfg.min_valid_rate = 0.0;
        let culled = cull_invalid_trees(&forest, &[1u64], &cfg);
        assert_eq!(culled.trees.len(), forest.trees.len());
    }

    #[test]
    fn train_writes_champion_and_csv() {
        let cfg = TrainConfig {
            population: 12,
            generations: 4,
            elite: 3,
            tournament_k: 3,
            seeds: 3,
            start_cap: 150,
            init_trees: 8,
            ..TrainConfig::default()
        };
        let dir = std::env::temp_dir().join("df_train_test");
        let artifacts = TrainArtifacts {
            out_dir: dir.clone(),
            run_name: "unit".to_string(),
        };
        let stop = AtomicBool::new(false);
        let champion = train(&cfg, &artifacts, 7, None, &stop).unwrap();
        assert!(dir.join("unit_champion.json").exists(), "champion not written");
        let reloaded = DecisionForest::load_json(&dir.join("unit_champion.json")).unwrap();
        assert_eq!(reloaded.trees.len(), champion.trees.len());
        assert!(dir.join("unit_log.csv").exists(), "csv log not written");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```sh
cargo test -p tetris-playground --bin tetris_decision_forest train::tests
```
Expected: FAIL to compile — `fitness`, `cull_invalid_trees`, `train`, `TrainArtifacts` not found.

- [ ] **Step 3: Write the implementation**

Add above the test module in `train.rs`:
```rust
use std::io::Write;

/// Survival stats of a forest over a seed set at a piece cap. The fitness
/// scalar the loop optimizes is `mean_pieces` — pure survival.
#[derive(Clone, Copy, Debug)]
pub struct FitnessResult {
    pub mean_pieces: f32,
    pub min_pieces: u64,
    pub max_pieces: u64,
    pub mean_lines: f32,
}

/// Play `forest` on every seed up to `cap`, aggregating survival.
pub fn fitness(forest: &DecisionForest, seeds: &[u64], cap: u64) -> FitnessResult {
    let outcomes: Vec<_> = seeds.iter().map(|&s| play_game(forest, s, cap)).collect();
    let n = outcomes.len().max(1) as f32;
    let sum_pieces: u64 = outcomes.iter().map(|o| o.pieces_placed).sum();
    let sum_lines: u64 = outcomes.iter().map(|o| o.lines_cleared).sum();
    let min_pieces = outcomes.iter().map(|o| o.pieces_placed).min().unwrap_or(0);
    let max_pieces = outcomes.iter().map(|o| o.pieces_placed).max().unwrap_or(0);
    FitnessResult {
        mean_pieces: sum_pieces as f32 / n,
        min_pieces,
        max_pieces,
        mean_lines: sum_lines as f32 / n,
    }
}

/// Per-tree valid-output counts over one sample game from `seed`, plus the
/// number of decisions. A tree is "valid" at a decision when its placement id
/// is legal for the current piece (`DecisionTree::output_is_valid`). The game
/// advances by the forest's actual vote.
pub fn tree_valid_counts(
    forest: &DecisionForest,
    seed: u64,
    sample_pieces: u64,
) -> (Vec<u64>, u64) {
    let mut game = TetrisGame::<StandardTetris>::new_with_seed(seed);
    let mut valid = vec![0u64; forest.trees.len()];
    let mut decisions = 0u64;
    while decisions < sample_pieces {
        if game.board.is_lost() {
            break;
        }
        let bits = encode_game(&game);
        let legal = game.current_placements();
        if legal.is_empty() {
            break;
        }
        for (i, tree) in forest.trees.iter().enumerate() {
            if tree.output_is_valid(&bits, legal) {
                valid[i] += 1;
            }
        }
        let Some(placement) = forest.vote(&bits, legal) else {
            break;
        };
        let lost = game.apply_placement(placement).is_lost == IsLost::LOST;
        decisions += 1;
        if lost {
            break;
        }
    }
    (valid, decisions)
}

/// **Autokill invalid-output trees.** Measure each tree's valid-output rate over
/// sample games from `seeds` and drop trees below `cfg.min_valid_rate` (default
/// 1.0 = any tree that ever outputs an invalid placement is culled). Never
/// empties a forest — if every tree is below threshold, the single most-valid
/// tree is kept. `min_valid_rate <= 0.0` disables culling.
pub fn cull_invalid_trees(
    forest: &DecisionForest,
    seeds: &[u64],
    cfg: &EvolveConfig,
) -> DecisionForest {
    if cfg.min_valid_rate <= 0.0 || forest.trees.is_empty() {
        return forest.clone();
    }
    let n = forest.trees.len();
    let mut total_valid = vec![0u64; n];
    let mut total_dec = 0u64;
    for &seed in seeds {
        let (valid, dec) = tree_valid_counts(forest, seed, cfg.cull_sample_pieces);
        for i in 0..n {
            total_valid[i] += valid[i];
        }
        total_dec += dec;
    }
    if total_dec == 0 {
        return forest.clone();
    }
    let kept: Vec<_> = forest
        .trees
        .iter()
        .enumerate()
        .filter(|(i, _)| total_valid[*i] as f64 / total_dec as f64 >= cfg.min_valid_rate)
        .map(|(_, t)| t.clone())
        .collect();
    let trees = if kept.is_empty() {
        let best = (0..n).max_by_key(|&i| total_valid[i]).unwrap();
        vec![forest.trees[best].clone()]
    } else {
        kept
    };
    DecisionForest {
        version: crate::forest::FOREST_FORMAT_VERSION,
        trees,
    }
}

/// Where to write outputs.
pub struct TrainArtifacts {
    pub out_dir: PathBuf,
    pub run_name: String,
}

/// The curriculum ladder of per-game piece caps.
const CURRICULUM: [u64; 5] = [200, 1_000, 10_000, 100_000, 1_000_000];

/// Run the genetic-programming training loop. Fitness is survival (mean pieces).
/// Offspring are autokilled of invalid-output trees each generation. Escalates
/// the piece cap up `CURRICULUM` once the champion survives the current cap on
/// every eval seed. Writes `<run>_champion.json` on improvement, periodic
/// `<run>_gen_<n>.json` snapshots, and a `<run>_log.csv`.
pub fn train(
    cfg: &TrainConfig,
    artifacts: &TrainArtifacts,
    seed: u64,
    resume: Option<DecisionForest>,
    stop: &AtomicBool,
) -> Result<DecisionForest> {
    std::fs::create_dir_all(&artifacts.out_dir)?;
    let csv_path = artifacts.out_dir.join(format!("{}_log.csv", artifacts.run_name));
    let mut csv = std::io::BufWriter::new(std::fs::File::create(&csv_path)?);
    writeln!(
        csv,
        "generation,cap,cap_stage,best_survival,mean_survival,champ_min_pieces,champ_mean_pieces,champ_max_pieces,champ_mean_lines,champ_trees,champ_nodes"
    )?;

    let mut master = SmallRng::seed_from_u64(seed);
    let mut population: Vec<DecisionForest> = Vec::with_capacity(cfg.population);
    if let Some(seed_forest) = resume {
        population.push(seed_forest.clone());
        for _ in 1..cfg.population {
            population.push(mutate_forest(&seed_forest, &mut master, &cfg.evolve));
        }
    } else {
        for _ in 0..cfg.population {
            population.push(DecisionForest::random(
                &mut master,
                cfg.init_trees,
                cfg.evolve.max_tree_depth,
                cfg.evolve.max_cond_depth,
            ));
        }
    }

    let mut cap_stage = CURRICULUM
        .iter()
        .position(|&c| c >= cfg.start_cap)
        .unwrap_or(0);
    let mut champion = population[0].clone();
    let mut champion_fit = f32::NEG_INFINITY;

    for generation in 0..cfg.generations {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let cap = CURRICULUM[cap_stage];
        let seeds: Vec<u64> = (0..cfg.seeds)
            .map(|i| generation as u64 * 1_000_003 + i as u64 + 1)
            .collect();

        let scored: Vec<(f32, DecisionForest)> = population
            .par_iter()
            .map(|f| (fitness(f, &seeds, cap).mean_pieces, f.clone()))
            .collect();

        let (best_fit, best_forest) = scored
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .cloned()
            .unwrap();
        let mean_fit = scored.iter().map(|s| s.0).sum::<f32>() / scored.len() as f32;

        if best_fit > champion_fit {
            champion_fit = best_fit;
            champion = best_forest.clone();
            champion.save_json(
                &artifacts.out_dir.join(format!("{}_champion.json", artifacts.run_name)),
            )?;
        }

        let champ_stats = fitness(&champion, &seeds, cap);
        writeln!(
            csv,
            "{generation},{cap},{cap_stage},{best_fit:.2},{mean_fit:.2},{},{:.2},{},{:.2},{},{}",
            champ_stats.min_pieces,
            champ_stats.mean_pieces,
            champ_stats.max_pieces,
            champ_stats.mean_lines,
            champion.trees.len(),
            champion.total_nodes()
        )?;
        csv.flush()?;

        if generation % 25 == 0 {
            println!(
                "gen {generation} | cap {cap} | best_survival {best_fit:.1} | mean {mean_fit:.1} | champ min/mean/max pieces {}/{:.0}/{} | lines {:.1} | trees {} nodes {}",
                champ_stats.min_pieces,
                champ_stats.mean_pieces,
                champ_stats.max_pieces,
                champ_stats.mean_lines,
                champion.trees.len(),
                champion.total_nodes()
            );
            champion.save_json(
                &artifacts.out_dir.join(format!("{}_gen_{generation:08}.json", artifacts.run_name)),
            )?;
        }

        if champ_stats.min_pieces >= cap && cap_stage + 1 < CURRICULUM.len() {
            cap_stage += 1;
            champion_fit = f32::NEG_INFINITY;
            println!(
                "── curriculum: champion cleared cap {cap} on all seeds; escalating to {} ──",
                CURRICULUM[cap_stage]
            );
        }

        // Next generation: elitism, then tournament crossover + mutation, then
        // autokill invalid-output trees from each offspring (one short sample).
        let mut ranked = scored.clone();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let mut next: Vec<DecisionForest> = Vec::with_capacity(cfg.population);
        for entry in ranked.iter().take(cfg.elite.min(ranked.len())) {
            next.push(entry.1.clone());
        }
        let cull_seed = &seeds[..seeds.len().min(1)];
        while next.len() < cfg.population {
            let parent_a = tournament_select(&scored, &mut master, cfg.tournament_k);
            let parent_b = tournament_select(&scored, &mut master, cfg.tournament_k);
            let crossed = crossover(parent_a, parent_b, &mut master);
            let child = mutate_forest(&crossed, &mut master, &cfg.evolve);
            let child = cull_invalid_trees(&child, cull_seed, &cfg.evolve);
            next.push(child);
        }
        population = next;
    }

    champion.save_json(
        &artifacts.out_dir.join(format!("{}_champion.json", artifacts.run_name)),
    )?;
    Ok(champion)
}
```

- [ ] **Step 4: Declare the module in main.rs**

Add `mod train;` to `main.rs`.

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```sh
cargo test -p tetris-playground --bin tetris_decision_forest train::tests
```
Expected: all four tests PASS.

- [ ] **Step 6: Leave changes unstaged** (no `git add`/`git commit`).

---

### Task 8: CLI wiring (`main.rs`)

Replace the random-play stub with a full clap CLI: `train`, `play`, `eval`, `bench`, `show`.

**Files:**
- Modify: `crates/tetris-playground/src/bin/research/tetris_decision_forest/main.rs`

**Interfaces:**
- Consumes: everything above, plus `clap::{Parser, Subcommand}`, `ctrlc`, `std::sync::atomic::{AtomicBool, Ordering}`, `std::sync::Arc`.
- Produces: the final CLI.

- [ ] **Step 1: Rewrite `main.rs` with the full CLI**

Replace the entire contents of `main.rs` with:
```rust
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

mod config;
mod encode;
mod evolve;
mod forest;
mod policy;
mod train;
mod tree;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::config::{EvolveConfig, TrainConfig};
use crate::forest::DecisionForest;
use crate::policy::play_game;
use crate::train::{fitness, train, TrainArtifacts};

#[derive(Parser)]
#[command(name = "tetris_decision_forest")]
#[command(about = "Evolvable decision forest (boolean-op trees, placement votes) that plays Tetris")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Evolve a population of forests up the piece-cap curriculum (fitness = survival).
    Train(TrainArgs),
    /// Play one game with a saved forest on a given seed.
    Play(PlayArgs),
    /// Evaluate a saved (or random) forest across N seeds.
    Eval(EvalArgs),
    /// Stress-test a saved forest at the 1,000,000-piece cap.
    Bench(BenchArgs),
    /// Print structural stats of a saved forest.
    Show(ShowArgs),
}

#[derive(clap::Args)]
struct TrainArgs {
    #[arg(long, default_value = "df-run")]
    run_name: String,
    #[arg(long, default_value_t = 64)]
    population: usize,
    #[arg(long, default_value_t = 1000)]
    generations: usize,
    #[arg(long, default_value_t = 8)]
    elite: usize,
    #[arg(long, default_value_t = 8)]
    seeds: usize,
    #[arg(long, default_value_t = 5)]
    tournament_k: usize,
    #[arg(long, default_value_t = 64)]
    init_trees: usize,
    /// Max decision-tree depth (deep trees).
    #[arg(long, default_value_t = 24)]
    max_tree_depth: u32,
    /// Max boolean-condition depth per branch (wide conditions).
    #[arg(long, default_value_t = 6)]
    max_cond_depth: u32,
    /// Max trees per forest (many trees).
    #[arg(long, default_value_t = 256)]
    max_trees: usize,
    /// Autokill: cull trees whose valid-output rate is below this (1.0 = cull any invalid; 0 disables).
    #[arg(long, default_value_t = 1.0)]
    min_valid_rate: f64,
    #[arg(long, default_value_t = 200)]
    start_cap: u64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "./artifacts/decision_forest")]
    out_dir: PathBuf,
    /// Resume by seeding the population from this champion JSON.
    #[arg(long)]
    resume: Option<PathBuf>,
}

#[derive(clap::Args)]
struct PlayArgs {
    #[arg(long)]
    forest: PathBuf,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 1_000_000)]
    max_pieces: u64,
}

#[derive(clap::Args)]
struct EvalArgs {
    /// Forest JSON; if omitted, a fixed-seed random forest is used.
    #[arg(long)]
    forest: Option<PathBuf>,
    #[arg(long, default_value_t = 32)]
    seeds: usize,
    #[arg(long, default_value_t = 100_000)]
    max_pieces: u64,
}

#[derive(clap::Args)]
struct BenchArgs {
    #[arg(long)]
    forest: PathBuf,
    #[arg(long, default_value_t = 16)]
    seeds: usize,
}

#[derive(clap::Args)]
struct ShowArgs {
    #[arg(long)]
    forest: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Train(args) => run_train(args),
        Command::Play(args) => run_play(args),
        Command::Eval(args) => run_eval(args),
        Command::Bench(args) => run_bench(args),
        Command::Show(args) => run_show(args),
    }
}

fn run_train(args: TrainArgs) -> Result<()> {
    let evolve = EvolveConfig {
        max_tree_depth: args.max_tree_depth,
        max_cond_depth: args.max_cond_depth,
        max_trees: args.max_trees,
        min_valid_rate: args.min_valid_rate,
        ..EvolveConfig::default()
    };
    let cfg = TrainConfig {
        population: args.population,
        generations: args.generations,
        elite: args.elite,
        tournament_k: args.tournament_k,
        seeds: args.seeds,
        start_cap: args.start_cap,
        init_trees: args.init_trees,
        evolve,
    };
    let artifacts = TrainArtifacts {
        out_dir: args.out_dir,
        run_name: args.run_name,
    };
    let resume = match args.resume {
        Some(path) => Some(DecisionForest::load_json(&path)?),
        None => None,
    };

    let stop = Arc::new(AtomicBool::new(false));
    let stop_handler = Arc::clone(&stop);
    ctrlc::set_handler(move || {
        eprintln!("\n[ctrl-c] finishing current generation, then stopping…");
        stop_handler.store(true, Ordering::Relaxed);
    })
    .ok();

    let champion = train(&cfg, &artifacts, args.seed, resume, &stop)?;
    println!(
        "training complete: champion has {} trees, {} nodes",
        champion.trees.len(),
        champion.total_nodes()
    );
    Ok(())
}

fn run_play(args: PlayArgs) -> Result<()> {
    let forest = DecisionForest::load_json(&args.forest)?;
    let outcome = play_game(&forest, args.seed, args.max_pieces);
    println!(
        "seed {} | pieces {} | lines {} | max_height {} | {}",
        args.seed,
        outcome.pieces_placed,
        outcome.lines_cleared,
        outcome.max_height,
        if outcome.alive { "ALIVE (cap reached)" } else { "TOPPED OUT" }
    );
    Ok(())
}

fn run_eval(args: EvalArgs) -> Result<()> {
    let forest = match args.forest {
        Some(path) => DecisionForest::load_json(&path)?,
        None => {
            use rand::rngs::SmallRng;
            use rand::SeedableRng;
            let mut rng = SmallRng::seed_from_u64(0);
            DecisionForest::random(&mut rng, 64, 24, 6)
        }
    };
    let seeds: Vec<u64> = (0..args.seeds as u64).map(|i| i + 1).collect();
    let r = fitness(&forest, &seeds, args.max_pieces);
    println!(
        "eval over {} seeds @ cap {} | survival mean {:.1} min {} max {} | mean_lines {:.1}",
        args.seeds, args.max_pieces, r.mean_pieces, r.min_pieces, r.max_pieces, r.mean_lines
    );
    Ok(())
}

fn run_bench(args: BenchArgs) -> Result<()> {
    let forest = DecisionForest::load_json(&args.forest)?;
    let seeds: Vec<u64> = (0..args.seeds as u64).map(|i| i + 1).collect();
    let r = fitness(&forest, &seeds, 1_000_000);
    let cleared_all = r.min_pieces >= 1_000_000;
    println!(
        "BENCH @ 1,000,000 cap over {} seeds | survival mean {:.0} min {} max {} | mean_lines {:.0} | {}",
        args.seeds,
        r.mean_pieces,
        r.min_pieces,
        r.max_pieces,
        r.mean_lines,
        if cleared_all { "*** GOAL REACHED: survived 1M on every seed ***" } else { "not yet 1M on all seeds" }
    );
    Ok(())
}

fn run_show(args: ShowArgs) -> Result<()> {
    let forest = DecisionForest::load_json(&args.forest)?;
    println!("forest: {} trees, {} total nodes", forest.trees.len(), forest.total_nodes());
    for (i, t) in forest.trees.iter().enumerate() {
        println!("  tree {i}: {} nodes, depth {}", t.count_nodes(), t.depth());
    }
    Ok(())
}
```

- [ ] **Step 2: Build and check `--help`**

Run:
```sh
cargo run --release -p tetris-playground --bin tetris_decision_forest -- --help
```
Expected: compiles; prints the top-level help listing `train`, `play`, `eval`, `bench`, `show`.

- [ ] **Step 3: Smoke-test `eval` on a random forest**

Run:
```sh
cargo run --release -p tetris-playground --bin tetris_decision_forest -- eval --seeds 4 --max-pieces 500
```
Expected: prints an `eval over 4 seeds …` line with `survival mean` ≥ 0.

- [ ] **Step 4: Run the full test suite**

Run:
```sh
cargo test -p tetris-playground --bin tetris_decision_forest
```
Expected: all tests from Tasks 2–7 PASS.

- [ ] **Step 5: Leave changes unstaged** (no `git add`/`git commit`).

---

### Task 9: End-to-end verification, lint, and docs sync

Prove the whole system runs a real short training, then clean up and align docs.

**Files:**
- Modify (only if flags drifted): `crates/tetris-playground/src/bin/research/tetris_decision_forest/README.md`

**Interfaces:**
- Consumes: the finished binary.
- Produces: a verified, lint-clean system and accurate docs — all left unstaged.

- [ ] **Step 1: Run a short real training to a checkpoint**

Run:
```sh
cargo run --release -p tetris-playground --bin tetris_decision_forest -- train \
  --run-name smoke --population 24 --generations 30 --seeds 4 --init-trees 32 --start-cap 200 \
  --out-dir ./artifacts/decision_forest_smoke
```
Expected: prints per-generation lines; on completion `artifacts/decision_forest_smoke/smoke_champion.json` and `smoke_log.csv` exist. Note: with `min_valid_rate = 1.0` (cull any invalid-output tree), forests start small and grow as piece-conditional trees evolve; if survival stays stuck at ~0 for many generations, lower `--min-valid-rate` (e.g. 0.9) to confirm the pipeline, then decide on the training regime.

- [ ] **Step 2: Confirm the champion plays and is inspectable**

Run:
```sh
cargo run --release -p tetris-playground --bin tetris_decision_forest -- play \
  --forest ./artifacts/decision_forest_smoke/smoke_champion.json --seed 99 --max-pieces 5000
cargo run --release -p tetris-playground --bin tetris_decision_forest -- show \
  --forest ./artifacts/decision_forest_smoke/smoke_champion.json
```
Expected: `play` prints a survival line; `show` prints per-tree node/depth counts.

- [ ] **Step 3: Format and lint**

Run:
```sh
cargo fmt -p tetris-playground
cargo clippy -p tetris-playground --bin tetris_decision_forest -- -D warnings
```
Expected: fmt clean; clippy passes with zero warnings. If clippy flags the reserved `leaf_mutation_prob` knob, wire it into a real mutation branch or remove the field.

- [ ] **Step 4: Sync the README CLI section**

Open `README.md` and confirm every flag matches the final clap args from Task 8 (`--run-name`, `--population`, `--generations`, `--elite`, `--seeds`, `--tournament-k`, `--init-trees`, `--max-tree-depth`, `--max-cond-depth`, `--max-trees`, `--min-valid-rate`, `--start-cap`, `--seed`, `--out-dir`, `--resume` for `train`; `--forest`, `--seed`, `--max-pieces` for `play`; etc.). Fix any drift.

- [ ] **Step 5: Clean up smoke artifacts**

Run:
```sh
rm -rf ./artifacts/decision_forest_smoke
```

- [ ] **Step 6: Final review — everything unstaged**

Run:
```sh
git status --short
```
Confirm the whole system (all bin files + the `Cargo.toml` edit + any README fix) is present as **unstaged** changes. Do not `git add` or `git commit`; hand off to the user for review.

---

## Post-plan usage: driving toward 1,000,000 pieces

1. **Launch a long train** (deep, many trees):
   ```sh
   cargo run --release -p tetris-playground --bin tetris_decision_forest -- train \
     --run-name df-main --population 96 --generations 100000 --seeds 12 \
     --init-trees 128 --start-cap 200 --out-dir ./artifacts/decision_forest
   ```
   The curriculum escalates `200 → 1k → 10k → 100k → 1M` as the champion masters each rung. Ctrl-C stops cleanly after the current generation.

2. **Periodically bench** the champion at the 1M cap:
   ```sh
   cargo run --release -p tetris-playground --bin tetris_decision_forest -- bench \
     --forest ./artifacts/decision_forest/df-main_champion.json --seeds 32
   ```
   Goal reached at `*** GOAL REACHED: survived 1M on every seed ***`.

3. **Resume / warm-start** from a good champion:
   ```sh
   cargo run --release -p tetris-playground --bin tetris_decision_forest -- train \
     --run-name df-main2 --resume ./artifacts/decision_forest/df-main_champion.json \
     --population 96 --generations 100000 --seeds 24 --start-cap 100000
   ```

If aggressive culling (`min_valid_rate 1.0`) stalls early learning, relax it (`--min-valid-rate 0.9`) to let partially-piece-aware trees survive and improve. Because trees see the current piece and bag state, they can learn genuinely piece-conditional and lookahead-aware placements — the reason the input includes those bits.

---

## Execution deviations (applied during the build — the code is the source of truth)

The plan was executed inline on 2026-07-09; all tasks completed and all 21 tests
pass in ~0.2 s. Three deviations were made against this document, all in the
direction of correctness/performance, and the code + README/DESIGN reflect them:

1. **`EvolveConfig.init_tree_depth` (default 6) was added, and random tree
   generation uses it instead of `max_tree_depth`.** Random generation is a
   supercritical branching process (leaf prob 0.25 ⇒ mean 1.5 offspring ⇒
   expected size ~1.5^depth): generating at depth 24 produced astronomically
   large trees and hung the first test run. Trees now start small and GROW via
   mutation; `max_tree_depth` is the structural cap — `mutate_forest` computes
   `allow_grow = tree.depth() < max_tree_depth` and grow mutations are gated on
   it. The CLI gained `--init-tree-depth` (and `run_eval`'s fallback random
   forest uses depth 6).
2. **Test parameters were shrunk so the whole suite is fast** (the train test
   uses population 6 / generations 2 / seeds 2 / init_trees 4 / init depth 3 /
   cull sample 30; the cull tests use small forests and 40-piece samples).
   Tests must stay quick — do not reintroduce real-sized training in unit tests.
3. **TDD red-steps were verified by compile-failure of the test-only file**, as
   written, for encode/tree/forest; later modules (policy/config/evolve/train)
   were written implementation+tests together since their interfaces were
   already triple-reviewed — every module still carries its full test set.

4. **Post-build hardening pass (same day):** (a) *performance* — the training
   loop no longer clones the population twice per generation (scores are kept
   by index; elites cloned by ranked index), offspring creation + the
   per-offspring cull run in parallel with per-child derived RNG seeds
   (determinism preserved — covered by a test), `tree_valid_counts` evaluates
   each tree once per turn against an O(1) legality mask and shares the vote's
   `winner_from_tally`, and a held-over champion is only re-evaluated when it
   did not come from the current generation. Measured on an identical
   20-generation workload: 9.53 s → 6.78 s (~29% faster). (b) *observability* —
   `TrainConfig.log_every` / `--log-every` (default 1: a progress line every
   generation); `eval`/`bench` report pieces/sec. (c) *tests* — five added:
   aggressive cull keeps a hand-built piece-conditional (100%-valid) tree and
   kills a constant tree; train determinism (same seed ⇒ byte-identical
   champion JSON); vote-always-legal property across random forests × real
   game states; placement-id round-trip over the full range; mutation respects
   the depth cap. Suite: 26 tests in ~0.25 s.

5. **Abstention (user request, same day):** `DecisionTree` gained an `Abstain`
   leaf ("no recommendation"); `decide` returns `Option<u8>`; abstainers add
   nothing to the tally; abstention counts as VALID for the autokill (silence
   is sanctioned — it is the specialization mechanism, letting a tree vote only
   on boards it recognizes). Random generation emits ~15% abstain leaves
   (`ABSTAIN_LEAF_PROB`), and mutation toggles vote ↔ silence. TOTALITY is
   guaranteed and tested: the forest always answers in a live game — `vote` is
   `None` only for an empty legal set (impossible live); all-abstain tallies
   fall back to the lowest-index legal placement. Suite: 30 tests, ~0.25 s.

6. **Adversarial-review fixes (same day):** a two-reviewer verification pass
   confirmed the perf refactor semantically equivalent (0 defects) and surfaced
   four items, all fixed: crossover now enforces `max_tree_depth` (a graft that
   would exceed the cap is skipped — previously depth could stack unboundedly
   across generations, contradicting the docs); condition-grow now respects
   `max_cond_depth`; README's min-valid-rate wording qualified (rate is over the
   per-generation sample game; forests are never emptied); and two tests added
   (JSON version rejection; cull most-valid-survivor fallback), plus
   `crossover_respects_the_depth_cap` and a cond-depth assertion.
   `FOREST_FORMAT_VERSION` bumped to 2 (node encoding gained `Abstain`).
   Suite: 33 tests, ~0.3 s.

7. **Evaluation hot-path optimization + opt-in snapshots (same day):**
   (a) `CompiledForest` (new `compiled.rs`): the boxed genome is flattened to
   contiguous index arenas once per `fitness` call and evaluated in that form —
   a pure representation change with a bit-identical-semantics property test
   (`compiled_forest_matches_boxed_semantics_exactly`). (b) `fitness` compiles
   once and shares it across all seeds; per-seed games run in PARALLEL only at
   caps ≥ 10,000 (`PAR_SEEDS_MIN_CAP`) — below that, population-level
   parallelism already saturates cores and rayon task overhead dominates
   (measured: naive per-game compile + unconditional seed-par cost 4× eval
   throughput before this gating). This makes `bench` at the 1M cap and
   high-cap champion re-evals parallel instead of serial. Measured at the
   short-game regime: train 12.53 s → 10.27 s on the identical 20-generation
   workload; eval throughput within noise of baseline. The wins grow with game
   length and tree size — exactly the production regime. (c) The every-25-gen
   champion snapshot caused a visible pause; snapshots are now opt-in via
   `--snapshot-every N` / `TrainConfig.snapshot_every: Option<usize>` (default
   off; the champion JSON is still saved on every improvement). Suite: 34
   tests, ~0.3 s.

8. **Piece-gated generation + regrowth — fix for the trees-1 collapse
   (2026-07-09, post-prod-run):** production runs stalled with every forest at
   1 tree. Diagnosis from run data (`df-prod1`: champion 128 → 1 trees at gen
   457, then 36k generations flat at ~38 pieces): random leaves drew placement
   ids uniformly (`random_pid`), but a placement id belongs to exactly one
   piece, so a random voting leaf is legal ~1/7 of decisions — under
   cull-at-1.0 essentially no random tree passes, the never-empty fallback
   keeps exactly one tree, and no mechanism regrows (tree-add's random tree
   dies to the same cull), so tree count ratchets 128 → 1 and forest-level
   voting degenerates to near-constant play. Fix in three parts, keeping
   cull-at-1.0 semantics intact: (a) all randomly generated trees (init,
   tree-add, regrowth) are now **piece-gated** (`random_piece_gated_tree`,
   `PIECE_GATE_DEPTH = 7`): root chain tests the 7 current-piece bits, the
   subtree under gate `i` votes only placements of piece `i` — 100 % valid by
   construction since `current_placements` is exactly the piece's table;
   (b) leaf retargets stay within the leaf's piece 80 % of the time
   (`SAME_PIECE_RETARGET_PROB`); (c) `regrow_forest` tops offspring back up to
   `EvolveConfig.min_trees` (default 32, `--min-trees`) with fresh gated trees
   after the cull. Depth accounting: gated trees are `PIECE_GATE_DEPTH + sub`
   deep, so generation clamps sub-depth against `max_tree_depth` and the two
   depth-cap tests initialize parents within the cap. Verified: fresh forests
   pass cull-at-1.0 intact (16/16), regrown trees 100 % valid on real games,
   gated trees vote only current-piece placements across random boards;
   120-gen rerun holds 32–64 trees per forest (population mean fitness 15 →
   23) instead of collapsing. Suite: 39 tests, ~0.6 s.

9. **Anti-stagnation overhaul (2026-07-09, after the piece-gating fix):** even
   with piece-gated generation, runs plateaued (~40 pieces) with the champion
   frozen and #trees pinned at exactly the `min_trees` floor. A multi-agent
   analysis (3 analysts + 3-lens judge panel over the code, the live
   `df-prod1` CSV, and the EC/GP technique space) found three root causes:
   (i) the champion bar was a high-water mark across ROTATING seed sets — a
   winner's-curse ratchet sitting ~3.4 σ of seed noise above the incumbent's
   true skill (updates ≈ 1-in-3000-generation events; the gen-1633 "update"
   installed a truly-worse genome); (ii) the floor was absorbing — tree-add
   injected a fresh RANDOM tree that perturbs the plurality randomly, so
   33-tree children always lost and attrition (cull + remove) ratcheted the
   count down onto the floor; (iii) the selection gradient vanished — one
   node mutation in one of 32 piece-gated trees is exercised on ~3 % of
   decisions and must flip a plurality, so most offspring were exact
   behavioral clones, and subtree crossover crossed piece gates 6/7 of the
   time (invalid votes → autokill destroys the evolved recipient tree).
   Five composable fixes landed (survival-only fitness, cull-at-1.0, and
   forest voting untouched): (a) SAME-SEED champion gate — the incumbent is
   re-scored on the current generation's seeds and replaced only if beaten on
   the identical set (train.rs; `champion_fit` high-water bar deleted);
   (b) seed-block freeze — `eval_seeds` freezes the seed set for
   `TrainConfig.seed_block` (default 20, `--seed-block`) generations;
   (c) WHOLE-TREE crossover — per-slot fair-coin mix of complete trees
   (evolve.rs; the depth cap now holds trivially); (d) duplicate-and-diverge
   tree-add (`DUP_TREE_PROB = 0.7`, `tree_add_prob` 0.05 → 0.10) — gene
   duplication as the fixation path above the floor; (e) size-scaled mutation
   (one node mutation per 8 trees, `TREES_PER_MUTATION`) + silent-specialist
   regrowth (`SILENT_TREE_ABSTAIN_PROB = 0.75` mostly-abstain injections) +
   branch mutation flipped so LOCAL condition tweaks are the default arm and
   wholesale regrow the rare one. Suite: 43 tests, ~0.8 s.

10. **Validity-aware mutation (same day, after adversarial review of #9):**
    a 3-lens review panel found the residual plateau driver: ~24 % of
    structural mutations still invalidated a gated tree (uniform node
    targeting hit the 7 gate-spine nodes ~8 % of the time and every arm is
    fatal there — prune collapses the chain, the new local-tweak default
    rerolls the gate's Cell 213/214 into breakage; grow arms and
    Abstain→Leaf used piece-blind `random_pid`, ~73 % / ~86 % invalid), so
    with size-scaled mutation each child lost ~1 evolved tree per generation
    to the cull, replaced by 75 %-abstain silent trees — churn + an abstain
    ratchet toward the constant fallback. Fix (evolve.rs/tree.rs):
    `mutate_tree_structurally` redraws targets off the gate spine
    (`piece_bit_piece`) and passes the target's `governing_piece` (innermost
    gate whose `yes` side contains the node) into `mutate_tree_node`; leaf
    retargets / Abstain revivals / grown leaves now draw pids of exactly the
    governing piece. New tests: `governing_piece_reads_the_gate_context`,
    `mutation_preserves_gated_tree_validity` (60-mutation chains keep ≥95 %
    of trees fully valid). Review defects NOT addressed (documented,
    deliberate): champion regression across seed-block boundaries is raw-data
    visible but unflagged; curriculum cannot de-escalate; cull samples one
    seed. Suite: 45 tests, ~0.9 s.

11. **Generation-time optimization, 37.7× measured (2026-07-09, user asked
    for 10×):** at the user's production flags (pop 100, seeds 12,
    init-trees 128, --max-cond-depth 16) one generation took ~148 s, with
    the `time -p` split user 272 s / **sys 2063 s** over 15 gens — i.e.
    ~90 % of wall time was kernel-side memory management from allocating and
    dropping millions of boxed nodes per generation. Three fixes:
    (a) `COND_GEN_DEPTH_CAP = 4` (tree.rs) — random condition GENERATION is
    capped at depth 4 regardless of `max_cond_depth` (which remains the
    growth cap): `BoolExpr::random` is a supercritical branching process, so
    generating at depth 16 emitted ~50–130-node random formulas per branch —
    unselected structure that every clone, cull game, and condition eval then
    paid for. Applied at population init, fresh/silent gated trees, and the
    wholesale cond-regrow mutation arm. (b) Clone elimination (evolve.rs,
    train.rs): crossover builds the child directly (one clone per chosen
    tree, was clone-all-of-a-then-overwrite), `mutate_forest_owned` and
    `cull_invalid_trees_owned` take ownership and edit in place (the cull
    retains survivors instead of cloning them) — the pipeline went from ~3
    full deep clones + survivor clones per child to ~1 partial clone.
    (c) Elite + incumbent fitness caching within a frozen seed block
    (fitness is deterministic there): elites and a held champion reuse last
    generation's scores — ~17 % fewer games at elite 16 / pop 100.
    MEASURED (same machine, same contention, /usr/bin/time -p, 15 gens from
    init at the user's exact flags): old 2226.55 s real (148 s/gen) → new
    59.01 s real (3.9 s/gen) = **37.7×**; sys time 2062.8 s → 52.8 s.
    Suite: 45 tests green; determinism test unchanged.

12. **PUCT teacher distillation (2026-07-13, user request; role confirmed
    via question = training-time teacher):** new `puct.rs` + `distill.rs` +
    `distill` subcommand — the AlphaZero improve→project cycle with the
    forest in the network's role, keeping the played artifact a pure
    state→move boolean circuit (no search at inference). Root-PUCT teacher:
    prior = the forest's Laplace-smoothed vote tally over legal placements;
    value = survival fraction of a rollout following the forest policy;
    selection = `Q + c_puct·P·√N/(1+N)`; each simulation copies the root
    (`TetrisGame` is `Copy`) and reseeds the game RNG so unknown piece
    futures are resampled (labels leak no hidden future); label = most
    visits (ties → higher Q → lower index). `distill` samples states by
    playing the prior, labels them in parallel, then runs the standard GP
    loop with fitness = AGREEMENT with the labels (deterministic set ⇒
    strict champion gate); output `<run>_distilled.json` resumes into
    `train`. Smoke on the prod champion: prior-teacher agreement 0.463 at
    gen 0 (the search disagrees with the forest on half the states — real
    improvement signal); resume verified. Also: `save_json` is now ATOMIC
    (temp + rename) — the live run's frequent champion rewrites made
    concurrent reads observably torn (5/5 copies mid-write were corrupt).
    Suite: 50 tests, ~3.7 s.

13. **PUCT refinement round (2026-07-13, "improve and refine and further
    test"):** four inline refinements + a 2-lens adversarial review + a real
    improvement-round experiment. Refinements: (a) distill-set DEDUP on the
    214-bit encoding (repeat labels wasted sims and skewed toward openings);
    (b) `--min-q-gap` label-noise filter — a disagreeing teacher label is
    kept only if its Q beats the prior move's Q by the gap (flat-Q states
    give arbitrary most-visited labels); (c) end-of-distill SURVIVAL GUARD
    (prior vs distilled fitness on 8 held-out seeds, cap 1000, WARN on
    regression) — catastrophic-forgetting detector for the proxy objective;
    (d) behavioral test `teacher_avoids_immediately_losing_placements`.
    Review found one MAJOR semantic bug, fixed: reseeding the game RNG does
    NOT resample the current bag — `TetrisPieceBag` stores an
    already-shuffled queue and only consults the RNG at refill, so every
    simulation replayed the root's true hidden piece order and labels leaked
    information the student's bag-membership bits cannot express. Fix: each
    simulation now also calls `bag.shuffle(&mut g.rng)` (membership
    preserved, order resampled; assumption pinned by
    `simulations_resample_bag_order_but_preserve_membership`). Also from
    review: round-robin first pass in `puct_root` (every action gets one
    real Q before PUCT allocation — unvisited Q=0 biased selection and made
    the Q-gap filter compare against a fake worst-case; the filter now also
    requires the prior move to be visited); rollout value off-by-one (fatal
    placement no longer counts as survived — uniform 1-unit death-depth
    scale); vote prior mixed 3:1 with uniform instead of +1-Laplace (which
    swamped small vote tallies to near-uniform); argmax rewritten
    first-max-wins (ties → lowest index, the codebase convention);
    `--sims` clap range ≥ 1 (zero sims would emit evidence-free labels);
    `TetrisGameRng::new` instead of a struct literal; test renamed to drop
    an unasserted "improves agreement" claim. Experiment (pre-fix teacher,
    768 states → 724 after Q-gap, sims 256, 300 gens): agreement 0.488 →
    0.575 monotone; survival guard 27.5 → 27.2 (no forgetting, no gain yet);
    re-run with the fixed teacher recorded in the session log. Suite:
    54 tests, ~1.8 s.

---

## Self-Review

**Spec coverage (incl. every user edit):**
- New research bin under `src/bin/research/tetris_decision_forest/`. ✓
- Decision tree/forest of boolean operations → `tree.rs` (`BoolExpr` And/Or/Not/Xor conditions; `DecisionTree` boolean branches → placement leaves), `forest.rs` (vote). ✓
- Unstaged / no commit / no branch — every task ends "leave unstaged"; global constraint + Task 1 Step 4 + Task 9 Step 6 enforce. ✓
- No `board_cells`; board via engine `to_cell_array` (inside `encode_game`). ✓
- Tree output is a placement (`DecisionTree::decide -> u8` placement id); no score type. ✓
- Deep/wide/many → defaults 24/6/256/64, all CLI-exposed. ✓
- Forest output is a vote → `DecisionForest::vote` tallies, plays most-voted legal. ✓
- Fitness = survival → `fitness.mean_pieces` is the sole optimized scalar. ✓
- Trees can't output invalid placements played (vote restricts to legal) AND invalid-output trees autokilled (`cull_invalid_trees`, default `min_valid_rate 1.0` = cull any invalid). ✓
- Inputs = board + bag + current piece as binary (`encode.rs`, `NUM_INPUTS = 214`). ✓
- Separate files: logic (`encode`/`tree`/`forest`/`policy`) vs training (`evolve`/`train`/`config`). ✓
- 1M pieces → curriculum + `bench`. ✓
- Plan + README + DESIGN. ✓
- All new code in one bin dir (only `Cargo.toml` outside). ✓

**Type consistency:**
- `encode_game(&TetrisGame<StandardTetris>) -> [u8; NUM_INPUTS]`; `&bits[..]` → `&[u8]` into `decide`/`eval`/`vote`/`output_is_valid`. ✓
- `BoolExpr`/`DecisionTree` methods + `random_pid`/`random_boolexpr_leaf` consistent across `tree.rs`/`evolve.rs`. ✓
- `DecisionForest{version,trees}` + `vote`/`random(rng,num_trees,max_depth,cond_depth)`/`total_nodes`/`save_json`/`load_json` consistent across `forest.rs`/`policy.rs`/`evolve.rs`/`train.rs`/`main.rs`. ✓
- `fitness`/`tree_valid_counts`/`cull_invalid_trees`/`train` signatures consistent in `train.rs`/`main.rs`. ✓
- `EvolveConfig` (incl. `min_valid_rate`, `cull_sample_pieces`) / `TrainConfig` field names consistent across `config.rs`/`evolve.rs`/`train.rs`/`main.rs`. ✓
- Placement-id API (`TetrisPiecePlacement::{index, from_index, NUM_PLACEMENTS}`); `TetrisPiece::all()`, `bag.contains`, `current_piece`. ✓

**Placeholder scan:** no `TODO`; every code step complete; every test step real asserts. Advisory notes (engine-path checks in Tasks 1–2, the reserved `leaf_mutation_prob` knob) each carry a concrete resolution. ✓
