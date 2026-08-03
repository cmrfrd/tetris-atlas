# Tetris Atlas PUCT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single-file binary `tetris_atlas_puct` that grows a canonical Tetris state graph with integer-only UCT rollouts plus an explicit expansion frontier, then extracts a candidate safe set via gfp closure.

**Architecture:** One flat file `crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs` organized as commented sections. Preallocated arenas hold fixed-size records addressed by dense `u32` ids; a DashMap is used only for interning; rollouts traverse cached ids; epochs are barriers (expand → rollout → closure → metrics).

**Tech Stack:** Rust nightly (workspace toolchain), `tetris-game` engine, `dashmap` (FxHash), `rayon`, `crossbeam-queue`, `crossbeam` (Backoff), `rustc-hash`, `clap`, `ctrlc`. All already dependencies of `tetris-playground`.

**Spec:** `docs/superpowers/specs/2026-07-13-tetris-atlas-puct-design.md` — read it before starting; §16 lists approved deviations from the user's original plan text.

## Global Constraints

- Single flat file: `crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs`. All tasks append/modify sections of this one file. Tests live in one `#[cfg(test)] mod tests` at the bottom.
- File header: `#![feature(generic_const_exprs)]` + `#![allow(incomplete_features)]` (required by `tetris-game` types; see sibling `tetris_atlas_inmemory.rs`).
- **No `f32`/`f64` outside `mod tests`.** Reporting uses integer math (ms, integer rates).
- **No `panic!`, `todo!`, `unimplemented!`, `unreachable!`** (denied by workspace lints). `unwrap`/`expect` only in `mod tests` (annotated `#[allow(clippy::unwrap_used, clippy::expect_used)]`) and at the single documented arena-invariant site (`Arena::get`).
- Sentinels: `UNEXPANDED = u32::MAX`, `EXPANDING = u32::MAX - 1`, `MAX_VALID_STATE_ID = u32::MAX - 2`. A branch `child <= MAX_VALID_STATE_ID` is a valid id.
- Atomic orderings: `Relaxed` for statistics counters; `Acquire`/`Release`/`AcqRel` for child publication and `expansion_status`; `SeqCst` for the rare `stop`/`exhausted` flags.
- Test command for every task: `cargo test -p tetris-playground --bin tetris_atlas_puct`. Also run `cargo fmt --all` before each commit.
- **Git discipline:** work on branch `tetris-atlas-puct` (Task 0 creates it off the current HEAD). Commit ONLY `crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs`. Do NOT commit `Cargo.toml` — it carries the user's unrelated pending edits; our `[[bin]]` entry rides uncommitted in the working tree and the user folds it into their own Cargo.toml commit. Never commit `Cargo.lock`, `docs/`, or anything else.
- Builds/tests run in the foreground, never backgrounded.

## Key `tetris-game` API facts (verified against `crates/tetris-game/src/tetris.rs`)

- `TetrisBoard` (default = `TetrisBoard<StandardTetris>`): `Copy + Hash + Ord`, `EMPTY_BOARD`, `.apply_piece_placement(p) -> PlacementResult { lines_cleared, is_lost: IsLost }` (mutates self), `.total_holes() -> u32`, `.height() -> u32`, `.heights() -> [u32; 10]`, `.count() -> u32`, `from_limbs([u32; 10])`.
- `bool::from(is_lost)` converts `IsLost`.
- `TetrisPiece`: `.index() -> u8` (0..7), `TetrisPiece::from_index(u8)`, `TetrisPiece::all() -> [TetrisPiece; 7]`, `NUM_PIECES = 7`.
- `TetrisPieceBagState`: `u8` newtype mask. `FULL_MASK: u8 = 0b0111_1111`, `From<u8>`/`Into<u8>`, `.contains(piece)`, `.remove(piece)` (`&mut`), `.fill()` (`&mut`), `.is_empty()`, `.count() -> u8`, `.iter_pieces()`, `.iter_next_states() -> impl Iterator<Item = (TetrisPiece, TetrisPieceBagState)>` (removes piece, refills if empty — the canonical bag transition).
- `TetrisPiecePlacement`: `Copy`, `.piece`, `.orientation`, `.index() -> u8`, `from_index(u8)`, `all_from_piece(piece) -> &[TetrisPiecePlacement]`, `MAX_PIECE_PLACEMENT_COUNT: usize` (= 34).
- `StandardTetris::COLS = 10`; import `TetrisGameConfig` trait to use it.

---

### Task 0: Branch, Cargo entry, compiling skeleton

**Files:**
- Create: `crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs`
- Modify: `crates/tetris-playground/Cargo.toml` (append `[[bin]]` after the `tetris_atlas_bagstate` entry; NOT committed — see Global Constraints)

**Interfaces:**
- Produces: `struct Args` (clap), `fn main() -> anyhow::Result<()>`, section comment scaffolding all later tasks append into.

- [ ] **Step 1: Create branch**

```bash
git checkout -b tetris-atlas-puct
```

- [ ] **Step 2: Add the bin entry to Cargo.toml**

After the `tetris_atlas_bagstate` `[[bin]]` block:

```toml
[[bin]]
name = "tetris_atlas_puct"
path = "src/bin/atlas/tetris_atlas_puct.rs"
```

(No `required-features` — this bin uses no optional backend.)

- [ ] **Step 3: Write the skeleton file**

```rust
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! # Tetris Atlas PUCT — integer-UCT graph search + gfp safe-set extraction
//!
//! Grows a canonical AND-OR state graph over complete Tetris states
//! `(board, remaining_bag, current_piece)`: an explicit expansion frontier
//! guarantees coverage, integer-only UCT rollouts prioritize promising and
//! dangerous regions, and a periodic greatest-fixpoint closure pass extracts
//! the candidate safe set
//! `K = { s | ∃ placement a, ∀ next piece q ∈ bag(s): T(s,a,q) ∈ K }`.
//!
//! Design spec: docs/superpowers/specs/2026-07-13-tetris-atlas-puct-design.md
//!
//! Memory is paid up-front: three preallocated arenas (~5 KB per state at the
//! configured `--max-states`) hold fixed-size records addressed by dense u32
//! ids. The DashMap is used only for interning; rollouts traverse cached ids.
//! The whole search is deterministic — no RNG anywhere; only thread scheduling
//! perturbs visit statistics.

use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// ============================================================================
// Config & CLI
// ============================================================================

#[derive(Parser, Debug, Clone)]
#[command(name = "tetris-atlas-puct")]
#[command(about = "PUCT-style graph search + gfp closure for a candidate Tetris safe set")]
struct Args {
    /// Worker threads (0 = all cores).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Number of epochs to run (default: until Ctrl-C).
    #[arg(long)]
    epochs: Option<u64>,
    /// Rollouts per epoch.
    #[arg(long, default_value_t = 100_000)]
    rollouts_per_epoch: u64,
    /// Frontier states fully expanded per epoch.
    #[arg(long, default_value_t = 10_000)]
    expansions_per_epoch: u64,
    /// Run a closure pass every N epochs.
    #[arg(long, default_value_t = 10)]
    closure_every: u64,
    /// Rollout horizon in pieces. Statistics are only comparable within one horizon.
    #[arg(long, default_value_t = 128)]
    horizon: usize,
    /// State arena capacity. ~5 KB of up-front arena memory per state.
    #[arg(long, default_value_t = 500_000)]
    max_states: u32,
    /// Placement arena capacity (default: max_states * 34).
    #[arg(long)]
    max_placements: Option<u32>,
    /// Branch arena capacity (default: placements * 7).
    #[arg(long)]
    max_branches: Option<u32>,
    /// UCT exploration constant, fixed-point with 20 fractional bits (1_048_576 = 1.0).
    #[arg(long, default_value_t = 1_048_576)]
    c_uct: u64,
    /// LCB uncertainty constant, fixed-point with 20 fractional bits.
    #[arg(long, default_value_t = 1_048_576)]
    c_lcb: u64,
    /// Carrier: max total holes (u32::MAX = unbounded).
    #[arg(long, default_value_t = u32::MAX)]
    max_holes: u32,
    /// Carrier: max stack height.
    #[arg(long, default_value_t = u32::MAX)]
    max_height: u32,
    /// Carrier: max surface roughness (sum of adjacent |Δheight|).
    #[arg(long, default_value_t = u32::MAX)]
    max_roughness: u32,
    /// Expansion policy: `all` queues every interned state; `candidate` queues on first rollout visit.
    #[arg(long, value_enum, default_value_t = ExpansionPolicyArg::All)]
    expansion_policy: ExpansionPolicyArg,
    /// Optional CSV metrics output path.
    #[arg(long)]
    csv: Option<PathBuf>,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum ExpansionPolicyArg {
    All,
    Candidate,
}

// ============================================================================
// Domain (Task 1)
// ============================================================================

// ============================================================================
// Arenas (Task 2)
// ============================================================================

// ============================================================================
// Records (Task 3)
// ============================================================================

// ============================================================================
// Graph: interning + expansion (Tasks 3-4)
// ============================================================================

// ============================================================================
// Integer scoring (Task 5)
// ============================================================================

// ============================================================================
// Selection (Task 6)
// ============================================================================

// ============================================================================
// Rollout (Task 7)
// ============================================================================

// ============================================================================
// Closure (Task 8)
// ============================================================================

// ============================================================================
// Main / epochs (Task 9)
// ============================================================================

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst))?;
    }
    println!("{args:?}");
    let _ = stop;
    Ok(())
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check -p tetris-playground --bin tetris_atlas_puct`
Expected: success (warnings about unused enum are fine at this stage).

- [ ] **Step 5: Commit**

```bash
git add crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs
git commit -m "atlas(puct): scaffold single-file binary with CLI"
```

---

### Task 1: Domain layer

**Files:**
- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs` (fill `Domain` section; start `mod tests`)

**Interfaces:**
- Produces:
  - `struct StateKey { board: TetrisBoard, bag_bits: u8, piece_idx: u8 }` with `fn new(TetrisBoard, TetrisPieceBagState, TetrisPiece) -> Self`, `fn bag(&self) -> TetrisPieceBagState`, `fn piece(&self) -> TetrisPiece`.
  - `const UNEXPANDED: u32; const EXPANDING: u32; const MAX_VALID_STATE_ID: u32;`
  - `struct Carrier { max_holes: u32, max_height: u32, max_roughness: u32 }` with `fn contains(&self, &TetrisBoard) -> bool` and `const UNBOUNDED: Carrier`.
  - `fn surface_roughness(&TetrisBoard) -> u32`
  - `fn apply_move(&TetrisBoard, TetrisPiecePlacement, &Carrier) -> Option<TetrisBoard>`
  - `fn child_bag(TetrisPieceBagState, TetrisPiece) -> TetrisPieceBagState`
  - `fn root_keys() -> [StateKey; 7]`

- [ ] **Step 1: Write the failing tests**

Add at the bottom of the file:

```rust
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // --- Task 1: domain ---

    #[test]
    fn key_identity_axes() {
        let full = TetrisPieceBagState::from(TetrisPieceBagState::FULL_MASK);
        // base uses the refill case (current piece present in a FULL bag) — canonical.
        let base = StateKey::new(TetrisBoard::EMPTY_BOARD, full, TetrisPiece::O_PIECE);
        let same = StateKey::new(TetrisBoard::EMPTY_BOARD, full, TetrisPiece::O_PIECE);
        assert_eq!(base, same);

        // board axis
        let mut other_board = TetrisBoard::EMPTY_BOARD;
        other_board.set_bit(0, 0);
        assert_ne!(base, StateKey::new(other_board, full, TetrisPiece::O_PIECE));

        // bag axis: a canonical bag missing the current piece (O drawn) — differs
        // from `full`, and keeps the invariant (O absent) so `new` accepts it.
        let mut dropped = full;
        dropped.remove(TetrisPiece::O_PIECE);
        dropped.remove(TetrisPiece::I_PIECE);
        assert_ne!(
            base,
            StateKey::new(TetrisBoard::EMPTY_BOARD, dropped, TetrisPiece::O_PIECE)
        );

        // piece axis
        assert_ne!(
            base,
            StateKey::new(TetrisBoard::EMPTY_BOARD, full, TetrisPiece::T_PIECE)
        );
    }

    #[test]
    fn root_keys_are_canonical() {
        let roots = root_keys();
        assert_eq!(roots.len(), 7);
        for (i, root) in roots.iter().enumerate() {
            assert_eq!(root.board, TetrisBoard::EMPTY_BOARD);
            assert_eq!(root.piece().index(), i as u8);
            // Current piece was removed from a full bag: 6 remain, piece not among them.
            assert_eq!(root.bag().count(), 6);
            assert!(!root.bag().contains(root.piece()));
        }
        // All roots distinct.
        for i in 0..7 {
            for j in (i + 1)..7 {
                assert_ne!(roots[i], roots[j]);
            }
        }
    }

    #[test]
    fn bag_transition_refills() {
        // Singleton bag {O}: drawing O empties it -> refill to FULL (which contains O).
        let mut bag = TetrisPieceBagState::from(TetrisPieceBagState::FULL_MASK);
        for p in TetrisPiece::all() {
            if p != TetrisPiece::O_PIECE {
                bag.remove(p);
            }
        }
        assert_eq!(bag.count(), 1);
        let next = child_bag(bag, TetrisPiece::O_PIECE);
        assert_eq!(u8::from(next), TetrisPieceBagState::FULL_MASK);
        // Mid-bag draw: no refill.
        let full = TetrisPieceBagState::from(TetrisPieceBagState::FULL_MASK);
        let after = child_bag(full, TetrisPiece::I_PIECE);
        assert_eq!(after.count(), 6);
        assert!(!after.contains(TetrisPiece::I_PIECE));
    }

    #[test]
    fn apply_move_detects_loss() {
        // Columns 0-8 stacked 19 high, column 9 EMPTY: no row can ever complete
        // (so nothing clears), and a vertical I on column 0 reaches row 22 > 20.
        let mut limbs = [(1u32 << 19) - 1; 10];
        limbs[9] = 0;
        let board = TetrisBoard::from_limbs(limbs);
        let vertical_i_col0 = TetrisPiecePlacement::all_from_piece(TetrisPiece::I_PIECE)
            .iter()
            .copied()
            .find(|p| {
                p.piece.width(p.orientation.rotation) == 1 && p.orientation.column.index() == 0
            })
            .unwrap();
        assert!(apply_move(&board, vertical_i_col0, &Carrier::UNBOUNDED).is_none());
        // Empty board: every placement survives.
        for &p in TetrisPiecePlacement::all_from_piece(TetrisPiece::T_PIECE) {
            assert!(apply_move(&TetrisBoard::EMPTY_BOARD, p, &Carrier::UNBOUNDED).is_some());
        }
    }

    #[test]
    fn apply_move_respects_carrier() {
        let tight = Carrier { max_holes: 0, max_height: 0, max_roughness: 0 };
        // Height cap 0: every placement leaves the carrier.
        for &p in TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE) {
            assert!(apply_move(&TetrisBoard::EMPTY_BOARD, p, &tight).is_none());
        }
        let loose = Carrier { max_holes: 0, max_height: 2, max_roughness: 4 };
        // Flat O placement at height 2 fits.
        let flat_o = TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE)[0];
        assert!(apply_move(&TetrisBoard::EMPTY_BOARD, flat_o, &loose).is_some());
    }

    #[test]
    fn surface_roughness_matches_reference() {
        assert_eq!(surface_roughness(&TetrisBoard::EMPTY_BOARD), 0);
        let mut board = TetrisBoard::EMPTY_BOARD;
        board.set_bit(3, 0); // one cell at column 3 -> |0-1| + |1-0| = 2
        assert_eq!(surface_roughness(&board), 2);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: compile FAILURE — `StateKey`, `apply_move`, etc. not defined.

- [ ] **Step 3: Implement the domain section**

Fill the `Domain` section (and extend the `use` block at the top):

```rust
use tetris_game::{
    StandardTetris, TetrisBoard, TetrisGameConfig, TetrisPiece, TetrisPieceBagState,
    TetrisPiecePlacement,
};

/// Edge/child sentinels. A valid arena `StateId` never equals a sentinel.
const UNEXPANDED: u32 = u32::MAX;
const EXPANDING: u32 = u32::MAX - 1;
const MAX_VALID_STATE_ID: u32 = u32::MAX - 2;

/// A complete Tetris decision state: board, remaining-bag mask, and the visible
/// current piece. 42 bytes, `Copy`, no heap.
///
/// Canonical form (matches `TetrisPieceBagState::iter_next_states`): refill
/// happens at draw time, so `bag_bits` is never empty. Invariant:
/// `bag.contains(piece)  ⟹  bag == FULL_MASK` (a fresh refill legitimately
/// contains the just-drawn current piece; otherwise the draw removed it).
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct StateKey {
    board: TetrisBoard,
    bag_bits: u8,
    piece_idx: u8,
}

impl StateKey {
    fn new(board: TetrisBoard, bag: TetrisPieceBagState, piece: TetrisPiece) -> Self {
        // Canonical invariant: the current piece is absent from the remaining
        // bag unless a refill just restored the full bag. Real construction
        // sites (root_keys, expansion via child_bag) always satisfy this;
        // synthetic test helpers bypass `new` via a struct literal.
        debug_assert!(
            !bag.contains(piece) || u8::from(bag) == TetrisPieceBagState::FULL_MASK,
            "non-canonical StateKey: current piece in a non-full remaining bag"
        );
        Self { board, bag_bits: u8::from(bag), piece_idx: piece.index() }
    }

    fn bag(&self) -> TetrisPieceBagState {
        TetrisPieceBagState::from(self.bag_bits)
    }

    fn piece(&self) -> TetrisPiece {
        TetrisPiece::from_index(self.piece_idx)
    }
}

/// Optional bound on the board universe. Restricting the *player's* placements
/// to stay inside the carrier is sound: it only weakens the player, so any safe
/// set found is a safe set of the unrestricted game.
#[derive(Clone, Copy, Debug)]
struct Carrier {
    max_holes: u32,
    max_height: u32,
    max_roughness: u32,
}

impl Carrier {
    const UNBOUNDED: Self =
        Self { max_holes: u32::MAX, max_height: u32::MAX, max_roughness: u32::MAX };

    fn contains(&self, board: &TetrisBoard) -> bool {
        board.total_holes() <= self.max_holes
            && board.height() <= self.max_height
            && surface_roughness(board) <= self.max_roughness
    }
}

/// Total surface roughness (bumpiness): sum over adjacent columns of
/// |height[i] - height[i+1]|. Same definition as tetris_atlas_inmemory, for
/// comparability across runs.
fn surface_roughness(board: &TetrisBoard) -> u32 {
    let heights = board.heights();
    let mut rough = 0u32;
    for i in 0..StandardTetris::COLS - 1 {
        rough += heights[i].abs_diff(heights[i + 1]);
    }
    rough
}

/// Apply a placement. `None` means the move is not available to the player:
/// either it tops out or the resulting board leaves the carrier.
///
/// The result is independent of the *next* piece draw, so loss is decided per
/// placement, before piece branching — stored branches always point at live states.
fn apply_move(
    board: &TetrisBoard,
    placement: TetrisPiecePlacement,
    carrier: &Carrier,
) -> Option<TetrisBoard> {
    let mut next = *board;
    let result = next.apply_piece_placement(placement);
    if bool::from(result.is_lost) || !carrier.contains(&next) {
        return None;
    }
    Some(next)
}

/// The canonical bag transition: draw `piece`, refill if the bag empties.
/// Mirrors `TetrisPieceBagState::iter_next_states` for a single piece.
fn child_bag(bag: TetrisPieceBagState, piece: TetrisPiece) -> TetrisPieceBagState {
    let mut next = bag;
    next.remove(piece);
    if next.is_empty() {
        next.fill();
    }
    next
}

/// The seven initial states: empty board, full bag minus the first draw.
/// "init ∈ K" means all seven are in the extracted safe set.
fn root_keys() -> [StateKey; 7] {
    let pieces = TetrisPiece::all();
    pieces.map(|piece| {
        let mut bag = TetrisPieceBagState::from(TetrisPieceBagState::FULL_MASK);
        bag.remove(piece);
        StateKey::new(TetrisBoard::EMPTY_BOARD, bag, piece)
    })
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: all 6 tests PASS. (If `apply_move_detects_loss` fails because the hand-built board clears rows or doesn't top out, adjust the limb pattern — the intent is a board where a vertical I exceeds 20 rows without completing any row.)

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs
git commit -m "atlas(puct): domain layer — StateKey, carrier, apply_move, roots"
```

---

### Task 2: Arenas

**Files:**
- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs` (fill `Arenas` section; extend `mod tests`)

**Interfaces:**
- Produces:
  - `enum CapacityError { StatesExhausted, PlacementsExhausted, BranchesExhausted }` (implements `Display` + `std::error::Error`).
  - `struct Arena<T>` with `fn new(capacity: u32) -> Self`, `fn reserve(&self, count: u32) -> Option<u32>` (base index; `None` = full), `fn set(&self, idx: u32, value: T)`, `fn get(&self, idx: u32) -> &T`, `fn len(&self) -> u32`.
- Consumes: nothing from earlier tasks.
- Note: the branch arena (plain pre-initialized slice, no OnceLock) is defined in Task 3 next to its record type.

- [ ] **Step 1: Write the failing tests**

Append to `mod tests`:

```rust
    // --- Task 2: arena ---

    #[test]
    fn arena_reserve_is_contiguous_and_bounded() {
        let arena: Arena<u64> = Arena::new(10);
        assert_eq!(arena.reserve(3), Some(0));
        assert_eq!(arena.reserve(4), Some(3));
        assert_eq!(arena.reserve(3), Some(7)); // exactly at capacity
        assert_eq!(arena.reserve(1), None); // exhausted
        assert_eq!(arena.len(), 10);
    }

    #[test]
    fn arena_set_get_roundtrip() {
        let arena: Arena<u64> = Arena::new(4);
        let base = arena.reserve(2).unwrap();
        arena.set(base, 42);
        arena.set(base + 1, 43);
        assert_eq!(*arena.get(base), 42);
        assert_eq!(*arena.get(base + 1), 43);
    }

    #[test]
    fn arena_concurrent_reserves_do_not_overlap() {
        use rayon::prelude::*;
        let arena: Arena<u32> = Arena::new(8_000);
        let bases: Vec<u32> = (0..1_000u32)
            .into_par_iter()
            .map(|i| {
                let base = arena.reserve(8).unwrap();
                for j in 0..8 {
                    arena.set(base + j, i);
                }
                base
            })
            .collect();
        let mut sorted = bases.clone();
        sorted.sort_unstable();
        for pair in sorted.windows(2) {
            assert!(pair[1] - pair[0] >= 8, "overlapping reservations");
        }
        // Every slot readable and tagged by exactly the reserving worker.
        for (i, &base) in bases.iter().enumerate() {
            for j in 0..8 {
                assert_eq!(*arena.get(base + j), i as u32);
            }
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: compile FAILURE — `Arena` not defined.

- [ ] **Step 3: Implement the arena section**

Extend the `use` block (`std::sync::OnceLock`, `AtomicU32`), then:

```rust
/// Structured capacity exhaustion — never silently wrap ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CapacityError {
    StatesExhausted,
    PlacementsExhausted,
    BranchesExhausted,
}

impl std::fmt::Display for CapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let which = match self {
            Self::StatesExhausted => "state arena exhausted",
            Self::PlacementsExhausted => "placement arena exhausted",
            Self::BranchesExhausted => "branch arena exhausted",
        };
        f.write_str(which)
    }
}

impl std::error::Error for CapacityError {}

/// Preallocated write-once slot arena. `reserve` bumps atomically; each slot is
/// published exactly once via `OnceLock::set` and read via `get` only after its
/// id has been made reachable (interner map / branch child / frontier).
struct Arena<T> {
    slots: Box<[OnceLock<T>]>,
    next: AtomicU32,
}

impl<T> Arena<T> {
    fn new(capacity: u32) -> Self {
        debug_assert!(capacity <= MAX_VALID_STATE_ID);
        let slots: Vec<OnceLock<T>> = (0..capacity).map(|_| OnceLock::new()).collect();
        Self { slots: slots.into_boxed_slice(), next: AtomicU32::new(0) }
    }

    /// Reserve `count` contiguous slots; returns the base index, or `None` when
    /// the arena cannot satisfy the request (capacity exhaustion is terminal
    /// for the run, so overshoot of the bump pointer is harmless).
    fn reserve(&self, count: u32) -> Option<u32> {
        let base = self.next.fetch_add(count, Ordering::Relaxed);
        if (base as u64) + (count as u64) > self.slots.len() as u64 {
            return None;
        }
        Some(base)
    }

    fn set(&self, idx: u32, value: T) {
        let published = self.slots[idx as usize].set(value).is_ok();
        debug_assert!(published, "arena slot {idx} double-set");
    }

    /// INVARIANT: callers only hold an id after its slot was published
    /// (payload-before-id, see SearchGraph::intern). The expect documents that
    /// invariant; it is the single non-test expect in this binary.
    fn get(&self, idx: u32) -> &T {
        self.slots[idx as usize].get().expect("arena slot read before publication")
    }

    /// Number of reserved slots (occupancy), capped at capacity.
    fn len(&self) -> u32 {
        self.next.load(Ordering::Acquire).min(self.slots.len() as u32)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: all Task 1 + Task 2 tests PASS.

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs
git commit -m "atlas(puct): write-once slot arena with structured capacity errors"
```

---

### Task 3: Records, graph, interning with structural enumeration

**Files:**
- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs` (fill `Records` + first half of `Graph` section; extend `mod tests`)

**Interfaces:**
- Consumes: `StateKey`, `Carrier`, `apply_move`, `Arena<T>`, `CapacityError`, sentinels.
- Produces:
  - `const STATUS_QUEUED: u8 = 0; const STATUS_EXPANDING: u8 = 1; const STATUS_FULLY_EXPANDED: u8 = 2;`
  - `struct StateRecord { key: StateKey, visits: AtomicU64, horizon_survivals: AtomicU64, placement_base: u32, placement_len: u8, expansion_status: AtomicU8, frontier_queued: AtomicU8 }`
  - `struct PlacementRecord { placement: TetrisPiecePlacement, branch_base: u32, visits: AtomicU64, horizon_survivals: AtomicU64 }`
  - `struct PieceBranchRecord { child: AtomicU32, horizon_survivals: AtomicU32, visits: AtomicU64 }` with `fn new() -> Self`, `fn saturating_inc_survivals(&self)`
  - `struct BranchArena` (plain pre-initialized slice) with `fn new(capacity: u32)`, `fn reserve(&self, count: u32) -> Option<u32>`, `fn get(&self, idx: u32) -> &PieceBranchRecord`
  - `enum ExpansionPolicy { AllDiscovered, CandidateOnly }`
  - `struct Counters` (all `AtomicU64`: `rollouts, deaths, horizons, cycles, aborted, edges_expanded, full_expansions`, plus `max_rollout_depth: AtomicU32`)
  - `struct SearchGraph { ids: DashMap<StateKey, u32, FxBuildHasher>, states: Arena<StateRecord>, placements: Arena<PlacementRecord>, branches: BranchArena, frontier: ArrayQueue<u32>, policy: ExpansionPolicy, carrier: Carrier, exhausted: AtomicBool, counters: Counters }`
  - `impl SearchGraph`: `fn new(max_states: u32, max_placements: u32, max_branches: u32, carrier: Carrier, policy: ExpansionPolicy) -> Self`, `fn intern(&self, key: StateKey) -> Result<InternResult, CapacityError>`, `fn mark_queued_and_push(&self, id: u32)`, `fn note_exhausted(&self, e: CapacityError) -> CapacityError`
  - `struct InternResult { id: u32, newly_created: bool }`

- [ ] **Step 1: Write the failing tests**

Append to `mod tests` (plus a shared helper at the top of the mod):

```rust
    // --- shared test helper ---

    fn test_graph(max_states: u32, carrier: Carrier, policy: ExpansionPolicy) -> SearchGraph {
        SearchGraph::new(max_states, max_states * 34, max_states * 34 * 7, carrier, policy)
    }

    // --- Task 3: interning ---

    #[test]
    fn intern_same_key_returns_same_id() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let key = root_keys()[0];
        let first = graph.intern(key).unwrap();
        let second = graph.intern(key).unwrap();
        assert!(first.newly_created);
        assert!(!second.newly_created);
        assert_eq!(first.id, second.id);
        assert_eq!(graph.states.len(), 1);
    }

    #[test]
    fn concurrent_intern_one_record() {
        use rayon::prelude::*;
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let key = root_keys()[3];
        let ids: Vec<u32> =
            (0..1_000).into_par_iter().map(|_| graph.intern(key).unwrap().id).collect();
        assert!(ids.iter().all(|&id| id == ids[0]));
        assert_eq!(graph.states.len(), 1);
        assert_eq!(graph.ids.len(), 1);
        assert_eq!(graph.frontier.len(), 1); // queued exactly once
    }

    #[test]
    fn structural_enumeration_filters_placements() {
        // Unbounded carrier, empty board: every placement of the current piece survives.
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let key = root_keys()[TetrisPiece::T_PIECE.index() as usize];
        let id = graph.intern(key).unwrap().id;
        let state = graph.states.get(id);
        assert_eq!(
            state.placement_len as usize,
            TetrisPiecePlacement::all_from_piece(TetrisPiece::T_PIECE).len()
        );
        // Stored placements are exactly the surviving ones, in order, each with
        // 7 reserved branch slots all UNEXPANDED.
        for slot in 0..state.placement_len as u32 {
            let pr = graph.placements.get(state.placement_base + slot);
            assert_eq!(pr.placement.piece, TetrisPiece::T_PIECE);
            for piece_slot in 0..7 {
                let br = graph.branches.get(pr.branch_base + piece_slot);
                assert_eq!(br.child.load(Ordering::Acquire), UNEXPANDED);
            }
        }
    }

    #[test]
    fn dead_state_has_zero_placements() {
        // Carrier height 0: every placement leaves the carrier -> terminal death.
        let tight = Carrier { max_holes: 0, max_height: 0, max_roughness: 0 };
        let graph = test_graph(100, tight, ExpansionPolicy::AllDiscovered);
        let id = graph.intern(root_keys()[0]).unwrap().id;
        assert_eq!(graph.states.get(id).placement_len, 0);
    }

    #[test]
    fn candidate_policy_defers_queueing() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::CandidateOnly);
        let id = graph.intern(root_keys()[0]).unwrap().id;
        assert_eq!(graph.frontier.len(), 0);
        graph.mark_queued_and_push(id);
        graph.mark_queued_and_push(id); // second push is a no-op
        assert_eq!(graph.frontier.len(), 1);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: compile FAILURE — records/graph not defined.

- [ ] **Step 3: Implement records and graph**

Extend `use`: `crossbeam_queue::ArrayQueue`, `dashmap::DashMap`, `dashmap::mapref::entry::Entry`, `rustc_hash::FxBuildHasher`, `std::sync::atomic::{AtomicU8, AtomicU64}`.

```rust
// --- Records ---

const STATUS_QUEUED: u8 = 0;
const STATUS_EXPANDING: u8 = 1;
const STATUS_FULLY_EXPANDED: u8 = 2;

/// One canonical state. Structurally complete (placements enumerated, branch
/// blocks reserved) the moment it is published — `expansion_status` tracks only
/// child interning.
struct StateRecord {
    key: StateKey,
    visits: AtomicU64,
    horizon_survivals: AtomicU64,
    /// Base slot of `placement_len` contiguous PlacementRecords (surviving
    /// placements only; dead/out-of-carrier placements are not stored).
    placement_base: u32,
    /// 0 ⟹ terminally dead state (no placement keeps the board alive in-carrier).
    placement_len: u8,
    expansion_status: AtomicU8,
    /// 0/1: guarantees the frontier sees each state at most once.
    frontier_queued: AtomicU8,
}

struct PlacementRecord {
    placement: TetrisPiecePlacement,
    /// 7 contiguous branch slots; the slot for piece `q` is `branch_base + q.index()`.
    /// Slots for pieces not in `key.bag` stay UNEXPANDED forever.
    branch_base: u32,
    visits: AtomicU64,
    horizon_survivals: AtomicU64,
}

/// All-atomic 16-byte branch record (spec §16.12): plain pre-initialized arena,
/// no OnceLock. Survivals saturate at u32::MAX — pessimistic, hence sound for LCB.
struct PieceBranchRecord {
    child: AtomicU32,
    horizon_survivals: AtomicU32,
    visits: AtomicU64,
}

impl PieceBranchRecord {
    fn new() -> Self {
        Self {
            child: AtomicU32::new(UNEXPANDED),
            horizon_survivals: AtomicU32::new(0),
            visits: AtomicU64::new(0),
        }
    }

    fn saturating_inc_survivals(&self) {
        let _ = self
            .horizon_survivals
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1));
    }
}

/// Branch arena: every slot is pre-initialized at construction (child =
/// UNEXPANDED), so publication needs no OnceLock — all fields are atomics.
struct BranchArena {
    slots: Box<[PieceBranchRecord]>,
    next: AtomicU32,
}

impl BranchArena {
    fn new(capacity: u32) -> Self {
        debug_assert!(capacity <= MAX_VALID_STATE_ID);
        let slots: Vec<PieceBranchRecord> =
            (0..capacity).map(|_| PieceBranchRecord::new()).collect();
        Self { slots: slots.into_boxed_slice(), next: AtomicU32::new(0) }
    }

    fn reserve(&self, count: u32) -> Option<u32> {
        let base = self.next.fetch_add(count, Ordering::Relaxed);
        if (base as u64) + (count as u64) > self.slots.len() as u64 {
            return None;
        }
        Some(base)
    }

    fn get(&self, idx: u32) -> &PieceBranchRecord {
        &self.slots[idx as usize]
    }
}

// --- Graph ---

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpansionPolicy {
    AllDiscovered,
    CandidateOnly,
}

#[derive(Default)]
struct Counters {
    rollouts: AtomicU64,
    deaths: AtomicU64,
    horizons: AtomicU64,
    cycles: AtomicU64,
    aborted: AtomicU64,
    edges_expanded: AtomicU64,
    full_expansions: AtomicU64,
    max_rollout_depth: AtomicU32,
}

struct InternResult {
    id: u32,
    newly_created: bool,
}

/// Number of DashMap shards (power of two). Hashing 42-byte keys is the hot
/// intern path; FxHash + generous sharding cuts contention (same configuration
/// as tetris_atlas_inmemory).
const DASHMAP_SHARDS: usize = 256;

struct SearchGraph {
    /// Interning only. Hot traversal never touches this map.
    ids: DashMap<StateKey, u32, FxBuildHasher>,
    states: Arena<StateRecord>,
    placements: Arena<PlacementRecord>,
    branches: BranchArena,
    frontier: ArrayQueue<u32>,
    policy: ExpansionPolicy,
    carrier: Carrier,
    exhausted: AtomicBool,
    counters: Counters,
}

impl SearchGraph {
    fn new(
        max_states: u32,
        max_placements: u32,
        max_branches: u32,
        carrier: Carrier,
        policy: ExpansionPolicy,
    ) -> Self {
        Self {
            ids: DashMap::with_hasher_and_shard_amount(FxBuildHasher, DASHMAP_SHARDS),
            states: Arena::new(max_states),
            placements: Arena::new(max_placements),
            branches: BranchArena::new(max_branches),
            frontier: ArrayQueue::new(max_states.max(1) as usize),
            policy,
            carrier,
            exhausted: AtomicBool::new(false),
            counters: Counters::default(),
        }
    }

    fn note_exhausted(&self, e: CapacityError) -> CapacityError {
        self.exhausted.store(true, Ordering::SeqCst);
        e
    }

    /// Canonical interning: one StateId per StateKey, all statistics shared
    /// across histories. The record is fully built and published into the arena
    /// BEFORE its id enters the map (payload-before-id — a consumer that can
    /// see the id can always `get` the record).
    fn intern(&self, key: StateKey) -> Result<InternResult, CapacityError> {
        if let Some(existing) = self.ids.get(&key) {
            return Ok(InternResult { id: *existing, newly_created: false });
        }
        let id = match self.ids.entry(key) {
            Entry::Occupied(entry) => {
                return Ok(InternResult { id: *entry.get(), newly_created: false });
            }
            Entry::Vacant(entry) => {
                let record = self.build_state_record(key).map_err(|e| self.note_exhausted(e))?;
                let id = self
                    .states
                    .reserve(1)
                    .ok_or_else(|| self.note_exhausted(CapacityError::StatesExhausted))?;
                self.states.set(id, record);
                entry.insert(id);
                id
            }
        };
        if self.policy == ExpansionPolicy::AllDiscovered {
            self.mark_queued_and_push(id);
        }
        Ok(InternResult { id, newly_created: true })
    }

    /// Structural enumeration: filter the current piece's placements through
    /// `apply_move`, storing only survivors, each with 7 UNEXPANDED branch slots.
    fn build_state_record(&self, key: StateKey) -> Result<StateRecord, CapacityError> {
        let mut surviving =
            [TetrisPiecePlacement::from_index(0); TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT];
        let mut n: u32 = 0;
        for &placement in TetrisPiecePlacement::all_from_piece(key.piece()) {
            if apply_move(&key.board, placement, &self.carrier).is_some() {
                surviving[n as usize] = placement;
                n += 1;
            }
        }
        let placement_base = if n == 0 {
            0
        } else {
            let base = self.placements.reserve(n).ok_or(CapacityError::PlacementsExhausted)?;
            let branch_base_all =
                self.branches.reserve(n * 7).ok_or(CapacityError::BranchesExhausted)?;
            for i in 0..n {
                self.placements.set(
                    base + i,
                    PlacementRecord {
                        placement: surviving[i as usize],
                        branch_base: branch_base_all + i * 7,
                        visits: AtomicU64::new(0),
                        horizon_survivals: AtomicU64::new(0),
                    },
                );
            }
            base
        };
        Ok(StateRecord {
            key,
            visits: AtomicU64::new(0),
            horizon_survivals: AtomicU64::new(0),
            placement_base,
            placement_len: n as u8,
            expansion_status: AtomicU8::new(STATUS_QUEUED),
            frontier_queued: AtomicU8::new(0),
        })
    }

    /// Queue a state for full expansion at most once, ever.
    fn mark_queued_and_push(&self, id: u32) {
        let state = self.states.get(id);
        if state
            .frontier_queued
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            // Queued-once + capacity == max_states makes push infallible; a
            // failure would indicate a broken invariant, so count it visibly.
            if self.frontier.push(id).is_err() {
                self.counters.aborted.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: all tests PASS.

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs
git commit -m "atlas(puct): records + canonical interner with structural enumeration"
```

---

### Task 4: Edge and full expansion

**Files:**
- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs` (second half of `Graph` section; extend `mod tests`)

**Interfaces:**
- Consumes: everything from Task 3; `child_bag`, `apply_move`, `Backoff`.
- Produces (all on `impl SearchGraph`):
  - `fn expand_edge(&self, branch: &PieceBranchRecord, child_board: TetrisBoard, child_bag_state: TetrisPieceBagState, piece: TetrisPiece) -> Result<u32, CapacityError>`
  - `fn ensure_fully_expanded(&self, id: u32) -> Result<(), CapacityError>`
  - `fn resolve_child(&self, state: &StateRecord, pr: &PlacementRecord, br: &PieceBranchRecord, piece: TetrisPiece) -> Result<u32, CapacityError>`

- [ ] **Step 1: Write the failing tests**

Append to `mod tests`:

```rust
    // --- Task 4: expansion ---

    #[test]
    fn ensure_fully_expanded_covers_exactly_the_bag() {
        let graph = test_graph(10_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let key = root_keys()[0];
        let id = graph.intern(key).unwrap().id;
        graph.ensure_fully_expanded(id).unwrap();
        let state = graph.states.get(id);
        assert_eq!(state.expansion_status.load(Ordering::Acquire), STATUS_FULLY_EXPANDED);
        for slot in 0..state.placement_len as u32 {
            let pr = graph.placements.get(state.placement_base + slot);
            for piece in TetrisPiece::all() {
                let child = graph.branches.get(pr.branch_base + piece.index() as u32)
                    .child
                    .load(Ordering::Acquire);
                if key.bag().contains(piece) {
                    assert!(child <= MAX_VALID_STATE_ID, "bag piece must be expanded");
                    // Child key is the canonical successor.
                    let expect_board = apply_move(&key.board, pr.placement, &graph.carrier).unwrap();
                    let expect =
                        StateKey::new(expect_board, child_bag(key.bag(), piece), piece);
                    assert_eq!(graph.states.get(child).key, expect);
                } else {
                    assert_eq!(child, UNEXPANDED, "non-bag piece slot must stay unused");
                }
            }
        }
    }

    #[test]
    fn expand_edge_is_idempotent_and_canonical() {
        let graph = test_graph(10_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let id = graph.intern(root_keys()[1]).unwrap().id;
        graph.ensure_fully_expanded(id).unwrap();
        let state = graph.states.get(id);
        let pr = graph.placements.get(state.placement_base);
        let piece = state.key.bag().iter_pieces().next().unwrap();
        let br = graph.branches.get(pr.branch_base + piece.index() as u32);
        let first = br.child.load(Ordering::Acquire);
        // Re-resolving the same edge returns the same cached child.
        let again = graph.resolve_child(state, pr, br, piece).unwrap();
        assert_eq!(first, again);
        // A second full expansion is a no-op.
        graph.ensure_fully_expanded(id).unwrap();
        assert_eq!(br.child.load(Ordering::Acquire), first);
    }

    #[test]
    fn concurrent_full_expansion_no_stuck_expanding() {
        use rayon::prelude::*;
        let graph = test_graph(50_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let ids: Vec<u32> =
            root_keys().iter().map(|&k| graph.intern(k).unwrap().id).collect();
        // Hammer all roots from many workers simultaneously.
        (0..64usize).into_par_iter().for_each(|i| {
            graph.ensure_fully_expanded(ids[i % 7]).unwrap();
        });
        for &id in &ids {
            let state = graph.states.get(id);
            assert_eq!(state.expansion_status.load(Ordering::Acquire), STATUS_FULLY_EXPANDED);
            for slot in 0..state.placement_len as u32 {
                let pr = graph.placements.get(state.placement_base + slot);
                for piece in state.key.bag().iter_pieces() {
                    let child = graph.branches.get(pr.branch_base + piece.index() as u32)
                        .child
                        .load(Ordering::Acquire);
                    assert!(child <= MAX_VALID_STATE_ID, "no EXPANDING/UNEXPANDED left behind");
                }
            }
        }
        // Same (s,a,q) reached from different roots shares one child id:
        // canonicality means the map and arena agree.
        assert_eq!(graph.ids.len() as u32, graph.states.len());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: compile FAILURE — `ensure_fully_expanded`/`resolve_child` not defined.

- [ ] **Step 3: Implement expansion**

Extend `use`: `crossbeam::utils::Backoff`. Append to `impl SearchGraph`:

```rust
    /// Claim-and-publish one `(s, a, q)` edge. Loop shape handles all races:
    /// valid child -> return it; UNEXPANDED -> try to claim; EXPANDING (or lost
    /// claim) -> back off and re-read. On capacity error the claim is rolled
    /// back to UNEXPANDED so no edge is ever stuck EXPANDING.
    fn expand_edge(
        &self,
        branch: &PieceBranchRecord,
        child_board: TetrisBoard,
        child_bag_state: TetrisPieceBagState,
        piece: TetrisPiece,
    ) -> Result<u32, CapacityError> {
        let backoff = Backoff::new();
        loop {
            let current = branch.child.load(Ordering::Acquire);
            if current <= MAX_VALID_STATE_ID {
                return Ok(current);
            }
            if current == UNEXPANDED
                && branch
                    .child
                    .compare_exchange(UNEXPANDED, EXPANDING, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                let key = StateKey::new(child_board, child_bag_state, piece);
                return match self.intern(key) {
                    Ok(result) => {
                        branch.child.store(result.id, Ordering::Release);
                        self.counters.edges_expanded.fetch_add(1, Ordering::Relaxed);
                        Ok(result.id)
                    }
                    Err(e) => {
                        branch.child.store(UNEXPANDED, Ordering::Release);
                        Err(self.note_exhausted(e))
                    }
                };
            }
            backoff.snooze();
        }
    }

    /// Recompute the successor board for a stored (surviving) placement and
    /// resolve the branch child, expanding the edge on first touch.
    fn resolve_child(
        &self,
        state: &StateRecord,
        pr: &PlacementRecord,
        br: &PieceBranchRecord,
        piece: TetrisPiece,
    ) -> Result<u32, CapacityError> {
        let cached = br.child.load(Ordering::Acquire);
        if cached <= MAX_VALID_STATE_ID {
            return Ok(cached);
        }
        // Stored placements survived structural enumeration; recompute is
        // deterministic, so this cannot fail. Skip defensively if it somehow
        // does (an unexpanded branch is treated as unsafe everywhere).
        let Some(child_board) = apply_move(&state.key.board, pr.placement, &self.carrier) else {
            debug_assert!(false, "stored placement failed re-application");
            return Err(CapacityError::StatesExhausted);
        };
        self.expand_edge(br, child_board, child_bag_state_for(state.key.bag(), piece), piece)
    }

    /// Full expansion: intern every `(a, q ∈ bag)` child. One function, two
    /// drivers (frontier batches and rollouts). Losers of the status CAS wait
    /// until the winner publishes FULLY_EXPANDED. On capacity error the status
    /// is still published (the run is ending) — closure never trusts the label
    /// alone, it re-checks every branch child, so a partially expanded state
    /// simply fails the ∀ and is pruned.
    fn ensure_fully_expanded(&self, id: u32) -> Result<(), CapacityError> {
        let state = self.states.get(id);
        match state.expansion_status.compare_exchange(
            STATUS_QUEUED,
            STATUS_EXPANDING,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                let mut result = Ok(());
                'placements: for slot in 0..state.placement_len as u32 {
                    let pr = self.placements.get(state.placement_base + slot);
                    let Some(child_board) =
                        apply_move(&state.key.board, pr.placement, &self.carrier)
                    else {
                        debug_assert!(false, "stored placement failed re-application");
                        continue;
                    };
                    for (piece, bag) in state.key.bag().iter_next_states() {
                        let br = self.branches.get(pr.branch_base + piece.index() as u32);
                        if let Err(e) = self.expand_edge(br, child_board, bag, piece) {
                            result = Err(e);
                            break 'placements;
                        }
                    }
                }
                state.expansion_status.store(STATUS_FULLY_EXPANDED, Ordering::Release);
                self.counters.full_expansions.fetch_add(1, Ordering::Relaxed);
                result
            }
            Err(STATUS_EXPANDING) => {
                let backoff = Backoff::new();
                while state.expansion_status.load(Ordering::Acquire) != STATUS_FULLY_EXPANDED {
                    if self.exhausted.load(Ordering::SeqCst) {
                        return Err(CapacityError::StatesExhausted);
                    }
                    backoff.snooze();
                }
                Ok(())
            }
            Err(_) => Ok(()), // already FULLY_EXPANDED
        }
    }
```

And the small free-function alias used above (place next to `child_bag` in the Domain section):

```rust
/// Alias with the argument order used at expansion call sites.
fn child_bag_state_for(bag: TetrisPieceBagState, piece: TetrisPiece) -> TetrisPieceBagState {
    child_bag(bag, piece)
}
```

(If this alias feels redundant during implementation, call `child_bag` directly and skip it — it exists only to keep the resolve_child line under the 100-char format width.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: all tests PASS.

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs
git commit -m "atlas(puct): edge claim/publish and full state expansion"
```

---

### Task 5: Integer scoring

**Files:**
- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs` (fill `Integer scoring` section; extend `mod tests`)

**Interfaces:**
- Produces:
  - `const SCORE_SHIFT: u32 = 20; const SCORE_ONE: u64 = 1 << SCORE_SHIFT;`
  - `fn mean_fixed(successes: u64, visits: u64) -> u64`
  - `fn exploration_bonus(parent_visits: u64, action_visits: u64, c: u64) -> u64`
  - `fn branch_lcb(branch_visits: u64, branch_survivals: u64, placement_visits: u64, c_lcb: u64) -> u64`
- Consumes: nothing (pure functions).

- [ ] **Step 1: Write the failing tests**

Append to `mod tests`:

```rust
    // --- Task 5: integer scoring ---

    #[test]
    fn mean_fixed_basics() {
        assert_eq!(mean_fixed(0, 0), 0); // no visits -> 0, no division
        assert_eq!(mean_fixed(5, 5), SCORE_ONE); // certainty == 1.0
        assert_eq!(mean_fixed(1, 2), SCORE_ONE / 2);
        assert_eq!(mean_fixed(u64::MAX, u64::MAX), SCORE_ONE); // u128 path, no overflow
    }

    #[test]
    fn isqrt_property_holds() {
        // std u128::isqrt is the exact floor root; verify the contract we rely on.
        for n in [0u128, 1, 2, 3, 4, 5, 24, 25, 26, u64::MAX as u128, u128::MAX] {
            let r = n.isqrt();
            assert!(r * r <= n);
            assert!((r + 1).checked_mul(r + 1).map_or(true, |sq| sq > n));
        }
        // Pseudo-random walk (deterministic; no RNG in this crate).
        let mut x: u128 = 0x9E3779B97F4A7C15;
        for _ in 0..1_000 {
            x = x.wrapping_mul(0x2545F4914F6CDD1D).wrapping_add(1);
            let r = x.isqrt();
            assert!(r * r <= x && (r + 1).checked_mul(r + 1).map_or(true, |sq| sq > x));
        }
    }

    #[test]
    fn exploration_bonus_shape() {
        let c = SCORE_ONE;
        // Unvisited actions get flat priority.
        assert_eq!(exploration_bonus(100, 0, c), u64::MAX / 4);
        // More action visits -> less pressure.
        assert!(exploration_bonus(1_000, 1, c) > exploration_bonus(1_000, 100, c));
        // More parent visits -> more pressure (log grows).
        assert!(exploration_bonus(1 << 20, 10, c) > exploration_bonus(1 << 4, 10, c));
        // Scales linearly in c.
        assert_eq!(
            exploration_bonus(1_000, 10, 2 * SCORE_ONE),
            2 * exploration_bonus(1_000, 10, SCORE_ONE)
        );
    }

    #[test]
    fn exploration_bonus_matches_float_reference() {
        // Floats allowed ONLY in tests. Reference: c * sqrt(log2(N+1) / n).
        for (parent, action) in [(10u64, 3u64), (1_000, 7), (1 << 30, 999), (5, 1)] {
            let got = exploration_bonus(parent, action, SCORE_ONE) as f64 / SCORE_ONE as f64;
            let want = (((parent + 1) as f64).log2().floor() / action as f64).sqrt();
            assert!(
                (got - want).abs() <= want * 0.01 + 0.002,
                "parent={parent} action={action}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn branch_lcb_is_pessimistic() {
        // Unvisited branch -> 0 (never optimistic about unknowns).
        assert_eq!(branch_lcb(0, 0, 100, SCORE_ONE), 0);
        // Uncertainty subtracts (saturating at 0).
        let mean = mean_fixed(50, 100);
        assert!(branch_lcb(100, 50, 200, SCORE_ONE) < mean);
        // Huge uncertainty saturates to 0, never wraps.
        assert_eq!(branch_lcb(1, 1, u64::MAX / 2, u64::MAX / 2), 0);
        // More evidence at the same mean -> higher LCB.
        assert!(branch_lcb(10_000, 5_000, 20_000, SCORE_ONE)
            > branch_lcb(100, 50, 20_000, SCORE_ONE));
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: compile FAILURE — scoring functions not defined.

- [ ] **Step 3: Implement scoring**

```rust
/// Fixed-point scale: 20 fractional bits. A probability of 1.0 is SCORE_ONE.
/// No f32/f64 exists anywhere in search code — only u64/u128 arithmetic.
const SCORE_SHIFT: u32 = 20;
const SCORE_ONE: u64 = 1 << SCORE_SHIFT;

/// successes/visits in fixed point. u128 intermediates: no overflow for any u64 inputs.
fn mean_fixed(successes: u64, visits: u64) -> u64 {
    if visits == 0 {
        return 0;
    }
    (((successes as u128) << SCORE_SHIFT) / (visits as u128)) as u64
}

/// c * sqrt(log2(parent+1) / action_visits), all fixed point.
/// log2 replaces ln (the base change is absorbed by c); std u128::isqrt is the
/// exact floor root, so sqrt(log << 2*SHIFT) is the root in SHIFT fixed point.
fn exploration_bonus(parent_visits: u64, action_visits: u64, c: u64) -> u64 {
    if action_visits == 0 {
        return u64::MAX / 4;
    }
    let log = parent_visits.saturating_add(1).ilog2() as u128;
    let radicand = (log << (2 * SCORE_SHIFT)) / (action_visits as u128);
    let sqrt_fixed = radicand.isqrt() as u64;
    (((c as u128) * (sqrt_fixed as u128)) >> SCORE_SHIFT) as u64
}

/// Lower-confidence bound on a piece branch: mean minus uncertainty, floored
/// at 0. Unvisited branches score 0 — unknown is never treated as safe.
/// The parent count for the uncertainty term is the owning placement's visits.
fn branch_lcb(branch_visits: u64, branch_survivals: u64, placement_visits: u64, c_lcb: u64) -> u64 {
    if branch_visits == 0 {
        return 0;
    }
    mean_fixed(branch_survivals, branch_visits)
        .saturating_sub(exploration_bonus(placement_visits, branch_visits, c_lcb))
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: all tests PASS. If `exploration_bonus_matches_float_reference` fails on tolerance, check the reference uses `log2().floor()` (integer log semantics), not `ln`.

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs
git commit -m "atlas(puct): integer-only fixed-point scoring (mean, bonus, LCB)"
```

---

### Task 6: Selection

**Files:**
- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs` (fill `Selection` section; extend `mod tests`)

**Interfaces:**
- Consumes: `SearchGraph`, records, scoring functions.
- Produces:
  - `struct SearchParams { horizon: usize, c_uct: u64, c_lcb: u64 }`
  - `fn placement_value(graph: &SearchGraph, state: &StateRecord, pr: &PlacementRecord, c_lcb: u64) -> u64` — min over bag pieces of branch LCB.
  - `fn choose_search_placement(graph: &SearchGraph, state: &StateRecord, params: &SearchParams) -> Option<u32>` — slot index; exploration included; unvisited-first.
  - `fn choose_best_placement(graph: &SearchGraph, state: &StateRecord, params: &SearchParams) -> Option<u32>` — no exploration term.
  - `fn choose_piece(graph: &SearchGraph, state: &StateRecord, pr: &PlacementRecord, c_lcb: u64) -> TetrisPiece` — adversarial: unexpanded → unvisited → min LCB.

- [ ] **Step 1: Write the failing tests**

Append to `mod tests`:

```rust
    // --- Task 6: selection ---

    fn expanded_root(graph: &SearchGraph, root_idx: usize) -> u32 {
        let id = graph.intern(root_keys()[root_idx]).unwrap().id;
        graph.ensure_fully_expanded(id).unwrap();
        id
    }

    #[test]
    fn unvisited_placement_selected_first_in_order() {
        let graph = test_graph(50_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let params = SearchParams { horizon: 8, c_uct: SCORE_ONE, c_lcb: SCORE_ONE };
        let id = expanded_root(&graph, 0);
        let state = graph.states.get(id);
        // All unvisited: slot 0 first.
        assert_eq!(choose_search_placement(&graph, state, &params), Some(0));
        // Visit slot 0 -> slot 1 becomes the priority.
        graph.placements.get(state.placement_base).visits.fetch_add(1, Ordering::Relaxed);
        assert_eq!(choose_search_placement(&graph, state, &params), Some(1));
    }

    #[test]
    fn selection_none_when_dead() {
        let tight = Carrier { max_holes: 0, max_height: 0, max_roughness: 0 };
        let graph = test_graph(100, tight, ExpansionPolicy::AllDiscovered);
        let params = SearchParams { horizon: 8, c_uct: SCORE_ONE, c_lcb: SCORE_ONE };
        let id = graph.intern(root_keys()[0]).unwrap().id;
        let state = graph.states.get(id);
        assert_eq!(choose_search_placement(&graph, state, &params), None);
        assert_eq!(choose_best_placement(&graph, state, &params), None);
    }

    #[test]
    fn best_placement_tracks_min_branch_and_ignores_exploration() {
        let graph = test_graph(50_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let params = SearchParams { horizon: 8, c_uct: SCORE_ONE, c_lcb: 0 };
        let id = expanded_root(&graph, 0);
        let state = graph.states.get(id);
        // Give every placement/branch one visit, then make slot 2 clearly best:
        // all its branches survive; every other placement has a 0-survival branch.
        for slot in 0..state.placement_len as u32 {
            let pr = graph.placements.get(state.placement_base + slot);
            pr.visits.fetch_add(1_000, Ordering::Relaxed);
            for piece in state.key.bag().iter_pieces() {
                let br = graph.branches.get(pr.branch_base + piece.index() as u32);
                br.visits.fetch_add(1_000, Ordering::Relaxed);
                if slot == 2 {
                    br.horizon_survivals.fetch_add(1_000, Ordering::Relaxed);
                }
            }
        }
        assert_eq!(choose_best_placement(&graph, state, &params), Some(2));
        // placement_value is the min over branches: zero out ONE branch of slot 2.
        let pr2 = graph.placements.get(state.placement_base + 2);
        let piece = state.key.bag().iter_pieces().next().unwrap();
        let br = graph.branches.get(pr2.branch_base + piece.index() as u32);
        // visits 2000, survivals still 1000 -> mean 0.5 on that branch.
        br.visits.fetch_add(1_000, Ordering::Relaxed);
        let value = placement_value(&graph, state, pr2, 0);
        assert_eq!(value, SCORE_ONE / 2);
    }

    #[test]
    fn adversarial_piece_prefers_unexpanded_then_unvisited_then_min_lcb() {
        let graph = test_graph(50_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let id = expanded_root(&graph, 0);
        let state = graph.states.get(id);
        let pr = graph.placements.get(state.placement_base);
        let bag_pieces: Vec<TetrisPiece> = state.key.bag().iter_pieces().collect();
        // Fully expanded: no UNEXPANDED among bag pieces -> first unvisited wins.
        assert_eq!(choose_piece(&graph, state, pr, SCORE_ONE), bag_pieces[0]);
        // Visit all but one branch; the unvisited one must be chosen.
        for &piece in &bag_pieces[..bag_pieces.len() - 1] {
            let br = graph.branches.get(pr.branch_base + piece.index() as u32);
            br.visits.fetch_add(10, Ordering::Relaxed);
            br.horizon_survivals.fetch_add(10, Ordering::Relaxed);
        }
        assert_eq!(
            choose_piece(&graph, state, pr, SCORE_ONE),
            bag_pieces[bag_pieces.len() - 1]
        );
        // All visited: min LCB wins. Make bag_pieces[1] the weakest (0 survivals).
        let last = graph
            .branches
            .get(pr.branch_base + bag_pieces[bag_pieces.len() - 1].index() as u32);
        last.visits.fetch_add(10, Ordering::Relaxed);
        last.horizon_survivals.fetch_add(10, Ordering::Relaxed);
        let weak = graph.branches.get(pr.branch_base + bag_pieces[1].index() as u32);
        weak.visits.fetch_add(1_000, Ordering::Relaxed); // mean collapses toward 0
        assert_eq!(choose_piece(&graph, state, pr, SCORE_ONE), bag_pieces[1]);
    }

    #[test]
    fn tie_break_is_deterministic_lowest_slot() {
        let graph = test_graph(50_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let params = SearchParams { horizon: 8, c_uct: 0, c_lcb: 0 };
        let id = expanded_root(&graph, 0);
        let state = graph.states.get(id);
        // Identical statistics everywhere -> equal scores -> lowest slot wins.
        for slot in 0..state.placement_len as u32 {
            let pr = graph.placements.get(state.placement_base + slot);
            pr.visits.fetch_add(5, Ordering::Relaxed);
            for piece in state.key.bag().iter_pieces() {
                let br = graph.branches.get(pr.branch_base + piece.index() as u32);
                br.visits.fetch_add(5, Ordering::Relaxed);
                br.horizon_survivals.fetch_add(5, Ordering::Relaxed);
            }
        }
        assert_eq!(choose_search_placement(&graph, state, &params), Some(0));
        assert_eq!(choose_best_placement(&graph, state, &params), Some(0));
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: compile FAILURE — selection functions not defined.

- [ ] **Step 3: Implement selection**

```rust
/// Per-run search parameters (immutable after startup).
struct SearchParams {
    horizon: usize,
    c_uct: u64,
    c_lcb: u64,
}

/// Conservative placement value: the weakest bag-piece branch (adversary picks
/// the piece). Unexpanded/unvisited branches contribute 0 via branch_lcb.
fn placement_value(
    graph: &SearchGraph,
    state: &StateRecord,
    pr: &PlacementRecord,
    c_lcb: u64,
) -> u64 {
    let placement_visits = pr.visits.load(Ordering::Relaxed);
    let mut min = u64::MAX;
    for piece in state.key.bag().iter_pieces() {
        let br = graph.branches.get(pr.branch_base + piece.index() as u32);
        let lcb = branch_lcb(
            br.visits.load(Ordering::Relaxed),
            br.horizon_survivals.load(Ordering::Relaxed) as u64,
            placement_visits,
            c_lcb,
        );
        min = min.min(lcb);
    }
    min // bag is never empty, so min was set
}

/// UCT selection (exploration included). Unvisited placements first, in slot
/// order; then argmax(value + bonus) with deterministic ties:
/// higher score > fewer visits > lower slot.
fn choose_search_placement(
    graph: &SearchGraph,
    state: &StateRecord,
    params: &SearchParams,
) -> Option<u32> {
    let n = state.placement_len as u32;
    if n == 0 {
        return None;
    }
    let parent_visits = state.visits.load(Ordering::Relaxed);
    let mut best: Option<(u64, u64, u32)> = None; // (score, visits, slot)
    for slot in 0..n {
        let pr = graph.placements.get(state.placement_base + slot);
        let visits = pr.visits.load(Ordering::Relaxed);
        if visits == 0 {
            return Some(slot);
        }
        let score = placement_value(graph, state, pr, params.c_lcb)
            .saturating_add(exploration_bonus(parent_visits, visits, params.c_uct));
        let better = match best {
            None => true,
            Some((bs, bv, _)) => score > bs || (score == bs && visits < bv),
        };
        if better {
            best = Some((score, visits, slot));
        }
    }
    best.map(|(_, _, slot)| slot)
}

/// Solver view: pure worst-case exploitation, no exploration term. This is the
/// placement a policy/witness would play.
fn choose_best_placement(
    graph: &SearchGraph,
    state: &StateRecord,
    params: &SearchParams,
) -> Option<u32> {
    let n = state.placement_len as u32;
    if n == 0 {
        return None;
    }
    let mut best: Option<(u64, u64, u32)> = None;
    for slot in 0..n {
        let pr = graph.placements.get(state.placement_base + slot);
        let visits = pr.visits.load(Ordering::Relaxed);
        let value = placement_value(graph, state, pr, params.c_lcb);
        let better = match best {
            None => true,
            Some((bs, bv, _)) => value > bs || (value == bs && visits < bv),
        };
        if better {
            best = Some((value, visits, slot));
        }
    }
    best.map(|(_, _, slot)| slot)
}

/// Adversarial next-piece selection: (1) first unexpanded bag branch, (2) first
/// unvisited, (3) minimum LCB — dangerous and uncertain branches get the visits.
/// Deterministic: bag iteration order is piece-index order.
fn choose_piece(
    graph: &SearchGraph,
    state: &StateRecord,
    pr: &PlacementRecord,
    c_lcb: u64,
) -> TetrisPiece {
    let placement_visits = pr.visits.load(Ordering::Relaxed);
    let mut fallback = state.key.piece(); // unreachable in practice: bag is never empty
    let mut best: Option<(u64, TetrisPiece)> = None;
    for (i, piece) in state.key.bag().iter_pieces().enumerate() {
        let br = graph.branches.get(pr.branch_base + piece.index() as u32);
        if br.child.load(Ordering::Acquire) > MAX_VALID_STATE_ID {
            return piece; // pass 1: unexpanded
        }
        let visits = br.visits.load(Ordering::Relaxed);
        if visits == 0 {
            return piece; // pass 2: unvisited (valid because expansion precedes visits)
        }
        let lcb = branch_lcb(
            visits,
            br.horizon_survivals.load(Ordering::Relaxed) as u64,
            placement_visits,
            c_lcb,
        );
        if i == 0 {
            fallback = piece;
        }
        if best.map_or(true, |(bl, _)| lcb < bl) {
            best = Some((lcb, piece));
        }
    }
    best.map_or(fallback, |(_, piece)| piece)
}
```

Note the one subtlety: pass 1 (unexpanded) and pass 2 (unvisited) collapse into a single loop **only because** an unexpanded branch always has 0 visits — the `child` check runs before the `visits` check per piece, and iteration order is fixed, so precedence unexpanded > unvisited > min-LCB holds piece-by-piece. If an earlier piece is unvisited and a later one unexpanded, the earlier unvisited one returns first — that is acceptable because both cases mean "no information; must sample", and the plan's ordering intent (never run UCT before local coverage exists) is preserved.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: all tests PASS.

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs
git commit -m "atlas(puct): deterministic integer UCT selection + adversarial piece choice"
```

---

### Task 7: Rollout and backup

**Files:**
- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs` (fill `Rollout` section; extend `mod tests` incl. synthetic-graph helpers)

**Interfaces:**
- Consumes: `SearchGraph`, selection functions, `SearchParams`.
- Produces:
  - `struct PathEntry { state: u32, placement_slot: u32, piece_idx: u8 }`
  - `struct WorkerScratch { path: Vec<PathEntry>, visited: FxHashSet<u32> }` with `fn new(horizon: usize) -> Self`
  - `enum RolloutOutcome { Death, Horizon, Cycle, Aborted }`
  - `fn rollout(graph: &SearchGraph, params: &SearchParams, root: u32, scratch: &mut WorkerScratch) -> RolloutOutcome`
  - `fn backup(graph: &SearchGraph, path: &[PathEntry], final_state: u32, survived: bool)`
  - `impl Counters { fn record_rollout(&self, outcome: RolloutOutcome, depth: usize) }`
- Test helpers produced for Task 8 reuse: `fn synth_state(graph: &SearchGraph, marker: u32, bag_bits: u8, n_placements: u8) -> u32`, `fn wire(graph: &SearchGraph, parent: u32, slot: u32, piece: TetrisPiece, child: u32)`

- [ ] **Step 1: Write the synthetic-graph test helpers and failing tests**

Append to `mod tests`:

```rust
    // --- synthetic graph helpers (Tasks 7-8) ---

    /// Hand-built state bypassing the interner: `marker` differentiates keys,
    /// `bag_bits` sets the AND branching, `n_placements` placements each get 7
    /// UNEXPANDED branch slots. Status starts FULLY_EXPANDED (wire children
    /// yourself; leave a branch unwired to model an unknown edge).
    fn synth_state(graph: &SearchGraph, marker: u32, bag_bits: u8, n_placements: u8) -> u32 {
        let mut limbs = [0u32; 10];
        limbs[0] = marker;
        let key = StateKey {
            board: TetrisBoard::from_limbs(limbs),
            bag_bits,
            piece_idx: 0,
        };
        let placement_base = if n_placements == 0 {
            0
        } else {
            let base = graph.placements.reserve(n_placements as u32).unwrap();
            let branch_base = graph.branches.reserve(n_placements as u32 * 7).unwrap();
            for i in 0..n_placements as u32 {
                graph.placements.set(
                    base + i,
                    PlacementRecord {
                        placement: TetrisPiecePlacement::from_index(0),
                        branch_base: branch_base + i * 7,
                        visits: AtomicU64::new(0),
                        horizon_survivals: AtomicU64::new(0),
                    },
                );
            }
            base
        };
        let id = graph.states.reserve(1).unwrap();
        graph.states.set(
            id,
            StateRecord {
                key,
                visits: AtomicU64::new(0),
                horizon_survivals: AtomicU64::new(0),
                placement_base,
                placement_len: n_placements,
                expansion_status: AtomicU8::new(STATUS_FULLY_EXPANDED),
                frontier_queued: AtomicU8::new(1),
            },
        );
        id
    }

    fn wire(graph: &SearchGraph, parent: u32, slot: u32, piece: TetrisPiece, child: u32) {
        let state = graph.states.get(parent);
        let pr = graph.placements.get(state.placement_base + slot);
        graph
            .branches
            .get(pr.branch_base + piece.index() as u32)
            .child
            .store(child, Ordering::Release);
    }

    // --- Task 7: rollout ---

    const O_BAG: u8 = 0b0000_0001; // bag containing only the O piece

    #[test]
    fn rollout_death_on_dead_state() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let params = SearchParams { horizon: 8, c_uct: SCORE_ONE, c_lcb: SCORE_ONE };
        // A -> B where B has no placements (terminal death).
        let a = synth_state(&graph, 1, O_BAG, 1);
        let b = synth_state(&graph, 2, O_BAG, 0);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, b);
        let mut scratch = WorkerScratch::new(params.horizon);
        assert!(matches!(rollout(&graph, &params, a, &mut scratch), RolloutOutcome::Death));
        // Each traversed node updated exactly once.
        assert_eq!(graph.states.get(a).visits.load(Ordering::Relaxed), 1);
        assert_eq!(graph.states.get(b).visits.load(Ordering::Relaxed), 1);
        let pr = graph.placements.get(graph.states.get(a).placement_base);
        assert_eq!(pr.visits.load(Ordering::Relaxed), 1);
        assert_eq!(pr.horizon_survivals.load(Ordering::Relaxed), 0);
        let br = graph.branches.get(pr.branch_base + TetrisPiece::O_PIECE.index() as u32);
        assert_eq!(br.visits.load(Ordering::Relaxed), 1);
        assert_eq!(graph.states.get(a).horizon_survivals.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn rollout_survives_exact_horizon() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        // Self-loop; horizon 1 stops before the cycle check fires twice.
        let a = synth_state(&graph, 1, O_BAG, 1);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, a);
        let params = SearchParams { horizon: 1, c_uct: SCORE_ONE, c_lcb: SCORE_ONE };
        let mut scratch = WorkerScratch::new(params.horizon);
        assert!(matches!(rollout(&graph, &params, a, &mut scratch), RolloutOutcome::Horizon));
        assert_eq!(graph.states.get(a).horizon_survivals.load(Ordering::Relaxed), 2);
        // ^ once as path entry, once as final state (visits likewise 2).
        assert_eq!(graph.states.get(a).visits.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn rollout_cycle_detected_and_scored_as_survival() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let a = synth_state(&graph, 1, O_BAG, 1);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, a);
        let params = SearchParams { horizon: 100, c_uct: SCORE_ONE, c_lcb: SCORE_ONE };
        let mut scratch = WorkerScratch::new(params.horizon);
        assert!(matches!(rollout(&graph, &params, a, &mut scratch), RolloutOutcome::Cycle));
        // Survival credit on the traversed placement/branch.
        let pr = graph.placements.get(graph.states.get(a).placement_base);
        assert_eq!(pr.horizon_survivals.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn rollout_scratch_does_not_grow() {
        let graph = test_graph(10_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let params = SearchParams { horizon: 16, c_uct: SCORE_ONE, c_lcb: SCORE_ONE };
        let roots: Vec<u32> =
            root_keys().iter().map(|&k| graph.intern(k).unwrap().id).collect();
        let mut scratch = WorkerScratch::new(params.horizon);
        let path_cap = scratch.path.capacity();
        let visited_cap = scratch.visited.capacity();
        for i in 0..200 {
            let _ = rollout(&graph, &params, roots[i % 7], &mut scratch);
        }
        assert_eq!(scratch.path.capacity(), path_cap, "path reallocated");
        assert_eq!(scratch.visited.capacity(), visited_cap, "visited set reallocated");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: compile FAILURE — rollout types not defined.

- [ ] **Step 3: Implement rollout**

Extend `use`: `rustc_hash::FxHashSet`.

```rust
/// One step of a recorded rollout path: enough to find the state, placement,
/// and branch records again at backup time.
struct PathEntry {
    state: u32,
    placement_slot: u32,
    piece_idx: u8,
}

/// Per-worker reusable scratch. All capacity is allocated once; rollouts only
/// clear. (`for_each_init` may build one of these per rayon job split — the
/// no-allocation guarantee is per-rollout, not per-run.)
struct WorkerScratch {
    path: Vec<PathEntry>,
    visited: FxHashSet<u32>,
}

impl WorkerScratch {
    fn new(horizon: usize) -> Self {
        Self {
            path: Vec::with_capacity(horizon + 1),
            visited: FxHashSet::with_capacity_and_hasher(horizon * 2 + 16, Default::default()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum RolloutOutcome {
    /// Reached a state with no surviving placement.
    Death,
    /// Survived the full configured horizon.
    Horizon,
    /// Revisited a StateId within this rollout — scored as survival (spec §9):
    /// a loop under adversarial-leaning piece selection is the strongest
    /// empirical signal available; counted separately in metrics.
    Cycle,
    /// Capacity exhaustion mid-walk; no backup performed.
    Aborted,
}

/// One rollout from `root`: UCT placement, adversarial piece, cached-id
/// traversal (edges expand on first touch). No allocation, no floats, no RNG.
fn rollout(
    graph: &SearchGraph,
    params: &SearchParams,
    root: u32,
    scratch: &mut WorkerScratch,
) -> RolloutOutcome {
    scratch.path.clear();
    scratch.visited.clear();
    let mut current = root;
    loop {
        if scratch.path.len() >= params.horizon {
            backup(graph, &scratch.path, current, true);
            return RolloutOutcome::Horizon;
        }
        if !scratch.visited.insert(current) {
            backup(graph, &scratch.path, current, true);
            return RolloutOutcome::Cycle;
        }
        let state = graph.states.get(current);
        if state.placement_len == 0 {
            backup(graph, &scratch.path, current, false);
            return RolloutOutcome::Death;
        }
        if graph.policy == ExpansionPolicy::CandidateOnly {
            graph.mark_queued_and_push(current); // admission by search visit
        }
        let Some(slot) = choose_search_placement(graph, state, params) else {
            backup(graph, &scratch.path, current, false);
            return RolloutOutcome::Death; // defensive: placement_len > 0 makes this unreachable
        };
        let pr = graph.placements.get(state.placement_base + slot);
        let piece = choose_piece(graph, state, pr, params.c_lcb);
        let br = graph.branches.get(pr.branch_base + piece.index() as u32);
        let child = match graph.resolve_child(state, pr, br, piece) {
            Ok(id) => id,
            Err(_) => return RolloutOutcome::Aborted,
        };
        scratch.path.push(PathEntry { state: current, placement_slot: slot, piece_idx: piece.index() });
        current = child;
    }
}

/// Single-pass backup along the recorded path plus the terminal state.
/// Relaxed ordering throughout: counters publish nothing.
fn backup(graph: &SearchGraph, path: &[PathEntry], final_state: u32, survived: bool) {
    let terminal = graph.states.get(final_state);
    terminal.visits.fetch_add(1, Ordering::Relaxed);
    if survived {
        terminal.horizon_survivals.fetch_add(1, Ordering::Relaxed);
    }
    for entry in path {
        let state = graph.states.get(entry.state);
        state.visits.fetch_add(1, Ordering::Relaxed);
        let pr = graph.placements.get(state.placement_base + entry.placement_slot);
        pr.visits.fetch_add(1, Ordering::Relaxed);
        let br = graph.branches.get(pr.branch_base + entry.piece_idx as u32);
        br.visits.fetch_add(1, Ordering::Relaxed);
        if survived {
            state.horizon_survivals.fetch_add(1, Ordering::Relaxed);
            pr.horizon_survivals.fetch_add(1, Ordering::Relaxed);
            br.saturating_inc_survivals();
        }
    }
}

impl Counters {
    fn record_rollout(&self, outcome: RolloutOutcome, depth: usize) {
        self.rollouts.fetch_add(1, Ordering::Relaxed);
        let bucket = match outcome {
            RolloutOutcome::Death => &self.deaths,
            RolloutOutcome::Horizon => &self.horizons,
            RolloutOutcome::Cycle => &self.cycles,
            RolloutOutcome::Aborted => &self.aborted,
        };
        bucket.fetch_add(1, Ordering::Relaxed);
        self.max_rollout_depth.fetch_max(depth as u32, Ordering::Relaxed);
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: all tests PASS. Check the arithmetic in `rollout_survives_exact_horizon`: with a self-loop and horizon 1, the state is visited once as a path entry and once as the terminal state, hence counts of 2.

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs
git commit -m "atlas(puct): allocation-free rollout with cycle detection and backup"
```

---

### Task 8: gfp closure

**Files:**
- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs` (fill `Closure` section; extend `mod tests`)

**Interfaces:**
- Consumes: `SearchGraph`, records, sentinels; test helpers `synth_state`/`wire`.
- Produces:
  - `struct ClosureReport { k_size: usize, iterations: u32, roots_in_k: [bool; 7], witnesses: Vec<u8> }` (`witnesses[id] == u8::MAX` ⟺ id ∉ K; otherwise the witness placement slot)
  - `fn closure_qualifies(graph: &SearchGraph, id: u32, k: &[bool]) -> Option<u8>`
  - `fn extract_safe_set(graph: &SearchGraph, roots: &[u32; 7]) -> ClosureReport`

- [ ] **Step 1: Write the failing tests**

Append to `mod tests`:

```rust
    // --- Task 8: closure ---

    fn closure_of(graph: &SearchGraph) -> ClosureReport {
        // Roots don't matter for the synthetic tests; state 0 always exists.
        extract_safe_set(graph, &[0, 0, 0, 0, 0, 0, 0])
    }

    #[test]
    fn safe_cycle_is_retained() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let a = synth_state(&graph, 1, O_BAG, 1);
        let b = synth_state(&graph, 2, O_BAG, 1);
        let c = synth_state(&graph, 3, O_BAG, 1);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, b);
        wire(&graph, b, 0, TetrisPiece::O_PIECE, c);
        wire(&graph, c, 0, TetrisPiece::O_PIECE, a);
        let report = closure_of(&graph);
        assert_eq!(report.k_size, 3);
        assert_eq!(report.witnesses[a as usize], 0);
    }

    #[test]
    fn poisoned_cycle_is_rejected() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let a = synth_state(&graph, 1, O_BAG, 1);
        let b = synth_state(&graph, 2, O_BAG, 1);
        let c = synth_state(&graph, 3, O_BAG, 1);
        let dead = synth_state(&graph, 4, O_BAG, 0); // no placements: not in K0
        wire(&graph, a, 0, TetrisPiece::O_PIECE, b);
        wire(&graph, b, 0, TetrisPiece::O_PIECE, dead); // poison
        wire(&graph, c, 0, TetrisPiece::O_PIECE, a);
        let report = closure_of(&graph);
        assert_eq!(report.k_size, 0, "death propagates back through the whole chain");
    }

    #[test]
    fn unknown_branch_blocks_certification() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        const OI_BAG: u8 = 0b0000_0011; // bag {O, I}: two required branches
        let a = synth_state(&graph, 1, OI_BAG, 1);
        let b = synth_state(&graph, 2, O_BAG, 1);
        wire(&graph, b, 0, TetrisPiece::O_PIECE, b); // b: safe self-loop
        wire(&graph, a, 0, TetrisPiece::O_PIECE, b);
        // a's I branch stays UNEXPANDED -> a fails the ∀; b survives alone.
        let report = closure_of(&graph);
        assert_eq!(report.k_size, 1);
        assert_eq!(report.witnesses[b as usize], 0);
        assert_eq!(report.witnesses[a as usize], u8::MAX);
    }

    #[test]
    fn witness_skips_failing_placement() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let s = synth_state(&graph, 1, O_BAG, 2); // two placements
        let dead = synth_state(&graph, 2, O_BAG, 0);
        wire(&graph, s, 0, TetrisPiece::O_PIECE, dead); // slot 0 fails
        wire(&graph, s, 1, TetrisPiece::O_PIECE, s); // slot 1 keeps s safe
        let report = closure_of(&graph);
        assert_eq!(report.k_size, 1);
        assert_eq!(report.witnesses[s as usize], 1, "witness is the qualifying slot");
    }

    #[test]
    fn k0_requires_fully_expanded() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let a = synth_state(&graph, 1, O_BAG, 1);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, a);
        // Force status back to QUEUED: a would be safe, but is not fully expanded.
        graph.states.get(a).expansion_status.store(STATUS_QUEUED, Ordering::Release);
        let report = closure_of(&graph);
        assert_eq!(report.k_size, 0);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: compile FAILURE — closure types not defined.

- [ ] **Step 3: Implement closure**

Extend `use`: `rayon::prelude::*` (top-level, also used by Task 9).

```rust
/// Result of one gfp pass over a quiescent graph.
struct ClosureReport {
    k_size: usize,
    iterations: u32,
    roots_in_k: [bool; 7],
    /// Per state id: the witness placement slot, or u8::MAX if the state ∉ K.
    witnesses: Vec<u8>,
}

/// Does `id` have a placement whose every bag-piece successor is in `k`?
/// Unknown (UNEXPANDED/EXPANDING) branches fail the ∀ — never treated as safe.
/// Returns the first qualifying slot (deterministic witness).
fn closure_qualifies(graph: &SearchGraph, id: u32, k: &[bool]) -> Option<u8> {
    let state = graph.states.get(id);
    'placements: for slot in 0..state.placement_len {
        let pr = graph.placements.get(state.placement_base + slot as u32);
        for piece in state.key.bag().iter_pieces() {
            let child = graph
                .branches
                .get(pr.branch_base + piece.index() as u32)
                .child
                .load(Ordering::Acquire);
            if child > MAX_VALID_STATE_ID || !k[child as usize] {
                continue 'placements;
            }
        }
        return Some(slot);
    }
    None
}

/// Greatest-fixpoint pruning:
///   K0    = { alive ∧ fully expanded }
///   K_n+1 = { s ∈ K_n | ∃a ∀q∈bag(s): child(a,q) ∈ K_n }
/// iterated (rayon-parallel scans) until no state is removed. Runs only
/// between epochs — the graph is quiescent, no snapshot machinery needed.
fn extract_safe_set(graph: &SearchGraph, roots: &[u32; 7]) -> ClosureReport {
    let n = graph.states.len() as usize;
    let mut k: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|id| {
            let state = graph.states.get(id as u32);
            state.placement_len > 0
                && state.expansion_status.load(Ordering::Acquire) == STATUS_FULLY_EXPANDED
        })
        .collect();
    let mut iterations = 0u32;
    loop {
        iterations += 1;
        let next: Vec<bool> = (0..n)
            .into_par_iter()
            .map(|id| k[id] && closure_qualifies(graph, id as u32, &k).is_some())
            .collect();
        let removed = k.iter().zip(&next).filter(|(before, after)| **before && !**after).count();
        k = next;
        if removed == 0 {
            break;
        }
    }
    let witnesses: Vec<u8> = (0..n)
        .into_par_iter()
        .map(|id| {
            if k[id] {
                closure_qualifies(graph, id as u32, &k).unwrap_or(u8::MAX)
            } else {
                u8::MAX
            }
        })
        .collect();
    let k_size = k.iter().filter(|in_k| **in_k).count();
    let roots_in_k = roots.map(|r| k.get(r as usize).copied().unwrap_or(false));
    ClosureReport { k_size, iterations, roots_in_k, witnesses }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: all tests PASS.

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs
git commit -m "atlas(puct): gfp safe-set extraction with deterministic witnesses"
```

---

### Task 9: Epoch executor, metrics, main

**Files:**
- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs` (fill `Main / epochs` section, replace the stub `main`; extend `mod tests`)

**Interfaces:**
- Consumes: everything.
- Produces:
  - `fn run_search(args: &Args, stop: &AtomicBool) -> anyhow::Result<ClosureReport>`
  - `fn expansion_batch(graph: &SearchGraph, pool: &rayon::ThreadPool, limit: u64) -> usize`
  - `fn main() -> anyhow::Result<()>` (final form)

- [ ] **Step 1: Write the failing tests**

Append to `mod tests`:

```rust
    // --- Task 9: epochs / integration ---

    fn drain_expand_universe(graph: &SearchGraph, pool: &rayon::ThreadPool) {
        loop {
            let drained = expansion_batch(graph, pool, u64::MAX);
            if drained == 0 {
                break;
            }
        }
    }

    #[test]
    fn expansion_topology_is_thread_count_invariant() {
        // Tiny bounded universe: carrier holes 0 / height 1 / roughness 2 —
        // only horizontal-I placements survive, so the FULL universe fits the
        // arena. (Determinism only holds below capacity: once the arena
        // saturates, WHICH states get in depends on scheduling. The exhausted
        // guard below pins that precondition.)
        let carrier = Carrier { max_holes: 0, max_height: 1, max_roughness: 2 };
        let collect_keys = |threads: usize| -> Vec<StateKey> {
            let graph = test_graph(50_000, carrier, ExpansionPolicy::AllDiscovered);
            for key in root_keys() {
                graph.intern(key).unwrap();
            }
            let pool =
                rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            drain_expand_universe(&graph, &pool);
            let mut keys: Vec<StateKey> =
                (0..graph.states.len()).map(|id| graph.states.get(id).key).collect();
            keys.sort_unstable();
            keys
        };
        let single = collect_keys(1);
        let multi = collect_keys(4);
        assert!(!single.is_empty());
        assert_eq!(single, multi, "fully expanded topology must not depend on scheduling");
    }

    #[test]
    fn smoke_run_bounded() {
        let args = Args {
            threads: 2,
            epochs: Some(3),
            rollouts_per_epoch: 2_000,
            expansions_per_epoch: 2_000,
            closure_every: 1,
            horizon: 32,
            max_states: 50_000,
            max_placements: None,
            max_branches: None,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            max_holes: 1,
            max_height: 4,
            max_roughness: 16,
            expansion_policy: ExpansionPolicyArg::All,
            csv: None,
        };
        let stop = AtomicBool::new(false);
        let report = run_search(&args, &stop).unwrap();
        assert_eq!(report.roots_in_k.len(), 7);
        // One witness entry per interned state; at minimum the 7 roots exist.
        assert!(report.witnesses.len() >= 7);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: compile FAILURE — `run_search`/`expansion_batch` not defined.

- [ ] **Step 3: Implement the executor**

Extend `use`: `std::io::Write as _`, `std::time::Instant`.

```rust
/// Drain up to `limit` frontier states and fully expand them in parallel.
/// Returns the number of states expanded.
fn expansion_batch(graph: &SearchGraph, pool: &rayon::ThreadPool, limit: u64) -> usize {
    let mut batch: Vec<u32> = Vec::new();
    while (batch.len() as u64) < limit {
        match graph.frontier.pop() {
            Some(id) => batch.push(id),
            None => break,
        }
    }
    if !batch.is_empty() {
        pool.install(|| {
            batch.par_iter().for_each(|&id| {
                let _ = graph.ensure_fully_expanded(id);
            });
        });
    }
    batch.len()
}

/// Epoch loop: expansion barrier -> rollout barrier -> (periodic) closure ->
/// metrics. Returns the final closure report. Deterministic modulo scheduling.
fn run_search(args: &Args, stop: &AtomicBool) -> anyhow::Result<ClosureReport> {
    let max_placements = args.max_placements.unwrap_or_else(|| {
        args.max_states.saturating_mul(TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT as u32)
    });
    let max_branches = args.max_branches.unwrap_or_else(|| max_placements.saturating_mul(7));
    let carrier = Carrier {
        max_holes: args.max_holes,
        max_height: args.max_height,
        max_roughness: args.max_roughness,
    };
    let policy = match args.expansion_policy {
        ExpansionPolicyArg::All => ExpansionPolicy::AllDiscovered,
        ExpansionPolicyArg::Candidate => ExpansionPolicy::CandidateOnly,
    };
    let params =
        SearchParams { horizon: args.horizon, c_uct: args.c_uct, c_lcb: args.c_lcb };

    let graph = SearchGraph::new(args.max_states, max_placements, max_branches, carrier, policy);
    let mut roots = [0u32; 7];
    for (i, key) in root_keys().into_iter().enumerate() {
        roots[i] = graph.intern(key).map_err(anyhow::Error::from)?.id;
        if policy == ExpansionPolicy::CandidateOnly {
            graph.mark_queued_and_push(roots[i]); // roots are always admitted
        }
    }

    let pool = rayon::ThreadPoolBuilder::new().num_threads(args.threads).build()?;
    let mut csv = match &args.csv {
        Some(path) => {
            let mut file = std::fs::File::create(path)?;
            writeln!(
                file,
                "epoch,states,frontier,rollouts,deaths,horizons,cycles,aborted,\
                 edges,k_size,roots_in_k,elapsed_ms"
            )?;
            Some(file)
        }
        None => None,
    };

    let started = Instant::now();
    let mut epoch: u64 = 0;
    let mut last_report: Option<ClosureReport> = None;
    loop {
        if stop.load(Ordering::SeqCst) || graph.exhausted.load(Ordering::SeqCst) {
            break;
        }
        if let Some(max) = args.epochs {
            if epoch >= max {
                break;
            }
        }

        let expanded = expansion_batch(&graph, &pool, args.expansions_per_epoch);

        pool.install(|| {
            (0..args.rollouts_per_epoch).into_par_iter().for_each_init(
                || WorkerScratch::new(params.horizon),
                |scratch, i| {
                    let root = roots[(i % 7) as usize];
                    let outcome = rollout(&graph, &params, root, scratch);
                    graph.counters.record_rollout(outcome, scratch.path.len());
                },
            );
        });

        let closure = if epoch % args.closure_every.max(1) == 0 {
            Some(pool.install(|| extract_safe_set(&graph, &roots)))
        } else {
            None
        };

        let elapsed_ms = started.elapsed().as_millis() as u64;
        let counters = &graph.counters;
        let (k_size, roots_in_k) = closure
            .as_ref()
            .map(|r| (r.k_size, r.roots_in_k.iter().filter(|b| **b).count()))
            .or_else(|| {
                last_report
                    .as_ref()
                    .map(|r| (r.k_size, r.roots_in_k.iter().filter(|b| **b).count()))
            })
            .unwrap_or((0, 0));
        println!(
            "epoch {epoch} | states {} | frontier {} | expanded {expanded} | \
             rollouts {} (d {} / h {} / c {} / a {}) | edges {} | K {k_size} \
             (roots {roots_in_k}/7) | {elapsed_ms}ms",
            graph.states.len(),
            graph.frontier.len(),
            counters.rollouts.load(Ordering::Relaxed),
            counters.deaths.load(Ordering::Relaxed),
            counters.horizons.load(Ordering::Relaxed),
            counters.cycles.load(Ordering::Relaxed),
            counters.aborted.load(Ordering::Relaxed),
            counters.edges_expanded.load(Ordering::Relaxed),
        );
        if let Some(file) = csv.as_mut() {
            writeln!(
                file,
                "{epoch},{},{},{},{},{},{},{},{},{k_size},{roots_in_k},{elapsed_ms}",
                graph.states.len(),
                graph.frontier.len(),
                counters.rollouts.load(Ordering::Relaxed),
                counters.deaths.load(Ordering::Relaxed),
                counters.horizons.load(Ordering::Relaxed),
                counters.cycles.load(Ordering::Relaxed),
                counters.aborted.load(Ordering::Relaxed),
                counters.edges_expanded.load(Ordering::Relaxed),
            )?;
        }
        if let Some(report) = closure {
            last_report = Some(report);
        }
        epoch += 1;
    }

    let final_report = pool.install(|| extract_safe_set(&graph, &roots));
    println!(
        "FINAL | states {} | K {} | iterations {} | roots in K: {:?} | \
         max rollout depth {} | exhausted: {}",
        graph.states.len(),
        final_report.k_size,
        final_report.iterations,
        final_report.roots_in_k,
        graph.counters.max_rollout_depth.load(Ordering::Relaxed),
        graph.exhausted.load(Ordering::SeqCst),
    );
    Ok(final_report)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst))?;
    }
    println!("tetris_atlas_puct starting: {args:?}");
    let report = run_search(&args, &stop)?;
    if report.roots_in_k.iter().all(|b| *b) && report.k_size > 0 {
        println!("SUCCESS: all 7 roots are in the candidate safe set (|K| = {})", report.k_size);
    } else {
        println!("No root-covering safe set yet (|K| = {}).", report.k_size);
    }
    Ok(())
}
```

Note: `CapacityError` must be convertible for `?`/`map_err(anyhow::Error::from)` — it implements `std::error::Error` (Task 2), so `anyhow::Error::from` works.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p tetris-playground --bin tetris_atlas_puct`
Expected: all tests PASS (the smoke test takes a few seconds in debug; if it is too slow, drop `rollouts_per_epoch` to 500).

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs
git commit -m "atlas(puct): epoch executor, metrics, CSV, final closure report"
```

---

### Task 10: Float ban, lint sweep, live run

**Files:**
- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs` (one test + any lint fixes)

**Interfaces:** none new.

- [ ] **Step 1: Add the float-ban test**

Append to `mod tests`:

```rust
    // --- Task 10: hygiene ---

    #[test]
    fn no_floats_outside_tests() {
        // The search engine is integer-only by spec. Floats may appear only in
        // this test module (reference UCT). Scan our own source up to `mod tests`.
        let source = include_str!("tetris_atlas_puct.rs");
        let (engine, _tests) =
            source.split_once("mod tests").expect("tests module marker present");
        for token in ["f32", "f64"] {
            assert!(
                !engine.contains(token),
                "float type `{token}` found in search engine code"
            );
        }
    }
```

- [ ] **Step 2: Run the full gate**

```bash
cargo fmt --all -- --check
cargo clippy -p tetris-playground --bin tetris_atlas_puct -- -D warnings
cargo test -p tetris-playground --bin tetris_atlas_puct
```

Expected: fmt clean; clippy clean (fix any lints — the only allowed `expect` outside tests is `Arena::get`; if clippy's `expect_used` fires there under `-D warnings`, add a targeted `#[allow(clippy::expect_used)]` on `Arena::get` with the invariant comment); all tests PASS.

- [ ] **Step 3: Live sanity run (bounded carrier, ~1 minute)**

```bash
cargo run --release -p tetris-playground --bin tetris_atlas_puct -- \
  --epochs 20 --max-states 500000 --max-holes 1 --max-height 4 --max-roughness 16 \
  --rollouts-per-epoch 50000 --expansions-per-epoch 20000 --closure-every 5 --horizon 128
```

Expected: epochs log with growing `states`, nonzero rollout deaths/horizons, closure `K` counts (likely 0 at this tight carrier — consistent with all prior atlas results), clean FINAL report, no panic. Record the observed states/sec and the final line in the commit message body.

- [ ] **Step 4: Commit**

```bash
cargo fmt --all
git add crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs
git commit -m "atlas(puct): float ban test + lint sweep + live smoke run"
```

---

## Plan self-review notes

- **Spec coverage:** spec §3 domain → Task 1; §4 arenas/records → Tasks 2–3; §5 interning → Task 3; §6 expansion → Task 4; §7 scoring → Task 5; §8 selection → Task 6; §9 rollout/backup → Task 7; §11 closure → Task 8; §10/§12/§13 executor/CLI/metrics → Tasks 0+9; §14 testing → distributed per task + Task 10; §15 phases → task order. §16.9 (certificate deferred) and §17 non-goals: intentionally absent.
- **Known simplification vs spec §6:** rollouts expand single edges (`resolve_child`) rather than calling `ensure_fully_expanded`; the frontier remains the sole driver of full expansion. This matches the user's original plan §17 ("if edge is unknown: claim and expand edge") and keeps rollout latency flat. Closure still requires `FULLY_EXPANDED`, so soundness is unaffected.
- **Type consistency:** ids are raw `u32` everywhere (a `StateId` newtype adds no safety inside one file and was dropped — sentinels are `u32` consts); `witnesses: Vec<u8>` uses `u8::MAX` as "not in K" (placement_len ≤ 34 < 255 so no collision); branch survivals are `AtomicU32` widened to `u64` at every `branch_lcb` call site.
