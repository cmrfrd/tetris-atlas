//! Exact Rust mirror of the **Lean proof engine** (`proofs/Proofs/*.lean`),
//! used to search for funneling policies and forceable target boards for the
//! 5-bag reset experiment (`proofs/Proofs/Experiments/FiveBagReset.lean`).
//!
//! This binary deliberately does NOT depend on `tetris-game`: the Lean
//! development is the ground truth here. Anything found by this search is
//! re-certified inside Lean (`Winning_35_of_tables` + `native_decide`), so a
//! semantic mismatch can only waste search time, never produce a false proof.
//!
//! Mirrored definitions (proofs/Proofs/):
//! - `Board = Finset (ℕ × ℕ)` — here `[u64; 10]`: bit `r` of word `c` is cell
//!   `(c, r)`, row 0 at the *bottom*. Loss iff any cell has row ≥ 20
//!   (`Board.isLost`). u64 rows keep all in-scope arithmetic exact: a
//!   non-lost board has height ≤ 19 and one 7-piece leg adds < 32 rows.
//! - `Piece.shape` (Piece.lean:40) — verbatim `(col, rowFromTop)` tables;
//!   `Piece.shapeUp` flips vertically to `(col, maxT − rowFromTop)`.
//! - `Placement.Valid` — `∀ cell ∈ shapeUp, col + cell.col < 10`;
//!   `allValidFor` = all `(rot ∈ Fin 4, col ∈ range 10)` passing `Valid`.
//! - `Placement.dropOffset` — `sup over cells of (colHeight (col+dc) − dr)`
//!   with ℕ truncated subtraction; `place` = union (always disjoint);
//!   `applyStep` = place then `clearLines`.
//! - `Board.clearLines` — remove full rows; every remaining cell shifts down
//!   by the number of full rows strictly below it.
//! - `Bag.draw` (Bag.lean:32) — erase the piece, refilling to the full bag
//!   when that empties it.
//! - `adversarialStep g p pl` — applies `{pl with piece := p}`, then draws.
//!
//! `validate` reproduces the leg-1 fan-outs measured in Lean (2026-06-11,
//! `lake env lean` playout harnesses over all 5040 bag orderings):
//! col0 → 4980, canon → 3145, clear → 1892. That agreement is the
//! equivalence evidence for this mirror.
//!
//! The search side answers: for which target boards `B*` can the player
//! *force* `board = B*` exactly at the end of a 7-piece leg (bag returns to
//! full), surviving every intermediate step, regardless of draw order? A
//! forceable singleton target gives a leg certificate with frontier size 1 —
//! the funneling the Lean-side certification needs (`checkPolicy` cost is
//! quadratic in frontier size; greedy policies plateau at ~1892).

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::io::Write as _;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

// ---------------------------------------------------------------------------
// Pieces (Piece.lean)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
enum Piece {
    O = 0,
    I = 1,
    S = 2,
    Z = 3,
    T = 4,
    L = 5,
    J = 6,
}

const PIECES: [Piece; 7] = [
    Piece::O,
    Piece::I,
    Piece::S,
    Piece::Z,
    Piece::T,
    Piece::L,
    Piece::J,
];

impl Piece {
    fn name(self) -> &'static str {
        match self {
            Piece::O => "O",
            Piece::I => "I",
            Piece::S => "S",
            Piece::Z => "Z",
            Piece::T => "T",
            Piece::L => "L",
            Piece::J => "J",
        }
    }
}

/// `Piece.shape` (Piece.lean:40), transcribed verbatim: cells as
/// `(col, rowFromTop)` offsets.
fn shape(p: Piece, rot: u8) -> [(u32, u32); 4] {
    match (p, rot & 3) {
        (Piece::O, _) => [(0, 0), (0, 1), (1, 0), (1, 1)],
        (Piece::I, 0) | (Piece::I, 2) => [(0, 0), (1, 0), (2, 0), (3, 0)],
        (Piece::I, _) => [(0, 0), (0, 1), (0, 2), (0, 3)],
        (Piece::S, 0) | (Piece::S, 2) => [(0, 1), (1, 0), (1, 1), (2, 0)],
        (Piece::S, _) => [(0, 0), (0, 1), (1, 1), (1, 2)],
        (Piece::Z, 0) | (Piece::Z, 2) => [(0, 0), (1, 0), (1, 1), (2, 1)],
        (Piece::Z, _) => [(0, 1), (0, 2), (1, 0), (1, 1)],
        (Piece::T, 0) => [(0, 0), (1, 0), (1, 1), (2, 0)],
        (Piece::T, 1) => [(0, 1), (1, 0), (1, 1), (1, 2)],
        (Piece::T, 2) => [(0, 1), (1, 0), (1, 1), (2, 1)],
        (Piece::T, _) => [(0, 0), (0, 1), (0, 2), (1, 1)],
        (Piece::L, 0) => [(0, 1), (1, 1), (2, 0), (2, 1)],
        (Piece::L, 1) => [(0, 0), (0, 1), (0, 2), (1, 2)],
        (Piece::L, 2) => [(0, 0), (0, 1), (1, 0), (2, 0)],
        (Piece::L, _) => [(0, 0), (1, 0), (1, 1), (1, 2)],
        (Piece::J, 0) => [(0, 0), (0, 1), (1, 1), (2, 1)],
        (Piece::J, 1) => [(0, 0), (0, 1), (0, 2), (1, 0)],
        (Piece::J, 2) => [(0, 0), (1, 0), (2, 0), (2, 1)],
        (Piece::J, _) => [(0, 2), (1, 0), (1, 1), (1, 2)],
    }
}

/// `Piece.shapeUp`: the shape re-expressed bottom-up,
/// `cell ↦ (col, maxT − rowFromTop)`.
fn shape_up(p: Piece, rot: u8) -> [(u32, u32); 4] {
    let s = shape(p, rot);
    let max_t = s.iter().map(|c| c.1).max().unwrap_or(0);
    s.map(|(c, r)| (c, max_t - r))
}

/// Precomputed `shapeUp` cells and `allValidFor` (rot, col) lists per piece.
struct Shapes {
    up: [[[(u32, u32); 4]; 4]; 7],
    valid: [Vec<(u8, u32)>; 7],
}

static SHAPES: LazyLock<Shapes> = LazyLock::new(|| {
    let mut up = [[[(0u32, 0u32); 4]; 4]; 7];
    let mut valid: [Vec<(u8, u32)>; 7] = Default::default();
    for (pi, &p) in PIECES.iter().enumerate() {
        for rot in 0..4u8 {
            let s = shape_up(p, rot);
            up[pi][rot as usize] = s;
            let max_dc = s.iter().map(|c| c.0).max().unwrap_or(0);
            for col in 0..COLS as u32 {
                // `Placement.Valid`: every cell's column fits, i.e. col + max_dc < 10.
                if col + max_dc < COLS as u32 {
                    valid[pi].push((rot, col));
                }
            }
        }
    }
    Shapes { up, valid }
});

// ---------------------------------------------------------------------------
// Board (Board.lean)
// ---------------------------------------------------------------------------

const COLS: usize = 10;
const ROWS: u32 = 20;

/// Column-major bitboard mirror of `Board = Finset (ℕ × ℕ)`.
type Board = [u64; COLS];
const EMPTY: Board = [0; COLS];

/// `Board.colHeight`: highest occupied row + 1; 0 for an empty column.
fn col_height(b: &Board, j: usize) -> u32 {
    64 - b[j].leading_zeros()
}

/// `Board.isLost`: some cell at row ≥ 20.
fn is_lost(b: &Board) -> bool {
    b.iter().any(|&w| w >> ROWS != 0)
}

/// `Finset.card` of the board.
fn card(b: &Board) -> u32 {
    b.iter().map(|w| w.count_ones()).sum()
}

/// `holes` (funnel_v2.lean): per column, (top row + 1) − cell count; 0 if empty.
fn holes(b: &Board) -> u32 {
    b.iter()
        .map(|&w| {
            if w == 0 {
                0
            } else {
                (64 - w.leading_zeros()) - w.count_ones()
            }
        })
        .sum()
}

/// `Board.clearLines`: keep cells of non-full rows, shifting each down by the
/// number of full rows strictly below it.
fn clear_lines(b: &mut Board) {
    let full = b.iter().fold(!0u64, |acc, &w| acc & w);
    if full == 0 {
        return;
    }
    for w in b.iter_mut() {
        let mut out = 0u64;
        let mut rest = *w & !full;
        while rest != 0 {
            let r = rest.trailing_zeros();
            let below = (full & ((1u64 << r) - 1)).count_ones();
            out |= 1u64 << (r - below);
            rest &= rest - 1;
        }
        *w = out;
    }
}

/// `Placement.applyStep` for a valid `(piece, rot, col)`: hard-drop
/// (`dropOffset` = sup of `colHeight(col+dc) − dr`, truncated), union, then
/// `clearLines`.
fn apply_step(b: &Board, p: Piece, rot: u8, col: u32) -> Board {
    let cells = &SHAPES.up[p as usize][(rot & 3) as usize];
    let off = cells
        .iter()
        .map(|&(dc, dr)| col_height(b, (col + dc) as usize).saturating_sub(dr))
        .max()
        .unwrap_or(0);
    let mut nb = *b;
    for &(dc, dr) in cells {
        nb[(col + dc) as usize] |= 1u64 << (off + dr);
    }
    clear_lines(&mut nb);
    nb
}

// ---------------------------------------------------------------------------
// Bag (Bag.lean) and the adversarial step (Adversarial.lean:669)
// ---------------------------------------------------------------------------

/// `Bag = Finset Piece` as a 7-bit mask; bit `p as u8` = piece present.
type Bag = u8;
const FULL_BAG: Bag = 0x7F;

/// `Bag.draw`: erase `p`, refilling to the full bag if that empties it.
fn draw(bag: Bag, p: Piece) -> Bag {
    let nb = bag & !(1u8 << (p as u8));
    if nb == 0 { FULL_BAG } else { nb }
}

fn bag_contains(bag: Bag, p: Piece) -> bool {
    bag & (1u8 << (p as u8)) != 0
}

// ---------------------------------------------------------------------------
// Policies (mirrors of the Lean scratch policies measured on 2026-06-11)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PolicyKind {
    /// `fun _ p => ⟨p, 0, 0⟩` — always rot 0, col 0.
    Col0,
    /// Greedy argmin of the row-major board encoding `Σ 2^(col + 10·row)`.
    Canon,
    /// Greedy argmin of the lex key (card·256 + holes, encoding).
    Clear,
}

impl PolicyKind {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "col0" => Ok(PolicyKind::Col0),
            "canon" => Ok(PolicyKind::Canon),
            "clear" => Ok(PolicyKind::Clear),
            _ => bail!("unknown policy {s:?} (expected col0|canon|clear)"),
        }
    }
}

/// The 640-bit row-major cell encoding `Σ 2^(col + 10·row)` as 10
/// little-endian u64 words — exact ℕ comparison via word-wise compare.
fn encode_bits(b: &Board) -> [u64; 10] {
    let mut out = [0u64; 10];
    for (c, &w) in b.iter().enumerate() {
        let mut rest = w;
        while rest != 0 {
            let r = rest.trailing_zeros();
            let idx = c as u32 + 10 * r;
            out[(idx / 64) as usize] |= 1u64 << (idx % 64);
            rest &= rest - 1;
        }
    }
    out
}

fn cmp_words(a: &[u64; 10], b: &[u64; 10]) -> Ordering {
    for i in (0..10).rev() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    Ordering::Equal
}

/// One policy step, mirroring the Lean argmin form
/// `((allValidFor p).image (fun pl => key(applyStep b pl) * 40 + rot*10+col)).min`,
/// decoded back to `(rot, col)` via `k % 40`.
///
/// `Clear`'s Lean key is `((card·256 + holes)·2^256 + encode)·40 + rc`; the
/// lex comparison here is exact iff `encode < 2^256`, i.e. board height ≤ 25
/// — guaranteed for loss-checked legs (height ≤ 19 + piece ≤ 4).
fn policy_step(kind: PolicyKind, b: &Board, p: Piece) -> (u8, u32) {
    match kind {
        PolicyKind::Col0 => (0, 0),
        PolicyKind::Canon | PolicyKind::Clear => {
            let mut best: Option<(u64, [u64; 10], u32)> = None;
            for &(rot, col) in &SHAPES.valid[p as usize] {
                let nb = apply_step(b, p, rot, col);
                let tier = if kind == PolicyKind::Clear {
                    u64::from(card(&nb)) * 256 + u64::from(holes(&nb))
                } else {
                    0
                };
                let bits = encode_bits(&nb);
                let rc = u32::from(rot) * 10 + col;
                let better = match &best {
                    None => true,
                    Some((bt, bb, brc)) => match tier.cmp(bt) {
                        Ordering::Less => true,
                        Ordering::Greater => false,
                        Ordering::Equal => match cmp_words(&bits, bb) {
                            Ordering::Less => true,
                            Ordering::Greater => false,
                            Ordering::Equal => rc < *brc,
                        },
                    },
                };
                if better {
                    best = Some((tier, bits, rc));
                }
            }
            match best {
                Some((_, _, rc)) => ((rc / 10) as u8, rc % 10),
                None => (0, 0),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Playouts over all 5040 bag orderings
// ---------------------------------------------------------------------------

/// Fold `adversarialStep` over one fixed draw order (policies here only read
/// the board, matching the Lean scratch policies).
fn playout(kind: PolicyKind, order: &[Piece; 7], from: &Board) -> Board {
    let mut b = *from;
    for &p in order {
        let (rot, col) = policy_step(kind, &b, p);
        b = apply_step(&b, p, rot, col);
    }
    b
}

fn for_each_perm(f: &mut impl FnMut(&[Piece; 7])) {
    fn rec(items: &mut [Piece; 7], k: usize, f: &mut impl FnMut(&[Piece; 7])) {
        if k == 7 {
            f(items);
            return;
        }
        for i in k..7 {
            items.swap(k, i);
            rec(items, k + 1, f);
            items.swap(k, i);
        }
    }
    let mut items = PIECES;
    rec(&mut items, 0, f);
}

/// Number of distinct final boards over all 5040 orderings (final bags are
/// all full again, so distinct boards = distinct states).
fn fan_out(kind: PolicyKind, from: &Board) -> usize {
    let mut set: FxHashSet<Board> = FxHashSet::default();
    for_each_perm(&mut |ord| {
        set.insert(playout(kind, ord, from));
    });
    set.len()
}

/// Histogram of final boards over all 5040 orderings, most frequent first.
fn mine_finals(kind: PolicyKind, from: &Board) -> Vec<(Board, u32)> {
    let mut hist: FxHashMap<Board, u32> = FxHashMap::default();
    for_each_perm(&mut |ord| {
        *hist.entry(playout(kind, ord, from)).or_insert(0) += 1;
    });
    let mut v: Vec<(Board, u32)> = hist.into_iter().collect();
    v.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| cmp_words(&encode_bits(&a.0), &encode_bits(&b.0)))
    });
    v
}

// ---------------------------------------------------------------------------
// AND-OR forcibility: can the player force `board = target` at leg end?
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    Forced,
    Refuted,
    Unknown,
}

/// Backward-complete AND-OR search for one leg (7 pieces, bag full → full):
/// `forces (b, bag, depth)` iff for every piece the adversary can draw there
/// is a valid, surviving placement whose successor still forces, and at
/// depth 7 the board equals `target` exactly. This is `BagWinning` with a
/// singleton target set, plus per-step survival (`¬ isLost`).
struct Forcer {
    target: Board,
    t_card: u32,
    t_colcnt: [u32; COLS],
    memo: FxHashMap<(Board, Bag), bool>,
    /// Winning placement per (state, piece), recorded on success — the
    /// strategy table the Lean `tablePolicy` certificate consumes.
    choice: FxHashMap<(Board, Bag, Piece), (u8, u32)>,
    nodes: u64,
    max_nodes: u64,
}

impl Forcer {
    fn new(target: Board, max_nodes: u64) -> Self {
        Forcer {
            target,
            t_card: card(&target),
            t_colcnt: std::array::from_fn(|j| target[j].count_ones()),
            memo: FxHashMap::default(),
            choice: FxHashMap::default(),
            nodes: 0,
            max_nodes,
        }
    }

    /// `None` = node budget exhausted (verdict unknown).
    fn forces(&mut self, b: &Board, bag: Bag, depth: u32) -> Option<bool> {
        if depth == 7 {
            return Some(*b == self.target);
        }
        self.nodes += 1;
        if self.nodes > self.max_nodes {
            return None;
        }
        // Exact arithmetic prunes. Over the remaining k pieces, with ℓ total
        // line clears: card(final) = card(b) + 4k − 10ℓ, and every clear
        // removes exactly one cell from every column, so per column
        // t_colcnt[j] = colcnt[j] + added_j − ℓ with added_j ≥ 0.
        let k = 7 - depth;
        let need = card(b) + 4 * k;
        if need < self.t_card {
            return Some(false);
        }
        let diff = need - self.t_card;
        if diff % 10 != 0 {
            return Some(false);
        }
        let l = diff / 10;
        for j in 0..COLS {
            if b[j].count_ones() > self.t_colcnt[j] + l {
                return Some(false);
            }
        }
        if let Some(&v) = self.memo.get(&(*b, bag)) {
            return Some(v);
        }
        let mut result = true;
        'pieces: for p in PIECES {
            if !bag_contains(bag, p) {
                continue;
            }
            for &(rot, col) in &SHAPES.valid[p as usize] {
                let nb = apply_step(b, p, rot, col);
                if is_lost(&nb) {
                    continue;
                }
                match self.forces(&nb, draw(bag, p), depth + 1) {
                    None => return None,
                    Some(true) => {
                        self.choice.insert((*b, bag, p), (rot, col));
                        continue 'pieces;
                    }
                    Some(false) => {}
                }
            }
            result = false;
            break;
        }
        self.memo.insert((*b, bag), result);
        Some(result)
    }

    fn run(&mut self, from: &Board) -> Verdict {
        if is_lost(from) {
            return Verdict::Refuted;
        }
        match self.forces(from, FULL_BAG, 0) {
            Some(true) => Verdict::Forced,
            Some(false) => Verdict::Refuted,
            None => Verdict::Unknown,
        }
    }
}

/// Set-valued leg target: either an explicit board set or the predicate
/// "all cells within the bottom `max_rows` rows" (cards are forced to
/// 28 − 10ℓ by the leg arithmetic; boards with a full row never occur
/// post-`clearLines`).
enum TargetSet {
    Explicit(FxHashSet<Board>),
    FlatPred { max_rows: u32 },
}

impl TargetSet {
    fn contains(&self, b: &Board) -> bool {
        match self {
            TargetSet::Explicit(s) => s.contains(b),
            TargetSet::FlatPred { max_rows } => {
                b.iter().all(|&w| w >> max_rows == 0)
                    && b.iter().fold(!0u64, |acc, &w| acc & w) == 0
            }
        }
    }

    /// Per reachable final-card class (8, 18, 28), the column-wise max cell
    /// count over members — the data the arithmetic prunes need.
    fn classes(&self) -> Vec<(u32, [u32; COLS])> {
        match self {
            TargetSet::Explicit(s) => {
                let mut by_card: FxHashMap<u32, [u32; COLS]> = FxHashMap::default();
                for b in s {
                    let entry = by_card.entry(card(b)).or_insert([0; COLS]);
                    for j in 0..COLS {
                        entry[j] = entry[j].max(b[j].count_ones());
                    }
                }
                by_card.into_iter().collect()
            }
            // Source-agnostic: a leg from card c₀ ends at c₀ + 28 − 10ℓ, so
            // every card up to the no-full-row cap 9·max_rows is a class.
            TargetSet::FlatPred { max_rows } => (0..=9 * *max_rows)
                .map(|c| (c, [*max_rows; COLS]))
                .collect(),
        }
    }
}

/// AND-OR forcibility into a target *set*, with greedy funnel move ordering:
/// placements are tried in ascending clear-key order of the successor
/// (card, holes, row-major encoding), so the first-success strategy funnels
/// branches toward common low flat boards where possible.
struct SetForcer<'a> {
    targets: &'a TargetSet,
    /// Boards excluded from the target (learned dead during closure repair).
    exclude: Option<&'a FxHashSet<Board>>,
    /// Known-obligation boards (core ∪ worklist): at the last placement of a
    /// leg, successors inside this set are tried first so recorded frontiers
    /// reuse existing obligations instead of spawning new ones.
    prefer: Option<&'a FxHashSet<Board>>,
    /// Accumulated union strategy: at any node with a recorded placement, try
    /// it first, so new trees replay established strategy and only diverge
    /// where they must — frontier novelty collapses onto known orbits.
    guide: Option<&'a StrategyTable>,
    classes: &'a [(u32, [u32; COLS])],
    memo: FxHashMap<(Board, Bag), bool>,
    choice: FxHashMap<(Board, Bag, Piece), (u8, u32)>,
    nodes: u64,
    max_nodes: u64,
}

impl<'a> SetForcer<'a> {
    fn new(targets: &'a TargetSet, classes: &'a [(u32, [u32; COLS])], max_nodes: u64) -> Self {
        SetForcer {
            targets,
            exclude: None,
            prefer: None,
            guide: None,
            classes,
            memo: FxHashMap::default(),
            choice: FxHashMap::default(),
            nodes: 0,
            max_nodes,
        }
    }

    fn with_exclude(
        targets: &'a TargetSet,
        exclude: &'a FxHashSet<Board>,
        classes: &'a [(u32, [u32; COLS])],
        max_nodes: u64,
    ) -> Self {
        SetForcer {
            targets,
            exclude: Some(exclude),
            prefer: None,
            guide: None,
            classes,
            memo: FxHashMap::default(),
            choice: FxHashMap::default(),
            nodes: 0,
            max_nodes,
        }
    }

    fn prefer(mut self, prefer: &'a FxHashSet<Board>) -> Self {
        self.prefer = Some(prefer);
        self
    }

    fn guide(mut self, guide: &'a StrategyTable) -> Self {
        self.guide = Some(guide);
        self
    }

    /// Sound necessary condition: some card class is arithmetically reachable
    /// (ℓ = (card(b) + 4k − c)/10 a nonnegative integer and every column fits
    /// under the class-wise max plus ℓ).
    fn feasible(&self, b: &Board, k: u32) -> bool {
        let need = card(b) + 4 * k;
        self.classes.iter().any(|&(c, ref maxcol)| {
            if need < c {
                return false;
            }
            let diff = need - c;
            if diff % 10 != 0 {
                return false;
            }
            let l = diff / 10;
            (0..COLS).all(|j| b[j].count_ones() <= maxcol[j] + l)
        })
    }

    fn forces(&mut self, b: &Board, bag: Bag, depth: u32) -> Option<bool> {
        if depth == 7 {
            return Some(self.targets.contains(b) && self.exclude.is_none_or(|d| !d.contains(b)));
        }
        self.nodes += 1;
        if self.nodes > self.max_nodes {
            return None;
        }
        if !self.feasible(b, 7 - depth) {
            return Some(false);
        }
        if let Some(&v) = self.memo.get(&(*b, bag)) {
            return Some(v);
        }
        let mut result = true;
        'pieces: for p in PIECES {
            if !bag_contains(bag, p) {
                continue;
            }
            // Expand successors once, then funnel-order them. The guided
            // placement (recorded union strategy) is absolute-first (tier 0);
            // at the leg's last placement, known obligations come before
            // unknown ones (tier bit 63) so recorded frontiers reuse boards
            // the closure already tracks.
            let guided = self.guide.and_then(|g| g.get(&(*b, bag, p)).copied());
            let mut succs: Vec<(u64, [u64; 10], Board, u8, u32)> = SHAPES.valid[p as usize]
                .iter()
                .filter_map(|&(rot, col)| {
                    let nb = apply_step(b, p, rot, col);
                    if is_lost(&nb) {
                        return None;
                    }
                    let mut tier = if guided == Some((rot, col)) {
                        0
                    } else {
                        (u64::from(card(&nb)) * 256 + u64::from(holes(&nb))) | 1 << 62
                    };
                    if depth == 6
                        && let Some(pref) = self.prefer
                        && !pref.contains(&nb)
                    {
                        tier |= 1 << 63;
                    }
                    Some((tier, encode_bits(&nb), nb, rot, col))
                })
                .collect();
            succs.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| cmp_words(&a.1, &b.1)));
            for (_, _, nb, rot, col) in &succs {
                match self.forces(nb, draw(bag, p), depth + 1) {
                    None => return None,
                    Some(true) => {
                        self.choice.insert((*b, bag, p), (*rot, *col));
                        continue 'pieces;
                    }
                    Some(false) => {}
                }
            }
            result = false;
            break;
        }
        self.memo.insert((*b, bag), result);
        Some(result)
    }

    fn run(&mut self, from: &Board) -> Verdict {
        if is_lost(from) {
            return Verdict::Refuted;
        }
        match self.forces(from, FULL_BAG, 0) {
            Some(true) => Verdict::Forced,
            Some(false) => Verdict::Refuted,
            None => Verdict::Unknown,
        }
    }

    /// Walk the recorded strategy: collect its table rows and its depth-7
    /// frontier (the boards the next leg must handle).
    fn extract(&self, from: &Board) -> (Vec<TableRow>, Vec<Board>) {
        let mut rows: Vec<TableRow> = Vec::new();
        let mut frontier: FxHashSet<Board> = FxHashSet::default();
        let mut seen: FxHashSet<(Board, Bag)> = FxHashSet::default();
        let mut stack: Vec<(Board, Bag, u32)> = vec![(*from, FULL_BAG, 0)];
        while let Some((b, bag, depth)) = stack.pop() {
            if depth == 7 {
                frontier.insert(b);
                continue;
            }
            if !seen.insert((b, bag)) {
                continue;
            }
            for p in PIECES {
                if !bag_contains(bag, p) {
                    continue;
                }
                if let Some(&(rot, col)) = self.choice.get(&(b, bag, p)) {
                    rows.push(((b, bag, p), (rot, col)));
                    stack.push((apply_step(&b, p, rot, col), draw(bag, p), depth + 1));
                }
            }
        }
        let mut fr: Vec<Board> = frontier.into_iter().collect();
        fr.sort_by(|a, b| cmp_words(&encode_bits(a), &encode_bits(b)));
        (rows, fr)
    }
}

type TableRow = ((Board, Bag, Piece), (u8, u32));

/// Walk the recorded winning strategy from `from`, collecting exactly the
/// (state, piece) → placement rows that play can reach — the `tablePolicy`
/// artifact for one leg.
fn extract_table(f: &Forcer, from: &Board) -> Vec<TableRow> {
    let mut rows: Vec<TableRow> = Vec::new();
    let mut seen: FxHashSet<(Board, Bag)> = FxHashSet::default();
    let mut stack: Vec<(Board, Bag, u32)> = vec![(*from, FULL_BAG, 0)];
    while let Some((b, bag, depth)) = stack.pop() {
        if depth == 7 || !seen.insert((b, bag)) {
            continue;
        }
        for p in PIECES {
            if !bag_contains(bag, p) {
                continue;
            }
            if let Some(&(rot, col)) = f.choice.get(&(b, bag, p)) {
                rows.push(((b, bag, p), (rot, col)));
                stack.push((apply_step(&b, p, rot, col), draw(bag, p), depth + 1));
            }
        }
    }
    rows
}

// ---------------------------------------------------------------------------
// Deterministic policy orbits: the policy generates its own atlas
// ---------------------------------------------------------------------------

const NFEAT: usize = 8;
const FEAT_NAMES: [&str; NFEAT] = [
    "card", "holes", "agg", "bump", "maxh", "cleared", "wells", "steps",
];

/// Feature vector of a post-placement, post-clear board. `cleared` is the
/// line count of the step that produced it. `agg = card + holes` and
/// `card = const − 10·cleared` within one argmin — redundancies are fine,
/// the sweep samples a non-orthogonal basis.
fn features(nb: &Board, cleared: u32) -> [i64; NFEAT] {
    let h: [u32; COLS] = std::array::from_fn(|j| col_height(nb, j));
    let agg: u32 = h.iter().sum();
    let maxh: u32 = h.iter().copied().max().unwrap_or(0);
    let bump: u32 = (0..COLS - 1).map(|j| h[j].abs_diff(h[j + 1])).sum();
    let wells: u32 = (0..COLS)
        .map(|j| {
            let l = if j == 0 { u32::MAX } else { h[j - 1] };
            let r = if j == COLS - 1 { u32::MAX } else { h[j + 1] };
            l.min(r).saturating_sub(h[j])
        })
        .sum();
    let steps: u32 = (0..COLS - 1)
        .map(|j| u32::from(h[j].abs_diff(h[j + 1]) == 1))
        .sum();
    [
        i64::from(card(nb)),
        i64::from(holes(nb)),
        i64::from(agg),
        i64::from(bump),
        i64::from(maxh),
        i64::from(cleared),
        i64::from(wells),
        i64::from(steps),
    ]
}

/// One deterministic step: argmin over *surviving* valid placements of
/// `w · features`, tiebroken by (row-major encoding, rot·10+col) — the
/// canonicalizing tiebreak is what funnels equal-score branches onto common
/// boards. `None` = every valid placement tops out.
fn weighted_step(w: &[i64; NFEAT], b: &Board, p: Piece) -> Option<(u8, u32, Board)> {
    let c0 = card(b);
    let mut best: Option<(i64, [u64; 10], u32, Board)> = None;
    for &(rot, col) in &SHAPES.valid[p as usize] {
        let nb = apply_step(b, p, rot, col);
        if is_lost(&nb) {
            continue;
        }
        let cleared = (c0 + 4 - card(&nb)) / 10;
        let f = features(&nb, cleared);
        let score: i64 = w.iter().zip(f.iter()).map(|(a, b)| a * b).sum();
        let bits = encode_bits(&nb);
        let rc = u32::from(rot) * 10 + col;
        let better = match &best {
            None => true,
            Some((bs, bb, brc, _)) => match score.cmp(bs) {
                Ordering::Less => true,
                Ordering::Greater => false,
                Ordering::Equal => match cmp_words(&bits, bb) {
                    Ordering::Less => true,
                    Ordering::Greater => false,
                    Ordering::Equal => rc < *brc,
                },
            },
        };
        if better {
            best = Some((score, bits, rc, nb));
        }
    }
    best.map(|(_, _, rc, nb)| ((rc / 10) as u8, rc % 10, nb))
}

enum OrbitOutcome {
    /// Finite orbit, every step surviving: boundary states (bag full again),
    /// total (board, bag) states, and the full replay table. This IS an
    /// atlas — `lean-emit` consumes boundary + rows unchanged.
    Closed {
        boundary: Vec<Board>,
        states: usize,
        rows: Vec<TableRow>,
    },
    /// Some reachable (state, piece) has no surviving placement.
    Dead(Board, Bag, Piece),
    /// Some reachable board exceeds the height bound (sprawl proxy).
    TooTall(Board),
    /// Orbit exceeded the state cap without closing.
    CapHit { states: usize, boundary: usize },
}

/// BFS the full adversarial orbit of a deterministic policy from (∅, full):
/// the adversary branches on every in-bag piece, the policy answers with
/// exactly one placement. A closed orbit is a self-certifying atlas — no
/// AND-OR search anywhere, so candidate policies are cheap to evaluate.
fn policy_orbit(w: &[i64; NFEAT], max_h: u32, cap: usize) -> OrbitOutcome {
    let mut seen: FxHashSet<(Board, Bag)> = FxHashSet::default();
    let mut boundary: FxHashSet<Board> = FxHashSet::default();
    let mut rows: Vec<TableRow> = Vec::new();
    let mut stack: Vec<(Board, Bag)> = vec![(EMPTY, FULL_BAG)];
    seen.insert((EMPTY, FULL_BAG));
    boundary.insert(EMPTY);
    while let Some((b, bag)) = stack.pop() {
        for p in PIECES {
            if !bag_contains(bag, p) {
                continue;
            }
            let Some((rot, col, nb)) = weighted_step(w, &b, p) else {
                return OrbitOutcome::Dead(b, bag, p);
            };
            if (0..COLS).any(|j| col_height(&nb, j) > max_h) {
                return OrbitOutcome::TooTall(nb);
            }
            rows.push(((b, bag, p), (rot, col)));
            let nbag = draw(bag, p);
            if seen.insert((nb, nbag)) {
                if nbag == FULL_BAG {
                    boundary.insert(nb);
                }
                stack.push((nb, nbag));
                if seen.len() > cap {
                    return OrbitOutcome::CapHit {
                        states: seen.len(),
                        boundary: boundary.len(),
                    };
                }
            }
        }
    }
    let mut bd: Vec<Board> = boundary.into_iter().collect();
    bd.sort_by(|a, b| cmp_words(&encode_bits(a), &encode_bits(b)));
    OrbitOutcome::Closed {
        states: seen.len(),
        boundary: bd,
        rows,
    }
}

/// Parse `--weights`: a named policy (`clear` = card·256 + holes with the
/// canonical tiebreak, `canon` = pure tiebreak) or NFEAT comma-separated
/// integer weights.
fn parse_weights(s: &str) -> Result<[i64; NFEAT]> {
    match s {
        "clear" => {
            let mut w = [0i64; NFEAT];
            w[0] = 256;
            w[1] = 1;
            Ok(w)
        }
        "canon" => Ok([0i64; NFEAT]),
        _ => {
            let v: Vec<i64> = s
                .split(',')
                .map(|t| {
                    t.trim()
                        .parse::<i64>()
                        .with_context(|| format!("weight {t:?}"))
                })
                .collect::<Result<_>>()?;
            if v.len() != NFEAT {
                bail!(
                    "expected {NFEAT} weights ({}), got {}",
                    FEAT_NAMES.join(","),
                    v.len()
                );
            }
            Ok(std::array::from_fn(|i| v[i]))
        }
    }
}

fn fmt_weights(w: &[i64; NFEAT]) -> String {
    w.iter()
        .zip(FEAT_NAMES.iter())
        .filter(|(v, _)| **v != 0)
        .map(|(v, n)| format!("{n}={v}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn cmd_orbit(
    weights: &str,
    max_h: u32,
    cap: usize,
    out: &Option<String>,
    emit: &Option<String>,
) -> Result<()> {
    let w = parse_weights(weights)?;
    let t0 = Instant::now();
    println!("policy [{}], max_h {max_h}, cap {cap}", fmt_weights(&w));
    match policy_orbit(&w, max_h, cap) {
        OrbitOutcome::Closed {
            boundary,
            states,
            rows,
        } => {
            let card_max = boundary.iter().map(card).max().unwrap_or(0);
            println!(
                "CLOSED: {} boundary boards (card ≤ {card_max}), {states} states, {} table rows [{:.2?}]",
                boundary.len(),
                rows.len(),
                t0.elapsed()
            );
            if let Some(path) = out {
                let text: String = boundary
                    .iter()
                    .map(|b| format!("{}\n", to_hex(b)))
                    .collect();
                std::fs::write(path, text).with_context(|| format!("writing {path}"))?;
                println!("wrote core to {path}");
            }
            if let Some(path) = emit {
                write_table(path, &rows)?;
                println!("wrote strategy table to {path}");
            }
        }
        OrbitOutcome::Dead(b, bag, p) => {
            println!(
                "DEAD: no surviving placement for {} at bag {bag:02x} on board {} [{:.2?}]",
                p.name(),
                to_hex(&b),
                t0.elapsed()
            );
            print!("{}", render(&b));
        }
        OrbitOutcome::TooTall(b) => {
            println!(
                "TOO TALL: reached height > {max_h} at {} [{:.2?}]",
                to_hex(&b),
                t0.elapsed()
            );
            print!("{}", render(&b));
        }
        OrbitOutcome::CapHit { states, boundary } => {
            println!(
                "CAP HIT: {states} states ({boundary} boundary) without closing [{:.2?}]",
                t0.elapsed()
            );
        }
    }
    Ok(())
}

/// Deterministic candidate weights for sweep index `i`: half "ladders"
/// (lexicographic-style magnitude tiers over a random feature subset), half
/// small random integers; signs biased toward the sane direction (penalize
/// holes/height/bumpiness, reward clears and S/Z-receptive steps).
fn sweep_candidate(seed: u64, i: u64) -> [i64; NFEAT] {
    let mut s = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(i)
        .wrapping_mul(0xBF58_476D_1CE4_E5B9)
        | 1;
    let mut rng = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let sane: [i64; NFEAT] = [1, 1, 1, 1, 1, -1, 1, -1];
    let mut w = [0i64; NFEAT];
    if rng() % 2 == 0 {
        // Ladder: 4 magnitude tiers on 4 distinct random features.
        let mut feats: Vec<usize> = (0..NFEAT).collect();
        for k in 0..NFEAT {
            let j = k + (rng() as usize) % (NFEAT - k);
            feats.swap(k, j);
        }
        for (tier, &f) in [4096i64, 256, 16, 1].iter().zip(feats.iter()) {
            let sign = if rng() % 8 == 0 { -sane[f] } else { sane[f] };
            w[f] = sign * tier;
        }
    } else {
        for f in 0..NFEAT {
            if rng() % 3 == 0 {
                continue;
            }
            let mag = 1i64 << (rng() % 9);
            let sign = if rng() % 8 == 0 { -sane[f] } else { sane[f] };
            w[f] = sign * mag;
        }
    }
    w
}

fn cmd_orbit_sweep(candidates: usize, seed: u64, max_h: u32, cap: usize) -> Result<()> {
    let t0 = Instant::now();
    let named: Vec<(String, [i64; NFEAT])> = vec![
        ("clear".into(), parse_weights("clear")?),
        ("canon".into(), parse_weights("canon")?),
    ];
    let all: Vec<(String, [i64; NFEAT])> = named
        .into_iter()
        .chain((0..candidates as u64).map(|i| (format!("#{i}"), sweep_candidate(seed, i))))
        .collect();
    let done = AtomicU64::new(0);
    let results: Vec<(String, [i64; NFEAT], OrbitOutcome)> = all
        .par_iter()
        .map(|(name, w)| {
            let r = policy_orbit(w, max_h, cap);
            let n = done.fetch_add(1, AtomicOrdering::Relaxed) + 1;
            if n % 64 == 0 {
                eprintln!(
                    "[sweep] {n}/{} candidates [{:.2?}]",
                    all.len(),
                    t0.elapsed()
                );
            }
            (name.clone(), *w, r)
        })
        .collect();
    let (mut closed, mut dead, mut tall, mut caphit) = (0usize, 0usize, 0usize, 0usize);
    let mut winners: Vec<(usize, usize, usize, String, [i64; NFEAT])> = Vec::new();
    let mut best_cap: Option<(usize, usize, String, [i64; NFEAT])> = None;
    for (name, w, r) in results {
        match r {
            OrbitOutcome::Closed {
                boundary,
                states,
                rows,
            } => {
                closed += 1;
                winners.push((states, boundary.len(), rows.len(), name, w));
            }
            OrbitOutcome::Dead(..) => dead += 1,
            OrbitOutcome::TooTall(_) => tall += 1,
            OrbitOutcome::CapHit { states, boundary } => {
                caphit += 1;
                if best_cap.as_ref().is_none_or(|(bs, _, _, _)| boundary < *bs) {
                    best_cap = Some((boundary, states, name, w));
                }
            }
        }
    }
    println!(
        "sweep: {} candidates → {closed} closed, {dead} dead, {tall} too-tall, {caphit} cap-hit [{:.2?}]",
        all.len(),
        t0.elapsed()
    );
    winners.sort();
    for (states, boundary, rows, name, w) in winners.iter().take(20) {
        println!(
            "  CLOSED {name}: {states} states, {boundary} boundary, {rows} rows — [{}]",
            fmt_weights(w)
        );
    }
    if winners.is_empty()
        && let Some((boundary, states, name, w)) = best_cap
    {
        println!(
            "  (least-sprawling cap-hit {name}: {states} states, {boundary} boundary — [{}])",
            fmt_weights(&w)
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Global safety-game solver: greedy strategy + backward loss patching
// ---------------------------------------------------------------------------

/// On-the-fly solver for the height-bounded safety game on states
/// `(board, bag)`: the adversary draws any in-bag piece, the player answers
/// with one placement; a state is *lost* when for some piece every placement
/// either tops out, exceeds `max_h`, or lands in a lost state. The greedy
/// weighted ordering supplies default choices (the orbit-sweep policies);
/// loss discovery propagates backward through `deps` and forces the affected
/// states to advance to their next placement — greedy where possible,
/// patched only where refuted. At quiescence the surviving strategy table is
/// replay-validated from scratch (`policy_orbit`-style BFS), so the solver's
/// bookkeeping is not trusted by the certificate path.
/// Interned game state: (board id, bag).
type SId = (u32, Bag);
/// Strategy key: (board id, bag, piece to place).
type SKey = (u32, Bag, Piece);

struct GameSolver {
    arena: Vec<Board>,
    ids: FxHashMap<Board, u32>,
    w: [i64; NFEAT],
    max_h: u32,
    lost: FxHashSet<SId>,
    /// Current tentative strategy (board id, bag, piece) → placement.
    choice: FxHashMap<SKey, (u8, u32)>,
    /// Every state ever enqueued; `ordered` prefers successors in this set,
    /// funneling the strategy back into the explored graph (closure pressure).
    states: FxHashSet<SId>,
    /// successor state → the (state, piece) entries currently relying on it.
    deps: FxHashMap<SId, FxHashSet<SKey>>,
    /// Loss wakeups, processed before any fresh expansion so refutations
    /// climb toward the root ahead of new frontier growth.
    wake: std::collections::VecDeque<SId>,
    /// Fresh states, LIFO so strategy lines reach full-bag boundaries early,
    /// giving `ordered`'s seen tiers real funnel targets.
    fresh: Vec<SId>,
    queued: FxHashSet<SId>,
    advances: u64,
}

impl GameSolver {
    fn new(w: [i64; NFEAT], max_h: u32) -> Self {
        GameSolver {
            arena: Vec::new(),
            ids: FxHashMap::default(),
            w,
            max_h,
            lost: FxHashSet::default(),
            choice: FxHashMap::default(),
            states: FxHashSet::default(),
            deps: FxHashMap::default(),
            wake: std::collections::VecDeque::new(),
            fresh: Vec::new(),
            queued: FxHashSet::default(),
            advances: 0,
        }
    }

    /// Dynamic greedy ordering of surviving, height-admissible placements of
    /// `p` on `b` with successor bag `nbag`: successor states already in
    /// `states` first, then successor boards already interned, then ascending
    /// (w·features, row-major encoding, rot·10+col).
    fn ordered(&self, b: &Board, p: Piece, nbag: Bag) -> Vec<(u8, u32, Board)> {
        let c0 = card(b);
        let mut v: Vec<(bool, bool, i64, [u64; 10], u32, Board)> = SHAPES.valid[p as usize]
            .iter()
            .filter_map(|&(rot, col)| {
                let nb = apply_step(b, p, rot, col);
                if is_lost(&nb) || (0..COLS).any(|j| col_height(&nb, j) > self.max_h) {
                    return None;
                }
                let (pair_unseen, board_unseen) = match self.ids.get(&nb) {
                    Some(&i) => (!self.states.contains(&(i, nbag)), false),
                    None => (true, true),
                };
                let cleared = (c0 + 4 - card(&nb)) / 10;
                let f = features(&nb, cleared);
                let score: i64 = self.w.iter().zip(f.iter()).map(|(a, b)| a * b).sum();
                Some((
                    pair_unseen,
                    board_unseen,
                    score,
                    encode_bits(&nb),
                    u32::from(rot) * 10 + col,
                    nb,
                ))
            })
            .collect();
        v.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| cmp_words(&a.3, &b.3))
                .then_with(|| a.4.cmp(&b.4))
        });
        v.into_iter()
            .map(|(_, _, _, _, rc, nb)| ((rc / 10) as u8, rc % 10, nb))
            .collect()
    }

    fn enqueue_fresh(&mut self, s: SId) {
        if self.queued.insert(s) {
            self.fresh.push(s);
        }
    }

    fn enqueue_wake(&mut self, s: SId) {
        if self.queued.insert(s) {
            self.wake.push_back(s);
        }
    }

    fn pop(&mut self) -> Option<SId> {
        let s = self.wake.pop_front().or_else(|| self.fresh.pop())?;
        self.queued.remove(&s);
        Some(s)
    }

    /// Re-establish a valid choice for every in-bag piece of `s`, advancing
    /// past lost successors; mark `s` lost and wake its dependents if some
    /// piece runs out of placements.
    fn process(&mut self, s: (u32, Bag)) {
        if self.lost.contains(&s) {
            return;
        }
        let (bid, bag) = s;
        let b = self.arena[bid as usize];
        for p in PIECES {
            if !bag_contains(bag, p) {
                continue;
            }
            let key = (bid, bag, p);
            let nbag = draw(bag, p);
            // Keep the current choice if its successor is still alive
            // (boards never interned were never marked lost).
            if let Some(&(rot, col)) = self.choice.get(&key) {
                let nb = apply_step(&b, p, rot, col);
                let alive = match self.ids.get(&nb) {
                    Some(&i) => !self.lost.contains(&(i, nbag)),
                    None => true,
                };
                if alive {
                    continue;
                }
            }
            // Rescan from scratch under the current (dynamic) ordering,
            // skipping lost successors. Sound and terminating: a rescan only
            // happens when the current choice's successor became lost, and
            // lost-ness is monotone, so each rescan permanently burns at
            // least one placement for this key.
            let ordered = self.ordered(&b, p, nbag);
            let mut found = None;
            for (rot, col, nb) in &ordered {
                let dead = self
                    .ids
                    .get(nb)
                    .is_some_and(|&i2| self.lost.contains(&(i2, nbag)));
                if dead {
                    continue;
                }
                found = Some((*rot, *col, *nb));
                break;
            }
            self.advances += 1;
            match found {
                Some((rot, col, nb)) => {
                    let sid = (intern(&mut self.arena, &mut self.ids, nb), nbag);
                    self.choice.insert(key, (rot, col));
                    self.deps.entry(sid).or_default().insert(key);
                    if self.states.insert(sid) {
                        self.enqueue_fresh(sid);
                    }
                }
                None => {
                    self.lost.insert(s);
                    if let Some(dependents) = self.deps.get(&s).cloned() {
                        for (dbid, dbag, _) in dependents {
                            let ds = (dbid, dbag);
                            if !self.lost.contains(&ds) {
                                self.enqueue_wake(ds);
                            }
                        }
                    }
                    return;
                }
            }
        }
    }

    /// Run to quiescence from (∅, full). `Some(..)` = the root survives and
    /// the table replays to a closed orbit; `None` = root lost at this
    /// height bound (or the state cap was exceeded).
    fn solve(&mut self, cap: usize) -> Result<Option<(Vec<Board>, Vec<TableRow>)>> {
        let t0 = Instant::now();
        let root = (intern(&mut self.arena, &mut self.ids, EMPTY), FULL_BAG);
        self.states.insert(root);
        self.enqueue_fresh(root);
        let mut processed: u64 = 0;
        while let Some(s) = self.pop() {
            self.process(s);
            processed += 1;
            if processed % 1_000_000 == 0 {
                println!(
                    "[solve] processed {processed}: states {}, lost {}, wake {}, fresh {}, advances {} [{:.2?}]",
                    self.arena.len(),
                    self.lost.len(),
                    self.wake.len(),
                    self.fresh.len(),
                    self.advances,
                    t0.elapsed()
                );
            }
            if self.lost.contains(&root) {
                println!(
                    "[solve] root lost: states {}, lost {} [{:.2?}]",
                    self.arena.len(),
                    self.lost.len(),
                    t0.elapsed()
                );
                return Ok(None);
            }
            if self.arena.len() > cap {
                println!(
                    "[solve] state cap {cap} exceeded: lost {}, wake {}, fresh {} [{:.2?}]",
                    self.lost.len(),
                    self.wake.len(),
                    self.fresh.len(),
                    t0.elapsed()
                );
                return Ok(None);
            }
        }
        println!(
            "[solve] quiescent: processed {processed}, states {}, lost {}, advances {} [{:.2?}]",
            self.arena.len(),
            self.lost.len(),
            self.advances,
            t0.elapsed()
        );
        // Untrusted-bookkeeping validation: replay the table from scratch.
        let mut seen: FxHashSet<(Board, Bag)> = FxHashSet::default();
        let mut boundary: FxHashSet<Board> = FxHashSet::default();
        let mut rows: Vec<TableRow> = Vec::new();
        let mut stack: Vec<(Board, Bag)> = vec![(EMPTY, FULL_BAG)];
        seen.insert((EMPTY, FULL_BAG));
        boundary.insert(EMPTY);
        while let Some((b, bag)) = stack.pop() {
            let bid = *self.ids.get(&b).context("replay reached unknown board")?;
            for p in PIECES {
                if !bag_contains(bag, p) {
                    continue;
                }
                let &(rot, col) = self
                    .choice
                    .get(&(bid, bag, p))
                    .context("replay reached a state with no recorded choice")?;
                let nb = apply_step(&b, p, rot, col);
                if is_lost(&nb) || (0..COLS).any(|j| col_height(&nb, j) > self.max_h) {
                    bail!("replay violated the height bound — solver bug");
                }
                rows.push(((b, bag, p), (rot, col)));
                let nbag = draw(bag, p);
                if seen.insert((nb, nbag)) {
                    if nbag == FULL_BAG {
                        boundary.insert(nb);
                    }
                    stack.push((nb, nbag));
                }
            }
        }
        let mut bd: Vec<Board> = boundary.into_iter().collect();
        bd.sort_by(|a, b| cmp_words(&encode_bits(a), &encode_bits(b)));
        println!(
            "[solve] replay: {} orbit states, {} boundary, {} rows",
            seen.len(),
            bd.len(),
            rows.len()
        );
        Ok(Some((bd, rows)))
    }
}

fn cmd_solve(
    weights: &str,
    max_h: u32,
    cap: usize,
    out: &Option<String>,
    emit: &Option<String>,
) -> Result<()> {
    let w = parse_weights(weights)?;
    println!(
        "solve: policy [{}], max_h {max_h}, cap {cap}",
        fmt_weights(&w)
    );
    let mut solver = GameSolver::new(w, max_h);
    match solver.solve(cap)? {
        Some((boundary, rows)) => {
            let card_max = boundary.iter().map(card).max().unwrap_or(0);
            println!(
                "SOLVED: {} boundary boards (card ≤ {card_max}), {} table rows",
                boundary.len(),
                rows.len()
            );
            if let Some(path) = out {
                let text: String = boundary
                    .iter()
                    .map(|b| format!("{}\n", to_hex(b)))
                    .collect();
                std::fs::write(path, text).with_context(|| format!("writing {path}"))?;
                println!("wrote core to {path}");
            }
            if let Some(path) = emit {
                write_table(path, &rows)?;
                println!("wrote strategy table to {path}");
            }
        }
        None => println!("UNSOLVED at max_h {max_h} (root lost or cap exceeded)"),
    }
    Ok(())
}

enum CoreResult {
    /// Closed self-recurrent core + the union strategy table witnessing it.
    Closed(Vec<Board>, Vec<TableRow>),
    /// The seed itself was learned dead — no closure exists from it within
    /// this family (relative to the searches that fit the node budget).
    Refuted(Board),
}

fn intern(arena: &mut Vec<Board>, ids: &mut FxHashMap<Board, u32>, b: Board) -> u32 {
    if let Some(&i) = ids.get(&b) {
        return i;
    }
    let i = u32::try_from(arena.len()).unwrap_or(u32::MAX);
    arena.push(b);
    ids.insert(b, i);
    i
}

/// Resumable closure state: interned boards, frontier graph, worklist, dead
/// set, and the union strategy guide (the guide lives in `<path>.table`).
struct ClosureState {
    arena: Vec<Board>,
    ids: FxHashMap<Board, u32>,
    dead: FxHashSet<Board>,
    /// member id → frontier ids of its recorded forcing tree
    core: FxHashMap<u32, Vec<u32>>,
    /// frontier id → member ids whose recorded tree lands on it
    used_by: FxHashMap<u32, FxHashSet<u32>>,
    work: BinaryHeap<Reverse<(u32, u32, u32)>>,
    queued: FxHashSet<u32>,
    guide: StrategyTable,
}

impl ClosureState {
    fn push_work(&mut self, id: u32) {
        if self.queued.insert(id) {
            let b = self.arena[id as usize];
            self.work.push(Reverse((card(&b), holes(&b), id)));
        }
    }

    fn save(&self, path: &str) -> Result<()> {
        let mut s = String::new();
        s.push_str("#arena\n");
        for b in &self.arena {
            s.push_str(&to_hex(b));
            s.push('\n');
        }
        s.push_str("#dead\n");
        for b in &self.dead {
            s.push_str(&to_hex(b));
            s.push('\n');
        }
        s.push_str("#core\n");
        for (m, fr) in &self.core {
            s.push_str(&format!("{m}:"));
            for f in fr {
                s.push_str(&format!(" {f}"));
            }
            s.push('\n');
        }
        s.push_str("#work\n");
        for Reverse((_, _, id)) in &self.work {
            s.push_str(&format!("{id}\n"));
        }
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, s).with_context(|| format!("writing {tmp}"))?;
        std::fs::rename(&tmp, path).with_context(|| format!("renaming {tmp}"))?;
        let rows: Vec<TableRow> = self.guide.iter().map(|(k, v)| (*k, *v)).collect();
        write_table(&format!("{path}.table"), &rows)?;
        Ok(())
    }

    fn load(path: &str) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        let mut st = ClosureState {
            arena: Vec::new(),
            ids: FxHashMap::default(),
            dead: FxHashSet::default(),
            core: FxHashMap::default(),
            used_by: FxHashMap::default(),
            work: BinaryHeap::new(),
            queued: FxHashSet::default(),
            guide: StrategyTable::default(),
        };
        let mut section = "";
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(h) = line.strip_prefix('#') {
                section = match h {
                    "arena" => "arena",
                    "dead" => "dead",
                    "core" => "core",
                    "work" => "work",
                    _ => bail!("unknown checkpoint section {h}"),
                };
                continue;
            }
            match section {
                "arena" => {
                    let b = from_hex(line)?;
                    intern(&mut st.arena, &mut st.ids, b);
                }
                "dead" => {
                    st.dead.insert(from_hex(line)?);
                }
                "core" => {
                    let (m, fr) = line
                        .split_once(':')
                        .ok_or_else(|| anyhow::anyhow!("bad core line {line}"))?;
                    let m: u32 = m.trim().parse()?;
                    let fr: Vec<u32> = fr
                        .split_whitespace()
                        .map(str::parse)
                        .collect::<std::result::Result<_, _>>()?;
                    for f in &fr {
                        st.used_by.entry(*f).or_default().insert(m);
                    }
                    st.core.insert(m, fr);
                }
                "work" => {
                    let id: u32 = line.parse()?;
                    st.push_work(id);
                }
                _ => bail!("checkpoint line before any section: {line}"),
            }
        }
        let tpath = format!("{path}.table");
        if std::path::Path::new(&tpath).exists() {
            st.guide = parse_table(&tpath)?;
        }
        Ok(st)
    }
}

/// Repairing closure (lazy GFP): close `seed` under forcing-tree frontiers.
/// Each worklist board is first forced against the current obligation set
/// (success ⇒ zero growth), falling back to `family ∖ dead` (new frontier
/// boards join the worklist). Both searches replay the accumulated union
/// strategy first (`guide`), so trees diverge from established play only
/// where they must. A board that cannot force one bag into `family ∖ dead`
/// (or blows the node budget) is learned `dead`, and every member whose tree
/// referenced it is re-forced with it excluded. On success every member's
/// recorded frontier lies inside the core, so `BagWinning core core` holds;
/// the witness table is then re-extracted per member against the *final*
/// core (all targets equal ⇒ the union table is replay-consistent).
fn extract_core(
    family: &TargetSet,
    seed: Board,
    max_nodes: u64,
    ckpt: Option<&str>,
    resume: bool,
) -> Result<CoreResult> {
    const BATCH: usize = 32;
    const CKPT_SECS: u64 = 600;
    let full_classes = family.classes();
    let mut st = if resume {
        let Some(path) = ckpt else {
            bail!("--resume needs --ckpt");
        };
        let st = ClosureState::load(path)?;
        eprintln!(
            "[core] resumed: arena {}, core {}, dead {}, worklist {}, guide {}",
            st.arena.len(),
            st.core.len(),
            st.dead.len(),
            st.work.len(),
            st.guide.len()
        );
        st
    } else {
        let mut st = ClosureState {
            arena: Vec::new(),
            ids: FxHashMap::default(),
            dead: FxHashSet::default(),
            core: FxHashMap::default(),
            used_by: FxHashMap::default(),
            work: BinaryHeap::new(),
            queued: FxHashSet::default(),
            guide: StrategyTable::default(),
        };
        let sid = intern(&mut st.arena, &mut st.ids, seed);
        st.push_work(sid);
        st
    };
    let mut processed: u64 = st.core.len() as u64;
    let mut last_ckpt = Instant::now();
    loop {
        let mut batch: Vec<Board> = Vec::new();
        while batch.len() < BATCH {
            let Some(Reverse((_, _, id))) = st.work.pop() else {
                break;
            };
            st.queued.remove(&id);
            let b = st.arena[id as usize];
            if !st.dead.contains(&b) && !st.core.contains_key(&id) {
                batch.push(b);
            }
        }
        if batch.is_empty() {
            break;
        }
        // Zero-growth probe target: everything the closure already tracks.
        let mut known: FxHashSet<Board> = st.core.keys().map(|&i| st.arena[i as usize]).collect();
        known.extend(st.queued.iter().map(|&i| st.arena[i as usize]));
        known.extend(batch.iter().copied());
        let probe = TargetSet::Explicit(known.clone());
        let probe_classes = probe.classes();
        let dead_snapshot = st.dead.clone();
        let guide = &st.guide;
        type Outcome = (Board, Option<(Vec<TableRow>, Vec<Board>)>, Verdict, bool);
        let results: Vec<Outcome> = batch
            .par_iter()
            .map(|&t| {
                // Probe: land the whole tree on known obligations (no growth).
                // Fail-fast budget: a probe miss is a (useless) refutation of
                // a huge explicit target, so cap its cost; the guided family
                // fallback below is what actually limits frontier novelty.
                let mut f = SetForcer::new(&probe, &probe_classes, max_nodes / 8).guide(guide);
                if f.run(&t) == Verdict::Forced {
                    return (t, Some(f.extract(&t)), Verdict::Forced, true);
                }
                // Fallback: the full family, biased toward known leaves.
                let mut f2 =
                    SetForcer::with_exclude(family, &dead_snapshot, &full_classes, max_nodes)
                        .prefer(&known)
                        .guide(guide);
                let v = f2.run(&t);
                if v == Verdict::Forced {
                    (t, Some(f2.extract(&t)), v, false)
                } else {
                    (t, None, v, false)
                }
            })
            .collect();
        let mut probe_hits = 0u32;
        let batch_n = results.len();
        for (t, res, v, probed) in results {
            probe_hits += u32::from(probed);
            let tid = intern(&mut st.arena, &mut st.ids, t);
            match res {
                Some((rows, frontier)) => {
                    // Tree must avoid boards declared dead within this batch.
                    if frontier.iter().any(|fb| st.dead.contains(fb)) {
                        st.push_work(tid);
                        continue;
                    }
                    let mut fr_ids: Vec<u32> = Vec::with_capacity(frontier.len());
                    for fb in &frontier {
                        let fid = intern(&mut st.arena, &mut st.ids, *fb);
                        fr_ids.push(fid);
                        st.used_by.entry(fid).or_default().insert(tid);
                        if fid != tid && !st.core.contains_key(&fid) {
                            st.push_work(fid);
                        }
                    }
                    st.core.insert(tid, fr_ids);
                    st.guide.extend(rows);
                    processed += 1;
                }
                None => {
                    st.dead.insert(t);
                    eprintln!(
                        "[core] dead #{} ({v:?}): {} — repairing dependents",
                        st.dead.len(),
                        to_hex(&t)
                    );
                    if t == seed {
                        return Ok(CoreResult::Refuted(seed));
                    }
                    for parent in st.used_by.remove(&tid).unwrap_or_default() {
                        if let Some(fr) = st.core.remove(&parent) {
                            for fid in &fr {
                                if let Some(s) = st.used_by.get_mut(fid) {
                                    s.remove(&parent);
                                }
                            }
                            st.push_work(parent);
                        }
                    }
                }
            }
        }
        eprintln!(
            "[core] processed {processed}: core {}, dead {}, worklist {}, probe {probe_hits}/{batch_n}, guide {}",
            st.core.len(),
            st.dead.len(),
            st.work.len(),
            st.guide.len()
        );
        if let Some(path) = ckpt
            && last_ckpt.elapsed().as_secs() >= CKPT_SECS
        {
            st.save(path)?;
            last_ckpt = Instant::now();
            eprintln!("[core] checkpoint written to {path}");
        }
    }
    if let Some(path) = ckpt {
        st.save(path)?;
    }
    // Invariant audit: every member's recorded frontier lies inside the core.
    let violations = st
        .core
        .values()
        .flat_map(|fr| fr.iter())
        .filter(|fid| !st.core.contains_key(fid))
        .count();
    if violations > 0 {
        eprintln!("[core] WARNING: {violations} frontier refs escape the core (bug)");
    }
    // Re-extract every member's strategy against the *final* core, so the
    // union table is consistent under mixed replay: any path's last row was
    // recorded by a search whose every leaf lies in the final core.
    let mut boards: Vec<Board> = st.core.keys().map(|&i| st.arena[i as usize]).collect();
    boards.sort_by(|a, b| cmp_words(&encode_bits(a), &encode_bits(b)));
    let core_set: FxHashSet<Board> = boards.iter().copied().collect();
    let final_target = TargetSet::Explicit(core_set);
    let final_classes = final_target.classes();
    let guide = &st.guide;
    eprintln!("[core] re-extracting {} member strategies", boards.len());
    let per_member: Vec<Option<Vec<TableRow>>> = boards
        .par_iter()
        .map(|&t| {
            let mut f = SetForcer::new(&final_target, &final_classes, max_nodes).guide(guide);
            if f.run(&t) == Verdict::Forced {
                return Some(f.extract(&t).0);
            }
            let mut f2 = SetForcer::new(&final_target, &final_classes, 4 * max_nodes);
            if f2.run(&t) == Verdict::Forced {
                Some(f2.extract(&t).0)
            } else {
                None
            }
        })
        .collect();
    let mut table: StrategyTable = StrategyTable::default();
    let mut failures = 0usize;
    for (b, rows) in boards.iter().zip(per_member) {
        match rows {
            Some(rows) => table.extend(rows),
            None => {
                failures += 1;
                eprintln!("[core] re-extraction FAILED for {}", to_hex(b));
            }
        }
    }
    if failures > 0 {
        bail!("{failures} members failed final re-extraction — table incomplete");
    }
    let table: Vec<TableRow> = table.into_iter().collect();
    Ok(CoreResult::Closed(boards, table))
}

// ---------------------------------------------------------------------------
// Rendering / parsing
// ---------------------------------------------------------------------------

fn render(b: &Board) -> String {
    let h = (0..COLS)
        .map(|j| col_height(b, j))
        .max()
        .unwrap_or(0)
        .max(1);
    let mut s = String::new();
    for r in (0..h).rev() {
        for w in b.iter() {
            s.push(if w >> r & 1 == 1 { '#' } else { '.' });
        }
        s.push('\n');
    }
    s
}

fn to_hex(b: &Board) -> String {
    b.iter()
        .map(|w| format!("{w:x}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn from_hex(s: &str) -> Result<Board> {
    let words: Vec<u64> = s
        .split(',')
        .map(|t| u64::from_str_radix(t.trim(), 16).with_context(|| format!("bad hex word {t:?}")))
        .collect::<Result<_>>()?;
    if words.len() != COLS {
        bail!(
            "expected {COLS} comma-separated hex words, got {}",
            words.len()
        );
    }
    let mut b = EMPTY;
    b.copy_from_slice(&words);
    Ok(b)
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "tetris_lean_funnel",
    about = "Lean-proof-engine mirror: funneling policy + forceable-target search for FiveBagReset"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Reproduce the Lean-measured leg-1 fan-outs (col0 4980, canon 3145,
    /// clear 1892) as engine-equivalence evidence.
    Validate,
    /// Histogram the distinct leg finals of a greedy policy over all 5040
    /// orderings.
    Mine {
        #[arg(long, default_value = "clear")]
        policy: String,
        #[arg(long, default_value_t = 30)]
        top: usize,
        /// Source board (10 comma-separated hex column words), default empty.
        #[arg(long)]
        from: Option<String>,
    },
    /// AND-OR force-check every mined final as a singleton leg target.
    Search {
        #[arg(long, default_value = "clear")]
        policy: String,
        /// Only try the N most frequent finals (0 = all).
        #[arg(long, default_value_t = 0)]
        top: usize,
        #[arg(long, default_value_t = 5_000_000)]
        max_nodes: u64,
        #[arg(long)]
        from: Option<String>,
    },
    /// AND-OR force-check one explicit target board.
    Force {
        /// Target board (10 comma-separated hex column words).
        #[arg(long)]
        target: String,
        #[arg(long, default_value_t = 200_000_000)]
        max_nodes: u64,
        #[arg(long)]
        from: Option<String>,
        /// Write the winning strategy table (one row per (state, piece)).
        #[arg(long)]
        emit: Option<String>,
    },
    /// Exhaustively force-check every board with `card` cells confined to the
    /// bottom `rows` rows (boards containing a full row are skipped:
    /// post-`clearLines` boards never contain one, so they are unreachable).
    Enum {
        #[arg(long)]
        card: u32,
        #[arg(long, default_value_t = 2)]
        rows: u32,
        #[arg(long, default_value_t = 50_000_000)]
        max_nodes: u64,
        #[arg(long)]
        from: Option<String>,
    },
    /// Backward-cascade level: enumerate every flat board (card cells within
    /// the bottom `rows` rows) as a leg SOURCE and force-check it into a fixed
    /// target — the empty board by default, a singleton --target, or a target
    /// set read from --targets-file (one board hex per line, e.g. a previous
    /// sweep's --out). Writes the forced sources to --out for the next level.
    Sweep {
        /// Source cardinality to enumerate (ignored with --sources-file).
        #[arg(long)]
        card: Option<u32>,
        #[arg(long, default_value_t = 2)]
        rows: u32,
        /// File of source boards (one hex per line); overrides --card.
        #[arg(long)]
        sources_file: Option<String>,
        /// Singleton target board hex (default: empty board).
        #[arg(long)]
        target: Option<String>,
        /// File of target boards (one hex per line); overrides --target.
        #[arg(long)]
        targets_file: Option<String>,
        #[arg(long, default_value_t = 50_000_000)]
        max_nodes: u64,
        /// Write forced source boards here, one hex per line.
        #[arg(long)]
        out: Option<String>,
    },
    /// Force into the set of all flat boards (cells within the bottom `rows`
    /// rows), then iteratively re-force with the found strategy's frontier as
    /// the target until the frontier size reaches a fixpoint.
    Shrink {
        #[arg(long, default_value_t = 2)]
        rows: u32,
        #[arg(long, default_value_t = 50_000_000)]
        max_nodes: u64,
        #[arg(long)]
        from: Option<String>,
        /// Write the fixpoint strategy table here.
        #[arg(long)]
        emit: Option<String>,
    },
    /// Bag-granularity greatest fixed point: start from the universe of flat
    /// boards (cells within the bottom `rows` rows, ≤ `max_holes` holes, no
    /// full row) and iteratively remove boards that cannot force one full bag
    /// back into the surviving set. A nonempty fixpoint `T` is a
    /// self-recurrent family — `BagWinning T T` — which together with a
    /// forced reach from the empty board certifies `TetrisSolvable`
    /// (Lean: `tetrisSolvable_of_selfBagWinningBool`). If the empty board
    /// itself survives, the reach is trivial (`init ∈ T`).
    Gfp {
        #[arg(long, default_value_t = 3)]
        rows: u32,
        #[arg(long, default_value_t = 0)]
        max_holes: u32,
        #[arg(long, default_value_t = 50_000_000)]
        max_nodes: u64,
        /// Calibration mode: force only `n` evenly spaced sources against the
        /// full universe, report the verdict mix and cost, and exit (no
        /// fixpoint iteration, no convergence claim).
        #[arg(long)]
        sample: Option<usize>,
        /// Write the converged family here, one board hex per line (in
        /// `--sample` mode: the forced sample sources).
        #[arg(long)]
        out: Option<String>,
    },
    /// Piece-granularity product GFP: over all `(board, bagmask)` pairs of a
    /// flat universe, iteratively remove states where some bag piece has no
    /// placement landing on a surviving in-universe state. Exact and
    /// search-free — no forcing trees, no node budgets — the fixpoint is the
    /// maximal closed table (depth-1 atlas) inside the universe, and the
    /// verdict for `(∅, full)` is definitive both ways. On success, writes
    /// the reachable core and strategy table for `lean-emit-closed`.
    ClosedGfp {
        #[arg(long, default_value_t = 3)]
        rows: u32,
        #[arg(long, default_value_t = 0)]
        max_holes: u32,
        /// Build the universe lazily: BFS the boards reachable from ∅ under
        /// placements whose successors satisfy the predicate, instead of
        /// enumerating every predicate board. Any closed table containing
        /// `(∅, full)` lives inside this set, so the verdict for `(∅, full)`
        /// is unchanged; memory scales with the reachable set.
        #[arg(long, default_value_t = false)]
        lazy: bool,
        /// Safety cap on the lazy universe size.
        #[arg(long, default_value_t = 150_000_000)]
        max_boards: usize,
        /// Write the reachable full-bag boards here, one hex per line.
        #[arg(long)]
        out: Option<String>,
        /// Write the reachable strategy table here (`write_table` format).
        #[arg(long)]
        emit: Option<String>,
        /// Keep iterating to the fixpoint after `(∅, full)` dies and report
        /// the survivors — a nonempty fixpoint is an M2-grade closed cycle
        /// set even when unreachable from the empty board.
        #[arg(long, default_value_t = false)]
        fixpoint: bool,
        /// Restrict each (board, piece) to the K best placements by successor
        /// (holes, max height) — the policy generates its own universe, so
        /// memory and certificate size shrink together. A DEAD verdict is
        /// only definitive for the restricted relation; any table found is a
        /// genuine closed table of the full game. 0 = unrestricted.
        #[arg(long, default_value_t = 0)]
        top_k: usize,
    },
    /// Footprint histogram of a strategy table: over the distinct boards in
    /// the table's keys, tabulate (max column height, total holes) — the
    /// minimal predicate universe a closed table built from these
    /// strategies would have to contain.
    TableStats {
        /// Strategy table file (from `core --emit` or a checkpoint sibling).
        #[arg(long)]
        table: String,
    },
    /// Enumerate a flat universe (cells in bottom `rows` rows, holes ≤
    /// `max_holes`, no full row) and write it to a file, one hex per line —
    /// e.g. as an explicit `core --family` fallback.
    Universe {
        #[arg(long, default_value_t = 2)]
        rows: u32,
        #[arg(long, default_value_t = 0)]
        max_holes: u32,
        #[arg(long)]
        out: String,
    },
    /// Greedily close a seed board under forcing-tree frontiers into a small
    /// self-recurrent core (`BagWinning core core`), preferring the current
    /// core as target before falling back to the full family. The family is
    /// only the fallback target and need not be closed — a converged `gfp`
    /// fixpoint, or just a flat universe via `--rows`. Fails loudly at the
    /// first closure member that cannot force into the family at all.
    Core {
        /// Family file: one board hex per line (e.g. output of `gfp --out`).
        #[arg(long, conflicts_with = "rows")]
        family: Option<String>,
        /// Alternatively: the flat predicate family — every board whose cells
        /// sit within the bottom `rows` rows (holes fine, no full row).
        #[arg(long)]
        rows: Option<u32>,
        /// Seed board hex; defaults to the empty board if present in the
        /// family, else the minimum-card member.
        #[arg(long)]
        seed: Option<String>,
        #[arg(long, default_value_t = 50_000_000)]
        max_nodes: u64,
        /// Write the core family here, one board hex per line.
        #[arg(long)]
        out: Option<String>,
        /// Write the union strategy table here.
        #[arg(long)]
        emit: Option<String>,
        /// Checkpoint path: closure state is snapshotted here every ~10
        /// minutes (the union strategy guide goes to `<ckpt>.table`).
        #[arg(long)]
        ckpt: Option<String>,
        /// Resume from --ckpt instead of starting fresh.
        #[arg(long, default_value_t = false)]
        resume: bool,
    },
    /// Emit a self-contained Lean certificate file from a closed core: the
    /// family as a packed state stream plus one pooled (DAG-shared) ICert
    /// stream, certified by `tetrisSolvable_of_checkPICerts_mem` via
    /// `native_decide`. The emission replays the strategy table end-to-end
    /// (every leaf must land in the core), then self-checks by decoding the
    /// stream back with an exact Lean-mirror walk.
    LeanEmit {
        /// Core family file (one board hex per line, from `core --out`).
        #[arg(long)]
        core: String,
        /// Strategy table file (from `core --emit`).
        #[arg(long)]
        table: String,
        /// Output `.lean` path.
        #[arg(long)]
        out: String,
    },
    /// Emit the depth-1 closed-table certificate from a closed core: BFS the
    /// strategy table from `(∅, full)` at piece granularity so every reached
    /// `(board, bag)` state is a first-class member, each carrying one
    /// placement per bag piece that lands directly on another member. The
    /// table IS the atlas; Lean checks seven placement replays per state
    /// (`tetrisSolvable_of_checkClosedStream_mem`) instead of walking
    /// virtual 5040-ordering trees.
    LeanEmitClosed {
        /// Core family file (one board hex per line, from `core --out`).
        #[arg(long)]
        core: String,
        /// Strategy table file (from `core --emit`).
        #[arg(long)]
        table: String,
        /// Output `.lean` path.
        #[arg(long)]
        out: String,
        /// Node budget for the one-bag reach search used when the core does
        /// not contain ∅ (`tetrisSolvable_of_checkClosedStream_reach` form).
        #[arg(long, default_value_t = 50_000_000)]
        reach_max_nodes: u64,
    },
    /// BFS the full adversarial orbit of one deterministic feature-weighted
    /// policy from the empty board. A closed orbit IS an atlas: the policy
    /// generated it, `--out`/`--emit` feed `lean-emit` unchanged.
    Orbit {
        /// Named policy (`clear`, `canon`) or comma-separated integer
        /// weights `card,holes,agg,bump,maxh,cleared,wells,steps`.
        #[arg(long, default_value = "clear")]
        weights: String,
        /// Reject any reachable board with a column height above this.
        #[arg(long, default_value_t = 8)]
        max_h: u32,
        /// Abort once the orbit exceeds this many (board, bag) states.
        #[arg(long, default_value_t = 2_000_000)]
        cap: usize,
        /// Write boundary boards here (core file for `lean-emit`).
        #[arg(long)]
        out: Option<String>,
        /// Write the replay strategy table here (table for `lean-emit`).
        #[arg(long)]
        emit: Option<String>,
    },
    /// Random-search feature weights for closed policy orbits, smallest
    /// orbits first.
    OrbitSweep {
        #[arg(long, default_value_t = 2000)]
        candidates: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 8)]
        max_h: u32,
        #[arg(long, default_value_t = 200_000)]
        cap: usize,
    },
    /// Solve the height-bounded safety game from (∅, full bag): greedy
    /// weighted defaults patched by backward loss propagation. On success the
    /// surviving strategy's closed orbit is replay-validated and written as
    /// core + table for `lean-emit`.
    Solve {
        /// Ordering heuristic: named policy or comma-separated weights
        /// (see `orbit --weights`).
        #[arg(long, default_value = "clear")]
        weights: String,
        /// Mark any board with a column height above this as lost.
        #[arg(long, default_value_t = 6)]
        max_h: u32,
        /// Abort once this many distinct boards have been chosen.
        #[arg(long, default_value_t = 20_000_000)]
        cap: usize,
        /// Write boundary boards here (core file for `lean-emit`).
        #[arg(long)]
        out: Option<String>,
        /// Write the replay strategy table here (table for `lean-emit`).
        #[arg(long)]
        emit: Option<String>,
    },
}

fn write_table(path: &str, rows: &[TableRow]) -> Result<()> {
    let mut out = String::new();
    for ((b, bag, p), (rot, col)) in rows {
        out.push_str(&format!(
            "board={} bag={bag:02x} piece={} rot={rot} col={col}\n",
            to_hex(b),
            p.name()
        ));
    }
    std::fs::write(path, out).with_context(|| format!("writing {path}"))?;
    Ok(())
}

fn cmd_shrink(
    rows: u32,
    max_nodes: u64,
    from: &Option<String>,
    emit: &Option<String>,
) -> Result<()> {
    let from = parse_from(from)?;
    let t0 = Instant::now();
    let mut target = TargetSet::FlatPred { max_rows: rows };
    let mut prev: Option<usize> = None;
    for iter in 0.. {
        let classes = target.classes();
        let mut f = SetForcer::new(&target, &classes, max_nodes);
        match f.run(&from) {
            Verdict::Refuted => {
                println!("iter {iter}: REFUTED [{:.2?}]", t0.elapsed());
                return Ok(());
            }
            Verdict::Unknown => {
                println!(
                    "iter {iter}: UNKNOWN (node budget {max_nodes} exhausted) [{:.2?}]",
                    t0.elapsed()
                );
                return Ok(());
            }
            Verdict::Forced => {}
        }
        let (table, frontier) = f.extract(&from);
        println!(
            "iter {iter}: forced, frontier {} (table rows {}, nodes {}, memo {}) [{:.2?}]",
            frontier.len(),
            table.len(),
            f.nodes,
            f.memo.len(),
            t0.elapsed()
        );
        if prev == Some(frontier.len()) {
            println!("fixpoint frontier ({}):", frontier.len());
            for b in &frontier {
                println!("card={} holes={} hex={}", card(b), holes(b), to_hex(b));
                print!("{}", render(b));
            }
            if let Some(path) = emit {
                write_table(path, &table)?;
                println!("wrote {} rows to {path}", table.len());
            }
            return Ok(());
        }
        prev = Some(frontier.len());
        target = TargetSet::Explicit(frontier.into_iter().collect());
    }
    Ok(())
}

fn parse_from(from: &Option<String>) -> Result<Board> {
    match from {
        Some(s) => from_hex(s),
        None => Ok(EMPTY),
    }
}

fn cmd_validate() -> Result<()> {
    let checks = [
        (PolicyKind::Col0, 4980usize),
        (PolicyKind::Canon, 3145),
        (PolicyKind::Clear, 1892),
    ];
    let mut all_ok = true;
    for (kind, expect) in checks {
        let t0 = Instant::now();
        let got = fan_out(kind, &EMPTY);
        let ok = got == expect;
        all_ok &= ok;
        println!(
            "{kind:?}: fan-out {got} (Lean: {expect}) {} [{:.2?}]",
            if ok { "PASS" } else { "FAIL" },
            t0.elapsed()
        );
    }
    if !all_ok {
        bail!("fan-out mismatch against the Lean-measured values");
    }
    println!("Rust mirror matches the Lean engine on all three leg-1 fan-outs.");
    Ok(())
}

fn cmd_mine(policy: &str, top: usize, from: &Option<String>) -> Result<()> {
    let kind = PolicyKind::parse(policy)?;
    let from = parse_from(from)?;
    let t0 = Instant::now();
    let finals = mine_finals(kind, &from);
    println!(
        "{kind:?}: {} distinct finals over 5040 orderings [{:.2?}]",
        finals.len(),
        t0.elapsed()
    );
    for (i, (b, freq)) in finals.iter().take(top).enumerate() {
        println!(
            "#{i} freq={freq} card={} holes={} hex={}",
            card(b),
            holes(b),
            to_hex(b)
        );
        print!("{}", render(b));
    }
    Ok(())
}

fn cmd_search(policy: &str, top: usize, max_nodes: u64, from: &Option<String>) -> Result<()> {
    let kind = PolicyKind::parse(policy)?;
    let from = parse_from(from)?;
    let t0 = Instant::now();
    let finals = mine_finals(kind, &from);
    println!(
        "{kind:?}: {} distinct finals over 5040 orderings [{:.2?}]; force-checking {}",
        finals.len(),
        t0.elapsed(),
        if top == 0 {
            "all".to_string()
        } else {
            format!("top {top}")
        }
    );
    let cands: Vec<(Board, u32)> = if top == 0 {
        finals
    } else {
        finals.into_iter().take(top).collect()
    };
    force_check_pool(&cands, &from, max_nodes, t0);
    Ok(())
}

/// Force-check a pool of candidate singleton targets in parallel and print a
/// verdict summary plus details for any forced target.
fn force_check_pool(cands: &[(Board, u32)], from: &Board, max_nodes: u64, t0: Instant) {
    let total = cands.len();
    let done = AtomicU64::new(0);
    let results: Vec<(Board, u32, Verdict, u64)> = cands
        .par_iter()
        .map(|(target, freq)| {
            let mut f = Forcer::new(*target, max_nodes);
            let v = f.run(from);
            let n = done.fetch_add(1, AtomicOrdering::Relaxed) + 1;
            if n % 10000 == 0 || v == Verdict::Forced {
                eprintln!(
                    "[{n}/{total}] {v:?} freq={freq} card={} nodes={} hex={}",
                    card(target),
                    f.nodes,
                    to_hex(target)
                );
            }
            (*target, *freq, v, f.nodes)
        })
        .collect();
    let forced: Vec<_> = results.iter().filter(|r| r.2 == Verdict::Forced).collect();
    let refuted = results.iter().filter(|r| r.2 == Verdict::Refuted).count();
    let unknown = results.iter().filter(|r| r.2 == Verdict::Unknown).count();
    let max_nodes_seen = results.iter().map(|r| r.3).max().unwrap_or(0);
    println!(
        "verdicts: {} forced, {refuted} refuted, {unknown} unknown (budget {max_nodes}); max nodes/candidate {max_nodes_seen} [{:.2?}]",
        forced.len(),
        t0.elapsed()
    );
    for (b, freq, _, nodes) in &forced {
        println!(
            "FORCED freq={freq} card={} nodes={nodes} hex={}",
            card(b),
            to_hex(b)
        );
        print!("{}", render(b));
        // Re-run to extract the table (cheap relative to the search).
        let mut f = Forcer::new(*b, u64::MAX);
        if f.run(from) == Verdict::Forced {
            let rows = extract_table(&f, from);
            println!("strategy table rows: {}", rows.len());
        }
    }
}

/// All boards with exactly `cardinality` cells confined to the bottom `rows`
/// rows and no full row.
fn enum_flat_boards(cardinality: u32, rows: u32) -> Vec<Board> {
    let n = COLS as u32 * rows;
    let mut out = Vec::new();
    // Gosper's hack over n-bit masks with exactly `cardinality` set bits.
    if cardinality == 0 || cardinality > n {
        return out;
    }
    let limit: u64 = 1u64 << n;
    let mut mask: u64 = (1u64 << cardinality) - 1;
    while mask < limit {
        let mut b = EMPTY;
        let mut rest = mask;
        while rest != 0 {
            let i = rest.trailing_zeros();
            b[(i % 10) as usize] |= 1u64 << (i / 10);
            rest &= rest - 1;
        }
        let full = b.iter().fold(!0u64, |acc, &w| acc & w);
        if full == 0 {
            out.push(b);
        }
        // next mask with same popcount
        let c = mask & mask.wrapping_neg();
        let r = mask + c;
        mask = (((r ^ mask) >> 2) / c) | r;
    }
    out
}

fn cmd_enum(cardinality: u32, rows: u32, max_nodes: u64, from: &Option<String>) -> Result<()> {
    if rows > 3 {
        bail!("rows > 3 produces an intractable candidate pool");
    }
    let from = parse_from(from)?;
    let t0 = Instant::now();
    let boards = enum_flat_boards(cardinality, rows);
    println!(
        "enumerated {} candidate boards (card={cardinality}, rows≤{rows}, no full row) [{:.2?}]",
        boards.len(),
        t0.elapsed()
    );
    let cands: Vec<(Board, u32)> = boards.into_iter().map(|b| (b, 0)).collect();
    force_check_pool(&cands, &from, max_nodes, t0);
    Ok(())
}

fn cmd_sweep(
    cardinality: Option<u32>,
    rows: u32,
    sources_file: &Option<String>,
    target: &Option<String>,
    targets_file: &Option<String>,
    max_nodes: u64,
    out: &Option<String>,
) -> Result<()> {
    let t0 = Instant::now();
    let (sources, src_desc) = if let Some(path) = sources_file {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        let v: Vec<Board> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(from_hex)
            .collect::<Result<_>>()?;
        if v.is_empty() {
            bail!("no source boards in {path}");
        }
        (v, format!("from {path}"))
    } else {
        let card = cardinality.context("--card is required without --sources-file")?;
        if rows > 3 {
            bail!("rows > 3 produces an intractable source pool");
        }
        (
            enum_flat_boards(card, rows),
            format!("card={card}, rows≤{rows}"),
        )
    };
    let total = sources.len();
    let done = AtomicU64::new(0);
    let forced_ct = AtomicU64::new(0);
    let step = (total / 40).max(1) as u64;
    let progress = |v: Verdict| {
        let n = done.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        if v == Verdict::Forced {
            forced_ct.fetch_add(1, AtomicOrdering::Relaxed);
        }
        if n % step == 0 || n as usize == total {
            eprintln!(
                "[{n}/{total}] forced so far: {} [{:.2?}]",
                forced_ct.load(AtomicOrdering::Relaxed),
                t0.elapsed()
            );
        }
    };
    let results: Vec<(Board, Verdict, u64)> = if let Some(path) = targets_file {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        let set: FxHashSet<Board> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(from_hex)
            .collect::<Result<_>>()?;
        if set.is_empty() {
            bail!("no target boards in {path}");
        }
        println!(
            "sweep: {total} sources ({src_desc}) → {} targets from {path}",
            set.len()
        );
        let targets = TargetSet::Explicit(set);
        let classes = targets.classes();
        sources
            .par_iter()
            .map(|src| {
                let mut f = SetForcer::new(&targets, &classes, max_nodes);
                let v = f.run(src);
                progress(v);
                (*src, v, f.nodes)
            })
            .collect()
    } else {
        let tb = match target {
            Some(s) => from_hex(s)?,
            None => EMPTY,
        };
        println!(
            "sweep: {total} sources ({src_desc}) → singleton target {}",
            to_hex(&tb)
        );
        sources
            .par_iter()
            .map(|src| {
                let mut f = Forcer::new(tb, max_nodes);
                let v = f.run(src);
                progress(v);
                (*src, v, f.nodes)
            })
            .collect()
    };
    let forced: Vec<Board> = results
        .iter()
        .filter(|r| r.1 == Verdict::Forced)
        .map(|r| r.0)
        .collect();
    let refuted = results.iter().filter(|r| r.1 == Verdict::Refuted).count();
    let unknown = results.iter().filter(|r| r.1 == Verdict::Unknown).count();
    let max_seen = results.iter().map(|r| r.2).max().unwrap_or(0);
    println!(
        "verdicts: {} forced, {refuted} refuted, {unknown} unknown (budget {max_nodes}); max nodes/source {max_seen} [{:.2?}]",
        forced.len(),
        t0.elapsed()
    );
    for b in forced.iter().take(8) {
        println!("FORCED card={} hex={}", card(b), to_hex(b));
        print!("{}", render(b));
    }
    if forced.len() > 8 {
        println!("... and {} more forced sources", forced.len() - 8);
    }
    if unknown > 0 {
        println!("warning: {unknown} unknowns — the forced list is sound but possibly incomplete");
    }
    if let Some(path) = out {
        let text: String = forced.iter().map(|b| format!("{}\n", to_hex(b))).collect();
        std::fs::write(path, text).with_context(|| format!("writing {path}"))?;
        println!("wrote {} forced sources to {path}", forced.len());
    }
    Ok(())
}

/// All boards whose cells sit in the bottom `rows` rows with at most
/// `max_holes` total holes and no full row (post-`clearLines` boards never
/// contain one). Column-wise DFS over the `2^rows` per-column patterns with a
/// running hole budget.
fn enum_universe(rows: u32, max_holes: u32) -> Vec<Board> {
    // (pattern, holes): holes of a column = (top row + 1) − cell count.
    let pats: Vec<(u64, u32)> = (0..(1u64 << rows))
        .map(|p| {
            let holes = if p == 0 {
                0
            } else {
                (64 - p.leading_zeros()) - p.count_ones()
            };
            (p, holes)
        })
        .filter(|&(_, h)| h <= max_holes)
        .collect();
    fn rec(pats: &[(u64, u32)], b: &mut Board, j: usize, budget: u32, out: &mut Vec<Board>) {
        if j == COLS {
            if b.iter().fold(!0u64, |acc, &w| acc & w) == 0 {
                out.push(*b);
            }
            return;
        }
        for &(p, h) in pats {
            if h > budget {
                continue;
            }
            b[j] = p;
            rec(pats, b, j + 1, budget - h, out);
        }
        b[j] = 0;
    }
    let mut out = Vec::new();
    rec(&pats, &mut EMPTY.clone(), 0, max_holes, &mut out);
    out
}

/// Pack a board with all columns < 2^rows into `rows` bits per column
/// (requires rows ≤ 6); `None` if any column exceeds the height cap.
fn pack_board(b: &Board, rows: u32) -> Option<u64> {
    let mut acc = 0u64;
    for (j, &w) in b.iter().enumerate() {
        if w >> rows != 0 {
            return None;
        }
        acc |= w << (j as u32 * rows);
    }
    Some(acc)
}

fn unpack_board(p: u64, rows: u32) -> Board {
    let mask = (1u64 << rows) - 1;
    std::array::from_fn(|j| (p >> (j as u32 * rows)) & mask)
}

/// Upper bound on `SHAPES.valid[p].len()` across pieces (O: 4 rotations × 9
/// cols = 36; I horizontal+vertical mix = 34; others ≤ 36).
const MAX_PLACEMENTS: usize = 40;

/// The `k` best predicate-passing placements for `(b, p)`, ranked by successor
/// `(holes, max height, packed board)` with duplicate successor boards
/// collapsed. One deterministic restricted successor relation shared by the
/// lazy BFS, the GFP sweep, and the extraction — any closed table found under
/// it is a genuine closed table of the full game (soundness unaffected;
/// completeness restricted). Returns `(rot, col, packed successor)` triples;
/// requires `k ≥ 1`.
fn topk_placements(
    b: &Board,
    p: Piece,
    rows: u32,
    max_holes: u32,
    k: usize,
) -> ([(u8, u32, u64); MAX_PLACEMENTS], usize) {
    let mut scored = [(0u32, 0u32, 0u64, 0u8, 0u32); MAX_PLACEMENTS];
    let mut n = 0;
    for &(rot, col) in &SHAPES.valid[p as usize] {
        let nb = apply_step(b, p, rot, col);
        let Some(np) = pack_board(&nb, rows) else {
            continue;
        };
        let h = holes(&nb);
        if h > max_holes {
            continue;
        }
        let mh = (0..COLS).map(|j| col_height(&nb, j)).max().unwrap_or(0);
        scored[n] = (h, mh, np, rot, col);
        n += 1;
    }
    scored[..n]
        .sort_unstable_by_key(|&(h, mh, np, rot, col)| (h, mh, np, u32::from(rot) * 10 + col));
    let mut out = [(0u8, 0u32, 0u64); MAX_PLACEMENTS];
    let mut len = 0;
    for &(_, _, np, rot, col) in &scored[..n] {
        // Equal packed successors are contiguous after the sort (same board ⇒
        // same holes/height), so adjacent-dedup against the last kept entry
        // collapses them.
        if len > 0 && out[len - 1].2 == np {
            continue;
        }
        out[len] = (rot, col, np);
        len += 1;
        if len == k {
            break;
        }
    }
    (out, len)
}

fn cmd_gfp(
    rows: u32,
    max_holes: u32,
    max_nodes: u64,
    sample: Option<usize>,
    out: &Option<String>,
) -> Result<()> {
    if rows > 6 {
        bail!("rows > 6 produces an intractable universe");
    }
    let t0 = Instant::now();
    let universe = enum_universe(rows, max_holes);
    println!(
        "universe: {} boards (rows≤{rows}, holes≤{max_holes}, no full row) [{:.2?}]",
        universe.len(),
        t0.elapsed()
    );
    if universe.len() > 64_000_000 {
        bail!("universe too large; tighten --rows/--max-holes");
    }
    if let Some(n) = sample {
        let target = TargetSet::Explicit(universe.iter().copied().collect());
        let classes = target.classes();
        let stride = (universe.len() / n.max(1)).max(1);
        let sources: Vec<Board> = universe.iter().step_by(stride).take(n).copied().collect();
        let done = AtomicU64::new(0);
        let forced_ct = AtomicU64::new(0);
        let refuted_ct = AtomicU64::new(0);
        let verdicts: Vec<(Verdict, u64)> = sources
            .par_iter()
            .map(|src| {
                let mut f = SetForcer::new(&target, &classes, max_nodes);
                let v = f.run(src);
                match v {
                    Verdict::Forced => forced_ct.fetch_add(1, AtomicOrdering::Relaxed),
                    Verdict::Refuted => refuted_ct.fetch_add(1, AtomicOrdering::Relaxed),
                    Verdict::Unknown => 0,
                };
                let n_done = done.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                if n_done % 10 == 0 {
                    eprintln!(
                        "[sample] [{n_done}/{}] forced {} refuted {} [{:.2?}]",
                        sources.len(),
                        forced_ct.load(AtomicOrdering::Relaxed),
                        refuted_ct.load(AtomicOrdering::Relaxed),
                        t0.elapsed()
                    );
                }
                (v, f.nodes)
            })
            .collect();
        let forced = verdicts.iter().filter(|r| r.0 == Verdict::Forced).count();
        let refuted = verdicts.iter().filter(|r| r.0 == Verdict::Refuted).count();
        let unknown = verdicts.iter().filter(|r| r.0 == Verdict::Unknown).count();
        let mut nodes: Vec<u64> = verdicts.iter().map(|r| r.1).collect();
        nodes.sort_unstable();
        println!(
            "sample {} of {}: forced {forced}, refuted {refuted}, unknown {unknown}; nodes p50 {} p90 {} max {} [{:.2?}]",
            sources.len(),
            universe.len(),
            nodes[nodes.len() / 2],
            nodes[nodes.len() * 9 / 10],
            nodes.last().copied().unwrap_or(0),
            t0.elapsed()
        );
        for (src, (v, _)) in sources.iter().zip(verdicts.iter()) {
            if *v == Verdict::Forced {
                println!("forced: {} (card {})", to_hex(src), card(src));
            }
        }
        if let Some(path) = out {
            let text: String = sources
                .iter()
                .zip(verdicts.iter())
                .filter(|(_, (v, _))| *v == Verdict::Forced)
                .map(|(src, _)| format!("{}\n", to_hex(src)))
                .collect();
            std::fs::write(path, text).with_context(|| format!("writing {path}"))?;
        }
        return Ok(());
    }
    let mut family = universe;
    for iter in 1u32.. {
        let target = TargetSet::Explicit(family.iter().copied().collect());
        let classes = target.classes();
        let total = family.len();
        let done = AtomicU64::new(0);
        let kept_ct = AtomicU64::new(0);
        let step = (total / 40).max(1) as u64;
        let verdicts: Vec<(Board, Verdict)> = family
            .par_iter()
            .map(|src| {
                let mut f = SetForcer::new(&target, &classes, max_nodes);
                let v = f.run(src);
                let n = done.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                if v == Verdict::Forced {
                    kept_ct.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if n % step == 0 || n as usize == total {
                    eprintln!(
                        "[iter {iter}] [{n}/{total}] kept so far: {} [{:.2?}]",
                        kept_ct.load(AtomicOrdering::Relaxed),
                        t0.elapsed()
                    );
                }
                (*src, v)
            })
            .collect();
        let kept: Vec<Board> = verdicts
            .iter()
            .filter(|r| r.1 == Verdict::Forced)
            .map(|r| r.0)
            .collect();
        let unknown = verdicts.iter().filter(|r| r.1 == Verdict::Unknown).count();
        println!(
            "iter {iter}: {total} → {} kept ({unknown} unknown dropped) [{:.2?}]",
            kept.len(),
            t0.elapsed()
        );
        if kept.is_empty() {
            println!("GFP EMPTY — no self-recurrent family within this universe");
            return Ok(());
        }
        if kept.len() == total {
            println!(
                "GFP CONVERGED: self-recurrent family of {} boards (BagWinning T T)",
                kept.len()
            );
            let card_min = kept.iter().map(card).min().unwrap_or(0);
            let card_max = kept.iter().map(card).max().unwrap_or(0);
            println!("family card range: {card_min}..{card_max}");
            if kept.contains(&EMPTY) {
                println!("family contains the EMPTY board — init ∈ T, reach is trivial (j = 0)");
            } else {
                let mut f = SetForcer::new(&target, &classes, max_nodes);
                let v = f.run(&EMPTY);
                println!(
                    "reach check (one leg from empty): {v:?} (nodes {})",
                    f.nodes
                );
            }
            if let Some(path) = out {
                let text: String = kept.iter().map(|b| format!("{}\n", to_hex(b))).collect();
                std::fs::write(path, text).with_context(|| format!("writing {path}"))?;
                println!("wrote {} family boards to {path}", kept.len());
            }
            let fam_set: FxHashSet<Board> = kept.iter().copied().collect();
            let seed = pick_seed(&fam_set)?;
            let fam = TargetSet::Explicit(fam_set);
            println!(
                "core extraction from seed {} (card {})",
                to_hex(&seed),
                card(&seed)
            );
            match extract_core(&fam, seed, max_nodes, None, false)? {
                CoreResult::Closed(core, _) => {
                    println!(
                        "core: {} boards (self-recurrent closure of seed)",
                        core.len()
                    );
                    if let Some(path) = out {
                        let cpath = format!("{path}.core");
                        let text: String =
                            core.iter().map(|b| format!("{}\n", to_hex(b))).collect();
                        std::fs::write(&cpath, text).with_context(|| format!("writing {cpath}"))?;
                        println!("wrote core to {cpath}");
                    }
                }
                CoreResult::Refuted(b) => println!(
                    "core extraction: seed {} learned dead — impossible for a converged family (bug?)",
                    to_hex(&b)
                ),
            }
            return Ok(());
        }
        family = kept;
    }
    Ok(())
}

fn cmd_table_stats(table_path: &str) -> Result<()> {
    let t0 = Instant::now();
    let choice = parse_table(table_path)?;
    let boards: FxHashSet<Board> = choice.keys().map(|(b, _, _)| *b).collect();
    let mut hist: FxHashMap<(u32, u32), usize> = FxHashMap::default();
    for b in &boards {
        let h = (0..COLS).map(|j| col_height(b, j)).max().unwrap_or(0);
        let holes: u32 = b
            .iter()
            .map(|&w| {
                if w == 0 {
                    0
                } else {
                    (64 - w.leading_zeros()) - w.count_ones()
                }
            })
            .sum();
        *hist.entry((h, holes)).or_insert(0) += 1;
    }
    println!(
        "table {} rows, {} distinct boards [{:.2?}]",
        choice.len(),
        boards.len(),
        t0.elapsed()
    );
    let max_h = hist.keys().map(|k| k.0).max().unwrap_or(0);
    let max_k = hist.keys().map(|k| k.1).max().unwrap_or(0);
    println!("max height {max_h}, max holes {max_k}");
    let total = boards.len();
    println!("cumulative coverage by (height ≤ H, holes ≤ K):");
    for h in 0..=max_h {
        let mut line = format!("  H≤{h}:");
        for k in 0..=max_k {
            let cum: usize = hist
                .iter()
                .filter(|((hh, kk), _)| *hh <= h && *kk <= k)
                .map(|(_, c)| c)
                .sum();
            line.push_str(&format!(" K≤{k} {:.1}%", 100.0 * cum as f64 / total as f64));
        }
        println!("{line}");
    }
    Ok(())
}

/// Exact piece-granularity GFP over `(board, bagmask)` pairs. `alive[i]`
/// holds one bit per mask `m ∈ 1..=127`: state `(i, m)` is alive while every
/// piece in `m` has a placement whose `(successor, draw(m, p))` is alive and
/// in-universe. Jacobi sweeps from all-alive are monotone decreasing, so the
/// fixpoint is the maximal closed table inside the universe.
fn cmd_closed_gfp(
    rows: u32,
    max_holes: u32,
    lazy: bool,
    max_boards: usize,
    out: &Option<String>,
    emit: &Option<String>,
    fixpoint: bool,
    top_k: usize,
) -> Result<()> {
    let t0 = Instant::now();
    if !(1..=6).contains(&rows) {
        bail!("rows must be in 1..=6 (boards pack into ≤60 bits)");
    }
    if !lazy && rows > 4 {
        bail!("rows > 4: dense universe too large to enumerate; use --lazy");
    }
    if top_k > 0 {
        println!(
            "policy restriction: top-{top_k} placements per (board, piece) by successor \
             (holes, max height) — DEAD verdicts are relative to this relation"
        );
    }
    // Universe boards packed `rows` bits per column; ∅ packs to 0.
    let (boards, index): (Vec<u64>, FxHashMap<u64, u32>) = if lazy {
        // BFS from ∅ over placements whose successors satisfy the predicate.
        // Every in-predicate successor of a reachable board is reachable, so
        // the maximal closed table here is the reachable part of the dense
        // one and the verdict for `(∅, full)` matches the dense run.
        let mut index: FxHashMap<u64, u32> = FxHashMap::default();
        let mut boards: Vec<u64> = vec![0];
        index.insert(0, 0);
        let mut frontier: Vec<u64> = vec![0];
        let mut layer = 0u32;
        while !frontier.is_empty() {
            layer += 1;
            let mut next: Vec<u64> = Vec::new();
            for chunk in frontier.chunks(1 << 21) {
                let mut cands: Vec<u64> = chunk
                    .par_iter()
                    .flat_map_iter(|&pk| {
                        let b = unpack_board(pk, rows);
                        let mut out: Vec<u64> = Vec::new();
                        for &p in &PIECES {
                            if top_k > 0 {
                                let (pl, len) = topk_placements(&b, p, rows, max_holes, top_k);
                                for &(_, _, np) in &pl[..len] {
                                    if !index.contains_key(&np) {
                                        out.push(np);
                                    }
                                }
                                continue;
                            }
                            for &(rot, col) in &SHAPES.valid[p as usize] {
                                let nb = apply_step(&b, p, rot, col);
                                if let Some(np) = pack_board(&nb, rows) {
                                    if holes(&nb) <= max_holes && !index.contains_key(&np) {
                                        out.push(np);
                                    }
                                }
                            }
                        }
                        out.into_iter()
                    })
                    .collect();
                // Intra-chunk duplicates vastly outnumber unique candidates at
                // 100M+-board scale; collapse them in parallel before the
                // sequential cache-missing index inserts.
                cands.par_sort_unstable();
                cands.dedup();
                index.reserve(cands.len());
                for np in cands {
                    if let std::collections::hash_map::Entry::Vacant(e) = index.entry(np) {
                        e.insert(boards.len() as u32);
                        boards.push(np);
                        next.push(np);
                    }
                }
                if boards.len() > max_boards {
                    bail!(
                        "lazy universe exceeded --max-boards {max_boards} at layer {layer}; \
                         tighten the predicate or raise the cap"
                    );
                }
            }
            println!(
                "[lazy] layer {layer}: +{} boards, total {} [{:.2?}]",
                next.len(),
                boards.len(),
                t0.elapsed()
            );
            frontier = next;
        }
        (boards, index)
    } else {
        let universe = enum_universe(rows, max_holes);
        let boards = universe
            .iter()
            .map(|b| pack_board(b, rows).context("enum_universe board exceeds rows"))
            .collect::<Result<Vec<u64>>>()?;
        let index = boards
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i as u32))
            .collect();
        (boards, index)
    };
    let n = boards.len();
    println!(
        "universe: {n} boards ({}rows≤{rows}, holes≤{max_holes}, no full row) [{:.2?}]",
        if lazy { "reachable from ∅, " } else { "" },
        t0.elapsed()
    );
    let empty_idx = *index
        .get(&0)
        .ok_or_else(|| anyhow::anyhow!("universe must contain the empty board"))?
        as usize;
    const FULL_BIT: u128 = 1u128 << 127;
    const ALL_MASKS: u128 = !1u128;
    let mut alive: Vec<u128> = vec![ALL_MASKS; n];
    let mut sweep = 0u32;
    loop {
        sweep += 1;
        let next: Vec<u128> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut acc = alive[i];
                if acc == 0 {
                    return 0;
                }
                let b = unpack_board(boards[i], rows);
                for (k, &p) in PIECES.iter().enumerate() {
                    // a = OR of alive masks over in-universe successors,
                    // recomputed on the fly (no materialized edge table).
                    let mut a: u128 = 0;
                    if top_k > 0 {
                        let (pl, len) = topk_placements(&b, p, rows, max_holes, top_k);
                        for &(_, _, np) in &pl[..len] {
                            if let Some(&j) = index.get(&np) {
                                a |= alive[j as usize];
                                if a == ALL_MASKS {
                                    break;
                                }
                            }
                        }
                    } else {
                        for &(rot, col) in &SHAPES.valid[p as usize] {
                            let nb = apply_step(&b, p, rot, col);
                            if let Some(&j) = pack_board(&nb, rows).and_then(|np| index.get(&np)) {
                                a |= alive[j as usize];
                                if a == ALL_MASKS {
                                    break;
                                }
                            }
                        }
                    }
                    if a == ALL_MASKS {
                        continue;
                    }
                    let pb = 1u32 << k;
                    let mut ok: u128 = !0;
                    for m in 1u32..128 {
                        if m & pb == 0 {
                            continue;
                        }
                        let child = if m == pb { 127 } else { m ^ pb };
                        if a & (1u128 << child) == 0 {
                            ok &= !(1u128 << m);
                        }
                    }
                    acc &= ok;
                    if acc == 0 {
                        break;
                    }
                }
                acc
            })
            .collect();
        let changed = next
            .par_iter()
            .zip(alive.par_iter())
            .filter(|(a, b)| a != b)
            .count();
        let live_states: u64 = next.par_iter().map(|a| u64::from(a.count_ones())).sum();
        let live_boards = next.par_iter().filter(|a| **a != 0).count();
        let empty_alive = next[empty_idx] & FULL_BIT != 0;
        println!(
            "[gfp] sweep {sweep}: changed {changed}, live (board,bag) {live_states}, live boards {live_boards}, (∅,full) {} [{:.2?}]",
            if empty_alive { "alive" } else { "DEAD" },
            t0.elapsed()
        );
        alive = next;
        if !empty_alive && !fixpoint {
            bail!(
                "(∅, full) died at sweep {sweep} — no closed table exists within this universe; \
                 widen it (--rows/--max-holes)"
            );
        }
        if changed == 0 {
            break;
        }
    }
    let seed_idx: usize = if alive[empty_idx] & FULL_BIT != 0 {
        println!("fixpoint reached: the maximal closed table contains (∅, full)");
        empty_idx
    } else {
        // Only reachable under --fixpoint: ∅ is dead, but a nonempty fixpoint
        // is still a genuine closed table (an M2 cycle set) — extract a
        // compact closure from a seed and certify it with a reach leg
        // (`tetrisSolvable_of_checkClosedStream_reach`).
        let live_states: u64 = alive.par_iter().map(|a| u64::from(a.count_ones())).sum();
        let survivors: Vec<usize> = (0..n).filter(|&i| alive[i] != 0).collect();
        println!(
            "fixpoint WITHOUT (∅, full): {live_states} live (board,bag) states over {} boards [{:.2?}]",
            survivors.len(),
            t0.elapsed()
        );
        let mut sample: Vec<usize> = survivors.clone();
        sample.sort_by_key(|&i| (card(&unpack_board(boards[i], rows)), i));
        for &i in sample.iter().take(8) {
            let b = unpack_board(boards[i], rows);
            println!(
                "  survivor: {} (card {}, masks {})",
                to_hex(&b),
                card(&b),
                alive[i].count_ones()
            );
        }
        let full_bag_survivors = survivors
            .iter()
            .filter(|&&i| alive[i] & FULL_BIT != 0)
            .count();
        println!("full-bag survivors: {full_bag_survivors}");
        let seed = survivors
            .iter()
            .filter(|&&i| alive[i] & FULL_BIT != 0)
            .min_by_key(|&&i| (card(&unpack_board(boards[i], rows)), i))
            .copied();
        match seed {
            Some(i) => {
                println!(
                    "seeding extraction from min-card full-bag survivor {}",
                    to_hex(&unpack_board(boards[i], rows))
                );
                i
            }
            None => {
                println!("no full-bag survivors — unusable for a bag-aligned reach leg");
                return Ok(());
            }
        }
    };
    // Reachable extraction from the seed: prefer successors already reached
    // so the emitted table stays compact, else take the first alive one.
    let enc = |i: u32, m: u32| (u64::from(i) << 7) | u64::from(m);
    let mut seen: FxHashSet<u64> = FxHashSet::default();
    let mut queue: std::collections::VecDeque<(u32, u32)> = std::collections::VecDeque::new();
    let mut table: Vec<TableRow> = Vec::new();
    seen.insert(enc(seed_idx as u32, 127));
    queue.push_back((seed_idx as u32, 127));
    while let Some((i, m)) = queue.pop_front() {
        let b = unpack_board(boards[i as usize], rows);
        for (k, &p) in PIECES.iter().enumerate() {
            let pb = 1u32 << k;
            if m & pb == 0 {
                continue;
            }
            let child_mask = if m == pb { 127 } else { m ^ pb };
            let live: Vec<(u8, u32)> = if top_k > 0 {
                let (pl, len) = topk_placements(&b, p, rows, max_holes, top_k);
                pl[..len]
                    .iter()
                    .filter_map(|&(rot, col, np)| {
                        index
                            .get(&np)
                            .filter(|&&j| alive[j as usize] & (1u128 << child_mask) != 0)
                            .map(|&j| (rot * 10 + col as u8, j))
                    })
                    .collect()
            } else {
                SHAPES.valid[p as usize]
                    .iter()
                    .filter_map(|&(rot, col)| {
                        let nb = apply_step(&b, p, rot, col);
                        pack_board(&nb, rows)
                            .and_then(|np| index.get(&np))
                            .filter(|&&j| alive[j as usize] & (1u128 << child_mask) != 0)
                            .map(|&j| (rot * 10 + col as u8, j))
                    })
                    .collect()
            };
            let &(code, j) = live
                .iter()
                .find(|&&(_, j)| seen.contains(&enc(j, child_mask)))
                .or_else(|| live.first())
                .ok_or_else(|| {
                    anyhow::anyhow!("fixpoint inconsistency: alive state without alive successor")
                })?;
            table.push(((b, m as Bag, p), (code / 10, u32::from(code % 10))));
            if seen.insert(enc(j, child_mask)) {
                queue.push_back((j, child_mask));
            }
        }
    }
    let reach_boards: FxHashSet<u32> = seen.iter().map(|&s| (s >> 7) as u32).collect();
    let reach_full: Vec<Board> = {
        let mut v: Vec<Board> = seen
            .iter()
            .filter(|&&s| s & 127 == 127)
            .map(|&s| unpack_board(boards[(s >> 7) as usize], rows))
            .collect();
        v.sort_by(|a, b| cmp_words(&encode_bits(a), &encode_bits(b)));
        v
    };
    println!(
        "reachable closed table: {} (board, bag) states over {} boards ({} full-bag), {} table rows [{:.2?}]",
        seen.len(),
        reach_boards.len(),
        reach_full.len(),
        table.len(),
        t0.elapsed()
    );
    if let Some(path) = out {
        let text: String = reach_full
            .iter()
            .map(|b| format!("{}\n", to_hex(b)))
            .collect();
        std::fs::write(path, text).with_context(|| format!("writing {path}"))?;
        println!("wrote core to {path}");
    }
    if let Some(path) = emit {
        write_table(path, &table)?;
        println!("wrote strategy table to {path}");
    }
    Ok(())
}

/// Empty board if present, else the minimum-card member (encoding tiebreak).
fn pick_seed(family: &FxHashSet<Board>) -> Result<Board> {
    if family.contains(&EMPTY) {
        return Ok(EMPTY);
    }
    family
        .iter()
        .min_by(|a, b| {
            card(a)
                .cmp(&card(b))
                .then_with(|| cmp_words(&encode_bits(a), &encode_bits(b)))
        })
        .copied()
        .ok_or_else(|| anyhow::anyhow!("empty family"))
}

fn cmd_core(
    family_path: &Option<String>,
    rows: Option<u32>,
    seed: &Option<String>,
    max_nodes: u64,
    out: &Option<String>,
    emit: &Option<String>,
    ckpt: &Option<String>,
    resume: bool,
) -> Result<()> {
    let t0 = Instant::now();
    let family: TargetSet = match (family_path, rows) {
        (Some(path), None) => {
            let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
            let set: FxHashSet<Board> = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(from_hex)
                .collect::<Result<_>>()?;
            if set.is_empty() {
                bail!("family file is empty");
            }
            TargetSet::Explicit(set)
        }
        (None, Some(r)) => {
            if r > 6 {
                bail!("rows > 6 gives the forcer no useful card classes");
            }
            TargetSet::FlatPred { max_rows: r }
        }
        _ => bail!("provide exactly one of --family or --rows"),
    };
    let seed: Board = match seed {
        Some(s) => from_hex(s)?,
        None => match &family {
            TargetSet::Explicit(set) => pick_seed(set)?,
            TargetSet::FlatPred { .. } => EMPTY,
        },
    };
    if !family.contains(&seed) {
        bail!("seed {} is not in the family", to_hex(&seed));
    }
    let desc = match &family {
        TargetSet::Explicit(set) => format!("{} boards (explicit)", set.len()),
        TargetSet::FlatPred { max_rows } => format!("flat ≤{max_rows} rows (predicate)"),
    };
    println!(
        "family {desc}; seed {} (card {})",
        to_hex(&seed),
        card(&seed)
    );
    let (core, table) = match extract_core(&family, seed, max_nodes, ckpt.as_deref(), resume)? {
        CoreResult::Closed(core, table) => (core, table),
        CoreResult::Refuted(b) => bail!(
            "seed {} learned dead — no closure from this seed within the family \
             (under the node budget); try another seed or a larger universe",
            to_hex(&b)
        ),
    };
    let card_min = core.iter().map(card).min().unwrap_or(0);
    let card_max = core.iter().map(card).max().unwrap_or(0);
    println!(
        "core: {} boards (cards {card_min}..{card_max}), table {} rows [{:.2?}]",
        core.len(),
        table.len(),
        t0.elapsed()
    );
    if let Some(path) = out {
        let text: String = core.iter().map(|b| format!("{}\n", to_hex(b))).collect();
        std::fs::write(path, text).with_context(|| format!("writing {path}"))?;
        println!("wrote core to {path}");
    }
    if let Some(path) = emit {
        write_table(path, &table)?;
        println!("wrote strategy table to {path}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Lean certificate emission (pooled ICert streams)
// ---------------------------------------------------------------------------

fn piece_by_name(s: &str) -> Option<Piece> {
    PIECES.iter().copied().find(|p| p.name() == s)
}

/// `(board, bag, piece) -> (rot, col)` strategy table.
type StrategyTable = FxHashMap<(Board, Bag, Piece), (u8, u32)>;

/// Parse a strategy table written by `write_table`. Duplicate keys (the same
/// `(board, bag, piece)` recorded by different members' searches) overwrite
/// silently — any mixed replay the union produces is validated end-to-end by
/// the emitter (every leaf must land in the core), and re-validated in Lean.
fn parse_table(path: &str) -> Result<StrategyTable> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
    let mut map: StrategyTable = FxHashMap::default();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ctx = || format!("{path}:{}", ln + 1);
        let (mut board, mut bag, mut piece, mut rot, mut col) = (None, None, None, None, None);
        for tok in line.split_whitespace() {
            let (k, v) = tok
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("{}: bad token {tok}", ctx()))?;
            match k {
                "board" => board = Some(from_hex(v)?),
                "bag" => bag = Some(u8::from_str_radix(v, 16).with_context(ctx)?),
                "piece" => {
                    piece = Some(
                        piece_by_name(v)
                            .ok_or_else(|| anyhow::anyhow!("{}: bad piece {v}", ctx()))?,
                    );
                }
                "rot" => rot = Some(v.parse::<u8>().with_context(ctx)?),
                "col" => col = Some(v.parse::<u32>().with_context(ctx)?),
                _ => bail!("{}: unknown key {k}", ctx()),
            }
        }
        let missing = || anyhow::anyhow!("{}: missing field", ctx());
        map.insert(
            (
                board.ok_or_else(missing)?,
                bag.ok_or_else(missing)?,
                piece.ok_or_else(missing)?,
            ),
            (rot.ok_or_else(missing)?, col.ok_or_else(missing)?),
        );
    }
    Ok(map)
}

/// Pooled preorder serializer mirroring `decodePICert` (FiveBagReset.lean):
/// leaf = `0, i` (core index); back-ref = `2, j` (j-th completed node); node
/// = `1, cO..cJ` (code = rot·10 + col; 0 for out-of-bag pieces) followed by
/// the seven child trees in `O,I,S,Z,T,L,J` order (out-of-bag children are
/// dummy `0, 0` leaves), with the node appended to the pool on completion.
struct PoolEmitter<'a> {
    index: &'a FxHashMap<Board, usize>,
    choice: &'a StrategyTable,
    stream: Vec<u64>,
    pool: FxHashMap<(Board, Bag), u64>,
}

impl PoolEmitter<'_> {
    fn emit(&mut self, b: &Board, bag: Bag, depth: u32) -> Result<()> {
        if depth == 7 {
            let i = self.index.get(b).ok_or_else(|| {
                anyhow::anyhow!(
                    "leaf {} escapes the core — table replay is not closed",
                    to_hex(b)
                )
            })?;
            self.stream.push(0);
            self.stream.push(*i as u64);
            return Ok(());
        }
        if let Some(&j) = self.pool.get(&(*b, bag)) {
            self.stream.push(2);
            self.stream.push(j);
            return Ok(());
        }
        self.stream.push(1);
        let mut chosen: [Option<(u8, u32)>; 7] = [None; 7];
        for (k, &p) in PIECES.iter().enumerate() {
            if bag_contains(bag, p) {
                let &(rot, col) = self.choice.get(&(*b, bag, p)).ok_or_else(|| {
                    anyhow::anyhow!(
                        "missing table row: board={} bag={bag:02x} piece={}",
                        to_hex(b),
                        p.name()
                    )
                })?;
                if rot > 3 || col > 9 {
                    bail!("placement out of code range: rot={rot} col={col}");
                }
                chosen[k] = Some((rot, col));
                self.stream.push(u64::from(rot) * 10 + u64::from(col));
            } else {
                self.stream.push(0);
            }
        }
        for (k, &p) in PIECES.iter().enumerate() {
            match chosen[k] {
                Some((rot, col)) => {
                    self.emit(&apply_step(b, p, rot, col), draw(bag, p), depth + 1)?;
                }
                None => {
                    self.stream.push(0);
                    self.stream.push(0);
                }
            }
        }
        let j = self.pool.len() as u64;
        self.pool.insert((*b, bag), j);
        Ok(())
    }
}

/// Decoded mirror node (arena-allocated).
enum RNode {
    Leaf(u64),
    Node {
        codes: [u64; 7],
        children: [usize; 7],
    },
}

/// Exact Lean-mirror decode of one tree off `stream` at `pos`; completed
/// nodes append their arena index to `pool`. Malformed input is a hard error
/// here (the emitter must never produce it; Lean would decode junk and fail).
fn decode_rnode(
    stream: &[u64],
    pos: &mut usize,
    arena: &mut Vec<RNode>,
    pool: &mut Vec<usize>,
) -> Result<usize> {
    let op = *stream
        .get(*pos)
        .ok_or_else(|| anyhow::anyhow!("stream truncated"))?;
    *pos += 1;
    match op {
        0 => {
            let i = *stream
                .get(*pos)
                .ok_or_else(|| anyhow::anyhow!("stream truncated"))?;
            *pos += 1;
            arena.push(RNode::Leaf(i));
            Ok(arena.len() - 1)
        }
        2 => {
            let j = *stream
                .get(*pos)
                .ok_or_else(|| anyhow::anyhow!("stream truncated"))?;
            *pos += 1;
            pool.get(j as usize)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("back-ref {j} out of pool range {}", pool.len()))
        }
        1 => {
            let mut codes = [0u64; 7];
            for c in &mut codes {
                *c = *stream
                    .get(*pos)
                    .ok_or_else(|| anyhow::anyhow!("stream truncated"))?;
                *pos += 1;
            }
            let mut children = [0usize; 7];
            for ch in &mut children {
                *ch = decode_rnode(stream, pos, arena, pool)?;
            }
            arena.push(RNode::Node { codes, children });
            pool.push(arena.len() - 1);
            Ok(arena.len() - 1)
        }
        op => bail!("bad opcode {op} at {}", *pos - 1),
    }
}

/// Walk a decoded root exactly as `checkICert` does: alive at every interior
/// node, every in-bag piece's code names a valid placement, and depth-7
/// leaves equal the indexed core state. Shared nodes are walked once (their
/// `(board, bag)` is pinned by first visit).
fn walk_rnode(
    arena: &[RNode],
    seen: &mut FxHashMap<usize, (Board, Bag)>,
    boards: &[Board],
    idx: usize,
    b: &Board,
    bag: Bag,
    depth: u32,
) -> Result<()> {
    match &arena[idx] {
        RNode::Leaf(i) => {
            if depth != 7 {
                bail!("leaf at depth {depth}");
            }
            let t = boards
                .get(*i as usize)
                .ok_or_else(|| anyhow::anyhow!("leaf index {i} out of range"))?;
            if t != b {
                bail!(
                    "leaf mismatch: tree lands on {} but names {}",
                    to_hex(b),
                    to_hex(t)
                );
            }
            Ok(())
        }
        RNode::Node { codes, children } => {
            if depth == 7 {
                bail!("node at depth 7");
            }
            if let Some(prev) = seen.get(&idx) {
                if *prev != (*b, bag) {
                    bail!("shared node reused with a different (board, bag)");
                }
                return Ok(());
            }
            seen.insert(idx, (*b, bag));
            if is_lost(b) {
                bail!("interior board is lost: {}", to_hex(b));
            }
            for (k, &p) in PIECES.iter().enumerate() {
                if !bag_contains(bag, p) {
                    continue;
                }
                let (rot, col) = ((codes[k] / 10) as u8, (codes[k] % 10) as u32);
                if !SHAPES.valid[p as usize].contains(&(rot, col)) {
                    bail!("invalid placement code {} for {}", codes[k], p.name());
                }
                walk_rnode(
                    arena,
                    seen,
                    boards,
                    children[k],
                    &apply_step(b, p, rot, col),
                    draw(bag, p),
                    depth + 1,
                )?;
            }
            Ok(())
        }
    }
}

/// Decode the full pooled stream back and verify every member against the
/// Lean `checkICert` semantics. Linear in the stream.
fn selfcheck(boards: &[Board], stream: &[u64]) -> Result<()> {
    let mut arena: Vec<RNode> = Vec::new();
    let mut pool: Vec<usize> = Vec::new();
    let mut pos = 0usize;
    let mut roots: Vec<usize> = Vec::with_capacity(boards.len());
    for _ in boards {
        roots.push(decode_rnode(stream, &mut pos, &mut arena, &mut pool)?);
    }
    if pos != stream.len() {
        bail!("stream has {} trailing nats", stream.len() - pos);
    }
    let mut seen: FxHashMap<usize, (Board, Bag)> = FxHashMap::default();
    for (i, &r) in roots.iter().enumerate() {
        walk_rnode(&arena, &mut seen, boards, r, &boards[i], FULL_BAG, 0)?;
    }
    Ok(())
}

/// Transitive closure of ∅ under table-replay leaf edges: walk each member's
/// leg through the union table (every reached `(board, bag, piece)` must have
/// a row), collecting depth-7 leaves as new members. The result is closed by
/// construction and is exactly the family the certificate must cover —
/// obligations nothing references from ∅ are dropped.
fn live_set(choice: &StrategyTable) -> Result<FxHashSet<Board>> {
    let mut live: FxHashSet<Board> = [EMPTY].into_iter().collect();
    let mut stack: Vec<Board> = vec![EMPTY];
    let mut seen: FxHashSet<(Board, Bag)> = FxHashSet::default();
    while let Some(root) = stack.pop() {
        let mut walk: Vec<(Board, Bag, u32)> = vec![(root, FULL_BAG, 0)];
        while let Some((b, bag, depth)) = walk.pop() {
            if depth == 7 {
                if live.insert(b) {
                    stack.push(b);
                }
                continue;
            }
            if !seen.insert((b, bag)) {
                continue;
            }
            for p in PIECES {
                if !bag_contains(bag, p) {
                    continue;
                }
                let Some(&(rot, col)) = choice.get(&(b, bag, p)) else {
                    bail!(
                        "table has no row for board={} bag={bag:02x} piece={}",
                        to_hex(&b),
                        p.name()
                    );
                };
                walk.push((apply_step(&b, p, rot, col), draw(bag, p), depth + 1));
            }
        }
    }
    Ok(live)
}

fn cmd_lean_emit(core_path: &str, table_path: &str, out_path: &str) -> Result<()> {
    let t0 = Instant::now();
    let text =
        std::fs::read_to_string(core_path).with_context(|| format!("reading {core_path}"))?;
    let mut boards: Vec<Board> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(from_hex)
        .collect::<Result<_>>()?;
    boards.sort_by(|a, b| cmp_words(&encode_bits(a), &encode_bits(b)));
    boards.dedup();
    if !boards.contains(&EMPTY) {
        bail!("core must contain the empty board (membership reach is the only emitted form)");
    }
    let choice = parse_table(table_path)?;
    let live = live_set(&choice)?;
    let before = boards.len();
    boards.retain(|b| live.contains(b));
    if boards.len() != live.len() {
        bail!(
            "{} live boards are missing from the core file (table/core mismatch)",
            live.len() - boards.len()
        );
    }
    let index: FxHashMap<Board, usize> = boards.iter().enumerate().map(|(i, b)| (*b, i)).collect();
    println!(
        "core {} of {before} states live from ∅, table {} rows [{:.2?}]",
        boards.len(),
        choice.len(),
        t0.elapsed()
    );
    let mut em = PoolEmitter {
        index: &index,
        choice: &choice,
        stream: Vec::new(),
        pool: FxHashMap::default(),
    };
    for b in &boards {
        em.emit(b, FULL_BAG, 0)?;
    }
    let stream = em.stream;
    println!(
        "cert stream {} nats (pool {} shared nodes) [{:.2?}]",
        stream.len(),
        em.pool.len(),
        t0.elapsed()
    );
    selfcheck(&boards, &stream)?;
    println!("self-check passed: Lean-mirror decode + full checkICert walk");
    let mut bstream: Vec<u64> = Vec::new();
    for b in &boards {
        bstream.push(u64::from(card(b)));
        for (c, &w) in b.iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let r = w.trailing_zeros();
                bstream.push(c as u64);
                bstream.push(u64::from(r));
                w &= w - 1;
            }
        }
    }
    let core_str: String = bstream
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    let cert_str: String = stream
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    let n = boards.len();
    let lean = format!(
        r#"import Proofs.Experiments.FiveBagReset

/-! Auto-generated by `tetris_lean_funnel lean-emit` — do not edit.
Core: {n} flat states; cert stream: {} nats ({} pooled nodes). -/

namespace Tetris
namespace Experiments
namespace FiveBagReset

def coreStr : String := "{core_str}"

def certStr : String := "{cert_str}"

/-- The certified self-recurrent family, decoded from `coreStr`. -/
def coreA : Array GameState := decodeStates {n} (natsOfString coreStr) #[]

set_option linter.style.nativeDecide false in
example : TetrisSolvable :=
  tetrisSolvable_of_checkPICerts_mem (A := coreA) (s := natsOfString certStr)
    (by native_decide) (by native_decide)

end FiveBagReset
end Experiments
end Tetris
"#,
        stream.len(),
        em.pool.len(),
    );
    std::fs::write(out_path, &lean).with_context(|| format!("writing {out_path}"))?;
    println!(
        "wrote {out_path}: {} bytes ({n} states, {} stream nats) [{:.2?}]",
        lean.len(),
        stream.len(),
        t0.elapsed()
    );
    Ok(())
}

/// Mirror of the Lean closed-table check: decode the pooled stream back
/// (`decodePICerts` semantics) and verify each root against `checkICert` at
/// depth 1 — node root, live board, valid placement per bag piece, each
/// child a leaf naming the exact `(board, bag)` successor.
fn selfcheck_closed(states: &[(Board, Bag)], stream: &[u64]) -> Result<()> {
    let mut arena: Vec<RNode> = Vec::new();
    let mut pool: Vec<usize> = Vec::new();
    let mut pos = 0usize;
    let mut roots: Vec<usize> = Vec::with_capacity(states.len());
    for _ in states {
        roots.push(decode_rnode(stream, &mut pos, &mut arena, &mut pool)?);
    }
    if pos != stream.len() {
        bail!("stream has {} trailing nats", stream.len() - pos);
    }
    for (i, &r) in roots.iter().enumerate() {
        let (b, bag) = states[i];
        let RNode::Node { codes, children } = &arena[r] else {
            bail!("root {i} is not a node");
        };
        if is_lost(&b) {
            bail!("member board is lost: {}", to_hex(&b));
        }
        for (k, &p) in PIECES.iter().enumerate() {
            if !bag_contains(bag, p) {
                continue;
            }
            let (rot, col) = ((codes[k] / 10) as u8, (codes[k] % 10) as u32);
            if !SHAPES.valid[p as usize].contains(&(rot, col)) {
                bail!("invalid placement code {} for {}", codes[k], p.name());
            }
            let RNode::Leaf(j) = arena[children[k]] else {
                bail!("root {i} piece {} child is not a leaf", p.name());
            };
            let t = states
                .get(j as usize)
                .ok_or_else(|| anyhow::anyhow!("leaf index {j} out of range"))?;
            if *t != (apply_step(&b, p, rot, col), draw(bag, p)) {
                bail!("root {i} piece {} leaf names the wrong state", p.name());
            }
        }
    }
    Ok(())
}

/// Serialize the recorded winning strategy from `(b, bag)` as a plain
/// (pool-free) preorder ICert stream — node `1 :: cO..cJ`, leaf `0 :: i`,
/// out-of-bag children as dummy `0 :: 0` leaves — with depth-7 leaves naming
/// `index` entries. Shared subtrees re-serialize: the stream is tree-sized.
fn emit_reach_stream(
    f: &SetForcer,
    b: &Board,
    bag: Bag,
    depth: u32,
    index: &FxHashMap<(Board, Bag), usize>,
    out: &mut Vec<u64>,
) -> Result<()> {
    if depth == 7 {
        let &i = index
            .get(&(*b, bag))
            .ok_or_else(|| anyhow::anyhow!("reach leaf {} escapes the table", to_hex(b)))?;
        out.push(0);
        out.push(i as u64);
        return Ok(());
    }
    let mut codes = [0u64; 7];
    let mut children: [Option<(Board, Bag)>; 7] = [None; 7];
    for (k, &p) in PIECES.iter().enumerate() {
        if !bag_contains(bag, p) {
            continue;
        }
        let &(rot, col) = f
            .choice
            .get(&(*b, bag, p))
            .ok_or_else(|| anyhow::anyhow!("missing reach choice at {}", to_hex(b)))?;
        codes[k] = u64::from(rot) * 10 + u64::from(col);
        children[k] = Some((apply_step(b, p, rot, col), draw(bag, p)));
    }
    out.push(1);
    out.extend_from_slice(&codes);
    for child in &children {
        match child {
            Some((nb, nbag)) => emit_reach_stream(f, nb, *nbag, depth + 1, index, out)?,
            None => {
                out.push(0);
                out.push(0);
            }
        }
    }
    Ok(())
}

/// Exact mirror of `checkICert A depth init (decodeICert r.length r).1`:
/// decode the stream (plain streams decode identically under the pooled
/// decoder), then replay — interior nodes need a live board and, per bag
/// piece, a valid placement whose child accepts; depth-0 leaves must name
/// the exact reached state.
fn selfcheck_reach(
    states: &[(Board, Bag)],
    stream: &[u64],
    init: (Board, Bag),
    depth: u32,
) -> Result<()> {
    let mut arena: Vec<RNode> = Vec::new();
    let mut pool: Vec<usize> = Vec::new();
    let mut pos = 0usize;
    let root = decode_rnode(stream, &mut pos, &mut arena, &mut pool)?;
    if pos != stream.len() {
        bail!("reach stream has {} trailing nats", stream.len() - pos);
    }
    fn check(
        arena: &[RNode],
        node: usize,
        b: &Board,
        bag: Bag,
        n: u32,
        states: &[(Board, Bag)],
    ) -> Result<()> {
        if n == 0 {
            let RNode::Leaf(j) = arena[node] else {
                bail!("depth-0 reach node is not a leaf");
            };
            let t = states
                .get(j as usize)
                .ok_or_else(|| anyhow::anyhow!("reach leaf index {j} out of range"))?;
            if *t != (*b, bag) {
                bail!("reach leaf names the wrong state");
            }
            return Ok(());
        }
        let RNode::Node { codes, children } = &arena[node] else {
            bail!("interior reach node is not a node");
        };
        if is_lost(b) {
            bail!("reach path hits a lost board: {}", to_hex(b));
        }
        for (k, &p) in PIECES.iter().enumerate() {
            if !bag_contains(bag, p) {
                continue;
            }
            let (rot, col) = ((codes[k] / 10) as u8, (codes[k] % 10) as u32);
            if !SHAPES.valid[p as usize].contains(&(rot, col)) {
                bail!("invalid reach placement code {} for {}", codes[k], p.name());
            }
            check(
                arena,
                children[k],
                &apply_step(b, p, rot, col),
                draw(bag, p),
                n - 1,
                states,
            )?;
        }
        Ok(())
    }
    check(&arena, root, &init.0, init.1, depth, states)
}

fn cmd_lean_emit_closed(
    core_path: &str,
    table_path: &str,
    out_path: &str,
    reach_max_nodes: u64,
) -> Result<()> {
    let t0 = Instant::now();
    let text =
        std::fs::read_to_string(core_path).with_context(|| format!("reading {core_path}"))?;
    let core_list: Vec<Board> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(from_hex)
        .collect::<Result<_>>()?;
    let core: FxHashSet<Board> = core_list.iter().copied().collect();
    if core.is_empty() {
        bail!("empty core");
    }
    // With ∅ in the core the table is entered at (∅, full) and the mem-form
    // theorem applies; otherwise a one-bag reach certificate is searched and
    // the reach-form theorem is emitted.
    let mem_form = core.contains(&EMPTY);
    let choice = parse_table(table_path)?;
    // Piece-granular BFS from every full-bag core state: the closure of the
    // whole core, so reach legs may land on any member. Core closure
    // guarantees every full-bag stop is a core member with table rows.
    let mut states: Vec<(Board, Bag)> = Vec::new();
    let mut index: FxHashMap<(Board, Bag), usize> = FxHashMap::default();
    for &b in &core_list {
        if let std::collections::hash_map::Entry::Vacant(e) = index.entry((b, FULL_BAG)) {
            e.insert(states.len());
            states.push((b, FULL_BAG));
        }
    }
    let mut codes: Vec<[u64; 7]> = Vec::new();
    let mut succ: Vec<[usize; 7]> = Vec::new();
    let mut i = 0usize;
    while i < states.len() {
        let (b, bag) = states[i];
        if is_lost(&b) {
            bail!("reached a lost board: {}", to_hex(&b));
        }
        if bag == FULL_BAG && !core.contains(&b) {
            bail!("full-bag state {} escapes the core", to_hex(&b));
        }
        let mut row_codes = [0u64; 7];
        let mut row_succ = [usize::MAX; 7];
        for (k, &p) in PIECES.iter().enumerate() {
            if !bag_contains(bag, p) {
                continue;
            }
            let &(rot, col) = choice.get(&(b, bag, p)).ok_or_else(|| {
                anyhow::anyhow!(
                    "missing table row: board={} bag={bag:02x} piece={}",
                    to_hex(&b),
                    p.name()
                )
            })?;
            if !SHAPES.valid[p as usize].contains(&(rot, col)) {
                bail!("invalid placement rot={rot} col={col} for {}", p.name());
            }
            row_codes[k] = u64::from(rot) * 10 + u64::from(col);
            let next = (apply_step(&b, p, rot, col), draw(bag, p));
            let j = match index.get(&next) {
                Some(&j) => j,
                None => {
                    let j = states.len();
                    index.insert(next, j);
                    states.push(next);
                    j
                }
            };
            row_succ[k] = j;
        }
        codes.push(row_codes);
        succ.push(row_succ);
        i += 1;
    }
    let n = states.len();
    let full_bag = states.iter().filter(|(_, m)| *m == FULL_BAG).count();
    println!(
        "closed table: {n} (board, bag) states ({full_bag} full-bag) from {} core boards, table {} rows [{:.2?}]",
        core.len(),
        choice.len(),
        t0.elapsed()
    );
    let mut stream: Vec<u64> = Vec::with_capacity(n * 22);
    for i in 0..n {
        stream.push(1);
        stream.extend_from_slice(&codes[i]);
        for k in 0..7 {
            stream.push(0);
            stream.push(if succ[i][k] == usize::MAX {
                0
            } else {
                succ[i][k] as u64
            });
        }
    }
    selfcheck_closed(&states, &stream)?;
    println!(
        "cert stream {} nats; self-check passed: Lean-mirror decode + depth-1 replay",
        stream.len()
    );
    let mut sstream: Vec<u64> = Vec::new();
    for (b, bag) in &states {
        sstream.push(u64::from(*bag));
        sstream.push(u64::from(card(b)));
        for (c, &w) in b.iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let r = w.trailing_zeros();
                sstream.push(c as u64);
                sstream.push(u64::from(r));
                w &= w - 1;
            }
        }
    }
    let states_str: String = sstream
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    let cert_str: String = stream
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    let (reach_def, example) = if mem_form {
        (
            String::new(),
            "example : TetrisSolvable :=\n  tetrisSolvable_of_checkClosedStream_mem (A := closedA)\n    (s := natsOfString closedCertStr) (by native_decide) (by native_decide)"
                .to_string(),
        )
    } else {
        let target = TargetSet::Explicit(core.clone());
        let classes = target.classes();
        let mut f = SetForcer::new(&target, &classes, reach_max_nodes);
        let verdict = f.run(&EMPTY);
        if !matches!(verdict, Verdict::Forced) {
            bail!(
                "reach leg: (∅, full) not forced into the core in one bag \
                 ({verdict:?} after {} nodes); deepen the search or pick another core",
                f.nodes
            );
        }
        println!(
            "reach: (∅, full) forced into the core in one bag ({} nodes) [{:.2?}]",
            f.nodes,
            t0.elapsed()
        );
        let mut rstream: Vec<u64> = Vec::new();
        emit_reach_stream(&f, &EMPTY, FULL_BAG, 0, &index, &mut rstream)?;
        selfcheck_reach(&states, &rstream, (EMPTY, FULL_BAG), 7)?;
        println!(
            "reach stream {} nats; self-check passed: Lean-mirror decode + depth-7 replay",
            rstream.len()
        );
        let reach_str: String = rstream
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        (
            format!("\ndef closedReachStr : String := \"{reach_str}\"\n"),
            "example : TetrisSolvable :=\n  tetrisSolvable_of_checkClosedStream_reach (A := closedA)\n    (s := natsOfString closedCertStr) (n := 7)\n    (r := natsOfString closedReachStr) (by native_decide) (by native_decide)"
                .to_string(),
        )
    };
    let lean = format!(
        r#"import Proofs.Experiments.FiveBagReset

/-! Auto-generated by `tetris_lean_funnel lean-emit-closed` — do not edit.
Closed table: {n} (board, bag) states; cert stream: {} nats (22 per state). -/

namespace Tetris
namespace Experiments
namespace FiveBagReset

def closedStr : String := "{states_str}"

def closedCertStr : String := "{cert_str}"
{reach_def}
/-- The certified closed table — the depth-1 atlas, decoded from `closedStr`. -/
def closedA : Array GameState := decodeStatesBag {n} (natsOfString closedStr) #[]

set_option linter.style.nativeDecide false in
{example}

end FiveBagReset
end Experiments
end Tetris
"#,
        stream.len(),
    );
    std::fs::write(out_path, &lean).with_context(|| format!("writing {out_path}"))?;
    println!(
        "wrote {out_path}: {} bytes ({n} states, {} cert nats, {} state nats) [{:.2?}]",
        lean.len(),
        stream.len(),
        sstream.len(),
        t0.elapsed()
    );
    Ok(())
}

fn cmd_force(
    target: &str,
    max_nodes: u64,
    from: &Option<String>,
    emit: &Option<String>,
) -> Result<()> {
    let target = from_hex(target)?;
    let from = parse_from(from)?;
    let t0 = Instant::now();
    let mut f = Forcer::new(target, max_nodes);
    let v = f.run(&from);
    println!(
        "{v:?} nodes={} memo={} [{:.2?}]",
        f.nodes,
        f.memo.len(),
        t0.elapsed()
    );
    if v == Verdict::Forced {
        let rows = extract_table(&f, &from);
        println!("strategy table rows: {}", rows.len());
        if let Some(path) = emit {
            let mut out = String::new();
            for ((b, bag, p), (rot, col)) in &rows {
                out.push_str(&format!(
                    "board={} bag={bag:02x} piece={} rot={rot} col={col}\n",
                    to_hex(b),
                    p.name()
                ));
            }
            std::fs::write(path, out).with_context(|| format!("writing {path}"))?;
            println!("wrote {} rows to {path}", rows.len());
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.cmd {
        Cmd::Validate => cmd_validate(),
        Cmd::Mine { policy, top, from } => cmd_mine(policy, *top, from),
        Cmd::Search {
            policy,
            top,
            max_nodes,
            from,
        } => cmd_search(policy, *top, *max_nodes, from),
        Cmd::Force {
            target,
            max_nodes,
            from,
            emit,
        } => cmd_force(target, *max_nodes, from, emit),
        Cmd::Enum {
            card,
            rows,
            max_nodes,
            from,
        } => cmd_enum(*card, *rows, *max_nodes, from),
        Cmd::Sweep {
            card,
            rows,
            sources_file,
            target,
            targets_file,
            max_nodes,
            out,
        } => cmd_sweep(
            *card,
            *rows,
            sources_file,
            target,
            targets_file,
            *max_nodes,
            out,
        ),
        Cmd::Shrink {
            rows,
            max_nodes,
            from,
            emit,
        } => cmd_shrink(*rows, *max_nodes, from, emit),
        Cmd::Gfp {
            sample,
            rows,
            max_holes,
            max_nodes,
            out,
        } => cmd_gfp(*rows, *max_holes, *max_nodes, *sample, out),
        Cmd::ClosedGfp {
            rows,
            max_holes,
            lazy,
            max_boards,
            out,
            emit,
            fixpoint,
            top_k,
        } => cmd_closed_gfp(
            *rows,
            *max_holes,
            *lazy,
            *max_boards,
            out,
            emit,
            *fixpoint,
            *top_k,
        ),
        Cmd::TableStats { table } => cmd_table_stats(table),
        Cmd::Universe {
            rows,
            max_holes,
            out,
        } => {
            let universe = enum_universe(*rows, *max_holes);
            let text: String = universe
                .iter()
                .map(|b| format!("{}\n", to_hex(b)))
                .collect();
            std::fs::write(out, text).with_context(|| format!("writing {out}"))?;
            println!(
                "wrote {} boards (rows≤{rows}, holes≤{max_holes}) to {out}",
                universe.len()
            );
            Ok(())
        }
        Cmd::Core {
            family,
            rows,
            seed,
            max_nodes,
            out,
            emit,
            ckpt,
            resume,
        } => cmd_core(family, *rows, seed, *max_nodes, out, emit, ckpt, *resume),
        Cmd::LeanEmit { core, table, out } => cmd_lean_emit(core, table, out),
        Cmd::LeanEmitClosed {
            core,
            table,
            out,
            reach_max_nodes,
        } => cmd_lean_emit_closed(core, table, out, *reach_max_nodes),
        Cmd::Orbit {
            weights,
            max_h,
            cap,
            out,
            emit,
        } => cmd_orbit(weights, *max_h, *cap, out, emit),
        Cmd::OrbitSweep {
            candidates,
            seed,
            max_h,
            cap,
        } => cmd_orbit_sweep(*candidates, *seed, *max_h, *cap),
        Cmd::Solve {
            weights,
            max_h,
            cap,
            out,
            emit,
        } => cmd_solve(weights, *max_h, *cap, out, emit),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_are_bottom_aligned_quads() {
        for &p in &PIECES {
            for rot in 0..4u8 {
                let s = shape_up(p, rot);
                let min_dr = s.iter().map(|c| c.1).min().unwrap_or(99);
                assert_eq!(min_dr, 0, "{p:?} rot {rot} not bottom-aligned");
                let distinct: FxHashSet<(u32, u32)> = s.iter().copied().collect();
                assert_eq!(distinct.len(), 4, "{p:?} rot {rot} has duplicate cells");
            }
        }
    }

    #[test]
    fn apply_step_drops_and_clears() {
        // Vertical I in column 0 of the empty board: cells (0,0..3).
        let b = apply_step(&EMPTY, Piece::I, 1, 0);
        assert_eq!(b[0], 0b1111);
        assert_eq!(card(&b), 4);

        // Fill row 0 with horizontal I at cols 0..3 and 4..7, then O at 8:
        // 10 cells in row 0 plus 2 in row 1 → row 0 clears, O remnant drops.
        let b1 = apply_step(&EMPTY, Piece::I, 0, 0);
        let b2 = apply_step(&b1, Piece::I, 0, 4);
        assert_eq!(card(&b2), 8);
        let b3 = apply_step(&b2, Piece::O, 0, 8);
        // Before clear: row 0 full except nothing — cols 8,9 get rows 0,1;
        // row 0 is full → cleared; remaining: cols 8,9 at row 0.
        assert_eq!(card(&b3), 2);
        assert_eq!(b3[8], 0b1);
        assert_eq!(b3[9], 0b1);
    }

    #[test]
    fn truncated_drop_offset() {
        // S at rot 0 on a flat-left board: shapeUp = {(0,0),(1,0),(1,1),(2,1)}.
        // On empty board all heights 0 → offset 0; cells rows (0,0,1,1).
        let b = apply_step(&EMPTY, Piece::S, 0, 0);
        assert_eq!(b[0], 0b01);
        assert_eq!(b[1], 0b11);
        assert_eq!(b[2], 0b10);
    }

    #[test]
    fn bag_draw_refills() {
        let mut bag = FULL_BAG;
        for &p in &PIECES[..6] {
            bag = draw(bag, p);
            assert_ne!(bag, FULL_BAG);
        }
        // Seventh draw empties → refills.
        bag = draw(bag, Piece::J);
        assert_eq!(bag, FULL_BAG);
    }

    #[test]
    fn s_z_never_complete_perfect_clear() {
        // Geometric obstruction behind the ∅-forcing refutations: a final S or
        // Z placement can never leave the board empty (the full-row condition
        // forces a pre-existing cell directly above the piece's bottom cell in
        // some column, which straight-drop gravity cannot tuck under).
        // Exhaustive over every flat board that could even arithmetically PC
        // with one piece (card + 4 ≡ 0 mod 10, within 2-3 rows).
        let pools = [(6u32, 2u32), (16, 2), (26, 3)];
        for (c, rows) in pools {
            for b in enum_flat_boards(c, rows) {
                for p in [Piece::S, Piece::Z] {
                    for &(rot, col) in &SHAPES.valid[p as usize] {
                        let nb = apply_step(&b, p, rot, col);
                        assert_ne!(card(&nb), 0, "PC by {p:?} from {}", to_hex(&b));
                    }
                }
            }
        }
        // Positive control: O and I genuinely can complete perfect clears.
        let mut o_notch = EMPTY;
        for w in o_notch.iter_mut().take(8) {
            *w = 0b11;
        }
        assert_eq!(card(&apply_step(&o_notch, Piece::O, 0, 8)), 0);
        let mut i_notch = EMPTY;
        for w in i_notch.iter_mut().skip(1) {
            *w = 0b1111;
        }
        assert_eq!(card(&apply_step(&i_notch, Piece::I, 1, 0)), 0);
    }

    #[test]
    fn enum_universe_counts() {
        // Hole-free flat boards are solid column prefixes: heights in
        // {0..rows}^10. No full row ⟺ some column empty, so the count is
        // (rows+1)^10 − rows^10.
        assert_eq!(enum_universe(1, 0).len(), 1024 - 1);
        assert_eq!(enum_universe(2, 0).len(), 59049 - 1024);
        assert_eq!(enum_universe(3, 0).len(), 1048576 - 59049);
        // rows=2 unbounded holes: all 2^20 masks minus those containing a
        // full row (row 0 full or row 1 full: 2·2^10 − 1 masks).
        assert_eq!(enum_universe(2, 20).len(), (1 << 20) - 2047);
        // Universe members are within bounds and hole-bounded.
        for b in enum_universe(2, 1) {
            assert!(b.iter().all(|&w| w >> 2 == 0));
            assert!(holes(&b) <= 1);
            assert_eq!(b.iter().fold(!0u64, |acc, &w| acc & w), 0);
        }
    }

    #[test]
    fn fanout_col0_matches_lean() {
        assert_eq!(fan_out(PolicyKind::Col0, &EMPTY), 4980);
    }

    #[test]
    fn fanout_canon_matches_lean() {
        assert_eq!(fan_out(PolicyKind::Canon, &EMPTY), 3145);
    }

    #[test]
    fn fanout_clear_matches_lean() {
        assert_eq!(fan_out(PolicyKind::Clear, &EMPTY), 1892);
    }

    #[test]
    fn features_identities() {
        // agg = card + holes on any board; steps/bump/wells consistent on a
        // hand-built profile: heights [2,1,0,0,0,0,0,0,0,3].
        let mut b = EMPTY;
        b[0] = 0b11;
        b[1] = 0b1;
        b[9] = 0b111;
        let f = features(&b, 0);
        assert_eq!(f[2], f[0] + f[1]); // agg = card + holes
        assert_eq!(f[0], 6); // card
        assert_eq!(f[1], 0); // holes
        assert_eq!(f[3], 1 + 1 + 3); // bump
        assert_eq!(f[4], 3); // maxh
        assert_eq!(f[6], 0); // wells (no column strictly below both neighbors)
        assert_eq!(f[7], 2); // steps: |2-1|=1, |1-0|=1
        // A central well of depth 2: heights [2,0,2,...].
        let mut wb = EMPTY;
        wb[0] = 0b11;
        wb[2] = 0b11;
        assert_eq!(features(&wb, 0)[6], 2);
    }

    #[test]
    fn weighted_step_survives_and_canonicalizes() {
        // canon weights (all zero): pure tiebreak picks the row-major-least
        // successor, matching PolicyKind::Canon on the empty board where no
        // placement is lost.
        let w = [0i64; NFEAT];
        for &p in &PIECES {
            let (rot, col, nb) = weighted_step(&w, &EMPTY, p).unwrap();
            assert_eq!((rot, col), policy_step(PolicyKind::Canon, &EMPTY, p));
            assert_eq!(nb, apply_step(&EMPTY, p, rot, col));
        }
        // Near-top board: column 0 at height 19 — vertical I there is lost
        // and must be filtered, but survivable placements remain.
        let mut tall = EMPTY;
        tall[0] = (1u64 << 19) - 1;
        let stepped = weighted_step(&w, &tall, Piece::I);
        let (_, _, nb) = stepped.unwrap();
        assert!(!is_lost(&nb));
    }

    #[test]
    fn orbit_replay_consistency() {
        // Whatever the outcome, a closed orbit's rows must replay from ∅ to
        // exactly the boundary set (the invariant lean-emit's live_set
        // relies on). Use a tiny cap so the test stays fast even when the
        // clear policy sprawls; only verify on closure.
        let w = parse_weights("clear").unwrap();
        if let OrbitOutcome::Closed {
            boundary,
            states,
            rows,
        } = policy_orbit(&w, 8, 50_000)
        {
            let table: StrategyTable = rows.iter().copied().collect();
            assert_eq!(table.len(), rows.len(), "duplicate (state, piece) rows");
            let mut seen: FxHashSet<(Board, Bag)> = FxHashSet::default();
            let mut reached: FxHashSet<Board> = FxHashSet::default();
            let mut stack = vec![(EMPTY, FULL_BAG)];
            seen.insert((EMPTY, FULL_BAG));
            reached.insert(EMPTY);
            while let Some((b, bag)) = stack.pop() {
                for p in PIECES {
                    if !bag_contains(bag, p) {
                        continue;
                    }
                    let &(rot, col) = table.get(&(b, bag, p)).expect("missing row");
                    let nb = apply_step(&b, p, rot, col);
                    assert!(!is_lost(&nb));
                    let nbag = draw(bag, p);
                    if seen.insert((nb, nbag)) {
                        if nbag == FULL_BAG {
                            reached.insert(nb);
                        }
                        stack.push((nb, nbag));
                    }
                }
            }
            assert_eq!(seen.len(), states);
            let bd: FxHashSet<Board> = boundary.into_iter().collect();
            assert_eq!(reached, bd);
        }
    }
}
