#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! tetris_eigen — the homing-carrier / eigen-cycle existence probe.
//!
//! # Why this exists
//!
//! The tropical theory (`proofs/Proofs/Experiments/TopicalTetris.lean`) reduced adversarial
//! survival to ONE object: a recurrent shape the player can return to, whose existence — via
//! `eigen_global_roughness` — bounds the roughness of EVERY trajectory and hence proves
//! survival. A cell-counting fact sharpens the target: a 7-bag adds 28 cells = 2.8 lines, never
//! an integer clear, so no shape is fixed after a single bag; the recurrent object is an
//! eigen-CYCLE of period ≥ 5 bags (`IsEigenCycle`). The theory also says the RIGHT player
//! contracts the oscillation (roughness) — the dimension every prior probe found to be the sole
//! explosion axis.
//!
//! This binary tests that directly. It computes the **adversarial reachable closure** from the
//! empty board under a **roughness-homing, debt-bounded player** vs the full 7-bag adversary
//! (every piece of the remaining bag is an AND-branch the player must answer), inside a band
//! `(roughness ≤ R, holes ≤ D, height ≤ H)`. Three outcomes:
//!
//!   * **CLOSED(size)** — the worklist empties with every reachable state in-band and dead-end
//!     free ⇒ a finite closed carrier + a concrete strategy ⇒ SURVIVAL PROOF for the band, and
//!     (being finite + closed under a total adversary) it contains a recurrent eigen-cycle. If
//!     small, Lean-importable via `tetrisSolvableValid_of_maxHeight_invariant`.
//!   * **DIED(board)** — the homing player hits a state where the revealed piece has no in-band
//!     admissible placement. SUFFICIENT-test failure (greedy/depth-d cornered itself); a witness,
//!     not a floor.
//!   * **EXPLODED** — closure exceeds the cap ⇒ no small carrier at this band.
//!
//! Sweeping R upward finds the smallest certifiable roughness band, or floors the route with a
//! number. Debt budget defaults to D=1 (the proven `K=1`); strict hole-free is D=0.
//!
//! Run:
//!   cargo run --release -p tetris-playground --bin tetris_eigen -- sweep d1 depth1
//!   cargo run --release -p tetris-playground --bin tetris_eigen -- r6 d1 depth4 cap5

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet};
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};

/// Canonical loss line.
const ROWS: u32 = 20;
/// Value sentinel: a forced top-out / no admissible response (higher = worse for the player).
const LOSS: i64 = 1 << 60;

/// The 7 pieces in a fixed index order; bit `i` of a bag mask ↔ `PIECES[i]`.
const PIECES: [TetrisPiece; 7] = [
    TetrisPiece::O_PIECE,
    TetrisPiece::I_PIECE,
    TetrisPiece::S_PIECE,
    TetrisPiece::Z_PIECE,
    TetrisPiece::T_PIECE,
    TetrisPiece::L_PIECE,
    TetrisPiece::J_PIECE,
];
const FULL_MASK: u8 = 0b111_1111;

/// A certification band: caps the player is required to stay within.
#[derive(Clone, Copy)]
struct Band {
    /// **Osc** cap = max−min column height (the swept knob; the Lean theory's `roughness`).
    /// Note osc ≤ height, so `r = 20` coincides with "height-only" (no extra osc constraint).
    r: u32,
    /// Hole / debt budget (`D = 1` is the proven `K = 1`).
    d: u32,
    /// Height cap (the real loss line unless tightened).
    h: u32,
}

/// **Osc** (the theory's `roughness`): max column height − min column height, over all 10 columns.
/// This is `eigen_global_roughness`'s metric — bounded ⇒ bounded shape ⇒ survival. (Distinct from
/// the engine's `roughness()`, which is bumpiness Σ|Δ| and ranges far higher.)
fn osc(b: &TetrisBoard) -> u32 {
    let h = b.heights();
    let mx = h.iter().copied().max().unwrap_or(0);
    let mn = h.iter().copied().min().unwrap_or(0);
    mx - mn
}

/// When set, the leaf value prepends a **robustness** term: the number of next pieces this board
/// CANNOT answer in-band, weighted to dominate everything. This is the 1-ply embodiment of "keep
/// a board that absorbs every adversary piece" — the AND-structure that pure roughness misses.
static ROBUST: AtomicBool = AtomicBool::new(false);

/// **Leaf value** (lower = better). Base: minimize HEIGHT (the loss line) first, then OSC (the
/// theory's contraction Lyapunov), then bumpiness (maneuverability for S/Z notches), then debt,
/// then mass. With `ROBUST`, a board that strands any next piece is penalized above all else.
fn leaf_cost(b: &TetrisBoard, band: Band) -> i64 {
    let maxh = b.height() as i64;
    let o = osc(b) as i64;
    let bump = b.roughness() as i64;
    let holes = b.total_holes() as i64;
    let agg = b.heights().iter().sum::<u32>() as i64;
    let base = maxh * 1_000_000_000 + o * 1_000_000 + bump * 1_000 + holes * 10 + agg;
    if ROBUST.load(Ordering::Relaxed) {
        let unanswerable = PIECES
            .iter()
            .filter(|&&p| admissible(b, p, band).is_empty())
            .count() as i64;
        unanswerable * 1_000_000_000_000_000 + base
    } else {
        base
    }
}

/// All in-band landings of `piece` on `board`: not lost, height ≤ `h`, holes ≤ `d`, roughness ≤ `r`.
fn admissible(board: &TetrisBoard, piece: TetrisPiece, band: Band) -> Vec<TetrisBoard> {
    let mut out = Vec::new();
    for &pl in TetrisPiecePlacement::all_from_piece(piece) {
        let mut nb = *board;
        let res = nb.apply_piece_placement(pl);
        if res.is_lost == IsLost::LOST {
            continue;
        }
        if nb.height() > band.h || nb.total_holes() > band.d || osc(&nb) > band.r {
            continue;
        }
        out.push(nb);
    }
    out
}

/// Depth-limited homing minimax: player MIN over in-band landings, adversary MAX over the
/// remaining bag. `LOSS` if the adversary can force out-of-band within the horizon. Memoized.
fn homing_minimax(
    board: TetrisBoard,
    mask: u8,
    depth: u8,
    band: Band,
    memo: &mut FxHashMap<(TetrisBoard, u8, u8), i64>,
) -> i64 {
    if mask == 0 || depth == 0 {
        return leaf_cost(&board, band);
    }
    if let Some(&v) = memo.get(&(board, mask, depth)) {
        return v;
    }
    let mut best_adv = i64::MIN;
    for (pi, p) in PIECES.iter().enumerate() {
        let bit = 1u8 << pi;
        if mask & bit == 0 {
            continue;
        }
        let mut best_play = LOSS;
        for nb in admissible(&board, *p, band) {
            let v = homing_minimax(nb, mask & !bit, depth - 1, band, memo);
            if v < best_play {
                best_play = v;
            }
        }
        if best_play > best_adv {
            best_adv = best_play; // an empty admissible set leaves LOSS — adversary takes it
        }
    }
    memo.insert((board, mask, depth), best_adv);
    best_adv
}

/// The player's deterministic homing choice for a revealed `piece`, given the bag remaining
/// AFTER it (`rem_mask`). Returns the chosen landing, or `None` if no in-band placement exists.
fn choose(
    board: &TetrisBoard,
    piece: TetrisPiece,
    rem_mask: u8,
    depth: u8,
    band: Band,
) -> Option<TetrisBoard> {
    let cands = admissible(board, piece, band);
    if cands.is_empty() {
        return None;
    }
    if depth <= 1 {
        return cands.into_iter().min_by_key(|nb| leaf_cost(nb, band));
    }
    let mut memo: FxHashMap<(TetrisBoard, u8, u8), i64> = FxHashMap::default();
    let mut best: Option<(i64, TetrisBoard)> = None;
    for nb in cands {
        let v = homing_minimax(nb, rem_mask, depth - 1, band, &mut memo);
        if best.as_ref().map_or(true, |(bv, _)| v < *bv) {
            best = Some((v, nb));
        }
    }
    best.map(|(_, b)| b)
}

/// The verdict for one band. `mo` = max osc, `mb` = max bumpiness, `mh` = max height, `md` = max debt.
enum Outcome {
    Closed {
        size: usize,
        mo: u32,
        mb: u32,
        mh: u32,
        md: u32,
    },
    Exploded {
        size: usize,
        mo: u32,
        mb: u32,
        mh: u32,
        md: u32,
    },
    Died {
        mask: u8,
        piece: TetrisPiece,
        size: usize,
        mo: u32,
        mb: u32,
        mh: u32,
        md: u32,
    },
}

/// Compute the adversarial reachable closure from the empty board under the depth-`depth` homing
/// player, inside `band`, capped at `cap` states.
fn closure(band: Band, depth: u8, cap: usize) -> Outcome {
    let root = TetrisBoard::new();
    let mut seen: FxHashSet<(TetrisBoard, u8)> = FxHashSet::default();
    let mut stack: Vec<(TetrisBoard, u8)> = vec![(root, FULL_MASK)];
    seen.insert((root, FULL_MASK));
    let (mut mo, mut mb, mut mh, mut md) = (0u32, 0u32, 0u32, 0u32);

    while let Some((b, mask)) = stack.pop() {
        for (pi, p) in PIECES.iter().enumerate() {
            let bit = 1u8 << pi;
            if mask & bit == 0 {
                continue;
            }
            let rem = mask & !bit;
            match choose(&b, *p, if rem == 0 { FULL_MASK } else { rem }, depth, band) {
                None => {
                    return Outcome::Died {
                        mask,
                        piece: *p,
                        size: seen.len(),
                        mo,
                        mb,
                        mh,
                        md,
                    };
                }
                Some(nb) => {
                    mo = mo.max(osc(&nb));
                    mb = mb.max(nb.roughness());
                    mh = mh.max(nb.height());
                    md = md.max(nb.total_holes());
                    let nmask = if rem == 0 { FULL_MASK } else { rem };
                    if seen.insert((nb, nmask)) {
                        if seen.len() > cap {
                            return Outcome::Exploded {
                                size: seen.len(),
                                mo,
                                mb,
                                mh,
                                md,
                            };
                        }
                        stack.push((nb, nmask));
                    }
                }
            }
        }
    }
    Outcome::Closed {
        size: seen.len(),
        mo,
        mb,
        mh,
        md,
    }
}

/// Print one band's outcome line.
fn report(band: Band, depth: u8, cap: usize) -> bool {
    let t0 = Instant::now();
    let out = closure(band, depth, cap);
    let dt = t0.elapsed().as_secs_f64();
    let closed = matches!(out, Outcome::Closed { .. });
    match out {
        Outcome::Closed {
            size,
            mo,
            mb,
            mh,
            md,
        } => println!(
            "  osc≤{:<2} d{} depth{}: CLOSED  |Σ|={:>9}  (max osc={mo} bump={mb} h={mh} holes={md})  [{dt:.1}s]  \
             *** finite closed carrier — survival proof for this band ***",
            band.r, band.d, depth, size
        ),
        Outcome::Exploded {
            size,
            mo,
            mb,
            mh,
            md,
        } => println!(
            "  osc≤{:<2} d{} depth{}: EXPLODED >|Σ|={:>9}  (max osc={mo} bump={mb} h={mh} holes={md})  [{dt:.1}s]",
            band.r, band.d, depth, size
        ),
        Outcome::Died {
            mask,
            piece,
            size,
            mo,
            mb,
            mh,
            md,
        } => println!(
            "  osc≤{:<2} d{} depth{}: DIED at |Σ|={:>9} (piece {:?}, bag-rem {:#05b}; max osc={mo} bump={mb} h={mh} holes={md})  [{dt:.1}s]",
            band.r, band.d, depth, size, piece, mask
        ),
    }
    closed
}

/// Sweep the roughness cap upward for a fixed debt budget + depth.
fn sweep(d: u32, depth: u8, cap: usize) {
    println!(
        "MODE: roughness-band sweep — homing player (depth {depth}, debt≤{d}, height≤{ROWS}) vs 7-bag adversary"
    );
    println!(
        "(CLOSED = finite carrier ⇒ survival proof for the band; smallest such R = minimal certifiable band)\n"
    );
    for r in [0u32, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20] {
        let closed = report(Band { r, d, h: ROWS }, depth, cap);
        if closed {
            println!(
                "\n=== minimal certifiable roughness band found: R = {r} (debt≤{d}, depth {depth}) ===\n\
                 Every reachable (board,bag) under the homing player stays within rough≤{r}, holes≤{d}, \
                 height≤{ROWS}, with an answer to every adversary piece. This is a finite closed carrier."
            );
            return;
        }
    }
    println!(
        "\n=== no band CLOSED up to R=20 at depth {depth}, debt≤{d} ===\n\
         The depth-{depth} homing player cannot close a finite carrier in any roughness band. Either \
         deepen lookahead (depth) / widen debt (d), or this floors the homing route with numbers."
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let parse = |pfx: &str| -> Option<u32> {
        args.iter()
            .find_map(|a| a.strip_prefix(pfx).and_then(|s| s.parse::<u32>().ok()))
    };
    let d = parse("d").unwrap_or(1);
    let depth = parse("depth").unwrap_or(1) as u8;
    let cap = parse("cap").map_or(2_000_000usize, |m| m as usize * 1_000_000);
    if args.iter().any(|a| a == "robust") {
        ROBUST.store(true, Ordering::Relaxed);
    }

    println!(
        "tetris_eigen — homing-carrier / eigen-cycle existence probe (cap={} states, player={})\n",
        cap,
        if ROBUST.load(Ordering::Relaxed) {
            "robust (1-ply safe-set greedy)"
        } else {
            "roughness-homing"
        }
    );

    if let Some(r) = parse("r") {
        report(Band { r, d, h: ROWS }, depth, cap);
    } else {
        sweep(d, depth, cap);
    }
}
