#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! tetris_preview — does FULL-BAG PREVIEW beat the order-switching adversary?
//!
//! # Why this exists
//!
//! `tetris_policy` showed a no-preview robust policy LEAKS: the switching adversary denies clears and
//! pumps height to the ceiling. But real Tetris shows the upcoming pieces. `tetris_tropical` showed
//! that with the bag KNOWN, a beam player survives 100% of fixed orders at height ≤ 7. This binary
//! gives the player exactly that — **full within-bag preview** (it sees the whole current bag and
//! plans it with beam search) — but keeps the adversary's power to choose a DIFFERENT order each bag
//! and hide the NEXT bag.
//!
//! It computes the reachable closure over bag-boundary boards: from each board, the adversary may
//! play any of the 5040 orders, the player answers each with full-preview beam play.
//!   * **CLOSED(size)** — finite, no order ever tops the player from any reachable board ⇒ full-bag
//!     preview SURVIVES the switching adversary forever ⇒ Tetris solved (boundary carrier = closure).
//!   * **LEAK(board,order)** — some order tops the player from some reachable board.
//!   * **EXPLODED** — the boundary carrier exceeds the cap.
//!
//! Run:
//!   cargo run --release -p tetris-playground --bin tetris_preview -- width6 cap300

use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::FxHashSet;
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};

const ROWS: u32 = 20;

const PIECES: [TetrisPiece; 7] = [
    TetrisPiece::O_PIECE,
    TetrisPiece::I_PIECE,
    TetrisPiece::S_PIECE,
    TetrisPiece::Z_PIECE,
    TetrisPiece::T_PIECE,
    TetrisPiece::L_PIECE,
    TetrisPiece::J_PIECE,
];
const NAMES: [&str; 7] = ["O", "I", "S", "Z", "T", "L", "J"];

/// When set, the eval adds the proven lever from PieceCharge.level_clear_capacity: reward a high
/// MINIMUM column height (clearing capacity = min height), i.e. penalize deep wells.
static MIN_LEVER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Survival eval (lower=better): hole-averse, then low max-height, then flat, then light. With the
/// `minlever` flag, also reward a high minimum column height (the clearing-capacity lever).
fn eval(b: &TetrisBoard) -> i64 {
    let h = b.heights();
    let base = (b.total_holes() as i64) * 1_000_000_000
        + (b.height() as i64) * 1_000_000
        + (b.roughness() as i64) * 1_000
        + h.iter().sum::<u32>() as i64;
    if MIN_LEVER.load(std::sync::atomic::Ordering::Relaxed) {
        let minh = h.iter().copied().min().unwrap_or(0) as i64;
        base - minh * 100_000 // reward high min height (more clearing capacity)
    } else {
        base
    }
}

/// Play one full bag (the 7 pieces of `order`, all known = full preview) from `board` with a
/// width-`width` beam. Returns the best surviving end-of-bag board, or `None` if forced to top out.
fn play_bag_beam(
    board: TetrisBoard,
    order: &[TetrisPiece; 7],
    width: usize,
) -> Option<TetrisBoard> {
    let mut beam = vec![board];
    for &p in order.iter() {
        let mut next: Vec<TetrisBoard> = Vec::with_capacity(beam.len() * 8);
        for b in &beam {
            for &pl in TetrisPiecePlacement::all_from_piece(p) {
                let mut nb = *b;
                let res = nb.apply_piece_placement(pl);
                if res.is_lost == IsLost::LOST || nb.height() > ROWS {
                    continue;
                }
                next.push(nb);
            }
        }
        if next.is_empty() {
            return None;
        }
        if next.len() > width {
            next.select_nth_unstable_by_key(width, eval);
            next.truncate(width);
        }
        beam = next;
    }
    beam.into_iter().min_by_key(|b| eval(b))
}

/// All 5040 bag orders (permutations of the 7 pieces).
fn all_orders() -> Vec<[TetrisPiece; 7]> {
    let mut idx = [0usize, 1, 2, 3, 4, 5, 6];
    let mut out = Vec::with_capacity(5040);
    loop {
        out.push(std::array::from_fn(|i| PIECES[idx[i]]));
        if !next_perm(&mut idx) {
            break;
        }
    }
    out
}

fn next_perm(a: &mut [usize; 7]) -> bool {
    let n = a.len();
    let mut i = n - 1;
    while i > 0 && a[i - 1] >= a[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let mut j = n - 1;
    while a[j] <= a[i - 1] {
        j -= 1;
    }
    a.swap(i - 1, j);
    a[i..].reverse();
    true
}

fn order_name(order: &[TetrisPiece; 7]) -> String {
    order
        .iter()
        .map(|p| {
            let k = PIECES.iter().position(|q| q == p).unwrap();
            NAMES[k]
        })
        .collect::<Vec<_>>()
        .join("")
}

enum Outcome {
    Closed {
        size: usize,
        mh: u32,
        md: u32,
    },
    Exploded {
        size: usize,
        mh: u32,
        md: u32,
    },
    Leak {
        order: [TetrisPiece; 7],
        board: TetrisBoard,
        size: usize,
    },
}

fn closure(width: usize, cap: usize) -> Outcome {
    let orders = all_orders();
    let root = TetrisBoard::new();
    let mut seen: FxHashSet<TetrisBoard> = FxHashSet::default();
    let mut stack = vec![root];
    seen.insert(root);
    let (mut mh, mut md) = (0u32, 0u32);

    while let Some(b) = stack.pop() {
        // Expand all 5040 orders in parallel; each yields a next boundary board or a top-out.
        let results: Vec<Option<TetrisBoard>> = orders
            .par_iter()
            .map(|o| play_bag_beam(b, o, width))
            .collect();
        for (i, r) in results.into_iter().enumerate() {
            match r {
                None => {
                    return Outcome::Leak {
                        order: orders[i],
                        board: b,
                        size: seen.len(),
                    };
                }
                Some(nb) => {
                    mh = mh.max(nb.height());
                    md = md.max(nb.total_holes());
                    if seen.insert(nb) {
                        if seen.len() > cap {
                            return Outcome::Exploded {
                                size: seen.len(),
                                mh,
                                md,
                            };
                        }
                        stack.push(nb);
                    }
                }
            }
        }
    }
    Outcome::Closed {
        size: seen.len(),
        mh,
        md,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let parse = |pfx: &str| -> Option<usize> {
        args.iter()
            .find_map(|a| a.strip_prefix(pfx).and_then(|s| s.parse::<usize>().ok()))
    };
    let width = parse("width").unwrap_or(6);
    let cap = parse("cap").map_or(300_000usize, |m| m * 1_000);
    if args.iter().any(|a| a == "minlever") {
        MIN_LEVER.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    println!(
        "tetris_preview — full-bag-preview beam (width {width}) vs the ORDER-SWITCHING adversary{}",
        if MIN_LEVER.load(std::sync::atomic::Ordering::Relaxed) {
            "  [+min-height lever]"
        } else {
            ""
        }
    );
    println!(
        "(CLOSED = preview survives switching forever ⇒ Tetris solved; LEAK/EXPLODED otherwise)\n"
    );

    let t0 = Instant::now();
    let out = closure(width, cap);
    let dt = t0.elapsed().as_secs_f64();
    match out {
        Outcome::Closed { size, mh, md } => println!(
            "CLOSED  |boundary carrier|={size}  (max height={mh}, max debt={md})  [{dt:.1}s]\n\
             *** no bag order tops the full-preview player from any reachable board ***\n\
             *** finite closed boundary carrier vs the SWITCHING adversary ⇒ Tetris SOLVED with full-bag preview ***"
        ),
        Outcome::Exploded { size, mh, md } => println!(
            "EXPLODED  |boundary carrier|>{size}  (max height={mh}, max debt={md})  [{dt:.1}s]\n\
             the full-preview player survives but the boundary carrier exceeds the cap — bounded play, large carrier."
        ),
        Outcome::Leak { order, board, size } => println!(
            "LEAK  order {} tops out the full-preview player from a reachable board\n\
             (reached |boundary carrier|={size} boards, heights {:?})  [{dt:.1}s]\n\
             ⇒ even full-bag preview can be driven to a board where some next order kills.",
            order_name(&order),
            board.heights()
        ),
    }
}
