#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! tetris_policy — policy evaluation for the ORDER-SWITCHING adversary (the real mean-payoff game).
//!
//! # Why this exists
//!
//! `tetris_tropical` showed every FIXED periodic bag order is comfortably survivable (100%, cycle
//! height ≤ 7) — so all residual difficulty is in the adversary's freedom to pick a DIFFERENT order
//! each bag and exploit the transitions. That is the real game. Unlike the fixed-order beam (which
//! legitimately used known future), here the player must be robust to ANY future: a depth-`d`
//! bag-minimax (adversary MAXimizes over the remaining bag, player MINimizes), the survival eval.
//!
//! This is policy-iteration step 0: take the strongest simple robust policy and EVALUATE it under
//! the switching adversary by computing the reachable closure over `(board, bag_mask)` from empty:
//!   * **CLOSED(size)** — finite, every state answers every adversary piece in-budget ⇒ that policy
//!     SURVIVES the switching adversary forever ⇒ Tetris solved (carrier = the closure).
//!   * **LEAK(board,piece)** — the policy is stranded somewhere ⇒ improve it there (the next PI step).
//!   * **EXPLODED** — the policy's carrier exceeds the cap ⇒ no small carrier for this policy.
//!
//! Run:
//!   cargo run --release -p tetris-playground --bin tetris_policy -- d1 depth4 cap2

use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet};
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};

const ROWS: u32 = 20;
const LOSS: i64 = 1 << 60;

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
const FULL_MASK: u8 = 0b111_1111;

/// Debt budget (holes allowed) — the proven `K=1` by default.
static DEBT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

/// Survival eval (lower=better): hole-averse, then low max-height, then flat, then light.
fn eval(b: &TetrisBoard) -> i64 {
    (b.total_holes() as i64) * 1_000_000_000
        + (b.height() as i64) * 1_000_000
        + (b.roughness() as i64) * 1_000
        + b.heights().iter().sum::<u32>() as i64
}

/// In-budget landings of `piece` on `board`: not lost, height ≤ ROWS, holes ≤ DEBT.
fn admissible(board: &TetrisBoard, piece: TetrisPiece) -> Vec<TetrisBoard> {
    let d = DEBT.load(std::sync::atomic::Ordering::Relaxed);
    let mut out = Vec::new();
    for &pl in TetrisPiecePlacement::all_from_piece(piece) {
        let mut nb = *board;
        let res = nb.apply_piece_placement(pl);
        if res.is_lost == IsLost::LOST || nb.height() > ROWS || nb.total_holes() > d {
            continue;
        }
        out.push(nb);
    }
    out
}

/// Depth-limited bag-minimax: adversary MAX over remaining bag, player MIN over in-budget landings.
/// `LOSS` if the adversary can force out-of-budget within the horizon. Memoized.
fn bag_minimax(
    board: TetrisBoard,
    mask: u8,
    depth: u8,
    memo: &mut FxHashMap<(TetrisBoard, u8, u8), i64>,
) -> i64 {
    if mask == 0 || depth == 0 {
        return eval(&board);
    }
    if let Some(&v) = memo.get(&(board, mask, depth)) {
        return v;
    }
    let mut worst = i64::MIN;
    for (pi, p) in PIECES.iter().enumerate() {
        let bit = 1u8 << pi;
        if mask & bit == 0 {
            continue;
        }
        let mut best = LOSS;
        for nb in admissible(&board, *p) {
            let v = bag_minimax(nb, mask & !bit, depth - 1, memo);
            if v < best {
                best = v;
            }
        }
        if best > worst {
            worst = best;
        }
    }
    memo.insert((board, mask, depth), worst);
    worst
}

/// The robust policy's placement for revealed `piece`, with the bag remaining AFTER it (`rem`).
fn policy(board: &TetrisBoard, piece: TetrisPiece, rem: u8, depth: u8) -> Option<TetrisBoard> {
    let cands = admissible(board, piece);
    if cands.is_empty() {
        return None;
    }
    if depth <= 1 {
        return cands.into_iter().min_by_key(eval_key);
    }
    let mut memo: FxHashMap<(TetrisBoard, u8, u8), i64> = FxHashMap::default();
    let mut best: Option<(i64, TetrisBoard)> = None;
    for nb in cands {
        let v = bag_minimax(nb, rem, depth - 1, &mut memo);
        if best.as_ref().is_none_or(|(bv, _)| v < *bv) {
            best = Some((v, nb));
        }
    }
    best.map(|(_, b)| b)
}

fn eval_key(b: &TetrisBoard) -> i64 {
    eval(b)
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
        piece: TetrisPiece,
        mask: u8,
        size: usize,
        mh: u32,
        md: u32,
    },
}

/// Reachable closure over `(board, bag_mask)` from empty, under the depth-`depth` robust policy vs
/// the switching adversary (every bag piece is an AND-branch).
fn closure(depth: u8, cap: usize) -> Outcome {
    let root = TetrisBoard::new();
    let mut seen: FxHashSet<(TetrisBoard, u8)> = FxHashSet::default();
    let mut stack = vec![(root, FULL_MASK)];
    seen.insert((root, FULL_MASK));
    let (mut mh, mut md) = (0u32, 0u32);

    while let Some((b, mask)) = stack.pop() {
        for (pi, p) in PIECES.iter().enumerate() {
            let bit = 1u8 << pi;
            if mask & bit == 0 {
                continue;
            }
            let rem = mask & !bit;
            let rem_full = if rem == 0 { FULL_MASK } else { rem };
            match policy(&b, *p, rem_full, depth) {
                None => {
                    return Outcome::Leak {
                        piece: *p,
                        mask,
                        size: seen.len(),
                        mh,
                        md,
                    };
                }
                Some(nb) => {
                    mh = mh.max(nb.height());
                    md = md.max(nb.total_holes());
                    if seen.insert((nb, rem_full)) {
                        if seen.len() > cap {
                            return Outcome::Exploded {
                                size: seen.len(),
                                mh,
                                md,
                            };
                        }
                        stack.push((nb, rem_full));
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
    let parse = |pfx: &str| -> Option<u32> {
        args.iter()
            .find_map(|a| a.strip_prefix(pfx).and_then(|s| s.parse::<u32>().ok()))
    };
    let d = parse("d").unwrap_or(1);
    let depth = parse("depth").unwrap_or(4) as u8;
    let cap = parse("cap").map_or(2_000_000usize, |m| m as usize * 1_000_000);
    DEBT.store(d, std::sync::atomic::Ordering::Relaxed);

    println!(
        "tetris_policy — switching-adversary policy evaluation (robust depth-{depth} minimax, debt≤{d}, cap={cap})"
    );
    println!(
        "(CLOSED = this policy SURVIVES the order-switching adversary ⇒ Tetris solved; LEAK = patch point)\n"
    );

    let t0 = Instant::now();
    let out = closure(depth, cap);
    let dt = t0.elapsed().as_secs_f64();
    match out {
        Outcome::Closed { size, mh, md } => println!(
            "CLOSED  |carrier|={size}  (max height={mh}, max debt={md})  [{dt:.1}s]\n\
             *** the robust policy answers EVERY adversary piece from every reachable state, forever ***\n\
             *** finite closed carrier vs the SWITCHING adversary ⇒ Tetris SOLVED for this policy ***"
        ),
        Outcome::Exploded { size, mh, md } => println!(
            "EXPLODED  |carrier|>{size}  (max height={mh}, max debt={md})  [{dt:.1}s]\n\
             the policy survives but its carrier exceeds the cap — no SMALL carrier for this policy."
        ),
        Outcome::Leak {
            piece,
            mask,
            size,
            mh,
            md,
        } => {
            let rem: Vec<&str> = (0..7)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| NAMES[i])
                .collect();
            println!(
                "LEAK  the policy is stranded: piece {piece:?} unanswerable in-budget (bag-remaining {})\n\
                 reached |carrier|={size} states (max height={mh}, max debt={md})  [{dt:.1}s]\n\
                 ⇒ the depth-{depth} robust policy has a switching hole here; improving it at this state\n\
                   is the next policy-iteration step (deeper search / wider debt / a better eval).",
                rem.join("")
            );
        }
    }
}
