#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! tetris_tropical — the max-plus CYCLE-TIME / eigen-cycle computation on the REAL drop+clear dynamics.
//!
//! # Why this exists
//!
//! `TopicalTetris.lean` proved survival ⟺ the sign of one spectral number (the cycle time of the
//! bag operator) and gave a computable `eigenRate` — but we only ran it on toy 5-O bags. This binary
//! runs the same idea on the REAL engine. For each of the 5040 fixed bag orders σ (repeat σ forever
//! = the bag "matrix" A_σ iterated), it plays a deterministic strategy from the empty board and
//! detects the exact board EIGEN-CYCLE it settles into (a repeat at a bag boundary), or a top-out.
//!
//! Because the engine board is bottom-anchored, an exact board-repeat IS an eigen-cycle of rate 0
//! (bounded forever); a top-out is rate > 0 (the strategy loses to that periodic order). So per order
//! we LEARN: does the strategy reach a bounded eigen-cycle, and if so its PERIOD, max HEIGHT, the
//! CRITICAL COLUMN (tallest = the max-plus critical cell), and the CLEARING RATE (should cluster at
//! the proven 2.8 lines/bag equilibrium). Across all 5040 orders: the survival fraction, the cycle
//! statistics, and the HARD orders (which piece arrangements break the strategy).
//!
//! This is the tropical theory used as an INSTRUMENT — measuring the margins, periods, and
//! bottlenecks the abstract theorems leave open. (Fixed-order eigen-cycles are a periodic
//! sub-adversary; the full order-switching adversary is the mean-payoff GAME = policy iteration,
//! the natural next step.)
//!
//! Run:
//!   cargo run --release -p tetris-playground --bin tetris_tropical

use std::time::Instant;

use rustc_hash::FxHashMap;
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};

/// Canonical loss line.
const ROWS: u32 = 20;
/// Max bags to look for a cycle before declaring "no cycle within horizon".
const MAX_BAGS: usize = 2000;
/// Beam lookahead (pieces) and width — set from args in `main`.
static LOOK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(7);
static WIDTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(8);

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

/// Index of the tallest column (the max-plus critical cell of the surface).
fn critical_col(b: &TetrisBoard) -> usize {
    let h = b.heights();
    (0..h.len()).max_by_key(|&i| h[i]).unwrap_or(0)
}

/// Survival eval (lower = better): hole-averse, then low max-height (loss line), then flat, then light.
fn eval(b: &TetrisBoard) -> i64 {
    (b.total_holes() as i64) * 1_000_000_000
        + (b.height() as i64) * 1_000_000
        + (b.roughness() as i64) * 1_000
        + b.heights().iter().sum::<u32>() as i64
}

/// Beam move-selector with one-bag (default) lookahead over the KNOWN periodic order. Returns the
/// best immediate `(next_board, lines_cleared)`, or `None` only if the current piece cannot be
/// placed at all (a true top-out). Plans `LOOK` pieces ahead keeping the best `WIDTH` lines.
fn choose(board: &TetrisBoard, order: &[TetrisPiece; 7], pos: usize) -> Option<(TetrisBoard, u32)> {
    let look = LOOK.load(std::sync::atomic::Ordering::Relaxed).max(1);
    let width = WIDTH.load(std::sync::atomic::Ordering::Relaxed).max(1);
    // beam entry: (current board, the first move that started this line)
    let mut beam: Vec<(TetrisBoard, (TetrisBoard, u32))> = Vec::new();
    for &pl in TetrisPiecePlacement::all_from_piece(order[pos % 7]) {
        let mut nb = *board;
        let res = nb.apply_piece_placement(pl);
        if res.is_lost == IsLost::LOST || nb.height() > ROWS {
            continue;
        }
        beam.push((nb, (nb, res.lines_cleared)));
    }
    if beam.is_empty() {
        return None; // truly dead: current piece has no landing
    }
    if beam.len() > width {
        beam.select_nth_unstable_by_key(width, |(b, _)| eval(b));
        beam.truncate(width);
    }
    for step in 1..look {
        let p = order[(pos + step) % 7];
        let mut next: Vec<(TetrisBoard, (TetrisBoard, u32))> = Vec::new();
        for (b, first) in &beam {
            for &pl in TetrisPiecePlacement::all_from_piece(p) {
                let mut nb = *b;
                let res = nb.apply_piece_placement(pl);
                if res.is_lost == IsLost::LOST || nb.height() > ROWS {
                    continue;
                }
                next.push((nb, *first));
            }
        }
        if next.is_empty() {
            break; // can't extend within horizon — fall back to current beam's first moves
        }
        if next.len() > width {
            next.select_nth_unstable_by_key(width, |(b, _)| eval(b));
            next.truncate(width);
        }
        beam = next;
    }
    beam.into_iter()
        .min_by_key(|(b, _)| eval(b))
        .map(|(_, f)| f)
}

/// Play exactly one bag (the 7 pieces of `order`) from `b`. `None` on top-out.
fn play_bag(mut b: TetrisBoard, order: &[TetrisPiece; 7]) -> Option<(TetrisBoard, u32)> {
    let mut lines = 0u32;
    for i in 0..7 {
        let (nb, lc) = choose(&b, order, i)?;
        b = nb;
        lines += lc;
    }
    Some((b, lines))
}

/// The verdict for one fixed periodic order.
enum OrderResult {
    /// Reached an exact board eigen-cycle (bounded survival, rate 0).
    Cycle {
        period: usize,
        max_h: u32,
        max_rough: u32,
        crit_col: usize,
        lines_per_bag: f64,
    },
    /// Topped out at this bag index (the strategy loses to this periodic order).
    Died { bag: usize },
    /// Survived MAX_BAGS without an exact repeat (long transient / large period).
    NoCycle { max_h: u32 },
}

/// Run order σ repeated forever under the deterministic strategy from empty; detect the eigen-cycle.
fn run_order(order: &[TetrisPiece; 7]) -> OrderResult {
    let mut b = TetrisBoard::new();
    // board -> (bag index first seen, cumulative lines at that point)
    let mut seen: FxHashMap<TetrisBoard, (usize, u64)> = FxHashMap::default();
    seen.insert(b, (0, 0));
    let mut cum_lines = 0u64;
    let mut max_h = 0u32;

    for bag in 1..=MAX_BAGS {
        match play_bag(b, order) {
            None => return OrderResult::Died { bag },
            Some((nb, lc)) => {
                b = nb;
                cum_lines += lc as u64;
                max_h = max_h.max(b.height());
            }
        }
        if let Some(&(first_bag, first_lines)) = seen.get(&b) {
            let period = bag - first_bag;
            let cyc_lines = cum_lines - first_lines;
            let (mh, mr, cc) = cycle_stats(&b, order, period);
            return OrderResult::Cycle {
                period,
                max_h: mh,
                max_rough: mr,
                crit_col: cc,
                lines_per_bag: cyc_lines as f64 / period as f64,
            };
        }
        seen.insert(b, (bag, cum_lines));
    }
    OrderResult::NoCycle { max_h }
}

/// Replay `period` bags from the cycle start to characterize the eigen-cycle (max height, max
/// roughness, critical column).
fn cycle_stats(start: &TetrisBoard, order: &[TetrisPiece; 7], period: usize) -> (u32, u32, usize) {
    let mut b = *start;
    let (mut mh, mut mr, mut cc) = (b.height(), b.roughness(), critical_col(&b));
    for _ in 0..period {
        let (nb, _) = play_bag(b, order).expect("cycle replays without top-out");
        b = nb;
        if b.height() > mh {
            mh = b.height();
            cc = critical_col(&b);
        }
        mr = mr.max(b.roughness());
    }
    (mh, mr, cc)
}

/// In-place next lexicographic permutation of `a`; returns false when wrapped to sorted.
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

fn order_name(order: &[TetrisPiece; 7], idx: &[usize; 7]) -> String {
    let _ = order;
    idx.iter().map(|&k| NAMES[k]).collect::<Vec<_>>().join("")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let parse = |pfx: &str| -> Option<usize> {
        args.iter()
            .find_map(|a| a.strip_prefix(pfx).and_then(|s| s.parse::<usize>().ok()))
    };
    if let Some(l) = parse("look") {
        LOOK.store(l, std::sync::atomic::Ordering::Relaxed);
    }
    if let Some(w) = parse("width") {
        WIDTH.store(w, std::sync::atomic::Ordering::Relaxed);
    }
    println!(
        "tetris_tropical — max-plus eigen-cycle of the REAL drop+clear dynamics, over all 5040 bag orders"
    );
    println!(
        "(beam strategy: lookahead={} pieces, width={}; exact board-repeat = bounded eigen-cycle = survives)\n",
        LOOK.load(std::sync::atomic::Ordering::Relaxed),
        WIDTH.load(std::sync::atomic::Ordering::Relaxed),
    );
    let t0 = Instant::now();

    let mut idx: [usize; 7] = [0, 1, 2, 3, 4, 5, 6];
    let mut n_cycle = 0usize;
    let mut n_died = 0usize;
    let mut n_nocycle = 0usize;
    let mut period_hist: FxHashMap<usize, usize> = FxHashMap::default();
    let mut max_h_overall = 0u32;
    let mut max_period = 0usize;
    let mut lpb_sum = 0f64;
    let mut crit_hist = [0usize; 10];
    let mut died_examples: Vec<(String, usize)> = Vec::new();
    let mut deep_cycle_examples: Vec<(String, usize, u32)> = Vec::new();

    loop {
        let order: [TetrisPiece; 7] = std::array::from_fn(|i| PIECES[idx[i]]);
        match run_order(&order) {
            OrderResult::Cycle {
                period,
                max_h,
                max_rough,
                crit_col,
                lines_per_bag,
            } => {
                n_cycle += 1;
                *period_hist.entry(period).or_insert(0) += 1;
                max_h_overall = max_h_overall.max(max_h);
                max_period = max_period.max(period);
                lpb_sum += lines_per_bag;
                crit_hist[crit_col] += 1;
                if period >= 5 && deep_cycle_examples.len() < 8 {
                    deep_cycle_examples.push((order_name(&order, &idx), period, max_h));
                }
                let _ = max_rough;
            }
            OrderResult::Died { bag } => {
                n_died += 1;
                if died_examples.len() < 12 {
                    died_examples.push((order_name(&order, &idx), bag));
                }
            }
            OrderResult::NoCycle { max_h } => {
                n_nocycle += 1;
                max_h_overall = max_h_overall.max(max_h);
            }
        }
        if !next_perm(&mut idx) {
            break;
        }
    }

    let total = n_cycle + n_died + n_nocycle;
    println!(
        "=== RESULTS over {total} bag orders  [{:.1}s] ===",
        t0.elapsed().as_secs_f64()
    );
    println!(
        "  bounded eigen-cycle (SURVIVES periodic order): {n_cycle}  ({:.1}%)",
        100.0 * n_cycle as f64 / total as f64
    );
    println!(
        "  topped out (LOSES to periodic order):          {n_died}  ({:.1}%)",
        100.0 * n_died as f64 / total as f64
    );
    if n_nocycle > 0 {
        println!("  no cycle within {MAX_BAGS} bags:                 {n_nocycle}");
    }
    if n_cycle > 0 {
        println!(
            "\n  cycle stats (survivors): max height in any cycle = {max_h_overall}, longest period = {max_period} bags, \
             mean clearing rate = {:.3} lines/bag  (theory equilibrium = 2.800)",
            lpb_sum / n_cycle as f64
        );
        let mut periods: Vec<(usize, usize)> = period_hist.into_iter().collect();
        periods.sort();
        let shown: Vec<String> = periods.iter().map(|(p, c)| format!("{p}bag×{c}")).collect();
        println!("  period histogram: {}", shown.join("  "));
        let crit: Vec<String> = (0..10).map(|c| format!("c{c}:{}", crit_hist[c])).collect();
        println!("  critical (tallest) column histogram: {}", crit.join(" "));
        if !deep_cycle_examples.is_empty() {
            println!("  example long cycles (order, period, max-h):");
            for (o, p, h) in &deep_cycle_examples {
                println!("    {o}  period={p}bags  max-h={h}");
            }
        }
    }
    if !died_examples.is_empty() {
        println!("\n  example HARD orders that broke the strategy (order, top-out bag):");
        for (o, bag) in &died_examples {
            println!("    {o}  died at bag {bag}");
        }
    }

    println!("\n=== READING ===");
    if n_died == 0 && n_nocycle == 0 {
        println!(
            "The strategy reaches a bounded eigen-cycle on EVERY one of the {total} periodic bag orders — \
             the tropical eigen-cycle picture is real on actual Tetris, and every fixed order is survivable. \
             The open question is the ORDER-SWITCHING adversary (the mean-payoff game = policy iteration)."
        );
    } else {
        println!(
            "{n_died} periodic orders top out this simple strategy ⇒ those piece arrangements are the \
             spectral bottleneck (look at the hard-order list — expect S/Z-clustered tails). A smarter \
             per-order strategy (or lookahead) may still cycle on them; that gap is the make-or-break."
        );
    }
}
