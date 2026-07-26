#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! tetris_energy — the energy-informed lookahead controller.
//!
//! # Why this exists
//!
//! The Lean energy-game theory (`proofs/Proofs/Experiments/EnergyGame.lean` et al.) proved,
//! sorry-free, that controlled Tetris survival is governed by THREE quantities of which two
//! are tame:
//!   * **debt** (buried holes) is the ONLY killer — and `K = 1` is provably sufficient for
//!     the S/Z core (`no_holeFree_invariant`, `SZ_handled_with_budget1`);
//!   * **height** is bounded by the loss line;
//!   * the entire residual difficulty lives in the **roughness** dimension + sustaining the
//!     `4/cols` clearing equilibrium (`survival_forces_clears`, `card_applyStep`).
//!
//! The existing `tetris_carrier_probe` minimax used an ADDITIVE eval
//! (`agg + 6·holes + bump/2 + 2·maxh`) where height competes with debt — and it topped out
//! at bag 30 with 23 holes. The theory says debt must DOMINATE: tolerate height, never bury.
//! This binary is that controller — a **debt-dominant** (lexicographic) deep bag-aware
//! minimax — and its whole point is one diagnostic:
//!
//!   **Does the survival horizon DIVERGE as lookahead depth grows (⇒ the optimal player
//!   survives ⇒ adversarial 7-bag is SOLVABLE), or PLATEAU (⇒ the worst-order adversary
//!   wins ⇒ prove impossibility instead)?**
//!
//! Run:
//!   cargo run --release -p tetris-playground --bin tetris_energy -- run bags2000 depth6
//!   cargo run --release -p tetris-playground --bin tetris_energy -- sweep bags2000

use std::time::Instant;

use rustc_hash::FxHashMap;
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};

/// Board height ceiling (canonical Tetris loses above row 19).
const ROWS: u32 = 20;
/// Loss sentinel for the minimax value (lower = better for the player).
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

/// **The energy cost** (lower = better). STRICTLY LEXICOGRAPHIC `(debt, height, roughness,
/// mass)` — debt dominates everything, exactly as the proofs require: the player will never
/// accept a buried hole to lower its height or surface. Among hole-free options it then
/// minimizes height (the loss line), then roughness (maneuverability for S/Z notches), then
/// total mass. Line clears are rewarded implicitly (they drop height and can expose holes).
fn energy_cost(b: &TetrisBoard) -> i64 {
    let holes = b.total_holes() as i64;
    let maxh = b.height() as i64;
    let rough = b.roughness() as i64;
    let agg = b.heights().iter().sum::<u32>() as i64;
    holes * 1_000_000_000 + maxh * 1_000_000 + rough * 1_000 + agg
}

/// Player-node beam width (top-K placements explored deeper). `usize::MAX` = exact.
static BEAM_K: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Depth-limited bag-aware minimax with the **energy** leaf eval. Adversary (revealed piece)
/// MAXIMIZES the player's value; player MINIMIZES over its top-`BEAM_K` placements. Memoized
/// on `(board, mask, depth)`. Returns `LOSS` if a top-out is forced within the horizon.
fn bag_minimax(
    board: TetrisBoard,
    mask: u8,
    depth: u8,
    memo: &mut FxHashMap<(TetrisBoard, u8, u8), i64>,
) -> i64 {
    if mask == 0 || depth == 0 {
        return energy_cost(&board);
    }
    if let Some(&v) = memo.get(&(board, mask, depth)) {
        return v;
    }
    let beam_k = BEAM_K.load(std::sync::atomic::Ordering::Relaxed);
    let mut best_adv = i64::MIN;
    for (pi, p) in PIECES.iter().enumerate() {
        let bit = 1u8 << pi;
        if mask & bit == 0 {
            continue;
        }
        let mut cands: Vec<(i64, TetrisBoard)> = Vec::new();
        for &pl in TetrisPiecePlacement::all_from_piece(*p) {
            let mut nb = board;
            let res = nb.apply_piece_placement(pl);
            if res.is_lost == IsLost::LOST || nb.height() > ROWS {
                continue;
            }
            cands.push((energy_cost(&nb), nb));
        }
        let mut best_play = LOSS;
        if !cands.is_empty() {
            if cands.len() > beam_k {
                cands.select_nth_unstable_by_key(beam_k, |c| c.0);
                cands.truncate(beam_k);
            }
            for (_, nb) in cands {
                let v = bag_minimax(nb, mask & !bit, depth - 1, memo);
                if v < best_play {
                    best_play = v;
                }
            }
        }
        if best_play > best_adv {
            best_adv = best_play;
        }
    }
    memo.insert((board, mask, depth), best_adv);
    best_adv
}

/// Play one bag vs the worst-order adversary with the depth-`depth` energy minimax. Each
/// step the adversary reveals the piece maximizing the player's best-response value; the
/// player picks the minimizing placement. Returns `(end_board, lines_cleared)` or `None` on
/// forced top-out.
fn play_bag(board: TetrisBoard, depth: u8) -> Option<(TetrisBoard, u32)> {
    let mut b = board;
    let mut mask = FULL_MASK;
    let mut lines = 0u32;
    let mut memo: FxHashMap<(TetrisBoard, u8, u8), i64> = FxHashMap::default();
    while mask != 0 {
        let look = depth.min(mask.count_ones() as u8);
        // adversary: argmax over remaining pieces of the player's best response
        let mut adv: Option<(usize, TetrisPiece, i64)> = None;
        for (pi, p) in PIECES.iter().enumerate() {
            let bit = 1u8 << pi;
            if mask & bit == 0 {
                continue;
            }
            let mut best_play = LOSS;
            for &pl in TetrisPiecePlacement::all_from_piece(*p) {
                let mut nb = b;
                let res = nb.apply_piece_placement(pl);
                if res.is_lost == IsLost::LOST || nb.height() > ROWS {
                    continue;
                }
                let v = bag_minimax(nb, mask & !bit, look.saturating_sub(1), &mut memo);
                if v < best_play {
                    best_play = v;
                }
            }
            if adv.as_ref().is_none_or(|(_, _, av)| best_play > *av) {
                adv = Some((pi, *p, best_play));
            }
        }
        let (pi, p, av) = adv.unwrap();
        if av >= LOSS {
            return None; // adversary can force a top-out
        }
        // player: pick the minimizing placement for the revealed piece
        let bit = 1u8 << pi;
        let mut best_board = b;
        let mut best_v = LOSS;
        let mut best_lines = 0u32;
        for &pl in TetrisPiecePlacement::all_from_piece(p) {
            let mut nb = b;
            let res = nb.apply_piece_placement(pl);
            if res.is_lost == IsLost::LOST || nb.height() > ROWS {
                continue;
            }
            let v = bag_minimax(nb, mask & !bit, look.saturating_sub(1), &mut memo);
            if v < best_v {
                best_v = v;
                best_board = nb;
                best_lines = res.lines_cleared;
            }
        }
        b = best_board;
        lines += best_lines;
        mask &= !bit;
    }
    Some((b, lines))
}

/// Long-run from empty: chain bags with the depth-`depth` energy controller vs the
/// worst-order adversary. Returns `Some(bag_index_of_topout)` or `None` if it survived all
/// `nbags` (a bounded steady state). Prints the trajectory.
fn run(nbags: usize, depth: u8, verbose: bool) -> Option<usize> {
    if verbose {
        println!(
            "MODE: energy controller (debt-dominant minimax, depth={depth}, beam={}) vs worst-order adversary",
            match BEAM_K.load(std::sync::atomic::Ordering::Relaxed) {
                usize::MAX => "exact".to_string(),
                k => k.to_string(),
            }
        );
    }
    let t0 = Instant::now();
    let mut b = TetrisBoard::new();
    let (mut max_h, mut max_holes, mut max_rough) = (0u32, 0u32, 0u32);
    let mut total_lines = 0u64;
    for bag in 0..nbags {
        match play_bag(b, depth) {
            None => {
                if verbose {
                    println!(
                        "TOPPED OUT at bag {bag} ({} pieces). max_h={max_h} max_holes={max_holes} \
                         max_rough={max_rough} lines={total_lines} ({:.1}s)",
                        bag * 7,
                        t0.elapsed().as_secs_f64()
                    );
                }
                return Some(bag);
            }
            Some((nb, lc)) => {
                b = nb;
                total_lines += lc as u64;
                max_h = max_h.max(b.height());
                max_holes = max_holes.max(b.total_holes());
                max_rough = max_rough.max(b.roughness());
                if verbose && (bag < 12 || (bag + 1) % 100 == 0) {
                    println!(
                        "  bag {:5} ({:7} pc): h={:2} (max {:2})  debt={:2} (max {:2})  \
                         rough={:3} (max {:3})  lines={}",
                        bag + 1,
                        (bag + 1) * 7,
                        b.height(),
                        max_h,
                        b.total_holes(),
                        max_holes,
                        b.roughness(),
                        max_rough,
                        total_lines
                    );
                }
            }
        }
    }
    if verbose {
        println!(
            "\nSURVIVED all {nbags} bags ({} pieces) in {:.1}s — NO top-out.\n\
             BOUNDED steady state: max_h={max_h}  max_debt={max_holes}  max_rough={max_rough}  \
             total_lines={total_lines}",
            nbags * 7,
            t0.elapsed().as_secs_f64()
        );
    }
    None
}

/// **The diagnostic sweep.** Run the energy controller at increasing lookahead depths and
/// report each survival horizon. A DIVERGING horizon ⇒ the optimal player survives ⇒
/// SOLVABLE; a PLATEAU ⇒ the worst-order adversary wins ⇒ prove impossibility.
fn run_sweep(nbags: usize) {
    println!("MODE: depth sweep — survival horizon vs lookahead depth (energy controller)");
    println!("(diverging horizon ⇒ solvable; plateau ⇒ adversary wins)\n");
    let depths = [1u8, 2, 3, 4, 5, 6, 7];
    let mut results: Vec<(u8, String)> = Vec::new();
    for &d in depths.iter() {
        let t0 = Instant::now();
        let outcome = run(nbags, d, false);
        let label = match outcome {
            Some(bag) => format!("topped out at bag {bag} ({} pieces)", bag * 7),
            None => format!("SURVIVED all {nbags} bags ({} pieces)", nbags * 7),
        };
        println!(
            "  depth {d}: {label}   [{:.1}s]",
            t0.elapsed().as_secs_f64()
        );
        results.push((d, label));
    }
    println!("\n=== VERDICT ===");
    let survived_all = results.iter().any(|(_, l)| l.starts_with("SURVIVED"));
    if survived_all {
        println!(
            "At least one depth SURVIVED all {nbags} bags ⇒ a finite-lookahead energy controller \
             reaches a BOUNDED steady state ⇒ strong evidence adversarial 7-bag is SOLVABLE. \
             Next: extract the controller's visited boundary surfaces as Σ and certify in Lean."
        );
    } else {
        println!(
            "Every depth topped out. If the horizon is DIVERGING with depth, deeper lookahead \
             (or the true optimal player) may still survive — solvable but needs depth. If it \
             PLATEAUED, the worst-order adversary likely wins ⇒ pivot to proving impossibility."
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let parse = |pfx: &str| -> Option<usize> {
        args.iter()
            .find_map(|a| a.strip_prefix(pfx).and_then(|s| s.parse::<usize>().ok()))
    };
    if let Some(k) = parse("beam") {
        BEAM_K.store(k, std::sync::atomic::Ordering::Relaxed);
    }
    let nbags = parse("bags").unwrap_or(2000);
    println!(
        "tetris_energy — debt-dominant deep-lookahead controller vs worst-order 7-bag adversary"
    );
    if args.iter().any(|a| a == "sweep") {
        run_sweep(nbags);
    } else {
        let depth = parse("depth").unwrap_or(6) as u8;
        run(nbags, depth, true);
    }
}
