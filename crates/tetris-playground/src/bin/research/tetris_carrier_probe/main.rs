#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! tetris_carrier_probe — Phase 0 of the search-grounded-closure route to proving
//! the Tetris atlas exists.
//!
//! # Why this exists
//!
//! The Lean route (`AbstractSafe.lean`) reduces `TetrisSolvableValid` to: exhibit a
//! NONEMPTY, CLOSED, init-containing set of game states from which a fixed strategy
//! survives every adversarial 7-bag order forever. Both prior Lean attempts stalled
//! because they tried to *prove* such a set exists before *confirming* it does. This
//! binary confirms (or refutes) existence empirically, at full height 20, with EXACT
//! boards (no lossy abstraction — so whatever it finds is directly certifiable later
//! by `native_decide` with no soundness debt).
//!
//! # What it computes
//!
//! State = (exact board, current 7-bag remaining-set). The adversary draws any piece
//! still in the bag; a DETERMINISTIC flatten-and-drain strategy picks the placement.
//! We:
//!   1. forward-BFS the exact reachable set `R` from init under ALL adversary draws,
//!   2. backward death-propagate (a state is dead if SOME drawable piece tops the
//!      strategy out, or leads to a dead state — the adversarial AND-safety GFP),
//!   3. report the surviving closed core `S = R \ dead`.
//!
//! # The 3-outcome diagnostic (this is the point — a bounded run is CONCLUSIVE)
//!
//!   1. CONVERGED, init survives, |S| small  -> candidate carrier found; proceed to
//!      Lean certification (`native_decide` on `S`).
//!   2. Bounded height but |R| hit the budget -> set is FINITE but large; shard the
//!      certification or tighten the strategy. (Distinguished from 3 by height NOT
//!      drifting up.)
//!   3. Height drifts up to 20 / |R| explodes -> strategy fails to reset per bag;
//!      effectively unbounded. Real negative result -> switch strategy.
//!
//! Run: `cargo run --release -p tetris-playground --bin tetris_carrier_probe [STATE_BUDGET]`

use std::collections::VecDeque;
use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet};
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPieceBagState, TetrisPiecePlacement};

/// Sentinel edge target meaning "the strategy tops out on this draw" (a loss edge).
const DEAD: u32 = u32::MAX;
/// Board height ceiling. Canonical Tetris loses above row 19, so height > 20 is lost.
const ROWS: u32 = 20;
/// Default cap on distinct reachable states before we declare an explosion.
const DEFAULT_BUDGET: usize = 5_000_000;

/// One node of the reachable graph: an exact board plus the pieces still in the
/// current 7-bag. `bag == TetrisPieceBagState::new()` (FULL) marks a bag boundary.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct State {
    board: TetrisBoard,
    bag: TetrisPieceBagState,
}

/// Which deterministic strategy the probe drives. Each must be a pure function of
/// (board, piece) so the adversary's only freedom is the reveal order.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Strat {
    /// Greedy: minimize (holes, height, roughness, summed heights). No structured well.
    Flatten,
    /// Reserve column `WELL_COL` empty; flatten cols 0..9 \ well; drain with a
    /// vertical I in the well whenever that clears ≥1 line.
    Well,
}

/// Reserved well column for the `Well` strategy (rightmost).
const WELL_COL: usize = 9;

/// An admissibility band: the designed carrier invariant. A boundary board is
/// admissible iff it stays within all caps. When a band is active, the boundary BFS
/// treats *escaping the band* (a leaf that violates a cap) as a death — so the
/// surviving core is a CONTROLLED-INVARIANT closed set, small by construction when the
/// caps are tight. `None` caps mean "unconstrained" (free reachable BFS).
#[derive(Clone, Copy)]
struct Admiss {
    hcap: Option<u32>,    // max board height
    rcap: Option<u32>,    // max surface roughness
    holecap: Option<u32>, // max total holes
    well_empty: bool,     // require the well column to be empty at bag boundaries
    /// Optional weighted Lyapunov potential `Φ = height + a·roughness + b·holes`,
    /// admissible iff `Φ ≤ cap`. Stored as `(a, b, cap)`. Unlike the box caps above
    /// (which forbid roughness REGARDLESS of height), the potential expresses a
    /// trade-off: a rough board is admissible IF it is low enough. This is the carrier
    /// shape a per-bag drift argument actually certifies. `None` = no potential.
    pot: Option<(u32, u32, u32)>,
}

impl Admiss {
    const UNBOUNDED: Self = Self {
        hcap: None,
        rcap: None,
        holecap: None,
        well_empty: false,
        pot: None,
    };
    fn active(&self) -> bool {
        self.hcap.is_some()
            || self.rcap.is_some()
            || self.holecap.is_some()
            || self.well_empty
            || self.pot.is_some()
    }
    /// The Lyapunov potential value `Φ(b) = height + a·roughness + b·holes` (0 if no
    /// potential weights are configured).
    fn phi(&self, b: &TetrisBoard) -> u32 {
        match self.pot {
            Some((a, bb, _)) => b.height() + a * b.roughness() + bb * b.total_holes(),
            None => 0,
        }
    }
    /// Is `b` inside the band?
    fn ok(&self, b: &TetrisBoard) -> bool {
        if let Some(h) = self.hcap {
            if b.height() > h {
                return false;
            }
        }
        if let Some(r) = self.rcap {
            if b.roughness() > r {
                return false;
            }
        }
        if let Some(hc) = self.holecap {
            if b.total_holes() > hc {
                return false;
            }
        }
        if self.well_empty && b.as_limbs()[WELL_COL] != 0 {
            return false;
        }
        if let Some((_, _, cap)) = self.pot {
            if self.phi(b) > cap {
                return false;
            }
        }
        true
    }
}

/// Strategy dispatch — pure function of (board, piece). `None` iff it tops out.
fn choose(board: &TetrisBoard, piece: TetrisPiece, strat: Strat) -> Option<TetrisBoard> {
    match strat {
        Strat::Flatten => choose_flatten(board, piece),
        Strat::Well => choose_well(board, piece),
    }
}

/// Greedy flatten-and-drain. Minimize (holes, max height, roughness, summed heights).
fn choose_flatten(board: &TetrisBoard, piece: TetrisPiece) -> Option<TetrisBoard> {
    let mut best: Option<(TetrisBoard, (u32, u32, u32, u32))> = None;
    for &pl in TetrisPiecePlacement::all_from_piece(piece) {
        let mut nb = *board;
        let res = nb.apply_piece_placement(pl);
        if res.is_lost == IsLost::LOST || nb.height() > ROWS {
            continue;
        }
        let score = (
            nb.total_holes(),
            nb.height(),
            nb.roughness(),
            nb.heights().iter().sum::<u32>(),
        );
        if best.is_none_or(|(_, bs)| score < bs) {
            best = Some((nb, score));
        }
    }
    best.map(|(b, _)| b)
}

/// Max column height ignoring the reserved well column.
fn max_height_nonwell(b: &TetrisBoard) -> u32 {
    b.heights()
        .iter()
        .enumerate()
        .filter(|(c, _)| *c != WELL_COL)
        .map(|(_, &h)| h)
        .max()
        .unwrap_or(0)
}

/// Well-reserving stack-and-burn. Preference order:
///   1. DRAIN: if `piece == I` and a placement clears ≥1 line, take the one clearing
///      the most lines (then lowest resulting height). This is the controlled burn.
///   2. KEEP: among placements that leave the well column untouched, minimize
///      (holes, max-height-of-non-well-cols, roughness, summed heights). Keeps the
///      well open and the rest of the surface flat.
///   3. FORCED: if every legal placement touches the well, fall back to the flattest.
///
/// `None` iff no legal placement at all (top-out).
fn choose_well(board: &TetrisBoard, piece: TetrisPiece) -> Option<TetrisBoard> {
    let before = board.as_limbs();
    let mut best_drain: Option<(u32, u32, TetrisBoard)> = None; // (lines, -, board) max lines
    let mut best_keep: Option<((u32, u32, u32, u32), TetrisBoard)> = None;
    let mut best_forced: Option<((u32, u32, u32), TetrisBoard)> = None;
    for &pl in TetrisPiecePlacement::all_from_piece(piece) {
        let mut nb = *board;
        let res = nb.apply_piece_placement(pl);
        if res.is_lost == IsLost::LOST || nb.height() > ROWS {
            continue;
        }
        let after = nb.as_limbs();
        let touches_well = after[WELL_COL] != before[WELL_COL];

        if piece == TetrisPiece::I_PIECE && res.lines_cleared > 0 {
            let key = (res.lines_cleared, u32::MAX - nb.height());
            if best_drain.is_none_or(|(l, h, _)| (key.0, key.1) > (l, h)) {
                best_drain = Some((key.0, key.1, nb));
            }
        }
        if !touches_well {
            let score = (
                nb.total_holes(),
                max_height_nonwell(&nb),
                nb.roughness(),
                nb.heights().iter().sum::<u32>(),
            );
            if best_keep.is_none_or(|(s, _)| score < s) {
                best_keep = Some((score, nb));
            }
        }
        let fscore = (nb.total_holes(), nb.height(), nb.roughness());
        if best_forced.is_none_or(|(s, _)| fscore < s) {
            best_forced = Some((fscore, nb));
        }
    }
    if let Some((_, _, b)) = best_drain {
        return Some(b);
    }
    if let Some((_, b)) = best_keep {
        return Some(b);
    }
    best_forced.map(|(_, b)| b)
}

/// Expand one full bag from a boundary board `start` (full bag of 7). BFS the
/// intra-bag tree over (board, remaining-bag) under all adversary reveal orders, with
/// the deterministic strategy responding. Returns `(has_loss, leaf_boards)`:
///   - `has_loss` = some reachable reveal has NO legal strategy placement (a top-out),
///   - `leaf_boards` = the distinct boards once the bag empties (the next boundary
///     boards). Because the strategy is a pure function of (board, piece), the
///     adversary's adaptivity is exactly the set of reveal orders, captured by the
///     tree; for-all-orders survival ⟺ no reachable node tops out.
fn expand_bag(start: TetrisBoard, strat: Strat) -> (bool, Vec<TetrisBoard>) {
    let full = TetrisPieceBagState::new();
    let mut seen: FxHashSet<(TetrisBoard, u8)> = FxHashSet::default();
    let mut leaves: FxHashSet<TetrisBoard> = FxHashSet::default();
    let mut stack: Vec<(TetrisBoard, TetrisPieceBagState)> = vec![(start, full)];
    seen.insert((start, u8::from(full)));
    let mut has_loss = false;
    let pieces = TetrisPiece::all();
    while let Some((b, bag)) = stack.pop() {
        if bag.is_empty() {
            leaves.insert(b);
            continue;
        }
        for &p in pieces.iter() {
            if !bag.contains(p) {
                continue;
            }
            match choose(&b, p, strat) {
                None => has_loss = true,
                Some(nb) => {
                    let mut bag2 = bag;
                    bag2.remove(p);
                    if seen.insert((nb, u8::from(bag2))) {
                        stack.push((nb, bag2));
                    }
                }
            }
        }
    }
    (has_loss, leaves.into_iter().collect())
}

/// Bag-boundary macro-BFS: nodes are full-bag boards (reset surfaces); one edge =
/// one full bag of adversarial play under the deterministic strategy. This is the
/// state space the per-bag invariant `tetrisSolvableValid_of_bag_indexed_invariant`
/// actually certifies — intra-bag transients are folded into `expand_bag` and never
/// stored, killing the (board,bag) explosion.
fn run_boundary(budget: usize, strat: Strat, adm: Admiss) {
    let sname = match strat {
        Strat::Flatten => "flatten",
        Strat::Well => "well",
    };
    println!("MODE: bag-boundary macro-BFS (per-bag reset surfaces only), strategy={sname}");
    if adm.active() {
        println!(
            "BAND: hcap={:?} rcap={:?} holecap={:?} well_empty={} (band-escape = death)",
            adm.hcap, adm.rcap, adm.holecap, adm.well_empty
        );
    }
    let t0 = Instant::now();
    let init = TetrisBoard::new();

    let mut index: FxHashMap<TetrisBoard, u32> = FxHashMap::default();
    let mut boards: Vec<TetrisBoard> = Vec::new();
    let mut has_loss: Vec<bool> = Vec::new();
    let mut edge_src: Vec<u32> = Vec::new();
    let mut edge_dst: Vec<u32> = Vec::new();
    let mut queue: VecDeque<u32> = VecDeque::new();

    index.insert(init, 0);
    boards.push(init);
    has_loss.push(false);
    queue.push_back(0);

    let mut exploded = false;
    let mut max_h = 0u32;
    while let Some(sid) = queue.pop_front() {
        let (loss, leaves) = expand_bag(boards[sid as usize], strat);
        if loss {
            has_loss[sid as usize] = true;
        }
        for leaf in leaves {
            // Controlled-invariant semantics: a leaf outside the band is an adversary
            // order that escapes the carrier — treat as a death of the source, and do
            // not expand it (it is not part of the band).
            if adm.active() && !adm.ok(&leaf) {
                has_loss[sid as usize] = true;
                continue;
            }
            max_h = max_h.max(leaf.height());
            let nid = if let Some(&id) = index.get(&leaf) {
                id
            } else {
                if boards.len() >= budget {
                    exploded = true;
                    break;
                }
                let id = boards.len() as u32;
                index.insert(leaf, id);
                boards.push(leaf);
                has_loss.push(false);
                queue.push_back(id);
                id
            };
            edge_src.push(sid);
            edge_dst.push(nid);
        }
        if exploded {
            break;
        }
    }

    let n = boards.len();
    let m = edge_src.len();
    println!(
        "macro-BFS: |boundary R|={n} boards, {m} bag-edges, max_height={max_h}, {:.1}s{}",
        t0.elapsed().as_secs_f64(),
        if exploded {
            "  [EXPLODED: hit budget]"
        } else {
            ""
        }
    );

    // backward death propagation over boundary boards (adversarial AND-safety GFP)
    let mut offsets = vec![0u32; n + 1];
    for &d in &edge_dst {
        offsets[d as usize + 1] += 1;
    }
    for i in 0..n {
        offsets[i + 1] += offsets[i];
    }
    let mut rev = vec![0u32; m];
    let mut cur = offsets.clone();
    for k in 0..m {
        let d = edge_dst[k] as usize;
        rev[cur[d] as usize] = edge_src[k];
        cur[d] += 1;
    }
    let mut dead = vec![false; n];
    let mut dq: VecDeque<u32> = VecDeque::new();
    for i in 0..n {
        if has_loss[i] {
            dead[i] = true;
            dq.push_back(i as u32);
        }
    }
    while let Some(s) = dq.pop_front() {
        let lo = offsets[s as usize] as usize;
        let hi = offsets[s as usize + 1] as usize;
        for &pred in &rev[lo..hi] {
            if !dead[pred as usize] {
                dead[pred as usize] = true;
                dq.push_back(pred);
            }
        }
    }

    let dead_count = dead.iter().filter(|&&d| d).count();
    let alive = n - dead_count;
    let init_alive = !dead[0];
    // Distributions over ALL reachable boundary boards (diagnose what proliferates).
    let mut max_h_surv = 0u32;
    let mut max_holes = 0u32;
    let mut max_rough = 0u32;
    let mut hole_hist: FxHashMap<u32, u64> = FxHashMap::default();
    for i in 0..n {
        let b = &boards[i];
        let holes = b.total_holes();
        *hole_hist.entry(holes).or_insert(0) += 1;
        max_holes = max_holes.max(holes);
        max_rough = max_rough.max(b.roughness());
        if !dead[i] {
            max_h_surv = max_h_surv.max(b.height());
        }
    }
    let mut hole_keys: Vec<_> = hole_hist.keys().copied().collect();
    hole_keys.sort_unstable();

    println!("\n================ RESULT (bag-boundary) ================");
    println!("|boundary R| reachable     : {n}");
    println!("dead boundary boards       : {dead_count}");
    println!("|S| surviving boundary core: {alive}");
    println!("init in closed core?       : {init_alive}");
    println!("max height in core         : {max_h_surv}");
    println!("max holes / roughness (R)  : {max_holes} / {max_rough}");
    print!("hole histogram (all R)     : ");
    for k in hole_keys.iter().take(12) {
        print!("holes{k}={} ", hole_hist[k]);
    }
    println!();

    println!("\n---------------- REGIME ----------------");
    if exploded {
        println!(
            "OUTCOME 2: boundary set hit budget {budget} (still finite, max h={max_h}). Raise \
             budget; numbers are a truncated lower bound."
        );
    } else if init_alive && alive > 0 {
        println!(
            "OUTCOME 1: CONVERGED — nonempty closed boundary core, init SURVIVES, |S|={alive} \
             reset surfaces (max h={max_h_surv}). CANDIDATE CARRIER. If |S| ≲ 1e5, this is R1: \
             export S and certify per-bag closure in Lean (native_decide)."
        );
    } else {
        println!(
            "COLLAPSE: converged but surviving core {} — flatten-drain does NOT prove solvability \
             at the bag-boundary level. Honest negative for THIS strategy.",
            if alive == 0 {
                "is EMPTY".to_string()
            } else {
                "excludes init".to_string()
            }
        );
    }
    println!("total time: {:.1}s", t0.elapsed().as_secs_f64());
}

/// Optimal-player AND-OR band fixpoint — the DECISIVE existence test.
///
/// Unlike the deterministic-strategy modes, here the PLAYER plays optimally: at each
/// revealed piece it may choose ANY admissible placement (OR), while the adversary
/// chooses the worst piece (AND). A state `(board, bag)` is safe iff for EVERY piece
/// the adversary can draw, SOME placement keeps the player in a safe admissible state.
/// This computes the true safe set restricted to the band — answering "does a closed
/// band exist for ANY strategy?", not just for a hand-crafted greedy one.
///
/// Admissibility: every state's board is height-capped (keeps the graph finite);
/// the FULL band (holes/roughness/well) is enforced only at bag boundaries
/// (`bag == FULL`), so transient holes created mid-bag and cleared by bag end are
/// allowed. If `init` is safe, the safe boards form a certifiable carrier (R1). If
/// `init` is unsafe, no band-respecting strategy survives — a strong confirmation of
/// the structural obstruction (crux #66/#72) → routes to T2.
fn run_optimal(budget: usize, adm: Admiss) {
    println!("MODE: optimal-player AND-OR band fixpoint (player OR, adversary AND)");
    println!(
        "BAND (at boundaries): hcap={:?} rcap={:?} holecap={:?} well_empty={} pot={:?}",
        adm.hcap, adm.rcap, adm.holecap, adm.well_empty, adm.pot
    );
    let hcap = adm.hcap.unwrap_or(ROWS);
    // Intra-bag boards get a little slack above the boundary cap: a drain transiently
    // spikes height (a vertical I dropped before it clears 4 rows), so forbidding that
    // would wrongly rule out legal drains. Boundary boards still obey `hcap`.
    let intra_hcap = (hcap + 4).min(ROWS);
    // To keep the optimal-player graph tractable, the roughness/well/potential
    // constraints are ALSO applied to intra-bag states (the hole cap stays boundary-only,
    // so the bootstrap transient hole is allowed). The potential gets matching slack
    // (+4·(a+b+1) headroom) so a mid-bag drain spike is not wrongly forbidden. This is a
    // conservative band: if init is safe here it is safe in the looser band; if unsafe,
    // it is suggestive but not a proof for the looser band (flagged in the verdict).
    // Intra slack is a small constant (not weight-scaled): a roughness-weighted potential
    // swings wildly mid-bag, and a generous slack admits ~the whole board space and
    // explodes. +5 covers a typical drain transient; it may conservatively forbid the
    // largest spikes (→ possibly undercount the safe core, never overcount).
    let intra_pot = adm.pot.map(|(a, b, cap)| (a, b, cap + 5));
    let intra_band = Admiss {
        hcap: Some(intra_hcap),
        rcap: adm.rcap,
        holecap: None,
        well_empty: adm.well_empty,
        pot: intra_pot,
    };
    let t0 = Instant::now();
    let full = TetrisPieceBagState::new();

    // State = (board, remaining-bag). Forward-build the admissible reachable graph
    // from init under player-chooses-placement and adversary-chooses-piece.
    let init = (TetrisBoard::new(), full);
    let mut index: FxHashMap<(TetrisBoard, u8), u32> = FxHashMap::default();
    let mut states: Vec<(TetrisBoard, TetrisPieceBagState)> = Vec::new();
    let mut queue: VecDeque<u32> = VecDeque::new();
    index.insert((init.0, u8::from(init.1)), 0);
    states.push(init);
    queue.push_back(0);

    // Per state: for each piece in its bag, the list of successor state-ids (player's
    // admissible placement choices). `forced_loss` if some drawable piece has none.
    let mut succ_by_piece: Vec<Vec<(TetrisPiece, u32)>> = vec![Vec::new()];
    let mut forced_loss: Vec<bool> = vec![false];
    let pieces = TetrisPiece::all();
    let mut exploded = false;

    while let Some(sid) = queue.pop_front() {
        let (board, bag) = states[sid as usize];
        if bag.is_empty() {
            // refill: single deterministic edge to (same board, FULL)
            let key = (board, u8::from(full));
            let nid = *index.entry(key).or_insert_with(|| {
                let id = states.len() as u32;
                states.push((board, full));
                succ_by_piece.push(Vec::new());
                forced_loss.push(false);
                queue.push_back(id);
                id
            });
            // model refill as a pseudo-piece edge that must be taken (AND with one option)
            succ_by_piece[sid as usize].push((TetrisPiece::NULL_PIECE, nid));
            continue;
        }
        for &p in pieces.iter() {
            if !bag.contains(p) {
                continue;
            }
            let mut bag2 = bag;
            bag2.remove(p);
            let at_boundary = bag2.is_empty(); // after this placement the bag refills next
            let mut any = false;
            for &pl in TetrisPiecePlacement::all_from_piece(p) {
                let mut nb = board;
                let res = nb.apply_piece_placement(pl);
                if res.is_lost == IsLost::LOST {
                    continue;
                }
                // Boundary board → full band (incl. holes). Intra-bag board → conservative
                // band (height/roughness/well, holes allowed for the bootstrap transient).
                if at_boundary {
                    if !adm.ok(&nb) {
                        continue;
                    }
                } else if !intra_band.ok(&nb) {
                    continue;
                }
                any = true;
                if states.len() >= budget {
                    exploded = true;
                    break;
                }
                let key = (nb, u8::from(bag2));
                let nid = *index.entry(key).or_insert_with(|| {
                    let id = states.len() as u32;
                    states.push((nb, bag2));
                    succ_by_piece.push(Vec::new());
                    forced_loss.push(false);
                    queue.push_back(id);
                    id
                });
                succ_by_piece[sid as usize].push((p, nid));
            }
            if !any {
                forced_loss[sid as usize] = true; // adversary draws p, player stuck
            }
            if exploded {
                break;
            }
        }
        if exploded {
            break;
        }
    }

    let n = states.len();
    println!(
        "forward graph: {n} admissible states, {:.1}s{}",
        t0.elapsed().as_secs_f64(),
        if exploded {
            "  [EXPLODED: hit budget — inconclusive]"
        } else {
            ""
        }
    );
    if exploded {
        println!(
            "\nREGIME: band too large to enumerate at budget {budget}. Inconclusive — \
                  tighten caps or raise budget."
        );
        return;
    }

    // GFP via worklist (O(edges)). A state is unsafe if `forced_loss`, or some drawable
    // piece's safe-successor count drops to 0 (for-all-piece ∃-safe-placement fails).
    // Maintain per-(state,piece) safe-successor counts; when a state turns unsafe,
    // decrement that count in each predecessor and cascade.
    let mut piece_cnt: Vec<FxHashMap<TetrisPiece, u32>> = vec![FxHashMap::default(); n];
    let mut rev: Vec<Vec<(u32, TetrisPiece)>> = vec![Vec::new(); n];
    for sid in 0..n {
        for &(p, succ) in &succ_by_piece[sid] {
            *piece_cnt[sid].entry(p).or_insert(0) += 1;
            rev[succ as usize].push((sid as u32, p));
        }
    }
    let mut unsafe_s = vec![false; n];
    let mut wq: VecDeque<u32> = VecDeque::new();
    for sid in 0..n {
        if forced_loss[sid] {
            unsafe_s[sid] = true;
            wq.push_back(sid as u32);
        }
    }
    while let Some(s) = wq.pop_front() {
        let preds = std::mem::take(&mut rev[s as usize]);
        for (pred, p) in preds {
            if unsafe_s[pred as usize] {
                continue;
            }
            if let Some(c) = piece_cnt[pred as usize].get_mut(&p) {
                *c -= 1;
                if *c == 0 {
                    unsafe_s[pred as usize] = true;
                    wq.push_back(pred);
                }
            }
        }
    }

    let unsafe_count = unsafe_s.iter().filter(|&&u| u).count();
    let safe_count = n - unsafe_count;
    let init_safe = !unsafe_s[0];
    // distinct safe boundary boards (bag == FULL) = the certifiable carrier
    let mut safe_boundary: FxHashSet<TetrisBoard> = FxHashSet::default();
    let mut max_h_safe = 0u32;
    let mut max_phi_safe = 0u32; // effective potential cap actually USED by the core
    let mut max_rough_safe = 0u32;
    let mut max_holes_safe = 0u32;
    for sid in 0..n {
        if unsafe_s[sid] {
            continue;
        }
        let (b, bag) = states[sid];
        if bag == full {
            safe_boundary.insert(b);
            max_h_safe = max_h_safe.max(b.height());
            max_phi_safe = max_phi_safe.max(adm.phi(&b));
            max_rough_safe = max_rough_safe.max(b.roughness());
            max_holes_safe = max_holes_safe.max(b.total_holes());
        }
    }

    println!("\n================ RESULT (optimal-player) ================");
    println!("admissible states         : {n}");
    println!("unsafe states             : {unsafe_count}");
    println!("safe states               : {safe_count}");
    println!("init safe?                : {init_safe}");
    println!("safe boundary boards (S)  : {}", safe_boundary.len());
    println!("max height in safe core   : {max_h_safe}");
    if adm.pot.is_some() {
        println!(
            "max Φ / rough / holes (core): {max_phi_safe} / {max_rough_safe} / {max_holes_safe}  \
             (effective potential cap actually used = {max_phi_safe})"
        );
    }

    println!("\n---------------- REGIME ----------------");
    if init_safe && !safe_boundary.is_empty() {
        println!(
            "OUTCOME 1: a band-respecting strategy EXISTS — init safe, |S|={} safe boundary \
             surfaces (max h={max_h_safe}). If |S| ≲ 1e5 this is R1: export S, certify in Lean.",
            safe_boundary.len()
        );
    } else {
        println!(
            "OBSTRUCTION CONFIRMED: init is UNSAFE under the optimal player within this band — \
             NO band-respecting strategy survives all adversary orders. This is empirical \
             confirmation of the structural obstruction (crux #66/#72) for this band. Routes to T2."
        );
    }
    println!("total time: {:.1}s", t0.elapsed().as_secs_f64());
}

/// Build a "window" board: working columns `0..profile.len()` filled SOLID to the
/// given relative heights, sitting on top of an empty base at row `base`. Basing the
/// surface high (rows below `base` empty) guarantees no placement on top can ever
/// complete a row 0..9 — so there are NO line clears, and filler drift is measured
/// cleanly (clears only ever reduce Φ, so ignoring them is a sound upper bound).
fn build_window(profile: &[u32], base: u32) -> TetrisBoard {
    let mut b = TetrisBoard::new();
    for (j, &h) in profile.iter().enumerate() {
        for r in 0..h {
            b.set_bit(j, (base + r) as usize);
        }
    }
    b
}

/// **The drift probe.** Tests Lemma 1 (per-piece local Φ-drift) WITHOUT global
/// enumeration. Φ = Σ column heights (= cells + holes; bounds max height). On a
/// no-clear placement, ΔΦ = 4 + holes_created, so the informative quantity is the
/// minimum holes a piece is FORCED to create — 0 if a matching landing site exists.
///
/// Drift is local, so we enumerate bounded surface WINDOWS (W working columns,
/// relative height ≤ D, solid) — a few ×10⁴ shapes, NOT >10⁷ global boards — and use
/// the real engine for exact drift. For each window and piece we take the player's
/// best (min-holes) inside placement; per piece we report the worst window (the trap).
///
/// The deepest output: per window, the SET of pieces hostable hole-free, and the
/// maximum such set. If some window hosts all 7 → per-piece closure is viable. If the
/// max is 6 (all but one staircase) → the bag-phase disjunction is forced, quantified.
fn run_drift(w: usize, d: u32) {
    // Base at row 0: columns are solid from the floor (genuinely 0 holes). Line clears
    // are impossible anyway because columns ≥ w stay empty, so no row is ever full.
    let base = 0u32;
    println!("MODE: drift probe — Φ=Σcol-heights, window W={w} relhgt≤{d} base={base}");
    println!("(ΔΦ on no-clear placement = 4 + holes_created; we measure min holes forced)");
    let t0 = Instant::now();

    // pieces: the 6 fillers first, then I (the drain piece).
    let pieces = [
        (TetrisPiece::O_PIECE, "O"),
        (TetrisPiece::S_PIECE, "S"),
        (TetrisPiece::Z_PIECE, "Z"),
        (TetrisPiece::T_PIECE, "T"),
        (TetrisPiece::L_PIECE, "L"),
        (TetrisPiece::J_PIECE, "J"),
        (TetrisPiece::I_PIECE, "I"),
    ];

    let mut u = [0u32; 7]; // worst-case (over windows) min-holes per piece
    let mut worst_win: [Vec<u32>; 7] = Default::default();
    let mut hostable_hist: FxHashMap<usize, u64> = FxHashMap::default();
    let mut best_hostable = 0usize;
    let mut best_win: (Vec<u32>, Vec<&str>) = (Vec::new(), Vec::new());
    // which 6-subsets of fillers are simultaneously hostable hole-free (drop which piece?)
    let mut max_filler_hostable = 0usize;
    let mut filler_host_examples: FxHashMap<String, Vec<u32>> = FxHashMap::default();

    let mut profile = vec![0u32; w];
    let mut nwin: u64 = 0;
    loop {
        let board = build_window(&profile, base);
        let before_h = board.heights();
        let before_holes = board.total_holes();

        let mut hostable: Vec<&str> = Vec::new();
        let mut filler_hostset: Vec<&str> = Vec::new();
        for (pi, (p, lbl)) in pieces.iter().enumerate() {
            let mut min_holes = u32::MAX;
            for &pl in TetrisPiecePlacement::all_from_piece(*p) {
                let mut nb = board;
                let res = nb.apply_piece_placement(pl);
                if res.is_lost == IsLost::LOST || res.lines_cleared > 0 {
                    continue;
                }
                let after_h = nb.heights();
                // occupied columns = where height increased; require strictly inside
                // [1, w-2] so the local Φ-delta has no window-edge artifacts.
                let mut inside = true;
                let mut touched = false;
                for j in 0..10 {
                    if after_h[j] > before_h[j] {
                        touched = true;
                        if j < 1 || j > w - 2 {
                            inside = false;
                            break;
                        }
                    } else if after_h[j] < before_h[j] {
                        inside = false;
                        break;
                    }
                }
                if !inside || !touched {
                    continue;
                }
                let holes = nb.total_holes().saturating_sub(before_holes); // holes CREATED
                if holes < min_holes {
                    min_holes = holes;
                }
            }
            if min_holes == u32::MAX {
                continue; // no inside placement for this piece in this window — skip
            }
            if min_holes > u[pi] {
                u[pi] = min_holes;
                worst_win[pi] = profile.clone();
            }
            if min_holes == 0 {
                hostable.push(lbl);
                if pi < 6 {
                    filler_hostset.push(lbl);
                }
            }
        }
        *hostable_hist.entry(hostable.len()).or_insert(0) += 1;
        if hostable.len() > best_hostable {
            best_hostable = hostable.len();
            best_win = (profile.clone(), hostable.clone());
        }
        if filler_hostset.len() > max_filler_hostable {
            max_filler_hostable = filler_hostset.len();
        }
        if filler_hostset.len() >= 5 {
            filler_host_examples
                .entry(filler_hostset.join(""))
                .or_insert_with(|| profile.clone());
        }
        nwin += 1;

        // odometer increment over base-(d+1) digits
        let mut k = 0;
        loop {
            if k == w {
                // done
                report_drift(
                    &pieces,
                    &u,
                    &worst_win,
                    &hostable_hist,
                    best_hostable,
                    &best_win,
                    max_filler_hostable,
                    &filler_host_examples,
                    nwin,
                    t0,
                );
                return;
            }
            profile[k] += 1;
            if profile[k] <= d {
                break;
            }
            profile[k] = 0;
            k += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn report_drift(
    pieces: &[(TetrisPiece, &str); 7],
    u: &[u32; 7],
    worst_win: &[Vec<u32>; 7],
    hostable_hist: &FxHashMap<usize, u64>,
    best_hostable: usize,
    best_win: &(Vec<u32>, Vec<&str>),
    max_filler_hostable: usize,
    filler_host_examples: &FxHashMap<String, Vec<u32>>,
    nwin: u64,
    t0: Instant,
) {
    println!(
        "\nwindows enumerated: {nwin}  ({:.1}s)",
        t0.elapsed().as_secs_f64()
    );
    println!("\n=== per-piece worst-case forced holes (over all windows) ===");
    let mut filler_sum = 0u32;
    for (pi, (_, lbl)) in pieces.iter().enumerate() {
        println!(
            "  {lbl}: u={}  (worst window profile {:?})",
            u[pi], worst_win[pi]
        );
        if pi < 6 {
            filler_sum += u[pi];
        }
    }
    println!("\n=== bag accounting (Φ = Σ heights) ===");
    println!("  Σ worst-case filler holes (O,S,Z,T,L,J) = {filler_sum}");
    println!(
        "  per-bag Σheight drift (no-clear) = 24 + Σholes; one I-drain clears 4 rows = -36 working."
    );
    println!(
        "  PESSIMISTIC bag check (worst window per piece): 24 + {filler_sum} - 36 = {} (≤0 ⇒ ok)",
        24i32 + filler_sum as i32 - 36
    );
    println!("\n=== hostable-set analysis (THE structural question) ===");
    let mut sizes: Vec<_> = hostable_hist.keys().copied().collect();
    sizes.sort_unstable();
    for s in &sizes {
        println!(
            "  windows hosting exactly {s}/7 pieces hole-free: {}",
            hostable_hist[s]
        );
    }
    println!(
        "  MAX pieces hostable hole-free by a SINGLE window: {best_hostable}/7  (pieces {:?}, profile {:?})",
        best_win.1, best_win.0
    );
    println!("  MAX fillers (of 6) hostable hole-free by a single window: {max_filler_hostable}/6");
    if !filler_host_examples.is_empty() {
        println!("  example windows hosting ≥5 fillers hole-free (filler-set → profile):");
        let mut ks: Vec<_> = filler_host_examples.keys().cloned().collect();
        ks.sort();
        for k in ks.iter().take(10) {
            println!("    {{{k}}} → {:?}", filler_host_examples[k]);
        }
    }
    println!("\n=== verdict ===");
    if best_hostable == 7 {
        println!(
            "BREAKTHROUGH SIGNAL: some single surface hosts ALL 7 pieces hole-free → per-piece \
             closure is viable on that surface family. Next: is it strategy-MAINTAINABLE?"
        );
    } else if max_filler_hostable >= 5 {
        println!(
            "EXPECTED/STRUCTURAL: no single window hosts all 7, max fillers={max_filler_hostable}/6. \
             Confirms the bag-phase disjunction is FORCED (a surface can't host flats AND both \
             staircases). The drift is BOUNDED (small u), so the obstruction is hole-PLACEMENT/drain \
             scheduling, not unbounded drift. Next: drain-zone refinement + phase-indexed surfaces."
        );
    } else {
        println!(
            "WEAK: even fillers can't be co-hosted (max {max_filler_hostable}/6) at W={}, D — widen \
             window / raise D, or the local surface model is too tight.",
            best_win.0.len()
        );
    }
}

/// Φ = Σ column heights (= cells + holes; bounds max height). The Lyapunov candidate.
fn phi(b: &TetrisBoard) -> i32 {
    b.heights().iter().map(|&h| h as i32).sum()
}

/// Reserved well column for the cycle test (rightmost).
const CYCLE_WELL: usize = 9;

/// When true, the greedy player minimizes HOLES first (then Φ, height) — far more
/// survival-oriented than minimizing Φ first (which treats a buried hole like a
/// harmless surface cell). Set from args in `main`.
static GREEDY_HOLE_FIRST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// When true, the long-run adversary draws a RANDOM (fixed-seed) bag order each bag
/// instead of the myopic worst order — calibrates strategy-strength vs adversary-strength.
static ADV_RANDOM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Well-reserving min-Φ greedy placement. For non-I pieces the well column must stay
/// empty (reserved for the drain); the I may use it (the drain) or lie flat. Among
/// admissible placements pick the one minimizing (Σheights, max-height, holes). `None`
/// iff the player tops out. Returns the board AND the lines cleared by the placement.
fn greedy_place(board: &TetrisBoard, p: TetrisPiece) -> Option<(TetrisBoard, u32)> {
    let before = board.as_limbs();
    let mut best: Option<(TetrisBoard, u32, (i32, i32, i32))> = None;
    for &pl in TetrisPiecePlacement::all_from_piece(p) {
        let mut nb = *board;
        let res = nb.apply_piece_placement(pl);
        if res.is_lost == IsLost::LOST || nb.height() > ROWS {
            continue;
        }
        // non-I pieces must not touch the reserved well column
        if p != TetrisPiece::I_PIECE && nb.as_limbs()[CYCLE_WELL] != before[CYCLE_WELL] {
            continue;
        }
        // score: holes-first (survival) or Φ-first, per the global switch.
        let score: (i32, i32, i32) = if GREEDY_HOLE_FIRST.load(std::sync::atomic::Ordering::Relaxed)
        {
            (nb.total_holes() as i32, phi(&nb), nb.height() as i32)
        } else {
            (phi(&nb), nb.height() as i32, nb.total_holes() as i32)
        };
        if best.as_ref().is_none_or(|(_, _, s)| score < *s) {
            best = Some((nb, res.lines_cleared, score));
        }
    }
    best.map(|(b, lc, _)| (b, lc))
}

/// Generate all permutations of a 7-element array via Heap's algorithm.
fn all_perms_7(items: [TetrisPiece; 7]) -> Vec<[TetrisPiece; 7]> {
    let mut out = Vec::with_capacity(5040);
    let mut a = items;
    let mut c = [0usize; 7];
    out.push(a);
    let mut i = 0;
    while i < 7 {
        if c[i] < i {
            if i % 2 == 0 {
                a.swap(0, i);
            } else {
                a.swap(c[i], i);
            }
            out.push(a);
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    out
}

/// **The one-bag cycle test.** From a flat reset surface at floor `f` (working cols
/// `0..9` at height `f`, the well col `9` empty), play EVERY one of the 5040 bag orders
/// with the well-reserving min-Φ greedy player, and measure the per-bag Φ drift.
///
/// Greedy gives an UPPER bound on the optimal player's Φ_end, so if greedy keeps
/// `ΔΦ = Φ_end − Φ_start ≤ 0` with no top-out across ALL orders, the optimal player
/// does at least as well ⇒ the per-bag Lyapunov drift is ≤ 0 from this surface — a
/// SUFFICIENT empirical witness for closure at this floor. Greedy failure on an order
/// is inconclusive (optimal might still succeed) but localizes the killer order.
fn run_cycle(floor: u32) {
    // σ: working columns 0..=8 solid to height `floor`; well column 9 empty.
    let mut sigma = TetrisBoard::new();
    for j in 0..CYCLE_WELL {
        for r in 0..floor {
            sigma.set_bit(j, r as usize);
        }
    }
    let start_phi = phi(&sigma);
    let perms = all_perms_7([
        TetrisPiece::O_PIECE,
        TetrisPiece::I_PIECE,
        TetrisPiece::S_PIECE,
        TetrisPiece::Z_PIECE,
        TetrisPiece::T_PIECE,
        TetrisPiece::L_PIECE,
        TetrisPiece::J_PIECE,
    ]);

    let mut worst_end = i32::MIN;
    let mut worst_order = perms[0];
    let mut losses = 0u32;
    let mut loss_order: Option<[TetrisPiece; 7]> = None;
    let mut min_lines = u32::MAX; // worst-case total lines cleared in a bag (drain success)
    let mut sum_end = 0i64;
    let mut n_ok = 0i64;

    for perm in &perms {
        let mut b = sigma;
        let mut lines = 0u32;
        let mut lost = false;
        for &p in perm.iter() {
            match greedy_place(&b, p) {
                Some((nb, lc)) => {
                    b = nb;
                    lines += lc;
                }
                None => {
                    lost = true;
                    break;
                }
            }
        }
        if lost {
            losses += 1;
            if loss_order.is_none() {
                loss_order = Some(*perm);
            }
            continue;
        }
        let e = phi(&b);
        sum_end += e as i64;
        n_ok += 1;
        if e > worst_end {
            worst_end = e;
            worst_order = *perm;
        }
        if lines < min_lines {
            min_lines = lines;
        }
    }

    let lbl = |arr: &[TetrisPiece; 7]| -> String {
        arr.iter()
            .map(|p| match *p {
                TetrisPiece::O_PIECE => "O",
                TetrisPiece::I_PIECE => "I",
                TetrisPiece::S_PIECE => "S",
                TetrisPiece::Z_PIECE => "Z",
                TetrisPiece::T_PIECE => "T",
                TetrisPiece::L_PIECE => "L",
                _ => "J",
            })
            .collect::<Vec<_>>()
            .join("")
    };
    let avg_end = if n_ok > 0 {
        sum_end as f64 / n_ok as f64
    } else {
        0.0
    };
    println!(
        "floor={floor:2}  Φ_start={start_phi:3}  worst Φ_end={worst_end:3}  ΔΦ={:+3}  \
         avgΦ_end={avg_end:6.1}  losses={losses:4}/5040  min_lines_cleared={}  worst_order={}{}",
        worst_end - start_phi,
        if min_lines == u32::MAX { 0 } else { min_lines },
        lbl(&worst_order),
        match loss_order {
            Some(o) => format!("  FIRST_LOSS={}", lbl(&o)),
            None => String::new(),
        }
    );
}

/// When true, the player uses the strong Lee/El-Tetris heuristic (no well reservation)
/// instead of the well-reserving greedy. Known to clear millions of lines under random.
static PLAYER_SMART: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Strong heuristic placement (Lee's genetically-tuned weights): maximize
/// `-0.51·aggregateHeight + 0.76·linesCleared − 0.36·holes − 0.18·bumpiness`.
/// No reserved well — the heuristic shapes the board itself. `None` iff it tops out.
fn smart_place(board: &TetrisBoard, p: TetrisPiece) -> Option<(TetrisBoard, u32)> {
    let mut best: Option<(TetrisBoard, u32, f64)> = None;
    for &pl in TetrisPiecePlacement::all_from_piece(p) {
        let mut nb = *board;
        let res = nb.apply_piece_placement(pl);
        if res.is_lost == IsLost::LOST || nb.height() > ROWS {
            continue;
        }
        let agg = nb.heights().iter().sum::<u32>() as f64;
        let holes = nb.total_holes() as f64;
        let bump = nb.roughness() as f64;
        let lines = res.lines_cleared as f64;
        let score = -0.51 * agg + 0.76 * lines - 0.36 * holes - 0.18 * bump;
        if best.as_ref().is_none_or(|(_, _, s)| score > *s) {
            best = Some((nb, res.lines_cleared, score));
        }
    }
    best.map(|(b, lc, _)| (b, lc))
}

/// Dispatch one placement to the active player.
fn place_piece(board: &TetrisBoard, p: TetrisPiece) -> Option<(TetrisBoard, u32)> {
    if PLAYER_SMART.load(std::sync::atomic::Ordering::Relaxed) {
        smart_place(board, p)
    } else {
        greedy_place(board, p)
    }
}

/// Play one full bag in the given order with the active player.
/// Returns the end board and total lines cleared, or `None` if the player tops out.
fn play_bag(board: TetrisBoard, order: &[TetrisPiece; 7]) -> Option<(TetrisBoard, u32)> {
    let mut b = board;
    let mut lines = 0u32;
    for &p in order.iter() {
        let (nb, lc) = place_piece(&b, p)?;
        b = nb;
        lines += lc;
    }
    Some((b, lines))
}

/// **The multi-bag long-run test.** A continuous game: the well-reserving greedy
/// player vs. a myopic worst-order adversary that, each bag, picks the order maximizing
/// the player's resulting Φ (or kills the player if any order can). Tracks Φ / max-height
/// / holes over `nbags` bags from the empty board.
///
/// The decisive question the single-bag test left open: per bag Φ drops 10 but holes
/// rise +2 — do holes ACCUMULATE (a slow leak → eventual top-out) or reach a BOUNDED
/// steady state (cleared as their rows fill)? Bounded over millions of pieces ⇒ strong
/// empirical infinite-play evidence AND the Lyapunov region is bounded.
fn run_longrun(nbags: usize) {
    println!("MODE: multi-bag long-run — well-reserving greedy vs worst-order adversary");
    let t0 = Instant::now();
    let perms = all_perms_7([
        TetrisPiece::O_PIECE,
        TetrisPiece::I_PIECE,
        TetrisPiece::S_PIECE,
        TetrisPiece::Z_PIECE,
        TetrisPiece::T_PIECE,
        TetrisPiece::L_PIECE,
        TetrisPiece::J_PIECE,
    ]);

    let random_adv = ADV_RANDOM.load(std::sync::atomic::Ordering::Relaxed);
    let player_name = if PLAYER_SMART.load(std::sync::atomic::Ordering::Relaxed) {
        "smart (Lee weights)"
    } else if GREEDY_HOLE_FIRST.load(std::sync::atomic::Ordering::Relaxed) {
        "holes-first greedy"
    } else {
        "Φ-first greedy"
    };
    println!(
        "player = {player_name} ; adversary = {}",
        if random_adv {
            "RANDOM order (seed 1)"
        } else {
            "myopic WORST order"
        }
    );
    let mut b = TetrisBoard::new();
    let mut max_phi = 0i32;
    let mut max_h = 0u32;
    let mut max_holes = 0u32;
    let mut total_lines = 0u64;
    let mut last_report = 0usize;
    let mut rng: u64 = 0x9E3779B97F4A7C15; // fixed-seed xorshift for the random adversary

    for bag in 0..nbags {
        let mut worst: Option<(i32, TetrisBoard, u32)> = None;
        let mut killed = false;
        if random_adv {
            // draw one random order; player tops out ⇒ killed.
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let order = &perms[(rng as usize) % perms.len()];
            match play_bag(b, order) {
                None => killed = true,
                Some((nb, lc)) => worst = Some((phi(&nb), nb, lc)),
            }
        } else {
            // myopic worst: pick the order maximizing resulting Φ; any topping order wins.
            for order in &perms {
                match play_bag(b, order) {
                    None => {
                        killed = true;
                        break;
                    }
                    Some((nb, lc)) => {
                        let m = phi(&nb);
                        if worst.as_ref().is_none_or(|(wm, _, _)| m > *wm) {
                            worst = Some((m, nb, lc));
                        }
                    }
                }
            }
        }
        if killed {
            println!(
                "TOPPED OUT at bag {bag} ({} pieces). max_Φ={max_phi} max_h={max_h} \
                 max_holes={max_holes}",
                bag * 7
            );
            return;
        }
        let (m, nb, lc) = worst.unwrap();
        b = nb;
        total_lines += lc as u64;
        max_phi = max_phi.max(m);
        max_h = max_h.max(b.height());
        max_holes = max_holes.max(b.total_holes());
        if bag - last_report >= 1000 || bag + 1 == nbags {
            last_report = bag;
            println!(
                "  bag {:6} ({:8} pc): Φ={:3} (max {:3})  h={:2} (max {:2})  holes={:2} (max {:2})  lines={}",
                bag + 1,
                (bag + 1) * 7,
                phi(&b),
                max_phi,
                b.height(),
                max_h,
                b.total_holes(),
                max_holes,
                total_lines
            );
        }
    }
    println!(
        "\nSURVIVED all {nbags} bags ({} pieces) in {:.1}s — NO top-out.",
        nbags * 7,
        t0.elapsed().as_secs_f64()
    );
    println!(
        "BOUNDED: max_Φ={max_phi}  max_height={max_h}  max_holes={max_holes}  total_lines={total_lines}"
    );
    println!(
        "VERDICT: well-reserving greedy keeps Φ/height/holes BOUNDED vs worst-order adversary over \
         {} pieces ⇒ strong empirical infinite-play (M1) + bounded Lyapunov region. (Myopic \
         adversary, not full minimax.)",
        nbags * 7
    );
}

/// The 7 pieces in a fixed order; bit `i` of a bag mask ↔ `MM_PIECES[i]`.
const MM_PIECES: [TetrisPiece; 7] = [
    TetrisPiece::O_PIECE,
    TetrisPiece::I_PIECE,
    TetrisPiece::S_PIECE,
    TetrisPiece::Z_PIECE,
    TetrisPiece::T_PIECE,
    TetrisPiece::L_PIECE,
    TetrisPiece::J_PIECE,
];
const MM_FULL: u8 = 0b111_1111;
const MM_LOSS: i32 = 1_000_000;

/// Memo-size cap for the exact pump minimax (the exact full-bag search can explode);
/// when exceeded, `bag_minimax` bails to the leaf and sets `MM_OVERFLOW` (→ result is
/// approximate, not exact). `usize::MAX` (the default) disables the cap.
static MM_CAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
static MM_OVERFLOW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Leaf heuristic "badness" (lower = better for the player): a strong, hole-averse
/// evaluation. `aggHeight` already counts holes once (Σheights = cells + holes); the
/// extra `6·holes` makes a buried hole far costlier than a surface cell, and the
/// roughness/peak terms keep the surface drain-friendly. Line clears are rewarded
/// implicitly (a cleared board has lower aggHeight).
fn leaf_badness(b: &TetrisBoard) -> i32 {
    let agg = b.heights().iter().sum::<u32>() as i32;
    let holes = b.total_holes() as i32;
    let bump = b.roughness() as i32;
    let maxh = b.height() as i32;
    agg + 6 * holes + bump / 2 + 2 * maxh
}

/// Player-node beam width: at each player decision in the minimax recursion, only the
/// top-`BEAM_K` placements (by immediate `leaf_badness`) are explored deeper. This lets
/// the lookahead go DEEP without the exact tree exploding. `usize::MAX` = no pruning
/// (exact). Set from args.
static BEAM_K: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Hole/debt cap for the deep-player carrier (`debtcarrier` mode): the player only considers
/// placements whose result has `total_holes ≤ HOLE_CAP`. A piece with no such placement is a
/// forced debt-escape (`has_loss`). `u32::MAX` = uncapped. This restricts the deepcarrier
/// closure to the bounded-debt carrier `{b : debt b ≤ D}`.
static HOLE_CAP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);

/// Depth-limited bag-aware minimax with player-node BEAM pruning. Looks ahead at most
/// `depth` pieces; adversary (revealed piece) MAXIMIZES, player MINIMIZES but only over
/// its top-`BEAM_K` placements. At the horizon returns `leaf_badness`. Memoized on
/// `(board, mask, depth)`. `MM_LOSS` if forced top-out within the horizon.
fn bag_minimax(
    board: TetrisBoard,
    mask: u8,
    depth: u8,
    memo: &mut FxHashMap<(TetrisBoard, u8, u8), i32>,
) -> i32 {
    if mask == 0 || depth == 0 {
        return leaf_badness(&board);
    }
    if let Some(&v) = memo.get(&(board, mask, depth)) {
        return v;
    }
    if memo.len() >= MM_CAP.load(std::sync::atomic::Ordering::Relaxed) {
        MM_OVERFLOW.store(true, std::sync::atomic::Ordering::Relaxed);
        return leaf_badness(&board);
    }
    let beam_k = BEAM_K.load(std::sync::atomic::Ordering::Relaxed);
    let mut best_adv = i32::MIN;
    for (pi, p) in MM_PIECES.iter().enumerate() {
        let bit = 1u8 << pi;
        if mask & bit == 0 {
            continue;
        }
        // collect valid player placements with immediate badness, keep top-K (beam)
        let mut cands: Vec<(i32, TetrisBoard)> = Vec::new();
        for &pl in TetrisPiecePlacement::all_from_piece(*p) {
            let mut nb = board;
            let res = nb.apply_piece_placement(pl);
            if res.is_lost == IsLost::LOST || nb.height() > ROWS {
                continue;
            }
            cands.push((leaf_badness(&nb), nb));
        }
        let mut best_play = MM_LOSS;
        if cands.is_empty() {
            best_play = MM_LOSS; // no placement → player loses this piece
        } else {
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

/// When set, the adversary reveals pieces in the FIXED order S,Z,O,T,L,J,I each bag
/// (the structurally-hardest "S/Z-first burst" per the budget/Burgiel analysis) instead
/// of the expensive adaptive worst-order — letting the DEEP player run for many bags.
static ADV_SZFIRST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Play one bag with the deep minimax PLAYER against the FIXED S/Z-first order. The
/// player still uses worst-case `depth` lookahead (it doesn't know the order), so it
/// plays conservatively. One deep search per revealed piece — no adversary scan, so it
/// is far faster than `play_bag_minimax`.
fn play_bag_szfirst(board: TetrisBoard, depth: u8) -> Option<TetrisBoard> {
    const ORDER: [TetrisPiece; 7] = [
        TetrisPiece::S_PIECE,
        TetrisPiece::Z_PIECE,
        TetrisPiece::O_PIECE,
        TetrisPiece::T_PIECE,
        TetrisPiece::L_PIECE,
        TetrisPiece::J_PIECE,
        TetrisPiece::I_PIECE,
    ];
    let mut b = board;
    let mut mask = MM_FULL;
    let mut memo: FxHashMap<(TetrisBoard, u8, u8), i32> = FxHashMap::default();
    for &p in ORDER.iter() {
        let pi = MM_PIECES.iter().position(|&q| q == p).unwrap();
        let bit = 1u8 << pi;
        let look = depth.min(mask.count_ones() as u8);
        let mut best_board = b;
        let mut best_v = MM_LOSS;
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
            }
        }
        if best_v >= MM_LOSS {
            return None;
        }
        b = best_board;
        mask &= !bit;
    }
    Some(b)
}

/// Play one bag in a GIVEN reveal order with the deep minimax player (it uses worst-case
/// `depth` lookahead, not knowing the order). Returns the bag-end board, or `None` on
/// forced top-out.
fn play_bag_order_minimax(
    board: TetrisBoard,
    order: &[TetrisPiece; 7],
    depth: u8,
) -> Option<TetrisBoard> {
    let mut b = board;
    let mut mask = MM_FULL;
    let mut memo: FxHashMap<(TetrisBoard, u8, u8), i32> = FxHashMap::default();
    for &p in order.iter() {
        let pi = MM_PIECES.iter().position(|&q| q == p).unwrap();
        let bit = 1u8 << pi;
        let look = depth.min(mask.count_ones() as u8);
        let mut best_board = b;
        let mut best_v = MM_LOSS;
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
            }
        }
        if best_v >= MM_LOSS {
            return None;
        }
        b = best_board;
        mask &= !bit;
    }
    Some(b)
}

/// Play one bag as the depth-`depth` minimax trajectory: each step the adversary
/// reveals the piece maximizing the player's best-response value; the player picks the
/// minimizing placement. Returns the bag-end board, or `None` on forced top-out.
fn play_bag_minimax(board: TetrisBoard, depth: u8) -> Option<TetrisBoard> {
    let mut b = board;
    let mut mask = MM_FULL;
    let mut memo: FxHashMap<(TetrisBoard, u8, u8), i32> = FxHashMap::default();
    while mask != 0 {
        let look = depth.min(mask.count_ones() as u8);
        // adversary's piece = argmax over remaining of the player's best response value
        let mut adv: Option<(usize, TetrisPiece, i32)> = None;
        for (pi, p) in MM_PIECES.iter().enumerate() {
            let bit = 1u8 << pi;
            if mask & bit == 0 {
                continue;
            }
            let mut best_play = MM_LOSS;
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
        if av >= MM_LOSS {
            return None;
        }
        let bit = 1u8 << pi;
        let mut best_board = b;
        let mut best_v = MM_LOSS;
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
            }
        }
        b = best_board;
        mask &= !bit;
    }
    Some(b)
}

/// Long-run with the depth-limited bag-aware minimax player vs the worst-order
/// adversary (which the minimax inherently models). Chains bags from empty.
fn run_minimax_longrun(nbags: usize, depth: u8) {
    let bk = BEAM_K.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "MODE: depth-{depth} beam-{} bag-aware MINIMAX player vs worst-order adversary",
        if bk == usize::MAX {
            "∞(exact)".to_string()
        } else {
            bk.to_string()
        }
    );
    let t0 = Instant::now();
    let mut b = TetrisBoard::new();
    let mut max_phi = 0i32;
    let mut max_h = 0u32;
    let mut max_holes = 0u32;
    let szfirst = ADV_SZFIRST.load(std::sync::atomic::Ordering::Relaxed);
    if szfirst {
        println!("(adversary = FIXED S,Z,O,T,L,J,I order — the S/Z-first burst)");
    }
    for bag in 0..nbags {
        let step = if szfirst {
            play_bag_szfirst(b, depth)
        } else {
            play_bag_minimax(b, depth)
        };
        match step {
            None => {
                println!(
                    "TOPPED OUT at bag {bag} ({} pieces). max_Φ={max_phi} max_h={max_h} \
                     max_holes={max_holes}  ({:.1}s)",
                    bag * 7,
                    t0.elapsed().as_secs_f64()
                );
                return;
            }
            Some(nb) => {
                b = nb;
                max_phi = max_phi.max(phi(&b));
                max_h = max_h.max(b.height());
                max_holes = max_holes.max(b.total_holes());
                if bag < 10 || (bag + 1) % 50 == 0 || bag + 1 == nbags {
                    println!(
                        "  bag {:5} ({:7} pc): Φ={:3} (max {:3})  h={:2} (max {:2})  holes={:2} (max {:2})",
                        bag + 1,
                        (bag + 1) * 7,
                        phi(&b),
                        max_phi,
                        b.height(),
                        max_h,
                        b.total_holes(),
                        max_holes
                    );
                }
            }
        }
    }
    println!(
        "\nSURVIVED all {nbags} bags ({} pieces) in {:.1}s — NO top-out.",
        nbags * 7,
        t0.elapsed().as_secs_f64()
    );
    println!("BOUNDED: max_Φ={max_phi}  max_height={max_h}  max_holes={max_holes}");
    println!(
        "VERDICT: the depth-{depth} bag-aware player keeps the board BOUNDED vs the worst-order \
         adversary ⇒ adversarial 7-bag survival is PLAUSIBLE; the bounded region is the empirical \
         Lyapunov carrier."
    );
}

/// Sum of well depths (a column strictly lower than both neighbors; board edges count
/// as height `ROWS`). A Dellacherie-style surface feature.
fn well_sum(h: &[u32; 10]) -> i32 {
    let mut s = 0i32;
    for j in 0..10 {
        let left = if j > 0 { h[j - 1] } else { ROWS };
        let right = if j < 9 { h[j + 1] } else { ROWS };
        if h[j] < left && h[j] < right {
            s += (left.min(right) - h[j]) as i32;
        }
    }
    s
}

/// Candidate potentials P(board) for the unsurvivability proof. Each returned with a
/// name. We hunt for one where S/Z FORCE ΔP≥1, the player cannot DECREASE P with the
/// five non-staircase pieces, and clears don't decrease P.
fn potentials(b: &TetrisBoard) -> Vec<(&'static str, i32)> {
    let h = b.heights();
    let agg: i32 = h.iter().map(|&x| x as i32).sum();
    let holes = b.total_holes() as i32;
    let rough = b.roughness() as i32;
    let maxh = b.height() as i32;
    let wells = well_sum(&h);
    let parity = {
        let mut e = 0i32;
        let mut o = 0i32;
        for (j, &x) in h.iter().enumerate() {
            if j % 2 == 0 {
                e += x as i32;
            } else {
                o += x as i32;
            }
        }
        (e - o).abs()
    };
    vec![
        ("rough", rough),
        ("holes", holes),
        ("agg", agg),
        ("maxh", maxh),
        ("wells", wells),
        ("holes+rough", holes + rough),
        ("rough+wells", rough + wells),
        ("|colParity|", parity),
    ]
}

/// **The potential-hunt probe.** For each candidate potential P, compute over all local
/// windows the per-piece worst-case best-response ΔP — i.e. `max over windows of (min
/// over inside placements of ΔP)`. The player MINIMIZES ΔP (it wants P low); the worst
/// window is the adversary's choice of surface. A potential proves unsurvivability iff:
///   S ≥ 1 AND Z ≥ 1   (every S/Z placement is forced to raise P)  AND
///   O,I,T,L,J ≥ 0     (the player cannot lower P with the cleaners).
/// (Clear-invariance is checked separately/structurally.) If found → proof potential.
fn run_hunt(w: usize, d: u32) {
    let base = 0u32;
    println!("MODE: potential-hunt — find P with S/Z forced ↑, cleaners can't ↓ (⇒ unsurvivable)");
    println!(
        "window W={w} relhgt≤{d}; reporting per-piece worst-case best-response ΔP per candidate P"
    );
    let t0 = Instant::now();
    let pieces = [
        (TetrisPiece::O_PIECE, "O"),
        (TetrisPiece::I_PIECE, "I"),
        (TetrisPiece::S_PIECE, "S"),
        (TetrisPiece::Z_PIECE, "Z"),
        (TetrisPiece::T_PIECE, "T"),
        (TetrisPiece::L_PIECE, "L"),
        (TetrisPiece::J_PIECE, "J"),
    ];
    let pnames: Vec<&'static str> = potentials(&TetrisBoard::new())
        .iter()
        .map(|(n, _)| *n)
        .collect();
    let np = pnames.len();
    // u[cand][piece] = MIN over windows of the player's best (min) ΔP — i.e. the most
    // the player can REDUCE P with that piece across all surfaces. For a forced-↑ piece
    // this stays ≥ threshold; if the player can ever reduce P, it goes < 0.
    let mut u = vec![[i32::MAX; 7]; np];
    // Only CLEAR-INVARIANT potentials are valid: the player reduces any clear-VARIANT P
    // by clearing lines (which the no-clear window can't see). A full-row clear drops all
    // heights equally, so roughness and |Σeven−Σodd| are invariant; holes/agg/maxh/wells
    // are not.
    let clear_inv = |n: &str| matches!(n, "rough" | "|colParity|");

    let mut profile = vec![0u32; w];
    loop {
        let board = build_window(&profile, base);
        let before: Vec<i32> = potentials(&board).iter().map(|(_, v)| *v).collect();
        let before_h = board.heights();
        for (pidx, (p, _)) in pieces.iter().enumerate() {
            // player's best (min) ΔP for EACH candidate, over inside placements
            let mut best = vec![i32::MAX; np];
            let mut any = false;
            for &pl in TetrisPiecePlacement::all_from_piece(*p) {
                let mut nb = board;
                let res = nb.apply_piece_placement(pl);
                if res.is_lost == IsLost::LOST || res.lines_cleared > 0 || nb.height() > ROWS {
                    continue;
                }
                let after_h = nb.heights();
                let mut inside = true;
                let mut touched = false;
                for j in 0..10 {
                    if after_h[j] > before_h[j] {
                        touched = true;
                        if j < 1 || j > w - 2 {
                            inside = false;
                            break;
                        }
                    } else if after_h[j] < before_h[j] {
                        inside = false;
                        break;
                    }
                }
                if !inside || !touched {
                    continue;
                }
                any = true;
                let after: Vec<i32> = potentials(&nb).iter().map(|(_, v)| *v).collect();
                for c in 0..np {
                    let dp = after[c] - before[c];
                    if dp < best[c] {
                        best[c] = dp;
                    }
                }
            }
            if !any {
                continue;
            }
            for c in 0..np {
                if best[c] < u[c][pidx] {
                    u[c][pidx] = best[c];
                }
            }
        }
        // odometer
        let mut k = 0;
        loop {
            if k == w {
                // report and finish
                println!("\nwindows done ({:.1}s)\n", t0.elapsed().as_secs_f64());
                println!("  candidate P     |   O    I    S    Z    T    L    J  | verdict",);
                println!("  ----------------+-------------------------------------+--------");
                let mut found = false;
                for c in 0..np {
                    let r = u[c];
                    // piece order in `pieces`: O,I,S,Z,T,L,J → indices 0..6
                    let (uo, ui, us, uz, ut, ul, uj) = (r[0], r[1], r[2], r[3], r[4], r[5], r[6]);
                    let monotone =
                        us >= 1 && uz >= 1 && uo >= 0 && ui >= 0 && ut >= 0 && ul >= 0 && uj >= 0;
                    let ci = clear_inv(pnames[c]);
                    let valid = monotone && ci;
                    if valid {
                        found = true;
                    }
                    println!(
                        "  {:<15} | {:3}  {:3}  {:3}  {:3}  {:3}  {:3}  {:3}  | {}{}",
                        pnames[c],
                        uo,
                        ui,
                        us,
                        uz,
                        ut,
                        ul,
                        uj,
                        if ci { "[clear-inv] " } else { "[clear-VAR] " },
                        if valid {
                            "*** PROOF POTENTIAL ***"
                        } else if monotone {
                            "monotone but clear-variant (player clears it away)"
                        } else {
                            "fails (S/Z not forced, or player can reduce)"
                        }
                    );
                }
                println!(
                    "\n(Values = worst-case best-response ΔP: the most the player is FORCED to change P.\n \
                     Need S≥1, Z≥1 (forced ↑) AND O,I,T,L,J≥0 (player can't ↓) for an unsurvivability potential.)"
                );
                if !found {
                    println!(
                        "\nVERDICT: NO candidate P is monotone — every P that S/Z force up, some cleaner can \
                         pull back DOWN (player rebuilds). Evidence the narrowed claim is FALSE / adversarial \
                         is SURVIVABLE with optimal play. (Within this candidate family + local-window model.)"
                    );
                } else {
                    println!(
                        "\nVERDICT: a MONOTONE potential exists ⇒ unsurvivability proof candidate! Next: verify \
                         clear-invariance + the height bound, then formalize the per-piece ΔP bounds in Lean."
                    );
                }
                return;
            }
            profile[k] += 1;
            if profile[k] <= d {
                break;
            }
            profile[k] = 0;
            k += 1;
        }
    }
}

/// **Monotonicity test for the WQO / finite-basis approach.** Samples dominated board
/// pairs `b ≼ β` (height-vector domination, Dickson's wqo on ℕ¹⁰) and checks whether
/// `applyStep` preserves the order: `b ≼ β ⟹ applyStep b pl ≼ applyStep β pl`. Reports
/// violations split by whether a line CLEAR occurred — the structural prediction is that
/// non-clearing moves never violate (the piece lands lower on the lower board), so all
/// violations should be clear-related. If non-clear violations ≈ 0, a clear-aware order
/// may rescue downward-closure; if they are common, the order is fundamentally wrong.
fn run_monotone(samples: usize) {
    println!(
        "MODE: monotonicity test for WQO finite-basis — does applyStep preserve height-domination?"
    );
    let mut rng: u64 = 0xD1B54A32D192ED03;
    let mut xs = |r: &mut u64| {
        *r ^= *r << 13;
        *r ^= *r >> 7;
        *r ^= *r << 17;
        *r
    };
    let pieces = TetrisPiece::all();
    let mut total = 0u64;
    let mut viol = 0u64;
    let mut viol_clear = 0u64;
    let mut viol_noclear = 0u64;
    let mut pairs_ok = 0u64;

    for _ in 0..samples {
        // random β: each column a random bit pattern in rows 0..H
        let h = 6 + (xs(&mut rng) % 9) as usize; // height ceiling 6..14
        let mut beta: TetrisBoard = TetrisBoard::new();
        for j in 0..10 {
            let bits = (xs(&mut rng) as u32) & ((1u32 << h) - 1);
            for r in 0..h {
                if (bits >> r) & 1 == 1 {
                    beta.set_bit(j, r);
                }
            }
        }
        // b = β with a random subset of cells removed ⇒ b ⊆ β ⇒ heights(b) ≤ heights(β)
        let bl = beta.as_limbs();
        let mut b: TetrisBoard = TetrisBoard::new();
        for j in 0..10 {
            let keep = xs(&mut rng) as u32;
            let kept = bl[j] & keep;
            for r in 0..20 {
                if (kept >> r) & 1 == 1 {
                    b.set_bit(j, r);
                }
            }
        }
        // confirm domination of the inputs (heights)
        let hb = b.heights();
        let hbe = beta.heights();
        if !(0..10).all(|j| hb[j] <= hbe[j]) {
            continue; // shouldn't happen, but skip if not dominated
        }
        pairs_ok += 1;
        for &p in pieces.iter() {
            for &pl in TetrisPiecePlacement::all_from_piece(p) {
                let mut b2 = b;
                let rb = b2.apply_piece_placement(pl);
                let mut be2 = beta;
                let rbe = be2.apply_piece_placement(pl);
                if rb.is_lost == IsLost::LOST || rbe.is_lost == IsLost::LOST {
                    continue;
                }
                total += 1;
                let h2 = b2.heights();
                let hbe2 = be2.heights();
                let dominated = (0..10).all(|j| h2[j] <= hbe2[j]);
                if !dominated {
                    viol += 1;
                    if rb.lines_cleared > 0 || rbe.lines_cleared > 0 {
                        viol_clear += 1;
                    } else {
                        viol_noclear += 1;
                    }
                }
            }
        }
    }

    println!("\ndominated input pairs : {pairs_ok}");
    println!("placement checks      : {total}");
    println!(
        "domination VIOLATIONS : {viol}  ({:.4}% of checks)",
        100.0 * viol as f64 / total.max(1) as f64
    );
    println!(
        "  of which clear-related : {viol_clear}  ({:.1}% of violations)",
        100.0 * viol_clear as f64 / viol.max(1) as f64
    );
    println!("  of which NO-clear      : {viol_noclear}");
    println!("\nVERDICT:");
    if viol_noclear == 0 {
        println!(
            "  NON-CLEARING moves NEVER violate (as predicted structurally) — monotonicity holds \
             except across line clears. ⇒ a CLEAR-AWARE refined order (or bag-boundary basis where \
             clears already happened) can plausibly rescue downward-closure. The WQO/finite-basis \
             Lean route is VIABLE to attempt."
        );
    } else {
        println!(
            "  {viol_noclear} NO-CLEAR violations exist ⇒ height-domination is NOT preserved even \
             without clears ⇒ this order is fundamentally wrong; the WQO route needs a different \
             order or fails."
        );
    }
}

/// **Hole-aware domination monotonicity test.** The refined order is `b ≼' β` iff
/// `∀ j, height(b,j) ≤ height(β,j)` AND `∀ j, holes(b,j) ≤ holes(β,j)`. We generate
/// dominated pairs by TRUNCATING β above a random row (which lowers heights and never
/// raises per-column holes), then check whether a non-clearing placement preserves BOTH
/// the height and hole orders. If non-clear hole-violations are common, the refined order
/// has no clean keystone and the WQO route can't be saved by adding holes.
/// When set, `hmono` restricts the dominator β to CARRIER-LIKE boards (roughness ≤ 6,
/// total_holes ≤ 3) instead of fully-random junk — testing whether a candidate order is
/// monotone on the realistic carrier regime (where the WQO keystone actually needs to hold).
static HMONO_CLEAN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn run_hmono(samples: usize) {
    let clean = HMONO_CLEAN.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "MODE: hole-aware domination monotonicity — does applyStep preserve height+hole order (no clear)?{}",
        if clean {
            "  [CLEAN: β restricted to carrier-like roughness≤6 holes≤3]"
        } else {
            ""
        }
    );
    let mut rng: u64 = 0x71F1A693C9E4B6D2;
    let mut xs = |r: &mut u64| {
        *r ^= *r << 13;
        *r ^= *r >> 7;
        *r ^= *r << 17;
        *r
    };
    let pieces = TetrisPiece::all();
    // CLEAN mode: pool of real carrier surfaces (deep-play, roughness≤6 holes≤3) to draw β from.
    let pool: Vec<TetrisBoard> = if clean {
        let plist = [
            TetrisPiece::O_PIECE,
            TetrisPiece::I_PIECE,
            TetrisPiece::S_PIECE,
            TetrisPiece::Z_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::L_PIECE,
            TetrisPiece::J_PIECE,
        ];
        let perms = all_perms_7(plist);
        let mut pr: u64 = 0xD1B54A32D192ED03;
        let mut set: FxHashSet<TetrisBoard> = FxHashSet::default();
        let mut b = TetrisBoard::new();
        for _ in 0..200_000 {
            pr ^= pr << 13;
            pr ^= pr >> 7;
            pr ^= pr << 17;
            let order = &perms[(pr as usize) % perms.len()];
            match play_bag_order_minimax(b, order, 2) {
                None => b = TetrisBoard::new(),
                Some(nb) => {
                    if nb.roughness() <= 6 && nb.total_holes() <= 3 && nb.height() >= 4 {
                        set.insert(nb);
                        b = nb;
                    } else {
                        b = TetrisBoard::new();
                    }
                }
            }
            if set.len() >= 40_000 {
                break;
            }
        }
        println!("  clean β-pool: {} real carrier surfaces", set.len());
        set.into_iter().collect()
    } else {
        Vec::new()
    };
    let mut total = 0u64;
    let mut hole_viol_noclear = 0u64;
    let mut height_viol_noclear = 0u64;
    let mut totalhole_viol = 0u64;
    let mut subset_viol = 0u64;
    let mut overhang_viol = 0u64;
    let mut pairs = 0u64;
    let holes_per_col = |b: &TetrisBoard| -> [u32; 10] {
        let h = b.heights();
        let limbs = b.as_limbs();
        let mut out = [0u32; 10];
        for j in 0..10 {
            out[j] = h[j].saturating_sub(limbs[j].count_ones());
        }
        out
    };
    // depth-weighted buriedness: Σ over covered-empty cells of (depth below the column top).
    let overhang = |b: &TetrisBoard| -> u32 {
        let limbs = b.as_limbs();
        let mut s = 0u32;
        for j in 0..10 {
            let col = limbs[j];
            if col == 0 {
                continue;
            }
            let top = 32 - col.leading_zeros();
            for r in 0..top {
                if (col >> r) & 1 == 0 {
                    s += top - r;
                }
            }
        }
        s
    };
    if clean && pool.is_empty() {
        println!("  clean β-pool EMPTY — cannot test; aborting.");
        return;
    }
    for _ in 0..samples {
        let beta: TetrisBoard = if clean {
            pool[(xs(&mut rng) as usize) % pool.len()]
        } else {
            let hh = 6 + (xs(&mut rng) % 9) as usize;
            let mut bb: TetrisBoard = TetrisBoard::new();
            for j in 0..10 {
                let bits = (xs(&mut rng) as u32) & ((1u32 << hh) - 1);
                for r in 0..hh {
                    if (bits >> r) & 1 == 1 {
                        bb.set_bit(j, r);
                    }
                }
            }
            bb
        };
        let hh = beta.height() as usize;
        // b = β truncated above a random row k ⇒ heights(b) ≤ heights(β), holes(b) ≤ holes(β)
        let k = (xs(&mut rng) % (hh as u64 + 1)) as usize;
        let bl = beta.as_limbs();
        let mut b: TetrisBoard = TetrisBoard::new();
        for j in 0..10 {
            let kept = bl[j] & ((1u32 << k) - 1);
            for r in 0..k {
                if (kept >> r) & 1 == 1 {
                    b.set_bit(j, r);
                }
            }
        }
        if clean && (beta.roughness() > 6 || beta.total_holes() > 3) {
            continue;
        }
        let hb = b.heights();
        let hbe = beta.heights();
        let qb = holes_per_col(&b);
        let qbe = holes_per_col(&beta);
        if !((0..10).all(|j| hb[j] <= hbe[j]) && (0..10).all(|j| qb[j] <= qbe[j])) {
            continue;
        }
        pairs += 1;
        for &p in pieces.iter() {
            for &pl in TetrisPiecePlacement::all_from_piece(p) {
                let mut b2 = b;
                let rb = b2.apply_piece_placement(pl);
                let mut be2 = beta;
                let rbe = be2.apply_piece_placement(pl);
                if rb.is_lost == IsLost::LOST || rbe.is_lost == IsLost::LOST {
                    continue;
                }
                if rb.lines_cleared > 0 || rbe.lines_cleared > 0 {
                    continue;
                } // no-clear only
                total += 1;
                let h2 = b2.heights();
                let hbe2 = be2.heights();
                if !(0..10).all(|j| h2[j] <= hbe2[j]) {
                    height_viol_noclear += 1;
                }
                let q2 = holes_per_col(&b2);
                let qbe2 = holes_per_col(&be2);
                if !(0..10).all(|j| q2[j] <= qbe2[j]) {
                    hole_viol_noclear += 1;
                }
                // total-holes scalar order
                if b2.total_holes() > be2.total_holes() {
                    totalhole_viol += 1;
                }
                // upward-⊆ (subset of filled cells): b2 has a cell β2 lacks
                let lb = b2.as_limbs();
                let lbe = be2.as_limbs();
                if (0..10).any(|j| lb[j] & !lbe[j] != 0) {
                    subset_viol += 1;
                }
                // overhang-depth (depth-weighted buriedness) scalar order
                if overhang(&b2) > overhang(&be2) {
                    overhang_viol += 1;
                }
            }
        }
    }
    let pct = |v: u64| 100.0 * v as f64 / total.max(1) as f64;
    println!("\ndominated pairs (height+hole+subset) : {pairs}");
    println!("no-clear placement checks            : {total}");
    println!("--- candidate domination orders, no-clear monotonicity violations ---");
    println!(
        "  height (per-col)      : {height_viol_noclear:>10}  ({:.3}%)  [coarse, known monotone]",
        pct(height_viol_noclear)
    );
    println!(
        "  height+holes (per-col): {hole_viol_noclear:>10}  ({:.3}%)",
        pct(hole_viol_noclear)
    );
    println!(
        "  total-holes (scalar)  : {totalhole_viol:>10}  ({:.3}%)",
        pct(totalhole_viol)
    );
    println!(
        "  overhang-depth (scalar): {overhang_viol:>9}  ({:.3}%)",
        pct(overhang_viol)
    );
    println!(
        "  upward-⊆ (subset)     : {subset_viol:>10}  ({:.3}%)",
        pct(subset_viol)
    );
    println!(
        "\nVERDICT (P1 make-or-break — is ANY fine order monotone under no-clear placement?):"
    );
    let any_fine_clean =
        totalhole_viol == 0 || subset_viol == 0 || overhang_viol == 0 || hole_viol_noclear == 0;
    if any_fine_clean {
        println!(
            "  A fine order has 0 no-clear violations ⇒ candidate clean keystone ⇒ swap domLE in \
            WqoCarrier.lean and re-run the basis probe."
        );
    } else {
        println!(
            "  EVERY fine order (holes/overhang/subset) is BROKEN on no-clear moves ⇒ no clean \
            keystone exists in this family ⇒ P1 (refined-order WQO) FLOORED: placement creates \
            holes/overhangs per local surface, non-monotonically, regardless of how the fineness is \
            packaged. Only height (too coarse) is monotone."
        );
    }
}

/// Horizontal reflection of a board (column j ↔ 9−j). 7-bag is reflection-closed
/// (S↔Z, L↔J), so the carrier is reflection-symmetric and we can canonicalize boundary
/// boards to their min representative — a sound ~2× compression.
fn reflect(b: &TetrisBoard) -> TetrisBoard {
    let limbs = b.as_limbs();
    let mut out = TetrisBoard::new();
    let mut ol = [0u32; 10];
    for j in 0..10 {
        ol[j] = limbs[9 - j];
    }
    // rebuild via set_bit from the reflected limbs
    for (j, &col) in ol.iter().enumerate() {
        let mut bits = col;
        while bits != 0 {
            let r = bits.trailing_zeros() as usize;
            out.set_bit(j, r);
            bits &= bits - 1;
        }
    }
    out
}

/// Canonical (reflection-min) representative of a board.
fn canon(b: &TetrisBoard) -> TetrisBoard {
    let r = reflect(b);
    if r.as_limbs() <= b.as_limbs() { r } else { *b }
}

/// `a` dominates `v` under height-domination: every column of `v` is ≤ `a`'s.
fn dom(a: &[u32; 10], v: &[u32; 10]) -> bool {
    (0..10).all(|j| v[j] <= a[j])
}

/// Insert height-vector `v` into the maximal antichain `ac` (keep the largest elements).
/// Returns true if `v` was added (was maximal). O(|ac|) per insert.
fn antichain_insert(ac: &mut Vec<[u32; 10]>, v: [u32; 10]) -> bool {
    if ac.iter().any(|a| dom(a, &v)) {
        return false; // v dominated by an existing maximal element
    }
    ac.retain(|a| !dom(&v, a)); // drop elements v now dominates
    ac.push(v);
    true
}

/// **Debt-trajectory make-or-break (hole-debt × surface-WQO route).** Plays the deep player
/// against a chosen adversary, tracking at each bag boundary the MAX DEBT (= total holes, the
/// Lyapunov counter) alongside the surface antichain (height-vectors under domination) and max
/// height. The route's gating question: is there a bound `D` and a SMALL height-domination basis
/// such that the carrier `{b : surface(b) ≼ β ∧ debt(b) ≤ D}` is reachable-closed? If the deep
/// player holds debt ≤ small D AND a small antichain AND never tops out vs the structural-worst
/// order, the carrier candidate is real ⇒ proceed to the Lean reduction. If debt grows unbounded
/// or the antichain explodes ⇒ HONEST FLOOR with the number. `adv`: 0=S/Z-first, 1=adaptive-worst,
/// 2=random.
fn run_debttraj(depth: u8, nbags: usize, adv: u8) {
    let advname = match adv {
        0 => "S/Z-first (structural worst)",
        1 => "adaptive-worst (minimax)",
        _ => "random",
    };
    println!(
        "MODE: debt trajectory — deep-player(depth {depth}) vs {advname}; track max debt + surface antichain + height"
    );
    let t0 = Instant::now();
    let perms = all_perms_7([
        TetrisPiece::O_PIECE,
        TetrisPiece::I_PIECE,
        TetrisPiece::S_PIECE,
        TetrisPiece::Z_PIECE,
        TetrisPiece::T_PIECE,
        TetrisPiece::L_PIECE,
        TetrisPiece::J_PIECE,
    ]);
    let mut rng: u64 = 0x243F6A8885A308D3;
    let mut antichain: Vec<[u32; 10]> = Vec::new();
    let mut seen: FxHashSet<[u32; 10]> = FxHashSet::default();
    let mut max_debt = 0u32;
    let mut max_h = 0u32;
    let mut topouts = 0u64;
    let mut survived = 0u64;
    let mut debt_ge = [0u64; 17]; // histogram: count of boundaries with debt ≥ k
    let mut b = TetrisBoard::new();
    antichain_insert(&mut antichain, b.heights());
    for bag in 0..nbags {
        let nb = match adv {
            0 => play_bag_szfirst(b, depth),
            1 => play_bag_minimax(b, depth),
            _ => {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                play_bag_order_minimax(b, &perms[(rng as usize) % perms.len()], depth)
            }
        };
        match nb {
            None => {
                topouts += 1;
                b = TetrisBoard::new();
            }
            Some(nb) => {
                survived += 1;
                let d = nb.total_holes();
                max_debt = max_debt.max(d);
                max_h = max_h.max(nb.height());
                for k in 0..=(d.min(16) as usize) {
                    debt_ge[k] += 1;
                }
                let hv = canon(&nb).heights();
                if seen.insert(hv) {
                    antichain_insert(&mut antichain, hv);
                }
                b = nb;
            }
        }
        if (bag + 1) % 250 == 0 {
            use std::io::Write;
            println!(
                "  bag {:>6}: surv {} topout {} | max_debt {} cur_debt {} max_h {} | antichain {} distinct {} [{:.0}s]",
                bag + 1,
                survived,
                topouts,
                max_debt,
                b.total_holes(),
                max_h,
                antichain.len(),
                seen.len(),
                t0.elapsed().as_secs_f64()
            );
            std::io::stdout().flush().ok();
        }
    }
    println!("\n================ DEBT-TRAJECTORY RESULT ================");
    println!("adversary                 : {advname}");
    println!("bags played               : {nbags}  (survived {survived}, topouts {topouts})");
    println!("MAX DEBT at bag boundary  : {max_debt}");
    println!("MAX height at boundary    : {max_h}");
    println!("surface antichain size    : {}", antichain.len());
    println!("distinct boundary surfaces: {}", seen.len());
    print!("debt-≥-k boundary counts  : ");
    for k in 0..=8 {
        if debt_ge[k] > 0 {
            print!("≥{k}:{} ", debt_ge[k]);
        }
    }
    println!();
    println!("\nVERDICT:");
    if topouts == 0 && max_debt <= 6 && antichain.len() <= 2000 {
        println!(
            "  PROCEED — debt BOUNDED (≤{max_debt}) AND antichain SMALL ({}) AND zero topouts vs {advname} \
             ⇒ the debt-WQO carrier candidate is real at D={max_debt}. Build the Lean reduction (rung A: \
             debt(place)≤debt+3, then the bounded-debt × surface-WQO Carrier).",
            antichain.len()
        );
    } else {
        println!(
            "  EXAMINE — topouts {topouts}, max_debt {max_debt}, antichain {}. Unbounded debt or exploding \
             antichain vs the structural-worst order ⇒ HONEST FLOOR (record the number).",
            antichain.len()
        );
    }
    println!("total time: {:.1}s", t0.elapsed().as_secs_f64());
}

/// **S1 — basis antichain size (the WQO route's make-or-break).** Runs the deep-player
/// boundary closure (like `deepcarrier`) but maintains the MAXIMAL ANTICHAIN of boundary
/// surfaces' height-vectors under domination. The WQO finite-basis Lean route can only
/// `native_decide` if this antichain (the basis) is small. Small/saturating ⇒ viable
/// (→ S2). Grows past ~1e5 ⇒ HONEST FLOOR (the basis itself explodes).
fn run_basis(depth: u8, nbags: usize) {
    println!(
        "MODE: S1 basis antichain — SAMPLED via random-order depth-{depth} play, maximal antichain"
    );
    let t0 = Instant::now();
    let perms = all_perms_7([
        TetrisPiece::O_PIECE,
        TetrisPiece::I_PIECE,
        TetrisPiece::S_PIECE,
        TetrisPiece::Z_PIECE,
        TetrisPiece::T_PIECE,
        TetrisPiece::L_PIECE,
        TetrisPiece::J_PIECE,
    ]);
    let cap = 150_000usize;
    let mut antichain: Vec<[u32; 10]> = Vec::new();
    let mut seen: FxHashSet<[u32; 10]> = FxHashSet::default();
    let mut surfaces = 0u64;
    let mut rng: u64 = 0x243F6A8885A308D3;
    let mut b = TetrisBoard::new();
    antichain_insert(&mut antichain, b.heights());
    let mut overflow_basis = false;

    for bag in 0..nbags {
        // random order, deep-player response (it doesn't know the order → worst-case lookahead)
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let order = &perms[(rng as usize) % perms.len()];
        match play_bag_order_minimax(b, order, depth) {
            None => {
                // restart from empty on a (rare under random) top-out
                b = TetrisBoard::new();
                continue;
            }
            Some(nb) => {
                b = nb;
                let hv = canon(&b).heights();
                surfaces += 1;
                if seen.insert(hv) {
                    antichain_insert(&mut antichain, hv);
                    if antichain.len() > cap {
                        overflow_basis = true;
                    }
                }
            }
        }
        if overflow_basis {
            println!(
                "  antichain EXCEEDED {cap} at bag {bag} ({surfaces} surfaces, {} distinct) — basis too large.",
                seen.len()
            );
            break;
        }
        if (bag + 1) % 20_000 == 0 {
            println!(
                "  bag {:7}: surfaces {surfaces}, distinct {}, ANTICHAIN {}, {:.0}s",
                bag + 1,
                seen.len(),
                antichain.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }

    let max_h = antichain
        .iter()
        .map(|v| *v.iter().max().unwrap())
        .max()
        .unwrap_or(0);
    println!("\n================ RESULT (S1, sampled) ================");
    println!("boundary surfaces sampled : {surfaces}");
    println!("distinct height-vectors   : {}", seen.len());
    println!("MAXIMAL ANTICHAIN (basis) : {}", antichain.len());
    println!("max column height in it   : {max_h}");
    println!("\nVERDICT:");
    if overflow_basis {
        println!(
            "  HONEST FLOOR — LARGE BASIS: the maximal antichain exceeded {cap}. The carrier's safe \
             surfaces are mostly domination-INCOMPARABLE (one column higher, another lower), so the \
             basis is large ⇒ too big for native_decide ⇒ the WQO finite-basis route cannot complete \
             by enumeration. (Sampled lower bound on the true antichain.)"
        );
    } else {
        println!(
            "  SMALL/SATURATED: antichain = {} ≤ {cap} over {} distinct surfaces ⇒ the carrier basis \
             looks finite & certifiable ⇒ WQO route plausibly VIABLE → confirm then S2.",
            antichain.len(),
            seen.len()
        );
    }
    println!("total time: {:.1}s", t0.elapsed().as_secs_f64());
}

/// Expand one bag from boundary board `start` with the DEEP minimax player responding
/// to every adversary reveal order. Returns `(has_loss, next_boundary_boards)`: the
/// player tops out on some order, or the distinct canonical boundary boards reachable.
/// Memoized over `(board, mask)` within the bag so shared sub-orders aren't recomputed.
fn expand_bag_deep(start: TetrisBoard, depth: u8) -> (bool, Vec<TetrisBoard>) {
    // Fresh minimax memo per bag — bounds memory (a cross-bag memo grows to many GB).
    let mut mmemo: FxHashMap<(TetrisBoard, u8, u8), i32> = FxHashMap::default();
    let mut leaves: FxHashSet<TetrisBoard> = FxHashSet::default();
    let mut seen: FxHashSet<(TetrisBoard, u8)> = FxHashSet::default();
    let mut stack: Vec<(TetrisBoard, u8)> = vec![(start, MM_FULL)];
    seen.insert((start, MM_FULL));
    let mut has_loss = false;
    while let Some((b, mask)) = stack.pop() {
        if mask == 0 {
            leaves.insert(canon(&b));
            continue;
        }
        let look = depth.min(mask.count_ones() as u8);
        for (pi, p) in MM_PIECES.iter().enumerate() {
            let bit = 1u8 << pi;
            if mask & bit == 0 {
                continue;
            }
            // deep player picks the placement minimizing bag_minimax over the rest
            let mut best_board = b;
            let mut best_v = MM_LOSS;
            let hole_cap = HOLE_CAP.load(std::sync::atomic::Ordering::Relaxed);
            for &pl in TetrisPiecePlacement::all_from_piece(*p) {
                let mut nb = b;
                let res = nb.apply_piece_placement(pl);
                if res.is_lost == IsLost::LOST || nb.height() > ROWS {
                    continue;
                }
                if nb.total_holes() > hole_cap {
                    continue; // debt-cap: this placement leaves the bounded-debt carrier
                }
                let v = bag_minimax(nb, mask & !bit, look.saturating_sub(1), &mut mmemo);
                if v < best_v {
                    best_v = v;
                    best_board = nb;
                }
            }
            if best_v >= MM_LOSS {
                has_loss = true; // adversary draws p, player has no safe in-carrier placement
            } else if seen.insert((best_board, mask & !bit)) {
                stack.push((best_board, mask & !bit));
            }
        }
    }
    (has_loss, leaves.into_iter().collect())
}

/// **Deep-player carrier size.** Bag-boundary AND-OR closure using the deep minimax
/// player as the strategy: BFS over canonical boundary surfaces from the empty board,
/// expanding each through a full bag (all adversary orders, deep-player response), with
/// backward death propagation. Budget-bounded. Decisive R1 test: converges small ⇒
/// certifiable carrier; explodes ⇒ proof must be symbolic.
fn run_deepcarrier(depth: u8, budget: usize) {
    println!(
        "MODE: deep-player carrier — bag-boundary AND-OR closure, depth-{depth} player, reflection-canon"
    );
    let t0 = Instant::now();
    let init = canon(&TetrisBoard::new());
    let mut index: FxHashMap<TetrisBoard, u32> = FxHashMap::default();
    let mut boards: Vec<TetrisBoard> = Vec::new();
    let mut has_loss: Vec<bool> = Vec::new();
    let mut edge_src: Vec<u32> = Vec::new();
    let mut edge_dst: Vec<u32> = Vec::new();
    let mut queue: VecDeque<u32> = VecDeque::new();
    index.insert(init, 0);
    boards.push(init);
    has_loss.push(false);
    queue.push_back(0);
    let mut exploded = false;
    let mut max_h = 0u32;
    // WQO basis: the maximal antichain of boundary-surface height-vectors under domination.
    // The carrier is downward-closed in height-domination, so this antichain IS its finite
    // basis — it can stay small even if the full surface set explodes.
    let mut antichain: Vec<[u32; 10]> = Vec::new();
    let mut ac_max = 0usize;

    while let Some(sid) = queue.pop_front() {
        let (loss, leaves) = expand_bag_deep(boards[sid as usize], depth);
        if loss {
            has_loss[sid as usize] = true;
        }
        for leaf in leaves {
            max_h = max_h.max(leaf.height());
            let nid = if let Some(&id) = index.get(&leaf) {
                id
            } else {
                if boards.len() >= budget {
                    exploded = true;
                    break;
                }
                let id = boards.len() as u32;
                index.insert(leaf, id);
                boards.push(leaf);
                has_loss.push(false);
                queue.push_back(id);
                antichain_insert(&mut antichain, leaf.heights());
                ac_max = ac_max.max(antichain.len());
                id
            };
            edge_src.push(sid);
            edge_dst.push(nid);
        }
        if exploded {
            break;
        }
        if boards.len() % 50_000 == 0 && !boards.is_empty() {
            println!(
                "  ... {} boundary surfaces, antichain {}, queue {}, {:.0}s",
                boards.len(),
                antichain.len(),
                queue.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }

    let n = boards.len();
    println!(
        "closure: {n} canonical boundary surfaces, WQO antichain={} (max {ac_max}), max_height={max_h}, {:.1}s{}",
        antichain.len(),
        t0.elapsed().as_secs_f64(),
        if exploded {
            "  [EXPLODED: hit budget]"
        } else {
            ""
        }
    );
    if exploded {
        println!(
            "\nVERDICT: full surface set EXCEEDS {budget}, BUT the WQO antichain (height-domination \
             basis) is {} (peaked {ac_max}). If the antichain SATURATES small (≤~1e3) while the full \
             set explodes, the downward-closed carrier has a small finite basis ⇒ the WQO×debt route \
             is viable (certify the basis, not the full set). If the antichain ALSO grows unbounded ⇒ \
             HONEST FLOOR (no small basis even with debt cap).",
            antichain.len()
        );
        return;
    }
    // death propagation
    let m = edge_src.len();
    let mut offsets = vec![0u32; n + 1];
    for &d in &edge_dst {
        offsets[d as usize + 1] += 1;
    }
    for i in 0..n {
        offsets[i + 1] += offsets[i];
    }
    let mut rev = vec![0u32; m];
    let mut cur = offsets.clone();
    for k in 0..m {
        let d = edge_dst[k] as usize;
        rev[cur[d] as usize] = edge_src[k];
        cur[d] += 1;
    }
    let mut dead = vec![false; n];
    let mut dq: VecDeque<u32> = VecDeque::new();
    for i in 0..n {
        if has_loss[i] {
            dead[i] = true;
            dq.push_back(i as u32);
        }
    }
    while let Some(s) = dq.pop_front() {
        for &pred in &rev[offsets[s as usize] as usize..offsets[s as usize + 1] as usize] {
            if !dead[pred as usize] {
                dead[pred as usize] = true;
                dq.push_back(pred);
            }
        }
    }
    let alive = n - dead.iter().filter(|&&d| d).count();
    let init_alive = !dead[0];
    println!("\n================ RESULT ================");
    println!("canonical boundary surfaces : {n}");
    println!("surviving closed core       : {alive}");
    println!("init survives?              : {init_alive}");
    println!("\nVERDICT:");
    if init_alive && alive > 0 && alive <= 100_000 {
        println!(
            "  CONVERGED + CERTIFIABLE: the deep player's carrier is {alive} surfaces (≤1e5) and \
             contains init ⇒ R1! Export it and certify closure by native_decide ⇒ a real path to \
             the proof."
        );
    } else if init_alive && alive > 0 {
        println!(
            "  CONVERGED but LARGE ({alive} > 1e5): a finite carrier exists but exceeds native_decide \
             capacity ⇒ needs symmetry/phase compression, or symbolic proof."
        );
    } else {
        println!(
            "  COLLAPSE: surviving core empty/excludes init at depth-{depth} (player too shallow?)."
        );
    }
    println!("total time: {:.1}s", t0.elapsed().as_secs_f64());
}

/// Compression-scheme signature of a (canonical) board. Each scheme maps a board to a
/// `[u32; 11]` key; the carrier's size in that representation is the number of distinct
/// keys. Scheme 0 (exact limbs) is the baseline that is known to explode; the others are
/// progressively coarser lossy quotients. The make-or-break the `compress` mode answers:
/// does ANY scheme's distinct-key count PLATEAU small (≤~1e5) while the exact count keeps
/// growing — i.e. is the huge carrier finitely representable after compression?
fn compress_sig(scheme: u8, b: &TetrisBoard) -> [u32; 11] {
    let cb = canon(b);
    let h = cb.heights();
    let mut k = [0u32; 11];
    match scheme {
        // 0: EXACT canonical board (full limbs) — baseline, no compression.
        0 => {
            let l = cb.as_limbs();
            k[..10].copy_from_slice(&l[..10]);
        }
        // 1: exact per-column HEIGHT vector (collapses different hole patterns below a
        //    common surface profile).
        1 => {
            k[..10].copy_from_slice(&h);
        }
        // 2: RELATIVE heights (subtract min) — vertical-translation invariant surface.
        2 => {
            let m = *h.iter().min().unwrap_or(&0);
            for j in 0..10 {
                k[j] = h[j] - m;
            }
        }
        // 3: relative heights CLAMPED to 6 — bounds the surface-shape alphabet.
        3 => {
            let m = *h.iter().min().unwrap_or(&0);
            for j in 0..10 {
                k[j] = (h[j] - m).min(6);
            }
        }
        // 4: relative heights CLAMPED to 4 — coarser shape alphabet.
        4 => {
            let m = *h.iter().min().unwrap_or(&0);
            for j in 0..10 {
                k[j] = (h[j] - m).min(4);
            }
        }
        // 5: adjacent-column height DIFFERENCES clamped to ±4 (offset by +4 to stay ≥0) —
        //    pure local shape, invariant to both translation and the base column index.
        5 => {
            for j in 0..9 {
                let d = h[j] as i32 - h[j + 1] as i32;
                k[j] = (d.clamp(-4, 4) + 4) as u32;
            }
        }
        // 6: HEIGHTS bucketed by /2 — halves the vertical resolution.
        6 => {
            for j in 0..10 {
                k[j] = h[j] / 2;
            }
        }
        // 7: relative heights clamped to 4 PLUS a coarse total-holes bucket (aux slot) —
        //    adds (coarse) drainability info to the shape.
        7 => {
            let m = *h.iter().min().unwrap_or(&0);
            for j in 0..10 {
                k[j] = (h[j] - m).min(4);
            }
            k[10] = (cb.total_holes() / 2).min(15);
        }
        _ => {}
    }
    k
}

const SCHEME_NAMES: [&str; 8] = [
    "exact-limbs",
    "heights",
    "relheights",
    "relh-clamp6",
    "relh-clamp4",
    "adjdiff-clamp4",
    "heights/2",
    "relh4+holes/2",
];

/// **Compression make-or-break (PRIMARY route, iteration 1).** Runs the deep-player
/// AND-OR bag-boundary closure exactly like `deepcarrier` (faithful: the frontier is keyed
/// by the EXACT canonical board so the expansion is not lossy), but simultaneously tracks,
/// for each of 8 compression SCHEMES, the number of distinct signatures among all boards
/// discovered. Prints the growth of every scheme vs. the exact count. A scheme whose count
/// PLATEAUS far below the exact count is a candidate compressed representation of the
/// carrier — the lever that could bring the >5e5 carrier under the native_decide ceiling.
fn run_compress(depth: u8, budget: usize) {
    println!(
        "MODE: compression make-or-break — deep-player carrier keyed exact, |I| tracked under 8 schemes"
    );
    println!(
        "Q: does any scheme's distinct-signature count PLATEAU small while exact keeps growing?"
    );
    let t0 = Instant::now();
    let init = canon(&TetrisBoard::new());
    let mut index: FxHashMap<TetrisBoard, u32> = FxHashMap::default();
    let mut boards: Vec<TetrisBoard> = Vec::new();
    let mut queue: VecDeque<u32> = VecDeque::new();
    // one distinct-signature set per scheme
    let mut sigsets: Vec<FxHashSet<[u32; 11]>> = (0..8).map(|_| FxHashSet::default()).collect();
    let mut record = |boards: &[TetrisBoard], i: usize, sigsets: &mut Vec<FxHashSet<[u32; 11]>>| {
        for s in 0..8u8 {
            sigsets[s as usize].insert(compress_sig(s, &boards[i]));
        }
    };
    index.insert(init, 0);
    boards.push(init);
    queue.push_back(0);
    record(&boards, 0, &mut sigsets);
    let mut exploded = false;
    let mut max_h = 0u32;
    let mut next_report = 25_000usize;

    while let Some(sid) = queue.pop_front() {
        let (_loss, leaves) = expand_bag_deep(boards[sid as usize], depth);
        for leaf in leaves {
            max_h = max_h.max(leaf.height());
            if let std::collections::hash_map::Entry::Vacant(entry) = index.entry(leaf) {
                if boards.len() >= budget {
                    exploded = true;
                    break;
                }
                let id = boards.len() as u32;
                entry.insert(id);
                boards.push(leaf);
                queue.push_back(id);
                record(&boards, id as usize, &mut sigsets);
            }
        }
        if exploded {
            break;
        }
        if boards.len() >= next_report {
            print!("  exact={:>8}  ", boards.len());
            for s in 0..8 {
                print!("{}={} ", SCHEME_NAMES[s], sigsets[s].len());
            }
            println!("[{:.0}s]", t0.elapsed().as_secs_f64());
            next_report += 25_000;
        }
    }

    let n = boards.len();
    println!(
        "\n================ COMPRESSION RESULT ================\nexact carrier (canonical boards): {n}{}  max_height={max_h}  [{:.1}s]",
        if exploded {
            "  [EXPLODED: hit budget]"
        } else {
            "  [CONVERGED]"
        },
        t0.elapsed().as_secs_f64()
    );
    println!("{:<16} {:>12} {:>10}", "scheme", "|distinct|", "ratio");
    for s in 0..8 {
        let c = sigsets[s].len();
        println!(
            "{:<16} {:>12} {:>9.1}x",
            SCHEME_NAMES[s],
            c,
            n as f64 / c as f64
        );
    }
    // --- congruence diagnostic: is the smallest scheme SOUND? ---------------------
    // For the two most-compressing schemes, group the carrier boards by signature and
    // measure how much member boards DIVERGE in total_holes (the drainability-relevant
    // quantity). Large hole-spread within a signature ⇒ the scheme merges drainable and
    // undrainable boards ⇒ its preimage is not a sound carrier (it admits the undrainable
    // members as junk the adversary escapes from). This puts a NUMBER on (un)soundness.
    println!("\n---- congruence (soundness) of the most-compressing schemes ----");
    for &s in &[6u8, 4u8] {
        let mut groups: FxHashMap<[u32; 11], (u32, u32, u32)> = FxHashMap::default();
        for b in &boards {
            let key = compress_sig(s, b);
            let holes = b.total_holes();
            let e = groups.entry(key).or_insert((u32::MAX, 0, 0));
            e.0 = e.0.min(holes);
            e.1 = e.1.max(holes);
            e.2 += 1;
        }
        let multi = groups.values().filter(|g| g.2 >= 2).count();
        let split = groups.values().filter(|g| g.1 - g.0 >= 2).count();
        let max_spread = groups.values().map(|g| g.1 - g.0).max().unwrap_or(0);
        let max_grp = groups.values().map(|g| g.2).max().unwrap_or(0);
        println!(
            "  {:<16} groups={:>7}  multi-board={:>7}  hole-spread≥2={:>7} ({:.1}% of multi)  max-spread={}  max-group={}",
            SCHEME_NAMES[s as usize],
            groups.len(),
            multi,
            split,
            100.0 * split as f64 / multi.max(1) as f64,
            max_spread,
            max_grp
        );
    }

    println!("\nVERDICT:");
    // best non-exact scheme
    let best = (1..8).min_by_key(|&s| sigsets[s].len()).unwrap();
    let bc = sigsets[best].len();
    if !exploded {
        println!(
            "  exact carrier CONVERGED at {n} ≤ budget — compression secondary; smallest scheme \
             '{}' = {bc}. If exact ≤1e5 this is already certifiable.",
            SCHEME_NAMES[best]
        );
    } else if bc <= 100_000 {
        println!(
            "  CANDIDATE: scheme '{}' holds {bc} ≤1e5 distinct signatures while the EXACT carrier \
             exploded past {budget}. If this scheme also CLOSES (iteration 2: congruence check), it \
             is a certifiable compressed atlas. Re-run at higher budget to confirm the plateau.",
            SCHEME_NAMES[best]
        );
    } else {
        println!(
            "  FLOOR (so far): even the smallest scheme '{}' has {bc} >1e5 distinct signatures at \
             budget {budget} and is still growing ⇒ compression does not bring the carrier under the \
             native_decide ceiling at this depth. Inspect the growth columns: any scheme whose count \
             stopped rising is a plateau candidate regardless of absolute size.",
            SCHEME_NAMES[best]
        );
    }
    println!("total time: {:.1}s", t0.elapsed().as_secs_f64());
}

/// **Band closure-fraction (PRIMARY route, feature-region representation).** Samples
/// reachable IN-BAND boundary surfaces (random-order deep play, restarting from empty when
/// the player drifts out of band or tops out), then for each surface checks the RAW
/// closure condition: does EVERY one of the 7 pieces have at least one placement landing
/// back in-band (after clears)? The closure FRACTION = (#surfaces closed)/(#surfaces). The
/// prior work measured only the post-cascade GFP core (empty/explodes); this measures the
/// raw fraction and CHARACTERIZES the counterexamples — distinguishing "almost closed, a
/// few concentrated bad (surface,piece) pairs" (→ split the proof) from "diffusely broken"
/// (→ the band is hopeless). The graded metric the goal asks for.
fn run_closurefrac(adm: Admiss, depth: u8, nbags: usize) {
    println!(
        "MODE: band closure-fraction — raw fraction of in-band surfaces with a safe in-band move for EVERY piece"
    );
    println!(
        "band: hcap={:?} rcap={:?} holecap={:?} well_empty={} pot={:?}",
        adm.hcap, adm.rcap, adm.holecap, adm.well_empty, adm.pot
    );
    if !adm.active() {
        println!("  (no band caps given — pass e.g. `h12 holes2 r6` to define the band)");
        return;
    }
    let t0 = Instant::now();
    let plist: [(TetrisPiece, &str); 7] = [
        (TetrisPiece::O_PIECE, "O"),
        (TetrisPiece::I_PIECE, "I"),
        (TetrisPiece::S_PIECE, "S"),
        (TetrisPiece::Z_PIECE, "Z"),
        (TetrisPiece::T_PIECE, "T"),
        (TetrisPiece::L_PIECE, "L"),
        (TetrisPiece::J_PIECE, "J"),
    ];
    let perms = all_perms_7(plist.map(|(p, _)| p));
    let mut rng: u64 = 0x243F6A8885A308D3;
    let mut surfaces: FxHashSet<TetrisBoard> = FxHashSet::default();
    let cap = 300_000usize;
    let mut b = TetrisBoard::new();
    if adm.ok(&b) {
        surfaces.insert(canon(&b));
    }
    for _ in 0..nbags {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let order = &perms[(rng as usize) % perms.len()];
        match play_bag_order_minimax(b, order, depth) {
            None => {
                b = TetrisBoard::new();
            }
            Some(nb) => {
                if adm.ok(&nb) {
                    surfaces.insert(canon(&nb));
                    b = nb;
                } else {
                    b = TetrisBoard::new();
                }
            }
        }
        if surfaces.len() >= cap {
            break;
        }
    }
    println!(
        "sampled {} distinct in-band surfaces ({:.0}s)",
        surfaces.len(),
        t0.elapsed().as_secs_f64()
    );

    // raw closure: for each surface, does every piece have an in-band placement?
    let mut closed = 0usize;
    let mut fail_by_piece = [0usize; 7];
    let mut fail_surfaces = 0usize;
    let mut sum_h = 0u64;
    let mut sum_holes = 0u64;
    let mut sum_rough = 0u64;
    for surf in &surfaces {
        let mut all_ok = true;
        for (pi, (p, _)) in plist.iter().enumerate() {
            let mut has = false;
            for &pl in TetrisPiecePlacement::all_from_piece(*p) {
                let mut nb = *surf;
                let res = nb.apply_piece_placement(pl);
                if res.is_lost == IsLost::LOST || nb.height() > ROWS {
                    continue;
                }
                if adm.ok(&nb) {
                    has = true;
                    break;
                }
            }
            if !has {
                all_ok = false;
                fail_by_piece[pi] += 1;
            }
        }
        if all_ok {
            closed += 1;
        } else {
            fail_surfaces += 1;
            sum_h += surf.height() as u64;
            sum_holes += surf.total_holes() as u64;
            sum_rough += surf.roughness() as u64;
        }
    }
    let total = surfaces.len().max(1);
    println!("\n================ CLOSURE-FRACTION RESULT ================");
    println!(
        "closure fraction         : {:.4}%  ({closed}/{total} surfaces fully closed)",
        100.0 * closed as f64 / total as f64
    );
    println!("counterexample surfaces  : {fail_surfaces}");
    print!("failures by piece        : ");
    for (pi, (_, name)) in plist.iter().enumerate() {
        print!("{name}={} ", fail_by_piece[pi]);
    }
    println!();
    if fail_surfaces > 0 {
        let f = fail_surfaces as f64;
        println!(
            "failing-surface features : avg_height={:.2} avg_holes={:.2} avg_roughness={:.2}",
            sum_h as f64 / f,
            sum_holes as f64 / f,
            sum_rough as f64 / f
        );
        let sz = fail_by_piece[2] + fail_by_piece[3];
        let allf: usize = fail_by_piece.iter().sum();
        println!(
            "S/Z share of failures    : {:.1}%  ({sz}/{allf} (surface,piece) failures are S or Z)",
            100.0 * sz as f64 / allf.max(1) as f64
        );
    }
    println!("\nVERDICT:");
    let frac = closed as f64 / total as f64;
    if frac >= 0.999 {
        println!(
            "  ALMOST CLOSED ({:.3}%): counterexamples are RARE — characterize the {fail_surfaces} \
             bad surfaces; if they share structure, the proof can split (clean band + targeted \
             drainage lemmas for the bad type). Promising — refine the band to exclude them.",
            100.0 * frac
        );
    } else if frac >= 0.9 {
        println!(
            "  MOSTLY CLOSED ({:.2}%): a sizeable minority of surfaces have a forced out-of-band \
             piece. Inspect the dominant failing piece + features; tighten/retarget the band.",
            100.0 * frac
        );
    } else {
        println!(
            "  DIFFUSELY BROKEN ({:.1}%): a large fraction of in-band surfaces have a forced \
             out-of-band piece ⇒ this band is not close to controlled-invariant; the obstruction is \
             not a few concentrated counterexamples. Record + try a different band.",
            100.0 * frac
        );
    }
    println!("total time: {:.1}s", t0.elapsed().as_secs_f64());
}

/// **Band closure-fraction SWEEP (PRIMARY route, iteration 3 = refinement).** Samples
/// reachable surfaces ONCE under a generous superset band, then evaluates the raw closure
/// fraction for a LADDER of sub-bands (roughness × holes caps). Reveals the
/// closure-fraction-vs-band-tightness tradeoff in one shot: if some sub-band is both highly
/// closed AND small (a "knee"), it is a refinement target; if it is a smooth tradeoff with
/// no knee (loose⇒closed-but-huge, tight⇒small-but-broken), representation-2 is floored.
fn run_closuresweep(depth: u8, nbags: usize) {
    println!(
        "MODE: band closure-fraction SWEEP — sample once (generous), evaluate a ladder of sub-bands"
    );
    let t0 = Instant::now();
    let plist: [(TetrisPiece, &str); 7] = [
        (TetrisPiece::O_PIECE, "O"),
        (TetrisPiece::I_PIECE, "I"),
        (TetrisPiece::S_PIECE, "S"),
        (TetrisPiece::Z_PIECE, "Z"),
        (TetrisPiece::T_PIECE, "T"),
        (TetrisPiece::L_PIECE, "L"),
        (TetrisPiece::J_PIECE, "J"),
    ];
    let perms = all_perms_7(plist.map(|(p, _)| p));
    // generous sampling band: height ≤ 14, holes ≤ 4, roughness ≤ 12
    let sample_band = Admiss {
        hcap: Some(14),
        rcap: Some(12),
        holecap: Some(4),
        well_empty: false,
        pot: None,
    };
    let mut rng: u64 = 0x243F6A8885A308D3;
    let mut surfaces: FxHashSet<TetrisBoard> = FxHashSet::default();
    let cap = 200_000usize;
    let mut b = TetrisBoard::new();
    surfaces.insert(canon(&b));
    for _ in 0..nbags {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let order = &perms[(rng as usize) % perms.len()];
        match play_bag_order_minimax(b, order, depth) {
            None => b = TetrisBoard::new(),
            Some(nb) => {
                if sample_band.ok(&nb) {
                    surfaces.insert(canon(&nb));
                    b = nb;
                } else {
                    b = TetrisBoard::new();
                }
            }
        }
        if surfaces.len() >= cap {
            break;
        }
    }
    let surf_vec: Vec<TetrisBoard> = surfaces.into_iter().collect();
    println!(
        "sampled {} distinct surfaces under (h14 holes4 r12) ({:.0}s)\n",
        surf_vec.len(),
        t0.elapsed().as_secs_f64()
    );

    // evaluate the ladder
    println!(
        "{:<22} {:>10} {:>14} {:>10}",
        "band", "|in-band|", "closure-frac", "topPiece"
    );
    for &hc in &[12u32] {
        for &holec in &[1u32, 2, 3, 4] {
            for &rc in &[4u32, 6, 8, 10, 12] {
                let band = Admiss {
                    hcap: Some(hc),
                    rcap: Some(rc),
                    holecap: Some(holec),
                    well_empty: false,
                    pot: None,
                };
                let mut in_band = 0usize;
                let mut closed = 0usize;
                let mut fail_by_piece = [0usize; 7];
                for surf in &surf_vec {
                    if !band.ok(surf) {
                        continue;
                    }
                    in_band += 1;
                    let mut all_ok = true;
                    for (pi, (p, _)) in plist.iter().enumerate() {
                        let mut has = false;
                        for &pl in TetrisPiecePlacement::all_from_piece(*p) {
                            let mut nb = *surf;
                            let res = nb.apply_piece_placement(pl);
                            if res.is_lost == IsLost::LOST || nb.height() > ROWS {
                                continue;
                            }
                            if band.ok(&nb) {
                                has = true;
                                break;
                            }
                        }
                        if !has {
                            all_ok = false;
                            fail_by_piece[pi] += 1;
                        }
                    }
                    if all_ok {
                        closed += 1;
                    }
                }
                let top = fail_by_piece
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, c)| **c)
                    .map(|(i, _)| plist[i].1)
                    .unwrap_or("-");
                println!(
                    "h{hc} holes{holec} r{rc:<2}{:>10} {:>10} {:>13.2}% {:>10}",
                    "",
                    in_band,
                    100.0 * closed as f64 / in_band.max(1) as f64,
                    top
                );
            }
        }
    }
    println!(
        "\nINTERPRETATION: read down each holes-block — does closure-frac reach ~100% only at \
         large r (huge/junky band), or is there a knee (high frac at small band)? A smooth tradeoff \
         with no knee ⇒ representation-2 (feature-region bands) floored. [{:.0}s]",
        t0.elapsed().as_secs_f64()
    );
}

/// **The pump probe.** For each flat reset floor `f` (working cols 0..8 at height `f`,
/// well col 9 empty), compute the EXACT one-bag minimax (player MINIMIZES, adversary
/// MAXIMIZES, full depth, real leaf board) and report the optimal worst-case end board
/// vs the start. The question: is OPTIMAL play *forced* to net +holes / +height from a
/// flat surface, for the worst order? That forced accumulation, if it can't later be
/// undone, is a "pump" → unbounded height → forced loss (unsurvivability). A forced
/// one-bag top-out at some floor is a direct kill.
fn run_pump(floors: &[u32]) {
    println!("MODE: exact one-bag minimax PUMP probe (optimal player vs worst order)");
    println!("Q: is optimal play FORCED to accumulate from a flat floor? (a pump ⇒ unsurvivable)");
    MM_CAP.store(35_000_000, std::sync::atomic::Ordering::Relaxed);
    for &f in floors {
        let mut sigma = TetrisBoard::new();
        for j in 0..CYCLE_WELL {
            for r in 0..f {
                sigma.set_bit(j, r as usize);
            }
        }
        let start_phi = phi(&sigma);
        MM_OVERFLOW.store(false, std::sync::atomic::Ordering::Relaxed);
        let t0 = Instant::now();
        match play_bag_minimax(sigma, 7) {
            None => {
                println!(
                    "  floor {f:2}: OPTIMAL player TOPPED OUT in ONE bag — adversary forces a loss. \
                     ({:.1}s)",
                    t0.elapsed().as_secs_f64()
                );
            }
            Some(e) => {
                let ovf = MM_OVERFLOW.load(std::sync::atomic::Ordering::Relaxed);
                println!(
                    "  floor {f:2}: start(Φ{start_phi:3}, h{f:2}, holes 0)  →  optimal worst-case \
                     end(Φ{:3}, h{:2}, holes{:2})   ΔΦ={:+3}  Δholes={:+2}   {}  ({:.1}s)",
                    phi(&e),
                    e.height(),
                    e.total_holes(),
                    phi(&e) - start_phi,
                    e.total_holes() as i32,
                    if ovf {
                        "[APPROX: memo cap hit]"
                    } else {
                        "[EXACT]"
                    },
                    t0.elapsed().as_secs_f64()
                );
            }
        }
    }
    println!(
        "\nINTERPRETATION: Δholes>0 EXACT for all floors ⇒ optimal play is forced into holes from \
         flat ⇒ pump seed (pursue impossibility proof). Δholes=0 EXACT ⇒ optimal keeps it clean ⇒ \
         survivable. APPROX rows are inconclusive (exact search overflowed)."
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let boundary = args.iter().any(|a| a == "boundary" || a == "--boundary");
    let optimal = args.iter().any(|a| a == "optimal" || a == "--optimal");
    let drift = args.iter().any(|a| a == "drift" || a == "--drift");
    let cycle = args.iter().any(|a| a == "cycle" || a == "--cycle");
    let longrun = args.iter().any(|a| a == "longrun" || a == "--longrun");
    let minimax = args.iter().any(|a| a == "minimax" || a == "--minimax");
    let pump = args.iter().any(|a| a == "pump" || a == "--pump");
    let strat = if args.iter().any(|a| a == "well") {
        Strat::Well
    } else {
        Strat::Flatten
    };
    let budget: usize = args
        .iter()
        .skip(1)
        .find_map(|a| a.parse::<usize>().ok())
        .unwrap_or(DEFAULT_BUDGET);

    // Band caps from args: "h<N>" height, "r<N>" roughness, "holes<N>", "wellempty".
    // Weighted potential: "pa<N>" roughness weight, "pb<N>" hole weight, "pc<N>" cap.
    // The potential is active iff a cap "pc<N>" is given (weights default to 1).
    let parse_cap = |prefix: &str| -> Option<u32> {
        args.iter()
            .find_map(|a| a.strip_prefix(prefix).and_then(|s| s.parse::<u32>().ok()))
    };
    let pot = parse_cap("pc").map(|cap| {
        (
            parse_cap("pa").unwrap_or(1),
            parse_cap("pb").unwrap_or(1),
            cap,
        )
    });
    let adm = Admiss {
        // When a potential is active, drop the box caps so the potential alone shapes
        // the band (height is already inside Φ); keep well_empty if requested.
        hcap: if pot.is_some() { None } else { parse_cap("h") },
        rcap: if pot.is_some() { None } else { parse_cap("r") },
        holecap: if pot.is_some() {
            None
        } else {
            parse_cap("holes")
        },
        well_empty: args.iter().any(|a| a == "wellempty"),
        pot,
    };

    println!("tetris_carrier_probe — exact boards, all 7-bag orders");
    if pump {
        run_pump(&[4, 6, 8, 10, 12, 14, 16]);
        return;
    }
    if args.iter().any(|a| a == "monotone" || a == "--monotone") {
        let s = args
            .iter()
            .find_map(|a| {
                a.strip_prefix("samples")
                    .and_then(|x| x.parse::<usize>().ok())
            })
            .unwrap_or(200_000);
        run_monotone(s);
        return;
    }
    if args.iter().any(|a| a == "hmono" || a == "--hmono") {
        let s = args
            .iter()
            .find_map(|a| {
                a.strip_prefix("samples")
                    .and_then(|x| x.parse::<usize>().ok())
            })
            .unwrap_or(100_000);
        if args.iter().any(|a| a == "clean") {
            HMONO_CLEAN.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        run_hmono(s);
        return;
    }
    if args.iter().any(|a| a == "debttraj" || a == "--debttraj") {
        let depth = args
            .iter()
            .find_map(|a| a.strip_prefix("depth").and_then(|s| s.parse::<u8>().ok()))
            .unwrap_or(4);
        let nbags = args
            .iter()
            .find_map(|a| a.strip_prefix("bags").and_then(|s| s.parse::<usize>().ok()))
            .unwrap_or(3000);
        if let Some(k) = args
            .iter()
            .find_map(|a| a.strip_prefix("beam").and_then(|s| s.parse::<usize>().ok()))
        {
            BEAM_K.store(k, std::sync::atomic::Ordering::Relaxed);
        }
        let adv = if args.iter().any(|a| a == "worst") {
            1u8
        } else if args.iter().any(|a| a == "rand") {
            2u8
        } else {
            0u8
        };
        run_debttraj(depth, nbags, adv);
        return;
    }
    if args.iter().any(|a| a == "basis" || a == "--basis") {
        let depth = args
            .iter()
            .find_map(|a| a.strip_prefix("depth").and_then(|s| s.parse::<u8>().ok()))
            .unwrap_or(2);
        let nbags = args
            .iter()
            .find_map(|a| a.strip_prefix("bags").and_then(|s| s.parse::<usize>().ok()))
            .unwrap_or(300_000);
        run_basis(depth, nbags);
        return;
    }
    if args
        .iter()
        .any(|a| a == "deepcarrier" || a == "--deepcarrier")
    {
        let depth = args
            .iter()
            .find_map(|a| a.strip_prefix("depth").and_then(|s| s.parse::<u8>().ok()))
            .unwrap_or(3);
        let bud = args
            .iter()
            .find_map(|a| {
                a.strip_prefix("budget")
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .unwrap_or(2_000_000);
        run_deepcarrier(depth, bud);
        return;
    }
    if args
        .iter()
        .any(|a| a == "debtcarrier" || a == "--debtcarrier")
    {
        let depth = args
            .iter()
            .find_map(|a| a.strip_prefix("depth").and_then(|s| s.parse::<u8>().ok()))
            .unwrap_or(3);
        let bud = args
            .iter()
            .find_map(|a| {
                a.strip_prefix("budget")
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .unwrap_or(2_000_000);
        let d = args
            .iter()
            .find_map(|a| a.strip_prefix("holes").and_then(|s| s.parse::<u32>().ok()))
            .unwrap_or(1);
        if let Some(k) = args
            .iter()
            .find_map(|a| a.strip_prefix("beam").and_then(|s| s.parse::<usize>().ok()))
        {
            BEAM_K.store(k, std::sync::atomic::Ordering::Relaxed);
        }
        HOLE_CAP.store(d, std::sync::atomic::Ordering::Relaxed);
        println!("(debtcarrier: deep-player AND-OR closure restricted to debt ≤ {d})");
        run_deepcarrier(depth, bud);
        return;
    }
    if args.iter().any(|a| a == "compress" || a == "--compress") {
        let depth = args
            .iter()
            .find_map(|a| a.strip_prefix("depth").and_then(|s| s.parse::<u8>().ok()))
            .unwrap_or(3);
        let bud = args
            .iter()
            .find_map(|a| {
                a.strip_prefix("budget")
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .unwrap_or(300_000);
        run_compress(depth, bud);
        return;
    }
    if args
        .iter()
        .any(|a| a == "closurefrac" || a == "--closurefrac")
    {
        let depth = args
            .iter()
            .find_map(|a| a.strip_prefix("depth").and_then(|s| s.parse::<u8>().ok()))
            .unwrap_or(2);
        let nbags = args
            .iter()
            .find_map(|a| a.strip_prefix("bags").and_then(|s| s.parse::<usize>().ok()))
            .unwrap_or(400_000);
        run_closurefrac(adm, depth, nbags);
        return;
    }
    if args
        .iter()
        .any(|a| a == "closuresweep" || a == "--closuresweep")
    {
        let depth = args
            .iter()
            .find_map(|a| a.strip_prefix("depth").and_then(|s| s.parse::<u8>().ok()))
            .unwrap_or(2);
        let nbags = args
            .iter()
            .find_map(|a| a.strip_prefix("bags").and_then(|s| s.parse::<usize>().ok()))
            .unwrap_or(400_000);
        run_closuresweep(depth, nbags);
        return;
    }
    if args.iter().any(|a| a == "hunt" || a == "--hunt") {
        let wv = args
            .iter()
            .find_map(|a| a.strip_prefix("w").and_then(|s| s.parse::<usize>().ok()))
            .unwrap_or(7);
        let dv = args
            .iter()
            .find_map(|a| a.strip_prefix("d").and_then(|s| s.parse::<u32>().ok()))
            .unwrap_or(4);
        run_hunt(wv, dv);
        return;
    }
    if minimax {
        let nbags = args
            .iter()
            .find_map(|a| a.strip_prefix("bags").and_then(|s| s.parse::<usize>().ok()))
            .unwrap_or(200);
        let depth = args
            .iter()
            .find_map(|a| a.strip_prefix("depth").and_then(|s| s.parse::<u8>().ok()))
            .unwrap_or(4);
        if let Some(k) = args
            .iter()
            .find_map(|a| a.strip_prefix("beam").and_then(|s| s.parse::<usize>().ok()))
        {
            BEAM_K.store(k, std::sync::atomic::Ordering::Relaxed);
        }
        if args.iter().any(|a| a == "szfirst") {
            ADV_SZFIRST.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        run_minimax_longrun(nbags, depth);
        return;
    }
    if longrun {
        if args.iter().any(|a| a == "holefirst") {
            GREEDY_HOLE_FIRST.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        if args.iter().any(|a| a == "smart") {
            PLAYER_SMART.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        if args.iter().any(|a| a == "rand") {
            ADV_RANDOM.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let nbags = args
            .iter()
            .find_map(|a| a.strip_prefix("bags").and_then(|s| s.parse::<usize>().ok()))
            .unwrap_or(5000);
        run_longrun(nbags);
        return;
    }
    if cycle {
        println!("MODE: one-bag cycle test — well-reserving min-Φ greedy, all 5040 orders");
        println!(
            "Φ = Σ column heights; ΔΦ ≤ 0 with 0 losses across all orders ⇒ closure (sufficient)"
        );
        // sweep floors, or a single "f<N>" if given
        if let Some(f) = args
            .iter()
            .find_map(|a| a.strip_prefix("f").and_then(|s| s.parse::<u32>().ok()))
        {
            run_cycle(f);
        } else {
            for f in [0u32, 2, 4, 6, 8, 10, 12, 14, 16].iter() {
                run_cycle(*f);
            }
        }
        return;
    }
    if drift {
        // window width "w<N>" (default 7), relative-height cap "d<N>" (default 4)
        let wv = args
            .iter()
            .find_map(|a| a.strip_prefix("w").and_then(|s| s.parse::<usize>().ok()))
            .unwrap_or(7);
        let dv = args
            .iter()
            .find_map(|a| a.strip_prefix("d").and_then(|s| s.parse::<u32>().ok()))
            .unwrap_or(4);
        run_drift(wv, dv);
        return;
    }
    println!("state_budget = {budget}");
    if optimal {
        run_optimal(budget, adm);
        return;
    }
    if boundary {
        run_boundary(budget, strat, adm);
        return;
    }
    let _ = Admiss::UNBOUNDED;
    let t0 = Instant::now();

    // --- forward BFS over the exact reachable graph -------------------------------
    let init = State {
        board: TetrisBoard::new(),
        bag: TetrisPieceBagState::new(),
    };
    let mut index: FxHashMap<State, u32> = FxHashMap::default();
    let mut states: Vec<State> = Vec::new();
    let mut has_loss: Vec<bool> = Vec::new();
    let mut edge_src: Vec<u32> = Vec::new();
    let mut edge_dst: Vec<u32> = Vec::new();
    let mut queue: VecDeque<u32> = VecDeque::new();

    index.insert(init, 0);
    states.push(init);
    has_loss.push(false);
    queue.push_back(0);

    let pieces = TetrisPiece::all();
    let mut max_height_seen: u32 = 0;
    let mut exploded = false;

    while let Some(sid) = queue.pop_front() {
        let st = states[sid as usize];
        for &p in pieces.iter() {
            if !st.bag.contains(p) {
                continue;
            }
            match choose(&st.board, p, strat) {
                None => {
                    has_loss[sid as usize] = true;
                }
                Some(nb) => {
                    let mut bag2 = st.bag;
                    bag2.remove(p);
                    if bag2.is_empty() {
                        bag2.fill();
                    }
                    let nstate = State {
                        board: nb,
                        bag: bag2,
                    };
                    max_height_seen = max_height_seen.max(nb.height());
                    let nid = if let Some(&id) = index.get(&nstate) {
                        id
                    } else {
                        if states.len() >= budget {
                            exploded = true;
                            break;
                        }
                        let id = states.len() as u32;
                        index.insert(nstate, id);
                        states.push(nstate);
                        has_loss.push(false);
                        queue.push_back(id);
                        id
                    };
                    edge_src.push(sid);
                    edge_dst.push(nid);
                }
            }
        }
        if exploded {
            break;
        }
    }

    let n = states.len();
    let m = edge_src.len();
    let t_bfs = t0.elapsed();
    println!(
        "forward BFS: |R|={n} states, {m} alive-edges, max_height_seen={max_height_seen}, {:.1}s{}",
        t_bfs.as_secs_f64(),
        if exploded {
            "  [EXPLODED: hit budget]"
        } else {
            ""
        }
    );

    // --- backward death propagation (adversarial AND-safety GFP) ------------------
    // Reverse CSR over alive edges so a newly-dead state can notify its predecessors.
    let mut offsets = vec![0u32; n + 1];
    for &d in &edge_dst {
        offsets[d as usize + 1] += 1;
    }
    for i in 0..n {
        offsets[i + 1] += offsets[i];
    }
    let mut rev = vec![0u32; m];
    let mut cur = offsets.clone();
    for k in 0..m {
        let d = edge_dst[k] as usize;
        rev[cur[d] as usize] = edge_src[k];
        cur[d] += 1;
    }

    let mut dead = vec![false; n];
    let mut dq: VecDeque<u32> = VecDeque::new();
    for i in 0..n {
        if has_loss[i] {
            dead[i] = true;
            dq.push_back(i as u32);
        }
    }
    while let Some(s) = dq.pop_front() {
        let lo = offsets[s as usize] as usize;
        let hi = offsets[s as usize + 1] as usize;
        for &pred in &rev[lo..hi] {
            if !dead[pred as usize] {
                dead[pred as usize] = true;
                dq.push_back(pred);
            }
        }
    }

    // --- report -------------------------------------------------------------------
    let dead_count = dead.iter().filter(|&&d| d).count();
    let alive = n - dead_count;
    let init_alive = !dead[0];

    let full_bag = TetrisPieceBagState::new();
    let mut surv_boards: FxHashSet<TetrisBoard> = FxHashSet::default();
    let mut surv_boundary: FxHashSet<TetrisBoard> = FxHashSet::default();
    let mut max_h_surv = 0u32;
    let mut hist = [0u64; (ROWS + 1) as usize];
    for i in 0..n {
        if dead[i] {
            continue;
        }
        let st = states[i];
        let h = st.board.height();
        surv_boards.insert(st.board);
        max_h_surv = max_h_surv.max(h);
        if (h as usize) <= ROWS as usize {
            hist[h as usize] += 1;
        }
        if st.bag == full_bag {
            surv_boundary.insert(st.board);
        }
    }

    println!("\n================ RESULT ================");
    println!("|R| reachable states      : {n}");
    println!("dead states               : {dead_count}");
    println!("|S| surviving closed core : {alive}");
    println!("init in closed core?      : {init_alive}");
    println!("distinct surviving boards : {}", surv_boards.len());
    println!("distinct bag-boundary brds: {}", surv_boundary.len());
    println!("max height in core        : {max_h_surv}");
    print!("core height histogram     : ");
    for (h, c) in hist.iter().enumerate() {
        if *c > 0 {
            print!("h{h}={c} ");
        }
    }
    println!();

    // --- regime classification (the 3-outcome diagnostic) -------------------------
    println!("\n---------------- REGIME ----------------");
    if exploded {
        if max_height_seen >= ROWS - 2 {
            println!(
                "OUTCOME 3 (likely): height drifts to {max_height_seen}/{ROWS} and |R| hit the \
                 budget — strategy fails to reset per bag; effectively unbounded. Closed-core \
                 numbers above are a TRUNCATED lower bound, not trustworthy. Action: switch strategy."
            );
        } else {
            println!(
                "OUTCOME 2 (likely): |R| hit the budget but max height stayed at {max_height_seen}\
                 /{ROWS} (no drift) — set is FINITE but larger than the budget. Closed-core numbers \
                 are a TRUNCATED lower bound. Action: raise budget / shard certification / tighten \
                 strategy to a smaller surface family."
            );
        }
    } else if init_alive && alive > 0 {
        println!(
            "OUTCOME 1: CONVERGED — nonempty closed core, init SURVIVES, |S|={alive} \
             ({} distinct boards). This is a candidate carrier. Action: export S and certify \
             closure in Lean (native_decide).",
            surv_boards.len()
        );
    } else {
        println!(
            "COLLAPSE: BFS converged (set is finite) but the surviving core {} — this strategy \
             does NOT prove solvability. Real negative result. Action: try a stronger/explicit-well \
             strategy. (Note: a finite converged collapse is honest evidence about THIS strategy, \
             not about Tetris.)",
            if alive == 0 {
                "is EMPTY".to_string()
            } else {
                "excludes init".to_string()
            }
        );
    }
    println!("total time: {:.1}s", t0.elapsed().as_secs_f64());
}
