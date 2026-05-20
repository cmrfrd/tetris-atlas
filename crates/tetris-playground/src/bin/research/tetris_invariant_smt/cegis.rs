use std::time::Instant;

use tracing::{info, warn};
use z3::ast::{Ast, BV, Bool};
use z3::{SatResult, Solver};

use tetris_game::{TetrisBoard, TetrisPiece, TetrisPieceBagState};

use crate::config::*;
use crate::encoding::*;

// ---------------------------------------------------------------------------
// Learned constraint types (reused from previous CEGIS)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum LearnedConstraint {
    MaxWellDepth(u32),
    MaxCells(u32),
    MaxColumnsAtMaxHeight(u32),
    MaxConsecutiveHigh(u32),
    MaxHeightSpan(u32),
    ExcludeBoard([u64; BOARD_COLS as usize]),
}

impl std::fmt::Display for LearnedConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxWellDepth(w) => write!(f, "max_well_depth <= {w}"),
            Self::MaxCells(c) => write!(f, "total_cells <= {c}"),
            Self::MaxColumnsAtMaxHeight(k) => write!(f, "cols_at_max_height <= {k}"),
            Self::MaxConsecutiveHigh(k) => write!(f, "consecutive_high <= {k}"),
            Self::MaxHeightSpan(s) => write!(f, "height_span <= {s}"),
            Self::ExcludeBoard(_) => write!(f, "exclude_specific_board"),
        }
    }
}

// ---------------------------------------------------------------------------
// z3 encoding of learned constraints
// ---------------------------------------------------------------------------

pub fn encode_learned(board: &[BV], constraint: &LearnedConstraint) -> Bool {
    match constraint {
        LearnedConstraint::MaxWellDepth(max_w) => encode_max_well_depth(board, *max_w),
        LearnedConstraint::MaxCells(max_c) => encode_max_cells(board, *max_c),
        LearnedConstraint::MaxColumnsAtMaxHeight(max_k) => {
            encode_max_cols_at_max_height(board, *max_k)
        }
        LearnedConstraint::MaxConsecutiveHigh(max_k) => encode_max_consecutive_high(board, *max_k),
        LearnedConstraint::MaxHeightSpan(max_s) => encode_max_height_span(board, *max_s),
        LearnedConstraint::ExcludeBoard(vals) => encode_exclude_board(board, vals),
    }
}

fn bv_umin(a: &BV, b: &BV) -> BV {
    a.bvule(b).ite(a, b)
}

fn encode_max_well_depth(board: &[BV], max_w: u32) -> Bool {
    let n = BOARD_COLS as usize;
    let heights: Vec<BV> = board.iter().map(|col| col_height(col)).collect();
    let max_w_bv = BV::from_u64(max_w as u64, BV_WIDTH);
    let bv_zero = BV::from_u64(0, BV_WIDTH);
    let mut constraints = Vec::new();

    for c in 0..n {
        let neighbor_min = if c == 0 {
            heights[1].clone()
        } else if c == n - 1 {
            heights[c - 1].clone()
        } else {
            bv_umin(&heights[c - 1], &heights[c + 1])
        };
        let ge = neighbor_min.bvuge(&heights[c]);
        let diff = neighbor_min.bvsub(&heights[c]);
        let well_depth = ge.ite(&diff, &bv_zero);
        constraints.push(well_depth.bvule(&max_w_bv));
    }
    Bool::and(&constraints)
}

fn encode_max_cells(board: &[BV], max_c: u32) -> Bool {
    let mut total = BV::from_u64(0, BV_WIDTH);
    for col in board {
        total = total.bvadd(symbolic_popcount(col, MAX_HEIGHT));
    }
    total.bvule(BV::from_u64(max_c as u64, BV_WIDTH))
}

fn encode_max_cols_at_max_height(board: &[BV], max_k: u32) -> Bool {
    let max_h_bv = BV::from_u64(MAX_HEIGHT as u64, BV_WIDTH);
    let one = BV::from_u64(1, BV_WIDTH);
    let zero = BV::from_u64(0, BV_WIDTH);
    let mut count = BV::from_u64(0, BV_WIDTH);
    for col in board {
        let at_max = col_height(col).eq(&max_h_bv);
        count = count.bvadd(at_max.ite(&one, &zero));
    }
    count.bvule(BV::from_u64(max_k as u64, BV_WIDTH))
}

fn encode_max_consecutive_high(board: &[BV], max_k: u32) -> Bool {
    let threshold = BV::from_u64(MAX_HEIGHT.saturating_sub(1) as u64, BV_WIDTH);
    let heights: Vec<BV> = board.iter().map(|col| col_height(col)).collect();
    let n = BOARD_COLS as usize;

    let window = (max_k + 1) as usize;
    if window > n {
        return Bool::new_const("true_literal").ite(
            &Bool::new_const("true_literal"),
            &Bool::new_const("true_literal"),
        );
    }

    let mut constraints = Vec::new();
    for start in 0..=(n - window) {
        let mut at_least_one_low = Vec::new();
        for c in start..(start + window) {
            at_least_one_low.push(heights[c].bvult(&threshold));
        }
        constraints.push(Bool::or(&at_least_one_low));
    }
    Bool::and(&constraints)
}

fn encode_max_height_span(board: &[BV], max_s: u32) -> Bool {
    let heights: Vec<BV> = board.iter().map(|col| col_height(col)).collect();
    let bv_zero = BV::from_u64(0, BV_WIDTH);
    let max_s_bv = BV::from_u64(max_s as u64, BV_WIDTH);

    let mut constraints = Vec::new();
    for i in 0..heights.len() {
        for j in (i + 1)..heights.len() {
            let i_nonzero = heights[i].eq(&bv_zero).not();
            let j_nonzero = heights[j].eq(&bv_zero).not();
            let both_nonzero = Bool::and(&[i_nonzero, j_nonzero]);
            let ij_ok = heights[i].bvsub(&heights[j]).bvule(&max_s_bv);
            let ji_ok = heights[j].bvsub(&heights[i]).bvule(&max_s_bv);
            let diff_ok = Bool::or(&[ij_ok, ji_ok]);
            constraints.push(both_nonzero.implies(&diff_ok));
        }
    }
    Bool::and(&constraints)
}

fn encode_exclude_board(board: &[BV], vals: &[u64; BOARD_COLS as usize]) -> Bool {
    let diffs: Vec<Bool> = board
        .iter()
        .zip(vals.iter())
        .map(|(bv, &val)| bv.eq(BV::from_u64(val, BV_WIDTH)).not())
        .collect();
    Bool::or(&diffs)
}

// ---------------------------------------------------------------------------
// Concrete evaluation of learned constraints
// ---------------------------------------------------------------------------

fn concrete_heights(board: &TetrisBoard) -> [u32; BOARD_COLS as usize] {
    let mut heights = [0u32; BOARD_COLS as usize];
    for c in 0..BOARD_COLS as usize {
        for r in 0..BOARD_ROWS {
            if board.get_bit(c, r as usize) {
                heights[c] = r + 1;
            }
        }
    }
    heights
}

fn board_satisfies_learned(board: &TetrisBoard, constraint: &LearnedConstraint) -> bool {
    let heights = concrete_heights(board);
    let n = BOARD_COLS as usize;
    match constraint {
        LearnedConstraint::MaxWellDepth(max_w) => {
            for c in 0..n {
                let nm = if c == 0 {
                    heights[1]
                } else if c == n - 1 {
                    heights[c - 1]
                } else {
                    heights[c - 1].min(heights[c + 1])
                };
                if nm.saturating_sub(heights[c]) > *max_w {
                    return false;
                }
            }
            true
        }
        LearnedConstraint::MaxCells(max_c) => board.count() <= *max_c,
        LearnedConstraint::MaxColumnsAtMaxHeight(max_k) => {
            heights.iter().filter(|&&h| h == MAX_HEIGHT).count() as u32 <= *max_k
        }
        LearnedConstraint::MaxConsecutiveHigh(max_k) => {
            let threshold = MAX_HEIGHT.saturating_sub(1);
            let mut run = 0u32;
            for &h in &heights {
                if h >= threshold {
                    run += 1;
                    if run > *max_k {
                        return false;
                    }
                } else {
                    run = 0;
                }
            }
            true
        }
        LearnedConstraint::MaxHeightSpan(max_s) => {
            let nonempty: Vec<u32> = heights.iter().copied().filter(|&h| h > 0).collect();
            if nonempty.len() < 2 {
                return true;
            }
            let mn = *nonempty.iter().min().unwrap_or(&0);
            let mx = *nonempty.iter().max().unwrap_or(&0);
            mx - mn <= *max_s
        }
        LearnedConstraint::ExcludeBoard(vals) => {
            !(0..n).all(|c| board_col_bits(board, c) == vals[c])
        }
    }
}

fn board_satisfies_with_learned(
    board: &TetrisBoard,
    base: &InvariantParams,
    learned: &[LearnedConstraint],
) -> bool {
    if !board_satisfies_invariant_with_params(board, base) {
        return false;
    }
    learned.iter().all(|c| board_satisfies_learned(board, c))
}

// ---------------------------------------------------------------------------
// Counterexample analysis
// ---------------------------------------------------------------------------

struct BoardFeatures {
    heights: [u32; BOARD_COLS as usize],
    total_cells: u32,
    max_well_depth: u32,
    cols_at_max_height: u32,
    max_consecutive_high: u32,
    height_span: u32,
}

fn extract_features(board: &TetrisBoard) -> BoardFeatures {
    let heights = concrete_heights(board);
    let n = BOARD_COLS as usize;
    let total_cells = board.count();

    let mut max_well = 0u32;
    for c in 0..n {
        let nm = if c == 0 {
            heights[1]
        } else if c == n - 1 {
            heights[c - 1]
        } else {
            heights[c - 1].min(heights[c + 1])
        };
        max_well = max_well.max(nm.saturating_sub(heights[c]));
    }

    let cols_at_max = heights.iter().filter(|&&h| h == MAX_HEIGHT).count() as u32;

    let threshold = MAX_HEIGHT.saturating_sub(1);
    let mut max_consecutive_high = 0u32;
    let mut run = 0u32;
    for &h in &heights {
        if h >= threshold {
            run += 1;
            max_consecutive_high = max_consecutive_high.max(run);
        } else {
            run = 0;
        }
    }

    let nonempty: Vec<u32> = heights.iter().copied().filter(|&h| h > 0).collect();
    let height_span = if nonempty.len() >= 2 {
        nonempty.iter().max().unwrap() - nonempty.iter().min().unwrap()
    } else {
        0
    };

    BoardFeatures {
        heights,
        total_cells,
        max_well_depth: max_well,
        cols_at_max_height: cols_at_max,
        max_consecutive_high,
        height_span,
    }
}

fn current_bound(existing: &[LearnedConstraint], tag: &str) -> u32 {
    existing
        .iter()
        .filter_map(|c| match (c, tag) {
            (LearnedConstraint::MaxWellDepth(v), "well") => Some(*v),
            (LearnedConstraint::MaxCells(v), "cells") => Some(*v),
            (LearnedConstraint::MaxColumnsAtMaxHeight(v), "cols_max") => Some(*v),
            (LearnedConstraint::MaxConsecutiveHigh(v), "consec") => Some(*v),
            (LearnedConstraint::MaxHeightSpan(v), "span") => Some(*v),
            _ => None,
        })
        .min()
        .unwrap_or(u32::MAX)
}

fn analyze_counterexample(
    board: &TetrisBoard,
    _piece: TetrisPiece,
    existing: &[LearnedConstraint],
) -> LearnedConstraint {
    let f = extract_features(board);

    info!(
        "    analysis: cells={}, well={}, cols_max={}, consec_high={}, span={}",
        f.total_cells,
        f.max_well_depth,
        f.cols_at_max_height,
        f.max_consecutive_high,
        f.height_span
    );

    // Priority 1: well depth
    if f.max_well_depth > 0 {
        let cur = current_bound(existing, "well");
        let bound = f.max_well_depth - 1;
        if bound < cur {
            info!("    -> learned: max_well_depth <= {bound}");
            return LearnedConstraint::MaxWellDepth(bound);
        }
    }

    // Priority 2: consecutive high columns
    if f.max_consecutive_high >= 3 {
        let cur = current_bound(existing, "consec");
        let bound = f.max_consecutive_high - 1;
        if bound < cur && bound >= 2 {
            info!("    -> learned: consecutive_high <= {bound}");
            return LearnedConstraint::MaxConsecutiveHigh(bound);
        }
    }

    // Priority 3: columns at max height
    if f.cols_at_max_height >= 4 {
        let cur = current_bound(existing, "cols_max");
        let bound = f.cols_at_max_height - 1;
        if bound < cur && bound >= 2 {
            info!("    -> learned: cols_at_max_height <= {bound}");
            return LearnedConstraint::MaxColumnsAtMaxHeight(bound);
        }
    }

    // Priority 4: height span
    if f.height_span >= 3 {
        let cur = current_bound(existing, "span");
        let bound = f.height_span - 1;
        if bound < cur && bound >= 2 {
            info!("    -> learned: height_span <= {bound}");
            return LearnedConstraint::MaxHeightSpan(bound);
        }
    }

    // Priority 5: cell count
    let cells_floor = 4u32;
    if f.total_cells > cells_floor {
        let cur = current_bound(existing, "cells");
        let bound = f.total_cells - 1;
        if bound < cur && bound >= cells_floor {
            info!("    -> learned: total_cells <= {bound}");
            return LearnedConstraint::MaxCells(bound);
        }
    }

    // Fallback: exclude specific board
    let mut vals = [0u64; BOARD_COLS as usize];
    for c in 0..BOARD_COLS as usize {
        vals[c] = board_col_bits(board, c);
    }
    info!(
        "    -> learned: exclude_specific_board (heights={:?})",
        &f.heights[..]
    );
    LearnedConstraint::ExcludeBoard(vals)
}

// ---------------------------------------------------------------------------
// Per-bag-state base invariant parameters
// ---------------------------------------------------------------------------

/// Return the base invariant parameters for a given bag state.
///
/// Full bag (7 pieces): tight constraints (reset point — board must be flat).
/// Partial bags (1-6 pieces): looser constraints (temporary excursions allowed).
fn base_for_bag(bag_bits: u8) -> InvariantParams {
    if bag_bits == TetrisPieceBagState::FULL_MASK {
        InvariantParams {
            max_height: FULL_BAG_MAX_HEIGHT,
            max_total_holes: 0,
            max_roughness: FULL_BAG_MAX_ROUGHNESS,
            min_flat_block: 4,
        }
    } else {
        InvariantParams {
            max_height: MAX_HEIGHT,
            max_total_holes: MAX_TOTAL_HOLES,
            max_roughness: MAX_ROUGHNESS,
            min_flat_block: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Transition precomputation
// ---------------------------------------------------------------------------

struct Transition {
    source_bag: u8,
    piece: TetrisPiece,
    target_bag: u8,
}

fn precompute_transitions() -> Vec<Transition> {
    let mut transitions = Vec::new();
    for bag_bits in 1u8..=TetrisPieceBagState::FULL_MASK {
        let bag = TetrisPieceBagState::from(bag_bits);
        for (piece, next_bag) in bag.iter_next_states() {
            transitions.push(Transition {
                source_bag: bag_bits,
                piece,
                target_bag: u8::from(next_bag),
            });
        }
    }
    // Sort by bag size descending (hardest transitions first — more adversary choice)
    transitions.sort_by(|a, b| {
        let sa = a.source_bag.count_ones();
        let sb = b.source_bag.count_ones();
        sb.cmp(&sa)
    });
    transitions
}

// ---------------------------------------------------------------------------
// Single transition inductiveness check
// ---------------------------------------------------------------------------

/// Check whether the transition (source_bag, piece) -> target_bag is inductive.
///
/// Query: exists board satisfying P_source such that ALL placements of piece
/// fail to produce a board satisfying P_target (or cause loss).
///
/// Returns None if safe (UNSAT), or the counterexample board (SAT).
fn check_transition(
    piece: TetrisPiece,
    placements: &[PlacementInfo],
    source_base: &InvariantParams,
    target_base: &InvariantParams,
    source_learned: &[LearnedConstraint],
    target_learned: &[LearnedConstraint],
) -> Option<TetrisBoard> {
    let solver = Solver::new();
    let board = make_symbolic_board("b");

    // Source: source base invariant + source learned constraints
    solver.assert(encode_invariant_with_params(&board, source_base));
    for c in source_learned {
        solver.assert(encode_learned(&board, c));
    }

    // Piece placements for this piece type
    let piece_placements: Vec<&PlacementInfo> =
        placements.iter().filter(|p| p.piece == piece).collect();

    // Each placement "fails" if: result violates target invariant OR causes loss
    let mut fail_constraints: Vec<Bool> = Vec::new();
    for info in &piece_placements {
        let enc = encode_placement(&board, info);

        // Target: target base invariant + target learned constraints on result board
        let mut ok_parts = vec![encode_invariant_with_params(&enc.result_board, target_base)];
        for c in target_learned {
            ok_parts.push(encode_learned(&enc.result_board, c));
        }
        let result_ok = Bool::and(&ok_parts);

        let fails = Bool::or(&[result_ok.not(), enc.is_lost]);
        fail_constraints.push(fails);
    }

    // Assert ALL placements fail
    solver.assert(Bool::and(&fail_constraints));

    match solver.check() {
        SatResult::Unsat => None,
        SatResult::Sat => {
            let model = solver.get_model().unwrap();
            let mut tb = TetrisBoard::EMPTY_BOARD;
            for (i, col_bv) in board.iter().enumerate() {
                if let Some(val) = model.eval(col_bv, true) {
                    if let Some(bits) = val.as_u64() {
                        for r in 0..BOARD_ROWS {
                            if bits & (1u64 << r) != 0 {
                                tb.set_bit(i, r as usize);
                            }
                        }
                    }
                }
            }
            Some(tb)
        }
        SatResult::Unknown => {
            warn!("z3 returned UNKNOWN");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Bag-aware CEGIS main loop
// ---------------------------------------------------------------------------

/// Run bag-aware CEGIS: find per-bag-state invariants proving infinite play.
///
/// For each non-empty bag state B, we maintain an invariant P_B(board).
/// The check for each transition (B, piece p) -> B' is:
///   forall board in P_B: exists placement of p: result in P_{B'} and not lost.
///
/// If all 448 transitions are safe, we have an inductive chain of invariants
/// proving infinite play under the 7-bag randomizer.
pub fn run_bag_cegis(placements: &[PlacementInfo], max_rounds: u32) -> bool {
    let transitions = precompute_transitions();

    let full_base = base_for_bag(TetrisPieceBagState::FULL_MASK);
    let partial_base = base_for_bag(1); // any non-full bag
    info!(
        "bag-aware CEGIS: {} transitions over {} bag states",
        transitions.len(),
        (1u8..=TetrisPieceBagState::FULL_MASK).count()
    );
    info!(
        "  full-bag base  = height <= {}, holes <= {}, roughness <= {}",
        full_base.max_height, full_base.max_total_holes, full_base.max_roughness
    );
    info!(
        "  partial base   = height <= {}, holes <= {}, roughness <= {}",
        partial_base.max_height, partial_base.max_total_holes, partial_base.max_roughness
    );

    // Per-bag-state learned constraints. Index = bag_bits (0 unused, 1..=127 valid).
    let mut learned: Vec<Vec<LearnedConstraint>> = vec![Vec::new(); 128];

    let start = Instant::now();
    let full_bag = TetrisPieceBagState::FULL_MASK;

    // Stuck detection: consecutive board exclusions on the same bag state
    let mut consecutive_excludes: u32 = 0;
    let mut last_exclude_bag: u8 = 0;

    for round in 0..max_rounds {
        let total_constraints: usize = learned.iter().map(|v| v.len()).sum();
        let bags_with_constraints = learned.iter().filter(|v| !v.is_empty()).count();
        info!(
            "=== round {} ({:.1}s, {} constraints across {} bags) ===",
            round,
            start.elapsed().as_secs_f64(),
            total_constraints,
            bags_with_constraints,
        );

        let mut all_safe = true;

        for t in &transitions {
            let bag_size = t.source_bag.count_ones();

            let source_base = base_for_bag(t.source_bag);
            let target_base = base_for_bag(t.target_bag);

            let t0 = Instant::now();
            let result = check_transition(
                t.piece,
                placements,
                &source_base,
                &target_base,
                &learned[t.source_bag as usize],
                &learned[t.target_bag as usize],
            );
            let elapsed = t0.elapsed().as_secs_f64();

            match result {
                None => {
                    // Safe — log slow checks
                    if elapsed > 2.0 {
                        info!(
                            "  bag={:07b}({}p) {:?} -> {:07b} SAFE ({:.1}s)",
                            t.source_bag, bag_size, t.piece, t.target_bag, elapsed
                        );
                    }
                }
                Some(cex) => {
                    let bag_display = TetrisPieceBagState::from(t.source_bag);
                    warn!(
                        "  bag={:07b}({}p) {:?} -> {:07b} CEX ({:.1}s)  {}",
                        t.source_bag, bag_size, t.piece, t.target_bag, elapsed, bag_display
                    );
                    print_counterexample(&cex);

                    let constraint =
                        analyze_counterexample(&cex, t.piece, &learned[t.source_bag as usize]);

                    // Stuck detection
                    if matches!(constraint, LearnedConstraint::ExcludeBoard(_)) {
                        if last_exclude_bag == t.source_bag {
                            consecutive_excludes += 1;
                        } else {
                            consecutive_excludes = 1;
                            last_exclude_bag = t.source_bag;
                        }
                        if consecutive_excludes >= 30 {
                            warn!(
                                "30 consecutive board exclusions on bag {:07b}. Stuck.",
                                t.source_bag
                            );
                            print_summary(&learned);
                            return false;
                        }
                    } else {
                        consecutive_excludes = 0;
                    }

                    learned[t.source_bag as usize].push(constraint);

                    // Verify empty board still satisfies P_full
                    let empty = TetrisBoard::EMPTY_BOARD;
                    if !board_satisfies_with_learned(
                        &empty,
                        &base_for_bag(full_bag),
                        &learned[full_bag as usize],
                    ) {
                        warn!("Empty board excluded from full-bag invariant. Failed.");
                        print_summary(&learned);
                        return false;
                    }

                    // Verify this bag state's invariant is still satisfiable
                    if !bag_invariant_satisfiable(&source_base, &learned[t.source_bag as usize]) {
                        warn!(
                            "Invariant for bag {:07b} became empty. Failed.",
                            t.source_bag
                        );
                        print_summary(&learned);
                        return false;
                    }

                    all_safe = false;
                    break; // restart round
                }
            }
        }

        if all_safe {
            info!(
                "=== SUCCESS — bag-aware inductive invariant found ({} rounds, {:.1}s) ===",
                round + 1,
                start.elapsed().as_secs_f64()
            );
            info!("For each bag state, for each adversary piece choice,");
            info!(
                "the player has a placement keeping the board in the next bag-state's invariant."
            );
            info!("This proves infinite play under the 7-bag randomizer.");
            print_summary(&learned);
            return true;
        }
    }

    warn!(
        "Did not converge within {} rounds ({:.1}s).",
        max_rounds,
        start.elapsed().as_secs_f64()
    );
    print_summary(&learned);
    false
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bag_invariant_satisfiable(base: &InvariantParams, learned: &[LearnedConstraint]) -> bool {
    let solver = Solver::new();
    let board = make_symbolic_board("sat");
    solver.assert(encode_invariant_with_params(&board, base));
    for c in learned {
        solver.assert(encode_learned(&board, c));
    }
    solver.check() == SatResult::Sat
}

fn print_counterexample(board: &TetrisBoard) {
    let heights = concrete_heights(board);
    let n = BOARD_COLS as usize;
    let holes: Vec<u32> = (0..n)
        .map(|c| {
            let h = heights[c];
            let filled = (0..h).filter(|&r| board.get_bit(c, r as usize)).count() as u32;
            h - filled
        })
        .collect();

    info!("    heights: {:?}", &heights[..]);
    info!("    holes:   {:?}", holes);
    info!("    cells:   {}", board.count());

    let max_h = *heights.iter().max().unwrap_or(&0);
    for r in (0..max_h).rev() {
        let mut row_str = String::from("    ");
        for c in 0..n {
            if board.get_bit(c, r as usize) {
                row_str.push('#');
            } else {
                row_str.push('.');
            }
        }
        info!("{}", row_str);
    }
}

fn print_summary(learned: &[Vec<LearnedConstraint>]) {
    let total: usize = learned.iter().map(|v| v.len()).sum();
    let bags_used = learned.iter().filter(|v| !v.is_empty()).count();
    info!(
        "Summary: {} total constraints across {} bag states",
        total, bags_used
    );
    for bag_bits in 1u8..=TetrisPieceBagState::FULL_MASK {
        let constraints = &learned[bag_bits as usize];
        if !constraints.is_empty() {
            let bag = TetrisPieceBagState::from(bag_bits);
            info!(
                "  bag={:07b} ({}p) {}: {} constraints",
                bag_bits,
                bag_bits.count_ones(),
                bag,
                constraints.len()
            );
            for (i, c) in constraints.iter().enumerate() {
                info!("    [{i}] {c}");
            }
        }
    }
}
