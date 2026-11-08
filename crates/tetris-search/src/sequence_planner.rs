use std::cmp::{Ordering, Reverse};

use tetris_game::{
    StandardTetris, TetrisBoard, TetrisGameConfig, TetrisPiece, TetrisPiecePlacement,
};
use tetris_utils::{FixedBinMinHeap, HeaplessVec};

use crate::scoring::{TetrisBoardScoreState, height_mse_board_score};

#[derive(Debug, Clone, Copy)]
pub struct TetrisPlanScoreState {
    pub board: TetrisBoard<10, 20>,
    pub recent_lines_cleared: u32,
    pub depth: usize,
    pub placement: TetrisPiecePlacement<10>,
}

#[inline(always)]
pub fn height_mse_plan_score(state: &TetrisPlanScoreState) -> f32 {
    height_mse_board_score(&TetrisBoardScoreState {
        board: state.board,
        recent_lines_cleared: state.recent_lines_cleared,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TetrisSequenceWitness<const DEPTH: usize> {
    pub board: TetrisBoard<10, 20>,
    pub placements: [TetrisPiecePlacement<10>; DEPTH],
    pub nodes_expanded: usize,
    pub terminal_candidates_examined: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TetrisSequencePlanOutcome<const DEPTH: usize> {
    Success(TetrisSequenceWitness<DEPTH>),
    Exhausted {
        nodes_expanded: usize,
        terminal_candidates_examined: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF32(f32);

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrontierEntry {
    node_idx: u32,
    priority: OrderedF32,
    depth: u16,
    board_limbs: [u32; StandardTetris::COLS],
}

impl FrontierEntry {
    #[inline]
    fn new(node_idx: u32, priority: f32, depth: u16, board: TetrisBoard<10, 20>) -> Self {
        Self {
            node_idx,
            priority: OrderedF32(priority),
            depth,
            board_limbs: *board.as_limbs(),
        }
    }
}

impl PartialOrd for FrontierEntry {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FrontierEntry {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| self.depth.cmp(&other.depth))
            .then_with(|| other.board_limbs.cmp(&self.board_limbs))
            .then_with(|| other.node_idx.cmp(&self.node_idx))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CandidateNode {
    board: TetrisBoard<10, 20>,
    placement: TetrisPiecePlacement<10>,
    score: f32,
    recent_lines_cleared: u32,
}

impl Default for CandidateNode {
    fn default() -> Self {
        Self {
            board: TetrisBoard::<10, 20>::new(),
            placement: <TetrisPiecePlacement<10>>::from_index(0),
            score: f32::NEG_INFINITY,
            recent_lines_cleared: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SearchNode {
    board: TetrisBoard<10, 20>,
    depth: u16,
    parent_slot: u32,
    placement: u8,
}

/// Computes the full width-limited best-first tree capacity for a fixed depth.
pub const fn best_first_capacity(width: usize, depth: usize) -> usize {
    let mut total = 1usize;
    let mut layer = 1usize;
    let mut level = 0usize;
    while level < depth {
        layer *= width;
        total += layer;
        level += 1;
    }
    total
}

pub struct TetrisSequencePlanner<const WIDTH: usize, const DEPTH: usize, const NODE_CAPACITY: usize>
{
    nodes: HeaplessVec<SearchNode, NODE_CAPACITY>,
    frontier: FixedBinMinHeap<Reverse<FrontierEntry>, NODE_CAPACITY>,
    nodes_expanded: usize,
    terminal_candidates_examined: usize,
    // Optional predicate applied to full-depth boards. Return `true` to accept.
    filter_fn: Option<fn(TetrisBoard<10, 20>) -> bool>,
}

impl<const WIDTH: usize, const DEPTH: usize, const NODE_CAPACITY: usize> Default
    for TetrisSequencePlanner<WIDTH, DEPTH, NODE_CAPACITY>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const WIDTH: usize, const DEPTH: usize, const NODE_CAPACITY: usize>
    TetrisSequencePlanner<WIDTH, DEPTH, NODE_CAPACITY>
{
    pub const fn new() -> Self {
        Self {
            nodes: HeaplessVec::new(),
            frontier: FixedBinMinHeap::new(),
            nodes_expanded: 0,
            terminal_candidates_examined: 0,
            filter_fn: None,
        }
    }

    /// Heap-allocates a fresh planner without materializing it on the stack.
    ///
    /// The fixed-capacity buffers reach hundreds of megabytes at production
    /// widths, so `Box::new(Self::new())` would overflow the stack, and storing
    /// the planner in const-initialized TLS makes every `thread::spawn` fault in
    /// the full buffer size.
    pub fn new_boxed() -> Box<Self> {
        unsafe { Box::<Self>::new_zeroed().assume_init() }
    }

    /// Set a terminal filter. Full-depth boards where `f(board)` returns `false` are rejected.
    pub fn with_filter(mut self, f: fn(TetrisBoard<10, 20>) -> bool) -> Self {
        self.filter_fn = Some(f);
        self
    }

    pub fn set_filter(&mut self, f: fn(TetrisBoard<10, 20>) -> bool) {
        self.filter_fn = Some(f);
    }

    pub fn clear_filter(&mut self) {
        self.filter_fn = None;
    }

    #[inline]
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.frontier.clear();
        self.nodes_expanded = 0;
        self.terminal_candidates_examined = 0;
    }

    pub fn search<ScoreFn>(
        &mut self,
        start_board: TetrisBoard<10, 20>,
        forced_sequence: &[TetrisPiece; DEPTH],
        score_fn: &ScoreFn,
    ) -> TetrisSequencePlanOutcome<DEPTH>
    where
        ScoreFn: Fn(&TetrisPlanScoreState) -> f32,
    {
        self.reset();
        self.push_root(start_board);

        while let Some(Reverse(entry)) = self.frontier.pop_min() {
            let node = self.nodes[entry.node_idx as usize];
            if node.depth as usize == DEPTH {
                self.terminal_candidates_examined =
                    self.terminal_candidates_examined.saturating_add(1);
                if self.filter_fn.is_none_or(|f| f(node.board)) {
                    return TetrisSequencePlanOutcome::Success(self.build_witness(entry.node_idx));
                }
                continue;
            }

            self.nodes_expanded = self.nodes_expanded.saturating_add(1);
            if !self.expand_node(entry.node_idx, node, forced_sequence, score_fn) {
                return TetrisSequencePlanOutcome::Exhausted {
                    nodes_expanded: self.nodes_expanded,
                    terminal_candidates_examined: self.terminal_candidates_examined,
                };
            }
        }

        TetrisSequencePlanOutcome::Exhausted {
            nodes_expanded: self.nodes_expanded,
            terminal_candidates_examined: self.terminal_candidates_examined,
        }
    }

    pub fn collect_witnesses<ScoreFn, const OUT: usize>(
        &mut self,
        start_board: TetrisBoard<10, 20>,
        forced_sequence: &[TetrisPiece; DEPTH],
        score_fn: &ScoreFn,
        limit: usize,
    ) -> HeaplessVec<TetrisSequenceWitness<DEPTH>, OUT>
    where
        ScoreFn: Fn(&TetrisPlanScoreState) -> f32,
    {
        let mut witnesses = HeaplessVec::new();
        let limit = limit.min(OUT);
        if limit == 0 {
            return witnesses;
        }

        self.reset();
        self.push_root(start_board);

        while let Some(Reverse(entry)) = self.frontier.pop_min() {
            let node = self.nodes[entry.node_idx as usize];
            if node.depth as usize == DEPTH {
                self.terminal_candidates_examined =
                    self.terminal_candidates_examined.saturating_add(1);
                if self.filter_fn.is_none_or(|f| f(node.board)) {
                    let pushed = witnesses.try_push(self.build_witness(entry.node_idx));
                    debug_assert!(pushed, "witness buffer capacity must cover limit");
                    if witnesses.len() >= limit {
                        break;
                    }
                }
                continue;
            }

            self.nodes_expanded = self.nodes_expanded.saturating_add(1);
            if !self.expand_node(entry.node_idx, node, forced_sequence, score_fn) {
                return witnesses;
            }
        }

        witnesses
    }

    #[inline]
    fn push_root(&mut self, start_board: TetrisBoard<10, 20>) {
        let root = SearchNode {
            board: start_board,
            depth: 0,
            parent_slot: u32::MAX,
            placement: 0, // Unused placeholder until a real placement is assigned.
        };
        let pushed = self.nodes.try_push(root);
        debug_assert!(pushed, "planner root must fit in node arena");
        self.frontier.push(Reverse(FrontierEntry::new(
            0,
            f32::INFINITY,
            0,
            start_board,
        )));
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "The node capacity is compile-time bounded by u32 and child_idx is checked against it before insertion."
    )]
    fn expand_node<ScoreFn>(
        &mut self,
        parent_idx: u32,
        node: SearchNode,
        forced_sequence: &[TetrisPiece; DEPTH],
        score_fn: &ScoreFn,
    ) -> bool
    where
        ScoreFn: Fn(&TetrisPlanScoreState) -> f32,
    {
        let _ = const {
            tetris_utils::const_assert!(
                NODE_CAPACITY <= u32::MAX as usize,
                "planner node capacity must fit u32 slots"
            )
        };
        let piece = forced_sequence[node.depth as usize];
        let child_depth = node.depth + 1;
        let mut top_candidates = [CandidateNode::default(); WIDTH];
        let mut count = 0usize;

        for &placement in <TetrisPiecePlacement<10>>::all_from_piece(piece) {
            let mut child_board = node.board;
            let outcome = child_board.apply_piece_placement(placement);
            if outcome.is_lost {
                continue;
            }

            let score = score_fn(&TetrisPlanScoreState {
                board: child_board,
                recent_lines_cleared: outcome.lines_cleared,
                depth: child_depth as usize,
                placement,
            });
            insert_top_candidate(
                &mut top_candidates,
                &mut count,
                CandidateNode {
                    board: child_board,
                    placement,
                    score,
                    recent_lines_cleared: outcome.lines_cleared,
                },
            );
        }

        for child in top_candidates.into_iter().take(count.min(WIDTH)) {
            let child_idx = self.nodes.len();
            if child_idx >= NODE_CAPACITY {
                return false;
            }
            if !self.nodes.try_push(SearchNode {
                board: child.board,
                depth: child_depth,
                parent_slot: parent_idx,
                placement: child.placement.index(),
            }) {
                return false;
            }
            self.frontier.push(Reverse(FrontierEntry::new(
                child_idx as u32,
                child.score,
                child_depth,
                child.board,
            )));
        }

        true
    }

    fn build_witness(&self, final_slot: u32) -> TetrisSequenceWitness<DEPTH> {
        let mut placements = [<TetrisPiecePlacement<10>>::from_index(0); DEPTH];
        let mut cursor = final_slot;
        let mut remaining = DEPTH;
        while cursor != 0 && cursor != u32::MAX {
            let node = self.nodes[cursor as usize];
            remaining -= 1;
            placements[remaining] = <TetrisPiecePlacement<10>>::from_index(node.placement);
            cursor = node.parent_slot;
        }
        debug_assert_eq!(remaining, 0);
        TetrisSequenceWitness {
            board: self.nodes[final_slot as usize].board,
            placements,
            nodes_expanded: self.nodes_expanded,
            terminal_candidates_examined: self.terminal_candidates_examined,
        }
    }
}

fn compare_candidate_nodes(left: &CandidateNode, right: &CandidateNode) -> Ordering {
    left.score
        .total_cmp(&right.score)
        .then_with(|| right.board.as_limbs().cmp(left.board.as_limbs()))
}

fn insert_top_candidate<const WIDTH: usize>(
    top_candidates: &mut [CandidateNode; WIDTH],
    count: &mut usize,
    candidate: CandidateNode,
) {
    let mut insert_at = (*count).min(WIDTH);
    while insert_at > 0
        && compare_candidate_nodes(&candidate, &top_candidates[insert_at.saturating_sub(1)]).is_gt()
    {
        insert_at -= 1;
    }

    if *count < WIDTH {
        for idx in (insert_at..*count).rev() {
            top_candidates[idx + 1] = top_candidates[idx];
        }
        top_candidates[insert_at] = candidate;
        *count += 1;
        return;
    }

    if insert_at == WIDTH {
        return;
    }

    for idx in (insert_at..(WIDTH - 1)).rev() {
        top_candidates[idx + 1] = top_candidates[idx];
    }
    top_candidates[insert_at] = candidate;
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEPTH: usize = 4;
    const WIDTH: usize = 4;
    const NODE_CAPACITY: usize = best_first_capacity(WIDTH, DEPTH);

    fn sequence() -> [TetrisPiece; DEPTH] {
        [
            TetrisPiece::O,
            TetrisPiece::I,
            TetrisPiece::S,
            TetrisPiece::Z,
        ]
    }

    #[test]
    fn best_first_search_returns_full_depth_plan() {
        let mut planner = TetrisSequencePlanner::<WIDTH, DEPTH, NODE_CAPACITY>::new();
        let outcome = planner.search(
            TetrisBoard::<10, 20>::new(),
            &sequence(),
            &height_mse_plan_score,
        );
        assert!(
            matches!(outcome, TetrisSequencePlanOutcome::Success(_)),
            "sequence planner should produce a full script"
        );
        if let TetrisSequencePlanOutcome::Success(witness) = outcome {
            assert_eq!(witness.placements.len(), DEPTH);
            assert!(witness.nodes_expanded > 0);
            assert!(witness.terminal_candidates_examined > 0);
        }
    }

    /// `new_boxed` (zeroed heap allocation) must behave identically to `new`.
    #[test]
    fn new_boxed_matches_new() {
        let mut stack_planner = TetrisSequencePlanner::<WIDTH, DEPTH, NODE_CAPACITY>::new();
        let mut boxed_planner = TetrisSequencePlanner::<WIDTH, DEPTH, NODE_CAPACITY>::new_boxed();
        let stack_outcome = stack_planner.search(
            TetrisBoard::<10, 20>::new(),
            &sequence(),
            &height_mse_plan_score,
        );
        let boxed_outcome = boxed_planner.search(
            TetrisBoard::<10, 20>::new(),
            &sequence(),
            &height_mse_plan_score,
        );
        assert!(
            matches!(stack_outcome, TetrisSequencePlanOutcome::Success(_)),
            "planner should produce a full script"
        );
        assert_eq!(stack_outcome, boxed_outcome);
    }

    #[test]
    fn filter_can_reject_all_full_depth_plans() {
        let mut planner =
            TetrisSequencePlanner::<WIDTH, DEPTH, NODE_CAPACITY>::new().with_filter(|_board| false);
        let outcome = planner.search(
            TetrisBoard::<10, 20>::new(),
            &sequence(),
            &height_mse_plan_score,
        );
        assert!(
            matches!(outcome, TetrisSequencePlanOutcome::Exhausted { .. }),
            "filter should reject every full-depth script"
        );
        if let TetrisSequencePlanOutcome::Exhausted {
            terminal_candidates_examined,
            ..
        } = outcome
        {
            assert!(terminal_candidates_examined > 0);
        }
    }

    #[test]
    fn collect_witnesses_respects_limit() {
        let mut planner = TetrisSequencePlanner::<WIDTH, DEPTH, NODE_CAPACITY>::new();
        let witnesses = planner.collect_witnesses::<_, 4>(
            TetrisBoard::<10, 20>::new(),
            &sequence(),
            &height_mse_plan_score,
            3,
        );
        assert_eq!(witnesses.len(), 3);
    }

    #[test]
    fn tiny_node_capacity_exhausts() {
        let mut planner = TetrisSequencePlanner::<WIDTH, DEPTH, 1>::new();
        let outcome = planner.search(
            TetrisBoard::<10, 20>::new(),
            &sequence(),
            &height_mse_plan_score,
        );
        assert!(matches!(
            outcome,
            TetrisSequencePlanOutcome::Exhausted {
                nodes_expanded: 1,
                ..
            }
        ));
    }
}
