use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

use tetris_game::{
    IsLost, StandardTetris, TetrisBoard, TetrisGameConfig, TetrisPiece, TetrisPiecePlacement,
    constants,
};
use tetris_utils::repeat_idx_unroll;

#[derive(Debug, Clone, Copy)]
struct ScoreState {
    board: TetrisBoard,
    recent_lines_cleared: u32,
}

#[inline(always)]
fn score_board(state: &ScoreState) -> f32 {
    let lines = state.recent_lines_cleared as f32;
    let holes = state.board.total_holes() as f32;
    let heights = state.board.heights();

    let mut aggregate_height = 0.0;
    repeat_idx_unroll!(StandardTetris::COLS, I, {
        aggregate_height += heights[I] as f32;
    });

    let mut bumpiness = 0.0;
    repeat_idx_unroll!(StandardTetris::COLS - 1, I, {
        bumpiness += (heights[I] as f32 - heights[I + 1] as f32).abs();
    });

    0.760666 * lines + (-0.510066) * aggregate_height + (-0.35663) * holes + (-0.184483) * bumpiness
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF32(f32);

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[derive(Debug, Clone, Copy)]
struct SearchNode {
    board: TetrisBoard,
    depth: u16,
    parent_slot: u32,
    placement: u8,
    lines_cleared: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrontierEntry {
    node_idx: u32,
    priority: OrderedF32,
    depth: u16,
}

impl FrontierEntry {
    fn new(node_idx: u32, priority: f32, depth: u16) -> Self {
        Self {
            node_idx,
            priority: OrderedF32(priority),
            depth,
        }
    }
}

impl PartialOrd for FrontierEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FrontierEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| self.depth.cmp(&other.depth))
            .then_with(|| other.node_idx.cmp(&self.node_idx))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CandidateNode {
    board: TetrisBoard,
    placement: u8,
    score: f32,
    lines_cleared: u32,
}

impl Default for CandidateNode {
    fn default() -> Self {
        Self {
            board: TetrisBoard::EMPTY_BOARD,
            placement: TetrisPiecePlacement::default().index(),
            score: f32::NEG_INFINITY,
            lines_cleared: 0,
        }
    }
}

fn compare_candidates(left: &CandidateNode, right: &CandidateNode) -> Ordering {
    left.score
        .total_cmp(&right.score)
        .then_with(|| right.board.as_limbs().cmp(&left.board.as_limbs()))
}

fn insert_top_candidate(top: &mut Vec<CandidateNode>, width: usize, candidate: CandidateNode) {
    let count = top.len();
    let mut insert_at = count.min(width);
    while insert_at > 0 && compare_candidates(&candidate, &top[insert_at.saturating_sub(1)]).is_gt()
    {
        insert_at -= 1;
    }

    if count < width {
        top.insert(insert_at, candidate);
        return;
    }

    if insert_at == width {
        return;
    }

    top.pop();
    top.insert(insert_at, candidate);
}

pub struct SearchResult {
    pub placements: Vec<TetrisPiecePlacement>,
    pub final_board: TetrisBoard,
    pub lines_cleared: u32,
    pub nodes_expanded: usize,
}

pub enum SearchOutcome {
    Found(SearchResult),
    Exhausted { nodes_expanded: usize },
}

pub fn search(
    sequence: &[TetrisPiece],
    target: TetrisBoard,
    width: usize,
    max_nodes: usize,
    verbose: bool,
) -> SearchOutcome {
    let total_depth = sequence.len();
    let mut arena: Vec<SearchNode> = Vec::with_capacity(max_nodes.min(1 << 20));
    let mut frontier: BinaryHeap<FrontierEntry> = BinaryHeap::new();

    let root = SearchNode {
        board: TetrisBoard::EMPTY_BOARD,
        depth: 0,
        parent_slot: u32::MAX,
        placement: TetrisPiecePlacement::default().index(),
        lines_cleared: 0,
    };
    arena.push(root);
    frontier.push(FrontierEntry::new(0, f32::INFINITY, 0));

    let mut nodes_expanded: usize = 0;
    let mut best_depth: u16 = 0;
    let mut terminal_checked: usize = 0;
    let search_start = Instant::now();
    let mut last_log = search_start;

    while let Some(entry) = frontier.pop() {
        let node = arena[entry.node_idx as usize];

        if node.depth as usize == total_depth {
            terminal_checked += 1;
            if node.board == target {
                let placements = build_path(&arena, entry.node_idx, total_depth);
                return SearchOutcome::Found(SearchResult {
                    placements,
                    final_board: node.board,
                    lines_cleared: node.lines_cleared,
                    nodes_expanded,
                });
            }
            continue;
        }

        nodes_expanded += 1;

        if verbose {
            let mut should_log = false;
            if node.depth > best_depth {
                best_depth = node.depth;
                should_log = true;
            }
            let now = Instant::now();
            if now.duration_since(last_log).as_secs() >= 2 {
                should_log = true;
            }
            if should_log {
                last_log = now;
                let elapsed = now.duration_since(search_start);
                eprintln!(
                    "  [{:.1?}] best_depth={}/{} expanded={} terminal_checked={} arena={} frontier={} score={:.4}",
                    elapsed,
                    best_depth,
                    total_depth,
                    nodes_expanded,
                    terminal_checked,
                    arena.len(),
                    frontier.len(),
                    entry.priority.0,
                );
            }
        }

        let piece = sequence[node.depth as usize];
        let mut top_candidates: Vec<CandidateNode> = Vec::with_capacity(width);

        for &placement in TetrisPiecePlacement::all_from_piece(piece) {
            let mut child_board = node.board;
            let outcome = child_board.apply_piece_placement(placement);
            if outcome.is_lost == IsLost::LOST {
                continue;
            }
            let cumulative_lines = node.lines_cleared + outcome.lines_cleared;
            let score = score_board(&ScoreState {
                board: child_board,
                recent_lines_cleared: outcome.lines_cleared,
            });
            insert_top_candidate(
                &mut top_candidates,
                width,
                CandidateNode {
                    board: child_board,
                    placement: placement.index(),
                    score,
                    lines_cleared: cumulative_lines,
                },
            );
        }

        for child in &top_candidates {
            if arena.len() >= max_nodes {
                return SearchOutcome::Exhausted { nodes_expanded };
            }
            let child_depth = node.depth + 1;
            let child_idx = arena.len() as u32;
            arena.push(SearchNode {
                board: child.board,
                depth: child_depth,
                parent_slot: entry.node_idx,
                placement: child.placement,
                lines_cleared: child.lines_cleared,
            });
            frontier.push(FrontierEntry::new(child_idx, child.score, child_depth));
        }
    }

    SearchOutcome::Exhausted { nodes_expanded }
}

fn build_path(arena: &[SearchNode], final_slot: u32, depth: usize) -> Vec<TetrisPiecePlacement> {
    let mut placements = vec![TetrisPiecePlacement::default(); depth];
    let mut cursor = final_slot;
    let mut remaining = depth;
    while cursor != 0 && cursor != u32::MAX {
        let node = &arena[cursor as usize];
        remaining -= 1;
        placements[remaining] = TetrisPiecePlacement::from_index(node.placement);
        cursor = node.parent_slot;
    }
    debug_assert_eq!(remaining, 0);
    placements
}
