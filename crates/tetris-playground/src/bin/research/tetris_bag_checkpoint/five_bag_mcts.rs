//! Typed MCTS experiments for five-bag cycle searches.
//!
//! This module keeps the tree-search machinery separate from the CLI plumbing in
//! `five_bag_cycle`. The search is still a research tool, not a proof checker:
//! it can find witnesses and compare sampling policies, but closure must still
//! be certified by the exact verifier.

use anyhow::{Result, bail};
use clap::ValueEnum;
use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHasher};
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};

use super::common::{BoardKey, board_key, has_too_many_holes};

/// MCTS implementation selected by the five-bag CLI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum MctsPolicy {
    /// Transposition-table PUCT search.
    Puct,
    /// Legacy independent weighted rollouts.
    Rollout,
}

/// Terminal predicate used by MCTS experiments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum MctsTargetKind {
    /// Final board must be empty.
    Empty,
    /// Final board must have one of `--target-cells`.
    CellCount,
    /// Final board must satisfy low-board height/hole bounds.
    LowBoard,
}

/// How terminal misses are scored during PUCT backup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum MctsTerminalMode {
    /// Terminal misses keep the same shaped score as non-terminal leaves.
    Shaped,
    /// A terminal miss is a hard failure, even if it is close to the target.
    EmptyOrBust,
}

/// Height and hole bounds applied after every placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoardConstraints {
    max_height: u32,
    max_holes: u32,
}

impl BoardConstraints {
    pub const fn new(max_height: u32, max_holes: u32) -> Self {
        Self {
            max_height,
            max_holes,
        }
    }

    pub fn max_height(self) -> u32 {
        self.max_height
    }

    pub fn max_holes(self) -> u32 {
        self.max_holes
    }

    pub fn allows_after_placement(self, board: &TetrisBoard, is_lost: IsLost) -> bool {
        is_lost != IsLost::LOST
            && board.height() <= self.max_height
            && (self.max_holes == u32::MAX || !has_too_many_holes(board, self.max_holes))
    }

    pub fn allows_board(self, board: &TetrisBoard) -> bool {
        board.height() <= self.max_height
            && (self.max_holes == u32::MAX || !has_too_many_holes(board, self.max_holes))
    }
}

/// Validated set of acceptable terminal cell counts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellCountSet(Vec<u32>);

impl CellCountSet {
    pub fn new(mut cells: Vec<u32>) -> Result<Self> {
        if cells.is_empty() {
            bail!("--target cell-count requires at least one --target-cells value");
        }
        cells.sort_unstable();
        cells.dedup();
        for &cell_count in &cells {
            if cell_count > 200 {
                bail!("target cell count {cell_count} is outside a 10x20 board");
            }
        }
        Ok(Self(cells))
    }

    fn contains(&self, cell_count: u32) -> bool {
        self.0.binary_search(&cell_count).is_ok()
    }

    fn distance(&self, cell_count: u32) -> u32 {
        self.0
            .iter()
            .map(|&target| target.abs_diff(cell_count))
            .min()
            .unwrap_or(u32::MAX)
    }

    fn as_slice(&self) -> &[u32] {
        &self.0
    }
}

/// Typed terminal objective for an MCTS experiment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetSpec {
    Empty,
    CellCount {
        cells: CellCountSet,
        max_height: Option<u32>,
        max_holes: Option<u32>,
    },
    LowBoard {
        max_height: u32,
        max_holes: u32,
    },
}

impl TargetSpec {
    pub fn from_cli(
        kind: MctsTargetKind,
        cells: Vec<u32>,
        max_height: Option<u32>,
        max_holes: Option<u32>,
    ) -> Result<Self> {
        match kind {
            MctsTargetKind::Empty => {
                if !cells.is_empty() || max_height.is_some() || max_holes.is_some() {
                    bail!(
                        "--target empty does not use --target-cells, --target-max-height, or --target-max-holes"
                    );
                }
                Ok(Self::Empty)
            }
            MctsTargetKind::CellCount => Ok(Self::CellCount {
                cells: CellCountSet::new(cells)?,
                max_height,
                max_holes,
            }),
            MctsTargetKind::LowBoard => {
                if !cells.is_empty() {
                    bail!("--target low-board does not use --target-cells");
                }
                Ok(Self::LowBoard {
                    max_height: max_height.unwrap_or(1),
                    max_holes: max_holes.unwrap_or(0),
                })
            }
        }
    }

    pub fn kind(&self) -> MctsTargetKind {
        match self {
            Self::Empty => MctsTargetKind::Empty,
            Self::CellCount { .. } => MctsTargetKind::CellCount,
            Self::LowBoard { .. } => MctsTargetKind::LowBoard,
        }
    }

    pub fn is_success(&self, board: &TetrisBoard) -> bool {
        match self {
            Self::Empty => board.count() == 0,
            Self::CellCount {
                cells,
                max_height,
                max_holes,
            } => {
                cells.contains(board.count())
                    && max_height.is_none_or(|height| board.height() <= height)
                    && max_holes
                        .is_none_or(|holes| holes == u32::MAX || !has_too_many_holes(board, holes))
            }
            Self::LowBoard {
                max_height,
                max_holes,
            } => {
                board.height() <= *max_height
                    && (*max_holes == u32::MAX || !has_too_many_holes(board, *max_holes))
            }
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Empty => "empty board".to_owned(),
            Self::CellCount {
                cells,
                max_height,
                max_holes,
            } => format!(
                "cell-count {:?}, target_max_height={:?}, target_max_holes={:?}",
                cells.as_slice(),
                max_height,
                max_holes
            ),
            Self::LowBoard {
                max_height,
                max_holes,
            } => format!("low-board height<={max_height}, holes<={max_holes}"),
        }
    }

    fn terminal_score(
        &self,
        board: &TetrisBoard,
        mode: MctsTerminalMode,
        repeat_count: u32,
        repeat_penalty: f32,
    ) -> f32 {
        if self.is_success(board) {
            return 1.0;
        }

        let novelty_penalty = repeat_count as f32 * repeat_penalty;
        match mode {
            MctsTerminalMode::Shaped => self.progress_score(board).min(0.95) - novelty_penalty,
            MctsTerminalMode::EmptyOrBust => {
                -1.0 - self.terminal_failure_penalty(board) - novelty_penalty
            }
        }
    }

    fn progress_score(&self, board: &TetrisBoard) -> f32 {
        let cells = board.count();
        let height = board.height();
        let holes = count_holes(board);

        let raw = match self {
            Self::Empty => 1.0 - cells as f32 / 80.0 - height as f32 / 40.0 - holes as f32 / 60.0,
            Self::CellCount { cells: targets, .. } => {
                let distance = targets.distance(cells);
                1.0 - distance as f32 / 60.0 - height as f32 / 50.0 - holes as f32 / 80.0
            }
            Self::LowBoard {
                max_height,
                max_holes,
            } => {
                let excess_height = height.saturating_sub(*max_height);
                let excess_holes = holes.saturating_sub(*max_holes);
                1.0 - excess_height as f32 / 20.0 - excess_holes as f32 / 40.0
            }
        };

        raw.clamp(-1.0, 1.0)
    }

    fn terminal_failure_penalty(&self, board: &TetrisBoard) -> f32 {
        let cells = board.count();
        let height = board.height();
        let holes = count_holes(board);

        match self {
            Self::Empty => cells.min(80) as f32 / 40.0 + height as f32 / 80.0 + holes as f32 / 80.0,
            Self::CellCount { cells: targets, .. } => {
                targets.distance(cells) as f32 / 40.0 + height as f32 / 80.0 + holes as f32 / 80.0
            }
            Self::LowBoard {
                max_height,
                max_holes,
            } => {
                let excess_height = height.saturating_sub(*max_height);
                let excess_holes = holes.saturating_sub(*max_holes);
                excess_height as f32 / 20.0 + excess_holes as f32 / 40.0
            }
        }
    }
}

/// Configuration for transposition-table PUCT.
#[derive(Clone, Debug)]
pub struct PuctConfig {
    pub start_board: TetrisBoard,
    pub constraints: BoardConstraints,
    pub target: TargetSpec,
    pub iterations: u64,
    pub exploration: f32,
    pub prior_temperature: f32,
    pub prior_noise: f32,
    pub max_children: usize,
    pub terminal_mode: MctsTerminalMode,
    pub terminal_repeat_penalty: f32,
    pub noise_seed: u64,
}

/// Snapshot emitted by long-running PUCT searches.
#[derive(Clone, Copy, Debug)]
pub struct PuctProgress {
    pub iterations: u64,
    pub tree_nodes: usize,
    pub terminal_evaluations: u64,
    pub best_cells: u32,
    pub best_score: f32,
}

impl PuctConfig {
    pub fn validate(&self, piece_count: usize) -> Result<()> {
        if piece_count == 0 && !self.target.is_success(&self.start_board) {
            bail!("zero-piece PUCT search has no legal way to change a non-target start board");
        }
        if self.iterations == 0 {
            bail!("--mcts-iterations must be greater than zero");
        }
        if self.max_children == 0 {
            bail!("--mcts-max-children must be greater than zero");
        }
        if !self.exploration.is_finite() || self.exploration < 0.0 {
            bail!("--mcts-exploration must be finite and non-negative");
        }
        if !self.prior_temperature.is_finite() || self.prior_temperature <= 0.0 {
            bail!("--mcts-prior-temperature must be finite and positive");
        }
        if !self.prior_noise.is_finite() || self.prior_noise < 0.0 {
            bail!("--mcts-prior-noise must be finite and non-negative");
        }
        if !self.terminal_repeat_penalty.is_finite() || self.terminal_repeat_penalty < 0.0 {
            bail!("--mcts-terminal-repeat-penalty must be finite and non-negative");
        }
        if !self.constraints.allows_board(&self.start_board) {
            bail!("start board violates the configured board constraints");
        }
        Ok(())
    }
}

/// Aggregate result for one PUCT run over a fixed piece sequence.
#[derive(Clone, Debug)]
pub struct PuctSearchResult {
    pub success: bool,
    pub best_cells: u32,
    pub best_score: f32,
    pub iterations: u64,
    pub tree_nodes: usize,
    pub terminal_evaluations: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct StateKey {
    board: BoardKey,
    step: usize,
}

#[derive(Clone, Copy, Debug)]
struct SearchState {
    board: TetrisBoard,
    step: usize,
}

#[derive(Clone, Debug)]
struct SearchNode {
    state: SearchState,
    visits: u32,
    value_sum: f32,
    expanded: bool,
    children: Vec<ActionEdge>,
}

impl SearchNode {
    fn new(state: SearchState) -> Self {
        Self {
            state,
            visits: 0,
            value_sum: 0.0,
            expanded: false,
            children: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
struct ActionEdge {
    #[allow(dead_code)]
    placement: TetrisPiecePlacement,
    child: usize,
    visits: u32,
    value_sum: f32,
    prior: f32,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    placement: TetrisPiecePlacement,
    board: TetrisBoard,
    cost: f32,
}

struct PuctSearch<'a> {
    pieces: &'a [TetrisPiece],
    config: &'a PuctConfig,
    nodes: Vec<SearchNode>,
    index: FxHashMap<StateKey, usize>,
    best_cells: u32,
    best_score: f32,
    success: bool,
    terminal_evaluations: u64,
    terminal_seen: FxHashMap<BoardKey, u32>,
}

impl<'a> PuctSearch<'a> {
    fn new(pieces: &'a [TetrisPiece], config: &'a PuctConfig) -> Self {
        let root_state = SearchState {
            board: config.start_board,
            step: 0,
        };
        let root_key = state_key(root_state);
        let mut index = FxHashMap::default();
        index.insert(root_key, 0);
        Self {
            pieces,
            config,
            nodes: vec![SearchNode::new(root_state)],
            index,
            best_cells: u32::MAX,
            best_score: f32::NEG_INFINITY,
            success: false,
            terminal_evaluations: 0,
            terminal_seen: FxHashMap::default(),
        }
    }

    fn search(
        mut self,
        progress_every: u64,
        mut on_progress: impl FnMut(PuctProgress),
    ) -> PuctSearchResult {
        let mut iterations = 0;
        for _ in 0..self.config.iterations {
            iterations += 1;
            self.run_iteration();
            if progress_every > 0 && iterations % progress_every == 0 {
                on_progress(self.progress(iterations));
            }
            if self.success {
                break;
            }
        }

        PuctSearchResult {
            success: self.success,
            best_cells: self.best_cells,
            best_score: self.best_score,
            iterations,
            tree_nodes: self.nodes.len(),
            terminal_evaluations: self.terminal_evaluations,
        }
    }

    fn progress(&self, iterations: u64) -> PuctProgress {
        PuctProgress {
            iterations,
            tree_nodes: self.nodes.len(),
            terminal_evaluations: self.terminal_evaluations,
            best_cells: self.best_cells,
            best_score: self.best_score,
        }
    }

    fn run_iteration(&mut self) {
        let mut node_id = 0;
        let mut path = Vec::new();

        let value = loop {
            if self.is_terminal(node_id) {
                break self.evaluate_terminal(node_id);
            }

            if !self.nodes[node_id].expanded {
                let expanded = self.expand_node(node_id);
                if expanded {
                    break self.evaluate_leaf(node_id);
                }
                break -1.0;
            }

            if self.nodes[node_id].children.is_empty() {
                break -1.0;
            }

            let edge_id = self.select_child(node_id);
            let child_id = self.nodes[node_id].children[edge_id].child;
            path.push((node_id, edge_id));
            node_id = child_id;
        };

        self.nodes[node_id].visits = self.nodes[node_id].visits.saturating_add(1);
        self.nodes[node_id].value_sum += value;

        for (parent_id, edge_id) in path {
            self.nodes[parent_id].visits = self.nodes[parent_id].visits.saturating_add(1);
            self.nodes[parent_id].value_sum += value;
            let edge = &mut self.nodes[parent_id].children[edge_id];
            edge.visits = edge.visits.saturating_add(1);
            edge.value_sum += value;
        }
    }

    fn is_terminal(&self, node_id: usize) -> bool {
        self.nodes[node_id].state.step == self.pieces.len()
    }

    fn evaluate_terminal(&mut self, node_id: usize) -> f32 {
        self.terminal_evaluations += 1;
        let board = self.nodes[node_id].state.board;
        let cells = board.count();
        let board_key = board_key(&board);
        let repeat_count = *self.terminal_seen.get(&board_key).unwrap_or(&0);
        let score = self.config.target.terminal_score(
            &board,
            self.config.terminal_mode,
            repeat_count,
            self.config.terminal_repeat_penalty,
        );
        *self.terminal_seen.entry(board_key).or_insert(0) += 1;

        self.best_cells = self.best_cells.min(cells);
        self.best_score = self.best_score.max(score);
        self.success |= self.config.target.is_success(&board);

        score
    }

    fn evaluate_leaf(&self, node_id: usize) -> f32 {
        let board = self.nodes[node_id].state.board;
        self.config.target.progress_score(&board)
    }

    fn expand_node(&mut self, node_id: usize) -> bool {
        let state = self.nodes[node_id].state;
        let piece = self.pieces[state.step];
        let mut candidates = Vec::new();

        for &placement in TetrisPiecePlacement::all_from_piece(piece) {
            let mut next = state.board;
            let result = next.apply_piece_placement(placement);
            if !self
                .config
                .constraints
                .allows_after_placement(&next, result.is_lost)
            {
                continue;
            }

            let mut cost =
                placement_cost(&next, result.lines_cleared, state.step, self.pieces.len());
            if self.config.prior_noise > 0.0 {
                cost +=
                    self.config.prior_noise * prior_noise(state, placement, self.config.noise_seed);
            }
            candidates.push(Candidate {
                placement,
                board: next,
                cost,
            });
        }

        candidates.sort_unstable_by(|left, right| {
            left.cost
                .partial_cmp(&right.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(self.config.max_children);

        let priors = normalized_priors(&candidates, self.config.prior_temperature);
        let mut children = Vec::with_capacity(candidates.len());

        for (candidate, prior) in candidates.into_iter().zip(priors) {
            let child_state = SearchState {
                board: candidate.board,
                step: state.step + 1,
            };
            let child_key = state_key(child_state);
            let child = if let Some(&child_id) = self.index.get(&child_key) {
                child_id
            } else {
                let child_id = self.nodes.len();
                self.nodes.push(SearchNode::new(child_state));
                self.index.insert(child_key, child_id);
                child_id
            };
            children.push(ActionEdge {
                placement: candidate.placement,
                child,
                visits: 0,
                value_sum: 0.0,
                prior,
            });
        }

        self.nodes[node_id].children = children;
        self.nodes[node_id].expanded = true;
        !self.nodes[node_id].children.is_empty()
    }

    fn select_child(&self, node_id: usize) -> usize {
        let node = &self.nodes[node_id];
        let parent_visits = node.visits.max(1) as f32;
        let exploration_scale = self.config.exploration * parent_visits.sqrt();

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (idx, edge) in node.children.iter().enumerate() {
            let q = if edge.visits == 0 {
                0.0
            } else {
                edge.value_sum / edge.visits as f32
            };
            let u = exploration_scale * edge.prior / (1.0 + edge.visits as f32);
            let score = q + u;
            if score > best_score {
                best_idx = idx;
                best_score = score;
            }
        }

        best_idx
    }
}

/// Run PUCT over one fixed piece sequence.
pub fn run_puct_search(pieces: &[TetrisPiece], config: &PuctConfig) -> PuctSearchResult {
    run_puct_search_with_progress(pieces, config, 0, |_| {})
}

/// Run PUCT and invoke `on_progress` every `progress_every` iterations.
pub fn run_puct_search_with_progress(
    pieces: &[TetrisPiece],
    config: &PuctConfig,
    progress_every: u64,
    on_progress: impl FnMut(PuctProgress),
) -> PuctSearchResult {
    PuctSearch::new(pieces, config).search(progress_every, on_progress)
}

fn state_key(state: SearchState) -> StateKey {
    StateKey {
        board: board_key(&state.board),
        step: state.step,
    }
}

fn prior_noise(state: SearchState, placement: TetrisPiecePlacement, seed: u64) -> f32 {
    let mut hasher = FxHasher::default();
    seed.hash(&mut hasher);
    board_key(&state.board).hash(&mut hasher);
    state.step.hash(&mut hasher);
    placement.index().hash(&mut hasher);
    let bits = hasher.finish();
    let unit = (bits as f64 / u64::MAX as f64) as f32;
    unit * 2.0 - 1.0
}

fn normalized_priors(candidates: &[Candidate], temperature: f32) -> Vec<f32> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let max_logit = candidates
        .iter()
        .map(|candidate| -candidate.cost / temperature)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut weights = Vec::with_capacity(candidates.len());
    let mut total = 0.0;
    for candidate in candidates {
        let weight = (-candidate.cost / temperature - max_logit).exp();
        let finite_weight = if weight.is_finite() { weight } else { 0.0 };
        weights.push(finite_weight);
        total += finite_weight;
    }

    if total <= f32::EPSILON {
        let uniform = 1.0 / candidates.len() as f32;
        return vec![uniform; candidates.len()];
    }

    weights.into_iter().map(|weight| weight / total).collect()
}

fn placement_cost(
    board: &TetrisBoard,
    lines_just_cleared: u32,
    step: usize,
    total_steps: usize,
) -> f32 {
    let cells = board.count();
    let height = board.height() as usize;
    let holes = count_holes(board);
    let roughness = roughness(board);
    let near_clear_cells = near_clear_cells(board);

    let progress = (step + 1) as f32 / total_steps.max(1) as f32;
    let target_lines = total_steps as f32 * 0.4;
    let expected_lines = target_lines * progress;
    let observed_lines = ((step + 1) as u32 * 4).saturating_sub(cells) as f32 / 10.0;
    let line_deficit = (expected_lines - observed_lines).max(0.0);
    let clear_bonus = -(lines_just_cleared as f32) * (2.0 + line_deficit);

    cells as f32 * 0.45 + height as f32 * 0.8 + holes as f32 * 4.0 + roughness as f32 * 0.8
        - near_clear_cells as f32 * 0.25
        + line_deficit * 3.0
        + clear_bonus
}

fn count_holes(board: &TetrisBoard) -> u32 {
    board
        .as_limbs()
        .iter()
        .map(|col| {
            let height = u32::BITS - col.leading_zeros();
            height - col.count_ones()
        })
        .sum()
}

fn roughness(board: &TetrisBoard) -> u32 {
    let limbs = board.as_limbs();
    let mut heights = [0u32; 10];
    for (idx, col) in limbs.iter().enumerate() {
        heights[idx] = u32::BITS - col.leading_zeros();
    }

    heights
        .windows(2)
        .map(|pair| pair[0].abs_diff(pair[1]))
        .sum()
}

fn near_clear_cells(board: &TetrisBoard) -> u32 {
    let mut cells = 0;
    for row in 0..board.height() as usize {
        let mut row_cells = 0;
        for col in 0..10 {
            if board.get_bit(col, row) {
                row_cells += 1;
            }
        }
        if row_cells >= 8 {
            cells += row_cells;
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(target: TargetSpec) -> PuctConfig {
        PuctConfig {
            start_board: TetrisBoard::new(),
            constraints: BoardConstraints::new(20, u32::MAX),
            target,
            iterations: 64,
            exploration: 1.4,
            prior_temperature: 2.0,
            prior_noise: 0.0,
            max_children: 64,
            terminal_mode: MctsTerminalMode::Shaped,
            terminal_repeat_penalty: 0.0,
            noise_seed: 7,
        }
    }

    #[test]
    fn empty_target_accepts_only_empty_board() {
        let target = TargetSpec::Empty;
        assert!(target.is_success(&TetrisBoard::new()));

        let mut board = TetrisBoard::new();
        let placement = TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE)[0];
        let result = board.apply_piece_placement(placement);
        assert_eq!(result.is_lost, IsLost::NOT_LOST);
        assert!(!target.is_success(&board));
    }

    #[test]
    fn cell_count_target_dedupes_and_matches_counts() -> Result<()> {
        let target = TargetSpec::from_cli(MctsTargetKind::CellCount, vec![4, 4], None, None)?;
        let mut board = TetrisBoard::new();
        let placement = TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE)[0];
        let result = board.apply_piece_placement(placement);
        assert_eq!(result.is_lost, IsLost::NOT_LOST);
        assert!(target.is_success(&board));
        Ok(())
    }

    #[test]
    fn low_board_target_uses_default_bounds() -> Result<()> {
        let target = TargetSpec::from_cli(MctsTargetKind::LowBoard, Vec::new(), None, None)?;
        assert!(target.is_success(&TetrisBoard::new()));
        Ok(())
    }

    #[test]
    fn puct_finds_zero_piece_empty_target() {
        let config = test_config(TargetSpec::Empty);
        let result = run_puct_search(&[], &config);
        assert!(result.success);
        assert_eq!(result.best_cells, 0);
    }

    #[test]
    fn puct_can_target_single_piece_cell_count() -> Result<()> {
        let target = TargetSpec::from_cli(MctsTargetKind::CellCount, vec![4], None, None)?;
        let config = test_config(target);
        config.validate(1)?;

        let result = run_puct_search(&[TetrisPiece::O_PIECE], &config);
        assert!(result.success);
        assert_eq!(result.best_cells, 4);
        assert!(result.tree_nodes > 1);
        Ok(())
    }

    #[test]
    fn puct_progress_callback_fires() -> Result<()> {
        let target = TargetSpec::from_cli(MctsTargetKind::CellCount, vec![5], None, None)?;
        let mut config = test_config(target);
        config.iterations = 4;
        config.validate(1)?;

        let mut progress_events = 0;
        let result =
            run_puct_search_with_progress(&[TetrisPiece::O_PIECE], &config, 1, |progress| {
                progress_events += 1;
                assert!(progress.iterations > 0);
                assert!(progress.tree_nodes > 0);
            });

        assert!(!result.success);
        assert!(progress_events > 0);
        Ok(())
    }

    #[test]
    fn empty_or_bust_penalizes_terminal_misses() {
        let mut board = TetrisBoard::new();
        let placement = TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE)[0];
        let result = board.apply_piece_placement(placement);
        assert_eq!(result.is_lost, IsLost::NOT_LOST);

        let shaped = TargetSpec::Empty.terminal_score(&board, MctsTerminalMode::Shaped, 0, 0.0);
        let empty_or_bust =
            TargetSpec::Empty.terminal_score(&board, MctsTerminalMode::EmptyOrBust, 0, 0.0);

        assert!(shaped > 0.0);
        assert!(empty_or_bust < 0.0);
    }
}
