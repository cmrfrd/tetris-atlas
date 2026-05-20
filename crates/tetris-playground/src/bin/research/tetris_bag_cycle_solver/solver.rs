use std::time::Instant;

use rustc_hash::FxHashMap;
use serde::Serialize;
use tetris_game::{TetrisBoard, TetrisPiece, TetrisPieceBagState, TetrisPiecePlacement};

use crate::config::SolverConfig;
use crate::state::{BoardCache, BoardId, MemoKey, WitnessKey};
use crate::transition::{BoardTransition, TransitionCache, TransitionStats};

/// Statistics collected during the search.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct SolveStats {
    pub nodes_explored: u64,
    pub memo_hits: u64,
    pub memo_true: u64,
    pub memo_false: u64,
    pub cell_prunes: u64,
    pub target_hits: u64,
    pub piece_branches_attempted: u64,
    pub piece_branches_proven: u64,
    pub piece_branches_failed: u64,
    pub placement_branches_scanned: u64,
    pub witness_placements_selected: u64,
    pub deepest_ply_reached: u32,
    pub total_plies: u32,
    pub transition_cache_entries: usize,
    pub transition: TransitionStats,
    pub retrograde_states: usize,
    pub retrograde_edges: usize,
    pub retrograde_winning_states: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct WitnessEntry {
    pub placement_index: u8,
    pub child_board_id: Option<BoardId>,
    pub terminal: bool,
}

pub type WitnessPolicy = FxHashMap<WitnessKey, WitnessEntry>;

#[derive(Debug, Clone, Serialize)]
pub struct RootPieceDiagnostic {
    pub piece_index: u8,
    pub piece_name: String,
    pub proven: bool,
    pub placement_index: Option<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RootDiagnostics {
    pub proven_pieces: usize,
    pub total_pieces: usize,
    pub piece_coverage: f64,
    pub pieces: Vec<RootPieceDiagnostic>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RootPieceStatus {
    NotStarted,
    Running,
    Proven,
    Failed,
}

impl RootPieceStatus {
    fn short(self) -> &'static str {
        match self {
            RootPieceStatus::NotStarted => ".",
            RootPieceStatus::Running => "*",
            RootPieceStatus::Proven => "+",
            RootPieceStatus::Failed => "x",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct RootProgress {
    pub statuses: [RootPieceStatus; 7],
}

impl Default for RootProgress {
    fn default() -> Self {
        Self {
            statuses: [RootPieceStatus::NotStarted; 7],
        }
    }
}

impl RootProgress {
    pub fn proven_count(self) -> usize {
        self.statuses
            .iter()
            .filter(|&&status| status == RootPieceStatus::Proven)
            .count()
    }

    pub fn failed_count(self) -> usize {
        self.statuses
            .iter()
            .filter(|&&status| status == RootPieceStatus::Failed)
            .count()
    }

    pub fn running_piece_name(self) -> Option<&'static str> {
        self.statuses
            .iter()
            .position(|&status| status == RootPieceStatus::Running)
            .map(piece_name_from_index)
    }

    pub fn compact(self) -> String {
        TetrisPiece::all()
            .into_iter()
            .map(|piece| {
                let idx = piece.index() as usize;
                format!(
                    "{}:{}",
                    piece_name_from_index(idx),
                    self.statuses[idx].short()
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Result of the solver run.
pub struct SolveResult {
    pub winning: bool,
    pub stats: SolveStats,
    pub boards: BoardCache,
    pub memo_entries: usize,
    pub policy: WitnessPolicy,
    pub root_diagnostics: Option<RootDiagnostics>,
    pub root_progress: RootProgress,
}

impl SolveResult {
    #[inline]
    pub fn boards_interned(&self) -> usize {
        self.boards.len()
    }
}

/// Core AND-OR memoized search solver.
pub struct Solver {
    config: SolverConfig,
    boards: BoardCache,
    memo: FxHashMap<MemoKey, bool>,
    policy: WitnessPolicy,
    transitions: TransitionCache,
    empty_board_id: BoardId,
    stats: SolveStats,
    root_progress: RootProgress,
    last_report: Instant,
    start_time: Instant,
    diagnostics_enabled: bool,
}

impl Solver {
    pub fn new(config: SolverConfig, diagnostics_enabled: bool) -> Self {
        let mut boards = BoardCache::new();
        let empty_board_id = boards.get_or_insert(TetrisBoard::EMPTY_BOARD);
        let now = Instant::now();
        Self {
            config,
            boards,
            memo: FxHashMap::default(),
            policy: FxHashMap::default(),
            transitions: TransitionCache::new(config.admissibility),
            empty_board_id,
            stats: SolveStats {
                total_plies: config.total_pieces(),
                ..SolveStats::default()
            },
            root_progress: RootProgress::default(),
            last_report: now,
            start_time: now,
            diagnostics_enabled,
        }
    }

    pub fn solve(mut self) -> SolveResult {
        self.start_time = Instant::now();
        self.last_report = self.start_time;

        let bag = TetrisPieceBagState::new();
        let cycles = self.config.bag_cycles;
        let winning = self.search(self.empty_board_id, bag, cycles);
        let root_diagnostics = self
            .diagnostics_enabled
            .then(|| self.diagnose_root_piece_coverage());
        self.stats.transition_cache_entries = self.transitions.len();
        self.stats.transition = self.transitions.stats();
        let memo_entries = self.memo.len();

        SolveResult {
            winning,
            stats: self.stats,
            boards: self.boards,
            memo_entries,
            policy: self.policy,
            root_diagnostics,
            root_progress: self.root_progress,
        }
    }

    /// Recursive AND-OR search.
    ///
    /// Returns `true` if the player can guarantee reaching the empty board
    /// within the remaining bag cycles, regardless of adversarial piece ordering.
    fn search(
        &mut self,
        board_id: BoardId,
        bag: TetrisPieceBagState,
        cycles_remaining: u32,
    ) -> bool {
        // Progress reporting
        self.stats.nodes_explored += 1;
        self.update_depth_progress(bag, cycles_remaining);
        if self.stats.nodes_explored % 100_000 == 0 {
            self.maybe_report();
        }

        // 1. Memo lookup
        let key = MemoKey::new(board_id, bag, cycles_remaining);
        if let Some(&result) = self.memo.get(&key) {
            self.stats.memo_hits += 1;
            return result;
        }

        let board = self.boards.boards[board_id as usize];

        // 2. Cell-count feasibility prune.
        //    Total cells after placing all remaining pieces = current + 4*pieces_remaining.
        //    To reach empty board, must clear complete rows of 10. So the total must be
        //    divisible by 10.
        if self.config.uses_empty_reset() {
            let pieces_in_bag = bag.count() as u32;
            let full_bags_remaining = if cycles_remaining > 0 {
                (cycles_remaining - 1) * 7
            } else {
                0
            };
            let pieces_remaining = pieces_in_bag + full_bags_remaining;
            let total_cells = board.count() + 4 * pieces_remaining;
            if total_cells % 10 != 0 {
                self.stats.cell_prunes += 1;
                self.stats.memo_false += 1;
                self.memo.insert(key, false);
                return false;
            }
        }

        // 3. AND over adversary piece choices — all pieces in bag must be survivable.
        //    Order pieces by fewest valid placements first (fail-fast).
        let pieces: Vec<(TetrisPiece, TetrisPieceBagState)> = bag.iter_next_states().collect();

        // Sort by number of valid placements (ascending) for fail-fast AND
        let mut piece_order: Vec<usize> = (0..pieces.len()).collect();
        piece_order.sort_unstable_by_key(|&i| {
            let (piece, _) = pieces[i];
            TetrisPiecePlacement::all_from_piece(piece).len()
        });

        let is_root = self.is_root_state(board_id, bag, cycles_remaining);
        for &i in &piece_order {
            let (piece, next_bag) = pieces[i];
            if is_root {
                self.root_progress.statuses[piece.index() as usize] = RootPieceStatus::Running;
            }
            let witness =
                self.prove_piece_branch(board_id, board, bag, cycles_remaining, piece, next_bag);
            if witness.is_none() {
                if is_root {
                    self.root_progress.statuses[piece.index() as usize] = RootPieceStatus::Failed;
                }
                self.stats.memo_false += 1;
                self.memo.insert(key, false);
                return false;
            }
            if is_root {
                self.root_progress.statuses[piece.index() as usize] = RootPieceStatus::Proven;
            }
        }

        self.stats.memo_true += 1;
        self.memo.insert(key, true);
        true
    }

    fn prove_piece_branch(
        &mut self,
        board_id: BoardId,
        board: TetrisBoard,
        bag: TetrisPieceBagState,
        cycles_remaining: u32,
        piece: TetrisPiece,
        next_bag: TetrisPieceBagState,
    ) -> Option<WitnessEntry> {
        self.stats.piece_branches_attempted += 1;
        let witness = if bag.count() == 1 {
            if cycles_remaining <= 1 {
                self.try_placements_for_target(board, piece)
            } else {
                self.try_placements_recursive(board, piece, next_bag, cycles_remaining - 1)
            }
        } else {
            self.try_placements_recursive(board, piece, next_bag, cycles_remaining)
        };

        let Some(witness) = witness else {
            self.stats.piece_branches_failed += 1;
            return None;
        };

        self.policy.insert(
            WitnessKey::new(board_id, bag, cycles_remaining, piece),
            witness,
        );
        self.stats.piece_branches_proven += 1;
        Some(witness)
    }

    /// OR node: try all placements of `piece` on `board`, looking for one that
    /// reaches the configured terminal target.
    fn try_placements_for_target(
        &mut self,
        board: TetrisBoard,
        piece: TetrisPiece,
    ) -> Option<WitnessEntry> {
        let mut transitions = self.transitions.successors(board, piece).to_vec();
        self.stats.placement_branches_scanned += transitions.len() as u64;
        transitions.sort_unstable_by(|a, b| b.lines_cleared.cmp(&a.lines_cleared));

        let target = transitions
            .into_iter()
            .find(|transition| self.config.target_contains(&transition.board));
        if target.is_some() {
            self.stats.target_hits += 1;
            self.stats.deepest_ply_reached = self.stats.total_plies;
            self.stats.witness_placements_selected += 1;
        }

        target.map(|transition| WitnessEntry {
            placement_index: transition.placement.index(),
            child_board_id: None,
            terminal: true,
        })
    }

    /// OR node: try all placements of `piece` on `board`, looking for one where
    /// the recursive search succeeds. Placements that clear more lines are tried first.
    fn try_placements_recursive(
        &mut self,
        board: TetrisBoard,
        piece: TetrisPiece,
        next_bag: TetrisPieceBagState,
        cycles_remaining: u32,
    ) -> Option<WitnessEntry> {
        let mut transitions = self.transitions.successors(board, piece).to_vec();
        self.stats.placement_branches_scanned += transitions.len() as u64;

        // Sort by lines cleared descending (succeed fast).
        transitions.sort_unstable_by(|a, b| b.lines_cleared.cmp(&a.lines_cleared));

        transitions.into_iter().find_map(|transition| {
            self.recursive_witness_from_transition(transition, next_bag, cycles_remaining)
        })
    }

    fn recursive_witness_from_transition(
        &mut self,
        transition: BoardTransition,
        next_bag: TetrisPieceBagState,
        cycles_remaining: u32,
    ) -> Option<WitnessEntry> {
        let new_id = self.boards.get_or_insert(transition.board);
        self.search(new_id, next_bag, cycles_remaining).then(|| {
            self.stats.witness_placements_selected += 1;
            WitnessEntry {
                placement_index: transition.placement.index(),
                child_board_id: Some(new_id),
                terminal: false,
            }
        })
    }

    fn diagnose_root_piece_coverage(&mut self) -> RootDiagnostics {
        let bag = TetrisPieceBagState::new();
        let cycles = self.config.bag_cycles;
        let board_id = self.empty_board_id;
        let board = self.boards.boards[board_id as usize];
        let mut pieces = Vec::new();
        let mut proven_pieces = 0;

        for (piece, next_bag) in bag.iter_next_states() {
            let witness = self.prove_piece_branch(board_id, board, bag, cycles, piece, next_bag);
            if witness.is_some() {
                proven_pieces += 1;
            }
            pieces.push(RootPieceDiagnostic {
                piece_index: piece.index(),
                piece_name: piece.to_string(),
                proven: witness.is_some(),
                placement_index: witness.map(|entry| entry.placement_index),
            });
        }

        let total_pieces = pieces.len();
        let piece_coverage = if total_pieces == 0 {
            1.0
        } else {
            proven_pieces as f64 / total_pieces as f64
        };

        RootDiagnostics {
            proven_pieces,
            total_pieces,
            piece_coverage,
            pieces,
        }
    }

    fn maybe_report(&mut self) {
        if self.last_report.elapsed().as_secs() >= 2 {
            let elapsed = self.start_time.elapsed().as_secs_f64();
            let rate = self.stats.nodes_explored as f64 / elapsed;
            let memo_resolved = self.stats.memo_true + self.stats.memo_false;
            let memo_win_pct = percentage(self.stats.memo_true, memo_resolved);
            let branch_win_pct = percentage(
                self.stats.piece_branches_proven,
                self.stats.piece_branches_proven + self.stats.piece_branches_failed,
            );
            let depth_pct = percentage(
                self.stats.deepest_ply_reached as u64,
                self.stats.total_plies as u64,
            );
            eprintln!(
                "[search] {:.1}s | nodes={} memo_hits={} memo_size={} memo_true={} memo_false={} memo_win={:.1}% boards={} depth={}/{} ({:.1}%) root=[{}] root_done={}/7 root_failed={} running={} target_hits={} branch_win={:.1}% placements_scanned={} witnesses={} cell_prunes={} lost_prunes={} adm_prunes={} transition_cache={} cache_hits={} cache_misses={} ({:.0} nodes/s)",
                elapsed,
                self.stats.nodes_explored,
                self.stats.memo_hits,
                self.memo.len(),
                self.stats.memo_true,
                self.stats.memo_false,
                memo_win_pct,
                self.boards.len(),
                self.stats.deepest_ply_reached,
                self.stats.total_plies,
                depth_pct,
                self.root_progress.compact(),
                self.root_progress.proven_count(),
                self.root_progress.failed_count(),
                self.root_progress.running_piece_name().unwrap_or("-"),
                self.stats.target_hits,
                branch_win_pct,
                self.stats.placement_branches_scanned,
                self.stats.witness_placements_selected,
                self.stats.cell_prunes,
                self.transitions.stats().lost_prunes,
                self.transitions.stats().admissibility_prunes,
                self.transitions.len(),
                self.transitions.stats().cache_hits,
                self.transitions.stats().cache_misses,
                rate,
            );
            self.last_report = Instant::now();
        }
    }

    fn update_depth_progress(&mut self, bag: TetrisPieceBagState, cycles_remaining: u32) {
        let plies_remaining = bag.count() as u32 + cycles_remaining.saturating_sub(1) * 7;
        let ply = self.stats.total_plies.saturating_sub(plies_remaining);
        self.stats.deepest_ply_reached = self.stats.deepest_ply_reached.max(ply);
    }

    #[inline]
    fn is_root_state(
        &self,
        board_id: BoardId,
        bag: TetrisPieceBagState,
        cycles_remaining: u32,
    ) -> bool {
        board_id == self.empty_board_id
            && bag == TetrisPieceBagState::new()
            && cycles_remaining == self.config.bag_cycles
    }
}

fn piece_name_from_index(index: usize) -> &'static str {
    match index {
        0 => "O",
        1 => "I",
        2 => "S",
        3 => "Z",
        4 => "T",
        5 => "L",
        6 => "J",
        _ => "?",
    }
}

fn percentage(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 * 100.0 / denominator as f64
    }
}
