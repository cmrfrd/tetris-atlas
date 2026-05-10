use clap::ValueEnum;
use serde::Serialize;
use tetris_game::TetrisBoard;

/// Admissibility constraints bounding the reachable board space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct BoardAdmissibility {
    pub max_height: u32,
    pub max_holes: u32,
    pub max_roughness: u32,
    pub max_count: u32,
}

impl BoardAdmissibility {
    #[inline]
    pub fn contains(self, board: &TetrisBoard) -> bool {
        if self.max_height != u32::MAX && board.height() > self.max_height {
            return false;
        }
        if self.max_holes != u32::MAX && board.total_holes() > self.max_holes {
            return false;
        }
        if self.max_roughness != u32::MAX && board.roughness() > self.max_roughness {
            return false;
        }
        if self.max_count != u32::MAX && board.count() > self.max_count {
            return false;
        }
        true
    }
}

impl Default for BoardAdmissibility {
    fn default() -> Self {
        Self {
            max_height: u32::MAX,
            max_holes: u32::MAX,
            max_roughness: u32::MAX,
            max_count: u32::MAX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
pub enum SearchAlgorithm {
    /// Memoized recursive AND-OR search.
    Dfs,
    /// Forward finite universe plus bottom-up dynamic programming.
    Retrograde,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
pub enum TargetMode {
    /// The terminal board must be exactly empty.
    Empty,
    /// The terminal board may be any board satisfying `closed_set_admissibility`.
    LowClean,
}

/// Configuration for the bag-cycle reset solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SolverConfig {
    /// Number of complete bag cycles. Must be a multiple of 5 for strict empty reset.
    pub bag_cycles: u32,
    /// Board admissibility constraints for pruning.
    pub admissibility: BoardAdmissibility,
    /// Terminal target predicate.
    pub target_mode: TargetMode,
    /// Terminal board constraints when `target_mode == TargetMode::LowClean`.
    pub closed_set_admissibility: BoardAdmissibility,
}

impl SolverConfig {
    #[inline]
    pub const fn total_pieces(self) -> u32 {
        self.bag_cycles * 7
    }

    #[inline]
    pub fn target_contains(self, board: &TetrisBoard) -> bool {
        match self.target_mode {
            TargetMode::Empty => *board == TetrisBoard::new(),
            TargetMode::LowClean => self.closed_set_admissibility.contains(board),
        }
    }

    #[inline]
    pub const fn uses_empty_reset(self) -> bool {
        matches!(self.target_mode, TargetMode::Empty)
    }
}
