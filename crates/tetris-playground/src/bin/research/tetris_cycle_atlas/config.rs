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

/// Configuration for the cycle-boundary adversarial atlas solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SolverConfig {
    /// Admissibility for intermediate boards during AND-OR search within a bag cycle.
    /// Can be looser than safe_set_admissibility to allow temporary tall boards.
    pub admissibility: BoardAdmissibility,
    /// Admissibility for boards in the safe set (GFP). Tighter constraints.
    pub safe_set_admissibility: BoardAdmissibility,
    /// Number of complete bag cycles the player gets to return to a safe board.
    /// 1 = must return within a single 7-piece cycle (original behavior).
    /// Higher values give the player more room to maneuver.
    pub bag_cycles: u32,
    /// Safety cap on board discovery.
    pub max_boards: usize,
    /// Safety cap on the extended per-GFP registry, including intermediate boards.
    pub max_registry_boards: usize,
}
