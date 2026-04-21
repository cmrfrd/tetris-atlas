use tetris_game::TetrisPieceBagState;

/// Admissibility constraints that bound the reachable state space.
///
/// Each field maps 1:1 to a `TetrisBoard` method:
/// - `max_height` -> `board.height()`
/// - `max_holes` -> `board.total_holes()`
/// - `max_roughness` -> `board.roughness()`
/// - `max_count` -> `board.count()`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoardAdmissibility {
    pub max_height: u32,
    pub max_holes: u32,
    pub max_roughness: u32,
    pub max_count: u32,
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

/// Configuration for the full 7-bag adversarial solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverConfig {
    pub admissibility: BoardAdmissibility,
    /// Maximum states to expand before halting (safety cap).
    pub max_states: usize,
    /// Root bag state (defaults to full bag).
    pub root_bag: TetrisPieceBagState,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            admissibility: BoardAdmissibility::default(),
            max_states: usize::MAX,
            root_bag: TetrisPieceBagState::new(),
        }
    }
}
