use tetris_game::{TetrisBoard, TetrisPieceBagState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoardAdmissibility {
    pub max_height: u8,
    pub max_holes: u32,
    pub max_cells: u32,
    pub max_roughness: u32,
    pub max_height_spread: u32,
}

impl Default for BoardAdmissibility {
    fn default() -> Self {
        Self {
            max_height: 6,
            max_holes: u32::MAX,
            max_cells: u32::MAX,
            max_roughness: u32::MAX,
            max_height_spread: u32::MAX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootStateConfig {
    pub board: TetrisBoard,
    pub bag: TetrisPieceBagState,
}

impl Default for RootStateConfig {
    fn default() -> Self {
        Self {
            board: TetrisBoard::new(),
            bag: TetrisPieceBagState::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SolverConfig {
    pub admissibility: BoardAdmissibility,
    pub root: RootStateConfig,
}
