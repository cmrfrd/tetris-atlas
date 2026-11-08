//! Board bounds shared by the proof search and its reference oracle.

use crate::config::BoardAdmissibility;
use tetris_game::TetrisBoard;

#[inline]
pub fn board_is_admissible(board: TetrisBoard<10, 20>, admissibility: BoardAdmissibility) -> bool {
    board.height() <= u32::from(admissibility.max_height)
        && board.total_holes() <= admissibility.max_holes
        && board.count() <= admissibility.max_cells
        && board_surface_roughness(board) <= admissibility.max_roughness
        && board_height_spread(board) <= admissibility.max_height_spread
}

pub fn board_surface_roughness(board: TetrisBoard<10, 20>) -> u32 {
    let heights = board.heights();
    heights
        .windows(2)
        .map(|window| window[0].abs_diff(window[1]))
        .sum()
}

pub fn board_height_spread(board: TetrisBoard<10, 20>) -> u32 {
    let heights = board.heights();
    let min_height = heights.into_iter().min().unwrap_or(0);
    let max_height = heights.into_iter().max().unwrap_or(0);
    max_height.saturating_sub(min_height)
}
