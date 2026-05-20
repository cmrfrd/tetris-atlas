use std::time::Instant;

use rustc_hash::FxHashMap;
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};

use crate::config::{BoardAdmissibility, SolverConfig};

pub type BoardId = u32;

/// Result of board-only BFS discovery.
pub struct BoardUniverse {
    pub boards: Vec<TetrisBoard>,
    pub board_to_id: FxHashMap<TetrisBoard, BoardId>,
    pub empty_board_id: BoardId,
}

impl BoardUniverse {
    /// BFS from empty board. For each board, try ALL 7 pieces x ALL placements.
    /// Intern resulting boards if they pass safe_set_admissibility.
    pub fn discover(config: &SolverConfig) -> Self {
        let start = Instant::now();

        let mut boards: Vec<TetrisBoard> = Vec::new();
        let mut board_to_id: FxHashMap<TetrisBoard, BoardId> = FxHashMap::default();

        // Intern the empty board
        let empty_board = TetrisBoard::EMPTY_BOARD;
        let empty_board_id = intern_board(&mut boards, &mut board_to_id, &empty_board);

        // BFS frontier
        let mut frontier: Vec<BoardId> = vec![empty_board_id];
        let mut frontier_idx: usize = 0;
        let mut hit_cap = false;

        let mut last_report = Instant::now();

        'bfs: while frontier_idx < frontier.len() {
            let bid = frontier[frontier_idx];
            frontier_idx += 1;

            if boards.len() >= config.max_boards {
                hit_cap = true;
                break;
            }

            let board = boards[bid as usize];

            // Try all 7 pieces x all placements
            for piece in TetrisPiece::all() {
                for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                    let mut next_board = board;
                    let result = next_board.apply_piece_placement(placement);
                    if result.is_lost == IsLost::LOST {
                        continue;
                    }
                    if !config.safe_set_admissibility.contains(&next_board) {
                        continue;
                    }

                    if board_to_id.contains_key(&next_board) {
                        continue;
                    }

                    if boards.len() >= config.max_boards {
                        hit_cap = true;
                        break 'bfs;
                    }

                    let new_id = intern_board(&mut boards, &mut board_to_id, &next_board);
                    frontier.push(new_id);
                }
            }

            // Progress reporting
            if last_report.elapsed().as_secs() >= 2 {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = boards.len() as f64 / elapsed;
                eprintln!(
                    "[discover] {:.1}s | boards={} frontier={} ({:.0} boards/s)",
                    elapsed,
                    boards.len(),
                    frontier.len() - frontier_idx,
                    rate,
                );
                last_report = Instant::now();
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        if hit_cap {
            eprintln!(
                "[discover] hit max_boards cap ({}) -- stopped expansion with frontier={}",
                config.max_boards,
                frontier.len().saturating_sub(frontier_idx),
            );
        }
        eprintln!(
            "[discover] done in {:.2}s | boards={}",
            elapsed,
            boards.len(),
        );

        BoardUniverse {
            boards,
            board_to_id,
            empty_board_id,
        }
    }

    #[inline]
    pub fn board_count(&self) -> usize {
        self.boards.len()
    }
}

fn intern_board(
    boards: &mut Vec<TetrisBoard>,
    board_to_id: &mut FxHashMap<TetrisBoard, BoardId>,
    board: &TetrisBoard,
) -> BoardId {
    if let Some(&id) = board_to_id.get(board) {
        return id;
    }
    let id = boards.len() as BoardId;
    boards.push(*board);
    board_to_id.insert(*board, id);
    id
}

/// Check if a board passes admissibility (used externally for intermediate boards).
#[inline]
pub fn board_is_admissible(board: &TetrisBoard, adm: &BoardAdmissibility) -> bool {
    adm.contains(board)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_includes_empty_board() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 4,
                max_holes: 0,
                max_roughness: 4,
                max_count: u32::MAX,
            },
            safe_set_admissibility: BoardAdmissibility {
                max_height: 4,
                max_holes: 0,
                max_roughness: 4,
                max_count: u32::MAX,
            },
            bag_cycles: 1,
            max_boards: 100_000,
            max_registry_boards: 100_000,
        };
        let universe = BoardUniverse::discover(&config);
        assert!(universe.board_count() >= 1);
        assert_eq!(universe.empty_board_id, 0);
        assert_eq!(universe.boards[0], TetrisBoard::EMPTY_BOARD);
    }

    #[test]
    fn discover_respects_max_boards_cap() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 4,
                max_holes: 0,
                max_roughness: 4,
                max_count: u32::MAX,
            },
            safe_set_admissibility: BoardAdmissibility {
                max_height: 4,
                max_holes: 0,
                max_roughness: 4,
                max_count: u32::MAX,
            },
            bag_cycles: 1,
            max_boards: 1,
            max_registry_boards: 1,
        };

        let universe = BoardUniverse::discover(&config);

        assert_eq!(universe.board_count(), 1);
    }
}
