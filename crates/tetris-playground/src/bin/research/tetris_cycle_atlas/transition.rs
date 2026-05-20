use rustc_hash::FxHashMap;
use serde::Serialize;
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};

use crate::config::BoardAdmissibility;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoardTransition {
    pub placement: TetrisPiecePlacement,
    pub board: TetrisBoard,
    pub lines_cleared: u32,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct TransitionStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub successors_generated: u64,
    pub lost_prunes: u64,
    pub admissibility_prunes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TransitionKey {
    board: TetrisBoard,
    piece: TetrisPiece,
}

pub struct TransitionCache {
    admissibility: BoardAdmissibility,
    cache: FxHashMap<TransitionKey, Vec<BoardTransition>>,
    stats: TransitionStats,
}

impl TransitionCache {
    pub fn new(admissibility: BoardAdmissibility) -> Self {
        Self {
            admissibility,
            cache: FxHashMap::default(),
            stats: TransitionStats::default(),
        }
    }

    pub fn successors(&mut self, board: TetrisBoard, piece: TetrisPiece) -> &[BoardTransition] {
        let key = TransitionKey { board, piece };
        if self.cache.contains_key(&key) {
            self.stats.cache_hits += 1;
            return self.cache.get(&key).map_or(&[], Vec::as_slice);
        }

        self.stats.cache_misses += 1;
        let mut successors = Vec::new();
        for &placement in TetrisPiecePlacement::all_from_piece(piece) {
            let mut next_board = board;
            let result = next_board.apply_piece_placement(placement);
            if result.is_lost == IsLost::LOST {
                self.stats.lost_prunes += 1;
                continue;
            }
            if !self.admissibility.contains(&next_board) {
                self.stats.admissibility_prunes += 1;
                continue;
            }
            self.stats.successors_generated += 1;
            successors.push(BoardTransition {
                placement,
                board: next_board,
                lines_cleared: result.lines_cleared,
            });
        }

        self.cache.insert(key, successors);
        self.cache.get(&key).map_or(&[], Vec::as_slice)
    }

    #[inline]
    pub const fn stats(&self) -> TransitionStats {
        self.stats
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_cache_reuses_board_piece_successors() {
        let mut cache = TransitionCache::new(BoardAdmissibility::default());
        let board = TetrisBoard::EMPTY_BOARD;

        let first_count = cache.successors(board, TetrisPiece::O_PIECE).len();
        let after_first = cache.stats();
        let second_count = cache.successors(board, TetrisPiece::O_PIECE).len();
        let after_second = cache.stats();

        assert_eq!(first_count, second_count);
        assert_eq!(after_first.cache_misses, 1);
        assert_eq!(after_second.cache_hits, 1);
        assert_eq!(cache.len(), 1);
    }
}
