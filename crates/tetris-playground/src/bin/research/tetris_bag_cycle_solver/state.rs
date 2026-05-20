use rustc_hash::FxHashMap;
use tetris_game::{TetrisBoard, TetrisPiece, TetrisPieceBagState};

pub type BoardId = u32;

/// Interning cache for boards — deduplicates identical boards into compact IDs.
pub struct BoardCache {
    pub boards: Vec<TetrisBoard>,
    board_to_id: FxHashMap<TetrisBoard, BoardId>,
}

impl BoardCache {
    pub fn new() -> Self {
        Self {
            boards: Vec::new(),
            board_to_id: FxHashMap::default(),
        }
    }

    /// Get or insert a board, returning its stable ID.
    #[inline]
    pub fn get_or_insert(&mut self, board: TetrisBoard) -> BoardId {
        if let Some(&id) = self.board_to_id.get(&board) {
            return id;
        }
        let id = self.boards.len() as BoardId;
        self.boards.push(board);
        self.board_to_id.insert(board, id);
        id
    }

    #[inline]
    pub fn get_id(&self, board: &TetrisBoard) -> Option<BoardId> {
        self.board_to_id.get(board).copied()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.boards.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoKey {
    board_id: BoardId,
    bag: TetrisPieceBagState,
    cycles_remaining: u32,
}

impl MemoKey {
    #[inline]
    pub fn new(board_id: BoardId, bag: TetrisPieceBagState, cycles_remaining: u32) -> Self {
        Self {
            board_id,
            bag,
            cycles_remaining,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WitnessKey {
    board_id: BoardId,
    bag: TetrisPieceBagState,
    cycles_remaining: u32,
    piece: TetrisPiece,
}

impl WitnessKey {
    #[inline]
    pub fn new(
        board_id: BoardId,
        bag: TetrisPieceBagState,
        cycles_remaining: u32,
        piece: TetrisPiece,
    ) -> Self {
        Self {
            board_id,
            bag,
            cycles_remaining,
            piece,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn board_cache_deduplicates() {
        let mut cache = BoardCache::new();
        let board = TetrisBoard::EMPTY_BOARD;
        let id1 = cache.get_or_insert(board);
        let id2 = cache.get_or_insert(board);
        assert_eq!(id1, id2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn memo_key_distinct() {
        let bag = TetrisPieceBagState::new();
        let k1 = MemoKey::new(0, bag, 5);
        let k2 = MemoKey::new(0, bag, 4);
        let k3 = MemoKey::new(1, bag, 5);
        let k4 = MemoKey::new(0, bag, 261);
        assert_ne!(k1, k2);
        assert_ne!(k1, k3);
        assert_ne!(k1, k4);
    }

    #[test]
    fn witness_key_distinguishes_piece_and_state() {
        let bag = TetrisPieceBagState::new();
        let k1 = WitnessKey::new(0, bag, 5, TetrisPiece::I_PIECE);
        let k2 = WitnessKey::new(0, bag, 5, TetrisPiece::O_PIECE);
        let k3 = WitnessKey::new(1, bag, 5, TetrisPiece::I_PIECE);
        let k4 = WitnessKey::new(0, bag, 261, TetrisPiece::I_PIECE);
        assert_ne!(k1, k2);
        assert_ne!(k1, k3);
        assert_ne!(k1, k4);
    }
}
