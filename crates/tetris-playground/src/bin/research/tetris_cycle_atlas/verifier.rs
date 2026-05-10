use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet};
use tetris_game::{TetrisBoard, TetrisPiece, TetrisPieceBagState, TetrisPiecePlacement};

use crate::config::SolverConfig;
use crate::transition::TransitionCache;
use crate::universe::{BoardId, BoardUniverse};

/// Memo key for verification AND-OR search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MemoKey {
    board_id: BoardId,
    bag: TetrisPieceBagState,
    cycles_remaining: u32,
}

/// Verification result.
pub struct VerificationResult {
    pub boards_checked: usize,
    pub boards_passed: usize,
    pub boards_failed: usize,
}

impl VerificationResult {
    #[inline]
    pub const fn is_clean(&self) -> bool {
        self.boards_failed == 0
    }
}

/// Board registry for verification (same as in gfp.rs).
struct BoardRegistry {
    boards: Vec<TetrisBoard>,
    board_to_id: FxHashMap<TetrisBoard, BoardId>,
    universe_count: u32,
    max_boards: usize,
    cap_hits: u64,
}

impl BoardRegistry {
    fn from_universe(universe: &BoardUniverse, max_boards: usize) -> Self {
        Self {
            boards: universe.boards.clone(),
            board_to_id: universe.board_to_id.clone(),
            universe_count: universe.board_count() as u32,
            max_boards,
            cap_hits: 0,
        }
    }

    fn get_or_insert(&mut self, board: TetrisBoard) -> Option<BoardId> {
        if let Some(&id) = self.board_to_id.get(&board) {
            return Some(id);
        }
        if self.boards.len() >= self.max_boards {
            self.cap_hits += 1;
            return None;
        }
        let id = self.boards.len() as BoardId;
        self.boards.push(board);
        self.board_to_id.insert(board, id);
        Some(id)
    }

    #[inline]
    fn is_universe_board(&self, id: BoardId) -> bool {
        id < self.universe_count
    }
}

/// Independently verify every board in the safe set by re-running the inner
/// AND-OR check with a fresh memo.
pub fn verify_safe_set(
    universe: &BoardUniverse,
    safe_set: &FxHashSet<BoardId>,
    config: &SolverConfig,
) -> VerificationResult {
    let start = Instant::now();
    let mut transitions = TransitionCache::new(config.admissibility);
    let mut registry = BoardRegistry::from_universe(universe, config.max_registry_boards);
    let mut boards_passed: usize = 0;
    let mut boards_failed: usize = 0;

    for &board_id in safe_set {
        // Fresh memo for each board
        let mut memo: FxHashMap<MemoKey, bool> = FxHashMap::default();
        let survives = verify_inner_check(
            board_id,
            TetrisPieceBagState::new(),
            config.bag_cycles,
            &mut registry,
            safe_set,
            &mut transitions,
            &mut memo,
        );
        if survives {
            boards_passed += 1;
        } else {
            boards_failed += 1;
            eprintln!(
                "[verify] FAILED: board_id={} board={:?}",
                board_id, universe.boards[board_id as usize]
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "[verify] done in {:.2}s | checked={} passed={} failed={} registry={}/{} cap_hits={}",
        elapsed,
        safe_set.len(),
        boards_passed,
        boards_failed,
        registry.boards.len(),
        registry.max_boards,
        registry.cap_hits,
    );

    VerificationResult {
        boards_checked: safe_set.len(),
        boards_passed,
        boards_failed,
    }
}

/// Same AND-OR search as in gfp.rs, but independent implementation for verification.
fn verify_inner_check(
    board_id: BoardId,
    bag: TetrisPieceBagState,
    cycles_remaining: u32,
    registry: &mut BoardRegistry,
    safe_set: &FxHashSet<BoardId>,
    transitions: &mut TransitionCache,
    memo: &mut FxHashMap<MemoKey, bool>,
) -> bool {
    // Terminal: all cycles exhausted
    if cycles_remaining == 0 {
        return registry.is_universe_board(board_id) && safe_set.contains(&board_id);
    }

    let key = MemoKey {
        board_id,
        bag,
        cycles_remaining,
    };
    if let Some(&result) = memo.get(&key) {
        return result;
    }

    let board = registry.boards[board_id as usize];

    // AND: all adversary piece choices must be survivable
    let pieces: Vec<(TetrisPiece, TetrisPieceBagState)> = bag.iter_next_states().collect();

    let mut result = true;
    for &(piece, next_bag) in &pieces {
        let next_cycles = if next_bag == TetrisPieceBagState::new() {
            cycles_remaining - 1
        } else {
            cycles_remaining
        };

        // OR: player picks placement
        let transitions_vec = transitions.successors(board, piece).to_vec();
        let mut found = false;
        for transition in &transitions_vec {
            let Some(next_board_id) = registry.get_or_insert(transition.board) else {
                continue;
            };
            if verify_inner_check(
                next_board_id,
                next_bag,
                next_cycles,
                registry,
                safe_set,
                transitions,
                memo,
            ) {
                found = true;
                break;
            }
        }
        if !found {
            result = false;
            break;
        }
    }

    memo.insert(key, result);
    result
}
