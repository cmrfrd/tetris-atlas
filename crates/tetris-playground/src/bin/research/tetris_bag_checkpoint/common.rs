use std::hash::{Hash, Hasher};

use anyhow::{Result, bail};
use rustc_hash::{FxHashSet, FxHasher};
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};

const BOARD_KEY_BYTES: usize = 40;

/// Canonical proof identity for a board: ten column-major `u32` limbs encoded
/// little-endian. Unlike `board_hash`, this is exact and stable across runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BoardKey([u8; BOARD_KEY_BYTES]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BoardStepKey {
    board: BoardKey,
    step: u8,
}

impl BoardKey {
    pub fn from_board(board: &TetrisBoard) -> Self {
        let limbs = board.as_limbs();
        let mut bytes = [0u8; BOARD_KEY_BYTES];
        for (idx, limb) in limbs.iter().enumerate() {
            let start = idx * 4;
            bytes[start..start + 4].copy_from_slice(&limb.to_le_bytes());
        }
        Self(bytes)
    }

    pub fn from_blob(blob: &[u8]) -> Result<Self> {
        match blob.len() {
            BOARD_KEY_BYTES => {
                let mut bytes = [0u8; BOARD_KEY_BYTES];
                bytes.copy_from_slice(blob);
                Ok(Self(bytes))
            }
            TetrisBoard::SIZE => {
                let mut bytes = [0u8; TetrisBoard::SIZE];
                bytes.copy_from_slice(blob);
                Ok(Self::from_board(&TetrisBoard::from_binary_slice(bytes)))
            }
            n => bail!(
                "unsupported board blob length {n}; expected {} or {} bytes",
                BOARD_KEY_BYTES,
                TetrisBoard::SIZE
            ),
        }
    }

    pub fn to_board(self) -> TetrisBoard {
        let mut native_bytes = [0u8; BOARD_KEY_BYTES];
        for idx in 0..TetrisBoard::WIDTH {
            let start = idx * 4;
            let limb = u32::from_le_bytes([
                self.0[start],
                self.0[start + 1],
                self.0[start + 2],
                self.0[start + 3],
            ]);
            native_bytes[start..start + 4].copy_from_slice(&limb.to_ne_bytes());
        }
        TetrisBoard::from(native_bytes)
    }

    pub fn as_bytes(&self) -> &[u8; BOARD_KEY_BYTES] {
        &self.0
    }
}

// ---------------------------------------------------------------------------
// Permutation generation (Heap's algorithm)
// ---------------------------------------------------------------------------

pub fn generate_permutations() -> Vec<[TetrisPiece; 7]> {
    let pieces = TetrisPiece::all();
    let mut perms = Vec::with_capacity(5040);
    let mut indices: [usize; 7] = [0, 1, 2, 3, 4, 5, 6];

    let mut c = [0usize; 7];
    perms.push(std::array::from_fn(|i| pieces[indices[i]]));

    let mut i = 0;
    while i < 7 {
        if c[i] < i {
            if i % 2 == 0 {
                indices.swap(0, i);
            } else {
                indices.swap(c[i], i);
            }
            perms.push(std::array::from_fn(|j| pieces[indices[j]]));
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }

    debug_assert_eq!(perms.len(), 5040);
    perms
}

// ---------------------------------------------------------------------------
// Board hashing
// ---------------------------------------------------------------------------

pub fn board_hash(board: &TetrisBoard) -> u64 {
    let mut hasher = FxHasher::default();
    board.hash(&mut hasher);
    hasher.finish()
}

pub fn board_key(board: &TetrisBoard) -> BoardKey {
    BoardKey::from_board(board)
}

// ---------------------------------------------------------------------------
// DFS: collect all board hashes reachable after placing all 7 pieces
// ---------------------------------------------------------------------------

pub fn collect_reachable_hashes(
    board: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    step: usize,
    max_height: u32,
    max_holes: u32,
    out: &mut FxHashSet<u64>,
    nodes: &mut u64,
) {
    if step == 7 {
        out.insert(board_hash(&board));
        return;
    }

    let piece = pieces[step];
    for &placement in TetrisPiecePlacement::all_from_piece(piece) {
        let mut next = board;
        let result = next.apply_piece_placement(placement);
        *nodes += 1;

        if result.is_lost == IsLost::LOST {
            continue;
        }
        if next.height() > max_height {
            continue;
        }
        if max_holes != u32::MAX && has_too_many_holes(&next, max_holes) {
            continue;
        }

        collect_reachable_hashes(next, pieces, step + 1, max_height, max_holes, out, nodes);
    }
}

/// DFS that collects all terminal `(hash, board)` pairs.
pub fn collect_reachable_boards(
    board: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    step: usize,
    max_height: u32,
    max_holes: u32,
    out: &mut Vec<(u64, TetrisBoard)>,
    nodes: &mut u64,
) {
    if step == 7 {
        out.push((board_hash(&board), board));
        return;
    }

    let piece = pieces[step];
    for &placement in TetrisPiecePlacement::all_from_piece(piece) {
        let mut next = board;
        let result = next.apply_piece_placement(placement);
        *nodes += 1;

        if result.is_lost == IsLost::LOST {
            continue;
        }
        if next.height() > max_height {
            continue;
        }
        if max_holes != u32::MAX && has_too_many_holes(&next, max_holes) {
            continue;
        }

        collect_reachable_boards(next, pieces, step + 1, max_height, max_holes, out, nodes);
    }
}

/// DFS that only records boards whose hash is in `target_set`.
pub fn collect_reachable_filtered(
    board: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    step: usize,
    max_height: u32,
    max_holes: u32,
    target_set: &FxHashSet<u64>,
    out: &mut Vec<(u64, TetrisBoard)>,
    nodes: &mut u64,
) {
    if step == 7 {
        let h = board_hash(&board);
        if target_set.contains(&h) {
            out.push((h, board));
        }
        return;
    }

    let piece = pieces[step];
    for &placement in TetrisPiecePlacement::all_from_piece(piece) {
        let mut next = board;
        let result = next.apply_piece_placement(placement);
        *nodes += 1;

        if result.is_lost == IsLost::LOST {
            continue;
        }
        if next.height() > max_height {
            continue;
        }
        if max_holes != u32::MAX && has_too_many_holes(&next, max_holes) {
            continue;
        }

        collect_reachable_filtered(
            next,
            pieces,
            step + 1,
            max_height,
            max_holes,
            target_set,
            out,
            nodes,
        );
    }
}

/// Short-circuit holes check: returns true as soon as accumulated holes exceeds max.
/// Cheaper than `board.total_holes() > max` when early columns already exceed the budget.
#[inline]
pub fn has_too_many_holes(board: &TetrisBoard, max_holes: u32) -> bool {
    let limbs = board.as_limbs();
    let mut sum = 0u32;
    for col in &limbs {
        let h = u32::BITS - col.leading_zeros();
        let p = col.count_ones();
        sum += h - p;
        if sum > max_holes {
            return true;
        }
    }
    false
}

/// DFS that collects terminal boards matching a specific cell count.
/// Includes cell-count feasibility pruning: if `current_cells + 4*(7-step) < target_cells`,
/// the subtree is pruned since no amount of line-clear-free placement can reach the target.
pub fn collect_reachable_targeted(
    board: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    step: usize,
    max_height: u32,
    max_holes: u32,
    target_cells: u32,
    out: &mut Vec<(u64, TetrisBoard)>,
    nodes: &mut u64,
) {
    if step == 7 {
        if board.count() == target_cells {
            out.push((board_hash(&board), board));
        }
        return;
    }

    // Cell count feasibility: maximum possible final cells is current + 4*remaining (no clears).
    // If that's below target, this subtree can never produce a board with target_cells.
    let remaining = (7 - step) as u32;
    let current_cells = board.count();
    if current_cells + 4 * remaining < target_cells {
        return;
    }

    let piece = pieces[step];
    for &placement in TetrisPiecePlacement::all_from_piece(piece) {
        let mut next = board;
        let result = next.apply_piece_placement(placement);
        *nodes += 1;

        if result.is_lost == IsLost::LOST {
            continue;
        }
        if next.height() > max_height {
            continue;
        }
        if max_holes != u32::MAX && has_too_many_holes(&next, max_holes) {
            continue;
        }

        collect_reachable_targeted(
            next,
            pieces,
            step + 1,
            max_height,
            max_holes,
            target_cells,
            out,
            nodes,
        );
    }
}

/// DFS that short-circuits: returns true as soon as ANY board in `targets` is reached.
pub fn can_reach_any(
    board: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    step: usize,
    max_height: u32,
    max_holes: u32,
    targets: &FxHashSet<u64>,
    nodes: &mut u64,
) -> bool {
    let mut memo: FxHashSet<u64> = FxHashSet::default();
    can_reach_any_memo(
        board, pieces, step, max_height, max_holes, targets, nodes, &mut memo,
    )
}

/// Short-circuit DFS that returns a concrete 7-placement witness as soon as any
/// exact target board is reached.
pub fn find_bag_witness(
    board: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    max_height: u32,
    max_holes: u32,
    targets: &FxHashSet<BoardKey>,
    nodes: &mut u64,
) -> Option<(BoardKey, [u8; 7])> {
    let target_cells = target_cell_counts(targets.iter().copied());
    find_bag_witness_with_target_cells(
        board,
        pieces,
        max_height,
        max_holes,
        targets,
        &target_cells,
        nodes,
    )
}

pub fn target_cell_counts(keys: impl Iterator<Item = BoardKey>) -> Vec<u32> {
    let mut cells = FxHashSet::default();
    for key in keys {
        cells.insert(key.to_board().count());
    }
    let mut cells: Vec<u32> = cells.into_iter().collect();
    cells.sort_unstable();
    cells
}

/// Short-circuit DFS that returns a concrete 7-placement witness as soon as any
/// exact target board is reached, with target cell-count residue pruning.
pub fn find_bag_witness_with_target_cells(
    board: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    max_height: u32,
    max_holes: u32,
    targets: &FxHashSet<BoardKey>,
    target_cells: &[u32],
    nodes: &mut u64,
) -> Option<(BoardKey, [u8; 7])> {
    let mut memo: FxHashSet<BoardStepKey> = FxHashSet::default();
    let mut placements = [0u8; 7];
    find_bag_witness_memo(
        board,
        pieces,
        0,
        max_height,
        max_holes,
        targets,
        target_cells,
        nodes,
        &mut memo,
        &mut placements,
    )
}

fn any_target_cell_feasible(board: TetrisBoard, step: usize, target_cells: &[u32]) -> bool {
    let remaining = (7 - step) as u32;
    let max_final_cells = board.count() + 4 * remaining;
    let required_residue = max_final_cells % 10;
    target_cells
        .iter()
        .any(|&cells| cells <= max_final_cells && cells % 10 == required_residue)
}

fn find_bag_witness_memo(
    board: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    step: usize,
    max_height: u32,
    max_holes: u32,
    targets: &FxHashSet<BoardKey>,
    target_cells: &[u32],
    nodes: &mut u64,
    memo: &mut FxHashSet<BoardStepKey>,
    placements: &mut [u8; 7],
) -> Option<(BoardKey, [u8; 7])> {
    if step == 7 {
        let key = board_key(&board);
        return targets.contains(&key).then_some((key, *placements));
    }

    if !any_target_cell_feasible(board, step, target_cells) {
        return None;
    }

    let memo_key = BoardStepKey {
        board: board_key(&board),
        step: step as u8,
    };
    if memo.contains(&memo_key) {
        return None;
    }

    let piece = pieces[step];
    for &placement in TetrisPiecePlacement::all_from_piece(piece) {
        let mut next = board;
        let result = next.apply_piece_placement(placement);
        *nodes += 1;

        if result.is_lost == IsLost::LOST {
            continue;
        }
        if next.height() > max_height {
            continue;
        }
        if max_holes != u32::MAX && has_too_many_holes(&next, max_holes) {
            continue;
        }

        placements[step] = placement.index();
        if let Some(witness) = find_bag_witness_memo(
            next,
            pieces,
            step + 1,
            max_height,
            max_holes,
            targets,
            target_cells,
            nodes,
            memo,
            placements,
        ) {
            return Some(witness);
        }
    }

    memo.insert(memo_key);
    None
}

pub fn replay_bag(
    mut board: TetrisBoard,
    placements: &[u8; 7],
    max_height: u32,
    max_holes: u32,
) -> Option<TetrisBoard> {
    for placement_idx in placements {
        let placement = TetrisPiecePlacement::from_index(*placement_idx);
        let result = board.apply_piece_placement(placement);
        if result.is_lost == IsLost::LOST {
            return None;
        }
        if board.height() > max_height {
            return None;
        }
        if max_holes != u32::MAX && has_too_many_holes(&board, max_holes) {
            return None;
        }
    }
    Some(board)
}

/// Inner DFS with memoization of failed (board, step) states.
/// The memo key packs board_hash and step into a single u64.
fn can_reach_any_memo(
    board: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    step: usize,
    max_height: u32,
    max_holes: u32,
    targets: &FxHashSet<u64>,
    nodes: &mut u64,
    memo: &mut FxHashSet<u64>,
) -> bool {
    if step == 7 {
        return targets.contains(&board_hash(&board));
    }

    // Memo key: mix board hash with step to avoid collisions across steps.
    let bh = board_hash(&board);
    let memo_key = bh.wrapping_mul(7).wrapping_add(step as u64);
    if memo.contains(&memo_key) {
        return false;
    }

    let piece = pieces[step];
    for &placement in TetrisPiecePlacement::all_from_piece(piece) {
        let mut next = board;
        let result = next.apply_piece_placement(placement);
        *nodes += 1;

        if result.is_lost == IsLost::LOST {
            continue;
        }
        if next.height() > max_height {
            continue;
        }
        if max_holes != u32::MAX && has_too_many_holes(&next, max_holes) {
            continue;
        }

        if can_reach_any_memo(
            next,
            pieces,
            step + 1,
            max_height,
            max_holes,
            targets,
            nodes,
            memo,
        ) {
            return true;
        }
    }

    memo.insert(memo_key);
    false
}

// ---------------------------------------------------------------------------
// Board display
// ---------------------------------------------------------------------------

pub fn print_board(board: &TetrisBoard) {
    let h = board.height() as usize;
    for row in (0..h).rev() {
        let mut s = String::with_capacity(10);
        for col in 0..10 {
            s.push(if board.get_bit(col, row) { '#' } else { '.' });
        }
        println!("    {row}: {s}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn board_key_round_trips_exact_board() {
        let mut board = TetrisBoard::new();
        board.set_bit(0, 0);
        board.set_bit(9, 19);

        let key = BoardKey::from_board(&board);

        assert_eq!(key.to_board(), board);
        assert_eq!(BoardKey::from_blob(key.as_bytes()).ok(), Some(key));
        assert_eq!(
            BoardKey::from_blob(&board.to_binary_slice()).ok(),
            Some(key)
        );
    }

    #[test]
    fn witness_replay_reaches_exact_target() {
        let perm = generate_permutations()[0];
        let mut target = TetrisBoard::new();
        let mut expected_placements = [0u8; 7];

        for (idx, piece) in perm.iter().enumerate() {
            let placement = TetrisPiecePlacement::all_from_piece(*piece)[0];
            expected_placements[idx] = placement.index();
            let result = target.apply_piece_placement(placement);
            assert_eq!(result.is_lost, IsLost::NOT_LOST);
        }

        let target_key = BoardKey::from_board(&target);
        let targets = FxHashSet::from_iter([target_key]);
        let mut nodes = 0;
        let witness = find_bag_witness(
            TetrisBoard::new(),
            &perm,
            TetrisBoard::HEIGHT as u32,
            u32::MAX,
            &targets,
            &mut nodes,
        );

        assert!(witness.is_some());
        let (found_key, placements) = witness.unwrap_or((target_key, [u8::MAX; 7]));
        assert_eq!(found_key, target_key);
        assert_eq!(
            replay_bag(
                TetrisBoard::new(),
                &placements,
                TetrisBoard::HEIGHT as u32,
                u32::MAX
            ),
            Some(target)
        );
        assert_eq!(placements, expected_placements);
    }

    #[test]
    fn witness_cell_feasibility_uses_count_and_residue() {
        let board = TetrisBoard::new();

        assert!(any_target_cell_feasible(board, 0, &[18]));
        assert!(!any_target_cell_feasible(board, 0, &[17]));
        assert!(!any_target_cell_feasible(board, 0, &[38]));
    }
}
