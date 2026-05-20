use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use rustc_hash::FxHashSet;
use tetris_game::TetrisBoard;

use crate::universe::BoardId;

const MAGIC: &[u8; 4] = b"CATL";
const VERSION: u32 = 2;

/// The cycle atlas: just the safe set of boards.
/// During play, run online AND-OR search each cycle to pick placements.
pub struct CycleAtlas {
    pub bag_cycles: u32,
    pub boards: Vec<TetrisBoard>,
    pub safe_board_ids: FxHashSet<BoardId>,
}

/// Extract the atlas from the safe set.
pub fn extract(
    all_boards: &[TetrisBoard],
    safe_set: &FxHashSet<BoardId>,
    bag_cycles: u32,
) -> CycleAtlas {
    CycleAtlas {
        bag_cycles,
        boards: all_boards.to_vec(),
        safe_board_ids: safe_set.clone(),
    }
}

/// Save the atlas to a binary file.
pub fn save(atlas: &CycleAtlas, all_boards: &[TetrisBoard], path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating atlas directory {}", parent.display()))?;
    }

    let mut f = std::fs::File::create(path).context("creating atlas file")?;

    // Header
    f.write_all(MAGIC)?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.write_all(&atlas.bag_cycles.to_le_bytes())?;

    // Total board count
    let total = all_boards.len() as u32;
    f.write_all(&total.to_le_bytes())?;

    // All board limbs
    for board in all_boards {
        for limb in &board.as_limbs() {
            f.write_all(&limb.to_le_bytes())?;
        }
    }

    // Safe set count
    let safe_count = atlas.safe_board_ids.len() as u32;
    f.write_all(&safe_count.to_le_bytes())?;

    // Safe board IDs
    for &bid in &atlas.safe_board_ids {
        f.write_all(&bid.to_le_bytes())?;
    }

    Ok(())
}

/// Load the atlas from a binary file.
pub fn load(path: &Path) -> Result<CycleAtlas> {
    let data = std::fs::read(path).context("reading atlas file")?;
    let mut cursor = &data[..];

    // Header
    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("not a cycle atlas file (bad magic)");
    }
    let mut ver_buf = [0u8; 4];
    cursor.read_exact(&mut ver_buf)?;
    let version = u32::from_le_bytes(ver_buf);
    if version != VERSION {
        bail!("unsupported atlas version {version}");
    }

    let mut bag_cycles_buf = [0u8; 4];
    cursor.read_exact(&mut bag_cycles_buf)?;
    let bag_cycles = u32::from_le_bytes(bag_cycles_buf);

    // Total board count
    let mut count_buf = [0u8; 4];
    cursor.read_exact(&mut count_buf)?;
    let total = u32::from_le_bytes(count_buf) as usize;

    // Read all boards
    let mut boards = Vec::with_capacity(total);
    for _ in 0..total {
        let mut limbs = [0u32; 10];
        for limb in &mut limbs {
            let mut buf = [0u8; 4];
            cursor.read_exact(&mut buf)?;
            *limb = u32::from_le_bytes(buf);
        }
        boards.push(board_from_limbs(limbs));
    }

    // Safe set count
    let mut safe_count_buf = [0u8; 4];
    cursor.read_exact(&mut safe_count_buf)?;
    let safe_count = u32::from_le_bytes(safe_count_buf) as usize;

    // Safe board IDs
    let mut safe_board_ids = FxHashSet::with_capacity_and_hasher(safe_count, Default::default());
    for _ in 0..safe_count {
        let mut bid_buf = [0u8; 4];
        cursor.read_exact(&mut bid_buf)?;
        let board_id = u32::from_le_bytes(bid_buf);
        if board_id as usize >= boards.len() {
            bail!(
                "atlas safe board id {board_id} is out of range for {} boards",
                boards.len()
            );
        }
        safe_board_ids.insert(board_id);
    }

    if !cursor.is_empty() {
        bail!("atlas file has {} trailing byte(s)", cursor.len());
    }

    Ok(CycleAtlas {
        bag_cycles,
        boards,
        safe_board_ids,
    })
}

/// Reconstruct a TetrisBoard from column limbs.
fn board_from_limbs(limbs: [u32; 10]) -> TetrisBoard {
    let mut board = TetrisBoard::EMPTY_BOARD;
    for (col, mut limb) in limbs.into_iter().enumerate() {
        while limb != 0 {
            let row = limb.trailing_zeros() as usize;
            board.set_bit(col, row);
            limb &= limb - 1;
        }
    }
    board
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn board_from_limbs_reconstructs_bits_without_unsafe_layout_assumption() {
        let mut limbs = [0u32; 10];
        limbs[0] = 0b101;
        limbs[9] = 1 << 19;

        let board = board_from_limbs(limbs);

        assert!(board.get_bit(0, 0));
        assert!(!board.get_bit(0, 1));
        assert!(board.get_bit(0, 2));
        assert!(board.get_bit(9, 19));
        assert_eq!(board.count(), 3);
    }

    #[test]
    fn extract_records_bag_cycles() {
        let boards = vec![TetrisBoard::EMPTY_BOARD];
        let mut safe = FxHashSet::default();
        safe.insert(0);

        let atlas = extract(&boards, &safe, 3);

        assert_eq!(atlas.bag_cycles, 3);
        assert_eq!(atlas.boards, boards);
        assert_eq!(atlas.safe_board_ids.len(), 1);
    }
}
