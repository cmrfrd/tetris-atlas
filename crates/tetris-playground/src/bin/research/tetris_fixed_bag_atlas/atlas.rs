use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use rustc_hash::FxHashMap;
use tetris_game::{TetrisBoard, TetrisPiece, TetrisPiecePlacement};

use crate::retrograde::SolveResult;
use crate::state::StateId;
use crate::universe::Universe;

const MAGIC: &[u8; 4] = b"FBAG";
const VERSION: u32 = 1;

/// Compact atlas for play: maps (board, cycle_pos) -> list of winning placements.
pub struct PlayAtlas {
    pub cycle: [TetrisPiece; 7],
    pub lookup: FxHashMap<([u32; 10], u8), Vec<TetrisPiecePlacement>>,
}

/// Extract the play atlas from the solved universe.
pub fn extract(universe: &Universe, result: &SolveResult) -> PlayAtlas {
    let mut lookup: FxHashMap<([u32; 10], u8), Vec<TetrisPiecePlacement>> = FxHashMap::default();

    for sid in 0..universe.state_count() {
        if !result.alive[sid] {
            continue;
        }

        let key = &universe.states[sid];
        let board = &universe.boards[key.board_id as usize];
        let limbs = board.as_limbs();

        let mut actions: Vec<TetrisPiecePlacement> = Vec::new();
        for edge in universe.successors(sid as StateId) {
            if result.alive[edge.succ as usize] {
                actions.push(TetrisPiecePlacement::from_index(edge.placement));
            }
        }

        if !actions.is_empty() {
            actions.dedup();
            lookup.insert((limbs, key.cycle_pos), actions);
        }
    }

    PlayAtlas {
        cycle: universe.config.cycle,
        lookup,
    }
}

/// Save the atlas to a binary file.
pub fn save(atlas: &PlayAtlas, path: &Path) -> Result<()> {
    let mut f = std::fs::File::create(path).context("creating atlas file")?;

    // Header
    f.write_all(MAGIC)?;
    f.write_all(&VERSION.to_le_bytes())?;

    // Cycle: 7 piece indices
    for piece in &atlas.cycle {
        f.write_all(&[piece.index()])?;
    }

    // Entry count
    let count = atlas.lookup.len() as u32;
    f.write_all(&count.to_le_bytes())?;

    // Entries
    for (&(limbs, cycle_pos), actions) in &atlas.lookup {
        // Board: 10 x u32
        for limb in &limbs {
            f.write_all(&limb.to_le_bytes())?;
        }
        // Cycle pos
        f.write_all(&[cycle_pos])?;
        // Action count + packed indices
        let n = actions.len() as u8;
        f.write_all(&[n])?;
        for a in actions {
            f.write_all(&[a.index()])?;
        }
    }

    Ok(())
}

/// Load the atlas from a binary file.
pub fn load(path: &Path) -> Result<PlayAtlas> {
    let data = std::fs::read(path).context("reading atlas file")?;
    let mut cursor = &data[..];

    // Header
    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("not a fixed-bag atlas file (bad magic)");
    }
    let mut ver_buf = [0u8; 4];
    cursor.read_exact(&mut ver_buf)?;
    let version = u32::from_le_bytes(ver_buf);
    if version != VERSION {
        bail!("unsupported atlas version {version}");
    }

    // Cycle
    let mut cycle_raw = [0u8; 7];
    cursor.read_exact(&mut cycle_raw)?;
    let cycle = cycle_raw.map(TetrisPiece::new);

    // Entry count
    let mut count_buf = [0u8; 4];
    cursor.read_exact(&mut count_buf)?;
    let count = u32::from_le_bytes(count_buf) as usize;

    let mut lookup: FxHashMap<([u32; 10], u8), Vec<TetrisPiecePlacement>> =
        FxHashMap::with_capacity_and_hasher(count, Default::default());

    for _ in 0..count {
        // Board limbs
        let mut limbs = [0u32; 10];
        for limb in &mut limbs {
            let mut buf = [0u8; 4];
            cursor.read_exact(&mut buf)?;
            *limb = u32::from_le_bytes(buf);
        }
        // Cycle pos
        let mut pos_buf = [0u8; 1];
        cursor.read_exact(&mut pos_buf)?;
        let cycle_pos = pos_buf[0];
        // Actions
        let mut n_buf = [0u8; 1];
        cursor.read_exact(&mut n_buf)?;
        let n = n_buf[0] as usize;
        let mut actions = Vec::with_capacity(n);
        for _ in 0..n {
            let mut idx_buf = [0u8; 1];
            cursor.read_exact(&mut idx_buf)?;
            actions.push(TetrisPiecePlacement::from_index(idx_buf[0]));
        }
        lookup.insert((limbs, cycle_pos), actions);
    }

    Ok(PlayAtlas { cycle, lookup })
}

/// Reconstruct a TetrisBoard from column limbs.
pub fn board_from_limbs(limbs: [u32; 10]) -> TetrisBoard {
    // SAFETY: TetrisBoard is a newtype wrapper around [u32; 10] with identical layout.
    unsafe { std::mem::transmute(limbs) }
}
