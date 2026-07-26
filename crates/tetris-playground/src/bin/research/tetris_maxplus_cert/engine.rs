//! Board ↔ surface adapter. The certificate lives on the surface (column heights);
//! the ground-truth dynamics stay in the real `tetris-game` engine.

use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};

pub const W: usize = 10;

/// Per-column surface heights (top of each column) as i64, so v-subtraction can go negative.
pub fn heights_i64(b: &TetrisBoard) -> [i64; W] {
    let h = b.heights(); // [u32; 10]
    let mut out = [0i64; W];
    for j in 0..W {
        out[j] = h[j] as i64;
    }
    out
}

/// Hilbert oscillation = max - min. NOT the engine's bumpiness `roughness()`.
pub fn osc(h: &[i64; W]) -> i64 {
    let mx = h.iter().copied().max().unwrap_or(0);
    let mn = h.iter().copied().min().unwrap_or(0);
    mx - mn
}

pub fn max_height(h: &[i64; W]) -> i64 {
    h.iter().copied().max().unwrap_or(0)
}

pub fn holes_of(b: &TetrisBoard) -> u32 {
    b.total_holes()
}

/// Apply a placement to a copy of the board. Returns None if it tops out,
/// else the resulting board and the number of lines cleared.
pub fn successor(b: &TetrisBoard, pl: TetrisPiecePlacement) -> Option<(TetrisBoard, u32)> {
    let mut nb = *b;
    let res = nb.apply_piece_placement(pl);
    if res.is_lost == IsLost::LOST {
        None
    } else {
        Some((nb, res.lines_cleared))
    }
}

pub fn placements(p: TetrisPiece) -> &'static [TetrisPiecePlacement] {
    TetrisPiecePlacement::all_from_piece(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_game::{TetrisBoard, TetrisPiece, TetrisPiecePlacement};

    #[test]
    fn empty_board_is_flat_zero() {
        let h = heights_i64(&TetrisBoard::new());
        assert_eq!(h, [0i64; W]);
        assert_eq!(osc(&h), 0);
        assert_eq!(max_height(&h), 0);
    }

    #[test]
    fn osc_is_max_minus_min() {
        let h = [3, 0, 5, 1, 1, 1, 1, 1, 1, 2];
        assert_eq!(osc(&h), 5); // 5 - 0
        assert_eq!(max_height(&h), 5);
    }

    #[test]
    fn some_placement_raises_height_and_succeeds() {
        let b = TetrisBoard::new();
        let pl = placements(TetrisPiece::O_PIECE)[0];
        let (nb, cleared) = successor(&b, pl).expect("O on empty must not lose");
        assert_eq!(cleared, 0);
        assert_eq!(max_height(&heights_i64(&nb)), 2); // O is 2 tall
        assert_eq!(holes_of(&nb), 0);
    }
}
