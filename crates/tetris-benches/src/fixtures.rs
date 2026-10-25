//! Deterministic live inputs for the placement and line-clear benchmarks.

use tetris_game::{
    StandardTetris, TetrisBoard, TetrisGame, TetrisGameConfig, TetrisGameRng, TetrisPiecePlacement,
};

pub(crate) const SEED: u64 = 0xDEAD_BEEF_CAFE;
pub(crate) const GAME_SEEDS: std::ops::RangeInclusive<u64> = 1..=16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlacementFixture {
    pub board: TetrisBoard,
    pub placement: TetrisPiecePlacement,
}

/// All candidate moves at 64 successive reachable positions for each of 16 seeds.
/// The fixture policy is deliberately fixed and uses integer features only.
pub(crate) fn placements() -> Vec<PlacementFixture> {
    let mut fixtures = Vec::new();
    for seed in GAME_SEEDS {
        let mut game: TetrisGame = TetrisGame::new_with_seed(seed);
        for _ in 0..64 {
            let mut best = None;
            for &placement in game.current_placements() {
                fixtures.push(PlacementFixture {
                    board: game.board,
                    placement,
                });
                let mut next = game.board;
                let result = next.apply_piece_placement(placement);
                let key = (
                    bool::from(result.is_lost),
                    next.total_holes() * 16 + next.height() * 4 + next.roughness(),
                    placement.index(),
                );
                if best.is_none_or(|(old, _)| key < old) {
                    best = Some((key, placement));
                }
            }
            if let Some((_, placement)) = best {
                if game.apply_placement(placement).is_lost.into() {
                    game.reset(Some(seed));
                }
            }
        }
    }
    fixtures
}

/// Independent columns, with exactly 0..4 full playable rows per board.
pub(crate) fn rows() -> Vec<TetrisBoard> {
    let mut rng = TetrisGameRng::new(SEED);
    (0..1_280)
        .map(|i| {
            let mut mask = 0u32;
            while mask.count_ones() < i % 5 {
                mask |= 1 << (rng.next_u64() % StandardTetris::ROWS as u64);
            }
            let mut cols = [0; 10];
            for col in &mut cols {
                *col = (rng.next_u64() as u32 & StandardTetris::ROW_MASK) | mask;
            }
            let full = cols.iter().copied().fold(u32::MAX, |a, b| a & b);
            cols[0] &= !(full & !mask);
            TetrisBoard::from_limbs(cols)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixtures_repeat_and_cover_real_moves_and_clear_counts() {
        let placements = placements();
        assert_eq!(placements, super::placements());
        assert!(placements.len() > 20_000);
        assert!(placements.iter().all(|f| !f.board.is_lost()));
        assert!(placements.iter().any(|f| f.board.count() > 0));
        for (index, mut board) in rows().into_iter().enumerate() {
            assert_eq!(board.clear_filled_rows(), index as u32 % 5);
        }
    }
}
