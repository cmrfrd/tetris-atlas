//! Deterministic board populations shared by BoardSet operation benchmarks.

use tetris_game::{StandardTetris, TetrisGame};

#[allow(
    clippy::cast_possible_truncation,
    reason = "The fixed-seed generator intentionally returns the low 32 bits of its xorshift state."
)]
pub fn random(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state as u32
}

pub fn boards(kind: &str, n: usize) -> Vec<[u32; 10]> {
    let mut seed = 0xDEADBEEFCAFE;
    let mut game = TetrisGame::<StandardTetris>::new_with_seed(1);
    let mut game_seed = 1;
    (0..n)
        .map(|_| match kind {
            "engine" => {
                let placements = game.current_placements();
                let placement = placements[random(&mut seed) as usize % placements.len()];
                game.apply_placement(placement);
                let board = *game.board.as_limbs();
                if game.board.is_lost() {
                    game_seed += 1;
                    game = TetrisGame::new_with_seed(game_seed);
                }
                board
            }
            "shared" => std::array::from_fn(|i| {
                if i < 8 {
                    0x55
                } else {
                    random(&mut seed) & 0xffff
                }
            }),
            _ => std::array::from_fn(|_| random(&mut seed)),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn benchmark_populations_keep_their_fixed_seed_fingerprints() {
        // Freeze the existing 16k populations, including normalized terminal
        // markers in the engine trace. Changing these requires fresh baselines.
        let observed = ["engine", "random", "shared"].map(|kind| {
            let keys = boards(kind, 16_384);
            assert_eq!(keys.len(), 16_384);
            let unique = keys.iter().copied().collect::<HashSet<_>>().len();
            let fingerprint = keys
                .iter()
                .flatten()
                .flat_map(|col| col.to_le_bytes())
                .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
                    (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
                });
            (unique, fingerprint)
        });
        assert_eq!(
            observed,
            [
                (14_884, 2_766_784_593_511_881_075),
                (16_384, 2_173_367_724_953_556_083),
                (16_366, 4_294_118_890_815_904_933),
            ]
        );
    }
}
