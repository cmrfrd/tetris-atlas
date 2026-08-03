use tetris_game::config::StandardTetris;
use tetris_game::tetris::{IsLost, TetrisGame, TetrisPiecePlacement};

use crate::compiled::CompiledForest;
use crate::encode::encode_game;
use crate::forest::DecisionForest;

/// The result of a full game playout.
#[derive(Clone, Copy, Debug)]
pub struct GameOutcome {
    pub pieces_placed: u64,
    pub lines_cleared: u64,
    pub max_height: u32,
    pub alive: bool,
}

/// The forest's move for the current game: encode the game to 214 bits and let
/// the forest vote among the legal placements. No lookahead — survival fitness
/// teaches the trees to avoid losing votes.
pub fn choose_placement(
    forest: &DecisionForest,
    game: &TetrisGame<StandardTetris>,
) -> Option<TetrisPiecePlacement> {
    let bits = encode_game(game);
    forest.vote(&bits, game.current_placements())
}

/// Play a full game with `forest` from `seed`, stopping at loss or after
/// `max_pieces` placements. Deterministic given `(seed, forest)`. One-shot
/// convenience: compiles the forest, then delegates. Batch callers (fitness)
/// should compile once and use `play_game_compiled` for every seed.
pub fn play_game(forest: &DecisionForest, seed: u64, max_pieces: u64) -> GameOutcome {
    play_game_compiled(&CompiledForest::compile(forest), seed, max_pieces)
}

/// Play a full game with an already-compiled forest — the hot path. Identical
/// results to `play_game` (the compiled form is semantics-preserving).
pub fn play_game_compiled(compiled: &CompiledForest, seed: u64, max_pieces: u64) -> GameOutcome {
    let mut game = TetrisGame::<StandardTetris>::new_with_seed(seed);
    let mut pieces_placed = 0u64;
    let mut lines_cleared = 0u64;
    let mut max_height = 0u32;

    while pieces_placed < max_pieces {
        if game.board.is_lost() {
            break;
        }
        let bits = encode_game(&game);
        let Some(placement) = compiled.vote(&bits, game.current_placements()) else {
            break;
        };
        let result = game.apply_placement(placement);
        pieces_placed += 1;
        lines_cleared += u64::from(result.lines_cleared);
        max_height = max_height.max(game.board.height());
        if result.is_lost == IsLost::LOST {
            return GameOutcome {
                pieces_placed,
                lines_cleared,
                max_height,
                alive: false,
            };
        }
    }

    GameOutcome {
        pieces_placed,
        lines_cleared,
        max_height,
        alive: !game.board.is_lost(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forest::FOREST_FORMAT_VERSION;
    use crate::tree::DecisionTree;

    fn always_pid0() -> DecisionForest {
        DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![DecisionTree::Leaf(0)],
        }
    }

    #[test]
    fn choose_placement_returns_a_legal_move() {
        let game = TetrisGame::<StandardTetris>::new_with_seed(5);
        let chosen = choose_placement(&always_pid0(), &game).unwrap();
        assert!(game.current_placements().contains(&chosen));
    }

    #[test]
    fn vote_is_always_legal_across_random_forests_and_states() {
        // Property: whatever the forest, the chosen placement is legal for the
        // current piece at every turn of a real game.
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        for fseed in 0..4u64 {
            let mut rng = SmallRng::seed_from_u64(fseed);
            let forest = DecisionForest::random(&mut rng, 6, 4, 3);
            let mut game = TetrisGame::<StandardTetris>::new_with_seed(fseed + 100);
            for _ in 0..40 {
                if game.board.is_lost() {
                    break;
                }
                let Some(chosen) = choose_placement(&forest, &game) else {
                    break;
                };
                assert!(
                    game.current_placements().contains(&chosen),
                    "illegal placement chosen"
                );
                if game.apply_placement(chosen).is_lost == IsLost::LOST {
                    break;
                }
            }
        }
    }

    #[test]
    fn forest_always_answers_while_the_game_is_live() {
        // Totality guarantee: even a forest of pure abstainers gets a move
        // every turn. vote() returns None only for an empty legal set, and the
        // engine's per-piece placement table is never empty in a live game —
        // so "no answer" is impossible while the game is alive.
        use crate::tree::DecisionTree;
        let silent = DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![DecisionTree::Abstain, DecisionTree::Abstain],
        };
        for seed in 0..3u64 {
            let mut game = TetrisGame::<StandardTetris>::new_with_seed(seed);
            let mut turns = 0u32;
            while !game.board.is_lost() && turns < 200 {
                assert!(
                    !game.current_placements().is_empty(),
                    "engine placement table empty in a live game"
                );
                let chosen = choose_placement(&silent, &game)
                    .expect("forest must always answer in a live game");
                assert!(game.current_placements().contains(&chosen));
                if game.apply_placement(chosen).is_lost == IsLost::LOST {
                    break;
                }
                turns += 1;
            }
        }
    }

    #[test]
    fn play_game_is_deterministic_and_makes_progress() {
        let forest = always_pid0();
        let a = play_game(&forest, 123, 500);
        let b = play_game(&forest, 123, 500);
        assert_eq!(a.pieces_placed, b.pieces_placed);
        assert_eq!(a.lines_cleared, b.lines_cleared);
        assert!(a.pieces_placed > 0);
    }
}
