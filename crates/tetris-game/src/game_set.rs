use std::ops::{Index, IndexMut};

use tetris_utils::HeaplessVec;

use crate::tetris::{
    IsLost, TetrisBoard, TetrisGame, TetrisPiece, TetrisPieceBag, TetrisPieceBagState,
    TetrisPieceOrientation, TetrisPiecePlacement,
};

/// Maximum number of games in a [`TetrisGameSet`].
///
/// This limit exists because `TetrisGameSet` is stack-allocated for performance.
/// Adjust this constant if you need larger batch sizes.
pub const MAX_GAMES: usize = 1024;

/// Error type for `TetrisGameSet` operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TetrisGameSetError {
    /// The requested number of games exceeds the maximum capacity.
    TooManyGames { requested: usize, max: usize },
}

impl std::fmt::Display for TetrisGameSetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyGames { requested, max } => {
                write!(f, "requested {requested} games, but maximum is {max}")
            }
        }
    }
}

impl std::error::Error for TetrisGameSetError {}

/// A collection of Tetris games for parallel/batch operations.
///
/// This struct enables efficient batch processing of multiple games simultaneously,
/// useful for:
/// - Monte Carlo simulations
/// - Reinforcement learning with parallel environments
/// - Batch evaluation of AI agents
///
/// The set is stack-allocated with a maximum capacity of [`MAX_GAMES`].
#[derive(Clone, Copy, Debug)]
pub struct TetrisGameSet(pub HeaplessVec<TetrisGame, MAX_GAMES>);

impl TetrisGameSet {
    /// Create a new TetrisGameSet with N default games.
    pub fn new(num_games: usize) -> Self {
        assert!(
            num_games <= MAX_GAMES,
            "Too many games. MAX_GAMES = {}",
            MAX_GAMES
        );
        let mut games = HeaplessVec::new();
        (0..num_games).for_each(|_| games.push(TetrisGame::new()));
        Self(games)
    }

    /// Create a new TetrisGameSet with N games using the provided seed.
    ///
    /// Each game gets a slightly different seed (seed + index).
    ///
    pub fn new_with_seed(seed: u64, num_games: usize) -> Self {
        assert!(
            num_games <= MAX_GAMES,
            "Too many games. MAX_GAMES = {}",
            MAX_GAMES
        );
        let mut games = HeaplessVec::new();
        (0..num_games).for_each(|i| games.push(TetrisGame::new_with_seed(seed + i as u64)));
        Self(games)
    }

    /// Create a new TetrisGameSet with N games using the same seed.
    pub fn new_with_same_seed(seed: u64, num_games: usize) -> Self {
        assert!(
            num_games <= MAX_GAMES,
            "Too many games. MAX_GAMES = {}",
            MAX_GAMES
        );
        let mut games = HeaplessVec::new();
        (0..num_games).for_each(|_| games.push(TetrisGame::new_with_seed(seed)));
        Self(games)
    }

    /// Returns the number of games in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the set contains no games.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Create a `TetrisGameSet` from a slice of games.
    pub fn from_games(games: &[TetrisGame]) -> Self {
        assert!(
            games.len() <= MAX_GAMES,
            "games.len() must be less than or equal to MAX_GAMES"
        );
        let mut input_games = HeaplessVec::new();
        input_games.fill_from_slice(games);
        Self(input_games)
    }

    /// Returns all boards as a vector.
    #[must_use]
    pub fn boards(&self) -> HeaplessVec<TetrisBoard, MAX_GAMES> {
        self.0.map(|game| game.board)
    }

    /// Returns all current pieces as a vector.
    #[must_use]
    pub fn current_pieces(&self) -> HeaplessVec<TetrisPiece, MAX_GAMES> {
        self.0.map(|game| game.current_piece)
    }

    /// Returns all piece counts as a vector.
    #[must_use]
    pub fn piece_counts(&self) -> HeaplessVec<u32, MAX_GAMES> {
        self.0.map(|game| game.piece_count)
    }

    /// Get the current placements for all games.
    ///
    /// These are the placements that can be applied to the current piece.
    #[must_use]
    pub fn current_placements(&self) -> Vec<&[TetrisPiecePlacement]> {
        self.0
            .into_iter()
            .map(|game| game.current_placements())
            .collect()
    }

    /// Apply a placement to the board.
    ///
    /// This will return true if the game is lost, false otherwise.
    /// Lines cleared are tracked by measuring the difference in height before and after the placement.
    ///
    /// If the game is not lost, the current piece is replaced with a new random piece.
    pub fn apply_placement(&mut self, placements: &[TetrisPiecePlacement]) -> Vec<IsLost> {
        self.0
            .into_iter_mut()
            .zip(placements)
            .map(|(game, &placement)| game.apply_placement(placement).is_lost)
            .collect()
    }

    /// Apply a placement from orientations to the board.
    ///
    /// This will return true if the game is lost, false otherwise.
    /// Lines cleared are tracked by measuring the difference in height before and after the placement.
    ///
    /// If the game is not lost, the current piece is replaced with a new random piece.
    pub fn apply_placement_from_orientations(
        &mut self,
        orientations: &[TetrisPieceOrientation],
    ) -> Vec<IsLost> {
        self.0
            .into_iter_mut()
            .zip(orientations)
            .map(|(game, &orientation)| {
                game.apply_placement(TetrisPiecePlacement {
                    piece: game.current_piece,
                    orientation,
                })
                .is_lost
            })
            .collect()
    }

    /// Resets any games that are in a lost state.
    ///
    /// Each lost game is reset with a new seed derived from its RNG.
    /// Returns the number of games that were reset.
    pub fn reset_lost_games(&mut self) -> usize {
        self.0
            .into_iter_mut()
            .map(|game| {
                if game.board.is_lost() {
                    let next_seed = game.rng.next_u64();
                    game.reset(Some(next_seed));
                    1
                } else {
                    0
                }
            })
            .sum()
    }

    /// Resets all games to their initial state (original seeds).
    pub fn reset_all(&mut self) {
        self.0.into_iter_mut().for_each(|game| game.reset(None));
    }

    /// Permute the gameset using the provided permutation vector.
    ///
    /// The permutation vector must be the same length as the gameset and contain
    /// valid indices (0..len). Each index should appear exactly once.
    pub fn permute(&mut self, permutation: &[usize]) {
        assert_eq!(permutation.len(), self.len(), "Permutation length mismatch");
        let mut new_games = HeaplessVec::new();
        for &idx in permutation {
            new_games.push(*self.0.get(idx).expect("permutation index out of bounds"));
        }
        self.0 = new_games;
    }

    /// Removes all games that are in a lost state from the set.
    pub fn drop_lost_games(&mut self) {
        self.0.retain(|game| !game.board.is_lost());
    }
}

impl Index<usize> for TetrisGameSet {
    type Output = TetrisGame;

    fn index(&self, index: usize) -> &Self::Output {
        self.0
            .get(index)
            .expect("TetrisGameSet index out of bounds")
    }
}

impl IndexMut<usize> for TetrisGameSet {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0
            .get_mut(index)
            .expect("TetrisGameSet index out of bounds")
    }
}
