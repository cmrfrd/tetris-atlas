use std::{
    array,
    marker::PhantomData,
    ops::Range,
    sync::atomic::{AtomicUsize, Ordering},
};

use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, DatasetIterator},
    },
    prelude::*,
};
use rand::{Rng, RngCore, seq::IndexedRandom};

use crate::tetris::{
    BOARD_SIZE, IsLost, TetrisBoardRaw, TetrisGame, TetrisGameSet, TetrisPiece,
    TetrisPiecePlacement,
};

pub fn gameset_into_board_dist_tensor<B: Backend>(
    game_set: &TetrisGameSet,
    device: &B::Device,
) -> Tensor<B, 3> {
    let batch_size = game_set.len();
    let board_size = BOARD_SIZE;

    let current_boards = game_set
        .boards()
        .to_vec()
        .into_iter()
        .map(|board| Tensor::<B, 1>::from_floats(board.to_binary_slice(), device))
        .collect::<Vec<_>>();
    let current_boards: Tensor<B, 2> = Tensor::stack(current_boards, 0);

    let a: Tensor<B, 3> = current_boards.reshape([batch_size, board_size, 1]);
    let b = Tensor::<B, 3>::ones([batch_size, board_size, 1], &device) - a.clone();

    let result = Tensor::cat(vec![a, b], 2);
    debug_assert_eq!(result.dims(), [batch_size, board_size, 2]);
    result
}

#[derive(Clone, Copy, Debug)]
pub struct CopyRange<T> {
    pub start: T,
    pub end: T,
}

impl<T> Into<Range<T>> for CopyRange<T> {
    fn into(self) -> Range<T> {
        Range {
            start: self.start,
            end: self.end,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TetrisDatasetUniformConfig {
    pub seed: u64,
    pub num_pieces_range: CopyRange<usize>,
    pub length: usize,
}

pub enum TetrisDatasetConfig {
    Uniform(TetrisDatasetUniformConfig),
}

#[derive(Clone, Debug, Copy)]

pub struct TetrisDatasetItem {
    pub current_board: TetrisBoardRaw,
    pub placement: TetrisPiecePlacement,
    pub result_board: TetrisBoardRaw,
    pub is_lost: IsLost,
}

pub struct TetrisDataset {
    config: TetrisDatasetConfig,
}

impl TetrisDataset {
    pub fn new(config: TetrisDatasetConfig) -> Self {
        Self { config }
    }

    pub fn train(config: TetrisDatasetConfig) -> Self {
        Self::new(config)
    }

    pub fn validation(config: TetrisDatasetConfig) -> Self {
        Self::new(config)
    }

    pub fn test(config: TetrisDatasetConfig) -> Self {
        Self::new(config)
    }
}

impl Dataset<TetrisDatasetItem> for TetrisDataset {
    fn get(&self, _index: usize) -> Option<TetrisDatasetItem> {
        let seed = rand::rng().next_u64();
        let mut tetris_game = TetrisGame::new_with_seed(seed);

        match self.config {
            TetrisDatasetConfig::Uniform(config) => {
                // Seed a random board by playing random pieces
                let range: Range<usize> = config.num_pieces_range.into();
                let num_pieces = rand::rng().random_range(range);
                for _i in 0..num_pieces {
                    let placement = *tetris_game
                        .current_placements()
                        .choose(&mut rand::rng())
                        .unwrap();
                    tetris_game.apply_placement(placement);
                }

                // Now we generate the next dataset item
                // get our current board
                let current_board = tetris_game.board();
                let placement = *tetris_game
                    .current_placements()
                    .choose(&mut rand::rng())
                    .unwrap();
                let is_lost = tetris_game.apply_placement(placement);
                let result_board = tetris_game.board();

                Some(TetrisDatasetItem {
                    current_board,
                    placement,
                    result_board,
                    is_lost,
                })
            }
        }
    }

    fn len(&self) -> usize {
        match self.config {
            TetrisDatasetConfig::Uniform(config) => config.length,
        }
    }

    fn iter(&self) -> DatasetIterator<'_, TetrisDatasetItem> {
        DatasetIterator::new(self)
    }
}

#[derive(Clone, Debug)]
pub struct TetrisBatcher<B: Backend> {
    _marker: PhantomData<B>,
}

impl<B: Backend> Default for TetrisBatcher<B> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TetrisBatch<B: Backend> {
    /// B := Batch size
    /// T := Tetris board size (vectorized)
    /// P := Piece placement size (vectorized) (scalar index)
    pub current_boards: Tensor<B, 2>, // [batch_size, T]
    pub placements: Tensor<B, 2>,    // [batch_size, 1]
    pub result_boards: Tensor<B, 2>, // [batch_size, T]
}

impl<B: Backend> Batcher<B, TetrisDatasetItem, TetrisBatch<B>> for TetrisBatcher<B> {
    fn batch(&self, items: Vec<TetrisDatasetItem>, device: &B::Device) -> TetrisBatch<B> {
        let current_boards = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats(item.current_board.to_binary_slice(), device))
            .collect();
        let current_boards = Tensor::stack(current_boards, 0);

        let placements = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats([item.placement.index()], device))
            .collect();
        let placements = Tensor::stack(placements, 0);

        let result_boards = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats(item.result_board.to_binary_slice(), device))
            .collect();
        let result_boards = Tensor::stack(result_boards, 0);

        TetrisBatch {
            current_boards,
            placements,
            result_boards,
        }
    }
}

/// A batch of tokenized tetris boards.
///
/// The main difference between this and tetris batch
/// is the resulting board outputs a [a, b] output meaning
/// which "class" the cell is likely to be (filled or empty)
#[derive(Clone, Debug)]
pub struct TetrisDistBatch<B: Backend> {
    /// B := Batch size
    /// T := Tetris board size (vectorized) (2 cell states)
    /// P := Piece placement size (vectorized) (162 possible placements)
    /// S := Tetris cell state size (2)
    pub current_boards_dist: Tensor<B, 3>, // [batch_size, T, S]
    pub placements_dist: Tensor<B, 2>,    // [batch_size, P]
    pub result_boards_dist: Tensor<B, 3>, // [batch_size, T, S]
}

#[derive(Clone, Debug)]
pub struct TetrisDistBatcher<B: Backend> {
    tetris_batcher: TetrisBatcher<B>,
}

impl<B: Backend> Default for TetrisDistBatcher<B> {
    fn default() -> Self {
        Self {
            tetris_batcher: TetrisBatcher::default(),
        }
    }
}

impl<B: Backend> Batcher<B, TetrisDatasetItem, TetrisDistBatch<B>> for TetrisDistBatcher<B> {
    fn batch(&self, items: Vec<TetrisDatasetItem>, device: &B::Device) -> TetrisDistBatch<B> {
        let tetris_batch = self.tetris_batcher.batch(items, device);

        let current_boards = tetris_batch.current_boards;
        let result_boards = tetris_batch.result_boards;

        let [batch_size, board_size] = result_boards.dims();

        let a = current_boards.reshape([batch_size, board_size, 1]);
        let b = Tensor::<B, 3>::ones([batch_size, board_size, 1], device) - a.clone();
        let current_boards_dist = Tensor::cat(vec![a, b], 2);

        let a = result_boards.reshape([batch_size, board_size, 1]);
        let b = Tensor::<B, 3>::ones([batch_size, board_size, 1], device) - a.clone();
        let result_boards_dist = Tensor::cat(vec![a, b], 2);

        let placements_dist: Tensor<B, 2> = tetris_batch
            .placements
            .squeeze::<1>(1)
            .one_hot(TetrisPiecePlacement::NUM_PLACEMENTS);

        TetrisDistBatch {
            current_boards_dist,
            placements_dist,
            result_boards_dist,
        }
    }
}

/// A dataset of tetris sequences.
///
/// The main difference between this and tetris batch
/// is the resulting board outputs a [a, b] output meaning
/// which "class" the cell is likely to be (filled or empty)

#[derive(Clone, Copy, Debug)]
pub struct TetrisSequenceDatasetUniformConfig {
    pub seed: u64,
    pub num_pieces_range: CopyRange<usize>,
    pub length: usize,
    pub sequence_length: usize,
}

#[derive(Clone, Debug)]
pub enum TetrisSequenceDatasetConfig {
    Uniform(TetrisSequenceDatasetUniformConfig),
}

#[derive(Clone, Debug)]
pub struct TetrisSequenceDatasetItem {
    pub current_boards: Vec<TetrisBoardRaw>,
    pub placements: Vec<TetrisPiecePlacement>,
    pub result_boards: Vec<TetrisBoardRaw>,
    pub is_lost: Vec<IsLost>,
}

impl Into<Vec<TetrisDatasetItem>> for &TetrisSequenceDatasetItem {
    fn into(self) -> Vec<TetrisDatasetItem> {
        let mut items = Vec::new();
        for i in 0..self.current_boards.len() {
            items.push(TetrisDatasetItem {
                current_board: self.current_boards[i],
                placement: self.placements[i],
                result_board: self.result_boards[i],
                is_lost: self.is_lost[i],
            });
        }
        items
    }
}

pub struct TetrisSequenceDataset {
    config: TetrisSequenceDatasetConfig,
}

impl TetrisSequenceDataset {
    pub fn new(config: TetrisSequenceDatasetConfig) -> Self {
        Self { config }
    }

    pub fn train(config: TetrisSequenceDatasetConfig) -> Self {
        Self::new(config)
    }

    pub fn validation(config: TetrisSequenceDatasetConfig) -> Self {
        Self::new(config)
    }

    pub fn test(config: TetrisSequenceDatasetConfig) -> Self {
        Self::new(config)
    }
}

impl Dataset<TetrisSequenceDatasetItem> for TetrisSequenceDataset {
    fn get(&self, _index: usize) -> Option<TetrisSequenceDatasetItem> {
        let seed = rand::rng().next_u64();
        let mut tetris_game = TetrisGame::new_with_seed(seed);

        match self.config {
            TetrisSequenceDatasetConfig::Uniform(config) => {
                // Seed a random board by playing random pieces
                let range: Range<usize> = config.num_pieces_range.into();
                let num_pieces = rand::rng().random_range(range);
                for _i in 0..num_pieces {
                    let placement = *tetris_game
                        .current_placements()
                        .choose(&mut rand::rng())
                        .unwrap();
                    tetris_game.apply_placement(placement);
                }

                let mut current_boards: Vec<TetrisBoardRaw> = Vec::new();
                let mut placements: Vec<TetrisPiecePlacement> = Vec::new();
                let mut result_boards: Vec<TetrisBoardRaw> = Vec::new();
                let mut is_losts: Vec<IsLost> = Vec::new();

                for i in 0..config.sequence_length {
                    let current_board = tetris_game.board();
                    current_boards.push(current_board);

                    let placement = *tetris_game
                        .current_placements()
                        .choose(&mut rand::rng())
                        .unwrap();
                    placements.push(placement);
                    let is_lost = tetris_game.apply_placement(placement);
                    is_losts.push(is_lost);

                    let result_board = tetris_game.board();
                    result_boards.push(result_board);
                }

                Some(TetrisSequenceDatasetItem {
                    current_boards,
                    placements,
                    result_boards,
                    is_lost: is_losts,
                })
            }
        }
    }

    fn len(&self) -> usize {
        match self.config {
            TetrisSequenceDatasetConfig::Uniform(config) => config.length,
        }
    }

    fn iter(&self) -> DatasetIterator<'_, TetrisSequenceDatasetItem> {
        DatasetIterator::new(self)
    }
}

#[derive(Clone, Debug)]
pub struct TetrisSequenceDistBatch<B: Backend> {
    /// B := Batch size
    /// T := Tetris board size (vectorized) (2 cell states)
    /// P := Piece placement size (vectorized) (162 possible placements)
    /// S := Tetris cell state size (2)
    /// L := Sequence length
    pub current_boards_dist: Tensor<B, 4>, // [batch_size, L, T, S]
    pub placements_dist: Tensor<B, 3>,    // [batch_size, L, P]
    pub result_boards_dist: Tensor<B, 4>, // [batch_size, L, T, S]
}

impl<B: Backend> TetrisSequenceDistBatch<B> {
    fn get_seq(&self, index: usize) -> TetrisDistBatch<B> {
        let [_, seq_len, _, _] = self.current_boards_dist.dims();
        debug_assert!(0 <= index && index < seq_len, "Index out of bounds");

        let device = self.current_boards_dist.device();

        let indices = Tensor::<B, 1, Int>::from_data([index], &device);
        let current_boards_dist = self
            .current_boards_dist
            .clone()
            .select(1, indices.clone())
            .squeeze::<3>(1);
        let placements_dist = self
            .placements_dist
            .clone()
            .select(1, indices.clone())
            .squeeze::<2>(1);
        let result_boards_dist = self
            .result_boards_dist
            .clone()
            .select(1, indices.clone())
            .squeeze::<3>(1);

        TetrisDistBatch {
            current_boards_dist,
            placements_dist,
            result_boards_dist,
        }
    }

    pub fn iter_seq(&self) -> impl Iterator<Item = TetrisDistBatch<B>> {
        let [_, seq_len, _, _] = self.current_boards_dist.dims();
        (0..seq_len).map(|i| self.get_seq(i))
    }
}

#[derive(Clone, Debug)]
pub struct TetrisSequenceDistBatcher<B: Backend> {
    tetris_dist_batcher: TetrisDistBatcher<B>,
}

impl<B: Backend> Default for TetrisSequenceDistBatcher<B> {
    fn default() -> Self {
        Self {
            tetris_dist_batcher: TetrisDistBatcher::default(),
        }
    }
}

impl<B: Backend> Batcher<B, TetrisSequenceDatasetItem, TetrisSequenceDistBatch<B>>
    for TetrisSequenceDistBatcher<B>
{
    fn batch(
        &self,
        items: Vec<TetrisSequenceDatasetItem>,
        device: &B::Device,
    ) -> TetrisSequenceDistBatch<B> {
        let sequence_length = items[0].current_boards.len();
        let batch_size = items.len();

        let mut dist_batches: Vec<TetrisDistBatch<B>> = Vec::new();
        for item in items.iter() {
            // Conver a single sequence into a "batch" representation
            // ex: Boards are now [L, T, S]
            let seq_as_vec_items: Vec<TetrisDatasetItem> = item.into();
            let seq_as_dist_batch = self.tetris_dist_batcher.batch(seq_as_vec_items, device);
            dist_batches.push(seq_as_dist_batch);
        }

        let current_boards_dist = Tensor::stack(
            dist_batches
                .iter()
                .map(|batch| batch.current_boards_dist.clone())
                .collect(),
            0,
        );
        assert_eq!(
            current_boards_dist.dims(),
            [batch_size, sequence_length, BOARD_SIZE, 2]
        );

        let placements_dist = Tensor::stack(
            dist_batches
                .iter()
                .map(|batch| batch.placements_dist.clone())
                .collect(),
            0,
        );
        assert_eq!(
            placements_dist.dims(),
            [
                batch_size,
                sequence_length,
                TetrisPiecePlacement::NUM_PLACEMENTS
            ]
        );

        let result_boards_dist = Tensor::stack(
            dist_batches
                .iter()
                .map(|batch| batch.result_boards_dist.clone())
                .collect(),
            0,
        );
        assert_eq!(
            result_boards_dist.dims(),
            [batch_size, sequence_length, BOARD_SIZE, 2]
        );

        TetrisSequenceDistBatch {
            current_boards_dist,
            placements_dist,
            result_boards_dist,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TetrisInitDistDatasetItem {
    pub game: TetrisGame,
}

pub struct TetrisInitDistDataset {
    config: TetrisDatasetConfig,
}

impl TetrisInitDistDataset {
    pub fn new(config: TetrisDatasetConfig) -> Self {
        Self { config }
    }

    pub fn train(config: TetrisDatasetConfig) -> Self {
        Self::new(config)
    }
}

impl Dataset<TetrisInitDistDatasetItem> for TetrisInitDistDataset {
    fn get(&self, _index: usize) -> Option<TetrisInitDistDatasetItem> {
        let seed = rand::rng().next_u64();
        let mut tetris_game = TetrisGame::new_with_seed(seed);

        match self.config {
            TetrisDatasetConfig::Uniform(config) => {
                // Seed a random board by playing random pieces
                let range: Range<usize> = config.num_pieces_range.into();
                let num_pieces = rand::rng().random_range(range);
                for _i in 0..num_pieces {
                    let placement = *tetris_game
                        .current_placements()
                        .choose(&mut rand::rng())
                        .unwrap();
                    tetris_game.apply_placement(placement);
                }

                Some(TetrisInitDistDatasetItem { game: tetris_game })
            }
        }
    }

    fn len(&self) -> usize {
        match self.config {
            TetrisDatasetConfig::Uniform(config) => config.length,
        }
    }

    fn iter(&self) -> DatasetIterator<'_, TetrisInitDistDatasetItem> {
        DatasetIterator::new(self)
    }
}

#[derive(Clone, Debug)]
pub struct TetrisInitDistBatch<B: Backend> {
    /// B := Batch size
    /// T := Tetris board size (vectorized) (2 cell states)
    /// P := Piece placement size (vectorized) (162 possible placements)
    /// S := Tetris cell state size (2)
    pub game_set: TetrisGameSet,
    pub current_boards: Tensor<B, 3>, // [batch_size, T, S]
}

#[derive(Clone, Debug)]
pub struct TetrisInitDistBatcher<B: Backend> {
    _marker: PhantomData<B>,
}

impl<B: Backend> Default for TetrisInitDistBatcher<B> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, TetrisInitDistDatasetItem, TetrisInitDistBatch<B>>
    for TetrisInitDistBatcher<B>
{
    fn batch(
        &self,
        items: Vec<TetrisInitDistDatasetItem>,
        device: &B::Device,
    ) -> TetrisInitDistBatch<B> {
        let game_set = TetrisGameSet::from_games(
            items
                .iter()
                .map(|item| item.game)
                .collect::<Vec<TetrisGame>>()
                .as_slice(),
        );
        let current_boards = gameset_into_board_dist_tensor(&game_set, device);

        TetrisInitDistBatch {
            game_set,
            current_boards,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tetris::{BOARD_SIZE, Clearer};
    use burn::backend::ndarray::NdArray;

    #[test]
    fn test_tetris_batcher_shapes() {
        let dataset =
            TetrisDataset::train(TetrisDatasetConfig::Uniform(TetrisDatasetUniformConfig {
                seed: 123,
                num_pieces_range: CopyRange { start: 0, end: 64 },
                length: 1_000_000,
            }));

        let tetris_batcher = TetrisBatcher::<NdArray<f32, i32, i8>>::default();
        let tetris_dist_batcher = TetrisDistBatcher::<NdArray<f32, i32, i8>>::default();

        let item = dataset.get(0).unwrap();
        let batch = tetris_batcher.batch(vec![item], &Default::default());
        assert_eq!(batch.current_boards.dims(), [1, BOARD_SIZE]);
        assert_eq!(batch.placements.dims(), [1, 1]);
        assert_eq!(batch.result_boards.dims(), [1, BOARD_SIZE]);

        let dist_batch = tetris_dist_batcher.batch(vec![item], &Default::default());
        assert_eq!(dist_batch.current_boards_dist.dims(), [1, BOARD_SIZE, 2]);
        assert_eq!(
            dist_batch.placements_dist.dims(),
            [1, TetrisPiecePlacement::NUM_PLACEMENTS]
        );
        assert_eq!(dist_batch.result_boards_dist.dims(), [1, BOARD_SIZE, 2]);
    }

    #[test]
    fn test_tetris_lost_datum() {
        let dataset =
            TetrisDataset::train(TetrisDatasetConfig::Uniform(TetrisDatasetUniformConfig {
                seed: 123,
                num_pieces_range: CopyRange { start: 0, end: 64 },
                length: 1_000_000,
            }));

        // check that we encounter a data point that is a lost state. But the board
        // and result board are not filled.
        let lost_item = dataset.iter().find(|item| {
            let is_lost = item.is_lost.into();
            let current_board = item.current_board;
            let result_board = item.result_board;
            is_lost && !current_board.is_full() && !result_board.is_full()
        });
        assert!(lost_item.is_some());

        // check that we encounter a data point that is a lost state. But only
        // the result board is filled.
        let lost_item = dataset.iter().find(|item| {
            let is_lost = item.is_lost.into();
            let current_board = item.current_board;
            let result_board = item.result_board;
            is_lost && !current_board.is_full() && result_board.is_full()
        });
        assert!(lost_item.is_some());

        // Also find a situation where the board and result board are filled.
        let filled_item = dataset.iter().find(|item| {
            let is_lost = item.is_lost.into();
            let current_board = item.current_board;
            let result_board = item.result_board;
            is_lost && current_board.is_full() && result_board.is_full()
        });
        assert!(filled_item.is_some());
    }

    #[test]
    fn test_tetris_sequence_dist_batcher_shapes() {
        let dataset = TetrisSequenceDataset::train(TetrisSequenceDatasetConfig::Uniform(
            TetrisSequenceDatasetUniformConfig {
                seed: 123,
                num_pieces_range: CopyRange { start: 0, end: 64 },
                length: 1_000_000,
                sequence_length: 10,
            },
        ));

        let datum = dataset.get(0).unwrap();

        let seq_dist_batcher = TetrisSequenceDistBatcher::<NdArray<f32, i32, i8>>::default();
        let seq_dist_batch = seq_dist_batcher.batch(vec![datum], &Default::default());

        assert_eq!(
            seq_dist_batch.current_boards_dist.dims(),
            [1, 10, BOARD_SIZE, 2]
        );
        assert_eq!(
            seq_dist_batch.placements_dist.dims(),
            [1, 10, TetrisPiecePlacement::NUM_PLACEMENTS]
        );
        assert_eq!(
            seq_dist_batch.result_boards_dist.dims(),
            [1, 10, BOARD_SIZE, 2]
        );

        let seq_dist_batch = seq_dist_batch.iter_seq().next().unwrap();
        assert_eq!(
            seq_dist_batch.current_boards_dist.dims(),
            [1, BOARD_SIZE, 2]
        );
        assert_eq!(
            seq_dist_batch.placements_dist.dims(),
            [1, TetrisPiecePlacement::NUM_PLACEMENTS]
        );
        assert_eq!(seq_dist_batch.result_boards_dist.dims(), [1, BOARD_SIZE, 2]);
    }

    #[test]
    fn test_tetris_init_dist_batcher_shapes() {
        let dataset = TetrisInitDistDataset::train(TetrisDatasetConfig::Uniform(
            TetrisDatasetUniformConfig {
                seed: 123,
                num_pieces_range: CopyRange { start: 0, end: 64 },
                length: 1_000_000,
            },
        ));

        let init_dist_batcher = TetrisInitDistBatcher::<NdArray<f32, i32, i8>>::default();
        let init_dist_batch =
            init_dist_batcher.batch(vec![dataset.get(0).unwrap()], &Default::default());

        assert_eq!(init_dist_batch.current_boards.dims(), [1, BOARD_SIZE, 2]);

        let test_batch_size = 8;
        let test_items = (0..test_batch_size)
            .map(|_| dataset.get(0).unwrap())
            .collect();
        let init_dist_batch = init_dist_batcher.batch(test_items, &Default::default());

        assert_eq!(
            init_dist_batch.current_boards.dims(),
            [test_batch_size, BOARD_SIZE, 2]
        );
    }
}
