use std::range::Range;
use std::thread;

use crate::{
    tensors::{TetrisBoardsTensor, TetrisPieceOrientationTensor, TetrisPiecePlacementTensor},
    tetris::{TetrisGame, TetrisGameSet, TetrisPiece},
};
use anyhow::Result;
use candle_core::Device;
use crossbeam::channel::{Receiver, bounded};
use rand::{
    Rng,
    distr::{Distribution, Uniform},
    seq::IndexedRandom,
};

pub struct TetrisTransition {
    pub current_gameset: TetrisGameSet,

    pub current_board: TetrisBoardsTensor,
    pub result_board: TetrisBoardsTensor,

    pub placement: TetrisPiecePlacementTensor,
    pub orientation: TetrisPieceOrientationTensor,

    pub piece: Vec<TetrisPiece>,
}

pub struct TetrisTransitionSequence {
    pub init_gameset: TetrisGameSet,

    pub current_boards: Vec<TetrisBoardsTensor>,
    pub result_boards: Vec<TetrisBoardsTensor>,

    pub placements: Vec<TetrisPiecePlacementTensor>,
    pub orientations: Vec<TetrisPieceOrientationTensor>,
    pub pieces: Vec<Vec<TetrisPiece>>,
}

pub struct TetrisDatasetGenerator {}

impl TetrisDatasetGenerator {
    pub fn new() -> Self {
        Self {}
    }

    pub fn gen_uniform_sampled_gameset<R: Rng>(
        &self,
        num_piece_range: Range<usize>,
        batch_size: usize,
        rng: &mut R,
    ) -> Result<TetrisGameSet> {
        let base_seed = rng.next_u64();

        // Seed each game with its own random number of placements
        let distribution = Uniform::new(num_piece_range.start, num_piece_range.end)?;
        let mut games: Vec<TetrisGame> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut game = TetrisGame::new_with_seed(base_seed + i as u64);
            let num_pieces = distribution.sample(rng);
            for _ in 0..num_pieces {
                let placement = *game.current_placements().choose(rng).unwrap();
                game.apply_placement(placement);
            }
            games.push(game);
        }

        let gameset = TetrisGameSet::from_games(&games);
        Ok(gameset)
    }

    pub fn gen_uniform_sampled_transition<R: Rng>(
        &self,
        num_piece_range: Range<usize>,
        batch_size: usize,
        device: &Device,
        rng: &mut R,
    ) -> Result<TetrisTransition> {
        let mut gameset = self.gen_uniform_sampled_gameset(num_piece_range, batch_size, rng)?;

        // Now we generate a transition
        let current_board = TetrisBoardsTensor::from_gameset(gameset, device)?;
        let (placement, orientation, piece) = {
            let placements = gameset
                .current_placements()
                .iter()
                .map(|&pls| *pls.choose(rng).unwrap())
                .collect::<Box<[_]>>();
            let pieces = placements.iter().map(|&p| p.piece).collect::<Vec<_>>();
            gameset.apply_placement(&placements);
            (
                TetrisPiecePlacementTensor::from_placements(&placements, device)
                    .expect("Failed to create placement tensor"),
                TetrisPieceOrientationTensor::from_orientations(
                    &placements
                        .iter()
                        .map(|&p| p.orientation)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    device,
                )
                .expect("Failed to create orientation tensor"),
                pieces,
            )
        };
        let result_board = TetrisBoardsTensor::from_gameset(gameset, device)?;
        Ok(TetrisTransition {
            current_gameset: gameset,
            current_board,
            placement,
            orientation,
            piece,
            result_board,
        })
    }

    pub fn gen_sequence<R: Rng>(
        &self,
        num_piece_range: Range<usize>,
        batch_size: usize,
        sequence_length: usize,
        device: &Device,
        rng: &mut R,
    ) -> Result<TetrisTransitionSequence> {
        let mut gameset = self.gen_uniform_sampled_gameset(num_piece_range, batch_size, rng)?;

        let init_gameset = gameset;

        let mut current_boards: Vec<TetrisBoardsTensor> = Vec::new();
        let mut placements: Vec<TetrisPiecePlacementTensor> = Vec::new();
        let mut orientations: Vec<TetrisPieceOrientationTensor> = Vec::new();
        let mut result_boards: Vec<TetrisBoardsTensor> = Vec::new();
        let mut pieces: Vec<Vec<TetrisPiece>> = Vec::new();

        for _ in 0..sequence_length {
            let current_board = TetrisBoardsTensor::from_gameset(gameset, device)?;
            current_boards.push(current_board);

            let (placement, orientation) = {
                let placements = gameset
                    .current_placements()
                    .iter()
                    .map(|&pls| *pls.choose(rng).unwrap())
                    .collect::<Box<[_]>>();
                pieces.push(placements.iter().map(|&p| p.piece).collect::<Vec<_>>());
                gameset.apply_placement(&placements);
                let placement = TetrisPiecePlacementTensor::from_placements(&placements, device)
                    .expect("Failed to create placement tensor");
                let orientation = TetrisPieceOrientationTensor::from_orientations(
                    &placements
                        .iter()
                        .map(|&p| p.orientation)
                        .collect::<Vec<_>>(),
                    device,
                )
                .expect("Failed to create orientation tensor");
                (placement, orientation)
            };
            placements.push(placement);
            orientations.push(orientation);

            let result_board = TetrisBoardsTensor::from_gameset(gameset, device)?;
            result_boards.push(result_board);
        }

        Ok(TetrisTransitionSequence {
            init_gameset,
            current_boards,
            placements,
            orientations,
            result_boards,
            pieces,
        })
    }

    /// Spawn a background worker that continuously generates `TetrisTransition` items
    /// and pushes them into a bounded channel. The worker runs until the returned
    /// receiver is dropped. The channel capacity provides backpressure.
    pub fn spawn_transition_channel(
        &self,
        num_piece_range: Range<usize>,
        batch_size: usize,
        device: Device,
        buffer_capacity: usize,
    ) -> Receiver<TetrisTransition> {
        let (tx, rx) = bounded::<TetrisTransition>(buffer_capacity);

        let _ = thread::Builder::new()
            .name("tetris-trans-pref".to_string())
            .spawn(move || {
                let generator = TetrisDatasetGenerator::new();
                let mut rng = rand::rng();

                loop {
                    let item = generator
                        .gen_uniform_sampled_transition(
                            num_piece_range.clone(),
                            batch_size,
                            &device,
                            &mut rng,
                        )
                        .expect("Failed to generate TetrisTransition");

                    // If the receiver is dropped, exit the worker.
                    if tx.send(item).is_err() {
                        break;
                    }
                }
            })
            .expect("Failed to spawn tetris-trans-pref thread");

        rx
    }

    /// Spawn a background worker that continuously generates `TetrisTransitionSequence`
    /// items and pushes them into a bounded channel. The worker runs until the returned
    /// receiver is dropped. The channel capacity provides backpressure.
    pub fn spawn_sequence_channel(
        &self,
        num_piece_range: Range<usize>,
        batch_size: usize,
        sequence_length: usize,
        device: Device,
        buffer_capacity: usize,
    ) -> Receiver<TetrisTransitionSequence> {
        let (tx, rx) = bounded::<TetrisTransitionSequence>(buffer_capacity);

        let _ = thread::Builder::new()
            .name("tetris-seq-pref".to_string())
            .spawn(move || {
                let generator = TetrisDatasetGenerator::new();
                let mut rng = rand::rng();

                loop {
                    let item = generator
                        .gen_sequence(
                            num_piece_range.clone(),
                            batch_size,
                            sequence_length,
                            &device,
                            &mut rng,
                        )
                        .expect("Failed to generate TetrisTransitionSequence");

                    // If the receiver is dropped, exit the worker.
                    if tx.send(item).is_err() {
                        break;
                    }
                }
            })
            .expect("Failed to spawn tetris-seq-pref thread");

        rx
    }
}

mod test {
    #[allow(unused_imports)]
    use super::*;

    /// Test that the sequence of boards is chained correctly
    /// i.e. result_boards[i] == current_boards[i+1]
    #[test]
    fn test_sequence_boards_full_chain() {
        let generator = TetrisDatasetGenerator::new();
        let seq_len = 6;
        let sequence = generator
            .gen_sequence((0..10).into(), 1, seq_len, &Device::Cpu, &mut rand::rng())
            .unwrap();

        for i in 0..(seq_len - 1) {
            let result_i: Vec<u8> = sequence.result_boards[i]
                .flatten_all()
                .unwrap()
                .to_vec1::<u8>()
                .unwrap();
            let current_next: Vec<u8> = sequence.current_boards[i + 1]
                .flatten_all()
                .unwrap()
                .to_vec1::<u8>()
                .unwrap();

            assert_eq!(result_i, current_next, "Mismatch at step {}", i);
        }
    }

    // #[test]
    // fn test_test() {
    //     // create a full board
    //     let rng = rand::rng();

    //     let empty_board = TetrisBoardRaw::default();
    //     let full_board = *TetrisBoardRaw::default().fill_all();
    //     let rand_board = *TetrisBoardRaw::default().flip_random_bits(1_000, &mut rand::rng());

    //     // tetris board tensor
    //     let empty_tetris_board_tensor =
    //         TetrisBoardsTensor::from_boards(&[empty_board], Device::Cpu).unwrap();
    //     let empty_tetris_board_tensor_dist =
    //         TetrisBoardsDistTensor::try_from(empty_tetris_board_tensor).unwrap();

    //     let full_tetris_board_tensor =
    //         TetrisBoardsTensor::from_boards(&[full_board], Device::Cpu).unwrap();
    //     let full_tetris_board_tensor_dist =
    //         TetrisBoardsDistTensor::try_from(full_tetris_board_tensor).unwrap();

    //     let rand_tetris_board_tensor =
    //         TetrisBoardsTensor::from_boards(&[rand_board], Device::Cpu).unwrap();
    //     let rand_tetris_board_tensor_dist =
    //         TetrisBoardsDistTensor::try_from(rand_tetris_board_tensor).unwrap();

    //     let max_entropy_board_tensor_dist = TetrisBoardsDistTensor::try_from(
    //         Tensor::full(
    //             0.5f32,
    //             Shape::from_dims(&[1, TetrisBoardRaw::SIZE, NUM_TETRIS_CELL_STATES]),
    //             &Device::Cpu,
    //         )
    //         .unwrap(),
    //     )
    //     .unwrap();

    //     // check the dist is correct
    //     let dist_empty_full = empty_tetris_board_tensor_dist
    //         .similarity(&full_tetris_board_tensor_dist)
    //         .unwrap()
    //         .to_scalar::<f32>()
    //         .unwrap();
    //     println!("dist_empty_full: {}", dist_empty_full);

    //     let dist_empty_rand = empty_tetris_board_tensor_dist
    //         .similarity(&rand_tetris_board_tensor_dist)
    //         .unwrap()
    //         .to_scalar::<f32>()
    //         .unwrap();
    //     println!("dist_empty_rand: {}", dist_empty_rand);

    //     let dist_full_rand = full_tetris_board_tensor_dist
    //         .similarity(&rand_tetris_board_tensor_dist)
    //         .unwrap()
    //         .to_scalar::<f32>()
    //         .unwrap();
    //     println!("dist_full_rand: {}", dist_full_rand);

    //     let dist_max_entropy_rand = max_entropy_board_tensor_dist
    //         .similarity(&rand_tetris_board_tensor_dist)
    //         .unwrap()
    //         .to_scalar::<f32>()
    //         .unwrap();
    //     println!("dist_max_entropy_rand: {}", dist_max_entropy_rand);

    //     let dist_max_entropy_full = max_entropy_board_tensor_dist
    //         .similarity(&full_tetris_board_tensor_dist)
    //         .unwrap()
    //         .to_scalar::<f32>()
    //         .unwrap();
    //     println!("dist_max_entropy_full: {}", dist_max_entropy_full);

    //     let dist_max_entropy_empty = max_entropy_board_tensor_dist
    //         .similarity(&empty_tetris_board_tensor_dist)
    //         .unwrap()
    //         .to_scalar::<f32>()
    //         .unwrap();
    //     println!("dist_max_entropy_empty: {}", dist_max_entropy_empty);
    // }
}
