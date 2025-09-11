use std::{ops::Deref, range::Range};

use crate::{
    ops::kl_div,
    tetris::{
        NUM_TETRIS_CELL_STATES, TetrisBoardRaw, TetrisGame, TetrisGameSet, TetrisPiece,
        TetrisPieceOrientation, TetrisPiecePlacement,
    },
};
use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Shape, Tensor};
use candle_nn::{encoding::one_hot, ops::softmax};
use rand::{
    Rng,
    distr::{Distribution, Uniform},
    seq::IndexedRandom,
};

/// Helper function to assert a 3D tensor for tetris board distributions/logits
/// Validates shape (_, TetrisBoardRaw::SIZE, NUM_TETRIS_CELL_STATES) and F32 dtype
fn assert_tetris_board_3d_tensor(tensor: &Tensor) {
    let (_batch_size, board_size, num_states) = tensor.dims3().expect("Tensor must be 3D");
    assert!(
        board_size == TetrisBoardRaw::SIZE,
        "Tensor must have {} board size, got {}",
        TetrisBoardRaw::SIZE,
        board_size
    );
    assert!(
        num_states == NUM_TETRIS_CELL_STATES,
        "Tensor must have {} states, got {}",
        NUM_TETRIS_CELL_STATES,
        num_states
    );
    assert!(
        tensor.dtype() == DType::F32,
        "Tensor must be f32, got {:?}",
        tensor.dtype()
    );
}

pub struct TetrisTransition {
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
        let u = Uniform::new(num_piece_range.start, num_piece_range.end)?;
        let mut games: Vec<TetrisGame> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut game = TetrisGame::new_with_seed(base_seed + i as u64);
            let num_pieces = u.sample(rng);
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

        let init_gameset = gameset.clone();

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
}

/// Tetris boards tensor
/// Tensor of shape (num_games, TetrisBoardRaw::SIZE)
///
/// Represents a "set" of tetris games
#[derive(Debug, Clone)]
pub struct TetrisBoardsTensor(Tensor);

impl TetrisBoardsTensor {
    /// Create a tetris boards tensor from a gameset
    pub fn from_gameset(games: TetrisGameSet, device: &Device) -> Result<Self> {
        let shape = Shape::from_dims(&[games.len(), TetrisBoardRaw::SIZE]);
        let boards = games
            .boards()
            .map(|board| board.to_binary_slice())
            .to_slice()
            .iter()
            .flat_map(|board| board.iter())
            .map(|&b| b as u32)
            .collect::<Vec<_>>();
        let boards = Tensor::from_vec(boards, &shape, device)?.to_dtype(DType::U32)?;
        Ok(Self(boards))
    }

    /// Create a tetris boards tensor from a slice of raw boards
    pub fn from_boards(boards: &[TetrisBoardRaw], device: &Device) -> Result<Self> {
        let shape = Shape::from_dims(&[boards.len(), TetrisBoardRaw::SIZE]);
        let mut flattened: Vec<u32> = Vec::with_capacity(boards.len() * TetrisBoardRaw::SIZE);
        for board in boards.iter() {
            flattened.extend(board.to_binary_slice().iter().map(|&v| v as u32));
        }
        let boards = Tensor::from_vec(flattened, &shape, device)?.to_dtype(DType::U32)?;
        Ok(Self(boards))
    }

    /// Convert tetris board tensor into tetris boards
    pub fn into_boards(self) -> Result<Vec<TetrisBoardRaw>> {
        let (batch_size, _board_size) = self.dims2()?;
        let boards = {
            let mut boards = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let board = self
                    .0
                    .i((i, ..))?
                    .flatten_all()?
                    .to_dtype(DType::U8)?
                    .to_vec1::<u8>()?;
                boards.push(TetrisBoardRaw::from_binary_slice(board.try_into().unwrap()));
            }
            boards
        };
        Ok(boards)
    }

    /// Count the number of boards that are equal between two tetris boards tensors
    /// Returns a tensor of shape (1)
    pub fn num_boards_equal(&self, other: &Self) -> Result<Tensor> {
        let (batch_size, board_size) = self.dims2()?;
        let (other_batch_size, other_board_size) = other.dims2()?;
        assert!(
            batch_size == other_batch_size,
            "Batch sizes must match, got {} and {}",
            batch_size,
            other_batch_size
        );
        assert!(
            board_size == other_board_size,
            "Board sizes must match, got {} and {}",
            board_size,
            other_board_size
        );

        // First check each batch * cell wise are equal
        // then sum how many cells are equal
        // then check that "board_size" cells are equal
        // then sum how many boards are equal
        let equal = self.0.eq(&other.0)?.to_dtype(DType::U32)?;
        let num_cells_equal = equal.sum(D::Minus1)?;
        let boards_equal = num_cells_equal
            .eq(&Tensor::full(
                board_size as u32,
                num_cells_equal.shape(),
                equal.device(),
            )?)?
            .to_dtype(DType::U32)?;
        let num_boards_equal = boards_equal.sum_all()?;
        Ok(num_boards_equal)
    }

    /// Count the number of cells that are equal between two tetris boards tensors
    /// Returns a tensor of shape (1)
    pub fn num_cells_equal(&self, other: &Self) -> Result<Tensor> {
        let equal = self.0.eq(&other.0)?.to_dtype(DType::U32)?;
        let num_cells_equal = equal.sum_all()?;
        Ok(num_cells_equal)
    }
}

impl Into<Tensor> for TetrisBoardsTensor {
    fn into(self) -> Tensor {
        self.0
    }
}

impl Deref for TetrisBoardsTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Tensor> for TetrisBoardsTensor {
    fn from(tensor: Tensor) -> Self {
        let (_batch_size, board_size) = tensor.dims2().expect("Tensor must be 2D");
        assert!(
            board_size == TetrisBoardRaw::SIZE,
            "Tensor must have {} board size, got {}",
            TetrisBoardRaw::SIZE,
            board_size
        );
        Self(tensor)
    }
}

/// Tetris boards logits tensor
/// Tensor of shape (num_games, TetrisBoardRaw::SIZE, 2)
///
/// This tensor represents the logits of a board across the 2 possible states:
/// - 0: The board is empty
/// - 1: The board is not empty
///
/// This tensor does not sum to 1 along the last dimension (raw logits)
#[derive(Debug, Clone)]
pub struct TetrisBoardLogitsTensor(Tensor);

impl TetrisBoardLogitsTensor {
    /// Convert the logits tensor to a distribution tensor by applying softmax along the last dimension
    /// Returns a tetris boards distribution tensor of shape (num_games, TetrisBoardRaw::SIZE, 2)
    pub fn into_dist(&self) -> Result<TetrisBoardsDistTensor> {
        let dist = softmax(&self.0, D::Minus1)?;
        Ok(TetrisBoardsDistTensor::from(dist))
    }
}

impl TryFrom<TetrisBoardsTensor> for TetrisBoardLogitsTensor {
    type Error = candle_core::Error;

    fn try_from(tensor: TetrisBoardsTensor) -> Result<Self, Self::Error> {
        // Tensor info and unwrapping
        let tensor = tensor.0.to_dtype(DType::F32)?;
        let (num_games, board_size) = tensor.shape().dims2().expect("Base tensor must be 2D");

        let a = tensor.reshape(&[num_games, board_size, 1])?;
        let b = (1.0 - a.clone())?;
        let logits = Tensor::cat(&[&a, &b], 2)?;
        Ok(Self(logits))
    }
}

impl Into<Tensor> for TetrisBoardLogitsTensor {
    fn into(self) -> Tensor {
        self.0
    }
}

impl Deref for TetrisBoardLogitsTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Tensor> for TetrisBoardLogitsTensor {
    fn from(tensor: Tensor) -> Self {
        assert_tetris_board_3d_tensor(&tensor);
        Self(tensor)
    }
}

/// Tetris boards distribution tensor
/// Tensor of shape (num_games, TetrisBoardRaw::SIZE, 2)
///
/// This tensor represents the distribution of a board across the 2 possible states:
/// - 0: The board is empty
/// - 1: The board is not empty
///
/// This tensor also sums to 1 along the last dimension
#[derive(Debug, Clone)]
pub struct TetrisBoardsDistTensor(Tensor);

impl TetrisBoardsDistTensor {
    /// Add random noise to the distribution tensor along the last dimension
    /// The noise is sampled from a uniform distribution between -epsilon and epsilon
    /// The resulting distribution is normalized to sum to 1 along the last dimension
    pub fn noise(&self, epsilon: f32) -> Result<Self> {
        let (batch, size, states) = self.dims3()?;
        let noise = Tensor::rand(-epsilon, epsilon, &[batch, size, states], self.device())?;
        let noised = (&self.0 + &noise)?;
        // Ensure values stay positive
        let noised =
            noised.maximum(&Tensor::zeros(noised.shape(), DType::F32, noised.device())?)?;
        // Normalize along last dim
        let sums = noised.sum(D::Minus1)?.reshape(&[batch, size, 1])?;
        let normalized = (noised / sums)?;
        Ok(Self(normalized))
    }

    /// Convert a tetris boards distribution tensor to a tetris boards tensor
    /// by taking the argmax of the last dimension
    /// Returns a tetris boards tensor of shape (num_games, TetrisBoardRaw::SIZE)
    pub fn argmax(&self) -> Result<TetrisBoardsTensor> {
        let argmax = self.0.argmax(D::Minus1)?;
        Ok(TetrisBoardsTensor::from(argmax))
    }

    /// Compares two tetris boards distribution tensors
    /// Returns a tensor of shape (batch_size)
    pub fn similarity(&self, other: &Self) -> Result<Tensor> {
        let (batch_size, _board_size, _states) = self.dims3()?;
        let (other_batch_size, _other_board_size, _other_states) = other.dims3()?;
        assert!(
            batch_size == other_batch_size,
            "Batch sizes must match, got {} and {}",
            batch_size,
            other_batch_size
        );

        let kl = kl_div(&self.0, &other.0, D::Minus1)?;
        Ok(kl)
    }

    // Calculate the entropy of the distribution for each board
    pub fn entropy(&self) -> Result<Tensor> {
        let eps = Tensor::new(1e-12f32, self.0.device())?.broadcast_as(self.0.shape().dims())?;
        let p = self.0.maximum(&eps)?;

        let plogp = p.log()?.mul(&p)?; // [B, SIZE, 2]
        let h_per_cell = plogp.sum(D::Minus1)?; // [B, SIZE]
        let h_per_game = h_per_cell.sum(D::Minus1)?; // [B]
        Ok(h_per_game.neg()?)
    }
}

impl TryFrom<TetrisBoardsTensor> for TetrisBoardsDistTensor {
    type Error = candle_core::Error;

    fn try_from(tensor: TetrisBoardsTensor) -> Result<Self, Self::Error> {
        // Tensor info and unwrapping
        let tensor = tensor.0.to_dtype(DType::F32)?;
        let (num_games, board_size) = tensor.shape().dims2().expect("Base tensor must be 2D");

        let a = tensor.reshape(&[num_games, board_size, 1])?;
        let b = (1.0 - a.clone())?;
        let dists = Tensor::cat(&[&a, &b], 2)?;
        Ok(Self(dists))
    }
}

impl Into<Tensor> for TetrisBoardsDistTensor {
    fn into(self) -> Tensor {
        self.0
    }
}

impl Deref for TetrisBoardsDistTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Tensor> for TetrisBoardsDistTensor {
    fn from(tensor: Tensor) -> Self {
        assert_tetris_board_3d_tensor(&tensor);
        Self(tensor)
    }
}

/// Tetris piece placement tensor
/// Tensor of shape (batch_size, 1)
///
/// Represents the index of a piece placement
#[derive(Debug, Clone)]
pub struct TetrisPiecePlacementTensor(Tensor);

impl TetrisPiecePlacementTensor {
    /// Convert a list of placements to a tensor
    /// using the index of the placement    dyn_tanh: DynamicTanh,

    pub fn from_placements(placements: &[TetrisPiecePlacement], device: &Device) -> Result<Self> {
        let shape = Shape::from_dims(&[placements.len(), 1]);
        let placement_indices: Vec<u8> = placements.iter().map(|p| p.index()).collect();
        let tensor = Tensor::from_vec(placement_indices, &shape, device)?;
        Ok(Self(tensor))
    }

    /// Convert a tensor of placement indices to a list of placements
    pub fn into_placements(self) -> Result<Vec<TetrisPiecePlacement>> {
        let placement_ints = self.0.flatten_all()?.to_dtype(DType::U8)?.to_vec1::<u8>()?;
        Ok(placement_ints
            .into_iter()
            .map(TetrisPiecePlacement::from_index)
            .collect())
    }
}

impl Into<Tensor> for TetrisPiecePlacementTensor {
    fn into(self) -> Tensor {
        self.0
    }
}

impl Deref for TetrisPiecePlacementTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Tensor> for TetrisPiecePlacementTensor {
    fn from(tensor: Tensor) -> Self {
        let (_batch_size, placement_dim) = tensor.dims2().expect("Tensor must be 2D");
        assert!(
            placement_dim == 1,
            "Tensor must have 1 placement dim, got {}",
            placement_dim
        );
        assert!(
            tensor.dtype() == DType::U8,
            "Tensor must be u8, got {:?}",
            tensor.dtype()
        );
        Self(tensor)
    }
}

/// Tetris piece placement distribution tensor
/// Tensor of shape (batch_size, TetrisPiecePlacement::NUM_PLACEMENTS)
///
/// Represents the distribution of a piece placement across all possible placements
#[repr(transparent)]
pub struct TetrisPiecePlacementDistTensor(Tensor);

impl TetrisPiecePlacementDistTensor {
    pub fn into_placements_argmax(self) -> Result<Vec<TetrisPiecePlacement>> {
        let argmax = self.0.argmax(D::Minus1)?;
        let placements = argmax.to_vec1::<u8>()?;
        Ok(placements
            .into_iter()
            .map(TetrisPiecePlacement::from_index)
            .collect())
    }
}

impl TryFrom<TetrisPiecePlacementTensor> for TetrisPiecePlacementDistTensor {
    type Error = candle_core::Error;

    fn try_from(tensor: TetrisPiecePlacementTensor) -> Result<Self, Self::Error> {
        let dists = one_hot(tensor.0, TetrisPiecePlacement::NUM_PLACEMENTS, 1f32, 0f32)?
            .to_dtype(DType::F32)?;
        Ok(Self(dists))
    }
}

impl Into<Tensor> for TetrisPiecePlacementDistTensor {
    fn into(self) -> Tensor {
        self.0
    }
}

impl Deref for TetrisPiecePlacementDistTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Tensor> for TetrisPiecePlacementDistTensor {
    fn from(tensor: Tensor) -> Self {
        let (_batch_size, placement_dim) = tensor.dims2().expect("Tensor must be 2D");
        assert!(
            placement_dim == TetrisPiecePlacement::NUM_PLACEMENTS,
            "Tensor must have {} placement dim, got {}",
            TetrisPiecePlacement::NUM_PLACEMENTS,
            placement_dim
        );
        assert!(
            tensor.dtype() == DType::F32,
            "Tensor must be f32, got {:?}",
            tensor.dtype()
        );
        Self(tensor)
    }
}

/// Tetris piece orientation tensor
/// Tensor of shape (batch_size, 1)
///
/// Represents the index of a piece orientation
#[derive(Debug, Clone)]
pub struct TetrisPieceOrientationTensor(Tensor);

impl TetrisPieceOrientationTensor {
    /// Convert a list of orientations to a tensor
    /// using the index of the orientation
    pub fn from_orientations(
        orientations: &[TetrisPieceOrientation],
        device: &Device,
    ) -> Result<Self> {
        let shape = Shape::from_dims(&[orientations.len(), 1]);
        let orientation_indices: Vec<u8> = orientations.iter().map(|o| o.index()).collect();
        let tensor = Tensor::from_vec(orientation_indices, &shape, device)?;
        Ok(Self(tensor))
    }

    /// Convert a tensor of orientation indices to a list of orientations
    pub fn into_orientations(self) -> Result<Vec<TetrisPieceOrientation>> {
        let orientation_ints = self.0.flatten_all()?.to_vec1::<u8>()?;
        Ok(orientation_ints
            .into_iter()
            .map(TetrisPieceOrientation::from_index)
            .collect())
    }
}

impl Into<Tensor> for TetrisPieceOrientationTensor {
    fn into(self) -> Tensor {
        self.0
    }
}

impl Deref for TetrisPieceOrientationTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Tensor> for TetrisPieceOrientationTensor {
    fn from(tensor: Tensor) -> Self {
        let (_batch_size, orientation_dim) = tensor.dims2().expect("Tensor must be 2D");
        assert!(
            orientation_dim == 1,
            "Tensor must have 1 orientation dim, got {}",
            orientation_dim
        );
        assert!(
            tensor.dtype() == DType::U8,
            "Tensor must be u8, got {:?}",
            tensor.dtype()
        );
        Self(tensor)
    }
}

/// Tetris piece orientation logits tensor
/// Tensor of shape (batch_size, num_orientations)
///
/// Represents the logits of a piece orientation
#[derive(Debug, Clone)]
pub struct TetrisPieceOrientationLogitsTensor(Tensor);

impl Into<Tensor> for TetrisPieceOrientationLogitsTensor {
    fn into(self) -> Tensor {
        self.0
    }
}

impl Deref for TetrisPieceOrientationLogitsTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Tensor> for TetrisPieceOrientationLogitsTensor {
    fn from(tensor: Tensor) -> Self {
        let (_batch_size, orientation_dim) = tensor.dims2().expect("Tensor must be 2D");
        assert!(
            orientation_dim == TetrisPieceOrientation::NUM_ORIENTATIONS,
            "Tensor must have {} orientation dim, got {}",
            TetrisPieceOrientation::NUM_ORIENTATIONS,
            orientation_dim
        );
        Self(tensor)
    }
}

/// Tetris piece orientation tensor
/// Tensor of shape (batch_size, num_orientations)
///
/// Represents the index of a piece orientation
#[derive(Debug, Clone)]
pub struct TetrisPieceOrientationDistTensor(Tensor);

impl TetrisPieceOrientationDistTensor {
    /// Convert a list of orientations to a tensor
    /// using the index of the orientation
    pub fn from_orientations(
        orientations: &[TetrisPieceOrientation],
        device: &Device,
    ) -> Result<Self> {
        let base_tensor = TetrisPieceOrientationTensor::from_orientations(orientations, device)?;
        let self_tensor = Self::try_from(base_tensor)?;
        Ok(self_tensor)
    }

    /// Convert a tensor of distribution of orientations to a list of orientations
    pub fn into_orientations_argmax(self) -> Result<Vec<TetrisPieceOrientation>> {
        let argmax = self.0.argmax(D::Minus1)?;
        let orientations = argmax.to_vec1::<u8>()?;
        Ok(orientations
            .into_iter()
            .map(TetrisPieceOrientation::from_index)
            .collect())
    }
}

impl TryFrom<TetrisPieceOrientationTensor> for TetrisPieceOrientationDistTensor {
    type Error = candle_core::Error;

    fn try_from(tensor: TetrisPieceOrientationTensor) -> Result<Self, Self::Error> {
        let dists = one_hot(
            tensor.0,
            TetrisPieceOrientation::NUM_ORIENTATIONS,
            1f32,
            0f32,
        )?
        .to_dtype(DType::F32)?;
        Ok(Self(dists))
    }
}

impl Into<Tensor> for TetrisPieceOrientationDistTensor {
    fn into(self) -> Tensor {
        self.0
    }
}

impl Deref for TetrisPieceOrientationDistTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Tensor> for TetrisPieceOrientationDistTensor {
    fn from(tensor: Tensor) -> Self {
        let (_batch_size, orientation_dim) = tensor.dims2().expect("Tensor must be 2D");
        assert!(
            orientation_dim == TetrisPieceOrientation::NUM_ORIENTATIONS,
            "Tensor must have {} orientation dim, got {}",
            TetrisPieceOrientation::NUM_ORIENTATIONS,
            orientation_dim
        );
        assert!(
            tensor.dtype() == DType::F32,
            "Tensor must be f32, got {:?}",
            tensor.dtype()
        );
        Self(tensor)
    }
}

/// Tetris Piece tensor
/// Tensor of shape (batch_size, 1)
///
/// Represents the index of a piece placement
#[derive(Debug, Clone)]
pub struct TetrisPieceTensor(Tensor);

impl TetrisPieceTensor {
    pub fn from_pieces(pieces: &[TetrisPiece], device: &Device) -> Result<Self> {
        let shape = Shape::from_dims(&[pieces.len(), 1]);
        let pieces = pieces.iter().map(|&p| p.index() as u32).collect::<Vec<_>>();
        let pieces = Tensor::from_vec(pieces, &shape, device)?.to_dtype(DType::U32)?;
        Ok(Self(pieces))
    }

    /// Convert a tensor of distribution of pieces to a list of pieces
    pub fn into_pieces(self) -> Result<Vec<TetrisPiece>> {
        let pieces = self.0.flatten_all()?.to_vec1::<u8>()?;
        Ok(pieces.into_iter().map(TetrisPiece::from_index).collect())
    }
}

impl Into<Tensor> for TetrisPieceTensor {
    fn into(self) -> Tensor {
        self.0
    }
}

impl Deref for TetrisPieceTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl TryFrom<Tensor> for TetrisPieceTensor {
    type Error = candle_core::Error;

    fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
        let (_batch_size, piece_dim) = tensor.dims2().expect("Tensor must be 2D");
        assert!(
            piece_dim == 1,
            "Tensor must have 1 piece dim, got {}",
            piece_dim
        );
        assert!(
            tensor.dtype() == DType::U32,
            "Tensor must be u32, got {:?}",
            tensor.dtype()
        );
        Ok(Self(tensor))
    }
}

/// Tetris context tensor
/// Tensor of shape (batch_size, seq_len, dim)
///
/// Represents the context of a sequence of tokens
#[derive(Debug, Clone)]
pub struct TetrisContextTensor(Tensor);

impl TetrisContextTensor {
    pub fn push_tokens(&self, tokens: Tensor) -> Result<Self> {
        // cat([B, S_A, D], [B, S_B, D]) -> [B, S_A+S_B, D]
        let new_tensor = Tensor::cat(&[self.deref(), &tokens], 1)?;
        Ok(Self(new_tensor))
    }

    pub fn swap_goal_token(&self, new_goal_token: Tensor) -> Result<Self> {
        // self: [B, S, D] where S >= 2; new_goal_token: [B, 1, D]
        let (batch_size, seq_len, dim) = self.dims3()?;
        let (ng_batch, ng_seq, ng_dim) = new_goal_token.dims3()?;
        assert!(
            ng_seq == 1,
            "new_goal_token must have sequence length 1, got {}",
            ng_seq
        );
        assert!(
            ng_batch == batch_size,
            "Batch sizes must match, got {} and {}",
            ng_batch,
            batch_size
        );
        assert!(
            ng_dim == dim,
            "Embedding dims must match, got {} and {}",
            ng_dim,
            dim
        );

        // Ensure device match
        let new_goal_token = if !std::ptr::eq(new_goal_token.device(), self.device()) {
            new_goal_token.to_device(self.device())?
        } else {
            new_goal_token
        };

        // Replace [:, 0, :] by building [new_goal_token, self[:, 1:, :]] along dim=1
        let tail = self.narrow(1, 1, seq_len - 1)?; // [B, S-1, D]
        let new_tensor = Tensor::cat(&[&new_goal_token, &tail], 1)?;
        Ok(Self(new_tensor))
    }
}

impl Into<Tensor> for TetrisContextTensor {
    fn into(self) -> Tensor {
        self.0
    }
}

impl Deref for TetrisContextTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Tensor> for TetrisContextTensor {
    fn from(tensor: Tensor) -> Self {
        let (_batch_size, seq_len, _dim) = tensor.dims3().expect("Tensor must be 3D");
        assert!(
            seq_len >= 2,
            "Tensor must have >=2 sequence length, got {}",
            seq_len
        );
        Self(tensor)
    }
}

mod test {
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
            let result_i: Vec<u32> = sequence.result_boards[i]
                .flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            let current_next: Vec<u32> = sequence.current_boards[i + 1]
                .flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();

            assert_eq!(result_i, current_next, "Mismatch at step {}", i);
        }
    }

    /// Test that the orientations are correctly converted to and from tensors
    #[test]
    fn test_into_from_orientations_tensor() {
        let batch_size = 16;
        let generator = TetrisDatasetGenerator::new();
        let transition = generator
            .gen_uniform_sampled_transition(
                (0..10).into(),
                batch_size,
                &Device::Cpu,
                &mut rand::rng(),
            )
            .unwrap();

        let original_orientations_tensor = transition.orientation;
        let original_orientations = original_orientations_tensor
            .clone()
            .into_orientations()
            .unwrap();

        let recovered_orientations_tensor =
            TetrisPieceOrientationTensor::from_orientations(&original_orientations, &Device::Cpu)
                .unwrap();
        let recovered_orientations = recovered_orientations_tensor
            .clone()
            .into_orientations()
            .unwrap();

        // Check original / recovered tensor equality
        let eq_result = original_orientations_tensor
            .0
            .eq(&recovered_orientations_tensor.0)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();
        assert_eq!(eq_result, batch_size as u32);

        // Check original / recovered orientations equality
        assert_eq!(original_orientations, recovered_orientations);
    }

    /// Test that the placements are correctly converted to and from tensors
    #[test]
    fn test_into_from_placements_tensor() {
        let batch_size = 16;
        let generator = TetrisDatasetGenerator::new();
        let transition = generator
            .gen_uniform_sampled_transition(
                (0..10).into(),
                batch_size,
                &Device::Cpu,
                &mut rand::rng(),
            )
            .unwrap();

        let original_placements_tensor = transition.placement;
        let original_placements = original_placements_tensor
            .clone()
            .into_placements()
            .unwrap();

        let recovered_placements_tensor =
            TetrisPiecePlacementTensor::from_placements(&original_placements, &Device::Cpu)
                .unwrap();
        let recovered_placements = recovered_placements_tensor
            .clone()
            .into_placements()
            .unwrap();

        // Check original / recovered tensor equality
        let eq_result = original_placements_tensor
            .0
            .eq(&recovered_placements_tensor.0)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();
        assert_eq!(eq_result, batch_size as u32);

        // Check original / recovered placements equality
        assert_eq!(original_placements, recovered_placements);
    }

    /// Test that the boards are correctly counted as equal
    #[test]
    fn test_count_equal() {
        let generator = TetrisDatasetGenerator::new();
        let transition = generator
            .gen_uniform_sampled_transition((0..10).into(), 1, &Device::Cpu, &mut rand::rng())
            .unwrap();

        let result_board = transition.result_board;

        let equal = result_board.num_boards_equal(&result_board).unwrap();
        assert_eq!(equal.to_scalar::<u32>().unwrap(), 1);
    }

    /// Test that the boards are correctly converted to and from tensors
    #[test]
    fn test_into_from_boards_tensor() {
        let batch_size = 16;
        let generator = TetrisDatasetGenerator::new();
        let transition = generator
            .gen_uniform_sampled_transition(
                (0..10).into(),
                batch_size,
                &Device::Cpu,
                &mut rand::rng(),
            )
            .unwrap();
        let original_boards_tensor = transition.result_board;
        let original_boards = original_boards_tensor.clone().into_boards().unwrap();

        let recovered_boards_tensor =
            TetrisBoardsTensor::from_boards(&original_boards, &Device::Cpu).unwrap();
        let recovered_boards = recovered_boards_tensor.clone().into_boards().unwrap();

        // Check original / recovered boards equality
        assert_eq!(original_boards, recovered_boards);
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
