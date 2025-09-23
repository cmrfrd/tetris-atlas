use crate::{
    impl_wrapped_tensor,
    ops::kl_div,
    tetris::{
        NUM_TETRIS_CELL_STATES, TetrisBoardRaw, TetrisGameSet, TetrisPiece, TetrisPieceOrientation,
        TetrisPiecePlacement,
    },
    wrapped_tensor::{ShapeDim, WrappedTensor},
};
use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Shape, Tensor};
use candle_nn::{encoding::one_hot, ops::softmax};

/// Tetris boards tensor
/// Tensor of shape (num_games, TetrisBoardRaw::SIZE)
///
/// Represents a "set" of tetris games
#[derive(Debug, Clone)]
pub struct TetrisBoardsTensor(Tensor);

impl_wrapped_tensor!(
    TetrisBoardsTensor,
    dtype = DType::U8,
    shape_spec = (ShapeDim::Any, ShapeDim::Dim(TetrisBoardRaw::SIZE))
);

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
            .map(|&b| b)
            .collect::<Vec<_>>();
        let boards = Tensor::from_vec(boards, &shape, device)?;
        Ok(Self::try_from(boards)?)
    }

    /// Create a tetris boards tensor from a slice of raw boards
    pub fn from_boards(boards: &[TetrisBoardRaw], device: &Device) -> Result<Self> {
        let shape = Shape::from_dims(&[boards.len(), TetrisBoardRaw::SIZE]);
        let mut flattened: Vec<u8> = Vec::with_capacity(boards.len() * TetrisBoardRaw::SIZE);
        for board in boards.iter() {
            flattened.extend(board.to_binary_slice().iter());
        }
        let boards = Tensor::from_vec(flattened, &shape, device)?;
        Ok(Self::try_from(boards)?)
    }

    /// Convert tetris board tensor into tetris boards
    pub fn into_boards(self) -> Result<Vec<TetrisBoardRaw>> {
        let (batch_size, _board_size) = self.shape_tuple();
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

    pub fn into_dist(&self) -> Result<TetrisBoardsDistTensor> {
        let dist = one_hot(self.inner().clone(), NUM_TETRIS_CELL_STATES, 1f32, 0f32)?;
        Ok(TetrisBoardsDistTensor::try_from(dist)?)
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
        let equal = self.inner().eq(other.inner())?.to_dtype(DType::U32)?;
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
    pub fn perc_cells_equal(&self, other: &Self) -> Result<f32> {
        let (batch_size, _) = self.shape_tuple();
        let (other_batch_size, _) = other.shape_tuple();
        assert!(
            batch_size == other_batch_size,
            "Batch sizes must match, got {} and {}",
            batch_size,
            other_batch_size
        );
        let equal = self.inner().eq(other.inner())?.to_dtype(DType::U32)?;
        let num_cells_equal = equal.sum(D::Minus1)?.to_dtype(DType::F64)?;
        let perc_cells_equal = (num_cells_equal / ((TetrisBoardRaw::SIZE) as f64))?;
        let perc_equal = perc_cells_equal
            .to_dtype(DType::F32)?
            .mean([0])?
            .to_scalar::<f32>()?;
        Ok(perc_equal)
    }
}

/// Tetris boards logits tensor
/// Tensor of shape (num_games, TetrisBoardRaw::SIZE)
///
/// This tensor represents the logits of a board for binary classification:
/// - Positive values indicate the cell is likely occupied (state 1)
/// - Negative values indicate the cell is likely empty (state 0)
/// - Zero is the decision boundary
///
/// This tensor contains raw logits (not probabilities)
#[derive(Debug, Clone)]
pub struct TetrisBoardLogitsTensor(Tensor);

impl_wrapped_tensor!(
    TetrisBoardLogitsTensor,
    dtype = DType::F32,
    shape_spec = (ShapeDim::Any, ShapeDim::Dim(TetrisBoardRaw::SIZE))
);

impl TetrisBoardLogitsTensor {
    /// Convert the logits tensor to a distribution tensor by applying sigmoid to get probabilities
    /// Returns a tetris boards distribution tensor of shape (num_games, TetrisBoardRaw::SIZE, 2)
    pub fn into_dist(&self) -> Result<TetrisBoardsDistTensor> {
        let p1 = candle_nn::ops::sigmoid(self.inner())?;
        let p0 = (1.0 - &p1)?;

        // Cat to create [B, SIZE, 2] where last dim is [p0, p1]
        let p0_expanded = p0.unsqueeze(2)?; // [B, SIZE, 1]
        let p1_expanded = p1.unsqueeze(2)?; // [B, SIZE, 1]
        let dist = Tensor::cat(&[p0_expanded, p1_expanded], 2)?; // [B, SIZE, 2]

        Ok(TetrisBoardsDistTensor::try_from(dist)?)
    }

    pub fn argmax(&self) -> Result<TetrisBoardsTensor> {
        // For single logits, threshold at 0: positive -> 1, negative/zero -> 0
        let argmax = self.inner().gt(0.0f64)?.to_dtype(DType::U8)?;
        Ok(TetrisBoardsTensor::try_from(argmax)?)
    }

    pub fn sample(&self, temperature: f32) -> Result<TetrisBoardsTensor> {
        let (batch, size) = self.shape_tuple();
        if temperature == 0.0 {
            return self.argmax();
        }

        // Temperature scaling: scale logits by 1/T, then apply sigmoid
        let scaled_logits = (self.inner() / (temperature as f64))?;
        let p1 = candle_nn::ops::sigmoid(&scaled_logits)?; // [B, SIZE]

        // Sample Bernoulli using probability of state 1
        let u = Tensor::rand(0f32, 1f32, &[batch, size], self.device())?; // [B, SIZE]
        let sampled = u.lt(&p1)?.to_dtype(DType::U8)?; // 1 if u < p1, else 0
        Ok(TetrisBoardsTensor::try_from(sampled)?)
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

impl_wrapped_tensor!(
    TetrisBoardsDistTensor,
    dtype = DType::F32,
    shape_spec = (
        ShapeDim::Any,
        ShapeDim::Dim(TetrisBoardRaw::SIZE),
        ShapeDim::Dim(2)
    )
);

impl TetrisBoardsDistTensor {
    pub fn from_boards_tensor(tensor: TetrisBoardsTensor) -> Result<Self> {
        let (num_games, board_size) = tensor.shape_tuple();
        let tensor = tensor.inner().to_dtype(DType::F32)?;
        let a = tensor.reshape(&[num_games, board_size, 1])?;
        let b = (1.0 - a.clone())?;
        let dists = Tensor::cat(&[&a, &b], 2)?;
        Ok(Self::try_from(dists)?)
    }

    /// Add random noise to the distribution tensor along the last dimension
    /// The noise is sampled from a uniform distribution between -epsilon and epsilon
    /// The resulting distribution is normalized to sum to 1 along the last dimension
    pub fn noise(&self, epsilon: f32) -> Result<Self> {
        let (batch, size, states) = self.shape_tuple();
        let noise = Tensor::rand(-epsilon, epsilon, &[batch, size, states], self.device())?;
        let noised = (&self.0 + &noise)?;
        // Ensure values stay positive
        let noised =
            noised.maximum(&Tensor::zeros(noised.shape(), DType::F32, noised.device())?)?;
        // Normalize along last dim
        let sums = noised.sum(D::Minus1)?.reshape(&[batch, size, 1])?;
        let normalized = (noised / sums)?;
        Ok(Self::try_from(normalized)?)
    }

    /// Convert a tetris boards distribution tensor to a tetris boards tensor
    /// by taking the argmax of the last dimension
    /// Returns a tetris boards tensor of shape (num_games, TetrisBoardRaw::SIZE)
    pub fn argmax(&self) -> Result<TetrisBoardsTensor> {
        let argmax = self.inner().argmax(D::Minus1)?;
        Ok(TetrisBoardsTensor::try_from(argmax)?)
    }

    pub fn sample(&self, temperature: f32) -> Result<TetrisBoardsTensor> {
        let (batch, size, states) = self.shape_tuple();
        if temperature == 0.0 {
            return self.argmax();
        }
        // Temperature scaling: q = softmax(log(p) / T)
        let device = self.device();
        let eps = Tensor::new(f32::EPSILON, device)?.broadcast_as((batch, size, states))?;
        let p = self.inner().maximum(&eps)?;
        let logits = p.log()?;
        let scaled = softmax(&(logits / (temperature as f64))?, D::Minus1)?; // [B, SIZE, 2]

        // For binary distribution, sample Bernoulli using probability of state 1
        let p1 = scaled.i((.., .., 1))?; // [B, SIZE]
        let u = Tensor::rand(0f32, 1f32, &[batch, size], device)?; // [B, SIZE]
        let sampled = u.lt(&p1)?.to_dtype(DType::U8)?; // 1 if choose state 1, else 0
        Ok(TetrisBoardsTensor::try_from(sampled)?)
    }

    /// Compares two tetris boards distribution tensors
    /// Returns a tensor of shape (batch_size)
    pub fn similarity(&self, other: &Self) -> Result<Tensor> {
        let (batch_size, _board_size, _states) = self.shape_tuple();
        let (other_batch_size, _other_board_size, _other_states) = other.shape_tuple();
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
        let device = self.device();
        let eps = Tensor::new(f32::EPSILON, device)?.broadcast_as(self.shape_tuple())?;
        let p = self.inner().maximum(&eps)?;

        let plogp = p.log()?.mul(&p)?; // [B, SIZE, 2]
        let h_per_cell = plogp.sum(D::Minus1)?; // [B, SIZE]
        let h_per_game = h_per_cell.sum(D::Minus1)?; // [B]
        Ok(h_per_game.neg()?)
    }
}

/// Tetris piece placement tensor
/// Tensor of shape (batch_size, 1)
///
/// Represents the index of a piece placement
#[derive(Debug, Clone)]
pub struct TetrisPiecePlacementTensor(Tensor);

impl_wrapped_tensor!(
    TetrisPiecePlacementTensor,
    dtype = DType::U8,
    shape_spec = (ShapeDim::Any, ShapeDim::Dim(1))
);

impl TetrisPiecePlacementTensor {
    /// Convert a list of placements to a tensor
    /// using the index of the placement
    pub fn from_placements(placements: &[TetrisPiecePlacement], device: &Device) -> Result<Self> {
        let shape = Shape::from_dims(&[placements.len(), 1]);
        let placement_indices: Vec<u8> = placements.iter().map(|p| p.index()).collect();
        let tensor = Tensor::from_vec(placement_indices, &shape, device)?;
        Ok(Self::try_from(tensor)?)
    }

    /// Convert a tensor of placement indices to a list of placements
    pub fn into_placements(&self) -> Result<Vec<TetrisPiecePlacement>> {
        let placement_ints = self
            .inner()
            .flatten_all()?
            .to_dtype(DType::U8)?
            .to_vec1::<u8>()?;
        Ok(placement_ints
            .into_iter()
            .map(TetrisPiecePlacement::from_index)
            .collect())
    }
}

/// Tetris piece placement distribution tensor
/// Tensor of shape (batch_size, TetrisPiecePlacement::NUM_PLACEMENTS)
///
/// Represents the distribution of a piece placement across all possible placements
#[repr(transparent)]
pub struct TetrisPiecePlacementDistTensor(Tensor);

impl_wrapped_tensor!(
    TetrisPiecePlacementDistTensor,
    dtype = DType::F32,
    shape_spec = (
        ShapeDim::Any,
        ShapeDim::Dim(TetrisPiecePlacement::NUM_PLACEMENTS)
    )
);

impl TetrisPiecePlacementDistTensor {
    pub fn into_placements_argmax(&self) -> Result<Vec<TetrisPiecePlacement>> {
        let argmax = self.argmax(D::Minus1)?;
        let placements = argmax.to_vec1::<u8>()?;
        Ok(placements
            .into_iter()
            .map(TetrisPiecePlacement::from_index)
            .collect())
    }

    pub fn from_placements_tensor(tensor: TetrisPiecePlacementTensor) -> Result<Self> {
        let (batch_size, _) = tensor.shape_tuple();
        let dists = one_hot(
            tensor.inner().clone(),
            TetrisPiecePlacement::NUM_PLACEMENTS,
            1f32,
            0f32,
        )?
        .to_dtype(DType::F32)?
        .reshape(&[batch_size, TetrisPiecePlacement::NUM_PLACEMENTS])?;
        Ok(Self::try_from(dists)?)
    }
}

/// Tetris piece orientation tensor
/// Tensor of shape (batch_size, 1)
///
/// Represents the index of a piece orientation
#[derive(Debug, Clone)]
pub struct TetrisPieceOrientationTensor(Tensor);

impl_wrapped_tensor!(
    TetrisPieceOrientationTensor,
    dtype = DType::U8,
    shape_spec = (ShapeDim::Any, ShapeDim::Dim(1))
);

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

    pub fn into_dist(&self) -> Result<TetrisPieceOrientationDistTensor> {
        let (batch_size, _) = self.shape_tuple();
        let dist = one_hot(
            self.0.clone(),
            TetrisPieceOrientation::NUM_ORIENTATIONS,
            1f32,
            0f32,
        )?
        .to_dtype(DType::F32)?
        .reshape(&[batch_size, TetrisPieceOrientation::NUM_ORIENTATIONS])?;
        Ok(TetrisPieceOrientationDistTensor::try_from(dist)?)
    }
}

/// Tetris piece orientation logits tensor
/// Tensor of shape (batch_size, num_orientations)
///
/// Represents the logits of a piece orientation
#[derive(Debug, Clone)]
pub struct TetrisPieceOrientationLogitsTensor(Tensor);

impl_wrapped_tensor!(
    TetrisPieceOrientationLogitsTensor,
    dtype = DType::F32,
    shape_spec = (
        ShapeDim::Any,
        ShapeDim::Dim(TetrisPieceOrientation::NUM_ORIENTATIONS)
    )
);

impl TetrisPieceOrientationLogitsTensor {
    pub fn into_dist(&self) -> Result<TetrisPieceOrientationDistTensor> {
        let dist = softmax(&self.0, D::Minus1)?;
        Ok(TetrisPieceOrientationDistTensor::try_from(dist)?)
    }
}

/// Tetris piece orientation tensor
/// Tensor of shape (batch_size, num_orientations)
///
/// Represents the index of a piece orientation
#[derive(Debug, Clone)]
pub struct TetrisPieceOrientationDistTensor(Tensor);

impl_wrapped_tensor!(
    TetrisPieceOrientationDistTensor,
    dtype = DType::F32,
    shape_spec = (
        ShapeDim::Any,
        ShapeDim::Dim(TetrisPieceOrientation::NUM_ORIENTATIONS)
    )
);

impl TetrisPieceOrientationDistTensor {
    /// Convert a list of orientations to a tensor
    /// using the index of the orientation
    pub fn from_orientations(
        orientations: &[TetrisPieceOrientation],
        device: &Device,
    ) -> Result<Self> {
        let base_tensor = TetrisPieceOrientationTensor::from_orientations(orientations, device)?;
        let self_tensor = Self::from_orientations_tensor(base_tensor)?;
        Ok(self_tensor)
    }

    pub fn from_orientations_tensor(tensor: TetrisPieceOrientationTensor) -> Result<Self> {
        let (batch_size, _) = tensor.shape_tuple();
        let dists = one_hot(
            tensor.inner().clone(),
            TetrisPieceOrientation::NUM_ORIENTATIONS,
            1f32,
            0f32,
        )?
        .to_dtype(DType::F32)?
        .reshape(&[batch_size, TetrisPieceOrientation::NUM_ORIENTATIONS])?;
        Ok(Self::try_from(dists)?)
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

/// Tetris Piece tensor
/// Tensor of shape (batch_size, 1)
///
/// Represents the index of a piece placement
#[derive(Debug, Clone)]
pub struct TetrisPieceTensor(Tensor);

impl_wrapped_tensor!(
    TetrisPieceTensor,
    dtype = DType::U32,
    shape_spec = (ShapeDim::Any, ShapeDim::Dim(1))
);

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

/// Tetris context tensor
/// Tensor of shape (batch_size, seq_len, dim)
///
/// Represents the context of a sequence of tokens
#[derive(Debug, Clone)]
pub struct TetrisContextTensor(Tensor);

impl_wrapped_tensor!(
    TetrisContextTensor,
    dtype = DType::F32,
    shape_spec = (ShapeDim::Any, ShapeDim::Any, ShapeDim::Any)
);

impl TetrisContextTensor {
    pub fn push_tokens(&self, tokens: &Tensor) -> Result<Self> {
        // cat([B, S_A, D], [B, S_B, D]) -> [B, S_A+S_B, D]
        let new_tensor = Tensor::cat(&[self.inner(), tokens], 1)?;
        Ok(Self::try_from(new_tensor)?)
    }

    pub fn swap_goal_token(&self, new_goal_token: &Tensor) -> Result<Self> {
        // self: [B, S, D] where S >= 2; new_goal_token: [B, 1, D]
        let (batch_size, seq_len, dim) = self.shape_tuple();
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
        let new_goal_token = new_goal_token.to_device(self.device())?;

        // Replace [:, 0, :] by building [new_goal_token, self[:, 1:, :]] along dim=1
        let tail = self.narrow(1, 1, seq_len - 1)?; // [B, S-1, D]
        let new_tensor = Tensor::cat(&[&new_goal_token, &tail], 1)?;
        Ok(Self(new_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TetrisDatasetGenerator;

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
            .inner()
            .eq(recovered_orientations_tensor.inner())
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
            .inner()
            .eq(recovered_placements_tensor.inner())
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
}
