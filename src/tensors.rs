use crate::{
    impl_wrapped_tensor,
    ops::{create_orientation_mask, kl_div},
    tetris::{
        TetrisBoard, TetrisGameSet, TetrisPiece, TetrisPieceOrientation, TetrisPiecePlacement,
    },
    wrapped_tensor::{ShapeDim, WrappedTensor},
};
use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Shape, Tensor};
use candle_nn::{encoding::one_hot, ops::softmax};

/// Tensor wrapper for Tetris boards.
/// Shape: (num_games, TetrisBoard::SIZE)
///
/// Each row represents a single board, flattened to a 1D binary vector of length TetrisBoard::SIZE.
#[derive(Debug, Clone)]
pub struct TetrisBoardsTensor(Tensor);

impl_wrapped_tensor!(
    TetrisBoardsTensor,
    dtype = DType::U8,
    shape_spec = (ShapeDim::Any, ShapeDim::Dim(TetrisBoard::SIZE))
);

impl TetrisBoardsTensor {
    /// Create a tetris boards tensor from a gameset
    pub fn from_gameset(games: TetrisGameSet, device: &Device) -> Result<Self> {
        let mut boards = Vec::with_capacity(games.len() * TetrisBoard::SIZE);
        games.boards().iter().for_each(|board| {
            boards.extend_from_slice(&board.to_binary_slice());
        });
        let shape = Shape::from_dims(&[games.len(), TetrisBoard::SIZE]);
        let boards = Tensor::from_vec(boards, &shape, device)?;
        Self::try_from(boards)
    }

    /// Create a tetris boards tensor from a slice of raw boards
    pub fn from_boards(boards: &[TetrisBoard], device: &Device) -> Result<Self> {
        let shape = Shape::from_dims(&[boards.len(), TetrisBoard::SIZE]);
        let mut flattened: Vec<u8> = Vec::with_capacity(boards.len() * TetrisBoard::SIZE);
        for board in boards.iter() {
            flattened.extend(board.to_binary_slice().iter());
        }
        let boards = Tensor::from_vec(flattened, &shape, device)?;
        Self::try_from(boards)
    }

    /// Convert tetris board tensor into tetris boards
    pub fn into_boards(self) -> Result<Vec<TetrisBoard>> {
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
                boards.push(TetrisBoard::from_binary_slice(board.try_into().unwrap()));
            }
            boards
        };
        Ok(boards)
    }

    pub fn into_dist(&self) -> Result<TetrisBoardsDistTensor> {
        let dist = one_hot(
            self.inner().clone(),
            TetrisBoard::NUM_TETRIS_CELL_STATES,
            1f32,
            0f32,
        )?;
        TetrisBoardsDistTensor::try_from(dist)
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
        let perc_cells_equal = (num_cells_equal / ((TetrisBoard::SIZE) as f64))?;
        let perc_equal = perc_cells_equal
            .to_dtype(DType::F32)?
            .mean([0])?
            .to_scalar::<f32>()?;
        Ok(perc_equal)
    }
}

/// Tetris boards logits tensor
/// Tensor of shape (num_games, TetrisBoard::SIZE)
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
    dtype = crate::fdtype(),
    shape_spec = (ShapeDim::Any, ShapeDim::Dim(TetrisBoard::SIZE))
);

impl TetrisBoardLogitsTensor {
    /// Convert the logits tensor to a distribution tensor by applying sigmoid to get probabilities
    /// Returns a tetris boards distribution tensor of shape (num_games, TetrisBoard::SIZE, 2)
    pub fn into_dist(&self) -> Result<TetrisBoardsDistTensor> {
        let p1 = candle_nn::ops::sigmoid(self.inner())?;
        let p0 = (1.0 - &p1)?;

        // Cat to create [B, SIZE, 2] where last dim is [p0, p1]
        let p0_expanded = p0.unsqueeze(2)?; // [B, SIZE, 1]
        let p1_expanded = p1.unsqueeze(2)?; // [B, SIZE, 1]
        let dist = Tensor::cat(&[p0_expanded, p1_expanded], 2)?; // [B, SIZE, 2]

        TetrisBoardsDistTensor::try_from(dist)
    }

    pub fn argmax(&self) -> Result<TetrisBoardsTensor> {
        // For single logits, threshold at 0: positive -> 1, negative/zero -> 0
        let argmax = self.inner().gt(0.0f64)?.to_dtype(DType::U8)?;
        TetrisBoardsTensor::try_from(argmax)
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
        TetrisBoardsTensor::try_from(sampled)
    }
}

/// Tetris boards distribution tensor
/// Tensor of shape (num_games, TetrisBoard::SIZE, 2)
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
    dtype = crate::fdtype(),
    shape_spec = (
        ShapeDim::Any,
        ShapeDim::Dim(TetrisBoard::SIZE),
        ShapeDim::Dim(2)
    )
);

impl TetrisBoardsDistTensor {
    pub fn from_boards_tensor(tensor: TetrisBoardsTensor) -> Result<Self> {
        let dtype = Self::expected_dtype();
        let (num_games, board_size) = tensor.shape_tuple();
        let tensor = tensor.inner().to_dtype(dtype)?;
        let a = tensor.reshape(&[num_games, board_size, 1])?;
        let b = (1.0 - a.clone())?;
        let dists = Tensor::cat(&[&a, &b], 2)?;
        Self::try_from(dists)
    }

    /// Add random noise to the distribution tensor along the last dimension
    /// The noise is sampled from a uniform distribution between -epsilon and epsilon
    /// The resulting distribution is normalized to sum to 1 along the last dimension
    pub fn noise(&self, epsilon: f32) -> Result<Self> {
        let dtype = Self::expected_dtype();
        let (batch, size, states) = self.shape_tuple();
        let noise = Tensor::rand(-epsilon, epsilon, &[batch, size, states], self.device())?;
        let noised = (&self.0 + &noise)?;
        // Ensure values stay positive
        let noised = noised.maximum(&Tensor::zeros(noised.shape(), dtype, noised.device())?)?;
        // Normalize along last dim
        let sums = noised.sum(D::Minus1)?.reshape(&[batch, size, 1])?;
        let normalized = (noised / sums)?;
        Self::try_from(normalized)
    }

    /// Convert a tetris boards distribution tensor to a tetris boards tensor
    /// by taking the argmax of the last dimension
    /// Returns a tetris boards tensor of shape (num_games, TetrisBoard::SIZE)
    pub fn argmax(&self) -> Result<TetrisBoardsTensor> {
        let argmax = self.inner().argmax(D::Minus1)?;
        TetrisBoardsTensor::try_from(argmax)
    }

    pub fn sample(&self, temperature: f32) -> Result<TetrisBoardsTensor> {
        let dtype = Self::expected_dtype();
        let (batch, size, states) = self.shape_tuple();
        if temperature == 0.0 {
            return self.argmax();
        }
        // Temperature scaling: q = softmax(log(p) / T)
        let device = self.device();
        let eps = Tensor::new(f32::EPSILON, device)?
            .broadcast_as((batch, size, states))?
            .to_dtype(dtype)?;
        let p = self.inner().maximum(&eps)?;
        let logits = p.log()?;
        let scaled = softmax(&(logits / (temperature as f64))?, D::Minus1)?; // [B, SIZE, 2]

        // For binary distribution, sample Bernoulli using probability of state 1
        let p1 = scaled.i((.., .., 1))?; // [B, SIZE]
        let u = Tensor::rand(0f32, 1f32, &[batch, size], device)?; // [B, SIZE]
        let sampled = u.lt(&p1)?.to_dtype(DType::U8)?; // 1 if choose state 1, else 0
        TetrisBoardsTensor::try_from(sampled)
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
        let dtype = Self::expected_dtype();
        let device = self.device();
        let eps = Tensor::new(f32::EPSILON, device)?
            .broadcast_as(self.shape_tuple())?
            .to_dtype(dtype)?;
        let p = self.inner().maximum(&eps)?.to_dtype(dtype)?;
        let plogp = p.log()?.mul(&p)?; // [B, SIZE, 2]
        let h_per_cell = plogp.sum(D::Minus1)?.to_dtype(dtype)?; // [B, SIZE]
        let h_per_game = h_per_cell.sum(D::Minus1)?.to_dtype(dtype)?; // [B]
        Ok(h_per_game.neg()?)
    }
}

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
        Self::try_from(tensor)
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
    dtype = crate::fdtype(),
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
        let dtype = Self::expected_dtype();
        let (batch_size, _) = tensor.shape_tuple();
        let dists = one_hot(
            tensor.inner().clone(),
            TetrisPiecePlacement::NUM_PLACEMENTS,
            1f32,
            0f32,
        )?
        .to_dtype(dtype)?
        .reshape(&[batch_size, TetrisPiecePlacement::NUM_PLACEMENTS])?;
        Self::try_from(dists)
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
            TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS,
            1f32,
            0f32,
        )?
        .to_dtype(TetrisPieceOrientationDistTensor::expected_dtype())?
        .reshape(&[batch_size, TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS])?;
        TetrisPieceOrientationDistTensor::try_from(dist)
    }

    pub fn perc_equal(&self, other: &Self) -> Result<f32> {
        let (batch_size, _) = self.shape_tuple();
        let (other_batch_size, _) = other.shape_tuple();
        assert!(
            batch_size == other_batch_size,
            "Batch sizes must match, got {} and {}",
            batch_size,
            other_batch_size
        );
        let equal = self.inner().eq(other.inner())?.to_dtype(DType::F32)?;
        let perc_equal = (equal.sum_all()? / (batch_size as f64))?.to_scalar::<f32>()?;
        Ok(perc_equal)
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
    dtype = crate::fdtype(),
    shape_spec = (
        ShapeDim::Any,
        ShapeDim::Dim(TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS)
    )
);

impl TetrisPieceOrientationLogitsTensor {
    pub fn into_dist(&self) -> Result<TetrisPieceOrientationDistTensor> {
        let dist = softmax(&self.0, D::Minus1)?;
        TetrisPieceOrientationDistTensor::try_from(dist)
    }

    pub fn sample(
        &self,
        temperature: f32,
        pieces: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationTensor> {
        let device = self.device();
        let dtype = Self::expected_dtype();
        let (batch_size, num_orientations) = self.shape_tuple();

        if pieces.shape_tuple() != (batch_size, 1) {
            return Err(anyhow::anyhow!(
                "pieces must have shape (batch_size, 1), got {:?}",
                pieces.shape_tuple()
            ));
        }

        let keep_mask = create_orientation_mask(pieces)?.gt(0u32)?;

        let neg_inf = Tensor::full(f64::NEG_INFINITY, (), device)?
            .to_dtype(dtype)?
            .broadcast_as(self.dims())?;
        let logits = keep_mask.where_cond(&self.inner(), &neg_inf)?;

        if temperature == 0.0 {
            let argmax = logits
                .argmax(D::Minus1)?
                .to_dtype(DType::U8)?
                .reshape(&[batch_size, 1])?;
            return TetrisPieceOrientationTensor::try_from(argmax);
        }

        // Gumbel-max sampling with temperature
        let scaled_logits = (logits / (temperature as f64))?;

        let u =
            Tensor::rand(0f32, 1f32, &[batch_size, num_orientations], device)?.to_dtype(dtype)?;
        let eps = Tensor::new(f32::EPSILON, device)?
            .broadcast_as(u.shape())?
            .to_dtype(dtype)?;
        let one = Tensor::new(1f32, device)?
            .broadcast_as(u.shape())?
            .to_dtype(dtype)?;
        let u = u.maximum(&eps)?.minimum(&(&one - &eps)?)?;
        let gumbel = u.log()?.neg()?.log()?.neg()?; // -log(-log(u))

        let y = (scaled_logits + gumbel)?;
        let sampled = y
            .argmax(D::Minus1)?
            .to_dtype(DType::U8)?
            .reshape(&[batch_size, 1])?;
        TetrisPieceOrientationTensor::try_from(sampled)
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
    dtype = crate::fdtype(),
    shape_spec = (
        ShapeDim::Any,
        ShapeDim::Dim(TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS)
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
        let dtype = Self::expected_dtype();
        let (batch_size, _) = tensor.shape_tuple();
        let dists = one_hot(
            tensor.inner().clone(),
            TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS,
            1f32,
            0f32,
        )?
        .to_dtype(dtype)?
        .reshape(&[batch_size, TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS])?;
        Self::try_from(dists)
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

    pub fn entropy(&self) -> Result<Tensor> {
        let dtype = Self::expected_dtype();
        let device = self.device();
        let eps = Tensor::new(f32::EPSILON, device)?
            .broadcast_as(self.shape_tuple())?
            .to_dtype(dtype)?;
        let p = self.inner().maximum(&eps)?;

        let plogp = p.log()?.mul(&p)?; // [B, NUM_ORIENTATIONS]
        let h = plogp.sum(D::Minus1)?; // [B]
        Ok(h.neg()?)
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

/// Tetris piece one-hot tensor
/// Tensor of shape (batch_size, NUM_PIECES)
///
/// One-hot encoded representation of pieces
#[derive(Debug, Clone)]
pub struct TetrisPieceOneHotTensor(Tensor);

impl_wrapped_tensor!(
    TetrisPieceOneHotTensor,
    dtype = crate::fdtype(),
    shape_spec = (ShapeDim::Any, ShapeDim::Dim(TetrisPiece::NUM_PIECES))
);

impl TetrisPieceOneHotTensor {
    pub fn from_piece_tensor(piece_tensor: TetrisPieceTensor) -> Result<Self> {
        let dtype = Self::expected_dtype();
        let (batch_size, _) = piece_tensor.shape_tuple();
        let one_hot = one_hot(
            piece_tensor.inner().clone(),
            TetrisPiece::NUM_PIECES,
            1f32,
            0f32,
        )?
        .to_dtype(dtype)?
        .reshape(&[batch_size, TetrisPiece::NUM_PIECES])?;
        Self::try_from(one_hot)
    }
}

/// Tetris piece sequence tensor
/// Tensor of shape (batch_size, seq_len)
///
/// Represents the sequence of pieces
#[derive(Debug, Clone)]
pub struct TetrisPieceSequenceTensor(Tensor);

impl_wrapped_tensor!(
    TetrisPieceSequenceTensor,
    dtype = DType::U32,
    shape_spec = (ShapeDim::Any, ShapeDim::Any)
);

impl TetrisPieceSequenceTensor {
    pub fn from_vec_pieces(pieces: Vec<TetrisPieceTensor>) -> Result<Self> {
        let inners = pieces.iter().map(|p| p.inner()).collect::<Vec<_>>();
        Self::try_from(Tensor::cat(&inners, 1)?)
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
    dtype = crate::fdtype(),
    shape_spec = (ShapeDim::Any, ShapeDim::Any, ShapeDim::Any)
);

impl TetrisContextTensor {
    pub fn push_tokens(&self, tokens: &Tensor) -> Result<Self> {
        let dtype = Self::expected_dtype();
        // cat([B, S_A, D], [B, S_B, D]) -> [B, S_A+S_B, D]
        let new_tensor = Tensor::cat(&[self.inner(), tokens], 1)?.to_dtype(dtype)?;
        Self::try_from(new_tensor)
    }

    pub fn swap_goal_token(&self, new_goal_token: &Tensor) -> Result<Self> {
        let dtype = Self::expected_dtype();
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
        let new_goal_token = new_goal_token.to_device(self.device())?.to_dtype(dtype)?;

        // Replace [:, 0, :] by building [new_goal_token, self[:, 1:, :]] along dim=1
        let tail = self.narrow(1, 1, seq_len - 1)?; // [B, S-1, D]
        let new_tensor = Tensor::cat(&[&new_goal_token, &tail], 1)?.to_dtype(dtype)?;
        Ok(Self(new_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TetrisDatasetGenerator;

    /// Test round-trip conversion: TetrisPieceOrientationTensor -> Vec<Orientation> -> TetrisPieceOrientationTensor
    ///
    /// Verifies that:
    /// 1. Converting a tensor to orientations and back produces an identical tensor
    /// 2. The orientation values are preserved through the conversion
    #[test]
    fn test_into_from_orientations_tensor() {
        let batch_size = 16;
        let device = Device::Cpu;

        // Generate test data with random orientations
        let generator = TetrisDatasetGenerator::new();
        let transition = generator
            .gen_uniform_sampled_transition((0..10).into(), batch_size, &device, &mut rand::rng())
            .unwrap();

        // Extract original orientations tensor and convert to Vec
        let original_tensor = transition.orientation;
        let orientations = original_tensor.clone().into_orientations().unwrap();

        // Convert back to tensor
        let recovered_tensor =
            TetrisPieceOrientationTensor::from_orientations(&orientations, &device).unwrap();
        let recovered_orientations = recovered_tensor.clone().into_orientations().unwrap();

        // Verify tensors are identical by counting equal elements
        let num_equal = original_tensor
            .inner()
            .eq(recovered_tensor.inner())
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();
        assert_eq!(
            num_equal, batch_size as u32,
            "All tensor elements should be equal after round-trip conversion"
        );

        // Verify orientation values are identical
        assert_eq!(
            orientations, recovered_orientations,
            "Orientation values should be preserved through conversion"
        );
    }

    /// Test round-trip conversion: TetrisPiecePlacementTensor -> Vec<Placement> -> Tensor
    ///
    /// Verifies that:
    /// 1. Converting a tensor to placements and back produces an identical tensor
    /// 2. The placement values are preserved through the conversion
    #[test]
    fn test_into_from_placements_tensor() {
        let batch_size = 16;
        let device = Device::Cpu;

        // Generate test data with random placements
        let generator = TetrisDatasetGenerator::new();
        let transition = generator
            .gen_uniform_sampled_transition((0..10).into(), batch_size, &device, &mut rand::rng())
            .unwrap();

        // Extract original placements tensor and convert to Vec
        let original_tensor = transition.placement;
        let placements = original_tensor.clone().into_placements().unwrap();

        // Convert back to tensor
        let recovered_tensor =
            TetrisPiecePlacementTensor::from_placements(&placements, &device).unwrap();
        let recovered_placements = recovered_tensor.clone().into_placements().unwrap();

        // Verify tensors are identical by counting equal elements
        let num_equal = original_tensor
            .inner()
            .eq(recovered_tensor.inner())
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();
        assert_eq!(
            num_equal, batch_size as u32,
            "All tensor elements should be equal after round-trip conversion"
        );

        // Verify placement values are identical
        assert_eq!(
            placements, recovered_placements,
            "Placement values should be preserved through conversion"
        );
    }

    /// Test round-trip conversion: TetrisBoardsTensor -> Vec<TetrisBoard> -> Tensor
    ///
    /// Verifies that:
    /// 1. Converting a board tensor to boards and back preserves the board state
    /// 2. The board data is correctly serialized and deserialized
    #[test]
    fn test_into_from_boards_tensor() {
        let batch_size = 16;
        let device = Device::Cpu;

        // Generate test data with random board states
        let generator = TetrisDatasetGenerator::new();
        let transition = generator
            .gen_uniform_sampled_transition((0..10).into(), batch_size, &device, &mut rand::rng())
            .unwrap();

        // Extract original boards tensor and convert to Vec
        let original_tensor = transition.result_board;
        let boards = original_tensor.clone().into_boards().unwrap();

        // Convert back to tensor
        let recovered_tensor = TetrisBoardsTensor::from_boards(&boards, &device).unwrap();
        let recovered_boards = recovered_tensor.into_boards().unwrap();

        // Verify board states are identical
        assert_eq!(
            boards, recovered_boards,
            "Board states should be preserved through round-trip conversion"
        );
    }

    /// Test masked orientation sampling with various mask configurations
    ///
    /// This comprehensive test verifies that:
    /// 1. Sampling works correctly for ALL 7 tetris pieces (I, O, T, S, Z, J, L)
    /// 2. Sampling works with the configured floating point dtype:
    ///    - Default: F32
    ///    - With `--features dtype-f16`: F16
    ///    - With `--features dtype-bf16`: BF16
    /// 3. Only valid (non-masked) orientations are sampled
    /// 4. Sampled orientations can be combined with pieces to create valid placements
    /// 5. Different temperature values produce valid results (0.0, 0.5, 1.0, 2.0, 10.0)
    /// 6. Argmax mode (temperature=0) respects masks
    /// 7. Works with various batch configurations (homogeneous, mixed, large)
    #[test]
    fn test_masked_orientation_sampling() {
        let device = Device::Cpu;
        let dtype = crate::fdtype();
        let num_orientations = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;

        // Get all piece types

        let all_pieces = TetrisPiece::all().to_vec();
        // --- Test 1: All pieces with various temperatures ---
        // Create batch with all pieces
        let batch_size = all_pieces.len();
        let pieces_tensor = TetrisPieceTensor::from_pieces(&all_pieces, &device).unwrap();

        // Precompute valid placements for each piece
        let valid_placements_per_piece: Vec<Vec<TetrisPiecePlacement>> = all_pieces
            .iter()
            .map(|&piece| TetrisPiecePlacement::all_from_piece(piece).to_vec())
            .collect();

        // Test with multiple temperatures
        let temperatures = vec![0.0, 0.5, 1.0, 2.0, 10.0];
        let num_samples_per_temp = 64;

        for &temp in temperatures.iter() {
            for _ in 0..num_samples_per_temp {
                // Generate random logits for each sample
                let logits = Tensor::randn(0f32, 1f32, (batch_size, num_orientations), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let logits_tensor = TetrisPieceOrientationLogitsTensor::try_from(logits).unwrap();

                // Sample orientations
                let sampled = logits_tensor.sample(temp, &pieces_tensor).unwrap();
                let orientations = sampled.into_orientations().unwrap();

                // Verify each sampled orientation creates a valid placement
                for (i, (piece, orientation)) in
                    all_pieces.iter().zip(orientations.iter()).enumerate()
                {
                    let placement = TetrisPiecePlacement {
                        piece: *piece,
                        orientation: *orientation,
                    };

                    assert!(
                        valid_placements_per_piece[i].contains(&placement),
                        "temp={}: Piece {:?} with sampled orientation {} creates invalid placement. \
                             Valid placements: {}",
                        temp,
                        piece,
                        orientation.index(),
                        valid_placements_per_piece[i].len()
                    );
                }
            }
        }

        // --- Test 2: Each piece individually with repeated sampling ---
        // Test each piece type separately to ensure piece-specific logic works
        for &piece in all_pieces.iter() {
            let valid_placements = TetrisPiecePlacement::all_from_piece(piece);
            let piece_batch = vec![piece; 16]; // Test with batch of same piece
            let pieces_tensor = TetrisPieceTensor::from_pieces(&piece_batch, &device).unwrap();

            for _ in 0..32 {
                let logits =
                    Tensor::randn(0f32, 1f32, (piece_batch.len(), num_orientations), &device)
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap();
                let logits_tensor = TetrisPieceOrientationLogitsTensor::try_from(logits).unwrap();

                let sampled = logits_tensor.sample(1.0, &pieces_tensor).unwrap();
                let orientations = sampled.into_orientations().unwrap();

                // All sampled orientations should create valid placements
                for orientation in orientations.iter() {
                    let placement = TetrisPiecePlacement {
                        piece,
                        orientation: *orientation,
                    };
                    assert!(
                        valid_placements.contains(&placement),
                        "Piece {:?} sampled orientation {} creates invalid placement",
                        piece,
                        orientation.index()
                    );
                }
            }
        }

        // --- Test 3: Argmax mode with targeted logits ---
        // Test argmax by setting strong preferences and verifying results
        for &piece in all_pieces.iter() {
            let valid_placements = TetrisPiecePlacement::all_from_piece(piece);
            let pieces_tensor = TetrisPieceTensor::from_pieces(&vec![piece; 1], &device).unwrap();

            // Try setting maximum at each possible orientation index
            for target_ori in 0..num_orientations {
                let mut logits_data = vec![-10.0f32; num_orientations];
                logits_data[target_ori] = 10.0;

                let logits = Tensor::from_vec(logits_data, (1, num_orientations), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let logits_tensor = TetrisPieceOrientationLogitsTensor::try_from(logits).unwrap();

                let sampled = logits_tensor.sample(0.0, &pieces_tensor).unwrap();
                let orientations = sampled.into_orientations().unwrap();

                let placement = TetrisPiecePlacement {
                    piece,
                    orientation: orientations[0],
                };

                // The argmax result should be valid (if invalid, it should pick a valid one)
                assert!(
                    valid_placements.contains(&placement),
                    "Argmax mode: Piece {:?} with target_ori={} resulted in orientation {} \
                         which creates invalid placement",
                    piece,
                    target_ori,
                    orientations[0].index()
                );
            }
        }

        // --- Test 4: Mixed piece batch ---
        // Create diverse batch mixing multiple copies of different pieces
        let mixed_pieces: Vec<TetrisPiece> = all_pieces
            .iter()
            .flat_map(|&p| vec![p, p, p]) // Triple each piece
            .collect();
        let pieces_tensor = TetrisPieceTensor::from_pieces(&mixed_pieces, &device).unwrap();

        for _ in 0..50 {
            let logits = Tensor::randn(0f32, 1f32, (mixed_pieces.len(), num_orientations), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let logits_tensor = TetrisPieceOrientationLogitsTensor::try_from(logits).unwrap();

            let sampled = logits_tensor.sample(1.5, &pieces_tensor).unwrap();
            let orientations = sampled.into_orientations().unwrap();

            for (piece, orientation) in mixed_pieces.iter().zip(orientations.iter()) {
                let placement = TetrisPiecePlacement {
                    piece: *piece,
                    orientation: *orientation,
                };
                let valid_placements = TetrisPiecePlacement::all_from_piece(*piece);

                assert!(
                    valid_placements.contains(&placement),
                    "Mixed batch: Piece {:?} with orientation {} creates invalid placement",
                    piece,
                    orientation.index()
                );
            }
        }

        // --- Test 5: Large homogeneous batch for ALL pieces ---
        // Stress test with large batches of each individual piece type
        // This tests every piece (I, O, T, S, Z, J, L) with 64 copies
        assert_eq!(
            all_pieces.len(),
            7,
            "Sanity check: should have all 7 Tetris pieces"
        );

        for &piece in all_pieces.iter() {
            let large_batch = vec![piece; 64];
            let pieces_tensor = TetrisPieceTensor::from_pieces(&large_batch, &device).unwrap();
            let valid_placements = TetrisPiecePlacement::all_from_piece(piece);

            // Sample multiple times for each piece to ensure robustness
            for _ in 0..3 {
                let logits = Tensor::randn(0f32, 1f32, (64, num_orientations), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let logits_tensor = TetrisPieceOrientationLogitsTensor::try_from(logits).unwrap();

                let sampled = logits_tensor.sample(2.0, &pieces_tensor).unwrap();
                let orientations = sampled.into_orientations().unwrap();

                // Verify all 64 sampled orientations create valid placements
                for orientation in orientations.iter() {
                    let placement = TetrisPiecePlacement {
                        piece,
                        orientation: *orientation,
                    };
                    assert!(
                        valid_placements.contains(&placement),
                        "Large batch: Piece {:?} with orientation {} creates invalid placement",
                        piece,
                        orientation.index()
                    );
                }
            }
        }
    }

    /// Test that masked logits produce valid (non-NaN) values in loss calculations
    ///
    /// When masked logits (with -inf for invalid options) go through softmax and log,
    /// they can potentially produce NaN values. This test verifies that the masking
    /// strategy produces numerically stable results throughout a typical loss calculation.
    ///
    /// Tests the following pipeline:
    /// 1. Masked logits -> softmax -> probabilities (should be finite)
    /// 2. Probabilities -> log -> log probabilities (should be finite)
    /// 3. Log probabilities -> cross-entropy loss (should be finite and non-negative)
    #[test]
    fn test_masked_logits_no_nan_in_loss() {
        use crate::ops::create_orientation_mask;
        use candle_nn::ops::softmax;

        let device = Device::Cpu;
        let dtype = crate::fdtype();
        let batch_size = 4;
        let num_orientations = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;

        // --- Step 1: Create masked logits ---
        // Generate random logits
        let logits = Tensor::randn(0f32, 1f32, (batch_size, num_orientations), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        // Create piece-based mask (each piece has different valid orientations)
        let pieces = vec![
            TetrisPiece::I_PIECE,
            TetrisPiece::O_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::L_PIECE,
        ];
        let pieces_tensor = TetrisPieceTensor::from_pieces(&pieces, &device).unwrap();
        let mask = create_orientation_mask(&pieces_tensor)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        // Apply masking: set masked-out positions to -inf
        let keep_mask = mask.gt(0.0).unwrap();
        let neg_inf = match dtype {
            DType::F16 => Tensor::full(f32::NEG_INFINITY, logits.dims(), &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap(),
            DType::BF16 => Tensor::full(f32::NEG_INFINITY, logits.dims(), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
            DType::F32 => Tensor::full(f32::NEG_INFINITY, logits.dims(), &device).unwrap(),
            DType::F64 => Tensor::full(f64::NEG_INFINITY, logits.dims(), &device).unwrap(),
            _ => panic!("Unsupported dtype for masking"),
        };
        let masked_logits = keep_mask.where_cond(&logits, &neg_inf).unwrap();

        // --- Step 2: Compute softmax probabilities ---
        let probs = softmax(&masked_logits, D::Minus1).unwrap();

        // Verify probabilities are finite
        let probs_f32 = probs.to_dtype(DType::F32).unwrap();
        let probs_vec = probs_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, &p) in probs_vec.iter().enumerate() {
            assert!(
                p.is_finite(),
                "Non-finite probability at index {}: {}",
                i,
                p
            );
        }

        // --- Step 3: Compute log probabilities with numerical stability ---
        // Use dtype-appropriate epsilon to avoid log(0) = -inf
        let eps_val = match dtype {
            DType::F16 => 1e-4f32,  // F16 has ~3 decimal digits of precision
            DType::BF16 => 1e-6f32, // BF16 has ~2 decimal digits of precision
            _ => 1e-10f32,          // F32/F64 can handle much smaller epsilon
        };
        let eps = Tensor::full(eps_val, probs.dims(), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let probs_stable = (probs + eps).unwrap();
        let log_probs = probs_stable.log().unwrap();

        // Verify log probabilities are finite
        let log_probs_f32 = log_probs.to_dtype(DType::F32).unwrap();
        let log_probs_vec = log_probs_f32
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, &lp) in log_probs_vec.iter().enumerate() {
            assert!(
                lp.is_finite(),
                "Non-finite log probability at index {}: {}",
                i,
                lp
            );
        }

        // --- Step 4: Compute cross-entropy loss ---
        // Create target indices that point to valid (non-masked) orientations
        let mut target_indices = Vec::new();
        let mask_f32 = mask.to_dtype(DType::F32).unwrap();
        for i in 0..batch_size {
            let mask_row = mask_f32.i(i).unwrap().to_vec1::<f32>().unwrap();

            // Find first valid orientation for this batch item
            let valid_idx = mask_row
                .iter()
                .position(|&x| x > 0.0)
                .expect("Each piece should have at least one valid orientation");
            target_indices.push(valid_idx as u32);
        }

        // Convert targets to one-hot encoding
        let targets = Tensor::from_vec(target_indices, (batch_size,), &device).unwrap();
        let targets_one_hot = candle_nn::encoding::one_hot(targets, num_orientations, 1.0, 0.0)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        // Calculate cross-entropy loss: -sum(target * log(pred))
        let loss_per_sample = (&targets_one_hot * &log_probs).unwrap();
        let loss = loss_per_sample.sum_all().unwrap().neg().unwrap();

        // Verify loss is valid
        let loss_val = loss
            .to_dtype(DType::F32)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            loss_val.is_finite(),
            "Cross-entropy loss should be finite, got: {}",
            loss_val
        );
        assert!(
            loss_val >= 0.0,
            "Cross-entropy loss should be non-negative, got: {}",
            loss_val
        );
    }
}
