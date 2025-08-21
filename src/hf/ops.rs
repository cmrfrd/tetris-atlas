use crate::{
    data::{TetrisBoardsTensor, TetrisPieceTensor},
    hf::{EPS, data::TetrisBoardsDistTensor},
    tetris::{TetrisPiece, TetrisPieceOrientation},
};
use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::loss::cross_entropy;

/// Create the mask over all orientations for a specific piece.
const PIECE_MASK_LOOKUP: [u8; TetrisPiece::NUM_PIECES * TetrisPieceOrientation::NUM_ORIENTATIONS] = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, // I
    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, // O
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, // T
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, // L
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 0, // J
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 0, // Z
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 0, // S
];
pub fn create_orientation_mask(pieces: &TetrisPieceTensor) -> Result<Tensor> {
    let device = pieces.device();
    let masks: Tensor = Tensor::from_slice(
        &PIECE_MASK_LOOKUP,
        (
            TetrisPiece::NUM_PIECES,
            TetrisPieceOrientation::NUM_ORIENTATIONS,
        ),
        &device,
    )?;
    let result = masks.index_select(&pieces.squeeze(1)?, 0)?;
    Ok(result)
}

/// Masked softmax over the last dim of [B, N].
/// - `x`:    [B, N] f32
/// - `mask`: [B, N] (bool/int/float; >0 => keep, 0 => mask)
/// Returns [B, N] probs; masked entries are exactly 0.
/// Rows that are fully masked become all-zeros (no NaNs).
pub fn masked_softmax_2d(x: &Tensor, mask: &Tensor) -> Result<Tensor> {
    if x.dims().len() != 2 || mask.dims().len() != 2 {
        return Err(anyhow::Error::msg(
            "masked_softmax_2d: both inputs must be 2D [B, N]",
        ));
    }
    if x.dims() != mask.dims() {
        return Err(anyhow::Error::msg(
            "masked_softmax_2d: x and mask must have identical shapes",
        ));
    }
    if x.dtype() != DType::F32 {
        return Err(anyhow::Error::msg("masked_softmax_2d: x must be f32"));
    }

    // build an integer predicate: non-zero => keep
    let zero = mask.zeros_like()?;
    let keep = mask.gt(&zero)?; // integer dtype (u8/u32/i64)
    let _keep_f32 = keep.to_dtype(DType::F32)?; // kept for potential downstream use

    // stable masked max: exclude masked positions by setting them to -inf for the max step
    let neg_inf = Tensor::full(f32::NEG_INFINITY, x.dims(), x.device())?;
    let masked_for_max = keep.where_cond(x, &neg_inf)?; // [B,N]
    let max_keep = masked_for_max.max_keepdim(1)?; // [B,1]

    // shift by per-row max and set masked positions to -inf BEFORE exp to avoid 0 * inf => NaN
    let shifted = x.broadcast_sub(&max_keep)?; // [B,N]
    let shifted_masked = keep.where_cond(&shifted, &neg_inf)?; // [B,N]
    let exps = shifted_masked.exp()?; // [B,N], masked -> exp(-inf)=0
    let denom = exps.sum_keepdim(1)?; // [B,1]
    let eps = Tensor::full(1e-12f32, denom.dims(), x.device())?;
    let denom = (denom + &eps)?; // avoid division by zero
    let probs = exps.broadcast_div(&denom)?; // [B,N]

    // Optional safeguard: if a row of x is exactly all zeros, force output row to zeros
    // This detects rows with sum(|x|) == 0 and gates the probabilities off for those rows.
    let row_nonzero = x.abs()?.sum_keepdim(1)?.gt(&denom.zeros_like()?)?; // [B,1] int
    let row_nonzero_f32 = row_nonzero.to_dtype(DType::F32)?; // [B,1]
    Ok(probs.broadcast_mul(&row_nonzero_f32)?)
}

pub fn board_loss(
    outputs: &TetrisBoardsDistTensor,
    targets: &TetrisBoardsTensor,
) -> Result<Tensor> {
    let (outputs_batch_size, outputs_seq_len, outputs_states) = outputs.dims3()?;
    let (targets_batch_size, targets_seq_len) = targets.dims2()?;
    if outputs_batch_size != targets_batch_size {
        return Err(anyhow::Error::msg(
            "board_loss: outputs and targets must have identical shapes",
        ));
    }

    let outputs_flat = outputs.reshape(&[outputs_batch_size * outputs_seq_len, outputs_states])?;
    let targets_flat = targets.reshape(&[targets_batch_size * targets_seq_len])?;
    let loss = cross_entropy(&outputs_flat, &targets_flat)?;
    Ok(loss)
}

/// KL(P || Q) where `p` and `q` are probability distributions along `dim`.
/// Uses PyTorch-style "batchmean": sum along `dim`, then mean across batches.
pub fn kl_div<D: candle_core::shape::Dim>(p: &Tensor, q: &Tensor, dim: D) -> Result<Tensor> {
    let (p_batch_size, p_dim) = p.dims2()?;
    let (q_batch_size, q_dim) = q.dims2()?;
    if p_batch_size != q_batch_size {
        return Err(anyhow::Error::msg(
            "kl_div: p and q must have identical batch sizes",
        ));
    }
    if p_dim != q_dim {
        return Err(anyhow::Error::msg(
            "kl_div: p and q must have identical dimensions",
        ));
    }

    let p_stable = (p + EPS)?;
    let q_stable = (q + EPS)?;

    let log_p = p_stable.log()?;
    let log_q = q_stable.log()?;
    let elem = (p_stable * (log_p - log_q)?)?;
    Ok(elem.sum(dim)?.mean_all()?)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn kl_simple_probabilities() -> Result<()> {
        let dev = Device::Cpu;
        let p = Tensor::from_vec(vec![0.5f32, 0.5], (1, 2), &dev)?;
        let q = Tensor::from_vec(vec![0.75f32, 0.25], (1, 2), &dev)?;
        let kl = kl_div(&p, &q, 1)?;
        let v = kl.to_scalar::<f32>()?;
        let expected = 0.5_f32 * (4.0_f32 / 3.0).ln();
        assert!((v - expected).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_create_orientation_mask() -> Result<()> {
        let pieces = TetrisPiece::all();
        let pieces_tensor = TetrisPieceTensor::from_pieces(&pieces, &Device::Cpu)?;
        let mask = create_orientation_mask(&pieces_tensor)?.to_dtype(DType::F32)?;
        let expected_mask = vec![
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            ],
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            ],
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            ],
        ];
        let mask_as_vec = mask.to_vec2::<f32>().unwrap();
        assert_eq!(mask_as_vec, expected_mask);

        // Also just check reversed for the heck of it
        let expected_mask_reversed = expected_mask.iter().rev().cloned().collect::<Vec<_>>();
        let reversed_pieces = pieces.iter().rev().cloned().collect::<Vec<_>>();
        let reversed_pieces_tensor =
            TetrisPieceTensor::from_pieces(&reversed_pieces, &Device::Cpu)?;
        let reversed_mask =
            create_orientation_mask(&reversed_pieces_tensor)?.to_dtype(DType::F32)?;
        let reversed_mask_as_vec = reversed_mask.to_vec2::<f32>().unwrap();
        assert_eq!(reversed_mask_as_vec, expected_mask_reversed);

        // test indexing the same piece 100 times
        let pieces = vec![TetrisPiece::O_PIECE; 100];
        let pieces_tensor = TetrisPieceTensor::from_pieces(&pieces, &Device::Cpu)?;
        let mask = create_orientation_mask(&pieces_tensor)?.to_dtype(DType::F32)?;
        let mask_as_vec = mask.to_vec2::<f32>().unwrap();
        let expected_mask = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        for (i, row) in mask_as_vec.iter().enumerate() {
            assert_eq!(row, &expected_mask);
        }

        Ok(())
    }
}
