use std::sync::RwLock;

use crate::{
    grad_accum::get_l2_norm,
    tensors::TetrisPieceTensor,
    tetris::{TetrisPiece, TetrisPieceOrientation},
};
use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var, backprop::GradStore};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::instrument;

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
#[instrument(level = "debug", skip(pieces))]
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
#[instrument(level = "debug", fields(x_shape = ?x.dims(), mask_shape = ?mask.dims()), skip(x, mask))]
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
    let device = x.device();

    // build an integer predicate: non-zero => keep
    let zero = mask.zeros_like()?;
    let keep = mask.gt(&zero)?; // integer dtype (u8/u32/i64)
    let _keep_f32 = keep.to_dtype(DType::F32)?; // kept for potential downstream use

    // stable masked max: exclude masked positions by setting them to -inf for the max step
    let neg_inf = Tensor::full(f32::NEG_INFINITY, x.dims(), device)?;
    let masked_for_max = keep.where_cond(x, &neg_inf)?; // [B,N]
    let max_keep = masked_for_max.max_keepdim(1)?; // [B,1]

    // shift by per-row max and set masked positions to -inf BEFORE exp to avoid 0 * inf => NaN
    let shifted = x.broadcast_sub(&max_keep)?; // [B,N]
    let shifted_masked = keep.where_cond(&shifted, &neg_inf)?; // [B,N]
    let exps = shifted_masked.exp()?; // [B,N], masked -> exp(-inf)=0
    let denom = exps.sum_keepdim(1)?; // [B,1]
    let eps = Tensor::full(f32::EPSILON, denom.dims(), device)?;
    let denom = (denom + &eps)?; // avoid division by zero
    let probs = exps.broadcast_div(&denom)?; // [B,N]

    // Optional safeguard: if a row of x is exactly all zeros, force output row to zeros
    // This detects rows with sum(|x|) == 0 and gates the probabilities off for those rows.
    let row_nonzero = x.abs()?.sum_keepdim(1)?.gt(&denom.zeros_like()?)?; // [B,1] int
    let row_nonzero_f32 = row_nonzero.to_dtype(DType::F32)?; // [B,1]
    Ok(probs.broadcast_mul(&row_nonzero_f32)?)
}

/// tril - lower triangular part of the matrix
/// - `t`:    sequence length
/// - `device`: device to create the mask on
/// Returns [t, t] mask; masked entries are exactly 0.
#[instrument(level = "trace", fields(t))]
pub fn triu2d(t: usize, device: &Device) -> Result<Tensor> {
    let mask = (0..t)
        .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
        .collect::<Vec<_>>();
    let mask = Tensor::from_slice(&mask, (t, t), &device)?;
    Ok(mask)
}

#[instrument(level = "trace", fields(on_true, shape = ?mask.shape()), skip(on_false, mask))]
pub fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?
        .to_dtype(on_false.dtype())?
        .broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

/// KL(P || Q) where `p` and `q` are probability distributions along `dim`.
/// Uses PyTorch-style "batchmean": sum along `dim`, then mean across batches.
#[instrument(level = "debug", fields(p_shape = ?p.shape(), q_shape = ?q.shape()), skip(p, q, dim))]
pub fn kl_div<D: candle_core::shape::Dim>(p: &Tensor, q: &Tensor, dim: D) -> Result<Tensor> {
    if p.shape() != q.shape() {
        return Err(anyhow::Error::msg(
            "kl_div: p and q must have identical shapes",
        ));
    }

    let p_stable = (p + f32::EPSILON as f64)?;
    let q_stable = (q + f32::EPSILON as f64)?;

    let log_p = p_stable.log()?;
    let log_q = q_stable.log()?;
    let elem = (p_stable * (log_p - log_q)?)?;
    Ok(elem.sum(dim)?.mean_all()?)
}

/// Numerically stable binary cross entropy loss.
/// - `logits`: `[B, ...]` raw predictions (not probabilities)
/// - `targets`: `[B, ...]` binary targets (0 or 1)
/// Returns scalar mean loss.
///
/// Numerically stable computation using the log-sum-exp trick:
/// ```text
/// BCE = -[y * log(σ(x)) + (1-y) * log(1-σ(x))]
/// where σ(x) = 1/(1+exp(-x))
/// ```
///
/// This can be rewritten as:
/// ```text
/// BCE = max(x, 0) - x*y + log(1 + exp(-|x|))
/// ```
/// This formulation avoids overflow/underflow issues.
#[instrument(level = "debug", fields(logits_shape = ?logits.shape(), targets_shape = ?targets.shape()), skip(logits, targets))]
pub fn binary_cross_entropy_with_logits_stable(
    logits: &Tensor,
    targets: &Tensor,
) -> Result<Tensor> {
    if logits.shape() != targets.shape() {
        return Err(anyhow::Error::msg(
            "binary_cross_entropy_with_logits: logits and targets must have identical shapes",
        ));
    }

    // max(x, 0)
    let zero = logits.zeros_like()?;
    let max_val = logits.maximum(&zero)?;

    // log(1 + exp(-|x|))
    let abs_logits = logits.abs()?;
    let neg_abs_logits = abs_logits.neg()?;
    let log_exp_term = neg_abs_logits
        .exp()?
        .add(&Tensor::ones_like(&neg_abs_logits)?)?
        .log()?;

    // x * y
    let xy_term = logits.mul(&targets.to_dtype(DType::F32)?)?;

    // BCE = max(x, 0) - x*y + log(1 + exp(-|x|))
    let loss = (max_val - xy_term)?.add(&log_exp_term)?;

    Ok(loss.mean_all()?)
}

pub fn clip_grad_norm(vars: &[Var], grad_store: &mut GradStore, max_norm: f64) -> Result<f64> {
    let norm = get_l2_norm(grad_store)? as f64;
    let scale = if norm > max_norm {
        max_norm / (norm + f32::EPSILON as f64)
    } else {
        1.0
    };

    // Scale gradients - still need vars for the mutable operations
    let grad_store_mutex = RwLock::new(grad_store);
    vars.par_iter().try_for_each(|var| -> Result<()> {
        let read_guard = grad_store_mutex.read().unwrap();
        if let Some(grad) = read_guard.get_id(var.id()) {
            let scaled_grad = grad.affine(scale, 0.0)?;
            drop(read_guard);
            let mut write_guard = grad_store_mutex.write().unwrap();
            write_guard.insert(var, scaled_grad);
        }
        Ok(())
    })?;

    Ok(norm)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use rand::seq::SliceRandom;

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
        for row in mask_as_vec.iter() {
            assert_eq!(row, &expected_mask);
        }

        Ok(())
    }

    #[test]
    fn test_tril2d() -> Result<()> {
        let mask = triu2d(3, &Device::Cpu)?;
        let mask_as_vec = mask.to_vec2::<u8>().unwrap();
        let expected_mask = vec![vec![0, 1, 1], vec![0, 0, 1], vec![0, 0, 0]];
        assert_eq!(mask_as_vec, expected_mask);
        Ok(())
    }

    #[test]
    fn test_binary_cross_entropy_with_logits() -> Result<()> {
        let dev = Device::Cpu;

        // Test case 1: Perfect predictions (logits = +∞ for target=1, -∞ for target=0)
        // Using large but finite values to approximate
        let logits = Tensor::from_vec(vec![10.0f32, -10.0, 10.0, -10.0], (2, 2), &dev)?;
        let targets = Tensor::from_vec(vec![1.0f32, 0.0, 1.0, 0.0], (2, 2), &dev)?;
        let loss = binary_cross_entropy_with_logits_stable(&logits, &targets)?;
        let loss_val = loss.to_scalar::<f32>()?;
        assert!(
            loss_val < 1e-4,
            "Perfect predictions should have near-zero loss"
        );

        // Test case 2: Worst predictions (logits = -∞ for target=1, +∞ for target=0)
        let logits = Tensor::from_vec(vec![-10.0f32, 10.0, -10.0, 10.0], (2, 2), &dev)?;
        let targets = Tensor::from_vec(vec![1.0f32, 0.0, 1.0, 0.0], (2, 2), &dev)?;
        let loss = binary_cross_entropy_with_logits_stable(&logits, &targets)?;
        let loss_val = loss.to_scalar::<f32>()?;
        assert!(loss_val > 9.0, "Worst predictions should have high loss");

        // Test case 3: Neutral predictions (logits = 0)
        let logits = Tensor::zeros((2, 2), DType::F32, &dev)?;
        let targets = Tensor::from_vec(vec![1.0f32, 0.0, 1.0, 0.0], (2, 2), &dev)?;
        let loss = binary_cross_entropy_with_logits_stable(&logits, &targets)?;
        let loss_val = loss.to_scalar::<f32>()?;
        let expected = (2.0f32).ln(); // log(2) ≈ 0.693
        assert!(
            (loss_val - expected).abs() < 1e-6,
            "Neutral predictions should have log(2) loss"
        );

        // Test case 4: Test numerical stability with extreme values
        let logits = Tensor::from_vec(vec![100.0f32, -100.0], (2,), &dev)?;
        let targets = Tensor::from_vec(vec![1.0f32, 0.0], (2,), &dev)?;
        let loss = binary_cross_entropy_with_logits_stable(&logits, &targets)?;
        let loss_val = loss.to_scalar::<f32>()?;
        assert!(
            loss_val.is_finite() && loss_val < 1e-4,
            "Should handle extreme values without overflow"
        );

        Ok(())
    }

    #[test]
    fn test_masked_softmax_2d_basic() -> Result<()> {
        let dev = Device::Cpu;

        // x: [1, 2, 3, 4], mask: keep indices 0,1,3; mask out index 2
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &dev)?;
        let mask = Tensor::from_vec(vec![1u8, 1, 0, 1], (1, 4), &dev)?;

        let probs = masked_softmax_2d(&x, &mask)?;
        let v = probs.to_vec2::<f32>().unwrap();

        let e1 = (1.0f32 - 4.0).exp(); // exp(-3)
        let e2 = (2.0f32 - 4.0).exp(); // exp(-2)
        let e4 = (4.0f32 - 4.0).exp(); // 1
        let den = e1 + e2 + e4;
        let expected = [vec![e1 / den, e2 / den, 0.0, e4 / den]];

        for (row_v, row_e) in v.iter().zip(expected.iter()) {
            for (a, b) in row_v.iter().zip(row_e.iter()) {
                assert!((a - b).abs() < 1e-6, "{} vs {}", a, b);
            }
        }

        Ok(())
    }

    #[test]
    fn test_masked_softmax_2d_all_masked_row() -> Result<()> {
        let dev = Device::Cpu;

        let x = Tensor::from_vec(vec![0.5f32, -1.0, 2.0], (1, 3), &dev)?;
        let mask = Tensor::from_vec(vec![0u8, 0, 0], (1, 3), &dev)?;

        let probs = masked_softmax_2d(&x, &mask)?;
        let v = probs.to_vec2::<f32>().unwrap();
        assert_eq!(v, vec![vec![0.0, 0.0, 0.0]]);

        Ok(())
    }

    #[test]
    fn test_masked_softmax_2d_all_zero_inputs_row() -> Result<()> {
        let dev = Device::Cpu;

        // Even with all mask=1, a row with all-zero inputs should output all zeros
        let x = Tensor::zeros((1, 4), DType::F32, &dev)?;
        let mask = Tensor::from_vec(vec![1u8, 1, 1, 1], (1, 4), &dev)?;

        let probs = masked_softmax_2d(&x, &mask)?;
        let v = probs.to_vec2::<f32>().unwrap();
        assert_eq!(v, vec![vec![0.0, 0.0, 0.0, 0.0]]);

        Ok(())
    }

    #[test]
    fn test_masked_softmax_2d_shape_mismatch_errors() -> Result<()> {
        let dev = Device::Cpu;

        let x = Tensor::zeros((2, 3), DType::F32, &dev)?;
        let mask = Tensor::zeros((2, 2), DType::U8, &dev)?;

        let res = masked_softmax_2d(&x, &mask);
        assert!(res.is_err());

        Ok(())
    }

    #[test]
    fn test_scatter_set() -> Result<()> {
        let dev = Device::Cpu;

        // This test demonstrates permuting the rows of a matrix using scatter_set.
        // Source matrix: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        let source = Tensor::arange(0f32, 9., &dev)?.reshape((3, 3))?;

        // The tensor to be modified in-place, starts as zeros.
        let x = Tensor::zeros((3, 3), DType::F32, &dev)?;

        // We want to permute rows: 0->1, 1->2, 2->0. The indices tensor specifies
        // the destination row for each element of the source.
        let indices = Tensor::from_vec(vec![1u32, 2, 0], (3,), &dev)?
            .reshape((3, 1))?
            .broadcast_as((3, 3))?
            .contiguous()?;

        // Perform scatter_set along dimension 0 (rows).
        x.scatter_set(&indices, &source, 0)?;

        let y_vec = x.to_vec2::<f32>()?;
        // Expected permuted matrix: [[6, 7, 8], [0, 1, 2], [3, 4, 5]]
        let expected = vec![
            vec![6.0, 7.0, 8.0], // Original row 2
            vec![0.0, 1.0, 2.0], // Original row 0
            vec![3.0, 4.0, 5.0], // Original row 1
        ];
        assert_eq!(y_vec, expected);

        fn simple_gather(x: Vec<usize>, indices: Vec<usize>) -> Vec<usize> {
            let mut y = Vec::with_capacity(indices.len());
            for idx in indices {
                y.push(x[idx]);
            }
            y
        }
        /// Reorder `x` in place so that after[i] = before[indices[i]].
        /// - `indices` must be a permutation of 0..n-1
        /// - Modifies `indices` (marks visited by adding n)
        /// - O(1) aux, T: Copy
        pub fn gather_permute_in_place<T: Copy>(
            x: &mut [T],
            indices: &mut [usize],
        ) -> Result<(), &'static str> {
            let n = x.len();
            if indices.len() != n {
                return Err("indices.len() must equal x.len()");
            }
            if n == 0 {
                return Ok(());
            }
            if n > usize::MAX / 2 {
                return Err("n too large for add-n marking");
            }

            // Verify indices is a valid permutation
            let mut seen = vec![false; n];
            for &v in indices.iter() {
                if v >= n || std::mem::replace(&mut seen[v], true) {
                    return Err("indices is not a valid permutation");
                }
            }

            let nn = n;

            let mut i = 0usize;
            while i < n {
                // already visited/placed?
                if indices[i] >= nn {
                    i += 1;
                    continue;
                }

                // follow the cycle starting at i
                let mut cur = i;
                let tmp = x[i]; // displaced value from position i

                loop {
                    let src = indices[cur]; // src < n (by contract)
                    indices[cur] = src + nn; // mark visited

                    // if the next link is visited, close the cycle
                    if indices[src] >= nn {
                        x[cur] = tmp;
                        break;
                    } else {
                        x[cur] = x[src];
                        cur = src;
                    }
                }

                i += 1;
            }

            Ok(())
        }

        // Test that both implementations produce the same result
        let mut rng = rand::rng();
        let num_samples = 100_000;
        let vec_size = 128;
        let _ = (0..num_samples)
            .map(|_| {
                let x = (0..vec_size)
                    .map(|_| rand::random::<u32>() as usize)
                    .collect::<Vec<_>>();
                let mut indices = (0..vec_size).collect::<Vec<_>>();
                indices.shuffle(&mut rng);

                let x_simple_gather = x.clone();
                let indices_simple_gather = indices.clone();
                let mut x_permute_in_place_fast = x.clone();
                let mut indices_permute_in_place_fast = indices.clone();
                let output_simple_gather = simple_gather(x_simple_gather, indices_simple_gather);
                let _ = gather_permute_in_place(
                    &mut x_permute_in_place_fast,
                    &mut indices_permute_in_place_fast,
                );
                assert_eq!(output_simple_gather, x_permute_in_place_fast);
            })
            .count();

        Ok(())
    }
}
