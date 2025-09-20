use std::collections::HashMap;

use candle_core::{Tensor, Var};
use candle_nn::{AdamW, Optimizer};

use anyhow::Result;

pub fn get_l2_norm(grad_store: &candle_core::backprop::GradStore, params: &[Var]) -> Result<f32> {
    let mut total_squared_sum: f32 = 0.0;
    for param in params {
        let grad = grad_store
            .get(param)
            .ok_or(anyhow::anyhow!("Gradient not found for parameter"))?;
        let sum_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
        total_squared_sum += sum_sq;
    }
    Ok(total_squared_sum.sqrt())
}

/// Gradient accumulator:
/// Accumulates gradients for individual parameters using their tensor IDs as keys
pub struct GradientAccumulator {
    accumulated_grads: HashMap<candle_core::TensorId, Tensor>,
    step_count: usize,
    accumulation_steps: usize,
}

impl GradientAccumulator {
    /// Create a new gradient accumulator
    pub fn new(accumulation_steps: usize) -> Self {
        Self {
            accumulated_grads: HashMap::new(),
            step_count: 0,
            accumulation_steps,
        }
    }

    /// Accumulate gradients from a GradStore by extracting individual tensor gradients
    /// Uses in-place accumulation and immediate scaling to reduce memory usage
    pub fn accumulate(
        &mut self,
        grads: candle_core::backprop::GradStore,
        params: &[Var],
    ) -> Result<(), candle_core::Error> {
        // Scale factor for immediate averaging to prevent gradient explosion
        let scale = 1.0 / self.accumulation_steps as f64;

        for param in params {
            if let Some(grad) = grads.get(param) {
                let param_id = param.id();
                let scaled_grad = grad.affine(scale, 0.0)?;

                match self.accumulated_grads.get_mut(&param_id) {
                    Some(accumulated_grad) => {
                        *accumulated_grad = accumulated_grad.add(&scaled_grad)?;
                    }
                    None => {
                        self.accumulated_grads.insert(param_id, scaled_grad);
                    }
                }
            }
        }

        self.step_count += 1;
        Ok(())
    }

    /// Check if we should apply the accumulated gradients
    pub fn should_step(&self) -> bool {
        self.step_count >= self.accumulation_steps
    }

    /// Apply the accumulated gradients and reset the accumulator
    /// This creates a new GradStore with the accumulated gradients and applies them via the optimizer
    pub fn apply_and_reset(
        &mut self,
        optimizer: &mut AdamW,
        params: &[Var],
    ) -> Result<bool, candle_core::Error> {
        if !self.should_step() {
            return Ok(false);
        }

        // Create a simple dummy computation to get a GradStore
        let dummy_tensor = params[0].zeros_like()?;
        let dummy_loss = dummy_tensor.sum_all()?;
        let mut grad_store = dummy_loss.backward()?;

        // Clear the dummy gradients and insert our accumulated gradients
        grad_store.remove(&params[0]);

        // Insert accumulated gradients for each parameter (already scaled)
        for param in params {
            if let Some(accumulated_grad) = self.accumulated_grads.remove(&param.id()) {
                grad_store.insert(param, accumulated_grad);
            }
        }

        // Apply the optimizer step
        optimizer.step(&grad_store)?;

        // Reset the accumulator state
        self.step_count = 0;
        Ok(true)
    }

    /// Reset the accumulator state
    pub fn reset(&mut self) {
        self.accumulated_grads.clear();
        self.step_count = 0;
    }
}
