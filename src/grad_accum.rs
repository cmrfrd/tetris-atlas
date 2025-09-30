use std::collections::HashMap;

use candle_core::{Tensor, Var};
use candle_nn::Optimizer;

use anyhow::Result;
use rayon::prelude::*;

use crate::ops::clip_grad_norm;

pub fn get_l2_norm(grad_store: &candle_core::backprop::GradStore) -> Result<f32> {
    let total_squared_sum: f32 = grad_store
        .get_ids()
        .par_bridge()
        .map(|param_id| -> Result<f32> {
            let grad = grad_store
                .get_id(*param_id)
                .ok_or(anyhow::anyhow!("Gradient not found for parameter"))?;
            let sum_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
            Ok(sum_sq)
        })
        .try_reduce(|| 0.0f32, |acc, x| Ok(acc + x))?;

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
                // Detach before scaling so we don't retain the forward graph.
                let scaled_grad = grad.detach().affine(scale, 0.0)?.detach();

                match self.accumulated_grads.get_mut(&param_id) {
                    Some(accumulated_grad) => {
                        // Ensure the accumulator never carries a grad tape across steps.
                        *accumulated_grad = accumulated_grad.add(&scaled_grad)?.detach();
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
    pub fn apply_and_reset<O: OptimStep>(
        &mut self,
        optimizer: &mut O,
        params: &[Var],
        clip_grad_max_norm: Option<f64>,
    ) -> Result<bool> {
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

        if let Some(clip_grad_max_norm) = clip_grad_max_norm {
            let _ = clip_grad_norm(&params, &mut grad_store, clip_grad_max_norm)?;
        }

        // Apply the optimizer step
        optimizer.step_like(&grad_store)?;

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

/// Small abstraction to allow using either the built-in candle AdamW or the local AdamW.
pub trait OptimStep {
    fn step_like(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()>;
}

impl OptimStep for candle_nn::AdamW {
    fn step_like(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        Ok(candle_nn::Optimizer::step(self, grads)?)
    }
}

impl OptimStep for crate::optim::AdamW {
    fn step_like(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        Ok(Optimizer::step(self, grads)?)
    }
}
