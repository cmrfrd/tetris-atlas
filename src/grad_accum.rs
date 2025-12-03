use std::collections::HashMap;

use candle_core::{DType, Tensor, Var};
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
            let sum_sq = grad
                .to_dtype(DType::F32)?
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()?;
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
                let scaled_grad = grad.detach().affine(scale, 0.0)?;

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
        clip_grad_max_value: Option<f64>,
    ) -> Result<bool> {
        if !self.should_step() {
            return Ok(false);
        }

        if params.is_empty() {
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
            let _ = clip_grad_norm(
                &params,
                &mut grad_store,
                clip_grad_max_norm,
                clip_grad_max_value,
            )?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, Var};

    /// Helper to create a simple loss and get its gradients
    /// The gradient for each element will be `scale` (derivative of sum(x * scale) = scale)
    fn create_grads_for_var(var: &Var, scale: f64) -> Result<candle_core::backprop::GradStore> {
        // Create a simple loss: sum of (var * scale)
        let loss = (var.as_tensor() * scale)?.sum_all()?;
        Ok(loss.backward()?)
    }

    #[test]
    fn test_get_l2_norm_single_tensor() -> Result<()> {
        let dev = Device::Cpu;
        // Use loss = 0.5 * sum(var^2) so gradient = var
        // For var = [3, 4], gradient = [3, 4], L2 norm = 5
        let var = Var::from_tensor(&Tensor::from_vec(vec![3.0f32, 4.0], (2,), &dev)?)?;
        let loss = (var.as_tensor().sqr()?.sum_all()? * 0.5)?;
        let grads = loss.backward()?;

        let norm = get_l2_norm(&grads)?;
        // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
        assert!((norm - 5.0).abs() < 1e-5, "Expected 5.0, got {}", norm);
        Ok(())
    }

    #[test]
    fn test_get_l2_norm_multiple_tensors() -> Result<()> {
        let dev = Device::Cpu;
        // Create two tensors
        let var1 = Var::from_tensor(&Tensor::from_vec(vec![1.0f32, 2.0], (2,), &dev)?)?;
        let var2 = Var::from_tensor(&Tensor::from_vec(vec![2.0f32, 0.0], (2,), &dev)?)?;

        // Create combined loss using sum, so gradients are all 1s
        let loss = (var1.as_tensor().sum_all()? + var2.as_tensor().sum_all()?)?;
        let grads = loss.backward()?;

        let norm = get_l2_norm(&grads)?;
        // Gradients are all 1s (derivative of sum), so for 4 elements: sqrt(1+1+1+1) = 2
        assert!((norm - 2.0).abs() < 1e-5, "Expected 2.0, got {}", norm);
        Ok(())
    }

    #[test]
    fn test_get_l2_norm_zeros() -> Result<()> {
        let dev = Device::Cpu;
        // Use loss = 0.5 * sum(var^2), so gradient = var = zeros
        let var = Var::zeros((3, 3), DType::F32, &dev)?;
        let loss = (var.as_tensor().sqr()?.sum_all()? * 0.5)?;
        let grads = loss.backward()?;

        let norm = get_l2_norm(&grads)?;
        assert!((norm - 0.0).abs() < 1e-5, "Expected 0.0, got {}", norm);
        Ok(())
    }

    #[test]
    fn test_gradient_accumulator_new() {
        let accum = GradientAccumulator::new(4);
        assert_eq!(accum.accumulation_steps, 4);
        assert_eq!(accum.step_count, 0);
        assert!(accum.accumulated_grads.is_empty());
    }

    #[test]
    fn test_gradient_accumulator_should_step() {
        let mut accum = GradientAccumulator::new(2);
        assert!(!accum.should_step());

        accum.step_count = 1;
        assert!(!accum.should_step());

        accum.step_count = 2;
        assert!(accum.should_step());

        accum.step_count = 3;
        assert!(accum.should_step());
    }

    #[test]
    fn test_gradient_accumulator_reset() -> Result<()> {
        let dev = Device::Cpu;
        let var = Var::from_tensor(&Tensor::ones((2, 2), DType::F32, &dev)?)?;
        let grads = create_grads_for_var(&var, 1.0)?;

        let mut accum = GradientAccumulator::new(4);
        accum.accumulate(grads, &[var])?;

        assert!(!accum.accumulated_grads.is_empty());
        assert_eq!(accum.step_count, 1);

        accum.reset();

        assert!(accum.accumulated_grads.is_empty());
        assert_eq!(accum.step_count, 0);
        Ok(())
    }

    #[test]
    fn test_gradient_accumulator_accumulate_single() -> Result<()> {
        let dev = Device::Cpu;
        let var = Var::from_tensor(&Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (4,), &dev)?)?;
        let grads = create_grads_for_var(&var, 2.0)?;

        let mut accum = GradientAccumulator::new(4);
        accum.accumulate(grads, &[var.clone()])?;

        assert_eq!(accum.step_count, 1);
        assert!(accum.accumulated_grads.contains_key(&var.id()));

        // Gradient should be scaled by 1/4 (accumulation_steps)
        let accumulated = accum.accumulated_grads.get(&var.id()).unwrap();
        let values = accumulated.to_vec1::<f32>()?;
        // Original gradient is 2.0 for each element (scale factor), scaled by 1/4 = 0.5
        for v in values {
            assert!((v - 0.5).abs() < 1e-5, "Expected 0.5, got {}", v);
        }
        Ok(())
    }

    #[test]
    fn test_gradient_accumulator_accumulate_multiple() -> Result<()> {
        let dev = Device::Cpu;
        let var = Var::from_tensor(&Tensor::ones((2, 2), DType::F32, &dev)?)?;

        let mut accum = GradientAccumulator::new(4);

        // Accumulate 4 identical gradients
        for _ in 0..4 {
            let grads = create_grads_for_var(&var, 1.0)?;
            accum.accumulate(grads, &[var.clone()])?;
        }

        assert_eq!(accum.step_count, 4);
        assert!(accum.should_step());

        // After 4 accumulations of gradients of 1.0, each scaled by 1/4:
        // Total = 4 * (1.0 * 1/4) = 1.0
        let accumulated = accum.accumulated_grads.get(&var.id()).unwrap();
        let values = accumulated.to_vec2::<f32>()?;
        for row in values {
            for v in row {
                assert!((v - 1.0).abs() < 1e-5, "Expected 1.0, got {}", v);
            }
        }
        Ok(())
    }

    #[test]
    fn test_gradient_accumulator_accumulate_different_gradients() -> Result<()> {
        let dev = Device::Cpu;
        let var = Var::from_tensor(&Tensor::ones((2,), DType::F32, &dev)?)?;

        let mut accum = GradientAccumulator::new(2);

        // First gradient: scale 2.0
        let grads1 = create_grads_for_var(&var, 2.0)?;
        accum.accumulate(grads1, &[var.clone()])?;

        // Second gradient: scale 4.0
        let grads2 = create_grads_for_var(&var, 4.0)?;
        accum.accumulate(grads2, &[var.clone()])?;

        // Expected: (2.0/2 + 4.0/2) = 1.0 + 2.0 = 3.0
        let accumulated = accum.accumulated_grads.get(&var.id()).unwrap();
        let values = accumulated.to_vec1::<f32>()?;
        for v in values {
            assert!((v - 3.0).abs() < 1e-5, "Expected 3.0, got {}", v);
        }
        Ok(())
    }

    #[test]
    fn test_gradient_accumulator_apply_and_reset_not_ready() -> Result<()> {
        let dev = Device::Cpu;
        let var = Var::from_tensor(&Tensor::ones((2,), DType::F32, &dev)?)?;
        let params = vec![var.clone()];

        let mut accum = GradientAccumulator::new(4);

        // Only accumulate once (not enough steps)
        let grads = create_grads_for_var(&var, 1.0)?;
        accum.accumulate(grads, &params)?;

        let mut optimizer = candle_nn::AdamW::new(params.clone(), Default::default())?;
        let applied = accum.apply_and_reset(&mut optimizer, &params, None, None)?;

        assert!(
            !applied,
            "Should not apply when step_count < accumulation_steps"
        );
        assert_eq!(accum.step_count, 1, "Step count should remain unchanged");
        Ok(())
    }

    #[test]
    fn test_gradient_accumulator_apply_and_reset_ready() -> Result<()> {
        let dev = Device::Cpu;
        let initial_value = Tensor::ones((2,), DType::F32, &dev)?;
        let var = Var::from_tensor(&initial_value)?;
        let params = vec![var.clone()];

        let mut accum = GradientAccumulator::new(2);

        // Accumulate enough steps
        for _ in 0..2 {
            let grads = create_grads_for_var(&var, 1.0)?;
            accum.accumulate(grads, &params)?;
        }

        assert!(accum.should_step());

        let mut optimizer = candle_nn::AdamW::new(params.clone(), Default::default())?;
        let original_value = var.as_tensor().to_vec1::<f32>()?;

        let applied = accum.apply_and_reset(&mut optimizer, &params, None, None)?;

        assert!(
            applied,
            "Should apply when step_count >= accumulation_steps"
        );
        assert_eq!(accum.step_count, 0, "Step count should be reset to 0");
        assert!(
            accum.accumulated_grads.is_empty(),
            "Accumulated grads should be empty after apply"
        );

        // Verify the variable was updated
        let new_value = var.as_tensor().to_vec1::<f32>()?;
        assert_ne!(original_value, new_value, "Variable should be updated");
        Ok(())
    }

    #[test]
    fn test_gradient_accumulator_apply_with_grad_clipping() -> Result<()> {
        let dev = Device::Cpu;
        let var = Var::from_tensor(&Tensor::from_vec(vec![1.0f32, 1.0], (2,), &dev)?)?;
        let params = vec![var.clone()];

        let mut accum = GradientAccumulator::new(1);

        // Large gradient
        let grads = create_grads_for_var(&var, 100.0)?;
        accum.accumulate(grads, &params)?;

        let mut optimizer = candle_nn::AdamW::new(params.clone(), Default::default())?;
        let applied = accum.apply_and_reset(&mut optimizer, &params, Some(1.0), None)?;

        assert!(applied);
        Ok(())
    }

    #[test]
    fn test_gradient_accumulator_empty_params() -> Result<()> {
        let params: Vec<Var> = vec![];
        let mut accum = GradientAccumulator::new(1);
        accum.step_count = 1; // Fake being ready

        let mut optimizer = candle_nn::AdamW::new(vec![], Default::default())?;
        let applied = accum.apply_and_reset(&mut optimizer, &params, None, None)?;

        assert!(!applied, "Should return false for empty params");
        Ok(())
    }

    #[test]
    fn test_gradient_accumulator_multiple_params() -> Result<()> {
        let dev = Device::Cpu;
        let var1 = Var::from_tensor(&Tensor::ones((2,), DType::F32, &dev)?)?;
        let var2 = Var::from_tensor(&Tensor::ones((3,), DType::F32, &dev)?)?;
        let params = vec![var1.clone(), var2.clone()];

        let mut accum = GradientAccumulator::new(2);

        // Create gradients for both vars
        for _ in 0..2 {
            let loss = (var1.as_tensor().sum_all()? + var2.as_tensor().sum_all()?)?;
            let grads = loss.backward()?;
            accum.accumulate(grads, &params)?;
        }

        assert!(accum.should_step());
        assert!(accum.accumulated_grads.contains_key(&var1.id()));
        assert!(accum.accumulated_grads.contains_key(&var2.id()));

        let mut optimizer = candle_nn::AdamW::new(params.clone(), Default::default())?;
        let applied = accum.apply_and_reset(&mut optimizer, &params, None, None)?;

        assert!(applied);
        assert!(accum.accumulated_grads.is_empty());
        Ok(())
    }

    #[test]
    fn test_gradient_accumulator_preserves_averaging() -> Result<()> {
        // Test that accumulation properly averages gradients over N steps
        let dev = Device::Cpu;
        let var = Var::from_tensor(&Tensor::ones((4,), DType::F32, &dev)?)?;

        let accumulation_steps = 8;
        let mut accum = GradientAccumulator::new(accumulation_steps);

        // Accumulate gradients with varying scales
        let scales = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for scale in scales {
            let grads = create_grads_for_var(&var, scale)?;
            accum.accumulate(grads, &[var.clone()])?;
        }

        // Expected average: (1+2+3+4+5+6+7+8)/8 = 36/8 = 4.5
        let accumulated = accum.accumulated_grads.get(&var.id()).unwrap();
        let values = accumulated.to_vec1::<f32>()?;
        for v in values {
            assert!((v - 4.5).abs() < 1e-5, "Expected average of 4.5, got {}", v);
        }
        Ok(())
    }

    #[test]
    fn test_optim_step_trait_candle_adamw() -> Result<()> {
        let dev = Device::Cpu;
        let var = Var::from_tensor(&Tensor::ones((2,), DType::F32, &dev)?)?;
        let params = vec![var.clone()];

        let mut optimizer = candle_nn::AdamW::new(params.clone(), Default::default())?;

        let grads = create_grads_for_var(&var, 1.0)?;
        optimizer.step_like(&grads)?;

        // Just verify it doesn't error - the optimizer internals are tested elsewhere
        Ok(())
    }

    #[test]
    fn test_optim_step_trait_custom_adamw() -> Result<()> {
        let dev = Device::Cpu;
        let var = Var::from_tensor(&Tensor::ones((2,), DType::F32, &dev)?)?;
        let params = vec![var.clone()];

        let mut optimizer = crate::optim::AdamW::new(params.clone(), Default::default())?;

        let grads = create_grads_for_var(&var, 1.0)?;
        optimizer.step_like(&grads)?;

        // Just verify it doesn't error
        Ok(())
    }

    #[test]
    fn test_gradient_accumulator_detaches_gradients() -> Result<()> {
        // Verify that accumulated gradients don't retain computation graph
        let dev = Device::Cpu;
        let var = Var::from_tensor(&Tensor::ones((2, 2), DType::F32, &dev)?)?;

        let mut accum = GradientAccumulator::new(2);

        for _ in 0..2 {
            let grads = create_grads_for_var(&var, 1.0)?;
            accum.accumulate(grads, &[var.clone()])?;
        }

        // The accumulated gradient should not track operations
        let accumulated = accum.accumulated_grads.get(&var.id()).unwrap();
        // Verify we can perform operations without graph issues
        let _ = accumulated.sum_all()?.to_scalar::<f32>()?;
        Ok(())
    }
}
