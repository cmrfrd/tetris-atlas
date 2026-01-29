use std::{collections::HashMap, path::Path};

use candle_core::{Device, Result, Tensor, Var, safetensors};
use candle_nn::Optimizer;

use crate::checkpointer::Checkpointable;

#[derive(Clone, Debug)]
pub struct ParamsAdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for ParamsAdamW {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

#[derive(Debug)]
pub struct VarAdamW {
    pub var: Var,
    pub first_moment: Var,
    pub second_moment: Var,
}

#[derive(Debug)]
pub struct AdamW {
    pub vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}

impl Optimizer for AdamW {
    type Config = ParamsAdamW;

    fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(VarAdamW {
                    var,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;
            if let Some(g) = grads.get(theta) {
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

impl AdamW {
    pub fn new_lr(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let params = ParamsAdamW {
            lr: learning_rate,
            ..ParamsAdamW::default()
        };
        Self::new(vars, params)
    }

    pub fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    pub fn set_params(&mut self, params: ParamsAdamW) {
        self.params = params;
    }

    /// Read-only access to the current optimization step counter.
    pub fn step_t(&self) -> usize {
        self.step_t
    }

    /// A lightweight state container for optimizer buffers.
    ///
    /// The buffers are ordered to match the order of parameters passed to `new`.
    pub fn export_state(&self) -> Result<AdamWState> {
        let mut first_moments = Vec::with_capacity(self.vars.len());
        let mut second_moments = Vec::with_capacity(self.vars.len());
        for v in &self.vars {
            first_moments.push(v.first_moment.as_tensor().clone());
            second_moments.push(v.second_moment.as_tensor().clone());
        }
        Ok(AdamWState {
            step_t: self.step_t,
            first_moments,
            second_moments,
        })
    }

    /// Restore optimizer buffers from a previously exported state.
    ///
    /// The provided state must match the number, shapes and dtypes of the
    /// parameters managed by this optimizer (same order).
    pub fn import_state(&mut self, state: &AdamWState) -> Result<()> {
        if state.first_moments.len() != self.vars.len()
            || state.second_moments.len() != self.vars.len()
        {
            candle_core::bail!(
                "AdamW.import_state: mismatched number of buffers: expected {}, got {}/{}",
                self.vars.len(),
                state.first_moments.len(),
                state.second_moments.len()
            );
        }

        for (i, v) in self.vars.iter_mut().enumerate() {
            let m_src = &state.first_moments[i];
            let v_src = &state.second_moments[i];

            if m_src.shape() != v.first_moment.shape()
                || v_src.shape() != v.second_moment.shape()
                || m_src.dtype() != v.first_moment.dtype()
                || v_src.dtype() != v.second_moment.dtype()
            {
                candle_core::bail!("AdamW.import_state: shape/dtype mismatch at index {}", i);
            }

            v.first_moment.set(m_src)?;
            v.second_moment.set(v_src)?;
        }

        self.step_t = state.step_t;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct AdamWState {
    pub step_t: usize,
    pub first_moments: Vec<Tensor>,
    pub second_moments: Vec<Tensor>,
}

impl Checkpointable for AdamW {
    fn save_to_path(&self, path: &Path) -> candle_core::Result<()> {
        let state = self.export_state()?;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "step_t".to_string(),
            Tensor::new(state.step_t as i64, &Device::Cpu)?,
        );
        for (i, t) in state.first_moments.iter().enumerate() {
            map.insert(format!("first_moment_{}", i), t.clone());
        }
        for (i, t) in state.second_moments.iter().enumerate() {
            map.insert(format!("second_moment_{}", i), t.clone());
        }
        safetensors::save(&map, path)?;
        Ok(())
    }

    fn load_from_path(&mut self, path: &Path) -> candle_core::Result<()> {
        let first_device = self
            .vars.first()
            .map(|v| v.var.device())
            .unwrap_or(&Device::Cpu);
        let map = candle_core::safetensors::load(path, &first_device)?;

        let step_t = map
            .get("step_t")
            .ok_or_else(|| candle_core::Error::Msg("missing step_t in optimizer state".into()))?
            .to_scalar::<i64>()? as usize;

        let mut first_moments = Vec::with_capacity(self.vars.len());
        let mut second_moments = Vec::with_capacity(self.vars.len());
        for i in 0..self.vars.len() {
            let m_name = format!("first_moment_{}", i);
            let v_name = format!("second_moment_{}", i);
            let m = map
                .get(&m_name)
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!("missing {} in optimizer state", m_name))
                })?
                .clone();
            let v = map
                .get(&v_name)
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!("missing {} in optimizer state", v_name))
                })?
                .clone();
            first_moments.push(m);
            second_moments.push(v);
        }
        let state = AdamWState {
            step_t,
            first_moments,
            second_moments,
        };
        self.import_state(&state)
    }
}
