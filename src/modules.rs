use std::{collections::HashMap, ops::Deref};

use candle_core::{D, DType, Device, IndexOp, Shape, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, Embedding, GroupNorm, Linear, Module, VarBuilder, conv2d, embedding,
    group_norm, linear, linear_no_bias, ops::softmax_last_dim,
};

use anyhow::{Result, ensure};

use crate::{
    ops::{create_orientation_mask, masked_fill, triu2d},
    tensors::{
        TetrisBoardLogitsTensor, TetrisBoardsDistTensor, TetrisBoardsTensor, TetrisContextTensor,
        TetrisPieceOrientationDistTensor, TetrisPieceOrientationLogitsTensor,
        TetrisPieceOrientationTensor, TetrisPiecePlacementDistTensor, TetrisPiecePlacementTensor,
        TetrisPieceTensor,
    },
    tetris::{
        NUM_TETRIS_CELL_STATES, TetrisBoardRaw, TetrisPiece, TetrisPieceOrientation,
        TetrisPiecePlacement,
    },
    wrapped_tensor::WrappedTensor,
};

fn calculate_default_inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / theta.powf(i as f32 / head_dim as f32))
        .collect()
}

#[derive(Clone, Debug)]
pub struct RopeEncodingConfig {
    pub max_position_embeddings: usize,
    pub head_dim: usize,
    pub theta: f32,
}

#[derive(Debug, Clone)]
pub struct RopeEncoding {
    cos: Tensor,
    sin: Tensor,
    max_position_embeddings: usize,
}

impl RopeEncoding {
    pub fn new(config: RopeEncodingConfig, vb: &VarBuilder) -> Result<Self> {
        let dim = config.head_dim;
        let theta = Tensor::new(calculate_default_inv_freq(dim, config.theta), vb.device())?;
        let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, vb.device())?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;

        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let cos = idx_theta.cos()?.to_dtype(DType::F32)?;
        let sin = idx_theta.sin()?.to_dtype(DType::F32)?;
        Ok(Self {
            cos,
            sin,
            max_position_embeddings: config.max_position_embeddings,
        })
    }

    pub fn forward(&self, x: Tensor) -> Result<Tensor> {
        // Expect [B, H, T, D]
        let (_b, _h, seq_len, _d) = x.dims4()?;
        ensure!(
            seq_len <= self.max_position_embeddings,
            "RoPE seq_len {} exceeds max_position_embeddings {}",
            seq_len,
            self.max_position_embeddings
        );
        ensure!(
            seq_len > 0,
            "RoPE seq_len {} must be greater than 0",
            seq_len
        );
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let y = candle_nn::rotary_emb::rope(&x, &cos, &sin)?;
        Ok(y)
    }
}

#[derive(Clone, Debug)]
pub struct CausalSelfAttentionConfig {
    pub d_model: usize,
    pub n_attention_heads: usize,
    pub n_kv_heads: usize,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

#[derive(Debug, Clone)]
pub struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rope: RopeEncoding,

    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
}

impl CausalSelfAttention {
    pub fn init(vb: &VarBuilder, config: &CausalSelfAttentionConfig) -> Result<Self> {
        let size_in = config.d_model;
        ensure!(
            config.d_model % config.n_attention_heads == 0,
            "d_model ({}) must be divisible by n_attention_heads ({})",
            config.d_model,
            config.n_attention_heads
        );
        let size_q = (config.d_model / config.n_attention_heads) * config.n_attention_heads;
        let size_kv = (config.d_model / config.n_attention_heads) * config.n_kv_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;

        let head_dim = config.d_model / config.n_attention_heads;
        ensure!(head_dim % 2 == 0, "head_dim must be even");
        let rope = RopeEncoding::new(
            RopeEncodingConfig {
                max_position_embeddings: config.max_position_embeddings,
                head_dim,
                theta: config.rope_theta,
            },
            &vb,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: config.n_attention_heads,
            num_kv_heads: config.n_kv_heads,
            head_dim: config.d_model / config.n_attention_heads,
            max_position_embeddings: config.max_position_embeddings,
            rope,
        })
    }

    pub fn forward(&self, x: Tensor, with_causal_mask: bool) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len, hidden_dim) = x.dims3()?;
        ensure!(
            hidden_dim == self.num_attention_heads * self.head_dim,
            "Attention expected d_model={}, got {}",
            self.num_attention_heads * self.head_dim,
            hidden_dim
        );
        ensure!(
            seq_len <= self.max_position_embeddings,
            "Attention seq_len {} exceeds max_position_embeddings {}",
            seq_len,
            self.max_position_embeddings
        );
        let dtype = x.dtype();
        let device = x.device();

        let q = self
            .q_proj
            .forward(&x)?
            .reshape(&[batch_size, seq_len, self.num_attention_heads, self.head_dim])?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(&x)?
            .reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(&x)?
            .reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;

        let q = self.rope.forward(q)?;
        let k = self.rope.forward(k)?;

        // repeat kv
        let num_rep = self.num_attention_heads / self.num_kv_heads;
        let k = match num_rep {
            1 => k,
            _ => {
                let (b_sz, n_kv_head, seq_len, head_dim) = k.dims4()?;
                Tensor::cat(&vec![&k; num_rep], 2)?.reshape((
                    b_sz,
                    n_kv_head * num_rep,
                    seq_len,
                    head_dim,
                ))?
            }
        };
        let v = match num_rep {
            1 => v,
            _ => {
                let (b_sz, n_kv_head, seq_len, head_dim) = v.dims4()?;
                Tensor::cat(&vec![&v; num_rep], 2)?.reshape((
                    b_sz,
                    n_kv_head * num_rep,
                    seq_len,
                    head_dim,
                ))?
            }
        };

        // attn_scores: [B, H, T, T]
        let attn_scores = (q.matmul(&k.t()?)? / f64::sqrt(self.head_dim as f64))?;

        // apply causal mask
        let attn_scores = if seq_len == 1 || !with_causal_mask {
            attn_scores
        } else {
            let mask = triu2d(seq_len, &device)?
                .reshape(&[1, 1, seq_len, seq_len])?
                .broadcast_as(attn_scores.shape())?;
            let m = masked_fill(&attn_scores, &mask, -1e9_f32)?;
            m
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scores)?;
        let y = attn_weights
            .matmul(&v.contiguous()?)?
            .to_dtype(dtype)?
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, hidden_dim])?;

        Ok((self.o_proj.forward(&y)?, attn_weights))
    }
}

#[derive(Debug, Clone)]
pub struct MlpConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub output_size: usize,
}

#[derive(Debug, Clone)]
pub struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    pub fn init(vb: &VarBuilder, cfg: &MlpConfig) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let o_size = cfg.output_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, o_size, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        Ok(self.c_proj.forward(&x)?)
    }
}

#[derive(Debug, Clone)]
pub struct DynamicTanhConfig {
    pub alpha_init_value: f64,
    pub normalized_shape: usize,
}

#[derive(Debug, Clone)]
pub struct DynamicTanh {
    pub alpha: Tensor,
    pub weight: Tensor,
    pub bias: Tensor,
    normalized_shape: usize,
    span: tracing::Span,
}

impl DynamicTanh {
    pub fn init(vb: &VarBuilder, cfg: &DynamicTanhConfig) -> Result<DynamicTanh> {
        let span = tracing::span!(tracing::Level::TRACE, "dyn_tanh");
        // Make these parameters (trainable), with constant inits
        let weight = vb.get_with_hints(
            &[cfg.normalized_shape],
            "weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;
        let bias = vb.get_with_hints(
            &[cfg.normalized_shape],
            "bias",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;
        let alpha =
            vb.get_with_hints(&[1], "alpha", candle_nn::Init::Const(cfg.alpha_init_value))?;
        Ok(DynamicTanh {
            alpha,
            weight,
            bias,
            normalized_shape: cfg.normalized_shape,
            span,
        })
    }

    pub fn forward(&self, x: Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        // Expect [B, T, D]
        let (_b, _t, d) = x.dims3()?;
        ensure!(
            d == self.normalized_shape,
            "DynamicTanh expected last dim {}, got {}",
            self.normalized_shape,
            d
        );

        // y = tanh(alpha * x) * weight + bias
        let alpha_scalar = self.alpha.reshape(&[])?; // scalar
        let x_scaled = x.broadcast_mul(&alpha_scalar)?; // [B,T,D]
        let x_act = x_scaled.tanh()?; // [B,T,D]
        let weight_b = self.weight.reshape(&[1, 1, self.normalized_shape])?; // [1,1,D]
        let bias_b = self.bias.reshape(&[1, 1, self.normalized_shape])?; // [1,1,D]
        let y = x_act.broadcast_mul(&weight_b)?.broadcast_add(&bias_b)?;
        Ok(y)
    }
}

#[derive(Debug, Clone)]
pub struct TransformerBlock {
    dyn_tanh_1: DynamicTanh,
    attn: CausalSelfAttention,
    dyn_tanh_2: DynamicTanh,
    mlp: Mlp,
}

#[derive(Debug, Clone)]
pub struct TransformerBlockConfig {
    pub dyn_tanh_config: DynamicTanhConfig,
    pub attn_config: CausalSelfAttentionConfig,
    pub mlp_config: MlpConfig,
}

impl TransformerBlock {
    fn init(vb: &VarBuilder, cfg: &TransformerBlockConfig) -> Result<TransformerBlock> {
        let dyn_tanh_1 = DynamicTanh::init(&vb.pp("dyn_tanh_1"), &cfg.dyn_tanh_config)?;
        let attn = CausalSelfAttention::init(&vb.pp("attn"), &cfg.attn_config)?;
        let dyn_tanh_2 = DynamicTanh::init(&vb.pp("dyn_tanh_2"), &cfg.dyn_tanh_config)?;
        let mlp = Mlp::init(&vb.pp("mlp"), &cfg.mlp_config)?;
        Ok(TransformerBlock {
            dyn_tanh_1,
            attn,
            dyn_tanh_2,
            mlp,
        })
    }

    fn forward(&self, x: Tensor, with_causal_mask: bool) -> Result<Tensor> {
        let (_b, _t, d) = x.dims3()?;
        ensure!(
            d == self.attn.head_dim * self.attn.num_attention_heads,
            "Block expected hidden dim {}, got {}",
            self.attn.head_dim * self.attn.num_attention_heads,
            d
        );

        // x = x + attention(norm(x))
        let normed_x = self.dyn_tanh_1.forward(x.clone())?;
        let (attn_out, _) = self.attn.forward(normed_x, with_causal_mask)?;
        let x = (x + attn_out)?;

        // x = x + mlp(norm(x))
        let normed_x = self.dyn_tanh_2.forward(x.clone())?;
        let mlp_out = self.mlp.forward(&normed_x)?;
        let x = (x + mlp_out)?;

        Ok(x)
    }
}

#[derive(Debug, Clone)]
pub struct TransformerBody {
    blocks: Vec<TransformerBlock>,
    dyn_tanh: DynamicTanh,
}

#[derive(Debug, Clone)]
pub struct TransformerBodyConfig {
    pub blocks_config: TransformerBlockConfig,
    pub num_blocks: usize,
    pub dyn_tanh_config: DynamicTanhConfig,
}

impl TransformerBody {
    pub fn init(vb: &VarBuilder, cfg: &TransformerBodyConfig) -> Result<TransformerBody> {
        let blocks = (0..cfg.num_blocks)
            .map(|i| TransformerBlock::init(&vb.pp(&format!("block_{}", i)), &cfg.blocks_config))
            .collect::<Result<Vec<TransformerBlock>>>()?;
        let dyn_tanh = DynamicTanh::init(&vb.pp("dyn_tanh"), &cfg.dyn_tanh_config)?;
        Ok(TransformerBody { blocks, dyn_tanh })
    }

    pub fn forward(&self, x: Tensor, with_causal_mask: bool) -> Result<Tensor> {
        let mut x = x;
        for block in self.blocks.iter() {
            x = block.forward(x, with_causal_mask)?;
        }
        let x = self.dyn_tanh.forward(x)?;
        Ok(x)
    }
}

#[derive(Debug, Clone)]
pub struct ConvBlockSpec {
    pub out_channels: usize,
    pub kernel_size: usize,
    pub conv_cfg: Conv2dConfig,
    pub gn_groups: usize, // must divide out_channels
}

#[derive(Debug, Clone)]
pub struct ConvEncoderConfig {
    pub in_channels: usize,
    pub input_hw: (usize, usize),
    pub blocks: Vec<ConvBlockSpec>,
    pub mlp: MlpConfig,
}

#[derive(Debug, Clone)]
struct ConvBlock {
    conv: Conv2d,
    gn: GroupNorm,
}

impl ConvBlock {
    fn init(
        vb: &VarBuilder,
        in_ch: usize,
        spec: &ConvBlockSpec,
        idx: usize,
    ) -> Result<(Self, usize)> {
        assert!(
            spec.out_channels % spec.gn_groups == 0,
            "gn_groups must divide out_channels"
        );
        let conv = conv2d(
            in_ch,
            spec.out_channels,
            spec.kernel_size,
            spec.conv_cfg,
            vb.pp(&format!("blocks_{idx}_conv")),
        )?;
        let gn = group_norm(
            spec.gn_groups,
            spec.out_channels,
            1e-8,
            vb.pp(&format!("blocks_{idx}_gn")),
        )?;
        Ok((Self { conv, gn }, spec.out_channels))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.gn.forward(&x)?;
        Ok(candle_nn::ops::silu(&x)?)
    }
}

#[derive(Debug, Clone)]
pub struct ConvEncoder {
    blocks: Vec<ConvBlock>,
    flatten_dim: usize,
    mlp: Mlp,
}

impl ConvEncoder {
    pub fn init(vb: &VarBuilder, cfg: &ConvEncoderConfig) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.blocks.len());
        let mut in_ch = cfg.in_channels;
        for (i, spec) in cfg.blocks.iter().enumerate() {
            let (b, out_ch) = ConvBlock::init(vb, in_ch, spec, i)?;
            blocks.push(b);
            in_ch = out_ch;
        }

        // infer flatten dim using vb's device/dtype
        let (h, w) = cfg.input_hw;
        let mut y = Tensor::zeros((1, cfg.in_channels, h, w), vb.dtype(), vb.device())?;
        for b in &blocks {
            y = b.forward(&y)?;
        }
        let y = y.flatten_from(1)?;
        let flatten_dim = y.dims()[1];

        assert_eq!(
            flatten_dim, cfg.mlp.hidden_size,
            "flatten_dim ({flatten_dim}) must equal mlp.hidden_size ({})",
            cfg.mlp.hidden_size
        );

        let mlp = Mlp::init(&vb.pp("mlp"), &cfg.mlp)?;
        Ok(Self {
            blocks,
            flatten_dim,
            mlp,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = x.clone();
        for b in &self.blocks {
            y = b.forward(&y)?;
        }
        let y = y.flatten_from(1)?;
        self.mlp.forward(&y)
    }

    pub fn flatten_dim(&self) -> usize {
        self.flatten_dim
    }
}

#[derive(Debug, Clone)]
pub struct FilmConfig {
    pub cond_dim: usize,
    pub feat_dim: usize, // D of x: [B, D]
    pub hidden: usize,
}

#[derive(Debug, Clone)]
pub struct Film {
    proj1: Linear, // cond_dim -> hidden
    proj2: Linear, // hidden -> 2*feat_dim
    cond_dim: usize,
    feat_dim: usize,
}

impl Film {
    pub fn init(vb: &VarBuilder, cfg: &FilmConfig) -> Result<Self> {
        let proj1 = linear(cfg.cond_dim, cfg.hidden, vb.pp("proj1"))?;
        let proj2 = linear(cfg.hidden, 2 * cfg.feat_dim, vb.pp("proj2"))?;
        Ok(Self {
            proj1,
            proj2,
            cond_dim: cfg.cond_dim,
            feat_dim: cfg.feat_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        assert!(
            x.dims().len() == 2 && cond.dims().len() == 2,
            "FiLM expects [B,D] and [B,cond]"
        );
        let (bx, dx) = (x.dims()[0], x.dims()[1]);
        let (bc, dc) = (cond.dims()[0], cond.dims()[1]);
        assert_eq!(bx, bc, "batch mismatch");
        assert_eq!(dx, self.feat_dim, "x dim != feat_dim");
        assert_eq!(dc, self.cond_dim, "cond dim != cond_dim");

        let h = candle_nn::ops::silu(&self.proj1.forward(cond)?)?;
        let gb = self.proj2.forward(&h)?; // [B, 2*D]
        let gamma = gb.narrow(1, 0, self.feat_dim)?; // [B, D]
        let beta = gb.narrow(1, self.feat_dim, self.feat_dim)?; // [B, D]
        let y = ((x * (&gamma + 1.0f64)?)? + &beta)?;
        Ok(y)
    }
}
