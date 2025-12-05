use candle_core::{D, Device, IndexOp, Tensor};
use candle_nn::Conv2dConfig as CandleConv2dConfig;
use candle_nn::{Conv2d, GroupNorm, Linear, Module, VarBuilder, conv2d, group_norm, linear};

use anyhow::{Result, ensure};
use image::{Rgb, RgbImage};
use serde::{Deserialize, Serialize};

use crate::fdtype;
use crate::ops::{masked_fill, triu2d};

fn calculate_default_inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / theta.powf(i as f32 / head_dim as f32))
        .collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
            .to_dtype(fdtype())?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;

        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let cos = idx_theta.cos()?.to_dtype(fdtype())?;
        let sin = idx_theta.sin()?.to_dtype(fdtype())?;
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
        let cos = self.cos.narrow(0, 0, seq_len)?.contiguous()?;
        let sin = self.sin.narrow(0, 0, seq_len)?.contiguous()?;
        let y = candle_nn::rotary_emb::rope(&x, &cos, &sin)?;
        Ok(y)
    }

    pub fn forward_with_pos_emb(&self, x: Tensor, pos_emb: usize) -> Result<Tensor> {
        let cos = self.cos.narrow(0, pos_emb, 1)?.contiguous()?;
        let sin = self.sin.narrow(0, pos_emb, 1)?.contiguous()?;
        let y = candle_nn::rotary_emb::rope(&x, &cos, &sin)?;
        Ok(y)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
            config.d_model.is_multiple_of(config.n_attention_heads),
            "d_model ({}) must be divisible by n_attention_heads ({})",
            config.d_model,
            config.n_attention_heads
        );
        ensure!(
            config.n_attention_heads.is_multiple_of(config.n_kv_heads),
            "n_attention_heads ({}) must be divisible by n_kv_heads ({})",
            config.n_attention_heads,
            config.n_kv_heads
        );
        let size_q = (config.d_model / config.n_attention_heads) * config.n_attention_heads;
        let size_kv = (config.d_model / config.n_attention_heads) * config.n_kv_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;

        let head_dim = config.d_model / config.n_attention_heads;
        ensure!(head_dim.is_multiple_of(2), "head_dim must be even");
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

        // repeat kv along the head dimension
        let num_rep = self.num_attention_heads / self.num_kv_heads;
        let k = match num_rep {
            1 => k,
            _ => Tensor::cat(&vec![&k; num_rep], 1)?,
        };
        let v = match num_rep {
            1 => v,
            _ => Tensor::cat(&vec![&v; num_rep], 1)?,
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

            masked_fill(&attn_scores, &mask, -1e9_f32)?
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub output_size: usize,
    #[serde(default)]
    pub dropout: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    dropout: Option<candle_nn::Dropout>,
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
        let dropout = cfg.dropout.map(candle_nn::Dropout::new);
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            dropout,
            span,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;

        // Apply dropout if configured
        if let Some(ref dropout) = self.dropout {
            x = dropout.forward(&x, true)?;
        }

        Ok(self.c_proj.forward(&x)?)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerBlockConfig {
    pub dyn_tanh_config: DynamicTanhConfig,
    pub attn_config: CausalSelfAttentionConfig,
    pub mlp_config: MlpConfig,
}

impl TransformerBlock {
    pub fn init(vb: &VarBuilder, cfg: &TransformerBlockConfig) -> Result<TransformerBlock> {
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

    pub fn forward(&self, x: Tensor, with_causal_mask: bool) -> Result<Tensor> {
        let (output, _) = self.forward_internal(x, with_causal_mask, false)?;
        Ok(output)
    }

    pub fn forward_with_attn(&self, x: Tensor, with_causal_mask: bool) -> Result<(Tensor, Tensor)> {
        let (output, attn_opt) = self.forward_internal(x, with_causal_mask, true)?;
        Ok((output, attn_opt.unwrap()))
    }

    fn forward_internal(
        &self,
        x: Tensor,
        with_causal_mask: bool,
        return_attention: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (_b, _t, d) = x.dims3()?;
        ensure!(
            d == self.attn.head_dim * self.attn.num_attention_heads,
            "Block expected hidden dim {}, got {}",
            self.attn.head_dim * self.attn.num_attention_heads,
            d
        );

        // x = x + attention(norm(x))
        let x_residual = x.clone();
        let x = self.dyn_tanh_1.forward(x)?;
        let (x, attn) = self.attn.forward(x, with_causal_mask)?;
        let x = (x + x_residual)?;

        // x = x + mlp(norm(x))
        let x_residual = x.clone();
        let x = self.dyn_tanh_2.forward(x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + x_residual)?;

        let attn_opt = if return_attention { Some(attn) } else { None };
        Ok((x, attn_opt))
    }
}

#[derive(Debug, Clone)]
pub struct TransformerBody {
    blocks: Vec<TransformerBlock>,
    dyn_tanh: DynamicTanh,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerBodyConfig {
    pub blocks_config: TransformerBlockConfig,
    pub num_blocks: usize,
    pub dyn_tanh_config: DynamicTanhConfig,
}

impl TransformerBody {
    pub fn init(vb: &VarBuilder, cfg: &TransformerBodyConfig) -> Result<TransformerBody> {
        let blocks = (0..cfg.num_blocks)
            .map(|i| TransformerBlock::init(&vb.pp(format!("block_{}", i)), &cfg.blocks_config))
            .collect::<Result<Vec<TransformerBlock>>>()?;
        let dyn_tanh = DynamicTanh::init(&vb.pp("dyn_tanh"), &cfg.dyn_tanh_config)?;
        Ok(TransformerBody { blocks, dyn_tanh })
    }

    pub fn forward(&self, x: Tensor, with_causal_mask: bool) -> Result<Tensor> {
        let (output, _) = self.forward_internal(x, with_causal_mask, false)?;
        Ok(output)
    }

    pub fn forward_with_attn(
        &self,
        x: Tensor,
        with_causal_mask: bool,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let (output, attns) = self.forward_internal(x, with_causal_mask, true)?;
        Ok((output, attns.unwrap_or_default()))
    }

    fn forward_internal(
        &self,
        x: Tensor,
        with_causal_mask: bool,
        return_attention: bool,
    ) -> Result<(Tensor, Option<Vec<Tensor>>)> {
        let mut x = x;
        let mut attns = if return_attention {
            Some(Vec::with_capacity(self.blocks.len()))
        } else {
            None
        };

        for block in self.blocks.iter() {
            if return_attention {
                let (y, attn) = block.forward_with_attn(x, with_causal_mask)?;
                x = y;
                attns.as_mut().unwrap().push(attn);
            } else {
                x = block.forward(x, with_causal_mask)?;
            }
        }

        let x = self.dyn_tanh.forward(x)?;
        Ok((x, attns))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Conv2dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl From<Conv2dConfig> for CandleConv2dConfig {
    fn from(val: Conv2dConfig) -> Self {
        CandleConv2dConfig {
            padding: val.padding,
            stride: val.stride,
            dilation: val.dilation,
            groups: val.groups,
            cudnn_fwd_algo: None,
        }
    }
}

impl From<CandleConv2dConfig> for Conv2dConfig {
    fn from(cfg: CandleConv2dConfig) -> Self {
        Self {
            padding: cfg.padding,
            stride: cfg.stride,
            dilation: cfg.dilation,
            groups: cfg.groups,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvBlockSpec {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub conv_cfg: Conv2dConfig,
    pub gn_groups: usize, // must divide out_channels
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvEncoderConfig {
    pub input_hw: (usize, usize),
    pub blocks: Vec<ConvBlockSpec>,
    pub mlp: MlpConfig,
}

#[derive(Debug, Clone)]
pub struct ConvBlock {
    conv: Conv2d,
    gn: GroupNorm,
}

impl ConvBlock {
    pub fn init(vb: &VarBuilder, spec: &ConvBlockSpec, idx: usize) -> Result<(Self, usize)> {
        assert!(
            spec.out_channels.is_multiple_of(spec.gn_groups),
            "gn_groups must divide out_channels"
        );
        let conv = conv2d(
            spec.in_channels,
            spec.out_channels,
            spec.kernel_size,
            spec.conv_cfg.into(),
            vb.pp(format!("blocks_{idx}_conv")),
        )?;
        let gn = group_norm(
            spec.gn_groups,
            spec.out_channels,
            1e-8,
            vb.pp(format!("blocks_{idx}_gn")),
        )?;
        Ok((Self { conv, gn }, spec.out_channels))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.gn.forward(&x)?;
        Ok(candle_nn::ops::silu(&x)?)
    }

    pub fn get_conv_filters(&self) -> Result<RgbImage> {
        let w = self.conv.weight();
        let w = w.to_dtype(fdtype())?;
        let w = w.to_device(&Device::Cpu)?;
        let (out_channels, in_per_group, kh, kw) = w.dims4()?;

        // Layout a grid for visualization
        let pad: usize = 1;
        let cols: usize = (f32::sqrt(out_channels as f32).ceil() as usize).max(1);
        let rows: usize = out_channels.div_ceil(cols);
        let img_w: usize = cols * kw + (cols.saturating_sub(1)) * pad;
        let img_h: usize = rows * kh + (rows.saturating_sub(1)) * pad;
        let mut img = RgbImage::new(img_w as u32, img_h as u32);

        for o in 0..out_channels {
            let row = o / cols;
            let col = o % cols;
            let x0 = col * (kw + pad);
            let y0 = row * (kh + pad);

            // [in_per_group, kh, kw]
            let k = w.i(o)?.contiguous()?.reshape(&[in_per_group, kh, kw])?;

            // Universal path: average across input channels to a single map
            let mean_2d = k.mean([0])?; // [kh, kw]
            let gray_flat: Vec<f32> = mean_2d.reshape((kh * kw,))?.to_vec1::<f32>()?;
            // Per-filter min/max normalization (single pass using built-in min/max)
            let (mn, mx) = gray_flat
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
                    (mn.min(v), mx.max(v))
                });
            let denom = if mx > mn { mx - mn } else { 1.0 };
            for yy in 0..kh {
                for xx in 0..kw {
                    let v = gray_flat[yy * kw + xx];
                    let g = (((v - mn) / denom) * 255.0).clamp(0.0, 255.0) as u8;
                    img.put_pixel((x0 + xx) as u32, (y0 + yy) as u32, Rgb([g, g, g]));
                }
            }
        }

        Ok(img)
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
        // Validate that blocks are properly chained
        if cfg.blocks.is_empty() {
            return Err(anyhow::anyhow!("ConvEncoder must have at least one block"));
        }

        // Validate that consecutive blocks have matching channels
        for (i, block) in cfg.blocks.windows(2).enumerate() {
            let curr = &block[1];
            let prev = &block[0];
            if curr.in_channels != prev.out_channels {
                return Err(anyhow::anyhow!(
                    "Block {} in_channels ({}) must match block {} out_channels ({})",
                    i + 1,
                    curr.in_channels,
                    i,
                    prev.out_channels
                ));
            }
        }

        let mut blocks = Vec::with_capacity(cfg.blocks.len());
        for (i, spec) in cfg.blocks.iter().enumerate() {
            let (b, _out_ch) = ConvBlock::init(vb, spec, i)?;
            blocks.push(b);
        }

        // infer flatten dim using vb's device/dtype
        let (h, w) = cfg.input_hw;
        let mut y = Tensor::zeros(
            (1, cfg.blocks[0].in_channels, h, w),
            vb.dtype(),
            vb.device(),
        )?;
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

    pub fn get_conv_filters(&self) -> Result<Vec<RgbImage>> {
        self.blocks.iter().map(|b| b.get_conv_filters()).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiLMConfig {
    pub cond_dim: usize,
    pub feat_dim: usize, // D of x: [B, D]
    pub hidden: usize,
    pub output_dim: usize,
}

#[derive(Debug, Clone)]
pub struct FiLM {
    proj1: Linear, // cond_dim -> hidden
    proj2: Linear, // hidden -> 2*feat_dim
    post: Linear,  // feat_dim -> feat_dim
    cond_dim: usize,
    feat_dim: usize,
    _output_dim: usize,
}

impl FiLM {
    pub fn init(vb: &VarBuilder, cfg: &FiLMConfig) -> Result<Self> {
        let proj1 = linear(cfg.cond_dim, cfg.hidden, vb.pp("proj1"))?;
        let proj2 = linear(cfg.hidden, 2 * cfg.feat_dim, vb.pp("proj2"))?;
        let post = linear(cfg.feat_dim, cfg.output_dim, vb.pp("post"))?;
        Ok(Self {
            proj1,
            proj2,
            post,
            cond_dim: cfg.cond_dim,
            feat_dim: cfg.feat_dim,
            _output_dim: cfg.output_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        assert!(
            x.rank() >= 2,
            "FiLM expects x to have at least 2 dimensions"
        );
        assert!(
            cond.rank() >= 2,
            "FiLM expects cond to have at least 2 dimensions"
        );
        let (bx, dx) = (x.dims()[0], x.dims()[x.rank() - 1]);
        let (bc, dc) = (cond.dims()[0], cond.dims()[cond.rank() - 1]);
        assert_eq!(bx, bc, "batch mismatch");
        assert_eq!(dx, self.feat_dim, "x dim != feat_dim");
        assert_eq!(dc, self.cond_dim, "cond dim != cond_dim");

        let h = candle_nn::ops::silu(&self.proj1.forward(cond)?)?;
        let gb = self.proj2.forward(&h)?; // [B, 2*D]
        let gamma = gb.narrow(D::Minus1, 0, self.feat_dim)?; // [B, D]
        let beta = gb.narrow(D::Minus1, self.feat_dim, self.feat_dim)?; // [B, D]

        // Broadcast gamma/beta across all non-batch, non-feature dims of x
        // x has rank >= 2, with last dim = D and first dim = B
        // Build target shape [B, 1, 1, ..., D] to match x's rank
        let x_rank = x.rank();
        let mut target_shape: Vec<usize> = Vec::with_capacity(x_rank);
        target_shape.push(bx);
        for _ in 0..(x_rank - 2) {
            target_shape.push(1);
        }
        target_shape.push(self.feat_dim);
        let gamma_b = gamma.reshape(target_shape.as_slice())?; // [B, 1, ..., D]
        let beta_b = beta.reshape(target_shape.as_slice())?; // [B, 1, ..., D]

        let gamma1 = (&gamma_b + 1.0f64)?; // [B, 1, ..., D]
        let scaled = x.broadcast_mul(&gamma1)?; // [B, ..., D]
        let y = scaled.broadcast_add(&beta_b)?; // [B, ..., D]
        let y = self.post.forward(&candle_nn::ops::silu(&y)?)?;
        Ok(y)
    }
}
