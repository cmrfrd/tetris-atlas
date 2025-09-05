use std::{collections::HashMap, ops::Deref};

use candle_core::{D, DType, IndexOp, Shape, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, Embedding, Linear, Module, VarBuilder, conv2d, embedding, linear,
    ops::softmax_last_dim,
};

use anyhow::{Result, ensure};

use crate::{
    data::{
        TetrisBoardLogitsTensor, TetrisBoardsDistTensor, TetrisBoardsTensor, TetrisContextTensor,
        TetrisPieceOrientationDistTensor, TetrisPieceOrientationLogitsTensor,
        TetrisPiecePlacementDistTensor, TetrisPiecePlacementTensor, TetrisPieceTensor,
    },
    ops::create_orientation_mask,
    tetris::{
        NUM_TETRIS_CELL_STATES, TetrisBoardRaw, TetrisPiece, TetrisPieceOrientation,
        TetrisPiecePlacement,
    },
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
    pub d_model: usize,
    pub theta: f32,
}

#[derive(Debug, Clone)]
struct RopeEncoding {
    cos: Tensor,
    sin: Tensor,
    max_position_embeddings: usize,
}

impl RopeEncoding {
    fn new(config: RopeEncodingConfig, vb: &VarBuilder) -> Result<Self> {
        let theta = Tensor::from_vec(
            calculate_default_inv_freq(config.d_model, config.theta),
            &[config.d_model / 2],
            vb.device(),
        )?;
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

    fn forward(&self, x: Tensor) -> Result<Tensor> {
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

#[derive(Debug, Clone)]
pub struct RopeEncoding2D {
    max_position_embeddings: usize,
    d_model: usize,
    theta: f32,
    width: usize,
    height: usize,
    cos_img: Tensor,
    sin_img: Tensor,
}

#[derive(Debug, Clone)]
pub struct RopeEncoding2DConfig {
    pub max_position_embeddings: usize,
    pub d_model: usize,
    pub theta: f32,
    pub width: usize,
    pub height: usize,
}

impl RopeEncoding2D {
    fn new(config: RopeEncoding2DConfig, vb: &VarBuilder) -> Result<Self> {
        ensure!(
            config.width * config.height <= config.max_position_embeddings,
            "Rope2D: width*height {} exceeds max_position_embeddings {}",
            config.width * config.height,
            config.max_position_embeddings
        );
        ensure!(
            config.d_model % 2 == 0,
            "Rope2D: d_model {} must be even (pairs of dims)",
            config.d_model
        );

        // Precompute 2D cos/sin tables for the fixed (height, width)
        let d_half = config.d_model / 2;
        // Split the half-dimension between X and Y without requiring divisibility by 4
        let x_pairs = d_half / 2;
        let y_pairs = d_half - x_pairs;
        let height = config.height;
        let width = config.width;
        let device = vb.device();
        // Build per-axis inv_freq tensors on the parameter device
        let inv_freq_x = Tensor::from_vec(
            calculate_default_inv_freq(2 * x_pairs, config.theta),
            &[x_pairs],
            device,
        )?;
        let inv_freq_y = Tensor::from_vec(
            calculate_default_inv_freq(2 * y_pairs, config.theta),
            &[y_pairs],
            device,
        )?;

        // X positions -> [W, x_pairs]
        let x_pos = Tensor::arange(0, width as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((width, 1))?;
        let idx_theta_x = x_pos.matmul(&inv_freq_x.reshape((1, x_pairs))?)?; // [W, x_pairs]
        let cos_x = idx_theta_x.cos()?.to_dtype(DType::F32)?; // [W, x_pairs]
        let sin_x = idx_theta_x.sin()?.to_dtype(DType::F32)?; // [W, x_pairs]

        // Y positions -> [H, y_pairs]
        let y_pos = Tensor::arange(0, height as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((height, 1))?;
        let idx_theta_y = y_pos.matmul(&inv_freq_y.reshape((1, y_pairs))?)?; // [H, y_pairs]
        let cos_y = idx_theta_y.cos()?.to_dtype(DType::F32)?; // [H, y_pairs]
        let sin_y = idx_theta_y.sin()?.to_dtype(DType::F32)?; // [H, y_pairs]

        // Broadcast to [H, W, x_pairs] and [H, W, y_pairs]
        let zeros_hw_x = Tensor::zeros(&[height, width, x_pairs], DType::F32, device)?;
        let zeros_hw_y = Tensor::zeros(&[height, width, y_pairs], DType::F32, device)?;
        let cos_x_map = zeros_hw_x.broadcast_add(&cos_x.reshape((1, width, x_pairs))?)?; // [H, W, x_pairs]
        let sin_x_map = zeros_hw_x.broadcast_add(&sin_x.reshape((1, width, x_pairs))?)?; // [H, W, x_pairs]
        let cos_y_map = zeros_hw_y.broadcast_add(&cos_y.reshape((height, 1, y_pairs))?)?; // [H, W, y_pairs]
        let sin_y_map = zeros_hw_y.broadcast_add(&sin_y.reshape((height, 1, y_pairs))?)?; // [H, W, y_pairs]

        // Concatenate along feature to get [H, W, D/2]
        let cos_hw = Tensor::cat(&[&cos_x_map, &cos_y_map], 2)?; // [H, W, D/2]
        let sin_hw = Tensor::cat(&[&sin_x_map, &sin_y_map], 2)?; // [H, W, D/2]

        // Flatten to [H*W, D/2]
        let cos_img = cos_hw.reshape((height * width, d_half))?;
        let sin_img = sin_hw.reshape((height * width, d_half))?;
        Ok(Self {
            max_position_embeddings: config.max_position_embeddings,
            d_model: config.d_model,
            theta: config.theta,
            width,
            height,
            cos_img,
            sin_img,
        })
    }

    // Note: Only 2D image variant is supported for this encoding.

    // Apply 2D RoPE to a sequence with explicit heads: x is [B, H, T=H*W, D]
    // First D/2 dims encode X (width), second D/2 dims encode Y (height)
    pub fn forward(&self, x: Tensor) -> Result<Tensor> {
        let (_b, _num_heads, seq_len, d) = x.dims4()?;
        ensure!(
            seq_len == self.height * self.width,
            "Rope2D.forward: seq_len {} != height*width {}*{}",
            seq_len,
            self.height,
            self.width
        );
        ensure!(
            seq_len <= self.max_position_embeddings,
            "Rope2D.forward: seq_len {} exceeds max_position_embeddings {}",
            seq_len,
            self.max_position_embeddings
        );
        ensure!(d % 2 == 0, "Rope2D.forward: D={} must be even", d);
        ensure!(
            d == self.d_model,
            "Rope2D.forward: D={} must equal configured d_model {}",
            d,
            self.d_model
        );

        let cos = self.cos_img.narrow(0, 0, seq_len)?; // [T, D/2]
        let sin = self.sin_img.narrow(0, 0, seq_len)?; // [T, D/2]

        let y = candle_nn::rotary_emb::rope(&x, &cos, &sin)?; // [B, H, T, D]
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
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rope: RopeEncoding,
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
        let rope = RopeEncoding::new(
            RopeEncodingConfig {
                max_position_embeddings: config.max_position_embeddings,
                d_model: head_dim,
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

    pub fn forward(&self, x: Tensor, with_causal_mask: bool) -> Result<Tensor> {
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

        // q, k: [B, H, T, D], k^T over last two dims
        let attn_scores = (q.matmul(&k.t()?)? / f64::sqrt(self.head_dim as f64))?;

        // apply causal mask
        let attn_scores = if seq_len == 1 {
            attn_scores
        } else if with_causal_mask {
            let mask = self.mask(seq_len)?.broadcast_as(attn_scores.shape())?;
            let shape = mask.shape();
            let on_true =
                Tensor::new(f32::NEG_INFINITY, attn_scores.device())?.broadcast_as(shape.dims())?;
            let m = mask.where_cond(&on_true, &attn_scores)?;
            m
        } else {
            attn_scores
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scores)?;
        let y = attn_weights
            .matmul(&v.contiguous()?)?
            .to_dtype(DType::F32)?
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, hidden_dim])?;

        Ok(self.o_proj.forward(&y)?)
    }

    fn mask(&self, t: usize) -> Result<Tensor> {
        let device = self.q_proj.weight().device();
        let mask: Vec<_> = (0..t)
            .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask, (t, t), &device)?;
        Ok(mask)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    span: tracing::Span,
    hidden_size: usize,
}

#[derive(Debug, Clone)]
pub struct MlpConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub output_size: usize,
}

impl Mlp {
    fn init(vb: &VarBuilder, cfg: &MlpConfig) -> Result<Self> {
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
            hidden_size: h_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        Ok(self.c_proj.forward(&x)?)
    }
}

#[derive(Debug, Clone)]
pub struct DynamicTanh {
    pub alpha: Tensor,
    pub weight: Tensor,
    pub bias: Tensor,
    normalized_shape: usize,
    span: tracing::Span,
}

#[derive(Debug, Clone)]
pub struct DynamicTanhConfig {
    pub alpha_init_value: f64,
    pub normalized_shape: usize,
}

impl DynamicTanh {
    fn init(vb: &VarBuilder, cfg: &DynamicTanhConfig) -> Result<DynamicTanh> {
        let span = tracing::span!(tracing::Level::TRACE, "dyn_tanh");
        // Make these parameters (trainable), with constant inits
        let weight = vb.get_with_hints(
            &[cfg.normalized_shape],
            "weight",
            candle_nn::Init::Const(1.0),
        )?;
        let bias =
            vb.get_with_hints(&[cfg.normalized_shape], "bias", candle_nn::Init::Const(0.0))?;
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

    fn forward(&self, x: Tensor) -> Result<Tensor> {
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
pub struct TetrisBoardEncoderTransformer {
    board_embedding: Embedding,
    blocks: Vec<TetrisGameTransformerBlock>,
    dyn_tanh_1: DynamicTanh,
    mlp_1: Mlp,
    dyn_tanh_2: DynamicTanh,
    mlp_2: Mlp,
}

#[derive(Debug, Clone)]
pub struct TetrisBoardEncoderTransformerConfig {
    pub board_embedding_config: (usize, usize),
    pub blocks_config: TetrisGameTransformerBlockConfig,
    pub num_blocks: usize,
    pub dyn_tanh_1_config: DynamicTanhConfig,
    pub mlp_1_config: MlpConfig,
    pub dyn_tanh_2_config: DynamicTanhConfig,
    pub mlp_2_config: MlpConfig,
}

impl TetrisBoardEncoderTransformer {
    pub fn init(
        vb: &VarBuilder,
        cfg: &TetrisBoardEncoderTransformerConfig,
    ) -> Result<TetrisBoardEncoderTransformer> {
        let board_embedding = embedding(
            cfg.board_embedding_config.0,
            cfg.board_embedding_config.1,
            vb.pp("board_embedding"),
        )?;
        let dyn_tanh_1 =
            DynamicTanh::init(&vb.pp("board_encoder_dyn_tanh_1"), &cfg.dyn_tanh_1_config)?;
        let dyn_tanh_2 =
            DynamicTanh::init(&vb.pp("board_encoder_dyn_tanh_2"), &cfg.dyn_tanh_2_config)?;
        let blocks = (0..cfg.num_blocks)
            .map(|i| {
                TetrisGameTransformerBlock::init(
                    &vb.pp(&format!("transformer_block_{}", i)),
                    &cfg.blocks_config,
                )
            })
            .collect::<Result<Vec<TetrisGameTransformerBlock>>>()?;
        let mlp_1 = Mlp::init(&vb.pp("mlp_1"), &cfg.mlp_1_config)?;
        let mlp_2 = Mlp::init(&vb.pp("mlp_2"), &cfg.mlp_2_config)?;

        Ok(TetrisBoardEncoderTransformer {
            board_embedding,
            blocks,
            dyn_tanh_1,
            mlp_1,
            dyn_tanh_2,
            mlp_2,
        })
    }

    fn soft_forward_board_embeddings(&self, board_dist: &TetrisBoardsDistTensor) -> Result<Tensor> {
        // embeddings: [S, D]
        let embeddings = self.board_embedding.embeddings();
        let (in_size, out_size) = embeddings.dims2()?;
        // board_dist: [B, T, S]
        let (b, t, s) = board_dist.dims3()?;
        ensure!(
            t == TetrisBoardRaw::SIZE,
            "Expected size {} got {}",
            TetrisBoardRaw::SIZE,
            t
        );
        ensure!(
            s == in_size,
            "Board dist last dim {} must equal vocab size {}",
            s,
            in_size
        );
        // Expand to exact batched shapes: [B,T,1,S] @ [B,T,S,D] -> [B,T,1,D] -> [B,T,D]
        let lhs = board_dist.unsqueeze(2)?; // [B,T,1,S]
        let rhs = embeddings
            .reshape(&[1, 1, in_size, out_size])? // [1,1,S,D]
            .repeat(&[b, t, 1, 1])?; // [B,T,S,D]
        let out = lhs.matmul(&rhs)?.squeeze(2)?; // [B,T,D]
        Ok(out)
    }

    pub fn soft_forward(&self, board_dist: &TetrisBoardsDistTensor) -> Result<Tensor> {
        let board_embeddings = self.soft_forward_board_embeddings(board_dist)?;

        let mut out = board_embeddings;
        for block in self.blocks.iter() {
            out = block.forward(out, false)?;
        }
        out = self.dyn_tanh_1.forward(out)?;

        // // [B, T, D] -> [B, D, T]
        // out = out.transpose(1, 2)?;
        // // [B, D, T] -> [B, D, 1] -> [B, 1, D]
        // out = self.mlp_1.forward(&out)?.transpose(1, 2)?;
        out = out.mean_keepdim(1)?;
        out = self.dyn_tanh_2.forward(out)?;
        // [B, 1, D] -> [B, 1, D_2] -> [B, D_2]
        out = self.mlp_2.forward(&out)?.squeeze(1)?;
        Ok(out)
    }
}

/// Tetris Fuse Tokenizer
///
/// This is a tokenizer that Outputs all tokens for the tetris world model.
/// The goal of this tokenizer is to fuse states and actions into a single token.
///
/// Every token is a vector of size `token_dim`, composing of:
/// fuse(board_embedding, piece_embedding, placement_embedding)
///
/// Token Types:
/// - Goal token: fuse(board_embedding, <goal_token>)
/// - Current token: fuse(board_embedding, <current_token>)
/// - Action token: fuse(board_embedding, piece_embedding, placement_embedding)
#[derive(Debug, Clone)]
pub struct TetrisWorldModelTokenizer {
    pub board_encoder: TetrisBoardEncoderTransformer,

    piece_embedding: Embedding,
    placement_embedding: Embedding,

    token_encoder: Mlp,
    dyn_tanh: DynamicTanh,
}

#[derive(Debug, Clone)]
pub struct TetrisWorldModelTokenizerConfig {
    pub board_encoder_config: TetrisBoardEncoderTransformerConfig,
    pub piece_embedding_dim: usize,
    pub placement_embedding_dim: usize,

    pub token_encoder_config: MlpConfig,
}

impl TetrisWorldModelTokenizer {
    const GOAL_TOKEN_IDX: u32 = 0;
    const CURRENT_TOKEN_IDX: u32 = 1;
    const REGULAR_TOKEN_IDX: u32 = 2;

    pub fn init(
        vb: &VarBuilder,
        cfg: &TetrisWorldModelTokenizerConfig,
    ) -> Result<TetrisWorldModelTokenizer> {
        let board_encoder = TetrisBoardEncoderTransformer::init(vb, &cfg.board_encoder_config)?;
        let piece_embedding = embedding(
            TetrisPiece::NUM_PIECES + 2,
            cfg.piece_embedding_dim,
            vb.pp("piece_embedding"),
        )?;
        let placement_embedding = embedding(
            TetrisPiecePlacement::NUM_PLACEMENTS + 2,
            cfg.placement_embedding_dim,
            vb.pp("placement_embedding"),
        )?;
        let token_encoder = Mlp::init(&vb.pp("token_encoder_"), &cfg.token_encoder_config)?;
        let dyn_tanh = DynamicTanh::init(
            &vb.pp("dyn_tanh"),
            &DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: cfg.token_encoder_config.output_size,
            },
        )?;
        Ok(TetrisWorldModelTokenizer {
            board_encoder,
            piece_embedding,
            placement_embedding,
            token_encoder,
            dyn_tanh,
        })
    }

    /// Create a "goal" token for a given board
    ///
    /// A goal token contains an embedding of the goal board,
    /// and "noop" goal embeddings for the piece and placement.
    pub fn forward_goal_token(&self, goal_board: &TetrisBoardsDistTensor) -> Result<Tensor> {
        let device = goal_board.device();
        let (b, _, _) = goal_board.dims3()?;

        let board_embedding = self.board_encoder.soft_forward(goal_board)?;

        let goal_indexer = Tensor::full(Self::GOAL_TOKEN_IDX, (b,), &device)?;

        let piece_embedding = self.piece_embedding.forward(&goal_indexer)?;
        let placement_embedding = self.placement_embedding.forward(&goal_indexer)?;

        // [B, 3*D] -> [B, 1, 3*D]
        let token = Tensor::cat(&[board_embedding, piece_embedding, placement_embedding], 1)?
            .unsqueeze(1)?;

        // [B, 1, 3*D] -> [B, 1, D]
        let token_encoded = self.token_encoder.forward(&token)?;
        let token_encoded = self.dyn_tanh.forward(token_encoded)?;
        // [B, 1, D] -> [B, D]
        let token_encoded = token_encoded.squeeze(1)?;
        Ok(token_encoded)
    }

    /// Create a "current" token for a given board
    ///
    /// A current token contains an embedding of the current board,
    /// an embedding of the current piece,
    /// and a "noop" embedding for placement.
    pub fn forward_current_token(
        &self,
        current_board: &TetrisBoardsDistTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<Tensor> {
        let device = current_board.device();
        let (b, _, _) = current_board.dims3()?;
        let board_embedding = self.board_encoder.soft_forward(current_board)?;

        let piece_offset = Tensor::full(Self::CURRENT_TOKEN_IDX, current_piece.shape(), &device)?;
        let piece_indices = (current_piece.deref() + &piece_offset)?.squeeze(1)?;
        let piece_embedding = self.piece_embedding.forward(&piece_indices)?;

        let placement_offset = Tensor::full(Self::CURRENT_TOKEN_IDX, (b,), &device)?;
        let placement_embedding = self.placement_embedding.forward(&placement_offset)?;

        // [B, 3*D] -> [B, 1, 3*D]
        let token = Tensor::cat(&[board_embedding, piece_embedding, placement_embedding], 1)?
            .unsqueeze(1)?;

        // [B, 3*D] -> [B, D]
        let token_encoded = self.token_encoder.forward(&token)?;
        let token_encoded = self.dyn_tanh.forward(token_encoded)?;
        // [B, 1, D] -> [B, D]
        let token_encoded = token_encoded.squeeze(1)?;
        Ok(token_encoded)
    }

    /// Create a token for a given board, piece, and placement
    ///
    /// A token contains an embedding of the boards, piece, and placement.
    pub fn forward_token(
        &self,
        board: &TetrisBoardsDistTensor,
        piece: &TetrisPieceTensor,
        placement: &TetrisPiecePlacementTensor,
    ) -> Result<Tensor> {
        let device = board.device();
        let (b, _, _) = board.dims3()?;
        let board_embedding = self.board_encoder.soft_forward(board)?;

        // get the piece embedding one offset to avoid the goal embedding
        let piece_offset = Tensor::full(Self::REGULAR_TOKEN_IDX, piece.shape(), &device)?;
        let piece_indices = (piece.deref() + &piece_offset)?.squeeze(1)?;
        let piece_embedding = self.piece_embedding.forward(&piece_indices)?;

        // get the placement embedding
        let placement_indices = placement.deref().to_dtype(DType::U32)?.flatten_all()?;
        let placement_offset = Tensor::full(Self::REGULAR_TOKEN_IDX, (b,), &device)?;
        let placement_indices = (placement_indices + &placement_offset)?;
        let placement_embedding = self.placement_embedding.forward(&placement_indices)?;

        // Merge the embeddings
        // ([B, D], [B, D], [B, D]) -> [B, 3*D] -> [B, 1, 3*D]
        let token = Tensor::cat(&[board_embedding, piece_embedding, placement_embedding], 1)?
            .unsqueeze(1)?;

        // [B, 3*D] -> [B, D]
        let token_encoded = self.token_encoder.forward(&token)?;
        let token_encoded = self.dyn_tanh.forward(token_encoded)?;
        // [B, 1, D] -> [B, D]
        let token_encoded = token_encoded.squeeze(1)?;
        Ok(token_encoded)
    }

    /// Create the embedding context
    pub fn forward_context(
        &self,
        goal_board: &TetrisBoardsDistTensor,
        boards: &Vec<TetrisBoardsDistTensor>,
        pieces: &Vec<TetrisPieceTensor>,
        placements: &Vec<TetrisPiecePlacementTensor>,
    ) -> Result<TetrisContextTensor> {
        let (b, _, _) = goal_board.dims3()?;

        // [B, D] -> [B, 1, D]
        let goal_token = self.forward_goal_token(goal_board)?.unsqueeze(1)?;

        // [B, D] -> [B, 1, D]
        let current_token = self
            .forward_current_token(&boards[0], &pieces[0])?
            .unsqueeze(1)?;

        // aggregate([B, D] -> [B, 1, D], S) -> Vec<[B, 1, D]>
        let context_tokens = boards
            .iter()
            .zip(pieces)
            .zip(placements)
            .map(|((b, p), pl)| Ok(self.forward_token(b, p, pl)?.unsqueeze(1)?))
            .collect::<Result<Vec<Tensor>>>()?;

        // aggregate([B, 1, D], [B, 1, D], Vec<[B, 1, D]>) -> [B, 2+S, D]
        let tokens: Vec<&Tensor> = [&goal_token, &current_token]
            .into_iter()
            .chain(context_tokens.iter())
            .collect();
        let tokens = Tensor::cat(&tokens, 1)?;

        Ok(TetrisContextTensor::from(tokens))
    }
}

#[derive(Debug, Clone)]
struct TetrisWorldModelBlock {
    dyn_tanh_1: DynamicTanh,
    attn: CausalSelfAttention,
    dyn_tanh_2: DynamicTanh,
    mlp: Mlp,
}

#[derive(Debug, Clone)]
pub struct TetrisWorldModelBlockConfig {
    pub dyn_tan_config: DynamicTanhConfig,
    pub attn_config: CausalSelfAttentionConfig,
    pub mlp_config: MlpConfig,
}

impl TetrisWorldModelBlock {
    fn init(vb: &VarBuilder, cfg: &TetrisWorldModelBlockConfig) -> Result<TetrisWorldModelBlock> {
        let dyn_tanh_1 = DynamicTanh::init(vb, &cfg.dyn_tan_config)?;
        let attn = CausalSelfAttention::init(vb, &cfg.attn_config)?;
        let dyn_tanh_2 = DynamicTanh::init(vb, &cfg.dyn_tan_config)?;
        let mlp = Mlp::init(vb, &cfg.mlp_config)?;
        Ok(TetrisWorldModelBlock {
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
        let x_residual = x.clone();
        let x = self.dyn_tanh_1.forward(x)?;
        let x = (self.attn.forward(x, with_causal_mask)? + x_residual)?;
        let x_residual = x.clone();
        let x = self.dyn_tanh_2.forward(x)?;
        let x = (self.mlp.forward(&x)? + x_residual)?;
        Ok(x)
    }
}

#[derive(Debug, Clone)]
pub struct TetrisWorldModel {
    blocks: Vec<TetrisWorldModelBlock>,
    dyn_tanh: DynamicTanh,
    board_head: (Tensor, Mlp),
    orientation_head: Mlp,
}

#[derive(Debug, Clone)]
pub struct TetrisWorldModelConfig {
    pub blocks_config: TetrisWorldModelBlockConfig,
    pub num_blocks: usize,
    pub orientation_head_config: MlpConfig,
    pub dyn_tan_config: DynamicTanhConfig,
}

impl TetrisWorldModel {
    pub fn init(vb: &VarBuilder, cfg: &TetrisWorldModelConfig) -> Result<TetrisWorldModel> {
        let blocks = (0..cfg.num_blocks)
            .map(|i| {
                TetrisWorldModelBlock::init(
                    &vb.pp(&format!("transformer_block_{}", i)),
                    &cfg.blocks_config,
                )
            })
            .collect::<Result<Vec<TetrisWorldModelBlock>>>()?;
        let dyn_tanh = DynamicTanh::init(vb, &cfg.dyn_tan_config)?;

        let d_model = cfg.blocks_config.attn_config.d_model;

        let board_pos_embed = vb.pp("board_pos_embed").get_with_hints(
            &[TetrisBoardRaw::SIZE, d_model],
            "board_pos_embed",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;
        let board_head_config = MlpConfig {
            hidden_size: d_model,
            intermediate_size: 3 * d_model,
            output_size: TetrisBoardRaw::NUM_TETRIS_CELL_STATES,
        };
        let board_head = (
            board_pos_embed,
            Mlp::init(&vb.pp("board_head"), &board_head_config)?,
        );
        let orientation_head = Mlp::init(&vb.pp("orientation_head"), &cfg.orientation_head_config)?;
        Ok(TetrisWorldModel {
            board_head,
            orientation_head,
            blocks,
            dyn_tanh,
        })
    }

    fn inner_forward(&self, input_ctx: &Tensor) -> Result<Tensor> {
        let mut x = input_ctx.clone();
        for block in self.blocks.iter() {
            x = block.forward(x, true)?;
        }
        let x = self.dyn_tanh.forward(x)?;

        Ok(x)
    }

    pub fn forward(
        &self,
        input_ctx: &Tensor,                // [B, T, D]
        current_piece: &TetrisPieceTensor, // [B, 1]
        mask_orientations: bool,
    ) -> Result<(TetrisBoardLogitsTensor, TetrisPieceOrientationLogitsTensor)> {
        let (b, seq_len, d) = input_ctx.dims3()?;

        let x = self.inner_forward(input_ctx)?;

        // Slice the last elem of the sequence
        // [B, S, D] -> [B, 1, D]
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?.unsqueeze(1)?;

        // Produce the output board logits
        // [B,1,D] -> [B,1,1,D] -> [B,1,T,D]
        let x_board_out = &x
            .unsqueeze(2)?
            .broadcast_as(&[b, 1, TetrisBoardRaw::SIZE, d])?;
        let pos_embed = self
            .board_head
            .0
            .reshape(&[1, 1, TetrisBoardRaw::SIZE, d])?
            .repeat(&[b, 1, 1, 1])?;
        let x_board_out = (x_board_out + pos_embed)?;

        // [B, 1, T, D] -> [B, 1, T, 2] (linear)
        let x_board_out = self.board_head.1.forward(&x_board_out)?;
        // [B, 1, T, 2] -> [B, 1, T, 2]
        let x_board_out = x_board_out.reshape(&[
            b,
            1,
            TetrisBoardRaw::SIZE,
            TetrisBoardRaw::NUM_TETRIS_CELL_STATES,
        ])?;
        let board_out = TetrisBoardLogitsTensor::try_from(x_board_out.squeeze(1)?.reshape(&[
            b,
            TetrisBoardRaw::SIZE,
            TetrisBoardRaw::NUM_TETRIS_CELL_STATES,
        ])?)?;

        // Produce the orientation logits
        // [B, 1, D] -> [B, 1, O] -> [B, O]
        let mut orientation_out = self.orientation_head.forward(&x)?.squeeze(1)?;
        if mask_orientations {
            let mask = create_orientation_mask(&current_piece)?; // u8/bool mask, shape [B, P]
            let neg_inf = Tensor::new(f32::NEG_INFINITY, orientation_out.device())?
                .broadcast_as(orientation_out.shape().dims())?;
            // Keep logits where mask==1, set to -inf where mask==0
            orientation_out = mask.where_cond(&orientation_out, &neg_inf)?;
        }
        let orientation_out = TetrisPieceOrientationLogitsTensor::try_from(orientation_out)?;

        Ok((board_out, orientation_out))
    }

    // Training-time variant: returns per-step logits for all tokens
    // pieces_per_step: [B, T] (piece id at each step; needed to build an orientation mask per step)
    // valid_steps: [B, T] boolean (1 for steps with targets, 0 otherwise)
    pub fn forward_all(
        &self,
        input_ctx: &TetrisContextTensor, // [B, T, D]
        pieces_seq: &Tensor,             // [B, T]
        mask_orientations: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (b, seq_len, d) = input_ctx.dims3()?;
        let device = input_ctx.device();

        // Core transformer over the whole sequence
        let x = self.inner_forward(input_ctx)?; // [B, S, D]

        // [B,S,D] -> [B, S, 1, D] -> +[B, S, T, D]
        let x_board_logits_out =
            &x.unsqueeze(2)?
                .broadcast_as(&[b, seq_len, TetrisBoardRaw::SIZE, d])?;
        let pos_embed = self
            .board_head
            .0
            .reshape(&[1, 1, TetrisBoardRaw::SIZE, d])?
            .repeat(&[b, seq_len, 1, 1])?;
        let x_board_logits_out = (x_board_logits_out + pos_embed)?;

        // [B,S,T*2] -> [B,S,T,2] (linear)
        let x_board_logits_out = self.board_head.1.forward(&x_board_logits_out)?;
        let board_logits = x_board_logits_out.reshape(&[
            b,
            seq_len,
            TetrisBoardRaw::SIZE,
            TetrisBoardRaw::NUM_TETRIS_CELL_STATES,
        ])?;

        // Orientation logits for every step
        // [B,T,D] -> [B,T,P]
        let mut orientation_logits = self.orientation_head.forward(&x)?;
        if mask_orientations {
            // Build per-step orientation mask: [B,T,P]
            // Implement a helper like `create_orientation_mask_seq(pieces_per_step)` that
            // applies your existing per-piece mask at each time index and stacks along T.
            // also 'cat' zero mask for the first two tokens
            let (b, t) = pieces_seq.dims2()?;
            let pieces_tensor =
                TetrisPieceTensor::try_from(pieces_seq.flatten_all()?.unsqueeze(1)?)?;
            let mask_btp = create_orientation_mask(&pieces_tensor)?.reshape(&[
                b,
                t,
                TetrisPieceOrientation::NUM_ORIENTATIONS,
            ])?;
            let mask_btp = Tensor::cat(
                &[
                    Tensor::full(
                        0u8,
                        (b, 2, TetrisPieceOrientation::NUM_ORIENTATIONS),
                        &device,
                    )?,
                    mask_btp,
                ],
                1,
            )?;

            // Set invalid logits to -inf
            let neg_inf = Tensor::new(f32::NEG_INFINITY, &device)?
                .broadcast_as(orientation_logits.shape().dims())?;
            orientation_logits = mask_btp.where_cond(&orientation_logits, &neg_inf)?;
        }

        Ok((board_logits, orientation_logits))
    }
}

#[derive(Debug, Clone)]
pub struct TetrisGameTransformerBlock {
    dyn_tanh_1: DynamicTanh,
    attn: CausalSelfAttention,
    dyn_tanh_2: DynamicTanh,
    mlp: Mlp,
}

#[derive(Debug, Clone)]
pub struct TetrisGameTransformerBlockConfig {
    pub dyn_tan_config: DynamicTanhConfig,
    pub attn_config: CausalSelfAttentionConfig,
    pub mlp_config: MlpConfig,
}

impl TetrisGameTransformerBlock {
    fn init(
        vb: &VarBuilder,
        cfg: &TetrisGameTransformerBlockConfig,
    ) -> Result<TetrisGameTransformerBlock> {
        let dyn_tanh_1 = DynamicTanh::init(&vb.pp("dyn_tanh_1"), &cfg.dyn_tan_config)?;
        let attn = CausalSelfAttention::init(&vb.pp("attn"), &cfg.attn_config)?;
        let dyn_tanh_2 = DynamicTanh::init(&vb.pp("dyn_tanh_2"), &cfg.dyn_tan_config)?;
        let mlp = Mlp::init(&vb.pp("mlp"), &cfg.mlp_config)?;
        Ok(TetrisGameTransformerBlock {
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
        let x_residual = x.clone();
        let x = self.dyn_tanh_1.forward(x)?;
        let x = (self.attn.forward(x, with_causal_mask)? + x_residual)?;
        let x_residual = x.clone();
        let x = self.dyn_tanh_2.forward(x)?;
        let x = (self.mlp.forward(&x)? + x_residual)?;
        Ok(x)
    }
}

#[derive(Debug, Clone)]
pub struct TetrisGameTransformer {
    board_embedding: Embedding,
    placement_embedding: Embedding,
    blocks: Vec<TetrisGameTransformerBlock>,
    dyn_tanh: DynamicTanh,
    output_layer: Linear,

    num_placement_embedding_residuals: usize,
}

#[derive(Debug, Clone)]
pub struct TetrisGameTransformerConfig {
    pub board_embedding_config: (usize, usize),
    pub placement_embedding_config: (usize, usize),
    pub num_placement_embedding_residuals: usize,
    pub blocks_config: TetrisGameTransformerBlockConfig,
    pub num_blocks: usize,
    pub dyn_tan_config: DynamicTanhConfig,
    pub output_layer_config: (usize, usize),
}

impl TetrisGameTransformer {
    pub fn init(
        vb: &VarBuilder,
        cfg: &TetrisGameTransformerConfig,
    ) -> Result<TetrisGameTransformer> {
        let board_embedding = embedding(
            cfg.board_embedding_config.0,
            cfg.board_embedding_config.1,
            vb.pp("board_embedding"),
        )?;
        let placement_embedding = embedding(
            cfg.placement_embedding_config.0,
            cfg.placement_embedding_config.1,
            vb.pp("placement_embedding"),
        )?;
        let dyn_tanh = DynamicTanh::init(&vb.pp("dyn_tanh"), &cfg.dyn_tan_config)?;
        let blocks = (0..cfg.num_blocks)
            .map(|i| {
                TetrisGameTransformerBlock::init(
                    &vb.pp(&format!("transformer_block_{}", i)),
                    &cfg.blocks_config,
                )
            })
            .collect::<Result<Vec<TetrisGameTransformerBlock>>>()?;
        let output_layer = linear(
            cfg.output_layer_config.0,
            cfg.output_layer_config.1,
            vb.pp("output_layer"),
        )?;
        Ok(TetrisGameTransformer {
            board_embedding,
            placement_embedding,
            blocks,
            dyn_tanh,
            output_layer,
            num_placement_embedding_residuals: cfg.num_placement_embedding_residuals,
        })
    }

    fn soft_forward_board_embeddings(&self, board_dist: &TetrisBoardsDistTensor) -> Result<Tensor> {
        // embeddings: [S, D]
        let embeddings = self.board_embedding.embeddings();
        let (in_size, out_size) = embeddings.dims2()?;
        // board_dist: [B, T, S]
        let (b, t, s) = board_dist.dims3()?;
        ensure!(
            t == TetrisBoardRaw::SIZE,
            "Expected seq_len {} got {}",
            TetrisBoardRaw::SIZE,
            t
        );
        ensure!(
            s == in_size,
            "Board dist last dim {} must equal vocab size {}",
            s,
            in_size
        );
        // Expand to exact batched shapes: [B,T,1,S] @ [B,T,S,D] -> [B,T,1,D] -> [B,T,D]
        let lhs = board_dist.unsqueeze(2)?; // [B,T,1,S]
        let rhs = embeddings
            .reshape(&[1, 1, in_size, out_size])? // [1,1,S,D]
            .repeat(&[b, t, 1, 1])?; // [B,T,S,D]
        let out = lhs.matmul(&rhs)?.squeeze(2)?; // [B,T,D]
        Ok(out)
    }

    fn soft_forward_placement_embeddings(
        &self,
        placement_dist: &TetrisPiecePlacementDistTensor,
    ) -> Result<Tensor> {
        // embeddings: [P, D]
        let embeddings = self.placement_embedding.embeddings();
        let (in_size, out_size) = embeddings.dims2()?;
        // placement_dist: [B, P]
        let (b, p) = placement_dist.dims2()?;
        ensure!(
            p == in_size,
            "Placement dist last dim {} must equal vocab size {}",
            p,
            in_size
        );
        // [B,1,P] @ [B,P,D] -> [B,1,D] -> [B,D]
        let lhs = placement_dist.unsqueeze(1)?; // [B,1,P]
        let rhs = embeddings
            .reshape(&[1, in_size, out_size])? // [1,P,D]
            .repeat(&[b, 1, 1])?; // [B,P,D]
        let out = lhs.matmul(&rhs)?.squeeze(1)?; // [B,D]
        Ok(out)
    }

    pub fn soft_forward(
        &self,
        tetris_boards: &TetrisBoardsDistTensor,
        placements: &TetrisPiecePlacementDistTensor,
    ) -> Result<TetrisBoardsDistTensor> {
        let (batch_size_boards, seq_len, _hidden_dim) = tetris_boards.dims3()?;
        let (batch_size_placements, placements_dist) = placements.dims2()?;
        assert_eq!(batch_size_boards, batch_size_placements);
        assert_eq!(seq_len, TetrisBoardRaw::SIZE);
        assert_eq!(placements_dist, TetrisPiecePlacement::NUM_PLACEMENTS);

        let soft_board_embeddings = self.soft_forward_board_embeddings(tetris_boards)?;
        let soft_placement_embeddings = self
            .soft_forward_placement_embeddings(placements)?
            .unsqueeze(1)?;
        let mut x = soft_board_embeddings;
        for (i, block) in self.blocks.iter().enumerate() {
            if i < self.num_placement_embedding_residuals {
                x = block
                    .forward(x, false)?
                    .broadcast_add(&soft_placement_embeddings.clone())?;
            } else {
                x = block.forward(x, false)?;
            }
        }

        let x = self.dyn_tanh.forward(x)?;
        let output = self.output_layer.forward(&x)?;
        let output = softmax_last_dim(&output)?;
        Ok(TetrisBoardsDistTensor::from(output))
    }

    pub fn forward(
        &self,
        tetris_boards: &TetrisBoardsTensor,
        placements: &TetrisPiecePlacementTensor,
    ) -> Result<TetrisBoardsDistTensor> {
        let (batch_size_boards, state) = tetris_boards.dims2()?;
        let (batch_size_placements, placement_dim) = placements.dims2()?;
        assert_eq!(batch_size_boards, batch_size_placements);
        assert_eq!(placement_dim, 1);
        assert_eq!(state, TetrisBoardRaw::SIZE);

        // A placement is a single integer token. so when getting a placement embedding,
        // we get [B, 1, E] which we need to reshape to [B, 1, E] for broadcasting with [B, T, E]
        let placement_embeddings = self.placement_embedding.forward(&placements)?;

        let board_embeddings = self.board_embedding.forward(&tetris_boards)?;

        let mut x = board_embeddings;
        for (i, block) in self.blocks.iter().enumerate() {
            if i < self.num_placement_embedding_residuals {
                x = block
                    .forward(x, false)?
                    .broadcast_add(&placement_embeddings)?;
            } else {
                x = block.forward(x, false)?;
            }
        }

        let x = self.dyn_tanh.forward(x)?;

        // Output is a distribution over the cell states
        // [B, T, S]
        let output_logits = self.output_layer.forward(&x)?;
        Ok(TetrisBoardsDistTensor::from(output_logits))
    }

    pub fn soft_forward_sequence(
        &self,
        mut current_board: TetrisBoardsDistTensor,
        placements: Vec<TetrisPiecePlacementDistTensor>,
    ) -> Result<Vec<TetrisBoardsDistTensor>> {
        let mut board_trajectories = Vec::with_capacity(placements.len());

        for placement in placements {
            let next_board = self.soft_forward(&current_board, &placement)?;
            board_trajectories.push(next_board.clone());
            current_board = next_board;
        }

        Ok(board_trajectories)
    }
}

#[derive(Debug, Clone)]
pub struct TetrisPlayerTransformer {
    board_embedding: Embedding,
    piece_embedding: Embedding,

    // Condition board mlp
    condition_board_mlp: Mlp,

    // Tail model
    blocks: Vec<TetrisGameTransformerBlock>,
    dyn_tanh: DynamicTanh,

    output_layer: Linear,
}

#[derive(Debug, Clone)]
pub struct TetrisPlayerTransformerConfig {
    pub board_embedding_config: (usize, usize),
    pub piece_embedding_config: (usize, usize),

    // fusion config: delta + projection
    pub condition_board_mlp_config: MlpConfig,

    // Model Body
    pub blocks_config: TetrisGameTransformerBlockConfig,
    pub num_blocks: usize,
    pub dyn_tan_config: DynamicTanhConfig,

    // Output layer
    pub output_layer_config: (usize, usize),
}

impl TetrisPlayerTransformer {
    pub fn init(
        vb: &VarBuilder,
        cfg: &TetrisPlayerTransformerConfig,
    ) -> Result<TetrisPlayerTransformer> {
        let board_embedding = embedding(
            cfg.board_embedding_config.0,
            cfg.board_embedding_config.1,
            vb.pp("board_embedding"),
        )?;
        let piece_embedding = embedding(
            cfg.piece_embedding_config.0,
            cfg.piece_embedding_config.1,
            vb.pp("piece_embedding"),
        )?;

        // Condition board mlp
        let condition_board_mlp = Mlp::init(
            &vb.pp("condition_board_mlp"),
            &cfg.condition_board_mlp_config,
        )?;

        let dyn_tanh = DynamicTanh::init(&vb.pp("dyn_tanh"), &cfg.dyn_tan_config)?;
        let blocks = (0..cfg.num_blocks)
            .map(|i| {
                TetrisGameTransformerBlock::init(
                    &vb.pp(&format!("block_{}", i)),
                    &cfg.blocks_config,
                )
            })
            .collect::<Result<Vec<TetrisGameTransformerBlock>>>()?;

        let output_layer = linear(
            cfg.output_layer_config.0,
            cfg.output_layer_config.1,
            vb.pp("output_layer"),
        )?;
        Ok(TetrisPlayerTransformer {
            board_embedding,
            piece_embedding,
            condition_board_mlp,
            blocks,
            dyn_tanh,
            output_layer,
        })
    }

    pub fn soft_forward_board_embeddings(
        &self,
        board_dist: &TetrisBoardsDistTensor,
    ) -> Result<Tensor> {
        // embeddings: [S, D]
        let embeddings = self.board_embedding.embeddings();
        let (in_size, out_size) = embeddings.dims2()?;
        // board_dist: [B, T, S]
        let (b, t, s) = board_dist.dims3()?;
        ensure!(
            t == TetrisBoardRaw::SIZE,
            "Expected seq_len {} got {}",
            TetrisBoardRaw::SIZE,
            t
        );
        ensure!(
            s == in_size,
            "Board dist last dim {} must equal vocab size {}",
            s,
            in_size
        );
        // Expand to exact batched shapes: [B,T,1,S] @ [B,T,S,D] -> [B,T,1,D] -> [B,T,D]
        let lhs = board_dist.unsqueeze(2)?; // [B,T,1,S]
        let rhs = embeddings
            .reshape(&[1, 1, in_size, out_size])? // [1,1,S,D]
            .repeat(&[b, t, 1, 1])?; // [B,T,S,D]
        let out = lhs.matmul(&rhs)?.squeeze(2)?; // [B,T,D]
        Ok(out)
    }

    pub fn soft_forward(
        &self,
        current_pieces: &[TetrisPiece],
        current_boards: &TetrisBoardsDistTensor,
        condition_boards: &TetrisBoardsDistTensor,
        iterations: usize,
    ) -> Result<Tensor> {
        assert!(iterations >= 1, "Iterations must be at least 1");

        let device = current_boards.device();

        // Encode current board, condition board, and piece embeddings
        // [B, T, D]
        let current_board_encoded = self.soft_forward_board_embeddings(current_boards)?;

        // Convert current pieces to tensor [B]
        let current_pieces_tensor = TetrisPieceTensor::from_pieces(current_pieces, device)?;
        let piece_embeddings = self.piece_embedding.forward(&current_pieces_tensor)?; // [B, 1, D]

        // Apply condition board mlp
        // [B, T, D] -> [B, D, T]
        let condition_board_encoded = self
            .soft_forward_board_embeddings(condition_boards)?
            .transpose(1, 2)?;
        // [B, D, T] -> [B, D, 1] -> [B, 1, D]
        let condition_board_encoded = self
            .condition_board_mlp
            .forward(&condition_board_encoded)?
            .transpose(1, 2)?;

        // Input to main transformer: [B, 2*T + 1, D]
        let initial_input = Tensor::cat(
            &[
                current_board_encoded,
                condition_board_encoded,
                piece_embeddings,
            ],
            1,
        )?;
        let mut x = initial_input.clone();
        for block in self.blocks.iter() {
            x = block.forward(x, true)?.broadcast_add(&initial_input)?;
        }
        x = self.dyn_tanh.forward(x)?;
        // for iter in 0..iterations {
        //     let last_iter = iter + 1 == iterations;
        //     for block in self.blocks.iter() {
        //         x = block.forward(x)?.broadcast_add(&initial_input)?;
        //     }
        //     x = self.dyn_tanh.forward(x)?;
        //     if !last_iter {
        //         x = x.detach();
        //     }
        // }

        // [B, 2*T + 1, D] -> [B, 1, D] -> [B, D]
        let x = x
            .i((.., TetrisBoardRaw::SIZE, ..))?
            .contiguous()?
            .squeeze(1)?;
        // [B, D] -> [B, P]
        let logits = self.output_layer.forward(&x)?;
        Ok(logits)
    }

    pub fn soft_forward_masked(
        &self,
        current_pieces: &[TetrisPiece],
        current_boards: &TetrisBoardsDistTensor,
        condition_boards: &TetrisBoardsDistTensor,
        iterations: usize,
    ) -> Result<TetrisPieceOrientationDistTensor> {
        let device = current_boards.device();
        let current_pieces_tensor = TetrisPieceTensor::from_pieces(current_pieces, device)?;
        let logits =
            self.soft_forward(current_pieces, current_boards, condition_boards, iterations)?;

        let mask = create_orientation_mask(&current_pieces_tensor)?.to_dtype(DType::F32)?;
        // let output = masked_softmax_2d(&logits, &mask)?;
        let output = (logits * mask)?;
        Ok(TetrisPieceOrientationDistTensor::from(output))
    }

    pub fn forward_get_placement(
        &self,
        current_pieces: &[TetrisPiece],
        current_boards: &TetrisBoardsDistTensor,
        condition_boards: &TetrisBoardsDistTensor,
        iterations: usize,
    ) -> Result<(
        TetrisPieceOrientationDistTensor,
        Vec<crate::tetris::TetrisPieceOrientation>,
    )> {
        let output =
            self.soft_forward_masked(current_pieces, current_boards, condition_boards, iterations)?;
        let placements_indices = output.argmax(D::Minus1)?.flatten_all()?.to_vec1::<u32>()?;

        let orientations: Vec<TetrisPieceOrientation> = placements_indices
            .into_iter()
            .map(|orientation_idx| TetrisPieceOrientation::from_index(orientation_idx as u8))
            .collect();

        Ok((output, orientations))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        data::TetrisDatasetGenerator,
        grad_accum::GradientAccumulator,
        ops::masked_softmax_2d,
        tetris::{NUM_TETRIS_CELL_STATES, TetrisPiece},
    };

    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap, var_builder::VarBuilder};

    fn create_var_builder() -> VarBuilder<'static> {
        let device = Device::Cpu;
        let vs = candle_nn::VarMap::new();
        VarBuilder::from_varmap(&vs, DType::F32, &device)
    }

    #[test]
    fn test_rope_encoding() -> Result<()> {
        let vb = create_var_builder();
        let config = RopeEncodingConfig {
            max_position_embeddings: 16,
            d_model: 8,
            theta: 10000.0,
        };

        let rope = RopeEncoding::new(config, &vb)?;

        // Test forward pass with 4D tensor (batch, num_heads, seq_len, head_dim)
        let batch_size = 2;
        let num_heads = 2;
        let seq_len = 4;
        let head_dim = 8;
        let input = Tensor::randn(
            0f32,
            1f32,
            &[batch_size, num_heads, seq_len, head_dim],
            &Device::Cpu,
        )?;

        let output = rope.forward(input)?;
        assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);

        Ok(())
    }

    #[test]
    fn test_causal_self_attention() -> Result<()> {
        let vb = create_var_builder();
        let config = CausalSelfAttentionConfig {
            d_model: 16,
            n_attention_heads: 4,
            n_kv_heads: 2,
            rope_theta: 10000.0,
            max_position_embeddings: TetrisBoardRaw::SIZE,
        };

        let attention = CausalSelfAttention::init(&vb, &config)?;

        // Test forward pass
        let batch_size = 2;
        let seq_len = TetrisBoardRaw::SIZE;
        let hidden_dim = 16;
        let input = Tensor::randn(0f32, 1f32, &[batch_size, seq_len, hidden_dim], &Device::Cpu)?;

        let output = attention.forward(input, true)?;
        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_dim]);

        Ok(())
    }

    #[test]
    fn test_mlp() -> Result<()> {
        let vb = create_var_builder();
        let config = MlpConfig {
            hidden_size: 16,
            intermediate_size: 32,
            output_size: 16,
        };

        let mlp = Mlp::init(&vb, &config)?;

        // Test forward pass
        let batch_size = 2;
        let seq_len = TetrisBoardRaw::SIZE;
        let hidden_size = 16;
        let input = Tensor::randn(
            0f32,
            1f32,
            &[batch_size, seq_len, hidden_size],
            &Device::Cpu,
        )?;

        let output = mlp.forward(&input)?;
        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);

        Ok(())
    }

    #[test]
    fn test_dynamic_tanh() -> Result<()> {
        let vb = create_var_builder();
        let config = DynamicTanhConfig {
            alpha_init_value: 1.0,
            normalized_shape: 16,
        };

        let dyn_tanh = DynamicTanh::init(&vb, &config)?;

        // Test forward pass
        let batch_size = 2;
        let seq_len = TetrisBoardRaw::SIZE;
        let hidden_size = 16;
        let input = Tensor::randn(
            0f32,
            1f32,
            &[batch_size, seq_len, hidden_size],
            &Device::Cpu,
        )?;

        let output = dyn_tanh.forward(input)?;
        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);

        Ok(())
    }

    #[test]
    fn test_tetris_game_transformer_block() -> Result<()> {
        let vb = create_var_builder();
        let config = TetrisGameTransformerBlockConfig {
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: 16,
            },
            attn_config: CausalSelfAttentionConfig {
                d_model: 16,
                n_attention_heads: 2,
                n_kv_heads: 2,
                rope_theta: 10000.0,
                max_position_embeddings: TetrisBoardRaw::SIZE,
            },
            mlp_config: MlpConfig {
                hidden_size: 16,
                intermediate_size: 32,
                output_size: 16,
            },
        };

        let block = TetrisGameTransformerBlock::init(&vb, &config)?;

        // Test forward pass
        let batch_size = 2;
        let seq_len = TetrisBoardRaw::SIZE;
        let hidden_size = 16;
        let input = Tensor::randn(
            0f32,
            1f32,
            &[batch_size, seq_len, hidden_size],
            &Device::Cpu,
        )?;

        let output = block.forward(input, true)?;
        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);

        Ok(())
    }

    #[test]
    fn test_tetris_game_transformer() -> Result<()> {
        let vb = create_var_builder();

        let num_states = NUM_TETRIS_CELL_STATES;
        let d_model = 16;
        let num_placements = TetrisPiecePlacement::NUM_PLACEMENTS;

        let config = TetrisGameTransformerConfig {
            board_embedding_config: (num_states, d_model),
            placement_embedding_config: (num_placements, d_model),
            num_placement_embedding_residuals: 1,
            blocks_config: TetrisGameTransformerBlockConfig {
                dyn_tan_config: DynamicTanhConfig {
                    alpha_init_value: 1.0,
                    normalized_shape: d_model,
                },
                attn_config: CausalSelfAttentionConfig {
                    d_model: 16,
                    n_attention_heads: 2,
                    n_kv_heads: 2,
                    rope_theta: 10000.0,
                    max_position_embeddings: TetrisBoardRaw::SIZE,
                },
                mlp_config: MlpConfig {
                    hidden_size: d_model,
                    intermediate_size: 2 * d_model,
                    output_size: d_model,
                },
            },
            num_blocks: 2,
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: d_model,
            },
            output_layer_config: (d_model, num_states), // 16 hidden, 10 output classes
        };

        let transformer = TetrisGameTransformer::init(&vb, &config)?;

        // Test regular forward pass
        let batch_size = 4;
        let board_size = TetrisBoardRaw::SIZE; // Should be defined in tetris module
        let tetris_boards = Tensor::zeros(&[batch_size, board_size], DType::U8, &Device::Cpu)?;
        let placements = Tensor::zeros(&[batch_size, 1], DType::U8, &Device::Cpu)?;

        let tetris_boards = TetrisBoardsTensor::from(tetris_boards);
        let placements = TetrisPiecePlacementTensor::from(placements);

        let output = transformer.forward(&tetris_boards, &placements)?;
        assert_eq!(output.dims(), &[batch_size, board_size, num_states]);

        // Test soft forward pass
        let tetris_boards_soft = Tensor::rand(
            0f32,
            1f32,
            &[batch_size, board_size, num_states],
            &Device::Cpu,
        )?;
        let placements_soft = Tensor::rand(
            0f32,
            1f32,
            &[batch_size, TetrisPiecePlacement::NUM_PLACEMENTS],
            &Device::Cpu,
        )?;

        let tetris_boards_soft = TetrisBoardsDistTensor::from(tetris_boards_soft);
        let placements_soft = TetrisPiecePlacementDistTensor::from(placements_soft);

        let output_soft = transformer.soft_forward(&tetris_boards_soft, &placements_soft)?;
        assert_eq!(output_soft.dims(), &[batch_size, board_size, num_states]);

        // Verify all outputs are between 0 and 1
        let output_soft_tensor: Tensor = output_soft.into();
        let output_soft_data: Vec<f32> = output_soft_tensor.flatten_all()?.to_vec1()?;
        let min = output_soft_data
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let max = output_soft_data
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(min >= 0.0, "Min value {} is less than 0", min);
        assert!(max <= 1.0, "Max value {} is greater than 1", max);

        Ok(())
    }

    #[test]
    fn test_tetris_game_transformer_sequence() -> Result<()> {
        let vb = create_var_builder();

        let num_states = NUM_TETRIS_CELL_STATES;
        let d_model = 16;
        let num_placements = TetrisPiecePlacement::NUM_PLACEMENTS;

        let config = TetrisGameTransformerConfig {
            board_embedding_config: (num_states, d_model),
            placement_embedding_config: (num_placements, d_model),
            num_placement_embedding_residuals: 1,
            blocks_config: TetrisGameTransformerBlockConfig {
                dyn_tan_config: DynamicTanhConfig {
                    alpha_init_value: 1.0,
                    normalized_shape: d_model,
                },
                attn_config: CausalSelfAttentionConfig {
                    d_model,
                    n_attention_heads: 2,
                    n_kv_heads: 2,
                    rope_theta: 10000.0,
                    max_position_embeddings: TetrisBoardRaw::SIZE,
                },
                mlp_config: MlpConfig {
                    hidden_size: d_model,
                    intermediate_size: 2 * d_model,
                    output_size: d_model,
                },
            },
            num_blocks: 2,
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: d_model,
            },
            output_layer_config: (d_model, num_states),
        };

        let transformer = TetrisGameTransformer::init(&vb, &config)?;

        // Test sequence forward pass
        let batch_size = 2;
        let board_size = TetrisBoardRaw::SIZE;
        let num_states = NUM_TETRIS_CELL_STATES;
        let starting_boards = TetrisBoardsDistTensor::from(Tensor::rand(
            0f32,
            1f32,
            &[batch_size, board_size, num_states],
            &Device::Cpu,
        )?);

        let placements = vec![
            TetrisPiecePlacementDistTensor::from(Tensor::rand(
                0f32,
                1f32,
                &[batch_size, TetrisPiecePlacement::NUM_PLACEMENTS],
                &Device::Cpu,
            )?),
            TetrisPiecePlacementDistTensor::from(Tensor::rand(
                0f32,
                1f32,
                &[batch_size, TetrisPiecePlacement::NUM_PLACEMENTS],
                &Device::Cpu,
            )?),
            TetrisPiecePlacementDistTensor::from(Tensor::rand(
                0f32,
                1f32,
                &[batch_size, TetrisPiecePlacement::NUM_PLACEMENTS],
                &Device::Cpu,
            )?),
        ];

        let trajectory = transformer.soft_forward_sequence(starting_boards, placements)?;
        assert_eq!(trajectory.len(), 3);
        for board in trajectory {
            assert_eq!(board.dims(), &[batch_size, board_size, num_states]);
        }

        Ok(())
    }

    #[test]
    fn test_soft_forward_board_embeddings() -> Result<()> {
        let vb = create_var_builder();
        let d_model = 16;
        let board_vocab = NUM_TETRIS_CELL_STATES;
        let placement_vocab = TetrisPiecePlacement::NUM_PLACEMENTS;

        let cfg = TetrisGameTransformerConfig {
            board_embedding_config: (board_vocab, d_model),
            placement_embedding_config: (placement_vocab, d_model),
            num_placement_embedding_residuals: 0,
            blocks_config: TetrisGameTransformerBlockConfig {
                dyn_tan_config: DynamicTanhConfig {
                    alpha_init_value: 1.0,
                    normalized_shape: d_model,
                },
                attn_config: CausalSelfAttentionConfig {
                    d_model,
                    n_attention_heads: 2,
                    n_kv_heads: 2,
                    rope_theta: 10000.0,
                    max_position_embeddings: TetrisBoardRaw::SIZE,
                },
                mlp_config: MlpConfig {
                    hidden_size: d_model,
                    intermediate_size: 2 * d_model,
                    output_size: d_model,
                },
            },
            num_blocks: 0,
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: d_model,
            },
            output_layer_config: (d_model, board_vocab),
        };
        let model = TetrisGameTransformer::init(&vb, &cfg)?;

        let batch_size = 2;
        let t = TetrisBoardRaw::SIZE;
        let s = board_vocab;
        let board_dist = Tensor::randn(0.0f32, 1.0f32, &[batch_size, t, s], &Device::Cpu)?;

        let board_dist = TetrisBoardsDistTensor::from(board_dist);

        let out = model.soft_forward_board_embeddings(&board_dist)?;
        assert_eq!(out.dims(), &[batch_size, t, d_model]);
        Ok(())
    }

    #[test]
    fn test_soft_forward_placement_embeddings() -> Result<()> {
        let vb = create_var_builder();
        let d_model = 16;
        let board_vocab = NUM_TETRIS_CELL_STATES;
        let placement_vocab = TetrisPiecePlacement::NUM_PLACEMENTS;

        let cfg = TetrisGameTransformerConfig {
            board_embedding_config: (board_vocab, d_model),
            placement_embedding_config: (placement_vocab, d_model),
            num_placement_embedding_residuals: 0,
            blocks_config: TetrisGameTransformerBlockConfig {
                dyn_tan_config: DynamicTanhConfig {
                    alpha_init_value: 1.0,
                    normalized_shape: d_model,
                },
                attn_config: CausalSelfAttentionConfig {
                    d_model,
                    n_attention_heads: 2,
                    n_kv_heads: 2,
                    rope_theta: 10000.0,
                    max_position_embeddings: TetrisBoardRaw::SIZE,
                },
                mlp_config: MlpConfig {
                    hidden_size: d_model,
                    intermediate_size: 2 * d_model,
                    output_size: d_model,
                },
            },
            num_blocks: 0,
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: d_model,
            },
            output_layer_config: (d_model, board_vocab),
        };
        let model = TetrisGameTransformer::init(&vb, &cfg)?;

        let batch_size = 2;
        let p = placement_vocab;
        let placements_dist = Tensor::randn(0.0f32, 1.0f32, &[batch_size, p], &Device::Cpu)?;

        let placements_dist = TetrisPiecePlacementDistTensor::from(placements_dist);

        let out = model.soft_forward_placement_embeddings(&placements_dist)?;
        assert_eq!(out.dims(), &[batch_size, d_model]);
        Ok(())
    }

    #[test]
    fn test_tetris_player_transformer() -> Result<()> {
        use crate::tetris::{TetrisGameSet, TetrisPiece};
        use candle_core::D;

        let vb = create_var_builder();

        let num_states = NUM_TETRIS_CELL_STATES;
        let d_model = 16;
        let num_orientations = TetrisPieceOrientation::NUM_ORIENTATIONS;

        let config = TetrisPlayerTransformerConfig {
            board_embedding_config: (num_states, d_model),
            piece_embedding_config: (TetrisPiece::NUM_PIECES, d_model),
            num_blocks: 2,
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: d_model,
            },
            condition_board_mlp_config: MlpConfig {
                hidden_size: TetrisBoardRaw::SIZE,
                intermediate_size: d_model,
                output_size: 1,
            },
            blocks_config: TetrisGameTransformerBlockConfig {
                dyn_tan_config: DynamicTanhConfig {
                    alpha_init_value: 1.0,
                    normalized_shape: d_model,
                },
                attn_config: CausalSelfAttentionConfig {
                    d_model: 16,
                    n_attention_heads: 2,
                    n_kv_heads: 2,
                    rope_theta: 10000.0,
                    max_position_embeddings: TetrisBoardRaw::SIZE + 2,
                },
                mlp_config: MlpConfig {
                    hidden_size: d_model,
                    intermediate_size: 2 * d_model,
                    output_size: d_model,
                },
            },
            output_layer_config: (d_model, num_orientations),
        };

        let player_transformer = TetrisPlayerTransformer::init(&vb, &config)?;

        // Create a simple game set for testing
        let batch_size = 4;
        let game_set = TetrisGameSet::new(batch_size);
        let iterations = 2;

        // Test soft forward pass
        let board_size = TetrisBoardRaw::SIZE;
        let tetris_boards_soft = Tensor::rand(
            0f32,
            1f32,
            &[batch_size, board_size, num_states],
            &Device::Cpu,
        )?;
        let input_boards = TetrisBoardsDistTensor::from(tetris_boards_soft.clone());
        let condition_boards = TetrisBoardsDistTensor::from(tetris_boards_soft.clone());
        let current_pieces = game_set.current_pieces().to_vec();
        let current_pieces = current_pieces.as_slice();

        let output = player_transformer.soft_forward_masked(
            &current_pieces,
            &input_boards,
            &condition_boards,
            iterations,
        )?;
        assert_eq!(output.dims(), &[batch_size, num_orientations]);

        // Test forward_get_placement
        let (output_logits, placements) = player_transformer.forward_get_placement(
            &current_pieces,
            &input_boards,
            &condition_boards,
            iterations,
        )?;
        assert_eq!(output_logits.dims(), &[batch_size, num_orientations]);
        assert_eq!(placements.len(), batch_size);
        // Verify all outputs are between 0 and 1 (probabilities)

        // Backward pass: compute CE loss and assert gradients are non-zero
        let varmap = candle_nn::VarMap::new();
        let vb2 = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let player_transformer2 = TetrisPlayerTransformer::init(&vb2, &config)?;

        let input_boards2 = TetrisBoardsDistTensor::from(tetris_boards_soft.clone());
        let condition_boards2 = TetrisBoardsDistTensor::from(tetris_boards_soft.clone());

        // Use raw logits with cross-entropy
        let logits = player_transformer2.soft_forward(
            &current_pieces,
            &input_boards2,
            &condition_boards2,
            iterations,
        )?;
        let targets = logits.argmax(D::Minus1)?.to_dtype(DType::U32)?;

        let loss = candle_nn::loss::cross_entropy(&logits, &targets)?;
        let grads = loss.backward()?;

        let mut total_grad_mag = 0.0f32;
        for v in varmap.all_vars() {
            if let Some(g) = grads.get(&v) {
                total_grad_mag += g.abs()?.sum_all()?.to_scalar::<f32>()?;
            }
        }
        assert!(
            total_grad_mag > 0.0,
            "expected non-zero gradient sum, got {}",
            total_grad_mag
        );

        // Also test gradients for soft_forward_softmax by training with cross-entropy on logits
        let varmap3 = candle_nn::VarMap::new();
        let vb3 = VarBuilder::from_varmap(&varmap3, DType::F32, &Device::Cpu);
        let player_transformer3 = TetrisPlayerTransformer::init(&vb3, &config)?;

        let input_boards3 = TetrisBoardsDistTensor::from(tetris_boards_soft.clone());
        let condition_boards3 = TetrisBoardsDistTensor::from(tetris_boards_soft.clone());

        let logits_softmax = player_transformer3.soft_forward(
            &current_pieces,
            &input_boards3,
            &condition_boards3,
            iterations,
        )?;
        let targets_softmax = logits_softmax.argmax(D::Minus1)?.to_dtype(DType::U32)?;

        let loss_softmax = candle_nn::loss::cross_entropy(&logits_softmax, &targets_softmax)?;
        let grads_softmax = loss_softmax.backward()?;

        let mut total_grad_mag_softmax = 0.0f32;
        for v in varmap3.all_vars() {
            if let Some(g) = grads_softmax.get(&v) {
                total_grad_mag_softmax += g.abs()?.sum_all()?.to_scalar::<f32>()?;
            }
        }
        assert!(
            total_grad_mag_softmax > 0.0,
            "expected non-zero gradient sum for soft_forward_softmax, got {}",
            total_grad_mag_softmax
        );

        Ok(())
    }

    #[test]
    fn test_masked_softmax_2d_partial_mask() -> Result<()> {
        let device = Device::Cpu;
        // x = [1, 2, 3], mask = [1, 0, 1] -> softmax over [1, 3], middle is zero
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3], &device)?;
        let mask = Tensor::from_vec(vec![1.0f32, 0.0, 1.0], &[1, 3], &device)?;
        let out = masked_softmax_2d(&x, &mask)?;
        assert_eq!(out.dims(), &[1, 3]);
        let v: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert_eq!(v.len(), 3);

        // Expected: [exp(-2)/(1+exp(-2)), 0, 1/(1+exp(-2))]
        let exp_m2 = (-2.0f32).exp();
        let z = 1.0 + exp_m2;
        let expected0 = exp_m2 / z;
        let expected2 = 1.0 / z;

        assert!((v[0] - expected0).abs() < 1e-6);
        assert_eq!(v[1], 0.0);
        assert!((v[2] - expected2).abs() < 1e-6);

        // Verify sum is exactly 1.0
        let sum = v.iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 1e-6, "Sum {} should be 1.0", sum);
        Ok(())
    }

    #[test]
    fn test_masked_softmax_2d_fully_masked_row() -> Result<()> {
        let device = Device::Cpu;
        // Row 0 fully masked -> zeros; Row 1 all valid -> sums to 1
        let x = Tensor::from_vec(vec![0.5f32, -1.0, 0.0, 2.0, -2.0, 0.5], &[2, 3], &device)?;
        let mask = Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], &[2, 3], &device)?;
        let out = masked_softmax_2d(&x, &mask)?;
        assert_eq!(out.dims(), &[2, 3]);
        let v: Vec<f32> = out.flatten_all()?.to_vec1()?;

        // First row: exactly zeros
        for &p in &v[0..3] {
            assert_eq!(p, 0.0);
        }
        // Second row: probabilities sum to 1 and are within [0,1]
        let row1_sum = v[3] + v[4] + v[5];
        assert!((row1_sum - 1.0).abs() < 1e-6);
        for &p in &v[3..6] {
            assert!(p >= 0.0 && p <= 1.0);
        }
        Ok(())
    }

    #[test]
    fn test_masked_softmax_2d_errors() -> Result<()> {
        let device = Device::Cpu;
        // Mismatched shapes
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3], &device)?;
        let bad_mask = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 3], &device)?;
        assert!(masked_softmax_2d(&x, &bad_mask).is_err());

        // Non-2D inputs
        let x_3d = Tensor::zeros(&[1, 2, 3], DType::F32, &device)?;
        let m_3d = Tensor::zeros(&[1, 2, 3], DType::F32, &device)?;
        assert!(masked_softmax_2d(&x_3d, &m_3d).is_err());

        // Wrong dtype for x
        let x_u8 = Tensor::zeros(&[2, 3], DType::U8, &device)?;
        let mask = Tensor::zeros(&[2, 3], DType::F32, &device)?;
        assert!(masked_softmax_2d(&x_u8, &mask).is_err());
        Ok(())
    }

    #[test]
    fn test_tetris_world_model_tokenizer() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let embedding_dim = 16;
        let cfg = TetrisWorldModelTokenizerConfig {
            board_encoder_config: TetrisBoardEncoderTransformerConfig {
                board_embedding_config: (NUM_TETRIS_CELL_STATES, embedding_dim),
                blocks_config: TetrisGameTransformerBlockConfig {
                    dyn_tan_config: DynamicTanhConfig {
                        alpha_init_value: 1.0,
                        normalized_shape: embedding_dim,
                    },
                    attn_config: CausalSelfAttentionConfig {
                        d_model: embedding_dim,
                        n_attention_heads: 1,
                        n_kv_heads: 1,
                        rope_theta: 10000.0,
                        max_position_embeddings: 1024,
                    },
                    mlp_config: MlpConfig {
                        hidden_size: embedding_dim,
                        intermediate_size: embedding_dim,
                        output_size: embedding_dim,
                    },
                },
                num_blocks: 3,
                dyn_tanh_1_config: DynamicTanhConfig {
                    alpha_init_value: 1.0,
                    normalized_shape: embedding_dim,
                },
                mlp_1_config: MlpConfig {
                    hidden_size: TetrisBoardRaw::SIZE,
                    intermediate_size: embedding_dim,
                    output_size: 1,
                },
                dyn_tanh_2_config: DynamicTanhConfig {
                    alpha_init_value: 1.0,
                    normalized_shape: embedding_dim,
                },
                mlp_2_config: MlpConfig {
                    hidden_size: embedding_dim,
                    intermediate_size: embedding_dim,
                    output_size: embedding_dim,
                },
            },
            piece_embedding_dim: embedding_dim,
            placement_embedding_dim: embedding_dim,
            token_encoder_config: MlpConfig {
                hidden_size: 3 * embedding_dim,
                intermediate_size: 3 * embedding_dim,
                output_size: embedding_dim,
            },
        };

        let tokenizer = TetrisWorldModelTokenizer::init(&vb, &cfg)?;

        let dataset_generator = TetrisDatasetGenerator::new();

        let batch_size = 4;
        let transition = dataset_generator.gen_uniform_sampled_transition(
            (0..8).into(),
            batch_size,
            &device,
            &mut rand::rng(),
        )?;

        let goal_board = TetrisBoardsDistTensor::try_from(transition.result_board.clone())?;
        let goal_token = tokenizer.forward_goal_token(&goal_board)?;
        assert_eq!(goal_token.shape().dims(), &[batch_size, embedding_dim]);

        let current_board = TetrisBoardsDistTensor::try_from(transition.current_board.clone())?;
        let current_pieces_tensor = TetrisPieceTensor::from_pieces(&transition.piece, &device)?;
        let current_token =
            tokenizer.forward_current_token(&current_board, &current_pieces_tensor)?;
        assert_eq!(current_token.shape().dims(), &[batch_size, embedding_dim]);

        Ok(())
    }

    #[test]
    fn test_tetris_board_encoder() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // 3 conv layers then an MLP
        let enc_cfg = TetrisBoardEncoderTransformerConfig {
            board_embedding_config: (NUM_TETRIS_CELL_STATES, 16),
            blocks_config: TetrisGameTransformerBlockConfig {
                dyn_tan_config: DynamicTanhConfig {
                    alpha_init_value: 1.0,
                    normalized_shape: 16,
                },
                attn_config: CausalSelfAttentionConfig {
                    d_model: 16,
                    n_attention_heads: 1,
                    n_kv_heads: 1,
                    rope_theta: 10000.0,
                    max_position_embeddings: 1024,
                },
                mlp_config: MlpConfig {
                    hidden_size: 16,
                    intermediate_size: 16,
                    output_size: 16,
                },
            },
            num_blocks: 3,
            dyn_tanh_1_config: DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: 16,
            },
            mlp_1_config: MlpConfig {
                hidden_size: TetrisBoardRaw::SIZE,
                intermediate_size: 16,
                output_size: 1,
            },
            dyn_tanh_2_config: DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: 16,
            },
            mlp_2_config: MlpConfig {
                hidden_size: 16,
                intermediate_size: 16,
                output_size: 16,
            },
        };
        let encoder = TetrisBoardEncoderTransformer::init(&vb, &enc_cfg)?;

        // Input: [B, T, D] where T == BOARD_SIZE and D == first in_channels
        let batch_size = 1;
        let input = Tensor::randn(
            0f32,
            1f32,
            &[batch_size, TetrisBoardRaw::SIZE, NUM_TETRIS_CELL_STATES],
            vb.device(),
        )?;
        let input = TetrisBoardsDistTensor::from(input);

        let out = encoder.soft_forward(&input)?;

        // Expected output dims: [B, 16]
        assert_eq!(out.dims(), &[batch_size, 16]);

        // ensure gradients are non-zero
        let mut grad_accum = GradientAccumulator::new(1);
        let grads = out.backward()?;
        grad_accum.accumulate(grads, &varmap.all_vars())?;
        assert!(grad_accum.gradient_norm()? > 0.0);

        // Create a simple target tensor of ones to overfit to
        let target = Tensor::ones(&[batch_size, 16], DType::F32, vb.device())?;

        // Create optimizer
        let mut opt = AdamW::new_lr(varmap.all_vars(), 1e-3)?;

        // Training loop
        let mut loss_history = Vec::new();
        for _ in 0..50 {
            let out = encoder.soft_forward(&input)?;
            let loss = out.sub(&target)?.abs()?.sum_all()?;
            let grads = loss.backward()?;
            opt.step(&grads)?;
            loss_history.push(loss.to_vec0::<f32>()?);
        }

        // Assert that the loss decreased during training
        let initial_loss = loss_history[0];
        let final_loss = loss_history[loss_history.len() - 1];
        assert!(
            final_loss < initial_loss,
            "Loss did not decrease: initial={}, final={}",
            initial_loss,
            final_loss
        );

        Ok(())
    }

    #[test]
    fn test_tetris_world_model_forward() -> Result<()> {
        let device = Device::Cpu;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let embedding_dim = 16;

        let tokenizer_cfg = TetrisWorldModelTokenizerConfig {
            board_encoder_config: TetrisBoardEncoderTransformerConfig {
                board_embedding_config: (NUM_TETRIS_CELL_STATES, 16),
                blocks_config: TetrisGameTransformerBlockConfig {
                    dyn_tan_config: DynamicTanhConfig {
                        alpha_init_value: 1.0,
                        normalized_shape: 16,
                    },
                    attn_config: CausalSelfAttentionConfig {
                        d_model: 16,
                        n_attention_heads: 1,
                        n_kv_heads: 1,
                        rope_theta: 10000.0,
                        max_position_embeddings: TetrisBoardRaw::SIZE,
                    },
                    mlp_config: MlpConfig {
                        hidden_size: 16,
                        intermediate_size: 16,
                        output_size: 16,
                    },
                },
                num_blocks: 3,
                dyn_tanh_1_config: DynamicTanhConfig {
                    alpha_init_value: 1.0,
                    normalized_shape: 16,
                },
                mlp_1_config: MlpConfig {
                    hidden_size: TetrisBoardRaw::SIZE,
                    intermediate_size: 16,
                    output_size: 1,
                },
                dyn_tanh_2_config: DynamicTanhConfig {
                    alpha_init_value: 1.0,
                    normalized_shape: 16,
                },
                mlp_2_config: MlpConfig {
                    hidden_size: 16,
                    intermediate_size: 16,
                    output_size: embedding_dim,
                },
            },
            piece_embedding_dim: embedding_dim,
            placement_embedding_dim: embedding_dim,
            token_encoder_config: MlpConfig {
                hidden_size: 3 * embedding_dim,
                intermediate_size: 3 * embedding_dim,
                output_size: embedding_dim,
            },
        };

        let tokenizer = TetrisWorldModelTokenizer::init(&vb, &tokenizer_cfg)?;

        let model_cfg = TetrisWorldModelConfig {
            blocks_config: TetrisWorldModelBlockConfig {
                dyn_tan_config: DynamicTanhConfig {
                    alpha_init_value: 1.0,
                    normalized_shape: embedding_dim,
                },
                attn_config: CausalSelfAttentionConfig {
                    d_model: embedding_dim,
                    n_attention_heads: 4,
                    n_kv_heads: 4,
                    rope_theta: 10000.0,
                    max_position_embeddings: 128,
                },
                mlp_config: MlpConfig {
                    hidden_size: embedding_dim,
                    intermediate_size: embedding_dim,
                    output_size: embedding_dim,
                },
            },
            num_blocks: 4,
            orientation_head_config: MlpConfig {
                hidden_size: embedding_dim,
                intermediate_size: embedding_dim,
                output_size: TetrisPieceOrientation::NUM_ORIENTATIONS,
            },
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: embedding_dim,
            },
        };

        let model = TetrisWorldModel::init(&vb, &model_cfg)?;

        let dataset_generator = TetrisDatasetGenerator::new();
        let batch_size = 4;
        let sequence_length = 10;
        let datum_sequence = dataset_generator.gen_sequence(
            (0..8).into(),
            batch_size,
            sequence_length,
            &device,
            &mut rand::rng(),
        )?;

        let goal_board = TetrisBoardsDistTensor::try_from(
            datum_sequence.current_boards.last().unwrap().clone(),
        )?;
        let _current_board =
            TetrisBoardsDistTensor::try_from(datum_sequence.current_boards[0].clone())?;
        let current_pieces_tensor =
            TetrisPieceTensor::from_pieces(&datum_sequence.pieces[0], &device)?;

        let context_tokens = tokenizer.forward_context(
            &goal_board,
            &datum_sequence
                .current_boards
                .iter()
                .map(|b| Ok(TetrisBoardsDistTensor::try_from(b.clone())?))
                .collect::<Result<Vec<TetrisBoardsDistTensor>>>()?,
            &datum_sequence
                .pieces
                .iter()
                .map(|p| Ok(TetrisPieceTensor::from_pieces(p, &device)?))
                .collect::<Result<Vec<TetrisPieceTensor>>>()?,
            &datum_sequence.placements,
        )?;
        assert_eq!(
            context_tokens.shape().dims(),
            &[batch_size, 2 + sequence_length, embedding_dim]
        );

        let (board_logits, orientation_logits) = model
            .forward(&context_tokens, &current_pieces_tensor, false)
            .unwrap();
        assert_eq!(
            board_logits.shape().dims(),
            &[
                batch_size,
                TetrisBoardRaw::SIZE,
                TetrisBoardRaw::NUM_TETRIS_CELL_STATES
            ]
        );
        assert_eq!(
            orientation_logits.shape().dims(),
            &[batch_size, TetrisPieceOrientation::NUM_ORIENTATIONS]
        );

        let pieces_tensor = {
            // [..] -> [B*S, 1] -> [B, S]
            let pieces_seq = datum_sequence
                .pieces
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            let pieces_seq = TetrisPieceTensor::from_pieces(&pieces_seq.as_slice(), &device)?;
            &pieces_seq.reshape(&[batch_size, sequence_length])?
        };

        let (board_logits_all, orientation_logits_all) =
            model.forward_all(&context_tokens, &pieces_tensor, false)?;
        assert_eq!(
            board_logits_all.shape().dims(),
            &[
                batch_size,
                2 + sequence_length,
                TetrisBoardRaw::SIZE,
                TetrisBoardRaw::NUM_TETRIS_CELL_STATES
            ]
        );
        assert_eq!(
            orientation_logits_all.shape().dims(),
            &[
                batch_size,
                2 + sequence_length,
                TetrisPieceOrientation::NUM_ORIENTATIONS
            ]
        );

        Ok(())
    }
}
