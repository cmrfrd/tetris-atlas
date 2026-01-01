use std::path::PathBuf;

use anyhow::Result;
use candle_core::{D, DType, Shape, Tensor};
use candle_nn::{AdamW, Embedding, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, embedding};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tensorboard::summary_writer::SummaryWriter;
use tracing::info;

use crate::beam_search::TetrisGameIter;
use crate::checkpointer::Checkpointer;
use crate::grad_accum::{GradientAccumulator, get_l2_norm};
use crate::modules::{
    CausalSelfAttentionConfig, DynamicTanhConfig, Mlp, MlpConfig, TransformerBlockConfig,
    TransformerBody, TransformerBodyConfig,
};
use crate::ops::create_orientation_mask;
use crate::tensors::{
    TetrisBoardsTensor, TetrisPieceOrientationLogitsTensor, TetrisPieceOrientationTensor,
    TetrisPieceTensor,
};
use crate::tetris::{TetrisBoard, TetrisPiece, TetrisPieceOrientation};
use crate::wrapped_tensor::WrappedTensor;
use crate::{device, fdtype};

/// Supervised policy trained on BeamSearch-generated actions.
///
/// Architecture mirrors `tetris_policy_gradients.rs` transformer policy, but **no value head**.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetrisBeamSupervisedPolicyConfig {
    pub d_model: usize,
    pub num_blocks: usize,
    pub attn_config: CausalSelfAttentionConfig,
    pub block_mlp_config: MlpConfig,
    pub head_mlp_config: MlpConfig,
    #[serde(default)]
    pub with_causal_mask: bool,
}

#[derive(Debug, Clone)]
pub struct TetrisBeamSupervisedPolicy {
    token_embed: Embedding, // [0/1] -> D
    pos_embed: Embedding,   // [0..199] -> D
    piece_embed: Embedding, // piece id -> D
    body: TransformerBody,
    head_mlp: Mlp, // pooled -> logits over orientations
    #[allow(dead_code)]
    d_model: usize,
    with_causal_mask: bool,
}

impl TetrisBeamSupervisedPolicy {
    pub fn init(vb: &VarBuilder, cfg: &TetrisBeamSupervisedPolicyConfig) -> Result<Self> {
        let token_embed = embedding(
            TetrisBoard::NUM_TETRIS_CELL_STATES,
            cfg.d_model,
            vb.pp("token_embed"),
        )?;
        let pos_embed = embedding(TetrisBoard::SIZE, cfg.d_model, vb.pp("pos_embed"))?;
        let piece_embed = embedding(TetrisPiece::NUM_PIECES, cfg.d_model, vb.pp("piece_embed"))?;

        let body_cfg = TransformerBodyConfig {
            blocks_config: TransformerBlockConfig {
                dyn_tanh_config: DynamicTanhConfig {
                    alpha_init_value: 1.0,
                    normalized_shape: cfg.d_model,
                },
                attn_config: cfg.attn_config.clone(),
                mlp_config: cfg.block_mlp_config.clone(),
            },
            num_blocks: cfg.num_blocks,
            dyn_tanh_config: DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: cfg.d_model,
            },
        };
        let body = TransformerBody::init(&vb.pp("transformer_body"), &body_cfg)?;
        let head_mlp = Mlp::init(&vb.pp("head_mlp"), &cfg.head_mlp_config)?;

        Ok(Self {
            token_embed,
            pos_embed,
            piece_embed,
            body,
            head_mlp,
            d_model: cfg.d_model,
            with_causal_mask: cfg.with_causal_mask,
        })
    }

    fn forward_pooled(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<Tensor> {
        let (b, _t) = current_board.shape_tuple(); // [B, 200]
        let tokens = current_board.inner().to_dtype(DType::U32)?;

        // Token + pos embeddings: [B, 200, D]
        let x_tokens = self.token_embed.forward(&tokens)?;
        let device = tokens.device();
        let pos_ids = Tensor::arange(0, TetrisBoard::SIZE as u32, device)?
            .to_dtype(DType::U32)?
            .reshape(&[1, TetrisBoard::SIZE])?
            .repeat(&[b, 1])?;
        let x_pos = self.pos_embed.forward(&pos_ids)?;

        // Piece conditioning: [B, D] -> [B, 200, D]
        let piece_embed = self.piece_embed.forward(current_piece)?.squeeze(1)?;
        let piece_b = piece_embed.unsqueeze(1)?.broadcast_as(x_tokens.dims())?;

        let x = ((&x_tokens + &x_pos)? + &piece_b)?;
        let x = self.body.forward(x, self.with_causal_mask)?;
        Ok(x.mean(1)?)
    }

    pub fn forward_logits(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationLogitsTensor> {
        let pooled = self.forward_pooled(current_board, current_piece)?;
        let logits = self.head_mlp.forward(&pooled)?;
        TetrisPieceOrientationLogitsTensor::try_from(logits)
    }

    /// Mask invalid orientations for the current piece.
    pub fn forward_masked(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationLogitsTensor> {
        let logits = self.forward_logits(current_board, current_piece)?;
        let mask = create_orientation_mask(current_piece)?;
        let device = logits.device();
        let zero = mask.zeros_like()?;
        let keep = mask.gt(&zero)?;
        let neg_inf = Tensor::new(-1e9f32, &device)?.broadcast_as(logits.dims())?;
        let masked = keep.where_cond(logits.inner(), &neg_inf)?;
        TetrisPieceOrientationLogitsTensor::try_from(masked)
    }
}

/// Train a transformer policy with supervised learning on BeamSearch-generated labels.
pub fn train_tetris_beam_supervised(
    run_name: String,
    logdir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
    resume: bool,
) -> Result<()> {
    let device = device();
    let dtype = fdtype();

    // --- Data generation (teacher) ---
    const TEACHER_BEAM_WIDTH: usize = 128;
    const TEACHER_DEPTH: usize = 8;
    const TEACHER_MAX_MOVES: usize = 64; // safe upper bound for placements
    const SEED: u64 = 123;

    // --- Training hyperparams ---
    const NUM_ITERATIONS: usize = 50_000;
    const BATCH_SIZE: usize = 512;
    const CHECKPOINT_INTERVAL: usize = 200;
    const LEARNING_RATE: f64 = 1e-4;
    const CLIP_GRAD_MAX_NORM: Option<f64> = Some(1.0);
    const CLIP_GRAD_MAX_VALUE: Option<f64> = None;

    // Model hyperparams
    let model_dim = 64;
    let num_blocks = 4;

    let mut model_varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&model_varmap, dtype, &device);

    let policy_cfg = TetrisBeamSupervisedPolicyConfig {
        d_model: model_dim,
        num_blocks,
        attn_config: CausalSelfAttentionConfig {
            d_model: model_dim,
            n_attention_heads: 8,
            n_kv_heads: 8,
            rope_theta: 10_000.0,
            max_position_embeddings: TetrisBoard::SIZE,
        },
        block_mlp_config: MlpConfig {
            hidden_size: model_dim,
            intermediate_size: 2 * model_dim,
            output_size: model_dim,
            dropout: None,
        },
        head_mlp_config: MlpConfig {
            hidden_size: model_dim,
            intermediate_size: 2 * model_dim,
            output_size: TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS,
            dropout: None,
        },
        with_causal_mask: false,
    };

    let policy = TetrisBeamSupervisedPolicy::init(&vb, &policy_cfg)?;
    let model_params = model_varmap.all_vars();
    let mut optimizer = AdamW::new(
        model_params.clone(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            ..ParamsAdamW::default()
        },
    )?;
    let mut grad_accum = GradientAccumulator::new(1);

    let mut summary_writer = logdir.map(SummaryWriter::new);
    let checkpointer = checkpoint_dir.as_ref().map(|dir| {
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(
            dir.join("beam_supervised_policy_config.json"),
            serde_json::to_string_pretty(&policy_cfg).unwrap(),
        );
        Checkpointer::new(CHECKPOINT_INTERVAL, dir.clone(), run_name.clone())
            .expect("Failed to create checkpointer")
    });

    if resume {
        if let Some(cp) = checkpointer.as_ref() {
            let _ = cp.load_latest_checkpoint(&mut model_varmap)?;
        }
    }

    // Teacher iterator: uses BeamSearch to pick actions.
    let mut teacher =
        TetrisGameIter::<TEACHER_BEAM_WIDTH, TEACHER_DEPTH, TEACHER_MAX_MOVES>::new_with_seed(SEED);

    info!("Starting BeamSupervised training: iterations={NUM_ITERATIONS} batch={BATCH_SIZE}");

    let mut rng = rand::rng();

    for iteration in 0..NUM_ITERATIONS {
        let mut batch = (&mut teacher).take(BATCH_SIZE).collect::<Vec<_>>();
        batch.shuffle(&mut rng);

        let boards = batch.iter().map(|(b, _p)| *b).collect::<Vec<_>>();
        let board_tensor = TetrisBoardsTensor::from_boards(&boards, &device)?;
        let pieces = batch.iter().map(|(_, p)| p.piece).collect::<Vec<_>>();
        let piece_tensor = TetrisPieceTensor::from_pieces(&pieces, &device)?;
        let targets = batch.iter().map(|(_, p)| p.orientation).collect::<Vec<_>>();
        let target_tensor = TetrisPieceOrientationTensor::from_orientations(&targets, &device)?;

        // Forward
        let logits = policy.forward_masked(&board_tensor, &piece_tensor)?;
        let probs = candle_nn::ops::softmax(logits.inner(), D::Minus1)?;
        let log_probs = candle_nn::ops::log_softmax(logits.inner(), D::Minus1)?;

        // Cross-entropy via one-hot targets: -E[log π(a|s)]
        let target_one_hot = target_tensor.into_dist()?;
        let nll = (&log_probs * target_one_hot.inner())?
            .sum(D::Minus1)?
            .neg()?; // [B]
        let loss = nll.mean_all()?;

        // Metrics
        let entropy = (&probs * &log_probs)?.sum(D::Minus1)?.neg()?;
        let avg_entropy = entropy.mean_all()?;
        let predicted = logits.inner().argmax(D::Minus1)?;
        let target_indices: Vec<u32> = targets.iter().map(|o| o.index() as u32).collect();
        let target_indices_tensor =
            Tensor::from_vec(target_indices, Shape::from_dims(&[BATCH_SIZE]), &device)?;
        let correct = predicted
            .eq(&target_indices_tensor)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let accuracy = correct.to_scalar::<f32>()? / BATCH_SIZE as f32;

        let loss_scalar = loss.to_scalar::<f32>()?;
        let entropy_scalar = avg_entropy.to_scalar::<f32>()?;

        if !loss_scalar.is_finite() {
            eprintln!("⚠️  Non-finite loss at iter {iteration}, skipping");
            continue;
        }

        // Backward
        let grads = loss.backward()?;
        let grad_norm = get_l2_norm(&grads)?;
        grad_accum.accumulate(grads, &model_params)?;
        grad_accum.apply_and_reset(
            &mut optimizer,
            &model_params,
            CLIP_GRAD_MAX_NORM,
            CLIP_GRAD_MAX_VALUE,
        )?;

        // Log
        if let Some(s) = summary_writer.as_mut() {
            s.add_scalar("loss/supervised", loss_scalar, iteration);
            s.add_scalar("metrics/accuracy", accuracy, iteration);
            s.add_scalar("metrics/entropy", entropy_scalar, iteration);
            s.add_scalar("metrics/grad_norm", grad_norm as f32, iteration);
        }

        // Checkpoint
        if let Some(cp) = checkpointer.as_ref() {
            let _ = cp.checkpoint_item(iteration, &model_varmap, None, None)?;
        }

        println!(
            "iter={iteration} loss={loss_scalar:.4} acc={accuracy:.3} ent={entropy_scalar:.3} grad_norm={grad_norm:.3}"
        );
    }

    Ok(())
}
