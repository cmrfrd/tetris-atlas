use std::path::PathBuf;

use crate::data::TetrisTransition;
use crate::optim::{AdamW, ParamsAdamW};
use candle_core::{DType, Device, Tensor};
use candle_nn::Optimizer;
use candle_nn::{Embedding, Module, VarBuilder, VarMap, embedding};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tensorboard::summary_writer::SummaryWriter;
use tracing::{debug, info};

use crate::{
    checkpointer::Checkpointer,
    data::TetrisDatasetGenerator,
    grad_accum::{GradientAccumulator, get_l2_norm},
    modules::{
        CausalSelfAttentionConfig, DynamicTanhConfig, FiLM, FiLMConfig, Mlp, MlpConfig,
        TransformerBlockConfig, TransformerBody, TransformerBodyConfig,
    },
    ops::binary_cross_entropy_with_logits_stable,
    tensors::{
        TetrisBoardLogitsTensor, TetrisBoardsTensor, TetrisPieceOrientationTensor,
        TetrisPieceTensor,
    },
    tetris::{NUM_TETRIS_CELL_STATES, TetrisBoard, TetrisPiece, TetrisPieceOrientation},
    wrapped_tensor::WrappedTensor,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetrisGameTransitionModelConfig {
    pub piece_embedding_config: (usize, usize),
    pub orientation_embedding_config: (usize, usize),

    pub piece_film_config: FiLMConfig,
    pub orientation_film_config: FiLMConfig,

    pub d_model: usize,
    pub num_blocks: usize,
    pub attn_config: CausalSelfAttentionConfig, // d_model, n_heads, kv_heads, rope, max_pos
    pub block_mlp_config: MlpConfig,            // hidden=d_model, inter, output=d_model
    pub cell_head_config: MlpConfig,            // hidden=d_model, inter, output=1
    pub with_causal_mask: bool,
}

#[derive(Debug, Clone)]
pub struct TetrisGameTransitionModel {
    // tokenization of the board cells
    token_embed: Embedding, // [0/1] -> D
    pos_embed: Embedding,   // [0..T-1] -> D

    // conditioning
    piece_embedding: Embedding,       // [B,1] -> [B,1,D]
    orientation_embedding: Embedding, // [B,1] -> [B,1,D]
    piece_film: FiLM,                 // applies to [B,T,D]
    orientation_film: FiLM,           // applies to [B,T,D]

    // transformer body
    body: TransformerBody,

    // output head per token -> one logit per cell
    cell_head: Mlp, // D -> 1

    // config
    with_causal_mask: bool,
}

impl TetrisGameTransitionModel {
    pub fn init(
        vb: &VarBuilder,
        cfg: &TetrisGameTransitionModelConfig,
    ) -> Result<TetrisGameTransitionModel> {
        // Validate dims
        assert_eq!(
            cfg.piece_embedding_config.1, cfg.d_model,
            "piece embedding dim must equal d_model"
        );
        assert_eq!(
            cfg.orientation_embedding_config.1, cfg.d_model,
            "orientation embedding dim must equal d_model"
        );
        assert_eq!(
            cfg.block_mlp_config.hidden_size, cfg.d_model,
            "block mlp hidden_size must equal d_model"
        );
        assert_eq!(
            cfg.block_mlp_config.output_size, cfg.d_model,
            "block mlp output_size must equal d_model"
        );
        assert_eq!(
            cfg.attn_config.d_model, cfg.d_model,
            "attn d_model must equal d_model"
        );

        // Board token + positional embeddings
        let token_embed = embedding(NUM_TETRIS_CELL_STATES, cfg.d_model, vb.pp("token_embed"))?;
        let pos_embed = embedding(TetrisBoard::SIZE, cfg.d_model, vb.pp("pos_embed"))?;

        // Conditioning embeddings
        let piece_embedding = embedding(
            cfg.piece_embedding_config.0,
            cfg.piece_embedding_config.1,
            vb.pp("piece_embedding"),
        )?;
        let orientation_embedding = embedding(
            cfg.orientation_embedding_config.0,
            cfg.orientation_embedding_config.1,
            vb.pp("orientation_embedding"),
        )?;

        // Conditioning FiLMs over token sequence
        let piece_film = FiLM::init(&vb.pp("piece_film"), &cfg.piece_film_config)?;
        let orientation_film =
            FiLM::init(&vb.pp("orientation_film"), &cfg.orientation_film_config)?;

        // Transformer body
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

        // Output head: per-token to single logit
        let cell_head = Mlp::init(&vb.pp("cell_head"), &cfg.cell_head_config)?;

        Ok(TetrisGameTransitionModel {
            token_embed,
            pos_embed,
            piece_embedding,
            orientation_embedding,
            piece_film,
            orientation_film,
            body,
            cell_head,
            with_causal_mask: cfg.with_causal_mask,
        })
    }

    pub fn forward(
        &self,
        current_board: &TetrisBoardsTensor,
        piece: &TetrisPieceTensor,
        orientation: &TetrisPieceOrientationTensor,
    ) -> Result<TetrisBoardLogitsTensor> {
        let (batch_size, board_size) = current_board.shape_tuple();
        assert_eq!(board_size, TetrisBoard::SIZE, "Unexpected board size");

        // Token + positional embeddings
        let tokens = current_board.inner().to_dtype(DType::U32)?; // [B, T]
        let x_tokens = self.token_embed.forward(&tokens)?; // [B, T, D]

        let pos_ids = Tensor::arange(0, TetrisBoard::SIZE as u32, tokens.device())?
            .to_dtype(DType::U32)?
            .reshape(&[1, TetrisBoard::SIZE])?
            .repeat(&[batch_size, 1])?; // [B, T]
        let x_pos = self.pos_embed.forward(&pos_ids)?; // [B, T, D]

        let mut x = (&x_tokens + &x_pos)?; // [B, T, D]

        // Conditioning with FiLM using piece and orientation embeddings
        let piece_embed = self.piece_embedding.forward(piece)?.squeeze(1)?; // [B, D]
        let orient_embed = self
            .orientation_embedding
            .forward(orientation)?
            .squeeze(1)?; // [B, D]
        x = self.piece_film.forward(&x, &piece_embed)?; // [B, T, D]
        x = self.orientation_film.forward(&x, &orient_embed)?; // [B, T, D]

        // Transformer body over sequence of cells
        x = self.body.forward(x, self.with_causal_mask)?; // [B, T, D]

        // Per-token cell head to single logit per cell
        let logits = self.cell_head.forward(&x)?.squeeze(2)?; // [B, T]
        TetrisBoardLogitsTensor::try_from(logits)
    }
}

pub fn train_game_transition_model(
    run_name: String,
    logdir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
) -> Result<()> {
    info!("Training world model");
    const NUM_ITERATIONS: usize = 1_000_000;
    const BATCH_SIZE: usize = 128;
    const ACCUMULATE_GRADIENTS_STEPS: usize = 16;
    const CHECKPOINT_INTERVAL: usize = 10_000;
    const ENTROPY_LOSS_WEIGHT: f64 = 0.001;
    const CLIP_GRAD_MAX_NORM: f64 = 0.1;

    let model_dim = 32;

    let mut summary_writer = logdir.map(|s| SummaryWriter::new(s));

    // let device = Device::Cpu;
    let device = Device::new_cuda(0).unwrap();

    let model_varmap = VarMap::new();
    let model_vs = VarBuilder::from_varmap(&model_varmap, DType::F32, &device);

    let model_cfg = TetrisGameTransitionModelConfig {
        piece_embedding_config: (TetrisPiece::NUM_PIECES, model_dim),
        orientation_embedding_config: (TetrisPieceOrientation::NUM_ORIENTATIONS, model_dim),

        piece_film_config: FiLMConfig {
            cond_dim: model_dim,
            feat_dim: model_dim,
            hidden: model_dim,
            output_dim: model_dim,
        },
        orientation_film_config: FiLMConfig {
            cond_dim: model_dim,
            feat_dim: model_dim,
            hidden: model_dim,
            output_dim: model_dim,
        },

        d_model: model_dim,
        num_blocks: 8,
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
        with_causal_mask: false,

        cell_head_config: MlpConfig {
            hidden_size: model_dim,
            intermediate_size: 2 * model_dim,
            output_size: 1,
            dropout: None,
        },
    };

    let checkpointer = checkpoint_dir.as_ref().map(|dir| {
        let config_path = dir.join("model_config.json");
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(
            &config_path,
            serde_json::to_string_pretty(&model_cfg).unwrap(),
        );
        Checkpointer::new(CHECKPOINT_INTERVAL, dir.clone(), run_name.clone())
            .expect("Failed to create checkpointer")
    });

    let model = TetrisGameTransitionModel::init(&model_vs, &model_cfg)?;
    let mut model_optimizer = AdamW::new(model_varmap.all_vars(), ParamsAdamW::default()).unwrap();
    let model_params = model_varmap.all_vars();
    let mut model_grad_accumulator = GradientAccumulator::new(ACCUMULATE_GRADIENTS_STEPS);
    info!("Model optimizer and grad accumulator initialized");

    let data_generator = TetrisDatasetGenerator::new();
    let mut rng = rand::rng();

    let mut total_boards = 0;
    let start_time = std::time::Instant::now();

    for i in 0..NUM_ITERATIONS {
        let datum = data_generator
            .gen_uniform_sampled_transition((1..20).into(), BATCH_SIZE, &device, &mut rng)
            .unwrap();

        let current_board = datum.current_board;
        let piece = TetrisPieceTensor::from_pieces(&datum.piece, &device).unwrap();
        let orientation = datum.orientation;

        let output_logits = model.forward(&current_board, &piece, &orientation).unwrap();
        let target = datum.result_board;

        let bce_loss =
            binary_cross_entropy_with_logits_stable(&output_logits.inner(), &target.inner())
                .unwrap();
        let bce_loss_value = bce_loss.to_scalar::<f32>()?;
        let mean_entropy_loss = output_logits.into_dist()?.entropy()?.mean_all()?;
        let mean_entropy_loss_value = mean_entropy_loss.to_scalar::<f32>()?;

        let sample_0 = target
            .perc_cells_equal(&output_logits.sample(0.0).unwrap())
            .unwrap();

        let grads = (bce_loss + (mean_entropy_loss * ENTROPY_LOSS_WEIGHT)?)?.backward()?;
        let grad_norm = get_l2_norm(&grads).unwrap();

        total_boards += BATCH_SIZE;
        let elapsed_time = start_time.elapsed().as_secs_f32();
        let boards_per_sec = (total_boards as f32) / elapsed_time;

        info!(
            "Iteration {} | BCE Loss: {:.4} | Entropy Loss: {:.4} | Batch Accuracy: {:.4} | Grad Norm: {:.4} | Board/Sec: {:.4}",
            i, bce_loss_value, mean_entropy_loss_value, sample_0, grad_norm, boards_per_sec
        );

        summary_writer.as_mut().map(|s| {
            s.add_scalar("loss", bce_loss_value, i);
            s.add_scalar("avg_output_board_entropy", mean_entropy_loss_value, i);
            s.add_scalar("board_sample_accuracy/sample_0.0", sample_0, i);
            s.add_scalar(
                "board_sample_accuracy/sample_0.1",
                target
                    .perc_cells_equal(&output_logits.sample(0.1).unwrap())
                    .unwrap(),
                i,
            );
            s.add_scalar(
                "board_sample_accuracy/sample_0.2",
                target
                    .perc_cells_equal(&output_logits.sample(0.2).unwrap())
                    .unwrap(),
                i,
            );
            s.add_scalar("grad_norm", grad_norm, i);
        });

        model_grad_accumulator
            .accumulate(grads, &model_params)
            .unwrap();

        let should_step = model_grad_accumulator
            .apply_and_reset(
                &mut model_optimizer,
                &model_params,
                Some(CLIP_GRAD_MAX_NORM),
                None,
            )
            .unwrap();

        if should_step {
            debug!("Stepping model");
        }

        if let Some(ref checkpointer) = checkpointer {
            let _ = checkpointer.save_item(i, &model_varmap, None, Some("model"));
            let _ = checkpointer.save_item(i, &model_optimizer, None, Some("optimizer"));
        }
    }

    if let Some(ref checkpointer) = checkpointer {
        let _ =
            checkpointer.force_checkpoint_item(NUM_ITERATIONS, &model_varmap, None, Some("model"));
        let _ = checkpointer.save_item(NUM_ITERATIONS, &model_optimizer, None, Some("optimizer"));
    }

    Ok(())
}
