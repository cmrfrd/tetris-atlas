use clap::Parser;
use std::str::FromStr;
use tetris_ml::set_global_threadpool;
use tracing::{Level, info};
use tracing_subscriber::prelude::*;

use std::{collections::HashMap, ops::Deref, path::PathBuf};

use candle_core::{DType, Device};
use candle_nn::{AdamW, Embedding, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, embedding};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tensorboard::summary_writer::SummaryWriter;
use tracing::debug;

use tetris_game::{TetrisBoard, TetrisPiece, TetrisPieceOrientation};
use tetris_ml::{
    checkpointer::Checkpointer,
    data::TetrisDatasetGenerator,
    grad_accum::GradientAccumulator,
    modules::{
        AttnMlp, AttnMlpConfig, Conv2dConfig, ConvBlockSpec, ConvEncoder, ConvEncoderConfig, FiLM,
        FiLMConfig, Mlp, MlpConfig,
    },
};
use tetris_ml::ops::{binary_cross_entropy_with_logits_stable, get_l2_norm};
use tetris_ml::tensors::{
    TetrisBoardLogitsTensor, TetrisBoardsTensor, TetrisPieceOrientationTensor, TetrisPieceTensor,
};
use tetris_ml::wrapped_tensor::WrappedTensor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetrisGameTransitionModelConfig {
    pub piece_embedding_config: (usize, usize),
    pub orientation_embedding_config: (usize, usize),

    pub board_encoder_config: ConvEncoderConfig,

    pub piece_film_config: FiLMConfig,
    pub orientation_film_config: FiLMConfig,

    pub mlp_config: AttnMlpConfig,
}

#[derive(Debug, Clone)]
pub struct TetrisGameTransitionModel {
    piece_embedding: Embedding,
    orientation_embedding: Embedding,

    board_encoder: ConvEncoder,

    piece_film: FiLM,
    orientation_film: FiLM,

    mlp: AttnMlp,
}

impl TetrisGameTransitionModel {
    pub fn init(
        vb: &VarBuilder,
        cfg: &TetrisGameTransitionModelConfig,
    ) -> Result<TetrisGameTransitionModel> {
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
        assert_eq!(
            cfg.piece_embedding_config.1, cfg.orientation_embedding_config.1,
            "Piece and orientation embedding sizes must match"
        );

        let board_encoder = ConvEncoder::init(vb, &cfg.board_encoder_config)?;
        let piece_film = FiLM::init(vb, &cfg.piece_film_config)?;
        let orientation_film = FiLM::init(vb, &cfg.orientation_film_config)?;
        let mlp = AttnMlp::init(vb, &cfg.mlp_config)?;
        Ok(TetrisGameTransitionModel {
            piece_embedding,
            orientation_embedding,
            board_encoder,
            piece_film,
            orientation_film,
            mlp,
        })
    }

    pub fn forward(
        &self,
        current_board: &TetrisBoardsTensor,
        piece: &TetrisPieceTensor,
        orientation: &TetrisPieceOrientationTensor,
    ) -> Result<TetrisBoardLogitsTensor> {
        let (b, _) = current_board.shape_tuple();
        let current_board = current_board
            .reshape(&[b, 1, TetrisBoard::HEIGHT, TetrisBoard::WIDTH])?
            .to_dtype(DType::F32)?;

        // [B, 1, H, W] -> [B, D]
        let board_embedding = self.board_encoder.forward(&current_board)?;
        // [B, 1] -> [B, 1, D] -> [B, D]
        let piece_embedding = self.piece_embedding.forward(&piece)?.squeeze(1)?;
        // [B, 1] -> [B, 1, D] -> [B, D]
        let orientation_embedding = self
            .orientation_embedding
            .forward(&orientation)?
            .squeeze(1)?;

        let x = self
            .piece_film
            .forward(&board_embedding, &piece_embedding)?;
        let x = self.orientation_film.forward(&x, &orientation_embedding)?;
        let x = self.mlp.forward(&x)?;
        let x = TetrisBoardLogitsTensor::try_from(x)?;
        Ok(x)
    }
}

pub fn train_game_transition_model(
    run_name: String,
    logdir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
) -> Result<()> {
    info!("Training world model");
    const NUM_ITERATIONS: usize = 10_000;
    const BATCH_SIZE: usize = 32;
    const ACCUMULATE_GRADIENTS_STEPS: usize = 1;
    const CHECKPOINT_INTERVAL: usize = 1_000;
    const CLIP_GRAD_MAX_NORM: f64 = 0.1;
    let model_dim = 8;
    let mut summary_writer = logdir.map(|s| SummaryWriter::new(s));

    // let device = Device::Cpu;
    let device = Device::new_cuda(0).unwrap();

    let model_varmap = VarMap::new();
    let model_vs = VarBuilder::from_varmap(&model_varmap, tetris_ml::fdtype(), &device);

    let model_cfg = TetrisGameTransitionModelConfig {
        piece_embedding_config: (TetrisPiece::NUM_PIECES, model_dim),
        orientation_embedding_config: (TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS, model_dim),
        board_encoder_config: ConvEncoderConfig {
            blocks: vec![
                ConvBlockSpec {
                    in_channels: 1,
                    out_channels: 128,
                    kernel_size: 3,
                    conv_cfg: Conv2dConfig {
                        padding: 1,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                    },
                    gn_groups: 64,
                },
                ConvBlockSpec {
                    in_channels: 128,
                    out_channels: 128,
                    kernel_size: 3,
                    conv_cfg: Conv2dConfig {
                        padding: 1,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                    },
                    gn_groups: 64,
                },
                ConvBlockSpec {
                    in_channels: 128,
                    out_channels: 64,
                    kernel_size: 3,
                    conv_cfg: Conv2dConfig {
                        padding: 1,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                    },
                    gn_groups: 32,
                },
                ConvBlockSpec {
                    in_channels: 64,
                    out_channels: 32,
                    kernel_size: 3,
                    conv_cfg: Conv2dConfig {
                        padding: 1,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                    },
                    gn_groups: 16,
                },
                ConvBlockSpec {
                    in_channels: 32,
                    out_channels: 16,
                    kernel_size: 3,
                    conv_cfg: Conv2dConfig {
                        padding: 1,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                    },
                    gn_groups: 8,
                },
                ConvBlockSpec {
                    in_channels: 16,
                    out_channels: 8,
                    kernel_size: 3,
                    conv_cfg: Conv2dConfig {
                        padding: 1,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                    },
                    gn_groups: 8,
                },
            ],
            input_hw: (TetrisBoard::HEIGHT, TetrisBoard::WIDTH),
            mlp: AttnMlpConfig {
                hidden_size: 1920,
                intermediate_size: 4 * model_dim,
                output_size: model_dim,
                dropout: None,
            },
        },

        piece_film_config: FiLMConfig {
            cond_dim: model_dim,
            feat_dim: model_dim,
            hidden: 2 * model_dim,
            output_dim: model_dim,
        },
        orientation_film_config: FiLMConfig {
            cond_dim: model_dim,
            feat_dim: model_dim,
            hidden: 2 * model_dim,
            output_dim: model_dim,
        },

        mlp_config: AttnMlpConfig {
            hidden_size: model_dim,
            intermediate_size: 2 * model_dim,
            output_size: TetrisBoard::SIZE,
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
    let datum_rx =
        data_generator.spawn_transition_channel((3..18).into(), BATCH_SIZE, device.clone(), 256);

    let mut total_boards = 0;
    let start_time = std::time::Instant::now();

    for i in 0..NUM_ITERATIONS {
        let span = tracing::info_span!("datum_recv", iteration = i, batch = BATCH_SIZE);
        let _enter = span.enter();
        let datum = datum_rx.recv().expect("prefetch thread stopped");

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

        let grads = (bce_loss + (mean_entropy_loss * 0.01)?)?.backward()?;
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
            s.add_scalars(
                "board_sample_accuracy",
                &HashMap::from([
                    ("sample_0.0".to_string(), sample_0),
                    (
                        "sample_0.1".to_string(),
                        target
                            .perc_cells_equal(&output_logits.sample(0.1).unwrap())
                            .unwrap(),
                    ),
                    (
                        "sample_0.2".to_string(),
                        target
                            .perc_cells_equal(&output_logits.sample(0.2).unwrap())
                            .unwrap(),
                    ),
                ]),
                i,
            );
            s.add_scalar("grad_norm", grad_norm, i);

            let conv_imgs = model.board_encoder.get_conv_filters().unwrap();
            for (j, img) in conv_imgs.iter().enumerate() {
                let tag = format!("conv_filter_{}", j);
                let data = img.deref();
                let dim = [3, img.height() as usize, img.width() as usize];
                let step = i;
                s.add_image(&tag, &data, &dim, step);
            }
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
            let _ = checkpointer.checkpoint_item(i, &model_varmap, None, None);
        }
    }

    if let Some(ref checkpointer) = checkpointer {
        let _ = checkpointer.force_checkpoint_item(NUM_ITERATIONS, &model_varmap, None, None);
    }

    Ok(())
}

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count, help = "Increase verbosity level (-v = ERROR, -vv = WARN, -vvv = INFO, -vvvv = DEBUG, -vvvvv = TRACE)")]
    verbose: u8,

    #[arg(long, help = "Path to save the tensorboard logs")]
    logdir: Option<String>,

    #[arg(long, help = "Path to load/save model checkpoints")]
    checkpoint_dir: Option<String>,

    #[arg(long, help = "Name of the training run")]
    run_name: String,
}

fn main() {
    info!("Starting tetris transition model training");
    set_global_threadpool();

    let cli = Cli::parse();

    let verbosity = cli.verbose.saturating_add(2).clamp(0, 5);
    let level = Level::from_str(verbosity.to_string().as_str()).unwrap();

    let registry = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .with(tracing_subscriber::filter::LevelFilter::from_level(level));

    registry.init();
    info!("Logging initialized at level: {}", level);

    let ulid = ulid::Ulid::new().to_string();
    let run_name = format!("{}_{}", cli.run_name, ulid);
    let logdir = cli.logdir.as_ref().map(|s| {
        let path = std::path::Path::new(s).join(&run_name);
        std::fs::create_dir_all(&path).expect("Failed to create log directory");
        path
    });
    let checkpoint_dir = cli.checkpoint_dir.as_ref().map(|s| {
        let path = std::path::Path::new(s).join(&run_name);
        std::fs::create_dir_all(&path).expect("Failed to create checkpoint directory");
        path
    });

    train_game_transition_model(run_name, logdir, checkpoint_dir).unwrap();
}
