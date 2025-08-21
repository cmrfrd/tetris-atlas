use crate::ARTIFACT_DIR;
use crate::dataset::{
    CopyRange, TetrisBatcher, TetrisDataset, TetrisDatasetConfig, TetrisDatasetUniformConfig,
    TetrisInitDistBatch, TetrisInitDistBatcher, TetrisInitDistDataset, TetrisSequenceDataset,
    TetrisSequenceDatasetConfig, TetrisSequenceDatasetUniformConfig, TetrisSequenceDistBatcher,
};
use crate::model::{
    CausalSelfAttentionConfig, DynamicTanhConfig, MlpConfig, TetrisGameTransformerBlockConfig,
    TetrisGameTransformerConfig, TetrisPlayerTransformer, TetrisPlayerTransformerConfig,
};
use crate::tetris::{
    BOARD_SIZE, TetrisBoardRaw, TetrisGame, TetrisPiece, TetrisPieceBag, TetrisPiecePlacement,
};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::cast::ToElement;
use burn::train::metric::{AccuracyMetric, LearningRateMetric};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, metric::LossMetric},
};

#[derive(Config)]
pub struct ExpConfig {
    #[config(default = 100)]
    pub num_epochs: usize,

    #[config(default = 10_000)]
    pub items_per_epoch: usize,

    #[config(default = 2)]
    pub num_workers: usize,

    #[config(default = 1337)]
    pub seed: u64,

    #[config(default = 64)]
    pub batch_size: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run_train2<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    create_artifact_dir(artifact_dir);

    // Config
    let tetris_model_optimizer = AdamWConfig::new();
    let tetris_model_config = ExpConfig::new();

    let num_cell_states = 2;
    let num_placements = TetrisPiecePlacement::NUM_PLACEMENTS;
    // let d_model = 32;
    let d_model = 16;
    // let num_transformer_blocks = 8;
    let num_transformer_blocks = 4;
    let num_placement_embedding_residuals = num_transformer_blocks / 2;

    let tetris_game_transformer_config = TetrisGameTransformerConfig {
        board_embedding_config: EmbeddingConfig::new(num_cell_states, d_model),
        placement_embedding_config: EmbeddingConfig::new(num_placements, d_model),
        num_placement_embedding_residuals,
        blocks_config: TetrisGameTransformerBlockConfig {
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 0.5,
                normalized_shape: d_model,
            },
            attn_config: CausalSelfAttentionConfig {
                d_model,
                n_heads: 4,
                rope_theta: 10000.0,
                max_position_embeddings: BOARD_SIZE,
            },
            mlp_config: MlpConfig {
                hidden_size: d_model,
                intermediate_size: d_model * 2,
            },
        },
        num_blocks: num_transformer_blocks,
        dyn_tan_config: DynamicTanhConfig {
            alpha_init_value: 0.5,
            normalized_shape: d_model,
        },
        output_layer_config: LinearConfig::new(d_model, num_cell_states),
    };
    let model = tetris_game_transformer_config.init(&device);

    // Define train/valid datasets and dataloaders
    let train_dataset_config = TetrisDatasetConfig::Uniform(TetrisDatasetUniformConfig {
        seed: 42,
        num_pieces_range: CopyRange { start: 0, end: 32 },
        length: tetris_model_config.items_per_epoch,
    });
    let train_dataset = TetrisDataset::train(train_dataset_config);

    let test_dataset_config = TetrisDatasetConfig::Uniform(TetrisDatasetUniformConfig {
        seed: 42,
        num_pieces_range: CopyRange { start: 0, end: 10 },
        length: 100,
    });
    let test_dataset = TetrisDataset::test(test_dataset_config);

    let valid_dataset_config = TetrisDatasetConfig::Uniform(TetrisDatasetUniformConfig {
        seed: 42,
        num_pieces_range: CopyRange { start: 0, end: 10 },
        length: 100,
    });
    let valid_dataset = TetrisDataset::validation(valid_dataset_config);

    println!("Train Dataset Size: {}", train_dataset.len());
    println!("Test Dataset Size: {}", test_dataset.len());
    println!("Valid Dataset Size: {}", valid_dataset.len());

    let batcher_train = TetrisBatcher::<B>::default();
    let batcher_test = TetrisBatcher::<B::InnerBackend>::default();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(tetris_model_config.batch_size)
        .num_workers(tetris_model_config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(tetris_model_config.batch_size)
        .num_workers(tetris_model_config.num_workers)
        .build(valid_dataset);

    let optimizer = tetris_model_optimizer.init();

    // Model
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![
            device.clone(),
            device.clone(),
            device.clone(),
            device.clone(),
        ])
        .num_epochs(tetris_model_config.num_epochs)
        .grads_accumulation(4)
        .summary()
        .build(model, optimizer, 1e-3);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    tetris_model_config
        .save(format!("{artifact_dir}/config.json").as_str())
        .unwrap();

    model_trained
        .save_file(
            format!("{artifact_dir}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}

pub fn run_train_sequence<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    let artifact_dir = format!("{artifact_dir}_sequence");
    create_artifact_dir(artifact_dir.as_str());

    // Config
    let tetris_model_optimizer = AdamWConfig::new();
    let tetris_model_config = ExpConfig::new();

    let num_cell_states = 2;
    let num_placements = TetrisPiecePlacement::NUM_PLACEMENTS;
    let d_model = 8;
    let num_transformer_blocks = 4;
    let num_placement_embedding_residuals = num_transformer_blocks / 2;
    let n_heads = 4;

    let tetris_game_transformer_config = TetrisGameTransformerConfig {
        board_embedding_config: EmbeddingConfig::new(num_cell_states, d_model),
        placement_embedding_config: EmbeddingConfig::new(num_placements, d_model),
        num_placement_embedding_residuals,
        blocks_config: TetrisGameTransformerBlockConfig {
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 0.5,
                normalized_shape: d_model,
            },
            attn_config: CausalSelfAttentionConfig {
                d_model,
                n_heads,
                rope_theta: 10000.0,
                max_position_embeddings: BOARD_SIZE,
            },
            mlp_config: MlpConfig {
                hidden_size: d_model,
                intermediate_size: d_model * 2,
            },
        },
        num_blocks: num_transformer_blocks,
        dyn_tan_config: DynamicTanhConfig {
            alpha_init_value: 0.5,
            normalized_shape: d_model,
        },
        output_layer_config: LinearConfig::new(d_model, num_cell_states),
    };
    let model = tetris_game_transformer_config.init(&device);
    println!("Number of parameters: {}", model.num_params());

    // Define train/valid datasets and dataloaders
    let train_dataset_config =
        TetrisSequenceDatasetConfig::Uniform(TetrisSequenceDatasetUniformConfig {
            seed: 42,
            num_pieces_range: CopyRange { start: 0, end: 32 },
            length: tetris_model_config.items_per_epoch,
            sequence_length: 10,
        });
    let train_dataset = TetrisSequenceDataset::train(train_dataset_config);

    let test_dataset_config =
        TetrisSequenceDatasetConfig::Uniform(TetrisSequenceDatasetUniformConfig {
            seed: 42,
            num_pieces_range: CopyRange { start: 0, end: 10 },
            length: 100,
            sequence_length: 10,
        });
    let test_dataset = TetrisSequenceDataset::test(test_dataset_config);

    let valid_dataset_config =
        TetrisSequenceDatasetConfig::Uniform(TetrisSequenceDatasetUniformConfig {
            seed: 42,
            num_pieces_range: CopyRange { start: 0, end: 10 },
            length: 100,
            sequence_length: 10,
        });
    let valid_dataset = TetrisSequenceDataset::validation(valid_dataset_config);

    println!("Train Dataset Size: {}", train_dataset.len());
    println!("Test Dataset Size: {}", test_dataset.len());
    println!("Valid Dataset Size: {}", valid_dataset.len());

    let batcher_train = TetrisSequenceDistBatcher::<B>::default();
    let batcher_test = TetrisSequenceDistBatcher::<B::InnerBackend>::default();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(tetris_model_config.batch_size)
        .num_workers(tetris_model_config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(tetris_model_config.batch_size)
        .num_workers(tetris_model_config.num_workers)
        .build(valid_dataset);

    let optimizer = tetris_model_optimizer.init();

    // Model
    let learner = LearnerBuilder::new(artifact_dir.clone())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![
            device.clone(),
            device.clone(),
            device.clone(),
            device.clone(),
        ])
        .num_epochs(tetris_model_config.num_epochs)
        .grads_accumulation(4)
        .summary()
        .build(model, optimizer, 1e-3);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    tetris_model_config
        .save(format!("{artifact_dir}/config.json").as_str())
        .unwrap();

    model_trained
        .save_file(
            format!("{artifact_dir}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}

pub fn run_train<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    let artifact_dir = format!("{artifact_dir}_both");
    create_artifact_dir(artifact_dir.as_str());

    // Config
    let tetris_model_optimizer = AdamWConfig::new();
    let tetris_model_config = ExpConfig::new();

    let num_cell_states = 2;
    let num_placements = TetrisPiecePlacement::NUM_PLACEMENTS;
    let d_model = 16;
    let num_transformer_blocks = 5;
    let n_heads = 8;
    let num_placement_embedding_residuals = num_transformer_blocks / 2;

    let tetris_game_transformer_config = TetrisGameTransformerConfig {
        board_embedding_config: EmbeddingConfig::new(num_cell_states, d_model),
        placement_embedding_config: EmbeddingConfig::new(num_placements, d_model),
        num_placement_embedding_residuals,
        blocks_config: TetrisGameTransformerBlockConfig {
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 0.5,
                normalized_shape: d_model,
            },
            attn_config: CausalSelfAttentionConfig {
                d_model,
                n_heads,
                rope_theta: 10000.0,
                max_position_embeddings: BOARD_SIZE,
            },
            mlp_config: MlpConfig {
                hidden_size: d_model,
                intermediate_size: d_model * 2,
            },
        },
        num_blocks: num_transformer_blocks,
        dyn_tan_config: DynamicTanhConfig {
            alpha_init_value: 0.5,
            normalized_shape: d_model,
        },
        output_layer_config: LinearConfig::new(d_model, num_cell_states),
    };
    let mut tetris_game_model = tetris_game_transformer_config.init(&device);
    let mut tetris_game_optimizer = tetris_model_optimizer.init();
    println!(
        "Tetris Game Model Number of parameters: {}",
        tetris_game_model.num_params()
    );

    let num_pieces = TetrisPiece::NUM_PIECES;
    let d_model = 16;
    let num_transformer_blocks = 4;
    let n_heads = 4;
    let num_placement_embedding_residuals = num_transformer_blocks / 2;
    let tetris_player_transformer_config = TetrisPlayerTransformerConfig {
        board_embedding_config: EmbeddingConfig::new(num_cell_states, d_model),
        piece_embedding_config: EmbeddingConfig::new(num_pieces, d_model),
        num_placement_embedding_residuals,
        blocks_config: TetrisGameTransformerBlockConfig {
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 0.5,
                normalized_shape: d_model,
            },
            attn_config: CausalSelfAttentionConfig {
                d_model,
                n_heads,
                rope_theta: 10000.0,
                max_position_embeddings: BOARD_SIZE,
            },
            mlp_config: MlpConfig {
                hidden_size: d_model,
                intermediate_size: d_model * 2,
            },
        },
        num_blocks: num_transformer_blocks,
        dyn_tan_config: DynamicTanhConfig {
            alpha_init_value: 0.5,
            normalized_shape: d_model,
        },
        output_layer_config: LinearConfig::new(d_model, num_placements),
    };
    let mut tetris_player_model: TetrisPlayerTransformer<B> =
        tetris_player_transformer_config.init(&device);
    let mut tetris_player_optimizer: OptimizerAdaptor<_, TetrisPlayerTransformer<B>, _> =
        tetris_model_optimizer.init();
    println!(
        "Tetris Player Model Number of parameters: {}",
        tetris_player_model.num_params()
    );

    // Define train/valid datasets and dataloaders
    let train_dataset_config = TetrisDatasetConfig::Uniform(TetrisDatasetUniformConfig {
        seed: 42,
        num_pieces_range: CopyRange { start: 0, end: 16 },
        length: tetris_model_config.items_per_epoch,
    });
    let train_dataset = TetrisInitDistDataset::train(train_dataset_config);
    let batcher_train = TetrisInitDistBatcher::<B>::default();
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(tetris_model_config.batch_size)
        .num_workers(tetris_model_config.num_workers)
        .build(train_dataset)
        .to_device(&device);

    // Dataset for single batch world model
    let train_dataset_config =
        TetrisSequenceDatasetConfig::Uniform(TetrisSequenceDatasetUniformConfig {
            seed: 123,
            num_pieces_range: CopyRange { start: 0, end: 20 },
            length: 10_000,
            sequence_length: 8,
        });
    let game_model_train_dataset = TetrisSequenceDataset::train(train_dataset_config.clone());
    let game_model_valid_dataset = TetrisSequenceDataset::validation(train_dataset_config);
    let game_model_batcher_train = TetrisSequenceDistBatcher::<B>::default();
    let game_model_batcher_valid = TetrisSequenceDistBatcher::<B::InnerBackend>::default();
    let dataloader_game_model_train = DataLoaderBuilder::new(game_model_batcher_train)
        .batch_size(80)
        .num_workers(tetris_model_config.num_workers)
        .build(game_model_train_dataset)
        .to_device(&device);
    let dataloader_game_model_valid = DataLoaderBuilder::new(game_model_batcher_valid)
        .batch_size(8)
        .num_workers(tetris_model_config.num_workers)
        .build(game_model_valid_dataset)
        .to_device(&device);

    // pre train 500 iterations for the game model
    let learner = LearnerBuilder::new(artifact_dir.clone())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![
            device.clone(),
            device.clone(),
            device.clone(),
            device.clone(),
        ])
        .num_epochs(64)
        .grads_accumulation(8)
        .summary()
        .build(tetris_game_model, tetris_game_optimizer.clone(), 1e-3);
    let mut tetris_game_model = learner.fit(
        dataloader_game_model_train.clone(),
        dataloader_game_model_valid,
    );

    // let num_pre_train_accumulation = 8;
    // let mut pre_train_grad_accumulator = GradientsAccumulator::new();
    // let mut accumulation_current = 0;
    // let num_pre_train_iterations = 256 + 1;
    let also_dataloader_game_model_train = dataloader_game_model_train.clone();
    let mut dataloader_game_model_train_iter = also_dataloader_game_model_train.iter();
    // for pre_train_iteration in 0..num_pre_train_iterations {
    //     let batch = dataloader_game_model_train_iter.next().unwrap();
    //     let output = tetris_game_model.forward_sequence(batch);
    //     let loss = output.loss.clone().into_scalar();
    //     let grads = output.loss.backward();
    //     let grads = GradientsParams::from_grads(grads, &tetris_game_model);
    //     pre_train_grad_accumulator.accumulate(&tetris_game_model, grads);
    //     accumulation_current += 1;
    //     if num_pre_train_accumulation <= accumulation_current {
    //         let grads = pre_train_grad_accumulator.grads();
    //         tetris_game_model = tetris_game_optimizer.step(1e-3, tetris_game_model, grads);
    //         accumulation_current = 0;
    //     }
    //     println!("{}: Game Model Loss: {}", pre_train_iteration, loss);
    // }

    for epoch in 1..tetris_model_config.num_epochs + 1 {
        let mut game_model_accumulator = GradientsAccumulator::new();

        let mut player_model_accumulator = GradientsAccumulator::new();
        let mut player_model_accumulation_current = 0;
        let player_model_accumulation = 4;

        for (iteration, batch) in dataloader_train.iter().enumerate() {
            {
                for _ in 0..16 {
                    let batch = dataloader_game_model_train_iter.next().unwrap();
                    let output = tetris_game_model.forward_sequence(batch);
                    let loss = output.loss.clone().into_scalar();
                    let grads = output.loss.backward();
                    let grads = GradientsParams::from_grads(grads, &tetris_game_model);
                    game_model_accumulator.accumulate(&tetris_game_model, grads);
                    println!("Game Model Loss: {}", loss);
                }
            }

            let num_moves = 16;

            let mut game_set = batch.game_set.clone();
            let mut current_boards = batch.current_boards.clone().to_device(&device);
            let mut boards_trajectory = vec![];
            let mut target_boards_trajectory = vec![];
            let mut placement_dist_trajectory = vec![];

            // Single unrolling that captures data for both models
            let mut unroll_game_model_grad_accumulator = GradientsAccumulator::new();
            for move_idx in 0..num_moves {
                // Do inference on the player model
                let (placement_dist_output, model_placements) =
                    tetris_player_model.forward_get_placement(&game_set, current_boards.clone());
                placement_dist_trajectory.push(placement_dist_output.clone());

                // Do inference on the game model
                let next_boards_dist_output = tetris_game_model
                    .soft_forward(current_boards.clone(), placement_dist_output.clone());
                current_boards = next_boards_dist_output.clone();
                boards_trajectory.push(next_boards_dist_output);

                // Get correct boards for target
                let model_placements = game_set
                    .current_placements()
                    .into_iter()
                    .zip(model_placements.into_iter())
                    .map(|(game_placements, model_placement)| {
                        game_placements
                            .into_iter()
                            .find(|p| *p == &model_placement)
                            .map(|p| *p)
                            .expect("Model placement should be valid")
                    })
                    .collect::<Vec<TetrisPiecePlacement>>();
                game_set.apply_placement(&model_placements);

                // Get correct boards for target
                let correct_boards = game_set
                    .boards()
                    .to_vec()
                    .into_iter()
                    .map(|b| b.to_binary_slice())
                    .map(|b| Tensor::<B, 1>::from_floats(b, &device))
                    .map(|b| b.reshape([1, TetrisBoardRaw::SIZE]))
                    .reduce(|a, b| Tensor::cat(vec![a, b], 0))
                    .unwrap();
                target_boards_trajectory.push(correct_boards);

                // STEP 1: Train game model (detach player model gradients)
                let game_loss_value = {
                    let loss_fn = CrossEntropyLossConfig::new().init(&device);
                    let agg_boards_trajectory: Tensor<B, 3> = Tensor::cat(
                        boards_trajectory
                            .iter()
                            .map(|t| t.clone().detach())
                            .collect(),
                        0,
                    );
                    let [batch_size, seq_len, states] = agg_boards_trajectory.dims();
                    let agg_boards_trajectory =
                        agg_boards_trajectory.reshape([batch_size * seq_len, states]);

                    let agg_target_boards_trajectory: Tensor<B, 2> =
                        Tensor::cat(target_boards_trajectory.clone(), 0);
                    let [batch_size, seq_len] = agg_target_boards_trajectory.dims();
                    let agg_target_boards_trajectory = agg_target_boards_trajectory
                        .int()
                        .reshape([batch_size * seq_len]);

                    let game_loss =
                        loss_fn.forward(agg_boards_trajectory, agg_target_boards_trajectory);
                    let loss_value = game_loss.clone().into_scalar();
                    let grads = game_loss.backward();
                    let grads = GradientsParams::from_grads(grads, &tetris_game_model);
                    unroll_game_model_grad_accumulator.accumulate(&tetris_game_model, grads);
                    loss_value
                };
                println!(
                    "Unrolling {} Game Model Loss: {}",
                    move_idx, game_loss_value
                );
            }
            let grads = unroll_game_model_grad_accumulator.grads();
            tetris_game_model = tetris_game_optimizer.step(1e-3, tetris_game_model, grads);

            // STEP 2: Train player model (detach game model gradients)
            let player_loss_value = {
                let final_boards = current_boards.clone(); // This has gradients from player model
                let [batch_size, seq_len, states] = final_boards.dims();
                let final_boards_flat = final_boards.reshape([batch_size * seq_len, states]);
                let target_boards = Tensor::<B, 1>::zeros([batch_size * seq_len], &device).int();

                let loss_fn = CrossEntropyLossConfig::new().init(&device);
                let player_loss = loss_fn.forward(final_boards_flat, target_boards);
                let loss_value = player_loss.clone().into_scalar();
                let grads = player_loss.backward();
                let grads = GradientsParams::from_grads(grads, &tetris_player_model);

                player_model_accumulator.accumulate(&tetris_player_model, grads);
                player_model_accumulation_current += 1;
                if player_model_accumulation <= player_model_accumulation_current {
                    let grads = player_model_accumulator.grads();
                    tetris_player_model =
                        tetris_player_optimizer.step(1e-3, tetris_player_model, grads);
                    player_model_accumulation_current = 0;
                }

                loss_value
            };

            // Print losses for both models
            println!(
                "Epoch {}, Iteration {}: Player Loss = {:.6}",
                epoch, iteration, player_loss_value
            );
        }
    }
}

#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use burn::backend::{
        Autodiff,
        ndarray::{NdArray, NdArrayDevice},
    };

    pub fn run() {
        let device = NdArrayDevice::Cpu;
        super::run_train::<Autodiff<NdArray>>(super::ARTIFACT_DIR, device);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };

    pub fn run() {
        let device = LibTorchDevice::Cuda(0);
        super::run_train::<Autodiff<LibTorch>>(super::ARTIFACT_DIR, device);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::backend::{
        Autodiff,
        wgpu::{Wgpu, WgpuDevice},
    };

    pub fn run() {
        let device = WgpuDevice::default();
        super::run_train::<Autodiff<Wgpu>>(super::ARTIFACT_DIR, device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };

    pub fn run() {
        let device = LibTorchDevice::Cpu;
        super::run_train::<Autodiff<LibTorch>>(super::ARTIFACT_DIR, device);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use burn::backend::{Autodiff, Cuda, cuda::CudaDevice};

    pub fn run() {
        let device = CudaDevice::default();
        super::run_train::<Autodiff<Cuda>>(super::ARTIFACT_DIR, device);
    }
}

#[cfg(feature = "candle")]
mod candle {
    use burn::backend::{Autodiff, Candle, candle::CandleDevice};

    pub fn run() {
        let device = CandleDevice::Cpu;
        super::run_train::<Autodiff<Candle>>(super::ARTIFACT_DIR, device);
    }
}

#[cfg(feature = "candle-cuda")]
mod candle_cuda {
    use burn::backend::{Autodiff, Candle, candle::CandleDevice};

    pub fn run() {
        let device = CandleDevice::cuda(1);
        super::run_train::<Autodiff<Candle>>(super::ARTIFACT_DIR, device);
    }
}

pub fn train() {
    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-blas-accelerate",
    ))]
    ndarray::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
    #[cfg(feature = "candle")]
    candle::run();
    #[cfg(feature = "candle-cuda")]
    candle_cuda::run();
}

// mod ndarray {
//     use burn::backend::{
//         Autodiff,
//         ndarray::{NdArray, NdArrayDevice},
//     };
//     pub fn run() {
//         let device = NdArrayDevice::Cpu;
//         super::run_train::<Autodiff<NdArray>>(super::ARTIFACT_DIR, device);
//     }
// }

// #[cfg(feature = "tch-gpu")]
// mod tch_gpu {
//     use burn::backend::libtorch::{LibTorch, LibTorchDevice};

//     pub fn run() {
//         #[cfg(not(target_os = "macos"))]
//         let device = LibTorchDevice::Cuda(0);
//         #[cfg(target_os = "macos")]
//         let device = LibTorchDevice::Mps;

//         super::run::<LibTorch>(device);
//     }
// }

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::default();
        super::run::<Wgpu>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    use simple_regression::training;
    pub fn run() {
        let device = LibTorchDevice::Cpu;
        super::run::<LibTorch>(device);
    }
}

#[cfg(feature = "remote")]
mod remote {
    use burn::backend::{RemoteBackend, remote::RemoteDevice};

    pub fn run() {
        let device = RemoteDevice::default();
        super::run::<RemoteBackend>(device);
    }
}

// /// Train a regression model and predict results on a number of samples.
// pub fn train() {
//     use burn::backend::{
//         Autodiff,
//         ndarray::{NdArray, NdArrayDevice},
//     };

//     use burn::backend::libtorch::{LibTorch, LibTorchDevice};
//     let device = LibTorchDevice::Cuda(0);

//     // type Backend = burn::backend::Autodiff<burn::backend::Candle<f32>>;
//     // let device = CandleDevice::cuda(0);
//     run_train::<Autodiff<LibTorch>>(ARTIFACT_DIR, device);
// }
