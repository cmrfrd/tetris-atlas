use std::{collections::HashMap, range::Range};

use candle_core::{D, DType, Device, IndexOp, Tensor, Var};
use candle_nn::{
    AdamW, Conv2dConfig, Optimizer, ParamsAdamW, VarBuilder, VarMap,
    loss::{cross_entropy, nll},
};
use rand::{Rng, distr::Uniform};

use crate::{
    data::{
        TetrisBoardsDistTensor, TetrisDatasetGenerator, TetrisPiecePlacementDistTensor,
        TetrisPieceTensor,
    },
    grad_accum::GradientAccumulator,
    model::{
        CausalSelfAttentionConfig, DynamicTanhConfig, MlpConfig, TetrisGameTransformer,
        TetrisGameTransformerBlockConfig, TetrisGameTransformerConfig, TetrisPlayerTransformer,
        TetrisPlayerTransformerConfig, TetrisWorldModel, TetrisWorldModelBlockConfig,
        TetrisWorldModelConfig, TetrisWorldModelTokenizer, TetrisWorldModelTokenizerConfig,
        TetrisWorldModelTokenizerTransformerBlockConfig,
    },
    tensorboard::summary_writer::SummaryWriter,
    tetris::{
        NUM_TETRIS_CELL_STATES, TetrisBoardRaw, TetrisPiece, TetrisPieceOrientation,
        TetrisPiecePlacement,
    },
};

pub fn train_game_transformer(tensorboard_logdir: String) {
    println!("Training world model");
    const NUM_ITERATIONS: usize = 10_000;
    const BATCH_SIZE: usize = 256;

    const ACCUMULATE_GRADIENTS_STEPS: usize = 8;

    let mut summary_writer = SummaryWriter::new(tensorboard_logdir);

    let device = Device::new_cuda(0).unwrap();
    let model_varmap = VarMap::new();
    let model_vs = VarBuilder::from_varmap(&model_varmap, DType::F32, &device);

    let model_dim = 32;

    let game_transformer_cfg = TetrisGameTransformerConfig {
        board_embedding_config: (NUM_TETRIS_CELL_STATES, model_dim),
        placement_embedding_config: (TetrisPiecePlacement::NUM_PLACEMENTS, model_dim),
        num_blocks: 16,
        num_placement_embedding_residuals: 8,
        blocks_config: TetrisGameTransformerBlockConfig {
            attn_config: CausalSelfAttentionConfig {
                d_model: model_dim,
                n_attention_heads: 16,
                n_kv_heads: 16,
                rope_theta: 10000.0,
                max_position_embeddings: TetrisBoardRaw::SIZE,
            },
            mlp_config: MlpConfig {
                hidden_size: model_dim,
                intermediate_size: 2 * model_dim,
                output_size: model_dim,
            },
            dyn_tanh_config: DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: model_dim,
            },
        },
        dyn_tanh_config: DynamicTanhConfig {
            alpha_init_value: 1.0,
            normalized_shape: model_dim,
        },
        output_layer_config: (model_dim, NUM_TETRIS_CELL_STATES),
    };

    let game_transformer = TetrisGameTransformer::init(&model_vs, &game_transformer_cfg).unwrap();
    let mut game_transformer_optimizer =
        AdamW::new(model_varmap.all_vars(), ParamsAdamW::default()).unwrap();
    let game_transformer_params = model_varmap.all_vars();
    let mut game_transformer_grad_accumulator =
        GradientAccumulator::new(ACCUMULATE_GRADIENTS_STEPS);
    println!("Game transformer optimizer and grad accumulator initialized");

    let data_generator = TetrisDatasetGenerator::new();
    let mut rng = rand::rng();

    for i in 0..NUM_ITERATIONS {
        let datum = data_generator
            .gen_uniform_sampled_transition((10..11).into(), BATCH_SIZE, &device, &mut rng)
            .unwrap();

        let current_board = datum.current_board;
        let placement = datum.placement;

        let output = game_transformer
            .forward(&current_board, &placement)
            .unwrap();
        let target = datum.result_board;

        let output_flat = output
            .reshape(&[BATCH_SIZE * TetrisBoardRaw::SIZE, NUM_TETRIS_CELL_STATES])
            .unwrap();
        let target_flat = target
            .reshape(&[BATCH_SIZE * TetrisBoardRaw::SIZE])
            .unwrap();

        let loss = cross_entropy(&output_flat, &target_flat).unwrap();
        let loss_value = loss.to_scalar::<f32>().unwrap();

        let board_logits_argmax = output_flat
            .argmax(D::Minus1)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap();
        let correct_mask = board_logits_argmax
            .eq(&target_flat)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let correct_sum = correct_mask.sum_all().unwrap().to_scalar::<f32>().unwrap();
        let total_preds = (BATCH_SIZE * TetrisBoardRaw::SIZE) as f32;
        let board_logits_accuracy = correct_sum / total_preds;

        let grad_norm = game_transformer_grad_accumulator.gradient_norm().unwrap();

        summary_writer.add_scalar("loss", loss_value, i);
        summary_writer.add_scalar("board_logits_accuracy", board_logits_accuracy, i);
        summary_writer.add_scalar("grad_norm", grad_norm, i);

        println!(
            "Iteration {:>6} | loss {:>10.4} | batch_accuracy {:>7.4} | grad_norm {:>10.4}",
            i, loss_value, board_logits_accuracy, grad_norm
        );

        let grads = loss.backward().unwrap();
        game_transformer_grad_accumulator
            .accumulate(grads, &game_transformer_params)
            .unwrap();

        let should_step = game_transformer_grad_accumulator
            .apply_and_reset(&mut game_transformer_optimizer, &game_transformer_params)
            .unwrap();

        if should_step {
            println!("Stepping game transformer");
        }
    }
}

pub fn train() {
    println!("Training world model");
    const NUM_ITERATIONS: usize = 10_000;
    const BATCH_SIZE: usize = 64;

    let mut sequence_length = 12;
    const ACCUMULATE_GRADIENTS_STEPS: usize = 4;

    let device = Device::new_cuda(0).unwrap();
    let model_varmap = VarMap::new();
    let model_vs = VarBuilder::from_varmap(&model_varmap, DType::F32, &device);

    let world_model_dim = 32;

    let tokenizer_cfg = TetrisWorldModelTokenizerConfig {
        board_embedding_dim: world_model_dim,
        piece_embedding_dim: world_model_dim,
        orientation_embedding_dim: world_model_dim,
        blocks_config: TetrisWorldModelTokenizerTransformerBlockConfig {
            attn_config: CausalSelfAttentionConfig {
                d_model: world_model_dim,
                n_attention_heads: 8,
                n_kv_heads: 8,
                rope_theta: 10000.0,
                max_position_embeddings: TetrisBoardRaw::SIZE,
            },
            dyn_tanh_config: DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: world_model_dim,
            },
            mlp_config: MlpConfig {
                hidden_size: world_model_dim,
                intermediate_size: 3 * world_model_dim,
                output_size: world_model_dim,
            },
        },
        num_blocks: 8,
        num_residuals: 4,
        dyn_tanh_config: DynamicTanhConfig {
            alpha_init_value: 1.0,
            normalized_shape: world_model_dim,
        },
        token_encoder_config: MlpConfig {
            hidden_size: TetrisBoardRaw::SIZE,
            intermediate_size: 3 * world_model_dim,
            output_size: 1,
        },
    };

    let num_blocks = 16;
    let model_cfg = TetrisWorldModelConfig {
        blocks_config: TetrisWorldModelBlockConfig {
            dyn_tanh_config: DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: world_model_dim,
            },
            attn_config: CausalSelfAttentionConfig {
                d_model: world_model_dim,
                n_attention_heads: 16,
                n_kv_heads: 16,
                rope_theta: 10000.0,
                max_position_embeddings: 128,
            },
            mlp_config: MlpConfig {
                hidden_size: world_model_dim,
                intermediate_size: 3 * world_model_dim,
                output_size: world_model_dim,
            },
        },
        num_blocks,
        orientation_head_config: MlpConfig {
            hidden_size: world_model_dim,
            intermediate_size: 2 * world_model_dim,
            output_size: TetrisPieceOrientation::NUM_ORIENTATIONS,
        },
        dyn_tanh_config: DynamicTanhConfig {
            alpha_init_value: 1.0,
            normalized_shape: world_model_dim,
        },
    };

    let model = TetrisWorldModel::init(&model_vs, &model_cfg).unwrap();
    let tokenizer = TetrisWorldModelTokenizer::init(&model_vs, &tokenizer_cfg).unwrap();
    println!("Model and tokenizer initialized");

    let mut model_optimizer = AdamW::new(model_varmap.all_vars(), ParamsAdamW::default()).unwrap();
    let model_params = model_varmap.all_vars();
    let mut model_grad_accumulator = GradientAccumulator::new(ACCUMULATE_GRADIENTS_STEPS);
    println!("Model optimizer and grad accumulator initialized");

    let data_generator = TetrisDatasetGenerator::new();
    let mut rng = rand::rng();

    for i in 0..NUM_ITERATIONS {
        let datum_sequence = data_generator
            .gen_sequence(
                (0..4).into(),
                BATCH_SIZE,
                sequence_length,
                &device,
                &mut rng,
            )
            .unwrap();

        let goal_board =
            TetrisBoardsDistTensor::try_from(datum_sequence.current_boards.last().unwrap().clone())
                .unwrap();
        let board_sequence = datum_sequence
            .current_boards
            .iter()
            .map(|b| TetrisBoardsDistTensor::try_from(b.clone()).unwrap())
            .collect::<Vec<TetrisBoardsDistTensor>>();
        let piece_sequence = datum_sequence
            .pieces
            .iter()
            .map(|p| TetrisPieceTensor::from_pieces(p, &device).unwrap())
            .collect::<Vec<TetrisPieceTensor>>();
        let orientation_sequence = datum_sequence.orientations;

        // [B, 2+S, D]
        let context_tokens = tokenizer
            .forward_context(
                &goal_board,
                &board_sequence,
                &piece_sequence,
                &orientation_sequence,
            )
            .unwrap();

        // [B, S]
        let current_pieces_tensor = {
            // [..] -> [B*S, 1] -> [B, S]
            let pieces_seq = datum_sequence
                .pieces
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            let pieces_seq =
                TetrisPieceTensor::from_pieces(&pieces_seq.as_slice(), &device).unwrap();
            pieces_seq.reshape(&[BATCH_SIZE, sequence_length]).unwrap()
        };

        let (world_model_board_output_logits, world_model_orientation_output_logits) = model
            .forward_all(&context_tokens, &current_pieces_tensor, false)
            .unwrap();

        // Compute sequence board loss over tokens after [goal, current]
        // world_model_board_output_logits: [B, 2+S, T, NUM_STATES]
        let board_logits_seq = world_model_board_output_logits
            .i((.., 2.., .., ..))
            .unwrap(); // [B, S, T, NUM_STATES]
        assert_eq!(
            board_logits_seq.shape().dims(),
            &[
                BATCH_SIZE,
                sequence_length,
                TetrisBoardRaw::SIZE,
                NUM_TETRIS_CELL_STATES
            ]
        );

        // Targets per step: next boards (result_boards[t]) → [B, S, T]
        let target_seq = {
            let mut per_step: Vec<Tensor> = Vec::with_capacity(sequence_length);
            for t in 0..sequence_length {
                let step_boards = datum_sequence.result_boards[t].clone(); // [B, T]
                let step_labels = step_boards.unsqueeze(1).unwrap(); // [B, 1, T]
                per_step.push(step_labels);
            }
            let refs: Vec<&Tensor> = per_step.iter().collect();
            Tensor::cat(&refs, 1).unwrap() // [B, S, T]
        };
        assert_eq!(
            target_seq.shape().dims(),
            &[BATCH_SIZE, sequence_length, TetrisBoardRaw::SIZE]
        );

        // Flatten to [B*S*T, NUM_STATES] vs [B*S*T]
        let board_logits_flat = board_logits_seq
            .reshape(&[
                (BATCH_SIZE * sequence_length * TetrisBoardRaw::SIZE) as usize,
                NUM_TETRIS_CELL_STATES as usize,
            ])
            .unwrap();
        let board_targets_flat = target_seq
            .reshape(&[(BATCH_SIZE * sequence_length * TetrisBoardRaw::SIZE) as usize])
            .unwrap();

        // Get the placement logits
        let orientation_logits_seq = world_model_orientation_output_logits
            .i((.., 2.., ..))
            .unwrap()
            .contiguous()
            .unwrap();

        let orientations_logits_flat = orientation_logits_seq
            .reshape(&[
                (BATCH_SIZE * sequence_length) as usize,
                TetrisPieceOrientation::NUM_ORIENTATIONS,
            ])
            .unwrap();
        // [B, S] -> [B*S]
        let orientations_targets_flat = Tensor::cat(
            &orientation_sequence
                .into_iter()
                .map(Into::<Tensor>::into)
                .collect::<Vec<Tensor>>(),
            1,
        )
        .unwrap()
        .reshape(&[(BATCH_SIZE * sequence_length) as usize])
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap();

        // Cross-entropy over all sequence steps and cells
        let board_loss = cross_entropy(&board_logits_flat, &board_targets_flat).unwrap();
        let board_loss_value = board_loss.to_scalar::<f32>().unwrap();

        let orientation_loss =
            cross_entropy(&orientations_logits_flat, &orientations_targets_flat).unwrap();
        let orientation_loss_value = orientation_loss
            .to_device(&Device::Cpu)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        let grad_norm = model_grad_accumulator.gradient_norm().unwrap();

        // Backprop + gradient accumulation (train both heads)
        // let loss = (board_loss + orientation_loss).unwrap();
        let loss = board_loss;
        let loss_value = loss.to_scalar::<f32>().unwrap();

        // calculate board logits accuracy (mean over all predictions)
        let board_logits_argmax = board_logits_flat
            .argmax(D::Minus1)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap();
        let correct_mask = board_logits_argmax
            .eq(&board_targets_flat)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let correct_sum = correct_mask.sum_all().unwrap().to_scalar::<f32>().unwrap();
        let total_preds = (BATCH_SIZE * sequence_length * TetrisBoardRaw::SIZE) as f32;
        let board_logits_accuracy = correct_sum / total_preds;

        // log
        println!(
            "Iteration {:>6} | board_loss {:>10.4} | board_acc {:>7.4} | orientation_loss {:>10.4} | grad_norm {:>10.4} | loss {:>10.4}",
            i,
            board_loss_value,
            board_logits_accuracy,
            orientation_loss_value,
            grad_norm,
            loss_value,
        );

        let grads = loss.backward().unwrap();
        model_grad_accumulator
            .accumulate(grads, &model_params)
            .unwrap();

        // Apply optimizer step if accumulation is complete
        let should_step = model_grad_accumulator
            .apply_and_reset(&mut model_optimizer, &model_params)
            .unwrap();

        if should_step {
            println!("Stepping model");
        }
    }

    // ///////////////////////////////////
    // ///
    // /// RL training loop
    // ///////////////////////////////////
    // for i in 0..NUM_ITERATIONS {
    //     let datum_sequence = data_generator
    //         .gen_sequence(
    //             (0..4).into(),
    //             BATCH_SIZE,
    //             sequence_length,
    //             &device,
    //             &mut rng,
    //         )
    //         .unwrap();

    //     let goal_board =
    //         TetrisBoardsDistTensor::try_from(datum_sequence.current_boards.last().unwrap().clone())
    //             .unwrap();
    //     let board_sequence = datum_sequence
    //         .current_boards
    //         .iter()
    //         .map(|b| TetrisBoardsDistTensor::try_from(b.clone()).unwrap())
    //         .collect::<Vec<TetrisBoardsDistTensor>>();
    //     let piece_sequence = datum_sequence
    //         .pieces
    //         .iter()
    //         .map(|p| TetrisPieceTensor::from_pieces(p, &device).unwrap())
    //         .collect::<Vec<TetrisPieceTensor>>();
    //     let placement_sequence = datum_sequence.placements;
    //     let orientation_sequence = datum_sequence.orientations;

    //     // [B, 2+S, D]
    //     let context_tokens = tokenizer
    //         .forward_context(
    //             &goal_board,
    //             &board_sequence,
    //             &piece_sequence,
    //             &placement_sequence,
    //         )
    //         .unwrap();

    //     // [B, S]
    //     let current_pieces_tensor = {
    //         // [..] -> [B*S, 1] -> [B, S]
    //         let pieces_seq = datum_sequence
    //             .pieces
    //             .iter()
    //             .flatten()
    //             .copied()
    //             .collect::<Vec<_>>();
    //         let pieces_seq =
    //             TetrisPieceTensor::from_pieces(&pieces_seq.as_slice(), &device).unwrap();
    //         pieces_seq.reshape(&[BATCH_SIZE, sequence_length]).unwrap()
    //     };

    //     let (world_model_board_output_logits, world_model_orientation_output_logits) = model
    //         .forward_all(&context_tokens, &current_pieces_tensor, false)
    //         .unwrap();

    //     // Compute sequence board loss over tokens after [goal, current]
    //     // world_model_board_output_logits: [B, 2+S, T, NUM_STATES]
    //     let board_logits_seq = world_model_board_output_logits
    //         .i((.., 2.., .., ..))
    //         .unwrap(); // [B, S, T, NUM_STATES]

    //     // Targets per step: next boards (result_boards[t]) → [B, S, T]
    //     let target_seq = {
    //         let mut per_step: Vec<Tensor> = Vec::with_capacity(sequence_length);
    //         for t in 0..sequence_length {
    //             let step_boards = datum_sequence.result_boards[t].clone(); // [B, T]
    //             let step_labels = step_boards.unsqueeze(1).unwrap(); // [B, 1, T]
    //             per_step.push(step_labels);
    //         }
    //         let refs: Vec<&Tensor> = per_step.iter().collect();
    //         Tensor::cat(&refs, 1).unwrap() // [B, S, T]
    //     };
    //     assert_eq!(
    //         target_seq.shape().dims(),
    //         &[BATCH_SIZE, sequence_length, TetrisBoardRaw::SIZE]
    //     );

    //     // Flatten to [B*S*T, NUM_STATES] vs [B*S*T]
    //     let board_logits_flat = board_logits_seq
    //         .reshape(&[
    //             (BATCH_SIZE * sequence_length * TetrisBoardRaw::SIZE) as usize,
    //             NUM_TETRIS_CELL_STATES as usize,
    //         ])
    //         .unwrap();
    //     let board_targets_flat = target_seq
    //         .reshape(&[(BATCH_SIZE * sequence_length * TetrisBoardRaw::SIZE) as usize])
    //         .unwrap();

    //     // Get the placement logits
    //     let orientation_logits_seq = world_model_orientation_output_logits
    //         .i((.., 2.., ..))
    //         .unwrap()
    //         .contiguous()
    //         .unwrap();

    //     let orientations_logits_flat = orientation_logits_seq
    //         .reshape(&[
    //             (BATCH_SIZE * sequence_length) as usize,
    //             TetrisPieceOrientation::NUM_ORIENTATIONS,
    //         ])
    //         .unwrap();
    //     // [B, S] -> [B*S]
    //     let orientations_targets_flat = Tensor::cat(
    //         &orientation_sequence
    //             .into_iter()
    //             .map(Into::<Tensor>::into)
    //             .collect::<Vec<Tensor>>(),
    //         1,
    //     )
    //     .unwrap()
    //     .reshape(&[(BATCH_SIZE * sequence_length) as usize])
    //     .unwrap()
    //     .to_dtype(DType::U32)
    //     .unwrap();

    //     // Cross-entropy over all sequence steps and cells
    //     let board_loss = cross_entropy(&board_logits_flat, &board_targets_flat).unwrap();
    //     let board_loss_value = board_loss.to_scalar::<f32>().unwrap();

    //     let orientation_loss =
    //         cross_entropy(&orientations_logits_flat, &orientations_targets_flat).unwrap();
    //     let orientation_loss_value = orientation_loss
    //         .to_device(&Device::Cpu)
    //         .unwrap()
    //         .to_scalar::<f32>()
    //         .unwrap();

    //     let grad_norm = model_grad_accumulator.gradient_norm().unwrap();

    //     // Backprop + gradient accumulation
    //     let loss = (board_loss + orientation_loss).unwrap();
    //     let loss_value = loss.to_scalar::<f32>().unwrap();

    //     // calculate board logits accuracy (mean over all predictions)
    //     let board_logits_argmax = board_logits_flat
    //         .argmax(D::Minus1)
    //         .unwrap()
    //         .to_dtype(DType::U32)
    //         .unwrap();
    //     let correct_mask = board_logits_argmax
    //         .eq(&board_targets_flat)
    //         .unwrap()
    //         .to_dtype(DType::F32)
    //         .unwrap();
    //     let correct_sum = correct_mask.sum_all().unwrap().to_scalar::<f32>().unwrap();
    //     let total_preds = (BATCH_SIZE * sequence_length * TetrisBoardRaw::SIZE) as f32;
    //     let board_logits_accuracy = correct_sum / total_preds;

    //     // log
    //     println!(
    //         "Iteration {:>6} | board_loss {:>10.4} | board_acc {:>7.4} | orientation_loss {:>10.4} | grad_norm {:>10.4} | loss {:>10.4}",
    //         i,
    //         board_loss_value,
    //         board_logits_accuracy,
    //         orientation_loss_value,
    //         grad_norm,
    //         loss_value,
    //     );

    //     let grads = loss.backward().unwrap();
    //     model_grad_accumulator
    //         .accumulate(grads, &model_params)
    //         .unwrap();

    //     // Apply optimizer step if accumulation is complete
    //     let should_step = model_grad_accumulator
    //         .apply_and_reset(&mut model_optimizer, &model_params)
    //         .unwrap();

    //     if should_step {
    //         println!("Stepping model");
    //     }
    // }
}
