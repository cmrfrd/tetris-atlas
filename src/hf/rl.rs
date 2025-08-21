use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap, encoding::one_hot, loss::cross_entropy,
    ops::log_softmax,
};

use crate::{
    data::TetrisDatasetGenerator,
    grad_accum::GradientAccumulator,
    hf::model::{
        CausalSelfAttentionConfig, DynamicTanhConfig, MlpConfig, TetrisGameTransformer,
        TetrisGameTransformerBlockConfig, TetrisGameTransformerConfig, TetrisPlayerTransformer,
        TetrisPlayerTransformerConfig,
    },
    tetris::{
        BOARD_SIZE, NUM_TETRIS_CELL_STATES, NUM_TETRIS_PIECES, TetrisBoardRaw,
        TetrisPieceOrientation, TetrisPiecePlacement,
    },
};

pub fn train() {
    const WORLD_MODEL_PRETRAIN_SEQUENCE_LENGTH: usize = 8;
    const WORLD_MODEL_NUM_GRADIENT_ACCUMULATE_STEPS: usize =
        4 * WORLD_MODEL_PRETRAIN_SEQUENCE_LENGTH;
    const NUM_WORLD_MODEL_ITERATIONS: usize = 2_000;
    const NUM_ITERATIONS: usize = 100_000;
    const BATCH_SIZE: usize = 256;
    // Number of planning iterations the model uses to reach the goal (empty board)
    const NUM_PLANNING_ITERATIONS: usize = 8;

    let mut sequence_length = 8;
    const ACCUMULATE_GRADIENTS_STEPS: usize = 4;

    let device = Device::new_cuda(0).unwrap();

    let preplace_piece_range = (1..8).into();
    let data_generator = TetrisDatasetGenerator::new();

    let mut rng = rand::rng();

    // Tetris model
    let d_model = 16;
    let num_blocks = 16;
    let num_placement_embedding_residuals = num_blocks / 2;

    let game_model_config = TetrisGameTransformerConfig {
        board_embedding_config: (NUM_TETRIS_CELL_STATES, d_model),
        placement_embedding_config: (TetrisPiecePlacement::NUM_PLACEMENTS, d_model),
        num_placement_embedding_residuals,
        blocks_config: TetrisGameTransformerBlockConfig {
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 0.5,
                normalized_shape: d_model,
            },
            attn_config: CausalSelfAttentionConfig {
                d_model,
                n_attention_heads: 8,
                rope_theta: 10000.0,
                max_position_embeddings: BOARD_SIZE,
            },
            mlp_config: MlpConfig {
                hidden_size: d_model,
                intermediate_size: d_model * 2,
                output_size: d_model,
            },
        },
        num_blocks,
        dyn_tan_config: DynamicTanhConfig {
            alpha_init_value: 0.5,
            normalized_shape: d_model,
        },
        output_layer_config: (d_model, NUM_TETRIS_CELL_STATES),
    };

    let game_model_varmap = VarMap::new();
    let game_model_vs = VarBuilder::from_varmap(&game_model_varmap, DType::F32, &device);

    let game_model = TetrisGameTransformer::init(&game_model_vs, &game_model_config).unwrap();
    let mut game_model_optimizer =
        AdamW::new(game_model_varmap.all_vars(), ParamsAdamW::default()).unwrap();

    let game_model_params = game_model_varmap.all_vars();
    let mut game_model_grad_accumulator =
        GradientAccumulator::new(WORLD_MODEL_NUM_GRADIENT_ACCUMULATE_STEPS);

    // Player model
    let d_model = 16;
    let num_blocks = 16;
    let player_model_config = TetrisPlayerTransformerConfig {
        board_embedding_config: (NUM_TETRIS_CELL_STATES, d_model),
        piece_embedding_config: (NUM_TETRIS_PIECES, d_model),

        blocks_config: TetrisGameTransformerBlockConfig {
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 0.5,
                normalized_shape: d_model,
            },
            attn_config: CausalSelfAttentionConfig {
                d_model,
                n_attention_heads: 8,
                rope_theta: 10000.0,
                max_position_embeddings: TetrisBoardRaw::SIZE,
            },
            mlp_config: MlpConfig {
                hidden_size: d_model,
                intermediate_size: d_model * 2,
                output_size: d_model,
            },
        },
        num_blocks,
        dyn_tan_config: DynamicTanhConfig {
            alpha_init_value: 0.5,
            normalized_shape: d_model,
        },

        condition_board_mlp_config: MlpConfig {
            hidden_size: BOARD_SIZE,
            intermediate_size: d_model,
            output_size: 1,
        },

        output_layer_config: (d_model, TetrisPieceOrientation::NUM_ORIENTATIONS),
    };

    let player_model_varmap = VarMap::new();
    let player_model_vs = VarBuilder::from_varmap(&player_model_varmap, DType::F32, &device);

    let player_model =
        TetrisPlayerTransformer::init(&player_model_vs, &player_model_config).unwrap();
    let mut player_model_optimizer =
        AdamW::new(player_model_varmap.all_vars(), ParamsAdamW::default()).unwrap();

    let player_model_params = player_model_varmap.all_vars();
    let mut player_model_grad_accumulator = GradientAccumulator::new(ACCUMULATE_GRADIENTS_STEPS);

    // Pre train the world model
    for i in 0..NUM_WORLD_MODEL_ITERATIONS {
        let datum_sequence = data_generator
            .gen_sequence(
                preplace_piece_range,
                BATCH_SIZE,
                WORLD_MODEL_PRETRAIN_SEQUENCE_LENGTH,
                &device,
                &mut rng,
            )
            .unwrap();

        let mut tetris_world_model_sum_loss = 0.0;
        let mut tetris_world_model_total_correct = 0;
        let mut tetris_world_model_total_cells_correct = 0;

        for t in 0..sequence_length {
            let datum_current_board = datum_sequence.current_boards[t].clone();
            let datum_placement = datum_sequence.placements[t].clone();
            let datum_result_board = datum_sequence.result_boards[t].clone();

            // let output_board_dist = game_model
            //     .forward(&datum_current_board, &datum_placement)
            //     .unwrap();

            // let (batch_size, seq_len, states) = output_board_dist.dims3().unwrap();
            // let outputs = output_board_dist
            //     .reshape(&[batch_size * seq_len, states])
            //     .unwrap();

            // let (batch_size, seq_len) = datum_result_board.dims2().unwrap();
            // let targets = datum_result_board.reshape(&[batch_size * seq_len]).unwrap();

            // // Calculate CE loss
            // let ce_loss = cross_entropy(&outputs, &targets).unwrap();

            // // Differentiable board-level wrongness penalty: expected wrong-board probability
            // let output_logits: Tensor = output_board_dist.clone().into(); // [B, T, S]
            // let log_probs = log_softmax(&output_logits, D::Minus1).unwrap(); // [B, T, S]
            // let targets_2d: Tensor = datum_result_board.clone().into(); // [B, T]
            // let one_hot_targets = one_hot(targets_2d, NUM_TETRIS_CELL_STATES, 1f32, 0f32)
            //     .unwrap()
            //     .to_dtype(DType::F32)
            //     .unwrap(); // [B, T, S]
            // let per_cell_logp = (log_probs * one_hot_targets)
            //     .unwrap()
            //     .sum(D::Minus1)
            //     .unwrap(); // [B, T]
            // let per_board_logp = per_cell_logp.sum(D::Minus1).unwrap(); // [B]
            // let board_correct_prob = per_board_logp.exp().unwrap(); // [B]
            // let wrong_board_prob = (Tensor::ones(
            //     board_correct_prob.shape(),
            //     DType::F32,
            //     board_correct_prob.device(),
            // )
            // .unwrap()
            //     - &board_correct_prob)
            //     .unwrap(); // [B]
            // // Concave shaping near zero so small errors yield large gains: sqrt(wrong_prob + eps)
            // let eps = Tensor::full(1e-6f32, &[], wrong_board_prob.device()).unwrap();
            // let shaped = (&wrong_board_prob + &eps).unwrap().sqrt().unwrap(); // [B]
            // let wrong_board_sum = shaped.sum_all().unwrap(); // scalar
            // let batch_f32 = Tensor::full(batch_size as f32, &[], wrong_board_sum.device()).unwrap();
            // let wrong_board_fraction = (wrong_board_sum / batch_f32).unwrap(); // scalar
            // let weight = Tensor::full(10.0f32, &[], wrong_board_fraction.device()).unwrap();
            // let wrong_board_loss = (wrong_board_fraction * weight).unwrap();

            // // Combined loss
            // let loss = (ce_loss + wrong_board_loss).unwrap();
            // let loss_value = loss.to_scalar::<f32>().unwrap();
            // tetris_world_model_sum_loss += loss_value;

            // // Count number of fully correct boards (all cells correct) for metrics
            // let output_argmax = output_board_dist.argmax().unwrap();
            // let num_all_correct = output_argmax.num_boards_equal(&datum_result_board).unwrap();
            // tetris_world_model_total_correct += num_all_correct.to_scalar::<u32>().unwrap();

            // // Count number of cells correct for metrics
            // let num_cells_correct = output_argmax.num_cells_equal(&datum_result_board).unwrap();
            // tetris_world_model_total_cells_correct += num_cells_correct.to_scalar::<u32>().unwrap();

            // // Compute gradients and accumulate them
            // let grads = loss.backward().unwrap();
            // game_model_grad_accumulator
            //     .accumulate(grads, &game_model_params)
            //     .unwrap();
        }

        let world_model_avg_loss =
            tetris_world_model_sum_loss / (WORLD_MODEL_PRETRAIN_SEQUENCE_LENGTH as f32);
        let world_model_avg_board_accuracy = (tetris_world_model_total_correct as f32
            / (WORLD_MODEL_PRETRAIN_SEQUENCE_LENGTH * BATCH_SIZE) as f32)
            * 100.0;
        let world_model_avg_cells_correct = (tetris_world_model_total_cells_correct as f32
            / (WORLD_MODEL_PRETRAIN_SEQUENCE_LENGTH * BATCH_SIZE * TetrisBoardRaw::SIZE) as f32)
            * 100.0;

        let grad_norm_before_step = game_model_grad_accumulator.gradient_norm().unwrap();
        let should_step = game_model_grad_accumulator
            .apply_and_reset(&mut game_model_optimizer, &game_model_params)
            .unwrap();

        if should_step {
            println!(
                "Pre-train iteration {}: world_model_avg_loss = {:.4}, world_model_avg_board_accuracy = {:.2}%, world_model_avg_cell_accuracy = {:.2}%, world_model_grad_norm = {:.4}",
                i,
                world_model_avg_loss,
                world_model_avg_board_accuracy,
                world_model_avg_cells_correct,
                grad_norm_before_step,
            );
        }
    }

    // RL training loop
    for i in 0..NUM_ITERATIONS {
        let datum_sequence = data_generator
            .gen_sequence(
                preplace_piece_range,
                BATCH_SIZE,
                sequence_length,
                &device,
                &mut rng,
            )
            .unwrap();

        for t in 0..sequence_length {
            let datum_current_board = datum_sequence.current_boards[t].clone();
            let datum_placement = datum_sequence.placements[t].clone();
            let datum_result_board = datum_sequence.result_boards[t].clone();
            let datum_pieces = datum_sequence.pieces[t].clone();
            let datum_orientation = datum_sequence.orientations[t].clone();
        }

        // let mut player_model_sequence_sum_loss = 0.0;
        // let mut player_model_sequence_sum_accuracy = 0.0;
        // let mut player_model_total_samples = 0usize;
        // {
        //     let datum_current_board = datum_transition.current_board.clone();
        //     let _datum_placement = datum_transition.placement.clone();
        //     let datum_pieces = datum_transition.piece.clone();
        //     let datum_orientation = datum_transition.orientation.clone();

        //     //
        //     // Perform grad update for player model
        //     //
        //     // Construct current board distribution tensor once per step
        //     let current_dist_board = TetrisBoardsDistTensor::try_from(datum_current_board).unwrap();

        //     // reshape targets once per step
        //     let (batch_size, orientations_len) = datum_orientation.dims2().unwrap();
        //     let targets = datum_orientation
        //         .reshape(&[batch_size * orientations_len])
        //         .unwrap();

        //     // Condition on the empty board goal for the whole batch
        //     let condition_board = &empty_boards_dist;

        //     // Use logits with cross-entropy for stable training gradients
        //     let logits = player_model
        //         .soft_forward_masked(
        //             &datum_pieces,
        //             &current_dist_board,
        //             condition_board,
        //             NUM_PLANNING_ITERATIONS,
        //         )
        //         .unwrap();

        //     // Calculate loss on logits
        //     let loss = cross_entropy(&logits, &targets).unwrap();
        //     let loss_value = loss.to_scalar::<f32>().unwrap();
        //     player_model_sequence_sum_loss += loss_value;

        //     // Calculate accuracy
        //     let outputs_argmax = logits
        //         .argmax(D::Minus1)
        //         .unwrap()
        //         .to_dtype(DType::U8)
        //         .unwrap();
        //     let correct_predictions = outputs_argmax
        //         .eq(&targets)
        //         .unwrap()
        //         .to_dtype(DType::U32)
        //         .unwrap()
        //         .sum_all()
        //         .unwrap();
        //     let total_predictions = (batch_size * orientations_len) as f32;
        //     let accuracy_percentage = (correct_predictions.to_scalar::<u32>().unwrap() as f32
        //         / total_predictions)
        //         * 100.0;
        //     player_model_sequence_sum_accuracy += accuracy_percentage;

        //     // Compute gradients and accumulate them
        //     let grads = loss.backward().unwrap();
        //     player_model_grad_accumulator
        //         .accumulate(grads, &player_model_params)
        //         .unwrap();

        //     player_model_total_samples += 1;
        // }

        // // Compute averages for curriculum and (conditional) logging
        // let player_model_sequence_avg_loss = if player_model_total_samples > 0 {
        //     player_model_sequence_sum_loss / (player_model_total_samples as f32)
        // } else {
        //     0.0
        // };
        // let player_model_sequence_avg_accuracy = if player_model_total_samples > 0 {
        //     player_model_sequence_sum_accuracy / (player_model_total_samples as f32)
        // } else {
        //     0.0
        // };

        // if player_model_sequence_avg_accuracy > 90. {
        //     sequence_length += 1;
        //     println!("Increasing sequence length to {}", sequence_length);
        // }

        // // Capture gradient norm before applying/resetting, otherwise it would be zero after reset
        // let grad_norm_before_step = player_model_grad_accumulator.gradient_norm().unwrap();

        // let should_step = player_model_grad_accumulator
        //     .apply_and_reset(&mut player_model_optimizer, &player_model_params)
        //     .unwrap();

        // if should_step {
        //     println!(
        //         "Iteration {}: player_model_avg_seq_loss = {:.4}, player_model_avg_seq_accuracy = {:.2}%, player_model_grad_norm = {:.4}",
        //         i,
        //         player_model_sequence_avg_loss,
        //         player_model_sequence_avg_accuracy,
        //         grad_norm_before_step,
        //     );
        //     println!("Stepping player model");
        // }
    }
}
