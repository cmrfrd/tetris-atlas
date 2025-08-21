use std::{collections::HashMap, range::Range};

use candle_core::{D, DType, Device, Tensor, Var};
use candle_nn::{
    AdamW, Conv2dConfig, Optimizer, ParamsAdamW, VarBuilder, VarMap,
    loss::{cross_entropy, nll},
};
use rand::{Rng, distr::Uniform};

use crate::{
    data::{TetrisBoardsDistTensor, TetrisDatasetGenerator, TetrisPieceTensor},
    grad_accum::GradientAccumulator,
    hf::model::{
        CausalSelfAttentionConfig, Conv2dInit, DynamicTanhConfig, MlpConfig,
        TetrisBoardEncoderConfig, TetrisGameTransformer, TetrisGameTransformerBlockConfig,
        TetrisGameTransformerConfig, TetrisPlayerTransformer, TetrisPlayerTransformerConfig,
        TetrisWorldModel, TetrisWorldModelBlockConfig, TetrisWorldModelConfig,
        TetrisWorldModelTokenizer, TetrisWorldModelTokenizerConfig,
    },
    tetris::{
        BOARD_SIZE, NUM_TETRIS_CELL_STATES, NUM_TETRIS_PIECES, TetrisBoardRaw,
        TetrisPieceOrientation, TetrisPiecePlacement,
    },
};

pub fn train() {
    const NUM_ITERATIONS: usize = 100_000;
    const BATCH_SIZE: usize = 256;

    let mut sequence_length = 4;
    const ACCUMULATE_GRADIENTS_STEPS: usize = 4;

    let device = Device::new_cuda(0).unwrap();
    let model_varmap = VarMap::new();
    let model_vs = VarBuilder::from_varmap(&model_varmap, DType::F32, &device);

    let d_model = 16;
    let piece_embedding_dim = 4;
    let placement_embedding_dim = 4;

    let conv_cfg = Conv2dConfig {
        padding: 1,
        stride: 1,
        dilation: 1,
        groups: 1,
        cudnn_fwd_algo: None,
    };
    let tokenizer_cfg = TetrisWorldModelTokenizerConfig {
        board_encoder_config: TetrisBoardEncoderConfig {
            conv_layers: vec![
                Conv2dInit {
                    in_channels: NUM_TETRIS_CELL_STATES,
                    out_channels: 16,
                    kernel_size: 3,
                    config: conv_cfg,
                },
                Conv2dInit {
                    in_channels: 16,
                    out_channels: 8,
                    kernel_size: 3,
                    config: conv_cfg,
                },
                Conv2dInit {
                    in_channels: 8,
                    out_channels: 4,
                    kernel_size: 3,
                    config: conv_cfg,
                },
            ],
            mlp_config: MlpConfig {
                hidden_size: 12,
                intermediate_size: 12,
                output_size: 16,
            },
        },
        piece_embedding_dim: piece_embedding_dim,
        placement_embedding_dim: placement_embedding_dim,
        token_encoder_config: MlpConfig {
            hidden_size: (piece_embedding_dim + placement_embedding_dim + d_model),
            intermediate_size: 3 * d_model,
            output_size: d_model,
        },
    };

    let num_blocks = 8;
    let world_model_dim = 16;
    let model_cfg = TetrisWorldModelConfig {
        blocks_config: TetrisWorldModelBlockConfig {
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: world_model_dim,
            },
            attn_config: CausalSelfAttentionConfig {
                d_model: world_model_dim,
                n_attention_heads: 4,
                rope_theta: 10000.0,
                max_position_embeddings: 128,
            },
            mlp_config: MlpConfig {
                hidden_size: world_model_dim,
                intermediate_size: world_model_dim,
                output_size: world_model_dim,
            },
        },
        num_blocks,
        board_head_config: MlpConfig {
            hidden_size: world_model_dim,
            intermediate_size: world_model_dim,
            output_size: TetrisBoardRaw::SIZE * TetrisBoardRaw::NUM_TETRIS_CELL_STATES,
        },
        orientation_head_config: MlpConfig {
            hidden_size: world_model_dim,
            intermediate_size: world_model_dim,
            output_size: TetrisPieceOrientation::NUM_ORIENTATIONS,
        },
        dyn_tan_config: DynamicTanhConfig {
            alpha_init_value: 1.0,
            normalized_shape: world_model_dim,
        },
    };

    let model = TetrisWorldModel::init(&model_vs, &model_cfg).unwrap();

    let tokenizer = TetrisWorldModelTokenizer::init(&model_vs, &tokenizer_cfg).unwrap();
    let mut model_optimizer = AdamW::new(model_varmap.all_vars(), ParamsAdamW::default()).unwrap();
    let model_params = model_varmap.all_vars();
    let mut model_grad_accumulator = GradientAccumulator::new(ACCUMULATE_GRADIENTS_STEPS);

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
        let current_board =
            TetrisBoardsDistTensor::try_from(datum_sequence.current_boards[0].clone()).unwrap();
        // let current_pieces_tensor =
        //     TetrisPieceTensor::from_pieces(&datum_sequence.pieces[0], &device).unwrap();
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

        let context_tokens = tokenizer
            .forward_context(
                &goal_board,
                &datum_sequence
                    .current_boards
                    .iter()
                    .map(|b| TetrisBoardsDistTensor::try_from(b.clone()).unwrap())
                    .collect::<Vec<TetrisBoardsDistTensor>>(),
                &datum_sequence
                    .pieces
                    .iter()
                    .map(|p| TetrisPieceTensor::from_pieces(p, &device).unwrap())
                    .collect::<Vec<TetrisPieceTensor>>(),
                &datum_sequence.placements,
            )
            .unwrap();

        let (world_model_output, world_model_logits) = model
            .forward_all(&context_tokens, &current_pieces_tensor)
            .unwrap();

        // let loss = board_loss(&world_model_output, &goal_board).unwrap();

        // Apply optimizer step if accumulation is complete
        let should_step = model_grad_accumulator
            .apply_and_reset(&mut model_optimizer, &model_params)
            .unwrap();

        if should_step {
            println!("Stepping model");
        }
    }
}
