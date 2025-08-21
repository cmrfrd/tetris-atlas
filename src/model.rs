use burn::config::Config;
use burn::module::Module;
use burn::tensor::{Int, Shape};
use burn::tensor::{Tensor, backend::Backend};
use burn::{
    module::Param,
    nn::{Linear, LinearConfig},
    tensor::backend::AutodiffBackend,
    train::{TrainOutput, TrainStep, ValidStep},
};
use burn::{
    nn::{
        Embedding, EmbeddingConfig, Gelu, RotaryEncoding, RotaryEncodingConfig,
        loss::CrossEntropyLossConfig,
    },
    train::ClassificationOutput,
};

use crate::dataset::{TetrisBatch, TetrisDistBatch, TetrisInitDistBatch, TetrisSequenceDistBatch};
use crate::tetris::{TetrisGameSet, TetrisPiecePlacement};

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    rope: RotaryEncoding<B>,
    n_heads: usize,
    head_dim: usize,
}

#[derive(Config)]
pub struct CausalSelfAttentionConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl CausalSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalSelfAttention<B> {
        let head_dim = self.d_model / self.n_heads;

        let q_proj = LinearConfig::new(self.d_model, self.d_model).init(device);
        let k_proj = LinearConfig::new(self.d_model, self.d_model).init(device);
        let v_proj = LinearConfig::new(self.d_model, self.d_model).init(device);
        let o_proj = LinearConfig::new(self.d_model, self.d_model).init(device);

        let rope = RotaryEncodingConfig::new(self.max_position_embeddings, head_dim)
            .with_theta(self.rope_theta)
            .init(device);

        CausalSelfAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            n_heads: self.n_heads,
            head_dim,
        }
    }
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();

        // Project to QKV
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape to (batch, n_heads, seq_len, head_dim)
        let q = q
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);

        // Apply RoPE (rotary positional encoding)
        let q = self.rope.apply(q, 0);
        let k = self.rope.apply(k, 0);

        // Compute scaled dot-product attention
        let attn_scores = q.matmul(k.swap_dims(2, 3)) / f32::sqrt(self.head_dim as f32);
        let attn_weights = burn::tensor::activation::softmax(attn_scores, 3);
        let context = attn_weights.matmul(v);

        // Restore shape to (batch, seq_len, d_model)
        let context =
            context
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, self.n_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(context)
    }
}

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    proj: Linear<B>,
    activation: Gelu,
}

#[derive(Config)]
pub struct MlpConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl MlpConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        let fc1 = LinearConfig::new(self.hidden_size, self.intermediate_size).init(device);
        let fc2 = LinearConfig::new(self.hidden_size, self.intermediate_size).init(device);
        let proj = LinearConfig::new(self.intermediate_size, self.hidden_size).init(device);

        Mlp {
            fc1,
            fc2,
            proj,
            activation: Gelu::new(),
        }
    }
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x0 = self.fc1.forward(x.clone());
        let x0_activation = self.activation.forward(x0);
        let x1 = self.fc2.forward(x);
        self.proj.forward(x0_activation * x1)
    }
}

#[derive(Module, Debug)]
struct DynamicTanh<B: Backend> {
    alpha: Param<Tensor<B, 1>>,
    weight: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
    normalized_shape: usize,
}

#[derive(Config)]
pub struct DynamicTanhConfig {
    #[config(default = 0.5)]
    pub alpha_init_value: f64,

    pub normalized_shape: usize,
}

impl DynamicTanhConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> DynamicTanh<B> {
        let weight = Tensor::ones([self.normalized_shape], device);
        let bias = Tensor::zeros([self.normalized_shape], device);
        let alpha = Tensor::from_data([self.alpha_init_value as f32], device);

        DynamicTanh {
            alpha: Param::from_tensor(alpha),
            weight: Param::from_tensor(weight),
            bias: Param::from_tensor(bias),
            normalized_shape: self.normalized_shape,
        }
    }
}

impl<B: Backend> DynamicTanh<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let alpha_value = self.alpha.val().reshape([1, 1, 1]);
        let x = burn::tensor::activation::tanh(x * alpha_value);
        let weight_clone = self
            .weight
            .val()
            .clone()
            .reshape([1, 1, self.normalized_shape]);
        let bias_clone = self
            .bias
            .val()
            .clone()
            .reshape([1, 1, self.normalized_shape]);

        x * weight_clone + bias_clone
    }
}

#[derive(Module, Debug)]
pub struct TetrisGameTransformerBlock<B: Backend> {
    dyn_tan_1: DynamicTanh<B>,
    attn: CausalSelfAttention<B>,
    dyn_tan_2: DynamicTanh<B>,
    mlp: Mlp<B>,
}

#[derive(Config)]
pub struct TetrisGameTransformerBlockConfig {
    pub dyn_tan_config: DynamicTanhConfig,
    pub attn_config: CausalSelfAttentionConfig,
    pub mlp_config: MlpConfig,
}

impl TetrisGameTransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TetrisGameTransformerBlock<B> {
        let dyn_tan_1 = self.dyn_tan_config.init(device);
        let attn = self.attn_config.init(device);
        let dyn_tan_2 = self.dyn_tan_config.init(device);
        let mlp = self.mlp_config.init(device);

        TetrisGameTransformerBlock {
            dyn_tan_1,
            attn,
            dyn_tan_2,
            mlp,
        }
    }
}

impl<B: Backend> TetrisGameTransformerBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x_residual = x.clone();
        let x = self.dyn_tan_1.forward(x);
        let x = self.attn.forward(x) + x_residual.clone();
        let x_residual = x.clone();
        let x = self.dyn_tan_2.forward(x);
        let x = self.mlp.forward(x) + x_residual;
        x
    }
}

#[derive(Module, Debug)]
pub struct TetrisGameTransformer<B: Backend> {
    board_embedding: Embedding<B>,
    placement_embedding: Embedding<B>,
    blocks: Vec<TetrisGameTransformerBlock<B>>,
    dyn_tan: DynamicTanh<B>,
    output_layer: Linear<B>,

    num_placement_embedding_residuals: usize,
}

#[derive(Config)]
pub struct TetrisGameTransformerConfig {
    pub board_embedding_config: EmbeddingConfig,
    pub placement_embedding_config: EmbeddingConfig,

    // Number of transformer blocks that will receive the placement embedding as a residual
    pub num_placement_embedding_residuals: usize,

    pub blocks_config: TetrisGameTransformerBlockConfig,
    pub num_blocks: usize,

    pub dyn_tan_config: DynamicTanhConfig,
    pub output_layer_config: LinearConfig,
}

impl TetrisGameTransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TetrisGameTransformer<B> {
        let board_embedding = self.board_embedding_config.init(device);
        let placement_embedding = self.placement_embedding_config.init(device);
        let dyn_tan = self.dyn_tan_config.init(device);
        let num_placement_embedding_residuals = self.num_placement_embedding_residuals;
        let blocks = (0..self.num_blocks)
            .map(|_| self.blocks_config.init(device))
            .collect();
        let output_layer = self.output_layer_config.init(device);

        TetrisGameTransformer {
            board_embedding,
            placement_embedding,
            blocks,
            dyn_tan,
            output_layer,
            num_placement_embedding_residuals,
        }
    }
}

impl<B: Backend> TetrisGameTransformer<B> {
    pub fn soft_forward_board_embeddings(&self, board_dist: Tensor<B, 3>) -> Tensor<B, 3> {
        // [S, D] -> [1, 1, S, D]
        let board_embedding_broadcast = self.board_embedding.weight.val().unsqueeze::<4>();
        // [B, T, S] -> [B, T, 1, S]
        let board_dist = board_dist.unsqueeze_dim(2);
        // [B, T, 1, S] @ [1, 1, S, D] -> [B, T, 1, D] -> [B, T, D]
        let soft_board_embeddings = board_dist.matmul(board_embedding_broadcast).squeeze(2);
        soft_board_embeddings
    }

    pub fn soft_forward_placement_embeddings(&self, placement_dist: Tensor<B, 2>) -> Tensor<B, 2> {
        // [P, D] -> [1, P, D]
        let placement_embedding_broadcast: Tensor<B, 3> =
            self.placement_embedding.weight.val().unsqueeze_dim(0);
        // [B, P] -> [B, 1, P]
        let placement_dist: Tensor<B, 3> = placement_dist.unsqueeze_dim(1);
        // [B, 1, P] @ [1, P, D] -> [B, 1, D] -> [B, D]
        let soft_placement_embeddings: Tensor<B, 2> = placement_dist
            .matmul(placement_embedding_broadcast)
            .squeeze(1);

        soft_placement_embeddings
    }

    pub fn soft_forward(
        &self,
        tetris_boards: Tensor<B, 3>,
        placements: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let soft_board_embeddings = self.soft_forward_board_embeddings(tetris_boards);

        // We apply placement embeddings along each token
        let soft_placement_embeddings: Tensor<B, 3> = self
            .soft_forward_placement_embeddings(placements)
            .unsqueeze_dim(1);

        let mut x = soft_board_embeddings;
        for (i, block) in self.blocks.iter().enumerate() {
            if i < self.num_placement_embedding_residuals {
                x = block.forward(x) + soft_placement_embeddings.clone();
            } else {
                x = block.forward(x);
            }
        }

        let x = self.dyn_tan.forward(x);

        // Output is a distribution over the cell states
        // [B, T, S]
        let output = self.output_layer.forward(x);
        output
    }

    pub fn forward(&self, item: TetrisBatch<B>) -> Tensor<B, 3> {
        // A placement is a single integer token. so when getting a placement embedding,
        // we get [B, 1, E] which we need to reshape to [B, E]
        let placements = item.placements;
        let placement_embeddings = self.placement_embedding.forward(placements.int());

        let current_boards = item.current_boards;
        let board_embeddings = self.board_embedding.forward(current_boards.int());

        let mut x = board_embeddings;
        for (i, block) in self.blocks.iter().enumerate() {
            if i < self.num_placement_embedding_residuals {
                x = block.forward(x) + placement_embeddings.clone();
            } else {
                x = block.forward(x);
            }
        }

        let x = self.dyn_tan.forward(x);

        // Output is a distribution over the cell states
        // [B, T, S]
        let output = self.output_layer.forward(x);
        output
    }

    pub fn forward_with_classification(&self, item: TetrisBatch<B>) -> ClassificationOutput<B> {
        let result_boards = item.result_boards.clone();
        let output = self.forward(item);

        // [B, T, S] -> [B * T, S]
        let [batch_size, seq_len, states] = output.dims();
        let output_flat = output.reshape([batch_size * seq_len, states]);

        // [B, T] -> [B * T]
        let targets_flat = result_boards.int().reshape([batch_size * seq_len]);

        let device = &self.devices()[0];
        let loss = CrossEntropyLossConfig::new().init(device);
        let loss = loss.forward(output_flat.clone(), targets_flat.clone());

        ClassificationOutput {
            loss,
            output: output_flat,
            targets: targets_flat,
        }
    }

    pub fn soft_forward_with_classification(
        &self,
        item: TetrisDistBatch<B>,
    ) -> ClassificationOutput<B> {
        let result_boards = item.result_boards_dist.clone();
        let output = self.soft_forward(item.current_boards_dist, item.placements_dist);

        // [B, T, S] -> [B * T, S]
        let [batch_size, seq_len, states] = output.dims();
        let output_flat = output.reshape([batch_size * seq_len, states]);

        // [B, T, S] -> [B, T] -> [B * T]
        let targets_flat = result_boards
            .int()
            .argmax(2)
            .reshape([batch_size * seq_len]);

        let device = &self.devices()[0];
        let loss = CrossEntropyLossConfig::new().init(device);
        let loss = loss.forward(output_flat.clone(), targets_flat.clone());

        ClassificationOutput {
            loss,
            output: output_flat,
            targets: targets_flat,
        }
    }

    pub fn forward_sequence(&self, item: TetrisSequenceDistBatch<B>) -> ClassificationOutput<B> {
        let mut targets = Vec::new();
        let mut soft_forward_outputs = Vec::new();
        let mut dist_iter = item.iter_seq();

        let first_seq_item_dist_batch = dist_iter.next().unwrap();
        targets.push(first_seq_item_dist_batch.result_boards_dist.clone());
        let mut soft_forward_output = self.soft_forward(
            first_seq_item_dist_batch.current_boards_dist,
            first_seq_item_dist_batch.placements_dist,
        );
        soft_forward_outputs.push(soft_forward_output.clone());

        for seq_item_dist_batch in dist_iter {
            let result_boards_dist = seq_item_dist_batch.result_boards_dist.clone();
            targets.push(result_boards_dist);

            let mut batch = seq_item_dist_batch;
            batch.current_boards_dist = soft_forward_output;
            soft_forward_output =
                self.soft_forward(batch.current_boards_dist, batch.placements_dist);
            soft_forward_outputs.push(soft_forward_output.clone());
        }

        let soft_forward_outputs_flat = soft_forward_outputs
            .iter()
            .map(|output| {
                let [batch_size, seq_len, states] = output.dims();
                output.clone().reshape([batch_size * seq_len, states])
            })
            .reduce(|a, b| Tensor::cat(vec![a, b], 0))
            .unwrap();
        let targets_flat = targets
            .iter()
            .map(|target| {
                let [batch_size, seq_len, _] = target.dims();
                target
                    .clone()
                    .int()
                    .argmax(2)
                    .reshape([batch_size * seq_len])
            })
            .reduce(|a, b| Tensor::cat(vec![a, b], 0))
            .unwrap();

        let device = &self.devices()[0];
        let loss = CrossEntropyLossConfig::new().init(device);
        let loss = loss.forward(soft_forward_outputs_flat.clone(), targets_flat.clone());

        ClassificationOutput {
            loss,
            output: soft_forward_outputs_flat,
            targets: targets_flat,
        }
    }
}

/// Step implementaiton for a batch of (input, placement, output) tuples
impl<B: AutodiffBackend> TrainStep<TetrisBatch<B>, ClassificationOutput<B>>
    for TetrisGameTransformer<B>
{
    fn step(&self, item: TetrisBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_with_classification(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<TetrisBatch<B>, ClassificationOutput<B>> for TetrisGameTransformer<B> {
    fn step(&self, item: TetrisBatch<B>) -> ClassificationOutput<B> {
        self.forward_with_classification(item)
    }
}

/// Step implementaiton for a batch of distributions of (input, placement, output) tuples
impl<B: AutodiffBackend> TrainStep<TetrisDistBatch<B>, ClassificationOutput<B>>
    for TetrisGameTransformer<B>
{
    fn step(&self, item: TetrisDistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.soft_forward_with_classification(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<TetrisDistBatch<B>, ClassificationOutput<B>>
    for TetrisGameTransformer<B>
{
    fn step(&self, item: TetrisDistBatch<B>) -> ClassificationOutput<B> {
        self.soft_forward_with_classification(item)
    }
}

/// Step implementaiton for a batch of sequences of distributions of (input, placement, output) tuples
impl<B: AutodiffBackend> TrainStep<TetrisSequenceDistBatch<B>, ClassificationOutput<B>>
    for TetrisGameTransformer<B>
{
    fn step(&self, item: TetrisSequenceDistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_sequence(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<TetrisSequenceDistBatch<B>, ClassificationOutput<B>>
    for TetrisGameTransformer<B>
{
    fn step(&self, item: TetrisSequenceDistBatch<B>) -> ClassificationOutput<B> {
        self.forward_sequence(item)
    }
}

#[derive(Module, Debug)]
pub struct TetrisPlayerTransformer<B: Backend> {
    board_embedding: Embedding<B>,
    piece_embedding: Embedding<B>,

    blocks: Vec<TetrisGameTransformerBlock<B>>,
    dyn_tan: DynamicTanh<B>,
    output_layer: Linear<B>,

    num_placement_embedding_residuals: usize,
}

#[derive(Config)]
pub struct TetrisPlayerTransformerConfig {
    pub board_embedding_config: EmbeddingConfig,
    pub piece_embedding_config: EmbeddingConfig,

    // Number of transformer blocks that will receive the placement embedding as a residual
    pub num_placement_embedding_residuals: usize,

    pub blocks_config: TetrisGameTransformerBlockConfig,
    pub num_blocks: usize,

    pub dyn_tan_config: DynamicTanhConfig,
    pub output_layer_config: LinearConfig,
}

impl TetrisPlayerTransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TetrisPlayerTransformer<B> {
        let board_embedding = self.board_embedding_config.init(device);
        let piece_embedding = self.piece_embedding_config.init(device);
        let dyn_tan = self.dyn_tan_config.init(device);
        let num_placement_embedding_residuals = self.num_placement_embedding_residuals;
        let blocks = (0..self.num_blocks)
            .map(|_| self.blocks_config.init(device))
            .collect();
        let output_layer = self.output_layer_config.init(device);

        TetrisPlayerTransformer {
            board_embedding,
            piece_embedding,
            blocks,
            dyn_tan,
            output_layer,
            num_placement_embedding_residuals,
        }
    }
}

impl<B: Backend> TetrisPlayerTransformer<B> {
    pub fn soft_forward_board_embeddings(&self, board_dist: Tensor<B, 3>) -> Tensor<B, 3> {
        // [S, D] -> [1, 1, S, D]
        let board_embedding_broadcast = self.board_embedding.weight.val().unsqueeze::<4>();
        // [B, T, S] -> [B, T, 1, S]
        let board_dist = board_dist.unsqueeze_dim(2);
        // [B, T, 1, S] @ [1, 1, S, D] -> [B, T, 1, D] -> [B, T, D]
        let soft_board_embeddings = board_dist.matmul(board_embedding_broadcast).squeeze(2);
        soft_board_embeddings
    }

    pub fn soft_forward(&self, game_set: &TetrisGameSet, board_dist: Tensor<B, 3>) -> Tensor<B, 2> {
        let device = &self.devices()[0];
        let pieces: Tensor<B, 2, Int> = Tensor::stack(
            game_set
                .current_pieces()
                .to_vec()
                .into_iter()
                .map(|piece| Tensor::<B, 1, Int>::from_ints([piece.index()], device))
                .collect::<Vec<_>>(),
            0,
        );

        let soft_board_embeddings = self.soft_forward_board_embeddings(board_dist);
        let soft_piece_embeddings = self.piece_embedding.forward(pieces);

        let mut x = soft_board_embeddings;
        for (i, block) in self.blocks.iter().enumerate() {
            if i < self.num_placement_embedding_residuals {
                x = block.forward(x) + soft_piece_embeddings.clone();
            } else {
                x = block.forward(x);
            }
        }

        let x = self.dyn_tan.forward(x);

        // Reduce to a single distribution vector
        // [B, T, D] -> [B, 1, D] -> [B, D]
        let x = x.mean_dim(1).squeeze(1);

        // Output is a distribution over placement indices
        // [B, D] -> [B, P]
        let output = self.output_layer.forward(x);

        // Apply softmax to get proper probability distribution
        let output = burn::tensor::activation::softmax(output, 1);

        // Mask out invalid placements
        // cat([1, P] ..., 0) -> [B, P]
        let mask = game_set
            .current_pieces()
            .to_vec()
            .into_iter()
            .map(|piece| TetrisPiecePlacement::indices_from_piece(piece))
            .map(|placement_indices| {
                let mask = Tensor::<B, 1>::zeros(
                    Shape::new([TetrisPiecePlacement::NUM_PLACEMENTS]),
                    device,
                );
                let mask_values =
                    Tensor::<B, 1>::ones(Shape::new([placement_indices.len()]), device);
                let mask: Tensor<B, 2> = mask
                    .slice_assign(placement_indices, mask_values)
                    .unsqueeze_dim(0);
                mask
            })
            .reduce(|a, b| Tensor::cat(vec![a, b], 0))
            .unwrap();

        output * mask
    }

    pub fn forward_get_placement(
        &self,
        game_set: &TetrisGameSet,
        board_dist: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Vec<TetrisPiecePlacement>) {
        let output = self.soft_forward(game_set, board_dist);
        let placements = output
            .clone()
            .argmax(1)
            .squeeze::<1>(1)
            .into_data()
            .into_vec::<i64>()
            .unwrap()
            .into_iter()
            .map(|placement| TetrisPiecePlacement::from_index(placement as u8))
            .collect();
        (output, placements)
    }
}

#[cfg(test)]
mod tests {
    use crate::tetris::{BOARD_SIZE, TetrisGame, TetrisGameSet, TetrisPiece, TetrisPiecePlacement};

    use super::*;
    use burn::backend::{Autodiff, NdArray, ndarray::NdArrayDevice};

    #[test]
    fn test_mlp_forward() {
        let device = NdArrayDevice::Cpu;

        let mlp_config = MlpConfig {
            hidden_size: 10,
            intermediate_size: 20,
        };

        let mlp: Mlp<Autodiff<NdArray>> = mlp_config.init(&device);

        let x = Tensor::zeros([1, 10, 10], &device);
        let y = mlp.forward(x);

        assert_eq!(y.dims(), [1, 10, 10]);
    }

    #[test]
    fn test_dyn_tanh_forward() {
        let device = NdArrayDevice::Cpu;

        let dyn_tanh_config = DynamicTanhConfig {
            alpha_init_value: 0.5,
            normalized_shape: 10,
        };

        let dyn_tanh: DynamicTanh<Autodiff<NdArray>> = dyn_tanh_config.init(&device);

        let x = Tensor::zeros([1, 100, 10], &device);
        let y = dyn_tanh.forward(x);

        assert_eq!(y.dims(), [1, 100, 10]);
    }

    #[test]
    fn test_causal_self_attention_forward() {
        let device = NdArrayDevice::Cpu;

        let causal_self_attention_config = CausalSelfAttentionConfig {
            d_model: 16,
            n_heads: 8,
            rope_theta: 10000.0,
            max_position_embeddings: 100,
        };

        let causal_self_attention: CausalSelfAttention<Autodiff<NdArray>> =
            causal_self_attention_config.init(&device);

        let x = Tensor::zeros([1, 100, 16], &device);
        let y = causal_self_attention.forward(x);

        assert_eq!(y.dims(), [1, 100, 16]);
    }

    #[test]
    fn test_tetris_game_transformer_forward() {
        let device = NdArrayDevice::Cpu;

        let num_cell_states = 2;
        let num_placements = TetrisPiecePlacement::NUM_PLACEMENTS;
        let d_model = 16;

        let tetris_game_transformer_config = TetrisGameTransformerConfig {
            board_embedding_config: EmbeddingConfig::new(num_cell_states, d_model),
            placement_embedding_config: EmbeddingConfig::new(num_placements, d_model),
            num_placement_embedding_residuals: 2,
            blocks_config: TetrisGameTransformerBlockConfig {
                dyn_tan_config: DynamicTanhConfig {
                    alpha_init_value: 0.5,
                    normalized_shape: d_model,
                },
                attn_config: CausalSelfAttentionConfig {
                    d_model,
                    n_heads: 8,
                    rope_theta: 10000.0,
                    max_position_embeddings: BOARD_SIZE,
                },
                mlp_config: MlpConfig {
                    hidden_size: d_model,
                    intermediate_size: 32,
                },
            },
            num_blocks: 3,
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 0.5,
                normalized_shape: d_model,
            },
            output_layer_config: LinearConfig::new(d_model, num_cell_states),
        };

        let tetris_game_transformer: TetrisGameTransformer<Autodiff<NdArray>> =
            tetris_game_transformer_config.init(&device);

        let item = TetrisBatch {
            current_boards: Tensor::zeros([1, BOARD_SIZE], &device),
            placements: Tensor::zeros([1, 1], &device),
            result_boards: Tensor::zeros([1, BOARD_SIZE], &device),
        };
        let y = tetris_game_transformer.forward(item.clone());
        let _ = tetris_game_transformer.forward_with_classification(item.clone());

        assert_eq!(y.dims(), [1, BOARD_SIZE, 2]);

        let item = TetrisDistBatch {
            current_boards_dist: Tensor::ones([1, BOARD_SIZE, 2], &device),
            placements_dist: Tensor::ones([1, TetrisPiecePlacement::NUM_PLACEMENTS], &device),
            result_boards_dist: Tensor::ones([1, BOARD_SIZE, 2], &device),
        };
        let y = tetris_game_transformer.soft_forward(
            item.current_boards_dist.clone(),
            item.placements_dist.clone(),
        );
        let _ = tetris_game_transformer.soft_forward_with_classification(item.clone());

        assert_eq!(y.dims(), [1, BOARD_SIZE, 2]);

        let item = TetrisSequenceDistBatch {
            current_boards_dist: Tensor::ones([1, 2, BOARD_SIZE, 2], &device),
            placements_dist: Tensor::ones([1, 2, TetrisPiecePlacement::NUM_PLACEMENTS], &device),
            result_boards_dist: Tensor::ones([1, 2, BOARD_SIZE, 2], &device),
        };
        let y = tetris_game_transformer.forward_sequence(item.clone());

        assert_eq!(y.output.dims(), [1 * 2 * BOARD_SIZE, 2]);
    }

    #[test]
    fn test_tetris_player_transformer_forward() {
        let device = NdArrayDevice::Cpu;

        let num_cell_states = 2;
        let num_pieces = TetrisPiece::NUM_PIECES;
        let num_placements = TetrisPiecePlacement::NUM_PLACEMENTS;
        let d_model = 8;

        let tetris_player_transformer_config = TetrisPlayerTransformerConfig {
            board_embedding_config: EmbeddingConfig::new(num_cell_states, d_model),
            piece_embedding_config: EmbeddingConfig::new(num_pieces, d_model),
            num_placement_embedding_residuals: 2,
            blocks_config: TetrisGameTransformerBlockConfig {
                dyn_tan_config: DynamicTanhConfig {
                    alpha_init_value: 0.5,
                    normalized_shape: d_model,
                },
                attn_config: CausalSelfAttentionConfig {
                    d_model,
                    n_heads: 2,
                    rope_theta: 10000.0,
                    max_position_embeddings: BOARD_SIZE,
                },
                mlp_config: MlpConfig {
                    hidden_size: d_model,
                    intermediate_size: 32,
                },
            },
            num_blocks: 4,
            dyn_tan_config: DynamicTanhConfig {
                alpha_init_value: 0.5,
                normalized_shape: d_model,
            },
            output_layer_config: LinearConfig::new(d_model, num_placements),
        };

        let tetris_player_transformer: TetrisPlayerTransformer<Autodiff<NdArray>> =
            tetris_player_transformer_config.init(&device);

        let batch_size = 8;
        let game_set = TetrisGameSet::new_with_seed(0, batch_size);
        let current_boards = Tensor::ones([batch_size, BOARD_SIZE, 2], &device);
        let pieces_raw = game_set
            .current_pieces()
            .to_vec()
            .into_iter()
            .map(|p| p.index())
            .collect::<Vec<_>>();

        let item = TetrisInitDistBatch {
            game_set,
            current_boards,
        };
        let y = tetris_player_transformer.soft_forward(&item.game_set, item.current_boards.clone());
        assert_eq!(y.dims(), [8, TetrisPiecePlacement::NUM_PLACEMENTS]);

        let (output, _) = tetris_player_transformer
            .forward_get_placement(&item.game_set, item.current_boards.clone());
        let [batch_size, _] = output.dims();
        assert_eq!(output.dims(), [8, TetrisPiecePlacement::NUM_PLACEMENTS]);
        let output_vec = output.into_data().into_vec::<f32>().unwrap();

        for i in 0..batch_size {
            let output_chunk = &output_vec[i * TetrisPiecePlacement::NUM_PLACEMENTS
                ..(i + 1) * TetrisPiecePlacement::NUM_PLACEMENTS];
            let epsilon = 1e-6;
            assert!(
                output_chunk.iter().filter(|x| x.abs() > epsilon).count()
                    == TetrisPiecePlacement::all_from_piece(TetrisPiece::new(pieces_raw[i] as u8))
                        .len()
            );
        }
    }
}
