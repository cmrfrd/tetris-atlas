use std::path::PathBuf;

use anyhow::Result;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarMap;
use candle_nn::{VarBuilder, init::Init};
use rand::RngCore;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use tensorboard::summary_writer::SummaryWriter;

use crate::ops::create_orientation_mask;
use crate::tensors::{
    TetrisBoardsTensor, TetrisPieceOneHotTensor, TetrisPieceOrientationLogitsTensor,
    TetrisPieceTensor,
};
use crate::tetris::{TetrisBoard, TetrisGameSet, TetrisPiece, TetrisPieceOrientation};
use crate::wrapped_tensor::WrappedTensor;

// /// Reorder `x` in place so that after[i] = before[indices[i]].
// /// - `indices` must be a permutation of 0..n-1
// /// - Modifies `indices` (marks visited by adding n)
// /// - O(1) aux, T: Copy
// pub fn gather_permute_in_place<T: Copy, X: std::ops::IndexMut<usize, Output = T>>(
//     x: &mut X,
//     indices: &mut [usize],
// ) -> Result<(), &'static str> {
//     let n = indices.len();
//     if indices.len() != n {
//         return Err("indices.len() must equal x.len()");
//     }
//     if n == 0 {
//         return Ok(());
//     }
//     if n > usize::MAX / 2 {
//         return Err("n too large for add-n marking");
//     }

//     // Verify indices is a valid permutation
//     let mut seen = vec![false; n];
//     for &v in indices.iter() {
//         if v >= n || std::mem::replace(&mut seen[v], true) {
//             return Err("indices is not a valid permutation");
//         }
//     }

//     let nn = n;

//     let mut i = 0usize;
//     while i < n {
//         // already visited/placed?
//         if indices[i] >= nn {
//             i += 1;
//             continue;
//         }

//         // follow the cycle starting at i
//         let mut cur = i;
//         let tmp = x[i]; // displaced value from position i

//         loop {
//             let src = indices[cur]; // src < n (by contract)
//             indices[cur] = src + nn; // mark visited

//             // if the next link is visited, close the cycle
//             if indices[src] >= nn {
//                 x[cur] = tmp;
//                 break;
//             } else {
//                 x[cur] = x[src];
//                 cur = src;
//             }
//         }

//         i += 1;
//     }

//     Ok(())
// }

/// Population of simple MLP policies with configurable layer dimensions and SiLU activations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetrisEvolutionPopulationConfig {
    /// Layer dimensions: [input_dim, hidden1, hidden2, ..., output_dim]
    /// The first dimension should match: board_size + num_pieces
    /// The last dimension should match: NUM_ORIENTATIONS
    pub layer_dims: Vec<usize>,
    /// Number of individuals in the population
    pub population_size: usize,
}

#[derive(Debug, Clone)]
pub struct TetrisEvolutionPopulation {
    population_size: usize,
    weights: Vec<Tensor>, // [P, in_dim, out_dim]
    biases: Vec<Tensor>,  // [P, out_dim]
}

impl TetrisEvolutionPopulation {
    pub fn init(vb: &VarBuilder, cfg: &TetrisEvolutionPopulationConfig) -> Result<Self> {
        if cfg.layer_dims.len() < 2 {
            anyhow::bail!("Need at least 2 layer dimensions (input and output)");
        }

        let expected_input = TetrisBoard::SIZE + TetrisPiece::NUM_PIECES;
        if cfg.layer_dims[0] != expected_input {
            anyhow::bail!(
                "expected first layer dim {} (board + piece features), got {}",
                expected_input,
                cfg.layer_dims[0]
            );
        }

        let expected_output = TetrisPieceOrientation::NUM_ORIENTATIONS;
        if *cfg.layer_dims.last().unwrap() != expected_output {
            anyhow::bail!(
                "expected last layer dim {} (num orientations), got {}",
                expected_output,
                cfg.layer_dims.last().unwrap()
            );
        }

        let pop = cfg.population_size;
        let mut weights = Vec::with_capacity(cfg.layer_dims.len() - 1);
        let mut biases = Vec::with_capacity(cfg.layer_dims.len() - 1);

        for (idx, dims) in cfg.layer_dims.windows(2).enumerate() {
            let (in_dim, out_dim) = (dims[0], dims[1]);
            let mut weights_layer = Vec::with_capacity(pop);
            let mut biases_layer = Vec::with_capacity(pop);
            for individual in 0..pop {
                let builder = vb.pp(format!("individual_{individual}_layer_{idx}"));
                let weight = builder.get_with_hints(
                    (in_dim, out_dim),
                    "weight",
                    Init::Randn {
                        mean: 0.0,
                        stdev: 0.02,
                    },
                )?;
                let bias = builder.get_with_hints(
                    (out_dim,),
                    "bias",
                    Init::Randn {
                        mean: 0.0,
                        stdev: 0.02,
                    },
                )?;
                weights_layer.push(weight);
                biases_layer.push(bias);
            }
            weights.push(Tensor::stack(&weights_layer, 0)?);
            biases.push(Tensor::stack(&biases_layer, 0)?);
        }

        Ok(Self {
            population_size: pop,
            weights,
            biases,
        })
    }

    fn check_dimensions(&self) -> Result<()> {
        if self.weights.is_empty() {
            anyhow::bail!("Population network must contain at least one layer");
        }
        Ok(())
    }

    /// Forward producing unmasked orientation logits [population, batch, NUM_ORIENTATIONS]
    pub fn forward_logits(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationLogitsTensor> {
        self.check_dimensions()?;
        let (batch, _) = current_board.shape_tuple();

        let board_flat = current_board
            .to_dtype(DType::F32)?
            .reshape(&[batch, TetrisBoard::HEIGHT * TetrisBoard::WIDTH])?;
        let piece_one_hot = TetrisPieceOneHotTensor::from_piece_tensor(current_piece.clone())?;
        let input = Tensor::cat(&[&board_flat, &piece_one_hot], 1)?; // [B, input_dim]

        // Input shape: [population, in_dim]
        let mut activations = input;

        for (layer_idx, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            // activations: [P, in_dim], w: [P, in_dim, out_dim]
            // Use unsqueeze to make activations [P, 1, in_dim] for batched matmul
            let activations_expanded = activations.unsqueeze(1)?; // [P, 1, in_dim]
            let matmul = activations_expanded.matmul(w)?; // [P, 1, out_dim]
            let matmul = matmul.squeeze(1)?; // [P, out_dim]
            activations = (&matmul + b)?;
            if layer_idx < self.weights.len() - 1 {
                activations = candle_nn::ops::silu(&activations)?;
            }
        }

        Ok(TetrisPieceOrientationLogitsTensor::try_from(activations)?)
    }

    /// Compute masked action distribution over orientations for the entire population.
    pub fn forward_masked(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationLogitsTensor> {
        let logits = self.forward_logits(current_board, current_piece)?; // [P, O]

        // Mask invalid orientations
        let mask = create_orientation_mask(current_piece)?; // [P, O] (0/1)
        let device = logits.device();
        let zero = mask.zeros_like()?;
        let keep = mask.gt(&zero)?;
        let neg_inf = Tensor::new(-1e9f32, &device)?.broadcast_as(logits.dims())?;
        let masked = keep.where_cond(&logits, &neg_inf)?;
        Ok(TetrisPieceOrientationLogitsTensor::try_from(masked)?)
    }

    pub fn permute_population(
        &mut self,
        indices: &[u32],
        num_lost: usize,
        perturb_norm: Option<f64>,
    ) -> Result<()> {
        let device = self.weights[0].device().clone();
        let mut inverse_indices = vec![0u32; self.population_size];
        for (new_pos, &old_pos) in indices.iter().enumerate() {
            inverse_indices[old_pos as usize] = new_pos as u32;
        }
        let inverse_indices_tensor =
            Tensor::from_slice(&inverse_indices, (self.population_size,), &device)?;

        let mutation_indices = (self.population_size - num_lost..self.population_size)
            .map(|i| i as u32)
            .collect::<Vec<_>>();
        let mutation_indices_tensor = Tensor::from_slice(&mutation_indices, (num_lost,), &device)?;

        self.weights
            .par_iter()
            .chain(self.biases.par_iter())
            .try_for_each(|param| -> Result<()> {
                let dims = param.dims();
                debug_assert!(dims[0] == self.population_size);

                let flatten_dims_product = dims[1..].iter().product();
                let output_dim = flatten_dims_product;

                let noise = Self::apply_norm(
                    Tensor::randn(0.0, 0.01, (num_lost, output_dim), &device)?
                        .to_dtype(param.dtype())?,
                    perturb_norm,
                )?;

                let noise = noise
                    .reshape(
                        std::iter::once(num_lost)
                            .chain(dims[1..].iter().copied())
                            .collect::<Vec<_>>(),
                    )?
                    .contiguous()?;

                let mutation_indices_broadcast = mutation_indices_tensor
                    .reshape(
                        std::iter::once(num_lost)
                            .chain(std::iter::repeat(1).take(dims.len() - 1))
                            .collect::<Vec<_>>(),
                    )?
                    .broadcast_as(noise.dims())?
                    .contiguous()?;

                let original = param.copy()?;
                let inverse_indices_broadcast = inverse_indices_tensor
                    .reshape(
                        std::iter::once(self.population_size)
                            .chain(std::iter::repeat(1).take(dims.len() - 1))
                            .collect::<Vec<_>>(),
                    )?
                    .broadcast_as(param.shape())?
                    .contiguous()?;
                param.scatter_set(&inverse_indices_broadcast, &original, 0)?;
                param.scatter_add_set(&mutation_indices_broadcast, &noise, 0)?;
                Ok(())
            })?;

        Ok(())
    }

    fn apply_norm(tensor: Tensor, norm: Option<f64>) -> Result<Tensor> {
        if let Some(target_norm) = norm {
            if target_norm <= 0.0 {
                anyhow::bail!("perturb_norm must be positive");
            }

            let dims = tensor.dims();
            let norms = tensor.sqr()?.sum_all()?.sqrt()?;
            let denom =
                norms.maximum(&Tensor::full(f32::EPSILON, norms.dims(), tensor.device())?)?;
            let scaled = tensor
                .broadcast_div(&denom)?
                .affine(target_norm as f64, 0.0)?;
            let scaled = scaled.reshape(dims)?;
            Ok(scaled)
        } else {
            Ok(tensor)
        }
    }

    pub fn weight_sample(&self) -> Result<Vec<Vec<f32>>> {
        Ok(self.biases[0].to_vec2::<f32>()?)
    }
}

pub fn train_population(
    _run_name: String,
    logdir: Option<PathBuf>,
    _checkpoint_dir: Option<PathBuf>,
) -> Result<()> {
    let device = Device::new_cuda(0).unwrap();

    const NUM_ITERATIONS: usize = 100_000_000;
    const POPULATION_SIZE: usize = 256;
    const SEED: usize = 41;
    const NOISE_SCALE: f64 = 0.01;

    let model_varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&model_varmap, DType::F32, &device);

    let population_cfg = TetrisEvolutionPopulationConfig {
        layer_dims: vec![
            TetrisBoard::SIZE + TetrisPiece::NUM_PIECES,
            64,
            64,
            TetrisPieceOrientation::NUM_ORIENTATIONS,
        ],
        population_size: POPULATION_SIZE,
    };
    let mut population = TetrisEvolutionPopulation::init(&vb, &population_cfg)?;

    let mut summary_writer = logdir.map(|dir| {
        let _ = std::fs::create_dir_all(&dir);
        SummaryWriter::new(dir)
    });

    let mut game_set = TetrisGameSet::new_with_seed(SEED as u64, POPULATION_SIZE);
    let mut rng = rand::rng();

    let mut max_num_pieces = 0;
    let mut transitions_per_second = 0.0f64;
    let mut iterations_per_second = 0.0f64;
    let mut last_log_iteration = 0usize;
    let mut last_log_time = std::time::Instant::now();
    for i in 0..NUM_ITERATIONS {
        if i % 50 == 0 {
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(last_log_time).as_secs_f64();
            let delta_iterations = i.saturating_sub(last_log_iteration);
            if elapsed > 0.0 && delta_iterations > 0 {
                let delta_transitions = (delta_iterations * POPULATION_SIZE) as f64;
                transitions_per_second = delta_transitions / elapsed;
                iterations_per_second = delta_iterations as f64 / elapsed;
            }
            last_log_time = now;
            last_log_iteration = i;

            println!(
                "Iteration {:5} | Max Pieces: {:5} | Transitions/Sec: {:8.1} | Iterations/Sec: {:8.1}",
                i, max_num_pieces, transitions_per_second, iterations_per_second
            );
        }

        // println!(
        //     "\nIteration {:5}: {}",
        //     i,
        //     (0..POPULATION_SIZE)
        //         .map(|idx| format!("{:5}", game_set[idx].piece_count))
        //         .collect::<Vec<_>>()
        //         .join(" ")
        // );
        // println!(
        //     "Heights: {}",
        //     (0..POPULATION_SIZE)
        //         .map(|idx| format!("{:2}", game_set[idx].board().height()))
        //         .collect::<Vec<_>>()
        //         .join(" ")
        // );

        // // println!("Before update");
        // let weights = population.weight_sample()?;
        // for weight_vec in weights {
        //     let rounded = weight_vec
        //         .iter()
        //         .map(|x| (x * 100.0).round() / 100.0)
        //         .collect::<Vec<_>>();
        //     println!("{:?}", rounded);
        // }

        let current_board = TetrisBoardsTensor::from_gameset(game_set, &device)?;
        let current_pieces_array: Vec<TetrisPiece> = game_set.current_pieces().to_vec();
        let current_pieces_vec: Vec<_> = current_pieces_array[0..game_set.len()].to_vec();
        let current_piece =
            TetrisPieceTensor::from_pieces(&current_pieces_vec.as_slice(), &device)?;
        let masked = population.forward_masked(&current_board, &current_piece)?;
        let entropy_mean = masked.into_dist()?.entropy()?.mean_all()?;

        let sampled_orientations = masked.sample(0.1)?;
        let action_orientations = sampled_orientations.into_orientations()?;

        let lost_flags = game_set.apply_placement_from_orientations(&action_orientations);
        let mut alive = Vec::with_capacity(POPULATION_SIZE);
        let mut dead = Vec::with_capacity(POPULATION_SIZE);
        let piece_counts_array: Vec<u32> = game_set.piece_counts().to_vec();
        let piece_counts: Vec<_> = piece_counts_array[0..game_set.len()].to_vec();
        for (idx, (is_lost, &piece_count)) in lost_flags.iter().zip(&piece_counts).enumerate() {
            if bool::from(*is_lost) {
                dead.push(idx);
            } else {
                alive.push((idx, piece_count));
            }
        }
        let num_dead = dead.len();

        let iteration_max_pieces = piece_counts.iter().copied().max().unwrap_or(0);
        max_num_pieces = std::cmp::max(max_num_pieces, iteration_max_pieces);

        if let Some(writer) = summary_writer.as_mut() {
            let histogram: Vec<f64> = piece_counts.iter().map(|&count| count as f64).collect();
            writer.add_histogram("population/pieces_distribution", &histogram, None, i);
            writer.add_scalar("population/max_pieces", max_num_pieces as f32, i);
            writer.add_scalar("population/deaths", num_dead as f32, i);
            writer.add_scalar(
                "population/avg_entropy",
                entropy_mean.to_scalar::<f32>()?,
                i,
            );
        }

        if num_dead == 0 {
            continue;
        }
        if alive.len() == 0 {
            // Reset all games since none survived
            // game_set.apply_mut(|game| game.reset(None));
            continue;
        }

        // select new parents
        let alive_indices = alive
            .iter()
            .map(|(idx, _)| *idx as usize)
            .collect::<Vec<_>>();
        let weights = alive
            .iter()
            .map(|(_, piece_count)| (*piece_count as f64).powf(3.).clamp(0.0, 100_000.0))
            .collect::<Vec<_>>();
        let weighted_index = WeightedIndex::new(weights)?;
        let new_parent_positions = weighted_index
            .sample_iter(&mut rng)
            .take(num_dead)
            .collect::<Vec<_>>();
        let new_parents = new_parent_positions
            .iter()
            .map(|&pos| alive_indices[pos])
            .collect::<Vec<_>>();
        // println!("New parents: {:?}", new_parents);

        // Update the gameset so the lost games, move to the bottom and reset
        {
            // permute the gameset using the index vector
            let permute_vec = alive
                .iter()
                .map(|(idx, _)| *idx as usize)
                .chain(new_parents.iter().map(|idx| *idx as usize))
                .collect::<Vec<_>>();
            // println!("Permute vec: {:?}", permute_vec);
            // let _ = gather_permute_in_place(&mut game_set, &mut permute_vec);
            game_set.permute(&permute_vec);

            // reset the last len(dead) games
            for i in (POPULATION_SIZE - dead.len())..POPULATION_SIZE {
                game_set[i].reset(Some(rng.next_u64()));
            }
        }

        // Update the population so the dead citizens are replaced by the new parents + mutation
        {
            // generate the index vector
            let mut index_vector = [0u32; POPULATION_SIZE];
            for (i, (idx, _)) in alive.iter().enumerate() {
                index_vector[i] = *idx as u32;
            }
            for (i, idx) in new_parents.iter().enumerate() {
                index_vector[alive.len() + i] = *idx as u32;
            }
            // println!("Index vector: {:?}", index_vector);

            population.permute_population(&index_vector, dead.len(), Some(NOISE_SCALE))?;
        }

        // // println!("After update");
        // let weights = population.weight_sample()?;
        // for weight_vec in weights {
        //     let rounded = weight_vec
        //         .iter()
        //         .map(|x| (x * 100.0).round() / 100.0)
        //         .collect::<Vec<_>>();
        //     println!("{:?}", rounded);
        // }

        // println!("--------------------------------");
        // // Wait for user input before continuing
        // let mut input = String::new();
        // std::io::stdin().read_line(&mut input)?;
    }

    Ok(())
}
