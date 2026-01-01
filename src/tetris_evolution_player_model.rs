use std::path::PathBuf;

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarMap;
use candle_nn::{VarBuilder, init::Init};
use rand::Rng;
use rand::RngCore;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use serde::{Deserialize, Serialize};
use tensorboard::summary_writer::SummaryWriter;

use crate::ops::create_orientation_mask;
use crate::tensors::{
    TetrisBoardsTensor, TetrisPieceOneHotTensor, TetrisPieceOrientationLogitsTensor,
    TetrisPieceTensor,
};
use crate::tetris::{TetrisBoard, TetrisGameSet, TetrisPiece, TetrisPieceOrientation};
use crate::wrapped_tensor::WrappedTensor;
use crate::{device, fdtype};

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

        let expected_output = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;
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
        if batch != self.population_size {
            anyhow::bail!(
                "forward_logits expects batch == population_size (one game per individual). got batch={}, population_size={}",
                batch,
                self.population_size
            );
        }
        if current_piece.shape_tuple().0 != self.population_size {
            anyhow::bail!(
                "forward_logits expects current_piece batch == population_size. got {:?}, population_size={}",
                current_piece.shape_tuple(),
                self.population_size
            );
        }

        let board_flat = current_board
            .to_dtype(crate::fdtype())?
            .reshape(&[batch, TetrisBoard::HEIGHT * TetrisBoard::WIDTH])?;
        let piece_one_hot = TetrisPieceOneHotTensor::from_piece_tensor(current_piece.clone())?;
        let input = Tensor::cat(&[&board_flat, &piece_one_hot], 1)?; // [B, input_dim]

        // Input shape: [population, in_dim] (one row per individual)
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
        let mask = create_orientation_mask(current_piece)?; // [B, O] (0/1)
        let device = logits.device();
        let dtype = logits.dtype();
        let keep = mask.gt(0u32)?;
        // NOTE: Metal backend does not support const-set for f64.
        // Use f32 and cast to the logits dtype (often f16/f32).
        let neg_inf = Tensor::full(f32::NEG_INFINITY, logits.dims(), &device)?
            .to_dtype(dtype)?
            .broadcast_as(logits.dims())?;
        let masked = keep.where_cond(logits.inner(), &neg_inf)?;
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

        // NOTE: Keep this sequential. Dispatching many small CUDA ops from multiple host threads
        // is often slower and can be unstable depending on the backend.
        for param in self.weights.iter().chain(self.biases.iter()) {
            let dims = param.dims();
            debug_assert!(dims[0] == self.population_size);

            let output_dim: usize = dims[1..].iter().product();

            let noise = Self::apply_norm(
                // NOTE: Metal backend does not support randn for f64. Use f32 and cast.
                Tensor::randn(0f32, 1f32, (num_lost, output_dim), &device)?
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
        }

        Ok(())
    }

    fn apply_norm(tensor: Tensor, norm: Option<f64>) -> Result<Tensor> {
        if let Some(target_norm) = norm {
            if target_norm <= 0.0 {
                anyhow::bail!("perturb_norm must be positive");
            }

            // Normalize *per individual* (per row), so each mutated individual's noise has the
            // same L2 norm in the flattened parameter space.
            //
            // tensor is expected to be [num_lost, output_dim] here.
            if tensor.dims().len() != 2 {
                anyhow::bail!(
                    "apply_norm expects a 2D tensor [num_lost, output_dim], got dims={:?}",
                    tensor.dims()
                );
            }
            let dims = tensor.dims();
            let (num_lost, _output_dim) = tensor.dims2()?;

            // norms: [num_lost, 1]
            let norms = tensor.sqr()?.sum_keepdim(1)?.sqrt()?;
            let eps = Tensor::full(f32::EPSILON, norms.dims(), tensor.device())?
                .to_dtype(tensor.dtype())?;
            let denom = norms.maximum(&eps)?;

            let scaled = tensor
                .broadcast_div(&denom)?
                .affine(target_norm as f64, 0.0)?
                .reshape((num_lost, dims[1]))?;
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
    let device = device();
    let dtype = fdtype();

    const NUM_GENERATIONS: usize = 100_000;
    const POPULATION_SIZE: usize = 1024;
    const SEED: usize = 42;
    // When true, all individuals are evaluated on the exact same game seed.
    // That means every reset restarts the identical piece sequence from the same start state.
    // This makes fitness comparisons extremely low-variance, but will heavily overfit to that one
    // deterministic "game".
    const USE_SHARED_GAME_SEEDS: bool = true;
    const NOISE_SCALE: f64 = 0.002; // IMPROVEMENT: Increased from 0.0001 for better exploration
    const N_GAMES_FOR_FITNESS: usize = 256; // Average fitness over N games per individual
    const MIN_READY_FOR_EVOLUTION: usize = 32; // Minimum individuals needed to trigger evolution
    const SAMPLING_TEMPERATURE: f32 = 0.5; // IMPROVEMENT: Increased from 0.1 for more exploration during training

    #[inline]
    fn eval_game_seed(base_seed: u64, _game_number: usize) -> u64 {
        // "Literally the exact same seed": ignore game number and always return base_seed.
        base_seed
    }

    let model_varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&model_varmap, dtype, &device);

    let population_cfg = TetrisEvolutionPopulationConfig {
        layer_dims: vec![
            TetrisBoard::SIZE + TetrisPiece::NUM_PIECES,
            64,
            64,
            64,
            64,
            TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS,
        ],
        population_size: POPULATION_SIZE,
    };
    let mut population = TetrisEvolutionPopulation::init(&vb, &population_cfg)?;

    let mut summary_writer = logdir.map(|dir| {
        let _ = std::fs::create_dir_all(&dir);
        SummaryWriter::new(dir)
    });

    let shared_seed_base = SEED as u64;
    let mut game_set = if USE_SHARED_GAME_SEEDS {
        TetrisGameSet::new_with_same_seed(eval_game_seed(shared_seed_base, 0), POPULATION_SIZE)
    } else {
        TetrisGameSet::new_with_seed(shared_seed_base, POPULATION_SIZE)
    };
    let mut rng = rand::rng();

    // Track cumulative pieces and game counts for averaging fitness over N games
    let mut cumulative_pieces: Vec<u64> = vec![0; POPULATION_SIZE];
    let mut games_completed: Vec<usize> = vec![0; POPULATION_SIZE];

    let mut max_num_pieces = 0;
    let mut total_steps = 0usize;
    let mut last_log_time = std::time::Instant::now();
    let mut last_log_steps = 0usize;

    for generation in 0..NUM_GENERATIONS {
        let generation_start_time = std::time::Instant::now();
        let generation_start_steps = total_steps;

        // Inner loop: play until enough individuals complete N games for evolution
        let (ready_for_eval, still_running, avg_entropy) = loop {
            // Forward pass for all individuals
            let current_board = TetrisBoardsTensor::from_gameset(&game_set, &device)?;
            let current_pieces_array: Vec<TetrisPiece> = game_set.current_pieces().to_vec();
            let current_pieces_vec: Vec<_> = current_pieces_array[0..game_set.len()].to_vec();
            let current_piece =
                TetrisPieceTensor::from_pieces(&current_pieces_vec.as_slice(), &device)?;
            let masked = population.forward_masked(&current_board, &current_piece)?;
            let entropy_mean = masked.into_dist()?.entropy()?.mean_all()?;

            // Sample actions and apply (using temperature for exploration)
            let sampled_orientations = masked.sample(SAMPLING_TEMPERATURE, &current_piece)?;
            let action_orientations = sampled_orientations.into_orientations()?;
            let lost_flags = game_set.apply_placement_from_orientations(&action_orientations);
            let piece_counts_array: Vec<u32> = game_set.piece_counts().to_vec();
            let piece_counts: Vec<_> = piece_counts_array[0..game_set.len()].to_vec();

            // Update max pieces
            let step_max_pieces = piece_counts.iter().copied().max().unwrap_or(0);
            max_num_pieces = std::cmp::max(max_num_pieces, step_max_pieces);

            // Update cumulative stats for individuals that lost a game
            for (idx, (is_lost, &piece_count)) in lost_flags.iter().zip(&piece_counts).enumerate() {
                if bool::from(*is_lost) {
                    cumulative_pieces[idx] += piece_count as u64;
                    games_completed[idx] += 1;
                    let next_seed = if USE_SHARED_GAME_SEEDS {
                        eval_game_seed(shared_seed_base, games_completed[idx])
                    } else {
                        rng.next_u64()
                    };
                    game_set[idx].reset(Some(next_seed));
                }
            }

            total_steps += 1;

            // Check which individuals are ready for evaluation
            let mut ready_for_eval: Vec<(usize, f64)> = Vec::with_capacity(POPULATION_SIZE);
            let mut still_running: Vec<usize> = Vec::with_capacity(POPULATION_SIZE);

            for idx in 0..POPULATION_SIZE {
                if games_completed[idx] >= N_GAMES_FOR_FITNESS {
                    let avg_pieces = cumulative_pieces[idx] as f64 / games_completed[idx] as f64;
                    ready_for_eval.push((idx, avg_pieces));
                } else {
                    still_running.push(idx);
                }
            }

            // Break when we have enough individuals ready for evolution
            if ready_for_eval.len() >= MIN_READY_FOR_EVOLUTION {
                break (ready_for_eval, still_running, entropy_mean);
            }
        };

        let num_ready = ready_for_eval.len();
        let steps_this_generation = total_steps - generation_start_steps;

        // Sort ready individuals by fitness (average pieces), descending
        let mut sorted_ready = ready_for_eval.clone();
        sorted_ready.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep the top half as survivors, replace the bottom half
        let num_survivors = num_ready / 2;
        let num_to_replace = num_ready - num_survivors;

        let survivors: Vec<(usize, f64)> = sorted_ready[..num_survivors].to_vec();

        // Compute fitness stats before evolution
        let avg_fitness: f64 =
            ready_for_eval.iter().map(|(_, avg)| *avg).sum::<f64>() / num_ready as f64;
        let best_fitness: f64 = sorted_ready[0].1;

        // Select new parents from survivors using fitness-weighted sampling
        // Using squared fitness for selection pressure while maintaining diversity
        let survivor_indices: Vec<usize> = survivors.iter().map(|(idx, _)| *idx).collect();
        let weights: Vec<f64> = survivors
            .iter()
            .map(|(_, avg_pieces)| avg_pieces.powf(2.).clamp(0.0, 100_000.0))
            .collect();
        let new_parents: Vec<usize> = match WeightedIndex::new(&weights) {
            Ok(weighted_index) => weighted_index
                .sample_iter(&mut rng)
                .take(num_to_replace)
                .map(|pos| survivor_indices[pos])
                .collect(),
            Err(_) => {
                // Early on it's common for all fitnesses (and thus weights) to be ~0.
                // In that case, fall back to uniform sampling over survivors.
                (0..num_to_replace)
                    .map(|_| {
                        let pos = rng.random_range(0..survivor_indices.len());
                        survivor_indices[pos]
                    })
                    .collect()
            }
        };

        // Build permutation: [still_running..., survivors..., new_parents...]
        // After permutation, the population structure will be:
        //   [0..still_running.len()]                                     = still_running individuals (continue current N-game cycle)
        //   [still_running.len()..still_running.len()+num_survivors]     = survivors (top 50%, start new N-game cycle)
        //   [POPULATION_SIZE-num_to_replace..POPULATION_SIZE]            = new_parents (mutated copies of survivors, start new N-game cycle)
        let permute_vec: Vec<usize> = still_running
            .iter()
            .copied()
            .chain(survivor_indices.iter().copied())
            .chain(new_parents.iter().copied())
            .collect();

        // === PERMUTATION PHASE ===
        // Permute game_set to match new population order
        game_set.permute(&permute_vec);

        // Permute the tracking arrays to match new population order
        let old_cumulative = cumulative_pieces.clone();
        let old_games = games_completed.clone();
        for (new_idx, &old_idx) in permute_vec.iter().enumerate() {
            cumulative_pieces[new_idx] = old_cumulative[old_idx];
            games_completed[new_idx] = old_games[old_idx];
        }

        // Permute population weights to match new order, adding noise to new_parents
        let index_vector: Vec<u32> = permute_vec.iter().map(|&idx| idx as u32).collect();
        population.permute_population(&index_vector, num_to_replace, Some(NOISE_SCALE))?;

        // === RESET PHASE ===
        // Reset survivors (indices: still_running.len() .. still_running.len()+num_survivors)
        // They completed N games and need to start a fresh N-game evaluation cycle
        for idx in still_running.len()..(still_running.len() + num_survivors) {
            cumulative_pieces[idx] = 0;
            games_completed[idx] = 0;
            let reset_seed = if USE_SHARED_GAME_SEEDS {
                eval_game_seed(shared_seed_base, 0)
            } else {
                rng.next_u64()
            };
            game_set[idx].reset(Some(reset_seed)); // BUG FIX: Reset games too!
        }

        // Reset new_parents (indices: POPULATION_SIZE-num_to_replace .. POPULATION_SIZE)
        // These are mutated copies of survivors that need fresh evaluation
        for idx in (POPULATION_SIZE - num_to_replace)..POPULATION_SIZE {
            cumulative_pieces[idx] = 0;
            games_completed[idx] = 0;
            let reset_seed = if USE_SHARED_GAME_SEEDS {
                eval_game_seed(shared_seed_base, 0)
            } else {
                rng.next_u64()
            };
            game_set[idx].reset(Some(reset_seed));
        }

        // Compute timing stats
        let generation_elapsed = generation_start_time.elapsed().as_secs_f64();
        let steps_per_second = if generation_elapsed > 0.0 {
            steps_this_generation as f64 / generation_elapsed
        } else {
            0.0
        };

        // Log to console
        if generation % 10 == 0 {
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(last_log_time).as_secs_f64();
            let delta_steps = total_steps.saturating_sub(last_log_steps);
            let overall_steps_per_sec = if elapsed > 0.0 {
                delta_steps as f64 / elapsed
            } else {
                0.0
            };
            last_log_time = now;
            last_log_steps = total_steps;

            println!(
                "Gen {:5} | Steps: {:8} | Best: {:6.1} | Avg: {:6.1} | Replaced: {:4} | Steps/Sec: {:8.1}",
                generation,
                total_steps,
                best_fitness,
                avg_fitness,
                num_to_replace,
                overall_steps_per_sec
            );
        }

        // Log to tensorboard
        if let Some(writer) = summary_writer.as_mut() {
            writer.add_scalar("generation/best_fitness", best_fitness as f32, generation);
            writer.add_scalar("generation/avg_fitness", avg_fitness as f32, generation);
            writer.add_scalar("generation/num_replaced", num_to_replace as f32, generation);
            writer.add_scalar("generation/num_survivors", num_survivors as f32, generation);
            writer.add_scalar(
                "generation/steps_this_gen",
                steps_this_generation as f32,
                generation,
            );
            writer.add_scalar(
                "generation/steps_per_second",
                steps_per_second as f32,
                generation,
            );
            writer.add_scalar("generation/total_steps", total_steps as f32, generation);
            writer.add_scalar(
                "generation/max_pieces_ever",
                max_num_pieces as f32,
                generation,
            );
            writer.add_scalar(
                "generation/avg_entropy",
                avg_entropy.to_scalar::<f32>()?,
                generation,
            );

            // Histogram of fitness values for ready individuals
            let fitness_values: Vec<f64> = ready_for_eval.iter().map(|(_, f)| *f).collect();
            writer.add_histogram(
                "generation/fitness_distribution",
                &fitness_values,
                None,
                generation,
            );
        }
    }

    Ok(())
}
