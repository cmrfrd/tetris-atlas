//! # Tetris Training - Genetic Algorithm
//!
//! Evolves optimal heuristic coefficients for Tetris board evaluation using a genetic algorithm.
//! The goal is to find coefficient values that maximize survival (pieces placed) while
//! minimizing board height.
//!
//! ## Purpose
//!
//! Automatically discovers good heuristic weights for evaluating board states, replacing
//! hand-tuned parameters with evolved ones optimized for long-term gameplay.
//!
//! ## Methodology
//!
//! 1. **Population**: 64 individuals, each with random initial coefficients:
//!    - lines_cleared, height, bumpiness, weighted_holes
//! 2. **Fitness**: Play games up to 10k pieces, fitness = pieces_placed - height_penalty
//! 3. **Evolution**: Tournament selection, crossover, mutation (30% rate)
//! 4. **Elitism**: Keep top 8 individuals unchanged each generation
//! 5. Run for 1,000,000 generations in parallel using rayon
//!
//! ## Output
//!
//! - **CSV File**: `beam_genetic_output.csv` - Generation stats and best coefficients
//! - **Console**: Real-time progress with best fitness and coefficient values
//! - Final best coefficients printed at completion

use proc_macros::inline_conditioned;
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;
use tetris_game::utils::HeaplessVec;
use tetris_game::{IsLost, TetrisBoard, TetrisGame, TetrisGameRng, TetrisPieceOrientation, TetrisPiecePlacement};
use tetris_ml::beam_search::{BeamSearch, BeamSearchState};
use tetris_ml::set_global_threadpool;

/// Output file path for genetic algorithm results
const OUTPUT_FILE: &str = "beam_genetic_output.csv";

/// Coefficients for genetic algorithm evaluation
#[derive(Clone, Copy, Debug)]
pub struct EvalCoefficients {
    pub lines_cleared: f32,
    pub height: f32,
    pub bumpiness: f32,
    pub weighted_holes: f32,
}

impl EvalCoefficients {
    /// Create random coefficients for initial population
    pub fn random<R: rand::Rng>(rng: &mut R) -> Self {
        Self {
            lines_cleared: rng.random_range(-1.0..1.0),
            height: rng.random_range(-1.0..1.0),
            bumpiness: rng.random_range(-1.0..1.0),
            weighted_holes: rng.random_range(-1.0..1.0),
        }
    }

    /// Mutate coefficients slightly
    pub fn mutate<R: rand::Rng>(&mut self, rng: &mut R, mutation_rate: f32) {
        if rng.random::<f32>() < mutation_rate {
            self.lines_cleared += rng.random_range(-0.01..0.01);
            self.lines_cleared = self.lines_cleared.clamp(-1.0, 1.0);
        }
        if rng.random::<f32>() < mutation_rate {
            self.height += rng.random_range(-0.01..0.01);
            self.height = self.height.clamp(-1.0, 1.0);
        }
        if rng.random::<f32>() < mutation_rate {
            self.bumpiness += rng.random_range(-0.01..0.01);
            self.bumpiness = self.bumpiness.clamp(-1.0, 1.0);
        }
        if rng.random::<f32>() < mutation_rate {
            self.weighted_holes += rng.random_range(-0.01..0.01);
            self.weighted_holes = self.weighted_holes.clamp(-1.0, 1.0);
        }
    }

    /// Crossover two coefficient sets
    pub fn crossover<R: rand::Rng>(parent1: &Self, parent2: &Self, rng: &mut R) -> Self {
        Self {
            lines_cleared: if rng.random_bool(0.5) {
                parent1.lines_cleared
            } else {
                parent2.lines_cleared
            },
            height: if rng.random_bool(0.5) {
                parent1.height
            } else {
                parent2.height
            },
            bumpiness: if rng.random_bool(0.5) {
                parent1.bumpiness
            } else {
                parent2.bumpiness
            },
            weighted_holes: if rng.random_bool(0.5) {
                parent1.weighted_holes
            } else {
                parent2.weighted_holes
            },
        }
    }
}

/// Tetris state with evolved evaluation coefficients for genetic algorithms
#[derive(Clone, Copy)]
pub struct BeamTetrisStateGenetic {
    game: TetrisGame,
    coefficients: EvalCoefficients,
}

impl BeamTetrisStateGenetic {
    pub fn new(game: TetrisGame, coefficients: EvalCoefficients) -> Self {
        Self { game, coefficients }
    }

    /// Evolved heuristic using genetic algorithm coefficients
    #[inline_conditioned(always)]
    pub fn score(&self) -> f32 {
        if self.game.board.is_lost() {
            return f32::NEG_INFINITY;
        }

        let lines = self.game.lines_cleared as f32;
        let height = self.game.board.height() as f32;

        // Calculate bumpiness (sum of absolute differences between adjacent column heights)
        let heights = self.game.board.heights();
        let mut bumpiness = 0.0;
        for i in 0..heights.len() - 1 {
            bumpiness += (heights[i] as f32 - heights[i + 1] as f32).abs();
        }

        // Calculate weighted holes (holes weighted by their depth)
        // Deeper holes are much worse than shallow ones
        let holes_per_column = self.game.board.holes();
        let mut weighted_holes = 0.0;
        for col in 0..10 {
            // Weight each hole by the height of blocks above it
            weighted_holes += holes_per_column[col] as f32 * heights[col] as f32;
        }

        self.coefficients.lines_cleared * lines
            + self.coefficients.height * height
            + self.coefficients.bumpiness * bumpiness
            + self.coefficients.weighted_holes * weighted_holes
    }
}

impl PartialEq for BeamTetrisStateGenetic {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.score().total_cmp(&other.score()).is_eq()
    }
}

impl Eq for BeamTetrisStateGenetic {}

impl PartialOrd for BeamTetrisStateGenetic {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BeamTetrisStateGenetic {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.score().total_cmp(&other.score())
    }
}

impl BeamSearchState for BeamTetrisStateGenetic {
    type Action = TetrisPiecePlacement;

    #[inline_conditioned(always)]
    fn apply_action(&self, action: &Self::Action) -> Self {
        let mut g = self.game;
        let _res = g.apply_placement(*action);
        Self {
            game: g,
            coefficients: self.coefficients,
        }
    }

    #[inline_conditioned(always)]
    fn generate_actions<const M: usize>(&self, buffer: &mut HeaplessVec<Self::Action, M>) -> usize {
        if self.game.board.is_lost() {
            return 0;
        }

        buffer.clear();
        let placements = self.game.current_placements();
        let n = placements.len();
        debug_assert!(
            n <= M,
            "MAX_MOVES too small: need at least {}, have {}",
            n,
            M
        );

        buffer.fill_from_slice(placements);
        n
    }
}

pub struct TetrisGameIterGenetic<
    const N: usize,
    const BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
> {
    pub game: TetrisGame,
    /// N parallel beam searches for voting
    searches: Vec<BeamSearch<BeamTetrisStateGenetic, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>>,
    pub coefficients: EvalCoefficients,
    step_counter: u64,
}

impl<const N: usize, const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize>
    TetrisGameIterGenetic<N, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    pub fn new_with_coefficients(coefficients: EvalCoefficients) -> Self {
        assert!(N > 0, "N must be greater than 0");
        let mut searches = Vec::with_capacity(N);
        for _ in 0..N {
            searches.push(BeamSearch::new());
        }
        Self {
            game: TetrisGame::new(),
            searches,
            coefficients,
            step_counter: 0,
        }
    }

    pub fn new_with_seed_and_coefficients(seed: u64, coefficients: EvalCoefficients) -> Self {
        assert!(N > 0, "N must be greater than 0");
        let mut searches = Vec::with_capacity(N);
        for _ in 0..N {
            searches.push(BeamSearch::new());
        }
        Self {
            game: TetrisGame::new_with_seed(seed),
            searches,
            coefficients,
            step_counter: seed,
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        self.game.reset(seed);
        if let Some(s) = seed {
            self.step_counter = s;
        }
    }
}

impl<const N: usize, const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize> Iterator
    for TetrisGameIterGenetic<N, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    type Item = (TetrisBoard, TetrisPiecePlacement);

    fn next(&mut self) -> Option<Self::Item> {
        if self.game.board.is_lost() {
            return None;
        }
        let board_before = self.game.board;

        // Run N searches with different seeds and vote on the best action
        let action_votes: HashMap<TetrisPiecePlacement, (usize, f32)> = self.searches
            .par_iter_mut()
            .enumerate()
            .filter_map(|(i, search)| {
                // Create state variant with different RNG seed
                let mut game = self.game;
                game.rng = TetrisGameRng::new(self.step_counter + i as u64);
                let state = BeamTetrisStateGenetic::new(game, self.coefficients);

                // Run search and extract first action + score
                search
                    .search_top_with_state(state, MAX_DEPTH)
                    .and_then(|scored| scored.root_action.map(|action| (action, scored.state.score())))
            })
            .fold(
                HashMap::new,
                |mut acc: HashMap<TetrisPiecePlacement, (usize, f32)>, (action, score)| {
                    let entry = acc.entry(action).or_insert((0, 0.0));
                    entry.0 += 1; // Count votes
                    entry.1 += score; // Sum scores
                    acc
                },
            )
            .reduce(
                HashMap::new,
                |mut a, b| {
                    for (action, (count, score)) in b {
                        let entry = a.entry(action).or_insert((0, 0.0));
                        entry.0 += count;
                        entry.1 += score;
                    }
                    a
                },
            );

        // Pick the action with the most votes, breaking ties by total score
        let first_action = action_votes
            .into_iter()
            .max_by(|(_, (count_a, score_a)), (_, (count_b, score_b))| {
                count_a
                    .cmp(count_b)
                    .then_with(|| score_a.total_cmp(score_b))
            })
            .map(|(action, _)| action)?;

        self.step_counter += 1;

        (self.game.apply_placement(first_action).is_lost != IsLost::LOST)
            .then_some((board_before, first_action))
    }
}

/// Individual in the genetic algorithm population
#[derive(Clone)]
struct Individual {
    coefficients: EvalCoefficients,
    fitness: f32,
    pieces_placed: usize,
    avg_height: f32,
}

impl Individual {
    fn new(coefficients: EvalCoefficients) -> Self {
        Self {
            coefficients,
            fitness: 0.0,
            pieces_placed: 0,
            avg_height: 0.0,
        }
    }

    /// Evaluate fitness by playing a game with these coefficients
    fn evaluate_fitness<const N: usize, const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize>(
        &mut self,
        seed: u64,
        max_pieces: usize,
    ) {
        let mut iter =
            TetrisGameIterGenetic::<N, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new_with_seed_and_coefficients(
                seed,
                self.coefficients,
            );

        let mut pieces_placed = 0;
        let mut cumulative_height_sum = 0.0;

        while pieces_placed < max_pieces {
            if iter.next().is_none() {
                break;
            }
            pieces_placed += 1;

            // Accumulate height statistics after each piece
            let heights = iter.game.board.heights();
            let step_avg_height = heights.iter().sum::<u32>() as f32 / heights.len() as f32;
            cumulative_height_sum += step_avg_height;
        }

        // Calculate average height across the entire game
        let avg_height = if pieces_placed > 0 {
            cumulative_height_sum / pieces_placed as f32
        } else {
            0.0
        };

        // Store stats
        self.pieces_placed = pieces_placed;
        self.avg_height = avg_height;

        // Fitness: First maximize pieces (normalized), then minimize height
        // pieces_component: 0.0 to 100.0 (scale up to emphasize survival)
        // height_penalty: exponential penalty that grows rapidly with height
        let pieces_component = (pieces_placed.min(max_pieces) as f32) / (max_pieces as f32) * 100.0;

        // Exponential penalty: scale * (exp(height/divisor) - 1)
        // This grows slowly at first, then rapidly for higher heights
        // height=1 -> ~1.6, height=2 -> ~4.4, height=3 -> ~10.1, height=4 -> ~21.6
        let height_penalty_scale = 10.0;
        let height_divisor = 2.0;
        let height_penalty = height_penalty_scale * ((avg_height / height_divisor).exp() - 1.0);

        self.fitness = pieces_component - height_penalty;
    }
}

/// Genetic algorithm for evolving Tetris evaluation coefficients
fn evolve_coefficients() {
    set_global_threadpool();

    // --- Tunables ---
    const N: usize = 4; // Number of parallel searches for multi-beam voting
    const BEAM_WIDTH: usize = 8;
    const MAX_DEPTH: usize = 8;
    const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;
    const POPULATION_SIZE: usize = 64;
    const GENERATIONS: usize = 1_000_000;
    const MUTATION_RATE: f32 = 0.3;
    const MAX_PIECES_PER_EVAL: usize = 10_000;
    const ELITE_COUNT: usize = 8; // Keep top individuals unchanged
    // --------------

    let mut rng = rand::rng();

    // Initialize population
    let mut population: Vec<Individual> = (0..POPULATION_SIZE)
        .map(|_| Individual::new(EvalCoefficients::random(&mut rng)))
        .collect();

    // Open output file
    let file = File::create(OUTPUT_FILE).expect("Failed to create output file");
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "generation,best_fitness,avg_fitness,pieces_placed,avg_height,lines_cleared,height,bumpiness,weighted_holes"
    )
    .expect("Failed to write header");

    let start = Instant::now();

    for generation in 0..GENERATIONS {
        let gen_start = Instant::now();

        // Generate seeds for all individuals upfront (for thread-safety)
        let seeds: Vec<u64> = (0..POPULATION_SIZE)
            .map(|_| generation as u64 * 1000 + rng.random::<u64>())
            .collect();

        // Evaluate fitness for all individuals in parallel
        population
            .par_iter_mut()
            .zip(seeds.par_iter())
            .for_each(|(individual, &seed)| {
                individual.evaluate_fitness::<N, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>(
                    seed,
                    MAX_PIECES_PER_EVAL,
                );
            });

        // Sort by fitness (descending)
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let best = &population[0];
        let best_fitness = best.fitness;
        let avg_fitness: f32 =
            population.iter().map(|i| i.fitness).sum::<f32>() / population.len() as f32;

        // Log results
        println!(
            "Generation {}: best_fitness={:.4}, avg_fitness={:.4}, time={:.2}s",
            generation,
            best_fitness,
            avg_fitness,
            gen_start.elapsed().as_secs_f64()
        );

        let pieces_comp = (best.pieces_placed.min(MAX_PIECES_PER_EVAL) as f32)
            / (MAX_PIECES_PER_EVAL as f32)
            * 100.0;
        let height_penalty_scale = 10.0;
        let height_divisor = 2.0;
        let height_penalty =
            height_penalty_scale * ((best.avg_height / height_divisor).exp() - 1.0);

        println!(
            "  Best: pieces={}, avg_height={:.6}, fitness_breakdown=(pieces_comp={:.2}, height_penalty={:.2})",
            best.pieces_placed, best.avg_height, pieces_comp, height_penalty
        );
        println!("  Best coefficients: {:?}", best.coefficients);

        writeln!(
            writer,
            "{},{:.4},{:.4},{},{:.6},{:.4},{:.4},{:.4},{:.4}",
            generation,
            best_fitness,
            avg_fitness,
            best.pieces_placed,
            best.avg_height,
            best.coefficients.lines_cleared,
            best.coefficients.height,
            best.coefficients.bumpiness,
            best.coefficients.weighted_holes
        )
        .expect("Failed to write to output file");

        if generation % 10 == 0 {
            writer.flush().expect("Failed to flush output file");
        }

        // Create next generation
        let mut next_generation = Vec::with_capacity(POPULATION_SIZE);

        // Elitism: keep top individuals
        for i in 0..ELITE_COUNT {
            next_generation.push(population[i].clone());
        }

        // Fill rest with crossover and mutation
        while next_generation.len() < POPULATION_SIZE {
            // Tournament selection
            let parent1_idx = (0..5)
                .map(|_| rng.random_range(0..POPULATION_SIZE))
                .min_by(|&a, &b| {
                    population[b]
                        .fitness
                        .partial_cmp(&population[a].fitness)
                        .unwrap()
                })
                .unwrap();
            let parent2_idx = (0..5)
                .map(|_| rng.random_range(0..POPULATION_SIZE))
                .min_by(|&a, &b| {
                    population[b]
                        .fitness
                        .partial_cmp(&population[a].fitness)
                        .unwrap()
                })
                .unwrap();

            // Crossover
            let mut child_coeffs = EvalCoefficients::crossover(
                &population[parent1_idx].coefficients,
                &population[parent2_idx].coefficients,
                &mut rng,
            );

            // Mutation
            child_coeffs.mutate(&mut rng, MUTATION_RATE);

            next_generation.push(Individual::new(child_coeffs));
        }

        population = next_generation;
    }

    writer.flush().expect("Failed to flush output file");
    println!(
        "Evolution complete! Total time: {:.2}s",
        start.elapsed().as_secs_f64()
    );
    println!("Best final coefficients: {:?}", population[0].coefficients);
}

fn main() {
    evolve_coefficients();
}
