//! # Tetris Experiment - Voting Distribution Across Positions
//!
//! Measures expert agreement across MANY different random board positions. For each of 2,048
//! different starting positions, runs 256 beam searches and calculates entropy/agreement statistics.
//!
//! ## Purpose
//!
//! Understands the distribution of expert agreement levels across the state space. Some positions
//! may have clear best moves (high agreement) while others are ambiguous (low agreement).
//!
//! ## Methodology
//!
//! 1. For each of 2,048 simulations:
//!    - Start from a random position (after warmup)
//!    - Run 256 beam searches with different seeds
//!    - Calculate normalized entropy and effective number of choices
//! 2. Analyze distribution of agreement levels
//!
//! ## Output
//!
//! - **CSV Files**:
//!   - `tetris_beam_multisearch_entropies.csv` - Entropy values per position
//!   - `tetris_beam_multisearch_effective_choices.csv` - Effective choice counts
//! - **Console**: Statistics on mean/variance of agreement, interpretation of results

use rand::RngCore;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tetris_game::{
    IsLost, PlacementResult, TetrisBoard, TetrisGame, TetrisGameRng, TetrisPieceOrientation,
    TetrisPiecePlacement,
};
use tetris_ml::beam_search::BeamTetrisState;
use tetris_ml::{BeamSearch, ScoredState, set_global_threadpool};

/*
python3 -c "import matplotlib.pyplot as plt; import numpy as np; data = [line.strip().split(',') for line in open('/Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas/beam_search_output.csv') if line.strip() and ',' in line]; data = [(int(row[0]), int(row[1])) for row in data if len(row) >= 2]; x, y = zip(*data); x, y = np.array(x), np.array(y); ratios = y / x; diffs_y = np.diff(y); diffs_x = np.diff(x); rates = diffs_y / diffs_x; fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12)); ax1.plot(x, ratios, 'b-', linewidth=2); ax1.set_xlabel('Number of Pieces Placed'); ax1.set_ylabel('Ratio (Unique Boards / Pieces)'); ax1.set_title('Board Uniqueness Ratio Over Pieces Placed'); ax1.grid(True, alpha=0.3); ax1.text(0.95, 0.05, f'{ratios[-1]:.4f}', transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), ha='right'); ax2.plot(x, y, 'g-', linewidth=3, label='Unique Boards'); ax2.plot(x, x, 'r-', linewidth=3, label='Total Pieces'); ax2.set_xlabel('Number of Pieces Placed'); ax2.set_ylabel('Count'); ax2.set_title('Unique Boards vs Total Pieces Placed'); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.text(0.95, 0.05, f'{y[-1]:,}', transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), ha='right'); ax3.plot(x[1:], rates, 'm-', linewidth=1, marker='o', markersize=2); ax3.set_xlabel('Number of Pieces Placed'); ax3.set_ylabel('Discovery Rate (New Unique / Pieces per Interval)'); ax3.set_title('Rate of New Unique Board Discovery'); ax3.grid(True, alpha=0.3); ax3.text(0.95, 0.05, f'{rates[-1]:.4f}', transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8), ha='right'); plt.tight_layout(); plt.show()"
 */

pub struct TetrisGameIter<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize> {
    pub game: TetrisGame,
    search: BeamSearch<BeamTetrisState, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>,
}

impl<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize>
    TetrisGameIter<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    pub fn new() -> Self {
        Self {
            game: TetrisGame::new(),
            search: BeamSearch::<BeamTetrisState, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new(),
        }
    }

    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            game: TetrisGame::new_with_seed(seed),
            search: BeamSearch::<BeamTetrisState, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new(),
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        self.game.reset(seed);
    }

    pub fn kink(&mut self) -> PlacementResult {
        self.game.play_random()
    }
}

impl<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize> Iterator
    for TetrisGameIter<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    type Item = (TetrisBoard, TetrisPiecePlacement);

    fn next(&mut self) -> Option<Self::Item> {
        if self.game.board.is_lost() {
            return None;
        }
        let board_before = self.game.board;
        let ScoredState { root_action: first_action, .. } = self
            .search
            .search_top_with_state(BeamTetrisState::new(self.game), MAX_DEPTH)?;
        (self.game.apply_placement(first_action.unwrap()).is_lost != IsLost::LOST)
            .then_some((board_before, first_action.unwrap()))
    }
}

fn main() {
    run_tetris_beam_multisearch();
}

fn run_simulation() -> (f64, usize, f64, f64) {
    let start_time = Instant::now();
    let mut rng = rand::rng();

    let mut base_agent =
        TetrisGameIter::<16, 16, { TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS }>::new_with_seed(
            rng.next_u64(),
        );
    for _ in 0..1000 {
        let _ = base_agent.next();
    }
    // continue until tetris bag has 1 piece left
    while base_agent.game.bag.remaining > 1 {
        let _ = base_agent.next();
    }

    let mut starting_state = base_agent.game.clone();

    // if we have "many" clairvoyant agents, how often do they disagree?
    let mut base_rng = rand::rng();
    let mut counts: HashMap<TetrisPiecePlacement, usize> = HashMap::new();
    for _ in 0..256 {
        let base_seed = base_rng.next_u64();
        starting_state.rng = TetrisGameRng::new(base_seed);

        const WIDTH: usize = 4;
        const DEPTH: usize = 4;
        let mut search = BeamSearch::<
            BeamTetrisState,
            WIDTH,
            DEPTH,
            { TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS },
        >::new();
        let first_action = search
            .search_top_with_state(BeamTetrisState::new(starting_state), DEPTH)
            .unwrap()
            .root_action
            .unwrap();
        counts
            .entry(first_action)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    // Calculate normalized entropy (0.0 = perfect agreement, 1.0 = uniform distribution)
    let total_votes = counts.values().sum::<usize>() as f64;
    let num_unique_actions = counts.len();

    // Shannon entropy
    let entropy: f64 = counts
        .values()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total_votes;
            -p * p.ln()
        })
        .sum();

    // Normalize by maximum possible entropy (uniform distribution)
    let max_entropy = (num_unique_actions as f64).ln();
    let normalized_entropy = if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0 // If only one action, perfect agreement
    };

    // Calculate effective number of choices: how many actions are "really" being considered
    // This is more intuitive than entropy: 1.0 = strong agreement, higher = more disagreement
    let effective_num_choices = entropy.exp();

    let elapsed = start_time.elapsed().as_secs_f64();

    (
        normalized_entropy,
        num_unique_actions,
        effective_num_choices,
        elapsed,
    )
}

/// Run a multi-beam-search-driven Tetris game with voting across multiple seeds.
///
/// This is intentionally argument-free: tweak the consts below while iterating.
pub fn run_tetris_beam_multisearch() {
    set_global_threadpool();

    const NUM_SIMULATIONS: usize = 2_048;
    let overall_start = Instant::now();

    let num_threads = rayon::current_num_threads();
    println!(
        "Starting {} simulations to measure expert disagreement...",
        NUM_SIMULATIONS
    );
    println!("Each simulation: 1000+ warmup steps, then 1024 expert votes");
    println!("Running in parallel with {} threads\n", num_threads);

    let completed = Arc::new(AtomicUsize::new(0));
    let completed_clone = Arc::clone(&completed);

    // Spawn a thread to print progress updates
    let progress_handle = std::thread::spawn(move || {
        let start = Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let count = completed_clone.load(Ordering::Relaxed);
            if count >= NUM_SIMULATIONS {
                break;
            }
            let elapsed = start.elapsed().as_secs_f64();
            let avg_time = if count > 0 {
                elapsed / count as f64
            } else {
                0.0
            };
            let remaining = (NUM_SIMULATIONS - count) as f64 * avg_time;
            println!(
                "[Progress: {}/{} ({:.1}%)] Elapsed: {:.1}s, ETA: {:.1}s",
                count,
                NUM_SIMULATIONS,
                (count as f64 / NUM_SIMULATIONS as f64) * 100.0,
                elapsed,
                remaining
            );
        }
    });

    // Run simulations in parallel
    let results: Vec<(f64, usize, f64, f64)> = (0..NUM_SIMULATIONS)
        .into_par_iter()
        .map(|_| {
            let result = run_simulation();
            completed.fetch_add(1, Ordering::Relaxed);
            result
        })
        .collect();

    // Wait for progress thread to finish
    let _ = progress_handle.join();

    let entropies: Vec<f64> = results.iter().map(|(entropy, _, _, _)| *entropy).collect();
    let effective_choices: Vec<f64> = results.iter().map(|(_, _, eff, _)| *eff).collect();

    // Write entropies to "tetris_beam_multisearch_entropies.csv"
    if let Ok(mut file) = std::fs::File::create("tetris_beam_multisearch_entropies.csv") {
        use std::io::Write;
        for val in &entropies {
            let _ = writeln!(file, "{}", val);
        }
    } else {
        eprintln!("Failed to create entropies output csv");
    }

    // Write effective choices to "tetris_beam_multisearch_effective_choices.csv"
    if let Ok(mut file) = std::fs::File::create("tetris_beam_multisearch_effective_choices.csv") {
        use std::io::Write;
        for val in &effective_choices {
            let _ = writeln!(file, "{}", val);
        }
    } else {
        eprintln!("Failed to create effective choices output csv");
    }

    let mean_entropy = entropies.iter().sum::<f64>() / NUM_SIMULATIONS as f64;
    let variance = entropies
        .iter()
        .map(|&e| (e - mean_entropy).powi(2))
        .sum::<f64>()
        / NUM_SIMULATIONS as f64;
    let std_dev = variance.sqrt();

    let total_elapsed = overall_start.elapsed();
    let avg_sim_time: f64 =
        results.iter().map(|(_, _, _, t)| t).sum::<f64>() / NUM_SIMULATIONS as f64;

    println!("\n{}", "=".repeat(60));
    println!("SIMULATION COMPLETE");
    println!("{}", "=".repeat(60));
    println!("Total simulations: {}", NUM_SIMULATIONS);
    println!(
        "Wall time: {:.2}s ({:.2}s per sim average)",
        total_elapsed.as_secs_f64(),
        avg_sim_time
    );
    println!(
        "Speedup: {:.2}x (parallel efficiency: {:.1}%)",
        avg_sim_time * NUM_SIMULATIONS as f64 / total_elapsed.as_secs_f64(),
        (avg_sim_time * NUM_SIMULATIONS as f64 / total_elapsed.as_secs_f64()) / num_threads as f64
            * 100.0
    );
    println!("\nExpert Disagreement Statistics:");

    // Effective number of choices (most intuitive metric)
    let mean_eff_choices = effective_choices.iter().sum::<f64>() / NUM_SIMULATIONS as f64;
    let min_eff_choices = effective_choices
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_eff_choices = effective_choices
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("\n  🎯 Effective Number of Choices:");
    println!("     Mean: {:.2} actions", mean_eff_choices);
    println!(
        "     Range: [{:.2}, {:.2}]",
        min_eff_choices, max_eff_choices
    );

    // Interpret the effective choices
    let interpretation = if mean_eff_choices < 2.0 {
        "Strong agreement - experts mostly agree on best action"
    } else if mean_eff_choices < 4.0 {
        "Moderate agreement - split between 2-4 viable strategies"
    } else if mean_eff_choices < 8.0 {
        "Weak agreement - many competing strategies"
    } else {
        "Very weak agreement - nearly random/highly uncertain"
    };
    println!("     Interpretation: {}", interpretation);

    // Technical entropy metrics
    println!("\n  📊 Normalized Entropy (technical):");
    println!(
        "     Mean: {:.3} (0.0=perfect agreement, 1.0=uniform)",
        mean_entropy
    );
    println!("     Std dev: {:.3}", std_dev);
    println!(
        "     Range: [{:.3}, {:.3}]",
        entropies.iter().cloned().fold(f64::INFINITY, f64::min),
        entropies.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    let avg_unique_actions: f64 =
        results.iter().map(|(_, u, _, _)| *u as f64).sum::<f64>() / NUM_SIMULATIONS as f64;
    println!("\n  📈 Other Statistics:");
    println!("     Avg unique actions per sim: {:.1}", avg_unique_actions);

    // Convert average entropy to "agreement percentage" for easier interpretation
    let avg_agreement = (1.0 - mean_entropy) * 100.0;
    println!("     Average agreement: {:.1}%", avg_agreement);

    println!("\nResults written to:");
    println!("  - tetris_beam_multisearch_entropies.csv");
    println!("  - tetris_beam_multisearch_effective_choices.csv");
}
