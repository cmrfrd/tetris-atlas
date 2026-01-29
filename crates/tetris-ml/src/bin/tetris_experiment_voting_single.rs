//! # Tetris Experiment - Voting from Single Position
//!
//! Measures how beam search "experts" vote when starting from the SAME board position.
//! This experiments with N=100,000 parallel searches from one specific state to analyze
//! the voting distribution and measure expert consensus.
//!
//! ## Purpose
//!
//! Tests whether multiple beam searches with different random seeds agree on the best move
//! from a single position, helping understand decision stability and confidence.
//!
//! ## Methodology
//!
//! 1. Warm up a game for 1000+ pieces to reach a complex state
//! 2. Run 100,000 beam searches from that exact position with different seeds
//! 3. Count how many times each placement is chosen
//! 4. Analyze voting distribution
//!
//! ## Output
//!
//! - **Console**: Sorted list of piece placements and their vote counts
//! - Shows whether experts strongly agree or are divided

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

    const NUM_SIMULATIONS: usize = 100_000;
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
    let starting_state = base_agent.game.clone();
    println!("{}", starting_state.board);
    println!("{}", starting_state.bag);
    println!("{}", starting_state.current_piece);

    // Run simulations in parallel
    let results: HashMap<TetrisPiecePlacement, usize> = (0..NUM_SIMULATIONS)
        .into_par_iter()
        .map(|_| {
            let new_base_seed = rand::rng().next_u64();
            let mut starting_state = starting_state.clone();
            starting_state.rng = TetrisGameRng::new(new_base_seed);

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

            completed.fetch_add(1, Ordering::Relaxed);
            first_action
        })
        .fold(
            || HashMap::<TetrisPiecePlacement, usize>::new(),
            |mut acc, action| {
                *acc.entry(action).or_insert(0) += 1;
                acc
            },
        )
        .reduce(
            || HashMap::new(),
            |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_insert(0) += v;
                }
                a
            },
        );

    // Wait for progress thread to finish
    let _ = progress_handle.join();

    // Print the sorted list from the results HashMap
    let mut sorted: Vec<_> = results.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1)); // Sort descending by count
    println!("Sorted Piece Placements and Counts:");
    for (placement, count) in sorted {
        println!("{:>4}: {}", placement.orientation, count);
    }
}
