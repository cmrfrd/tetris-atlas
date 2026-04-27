use proc_macros::inline_conditioned;
use rayon::prelude::*;
use tetris_game::{
    TetrisBoard, TetrisGame, TetrisGameRng, TetrisPieceOrientation, TetrisPiecePlacement,
};
use tetris_utils::{FixedBinMinHeap, HeaplessVec, repeat_idx_unroll};

/// Counts of actions by orientation index
#[derive(Clone, Copy)]
pub struct OrientationCounts([u32; TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS]);

impl Default for OrientationCounts {
    fn default() -> Self {
        Self([0; TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS])
    }
}

impl OrientationCounts {
    #[inline_conditioned(always)]
    pub fn inner(&self) -> [u32; TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS] {
        self.0
    }

    #[inline_conditioned(always)]
    pub fn merge(&mut self, other: Self) -> &mut Self {
        repeat_idx_unroll!(TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS, I, {
            self.0[I] += other.0[I];
        });
        self
    }

    #[inline_conditioned(always)]
    pub fn add_action(&mut self, action: TetrisPiecePlacement) -> &mut Self {
        self.0[action.orientation.index() as usize] += 1;
        self
    }

    #[inline_conditioned(always)]
    pub fn top_orientation(&self) -> (TetrisPieceOrientation, u32) {
        let (max_idx, &max_count) = self
            .0
            .iter()
            .enumerate()
            .max_by_key(|&(_, &count)| count)
            .unwrap_or((0, &0));
        (TetrisPieceOrientation::from_index(max_idx as u8), max_count)
    }

    #[inline_conditioned(always)]
    pub fn nonzero_orientations(&self) -> impl Iterator<Item = (TetrisPieceOrientation, u32)> + '_ {
        self.0
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count > 0)
            .map(|(i, &count)| (TetrisPieceOrientation::from_index(i as u8), count))
    }
}

/// Trait for states that can be used in beam search.
///
/// Ranking is supplied by [`BeamSearch`] scoring callbacks rather than by the state type itself.
pub trait BeamSearchState: Copy {
    /// Type representing an action in the search space
    type Action: Copy + Default;

    /// Apply an action to this state, returning a new state
    fn apply_action(&self, action: &Self::Action) -> Self;

    /// Generate all valid actions from this state
    /// Writes actions into the provided buffer and returns the count.
    fn generate_actions<const MAX_ACTIONS: usize>(
        &self,
        buffer: &mut HeaplessVec<Self::Action, MAX_ACTIONS>,
    ) -> usize;
}

/// Lightweight scored state for heap operations (lives in next_beam)
#[derive(Clone, Copy)]
pub struct ScoredState<S: BeamSearchState> {
    pub state: S,
    pub action: S::Action,              // Action from parent to this state
    pub root_action: Option<S::Action>, // First action from root (None for root, Some for descendants)
    pub depth: usize,
    pub score: f32,
}

/// Controls how per-state scorer outputs are combined while expanding a beam.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BeamScoreMode {
    /// Rank each beam node by only that node's scorer value.
    #[default]
    Leaf,
    /// Rank each beam node by the cumulative sum of scorer values along its path.
    Cumulative,
}

impl<S: BeamSearchState> PartialEq for ScoredState<S> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl<S: BeamSearchState> Eq for ScoredState<S> {}

impl<S: BeamSearchState> PartialOrd for ScoredState<S> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<S: BeamSearchState> Ord for ScoredState<S> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.depth.cmp(&other.depth))
    }
}

impl<S: BeamSearchState> ScoredState<S> {
    #[inline]
    pub fn new(
        state: S,
        action: S::Action,
        root_action: Option<S::Action>,
        depth: usize,
        score: f32,
    ) -> Self {
        Self {
            state,
            action,
            root_action,
            depth,
            score,
        }
    }
}

/// Beam search using a current beam + next beam (no large expansion buffer).
///
/// This approach maximizes performance on modern CPUs by:
/// 1. Sequential memory access (cache-friendly)
/// 2. Simple comparison loops (SIMD-friendly, auto-vectorizable)
/// 3. Predictable branches (pipeline-friendly)
///
/// Const generics:
/// - MAX_BEAM_WIDTH: Maximum number of states to keep per level
/// - MAX_DEPTH: Maximum search depth (for move history)
/// - MAX_MOVES: Maximum moves per state
pub struct BeamSearch<
    S: BeamSearchState,
    const MAX_BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
    ScoreFn,
> where
    ScoreFn: Fn(&S) -> f32 + Clone,
{
    // Two heaps instead of current_beam + next_beam
    beams: [FixedBinMinHeap<ScoredState<S>, MAX_BEAM_WIDTH>; 2],
    current_idx: usize, // 0 or 1 - tracks which beam is "current"
    beam_width: usize,  // Runtime beam width (bounded by MAX_BEAM_WIDTH)

    // Action generation scratch buffer (unchanged)
    action_buffer: HeaplessVec<S::Action, MAX_MOVES>,

    // Best state in the current beam (updated during expansion)
    // Reset at the start of each expand_level to track best in newly created beam
    best_state: Option<ScoredState<S>>,

    // Default scorer used by search_top/search_top_n convenience methods.
    score_fn: ScoreFn,

    // Score composition strategy.
    score_mode: BeamScoreMode,
}

impl<
    S: BeamSearchState,
    const MAX_BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
    ScoreFn,
> BeamSearch<S, MAX_BEAM_WIDTH, MAX_DEPTH, MAX_MOVES, ScoreFn>
where
    ScoreFn: Fn(&S) -> f32 + Clone,
{
    /// Create a new beam search
    pub fn new(score_fn: ScoreFn) -> Self {
        Self {
            beams: [FixedBinMinHeap::new(), FixedBinMinHeap::new()],
            current_idx: 0,
            beam_width: MAX_BEAM_WIDTH,
            action_buffer: HeaplessVec::new(),
            best_state: None,
            score_fn,
            score_mode: BeamScoreMode::Leaf,
        }
    }

    #[inline_conditioned(always)]
    pub fn with_score_mode(mut self, score_mode: BeamScoreMode) -> Self {
        self.score_mode = score_mode;
        self
    }

    #[inline_conditioned(always)]
    pub fn score_mode(&self) -> BeamScoreMode {
        self.score_mode
    }

    #[inline_conditioned(always)]
    pub fn set_score_mode(&mut self, score_mode: BeamScoreMode) {
        self.score_mode = score_mode;
    }

    #[inline_conditioned(always)]
    pub fn beam_width(&self) -> usize {
        self.beam_width
    }

    #[inline_conditioned(always)]
    pub fn set_beam_width(&mut self, beam_width: usize) {
        assert!(
            beam_width > 0 && beam_width <= MAX_BEAM_WIDTH,
            "beam_width must be in range [1, MAX_BEAM_WIDTH]; got beam_width={beam_width}, MAX_BEAM_WIDTH={MAX_BEAM_WIDTH}"
        );
        self.beam_width = beam_width;
    }

    /// Expand one level of the search tree
    ///
    /// This is the performance-critical function. It's optimized for:
    /// - Sequential memory access (cache-friendly)
    /// - Simple inner loops (auto-vectorizable)
    /// - Minimal branching (pipeline-friendly)
    ///
    /// Terminal states are filtered out during expansion to avoid wasting beam slots
    #[inline_conditioned(always)]
    fn expand_level_using<F>(&mut self, score_fn: &F) -> usize
    where
        F: Fn(&S) -> f32,
    {
        // Determine the current+next beam and clear the next beam for filling
        let next_idx = 1 - self.current_idx;
        let active_beam_width = self.beam_width;
        let current_beam: &FixedBinMinHeap<ScoredState<S>, MAX_BEAM_WIDTH> =
            unsafe { &*(&self.beams[self.current_idx] as *const _) };
        let next_beam: &mut FixedBinMinHeap<ScoredState<S>, MAX_BEAM_WIDTH> =
            unsafe { &mut *(&mut self.beams[next_idx] as *mut _) };
        next_beam.clear();
        self.best_state = None;

        for beam_idx in 0..current_beam.len() {
            // Fill action buffer from parent beam state
            let parent = unsafe { current_beam.as_slice().get_unchecked(beam_idx) };
            self.action_buffer.clear();
            let _action_count = parent
                .state
                .generate_actions::<MAX_MOVES>(&mut self.action_buffer);

            // For each action, create lightweight child and insert into NEXT beam
            for &action in self.action_buffer.into_iter() {
                let new_state = parent.state.apply_action(&action);
                let raw_score = score_fn(&new_state);
                if !raw_score.is_finite() {
                    continue;
                }
                let score = match self.score_mode {
                    BeamScoreMode::Leaf => raw_score,
                    BeamScoreMode::Cumulative => parent.score + raw_score,
                };

                let child = ScoredState {
                    state: new_state,
                    action,
                    root_action: parent.root_action.or(Some(action)), // Branchless: propagate or set at depth 1
                    depth: parent.depth + 1,
                    score,
                };

                // Track best state (only update when needed)
                match self.best_state {
                    None => self.best_state = Some(child),
                    Some(ref best) if child > *best => self.best_state = Some(child),
                    _ => {}
                }

                if active_beam_width == MAX_BEAM_WIDTH {
                    next_beam.push_if_better_min_heap(child);
                } else if next_beam.len() < active_beam_width {
                    next_beam.push(child);
                } else if let Some(min_kept) = next_beam.peek_min()
                    && child > min_kept
                {
                    next_beam.replace_min(child);
                }
            }
        }

        let next_len = next_beam.len();
        self.current_idx = next_idx;
        next_len
    }

    /// Search to specified depth and return the single best state.
    ///
    /// Uses the cached best_state for O(1) retrieval.
    /// Returns None if the search fails (beam becomes empty).
    #[inline_conditioned(always)]
    pub fn search_top(&mut self, depth: usize) -> Option<ScoredState<S>> {
        let score_fn = self.score_fn.clone();
        self.search_top_using(depth, score_fn)
    }

    /// Search to specified depth with a one-off scoring function.
    #[inline_conditioned(always)]
    pub fn search_top_using<F>(&mut self, depth: usize, score_fn: F) -> Option<ScoredState<S>>
    where
        F: Fn(&S) -> f32,
    {
        assert!(
            depth <= MAX_DEPTH,
            "search depth ({depth}) exceeds MAX_DEPTH ({MAX_DEPTH}); increase MAX_DEPTH or pass a smaller depth"
        );

        for _d in 0..depth {
            let beam_size = self.expand_level_using(&score_fn);
            if beam_size == 0 {
                return None;
            }
        }

        self.best_state
    }

    /// Search to specified depth, falling back to the deepest completed level if the beam empties.
    #[inline_conditioned(always)]
    pub fn search_top_best_effort(&mut self, depth: usize) -> Option<ScoredState<S>> {
        let score_fn = self.score_fn.clone();
        self.search_top_best_effort_using(depth, score_fn)
    }

    /// Search to specified depth with a one-off scorer, returning the deepest available best state.
    #[inline_conditioned(always)]
    pub fn search_top_best_effort_using<F>(
        &mut self,
        depth: usize,
        score_fn: F,
    ) -> Option<ScoredState<S>>
    where
        F: Fn(&S) -> f32,
    {
        assert!(
            depth <= MAX_DEPTH,
            "search depth ({depth}) exceeds MAX_DEPTH ({MAX_DEPTH}); increase MAX_DEPTH or pass a smaller depth"
        );

        let mut best_available = self.best_state;
        for _d in 0..depth {
            let beam_size = self.expand_level_using(&score_fn);
            if beam_size == 0 {
                return best_available.and_then(|state| state.root_action.map(|_| state));
            }
            best_available = self.best_state;
        }

        self.best_state
    }

    /// Load `state` as the only root node, then run `search_top(depth)`.
    ///
    /// This is the recommended API for receding-horizon control loops (e.g. Tetris)
    /// since it avoids heap allocation.
    #[inline_conditioned(always)]
    pub fn search_top_with_state(&mut self, state: S, depth: usize) -> Option<ScoredState<S>> {
        let score_fn = self.score_fn.clone();
        self.search_top_with_state_using(state, depth, score_fn)
    }

    /// Load `state` as the root node, then run `search_top_best_effort(depth)`.
    #[inline_conditioned(always)]
    pub fn search_top_with_state_best_effort(
        &mut self,
        state: S,
        depth: usize,
    ) -> Option<ScoredState<S>> {
        let score_fn = self.score_fn.clone();
        self.search_top_with_state_best_effort_using(state, depth, score_fn)
    }

    /// Load `state` as the only root node, then run `search_top_using(depth, score_fn)`.
    #[inline_conditioned(always)]
    pub fn search_top_with_state_using<F>(
        &mut self,
        state: S,
        depth: usize,
        score_fn: F,
    ) -> Option<ScoredState<S>>
    where
        F: Fn(&S) -> f32,
    {
        self.beams[self.current_idx].clear();
        let score = score_fn(&state);
        let root = ScoredState::new(
            state,
            S::Action::default(),
            None, // Root has no root_action
            0,
            score,
        );
        self.beams[self.current_idx].push(root);
        self.best_state = Some(root);
        self.search_top_using(depth, score_fn)
    }

    /// Load `state` as the root node, then run `search_top_best_effort_using(depth, score_fn)`.
    #[inline_conditioned(always)]
    pub fn search_top_with_state_best_effort_using<F>(
        &mut self,
        state: S,
        depth: usize,
        score_fn: F,
    ) -> Option<ScoredState<S>>
    where
        F: Fn(&S) -> f32,
    {
        self.beams[self.current_idx].clear();
        let score = score_fn(&state);
        let root = ScoredState::new(
            state,
            S::Action::default(),
            None, // Root has no root_action
            0,
            score,
        );
        self.beams[self.current_idx].push(root);
        self.best_state = Some(root);
        self.search_top_best_effort_using(depth, score_fn)
    }

    /// Search to specified depth and return the top N states (unordered).
    ///
    /// Returns None if the search fails (beam becomes empty) or if fewer than N states exist.
    /// Uses partial selection to extract the top N in O(active_beam_width) time.
    /// Results are not sorted - use this when you only need the best N states, not their order.
    #[inline_conditioned(always)]
    pub fn search_top_n<const N: usize>(&mut self, depth: usize) -> Option<[ScoredState<S>; N]> {
        let score_fn = self.score_fn.clone();
        self.search_top_n_using::<N, _>(depth, score_fn)
    }

    /// Search to specified depth with a one-off scoring function and return top N states.
    #[inline_conditioned(always)]
    pub fn search_top_n_using<const N: usize, F>(
        &mut self,
        depth: usize,
        score_fn: F,
    ) -> Option<[ScoredState<S>; N]>
    where
        F: Fn(&S) -> f32,
    {
        assert!(
            N > 0 && N <= self.beam_width,
            "N must be in range (0, beam_width]; got N={N}, beam_width={}",
            self.beam_width
        );
        assert!(
            depth <= MAX_DEPTH,
            "search depth ({depth}) exceeds MAX_DEPTH ({MAX_DEPTH}); increase MAX_DEPTH or pass a smaller depth"
        );

        for _d in 0..depth {
            let beam_size = self.expand_level_using(&score_fn);
            if beam_size == 0 {
                return None;
            }
        }

        let beam_slice = self.beams[self.current_idx].as_mut_slice();
        if beam_slice.len() < N {
            return None;
        }

        // Extract top N using partial sort (in-place, no allocation, unordered)
        // After select_nth_unstable_by, the top N elements are in beam_slice[..N] (unordered)
        beam_slice.select_nth_unstable_by(N - 1, |a, b| b.cmp(a));
        Some(beam_slice[..N].try_into().unwrap())
    }

    /// Load `state` as the only root node, then run `search_top_n::<N>(depth)`.
    ///
    /// This is the recommended API for receding-horizon control loops (e.g. Tetris)
    /// since it avoids heap allocation.
    #[inline_conditioned(always)]
    pub fn search_top_n_with_state<const N: usize>(
        &mut self,
        state: S,
        depth: usize,
    ) -> Option<[ScoredState<S>; N]> {
        let score_fn = self.score_fn.clone();
        self.search_top_n_with_state_using::<N, _>(state, depth, score_fn)
    }

    /// Load `state` as the only root node, then run `search_top_n_using::<N>(depth, score_fn)`.
    #[inline_conditioned(always)]
    pub fn search_top_n_with_state_using<const N: usize, F>(
        &mut self,
        state: S,
        depth: usize,
        score_fn: F,
    ) -> Option<[ScoredState<S>; N]>
    where
        F: Fn(&S) -> f32,
    {
        self.beams[self.current_idx].clear();
        let score = score_fn(&state);
        let root = ScoredState::new(
            state,
            S::Action::default(),
            None, // Root has no root_action
            0,
            score,
        );
        self.beams[self.current_idx].push(root);
        self.best_state = Some(root);
        self.search_top_n_using::<N, _>(depth, score_fn)
    }
}

/// Runs N parallel beam searches with different game states/seeds and votes on the best action.
///
/// Uses mode voting: picks the action that appears most frequently across searches,
/// breaking ties by total score.
///
/// This is useful for Tetris where piece sequences are stochastic - by searching multiple
/// possible futures and voting, we get more robust action selection.
///
/// Const generics:
/// - N: Number of parallel searches to run
/// - MAX_BEAM_WIDTH, MAX_DEPTH, MAX_MOVES: Same as BeamSearch
pub struct MultiBeamSearch<
    S: BeamSearchState,
    const NUM_BEAMS: usize,
    const TOP_N_PER_BEAM: usize,
    const MAX_BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
    ScoreFn,
> where
    S::Action: Eq + std::hash::Hash,
    ScoreFn: Fn(&S) -> f32 + Clone,
{
    pub searches: Vec<BeamSearch<S, MAX_BEAM_WIDTH, MAX_DEPTH, MAX_MOVES, ScoreFn>>,
}

impl<
    S: BeamSearchState + Send + Sync,
    const NUM_BEAMS: usize,
    const TOP_N_PER_BEAM: usize,
    const MAX_BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
    ScoreFn,
> MultiBeamSearch<S, NUM_BEAMS, TOP_N_PER_BEAM, MAX_BEAM_WIDTH, MAX_DEPTH, MAX_MOVES, ScoreFn>
where
    S::Action: Eq + std::hash::Hash + Send + Sync,
    ScoreFn: Fn(&S) -> f32 + Clone + Send + Sync,
{
    /// Create a new MultiBeamSearch with N independent search instances
    pub fn new(score_fn: ScoreFn) -> Self {
        Self {
            searches: (0..NUM_BEAMS)
                .map(|_| BeamSearch::new(score_fn.clone()))
                .collect(),
        }
    }
}

// Note: No Default impl since we require an initial state for the buffer

// Specialized implementation for BeamTetrisState to support multi-seed searches
impl<
    const NUM_BEAMS: usize,
    const TOP_N_PER_BEAM: usize,
    const MAX_BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
    ScoreFn,
>
    MultiBeamSearch<
        BeamTetrisState,
        NUM_BEAMS,
        TOP_N_PER_BEAM,
        MAX_BEAM_WIDTH,
        MAX_DEPTH,
        MAX_MOVES,
        ScoreFn,
    >
where
    ScoreFn: Fn(&BeamTetrisState) -> f32 + Clone + Send + Sync,
{
    pub fn search_count_actions_with_seeds(
        &mut self,
        base_state: BeamTetrisState,
        base_seed: u64,
        depth: usize,
        beam_width: usize,
    ) -> OrientationCounts {
        assert!(
            depth <= MAX_DEPTH,
            "search depth ({depth}) exceeds MAX_DEPTH ({MAX_DEPTH}); increase MAX_DEPTH or pass a smaller depth"
        );
        assert!(
            beam_width > 0 && beam_width <= MAX_BEAM_WIDTH,
            "beam_width must be in range [1, MAX_BEAM_WIDTH]; got beam_width={beam_width}, MAX_BEAM_WIDTH={MAX_BEAM_WIDTH}"
        );
        if TOP_N_PER_BEAM > 1 {
            assert!(
                TOP_N_PER_BEAM <= beam_width,
                "TOP_N_PER_BEAM ({TOP_N_PER_BEAM}) exceeds beam_width ({beam_width}); increase beam_width or lower TOP_N_PER_BEAM"
            );
        }

        let base_par_iter = self.searches.par_iter_mut().enumerate();
        let counts: OrientationCounts = match TOP_N_PER_BEAM {
            1 => base_par_iter
                .filter_map(|(i, search)| {
                    let mut game = base_state.0;
                    game.rng = TetrisGameRng::new(base_seed + i as u64);
                    search.set_beam_width(beam_width);
                    search
                        .search_top_with_state(BeamTetrisState(game), depth)
                        .and_then(|scored| scored.root_action)
                })
                .fold(
                    || OrientationCounts::default(),
                    |mut acc, action| {
                        acc.add_action(action);
                        acc
                    },
                )
                .reduce(
                    || OrientationCounts::default(),
                    |mut a, b| {
                        a.merge(b);
                        a
                    },
                ),
            _ => base_par_iter
                .flat_map(|(i, search)| {
                    let mut game = base_state.0;
                    game.rng = TetrisGameRng::new(base_seed + i as u64);
                    search.set_beam_width(beam_width);
                    search
                        .search_top_n_with_state::<TOP_N_PER_BEAM>(BeamTetrisState(game), depth)
                        .into_par_iter()
                        .flatten()
                        .filter_map(|scored| scored.root_action)
                })
                .fold(
                    || OrientationCounts::default(),
                    |mut acc, action| {
                        acc.add_action(action);
                        acc
                    },
                )
                .reduce(
                    || OrientationCounts::default(),
                    |mut a, b| {
                        a.merge(b);
                        a
                    },
                ),
        };

        counts
    }

    pub fn search_count_actions_with_seeds_using<F>(
        &mut self,
        base_state: BeamTetrisState,
        base_seed: u64,
        depth: usize,
        beam_width: usize,
        score_fn: F,
    ) -> OrientationCounts
    where
        F: Fn(&BeamTetrisState) -> f32 + Clone + Send + Sync,
    {
        assert!(
            depth <= MAX_DEPTH,
            "search depth ({depth}) exceeds MAX_DEPTH ({MAX_DEPTH}); increase MAX_DEPTH or pass a smaller depth"
        );
        assert!(
            beam_width > 0 && beam_width <= MAX_BEAM_WIDTH,
            "beam_width must be in range [1, MAX_BEAM_WIDTH]; got beam_width={beam_width}, MAX_BEAM_WIDTH={MAX_BEAM_WIDTH}"
        );
        if TOP_N_PER_BEAM > 1 {
            assert!(
                TOP_N_PER_BEAM <= beam_width,
                "TOP_N_PER_BEAM ({TOP_N_PER_BEAM}) exceeds beam_width ({beam_width}); increase beam_width or lower TOP_N_PER_BEAM"
            );
        }

        let base_par_iter = self.searches.par_iter_mut().enumerate();
        match TOP_N_PER_BEAM {
            1 => base_par_iter
                .filter_map(|(i, search)| {
                    let mut game = base_state.0;
                    game.rng = TetrisGameRng::new(base_seed + i as u64);
                    search.set_beam_width(beam_width);
                    search
                        .search_top_with_state_using(BeamTetrisState(game), depth, score_fn.clone())
                        .and_then(|scored| scored.root_action)
                })
                .fold(
                    || OrientationCounts::default(),
                    |mut acc, action| {
                        acc.add_action(action);
                        acc
                    },
                )
                .reduce(
                    || OrientationCounts::default(),
                    |mut a, b| {
                        a.merge(b);
                        a
                    },
                ),
            _ => base_par_iter
                .flat_map(|(i, search)| {
                    let mut game = base_state.0;
                    game.rng = TetrisGameRng::new(base_seed + i as u64);
                    search.set_beam_width(beam_width);
                    search
                        .search_top_n_with_state_using::<TOP_N_PER_BEAM, _>(
                            BeamTetrisState(game),
                            depth,
                            score_fn.clone(),
                        )
                        .into_par_iter()
                        .flatten()
                        .filter_map(|scored| scored.root_action)
                })
                .fold(
                    || OrientationCounts::default(),
                    |mut acc, action| {
                        acc.add_action(action);
                        acc
                    },
                )
                .reduce(
                    || OrientationCounts::default(),
                    |mut a, b| {
                        a.merge(b);
                        a
                    },
                ),
        }
    }

    /// Search with N different seeds applied to the same base game state.
    ///
    /// This creates N variants of the base game with identical board/bag/piece state
    /// but different RNG seeds for future piece generation. Seeds are generated as:
    /// base_seed, base_seed+1, base_seed+2, ..., base_seed+(N-1)
    ///
    /// The current bag order is preserved; only the RNG seed changes, affecting
    /// future bag refills when the current bag empties.
    ///
    /// Uses mode voting: picks the action that appears most frequently, breaking ties by total score.
    ///
    /// # Arguments
    /// * `base_state` - The starting game state to use for all searches
    /// * `base_seed` - Base seed value; each search gets base_seed + index
    /// * `depth` - Search depth
    ///
    /// # Returns
    /// (best_action, average_score) or None if all searches fail
    pub fn search_with_seeds(
        &mut self,
        base_state: BeamTetrisState,
        base_seed: u64,
        depth: usize,
        beam_width: usize,
    ) -> Option<TetrisPiecePlacement> {
        let counts = self.search_count_actions_with_seeds(base_state, base_seed, depth, beam_width);

        // Find best orientation by vote count
        let (best_orientation, best_count) = counts.top_orientation();
        if best_count == 0 {
            return None;
        }

        Some(TetrisPiecePlacement {
            piece: base_state.0.current_piece,
            orientation: best_orientation,
        })
    }

    pub fn search_with_seeds_using<F>(
        &mut self,
        base_state: BeamTetrisState,
        base_seed: u64,
        depth: usize,
        beam_width: usize,
        score_fn: F,
    ) -> Option<TetrisPiecePlacement>
    where
        F: Fn(&BeamTetrisState) -> f32 + Clone + Send + Sync,
    {
        let counts = self.search_count_actions_with_seeds_using(
            base_state, base_seed, depth, beam_width, score_fn,
        );

        let (best_orientation, best_count) = counts.top_orientation();
        if best_count == 0 {
            return None;
        }

        Some(TetrisPiecePlacement {
            piece: base_state.0.current_piece,
            orientation: best_orientation,
        })
    }
}

/// Wrapper for TetrisGame that implements BeamSearchState
#[derive(Clone, Copy)]
pub struct BeamTetrisState(pub TetrisGame);

#[derive(Debug, Clone, Copy)]
pub struct TetrisBoardScoreState {
    pub board: TetrisBoard,
    pub recent_lines_cleared: u32,
}

pub const HEIGHT_MSE_ROW_PENALTY_BASE: f32 = 1.25;

const fn pow_f32(base: f32, exponent: usize) -> f32 {
    let mut result = 1.0;
    let mut remaining = exponent;
    while remaining > 0 {
        result *= base;
        remaining -= 1;
    }
    result
}

/// Byte-level LUT for weighted popcount: `WPOP_LUT[byte]` = Σ (bit_i × 1.25^i) for i in 0..8.
///
/// Higher byte lanes reuse this table with a constant scale factor:
///   mid-byte score = WPOP_LUT[byte] × 1.25^8
///   hi-nibble score = WPOP_LUT[nibble] × 1.25^16
const WPOP_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut v = 0usize;
    while v < 256 {
        let mut sum = 0.0f32;
        let mut bit = 0usize;
        while bit < 8 {
            if (v >> bit) & 1 != 0 {
                sum += pow_f32(HEIGHT_MSE_ROW_PENALTY_BASE, bit);
            }
            bit += 1;
        }
        lut[v] = sum;
        v += 1;
    }
    lut
};

const WPOP_SCALE_MID: f32 = pow_f32(HEIGHT_MSE_ROW_PENALTY_BASE, 8);
const WPOP_SCALE_HI: f32 = pow_f32(HEIGHT_MSE_ROW_PENALTY_BASE, 16);

/// Compare a board against empty by summing per-cell weights `1.25^row` for every filled cell.
///
/// Uses a single byte-level LUT for branchless weighted popcount per column.
#[inline_conditioned(always)]
pub const fn height_mse_distance_from_empty(board: TetrisBoard) -> f32 {
    const PLAYABLE_MASK: u32 = (1u32 << TetrisBoard::HEIGHT) - 1;

    let limbs = board.as_limbs();
    let mut total = 0.0f32;

    let mut col = 0usize;
    while col < TetrisBoard::WIDTH {
        let v = limbs[col] & PLAYABLE_MASK;
        total += WPOP_LUT[(v & 0xFF) as usize]
            + WPOP_LUT[((v >> 8) & 0xFF) as usize] * WPOP_SCALE_MID
            + WPOP_LUT[((v >> 16) & 0xF) as usize] * WPOP_SCALE_HI;
        col += 1;
    }
    total
}

/// Default Tetris scorer for beam search.
///
/// Beam search maximizes scores, so this returns the negative height-weighted distance from empty.
#[inline_conditioned(always)]
pub const fn height_mse_board_score(state: &TetrisBoardScoreState) -> f32 {
    if state.board.is_lost() {
        return f32::NEG_INFINITY;
    }

    -height_mse_distance_from_empty(state.board)
}

#[inline_conditioned(always)]
pub fn height_mse_beam_tetris_score(state: &BeamTetrisState) -> f32 {
    height_mse_board_score(&TetrisBoardScoreState {
        board: state.0.board,
        recent_lines_cleared: state.0.recent_lines_cleared,
    })
}

pub type TetrisBeamSearch<
    const MAX_BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
> = BeamSearch<BeamTetrisState, MAX_BEAM_WIDTH, MAX_DEPTH, MAX_MOVES, fn(&BeamTetrisState) -> f32>;

pub type TetrisMultiBeamSearch<
    const NUM_BEAMS: usize,
    const TOP_N_PER_BEAM: usize,
    const MAX_BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
> = MultiBeamSearch<
    BeamTetrisState,
    NUM_BEAMS,
    TOP_N_PER_BEAM,
    MAX_BEAM_WIDTH,
    MAX_DEPTH,
    MAX_MOVES,
    fn(&BeamTetrisState) -> f32,
>;

impl BeamTetrisState {
    pub fn new(game: TetrisGame) -> Self {
        Self(game)
    }
}

impl BeamSearchState for BeamTetrisState {
    type Action = TetrisPiecePlacement;

    #[inline_conditioned(always)]
    fn apply_action(&self, action: &Self::Action) -> Self {
        let mut g = self.0;
        let _res = g.apply_placement(*action);
        Self(g)
    }

    #[inline_conditioned(always)]
    fn generate_actions<const M: usize>(&self, buffer: &mut HeaplessVec<Self::Action, M>) -> usize {
        if self.0.board.is_lost() {
            return 0;
        }
        buffer.fill_from_slice(self.0.current_placements())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use tetris_utils::HeaplessVec;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    enum Direction {
        #[default]
        Stay,
        Up,
        Down,
        Left,
        Right,
    }

    #[derive(Clone, Copy, Debug)]
    struct GridState<const N: usize> {
        position: (i32, i32),
        goal: (i32, i32),
    }

    impl<const N: usize> GridState<N> {
        #[inline]
        fn in_bounds((x, y): (i32, i32)) -> bool {
            x >= 0 && y >= 0 && (x as usize) < N && (y as usize) < N
        }

        /// 4-bit mask for (Up,Down,Left,Right) indicating which directions are blocked.
        #[inline]
        fn blocked_mask(&self) -> u8 {
            let mut rng = rand::rng();
            let mask = rng.random_range(0..16);
            mask as u8 & 0b1111
        }

        /// Score: negative Manhattan distance to goal (higher is better)
        #[inline]
        fn score(&self) -> f32 {
            let (x, y) = self.position;
            let (gx, gy) = self.goal;
            let distance = (x - gx).abs() + (y - gy).abs();
            -(distance as f32)
        }
    }

    impl<const N: usize> BeamSearchState for GridState<N> {
        type Action = Direction;

        fn apply_action(&self, action: &Self::Action) -> Self {
            let (x, y) = self.position;
            let new_pos = match action {
                Direction::Stay => (x, y),
                Direction::Up => (x, y - 1),
                Direction::Down => (x, y + 1),
                Direction::Left => (x - 1, y),
                Direction::Right => (x + 1, y),
            };

            Self {
                position: new_pos,
                goal: self.goal,
            }
        }

        fn generate_actions<const MAX_ACTIONS: usize>(
            &self,
            buffer: &mut HeaplessVec<Self::Action, MAX_ACTIONS>,
        ) -> usize {
            buffer.clear();
            let moves = [
                Direction::Stay,
                Direction::Up,
                Direction::Down,
                Direction::Left,
                Direction::Right,
            ];
            let blocked = self.blocked_mask();

            // If we're at the goal, allow a no-op move so the beam can retain the
            // goal state across levels.
            if self.position == self.goal {
                buffer.try_push(Direction::Stay);
            }

            for mv in moves {
                // `blocked` is a 4-bit mask for cardinal directions only:
                // bit 0=Up, 1=Down, 2=Left, 3=Right. `Stay` is never blocked.
                match mv {
                    Direction::Up | Direction::Down | Direction::Left | Direction::Right => {
                        let dir_idx: u8 = match mv {
                            Direction::Up => 0,
                            Direction::Down => 1,
                            Direction::Left => 2,
                            Direction::Right => 3,
                            Direction::Stay => continue,
                        };
                        if (blocked & (1u8 << dir_idx)) != 0 {
                            continue;
                        }
                    }
                    Direction::Stay => {}
                }

                let new_state = self.apply_action(&mv);
                let pos = new_state.position;
                if Self::in_bounds(pos) {
                    buffer.try_push(mv);
                }
            }

            buffer.len()
        }
    }

    #[test]
    fn test_height_mse_distance_from_empty_empty_board() {
        let board = TetrisBoard::new();
        assert_eq!(height_mse_distance_from_empty(board), 0.0);
    }

    #[test]
    fn test_height_mse_distance_from_empty_random_boards() {
        // Reference implementation: iterate row_diffs the old way.
        fn reference_score(board: TetrisBoard) -> f32 {
            let limbs = board.as_limbs();
            let mask = (1u32 << TetrisBoard::HEIGHT) - 1;
            let mut row_diffs = [0u8; TetrisBoard::HEIGHT];
            for col in 0..TetrisBoard::WIDTH {
                let mut bits = limbs[col] & mask;
                while bits != 0 {
                    let row = bits.trailing_zeros() as usize;
                    row_diffs[row] += 1;
                    bits &= bits - 1;
                }
            }
            let mut total = 0.0f32;
            for row in 0..TetrisBoard::HEIGHT {
                total += row_diffs[row] as f32 * pow_f32(HEIGHT_MSE_ROW_PENALTY_BASE, row);
            }
            total
        }

        let mut rng = rand::rng();
        for _ in 0..1000 {
            let mut board = TetrisBoard::new();
            board.set_random_bits(50, &mut rng);
            let expected = reference_score(board);
            let actual = height_mse_distance_from_empty(board);
            assert!(
                (expected - actual).abs() < 1e-3,
                "mismatch: expected={expected}, actual={actual}"
            );
        }
    }

    #[test]
    fn test_beam_search_grid_many_cases() {
        const NUM_TESTS: usize = 10;
        const N: usize = 100;
        const BEAM_WIDTH: usize = 256;
        const MAX_DEPTH: usize = 64;
        const MAX_MOVES: usize = 5;
        const ITERS: usize = 500;

        fn cell_to_pos<const N: usize>(idx: usize) -> (i32, i32) {
            let x = (idx % N) as i32;
            let y = (idx / N) as i32;
            (x, y)
        }

        let mut rng = rand::rng();
        let mut search =
            BeamSearch::<GridState<N>, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES, _>::new(|state| {
                state.score()
            });

        for _ in 0..NUM_TESTS {
            let start = cell_to_pos::<N>(rng.random_range(0..N * N));
            let goal = cell_to_pos::<N>(rng.random_range(0..N * N));
            let initial = GridState::<N> {
                position: start,
                goal,
            };

            let mut made = 0usize;
            let mut s = initial;
            while (made < ITERS) && (s.position != goal) {
                let Some(state) = search.search_top_with_state(s, 5) else {
                    continue;
                };
                s = s.apply_action(&state.root_action.unwrap());
                assert!(GridState::<N>::in_bounds(s.position), "moved out of bounds");

                made += 1;
            }
            assert_eq!(s.position, goal, "did not reach goal");
        }
    }
}
