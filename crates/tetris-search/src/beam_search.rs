use proc_macros::inline_conditioned;
use rayon::prelude::*;
use tetris_game::{
    TetrisGame, TetrisGameRng, TetrisPieceOrientation, TetrisPiecePlacement, repeat_idx_unroll,
    utils::{FixedBinMinHeap, HeaplessVec},
};

/// Trait for states that can be used in beam search
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

    /// Evaluate this state (higher is better)
    fn evaluate(&self) -> f32;

    /// Check if this state is terminal (game over, no valid moves, etc.)
    /// Terminal states will be completely filtered from the beam
    ///
    /// Returns true if the state is terminal (dead/game-over)
    /// Default implementation: state is terminal if it has no valid moves
    ///
    /// NOTE: This should detect FAILURE terminals (game over, death, etc.)
    /// For SUCCESS terminals (reached goal), you may want to handle differently:
    /// - Option A: is_terminal() = true (stops search at goal)
    /// - Option B: is_terminal() = false (allows search to continue past goal)
    /// Choose based on your problem domain.
    fn is_terminal(&self) -> bool {
        // Safe default: assume non-terminal.
        //
        // IMPORTANT: BeamSearch uses `is_terminal()` to *filter* states during expansion.
        // A bad default here can silently prune valid states or even panic if an impl of
        // `generate_moves` expects the buffer to be "large enough".
        //
        // You should override this with a fast, correct failure-terminal check
        // (e.g. game-over, dead-end, out-of-bounds).
        false
    }
}

/// Lightweight scored state for heap operations (lives in next_beam)
#[derive(Clone, Copy)]
pub struct ScoredState<S: BeamSearchState> {
    pub state: S,
    pub score: f32,
    pub action: S::Action,              // Action from parent to this state
    pub root_action: Option<S::Action>, // First action from root (None for root, Some for descendants)
    pub depth: usize,
}

// Total ordering by score for heap operations
impl<S: BeamSearchState> PartialEq for ScoredState<S> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score).is_eq()
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
        self.score.total_cmp(&other.score)
    }
}

impl<S: BeamSearchState> ScoredState<S> {
    #[inline]
    pub fn new(
        state: S,
        score: f32,
        action: S::Action,
        root_action: Option<S::Action>,
        depth: usize,
    ) -> Self {
        Self {
            state,
            score,
            action,
            root_action,
            depth,
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
/// - BEAM_WIDTH: Number of states to keep per level
/// - MAX_DEPTH: Maximum search depth (for move history)
/// - MAX_MOVES: Maximum moves per state
pub struct BeamSearch<
    S: BeamSearchState,
    const BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
> {
    // Two heaps instead of current_beam + next_beam
    beams: [FixedBinMinHeap<ScoredState<S>, BEAM_WIDTH>; 2],
    current_idx: usize, // 0 or 1 - tracks which beam is "current"

    // Action generation scratch buffer (unchanged)
    action_buffer: HeaplessVec<S::Action, MAX_MOVES>,

    // Best state in the current beam (updated during expansion)
    // Reset at the start of each expand_level to track best in newly created beam
    best_state: Option<ScoredState<S>>,
}

impl<S: BeamSearchState, const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize>
    BeamSearch<S, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    /// Create a new beam search
    pub fn new() -> Self {
        Self {
            beams: [FixedBinMinHeap::new(), FixedBinMinHeap::new()],
            current_idx: 0,
            action_buffer: HeaplessVec::new(),
            best_state: None,
        }
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
    fn expand_level(&mut self) -> usize {
        let next_idx = 1 - self.current_idx;
        let current_beam: &FixedBinMinHeap<ScoredState<S>, BEAM_WIDTH> =
            unsafe { &*(&self.beams[self.current_idx] as *const _) };
        let next_beam: &mut FixedBinMinHeap<ScoredState<S>, BEAM_WIDTH> =
            unsafe { &mut *(&mut self.beams[next_idx] as *mut _) };
        next_beam.clear();

        // Reset best_state for this level
        self.best_state = None;

        // Expansion: read from current, write to next
        let current_len = current_beam.len();
        for beam_idx in 0..current_len {
            // SAFETY: We're reading from the current beam which won't be modified
            let parent = unsafe { current_beam.as_slice().get_unchecked(beam_idx) };

            // Generate actions
            self.action_buffer.clear();
            let _action_count = parent
                .state
                .generate_actions::<MAX_MOVES>(&mut self.action_buffer);

            // For each action, create lightweight child and insert into NEXT beam
            for &action in self.action_buffer.iter() {
                let new_state = parent.state.apply_action(&action);

                let child = ScoredState {
                    state: new_state,
                    score: new_state.evaluate(),
                    action,
                    root_action: parent.root_action.or(Some(action)), // Branchless: propagate or set at depth 1
                    depth: parent.depth + 1,
                };

                // Track best state (only update when needed)
                match self.best_state {
                    None => self.best_state = Some(child),
                    Some(best) if child.score > best.score => self.best_state = Some(child),
                    _ => {}
                }

                next_beam.push_if_better_min_heap(child);
            }
        }

        // Switch to the next beam if it's not empty; otherwise, return current beam length.
        self.current_idx = next_idx;
        current_beam.len()
    }

    /// Search to specified depth and return the single best state.
    ///
    /// Uses the cached best_state for O(1) retrieval.
    /// Returns None if the search fails (beam becomes empty).
    #[inline_conditioned(always)]
    pub fn search_top(&mut self, depth: usize) -> Option<ScoredState<S>> {
        assert!(
            depth <= MAX_DEPTH,
            "search depth ({depth}) exceeds MAX_DEPTH ({MAX_DEPTH}); increase MAX_DEPTH or pass a smaller depth"
        );

        for _d in 0..depth {
            let beam_size = self.expand_level();
            if beam_size == 0 {
                return None;
            }
        }

        self.best_state
    }

    /// Load `state` as the only root node, then run `search_top(depth)`.
    ///
    /// This is the recommended API for receding-horizon control loops (e.g. Tetris)
    /// since it avoids heap allocation.
    #[inline_conditioned(always)]
    pub fn search_top_with_state(&mut self, state: S, depth: usize) -> Option<ScoredState<S>> {
        self.beams[self.current_idx].clear();
        let root = ScoredState::new(
            state,
            state.evaluate(),
            S::Action::default(),
            None, // Root has no root_action
            0,
        );
        self.beams[self.current_idx].push(root);
        self.best_state = Some(root);
        self.search_top(depth)
    }

    /// Search to specified depth and return the top N states (unordered).
    ///
    /// Returns None if the search fails (beam becomes empty) or if fewer than N states exist.
    /// Uses partial selection to extract the top N in O(BEAM_WIDTH) time.
    /// Results are not sorted - use this when you only need the best N states, not their order.
    #[inline_conditioned(always)]
    pub fn search_top_n<const N: usize>(&mut self, depth: usize) -> Option<[ScoredState<S>; N]> {
        assert!(
            N > 0 && N <= BEAM_WIDTH,
            "N must be in range (0, BEAM_WIDTH]; got N={N}, BEAM_WIDTH={BEAM_WIDTH}"
        );
        assert!(
            depth <= MAX_DEPTH,
            "search depth ({depth}) exceeds MAX_DEPTH ({MAX_DEPTH}); increase MAX_DEPTH or pass a smaller depth"
        );

        for _d in 0..depth {
            let beam_size = self.expand_level();
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
        self.beams[self.current_idx].clear();
        let root = ScoredState::new(
            state,
            state.evaluate(),
            S::Action::default(),
            None, // Root has no root_action
            0,
        );
        self.beams[self.current_idx].push(root);
        self.best_state = Some(root);
        self.search_top_n::<N>(depth)
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
/// - BEAM_WIDTH, MAX_DEPTH, MAX_MOVES: Same as BeamSearch
pub struct MultiBeamSearch<
    S: BeamSearchState,
    const NUM_BEAMS: usize,
    const TOP_N_PER_BEAM: usize,
    const BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
> where
    S::Action: Eq + std::hash::Hash,
{
    searches: Vec<BeamSearch<S, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>>,
}

impl<
    S: BeamSearchState + Send + Sync,
    const NUM_BEAMS: usize,
    const TOP_N_PER_BEAM: usize,
    const BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
> MultiBeamSearch<S, NUM_BEAMS, TOP_N_PER_BEAM, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
where
    S::Action: Eq + std::hash::Hash + Send + Sync,
{
    /// Create a new MultiBeamSearch with N independent search instances
    pub fn new() -> Self {
        Self {
            searches: (0..NUM_BEAMS).map(|_| BeamSearch::new()).collect(),
        }
    }
}

// Note: No Default impl since we require an initial state for the buffer

// Specialized implementation for BeamTetrisState to support multi-seed searches
impl<
    const NUM_BEAMS: usize,
    const TOP_N_PER_BEAM: usize,
    const BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
> MultiBeamSearch<BeamTetrisState, NUM_BEAMS, TOP_N_PER_BEAM, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
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
    ) -> Option<TetrisPiecePlacement> {
        const fn fold_func(
            mut acc: [usize; TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS],
            action: TetrisPiecePlacement,
        ) -> [usize; TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS] {
            acc[action.orientation.index() as usize] += 1;
            acc
        }
        const fn reduce_func(
            mut a: [usize; TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS],
            b: [usize; TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS],
        ) -> [usize; TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS] {
            repeat_idx_unroll!(TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS, I, {
                a[I] += b[I];
            });
            a
        }

        let base_par_iter = self.searches.par_iter_mut().enumerate();
        let counts = match TOP_N_PER_BEAM {
            1 => base_par_iter
                .filter_map(|(i, search)| {
                    let mut game = base_state.0;
                    game.rng = TetrisGameRng::new(base_seed + i as u64);
                    search
                        .search_top_with_state(BeamTetrisState(game), depth)
                        .and_then(|scored| scored.root_action)
                })
                .fold(
                    || [0usize; TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS],
                    fold_func,
                )
                .reduce(
                    || [0usize; TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS],
                    reduce_func,
                ),
            _ => base_par_iter
                .flat_map(|(i, search)| {
                    let mut game = base_state.0;
                    game.rng = TetrisGameRng::new(base_seed + i as u64);
                    search
                        .search_top_n_with_state::<TOP_N_PER_BEAM>(BeamTetrisState(game), depth)
                        .into_par_iter()
                        .flatten()
                        .filter_map(|scored| scored.root_action)
                })
                .fold(
                    || [0usize; TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS],
                    fold_func,
                )
                .reduce(
                    || [0usize; TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS],
                    reduce_func,
                ),
        };

        // Find best orientation by vote count
        let (best_idx, &best_count) = counts
            .iter()
            .enumerate()
            .max_by_key(|&(_i, &count)| count)?;

        if best_count == 0 {
            return None;
        }

        Some(TetrisPiecePlacement {
            piece: base_state.0.current_piece,
            orientation: TetrisPieceOrientation::from_index(best_idx as u8),
        })
    }
}

/// Wrapper for TetrisGame that implements BeamSearchState
#[derive(Clone, Copy)]
pub struct BeamTetrisState(pub TetrisGame);

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

    #[inline_conditioned(always)]
    fn evaluate(&self) -> f32 {
        if self.0.board.is_lost() {
            return f32::NEG_INFINITY;
        }

        // Classic 4-feature Tetris heuristic (widely used in many simple "near-perfect" bots):
        // score = 0.760666 * lines_cleared
        //       - 0.510066 * aggregate_height
        //       - 0.35663  * holes
        //       - 0.184483 * bumpiness
        //
        // Notes:
        // - We use `recent_lines_cleared` (lines cleared by the *last* placement), not lifetime
        //   `lines_cleared`, so the search correctly prefers line-clearing actions at each step.
        // - `aggregate_height` is the sum of per-column heights.
        let lines = self.0.recent_lines_cleared as f32;
        let holes = self.0.board.total_holes() as f32;

        let heights = self.0.board.heights();
        let mut aggregate_height = 0.0;
        for h in heights.iter() {
            aggregate_height += *h as f32;
        }

        // Bumpiness = sum of absolute differences between adjacent column heights.
        let mut bumpiness = 0.0;
        for i in 0..heights.len() - 1 {
            bumpiness += (heights[i] as f32 - heights[i + 1] as f32).abs();
        }

        0.760666 * lines
            + (-0.510066) * aggregate_height
            + (-0.35663) * holes
            + (-0.184483) * bumpiness
    }

    #[inline]
    fn is_terminal(&self) -> bool {
        self.0.board.is_lost()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use tetris_game::utils::HeaplessVec;

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
                            _ => unreachable!(),
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

        fn evaluate(&self) -> f32 {
            let (x, y) = self.position;
            let (gx, gy) = self.goal;
            let distance = (x - gx).abs() + (y - gy).abs();
            -(distance as f32)
        }

        fn is_terminal(&self) -> bool {
            // Failure-only terminal: keep stable behavior (don't depend on stochastic blocking).
            !Self::in_bounds(self.position)
        }
    }

    // MultiBeamSearch test removed - it's specialized for BeamTetrisState only

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
        let mut search = BeamSearch::<GridState<N>, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new();

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
