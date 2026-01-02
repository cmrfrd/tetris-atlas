use crate::tetris::{IsLost, TetrisBoard, TetrisGame, TetrisPiecePlacement};
use crate::utils::{FixedMinHeap, HeaplessVec};
use proc_macros::inline_conditioned;

/// Trait for states that can be used in beam search
pub trait BeamSearchState: Copy {
    /// Type representing an action in the search space
    type Action: Copy;

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

/// A state with its evaluation score and the *first move* taken from the root.
///
/// This is optimized for receding-horizon control (like Tetris), where we only
/// ever execute the first move of the best plan each step.
#[derive(Clone, Copy)]
struct ScoredState<S: BeamSearchState, const MAX_DEPTH: usize> {
    state: S,
    score: f32,
    first_action: Option<S::Action>,
    depth: usize,
}

// Total ordering by score using `f32::total_cmp` so we can store `ScoredState` directly
// inside `FixedMinHeap` (no wrapper type).
impl<S: BeamSearchState, const MAX_DEPTH: usize> PartialEq for ScoredState<S, MAX_DEPTH> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score).is_eq()
    }
}

impl<S: BeamSearchState, const MAX_DEPTH: usize> Eq for ScoredState<S, MAX_DEPTH> {}

impl<S: BeamSearchState, const MAX_DEPTH: usize> PartialOrd for ScoredState<S, MAX_DEPTH> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<S: BeamSearchState, const MAX_DEPTH: usize> Ord for ScoredState<S, MAX_DEPTH> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.score.total_cmp(&other.score)
    }
}

impl<S: BeamSearchState, const MAX_DEPTH: usize> ScoredState<S, MAX_DEPTH> {
    fn new(state: S) -> Self {
        Self {
            score: state.evaluate(),
            state,
            first_action: None,
            depth: 0,
        }
    }

    fn with_action(mut self, action: S::Action, new_state: S) -> Self {
        if self.depth < MAX_DEPTH {
            self.depth += 1;
        }
        if self.first_action.is_none() {
            self.first_action = Some(action);
        }
        Self {
            score: new_state.evaluate(),
            state: new_state,
            first_action: self.first_action,
            depth: self.depth,
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
    // Current beam: stores best `current_size` states from previous level.
    current_beam: HeaplessVec<ScoredState<S, MAX_DEPTH>, BEAM_WIDTH>,

    // Next beam stored as a fixed-capacity min-heap (root is the worst/lowest score).
    // This makes top-K maintenance O(log BEAM_WIDTH) per candidate instead of O(BEAM_WIDTH).
    next_heap: FixedMinHeap<ScoredState<S, MAX_DEPTH>, BEAM_WIDTH>,

    // Action generation scratch buffer
    action_buffer: HeaplessVec<S::Action, MAX_MOVES>,
}

impl<S: BeamSearchState, const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize>
    BeamSearch<S, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    /// Create a new beam search
    pub fn new() -> Self {
        Self {
            current_beam: HeaplessVec::new(),
            next_heap: FixedMinHeap::new(),
            action_buffer: HeaplessVec::new(),
        }
    }

    /// Reset internal buffers and load a single root state into the beam.
    ///
    /// This lets callers reuse the same `BeamSearch` instance across many searches
    /// without reallocating or rebuilding the struct.
    #[inline_conditioned(always)]
    fn load_state(&mut self, state: S) {
        self.current_beam.clear();
        self.current_beam.push(ScoredState::new(state));
        self.next_heap.clear();
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
        self.next_heap.clear();

        let beam_len = self.current_beam.len();
        for i in 0..beam_len {
            // Copy element from the current beam
            // Exit if terminal
            let scored_state = self.current_beam.to_slice()[i];
            if scored_state.state.is_terminal() {
                continue;
            }

            // Get all actions from the single beam state
            self.action_buffer.clear();
            let action_count = scored_state
                .state
                .generate_actions::<MAX_MOVES>(&mut self.action_buffer);
            debug_assert_eq!(action_count, self.action_buffer.len());

            // For each action, apply it to the state and insert the new state into the next heap
            for j in 0..action_count {
                let action = self.action_buffer.to_slice()[j];
                let new_state = scored_state.state.apply_action(&action);

                // Filter out terminal states immediately
                // This prevents wasting beam slots on dead states
                if new_state.is_terminal() {
                    continue;
                }

                let new_scored = scored_state.with_action(action, new_state);
                self.insert_next(new_scored);
            }
        }

        // Check if we generated any non-terminal states
        if self.next_heap.is_empty() {
            // All paths led to terminal states - search failed
            self.current_beam.clear();
            return 0;
        }

        // Move heap contents into `current_beam` for the next level.
        // Order doesn't matter for beam expansion.
        self.current_beam.clear();
        self.current_beam.fill_from_slice(self.next_heap.as_slice());
        self.current_beam.len()
    }

    #[inline_conditioned(always)]
    fn insert_next(&mut self, cand: ScoredState<S, MAX_DEPTH>) {
        // Maintain `next_heap` as a min-heap on `score` (root is the worst).
        self.next_heap.push_if_better_min_heap(cand);
    }

    /// Search to specified depth and return the *first move* of the best plan.
    ///
    /// This avoids heap allocation and avoids copying full move histories.
    #[inline_conditioned(always)]
    pub fn search_first_action(&mut self, depth: usize) -> Option<S::Action> {
        assert!(
            depth <= MAX_DEPTH,
            "search depth ({depth}) exceeds MAX_DEPTH ({MAX_DEPTH}); increase MAX_DEPTH or pass a smaller depth"
        );
        for _ in 0..depth {
            let beam_size = self.expand_level();
            if beam_size == 0 {
                return None;
            }
        }

        // Find best state in current beam
        let mut best_score = f32::NEG_INFINITY;
        let mut best_action: Option<S::Action> = None;
        for state in self.current_beam.to_slice() {
            if state.score > best_score {
                best_score = state.score;
                best_action = state.first_action;
            }
        }

        best_action
    }

    /// Load `state` as the only root node, then run `search_first_action(depth)`.
    ///
    /// This is the recommended API for receding-horizon control loops (e.g. Tetris)
    /// since it avoids heap allocation.
    #[inline_conditioned(always)]
    pub fn search_first_action_with_state(&mut self, state: S, depth: usize) -> Option<S::Action> {
        self.load_state(state);
        self.search_first_action(depth)
    }
}

#[derive(Clone, Copy)]
pub struct BeamTetrisState(pub(crate) TetrisGame);

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

        buffer.clear();
        let placements = self.0.current_placements();
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

    #[inline_conditioned(always)]
    fn evaluate(&self) -> f32 {
        if self.0.board.is_lost() {
            return f32::NEG_INFINITY;
        }

        // Simple heuristic: prefer more lines, fewer holes, lower height.
        // You can tune these weights aggressively.
        let lines = self.0.lines_cleared as f32;
        let lines_coeff = 10.0;

        let height = self.0.board.height() as f32;
        let height_coeff = -2.0;

        let holes = self.0.board.total_holes() as f32;
        let holes_coeff = -3.0;

        lines_coeff * lines + height_coeff * height + holes_coeff * holes
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

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Direction {
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
                let Some(action) = search.search_first_action_with_state(s, 5) else {
                    continue;
                };
                s = s.apply_action(&action);
                assert!(GridState::<N>::in_bounds(s.position), "moved out of bounds");

                made += 1;
            }
            assert_eq!(s.position, goal, "did not reach goal");
        }
    }
}

/*
TERMINAL STATE HANDLING:
========================

Terminal states are states where the game is over (either success or failure).
They are fundamentally different from "bad" states:

- Bad state: Low score, but can continue searching
- Terminal state: Cannot continue - no valid moves OR reached end condition

Why Filter Terminal States?
----------------------------
1. Efficiency: Don't waste beam slots on states that can't expand
2. Correctness: Avoid returning paths that lead to immediate failure
3. Quality: Focus beam capacity on viable paths

Implementation Strategy:
------------------------
We use a HYBRID approach for maximum performance:

1. FILTER during expansion (Approach 2):
   - When generating new states, check is_terminal()
   - Skip terminal states entirely - don't add to expansion buffer
   - Saves memory and computation

2. ALSO use in evaluation (Approach 1):
   - If terminal states slip through, they get very low scores
   - Double safety: both prevention and penalization

3. HANDLE all-terminal case:
   - If expansion generates zero non-terminal states, search fails
   - Return None from search() to indicate no solution

For Tetris Specifically:
------------------------
Terminal state = board height exceeds limit (game over)

fn is_terminal(&self) -> bool {
    self.max_height() >= 20  // Board is full
}

This is MUCH faster than checking via generate_actions(), since we can
check board height in O(1) vs generating all placements in O(40).

Performance Impact:
-------------------
Filtering terminal states early means:
- Fewer states in expansion buffer
- More beam slots for viable paths
- Faster partial selection (smaller n)
- Higher quality final solutions

Example: If 10% of expansions are terminal:
- Without filtering: 40,000 states → select 1,000
- With filtering: 36,000 states → select 1,000
- 10% fewer states to process + higher beam quality

CRITICAL: For infinite-play games like Tetris, terminal state filtering
is ESSENTIAL. Without it, the beam fills with dead states and you'll
return paths that immediately fail.
*/
