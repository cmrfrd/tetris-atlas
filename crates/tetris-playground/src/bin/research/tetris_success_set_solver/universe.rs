use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use tetris_game::{IsLost, TetrisBoard, TetrisPieceBagState, TetrisPiecePlacement};

use crate::config::{BoardAdmissibility, SolverConfig};
use crate::graph::{EdgeRange, FlatEdge, PredecessorRef, StateIndex};
use crate::state::{
    BoardId, FULL_BAG_MASK, PackedPlacement, StateId, StateKey, pack_placement, piece_branches,
};

#[derive(Debug)]
pub struct Universe {
    pub empty_board_id: BoardId,
    pub root_state_id: StateId,
    pub boards: Vec<TetrisBoard>,
    pub board_to_id: HashMap<TetrisBoard, BoardId>,
    pub states: Vec<StateKey>,
    pub state_to_id: HashMap<StateKey, StateId>,
    pub state_index: Vec<StateIndex>,
    pub edges: Vec<FlatEdge>,
    pub predecessor_ranges: Vec<EdgeRange>,
    pub predecessors: Vec<PredecessorRef>,
}

impl Universe {
    pub fn state_key(&self, state_id: StateId) -> StateKey {
        self.states[state_id as usize]
    }

    pub fn board(&self, board_id: BoardId) -> TetrisBoard {
        self.boards[board_id as usize]
    }

    pub fn board_of_state(&self, state_id: StateId) -> TetrisBoard {
        self.board(self.state_key(state_id).board_id)
    }

    pub fn state_index(&self, state_id: StateId) -> &StateIndex {
        &self.state_index[state_id as usize]
    }

    pub fn edge_slice(&self, range: EdgeRange) -> &[FlatEdge] {
        let start = range.start as usize;
        let end = start + range.len as usize;
        &self.edges[start..end]
    }

    pub fn states_with_empty_board(&self) -> impl Iterator<Item = StateId> + '_ {
        self.states
            .iter()
            .enumerate()
            .filter(move |(_, state)| state.board_id == self.empty_board_id)
            .map(|(idx, _)| idx as StateId)
    }

    pub fn predecessor_slice(&self, state_id: StateId) -> &[PredecessorRef] {
        let range = self.predecessor_ranges[state_id as usize];
        let start = range.start as usize;
        let end = start + range.len as usize;
        &self.predecessors[start..end]
    }

    pub fn target_states(&self) -> impl Iterator<Item = StateId> + '_ {
        self.states_with_empty_board()
            .filter(move |&state_id| state_id != self.root_state_id)
    }
}

#[derive(Debug, Default)]
pub struct UniverseBuilder {
    config: SolverConfig,
    boards: Vec<TetrisBoard>,
    board_to_id: HashMap<TetrisBoard, BoardId>,
    states: Vec<StateKey>,
    state_to_id: HashMap<StateKey, StateId>,
    frontier: VecDeque<StateId>,
    state_index: Vec<StateIndex>,
    edges: Vec<FlatEdge>,
}

impl UniverseBuilder {
    pub fn build(config: SolverConfig) -> Result<Universe> {
        const LOG_EVERY: Duration = Duration::from_secs(2);
        let mut builder = Self {
            config,
            ..Self::default()
        };

        ensure!(
            board_is_admissible(builder.config.root.board, builder.config.admissibility),
            "root board must be admissible"
        );

        let empty_board = TetrisBoard::new();
        let empty_board_id = builder.intern_board(empty_board);
        ensure!(empty_board_id == 0, "empty board should be interned first");
        let root_board_id = builder.intern_board(builder.config.root.board);
        let root_state_id =
            builder.intern_state(StateKey::new(root_board_id, builder.config.root.bag));

        let mut expanded_states = 0usize;
        let build_start = Instant::now();
        let mut last_log = Instant::now();
        let mut last_expanded_states = 0usize;
        let mut last_states = builder.states.len();
        let mut last_boards = builder.boards.len();
        let mut last_edges = builder.edges.len();
        while let Some(state_id) = builder.frontier.pop_front() {
            builder.expand_state(state_id)?;
            expanded_states += 1;
            if last_log.elapsed() >= LOG_EVERY {
                let log_elapsed = last_log.elapsed().as_secs_f64().max(1e-9);
                let total_elapsed = build_start.elapsed().as_secs_f64().max(1e-9);
                let delta_expanded = expanded_states.saturating_sub(last_expanded_states);
                let delta_states = builder.states.len().saturating_sub(last_states);
                let delta_boards = builder.boards.len().saturating_sub(last_boards);
                let delta_edges = builder.edges.len().saturating_sub(last_edges);
                eprintln!(
                    "[success-set:build] elapsed={:.1}s expanded_states={} states={} boards={} edges={} frontier={} frontier_gap={} new_states={} new_boards={} new_edges={} expand_rate={:.1}/s state_rate={:.1}/s edge_rate={:.1}/s avg_edges_per_expanded={:.2} approx_mem={:.1}MB",
                    total_elapsed,
                    expanded_states,
                    builder.states.len(),
                    builder.boards.len(),
                    builder.edges.len(),
                    builder.frontier.len(),
                    builder.states.len().saturating_sub(expanded_states),
                    delta_states,
                    delta_boards,
                    delta_edges,
                    delta_expanded as f64 / log_elapsed,
                    delta_states as f64 / log_elapsed,
                    delta_edges as f64 / log_elapsed,
                    builder.edges.len() as f64 / expanded_states.max(1) as f64,
                    approx_build_bytes(
                        builder.boards.len(),
                        builder.states.len(),
                        builder.edges.len()
                    ) as f64
                        / (1024.0 * 1024.0),
                );
                last_log = Instant::now();
                last_expanded_states = expanded_states;
                last_states = builder.states.len();
                last_boards = builder.boards.len();
                last_edges = builder.edges.len();
            }
        }

        let (predecessor_ranges, predecessors) =
            build_predecessors(&builder.states, &builder.state_index, &builder.edges);

        Ok(Universe {
            empty_board_id,
            root_state_id,
            boards: builder.boards,
            board_to_id: builder.board_to_id,
            states: builder.states,
            state_to_id: builder.state_to_id,
            state_index: builder.state_index,
            edges: builder.edges,
            predecessor_ranges,
            predecessors,
        })
    }

    fn intern_board(&mut self, board: TetrisBoard) -> BoardId {
        if let Some(&board_id) = self.board_to_id.get(&board) {
            return board_id;
        }

        let board_id = self.boards.len() as BoardId;
        self.boards.push(board);
        self.board_to_id.insert(board, board_id);
        board_id
    }

    fn intern_state(&mut self, state: StateKey) -> StateId {
        if let Some(&state_id) = self.state_to_id.get(&state) {
            return state_id;
        }

        let state_id = self.states.len() as StateId;
        self.states.push(state);
        self.state_to_id.insert(state, state_id);
        self.state_index.push(StateIndex {
            bag: state.bag,
            ..StateIndex::default()
        });
        self.frontier.push_back(state_id);
        state_id
    }

    fn expand_state(&mut self, state_id: StateId) -> Result<()> {
        let state = self.states[state_id as usize];
        let board = self.boards[state.board_id as usize];
        let mut piece_ranges = [EdgeRange::EMPTY; 7];

        for branch in piece_branches(state.bag) {
            let piece_idx = branch.piece.index() as usize;
            let start = self.edges.len() as u32;

            for &placement in TetrisPiecePlacement::all_from_piece(branch.piece) {
                let Some(edge) = self.build_edge(board, branch.next_bag, placement) else {
                    continue;
                };
                self.edges.push(edge);
            }

            piece_ranges[piece_idx] = EdgeRange {
                start,
                len: self.edges.len() as u32 - start,
            };
        }

        self.state_index[state_id as usize] = StateIndex {
            bag: state.bag,
            piece_ranges,
        };
        Ok(())
    }

    fn build_edge(
        &mut self,
        board: TetrisBoard,
        next_bag: TetrisPieceBagState,
        placement: TetrisPiecePlacement,
    ) -> Option<FlatEdge> {
        let mut next_board = board;
        let result = next_board.apply_piece_placement(placement);
        if result.is_lost == IsLost::LOST
            || !board_is_admissible(next_board, self.config.admissibility)
        {
            return None;
        }

        let next_board_id = self.intern_board(next_board);
        let succ = self.intern_state(StateKey::new(next_board_id, next_bag));
        Some(FlatEdge {
            succ,
            placement: pack_placement(placement),
        })
    }
}

fn approx_build_bytes(board_count: usize, state_count: usize, edge_count: usize) -> usize {
    board_count * std::mem::size_of::<TetrisBoard>()
        + state_count
            * (std::mem::size_of::<StateKey>()
                + std::mem::size_of::<StateIndex>()
                + std::mem::size_of::<StateId>())
        + edge_count * std::mem::size_of::<FlatEdge>()
}

#[inline]
pub fn board_is_admissible(board: TetrisBoard, admissibility: BoardAdmissibility) -> bool {
    board.height() <= u32::from(admissibility.max_height)
        && board.total_holes() <= admissibility.max_holes
        && board.count() <= admissibility.max_cells
        && board_surface_roughness(board) <= admissibility.max_roughness
        && board_height_spread(board) <= admissibility.max_height_spread
}

pub fn board_surface_roughness(board: TetrisBoard) -> u32 {
    let heights = board.heights();
    heights
        .windows(2)
        .map(|window| window[0].abs_diff(window[1]))
        .sum()
}

pub fn board_height_spread(board: TetrisBoard) -> u32 {
    let heights = board.heights();
    let min_height = heights.into_iter().min().unwrap_or(0);
    let max_height = heights.into_iter().max().unwrap_or(0);
    max_height.saturating_sub(min_height)
}

#[inline]
pub fn edge_contains_placement(edges: &[FlatEdge], placement: PackedPlacement) -> bool {
    edges.iter().any(|edge| edge.placement == placement)
}

fn build_predecessors(
    states: &[StateKey],
    state_index: &[StateIndex],
    edges: &[FlatEdge],
) -> (Vec<EdgeRange>, Vec<PredecessorRef>) {
    let mut counts = vec![0u32; states.len()];
    for (parent_idx, state) in states.iter().enumerate() {
        let index = &state_index[parent_idx];
        for branch in piece_branches(state.bag) {
            let range = index.piece_ranges[branch.piece.index() as usize];
            for edge in &edges[range.start as usize..(range.start + range.len) as usize] {
                counts[edge.succ as usize] += 1;
            }
        }
    }

    let mut predecessor_ranges = vec![EdgeRange::EMPTY; states.len()];
    let mut cursor = 0u32;
    for (state_id, count) in counts.iter().copied().enumerate() {
        predecessor_ranges[state_id] = EdgeRange {
            start: cursor,
            len: count,
        };
        cursor += count;
    }

    let mut write_positions = predecessor_ranges
        .iter()
        .map(|range| range.start)
        .collect::<Vec<_>>();
    let mut predecessors = vec![
        PredecessorRef {
            parent: 0,
            piece_idx: 0,
            placement: 0,
        };
        edges.len()
    ];

    for (parent_idx, state) in states.iter().enumerate() {
        let index = &state_index[parent_idx];
        for branch in piece_branches(state.bag) {
            let piece_idx = branch.piece.index() as usize;
            let range = index.piece_ranges[piece_idx];
            for edge in &edges[range.start as usize..(range.start + range.len) as usize] {
                let succ_idx = edge.succ as usize;
                let write_idx = write_positions[succ_idx] as usize;
                predecessors[write_idx] = PredecessorRef {
                    parent: parent_idx as StateId,
                    piece_idx: piece_idx as u8,
                    placement: edge.placement,
                };
                write_positions[succ_idx] += 1;
            }
        }
    }

    (predecessor_ranges, predecessors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RootStateConfig;

    #[test]
    fn empty_state_key_uses_provided_empty_board_id() {
        let universe = UniverseBuilder::build(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: u32::MAX,
                max_cells: u32::MAX,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: RootStateConfig::default(),
        })
        .expect("universe should build");

        assert_eq!(universe.empty_board_id, 0);
        assert_eq!(
            universe.state_key(0),
            StateKey::new(universe.empty_board_id, TetrisPieceBagState::new())
        );
    }

    #[test]
    fn board_admissibility_is_inclusive() {
        let mut board = TetrisBoard::new();
        board.set_bit(0, 1);
        assert!(board_is_admissible(
            board,
            BoardAdmissibility {
                max_height: 2,
                max_holes: 1,
                max_cells: 2,
                max_roughness: 2,
                max_height_spread: 2,
            }
        ));
        assert!(!board_is_admissible(
            board,
            BoardAdmissibility {
                max_height: 1,
                max_holes: 1,
                max_cells: 2,
                max_roughness: 2,
                max_height_spread: 2,
            }
        ));
        assert!(!board_is_admissible(
            board,
            BoardAdmissibility {
                max_height: 2,
                max_holes: 1,
                max_cells: 2,
                max_roughness: 1,
                max_height_spread: 2,
            }
        ));
        assert!(!board_is_admissible(
            board,
            BoardAdmissibility {
                max_height: 2,
                max_holes: 1,
                max_cells: 2,
                max_roughness: 2,
                max_height_spread: 1,
            }
        ));
    }

    #[test]
    fn all_seed_bags_are_present_for_empty_board() {
        let universe = UniverseBuilder::build(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 0,
                max_holes: u32::MAX,
                max_cells: u32::MAX,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: RootStateConfig::default(),
        })
        .expect("universe should build");

        assert_eq!(universe.root_state_id, 0);
        assert!(universe.states_with_empty_board().count() >= 1);
    }

    #[test]
    fn stored_edges_match_piece_branches() {
        let universe = UniverseBuilder::build(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: u32::MAX,
                max_cells: u32::MAX,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: RootStateConfig::default(),
        })
        .expect("universe should build");

        for state_id in 0..universe.states.len() as StateId {
            let state = universe.state_key(state_id);
            let state_index = universe.state_index(state_id);
            assert_eq!(state.bag, state_index.bag);

            for branch in piece_branches(state.bag) {
                let range = state_index.piece_ranges[branch.piece.index() as usize];
                for edge in universe.edge_slice(range) {
                    let placement = TetrisPiecePlacement::from_index(edge.placement);
                    assert_eq!(placement.piece, branch.piece);
                    let succ_board = universe.board_of_state(edge.succ);
                    assert!(board_is_admissible(
                        succ_board,
                        BoardAdmissibility {
                            max_height: 1,
                            max_holes: u32::MAX,
                            max_cells: u32::MAX,
                            max_roughness: u32::MAX,
                            max_height_spread: u32::MAX,
                        }
                    ));
                }
            }
        }
    }

    #[test]
    fn no_duplicate_seed_expansion_from_frontier() {
        let universe = UniverseBuilder::build(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 0,
                max_holes: u32::MAX,
                max_cells: u32::MAX,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: RootStateConfig::default(),
        })
        .expect("universe should build");

        let state_id = 0;
        let state = universe.state_key(state_id);
        let state_index = universe.state_index(state_id);
        let mut expected_edge_count = 0usize;
        for branch in piece_branches(state.bag) {
            let range = state_index.piece_ranges[branch.piece.index() as usize];
            expected_edge_count += range.len as usize;
        }

        assert_eq!(expected_edge_count, universe.edges.len());
    }

    #[test]
    fn inadmissible_root_is_rejected() {
        let mut root_board = TetrisBoard::new();
        root_board.set_bit(0, 1);
        let result = UniverseBuilder::build(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: 0,
                max_cells: 2,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: RootStateConfig {
                board: root_board,
                bag: TetrisPieceBagState::new(),
            },
        });

        assert!(result.is_err());
    }
}
