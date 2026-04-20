use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use dashmap::DashMap;
use rayon::prelude::*;
use serde::Serialize;
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPieceBagState, TetrisPiecePlacement};

use crate::config::SolverConfig;
use crate::state::{PackedPlacement, next_bag_state, pack_placement, piece_branches};
use crate::universe::{board_height_spread, board_is_admissible, board_surface_roughness};

pub type BagNodeId = u32;
pub type PieceNodeId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SolveMode {
    #[default]
    Full,
    CoreFirst,
    TemplateKernel,
    Optimistic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveConclusion {
    Yes,
    No,
    Unresolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum FrontierMode {
    #[default]
    FragileFirst,
    StableFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveOptions {
    pub mode: SolveMode,
    pub core_prepass_expansions: usize,
    pub core_certify_interval: usize,
    pub template_max_turning_points: u32,
    pub template_max_well_depth: u32,
    pub template_sample_expansions: usize,
    pub template_max_bridge_depth: usize,
    pub optimistic_max_repairs_per_piece: usize,
    pub optimistic_max_global_repairs: usize,
    pub optimistic_probe_width: usize,
    pub optimistic_fallback_on_thrash: bool,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            mode: SolveMode::Full,
            core_prepass_expansions: 200_000,
            core_certify_interval: 4_096,
            template_max_turning_points: 4,
            template_max_well_depth: 3,
            template_sample_expansions: 500_000,
            template_max_bridge_depth: 50,
            optimistic_max_repairs_per_piece: 32,
            optimistic_max_global_repairs: 1_000_000,
            optimistic_probe_width: 1,
            optimistic_fallback_on_thrash: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BagKey {
    pub board: TetrisBoard,
    pub bag: TetrisPieceBagState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PieceKey {
    pub board: TetrisBoard,
    pub bag: TetrisPieceBagState,
    pub piece: TetrisPiece,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeResolution {
    Won,
    Lost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieceChild {
    pub placement: PackedPlacement,
    pub succ: BagNodeId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OptimisticChildTarget {
    Activated(BagNodeId),
    Dormant(BagKey),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OptimisticChild {
    placement: PackedPlacement,
    target: OptimisticChildTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GeometryChild {
    placement: PackedPlacement,
    board: TetrisBoard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GeometryKey {
    board: TetrisBoard,
    piece: TetrisPiece,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OptimisticPolicyEntry {
    placements: Vec<PackedPlacement>,
    hits: u32,
    misses: u32,
    successful_repairs: u32,
    failed_repairs: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BoardFeatures {
    height: u32,
    holes: u32,
    cells: u32,
    roughness: u32,
    height_spread: u32,
    turning_points: u32,
    max_well_depth: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BoardSignature {
    heights: [u32; 10],
    holes: u32,
    roughness: u32,
    height_spread: u32,
    turning_points: u32,
    max_well_depth: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExpandEntry {
    score: u64,
    sequence: u64,
    bag_id: BagNodeId,
}

#[derive(Debug, Clone)]
struct ExpandTask {
    bag_id: BagNodeId,
    key: BagKey,
    branches: Vec<ExpandTaskBranch>,
}

#[derive(Debug, Clone)]
struct ExpandTaskBranch {
    piece: TetrisPiece,
    geometry: Option<Arc<[GeometryChild]>>,
}

#[derive(Debug, Clone)]
struct ExpandResult {
    bag_id: BagNodeId,
    key: BagKey,
    branches: Vec<PieceExpansionResult>,
}

#[derive(Debug, Clone)]
struct PieceExpansionResult {
    piece: TetrisPiece,
    geometry: Arc<[GeometryChild]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StagedPieceChild {
    placement: PackedPlacement,
    succ_key: BagKey,
}

#[derive(Debug, Clone)]
struct StagedPieceExpansion {
    piece: TetrisPiece,
    geometry: Arc<[GeometryChild]>,
    piece_key: PieceKey,
    children: Vec<StagedPieceChild>,
    unique_succ_keys: Vec<BagKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OptimisticActivationKind {
    ActivatedReuse,
    KnownReuse,
    NovelActivation,
}

#[derive(Debug, Clone)]
struct OptimisticExpandBranchProposal {
    piece: TetrisPiece,
    geometry: Arc<[GeometryChild]>,
    children: Vec<StagedPieceChild>,
    unique_succ_count: u32,
    active_slot: Option<u32>,
    activation_kind: Option<OptimisticActivationKind>,
    popularity_reordered: bool,
}

#[derive(Debug, Clone)]
struct OptimisticExpandProposal {
    bag_id: BagNodeId,
    key: BagKey,
    branches: Vec<OptimisticExpandBranchProposal>,
}

#[derive(Debug, Clone, Copy)]
struct OptimisticRepairProposal {
    piece_id: PieceNodeId,
    slot: Option<u32>,
    placement: Option<PackedPlacement>,
    succ_key: Option<BagKey>,
    succ_id: Option<BagNodeId>,
    activation_kind: Option<OptimisticActivationKind>,
    scanned: usize,
}

#[derive(Debug, Clone, Copy)]
struct OptimisticRepairResolution {
    slot: u32,
    placement: PackedPlacement,
    succ_key: Option<BagKey>,
    succ_id: Option<BagNodeId>,
    activation_kind: OptimisticActivationKind,
    scanned: usize,
}

#[derive(Debug, Clone)]
struct OptimisticSnapshot {
    live_won: Vec<bool>,
    activated: Vec<bool>,
    successor_popularity: Vec<usize>,
    known_bags: HashSet<BagKey>,
    probe_width: usize,
}

#[derive(Debug, Clone)]
struct StagedExpandResult {
    bag_id: BagNodeId,
    key: BagKey,
    branches: Vec<StagedPieceExpansion>,
}

#[derive(Debug, Clone)]
struct ResolvedPieceExpansion {
    piece: TetrisPiece,
    geometry: Arc<[GeometryChild]>,
    piece_id: PieceNodeId,
    children: Vec<PieceChild>,
    live_unique_succs: Vec<BagNodeId>,
    unique_succ_count: u32,
}

#[derive(Debug, Clone)]
struct ResolvedExpandResult {
    bag_id: BagNodeId,
    key: BagKey,
    branches: Vec<ResolvedPieceExpansion>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParentSegment {
    start: u32,
    len: u32,
    next: Option<u32>,
}

impl Ord for ExpandEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| other.sequence.cmp(&self.sequence))
            .then_with(|| self.bag_id.cmp(&other.bag_id))
    }
}

impl PartialOrd for ExpandEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
pub struct BagNode {
    pub key: BagKey,
    pub resolution: NodeResolution,
    pub expanded: bool,
    pub parents: Vec<PieceNodeId>,
    pub piece_nodes: [Option<PieceNodeId>; 7],
    activated: bool,
    branch_order: [u8; 7],
    branch_order_len: u8,
    next_branch_idx: u8,
    expanded_branch_count: u8,
    branch_order_initialized: bool,
}

#[derive(Debug, Clone)]
pub struct PieceNode {
    pub key: PieceKey,
    pub parent_bag: BagNodeId,
    pub resolution: NodeResolution,
    pub children: Vec<PieceChild>,
    pub unique_succ_count: u32,
    pub alive_succ_count: u32,
    pub best_action: Option<PackedPlacement>,
    pub best_child: Option<BagNodeId>,
    active_child_slot: Option<u32>,
    optimistic_children: Vec<OptimisticChild>,
    next_repair_cursor: u32,
    repair_attempts: u32,
    optimistic_exhausted: bool,
    optimistic_repair_pending: bool,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct ProofMetrics {
    pub solve_mode: SolveMode,
    pub used_full_fallback: bool,
    pub bag_node_count: usize,
    pub piece_node_count: usize,
    pub winning_count: usize,
    pub losing_count: usize,
    pub piece_winning_count: usize,
    pub piece_losing_count: usize,
    pub dependency_count: usize,
    pub deduped_edge_count: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub geometry_cache_hits: usize,
    pub geometry_cache_misses: usize,
    pub board_feature_cache_hits: usize,
    pub board_feature_cache_misses: usize,
    pub bags_expanded: usize,
    pub skipped_dead_bags: usize,
    pub parallel_batches_processed: usize,
    pub parallel_batch_size: usize,
    pub branches_expanded: usize,
    pub branches_skipped_due_to_parent_death: usize,
    pub bags_killed_by_first_failed_branch: usize,
    pub avg_branches_expanded_per_bag: f64,
    pub staged_branches_processed: usize,
    pub batch_unique_piece_keys: usize,
    pub batch_unique_bag_keys: usize,
    pub resolved_children_processed: usize,
    pub batched_parent_links_appended: usize,
    pub parent_segments_allocated: usize,
    pub parent_links_stored: usize,
    pub avg_parent_segments_per_bag: f64,
    pub optimistic_policy_entries: usize,
    pub optimistic_ranked_policy_entries: usize,
    pub optimistic_ranked_policy_capacity: usize,
    pub optimistic_policy_hits: usize,
    pub optimistic_policy_misses: usize,
    pub optimistic_repairs_attempted: usize,
    pub optimistic_repairs_succeeded: usize,
    pub optimistic_repairs_failed: usize,
    pub optimistic_repairs_promoted: usize,
    pub optimistic_repairs_demoted: usize,
    pub optimistic_reused_ranked_candidates: usize,
    pub optimistic_active_parent_links: usize,
    pub optimistic_popularity_nonzero_bags: usize,
    pub optimistic_max_successor_popularity: usize,
    pub optimistic_avg_successor_popularity: f64,
    pub optimistic_reorders_using_popularity: usize,
    pub optimistic_avg_candidates_scanned_per_repair: f64,
    pub optimistic_avg_rank_position_chosen: f64,
    pub optimistic_max_repairs_on_single_piece: usize,
    pub optimistic_activated_live_bags: usize,
    pub optimistic_dormant_child_targets: usize,
    pub optimistic_activated_frontier: usize,
    pub optimistic_novel_activations: usize,
    pub optimistic_reuse_activations: usize,
    pub optimistic_repairs_reusing_known_successor: usize,
    pub optimistic_repairs_forcing_novel_activation: usize,
    pub optimistic_fallback_triggered: bool,
    pub optimistic_expand_proposal_secs: f64,
    pub optimistic_repair_proposal_secs: f64,
    pub optimistic_repair_batches_processed: usize,
    pub optimistic_expand_proposals_revalidated: usize,
    pub optimistic_repair_proposals_revalidated: usize,
    pub candidate_core_count: usize,
    pub largest_scc_bags: usize,
    pub largest_scc_pieces: usize,
    pub closed_core_found: bool,
    pub core_certification_secs: f64,
    pub bridge_search_secs: f64,
    pub stage_secs: f64,
    pub commit_secs: f64,
    pub template_member_bags: usize,
    pub template_member_pieces: usize,
    pub template_signature_count: usize,
    pub largest_template_scc_bags: usize,
    pub largest_template_scc_pieces: usize,
    pub root_reaches_template_family: bool,
    pub template_attractor_bags: usize,
    pub template_closure_failed_piece_counts: [usize; 7],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct ProofVerification {
    pub bag_nodes_checked: usize,
    pub piece_nodes_checked: usize,
    pub witness_failures: usize,
    pub resolution_failures: usize,
}

impl ProofVerification {
    pub const fn is_clean(self) -> bool {
        self.witness_failures == 0 && self.resolution_failures == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootProofResult {
    pub conclusion: SolveConclusion,
    pub result: NodeResolution,
    pub best_action: [Option<PackedPlacement>; 7],
    pub failing_pieces: Vec<TetrisPiece>,
}

#[derive(Debug, Clone)]
pub struct ProofSolveResult {
    pub solve_mode: SolveMode,
    pub used_full_fallback: bool,
    pub root_bag: BagNodeId,
    pub bag_nodes: Vec<BagNode>,
    pub piece_nodes: Vec<PieceNode>,
    pub metrics: ProofMetrics,
    pub discover_secs: f64,
    pub propagate_secs: f64,
    pub total_secs: f64,
    pub root_result: RootProofResult,
}

#[derive(Debug, Default)]
struct ProofSolver {
    config: SolverConfig,
    solve_mode: SolveMode,
    frontier_mode: FrontierMode,
    root_bag: BagNodeId,
    bag_nodes: Vec<BagNode>,
    piece_nodes: Vec<PieceNode>,
    bag_map: HashMap<BagKey, BagNodeId>,
    piece_map: HashMap<PieceKey, PieceNodeId>,
    expand_queue: BinaryHeap<ExpandEntry>,
    dead_bag_queue: VecDeque<BagNodeId>,
    dead_piece_queue: VecDeque<PieceNodeId>,
    optimistic_repair_queue: VecDeque<PieceNodeId>,
    board_features: DashMap<TetrisBoard, BoardFeatures>,
    geometry_cache: DashMap<GeometryKey, Arc<[GeometryChild]>>,
    optimistic_policy: DashMap<GeometryKey, OptimisticPolicyEntry>,
    next_expand_sequence: u64,
    cache_hits: usize,
    cache_misses: usize,
    geometry_cache_hits: AtomicUsize,
    geometry_cache_misses: AtomicUsize,
    board_feature_cache_hits: AtomicUsize,
    board_feature_cache_misses: AtomicUsize,
    deduped_edges: usize,
    bags_expanded: usize,
    skipped_dead_bags: usize,
    parallel_local_expansion: bool,
    staged_merge: bool,
    parallel_batches_processed: usize,
    total_local_expand_secs: f64,
    total_merge_secs: f64,
    total_stage_secs: f64,
    total_commit_secs: f64,
    branches_expanded: usize,
    branches_skipped_due_to_parent_death: usize,
    bags_killed_by_first_failed_branch: usize,
    staged_branches_processed: usize,
    batch_unique_piece_keys: usize,
    batch_unique_bag_keys: usize,
    resolved_children_processed: usize,
    batched_parent_links_appended: usize,
    parent_edges: Vec<PieceNodeId>,
    parent_segments: Vec<ParentSegment>,
    bag_parent_head: Vec<Option<u32>>,
    bag_parent_tail: Vec<Option<u32>>,
    bag_parent_count: Vec<usize>,
    optimistic_policy_hits: usize,
    optimistic_policy_misses: usize,
    optimistic_repairs_attempted: usize,
    optimistic_repairs_succeeded: usize,
    optimistic_repairs_failed: usize,
    optimistic_repairs_promoted: usize,
    optimistic_repairs_demoted: usize,
    optimistic_reused_ranked_candidates: usize,
    optimistic_repair_candidates_scanned: usize,
    optimistic_rank_position_sum: usize,
    optimistic_rank_position_count: usize,
    optimistic_max_repairs_on_single_piece: usize,
    optimistic_active_parent_links: usize,
    optimistic_successor_popularity: Vec<usize>,
    optimistic_reorders_using_popularity: usize,
    optimistic_novel_activations: usize,
    optimistic_reuse_activations: usize,
    optimistic_repairs_reusing_known_successor: usize,
    optimistic_repairs_forcing_novel_activation: usize,
    optimistic_fallback_triggered: bool,
    optimistic_probe_width: usize,
    optimistic_expand_proposal_secs: f64,
    optimistic_repair_proposal_secs: f64,
    optimistic_repair_batches_processed: usize,
    optimistic_expand_proposals_revalidated: usize,
    optimistic_repair_proposals_revalidated: usize,
    candidate_core_count: usize,
    largest_scc_bags: usize,
    largest_scc_pieces: usize,
    closed_core_found: bool,
    core_certification_secs: f64,
    bridge_search_secs: f64,
    used_full_fallback: bool,
    lazy_branch_expansion: bool,
}

impl ProofSolver {
    const EXPAND_BATCH_SIZE: usize = 64;
    const OPTIMISTIC_POLICY_CAPACITY: usize = 8;

    fn is_live_and_activated(&self, bag_id: BagNodeId) -> bool {
        self.bag_nodes[bag_id as usize].resolution == NodeResolution::Won
            && self.bag_nodes[bag_id as usize].activated
    }

    fn is_live_and_dormant(&self, bag_id: BagNodeId) -> bool {
        self.bag_nodes[bag_id as usize].resolution == NodeResolution::Won
            && !self.bag_nodes[bag_id as usize].activated
    }

    fn new(config: SolverConfig) -> Self {
        Self::new_with_mode(config, SolveMode::Full, FrontierMode::FragileFirst)
    }

    fn new_with_mode(
        config: SolverConfig,
        solve_mode: SolveMode,
        frontier_mode: FrontierMode,
    ) -> Self {
        let mut solver = Self {
            config,
            solve_mode,
            frontier_mode,
            lazy_branch_expansion: matches!(solve_mode, SolveMode::Full | SolveMode::Optimistic),
            parallel_local_expansion: true,
            staged_merge: matches!(solve_mode, SolveMode::Full),
            ..Self::default()
        };
        let root_key = BagKey {
            board: solver.config.root.board,
            bag: solver.config.root.bag,
        };
        solver.board_features.insert(
            solver.config.root.board,
            solver.compute_board_features(solver.config.root.board),
        );
        solver.root_bag = solver.intern_bag(root_key);
        solver
    }

    fn is_optimistic(&self) -> bool {
        self.solve_mode == SolveMode::Optimistic
    }

    fn append_parent_links(&mut self, bag_id: BagNodeId, parents: &[PieceNodeId]) {
        if parents.is_empty() {
            return;
        }

        let start = self.parent_edges.len();
        self.parent_edges.extend_from_slice(parents);
        let segment_id = self.parent_segments.len() as u32;
        self.parent_segments.push(ParentSegment {
            start: start as u32,
            len: parents.len() as u32,
            next: None,
        });

        if let Some(tail) = self.bag_parent_tail[bag_id as usize] {
            self.parent_segments[tail as usize].next = Some(segment_id);
        } else {
            self.bag_parent_head[bag_id as usize] = Some(segment_id);
        }
        self.bag_parent_tail[bag_id as usize] = Some(segment_id);
        self.bag_parent_count[bag_id as usize] += parents.len();
    }

    fn increment_optimistic_successor_popularity(&mut self, bag_id: BagNodeId) {
        if !self.is_optimistic() {
            return;
        }
        self.optimistic_successor_popularity[bag_id as usize] =
            self.optimistic_successor_popularity[bag_id as usize].saturating_add(1);
    }

    fn decrement_optimistic_successor_popularity(&mut self, bag_id: BagNodeId) {
        if !self.is_optimistic() {
            return;
        }
        self.optimistic_successor_popularity[bag_id as usize] =
            self.optimistic_successor_popularity[bag_id as usize].saturating_sub(1);
    }

    fn materialized_bag_nodes(&self) -> Vec<BagNode> {
        let mut bag_nodes = self.bag_nodes.clone();
        for bag in &mut bag_nodes {
            bag.parents.clear();
        }
        for (piece_id, piece) in self.piece_nodes.iter().enumerate() {
            if piece.resolution != NodeResolution::Won {
                continue;
            }
            let mut seen = HashSet::new();
            for child in &piece.children {
                if self.bag_nodes[child.succ as usize].resolution == NodeResolution::Won
                    && seen.insert(child.succ)
                {
                    bag_nodes[child.succ as usize]
                        .parents
                        .push(piece_id as PieceNodeId);
                }
            }
        }
        bag_nodes
    }

    fn dependency_count(&self) -> usize {
        self.piece_nodes
            .iter()
            .filter(|piece| piece.resolution == NodeResolution::Won)
            .map(|piece| {
                piece
                    .children
                    .iter()
                    .filter_map(|child| {
                        (self.bag_nodes[child.succ as usize].resolution == NodeResolution::Won)
                            .then_some(child.succ)
                    })
                    .collect::<HashSet<_>>()
                    .len()
            })
            .sum()
    }

    fn ranked_placements_for(
        &self,
        board: TetrisBoard,
        piece: TetrisPiece,
    ) -> Option<Vec<PackedPlacement>> {
        self.optimistic_policy
            .get(&GeometryKey { board, piece })
            .map(|entry| entry.placements.clone())
    }

    fn ensure_optimistic_children(&mut self, piece_id: PieceNodeId) {
        if !self.is_optimistic() {
            return;
        }
        let piece = &mut self.piece_nodes[piece_id as usize];
        if piece.optimistic_children.is_empty() && !piece.children.is_empty() {
            piece.optimistic_children = piece
                .children
                .iter()
                .copied()
                .map(|child| OptimisticChild {
                    placement: child.placement,
                    target: OptimisticChildTarget::Activated(child.succ),
                })
                .collect();
        }
    }

    fn optimistic_child_succ(&self, piece_id: PieceNodeId, slot: u32) -> Option<BagNodeId> {
        let piece = &self.piece_nodes[piece_id as usize];
        piece
            .optimistic_children
            .get(slot as usize)
            .and_then(|child| {
                if let OptimisticChildTarget::Activated(succ) = child.target {
                    Some(succ)
                } else {
                    None
                }
            })
    }

    fn promote_policy_placement(
        &mut self,
        board: TetrisBoard,
        piece: TetrisPiece,
        placement: PackedPlacement,
        is_repair: bool,
    ) {
        self.optimistic_policy_hits += 1;
        match self.optimistic_policy.entry(GeometryKey { board, piece }) {
            dashmap::mapref::entry::Entry::Occupied(mut entry) => {
                let value = entry.get_mut();
                value.hits = value.hits.saturating_add(1);
                if is_repair {
                    value.successful_repairs = value.successful_repairs.saturating_add(1);
                    self.optimistic_repairs_promoted += 1;
                }
                if let Some(idx) = value.placements.iter().position(|&item| item == placement) {
                    if idx != 0 {
                        let placement = value.placements.remove(idx);
                        value.placements.insert(0, placement);
                    }
                } else {
                    value.placements.insert(0, placement);
                    value.placements.truncate(Self::OPTIMISTIC_POLICY_CAPACITY);
                }
            }
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(OptimisticPolicyEntry {
                    placements: vec![placement],
                    hits: 1,
                    misses: 0,
                    successful_repairs: u32::from(is_repair),
                    failed_repairs: 0,
                });
                if is_repair {
                    self.optimistic_repairs_promoted += 1;
                }
            }
        }
    }

    fn demote_policy_placement(
        &mut self,
        board: TetrisBoard,
        piece: TetrisPiece,
        placement: PackedPlacement,
        is_repair: bool,
    ) {
        self.optimistic_policy_misses += 1;
        if let Some(mut entry) = self
            .optimistic_policy
            .get_mut(&GeometryKey { board, piece })
        {
            entry.misses = entry.misses.saturating_add(1);
            if is_repair {
                entry.failed_repairs = entry.failed_repairs.saturating_add(1);
                self.optimistic_repairs_demoted += 1;
            }
            if let Some(idx) = entry.placements.iter().position(|&item| item == placement) {
                if idx + 1 < entry.placements.len() {
                    entry.placements.swap(idx, idx + 1);
                }
            }
        }
    }

    fn note_ranked_choice(
        &mut self,
        board: TetrisBoard,
        piece: TetrisPiece,
        placement: PackedPlacement,
    ) {
        if let Some(entry) = self.optimistic_policy.get(&GeometryKey { board, piece }) {
            if let Some(rank) = entry.placements.iter().position(|&item| item == placement) {
                self.optimistic_reused_ranked_candidates += 1;
                self.optimistic_rank_position_sum += rank;
                self.optimistic_rank_position_count += 1;
            }
        }
    }

    fn reorder_piece_children_optimistically(
        &mut self,
        board: TetrisBoard,
        piece: TetrisPiece,
        children: &mut [PieceChild],
    ) -> usize {
        if !self.is_optimistic() {
            return 0;
        }
        let mut target_idx = 0usize;
        if let Some(ranked) = self.ranked_placements_for(board, piece) {
            for preferred in ranked {
                if let Some(idx) = children[target_idx..]
                    .iter()
                    .position(|child| child.placement == preferred)
                {
                    let idx = target_idx + idx;
                    if idx != target_idx {
                        children[target_idx..=idx].rotate_right(1);
                    }
                    target_idx += 1;
                }
            }
        }
        target_idx
    }

    fn reorder_piece_children_by_successor_popularity(
        &mut self,
        children: &mut [PieceChild],
        locked_prefix: usize,
    ) {
        if !self.is_optimistic() || children.len() < 2 {
            return;
        }
        let locked_prefix = locked_prefix.min(children.len());
        if children.len().saturating_sub(locked_prefix) < 2 {
            return;
        }
        let mut used_popularity = false;
        children[locked_prefix..].sort_by(|lhs, rhs| {
            let lhs_pop = self.optimistic_successor_popularity[lhs.succ as usize];
            let rhs_pop = self.optimistic_successor_popularity[rhs.succ as usize];
            if lhs_pop != rhs_pop {
                used_popularity = true;
            }
            rhs_pop.cmp(&lhs_pop)
        });
        if used_popularity {
            self.optimistic_reorders_using_popularity += 1;
        }
    }

    fn optimistic_staged_child_rank(&self, child: &StagedPieceChild) -> (u8, usize) {
        if let Some(&bag_id) = self.bag_map.get(&child.succ_key) {
            let bag = &self.bag_nodes[bag_id as usize];
            if bag.resolution == NodeResolution::Lost {
                return (0, 0);
            }
            if bag.activated {
                return (3, self.optimistic_successor_popularity[bag_id as usize]);
            }
            return (2, self.optimistic_successor_popularity[bag_id as usize]);
        }
        (1, 0)
    }

    fn reorder_staged_children_optimistically(
        &mut self,
        board: TetrisBoard,
        piece: TetrisPiece,
        children: &mut [StagedPieceChild],
    ) {
        if !self.is_optimistic() {
            return;
        }
        let mut target_idx = 0usize;
        if let Some(ranked) = self.ranked_placements_for(board, piece) {
            for preferred in ranked {
                if let Some(idx) = children[target_idx..]
                    .iter()
                    .position(|child| child.placement == preferred)
                {
                    let idx = target_idx + idx;
                    if idx != target_idx {
                        children[target_idx..=idx].rotate_right(1);
                    }
                    target_idx += 1;
                }
            }
        }

        let locked_prefix = target_idx.min(children.len());
        if children.len().saturating_sub(locked_prefix) < 2 {
            return;
        }
        let mut used_popularity = false;
        children[locked_prefix..].sort_by(|lhs, rhs| {
            let lhs_rank = self.optimistic_staged_child_rank(lhs);
            let rhs_rank = self.optimistic_staged_child_rank(rhs);
            if lhs_rank != rhs_rank {
                used_popularity = true;
            }
            rhs_rank.cmp(&lhs_rank)
        });
        if used_popularity {
            self.optimistic_reorders_using_popularity += 1;
        }
    }

    fn choose_staged_optimistic_child_slot_strict_reuse(
        &self,
        children: &[StagedPieceChild],
        probe_width: usize,
    ) -> Option<(u32, bool)> {
        let probe_width = probe_width.max(1);
        let activated = children
            .iter()
            .enumerate()
            .filter_map(|(idx, child)| {
                self.bag_map
                    .get(&child.succ_key)
                    .copied()
                    .filter(|&bag_id| self.is_live_and_activated(bag_id))
                    .map(|_| idx)
            })
            .collect::<Vec<_>>();
        if let Some(&idx) = activated
            .iter()
            .take(activated.len().min(probe_width))
            .chain(activated.iter().skip(activated.len().min(probe_width)))
            .next()
        {
            return Some((idx as u32, true));
        }

        let dormant = children
            .iter()
            .enumerate()
            .filter_map(
                |(idx, child)| match self.bag_map.get(&child.succ_key).copied() {
                    Some(bag_id)
                        if self.bag_nodes[bag_id as usize].resolution == NodeResolution::Won =>
                    {
                        Some(idx)
                    }
                    None => Some(idx),
                    _ => None,
                },
            )
            .collect::<Vec<_>>();
        dormant
            .iter()
            .take(dormant.len().min(probe_width))
            .chain(dormant.iter().skip(dormant.len().min(probe_width)))
            .next()
            .copied()
            .map(|idx| (idx as u32, false))
    }

    fn choose_optimistic_child_slot_strict_reuse(
        &self,
        children: &[PieceChild],
        probe_width: usize,
    ) -> Option<(u32, bool)> {
        let probe_width = probe_width.max(1);
        let activated = children
            .iter()
            .enumerate()
            .filter_map(|(idx, child)| self.is_live_and_activated(child.succ).then_some(idx))
            .collect::<Vec<_>>();
        if !activated.is_empty() {
            let prefix = activated.len().min(probe_width);
            if let Some(&idx) = activated
                .iter()
                .take(prefix)
                .chain(activated.iter().skip(prefix))
                .next()
            {
                return Some((idx as u32, true));
            }
        }

        let dormant = children
            .iter()
            .enumerate()
            .filter_map(|(idx, child)| self.is_live_and_dormant(child.succ).then_some(idx))
            .collect::<Vec<_>>();
        let prefix = dormant.len().min(probe_width);
        dormant
            .iter()
            .take(prefix)
            .chain(dormant.iter().skip(prefix))
            .next()
            .copied()
            .map(|idx| (idx as u32, false))
    }

    fn optimistic_activation_metrics(&self) -> (usize, usize, usize) {
        let activated_live_bags = self
            .bag_nodes
            .iter()
            .filter(|bag| bag.resolution == NodeResolution::Won && bag.activated)
            .count();
        let dormant_child_targets = self
            .piece_nodes
            .iter()
            .map(|piece| {
                piece
                    .optimistic_children
                    .iter()
                    .filter(|child| matches!(child.target, OptimisticChildTarget::Dormant(_)))
                    .count()
            })
            .sum();
        let activated_frontier = self.expand_queue.len();
        (
            activated_live_bags,
            dormant_child_targets,
            activated_frontier,
        )
    }

    fn optimistic_snapshot(&self) -> OptimisticSnapshot {
        OptimisticSnapshot {
            live_won: self
                .bag_nodes
                .iter()
                .map(|bag| bag.resolution == NodeResolution::Won)
                .collect(),
            activated: self.bag_nodes.iter().map(|bag| bag.activated).collect(),
            successor_popularity: self.optimistic_successor_popularity.clone(),
            known_bags: self.bag_map.keys().copied().collect(),
            probe_width: self.optimistic_probe_width.max(1),
        }
    }

    fn optimistic_snapshot_child_rank(
        snapshot: &OptimisticSnapshot,
        child: &StagedPieceChild,
        known_bag_id: Option<BagNodeId>,
    ) -> (u8, usize) {
        if let Some(bag_id) = known_bag_id {
            if !snapshot.live_won[bag_id as usize] {
                return (0, 0);
            }
            if snapshot.activated[bag_id as usize] {
                return (3, snapshot.successor_popularity[bag_id as usize]);
            }
            return (2, snapshot.successor_popularity[bag_id as usize]);
        }
        (1, 0)
    }

    fn reorder_staged_children_optimistically_with_snapshot(
        ranked: Option<&[PackedPlacement]>,
        snapshot: &OptimisticSnapshot,
        known_bag_ids: &[Option<BagNodeId>],
        children: &mut [StagedPieceChild],
    ) -> bool {
        let mut target_idx = 0usize;
        if let Some(ranked) = ranked {
            for &preferred in ranked {
                if let Some(idx) = children[target_idx..]
                    .iter()
                    .position(|child| child.placement == preferred)
                {
                    let idx = target_idx + idx;
                    if idx != target_idx {
                        children[target_idx..=idx].rotate_right(1);
                    }
                    target_idx += 1;
                }
            }
        }

        let locked_prefix = target_idx.min(children.len());
        if children.len().saturating_sub(locked_prefix) < 2 {
            return false;
        }
        let original = children[locked_prefix..].to_vec();
        let original_known = known_bag_ids[locked_prefix..].to_vec();
        let mut suffix = original.into_iter().zip(original_known).collect::<Vec<_>>();
        suffix.sort_by(|(lhs_child, lhs_known), (rhs_child, rhs_known)| {
            let lhs_rank = Self::optimistic_snapshot_child_rank(snapshot, lhs_child, *lhs_known);
            let rhs_rank = Self::optimistic_snapshot_child_rank(snapshot, rhs_child, *rhs_known);
            rhs_rank.cmp(&lhs_rank)
        });
        let used_popularity = suffix.iter().map(|(child, _)| *child).collect::<Vec<_>>()
            != children[locked_prefix..].to_vec();
        for (offset, (child, _)) in suffix.into_iter().enumerate() {
            children[locked_prefix + offset] = child;
        }
        used_popularity
    }

    fn choose_staged_optimistic_child_slot_strict_reuse_with_snapshot(
        children: &[StagedPieceChild],
        known_bag_ids: &[Option<BagNodeId>],
        snapshot: &OptimisticSnapshot,
    ) -> Option<(u32, OptimisticActivationKind)> {
        let probe_width = snapshot.probe_width.max(1);
        let activated = children
            .iter()
            .enumerate()
            .filter_map(|(idx, _child)| {
                known_bag_ids[idx]
                    .filter(|&bag_id| {
                        snapshot.live_won[bag_id as usize] && snapshot.activated[bag_id as usize]
                    })
                    .map(|_| idx)
            })
            .collect::<Vec<_>>();
        if let Some(&idx) = activated
            .iter()
            .take(activated.len().min(probe_width))
            .chain(activated.iter().skip(activated.len().min(probe_width)))
            .next()
        {
            return Some((idx as u32, OptimisticActivationKind::ActivatedReuse));
        }

        let dormant = children
            .iter()
            .enumerate()
            .filter_map(|(idx, _child)| match known_bag_ids[idx] {
                Some(bag_id) if snapshot.live_won[bag_id as usize] => {
                    Some((idx, OptimisticActivationKind::KnownReuse))
                }
                None => Some((idx, OptimisticActivationKind::NovelActivation)),
                _ => None,
            })
            .collect::<Vec<_>>();
        dormant
            .iter()
            .take(dormant.len().min(probe_width))
            .chain(dormant.iter().skip(dormant.len().min(probe_width)))
            .next()
            .copied()
            .map(|(idx, kind)| (idx as u32, kind))
    }

    fn compute_optimistic_expand_proposals(
        &self,
        expand_results: Vec<ExpandResult>,
        snapshot: &OptimisticSnapshot,
    ) -> Vec<OptimisticExpandProposal> {
        let proposal_one = |result: ExpandResult| {
            let key = result.key;
            OptimisticExpandProposal {
                bag_id: result.bag_id,
                key,
                branches: result
                    .branches
                    .into_iter()
                    .map(|branch| {
                        let next_bag = next_bag_state(key.bag, branch.piece)
                            .expect("piece should exist in bag");
                        let mut children = branch
                            .geometry
                            .iter()
                            .copied()
                            .map(|child| StagedPieceChild {
                                placement: child.placement,
                                succ_key: BagKey {
                                    board: child.board,
                                    bag: next_bag,
                                },
                            })
                            .collect::<Vec<_>>();
                        children.sort_by_key(|child| {
                            let features = self.board_features(child.succ_key.board);
                            (
                                features.height,
                                features.holes,
                                features.cells,
                                features.roughness,
                                features.height_spread,
                                child.placement,
                            )
                        });
                        let ranked = self
                            .ranked_placements_for(key.board, branch.piece)
                            .unwrap_or_default();
                        let mut known_bag_ids = children
                            .iter()
                            .map(|child| self.bag_map.get(&child.succ_key).copied())
                            .collect::<Vec<_>>();
                        let popularity_reordered =
                            Self::reorder_staged_children_optimistically_with_snapshot(
                                Some(&ranked),
                                snapshot,
                                &known_bag_ids,
                                &mut children,
                            );
                        known_bag_ids = children
                            .iter()
                            .map(|child| self.bag_map.get(&child.succ_key).copied())
                            .collect::<Vec<_>>();
                        let active_slot =
                            Self::choose_staged_optimistic_child_slot_strict_reuse_with_snapshot(
                                &children,
                                &known_bag_ids,
                                snapshot,
                            );
                        let unique_succ_count = children
                            .iter()
                            .map(|child| child.succ_key)
                            .collect::<HashSet<_>>()
                            .len() as u32;
                        OptimisticExpandBranchProposal {
                            piece: branch.piece,
                            geometry: branch.geometry,
                            children,
                            unique_succ_count,
                            active_slot: active_slot.map(|(slot, _)| slot),
                            activation_kind: active_slot.map(|(_, kind)| kind),
                            popularity_reordered,
                        }
                    })
                    .collect(),
            }
        };

        if self.parallel_local_expansion {
            expand_results.into_par_iter().map(proposal_one).collect()
        } else {
            expand_results.into_iter().map(proposal_one).collect()
        }
    }

    fn collect_optimistic_repair_batch(&mut self) -> Vec<PieceNodeId> {
        let mut batch = Vec::with_capacity(Self::EXPAND_BATCH_SIZE);
        while batch.len() < Self::EXPAND_BATCH_SIZE {
            let Some(piece_id) = self.optimistic_repair_queue.pop_front() else {
                break;
            };
            self.ensure_optimistic_children(piece_id);
            self.piece_nodes[piece_id as usize].optimistic_repair_pending = false;
            if self.piece_nodes[piece_id as usize].resolution == NodeResolution::Lost {
                continue;
            }
            batch.push(piece_id);
        }
        batch
    }

    fn compute_optimistic_repair_proposals(
        &self,
        batch: Vec<PieceNodeId>,
        snapshot: &OptimisticSnapshot,
    ) -> Vec<OptimisticRepairProposal> {
        let proposal_one = |piece_id: PieceNodeId| {
            let piece = &self.piece_nodes[piece_id as usize];
            let child_len = piece.optimistic_children.len();
            let cursor = (piece.next_repair_cursor as usize).min(child_len);
            let mut scanned = 0usize;
            let replacement = (cursor..child_len)
                .chain(0..cursor)
                .find_map(|idx| {
                    let child = piece.optimistic_children[idx];
                    scanned += 1;
                    match child.target {
                        OptimisticChildTarget::Activated(succ)
                            if snapshot.live_won[succ as usize]
                                && snapshot.activated[succ as usize] =>
                        {
                            Some((
                                idx as u32,
                                child.placement,
                                None,
                                Some(succ),
                                OptimisticActivationKind::ActivatedReuse,
                            ))
                        }
                        _ => None,
                    }
                })
                .or_else(|| {
                    (cursor..child_len).chain(0..cursor).find_map(|idx| {
                        let child = piece.optimistic_children[idx];
                        scanned += 1;
                        match child.target {
                            OptimisticChildTarget::Activated(succ)
                                if snapshot.live_won[succ as usize] =>
                            {
                                Some((
                                    idx as u32,
                                    child.placement,
                                    None,
                                    Some(succ),
                                    OptimisticActivationKind::KnownReuse,
                                ))
                            }
                            OptimisticChildTarget::Dormant(succ_key)
                                if snapshot.known_bags.contains(&succ_key) =>
                            {
                                Some((
                                    idx as u32,
                                    child.placement,
                                    Some(succ_key),
                                    None,
                                    OptimisticActivationKind::KnownReuse,
                                ))
                            }
                            OptimisticChildTarget::Dormant(succ_key) => Some((
                                idx as u32,
                                child.placement,
                                Some(succ_key),
                                None,
                                OptimisticActivationKind::NovelActivation,
                            )),
                            _ => None,
                        }
                    })
                });
            if let Some((slot, placement, succ_key, succ_id, kind)) = replacement {
                OptimisticRepairProposal {
                    piece_id,
                    slot: Some(slot),
                    placement: Some(placement),
                    succ_key,
                    succ_id,
                    activation_kind: Some(kind),
                    scanned,
                }
            } else {
                OptimisticRepairProposal {
                    piece_id,
                    slot: None,
                    placement: None,
                    succ_key: None,
                    succ_id: None,
                    activation_kind: None,
                    scanned,
                }
            }
        };

        if self.parallel_local_expansion {
            batch.into_par_iter().map(proposal_one).collect()
        } else {
            batch.into_iter().map(proposal_one).collect()
        }
    }

    fn enqueue_optimistic_repair(&mut self, piece_id: PieceNodeId) {
        let piece = &mut self.piece_nodes[piece_id as usize];
        if piece.optimistic_repair_pending || piece.resolution == NodeResolution::Lost {
            return;
        }
        piece.optimistic_repair_pending = true;
        self.optimistic_repair_queue.push_back(piece_id);
    }

    fn solve(mut self) -> Result<ProofSolveResult> {
        let total_start = Instant::now();
        let discover_start = Instant::now();
        self.discover_and_propagate();
        self.select_witnesses()?;
        let total_secs = total_start.elapsed().as_secs_f64();
        let discover_secs = discover_start.elapsed().as_secs_f64();

        let metrics = self.compute_metrics();
        let root_result = self.root_result();
        Ok(ProofSolveResult {
            solve_mode: self.solve_mode,
            used_full_fallback: self.used_full_fallback,
            root_bag: self.root_bag,
            bag_nodes: self.materialized_bag_nodes(),
            piece_nodes: self.piece_nodes,
            metrics,
            discover_secs,
            propagate_secs: 0.0,
            total_secs,
            root_result,
        })
    }

    fn solve_core_first(config: SolverConfig, options: SolveOptions) -> Result<ProofSolveResult> {
        let total_start = Instant::now();
        let mut solver =
            Self::new_with_mode(config, SolveMode::CoreFirst, FrontierMode::StableFirst);
        let discover_start = Instant::now();
        if let Some(result) = solver.discover_core_first(
            options.core_prepass_expansions,
            options.core_certify_interval.max(1),
            total_start,
            discover_start,
        )? {
            return Ok(result);
        }

        let mut fallback =
            Self::new_with_mode(config, SolveMode::CoreFirst, FrontierMode::FragileFirst);
        fallback.used_full_fallback = true;
        fallback.solve()
    }

    fn solve_template_kernel(
        config: SolverConfig,
        options: SolveOptions,
    ) -> Result<ProofSolveResult> {
        let total_start = Instant::now();
        let mut solver =
            Self::new_with_mode(config, SolveMode::TemplateKernel, FrontierMode::StableFirst);
        let discover_start = Instant::now();
        solver.discover_sample(options.template_sample_expansions);
        solver.build_template_kernel_result(total_start, discover_start, options)
    }

    fn solve_optimistic(config: SolverConfig, options: SolveOptions) -> Result<ProofSolveResult> {
        let total_start = Instant::now();
        let discover_start = Instant::now();
        let mut solver =
            Self::new_with_mode(config, SolveMode::Optimistic, FrontierMode::StableFirst);
        solver.optimistic_probe_width = options.optimistic_probe_width.max(1);
        let completed_optimistically = solver.discover_optimistic(options);
        if !completed_optimistically && options.optimistic_fallback_on_thrash {
            let mut fallback =
                Self::new_with_mode(config, SolveMode::Optimistic, FrontierMode::FragileFirst);
            fallback.used_full_fallback = true;
            fallback.optimistic_fallback_triggered = true;
            fallback.solve()
        } else {
            solver.complete_optimistic_exact()?;
            solver.select_witnesses()?;
            let metrics = solver.compute_metrics();
            let root_result = solver.root_result();
            Ok(ProofSolveResult {
                solve_mode: solver.solve_mode,
                used_full_fallback: solver.used_full_fallback,
                root_bag: solver.root_bag,
                bag_nodes: solver.materialized_bag_nodes(),
                piece_nodes: solver.piece_nodes,
                metrics,
                discover_secs: discover_start.elapsed().as_secs_f64(),
                propagate_secs: 0.0,
                total_secs: total_start.elapsed().as_secs_f64(),
                root_result,
            })
        }
    }

    fn complete_optimistic_exact(&mut self) -> Result<()> {
        if !self.is_optimistic() {
            return Ok(());
        }

        for piece_id in 0..self.piece_nodes.len() {
            self.ensure_optimistic_children(piece_id as PieceNodeId);
            let mut newly_interned = Vec::new();
            for child in &self.piece_nodes[piece_id].optimistic_children {
                if let OptimisticChildTarget::Dormant(succ_key) = child.target
                    && !self.bag_map.contains_key(&succ_key)
                {
                    newly_interned.push(succ_key);
                }
            }
            for succ_key in newly_interned {
                let succ = self.intern_bag(succ_key);
                self.activate_bag_if_needed(succ);
            }
            let resolved = self.piece_nodes[piece_id]
                .optimistic_children
                .iter()
                .enumerate()
                .filter_map(|(idx, child)| {
                    if let OptimisticChildTarget::Dormant(succ_key) = child.target {
                        self.bag_map.get(&succ_key).copied().map(|succ| (idx, succ))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            for &(_, succ) in &resolved {
                self.activate_bag_if_needed(succ);
            }
            for (idx, succ) in resolved {
                self.piece_nodes[piece_id].optimistic_children[idx].target =
                    OptimisticChildTarget::Activated(succ);
            }
        }

        let options = SolveOptions {
            mode: SolveMode::Optimistic,
            optimistic_max_repairs_per_piece: usize::MAX,
            optimistic_max_global_repairs: usize::MAX,
            optimistic_probe_width: self.optimistic_probe_width.max(1),
            optimistic_fallback_on_thrash: false,
            ..SolveOptions::default()
        };
        let _ = self.discover_optimistic(options);

        for piece_id in 0..self.piece_nodes.len() {
            self.ensure_optimistic_children(piece_id as PieceNodeId);
            let children = self.piece_nodes[piece_id]
                .optimistic_children
                .iter()
                .filter_map(|child| {
                    if let OptimisticChildTarget::Activated(succ) = child.target {
                        Some(PieceChild {
                            placement: child.placement,
                            succ,
                        })
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            self.piece_nodes[piece_id].children = children;
        }
        Ok(())
    }

    fn discover_and_propagate(&mut self) {
        const LOG_EVERY: Duration = Duration::from_secs(2);
        let start = Instant::now();
        let mut last_log = Instant::now();
        let mut last_bag_count = 0usize;
        let mut last_piece_count = 0usize;
        let mut last_edge_count = 0usize;
        let mut last_hits = 0usize;
        let mut last_misses = 0usize;
        let mut last_expanded = 0usize;
        let mut last_skipped = 0usize;
        let mut last_lost_bags = 0usize;
        let mut last_lost_pieces = 0usize;
        let mut last_frontier = self.expand_queue.len();
        let mut last_local_expand_secs = 0.0f64;
        let mut last_merge_secs = 0.0f64;
        let mut last_stage_secs = 0.0f64;
        let mut last_commit_secs = 0.0f64;
        let mut last_geometry_hits = 0usize;
        let mut last_geometry_misses = 0usize;
        let mut last_board_feature_hits = 0usize;
        let mut last_board_feature_misses = 0usize;

        while !self.expand_queue.is_empty()
            || !self.dead_bag_queue.is_empty()
            || !self.dead_piece_queue.is_empty()
        {
            self.drain_death_queues();
            let batch = self.collect_expand_batch();
            if batch.is_empty() {
                continue;
            }

            let local_expand_start = Instant::now();
            let expand_results = self.compute_expand_results(batch);
            self.total_local_expand_secs += local_expand_start.elapsed().as_secs_f64();

            let merge_start = Instant::now();
            if self.staged_merge {
                let stage_start = Instant::now();
                let staged_results = self.stage_expand_results(expand_results);
                self.total_stage_secs += stage_start.elapsed().as_secs_f64();

                let commit_start = Instant::now();
                self.apply_staged_batch(staged_results);
                self.total_commit_secs += commit_start.elapsed().as_secs_f64();
            } else {
                for result in expand_results {
                    self.apply_expand_result(result);
                }
                self.total_commit_secs += merge_start.elapsed().as_secs_f64();
            }
            self.drain_death_queues();
            self.total_merge_secs += merge_start.elapsed().as_secs_f64();
            self.parallel_batches_processed += 1;

            if last_log.elapsed() >= LOG_EVERY {
                let elapsed = start.elapsed().as_secs_f64().max(1e-9);
                let log_elapsed = last_log.elapsed().as_secs_f64().max(1e-9);
                let delta_bags = self.bag_nodes.len().saturating_sub(last_bag_count);
                let delta_pieces = self.piece_nodes.len().saturating_sub(last_piece_count);
                let delta_edges = self.deduped_edges.saturating_sub(last_edge_count);
                let delta_hits = self.cache_hits.saturating_sub(last_hits);
                let delta_misses = self.cache_misses.saturating_sub(last_misses);
                let delta_expanded = self.bags_expanded.saturating_sub(last_expanded);
                let delta_skipped = self.skipped_dead_bags.saturating_sub(last_skipped);
                let lost_bags = self
                    .bag_nodes
                    .iter()
                    .filter(|node| node.resolution == NodeResolution::Lost)
                    .count();
                let lost_pieces = self
                    .piece_nodes
                    .iter()
                    .filter(|node| node.resolution == NodeResolution::Lost)
                    .count();
                let delta_lost_bags = lost_bags.saturating_sub(last_lost_bags);
                let delta_lost_pieces = lost_pieces.saturating_sub(last_lost_pieces);
                let delta_local_expand_secs = self.total_local_expand_secs - last_local_expand_secs;
                let delta_merge_secs = self.total_merge_secs - last_merge_secs;
                let delta_stage_secs = self.total_stage_secs - last_stage_secs;
                let delta_commit_secs = self.total_commit_secs - last_commit_secs;
                let geometry_hits = self.geometry_cache_hits.load(AtomicOrdering::Relaxed);
                let geometry_misses = self.geometry_cache_misses.load(AtomicOrdering::Relaxed);
                let board_feature_hits =
                    self.board_feature_cache_hits.load(AtomicOrdering::Relaxed);
                let board_feature_misses = self
                    .board_feature_cache_misses
                    .load(AtomicOrdering::Relaxed);
                let delta_geometry_hits = geometry_hits.saturating_sub(last_geometry_hits);
                let delta_geometry_misses = geometry_misses.saturating_sub(last_geometry_misses);
                let delta_board_feature_hits =
                    board_feature_hits.saturating_sub(last_board_feature_hits);
                let delta_board_feature_misses =
                    board_feature_misses.saturating_sub(last_board_feature_misses);
                let frontier = self.expand_queue.len();
                let frontier_delta = frontier as isize - last_frontier as isize;
                let frontier_gap = self.bag_nodes.len().saturating_sub(self.bags_expanded);
                eprintln!(
                    "[success-set:discover] elapsed={elapsed:.1}s bag_nodes={} piece_nodes={} deduped_edges={} frontier={} frontier_delta={} frontier_gap={} dead_bag_queue={} dead_piece_queue={} expanded={} skipped_dead={} lost_bags={} lost_pieces={} new_bags={} new_pieces={} new_deduped_edges={} new_lost_bags={} new_lost_pieces={} expand_rate={:.1}/s skip_rate={:.1}/s bag_rate={:.1}/s piece_rate={:.1}/s edge_rate={:.1}/s bag_loss_rate={:.1}/s piece_loss_rate={:.1}/s local_expand_secs={:.3} stage_secs={:.3} commit_secs={:.3} merge_secs={:.3} cache_hit_rate={:.2} geometry_cache_hit_rate={:.2} board_feature_cache_hit_rate={:.2} branches_expanded={} branches_skipped_due_to_parent_death={} bags_killed_by_first_failed_branch={} avg_branches_expanded_per_bag={:.2} staged_branches_processed={} batch_unique_piece_keys={} batch_unique_bag_keys={} resolved_children_processed={} batched_parent_links_appended={} parent_segments_allocated={} parent_links_stored={} avg_parent_segments_per_bag={:.2} new_geometry_cache_hits={} new_geometry_cache_misses={} new_board_feature_cache_hits={} new_board_feature_cache_misses={} approx_mem_lower_bound={:.1}MB",
                    self.bag_nodes.len(),
                    self.piece_nodes.len(),
                    self.deduped_edges,
                    frontier,
                    frontier_delta,
                    frontier_gap,
                    self.dead_bag_queue.len(),
                    self.dead_piece_queue.len(),
                    self.bags_expanded,
                    self.skipped_dead_bags,
                    lost_bags,
                    lost_pieces,
                    delta_bags,
                    delta_pieces,
                    delta_edges,
                    delta_lost_bags,
                    delta_lost_pieces,
                    delta_expanded as f64 / log_elapsed,
                    delta_skipped as f64 / log_elapsed,
                    delta_bags as f64 / log_elapsed,
                    delta_pieces as f64 / log_elapsed,
                    delta_edges as f64 / log_elapsed,
                    delta_lost_bags as f64 / log_elapsed,
                    delta_lost_pieces as f64 / log_elapsed,
                    delta_local_expand_secs,
                    delta_stage_secs,
                    delta_commit_secs,
                    delta_merge_secs,
                    delta_hits as f64 / (delta_hits + delta_misses).max(1) as f64,
                    delta_geometry_hits as f64
                        / (delta_geometry_hits + delta_geometry_misses).max(1) as f64,
                    delta_board_feature_hits as f64
                        / (delta_board_feature_hits + delta_board_feature_misses).max(1) as f64,
                    self.branches_expanded,
                    self.branches_skipped_due_to_parent_death,
                    self.bags_killed_by_first_failed_branch,
                    if self.bags_expanded == 0 {
                        0.0
                    } else {
                        self.branches_expanded as f64 / self.bags_expanded as f64
                    },
                    self.staged_branches_processed,
                    self.batch_unique_piece_keys,
                    self.batch_unique_bag_keys,
                    self.resolved_children_processed,
                    self.batched_parent_links_appended,
                    self.parent_segments.len(),
                    self.parent_edges.len(),
                    if self.bag_nodes.is_empty() {
                        0.0
                    } else {
                        self.parent_segments.len() as f64 / self.bag_nodes.len() as f64
                    },
                    delta_geometry_hits,
                    delta_geometry_misses,
                    delta_board_feature_hits,
                    delta_board_feature_misses,
                    self.approx_bytes_lower_bound() as f64 / (1024.0 * 1024.0),
                );
                let _ = std::io::stderr().flush();
                last_log = Instant::now();
                last_bag_count = self.bag_nodes.len();
                last_piece_count = self.piece_nodes.len();
                last_edge_count = self.deduped_edges;
                last_hits = self.cache_hits;
                last_misses = self.cache_misses;
                last_expanded = self.bags_expanded;
                last_skipped = self.skipped_dead_bags;
                last_lost_bags = lost_bags;
                last_lost_pieces = lost_pieces;
                last_frontier = frontier;
                last_local_expand_secs = self.total_local_expand_secs;
                last_merge_secs = self.total_merge_secs;
                last_stage_secs = self.total_stage_secs;
                last_commit_secs = self.total_commit_secs;
                last_geometry_hits = geometry_hits;
                last_geometry_misses = geometry_misses;
                last_board_feature_hits = board_feature_hits;
                last_board_feature_misses = board_feature_misses;
            }
        }

        self.drain_death_queues();
    }

    fn discover_optimistic(&mut self, options: SolveOptions) -> bool {
        const LOG_EVERY: Duration = Duration::from_secs(2);
        let start = Instant::now();
        let mut last_log = Instant::now();
        let mut last_repairs_attempted = 0usize;
        let mut last_repairs_succeeded = 0usize;
        let mut last_repairs_failed = 0usize;

        while !self.expand_queue.is_empty()
            || !self.dead_bag_queue.is_empty()
            || !self.dead_piece_queue.is_empty()
            || !self.optimistic_repair_queue.is_empty()
        {
            self.drain_death_queues();
            if !self.process_optimistic_repairs(options) {
                return false;
            }

            let batch = self.collect_expand_batch();
            if !batch.is_empty() {
                let local_expand_start = Instant::now();
                let expand_results = self.compute_expand_results(batch);
                self.total_local_expand_secs += local_expand_start.elapsed().as_secs_f64();

                let merge_start = Instant::now();
                if self.staged_merge {
                    let stage_start = Instant::now();
                    let staged_results = self.stage_expand_results(expand_results);
                    self.total_stage_secs += stage_start.elapsed().as_secs_f64();

                    let commit_start = Instant::now();
                    self.apply_staged_batch(staged_results);
                    self.total_commit_secs += commit_start.elapsed().as_secs_f64();
                } else {
                    let proposal_start = Instant::now();
                    let snapshot = self.optimistic_snapshot();
                    let proposals =
                        self.compute_optimistic_expand_proposals(expand_results, &snapshot);
                    self.optimistic_expand_proposal_secs += proposal_start.elapsed().as_secs_f64();
                    for proposal in proposals {
                        self.apply_optimistic_expand_proposal(proposal);
                    }
                    self.total_commit_secs += merge_start.elapsed().as_secs_f64();
                }
                self.drain_death_queues();
                if !self.process_optimistic_repairs(options) {
                    return false;
                }
                self.total_merge_secs += merge_start.elapsed().as_secs_f64();
                self.parallel_batches_processed += 1;
            }

            if last_log.elapsed() >= LOG_EVERY {
                let delta_repairs_attempted =
                    self.optimistic_repairs_attempted - last_repairs_attempted;
                let delta_repairs_succeeded =
                    self.optimistic_repairs_succeeded - last_repairs_succeeded;
                let delta_repairs_failed = self.optimistic_repairs_failed - last_repairs_failed;
                let lost_bags = self
                    .bag_nodes
                    .iter()
                    .filter(|node| node.resolution == NodeResolution::Lost)
                    .count();
                let live_bags = self.bag_nodes.len().saturating_sub(lost_bags);
                let (activated_live_bags, dormant_child_targets, activated_frontier) =
                    self.optimistic_activation_metrics();
                let nonzero_popularity = self
                    .optimistic_successor_popularity
                    .iter()
                    .copied()
                    .filter(|&value| value > 0)
                    .collect::<Vec<_>>();
                eprintln!(
                    "[success-set:optimistic] elapsed={:.1}s live_bags={} frontier={} activated_live_bags={} dormant_child_targets={} activated_frontier={} novel_activations={} reuse_activations={} repairs_reusing_known_successor={} repairs_forcing_novel_activation={} repairs_attempted={} repairs_succeeded={} repairs_failed={} policy_entries={} policy_hits={} policy_misses={} repairs_promoted={} repairs_demoted={} reused_ranked_candidates={} avg_candidates_scanned_per_repair={:.3} avg_rank_position_chosen={:.3} active_parent_links={} popularity_nonzero_bags={} max_successor_popularity={} avg_successor_popularity={:.3} reorders_using_popularity={} max_repairs_on_single_piece={} fallback_triggered={} expand_proposal_secs={:.3} repair_proposal_secs={:.3} repair_batches_processed={} expand_proposals_revalidated={} repair_proposals_revalidated={}",
                    start.elapsed().as_secs_f64(),
                    live_bags,
                    self.expand_queue.len(),
                    activated_live_bags,
                    dormant_child_targets,
                    activated_frontier,
                    self.optimistic_novel_activations,
                    self.optimistic_reuse_activations,
                    self.optimistic_repairs_reusing_known_successor,
                    self.optimistic_repairs_forcing_novel_activation,
                    delta_repairs_attempted,
                    delta_repairs_succeeded,
                    delta_repairs_failed,
                    self.optimistic_policy.len(),
                    self.optimistic_policy_hits,
                    self.optimistic_policy_misses,
                    self.optimistic_repairs_promoted,
                    self.optimistic_repairs_demoted,
                    self.optimistic_reused_ranked_candidates,
                    if self.optimistic_repairs_attempted == 0 {
                        0.0
                    } else {
                        self.optimistic_repair_candidates_scanned as f64
                            / self.optimistic_repairs_attempted as f64
                    },
                    if self.optimistic_rank_position_count == 0 {
                        0.0
                    } else {
                        self.optimistic_rank_position_sum as f64
                            / self.optimistic_rank_position_count as f64
                    },
                    self.optimistic_active_parent_links,
                    nonzero_popularity.len(),
                    nonzero_popularity.iter().copied().max().unwrap_or(0),
                    if nonzero_popularity.is_empty() {
                        0.0
                    } else {
                        nonzero_popularity.iter().sum::<usize>() as f64
                            / nonzero_popularity.len() as f64
                    },
                    self.optimistic_reorders_using_popularity,
                    self.optimistic_max_repairs_on_single_piece,
                    self.optimistic_fallback_triggered,
                    self.optimistic_expand_proposal_secs,
                    self.optimistic_repair_proposal_secs,
                    self.optimistic_repair_batches_processed,
                    self.optimistic_expand_proposals_revalidated,
                    self.optimistic_repair_proposals_revalidated,
                );
                last_log = Instant::now();
                last_repairs_attempted = self.optimistic_repairs_attempted;
                last_repairs_succeeded = self.optimistic_repairs_succeeded;
                last_repairs_failed = self.optimistic_repairs_failed;
            }
        }

        true
    }

    fn process_optimistic_repairs(&mut self, options: SolveOptions) -> bool {
        while !self.optimistic_repair_queue.is_empty() {
            let batch = self.collect_optimistic_repair_batch();
            if batch.is_empty() {
                continue;
            }
            for &piece_id in &batch {
                self.optimistic_repairs_attempted += 1;
                self.piece_nodes[piece_id as usize].repair_attempts = self.piece_nodes
                    [piece_id as usize]
                    .repair_attempts
                    .saturating_add(1);
                self.optimistic_max_repairs_on_single_piece = self
                    .optimistic_max_repairs_on_single_piece
                    .max(self.piece_nodes[piece_id as usize].repair_attempts as usize);
                if (self.piece_nodes[piece_id as usize].repair_attempts as usize)
                    > options.optimistic_max_repairs_per_piece
                    || self.optimistic_repairs_attempted > options.optimistic_max_global_repairs
                {
                    self.optimistic_fallback_triggered = true;
                    return false;
                }
            }
            let proposal_start = Instant::now();
            let snapshot = self.optimistic_snapshot();
            let proposals = self.compute_optimistic_repair_proposals(batch, &snapshot);
            self.optimistic_repair_proposal_secs += proposal_start.elapsed().as_secs_f64();
            self.optimistic_repair_batches_processed += 1;
            for proposal in proposals {
                self.apply_optimistic_repair_proposal(proposal);
            }
        }

        true
    }

    fn discover_sample(&mut self, max_expansions: usize) {
        while (!self.expand_queue.is_empty()
            || !self.dead_bag_queue.is_empty()
            || !self.dead_piece_queue.is_empty())
            && self.bags_expanded < max_expansions
        {
            self.drain_death_queues();
            let batch = self.collect_expand_batch();
            if batch.is_empty() {
                continue;
            }

            let local_expand_start = Instant::now();
            let expand_results = self.compute_expand_results(batch);
            self.total_local_expand_secs += local_expand_start.elapsed().as_secs_f64();

            let merge_start = Instant::now();
            if self.staged_merge {
                let stage_start = Instant::now();
                let staged_results = self.stage_expand_results(expand_results);
                self.total_stage_secs += stage_start.elapsed().as_secs_f64();

                let commit_start = Instant::now();
                self.apply_staged_batch(staged_results);
                self.total_commit_secs += commit_start.elapsed().as_secs_f64();
            } else {
                for result in expand_results {
                    self.apply_expand_result(result);
                }
                self.total_commit_secs += merge_start.elapsed().as_secs_f64();
            }
            self.drain_death_queues();
            self.total_merge_secs += merge_start.elapsed().as_secs_f64();
            self.parallel_batches_processed += 1;
        }

        self.drain_death_queues();
    }

    fn discover_core_first(
        &mut self,
        max_expansions: usize,
        certify_interval: usize,
        total_start: Instant,
        discover_start: Instant,
    ) -> Result<Option<ProofSolveResult>> {
        let mut since_last_cert = 0usize;
        while (!self.expand_queue.is_empty()
            || !self.dead_bag_queue.is_empty()
            || !self.dead_piece_queue.is_empty())
            && self.bags_expanded < max_expansions
        {
            self.drain_death_queues();
            let batch = self.collect_expand_batch();
            if batch.is_empty() {
                continue;
            }

            let local_expand_start = Instant::now();
            let expand_results = self.compute_expand_results(batch);
            self.total_local_expand_secs += local_expand_start.elapsed().as_secs_f64();

            let merge_start = Instant::now();
            if self.staged_merge {
                let stage_start = Instant::now();
                let staged_results = self.stage_expand_results(expand_results);
                self.total_stage_secs += stage_start.elapsed().as_secs_f64();

                let commit_start = Instant::now();
                self.apply_staged_batch(staged_results);
                self.total_commit_secs += commit_start.elapsed().as_secs_f64();
            } else {
                for result in expand_results {
                    self.apply_expand_result(result);
                }
                self.total_commit_secs += merge_start.elapsed().as_secs_f64();
            }
            self.drain_death_queues();
            self.total_merge_secs += merge_start.elapsed().as_secs_f64();
            self.parallel_batches_processed += 1;
            since_last_cert += Self::EXPAND_BATCH_SIZE;

            if since_last_cert >= certify_interval {
                since_last_cert = 0;
                if let Some(result) =
                    self.try_build_core_first_result(total_start, discover_start)?
                {
                    return Ok(Some(result));
                }
            }
        }

        self.drain_death_queues();
        Ok(self.try_build_core_first_result(total_start, discover_start)?)
    }

    fn collect_expand_batch(&mut self) -> Vec<ExpandTask> {
        let mut batch = Vec::with_capacity(Self::EXPAND_BATCH_SIZE);
        while batch.len() < Self::EXPAND_BATCH_SIZE {
            let Some(entry) = self.expand_queue.pop() else {
                break;
            };
            let bag_id = entry.bag_id;
            let bag_key = self.bag_nodes[bag_id as usize].key;
            let bag_resolution = self.bag_nodes[bag_id as usize].resolution;
            let bag_expanded = self.bag_nodes[bag_id as usize].expanded;
            if bag_resolution == NodeResolution::Lost {
                self.skipped_dead_bags += 1;
                continue;
            }
            if bag_expanded {
                continue;
            }

            let branches = if self.lazy_branch_expansion {
                let (board, bag_state, needs_init) = {
                    let bag = &self.bag_nodes[bag_id as usize];
                    (bag.key.board, bag.key.bag, !bag.branch_order_initialized)
                };
                if needs_init {
                    let (branch_order, branch_order_len) =
                        self.compute_branch_order(board, bag_state);
                    let bag = &mut self.bag_nodes[bag_id as usize];
                    bag.branch_order = branch_order;
                    bag.branch_order_len = branch_order_len;
                    bag.branch_order_initialized = true;
                }

                let maybe_piece = {
                    let bag = &self.bag_nodes[bag_id as usize];
                    (bag.next_branch_idx < bag.branch_order_len).then_some(TetrisPiece::from_index(
                        bag.branch_order[bag.next_branch_idx as usize],
                    ))
                };
                let Some(piece) = maybe_piece else {
                    self.bag_nodes[bag_id as usize].expanded = true;
                    continue;
                };
                vec![ExpandTaskBranch {
                    piece,
                    geometry: self
                        .geometry_cache
                        .get(&GeometryKey { board, piece })
                        .map(|entry| {
                            self.geometry_cache_hits
                                .fetch_add(1, AtomicOrdering::Relaxed);
                            Arc::clone(entry.value())
                        }),
                }]
            } else {
                piece_branches(bag_key.bag)
                    .map(|branch| ExpandTaskBranch {
                        piece: branch.piece,
                        geometry: self
                            .geometry_cache
                            .get(&GeometryKey {
                                board: bag_key.board,
                                piece: branch.piece,
                            })
                            .map(|entry| {
                                self.geometry_cache_hits
                                    .fetch_add(1, AtomicOrdering::Relaxed);
                                Arc::clone(entry.value())
                            }),
                    })
                    .collect::<Vec<_>>()
            };
            batch.push(ExpandTask {
                bag_id,
                key: bag_key,
                branches,
            });
        }
        batch
    }

    fn compute_branch_order(
        &mut self,
        board: TetrisBoard,
        bag: TetrisPieceBagState,
    ) -> ([u8; 7], u8) {
        let mut scored = piece_branches(bag)
            .map(|branch| {
                let geometry = self.geometry_children(board, branch.piece);
                let branch_features = geometry
                    .iter()
                    .map(|child| self.board_features(child.board))
                    .min_by_key(|features| {
                        (
                            features.height,
                            features.holes,
                            features.cells,
                            features.roughness,
                            features.height_spread,
                        )
                    })
                    .unwrap_or(BoardFeatures {
                        height: u32::MAX,
                        holes: u32::MAX,
                        cells: u32::MAX,
                        roughness: u32::MAX,
                        height_spread: u32::MAX,
                        turning_points: u32::MAX,
                        max_well_depth: u32::MAX,
                    });
                (
                    branch.piece.index(),
                    geometry.len(),
                    branch_features.height,
                    branch_features.holes,
                    branch_features.cells,
                    branch_features.roughness,
                    branch_features.height_spread,
                )
            })
            .collect::<Vec<_>>();
        scored.sort_unstable_by_key(
            |&(piece_idx, geom_len, height, holes, cells, roughness, spread)| {
                (geom_len, height, holes, cells, roughness, spread, piece_idx)
            },
        );

        let mut order = [u8::MAX; 7];
        for (idx, &(piece_idx, ..)) in scored.iter().enumerate() {
            order[idx] = piece_idx;
        }
        (order, scored.len() as u8)
    }

    fn compute_expand_results(&self, batch: Vec<ExpandTask>) -> Vec<ExpandResult> {
        let compute_one = |task: ExpandTask| ExpandResult {
            bag_id: task.bag_id,
            key: task.key,
            branches: task
                .branches
                .into_iter()
                .map(|branch| PieceExpansionResult {
                    piece: branch.piece,
                    geometry: branch.geometry.unwrap_or_else(|| {
                        self.geometry_cache_misses
                            .fetch_add(1, AtomicOrdering::Relaxed);
                        Arc::<[GeometryChild]>::from(Self::compute_geometry_children_for(
                            task.key.board,
                            branch.piece,
                            self.config.admissibility,
                        ))
                    }),
                })
                .collect(),
        };

        if self.parallel_local_expansion {
            batch.into_par_iter().map(compute_one).collect()
        } else {
            batch.into_iter().map(compute_one).collect()
        }
    }

    fn stage_expand_results(&self, expand_results: Vec<ExpandResult>) -> Vec<StagedExpandResult> {
        let stage_one = |result: ExpandResult| StagedExpandResult {
            bag_id: result.bag_id,
            key: result.key,
            branches: result
                .branches
                .into_iter()
                .map(|branch| {
                    let next_bag = next_bag_state(result.key.bag, branch.piece)
                        .expect("piece should exist in bag");
                    let mut children = branch
                        .geometry
                        .iter()
                        .copied()
                        .map(|child| StagedPieceChild {
                            placement: child.placement,
                            succ_key: BagKey {
                                board: child.board,
                                bag: next_bag,
                            },
                        })
                        .collect::<Vec<_>>();
                    children.sort_by_key(|child| {
                        let features = self.board_features(child.succ_key.board);
                        (
                            features.height,
                            features.holes,
                            features.cells,
                            features.roughness,
                            features.height_spread,
                            child.placement,
                        )
                    });
                    let mut seen_succs = HashSet::with_capacity(children.len());
                    let unique_succ_keys = children
                        .iter()
                        .filter_map(|child| {
                            seen_succs.insert(child.succ_key).then_some(child.succ_key)
                        })
                        .collect::<Vec<_>>();

                    StagedPieceExpansion {
                        piece: branch.piece,
                        geometry: branch.geometry,
                        piece_key: PieceKey {
                            board: result.key.board,
                            bag: result.key.bag,
                            piece: branch.piece,
                        },
                        children,
                        unique_succ_keys,
                    }
                })
                .collect(),
        };

        if self.parallel_local_expansion {
            expand_results.into_par_iter().map(stage_one).collect()
        } else {
            expand_results.into_iter().map(stage_one).collect()
        }
    }

    fn apply_staged_batch(&mut self, staged_results: Vec<StagedExpandResult>) {
        let mut batch_piece_ids = HashMap::<PieceKey, PieceNodeId>::new();
        let mut batch_bag_ids = HashMap::<BagKey, BagNodeId>::new();

        for staged in &staged_results {
            for branch in &staged.branches {
                if let std::collections::hash_map::Entry::Vacant(entry) =
                    batch_piece_ids.entry(branch.piece_key)
                {
                    let piece_id = self.intern_piece(branch.piece_key, staged.bag_id);
                    entry.insert(piece_id);
                    self.batch_unique_piece_keys += 1;
                }
                for &succ_key in &branch.unique_succ_keys {
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        batch_bag_ids.entry(succ_key)
                    {
                        let bag_id = self.intern_bag(succ_key);
                        entry.insert(bag_id);
                        self.batch_unique_bag_keys += 1;
                    }
                }
            }
        }

        self.staged_branches_processed += staged_results
            .iter()
            .map(|r| r.branches.len())
            .sum::<usize>();
        let resolved_results = staged_results
            .into_iter()
            .map(|staged| {
                let branches = staged
                    .branches
                    .into_iter()
                    .map(|branch| {
                        let piece_id = *batch_piece_ids
                            .get(&branch.piece_key)
                            .expect("staged piece key should be pre-interned");
                        let children = branch
                            .children
                            .iter()
                            .map(|child| PieceChild {
                                placement: child.placement,
                                succ: *batch_bag_ids
                                    .get(&child.succ_key)
                                    .expect("staged successor bag should be pre-interned"),
                            })
                            .collect::<Vec<_>>();
                        let live_unique_succs = branch
                            .unique_succ_keys
                            .iter()
                            .filter_map(|succ_key| {
                                let succ = *batch_bag_ids
                                    .get(succ_key)
                                    .expect("staged successor bag should be pre-interned");
                                (self.bag_nodes[succ as usize].resolution == NodeResolution::Won)
                                    .then_some(succ)
                            })
                            .collect::<Vec<_>>();
                        self.resolved_children_processed += children.len();
                        ResolvedPieceExpansion {
                            piece: branch.piece,
                            geometry: branch.geometry,
                            piece_id,
                            children,
                            live_unique_succs,
                            unique_succ_count: branch.unique_succ_keys.len() as u32,
                        }
                    })
                    .collect::<Vec<_>>();
                ResolvedExpandResult {
                    bag_id: staged.bag_id,
                    key: staged.key,
                    branches,
                }
            })
            .collect::<Vec<_>>();

        let mut parent_links_by_succ = HashMap::<BagNodeId, Vec<PieceNodeId>>::new();
        for resolved in resolved_results {
            self.apply_resolved_expand_result(&mut parent_links_by_succ, resolved);
        }
        for (succ, parents) in parent_links_by_succ {
            self.batched_parent_links_appended += parents.len();
            self.append_parent_links(succ, &parents);
        }
    }

    fn drain_death_queues(&mut self) {
        while !self.dead_piece_queue.is_empty() || !self.dead_bag_queue.is_empty() {
            while let Some(piece_id) = self.dead_piece_queue.pop_front() {
                let parent_bag = self.piece_nodes[piece_id as usize].parent_bag;
                self.kill_bag(parent_bag);
            }

            while let Some(bag_id) = self.dead_bag_queue.pop_front() {
                let mut segment = self.bag_parent_head[bag_id as usize];
                while let Some(segment_id) = segment {
                    let ParentSegment { start, len, next } =
                        self.parent_segments[segment_id as usize];
                    let start = start as usize;
                    let len = len as usize;
                    for edge_idx in start..start + len {
                        let piece_id = self.parent_edges[edge_idx];
                        if self.piece_nodes[piece_id as usize].resolution == NodeResolution::Lost {
                            continue;
                        }
                        if self.is_optimistic() {
                            self.ensure_optimistic_children(piece_id);
                            let active_succ = self.piece_nodes[piece_id as usize]
                                .active_child_slot
                                .and_then(|slot| self.optimistic_child_succ(piece_id, slot));
                            if active_succ == Some(bag_id) {
                                let piece = &self.piece_nodes[piece_id as usize];
                                if let Some(slot) = piece.active_child_slot
                                    && let Some(child) =
                                        piece.optimistic_children.get(slot as usize)
                                {
                                    let failed_placement = child.placement;
                                    self.demote_policy_placement(
                                        piece.key.board,
                                        piece.key.piece,
                                        failed_placement,
                                        true,
                                    );
                                }
                                self.optimistic_active_parent_links =
                                    self.optimistic_active_parent_links.saturating_sub(1);
                                self.decrement_optimistic_successor_popularity(bag_id);
                                self.enqueue_optimistic_repair(piece_id);
                            }
                        } else {
                            let piece = &mut self.piece_nodes[piece_id as usize];
                            piece.alive_succ_count = piece.alive_succ_count.saturating_sub(1);
                            if piece.alive_succ_count == 0 {
                                self.kill_piece(piece_id);
                            }
                        }
                    }
                    segment = next;
                }
            }
        }
    }

    fn kill_bag(&mut self, bag_id: BagNodeId) {
        if self.bag_nodes[bag_id as usize].resolution == NodeResolution::Lost {
            return;
        }
        let bag = &mut self.bag_nodes[bag_id as usize];
        bag.resolution = NodeResolution::Lost;
        if self.lazy_branch_expansion && !bag.expanded {
            let remaining = bag.branch_order_len.saturating_sub(bag.next_branch_idx);
            self.branches_skipped_due_to_parent_death += remaining as usize;
            if bag.expanded_branch_count == 1 {
                self.bags_killed_by_first_failed_branch += 1;
            }
            bag.expanded = true;
            self.bags_expanded += 1;
        }
        self.dead_bag_queue.push_back(bag_id);
    }

    fn kill_piece(&mut self, piece_id: PieceNodeId) {
        if self.piece_nodes[piece_id as usize].resolution == NodeResolution::Lost {
            return;
        }
        let piece = &mut self.piece_nodes[piece_id as usize];
        piece.resolution = NodeResolution::Lost;
        piece.best_action = None;
        piece.best_child = None;
        self.dead_piece_queue.push_back(piece_id);
    }

    fn select_witnesses(&mut self) -> Result<()> {
        let is_optimistic = self.is_optimistic();
        for piece_id in 0..self.piece_nodes.len() {
            let winning_unique_succs = self.piece_nodes[piece_id]
                .children
                .iter()
                .filter_map(|child| {
                    (self.bag_nodes[child.succ as usize].resolution == NodeResolution::Won)
                        .then_some(child.succ)
                })
                .collect::<HashSet<_>>();
            let unique_succ_count = self.piece_nodes[piece_id]
                .children
                .iter()
                .map(|child| child.succ)
                .collect::<HashSet<_>>()
                .len() as u32;

            let piece = &mut self.piece_nodes[piece_id];
            piece.unique_succ_count = unique_succ_count;
            piece.alive_succ_count = winning_unique_succs.len() as u32;
            if piece.resolution == NodeResolution::Lost {
                piece.best_action = None;
                piece.best_child = None;
                piece.active_child_slot = None;
                continue;
            }

            let mut best = None;
            if is_optimistic
                && let Some(slot) = piece.active_child_slot
                && let Some(child) = piece.children.get(slot as usize)
                && self.bag_nodes[child.succ as usize].resolution == NodeResolution::Won
                && self.bag_nodes[child.succ as usize].activated
            {
                best = Some((slot, child.placement, child.succ));
            }
            for (idx, child) in piece.children.iter().enumerate() {
                if best.is_some() {
                    break;
                }
                if self.bag_nodes[child.succ as usize].resolution == NodeResolution::Won {
                    best = Some((idx as u32, child.placement, child.succ));
                    break;
                }
            }
            if let Some((slot, placement, succ)) = best {
                piece.best_action = Some(placement);
                piece.best_child = Some(succ);
                piece.active_child_slot = Some(slot);
            } else {
                bail!(
                    "winning piece node {:?} has no surviving child during witness selection",
                    piece.key
                );
            }
        }
        Ok(())
    }

    fn intern_bag(&mut self, key: BagKey) -> BagNodeId {
        self.intern_bag_with_activation(key, true)
    }

    #[cfg(test)]
    fn intern_bag_dormant(&mut self, key: BagKey) -> BagNodeId {
        self.intern_bag_with_activation(key, false)
    }

    fn activate_bag_if_needed(&mut self, bag_id: BagNodeId) {
        if self.bag_nodes[bag_id as usize].activated {
            return;
        }
        self.bag_nodes[bag_id as usize].activated = true;
        if self.bag_nodes[bag_id as usize].resolution == NodeResolution::Lost
            || self.bag_nodes[bag_id as usize].expanded
        {
            return;
        }
        let score = self.expand_priority(self.bag_nodes[bag_id as usize].key.board);
        let sequence = self.next_expand_sequence;
        self.next_expand_sequence += 1;
        self.expand_queue.push(ExpandEntry {
            score,
            sequence,
            bag_id,
        });
    }

    fn intern_bag_with_activation(&mut self, key: BagKey, activate: bool) -> BagNodeId {
        if let Some(&id) = self.bag_map.get(&key) {
            self.cache_hits += 1;
            if activate {
                self.activate_bag_if_needed(id);
            }
            return id;
        }

        self.cache_misses += 1;
        let id = self.bag_nodes.len() as BagNodeId;
        self.bag_nodes.push(BagNode {
            key,
            resolution: NodeResolution::Won,
            expanded: false,
            parents: Vec::new(),
            piece_nodes: [None; 7],
            activated: activate,
            branch_order: [u8::MAX; 7],
            branch_order_len: 0,
            next_branch_idx: 0,
            expanded_branch_count: 0,
            branch_order_initialized: false,
        });
        self.bag_parent_head.push(None);
        self.bag_parent_tail.push(None);
        self.bag_parent_count.push(0);
        self.optimistic_successor_popularity.push(0);
        self.bag_map.insert(key, id);
        if activate {
            let score = self.expand_priority(key.board);
            let sequence = self.next_expand_sequence;
            self.next_expand_sequence += 1;
            self.expand_queue.push(ExpandEntry {
                score,
                sequence,
                bag_id: id,
            });
        }
        id
    }

    fn intern_piece(&mut self, key: PieceKey, parent_bag: BagNodeId) -> PieceNodeId {
        if let Some(&id) = self.piece_map.get(&key) {
            self.cache_hits += 1;
            return id;
        }

        self.cache_misses += 1;
        let id = self.piece_nodes.len() as PieceNodeId;
        self.piece_nodes.push(PieceNode {
            key,
            parent_bag,
            resolution: NodeResolution::Won,
            children: Vec::new(),
            optimistic_children: Vec::new(),
            unique_succ_count: 0,
            alive_succ_count: 0,
            best_action: None,
            best_child: None,
            active_child_slot: None,
            next_repair_cursor: 0,
            repair_attempts: 0,
            optimistic_exhausted: false,
            optimistic_repair_pending: false,
        });
        self.piece_map.insert(key, id);
        id
    }

    fn apply_resolved_expand_result(
        &mut self,
        parent_links_by_succ: &mut HashMap<BagNodeId, Vec<PieceNodeId>>,
        resolved: ResolvedExpandResult,
    ) {
        let bag_id = resolved.bag_id;
        if self.bag_nodes[bag_id as usize].expanded
            || self.bag_nodes[bag_id as usize].resolution == NodeResolution::Lost
        {
            return;
        }
        if self.lazy_branch_expansion {
            self.branches_expanded += resolved.branches.len();
        } else {
            self.bags_expanded += 1;
        }

        let mut piece_nodes = self.bag_nodes[bag_id as usize].piece_nodes;
        let expanded_before = self.bag_nodes[bag_id as usize].expanded_branch_count;
        let branch_order_len = self.bag_nodes[bag_id as usize].branch_order_len;

        for branch in resolved.branches {
            let piece_idx = branch.piece.index() as usize;
            let piece_id = branch.piece_id;
            piece_nodes[piece_idx] = Some(piece_id);

            if self.piece_nodes[piece_id as usize].children.is_empty()
                && self.piece_nodes[piece_id as usize]
                    .optimistic_children
                    .is_empty()
            {
                self.geometry_cache
                    .entry(GeometryKey {
                        board: resolved.key.board,
                        piece: branch.piece,
                    })
                    .or_insert_with(|| branch.geometry.clone());

                self.deduped_edges += branch.children.len();
                let mut children = branch.children;
                let locked_prefix = self.reorder_piece_children_optimistically(
                    resolved.key.board,
                    branch.piece,
                    &mut children,
                );
                self.reorder_piece_children_by_successor_popularity(&mut children, locked_prefix);
                self.piece_nodes[piece_id as usize].children = children;
                self.piece_nodes[piece_id as usize].unique_succ_count = branch.unique_succ_count;
                self.piece_nodes[piece_id as usize].alive_succ_count =
                    branch.live_unique_succs.len() as u32;
                for succ in branch.live_unique_succs {
                    parent_links_by_succ.entry(succ).or_default().push(piece_id);
                }
                if self.piece_nodes[piece_id as usize].alive_succ_count == 0 {
                    self.kill_piece(piece_id);
                }
            }
        }

        self.bag_nodes[bag_id as usize].piece_nodes = piece_nodes;
        if self.lazy_branch_expansion {
            let mut requeue_board = None;
            {
                let bag = &mut self.bag_nodes[bag_id as usize];
                bag.expanded_branch_count = bag.expanded_branch_count.saturating_add(1);
                bag.next_branch_idx = bag.next_branch_idx.saturating_add(1);
                if bag.resolution == NodeResolution::Lost {
                    let remaining = bag.branch_order_len.saturating_sub(bag.next_branch_idx);
                    self.branches_skipped_due_to_parent_death += remaining as usize;
                    if expanded_before == 0 {
                        self.bags_killed_by_first_failed_branch += 1;
                    }
                    bag.expanded = true;
                    self.bags_expanded += 1;
                } else if bag.expanded_branch_count >= branch_order_len {
                    bag.expanded = true;
                    self.bags_expanded += 1;
                } else {
                    requeue_board = Some(bag.key.board);
                }
            }
            if let Some(board) = requeue_board {
                let score = self.expand_priority(board);
                let sequence = self.next_expand_sequence;
                self.next_expand_sequence += 1;
                self.expand_queue.push(ExpandEntry {
                    score,
                    sequence,
                    bag_id,
                });
            }
        } else {
            self.bag_nodes[bag_id as usize].expanded = true;
        }
    }

    fn apply_expand_result(&mut self, result: ExpandResult) {
        let bag_id = result.bag_id;
        if self.bag_nodes[bag_id as usize].expanded
            || self.bag_nodes[bag_id as usize].resolution == NodeResolution::Lost
        {
            return;
        }
        if self.lazy_branch_expansion {
            self.branches_expanded += result.branches.len();
        } else {
            self.bags_expanded += 1;
        }

        let key = result.key;
        let mut piece_nodes = self.bag_nodes[bag_id as usize].piece_nodes;
        let expanded_before = self.bag_nodes[bag_id as usize].expanded_branch_count;
        let branch_order_len = self.bag_nodes[bag_id as usize].branch_order_len;

        for branch in result.branches {
            let piece_idx = branch.piece.index() as usize;
            let piece_key = PieceKey {
                board: key.board,
                bag: key.bag,
                piece: branch.piece,
            };
            let piece_id = self.intern_piece(piece_key, bag_id);
            piece_nodes[piece_idx] = Some(piece_id);

            if self.piece_nodes[piece_id as usize].children.is_empty()
                && self.piece_nodes[piece_id as usize]
                    .optimistic_children
                    .is_empty()
            {
                self.geometry_cache
                    .entry(GeometryKey {
                        board: key.board,
                        piece: branch.piece,
                    })
                    .or_insert_with(|| branch.geometry.clone());

                if self.is_optimistic() {
                    let next_bag =
                        next_bag_state(key.bag, branch.piece).expect("piece should exist in bag");
                    let mut children = branch
                        .geometry
                        .iter()
                        .copied()
                        .map(|child| {
                            let _ = self.board_features(child.board);
                            StagedPieceChild {
                                placement: child.placement,
                                succ_key: BagKey {
                                    board: child.board,
                                    bag: next_bag,
                                },
                            }
                        })
                        .collect::<Vec<_>>();
                    children.sort_by_key(|child| {
                        let features = self.board_features(child.succ_key.board);
                        (
                            features.height,
                            features.holes,
                            features.cells,
                            features.roughness,
                            features.height_spread,
                            child.placement,
                        )
                    });
                    self.reorder_staged_children_optimistically(
                        key.board,
                        branch.piece,
                        &mut children,
                    );
                    self.deduped_edges += children.len();
                    self.piece_nodes[piece_id as usize].optimistic_children = children
                        .iter()
                        .copied()
                        .map(|child| OptimisticChild {
                            placement: child.placement,
                            target: OptimisticChildTarget::Dormant(child.succ_key),
                        })
                        .collect();
                    self.piece_nodes[piece_id as usize].unique_succ_count = children
                        .iter()
                        .map(|child| child.succ_key)
                        .collect::<HashSet<_>>()
                        .len()
                        as u32;
                    self.piece_nodes[piece_id as usize].alive_succ_count =
                        self.piece_nodes[piece_id as usize].unique_succ_count;
                    let active_slot = self.choose_staged_optimistic_child_slot_strict_reuse(
                        &children,
                        self.optimistic_probe_width,
                    );
                    self.piece_nodes[piece_id as usize].active_child_slot =
                        active_slot.map(|(slot, _)| slot);
                    self.piece_nodes[piece_id as usize].next_repair_cursor =
                        active_slot.map_or(0, |(slot, _)| slot.saturating_add(1));
                    if let Some((slot, reused_activation)) = active_slot {
                        let child = children[slot as usize];
                        let (succ, novel_activation) =
                            if let Some(&known_succ) = self.bag_map.get(&child.succ_key) {
                                self.activate_bag_if_needed(known_succ);
                                (known_succ, false)
                            } else {
                                (self.intern_bag(child.succ_key), true)
                            };
                        self.piece_nodes[piece_id as usize].optimistic_children[slot as usize]
                            .target = OptimisticChildTarget::Activated(succ);
                        self.note_ranked_choice(key.board, branch.piece, child.placement);
                        self.promote_policy_placement(
                            key.board,
                            branch.piece,
                            child.placement,
                            false,
                        );
                        self.append_parent_links(succ, &[piece_id]);
                        self.optimistic_active_parent_links += 1;
                        self.increment_optimistic_successor_popularity(succ);
                        if reused_activation || !novel_activation {
                            self.optimistic_reuse_activations += 1;
                        } else {
                            self.optimistic_novel_activations += 1;
                        }
                    } else {
                        self.kill_piece(piece_id);
                    }
                } else {
                    let mut children = Vec::with_capacity(branch.geometry.len());
                    for child in branch.geometry.iter().copied() {
                        let _ = self.board_features(child.board);
                        let next_bag = next_bag_state(key.bag, branch.piece)
                            .expect("piece should exist in bag");
                        let succ = self.intern_bag(BagKey {
                            board: child.board,
                            bag: next_bag,
                        });
                        children.push(PieceChild {
                            placement: child.placement,
                            succ,
                        });
                    }
                    children.sort_by_key(|child| {
                        let features =
                            self.board_features(self.bag_nodes[child.succ as usize].key.board);
                        (
                            features.height,
                            features.holes,
                            features.cells,
                            features.roughness,
                            features.height_spread,
                            child.placement,
                        )
                    });
                    let mut unique_succs =
                        children.iter().map(|child| child.succ).collect::<Vec<_>>();
                    unique_succs.sort_unstable();
                    unique_succs.dedup();
                    self.deduped_edges += children.len();
                    self.piece_nodes[piece_id as usize].children = children;
                    let succ_count = unique_succs.len() as u32;
                    self.piece_nodes[piece_id as usize].unique_succ_count = succ_count;
                    let mut alive_succ_count = 0u32;
                    for &succ in &unique_succs {
                        if self.bag_nodes[succ as usize].resolution == NodeResolution::Won {
                            self.append_parent_links(succ, &[piece_id]);
                            alive_succ_count += 1;
                        }
                    }
                    self.piece_nodes[piece_id as usize].alive_succ_count = alive_succ_count;
                    if alive_succ_count == 0 {
                        self.kill_piece(piece_id);
                    }
                }
            }
        }

        self.bag_nodes[bag_id as usize].piece_nodes = piece_nodes;
        if self.lazy_branch_expansion {
            let mut requeue_board = None;
            {
                let bag = &mut self.bag_nodes[bag_id as usize];
                bag.expanded_branch_count = bag.expanded_branch_count.saturating_add(1);
                bag.next_branch_idx = bag.next_branch_idx.saturating_add(1);
                if bag.resolution == NodeResolution::Lost {
                    let remaining = bag.branch_order_len.saturating_sub(bag.next_branch_idx);
                    self.branches_skipped_due_to_parent_death += remaining as usize;
                    if expanded_before == 0 {
                        self.bags_killed_by_first_failed_branch += 1;
                    }
                    bag.expanded = true;
                    self.bags_expanded += 1;
                } else if bag.expanded_branch_count >= branch_order_len {
                    bag.expanded = true;
                    self.bags_expanded += 1;
                } else {
                    requeue_board = Some(bag.key.board);
                }
            }
            if let Some(board) = requeue_board {
                let score = self.expand_priority(board);
                let sequence = self.next_expand_sequence;
                self.next_expand_sequence += 1;
                self.expand_queue.push(ExpandEntry {
                    score,
                    sequence,
                    bag_id,
                });
            }
        } else {
            self.bag_nodes[bag_id as usize].expanded = true;
        }
    }

    fn resolve_optimistic_repair_replacement(
        &self,
        piece_id: PieceNodeId,
    ) -> Option<OptimisticRepairResolution> {
        let piece = &self.piece_nodes[piece_id as usize];
        let child_len = piece.optimistic_children.len();
        let cursor = (piece.next_repair_cursor as usize).min(child_len);
        let mut scanned = 0usize;
        (cursor..child_len)
            .chain(0..cursor)
            .find_map(|idx| {
                let child = piece.optimistic_children[idx];
                scanned += 1;
                match child.target {
                    OptimisticChildTarget::Activated(succ) if self.is_live_and_activated(succ) => {
                        Some(OptimisticRepairResolution {
                            slot: idx as u32,
                            placement: child.placement,
                            succ_key: None,
                            succ_id: Some(succ),
                            activation_kind: OptimisticActivationKind::ActivatedReuse,
                            scanned,
                        })
                    }
                    _ => None,
                }
            })
            .or_else(|| {
                (cursor..child_len).chain(0..cursor).find_map(|idx| {
                    let child = piece.optimistic_children[idx];
                    scanned += 1;
                    match child.target {
                        OptimisticChildTarget::Activated(succ)
                            if self.bag_nodes[succ as usize].resolution == NodeResolution::Won =>
                        {
                            Some(OptimisticRepairResolution {
                                slot: idx as u32,
                                placement: child.placement,
                                succ_key: None,
                                succ_id: Some(succ),
                                activation_kind: OptimisticActivationKind::KnownReuse,
                                scanned,
                            })
                        }
                        OptimisticChildTarget::Dormant(succ_key) => {
                            if let Some(&known_succ) = self.bag_map.get(&succ_key) {
                                (self.bag_nodes[known_succ as usize].resolution
                                    == NodeResolution::Won)
                                    .then_some(OptimisticRepairResolution {
                                        slot: idx as u32,
                                        placement: child.placement,
                                        succ_key: Some(succ_key),
                                        succ_id: Some(known_succ),
                                        activation_kind: OptimisticActivationKind::KnownReuse,
                                        scanned,
                                    })
                            } else {
                                Some(OptimisticRepairResolution {
                                    slot: idx as u32,
                                    placement: child.placement,
                                    succ_key: Some(succ_key),
                                    succ_id: None,
                                    activation_kind: OptimisticActivationKind::NovelActivation,
                                    scanned,
                                })
                            }
                        }
                        _ => None,
                    }
                })
            })
    }

    fn apply_optimistic_repair_proposal(&mut self, proposal: OptimisticRepairProposal) {
        let piece_id = proposal.piece_id;
        if self.piece_nodes[piece_id as usize].resolution == NodeResolution::Lost {
            return;
        }

        let mut replacement =
            proposal
                .slot
                .zip(proposal.placement)
                .and_then(|(slot, placement)| {
                    let succ_key = proposal.succ_key;
                    let succ_id = proposal.succ_id;
                    let kind = proposal.activation_kind?;
                    let valid = match kind {
                        OptimisticActivationKind::ActivatedReuse => {
                            succ_id.is_some_and(|succ| self.is_live_and_activated(succ))
                        }
                        OptimisticActivationKind::KnownReuse => {
                            if let Some(succ) = succ_id {
                                self.bag_nodes[succ as usize].resolution == NodeResolution::Won
                            } else if let Some(key) = succ_key {
                                self.bag_map.get(&key).copied().is_some_and(|succ| {
                                    self.bag_nodes[succ as usize].resolution == NodeResolution::Won
                                })
                            } else {
                                false
                            }
                        }
                        OptimisticActivationKind::NovelActivation => {
                            succ_key.is_some_and(|key| !self.bag_map.contains_key(&key))
                        }
                    };
                    valid.then_some(OptimisticRepairResolution {
                        slot,
                        placement,
                        succ_key,
                        succ_id,
                        activation_kind: kind,
                        scanned: proposal.scanned,
                    })
                });

        if replacement.is_none() {
            self.optimistic_repair_proposals_revalidated += 1;
            replacement = self.resolve_optimistic_repair_replacement(piece_id);
        }

        self.optimistic_repair_candidates_scanned += proposal.scanned;

        if let Some(replacement) = replacement {
            if replacement.scanned != proposal.scanned {
                self.optimistic_repair_candidates_scanned += replacement.scanned;
            }
            let succ = replacement
                .succ_id
                .or_else(|| {
                    replacement
                        .succ_key
                        .and_then(|key| self.bag_map.get(&key).copied())
                })
                .unwrap_or_else(|| {
                    self.intern_bag(replacement.succ_key.expect("novel repair needs succ key"))
                });
            let board = self.piece_nodes[piece_id as usize].key.board;
            let tetromino = self.piece_nodes[piece_id as usize].key.piece;
            let piece = &mut self.piece_nodes[piece_id as usize];
            piece.active_child_slot = Some(replacement.slot);
            piece.next_repair_cursor = replacement.slot.saturating_add(1);
            piece.optimistic_exhausted = false;
            piece.optimistic_children[replacement.slot as usize].target =
                OptimisticChildTarget::Activated(succ);
            self.optimistic_repairs_succeeded += 1;
            self.optimistic_active_parent_links += 1;
            self.activate_bag_if_needed(succ);
            self.increment_optimistic_successor_popularity(succ);
            self.append_parent_links(succ, &[piece_id]);
            self.note_ranked_choice(board, tetromino, replacement.placement);
            self.promote_policy_placement(board, tetromino, replacement.placement, true);
            match replacement.activation_kind {
                OptimisticActivationKind::ActivatedReuse | OptimisticActivationKind::KnownReuse => {
                    self.optimistic_reuse_activations += 1;
                    self.optimistic_repairs_reusing_known_successor += 1;
                }
                OptimisticActivationKind::NovelActivation => {
                    self.optimistic_novel_activations += 1;
                    self.optimistic_repairs_forcing_novel_activation += 1;
                }
            }
        } else {
            let piece = &mut self.piece_nodes[piece_id as usize];
            piece.active_child_slot = None;
            piece.next_repair_cursor = 0;
            piece.optimistic_exhausted = true;
            self.optimistic_repairs_failed += 1;
            self.kill_piece(piece_id);
        }
    }

    fn apply_optimistic_expand_proposal(&mut self, proposal: OptimisticExpandProposal) {
        let bag_id = proposal.bag_id;
        if self.bag_nodes[bag_id as usize].expanded
            || self.bag_nodes[bag_id as usize].resolution == NodeResolution::Lost
        {
            return;
        }
        if self.lazy_branch_expansion {
            self.branches_expanded += proposal.branches.len();
        } else {
            self.bags_expanded += 1;
        }

        let mut piece_nodes = self.bag_nodes[bag_id as usize].piece_nodes;
        let expanded_before = self.bag_nodes[bag_id as usize].expanded_branch_count;
        let branch_order_len = self.bag_nodes[bag_id as usize].branch_order_len;

        for branch in proposal.branches {
            let piece_idx = branch.piece.index() as usize;
            let piece_key = PieceKey {
                board: proposal.key.board,
                bag: proposal.key.bag,
                piece: branch.piece,
            };
            let piece_id = self.intern_piece(piece_key, bag_id);
            piece_nodes[piece_idx] = Some(piece_id);

            if self.piece_nodes[piece_id as usize].children.is_empty()
                && self.piece_nodes[piece_id as usize]
                    .optimistic_children
                    .is_empty()
            {
                self.geometry_cache
                    .entry(GeometryKey {
                        board: proposal.key.board,
                        piece: branch.piece,
                    })
                    .or_insert_with(|| branch.geometry.clone());

                self.deduped_edges += branch.children.len();
                self.piece_nodes[piece_id as usize].optimistic_children = branch
                    .children
                    .iter()
                    .copied()
                    .map(|child| OptimisticChild {
                        placement: child.placement,
                        target: OptimisticChildTarget::Dormant(child.succ_key),
                    })
                    .collect();
                self.piece_nodes[piece_id as usize].unique_succ_count = branch.unique_succ_count;
                self.piece_nodes[piece_id as usize].alive_succ_count = branch.unique_succ_count;
                if branch.popularity_reordered {
                    self.optimistic_reorders_using_popularity += 1;
                }

                let mut active_slot = branch.active_slot;
                let mut activation_kind = branch.activation_kind;
                if let Some(slot) = active_slot {
                    let child = branch.children[slot as usize];
                    let valid = match activation_kind.expect("active slot implies kind") {
                        OptimisticActivationKind::ActivatedReuse => self
                            .bag_map
                            .get(&child.succ_key)
                            .copied()
                            .is_some_and(|succ| self.is_live_and_activated(succ)),
                        OptimisticActivationKind::KnownReuse => self
                            .bag_map
                            .get(&child.succ_key)
                            .copied()
                            .is_some_and(|succ| {
                                self.bag_nodes[succ as usize].resolution == NodeResolution::Won
                            }),
                        OptimisticActivationKind::NovelActivation => {
                            !self.bag_map.contains_key(&child.succ_key)
                        }
                    };
                    if !valid {
                        self.optimistic_expand_proposals_revalidated += 1;
                        let recomputed = self.choose_staged_optimistic_child_slot_strict_reuse(
                            &branch.children,
                            self.optimistic_probe_width,
                        );
                        active_slot = recomputed.map(|(slot, _)| slot);
                        activation_kind = recomputed.map(|(slot, reused_activation)| {
                            let succ_key = branch.children[slot as usize].succ_key;
                            if reused_activation {
                                OptimisticActivationKind::ActivatedReuse
                            } else if self.bag_map.contains_key(&succ_key) {
                                OptimisticActivationKind::KnownReuse
                            } else {
                                OptimisticActivationKind::NovelActivation
                            }
                        });
                    }
                }

                self.piece_nodes[piece_id as usize].active_child_slot = active_slot;
                self.piece_nodes[piece_id as usize].next_repair_cursor =
                    active_slot.map_or(0, |slot| slot.saturating_add(1));
                if let Some(slot) = active_slot {
                    let child = branch.children[slot as usize];
                    let succ = if let Some(&known_succ) = self.bag_map.get(&child.succ_key) {
                        self.activate_bag_if_needed(known_succ);
                        known_succ
                    } else {
                        self.intern_bag(child.succ_key)
                    };
                    self.piece_nodes[piece_id as usize].optimistic_children[slot as usize].target =
                        OptimisticChildTarget::Activated(succ);
                    self.note_ranked_choice(proposal.key.board, branch.piece, child.placement);
                    self.promote_policy_placement(
                        proposal.key.board,
                        branch.piece,
                        child.placement,
                        false,
                    );
                    self.append_parent_links(succ, &[piece_id]);
                    self.optimistic_active_parent_links += 1;
                    self.increment_optimistic_successor_popularity(succ);
                    match activation_kind.expect("active slot implies kind") {
                        OptimisticActivationKind::ActivatedReuse
                        | OptimisticActivationKind::KnownReuse => {
                            self.optimistic_reuse_activations += 1;
                        }
                        OptimisticActivationKind::NovelActivation => {
                            self.optimistic_novel_activations += 1;
                        }
                    }
                } else {
                    self.kill_piece(piece_id);
                }
            }
        }

        self.bag_nodes[bag_id as usize].piece_nodes = piece_nodes;
        if self.lazy_branch_expansion {
            let mut requeue_board = None;
            {
                let bag = &mut self.bag_nodes[bag_id as usize];
                bag.expanded_branch_count = bag.expanded_branch_count.saturating_add(1);
                bag.next_branch_idx = bag.next_branch_idx.saturating_add(1);
                if bag.resolution == NodeResolution::Lost {
                    let remaining = bag.branch_order_len.saturating_sub(bag.next_branch_idx);
                    self.branches_skipped_due_to_parent_death += remaining as usize;
                    if expanded_before == 0 {
                        self.bags_killed_by_first_failed_branch += 1;
                    }
                    bag.expanded = true;
                    self.bags_expanded += 1;
                } else if bag.expanded_branch_count >= branch_order_len {
                    bag.expanded = true;
                    self.bags_expanded += 1;
                } else {
                    requeue_board = Some(bag.key.board);
                }
            }
            if let Some(board) = requeue_board {
                let score = self.expand_priority(board);
                let sequence = self.next_expand_sequence;
                self.next_expand_sequence += 1;
                self.expand_queue.push(ExpandEntry {
                    score,
                    sequence,
                    bag_id,
                });
            }
        } else {
            self.bag_nodes[bag_id as usize].expanded = true;
        }
    }

    fn try_build_core_first_result(
        &mut self,
        total_start: Instant,
        discover_start: Instant,
    ) -> Result<Option<ProofSolveResult>> {
        let certification_start = Instant::now();
        let expanded_live = self
            .bag_nodes
            .iter()
            .enumerate()
            .filter_map(|(bag_id, bag)| {
                (bag.expanded && bag.resolution == NodeResolution::Won)
                    .then_some(bag_id as BagNodeId)
            })
            .collect::<Vec<_>>();
        if expanded_live.is_empty() {
            self.core_certification_secs += certification_start.elapsed().as_secs_f64();
            return Ok(None);
        }

        let mut id_to_pos = vec![None; self.bag_nodes.len()];
        for (pos, &bag_id) in expanded_live.iter().enumerate() {
            id_to_pos[bag_id as usize] = Some(pos);
        }

        let mut adjacency = vec![Vec::<usize>::new(); expanded_live.len()];
        let mut reverse = vec![Vec::<usize>::new(); expanded_live.len()];
        for (pos, &bag_id) in expanded_live.iter().enumerate() {
            let bag = &self.bag_nodes[bag_id as usize];
            let mut succs = Vec::new();
            for branch in piece_branches(bag.key.bag) {
                let Some(piece_id) = bag.piece_nodes[branch.piece.index() as usize] else {
                    continue;
                };
                let piece = &self.piece_nodes[piece_id as usize];
                if piece.resolution == NodeResolution::Lost {
                    continue;
                }
                for child in &piece.children {
                    if self.bag_nodes[child.succ as usize].expanded
                        && self.bag_nodes[child.succ as usize].resolution == NodeResolution::Won
                    {
                        if let Some(next_pos) = id_to_pos[child.succ as usize] {
                            succs.push(next_pos);
                        }
                    }
                }
            }
            succs.sort_unstable();
            succs.dedup();
            for &next in &succs {
                adjacency[pos].push(next);
                reverse[next].push(pos);
            }
        }

        let mut order = Vec::with_capacity(expanded_live.len());
        let mut seen = vec![false; expanded_live.len()];
        for start in 0..expanded_live.len() {
            if seen[start] {
                continue;
            }
            let mut stack = vec![(start, 0usize)];
            seen[start] = true;
            while let Some((node, edge_idx)) = stack.pop() {
                if edge_idx < adjacency[node].len() {
                    stack.push((node, edge_idx + 1));
                    let next = adjacency[node][edge_idx];
                    if !seen[next] {
                        seen[next] = true;
                        stack.push((next, 0));
                    }
                } else {
                    order.push(node);
                }
            }
        }

        let mut comp_index = vec![usize::MAX; expanded_live.len()];
        let mut components = Vec::<Vec<usize>>::new();
        for &start in order.iter().rev() {
            if comp_index[start] != usize::MAX {
                continue;
            }
            let comp_id = components.len();
            let mut stack = vec![start];
            let mut component = Vec::new();
            comp_index[start] = comp_id;
            while let Some(node) = stack.pop() {
                component.push(node);
                for &prev in &reverse[node] {
                    if comp_index[prev] == usize::MAX {
                        comp_index[prev] = comp_id;
                        stack.push(prev);
                    }
                }
            }
            components.push(component);
        }

        self.largest_scc_bags = self
            .largest_scc_bags
            .max(components.iter().map(Vec::len).max().unwrap_or(0));

        let mut candidates = Vec::<Vec<BagNodeId>>::new();
        for component in components {
            let component_bags = component
                .into_iter()
                .map(|pos| expanded_live[pos])
                .collect::<Vec<_>>();
            let component_set = component_bags
                .iter()
                .map(|&bag_id| bag_id as usize)
                .collect::<std::collections::HashSet<_>>();
            let mut closed = true;
            let mut piece_count = 0usize;
            for &bag_id in &component_bags {
                let bag = &self.bag_nodes[bag_id as usize];
                for branch in piece_branches(bag.key.bag) {
                    let Some(piece_id) = bag.piece_nodes[branch.piece.index() as usize] else {
                        closed = false;
                        break;
                    };
                    let piece = &self.piece_nodes[piece_id as usize];
                    if piece.resolution == NodeResolution::Lost {
                        closed = false;
                        break;
                    }
                    piece_count += 1;
                    let has_internal_child = piece
                        .children
                        .iter()
                        .any(|child| component_set.contains(&(child.succ as usize)));
                    if !has_internal_child {
                        closed = false;
                        break;
                    }
                }
                if !closed {
                    break;
                }
            }
            self.largest_scc_pieces = self.largest_scc_pieces.max(piece_count);
            if closed {
                candidates.push(component_bags);
            }
        }

        self.candidate_core_count += candidates.len();
        self.core_certification_secs += certification_start.elapsed().as_secs_f64();

        if candidates.is_empty() {
            return Ok(None);
        }

        candidates.sort_by_key(|bags| std::cmp::Reverse(bags.len()));
        let bridge_start = Instant::now();
        for core_bags in candidates {
            let mut target = vec![false; self.bag_nodes.len()];
            for &bag_id in &core_bags {
                target[bag_id as usize] = true;
            }

            let mut changed = true;
            while changed {
                changed = false;
                for &bag_id in &expanded_live {
                    if target[bag_id as usize] {
                        continue;
                    }
                    let bag = &self.bag_nodes[bag_id as usize];
                    let reaches_target = piece_branches(bag.key.bag).all(|branch| {
                        let Some(piece_id) = bag.piece_nodes[branch.piece.index() as usize] else {
                            return false;
                        };
                        let piece = &self.piece_nodes[piece_id as usize];
                        piece.resolution == NodeResolution::Won
                            && piece
                                .children
                                .iter()
                                .any(|child| target[child.succ as usize])
                    });
                    if reaches_target {
                        target[bag_id as usize] = true;
                        changed = true;
                    }
                }
            }

            if !target[self.root_bag as usize] {
                continue;
            }

            self.bridge_search_secs += bridge_start.elapsed().as_secs_f64();
            self.closed_core_found = true;
            return Ok(Some(self.extract_target_proof(
                &target,
                total_start,
                discover_start,
            )?));
        }

        self.bridge_search_secs += bridge_start.elapsed().as_secs_f64();
        Ok(None)
    }

    fn build_template_kernel_result(
        mut self,
        total_start: Instant,
        discover_start: Instant,
        options: SolveOptions,
    ) -> Result<ProofSolveResult> {
        let member_start = Instant::now();
        let mut member_bags = Vec::<BagNodeId>::new();
        let mut member_piece_count = 0usize;
        let mut signatures = std::collections::HashSet::<BoardSignature>::new();
        for bag_id in 0..self.bag_nodes.len() {
            let bag = &self.bag_nodes[bag_id];
            if !bag.expanded || bag.resolution == NodeResolution::Lost {
                continue;
            }
            let board = bag.key.board;
            let bag_key = bag.key.bag;
            let features = self.board_features(board);
            if features.turning_points > options.template_max_turning_points
                || features.max_well_depth > options.template_max_well_depth
            {
                continue;
            }
            member_bags.push(bag_id as BagNodeId);
            signatures.insert(Self::board_signature_for(board));
            member_piece_count += piece_branches(bag_key).count();
        }

        let mut member_set = vec![false; self.bag_nodes.len()];
        for &bag_id in &member_bags {
            member_set[bag_id as usize] = true;
        }

        let mut failed_piece_counts = [0usize; 7];
        let mut family_closed = !member_bags.is_empty();
        for &bag_id in &member_bags {
            let bag = &self.bag_nodes[bag_id as usize];
            for branch in piece_branches(bag.key.bag) {
                let Some(piece_id) = bag.piece_nodes[branch.piece.index() as usize] else {
                    failed_piece_counts[branch.piece.index() as usize] += 1;
                    family_closed = false;
                    continue;
                };
                let piece = &self.piece_nodes[piece_id as usize];
                let has_internal_child = piece
                    .children
                    .iter()
                    .any(|child| member_set[child.succ as usize]);
                if !has_internal_child {
                    failed_piece_counts[branch.piece.index() as usize] += 1;
                    family_closed = false;
                }
            }
        }
        self.core_certification_secs += member_start.elapsed().as_secs_f64();

        let (largest_template_scc_bags, largest_template_scc_pieces) =
            self.compute_member_scc_metrics(&member_bags, &member_set);

        let mut target = member_set.clone();
        if family_closed {
            let bridge_start = Instant::now();
            let mut changed = true;
            let mut depth = 0usize;
            while changed && depth < options.template_max_bridge_depth {
                changed = false;
                depth += 1;
                for bag_id in 0..self.bag_nodes.len() {
                    if target[bag_id] {
                        continue;
                    }
                    let bag = &self.bag_nodes[bag_id];
                    if !bag.expanded || bag.resolution == NodeResolution::Lost {
                        continue;
                    }
                    let reaches_target = piece_branches(bag.key.bag).all(|branch| {
                        let Some(piece_id) = bag.piece_nodes[branch.piece.index() as usize] else {
                            return false;
                        };
                        let piece = &self.piece_nodes[piece_id as usize];
                        piece.resolution == NodeResolution::Won
                            && piece
                                .children
                                .iter()
                                .any(|child| target[child.succ as usize])
                    });
                    if reaches_target {
                        target[bag_id] = true;
                        changed = true;
                    }
                }
            }
            self.bridge_search_secs += bridge_start.elapsed().as_secs_f64();
        }

        let attractor_bags = target.iter().filter(|&&alive| alive).count();
        let root_reaches_family = family_closed
            && (self.root_bag as usize) < target.len()
            && target[self.root_bag as usize];

        self.closed_core_found = family_closed;
        self.candidate_core_count = usize::from(!member_bags.is_empty());
        self.largest_scc_bags = largest_template_scc_bags;
        self.largest_scc_pieces = largest_template_scc_pieces;

        if root_reaches_family {
            let mut result = self.extract_target_proof(&target, total_start, discover_start)?;
            result.root_result.conclusion = SolveConclusion::Yes;
            result.metrics.template_member_bags = member_bags.len();
            result.metrics.template_member_pieces = member_piece_count;
            result.metrics.template_signature_count = signatures.len();
            result.metrics.largest_template_scc_bags = largest_template_scc_bags;
            result.metrics.largest_template_scc_pieces = largest_template_scc_pieces;
            result.metrics.root_reaches_template_family = true;
            result.metrics.template_attractor_bags = attractor_bags;
            result.metrics.template_closure_failed_piece_counts = failed_piece_counts;
            return Ok(result);
        }

        let (
            optimistic_activated_live_bags,
            optimistic_dormant_child_targets,
            optimistic_activated_frontier,
        ) = self.optimistic_activation_metrics();
        let metrics = ProofMetrics {
            solve_mode: self.solve_mode,
            used_full_fallback: self.used_full_fallback,
            bag_node_count: self.bag_nodes.len(),
            piece_node_count: self.piece_nodes.len(),
            winning_count: self
                .bag_nodes
                .iter()
                .filter(|node| node.resolution == NodeResolution::Won)
                .count(),
            losing_count: self
                .bag_nodes
                .iter()
                .filter(|node| node.resolution == NodeResolution::Lost)
                .count(),
            piece_winning_count: self
                .piece_nodes
                .iter()
                .filter(|node| node.resolution == NodeResolution::Won)
                .count(),
            piece_losing_count: self
                .piece_nodes
                .iter()
                .filter(|node| node.resolution == NodeResolution::Lost)
                .count(),
            dependency_count: self.dependency_count(),
            deduped_edge_count: self.deduped_edges,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            geometry_cache_hits: self.geometry_cache_hits.load(AtomicOrdering::Relaxed),
            geometry_cache_misses: self.geometry_cache_misses.load(AtomicOrdering::Relaxed),
            board_feature_cache_hits: self.board_feature_cache_hits.load(AtomicOrdering::Relaxed),
            board_feature_cache_misses: self
                .board_feature_cache_misses
                .load(AtomicOrdering::Relaxed),
            bags_expanded: self.bags_expanded,
            skipped_dead_bags: self.skipped_dead_bags,
            parallel_batches_processed: self.parallel_batches_processed,
            parallel_batch_size: Self::EXPAND_BATCH_SIZE,
            branches_expanded: self.branches_expanded,
            branches_skipped_due_to_parent_death: self.branches_skipped_due_to_parent_death,
            bags_killed_by_first_failed_branch: self.bags_killed_by_first_failed_branch,
            avg_branches_expanded_per_bag: if self.bags_expanded == 0 {
                0.0
            } else {
                self.branches_expanded as f64 / self.bags_expanded as f64
            },
            staged_branches_processed: self.staged_branches_processed,
            batch_unique_piece_keys: self.batch_unique_piece_keys,
            batch_unique_bag_keys: self.batch_unique_bag_keys,
            resolved_children_processed: self.resolved_children_processed,
            batched_parent_links_appended: self.batched_parent_links_appended,
            parent_segments_allocated: self.parent_segments.len(),
            parent_links_stored: self.parent_edges.len(),
            avg_parent_segments_per_bag: if self.bag_nodes.is_empty() {
                0.0
            } else {
                self.parent_segments.len() as f64 / self.bag_nodes.len() as f64
            },
            optimistic_policy_entries: self.optimistic_policy.len(),
            optimistic_ranked_policy_entries: self.optimistic_policy.len(),
            optimistic_ranked_policy_capacity: Self::OPTIMISTIC_POLICY_CAPACITY,
            optimistic_policy_hits: self.optimistic_policy_hits,
            optimistic_policy_misses: self.optimistic_policy_misses,
            optimistic_repairs_attempted: self.optimistic_repairs_attempted,
            optimistic_repairs_succeeded: self.optimistic_repairs_succeeded,
            optimistic_repairs_failed: self.optimistic_repairs_failed,
            optimistic_repairs_promoted: self.optimistic_repairs_promoted,
            optimistic_repairs_demoted: self.optimistic_repairs_demoted,
            optimistic_reused_ranked_candidates: self.optimistic_reused_ranked_candidates,
            optimistic_active_parent_links: self.optimistic_active_parent_links,
            optimistic_popularity_nonzero_bags: self
                .optimistic_successor_popularity
                .iter()
                .filter(|&&value| value > 0)
                .count(),
            optimistic_max_successor_popularity: self
                .optimistic_successor_popularity
                .iter()
                .copied()
                .max()
                .unwrap_or(0),
            optimistic_avg_successor_popularity: {
                let nonzero = self
                    .optimistic_successor_popularity
                    .iter()
                    .copied()
                    .filter(|&value| value > 0)
                    .collect::<Vec<_>>();
                if nonzero.is_empty() {
                    0.0
                } else {
                    nonzero.iter().sum::<usize>() as f64 / nonzero.len() as f64
                }
            },
            optimistic_reorders_using_popularity: self.optimistic_reorders_using_popularity,
            optimistic_avg_candidates_scanned_per_repair: if self.optimistic_repairs_attempted == 0
            {
                0.0
            } else {
                self.optimistic_repair_candidates_scanned as f64
                    / self.optimistic_repairs_attempted as f64
            },
            optimistic_avg_rank_position_chosen: if self.optimistic_rank_position_count == 0 {
                0.0
            } else {
                self.optimistic_rank_position_sum as f64
                    / self.optimistic_rank_position_count as f64
            },
            optimistic_max_repairs_on_single_piece: self.optimistic_max_repairs_on_single_piece,
            optimistic_activated_live_bags,
            optimistic_dormant_child_targets,
            optimistic_activated_frontier,
            optimistic_novel_activations: self.optimistic_novel_activations,
            optimistic_reuse_activations: self.optimistic_reuse_activations,
            optimistic_repairs_reusing_known_successor: self
                .optimistic_repairs_reusing_known_successor,
            optimistic_repairs_forcing_novel_activation: self
                .optimistic_repairs_forcing_novel_activation,
            optimistic_fallback_triggered: self.optimistic_fallback_triggered,
            optimistic_expand_proposal_secs: self.optimistic_expand_proposal_secs,
            optimistic_repair_proposal_secs: self.optimistic_repair_proposal_secs,
            optimistic_repair_batches_processed: self.optimistic_repair_batches_processed,
            optimistic_expand_proposals_revalidated: self.optimistic_expand_proposals_revalidated,
            optimistic_repair_proposals_revalidated: self.optimistic_repair_proposals_revalidated,
            candidate_core_count: usize::from(!member_bags.is_empty()),
            largest_scc_bags: largest_template_scc_bags,
            largest_scc_pieces: largest_template_scc_pieces,
            closed_core_found: family_closed,
            core_certification_secs: self.core_certification_secs,
            bridge_search_secs: self.bridge_search_secs,
            stage_secs: self.total_stage_secs,
            commit_secs: self.total_commit_secs,
            template_member_bags: member_bags.len(),
            template_member_pieces: member_piece_count,
            template_signature_count: signatures.len(),
            largest_template_scc_bags,
            largest_template_scc_pieces,
            root_reaches_template_family: false,
            template_attractor_bags: attractor_bags,
            template_closure_failed_piece_counts: failed_piece_counts,
        };
        Ok(ProofSolveResult {
            solve_mode: self.solve_mode,
            used_full_fallback: self.used_full_fallback,
            root_bag: self.root_bag,
            bag_nodes: self.materialized_bag_nodes(),
            piece_nodes: self.piece_nodes,
            metrics,
            discover_secs: discover_start.elapsed().as_secs_f64(),
            propagate_secs: 0.0,
            total_secs: total_start.elapsed().as_secs_f64(),
            root_result: RootProofResult {
                conclusion: SolveConclusion::Unresolved,
                result: NodeResolution::Lost,
                best_action: [None; 7],
                failing_pieces: Vec::new(),
            },
        })
    }

    fn compute_member_scc_metrics(
        &self,
        member_bags: &[BagNodeId],
        member_set: &[bool],
    ) -> (usize, usize) {
        if member_bags.is_empty() {
            return (0, 0);
        }
        let mut id_to_pos = vec![None; self.bag_nodes.len()];
        for (pos, &bag_id) in member_bags.iter().enumerate() {
            id_to_pos[bag_id as usize] = Some(pos);
        }
        let mut adjacency = vec![Vec::<usize>::new(); member_bags.len()];
        let mut reverse = vec![Vec::<usize>::new(); member_bags.len()];
        for (pos, &bag_id) in member_bags.iter().enumerate() {
            let bag = &self.bag_nodes[bag_id as usize];
            for branch in piece_branches(bag.key.bag) {
                let Some(piece_id) = bag.piece_nodes[branch.piece.index() as usize] else {
                    continue;
                };
                let piece = &self.piece_nodes[piece_id as usize];
                for child in &piece.children {
                    if member_set[child.succ as usize]
                        && let Some(next_pos) = id_to_pos[child.succ as usize]
                    {
                        adjacency[pos].push(next_pos);
                        reverse[next_pos].push(pos);
                    }
                }
            }
            adjacency[pos].sort_unstable();
            adjacency[pos].dedup();
        }

        let mut order = Vec::with_capacity(member_bags.len());
        let mut seen = vec![false; member_bags.len()];
        for start in 0..member_bags.len() {
            if seen[start] {
                continue;
            }
            let mut stack = vec![(start, 0usize)];
            seen[start] = true;
            while let Some((node, edge_idx)) = stack.pop() {
                if edge_idx < adjacency[node].len() {
                    stack.push((node, edge_idx + 1));
                    let next = adjacency[node][edge_idx];
                    if !seen[next] {
                        seen[next] = true;
                        stack.push((next, 0));
                    }
                } else {
                    order.push(node);
                }
            }
        }

        let mut comp_index = vec![usize::MAX; member_bags.len()];
        let mut largest_bags = 0usize;
        let mut largest_pieces = 0usize;
        for &start in order.iter().rev() {
            if comp_index[start] != usize::MAX {
                continue;
            }
            let comp_id = start;
            let mut stack = vec![start];
            comp_index[start] = comp_id;
            let mut bag_count = 0usize;
            let mut piece_count = 0usize;
            while let Some(node) = stack.pop() {
                bag_count += 1;
                piece_count +=
                    piece_branches(self.bag_nodes[member_bags[node] as usize].key.bag).count();
                for &prev in &reverse[node] {
                    if comp_index[prev] == usize::MAX {
                        comp_index[prev] = comp_id;
                        stack.push(prev);
                    }
                }
            }
            largest_bags = largest_bags.max(bag_count);
            largest_pieces = largest_pieces.max(piece_count);
        }
        (largest_bags, largest_pieces)
    }

    fn extract_target_proof(
        &self,
        target: &[bool],
        total_start: Instant,
        discover_start: Instant,
    ) -> Result<ProofSolveResult> {
        let selected_bags = self
            .bag_nodes
            .iter()
            .enumerate()
            .filter_map(|(bag_id, _)| target[bag_id].then_some(bag_id as BagNodeId))
            .collect::<Vec<_>>();
        let mut bag_remap = vec![None; self.bag_nodes.len()];
        for (new_id, &old_id) in selected_bags.iter().enumerate() {
            bag_remap[old_id as usize] = Some(new_id as BagNodeId);
        }

        let mut bag_nodes = selected_bags
            .iter()
            .map(|&old_id| BagNode {
                key: self.bag_nodes[old_id as usize].key,
                resolution: NodeResolution::Won,
                expanded: true,
                parents: Vec::new(),
                piece_nodes: [None; 7],
                activated: true,
                branch_order: [u8::MAX; 7],
                branch_order_len: 0,
                next_branch_idx: 0,
                expanded_branch_count: 0,
                branch_order_initialized: false,
            })
            .collect::<Vec<_>>();
        let mut piece_nodes = Vec::new();
        let mut deduped_edges = 0usize;

        for (new_bag_id, &old_bag_id) in selected_bags.iter().enumerate() {
            let bag = &self.bag_nodes[old_bag_id as usize];
            for branch in piece_branches(bag.key.bag) {
                let old_piece_id =
                    bag.piece_nodes[branch.piece.index() as usize].ok_or_else(|| {
                        anyhow::anyhow!("target proof missing piece node for bag {}", old_bag_id)
                    })?;
                let old_piece = &self.piece_nodes[old_piece_id as usize];
                let children = old_piece
                    .children
                    .iter()
                    .filter_map(|child| {
                        target[child.succ as usize].then_some(PieceChild {
                            placement: child.placement,
                            succ: bag_remap[child.succ as usize]
                                .expect("selected successor should have remap"),
                        })
                    })
                    .collect::<Vec<_>>();
                if children.is_empty() {
                    bail!(
                        "target proof extraction found empty child set for winning piece {:?}",
                        old_piece.key
                    );
                }
                let best = children[0];
                let new_piece_id = piece_nodes.len() as PieceNodeId;
                deduped_edges += children.len();
                piece_nodes.push(PieceNode {
                    key: old_piece.key,
                    parent_bag: new_bag_id as BagNodeId,
                    resolution: NodeResolution::Won,
                    unique_succ_count: children.len() as u32,
                    alive_succ_count: children.len() as u32,
                    best_action: Some(best.placement),
                    best_child: Some(best.succ),
                    children,
                    active_child_slot: Some(0),
                    optimistic_children: Vec::new(),
                    next_repair_cursor: 1,
                    repair_attempts: 0,
                    optimistic_exhausted: false,
                    optimistic_repair_pending: false,
                });
                bag_nodes[new_bag_id].piece_nodes[branch.piece.index() as usize] =
                    Some(new_piece_id);
            }
        }

        for (piece_id, piece) in piece_nodes.iter().enumerate() {
            for child in &piece.children {
                bag_nodes[child.succ as usize]
                    .parents
                    .push(piece_id as PieceNodeId);
            }
        }

        let winning_count = bag_nodes.len();
        let piece_winning_count = piece_nodes.len();
        let dependency_count = bag_nodes.iter().map(|node| node.parents.len()).sum();
        let (
            optimistic_activated_live_bags,
            optimistic_dormant_child_targets,
            optimistic_activated_frontier,
        ) = self.optimistic_activation_metrics();
        let metrics = ProofMetrics {
            solve_mode: self.solve_mode,
            used_full_fallback: self.used_full_fallback,
            bag_node_count: bag_nodes.len(),
            piece_node_count: piece_nodes.len(),
            winning_count,
            losing_count: 0,
            piece_winning_count,
            piece_losing_count: 0,
            dependency_count,
            deduped_edge_count: deduped_edges,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            geometry_cache_hits: self.geometry_cache_hits.load(AtomicOrdering::Relaxed),
            geometry_cache_misses: self.geometry_cache_misses.load(AtomicOrdering::Relaxed),
            board_feature_cache_hits: self.board_feature_cache_hits.load(AtomicOrdering::Relaxed),
            board_feature_cache_misses: self
                .board_feature_cache_misses
                .load(AtomicOrdering::Relaxed),
            bags_expanded: self.bags_expanded,
            skipped_dead_bags: self.skipped_dead_bags,
            parallel_batches_processed: self.parallel_batches_processed,
            parallel_batch_size: Self::EXPAND_BATCH_SIZE,
            branches_expanded: self.branches_expanded,
            branches_skipped_due_to_parent_death: self.branches_skipped_due_to_parent_death,
            bags_killed_by_first_failed_branch: self.bags_killed_by_first_failed_branch,
            avg_branches_expanded_per_bag: if self.bags_expanded == 0 {
                0.0
            } else {
                self.branches_expanded as f64 / self.bags_expanded as f64
            },
            staged_branches_processed: self.staged_branches_processed,
            batch_unique_piece_keys: self.batch_unique_piece_keys,
            batch_unique_bag_keys: self.batch_unique_bag_keys,
            resolved_children_processed: self.resolved_children_processed,
            batched_parent_links_appended: self.batched_parent_links_appended,
            parent_segments_allocated: self.parent_segments.len(),
            parent_links_stored: self.parent_edges.len(),
            avg_parent_segments_per_bag: if self.bag_nodes.is_empty() {
                0.0
            } else {
                self.parent_segments.len() as f64 / self.bag_nodes.len() as f64
            },
            optimistic_policy_entries: self.optimistic_policy.len(),
            optimistic_ranked_policy_entries: self.optimistic_policy.len(),
            optimistic_ranked_policy_capacity: Self::OPTIMISTIC_POLICY_CAPACITY,
            optimistic_policy_hits: self.optimistic_policy_hits,
            optimistic_policy_misses: self.optimistic_policy_misses,
            optimistic_repairs_attempted: self.optimistic_repairs_attempted,
            optimistic_repairs_succeeded: self.optimistic_repairs_succeeded,
            optimistic_repairs_failed: self.optimistic_repairs_failed,
            optimistic_repairs_promoted: self.optimistic_repairs_promoted,
            optimistic_repairs_demoted: self.optimistic_repairs_demoted,
            optimistic_reused_ranked_candidates: self.optimistic_reused_ranked_candidates,
            optimistic_active_parent_links: self.optimistic_active_parent_links,
            optimistic_popularity_nonzero_bags: self
                .optimistic_successor_popularity
                .iter()
                .filter(|&&value| value > 0)
                .count(),
            optimistic_max_successor_popularity: self
                .optimistic_successor_popularity
                .iter()
                .copied()
                .max()
                .unwrap_or(0),
            optimistic_avg_successor_popularity: {
                let nonzero = self
                    .optimistic_successor_popularity
                    .iter()
                    .copied()
                    .filter(|&value| value > 0)
                    .collect::<Vec<_>>();
                if nonzero.is_empty() {
                    0.0
                } else {
                    nonzero.iter().sum::<usize>() as f64 / nonzero.len() as f64
                }
            },
            optimistic_reorders_using_popularity: self.optimistic_reorders_using_popularity,
            optimistic_avg_candidates_scanned_per_repair: if self.optimistic_repairs_attempted == 0
            {
                0.0
            } else {
                self.optimistic_repair_candidates_scanned as f64
                    / self.optimistic_repairs_attempted as f64
            },
            optimistic_avg_rank_position_chosen: if self.optimistic_rank_position_count == 0 {
                0.0
            } else {
                self.optimistic_rank_position_sum as f64
                    / self.optimistic_rank_position_count as f64
            },
            optimistic_max_repairs_on_single_piece: self.optimistic_max_repairs_on_single_piece,
            optimistic_activated_live_bags,
            optimistic_dormant_child_targets,
            optimistic_activated_frontier,
            optimistic_novel_activations: self.optimistic_novel_activations,
            optimistic_reuse_activations: self.optimistic_reuse_activations,
            optimistic_repairs_reusing_known_successor: self
                .optimistic_repairs_reusing_known_successor,
            optimistic_repairs_forcing_novel_activation: self
                .optimistic_repairs_forcing_novel_activation,
            optimistic_fallback_triggered: self.optimistic_fallback_triggered,
            optimistic_expand_proposal_secs: self.optimistic_expand_proposal_secs,
            optimistic_repair_proposal_secs: self.optimistic_repair_proposal_secs,
            optimistic_repair_batches_processed: self.optimistic_repair_batches_processed,
            optimistic_expand_proposals_revalidated: self.optimistic_expand_proposals_revalidated,
            optimistic_repair_proposals_revalidated: self.optimistic_repair_proposals_revalidated,
            candidate_core_count: self.candidate_core_count,
            largest_scc_bags: self.largest_scc_bags,
            largest_scc_pieces: self.largest_scc_pieces,
            closed_core_found: self.closed_core_found,
            core_certification_secs: self.core_certification_secs,
            bridge_search_secs: self.bridge_search_secs,
            stage_secs: self.total_stage_secs,
            commit_secs: self.total_commit_secs,
            template_member_bags: 0,
            template_member_pieces: 0,
            template_signature_count: 0,
            largest_template_scc_bags: 0,
            largest_template_scc_pieces: 0,
            root_reaches_template_family: false,
            template_attractor_bags: 0,
            template_closure_failed_piece_counts: [0; 7],
        };
        let root_bag = bag_remap[self.root_bag as usize]
            .ok_or_else(|| anyhow::anyhow!("root bag missing from target proof"))?;
        let root_result = RootProofResult {
            conclusion: SolveConclusion::Yes,
            result: NodeResolution::Won,
            best_action: {
                let mut best_action = [None; 7];
                for branch in piece_branches(self.bag_nodes[self.root_bag as usize].key.bag) {
                    let piece_id = bag_nodes[root_bag as usize].piece_nodes
                        [branch.piece.index() as usize]
                        .expect("root target piece should exist");
                    best_action[branch.piece.index() as usize] =
                        piece_nodes[piece_id as usize].best_action;
                }
                best_action
            },
            failing_pieces: Vec::new(),
        };
        Ok(ProofSolveResult {
            solve_mode: self.solve_mode,
            used_full_fallback: self.used_full_fallback,
            root_bag,
            bag_nodes,
            piece_nodes,
            metrics,
            discover_secs: discover_start.elapsed().as_secs_f64(),
            propagate_secs: 0.0,
            total_secs: total_start.elapsed().as_secs_f64(),
            root_result,
        })
    }

    fn compute_metrics(&self) -> ProofMetrics {
        let winning_count = self
            .bag_nodes
            .iter()
            .filter(|node| node.resolution == NodeResolution::Won)
            .count();
        let losing_count = self.bag_nodes.len().saturating_sub(winning_count);
        let piece_winning_count = self
            .piece_nodes
            .iter()
            .filter(|node| node.resolution == NodeResolution::Won)
            .count();
        let piece_losing_count = self.piece_nodes.len().saturating_sub(piece_winning_count);
        let dependency_count = self.dependency_count();
        let (
            optimistic_activated_live_bags,
            optimistic_dormant_child_targets,
            optimistic_activated_frontier,
        ) = self.optimistic_activation_metrics();

        ProofMetrics {
            solve_mode: self.solve_mode,
            used_full_fallback: self.used_full_fallback,
            bag_node_count: self.bag_nodes.len(),
            piece_node_count: self.piece_nodes.len(),
            winning_count,
            losing_count,
            piece_winning_count,
            piece_losing_count,
            dependency_count,
            deduped_edge_count: self.deduped_edges,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            geometry_cache_hits: self.geometry_cache_hits.load(AtomicOrdering::Relaxed),
            geometry_cache_misses: self.geometry_cache_misses.load(AtomicOrdering::Relaxed),
            board_feature_cache_hits: self.board_feature_cache_hits.load(AtomicOrdering::Relaxed),
            board_feature_cache_misses: self
                .board_feature_cache_misses
                .load(AtomicOrdering::Relaxed),
            bags_expanded: self.bags_expanded,
            skipped_dead_bags: self.skipped_dead_bags,
            parallel_batches_processed: self.parallel_batches_processed,
            parallel_batch_size: Self::EXPAND_BATCH_SIZE,
            branches_expanded: self.branches_expanded,
            branches_skipped_due_to_parent_death: self.branches_skipped_due_to_parent_death,
            bags_killed_by_first_failed_branch: self.bags_killed_by_first_failed_branch,
            avg_branches_expanded_per_bag: if self.bags_expanded == 0 {
                0.0
            } else {
                self.branches_expanded as f64 / self.bags_expanded as f64
            },
            staged_branches_processed: self.staged_branches_processed,
            batch_unique_piece_keys: self.batch_unique_piece_keys,
            batch_unique_bag_keys: self.batch_unique_bag_keys,
            resolved_children_processed: self.resolved_children_processed,
            batched_parent_links_appended: self.batched_parent_links_appended,
            parent_segments_allocated: self.parent_segments.len(),
            parent_links_stored: self.parent_edges.len(),
            avg_parent_segments_per_bag: if self.bag_nodes.is_empty() {
                0.0
            } else {
                self.parent_segments.len() as f64 / self.bag_nodes.len() as f64
            },
            optimistic_policy_entries: self.optimistic_policy.len(),
            optimistic_ranked_policy_entries: self.optimistic_policy.len(),
            optimistic_ranked_policy_capacity: Self::OPTIMISTIC_POLICY_CAPACITY,
            optimistic_policy_hits: self.optimistic_policy_hits,
            optimistic_policy_misses: self.optimistic_policy_misses,
            optimistic_repairs_attempted: self.optimistic_repairs_attempted,
            optimistic_repairs_succeeded: self.optimistic_repairs_succeeded,
            optimistic_repairs_failed: self.optimistic_repairs_failed,
            optimistic_repairs_promoted: self.optimistic_repairs_promoted,
            optimistic_repairs_demoted: self.optimistic_repairs_demoted,
            optimistic_reused_ranked_candidates: self.optimistic_reused_ranked_candidates,
            optimistic_active_parent_links: self.optimistic_active_parent_links,
            optimistic_popularity_nonzero_bags: self
                .optimistic_successor_popularity
                .iter()
                .filter(|&&value| value > 0)
                .count(),
            optimistic_max_successor_popularity: self
                .optimistic_successor_popularity
                .iter()
                .copied()
                .max()
                .unwrap_or(0),
            optimistic_avg_successor_popularity: {
                let nonzero = self
                    .optimistic_successor_popularity
                    .iter()
                    .copied()
                    .filter(|&value| value > 0)
                    .collect::<Vec<_>>();
                if nonzero.is_empty() {
                    0.0
                } else {
                    nonzero.iter().sum::<usize>() as f64 / nonzero.len() as f64
                }
            },
            optimistic_reorders_using_popularity: self.optimistic_reorders_using_popularity,
            optimistic_avg_candidates_scanned_per_repair: if self.optimistic_repairs_attempted == 0
            {
                0.0
            } else {
                self.optimistic_repair_candidates_scanned as f64
                    / self.optimistic_repairs_attempted as f64
            },
            optimistic_avg_rank_position_chosen: if self.optimistic_rank_position_count == 0 {
                0.0
            } else {
                self.optimistic_rank_position_sum as f64
                    / self.optimistic_rank_position_count as f64
            },
            optimistic_max_repairs_on_single_piece: self.optimistic_max_repairs_on_single_piece,
            optimistic_activated_live_bags,
            optimistic_dormant_child_targets,
            optimistic_activated_frontier,
            optimistic_novel_activations: self.optimistic_novel_activations,
            optimistic_reuse_activations: self.optimistic_reuse_activations,
            optimistic_repairs_reusing_known_successor: self
                .optimistic_repairs_reusing_known_successor,
            optimistic_repairs_forcing_novel_activation: self
                .optimistic_repairs_forcing_novel_activation,
            optimistic_fallback_triggered: self.optimistic_fallback_triggered,
            optimistic_expand_proposal_secs: self.optimistic_expand_proposal_secs,
            optimistic_repair_proposal_secs: self.optimistic_repair_proposal_secs,
            optimistic_repair_batches_processed: self.optimistic_repair_batches_processed,
            optimistic_expand_proposals_revalidated: self.optimistic_expand_proposals_revalidated,
            optimistic_repair_proposals_revalidated: self.optimistic_repair_proposals_revalidated,
            candidate_core_count: self.candidate_core_count,
            largest_scc_bags: self.largest_scc_bags,
            largest_scc_pieces: self.largest_scc_pieces,
            closed_core_found: self.closed_core_found,
            core_certification_secs: self.core_certification_secs,
            bridge_search_secs: self.bridge_search_secs,
            stage_secs: self.total_stage_secs,
            commit_secs: self.total_commit_secs,
            template_member_bags: 0,
            template_member_pieces: 0,
            template_signature_count: 0,
            largest_template_scc_bags: 0,
            largest_template_scc_pieces: 0,
            root_reaches_template_family: false,
            template_attractor_bags: 0,
            template_closure_failed_piece_counts: [0; 7],
        }
    }

    fn root_result(&self) -> RootProofResult {
        let root = &self.bag_nodes[self.root_bag as usize];
        let mut best_action = [None; 7];
        let mut failing_pieces = Vec::new();
        for branch in piece_branches(root.key.bag) {
            let piece_idx = branch.piece.index() as usize;
            let Some(piece_id) = root.piece_nodes[piece_idx] else {
                failing_pieces.push(branch.piece);
                continue;
            };
            let piece = &self.piece_nodes[piece_id as usize];
            match piece.resolution {
                NodeResolution::Won => best_action[piece_idx] = piece.best_action,
                NodeResolution::Lost => failing_pieces.push(branch.piece),
            }
        }

        RootProofResult {
            conclusion: match root.resolution {
                NodeResolution::Won => SolveConclusion::Yes,
                NodeResolution::Lost => SolveConclusion::No,
            },
            result: root.resolution,
            best_action,
            failing_pieces,
        }
    }

    fn approx_bytes_lower_bound(&self) -> usize {
        self.bag_nodes.len() * std::mem::size_of::<BagNode>()
            + self.piece_nodes.len() * std::mem::size_of::<PieceNode>()
            + self.deduped_edges * std::mem::size_of::<PieceChild>()
            + self.parent_edges.len() * std::mem::size_of::<PieceNodeId>()
            + self.parent_segments.len() * std::mem::size_of::<ParentSegment>()
    }

    fn compute_board_features(&self, board: TetrisBoard) -> BoardFeatures {
        let heights = board.heights();
        BoardFeatures {
            height: board.height(),
            holes: board.total_holes(),
            cells: board.count(),
            roughness: board_surface_roughness(board),
            height_spread: board_height_spread(board),
            turning_points: Self::count_turning_points(heights),
            max_well_depth: Self::max_well_depth(heights),
        }
    }

    fn board_features(&self, board: TetrisBoard) -> BoardFeatures {
        if let Some(features) = self.board_features.get(&board) {
            self.board_feature_cache_hits
                .fetch_add(1, AtomicOrdering::Relaxed);
            return *features.value();
        }
        self.board_feature_cache_misses
            .fetch_add(1, AtomicOrdering::Relaxed);
        let features = self.compute_board_features(board);
        self.board_features.insert(board, features);
        features
    }

    fn expand_priority(&mut self, board: TetrisBoard) -> u64 {
        let features = self.board_features(board);
        let fragile = ((features.height as u64) << 48)
            | ((features.holes as u64) << 32)
            | ((features.cells as u64) << 16)
            | ((features.roughness as u64) << 8)
            | (features.height_spread as u64);
        match self.frontier_mode {
            FrontierMode::FragileFirst => fragile,
            FrontierMode::StableFirst => u64::MAX - fragile,
        }
    }

    fn geometry_children(
        &mut self,
        board: TetrisBoard,
        piece: TetrisPiece,
    ) -> Arc<[GeometryChild]> {
        let key = GeometryKey { board, piece };
        if let Some(children) = self.geometry_cache.get(&key) {
            self.geometry_cache_hits
                .fetch_add(1, AtomicOrdering::Relaxed);
            return Arc::clone(children.value());
        }
        self.geometry_cache_misses
            .fetch_add(1, AtomicOrdering::Relaxed);
        let children = Arc::<[GeometryChild]>::from(Self::compute_geometry_children_for(
            board,
            piece,
            self.config.admissibility,
        ));
        for child in children.iter().copied() {
            let _ = self.board_features(child.board);
        }
        let children_for_cache = Arc::clone(&children);
        match self.geometry_cache.entry(key) {
            dashmap::mapref::entry::Entry::Occupied(entry) => Arc::clone(entry.get()),
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(children_for_cache);
                children
            }
        }
    }

    fn compute_geometry_children_for(
        board: TetrisBoard,
        piece: TetrisPiece,
        admissibility: crate::config::BoardAdmissibility,
    ) -> Vec<GeometryChild> {
        let mut children = Vec::new();
        for &placement in TetrisPiecePlacement::all_from_piece(piece) {
            let mut next_board = board;
            let outcome = next_board.apply_piece_placement(placement);
            if outcome.is_lost == IsLost::LOST || !board_is_admissible(next_board, admissibility) {
                continue;
            }
            children.push(GeometryChild {
                placement: pack_placement(placement),
                board: next_board,
            });
        }

        children.sort_unstable_by_key(|child| (child.board, child.placement));
        children.dedup_by(|right, left| right.board == left.board);

        let mut ranked_children = children
            .into_iter()
            .map(|child| (child, Self::compute_board_features_for(child.board)))
            .collect::<Vec<_>>();
        ranked_children.sort_by_key(|(child, features)| {
            (
                features.height,
                features.holes,
                features.cells,
                features.roughness,
                features.height_spread,
                child.placement,
            )
        });
        ranked_children
            .into_iter()
            .map(|(child, _)| child)
            .collect()
    }

    fn compute_board_features_for(board: TetrisBoard) -> BoardFeatures {
        let heights = board.heights();
        BoardFeatures {
            height: board.height(),
            holes: board.total_holes(),
            cells: board.count(),
            roughness: board_surface_roughness(board),
            height_spread: board_height_spread(board),
            turning_points: Self::count_turning_points(heights),
            max_well_depth: Self::max_well_depth(heights),
        }
    }

    fn board_signature_for(board: TetrisBoard) -> BoardSignature {
        let heights = board.heights();
        BoardSignature {
            heights,
            holes: board.total_holes(),
            roughness: board_surface_roughness(board),
            height_spread: board_height_spread(board),
            turning_points: Self::count_turning_points(heights),
            max_well_depth: Self::max_well_depth(heights),
        }
    }

    fn count_turning_points(heights: [u32; 10]) -> u32 {
        let mut prev_sign = 0i32;
        let mut turning_points = 0u32;
        for window in heights.windows(2) {
            let sign = (window[1] as i32 - window[0] as i32).signum();
            if sign == 0 {
                continue;
            }
            if prev_sign != 0 && sign != prev_sign {
                turning_points += 1;
            }
            prev_sign = sign;
        }
        turning_points
    }

    fn max_well_depth(heights: [u32; 10]) -> u32 {
        let mut max_depth = 0u32;
        for idx in 0..heights.len() {
            let left = if idx > 0 {
                Some(heights[idx - 1])
            } else {
                None
            };
            let right = if idx + 1 < heights.len() {
                Some(heights[idx + 1])
            } else {
                None
            };
            let support = match (left, right) {
                (Some(l), Some(r)) => l.min(r),
                (Some(l), None) => l,
                (None, Some(r)) => r,
                (None, None) => heights[idx],
            };
            if support > heights[idx] {
                max_depth = max_depth.max(support - heights[idx]);
            }
        }
        max_depth
    }
}

pub fn solve_root(config: SolverConfig) -> Result<ProofSolveResult> {
    ProofSolver::new(config).solve()
}

pub fn solve_root_with_options(
    config: SolverConfig,
    options: SolveOptions,
) -> Result<ProofSolveResult> {
    match options.mode {
        SolveMode::Full => {
            ProofSolver::new_with_mode(config, SolveMode::Full, FrontierMode::FragileFirst).solve()
        }
        SolveMode::CoreFirst => ProofSolver::solve_core_first(config, options),
        SolveMode::TemplateKernel => ProofSolver::solve_template_kernel(config, options),
        SolveMode::Optimistic => ProofSolver::solve_optimistic(config, options),
    }
}

#[cfg(test)]
fn solve_root_with_parallelism(
    config: SolverConfig,
    parallel_local_expansion: bool,
) -> Result<ProofSolveResult> {
    let mut solver = ProofSolver::new(config);
    solver.parallel_local_expansion = parallel_local_expansion;
    solver.solve()
}

#[cfg(test)]
fn solve_root_with_execution(
    config: SolverConfig,
    parallel_local_expansion: bool,
    staged_merge: bool,
) -> Result<ProofSolveResult> {
    let mut solver = ProofSolver::new(config);
    solver.parallel_local_expansion = parallel_local_expansion;
    solver.staged_merge = staged_merge;
    solver.solve()
}

pub fn verify_proof(result: &ProofSolveResult) -> ProofVerification {
    use std::collections::HashSet;

    let mut verification = ProofVerification::default();

    for (bag_id, bag) in result.bag_nodes.iter().enumerate() {
        match bag.resolution {
            NodeResolution::Won => {
                verification.bag_nodes_checked += 1;
                for branch in piece_branches(bag.key.bag) {
                    let Some(piece_id) = bag.piece_nodes[branch.piece.index() as usize] else {
                        verification.resolution_failures += 1;
                        continue;
                    };
                    let piece = &result.piece_nodes[piece_id as usize];
                    if piece.resolution != NodeResolution::Won {
                        verification.resolution_failures += 1;
                    }
                    if piece.parent_bag as usize != bag_id {
                        verification.resolution_failures += 1;
                    }
                    if piece.key.board != bag.key.board
                        || piece.key.bag != bag.key.bag
                        || piece.key.piece != branch.piece
                    {
                        verification.resolution_failures += 1;
                    }
                }
            }
            NodeResolution::Lost => {
                let has_failing_branch = piece_branches(bag.key.bag).any(|branch| {
                    bag.piece_nodes[branch.piece.index() as usize].is_none_or(|piece_id| {
                        result.piece_nodes[piece_id as usize].resolution == NodeResolution::Lost
                    })
                });
                if !has_failing_branch {
                    verification.resolution_failures += 1;
                }
            }
        }
    }

    for (piece_id, piece) in result.piece_nodes.iter().enumerate() {
        if piece.parent_bag as usize >= result.bag_nodes.len() {
            verification.resolution_failures += 1;
            continue;
        }
        let parent_bag = &result.bag_nodes[piece.parent_bag as usize];
        if parent_bag.piece_nodes[piece.key.piece.index() as usize] != Some(piece_id as PieceNodeId)
        {
            verification.resolution_failures += 1;
        }

        let unique_succs = piece
            .children
            .iter()
            .map(|child| child.succ)
            .collect::<HashSet<_>>();
        if piece.unique_succ_count as usize != unique_succs.len() {
            verification.resolution_failures += 1;
        }
        let live_unique_succs = piece
            .children
            .iter()
            .filter_map(|child| {
                (result.bag_nodes[child.succ as usize].resolution == NodeResolution::Won)
                    .then_some(child.succ)
            })
            .collect::<HashSet<_>>();
        if piece.alive_succ_count as usize != live_unique_succs.len() {
            verification.resolution_failures += 1;
        }

        match piece.resolution {
            NodeResolution::Won => {
                verification.piece_nodes_checked += 1;
                let Some(best_action) = piece.best_action else {
                    verification.witness_failures += 1;
                    continue;
                };
                if best_action as usize >= tetris_game::TetrisPiecePlacement::NUM_PLACEMENTS {
                    verification.witness_failures += 1;
                    continue;
                }
                if tetris_game::TetrisPiecePlacement::from_index(best_action).piece
                    != piece.key.piece
                {
                    verification.witness_failures += 1;
                    continue;
                }
                let witness_is_winning = piece.children.iter().any(|child| {
                    child.placement == best_action
                        && result.bag_nodes[child.succ as usize].resolution == NodeResolution::Won
                });
                if !witness_is_winning {
                    verification.witness_failures += 1;
                }
            }
            NodeResolution::Lost => {
                let has_surviving_child = piece.children.iter().any(|child| {
                    result.bag_nodes[child.succ as usize].resolution == NodeResolution::Won
                });
                if has_surviving_child {
                    verification.resolution_failures += 1;
                }
            }
        }
    }

    let root_bag = &result.bag_nodes[result.root_bag as usize];
    if result.root_result.result != root_bag.resolution {
        verification.resolution_failures += 1;
    }
    match result.root_result.result {
        NodeResolution::Won => {
            if result.root_result.conclusion != SolveConclusion::Yes {
                verification.resolution_failures += 1;
            }
            for branch in piece_branches(root_bag.key.bag) {
                let Some(piece_id) = root_bag.piece_nodes[branch.piece.index() as usize] else {
                    verification.resolution_failures += 1;
                    continue;
                };
                if result.root_result.best_action[branch.piece.index() as usize]
                    != result.piece_nodes[piece_id as usize].best_action
                {
                    verification.witness_failures += 1;
                }
            }
            if !result.root_result.failing_pieces.is_empty() {
                verification.resolution_failures += 1;
            }
        }
        NodeResolution::Lost => {
            if result.root_result.conclusion != SolveConclusion::No {
                verification.resolution_failures += 1;
            }
            let expected_failing = piece_branches(root_bag.key.bag)
                .filter_map(|branch| {
                    root_bag.piece_nodes[branch.piece.index() as usize]
                        .and_then(|piece_id| {
                            (result.piece_nodes[piece_id as usize].resolution
                                == NodeResolution::Lost)
                                .then_some(branch.piece)
                        })
                        .or(Some(branch.piece))
                })
                .collect::<Vec<_>>();
            if result.root_result.failing_pieces != expected_failing {
                verification.resolution_failures += 1;
            }
        }
    }

    verification
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct OracleWinningSet {
    winning: Vec<bool>,
}

#[cfg(test)]
fn oracle_compute_winning_set(config: SolverConfig) -> OracleWinningSet {
    use crate::universe::UniverseBuilder;

    let universe = UniverseBuilder::build(config).expect("oracle universe should build");
    let mut winning = vec![true; universe.states.len()];

    loop {
        let mut changed = false;
        let mut to_kill = Vec::new();
        for state_id in 0..universe.states.len() {
            if !winning[state_id] {
                continue;
            }
            let state_key = universe.states[state_id];
            let index = &universe.state_index[state_id];
            let all_pieces_survive = piece_branches(state_key.bag).all(|branch| {
                let range = index.piece_ranges[branch.piece.index() as usize];
                universe
                    .edge_slice(range)
                    .iter()
                    .any(|edge| winning[edge.succ as usize])
            });
            if !all_pieces_survive {
                to_kill.push(state_id);
            }
        }
        if to_kill.is_empty() {
            break;
        }
        for state_id in to_kill {
            if winning[state_id] {
                winning[state_id] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    OracleWinningSet { winning }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::config::{BoardAdmissibility, RootStateConfig};
    use crate::retrograde::{evaluate_root as evaluate_recovery_root, solve as solve_recovery};
    use crate::universe::UniverseBuilder;
    use crate::verifier::verify_replay;

    use super::*;

    fn placements_for_piece(piece: TetrisPiece, count: usize) -> Vec<PackedPlacement> {
        (0..117)
            .map(TetrisPiecePlacement::from_index)
            .filter(|placement| placement.piece == piece)
            .take(count)
            .map(pack_placement)
            .collect()
    }

    fn test_bag_node(
        key: BagKey,
        resolution: NodeResolution,
        piece_nodes: [Option<PieceNodeId>; 7],
    ) -> BagNode {
        BagNode {
            key,
            resolution,
            expanded: true,
            parents: Vec::new(),
            piece_nodes,
            activated: true,
            branch_order: [u8::MAX; 7],
            branch_order_len: 0,
            next_branch_idx: 0,
            expanded_branch_count: 0,
            branch_order_initialized: false,
        }
    }

    fn test_piece_node(
        key: PieceKey,
        parent_bag: BagNodeId,
        resolution: NodeResolution,
        best_action: Option<PackedPlacement>,
        best_child: Option<BagNodeId>,
    ) -> PieceNode {
        PieceNode {
            key,
            parent_bag,
            resolution,
            children: best_action
                .zip(best_child)
                .map(|(placement, succ)| vec![PieceChild { placement, succ }])
                .unwrap_or_default(),
            unique_succ_count: usize::from(best_child.is_some()) as u32,
            alive_succ_count: usize::from(best_child.is_some()) as u32,
            best_action,
            best_child,
            active_child_slot: best_child.map(|_| 0),
            optimistic_children: Vec::new(),
            next_repair_cursor: 0,
            repair_attempts: 0,
            optimistic_exhausted: false,
            optimistic_repair_pending: false,
        }
    }

    fn single_o_config() -> SolverConfig {
        SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 2,
                max_holes: u32::MAX,
                max_cells: u32::MAX,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: RootStateConfig {
                board: TetrisBoard::new(),
                bag: TetrisPieceBagState::from(u8::from(TetrisPiece::O_PIECE)),
            },
        }
    }

    fn malformed_o_won_result(
        config: SolverConfig,
        placement: TetrisPiecePlacement,
    ) -> ProofSolveResult {
        let root_key = BagKey {
            board: config.root.board,
            bag: config.root.bag,
        };
        let child_key = BagKey {
            board: TetrisBoard::new(),
            bag: TetrisPieceBagState::new(),
        };
        let packed = pack_placement(placement);
        let mut root_pieces = [None; 7];
        root_pieces[TetrisPiece::O_PIECE.index() as usize] = Some(0);

        ProofSolveResult {
            solve_mode: SolveMode::Full,
            used_full_fallback: false,
            root_bag: 0,
            bag_nodes: vec![
                test_bag_node(root_key, NodeResolution::Won, root_pieces),
                test_bag_node(child_key, NodeResolution::Won, [None; 7]),
            ],
            piece_nodes: vec![test_piece_node(
                PieceKey {
                    board: root_key.board,
                    bag: root_key.bag,
                    piece: TetrisPiece::O_PIECE,
                },
                0,
                NodeResolution::Won,
                Some(packed),
                Some(1),
            )],
            metrics: ProofMetrics::default(),
            discover_secs: 0.0,
            propagate_secs: 0.0,
            total_secs: 0.0,
            root_result: RootProofResult {
                conclusion: SolveConclusion::Yes,
                result: NodeResolution::Won,
                best_action: {
                    let mut actions = [None; 7];
                    actions[TetrisPiece::O_PIECE.index() as usize] = Some(packed);
                    actions
                },
                failing_pieces: Vec::new(),
            },
        }
    }

    #[test]
    fn closed_winning_set_is_no_when_no_admissible_move_exists() {
        let result = solve_root(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 0,
                max_holes: 0,
                max_cells: 0,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        })
        .expect("solver should succeed");

        assert_eq!(result.root_result.result, NodeResolution::Lost);
        assert_eq!(result.root_result.failing_pieces.len(), 7);
        assert_eq!(result.metrics.winning_count, 0);
        assert_eq!(result.metrics.losing_count, result.metrics.bag_node_count);
        assert_eq!(
            result.metrics.piece_losing_count,
            result.metrics.piece_node_count
        );
        assert_eq!(result.metrics.bags_expanded, 1);
        assert_eq!(result.metrics.branches_expanded, 1);
        assert_eq!(result.metrics.branches_skipped_due_to_parent_death, 6);
        assert_eq!(result.metrics.bags_killed_by_first_failed_branch, 1);
        assert_eq!(result.metrics.skipped_dead_bags, 1);
        assert!(verify_proof(&result).is_clean());
    }

    #[test]
    fn zero_successor_piece_branches_die_immediately_during_discovery() {
        let result = solve_root(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 0,
                max_holes: 0,
                max_cells: 0,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        })
        .expect("solver should succeed");

        assert_eq!(result.metrics.bag_node_count, 1);
        assert_eq!(result.metrics.piece_node_count, 1);
        assert_eq!(result.metrics.deduped_edge_count, 0);
        assert_eq!(result.metrics.bags_expanded, 1);
        assert_eq!(result.metrics.branches_expanded, 1);
        assert_eq!(result.metrics.branches_skipped_due_to_parent_death, 6);
        assert_eq!(result.root_result.result, NodeResolution::Lost);
        assert_eq!(result.piece_nodes.len(), 1);
        let piece = &result.piece_nodes[0];
        assert_eq!(piece.children.len(), 0);
        assert_eq!(piece.unique_succ_count, 0);
        assert_eq!(piece.alive_succ_count, 0);
        assert_eq!(piece.resolution, NodeResolution::Lost);
    }

    #[test]
    fn closed_winning_set_reports_precise_failing_root_pieces() {
        let mut board = TetrisBoard::new();
        for col in 4..10 {
            board.set_bit(col, 0);
        }
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: 0,
                max_cells: 10,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig {
                board,
                bag: TetrisPieceBagState::from(
                    u8::from(TetrisPiece::I_PIECE) | u8::from(TetrisPiece::T_PIECE),
                ),
            },
        };
        let result = solve_root(config).expect("solver should succeed");
        let oracle = oracle_compute_winning_set(config);
        let universe = UniverseBuilder::build(config).expect("oracle universe should build");
        let state = universe.state_key(universe.root_state_id);
        let index = universe.state_index(universe.root_state_id);
        let expected_failing_pieces = piece_branches(state.bag)
            .filter_map(|branch| {
                let range = index.piece_ranges[branch.piece.index() as usize];
                let has_surviving_child = universe
                    .edge_slice(range)
                    .iter()
                    .any(|edge| oracle.winning[edge.succ as usize]);
                (!has_surviving_child).then_some(branch.piece)
            })
            .collect::<Vec<_>>();

        assert_eq!(result.root_result.result, NodeResolution::Lost);
        assert_eq!(result.root_result.failing_pieces, expected_failing_pieces);
    }

    #[test]
    fn closed_winning_set_matches_naive_oracle_on_tiny_losing_universe() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 0,
                max_holes: 0,
                max_cells: 0,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        };
        let proof = solve_root(config).expect("solver should succeed");
        let oracle = oracle_compute_winning_set(config);

        assert_eq!(proof.root_result.result, NodeResolution::Lost);
        assert_eq!(proof.metrics.winning_count, 0);
        assert_eq!(oracle.winning.iter().filter(|&&alive| alive).count(), 0);
        assert_eq!(proof.metrics.losing_count, oracle.winning.len());
    }

    #[test]
    fn closed_winning_set_matches_naive_oracle_on_small_universe() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: 0,
                max_cells: 8,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        };
        let proof = solve_root(config).expect("solver should succeed");
        let oracle = oracle_compute_winning_set(config);
        let root_alive = oracle.winning[0];

        assert_eq!(proof.root_result.result == NodeResolution::Won, root_alive);
        assert_eq!(
            proof.metrics.winning_count,
            oracle.winning.iter().filter(|&&alive| alive).count()
        );
    }

    #[test]
    fn closed_winning_set_differs_from_recovery_semantics() {
        let mut board = TetrisBoard::new();
        for col in 4..10 {
            board.set_bit(col, 0);
        }
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: 0,
                max_cells: 10,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig {
                board,
                bag: TetrisPieceBagState::from(u8::from(TetrisPiece::I_PIECE)),
            },
        };
        let closed = solve_root(config).expect("solver should succeed");
        let universe =
            UniverseBuilder::build(config).expect("recovery oracle universe should build");
        let recovery = solve_recovery(&universe);

        assert_eq!(evaluate_recovery_root(&universe, &recovery).is_some(), true);
        assert_eq!(closed.root_result.result, NodeResolution::Lost);
    }

    #[test]
    fn max_holes_zero_rejects_required_s_branch_from_empty_root() {
        let result = solve_root(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: 0,
                max_cells: 4,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        })
        .expect("solver should succeed");

        assert_eq!(result.root_result.result, NodeResolution::Lost);
        assert!(
            result
                .root_result
                .failing_pieces
                .contains(&TetrisPiece::S_PIECE)
        );

        let root_piece_nodes = result.bag_nodes[result.root_bag as usize].piece_nodes;
        let maybe_s_piece_node = root_piece_nodes[TetrisPiece::S_PIECE.index() as usize];
        if let Some(s_piece_node) = maybe_s_piece_node {
            assert_eq!(result.piece_nodes[s_piece_node as usize].children.len(), 0);
            assert_eq!(
                result.piece_nodes[s_piece_node as usize].resolution,
                NodeResolution::Lost
            );
        }
        assert!(verify_proof(&result).is_clean());
    }

    #[test]
    fn replay_verifier_rejects_witness_for_wrong_piece() {
        let config = single_o_config();
        let wrong_piece_placement = TetrisPiecePlacement::all_from_piece(TetrisPiece::I_PIECE)[0];
        let result = malformed_o_won_result(config, wrong_piece_placement);

        let verification = verify_replay(config, &result);

        assert!(verification.replay_failures > 0);
        assert!(!verification.is_clean());
    }

    #[test]
    fn replay_verifier_rejects_out_of_range_witness_index_without_panicking() {
        let config = single_o_config();
        let placement = TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE)[0];
        let mut result = malformed_o_won_result(config, placement);
        result.piece_nodes[0].best_action = Some(u8::MAX);

        let verification = verify_replay(config, &result);

        assert!(verification.replay_failures > 0);
        assert!(!verification.is_clean());
    }

    #[test]
    fn replay_verifier_rejects_corrupt_successor_bag() {
        let config = single_o_config();
        let placement = TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE)[0];
        let mut result = malformed_o_won_result(config, placement);
        let mut expected_board = config.root.board;
        let outcome = expected_board.apply_piece_placement(placement);
        assert_eq!(outcome.is_lost, IsLost::NOT_LOST);
        result.bag_nodes[1].key.board = expected_board;
        result.bag_nodes[1].key.bag = TetrisPieceBagState::from(u8::from(TetrisPiece::I_PIECE));

        let verification = verify_replay(config, &result);

        assert!(verification.bag_transition_failures > 0);
        assert!(!verification.is_clean());
    }

    #[test]
    fn replay_verifier_rejects_lost_piece_with_winning_successor() {
        let config = single_o_config();
        let placement = TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE)[0];
        let mut child_board = config.root.board;
        let outcome = child_board.apply_piece_placement(placement);
        assert_eq!(outcome.is_lost, IsLost::NOT_LOST);
        let child_key = BagKey {
            board: child_board,
            bag: next_bag_state(config.root.bag, TetrisPiece::O_PIECE)
                .expect("single-piece bag should refill"),
        };

        let result = ProofSolveResult {
            solve_mode: SolveMode::Full,
            used_full_fallback: false,
            root_bag: 0,
            bag_nodes: vec![test_bag_node(child_key, NodeResolution::Won, [None; 7])],
            piece_nodes: vec![test_piece_node(
                PieceKey {
                    board: config.root.board,
                    bag: config.root.bag,
                    piece: TetrisPiece::O_PIECE,
                },
                0,
                NodeResolution::Lost,
                None,
                None,
            )],
            metrics: ProofMetrics::default(),
            discover_secs: 0.0,
            propagate_secs: 0.0,
            total_secs: 0.0,
            root_result: RootProofResult {
                conclusion: SolveConclusion::No,
                result: NodeResolution::Lost,
                best_action: [None; 7],
                failing_pieces: vec![TetrisPiece::O_PIECE],
            },
        };

        let verification = verify_replay(config, &result);

        assert!(verification.resolution_failures > 0);
        assert!(!verification.is_clean());
    }

    #[test]
    fn geometry_cache_reuses_board_piece_successors() {
        let mut solver = ProofSolver::new(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 2,
                max_holes: u32::MAX,
                max_cells: u32::MAX,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        });

        let board = TetrisBoard::new();
        let first_len = solver.geometry_children(board, TetrisPiece::I_PIECE).len();
        assert_eq!(solver.geometry_cache.len(), 1);
        let second_len = solver.geometry_children(board, TetrisPiece::I_PIECE).len();

        assert_eq!(first_len, second_len);
        assert_eq!(solver.geometry_cache.len(), 1);
    }

    #[test]
    fn geometry_cache_dedups_duplicate_successor_boards_and_keeps_smallest_placement() {
        let mut board = TetrisBoard::new();
        for &col in &[0usize, 1, 2, 3, 4, 5, 9] {
            board.set_bit(col, 0);
            board.set_bit(col, 1);
        }
        for &col in &[1usize, 2, 5] {
            board.set_bit(col, 2);
        }
        board.set_bit(4, 3);

        let mut dup_board = TetrisBoard::new();
        for &col in &[0usize, 1, 2, 3, 4, 5, 7, 9] {
            dup_board.set_bit(col, 0);
        }
        for &col in &[1usize, 2, 5] {
            dup_board.set_bit(col, 1);
        }
        dup_board.set_bit(4, 2);

        let min_placement = 66;
        let alt_placement = 83;
        let piece = TetrisPiecePlacement::from_index(min_placement).piece;
        let mut solver = ProofSolver::new(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 6,
                max_holes: u32::MAX,
                max_cells: u32::MAX,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        });

        let children = solver.geometry_children(board, piece);
        let matching = children
            .iter()
            .filter(|child| child.board == dup_board)
            .collect::<Vec<_>>();

        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0].placement, min_placement);
        assert_eq!(TetrisPiecePlacement::from_index(alt_placement).piece, piece);
    }

    #[test]
    fn expand_queue_prioritizes_more_fragile_boards() {
        let mut solver = ProofSolver::new(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 4,
                max_holes: u32::MAX,
                max_cells: u32::MAX,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        });
        solver.expand_queue = BinaryHeap::new();

        let short_board = TetrisBoard::new();
        let mut tall_board = TetrisBoard::new();
        tall_board.set_bit(0, 0);
        tall_board.set_bit(0, 1);

        let short_id = solver.intern_bag(BagKey {
            board: short_board,
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::I_PIECE)),
        });
        let tall_id = solver.intern_bag(BagKey {
            board: tall_board,
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::O_PIECE)),
        });

        let first = solver
            .expand_queue
            .pop()
            .expect("queue should contain entries")
            .bag_id;
        let second = solver
            .expand_queue
            .pop()
            .expect("queue should contain entries")
            .bag_id;

        assert_eq!(first, tall_id);
        assert_eq!(second, short_id);
    }

    #[test]
    fn piece_children_use_unique_successor_bags() {
        let result = solve_root(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: 0,
                max_cells: 8,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        })
        .expect("solver should succeed");

        for piece in &result.piece_nodes {
            let succs = piece
                .children
                .iter()
                .map(|child| child.succ)
                .collect::<Vec<_>>();
            let unique = succs.iter().copied().collect::<HashSet<_>>();
            assert_eq!(succs.len(), unique.len());
            assert_eq!(piece.unique_succ_count as usize, unique.len());
        }
    }

    #[test]
    fn winning_pieces_always_have_a_winning_child_after_solve() {
        let result = solve_root(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 2,
                max_holes: 1,
                max_cells: 8,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        })
        .expect("solver should succeed");

        for piece in &result.piece_nodes {
            if piece.resolution == NodeResolution::Won {
                assert!(piece.children.iter().any(|child| {
                    result.bag_nodes[child.succ as usize].resolution == NodeResolution::Won
                }));
            }
        }
    }

    #[test]
    fn alive_successor_counts_match_current_winning_successors() {
        let result = solve_root(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 2,
                max_holes: 1,
                max_cells: 8,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        })
        .expect("solver should succeed");

        for piece in &result.piece_nodes {
            let winning_unique_succs = piece
                .children
                .iter()
                .filter_map(|child| {
                    (result.bag_nodes[child.succ as usize].resolution == NodeResolution::Won)
                        .then_some(child.succ)
                })
                .collect::<HashSet<_>>();
            assert_eq!(piece.alive_succ_count as usize, winning_unique_succs.len());
        }
    }

    #[test]
    fn winning_bag_parents_match_winning_piece_children() {
        let result = solve_root(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 2,
                max_holes: 1,
                max_cells: 8,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        })
        .expect("solver should succeed");

        for (bag_id, bag) in result.bag_nodes.iter().enumerate() {
            if bag.resolution != NodeResolution::Won {
                continue;
            }
            let mut expected = result
                .piece_nodes
                .iter()
                .enumerate()
                .filter_map(|(piece_id, piece)| {
                    (piece.resolution == NodeResolution::Won
                        && piece
                            .children
                            .iter()
                            .any(|child| child.succ as usize == bag_id))
                    .then_some(piece_id as PieceNodeId)
                })
                .collect::<Vec<_>>();
            let mut actual = bag.parents.clone();
            expected.sort_unstable();
            actual.sort_unstable();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn dependency_count_matches_piece_child_backrefs() {
        let result = solve_root(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 2,
                max_holes: 1,
                max_cells: 8,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        })
        .expect("solver should succeed");

        let expected_dependency_count = result
            .bag_nodes
            .iter()
            .map(|bag| bag.parents.len())
            .sum::<usize>();

        assert_eq!(result.metrics.dependency_count, expected_dependency_count);
    }

    #[test]
    fn verifier_rejects_piece_parent_bag_mismatch() {
        let mut result = solve_root(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: 0,
                max_cells: 8,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        })
        .expect("solver should succeed");

        let piece_id = result
            .piece_nodes
            .iter()
            .position(|_| true)
            .expect("expected at least one piece");
        result.piece_nodes[piece_id].parent_bag = u32::MAX;

        let verification = verify_proof(&result);
        assert!(verification.resolution_failures > 0);
    }

    #[test]
    fn serial_and_parallel_local_expansion_match_on_small_universe() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 2,
                max_holes: 1,
                max_cells: 8,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        };

        let serial = solve_root_with_parallelism(config, false).expect("serial solve should work");
        let parallel =
            solve_root_with_parallelism(config, true).expect("parallel solve should work");

        assert_eq!(serial.root_result, parallel.root_result);
        assert_eq!(serial.metrics.winning_count, parallel.metrics.winning_count);
        assert_eq!(serial.metrics.losing_count, parallel.metrics.losing_count);
        assert_eq!(
            serial.metrics.piece_winning_count,
            parallel.metrics.piece_winning_count
        );
        assert_eq!(
            serial.metrics.piece_losing_count,
            parallel.metrics.piece_losing_count
        );
        assert_eq!(verify_proof(&serial), verify_proof(&parallel));
    }

    #[test]
    fn staged_and_direct_merge_paths_match_on_small_universe() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 2,
                max_holes: 1,
                max_cells: 8,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        };

        let direct =
            solve_root_with_execution(config, true, false).expect("direct merge solve should work");
        let staged =
            solve_root_with_execution(config, true, true).expect("staged merge solve should work");

        assert_eq!(direct.root_result, staged.root_result);
        assert_eq!(direct.metrics.winning_count, staged.metrics.winning_count);
        assert_eq!(direct.metrics.losing_count, staged.metrics.losing_count);
        assert_eq!(
            direct.metrics.piece_winning_count,
            staged.metrics.piece_winning_count
        );
        assert_eq!(
            direct.metrics.piece_losing_count,
            staged.metrics.piece_losing_count
        );
        assert_eq!(verify_proof(&direct), verify_proof(&staged));
    }

    #[test]
    fn staged_and_direct_merge_paths_match_on_immediate_loss_universe() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 0,
                max_holes: 0,
                max_cells: 0,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        };

        let direct =
            solve_root_with_execution(config, true, false).expect("direct merge solve should work");
        let staged =
            solve_root_with_execution(config, true, true).expect("staged merge solve should work");

        assert_eq!(direct.root_result, staged.root_result);
        assert_eq!(direct.metrics.winning_count, staged.metrics.winning_count);
        assert_eq!(direct.metrics.losing_count, staged.metrics.losing_count);
        assert_eq!(
            direct.metrics.branches_expanded,
            staged.metrics.branches_expanded
        );
        assert_eq!(
            direct.metrics.branches_skipped_due_to_parent_death,
            staged.metrics.branches_skipped_due_to_parent_death
        );
        assert_eq!(
            direct.metrics.bags_killed_by_first_failed_branch,
            staged.metrics.bags_killed_by_first_failed_branch
        );
        assert_eq!(verify_proof(&direct), verify_proof(&staged));
    }

    #[test]
    fn staged_merge_matches_direct_with_serial_local_expansion() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 2,
                max_holes: 1,
                max_cells: 8,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        };

        let direct = solve_root_with_execution(config, false, false)
            .expect("direct serial merge solve should work");
        let staged = solve_root_with_execution(config, false, true)
            .expect("staged serial merge solve should work");

        assert_eq!(direct.root_result, staged.root_result);
        assert_eq!(direct.metrics.winning_count, staged.metrics.winning_count);
        assert_eq!(direct.metrics.losing_count, staged.metrics.losing_count);
        assert_eq!(
            direct.metrics.piece_winning_count,
            staged.metrics.piece_winning_count
        );
        assert_eq!(
            direct.metrics.piece_losing_count,
            staged.metrics.piece_losing_count
        );
        assert_eq!(verify_proof(&direct), verify_proof(&staged));
    }

    #[test]
    fn core_first_fallback_matches_full_solver_on_small_universe() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: 0,
                max_cells: 8,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        };

        let full = solve_root(config).expect("full solver should succeed");
        let core_first = solve_root_with_options(
            config,
            SolveOptions {
                mode: SolveMode::CoreFirst,
                core_prepass_expansions: 64,
                core_certify_interval: 16,
                ..SolveOptions::default()
            },
        )
        .expect("core-first solver should succeed");

        assert_eq!(core_first.root_result, full.root_result);
        assert_eq!(verify_proof(&core_first), verify_proof(&full));
        assert_eq!(core_first.root_result.result, NodeResolution::Lost);
        assert!(core_first.used_full_fallback);
        assert_eq!(core_first.solve_mode, SolveMode::CoreFirst);
    }

    #[test]
    fn optimistic_matches_full_solver_on_small_universe() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: 0,
                max_cells: 8,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        };

        let full = solve_root(config).expect("full solver should succeed");
        let optimistic = solve_root_with_options(
            config,
            SolveOptions {
                mode: SolveMode::Optimistic,
                ..SolveOptions::default()
            },
        )
        .expect("optimistic solver should succeed");

        assert_eq!(optimistic.root_result, full.root_result);
        assert_eq!(optimistic.metrics.winning_count, full.metrics.winning_count);
        assert_eq!(optimistic.metrics.losing_count, full.metrics.losing_count);
        assert_eq!(
            optimistic.metrics.piece_winning_count,
            full.metrics.piece_winning_count
        );
        assert_eq!(
            optimistic.metrics.piece_losing_count,
            full.metrics.piece_losing_count
        );
        assert!(verify_proof(&optimistic).is_clean());
    }

    #[test]
    fn optimistic_can_fallback_to_full_when_repair_budget_is_tiny() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: 0,
                max_cells: 8,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        };

        let optimistic = solve_root_with_options(
            config,
            SolveOptions {
                mode: SolveMode::Optimistic,
                optimistic_max_repairs_per_piece: 0,
                optimistic_max_global_repairs: 0,
                optimistic_fallback_on_thrash: true,
                ..SolveOptions::default()
            },
        )
        .expect("optimistic fallback solver should succeed");

        assert_eq!(optimistic.solve_mode, SolveMode::Optimistic);
        assert!(verify_proof(&optimistic).is_clean());
        assert!(optimistic.used_full_fallback || !optimistic.metrics.optimistic_fallback_triggered);
    }

    #[test]
    fn optimistic_repair_uses_next_repair_cursor_and_updates_active_link_accounting() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 4,
                max_holes: 1,
                max_cells: 12,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        };
        let mut solver =
            ProofSolver::new_with_mode(config, SolveMode::Optimistic, FrontierMode::StableFirst);

        let live0 = solver.intern_bag(BagKey {
            board: TetrisBoard::new(),
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::O_PIECE)),
        });
        let doomed = solver.intern_bag(BagKey {
            board: {
                let mut board = TetrisBoard::new();
                board.set_bit(0, 0);
                board
            },
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::I_PIECE)),
        });
        let live2 = solver.intern_bag(BagKey {
            board: {
                let mut board = TetrisBoard::new();
                board.set_bit(1, 0);
                board
            },
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::T_PIECE)),
        });
        let piece_id = solver.intern_piece(
            PieceKey {
                board: solver.config.root.board,
                bag: solver.config.root.bag,
                piece: TetrisPiece::L_PIECE,
            },
            solver.root_bag,
        );

        solver.piece_nodes[piece_id as usize].children = vec![
            PieceChild {
                placement: pack_placement(TetrisPiecePlacement::from_index(88)),
                succ: live0,
            },
            PieceChild {
                placement: pack_placement(TetrisPiecePlacement::from_index(89)),
                succ: doomed,
            },
            PieceChild {
                placement: pack_placement(TetrisPiecePlacement::from_index(90)),
                succ: live2,
            },
        ];
        solver.piece_nodes[piece_id as usize].unique_succ_count = 3;
        solver.piece_nodes[piece_id as usize].alive_succ_count = 3;
        solver.piece_nodes[piece_id as usize].active_child_slot = Some(1);
        solver.piece_nodes[piece_id as usize].next_repair_cursor = 2;
        solver.append_parent_links(doomed, &[piece_id]);
        solver.optimistic_active_parent_links = 1;

        solver.kill_bag(doomed);
        solver.drain_death_queues();
        assert_eq!(solver.optimistic_active_parent_links, 0);
        assert_eq!(solver.optimistic_repair_queue.len(), 1);

        let completed = solver.process_optimistic_repairs(SolveOptions {
            mode: SolveMode::Optimistic,
            ..SolveOptions::default()
        });
        assert!(completed);
        assert_eq!(solver.optimistic_repairs_attempted, 1);
        assert_eq!(solver.optimistic_repairs_succeeded, 1);
        assert_eq!(solver.optimistic_repairs_failed, 0);
        assert_eq!(solver.optimistic_active_parent_links, 1);
        assert_eq!(
            solver.piece_nodes[piece_id as usize].active_child_slot,
            Some(2),
            "repair should continue from next_repair_cursor instead of rescanning from slot 0"
        );
        assert_eq!(solver.piece_nodes[piece_id as usize].next_repair_cursor, 3);
        assert_eq!(solver.bag_parent_count[live2 as usize], 1);
    }

    #[test]
    fn ranked_policy_reorders_multiple_cached_placements_preserving_fallback_order() {
        let mut solver = ProofSolver::new_with_mode(
            SolverConfig {
                admissibility: BoardAdmissibility {
                    max_height: 4,
                    max_holes: 1,
                    max_cells: 12,
                    max_roughness: u32::MAX,
                    max_height_spread: u32::MAX,
                },
                root: crate::config::RootStateConfig::default(),
            },
            SolveMode::Optimistic,
            FrontierMode::StableFirst,
        );
        let placements = placements_for_piece(TetrisPiece::T_PIECE, 4);
        let mut children = placements
            .iter()
            .enumerate()
            .map(|(idx, &placement)| PieceChild {
                placement,
                succ: idx as BagNodeId,
            })
            .collect::<Vec<_>>();

        solver.promote_policy_placement(
            solver.config.root.board,
            TetrisPiece::T_PIECE,
            placements[2],
            false,
        );
        solver.promote_policy_placement(
            solver.config.root.board,
            TetrisPiece::T_PIECE,
            placements[0],
            false,
        );

        let _ = solver.reorder_piece_children_optimistically(
            solver.config.root.board,
            TetrisPiece::T_PIECE,
            &mut children,
        );

        assert_eq!(
            children
                .iter()
                .map(|child| child.placement)
                .collect::<Vec<_>>(),
            vec![placements[0], placements[2], placements[1], placements[3]]
        );
    }

    #[test]
    fn optimistic_repair_promotion_reorders_future_identical_board_piece_expansion() {
        let mut solver = ProofSolver::new_with_mode(
            SolverConfig {
                admissibility: BoardAdmissibility {
                    max_height: 4,
                    max_holes: 1,
                    max_cells: 12,
                    max_roughness: u32::MAX,
                    max_height_spread: u32::MAX,
                },
                root: crate::config::RootStateConfig::default(),
            },
            SolveMode::Optimistic,
            FrontierMode::StableFirst,
        );
        let placements = placements_for_piece(TetrisPiece::L_PIECE, 3);
        let mut children = placements
            .iter()
            .enumerate()
            .map(|(idx, &placement)| PieceChild {
                placement,
                succ: idx as BagNodeId,
            })
            .collect::<Vec<_>>();

        solver.promote_policy_placement(
            solver.config.root.board,
            TetrisPiece::L_PIECE,
            placements[2],
            true,
        );
        let _ = solver.reorder_piece_children_optimistically(
            solver.config.root.board,
            TetrisPiece::L_PIECE,
            &mut children,
        );

        assert_eq!(children[0].placement, placements[2]);
        assert_eq!(solver.optimistic_repairs_promoted, 1);
    }

    #[test]
    fn optimistic_policy_demotes_failed_active_placement() {
        let mut solver = ProofSolver::new_with_mode(
            SolverConfig {
                admissibility: BoardAdmissibility {
                    max_height: 4,
                    max_holes: 1,
                    max_cells: 12,
                    max_roughness: u32::MAX,
                    max_height_spread: u32::MAX,
                },
                root: crate::config::RootStateConfig::default(),
            },
            SolveMode::Optimistic,
            FrontierMode::StableFirst,
        );
        let placements = placements_for_piece(TetrisPiece::T_PIECE, 3);
        let board = solver.config.root.board;

        solver.promote_policy_placement(board, TetrisPiece::T_PIECE, placements[2], false);
        solver.promote_policy_placement(board, TetrisPiece::T_PIECE, placements[1], false);
        solver.promote_policy_placement(board, TetrisPiece::T_PIECE, placements[0], false);
        solver.demote_policy_placement(board, TetrisPiece::T_PIECE, placements[0], true);

        let entry = solver
            .optimistic_policy
            .get(&GeometryKey {
                board,
                piece: TetrisPiece::T_PIECE,
            })
            .expect("policy entry should exist");
        assert_eq!(
            entry.placements,
            vec![placements[1], placements[0], placements[2]]
        );
        assert_eq!(entry.failed_repairs, 1);
        assert_eq!(solver.optimistic_repairs_demoted, 1);
    }

    #[test]
    fn optimistic_popularity_reorders_children_toward_common_successor() {
        let mut solver = ProofSolver::new_with_mode(
            SolverConfig {
                admissibility: BoardAdmissibility {
                    max_height: 4,
                    max_holes: 1,
                    max_cells: 12,
                    max_roughness: u32::MAX,
                    max_height_spread: u32::MAX,
                },
                root: crate::config::RootStateConfig::default(),
            },
            SolveMode::Optimistic,
            FrontierMode::StableFirst,
        );
        let succ0 = solver.intern_bag(BagKey {
            board: TetrisBoard::new(),
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::O_PIECE)),
        });
        let succ1 = solver.intern_bag(BagKey {
            board: {
                let mut board = TetrisBoard::new();
                board.set_bit(0, 0);
                board
            },
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::I_PIECE)),
        });
        let succ2 = solver.intern_bag(BagKey {
            board: {
                let mut board = TetrisBoard::new();
                board.set_bit(1, 0);
                board
            },
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::T_PIECE)),
        });
        solver.optimistic_successor_popularity[succ2 as usize] = 5;
        solver.optimistic_successor_popularity[succ1 as usize] = 2;

        let placements = placements_for_piece(TetrisPiece::S_PIECE, 3);
        let mut children = vec![
            PieceChild {
                placement: placements[0],
                succ: succ0,
            },
            PieceChild {
                placement: placements[1],
                succ: succ1,
            },
            PieceChild {
                placement: placements[2],
                succ: succ2,
            },
        ];

        solver.reorder_piece_children_by_successor_popularity(&mut children, 0);

        assert_eq!(children[0].succ, succ2);
        assert_eq!(children[1].succ, succ1);
        assert_eq!(children[2].succ, succ0);
        assert_eq!(solver.optimistic_reorders_using_popularity, 1);
    }

    #[test]
    fn optimistic_popularity_does_not_override_ranked_policy_prefix() {
        let mut solver = ProofSolver::new_with_mode(
            SolverConfig {
                admissibility: BoardAdmissibility {
                    max_height: 4,
                    max_holes: 1,
                    max_cells: 12,
                    max_roughness: u32::MAX,
                    max_height_spread: u32::MAX,
                },
                root: crate::config::RootStateConfig::default(),
            },
            SolveMode::Optimistic,
            FrontierMode::StableFirst,
        );
        let low_pop = solver.intern_bag(BagKey {
            board: TetrisBoard::new(),
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::O_PIECE)),
        });
        let high_pop = solver.intern_bag(BagKey {
            board: {
                let mut board = TetrisBoard::new();
                board.set_bit(0, 0);
                board
            },
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::I_PIECE)),
        });
        let tail = solver.intern_bag(BagKey {
            board: {
                let mut board = TetrisBoard::new();
                board.set_bit(1, 0);
                board
            },
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::T_PIECE)),
        });
        solver.optimistic_successor_popularity[high_pop as usize] = 10;
        let placements = placements_for_piece(TetrisPiece::Z_PIECE, 3);
        let mut children = vec![
            PieceChild {
                placement: placements[0],
                succ: low_pop,
            },
            PieceChild {
                placement: placements[1],
                succ: high_pop,
            },
            PieceChild {
                placement: placements[2],
                succ: tail,
            },
        ];

        solver.promote_policy_placement(
            solver.config.root.board,
            TetrisPiece::Z_PIECE,
            placements[0],
            false,
        );
        let locked_prefix = solver.reorder_piece_children_optimistically(
            solver.config.root.board,
            TetrisPiece::Z_PIECE,
            &mut children,
        );
        solver.reorder_piece_children_by_successor_popularity(&mut children, locked_prefix);

        assert_eq!(locked_prefix, 1);
        assert_eq!(children[0].placement, placements[0]);
        assert_eq!(children[1].succ, high_pop);
    }

    #[test]
    fn optimistic_strict_reuse_prefers_activated_successor_over_higher_ranked_dormant_one() {
        let mut solver = ProofSolver::new_with_mode(
            SolverConfig {
                admissibility: BoardAdmissibility {
                    max_height: 4,
                    max_holes: 1,
                    max_cells: 12,
                    max_roughness: u32::MAX,
                    max_height_spread: u32::MAX,
                },
                root: crate::config::RootStateConfig::default(),
            },
            SolveMode::Optimistic,
            FrontierMode::StableFirst,
        );
        let activated = solver.intern_bag(BagKey {
            board: TetrisBoard::new(),
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::O_PIECE)),
        });
        let dormant = solver.intern_bag_dormant(BagKey {
            board: {
                let mut board = TetrisBoard::new();
                board.set_bit(0, 0);
                board
            },
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::I_PIECE)),
        });
        let placements = placements_for_piece(TetrisPiece::T_PIECE, 2);
        let mut children = vec![
            PieceChild {
                placement: placements[0],
                succ: activated,
            },
            PieceChild {
                placement: placements[1],
                succ: dormant,
            },
        ];

        solver.promote_policy_placement(
            solver.config.root.board,
            TetrisPiece::T_PIECE,
            placements[1],
            false,
        );
        let locked_prefix = solver.reorder_piece_children_optimistically(
            solver.config.root.board,
            TetrisPiece::T_PIECE,
            &mut children,
        );
        solver.reorder_piece_children_by_successor_popularity(&mut children, locked_prefix);

        assert_eq!(
            children[0].succ, dormant,
            "ranked policy should still order children"
        );
        let selected = solver
            .choose_optimistic_child_slot_strict_reuse(&children, 1)
            .expect("a live successor should be selected");
        assert_eq!(selected, (1, true));
    }

    #[test]
    fn optimistic_successor_popularity_tracks_active_link_repair_lifecycle() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 4,
                max_holes: 1,
                max_cells: 12,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        };
        let mut solver =
            ProofSolver::new_with_mode(config, SolveMode::Optimistic, FrontierMode::StableFirst);

        let doomed = solver.intern_bag(BagKey {
            board: TetrisBoard::new(),
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::O_PIECE)),
        });
        let replacement = solver.intern_bag(BagKey {
            board: {
                let mut board = TetrisBoard::new();
                board.set_bit(2, 0);
                board
            },
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::I_PIECE)),
        });
        let piece_id = solver.intern_piece(
            PieceKey {
                board: solver.config.root.board,
                bag: solver.config.root.bag,
                piece: TetrisPiece::L_PIECE,
            },
            solver.root_bag,
        );
        let placements = placements_for_piece(TetrisPiece::L_PIECE, 2);

        solver.piece_nodes[piece_id as usize].children = vec![
            PieceChild {
                placement: placements[0],
                succ: doomed,
            },
            PieceChild {
                placement: placements[1],
                succ: replacement,
            },
        ];
        solver.piece_nodes[piece_id as usize].unique_succ_count = 2;
        solver.piece_nodes[piece_id as usize].alive_succ_count = 2;
        solver.piece_nodes[piece_id as usize].active_child_slot = Some(0);
        solver.piece_nodes[piece_id as usize].next_repair_cursor = 1;
        solver.optimistic_active_parent_links = 1;
        solver.increment_optimistic_successor_popularity(doomed);
        solver.append_parent_links(doomed, &[piece_id]);

        solver.kill_bag(doomed);
        solver.drain_death_queues();
        assert_eq!(solver.optimistic_successor_popularity[doomed as usize], 0);
        assert_eq!(solver.optimistic_repair_queue.len(), 1);

        let completed = solver.process_optimistic_repairs(SolveOptions {
            mode: SolveMode::Optimistic,
            ..SolveOptions::default()
        });
        assert!(completed);
        assert_eq!(
            solver.optimistic_successor_popularity[replacement as usize],
            1
        );
        assert_eq!(
            solver.piece_nodes[piece_id as usize].active_child_slot,
            Some(1)
        );
    }

    #[test]
    fn optimistic_repair_prefers_activated_replacement_before_dormant_replacement() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 4,
                max_holes: 1,
                max_cells: 12,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        };
        let mut solver =
            ProofSolver::new_with_mode(config, SolveMode::Optimistic, FrontierMode::StableFirst);

        let doomed = solver.intern_bag(BagKey {
            board: TetrisBoard::new(),
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::O_PIECE)),
        });
        let activated_replacement = solver.intern_bag(BagKey {
            board: {
                let mut board = TetrisBoard::new();
                board.set_bit(1, 0);
                board
            },
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::I_PIECE)),
        });
        let dormant_replacement = solver.intern_bag_dormant(BagKey {
            board: {
                let mut board = TetrisBoard::new();
                board.set_bit(2, 0);
                board
            },
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::T_PIECE)),
        });
        let piece_id = solver.intern_piece(
            PieceKey {
                board: solver.config.root.board,
                bag: solver.config.root.bag,
                piece: TetrisPiece::L_PIECE,
            },
            solver.root_bag,
        );
        let placements = placements_for_piece(TetrisPiece::L_PIECE, 3);

        solver.promote_policy_placement(
            solver.config.root.board,
            TetrisPiece::L_PIECE,
            placements[2],
            false,
        );
        solver.piece_nodes[piece_id as usize].children = vec![
            PieceChild {
                placement: placements[0],
                succ: doomed,
            },
            PieceChild {
                placement: placements[1],
                succ: activated_replacement,
            },
            PieceChild {
                placement: placements[2],
                succ: dormant_replacement,
            },
        ];
        solver.piece_nodes[piece_id as usize].unique_succ_count = 3;
        solver.piece_nodes[piece_id as usize].alive_succ_count = 3;
        solver.piece_nodes[piece_id as usize].active_child_slot = Some(0);
        solver.piece_nodes[piece_id as usize].next_repair_cursor = 1;
        solver.optimistic_active_parent_links = 1;
        solver.increment_optimistic_successor_popularity(doomed);
        solver.append_parent_links(doomed, &[piece_id]);

        solver.kill_bag(doomed);
        solver.drain_death_queues();
        let completed = solver.process_optimistic_repairs(SolveOptions {
            mode: SolveMode::Optimistic,
            ..SolveOptions::default()
        });
        assert!(completed);
        assert_eq!(
            solver.piece_nodes[piece_id as usize].active_child_slot,
            Some(1),
            "repair should reuse the already activated successor before activating a dormant one"
        );
        assert_eq!(solver.optimistic_repairs_reusing_known_successor, 1);
        assert_eq!(solver.optimistic_repairs_forcing_novel_activation, 0);
    }

    #[test]
    fn optimistic_repair_revalidates_stale_novel_proposal_as_known_reuse() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 4,
                max_holes: 1,
                max_cells: 12,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: crate::config::RootStateConfig::default(),
        };
        let mut solver =
            ProofSolver::new_with_mode(config, SolveMode::Optimistic, FrontierMode::StableFirst);

        let doomed = solver.intern_bag(BagKey {
            board: TetrisBoard::new(),
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::O_PIECE)),
        });
        let novel_key = BagKey {
            board: {
                let mut board = TetrisBoard::new();
                board.set_bit(3, 0);
                board
            },
            bag: TetrisPieceBagState::from(u8::from(TetrisPiece::I_PIECE)),
        };
        let piece_id = solver.intern_piece(
            PieceKey {
                board: solver.config.root.board,
                bag: solver.config.root.bag,
                piece: TetrisPiece::L_PIECE,
            },
            solver.root_bag,
        );
        let placements = placements_for_piece(TetrisPiece::L_PIECE, 2);

        solver.piece_nodes[piece_id as usize].optimistic_children = vec![
            OptimisticChild {
                placement: placements[0],
                target: OptimisticChildTarget::Activated(doomed),
            },
            OptimisticChild {
                placement: placements[1],
                target: OptimisticChildTarget::Dormant(novel_key),
            },
        ];
        solver.piece_nodes[piece_id as usize].active_child_slot = Some(0);
        solver.piece_nodes[piece_id as usize].next_repair_cursor = 1;
        solver.optimistic_active_parent_links = 1;
        solver.increment_optimistic_successor_popularity(doomed);
        solver.append_parent_links(doomed, &[piece_id]);

        solver.kill_bag(doomed);
        solver.drain_death_queues();
        let snapshot = solver.optimistic_snapshot();
        let proposal = solver
            .compute_optimistic_repair_proposals(vec![piece_id], &snapshot)
            .pop()
            .expect("repair proposal should exist");
        assert_eq!(
            proposal.activation_kind,
            Some(OptimisticActivationKind::NovelActivation)
        );

        let known_succ = solver.intern_bag(novel_key);
        solver.activate_bag_if_needed(known_succ);
        solver.apply_optimistic_repair_proposal(proposal);

        assert_eq!(solver.optimistic_repair_proposals_revalidated, 1);
        assert_eq!(solver.optimistic_repairs_reusing_known_successor, 1);
        assert_eq!(solver.optimistic_repairs_forcing_novel_activation, 0);
        assert_eq!(
            solver.piece_nodes[piece_id as usize].active_child_slot,
            Some(1)
        );
    }

    #[test]
    fn board_shape_metrics_match_expected_values() {
        let heights = [0, 2, 1, 3, 3, 1, 0, 2, 2, 1];
        assert_eq!(ProofSolver::count_turning_points(heights), 5);
        assert_eq!(ProofSolver::max_well_depth(heights), 2);
    }

    #[test]
    fn template_kernel_mode_reports_unresolved_when_no_closed_family_is_found() {
        let result = solve_root_with_options(
            SolverConfig {
                admissibility: BoardAdmissibility {
                    max_height: 1,
                    max_holes: 0,
                    max_cells: 8,
                    max_roughness: u32::MAX,
                    max_height_spread: u32::MAX,
                },
                root: crate::config::RootStateConfig::default(),
            },
            SolveOptions {
                mode: SolveMode::TemplateKernel,
                template_sample_expansions: 256,
                template_max_turning_points: 1,
                template_max_well_depth: 0,
                template_max_bridge_depth: 4,
                ..SolveOptions::default()
            },
        )
        .expect("template-kernel solver should succeed");

        assert_eq!(result.solve_mode, SolveMode::TemplateKernel);
        assert_eq!(result.root_result.conclusion, SolveConclusion::Unresolved);
        assert!(!result.metrics.root_reaches_template_family);
    }
}
