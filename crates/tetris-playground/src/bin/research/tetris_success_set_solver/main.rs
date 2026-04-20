mod config;
mod graph;
mod policy;
mod proof;
mod retrograde;
mod state;
mod universe;
mod verifier;

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, ensure};
use clap::{Parser, ValueEnum};
use serde::Serialize;

use crate::config::{BoardAdmissibility, SolverConfig};
use crate::proof::{
    ProofSolveResult, SolveConclusion, SolveMode, SolveOptions, solve_root,
    solve_root_with_options, verify_proof,
};
use crate::state::unpack_placement;
use crate::verifier::{ReplayVerification, verify_replay};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PrintMode {
    Summary,
    Roadmap,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliSolveMode {
    Full,
    CoreFirst,
    TemplateKernel,
    Optimistic,
}

#[derive(Debug, Parser)]
#[command(name = "tetris_success_set_solver")]
#[command(about = "Closed winning-set solver for no-preview, no-hold 7-bag Tetris")]
struct Cli {
    #[arg(long, default_value_t = 6)]
    max_height: u8,
    #[arg(long, default_value_t = u32::MAX)]
    max_holes: u32,
    #[arg(long, default_value_t = u32::MAX)]
    max_cells: u32,
    #[arg(long, default_value_t = u32::MAX)]
    max_roughness: u32,
    #[arg(long, default_value_t = u32::MAX)]
    max_height_spread: u32,
    #[arg(long, default_value_t = tetris_game::TetrisPieceBagState::FULL_MASK, value_parser = parse_nonzero_bag_mask)]
    root_bag_mask: u8,
    /// Root board encoded as 20 slash-separated rows of 10 chars each, top-to-bottom, using '.' and '#'.
    #[arg(long)]
    root_board_rows: Option<String>,
    /// Optional path for the summary JSON export. The proof graph artifact will be emitted beside it.
    #[arg(long)]
    export_path: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = CliSolveMode::CoreFirst)]
    solve_mode: CliSolveMode,
    #[arg(long, default_value_t = 200_000)]
    core_prepass_expansions: usize,
    #[arg(long, default_value_t = 4_096)]
    core_certify_interval: usize,
    #[arg(long, default_value_t = 4)]
    template_max_turning_points: u32,
    #[arg(long, default_value_t = 3)]
    template_max_well_depth: u32,
    #[arg(long, default_value_t = 500_000)]
    template_sample_expansions: usize,
    #[arg(long, default_value_t = 50)]
    template_max_bridge_depth: usize,
    #[arg(long, default_value_t = 32)]
    optimistic_max_repairs_per_piece: usize,
    #[arg(long, default_value_t = 1_000_000)]
    optimistic_max_global_repairs: usize,
    #[arg(long, default_value_t = 1)]
    optimistic_probe_width: usize,
    #[arg(long, default_value_t = true)]
    optimistic_fallback_on_thrash: bool,
    #[arg(long, value_enum, default_value_t = PrintMode::Summary)]
    print: PrintMode,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root_board = parse_root_board_rows(cli.root_board_rows.as_deref())?;
    let config = SolverConfig {
        admissibility: BoardAdmissibility {
            max_height: cli.max_height,
            max_holes: cli.max_holes,
            max_cells: cli.max_cells,
            max_roughness: cli.max_roughness,
            max_height_spread: cli.max_height_spread,
        },
        root: crate::config::RootStateConfig {
            board: root_board,
            bag: tetris_game::TetrisPieceBagState::from(cli.root_bag_mask),
        },
    };

    match cli.print {
        PrintMode::Roadmap => print_roadmap(),
        PrintMode::Summary => run_solver(
            config,
            SolveOptions {
                mode: match cli.solve_mode {
                    CliSolveMode::Full => SolveMode::Full,
                    CliSolveMode::CoreFirst => SolveMode::CoreFirst,
                    CliSolveMode::TemplateKernel => SolveMode::TemplateKernel,
                    CliSolveMode::Optimistic => SolveMode::Optimistic,
                },
                core_prepass_expansions: cli.core_prepass_expansions,
                core_certify_interval: cli.core_certify_interval,
                template_max_turning_points: cli.template_max_turning_points,
                template_max_well_depth: cli.template_max_well_depth,
                template_sample_expansions: cli.template_sample_expansions,
                template_max_bridge_depth: cli.template_max_bridge_depth,
                optimistic_max_repairs_per_piece: cli.optimistic_max_repairs_per_piece,
                optimistic_max_global_repairs: cli.optimistic_max_global_repairs,
                optimistic_probe_width: cli.optimistic_probe_width,
                optimistic_fallback_on_thrash: cli.optimistic_fallback_on_thrash,
            },
            cli.export_path,
        )?,
    }

    Ok(())
}

fn parse_nonzero_bag_mask(value: &str) -> Result<u8, String> {
    let mask = value
        .parse::<u8>()
        .map_err(|err| format!("invalid root bag mask: {err}"))?;
    if mask == 0 || mask > tetris_game::TetrisPieceBagState::FULL_MASK {
        return Err(format!(
            "root bag mask must be in 1..={}",
            tetris_game::TetrisPieceBagState::FULL_MASK
        ));
    }
    Ok(mask)
}

fn parse_root_board_rows(value: Option<&str>) -> Result<tetris_game::TetrisBoard> {
    let Some(value) = value else {
        return Ok(tetris_game::TetrisBoard::new());
    };

    let rows = value.split('/').collect::<Vec<_>>();
    ensure!(
        rows.len() == tetris_game::TetrisBoard::HEIGHT,
        "root board must contain exactly {} slash-separated rows",
        tetris_game::TetrisBoard::HEIGHT
    );

    let mut board = tetris_game::TetrisBoard::new();
    for (row_idx_from_top, row) in rows.iter().enumerate() {
        ensure!(
            row.len() == tetris_game::TetrisBoard::WIDTH,
            "each root-board row must contain exactly {} cells",
            tetris_game::TetrisBoard::WIDTH
        );
        let board_row = tetris_game::TetrisBoard::HEIGHT - 1 - row_idx_from_top;
        for (col, ch) in row.chars().enumerate() {
            match ch {
                '#' => board.set_bit(col, board_row),
                '.' => {}
                _ => anyhow::bail!("root board rows may only contain '.' and '#'"),
            }
        }
    }

    Ok(board)
}

fn print_roadmap() {
    println!("Implementation roadmap");
    println!("1. Discover the full admissible graph reachable from the configured root.");
    println!("2. Treat all discovered states as provisional winners.");
    println!("3. Propagate death backward until a closed winning set remains.");
    println!("4. Pick one surviving witness placement for each surviving piece branch.");
    println!("5. Export a winning-set summary JSON and proof-graph artifact.");
    println!("6. Cross-check tiny instances against a naive greatest-fixed-point oracle.");
}

fn run_solver(
    config: SolverConfig,
    options: SolveOptions,
    export_path: Option<PathBuf>,
) -> Result<()> {
    ensure!(
        crate::universe::board_is_admissible(config.root.board, config.admissibility),
        "root board is outside the admissible universe"
    );

    let result = solve_root_with_options(config, options)?;
    let (verification, replay_verification) = match result.root_result.conclusion {
        SolveConclusion::Unresolved => (
            crate::proof::ProofVerification::default(),
            ReplayVerification::default(),
        ),
        SolveConclusion::Yes | SolveConclusion::No => {
            let verification = verify_proof(&result);
            ensure!(
                verification.is_clean(),
                "proof verification failed: bag_nodes_checked={} piece_nodes_checked={} witness_failures={} resolution_failures={}",
                verification.bag_nodes_checked,
                verification.piece_nodes_checked,
                verification.witness_failures,
                verification.resolution_failures,
            );
            let replay_verification = verify_replay(config, &result);
            ensure!(
                replay_verification.is_clean(),
                "proof replay verification failed: won_bags_checked={} won_pieces_checked={} lost_pieces_checked={} replay_failures={} missing_node_failures={} bag_transition_failures={} resolution_failures={}",
                replay_verification.won_bags_checked,
                replay_verification.won_pieces_checked,
                replay_verification.lost_pieces_checked,
                replay_verification.replay_failures,
                replay_verification.missing_node_failures,
                replay_verification.bag_transition_failures,
                replay_verification.resolution_failures,
            );
            (verification, replay_verification)
        }
    };

    println!("tetris_success_set_solver");
    println!("max_height={}", config.admissibility.max_height);
    println!("max_holes={}", config.admissibility.max_holes);
    println!("max_cells={}", config.admissibility.max_cells);
    println!("max_roughness={}", config.admissibility.max_roughness);
    println!(
        "max_height_spread={}",
        config.admissibility.max_height_spread
    );
    println!("root_board_rows={}", board_to_rows(config.root.board));
    println!("root_bag_mask={}", u8::from(config.root.bag));
    println!("solve_mode={:?}", result.solve_mode);
    println!("used_full_fallback={}", result.used_full_fallback);
    println!("bag_node_count={}", result.metrics.bag_node_count);
    println!("piece_node_count={}", result.metrics.piece_node_count);
    println!("deduped_edge_count={}", result.metrics.deduped_edge_count);
    println!("winning_count={}", result.metrics.winning_count);
    println!("losing_count={}", result.metrics.losing_count);
    println!("piece_winning_count={}", result.metrics.piece_winning_count);
    println!("piece_losing_count={}", result.metrics.piece_losing_count);
    println!("bags_expanded={}", result.metrics.bags_expanded);
    println!("skipped_dead_bags={}", result.metrics.skipped_dead_bags);
    println!(
        "parallel_batches_processed={}",
        result.metrics.parallel_batches_processed
    );
    println!("parallel_batch_size={}", result.metrics.parallel_batch_size);
    println!("branches_expanded={}", result.metrics.branches_expanded);
    println!(
        "branches_skipped_due_to_parent_death={}",
        result.metrics.branches_skipped_due_to_parent_death
    );
    println!(
        "bags_killed_by_first_failed_branch={}",
        result.metrics.bags_killed_by_first_failed_branch
    );
    println!(
        "avg_branches_expanded_per_bag={:.3}",
        result.metrics.avg_branches_expanded_per_bag
    );
    println!(
        "staged_branches_processed={}",
        result.metrics.staged_branches_processed
    );
    println!(
        "batch_unique_piece_keys={}",
        result.metrics.batch_unique_piece_keys
    );
    println!(
        "batch_unique_bag_keys={}",
        result.metrics.batch_unique_bag_keys
    );
    println!(
        "resolved_children_processed={}",
        result.metrics.resolved_children_processed
    );
    println!(
        "batched_parent_links_appended={}",
        result.metrics.batched_parent_links_appended
    );
    println!(
        "parent_segments_allocated={}",
        result.metrics.parent_segments_allocated
    );
    println!("parent_links_stored={}", result.metrics.parent_links_stored);
    println!(
        "avg_parent_segments_per_bag={:.3}",
        result.metrics.avg_parent_segments_per_bag
    );
    println!(
        "optimistic_policy_entries={}",
        result.metrics.optimistic_policy_entries
    );
    println!(
        "optimistic_ranked_policy_entries={}",
        result.metrics.optimistic_ranked_policy_entries
    );
    println!(
        "optimistic_ranked_policy_capacity={}",
        result.metrics.optimistic_ranked_policy_capacity
    );
    println!(
        "optimistic_policy_hits={}",
        result.metrics.optimistic_policy_hits
    );
    println!(
        "optimistic_policy_misses={}",
        result.metrics.optimistic_policy_misses
    );
    println!(
        "optimistic_repairs_attempted={}",
        result.metrics.optimistic_repairs_attempted
    );
    println!(
        "optimistic_repairs_succeeded={}",
        result.metrics.optimistic_repairs_succeeded
    );
    println!(
        "optimistic_repairs_failed={}",
        result.metrics.optimistic_repairs_failed
    );
    println!(
        "optimistic_repairs_promoted={}",
        result.metrics.optimistic_repairs_promoted
    );
    println!(
        "optimistic_repairs_demoted={}",
        result.metrics.optimistic_repairs_demoted
    );
    println!(
        "optimistic_reused_ranked_candidates={}",
        result.metrics.optimistic_reused_ranked_candidates
    );
    println!(
        "optimistic_active_parent_links={}",
        result.metrics.optimistic_active_parent_links
    );
    println!(
        "optimistic_popularity_nonzero_bags={}",
        result.metrics.optimistic_popularity_nonzero_bags
    );
    println!(
        "optimistic_max_successor_popularity={}",
        result.metrics.optimistic_max_successor_popularity
    );
    println!(
        "optimistic_avg_successor_popularity={:.3}",
        result.metrics.optimistic_avg_successor_popularity
    );
    println!(
        "optimistic_reorders_using_popularity={}",
        result.metrics.optimistic_reorders_using_popularity
    );
    println!(
        "optimistic_avg_candidates_scanned_per_repair={:.3}",
        result.metrics.optimistic_avg_candidates_scanned_per_repair
    );
    println!(
        "optimistic_avg_rank_position_chosen={:.3}",
        result.metrics.optimistic_avg_rank_position_chosen
    );
    println!(
        "optimistic_max_repairs_on_single_piece={}",
        result.metrics.optimistic_max_repairs_on_single_piece
    );
    println!(
        "optimistic_activated_live_bags={}",
        result.metrics.optimistic_activated_live_bags
    );
    println!(
        "optimistic_dormant_child_targets={}",
        result.metrics.optimistic_dormant_child_targets
    );
    println!(
        "optimistic_activated_frontier={}",
        result.metrics.optimistic_activated_frontier
    );
    println!(
        "optimistic_novel_activations={}",
        result.metrics.optimistic_novel_activations
    );
    println!(
        "optimistic_reuse_activations={}",
        result.metrics.optimistic_reuse_activations
    );
    println!(
        "optimistic_repairs_reusing_known_successor={}",
        result.metrics.optimistic_repairs_reusing_known_successor
    );
    println!(
        "optimistic_repairs_forcing_novel_activation={}",
        result.metrics.optimistic_repairs_forcing_novel_activation
    );
    println!(
        "optimistic_fallback_triggered={}",
        result.metrics.optimistic_fallback_triggered
    );
    println!(
        "optimistic_expand_proposal_secs={:.3}",
        result.metrics.optimistic_expand_proposal_secs
    );
    println!(
        "optimistic_repair_proposal_secs={:.3}",
        result.metrics.optimistic_repair_proposal_secs
    );
    println!(
        "optimistic_repair_batches_processed={}",
        result.metrics.optimistic_repair_batches_processed
    );
    println!(
        "optimistic_expand_proposals_revalidated={}",
        result.metrics.optimistic_expand_proposals_revalidated
    );
    println!(
        "optimistic_repair_proposals_revalidated={}",
        result.metrics.optimistic_repair_proposals_revalidated
    );
    println!(
        "candidate_core_count={}",
        result.metrics.candidate_core_count
    );
    println!("largest_scc_bags={}", result.metrics.largest_scc_bags);
    println!("largest_scc_pieces={}", result.metrics.largest_scc_pieces);
    println!("closed_core_found={}", result.metrics.closed_core_found);
    println!(
        "core_certification_secs={:.3}",
        result.metrics.core_certification_secs
    );
    println!(
        "bridge_search_secs={:.3}",
        result.metrics.bridge_search_secs
    );
    println!("stage_secs={:.3}", result.metrics.stage_secs);
    println!("commit_secs={:.3}", result.metrics.commit_secs);
    println!(
        "template_member_bags={}",
        result.metrics.template_member_bags
    );
    println!(
        "template_member_pieces={}",
        result.metrics.template_member_pieces
    );
    println!(
        "template_signature_count={}",
        result.metrics.template_signature_count
    );
    println!(
        "largest_template_scc_bags={}",
        result.metrics.largest_template_scc_bags
    );
    println!(
        "largest_template_scc_pieces={}",
        result.metrics.largest_template_scc_pieces
    );
    println!(
        "root_reaches_template_family={}",
        result.metrics.root_reaches_template_family
    );
    println!(
        "template_attractor_bags={}",
        result.metrics.template_attractor_bags
    );
    println!(
        "template_closure_failed_piece_counts={:?}",
        result.metrics.template_closure_failed_piece_counts
    );
    println!("discover_secs={:.3}", result.discover_secs);
    println!("propagate_secs={:.3}", result.propagate_secs);
    println!("total_secs={:.3}", result.total_secs);
    println!(
        "bag_nodes_per_sec={:.1}",
        result.metrics.bag_node_count as f64 / result.total_secs.max(1e-9)
    );
    println!("dependency_count={}", result.metrics.dependency_count);
    println!("cache_hits={}", result.metrics.cache_hits);
    println!("cache_misses={}", result.metrics.cache_misses);
    println!("geometry_cache_hits={}", result.metrics.geometry_cache_hits);
    println!(
        "geometry_cache_misses={}",
        result.metrics.geometry_cache_misses
    );
    println!(
        "board_feature_cache_hits={}",
        result.metrics.board_feature_cache_hits
    );
    println!(
        "board_feature_cache_misses={}",
        result.metrics.board_feature_cache_misses
    );
    println!(
        "graph_verification=bag_nodes_checked:{} piece_nodes_checked:{} witness_failures:{} resolution_failures:{}",
        verification.bag_nodes_checked,
        verification.piece_nodes_checked,
        verification.witness_failures,
        verification.resolution_failures,
    );
    println!(
        "replay_verification=won_bags_checked:{} won_pieces_checked:{} lost_pieces_checked:{} witness_replays:{} lost_piece_placements_checked:{} replay_failures:{} missing_node_failures:{} bag_transition_failures:{} resolution_failures:{}",
        replay_verification.won_bags_checked,
        replay_verification.won_pieces_checked,
        replay_verification.lost_pieces_checked,
        replay_verification.witness_replays,
        replay_verification.lost_piece_placements_checked,
        replay_verification.replay_failures,
        replay_verification.missing_node_failures,
        replay_verification.bag_transition_failures,
        replay_verification.resolution_failures,
    );

    match result.root_result.conclusion {
        SolveConclusion::Yes => {
            println!("result=YES");
            println!("root_witnesses:");
            for piece in config.root.bag.iter_pieces() {
                let placement = result.root_result.best_action[piece.index() as usize]
                    .map(unpack_placement)
                    .expect("winning root must have a witness for each piece");
                println!("  piece={} placement={}", piece, placement.index());
            }
        }
        SolveConclusion::No => {
            println!("result=NO");
            println!("root_failing_pieces:");
            for piece in &result.root_result.failing_pieces {
                println!("  piece={}", piece);
            }
        }
        SolveConclusion::Unresolved => {
            println!("result=UNRESOLVED");
        }
    }

    let export_path = export_result(
        export_path,
        config,
        &result,
        &verification,
        &replay_verification,
    )?;
    println!("export_path={}", export_path.display());
    let _ = std::io::stdout().flush();

    Ok(())
}

fn board_to_rows(board: tetris_game::TetrisBoard) -> String {
    (0..tetris_game::TetrisBoard::HEIGHT)
        .rev()
        .map(|row| {
            (0..tetris_game::TetrisBoard::WIDTH)
                .map(|col| if board.get_bit(col, row) { '#' } else { '.' })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("/")
}

#[derive(Debug, Serialize)]
struct ExportConfig {
    max_height: u8,
    max_holes: u32,
    max_cells: u32,
    max_roughness: u32,
    max_height_spread: u32,
    root_board_rows: String,
    root_bag_mask: u8,
}

#[derive(Debug, Serialize)]
struct ExportGraphVerification {
    bag_nodes_checked: usize,
    piece_nodes_checked: usize,
    witness_failures: usize,
    resolution_failures: usize,
}

#[derive(Debug, Serialize)]
struct ExportVerification {
    graph: ExportGraphVerification,
    replay: ReplayVerification,
}

#[derive(Debug, Serialize)]
struct ExportRootResult {
    result: &'static str,
    root_witnesses: Vec<ExportWitness>,
    root_failing_pieces: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ExportWitness {
    piece: String,
    placement_index: u8,
}

#[derive(Debug, Serialize)]
struct ExportPayload {
    solver: &'static str,
    unix_timestamp_secs: u64,
    config: ExportConfig,
    metrics: crate::proof::ProofMetrics,
    verification: ExportVerification,
    root: ExportRootResult,
}

fn export_result(
    export_path: Option<PathBuf>,
    config: SolverConfig,
    result: &ProofSolveResult,
    verification: &crate::proof::ProofVerification,
    replay_verification: &ReplayVerification,
) -> Result<PathBuf> {
    let unix_timestamp_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("current time should be after unix epoch")
        .as_secs();

    let path = export_path.unwrap_or_else(|| {
        PathBuf::from(format!(
            "artifacts/output/tetris_success_set_solver_{unix_timestamp_secs}_summary.json"
        ))
    });
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let root = match result.root_result.conclusion {
        SolveConclusion::Yes => ExportRootResult {
            result: "YES",
            root_witnesses: config
                .root
                .bag
                .iter_pieces()
                .map(|piece| ExportWitness {
                    piece: piece.to_string(),
                    placement_index: result.root_result.best_action[piece.index() as usize]
                        .expect("winning root must have witness for every piece"),
                })
                .collect(),
            root_failing_pieces: Vec::new(),
        },
        SolveConclusion::No => ExportRootResult {
            result: "NO",
            root_witnesses: Vec::new(),
            root_failing_pieces: result
                .root_result
                .failing_pieces
                .iter()
                .map(ToString::to_string)
                .collect(),
        },
        SolveConclusion::Unresolved => ExportRootResult {
            result: "UNRESOLVED",
            root_witnesses: Vec::new(),
            root_failing_pieces: Vec::new(),
        },
    };

    let payload = ExportPayload {
        solver: "tetris_success_set_solver",
        unix_timestamp_secs,
        config: ExportConfig {
            max_height: config.admissibility.max_height,
            max_holes: config.admissibility.max_holes,
            max_cells: config.admissibility.max_cells,
            max_roughness: config.admissibility.max_roughness,
            max_height_spread: config.admissibility.max_height_spread,
            root_board_rows: board_to_rows(config.root.board),
            root_bag_mask: u8::from(config.root.bag),
        },
        metrics: result.metrics.clone(),
        verification: ExportVerification {
            graph: ExportGraphVerification {
                bag_nodes_checked: verification.bag_nodes_checked,
                piece_nodes_checked: verification.piece_nodes_checked,
                witness_failures: verification.witness_failures,
                resolution_failures: verification.resolution_failures,
            },
            replay: *replay_verification,
        },
        root,
    };

    fs::write(&path, serde_json::to_vec_pretty(&payload)?)?;
    fs::write(
        companion_proof_graph_path(&path),
        render_proof_graph_jsonl(result),
    )?;
    Ok(path)
}

fn companion_proof_graph_path(summary_path: &PathBuf) -> PathBuf {
    let file_name = summary_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("proof_summary.json");
    let proof_name = if let Some(stem) = file_name.strip_suffix("_summary.json") {
        format!("{stem}_proof_graph.jsonl")
    } else if let Some(stem) = file_name.strip_suffix(".json") {
        format!("{stem}_proof_graph.jsonl")
    } else {
        format!("{file_name}_proof_graph.jsonl")
    };
    summary_path.with_file_name(proof_name)
}

fn render_proof_graph_jsonl(result: &ProofSolveResult) -> String {
    let mut lines = Vec::with_capacity(result.bag_nodes.len() + result.piece_nodes.len());
    for (id, bag) in result.bag_nodes.iter().enumerate() {
        let piece_nodes = bag
            .piece_nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| node.map(|node_id| format!("{idx}:{node_id}")))
            .collect::<Vec<_>>();
        let value = serde_json::json!({
            "kind": "bag",
            "id": id,
            "board_rows": board_to_rows(bag.key.board),
            "bag_mask": u8::from(bag.key.bag),
            "resolution": bag.resolution,
            "is_root": id as u32 == result.root_bag,
            "piece_nodes": piece_nodes,
        });
        lines.push(value.to_string());
    }
    for (id, piece) in result.piece_nodes.iter().enumerate() {
        let children = piece
            .children
            .iter()
            .map(|child| {
                serde_json::json!({
                    "placement_index": child.placement,
                    "succ_bag": child.succ,
                })
            })
            .collect::<Vec<_>>();
        let value = serde_json::json!({
            "kind": "piece",
            "id": id,
            "board_rows": board_to_rows(piece.key.board),
            "bag_mask": u8::from(piece.key.bag),
            "piece": piece.key.piece.to_string(),
            "parent_bag": piece.parent_bag,
            "resolution": piece.resolution,
            "best_action": piece.best_action,
            "best_child": piece.best_child,
            "children": children,
        });
        lines.push(value.to_string());
    }
    format!("{}\n", lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn parse_nonzero_bag_mask_rejects_zero() {
        assert!(parse_nonzero_bag_mask("0").is_err());
    }

    #[test]
    fn parse_root_board_rows_accepts_valid_board() {
        let board = parse_root_board_rows(Some(
            "........../........../........../........../........../........../........../........../........../........../........../........../........../........../........../........../........../........../........../##########",
        ))
        .expect("board should parse");
        assert_eq!(board.count(), 10);
    }

    #[test]
    fn export_writes_summary_and_proof_graph() {
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
        let result = solve_root(config).expect("solver should succeed");
        let verification = verify_proof(&result);
        let replay_verification = verify_replay(config, &result);

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_nanos();
        let export_path = std::env::temp_dir().join(format!(
            "tetris_success_set_solver_test_{unique}_summary.json"
        ));
        let proof_path = companion_proof_graph_path(&export_path);

        let written_path = export_result(
            Some(export_path.clone()),
            config,
            &result,
            &verification,
            &replay_verification,
        )
        .expect("export should succeed");

        assert_eq!(written_path, export_path);
        assert!(written_path.exists());
        assert!(proof_path.exists());

        let summary = fs::read_to_string(&written_path).expect("summary should be readable");
        assert!(summary.contains("\"result\": \"NO\""));
        assert!(summary.contains("\"winning_count\": 0"));
        assert!(summary.contains("\"replay\""));

        let proof_graph = fs::read_to_string(&proof_path).expect("proof graph should be readable");
        assert!(proof_graph.contains("\"kind\":\"bag\""));
        assert!(proof_graph.contains("\"kind\":\"piece\""));

        let _ = fs::remove_file(written_path);
        let _ = fs::remove_file(proof_path);
    }

    #[test]
    fn export_no_summary_contains_precise_failing_pieces() {
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
        let result = solve_root(config).expect("solver should succeed");
        let verification = verify_proof(&result);
        let replay_verification = verify_replay(config, &result);

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_nanos();
        let export_path = std::env::temp_dir().join(format!(
            "tetris_success_set_solver_no_test_{unique}_summary.json"
        ));
        let proof_path = companion_proof_graph_path(&export_path);

        let written_path = export_result(
            Some(export_path.clone()),
            config,
            &result,
            &verification,
            &replay_verification,
        )
        .expect("export should succeed");

        let summary = fs::read_to_string(&written_path).expect("summary should be readable");
        assert!(summary.contains("\"result\": \"NO\""));
        assert!(summary.contains("\"winning_count\": 0"));
        assert!(summary.contains(&format!(
            "\"losing_count\": {}",
            result.metrics.losing_count
        )));
        for piece in ["O", "I", "S", "Z", "T", "L", "J"] {
            assert!(summary.contains(&format!("\"{piece}\"")));
        }

        let proof_graph = fs::read_to_string(&proof_path).expect("proof graph should be readable");
        assert!(proof_graph.contains("\"resolution\":\"lost\""));

        let _ = fs::remove_file(written_path);
        let _ = fs::remove_file(proof_path);
    }

    #[test]
    fn companion_proof_graph_path_uses_distinct_name_for_plain_json_path() {
        let summary_path = PathBuf::from("artifacts/output/custom.json");
        let proof_path = companion_proof_graph_path(&summary_path);

        assert_ne!(summary_path, proof_path);
        assert_eq!(
            proof_path,
            PathBuf::from("artifacts/output/custom_proof_graph.jsonl")
        );
    }
}
