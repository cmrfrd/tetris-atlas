//! Distillation: project the PUCT teacher's policy improvement back into
//! circuit form.
//!
//! Pipeline (all TRAINING-time; the played artifact stays a pure state→move
//! boolean function):
//! 1. Sample states by playing the prior forest from a seed set.
//! 2. Label each state with the root-PUCT teacher (`puct.rs`) — the forest's
//!    own votes are the search prior, so the labels are a one-step-improved
//!    version of the forest's policy.
//! 3. Evolve forests whose fitness is AGREEMENT with the labels (fraction of
//!    states where the forest's vote equals the teacher's move).
//!
//! The distilled champion is a normal forest JSON — hand it to
//! `train --resume` to continue survival training from the improved policy
//! (the AlphaZero improve→project cycle, one round per distill run).

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use tetris_game::config::StandardTetris;
use tetris_game::tetris::{IsLost, TetrisGame, TetrisPiecePlacement};

use crate::compiled::CompiledForest;
use crate::config::EvolveConfig;
use crate::encode::{NUM_INPUTS, encode_game};
use crate::evolve::{crossover, mutate_forest, mutate_forest_owned, tournament_select};
use crate::forest::DecisionForest;
use crate::puct::{PuctConfig, best_action, puct_root};
use crate::train::{TrainArtifacts, cull_invalid_trees_owned, fitness, regrow_forest};
use crate::tree::COND_GEN_DEPTH_CAP;

/// One teacher-labeled state.
pub struct DistillExample {
    pub bits: [u8; NUM_INPUTS],
    /// The legal set of the state (static per piece — see `encode.rs`).
    pub legal: &'static [TetrisPiecePlacement],
    /// The teacher's placement id.
    pub label: u8,
}

/// Distillation-loop tuning.
#[derive(Clone, Debug)]
pub struct DistillConfig {
    pub population: usize,
    pub generations: usize,
    pub elite: usize,
    pub tournament_k: usize,
    /// Number of teacher-labeled states in the dataset.
    pub states: usize,
    /// Game seeds used to sample states with the prior forest.
    pub game_seeds: usize,
    /// Label-noise filter: when the teacher DISAGREES with the prior's own
    /// move, keep the example only if the teacher's Q beats the prior move's
    /// Q by at least this much. On a flat-Q state the most-visited action is
    /// arbitrary — forcing the forest to match arbitrary labels wastes
    /// capacity. `0.0` keeps everything.
    pub min_q_gap: f32,
    pub log_every: usize,
    pub evolve: EvolveConfig,
    pub puct: PuctConfig,
}

impl Default for DistillConfig {
    fn default() -> Self {
        DistillConfig {
            population: 64,
            generations: 500,
            elite: 8,
            tournament_k: 5,
            states: 512,
            game_seeds: 8,
            min_q_gap: 0.0,
            log_every: 1,
            evolve: EvolveConfig::default(),
            puct: PuctConfig::default(),
        }
    }
}

/// Sample up to `num_states` DISTINCT visited states by playing `prior` from
/// the seed set (extending the seed range if games are too short), then label
/// every state with the PUCT teacher IN PARALLEL. States are deduplicated on
/// their 214-bit encoding (repeat labels would waste teacher sims and skew
/// the dataset toward openings). When the teacher disagrees with the prior's
/// own move, the example is kept only if the Q gap clears `cfg.min_q_gap`
/// (see `DistillConfig`). Deterministic given `(prior, cfg, master_seed)`.
pub fn build_distill_set(
    prior: &DecisionForest,
    cfg: &DistillConfig,
    master_seed: u64,
) -> Vec<DistillExample> {
    let compiled = CompiledForest::compile(prior);

    // Collect pre-move snapshots (`TetrisGame` is `Copy`) along prior-policy
    // games until the quota of DISTINCT states is met.
    let mut states: Vec<TetrisGame<StandardTetris>> = Vec::with_capacity(cfg.states);
    let mut seen: std::collections::HashSet<[u8; NUM_INPUTS]> = std::collections::HashSet::new();
    let mut seed = 0u64;
    while states.len() < cfg.states && seed < cfg.game_seeds.max(1) as u64 * 64 {
        let mut game = TetrisGame::<StandardTetris>::new_with_seed(seed + 1);
        while states.len() < cfg.states {
            if game.board.is_lost() {
                break;
            }
            let bits = encode_game(&game);
            if seen.insert(bits) {
                states.push(game);
            }
            let Some(placement) = compiled.vote(&bits, game.current_placements()) else {
                break;
            };
            if game.apply_placement(placement).is_lost == IsLost::LOST {
                break;
            }
        }
        seed += 1;
    }

    states
        .par_iter()
        .enumerate()
        .filter_map(|(i, g)| {
            let mut rng = SmallRng::seed_from_u64(
                master_seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            let bits = encode_game(g);
            let legal = TetrisPiecePlacement::all_from_piece(g.current_piece);
            let actions = puct_root(g, &compiled, &cfg.puct, &mut rng);
            let label = best_action(&actions)?;
            // Q-gap filter: on disagreement, demand a demonstrable
            // improvement over the prior's own move before teaching it. The
            // comparison is only meaningful when the prior's move was
            // actually SIMULATED (an unvisited action carries Q = 0 — the
            // worst case, not "unknown" — which would wave through exactly
            // the arbitrary labels this filter exists to drop). The
            // round-robin first pass in `puct_root` guarantees a visit
            // whenever sims >= |legal|; otherwise we drop conservatively.
            let prior_move = compiled.vote(&bits, legal)?;
            if label.placement != prior_move && cfg.min_q_gap > 0.0 {
                match actions.iter().find(|a| a.placement == prior_move) {
                    Some(pa) if pa.visits > 0 => {
                        if label.q - pa.q < cfg.min_q_gap {
                            return None;
                        }
                    }
                    _ => return None,
                }
            }
            Some(DistillExample {
                bits,
                legal,
                label: label.placement.index(),
            })
        })
        .collect()
}

/// Agreement fitness: the fraction of labeled states where the forest's vote
/// is exactly the teacher's move. Deterministic on a fixed set, so champion
/// updates are strict improvements.
pub fn agreement(forest: &DecisionForest, set: &[DistillExample]) -> f32 {
    if set.is_empty() {
        return 0.0;
    }
    let compiled = CompiledForest::compile(forest);
    let hits = set
        .iter()
        .filter(|ex| {
            compiled
                .vote(&ex.bits, ex.legal)
                .is_some_and(|p| p.index() == ex.label)
        })
        .count();
    hits as f32 / set.len() as f32
}

/// Run the distillation loop: evolve forests toward agreement with a PUCT
/// teacher built around `prior`. Writes `<run>_distilled.json` (the champion,
/// resumable by `train --resume`) and `<run>_distill_log.csv`.
pub fn distill(
    cfg: &DistillConfig,
    artifacts: &TrainArtifacts,
    seed: u64,
    prior: Option<DecisionForest>,
    stop: &AtomicBool,
) -> Result<DecisionForest> {
    std::fs::create_dir_all(&artifacts.out_dir)?;
    let csv_path = artifacts
        .out_dir
        .join(format!("{}_distill_log.csv", artifacts.run_name));
    let mut csv = std::io::BufWriter::new(std::fs::File::create(&csv_path)?);
    writeln!(
        csv,
        "generation,best_agreement,mean_agreement,champ_trees,champ_nodes"
    )?;

    let mut master = SmallRng::seed_from_u64(seed);
    let gen_cond_depth = cfg.evolve.max_cond_depth.min(COND_GEN_DEPTH_CAP);
    let prior_forest = prior.unwrap_or_else(|| {
        DecisionForest::random(&mut master, 64, cfg.evolve.init_tree_depth, gen_cond_depth)
    });

    println!(
        "building distill set: {} states, {} sims/state (teacher prior = {} trees)",
        cfg.states,
        cfg.puct.sims,
        prior_forest.trees.len()
    );
    let set = build_distill_set(&prior_forest, cfg, master.random::<u64>());
    println!("distill set ready: {} labeled states", set.len());

    // Population: the prior itself plus its mutants — distillation refines
    // the prior toward the teacher rather than starting from scratch.
    let mut population: Vec<DecisionForest> = Vec::with_capacity(cfg.population);
    population.push(prior_forest.clone());
    for _ in 1..cfg.population {
        population.push(mutate_forest(&prior_forest, &mut master, &cfg.evolve));
    }

    let prior_for_guard = prior_forest.clone();
    let mut champion = prior_forest;
    let mut champion_fit = agreement(&champion, &set);
    let cull_seed = [1u64];

    for generation in 0..cfg.generations {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let scores: Vec<f32> = population.par_iter().map(|f| agreement(f, &set)).collect();
        let best_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let best_fit = scores[best_idx];
        let mean_fit = scores.iter().sum::<f32>() / scores.len() as f32;

        // The set is fixed, so this is a strict, noise-free improvement gate.
        if best_fit > champion_fit {
            champion_fit = best_fit;
            champion = population[best_idx].clone();
            champion.save_json(
                &artifacts
                    .out_dir
                    .join(format!("{}_distilled.json", artifacts.run_name)),
            )?;
        }

        writeln!(
            csv,
            "{generation},{best_fit:.4},{mean_fit:.4},{},{}",
            champion.trees.len(),
            champion.total_nodes()
        )?;
        csv.flush()?;
        if generation % cfg.log_every.max(1) == 0 {
            println!(
                "distill gen {generation} | agreement best {best_fit:.3} champ {champion_fit:.3} mean {mean_fit:.3} | trees {} nodes {}",
                champion.trees.len(),
                champion.total_nodes()
            );
        }

        let mut order: Vec<usize> = (0..population.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        let mut next: Vec<DecisionForest> = order
            .iter()
            .take(cfg.elite.min(order.len()))
            .map(|&i| population[i].clone())
            .collect();
        let n_children = cfg.population.saturating_sub(next.len());
        let child_seeds: Vec<u64> = (0..n_children).map(|_| master.random::<u64>()).collect();
        let children: Vec<DecisionForest> = child_seeds
            .into_par_iter()
            .map(|cs| {
                let mut rng = SmallRng::seed_from_u64(cs);
                let parent_a = tournament_select(&scores, &population, &mut rng, cfg.tournament_k);
                let parent_b = tournament_select(&scores, &population, &mut rng, cfg.tournament_k);
                let crossed = crossover(parent_a, parent_b, &mut rng);
                let mutated = mutate_forest_owned(crossed, &mut rng, &cfg.evolve);
                let mut child = cull_invalid_trees_owned(mutated, &cull_seed, &cfg.evolve);
                regrow_forest(&mut child, &mut rng, &cfg.evolve);
                child
            })
            .collect();
        next.extend(children);
        population = next;
    }

    champion.save_json(
        &artifacts
            .out_dir
            .join(format!("{}_distilled.json", artifacts.run_name)),
    )?;
    println!("distill complete: champion agreement {champion_fit:.3}");

    // Survival guard (report-only): agreement is a proxy objective — verify
    // the distilled champion did not lose PLAYING strength before it goes
    // back into survival training. Held-out seeds, disjoint from sampling.
    let guard_seeds: Vec<u64> = (0..8u64).map(|i| 1_000_001 + i).collect();
    let prior_surv = fitness(&prior_for_guard, &guard_seeds, 1_000);
    let champ_surv = fitness(&champion, &guard_seeds, 1_000);
    println!(
        "survival check (8 held-out seeds, cap 1000): prior {:.1} -> distilled {:.1} mean pieces{}",
        prior_surv.mean_pieces,
        champ_surv.mean_pieces,
        if champ_surv.mean_pieces + 1.0 < prior_surv.mean_pieces {
            "  [WARN: distilled plays worse — try more generations or --min-q-gap]"
        } else {
            ""
        }
    );
    Ok(champion)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> DistillConfig {
        let evolve = EvolveConfig {
            init_tree_depth: 3,
            cull_sample_pieces: 20,
            min_trees: 3,
            ..EvolveConfig::default()
        };
        DistillConfig {
            population: 6,
            generations: 2,
            elite: 2,
            tournament_k: 2,
            states: 24,
            game_seeds: 2,
            min_q_gap: 0.0,
            log_every: 1,
            evolve,
            puct: PuctConfig {
                sims: 8,
                rollout_depth: 3,
                c_puct: 1.5,
            },
        }
    }

    #[test]
    fn distill_set_labels_are_legal_and_deterministic() {
        let mut rng = SmallRng::seed_from_u64(7);
        let prior = DecisionForest::random(&mut rng, 6, 3, 2);
        let cfg = tiny_cfg();
        let set = build_distill_set(&prior, &cfg, 99);
        assert_eq!(set.len(), cfg.states);
        for ex in &set {
            assert!(ex.legal.iter().any(|p| p.index() == ex.label));
        }
        let set2 = build_distill_set(&prior, &cfg, 99);
        for (a, b) in set.iter().zip(set2.iter()) {
            assert_eq!(a.label, b.label, "labeling must be deterministic");
        }
    }

    #[test]
    fn distill_set_states_are_distinct() {
        let mut rng = SmallRng::seed_from_u64(17);
        let prior = DecisionForest::random(&mut rng, 6, 3, 2);
        let set = build_distill_set(&prior, &tiny_cfg(), 5);
        let mut seen = std::collections::HashSet::new();
        for ex in &set {
            assert!(seen.insert(ex.bits), "duplicate state in distill set");
        }
    }

    #[test]
    fn infinite_q_gap_keeps_only_agreeing_labels() {
        // With an unpassable gap, every disagreeing example is filtered, so
        // all remaining labels must equal the prior's own vote.
        let mut rng = SmallRng::seed_from_u64(19);
        let prior = DecisionForest::random(&mut rng, 6, 3, 2);
        let compiled = CompiledForest::compile(&prior);
        let mut cfg = tiny_cfg();
        cfg.min_q_gap = f32::INFINITY;
        let set = build_distill_set(&prior, &cfg, 5);
        for ex in &set {
            let vote = compiled.vote(&ex.bits, ex.legal).unwrap();
            assert_eq!(vote.index(), ex.label);
        }
        if !set.is_empty() {
            assert!((agreement(&prior, &set) - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn agreement_is_one_when_labels_are_the_forest_votes() {
        // Metric sanity: labeling a set with the forest's own votes gives
        // that forest agreement 1.0.
        let mut rng = SmallRng::seed_from_u64(13);
        let forest = DecisionForest::random(&mut rng, 6, 3, 2);
        let compiled = CompiledForest::compile(&forest);
        let mut game = TetrisGame::<StandardTetris>::new_with_seed(3);
        let mut set = Vec::new();
        for _ in 0..20 {
            if game.board.is_lost() {
                break;
            }
            let bits = encode_game(&game);
            let legal = TetrisPiecePlacement::all_from_piece(game.current_piece);
            let vote = compiled.vote(&bits, legal).unwrap();
            set.push(DistillExample {
                bits,
                legal,
                label: vote.index(),
            });
            if game.apply_placement(vote).is_lost == IsLost::LOST {
                break;
            }
        }
        assert!(!set.is_empty());
        assert!((agreement(&forest, &set) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn distill_writes_a_resumable_champion() {
        let cfg = tiny_cfg();
        let dir = std::env::temp_dir().join("df_distill_test");
        let artifacts = TrainArtifacts {
            out_dir: dir.clone(),
            run_name: "unit".to_string(),
        };
        let stop = AtomicBool::new(false);
        let champion = distill(&cfg, &artifacts, 21, None, &stop).unwrap();
        let path = dir.join("unit_distilled.json");
        assert!(path.exists(), "distilled champion not written");
        let reloaded = DecisionForest::load_json(&path).unwrap();
        assert_eq!(reloaded.trees.len(), champion.trees.len());
        assert!(dir.join("unit_distill_log.csv").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
