use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use tetris_game::config::StandardTetris;
use tetris_game::tetris::{IsLost, TetrisGame};

use crate::compiled::CompiledForest;
use crate::config::{EvolveConfig, TrainConfig};
use crate::encode::encode_game;
use crate::evolve::{
    crossover, mutate_forest, mutate_forest_owned, silent_gated_tree, tournament_select,
};
use crate::forest::DecisionForest;
use crate::policy::{GameOutcome, play_game_compiled};
use crate::tree::{COND_GEN_DEPTH_CAP, PIECE_GATE_DEPTH};

/// Survival stats of a forest over a seed set at a piece cap. The fitness
/// scalar the loop optimizes is `mean_pieces` — pure survival.
#[derive(Clone, Copy, Debug)]
pub struct FitnessResult {
    pub mean_pieces: f32,
    pub min_pieces: u64,
    pub max_pieces: u64,
    pub mean_lines: f32,
}

/// Below this piece cap, per-seed games are too short for rayon task overhead
/// to pay off (population-level parallelism already saturates the cores);
/// above it, long games make per-seed parallelism a real win (and it fixes
/// the otherwise-serial champion re-evals and `bench`).
const PAR_SEEDS_MIN_CAP: u64 = 10_000;

/// Play `forest` on every seed up to `cap`, aggregating survival. The forest
/// is compiled ONCE and shared across all games; seeds run in parallel only
/// at high caps (see `PAR_SEEDS_MIN_CAP`). Aggregation is integer-based, so
/// results are deterministic regardless of scheduling.
pub fn fitness(forest: &DecisionForest, seeds: &[u64], cap: u64) -> FitnessResult {
    let compiled = CompiledForest::compile(forest);
    let outcomes: Vec<GameOutcome> = if cap >= PAR_SEEDS_MIN_CAP && seeds.len() > 1 {
        seeds
            .par_iter()
            .map(|&s| play_game_compiled(&compiled, s, cap))
            .collect()
    } else {
        seeds
            .iter()
            .map(|&s| play_game_compiled(&compiled, s, cap))
            .collect()
    };
    let n = outcomes.len().max(1) as f32;
    let sum_pieces: u64 = outcomes.iter().map(|o| o.pieces_placed).sum();
    let sum_lines: u64 = outcomes.iter().map(|o| o.lines_cleared).sum();
    let min_pieces = outcomes.iter().map(|o| o.pieces_placed).min().unwrap_or(0);
    let max_pieces = outcomes.iter().map(|o| o.pieces_placed).max().unwrap_or(0);
    FitnessResult {
        mean_pieces: sum_pieces as f32 / n,
        min_pieces,
        max_pieces,
        mean_lines: sum_lines as f32 / n,
    }
}

/// Per-tree valid-output counts over one sample game from `seed`, plus the
/// number of decisions. A tree is "valid" at a decision when its placement id
/// is legal for the current piece (`DecisionTree::output_is_valid`). The game
/// advances by the forest's actual vote.
pub fn tree_valid_counts(
    forest: &DecisionForest,
    seed: u64,
    sample_pieces: u64,
) -> (Vec<u64>, u64) {
    let compiled = CompiledForest::compile(forest);
    let mut game = TetrisGame::<StandardTetris>::new_with_seed(seed);
    let mut valid = vec![0u64; forest.trees.len()];
    let mut decisions = 0u64;
    while decisions < sample_pieces {
        if game.board.is_lost() {
            break;
        }
        let bits = encode_game(&game);
        let legal = game.current_placements();
        if legal.is_empty() {
            break;
        }
        // One `decide` per tree serves both the validity count and the vote
        // tally; legality is an O(1) mask lookup instead of a scan.
        let mut legal_mask = [false; 256];
        for p in legal {
            legal_mask[p.index() as usize] = true;
        }
        let mut votes = [0u32; 256];
        for i in 0..compiled.num_trees() {
            match compiled.decide(i, &bits) {
                Some(pid) => {
                    votes[pid as usize] += 1;
                    if legal_mask[pid as usize] {
                        valid[i] += 1;
                    }
                }
                // Abstaining is a sanctioned "no recommendation" — never invalid.
                None => valid[i] += 1,
            }
        }
        let Some(placement) = DecisionForest::winner_from_tally(&votes, legal) else {
            break;
        };
        let lost = game.apply_placement(placement).is_lost == IsLost::LOST;
        decisions += 1;
        if lost {
            break;
        }
    }
    (valid, decisions)
}

/// **Autokill invalid-output trees.** Measure each tree's valid-output rate over
/// sample games from `seeds` and drop trees below `cfg.min_valid_rate` (default
/// 1.0 = any tree that ever outputs an invalid placement is culled). Never
/// empties a forest — if every tree is below threshold, the single most-valid
/// tree is kept. `min_valid_rate <= 0.0` disables culling.
pub fn cull_invalid_trees(
    forest: &DecisionForest,
    seeds: &[u64],
    cfg: &EvolveConfig,
) -> DecisionForest {
    cull_invalid_trees_owned(forest.clone(), seeds, cfg)
}

/// `cull_invalid_trees` without cloning the survivors — trees are dropped or
/// kept IN PLACE (the reproduce pipeline owns its offspring, and forest
/// clones dominate generation wall-time).
pub fn cull_invalid_trees_owned(
    mut forest: DecisionForest,
    seeds: &[u64],
    cfg: &EvolveConfig,
) -> DecisionForest {
    if cfg.min_valid_rate <= 0.0 || forest.trees.is_empty() {
        return forest;
    }
    let n = forest.trees.len();
    let mut total_valid = vec![0u64; n];
    let mut total_dec = 0u64;
    for &seed in seeds {
        let (valid, dec) = tree_valid_counts(&forest, seed, cfg.cull_sample_pieces);
        for i in 0..n {
            total_valid[i] += valid[i];
        }
        total_dec += dec;
    }
    if total_dec == 0 {
        return forest;
    }
    let any_kept = total_valid
        .iter()
        .any(|&v| v as f64 / total_dec as f64 >= cfg.min_valid_rate);
    if any_kept {
        let mut i = 0;
        forest.trees.retain(|_| {
            let keep = total_valid[i] as f64 / total_dec as f64 >= cfg.min_valid_rate;
            i += 1;
            keep
        });
    } else {
        // Never empty a forest: keep the single most-valid tree.
        let best = (0..n).max_by_key(|&i| total_valid[i]).unwrap();
        forest.trees.swap(0, best);
        forest.trees.truncate(1);
    }
    forest
}

/// Top `forest` back up to the regrowth floor (`cfg.min_trees`, bounded by
/// `cfg.max_trees`) with SILENT-SPECIALIST gated trees — 100% valid by
/// construction (they pass the autokill whose losses they replace) and
/// mostly-abstaining, so regrowth barely perturbs the plurality: a
/// near-neutral insertion that can then evolve votes. The counterweight to
/// `cull_invalid_trees`: the cull removes trees that break, regrowth keeps
/// the forest MANY-treed so the vote stays a real accumulation.
pub fn regrow_forest(forest: &mut DecisionForest, rng: &mut SmallRng, cfg: &EvolveConfig) {
    let floor = cfg.min_trees.min(cfg.max_trees);
    while forest.trees.len() < floor {
        forest.trees.push(silent_gated_tree(rng, cfg));
    }
}

/// Eval seeds for a generation, FROZEN within each `seed_block`-generation
/// window and rotated across windows. Within a block the objective is
/// stationary, so selection climbs a fixed landscape instead of chasing
/// common-mode seed noise; rotation across blocks still prevents overfitting
/// any one sequence set.
pub fn eval_seeds(generation: usize, seed_block: usize, n: usize) -> Vec<u64> {
    let block = (generation / seed_block.max(1)) as u64;
    (0..n as u64).map(|i| block * 1_000_003 + i + 1).collect()
}

/// Where to write outputs.
pub struct TrainArtifacts {
    pub out_dir: PathBuf,
    pub run_name: String,
}

/// The curriculum ladder of per-game piece caps.
const CURRICULUM: [u64; 5] = [200, 1_000, 10_000, 100_000, 1_000_000];

/// Run the genetic-programming training loop. Fitness is survival (mean pieces)
/// over a seed set that freezes for `seed_block` generations (see `eval_seeds`).
/// The champion must defend its title each generation on the CURRENT seed set
/// (same-seed gate — no cross-seed-set high-water mark). Offspring are
/// autokilled of invalid-output trees and regrown to the `min_trees` floor
/// each generation. Escalates the piece cap up `CURRICULUM` once the champion
/// survives the current cap on every eval seed. Writes `<run>_champion.json`
/// on improvement, a `<run>_log.csv`, and (only with `snapshot_every`)
/// periodic `<run>_gen_<n>.json` snapshots.
pub fn train(
    cfg: &TrainConfig,
    artifacts: &TrainArtifacts,
    seed: u64,
    resume: Option<DecisionForest>,
    stop: &AtomicBool,
) -> Result<DecisionForest> {
    std::fs::create_dir_all(&artifacts.out_dir)?;
    let csv_path = artifacts
        .out_dir
        .join(format!("{}_log.csv", artifacts.run_name));
    let mut csv = std::io::BufWriter::new(std::fs::File::create(&csv_path)?);
    writeln!(
        csv,
        "generation,cap,cap_stage,best_survival,mean_survival,champ_min_pieces,champ_mean_pieces,champ_max_pieces,champ_mean_lines,champ_trees,champ_nodes"
    )?;

    let mut master = SmallRng::seed_from_u64(seed);
    let mut population: Vec<DecisionForest> = Vec::with_capacity(cfg.population);
    if let Some(seed_forest) = resume {
        population.push(seed_forest.clone());
        for _ in 1..cfg.population {
            population.push(mutate_forest(&seed_forest, &mut master, &cfg.evolve));
        }
    } else {
        // Gated trees add PIECE_GATE_DEPTH levels; clamp the body depth so
        // fresh forests respect max_tree_depth from generation zero.
        let init_sub_depth = cfg
            .evolve
            .init_tree_depth
            .min(cfg.evolve.max_tree_depth.saturating_sub(PIECE_GATE_DEPTH));
        for _ in 0..cfg.population {
            population.push(DecisionForest::random(
                &mut master,
                cfg.init_trees,
                init_sub_depth,
                cfg.evolve.max_cond_depth.min(COND_GEN_DEPTH_CAP),
            ));
        }
    }

    let mut cap_stage = CURRICULUM
        .iter()
        .position(|&c| c >= cfg.start_cap)
        .unwrap_or(0);
    let mut champion = population[0].clone();
    // Within a frozen seed block at an unchanged cap, fitness is
    // deterministic — elites (byte-identical copies) and a held incumbent
    // champion keep last generation's scores instead of replaying games.
    let mut elite_carry: Vec<FitnessResult> = Vec::new();
    let mut incumbent_cache: Option<FitnessResult> = None;
    let mut prev_block_key: Option<(usize, usize)> = None;

    for generation in 0..cfg.generations {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let cap = CURRICULUM[cap_stage];
        let seeds = eval_seeds(generation, cfg.seed_block, cfg.seeds);
        let block_key = (generation / cfg.seed_block.max(1), cap_stage);
        let reuse_scores = prev_block_key == Some(block_key);

        // Score by index — no per-generation clones of the population.
        let results: Vec<FitnessResult> = population
            .par_iter()
            .enumerate()
            .map(|(i, f)| {
                if reuse_scores && i < elite_carry.len() {
                    elite_carry[i]
                } else {
                    fitness(f, &seeds, cap)
                }
            })
            .collect();
        let scores: Vec<f32> = results.iter().map(|r| r.mean_pieces).collect();
        let best_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let best_fit = scores[best_idx];
        let mean_fit = scores.iter().sum::<f32>() / scores.len() as f32;

        // SAME-SEED champion gate: re-score the incumbent on THIS generation's
        // seeds and replace it only if the generation's best beats it on the
        // identical seed set. (The previous rule compared against a historical
        // high-water mark set on a DIFFERENT seed set — a winner's-curse
        // ratchet: the bar sat ~3.4σ of seed noise above the champion's true
        // skill, freezing champion identity/#trees/#nodes for thousands of
        // generations and sometimes installing lucky-but-worse genomes.)
        let incumbent_stats = match incumbent_cache {
            Some(stats) if reuse_scores => stats,
            _ => fitness(&champion, &seeds, cap),
        };
        let champ_stats = if best_fit > incumbent_stats.mean_pieces {
            champion = population[best_idx].clone();
            champion.save_json(
                &artifacts
                    .out_dir
                    .join(format!("{}_champion.json", artifacts.run_name)),
            )?;
            results[best_idx]
        } else {
            incumbent_stats
        };
        incumbent_cache = Some(champ_stats);
        writeln!(
            csv,
            "{generation},{cap},{cap_stage},{best_fit:.2},{mean_fit:.2},{},{:.2},{},{:.2},{},{}",
            champ_stats.min_pieces,
            champ_stats.mean_pieces,
            champ_stats.max_pieces,
            champ_stats.mean_lines,
            champion.trees.len(),
            champion.total_nodes()
        )?;
        csv.flush()?;

        if generation % cfg.log_every.max(1) == 0 {
            println!(
                "gen {generation} | cap {cap} | best_survival {best_fit:.1} | mean {mean_fit:.1} | champ min/mean/max pieces {}/{:.0}/{} | lines {:.1} | trees {} nodes {}",
                champ_stats.min_pieces,
                champ_stats.mean_pieces,
                champ_stats.max_pieces,
                champ_stats.mean_lines,
                champion.trees.len(),
                champion.total_nodes()
            );
        }
        // Periodic snapshots are opt-in (the synchronous write causes a
        // visible pause); the champion JSON is saved on every improvement
        // regardless.
        if let Some(every) = cfg.snapshot_every {
            if every > 0 && generation % every == 0 {
                champion.save_json(
                    &artifacts
                        .out_dir
                        .join(format!("{}_gen_{generation:08}.json", artifacts.run_name)),
                )?;
            }
        }

        if champ_stats.min_pieces >= cap && cap_stage + 1 < CURRICULUM.len() {
            cap_stage += 1;
            println!(
                "── curriculum: champion cleared cap {cap} on all seeds; escalating to {} ──",
                CURRICULUM[cap_stage]
            );
        }

        // Next generation: elitism by ranked index (clones only the elites),
        // then offspring built IN PARALLEL — each child gets its own RNG seed
        // drawn from the master stream, so the run stays deterministic
        // regardless of thread scheduling. Each offspring is autokilled of
        // invalid-output trees (one short sample game).
        let mut order: Vec<usize> = (0..population.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        let mut next: Vec<DecisionForest> = order
            .iter()
            .take(cfg.elite.min(order.len()))
            .map(|&i| population[i].clone())
            .collect();
        let n_children = cfg.population.saturating_sub(next.len());
        let child_seeds: Vec<u64> = (0..n_children).map(|_| master.random::<u64>()).collect();
        let cull_seed: Vec<u64> = seeds[..seeds.len().min(1)].to_vec();
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
        // Elites occupy the first slots of the next population, in ranked
        // order — carry their (deterministic) scores into the next
        // generation of this seed block.
        elite_carry = order
            .iter()
            .take(cfg.elite.min(order.len()))
            .map(|&i| results[i])
            .collect();
        prev_block_key = Some(block_key);
        next.extend(children);
        population = next;
    }

    champion.save_json(
        &artifacts
            .out_dir
            .join(format!("{}_champion.json", artifacts.run_name)),
    )?;
    Ok(champion)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitness_reports_survival() {
        let mut rng = SmallRng::seed_from_u64(2);
        let forest = DecisionForest::random(&mut rng, 4, 4, 2);
        let seeds = [1u64, 2];
        let r = fitness(&forest, &seeds, 120);
        assert!(r.min_pieces <= r.max_pieces);
        assert!(r.mean_pieces >= r.min_pieces as f32 - 0.001);
        assert!(r.mean_pieces <= r.max_pieces as f32 + 0.001);
    }

    #[test]
    fn cull_never_empties_and_only_removes() {
        let mut rng = SmallRng::seed_from_u64(4);
        let forest = DecisionForest::random(&mut rng, 8, 4, 2);
        let cfg = EvolveConfig {
            cull_sample_pieces: 40,
            ..EvolveConfig::default()
        }; // min_valid_rate 1.0 (aggressive)
        let culled = cull_invalid_trees(&forest, &[1u64], &cfg);
        assert!(!culled.trees.is_empty(), "cull emptied the forest");
        assert!(culled.trees.len() <= forest.trees.len());
    }

    #[test]
    fn eval_seeds_freeze_within_a_block_and_rotate_across_blocks() {
        // Same seeds throughout one block...
        assert_eq!(eval_seeds(0, 20, 4), eval_seeds(19, 20, 4));
        // ...different seeds in the next block...
        assert_ne!(eval_seeds(19, 20, 4), eval_seeds(20, 20, 4));
        // ...and a zero block size degrades to per-generation rotation.
        assert_ne!(eval_seeds(0, 0, 4), eval_seeds(1, 0, 4));
        assert_eq!(eval_seeds(0, 20, 3).len(), 3);
    }

    #[test]
    fn fresh_forests_survive_the_aggressive_cull_intact() {
        // Random forests are piece-gated — 100% valid by construction — so
        // cull-at-1.0 keeps every tree. This is the fix for the trees-1
        // collapse: the generator produces trees that can pass the bar.
        let mut rng = SmallRng::seed_from_u64(8);
        let forest = DecisionForest::random(&mut rng, 16, 4, 3);
        let cfg = EvolveConfig {
            cull_sample_pieces: 60,
            ..EvolveConfig::default()
        }; // min_valid_rate 1.0
        let culled = cull_invalid_trees(&forest, &[1u64, 2], &cfg);
        assert_eq!(
            culled.trees.len(),
            16,
            "piece-gated trees must all pass cull-at-1.0"
        );
    }

    #[test]
    fn regrow_tops_forests_back_up_with_fully_valid_trees() {
        use crate::forest::FOREST_FORMAT_VERSION;
        use crate::tree::DecisionTree;
        let mut rng = SmallRng::seed_from_u64(12);
        let mut forest = DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![DecisionTree::Abstain],
        };
        let cfg = EvolveConfig::default(); // min_trees 32
        regrow_forest(&mut forest, &mut rng, &cfg);
        assert_eq!(forest.trees.len(), cfg.min_trees);
        // Every regrown tree is 100% valid over a real game.
        let (valid, decisions) = tree_valid_counts(&forest, 5, 60);
        assert!(decisions > 0);
        for (i, v) in valid.iter().enumerate() {
            assert_eq!(*v, decisions, "regrown tree {i} not fully valid");
        }
    }

    #[test]
    fn cull_disabled_is_identity() {
        let mut rng = SmallRng::seed_from_u64(6);
        let forest = DecisionForest::random(&mut rng, 4, 4, 2);
        let cfg = EvolveConfig {
            min_valid_rate: 0.0,
            ..EvolveConfig::default()
        };
        let culled = cull_invalid_trees(&forest, &[1u64], &cfg);
        assert_eq!(culled.trees.len(), forest.trees.len());
    }

    /// Build a tree that branches on the 7 current-piece one-hot bits and
    /// outputs a placement that is LEGAL for that piece — a 100%-valid tree.
    /// This is the structure the aggressive cull selects for.
    fn piece_conditional_tree() -> crate::tree::DecisionTree {
        use crate::encode::{NUM_CELLS, NUM_PIECES};
        use crate::tree::{BoolExpr, DecisionTree};
        use tetris_game::tetris::{TetrisPiece, TetrisPiecePlacement};
        let mut tree = DecisionTree::Leaf(0); // unreachable fallback
        for (i, piece) in TetrisPiece::all().iter().enumerate() {
            let legal_pid = TetrisPiecePlacement::all_from_piece(*piece)[0].index();
            tree = DecisionTree::Branch {
                cond: BoolExpr::Cell(NUM_CELLS + NUM_PIECES + i),
                yes: Box::new(DecisionTree::Leaf(legal_pid)),
                no: Box::new(tree),
            };
        }
        tree
    }

    #[test]
    fn aggressive_cull_keeps_piece_conditional_tree_and_kills_constant() {
        use crate::forest::FOREST_FORMAT_VERSION;
        use crate::tree::DecisionTree;
        // One always-valid (piece-conditional) tree + one constant tree that is
        // invalid whenever the current piece is not the one pid 0 belongs to.
        let forest = DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![piece_conditional_tree(), DecisionTree::Leaf(0)],
        };
        let cfg = EvolveConfig {
            cull_sample_pieces: 40,
            ..EvolveConfig::default()
        }; // min_valid_rate = 1.0
        // Sanity: the conditional tree is valid on every decision; the
        // constant tree is not.
        let (valid, decisions) = tree_valid_counts(&forest, 3, 40);
        assert!(decisions > 0);
        assert_eq!(
            valid[0], decisions,
            "piece-conditional tree must be 100% valid"
        );
        assert!(
            valid[1] < decisions,
            "constant tree must sometimes be invalid"
        );
        // The aggressive cull keeps exactly the conditional tree.
        let culled = cull_invalid_trees(&forest, &[3u64], &cfg);
        assert_eq!(culled.trees.len(), 1, "constant tree should be culled");
        let (v2, d2) = tree_valid_counts(&culled, 5, 40);
        assert_eq!(v2[0], d2, "survivor must be the always-valid tree");
    }

    #[test]
    fn cull_keeps_the_most_valid_tree_when_all_fail_threshold() {
        use crate::forest::FOREST_FORMAT_VERSION;
        use crate::tree::DecisionTree;
        // Two constant voters: each is valid only when its piece is current,
        // so both fall below min_valid_rate = 1.0 — the fallback keeps exactly
        // the single most-valid tree instead of emptying the forest.
        let forest = DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![DecisionTree::Leaf(0), DecisionTree::Leaf(20)],
        };
        let cfg = EvolveConfig {
            cull_sample_pieces: 40,
            ..EvolveConfig::default()
        };
        let culled = cull_invalid_trees(&forest, &[1u64], &cfg);
        assert_eq!(culled.trees.len(), 1, "fallback keeps exactly one tree");
    }

    #[test]
    fn abstaining_tree_survives_aggressive_cull() {
        use crate::forest::FOREST_FORMAT_VERSION;
        use crate::tree::DecisionTree;
        // An all-abstain tree never emits an invalid placement, so cull-at-1.0
        // keeps it; the constant tree gets culled.
        let forest = DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![DecisionTree::Abstain, DecisionTree::Leaf(0)],
        };
        let cfg = EvolveConfig {
            cull_sample_pieces: 40,
            ..EvolveConfig::default()
        };
        let culled = cull_invalid_trees(&forest, &[3u64], &cfg);
        assert_eq!(culled.trees.len(), 1);
        let (v, d) = tree_valid_counts(&culled, 5, 40);
        assert!(d > 0);
        assert_eq!(v[0], d, "the abstainer is 100% valid");
    }

    #[test]
    fn train_is_deterministic_for_a_fixed_seed() {
        let evolve = EvolveConfig {
            init_tree_depth: 3,
            cull_sample_pieces: 20,
            min_trees: 3, // keep test forests tiny
            ..EvolveConfig::default()
        };
        let cfg = TrainConfig {
            population: 4,
            generations: 2,
            elite: 1,
            tournament_k: 2,
            seeds: 2,
            start_cap: 200,
            init_trees: 3,
            evolve,
            ..TrainConfig::default()
        };
        let stop = AtomicBool::new(false);
        let mut champs = Vec::new();
        for run in 0..2 {
            let dir = std::env::temp_dir().join(format!("df_det_test_{run}"));
            let artifacts = TrainArtifacts {
                out_dir: dir.clone(),
                run_name: "det".to_string(),
            };
            train(&cfg, &artifacts, 99, None, &stop).unwrap();
            champs.push(std::fs::read(dir.join("det_champion.json")).unwrap());
            let _ = std::fs::remove_dir_all(&dir);
        }
        assert_eq!(
            champs[0], champs[1],
            "same seed must give identical champions"
        );
    }

    #[test]
    fn train_writes_champion_and_csv() {
        let evolve = EvolveConfig {
            init_tree_depth: 3,
            cull_sample_pieces: 30, // keep the per-offspring cull cheap
            min_trees: 4,           // keep test forests tiny
            ..EvolveConfig::default()
        };
        let cfg = TrainConfig {
            population: 6,
            generations: 2,
            elite: 2,
            tournament_k: 2,
            seeds: 2,
            start_cap: 200, // curriculum floor; games end at loss long before this
            init_trees: 4,
            seed_block: 2,
            log_every: 1,
            snapshot_every: None,
            evolve,
        };
        let dir = std::env::temp_dir().join("df_train_test");
        let artifacts = TrainArtifacts {
            out_dir: dir.clone(),
            run_name: "unit".to_string(),
        };
        let stop = AtomicBool::new(false);
        let champion = train(&cfg, &artifacts, 7, None, &stop).unwrap();
        assert!(
            dir.join("unit_champion.json").exists(),
            "champion not written"
        );
        let reloaded = DecisionForest::load_json(&dir.join("unit_champion.json")).unwrap();
        assert_eq!(reloaded.trees.len(), champion.trees.len());
        assert!(dir.join("unit_log.csv").exists(), "csv log not written");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
