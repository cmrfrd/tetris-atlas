//! Root-PUCT teacher: a POLICY-IMPROVEMENT operator over the forest.
//!
//! The AlphaZero cycle with the forest in the network's role: the forest's
//! vote tally (mixed 3:1 with uniform) is the PUCT **prior** P(a), rollouts
//! that follow the forest policy are the **value** estimate, and the PUCT rule
//! `Q(a) + c·P(a)·√N/(1+N(a))` allocates a simulation budget across the legal
//! placements of the root state. The most-visited action is the teacher's
//! label — a one-step-improved version of the forest's own policy. The
//! `distill` loop (see `distill.rs`) then projects that improvement back into
//! circuit form, keeping the played artifact a pure state→move boolean
//! function (NO search at inference; PUCT exists only at training time).
//!
//! Chance handling: `TetrisGame` is `Copy`, so each simulation copies the
//! root state (board, bag membership, current piece all preserved), reseeds
//! the game RNG **and reshuffles the remaining bag stream** — the bag stores
//! the current bag as an already-shuffled queue, so reseeding alone would
//! replay the root's true hidden piece order into every simulation. After
//! the reshuffle, a simulation sees exactly what the student's inputs
//! expose (bag MEMBERSHIP bits) and resamples everything the student cannot
//! know (the order, and pieces beyond the bag) — labels leak no hidden
//! information.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use tetris_game::config::StandardTetris;
use tetris_game::tetris::{IsLost, TetrisGame, TetrisGameRng, TetrisPiecePlacement};

use crate::compiled::CompiledForest;
use crate::encode::encode_game;

/// PUCT teacher parameters.
#[derive(Clone, Copy, Debug)]
pub struct PuctConfig {
    /// Simulation budget per labeled state.
    pub sims: u32,
    /// Rollout horizon in pieces; value = pieces survived / horizon.
    pub rollout_depth: u32,
    /// Exploration constant.
    pub c_puct: f32,
}

impl Default for PuctConfig {
    fn default() -> Self {
        PuctConfig {
            sims: 192,
            rollout_depth: 24,
            c_puct: 1.5,
        }
    }
}

/// PUCT statistics for one legal root action.
#[derive(Clone, Copy, Debug)]
pub struct RootAction {
    pub placement: TetrisPiecePlacement,
    pub prior: f32,
    pub visits: u32,
    /// Mean rollout value in [0, 1] (survival fraction of the horizon).
    pub q: f32,
}

/// Play `game` forward with the forest policy for up to `depth` pieces;
/// return the number of pieces survived.
fn rollout_survived(
    mut game: TetrisGame<StandardTetris>,
    forest: &CompiledForest,
    depth: u32,
) -> u32 {
    let mut survived = 0u32;
    while survived < depth {
        if game.board.is_lost() {
            break;
        }
        let bits = encode_game(&game);
        let Some(placement) = forest.vote(&bits, game.current_placements()) else {
            break;
        };
        // The fatal placement does not count as survived — keeps the value
        // scale uniform (death at depth d scores exactly d, at every depth).
        if game.apply_placement(placement).is_lost == IsLost::LOST {
            break;
        }
        survived += 1;
    }
    survived
}

/// Run root-PUCT at `game`'s current state and return the per-action stats.
/// Deterministic given `(game, forest, cfg, rng state)`. Fitness of an action
/// is pure SURVIVAL of the rollout that follows it — consistent with the
/// training objective.
pub fn puct_root(
    game: &TetrisGame<StandardTetris>,
    forest: &CompiledForest,
    cfg: &PuctConfig,
    rng: &mut SmallRng,
) -> Vec<RootAction> {
    let legal = game.current_placements();
    let bits = encode_game(game);
    let tally = forest.tally(&bits);
    // Prior = the forest's vote distribution mixed 3:1 with uniform. A flat
    // +1-per-action Laplace floor swamps the (typically small) real vote
    // mass; mixing preserves the forest's preference while keeping nonzero
    // exploration mass everywhere. Uniform when the forest is silent.
    let n_legal = legal.len() as f32;
    let total_votes: u32 = legal.iter().map(|p| tally[p.index() as usize]).sum();
    let uniform = 1.0 / n_legal.max(1.0);
    let mut actions: Vec<RootAction> = legal
        .iter()
        .map(|&p| {
            let prior = if total_votes == 0 {
                uniform
            } else {
                0.75 * tally[p.index() as usize] as f32 / total_votes as f32 + 0.25 * uniform
            };
            RootAction {
                placement: p,
                prior,
                visits: 0,
                q: 0.0,
            }
        })
        .collect();
    if actions.is_empty() {
        return actions;
    }

    for sim in 0..cfg.sims {
        // Round-robin first pass: every action is evaluated once before PUCT
        // allocation starts — unvisited actions carry Q = 0 (worst case),
        // which would otherwise bias both selection and downstream Q
        // comparisons (the distill Q-gap filter relies on visited Qs).
        let pick = if (sim as usize) < actions.len() {
            sim as usize
        } else {
            let n_total: u32 = actions.iter().map(|a| a.visits).sum();
            let sqrt_n = ((n_total + 1) as f32).sqrt();
            let mut pick = 0;
            let mut best = f32::NEG_INFINITY;
            for (i, a) in actions.iter().enumerate() {
                let score = a.q + cfg.c_puct * a.prior * sqrt_n / (1.0 + a.visits as f32);
                // Strict '>' keeps the FIRST maximum: ties resolve to the
                // lowest placement index, the convention used everywhere.
                if score > best {
                    best = score;
                    pick = i;
                }
            }
            pick
        };

        // One simulation: copy the root, resample everything the student
        // cannot observe (reseed the RNG for future bags AND reshuffle the
        // current bag's remaining order — membership is preserved), apply
        // the action, then roll out with the forest policy.
        let mut g = *game;
        g.rng = TetrisGameRng::new(rng.random::<u64>());
        g.bag.shuffle(&mut g.rng);
        let lost = g.apply_placement(actions[pick].placement).is_lost == IsLost::LOST;
        let survived = if lost {
            0
        } else {
            1 + rollout_survived(g, forest, cfg.rollout_depth)
        };
        let value = survived as f32 / (1 + cfg.rollout_depth) as f32;
        let a = &mut actions[pick];
        a.visits += 1;
        a.q += (value - a.q) / a.visits as f32;
    }
    actions
}

/// The search's chosen action: most visits, ties broken by higher Q, then
/// lower placement index — a total, deterministic order.
pub fn best_action(actions: &[RootAction]) -> Option<&RootAction> {
    actions.iter().max_by(|a, b| {
        a.visits
            .cmp(&b.visits)
            .then(a.q.partial_cmp(&b.q).unwrap_or(std::cmp::Ordering::Equal))
            .then(b.placement.index().cmp(&a.placement.index()))
    })
}

/// The teacher's label for a state (see `best_action`).
pub fn puct_label(
    game: &TetrisGame<StandardTetris>,
    forest: &CompiledForest,
    cfg: &PuctConfig,
    rng: &mut SmallRng,
) -> Option<TetrisPiecePlacement> {
    best_action(&puct_root(game, forest, cfg, rng)).map(|a| a.placement)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forest::DecisionForest;
    use rand::SeedableRng;

    fn tiny_cfg() -> PuctConfig {
        PuctConfig {
            sims: 24,
            rollout_depth: 4,
            c_puct: 1.5,
        }
    }

    #[test]
    fn puct_root_spends_the_budget_on_legal_actions() {
        let mut rng = SmallRng::seed_from_u64(3);
        let forest = DecisionForest::random(&mut rng, 8, 3, 2);
        let compiled = CompiledForest::compile(&forest);
        let game = TetrisGame::<StandardTetris>::new_with_seed(5);
        let cfg = tiny_cfg();
        let actions = puct_root(&game, &compiled, &cfg, &mut rng);
        let legal = game.current_placements();
        assert_eq!(actions.len(), legal.len());
        assert_eq!(
            actions.iter().map(|a| a.visits).sum::<u32>(),
            cfg.sims,
            "every simulation must land on some action"
        );
        let prior_sum: f32 = actions.iter().map(|a| a.prior).sum();
        assert!(
            (prior_sum - 1.0).abs() < 1e-4,
            "prior must be a distribution"
        );
        for a in &actions {
            assert!(legal.contains(&a.placement));
            assert!((0.0..=1.0).contains(&a.q));
        }
    }

    #[test]
    fn simulations_resample_bag_order_but_preserve_membership() {
        use tetris_game::tetris::TetrisPiece;
        // The per-simulation reshuffle must randomize the hidden bag ORDER
        // (else every sim replays the root's true future — an information
        // leak into the labels) while preserving bag MEMBERSHIP (exactly
        // what the student's bag bits expose).
        let game = TetrisGame::<StandardTetris>::new_with_seed(4);
        let mut g1 = game;
        g1.rng = TetrisGameRng::new(1);
        g1.bag.shuffle(&mut g1.rng);
        let mut g2 = game;
        g2.rng = TetrisGameRng::new(2);
        g2.bag.shuffle(&mut g2.rng);
        for piece in TetrisPiece::all() {
            assert_eq!(g1.bag.contains(piece), game.bag.contains(piece));
            assert_eq!(g2.bag.contains(piece), game.bag.contains(piece));
        }
        assert_eq!(g1.bag.remaining, game.bag.remaining);
        assert_ne!(
            format!("{}", g1.bag),
            format!("{}", g2.bag),
            "different seeds must give different bag orders"
        );
    }

    #[test]
    fn teacher_avoids_immediately_losing_placements() {
        use crate::forest::FOREST_FORMAT_VERSION;
        use crate::tree::DecisionTree;
        // An all-abstain forest plays the constant lowest-index-legal policy,
        // which stacks one region until placements start topping out. At the
        // first state where SOME placements lose immediately and some do not,
        // the teacher (whose value is pure rollout survival) must pick a
        // survivor — the behavioral core of "PUCT improves on the prior".
        let silent = DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![DecisionTree::Abstain],
        };
        let compiled = CompiledForest::compile(&silent);
        let cfg = PuctConfig {
            sims: 64,
            rollout_depth: 4,
            c_puct: 1.5,
        };
        let mut game = TetrisGame::<StandardTetris>::new_with_seed(2);
        let mut tested = false;
        for _ in 0..300 {
            if game.board.is_lost() {
                break;
            }
            let legal = game.current_placements();
            let losers: Vec<bool> = legal
                .iter()
                .map(|&p| {
                    let mut g = game;
                    g.apply_placement(p).is_lost == IsLost::LOST
                })
                .collect();
            let n_losers = losers.iter().filter(|&&l| l).count();
            if n_losers > 0 && n_losers < legal.len() {
                let label =
                    puct_label(&game, &compiled, &cfg, &mut SmallRng::seed_from_u64(7)).unwrap();
                let idx = legal.iter().position(|&p| p == label).unwrap();
                assert!(
                    !losers[idx],
                    "teacher picked an immediately-losing placement"
                );
                tested = true;
                break;
            }
            // Advance with the constant policy.
            let bits = encode_game(&game);
            let Some(p) = compiled.vote(&bits, legal) else {
                break;
            };
            if game.apply_placement(p).is_lost == IsLost::LOST {
                break;
            }
        }
        assert!(tested, "never reached a mixed live/lose state");
    }

    #[test]
    fn puct_label_is_legal_and_deterministic() {
        let mut rng = SmallRng::seed_from_u64(9);
        let forest = DecisionForest::random(&mut rng, 8, 3, 2);
        let compiled = CompiledForest::compile(&forest);
        let game = TetrisGame::<StandardTetris>::new_with_seed(11);
        let cfg = tiny_cfg();
        let l1 = puct_label(&game, &compiled, &cfg, &mut SmallRng::seed_from_u64(42)).unwrap();
        let l2 = puct_label(&game, &compiled, &cfg, &mut SmallRng::seed_from_u64(42)).unwrap();
        assert_eq!(l1, l2, "same rng seed must give the same label");
        assert!(game.current_placements().contains(&l1));
    }
}
