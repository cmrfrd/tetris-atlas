use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use tetris_game::tetris::{TetrisPiece, TetrisPiecePlacement};

use crate::config::EvolveConfig;
use crate::forest::{DecisionForest, FOREST_FORMAT_VERSION};
use crate::tree::{
    BoolExpr, COND_GEN_DEPTH_CAP, DecisionTree, PIECE_GATE_DEPTH, SILENT_TREE_ABSTAIN_PROB,
    governing_piece, piece_bit_piece, random_boolexpr_leaf, random_pid, random_pid_of_piece,
    random_piece_gated_tree, random_piece_gated_tree_with_abstain, random_tree_leaf_for,
};

/// When a voting leaf is retargeted, how often the new placement is drawn
/// from the SAME piece as the old one. Same-piece retargets preserve the
/// validity of a piece-gated path; the remaining fraction explores across
/// pieces (and is policed by the autokill).
const SAME_PIECE_RETARGET_PROB: f64 = 0.8;

/// When the tree-add mutation fires, how often it DUPLICATES an existing
/// evolved tree (then diverges the copy with one mutation) instead of
/// injecting a fresh random one. Gene duplication is the fixation path past
/// the `min_trees` floor: a duplicate reinforces votes the forest already
/// casts (near-neutral), where a fresh random tree perturbs the plurality in
/// a random direction and its carrier loses tournaments.
const DUP_TREE_PROB: f64 = 0.7;

/// One structural node mutation per this many trees (minimum one mutation).
/// A single-node change in one tree of a 32-tree plurality is almost always
/// behaviorally silent — mutation reach must scale with forest size for
/// offspring to differ from their parents at all.
const TREES_PER_MUTATION: usize = 8;

/// A fresh piece-gated tree sized from the config: board-logic depth under
/// each gate is `init_tree_depth`, clamped so gate + body respects
/// `max_tree_depth`. The generator for the exploratory share of tree-add.
pub fn fresh_gated_tree(rng: &mut SmallRng, cfg: &EvolveConfig) -> DecisionTree {
    let sub_depth = cfg
        .init_tree_depth
        .min(cfg.max_tree_depth.saturating_sub(PIECE_GATE_DEPTH));
    random_piece_gated_tree(rng, sub_depth, cfg.max_cond_depth.min(COND_GEN_DEPTH_CAP))
}

/// A SILENT-SPECIALIST gated tree (mostly-abstain leaves): still 100% valid
/// by construction, but adding one barely moves the plurality — the
/// near-neutral generator for regrowth, so topping a forest back up does not
/// randomize its vote.
pub fn silent_gated_tree(rng: &mut SmallRng, cfg: &EvolveConfig) -> DecisionTree {
    let sub_depth = cfg
        .init_tree_depth
        .min(cfg.max_tree_depth.saturating_sub(PIECE_GATE_DEPTH));
    random_piece_gated_tree_with_abstain(
        rng,
        sub_depth,
        cfg.max_cond_depth.min(COND_GEN_DEPTH_CAP),
        SILENT_TREE_ABSTAIN_PROB,
    )
}

/// Return a mutated copy of `parent`: tree add/remove, plus structural node
/// mutations on random decision nodes of random trees (mutate a leaf's
/// placement id, mutate/regrow a branch's boolean condition, grow a leaf into
/// a branch, or prune a branch to a leaf). The mutation COUNT scales with
/// forest size (one per `TREES_PER_MUTATION` trees, min 1) so offspring of
/// large forests are not behavioral clones of their parents. Tree-add usually
/// duplicates-and-diverges an existing evolved tree (`DUP_TREE_PROB`), giving
/// forests a real path above the `min_trees` floor. Always leaves ≥ 1 tree
/// and ≤ `max_trees`.
pub fn mutate_forest(
    parent: &DecisionForest,
    rng: &mut SmallRng,
    cfg: &EvolveConfig,
) -> DecisionForest {
    mutate_forest_owned(parent.clone(), rng, cfg)
}

/// `mutate_forest` without the defensive parent clone — the reproduce
/// pipeline already owns the crossover output, and forest clones dominate
/// generation wall-time.
pub fn mutate_forest_owned(
    mut child: DecisionForest,
    rng: &mut SmallRng,
    cfg: &EvolveConfig,
) -> DecisionForest {
    if child.trees.len() < cfg.max_trees && rng.random_bool(cfg.tree_add_prob) {
        let new_tree = if !child.trees.is_empty() && rng.random_bool(DUP_TREE_PROB) {
            // Gene duplication: copy an evolved tree, then diverge the copy.
            let src = rng.random_range(0..child.trees.len());
            let mut t = child.trees[src].clone();
            mutate_tree_structurally(&mut t, rng, cfg);
            t
        } else {
            fresh_gated_tree(rng, cfg)
        };
        child.trees.push(new_tree);
    }
    if child.trees.len() > 1 && rng.random_bool(cfg.tree_remove_prob) {
        let idx = rng.random_range(0..child.trees.len());
        child.trees.remove(idx);
    }

    if !child.trees.is_empty() {
        let n_muts = (child.trees.len() / TREES_PER_MUTATION).max(1);
        for _ in 0..n_muts {
            let tidx = rng.random_range(0..child.trees.len());
            mutate_tree_structurally(&mut child.trees[tidx], rng, cfg);
        }
    }

    child.version = FOREST_FORMAT_VERSION;
    child
}

/// One structural node mutation on `tree`, VALIDITY-AWARE: the target is
/// redrawn away from the piece-gate spine (a gate hit is near-certain tree
/// death — prune collapses the chain to one piece's leaf, and a Cell reroll
/// breaks the gate 213/214 of the time — so the cull destroys the whole
/// evolved tree and the mutation budget just feeds churn), and the mutation
/// receives the target's GOVERNING PIECE so any vote it creates is legal
/// exactly where that node runs. Mutations on gated trees thereby preserve
/// 100%-validity instead of invalidating ~a quarter of children.
fn mutate_tree_structurally(tree: &mut DecisionTree, rng: &mut SmallRng, cfg: &EvolveConfig) {
    // Grow mutations are allowed only below the structural depth cap.
    let allow_grow = tree.depth() < cfg.max_tree_depth;
    let n = tree.count_nodes();
    let mut target = rng.random_range(0..n);
    for _ in 0..8 {
        let on_gate_spine = matches!(
            tree.node_at_mut(target),
            Some(DecisionTree::Branch { cond, .. }) if piece_bit_piece(cond).is_some()
        );
        if !on_gate_spine {
            break;
        }
        target = rng.random_range(0..n);
    }
    let ctx = governing_piece(tree, target);
    if let Some(node) = tree.node_at_mut(target) {
        mutate_tree_node(node, rng, cfg, allow_grow, ctx);
    }
}

fn mutate_tree_node(
    node: &mut DecisionTree,
    rng: &mut SmallRng,
    cfg: &EvolveConfig,
    allow_grow: bool,
    ctx: Option<TetrisPiece>,
) {
    match node {
        DecisionTree::Leaf(pid) => {
            if allow_grow && rng.random_bool(cfg.grow_prob) {
                *node = DecisionTree::Branch {
                    cond: BoolExpr::random(rng, cfg.max_cond_depth.min(3).max(1)),
                    yes: Box::new(random_tree_leaf_for(rng, ctx)),
                    no: Box::new(DecisionTree::Leaf(*pid)),
                };
            } else if rng.random_bool(crate::tree::ABSTAIN_LEAF_PROB) {
                // A vote can mutate into silence — the specialization operator.
                *node = DecisionTree::Abstain;
            } else if let Some(piece) = ctx {
                // The governing gate is known: retarget within ITS piece —
                // exactly the votes that are legal where this leaf runs.
                *pid = random_pid_of_piece(rng, piece);
            } else if rng.random_bool(SAME_PIECE_RETARGET_PROB) {
                // Same-piece retarget: keeps a piece-gated path valid.
                *pid = random_pid_of_piece(rng, TetrisPiecePlacement::from_index(*pid).piece);
            } else {
                let _ = cfg.leaf_mutation_prob; // reserved knob (kept referenced)
                *pid = random_pid(rng);
            }
        }
        DecisionTree::Abstain => {
            if allow_grow && rng.random_bool(cfg.grow_prob) {
                *node = DecisionTree::Branch {
                    cond: BoolExpr::random(rng, cfg.max_cond_depth.min(3).max(1)),
                    yes: Box::new(random_tree_leaf_for(rng, ctx)),
                    no: Box::new(DecisionTree::Abstain),
                };
            } else {
                // Silence can mutate back into a vote — legal for the
                // governing gate's piece, so specialists can wake up without
                // being culled for it.
                *node = match ctx {
                    Some(piece) => DecisionTree::Leaf(random_pid_of_piece(rng, piece)),
                    None => DecisionTree::Leaf(random_pid(rng)),
                };
            }
        }
        DecisionTree::Branch { cond, yes, .. } => {
            if rng.random_bool(cfg.prune_prob) {
                *node = first_leaf(yes);
            } else if rng.random_bool(cfg.cond_mutation_prob) {
                // Rare macro jump: regrow the whole condition from scratch
                // (at generation depth — deep formulas are earned via growth).
                *cond = BoolExpr::random(rng, cfg.max_cond_depth.min(COND_GEN_DEPTH_CAP));
            } else {
                // Default: a LOCAL tweak of the learned condition. (Wholesale
                // replacement as the default arm made ~34% of all node
                // mutations uncorrelated macro jumps — no local structure.)
                mutate_cond(cond, rng, cfg);
            }
        }
    }
}

/// The first leaf node (vote or abstention) reached by always taking `yes` —
/// what a pruned branch collapses to.
fn first_leaf(tree: &DecisionTree) -> DecisionTree {
    let mut node = tree;
    loop {
        match node {
            DecisionTree::Leaf(p) => return DecisionTree::Leaf(*p),
            DecisionTree::Abstain => return DecisionTree::Abstain,
            DecisionTree::Branch { yes, .. } => node = yes,
        }
    }
}

/// Mutate a boolean condition in place.
fn mutate_cond(cond: &mut BoolExpr, rng: &mut SmallRng, cfg: &EvolveConfig) {
    // Growing a leaf inside the condition adds up to 2 levels; only allow it
    // when the whole condition stays within the documented cap.
    let allow_cond_grow = cond.depth() + 2 <= cfg.max_cond_depth;
    let n = cond.count_nodes();
    let target = rng.random_range(0..n);
    let Some(node) = cond.node_at_mut(target) else {
        return;
    };
    let is_gate = !matches!(node, BoolExpr::Cell(_) | BoolExpr::Const(_));

    if is_gate && rng.random_bool(cfg.prune_prob) {
        *node = random_boolexpr_leaf(rng);
        return;
    }
    if !is_gate && allow_cond_grow && rng.random_bool(cfg.grow_prob) {
        *node = BoolExpr::random(rng, 2);
        return;
    }
    match node {
        BoolExpr::Cell(i) => {
            *i = rng.random_range(0..crate::encode::NUM_INPUTS);
        }
        BoolExpr::Const(b) => {
            *b = !*b;
        }
        BoolExpr::Not(_) => {
            let taken = std::mem::replace(node, BoolExpr::Const(false));
            if let BoolExpr::Not(inner) = taken {
                *node = *inner; // double-negation elimination
            } else {
                *node = taken;
            }
        }
        _ => {
            let taken = std::mem::replace(node, BoolExpr::Const(false));
            *node = match taken {
                BoolExpr::And(a, b) => BoolExpr::Or(a, b),
                BoolExpr::Or(a, b) => BoolExpr::Xor(a, b),
                BoolExpr::Xor(a, b) => BoolExpr::And(a, b),
                other => other,
            };
        }
    }
}

/// FOREST-LEVEL crossover: for each shared tree slot a fair coin picks
/// parent `a`'s or parent `b`'s WHOLE tree; `a`'s surplus trees (if any) are
/// kept as-is. Whole trees keep their piece gates intact, so crossover can
/// never manufacture invalid-output trees — the previous subtree-graft
/// operator pulled a donor subtree from under piece j's gate into piece i's
/// gate (P ≈ 6/7 cross-piece), producing invalid votes that the autokill then
/// destroyed together with the whole evolved recipient tree. This operator
/// actually recombines evolved trees, and trivially respects the depth cap
/// (both parents already do).
pub fn crossover(a: &DecisionForest, b: &DecisionForest, rng: &mut SmallRng) -> DecisionForest {
    if a.trees.is_empty() || b.trees.is_empty() {
        return a.clone();
    }
    // Build the child directly: exactly ONE clone per chosen tree. (Cloning
    // all of `a` and then overwriting half the slots with clones of `b`
    // doubled the reproduce cost — clones dominate generation wall-time.)
    let shared = a.trees.len().min(b.trees.len());
    let trees = a
        .trees
        .iter()
        .enumerate()
        .map(|(i, at)| {
            if i < shared && rng.random_bool(0.5) {
                b.trees[i].clone()
            } else {
                at.clone()
            }
        })
        .collect();
    DecisionForest {
        version: FOREST_FORMAT_VERSION,
        trees,
    }
}

/// Tournament selection: sample `k` competitors (by index, no forest clones)
/// and return a reference to the best. `scores[i]` is the fitness of
/// `population[i]`.
pub fn tournament_select<'a>(
    scores: &[f32],
    population: &'a [DecisionForest],
    rng: &mut SmallRng,
    k: usize,
) -> &'a DecisionForest {
    debug_assert!(!population.is_empty() && scores.len() == population.len());
    let k = k.max(1).min(population.len());
    let mut best_idx = rng.random_range(0..population.len());
    for _ in 1..k {
        let idx = rng.random_range(0..population.len());
        if scores[idx] > scores[best_idx] {
            best_idx = idx;
        }
    }
    &population[best_idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::NUM_INPUTS;
    use tetris_game::tetris::TetrisPiecePlacement;

    fn sample_forest(rng: &mut SmallRng) -> DecisionForest {
        DecisionForest::random(rng, 16, 8, 4)
    }

    #[test]
    fn mutation_keeps_forest_valid_and_bounded() {
        let cfg = EvolveConfig::default();
        let n = TetrisPiecePlacement::NUM_PLACEMENTS as u8;
        let mut rng = SmallRng::seed_from_u64(11);
        let parent = sample_forest(&mut rng);
        for _ in 0..100 {
            let child = mutate_forest(&parent, &mut rng, &cfg);
            assert!(!child.trees.is_empty());
            assert!(child.trees.len() <= cfg.max_trees);
            for t in &child.trees {
                if let Some(pid) = t.decide(&[0u8; NUM_INPUTS]) {
                    assert!(pid < n, "pid out of range");
                }
            }
        }
    }

    #[test]
    fn crossover_produces_valid_forest() {
        let mut rng = SmallRng::seed_from_u64(21);
        let a = sample_forest(&mut rng);
        let b = sample_forest(&mut rng);
        let child = crossover(&a, &b, &mut rng);
        assert!(!child.trees.is_empty());
        assert_eq!(child.version, FOREST_FORMAT_VERSION);
        let n = TetrisPiecePlacement::NUM_PLACEMENTS as u8;
        for t in &child.trees {
            if let Some(pid) = t.decide(&[0u8; NUM_INPUTS]) {
                assert!(pid < n);
            }
        }
    }

    #[test]
    fn crossover_takes_whole_trees_from_parents() {
        // Every child tree is byte-identical to the SAME-SLOT tree of one
        // parent — whole trees recombine, subtrees never cross piece gates.
        let mut rng = SmallRng::seed_from_u64(33);
        let a = sample_forest(&mut rng);
        let b = sample_forest(&mut rng);
        let child = crossover(&a, &b, &mut rng);
        assert_eq!(child.trees.len(), a.trees.len());
        let mut from_a = 0;
        let mut from_b = 0;
        for (i, t) in child.trees.iter().enumerate() {
            let s = serde_json::to_string(t).unwrap();
            let sa = serde_json::to_string(&a.trees[i]).unwrap();
            let sb = b
                .trees
                .get(i)
                .map(|bt| serde_json::to_string(bt).unwrap())
                .unwrap_or_default();
            if s == sa {
                from_a += 1;
            } else {
                assert_eq!(s, sb, "child tree {i} matches neither parent slot");
                from_b += 1;
            }
        }
        assert!(
            from_a > 0 && from_b > 0,
            "mix should draw from both parents"
        );
    }

    /// Deepest boolean-condition depth anywhere in a tree.
    fn max_cond_depth_in(tree: &DecisionTree) -> u32 {
        match tree {
            DecisionTree::Leaf(_) | DecisionTree::Abstain => 0,
            DecisionTree::Branch { cond, yes, no } => cond
                .depth()
                .max(max_cond_depth_in(yes))
                .max(max_cond_depth_in(no)),
        }
    }

    #[test]
    fn crossover_respects_the_depth_cap() {
        // Whole-tree mixing can never exceed the deepest parent tree, so the
        // structural cap holds across generations of crossover by induction.
        let cap = PIECE_GATE_DEPTH + 5;
        let mut rng = SmallRng::seed_from_u64(41);
        let mut a = DecisionForest::random(&mut rng, 6, 5, 3);
        let mut b = DecisionForest::random(&mut rng, 6, 5, 3);
        for _ in 0..200 {
            let c = crossover(&a, &b, &mut rng);
            for t in &c.trees {
                assert!(t.depth() <= cap, "crossover exceeded cap: {}", t.depth());
            }
            b = std::mem::replace(&mut a, c);
        }
    }

    #[test]
    fn mutation_respects_the_depth_cap() {
        // Repeated mutation may grow trees, but never past max_tree_depth.
        let cfg = EvolveConfig {
            max_tree_depth: PIECE_GATE_DEPTH + 1,
            grow_prob: 0.9, // growth-heavy regime to stress the cap
            ..EvolveConfig::default()
        };
        let mut rng = SmallRng::seed_from_u64(13);
        let mut forest = DecisionForest::random(&mut rng, 4, 1, 2);
        for _ in 0..300 {
            forest = mutate_forest(&forest, &mut rng, &cfg);
        }
        for t in &forest.trees {
            assert!(t.depth() <= cfg.max_tree_depth, "depth {} > cap", t.depth());
            assert!(
                max_cond_depth_in(t) <= cfg.max_cond_depth,
                "cond depth {} > cap",
                max_cond_depth_in(t)
            );
        }
    }

    /// Every vote a tree emits when piece `i` is current must be a placement
    /// of piece `i` — i.e. the tree is 100% valid in any live game state.
    fn is_fully_valid_gated(tree: &DecisionTree, rng: &mut SmallRng) -> bool {
        use crate::encode::{NUM_CELLS, NUM_PIECES};
        for (i, piece) in TetrisPiece::all().iter().enumerate() {
            for _ in 0..8 {
                let mut bits = [0u8; NUM_INPUTS];
                for b in bits.iter_mut().take(NUM_CELLS) {
                    *b = u8::from(rng.random_bool(0.4));
                }
                bits[NUM_CELLS + NUM_PIECES + i] = 1;
                if let Some(pid) = tree.decide(&bits) {
                    if TetrisPiecePlacement::from_index(pid).piece != *piece {
                        return false;
                    }
                }
            }
        }
        true
    }

    #[test]
    fn mutation_preserves_gated_tree_validity() {
        // The whole point of gate-spine avoidance + governing-piece context:
        // arbitrary chains of structural mutations keep a gated tree 100%
        // valid, so the mutation budget stops feeding the autokill (which was
        // destroying ~a quarter of children's evolved trees every generation).
        let cfg = EvolveConfig::default();
        let mut rng = SmallRng::seed_from_u64(97);
        let mut invalid = 0;
        let mut total = 0;
        for _ in 0..10 {
            let mut forest = DecisionForest::random(&mut rng, 4, 4, 3);
            for _ in 0..60 {
                forest = mutate_forest(&forest, &mut rng, &cfg);
            }
            for t in &forest.trees {
                total += 1;
                if !is_fully_valid_gated(t, &mut rng) {
                    invalid += 1;
                }
            }
        }
        // Rare escapes are tolerated (gate-spine redraw is probabilistic),
        // but validity must be overwhelmingly preserved.
        assert!(
            invalid * 20 <= total,
            "{invalid}/{total} trees invalid after mutation chains"
        );
    }

    #[test]
    fn governing_piece_reads_the_gate_context() {
        use crate::encode::{NUM_CELLS, NUM_PIECES};
        let pieces = TetrisPiece::all();
        // if piece0 { Leaf } else if piece1 { Abstain } else { Leaf }
        let tree = DecisionTree::Branch {
            cond: BoolExpr::Cell(NUM_CELLS + NUM_PIECES),
            yes: Box::new(DecisionTree::Leaf(0)),
            no: Box::new(DecisionTree::Branch {
                cond: BoolExpr::Cell(NUM_CELLS + NUM_PIECES + 1),
                yes: Box::new(DecisionTree::Abstain),
                no: Box::new(DecisionTree::Leaf(1)),
            }),
        };
        // Preorder: 0 = outer gate, 1 = leaf under piece0, 2 = inner gate,
        // 3 = Abstain under piece1, 4 = innermost leaf (no gate above).
        assert_eq!(governing_piece(&tree, 0), None);
        assert_eq!(governing_piece(&tree, 1), Some(pieces[0]));
        assert_eq!(governing_piece(&tree, 2), None);
        assert_eq!(governing_piece(&tree, 3), Some(pieces[1]));
        assert_eq!(governing_piece(&tree, 4), None);
    }

    #[test]
    fn silent_trees_mostly_abstain_but_stay_gated() {
        use crate::encode::{NUM_CELLS, NUM_PIECES};
        let cfg = EvolveConfig::default();
        let mut rng = SmallRng::seed_from_u64(61);
        let mut votes = 0u32;
        let mut total = 0u32;
        for _ in 0..30 {
            let t = silent_gated_tree(&mut rng, &cfg);
            for i in 0..NUM_PIECES {
                let mut bits = [0u8; NUM_INPUTS];
                bits[NUM_CELLS + NUM_PIECES + i] = 1;
                total += 1;
                if t.decide(&bits).is_some() {
                    votes += 1;
                }
            }
        }
        // Mostly silent (abstain-heavy leaves): a near-neutral insertion.
        assert!(
            (votes as f64) < 0.5 * total as f64,
            "silent trees voted {votes}/{total}"
        );
    }

    #[test]
    fn mutation_count_scales_with_forest_size() {
        // A 32-tree forest gets several node mutations per child; with only
        // one, most offspring are exact behavioral clones of their parent.
        let cfg = EvolveConfig::default();
        let mut rng = SmallRng::seed_from_u64(71);
        let parent = DecisionForest::random(&mut rng, 32, 4, 3);
        let mut multi_tree_changes = 0;
        for _ in 0..20 {
            let child = mutate_forest(&parent, &mut rng, &cfg);
            let changed = parent
                .trees
                .iter()
                .zip(child.trees.iter())
                .filter(|(p, c)| {
                    serde_json::to_string(p).unwrap() != serde_json::to_string(c).unwrap()
                })
                .count();
            if changed >= 2 {
                multi_tree_changes += 1;
            }
        }
        assert!(
            multi_tree_changes >= 10,
            "only {multi_tree_changes}/20 children changed >= 2 trees"
        );
    }

    #[test]
    fn fresh_gated_tree_respects_a_tight_depth_cap() {
        let cfg = EvolveConfig {
            max_tree_depth: PIECE_GATE_DEPTH + 2,
            init_tree_depth: 20, // would blow past the cap without the clamp
            ..EvolveConfig::default()
        };
        let mut rng = SmallRng::seed_from_u64(51);
        for _ in 0..20 {
            let t = fresh_gated_tree(&mut rng, &cfg);
            assert!(t.depth() <= cfg.max_tree_depth, "depth {}", t.depth());
        }
    }

    #[test]
    fn tournament_picks_the_best_of_its_sample() {
        let mut rng = SmallRng::seed_from_u64(31);
        let population: Vec<DecisionForest> = (0..8)
            .map(|_| DecisionForest::random(&mut rng, 2, 2, 2))
            .collect();
        let scores: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut rng2 = SmallRng::seed_from_u64(1);
        let winner = tournament_select(&scores, &population, &mut rng2, population.len());
        let bits = [0u8; NUM_INPUTS];
        let legal = [
            TetrisPiecePlacement::from_index(0),
            TetrisPiecePlacement::from_index(1),
        ];
        assert_eq!(
            winner.vote(&bits, &legal),
            population.last().unwrap().vote(&bits, &legal)
        );
    }
}
