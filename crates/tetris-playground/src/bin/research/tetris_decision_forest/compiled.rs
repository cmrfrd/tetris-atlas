use tetris_game::tetris::TetrisPiecePlacement;

use crate::forest::DecisionForest;
use crate::tree::{BoolExpr, DecisionTree};

/// Flat boolean-condition node; children are indices into the cond arena.
#[derive(Clone, Copy)]
enum CCond {
    Cell(u16),
    Const(bool),
    Not(u32),
    And(u32, u32),
    Or(u32, u32),
    Xor(u32, u32),
}

/// Flat decision node; `cond` indexes the cond arena, `yes`/`no` the node arena.
#[derive(Clone, Copy)]
enum CNode {
    Leaf(u8),
    Abstain,
    Branch { cond: u32, yes: u32, no: u32 },
}

/// A forest compiled to contiguous arenas for fast evaluation — no `Box`
/// pointer chasing during games. This is a pure REPRESSENTATION change: the
/// genome (boxed trees), mutations, JSON format, and all vote semantics are
/// untouched, and `decide`/`vote` results are bit-identical to the boxed
/// forms (see the equivalence test). Compile once per game (microseconds),
/// evaluate thousands of times.
pub struct CompiledForest {
    conds: Vec<CCond>,
    nodes: Vec<CNode>,
    roots: Vec<u32>,
}

impl CompiledForest {
    /// Flatten every tree of `forest` into the arenas.
    pub fn compile(forest: &DecisionForest) -> Self {
        let mut c = CompiledForest {
            conds: Vec::new(),
            nodes: Vec::new(),
            roots: Vec::with_capacity(forest.trees.len()),
        };
        for tree in &forest.trees {
            let root = c.add_tree(tree);
            c.roots.push(root);
        }
        c
    }

    fn add_cond(&mut self, e: &BoolExpr) -> u32 {
        let node = match e {
            BoolExpr::Cell(i) => CCond::Cell(*i as u16),
            BoolExpr::Const(b) => CCond::Const(*b),
            BoolExpr::Not(a) => {
                let ia = self.add_cond(a);
                CCond::Not(ia)
            }
            BoolExpr::And(a, b) => {
                let ia = self.add_cond(a);
                let ib = self.add_cond(b);
                CCond::And(ia, ib)
            }
            BoolExpr::Or(a, b) => {
                let ia = self.add_cond(a);
                let ib = self.add_cond(b);
                CCond::Or(ia, ib)
            }
            BoolExpr::Xor(a, b) => {
                let ia = self.add_cond(a);
                let ib = self.add_cond(b);
                CCond::Xor(ia, ib)
            }
        };
        self.conds.push(node);
        (self.conds.len() - 1) as u32
    }

    fn add_tree(&mut self, t: &DecisionTree) -> u32 {
        let node = match t {
            DecisionTree::Leaf(p) => CNode::Leaf(*p),
            DecisionTree::Abstain => CNode::Abstain,
            DecisionTree::Branch { cond, yes, no } => {
                let ic = self.add_cond(cond);
                let iy = self.add_tree(yes);
                let inn = self.add_tree(no);
                CNode::Branch {
                    cond: ic,
                    yes: iy,
                    no: inn,
                }
            }
        };
        self.nodes.push(node);
        (self.nodes.len() - 1) as u32
    }

    /// Short-circuiting condition evaluation, identical to `BoolExpr::eval`.
    fn eval_cond(&self, idx: u32, bits: &[u8]) -> bool {
        match self.conds[idx as usize] {
            CCond::Cell(i) => bits.get(i as usize).is_some_and(|&v| v != 0),
            CCond::Const(b) => b,
            CCond::Not(a) => !self.eval_cond(a, bits),
            CCond::And(a, b) => self.eval_cond(a, bits) && self.eval_cond(b, bits),
            CCond::Or(a, b) => self.eval_cond(a, bits) || self.eval_cond(b, bits),
            CCond::Xor(a, b) => self.eval_cond(a, bits) ^ self.eval_cond(b, bits),
        }
    }

    /// Number of trees in the compiled forest.
    pub fn num_trees(&self) -> usize {
        self.roots.len()
    }

    /// Per-tree decide, identical to `DecisionTree::decide`.
    pub fn decide(&self, tree_idx: usize, bits: &[u8]) -> Option<u8> {
        let mut i = self.roots[tree_idx];
        loop {
            match self.nodes[i as usize] {
                CNode::Leaf(p) => return Some(p),
                CNode::Abstain => return None,
                CNode::Branch { cond, yes, no } => {
                    i = if self.eval_cond(cond, bits) { yes } else { no };
                }
            }
        }
    }

    /// Per-placement vote counts, identical to `DecisionForest::tally`.
    pub fn tally(&self, bits: &[u8]) -> [u32; 256] {
        let mut votes = [0u32; 256];
        for t in 0..self.roots.len() {
            if let Some(pid) = self.decide(t, bits) {
                votes[pid as usize] += 1;
            }
        }
        votes
    }

    /// The forest's move, identical to `DecisionForest::vote`.
    pub fn vote(
        &self,
        bits: &[u8],
        legal: &[TetrisPiecePlacement],
    ) -> Option<TetrisPiecePlacement> {
        if legal.is_empty() {
            return None;
        }
        DecisionForest::winner_from_tally(&self.tally(bits), legal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::NUM_INPUTS;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn compiled_forest_matches_boxed_semantics_exactly() {
        // Property: per-tree decide and the forest vote are bit-identical
        // between the boxed genome and the compiled arenas, across random
        // forests and random input patterns.
        let mut rng = SmallRng::seed_from_u64(17);
        for _ in 0..8 {
            let forest = DecisionForest::random(&mut rng, 12, 6, 4);
            let compiled = CompiledForest::compile(&forest);
            assert_eq!(compiled.num_trees(), forest.trees.len());
            for _ in 0..16 {
                let mut bits = [0u8; NUM_INPUTS];
                for b in bits.iter_mut() {
                    *b = u8::from(rng.random_bool(0.3));
                }
                for (i, tree) in forest.trees.iter().enumerate() {
                    assert_eq!(compiled.decide(i, &bits), tree.decide(&bits));
                }
                let legal: Vec<_> = (0u8..17).map(TetrisPiecePlacement::from_index).collect();
                assert_eq!(compiled.vote(&bits, &legal), forest.vote(&bits, &legal));
            }
        }
    }
}
