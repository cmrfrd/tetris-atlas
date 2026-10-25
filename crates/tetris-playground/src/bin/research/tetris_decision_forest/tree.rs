use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

use tetris_game::tetris::{TetrisPiece, TetrisPiecePlacement};

use crate::encode::{NUM_CELLS, NUM_INPUTS, NUM_PIECES};

/// A boolean operation over the 214 input bits — the condition at a decision
/// node. Leaves are an input bit (`Cell`) or a constant (`Const`); internal
/// nodes are boolean gates. Evaluates to a single `bool`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BoolExpr {
    Cell(usize),
    Const(bool),
    Not(Box<BoolExpr>),
    And(Box<BoolExpr>, Box<BoolExpr>),
    Or(Box<BoolExpr>, Box<BoolExpr>),
    Xor(Box<BoolExpr>, Box<BoolExpr>),
}

/// A random `BoolExpr` leaf: usually an input bit, occasionally a constant.
pub fn random_boolexpr_leaf(rng: &mut SmallRng) -> BoolExpr {
    if rng.random_bool(0.9) {
        BoolExpr::Cell(rng.random_range(0..NUM_INPUTS))
    } else {
        BoolExpr::Const(rng.random_bool(0.5))
    }
}

impl BoolExpr {
    /// Evaluate on the input bits (out-of-range index reads as `false`).
    pub fn eval(&self, bits: &[u8]) -> bool {
        match self {
            BoolExpr::Cell(i) => bits.get(*i).is_some_and(|&v| v != 0),
            BoolExpr::Const(b) => *b,
            BoolExpr::Not(a) => !a.eval(bits),
            BoolExpr::And(a, b) => a.eval(bits) && b.eval(bits),
            BoolExpr::Or(a, b) => a.eval(bits) || b.eval(bits),
            BoolExpr::Xor(a, b) => a.eval(bits) ^ b.eval(bits),
        }
    }

    /// A random boolean condition of depth at most `max_depth`.
    pub fn random(rng: &mut SmallRng, max_depth: u32) -> Self {
        if max_depth == 0 || rng.random_bool(0.3) {
            return random_boolexpr_leaf(rng);
        }
        match rng.random_range(0..4) {
            0 => BoolExpr::Not(Box::new(BoolExpr::random(rng, max_depth - 1))),
            1 => BoolExpr::And(
                Box::new(BoolExpr::random(rng, max_depth - 1)),
                Box::new(BoolExpr::random(rng, max_depth - 1)),
            ),
            2 => BoolExpr::Or(
                Box::new(BoolExpr::random(rng, max_depth - 1)),
                Box::new(BoolExpr::random(rng, max_depth - 1)),
            ),
            _ => BoolExpr::Xor(
                Box::new(BoolExpr::random(rng, max_depth - 1)),
                Box::new(BoolExpr::random(rng, max_depth - 1)),
            ),
        }
    }

    pub fn count_nodes(&self) -> usize {
        match self {
            BoolExpr::Cell(_) | BoolExpr::Const(_) => 1,
            BoolExpr::Not(a) => 1 + a.count_nodes(),
            BoolExpr::And(a, b) | BoolExpr::Or(a, b) | BoolExpr::Xor(a, b) => {
                1 + a.count_nodes() + b.count_nodes()
            }
        }
    }

    pub fn depth(&self) -> u32 {
        match self {
            BoolExpr::Cell(_) | BoolExpr::Const(_) => 0,
            BoolExpr::Not(a) => 1 + a.depth(),
            BoolExpr::And(a, b) | BoolExpr::Or(a, b) | BoolExpr::Xor(a, b) => {
                1 + a.depth().max(b.depth())
            }
        }
    }

    pub fn node_at_mut(&mut self, target: usize) -> Option<&mut BoolExpr> {
        let mut cursor = 0usize;
        self.node_at_mut_inner(target, &mut cursor)
    }

    fn node_at_mut_inner(&mut self, target: usize, cursor: &mut usize) -> Option<&mut BoolExpr> {
        if *cursor == target {
            return Some(self);
        }
        *cursor += 1;
        match self {
            BoolExpr::Cell(_) | BoolExpr::Const(_) => None,
            BoolExpr::Not(a) => a.node_at_mut_inner(target, cursor),
            BoolExpr::And(a, b) | BoolExpr::Or(a, b) | BoolExpr::Xor(a, b) => {
                if let Some(found) = a.node_at_mut_inner(target, cursor) {
                    return Some(found);
                }
                b.node_at_mut_inner(target, cursor)
            }
        }
    }
}

/// A random valid placement id (index into `TetrisPiecePlacement::from_index`).
pub fn random_pid(rng: &mut SmallRng) -> u8 {
    rng.random_range(0..TetrisPiecePlacement::NUM_PLACEMENTS) as u8
}

/// Probability that a randomly generated tree leaf is an `Abstain` (a
/// "no recommendation" output) rather than a placement vote.
pub const ABSTAIN_LEAF_PROB: f64 = 0.15;

/// A random tree leaf: usually a placement vote, sometimes an abstention.
pub fn random_tree_leaf(rng: &mut SmallRng) -> DecisionTree {
    if rng.random_bool(ABSTAIN_LEAF_PROB) {
        DecisionTree::Abstain
    } else {
        DecisionTree::Leaf(random_pid(rng))
    }
}

/// A random placement id restricted to `piece`'s own placement table.
pub fn random_pid_of_piece(rng: &mut SmallRng, piece: TetrisPiece) -> u8 {
    let all = TetrisPiecePlacement::all_from_piece(piece);
    all[rng.random_range(0..all.len())].index()
}

/// Abstain probability for SILENT-SPECIALIST trees (regrowth injections):
/// mostly silent, so adding one barely perturbs the plurality — a
/// near-neutral insertion that can then evolve votes where it helps.
pub const SILENT_TREE_ABSTAIN_PROB: f64 = 0.75;

/// A random tree leaf whose vote (if any) is a placement of `piece`.
fn random_leaf_for_piece(
    rng: &mut SmallRng,
    piece: TetrisPiece,
    abstain_prob: f64,
) -> DecisionTree {
    if rng.random_bool(abstain_prob) {
        DecisionTree::Abstain
    } else {
        DecisionTree::Leaf(random_pid_of_piece(rng, piece))
    }
}

/// A random board-conditioned subtree ALL of whose votes are placements of
/// `piece` — the body under one arm of the piece gate.
fn random_subtree_for_piece(
    rng: &mut SmallRng,
    piece: TetrisPiece,
    max_depth: u32,
    cond_depth: u32,
    abstain_prob: f64,
) -> DecisionTree {
    if max_depth == 0 || rng.random_bool(0.25) {
        return random_leaf_for_piece(rng, piece, abstain_prob);
    }
    DecisionTree::Branch {
        cond: BoolExpr::random(rng, cond_depth),
        yes: Box::new(random_subtree_for_piece(
            rng,
            piece,
            max_depth - 1,
            cond_depth,
            abstain_prob,
        )),
        no: Box::new(random_subtree_for_piece(
            rng,
            piece,
            max_depth - 1,
            cond_depth,
            abstain_prob,
        )),
    }
}

/// Depth contributed by the piece-gate chain of `random_piece_gated_tree`:
/// one branch level per piece.
pub const PIECE_GATE_DEPTH: u32 = NUM_PIECES as u32;

/// Depth used when GENERATING random boolean conditions, regardless of the
/// structural cap (`max_cond_depth`, which growth mutations may still climb
/// to). `BoolExpr::random` is a supercritical branching process (~1.225 mean
/// children), so generating at depth 16 makes ~50-130-node formulas: every
/// clone, cull walk, and condition eval then pays for random structure that
/// was never selected for. Like `init_tree_depth`, condition depth is earned
/// through evolution, not granted at birth.
pub const COND_GEN_DEPTH_CAP: u32 = 4;

/// If `expr` is a bare test of a current-piece one-hot bit, which piece.
/// These are the GATE conditions of a piece-gated tree.
pub fn piece_bit_piece(expr: &BoolExpr) -> Option<TetrisPiece> {
    if let BoolExpr::Cell(i) = expr {
        let base = NUM_CELLS + NUM_PIECES;
        if (base..base + NUM_PIECES).contains(i) {
            return Some(TetrisPiece::all()[*i - base]);
        }
    }
    None
}

/// The PIECE CONTEXT governing preorder node `target`: the piece of the
/// innermost piece-gate branch whose `yes` side contains the target. `None`
/// for nodes on the gate spine itself (or trees without gates). Mutations use
/// this to create votes that are valid where the mutated node actually runs.
pub fn governing_piece(tree: &DecisionTree, target: usize) -> Option<TetrisPiece> {
    fn walk(
        node: &DecisionTree,
        target: usize,
        cursor: &mut usize,
        ctx: Option<TetrisPiece>,
    ) -> Option<Option<TetrisPiece>> {
        if *cursor == target {
            return Some(ctx);
        }
        *cursor += 1;
        match node {
            DecisionTree::Leaf(_) | DecisionTree::Abstain => None,
            DecisionTree::Branch { cond, yes, no } => {
                let yes_ctx = piece_bit_piece(cond).or(ctx);
                if let Some(found) = walk(yes, target, cursor, yes_ctx) {
                    return Some(found);
                }
                walk(no, target, cursor, ctx)
            }
        }
    }
    let mut cursor = 0usize;
    walk(tree, target, &mut cursor, None).unwrap_or(None)
}

/// A random tree leaf honoring a piece context: votes (if any) are placements
/// of `ctx` when known, so growing/reviving leaves inside a gated subtree
/// cannot manufacture invalid outputs.
pub fn random_tree_leaf_for(rng: &mut SmallRng, ctx: Option<TetrisPiece>) -> DecisionTree {
    if rng.random_bool(ABSTAIN_LEAF_PROB) {
        DecisionTree::Abstain
    } else {
        match ctx {
            Some(piece) => DecisionTree::Leaf(random_pid_of_piece(rng, piece)),
            None => DecisionTree::Leaf(random_pid(rng)),
        }
    }
}

/// A random PIECE-GATED tree: the root chain tests the 7 current-piece
/// one-hot bits; under the gate for piece `i` sits a random board-conditioned
/// subtree whose votes are all placements OF THAT PIECE. The engine's legal
/// set is exactly the current piece's placement table, so every vote such a
/// tree can emit is legal — the tree is 100% valid BY CONSTRUCTION and
/// survives the cull-at-1.0 autokill while staying board-sensitive. This is
/// the generator for all new trees (population init, tree-add mutation,
/// regrowth); mutations may later break the gate, and the cull polices that.
/// Per-piece subtree depth is drawn from `0..=sub_depth` for size and
/// diversity; total depth is at most `PIECE_GATE_DEPTH + sub_depth`.
pub fn random_piece_gated_tree(
    rng: &mut SmallRng,
    sub_depth: u32,
    cond_depth: u32,
) -> DecisionTree {
    random_piece_gated_tree_with_abstain(rng, sub_depth, cond_depth, ABSTAIN_LEAF_PROB)
}

/// `random_piece_gated_tree` with an explicit leaf-abstain probability —
/// `SILENT_TREE_ABSTAIN_PROB` yields silent-specialist trees for regrowth.
pub fn random_piece_gated_tree_with_abstain(
    rng: &mut SmallRng,
    sub_depth: u32,
    cond_depth: u32,
    abstain_prob: f64,
) -> DecisionTree {
    // Innermost fallback is unreachable in a live game (exactly one piece
    // bit is set), and abstention is the safe answer if it ever is reached.
    let mut tree = DecisionTree::Abstain;
    for (i, piece) in TetrisPiece::all().iter().enumerate().rev() {
        let d = rng.random_range(0..=sub_depth);
        tree = DecisionTree::Branch {
            cond: BoolExpr::Cell(NUM_CELLS + NUM_PIECES + i),
            yes: Box::new(random_subtree_for_piece(
                rng,
                *piece,
                d,
                cond_depth,
                abstain_prob,
            )),
            no: Box::new(tree),
        };
    }
    tree
}

/// A decision tree: internal `Branch` nodes route on a boolean condition;
/// leaves either hold a placement id (`Leaf`) or explicitly decline to vote
/// (`Abstain` — "no recommendation"). `decide` routes the input bits to a
/// leaf: `Some(pid)` is the tree's vote, `None` is silence. Abstention is what
/// lets a tree SPECIALIZE: vote only on the boards its conditions recognize,
/// stay silent everywhere else.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DecisionTree {
    Leaf(u8),
    Abstain,
    Branch {
        cond: BoolExpr,
        yes: Box<DecisionTree>,
        no: Box<DecisionTree>,
    },
}

impl DecisionTree {
    /// Route `bits` to a leaf: `Some(placement id)` or `None` (abstain).
    /// Iterative so deep trees do not risk stack overflow.
    pub fn decide(&self, bits: &[u8]) -> Option<u8> {
        let mut node = self;
        loop {
            match node {
                DecisionTree::Leaf(p) => return Some(*p),
                DecisionTree::Abstain => return None,
                DecisionTree::Branch { cond, yes, no } => {
                    node = if cond.eval(bits) { yes } else { no };
                }
            }
        }
    }

    pub fn random(rng: &mut SmallRng, max_depth: u32, cond_depth: u32) -> Self {
        if max_depth == 0 || rng.random_bool(0.25) {
            return random_tree_leaf(rng);
        }
        DecisionTree::Branch {
            cond: BoolExpr::random(rng, cond_depth),
            yes: Box::new(DecisionTree::random(rng, max_depth - 1, cond_depth)),
            no: Box::new(DecisionTree::random(rng, max_depth - 1, cond_depth)),
        }
    }

    pub fn count_nodes(&self) -> usize {
        match self {
            DecisionTree::Leaf(_) | DecisionTree::Abstain => 1,
            DecisionTree::Branch { yes, no, .. } => 1 + yes.count_nodes() + no.count_nodes(),
        }
    }

    pub fn depth(&self) -> u32 {
        match self {
            DecisionTree::Leaf(_) | DecisionTree::Abstain => 0,
            DecisionTree::Branch { yes, no, .. } => 1 + yes.depth().max(no.depth()),
        }
    }

    pub fn node_at_mut(&mut self, target: usize) -> Option<&mut DecisionTree> {
        let mut cursor = 0usize;
        self.node_at_mut_inner(target, &mut cursor)
    }

    fn node_at_mut_inner(
        &mut self,
        target: usize,
        cursor: &mut usize,
    ) -> Option<&mut DecisionTree> {
        if *cursor == target {
            return Some(self);
        }
        *cursor += 1;
        match self {
            DecisionTree::Leaf(_) | DecisionTree::Abstain => None,
            DecisionTree::Branch { yes, no, .. } => {
                if let Some(found) = yes.node_at_mut_inner(target, cursor) {
                    return Some(found);
                }
                no.node_at_mut_inner(target, cursor)
            }
        }
    }

    /// Whether this tree's output is VALID for the current piece: either it
    /// abstains (silence is never invalid) or its placement id is among the
    /// legal placements. Used by the training-time autokill — abstention is a
    /// sanctioned "no recommendation", not an invalid output.
    pub fn output_is_valid(&self, bits: &[u8], legal: &[TetrisPiecePlacement]) -> bool {
        match self.decide(bits) {
            None => true,
            Some(pid) => legal.iter().any(|p| p.index() == pid),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zeros() -> [u8; NUM_INPUTS] {
        [0u8; NUM_INPUTS]
    }

    #[test]
    fn boolexpr_gates_evaluate() {
        let mut c = zeros();
        c[0] = 1; // bit 0 true, bit 1 false
        let and = BoolExpr::And(Box::new(BoolExpr::Cell(0)), Box::new(BoolExpr::Cell(1)));
        let or = BoolExpr::Or(Box::new(BoolExpr::Cell(0)), Box::new(BoolExpr::Cell(1)));
        let xor = BoolExpr::Xor(Box::new(BoolExpr::Cell(0)), Box::new(BoolExpr::Cell(1)));
        let not = BoolExpr::Not(Box::new(BoolExpr::Cell(1)));
        assert!(!and.eval(&c));
        assert!(or.eval(&c));
        assert!(xor.eval(&c));
        assert!(not.eval(&c));
        assert_eq!(and.count_nodes(), 3);
        assert_eq!(and.depth(), 1);
    }

    #[test]
    fn decision_tree_routes_to_leaf_placement() {
        // if bit 0 set -> Leaf(5) else Leaf(9)
        let tree = DecisionTree::Branch {
            cond: BoolExpr::Cell(0),
            yes: Box::new(DecisionTree::Leaf(5)),
            no: Box::new(DecisionTree::Leaf(9)),
        };
        assert_eq!(tree.decide(&zeros()), Some(9));
        let mut on = zeros();
        on[0] = 1;
        assert_eq!(tree.decide(&on), Some(5));
        assert_eq!(tree.count_nodes(), 3);
        assert_eq!(tree.depth(), 1);
    }

    #[test]
    fn random_tree_is_bounded_and_decides_a_valid_pid() {
        let n = TetrisPiecePlacement::NUM_PLACEMENTS as u8;
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..50 {
            let t = DecisionTree::random(&mut rng, 6, 4);
            assert!(t.depth() <= 6, "depth {}", t.depth());
            if let Some(pid) = t.decide(&zeros()) {
                assert!(pid < n, "pid out of range");
            } // None = abstain, always fine
        }
    }

    #[test]
    fn abstain_leaf_decides_none_and_is_always_valid() {
        let tree = DecisionTree::Abstain;
        assert_eq!(tree.decide(&zeros()), None);
        // Abstention is never an invalid output, whatever the legal set.
        let legal = [TetrisPiecePlacement::from_index(3)];
        assert!(tree.output_is_valid(&zeros(), &legal));
        assert!(tree.output_is_valid(&zeros(), &[]));
        // A branch can route to abstention on one side and a vote on the other.
        let routed = DecisionTree::Branch {
            cond: BoolExpr::Cell(0),
            yes: Box::new(DecisionTree::Abstain),
            no: Box::new(DecisionTree::Leaf(7)),
        };
        assert_eq!(routed.decide(&zeros()), Some(7));
        let mut on = zeros();
        on[0] = 1;
        assert_eq!(routed.decide(&on), None);
    }

    #[test]
    fn random_pid_of_piece_stays_within_the_piece() {
        let mut rng = SmallRng::seed_from_u64(29);
        for piece in TetrisPiece::all() {
            for _ in 0..20 {
                let pid = random_pid_of_piece(&mut rng, piece);
                assert_eq!(TetrisPiecePlacement::from_index(pid).piece, piece);
            }
        }
    }

    #[test]
    fn piece_gated_tree_votes_only_placements_of_the_current_piece() {
        // For every current piece (one-hot bit set) and random board bits, a
        // gated tree either abstains or votes a placement OF THAT PIECE —
        // i.e. it is 100% valid by construction.
        let mut rng = SmallRng::seed_from_u64(23);
        for _ in 0..40 {
            let tree = random_piece_gated_tree(&mut rng, 4, 3);
            assert!(
                tree.depth() <= PIECE_GATE_DEPTH + 4,
                "depth {}",
                tree.depth()
            );
            for (i, piece) in TetrisPiece::all().iter().enumerate() {
                let mut bits = [0u8; NUM_INPUTS];
                for b in bits.iter_mut().take(NUM_CELLS) {
                    *b = u8::from(rng.random_bool(0.4));
                }
                bits[NUM_CELLS + NUM_PIECES + i] = 1;
                if let Some(pid) = tree.decide(&bits) {
                    assert_eq!(
                        TetrisPiecePlacement::from_index(pid).piece,
                        *piece,
                        "gated tree voted another piece's placement"
                    );
                }
            }
        }
    }

    #[test]
    fn output_validity_matches_legal_pids() {
        let cells = zeros();
        let tree = DecisionTree::Leaf(0);
        let legal_has0 = [
            TetrisPiecePlacement::from_index(0),
            TetrisPiecePlacement::from_index(1),
        ];
        assert!(tree.output_is_valid(&cells, &legal_has0)); // pid 0 in legal
        let legal_no0 = [
            TetrisPiecePlacement::from_index(3),
            TetrisPiecePlacement::from_index(5),
        ];
        assert!(!tree.output_is_valid(&cells, &legal_no0)); // pid 0 not in legal
    }

    #[test]
    fn placement_id_round_trips_over_the_full_range() {
        // The pid scheme (u8 index <-> placement) is the contract between tree
        // leaves and the engine; protect it across the whole range.
        for i in 0..TetrisPiecePlacement::NUM_PLACEMENTS as u8 {
            assert_eq!(TetrisPiecePlacement::from_index(i).index(), i);
        }
    }

    #[test]
    fn node_at_mut_addresses_every_node() {
        let mut tree = DecisionTree::Branch {
            cond: BoolExpr::Cell(0),
            yes: Box::new(DecisionTree::Leaf(1)),
            no: Box::new(DecisionTree::Leaf(2)),
        };
        let n = tree.count_nodes();
        assert_eq!(n, 3);
        for i in 0..n {
            assert!(tree.node_at_mut(i).is_some());
        }
        assert!(tree.node_at_mut(n).is_none());

        let mut e = BoolExpr::And(Box::new(BoolExpr::Cell(0)), Box::new(BoolExpr::Const(true)));
        let m = e.count_nodes();
        for i in 0..m {
            assert!(e.node_at_mut(i).is_some());
        }
        assert!(e.node_at_mut(m).is_none());
    }
}
