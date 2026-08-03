use std::path::Path;

use anyhow::Result;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};

use tetris_game::tetris::TetrisPiecePlacement;

use crate::encode::NUM_INPUTS;
use crate::tree::{DecisionTree, random_piece_gated_tree};

/// On-disk format version. Bump when the input encoding, node encoding, or
/// placement-id scheme changes so old checkpoints are rejected.
/// v2: `DecisionTree` gained the `Abstain` variant.
pub const FOREST_FORMAT_VERSION: u32 = 2;

/// A decision forest: each tree votes one placement id per input; the forest
/// plays the legal placement with the most votes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionForest {
    pub version: u32,
    pub trees: Vec<DecisionTree>,
}

impl DecisionForest {
    /// Tally every tree's placement vote and return the legal placement with the
    /// most votes. A vote for a pid that is not legal for the current piece
    /// contributes to that pid's tally but can never win (only legal placements
    /// are considered), so an invalid placement is never played. Ties (including
    /// the all-zero-legal-vote case — every tree abstained or voted illegally)
    /// break by lowest placement index.
    ///
    /// TOTALITY: returns `None` ONLY when `legal` is empty, which cannot happen
    /// in a live game (the engine's per-piece placement table is static and
    /// non-empty) — the forest always answers.
    pub fn vote(
        &self,
        bits: &[u8],
        legal: &[TetrisPiecePlacement],
    ) -> Option<TetrisPiecePlacement> {
        if legal.is_empty() {
            return None;
        }
        Self::winner_from_tally(&self.tally(bits), legal)
    }

    /// Per-placement vote counts across all trees. Abstaining trees
    /// (`decide` = `None`) contribute nothing. Placement ids are `u8`
    /// (< NUM_PLACEMENTS ≤ 162); a 256-wide tally covers the whole range
    /// without a const dependency.
    pub fn tally(&self, bits: &[u8]) -> [u32; 256] {
        let mut votes = [0u32; 256];
        for tree in &self.trees {
            if let Some(pid) = tree.decide(bits) {
                votes[pid as usize] += 1;
            }
        }
        votes
    }

    /// The most-voted legal placement from a tally (ties break by lowest
    /// placement index). The single source of truth for winner selection —
    /// used by `vote` and by callers that build the tally themselves.
    pub fn winner_from_tally(
        votes: &[u32; 256],
        legal: &[TetrisPiecePlacement],
    ) -> Option<TetrisPiecePlacement> {
        legal
            .iter()
            .copied()
            .max_by_key(|p| (votes[p.index() as usize], std::cmp::Reverse(p.index())))
    }

    /// A random forest of `num_trees` PIECE-GATED decision trees (see
    /// `random_piece_gated_tree`): every tree is 100% valid by construction,
    /// so a fresh forest survives the aggressive autokill and votes real
    /// placements from generation zero. `sub_depth` bounds the random
    /// board-logic depth under each piece gate (tree depth is at most
    /// `PIECE_GATE_DEPTH + sub_depth`).
    pub fn random(rng: &mut SmallRng, num_trees: usize, sub_depth: u32, cond_depth: u32) -> Self {
        let trees = (0..num_trees)
            .map(|_| random_piece_gated_tree(rng, sub_depth, cond_depth))
            .collect();
        DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees,
        }
    }

    /// Total decision-node count across all trees (a complexity measure).
    pub fn total_nodes(&self) -> usize {
        self.trees.iter().map(|t| t.count_nodes()).sum()
    }

    /// Write the forest as JSON ATOMICALLY (temp file + rename): the champion
    /// is rewritten frequently during training, and a concurrent reader (an
    /// `eval`/`distill` of the live champion, or a human copying it) must
    /// never observe a torn, half-written file.
    pub fn save_json(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("json.tmp");
        let file = std::fs::File::create(&tmp)?;
        serde_json::to_writer_pretty(file, self)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load_json(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let forest: DecisionForest = serde_json::from_reader(file)?;
        anyhow::ensure!(
            forest.version == FOREST_FORMAT_VERSION,
            "forest format version {} != expected {}",
            forest.version,
            FOREST_FORMAT_VERSION
        );
        Ok(forest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vote_picks_the_legal_placement_with_the_most_votes() {
        let forest = DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![
                DecisionTree::Leaf(0),
                DecisionTree::Leaf(0),
                DecisionTree::Leaf(1),
            ],
        };
        let bits = [0u8; NUM_INPUTS];
        let legal = [
            TetrisPiecePlacement::from_index(0),
            TetrisPiecePlacement::from_index(1),
        ];
        assert_eq!(
            forest.vote(&bits, &legal).unwrap(),
            TetrisPiecePlacement::from_index(0)
        );
    }

    #[test]
    fn invalid_votes_cannot_win_and_fallback_is_lowest_index() {
        // The only tree votes pid 0, which is NOT in the legal set.
        let forest = DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![DecisionTree::Leaf(0)],
        };
        let bits = [0u8; NUM_INPUTS];
        let legal = [
            TetrisPiecePlacement::from_index(5),
            TetrisPiecePlacement::from_index(3),
        ];
        // Invalid vote discarded -> no legal placement got a vote -> lowest index (3).
        assert_eq!(
            forest.vote(&bits, &legal).unwrap(),
            TetrisPiecePlacement::from_index(3)
        );
    }

    #[test]
    fn abstaining_trees_do_not_block_the_vote() {
        // Two abstainers + one voter: the voter decides the move.
        let forest = DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![
                DecisionTree::Abstain,
                DecisionTree::Abstain,
                DecisionTree::Leaf(1),
            ],
        };
        let bits = [0u8; NUM_INPUTS];
        let legal = [
            TetrisPiecePlacement::from_index(0),
            TetrisPiecePlacement::from_index(1),
        ];
        assert_eq!(
            forest.vote(&bits, &legal).unwrap(),
            TetrisPiecePlacement::from_index(1)
        );
        // All-abstain forest: fallback still plays the lowest-index legal move.
        let silent = DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![DecisionTree::Abstain, DecisionTree::Abstain],
        };
        assert_eq!(
            silent.vote(&bits, &legal).unwrap(),
            TetrisPiecePlacement::from_index(0)
        );
    }

    #[test]
    fn vote_returns_none_when_no_legal_moves() {
        let forest = DecisionForest {
            version: FOREST_FORMAT_VERSION,
            trees: vec![DecisionTree::Leaf(0)],
        };
        assert!(forest.vote(&[0u8; NUM_INPUTS], &[]).is_none());
    }

    #[test]
    fn vote_always_returns_a_legal_placement() {
        let mut rng = SmallRng::seed_from_u64(2);
        let forest = DecisionForest::random(&mut rng, 8, 6, 3);
        let bits = [0u8; NUM_INPUTS];
        let legal: Vec<_> = (0u8..9).map(TetrisPiecePlacement::from_index).collect();
        let chosen = forest.vote(&bits, &legal).unwrap();
        assert!(legal.contains(&chosen));
    }

    #[test]
    fn load_json_rejects_wrong_version() {
        let mut rng = SmallRng::seed_from_u64(5);
        let forest = DecisionForest::random(&mut rng, 2, 2, 2);
        let path = std::env::temp_dir().join("df_version_reject.json");
        forest.save_json(&path).unwrap();
        let txt = std::fs::read_to_string(&path).unwrap();
        let doctored = txt.replace(
            &format!("\"version\": {FOREST_FORMAT_VERSION}"),
            "\"version\": 999",
        );
        assert_ne!(txt, doctored, "version field not found to doctor");
        std::fs::write(&path, doctored).unwrap();
        assert!(DecisionForest::load_json(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn random_forest_and_json_round_trip() {
        let mut rng = SmallRng::seed_from_u64(9);
        let forest = DecisionForest::random(&mut rng, 8, 5, 3);
        assert_eq!(forest.trees.len(), 8);
        assert!(forest.total_nodes() >= 8);
        let path = std::env::temp_dir().join("df_forest_roundtrip.json");
        forest.save_json(&path).unwrap();
        let loaded = DecisionForest::load_json(&path).unwrap();
        let bits = [0u8; NUM_INPUTS];
        let legal = [
            TetrisPiecePlacement::from_index(0),
            TetrisPiecePlacement::from_index(1),
        ];
        assert_eq!(forest.vote(&bits, &legal), loaded.vote(&bits, &legal));
        let _ = std::fs::remove_file(&path);
    }
}
