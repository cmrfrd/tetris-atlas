use std::collections::VecDeque;

use rayon::prelude::*;
use tracing::debug;

use tetris_game::{TetrisBoard, TetrisGame, TetrisPiece, TetrisPiecePlacement};

use crate::config::{InvariantParams, SimulationConfig};
use crate::encoding::{extract_col_words, find_safe_placements, piece_global_placement_range};
use crate::synthesis::SynthesisExample;
use crate::vm::Program;

/// Number of trace entries to keep before failure.
const TRACE_WINDOW: usize = 10;

/// A single step in a game simulation trace.
#[derive(Debug, Clone)]
pub struct TraceEntry {
    pub board: TetrisBoard,
    pub piece: TetrisPiece,
    pub chosen_placement: u32,
}

/// Result of simulating a single game.
pub enum SimulationResult {
    Survived { pieces_placed: u32 },
    InvalidPlacement { trace: Vec<TraceEntry>, step: u32 },
    Loss { trace: Vec<TraceEntry>, step: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    InvalidPlacement,
    Loss,
}

#[derive(Debug, Clone)]
pub struct FailureSummary {
    pub seed: u32,
    pub kind: FailureKind,
    pub step: u32,
    pub trace: Vec<TraceEntry>,
}

/// Maximum number of failure traces to keep for counterexample extraction.
const MAX_FAILURE_TRACES: usize = 5;

#[derive(Debug, Clone)]
pub struct VerificationSummary {
    pub seeds: u32,
    pub target: u32,
    pub survived: u32,
    pub invalid_placements: u32,
    pub losses: u32,
    pub total_pieces: u64,
    pub min_pieces: u32,
    pub max_pieces: u32,
    pub first_failure: Option<FailureSummary>,
    /// Additional failure traces beyond the first, for diverse counterexamples.
    pub extra_failures: Vec<FailureSummary>,
}

impl VerificationSummary {
    pub fn mean_pieces(&self) -> f64 {
        if self.seeds == 0 {
            return 0.0;
        }
        self.total_pieces as f64 / f64::from(self.seeds)
    }

    pub fn failed(&self) -> u32 {
        self.invalid_placements + self.losses
    }
}

/// Result of verification across many seeds.
pub enum VerificationResult {
    /// All games survived the target number of pieces.
    Survived(VerificationSummary),
    /// At least one game failed; trace from the failure.
    Failure(VerificationSummary),
}

/// Run a single game using the program as the policy.
fn simulate_game(program: &Program, seed: u64, max_pieces: u32) -> SimulationResult {
    let mut game = TetrisGame::new_with_seed(seed);
    let mut trace: VecDeque<TraceEntry> = VecDeque::with_capacity(TRACE_WINDOW + 1);

    for step in 0..max_pieces {
        let col_words = extract_col_words(&game.board);
        let piece_idx = game.current_piece.index();
        let placement_idx = program.execute(&col_words, piece_idx);

        trace.push_back(TraceEntry {
            board: game.board,
            piece: game.current_piece,
            chosen_placement: placement_idx,
        });
        if trace.len() > TRACE_WINDOW {
            trace.pop_front();
        }

        // Validate: is this a legal placement for this piece?
        if placement_idx >= TetrisPiecePlacement::NUM_PLACEMENTS as u32 {
            return SimulationResult::InvalidPlacement {
                trace: trace.into_iter().collect(),
                step,
            };
        }

        let placement_idx = placement_idx as u8;
        let placement_range = piece_global_placement_range(game.current_piece);
        if !placement_range.contains(&placement_idx) {
            return SimulationResult::InvalidPlacement {
                trace: trace.into_iter().collect(),
                step,
            };
        }

        let placement = TetrisPiecePlacement::from_index(placement_idx);
        let result = game.apply_placement(placement);
        if bool::from(result.is_lost) {
            return SimulationResult::Loss {
                trace: trace.into_iter().collect(),
                step,
            };
        }
    }

    SimulationResult::Survived {
        pieces_placed: max_pieces,
    }
}

/// Run the program on many seeds in parallel. Return the first failure found,
/// or `Survived` if all games survive.
pub fn verification_oracle(
    program: &Program,
    config: &SimulationConfig,
    _params: &InvariantParams,
) -> VerificationResult {
    let results: Vec<(u32, SimulationResult)> = (0..config.num_seeds)
        .into_par_iter()
        .map(|seed| {
            (
                seed,
                simulate_game(program, seed as u64, config.survival_target),
            )
        })
        .collect();

    let mut summary = VerificationSummary {
        seeds: config.num_seeds,
        target: config.survival_target,
        survived: 0,
        invalid_placements: 0,
        losses: 0,
        total_pieces: 0,
        min_pieces: config.survival_target,
        max_pieces: 0,
        first_failure: None,
        extra_failures: Vec::new(),
    };

    let mut total_failures_stored = 0usize;

    for (seed, result) in results {
        match result {
            SimulationResult::Survived { pieces_placed } => {
                summary.survived += 1;
                summary.total_pieces += u64::from(pieces_placed);
                summary.min_pieces = summary.min_pieces.min(pieces_placed);
                summary.max_pieces = summary.max_pieces.max(pieces_placed);
            }
            SimulationResult::InvalidPlacement { trace, step } => {
                debug!("seed {seed}: invalid placement at step {step}");
                summary.invalid_placements += 1;
                summary.total_pieces += u64::from(step);
                summary.min_pieces = summary.min_pieces.min(step);
                summary.max_pieces = summary.max_pieces.max(step);
                if total_failures_stored < MAX_FAILURE_TRACES {
                    let fs = FailureSummary {
                        seed,
                        kind: FailureKind::InvalidPlacement,
                        step,
                        trace,
                    };
                    if summary.first_failure.is_none() {
                        summary.first_failure = Some(fs);
                    } else {
                        summary.extra_failures.push(fs);
                    }
                    total_failures_stored += 1;
                }
            }
            SimulationResult::Loss { trace, step } => {
                debug!("seed {seed}: loss at step {step}");
                summary.losses += 1;
                summary.total_pieces += u64::from(step);
                summary.min_pieces = summary.min_pieces.min(step);
                summary.max_pieces = summary.max_pieces.max(step);
                if total_failures_stored < MAX_FAILURE_TRACES {
                    let fs = FailureSummary {
                        seed,
                        kind: FailureKind::Loss,
                        step,
                        trace,
                    };
                    if summary.first_failure.is_none() {
                        summary.first_failure = Some(fs);
                    } else {
                        summary.extra_failures.push(fs);
                    }
                    total_failures_stored += 1;
                }
            }
        }
    }

    if summary.seeds == 0 {
        summary.min_pieces = 0;
    }

    if summary.failed() == 0 {
        VerificationResult::Survived(summary)
    } else {
        VerificationResult::Failure(summary)
    }
}

/// Extract CEGIS counterexamples from a failure trace.
/// For each trace entry where safe placements exist, create a synthesis example.
pub fn extract_counterexamples(
    trace: &[TraceEntry],
    params: &InvariantParams,
) -> Vec<SynthesisExample> {
    let mut examples = Vec::new();
    let mut seen: Vec<([u32; 10], u8)> = Vec::new();

    for entry in trace {
        let col_words = extract_col_words(&entry.board);
        let piece_idx = entry.piece.index();

        // Deduplicate by (col_words, piece)
        let key = (col_words, piece_idx);
        if seen.contains(&key) {
            continue;
        }

        let safe = find_safe_placements(&entry.board, entry.piece, params);
        if !safe.is_empty() {
            seen.push(key);
            examples.push(SynthesisExample {
                col_words,
                piece: piece_idx,
                valid_global_indices: safe,
            });
        }
    }

    examples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Instruction;

    #[test]
    fn test_bad_program_fails_immediately() {
        // A program that always outputs 255 (out of range for any piece)
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 6, // CONST_LO
                dst: 0,
                src1: 255,
                src2: 0,
            }],
        };

        let result = simulate_game(&prog, 42, 100);
        assert!(
            matches!(result, SimulationResult::InvalidPlacement { step: 0, .. }),
            "out-of-range program should fail at step 0"
        );
    }

    #[test]
    fn test_wrong_piece_range_fails_immediately() {
        let seed = 42;
        let game = TetrisGame::new_with_seed(seed);
        let wrong_piece = TetrisPiece::all()
            .into_iter()
            .find(|&piece| piece != game.current_piece)
            .unwrap();
        let wrong_idx = piece_global_placement_range(wrong_piece).start;

        let prog = Program {
            instructions: vec![Instruction {
                opcode: 6,
                dst: 0,
                src1: wrong_idx,
                src2: 0,
            }],
        };

        let result = simulate_game(&prog, seed, 1);
        assert!(
            matches!(result, SimulationResult::InvalidPlacement { step: 0, .. }),
            "wrong-piece placement should fail at step 0"
        );
    }

    #[test]
    fn test_bad_program_verification_fails() {
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 6,
                dst: 0,
                src1: 255,
                src2: 0,
            }],
        };

        let config = SimulationConfig {
            num_seeds: 5,
            survival_target: 100,
        };
        let params = InvariantParams::new(4, 0, 2);

        let result = verification_oracle(&prog, &config, &params);
        assert!(
            matches!(result, VerificationResult::Failure(_)),
            "bad program should fail verification"
        );
    }

    #[test]
    fn test_extract_counterexamples_from_trace() {
        let board = TetrisBoard::new();
        let piece = TetrisPiece::all()[0]; // O piece

        let trace = vec![TraceEntry {
            board,
            piece,
            chosen_placement: 255,
        }];

        let params = InvariantParams::new(4, 2, 4);
        let examples = extract_counterexamples(&trace, &params);

        assert!(!examples.is_empty(), "should extract at least one example");
        let ex = &examples[0];
        assert_eq!(ex.col_words, [0; 10]);
        assert_eq!(ex.piece, piece.index());
        assert!(!ex.valid_global_indices.is_empty());
    }

    #[test]
    fn test_trace_window_bounded() {
        // Even with many steps, trace should be bounded
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 6,
                dst: 0,
                src1: 255, // always invalid
                src2: 0,
            }],
        };

        // This will fail at step 0, but the trace should have at most TRACE_WINDOW entries
        let result = simulate_game(&prog, 0, 1000);
        match result {
            SimulationResult::InvalidPlacement { trace, .. } => {
                assert!(trace.len() <= TRACE_WINDOW);
            }
            other => assert!(
                matches!(other, SimulationResult::InvalidPlacement { .. }),
                "expected InvalidPlacement"
            ),
        }
    }
}
