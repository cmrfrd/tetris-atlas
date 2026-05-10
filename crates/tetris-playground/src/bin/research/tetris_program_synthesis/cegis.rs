use std::collections::HashSet;

use tracing::{info, warn};

use tetris_game::{TetrisBoard, TetrisPiece};

use crate::config::{InvariantParams, SimulationConfig};
use crate::encoding::{
    board_from_heights, board_satisfies_invariant_with_params, extract_col_words,
    find_safe_placements,
};
use crate::synthesis::{
    SynthesisCheckResult, SynthesisExample, SynthesisOptions, SynthesisSession,
};
use crate::verification::{
    FailureKind, TraceEntry, VerificationResult, VerificationSummary, extract_counterexamples,
    verification_oracle,
};
use crate::vm::Program;

/// Result of unified empirical CEGIS.
#[derive(Debug)]
pub enum CegisResult {
    /// Found a program that survived the configured finite simulation suite.
    Survived {
        program: Program,
        rounds: usize,
        examples: usize,
    },
    /// Synthesis failed for the accumulated examples.
    SynthesisFailed { round: usize, examples: usize },
    /// Synthesis timed out or otherwise returned unknown.
    SynthesisUnknown {
        round: usize,
        examples: usize,
        reason: Option<String>,
    },
    /// Verification failed, but the trace yielded no invariant-safe training examples.
    NoCounterexamples {
        round: usize,
        examples: usize,
        last_program: Program,
    },
    /// Verification failed, but all extracted examples were already known.
    NoNewCounterexamples {
        round: usize,
        examples: usize,
        last_program: Program,
    },
    /// Exceeded max rounds without finding a survivor.
    MaxRoundsExceeded {
        last_program: Option<Program>,
        examples: usize,
    },
}

fn push_example_for_board(
    examples: &mut Vec<SynthesisExample>,
    seen: &mut HashSet<([u32; 10], u8)>,
    board: &TetrisBoard,
    piece: TetrisPiece,
    params: &InvariantParams,
) {
    let col_words = extract_col_words(board);
    let piece_idx = piece.index();
    if !seen.insert((col_words, piece_idx)) {
        return;
    }

    let safe = find_safe_placements(board, piece, params);
    if safe.is_empty() {
        return;
    }

    examples.push(SynthesisExample {
        col_words,
        piece: piece_idx,
        valid_global_indices: safe,
    });
}

/// Seed initial examples: empty board with the first piece that has valid placements.
/// Starting with 1 example keeps the first synthesis call fast (~seconds).
/// CEGIS will add harder examples from simulation failures incrementally.
fn seed_examples(params: &InvariantParams) -> Vec<SynthesisExample> {
    let mut examples = Vec::new();
    let mut seen = HashSet::new();

    let board = TetrisBoard::new();
    for piece in TetrisPiece::all() {
        push_example_for_board(&mut examples, &mut seen, &board, piece, params);
        if !examples.is_empty() {
            break;
        }
    }

    examples
}

fn log_verification_summary(round: usize, summary: &VerificationSummary) {
    info!(
        "round {round}: verification summary: seeds={}, target={}, survived={}, failed={}, invalid={}, losses={}, pieces_mean={:.2}, pieces_min={}, pieces_max={}",
        summary.seeds,
        summary.target,
        summary.survived,
        summary.failed(),
        summary.invalid_placements,
        summary.losses,
        summary.mean_pieces(),
        summary.min_pieces,
        summary.max_pieces
    );

    if let Some(failure) = &summary.first_failure {
        let kind = match failure.kind {
            FailureKind::InvalidPlacement => "invalid-placement",
            FailureKind::Loss => "loss",
        };
        info!(
            "round {round}: first failure: seed={}, kind={}, step={}, trace_len={}",
            failure.seed,
            kind,
            failure.step,
            failure.trace.len()
        );
    }
}

/// Run CEGIS for one global policy program.
pub fn run_cegis(
    params: &InvariantParams,
    simulation: &SimulationConfig,
    synthesis_options: &SynthesisOptions,
    max_rounds: usize,
    program_length: usize,
) -> CegisResult {
    info!(
        "unified CEGIS: prog_len={}, max_rounds={}, seeds={}, survival_target={}",
        program_length, max_rounds, simulation.num_seeds, simulation.survival_target
    );

    let mut examples = seed_examples(params);
    let mut seen: HashSet<([u32; 10], u8)> =
        examples.iter().map(|ex| (ex.col_words, ex.piece)).collect();
    let mut last_program = None;
    let mut synthesis = SynthesisSession::new(program_length, synthesis_options.clone());
    let mut newly_asserted = examples.len();

    info!("seeded with {} examples", examples.len());

    for round in 0..max_rounds {
        if newly_asserted > 0 {
            let start = examples.len() - newly_asserted;
            let new_examples = &examples[start..];
            if !synthesis.assert_examples(new_examples) {
                warn!(
                    "round {round}: found an example with no valid placements while asserting incrementally"
                );
                return CegisResult::SynthesisFailed {
                    round,
                    examples: synthesis.asserted_examples(),
                };
            }
            info!(
                "round {round}: asserted {newly_asserted} new examples into incremental solver (total asserted={})",
                synthesis.asserted_examples()
            );
            newly_asserted = 0;
        }

        let program = match synthesis.check() {
            SynthesisCheckResult::Sat(program) => program,
            SynthesisCheckResult::Unsat => {
                warn!(
                    "round {round}: synthesis UNSAT with {} examples and program length {}",
                    synthesis.asserted_examples(),
                    program_length
                );
                return CegisResult::SynthesisFailed {
                    round,
                    examples: synthesis.asserted_examples(),
                };
            }
            SynthesisCheckResult::Unknown(reason) => {
                warn!(
                    "round {round}: synthesis returned Unknown with {} examples and program length {}: {:?}",
                    synthesis.asserted_examples(),
                    program_length,
                    reason
                );
                return CegisResult::SynthesisUnknown {
                    round,
                    examples: synthesis.asserted_examples(),
                    reason,
                };
            }
        };

        info!(
            "round {round}: synthesized program from {} examples; verifying...",
            examples.len()
        );
        info!("round {round}: synthesized program:\n{}", program);

        match verification_oracle(&program, simulation, params) {
            VerificationResult::Survived(summary) => {
                log_verification_summary(round, &summary);
                info!(
                    "round {round}: survived {} seeds for {} pieces",
                    simulation.num_seeds, simulation.survival_target
                );
                return CegisResult::Survived {
                    program,
                    rounds: round + 1,
                    examples: examples.len(),
                };
            }
            VerificationResult::Failure(summary) => {
                log_verification_summary(round, &summary);

                // Extract counterexamples from ALL failure traces for diversity
                let before = examples.len();
                let mut all_failures: Vec<&[TraceEntry]> = Vec::new();
                if let Some(failure) = &summary.first_failure {
                    all_failures.push(&failure.trace);
                }
                for extra in &summary.extra_failures {
                    all_failures.push(&extra.trace);
                }

                for trace in all_failures {
                    let counterexamples = extract_counterexamples(trace, params);
                    for ex in counterexamples {
                        if seen.insert((ex.col_words, ex.piece)) {
                            examples.push(ex);
                        }
                    }
                }

                let added = examples.len() - before;
                newly_asserted = added;
                info!(
                    "round {round}: added {added} new examples; total examples={}",
                    examples.len()
                );

                if added == 0 {
                    warn!("round {round}: all extracted counterexamples were duplicates");
                    return CegisResult::NoNewCounterexamples {
                        round,
                        examples: examples.len(),
                        last_program: program,
                    };
                }

                last_program = Some(program);
            }
        }
    }

    warn!("exceeded {max_rounds} rounds without empirical convergence");
    CegisResult::MaxRoundsExceeded {
        last_program,
        examples: examples.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_examples_has_at_least_one() {
        let params = InvariantParams::new(4, 2, 4);
        let examples = seed_examples(&params);
        assert!(
            !examples.is_empty(),
            "should have at least one seed example"
        );
        // Seed starts with just one example to keep first synthesis fast
        assert_eq!(examples.len(), 1);
    }

    #[test]
    fn test_push_example_for_board_with_high_bits() {
        let params = InvariantParams::new(8, 0, 8);
        let mut examples = Vec::new();
        let mut seen = HashSet::new();

        let mut board = TetrisBoard::new();
        for row in 0..=7 {
            board.set_bit(0, row);
        }
        push_example_for_board(
            &mut examples,
            &mut seen,
            &board,
            TetrisPiece::O_PIECE,
            &params,
        );

        // With 8-bit col_words, bit 7 should be visible
        assert!(examples.iter().any(|ex| ex.col_words[0] & (1 << 7) != 0));
    }
}
