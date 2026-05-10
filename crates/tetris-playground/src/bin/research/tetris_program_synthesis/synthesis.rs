use std::ffi::CString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use tracing::{debug, info, warn};
use z3::ast::{Ast, BV, Bool};
use z3::{Params, SatResult, Solver, StatisticsValue};

use crate::config::{NUM_OPCODES, VM_VALUE_BITS};
use crate::vm::{Instruction, Program};
use crate::vm_symbolic::{
    SymbolicInstruction, constrain_program_validity, make_symbolic_program,
    symbolic_execute_on_concrete,
};

/// A synthesis example: column words + piece → set of valid global placement indices.
#[derive(Debug, Clone)]
pub struct SynthesisExample {
    pub col_words: [u32; 10],
    pub piece: u8,
    pub valid_global_indices: Vec<u8>,
}

/// Optional observability and resource controls for Z3 synthesis queries.
#[derive(Debug, Clone, Default)]
pub struct SynthesisOptions {
    pub log_z3_stats: bool,
    pub dump_smt2_dir: Option<PathBuf>,
    pub z3_timeout_ms: Option<u32>,
    pub append_z3_log_comments: bool,
}

fn append_z3_log_comment(comment: &str) {
    let sanitized = comment.replace('\0', "\\0");
    if let Ok(c_string) = CString::new(format!(";; {sanitized}\n")) {
        unsafe {
            z3_sys::Z3_append_log(c_string.as_ptr());
        }
    }
}

fn format_z3_stats(solver: &Solver) -> String {
    solver
        .get_statistics()
        .entries()
        .map(|entry| match entry.value {
            StatisticsValue::UInt(value) => format!("{}={}", entry.key, value),
            StatisticsValue::Double(value) => format!("{}={:.3}", entry.key, value),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn dump_smt2(
    solver: &Solver,
    dir: &Path,
    examples_len: usize,
    program_length: usize,
) -> anyhow::Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("creating SMT2 dump dir {}", dir.display()))?;
    let path = dir.join(format!(
        "synthesis_examples-{examples_len}_prog-{program_length}.smt2"
    ));
    fs::write(&path, solver.to_smt2())
        .with_context(|| format!("writing SMT2 dump {}", path.display()))?;
    info!("wrote SMT2 query to {}", path.display());
    Ok(())
}

#[derive(Debug)]
pub enum SynthesisCheckResult {
    Sat(Program),
    Unsat,
    Unknown(Option<String>),
}

/// Persistent synthesis session for one fixed program length.
///
/// CEGIS examples only add constraints, so this keeps one Z3 solver and asserts
/// only the new examples instead of rebuilding the entire query each round.
pub struct SynthesisSession {
    solver: Solver,
    sym_prog: Vec<SymbolicInstruction>,
    program_length: usize,
    asserted_examples: usize,
    options: SynthesisOptions,
}

impl SynthesisSession {
    pub fn new(program_length: usize, options: SynthesisOptions) -> Self {
        let solver = Solver::new();
        if let Some(timeout_ms) = options.z3_timeout_ms {
            let mut params = Params::new();
            params.set_u32("timeout", timeout_ms);
            solver.set_params(&params);
        }

        let sym_prog = make_symbolic_program("syn", program_length);
        for c in constrain_program_validity(&sym_prog) {
            solver.assert(&c);
        }

        Self {
            solver,
            sym_prog,
            program_length,
            asserted_examples: 0,
            options,
        }
    }

    pub fn asserted_examples(&self) -> usize {
        self.asserted_examples
    }

    pub fn assert_examples(&mut self, examples: &[SynthesisExample]) -> bool {
        for (offset, example) in examples.iter().enumerate() {
            let ex_idx = self.asserted_examples + offset;
            if !self.assert_example(example, ex_idx) {
                return false;
            }
        }
        self.asserted_examples += examples.len();
        true
    }

    fn assert_example(&self, example: &SynthesisExample, ex_idx: usize) -> bool {
        if example.valid_global_indices.is_empty() {
            debug!("example {ex_idx} has no valid indices — unsatisfiable");
            return false;
        }

        let r0 = symbolic_execute_on_concrete(&self.sym_prog, &example.col_words, example.piece);
        // Cap disjunction size to keep the formula tractable for Z3.
        // More options = larger formula = slower solving. 3 is enough for
        // synthesis to find valid programs while keeping queries fast.
        let indices = if example.valid_global_indices.len() > 3 {
            &example.valid_global_indices[..3]
        } else {
            &example.valid_global_indices
        };
        let options: Vec<Bool> = indices
            .iter()
            .map(|&idx| r0.eq(BV::from_u64(idx as u64, VM_VALUE_BITS)))
            .collect();
        self.solver.assert(Bool::or(&options));
        true
    }

    pub fn check(&self) -> SynthesisCheckResult {
        if self.options.append_z3_log_comments {
            append_z3_log_comment(&format!(
                "incremental synthesis check start: asserted_examples={}, program_length={}",
                self.asserted_examples, self.program_length
            ));
        }

        if let Some(dir) = &self.options.dump_smt2_dir {
            if let Err(error) = dump_smt2(
                &self.solver,
                dir,
                self.asserted_examples,
                self.program_length,
            ) {
                warn!("{error:#}");
            }
        }

        let check_start = Instant::now();
        let result = self.solver.check();
        let check_elapsed = check_start.elapsed();
        info!(
            "z3 synthesis check: result={:?}, examples={}, program_length={}, elapsed={:.3}s",
            result,
            self.asserted_examples,
            self.program_length,
            check_elapsed.as_secs_f64()
        );
        if self.options.log_z3_stats {
            info!("z3 stats: {}", format_z3_stats(&self.solver));
        }
        if self.options.append_z3_log_comments {
            append_z3_log_comment(&format!(
                "incremental synthesis check result: result={:?}, asserted_examples={}, program_length={}, elapsed_s={:.3}",
                result,
                self.asserted_examples,
                self.program_length,
                check_elapsed.as_secs_f64()
            ));
        }

        match result {
            SatResult::Sat => match self.model_to_program() {
                Some(program) => SynthesisCheckResult::Sat(program),
                None => SynthesisCheckResult::Unknown(Some(
                    "Z3 returned SAT but no model was available".to_string(),
                )),
            },
            SatResult::Unsat => {
                debug!(
                    "synthesis UNSAT with {} instructions and {} examples",
                    self.program_length, self.asserted_examples
                );
                SynthesisCheckResult::Unsat
            }
            SatResult::Unknown => {
                let reason = self.solver.get_reason_unknown();
                debug!("synthesis returned Unknown: {:?}", reason);
                SynthesisCheckResult::Unknown(reason)
            }
        }
    }

    fn model_to_program(&self) -> Option<Program> {
        let model = self.solver.get_model()?;
        let mut instructions = Vec::with_capacity(self.program_length);

        for instr in &self.sym_prog {
            let opcode = model
                .eval(&instr.opcode, true)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u8;
            let dst = model
                .eval(&instr.dst, true)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u8;
            let src1 = model
                .eval(&instr.src1, true)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u8;
            let src2 = model
                .eval(&instr.src2, true)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u8;

            // Clamp opcode to valid range (should already be constrained)
            let opcode = if opcode < NUM_OPCODES { opcode } else { 0 };

            instructions.push(Instruction {
                opcode,
                dst,
                src1,
                src2,
            });
        }

        Some(Program { instructions })
    }
}

/// Attempt to synthesize a program of the given length that maps each example's
/// (col_words, piece) to one of its valid global placement indices.
///
/// Returns `Some(program)` if synthesis succeeds, `None` if UNSAT/Unknown.
pub fn synthesis_oracle(
    examples: &[SynthesisExample],
    program_length: usize,
    options: &SynthesisOptions,
) -> Option<Program> {
    let mut session = SynthesisSession::new(program_length, options.clone());
    if !session.assert_examples(examples) {
        return None;
    }
    match session.check() {
        SynthesisCheckResult::Sat(program) => Some(program),
        SynthesisCheckResult::Unsat | SynthesisCheckResult::Unknown(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_options() -> SynthesisOptions {
        SynthesisOptions::default()
    }

    #[test]
    fn test_synthesize_constant() {
        // Trivial: always output 3 regardless of inputs
        let examples = vec![
            SynthesisExample {
                col_words: [0; 10],
                piece: 0,
                valid_global_indices: vec![3],
            },
            SynthesisExample {
                col_words: [1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                piece: 1,
                valid_global_indices: vec![3],
            },
        ];

        let prog = synthesis_oracle(&examples, 4, &test_options());
        assert!(prog.is_some(), "should synthesize constant-3 program");
        let prog = prog.unwrap();

        for ex in &examples {
            let result = prog.execute(&ex.col_words, ex.piece);
            assert!(
                ex.valid_global_indices
                    .iter()
                    .any(|&idx| u32::from(idx) == result),
                "program output {result} not in {:?}",
                ex.valid_global_indices
            );
        }
    }

    #[test]
    fn test_synthesize_const_lo_above_operand_slots() {
        let examples = vec![SynthesisExample {
            col_words: [0; 10],
            piece: 0,
            valid_global_indices: vec![42],
        }];

        let prog = synthesis_oracle(&examples, 1, &test_options());
        assert!(prog.is_some(), "should synthesize one-instruction constant");
        let prog = prog.unwrap();
        assert_eq!(prog.execute(&examples[0].col_words, examples[0].piece), 42);
    }

    #[test]
    fn test_incremental_session_accepts_new_examples() {
        let mut session = SynthesisSession::new(4, test_options());
        let first = vec![SynthesisExample {
            col_words: [0; 10],
            piece: 0,
            valid_global_indices: vec![3],
        }];
        assert!(session.assert_examples(&first));
        assert_eq!(session.asserted_examples(), 1);
        assert!(matches!(session.check(), SynthesisCheckResult::Sat(_)));

        let second = vec![SynthesisExample {
            col_words: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            piece: 0,
            valid_global_indices: vec![3],
        }];
        assert!(session.assert_examples(&second));
        assert_eq!(session.asserted_examples(), 2);
        assert!(matches!(session.check(), SynthesisCheckResult::Sat(_)));
    }

    #[test]
    fn test_synthesize_col_word_based() {
        // Output 0 if C0 == 0, output 1 otherwise
        let examples = vec![
            SynthesisExample {
                col_words: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                piece: 0,
                valid_global_indices: vec![0],
            },
            SynthesisExample {
                col_words: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                piece: 0,
                valid_global_indices: vec![1],
            },
            SynthesisExample {
                col_words: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                piece: 0,
                valid_global_indices: vec![1],
            },
        ];

        let prog = synthesis_oracle(&examples, 8, &test_options());
        assert!(prog.is_some(), "should synthesize col-word-based program");
        let prog = prog.unwrap();

        for ex in &examples {
            let result = prog.execute(&ex.col_words, ex.piece);
            assert!(
                ex.valid_global_indices
                    .iter()
                    .any(|&idx| u32::from(idx) == result),
                "program output {result} not in {:?} for col_words {:?}",
                ex.valid_global_indices,
                ex.col_words
            );
        }
    }

    #[test]
    fn test_synthesize_multiple_valid() {
        let examples = vec![
            SynthesisExample {
                col_words: [0; 10],
                piece: 0,
                valid_global_indices: vec![0, 1, 2, 3, 4],
            },
            SynthesisExample {
                col_words: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                piece: 2,
                valid_global_indices: vec![2, 5, 7],
            },
        ];

        let prog = synthesis_oracle(&examples, 6, &test_options());
        assert!(prog.is_some());
        let prog = prog.unwrap();

        for ex in &examples {
            let result = prog.execute(&ex.col_words, ex.piece);
            assert!(
                ex.valid_global_indices
                    .iter()
                    .any(|&idx| u32::from(idx) == result),
                "output {result} not valid for col_words {:?}",
                ex.col_words
            );
        }
    }

    #[test]
    fn test_synthesize_piece_dependent() {
        // Different pieces should map to different outputs
        let examples = vec![
            SynthesisExample {
                col_words: [0; 10],
                piece: 0,
                valid_global_indices: vec![0],
            },
            SynthesisExample {
                col_words: [0; 10],
                piece: 3,
                valid_global_indices: vec![3],
            },
        ];

        let prog = synthesis_oracle(&examples, 8, &test_options());
        assert!(prog.is_some(), "should synthesize piece-dependent program");
        let prog = prog.unwrap();

        for ex in &examples {
            let result = prog.execute(&ex.col_words, ex.piece);
            assert!(
                ex.valid_global_indices
                    .iter()
                    .any(|&idx| u32::from(idx) == result),
                "output {result} not valid for piece {}",
                ex.piece
            );
        }
    }
}
