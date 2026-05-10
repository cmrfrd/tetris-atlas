use z3::ast::{Ast, BV, Bool};

use crate::config::{NUM_COL_INPUTS, NUM_OPCODES, NUM_OPERAND_SLOTS, NUM_REGISTERS, VM_VALUE_BITS};

/// Per-field BV widths — tight to reduce Z3 search space.
const OPCODE_BITS: u32 = 4; // 9 opcodes → fits in 4 bits
const DST_BITS: u32 = 3; // 8 registers → fits in 3 bits
const SRC1_BITS: u32 = 8; // 8 bits for CONST_LO immediate (0–255)
const SRC2_BITS: u32 = 5; // 19 operand slots → fits in 5 bits

/// Create symbolic variables for one instruction's fields.
pub struct SymbolicInstruction {
    pub opcode: BV,
    pub dst: BV,
    pub src1: BV,
    pub src2: BV,
}

/// Create all symbolic instruction variables for a program of given length.
pub fn make_symbolic_program(prefix: &str, length: usize) -> Vec<SymbolicInstruction> {
    (0..length)
        .map(|i| SymbolicInstruction {
            opcode: BV::new_const(format!("{prefix}_op{i}"), OPCODE_BITS),
            dst: BV::new_const(format!("{prefix}_dst{i}"), DST_BITS),
            src1: BV::new_const(format!("{prefix}_s1_{i}"), SRC1_BITS),
            src2: BV::new_const(format!("{prefix}_s2_{i}"), SRC2_BITS),
        })
        .collect()
}

/// Add validity constraints for symbolic program fields.
pub fn constrain_program_validity(prog: &[SymbolicInstruction]) -> Vec<Bool> {
    let mut constraints = Vec::new();
    let max_op = BV::from_u64(NUM_OPCODES as u64, OPCODE_BITS);
    let max_src1 = BV::from_u64(NUM_OPERAND_SLOTS as u64, SRC1_BITS);
    let max_src2 = BV::from_u64(NUM_OPERAND_SLOTS as u64, SRC2_BITS);
    let const_op = BV::from_u64(crate::vm::Opcode::ConstLo as u64, OPCODE_BITS);

    for instr in prog {
        constraints.push(instr.opcode.bvult(&max_op));
        // dst: 3 bits = 0–7, exactly NUM_REGISTERS values — no constraint needed
        // src1: unrestricted for CONST_LO (immediate 0–255), else < NUM_OPERAND_SLOTS
        let src1_is_operand = instr.src1.bvult(&max_src1);
        constraints.push(Bool::or(&[instr.opcode.eq(&const_op), src1_is_operand]));
        constraints.push(instr.src2.bvult(&max_src2));
    }
    constraints
}

/// Build a symbolic ITE tree to select an operand value from registers + col words + piece.
/// The `idx` BV can be any width — slot constants are created at the same width.
fn select_operand(
    idx: &BV,
    regs: &[BV; NUM_REGISTERS as usize],
    col_words: &[BV; NUM_COL_INPUTS as usize],
    piece: &BV,
) -> BV {
    let idx_width = idx.get_size();
    let mut result = BV::from_u64(0, VM_VALUE_BITS);
    for slot in (0..NUM_OPERAND_SLOTS).rev() {
        let slot_bv = BV::from_u64(slot as u64, idx_width);
        let is_match = idx.eq(&slot_bv);
        let val = if slot < NUM_REGISTERS {
            regs[slot as usize].clone()
        } else if slot < NUM_REGISTERS + NUM_COL_INPUTS {
            col_words[(slot - NUM_REGISTERS) as usize].clone()
        } else {
            piece.clone()
        };
        result = is_match.ite(&val, &result);
    }
    result
}

/// Symbolically execute a symbolic program on concrete column word inputs and piece.
/// Returns the symbolic value of R0 after execution.
///
/// All opcodes, dst, src1, src2 are symbolic — this is used in synthesis.
pub fn symbolic_execute_on_concrete(
    prog: &[SymbolicInstruction],
    concrete_col_words: &[u32; NUM_COL_INPUTS as usize],
    concrete_piece: u8,
) -> BV {
    let mut regs: [BV; NUM_REGISTERS as usize] =
        std::array::from_fn(|_| BV::from_u64(0, VM_VALUE_BITS));
    let col_words: [BV; NUM_COL_INPUTS as usize] =
        std::array::from_fn(|i| BV::from_u64(concrete_col_words[i] as u64, VM_VALUE_BITS));
    let piece = BV::from_u64(concrete_piece as u64, VM_VALUE_BITS);

    let zero = BV::from_u64(0, VM_VALUE_BITS);
    let one = BV::from_u64(1, VM_VALUE_BITS);

    for instr in prog {
        let s1 = select_operand(&instr.src1, &regs, &col_words, &piece);
        let s2 = select_operand(&instr.src2, &regs, &col_words, &piece);

        // Compute result for each possible opcode
        let add_val = s1.bvadd(&s2);
        let sub_val = s1.bvsub(&s2);
        let and_val = s1.bvand(&s2);
        let or_val = s1.bvor(&s2);
        let lt_val = s1.bvult(&s2).ite(&one, &zero);
        let min_val = s1.bvult(&s2).ite(&s1, &s2);

        let op0 = BV::from_u64(0, OPCODE_BITS);
        let op1 = BV::from_u64(1, OPCODE_BITS);
        let op2 = BV::from_u64(2, OPCODE_BITS);
        let op3 = BV::from_u64(3, OPCODE_BITS);
        let op4 = BV::from_u64(4, OPCODE_BITS);
        let op6 = BV::from_u64(6, OPCODE_BITS);

        // CONST_LO: dst = src1 field value (zero-extended to VM_VALUE_BITS)
        let const_val = if SRC1_BITS < VM_VALUE_BITS {
            instr.src1.zero_ext(VM_VALUE_BITS - SRC1_BITS)
        } else {
            instr.src1.extract(VM_VALUE_BITS - 1, 0)
        };

        let computed = instr.opcode.eq(&op0).ite(
            &add_val,
            &instr.opcode.eq(&op1).ite(
                &sub_val,
                &instr.opcode.eq(&op2).ite(
                    &and_val,
                    &instr.opcode.eq(&op3).ite(
                        &or_val,
                        &instr
                            .opcode
                            .eq(&op4)
                            .ite(&lt_val, &instr.opcode.eq(&op6).ite(&const_val, &min_val)),
                    ),
                ),
            ),
        );

        // MUX special: if opcode==5, dst = (s1!=0) ? s2 : old_dst
        let is_mux = instr.opcode.eq(BV::from_u64(5, OPCODE_BITS));
        let mux_fires = s1.eq(&zero).not();

        // NOP: if opcode==8, no register changes
        let is_nop = instr.opcode.eq(BV::from_u64(8, OPCODE_BITS));

        for r in 0..NUM_REGISTERS {
            let is_dst = instr.dst.eq(BV::from_u64(r as u64, DST_BITS));
            let mux_result = mux_fires.ite(&s2, &regs[r as usize]);
            let final_val = is_mux.ite(&mux_result, &computed);
            let final_val = is_nop.ite(&regs[r as usize], &final_val);
            let new_reg = is_dst.ite(&final_val, &regs[r as usize]);
            regs[r as usize] = new_reg;
        }
    }

    regs[0].clone()
}

/// Execute a concrete program on symbolic column word inputs and piece.
/// Opcodes are fixed (concrete), but col_words and piece are symbolic BV values.
///
/// This is kept for cross-validation tests but is not used in the main
/// simulation-based CEGIS flow.
pub fn concrete_execute_on_symbolic(
    program: &crate::vm::Program,
    symbolic_col_words: &[BV; NUM_COL_INPUTS as usize],
    symbolic_piece: &BV,
) -> BV {
    let mut regs: [BV; NUM_REGISTERS as usize] =
        std::array::from_fn(|_| BV::from_u64(0, VM_VALUE_BITS));

    let zero = BV::from_u64(0, VM_VALUE_BITS);
    let one = BV::from_u64(1, VM_VALUE_BITS);

    for instr in &program.instructions {
        let s1 = concrete_read_operand(&regs, symbolic_col_words, symbolic_piece, instr.src1);
        let s2 = concrete_read_operand(&regs, symbolic_col_words, symbolic_piece, instr.src2);
        let d = instr.dst as usize;

        match instr.opcode {
            8 => {} // NOP: skip
            _ => {
                let new_val = match instr.opcode {
                    0 => s1.bvadd(&s2),
                    1 => s1.bvsub(&s2),
                    2 => s1.bvand(&s2),
                    3 => s1.bvor(&s2),
                    4 => s1.bvult(&s2).ite(&one, &zero),
                    5 => {
                        let fires = s1.eq(&zero).not();
                        fires.ite(&s2, &regs[d])
                    }
                    6 => BV::from_u64(instr.src1 as u64, VM_VALUE_BITS),
                    7 => s1.bvult(&s2).ite(&s1, &s2),
                    _ => regs[d].clone(),
                };
                regs[d] = new_val;
            }
        }
    }

    regs[0].clone()
}

fn concrete_read_operand(
    regs: &[BV; NUM_REGISTERS as usize],
    col_words: &[BV; NUM_COL_INPUTS as usize],
    piece: &BV,
    idx: u8,
) -> BV {
    if idx < NUM_REGISTERS {
        regs[idx as usize].clone()
    } else if idx < NUM_REGISTERS + NUM_COL_INPUTS {
        col_words[(idx - NUM_REGISTERS) as usize].clone()
    } else if idx < NUM_OPERAND_SLOTS {
        piece.clone()
    } else {
        BV::from_u64(0, VM_VALUE_BITS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::{Instruction, Program};
    use z3::{SatResult, Solver};

    #[test]
    fn test_symbolic_execute_const() {
        let prog_sym = make_symbolic_program("p", 1);
        let solver = Solver::new();

        solver.assert(prog_sym[0].opcode.eq(&BV::from_u64(6, OPCODE_BITS)));
        solver.assert(prog_sym[0].dst.eq(&BV::from_u64(0, DST_BITS)));
        solver.assert(prog_sym[0].src1.eq(&BV::from_u64(42, SRC1_BITS)));
        solver.assert(prog_sym[0].src2.eq(&BV::from_u64(0, SRC2_BITS)));

        let r0 = symbolic_execute_on_concrete(&prog_sym, &[0; 10], 0);

        solver.assert(r0.eq(&BV::from_u64(42, VM_VALUE_BITS)));
        assert_eq!(solver.check(), SatResult::Sat);
    }

    #[test]
    fn test_symbolic_execute_piece_input() {
        let prog_sym = make_symbolic_program("p", 1);
        let solver = Solver::new();

        // ADD R0 <- P + R0 (R0=0, so R0=piece)
        solver.assert(prog_sym[0].opcode.eq(&BV::from_u64(0, OPCODE_BITS)));
        solver.assert(prog_sym[0].dst.eq(&BV::from_u64(0, DST_BITS)));
        solver.assert(prog_sym[0].src1.eq(&BV::from_u64(18, SRC1_BITS))); // P = slot 18
        solver.assert(prog_sym[0].src2.eq(&BV::from_u64(0, SRC2_BITS)));

        let r0 = symbolic_execute_on_concrete(&prog_sym, &[0; 10], 5);
        solver.assert(r0.eq(&BV::from_u64(5, VM_VALUE_BITS)));
        assert_eq!(solver.check(), SatResult::Sat);
    }

    #[test]
    fn test_concrete_on_symbolic_matches_concrete() {
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 7, // MIN R0 <- C0, C1
                dst: 0,
                src1: 8,
                src2: 9,
            }],
        };

        let solver = Solver::new();
        let sym_cw: [BV; NUM_COL_INPUTS as usize] = std::array::from_fn(|i| {
            let bv = BV::new_const(format!("c{i}"), VM_VALUE_BITS);
            solver.assert(bv.eq(&BV::from_u64(
                if i == 0 {
                    4
                } else if i == 1 {
                    2
                } else {
                    0
                },
                VM_VALUE_BITS,
            )));
            bv
        });
        let sym_p = BV::new_const("piece", VM_VALUE_BITS);
        solver.assert(sym_p.eq(&BV::from_u64(0, VM_VALUE_BITS)));

        let r0 = concrete_execute_on_symbolic(&prog, &sym_cw, &sym_p);
        solver.assert(r0.eq(&BV::from_u64(2, VM_VALUE_BITS)));
        assert_eq!(solver.check(), SatResult::Sat);

        // Verify doesn't equal wrong answer
        let solver2 = Solver::new();
        let sym_cw2: [BV; NUM_COL_INPUTS as usize] = std::array::from_fn(|i| {
            let bv = BV::new_const(format!("c{i}"), VM_VALUE_BITS);
            solver2.assert(bv.eq(&BV::from_u64(
                if i == 0 {
                    4
                } else if i == 1 {
                    2
                } else {
                    0
                },
                VM_VALUE_BITS,
            )));
            bv
        });
        let sym_p2 = BV::new_const("piece", VM_VALUE_BITS);
        solver2.assert(sym_p2.eq(&BV::from_u64(0, VM_VALUE_BITS)));
        let r0_2 = concrete_execute_on_symbolic(&prog, &sym_cw2, &sym_p2);
        solver2.assert(r0_2.eq(&BV::from_u64(4, VM_VALUE_BITS)));
        assert_eq!(solver2.check(), SatResult::Unsat);
    }

    #[test]
    fn test_symbolic_nop_preserves_r0() {
        let prog_sym = make_symbolic_program("p", 2);
        let solver = Solver::new();

        // Instruction 0: CONST_LO R0 <- 7
        solver.assert(prog_sym[0].opcode.eq(&BV::from_u64(6, OPCODE_BITS)));
        solver.assert(prog_sym[0].dst.eq(&BV::from_u64(0, DST_BITS)));
        solver.assert(prog_sym[0].src1.eq(&BV::from_u64(7, SRC1_BITS)));
        solver.assert(prog_sym[0].src2.eq(&BV::from_u64(0, SRC2_BITS)));

        // Instruction 1: NOP targeting R0
        solver.assert(prog_sym[1].opcode.eq(&BV::from_u64(8, OPCODE_BITS)));
        solver.assert(prog_sym[1].dst.eq(&BV::from_u64(0, DST_BITS)));
        solver.assert(prog_sym[1].src1.eq(&BV::from_u64(0, SRC1_BITS)));
        solver.assert(prog_sym[1].src2.eq(&BV::from_u64(0, SRC2_BITS)));

        let r0 = symbolic_execute_on_concrete(&prog_sym, &[0; 10], 0);
        solver.assert(r0.eq(&BV::from_u64(7, VM_VALUE_BITS)));
        assert_eq!(solver.check(), SatResult::Sat);
    }

    #[test]
    fn test_concrete_nop_on_symbolic() {
        let prog = Program {
            instructions: vec![
                Instruction {
                    opcode: 6,
                    dst: 0,
                    src1: 42,
                    src2: 0,
                },
                Instruction {
                    opcode: 8, // NOP
                    dst: 0,
                    src1: 0,
                    src2: 0,
                },
            ],
        };

        let solver = Solver::new();
        let sym_cw: [BV; NUM_COL_INPUTS as usize] = std::array::from_fn(|i| {
            let bv = BV::new_const(format!("c{i}"), VM_VALUE_BITS);
            solver.assert(bv.eq(&BV::from_u64(0, VM_VALUE_BITS)));
            bv
        });
        let sym_p = BV::new_const("piece", VM_VALUE_BITS);
        solver.assert(sym_p.eq(&BV::from_u64(0, VM_VALUE_BITS)));

        let r0 = concrete_execute_on_symbolic(&prog, &sym_cw, &sym_p);
        solver.assert(r0.eq(&BV::from_u64(42, VM_VALUE_BITS)));
        assert_eq!(solver.check(), SatResult::Sat);
    }

    /// Cross-validate: for every opcode, run concrete VM and symbolic VM on the
    /// same inputs and verify they agree.
    #[test]
    fn test_concrete_vs_symbolic_all_opcodes() {
        let test_cases: Vec<(u8, u8, u8, u8, [u32; 10], u8, u32)> = vec![
            // (opcode, dst, src1, src2, col_words, piece, expected_r0)
            (0, 0, 8, 9, [3, 4, 0, 0, 0, 0, 0, 0, 0, 0], 0, 7), // ADD
            (1, 0, 8, 9, [10, 3, 0, 0, 0, 0, 0, 0, 0, 0], 0, 7), // SUB
            (2, 0, 8, 9, [0xFF, 0x0F, 0, 0, 0, 0, 0, 0, 0, 0], 0, 0x0F), // AND
            (3, 0, 8, 9, [0xF0, 0x0F, 0, 0, 0, 0, 0, 0, 0, 0], 0, 0xFF), // OR
            (4, 0, 8, 9, [2, 5, 0, 0, 0, 0, 0, 0, 0, 0], 0, 1), // CMP_LT true
            (4, 0, 8, 9, [5, 2, 0, 0, 0, 0, 0, 0, 0, 0], 0, 0), // CMP_LT false
            (6, 0, 99, 0, [0; 10], 0, 99),                      // CONST_LO
            (7, 0, 8, 9, [6, 3, 0, 0, 0, 0, 0, 0, 0, 0], 0, 3), // MIN
            (8, 0, 8, 9, [5, 5, 0, 0, 0, 0, 0, 0, 0, 0], 0, 0), // NOP (R0 stays 0)
            (0, 0, 18, 8, [10, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3, 13), // ADD P + C0
        ];

        for (opcode, dst, src1, src2, col_words, piece, expected) in &test_cases {
            let prog = Program {
                instructions: vec![Instruction {
                    opcode: *opcode,
                    dst: *dst,
                    src1: *src1,
                    src2: *src2,
                }],
            };

            // Concrete
            let concrete_result = prog.execute(col_words, *piece);
            assert_eq!(
                concrete_result, *expected,
                "concrete mismatch for opcode {opcode}"
            );

            // Symbolic (concrete program on symbolic inputs)
            let solver = Solver::new();
            let sym_cw: [BV; NUM_COL_INPUTS as usize] = std::array::from_fn(|i| {
                let bv = BV::new_const(format!("c{i}"), VM_VALUE_BITS);
                solver.assert(bv.eq(&BV::from_u64(col_words[i] as u64, VM_VALUE_BITS)));
                bv
            });
            let sym_p = BV::new_const("piece", VM_VALUE_BITS);
            solver.assert(sym_p.eq(&BV::from_u64(*piece as u64, VM_VALUE_BITS)));

            let r0 = concrete_execute_on_symbolic(&prog, &sym_cw, &sym_p);
            solver.assert(r0.eq(&BV::from_u64(*expected as u64, VM_VALUE_BITS)));
            assert_eq!(
                solver.check(),
                SatResult::Sat,
                "symbolic mismatch for opcode {opcode}"
            );
        }
    }

    #[test]
    fn test_symbolic_synthesis_finds_program() {
        // Use symbolic execution to find a program that outputs C0
        let prog_sym = make_symbolic_program("p", 2);
        let solver = Solver::new();

        for c in constrain_program_validity(&prog_sym) {
            solver.assert(&c);
        }

        // Example 1: C0=3 → output 3
        let r0_1 = symbolic_execute_on_concrete(&prog_sym, &[3, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0);
        solver.assert(r0_1.eq(&BV::from_u64(3, VM_VALUE_BITS)));

        // Example 2: C0=7 → output 7
        let r0_2 = symbolic_execute_on_concrete(&prog_sym, &[7, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0);
        solver.assert(r0_2.eq(&BV::from_u64(7, VM_VALUE_BITS)));

        assert_eq!(solver.check(), SatResult::Sat);
    }
}
