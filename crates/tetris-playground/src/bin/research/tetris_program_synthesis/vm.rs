use crate::config::{NUM_COL_INPUTS, NUM_OPERAND_SLOTS, NUM_REGISTERS};

/// Opcodes for the bitvector VM (4-bit encoding, 0..8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    Add = 0,
    Sub = 1,
    And = 2,
    Or = 3,
    CmpLt = 4,
    Mux = 5,
    ConstLo = 6,
    Min = 7,
    Nop = 8,
}

impl Opcode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Add),
            1 => Some(Self::Sub),
            2 => Some(Self::And),
            3 => Some(Self::Or),
            4 => Some(Self::CmpLt),
            5 => Some(Self::Mux),
            6 => Some(Self::ConstLo),
            7 => Some(Self::Min),
            8 => Some(Self::Nop),
            _ => None,
        }
    }
}

impl std::fmt::Display for Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => write!(f, "ADD"),
            Self::Sub => write!(f, "SUB"),
            Self::And => write!(f, "AND"),
            Self::Or => write!(f, "OR"),
            Self::CmpLt => write!(f, "CMP_LT"),
            Self::Mux => write!(f, "MUX"),
            Self::ConstLo => write!(f, "CONST_LO"),
            Self::Min => write!(f, "MIN"),
            Self::Nop => write!(f, "NOP"),
        }
    }
}

/// A single VM instruction: opcode(4) | dst(3) | src1(8) | src2(5) = 20 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instruction {
    pub opcode: u8,
    pub dst: u8,
    pub src1: u8,
    pub src2: u8,
}

impl Instruction {
    pub fn encode(&self) -> u32 {
        ((self.opcode as u32 & 0xF) << 16)
            | ((self.dst as u32 & 0x7) << 13)
            | ((self.src1 as u32) << 5)
            | (self.src2 as u32 & 0x1F)
    }

    pub fn decode(bits: u32) -> Self {
        Self {
            opcode: ((bits >> 16) & 0xF) as u8,
            dst: ((bits >> 13) & 0x7) as u8,
            src1: ((bits >> 5) & 0xFF) as u8,
            src2: (bits & 0x1F) as u8,
        }
    }
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let op = Opcode::from_u8(self.opcode)
            .map(|o| format!("{o}"))
            .unwrap_or_else(|| format!("OP{}", self.opcode));
        write!(
            f,
            "{:<8} R{} <- {}  {}",
            op,
            self.dst,
            operand_name(self.src1),
            operand_name(self.src2)
        )
    }
}

/// Human-readable name for an operand index.
fn operand_name(idx: u8) -> String {
    if idx < NUM_REGISTERS {
        format!("R{idx}")
    } else if idx < NUM_REGISTERS + NUM_COL_INPUTS {
        format!("C{}", idx - NUM_REGISTERS)
    } else if idx < NUM_OPERAND_SLOTS {
        "P".to_string()
    } else {
        format!("?{idx}")
    }
}

/// A program is a sequence of instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub instructions: Vec<Instruction>,
}

impl Program {
    /// Read an operand value from registers, column word inputs, or piece input.
    fn read_operand(
        regs: &[u32; NUM_REGISTERS as usize],
        col_words: &[u32; 10],
        piece: u8,
        idx: u8,
    ) -> u32 {
        if idx < NUM_REGISTERS {
            regs[idx as usize]
        } else if idx < NUM_REGISTERS + NUM_COL_INPUTS {
            col_words[(idx - NUM_REGISTERS) as usize] & 0xFF
        } else if idx < NUM_OPERAND_SLOTS {
            piece as u32
        } else {
            0
        }
    }

    /// Execute the program on given column words and piece index, returning R0.
    pub fn execute(&self, col_words: &[u32; NUM_COL_INPUTS as usize], piece: u8) -> u32 {
        let mut regs = [0u32; NUM_REGISTERS as usize];

        for instr in &self.instructions {
            let s1 = Self::read_operand(&regs, col_words, piece, instr.src1);
            let s2 = Self::read_operand(&regs, col_words, piece, instr.src2);
            let d = instr.dst as usize;

            match instr.opcode {
                0 => regs[d] = s1.wrapping_add(s2), // ADD
                1 => regs[d] = s1.wrapping_sub(s2), // SUB
                2 => regs[d] = s1 & s2,             // AND
                3 => regs[d] = s1 | s2,             // OR
                4 => regs[d] = u32::from(s1 < s2),  // CMP_LT
                5 => {
                    // MUX: dst = (src1 != 0) ? src2 : dst
                    if s1 != 0 {
                        regs[d] = s2;
                    }
                }
                6 => regs[d] = instr.src1 as u32, // CONST_LO: dst = immediate (src1 field)
                7 => regs[d] = s1.min(s2),        // MIN
                8 => {}                           // NOP
                _ => {}
            }
            // Mask to VM_VALUE_BITS (8-bit) to match symbolic BV width.
            regs[d] &= 0xFF;
        }

        regs[0]
    }
}

impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, instr) in self.instructions.iter().enumerate() {
            writeln!(f, "  [{i:2}] {instr}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_lo() {
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 6, // CONST_LO
                dst: 0,
                src1: 42,
                src2: 0,
            }],
        };
        assert_eq!(prog.execute(&[0; 10], 0), 42);
    }

    #[test]
    fn test_read_col_word() {
        // R0 = C3
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 0, // ADD R0 <- C3 + R0 (R0=0, so R0=C3)
                dst: 0,
                src1: 8 + 3, // C3 = slot 11
                src2: 0,     // R0
            }],
        };
        let mut col_words = [0u32; 10];
        col_words[3] = 7;
        assert_eq!(prog.execute(&col_words, 0), 7);
    }

    #[test]
    fn test_read_piece_input() {
        // R0 = P (piece input at slot 18)
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 0, // ADD R0 <- P + R0 (R0=0, so R0=piece)
                dst: 0,
                src1: 18, // P = slot 18
                src2: 0,  // R0
            }],
        };
        assert_eq!(prog.execute(&[0; 10], 3), 3);
        assert_eq!(prog.execute(&[0; 10], 6), 6);
        assert_eq!(prog.execute(&[0; 10], 0), 0);
    }

    #[test]
    fn test_cmp_lt_and_mux() {
        let prog = Program {
            instructions: vec![
                Instruction {
                    opcode: 6, // CONST_LO R2 <- 5
                    dst: 2,
                    src1: 5,
                    src2: 0,
                },
                Instruction {
                    opcode: 4, // CMP_LT R1 <- C0 < C1
                    dst: 1,
                    src1: 8, // C0
                    src2: 9, // C1
                },
                Instruction {
                    opcode: 5, // MUX R0 <- (R1 != 0) ? R2 : R0
                    dst: 0,
                    src1: 1, // R1 (condition)
                    src2: 2, // R2 (value)
                },
            ],
        };

        let mut col_words = [0u32; 10];
        col_words[0] = 1;
        col_words[1] = 3;
        assert_eq!(prog.execute(&col_words, 0), 5); // C0 < C1 -> MUX fires

        col_words[0] = 5;
        col_words[1] = 2;
        assert_eq!(prog.execute(&col_words, 0), 0); // C0 >= C1 -> MUX doesn't fire
    }

    #[test]
    fn test_min() {
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 7, // MIN R0 <- C0, C1
                dst: 0,
                src1: 8,
                src2: 9,
            }],
        };
        let mut col_words = [0u32; 10];
        col_words[0] = 4;
        col_words[1] = 2;
        assert_eq!(prog.execute(&col_words, 0), 2);
    }

    #[test]
    fn test_sub_wrapping() {
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 1, // SUB R0 <- C0 - C1
                dst: 0,
                src1: 8,
                src2: 9,
            }],
        };
        let mut col_words = [0u32; 10];
        col_words[0] = 3;
        col_words[1] = 5;
        assert_eq!(prog.execute(&col_words, 0), 254); // 8-bit wrapping: 3 - 5 = 0xFE
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let instr = Instruction {
            opcode: 8,
            dst: 3,
            src1: 200,
            src2: 8,
        };
        let bits = instr.encode();
        let decoded = Instruction::decode(bits);
        assert_eq!(instr, decoded);
    }

    #[test]
    fn test_nop_does_nothing() {
        let prog = Program {
            instructions: vec![
                Instruction {
                    opcode: 6, // CONST_LO R0 <- 42
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
        assert_eq!(prog.execute(&[0; 10], 0), 42);
    }

    #[test]
    fn test_nop_with_various_dst() {
        let prog = Program {
            instructions: vec![
                Instruction {
                    opcode: 6,
                    dst: 0,
                    src1: 10,
                    src2: 0,
                },
                Instruction {
                    opcode: 6,
                    dst: 1,
                    src1: 20,
                    src2: 0,
                },
                Instruction {
                    opcode: 8, // NOP targeting R0
                    dst: 0,
                    src1: 17, // arbitrary operands
                    src2: 15,
                },
                Instruction {
                    opcode: 8, // NOP targeting R1
                    dst: 1,
                    src1: 8,
                    src2: 9,
                },
                Instruction {
                    opcode: 0, // ADD R0 <- R0 + R1
                    dst: 0,
                    src1: 0,
                    src2: 1,
                },
            ],
        };
        assert_eq!(prog.execute(&[0; 10], 0), 30); // 10 + 20
    }

    #[test]
    fn test_all_zeros_col_words() {
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 0, // ADD R0 <- C0 + C1
                dst: 0,
                src1: 8,
                src2: 9,
            }],
        };
        assert_eq!(prog.execute(&[0; 10], 0), 0);
    }

    #[test]
    fn test_max_col_words() {
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 0, // ADD R0 <- C0 + C1 (wrapping)
                dst: 0,
                src1: 8,
                src2: 9,
            }],
        };
        // Col words masked to 0xFF on read: 0xFF + 0xFF = 0x1FE masked to 0xFE
        assert_eq!(prog.execute(&[u32::MAX; 10], 0), 0xFE);
    }

    #[test]
    fn test_and_or_bitwise() {
        let prog = Program {
            instructions: vec![
                Instruction {
                    opcode: 6, // CONST_LO R0 <- 0b1100
                    dst: 0,
                    src1: 0b1100,
                    src2: 0,
                },
                Instruction {
                    opcode: 6, // CONST_LO R1 <- 0b1010
                    dst: 1,
                    src1: 0b1010,
                    src2: 0,
                },
                Instruction {
                    opcode: 2, // AND R2 <- R0 & R1
                    dst: 2,
                    src1: 0,
                    src2: 1,
                },
                Instruction {
                    opcode: 3, // OR R3 <- R0 | R1
                    dst: 3,
                    src1: 0,
                    src2: 1,
                },
                Instruction {
                    opcode: 0, // ADD R0 <- R2 + R3
                    dst: 0,
                    src1: 2,
                    src2: 3, // AND=0b1000=8, OR=0b1110=14, sum=22
                },
            ],
        };
        assert_eq!(prog.execute(&[0; 10], 0), 22);
    }

    #[test]
    fn test_cmp_lt_equal_values() {
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 4,
                dst: 0,
                src1: 8, // C0
                src2: 9, // C1
            }],
        };
        let col_words = [5, 5, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(prog.execute(&col_words, 0), 0); // 5 < 5 is false
    }

    #[test]
    fn test_mux_preserves_on_zero_condition() {
        let prog = Program {
            instructions: vec![
                Instruction {
                    opcode: 6,
                    dst: 0,
                    src1: 99,
                    src2: 0,
                },
                Instruction {
                    opcode: 6,
                    dst: 1,
                    src1: 0, // condition = 0
                    src2: 0,
                },
                Instruction {
                    opcode: 6,
                    dst: 2,
                    src1: 77, // value that should NOT be written
                    src2: 0,
                },
                Instruction {
                    opcode: 5, // MUX R0 <- (R1!=0) ? R2 : R0
                    dst: 0,
                    src1: 1,
                    src2: 2,
                },
            ],
        };
        assert_eq!(prog.execute(&[0; 10], 0), 99);
    }

    #[test]
    fn test_min_with_equal() {
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 7,
                dst: 0,
                src1: 8,
                src2: 9,
            }],
        };
        assert_eq!(prog.execute(&[3, 3, 0, 0, 0, 0, 0, 0, 0, 0], 0), 3);
    }

    #[test]
    fn test_min_with_zero() {
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 7,
                dst: 0,
                src1: 8,
                src2: 9,
            }],
        };
        assert_eq!(prog.execute(&[0, 5, 0, 0, 0, 0, 0, 0, 0, 0], 0), 0);
    }

    #[test]
    fn test_chained_operations() {
        // Compute (C0 + C1) - MIN(C2, C3)
        let prog = Program {
            instructions: vec![
                Instruction {
                    opcode: 0, // ADD R0 <- C0 + C1
                    dst: 0,
                    src1: 8,
                    src2: 9,
                },
                Instruction {
                    opcode: 7, // MIN R1 <- C2, C3
                    dst: 1,
                    src1: 10,
                    src2: 11,
                },
                Instruction {
                    opcode: 1, // SUB R0 <- R0 - R1
                    dst: 0,
                    src1: 0,
                    src2: 1,
                },
            ],
        };
        let col_words = [3, 4, 2, 5, 0, 0, 0, 0, 0, 0];
        // (3+4) - min(2,5) = 7 - 2 = 5
        assert_eq!(prog.execute(&col_words, 0), 5);
    }

    #[test]
    fn test_operand_out_of_range() {
        // Operand index >= NUM_OPERAND_SLOTS should read as 0
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 0, // ADD R0 <- ?20 + ?25
                dst: 0,
                src1: 20,
                src2: 25,
            }],
        };
        assert_eq!(prog.execute(&[1; 10], 0), 0); // both out of range -> 0+0=0
    }

    #[test]
    fn test_all_col_inputs_accessible() {
        for c_idx in 0..10u8 {
            let prog = Program {
                instructions: vec![Instruction {
                    opcode: 0, // ADD R0 <- C[c_idx] + R0
                    dst: 0,
                    src1: 8 + c_idx,
                    src2: 0,
                }],
            };
            let mut col_words = [0u32; 10];
            col_words[c_idx as usize] = 100 + u32::from(c_idx);
            assert_eq!(prog.execute(&col_words, 0), 100 + u32::from(c_idx));
        }
    }

    #[test]
    fn test_all_registers_writable() {
        for reg in 0..8u8 {
            let mut instrs = vec![Instruction {
                opcode: 6,
                dst: reg,
                src1: 50 + reg,
                src2: 0,
            }];
            if reg != 0 {
                instrs.push(Instruction {
                    opcode: 0, // ADD R0 <- R[reg] + R0
                    dst: 0,
                    src1: reg,
                    src2: 0,
                });
            }
            let prog = Program {
                instructions: instrs,
            };
            assert_eq!(prog.execute(&[0; 10], 0), 50 + u32::from(reg));
        }
    }

    #[test]
    fn test_empty_program() {
        let prog = Program {
            instructions: vec![],
        };
        assert_eq!(prog.execute(&[5; 10], 0), 0); // R0 starts at 0
    }

    #[test]
    fn test_opcode_display() {
        assert_eq!(format!("{}", Opcode::Nop), "NOP");
        assert_eq!(format!("{}", Opcode::Add), "ADD");
    }

    #[test]
    fn test_instruction_display() {
        let instr = Instruction {
            opcode: 8,
            dst: 0,
            src1: 0,
            src2: 0,
        };
        let s = format!("{instr}");
        assert!(s.contains("NOP"));
    }

    #[test]
    fn test_piece_and_col_words_combined() {
        // R0 = C0 + P
        let prog = Program {
            instructions: vec![Instruction {
                opcode: 0, // ADD R0 <- C0 + P
                dst: 0,
                src1: 8,  // C0
                src2: 18, // P
            }],
        };
        assert_eq!(prog.execute(&[10, 0, 0, 0, 0, 0, 0, 0, 0, 0], 5), 15);
    }
}
