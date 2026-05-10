# Tetris Program Synthesis

This binary is an empirical CEGIS experiment for synthesizing a compact Tetris policy as a small
bitvector VM program.

It is not currently a proof artifact. A successful run means the synthesized program survived the
configured finite simulation suite. It does not prove infinite play or Atlas completeness.

## Current Approach

The synthesizer searches for one global VM program:

```text
input:  board columns [u32; 10] + current piece index 0..6
output: global TetrisPiecePlacement index
```

The CEGIS loop is:

1. Seed examples from simple invariant-satisfying boards for all seven pieces.
2. Use Z3 to synthesize a VM program satisfying all known examples.
3. Run the candidate program through concrete Tetris simulations.
4. If a seed fails, extract recent `(board, piece)` counterexamples.
5. Add only new counterexamples to the same Z3 solver.
6. Repeat until the program survives or synthesis becomes `Unsat`/`Unknown`.

The solver is incremental for a fixed `--program-length`: program validity constraints are asserted
once, seed examples are asserted once, and later CEGIS rounds assert only newly discovered examples.
Changing program length requires a new solver because the symbolic program shape changes.

## VM Notes

The VM now reads full board columns instead of low bytes. This avoids aliasing high rows away.

Current instruction set:

```text
ADD
SUB
AND
OR
CMP_LT
MUX
CONST_LO
MIN
NOP
```

Values are `u32`. Instruction fields are still compact, and `CONST_LO` uses `src1` as an 8-bit
immediate. Symbolic validity intentionally allows `CONST_LO src1` values outside the operand-slot
range so Z3 can emit placement constants above slot 18.

The current generic VM is probably too weak for useful long-horizon policies. Z3 has to rediscover
Tetris features such as height, holes, roughness, valid placements, and placement outcomes from raw
bit operations. The likely next step is adding domain-specific opcodes such as `HEIGHT`, `POPCNT`,
`HOLES`, `ABS_DIFF`, `MAX`, `PIECE_BASE`, or a placement-scoring opcode.

## Running

On this machine, Z3 is installed through Homebrew. Export the library path before running tests or
the binary:

```sh
export Z3_LIB="$(brew --prefix z3)/lib"
export LIBRARY_PATH="$Z3_LIB:${LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$Z3_LIB:${DYLD_LIBRARY_PATH:-}"
```

Small smoke run:

```sh
RUST_LOG=tetris_program_synthesis=info cargo run -p tetris-playground --bin tetris_program_synthesis -- \
  --program-length 1 \
  --max-rounds 1 \
  --num-seeds 1 \
  --survival-target 1 \
  --z3-timeout-ms 1000
```

Larger empirical run:

```sh
RUST_LOG=tetris_program_synthesis=info cargo run --release -p tetris-playground --bin tetris_program_synthesis -- \
  --program-length 12 \
  --max-rounds 20 \
  --num-seeds 8 \
  --survival-target 1000 \
  --z3-timeout-ms 60000 \
  --log-z3-stats
```

If synthesis returns `Unknown` with reason `timeout`, verification never ran. Increase
`--z3-timeout-ms`, reduce initial difficulty, or improve the VM opcode set.

## Z3 Logging

There are three useful levels of Z3 observability.

Rust-side Z3 statistics:

```sh
--log-z3-stats
```

This prints `solver.get_statistics()` after each synthesis check.

Inline native Z3 progress:

```sh
--z3-verbose 1
```

This prints Z3's own progress lines to stderr, usually as `(sat.stats ...)`. These are native Z3
logs, not Rust logs.

Native Z3 interaction log on disk:

```sh
--z3-log-file artifacts/output/z3-program-synthesis.log
```

This writes the low-level Z3 API/formula interaction log. It can be very large and low-level, but it
is the reliable native Z3 log file.

SMT-LIB2 query dumps:

```sh
--dump-smt2-dir artifacts/output/program_synthesis_smt2
```

This writes human-inspectable SMT2 snapshots. For debugging, this is usually easier to read than the
native Z3 interaction log.

Example with all useful logging:

```sh
RUST_LOG=tetris_program_synthesis=info cargo run --release -p tetris-playground --bin tetris_program_synthesis -- \
  --program-length 12 \
  --max-rounds 20 \
  --num-seeds 8 \
  --survival-target 1000 \
  --z3-verbose 1 \
  --log-z3-stats \
  --z3-log-file artifacts/output/z3-program-synthesis.log \
  --dump-smt2-dir artifacts/output/program_synthesis_smt2 \
  --z3-timeout-ms 60000
```

## Round Logs

Each CEGIS round reports:

```text
newly asserted examples
total asserted examples
synthesized program
verification summary over all seeds
mean/min/max pieces survived
invalid-placement and loss counts
first failure seed/kind/step
new counterexamples added
```

No verification summary is printed if Z3 fails to produce a candidate program. In that case the run
ends at synthesis with either `Unsat` or `Unknown`.

## Result Meanings

`Sat` means Z3 found a VM program that satisfies all currently asserted examples.

`Unsat` means no program of the current fixed length can satisfy the current example set. Since CEGIS
only adds constraints, this run cannot recover without increasing `--program-length` or changing the
VM/opcode semantics.

`Unknown` usually means timeout. More examples will not necessarily help. Increase
`--z3-timeout-ms`, reduce constraints, or improve the VM.

`Survived` means the program survived the configured empirical verification suite. It is not a proof
of infinite play.

## Current Lessons

- Z3 can spend all available time in round 0 before any concrete verification happens.
- `--num-seeds` and `--survival-target` matter only after synthesis produces a candidate.
- Incremental solving avoids rebuilding the solver across CEGIS rounds, but it does not solve the
  initial-query complexity problem.
- Full-column inputs are necessary for correctness, but they make raw bit-level synthesis harder.
- Domain-specific VM instructions are likely required to make the search practical.

## Verification Commands

```sh
cargo fmt --all -- --check
cargo check -p tetris-playground --bin tetris_program_synthesis --tests
cargo clippy -p tetris-playground --bin tetris_program_synthesis -- -D warnings
LIBRARY_PATH="$(brew --prefix z3)/lib" cargo test -p tetris-playground --bin tetris_program_synthesis
```
