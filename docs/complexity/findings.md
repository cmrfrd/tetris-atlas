# Complexity classification of the tetris-atlas variant

**Date:** 2026-07-15 · **Branch:** tetris-atlas-puct (dirty, base 65ae016)
**Question:** what complexity class does our constrained Tetris occupy — skyline hard drop
(162 placements, rotation+column pre-chosen, no kicks/tucks/slides), 7-bag randomizer,
no hold, no lookahead, loss iff height > m *after* clears?
**Method:** 11-agent verified workflow — 3 paper readers (Temprano 2015, FUN 2024, FUN 2026),
3 theory provers, 1 gadget-design attempt, 4 adversarial verifiers (all verdicts
high-confidence). Raw agent outputs and machine-verification scripts in `raw/`.

## The problem family

NP-completeness is a statement about *offline* decision problems; our online game needs a
family. All boards generalized to n columns × m rows, board given as an explicit bitmap.

| | Problem | Input | Question |
|---|---|---|---|
| **P1** | OFFLINE-HD | (n, m, board, explicit sequence w) | ∃ placements surviving all of w? (variant: also clear the board) |
| **P1′** | OFFLINE-HD-BAG | P1 with w constrained 7-bag-legal | same |
| **P2** | ONLINE-HORIZON | (n, m, board, bag, piece, N unary) | player guarantees N pieces vs adaptive bag adversary? |
| **P3** | ATLAS-MEMBERSHIP | (n, m, board, bag, piece) | player guarantees survival **forever**? (= membership in the gfp safe set) |
| **P4** | EMPTY-DIMS | (n, m) only | empty board, fresh bag: survive forever? |

## Classification (all claims adversarially verified)

| Problem | Upper bound | Lower bound | Status |
|---|---|---|---|
| P1 survival | NP (proven) | NP-hard, even with only {I,O} pieces | **NP-complete** — FUN 2024 Thm 5.1 transfers verbatim |
| P1 clearing | NP (proven) | NP-hard ({I,O,T} and 5-type); #P-hard ({I,J},{I,L},{I,O},{I,T}) | **NP-complete** — Temprano 2015 + FUN 2024 Prop 4.1-H transfer |
| P1′ | NP (proven) | none | **OPEN** — our construction attempt refuted; missing primitive precisely characterized (below) |
| P2 | PSPACE (proven) | none | **OPEN**; NP-hardness plausible via P1′; PSPACE-hardness needs a ∀-bit fan-out gadget |
| P3 | EXPTIME (proven) | none | **OPEN** between PSPACE and EXPTIME; hardness *gated behind M2* (below) |
| P4 | decidable; EXPTIME (unary dims) / 2-EXPTIME (binary) | structurally excluded | wrong home for hardness: sparse language; NP-hard ⇒ P=NP (Mahaney) |

**Bottom line.** The offline analog of our variant (drop the bag constraint) is genuinely
NP-complete in our exact 162-action model — the DHLN 2002 result does *not* give this (its
gadgets need mid-fall slides; it left hard drop open), but Temprano 2015 and FUN 2024's
hard-drops-only results close it. Adding the bag constraint (P1′) is open — we attempted the
construction and the attempt was refuted (informatively). The online game itself is not an
NP-type question at all: it is a safety game, PSPACE/EXPTIME-bounded, whose lower-bound
theory is entangled with the Atlas conjecture.

## Proven theorems (this session; proofs in `raw/theory_upper.md`, `raw/theory_structural.md`, verifier repairs in `raw/verdict_0.md`, `raw/verdict_1.md`)

**Upper bounds.**
- **A (NP):** P1/P1′ ∈ NP; certificate = the placement list, O(|w|·log n) bits, O(|w|·nm) verification; bag-legality is a linear scan.
- **B (PSPACE):** P2 solvable by depth-N minimax DFS; space O(N·(nm+log n+log N)); branching ≤ 4n (player) / ≤ 7 (adversary); N unary makes this polynomial.
- **C (EXPTIME):** P3 is a safety game on ≤ 2^{nm}·2^7·7 + O(2^{nm+7}) states; attractor/gfp computation, positional determinacy, and W = νX.(Safe ∩ CPre(X)) all proven from scratch (Lemmas C1–C3). Both players have memoryless deterministic optimal strategies.
- **D (adversarial ≡ almost-sure):** the player survives adversarially from s ⟺ some strategy survives with probability 1 under the uniform random bag ⟺ the positional strategy survives *surely*. From the adversary's region every strategy dies with probability ≥ 7^{−K} (K = attractor depth); the correct dichotomy is "a.s.: die or eventually absorb into the safe region W" — note a.s. *death* from the losing region is FALSE (draws can rescue the player into W). This licenses the project's adversarial gfp as the right object for a probability-1 claim.
- **E (conditional PSPACE route for P3):** if every yes-instance admits a poly(nm)-size *circuit* inductive invariant (post-fixed point of F containing s₀), then P3 ∈ Σ₂ᵖ ⊆ PSPACE — one-sided certificates suffice since invariant-checking is sound (verifier-improved from Σ₃ᵖ). Empirical carrier evidence (>10⁶-state closed sets, no small ones found) points against easy succinctness. The naive gfp cannot land in PSPACE: poly-space alternation *characterizes* EXPTIME (APSPACE = EXPTIME).

**Structural theorems.**
- **Row monotonicity:** the dynamics never reference the ceiling; a surviving strategy for n×m survives *verbatim* on n×m′, m′ ≥ m, with identical trajectories (holds on the full state space — the verifier strengthened the original statement). Hence per width n a threshold m*(n) ∈ ℕ∪{∞} with P4 = {(n,m) : m ≥ m*(n)}.
- **Width non-monotonicity of transfer:** clears are full-row AND across all columns, so a width-n strategy lifted to width n+1 never clears and dies within ⌊nm/4⌋+1 pieces. Whether survivability itself is monotone in width is open in both directions.
- **Where the complexity lives:** P4 is the epigraph of one integer function; unary-encoded it is sparse, so NP-hardness would give P = NP (Mahaney) and even Turing-hardness collapses PH (Karp–Lipton). Hardness for this ruleset *must* take the state (P2/P3) or the sequence (P1/P1′) as input. Caveat: this does not make P4 easy — deciding "m*(10) ≤ 20" *is* the Atlas project, and even the computability of n ↦ m*(n) is open (certifying m*(n) = ∞ has no known finite reduction).
- **Adversary information rate:** the bag adversary injects log₂(7!)/7 ≈ 1.757 bits/piece, and every 7k-window contains each type k±1 times — Burgiel-style S/Z floods are bag-illegal. For P2/P3 the instance must be encoded entirely in board geometry (fixed 7-piece alphabet); for P1′ the within-bag *order* remains a Θ(1.757·|w|)-bit input channel (verifier-corrected).

## Transfer matrix (what published hardness survives our movement model)

Details and quotes in `raw/reader_*.md`, `raw/theory_composition.md`.

| Construction | Skyline hard drop | Bag-legal | Verdict for us |
|---|---|---|---|
| DHLN 2002/03 (all objectives) | ✗ needs mid-fall slides into roofed notches | ✗ 5 types, long I-runs | nothing transfers; hard-drop model posed open (their §8) |
| Temprano 2015 clearing (5 types) | ✓ Brzustowski model = ours on his boards | ✗ | **transfers**; soundness independently re-verified (counting reconstructed airtight); recommend one afternoon of engine enumeration of his figure-based Lemmas 5–10 |
| FUN 2024 Thm 5.1 survival {I,O} (hard-drops-only) | ✓ "no overhangs in any of the buckets"; kicks inert | ✗ runs of O's and I's | **transfers verbatim** — the strongest anchor |
| FUN 2024 Prop 4.1-H clearing {I,·} + #P | ✓ authors: "do not require the kicking system of SRS" | ✗ | **transfers**, incl. #P-hardness |
| FUN 2024 other 2-piece sets, ASP | ✗ SRS spin chains through covered necks | ✗ | breaks; those alphabets open in our model |
| FUN 2026 one-piece + 7-bag Thm 5.2 | ✗ I-pieces corner-turn through covered corridors via SRS end-pivot kicks | ✓ (the only bag-legal hardness in the literature) | breaks; the per-bag absorber *blueprint* (6 dedicated slots/stratum, redundant shafts, one useful piece per bag) ports |
| FUN 2026 Thm 3.1 (positive: 1×k, top k−1 rows empty) | ✓ | – | ports as a positive result: I-only near-empty-top instances are easy — I-only skyline hardness would need filled top rows |

## The P1′ attempt and its refutation (`raw/theory_gadget_v2.md`, `raw/verdict_3.md`)

Target: NP-hardness of bag-legal skyline-hard-drop clearing (would resolve FUN 2024's open
problem p.23 in the hard-drop model, complementing FUN 2026's SRS-only resolution).

**What was built (machine-verified, python skyline simulator matching engine semantics):**
per-type wells with hole-free periodic self-tilings for all 7 pieces (MV1–MV6, verified in
the engine itself by `cargo test` probes); a width-4 **ratchet** where S has exactly one
flush placement, then Z exactly one, re-arming itself (MV7); a uniform close/re-arm ops
chain (MV8); terminal fixes and anvils (MV9/MV10); Temprano-style no-clear gate + exact
cell-budget algebra, all arithmetic machine-checked on a concrete s=2 instance.

**Why it dies (verifier, high confidence):** the *concurrent-arming split cheat*. Nothing
forces at most one bucket to be armed at a time; a cheater arms two buckets, splits one
3-Partition item across them at unit granularity, and the design's own uniformity lemma
(MV8: all op-cells column-uniform) lets the I-count algebra compensate exactly — so every
instance whose totals balance (i.e., *all* of them, since fractional 3-Partition is always
feasible) becomes a YES. NO-instances map to YES; the reduction proves nothing. The
proposed mirror-alternation repair fails the same way. Three earlier in-session designs
died at the same point.

**The precise open kernel:** skyline column-independence makes "receptive" a locally cheap,
reversible, concurrency-friendly property. Count-based/flow-based forcing cannot serialize
it. P1′ hardness needs a **skyline-compatible exclusivity primitive** — a structure that
must be geometrically *dismantled* in one region before it can be *assembled* in another,
with exactly-balanced piece cost and O(1) terminal slack (the current gate leaks 6R junk
cells of slack, obligation O3). This is now a well-posed gadget problem, cheap to falsify
with the same machine-checking harness. What survives for any future attempt: the wells,
ratchet, budget algebra, and completeness machinery.

## P3 hardness is gated behind M2

The sharpest project-specific finding: any EXPTIME/PSPACE-hardness reduction *into* P3 must
contain YES-branches — gadget states from which adversarial survival is provable ("havens").
In our model no such state has ever been exhibited; that is exactly the open M2 milestone.
Until M2 produces a certified adversarial closed cycle, no reduction can certify its
intended play; conversely if the adversarial safe set were empty at all dimensions, P3
would be trivially decidable (always NO). The complexity theory of the Atlas variant is
entangled with the truth of the Atlas conjecture itself.

## Ranked open targets (from `raw/theory_composition.md`, post-refutation)

1. **Write-up of the inherited results** — P1 survival/clearing NP-complete in the skyline
   model (FUN 2024-H + Temprano), with engine machine-verification of Temprano's finite
   lemmas. Certainty high, novelty low but citable.
2. **P1′ NP-completeness** — needs the exclusivity primitive (one real idea, then
   MV-style enumeration). Directly answers FUN 2024's open problem in the hard-drop model.
3. P1′ survival variant (waste-budget flood) — small delta on 2.
4. P2 NP-hardness — order-robustify 2 (type-dedicated wells never see within-bag order).
5. P2 PSPACE-hardness — needs a ∀-bit fan-out/consistency-comparator gadget (QBF bits =
   within-bag order choices; a placed piece can't be re-read across walls).
6. P3 EXPTIME-hardness — speculative; gated behind M2 + would need renewable (clear-reset)
   forced gadgets, which no published Tetris reduction has.

## Corrections to project docs

- CLAUDE.md says "117 precomputed `TetrisPiecePlacement` values"; the engine's
  `NUM_PLACEMENTS` is **162** (rotations O=1, I/S/Z=2, T/L/J=4; columns 9+17+17+17+34·3;
  `tetris.rs:808`, verified twice independently this session). All session proofs use only
  "≤ 4n actions per piece", so nothing depends on the constant.
- Engine as model witness: the u32 columns make the engine a faithful realization of the
  abstract n×m model only for m ≤ 28 (pieces spawn at bit rows 28–31); canonical 10×20 fine.

## References

- Demaine, Hohenberger, Liben-Nowell, *Tetris is Hard, Even to Approximate*, MIT TR 2002
  (arXiv:cs/0210020); journal: Breukelaar et al., IJCGA 14(1–2), 2004.
- Temprano, *Complexity of a Tetris variant*, arXiv:1506.07204 (2015).
- MIT Hardness Group, Demaine, Hall, Li, *Tetris with Few Piece Types*, FUN 2024
  (arXiv:2404.10712) — "hard drops only" mode = our model; 7-bag posed open (p.23).
- MIT Hardness Group (Brunner, Demaine, Hendrickson, Li), *Tetris Is Hard with Just One
  Piece Type*, FUN 2026 (arXiv:2603.09958) — Thm 5.2: bag-legal clearing NP-hard under SRS.
- Asif et al., *Tetris is NP-hard even with O(1) rows or columns*, JIP 28, 2020.
- Gehnen, Venier, *Tetris Is Not Competitive*, FUN 2024 — online non-competitiveness.
- Chandra, Kozen, Stockmeyer, *Alternation*, JACM 28(1), 1981 (APSPACE = EXPTIME).
- Mahaney, *Sparse complete sets for NP*, JCSS 25(2), 1982.
