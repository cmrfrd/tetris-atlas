import Mathlib
import Proofs.Board
import Proofs.Placement
import Proofs.SafeSet
import Proofs.Experiments.WqoCarrier
import Proofs.Experiments.HoleyCarrier
import Proofs.Experiments.HoleDebt

/-!
# The energy-game framing — headroom, capacity conservation, and its monotonicity

Reframes controlled Tetris as a **lower-bounded energy game**. Define **headroom**
`headroom = cols·rows − surfaceArea` (the unused capacity above the stack). Survival ⟺ the
player can keep `headroom ≥ 0` forever against the adversary's pieces; energy games are
determined with *memoryless* strategies, so a winning strategy is a lookup table on
`(surface, bag)` — i.e. the Atlas is, without loss of generality, history-free.

This file lays the energy foundation on top of `HoleDebt`:

* **Capacity conservation** (`capacity_conservation`): for an in-field board,
  `headroom + debt + mass = cols·rows`. The board's total capacity splits into three
  ledgers — empty space above the stack (`headroom`), buried empties (`debt`), and filled
  cells (`mass = card`). This is the energy-game's master identity.
* **headroom is the energy**: placement *spends* it (`headroom_place_le`) and line clears
  *refund* it (`headroom_le_clearLines`) — exactly the energy-game weight structure
  (placement = cost edges, clears = reward edges, topping out = energy `< 0`).

Backlog note (recorded): the per-piece bound `debt(place) ≤ debt + 3` is FALSE — a single
piece on a jagged surface buries many cells (e.g. a horizontal I on heights `[10,0,0,0]`
leaves a deep overhang). Debt growth per move is roughness-dependent, which is precisely why
the bounded controller must control roughness. The clean invariant is the conservation law
below, not a constant per-piece bound.
-/

namespace Tetris.EnergyGame

open Tetris Tetris.WqoCarrier Tetris.HoleyCarrier Tetris.HoleDebt

/-- **Headroom**: unused capacity above the stack, `cols·rows − surfaceArea`. The energy of
the energy game — the player keeps it non-negative (never tops out). -/
def headroom (cfg : GameConfig) (b : Board) : ℕ :=
  cfg.cols * cfg.rows - surfaceArea cfg b

/-- Board energy is at most total capacity for an in-field board: every column height is at
most `rows`, so `surfaceArea = Σ colHeight ≤ cols·rows`. -/
theorem surfaceArea_le_capacity {cfg : GameConfig} {b : Board}
    (h : ∀ j, b.colHeight j ≤ cfg.rows) :
    surfaceArea cfg b ≤ cfg.cols * cfg.rows := by
  unfold surfaceArea
  calc ∑ j ∈ Finset.range cfg.cols, b.colHeight j
      ≤ ∑ _j ∈ Finset.range cfg.cols, cfg.rows := Finset.sum_le_sum (fun j _ => h j)
    _ = cfg.cols * cfg.rows := by rw [Finset.sum_const, Finset.card_range, smul_eq_mul]

/-- **Capacity conservation (the master energy identity):** for an in-field, well-formed
board, `headroom + debt + mass = cols·rows`. Total capacity is partitioned into empty space
above the stack, buried holes, and filled cells. -/
theorem capacity_conservation {cfg : GameConfig} {b : Board} (hwf : Board.WF cfg b)
    (h : ∀ j, b.colHeight j ≤ cfg.rows) :
    headroom cfg b + debt cfg b + b.card = cfg.cols * cfg.rows := by
  unfold headroom
  have hsa := debt_add_card_eq_sum_colHeight hwf
  have hle := surfaceArea_le_capacity h
  omega

/-- **Placement spends headroom.** A hard drop only raises column heights, so `surfaceArea`
grows and headroom shrinks — a cost edge in the energy game. -/
theorem headroom_place_le (cfg : GameConfig) (b : Board) (pl : Placement) :
    headroom cfg (pl.place b) ≤ headroom cfg b := by
  unfold headroom
  have := surfaceArea_le_place cfg b pl
  omega

/-- **Line clears refund headroom.** Clears only lower column heights, so `surfaceArea`
falls and headroom grows — a reward edge in the energy game. -/
theorem headroom_le_clearLines (cfg : GameConfig) (b : Board) :
    headroom cfg b ≤ headroom cfg (Board.clearLines cfg b) := by
  unfold headroom
  have := surfaceArea_clearLines_le cfg b
  omega

/-! ## The hole-free fiber is the surface arena

The energy game lives on surfaces. The bridge: when `debt = 0` every column is a *gapless*
stack, so a hole-free board is completely determined by its column-height vector. Hence the
hole-free fiber is in bijection with the surface space — the finite arena on which the
surface automaton (`SurfaceFiber`) and the energy (`headroom`) operate. -/

/-- **Hole-free columns are gapless:** for a board with `debt = 0`, a valid column is filled
in exactly the rows below its height. (`debt = 0` ⇒ each `colHoles = 0` ⇒ filled-count =
height ⇒ the filled rows are all of `[0, colHeight)`.) -/
theorem holeFree_filled_iff {cfg : GameConfig} {b : Board} (hd : debt cfg b = 0)
    {j : ℕ} (hj : j < cfg.cols) (r : ℕ) :
    (j, r) ∈ b ↔ r < b.colHeight j := by
  have hcolHoles : colHoles b j = 0 := by
    unfold debt at hd
    exact (Finset.sum_eq_zero_iff.mp hd) j (Finset.mem_range.mpr hj)
  have hcard : (b.colRows j).card = b.colHeight j := by
    unfold colHoles at hcolHoles
    have hle := card_colRows_le_colHeight b j
    omega
  have hsub : b.colRows j ⊆ Finset.range (b.colHeight j) := by
    intro x hx
    rw [Finset.mem_range]
    have hx1 : x + 1 ≤ b.colHeight j := by unfold Board.colHeight; exact Finset.le_sup hx
    omega
  have heq : b.colRows j = Finset.range (b.colHeight j) :=
    Finset.eq_of_subset_of_card_le hsub (by rw [Finset.card_range]; omega)
  have hmem : (j, r) ∈ b ↔ r ∈ b.colRows j := by
    simp only [Board.colRows, Finset.mem_image, Finset.mem_filter]
    constructor
    · intro h; exact ⟨(j, r), ⟨h, rfl⟩, rfl⟩
    · rintro ⟨x, ⟨hxb, hxj⟩, hxr⟩
      have hxe : x = (j, r) := Prod.ext hxj hxr
      exact hxe ▸ hxb
  rw [hmem, heq, Finset.mem_range]

/-- **Hole-free boards are determined by their surface:** two hole-free, well-formed boards
with equal column heights are equal. So the hole-free fiber injects into the surface space
(the energy-game arena), one board per height vector. -/
theorem holeFree_ext {cfg : GameConfig} {b β : Board}
    (hb : debt cfg b = 0) (hβ : debt cfg β = 0)
    (hwfb : Board.WF cfg b) (hwfβ : Board.WF cfg β)
    (h : ∀ j, b.colHeight j = β.colHeight j) : b = β := by
  ext ⟨j, r⟩
  by_cases hj : j < cfg.cols
  · rw [holeFree_filled_iff hb hj, holeFree_filled_iff hβ hj, h]
  · constructor
    · intro hmem; exact absurd (hwfb (j, r) hmem) hj
    · intro hmem; exact absurd (hwfβ (j, r) hmem) hj

/-- **The hole-free fiber is closed under line clears.** Clearing a gapless stack lowers
columns gaplessly, so no holes appear: `debt = 0 ⇒ debt(clearLines) = 0`. Immediate from
`clearLines_debt_le` (debt is non-increasing, and `0` is the floor). So a controller staying
hole-free only has to avoid *creating* holes on placement — clears never break the invariant. -/
theorem holeFree_clearLines {cfg : GameConfig} {b : Board} (hwf : Board.WF cfg b)
    (hd : debt cfg b = 0) : debt cfg (Board.clearLines cfg b) = 0 :=
  Nat.le_zero.mp (hd ▸ clearLines_debt_le hwf)

/-! ## The loss boundary is `maxHeight`, not total headroom

Subtlety (recorded): topping out is a *per-column max* condition — one tall column loses even
if total `surfaceArea` is small. So total `headroom ≥ 0` is necessary but NOT sufficient for
survival. The loss certificate is `maxHeight ≤ rows`. This splits the energy game into two
quantities: `maxHeight` (the safety boundary, a sup) and `surfaceArea`/`headroom` (the
progress/clearing measure, a sum). A controller survives by keeping `maxHeight ≤ rows`, which
requires bounding *roughness* so the sum-bound controls the max-bound. -/

/-- The maximum column height across the valid columns. Loss ⟺ this exceeds `rows`. -/
def maxHeight (cfg : GameConfig) (b : Board) : ℕ :=
  (Finset.range cfg.cols).sup b.colHeight

/-- A well-formed board has height `0` outside its valid columns. -/
theorem colHeight_eq_zero_of_cols_le {cfg : GameConfig} {b : Board} {j : ℕ}
    (hwf : Board.WF cfg b) (hj : cfg.cols ≤ j) : b.colHeight j = 0 := by
  have hempty : b.colRows j = ∅ := by
    rw [Finset.eq_empty_iff_forall_notMem]
    intro r hr
    simp only [Board.colRows, Finset.mem_image, Finset.mem_filter] at hr
    obtain ⟨x, ⟨hxb, hxj⟩, _⟩ := hr
    have hlt := hwf x hxb
    rw [hxj] at hlt
    omega
  unfold Board.colHeight
  rw [hempty]
  simp

/-- **The loss boundary as a single scalar:** a well-formed board is non-lost iff its
`maxHeight` is at most `rows`. This is the safety objective of the energy game. -/
theorem not_isLost_iff_maxHeight_le {cfg : GameConfig} {b : Board} (hwf : Board.WF cfg b) :
    ¬ Board.isLost cfg b ↔ maxHeight cfg b ≤ cfg.rows := by
  unfold maxHeight
  rw [Finset.sup_le_iff]
  constructor
  · intro hnl j _
    rw [Board.not_isLost_iff_forall_row_lt] at hnl
    exact Board.colHeight_le_rows_of_in_field hnl j
  · intro hsup
    apply Board.not_isLost_of_colHeight_le
    intro j
    by_cases hj : j < cfg.cols
    · exact hsup j (Finset.mem_range.mpr hj)
    · rw [colHeight_eq_zero_of_cols_le hwf (Nat.le_of_not_lt hj)]
      exact Nat.zero_le _

/-! ## Wiring to the finish line: a `maxHeight`-bounded controller suffices

Repackages the codebase reduction `tetrisSolvableValid_of_height_bounded_invariant` so the
safety obligation is the single scalar `maxHeight ≤ rows` (instead of a per-column bound).
This leaves the *only* remaining obligation as the controller's closure `hstep` — the
energy-game's strategy. Everything upstream (init, safety) is discharged here. -/

/-- **Survival from a `maxHeight`-bounded closed invariant.** If `S` contains `init`, every
state in `S` is well-formed with `maxHeight ≤ rows`, and `S` is closed under some valid
placement for every drawable piece, then Tetris is solvable. The height obligation of the
underlying reduction is discharged from `maxHeight ≤ rows` via `Finset.le_sup` (valid
columns) and `colHeight_eq_zero_of_cols_le` (the rest). -/
theorem tetrisSolvableValid_of_maxHeight_invariant
    (S : Set GameState) (hinit : GameState.init ∈ S)
    (hwf : ∀ g ∈ S, Board.WF GameConfig.standard g.board)
    (hmax : ∀ g ∈ S, maxHeight GameConfig.standard g.board ≤ GameConfig.standard.rows)
    (hstep : ∀ g ∈ S, ∀ p, p ∈ g.bag →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        adversarialStep GameConfig.standard g p pl ∈ S) :
    TetrisSolvableValid := by
  refine tetrisSolvableValid_of_height_bounded_invariant S hinit ?_ hstep
  intro g hg j
  by_cases hj : j < GameConfig.standard.cols
  · calc g.board.colHeight j
        ≤ maxHeight GameConfig.standard g.board := by
          unfold maxHeight; exact Finset.le_sup (Finset.mem_range.mpr hj)
      _ ≤ GameConfig.standard.rows := hmax g hg
  · rw [colHeight_eq_zero_of_cols_le (hwf g hg) (Nat.le_of_not_lt hj)]
    exact Nat.zero_le _

#print axioms capacity_conservation
#print axioms headroom_place_le
#print axioms headroom_le_clearLines
#print axioms holeFree_filled_iff
#print axioms holeFree_ext
#print axioms not_isLost_iff_maxHeight_le
#print axioms tetrisSolvableValid_of_maxHeight_invariant

/-! ## Make-or-break (item 4, K=0 case): strict hole-free invariants are IMPOSSIBLE

The S-piece on flat ground always overhangs — both orientations leave a covered empty — and
a single piece (4 cells) can never complete a row, so no clear fires. Hence from the empty
board EVERY valid S placement creates a hole. The adversary draws S from the `debt = 0` init
state and forces `debt ≥ 1` no matter how the player responds: a strictly hole-free invariant
containing `init` cannot exist. **Numbers-backed floor: the K=0 sub-path is impossible; the
controller must carry a transient-hole budget K ≥ 1** (matching the empirical S-bootstrap
hole). This localizes the make-or-break to the K ≥ 1 regime. -/

/-- Every valid S placement on the empty board creates a hole (`debt ≥ 1`). Verified by
`native_decide` over the full finite set of valid S placements (`allValidFor`). -/
theorem S_on_empty_forces_debt :
    ∀ pl ∈ Placement.allValidFor GameConfig.standard Piece.S,
      1 ≤ debt GameConfig.standard
        (Placement.applyStep GameConfig.standard Board.empty pl) := by
  native_decide

/-- **K=0 is impossible.** No set of states with `debt ≡ 0` that contains `init` can be
closed under adversarial steps: drawing S from `init` (full bag ⇒ S drawable) forces
`debt ≥ 1` under every valid placement (`S_on_empty_forces_debt`), contradicting closure in a
`debt = 0` set. The hole-free energy sub-game has no solution. -/
theorem no_holeFree_invariant
    (S : Set GameState) (hinit : GameState.init ∈ S)
    (hdebt0 : ∀ g ∈ S, debt GameConfig.standard g.board = 0)
    (hstep : ∀ g ∈ S, ∀ p, p ∈ g.bag →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        adversarialStep GameConfig.standard g p pl ∈ S) :
    False := by
  have hSbag : Piece.S ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full _
  obtain ⟨pl, hpiece, hvalid, hmem⟩ := hstep GameState.init hinit Piece.S hSbag
  have hpl_mem : pl ∈ Placement.allValidFor GameConfig.standard Piece.S :=
    (Placement.mem_allValidFor _ _ _).mpr ⟨hpiece, hvalid⟩
  have hpleq : ({pl with piece := Piece.S} : Placement) = pl := by rw [← hpiece]
  have hboard : (adversarialStep GameConfig.standard GameState.init Piece.S pl).board
      = Placement.applyStep GameConfig.standard Board.empty pl := by
    rw [show (adversarialStep GameConfig.standard GameState.init Piece.S pl).board
          = Placement.applyStep GameConfig.standard GameState.init.board
              {pl with piece := Piece.S} from rfl,
      GameState.init_board, hpleq]
  have hd0 := hdebt0 _ hmem
  rw [hboard] at hd0
  have key := S_on_empty_forces_debt pl hpl_mem
  omega

/-- Z, like S, forces a hole on flat ground: every valid Z placement on the empty board
yields `debt ≥ 1`. -/
theorem Z_on_empty_forces_debt :
    ∀ pl ∈ Placement.allValidFor GameConfig.standard Piece.Z,
      1 ≤ debt GameConfig.standard
        (Placement.applyStep GameConfig.standard Board.empty pl) := by
  native_decide

/-- **The other five pieces can be placed hole-free on flat ground.** Each of O, I, T, L, J
has a valid placement on the empty board with `debt = 0`. Together with
`S_on_empty_forces_debt`/`Z_on_empty_forces_debt`, this pins the debt sources to exactly
`{S, Z}`: on a flat surface, only S and Z force a hole. So per bag the *forced* debt is
bounded by the number of S/Z draws (≤ 2), and the `K ≥ 1` controller's job reduces to
absorbing S and Z (and discharging via clears) — the I/O/T/L/J draws can stay hole-free. -/
theorem nonSZ_has_holeFree_placement :
    ∀ p ∈ ({Piece.O, Piece.I, Piece.T, Piece.L, Piece.J} : Finset Piece),
      ∃ pl ∈ Placement.allValidFor GameConfig.standard p,
        debt GameConfig.standard
          (Placement.applyStep GameConfig.standard Board.empty pl) = 0 := by
  native_decide

/-- **S is hole-free on a stepped surface.** On the board `{(2,0)}` (a single one-cell step:
heights `[0,0,1]`), the S-piece has a valid placement landing flush with `debt = 0`. Contrast
`S_on_empty_forces_debt` (flat ⇒ every S placement holes): S-safety is a *roughness*
condition — the controller absorbs S exactly by maintaining an S-notch. This is the concrete
mechanism the `K ≥ 1` controller uses to keep the forced S/Z debt transient. -/
theorem S_holeFree_on_step :
    ∃ pl ∈ Placement.allValidFor GameConfig.standard Piece.S,
      debt GameConfig.standard
        (Placement.applyStep GameConfig.standard {((2 : ℕ), (0 : ℕ))} pl) = 0 := by
  native_decide

/-- **The certificate needs height ≥ 2.** Every valid O placement on the empty board reaches
`maxHeight = 2` (the 2×2 O occupies two rows and cannot clear). So any closed invariant
containing `init` must admit states with `maxHeight ≥ 2` (adversary draws O). Together with
`no_holeFree_invariant` (needs `K ≥ 1`), these are the proven lower bounds on the energy-game
certificate: `maxHeight ≥ 2`, `debt-budget ≥ 1`. -/
theorem O_on_empty_forces_height :
    ∀ pl ∈ Placement.allValidFor GameConfig.standard Piece.O,
      2 ≤ maxHeight GameConfig.standard
        (Placement.applyStep GameConfig.standard Board.empty pl) := by
  native_decide

#print axioms S_on_empty_forces_debt
#print axioms no_holeFree_invariant
#print axioms Z_on_empty_forces_debt
#print axioms nonSZ_has_holeFree_placement
#print axioms S_holeFree_on_step
#print axioms O_on_empty_forces_height

/-! ## K=1 viability on the hardest interaction: consecutive S then Z

The only debt-sources are `{S, Z}` (iter7), and they demand opposite notches — so the
worst case for a budget-1 controller is the adversary playing S and Z back-to-back (no
repair piece between). This checks both orders from the empty board: can the player keep
`debt ≤ 1` through both hole-forcers? A `native_decide` `∃∃` over the (finite) placement
choices for each order. A positive result is strong evidence that `K = 1` absorbs a whole
bag (the other five pieces stay hole-free, iter7); a negative result would raise the budget
floor to `K ≥ 2`. -/

/-- **The player absorbs consecutive S,Z (both orders) from empty with `debt ≤ 1`.** For
each of the orders `S→Z` and `Z→S`, there exist valid placements keeping the buried-hole
count at most one throughout. So the two hole-forcers, even adjacent, need only a single
unit of transient-hole budget. Verified by `native_decide`. -/
theorem SZ_handled_with_budget1 :
    (∃ pl1 ∈ Placement.allValidFor GameConfig.standard Piece.S,
        debt GameConfig.standard
          (Placement.applyStep GameConfig.standard Board.empty pl1) ≤ 1 ∧
        ∃ pl2 ∈ Placement.allValidFor GameConfig.standard Piece.Z,
          debt GameConfig.standard
            (Placement.applyStep GameConfig.standard
              (Placement.applyStep GameConfig.standard Board.empty pl1) pl2) ≤ 1)
    ∧
    (∃ pl1 ∈ Placement.allValidFor GameConfig.standard Piece.Z,
        debt GameConfig.standard
          (Placement.applyStep GameConfig.standard Board.empty pl1) ≤ 1 ∧
        ∃ pl2 ∈ Placement.allValidFor GameConfig.standard Piece.S,
          debt GameConfig.standard
            (Placement.applyStep GameConfig.standard
              (Placement.applyStep GameConfig.standard Board.empty pl1) pl2) ≤ 1) := by
  native_decide

#print axioms SZ_handled_with_budget1

/-! ## Next (energy-game backlog)

With headroom established as the energy, the next iterations build toward "survival ⟺ player
wins the lower-bounded energy game on the `(surface, bag)` arena":

1. Hole-free fiber: the full step `applyStep` is surface-determined when `debt = 0`
   (clears become surface-determined), giving a finite congruent sub-game.
2. The I-drain lemma: from every surface in a candidate `Σ`, some I-placement nets
   headroom change `≥ 0` (refills via a clear) — the localized form of crux #66/#72.
3. Exhibit-and-verify a concrete controller on a small `Σ` (`native_decide`), lifting
   soundness through `SurfaceFiber`/`HoleDebt`.
4. Compose into `tetrisSolvableValid_of_height_bounded_invariant`. -/

end Tetris.EnergyGame
