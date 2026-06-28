import Mathlib
import Proofs.Piece

/-!
# Coloring charges of the seven pieces — the additive (parity) theory

The max-plus / topical work studied the *height* face of the dynamics: each piece a tropical matrix,
survival a spectral sign. This file studies the complementary **ordinary-linear** face — *content
and parity* rather than stacking — by assigning every piece an algebraic **charge** under a coloring
(a linear functional on cells). Unlike piece-specific combinatorics, a charge classifies all seven
pieces *uniformly*: each becomes one element of an abelian group, and the seven together are the
object of study.

The first coloring is the checkerboard `(column + row) mod 2`. The headline is the classical
**T-tetromino theorem**, here over the real engine `shape`: the checkerboard charge is carried by
*exactly one* piece, the T, at *every* rotation — the other six are perfectly balanced. Consequently
a whole 7-bag carries intrinsic charge `1`, a parity defect no placement can erase.
-/

namespace Tetris.PieceCharge

/-- Checkerboard charge of a finite set of cells: the parity of `Σ (column + row)` over the cells —
i.e. the white-cell-count parity under the `(col + row) mod 2` 2-coloring. Valued in `ZMod 2`. -/
def chargeC (s : Finset (ℕ × ℕ)) : ZMod 2 := ∑ c ∈ s, ((c.1 + c.2 : ℕ) : ZMod 2)

/-- The checkerboard charge of piece `p` at rotation `r`. -/
def checkerCharge (p : Piece) (r : Rotation) : ZMod 2 := chargeC (p.shape r)

/-- **The T-tetromino theorem, uniform over all seven pieces.** Under the checkerboard 2-coloring
the charge is carried by exactly one piece — the T — and at *every* rotation; the other six are
perfectly balanced (charge `0`). One algebraic invariant classifying the whole piece set at once. -/
theorem checkerCharge_classification :
    ∀ (p : Piece) (r : Rotation), checkerCharge p r = if p = Piece.T then 1 else 0 := by
  decide

/-- The checkerboard charge depends only on the piece, never on its rotation. -/
theorem checkerCharge_rotation_indep (p : Piece) (r r' : Rotation) :
    checkerCharge p r = checkerCharge p r' := by
  rw [checkerCharge_classification, checkerCharge_classification]

/-- Only the T carries checkerboard charge. -/
theorem checkerCharge_ne_zero_iff (p : Piece) (r : Rotation) :
    checkerCharge p r ≠ 0 ↔ p = Piece.T := by
  rw [checkerCharge_classification]
  by_cases h : p = Piece.T <;> simp [h]

/-- **The bag is intrinsically charged.** However the seven pieces of a bag are rotated, their total
checkerboard charge is `1` — carried entirely by the T. A full bag can never be charge-balanced: a
parity obstruction every bag's placement must carry, independent of the player. -/
theorem bag_checkerCharge (rot : Piece → Rotation) :
    ∑ p : Piece, checkerCharge p (rot p) = 1 := by
  simp only [checkerCharge_classification]
  decide

/-- Shifting every cell by `(dc, dr)` shifts the charge by `|s| • (dc + dr)`. -/
theorem chargeC_shift (s : Finset (ℕ × ℕ)) (dc dr : ℕ) :
    chargeC (s.image (fun c => (c.1 + dc, c.2 + dr)))
      = chargeC s + s.card • ((dc + dr : ℕ) : ZMod 2) := by
  unfold chargeC
  rw [Finset.sum_image (fun a _ b _ hab => by
    simp only [Prod.mk.injEq, add_left_inj] at hab
    exact Prod.ext hab.1 hab.2)]
  have key : ∀ c : ℕ × ℕ, (((c.1 + dc) + (c.2 + dr) : ℕ) : ZMod 2)
      = ((c.1 + c.2 : ℕ) : ZMod 2) + ((dc + dr : ℕ) : ZMod 2) := by
    intro c; push_cast; ring
  simp_rw [key]
  rw [Finset.sum_add_distrib, Finset.sum_const]

/-- **The checkerboard charge is translation-invariant** on every 4-cell set — so it is a genuine
invariant of each tetromino, independent of *where* on the board it is placed (only its shape
matters). This is what makes `checkerCharge` a well-defined charge, not a positional artifact. -/
theorem chargeC_shift_invariant {s : Finset (ℕ × ℕ)} (hs : s.card = 4) (dc dr : ℕ) :
    chargeC (s.image (fun c => (c.1 + dc, c.2 + dr))) = chargeC s := by
  have h4 : (4 : ℕ) • ((dc + dr : ℕ) : ZMod 2) = 0 := by
    have : ((4 : ℕ) : ZMod 2) = 0 := by decide
    rw [nsmul_eq_mul, this, zero_mul]
  rw [chargeC_shift, hs, h4, add_zero]

/-! ## Which charges survive a line-clear

The checkerboard charge above is *broken* by clears: clearing a row shifts everything above it down,
flipping the colour of every shifted cell. This section finds the charges that ARE conserved. A
line-clear removes exactly one cell from *every* column, so a **column** charge — a weighting of the
columns alone, ignoring rows — with *zero total weight* is exactly preserved, holes and all. These
balanced column colourings are the genuine conservation laws of Tetris. -/

/-- The column-mass charge of a column-count vector `v` under column weights `w`. -/
def massCharge (w v : Fin 10 → ℤ) : ℤ := ∑ c, w c * v c

/-- Placement is linear: a piece adds its column profile, so the charge adds the profile's charge —
a *board-independent* increment (the landing height is irrelevant to a column charge). -/
theorem massCharge_add (w v p : Fin 10 → ℤ) :
    massCharge w (v + p) = massCharge w v + massCharge w p := by
  simp only [massCharge, Pi.add_apply, mul_add, Finset.sum_add_distrib]

/-- A line-clear removes one cell from every column. -/
def clearRow (v : Fin 10 → ℤ) : Fin 10 → ℤ := fun c => v c - 1

/-- A clear drops the charge by the total weight `∑ w`. -/
theorem massCharge_clearRow (w v : Fin 10 → ℤ) :
    massCharge w (clearRow v) = massCharge w v - ∑ c, w c := by
  simp only [massCharge, clearRow, mul_sub, mul_one, Finset.sum_sub_distrib]

/-- **The conservation law.** A column-mass charge with zero total weight is invariant under every
line-clear — regardless of holes, of which row clears, or how many. The conserved charges of Tetris
are exactly the balanced column colourings (the `9`-dimensional space `∑ w = 0`). -/
theorem massCharge_clearRow_invariant {w : Fin 10 → ℤ} (hw : ∑ c, w c = 0) (v : Fin 10 → ℤ) :
    massCharge w (clearRow v) = massCharge w v := by
  rw [massCharge_clearRow, hw, sub_zero]

/-- **The uniform level law.** Every one of the seven pieces is exactly four cells, at every
rotation — so every placement adds exactly `4` to the total mass, identically for all pieces. The
"level" direction is uniform; only the *shape* (the balanced charges) is piece-specific. -/
theorem shape_card : ∀ (p : Piece) (r : Rotation), (p.shape r).card = 4 := by decide

/-- The horizontal-moment weight `w c = 2c - 9`: balanced (`∑ w = 0`), so its charge — a signed
left/right mass imbalance — is a conserved quantity of the dynamics. A concrete conservation law. -/
def wLR : Fin 10 → ℤ := fun c => 2 * (c : ℤ) - 9

theorem wLR_balanced : ∑ c, wLR c = 0 := by decide

/-- The left/right horizontal-moment charge is conserved under every line-clear. -/
theorem massCharge_wLR_conserved (v : Fin 10 → ℤ) :
    massCharge wLR (clearRow v) = massCharge wLR v :=
  massCharge_clearRow_invariant wLR_balanced v

/-- **Characterization of the conservation laws.** A column-mass charge is conserved by clears (on
every board) *iff* its total weight is zero — iff it vanishes on a complete row. The conservation
laws of Tetris are exactly the column colourings blind to full rows: they read only the *incomplete*
part of the board — its jaggedness and holes — which is exactly the part that decides survival. -/
theorem massCharge_conserved_iff (w : Fin 10 → ℤ) :
    (∀ v, massCharge w (clearRow v) = massCharge w v) ↔ ∑ c, w c = 0 := by
  constructor
  · intro h
    have e := massCharge_clearRow w 0
    rw [h 0] at e
    linarith
  · intro hw v
    exact massCharge_clearRow_invariant hw v

/-! ## Are the conserved charges controllable?

A conserved charge is only an obstruction if the adversary can force it to *grow*. We test the
middle-vs-edge charge — the one that reads exactly the empirical "middle pump" — and find the player
can always push it *down*: the conserved shape-charges are freely controllable, so the middle pump
is not a charge obstruction but a coupling to height-feasibility. -/

/-- Column weight reading middle-vs-edge mass: the two edge columns weigh `-4`, the middle eight
weigh `+1`. Balanced (`∑_{0..9} = 0`), so its charge is conserved under clears. -/
def wMid : ℕ → ℤ := fun c => if c = 0 ∨ c = 9 then -4 else 1

/-- The middle-vs-edge charge increment of placing piece `p` (rotation `r`) at column offset `j`. -/
def incrMid (p : Piece) (r : Rotation) (j : ℕ) : ℤ := ∑ cell ∈ p.shape r, wMid (cell.1 + j)

/-- **The conserved shape-charge is player-reducible.** For *every* piece the player has a placement
that strictly decreases the middle-vs-edge charge — so the adversary can never force it to grow. The
conserved shape-charges are freely controllable; the empirically-observed middle pump is therefore
NOT a charge obstruction but a coupling to height-feasibility: the player cannot always place where
the charge says to, because the chosen columns may be too tall. -/
theorem incrMid_reducible :
    ∀ p : Piece, ∃ (r : Rotation) (j : Fin 7), incrMid p r j.val < 0 := by
  decide

/-! ## The level/shape dichotomy — where survival is actually decided

The conserved charges (`∑ w = 0`) are the SHAPE, and we just showed they are freely controllable.
The remaining direction is the LEVEL — total mass, the weight `w = 1`, the one weighting with
`∑ w ≠ 0`. It is the exact opposite of conserved: every placement raises it by the piece size `4`,
and only a clear lowers it (by the board width `10`). So the level is the sole *un*-controllable
coordinate, and survival reduces entirely to keeping it bounded — to the clearing rate — with the
shape free. This is the charge theory's dissection of the crux: not the shape, but the level. -/

/-- The level (total mass) of a column-count vector. -/
abbrev level (v : Fin 10 → ℤ) : ℤ := massCharge 1 v

/-- A clear lowers the level by the board width `10` (ten cells leave per cleared row). -/
theorem level_clearRow (v : Fin 10 → ℤ) : level (clearRow v) = level v - 10 := by
  unfold level
  rw [massCharge_clearRow, show (∑ c : Fin 10, (1 : Fin 10 → ℤ) c) = 10 from by simp]

/-- Placing any piece (total mass `4`, uniformly — see `shape_card`) raises the level by exactly
`4`. The level rises monotonically under placement; only clears reduce it. -/
theorem level_place {prof : Fin 10 → ℤ} (h : level prof = 4) (v : Fin 10 → ℤ) :
    level (v + prof) = level v + 4 := by
  unfold level at *
  rw [massCharge_add, h]

/-- **The clearing coupling — where the otherwise-free shape control finally bites.** Clearing `m`
lines lowers every column by `m` and the level by `10 m`, but it is legal only when *every* column
has height ≥ m: a complete row needs all ten columns. So the clearable depth — the only way down for
the level — is capped by the **minimum** column height. A well (one low column) starves clearing no
matter how tall the rest. This is the coupling: to flush the level you must fill the wells, yet the
wells are exactly what the I-piece drains, so the adversary attacks by delaying the I. -/
theorem level_clear (h : Fin 10 → ℤ) (m : ℤ) (hm : ∀ c, m ≤ h c) :
    level (fun c => h c - m) = level h - 10 * m ∧ ∀ c, 0 ≤ h c - m := by
  refine ⟨?_, fun c => by linarith [hm c]⟩
  unfold level massCharge
  simp only [Pi.one_apply, one_mul]
  rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin]
  ring

/-! ## An actionable lever: clearing capacity = the minimum column height

The dichotomy reduces survival to keeping the level bounded, and `level_clear` says a clear only
goes as deep as the lowest column. Turned into a strategy lever: a hole-free board can clear exactly
`minHt` lines right now, dropping the level by `10 · minHt`. So — *alongside* "reduce roughness"
from the tropical theory — **"raise the minimum column height" is a provable survival lever** an
agent can put straight into its evaluation function. -/

/-- The minimum column height of a (hole-free) board. -/
def minHt (h : Fin 10 → ℤ) : ℤ := Finset.univ.inf' ⟨0, Finset.mem_univ 0⟩ h

/-- **Clearing capacity = the minimum column height.** A depth-`m` clear is legal iff `m ≤` every
column iff `m ≤ minHt`; so the most lines a hole-free board can clear at once is exactly its minimum
column height. Raising the minimum raises the clearing capacity — a directly-optimizable lever. -/
theorem clearable_iff_le_minHt (h : Fin 10 → ℤ) (m : ℤ) :
    (∀ c, m ≤ h c) ↔ m ≤ minHt h := by
  rw [minHt, Finset.le_inf'_iff]
  simp

/-- Clearing at full capacity drops the level by `10 · minHt` — the level-reduction per clearing
event is exactly ten times the minimum column height. To flush more level, raise the minimum. -/
theorem level_clear_capacity (h : Fin 10 → ℤ) :
    level (fun c => h c - minHt h) = level h - 10 * minHt h :=
  (level_clear h (minHt h) ((clearable_iff_le_minHt h (minHt h)).mpr le_rfl)).1

end Tetris.PieceCharge
