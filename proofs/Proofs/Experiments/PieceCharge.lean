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

end Tetris.PieceCharge
