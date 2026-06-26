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

end Tetris.PieceCharge
