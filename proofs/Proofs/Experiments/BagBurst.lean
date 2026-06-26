import Mathlib
import Proofs.Piece

/-!
# The 7-bag renewal bound

The empirical probes (`tetris_policy`, `tetris_preview`) showed the order-switching adversary
beats every bounded player by injecting roughness with the S and Z pieces and delaying the I-piece
drain, pumping the middle of the board to the ceiling. This file proves the combinatorial backbone
that constrains that weapon: under the 7-bag randomizer **every bag delivers exactly two roughness
pieces (S, Z) and exactly one drain (I)**, so over any window the adversary's roughness budget and
the player's drain budget are pinned in a fixed ratio.

These are pure facts about permutations of the seven pieces — no board, no line clearing. They are
the resource accounting a make-or-break clearing argument consumes: the adversary cannot inject
roughness faster than two pieces per bag, and the player is guaranteed one four-line drain per bag.
-/

namespace Tetris.BagBurst

/-- The "rough" pieces S and Z: the only tetrominoes that cannot lie flat on a level surface, hence
the adversary's roughness injectors. -/
def isSZ (p : Piece) : Bool := p == Piece.S || p == Piece.Z

/-- The drain piece I: the only tetromino that can clear four lines at once. -/
def isI (p : Piece) : Bool := p == Piece.I

/-- The canonical 7-bag: one of each piece. -/
def bag : List Piece := [Piece.O, Piece.I, Piece.S, Piece.Z, Piece.T, Piece.L, Piece.J]

/-- A bag order is any permutation of the seven pieces — the 7-bag randomizer's output each bag. -/
def IsBagOrder (l : List Piece) : Prop := l.Perm bag

/-- Every bag order has length 7. -/
theorem isBagOrder_length {l : List Piece} (h : IsBagOrder l) : l.length = 7 := by
  rw [h.length_eq]; decide

/-- **Roughness budget:** every bag delivers exactly two S/Z pieces. -/
theorem countP_isSZ {l : List Piece} (h : IsBagOrder l) : l.countP isSZ = 2 := by
  rw [h.countP_eq isSZ]; decide

/-- **Drain budget:** every bag delivers exactly one I piece. -/
theorem countP_isI {l : List Piece} (h : IsBagOrder l) : l.countP isI = 1 := by
  rw [h.countP_eq isI]; decide

/-- The two budgets, side by side: one drain per two roughness pieces, every bag. -/
theorem renewal_ratio {l : List Piece} (h : IsBagOrder l) :
    l.countP isSZ = 2 * l.countP isI := by
  rw [countP_isSZ h, countP_isI h]

/-- **Two-bag roughness bound:** across any two consecutive bags, exactly four S/Z pieces — so a
burst of consecutive S/Z (which can only straddle a bag boundary) is capped at four. -/
theorem countP_isSZ_two {l₁ l₂ : List Piece} (h₁ : IsBagOrder l₁) (h₂ : IsBagOrder l₂) :
    (l₁ ++ l₂).countP isSZ = 4 := by
  rw [List.countP_append, countP_isSZ h₁, countP_isSZ h₂]

/-- Across any two consecutive bags, exactly two drains. -/
theorem countP_isI_two {l₁ l₂ : List Piece} (h₁ : IsBagOrder l₁) (h₂ : IsBagOrder l₂) :
    (l₁ ++ l₂).countP isI = 2 := by
  rw [List.countP_append, countP_isI h₁, countP_isI h₂]

end Tetris.BagBurst
