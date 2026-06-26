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

/-- **Global roughness density:** a stream of `n` bag orders contains exactly `2n` S/Z pieces. -/
theorem countP_isSZ_flatten {bags : List (List Piece)} (h : ∀ b ∈ bags, IsBagOrder b) :
    bags.flatten.countP isSZ = 2 * bags.length := by
  revert h
  induction bags with
  | nil => intro _; simp
  | cons b bs ih =>
    intro h
    have hb : IsBagOrder b := h b (List.mem_cons.mpr (Or.inl rfl))
    have hbs : bs.flatten.countP isSZ = 2 * bs.length :=
      ih (fun b' hb' => h b' (List.mem_cons.mpr (Or.inr hb')))
    rw [List.flatten_cons, List.countP_append, countP_isSZ hb, hbs]
    simp only [List.length_cons]; omega

/-- **Global drain density:** a stream of `n` bag orders contains exactly `n` I pieces. -/
theorem countP_isI_flatten {bags : List (List Piece)} (h : ∀ b ∈ bags, IsBagOrder b) :
    bags.flatten.countP isI = bags.length := by
  revert h
  induction bags with
  | nil => intro _; simp
  | cons b bs ih =>
    intro h
    have hb : IsBagOrder b := h b (List.mem_cons.mpr (Or.inl rfl))
    have hbs : bs.flatten.countP isI = bs.length :=
      ih (fun b' hb' => h b' (List.mem_cons.mpr (Or.inr hb')))
    rw [List.flatten_cons, List.countP_append, countP_isI hb, hbs]
    simp only [List.length_cons]; omega

/-- **Drain-budget feasibility — the make-or-break is geometry, not budget.**

Each bag adds `28` cells, i.e. `2.8` rows over the 10-wide board, so keeping height bounded over `n`
bags requires clearing on average `2.8` rows per bag — at most `14 n` rows per `5 n` bags. The I
pieces alone supply `n` potential four-line drains, i.e. `4 n = 20 n / 5` clearable rows. Hence the
drain *budget* strictly exceeds the clearing *requirement* with a fixed `20 : 14` margin: the player
is never starved of drains. What the order-switching adversary attacks (per `tetris_preview`) is the
*geometry* of cashing those drains — keeping a usable well while the middle is pumped — never the
count of I pieces. -/
theorem drain_budget_ge_clearing_need {bags : List (List Piece)} (h : ∀ b ∈ bags, IsBagOrder b) :
    14 * bags.length ≤ 20 * bags.flatten.countP isI := by
  rw [countP_isI_flatten h]; omega

/-- **Recovery budget:** every bag delivers exactly five non-S/Z pieces — the player's guaranteed
window to rebuild a usable well between roughness injections. This is the `≥5`-recovery half of the
renewal structure: a burst of consecutive S/Z (capped at 4, since it can only straddle one bag
boundary) is always separated from the next burst by these recovery pieces. -/
theorem countP_nonSZ {l : List Piece} (h : IsBagOrder l) : l.countP (fun p => !isSZ p) = 5 := by
  rw [h.countP_eq (fun p => !isSZ p)]; decide

/-- The seven pieces of every bag split exactly as **2 roughness + 5 recovery**. -/
theorem roughness_add_recovery {l : List Piece} (h : IsBagOrder l) :
    l.countP isSZ + l.countP (fun p => !isSZ p) = 7 := by
  rw [countP_isSZ h, countP_nonSZ h]

end Tetris.BagBurst
