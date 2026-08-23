import Mathlib
import Proofs.Survival.ClearRecurrence

/-!
# The mix of clear sizes: one equation, three degrees of freedom

`ClearRate` pins the *total* clearing rate at `2.8` rows per bag. This file asks
the next question: is the **mix** pinned too? Must a surviving player use
singles, doubles, triples and tetrises in some forced proportion, or is any mix
admissible?

Write `a₁, a₂, a₃, a₄` for the number of drops that cleared exactly one, two,
three and four rows. Every cleared row belongs to exactly one drop, so

  **`a₁ + 2a₂ + 3a₃ + 4a₄ = cleared`**  (`mix_identity`)

and feeding that through the mass ledger gives the exact law

  **`10·(a₁ + 2a₂ + 3a₃ + 4a₄) + occupancy = 4·pieces`**  (`mix_law`).

## The answer: the mix is free

That is **one linear equation in four unknowns**. Counting therefore leaves a
three-dimensional family of admissible mixes, and no proportion is forced. Per
bag the constraint reads `f₁ + 2f₂ + 3f₃ + 4f₄ = 2.8`, whose corners are all
arithmetically available:

| pure strategy | clears per bag | share of pieces that clear |
|---|---|---|
| singles only | `f₁ = 2.8` | 40% |
| doubles only | `f₂ = 1.4` | 20% |
| triples only | `f₃ = 14/15 ≈ 0.93` | ≈ 13% |
| tetrises only | `f₄ = 0.7` | 10% |

`tetris_only_count_ge` is the sharp form at the tetris corner: a tetris-only
player must land a tetris in at least `0.7m − 5` of its first `m` bags — **70%
of all bags, with a lifetime slack of five**. `singles_only_count_ge` is the
same statement at the other corner.

## The one genuine side-constraint

Clear sizes are not quite unconstrained: a `k`-row clear needs the dropped piece
to occupy `k` distinct rows, and only the I tetromino spans four
(`tetris_requires_I`). So `a₄` is capped by the number of I placements
(`sizeCount_four_le_iCount`), and since a bag holds exactly one I, **a bag admits
at most one tetris**.

This constrains *timing*, not the asymptotic mix: the rate law already forces
`4a₄ ≤ 0.4n`, i.e. `a₄ ≤ 0.1n`, which is below the I supply of `n/7 ≈ 0.143n`.
The two bounds do not collide, which is exactly why a tetris-only strategy is
arithmetically viable — it just has to convert 70% of the pieces whose arrival
time it does not control.

## What this does not say

Arithmetic admits every mix; it does not construct one. Whether a given mix is
*playable* — whether the board shapes needed to keep landing triples, say, are
reachable and recoverable — is a geometry question this file cannot reach. The
content here is a clean negative: **no counting argument will ever rule a mix
out**, so any obstruction to a pure strategy must come from board structure.
-/

namespace Tetris
namespace ClearRate

/-! ## Counting clears by size -/

/-- Number of the first `n` drops that cleared exactly `k` rows. -/
def sizeCount (cfg : GameConfig) (π : Policy cfg) (g0 : GameState) (k : ℕ) : ℕ → ℕ
  | 0 => 0
  | n + 1 =>
      sizeCount cfg π g0 k n
        + (if (Board.fullRows cfg
              ((π (trace cfg π g0 n)).place (trace cfg π g0 n).board)).card = k
           then 1 else 0)

@[simp] theorem sizeCount_zero (cfg : GameConfig) (π : Policy cfg) (g0 : GameState)
    (k : ℕ) : sizeCount cfg π g0 k 0 = 0 := rfl

theorem sizeCount_succ (cfg : GameConfig) (π : Policy cfg) (g0 : GameState)
    (k n : ℕ) :
    sizeCount cfg π g0 k (n + 1)
      = sizeCount cfg π g0 k n
        + (if (Board.fullRows cfg
              ((π (trace cfg π g0 n)).place (trace cfg π g0 n).board)).card = k
           then 1 else 0) := rfl

/-- Number of the first `n` drops that played an I piece. -/
def iCount (cfg : GameConfig) (π : Policy cfg) (g0 : GameState) : ℕ → ℕ
  | 0 => 0
  | n + 1 =>
      iCount cfg π g0 n
        + (if (π (trace cfg π g0 n)).piece = Piece.I then 1 else 0)

@[simp] theorem iCount_zero (cfg : GameConfig) (π : Policy cfg) (g0 : GameState) :
    iCount cfg π g0 0 = 0 := rfl

theorem iCount_succ (cfg : GameConfig) (π : Policy cfg) (g0 : GameState) (n : ℕ) :
    iCount cfg π g0 (n + 1)
      = iCount cfg π g0 n
        + (if (π (trace cfg π g0 n)).piece = Piece.I then 1 else 0) := rfl

/-! ## The mix identity -/

/-- **Every cleared row belongs to exactly one drop.** Weighting each drop by how
many rows it cleared recovers the cumulative clear count. Drops clear at most
four rows (`fullRows_card_le_four`), so the four sizes exhaust the possibilities. -/
theorem mix_identity {cfg : GameConfig} {π : Policy cfg} (n : ℕ) :
    sizeCount cfg π GameState.init 1 n + 2 * sizeCount cfg π GameState.init 2 n
        + 3 * sizeCount cfg π GameState.init 3 n
        + 4 * sizeCount cfg π GameState.init 4 n
      = cleared cfg π GameState.init n := by
  induction n with
  | zero => simp
  | succ k ih =>
    have h4 := fullRows_card_le_four (cfg := cfg) (π := π) k
    rw [sizeCount_succ, sizeCount_succ, sizeCount_succ, sizeCount_succ, cleared_succ]
    split_ifs <;> omega

/-- **The mix law.** The four clear-size counts, weighted by size, are pinned to
the delivered mass minus what is still on the board. One linear equation in four
unknowns — so the mix has three degrees of freedom and no proportion is
forced. -/
theorem mix_law {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (n : ℕ) :
    10 * (sizeCount GameConfig.standard π GameState.init 1 n
          + 2 * sizeCount GameConfig.standard π GameState.init 2 n
          + 3 * sizeCount GameConfig.standard π GameState.init 3 n
          + 4 * sizeCount GameConfig.standard π GameState.init 4 n)
        + (trace GameConfig.standard π GameState.init n).board.count
      = 4 * n := by
  rw [mix_identity]
  have h := init_ledger hv n
  rw [GameConfig.standard_cols] at h
  omega

/-! ## The corners -/

/-- **Tetris-only play must tetris in 70% of bags.** If every clear is a
four-row clear, then across `m` bags the tetris count satisfies
`28m ≤ 40·a₄ + 200`, i.e. `a₄ ≥ 0.7m − 5`. The lifetime slack is five bags,
whatever `m` is. -/
theorem tetris_only_count_ge {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init (7 * m)).lost
      GameConfig.standard)
    (h1 : sizeCount GameConfig.standard π GameState.init 1 (7 * m) = 0)
    (h2 : sizeCount GameConfig.standard π GameState.init 2 (7 * m) = 0)
    (h3 : sizeCount GameConfig.standard π GameState.init 3 (7 * m) = 0) :
    28 * m ≤ 40 * sizeCount GameConfig.standard π GameState.init 4 (7 * m) + 200 := by
  have h := mix_law hv (7 * m)
  have hcap := count_lt_two_hundred_one hv hlive
  rw [h1, h2, h3] at h
  omega

/-- **Singles-only play must clear on 40% of its pieces.** The other corner:
`28m ≤ 10·a₁ + 200`. -/
theorem singles_only_count_ge {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init (7 * m)).lost
      GameConfig.standard)
    (h2 : sizeCount GameConfig.standard π GameState.init 2 (7 * m) = 0)
    (h3 : sizeCount GameConfig.standard π GameState.init 3 (7 * m) = 0)
    (h4 : sizeCount GameConfig.standard π GameState.init 4 (7 * m) = 0) :
    28 * m ≤ 10 * sizeCount GameConfig.standard π GameState.init 1 (7 * m) + 200 := by
  have h := mix_law hv (7 * m)
  have hcap := count_lt_two_hundred_one hv hlive
  rw [h2, h3, h4] at h
  omega

/-! ## The one genuine side-constraint on the mix -/

/-- **Tetrises are capped by I placements.** Only the I tetromino spans four
rows, so every four-row clear is played with an I. Since a bag holds exactly one
I, a bag admits at most one tetris. -/
theorem sizeCount_four_le_iCount {cfg : GameConfig} {π : Policy cfg} (n : ℕ) :
    sizeCount cfg π GameState.init 4 n ≤ iCount cfg π GameState.init n := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [sizeCount_succ, iCount_succ]
    by_cases hc : (Board.fullRows cfg
        ((π (trace cfg π GameState.init k)).place
          (trace cfg π GameState.init k).board)).card = 4
    · have hI : (π (trace cfg π GameState.init k)).piece = Piece.I :=
        tetris_requires_I_trace (cfg := cfg) (π := π) (n := k) (by omega)
      rw [if_pos hc, if_pos hI]
      omega
    · rw [if_neg hc]
      split <;> omega

/-- The rate law already caps tetrises below the I supply: `4a₄ ≤ 4·pieces/10`
forces `10·a₄ ≤ pieces`, while the bag supplies one I every 7. The two ceilings
do not collide, which is why the tetris corner is arithmetically reachable. -/
theorem ten_mul_sizeCount_four_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (n : ℕ) :
    10 * sizeCount GameConfig.standard π GameState.init 4 n ≤ n := by
  have h := mix_law hv n
  omega

end ClearRate
end Tetris
