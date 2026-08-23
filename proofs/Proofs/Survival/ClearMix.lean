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

/-! ## Which pieces can clear how many rows

Spanning `k` rows is *necessary* to clear `k`, but it is not sufficient, and the
gap matters. A hard drop rests **on top** of the stack: in every column the piece
occupies, the board is empty at and above the piece's lowest cell there. So if a
cleared row `r` is not filled by the piece in some occupied column, the board
must supply that cell — which forces `r` to lie strictly *below* the piece's cell
in that column.

Reading that off the drop profile gives an exact per-shape ceiling
(`maxClears`), and it is strictly sharper than the row span:

| piece | rows spanned | rows actually clearable |
|---|---|---|
| I | 4 | **4** |
| L, J | 3 | **3** |
| S, Z, T | 3 | **2** |
| O | 2 | **2** |

S, Z and T span three rows but can never clear three: whichever column fails to
reach the top row would need a board cell above the piece's resting level there,
and the hard drop forbids it. Only **I, L and J** can clear three or more
(`three_clear_requires_I_L_or_J`), and only **I** can clear four
(`tetris_requires_I`). -/

/-- Every column the piece occupies is low enough for the piece to reach it:
`colHeight ≤ dropOffset + cell height` for each profile cell. -/
theorem colHeight_le_dropOffset_add {b : Board} {pl : Placement} {cell : Coord}
    (hcell : cell ∈ pl.shapeUp) :
    b.colHeight (pl.col + cell.1) ≤ pl.dropOffset b + cell.2 := by
  have hle : b.colHeight (pl.col + cell.1) - cell.2 ≤ pl.dropOffset b := by
    rw [Placement.dropOffset_eq_sup]
    exact Finset.le_sup
      (f := fun c : Coord => b.colHeight (pl.col + c.1) - c.2) hcell
  omega

/-- A relative row `t` of a drop profile is *clearable* when every occupied
column either has a profile cell at `t`, or has its lowest cell strictly above
`t` (so the board may legally supply the missing cell from underneath). -/
def ClearableRow (S : Finset Coord) (t : ℕ) : Prop :=
  ∀ cell ∈ S, ((cell.1, t) ∈ S) ∨ t < cell.2

instance (S : Finset Coord) (t : ℕ) : Decidable (ClearableRow S t) := by
  unfold ClearableRow; infer_instance

/-- The per-shape ceiling on rows cleared by one drop. -/
def maxClears (p : Piece) (r : Rotation) : ℕ :=
  (((p.shapeUp r).image Prod.snd).filter (fun t => ClearableRow (p.shapeUp r) t)).card

/-- **The hard-drop clearing ceiling.** One drop onto a board with no pending
full rows clears at most `maxClears` rows. -/
theorem fullRows_card_le_maxClears {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hnf : ∀ r, ¬ Board.isFull cfg b r) :
    (Board.fullRows cfg (pl.place b)).card ≤ maxClears pl.piece pl.rot := by
  classical
  have hsub : Board.fullRows cfg (pl.place b)
      ⊆ ((pl.shapeUp.image Prod.snd).filter
            (fun t => ClearableRow pl.shapeUp t)).image
          (fun t => pl.dropOffset b + t) := by
    intro r hr
    simp only [Board.fullRows, Finset.mem_filter] at hr
    -- the cleared row contains a cell of the drop
    obtain ⟨c, hc, hcb⟩ : ∃ c ∈ Finset.range cfg.cols, (c, r) ∉ b := by
      by_contra hcon
      push Not at hcon
      exact hnf r hcon
    have hcdrop : (c, r) ∈ pl.dropped b := by
      have hcplace : (c, r) ∈ pl.place b := hr.2 c hc
      simp only [Placement.place, Finset.mem_union] at hcplace
      rcases hcplace with hb' | hd
      · exact absurd hb' hcb
      · exact hd
    obtain ⟨cell₀, hcell₀, hEq⟩ : ∃ cell ∈ pl.shapeUp,
        (pl.col + cell.1, pl.dropOffset b + cell.2) = (c, r) := by
      unfold Placement.dropped Placement.cellsAt at hcdrop
      rw [Finset.mem_image] at hcdrop
      exact hcdrop
    have hrt : r = pl.dropOffset b + cell₀.2 := (congrArg Prod.snd hEq).symm
    -- and every occupied column either supplies it or sits above it
    have hclear : ClearableRow pl.shapeUp cell₀.2 := by
      intro cell hcell
      have hlt : pl.col + cell.1 < cfg.cols := hv cell hcell
      have hmem : (pl.col + cell.1, r) ∈ pl.place b :=
        hr.2 _ (Finset.mem_range.mpr hlt)
      simp only [Placement.place, Finset.mem_union] at hmem
      rcases hmem with hb' | hd
      · right
        have h1 : r < b.colHeight (pl.col + cell.1) := Board.lt_colHeight hb'
        have h2 := colHeight_le_dropOffset_add (b := b) (pl := pl) hcell
        omega
      · left
        unfold Placement.dropped Placement.cellsAt at hd
        rw [Finset.mem_image] at hd
        obtain ⟨cell', hcell', hEq'⟩ := hd
        have hc1 : pl.col + cell'.1 = pl.col + cell.1 := congrArg Prod.fst hEq'
        have hc2 : pl.dropOffset b + cell'.2 = r := congrArg Prod.snd hEq'
        have : cell' = (cell.1, cell₀.2) := by
          refine Prod.ext ?_ ?_ <;> simp <;> omega
        rwa [this] at hcell'
    rw [Finset.mem_image]
    exact ⟨cell₀.2, Finset.mem_filter.mpr
      ⟨Finset.mem_image.mpr ⟨cell₀, hcell₀, rfl⟩, hclear⟩, hrt.symm⟩
  calc (Board.fullRows cfg (pl.place b)).card
      ≤ (((pl.shapeUp.image Prod.snd).filter
            (fun t => ClearableRow pl.shapeUp t)).image
          (fun t => pl.dropOffset b + t)).card := Finset.card_le_card hsub
    _ ≤ ((pl.shapeUp.image Prod.snd).filter
            (fun t => ClearableRow pl.shapeUp t)).card := Finset.card_image_le

/-- **S, Z, T and O can never clear three rows.** A 28-case check of the
per-shape ceiling. -/
theorem maxClears_le_two (p : Piece) (r : Rotation)
    (hI : p ≠ Piece.I) (hL : p ≠ Piece.L) (hJ : p ≠ Piece.J) :
    maxClears p r ≤ 2 := by
  revert hI hL hJ
  revert r
  revert p
  decide

/-- **Only I, L and J can clear three or more rows in one drop.** Sharper than
the row-span bound: S, Z and T *span* three rows but the hard drop can never
convert that into a triple. -/
theorem three_clear_requires_I_L_or_J {cfg : GameConfig} {b : Board} {pl : Placement}
    (hv : pl.Valid cfg) (hnf : ∀ r, ¬ Board.isFull cfg b r)
    (h3 : 3 ≤ (Board.fullRows cfg (pl.place b)).card) :
    pl.piece = Piece.I ∨ pl.piece = Piece.L ∨ pl.piece = Piece.J := by
  by_contra hcon
  push Not at hcon
  have hle := maxClears_le_two pl.piece pl.rot hcon.1 hcon.2.1 hcon.2.2
  have hb := fullRows_card_le_maxClears hv hnf
  omega

/-- Trace form of the triple constraint. -/
theorem three_clear_requires_I_L_or_J_trace {cfg : GameConfig} {π : Policy cfg}
    (hv : ∀ g, (π g).Valid cfg) {n : ℕ}
    (h3 : 3 ≤ (Board.fullRows cfg
      ((π (trace cfg π GameState.init n)).place
        (trace cfg π GameState.init n).board)).card) :
    (π (trace cfg π GameState.init n)).piece = Piece.I
      ∨ (π (trace cfg π GameState.init n)).piece = Piece.L
      ∨ (π (trace cfg π GameState.init n)).piece = Piece.J :=
  three_clear_requires_I_L_or_J (hv _) (trace_board_no_full n) h3

/-! ## Big clears are capped by the I/L/J supply -/

/-- Number of the first `n` drops that played an I, L or J. -/
def iljCount (cfg : GameConfig) (π : Policy cfg) (g0 : GameState) : ℕ → ℕ
  | 0 => 0
  | n + 1 =>
      iljCount cfg π g0 n
        + (if (π (trace cfg π g0 n)).piece = Piece.I
              ∨ (π (trace cfg π g0 n)).piece = Piece.L
              ∨ (π (trace cfg π g0 n)).piece = Piece.J then 1 else 0)

@[simp] theorem iljCount_zero (cfg : GameConfig) (π : Policy cfg) (g0 : GameState) :
    iljCount cfg π g0 0 = 0 := rfl

theorem iljCount_succ (cfg : GameConfig) (π : Policy cfg) (g0 : GameState) (n : ℕ) :
    iljCount cfg π g0 (n + 1)
      = iljCount cfg π g0 n
        + (if (π (trace cfg π g0 n)).piece = Piece.I
              ∨ (π (trace cfg π g0 n)).piece = Piece.L
              ∨ (π (trace cfg π g0 n)).piece = Piece.J then 1 else 0) := rfl

/-- **Triples and tetrises are capped by the I/L/J supply.** Only I, L and J can
clear three or more rows (`three_clear_requires_I_L_or_J`), so
`a₃ + a₄ ≤ #{I,L,J} placements` — three sevenths of the pieces. Like the
tetris/I cap this constrains *timing* rather than the asymptotic mix: the rate
law caps `3a₃ + 4a₄` at `0.4n` already, well below the `3n/7` supply. -/
theorem sizeCount_big_le_iljCount {cfg : GameConfig} {π : Policy cfg}
    (hv : ∀ g, (π g).Valid cfg) (n : ℕ) :
    sizeCount cfg π GameState.init 3 n + sizeCount cfg π GameState.init 4 n
      ≤ iljCount cfg π GameState.init n := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [sizeCount_succ, sizeCount_succ, iljCount_succ]
    by_cases h3 : (Board.fullRows cfg
        ((π (trace cfg π GameState.init k)).place
          (trace cfg π GameState.init k).board)).card = 3
    · have hilj := three_clear_requires_I_L_or_J_trace hv (n := k) (by omega)
      rw [if_pos h3, if_neg (by omega), if_pos hilj]
      omega
    · by_cases h4 : (Board.fullRows cfg
          ((π (trace cfg π GameState.init k)).place
            (trace cfg π GameState.init k).board)).card = 4
      · have hilj := three_clear_requires_I_L_or_J_trace hv (n := k) (by omega)
        rw [if_neg h3, if_pos h4, if_pos hilj]
        omega
      · rw [if_neg h3, if_neg h4]
        split <;> omega

/-- Size counters never decrease. -/
theorem sizeCount_mono (cfg : GameConfig) (π : Policy cfg) (g0 : GameState)
    (k : ℕ) : Monotone (sizeCount cfg π g0 k) := by
  apply monotone_nat_of_le_succ
  intro n
  rw [sizeCount_succ]
  exact Nat.le_add_right _ _

/-- **The period mix.** Over any 35-placement cycle period the clear-size
increments weight-sum to exactly the period's fourteen rows:
`Δa₁ + 2Δa₂ + 3Δa₃ + 4Δa₄ = 14`. Combined with the period piece balance
(five I's per period) the admissible per-period mixes form a small explicit
polytope — e.g. three tetrises and a double leave `14 − 14 = 0` singles. -/
theorem period_mix_fourteen {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    (sizeCount GameConfig.standard π GameState.init 1 (n + 35)
        - sizeCount GameConfig.standard π GameState.init 1 n)
      + 2 * (sizeCount GameConfig.standard π GameState.init 2 (n + 35)
        - sizeCount GameConfig.standard π GameState.init 2 n)
      + 3 * (sizeCount GameConfig.standard π GameState.init 3 (n + 35)
        - sizeCount GameConfig.standard π GameState.init 3 n)
      + 4 * (sizeCount GameConfig.standard π GameState.init 4 (n + 35)
        - sizeCount GameConfig.standard π GameState.init 4 n)
      = 14 := by
  have h1 := mix_identity (cfg := GameConfig.standard) (π := π) n
  have h2 := mix_identity (cfg := GameConfig.standard) (π := π) (n + 35)
  have hbal := trace_eq_clears hv (Nat.le_add_right n 35) hcyc
  have hm1 := sizeCount_mono GameConfig.standard π GameState.init 1
    (Nat.le_add_right n 35)
  have hm2 := sizeCount_mono GameConfig.standard π GameState.init 2
    (Nat.le_add_right n 35)
  have hm3 := sizeCount_mono GameConfig.standard π GameState.init 3
    (Nat.le_add_right n 35)
  have hm4 := sizeCount_mono GameConfig.standard π GameState.init 4
    (Nat.le_add_right n 35)
  omega

/-- **At most three tetrises per cycle period** — sharper than the five-I
supply: fourteen rows simply cannot absorb a fourth tetris (`4·4 > 14`). The
mix polytope also caps triples at four, doubles at seven, singles at
fourteen. -/
theorem period_tetris_le_three {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    sizeCount GameConfig.standard π GameState.init 4 (n + 35)
      - sizeCount GameConfig.standard π GameState.init 4 n ≤ 3 := by
  have h := period_mix_fourteen hv hcyc
  omega

/-- At most four triples per cycle period. -/
theorem period_triples_le_four {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    sizeCount GameConfig.standard π GameState.init 3 (n + 35)
      - sizeCount GameConfig.standard π GameState.init 3 n ≤ 4 := by
  have h := period_mix_fourteen hv hcyc
  omega

/-- At most seven doubles per cycle period. -/
theorem period_doubles_le_seven {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    sizeCount GameConfig.standard π GameState.init 2 (n + 35)
      - sizeCount GameConfig.standard π GameState.init 2 n ≤ 7 := by
  have h := period_mix_fourteen hv hcyc
  omega

/-- At most fourteen singles per cycle period. -/
theorem period_singles_le_fourteen {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    sizeCount GameConfig.standard π GameState.init 1 (n + 35)
      - sizeCount GameConfig.standard π GameState.init 1 n ≤ 14 := by
  have h := period_mix_fourteen hv hcyc
  omega

/-- **The multi-period mix**: over `j` cycle periods the clear-size increments
weight-sum to exactly `14·j` rows. -/
theorem multi_period_mix_fourteen {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) (j : ℕ) :
    (sizeCount GameConfig.standard π GameState.init 1 (n + 35 * j)
        - sizeCount GameConfig.standard π GameState.init 1 n)
      + 2 * (sizeCount GameConfig.standard π GameState.init 2 (n + 35 * j)
        - sizeCount GameConfig.standard π GameState.init 2 n)
      + 3 * (sizeCount GameConfig.standard π GameState.init 3 (n + 35 * j)
        - sizeCount GameConfig.standard π GameState.init 3 n)
      + 4 * (sizeCount GameConfig.standard π GameState.init 4 (n + 35 * j)
        - sizeCount GameConfig.standard π GameState.init 4 n)
      = 14 * j := by
  have h1 := mix_identity (cfg := GameConfig.standard) (π := π) n
  have h2 := mix_identity (cfg := GameConfig.standard) (π := π) (n + 35 * j)
  have hcl := multi_period_clears hv hcyc j
  have hm1 := sizeCount_mono GameConfig.standard π GameState.init 1
    (Nat.le_add_right n (35 * j))
  have hm2 := sizeCount_mono GameConfig.standard π GameState.init 2
    (Nat.le_add_right n (35 * j))
  have hm3 := sizeCount_mono GameConfig.standard π GameState.init 3
    (Nat.le_add_right n (35 * j))
  have hm4 := sizeCount_mono GameConfig.standard π GameState.init 4
    (Nat.le_add_right n (35 * j))
  omega

/-- **At most `3·j` tetrises over `j` cycle periods** — the per-period row
budget telescopes, which is strictly sharper than the aggregate mix
(`⌊14j/4⌋ = 3.5j`): the periodic boundary re-arms the bound each lap. -/
theorem multi_period_tetris_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    ∀ j, sizeCount GameConfig.standard π GameState.init 4 (n + 35 * j)
      - sizeCount GameConfig.standard π GameState.init 4 n ≤ 3 * j := by
  intro j
  induction j with
  | zero => simp
  | succ j ih =>
    have hj := trace_period_multiples π GameState.init hcyc j
    have hj1 := trace_period_multiples π GameState.init hcyc (j + 1)
    have hcycj : trace GameConfig.standard π GameState.init (n + 35 * j)
        = trace GameConfig.standard π GameState.init ((n + 35 * j) + 35) := by
      rw [show (n + 35 * j) + 35 = n + (j + 1) * 35 by ring,
        show n + 35 * j = n + j * 35 by ring]
      exact hj.symm.trans hj1
    have hstep := period_tetris_le_three hv hcycj
    have hmono := sizeCount_mono GameConfig.standard π GameState.init 4
      (Nat.le_add_right n (35 * j))
    rw [show n + 35 * (j + 1) = (n + 35 * j) + 35 by ring]
    omega

/-- At most `4·j` triples over `j` cycle periods (telescoped; sharper than
the aggregate `⌊14j/3⌋`). -/
theorem multi_period_triples_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    ∀ j, sizeCount GameConfig.standard π GameState.init 3 (n + 35 * j)
      - sizeCount GameConfig.standard π GameState.init 3 n ≤ 4 * j := by
  intro j
  induction j with
  | zero => simp
  | succ j ih =>
    have hj := trace_period_multiples π GameState.init hcyc j
    have hj1 := trace_period_multiples π GameState.init hcyc (j + 1)
    have hcycj : trace GameConfig.standard π GameState.init (n + 35 * j)
        = trace GameConfig.standard π GameState.init ((n + 35 * j) + 35) := by
      rw [show (n + 35 * j) + 35 = n + (j + 1) * 35 by ring,
        show n + 35 * j = n + j * 35 by ring]
      exact hj.symm.trans hj1
    have hstep := period_triples_le_four hv hcycj
    have hmono := sizeCount_mono GameConfig.standard π GameState.init 3
      (Nat.le_add_right n (35 * j))
    rw [show n + 35 * (j + 1) = (n + 35 * j) + 35 by ring]
    omega

/-- At most `7·j` doubles over `j` cycle periods (direct from the mix). -/
theorem multi_period_doubles_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) (j : ℕ) :
    sizeCount GameConfig.standard π GameState.init 2 (n + 35 * j)
      - sizeCount GameConfig.standard π GameState.init 2 n ≤ 7 * j := by
  have h := multi_period_mix_fourteen hv hcyc j
  omega

/-- At most `14·j` singles over `j` cycle periods (direct from the mix). -/
theorem multi_period_singles_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) (j : ℕ) :
    sizeCount GameConfig.standard π GameState.init 1 (n + 35 * j)
      - sizeCount GameConfig.standard π GameState.init 1 n ≤ 14 * j := by
  have h := multi_period_mix_fourteen hv hcyc j
  omega

/-- **The tetris density law**: on a cycle, every horizon holds at most
`3·⌊(m−n)/35⌋ + 3` tetrises — asymptotic density ≤ 3/35 per placement
(≈ 0.086), pinned by the row budget alone. Round up to the enclosing period
boundary and telescope. -/
theorem cycle_tetris_density {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m : ℕ}
    (hnm : n ≤ m) :
    sizeCount GameConfig.standard π GameState.init 4 m
      - sizeCount GameConfig.standard π GameState.init 4 n
      ≤ 3 * ((m - n) / 35) + 3 := by
  set j := (m - n) / 35 with hj
  have hhi : m ≤ n + 35 * (j + 1) := by omega
  have hcap := multi_period_tetris_le hv hcyc (j + 1)
  have hmono := sizeCount_mono GameConfig.standard π GameState.init 4 hhi
  omega

/-- **The mix identity, windowed**: over *any* window the clear-size
increments weight-sum to exactly the rows cleared in that window —
`Δa₁ + 2Δa₂ + 3Δa₃ + 4Δa₄ = Δcleared`, no cycle hypothesis. -/
theorem mix_window_identity {cfg : GameConfig} {π : Policy cfg} {n m : ℕ}
    (hnm : n ≤ m) :
    (sizeCount cfg π GameState.init 1 m - sizeCount cfg π GameState.init 1 n)
      + 2 * (sizeCount cfg π GameState.init 2 m
        - sizeCount cfg π GameState.init 2 n)
      + 3 * (sizeCount cfg π GameState.init 3 m
        - sizeCount cfg π GameState.init 3 n)
      + 4 * (sizeCount cfg π GameState.init 4 m
        - sizeCount cfg π GameState.init 4 n)
      = cleared cfg π GameState.init m - cleared cfg π GameState.init n := by
  have h1 := mix_identity (cfg := cfg) (π := π) n
  have h2 := mix_identity (cfg := cfg) (π := π) m
  have hm1 := sizeCount_mono cfg π GameState.init 1 hnm
  have hm2 := sizeCount_mono cfg π GameState.init 2 hnm
  have hm3 := sizeCount_mono cfg π GameState.init 3 hnm
  have hm4 := sizeCount_mono cfg π GameState.init 4 hnm
  omega

/-- Every-horizon size caps on a cycle: triples ≤ `4⌊Δn/35⌋ + 4`,
doubles ≤ `7⌊Δn/35⌋ + 7`, singles ≤ `14⌊Δn/35⌋ + 14` — each telescoped
per-period cap extended to unaligned horizons by rounding up to the enclosing
period boundary (the tetris case is `cycle_tetris_density`). -/
theorem cycle_size_density {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m : ℕ}
    (hnm : n ≤ m) :
    sizeCount GameConfig.standard π GameState.init 3 m
        - sizeCount GameConfig.standard π GameState.init 3 n
      ≤ 4 * ((m - n) / 35) + 4
    ∧ sizeCount GameConfig.standard π GameState.init 2 m
        - sizeCount GameConfig.standard π GameState.init 2 n
      ≤ 7 * ((m - n) / 35) + 7
    ∧ sizeCount GameConfig.standard π GameState.init 1 m
        - sizeCount GameConfig.standard π GameState.init 1 n
      ≤ 14 * ((m - n) / 35) + 14 := by
  set j := (m - n) / 35 with hj
  have hhi : m ≤ n + 35 * (j + 1) := by omega
  have htr := multi_period_triples_le hv hcyc (j + 1)
  have hmix := multi_period_mix_fourteen hv hcyc (j + 1)
  have hm1 := sizeCount_mono GameConfig.standard π GameState.init 1 hhi
  have hm2 := sizeCount_mono GameConfig.standard π GameState.init 2 hhi
  have hm3 := sizeCount_mono GameConfig.standard π GameState.init 3 hhi
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- **Silence dominates a cycle**: of the 35 placements in a period, only
between 4 and 14 clear anything — at least 21 and at most 31 placements are
silent. Fourteen rows shared among events of size ≤ 4 need ≥ 4 events; events
of size ≥ 1 permit ≤ 14. -/
theorem period_clear_events_bounds {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    4 ≤ (sizeCount GameConfig.standard π GameState.init 1 (n + 35)
          - sizeCount GameConfig.standard π GameState.init 1 n)
        + (sizeCount GameConfig.standard π GameState.init 2 (n + 35)
          - sizeCount GameConfig.standard π GameState.init 2 n)
        + (sizeCount GameConfig.standard π GameState.init 3 (n + 35)
          - sizeCount GameConfig.standard π GameState.init 3 n)
        + (sizeCount GameConfig.standard π GameState.init 4 (n + 35)
          - sizeCount GameConfig.standard π GameState.init 4 n)
      ∧ (sizeCount GameConfig.standard π GameState.init 1 (n + 35)
          - sizeCount GameConfig.standard π GameState.init 1 n)
        + (sizeCount GameConfig.standard π GameState.init 2 (n + 35)
          - sizeCount GameConfig.standard π GameState.init 2 n)
        + (sizeCount GameConfig.standard π GameState.init 3 (n + 35)
          - sizeCount GameConfig.standard π GameState.init 3 n)
        + (sizeCount GameConfig.standard π GameState.init 4 (n + 35)
          - sizeCount GameConfig.standard π GameState.init 4 n)
        ≤ 14 := by
  have h := period_mix_fourteen hv hcyc
  exact ⟨by omega, by omega⟩

/-- Multi-period clear-event bounds: over `j` periods, `2·events ≥ 7j` (i.e.
events ≥ 3.5·j) and `events ≤ 14j`. -/
theorem multi_period_clear_events_bounds {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) (j : ℕ) :
    7 * j ≤ 2 * ((sizeCount GameConfig.standard π GameState.init 1 (n + 35 * j)
          - sizeCount GameConfig.standard π GameState.init 1 n)
        + (sizeCount GameConfig.standard π GameState.init 2 (n + 35 * j)
          - sizeCount GameConfig.standard π GameState.init 2 n)
        + (sizeCount GameConfig.standard π GameState.init 3 (n + 35 * j)
          - sizeCount GameConfig.standard π GameState.init 3 n)
        + (sizeCount GameConfig.standard π GameState.init 4 (n + 35 * j)
          - sizeCount GameConfig.standard π GameState.init 4 n))
      ∧ (sizeCount GameConfig.standard π GameState.init 1 (n + 35 * j)
          - sizeCount GameConfig.standard π GameState.init 1 n)
        + (sizeCount GameConfig.standard π GameState.init 2 (n + 35 * j)
          - sizeCount GameConfig.standard π GameState.init 2 n)
        + (sizeCount GameConfig.standard π GameState.init 3 (n + 35 * j)
          - sizeCount GameConfig.standard π GameState.init 3 n)
        + (sizeCount GameConfig.standard π GameState.init 4 (n + 35 * j)
          - sizeCount GameConfig.standard π GameState.init 4 n)
        ≤ 14 * j := by
  have h := multi_period_mix_fourteen hv hcyc j
  exact ⟨by omega, by omega⟩

/-- **No pure-tetris cycle**: a cycle period cannot clear exclusively via
tetrises — `4 ∤ 14`. A tetris-only strategy can never close a loop. -/
theorem no_pure_tetris_period {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35))
    (h1 : sizeCount GameConfig.standard π GameState.init 1 (n + 35)
      = sizeCount GameConfig.standard π GameState.init 1 n)
    (h2 : sizeCount GameConfig.standard π GameState.init 2 (n + 35)
      = sizeCount GameConfig.standard π GameState.init 2 n)
    (h3 : sizeCount GameConfig.standard π GameState.init 3 (n + 35)
      = sizeCount GameConfig.standard π GameState.init 3 n) :
    False := by
  have h := period_mix_fourteen hv hcyc
  omega

/-- **No pure-triple cycle** either: `3 ∤ 14`. Together with
`no_pure_tetris_period`: every cycle's clearing mix must involve a single or
a double (or mix triples with tetrises — `2·3 + 2·4 = 14` is the unique
singles-and-doubles-free period mix). -/
theorem no_pure_triple_period {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35))
    (h1 : sizeCount GameConfig.standard π GameState.init 1 (n + 35)
      = sizeCount GameConfig.standard π GameState.init 1 n)
    (h2 : sizeCount GameConfig.standard π GameState.init 2 (n + 35)
      = sizeCount GameConfig.standard π GameState.init 2 n)
    (h4 : sizeCount GameConfig.standard π GameState.init 4 (n + 35)
      = sizeCount GameConfig.standard π GameState.init 4 n) :
    False := by
  have h := period_mix_fourteen hv hcyc
  omega

/-- The unique singles-and-doubles-free period mix: exactly two triples and
two tetrises. -/
theorem period_mix_no_small_clears {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35))
    (h1 : sizeCount GameConfig.standard π GameState.init 1 (n + 35)
      = sizeCount GameConfig.standard π GameState.init 1 n)
    (h2 : sizeCount GameConfig.standard π GameState.init 2 (n + 35)
      = sizeCount GameConfig.standard π GameState.init 2 n) :
    sizeCount GameConfig.standard π GameState.init 3 (n + 35)
        - sizeCount GameConfig.standard π GameState.init 3 n = 2
      ∧ sizeCount GameConfig.standard π GameState.init 4 (n + 35)
        - sizeCount GameConfig.standard π GameState.init 4 n = 2 := by
  have h := period_mix_fourteen hv hcyc
  exact ⟨by omega, by omega⟩

/-- The period-mix polytope: all clear-size vectors `(a₁, a₂, a₃, a₄)`
weight-summing to fourteen. -/
def periodMixes : Finset (ℕ × ℕ × ℕ × ℕ) :=
  ((Finset.range 15) ×ˢ (Finset.range 8) ×ˢ (Finset.range 5) ×ˢ
      (Finset.range 4)).filter
    (fun v => v.1 + 2 * v.2.1 + 3 * v.2.2.1 + 4 * v.2.2.2 = 14)

set_option maxRecDepth 40000 in
/-- **The period-mix polytope has exactly 47 points**: a cycle period's
clearing profile is one of 47 explicit possibilities. -/
theorem periodMixes_card : periodMixes.card = 47 := by decide

/-- Every cycle period's clear-size delta vector lies in the 47-point
polytope. -/
theorem period_mix_mem_polytope {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    (sizeCount GameConfig.standard π GameState.init 1 (n + 35)
        - sizeCount GameConfig.standard π GameState.init 1 n,
      sizeCount GameConfig.standard π GameState.init 2 (n + 35)
        - sizeCount GameConfig.standard π GameState.init 2 n,
      sizeCount GameConfig.standard π GameState.init 3 (n + 35)
        - sizeCount GameConfig.standard π GameState.init 3 n,
      sizeCount GameConfig.standard π GameState.init 4 (n + 35)
        - sizeCount GameConfig.standard π GameState.init 4 n)
      ∈ periodMixes := by
  have h := period_mix_fourteen hv hcyc
  simp only [periodMixes, Finset.mem_filter, Finset.mem_product,
    Finset.mem_range]
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_⟩ <;> omega

/-- The cooperative size counter agrees with the windowed filter cardinality. -/
theorem sizeCount_eq_card_filter {cfg : GameConfig} {π : Policy cfg} (k n : ℕ) :
    sizeCount cfg π GameState.init k n
      = ((Finset.range n).filter (fun m => (Board.fullRows cfg
          ((π (trace cfg π GameState.init m)).place
            (trace cfg π GameState.init m).board)).card = k)).card := by
  classical
  induction n with
  | zero => simp
  | succ m ih =>
    rw [sizeCount_succ, ih, Finset.range_add_one, Finset.filter_insert]
    split_ifs with h
    · rw [Finset.card_insert_of_notMem (by simp)]
    · omega

/-- The I counter agrees with the windowed filter cardinality. -/
theorem iCount_eq_card_filter {cfg : GameConfig} {π : Policy cfg} (n : ℕ) :
    iCount cfg π GameState.init n
      = ((Finset.range n).filter (fun m =>
          (π (trace cfg π GameState.init m)).piece = Piece.I)).card := by
  classical
  induction n with
  | zero => simp
  | succ m ih =>
    rw [iCount_succ, ih, Finset.range_add_one, Finset.filter_insert]
    split_ifs with h
    · rw [Finset.card_insert_of_notMem (by simp)]
    · omega

/-- The size-counter *increment* over a window is the window's own filter
count — the bridge between the cumulative counters and the windowed cap
theorems. -/
theorem sizeCount_window {cfg : GameConfig} {π : Policy cfg} (k n : ℕ) :
    ∀ w, sizeCount cfg π GameState.init k (n + w)
        - sizeCount cfg π GameState.init k n
      = ((Finset.range w).filter (fun j => (Board.fullRows cfg
          ((π (trace cfg π GameState.init (n + j))).place
            (trace cfg π GameState.init (n + j)).board)).card = k)).card := by
  classical
  intro w
  induction w with
  | zero => simp
  | succ w ih =>
    have hmono := sizeCount_mono cfg π GameState.init k (Nat.le_add_right n w)
    rw [show n + (w + 1) = (n + w) + 1 by omega, sizeCount_succ,
      Finset.range_add_one, Finset.filter_insert]
    split_ifs with h
    · rw [Finset.card_insert_of_notMem (by simp)]
      omega
    · omega

/-- **The survivor's clearing floor, windowed**: a live trace clears at least
`(4w − 200)` tenths-of-rows over any `w`-window — the ledger forces
`10·Δcleared ≥ 4w − 200` whenever the endpoint is alive. -/
theorem survivor_window_clears_floor {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n w : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init (n + w)).lost
      GameConfig.standard) :
    4 * w ≤ 10 * (cleared GameConfig.standard π GameState.init (n + w)
        - cleared GameConfig.standard π GameState.init n) + 200 := by
  have h1 := init_ledger hv n
  have h2 := init_ledger hv (n + w)
  rw [GameConfig.standard_cols] at h1 h2
  have hcap := count_lt_two_hundred_one hv hlive
  have hmono := cleared_mono GameConfig.standard π GameState.init
    (Nat.le_add_right n w)
  omega

/-- **The survivor's event floor**: a live trace clears on at least
`(w − 50)/10` of any `w` placements — clearing events cannot be rarer than
one in ten, sustainably, because each event removes at most four rows and
the ledger demands `0.4` rows per placement. -/
theorem survivor_window_events_floor {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n w : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init (n + w)).lost
      GameConfig.standard) :
    4 * w ≤ 40 * ((sizeCount GameConfig.standard π GameState.init 1 (n + w)
          - sizeCount GameConfig.standard π GameState.init 1 n)
        + (sizeCount GameConfig.standard π GameState.init 2 (n + w)
          - sizeCount GameConfig.standard π GameState.init 2 n)
        + (sizeCount GameConfig.standard π GameState.init 3 (n + w)
          - sizeCount GameConfig.standard π GameState.init 3 n)
        + (sizeCount GameConfig.standard π GameState.init 4 (n + w)
          - sizeCount GameConfig.standard π GameState.init 4 n)) + 200 := by
  have hcl := survivor_window_clears_floor hv hlive
  have hmix := mix_window_identity (cfg := GameConfig.standard) (π := π)
    (Nat.le_add_right n w)
  omega

/-- The windowed clearing ceiling: `10·Δcleared ≤ 4w + 200` whenever the
window *starts* alive — you cannot clear rows that were never delivered. -/
theorem survivor_window_clears_ceiling {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n w : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init n).lost
      GameConfig.standard) :
    10 * (cleared GameConfig.standard π GameState.init (n + w)
        - cleared GameConfig.standard π GameState.init n) ≤ 4 * w + 200 := by
  have h1 := init_ledger hv n
  have h2 := init_ledger hv (n + w)
  rw [GameConfig.standard_cols] at h1 h2
  have hcap := count_lt_two_hundred_one hv hlive
  have hmono := cleared_mono GameConfig.standard π GameState.init
    (Nat.le_add_right n w)
  omega

/-- **The event-rate bracket**: on a live window, clearing events number at
most `(4w + 200)/10` — with the floor, the event rate is pinned to roughly
`[10%, 40%]` of placements at every scale. -/
theorem survivor_window_events_ceiling {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n w : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init n).lost
      GameConfig.standard) :
    10 * ((sizeCount GameConfig.standard π GameState.init 1 (n + w)
          - sizeCount GameConfig.standard π GameState.init 1 n)
        + (sizeCount GameConfig.standard π GameState.init 2 (n + w)
          - sizeCount GameConfig.standard π GameState.init 2 n)
        + (sizeCount GameConfig.standard π GameState.init 3 (n + w)
          - sizeCount GameConfig.standard π GameState.init 3 n)
        + (sizeCount GameConfig.standard π GameState.init 4 (n + w)
          - sizeCount GameConfig.standard π GameState.init 4 n))
      ≤ 4 * w + 200 := by
  have hcl := survivor_window_clears_ceiling hv (w := w) hlive
  have hmix := mix_window_identity (cfg := GameConfig.standard) (π := π)
    (Nat.le_add_right n w)
  omega

end ClearRate
end Tetris
