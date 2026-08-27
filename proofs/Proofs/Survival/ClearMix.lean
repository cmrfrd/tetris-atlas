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

/-- **One tetris in ten, from the very start**: at every horizon, the
number of tetrises is at most a tenth of the moves played — each tetris
clears forty cells' worth of rows on a four-cells-a-move income. Not a
cycle law: it binds every game from move one. -/
theorem tetris_count_le_tenth {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (m : ℕ) :
    10 * sizeCount GameConfig.standard π GameState.init 4 m ≤ m := by
  have hmix := mix_identity (cfg := GameConfig.standard) (π := π) m
  have hled := init_ledger (cfg := GameConfig.standard) hv m
  rw [GameConfig.standard_cols] at hled
  omega

/-- Triples run at most two in fifteen, at every horizon. -/
theorem triple_count_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (m : ℕ) :
    15 * sizeCount GameConfig.standard π GameState.init 3 m ≤ 2 * m := by
  have hmix := mix_identity (cfg := GameConfig.standard) (π := π) m
  have hled := init_ledger (cfg := GameConfig.standard) hv m
  rw [GameConfig.standard_cols] at hled
  omega

/-- **The lifetime clearing speed limit**: no game clears more than two
rows per five moves, at any horizon — `5·cleared(m) ≤ 2m` always. The 2.8
rows/bag cycle rate (`14/35 = 2/5` per move) is the universal ceiling,
binding from move one. -/
theorem cleared_le_two_fifths {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (m : ℕ) :
    5 * cleared GameConfig.standard π GameState.init m ≤ 2 * m := by
  have hled := init_ledger (cfg := GameConfig.standard) hv m
  rw [GameConfig.standard_cols] at hled
  omega

/-- **The clearing-event speed limit**: no game has more than two clearing
moments per five moves — each event costs at least one ten-cell row. -/
theorem clear_events_le_two_fifths {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (m : ℕ) :
    5 * (sizeCount GameConfig.standard π GameState.init 1 m
        + sizeCount GameConfig.standard π GameState.init 2 m
        + sizeCount GameConfig.standard π GameState.init 3 m
        + sizeCount GameConfig.standard π GameState.init 4 m)
      ≤ 2 * m := by
  have hmix := mix_identity (cfg := GameConfig.standard) (π := π) m
  have h := cleared_le_two_fifths hv m
  omega

/-- **Live games must have clearing moments**: `4m ≤ 40·events + 200` —
each event clears at most four rows, so the pinch's clearing floor forces
at least one clearing moment per ten moves past move fifty. -/
theorem live_clear_events_floor {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init m).lost
      GameConfig.standard) :
    4 * m ≤ 40 * (sizeCount GameConfig.standard π GameState.init 1 m
        + sizeCount GameConfig.standard π GameState.init 2 m
        + sizeCount GameConfig.standard π GameState.init 3 m
        + sizeCount GameConfig.standard π GameState.init 4 m)
      + 200 := by
  have hmix := mix_identity (cfg := GameConfig.standard) (π := π) m
  have hfloor := live_clear_floor hv hlive
  omega

/-- **The first clear lands by move fifty-one**: a game still alive at step
51 has cleared at least one row — with `cleared_two_eq_zero`, every live
game's first clear falls somewhere in moves three through fifty-one, both
ends sharp. -/
theorem first_clear_by_fifty_one {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hlive : ¬ (trace GameConfig.standard π GameState.init 51).lost
      GameConfig.standard) :
    1 ≤ cleared GameConfig.standard π GameState.init 51 := by
  have hfloor := live_clear_floor hv hlive
  omega

/-- **The tetris train law**: any `t` tetrises inside a `w`-move window
cost `40t ≤ count(start) + 4w` — a train of tetrises is financed by the
banked mass at the window's start plus the four-cells-a-move income, so
trains are as long as the bank is deep and no longer. Generalizes the
pair law to arbitrary bursts. -/
theorem tetris_train_law {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (m w : ℕ) :
    40 * (sizeCount GameConfig.standard π GameState.init 4 (m + w)
        - sizeCount GameConfig.standard π GameState.init 4 m)
      ≤ (trace GameConfig.standard π GameState.init m).board.count
        + 4 * w := by
  have hmix := mix_window_identity (cfg := GameConfig.standard) (π := π)
    (Nat.le_add_right m w)
  have hled := init_ledger (cfg := GameConfig.standard) hv m
  have hled' := init_ledger (cfg := GameConfig.standard) hv (m + w)
  rw [GameConfig.standard_cols] at hled hled'
  have hclm := cleared_mono GameConfig.standard π GameState.init
    (Nat.le_add_right m w)
  omega

/-- **The window tetris cap**: from any live moment, a `w`-move window
holds at most `5 + w/10` tetrises — the 200-cell bank buys at most five
beyond the steady one-in-ten income rate. The burst allowance of every
Tetris game, exactly quantified. -/
theorem tetris_window_cap {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init m).lost
      GameConfig.standard) (w : ℕ) :
    sizeCount GameConfig.standard π GameState.init 4 (m + w)
        - sizeCount GameConfig.standard π GameState.init 4 m
      ≤ 5 + w / 10 := by
  have h := tetris_train_law hv m w
  have hcap := count_lt_two_hundred_one hv hlive
  omega

/-- **The confinement cap**: even WITH clears, a burst of `w` drops
confined to one adjacent pair that ends low obeys `3w ≤ 2n + 40` — the
window demands two rows per drop but the whole game can only mint 0.4, so
confined bursts are capped by the clearing credit banked before they
start. -/
theorem window_confinement_cap {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n j w : ℕ}
    (hj : j + 1 < 10)
    (hcells : ∀ k < w,
      ∀ cell ∈ (π (trace GameConfig.standard π GameState.init (n + k))).shapeUp,
        (π (trace GameConfig.standard π GameState.init (n + k))).col + cell.1
          = j
        ∨ (π (trace GameConfig.standard π GameState.init (n + k))).col
            + cell.1 = j + 1)
    (hlow : (trace GameConfig.standard π GameState.init
          (n + w)).board.colHeight j + 4 ≤ 20
      ∧ (trace GameConfig.standard π GameState.init
          (n + w)).board.colHeight (j + 1) + 4 ≤ 20) :
    3 * w ≤ 2 * n + 40 := by
  have hrate := window_sustain_clear_rate (n := n) hj hcells hlow
  have hcap := cleared_le_two_fifths hv (n + w)
  omega

/-- **The opening window closes at thirteen**: from the empty board, play
confined to a single adjacent pair can stay low for at most thirteen
drops — clears included. There is no banked credit at move zero, so the
opening forces the window to migrate almost immediately. -/
theorem opening_window_cap {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {j w : ℕ}
    (hj : j + 1 < 10)
    (hcells : ∀ k < w,
      ∀ cell ∈ (π (trace GameConfig.standard π GameState.init k)).shapeUp,
        (π (trace GameConfig.standard π GameState.init k)).col + cell.1 = j
        ∨ (π (trace GameConfig.standard π GameState.init k)).col + cell.1
            = j + 1)
    (hlow : (trace GameConfig.standard π GameState.init
          w).board.colHeight j + 4 ≤ 20
      ∧ (trace GameConfig.standard π GameState.init
          w).board.colHeight (j + 1) + 4 ≤ 20) :
    w ≤ 13 := by
  have h := window_confinement_cap hv (n := 0) hj
    (by simpa using hcells) (by simpa using hlow)
  omega

/-- **The width-`k` clear demand**: `w` drops confined to a column set
`S` ending at a live board force `4w ≤ |S|·clearsΔ + 20|S|` — the set
must absorb four cells per drop and can bank at most twenty per column. -/
theorem confinement_clear_demand {π : Policy GameConfig.standard}
    {n w : ℕ} {S : Finset ℕ} (hS : ∀ j ∈ S, j < 10)
    (hcells : ∀ k < w,
      ∀ cell ∈ (π (trace GameConfig.standard π GameState.init (n + k))).shapeUp,
        (π (trace GameConfig.standard π GameState.init (n + k))).col + cell.1
          ∈ S)
    (hlive : ¬ (trace GameConfig.standard π GameState.init
      (n + w)).lost GameConfig.standard) :
    4 * w ≤ S.card * (cleared GameConfig.standard π GameState.init (n + w)
        - cleared GameConfig.standard π GameState.init n) + 20 * S.card := by
  have hled := window_feed_ledger_set (n := n) hS w hcells
  have hif := (GameState.not_lost_iff_forall_row_lt
    GameConfig.standard _).mp hlive
  have hcolbound : ∀ j ∈ S,
      (trace GameConfig.standard π GameState.init
        (n + w)).board.colCount j ≤ 20 := by
    intro j hj
    have h1 := colCount_le_colHeight
      (trace GameConfig.standard π GameState.init (n + w)).board j
    have h2 := Board.colHeight_le_rows_of_in_field
      (cfg := GameConfig.standard) hif j
    rw [GameConfig.standard_rows] at h2
    omega
  have hsum : (∑ j ∈ S, (trace GameConfig.standard π GameState.init
        (n + w)).board.colCount j) ≤ 20 * S.card := by
    calc (∑ j ∈ S, (trace GameConfig.standard π GameState.init
            (n + w)).board.colCount j)
        ≤ ∑ _j ∈ S, 20 := Finset.sum_le_sum hcolbound
      _ = 20 * S.card := by
          rw [Finset.sum_const, smul_eq_mul, Nat.mul_comm]
  omega

/-- **A survivor never abandons a column**: no surviving policy can
eventually confine its drops to nine or fewer columns. Width-nine play
demands 4/9 of a cleared row per move forever, but the whole game can
only mint 2/5 — and 4/9 > 2/5. Every survivor keeps using all ten
columns, from every point on. The migration crux is now two-sided:
windows must move (burnout), and the board's full width must stay in
play (this theorem). -/
theorem no_eventual_confinement {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hsurv : SurvivesForever GameConfig.standard π GameState.init)
    {S : Finset ℕ} (hS : ∀ j ∈ S, j < 10) (hcard : S.card ≤ 9) {N : ℕ}
    (hconf : ∀ k,
      ∀ cell ∈ (π (trace GameConfig.standard π GameState.init (N + k))).shapeUp,
        (π (trace GameConfig.standard π GameState.init (N + k))).col + cell.1
          ∈ S) :
    False := by
  set w := 9 * N + 451 with hw
  have hdem := confinement_clear_demand (n := N) (w := w) hS
    (fun k _ => hconf k) (hsurv (N + w))
  have hglob := cleared_le_two_fifths hv (N + w)
  have hmul : S.card * (cleared GameConfig.standard π GameState.init (N + w)
        - cleared GameConfig.standard π GameState.init N)
      ≤ 9 * (cleared GameConfig.standard π GameState.init (N + w)
        - cleared GameConfig.standard π GameState.init N) :=
    Nat.mul_le_mul hcard (le_refl _)
  have hmul2 : 20 * S.card ≤ 180 := by omega
  omega

/-- **Every column is fed on schedule**: a surviving policy touches every
column `j` within `9N + 451` moves of any point `N` — abstaining longer
would confine play to the other nine columns past the width-nine budget.
The quantitative form of column non-abandonment. -/
theorem every_column_fed_within {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hsurv : SurvivesForever GameConfig.standard π GameState.init)
    {j : ℕ} (hj : j < 10) (N : ℕ) :
    ∃ k < 9 * N + 451,
      ∃ cell ∈ (π (trace GameConfig.standard π GameState.init (N + k))).shapeUp,
        (π (trace GameConfig.standard π GameState.init (N + k))).col + cell.1
          = j := by
  by_contra hnone
  push Not at hnone
  have hcard : ((Finset.range 10).erase j).card = 9 := by
    rw [Finset.card_erase_of_mem (Finset.mem_range.mpr hj),
      Finset.card_range]
  have hdem := confinement_clear_demand (n := N) (w := 9 * N + 451)
    (S := (Finset.range 10).erase j)
    (fun i hi => Finset.mem_range.mp (Finset.mem_of_mem_erase hi))
    (fun k hk cell hcell => by
      have hlt := (hv _).col_add_lt hcell
      rw [GameConfig.standard_cols] at hlt
      exact Finset.mem_erase.mpr
        ⟨hnone k hk cell hcell, Finset.mem_range.mpr hlt⟩)
    (hsurv (N + (9 * N + 451)))
  rw [hcard] at hdem
  have hglob := cleared_le_two_fifths hv (N + (9 * N + 451))
  omega

/-- **Every column is fed infinitely often**: a surviving policy returns
to every column from every point on — no survivor ever retires a column. -/
theorem every_column_fed_infinitely {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hsurv : SurvivesForever GameConfig.standard π GameState.init)
    {j : ℕ} (hj : j < 10) (N : ℕ) :
    ∃ k, ∃ cell ∈ (π (trace GameConfig.standard π GameState.init (N + k))).shapeUp,
      (π (trace GameConfig.standard π GameState.init (N + k))).col + cell.1
        = j := by
  obtain ⟨k, _, hcell⟩ := every_column_fed_within hv hsurv hj N
  exact ⟨k, hcell⟩

/-- **No fifty-one-move drought, ever**: between any point and fifty-one
moves later, a live game must clear — the ledger banks four cells per
move and a live board holds at most two hundred. The opening bound
`first_clear_by_fifty_one`, promoted to every point of the game. -/
theorem no_clear_drought {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {N : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init
      (N + 51)).lost GameConfig.standard) :
    cleared GameConfig.standard π GameState.init N
      < cleared GameConfig.standard π GameState.init (N + 51) := by
  have h1 := init_ledger hv (N + 51)
  have h2 := init_ledger hv N
  rw [GameConfig.standard_cols] at h1 h2
  have hcap := count_lt_two_hundred_one hv hlive
  omega

/-- A surviving policy is never fifty-one moves from its last clear. -/
theorem survivor_no_clear_drought {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hsurv : SurvivesForever GameConfig.standard π GameState.init) (N : ℕ) :
    cleared GameConfig.standard π GameState.init N
      < cleared GameConfig.standard π GameState.init (N + 51) :=
  no_clear_drought hv (hsurv (N + 51))

/-- **A clearing moment sits in every fifty-one-move window**: some single
step within the window strictly raises the cleared count. The drought
bound, localized to an event. -/
theorem clear_moment_in_window {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {N : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init
      (N + 51)).lost GameConfig.standard) :
    ∃ k < 51,
      cleared GameConfig.standard π GameState.init (N + k)
        < cleared GameConfig.standard π GameState.init ((N + k) + 1) := by
  by_contra hnone
  push Not at hnone
  have hflat : ∀ v, v ≤ 51 →
      cleared GameConfig.standard π GameState.init (N + v)
        = cleared GameConfig.standard π GameState.init N := by
    intro v
    induction v with
    | zero =>
      intro _
      simp
    | succ k ih =>
      intro hv51
      have hle := hnone k (by omega)
      have hmono := cleared_mono GameConfig.standard π GameState.init
        (show N + k ≤ (N + k) + 1 by omega)
      have hik := ih (by omega)
      rw [show N + (k + 1) = (N + k) + 1 by omega]
      omega
  have hdr := no_clear_drought hv hlive
  have h51 := hflat 51 (le_refl _)
  omega

/-- **Full-width play recurs on a fifty-one-move clock**: in every
fifty-one-move live window some merged board occupies all ten columns —
the drought bound and the span certificate composed. Column recurrence
(`every_column_fed_within`) forces each column separately on a `9N+451`
clock; the clearing clock forces all ten AT ONCE, every fifty-one moves. -/
theorem full_width_moment_in_window {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {N : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init
      (N + 51)).lost GameConfig.standard) :
    ∃ k < 51, ∀ j < 10,
      1 ≤ ((π (trace GameConfig.standard π GameState.init (N + k))).place
        (trace GameConfig.standard π GameState.init (N + k)).board).colCount
        j := by
  obtain ⟨k, hk, hjump⟩ := clear_moment_in_window hv hlive
  exact ⟨k, hk, clearing_move_spans_board hjump⟩

/-- **The uniform dwell cap**: play confined to one adjacent pair between
two live moments, ending low, lasts at most forty-six moves — anywhere in
the game. The window demands a clear per drop less eight
(`window_sustain_clear_rate`) while the window band allows at most
0.4-per-move plus one boardful (`cleared_window_band`); the two rates
cross at `w = 46`. Upgrades the opening cap (13, credit-free) to a bound
independent of the starting position: windows must migrate on a
forty-six-move clock, forever. -/
theorem window_dwell_cap {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n j w : ℕ}
    (hj : j + 1 < 10)
    (hlive_n : ¬ (trace GameConfig.standard π GameState.init n).lost
      GameConfig.standard)
    (hlive_nw : ¬ (trace GameConfig.standard π GameState.init (n + w)).lost
      GameConfig.standard)
    (hcells : ∀ k < w,
      ∀ cell ∈ (π (trace GameConfig.standard π GameState.init (n + k))).shapeUp,
        (π (trace GameConfig.standard π GameState.init (n + k))).col + cell.1
          = j
        ∨ (π (trace GameConfig.standard π GameState.init (n + k))).col
            + cell.1 = j + 1)
    (hlow : (trace GameConfig.standard π GameState.init
          (n + w)).board.colHeight j + 4 ≤ 20
      ∧ (trace GameConfig.standard π GameState.init
          (n + w)).board.colHeight (j + 1) + 4 ≤ 20) :
    w ≤ 46 := by
  have hrate := window_sustain_clear_rate (n := n) hj hcells hlow
  have hband := (cleared_window_band hv hlive_n hlive_nw).2
  omega

/-- A survivor's window migrates at least every forty-seven moves: no
adjacent pair hosts forty-seven consecutive confined drops ending low. -/
theorem survivor_window_dwell_cap {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hsurv : SurvivesForever GameConfig.standard π GameState.init)
    {n j w : ℕ} (hj : j + 1 < 10)
    (hcells : ∀ k < w,
      ∀ cell ∈ (π (trace GameConfig.standard π GameState.init (n + k))).shapeUp,
        (π (trace GameConfig.standard π GameState.init (n + k))).col + cell.1
          = j
        ∨ (π (trace GameConfig.standard π GameState.init (n + k))).col
            + cell.1 = j + 1)
    (hlow : (trace GameConfig.standard π GameState.init
          (n + w)).board.colHeight j + 4 ≤ 20
      ∧ (trace GameConfig.standard π GameState.init
          (n + w)).board.colHeight (j + 1) + 4 ≤ 20) :
    w ≤ 46 :=
  window_dwell_cap hv hj (hsurv n) (hsurv (n + w)) hcells hlow

/-- **The set-width sustain rate**: keeping a column set `S` low through
`w` confined drops costs `S.card` clears per four cells beyond the set's
sixteen-per-column capacity: `4w ≤ |S|·clearsΔ + 16|S|`. The sharp form
of the width-`k` demand (each column banks at most sixteen when it ends
four below the ceiling). -/
theorem confinement_sustain_clear_rate {π : Policy GameConfig.standard}
    {n w : ℕ} {S : Finset ℕ} (hS : ∀ j ∈ S, j < 10)
    (hcells : ∀ k < w,
      ∀ cell ∈ (π (trace GameConfig.standard π GameState.init (n + k))).shapeUp,
        (π (trace GameConfig.standard π GameState.init (n + k))).col + cell.1
          ∈ S)
    (hlow : ∀ j ∈ S, (trace GameConfig.standard π GameState.init
      (n + w)).board.colHeight j + 4 ≤ 20) :
    4 * w ≤ S.card * (cleared GameConfig.standard π GameState.init (n + w)
        - cleared GameConfig.standard π GameState.init n) + 16 * S.card := by
  have hled := window_feed_ledger_set (n := n) hS w hcells
  have hsum : (∑ j ∈ S, (trace GameConfig.standard π GameState.init
        (n + w)).board.colCount j) ≤ 16 * S.card := by
    calc (∑ j ∈ S, (trace GameConfig.standard π GameState.init
            (n + w)).board.colCount j)
        ≤ ∑ _j ∈ S, 16 := by
          apply Finset.sum_le_sum
          intro j hj
          have h1 := colCount_le_colHeight
            (trace GameConfig.standard π GameState.init (n + w)).board j
          have h2 := hlow j hj
          omega
      _ = 16 * S.card := by
          rw [Finset.sum_const, smul_eq_mul, Nat.mul_comm]
  omega

/-- **The sharp dwell cap is twenty-two**: play confined to one adjacent
pair between two live moments, ending low, lasts at most twenty-two moves
— the full-strength pair ledger (a clear per two cells beyond capacity)
against the window band. The migration clock, sharpened from forty-six. -/
theorem window_dwell_cap_sharp {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n j w : ℕ}
    (hj : j + 1 < 10)
    (hlive_n : ¬ (trace GameConfig.standard π GameState.init n).lost
      GameConfig.standard)
    (hlive_nw : ¬ (trace GameConfig.standard π GameState.init (n + w)).lost
      GameConfig.standard)
    (hcells : ∀ k < w,
      ∀ cell ∈ (π (trace GameConfig.standard π GameState.init (n + k))).shapeUp,
        (π (trace GameConfig.standard π GameState.init (n + k))).col + cell.1
          = j
        ∨ (π (trace GameConfig.standard π GameState.init (n + k))).col
            + cell.1 = j + 1)
    (hlow : (trace GameConfig.standard π GameState.init
          (n + w)).board.colHeight j + 4 ≤ 20
      ∧ (trace GameConfig.standard π GameState.init
          (n + w)).board.colHeight (j + 1) + 4 ≤ 20) :
    w ≤ 22 := by
  have hled := window_feed_ledger (n := n) hj w hcells
  have hc1 := colCount_le_colHeight
    (trace GameConfig.standard π GameState.init (n + w)).board j
  have hc2 := colCount_le_colHeight
    (trace GameConfig.standard π GameState.init (n + w)).board (j + 1)
  have hband := (cleared_window_band hv hlive_n hlive_nw).2
  omega

/-- **The low-pair crux migrates on a twenty-three clock**: any policy
solving Tetris through the halfway capstone — a selection `jf` of one low
adjacent pair per step, always played into — can never hold the same pair
for twenty-four consecutive steps. Survival itself follows from the
selection (`survivesForever_of_low_pair_play`), and the sharp dwell cap
then bounds every constant run. The remaining crux is not just
maintaining a low window but maintaining a *moving* one: the window must
step to a fresh low pair at least once every twenty-three moves,
forever. -/
theorem low_pair_selection_migrates {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {jf : ℕ → ℕ}
    (hsel : ∀ n, jf n + 1 < 10
      ∧ ((trace GameConfig.standard π GameState.init n).board.colHeight
            (jf n) + 4 ≤ 20
        ∧ (trace GameConfig.standard π GameState.init n).board.colHeight
            (jf n + 1) + 4 ≤ 20)
      ∧ ∀ cell ∈ (π (trace GameConfig.standard π GameState.init n)).shapeUp,
          (π (trace GameConfig.standard π GameState.init n)).col + cell.1
            = jf n
          ∨ (π (trace GameConfig.standard π GameState.init n)).col + cell.1
            = jf n + 1)
    {N : ℕ} :
    ¬ (∀ k ≤ 23, jf (N + k) = jf N) := by
  intro hconst
  have hsurv : SurvivesForever GameConfig.standard π GameState.init := by
    apply survivesForever_of_low_pair_play
    intro n
    obtain ⟨h1, h2, h3⟩ := hsel n
    exact ⟨jf n, h1, h2.1, h2.2, h3⟩
  have hcap := window_dwell_cap_sharp hv (n := N) (j := jf N) (w := 23)
    (hsel N).1 (hsurv N) (hsurv (N + 23))
    (fun k hk => by
      have h3 := (hsel (N + k)).2.2
      have hje := hconst k (by omega)
      rw [hje] at h3
      exact h3)
    (by
      have h2 := (hsel (N + 23)).2.1
      have hje := hconst 23 (le_refl _)
      rw [hje] at h2
      exact h2)
  omega

/-- **The moving window sweeps the whole board**: a halfway-capstone
selection reaches every column from every point on — within `9N + 451`
steps of any moment `N`, the selected pair touches column `j`. Combined
with `low_pair_selection_migrates`, the crux's shape is now fully
constrained: a low pair that steps at least every twenty-three moves and
sweeps all ten columns on a linear schedule, forever. -/
theorem low_pair_selection_covers {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {jf : ℕ → ℕ}
    (hsel : ∀ n, jf n + 1 < 10
      ∧ ((trace GameConfig.standard π GameState.init n).board.colHeight
            (jf n) + 4 ≤ 20
        ∧ (trace GameConfig.standard π GameState.init n).board.colHeight
            (jf n + 1) + 4 ≤ 20)
      ∧ ∀ cell ∈ (π (trace GameConfig.standard π GameState.init n)).shapeUp,
          (π (trace GameConfig.standard π GameState.init n)).col + cell.1
            = jf n
          ∨ (π (trace GameConfig.standard π GameState.init n)).col + cell.1
            = jf n + 1)
    {j : ℕ} (hj : j < 10) (N : ℕ) :
    ∃ k < 9 * N + 451, jf (N + k) = j ∨ jf (N + k) + 1 = j := by
  have hsurv : SurvivesForever GameConfig.standard π GameState.init := by
    apply survivesForever_of_low_pair_play
    intro n
    obtain ⟨h1, h2, h3⟩ := hsel n
    exact ⟨jf n, h1, h2.1, h2.2, h3⟩
  obtain ⟨k, hk, cell, hcell, hcol⟩ := every_column_fed_within hv hsurv hj N
  refine ⟨k, hk, ?_⟩
  rcases (hsel (N + k)).2.2 cell hcell with h | h
  · left
    omega
  · right
    omega

/-- A cell of a placed board lying outside the drop's columns was already
on the pre-drop board. -/
theorem place_mem_of_col_notin {b : Board} {pl : Placement} {p : ℕ × ℕ}
    (hp : p ∈ pl.place b)
    (hout : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 ≠ p.1) : p ∈ b := by
  rw [Placement.place_eq_union_dropped] at hp
  rcases Finset.mem_union.mp hp with h | h
  · exact h
  · exfalso
    unfold Placement.dropped Placement.cellsAt at h
    obtain ⟨cell, hcell, heq⟩ := Finset.mem_image.mp h
    have h1 := congrArg Prod.fst heq
    exact hout cell hcell h1

/-- **The moving window harvests prepared rows**: when a capstone
selection clears, the cleared row's cells in the eight columns outside
the selected pair were already standing before the drop — the window
only reaps what earlier sweeps sowed. The inventory story of the moving
window, made pointwise. -/
theorem capstone_clear_harvests_prepared_rows {π : Policy GameConfig.standard}
    {jf : ℕ → ℕ}
    (hsel : ∀ n, jf n + 1 < 10
      ∧ ((trace GameConfig.standard π GameState.init n).board.colHeight
            (jf n) + 4 ≤ 20
        ∧ (trace GameConfig.standard π GameState.init n).board.colHeight
            (jf n + 1) + 4 ≤ 20)
      ∧ ∀ cell ∈ (π (trace GameConfig.standard π GameState.init n)).shapeUp,
          (π (trace GameConfig.standard π GameState.init n)).col + cell.1
            = jf n
          ∨ (π (trace GameConfig.standard π GameState.init n)).col + cell.1
            = jf n + 1)
    {m : ℕ}
    (hjump : cleared GameConfig.standard π GameState.init m
      < cleared GameConfig.standard π GameState.init (m + 1)) :
    ∃ r, Board.isFull GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board) r
      ∧ ∀ c < 10, c ≠ jf m → c ≠ jf m + 1 →
          (c, r) ∈ (trace GameConfig.standard π GameState.init m).board := by
  have hs := cleared_succ GameConfig.standard π GameState.init m
  have hpos : 0 < (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init m)).place
        (trace GameConfig.standard π GameState.init m).board)).card := by
    omega
  obtain ⟨r, hr⟩ := Finset.card_pos.mp hpos
  have hfull := (Finset.mem_filter.mp hr).2
  refine ⟨r, hfull, ?_⟩
  intro c hc hne1 hne2
  have hmem : (c, r) ∈ (π (trace GameConfig.standard π GameState.init m)).place
      (trace GameConfig.standard π GameState.init m).board := by
    apply hfull
    rw [GameConfig.standard_cols]
    exact Finset.mem_range.mpr hc
  apply place_mem_of_col_notin hmem
  intro cell hcell hceq
  have hceq' : (π (trace GameConfig.standard π GameState.init m)).col + cell.1
      = c := hceq
  rcases (hsel m).2.2 cell hcell with h | h
  · exact hne1 (by omega)
  · exact hne2 (by omega)

/-- **Cleared rows are prepared rows**: at any clearing moment, the
cleared row's cells in every column the drop does not touch were already
standing on the pre-drop board. -/
theorem clear_row_prepared_outside_touched {π : Policy GameConfig.standard}
    {m : ℕ}
    (hjump : cleared GameConfig.standard π GameState.init m
      < cleared GameConfig.standard π GameState.init (m + 1)) :
    ∃ r, Board.isFull GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board) r
      ∧ ∀ c < 10,
          (π (trace GameConfig.standard π GameState.init m)).colProfile c = 0 →
          (c, r) ∈ (trace GameConfig.standard π GameState.init m).board := by
  have hs := cleared_succ GameConfig.standard π GameState.init m
  have hpos : 0 < (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init m)).place
        (trace GameConfig.standard π GameState.init m).board)).card := by
    omega
  obtain ⟨r, hr⟩ := Finset.card_pos.mp hpos
  have hfull := (Finset.mem_filter.mp hr).2
  refine ⟨r, hfull, ?_⟩
  intro c hc hprof0
  have hmem : (c, r) ∈ (π (trace GameConfig.standard π GameState.init m)).place
      (trace GameConfig.standard π GameState.init m).board := by
    apply hfull
    rw [GameConfig.standard_cols]
    exact Finset.mem_range.mpr hc
  apply place_mem_of_col_notin hmem
  intro cell hcell hceq
  have hceq' : (π (trace GameConfig.standard π GameState.init m)).col + cell.1
      = c := hceq
  have hmemf : cell ∈ (π (trace GameConfig.standard π
      GameState.init m)).shapeUp.filter
      (fun cell => (π (trace GameConfig.standard π GameState.init m)).col
        + cell.1 = c) :=
    Finset.mem_filter.mpr ⟨hcell, hceq'⟩
  unfold Placement.colProfile at hprof0
  rw [Finset.card_eq_zero] at hprof0
  rw [hprof0] at hmemf
  exact absurd hmemf (Finset.notMem_empty cell)

/-- **Six cells banked in every cleared row**: any clearing moment reaps
a row at least six of whose ten cells were standing before the drop — a
piece touches at most four columns, so at least six columns' worth of
the row is prior inventory. -/
theorem clear_row_six_banked {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (hjump : cleared GameConfig.standard π GameState.init m
      < cleared GameConfig.standard π GameState.init (m + 1)) :
    ∃ r, Board.isFull GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board) r
      ∧ 6 ≤ Board.rowCount
          (trace GameConfig.standard π GameState.init m).board r := by
  classical
  obtain ⟨r, hfull, hprep⟩ := clear_row_prepared_outside_touched hjump
  refine ⟨r, hfull, ?_⟩
  have huntouch := placement_untouched_columns_ge_six
    (hv (trace GameConfig.standard π GameState.init m))
  have hsub : ((Finset.range 10).filter (fun j =>
        (π (trace GameConfig.standard π GameState.init m)).colProfile j
          = 0)).image (fun c => (c, r))
      ⊆ (trace GameConfig.standard π GameState.init m).board.filter
          (fun p => p.2 = r) := by
    intro p hp
    obtain ⟨c, hc, rfl⟩ := Finset.mem_image.mp hp
    rw [Finset.mem_filter] at hc ⊢
    exact ⟨hprep c (Finset.mem_range.mp hc.1) hc.2, rfl⟩
  have hcardim : (((Finset.range 10).filter (fun j =>
        (π (trace GameConfig.standard π GameState.init m)).colProfile j
          = 0)).image (fun c => (c, r))).card
      = ((Finset.range 10).filter (fun j =>
        (π (trace GameConfig.standard π GameState.init m)).colProfile j
          = 0)).card :=
    Finset.card_image_of_injective _ (fun a b h => (Prod.ext_iff.mp h).1)
  have hle := Finset.card_le_card hsub
  unfold Board.rowCount
  exact le_trans (le_trans huntouch hcardim.symm.le) hle

/-- **The sharp inventory law**: a step clearing `k` rows reaps at least
`10k − 4` cells that were banked on the board before the drop — the
cleared rows hold exactly ten cells each and the piece contributed at
most four in total. Sharpens the per-row six-banked bound to the whole
harvest. -/
theorem cleared_rows_prior_inventory {π : Policy GameConfig.standard}
    {m : ℕ} :
    10 * (cleared GameConfig.standard π GameState.init (m + 1)
        - cleared GameConfig.standard π GameState.init m)
      ≤ ((trace GameConfig.standard π GameState.init m).board.filter
          (fun p => p.2 ∈ Board.fullRows GameConfig.standard
            ((π (trace GameConfig.standard π GameState.init m)).place
              (trace GameConfig.standard π GameState.init m).board))).card
        + 4 := by
  classical
  set pl := π (trace GameConfig.standard π GameState.init m) with hpl
  set b := (trace GameConfig.standard π GameState.init m).board with hb
  set F := Board.fullRows GameConfig.standard (pl.place b) with hF
  have hs := cleared_succ GameConfig.standard π GameState.init m
  rw [← hpl, ← hb, ← hF] at hs
  -- the cleared rows' cells inside the merged board: at least 10·|F|
  have hsub : F.biUnion
      (fun r => (Finset.range 10).image (fun c => (c, r)))
      ⊆ (pl.place b).filter (fun p => p.2 ∈ F) := by
    intro p hp
    obtain ⟨r, hr, hpmem⟩ := Finset.mem_biUnion.mp hp
    obtain ⟨c, hc, rfl⟩ := Finset.mem_image.mp hpmem
    rw [Finset.mem_filter]
    have hfull := (Finset.mem_filter.mp hr).2
    refine ⟨?_, hr⟩
    apply hfull
    rw [GameConfig.standard_cols]
    exact hc
  have hdisj : ∀ r₁ ∈ F, ∀ r₂ ∈ F, r₁ ≠ r₂ →
      Disjoint ((Finset.range 10).image (fun c => (c, r₁)))
        ((Finset.range 10).image (fun c => (c, r₂))) := by
    intro r₁ _ r₂ _ hne
    rw [Finset.disjoint_left]
    intro p hp1 hp2
    obtain ⟨c1, _, rfl⟩ := Finset.mem_image.mp hp1
    obtain ⟨c2, _, heq⟩ := Finset.mem_image.mp hp2
    have hsnd := congrArg Prod.snd heq
    exact hne hsnd.symm
  have hbicard : (F.biUnion
      (fun r => (Finset.range 10).image (fun c => (c, r)))).card
      = 10 * F.card := by
    rw [Finset.card_biUnion hdisj]
    have himg : ∀ r : ℕ,
        ((Finset.range 10).image (fun c => (c, r))).card = 10 := by
      intro r
      rw [Finset.card_image_of_injective _
        (fun a b h => (Prod.ext_iff.mp h).1), Finset.card_range]
    rw [Finset.sum_congr rfl (fun r _ => himg r), Finset.sum_const,
      smul_eq_mul, Nat.mul_comm]
  have hlow := Finset.card_le_card hsub
  -- merged cells split into pre-board cells and the (≤ 4) dropped cells
  have hsplit : (pl.place b).filter (fun p => p.2 ∈ F)
      ⊆ (b.filter (fun p => p.2 ∈ F)) ∪ pl.dropped b := by
    intro p hp
    rw [Finset.mem_filter] at hp
    have hpm := hp.1
    rw [Placement.place_eq_union_dropped] at hpm
    rcases Finset.mem_union.mp hpm with h | h
    · exact Finset.mem_union_left _ (Finset.mem_filter.mpr ⟨h, hp.2⟩)
    · exact Finset.mem_union_right _ h
  have hdropcard : (pl.dropped b).card ≤ 4 := by
    unfold Placement.dropped Placement.cellsAt
    calc (pl.shapeUp.image
          (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2))).card
        ≤ pl.shapeUp.card := Finset.card_image_le
      _ = 4 := pl.shapeUp_card
  have hup := Finset.card_le_card hsplit
  have hunion := Finset.card_union_le (b.filter (fun p => p.2 ∈ F))
    (pl.dropped b)
  have hdelta : cleared GameConfig.standard π GameState.init (m + 1)
      - cleared GameConfig.standard π GameState.init m = F.card := by
    omega
  rw [hdelta, ← hbicard]
  exact le_trans hlow (le_trans hup (le_trans hunion
    (Nat.add_le_add_left hdropcard _)))

/-- **Clears draw on standing stock**: a step clearing `k` rows starts
from a board holding at least `10k − 4` cells — the harvest minus the
piece's own contribution must already be on the table. -/
theorem clearing_move_count_floor {π : Policy GameConfig.standard} {m : ℕ} :
    10 * (cleared GameConfig.standard π GameState.init (m + 1)
        - cleared GameConfig.standard π GameState.init m)
      ≤ (trace GameConfig.standard π GameState.init m).board.count + 4 := by
  have hinv := cleared_rows_prior_inventory (π := π) (m := m)
  have hsub := Finset.card_filter_le
    (trace GameConfig.standard π GameState.init m).board
    (fun p => p.2 ∈ Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init m)).place
        (trace GameConfig.standard π GameState.init m).board))
  unfold Board.count
  omega

/-- **A tetris needs thirty-six on the table**: any four-clear starts
from a board holding at least thirty-six cells. The inventory price of
the biggest harvest, as a pure counting fact. -/
theorem tetris_requires_thirty_six_banked {π : Policy GameConfig.standard}
    {m : ℕ}
    (h4 : cleared GameConfig.standard π GameState.init (m + 1)
      = cleared GameConfig.standard π GameState.init m + 4) :
    36 ≤ (trace GameConfig.standard π GameState.init m).board.count := by
  have hfloor := clearing_move_count_floor (π := π) (m := m)
  omega

/-- **Every harvest needs a tower**: a step clearing `k` rows starts from
a board with some column already standing at least `k` high — the
`10k − 4` banked cells cannot lie flatter than a tenth of their mass. -/
theorem clearing_move_requires_tower {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ} :
    ∃ j < 10,
      cleared GameConfig.standard π GameState.init (m + 1)
        - cleared GameConfig.standard π GameState.init m
      ≤ (trace GameConfig.standard π GameState.init m).board.colHeight j := by
  have hfloor := clearing_move_count_floor (π := π) (m := m)
  have hwf := trace_board_wf hv
    (GameState.init_board_wf GameConfig.standard) m
  obtain ⟨j, hj, htall⟩ := exists_tall_column hwf
  exact ⟨j, hj, by omega⟩

/-- **A tetris needs a four-high tower**: any four-clear starts from a
board with some column at least four high — the well's neighbour was
standing before the I arrived. -/
theorem tetris_requires_tower {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (h4 : cleared GameConfig.standard π GameState.init (m + 1)
      = cleared GameConfig.standard π GameState.init m + 4) :
    ∃ j < 10,
      4 ≤ (trace GameConfig.standard π GameState.init m).board.colHeight j := by
  obtain ⟨j, hj, htower⟩ := clearing_move_requires_tower hv (m := m)
  exact ⟨j, hj, by omega⟩

/-- **While the window dwells, the rest of the board only sinks**: over
any run of drops confined to the pair `(j, j+1)`, every other column's
height is non-increasing — off-pair columns are never fed, so merges
leave them alone and clears can only lower them. Dwelling is not idle
elsewhere: the eight spectator columns flatten (or hold) throughout. -/
theorem dwell_off_pair_heights_sink {π : Policy GameConfig.standard}
    {n j c : ℕ} (hc : c < 10) (hcj : c ≠ j) (hcj1 : c ≠ j + 1) :
    ∀ w, (∀ k < w,
      ∀ cell ∈ (π (trace GameConfig.standard π GameState.init (n + k))).shapeUp,
        (π (trace GameConfig.standard π GameState.init (n + k))).col + cell.1
          = j
        ∨ (π (trace GameConfig.standard π GameState.init (n + k))).col
            + cell.1 = j + 1) →
      (trace GameConfig.standard π GameState.init (n + w)).board.colHeight c
        ≤ (trace GameConfig.standard π GameState.init n).board.colHeight c := by
  intro w
  induction w with
  | zero =>
    intro _
    exact le_refl _
  | succ k ih =>
    intro hcells
    have hihk := ih (fun i hi => hcells i (by omega))
    have hz : (π (trace GameConfig.standard π
        GameState.init (n + k))).colProfile c = 0 := by
      unfold Placement.colProfile
      rw [Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
      intro cell hmem
      rw [Finset.mem_filter] at hmem
      rcases hcells k (by omega) cell hmem.1 with h | h
      · omega
      · omega
    have hstep := applyStep_unfed_colHeight_le (cfg := GameConfig.standard)
      (b := (trace GameConfig.standard π GameState.init (n + k)).board)
      (pl := π (trace GameConfig.standard π GameState.init (n + k)))
      (by rw [GameConfig.standard_cols]; omega) hz
    rw [show n + (k + 1) = (n + k) + 1 by omega, trace_succ,
      GameState.step_board]
    exact le_trans hstep hihk

/-- **The reserve pair never spoils**: if a second low pair `(j', j'+1)`
disjoint from the active window stands ready when a dwell begins, it is
still low when the dwell ends — spectator columns only sink. The
migration step of the moving-window crux is safe whenever a reserve
exists: the open question shrinks to *keeping a reserve*, not to landing
on one. -/
theorem dwell_reserve_stays_ready {π : Policy GameConfig.standard}
    {n j j' w : ℕ} (hj' : j' + 1 < 10)
    (hd1 : j' ≠ j) (hd2 : j' ≠ j + 1) (hd3 : j' + 1 ≠ j)
    (hd4 : j' + 1 ≠ j + 1)
    (hcells : ∀ k < w,
      ∀ cell ∈ (π (trace GameConfig.standard π GameState.init (n + k))).shapeUp,
        (π (trace GameConfig.standard π GameState.init (n + k))).col + cell.1
          = j
        ∨ (π (trace GameConfig.standard π GameState.init (n + k))).col
            + cell.1 = j + 1)
    (hlow : (trace GameConfig.standard π GameState.init n).board.colHeight j'
        + 4 ≤ 20
      ∧ (trace GameConfig.standard π GameState.init n).board.colHeight
          (j' + 1) + 4 ≤ 20) :
    (trace GameConfig.standard π GameState.init (n + w)).board.colHeight j'
        + 4 ≤ 20
      ∧ (trace GameConfig.standard π GameState.init (n + w)).board.colHeight
          (j' + 1) + 4 ≤ 20 := by
  have hs1 := dwell_off_pair_heights_sink (π := π) (n := n) (j := j)
    (c := j') (by omega) hd1 hd2 w hcells
  have hs2 := dwell_off_pair_heights_sink (π := π) (n := n) (j := j)
    (c := j' + 1) (by omega) hd3 hd4 w hcells
  exact ⟨by omega, by omega⟩

end ClearRate
end Tetris
