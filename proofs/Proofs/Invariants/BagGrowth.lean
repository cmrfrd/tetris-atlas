import Mathlib
import Proofs.Model.Placement
import Proofs.Model.Game
import Proofs.Combinatorics.BoardCount
import Proofs.Invariants.ColumnCount
import Proofs.Invariants.Holes

/-!
# Bag growth — the displacement program at full width (10 columns, 7-bag)

The Bousch–Mairesse displacement calculus (topical IFS, JAMS 2001)
instantiated on the real board: certified facts about the asymptotic
height-growth rate of adversarial stacking and the clearing rate any
surviving strategy must sustain.

**The mass floor (`height_floor`)**: clear-free stacking of `n` valid
placements from the empty board pushes the skyline peak to at least
`4n / cols` — at standard width, `2.8` rows per bag. This is the
unconditional lower half of the growth rate: *every* strategy, against
*every* sequence, grows at least this fast without clears.

**The exact clear balance (`count_clearLines_add_cols`,
`count_playFrom_add_cleared`)**: each cleared row removes exactly one cell
per column, so mass obeys `count + cols·cleared = 4·pieces` along any real
game. Corollary **`clearing_rate`**: an in-field game of `n` placements has
cleared at least `(4n − cols·rows) / cols` rows — any surviving strategy
clears `2.8` rows per bag asymptotically, the rate-coupling law behind the
zone-route refutations, now a theorem of the real dynamics.

**The T-parity renewal obstruction (`no_five_bag_perfect_rectangle`,
`perfect_rectangle_bag_period_even`)**: density-1 renewal — stacking some
number of bags into a perfect `10 × h` rectangle, the flat-to-flat
certificate that would pin the growth rate at the mass floor — carries a
parity law: the checkerboard charge (mod 2) of a placed tetromino is `1`
exactly for the T piece (`charge_shapeUp`), every translation preserving it
(4-cell shapes shift by `4·parity ≡ 0`). A perfect `10 × h` box has charge
`h (mod 2)`, so a perfect rectangle forces `#T ≡ h (mod 2)`
(`perfect_rectangle_T_parity`). Five bags carry five T pieces against a
`10 × 14` box: odd against even — **no five-bag perfect rectangle exists**,
upgrading the measured flush perfect-clear wall to a theorem (no validity or
multiset hypotheses beyond the T count). In general the bag period of any
perfect-rectangle renewal must be **even**: the shortest density-1
flat-to-flat cycle compatible with the 7-bag is 10 bags (28 rows), the
lattice point where the tessellation and rate analyses already converged.
-/

namespace Tetris
namespace BagGrowth

open Tetris

/-! ## Clear-free stacking and the real game -/

/-- Clear-free stacking: fold bare placements (heap growth, no line clears). -/
def stackFrom (b : Board) (L : List Placement) : Board :=
  L.foldl (fun b pl => pl.place b) b

/-- Clear-free stack from the empty board. -/
def stack (L : List Placement) : Board := stackFrom ∅ L

/-- The real game: fold full moves (hard drop, then line clears). -/
def playFrom (cfg : GameConfig) (b : Board) (L : List Placement) : Board :=
  L.foldl (fun b pl => pl.applyStep cfg b) b

/-- Total rows cleared along a real game. -/
def clearedRows (cfg : GameConfig) : Board → List Placement → ℕ
  | _, [] => 0
  | b, pl :: rest =>
    (Board.fullRows cfg (pl.place b)).card
      + clearedRows cfg (pl.applyStep cfg b) rest

@[simp] theorem stackFrom_nil (b : Board) : stackFrom b [] = b := rfl

theorem stackFrom_cons (b : Board) (pl : Placement) (rest : List Placement) :
    stackFrom b (pl :: rest) = stackFrom (pl.place b) rest := rfl

@[simp] theorem playFrom_nil (cfg : GameConfig) (b : Board) :
    playFrom cfg b [] = b := rfl

theorem playFrom_cons (cfg : GameConfig) (b : Board) (pl : Placement)
    (rest : List Placement) :
    playFrom cfg b (pl :: rest) = playFrom cfg (pl.applyStep cfg b) rest := rfl

/-- Clear-free stacking adds exactly 4 cells per placement. -/
theorem count_stackFrom (b : Board) (L : List Placement) :
    (stackFrom b L).count = b.count + 4 * L.length := by
  induction L generalizing b with
  | nil => simp
  | cons pl rest ih =>
    rw [stackFrom_cons, ih, Placement.count_place, List.length_cons]
    ring

/-- Clear-free stacking preserves well-formedness for valid placements. -/
theorem stackFrom_wf {cfg : GameConfig} {b : Board} (hb : Board.WF cfg b)
    {L : List Placement} (hL : ∀ pl ∈ L, pl.Valid cfg) :
    Board.WF cfg (stackFrom b L) := by
  induction L generalizing b with
  | nil => exact hb
  | cons pl rest ih =>
    rw [stackFrom_cons]
    exact ih (Placement.place_wf hb (hL pl List.mem_cons_self))
      (fun q hq => hL q (List.mem_cons_of_mem pl hq))

/-! ## The capacity ledger -/

/-- A column's cell count is its row-set's cardinality. -/
theorem colCount_eq_colRows_card (b : Board) (j : ℕ) :
    b.colCount j = (b.colRows j).card := by
  unfold Board.colCount Board.colRows
  rw [Finset.card_image_of_injOn]
  intro p hp q hq hpq
  rw [Finset.mem_coe, Finset.mem_filter] at hp hq
  exact Prod.ext (hp.2.trans hq.2.symm) hpq

/-- Mass is bounded by width times the skyline peak. -/
theorem count_le_cols_mul_sup {cfg : GameConfig} {b : Board}
    (hb : Board.WF cfg b) :
    b.count ≤ cfg.cols * (Finset.range cfg.cols).sup b.colHeight := by
  rw [← Board.sum_colCount hb]
  calc ∑ j ∈ Finset.range cfg.cols, b.colCount j
      ≤ ∑ _j ∈ Finset.range cfg.cols,
          (Finset.range cfg.cols).sup b.colHeight := by
        apply Finset.sum_le_sum
        intro j hj
        rw [colCount_eq_colRows_card]
        exact le_trans (Board.colRows_card_le_colHeight b j)
          (Finset.le_sup (f := b.colHeight) hj)
    _ = cfg.cols * (Finset.range cfg.cols).sup b.colHeight := by
        rw [Finset.sum_const, Finset.card_range, smul_eq_mul]

/-- Mass is bounded by the playfield capacity on in-field boards. -/
theorem count_le_capacity {cfg : GameConfig} {b : Board}
    (hb : Board.WF cfg b) (hf : ∀ p ∈ b, p.2 < cfg.rows) :
    b.count ≤ cfg.cols * cfg.rows := by
  calc b.count ≤ cfg.cols * (Finset.range cfg.cols).sup b.colHeight :=
        count_le_cols_mul_sup hb
    _ ≤ cfg.cols * cfg.rows := by
        apply Nat.mul_le_mul_left
        apply Finset.sup_le
        intro j _
        exact Board.colHeight_le_rows_of_in_field hf j

/-! ## The mass floor: λ ≥ 2.8 rows per bag, unconditionally -/

/-- **The mass floor.** Clear-free stacking of `n` valid placements from the
empty board forces the skyline peak up at rate `4/cols` rows per piece: no
strategy, against no sequence, grows slower. At standard width with 7-piece
bags this reads `28·bags ≤ 10·peak` — the growth rate of adversarial
clear-free Tetris is at least `2.8` rows per bag. The lower half of the
Bousch–Mairesse displacement sandwich, at full width. -/
theorem height_floor {cfg : GameConfig} {L : List Placement}
    (hL : ∀ pl ∈ L, pl.Valid cfg) :
    4 * L.length ≤ cfg.cols * (Finset.range cfg.cols).sup (stack L).colHeight := by
  have hwf : Board.WF cfg (stack L) := stackFrom_wf (Board.empty_wf cfg) hL
  have hcount : (stack L).count = 4 * L.length := by
    rw [stack, count_stackFrom]
    simp [Board.count]
  rw [← hcount]
  exact count_le_cols_mul_sup hwf

/-- The mass floor at standard width, per bag: `m` full bags (`7m` valid
placements) push the peak to at least `⌈2.8 m⌉`. -/
theorem height_floor_bags {L : List Placement} {m : ℕ}
    (hL : ∀ pl ∈ L, pl.Valid GameConfig.standard) (hlen : L.length = 7 * m) :
    28 * m ≤ 10 * (Finset.range 10).sup (stack L).colHeight := by
  have h := height_floor (cfg := GameConfig.standard) hL
  rw [hlen, GameConfig.standard_cols] at h
  omega

/-! ## The exact clear balance and the clearing rate -/

/-- **Board-level clear balance**: line clears remove exactly `cols` cells
per cleared row — one from every column (`colCount_clearLines_add`, summed). -/
theorem count_clearLines_add_cols {cfg : GameConfig} {b : Board}
    (hb : Board.WF cfg b) :
    b.count = (Board.clearLines cfg b).count
      + cfg.cols * (Board.fullRows cfg b).card := by
  rw [← Board.sum_colCount hb, ← Board.sum_colCount (Board.clearLines_wf hb)]
  calc ∑ j ∈ Finset.range cfg.cols, b.colCount j
      = ∑ j ∈ Finset.range cfg.cols,
          ((Board.clearLines cfg b).colCount j + (Board.fullRows cfg b).card) := by
        apply Finset.sum_congr rfl
        intro j hj
        exact Board.colCount_clearLines_add cfg b (Finset.mem_range.mp hj)
    _ = (∑ j ∈ Finset.range cfg.cols, (Board.clearLines cfg b).colCount j)
          + cfg.cols * (Board.fullRows cfg b).card := by
        rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range,
          smul_eq_mul]

/-- Per-move mass balance of the real game: a move adds 4 cells and removes
`cols` per cleared row. -/
theorem count_applyStep_add {cfg : GameConfig} {b : Board} {pl : Placement}
    (hb : Board.WF cfg b) (hv : pl.Valid cfg) :
    (pl.applyStep cfg b).count
      + cfg.cols * (Board.fullRows cfg (pl.place b)).card = b.count + 4 := by
  have h := count_clearLines_add_cols (Placement.place_wf hb hv)
  rw [Placement.count_place] at h
  rw [Placement.applyStep_eq_clearLines_place]
  omega

/-- **Mass conservation along the real game**: final mass plus `cols` cells
per cleared row equals 4 cells per placement. -/
theorem count_playFrom_add_cleared {cfg : GameConfig} {b : Board}
    (hb : Board.WF cfg b) {L : List Placement} (hL : ∀ pl ∈ L, pl.Valid cfg) :
    (playFrom cfg b L).count + cfg.cols * clearedRows cfg b L
      = b.count + 4 * L.length := by
  induction L generalizing b with
  | nil => simp [clearedRows]
  | cons pl rest ih =>
    have hv : pl.Valid cfg := hL pl List.mem_cons_self
    have hstep := count_applyStep_add hb hv
    have hrest := ih (Placement.applyStep_wf hb hv)
      (fun q hq => hL q (List.mem_cons_of_mem pl hq))
    rw [playFrom_cons, clearedRows, List.length_cons]
    rw [Nat.mul_add] at *
    omega

/-- The real game preserves well-formedness for valid placements. -/
theorem playFrom_wf {cfg : GameConfig} {b : Board} (hb : Board.WF cfg b)
    {L : List Placement} (hL : ∀ pl ∈ L, pl.Valid cfg) :
    Board.WF cfg (playFrom cfg b L) := by
  induction L generalizing b with
  | nil => exact hb
  | cons pl rest ih =>
    rw [playFrom_cons]
    exact ih (Placement.applyStep_wf hb (hL pl List.mem_cons_self))
      (fun q hq => hL q (List.mem_cons_of_mem pl hq))

/-- **The clearing rate.** Any real game of `n` valid placements from the
empty board that ends in-field (all cells below row `rows`) has cleared at
least `(4n − cols·rows) / cols` rows. At standard: a surviving strategy
clears at least `(28m − 200)/10` rows in `m` bags — `2.8` rows per bag
asymptotically. Survival is a treadmill: the clear duty is not optional and
its rate is pinned by mass conservation. -/
theorem clearing_rate {cfg : GameConfig} {L : List Placement}
    (hL : ∀ pl ∈ L, pl.Valid cfg)
    (hfield : ∀ p ∈ playFrom cfg (∅ : Board) L, p.2 < cfg.rows) :
    4 * L.length ≤ cfg.cols * clearedRows cfg ∅ L + cfg.cols * cfg.rows := by
  have hbal := count_playFrom_add_cleared (Board.empty_wf cfg) hL
  have hcap : (playFrom cfg (∅ : Board) L).count ≤ cfg.cols * cfg.rows :=
    count_le_capacity (playFrom_wf (Board.empty_wf cfg) hL) hfield
  have hz : (∅ : Board).count = 0 := by simp [Board.count]
  omega

/-! ## The T-parity obstruction to density-1 renewal

A flat-to-flat renewal certificate at the mass floor — some number of bags
stacked into a perfect `10 × h` rectangle — would pin the clear-free growth
rate at exactly `2.8` rows per bag. The checkerboard charge shows such
renewals exist at even bag periods only. -/

/-- Checkerboard charge of a cell set, mod 2. -/
def charge (s : Finset Coord) : ZMod 2 :=
  ∑ p ∈ s, ((p.1 + p.2 : ℕ) : ZMod 2)

@[simp] theorem charge_empty : charge (∅ : Finset Coord) = 0 := rfl

/-- **T is the unique charged tetromino**: in every rotation, the drop
profile's checkerboard charge is `1` for T and `0` for the other six. -/
theorem charge_shapeUp :
    ∀ (p : Piece) (r : Rotation),
      charge (p.shapeUp r) = if p = Piece.T then 1 else 0 := by
  decide

/-- Translation invariance for tetromino drops: shifting a 4-cell shape
changes every cell's color by the same parity, and `4 ≡ 0 (mod 2)` kills the
total. The dropped piece carries exactly its profile's charge. -/
theorem charge_dropped (b : Board) (pl : Placement) :
    charge (pl.dropped b) = charge pl.shapeUp := by
  unfold charge Placement.dropped Placement.cellsAt
  rw [Finset.sum_image (fun x _ y _ hxy =>
    Placement.cellsAt_injective pl (pl.dropOffset b) hxy)]
  have hterm : ∀ cell : Coord,
      ((pl.col + cell.1 + (pl.dropOffset b + cell.2) : ℕ) : ZMod 2)
        = ((cell.1 + cell.2 : ℕ) : ZMod 2)
          + ((pl.col + pl.dropOffset b : ℕ) : ZMod 2) := by
    intro cell
    push_cast
    ring
  rw [Finset.sum_congr rfl (fun cell _ => hterm cell), Finset.sum_add_distrib,
    Finset.sum_const, Placement.shapeUp_card pl]
  have h40 : ((4 : ℕ) : ZMod 2) = 0 := by decide
  rw [nsmul_eq_mul, h40, zero_mul, add_zero]

/-- `Placement.shapeUp` form of the classification. -/
theorem charge_shapeUp_pl (pl : Placement) :
    charge pl.shapeUp = if pl.piece = Piece.T then 1 else 0 :=
  charge_shapeUp pl.piece pl.rot

/-- Placement adds the piece's charge (the dropped cells are disjoint from
the stack). -/
theorem charge_place (b : Board) (pl : Placement) :
    charge (pl.place b) = charge b + charge pl.shapeUp := by
  rw [← charge_dropped b pl]
  unfold charge
  rw [Placement.place_eq_union_dropped]
  exact Finset.sum_union (pl.dropped_disjoint b).symm

/-- The charge of a clear-free stack counts its T placements, mod 2. -/
theorem charge_stackFrom (b : Board) (L : List Placement) :
    charge (stackFrom b L)
      = charge b + ((L.countP fun pl => pl.piece == Piece.T : ℕ) : ZMod 2) := by
  induction L generalizing b with
  | nil => simp
  | cons pl rest ih =>
    rw [stackFrom_cons, ih, charge_place, charge_shapeUp_pl, List.countP_cons]
    by_cases hp : pl.piece = Piece.T
    · have hbeq : (pl.piece == Piece.T) = true := beq_iff_eq.mpr hp
      rw [if_pos hp, hbeq, if_pos rfl]
      push_cast
      ring
    · have hbeq : (pl.piece == Piece.T) = false := beq_eq_false_iff_ne.mpr hp
      rw [if_neg hp, hbeq, if_neg (by decide : ¬ (false = true))]
      push_cast
      ring

/-- One full row of the width-10 board carries charge 1 (45 + 10·r is odd). -/
theorem charge_row (r : ℕ) :
    charge ((Finset.range 10).image fun c => ((c, r) : Coord)) = 1 := by
  unfold charge
  rw [Finset.sum_image (fun x _ y _ hxy => by
    simpa using congrArg Prod.fst hxy)]
  have hterm : ∀ c : ℕ,
      ((c + r : ℕ) : ZMod 2) = ((c : ℕ) : ZMod 2) + ((r : ℕ) : ZMod 2) := by
    intro c; push_cast; ring
  rw [Finset.sum_congr rfl (fun c _ => hterm c), Finset.sum_add_distrib,
    Finset.sum_const, Finset.card_range]
  have h1 : ∑ c ∈ Finset.range 10, ((c : ℕ) : ZMod 2) = 1 := by decide
  have h2 : (10 : ℕ) • ((r : ℕ) : ZMod 2) = 0 := by
    have h100 : ((10 : ℕ) : ZMod 2) = 0 := by decide
    rw [nsmul_eq_mul, h100, zero_mul]
  rw [h1, h2, add_zero]

/-- The width-10 box of height `h` has charge `h (mod 2)`: each row carries
charge 1. -/
theorem charge_rect (h : ℕ) :
    charge (Finset.range 10 ×ˢ Finset.range h) = (h : ZMod 2) := by
  induction h with
  | zero => simp [charge]
  | succ k ih =>
    have hsplit : Finset.range 10 ×ˢ Finset.range (k + 1)
        = (Finset.range 10 ×ˢ Finset.range k)
          ∪ (Finset.range 10).image (fun c => ((c, k) : Coord)) := by
      ext p
      simp only [Finset.mem_union, Finset.mem_product, Finset.mem_range,
        Finset.mem_image]
      constructor
      · rintro ⟨hc, hr⟩
        rcases Nat.lt_succ_iff_lt_or_eq.mp hr with hlt | heq
        · exact Or.inl ⟨hc, hlt⟩
        · exact Or.inr ⟨p.1, hc, Prod.ext rfl heq.symm⟩
      · rintro (⟨hc, hr⟩ | ⟨c, hc, rfl⟩)
        · exact ⟨hc, Nat.lt_succ_of_lt hr⟩
        · exact ⟨hc, Nat.lt_succ_self k⟩
    have hdisj : Disjoint (Finset.range 10 ×ˢ Finset.range k)
        ((Finset.range 10).image (fun c => ((c, k) : Coord))) := by
      rw [Finset.disjoint_left]
      rintro p hp hq
      simp only [Finset.mem_product, Finset.mem_range] at hp
      rw [Finset.mem_image] at hq
      obtain ⟨c, -, rfl⟩ := hq
      exact absurd hp.2 (lt_irrefl k)
    rw [hsplit, charge, Finset.sum_union hdisj]
    have hrow := charge_row k
    unfold charge at ih hrow
    rw [ih, hrow]
    push_cast
    ring

/-! ## The renewal theorems -/

/-- **T-parity conservation for perfect rectangles**: a clear-free stack
that fills the width-10 height-`h` box exactly must have used a number of
T placements congruent to `h` mod 2. No validity or reachability hypotheses:
the charge ledger alone forces it. -/
theorem perfect_rectangle_T_parity {L : List Placement} {h : ℕ}
    (hfill : stack L = Finset.range 10 ×ˢ Finset.range h) :
    ((L.countP fun pl => pl.piece == Piece.T : ℕ) : ZMod 2) = (h : ZMod 2) := by
  have hs := charge_stackFrom ∅ L
  rw [stack] at hfill
  rw [hfill, charge_rect, charge_empty, zero_add] at hs
  exact hs.symm

/-- **No five bags ever stack into a perfect `10 × 14` rectangle.** Five
bags carry exactly five T pieces — odd — while the box demands an even
count. This upgrades the measured flush perfect-clear wall (the universal
10-cells-short stall) to a theorem: density-1 renewal is impossible at bag
period 5, whatever the placements, wherever they drop. -/
theorem no_five_bag_perfect_rectangle {L : List Placement}
    (hT : (L.countP fun pl => pl.piece == Piece.T) = 5) :
    stack L ≠ Finset.range 10 ×ˢ Finset.range 14 := by
  intro hfill
  have hp := perfect_rectangle_T_parity hfill
  rw [hT] at hp
  exact absurd hp (by decide)

/-- **Density-1 renewal has even bag period.** A perfect rectangle stacked
by `k` full bags (`35k` placements, `5k` of them T) forces `k` even: mass
fixes the box height at `14k`, and T-parity demands `5k ≡ 14k (mod 2)`.
The shortest flat-to-flat certificate compatible with the 7-bag is therefore
10 bags — 28 rows — the lattice point the tessellation and rate analyses
independently identified. -/
theorem perfect_rectangle_bag_period_even {L : List Placement} {k h : ℕ}
    (hlen : L.length = 35 * k)
    (hT : (L.countP fun pl => pl.piece == Piece.T) = 5 * k)
    (hfill : stack L = Finset.range 10 ×ˢ Finset.range h) :
    k % 2 = 0 := by
  have hmass : (stack L).count = 4 * L.length := by
    rw [stack, count_stackFrom]
    simp [Board.count]
  rw [hfill] at hmass
  have hcard : (Finset.range 10 ×ˢ Finset.range h).card = 10 * h := by
    rw [Finset.card_product, Finset.card_range, Finset.card_range]
  rw [Board.count] at hmass
  rw [hcard] at hmass
  rw [hlen] at hmass
  have hh : h = 14 * k := by omega
  have hp := perfect_rectangle_T_parity hfill
  rw [hT, hh] at hp
  have hk : ((k : ℕ) : ZMod 2) = 0 := by
    push_cast at hp
    have h5 : (5 : ZMod 2) = 1 := by decide
    have h14 : (14 : ZMod 2) = 0 := by decide
    rw [h5, h14, one_mul, zero_mul] at hp
    exact hp
  have hdvd : (2 : ℕ) ∣ k := (CharP.cast_eq_zero_iff (ZMod 2) 2 k).mp hk
  omega

end BagGrowth
end Tetris
