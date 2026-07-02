import Proofs.Model.Game

/-!
# Skylines and flush placements — the hole-free surface core (green)

The **skyline** of a height profile is the canonical hole-free board. This file
promotes the skyline core from `Proofs/Experiments/SurfaceInvariant.lean` into
the green library (that file still carries its own private copies until it is
deduplicated), and proves the **general flush-placement closure**: a placement
whose drop profile seats flush on a skyline yields exactly the skyline of the
updated profile — one lemma covering every piece orientation — together with
the full-move (`applyStep`) form via `clearLines_skyline` (continuous clearing
subtracts the profile minimum).

These are the transition workhorses for any hole-free carrier automaton feeding
`tetrisSolvableValid_of_strategy_bagInvariant`-style reductions: each concrete
transition instantiates `place_flush_skyline` with the orientation's column
data and closes by arithmetic. Vertical S and Z instances (the step-self-similar
bricks missing from the per-piece library) are included.
-/

namespace Tetris
namespace Board

/-- The **skyline** board of a column-height profile `h`: across the first
`cfg.cols` columns, column `j` is filled solidly in rows `0 … h j − 1`. This is
the canonical hole-free board of a given surface profile. -/
def skyline (cfg : GameConfig) (h : ℕ → ℕ) : Board :=
  (Finset.range cfg.cols).biUnion
    (fun j => (Finset.range (h j)).image (fun r => (j, r)))

/-- Membership in a skyline: cell `(j, r)` is filled exactly when `j` is a real
column and `r` sits below the profile height there. -/
theorem mem_skyline (cfg : GameConfig) (h : ℕ → ℕ) (j r : ℕ) :
    (j, r) ∈ skyline cfg h ↔ j < cfg.cols ∧ r < h j := by
  unfold skyline
  simp only [Finset.mem_biUnion, Finset.mem_range, Finset.mem_image, Prod.mk.injEq]
  constructor
  · rintro ⟨i, hi, ρ, hρ, hij, hρr⟩
    subst hij; subst hρr; exact ⟨hi, hρ⟩
  · rintro ⟨hj, hr⟩
    exact ⟨j, hj, r, hr, rfl, rfl⟩

/-- Coordinate-form membership, convenient when the cell is not a literal pair. -/
theorem mem_skyline' (cfg : GameConfig) (h : ℕ → ℕ) (p : Coord) :
    p ∈ skyline cfg h ↔ p.1 < cfg.cols ∧ p.2 < h p.1 := by
  obtain ⟨j, r⟩ := p
  exact mem_skyline cfg h j r

/-- A real column of a skyline has exactly the profile's rows filled. -/
theorem colRows_skyline {cfg : GameConfig} {h : ℕ → ℕ} {j : ℕ} (hj : j < cfg.cols) :
    Board.colRows (skyline cfg h) j = Finset.range (h j) := by
  ext r
  unfold Board.colRows
  simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_range]
  constructor
  · rintro ⟨p, ⟨hpmem, hpj⟩, hpr⟩
    have hmem2 := ((mem_skyline' cfg h p).mp hpmem).2
    rw [hpj, hpr] at hmem2
    exact hmem2
  · intro hr
    exact ⟨(j, r), ⟨(mem_skyline cfg h j r).mpr ⟨hj, hr⟩, rfl⟩, rfl⟩

/-- An out-of-range column of a skyline is empty. -/
theorem colRows_skyline_eq_empty {cfg : GameConfig} {h : ℕ → ℕ} {j : ℕ}
    (hj : cfg.cols ≤ j) : Board.colRows (skyline cfg h) j = ∅ := by
  ext r
  unfold Board.colRows
  simp only [Finset.mem_image, Finset.mem_filter, Finset.notMem_empty, iff_false, not_exists,
    not_and]
  rintro p ⟨hpmem, hpj⟩
  have hlt := ((mem_skyline' cfg h p).mp hpmem).1
  rw [hpj] at hlt
  omega

/-- The supremum identity behind a column's height: `range n` summed via `(·+1)`
recovers `n`. -/
private theorem sup_range_succ (n : ℕ) : (Finset.range n).sup (· + 1) = n := by
  cases n with
  | zero => simp
  | succ m =>
    refine le_antisymm (Finset.sup_le ?_) ?_
    · intro x hx
      rw [Finset.mem_range] at hx
      omega
    · exact Finset.le_sup (f := fun x => x + 1)
        (Finset.mem_range.mpr (Nat.lt_succ_self m))

/-- A skyline's column height recovers the profile exactly on real columns. -/
theorem colHeight_skyline {cfg : GameConfig} {h : ℕ → ℕ} {j : ℕ} (hj : j < cfg.cols) :
    Board.colHeight (skyline cfg h) j = h j := by
  unfold Board.colHeight
  rw [colRows_skyline hj]
  exact sup_range_succ (h j)

/-- A skyline's out-of-range columns have height `0`. -/
theorem colHeight_skyline_eq_zero {cfg : GameConfig} {h : ℕ → ℕ} {j : ℕ}
    (hj : cfg.cols ≤ j) : Board.colHeight (skyline cfg h) j = 0 := by
  unfold Board.colHeight
  rw [colRows_skyline_eq_empty hj]
  simp

theorem isFull_skyline {cfg : GameConfig} {h : ℕ → ℕ} {r : ℕ} :
    Board.isFull cfg (skyline cfg h) r ↔ ∀ j < cfg.cols, r < h j := by
  unfold Board.isFull
  constructor
  · intro hf j hj
    exact ((mem_skyline cfg h j r).mp (hf j (Finset.mem_range.mpr hj))).2
  · intro hf c hc
    exact (mem_skyline cfg h c r).mpr ⟨Finset.mem_range.mp hc, hf c (Finset.mem_range.mp hc)⟩

/-- With `m` the real-column minimum profile height, a skyline's full rows are
exactly rows `0 … m − 1`. -/
theorem isFull_skyline_iff_lt {cfg : GameConfig} {h : ℕ → ℕ} {m : ℕ}
    (hm : ∀ j < cfg.cols, m ≤ h j) (hm0 : ∃ j < cfg.cols, h j = m) (r : ℕ) :
    Board.isFull cfg (skyline cfg h) r ↔ r < m := by
  rw [isFull_skyline]
  constructor
  · intro hf; obtain ⟨j, hj, hjm⟩ := hm0; have := hf j hj; omega
  · intro hr j hj; have := hm j hj; omega

/-- The set of full rows of a skyline is `range m` (the contiguous bottom block). -/
theorem fullRows_skyline {cfg : GameConfig} {h : ℕ → ℕ} {m : ℕ}
    (hm : ∀ j < cfg.cols, m ≤ h j) (hm0 : ∃ j < cfg.cols, h j = m) :
    Board.fullRows cfg (skyline cfg h) = Finset.range m := by
  ext r
  unfold Board.fullRows
  rw [Finset.mem_filter, isFull_skyline_iff_lt hm hm0, Finset.mem_range]
  constructor
  · rintro ⟨_, hr⟩; exact hr
  · intro hr
    refine ⟨?_, hr⟩
    obtain ⟨j, hj, hjm⟩ := hm0
    exact Finset.mem_image.mpr ⟨(j, r), (mem_skyline cfg h j r).mpr ⟨hj, by omega⟩, rfl⟩

/-- Below row `r`, a skyline has `min m r` full rows cleared. -/
theorem clearedBelow_skyline {cfg : GameConfig} {h : ℕ → ℕ} {m : ℕ}
    (hm : ∀ j < cfg.cols, m ≤ h j) (hm0 : ∃ j < cfg.cols, h j = m) (r : ℕ) :
    Board.clearedBelow cfg (skyline cfg h) r = min m r := by
  unfold Board.clearedBelow
  rw [fullRows_skyline hm hm0]
  have : (Finset.range m).filter (· < r) = Finset.range (min m r) := by
    ext x; simp only [Finset.mem_filter, Finset.mem_range, lt_min_iff]
  rw [this, Finset.card_range]

/-- **Gravity on a skyline.** Clearing lines subtracts the minimum real-column
profile height `m` from every column: `clearLines (skyline h) = skyline (h − m)`.
This is the surface-invariant height-bounding step. -/
theorem clearLines_skyline {cfg : GameConfig} {h : ℕ → ℕ} {m : ℕ}
    (hm : ∀ j < cfg.cols, m ≤ h j) (hm0 : ∃ j < cfg.cols, h j = m) :
    Board.clearLines cfg (skyline cfg h) = skyline cfg (fun j => h j - m) := by
  unfold Board.clearLines
  ext ⟨j, r⟩
  rw [Finset.mem_image, mem_skyline]
  constructor
  · rintro ⟨⟨a, b⟩, hmem, heq⟩
    rw [Finset.mem_filter] at hmem
    obtain ⟨hab, hnf⟩ := hmem
    rw [mem_skyline] at hab
    rw [isFull_skyline_iff_lt hm hm0, not_lt] at hnf
    rw [clearedBelow_skyline hm hm0] at heq
    simp only [Prod.mk.injEq] at heq
    obtain ⟨ha, hb⟩ := heq
    rw [ha] at hab
    have hmj := hm j hab.1
    exact ⟨hab.1, by omega⟩
  · rintro ⟨hj, hr⟩
    have hmj := hm j hj
    refine ⟨(j, r + m), ?_, ?_⟩
    · rw [Finset.mem_filter, mem_skyline, isFull_skyline_iff_lt hm hm0]
      exact ⟨⟨hj, by omega⟩, by omega⟩
    · rw [clearedBelow_skyline hm hm0]
      simp only [Prod.mk.injEq, true_and]
      omega



/-- **Flush hard-drop offset.** If every drop-profile column has lower bound
`bot` (attained in column `0`) and the skyline is flush — column `col + i`
has height exactly `off + bot i` — then the hard-drop offset is exactly `off`. -/
theorem dropOffset_flush_skyline {cfg : GameConfig} {h : ℕ → ℕ} {pl : Placement}
    {w off : ℕ} {bot : ℕ → ℕ}
    (hwidth : ∀ cell ∈ pl.shapeUp, cell.1 < w)
    (hbot : ∀ cell ∈ pl.shapeUp, bot cell.1 ≤ cell.2)
    (hbot0 : ((0 : ℕ), bot 0) ∈ pl.shapeUp)
    (hcols : ∀ i < w, pl.col + i < cfg.cols)
    (hflush : ∀ i < w, h (pl.col + i) = off + bot i) :
    pl.dropOffset (skyline cfg h) = off := by
  rw [Placement.dropOffset_eq_sup]
  apply le_antisymm
  · refine Finset.sup_le (fun cell hcell => ?_)
    have hiw : cell.1 < w := hwidth cell hcell
    rw [colHeight_skyline (hcols cell.1 hiw), hflush cell.1 hiw]
    have := hbot cell hcell
    omega
  · have h0w : (0 : ℕ) < w := hwidth _ hbot0
    have hle := Finset.le_sup (f := fun cell : ℕ × ℕ =>
      (skyline cfg h).colHeight (pl.col + cell.1) - cell.2) hbot0
    have hred : (skyline cfg h).colHeight (pl.col + 0) - bot 0
        ≤ pl.shapeUp.sup
            (fun cell => (skyline cfg h).colHeight (pl.col + cell.1) - cell.2) := hle
    rw [colHeight_skyline (hcols 0 h0w), hflush 0 h0w] at hred
    omega

/-- **The general flush-placement closure.** If the drop profile of `pl` is
exactly "columns `0 … w−1`, column `i` filled contiguously from `bot i` to
`top i`" (`hshape`), the skyline is flush for it (`hflush`), and `h'` is the
profile with each landing column raised to `off + top i + 1` and all others
unchanged, then placing on the skyline yields exactly `skyline cfg h'`.
Hole-freeness is preserved with no per-piece reasoning. -/
theorem place_flush_skyline {cfg : GameConfig} {h h' : ℕ → ℕ} {pl : Placement}
    {w off : ℕ} {bot top : ℕ → ℕ}
    (hshape : ∀ i ρ, ((i, ρ) ∈ pl.shapeUp ↔ i < w ∧ bot i ≤ ρ ∧ ρ ≤ top i))
    (hw0 : 0 < w)
    (hbt : ∀ i < w, bot i ≤ top i)
    (hcols : ∀ i < w, pl.col + i < cfg.cols)
    (hflush : ∀ i < w, h (pl.col + i) = off + bot i)
    (hupd : ∀ i < w, h' (pl.col + i) = off + top i + 1)
    (hkeep : ∀ j, (∀ i < w, j ≠ pl.col + i) → h' j = h j) :
    pl.place (skyline cfg h) = skyline cfg h' := by
  have hoff : pl.dropOffset (skyline cfg h) = off := by
    refine dropOffset_flush_skyline (w := w) (bot := bot)
      (fun cell hcell => ?_) (fun cell hcell => ?_) ?_ hcols hflush
    · obtain ⟨i, ρ⟩ := cell
      exact ((hshape i ρ).mp hcell).1
    · obtain ⟨i, ρ⟩ := cell
      exact ((hshape i ρ).mp hcell).2.1
    · exact (hshape 0 (bot 0)).mpr ⟨hw0, le_refl _, hbt 0 hw0⟩
  ext ⟨j, r⟩
  rw [Placement.place_eq_union_dropped, Finset.mem_union, Placement.dropped_eq_image,
      hoff, mem_skyline, mem_skyline]
  simp only [Finset.mem_image, Prod.mk.injEq]
  by_cases hj : ∃ i, i < w ∧ j = pl.col + i
  · obtain ⟨i, hiw, rfl⟩ := hj
    rw [hupd i hiw]
    constructor
    · rintro (⟨hjc, hr⟩ | ⟨⟨i', ρ⟩, hcell, hji, hρr⟩)
      · rw [hflush i hiw] at hr
        have := hbt i hiw
        exact ⟨hcols i hiw, by omega⟩
      · dsimp only at hji hρr
        have hii : i' = i := by omega
        subst hii
        obtain ⟨-, hbρ, hρt⟩ := (hshape i' ρ).mp hcell
        exact ⟨hcols i' hiw, by omega⟩
    · rintro ⟨hjc, hr⟩
      by_cases hlow : r < h (pl.col + i)
      · exact Or.inl ⟨hjc, hlow⟩
      · rw [hflush i hiw] at hlow
        refine Or.inr ⟨(i, r - off), (hshape i (r - off)).mpr ⟨hiw, ?_, ?_⟩, ?_, ?_⟩
        · show bot i ≤ r - off
          omega
        · show r - off ≤ top i
          omega
        · rfl
        · show off + (r - off) = r
          omega
  · push Not at hj
    have hkeepj : h' j = h j := hkeep j (fun i hiw => hj i hiw)
    rw [hkeepj]
    constructor
    · rintro (hl | ⟨⟨i', ρ⟩, hcell, hji, hρr⟩)
      · exact hl
      · dsimp only at hji
        have hiw : i' < w := ((hshape i' ρ).mp hcell).1
        exact absurd hji.symm (hj i' hiw)
    · exact fun hl => Or.inl hl

/-- **Flush placement, full move, with clears.** Composing
`place_flush_skyline`-style equalities with `clearLines_skyline`: if placing
gives `skyline cfg h'` and `m` is the (attained) minimum of `h'` over real
columns, the full move drops the board to the min-normalized skyline. With
`m = 0` this is the no-clear case; with `m > 0` exactly `m` rows clear. -/
theorem applyStep_flush_skyline {cfg : GameConfig} {h h' : ℕ → ℕ} {pl : Placement}
    {m : ℕ}
    (hplace : pl.place (skyline cfg h) = skyline cfg h')
    (hm : ∀ j < cfg.cols, m ≤ h' j) (hm0 : ∃ j < cfg.cols, h' j = m) :
    pl.applyStep cfg (skyline cfg h) = skyline cfg (fun j => h' j - m) := by
  rw [Placement.applyStep_eq_clearLines_place, hplace, clearLines_skyline hm hm0]

/-! ## Validation instances: the vertical S and Z bricks

Vertical S (rot 1) seats flush on a left-high step `h c = h (c+1) + 1` and
reproduces the same step two rows higher; vertical Z (rot 1) is its mirror on a
right-high step. These are the step-self-similar placements a rough-surface
carrier uses for the S/Z pieces, and the first orientation instances consumed
from the general lemma. -/

/-- The vertical S drop profile: column 0 rows 1–2, column 1 rows 0–1. -/
theorem shapeUp_vertS (c : ℕ) :
    ({ piece := Piece.S, rot := 1, col := c } : Placement).shapeUp
      = {((0 : ℕ), (1 : ℕ)), (0, 2), (1, 0), (1, 1)} := by
  show Piece.shapeUp Piece.S 1 = _
  decide

/-- The vertical Z drop profile: column 0 rows 0–1, column 1 rows 1–2. -/
theorem shapeUp_vertZ (c : ℕ) :
    ({ piece := Piece.Z, rot := 1, col := c } : Placement).shapeUp
      = {((0 : ℕ), (0 : ℕ)), (0, 1), (1, 1), (1, 2)} := by
  show Piece.shapeUp Piece.Z 1 = _
  decide

/-- **Vertical-S closure on a skyline.** On a left-high step
(`h c = h (c+1) + 1`), the vertical S piece seats flush and raises both columns
by two — the step shape is *reproduced* two rows higher. -/
theorem place_vertS_skyline {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc : c < cfg.cols) (hc1 : c + 1 < cfg.cols) (hstep : h c = h (c + 1) + 1) :
    Placement.place (skyline cfg h) { piece := Piece.S, rot := 1, col := c }
      = skyline cfg
          (Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2)) := by
  refine place_flush_skyline (w := 2) (off := h (c + 1))
    (bot := fun i => 1 - i) (top := fun i => 2 - i)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_vertS]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · show ({ piece := Piece.S, rot := 1, col := c } : Placement).col + i < cfg.cols
    interval_cases i <;> simpa
  · show h (({ piece := Piece.S, rot := 1, col := c } : Placement).col + i) = h (c + 1) + (1 - i)
    interval_cases i
    · simpa using hstep
    · simp
  · show Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2)
        (({ piece := Piece.S, rot := 1, col := c } : Placement).col + i)
        = h (c + 1) + (2 - i) + 1
    interval_cases i
    · show Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2) c
          = h (c + 1) + 2 + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]
      omega
    · rw [Function.update_self]
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    show Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2) j = h j
    rw [Function.update_of_ne hj1, Function.update_of_ne hj0]

/-- **Vertical-Z closure on a skyline.** Mirror of `place_vertS_skyline`: on a
right-high step (`h (c+1) = h c + 1`), the vertical Z piece seats flush and
raises both columns by two, reproducing the step. -/
theorem place_vertZ_skyline {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc : c < cfg.cols) (hc1 : c + 1 < cfg.cols) (hstep : h (c + 1) = h c + 1) :
    Placement.place (skyline cfg h) { piece := Piece.Z, rot := 1, col := c }
      = skyline cfg
          (Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2)) := by
  refine place_flush_skyline (w := 2) (off := h c)
    (bot := fun i => i) (top := fun i => i + 1)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_vertZ]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · show ({ piece := Piece.Z, rot := 1, col := c } : Placement).col + i < cfg.cols
    interval_cases i <;> simpa
  · show h (({ piece := Piece.Z, rot := 1, col := c } : Placement).col + i) = h c + i
    interval_cases i
    · simp
    · simpa using hstep
  · show Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2)
        (({ piece := Piece.Z, rot := 1, col := c } : Placement).col + i)
        = h c + (i + 1) + 1
    interval_cases i
    · show Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2) c
          = h c + 2
      rw [Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]
    · rw [Function.update_self]
      omega
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    show Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2) j = h j
    rw [Function.update_of_ne hj1, Function.update_of_ne hj0]

end Board
end Tetris
