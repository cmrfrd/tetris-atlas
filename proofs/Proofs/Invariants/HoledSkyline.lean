import Proofs.Invariants.Skyline
import Proofs.Invariants.SurfaceFiber

/-!
# Holed skylines — the debt-1 board shape

Exact search (`Search/CarrierSearch.lean`) shows hole-free play cannot survive
the 7-bag: every flush-only carrier family is empty (per-piece through spread 5,
bag-aware through spread 4). The surviving design carries **bounded hole debt**:
boards of the form `skyline h` minus one *strictly covered* cell. This file
sets up that board shape and its placement theory:

* `holedSkyline cfg h x` — the skyline with the single cell `x` removed;
* the surface identity: a strictly covered hole is invisible to `colHeight`
  (`colHeight_holedSkyline`), hence to `dropOffset`/`dropped`
  (via `SurfaceFiber.dropOffset_eq_of_colHeight_eq`);
* placement transport: placing on a holed skyline is placing on the skyline
  and re-removing the hole (`place_holedSkyline`) — dropped cells always land
  at or above the surface, never on the buried cell.

The remaining debt-1 obligation (next step, not here) is the clearing law:
`clearLines` on a holed skyline clears exactly the full rows *below the
minimum, excluding the hole row*, shifting the hole with the stack and
exposing it when it reaches its column top — the `settle` semantics of the
search harness.
-/

namespace Tetris
namespace Board

/-- The skyline of `h` with the single cell `x` removed — the canonical
debt-1 board. Meaningful when `x` is in-field and strictly covered
(`x.2 + 1 < h x.1`); the lemmas below carry those hypotheses. -/
def holedSkyline (cfg : GameConfig) (h : ℕ → ℕ) (x : Coord) : Board :=
  (skyline cfg h).erase x

/-- Membership in a holed skyline. -/
theorem mem_holedSkyline {cfg : GameConfig} {h : ℕ → ℕ} {x : Coord} (p : Coord) :
    p ∈ holedSkyline cfg h x ↔ p ≠ x ∧ p.1 < cfg.cols ∧ p.2 < h p.1 := by
  unfold holedSkyline
  rw [Finset.mem_erase, mem_skyline']

/-- Helper: the sup `(· + 1)` of `range n` minus one element `k` with
`k + 1 < n` is still `n` — removing a non-top element does not lower the top. -/
private theorem sup_range_erase (n k : ℕ) (hk : k + 1 < n) :
    ((Finset.range n).erase k).sup (· + 1) = n := by
  refine le_antisymm (Finset.sup_le fun r hr => ?_) ?_
  · have := Finset.mem_range.mp (Finset.mem_of_mem_erase hr)
    omega
  · have hmem : n - 1 ∈ (Finset.range n).erase k := by
      rw [Finset.mem_erase, Finset.mem_range]
      omega
    have hle := Finset.le_sup (f := fun r => r + 1) hmem
    dsimp only at hle
    omega

/-- **A strictly covered hole is invisible to the surface.** Every column of a
holed skyline has the same height as the intact skyline — in particular the
hole's own column, whose top cell `(x.1, h x.1 − 1)` survives the erasure. -/
theorem colHeight_holedSkyline {cfg : GameConfig} {h : ℕ → ℕ} {x : Coord}
    (hcov : x.2 + 1 < h x.1) (j : ℕ) :
    (holedSkyline cfg h x).colHeight j = (skyline cfg h).colHeight j := by
  by_cases hj : j < cfg.cols
  · unfold Board.colHeight
    rw [colRows_skyline hj]
    by_cases hjx : j = x.1
    · have hrows : (holedSkyline cfg h x).colRows j = (Finset.range (h j)).erase x.2 := by
        ext r
        unfold Board.colRows
        simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_erase, Finset.mem_range]
        constructor
        · rintro ⟨p, ⟨hpmem, hpj⟩, hpr⟩
          obtain ⟨hpx, hplt, hprh⟩ := (mem_holedSkyline p).mp hpmem
          subst hpj; subst hpr
          refine ⟨fun hrx => hpx ?_, hprh⟩
          obtain ⟨a, b⟩ := p
          obtain ⟨c, d⟩ := x
          simp_all
        · rintro ⟨hrx, hrh⟩
          refine ⟨(j, r), ⟨(mem_holedSkyline (j, r)).mpr ⟨?_, hj, hrh⟩, rfl⟩, rfl⟩
          intro hcontra
          exact hrx (congrArg Prod.snd hcontra)
      rw [hrows, sup_range_succ, hjx]
      exact sup_range_erase (h x.1) x.2 hcov
    · have hrows : (holedSkyline cfg h x).colRows j = Finset.range (h j) := by
        ext r
        unfold Board.colRows
        simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_range]
        constructor
        · rintro ⟨p, ⟨hpmem, hpj⟩, hpr⟩
          obtain ⟨_, _, hprh⟩ := (mem_holedSkyline p).mp hpmem
          subst hpj; subst hpr; exact hprh
        · intro hr
          refine ⟨(j, r), ⟨(mem_holedSkyline (j, r)).mpr ⟨?_, hj, hr⟩, rfl⟩, rfl⟩
          intro hcontra
          exact hjx (congrArg Prod.fst hcontra)
      rw [hrows, sup_range_succ]
  · unfold Board.colHeight
    have h1 : (holedSkyline cfg h x).colRows j = ∅ := by
      ext r
      unfold Board.colRows
      simp only [Finset.mem_image, Finset.mem_filter, Finset.notMem_empty, iff_false,
        not_exists, not_and]
      rintro p ⟨hpmem, hpj⟩
      obtain ⟨_, hplt, _⟩ := (mem_holedSkyline p).mp hpmem
      omega
    rw [h1, colRows_skyline_eq_empty (by omega)]

/-- **Dropped cells never touch a buried cell.** On a skyline, every cell of a
hard-dropped piece lands at or above its column's surface, so a cell strictly
below the surface is never among them. -/
theorem notMem_dropped_skyline_of_covered {cfg : GameConfig} {h : ℕ → ℕ}
    {x : Coord} (pl : Placement) (hxcols : x.1 < cfg.cols) (hcov : x.2 < h x.1) :
    x ∉ pl.dropped (skyline cfg h) := by
  intro hmem
  rw [Placement.dropped_eq_image] at hmem
  obtain ⟨⟨i, ρ⟩, hcell, heq⟩ := Finset.mem_image.mp hmem
  have hx1 : pl.col + i = x.1 := congrArg Prod.fst heq
  have hx2 : pl.dropOffset (skyline cfg h) + ρ = x.2 := congrArg Prod.snd heq
  have hle := Finset.le_sup (f := fun cell : ℕ × ℕ =>
    (skyline cfg h).colHeight (pl.col + cell.1) - cell.2) hcell
  dsimp only at hle
  rw [hx1, colHeight_skyline hxcols, ← Placement.dropOffset_eq_sup] at hle
  omega

/-- **Placement transport onto a holed skyline.** Placing on a holed skyline is
placing on the intact skyline and re-removing the hole: the drop is
surface-determined (`dropOffset_eq_of_colHeight_eq`) and never lands on the
buried cell (`notMem_dropped_skyline_of_covered`). Together with
`place_flush_skyline` this reduces every debt-1 placement transition to the
hole-free one. -/
theorem place_holedSkyline {cfg : GameConfig} {h : ℕ → ℕ} {x : Coord}
    (pl : Placement) (hxcols : x.1 < cfg.cols) (hcov : x.2 + 1 < h x.1) :
    pl.place (holedSkyline cfg h x) = (pl.place (skyline cfg h)).erase x := by
  have hheights : ∀ j, (holedSkyline cfg h x).colHeight j
      = (skyline cfg h).colHeight j := colHeight_holedSkyline hcov
  rw [Placement.place_eq_union_dropped, Placement.place_eq_union_dropped,
      SurfaceFiber.dropped_eq_of_colHeight_eq pl hheights]
  unfold holedSkyline
  ext p
  simp only [Finset.mem_union, Finset.mem_erase]
  constructor
  · rintro (⟨hpx, hpsky⟩ | hpdrop)
    · exact ⟨hpx, Or.inl hpsky⟩
    · refine ⟨?_, Or.inr hpdrop⟩
      rintro rfl
      exact notMem_dropped_skyline_of_covered pl hxcols (by omega) hpdrop
  · rintro ⟨hpx, hpsky | hpdrop⟩
    · exact Or.inl ⟨hpx, hpsky⟩
    · exact Or.inr hpdrop


/-! ## Toward the debt-1 clearing law: full rows of a holed skyline

The hole blocks exactly its own row. With `m` the real-column minimum of the
profile, the full rows of `holedSkyline cfg h x` are `range m` minus `x.2` —
the contiguous bottom block of the intact skyline with the hole row knocked
out. The full clearing law (`clearLines` = clear those rows, shift, and expose
the hole when it reaches its column top) builds on this characterization. -/

/-- A holed skyline's row `r` is full exactly when every real column rises
above `r` and `r` is not the hole row. -/
theorem isFull_holedSkyline {cfg : GameConfig} {h : ℕ → ℕ} {x : Coord}
    (hxcols : x.1 < cfg.cols) {r : ℕ} :
    Board.isFull cfg (holedSkyline cfg h x) r ↔
      (∀ j < cfg.cols, r < h j) ∧ r ≠ x.2 := by
  unfold Board.isFull
  constructor
  · intro hf
    have hx : (x.1, r) ∈ holedSkyline cfg h x :=
      hf x.1 (Finset.mem_range.mpr hxcols)
    obtain ⟨hne, -, -⟩ := (mem_holedSkyline (x.1, r)).mp hx
    refine ⟨fun j hj => ?_, ?_⟩
    · have := (mem_holedSkyline (j, r)).mp (hf j (Finset.mem_range.mpr hj))
      exact this.2.2
    · intro hrx
      exact hne (by rw [hrx])
  · rintro ⟨hall, hrx⟩ c hc
    have hc' := Finset.mem_range.mp hc
    refine (mem_holedSkyline (c, r)).mpr ⟨?_, hc', hall c hc'⟩
    intro hcontra
    exact hrx (congrArg Prod.snd hcontra)

/-- With `m` the real-column minimum profile height, a holed skyline's full
rows are the bottom block minus the hole row: `range m \ {x.2}`. -/
theorem fullRows_holedSkyline {cfg : GameConfig} {h : ℕ → ℕ} {x : Coord} {m : ℕ}
    (hxcols : x.1 < cfg.cols)
    (hm : ∀ j < cfg.cols, m ≤ h j) (hm0 : ∃ j < cfg.cols, h j = m) :
    Board.fullRows cfg (holedSkyline cfg h x) = (Finset.range m).erase x.2 := by
  ext r
  unfold Board.fullRows
  rw [Finset.mem_filter, isFull_holedSkyline hxcols, Finset.mem_erase, Finset.mem_range]
  constructor
  · rintro ⟨-, hall, hrx⟩
    obtain ⟨j, hj, hjm⟩ := hm0
    have := hall j hj
    exact ⟨hrx, by omega⟩
  · rintro ⟨hrx, hrm⟩
    refine ⟨?_, fun j hj => ?_, hrx⟩
    · refine Finset.mem_image.mpr ⟨(x.1, r), ?_, rfl⟩
      refine (mem_holedSkyline (x.1, r)).mpr ⟨?_, hxcols, ?_⟩
      · intro hcontra
        exact hrx (congrArg Prod.snd hcontra)
      · show r < h x.1
        have := hm x.1 hxcols
        omega
    · have := hm j hj
      omega


/-- **Shift count below a row.** Below row `r`, a holed skyline has cleared
exactly the full rows `< r`: all of `range (min m r)` except the hole row when
it lies in that range. This is the per-cell gravity shift the clearing law
applies. -/
theorem clearedBelow_holedSkyline {cfg : GameConfig} {h : ℕ → ℕ} {x : Coord} {m : ℕ}
    (hxcols : x.1 < cfg.cols)
    (hm : ∀ j < cfg.cols, m ≤ h j) (hm0 : ∃ j < cfg.cols, h j = m) (r : ℕ) :
    Board.clearedBelow cfg (holedSkyline cfg h x) r
      = min m r - (if x.2 < min m r then 1 else 0) := by
  unfold Board.clearedBelow
  rw [fullRows_holedSkyline hxcols hm hm0]
  have hset : ((Finset.range m).erase x.2).filter (· < r)
      = (Finset.range (min m r)).erase x.2 := by
    ext y
    simp only [Finset.mem_filter, Finset.mem_erase, Finset.mem_range, lt_min_iff]
    constructor
    · rintro ⟨⟨hyx, hym⟩, hyr⟩
      exact ⟨hyx, hym, hyr⟩
    · rintro ⟨hyx, hym, hyr⟩
      exact ⟨⟨hyx, hym⟩, hyr⟩
  rw [hset]
  by_cases hx : x.2 < min m r
  · rw [Finset.card_erase_of_mem (Finset.mem_range.mpr hx), Finset.card_range, if_pos hx]
  · rw [Finset.erase_eq_of_notMem (fun hc => hx (Finset.mem_range.mp hc)),
        Finset.card_range, if_neg hx]
    omega


/-- **Clearing law, case A: the hole sits at or above the clear zone.** When
`m ≤ x.2` (the hole row is not among the full bottom rows), the hole blocks
nothing: all `m` bottom rows clear, the stack shifts down by `m`, and the hole
rides along. The result is again a holed skyline, with profile `h − m` and
hole `(x.1, x.2 − m)`. -/
theorem clearLines_holedSkyline_of_le {cfg : GameConfig} {h : ℕ → ℕ} {x : Coord} {m : ℕ}
    (hxcols : x.1 < cfg.cols)
    (hm : ∀ j < cfg.cols, m ≤ h j) (hm0 : ∃ j < cfg.cols, h j = m)
    (hxm : m ≤ x.2) :
    Board.clearLines cfg (holedSkyline cfg h x)
      = holedSkyline cfg (fun j => h j - m) (x.1, x.2 - m) := by
  have hfull : ∀ y : ℕ, Board.isFull cfg (holedSkyline cfg h x) y ↔ y < m := by
    intro y
    rw [isFull_holedSkyline hxcols]
    constructor
    · rintro ⟨hall, -⟩
      obtain ⟨j, hj, hjm⟩ := hm0
      have := hall j hj
      omega
    · intro hy
      exact ⟨fun j hj => by have := hm j hj; omega, by omega⟩
  unfold Board.clearLines
  ext ⟨j, r⟩
  rw [Finset.mem_image, mem_holedSkyline]
  constructor
  · rintro ⟨⟨a, b⟩, hmem, heq⟩
    rw [Finset.mem_filter] at hmem
    obtain ⟨hab, hnf⟩ := hmem
    obtain ⟨habx, hacols, habh⟩ := (mem_holedSkyline (a, b)).mp hab
    dsimp only at hnf heq hacols habh
    rw [hfull b, not_lt] at hnf
    rw [clearedBelow_holedSkyline hxcols hm hm0] at heq
    have hshift : min m b - (if x.2 < min m b then 1 else 0) = m := by
      have : min m b = m := by omega
      rw [this, if_neg (by omega)]
      omega
    rw [hshift] at heq
    have ha : a = j := congrArg Prod.fst heq
    have hb : b - m = r := congrArg Prod.snd heq
    subst ha
    refine ⟨?_, hacols, by dsimp only; omega⟩
    intro hcontra
    have h1 : a = x.1 := congrArg Prod.fst hcontra
    have h2 : r = x.2 - m := congrArg Prod.snd hcontra
    apply habx
    have : b = x.2 := by omega
    rw [h1, this]
  · rintro ⟨hne, hj, hr⟩
    dsimp only at hj hr
    have hmj := hm j hj
    refine ⟨(j, r + m), ?_, ?_⟩
    · rw [Finset.mem_filter]
      refine ⟨(mem_holedSkyline (j, r + m)).mpr ⟨?_, hj, ?_⟩, ?_⟩
      · intro hcontra
        have h1 : j = x.1 := congrArg Prod.fst hcontra
        have h2 : r + m = x.2 := congrArg Prod.snd hcontra
        apply hne
        have : r = x.2 - m := by omega
        rw [h1, this]
      · show r + m < h j
        omega
      · show ¬ Board.isFull cfg (holedSkyline cfg h x) (r + m)
        rw [hfull]
        omega
    · show ((j : ℕ), r + m - Board.clearedBelow cfg (holedSkyline cfg h x) (r + m)) = (j, r)
      rw [clearedBelow_holedSkyline hxcols hm hm0]
      have hshift : min m (r + m) - (if x.2 < min m (r + m) then 1 else 0) = m := by
        have : min m (r + m) = m := by omega
        rw [this, if_neg (by omega)]
        omega
      rw [hshift]
      show ((j : ℕ), r + m - m) = (j, r)
      simp


/-- **Clearing law, case B: the hole sits strictly inside the clear zone and
its column rises strictly above the minimum.** Rows `0 … m−1` except the hole
row clear (`m − 1` rows). In every other column the cell at the hole row
survives and rides to row `0`; in the hole's column row `0` becomes the new
hole position. The result is again a holed skyline with profile
`h − (m − 1)` and hole `(x.1, 0)`. -/
theorem clearLines_holedSkyline_of_lt {cfg : GameConfig} {h : ℕ → ℕ} {x : Coord} {m : ℕ}
    (hxcols : x.1 < cfg.cols) (hcov : x.2 + 1 < h x.1)
    (hm : ∀ j < cfg.cols, m ≤ h j) (hm0 : ∃ j < cfg.cols, h j = m)
    (hxm : x.2 < m) (hgt : m < h x.1) :
    Board.clearLines cfg (holedSkyline cfg h x)
      = holedSkyline cfg (fun j => h j - (m - 1)) (x.1, 0) := by
  have hfull : ∀ y : ℕ, Board.isFull cfg (holedSkyline cfg h x) y ↔ y < m ∧ y ≠ x.2 := by
    intro y
    rw [isFull_holedSkyline hxcols]
    constructor
    · rintro ⟨hall, hyx⟩
      obtain ⟨j, hj, hjm⟩ := hm0
      have := hall j hj
      exact ⟨by omega, hyx⟩
    · rintro ⟨hy, hyx⟩
      exact ⟨fun j hj => by have := hm j hj; omega, hyx⟩
  unfold Board.clearLines
  ext ⟨j, r⟩
  rw [Finset.mem_image, mem_holedSkyline]
  constructor
  · rintro ⟨⟨a, b⟩, hmem, heq⟩
    rw [Finset.mem_filter] at hmem
    obtain ⟨hab, hnf⟩ := hmem
    obtain ⟨habx, hacols, habh⟩ := (mem_holedSkyline (a, b)).mp hab
    dsimp only at hnf heq hacols habh
    rw [hfull b] at hnf
    rw [clearedBelow_holedSkyline hxcols hm hm0] at heq
    have hcase : b = x.2 ∨ m ≤ b := by
      by_cases hbm : b < m
      · left
        by_contra hbx
        exact hnf ⟨hbm, hbx⟩
      · right; omega
    rcases hcase with rfl | hbm
    · -- the survivor at the hole row rides to row 0
      have hax : a ≠ x.1 := by
        intro hc
        exact habx (by rw [hc])
      have hshift : min m x.2 - (if x.2 < min m x.2 then 1 else 0) = x.2 := by
        have : min m x.2 = x.2 := by omega
        rw [this, if_neg (by omega)]
        omega
      rw [hshift] at heq
      have ha : a = j := congrArg Prod.fst heq
      have hb : x.2 - x.2 = r := congrArg Prod.snd heq
      subst ha
      refine ⟨?_, hacols, ?_⟩
      · intro hcontra
        exact hax (congrArg Prod.fst hcontra)
      · show r < h a - (m - 1)
        have := hm a hacols
        omega
    · -- cells at or above the zone shift down by m − 1
      have hshift : min m b - (if x.2 < min m b then 1 else 0) = m - 1 := by
        have : min m b = m := by omega
        rw [this, if_pos (by omega)]
      rw [hshift] at heq
      have ha : a = j := congrArg Prod.fst heq
      have hb : b - (m - 1) = r := congrArg Prod.snd heq
      subst ha
      refine ⟨?_, hacols, ?_⟩
      · intro hcontra
        have h2 : r = 0 := congrArg Prod.snd hcontra
        omega
      · show r < h a - (m - 1)
        omega
  · rintro ⟨hne, hj, hr⟩
    dsimp only at hj hr
    have hmj := hm j hj
    by_cases hr0 : r = 0
    · -- row 0 of a non-hole column comes from the hole-row survivor
      subst hr0
      have hjx : j ≠ x.1 := by
        intro hc
        exact hne (by rw [hc])
      refine ⟨(j, x.2), ?_, ?_⟩
      · rw [Finset.mem_filter]
        refine ⟨(mem_holedSkyline (j, x.2)).mpr ⟨?_, hj, ?_⟩, ?_⟩
        · intro hcontra
          exact hjx (congrArg Prod.fst hcontra)
        · show x.2 < h j
          omega
        · show ¬ Board.isFull cfg (holedSkyline cfg h x) (j, x.2).2
          rw [hfull]
          simp
      · show ((j : ℕ), x.2 - Board.clearedBelow cfg (holedSkyline cfg h x) (j, x.2).2) = (j, 0)
        rw [clearedBelow_holedSkyline hxcols hm hm0]
        have : min m x.2 = x.2 := by omega
        rw [this, if_neg (by omega)]
        simp
    · -- rows ≥ 1 come from cells at or above the zone
      refine ⟨(j, r + (m - 1)), ?_, ?_⟩
      · rw [Finset.mem_filter]
        refine ⟨(mem_holedSkyline (j, r + (m - 1))).mpr ⟨?_, hj, ?_⟩, ?_⟩
        · intro hcontra
          have h2 : r + (m - 1) = x.2 := congrArg Prod.snd hcontra
          omega
        · show r + (m - 1) < h j
          omega
        · show ¬ Board.isFull cfg (holedSkyline cfg h x) (j, r + (m - 1)).2
          rw [hfull]
          omega
      · show ((j : ℕ), r + (m - 1)
            - Board.clearedBelow cfg (holedSkyline cfg h x) (j, r + (m - 1)).2) = (j, r)
        rw [clearedBelow_holedSkyline hxcols hm hm0]
        have hmin : min m (r + (m - 1)) = m := by omega
        rw [hmin, if_pos (by omega)]
        show ((j : ℕ), r + (m - 1) - (m - 1)) = (j, r)
        simp


/-- **Clearing law, case B′ (exposure): the hole column attains the minimum.**
With the hole strictly inside the clear zone and `h x.1 = m`, the hole
column's only cells lie in full rows, so it empties entirely: the debt is
ERASED and the result is an intact skyline — profile `h − (m−1)` with the
hole column dropped to `0`. This is the recovery move of debt-1 play: burying
a cell is repaid by clearing down to it. -/
theorem clearLines_holedSkyline_exposed {cfg : GameConfig} {h : ℕ → ℕ} {x : Coord} {m : ℕ}
    (hxcols : x.1 < cfg.cols) (hcov : x.2 + 1 < h x.1)
    (hm : ∀ j < cfg.cols, m ≤ h j) (hm0 : ∃ j < cfg.cols, h j = m)
    (hxm : x.2 < m) (hexp : h x.1 = m) :
    Board.clearLines cfg (holedSkyline cfg h x)
      = skyline cfg (Function.update (fun j => h j - (m - 1)) x.1 0) := by
  have hfull : ∀ y : ℕ, Board.isFull cfg (holedSkyline cfg h x) y ↔ y < m ∧ y ≠ x.2 := by
    intro y
    rw [isFull_holedSkyline hxcols]
    constructor
    · rintro ⟨hall, hyx⟩
      obtain ⟨j, hj, hjm⟩ := hm0
      have := hall j hj
      exact ⟨by omega, hyx⟩
    · rintro ⟨hy, hyx⟩
      exact ⟨fun j hj => by have := hm j hj; omega, hyx⟩
  unfold Board.clearLines
  ext ⟨j, r⟩
  rw [Finset.mem_image, mem_skyline]
  constructor
  · rintro ⟨⟨a, b⟩, hmem, heq⟩
    rw [Finset.mem_filter] at hmem
    obtain ⟨hab, hnf⟩ := hmem
    obtain ⟨habx, hacols, habh⟩ := (mem_holedSkyline (a, b)).mp hab
    dsimp only at hnf heq hacols habh
    rw [hfull b] at hnf
    rw [clearedBelow_holedSkyline hxcols hm hm0] at heq
    have hcase : b = x.2 ∨ m ≤ b := by
      by_cases hbm : b < m
      · left
        by_contra hbx
        exact hnf ⟨hbm, hbx⟩
      · right; omega
    have hax : a ≠ x.1 := by
      intro hc
      rcases hcase with rfl | hbm
      · exact habx (by rw [hc])
      · rw [hc, hexp] at habh
        omega
    rcases hcase with rfl | hbm
    · have hshift : min m x.2 - (if x.2 < min m x.2 then 1 else 0) = x.2 := by
        have : min m x.2 = x.2 := by omega
        rw [this, if_neg (by omega)]
        omega
      rw [hshift] at heq
      have ha : a = j := congrArg Prod.fst heq
      have hb : x.2 - x.2 = r := congrArg Prod.snd heq
      subst ha
      refine ⟨hacols, ?_⟩
      rw [Function.update_of_ne hax]
      show r < h a - (m - 1)
      have := hm a hacols
      omega
    · have hshift : min m b - (if x.2 < min m b then 1 else 0) = m - 1 := by
        have : min m b = m := by omega
        rw [this, if_pos (by omega)]
      rw [hshift] at heq
      have ha : a = j := congrArg Prod.fst heq
      have hb : b - (m - 1) = r := congrArg Prod.snd heq
      subst ha
      refine ⟨hacols, ?_⟩
      rw [Function.update_of_ne hax]
      show r < h a - (m - 1)
      omega
  · rintro ⟨hj, hr⟩
    have hjx : j ≠ x.1 := by
      intro hc
      rw [hc, Function.update_self] at hr
      omega
    rw [Function.update_of_ne hjx] at hr
    have hr' : r < h j - (m - 1) := hr
    have hmj := hm j hj
    by_cases hr0 : r = 0
    · subst hr0
      refine ⟨(j, x.2), ?_, ?_⟩
      · rw [Finset.mem_filter]
        refine ⟨(mem_holedSkyline (j, x.2)).mpr ⟨?_, hj, ?_⟩, ?_⟩
        · intro hcontra
          exact hjx (congrArg Prod.fst hcontra)
        · show x.2 < h j
          omega
        · show ¬ Board.isFull cfg (holedSkyline cfg h x) (j, x.2).2
          rw [hfull]
          simp
      · show ((j : ℕ), x.2 - Board.clearedBelow cfg (holedSkyline cfg h x) (j, x.2).2) = (j, 0)
        rw [clearedBelow_holedSkyline hxcols hm hm0]
        have : min m x.2 = x.2 := by omega
        rw [this, if_neg (by omega)]
        simp
    · refine ⟨(j, r + (m - 1)), ?_, ?_⟩
      · rw [Finset.mem_filter]
        refine ⟨(mem_holedSkyline (j, r + (m - 1))).mpr ⟨?_, hj, ?_⟩, ?_⟩
        · intro hcontra
          exact hjx (congrArg Prod.fst hcontra)
        · show r + (m - 1) < h j
          omega
        · show ¬ Board.isFull cfg (holedSkyline cfg h x) (j, r + (m - 1)).2
          rw [hfull]
          omega
      · show ((j : ℕ), r + (m - 1)
            - Board.clearedBelow cfg (holedSkyline cfg h x) (j, r + (m - 1)).2) = (j, r)
        rw [clearedBelow_holedSkyline hxcols hm hm0]
        have hmin : min m (r + (m - 1)) = m := by omega
        rw [hmin, if_pos (by omega)]
        show ((j : ℕ), r + (m - 1) - (m - 1)) = (j, r)
        simp



/-- **Debt in the cell count.** A holed skyline carries exactly one cell less
than its surface area: `card = ∑ h − 1`. Combined with the drift identity
(`sum_profile_applyStep_flush`) this extends the conservation ledger to
debt-1 states — the buried cell is one unit of surface area the clearing
duty cannot cash until exposure repays it. -/
theorem card_holedSkyline {cfg : GameConfig} {h : ℕ → ℕ} {x : Coord}
    (hxcols : x.1 < cfg.cols) (hcov : x.2 < h x.1) :
    (holedSkyline cfg h x).card = (∑ j ∈ Finset.range cfg.cols, h j) - 1 := by
  unfold holedSkyline
  rw [Finset.card_erase_of_mem ((mem_skyline' cfg h x).mpr ⟨hxcols, hcov⟩), card_skyline]

/-! ## The debt-1 board realization

A debt-1 state is a profile plus an optional (strictly covered) hole;
`debtBoard` realizes it as a board. This is the state space of the debt-1
invariant bridge: hole-free states are skylines, debt states are holed
skylines, and the placement/clearing laws above compute every transition. -/

/-- The board realized by a debt-1 state: the skyline of `h`, minus the hole
if one is present. -/
def debtBoard (cfg : GameConfig) (h : ℕ → ℕ) : Option Coord → Board
  | none => skyline cfg h
  | some x => holedSkyline cfg h x

@[simp] theorem debtBoard_none (cfg : GameConfig) (h : ℕ → ℕ) :
    debtBoard cfg h none = skyline cfg h := rfl

@[simp] theorem debtBoard_some (cfg : GameConfig) (h : ℕ → ℕ) (x : Coord) :
    debtBoard cfg h (some x) = holedSkyline cfg h x := rfl

/-- The surface of a debt-1 board is its profile on real columns (the hole,
being strictly covered, is invisible) and `0` outside. -/
theorem colHeight_debtBoard {cfg : GameConfig} {h : ℕ → ℕ} {ho : Option Coord}
    (hcov : ∀ x, ho = some x → x.2 + 1 < h x.1) (j : ℕ) :
    (debtBoard cfg h ho).colHeight j = if j < cfg.cols then h j else 0 := by
  cases ho with
  | none =>
    rw [debtBoard_none]
    by_cases hj : j < cfg.cols
    · rw [colHeight_skyline hj, if_pos hj]
    · rw [colHeight_skyline_eq_zero (by omega), if_neg hj]
  | some x =>
    rw [debtBoard_some, colHeight_holedSkyline (hcov x rfl)]
    by_cases hj : j < cfg.cols
    · rw [colHeight_skyline hj, if_pos hj]
    · rw [colHeight_skyline_eq_zero (by omega), if_neg hj]

/-- **Hole creation: the S-on-flat bootstrap edge (skyline → holedSkyline).**
Placing a horizontal S on a flat surface `skyline (fun _ => base)` is *non-flush*:
it seats flush in columns `col, col+1` but its top-right cell overhangs column
`col+2`, burying exactly the cell `(col+2, base)` and producing a debt-1
`holedSkyline`. This is the missing skyline→holedSkyline transition — the
empty-board `S`-first bootstrap is the `base = 0` instance — completing the
debt-1 placement algebra beside `place_flush_skyline` (skyline→skyline),
`place_holedSkyline` (holedSkyline→holedSkyline), and the clearing laws
(holedSkyline→skyline). It is a transition-algebra brick only: it does not close
any invariant; the `DebtCertificate.step` all-orders closure (crux #66/#72)
remains the open obligation. -/
theorem place_horizS_flat_eq_holedSkyline (cfg : GameConfig) (base col : ℕ)
    (hcol : col + 2 < cfg.cols) :
    ({ piece := Piece.S, rot := 0, col := col } : Placement).place
        (skyline cfg (fun _ => base))
      = holedSkyline cfg
          (fun j => if j = col then base + 1
                    else if j = col + 1 then base + 2
                    else if j = col + 2 then base + 2 else base)
          (col + 2, base) := by
  have hsh : ({ piece := Piece.S, rot := 0, col := col } : Placement).shapeUp
      = {((0 : ℕ), (0 : ℕ)), (1, 0), (1, 1), (2, 1)} :=
    shapeUp_horizS col 0 (by decide)
  have hc0 : col + 0 < cfg.cols := by omega
  have hc1 : col + 1 < cfg.cols := by omega
  have hd : ({ piece := Piece.S, rot := 0, col := col } : Placement).dropOffset
      (skyline cfg (fun _ => base)) = base := by
    rw [Placement.dropOffset_eq_sup, hsh]
    simp only [Finset.sup_insert, Finset.sup_singleton,
      colHeight_skyline hc0, colHeight_skyline hc1, colHeight_skyline hcol]
    omega
  have hdr : ({ piece := Piece.S, rot := 0, col := col } : Placement).dropped
      (skyline cfg (fun _ => base))
      = {(col, base), (col + 1, base), (col + 1, base + 1), (col + 2, base + 1)} := by
    rw [Placement.dropped_eq_image, hsh, hd]
    simp only [Finset.image_insert, Finset.image_singleton]
    norm_num
  rw [Placement.place_eq_union_dropped, hdr]
  ext ⟨a, b⟩
  simp only [Finset.mem_union, mem_skyline', mem_holedSkyline, Finset.mem_insert,
    Finset.mem_singleton, ne_eq, Prod.mk.injEq]
  split_ifs <;> omega

end Board
end Tetris
