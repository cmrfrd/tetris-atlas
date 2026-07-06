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

/-- **Hole creation, the Z mirror.** Placing a horizontal Z on a flat surface is
non-flush and buries the *left* column's cell `(col, base)` (the mirror of
`place_horizS_flat_eq_holedSkyline`, which buries `(col+2, base)`). Together the
two cover the only pieces that force a hole on a flat surface: on a flat skyline
`O, I, T, L, J` all seat flush, while `S` and `Z` cannot. -/
theorem place_horizZ_flat_eq_holedSkyline (cfg : GameConfig) (base col : ℕ)
    (hcol : col + 2 < cfg.cols) :
    ({ piece := Piece.Z, rot := 0, col := col } : Placement).place
        (skyline cfg (fun _ => base))
      = holedSkyline cfg
          (fun j => if j = col then base + 2
                    else if j = col + 1 then base + 2
                    else if j = col + 2 then base + 1 else base)
          (col, base) := by
  have hsh : ({ piece := Piece.Z, rot := 0, col := col } : Placement).shapeUp
      = {((0 : ℕ), (1 : ℕ)), (1, 0), (1, 1), (2, 0)} :=
    shapeUp_horizZ col 0 (by decide)
  have hc0 : col + 0 < cfg.cols := by omega
  have hc1 : col + 1 < cfg.cols := by omega
  have hd : ({ piece := Piece.Z, rot := 0, col := col } : Placement).dropOffset
      (skyline cfg (fun _ => base)) = base := by
    rw [Placement.dropOffset_eq_sup, hsh]
    simp only [Finset.sup_insert, Finset.sup_singleton,
      colHeight_skyline hc0, colHeight_skyline hc1, colHeight_skyline hcol]
    omega
  have hdr : ({ piece := Piece.Z, rot := 0, col := col } : Placement).dropped
      (skyline cfg (fun _ => base))
      = {(col, base + 1), (col + 1, base), (col + 1, base + 1), (col + 2, base)} := by
    rw [Placement.dropped_eq_image, hsh, hd]
    simp only [Finset.image_insert, Finset.image_singleton]
    norm_num
  rw [Placement.place_eq_union_dropped, hdr]
  ext ⟨a, b⟩
  simp only [Finset.mem_union, mem_skyline', mem_holedSkyline, Finset.mem_insert,
    Finset.mem_singleton, ne_eq, Prod.mk.injEq]
  split_ifs <;> omega

/-- **The init-S bootstrap step, in `DebtCertificate.step` shape.** From the
empty board (`debtBoard (fun _ => 0) none`), announcing S, the full move
`applyStep` (drop + clear) lands in the debt-1 board that buries `(col+2, 0)` —
no rows clear (the empty board has none full), so `applyStep = place`. This is
the concrete first move of the empty-board bootstrap in the exact
`applyStep (debtBoard …) = debtBoard …` form the certificate's `step` demands.
Still a bootstrap brick: it is one `(state, piece)` pair, and the family `P` it
must land inside (the closure of crux #66/#72) is not constructed. -/
theorem applyStep_flat_horizS (col : ℕ)
    (hcol : col + 2 < GameConfig.standard.cols) :
    Placement.applyStep GameConfig.standard
        (debtBoard GameConfig.standard (fun _ => 0) none)
        { piece := Piece.S, rot := 0, col := col }
      = debtBoard GameConfig.standard
          (fun j => if j = col then 0 + 1 else if j = col + 1 then 0 + 2
                    else if j = col + 2 then 0 + 2 else 0)
          (some (col + 2, 0)) := by
  rw [debtBoard_none, Placement.applyStep_eq_clearLines_place,
      place_horizS_flat_eq_holedSkyline GameConfig.standard 0 col hcol, debtBoard_some]
  have hfree : ∃ j < GameConfig.standard.cols,
      (fun j => if j = col then (0 : ℕ) + 1 else if j = col + 1 then 0 + 2
                else if j = col + 2 then 0 + 2 else 0) j = 0 := by
    rcases Nat.lt_or_ge col 3 with h | h
    · exact ⟨9, by decide, by simp only []; split_ifs <;> omega⟩
    · exact ⟨0, by decide, by simp only []; split_ifs <;> omega⟩
  rw [clearLines_holedSkyline_of_le (m := 0) hcol (fun j _ => Nat.zero_le _) hfree
        (Nat.zero_le _)]
  simp only [Nat.sub_zero]

/-- **The init-Z bootstrap step**, the mirror of `applyStep_flat_horizS`: from
the empty board, announcing Z, the full move buries `(col, 0)`. Together the two
give the empty board's response, in `DebtCertificate.step` shape, to each of the
two pieces that force a hole from flat. -/
theorem applyStep_flat_horizZ (col : ℕ)
    (hcol : col + 2 < GameConfig.standard.cols) :
    Placement.applyStep GameConfig.standard
        (debtBoard GameConfig.standard (fun _ => 0) none)
        { piece := Piece.Z, rot := 0, col := col }
      = debtBoard GameConfig.standard
          (fun j => if j = col then 0 + 2 else if j = col + 1 then 0 + 2
                    else if j = col + 2 then 0 + 1 else 0)
          (some (col, 0)) := by
  rw [debtBoard_none, Placement.applyStep_eq_clearLines_place,
      place_horizZ_flat_eq_holedSkyline GameConfig.standard 0 col hcol, debtBoard_some]
  have hfree : ∃ j < GameConfig.standard.cols,
      (fun j => if j = col then (0 : ℕ) + 2 else if j = col + 1 then 0 + 2
                else if j = col + 2 then 0 + 1 else 0) j = 0 := by
    rcases Nat.lt_or_ge col 3 with h | h
    · exact ⟨9, by decide, by simp only []; split_ifs <;> omega⟩
    · exact ⟨0, by decide, by simp only []; split_ifs <;> omega⟩
  rw [clearLines_holedSkyline_of_le (m := 0) (show col < GameConfig.standard.cols by omega)
        (fun j _ => Nat.zero_le _) hfree (Nat.zero_le _)]
  simp only [Nat.sub_zero]

/-- **Flush placement of O on a flat surface (no hole).** The 2×2 `O` seats
flush on `skyline (fun _ => base)`, raising columns `col, col+1` by 2 with no
buried cell — the debt-free companion to the S/Z hole-creation lemmas. -/
theorem place_O_flat (cfg : GameConfig) (base col : ℕ) (hcol : col + 1 < cfg.cols) :
    ({ piece := Piece.O, rot := 0, col := col } : Placement).place
        (skyline cfg (fun _ => base))
      = skyline cfg (fun j => if j = col ∨ j = col + 1 then base + 2 else base) := by
  have hsh : ({ piece := Piece.O, rot := 0, col := col } : Placement).shapeUp
      = {((0 : ℕ), (0 : ℕ)), (0, 1), (1, 0), (1, 1)} := shapeUp_O col 0
  have hc0 : col + 0 < cfg.cols := by omega
  have hc1 : col + 1 < cfg.cols := by omega
  have hd : ({ piece := Piece.O, rot := 0, col := col } : Placement).dropOffset
      (skyline cfg (fun _ => base)) = base := by
    rw [Placement.dropOffset_eq_sup, hsh]
    simp only [Finset.sup_insert, Finset.sup_singleton,
      colHeight_skyline hc0, colHeight_skyline hc1]
    omega
  have hdr : ({ piece := Piece.O, rot := 0, col := col } : Placement).dropped
      (skyline cfg (fun _ => base))
      = {(col, base), (col, base + 1), (col + 1, base), (col + 1, base + 1)} := by
    rw [Placement.dropped_eq_image, hsh, hd]
    simp only [Finset.image_insert, Finset.image_singleton]
    norm_num
  rw [Placement.place_eq_union_dropped, hdr]
  ext ⟨a, b⟩
  simp only [Finset.mem_union, mem_skyline', Finset.mem_insert, Finset.mem_singleton,
    Prod.mk.injEq]
  split_ifs <;> omega

/-- **The init-O step, in `DebtCertificate.step` shape**: from the empty board,
announcing O, the full move lands in a *hole-free* `debtBoard … none` (no rows
clear on the empty board). The debt-free analogue of `applyStep_flat_horizS`. -/
theorem applyStep_flat_O (col : ℕ) (hcol : col + 1 < GameConfig.standard.cols) :
    Placement.applyStep GameConfig.standard
        (debtBoard GameConfig.standard (fun _ => 0) none)
        { piece := Piece.O, rot := 0, col := col }
      = debtBoard GameConfig.standard
          (fun j => if j = col ∨ j = col + 1 then 0 + 2 else 0) none := by
  rw [debtBoard_none, Placement.applyStep_eq_clearLines_place,
      place_O_flat GameConfig.standard 0 col hcol, debtBoard_none]
  have hfree : ∃ j < GameConfig.standard.cols,
      (fun j => if j = col ∨ j = col + 1 then (0 : ℕ) + 2 else 0) j = 0 := by
    rcases Nat.lt_or_ge col 3 with h | h
    · exact ⟨9, by decide, if_neg (by omega)⟩
    · exact ⟨0, by decide, if_neg (by omega)⟩
  rw [clearLines_skyline (m := 0) (fun j _ => Nat.zero_le _) hfree]
  simp only [Nat.sub_zero]

/-- The O placement is in-bounds when `col + 1 < cols`. -/
theorem valid_O (col : ℕ) (hcol : col + 1 < GameConfig.standard.cols) :
    ({ piece := Piece.O, rot := 0, col := col } : Placement).Valid GameConfig.standard := by
  intro cell hcell
  rw [show ({ piece := Piece.O, rot := 0, col := col } : Placement).shapeUp
        = {((0 : ℕ), (0 : ℕ)), (0, 1), (1, 0), (1, 1)} from shapeUp_O col 0] at hcell
  fin_cases hcell <;> simp_all <;> omega

/-- **`init` has a valid *hole-free* response to a first `O`** — the
`DebtCertificate.step` conjuncts (1)–(3) for `(init, O)`, landing in a
`debtBoard … none` (debt 0). Same open residual (conjunct 4, closed-family
membership) as the S/Z cases. -/
theorem exists_debtBoard_step_flat_O (col : ℕ)
    (hcol : col + 1 < GameConfig.standard.cols) :
    ∃ (pl : Placement) (h' : ℕ → ℕ) (ho' : Option Coord),
      pl.piece = Piece.O ∧ pl.Valid GameConfig.standard ∧
        Placement.applyStep GameConfig.standard
          (debtBoard GameConfig.standard (fun _ => 0) none) pl
          = debtBoard GameConfig.standard h' ho' :=
  ⟨_, _, _, rfl, valid_O col hcol, applyStep_flat_O col hcol⟩

/-- **Flush placement of a vertical I (the drain piece) on a flat surface.** The
1×4 vertical I seats flush, raising column `col` by 4 with no hole. -/
theorem place_vertI_flat (cfg : GameConfig) (base col : ℕ) (hcol : col < cfg.cols) :
    ({ piece := Piece.I, rot := 1, col := col } : Placement).place
        (skyline cfg (fun _ => base))
      = skyline cfg (fun j => if j = col then base + 4 else base) := by
  have hsh : ({ piece := Piece.I, rot := 1, col := col } : Placement).shapeUp
      = {((0 : ℕ), (0 : ℕ)), (0, 1), (0, 2), (0, 3)} := shapeUp_vertI' col 1 (by decide)
  have hc0 : col + 0 < cfg.cols := by omega
  have hd : ({ piece := Piece.I, rot := 1, col := col } : Placement).dropOffset
      (skyline cfg (fun _ => base)) = base := by
    rw [Placement.dropOffset_eq_sup, hsh]
    simp only [Finset.sup_insert, Finset.sup_singleton, colHeight_skyline hc0]
    omega
  have hdr : ({ piece := Piece.I, rot := 1, col := col } : Placement).dropped
      (skyline cfg (fun _ => base))
      = {(col, base), (col, base + 1), (col, base + 2), (col, base + 3)} := by
    rw [Placement.dropped_eq_image, hsh, hd]
    simp only [Finset.image_insert, Finset.image_singleton]
    norm_num
  rw [Placement.place_eq_union_dropped, hdr]
  ext ⟨a, b⟩
  simp only [Finset.mem_union, mem_skyline', Finset.mem_insert, Finset.mem_singleton,
    Prod.mk.injEq]
  split_ifs <;> omega

/-- **The init-I drain step, in `DebtCertificate.step` shape**: from the empty
board, announcing I, the vertical drop lands in a hole-free `debtBoard … none`
(column `col` at height 4). -/
theorem applyStep_flat_vertI (col : ℕ) (hcol : col < GameConfig.standard.cols) :
    Placement.applyStep GameConfig.standard
        (debtBoard GameConfig.standard (fun _ => 0) none)
        { piece := Piece.I, rot := 1, col := col }
      = debtBoard GameConfig.standard
          (fun j => if j = col then 0 + 4 else 0) none := by
  rw [debtBoard_none, Placement.applyStep_eq_clearLines_place,
      place_vertI_flat GameConfig.standard 0 col hcol, debtBoard_none]
  have hfree : ∃ j < GameConfig.standard.cols,
      (fun j => if j = col then (0 : ℕ) + 4 else 0) j = 0 := by
    rcases Nat.eq_zero_or_pos col with h | h
    · exact ⟨1, by decide, by simp only []; split_ifs <;> omega⟩
    · exact ⟨0, by decide, by simp only []; split_ifs <;> omega⟩
  rw [clearLines_skyline (m := 0) (fun j _ => Nat.zero_le _) hfree]
  simp only [Nat.sub_zero]

/-- The vertical-I placement is in-bounds when `col < cols`. -/
theorem valid_vertI (col : ℕ) (hcol : col < GameConfig.standard.cols) :
    ({ piece := Piece.I, rot := 1, col := col } : Placement).Valid GameConfig.standard := by
  intro cell hcell
  rw [show ({ piece := Piece.I, rot := 1, col := col } : Placement).shapeUp
        = {((0 : ℕ), (0 : ℕ)), (0, 1), (0, 2), (0, 3)} from shapeUp_vertI' col 1 (by decide)]
    at hcell
  fin_cases hcell <;> simp_all <;> omega

/-- **`init` has a valid response to a first `I`** (the drain piece) —
`DebtCertificate.step` conjuncts (1)–(3), landing hole-free. Same open residual
(conjunct 4) as S/Z/O. -/
theorem exists_debtBoard_step_flat_I (col : ℕ)
    (hcol : col < GameConfig.standard.cols) :
    ∃ (pl : Placement) (h' : ℕ → ℕ) (ho' : Option Coord),
      pl.piece = Piece.I ∧ pl.Valid GameConfig.standard ∧
        Placement.applyStep GameConfig.standard
          (debtBoard GameConfig.standard (fun _ => 0) none) pl
          = debtBoard GameConfig.standard h' ho' :=
  ⟨_, _, _, rfl, valid_vertI col hcol, applyStep_flat_vertI col hcol⟩

/-- The flat-bottom T (rotation 2) drop profile. -/
theorem shapeUp_flatT (c : ℕ) :
    ({ piece := Piece.T, rot := 2, col := c } : Placement).shapeUp
      = {((0 : ℕ), (0 : ℕ)), (1, 0), (1, 1), (2, 0)} := by
  show Piece.shapeUp Piece.T 2 = _
  decide

/-- **Flush placement of the flat-bottom T on a flat surface.** -/
theorem place_flatT (cfg : GameConfig) (base col : ℕ) (hcol : col + 2 < cfg.cols) :
    ({ piece := Piece.T, rot := 2, col := col } : Placement).place
        (skyline cfg (fun _ => base))
      = skyline cfg (fun j => if j = col then base + 1 else if j = col + 1 then base + 2
                              else if j = col + 2 then base + 1 else base) := by
  have hsh := shapeUp_flatT col
  have hc0 : col + 0 < cfg.cols := by omega
  have hc1 : col + 1 < cfg.cols := by omega
  have hd : ({ piece := Piece.T, rot := 2, col := col } : Placement).dropOffset
      (skyline cfg (fun _ => base)) = base := by
    rw [Placement.dropOffset_eq_sup, hsh]
    simp only [Finset.sup_insert, Finset.sup_singleton,
      colHeight_skyline hc0, colHeight_skyline hc1, colHeight_skyline hcol]
    omega
  have hdr : ({ piece := Piece.T, rot := 2, col := col } : Placement).dropped
      (skyline cfg (fun _ => base))
      = {(col, base), (col + 1, base), (col + 1, base + 1), (col + 2, base)} := by
    rw [Placement.dropped_eq_image, hsh, hd]
    simp only [Finset.image_insert, Finset.image_singleton]
    norm_num
  rw [Placement.place_eq_union_dropped, hdr]
  ext ⟨a, b⟩
  simp only [Finset.mem_union, mem_skyline', Finset.mem_insert, Finset.mem_singleton,
    Prod.mk.injEq]
  split_ifs <;> omega

/-- **The init-T step, in `DebtCertificate.step` shape** (hole-free). -/
theorem applyStep_flat_T (col : ℕ) (hcol : col + 2 < GameConfig.standard.cols) :
    Placement.applyStep GameConfig.standard
        (debtBoard GameConfig.standard (fun _ => 0) none)
        { piece := Piece.T, rot := 2, col := col }
      = debtBoard GameConfig.standard
          (fun j => if j = col then 0 + 1 else if j = col + 1 then 0 + 2
                    else if j = col + 2 then 0 + 1 else 0) none := by
  rw [debtBoard_none, Placement.applyStep_eq_clearLines_place,
      place_flatT GameConfig.standard 0 col hcol, debtBoard_none]
  have hfree : ∃ j < GameConfig.standard.cols,
      (fun j => if j = col then (0 : ℕ) + 1 else if j = col + 1 then 0 + 2
                else if j = col + 2 then 0 + 1 else 0) j = 0 := by
    rcases Nat.lt_or_ge col 3 with h | h
    · exact ⟨9, by decide, by simp only []; split_ifs <;> omega⟩
    · exact ⟨0, by decide, by simp only []; split_ifs <;> omega⟩
  rw [clearLines_skyline (m := 0) (fun j _ => Nat.zero_le _) hfree]
  simp only [Nat.sub_zero]

/-- The flat-bottom T placement is in-bounds when `col + 2 < cols`. -/
theorem valid_flatT (col : ℕ) (hcol : col + 2 < GameConfig.standard.cols) :
    ({ piece := Piece.T, rot := 2, col := col } : Placement).Valid GameConfig.standard := by
  intro cell hcell
  rw [shapeUp_flatT col] at hcell
  fin_cases hcell <;> simp_all <;> omega

/-- **`init` has a valid hole-free response to a first `T`.** -/
theorem exists_debtBoard_step_flat_T (col : ℕ)
    (hcol : col + 2 < GameConfig.standard.cols) :
    ∃ (pl : Placement) (h' : ℕ → ℕ) (ho' : Option Coord),
      pl.piece = Piece.T ∧ pl.Valid GameConfig.standard ∧
        Placement.applyStep GameConfig.standard
          (debtBoard GameConfig.standard (fun _ => 0) none) pl
          = debtBoard GameConfig.standard h' ho' :=
  ⟨_, _, _, rfl, valid_flatT col hcol, applyStep_flat_T col hcol⟩

/-- The flat-bottom L (rotation 0) drop profile. -/
theorem shapeUp_flatL (c : ℕ) :
    ({ piece := Piece.L, rot := 0, col := c } : Placement).shapeUp
      = {((0 : ℕ), (0 : ℕ)), (1, 0), (2, 0), (2, 1)} := by
  show Piece.shapeUp Piece.L 0 = _
  decide

/-- **Flush placement of the flat-bottom L on a flat surface.** -/
theorem place_flatL (cfg : GameConfig) (base col : ℕ) (hcol : col + 2 < cfg.cols) :
    ({ piece := Piece.L, rot := 0, col := col } : Placement).place
        (skyline cfg (fun _ => base))
      = skyline cfg (fun j => if j = col then base + 1 else if j = col + 1 then base + 1
                              else if j = col + 2 then base + 2 else base) := by
  have hsh := shapeUp_flatL col
  have hc0 : col + 0 < cfg.cols := by omega
  have hc1 : col + 1 < cfg.cols := by omega
  have hd : ({ piece := Piece.L, rot := 0, col := col } : Placement).dropOffset
      (skyline cfg (fun _ => base)) = base := by
    rw [Placement.dropOffset_eq_sup, hsh]
    simp only [Finset.sup_insert, Finset.sup_singleton,
      colHeight_skyline hc0, colHeight_skyline hc1, colHeight_skyline hcol]
    omega
  have hdr : ({ piece := Piece.L, rot := 0, col := col } : Placement).dropped
      (skyline cfg (fun _ => base))
      = {(col, base), (col + 1, base), (col + 2, base), (col + 2, base + 1)} := by
    rw [Placement.dropped_eq_image, hsh, hd]
    simp only [Finset.image_insert, Finset.image_singleton]
    norm_num
  rw [Placement.place_eq_union_dropped, hdr]
  ext ⟨a, b⟩
  simp only [Finset.mem_union, mem_skyline', Finset.mem_insert, Finset.mem_singleton,
    Prod.mk.injEq]
  split_ifs <;> omega

/-- **The init-L step, in `DebtCertificate.step` shape** (hole-free). -/
theorem applyStep_flat_L (col : ℕ) (hcol : col + 2 < GameConfig.standard.cols) :
    Placement.applyStep GameConfig.standard
        (debtBoard GameConfig.standard (fun _ => 0) none)
        { piece := Piece.L, rot := 0, col := col }
      = debtBoard GameConfig.standard
          (fun j => if j = col then 0 + 1 else if j = col + 1 then 0 + 1
                    else if j = col + 2 then 0 + 2 else 0) none := by
  rw [debtBoard_none, Placement.applyStep_eq_clearLines_place,
      place_flatL GameConfig.standard 0 col hcol, debtBoard_none]
  have hfree : ∃ j < GameConfig.standard.cols,
      (fun j => if j = col then (0 : ℕ) + 1 else if j = col + 1 then 0 + 1
                else if j = col + 2 then 0 + 2 else 0) j = 0 := by
    rcases Nat.lt_or_ge col 3 with h | h
    · exact ⟨9, by decide, by simp only []; split_ifs <;> omega⟩
    · exact ⟨0, by decide, by simp only []; split_ifs <;> omega⟩
  rw [clearLines_skyline (m := 0) (fun j _ => Nat.zero_le _) hfree]
  simp only [Nat.sub_zero]

/-- The flat-bottom L placement is in-bounds when `col + 2 < cols`. -/
theorem valid_flatL (col : ℕ) (hcol : col + 2 < GameConfig.standard.cols) :
    ({ piece := Piece.L, rot := 0, col := col } : Placement).Valid GameConfig.standard := by
  intro cell hcell
  rw [shapeUp_flatL col] at hcell
  fin_cases hcell <;> simp_all <;> omega

/-- **`init` has a valid hole-free response to a first `L`.** -/
theorem exists_debtBoard_step_flat_L (col : ℕ)
    (hcol : col + 2 < GameConfig.standard.cols) :
    ∃ (pl : Placement) (h' : ℕ → ℕ) (ho' : Option Coord),
      pl.piece = Piece.L ∧ pl.Valid GameConfig.standard ∧
        Placement.applyStep GameConfig.standard
          (debtBoard GameConfig.standard (fun _ => 0) none) pl
          = debtBoard GameConfig.standard h' ho' :=
  ⟨_, _, _, rfl, valid_flatL col hcol, applyStep_flat_L col hcol⟩

/-- The flat-bottom J (rotation 0) drop profile. -/
theorem shapeUp_flatJ (c : ℕ) :
    ({ piece := Piece.J, rot := 0, col := c } : Placement).shapeUp
      = {((0 : ℕ), (0 : ℕ)), (0, 1), (1, 0), (2, 0)} := by
  show Piece.shapeUp Piece.J 0 = _
  decide

/-- **Flush placement of the flat-bottom J on a flat surface.** -/
theorem place_flatJ (cfg : GameConfig) (base col : ℕ) (hcol : col + 2 < cfg.cols) :
    ({ piece := Piece.J, rot := 0, col := col } : Placement).place
        (skyline cfg (fun _ => base))
      = skyline cfg (fun j => if j = col then base + 2 else if j = col + 1 then base + 1
                              else if j = col + 2 then base + 1 else base) := by
  have hsh := shapeUp_flatJ col
  have hc0 : col + 0 < cfg.cols := by omega
  have hc1 : col + 1 < cfg.cols := by omega
  have hd : ({ piece := Piece.J, rot := 0, col := col } : Placement).dropOffset
      (skyline cfg (fun _ => base)) = base := by
    rw [Placement.dropOffset_eq_sup, hsh]
    simp only [Finset.sup_insert, Finset.sup_singleton,
      colHeight_skyline hc0, colHeight_skyline hc1, colHeight_skyline hcol]
    omega
  have hdr : ({ piece := Piece.J, rot := 0, col := col } : Placement).dropped
      (skyline cfg (fun _ => base))
      = {(col, base), (col, base + 1), (col + 1, base), (col + 2, base)} := by
    rw [Placement.dropped_eq_image, hsh, hd]
    simp only [Finset.image_insert, Finset.image_singleton]
    norm_num
  rw [Placement.place_eq_union_dropped, hdr]
  ext ⟨a, b⟩
  simp only [Finset.mem_union, mem_skyline', Finset.mem_insert, Finset.mem_singleton,
    Prod.mk.injEq]
  split_ifs <;> omega

/-- **The init-J step, in `DebtCertificate.step` shape** (hole-free). -/
theorem applyStep_flat_J (col : ℕ) (hcol : col + 2 < GameConfig.standard.cols) :
    Placement.applyStep GameConfig.standard
        (debtBoard GameConfig.standard (fun _ => 0) none)
        { piece := Piece.J, rot := 0, col := col }
      = debtBoard GameConfig.standard
          (fun j => if j = col then 0 + 2 else if j = col + 1 then 0 + 1
                    else if j = col + 2 then 0 + 1 else 0) none := by
  rw [debtBoard_none, Placement.applyStep_eq_clearLines_place,
      place_flatJ GameConfig.standard 0 col hcol, debtBoard_none]
  have hfree : ∃ j < GameConfig.standard.cols,
      (fun j => if j = col then (0 : ℕ) + 2 else if j = col + 1 then 0 + 1
                else if j = col + 2 then 0 + 1 else 0) j = 0 := by
    rcases Nat.lt_or_ge col 3 with h | h
    · exact ⟨9, by decide, by simp only []; split_ifs <;> omega⟩
    · exact ⟨0, by decide, by simp only []; split_ifs <;> omega⟩
  rw [clearLines_skyline (m := 0) (fun j _ => Nat.zero_le _) hfree]
  simp only [Nat.sub_zero]

/-- The flat-bottom J placement is in-bounds when `col + 2 < cols`. -/
theorem valid_flatJ (col : ℕ) (hcol : col + 2 < GameConfig.standard.cols) :
    ({ piece := Piece.J, rot := 0, col := col } : Placement).Valid GameConfig.standard := by
  intro cell hcell
  rw [shapeUp_flatJ col] at hcell
  fin_cases hcell <;> simp_all <;> omega

/-- **`init` has a valid hole-free response to a first `J`.** -/
theorem exists_debtBoard_step_flat_J (col : ℕ)
    (hcol : col + 2 < GameConfig.standard.cols) :
    ∃ (pl : Placement) (h' : ℕ → ℕ) (ho' : Option Coord),
      pl.piece = Piece.J ∧ pl.Valid GameConfig.standard ∧
        Placement.applyStep GameConfig.standard
          (debtBoard GameConfig.standard (fun _ => 0) none) pl
          = debtBoard GameConfig.standard h' ho' :=
  ⟨_, _, _, rfl, valid_flatJ col hcol, applyStep_flat_J col hcol⟩

/-- The horizontal-S placement is in-bounds when `col + 2 < cols`. -/
theorem valid_horizS (col : ℕ) (hcol : col + 2 < GameConfig.standard.cols) :
    ({ piece := Piece.S, rot := 0, col := col } : Placement).Valid GameConfig.standard := by
  intro cell hcell
  rw [shapeUp_horizS col 0 (by decide)] at hcell
  fin_cases hcell <;> simp_all <;> omega

/-- The horizontal-Z placement is in-bounds when `col + 2 < cols`. -/
theorem valid_horizZ (col : ℕ) (hcol : col + 2 < GameConfig.standard.cols) :
    ({ piece := Piece.Z, rot := 0, col := col } : Placement).Valid GameConfig.standard := by
  intro cell hcell
  rw [shapeUp_horizZ col 0 (by decide)] at hcell
  fin_cases hcell <;> simp_all <;> omega

/-- **`init` has a valid debt-1 response to a first `S`** — the first three
conjuncts of `DebtCertificate.step` for the pair `(init, S)`: a valid S placement
whose full move lands in a `debtBoard`. The missing fourth conjunct is
`P (bag.draw S) h' ho'`, membership in the *closed* family `P` — exactly what
crux #66/#72 leaves open (no such `P` is known). -/
theorem exists_debtBoard_step_flat_S (col : ℕ)
    (hcol : col + 2 < GameConfig.standard.cols) :
    ∃ (pl : Placement) (h' : ℕ → ℕ) (ho' : Option Coord),
      pl.piece = Piece.S ∧ pl.Valid GameConfig.standard ∧
        Placement.applyStep GameConfig.standard
          (debtBoard GameConfig.standard (fun _ => 0) none) pl
          = debtBoard GameConfig.standard h' ho' :=
  ⟨_, _, _, rfl, valid_horizS col hcol, applyStep_flat_horizS col hcol⟩

/-- **`init` has a valid debt-1 response to a first `Z`** (the `Z` mirror of
`exists_debtBoard_step_flat_S`). -/
theorem exists_debtBoard_step_flat_Z (col : ℕ)
    (hcol : col + 2 < GameConfig.standard.cols) :
    ∃ (pl : Placement) (h' : ℕ → ℕ) (ho' : Option Coord),
      pl.piece = Piece.Z ∧ pl.Valid GameConfig.standard ∧
        Placement.applyStep GameConfig.standard
          (debtBoard GameConfig.standard (fun _ => 0) none) pl
          = debtBoard GameConfig.standard h' ho' :=
  ⟨_, _, _, rfl, valid_horizZ col hcol, applyStep_flat_horizZ col hcol⟩

/-- **`init` is a fully live debt-1 state: it has a valid debt-≤1 response to
EVERY piece.** For each of the 7 pieces `p`, there is a valid placement of `p`
whose full move from the empty board lands in a `debtBoard` (debt ≤ 1). This is
the COMPLETE base case of `DebtCertificate.step` conjuncts (1)–(3) at `init`:
S/Z answered at debt 1 (a buried cell), O/I/T/L/J answered hole-free at debt 0.

The one obligation NOT established is conjunct (4): that these successors lie in
a *closed* family `P` (`P (Bag.full.draw p) h' ho'`). Constructing such a `P` —
the all-orders adversarial closure, crux #66/#72 — is the open keystone, and no
such `P` is known. So this does not prove `TetrisSolvable`; it exhausts exactly
the part of `step` that is discharge-able without the closed-family design. -/
theorem init_responds_to_all_pieces (p : Piece) :
    ∃ (pl : Placement) (h' : ℕ → ℕ) (ho' : Option Coord),
      pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        Placement.applyStep GameConfig.standard
          (debtBoard GameConfig.standard (fun _ => 0) none) pl
          = debtBoard GameConfig.standard h' ho' := by
  cases p with
  | O => exact exists_debtBoard_step_flat_O 0 (by decide)
  | I => exact exists_debtBoard_step_flat_I 0 (by decide)
  | S => exact exists_debtBoard_step_flat_S 0 (by decide)
  | Z => exact exists_debtBoard_step_flat_Z 0 (by decide)
  | T => exact exists_debtBoard_step_flat_T 0 (by decide)
  | L => exact exists_debtBoard_step_flat_L 0 (by decide)
  | J => exact exists_debtBoard_step_flat_J 0 (by decide)

/-! ## Toward the inductive step: self-reproducing roughness windows

The base case (`init_responds_to_all_pieces`) handles the flat board. The
inductive step must respond from arbitrary debt-1 boards. The hard pieces are S
and Z (roughness): they need a "window" (a unit step in the surface) to seat
hole-free. The lemmas below show those windows are **self-reproducing** — placing
the roughness piece on its window lands on a surface with the *same* window — so
an adversary cannot starve S/Z by throwing them repeatedly. The surface grows by
2 per placement, to be reclaimed by an I-drain; managing that growth against the
fillers (O/T/L/J) is the open closure (crux #66/#72). -/

/-- **The S-window is self-reproducing.** On an S-step `h c = h (c+1) + 1`, the
vertical S seats flush and lands on a skyline whose profile still has the S-step
at `c` (both columns rose by 2). Hence S is placeable hole-free again — S bursts
never starve. -/
theorem place_vertS_step_reproduces {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc : c < cfg.cols) (hc1 : c + 1 < cfg.cols) (hstep : h c = h (c + 1) + 1) :
    ∃ h', Placement.place (skyline cfg h)
            { piece := Piece.S, rot := 1, col := c } = skyline cfg h'
        ∧ h' c = h' (c + 1) + 1 := by
  refine ⟨Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2),
    place_vertS_skyline hc hc1 hstep, ?_⟩
  have e1 : Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2) c
      = h c + 2 := by
    rw [Function.update_of_ne (show c ≠ c + 1 by omega), Function.update_self]
  have e2 : Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2) (c + 1)
      = h (c + 1) + 2 := by rw [Function.update_self]
  rw [e1, e2]; omega

/-- **The Z-window is self-reproducing** (mirror of the S case). On a Z-step
`h (c+1) = h c + 1`, the vertical Z reproduces the step — Z bursts never
starve. -/
theorem place_vertZ_step_reproduces {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc : c < cfg.cols) (hc1 : c + 1 < cfg.cols) (hstep : h (c + 1) = h c + 1) :
    ∃ h', Placement.place (skyline cfg h)
            { piece := Piece.Z, rot := 1, col := c } = skyline cfg h'
        ∧ h' (c + 1) = h' c + 1 := by
  refine ⟨Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2),
    place_vertZ_skyline hc hc1 hstep, ?_⟩
  have e1 : Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2) c
      = h c + 2 := by
    rw [Function.update_of_ne (show c ≠ c + 1 by omega), Function.update_self]
  have e2 : Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2) (c + 1)
      = h (c + 1) + 2 := by rw [Function.update_self]
  rw [e1, e2]; omega

/-- **General vertical-I placement.** The vertical I is one column wide, so on
any skyline it just raises its column by 4 (independent of the rest of the
surface). Generalises `place_vertI_flat`. -/
theorem place_vertI_col (cfg : GameConfig) (h : ℕ → ℕ) (w : ℕ) (hw : w < cfg.cols) :
    ({ piece := Piece.I, rot := 1, col := w } : Placement).place (skyline cfg h)
      = skyline cfg (Function.update h w (h w + 4)) := by
  have hsh : ({ piece := Piece.I, rot := 1, col := w } : Placement).shapeUp
      = {((0 : ℕ), (0 : ℕ)), (0, 1), (0, 2), (0, 3)} := shapeUp_vertI' w 1 (by decide)
  have hd : ({ piece := Piece.I, rot := 1, col := w } : Placement).dropOffset
      (skyline cfg h) = h w := by
    rw [Placement.dropOffset_eq_sup, hsh]
    simp only [Finset.sup_insert, Finset.sup_singleton, Nat.add_zero, colHeight_skyline hw]
    omega
  have hdr : ({ piece := Piece.I, rot := 1, col := w } : Placement).dropped (skyline cfg h)
      = {(w, h w), (w, h w + 1), (w, h w + 2), (w, h w + 3)} := by
    rw [Placement.dropped_eq_image, hsh, hd]
    simp only [Finset.image_insert, Finset.image_singleton]
    norm_num
  rw [Placement.place_eq_union_dropped, hdr]
  ext ⟨a, b⟩
  simp only [Finset.mem_union, mem_skyline', Finset.mem_insert, Finset.mem_singleton,
    Prod.mk.injEq]
  by_cases haw : a = w
  · subst haw; rw [Function.update_self]; omega
  · rw [Function.update_of_ne haw]; omega

/-- **The I-drain (the height-reclaiming move).** On a "well" board — one empty
column `w`, every other column at height ≥ 4 — dropping a vertical I into `w`
fills rows 0–3 across the board, clearing exactly 4 lines and lowering every
column by 4 (the well returns to 0). This is the Tetris move that reclaims the
+2-per-piece growth of the roughness windows; the open closure is scheduling it
often enough (7-bag I-gap ≤ 12) against the fillers. -/
theorem vertI_drain (cfg : GameConfig) (h : ℕ → ℕ) (w : ℕ) (hw : w < cfg.cols)
    (hwell : h w = 0) (hfull : ∀ j < cfg.cols, j ≠ w → 4 ≤ h j) :
    Placement.applyStep cfg (skyline cfg h) { piece := Piece.I, rot := 1, col := w }
      = skyline cfg (fun j => if j = w then 0 else h j - 4) := by
  rw [Placement.applyStep_eq_clearLines_place, place_vertI_col cfg h w hw]
  have hu : Function.update h w (h w + 4) = Function.update h w 4 := by rw [hwell]
  rw [hu, clearLines_skyline (m := 4)
        (fun j hj => by
          by_cases hjw : j = w
          · subst hjw; rw [Function.update_self]
          · rw [Function.update_of_ne hjw]; exact hfull j hj hjw)
        ⟨w, hw, by rw [Function.update_self]⟩]
  congr 1
  ext j
  by_cases hjw : j = w
  · subst hjw; rw [Function.update_self, if_pos rfl]
  · rw [Function.update_of_ne hjw, if_neg hjw]

/-- **Locality of `colHeight` under a placement.** A placement only touches its
own columns `pl.col + cell.1`; a column `j` disjoint from all of them keeps its
height. This is the basis of "a piece placed away from a window does not disturb
it" — the ingredient (with self-reproduction) for keeping several windows alive
simultaneously between drains. -/
theorem colHeight_place_of_notMem_cols (b : Board) (pl : Placement) (j : ℕ)
    (hj : ∀ cell ∈ pl.shapeUp, j ≠ pl.col + cell.1) :
    (pl.place b).colHeight j = b.colHeight j := by
  have he : ∀ x ∈ pl.dropped b, ¬ (x.1 = j) := by
    intro x hx
    rw [Placement.dropped_eq_image, Finset.mem_image] at hx
    obtain ⟨cell, hcell, rfl⟩ := hx
    exact fun hc => hj cell hcell hc.symm
  unfold Board.colHeight Board.colRows
  rw [Placement.place_eq_union_dropped, Finset.filter_union,
      Finset.filter_false_of_mem he, Finset.union_empty]

/-- **Two windows coexist.** A surface carrying an S-window at `c` and a Z-window
at a disjoint `d` keeps BOTH after playing the vertical S at its window: the
S-window self-reproduces, and the Z-window survives because S touches only its
own two columns (locality). Symmetrically for Z. So an adversary interleaving S
and Z on a two-window surface never breaks either window — composing
`place_vertS_step_reproduces` with `colHeight_place_of_notMem_cols`. (This is why
disjoint roughness windows persist *between* drains; the open crux is the
height growth and the well upkeep across the coupling drain — #66/#72.) -/
theorem place_vertS_preserves_disjoint_Zwindow {cfg : GameConfig} {h : ℕ → ℕ} {c d : ℕ}
    (hc : c < cfg.cols) (hc1 : c + 1 < cfg.cols) (hstepS : h c = h (c + 1) + 1)
    (hdc : d ≠ c) (hdc1 : d ≠ c + 1) (hd1c : d + 1 ≠ c) (hd1c1 : d + 1 ≠ c + 1)
    (hstepZ : h (d + 1) = h d + 1) :
    ∃ h', Placement.place (skyline cfg h) { piece := Piece.S, rot := 1, col := c }
            = skyline cfg h'
        ∧ h' c = h' (c + 1) + 1 ∧ h' (d + 1) = h' d + 1 := by
  refine ⟨Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2),
    place_vertS_skyline hc hc1 hstepS, ?_, ?_⟩
  · have ec : Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2) c
        = h c + 2 := by
      rw [Function.update_of_ne (show c ≠ c + 1 by omega), Function.update_self]
    have ec1 : Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2) (c + 1)
        = h (c + 1) + 2 := by rw [Function.update_self]
    rw [ec, ec1]; omega
  · have ed : Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2) d
        = h d := by rw [Function.update_of_ne hdc1, Function.update_of_ne hdc]
    have ed1 : Function.update (Function.update h c (h c + 2)) (c + 1) (h (c + 1) + 2) (d + 1)
        = h (d + 1) := by rw [Function.update_of_ne hd1c1, Function.update_of_ne hd1c]
    rw [ed, ed1]; exact hstepZ

/-- **The drain is benign to windows.** The I-drain subtracts 4 from every column
uniformly, so a window (a unit step `h c = h (c+1) + 1`) that sits above the
drain zone — both columns ≥ 4, guaranteed here by the well board's `hfull` — is
preserved by the drain. So the height-reclaiming move does NOT break the
roughness windows: only the fillers and well-upkeep can, which narrows crux
#66/#72 to those. -/
theorem vertI_drain_preserves_step {cfg : GameConfig} {h : ℕ → ℕ} {w c : ℕ}
    (hw : w < cfg.cols) (hwell : h w = 0) (hfull : ∀ j < cfg.cols, j ≠ w → 4 ≤ h j)
    (hc : c < cfg.cols) (hc1 : c + 1 < cfg.cols) (hcw : c ≠ w) (hc1w : c + 1 ≠ w)
    (hstep : h c = h (c + 1) + 1) :
    Placement.applyStep cfg (skyline cfg h) { piece := Piece.I, rot := 1, col := w }
        = skyline cfg (fun j => if j = w then 0 else h j - 4)
    ∧ (fun j => if j = w then 0 else h j - 4) c
        = (fun j => if j = w then 0 else h j - 4) (c + 1) + 1 := by
  refine ⟨vertI_drain cfg h w hw hwell hfull, ?_⟩
  have h4 := hfull (c + 1) hc1 hc1w
  simp only [if_neg hcw, if_neg hc1w]
  omega

/-- **Well upkeep.** The reservoir reserves an empty column `w` (the well) for
the I-drain. Any piece placed away from `w` keeps it empty (locality), and the
drain refills it to 0 (`vertI_drain`) — so between and across drains the well is
maintained, provided the fillers are seated off it. -/
theorem well_preserved {b : Board} (pl : Placement) {w : ℕ} (hw0 : b.colHeight w = 0)
    (hj : ∀ cell ∈ pl.shapeUp, w ≠ pl.col + cell.1) :
    (pl.place b).colHeight w = 0 := by
  rw [colHeight_place_of_notMem_cols b pl w hj, hw0]

/-- **Flush O on a local flat pair.** On two equal-height columns `h c = h (c+1)`
the 2×2 O seats flush and raises both by 2 (the local generalisation of
`place_O_flat`). -/
theorem place_O_pair {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc : c < cfg.cols) (hc1 : c + 1 < cfg.cols) (hpair : h c = h (c + 1)) :
    Placement.place (skyline cfg h) { piece := Piece.O, rot := 0, col := c }
      = skyline cfg
          (Function.update (Function.update h c (h c + 2)) (c + 1) (h c + 2)) := by
  refine place_flush_skyline (w := 2) (off := h c) (bot := fun _ => 0) (top := fun _ => 1)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [show ({ piece := Piece.O, rot := 0, col := c } : Placement).shapeUp
          = {((0 : ℕ), (0 : ℕ)), (0, 1), (1, 0), (1, 1)} from shapeUp_O c 0]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · show ({ piece := Piece.O, rot := 0, col := c } : Placement).col + i < cfg.cols
    interval_cases i <;> simpa
  · show h (({ piece := Piece.O, rot := 0, col := c } : Placement).col + i) = h c + 0
    interval_cases i
    · simp
    · show h (c + 1) = h c + 0; omega
  · show Function.update (Function.update h c (h c + 2)) (c + 1) (h c + 2)
        (({ piece := Piece.O, rot := 0, col := c } : Placement).col + i) = h c + 1 + 1
    interval_cases i
    · show Function.update (Function.update h c (h c + 2)) (c + 1) (h c + 2) c = h c + 1 + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]
    · show Function.update (Function.update h c (h c + 2)) (c + 1) (h c + 2) (c + 1) = h c + 1 + 1
      rw [Function.update_self]
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    show Function.update (Function.update h c (h c + 2)) (c + 1) (h c + 2) j = h j
    rw [Function.update_of_ne hj1, Function.update_of_ne hj0]

/-- **The O-filler is self-reproducing.** O on a flat pair `h c = h (c+1)` lands
on a skyline with the *same* flat pair (both columns +2) — so a filler region
persists under O, just as the roughness windows persist under S/Z. -/
theorem place_O_pair_reproduces {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc : c < cfg.cols) (hc1 : c + 1 < cfg.cols) (hpair : h c = h (c + 1)) :
    ∃ h', Placement.place (skyline cfg h) { piece := Piece.O, rot := 0, col := c }
            = skyline cfg h' ∧ h' c = h' (c + 1) := by
  refine ⟨Function.update (Function.update h c (h c + 2)) (c + 1) (h c + 2),
    place_O_pair hc hc1 hpair, ?_⟩
  rw [Function.update_of_ne (show c ≠ c + 1 by omega), Function.update_self,
      Function.update_self]

end Board
end Tetris
