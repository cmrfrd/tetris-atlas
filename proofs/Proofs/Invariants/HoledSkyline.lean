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

end Board
end Tetris
