import Mathlib
import Proofs.Invariants.Confluence

/-!
# Zone views: the board as a composition of zone-local games

The rely-guarantee decomposition of Tetris. A ZONE is a set of columns; a
zone's VIEW of the board is its cells in those columns. This file proves the
algebra that makes the decomposition real:

* `zoneView_place` — a zone-local placement commutes with the zone view:
  the zone evolves as its own game (`place` on the view), because the drop
  offset reads only the zone's column heights.
* `zoneView_place_of_disjoint` — an off-zone placement leaves the view
  untouched: other zones' moves are invisible.
* `zoneView_deleteRows` / `zoneView_clearLines` — the CLEAR SERVICE lemma:
  a global line clear acts on each zone's view as an externally-given row
  deletion (`deleteRows`), parameterized by the global full-row data. The
  clears are exactly the interface between zones; a zone assumes a deletion
  service, and the composed game discharges it where all zones' full rows
  meet (`isFull_iff_of_cover`).

Together with the Confluence toolkit (commutation, the 7! collapse,
shift-equivariance) this reduces the composed game to zone-local games plus
one scheduling interface — the object the per-zone contract computations
(`scripts/zone_contract_probe.py`) decide exactly.
-/

namespace Tetris
namespace Board

/-- The zone view: the cells of `b` in columns satisfying `Z`. -/
def zoneView (Z : ℕ → Prop) [DecidablePred Z] (b : Board) : Board :=
  b.filter (fun p => Z p.1)

@[simp] theorem mem_zoneView {Z : ℕ → Prop} [DecidablePred Z] {b : Board}
    {p : Coord} : p ∈ zoneView Z b ↔ p ∈ b ∧ Z p.1 :=
  Finset.mem_filter

/-- On its own columns, the zone view has the true column heights. -/
theorem colHeight_zoneView {Z : ℕ → Prop} [DecidablePred Z] (b : Board)
    {j : ℕ} (hj : Z j) : (zoneView Z b).colHeight j = b.colHeight j := by
  unfold colHeight colRows zoneView
  congr 2
  ext p
  simp only [Finset.mem_filter]
  exact ⟨fun ⟨⟨hp, _⟩, hc⟩ => ⟨hp, hc⟩,
         fun ⟨hp, hc⟩ => ⟨⟨hp, hc ▸ hj⟩, hc⟩⟩

/-- Generic row deletion: remove cells in rows satisfying `D`, drop the
survivors by `delBelow` (gravity data supplied externally). `clearLines`
is exactly `deleteRows` of the board's OWN full-row data — and the service
lemma below shows each zone receives precisely this operation. -/
def deleteRows (D : ℕ → Prop) [DecidablePred D] (delBelow : ℕ → ℕ)
    (b : Board) : Board :=
  (b.filter (fun p => ¬ D p.2)).image (fun p => (p.1, p.2 - delBelow p.2))

/-- `clearLines` is `deleteRows` at the board's own full-row data. -/
theorem clearLines_eq_deleteRows (cfg : GameConfig) (b : Board) :
    clearLines cfg b = deleteRows (isFull cfg b) (clearedBelow cfg b) b := rfl

/-- **Row deletion is zone-blind**: it commutes with every zone view. The
deletion map preserves columns, and both filters (zone, row) commute. -/
theorem zoneView_deleteRows (Z D : ℕ → Prop) [DecidablePred Z]
    [DecidablePred D] (f : ℕ → ℕ) (b : Board) :
    zoneView Z (deleteRows D f b) = deleteRows D f (zoneView Z b) := by
  unfold zoneView deleteRows
  rw [Finset.filter_image]
  congr 1
  ext p
  simp only [Finset.mem_filter]
  tauto

/-- **The clear-service lemma.** Each zone's view of a global line clear is
a row deletion parameterized by the GLOBAL full-row data `(isFull cfg b,
clearedBelow cfg b)`. This is the zones' entire coupling: a zone's game
evolves by its own placements (`zoneView_place`) plus this externally-
scheduled deletion — the "service" whose rate and discipline form the
rely-guarantee interface. -/
theorem zoneView_clearLines (cfg : GameConfig) (Z : ℕ → Prop)
    [DecidablePred Z] (b : Board) :
    zoneView Z (clearLines cfg b)
      = deleteRows (isFull cfg b) (clearedBelow cfg b) (zoneView Z b) :=
  zoneView_deleteRows Z (isFull cfg b) (clearedBelow cfg b) b

/-- A zone-local placement drops identically on the zone view: the offset
reads only the zone's column heights. -/
theorem dropOffset_zoneView {Z : ℕ → Prop} [DecidablePred Z] (b : Board)
    (pl : Placement) (hz : ∀ c ∈ pl.shapeUp, Z (pl.col + c.1)) :
    pl.dropOffset (zoneView Z b) = pl.dropOffset b := by
  rw [Placement.dropOffset_eq_sup, Placement.dropOffset_eq_sup]
  refine Finset.sup_congr rfl fun c hc => ?_
  simp only [colHeight_zoneView b (hz c hc)]

/-- **Zone equivariance of placement.** A placement whose footprint lies in
the zone commutes with the zone view: the zone's own game IS `place` on the
view. -/
theorem zoneView_place {Z : ℕ → Prop} [DecidablePred Z] (b : Board)
    (pl : Placement) (hz : ∀ c ∈ pl.shapeUp, Z (pl.col + c.1)) :
    zoneView Z (pl.place b) = pl.place (zoneView Z b) := by
  have hdz : pl.dropped (zoneView Z b) = pl.dropped b := by
    rw [Placement.dropped_eq_cellsAt, Placement.dropped_eq_cellsAt,
        dropOffset_zoneView b pl hz]
  show zoneView Z (b ∪ pl.dropped b) = zoneView Z b ∪ pl.dropped (zoneView Z b)
  rw [hdz]
  unfold zoneView
  rw [Finset.filter_union]
  congr 1
  ext p
  simp only [Finset.mem_filter, Placement.dropped_eq_image, Finset.mem_image]
  constructor
  · rintro ⟨hp, _⟩
    exact hp
  · rintro ⟨c, hc, rfl⟩
    exact ⟨⟨c, hc, rfl⟩, hz c hc⟩

/-- **Zone blindness of off-zone placements.** A placement whose footprint
avoids the zone leaves the view untouched: other zones' moves are invisible
until the clear service fires. -/
theorem zoneView_place_of_disjoint {Z : ℕ → Prop} [DecidablePred Z]
    (b : Board) (pl : Placement)
    (hz : ∀ c ∈ pl.shapeUp, ¬ Z (pl.col + c.1)) :
    zoneView Z (pl.place b) = zoneView Z b := by
  show zoneView Z (b ∪ pl.dropped b) = zoneView Z b
  unfold zoneView
  rw [Finset.filter_union]
  have : (pl.dropped b).filter (fun p => Z p.1) = ∅ := by
    rw [Finset.filter_eq_empty_iff]
    intro p hp
    rw [Placement.dropped_eq_image, Finset.mem_image] at hp
    obtain ⟨c, hc, rfl⟩ := hp
    exact hz c hc
  rw [this, Finset.union_empty]

/-- **Full rows are the meet of zone-full rows.** For zones covering the
columns, a row is globally full iff each zone sees its own columns full —
so the clear service each zone assumes is discharged exactly when every
zone's guarantee (bottom-fill discipline) aligns. Stated for a cover by
three zones (the STZ / MID / LJO split). -/
theorem isFull_iff_of_cover (cfg : GameConfig) (Z₁ Z₂ Z₃ : ℕ → Prop)
    [DecidablePred Z₁] [DecidablePred Z₂] [DecidablePred Z₃]
    (hcover : ∀ j < cfg.cols, Z₁ j ∨ Z₂ j ∨ Z₃ j) (b : Board) (r : ℕ) :
    isFull cfg b r ↔
      ((∀ c ∈ Finset.range cfg.cols, Z₁ c → (c, r) ∈ zoneView Z₁ b) ∧
       (∀ c ∈ Finset.range cfg.cols, Z₂ c → (c, r) ∈ zoneView Z₂ b) ∧
       (∀ c ∈ Finset.range cfg.cols, Z₃ c → (c, r) ∈ zoneView Z₃ b)) := by
  simp only [mem_zoneView]
  constructor
  · intro h
    exact ⟨fun c hc hz => ⟨h c hc, hz⟩, fun c hc hz => ⟨h c hc, hz⟩,
           fun c hc hz => ⟨h c hc, hz⟩⟩
  · rintro ⟨h1, h2, h3⟩ c hc
    rcases hcover c (Finset.mem_range.mp hc) with hz | hz | hz
    · exact (h1 c hc hz).1
    · exact (h2 c hc hz).1
    · exact (h3 c hc hz).1


/-- **The static rate-balance obstruction (arithmetic core).** A global line
clear removes one row from EVERY zone simultaneously, so over a long run a
static piece→zone assignment keeps every zone's height bounded only if each
zone's fill rate equals the one global clear rate: per supercycle of 5 bags,
`20 * k` cells arrive in a zone owning `k` pieces per bag while `r` global
clears remove `r * w` of its cells — balance forces `10 * k = 7 * w` (with
`r = 14` from the total). No zone with `1 ≤ k ≤ 6` pieces and `1 ≤ w ≤ 9`
columns satisfies it (`7 ∤ 10k` below `k = 7`): ONLY the whole board balances
statically. Every proper static zoning provably drifts — the reason the
per-zone closures (SZ band 3, LJO band 296) cannot compose statically, and
piece-assignment ROTATION (periodic contracts) is forced. -/
theorem static_zone_balance_impossible :
    ∀ k w : ℕ, 1 ≤ k → k ≤ 6 → 1 ≤ w → w ≤ 9 → 10 * k ≠ 7 * w := by
  intro k w hk1 hk6 hw1 hw9
  omega

/-- **Heights compose from zone views.** A column's height equals its height
in the view of any zone owning it (`colHeight_zoneView` restated for the
composition): global height bounds — hence the not-lost condition — follow
from per-zone bounds. The survival side of the composition is free; the
whole content lives in the contracts and the clear-service realization. -/
theorem colHeight_eq_zoneView_of_owner {Z : ℕ → Prop} [DecidablePred Z]
    (b : Board) {j : ℕ} (hj : Z j) :
    b.colHeight j = (zoneView Z b).colHeight j :=
  (colHeight_zoneView b hj).symm

end Board
end Tetris
