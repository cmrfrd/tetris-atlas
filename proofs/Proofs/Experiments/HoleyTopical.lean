import Mathlib
import Proofs.Theorems.BoardCount
import Proofs.Theorems.Gameplay
import Proofs.Experiments.SurfaceInvariant

/-!
# Including holes in the topical picture — the debt × roughness height budget

`TopicalTetris.lean` studies the **hole-free** surface dynamics: the drop is a topical
(monotone + homogeneous) max-plus map, survival ⟺ the projective *roughness*
(`maxHt − minHt`) stays bounded, and the recurrent shape is a single eigen-surface. Its
closing note asks for the missing **defect theory for holes** — the glue that lets a holey
board reuse that picture.

This file supplies that glue, natively on the real engine `Board`. The reframe (justified by
`SurfaceFiber.colHeight_place_eq_of_colHeight_eq`, that placement is hole-blind on the
surface) is: **don't fold holes into the map — couple a scalar counter to it.** Holes never
perturb the *surface* trajectory; what they do is *throttle clears* (a buried empty makes its
row un-full, so it can never clear), and that blockage is exactly `Board.holes` (`= debt`).

## The coupling bridge (all sorry-free)

The genuinely new content is one counting inequality and its consequences:

* `minColHeight_le_holes_add_fullRows` — **`minHt ≤ holes + |fullRows|`.** Every row below the
  lowest column is either full or holds a buried empty; the non-full ones inject into the hole
  set, the full ones into `fullRows`.
* `maxColHeight_le_holes_add_fullRows_add_roughness` — **the height budget**
  `maxHt ≤ holes + |fullRows| + roughness`, and post-clear (`|fullRows| = 0`)
  `maxColHeight_clearLines_le_holes_add_roughness` : `maxHt ≤ holes + roughness`.
* `not_isLost_of_holes_add_fullRows_add_roughness_le` — **the survival certificate**: a WF board
  with `holes + |fullRows| + roughness ≤ rows` is not lost. Survival reduces to a single
  resource inequality — keep the topical quantity (`roughness`) plus the hole counter
  (`holes`/`debt`) within the row budget. `roughness` is what `TopicalTetris` bounds; `holes`
  is the Lyapunov counter `HoleDebt` controls.

## Holes break the clear's projective invariance (the honest characterization)

`TopicalTetris`'s `clearMap` is a *uniform shift* (shape-preserving). That is a hole-free
idealization: `clearLines_uniform_shift_false` exhibits a holey board where clearing drops
columns by *different* amounts (a column supported only by the cleared row collapses), so a
clear is **not** a uniform surface shift once holes are present — precisely why holes are the
homogeneity defect.
-/

namespace Tetris.HoleyTopical

open Tetris Tetris.Board

/-! ## A column's filled rows, characterised by membership -/

/-- `r` is a filled row of column `j` iff the cell `(j, r)` is filled. -/
theorem mem_colRows_char (b : Board) (j r : ℕ) : r ∈ b.colRows j ↔ (j, r) ∈ b := by
  unfold Board.colRows
  simp only [Finset.mem_image, Finset.mem_filter]
  constructor
  · rintro ⟨p, ⟨hp, hpj⟩, rfl⟩
    obtain ⟨a, c⟩ := p
    simp only at hpj
    subst hpj
    exact hp
  · intro h
    exact ⟨(j, r), ⟨h, rfl⟩, rfl⟩

/-! ## The minimum column height and the projective roughness on `Board` -/

/-- The shortest column height across the real field — the floor of the surface. -/
def minColHeight (cfg : GameConfig) (b : Board) : ℕ :=
  (Finset.range cfg.cols).inf' ⟨0, Finset.mem_range.mpr cfg.cols_pos⟩ (colHeight b)

/-- The projective **roughness** of the surface: `maxColHeight − minColHeight`, the real-board
analogue of `TopicalTetris.roughness` (the Hilbert projective diameter of the shape). -/
def roughness (cfg : GameConfig) (b : Board) : ℕ :=
  maxColHeight cfg b - minColHeight cfg b

/-- Every real column sits at or above the floor. -/
theorem minColHeight_le_colHeight {cfg : GameConfig} {b : Board} {j : ℕ}
    (hj : j < cfg.cols) : minColHeight cfg b ≤ colHeight b j := by
  unfold minColHeight
  exact Finset.inf'_le (colHeight b) (Finset.mem_range.mpr hj)

/-- The floor never exceeds the ceiling. -/
theorem minColHeight_le_maxColHeight (cfg : GameConfig) (b : Board) :
    minColHeight cfg b ≤ maxColHeight cfg b :=
  le_trans (minColHeight_le_colHeight cfg.cols_pos) (colHeight_le_maxColHeight cfg.cols_pos)

/-! ## The core counting lemma -/

/-- **Buried empties below a level are bounded by the column's holes.** If `m ≤ colHeight b j`
then the empty cells of column `j` below row `m` are a subset of all buried empties of the
column, which number `colHoles b j`. (The empties below `m` and the filled rows of the column
are disjoint subsets of `range (colHeight b j)`.) -/
theorem card_empties_below_le_colHoles (b : Board) (j m : ℕ) (hm : m ≤ b.colHeight j) :
    ((Finset.range m).filter (fun r => (j, r) ∉ b)).card ≤ colHoles b j := by
  classical
  have hMrange : (Finset.range m).filter (fun r => (j, r) ∉ b)
      ⊆ Finset.range (b.colHeight j) := by
    intro r hr
    rw [Finset.mem_filter, Finset.mem_range] at hr
    exact Finset.mem_range.mpr (lt_of_lt_of_le hr.1 hm)
  have hCrange : b.colRows j ⊆ Finset.range (b.colHeight j) := by
    intro r hr
    rw [mem_colRows_char] at hr
    exact Finset.mem_range.mpr (Board.lt_colHeight hr)
  have hdisj : Disjoint ((Finset.range m).filter (fun r => (j, r) ∉ b)) (b.colRows j) := by
    rw [Finset.disjoint_left]
    intro r hrM hrC
    rw [Finset.mem_filter] at hrM
    rw [mem_colRows_char] at hrC
    exact hrM.2 hrC
  have hunion : ((Finset.range m).filter (fun r => (j, r) ∉ b)).card + (b.colRows j).card
      ≤ b.colHeight j := by
    rw [← Finset.card_union_of_disjoint hdisj]
    calc (((Finset.range m).filter (fun r => (j, r) ∉ b)) ∪ b.colRows j).card
        ≤ (Finset.range (b.colHeight j)).card :=
          Finset.card_le_card (Finset.union_subset hMrange hCrange)
      _ = b.colHeight j := Finset.card_range _
  have hCle : (b.colRows j).card ≤ b.colHeight j := Board.colRows_card_le_colHeight b j
  unfold Board.colHoles
  omega

/-- **The core coupling inequality: `minHt ≤ holes + |fullRows|`.** Partition the rows below the
floor `minColHeight` into full and non-full. The full ones embed into `fullRows`; each non-full
one is missing a column, so it lands in that column's buried empties — and below the floor every
column is tall enough for a missing cell to be a genuine hole. Summing,
`#non-full ≤ holes` and `#full ≤ |fullRows|`. -/
theorem minColHeight_le_holes_add_fullRows (cfg : GameConfig) (b : Board) :
    minColHeight cfg b ≤ holes cfg b + (fullRows cfg b).card := by
  classical
  have hsplit :
      ((Finset.range (minColHeight cfg b)).filter (fun r => isFull cfg b r)).card
        + ((Finset.range (minColHeight cfg b)).filter (fun r => ¬ isFull cfg b r)).card
        = minColHeight cfg b := by
    rw [Finset.card_filter_add_card_filter_not, Finset.card_range]
  have hFull :
      ((Finset.range (minColHeight cfg b)).filter (fun r => isFull cfg b r)).card
        ≤ (fullRows cfg b).card := by
    apply Finset.card_le_card
    intro r hr
    rw [Finset.mem_filter, Finset.mem_range] at hr
    rw [mem_fullRows_iff]
    exact ⟨Finset.mem_image.mpr ⟨(0, r), mem_of_isFull cfg cfg.cols_pos hr.2, rfl⟩, hr.2⟩
  have hNon :
      ((Finset.range (minColHeight cfg b)).filter (fun r => ¬ isFull cfg b r)).card
        ≤ holes cfg b := by
    have hsub :
        (Finset.range (minColHeight cfg b)).filter (fun r => ¬ isFull cfg b r)
          ⊆ (Finset.range cfg.cols).biUnion
              (fun j => (Finset.range (minColHeight cfg b)).filter (fun r => (j, r) ∉ b)) := by
      intro r hr
      rw [Finset.mem_filter, Finset.mem_range] at hr
      obtain ⟨c, hc, hcnotin⟩ := (not_isFull_iff_exists_missing cfg b r).mp hr.2
      exact Finset.mem_biUnion.mpr
        ⟨c, hc, Finset.mem_filter.mpr ⟨Finset.mem_range.mpr hr.1, hcnotin⟩⟩
    refine le_trans (Finset.card_le_card hsub) ?_
    refine le_trans Finset.card_biUnion_le ?_
    unfold Board.holes
    apply Finset.sum_le_sum
    intro j hj
    exact card_empties_below_le_colHoles b j (minColHeight cfg b)
      (minColHeight_le_colHeight (Finset.mem_range.mp hj))
  omega

/-! ## The height budget and the survival certificate -/

/-- **The height budget: `maxHt ≤ holes + |fullRows| + roughness`.** Immediate from the core
inequality and `maxHt = minHt + roughness`. -/
theorem maxColHeight_le_holes_add_fullRows_add_roughness (cfg : GameConfig) (b : Board) :
    maxColHeight cfg b ≤ holes cfg b + (fullRows cfg b).card + roughness cfg b := by
  have hcore := minColHeight_le_holes_add_fullRows cfg b
  have hmm := minColHeight_le_maxColHeight cfg b
  unfold roughness
  omega

/-- **The survival certificate.** A well-formed board whose hole counter plus full-row count
plus roughness fits the row budget is not lost: the budget caps `maxColHeight ≤ rows`, and
`not_isLost_of_maxColHeight_le` converts that to non-loss. -/
theorem not_isLost_of_holes_add_fullRows_add_roughness_le {cfg : GameConfig} {b : Board}
    (hwf : WF cfg b)
    (h : holes cfg b + (fullRows cfg b).card + roughness cfg b ≤ cfg.rows) :
    ¬ isLost cfg b :=
  not_isLost_of_maxColHeight_le hwf
    (le_trans (maxColHeight_le_holes_add_fullRows_add_roughness cfg b) h)

/-! ## Post-clear: the clean `holes + roughness` form

In real play the operative state is always *after* a clear, where no full rows remain. There the
`|fullRows|` term vanishes and the budget is the clean `maxHt ≤ holes + roughness`. -/

/-- After any line clear there are no full rows (gravity is injective on survivors, so it never
recombines partial rows into a full one — `clearLines_no_full`). -/
theorem fullRows_clearLines_eq_empty (cfg : GameConfig) (b : Board) :
    fullRows cfg (clearLines cfg b) = ∅ := by
  apply Finset.eq_empty_of_forall_notMem
  intro r hr
  exact clearLines_no_full b cfg.cols_pos r (isFull_of_mem_fullRows hr)

/-- **The post-clear height budget: `maxHt ≤ holes + roughness`.** The `|fullRows|` term is `0`
on a cleared board. -/
theorem maxColHeight_clearLines_le_holes_add_roughness (cfg : GameConfig) (b : Board) :
    maxColHeight cfg (clearLines cfg b)
      ≤ holes cfg (clearLines cfg b) + roughness cfg (clearLines cfg b) := by
  have h := maxColHeight_le_holes_add_fullRows_add_roughness cfg (clearLines cfg b)
  rw [fullRows_clearLines_eq_empty cfg b, Finset.card_empty] at h
  simpa using h

/-- **The post-clear survival certificate**: `holes + roughness ≤ rows` on a cleared board
forbids loss. -/
theorem not_isLost_clearLines_of_holes_add_roughness_le {cfg : GameConfig} {b : Board}
    (hwf : WF cfg b)
    (h : holes cfg (clearLines cfg b) + roughness cfg (clearLines cfg b) ≤ cfg.rows) :
    ¬ isLost cfg (clearLines cfg b) :=
  not_isLost_of_maxColHeight_le (clearLines_wf hwf)
    (le_trans (maxColHeight_clearLines_le_holes_add_roughness cfg b) h)

/-! ## Holes break the clear's projective invariance -/

/-- **A clear is NOT a uniform surface shift once holes are present.** Witness on the 4×4 grid:
the board has a single full row (row 1) whose removal drops most columns by `1`, but column `0`
— supported *only* by that row, with row `0` buried-empty — collapses by `2`. So
`colHeight (clearLines b) 0 + |fullRows b| = 0 + 1 ≠ 2 = colHeight b 0`: clearing changes the
shape. This is exactly the homogeneity defect that keeps `TopicalTetris`'s `clearMap` hole-free,
and the reason holes must ride as a separate counter rather than inside the topical map. -/
theorem clearLines_uniform_shift_false :
    ¬ ∀ (cfg : GameConfig) (b : Board) (j : ℕ),
        colHeight (clearLines cfg b) j + (fullRows cfg b).card = colHeight b j := by
  intro H
  have hbad := H GameConfig.tiny
    {((0 : ℕ), (1 : ℕ)), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)} 0
  revert hbad
  decide

#print axioms minColHeight_le_holes_add_fullRows
#print axioms maxColHeight_le_holes_add_fullRows_add_roughness
#print axioms not_isLost_of_holes_add_fullRows_add_roughness_le
#print axioms maxColHeight_clearLines_le_holes_add_roughness
#print axioms not_isLost_clearLines_of_holes_add_roughness_le
#print axioms clearLines_uniform_shift_false

end Tetris.HoleyTopical
