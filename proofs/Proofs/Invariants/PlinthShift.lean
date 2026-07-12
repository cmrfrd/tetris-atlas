import Proofs.Invariants.BandShift
import Proofs.Invariants.BandMechanisms

/-!
# The plinth: floor-1 transport for the entombed-hole regime

Findings D1/D2 (see PROGRESS.md, 2026-07-12): the `bandLift` certificates
cannot be inhabited — the base never rises, and the forced row-0 bootstrap
hole blocks re-anchoring. The plinth regime fixes both: one bag-1 placement
(`place_wellPlug_flat`) plugs the well's row 0; row 0 becomes a permanent
floor of nine cells plus the entombed hole, which keeps row 0 from ever
being full — the floor is immortal. The well operates at height 1, drains
fill and clear rows 1–4 (`drain_debtBoard_plinthLift`), and the active band
rides `c + 1` above the floor in coordinates the hole never touches, so
re-anchoring (`ReanchorsTo`) is unobstructed.
-/

namespace Tetris
namespace Board

/-- The plinth lift: well pinned at height 1 (the plug), band riding `c + 1`
above the floor. -/
def plinthLift (w c : ℕ) (ρ : ℕ → ℕ) : ℕ → ℕ :=
  fun j => if j = w then 1 else ρ j + c + 1

@[simp] theorem plinthLift_well (w c : ℕ) (ρ : ℕ → ℕ) :
    plinthLift w c ρ w = 1 := if_pos rfl

theorem plinthLift_ne (w c : ℕ) (ρ : ℕ → ℕ) {j : ℕ} (hj : j ≠ w) :
    plinthLift w c ρ j = ρ j + c + 1 := if_neg hj

/-- Re-anchoring: the same absolute band at a different base split. The
board-level no-op that lets the base rise as the pattern grows (fix for
finding D1). -/
def ReanchorsTo (well : ℕ) (ρ : ℕ → ℕ) (c : ℕ) (ρ' : ℕ → ℕ) (c' : ℕ) : Prop :=
  (∀ j, j ≠ well → ρ j + c = ρ' j + c') ∧ ρ' well = 0

/-- Re-anchored splits denote the same plinth profile. -/
theorem plinthLift_congr_reanchor {well : ℕ} {ρ ρ' : ℕ → ℕ} {c c' : ℕ}
    (h : ReanchorsTo well ρ c ρ' c') :
    plinthLift well c ρ = plinthLift well c' ρ' := by
  funext j
  by_cases hj : j = well
  · subst hj; simp [plinthLift]
  · rw [plinthLift_ne well c ρ hj, plinthLift_ne well c' ρ' hj]
    have := h.1 j hj
    omega

/-- **The floor is immortal.** A plinth board has no full rows: the well
column stops at height 1 (blocking every row ≥ 1) and the entombed hole
blocks row 0. -/
theorem fullRows_plinth_eq_empty {cfg : GameConfig} {w hx c : ℕ}
    {ρ : ℕ → ℕ} (hw : w < cfg.cols) (hxw : hx ≠ w) (hxc : hx < cfg.cols) :
    fullRows cfg (debtBoard cfg (plinthLift w c ρ) (some (hx, 0))) = ∅ := by
  rw [Finset.eq_empty_iff_forall_notMem]
  intro r hr
  unfold fullRows at hr
  rw [Finset.mem_filter] at hr
  have hfull := hr.2
  by_cases hr0 : r = 0
  · subst hr0
    have hmem := hfull hx (Finset.mem_range.mpr hxc)
    rw [mem_debtBoard] at hmem
    exact (hmem.1 (hx, 0) rfl) rfl
  · have hmem := hfull w (Finset.mem_range.mpr hw)
    rw [mem_debtBoard] at hmem
    have hlt := hmem.2.2
    simp only [plinthLift_well] at hlt
    omega

/-- The well-plug shape: `J` rot 3 — one cell in its left column at row 0,
three cells in the right column rows 0–2. -/
theorem shapeUp_wellPlugJ (c : ℕ) :
    ({ piece := Piece.J, rot := 3, col := c } : Placement).shapeUp
      = {((0 : ℕ), (0 : ℕ)), (1, 0), (1, 1), (1, 2)} := by
  show Piece.shapeUp Piece.J 3 = _
  decide

/-- **The well plug**: `J` rot 3 straddling the well boundary on flat ground
— exactly one cell lands at `(w, base)` (the plug), three seat flush on the
neighbor. No hole is created. The bag-1 entombment move (finding D2). -/
theorem place_wellPlug_flat (cfg : GameConfig) (base w : ℕ)
    (hw1 : w + 1 < cfg.cols) :
    ({ piece := Piece.J, rot := 3, col := w } : Placement).place
        (skyline cfg (fun _ => base))
      = skyline cfg (fun j =>
          if j = w then base + 1 else if j = w + 1 then base + 3 else base) := by
  have hsh := shapeUp_wellPlugJ w
  have hc0 : w + 0 < cfg.cols := by omega
  have hd : ({ piece := Piece.J, rot := 3, col := w } : Placement).dropOffset
      (skyline cfg (fun _ => base)) = base := by
    rw [Placement.dropOffset_eq_sup, hsh]
    simp only [Finset.sup_insert, Finset.sup_singleton,
      colHeight_skyline hc0, colHeight_skyline hw1]
    omega
  have hdr : ({ piece := Piece.J, rot := 3, col := w } : Placement).dropped
      (skyline cfg (fun _ => base))
      = {(w, base), (w + 1, base), (w + 1, base + 1), (w + 1, base + 2)} := by
    rw [Placement.dropped_eq_image, hsh, hd]
    simp only [Finset.image_insert, Finset.image_singleton]
    norm_num
  rw [Placement.place_eq_union_dropped, hdr]
  ext ⟨a, b⟩
  simp only [Finset.mem_union, mem_skyline', Finset.mem_insert,
    Finset.mem_singleton, Prod.mk.injEq]
  split_ifs <;> omega

/-- Hard-drop shift onto the plinth: the offset is the bare-skyline offset
plus `c + 1` (the entombed hole is strictly covered, hence invisible). -/
theorem dropOffset_plinthLift {cfg : GameConfig} {w hx c : ℕ}
    {ρ : ℕ → ℕ} {pl : Placement}
    (hcols : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 < cfg.cols)
    (havoid : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 ≠ w)
    (hhx : hx ≠ w ∧ hx < cfg.cols ∧ 1 ≤ ρ hx) :
    pl.dropOffset (debtBoard cfg (plinthLift w c ρ) (some (hx, 0)))
      = pl.dropOffset (skyline cfg ρ) + (c + 1) := by
  have hcovL : ∀ x, (some ((hx : ℕ), (0 : ℕ)) : Option Coord) = some x →
      x.2 + 1 < plinthLift w c ρ x.1 := by
    rintro x hx'
    rw [Option.some.injEq] at hx'
    subst hx'
    rw [plinthLift_ne w c ρ hhx.1]
    have := hhx.2.2
    omega
  have hL : pl.dropOffset (debtBoard cfg (plinthLift w c ρ) (some (hx, 0)))
      = pl.shapeUp.sup (fun cell => ρ (pl.col + cell.1) + (c + 1) - cell.2) := by
    rw [Placement.dropOffset_eq_sup]
    refine Finset.sup_congr rfl fun cell hcell => ?_
    rw [colHeight_debtBoard hcovL, if_pos (hcols cell hcell),
      plinthLift_ne w c ρ (havoid cell hcell),
      show ρ (pl.col + cell.1) + c + 1 = ρ (pl.col + cell.1) + (c + 1) from
        by omega]
  have hR : pl.dropOffset (skyline cfg ρ)
      = pl.shapeUp.sup (fun cell => ρ (pl.col + cell.1) - cell.2) := by
    rw [Placement.dropOffset_eq_sup]
    refine Finset.sup_congr rfl fun cell hcell => ?_
    rw [colHeight_skyline (hcols cell hcell)]
  rw [hL, hR]
  exact sup_sub_add_shift pl.shapeUp (fun cell => ρ (pl.col + cell.1)) (c + 1)
    (Placement.shapeUp_exists_bottom pl.piece pl.rot)

/-- **T1′ — plinth transport.** A bare flush skyline transition holds on the
plinth board at every base: landing cells ride `c + 1` above the floor, the
well column is `{(w, 0)}` on both sides, and the entombed hole never moves.
Rep obligations are bare-skyline, so every existing mechanism applies. -/
theorem place_debtBoard_plinthLift {cfg : GameConfig} {w hx c : ℕ}
    {ρ ρ' : ℕ → ℕ} {pl : Placement}
    (hcols : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 < cfg.cols)
    (havoid : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 ≠ w)
    (hhx : hx ≠ w ∧ hx < cfg.cols ∧ 1 ≤ ρ hx)
    (hrep : pl.place (skyline cfg ρ) = skyline cfg ρ') :
    pl.place (debtBoard cfg (plinthLift w c ρ) (some (hx, 0)))
      = debtBoard cfg (plinthLift w c ρ') (some (hx, 0)) := by
  have hD := dropOffset_plinthLift (cfg := cfg) (w := w) (hx := hx) (c := c)
    (ρ := ρ) (pl := pl) hcols havoid hhx
  have hpt := fun p => Finset.ext_iff.mp hrep p
  ext ⟨a, r⟩
  simp only [Placement.place_eq_union_dropped, Finset.mem_union,
    Placement.dropped_eq_cellsAt, hD, mem_cellsAt_add, mem_debtBoard]
  have hrep_ar := hpt (a, r - (c + 1))
  simp only [Placement.place_eq_union_dropped, Finset.mem_union,
    Placement.dropped_eq_cellsAt, mem_skyline'] at hrep_ar
  by_cases haw : a = w
  · -- Column w: exactly {(w, 0)} on both sides.
    subst haw
    constructor
    · rintro (⟨hne, hac, hlt⟩ | ⟨-, hmem⟩)
      · rw [plinthLift_well] at hlt
        exact ⟨hne, hac, by rw [plinthLift_well]; omega⟩
      · rcases mem_cellsAt.mp hmem with ⟨cell, hcell, h1, -⟩
        exact absurd h1 (havoid cell hcell)
    · rintro ⟨hne, hac, hlt⟩
      rw [plinthLift_well] at hlt
      exact Or.inl ⟨hne, hac, by rw [plinthLift_well]; omega⟩
  · by_cases hrc : r < c + 1
    · -- The slab (rows ≤ c): filled on both sides except the hole point.
      by_cases hhole : a = hx ∧ r = 0
      · obtain ⟨rfl, rfl⟩ := hhole
        constructor
        · rintro (⟨hne, -, -⟩ | ⟨hcr, -⟩)
          · exact absurd rfl (hne (a, 0) rfl)
          · omega
        · rintro ⟨hne, -, -⟩
          exact absurd rfl (hne (a, 0) rfl)
      · constructor
        · rintro (⟨-, hac, -⟩ | ⟨hcr, -⟩)
          · refine ⟨?_, hac, ?_⟩
            · rintro x hx' hcontra
              rw [Option.some.injEq] at hx'
              subst hx'
              rw [Prod.mk.injEq] at hcontra
              exact hhole hcontra
            · rw [plinthLift_ne w c ρ' haw]; omega
          · omega
        · rintro ⟨-, hac, -⟩
          refine Or.inl ⟨?_, hac, ?_⟩
          · rintro x hx' hcontra
            rw [Option.some.injEq] at hx'
            subst hx'
            rw [Prod.mk.injEq] at hcontra
            exact hhole hcontra
          · rw [plinthLift_ne w c ρ haw]; omega
    · -- Above the slab: transfer through the bare representative.
      push_neg at hrc
      have hne_auto : ∀ x, (some ((hx : ℕ), (0 : ℕ)) : Option Coord) = some x →
          ((a : ℕ), (r : ℕ)) ≠ x := by
        rintro x hx' hcontra
        rw [Option.some.injEq] at hx'
        subst hx'
        have h2 := congrArg Prod.snd hcontra
        simp only at h2
        omega
      rw [plinthLift_ne w c ρ haw, plinthLift_ne w c ρ' haw]
      constructor
      · rintro (⟨-, h2, h3⟩ | ⟨-, hmem⟩)
        · have hout := hrep_ar.mp (Or.inl ⟨h2, by omega⟩)
          exact ⟨hne_auto, hout.1, by omega⟩
        · have hout := hrep_ar.mp (Or.inr hmem)
          exact ⟨hne_auto, hout.1, by omega⟩
      · rintro ⟨-, h2, h3⟩
        rcases hrep_ar.mpr ⟨h2, by omega⟩ with ⟨g2, g3⟩ | hmem
        · exact Or.inl ⟨hne_auto, g2, by omega⟩
        · exact Or.inr ⟨hrc, hmem⟩

/-- Membership in a debt board whose hole sits at the floor point `(hx, 0)`,
with the hole condition flattened to component arithmetic. -/
theorem mem_debtBoard_holePoint {cfg : GameConfig} {h : ℕ → ℕ} {hx : ℕ}
    (a r : ℕ) :
    ((a, r) : Coord) ∈ debtBoard cfg h (some ((hx : ℕ), (0 : ℕ)))
      ↔ ¬(a = hx ∧ r = 0) ∧ a < cfg.cols ∧ r < h a := by
  rw [mem_debtBoard]
  constructor
  · rintro ⟨hne, hac, hlt⟩
    refine ⟨?_, hac, hlt⟩
    rintro ⟨h1, h2⟩
    exact hne (hx, 0) rfl (by rw [Prod.mk.injEq]; exact ⟨h1, h2⟩)
  · rintro ⟨hne, hac, hlt⟩
    refine ⟨?_, hac, hlt⟩
    rintro x hx' hcontra
    rw [Option.some.injEq] at hx'
    subst hx'
    rw [Prod.mk.injEq] at hcontra
    exact hne hcontra

/-- The drain lands on the plug: offset 1, filling well rows 1–4. -/
theorem bandDrain_place_plinth {cfg : GameConfig} {w hx c : ℕ} {ρ : ℕ → ℕ}
    (hw : w < cfg.cols) (hhx : hx ≠ w ∧ hx < cfg.cols ∧ 1 ≤ ρ hx) :
    (bandDrain w).place (debtBoard cfg (plinthLift w c ρ) (some (hx, 0)))
      = debtBoard cfg (fun j => if j = w then 5 else ρ j + c + 1)
          (some (hx, 0)) := by
  have hcovL : ∀ x, (some ((hx : ℕ), (0 : ℕ)) : Option Coord) = some x →
      x.2 + 1 < plinthLift w c ρ x.1 := by
    rintro x hx'
    rw [Option.some.injEq] at hx'
    subst hx'
    rw [plinthLift_ne w c ρ hhx.1]
    have := hhx.2.2
    omega
  have hcolw : (debtBoard cfg (plinthLift w c ρ)
      (some ((hx : ℕ), (0 : ℕ)))).colHeight w = 1 := by
    rw [colHeight_debtBoard hcovL, if_pos hw, plinthLift_well]
  have hD : (bandDrain w).dropOffset
      (debtBoard cfg (plinthLift w c ρ) (some ((hx : ℕ), (0 : ℕ)))) = 1 := by
    rw [Placement.dropOffset_eq_sup, bandDrain_shapeUp]
    simp only [Finset.sup_insert, Finset.sup_singleton, Nat.add_zero,
      bandDrain_col, hcolw]
    decide
  have hdr : (bandDrain w).dropped
      (debtBoard cfg (plinthLift w c ρ) (some ((hx : ℕ), (0 : ℕ))))
      = {(w, 1), (w, 2), (w, 3), (w, 4)} := by
    rw [Placement.dropped_eq_image, bandDrain_shapeUp, hD]
    simp only [Finset.image_insert, Finset.image_singleton, bandDrain_col,
      Nat.add_zero]
  rw [Placement.place_eq_union_dropped, hdr]
  ext ⟨a, r⟩
  simp only [Finset.mem_union, mem_debtBoard_holePoint, Finset.mem_insert,
    Finset.mem_singleton, Prod.mk.injEq]
  by_cases haw : a = w
  · subst haw
    constructor
    · rintro (⟨hne, hac, hlt⟩ | hdrop)
      · rw [plinthLift_well] at hlt
        exact ⟨hne, hac, by rw [if_pos rfl]; omega⟩
      · refine ⟨?_, hw, by rw [if_pos rfl]; omega⟩
        rintro ⟨-, hr0⟩
        omega
    · rintro ⟨hne, hac, hlt⟩
      rw [if_pos rfl] at hlt
      by_cases hr0 : r = 0
      · subst hr0
        exact Or.inl ⟨hne, hac, by rw [plinthLift_well]; omega⟩
      · right
        omega
  · constructor
    · rintro (⟨hne, hac, hlt⟩ | hdrop)
      · rw [plinthLift_ne w c ρ haw] at hlt
        exact ⟨hne, hac, by rw [if_neg haw]; omega⟩
      · omega
    · rintro ⟨hne, hac, hlt⟩
      rw [if_neg haw] at hlt
      exact Or.inl ⟨hne, hac, by rw [plinthLift_ne w c ρ haw]; omega⟩

/-- **T2′ — the plinth drain.** At `4 ≤ c` the vertical I fills well rows
1–4; rows 1–4 are full while row 0 never is (the entombed hole): exactly
rows 1–4 clear, the band drops to `c − 4`, plinth and hole intact. -/
theorem drain_debtBoard_plinthLift {cfg : GameConfig} {w hx c : ℕ}
    {ρ : ℕ → ℕ} (hw : w < cfg.cols) (hc : 4 ≤ c)
    (hhx : hx ≠ w ∧ hx < cfg.cols ∧ 1 ≤ ρ hx) :
    (bandDrain w).applyStep cfg
        (debtBoard cfg (plinthLift w c ρ) (some (hx, 0)))
      = debtBoard cfg (plinthLift w (c - 4) ρ) (some (hx, 0)) := by
  rw [Placement.applyStep_eq_clearLines_place, bandDrain_place_plinth hw hhx]
  set B := debtBoard cfg (fun j => if j = w then 5 else ρ j + c + 1)
    (some ((hx : ℕ), (0 : ℕ))) with hBdef
  have hmemB : ∀ a r : ℕ, ((a, r) : Coord) ∈ B
      ↔ ¬(a = hx ∧ r = 0) ∧ a < cfg.cols ∧
        r < (if a = w then 5 else ρ a + c + 1) := by
    intro a r
    rw [hBdef, mem_debtBoard_holePoint]
  have hfull_iff : ∀ r, isFull cfg B r ↔ 1 ≤ r ∧ r ≤ 4 := by
    intro r
    constructor
    · intro hisfull
      constructor
      · by_contra h0
        have hmem := hisfull hx (Finset.mem_range.mpr hhx.2.1)
        rw [hmemB] at hmem
        exact hmem.1 ⟨rfl, by omega⟩
      · by_contra h5
        have hmem := hisfull w (Finset.mem_range.mpr hw)
        rw [hmemB] at hmem
        have hlt := hmem.2.2
        rw [if_pos rfl] at hlt
        omega
    · rintro ⟨h1, h4⟩ col hcol
      rw [Finset.mem_range] at hcol
      refine (hmemB col r).mpr ⟨?_, hcol, ?_⟩
      · rintro ⟨-, rfl⟩
        omega
      · by_cases hcw : col = w
        · rw [if_pos hcw]; omega
        · rw [if_neg hcw]; omega
  have hfullRows : fullRows cfg B = {1, 2, 3, 4} := by
    ext r
    unfold fullRows
    rw [Finset.mem_filter]
    simp only [Finset.mem_insert, Finset.mem_singleton]
    constructor
    · rintro ⟨-, hfull⟩
      have := (hfull_iff r).mp hfull
      omega
    · intro hr
      refine ⟨?_, (hfull_iff r).mpr (by omega)⟩
      rw [Finset.mem_image]
      refine ⟨(w, r), (hmemB w r).mpr ⟨?_, hw, by rw [if_pos rfl]; omega⟩, rfl⟩
      rintro ⟨heq, -⟩
      exact hhx.1 heq.symm
  have hcb0 : clearedBelow cfg B 0 = 0 := by
    unfold clearedBelow
    rw [hfullRows, Finset.filter_false_of_mem, Finset.card_empty]
    intro x hx'
    simp only [Finset.mem_insert, Finset.mem_singleton] at hx'
    omega
  have hcb5 : ∀ r, 5 ≤ r → clearedBelow cfg B r = 4 := by
    intro r hr
    unfold clearedBelow
    rw [hfullRows, Finset.filter_true_of_mem]
    · decide
    · intro x hx'
      simp only [Finset.mem_insert, Finset.mem_singleton] at hx'
      omega
  unfold clearLines
  ext ⟨a, s⟩
  rw [Finset.mem_image, mem_debtBoard_holePoint]
  constructor
  · rintro ⟨⟨a₀, r₀⟩, hp, heq⟩
    rw [Finset.mem_filter] at hp
    obtain ⟨hpB, hnf⟩ := hp
    have hr0 : r₀ = 0 ∨ 5 ≤ r₀ := by
      by_contra hcon
      push_neg at hcon
      exact hnf ((hfull_iff r₀).mpr (by omega))
    have hpm := (hmemB a₀ r₀).mp hpB
    obtain ⟨hne, hac, hlt⟩ := hpm
    rcases hr0 with rfl | hr5
    · have heq' : (a₀, 0 - clearedBelow cfg B 0) = ((a : ℕ), (s : ℕ)) := heq
      rw [hcb0] at heq'
      rw [Prod.mk.injEq] at heq'
      obtain ⟨rfl, hs⟩ := heq'
      refine ⟨?_, hac, ?_⟩
      · rintro ⟨h1, -⟩
        exact hne ⟨h1, rfl⟩
      · by_cases hcw : a₀ = w
        · rw [hcw, plinthLift_well]; omega
        · rw [plinthLift_ne w (c - 4) ρ hcw]; omega
    · have heq' : (a₀, r₀ - clearedBelow cfg B r₀) = ((a : ℕ), (s : ℕ)) := heq
      rw [hcb5 r₀ hr5] at heq'
      rw [Prod.mk.injEq] at heq'
      obtain ⟨rfl, hs⟩ := heq'
      have haw : a₀ ≠ w := by
        intro hcw
        rw [hcw, if_pos rfl] at hlt
        omega
      rw [if_neg haw] at hlt
      refine ⟨?_, hac, ?_⟩
      · rintro ⟨-, h0⟩
        omega
      · rw [plinthLift_ne w (c - 4) ρ haw]
        omega
  · rintro ⟨hne, hac, hlt⟩
    by_cases hs0 : s = 0
    · subst hs0
      refine ⟨(a, 0), ?_, ?_⟩
      · rw [Finset.mem_filter]
        refine ⟨(hmemB a 0).mpr ⟨hne, hac, ?_⟩, ?_⟩
        · by_cases hcw : a = w
          · rw [if_pos hcw]; omega
          · rw [if_neg hcw]; omega
        · intro hisfull
          have := (hfull_iff 0).mp hisfull
          omega
      · show (a, 0 - clearedBelow cfg B 0) = ((a : ℕ), (0 : ℕ))
        rw [hcb0]
    · refine ⟨(a, s + 4), ?_, ?_⟩
      · rw [Finset.mem_filter]
        have haw : a ≠ w := by
          intro hcw
          rw [hcw, plinthLift_well] at hlt
          omega
        rw [plinthLift_ne w (c - 4) ρ haw] at hlt
        refine ⟨(hmemB a (s + 4)).mpr ⟨?_, hac, ?_⟩, ?_⟩
        · rintro ⟨-, h0⟩
          omega
        · rw [if_neg haw]
          omega
        · intro hisfull
          have := (hfull_iff (s + 4)).mp hisfull
          omega
      · show (a, s + 4 - clearedBelow cfg B (s + 4)) = ((a : ℕ), (s : ℕ))
        rw [hcb5 (s + 4) (by omega), Prod.mk.injEq]
        exact ⟨rfl, by omega⟩

end Board
end Tetris
