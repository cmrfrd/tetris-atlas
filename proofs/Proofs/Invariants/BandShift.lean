import Proofs.Invariants.HoledSkyline
import Proofs.Invariants.Confluence

/-!
# Well-anchored band lifts: the translation-quotient transport layer

The translation symmetry of the debt-1 board algebra is *well-anchored*: one
well column is pinned at height `0` (which blocks every line clear) and the
remaining band columns shift up by `c`. `bandLift` lifts a base-0 profile to
band base `c`; `holeLift` rides the (optional) buried cell along. The
transport theorems (`place_debtBoard_bandLift`, `drain_debtBoard_bandLift`)
let a `DebtCertificate`-style closure obligation be proven once at the
representative and reused at every base — see
`Proofs/Safety/ShiftCertificate.lean` for the certificate that packages this.
-/

namespace Tetris

/-- Transport an optional buried cell up by `c` rows. -/
def holeLift (c : ℕ) : Option Coord → Option Coord :=
  Option.map (fun x => (x.1, x.2 + c))

@[simp] theorem holeLift_none (c : ℕ) : holeLift c none = none := rfl

@[simp] theorem holeLift_some (c : ℕ) (x : Coord) :
    holeLift c (some x) = some (x.1, x.2 + c) := rfl

namespace Board

/-- The well-anchored band lift: column `w` pinned at `0`, every other
column raised by `c`. `bandLift w 0 h = h` exactly when `h w = 0`. -/
def bandLift (w c : ℕ) (h : ℕ → ℕ) : ℕ → ℕ :=
  fun j => if j = w then 0 else h j + c

@[simp] theorem bandLift_well (w c : ℕ) (h : ℕ → ℕ) : bandLift w c h w = 0 :=
  if_pos rfl

theorem bandLift_ne (w c : ℕ) (h : ℕ → ℕ) {j : ℕ} (hj : j ≠ w) :
    bandLift w c h j = h j + c := if_neg hj

theorem bandLift_zero {w : ℕ} {h : ℕ → ℕ} (hw : h w = 0) :
    bandLift w 0 h = h := by
  funext j
  by_cases hj : j = w
  · subst hj; simp [bandLift, hw]
  · simp [bandLift, hj]

/-- Uniform membership for debt-≤1 boards, covering both `Option` cases. -/
theorem mem_debtBoard {cfg : GameConfig} {h : ℕ → ℕ} {ho : Option Coord}
    (p : Coord) :
    p ∈ debtBoard cfg h ho
      ↔ (∀ x, ho = some x → p ≠ x) ∧ p.1 < cfg.cols ∧ p.2 < h p.1 := by
  cases ho with
  | none => simp [debtBoard_none, mem_skyline']
  | some x =>
      rw [debtBoard_some, mem_holedSkyline]
      constructor
      · rintro ⟨hne, hc, hr⟩
        exact ⟨fun y hy => by cases hy; exact hne, hc, hr⟩
      · rintro ⟨hne, hc, hr⟩
        exact ⟨hne x rfl, hc, hr⟩

/-- Truncated-subtraction sup shift: raising every value by `c` raises the
sup of `value − row` by exactly `c`, provided some cell sits at row `0`. -/
theorem sup_sub_add_shift (s : Finset Coord) (f : Coord → ℕ) (c : ℕ)
    (hbot : ∃ cell ∈ s, cell.2 = 0) :
    s.sup (fun cell => f cell + c - cell.2)
      = s.sup (fun cell => f cell - cell.2) + c := by
  obtain ⟨c₀, hc₀, hc₀0⟩ := hbot
  refine le_antisymm (Finset.sup_le fun cell hcell => ?_) ?_
  · have hb : f cell - cell.2 ≤ s.sup (fun cell => f cell - cell.2) :=
      Finset.le_sup (f := fun cell => f cell - cell.2) hcell
    omega
  · rcases Finset.exists_mem_eq_sup s ⟨c₀, hc₀⟩ (fun cell => f cell - cell.2)
      with ⟨cm, hcm, heq⟩
    have heq' : s.sup (fun cell => f cell - cell.2) = f cm - cm.2 := heq
    rw [heq']
    by_cases hz : f cm - cm.2 = 0
    · rw [hz]
      have hb : f c₀ + c - c₀.2 ≤ s.sup (fun cell => f cell + c - cell.2) := by
        exact Finset.le_sup (f := fun cell => f cell + c - cell.2) hc₀
      omega
    · have hb : f cm + c - cm.2 ≤ s.sup (fun cell => f cell + c - cell.2) := by
        exact Finset.le_sup (f := fun cell => f cell + c - cell.2) hcm
      omega

/-- **Shift-equivariance of the hard drop on debt-1 boards.** For a
band placement (in-bounds, avoiding the well) the drop offset at band
base `c` is the base-0 offset plus `c`; the strictly covered hole is
invisible on both boards. -/
theorem dropOffset_debtBoard_bandLift {cfg : GameConfig} {w c : ℕ}
    {ρ : ℕ → ℕ} {ho : Option Coord} {pl : Placement}
    (hcols : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 < cfg.cols)
    (havoid : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 ≠ w)
    (hho : ∀ x, ho = some x →
      x.1 ≠ w ∧ x.1 < cfg.cols ∧ x.2 + 1 < ρ x.1) :
    pl.dropOffset (debtBoard cfg (bandLift w c ρ) (holeLift c ho))
      = pl.dropOffset (debtBoard cfg ρ ho) + c := by
  have hcov : ∀ x, ho = some x → x.2 + 1 < ρ x.1 := fun x hx => (hho x hx).2.2
  have hcovL : ∀ x, holeLift c ho = some x → x.2 + 1 < bandLift w c ρ x.1 := by
    rintro x hx
    cases ho with
    | none => simp at hx
    | some x₀ =>
        rw [holeLift_some, Option.some.injEq] at hx
        subst hx
        rw [bandLift_ne w c ρ (hho x₀ rfl).1]
        have := (hho x₀ rfl).2.2
        omega
  have hL : pl.dropOffset (debtBoard cfg (bandLift w c ρ) (holeLift c ho))
      = pl.shapeUp.sup (fun cell => ρ (pl.col + cell.1) + c - cell.2) := by
    rw [Placement.dropOffset_eq_sup]
    refine Finset.sup_congr rfl fun cell hcell => ?_
    rw [colHeight_debtBoard hcovL, if_pos (hcols cell hcell),
      bandLift_ne w c ρ (havoid cell hcell)]
  have hR : pl.dropOffset (debtBoard cfg ρ ho)
      = pl.shapeUp.sup (fun cell => ρ (pl.col + cell.1) - cell.2) := by
    rw [Placement.dropOffset_eq_sup]
    refine Finset.sup_congr rfl fun cell hcell => ?_
    rw [colHeight_debtBoard hcov, if_pos (hcols cell hcell)]
  rw [hL, hR]
  exact sup_sub_add_shift pl.shapeUp (fun cell => ρ (pl.col + cell.1)) c
    (Placement.shapeUp_exists_bottom pl.piece pl.rot)

/-- Landing-cell membership, unfolded to the shape/column data. -/
theorem mem_cellsAt {pl : Placement} {d a r : ℕ} :
    ((a, r) : Coord) ∈ pl.cellsAt d
      ↔ ∃ cell ∈ pl.shapeUp, pl.col + cell.1 = a ∧ d + cell.2 = r := by
  unfold Placement.cellsAt
  simp only [Finset.mem_image, Prod.mk.injEq]

/-- Landing cells at a raised drop offset are the base landing cells,
shifted up: the piece never reaches below the shift. -/
theorem mem_cellsAt_add {pl : Placement} {d c a r : ℕ} :
    ((a, r) : Coord) ∈ pl.cellsAt (d + c)
      ↔ c ≤ r ∧ ((a, r - c) : Coord) ∈ pl.cellsAt d := by
  rw [mem_cellsAt, mem_cellsAt]
  constructor
  · rintro ⟨cell, hcell, h1, h2⟩
    exact ⟨by omega, cell, hcell, h1, by omega⟩
  · rintro ⟨hcr, cell, hcell, h1, h2⟩
    exact ⟨cell, hcell, h1, by omega⟩

/-- **T1 — clear-free transport.** A witnessed placement transition between
debt-1 boards at the base-0 representative transports verbatim to every
band base `c`: the landing cells shift up by `c`, the band bottom slab
(rows `< c`) is filled on both sides, and the hole rides at `+c`. -/
theorem place_debtBoard_bandLift {cfg : GameConfig} {w c : ℕ}
    {ρ ρ' : ℕ → ℕ} {ho ho' : Option Coord} {pl : Placement}
    (hcols : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 < cfg.cols)
    (havoid : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 ≠ w)
    (hho : ∀ x, ho = some x →
      x.1 ≠ w ∧ x.1 < cfg.cols ∧ x.2 + 1 < ρ x.1)
    (hrep : pl.place (debtBoard cfg ρ ho) = debtBoard cfg ρ' ho') :
    pl.place (debtBoard cfg (bandLift w c ρ) (holeLift c ho))
      = debtBoard cfg (bandLift w c ρ') (holeLift c ho') := by
  have hD := dropOffset_debtBoard_bandLift (cfg := cfg) (w := w) (c := c)
    (ρ := ρ) (ho := ho) (pl := pl) hcols havoid hho
  ext ⟨a, r⟩
  simp only [Placement.place_eq_union_dropped, Finset.mem_union,
    Placement.dropped_eq_cellsAt, hD, mem_cellsAt_add, mem_debtBoard]
  have hpt := Finset.ext_iff.mp hrep (a, r - c)
  simp only [Placement.place_eq_union_dropped, Finset.mem_union,
    Placement.dropped_eq_cellsAt, mem_debtBoard] at hpt
  by_cases haw : a = w
  · -- Column w: erased on both lifted sides.
    subst haw
    constructor
    · rintro (⟨-, -, hlt⟩ | ⟨-, hmem⟩)
      · rw [bandLift_well] at hlt; omega
      · rcases mem_cellsAt.mp hmem with ⟨cell, hcell, h1, -⟩
        exact absurd h1 (havoid cell hcell)
    · rintro ⟨-, -, hlt⟩
      rw [bandLift_well] at hlt; omega
  · by_cases hrc : r < c
    · -- Bottom slab: filled on both sides for in-range band columns; the
      -- dropped cells and both lifted holes live at rows ≥ c.
      constructor
      · rintro (⟨-, hac, -⟩ | ⟨hcr, -⟩)
        · refine ⟨?_, hac, ?_⟩
          · rintro x hx
            cases ho' with
            | none => simp at hx
            | some x₀ =>
                rw [holeLift_some, Option.some.injEq] at hx
                subst hx
                intro hcontra
                have h2 := congrArg Prod.snd hcontra
                simp only at h2
                omega
          · rw [bandLift_ne w c ρ' haw]; omega
        · omega
      · rintro ⟨-, hac, -⟩
        refine Or.inl ⟨?_, hac, ?_⟩
        · rintro x hx
          cases ho with
          | none => simp at hx
          | some x₀ =>
              rw [holeLift_some, Option.some.injEq] at hx
              subst hx
              intro hcontra
              have h2 := congrArg Prod.snd hcontra
              simp only at h2
              omega
        · rw [bandLift_ne w c ρ haw]; omega
    · -- Band, above the slab: transfer through the representative at r − c.
      push_neg at hrc
      have hhole : (∀ x, holeLift c ho = some x → ((a, r) : Coord) ≠ x)
          ↔ (∀ x, ho = some x → ((a, r - c) : Coord) ≠ x) := by
        cases ho with
        | none => simp
        | some x₀ =>
            obtain ⟨x1, x2⟩ := x₀
            simp only [holeLift_some, Option.some.injEq, forall_eq', ne_eq,
              Prod.mk.injEq, not_and]
            constructor
            · intro h h1 h2; exact h h1 (by omega)
            · intro h h1 h2; exact h h1 (by omega)
      have hhole' : (∀ x, holeLift c ho' = some x → ((a, r) : Coord) ≠ x)
          ↔ (∀ x, ho' = some x → ((a, r - c) : Coord) ≠ x) := by
        cases ho' with
        | none => simp
        | some x₀ =>
            obtain ⟨x1, x2⟩ := x₀
            simp only [holeLift_some, Option.some.injEq, forall_eq', ne_eq,
              Prod.mk.injEq, not_and]
            constructor
            · intro h h1 h2; exact h h1 (by omega)
            · intro h h1 h2; exact h h1 (by omega)
      rw [bandLift_ne w c ρ haw, bandLift_ne w c ρ' haw, hhole, hhole']
      constructor
      · rintro (⟨h1, h2, h3⟩ | ⟨-, hmem⟩)
        · have hout := hpt.mp (Or.inl ⟨h1, h2, by omega⟩)
          exact ⟨hout.1, hout.2.1, by omega⟩
        · have hout := hpt.mp (Or.inr hmem)
          exact ⟨hout.1, hout.2.1, by omega⟩
      · rintro ⟨h1, h2, h3⟩
        rcases hpt.mpr ⟨h1, h2, by omega⟩ with ⟨g1, g2, g3⟩ | hmem
        · exact Or.inl ⟨g1, g2, by omega⟩
        · exact Or.inr ⟨hrc, hmem⟩

/-- The drain placement: vertical I in column `w`. Mirrors
`Safety/SeamBridge.drainPl`, which lives above this layer; the definitional
bridge is stated beside `ShiftCertificate`. -/
def bandDrain (w : ℕ) : Placement := ⟨Piece.I, 1, w⟩

theorem bandDrain_shapeUp (w : ℕ) :
    (bandDrain w).shapeUp = {((0 : ℕ), (0 : ℕ)), (0, 1), (0, 2), (0, 3)} :=
  shapeUp_vertI' w 1 (by decide)

@[simp] theorem bandDrain_col (w : ℕ) : (bandDrain w).col = w := rfl

/-- The drain drops to the floor of the (empty) well and fills rows 0–3. -/
theorem bandDrain_place {cfg : GameConfig} {w c : ℕ} {ρ : ℕ → ℕ}
    {ho : Option Coord} (hw : w < cfg.cols)
    (hho : ∀ x, ho = some x →
      x.1 ≠ w ∧ x.1 < cfg.cols ∧ x.2 + 1 < ρ x.1) :
    (bandDrain w).place (debtBoard cfg (bandLift w c ρ) (holeLift c ho))
      = debtBoard cfg (fun j => if j = w then 4 else ρ j + c)
          (holeLift c ho) := by
  have hcovL : ∀ x, holeLift c ho = some x → x.2 + 1 < bandLift w c ρ x.1 := by
    rintro x hx
    cases ho with
    | none => simp at hx
    | some x₀ =>
        rw [holeLift_some, Option.some.injEq] at hx
        subst hx
        rw [bandLift_ne w c ρ (hho x₀ rfl).1]
        have := (hho x₀ rfl).2.2
        omega
  have hcolw : (debtBoard cfg (bandLift w c ρ) (holeLift c ho)).colHeight w = 0 := by
    rw [colHeight_debtBoard hcovL, if_pos hw, bandLift_well]
  have hD : (bandDrain w).dropOffset
      (debtBoard cfg (bandLift w c ρ) (holeLift c ho)) = 0 := by
    rw [Placement.dropOffset_eq_sup, bandDrain_shapeUp]
    simp only [Finset.sup_insert, Finset.sup_singleton, Nat.add_zero,
      bandDrain_col, hcolw, Nat.zero_sub, Nat.sub_zero, max_self]
  have hdr : (bandDrain w).dropped
      (debtBoard cfg (bandLift w c ρ) (holeLift c ho))
      = {(w, 0), (w, 1), (w, 2), (w, 3)} := by
    rw [Placement.dropped_eq_image, bandDrain_shapeUp, hD]
    simp only [Finset.image_insert, Finset.image_singleton, bandDrain_col,
      Nat.add_zero, Nat.zero_add]
  rw [Placement.place_eq_union_dropped, hdr]
  ext ⟨a, r⟩
  simp only [Finset.mem_union, mem_debtBoard, Finset.mem_insert,
    Finset.mem_singleton, Prod.mk.injEq]
  constructor
  · rintro (⟨hh, hac, hlt⟩ | hdrop)
    · refine ⟨hh, hac, ?_⟩
      by_cases haw : a = w
      · subst haw; rw [bandLift_well] at hlt; omega
      · rw [bandLift_ne w c ρ haw] at hlt
        rw [if_neg haw]
        omega
    · obtain ⟨rfl, hr4⟩ : a = w ∧ r < 4 := by omega
      refine ⟨?_, hw, by rw [if_pos rfl]; omega⟩
      rintro x hx hcontra
      cases ho with
      | none => simp at hx
      | some x₀ =>
          rw [holeLift_some, Option.some.injEq] at hx
          subst hx
          have h1 := congrArg Prod.fst hcontra
          simp only at h1
          exact (hho x₀ rfl).1 h1.symm
  · rintro ⟨hh, hac, hlt⟩
    by_cases haw : a = w
    · subst haw
      rw [if_pos rfl] at hlt
      right
      omega
    · rw [if_neg haw] at hlt
      exact Or.inl ⟨hh, hac, by rw [bandLift_ne w c ρ haw]; omega⟩

/-- **T2 — the generic drain.** At any band base `c ≥ 4`, dropping the
vertical I into the well clears exactly rows 0–3: the pattern and the
rep-hole are unchanged and the base drops by 4. -/
theorem drain_debtBoard_bandLift {cfg : GameConfig} {w c : ℕ} {ρ : ℕ → ℕ}
    {ho : Option Coord} (hw : w < cfg.cols) (hc : 4 ≤ c)
    (hho : ∀ x, ho = some x →
      x.1 ≠ w ∧ x.1 < cfg.cols ∧ x.2 + 1 < ρ x.1) :
    (bandDrain w).applyStep cfg
        (debtBoard cfg (bandLift w c ρ) (holeLift c ho))
      = debtBoard cfg (bandLift w (c - 4) ρ) (holeLift (c - 4) ho) := by
  rw [Placement.applyStep_eq_clearLines_place, bandDrain_place hw hho]
  have hm : ∀ j < cfg.cols, 4 ≤ (fun j => if j = w then 4 else ρ j + c) j := by
    intro j hj
    by_cases hjw : j = w
    · simp [hjw]
    · simp only [if_neg hjw]
      omega
  have hm0 : ∃ j < cfg.cols, (fun j => if j = w then 4 else ρ j + c) j = 4 :=
    ⟨w, hw, by simp⟩
  cases ho with
  | none =>
      simp only [holeLift_none, debtBoard_none]
      rw [clearLines_skyline hm hm0]
      congr 1
      funext j
      by_cases hjw : j = w
      · subst hjw; simp [bandLift]
      · simp only [bandLift, if_neg hjw]
        omega
  | some x =>
      obtain ⟨hxw, hxc, hxcov⟩ := hho x rfl
      simp only [holeLift_some, debtBoard_some]
      rw [clearLines_holedSkyline_of_le (x := (x.1, x.2 + c)) hxc hm hm0
        (hc.trans (Nat.le_add_left c x.2))]
      congr 1
      · funext j
        by_cases hjw : j = w
        · subst hjw; simp [bandLift]
        · simp only [bandLift, if_neg hjw]
          omega
      · exact Prod.ext rfl (by omega)

end Board
end Tetris
