import Mathlib
import Proofs.Safety.HorizonCompactness

/-!
# Headroom-graded safety: no kill certificate of depth five exists

The compactness program (`solvable_iff_forall_horizon`) reduces solvability to
"for every `N`, no adversarial kill-tree of depth `N` defeats the player". This
file climbs the first five rungs of that ladder, with a theorem that grades
safety by *headroom*:

* **Four rows of headroom buy one guaranteed step, unconditionally**
  (`mem_safeIterate_of_headroom`): a state whose every cell has `4k` rows of
  clearance above it lies in `safeIterate k` — any valid placement works, no
  strategy required, because a hard drop raises the stack by at most four rows
  and line clears only lower it.
* The empty board has twenty rows of headroom, so
  (`init_mem_safeIterate_five`): **`init ∈ safeIterate 5` — no adversarial kill
  certificate of depth ≤ 5 exists**, and this is certified with zero search.

The grading is tight for strategy-free play: five vertical pieces stacked in
one column reach height 20, and a sixth tops out — so depth six is exactly
where survival first *requires* a decision. Every rung beyond five must engage
strategy, and the kill-certificate refutation program starts there.
-/

namespace Tetris

/-- On a wide-enough board every piece admits a valid placement: rotation `0`
at column `0` fits inside the four-column bounding box. -/
theorem exists_valid_placement_of_cols {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    (p : Piece) : ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg := by
  refine ⟨⟨p, 0, 0⟩, rfl, ?_⟩
  intro cell hcell
  have h4 := Piece.shapeUp_col_lt_four p 0 cell hcell
  dsimp only
  omega

/-- Column heights are bounded by any uniform strict row bound on the cells. -/
theorem colHeight_le_of_forall_row_lt {b : Board} {H : ℕ}
    (hb : ∀ q ∈ b, q.2 < H) (j : ℕ) : b.colHeight j ≤ H := by
  refine Finset.sup_le fun r hr => ?_
  unfold Board.colRows at hr
  rw [Finset.mem_image] at hr
  obtain ⟨q, hq, rfl⟩ := hr
  rw [Finset.mem_filter] at hq
  exact hb q hq.1

/-- A hard drop lands within four rows of the stack top: every dropped cell
sits strictly below `H + 4` whenever the board's cells sit strictly below `H`. -/
theorem dropped_row_lt {b : Board} {H : ℕ} (hb : ∀ q ∈ b, q.2 < H)
    (pl : Placement) {q : Coord} (hq : q ∈ pl.dropped b) : q.2 < H + 4 := by
  unfold Placement.dropped Placement.cellsAt at hq
  rw [Finset.mem_image] at hq
  obtain ⟨cell, hcell, rfl⟩ := hq
  have hcell4 : cell.2 < 4 := Piece.shapeUp_row_lt_four _ _ cell hcell
  have hoff : pl.dropOffset b ≤ H := by
    rw [Placement.dropOffset_eq_sup]
    refine Finset.sup_le fun c _ => ?_
    have := colHeight_le_of_forall_row_lt hb (pl.col + c.1)
    omega
  dsimp only
  omega

/-- Line clears never raise a cell: every surviving cell sits at or below the
row of some pre-clear cell. -/
theorem clearLines_row_le {cfg : GameConfig} {b : Board} {q : Coord}
    (hq : q ∈ Board.clearLines cfg b) : ∃ q' ∈ b, q.2 ≤ q'.2 := by
  unfold Board.clearLines at hq
  rw [Finset.mem_image] at hq
  obtain ⟨p, hp, rfl⟩ := hq
  rw [Finset.mem_filter] at hp
  exact ⟨p, hp.1, by dsimp only; omega⟩

/-- **One full move raises the stack by at most four rows**: cells of
`applyStep` sit strictly below `H + 4` whenever the board's cells sit strictly
below `H` — for every placement, valid or not. -/
theorem applyStep_row_lt {cfg : GameConfig} {b : Board} {H : ℕ}
    (hb : ∀ q ∈ b, q.2 < H) (pl : Placement) {q : Coord}
    (hq : q ∈ Placement.applyStep cfg b pl) : q.2 < H + 4 := by
  unfold Placement.applyStep at hq
  obtain ⟨q', hq', hle⟩ := clearLines_row_le hq
  unfold Placement.place at hq'
  rcases Finset.mem_union.mp hq' with h | h
  · have := hb q' h
    omega
  · have := dropped_row_lt hb pl h
    omega

/-- **Headroom-graded safety.** A state whose every cell has `4k` rows of
clearance lies in the `k`-th safety iterate: with four rows of headroom per
step, *any* valid placement survives — the adversary's piece choice is
irrelevant and no strategy is needed. -/
theorem mem_safeIterate_of_headroom {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) :
    ∀ k, 4 * k ≤ cfg.rows → ∀ g : GameState,
      (∀ q ∈ g.board, q.2 + 4 * k ≤ cfg.rows) → g ∈ safeIterate cfg k := by
  intro k
  induction k with
  | zero =>
    intro _ g _
    rw [safeIterate_zero]
    trivial
  | succ k ih =>
    intro hR g hg
    rw [safeIterate_succ]
    refine ⟨?_, ?_⟩
    · rw [GameState.not_lost_iff_forall_row_lt]
      intro q hq
      have := hg q hq
      omega
    · intro p hp
      obtain ⟨pl, hpiece, hvalid⟩ := exists_valid_placement_of_cols hcols p
      refine ⟨pl, hpiece, hvalid, ih (by omega) _ ?_⟩
      intro q hq
      rw [adversarialStep_board] at hq
      have hb : ∀ q' ∈ g.board, q'.2 < cfg.rows - (4 * k + 3) := by
        intro q' hq'
        have := hg q' hq'
        omega
      have hstep := applyStep_row_lt hb _ hq
      omega

/-! ## Doubling the horizon: the player picks the rotation

The four-rows-per-step grading charges for the worst rotation (vertical I).
But the placement is the *player's* choice, and **rotation `0` of every piece
is at most two rows tall** — so flat play buys a guaranteed step for only two
rows of headroom, and the empty board's twenty rows certify ten steps. -/

/-- Rotation `0` of every piece has a drop profile at most two rows tall. -/
theorem shapeUp_row_lt_two_rot_zero :
    ∀ p : Piece, ∀ cell ∈ p.shapeUp 0, cell.2 < 2 := by
  decide

/-- Profile-height-graded drop bound: a piece whose profile is under `h` rows
tall lands strictly below `H + h`. -/
theorem dropped_row_lt' {b : Board} {H h : ℕ} (hb : ∀ q ∈ b, q.2 < H)
    {pl : Placement} (hsh : ∀ cell ∈ pl.shapeUp, cell.2 < h)
    {q : Coord} (hq : q ∈ pl.dropped b) : q.2 < H + h := by
  unfold Placement.dropped Placement.cellsAt at hq
  rw [Finset.mem_image] at hq
  obtain ⟨cell, hcell, rfl⟩ := hq
  have hcellh : cell.2 < h := hsh cell hcell
  have hoff : pl.dropOffset b ≤ H := by
    rw [Placement.dropOffset_eq_sup]
    refine Finset.sup_le fun c _ => ?_
    have := colHeight_le_of_forall_row_lt hb (pl.col + c.1)
    omega
  dsimp only
  omega

/-- Profile-height-graded move bound. -/
theorem applyStep_row_lt' {cfg : GameConfig} {b : Board} {H h : ℕ}
    (hb : ∀ q ∈ b, q.2 < H) {pl : Placement}
    (hsh : ∀ cell ∈ pl.shapeUp, cell.2 < h) {q : Coord}
    (hq : q ∈ Placement.applyStep cfg b pl) : q.2 < H + h := by
  unfold Placement.applyStep at hq
  obtain ⟨q', hq', hle⟩ := clearLines_row_le hq
  unfold Placement.place at hq'
  rcases Finset.mem_union.mp hq' with hmem | hmem
  · have := hb q' hmem
    omega
  · have := dropped_row_lt' hb hsh hmem
    omega

/-- **Flat-play headroom grading: two rows per step.** Playing every piece in
its flat rotation, `2k` rows of clearance certify `k` safe steps. -/
theorem mem_safeIterate_of_flat_headroom {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) :
    ∀ k, 2 * k ≤ cfg.rows → ∀ g : GameState,
      (∀ q ∈ g.board, q.2 + 2 * k ≤ cfg.rows) → g ∈ safeIterate cfg k := by
  intro k
  induction k with
  | zero =>
    intro _ g _
    rw [safeIterate_zero]
    trivial
  | succ k ih =>
    intro hR g hg
    rw [safeIterate_succ]
    refine ⟨?_, ?_⟩
    · rw [GameState.not_lost_iff_forall_row_lt]
      intro q hq
      have := hg q hq
      omega
    · intro p hp
      refine ⟨⟨p, 0, 0⟩, rfl, ?_, ih (by omega) _ ?_⟩
      · intro cell hcell
        have h4 := Piece.shapeUp_col_lt_four p 0 cell hcell
        dsimp only
        omega
      · intro q hq
        rw [adversarialStep_board] at hq
        have hb : ∀ q' ∈ g.board, q'.2 < cfg.rows - (2 * k + 1) := by
          intro q' hq'
          have := hg q' hq'
          omega
        have hstep := applyStep_row_lt' hb
          (pl := ⟨p, 0, 0⟩) (h := 2) (shapeUp_row_lt_two_rot_zero p) hq
        omega

/-- The empty board's twenty rows of headroom certify five safe steps. -/
theorem init_mem_safeIterate_of_le {k : ℕ} (hk : k ≤ 5) :
    GameState.init ∈ safeIterate GameConfig.standard k := by
  refine mem_safeIterate_of_headroom (by norm_num) k ?_ GameState.init ?_
  · rw [GameConfig.standard_rows]
    omega
  · intro q hq
    exact absurd hq (GameState.init_board_no_mem q)

/-- **No adversarial kill certificate of depth five exists.** The first five
rungs of the compactness ladder (`solvable_iff_forall_horizon`), certified with
zero search. Tight for strategy-free play: five vertical pieces in one column
reach height 20 and a sixth tops out, so depth six is exactly where survival
first requires a decision. -/
theorem init_mem_safeIterate_five :
    GameState.init ∈ safeIterate GameConfig.standard 5 :=
  init_mem_safeIterate_of_le le_rfl

/-- **No adversarial kill certificate of depth ten exists.** Flat play doubles
the strategy-free horizon: rotation `0` of every piece is at most two rows
tall, so ten pieces fit in twenty rows. This is the flat-play ceiling — the
stack never clears (four columns cannot fill a ten-wide row), so rung eleven
of the compactness ladder is the first that requires clearing or spreading. -/
theorem init_mem_safeIterate_ten :
    GameState.init ∈ safeIterate GameConfig.standard 10 := by
  refine mem_safeIterate_of_flat_headroom (by norm_num) 10 ?_ GameState.init ?_
  · rw [GameConfig.standard_rows]
  · intro q hq
    exact absurd hq (GameState.init_board_no_mem q)

/-- Monotone form: every depth up to ten is certified. -/
theorem init_mem_safeIterate_of_le_ten {k : ℕ} (hk : k ≤ 10) :
    GameState.init ∈ safeIterate GameConfig.standard k :=
  safeIterate_antitone GameConfig.standard hk init_mem_safeIterate_ten

/-! ## The two-group schedule: depth twenty

One column group fills in ten flat placements. But a ten-wide board holds two
disjoint four-wide groups (columns 0–3 and 4–7), and a solver that always
plays flat into the *shorter* group shares the twenty rows across both: forty
rows of combined budget, two per placement — **twenty guaranteed steps**. The
first certificate on the ladder that requires an actual decision (which group),
though still no lookahead. -/

/-- Column-local height bound: a column whose cells sit strictly below `H` has
height at most `H`. -/
theorem colHeight_le_of_col_row_lt {b : Board} {j H : ℕ}
    (hb : ∀ q ∈ b, q.1 = j → q.2 < H) : b.colHeight j ≤ H := by
  refine Finset.sup_le fun r hr => ?_
  unfold Board.colRows at hr
  rw [Finset.mem_image] at hr
  obtain ⟨q, hq, rfl⟩ := hr
  rw [Finset.mem_filter] at hq
  exact hb q hq.1 hq.2

/-- Line clears preserve columns and never raise cells. -/
theorem clearLines_col_row_le {cfg : GameConfig} {b : Board} {q : Coord}
    (hq : q ∈ Board.clearLines cfg b) : ∃ q' ∈ b, q.1 = q'.1 ∧ q.2 ≤ q'.2 := by
  unfold Board.clearLines at hq
  rw [Finset.mem_image] at hq
  obtain ⟨p, hp, rfl⟩ := hq
  rw [Finset.mem_filter] at hp
  exact ⟨p, hp.1, rfl, by dsimp only; omega⟩

/-- **Group-local move bound.** Playing a flat piece at column base `c₀` (its
cells spanning columns `[c₀, c₀+4)`): every post-move cell either descends from
an old cell in its own column, or is a new cell in `[c₀, c₀+4)` strictly below
`H + 2`, where `H` bounds the rows of the group's own columns. -/
theorem applyStep_group_bound {cfg : GameConfig} {b : Board} {c₀ H : ℕ}
    {p : Piece}
    (hb : ∀ q ∈ b, c₀ ≤ q.1 → q.1 < c₀ + 4 → q.2 < H) {q : Coord}
    (hq : q ∈ Placement.applyStep cfg b (⟨p, 0, c₀⟩ : Placement)) :
    (∃ q' ∈ b, q.1 = q'.1 ∧ q.2 ≤ q'.2)
      ∨ (c₀ ≤ q.1 ∧ q.1 < c₀ + 4 ∧ q.2 < H + 2) := by
  unfold Placement.applyStep at hq
  obtain ⟨q', hq', hcol, hle⟩ := clearLines_col_row_le hq
  unfold Placement.place at hq'
  rcases Finset.mem_union.mp hq' with hmem | hmem
  · exact Or.inl ⟨q', hmem, hcol, hle⟩
  · right
    unfold Placement.dropped Placement.cellsAt at hmem
    rw [Finset.mem_image] at hmem
    obtain ⟨cell, hcell, hEq⟩ := hmem
    have hcol4 : cell.1 < 4 := Piece.shapeUp_col_lt_four p 0 cell hcell
    have hrow2 : cell.2 < 2 := shapeUp_row_lt_two_rot_zero p cell hcell
    have hoff : (⟨p, 0, c₀⟩ : Placement).dropOffset b ≤ H := by
      rw [Placement.dropOffset_eq_sup]
      refine Finset.sup_le fun c hc => ?_
      have hc4 : c.1 < 4 := Piece.shapeUp_col_lt_four p 0 c hc
      have hch : b.colHeight (c₀ + c.1) ≤ H := by
        refine colHeight_le_of_col_row_lt fun q'' hq'' hcol'' => ?_
        exact hb q'' hq'' (by omega) (by omega)
      dsimp only
      omega
    have hq1 : q'.1 = c₀ + cell.1 := (congrArg Prod.fst hEq).symm
    have hq2 : q'.2 = (⟨p, 0, c₀⟩ : Placement).dropOffset b + cell.2 :=
      (congrArg Prod.snd hEq).symm
    refine ⟨by omega, by omega, by omega⟩

/-- **The two-group flat schedule.** Even per-group budgets `HA, HB ≤ rows`
with `HA + HB + 2k ≤ 2·rows` certify `k` steps: always play flat into the
group with the smaller budget. Evenness plus even `rows` closes the parity gap
(`min` even and below even `rows` leaves two rows of clearance). -/
theorem mem_safeIterate_of_two_group {cfg : GameConfig} (hcols : 8 ≤ cfg.cols)
    (hre : 2 ∣ cfg.rows) :
    ∀ k HA HB, 2 ∣ HA → 2 ∣ HB → HA ≤ cfg.rows → HB ≤ cfg.rows →
      HA + HB + 2 * k ≤ 2 * cfg.rows →
      ∀ g : GameState,
        (∀ q ∈ g.board, q.1 < 8) →
        (∀ q ∈ g.board, q.1 < 4 → q.2 < HA) →
        (∀ q ∈ g.board, 4 ≤ q.1 → q.2 < HB) →
        g ∈ safeIterate cfg k := by
  intro k
  induction k with
  | zero =>
    intro _ _ _ _ _ _ _ g _ _ _
    rw [safeIterate_zero]
    trivial
  | succ k ih =>
    intro HA HB hA2 hB2 hAr hBr hsum g hcols8 hgA hgB
    rw [safeIterate_succ]
    have hnl : ¬ g.lost cfg := by
      rw [GameState.not_lost_iff_forall_row_lt]
      intro q hq
      by_cases hq4 : q.1 < 4
      · exact lt_of_lt_of_le (hgA q hq hq4) hAr
      · exact lt_of_lt_of_le (hgB q hq (by omega)) hBr
    refine ⟨hnl, ?_⟩
    intro p hp
    -- play into the group with the smaller budget
    rcases le_total HA HB with hmin | hmin
    · -- group A (columns 0–3); its new budget HA + 2 fits under even rows
      have hfit : HA + 2 ≤ cfg.rows := by
        obtain ⟨a, rfl⟩ := hA2
        obtain ⟨r, hr⟩ := hre
        omega
      refine ⟨⟨p, 0, 0⟩, rfl, ?_, ?_⟩
      · intro cell hcell
        have h4 := Piece.shapeUp_col_lt_four p 0 cell hcell
        dsimp only
        omega
      · have hpost : ∀ q ∈ (adversarialStep cfg g p (⟨p, 0, 0⟩ : Placement)).board,
            q.1 < 8 ∧ (q.1 < 4 → q.2 < HA + 2) ∧ (4 ≤ q.1 → q.2 < HB) := by
          intro q hq
          rw [adversarialStep_board] at hq
          dsimp only at hq
          rcases applyStep_group_bound (c₀ := 0) (H := HA)
              (fun q' hq' _ h4 => hgA q' hq' (by omega)) hq with
            ⟨q', hq', hcol, hle⟩ | ⟨h0, h4, hlt⟩
          · refine ⟨by have := hcols8 q' hq'; omega, ?_, ?_⟩
            · intro hq4
              have := hgA q' hq' (by omega)
              omega
            · intro hq4
              have := hgB q' hq' (by omega)
              omega
          · exact ⟨by omega, fun _ => by omega, fun h => by omega⟩
        exact ih (HA + 2) HB (by omega) hB2 hfit hBr (by omega) _
          (fun q hq => (hpost q hq).1)
          (fun q hq => (hpost q hq).2.1)
          (fun q hq => (hpost q hq).2.2)
    · -- group B (columns 4–7)
      have hfit : HB + 2 ≤ cfg.rows := by
        obtain ⟨b', rfl⟩ := hB2
        obtain ⟨r, hr⟩ := hre
        omega
      refine ⟨⟨p, 0, 4⟩, rfl, ?_, ?_⟩
      · intro cell hcell
        have h4 := Piece.shapeUp_col_lt_four p 0 cell hcell
        dsimp only
        omega
      · have hpost : ∀ q ∈ (adversarialStep cfg g p (⟨p, 0, 4⟩ : Placement)).board,
            q.1 < 8 ∧ (q.1 < 4 → q.2 < HA) ∧ (4 ≤ q.1 → q.2 < HB + 2) := by
          intro q hq
          rw [adversarialStep_board] at hq
          dsimp only at hq
          rcases applyStep_group_bound (c₀ := 4) (H := HB)
              (fun q' hq' h4 _ => hgB q' hq' h4) hq with
            ⟨q', hq', hcol, hle⟩ | ⟨h0, h4, hlt⟩
          · refine ⟨by have := hcols8 q' hq'; omega, ?_, ?_⟩
            · intro hq4
              have := hgA q' hq' (by omega)
              omega
            · intro hq4
              have := hgB q' hq' (by omega)
              omega
          · exact ⟨by omega, fun h => by omega, fun _ => by omega⟩
        exact ih HA (HB + 2) hA2 (by omega) hAr hfit (by omega) _
          (fun q hq => (hpost q hq).1)
          (fun q hq => (hpost q hq).2.1)
          (fun q hq => (hpost q hq).2.2)

/-- **No adversarial kill certificate of depth twenty exists.** The two-group
flat schedule shares the board's twenty rows across two four-wide column
groups — forty rows of budget at two rows per placement. The first certified
rung that requires a decision (which group is shorter), though still zero
lookahead. -/
theorem init_mem_safeIterate_twenty :
    GameState.init ∈ safeIterate GameConfig.standard 20 := by
  refine mem_safeIterate_of_two_group (by norm_num) (by norm_num)
    20 0 0 ⟨0, rfl⟩ ⟨0, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    GameState.init ?_ ?_ ?_ <;>
    exact fun q hq => absurd hq (GameState.init_board_no_mem q)

/-- Monotone form: every depth up to twenty is certified. -/
theorem init_mem_safeIterate_of_le_twenty {k : ℕ} (hk : k ≤ 20) :
    GameState.init ∈ safeIterate GameConfig.standard k :=
  safeIterate_antitone GameConfig.standard hk init_mem_safeIterate_twenty

/-! ## The third group: depth twenty-five

Columns 8–9 are still free, and **every piece has a rotation at most two
columns wide** (verticals; O is already 2×2, I stands 1×4). A third stack on
the two spare columns costs at most four rows per placement (the standing I),
buying five more steps: static quotas of `10 + 10 + 5` placements exhaust
`20 + 20 + 20` rows of budget — **depth 25**. The static-quota form is also
simpler than the two-group minimum-selection (no parity bookkeeping) and
subsumes it. -/

/-- A narrowest rotation for each piece: at most two columns wide, at most
four rows tall. -/
def narrowRot : Piece → Rotation
  | Piece.O => 0
  | Piece.I => 1
  | Piece.S => 1
  | Piece.Z => 1
  | Piece.T => 3
  | Piece.L => 1
  | Piece.J => 1

/-- The narrow rotation's profile fits in a `2 × 4` box. -/
theorem shapeUp_narrowRot_bounds :
    ∀ p : Piece, ∀ cell ∈ p.shapeUp (narrowRot p), cell.1 < 2 ∧ cell.2 < 4 := by
  decide

/-- **Parametric group-local move bound.** A piece whose chosen rotation fits a
`w × h` box, played at column base `c₀`: every post-move cell either descends
within its own column or is new in `[c₀, c₀+w)` strictly below `H + h`. -/
theorem applyStep_group_bound' {cfg : GameConfig} {b : Board} {c₀ H w h : ℕ}
    {p : Piece} {r : Rotation}
    (hw : ∀ cell ∈ p.shapeUp r, cell.1 < w)
    (hh : ∀ cell ∈ p.shapeUp r, cell.2 < h)
    (hb : ∀ q ∈ b, c₀ ≤ q.1 → q.1 < c₀ + w → q.2 < H) {q : Coord}
    (hq : q ∈ Placement.applyStep cfg b (⟨p, r, c₀⟩ : Placement)) :
    (∃ q' ∈ b, q.1 = q'.1 ∧ q.2 ≤ q'.2)
      ∨ (c₀ ≤ q.1 ∧ q.1 < c₀ + w ∧ q.2 < H + h) := by
  unfold Placement.applyStep at hq
  obtain ⟨q', hq', hcol, hle⟩ := clearLines_col_row_le hq
  unfold Placement.place at hq'
  rcases Finset.mem_union.mp hq' with hmem | hmem
  · exact Or.inl ⟨q', hmem, hcol, hle⟩
  · right
    unfold Placement.dropped Placement.cellsAt at hmem
    rw [Finset.mem_image] at hmem
    obtain ⟨cell, hcell, hEq⟩ := hmem
    have hcolw : cell.1 < w := hw cell hcell
    have hrowh : cell.2 < h := hh cell hcell
    have hoff : (⟨p, r, c₀⟩ : Placement).dropOffset b ≤ H := by
      rw [Placement.dropOffset_eq_sup]
      refine Finset.sup_le fun c hc => ?_
      have hcw : c.1 < w := hw c hc
      have hch : b.colHeight (c₀ + c.1) ≤ H := by
        refine colHeight_le_of_col_row_lt fun q'' hq'' hcol'' => ?_
        exact hb q'' hq'' (by omega) (by omega)
      dsimp only
      omega
    have hq1 : q'.1 = c₀ + cell.1 := (congrArg Prod.fst hEq).symm
    have hq2 : q'.2 = (⟨p, r, c₀⟩ : Placement).dropOffset b + cell.2 :=
      (congrArg Prod.snd hEq).symm
    exact ⟨by omega, by omega, by omega⟩

/-- **The three-group static-quota schedule.** Quotas `kA, kB, kC` for the
column groups `[0,4)`, `[4,8)`, `[8,·)` with per-group budget constraints
certify `kA + kB + kC` steps: exhaust group A flat, then group B flat, then
group C in narrow rotations. -/
theorem mem_safeIterate_of_three_group {cfg : GameConfig} (hcols : 10 ≤ cfg.cols) :
    ∀ n kA kB kC HA HB HC, kA + kB + kC = n →
      HA + 2 * kA ≤ cfg.rows → HB + 2 * kB ≤ cfg.rows →
      HC + 4 * kC ≤ cfg.rows →
      ∀ g : GameState,
        (∀ q ∈ g.board, q.1 < 4 → q.2 < HA) →
        (∀ q ∈ g.board, 4 ≤ q.1 → q.1 < 8 → q.2 < HB) →
        (∀ q ∈ g.board, 8 ≤ q.1 → q.2 < HC) →
        g ∈ safeIterate cfg n := by
  intro n
  induction n with
  | zero =>
    intro _ _ _ _ _ _ _ _ _ _ g _ _ _
    rw [safeIterate_zero]
    trivial
  | succ n ih =>
    intro kA kB kC HA HB HC hsum hA hB hC g hgA hgB hgC
    rw [safeIterate_succ]
    have hnl : ¬ g.lost cfg := by
      rw [GameState.not_lost_iff_forall_row_lt]
      intro q hq
      by_cases h4 : q.1 < 4
      · have := hgA q hq h4
        omega
      · by_cases h8 : q.1 < 8
        · have := hgB q hq (by omega) h8
          omega
        · have := hgC q hq (by omega)
          omega
    refine ⟨hnl, ?_⟩
    intro p hp
    rcases Nat.eq_zero_or_pos kA with hkA | hkA
    · rcases Nat.eq_zero_or_pos kB with hkB | hkB
      · -- group C: narrow rotation at column 8
        have hkC : 0 < kC := by omega
        refine ⟨⟨p, narrowRot p, 8⟩, rfl, ?_, ?_⟩
        · intro cell hcell
          have h2 := (shapeUp_narrowRot_bounds p cell hcell).1
          dsimp only
          omega
        · have hpost : ∀ q ∈ (adversarialStep cfg g p
              (⟨p, narrowRot p, 8⟩ : Placement)).board,
              (q.1 < 4 → q.2 < HA) ∧ (4 ≤ q.1 → q.1 < 8 → q.2 < HB)
                ∧ (8 ≤ q.1 → q.2 < HC + 4) := by
            intro q hq
            rw [adversarialStep_board] at hq
            dsimp only at hq
            rcases applyStep_group_bound' (c₀ := 8) (H := HC) (w := 2) (h := 4)
                (fun c hc => (shapeUp_narrowRot_bounds p c hc).1)
                (fun c hc => (shapeUp_narrowRot_bounds p c hc).2)
                (fun q' hq' h8 _ => hgC q' hq' h8) hq with
              ⟨q', hq', hcol, hle⟩ | ⟨h8, hw, hlt⟩
            · refine ⟨?_, ?_, ?_⟩
              · intro h4
                have := hgA q' hq' (by omega)
                omega
              · intro h4 h8
                have := hgB q' hq' (by omega) (by omega)
                omega
              · intro h8
                have := hgC q' hq' (by omega)
                omega
            · exact ⟨fun h => by omega, fun h _ => by omega, fun _ => by omega⟩
          exact ih kA kB (kC - 1) HA HB (HC + 4) (by omega) hA hB (by omega) _
            (fun q hq => (hpost q hq).1)
            (fun q hq => (hpost q hq).2.1)
            (fun q hq => (hpost q hq).2.2)
      · -- group B: flat at column 4
        refine ⟨⟨p, 0, 4⟩, rfl, ?_, ?_⟩
        · intro cell hcell
          have h4 := Piece.shapeUp_col_lt_four p 0 cell hcell
          dsimp only
          omega
        · have hpost : ∀ q ∈ (adversarialStep cfg g p
              (⟨p, 0, 4⟩ : Placement)).board,
              (q.1 < 4 → q.2 < HA) ∧ (4 ≤ q.1 → q.1 < 8 → q.2 < HB + 2)
                ∧ (8 ≤ q.1 → q.2 < HC) := by
            intro q hq
            rw [adversarialStep_board] at hq
            dsimp only at hq
            rcases applyStep_group_bound' (c₀ := 4) (H := HB) (w := 4) (h := 2)
                (fun c hc => Piece.shapeUp_col_lt_four p 0 c hc)
                (fun c hc => shapeUp_row_lt_two_rot_zero p c hc)
                (fun q' hq' h4 h8 => hgB q' hq' h4 h8) hq with
              ⟨q', hq', hcol, hle⟩ | ⟨h4, h8, hlt⟩
            · refine ⟨?_, ?_, ?_⟩
              · intro h4
                have := hgA q' hq' (by omega)
                omega
              · intro h4 h8
                have := hgB q' hq' (by omega) (by omega)
                omega
              · intro h8
                have := hgC q' hq' (by omega)
                omega
            · exact ⟨fun h => by omega, fun _ _ => by omega, fun h => by omega⟩
          exact ih kA (kB - 1) kC HA (HB + 2) HC (by omega) hA (by omega) hC _
            (fun q hq => (hpost q hq).1)
            (fun q hq => (hpost q hq).2.1)
            (fun q hq => (hpost q hq).2.2)
    · -- group A: flat at column 0
      refine ⟨⟨p, 0, 0⟩, rfl, ?_, ?_⟩
      · intro cell hcell
        have h4 := Piece.shapeUp_col_lt_four p 0 cell hcell
        dsimp only
        omega
      · have hpost : ∀ q ∈ (adversarialStep cfg g p
            (⟨p, 0, 0⟩ : Placement)).board,
            (q.1 < 4 → q.2 < HA + 2) ∧ (4 ≤ q.1 → q.1 < 8 → q.2 < HB)
              ∧ (8 ≤ q.1 → q.2 < HC) := by
          intro q hq
          rw [adversarialStep_board] at hq
          dsimp only at hq
          rcases applyStep_group_bound' (c₀ := 0) (H := HA) (w := 4) (h := 2)
              (fun c hc => Piece.shapeUp_col_lt_four p 0 c hc)
              (fun c hc => shapeUp_row_lt_two_rot_zero p c hc)
              (fun q' hq' _ h4 => hgA q' hq' (by omega)) hq with
            ⟨q', hq', hcol, hle⟩ | ⟨h0, h4, hlt⟩
          · refine ⟨?_, ?_, ?_⟩
            · intro h4
              have := hgA q' hq' (by omega)
              omega
            · intro h4 h8
              have := hgB q' hq' (by omega) (by omega)
              omega
            · intro h8
              have := hgC q' hq' (by omega)
              omega
          · exact ⟨fun _ => by omega, fun h _ => by omega, fun h => by omega⟩
        exact ih (kA - 1) kB kC (HA + 2) HB HC (by omega) (by omega) hB hC _
          (fun q hq => (hpost q hq).1)
          (fun q hq => (hpost q hq).2.1)
          (fun q hq => (hpost q hq).2.2)

/-- **No adversarial kill certificate of depth twenty-five exists.** Quotas
`10 + 10 + 5` across the three column groups exhaust sixty rows of budget at
`2 + 2 + 4` rows per placement. This is the ceiling of the disjoint-group
schedule family (all ten columns in use, worst-case per-piece charge); the
absolute clear-free ceiling is 50 by mass (`clear_free_le_fifty` in
`ClearRecurrence`), so rungs 26–50 need tighter packing and rungs 51+ provably
need clears. -/
theorem init_mem_safeIterate_twentyfive :
    GameState.init ∈ safeIterate GameConfig.standard 25 := by
  refine mem_safeIterate_of_three_group (by norm_num) 25 10 10 5 0 0 0
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    GameState.init ?_ ?_ ?_ <;>
    exact fun q hq => absurd hq (GameState.init_board_no_mem q)

/-- Monotone form: every depth up to twenty-five is certified. -/
theorem init_mem_safeIterate_of_le_twentyfive {k : ℕ} (hk : k ≤ 25) :
    GameState.init ∈ safeIterate GameConfig.standard k :=
  safeIterate_antitone GameConfig.standard hk init_mem_safeIterate_twentyfive

end Tetris
