import Mathlib
import Proofs.Model.Placement
import Proofs.Model.Game
import Proofs.Combinatorics.BoardCount
import Proofs.Safety.SafeSet
import Proofs.Safety.SurfaceLanguage

/-!
# The seam bridge — reducing the full game to the clear-free band game

The seam architecture holds one **well column** empty while packing the
other nine, then drains with a vertical I. This file proves the discipline
is *exact*: while the well is open no clear can fire (a full row needs the
well cell), so every band move is a bare placement — the real game
restricted to the discipline **is** the clear-free heap game, the object
where the topical/max-plus theory, the surface projection, and placement
monotonicity all hold. The drain move is computed in closed form: with the
bottom four band rows complete, the vertical I clears exactly rows 0–3 and
the board shifts down four (`drain_applyStep`).

The payoff is `SeamCert.sound`: a **seam certificate** — an invariant set
of game states with the well open, closed under band placements (stated
with `place`, no clears) and drain moves (stated as the explicit
shift-by-four rewrite) — puts `init` in the safe set. Composed with
`safe_extract_for`, inhabiting `SeamCert` proves `TetrisSolvable`. The open
question is thereby confined to the finite, clear-free band game: every
future closure argument may reason about bare placements and one explicit
row-deletion formula, never about `clearLines`.
-/

namespace Tetris
namespace Seam

open Tetris

/-- The well column of `b` is completely empty. -/
def WellFree (w : ℕ) (b : Board) : Prop := ∀ r : ℕ, ((w, r) : Coord) ∉ b

/-- A placement stays out of the well column. -/
def AvoidsWell (w : ℕ) (pl : Placement) : Prop :=
  ∀ cell ∈ pl.shapeUp, pl.col + cell.1 ≠ w

/-! ## Band moves: no clears while the well is open -/

/-- A row missing its well cell is not full. -/
theorem not_isFull_of_wellFree {cfg : GameConfig} {w : ℕ} (hw : w < cfg.cols)
    {b : Board} (hb : WellFree w b) (r : ℕ) : ¬ Board.isFull cfg b r := by
  intro hfull
  exact hb r (hfull w (Finset.mem_range.mpr hw))

/-- A well-free board has no full rows at all. -/
theorem fullRows_eq_empty_of_wellFree {cfg : GameConfig} {w : ℕ}
    (hw : w < cfg.cols) {b : Board} (hb : WellFree w b) :
    Board.fullRows cfg b = ∅ := by
  rw [Finset.eq_empty_iff_forall_notMem]
  intro r hr
  unfold Board.fullRows at hr
  rw [Finset.mem_filter] at hr
  exact not_isFull_of_wellFree hw hb r hr.2

/-- A well-avoiding placement keeps the well free. -/
theorem wellFree_place {w : ℕ} {b : Board} (hb : WellFree w b)
    {pl : Placement} (ha : AvoidsWell w pl) :
    WellFree w (pl.place b) := by
  intro r hr
  rw [Placement.place_eq_union_dropped, Finset.mem_union] at hr
  rcases hr with hmem | hmem
  · exact hb r hmem
  · rw [Placement.dropped_eq_image, Finset.mem_image] at hmem
    obtain ⟨cell, hcell, heq⟩ := hmem
    exact ha cell hcell (congrArg Prod.fst heq)

/-- **The band bridge**: on a well-free board, a well-avoiding move never
clears — the full game step is the bare placement. Every within-band move
of the seam discipline lives in the clear-free heap game. -/
theorem applyStep_eq_place_of_wellFree {cfg : GameConfig} {w : ℕ}
    (hw : w < cfg.cols) {b : Board} (hb : WellFree w b)
    {pl : Placement} (ha : AvoidsWell w pl) :
    pl.applyStep cfg b = pl.place b := by
  rw [Placement.applyStep_eq_clearLines_place,
    Board.clearLines_eq_self_of_no_fullRows cfg
      (fullRows_eq_empty_of_wellFree hw (wellFree_place hb ha))]

/-! ## The drain move, in closed form -/

/-- The vertical I profile. -/
theorem I_shapeUp_vertical :
    Piece.I.shapeUp 1 = ({(0, 0), (0, 1), (0, 2), (0, 3)} : Finset PieceCell) := by
  decide

/-- The drain placement: a vertical I in the well column. -/
def drainPl (w : ℕ) : Placement := ⟨Piece.I, 1, w⟩

theorem drainPl_valid {cfg : GameConfig} {w : ℕ} (hw : w < cfg.cols) :
    (drainPl w).Valid cfg := by
  intro cell hcell
  have hsh : (drainPl w).shapeUp
      = ({(0, 0), (0, 1), (0, 2), (0, 3)} : Finset PieceCell) :=
    I_shapeUp_vertical
  rw [hsh] at hcell
  simp only [Finset.mem_insert, Finset.mem_singleton] at hcell
  rcases hcell with rfl | rfl | rfl | rfl <;> simpa using hw

/-- The bottom four rows are complete outside the well. -/
def BandComplete (cfg : GameConfig) (w : ℕ) (b : Board) : Prop :=
  ∀ c < cfg.cols, c ≠ w → ∀ r < 4, ((c, r) : Coord) ∈ b

/-- The drain's landing: on a well-free board the vertical I falls to the
floor and occupies exactly the four bottom well cells. -/
theorem drain_dropped {w : ℕ} {b : Board} (hb : WellFree w b) :
    (drainPl w).dropped b
      = ({(w, 0), (w, 1), (w, 2), (w, 3)} : Finset Coord) := by
  have hch : b.colHeight w = 0 := by
    unfold Board.colHeight
    have hempty : b.colRows w = ∅ := by
      rw [Finset.eq_empty_iff_forall_notMem]
      intro r hr
      exact hb r (SurfaceLanguage.mem_colRows_iff.mp hr)
    rw [hempty, Finset.sup_empty]
    rfl
  have hoff : (drainPl w).dropOffset b = 0 := by
    rw [Placement.dropOffset_eq_sup]
    apply Nat.le_antisymm _ (Nat.zero_le _)
    apply Finset.sup_le
    intro cell hcell
    have hsh : (drainPl w).shapeUp
        = ({(0, 0), (0, 1), (0, 2), (0, 3)} : Finset PieceCell) :=
      I_shapeUp_vertical
    rw [hsh] at hcell
    simp only [Finset.mem_insert, Finset.mem_singleton] at hcell
    rcases hcell with rfl | rfl | rfl | rfl <;>
      simp [drainPl, hch]
  rw [Placement.dropped_eq_cellsAt, hoff]
  unfold Placement.cellsAt
  have hsh : (drainPl w).shapeUp
      = ({(0, 0), (0, 1), (0, 2), (0, 3)} : Finset PieceCell) :=
    I_shapeUp_vertical
  rw [hsh]
  ext q
  simp only [Finset.mem_image, Finset.mem_insert, Finset.mem_singleton]
  constructor
  · rintro ⟨cell, hcell, rfl⟩
    rcases hcell with rfl | rfl | rfl | rfl <;> simp [drainPl]
  · rintro (rfl | rfl | rfl | rfl)
    · exact ⟨(0, 0), by simp [drainPl]⟩
    · exact ⟨(0, 1), by simp [drainPl]⟩
    · exact ⟨(0, 2), by simp [drainPl]⟩
    · exact ⟨(0, 3), by simp [drainPl]⟩

/-- Membership in the post-drain merge. -/
theorem mem_drain_place {w : ℕ} {b : Board} (hb : WellFree w b) (q : Coord) :
    q ∈ (drainPl w).place b ↔ q ∈ b ∨ (q.1 = w ∧ q.2 < 4) := by
  rw [Placement.place_eq_union_dropped, Finset.mem_union, drain_dropped hb]
  constructor
  · rintro (hq | hq)
    · exact Or.inl hq
    · simp only [Finset.mem_insert, Finset.mem_singleton] at hq
      rcases hq with rfl | rfl | rfl | rfl <;> simp
  · rintro (hq | ⟨h1, h2⟩)
    · exact Or.inl hq
    · right
      obtain ⟨q1, q2⟩ := q
      simp only at h1 h2
      subst h1
      have hq : q2 = 0 ∨ q2 = 1 ∨ q2 = 2 ∨ q2 = 3 := by omega
      rcases hq with rfl | rfl | rfl | rfl <;> simp

/-- With the band complete, the drain fills rows 0–3 exactly. -/
theorem drain_fullRows {cfg : GameConfig} {w : ℕ} (hw : w < cfg.cols)
    {b : Board} (hb : WellFree w b) (hc : BandComplete cfg w b) :
    Board.fullRows cfg ((drainPl w).place b) = Finset.range 4 := by
  ext r
  unfold Board.fullRows
  rw [Finset.mem_filter, Finset.mem_range]
  constructor
  · rintro ⟨-, hfull⟩
    by_contra hr4
    rw [Nat.not_lt] at hr4
    have hwell := hfull w (Finset.mem_range.mpr hw)
    rw [mem_drain_place hb] at hwell
    rcases hwell with hmem | ⟨-, h2⟩
    · exact hb r hmem
    · omega
  · intro hr4
    constructor
    · rw [Finset.mem_image]
      refine ⟨(w, r), ?_, rfl⟩
      rw [mem_drain_place hb]
      exact Or.inr ⟨rfl, hr4⟩
    · intro c hc'
      rw [Finset.mem_range] at hc'
      rw [mem_drain_place hb]
      by_cases hcw : c = w
      · exact Or.inr ⟨hcw, hr4⟩
      · exact Or.inl (hc c hc' hcw r hr4)

/-- **The drain bridge**: with the well free and the band's bottom four
rows complete, the vertical I clears exactly rows 0–3 and the whole board
shifts down four — the full game step is the explicit row-deletion rewrite.
The I's own cells and the four completed band rows vanish; nothing else is
touched. -/
theorem drain_applyStep {cfg : GameConfig} {w : ℕ} (hw : w < cfg.cols)
    {b : Board} (hb : WellFree w b) (hc : BandComplete cfg w b) :
    (drainPl w).applyStep cfg b
      = (b.filter (fun q => 4 ≤ q.2)).image (fun q => (q.1, q.2 - 4)) := by
  rw [Placement.applyStep_eq_clearLines_place]
  unfold Board.clearLines
  have hfull := drain_fullRows hw hb hc
  have hfull_iff : ∀ r, Board.isFull cfg ((drainPl w).place b) r ↔ r < 4 := by
    intro r
    constructor
    · intro hisfull
      by_contra hr4
      have hr : r ∈ Board.fullRows cfg ((drainPl w).place b) := by
        unfold Board.fullRows
        rw [Finset.mem_filter, Finset.mem_image]
        refine ⟨⟨(w, r), ?_, rfl⟩, hisfull⟩
        exact hisfull w (Finset.mem_range.mpr hw)
      rw [hfull, Finset.mem_range] at hr
      exact hr4 hr
    · intro hr4
      have hr : r ∈ Board.fullRows cfg ((drainPl w).place b) := by
        rw [hfull, Finset.mem_range]
        exact hr4
      unfold Board.fullRows at hr
      rw [Finset.mem_filter] at hr
      exact hr.2
  have hfilter : ((drainPl w).place b).filter
      (fun p => ¬ Board.isFull cfg ((drainPl w).place b) p.2)
      = b.filter (fun q => 4 ≤ q.2) := by
    ext q
    rw [Finset.mem_filter, Finset.mem_filter, mem_drain_place hb, hfull_iff]
    constructor
    · rintro ⟨hmem | ⟨-, h2⟩, hnf⟩
      · exact ⟨hmem, by omega⟩
      · omega
    · rintro ⟨hmem, h4⟩
      exact ⟨Or.inl hmem, by omega⟩
  rw [hfilter]
  apply Finset.image_congr
  intro q hq
  rw [Finset.mem_coe, Finset.mem_filter] at hq
  change (q.1, q.2 - Board.clearedBelow cfg ((drainPl w).place b) q.2)
      = (q.1, q.2 - 4)
  have hcb : Board.clearedBelow cfg ((drainPl w).place b) q.2 = 4 := by
    unfold Board.clearedBelow
    rw [hfull]
    rw [Finset.filter_true_of_mem, Finset.card_range]
    intro r hr
    rw [Finset.mem_range] at hr
    omega
  rw [hcb]

/-! ## The seam certificate and its soundness -/

/-- **A seam certificate**: an invariant set of game states, well column
open throughout, closed under (a) well-avoiding band placements — stated
with bare `place`, because the band bridge shows no clear can fire — and
(b) drain moves — stated as the explicit shift-by-four rewrite. Inhabiting
this structure requires reasoning only about the clear-free band game;
`clearLines` never appears in its obligations. -/
structure SeamCert (cfg : GameConfig) where
  /-- The well column. -/
  well : ℕ
  hwell : well < cfg.cols
  /-- The invariant set of game states. -/
  S : Set GameState
  hinit : GameState.init ∈ S
  /-- The well is open in every certified state. -/
  hfree : ∀ g ∈ S, WellFree well g.board
  /-- No certified state is lost. -/
  hnotlost : ∀ g ∈ S, ¬ g.lost cfg
  /-- Closure: every bag piece has either a band response (bare placement,
  avoiding the well, landing in `S`) or a drain response (piece is I, band
  complete, shifted board in `S`). -/
  hstep : ∀ g ∈ S, ∀ p ∈ g.bag,
    (∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧ AvoidsWell well pl ∧
      (⟨pl.place g.board, g.bag.draw p⟩ : GameState) ∈ S)
    ∨ (p = Piece.I ∧ BandComplete cfg well g.board ∧
      (⟨(g.board.filter (fun q => 4 ≤ q.2)).image (fun q => (q.1, q.2 - 4)),
        g.bag.draw p⟩ : GameState) ∈ S)

/-- **Soundness of the seam certificate**: the certified set is closed
under the *real* safe-step operator — the two bridges convert each
clear-free obligation into the actual `adversarialStep` — so the initial
state is safe. -/
theorem SeamCert.sound {cfg : GameConfig} (C : SeamCert cfg) :
    GameState.init ∈ safe cfg := by
  apply safe_greatest C.S _ C.hinit
  intro g hg
  refine ⟨C.hnotlost g hg, ?_⟩
  intro p hp
  rcases C.hstep g hg p hp with ⟨pl, hpiece, hv, ha, hmem⟩ | ⟨hI, hcomp, hmem⟩
  · refine ⟨pl, hpiece, hv, ?_⟩
    have hpl_eta : ({ pl with piece := p } : Placement) = pl := by
      obtain ⟨piece, rot, col⟩ := pl
      have hp' : piece = p := hpiece
      rw [hp']
    have hboard : adversarialStep cfg g p pl
        = (⟨pl.place g.board, g.bag.draw p⟩ : GameState) := by
      change (⟨Placement.applyStep cfg g.board { pl with piece := p },
        g.bag.draw p⟩ : GameState) = _
      rw [hpl_eta,
        applyStep_eq_place_of_wellFree C.hwell (C.hfree g hg) ha]
    rw [hboard]
    exact hmem
  · subst hI
    refine ⟨drainPl C.well, rfl, drainPl_valid C.hwell, ?_⟩
    have hboard : adversarialStep cfg g Piece.I (drainPl C.well)
        = (⟨(g.board.filter (fun q => 4 ≤ q.2)).image (fun q => (q.1, q.2 - 4)),
            g.bag.draw Piece.I⟩ : GameState) := by
      change (⟨Placement.applyStep cfg g.board
          { drainPl C.well with piece := Piece.I },
        g.bag.draw Piece.I⟩ : GameState) = _
      have hpl_eta : ({ drainPl C.well with piece := Piece.I } : Placement)
          = drainPl C.well := rfl
      rw [hpl_eta,
        drain_applyStep C.hwell (C.hfree g hg) hcomp]
    rw [hboard]
    exact hmem

/-- A seam certificate proves solvability (parametric form). -/
theorem SeamCert.solvable {cfg : GameConfig} (C : SeamCert cfg) :
    TetrisSolvableFor cfg :=
  safe_extract_for C.sound

/-- **A seam certificate at the standard config proves `TetrisSolvable`.**
The open conjecture is reduced to inhabiting `SeamCert GameConfig.standard`
— a closure problem for the clear-free nine-column band game. -/
theorem SeamCert.tetrisSolvable (C : SeamCert GameConfig.standard) :
    TetrisSolvable :=
  safe_extract C.sound

end Seam
end Tetris
