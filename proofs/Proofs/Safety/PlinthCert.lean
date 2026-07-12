import Proofs.Safety.BandSchedule
import Proofs.Invariants.PlinthShift

/-!
# The plinth certificate — the corrected inhabitation target

Two regimes: `Boot` (raw bag-1 boards, absolute level) and the plinth
steady state (`Inv`/`okB` over `BandState` representatives riding above the
immortal floor). Every steady obligation concludes in membership up to
re-anchoring (`PlinthMem`) — the D1 fix — and the entombed hole lives
outside the pattern coordinates — the D2 fix. Reduces directly to
`DebtCertificate` (plinth boards have well height 1, outside
`ShiftCertificate`'s image).
-/

namespace Tetris

open Board Seam

/-- Membership up to re-anchoring: the successor may re-split its absolute
band into (pattern, base) any equivalent way. Hole slot and window anchors
survive (heights shift uniformly; columns do not move). -/
def PlinthMem (well : ℕ) (Inv : Bag → BandState → Prop)
    (okB : Bag → BandState → ℕ → Prop)
    (T : Bag) (σ : BandState) (c : ℕ) : Prop :=
  ∃ σ' c', ReanchorsTo well σ.ρ c σ'.ρ c' ∧ σ'.ho = σ.ho ∧
    σ'.cS = σ.cS ∧ σ'.cZ = σ.cZ ∧ σ'.cO = σ.cO ∧ Inv T σ' ∧ okB T σ' c'

/-- **The plinth certificate.** Boot: raw bag-1 family from the empty board
through the well plug and the forced hole into the plinth. Steady: the
band-schedule obligations over the immortal floor, with re-anchoring. -/
structure PlinthCert where
  well : ℕ
  hwell : well < GameConfig.standard.cols
  /-- The entombed hole's column. -/
  hx : ℕ
  hhx : hx ≠ well ∧ hx < GameConfig.standard.cols
  /-- The bag-1 (pre-plinth) family, at the absolute level. -/
  Boot : Bag → (ℕ → ℕ) → Option Coord → Prop
  bootInit : Boot Bag.full (fun _ => 0) none
  bootCover : ∀ T h x, Boot T h (some x) →
    x.1 < GameConfig.standard.cols ∧ x.2 + 1 < h x.1
  bootHeight : ∀ T h ho, Boot T h ho →
    ∀ j < GameConfig.standard.cols, h j ≤ GameConfig.standard.rows
  /-- The steady family and its base predicate. -/
  Inv : Bag → BandState → Prop
  okB : Bag → BandState → ℕ → Prop
  /-- Steady states carry no extra hole (the debt slot is the entombed hole). -/
  hoNone : ∀ T σ, Inv T σ → σ.ho = none
  winS : ∀ T σ c, Inv T σ → σ.cS = some c → σ.ρ c = σ.ρ (c + 1) + 1
  winZ : ∀ T σ c, Inv T σ → σ.cZ = some c → σ.ρ (c + 1) = σ.ρ c + 1
  winO : ∀ T σ c, Inv T σ → σ.cO = some c → σ.ρ c = σ.ρ (c + 1)
  winColsS : ∀ T σ c, Inv T σ → σ.cS = some c →
    c + 1 < GameConfig.standard.cols ∧ c ≠ well ∧ c + 1 ≠ well
  winColsZ : ∀ T σ c, Inv T σ → σ.cZ = some c →
    c + 1 < GameConfig.standard.cols ∧ c ≠ well ∧ c + 1 ≠ well
  winColsO : ∀ T σ c, Inv T σ → σ.cO = some c →
    c + 1 < GameConfig.standard.cols ∧ c ≠ well ∧ c + 1 ≠ well
  anchored : ∀ T σ, Inv T σ → σ.ρ well = 0
  /-- The hole column stays loaded (keeps the entombed hole covered). -/
  holeLoaded : ∀ T σ, Inv T σ → 1 ≤ σ.ρ hx
  height : ∀ T σ b, Inv T σ → okB T σ b →
    ∀ j < GameConfig.standard.cols,
      Board.plinthLift well b σ.ρ j ≤ GameConfig.standard.rows
  invS : ∀ T σ c b, Inv T σ → okB T σ b → σ.cS = some c → Piece.S ∈ T →
    PlinthMem well Inv okB (T.draw Piece.S) (σ.bump2 c) b
  invZ : ∀ T σ c b, Inv T σ → okB T σ b → σ.cZ = some c → Piece.Z ∈ T →
    PlinthMem well Inv okB (T.draw Piece.Z) (σ.bump2 c) b
  invO : ∀ T σ c b, Inv T σ → okB T σ b → σ.cO = some c → Piece.O ∈ T →
    PlinthMem well Inv okB (T.draw Piece.O) (σ.bump2 c) b
  stepT : ∀ T σ b, Inv T σ → okB T σ b → Piece.T ∈ T →
    ∃ pl σ', pl.piece = Piece.T ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.skyline GameConfig.standard σ.ρ)
        = Board.skyline GameConfig.standard σ'.ρ ∧ σ'.ho = none ∧
      PlinthMem well Inv okB (T.draw Piece.T) σ' b
  stepL : ∀ T σ b, Inv T σ → okB T σ b → Piece.L ∈ T →
    ∃ pl σ', pl.piece = Piece.L ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.skyline GameConfig.standard σ.ρ)
        = Board.skyline GameConfig.standard σ'.ρ ∧ σ'.ho = none ∧
      PlinthMem well Inv okB (T.draw Piece.L) σ' b
  stepJ : ∀ T σ b, Inv T σ → okB T σ b → Piece.J ∈ T →
    ∃ pl σ', pl.piece = Piece.J ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.skyline GameConfig.standard σ.ρ)
        = Board.skyline GameConfig.standard σ'.ρ ∧ σ'.ho = none ∧
      PlinthMem well Inv okB (T.draw Piece.J) σ' b
  stepI : ∀ T σ b, Inv T σ → okB T σ b → Piece.I ∈ T →
    (4 ≤ b ∧ PlinthMem well Inv okB (T.draw Piece.I) σ (b - 4))
    ∨ (∃ pl σ', pl.piece = Piece.I ∧ pl.Valid GameConfig.standard ∧
        AvoidsWell well pl ∧
        pl.place (Board.skyline GameConfig.standard σ.ρ)
          = Board.skyline GameConfig.standard σ'.ρ ∧ σ'.ho = none ∧
        PlinthMem well Inv okB (T.draw Piece.I) σ' b)
  stepSBoot : ∀ T σ b, Inv T σ → okB T σ b → Piece.S ∈ T → σ.cS = none →
    ∃ pl σ', pl.piece = Piece.S ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.skyline GameConfig.standard σ.ρ)
        = Board.skyline GameConfig.standard σ'.ρ ∧ σ'.ho = none ∧
      PlinthMem well Inv okB (T.draw Piece.S) σ' b
  stepZBoot : ∀ T σ b, Inv T σ → okB T σ b → Piece.Z ∈ T → σ.cZ = none →
    ∃ pl σ', pl.piece = Piece.Z ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.skyline GameConfig.standard σ.ρ)
        = Board.skyline GameConfig.standard σ'.ρ ∧ σ'.ho = none ∧
      PlinthMem well Inv okB (T.draw Piece.Z) σ' b
  stepOBoot : ∀ T σ b, Inv T σ → okB T σ b → Piece.O ∈ T → σ.cO = none →
    ∃ pl σ', pl.piece = Piece.O ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.skyline GameConfig.standard σ.ρ)
        = Board.skyline GameConfig.standard σ'.ρ ∧ σ'.ho = none ∧
      PlinthMem well Inv okB (T.draw Piece.O) σ' b
  /-- Boot closure: every pending piece has a full-move response staying in
  Boot or handing off into the plinth. -/
  bootStep : ∀ T h ho p, Boot T h ho → p ∈ T →
    ∃ pl h' ho', pl.piece = p ∧ pl.Valid GameConfig.standard ∧
      Placement.applyStep GameConfig.standard
        (Board.debtBoard GameConfig.standard h ho) pl
        = Board.debtBoard GameConfig.standard h' ho' ∧
      (Boot (T.draw p) h' ho' ∨
        ∃ σ c, Inv (T.draw p) σ ∧ okB (T.draw p) σ c ∧
          h' = Board.plinthLift well c σ.ρ ∧ ho' = some (hx, 0))

namespace PlinthCert

/-- Derive validity and well-avoidance for a 2-column window response from
its shape disjunction (keeps `cols` out of the shape-enumeration goals). -/
theorem window_valid_avoid (C : PlinthCert) {cA : ℕ} {pl : Placement}
    (hcols : cA + 1 < GameConfig.standard.cols)
    (hcw : cA ≠ C.well) (hc1w : cA + 1 ≠ C.well)
    (hshape : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = cA ∨ pl.col + cell.1 = cA + 1) :
    pl.Valid GameConfig.standard ∧ AvoidsWell C.well pl := by
  constructor
  · intro cell hcell
    rcases hshape cell hcell with h | h <;> omega
  · intro cell hcell
    rcases hshape cell hcell with h | h <;> rw [h] <;> assumption

/-- Any plinth response (bare flush rep + `PlinthMem` successor) discharges
one `DebtCertificate.step` case: transport by T1′, no clears (the floor is
immortal), then pack the re-anchored member. -/
theorem plinth_response (C : PlinthCert) {T : Bag} {σ σ' : BandState}
    {b : ℕ} {p : Piece} (pl : Placement)
    (hInv : C.Inv T σ)
    (hpiece : pl.piece = p) (hval : pl.Valid GameConfig.standard)
    (havd : AvoidsWell C.well pl)
    (hrep : pl.place (Board.skyline GameConfig.standard σ.ρ)
      = Board.skyline GameConfig.standard σ'.ρ)
    (hMem : PlinthMem C.well C.Inv C.okB (T.draw p) σ' b) :
    ∃ h' ho', pl.piece = p ∧ pl.Valid GameConfig.standard ∧
      Placement.applyStep GameConfig.standard
        (Board.debtBoard GameConfig.standard
          (Board.plinthLift C.well b σ.ρ) (some (C.hx, 0))) pl
        = Board.debtBoard GameConfig.standard h' ho' ∧
      (∃ σ'' c, C.Inv (T.draw p) σ'' ∧ C.okB (T.draw p) σ'' c ∧
        h' = Board.plinthLift C.well c σ''.ρ ∧ ho' = some (C.hx, 0)) := by
  obtain ⟨σ'', c'', hre, -, -, -, -, hInv'', hok''⟩ := hMem
  have hhx3 : C.hx ≠ C.well ∧ C.hx < GameConfig.standard.cols ∧ 1 ≤ σ.ρ C.hx :=
    ⟨C.hhx.1, C.hhx.2, C.holeLoaded T σ hInv⟩
  have hlift := Board.place_debtBoard_plinthLift (c := b)
    (fun cell hcell => hval cell hcell) havd hhx3 hrep
  refine ⟨Board.plinthLift C.well c'' σ''.ρ, some (C.hx, 0), hpiece, hval, ?_,
    σ'', c'', hInv'', hok'', rfl, rfl⟩
  rw [Placement.applyStep_eq_clearLines_place, hlift,
    Board.clearLines_eq_self_of_no_fullRows GameConfig.standard
      (Board.fullRows_plinth_eq_empty C.hwell C.hhx.1 C.hhx.2),
    Board.plinthLift_congr_reanchor hre]

/-- **The reduction**: a plinth certificate yields a debt certificate. The
anchored S/Z/O cases are discharged here — the inhabitant never proves a
board equality for them. -/
def toDebtCertificate (C : PlinthCert) : DebtCertificate where
  P := fun T h ho =>
    C.Boot T h ho ∨
    ∃ σ c, C.Inv T σ ∧ C.okB T σ c ∧
      h = Board.plinthLift C.well c σ.ρ ∧ ho = some (C.hx, 0)
  init := Or.inl C.bootInit
  cover := by
    rintro T h x (hB | ⟨σ, c, hInv, -, rfl, hx'⟩)
    · exact C.bootCover T h x hB
    · rw [Option.some.injEq] at hx'
      subst hx'
      refine ⟨C.hhx.2, ?_⟩
      rw [Board.plinthLift_ne C.well c σ.ρ C.hhx.1]
      have := C.holeLoaded T σ hInv
      omega
  height := by
    rintro T h ho (hB | ⟨σ, c, hInv, hok, rfl, -⟩) j hj
    · exact C.bootHeight T h ho hB j hj
    · exact C.height T σ c hInv hok j hj
  step := by
    rintro T h ho p hP hp
    rcases hP with hB | ⟨σ, b, hInv, hokB, rfl, rfl⟩
    · -- Boot regime: designer's step, successor packed in either arm.
      obtain ⟨pl, h', ho', h1, h2, h3, h4⟩ := C.bootStep T h ho p hB hp
      refine ⟨pl, h', ho', h1, h2, h3, ?_⟩
      rcases h4 with hB' | ⟨σ, c, hInv, hok, rfl, rfl⟩
      · exact Or.inl hB'
      · exact Or.inr ⟨σ, c, hInv, hok, rfl, rfl⟩
    · -- Plinth regime.
      cases p with
      | S =>
          cases hcS : σ.cS with
          | none =>
              obtain ⟨pl, σ', h1, h2, h3, h4, -, h6⟩ :=
                C.stepSBoot T σ b hInv hokB hp hcS
              obtain ⟨h', ho', g1, g2, g3, g4⟩ :=
                C.plinth_response pl hInv h1 h2 h3 h4 h6
              exact ⟨pl, h', ho', g1, g2, g3, Or.inr g4⟩
          | some cA =>
              have hwin := C.winS T σ cA hInv hcS
              obtain ⟨hcols, hcw, hc1w⟩ := C.winColsS T σ cA hInv hcS
              have hMem := C.invS T σ cA b hInv hokB hcS hp
              have hrep : Placement.place
                  (Board.skyline GameConfig.standard σ.ρ)
                  { piece := Piece.S, rot := 1, col := cA }
                  = Board.skyline GameConfig.standard (σ.bump2 cA).ρ :=
                Board.place_vertS_skyline (by omega) hcols hwin
              have hshape : ∀ cell ∈ ({ piece := Piece.S, rot := 1, col := cA } :
                  Placement).shapeUp, cA + cell.1 = cA ∨ cA + cell.1 = cA + 1 := by
                intro cell hcell
                rw [Board.shapeUp_vertS cA] at hcell
                fin_cases hcell <;> simp <;> omega
              obtain ⟨hval, havd⟩ :=
                C.window_valid_avoid hcols hcw hc1w hshape
              obtain ⟨h', ho', g1, g2, g3, g4⟩ :=
                C.plinth_response ({ piece := Piece.S, rot := 1, col := cA })
                  hInv rfl hval havd hrep hMem
              exact ⟨{ piece := Piece.S, rot := 1, col := cA }, h', ho',
                g1, g2, g3, Or.inr g4⟩
      | Z =>
          cases hcZ : σ.cZ with
          | none =>
              obtain ⟨pl, σ', h1, h2, h3, h4, -, h6⟩ :=
                C.stepZBoot T σ b hInv hokB hp hcZ
              obtain ⟨h', ho', g1, g2, g3, g4⟩ :=
                C.plinth_response pl hInv h1 h2 h3 h4 h6
              exact ⟨pl, h', ho', g1, g2, g3, Or.inr g4⟩
          | some cA =>
              have hwin := C.winZ T σ cA hInv hcZ
              obtain ⟨hcols, hcw, hc1w⟩ := C.winColsZ T σ cA hInv hcZ
              have hMem := C.invZ T σ cA b hInv hokB hcZ hp
              have hrep : Placement.place
                  (Board.skyline GameConfig.standard σ.ρ)
                  { piece := Piece.Z, rot := 1, col := cA }
                  = Board.skyline GameConfig.standard (σ.bump2 cA).ρ :=
                Board.place_vertZ_skyline (by omega) hcols hwin
              have hshape : ∀ cell ∈ ({ piece := Piece.Z, rot := 1, col := cA } :
                  Placement).shapeUp, cA + cell.1 = cA ∨ cA + cell.1 = cA + 1 := by
                intro cell hcell
                rw [Board.shapeUp_vertZ cA] at hcell
                fin_cases hcell <;> simp <;> omega
              obtain ⟨hval, havd⟩ :=
                C.window_valid_avoid hcols hcw hc1w hshape
              obtain ⟨h', ho', g1, g2, g3, g4⟩ :=
                C.plinth_response ({ piece := Piece.Z, rot := 1, col := cA })
                  hInv rfl hval havd hrep hMem
              exact ⟨{ piece := Piece.Z, rot := 1, col := cA }, h', ho',
                g1, g2, g3, Or.inr g4⟩
      | O =>
          cases hcO : σ.cO with
          | none =>
              obtain ⟨pl, σ', h1, h2, h3, h4, -, h6⟩ :=
                C.stepOBoot T σ b hInv hokB hp hcO
              obtain ⟨h', ho', g1, g2, g3, g4⟩ :=
                C.plinth_response pl hInv h1 h2 h3 h4 h6
              exact ⟨pl, h', ho', g1, g2, g3, Or.inr g4⟩
          | some cA =>
              have hpair := C.winO T σ cA hInv hcO
              obtain ⟨hcols, hcw, hc1w⟩ := C.winColsO T σ cA hInv hcO
              have hMem := C.invO T σ cA b hInv hokB hcO hp
              have hrep : Placement.place
                  (Board.skyline GameConfig.standard σ.ρ)
                  { piece := Piece.O, rot := 0, col := cA }
                  = Board.skyline GameConfig.standard (σ.bump2 cA).ρ := by
                have hprof : Function.update (Function.update σ.ρ cA
                    (σ.ρ cA + 2)) (cA + 1) (σ.ρ cA + 2) = (σ.bump2 cA).ρ := by
                  unfold BandState.bump2
                  rw [hpair]
                rw [Board.place_O_pair (by omega) hcols hpair, hprof]
              have hshape : ∀ cell ∈ ({ piece := Piece.O, rot := 0, col := cA } :
                  Placement).shapeUp, cA + cell.1 = cA ∨ cA + cell.1 = cA + 1 := by
                intro cell hcell
                rw [Board.shapeUp_O cA 0] at hcell
                fin_cases hcell <;> simp <;> omega
              obtain ⟨hval, havd⟩ :=
                C.window_valid_avoid hcols hcw hc1w hshape
              obtain ⟨h', ho', g1, g2, g3, g4⟩ :=
                C.plinth_response ({ piece := Piece.O, rot := 0, col := cA })
                  hInv rfl hval havd hrep hMem
              exact ⟨{ piece := Piece.O, rot := 0, col := cA }, h', ho',
                g1, g2, g3, Or.inr g4⟩
      | T =>
          obtain ⟨pl, σ', h1, h2, h3, h4, -, h6⟩ := C.stepT T σ b hInv hokB hp
          obtain ⟨h', ho', g1, g2, g3, g4⟩ :=
            C.plinth_response pl hInv h1 h2 h3 h4 h6
          exact ⟨pl, h', ho', g1, g2, g3, Or.inr g4⟩
      | L =>
          obtain ⟨pl, σ', h1, h2, h3, h4, -, h6⟩ := C.stepL T σ b hInv hokB hp
          obtain ⟨h', ho', g1, g2, g3, g4⟩ :=
            C.plinth_response pl hInv h1 h2 h3 h4 h6
          exact ⟨pl, h', ho', g1, g2, g3, Or.inr g4⟩
      | J =>
          obtain ⟨pl, σ', h1, h2, h3, h4, -, h6⟩ := C.stepJ T σ b hInv hokB hp
          obtain ⟨h', ho', g1, g2, g3, g4⟩ :=
            C.plinth_response pl hInv h1 h2 h3 h4 h6
          exact ⟨pl, h', ho', g1, g2, g3, Or.inr g4⟩
      | I =>
          rcases C.stepI T σ b hInv hokB hp with ⟨hb4, hMem⟩ |
            ⟨pl, σ', h1, h2, h3, h4, -, h6⟩
          · obtain ⟨σ'', c'', hre, -, -, -, -, hInv'', hok''⟩ := hMem
            refine ⟨Board.bandDrain C.well,
              Board.plinthLift C.well c'' σ''.ρ, some (C.hx, 0), rfl,
              ?_, ?_, Or.inr ⟨σ'', c'', hInv'', hok'', rfl, rfl⟩⟩
            · rw [Board.bandDrain_eq_drainPl]
              exact drainPl_valid C.hwell
            · rw [show Board.plinthLift C.well c'' σ''.ρ
                  = Board.plinthLift C.well (b - 4) σ.ρ from
                  (Board.plinthLift_congr_reanchor hre).symm]
              exact Board.drain_debtBoard_plinthLift C.hwell hb4
                ⟨C.hhx.1, C.hhx.2, C.holeLoaded T σ hInv⟩
          · obtain ⟨h', ho', g1, g2, g3, g4⟩ :=
              C.plinth_response pl hInv h1 h2 h3 h4 h6
            exact ⟨pl, h', ho', g1, g2, g3, Or.inr g4⟩

end PlinthCert

/-- **Inhabiting the plinth certificate proves Tetris solvable.** The open
content: the T/L/J/I schedule with re-anchored bookkeeping, the unanchored
S/Z/O cases, and the bag-1 boot tree through the well plug. -/
theorem tetrisSolvableValid_of_plinthCert (C : PlinthCert) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_debtCertificate C.toDebtCertificate

end Tetris
