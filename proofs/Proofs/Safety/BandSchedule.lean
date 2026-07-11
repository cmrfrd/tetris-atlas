import Proofs.Safety.ShiftCertificate
import Proofs.Invariants.BandMechanisms

/-!
# The band-schedule certificate

`BandScheduleCert` is the isolated remainder of the Tetris-solvability
construction after the translation quotient: a designer family over
`BandState` (base-0 profile + debt hole + `Option` window anchors) whose
anchored S/Z/O closure is discharged HERE, unconditionally, from the
reproduction mechanisms and the debt-carry wrapper. The open content is
exactly: the T/L/J responses, the I disjunction (drain guard or band
response), the un-anchored S/Z/O cases (the bag-1 bootstrap wiring — see
the pack in `Proofs/Invariants/BandMechanisms.lean`), and the `okB` rate
bookkeeping. Inhabiting this structure proves Tetris solvable
(`tetrisSolvableValid_of_bandSchedule`).
-/

namespace Tetris

open Board Seam

/-- A steady-state band representative: base-0 profile, optional debt hole,
and optional window anchors (unset during the bag-1 bootstrap). -/
structure BandState where
  ρ  : ℕ → ℕ
  ho : Option Coord
  cS : Option ℕ
  cZ : Option ℕ
  cO : Option ℕ

namespace BandState

/-- The forced flush successor at a 2-column window: both columns rise 2.
Matches the output profiles of `place_vertS_skyline`, `place_vertZ_skyline`
and (after rewriting by the pair equality) `place_O_pair` verbatim. -/
def bump2 (σ : BandState) (c : ℕ) : BandState :=
  { σ with
    ρ := Function.update (Function.update σ.ρ c (σ.ρ c + 2)) (c + 1)
      (σ.ρ (c + 1) + 2) }

@[simp] theorem bump2_ho (σ : BandState) (c : ℕ) : (σ.bump2 c).ho = σ.ho := rfl

end BandState

/-- The flat start state: empty board, no debt, no anchors. -/
def BandState.start : BandState := ⟨fun _ => 0, none, none, none, none⟩

/-- **The band-schedule certificate** — the isolated open remainder.
Anchored S/Z/O responses are forced (`bump2`) and their board content is
proven by the library; the inhabitant supplies the schedule: T/L/J, the I
case, the bootstrap (`none`-anchor) cases, and the rate bookkeeping. -/
structure BandScheduleCert where
  /-- The well column. -/
  well : ℕ
  hwell : well < GameConfig.standard.cols
  /-- The designer family. -/
  Inv : Bag → BandState → Prop
  /-- The designer base predicate (admissible band bases per state). -/
  okB : Bag → BandState → ℕ → Prop
  init : Inv Bag.full BandState.start
  initBase : okB Bag.full BandState.start 0
  /-- Window shape when the S anchor is set. -/
  winS : ∀ T σ c, Inv T σ → σ.cS = some c → σ.ρ c = σ.ρ (c + 1) + 1
  /-- Window shape when the Z anchor is set. -/
  winZ : ∀ T σ c, Inv T σ → σ.cZ = some c → σ.ρ (c + 1) = σ.ρ c + 1
  /-- Window shape when the O anchor is set. -/
  winO : ∀ T σ c, Inv T σ → σ.cO = some c → σ.ρ c = σ.ρ (c + 1)
  /-- The anchored S window sits in the band, off the well. -/
  winColsS : ∀ T σ c, Inv T σ → σ.cS = some c →
    c + 1 < GameConfig.standard.cols ∧ c ≠ well ∧ c + 1 ≠ well
  /-- The anchored Z window sits in the band, off the well. -/
  winColsZ : ∀ T σ c, Inv T σ → σ.cZ = some c →
    c + 1 < GameConfig.standard.cols ∧ c ≠ well ∧ c + 1 ≠ well
  /-- The anchored O window sits in the band, off the well. -/
  winColsO : ∀ T σ c, Inv T σ → σ.cO = some c →
    c + 1 < GameConfig.standard.cols ∧ c ≠ well ∧ c + 1 ≠ well
  /-- The well column is empty at every representative. -/
  anchored : ∀ T σ, Inv T σ → σ.ρ well = 0
  /-- The debt hole is in-band, in-field, strictly covered. -/
  cover : ∀ T σ x, Inv T σ → σ.ho = some x →
    x.1 ≠ well ∧ x.1 < GameConfig.standard.cols ∧ x.2 + 1 < σ.ρ x.1
  /-- Admissible bases respect the ceiling. -/
  height : ∀ T σ b, Inv T σ → okB T σ b →
    ∀ j < GameConfig.standard.cols,
      Board.bandLift well b σ.ρ j ≤ GameConfig.standard.rows
  /-- Bookkeeping: the forced S successor stays in the family. -/
  invS : ∀ T σ c, Inv T σ → σ.cS = some c → Piece.S ∈ T →
    Inv (T.draw Piece.S) (σ.bump2 c) ∧
    ∀ b, okB T σ b → okB (T.draw Piece.S) (σ.bump2 c) b
  /-- Bookkeeping: the forced Z successor stays in the family. -/
  invZ : ∀ T σ c, Inv T σ → σ.cZ = some c → Piece.Z ∈ T →
    Inv (T.draw Piece.Z) (σ.bump2 c) ∧
    ∀ b, okB T σ b → okB (T.draw Piece.Z) (σ.bump2 c) b
  /-- Bookkeeping: the forced O successor stays in the family. -/
  invO : ∀ T σ c, Inv T σ → σ.cO = some c → Piece.O ∈ T →
    Inv (T.draw Piece.O) (σ.bump2 c) ∧
    ∀ b, okB T σ b → okB (T.draw Piece.O) (σ.bump2 c) b
  /-- The open schedule content: T. -/
  stepT : ∀ T σ b, Inv T σ → okB T σ b → Piece.T ∈ T →
    ∃ pl σ', pl.piece = Piece.T ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
        = Board.debtBoard GameConfig.standard σ'.ρ σ'.ho ∧
      Inv (T.draw Piece.T) σ' ∧ okB (T.draw Piece.T) σ' b
  /-- The open schedule content: L. -/
  stepL : ∀ T σ b, Inv T σ → okB T σ b → Piece.L ∈ T →
    ∃ pl σ', pl.piece = Piece.L ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
        = Board.debtBoard GameConfig.standard σ'.ρ σ'.ho ∧
      Inv (T.draw Piece.L) σ' ∧ okB (T.draw Piece.L) σ' b
  /-- The open schedule content: J. -/
  stepJ : ∀ T σ b, Inv T σ → okB T σ b → Piece.J ∈ T →
    ∃ pl σ', pl.piece = Piece.J ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
        = Board.debtBoard GameConfig.standard σ'.ρ σ'.ho ∧
      Inv (T.draw Piece.J) σ' ∧ okB (T.draw Piece.J) σ' b
  /-- The open schedule content: I — drain when the base allows, else a
  band response (the bag-1 case). -/
  stepI : ∀ T σ b, Inv T σ → okB T σ b → Piece.I ∈ T →
    (4 ≤ b ∧ Inv (T.draw Piece.I) σ ∧ okB (T.draw Piece.I) σ (b - 4))
    ∨ (∃ pl σ', pl.piece = Piece.I ∧ pl.Valid GameConfig.standard ∧
        AvoidsWell well pl ∧
        pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
          = Board.debtBoard GameConfig.standard σ'.ρ σ'.ho ∧
        Inv (T.draw Piece.I) σ' ∧ okB (T.draw Piece.I) σ' b)
  /-- Bootstrap: S with no anchor (the bag-1 wiring; see the pack). -/
  stepSBoot : ∀ T σ b, Inv T σ → okB T σ b → Piece.S ∈ T → σ.cS = none →
    ∃ pl σ', pl.piece = Piece.S ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
        = Board.debtBoard GameConfig.standard σ'.ρ σ'.ho ∧
      Inv (T.draw Piece.S) σ' ∧ okB (T.draw Piece.S) σ' b
  /-- Bootstrap: Z with no anchor. -/
  stepZBoot : ∀ T σ b, Inv T σ → okB T σ b → Piece.Z ∈ T → σ.cZ = none →
    ∃ pl σ', pl.piece = Piece.Z ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
        = Board.debtBoard GameConfig.standard σ'.ρ σ'.ho ∧
      Inv (T.draw Piece.Z) σ' ∧ okB (T.draw Piece.Z) σ' b
  /-- Bootstrap: O with no anchor. -/
  stepOBoot : ∀ T σ b, Inv T σ → okB T σ b → Piece.O ∈ T → σ.cO = none →
    ∃ pl σ', pl.piece = Piece.O ∧ pl.Valid GameConfig.standard ∧
      AvoidsWell well pl ∧
      pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
        = Board.debtBoard GameConfig.standard σ'.ρ σ'.ho ∧
      Inv (T.draw Piece.O) σ' ∧ okB (T.draw Piece.O) σ' b

namespace BandScheduleCert

/-- The three anchored window responses share one discharge: a flush
2-column mechanism at the anchor + the debt-carry wrapper. -/
theorem window_response (C : BandScheduleCert) {T : Bag} {σ : BandState}
    {c : ℕ} (pl : Placement)
    (hcols : c + 1 < GameConfig.standard.cols)
    (hcw : c ≠ C.well) (hc1w : c + 1 ≠ C.well)
    (hshape : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = c ∨ pl.col + cell.1 = c + 1)
    (hInv : C.Inv T σ)
    (hflush : pl.place (Board.skyline GameConfig.standard σ.ρ)
      = Board.skyline GameConfig.standard (σ.bump2 c).ρ) :
    pl.Valid GameConfig.standard ∧ AvoidsWell C.well pl ∧
      pl.place (Board.debtBoard GameConfig.standard σ.ρ σ.ho)
        = Board.debtBoard GameConfig.standard (σ.bump2 c).ρ (σ.bump2 c).ho := by
  refine ⟨?_, ?_, ?_⟩
  · intro cell hcell
    rcases hshape cell hcell with h | h <;> omega
  · intro cell hcell
    rcases hshape cell hcell with h | h <;> rw [h] <;> assumption
  · rw [BandState.bump2_ho]
    exact Board.place_debtBoard_of_flush hflush
      (fun x hx => ⟨(C.cover T σ x hInv hx).2.1, (C.cover T σ x hInv hx).2.2⟩)

/-- **The reduction**: a band schedule yields a shift certificate. The
anchored S/Z/O cases are discharged here — the inhabitant never proves a
board equality for them. -/
def toShiftCertificate (C : BandScheduleCert) : ShiftCertificate where
  well := C.well
  hwell := C.hwell
  Q := fun T ρ ho => ∃ σ : BandState, C.Inv T σ ∧ σ.ρ = ρ ∧ σ.ho = ho
  okBase := fun T ρ ho b =>
    ∃ σ : BandState, C.Inv T σ ∧ σ.ρ = ρ ∧ σ.ho = ho ∧ C.okB T σ b
  init := ⟨BandState.start, C.init, rfl, rfl⟩
  initBase := ⟨BandState.start, C.init, rfl, rfl, C.initBase⟩
  cover := by
    rintro T ρ x ⟨σ, hInv, rfl, hho⟩
    exact C.cover T σ x hInv hho
  height := by
    rintro T ρ ho b hQ ⟨σ, hInv, rfl, -, hokB⟩ j hj
    exact C.height T σ b hInv hokB j hj
  step := by
    rintro T ρ ho b p hQ ⟨σ, hInv, rfl, rfl, hokB⟩ hp
    cases p with
    | S =>
        cases hcS : σ.cS with
        | none =>
            obtain ⟨pl, σ', h1, h2, h3, h4, h5, h6⟩ :=
              C.stepSBoot T σ b hInv hokB hp hcS
            exact Or.inl ⟨pl, σ'.ρ, σ'.ho, h1, h2, h3, h4,
              ⟨σ', h5, rfl, rfl⟩, ⟨σ', h5, rfl, rfl, h6⟩⟩
        | some c =>
            obtain ⟨hcols, hcw, hc1w⟩ := C.winColsS T σ c hInv hcS
            obtain ⟨hInv', hok'⟩ := C.invS T σ c hInv hcS hp
            have hwin := C.winS T σ c hInv hcS
            have hflush : Placement.place
                (Board.skyline GameConfig.standard σ.ρ)
                { piece := Piece.S, rot := 1, col := c }
                = Board.skyline GameConfig.standard (σ.bump2 c).ρ :=
              Board.place_vertS_skyline (by omega) hcols hwin
            obtain ⟨hval, havd, hplace⟩ := C.window_response
              ({ piece := Piece.S, rot := 1, col := c }) hcols hcw hc1w
              (by
                intro cell hcell
                rw [Board.shapeUp_vertS c] at hcell
                fin_cases hcell <;> simp <;> omega) hInv hflush
            exact Or.inl ⟨_, (σ.bump2 c).ρ, (σ.bump2 c).ho, rfl, hval, havd,
              hplace, ⟨σ.bump2 c, hInv', rfl, rfl⟩,
              ⟨σ.bump2 c, hInv', rfl, rfl, hok' b hokB⟩⟩
    | Z =>
        cases hcZ : σ.cZ with
        | none =>
            obtain ⟨pl, σ', h1, h2, h3, h4, h5, h6⟩ :=
              C.stepZBoot T σ b hInv hokB hp hcZ
            exact Or.inl ⟨pl, σ'.ρ, σ'.ho, h1, h2, h3, h4,
              ⟨σ', h5, rfl, rfl⟩, ⟨σ', h5, rfl, rfl, h6⟩⟩
        | some c =>
            obtain ⟨hcols, hcw, hc1w⟩ := C.winColsZ T σ c hInv hcZ
            obtain ⟨hInv', hok'⟩ := C.invZ T σ c hInv hcZ hp
            have hwin := C.winZ T σ c hInv hcZ
            have hflush : Placement.place
                (Board.skyline GameConfig.standard σ.ρ)
                { piece := Piece.Z, rot := 1, col := c }
                = Board.skyline GameConfig.standard (σ.bump2 c).ρ :=
              Board.place_vertZ_skyline (by omega) hcols hwin
            obtain ⟨hval, havd, hplace⟩ := C.window_response
              ({ piece := Piece.Z, rot := 1, col := c }) hcols hcw hc1w
              (by
                intro cell hcell
                rw [Board.shapeUp_vertZ c] at hcell
                fin_cases hcell <;> simp <;> omega) hInv hflush
            exact Or.inl ⟨_, (σ.bump2 c).ρ, (σ.bump2 c).ho, rfl, hval, havd,
              hplace, ⟨σ.bump2 c, hInv', rfl, rfl⟩,
              ⟨σ.bump2 c, hInv', rfl, rfl, hok' b hokB⟩⟩
    | O =>
        cases hcO : σ.cO with
        | none =>
            obtain ⟨pl, σ', h1, h2, h3, h4, h5, h6⟩ :=
              C.stepOBoot T σ b hInv hokB hp hcO
            exact Or.inl ⟨pl, σ'.ρ, σ'.ho, h1, h2, h3, h4,
              ⟨σ', h5, rfl, rfl⟩, ⟨σ', h5, rfl, rfl, h6⟩⟩
        | some c =>
            obtain ⟨hcols, hcw, hc1w⟩ := C.winColsO T σ c hInv hcO
            obtain ⟨hInv', hok'⟩ := C.invO T σ c hInv hcO hp
            have hpair := C.winO T σ c hInv hcO
            have hflush : Placement.place
                (Board.skyline GameConfig.standard σ.ρ)
                { piece := Piece.O, rot := 0, col := c }
                = Board.skyline GameConfig.standard (σ.bump2 c).ρ := by
              have hprof : Function.update (Function.update σ.ρ c (σ.ρ c + 2))
                  (c + 1) (σ.ρ c + 2) = (σ.bump2 c).ρ := by
                unfold BandState.bump2
                rw [hpair]
              rw [Board.place_O_pair (by omega) hcols hpair, hprof]
            obtain ⟨hval, havd, hplace⟩ := C.window_response
              ({ piece := Piece.O, rot := 0, col := c }) hcols hcw hc1w
              (by
                intro cell hcell
                rw [Board.shapeUp_O c 0] at hcell
                fin_cases hcell <;> simp <;> omega) hInv hflush
            exact Or.inl ⟨_, (σ.bump2 c).ρ, (σ.bump2 c).ho, rfl, hval, havd,
              hplace, ⟨σ.bump2 c, hInv', rfl, rfl⟩,
              ⟨σ.bump2 c, hInv', rfl, rfl, hok' b hokB⟩⟩
    | T =>
        obtain ⟨pl, σ', h1, h2, h3, h4, h5, h6⟩ := C.stepT T σ b hInv hokB hp
        exact Or.inl ⟨pl, σ'.ρ, σ'.ho, h1, h2, h3, h4,
          ⟨σ', h5, rfl, rfl⟩, ⟨σ', h5, rfl, rfl, h6⟩⟩
    | L =>
        obtain ⟨pl, σ', h1, h2, h3, h4, h5, h6⟩ := C.stepL T σ b hInv hokB hp
        exact Or.inl ⟨pl, σ'.ρ, σ'.ho, h1, h2, h3, h4,
          ⟨σ', h5, rfl, rfl⟩, ⟨σ', h5, rfl, rfl, h6⟩⟩
    | J =>
        obtain ⟨pl, σ', h1, h2, h3, h4, h5, h6⟩ := C.stepJ T σ b hInv hokB hp
        exact Or.inl ⟨pl, σ'.ρ, σ'.ho, h1, h2, h3, h4,
          ⟨σ', h5, rfl, rfl⟩, ⟨σ', h5, rfl, rfl, h6⟩⟩
    | I =>
        rcases C.stepI T σ b hInv hokB hp with ⟨hb4, hInv', hok'⟩ |
          ⟨pl, σ', h1, h2, h3, h4, h5, h6⟩
        · exact Or.inr ⟨rfl, hb4, ⟨σ, hInv', rfl, rfl⟩,
            ⟨σ, hInv', rfl, rfl, hok'⟩⟩
        · exact Or.inl ⟨pl, σ'.ρ, σ'.ho, h1, h2, h3, h4,
            ⟨σ', h5, rfl, rfl⟩, ⟨σ', h5, rfl, rfl, h6⟩⟩

end BandScheduleCert

/-- **Inhabiting the band schedule proves Tetris solvable.** The remaining
open content of the whole development is the schedule: T/L/J, the I
disjunction, the bootstrap wiring, and the rate bookkeeping. -/
theorem tetrisSolvableValid_of_bandSchedule (C : BandScheduleCert) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_shiftCertificate C.toShiftCertificate

end Tetris
