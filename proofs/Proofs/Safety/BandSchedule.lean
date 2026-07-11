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

end Tetris
