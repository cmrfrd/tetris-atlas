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

end Tetris
