import Proofs.Safety.SkylineInvariant
import Proofs.Safety.SeamBridge
import Proofs.Invariants.BandShift

/-!
# The translation-quotient certificate

`ShiftCertificate` is `DebtCertificate` quotiented by the well-anchored band
translation: states are base-0 representatives `(bag, pattern, hole)` plus a
designer base predicate `okBase`; every board-level closure obligation is
stated at the representative (no base in the `place` equality) and
transported to all bases by `Proofs/Invariants/BandShift.lean`. The drain
case is fully generic — the inhabitant only chooses *when* to drain
(`4 ≤ c` under `okBase`), never *how*.
-/

namespace Tetris

open Board Seam

/-- `bandDrain` is `SeamBridge`'s drain placement (they live in different
layers; the definition is duplicated, the equality is definitional). -/
theorem Board.bandDrain_eq_drainPl (w : ℕ) : bandDrain w = drainPl w := rfl

/-- A debt board over a well-anchored profile keeps its well column empty. -/
theorem Board.wellFree_debtBoard {cfg : GameConfig} {h : ℕ → ℕ}
    {ho : Option Coord} {w : ℕ} (hanch : h w = 0)
    (hho : ∀ x, ho = some x → x.1 ≠ w) :
    WellFree w (debtBoard cfg h ho) := by
  intro r hmem
  rw [mem_debtBoard] at hmem
  obtain ⟨hne, -, hlt'⟩ := hmem
  have hlt : r < h w := hlt'
  omega

/-- **The translation-quotient certificate.** A bag-indexed family of base-0
band representatives (profiles anchored at an empty well, debt ≤ 1) with a
designer base predicate, closed under one response per pending piece: either
a well-avoiding placement proven at the representative, or the generic
well drain. Inhabiting this proves Tetris solvable
(`tetrisSolvableValid_of_shiftCertificate`); it collapses the
`DebtCertificate` witness space from absolute boards to relative patterns. -/
structure ShiftCertificate where
  /-- The well column. -/
  well : ℕ
  hwell : well < GameConfig.standard.cols
  /-- The relative family: pending bag, base-0 profile, optional hole. -/
  Q : Bag → (ℕ → ℕ) → Option Coord → Prop
  /-- The designer base predicate: which band bases each state admits. -/
  okBase : Bag → (ℕ → ℕ) → Option Coord → ℕ → Prop
  /-- The empty board at a fresh bag, at base 0. -/
  init : Q Bag.full (fun _ => 0) none
  initBase : okBase Bag.full (fun _ => 0) none 0
  /-- Holes live in the band, in-field, strictly covered. -/
  cover : ∀ T ρ x, Q T ρ (some x) →
    x.1 ≠ well ∧ x.1 < GameConfig.standard.cols ∧ x.2 + 1 < ρ x.1
  /-- Admissible bases respect the ceiling. -/
  height : ∀ T ρ ho c, Q T ρ ho → okBase T ρ ho c →
    ∀ j < GameConfig.standard.cols,
      Board.bandLift well c ρ j ≤ GameConfig.standard.rows
  /-- Closure: every pending piece has a response — a well-avoiding
  placement proven at the representative, or the generic drain. -/
  step : ∀ T ρ ho c p, Q T ρ ho → okBase T ρ ho c → p ∈ T →
    (∃ pl ρ' ho', pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        AvoidsWell well pl ∧
        pl.place (Board.debtBoard GameConfig.standard ρ ho)
          = Board.debtBoard GameConfig.standard ρ' ho' ∧
        Q (T.draw p) ρ' ho' ∧ okBase (T.draw p) ρ' ho' c)
    ∨ (p = Piece.I ∧ 4 ≤ c ∧ Q (T.draw p) ρ ho ∧ okBase (T.draw p) ρ ho (c - 4))

namespace ShiftCertificate

/-- The absolute family realized by a shift certificate: every band lift of
every representative at an admissible base. -/
def toFamily (C : ShiftCertificate) :
    Bag → (ℕ → ℕ) → Option Coord → Prop := fun T h ho' =>
  ∃ ρ ho c, C.Q T ρ ho ∧ C.okBase T ρ ho c ∧
    h = Board.bandLift C.well c ρ ∧ ho' = holeLift c ho

/-- **The quotient reduction**: a shift certificate yields a debt certificate.
Every board-level obligation was discharged once at the representatives;
this constructor transports them to all admissible bases. -/
def toDebtCertificate (C : ShiftCertificate) : DebtCertificate where
  P := C.toFamily
  init :=
    ⟨fun _ => 0, none, 0, C.init, C.initBase, (bandLift_zero rfl).symm, rfl⟩
  cover := by
    rintro T h x ⟨ρ, ho, c, hQ, hok, rfl, hx⟩
    cases ho with
    | none => simp at hx
    | some x₀ =>
        rw [holeLift_some, Option.some.injEq] at hx
        subst hx
        obtain ⟨hxw, hxc, hxcov⟩ := C.cover T ρ x₀ hQ
        refine ⟨hxc, ?_⟩
        rw [bandLift_ne C.well c ρ hxw]
        omega
  height := by
    rintro T h ho' ⟨ρ, ho, c, hQ, hok, rfl, rfl⟩ j hj
    exact C.height T ρ ho c hQ hok j hj
  step := by
    rintro T h ho' p ⟨ρ, ho, c, hQ, hok, rfl, rfl⟩ hp
    have hhoW : ∀ x, ho = some x → x.1 ≠ C.well ∧ x.1 < GameConfig.standard.cols
        ∧ x.2 + 1 < ρ x.1 :=
      fun x hx => C.cover T ρ x (hx ▸ hQ)
    rcases C.step T ρ ho c p hQ hok hp with
      ⟨pl, ρ', ho'', hpiece, hvalid, havoidW, hrepEq, hQ', hok'⟩ |
      ⟨hpI, hc4, hQ', hok'⟩
    · -- Placement case: transport by T1, then applyStep = place (well open).
      refine ⟨pl, Board.bandLift C.well c ρ', holeLift c ho'', hpiece, hvalid,
        ?_, ρ', ho'', c, hQ', hok', rfl, rfl⟩
      have hlift := place_debtBoard_bandLift (w := C.well) (c := c)
        hvalid havoidW hhoW hrepEq
      have hWF : WellFree C.well (debtBoard GameConfig.standard
          (bandLift C.well c ρ') (holeLift c ho'')) := by
        refine wellFree_debtBoard (bandLift_well _ _ _) ?_
        rintro x hx
        cases ho'' with
        | none => simp at hx
        | some x₀ =>
            rw [holeLift_some, Option.some.injEq] at hx
            subst hx
            exact (C.cover (T.draw p) ρ' x₀ hQ').1
      rw [Placement.applyStep_eq_clearLines_place, hlift,
        clearLines_eq_self_of_no_fullRows GameConfig.standard
          (fullRows_eq_empty_of_wellFree C.hwell hWF)]
    · -- Drain case: T2 verbatim.
      subst hpI
      exact ⟨bandDrain C.well, Board.bandLift C.well (c - 4) ρ,
        holeLift (c - 4) ho, rfl, drainPl_valid C.hwell,
        drain_debtBoard_bandLift C.hwell hc4 hhoW,
        ρ, ho, c - 4, hQ', hok', rfl, rfl⟩

end ShiftCertificate

/-- Inhabiting the quotient certificate proves Tetris solvable. -/
theorem tetrisSolvableValid_of_shiftCertificate (C : ShiftCertificate) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_debtCertificate C.toDebtCertificate

end Tetris
