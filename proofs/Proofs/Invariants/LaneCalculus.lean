import Proofs.Invariants.Skyline
import Proofs.Invariants.HoledSkyline
import Proofs.Invariants.BandMechanisms

/-!
# The lane calculus: every local flush transition, at arbitrary profiles

The complete flush-placement mechanism pack for the coupled band game. The
existing library proves the self-reproducing 2-wide windows
(`place_vertS_skyline`, `place_vertZ_skyline`, `place_O_pair`) and the
flat-surface witnesses at *globally* flat profiles; this file supplies every
remaining local flush landing at an **arbitrary** surrounding profile, each an
instance of `place_flush_skyline`:

- the 3-wide step lanes: horizontal S on `(k, k, k+1)` and horizontal Z on
  `(k+1, k, k)` — the roving-step currency that lets S/Z consume lane shapes
  (and, dually, lets a standing ±1 step be flattened by its mirror piece
  using a runway column);
- the notch fills: T rot 0 on `(k+1, k, k+1)`, L rot 2 on `(k, k+1, k+1)`,
  J rot 2 on `(k+1, k+1, k)` — each restores a local flat;
- the T alternator at general profiles: T rot 1 consumes an S-step and
  leaves a Z-step; T rot 3 the mirror;
- the local flat-3 landings T rot 2 / L rot 0 / J rot 0 and the local
  flat-4 horizontal I (the global-flat witnesses, freed from the
  all-columns-equal hypothesis);
- the pair economy: L rot 1 / J rot 3 turn a flat pair into a ±2 step and
  L rot 3 / J rot 1 flatten it back (the `SlotAlgebra` exclusive currency,
  as skyline equalities).

Together with the three existing window mechanisms and `place_vertI_col`,
every flush landing shape in the slot algebra now has its transition law;
any hand-designed band schedule can be certified from these alone.
-/

namespace Tetris
namespace Board

/-! ## Drop profiles missing from the library -/

/-- T rot 0 drop profile: the notch filler. -/
theorem shapeUp_notchT (c : ℕ) :
    ({ piece := Piece.T, rot := 0, col := c } : Placement).shapeUp
      = {((0 : ℕ), (1 : ℕ)), (1, 1), (1, 0), (2, 1)} := by
  change Piece.shapeUp Piece.T 0 = _
  decide

/-- T rot 1 drop profile: column 0 row 1, column 1 rows 0–2. -/
theorem shapeUp_vertT1 (c : ℕ) :
    ({ piece := Piece.T, rot := 1, col := c } : Placement).shapeUp
      = {((0 : ℕ), (1 : ℕ)), (1, 2), (1, 1), (1, 0)} := by
  change Piece.shapeUp Piece.T 1 = _
  decide

/-- T rot 3 drop profile: column 0 rows 0–2, column 1 row 1. -/
theorem shapeUp_vertT3 (c : ℕ) :
    ({ piece := Piece.T, rot := 3, col := c } : Placement).shapeUp
      = {((0 : ℕ), (2 : ℕ)), (0, 1), (0, 0), (1, 1)} := by
  change Piece.shapeUp Piece.T 3 = _
  decide

/-- L rot 1 drop profile: column 0 rows 0–2, column 1 row 0. -/
theorem shapeUp_vertL1 (c : ℕ) :
    ({ piece := Piece.L, rot := 1, col := c } : Placement).shapeUp
      = {((0 : ℕ), (2 : ℕ)), (0, 1), (0, 0), (1, 0)} := by
  change Piece.shapeUp Piece.L 1 = _
  decide

/-- L rot 2 drop profile: the left-notch filler. -/
theorem shapeUp_notchL (c : ℕ) :
    ({ piece := Piece.L, rot := 2, col := c } : Placement).shapeUp
      = {((0 : ℕ), (1 : ℕ)), (0, 0), (1, 1), (2, 1)} := by
  change Piece.shapeUp Piece.L 2 = _
  decide

/-- L rot 3 drop profile: column 0 row 2, column 1 rows 0–2. -/
theorem shapeUp_vertL3 (c : ℕ) :
    ({ piece := Piece.L, rot := 3, col := c } : Placement).shapeUp
      = {((0 : ℕ), (2 : ℕ)), (1, 2), (1, 1), (1, 0)} := by
  change Piece.shapeUp Piece.L 3 = _
  decide

/-- J rot 1 drop profile: column 0 rows 0–2, column 1 row 2. -/
theorem shapeUp_vertJ1 (c : ℕ) :
    ({ piece := Piece.J, rot := 1, col := c } : Placement).shapeUp
      = {((0 : ℕ), (2 : ℕ)), (0, 1), (0, 0), (1, 2)} := by
  change Piece.shapeUp Piece.J 1 = _
  decide

/-- J rot 2 drop profile: the right-notch filler. -/
theorem shapeUp_notchJ (c : ℕ) :
    ({ piece := Piece.J, rot := 2, col := c } : Placement).shapeUp
      = {((0 : ℕ), (1 : ℕ)), (1, 1), (2, 1), (2, 0)} := by
  change Piece.shapeUp Piece.J 2 = _
  decide

/-- J rot 3 drop profile: column 0 row 0, column 1 rows 0–2. -/
theorem shapeUp_vertJ3 (c : ℕ) :
    ({ piece := Piece.J, rot := 3, col := c } : Placement).shapeUp
      = {((0 : ℕ), (0 : ℕ)), (1, 2), (1, 1), (1, 0)} := by
  change Piece.shapeUp Piece.J 3 = _
  decide

/-! ## The 3-wide step lanes: horizontal S and Z at flush shapes -/

/-- **Horizontal S on a step lane.** On `(k, k, k+1)` — a flat pair with an
up-step at the right — the horizontal S seats flush, no hole: the lane
becomes `(k+1, k+2, k+2)`. -/
theorem place_horizS_step {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc2 : c + 2 < cfg.cols) (h1 : h (c + 1) = h c) (h2 : h (c + 2) = h c + 1) :
    Placement.place (skyline cfg h) { piece := Piece.S, rot := 0, col := c }
      = skyline cfg (Function.update (Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1)) := by
  refine place_flush_skyline (w := 3) (off := h c)
    (bot := fun i => i / 2) (top := fun i => min i 1)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_horizS c 0 (by decide)]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · change c + i < cfg.cols
    omega
  · change h (c + i) = h c + i / 2
    interval_cases i <;> (try simp) <;> omega
  · change Function.update (Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1) (c + i)
        = h c + min i 1 + 1
    interval_cases i
    · change Function.update (Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1) c
          = h c + min 0 1 + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 2),
        Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]; omega
    · rw [Function.update_of_ne (by omega : c + 1 ≠ c + 2), Function.update_self]; omega
    · rw [Function.update_self]; omega
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    have hj2 : j ≠ c + 2 := hj 2 (by omega)
    change Function.update (Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1) j = h j
    rw [Function.update_of_ne hj2, Function.update_of_ne hj1,
      Function.update_of_ne hj0]

/-- **Horizontal Z on a step lane** (mirror). On `(k+1, k, k)` — a down-step
with a flat runway at the right — the horizontal Z seats flush, no hole:
the lane becomes `(k+2, k+2, k+1)`. Note the special case where the down
step is a standing S-window: the Z *flattens* it and leaves a fresh S-step
one column to the right. -/
theorem place_horizZ_step {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc2 : c + 2 < cfg.cols) (h0 : h c = h (c + 1) + 1) (h2 : h (c + 2) = h (c + 1)) :
    Placement.place (skyline cfg h) { piece := Piece.Z, rot := 0, col := c }
      = skyline cfg (Function.update (Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1)) := by
  refine place_flush_skyline (w := 3) (off := h (c + 1))
    (bot := fun i => 1 - i) (top := fun i => 1 - i / 2)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_horizZ c 0 (by decide)]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · change c + i < cfg.cols
    omega
  · change h (c + i) = h (c + 1) + (1 - i)
    interval_cases i <;> (try simp) <;> omega
  · change Function.update (Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1) (c + i)
        = h (c + 1) + (1 - i / 2) + 1
    interval_cases i
    · change Function.update (Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1) c
          = h (c + 1) + (1 - 0 / 2) + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 2),
        Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]; omega
    · rw [Function.update_of_ne (by omega : c + 1 ≠ c + 2), Function.update_self]
    · rw [Function.update_self]; omega
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    have hj2 : j ≠ c + 2 := hj 2 (by omega)
    change Function.update (Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1) j = h j
    rw [Function.update_of_ne hj2, Function.update_of_ne hj1,
      Function.update_of_ne hj0]

/-! ## The notch fills: T, L, J each restore a local flat -/

/-- **T rot 0 fills a notch.** On `(k+1, k, k+1)` the T seats flush with its
stem in the notch: the lane flattens to `(k+2, k+2, k+2)`. -/
theorem place_notchT {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc2 : c + 2 < cfg.cols) (h0 : h c = h (c + 1) + 1) (h2 : h (c + 2) = h (c + 1) + 1) :
    Placement.place (skyline cfg h) { piece := Piece.T, rot := 0, col := c }
      = skyline cfg (Function.update (Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1)) := by
  refine place_flush_skyline (w := 3) (off := h (c + 1))
    (bot := fun i => 1 - i % 2) (top := fun _ => 1)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_notchT]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · change c + i < cfg.cols
    omega
  · change h (c + i) = h (c + 1) + (1 - i % 2)
    interval_cases i <;> (try simp) <;> omega
  · change Function.update (Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1) (c + i)
        = h (c + 1) + 1 + 1
    interval_cases i
    · change Function.update (Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1) c
          = h (c + 1) + 1 + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 2),
        Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]; omega
    · rw [Function.update_of_ne (by omega : c + 1 ≠ c + 2), Function.update_self]
    · rw [Function.update_self]; omega
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    have hj2 : j ≠ c + 2 := hj 2 (by omega)
    change Function.update (Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1) j = h j
    rw [Function.update_of_ne hj2, Function.update_of_ne hj1,
      Function.update_of_ne hj0]

/-- **L rot 2 fills a left notch.** On `(k, k+1, k+1)` the L seats flush,
foot in the low column: the lane flattens to `(k+2, k+2, k+2)`. -/
theorem place_notchL {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc2 : c + 2 < cfg.cols) (h1 : h (c + 1) = h c + 1) (h2 : h (c + 2) = h c + 1) :
    Placement.place (skyline cfg h) { piece := Piece.L, rot := 2, col := c }
      = skyline cfg (Function.update (Function.update (Function.update h
          c (h c + 2)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 1)) := by
  refine place_flush_skyline (w := 3) (off := h c)
    (bot := fun i => min i 1) (top := fun _ => 1)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_notchL]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · change c + i < cfg.cols
    omega
  · change h (c + i) = h c + min i 1
    interval_cases i <;> (try simp) <;> omega
  · change Function.update (Function.update (Function.update h
        c (h c + 2)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 1) (c + i)
        = h c + 1 + 1
    interval_cases i
    · change Function.update (Function.update (Function.update h
          c (h c + 2)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 1) c
          = h c + 1 + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 2),
        Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]
    · rw [Function.update_of_ne (by omega : c + 1 ≠ c + 2), Function.update_self]; omega
    · rw [Function.update_self]; omega
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    have hj2 : j ≠ c + 2 := hj 2 (by omega)
    change Function.update (Function.update (Function.update h
        c (h c + 2)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 1) j = h j
    rw [Function.update_of_ne hj2, Function.update_of_ne hj1,
      Function.update_of_ne hj0]

/-- **J rot 2 fills a right notch** (mirror). On `(k+1, k+1, k)` the J seats
flush, foot in the low column: the lane flattens to `(k+2, k+2, k+2)`. -/
theorem place_notchJ {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc2 : c + 2 < cfg.cols) (h0 : h c = h (c + 2) + 1) (h1 : h (c + 1) = h (c + 2) + 1) :
    Placement.place (skyline cfg h) { piece := Piece.J, rot := 2, col := c }
      = skyline cfg (Function.update (Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 2)) := by
  refine place_flush_skyline (w := 3) (off := h (c + 2))
    (bot := fun i => 1 - i / 2) (top := fun _ => 1)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_notchJ]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · change c + i < cfg.cols
    omega
  · change h (c + i) = h (c + 2) + (1 - i / 2)
    interval_cases i <;> (try simp) <;> omega
  · change Function.update (Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 2) (c + i)
        = h (c + 2) + 1 + 1
    interval_cases i
    · change Function.update (Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 2) c
          = h (c + 2) + 1 + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 2),
        Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]; omega
    · rw [Function.update_of_ne (by omega : c + 1 ≠ c + 2), Function.update_self]; omega
    · rw [Function.update_self]
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    have hj2 : j ≠ c + 2 := hj 2 (by omega)
    change Function.update (Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 2) j = h j
    rw [Function.update_of_ne hj2, Function.update_of_ne hj1,
      Function.update_of_ne hj0]

/-! ## The T alternator at general profiles -/

/-- **T rot 1 consumes an S-step and leaves a Z-step.** On `h c = h (c+1) + 1`
the vertical T seats flush: heights become `(k+2, k+3)` from `(k+1, k)` —
the ±1 step flips orientation. -/
theorem place_stepT_toZ {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc1 : c + 1 < cfg.cols) (hstep : h c = h (c + 1) + 1) :
    Placement.place (skyline cfg h) { piece := Piece.T, rot := 1, col := c }
      = skyline cfg (Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 3)) := by
  refine place_flush_skyline (w := 2) (off := h (c + 1))
    (bot := fun i => 1 - i) (top := fun i => 1 + i)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_vertT1]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · change c + i < cfg.cols
    omega
  · change h (c + i) = h (c + 1) + (1 - i)
    interval_cases i
    · simp
      omega
    · omega
  · change Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 3) (c + i) = h (c + 1) + (1 + i) + 1
    interval_cases i
    · change Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 3) c = h (c + 1) + (1 + 0) + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]; omega
    · rw [Function.update_self]
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    change Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 3) j = h j
    rw [Function.update_of_ne hj1, Function.update_of_ne hj0]

/-- **T rot 3 consumes a Z-step and leaves an S-step** (mirror). On
`h (c+1) = h c + 1` the vertical T seats flush: heights become `(k+3, k+2)`
from `(k, k+1)`. -/
theorem place_stepT_toS {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc1 : c + 1 < cfg.cols) (hstep : h (c + 1) = h c + 1) :
    Placement.place (skyline cfg h) { piece := Piece.T, rot := 3, col := c }
      = skyline cfg (Function.update (Function.update h
          c (h c + 3)) (c + 1) (h (c + 1) + 1)) := by
  refine place_flush_skyline (w := 2) (off := h c)
    (bot := fun i => i) (top := fun i => 2 - i)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_vertT3]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · change c + i < cfg.cols
    omega
  · change h (c + i) = h c + i
    interval_cases i
    · simp
    · omega
  · change Function.update (Function.update h
        c (h c + 3)) (c + 1) (h (c + 1) + 1) (c + i) = h c + (2 - i) + 1
    interval_cases i
    · change Function.update (Function.update h
          c (h c + 3)) (c + 1) (h (c + 1) + 1) c = h c + (2 - 0) + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]
    · rw [Function.update_self]; omega
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    change Function.update (Function.update h
        c (h c + 3)) (c + 1) (h (c + 1) + 1) j = h j
    rw [Function.update_of_ne hj1, Function.update_of_ne hj0]

/-! ## Local flat landings: T, L, J, I on a flat stretch of any profile -/

/-- **T rot 2 on a local flat-3**: leaves the bump `(k+1, k+2, k+1)`. -/
theorem place_flatT_lane {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc2 : c + 2 < cfg.cols) (h1 : h (c + 1) = h c) (h2 : h (c + 2) = h c) :
    Placement.place (skyline cfg h) { piece := Piece.T, rot := 2, col := c }
      = skyline cfg (Function.update (Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1)) := by
  refine place_flush_skyline (w := 3) (off := h c)
    (bot := fun _ => 0) (top := fun i => i % 2)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_flatT]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · change c + i < cfg.cols
    omega
  · change h (c + i) = h c + 0
    interval_cases i <;> (try simp) <;> omega
  · change Function.update (Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1) (c + i)
        = h c + i % 2 + 1
    interval_cases i
    · change Function.update (Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1) c
          = h c + 0 % 2 + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 2),
        Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]
    · rw [Function.update_of_ne (by omega : c + 1 ≠ c + 2), Function.update_self]; omega
    · rw [Function.update_self]; omega
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    have hj2 : j ≠ c + 2 := hj 2 (by omega)
    change Function.update (Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 2)) (c + 2) (h (c + 2) + 1) j = h j
    rw [Function.update_of_ne hj2, Function.update_of_ne hj1,
      Function.update_of_ne hj0]

/-- **L rot 0 on a local flat-3**: leaves `(k+1, k+1, k+2)` — an S-h
receptor one row up. -/
theorem place_flatL_lane {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc2 : c + 2 < cfg.cols) (h1 : h (c + 1) = h c) (h2 : h (c + 2) = h c) :
    Placement.place (skyline cfg h) { piece := Piece.L, rot := 0, col := c }
      = skyline cfg (Function.update (Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 2)) := by
  refine place_flush_skyline (w := 3) (off := h c)
    (bot := fun _ => 0) (top := fun i => i / 2)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_flatL]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · change c + i < cfg.cols
    omega
  · change h (c + i) = h c + 0
    interval_cases i <;> (try simp) <;> omega
  · change Function.update (Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 2) (c + i)
        = h c + i / 2 + 1
    interval_cases i
    · change Function.update (Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 2) c
          = h c + 0 / 2 + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 2),
        Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]
    · rw [Function.update_of_ne (by omega : c + 1 ≠ c + 2), Function.update_self]; omega
    · rw [Function.update_self]; omega
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    have hj2 : j ≠ c + 2 := hj 2 (by omega)
    change Function.update (Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 2) j = h j
    rw [Function.update_of_ne hj2, Function.update_of_ne hj1,
      Function.update_of_ne hj0]

/-- **J rot 0 on a local flat-3** (mirror): leaves `(k+2, k+1, k+1)` — a Z-h
receptor one row up. -/
theorem place_flatJ_lane {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc2 : c + 2 < cfg.cols) (h1 : h (c + 1) = h c) (h2 : h (c + 2) = h c) :
    Placement.place (skyline cfg h) { piece := Piece.J, rot := 0, col := c }
      = skyline cfg (Function.update (Function.update (Function.update h
          c (h c + 2)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 1)) := by
  refine place_flush_skyline (w := 3) (off := h c)
    (bot := fun _ => 0) (top := fun i => 1 - i)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_flatJ]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · change c + i < cfg.cols
    omega
  · change h (c + i) = h c + 0
    interval_cases i <;> (try simp) <;> omega
  · change Function.update (Function.update (Function.update h
        c (h c + 2)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 1) (c + i)
        = h c + (1 - i) + 1
    interval_cases i
    · change Function.update (Function.update (Function.update h
          c (h c + 2)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 1) c
          = h c + (1 - 0) + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 2),
        Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]
    · rw [Function.update_of_ne (by omega : c + 1 ≠ c + 2), Function.update_self]; omega
    · rw [Function.update_self]; omega
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    have hj2 : j ≠ c + 2 := hj 2 (by omega)
    change Function.update (Function.update (Function.update h
        c (h c + 2)) (c + 1) (h (c + 1) + 1)) (c + 2) (h (c + 2) + 1) j = h j
    rw [Function.update_of_ne hj2, Function.update_of_ne hj1,
      Function.update_of_ne hj0]

/-- **Horizontal I on a local flat-4**: all four columns rise 1. -/
theorem place_horizI_lane {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc3 : c + 3 < cfg.cols) (h1 : h (c + 1) = h c) (h2 : h (c + 2) = h c)
    (h3 : h (c + 3) = h c) :
    Placement.place (skyline cfg h) { piece := Piece.I, rot := 0, col := c }
      = skyline cfg (Function.update (Function.update (Function.update
          (Function.update h c (h c + 1)) (c + 1) (h (c + 1) + 1))
          (c + 2) (h (c + 2) + 1)) (c + 3) (h (c + 3) + 1)) := by
  refine place_flush_skyline (w := 4) (off := h c)
    (bot := fun _ => 0) (top := fun _ => 0)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_horizI]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · change c + i < cfg.cols
    omega
  · change h (c + i) = h c + 0
    interval_cases i <;> (try simp) <;> omega
  · change Function.update (Function.update (Function.update
        (Function.update h c (h c + 1)) (c + 1) (h (c + 1) + 1))
        (c + 2) (h (c + 2) + 1)) (c + 3) (h (c + 3) + 1) (c + i)
        = h c + 0 + 1
    interval_cases i
    · change Function.update (Function.update (Function.update
          (Function.update h c (h c + 1)) (c + 1) (h (c + 1) + 1))
          (c + 2) (h (c + 2) + 1)) (c + 3) (h (c + 3) + 1) c = h c + 0 + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 3),
        Function.update_of_ne (by omega : c ≠ c + 2),
        Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]
    · rw [Function.update_of_ne (by omega : c + 1 ≠ c + 3),
        Function.update_of_ne (by omega : c + 1 ≠ c + 2), Function.update_self]; omega
    · rw [Function.update_of_ne (by omega : c + 2 ≠ c + 3), Function.update_self]; omega
    · rw [Function.update_self]; omega
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    have hj2 : j ≠ c + 2 := hj 2 (by omega)
    have hj3 : j ≠ c + 3 := hj 3 (by omega)
    change Function.update (Function.update (Function.update
        (Function.update h c (h c + 1)) (c + 1) (h (c + 1) + 1))
        (c + 2) (h (c + 2) + 1)) (c + 3) (h (c + 3) + 1) j = h j
    rw [Function.update_of_ne hj3, Function.update_of_ne hj2,
      Function.update_of_ne hj1, Function.update_of_ne hj0]

/-! ## The pair economy: L and J on flat pairs and ±2 steps -/

/-- **L rot 1 on a flat pair**: leaves the left-high ±2 step `(k+3, k+1)`. -/
theorem place_pairL {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc1 : c + 1 < cfg.cols) (h1 : h (c + 1) = h c) :
    Placement.place (skyline cfg h) { piece := Piece.L, rot := 1, col := c }
      = skyline cfg (Function.update (Function.update h
          c (h c + 3)) (c + 1) (h (c + 1) + 1)) := by
  refine place_flush_skyline (w := 2) (off := h c)
    (bot := fun _ => 0) (top := fun i => 2 - 2 * i)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_vertL1]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · change c + i < cfg.cols
    omega
  · change h (c + i) = h c + 0
    interval_cases i
    · simp
    · omega
  · change Function.update (Function.update h
        c (h c + 3)) (c + 1) (h (c + 1) + 1) (c + i) = h c + (2 - 2 * i) + 1
    interval_cases i
    · change Function.update (Function.update h
          c (h c + 3)) (c + 1) (h (c + 1) + 1) c = h c + (2 - 2 * 0) + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]
    · rw [Function.update_self]; omega
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    change Function.update (Function.update h
        c (h c + 3)) (c + 1) (h (c + 1) + 1) j = h j
    rw [Function.update_of_ne hj1, Function.update_of_ne hj0]

/-- **L rot 3 flattens a left-high ±2 step**: on `h c = h (c+1) + 2` both
columns rise to `k+3` — the step closes into a flat pair. -/
theorem place_fillL {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc1 : c + 1 < cfg.cols) (h0 : h c = h (c + 1) + 2) :
    Placement.place (skyline cfg h) { piece := Piece.L, rot := 3, col := c }
      = skyline cfg (Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 3)) := by
  refine place_flush_skyline (w := 2) (off := h (c + 1))
    (bot := fun i => 2 - 2 * i) (top := fun _ => 2)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_vertL3]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · change c + i < cfg.cols
    omega
  · change h (c + i) = h (c + 1) + (2 - 2 * i)
    interval_cases i
    · simp
      omega
    · omega
  · change Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 3) (c + i) = h (c + 1) + 2 + 1
    interval_cases i
    · change Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 3) c = h (c + 1) + 2 + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]; omega
    · rw [Function.update_self]
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    change Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 3) j = h j
    rw [Function.update_of_ne hj1, Function.update_of_ne hj0]

/-- **J rot 3 on a flat pair** (mirror of L rot 1): leaves the right-high
±2 step `(k+1, k+3)`. -/
theorem place_pairJ {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc1 : c + 1 < cfg.cols) (h1 : h (c + 1) = h c) :
    Placement.place (skyline cfg h) { piece := Piece.J, rot := 3, col := c }
      = skyline cfg (Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 3)) := by
  refine place_flush_skyline (w := 2) (off := h c)
    (bot := fun _ => 0) (top := fun i => 2 * i)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_vertJ3]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · change c + i < cfg.cols
    omega
  · change h (c + i) = h c + 0
    interval_cases i
    · simp
    · omega
  · change Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 3) (c + i) = h c + 2 * i + 1
    interval_cases i
    · change Function.update (Function.update h
          c (h c + 1)) (c + 1) (h (c + 1) + 3) c = h c + 2 * 0 + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]
    · rw [Function.update_self]; omega
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    change Function.update (Function.update h
        c (h c + 1)) (c + 1) (h (c + 1) + 3) j = h j
    rw [Function.update_of_ne hj1, Function.update_of_ne hj0]

/-- **J rot 1 flattens a right-high ±2 step** (mirror of L rot 3): on
`h (c+1) = h c + 2` both columns rise to `k+3`. -/
theorem place_fillJ {cfg : GameConfig} {h : ℕ → ℕ} {c : ℕ}
    (hc1 : c + 1 < cfg.cols) (h1 : h (c + 1) = h c + 2) :
    Placement.place (skyline cfg h) { piece := Piece.J, rot := 1, col := c }
      = skyline cfg (Function.update (Function.update h
          c (h c + 3)) (c + 1) (h (c + 1) + 1)) := by
  refine place_flush_skyline (w := 2) (off := h c)
    (bot := fun i => 2 * i) (top := fun _ => 2)
    (fun i ρ => ?_) (by omega) (fun i _ => by dsimp only; omega)
    (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) (fun j hj => ?_)
  · rw [shapeUp_vertJ1]
    simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
    omega
  · change c + i < cfg.cols
    omega
  · change h (c + i) = h c + 2 * i
    interval_cases i
    · simp
    · omega
  · change Function.update (Function.update h
        c (h c + 3)) (c + 1) (h (c + 1) + 1) (c + i) = h c + 2 + 1
    interval_cases i
    · change Function.update (Function.update h
          c (h c + 3)) (c + 1) (h (c + 1) + 1) c = h c + 2 + 1
      rw [Function.update_of_ne (by omega : c ≠ c + 1), Function.update_self]
    · rw [Function.update_self]; omega
  · have hj0 : j ≠ c := by simpa using hj 0 (by omega)
    have hj1 : j ≠ c + 1 := hj 1 (by omega)
    change Function.update (Function.update h
        c (h c + 3)) (c + 1) (h (c + 1) + 1) j = h j
    rw [Function.update_of_ne hj1, Function.update_of_ne hj0]

end Board
end Tetris
