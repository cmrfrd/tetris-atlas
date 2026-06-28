import Proofs.Experiments.FiveBagReset
import Proofs.Experiments.SurfaceInvariant
import Proofs.Experiments.OnlineControlMachine
import Proofs.Experiments.OnlineReservoir
import Proofs.Experiments.EnergyGame
import Proofs.Experiments.HoleyTopical
import Proofs.Experiments.TopicalTetris
import Proofs.Experiments.PieceCharge
-- Route-specific reductions whose PRIMITIVES were promoted to the green library
-- (Proofs.Invariants.Wqo / Proofs.Invariants.HoleyCarrier). These halves stay here
-- because they conclude `TetrisSolvableValid` and depend on `SafeSet`.
import Proofs.Experiments.WqoCarrier
import Proofs.Experiments.HoleyCarrier
import Proofs.Experiments.Scratch.RoutePhase.Probe
import Proofs.Experiments.Scratch.RouteSurface.Probe

/-!
# ProofsExperiments — research routes (NOT the green standard library)

This is the manual lake target for active/floored research. It is **deliberately
separate** from the `Proofs` default target so the standard library stays
`native_decide`-free and base-axioms-only.

- These files may use `native_decide` (→ `Lean.ofReduceBool`) and import the
  large carrier/surface scaffolds. That is fine here; it must never leak into
  `Proofs`.
- Build explicitly with: `lake build ProofsExperiments` (foreground only).
- The `sorry` scaffold lives in `Proofs/Archive/AbstractSafe.lean` — built by
  **neither** lake target (record only; its proofs are the realization crux #66/#72).

Promoted into the green `Proofs` library: `BagBurst` (→ `Combinatorics.BagBurst`),
the WQO/hole/fiber/debt **primitives** (→ `Invariants.{Wqo,HoleyCarrier,SurfaceFiber,
HoleDebt}`). Demoted back here: `PieceCharge` (clean and classic, but not yet used by
any survival argument — re-promote when it is).
-/
