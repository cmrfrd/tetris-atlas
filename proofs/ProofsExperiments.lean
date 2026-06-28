import Proofs.Experiments.FiveBagReset
import Proofs.Experiments.SurfaceInvariant
import Proofs.Experiments.OnlineControlMachine
import Proofs.Experiments.OnlineReservoir
import Proofs.Experiments.EnergyGame
import Proofs.Experiments.HoleDebt
import Proofs.Experiments.HoleyCarrier
import Proofs.Experiments.HoleyTopical
import Proofs.Experiments.SurfaceFiber
import Proofs.Experiments.TopicalTetris
import Proofs.Experiments.WqoCarrier
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
- `Proofs/Experiments/AbstractSafe.lean` is intentionally **excluded**: it is an
  unbuilt scaffold whose proofs are `sorry` (the realization crux #66/#72).

Promoted out of here into the green `Proofs` library: `BagBurst`
(→ `Proofs.Combinatorics.BagBurst`) and `PieceCharge`
(→ `Proofs.Invariants.Charge`).
-/
