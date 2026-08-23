-- Tetris standard library — curated green re-export root.
-- Organized bottom-up by dependency layer (see LIBRARY.md). Base-axiom-clean
-- (propext, Classical.choice, Quot.sound); no `sorry`, no `native_decide`.
-- Active/floored research lives in the separate `ProofsExperiments` lake target.

-- Model — the game model
import Proofs.Model.Config
import Proofs.Model.Piece
import Proofs.Model.PieceGeometry
import Proofs.Model.Board
import Proofs.Model.Placement
import Proofs.Model.Bag
import Proofs.Model.Game

-- Combinatorics — board/column cell-counting, 7-bag renewal
import Proofs.Combinatorics.BoardCount
import Proofs.Combinatorics.BagBurst

-- Invariants — reachability/WF/clear invariants, holes, debt, surface fiber, WQO, state space
import Proofs.Invariants.StepInvariants
import Proofs.Invariants.Gameplay
import Proofs.Invariants.GameplayExtra
import Proofs.Invariants.ColumnCount
import Proofs.Invariants.Holes
import Proofs.Invariants.StateSpace
import Proofs.Invariants.Wqo
import Proofs.Invariants.HoleyCarrier
import Proofs.Invariants.SurfaceFiber
import Proofs.Invariants.SurfaceCalculus
import Proofs.Invariants.BagPlan
import Proofs.Invariants.HoleDebt
import Proofs.Invariants.HeightControl
import Proofs.Invariants.Skyline
import Proofs.Invariants.HoledSkyline
import Proofs.Invariants.BandShift
import Proofs.Invariants.BandMechanisms
import Proofs.Invariants.PlinthShift
import Proofs.Invariants.SlotAlgebra
import Proofs.Invariants.LaneCalculus
import Proofs.Invariants.Confluence
import Proofs.Invariants.ZoneView
import Proofs.Invariants.BagGrowth

-- Survival — policy / trace / cycle vocabulary
import Proofs.Survival.Survival
import Proofs.Survival.ClearRate
import Proofs.Survival.ClearDeviation
import Proofs.Survival.ClearRecurrence
import Proofs.Survival.ClearMix
import Proofs.Survival.Lasso

-- Safety — safe-set GFP, Atlas, the solvability reduction, computable iteration, obstructions
import Proofs.Safety.Adversarial
import Proofs.Safety.BagBlock
import Proofs.Safety.SafeSet
import Proofs.Safety.SurfaceLanguage
import Proofs.Safety.SeamBridge
import Proofs.Safety.SkylineInvariant
import Proofs.Safety.ShiftCertificate
import Proofs.Safety.BandSchedule
import Proofs.Safety.PlinthCert
import Proofs.Safety.CycleQuantum
import Proofs.Safety.MaximalAtlas
import Proofs.Safety.SafeIterate
import Proofs.Safety.SafeIterateFinite
import Proofs.Safety.Safety
import Proofs.Safety.SolverProperties
import Proofs.Safety.SolverFunction
import Proofs.Safety.SolverConstruction
