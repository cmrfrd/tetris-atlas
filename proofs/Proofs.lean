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
import Proofs.Invariants.HoleDebt
import Proofs.Invariants.HeightControl
import Proofs.Invariants.Skyline

-- Survival — policy / trace / cycle vocabulary
import Proofs.Survival.Survival

-- Safety — safe-set GFP, Atlas, the solvability reduction, computable iteration, obstructions
import Proofs.Safety.Adversarial
import Proofs.Safety.BagBlock
import Proofs.Safety.SafeSet
import Proofs.Safety.SafeIterate
import Proofs.Safety.SafeIterateFinite
import Proofs.Safety.Safety
import Proofs.Safety.SolverProperties
import Proofs.Safety.SolverFunction
import Proofs.Safety.SolverConstruction
