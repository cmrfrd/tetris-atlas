import Proofs.Model.Config
import Proofs.Model.Piece
import Proofs.Model.Board
import Proofs.Model.Placement
import Proofs.Model.Bag
import Proofs.Model.Game
import Proofs.Combinatorics.PieceGeometry
import Proofs.Combinatorics.BoardCount
import Proofs.Invariants.StepInvariants
import Proofs.Invariants.Gameplay
import Proofs.Invariants.GameplayExtra
import Proofs.Combinatorics.ColumnCount
import Proofs.Survival.Survival
import Proofs.Safety.Adversarial
import Proofs.Safety.SafeSet
import Proofs.Safety.SafeIterate
import Proofs.Safety.SafeIterateFinite
import Proofs.Safety.Safety
import Proofs.Invariants.StateSpace
import Proofs.Invariants.Holes
-- Promoted from Experiments/: base-axiom-clean, native_decide-free reusable results.
-- (Active research lives in the separate `ProofsExperiments` lake_lib, not this green target.)
import Proofs.Combinatorics.BagBurst
import Proofs.Invariants.Wqo
import Proofs.Invariants.HoleyCarrier
import Proofs.Invariants.SurfaceFiber
import Proofs.Invariants.HoleDebt
