import Proofs.Experiments.SurfaceInvariant

/-!
# Certified online control machines

This experiment factors the positive Atlas route through one reusable interface.

Instead of proving a large theorem directly about a particular reservoir surface,
we describe a static online controller as a finite-state machine:

* an abstract state `q` records the controller's phase;
* `carrier q T b` says that concrete board `b` is represented by phase `q`
  while the current bag is `T`;
* `move q T p` is the placement the controller plays when the adversary draws
  piece `p`;
* `next q T p` is the next abstract phase.

The core proof obligation is `step`: whenever `carrier q T b` holds and `p ∈ T`,
playing `move q T p` lands the real Tetris board in
`carrier (next q T p) (T.draw p)`.  The theorem `OnlineControlMachine.solves`
then turns any such certified machine into `TetrisSolvableValid`.

This is meant to be the common front door for the current approaches:

* the reservoir phase graph should become one instance of this interface;
* a hand-written static algorithm can be verified by filling these fields;
* a future Lean-computed finite table can also target this interface.

The proof below is deliberately generic.  It contains no reservoir geometry and
does not solve Tetris by itself; it only states the exact contract a candidate
controller must satisfy.
-/

namespace Tetris
namespace Experiments

noncomputable section

/-- A certified static online controller.

The machine state is abstract.  It may be a small inductive phase type, a
normalised finite board profile, or a wrapper around a checked transition table.
The `solves` theorem below does not need finiteness; finiteness is recorded
separately by `FiniteOnlineControlMachine` for table-oriented instantiations. -/
structure OnlineControlMachine where
  State : Type
  initState : State
  carrier : State → Bag → Board → Prop
  move : State → Bag → Piece → Placement
  next : State → Bag → Piece → State

  init :
    carrier initState Bag.full Board.empty

  height :
    ∀ q T b, carrier q T b →
      ∀ j, Board.colHeight b j ≤ GameConfig.standard.rows

  piece :
    ∀ q T p, (move q T p).piece = p

  valid :
    ∀ q T b p, carrier q T b → p ∈ T →
      (move q T p).Valid GameConfig.standard

  step :
    ∀ q T b p, carrier q T b → p ∈ T →
      carrier (next q T p) (T.draw p)
        (Placement.applyStep GameConfig.standard b { move q T p with piece := p })

namespace OnlineControlMachine

/-- The concrete board-level invariant induced by a machine: the board is
represented by at least one abstract machine state. -/
def Carrier (M : OnlineControlMachine) (T : Bag) (b : Board) : Prop :=
  ∃ q : M.State, M.carrier q T b

/-- Pick a representing state for the current concrete `(bag, board)` pair.
Outside the certified carrier we return `initState`; the solver theorem only
uses the chosen state on carrier states. -/
noncomputable def currentState (M : OnlineControlMachine) (T : Bag) (b : Board) : M.State := by
  classical
  exact if h : Carrier M T b then Classical.choose h else M.initState

theorem currentState_carrier {M : OnlineControlMachine} {T : Bag} {b : Board}
    (h : Carrier M T b) :
    M.carrier (M.currentState T b) T b := by
  classical
  unfold currentState
  rw [dif_pos h]
  exact Classical.choose_spec h

/-- The concrete Tetris strategy extracted from a certified machine. -/
noncomputable def strategy (M : OnlineControlMachine) : Bag → Board → Piece → Placement :=
  fun T b p => M.move (M.currentState T b) T p

theorem carrier_init (M : OnlineControlMachine) :
    Carrier M Bag.full Board.empty :=
  ⟨M.initState, M.init⟩

theorem carrier_height (M : OnlineControlMachine) {T : Bag} {b : Board}
    (h : Carrier M T b) :
    ∀ j, Board.colHeight b j ≤ GameConfig.standard.rows := by
  rcases h with ⟨q, hq⟩
  exact M.height q T b hq

theorem strategy_piece (M : OnlineControlMachine) (T : Bag) (b : Board) (p : Piece) :
    (M.strategy T b p).piece = p := by
  unfold strategy
  exact M.piece (M.currentState T b) T p

theorem strategy_valid (M : OnlineControlMachine) {T : Bag} {b : Board} {p : Piece}
    (h : Carrier M T b) (hp : p ∈ T) :
    (M.strategy T b p).Valid GameConfig.standard := by
  unfold strategy
  exact M.valid (M.currentState T b) T b p (currentState_carrier h) hp

theorem strategy_step (M : OnlineControlMachine) {T : Bag} {b : Board} {p : Piece}
    (h : Carrier M T b) (hp : p ∈ T) :
    Carrier M (T.draw p)
      (Placement.applyStep GameConfig.standard b { M.strategy T b p with piece := p }) := by
  refine ⟨M.next (M.currentState T b) T p, ?_⟩
  unfold strategy
  exact M.step (M.currentState T b) T b p (currentState_carrier h) hp

/-- Main front-door theorem: any certified online control machine gives the
project's strengthened valid-solvability theorem. -/
theorem solves (M : OnlineControlMachine) : TetrisSolvableValid :=
  tetrisSolvableValid_of_strategy_bagInvariant
    M.strategy
    (Carrier M)
    M.strategy_piece
    M.carrier_init
    (fun _ _ _ h hp => M.strategy_step h hp)
    (fun _ _ _ h hp => M.strategy_valid h hp)
    (fun _ _ h => M.carrier_height h)

end OnlineControlMachine

/-- Metadata wrapper for instantiations whose abstract state space is actually
finite.  The generic `OnlineControlMachine.solves` theorem does not require this
field, but finite state and decidable equality are what a checked transition
table or Lean-computed controller will normally provide. -/
structure FiniteOnlineControlMachine where
  toMachine : OnlineControlMachine
  finiteState : Nonempty (Fintype toMachine.State)
  decidableStateEq : Nonempty (DecidableEq toMachine.State)

namespace FiniteOnlineControlMachine

theorem solves (M : FiniteOnlineControlMachine) : TetrisSolvableValid :=
  M.toMachine.solves

end FiniteOnlineControlMachine

end
end Experiments
end Tetris
