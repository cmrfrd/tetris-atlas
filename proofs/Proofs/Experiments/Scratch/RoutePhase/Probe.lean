import Proofs.Experiments.OnlineReservoir

/-!
# RoutePhase Probe

Empirical probe of whether extending the `OnlineReservoir` phase graph
(or discharging `frontier_step`) is mechanical replication of the existing
edge-building pattern, or hits a real geometric construction wall.

We do NOT edit OnlineReservoir.lean. We re-derive analogous facts in isolation.
-/

namespace RoutePhaseProbe

open Tetris
open Tetris.OnlineReservoir
open Tetris.OnlineReservoir.PhaseGraph

/-! ## TARGET 1: replicate an existing table edge as a standalone lemma.

The existing chain proves, for an edge `src --p--> dst`, that
  `applyStep cfg (PhaseBoard src base) (phasePlacement p) = PhaseBoard dst (baseOut base)`.

Question: is the board-rewrite obligation `rfl` (definitional), or does it need
real geometry? We test the LAST opening edge `start5 --p--> start6` directly.
-/

/-- The `start5 → start6` board rewrite, proved standalone. -/
theorem probe_start5_to_start6_board_eq
    (first second third fourth fifth p : Piece) (base : ℕ) :
    Placement.applyStep GameConfig.standard
        (PhaseBoard (ReservoirPhase.start5 first second third fourth fifth) base)
        (phasePlacement p)
      = PhaseBoard (ReservoirPhase.start6 first second third fourth fifth p) base := by
  rfl

/-- The `rich3 → rich4` board rewrite, proved standalone. -/
theorem probe_rich3_to_rich4_board_eq
    (first second third p : Piece) (base : ℕ) :
    Placement.applyStep GameConfig.standard
        (PhaseBoard (ReservoirPhase.rich3 first second third) base)
        (phasePlacement p)
      = PhaseBoard (ReservoirPhase.rich4 first second third p) base := by
  rfl

/-- And validity/piece for an arbitrary drawn piece is mechanical. -/
theorem probe_edge_placement_ok (p : Piece) :
    (phasePlacement p).piece = p ∧ (phasePlacement p).Valid GameConfig.standard :=
  ⟨phasePlacement_piece p, phasePlacement_valid p⟩

/-! ## TARGET 2: discharge `frontier_step` for the `start6` reset case.

`start6` is a `CompletionPhase`. The bag has 1 piece; drawing the 7th piece `p`
empties the current bag, forcing the 7-bag refill to `Bag.full`. The destination
must be a `PhaseCarrier Bag.full` of the post-placement board
`applyStep cfg (openingHexaBoard a..f) {pl with piece := p}`.

The ONLY phase whose `PhaseOK` is `T = Bag.full` is `fullReady`, whose board is
`richReadyBoard base' = Board.skyline cfg (ReservoirProfile fullReady base')`.

So discharging this requires CHOOSING `pl` and `base' ≤ 3` and proving the exact
symbolic board equation
  `applyStep cfg (openingHexaBoard a..f) {pl} = richReadyBoard base'`.
We expose that obligation here.
-/

-- BAG SIDE (mechanical): the 7th draw at `start6` refills to `Bag.full`.
-- We take `T.card = 1` as a hypothesis (provable from start6's bag prefix via
-- `six_draw_prefix_card_eq_one`); the point is this is pure bag arithmetic with
-- NO board geometry.
theorem probe_start6_seventh_draw_refill
    {T : Bag} {p : Piece}
    (hcard1T : T.card = 1)
    (hp : p ∈ T) :
    T.draw p = Bag.full := by
  -- drawing the unique element of a singleton bag yields the full refill
  have hcard7 : (T.draw p).card = 7 := Bag.card_draw_of_singleton hp hcard1T
  exact (Bag.card_eq_seven_iff_eq_full _).mp hcard7

/-! ### The actual wall: the `start6` board reset obligation.

`frontier_step` at `start6` must produce a `PhaseCarrier (T.draw p) (post board)`.
Since `T.draw p = Bag.full`, the carrier's `PhaseOK` forces `phase = fullReady`
(the only phase with `PhaseOK ... Bag.full`), whose board is `richReadyBoard base'`.

So the REMAINING obligation, after the (mechanical) bag side, is:

  ∃ pl, pl.piece = p ∧ pl.Valid cfg ∧ ∃ base' ≤ 3,
      applyStep cfg (openingHexaBoard a b c d e f) {pl with piece := p}
        = richReadyBoard base'

Below we DOCUMENT this obligation as a `def` (a Prop), and try to prove a
concrete instance of just the BOARD EQUATION with the existing edge tactic (`rfl`)
to show the pattern does NOT mechanically extend here. -/

/-- The exact board-equation obligation hidden inside `frontier_step start6`. -/
def Start6ResetBoardGoal (a b c d e f p : Piece) : Prop :=
  ∃ (pl : Placement) (base' : ℕ),
    pl.piece = p ∧
    pl.Valid GameConfig.standard ∧
    base' + PhaseMargin ReservoirPhase.fullReady ≤ GameConfig.standard.rows ∧
    Placement.applyStep GameConfig.standard
        (openingHexaBoard a b c d e f) { pl with piece := p }
      = richReadyBoard base'

/-! ### Experiment A: does the table-edge tactic (`rfl`) extend to the reset?

For every existing edge, `applyStep (PhaseBoard src) (phasePlacement p) = PhaseBoard dst`
is `rfl` because `PhaseBoard dst` is DEFINED as that application. Here the target
`richReadyBoard base'` is a `Board.skyline`, NOT defined as an `applyStep` of the
hexa board, so `rfl`/definitional unfolding cannot close it. We confirm the
post-reset board is decidably EQUAL to a concrete `richReadyBoard` value or not. -/

/-- Concrete instance: a fixed legal 6-prefix and 7th piece, with the natural
`phasePlacement` of the drawn piece. We ask `native_decide` whether the resulting
7-piece board (after line clears) equals the `fullReady` skyline at base 0.

EMPIRICAL RESULT (this build): `native_decide` evaluates the equality to `false`.
The natural `phasePlacement` of the 7th piece does NOT reset the 7-piece opening
stack to the reserved-well `fullReady` skyline. So the existing edge pattern
(destination board = `applyStep src (phasePlacement p)`, closed by `rfl`) does
NOT mechanically extend to the reset: a genuine geometric construction (choosing
WHERE the reset pieces land, and proving the post-clear surface) is required. -/
theorem probe_start6_reset_natural_placement_fails :
    decide (Placement.applyStep GameConfig.standard
        (openingHexaBoard Piece.O Piece.I Piece.S Piece.Z Piece.T Piece.L)
        (phasePlacement Piece.J)
      = richReadyBoard 0) = false := by
  native_decide

/-! ### Experiment B: is it just a wrong default placement, or structurally
impossible in one piece?

We search ALL 7th-piece `Piece.J` placements (every rotation 0..3, every column
0..9) onto the fixed hexa board, and ask whether ANY of them resets to
`richReadyBoard 0`. If none do, the reset is not reachable by re-choosing the
single 7th placement on this opening board — the whole opening placement schedule
would need to be redesigned so the 7 pieces jointly tile to the skyline. -/

/-- All candidate `Piece.J` placements (4 rotations × 10 columns). -/
def candidateJPlacements : List Placement :=
  ([0, 1, 2, 3] : List (Fin 4)).flatMap (fun r =>
    (List.range 10).map (fun c =>
      ({ piece := Piece.J, rot := r, col := c } : Placement)))

/-- No single `Piece.J` placement resets the fixed hexa opening board to the
`fullReady` skyline at base 0. (Checked over all rotations and columns.) -/
theorem probe_start6_no_single_J_placement_resets :
    candidateJPlacements.all (fun pl =>
      ¬ (Placement.applyStep GameConfig.standard
          (openingHexaBoard Piece.O Piece.I Piece.S Piece.Z Piece.T Piece.L) pl
        = richReadyBoard 0)) = true := by
  native_decide

/-- Strengthened: no placement of ANY of the 7 pieces, at ANY rotation/column,
resets the fixed hexa opening board to the `fullReady` skyline at ANY base 0..3
(the only bases satisfying the headroom bound `base + 17 ≤ 20`). This shows the
reset wall is not an artifact of piece choice, placement, or base height: a single
final piece simply cannot turn this 6-piece opening stack into the reserved-well
skyline. The opening placement *schedule* must be redesigned for the pieces to
jointly tile to the skyline. -/
def allCandidatePlacements : List Placement :=
  ([Piece.O, Piece.I, Piece.S, Piece.Z, Piece.T, Piece.L, Piece.J]).flatMap
    (fun pc => ([0, 1, 2, 3] : List (Fin 4)).flatMap (fun r =>
      (List.range 10).map (fun c =>
        ({ piece := pc, rot := r, col := c } : Placement))))

theorem probe_start6_no_single_placement_resets_any_base :
    (allCandidatePlacements.all (fun pl =>
      ([0, 1, 2, 3] : List ℕ).all (fun base' =>
        ¬ (Placement.applyStep GameConfig.standard
            (openingHexaBoard Piece.O Piece.I Piece.S Piece.Z Piece.T Piece.L) pl
          = richReadyBoard base')))) = true := by
  native_decide

end RoutePhaseProbe
