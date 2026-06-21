import Proofs.Experiments.SurfaceInvariant
import Proofs.Theorems.Safety

/-!
# Online reservoir strategy: symbolic Atlas roadmap

This experiment is the positive, non-search route for the Atlas proof.

The finite-table route certifies a discovered closed graph.  This file instead
sets up a purely Lean symbolic strategy: define an online invariant over the
current bag and board, prove that the invariant is preserved for every legal
adversarial draw, then obtain `TetrisSolvableValid` from the existing
bag-indexed strategy reduction.

The central obstruction is online order.  Many existing surface lemmas prove a
convenient two-stage fact of the form "place the drawn piece, then drain with an
`I`".  That is not an online strategy unless the adversary has actually drawn
`I` next.  The new math here is therefore an order-independent reservoir
invariant: before the next `I` arrives, every possible sequence of non-`I`
pieces must have reserved landing resources, and when `I` arrives the regulator
must restore the reservoir.

This file intentionally sketches the whole proof system end-to-end.  The older
ledger carrier is kept as a conditional legacy route, with its geometry isolated
in `ReservoirGeometryCert`.  The active route is the bag-indexed symbolic phase
graph in `PhaseGraph`: the relational strategy plumbing, exact edge equations
for the current table, phase/bag transition arithmetic, height bounds, and
conditional solver theorem are all proved.

The remaining active mathematical content is isolated in one certificate field:

* `PhaseGraph.PhaseGraphCompletion.frontier_step` — the exact symbolic phase
  graph now covers the opening first six draws, including the singleton-bag
  phase.  The delegated opening case is therefore the actual reset draw from
  `start6`.  The rich branch is still delegated at `rich4`; the next refinement
  should add rich five/six-piece phases so its delegated case is also the
  singleton reset draw.  The consumed phase indices are proved pairwise
  distinct, so these phases represent legal first-bag histories.

Once that field is supplied, the conditional solver theorem closes the Atlas
claim without Rust search or a hand-built external transition table.
-/

namespace Tetris
namespace OnlineReservoir

noncomputable section

/-- A relational strategy certificate.  `Move T b p pl` is the symbolic
permission to answer draw `p` at `(T,b)` with placement `pl`.  The `total` field
turns it into an actual total solver by choice; the remaining fields are exactly
the strategy-local obligations consumed by
`tetrisSolvableValid_of_strategy_bagInvariant`. -/
structure RelationalAtlasCert where
  P : Bag → Board → Prop
  Move : Bag → Board → Piece → Placement → Prop
  init : P Bag.full Board.empty
  height : ∀ T b, P T b → ∀ j, Board.colHeight b j ≤ GameConfig.standard.rows
  total : ∀ T b p, P T b → p ∈ T → ∃ pl, Move T b p pl
  piece : ∀ T b p pl, Move T b p pl → pl.piece = p
  valid : ∀ T b p pl, P T b → p ∈ T → Move T b p pl → pl.Valid GameConfig.standard
  step : ∀ T b p pl, P T b → p ∈ T → Move T b p pl →
    P (T.draw p) (Placement.applyStep GameConfig.standard b { pl with piece := p })

namespace RelationalAtlasCert

/-- The total strategy extracted from a relational certificate.  Outside the
certified reachable carrier we may return a harmless default; the global
`hpiece` obligation still holds because every certified move names the demanded
piece and the default does too. -/
noncomputable def strategy (C : RelationalAtlasCert) :
    Bag → Board → Piece → Placement := by
  classical
  exact fun T b p =>
    if h : ∃ pl, C.Move T b p pl then Classical.choose h
    else { piece := p, rot := 0, col := 0 }

theorem strategy_move {C : RelationalAtlasCert} {T : Bag} {b : Board} {p : Piece}
    (hP : C.P T b) (hp : p ∈ T) :
    C.Move T b p (C.strategy T b p) := by
  classical
  dsimp [strategy]
  have htotal : ∃ pl, C.Move T b p pl := C.total T b p hP hp
  rw [dif_pos htotal]
  exact Classical.choose_spec htotal

theorem strategy_piece (C : RelationalAtlasCert) (T : Bag) (b : Board) (p : Piece) :
    (C.strategy T b p).piece = p := by
  classical
  dsimp [strategy]
  by_cases h : ∃ pl, C.Move T b p pl
  · rw [dif_pos h]
    exact C.piece T b p (Classical.choose h) (Classical.choose_spec h)
  · rw [dif_neg h]

theorem strategy_valid {C : RelationalAtlasCert} {T : Bag} {b : Board} {p : Piece}
    (hP : C.P T b) (hp : p ∈ T) :
    (C.strategy T b p).Valid GameConfig.standard :=
  C.valid T b p (C.strategy T b p) hP hp (strategy_move hP hp)

theorem strategy_step {C : RelationalAtlasCert} {T : Bag} {b : Board} {p : Piece}
    (hP : C.P T b) (hp : p ∈ T) :
    C.P (T.draw p)
      (Placement.applyStep GameConfig.standard b { C.strategy T b p with piece := p }) :=
  C.step T b p (C.strategy T b p) hP hp (strategy_move hP hp)

/-- End-to-end solver extraction.  This is the generic proof plumbing:
constructing a `RelationalAtlasCert` is enough to prove the strengthened
standard Tetris solvability statement. -/
theorem solved (C : RelationalAtlasCert) : TetrisSolvableValid :=
  tetrisSolvableValid_of_strategy_bagInvariant
    C.strategy
    C.P
    C.strategy_piece
    C.init
    (fun _ _ _ hP hp => C.strategy_step hP hp)
    (fun _ _ _ hP hp => C.strategy_valid hP hp)
    C.height

end RelationalAtlasCert

/-- A conservative online upper bound for how many copies of `p` can be drawn
before the next `I`.

If `I` is still in the current bag, play stops at that `I`, so only a remaining
current-bag copy of `p` can appear.  If `I` is already spent, the adversary may
finish the current bag and then draw one `p` in the next bag before drawing `I`.
-/
def maxCopiesBeforeNextI (T : Bag) (p : Piece) : ℕ :=
  if p = Piece.I then 0
  else (if p ∈ T then 1 else 0) + (if Piece.I ∈ T then 0 else 1)

theorem maxCopiesBeforeNextI_le_two (T : Bag) (p : Piece) :
    maxCopiesBeforeNextI T p ≤ 2 := by
  unfold maxCopiesBeforeNextI
  by_cases hpI : p = Piece.I
  · simp [hpI]
  · by_cases hp : p ∈ T <;> by_cases hI : Piece.I ∈ T <;> simp [hpI, hp, hI]

/-- Ledger data carried by the online reservoir invariant.  `slots p` is the
number of remaining order-independent landing resources reserved for piece `p`
before the next `I` regulator. -/
structure ReservoirLedger where
  floor : ℕ
  spread : ℕ
  slots : Piece → ℕ

namespace ReservoirLedger

def supports (L : ReservoirLedger) (T : Bag) : Prop :=
  ∀ p, p ≠ Piece.I → maxCopiesBeforeNextI T p ≤ L.slots p

def heightOK (L : ReservoirLedger) : Prop :=
  L.floor + L.spread ≤ GameConfig.standard.rows

def consume (L : ReservoirLedger) (p : Piece) : ReservoirLedger :=
  { L with slots := fun q => if q = p then L.slots q - 1 else L.slots q }

def afterRegulator (L : ReservoirLedger) : ReservoirLedger :=
  { L with floor := L.floor % 4, spread := 2 }

end ReservoirLedger

def initialSlots (p : Piece) : ℕ :=
  if p = Piece.I then 0 else 2

def initialLedger : ReservoirLedger where
  floor := 0
  spread := 2
  slots := initialSlots

theorem initialLedger_supports_full :
    initialLedger.supports Bag.full := by
  intro p hpI
  simp [initialLedger, initialSlots, maxCopiesBeforeNextI, hpI, Bag.mem_full]

theorem initialLedger_heightOK : initialLedger.heightOK := by
  change 0 + 2 ≤ GameConfig.standard.rows
  decide

/-- The geometric shape carried by a reservoir ledger.  This deliberately uses
the existing floor-exposing pocket-band vocabulary, so the new file depends on
the surface library rather than duplicating geometry. -/
def IsReservoirShape (L : ReservoirLedger) (b : Board) : Prop :=
  Board.IsLevelPocketBandAt GameConfig.standard L.spread L.floor b

/-- The online reservoir carrier: a board is admissible in bag phase `T` when it
has a floor-exposing reservoir shape, enough remaining slots for every possible
pre-`I` draw, and enough height budget. -/
def Carrier (T : Bag) (b : Board) : Prop :=
  ∃ L : ReservoirLedger, IsReservoirShape L b ∧ L.supports T ∧ L.heightOK

theorem carrier_init : Carrier Bag.full Board.empty := by
  refine ⟨initialLedger, ?_, initialLedger_supports_full, initialLedger_heightOK⟩
  dsimp [IsReservoirShape, initialLedger]
  exact Board.isLevelPocketBandAt_empty GameConfig.standard (by decide) (by decide)

theorem carrier_height {T : Bag} {b : Board} (h : Carrier T b) :
    ∀ j, Board.colHeight b j ≤ GameConfig.standard.rows := by
  intro j
  rcases h with ⟨L, hshape, _hsupports, hheight⟩
  exact le_trans (Board.colHeight_le_of_isLevelPocketBandAt hshape j) hheight

theorem support_gives_live_slot {L : ReservoirLedger} {T : Bag} {p : Piece}
    (hsupports : L.supports T) (hpI : p ≠ Piece.I) (hp : p ∈ T) :
    0 < L.slots p := by
  have hneed := hsupports p hpI
  have hpos : 0 < maxCopiesBeforeNextI T p := by
    unfold maxCopiesBeforeNextI
    by_cases hI : Piece.I ∈ T <;> simp [hpI, hp, hI]
  omega

/-- Critical arithmetic: after a non-`I` draw consumes one reserved slot, the
remaining ledger still covers every order the adversary can choose before the
next `I`. -/
theorem supports_after_nonI_draw {L : ReservoirLedger} {T : Bag} {p : Piece}
    (hsupports : L.supports T) (hp : p ∈ T) (hpI : p ≠ Piece.I) :
    (L.consume p).supports (T.draw p) := by
  intro q hqI
  unfold ReservoirLedger.supports at hsupports
  unfold ReservoirLedger.consume
  dsimp
  by_cases hqp : q = p
  · subst q
    by_cases herase : T.erase p = ∅
    · have honly : ∀ r, r ∈ T → r = p := by
        intro r hr
        by_contra hrp
        have : r ∈ T.erase p := Finset.mem_erase.mpr ⟨hrp, hr⟩
        rw [herase] at this
        exact Finset.notMem_empty r this
      have hInot : Piece.I ∉ T := by
        intro hI
        exact hpI ((honly Piece.I hI).symm)
      have hslot := hsupports p hpI
      rw [Bag.draw, if_pos herase]
      simp [maxCopiesBeforeNextI, hpI, hp, hInot, Bag.mem_full] at hslot ⊢
      omega
    · have hpnot : p ∉ T.erase p := Finset.notMem_erase _ _
      have hslot := hsupports p hpI
      rw [Bag.draw, if_neg herase]
      by_cases hI : Piece.I ∈ T
      · have hIp : Piece.I ≠ p := fun h => hpI h.symm
        have hIdraw : Piece.I ∈ T.erase p :=
          Finset.mem_erase.mpr ⟨hIp, hI⟩
        simp [maxCopiesBeforeNextI, hpI, hpnot, hIdraw]
      · have hInDraw : Piece.I ∉ T.erase p := by
          intro h
          exact hI (Finset.mem_of_mem_erase h)
        simp [maxCopiesBeforeNextI, hpI, hp, hI, hpnot, hInDraw] at hslot ⊢
        omega
  · by_cases herase : T.erase p = ∅
    · have honly : ∀ r, r ∈ T → r = p := by
        intro r hr
        by_contra hrp
        have : r ∈ T.erase p := Finset.mem_erase.mpr ⟨hrp, hr⟩
        rw [herase] at this
        exact Finset.notMem_empty r this
      have hqnot : q ∉ T := by
        intro hq
        exact hqp (honly q hq)
      have hInot : Piece.I ∉ T := by
        intro hI
        exact hpI ((honly Piece.I hI).symm)
      have hslot := hsupports q hqI
      rw [Bag.draw, if_pos herase]
      simpa [maxCopiesBeforeNextI, hqI, hqp, hqnot, hInot, Bag.mem_full] using hslot
    · have hqdraw : q ∈ T.erase p ↔ q ∈ T := by
        simp [Finset.mem_erase, hqp]
      have hIdraw : Piece.I ∈ T.erase p ↔ Piece.I ∈ T := by
        have hIp : Piece.I ≠ p := fun h => hpI h.symm
        simp [Finset.mem_erase, hIp]
      have hslot := hsupports q hqI
      rw [Bag.draw, if_neg herase]
      simpa [maxCopiesBeforeNextI, hqI, hqp, hqdraw, hIdraw] using hslot

/-- Fresh-pocket reentry plumbing.  A bare spread-bounded skyline is not enough
to recover the reservoir carrier, because `IsReservoirShape` also requires a
width-four exposed floor pocket.  Once the geometric branch proof supplies that
pocket, rebuilding the carrier is just ledger bookkeeping. -/
theorem freshPocket_reenters_reservoir {T : Bag} {b : Board} {L : ReservoirLedger}
    {spread floor : ℕ}
    (hpocket : Board.IsLevelPocketBandAt GameConfig.standard spread floor b)
    (hsupports : L.supports T) :
    ∃ L' : ReservoirLedger, IsReservoirShape L' b ∧ L'.supports T ∧ L'.heightOK := by
  let L' : ReservoirLedger := { floor := floor, spread := spread, slots := L.slots }
  refine ⟨L', ?_, ?_, ?_⟩
  · exact hpocket
  · intro p hpI
    exact hsupports p hpI
  · dsimp [ReservoirLedger.heightOK, L']
    rcases hpocket with
      ⟨_h, _w, _c, _hb, _hw, _hw0, _hband, hcap, _hc3, _hwc, _hwc1, _hwc2,
        _hwc3, _hpc, _hpc1, _hpc2, _hpc3⟩
    exact hcap

/-- The two legacy reservoir geometry obligations.  The old ledger proof is no
longer the primary route, but keeping its solver plumbing conditional is useful:
it states exactly what geometry would be needed without hiding that work behind
an assumed proof body. -/
structure ReservoirGeometryCert where
  /-- Consuming one non-`I` reservoir slot must produce a legal placement for the
  drawn piece and a new reservoir carrier for the updated bag. -/
  nonI_slot_geometry : ∀ {T : Bag} {b : Board} {L : ReservoirLedger} {p : Piece},
    (hshape : IsReservoirShape L b) → (hsupports : L.supports T) → (hheight : L.heightOK) →
    (hp : p ∈ T) → (hpI : p ≠ Piece.I) →
    ∃ (pl : Placement) (L' : ReservoirLedger),
      pl.piece = p ∧
      pl.Valid GameConfig.standard ∧
      IsReservoirShape L'
        (Placement.applyStep GameConfig.standard b { pl with piece := p }) ∧
      L'.supports (T.draw p) ∧
      L'.heightOK

  /-- When the adversary draws `I`, the strategy must drain/fill with `I` and
  restore the online reservoir for the post-draw bag. -/
  I_regulator_geometry : ∀ {T : Bag} {b : Board} {L : ReservoirLedger},
    (hshape : IsReservoirShape L b) → (hsupports : L.supports T) → (hheight : L.heightOK) →
    (hp : Piece.I ∈ T) →
    ∃ (pl : Placement) (L' : ReservoirLedger),
      pl.piece = Piece.I ∧
      pl.Valid GameConfig.standard ∧
      IsReservoirShape L'
        (Placement.applyStep GameConfig.standard b { pl with piece := Piece.I }) ∧
      L'.supports (T.draw Piece.I) ∧
      L'.heightOK

theorem carrier_step_nonI (G : ReservoirGeometryCert) {T : Bag} {b : Board} {p : Piece}
    (h : Carrier T b) (hp : p ∈ T) (hpI : p ≠ Piece.I) :
    ∃ pl : Placement,
      pl.piece = p ∧
      pl.Valid GameConfig.standard ∧
      Carrier (T.draw p)
        (Placement.applyStep GameConfig.standard b { pl with piece := p }) := by
  rcases h with ⟨L, hshape, hsupports, hheight⟩
  obtain ⟨pl, L', hpiece, hvalid, hshape', hsupports', hheight'⟩ :=
    G.nonI_slot_geometry hshape hsupports hheight hp hpI
  exact ⟨pl, hpiece, hvalid, ⟨L', hshape', hsupports', hheight'⟩⟩

theorem carrier_step_I (G : ReservoirGeometryCert) {T : Bag} {b : Board}
    (h : Carrier T b) (hp : Piece.I ∈ T) :
    ∃ pl : Placement,
      pl.piece = Piece.I ∧
      pl.Valid GameConfig.standard ∧
      Carrier (T.draw Piece.I)
        (Placement.applyStep GameConfig.standard b { pl with piece := Piece.I }) := by
  rcases h with ⟨L, hshape, hsupports, hheight⟩
  obtain ⟨pl, L', hpiece, hvalid, hshape', hsupports', hheight'⟩ :=
    G.I_regulator_geometry hshape hsupports hheight hp
  exact ⟨pl, hpiece, hvalid, ⟨L', hshape', hsupports', hheight'⟩⟩

/-- The online reservoir closure theorem.  This is the theorem that replaces
the phase-indexed universal surface closure: the carrier is strategy-local and
bag-indexed, so it only has to preserve boards generated by the reservoir
solver. -/
theorem carrier_step (G : ReservoirGeometryCert) {T : Bag} {b : Board}
    (h : Carrier T b) :
    ∀ p, p ∈ T →
      ∃ pl : Placement,
        pl.piece = p ∧
        pl.Valid GameConfig.standard ∧
        Carrier (T.draw p)
          (Placement.applyStep GameConfig.standard b { pl with piece := p }) := by
  intro p hp
  by_cases hpI : p = Piece.I
  · subst hpI
    exact carrier_step_I G h hp
  · exact carrier_step_nonI G h hp hpI

def Move (T : Bag) (b : Board) (p : Piece) (pl : Placement) : Prop :=
  pl.piece = p ∧
  pl.Valid GameConfig.standard ∧
  Carrier (T.draw p)
    (Placement.applyStep GameConfig.standard b { pl with piece := p })

def onlineReservoirCert (G : ReservoirGeometryCert) : RelationalAtlasCert where
  P := Carrier
  Move := Move
  init := carrier_init
  height := fun _ _ h => carrier_height h
  total := by
    intro T b p hP hp
    obtain ⟨pl, hpiece, hvalid, hnext⟩ := carrier_step G hP p hp
    exact ⟨pl, hpiece, hvalid, hnext⟩
  piece := by
    intro T b p pl hmove
    exact hmove.1
  valid := by
    intro T b p pl _hP _hp hmove
    exact hmove.2.1
  step := by
    intro T b p pl _hP _hp hmove
    exact hmove.2.2

/-- Conditional end-to-end theorem for the legacy reservoir approach. -/
theorem onlineReservoir_solves_tetris (G : ReservoirGeometryCert) : TetrisSolvableValid :=
  (onlineReservoirCert G).solved

/-! ## Bag-indexed symbolic phase graph

The reservoir ledger above records numeric capacity separately from geometry.  The
replacement direction below makes the geometry exact: every carrier board is a
specific symbolic board, and the bag decides which phases are admissible.  Some
phases are skylines; opening phases are exact post-placement boards from
`Board.empty`, because the first `S`/`Z` need not be skyline-shaped.  The
remaining open obligation is now an explicit finite graph-completeness
certificate:

* `PhaseGraphCompletion.frontier_step` — the graph still needs a direct
  carrier-preservation theorem for `CompletionPhase`.  Opening play now reaches
  `start6`, a singleton-bag phase with exact Lean-certified height bounds, so
  its remaining obligation is the reset draw back into the ready surface.  The
  rich branch is still delegated at `rich4`; `phaseOK_rich4_next_pairwise_ne`
  certifies that appending the next legal draw gives a five-piece non-repeating
  prefix, which is the next phase to internalize.

The bag arithmetic theorem `phaseOK_step` and the current edge-correctness table
are already proved.

Once that certificate is supplied, `onlinePhaseGraph_solves_tetris` closes
through the same `RelationalAtlasCert` plumbing as the legacy reservoir attempt.
-/

namespace PhaseGraph

set_option linter.constructorNameAsVariable false

/-- Exact symbolic surfaces for the bag-indexed phase graph.  `emptyStart` is
separate because the rich ready surface has `base + 1` bumps and is not
definitionally the empty board at `base = 0`. -/
inductive ReservoirPhase
  | emptyStart
  | startO
  | startT
  | startL
  | startJ
  | startS
  | startZ
  | startI
  | start2 (first second : Piece)
  | start3 (first second third : Piece)
  | start4 (first second third fourth : Piece)
  | start5 (first second third fourth fifth : Piece)
  | start6 (first second third fourth fifth sixth : Piece)
  | fullReady
  | noO
  | noT
  | noL
  | noJ
  | noS
  | noZ
  | noI
  | rich2 (first second : Piece)
  | rich3 (first second third : Piece)
  | rich4 (first second third fourth : Piece)
deriving DecidableEq, Repr, Fintype

/-- The exact phases not yet represented by ordinary `phaseEdge?` table edges. -/
def FrontierPhase (phase : ReservoirPhase) : Prop :=
  (∃ first second third fourth fifth,
      phase = ReservoirPhase.start5 first second third fourth fifth) ∨
  (∃ first second third fourth,
      phase = ReservoirPhase.rich4 first second third fourth)

/-- The phases that still require an external completion theorem in the current
table.  Opening play has been pushed one step past `start5`, so its delegated
state is the singleton-bag `start6`.  The rich branch is still delegated at
`rich4` until the rich five/six-piece tail is added with sharp geometry. -/
def CompletionPhase (phase : ReservoirPhase) : Prop :=
  (∃ first second third fourth fifth sixth,
      phase = ReservoirPhase.start6 first second third fourth fifth sixth) ∨
  (∃ first second third fourth,
      phase = ReservoirPhase.rich4 first second third fourth)

/-- The exact skyline profile for each symbolic phase.  Most non-ready profiles
are deliberately conservative placeholders; the transition theorem is where the
next pass will replace them with the actual post-placement shapes. -/
def ReservoirProfile (phase : ReservoirPhase) (base : ℕ) : ℕ → ℕ :=
  match phase with
  | .emptyStart => fun _ => 0
  | .startO => fun _ => 0
  | .startT => fun _ => 0
  | .startL => fun _ => 0
  | .startJ => fun _ => 0
  | .startS => fun _ => 0
  | .startZ => fun _ => 0
  | .startI => fun _ => 0
  | .start2 _ _ => fun _ => 0
  | .start3 _ _ _ => fun _ => 0
  | .start4 _ _ _ _ => fun _ => 0
  | .start5 _ _ _ _ _ => fun _ => 0
  | .start6 _ _ _ _ _ _ => fun _ => 0
  | .fullReady =>
      fun j =>
        if j = 0 then 0
        else if j = 5 then base + 1
        else if j = 8 then base + 1
        else base
  | .noO => fun j => if j = 0 then 0 else base
  | .noT => fun j => if j = 0 then 0 else base
  | .noL => fun j => if j = 0 then 0 else base
  | .noJ => fun j => if j = 0 then 0 else base
  | .noS => fun j => if j = 0 then 0 else base
  | .noZ => fun j => if j = 0 then 0 else base
  | .noI => fun j => if j = 0 then 0 else base
  | .rich2 _ _ => fun j => if j = 0 then 0 else base
  | .rich3 _ _ _ => fun j => if j = 0 then 0 else base
  | .rich4 _ _ _ _ => fun j => if j = 0 then 0 else base

/-- Per-phase vertical headroom.  These are intentionally conservative until the
edge equations pin down sharper margins for each exact profile. -/
def PhaseMargin : ReservoirPhase → ℕ
  | .emptyStart => 0
  | .startO => 4
  | .startT => 4
  | .startL => 4
  | .startJ => 4
  | .startS => 4
  | .startZ => 4
  | .startI => 4
  | .start2 _ _ => 8
  | .start3 _ _ _ => 12
  | .start4 _ _ _ _ => 16
  | .start5 _ _ _ _ _ => 20
  | .start6 _ _ _ _ _ _ => 20
  | .fullReady => 17
  | .noO => 17
  | .noT => 17
  | .noL => 17
  | .noJ => 17
  | .noS => 17
  | .noZ => 17
  | .noI => 17
  | .rich2 _ _ => 17
  | .rich3 _ _ _ => 17
  | .rich4 _ _ _ _ => 17

/-- Bag compatibility for each symbolic phase.  The phase says which geometric
resources have already been consumed; the bag says which pieces can still be
drawn before refill. -/
def PhaseOK (phase : ReservoirPhase) (T : Bag) : Prop :=
  match phase with
  | .emptyStart => T = Bag.full
  | .startO => T = Bag.full.draw Piece.O
  | .startT => T = Bag.full.draw Piece.T
  | .startL => T = Bag.full.draw Piece.L
  | .startJ => T = Bag.full.draw Piece.J
  | .startS => T = Bag.full.draw Piece.S
  | .startZ => T = Bag.full.draw Piece.Z
  | .startI => T = Bag.full.draw Piece.I
  | .start2 first second =>
      T = (Bag.full.draw first).draw second ∧
      second ∈ Bag.full.draw first
  | .start3 first second third =>
      T = ((Bag.full.draw first).draw second).draw third ∧
      second ∈ Bag.full.draw first ∧
      third ∈ (Bag.full.draw first).draw second
  | .start4 first second third fourth =>
      T = (((Bag.full.draw first).draw second).draw third).draw fourth ∧
      second ∈ Bag.full.draw first ∧
      third ∈ (Bag.full.draw first).draw second ∧
      fourth ∈ ((Bag.full.draw first).draw second).draw third
  | .start5 first second third fourth fifth =>
      T = ((((Bag.full.draw first).draw second).draw third).draw fourth).draw fifth ∧
      second ∈ Bag.full.draw first ∧
      third ∈ (Bag.full.draw first).draw second ∧
      fourth ∈ ((Bag.full.draw first).draw second).draw third ∧
      fifth ∈ (((Bag.full.draw first).draw second).draw third).draw fourth
  | .start6 first second third fourth fifth sixth =>
      T =
        (((((Bag.full.draw first).draw second).draw third).draw fourth).draw fifth).draw sixth ∧
      second ∈ Bag.full.draw first ∧
      third ∈ (Bag.full.draw first).draw second ∧
      fourth ∈ ((Bag.full.draw first).draw second).draw third ∧
      fifth ∈ (((Bag.full.draw first).draw second).draw third).draw fourth ∧
      sixth ∈ ((((Bag.full.draw first).draw second).draw third).draw fourth).draw fifth
  | .fullReady => T = Bag.full
  | .noO => T = Bag.full.draw Piece.O
  | .noT => T = Bag.full.draw Piece.T
  | .noL => T = Bag.full.draw Piece.L
  | .noJ => T = Bag.full.draw Piece.J
  | .noS => T = Bag.full.draw Piece.S
  | .noZ => T = Bag.full.draw Piece.Z
  | .noI => T = Bag.full.draw Piece.I
  | .rich2 first second =>
      T = (Bag.full.draw first).draw second ∧
      second ∈ Bag.full.draw first
  | .rich3 first second third =>
      T = ((Bag.full.draw first).draw second).draw third ∧
      second ∈ Bag.full.draw first ∧
      third ∈ (Bag.full.draw first).draw second
  | .rich4 first second third fourth =>
      T = (((Bag.full.draw first).draw second).draw third).draw fourth ∧
      second ∈ Bag.full.draw first ∧
      third ∈ (Bag.full.draw first).draw second ∧
      fourth ∈ ((Bag.full.draw first).draw second).draw third

/-- A legal four-draw prefix from a full bag leaves exactly three pieces. -/
theorem four_draw_prefix_card_eq_three
    {first second third fourth : Piece}
    (hsecond : second ∈ Bag.full.draw first)
    (hthird : third ∈ (Bag.full.draw first).draw second)
    (hfourth : fourth ∈ ((Bag.full.draw first).draw second).draw third) :
    ((((Bag.full.draw first).draw second).draw third).draw fourth).card = 3 := by
  have h1 : (Bag.full.draw first).card = 6 := by simp
  have h2 : ((Bag.full.draw first).draw second).card = 5 := by
    rw [Bag.card_draw_of_ge_two hsecond (by rw [h1]; omega), h1]
  have h3 : (((Bag.full.draw first).draw second).draw third).card = 4 := by
    rw [Bag.card_draw_of_ge_two hthird (by rw [h2]; omega), h2]
  rw [Bag.card_draw_of_ge_two hfourth (by rw [h3]; omega), h3]

/-- A legal five-draw prefix from a full bag leaves exactly two pieces. -/
theorem five_draw_prefix_card_eq_two
    {first second third fourth fifth : Piece}
    (hsecond : second ∈ Bag.full.draw first)
    (hthird : third ∈ (Bag.full.draw first).draw second)
    (hfourth : fourth ∈ ((Bag.full.draw first).draw second).draw third)
    (hfifth : fifth ∈ (((Bag.full.draw first).draw second).draw third).draw fourth) :
    (((((Bag.full.draw first).draw second).draw third).draw fourth).draw fifth).card = 2 := by
  have h4 := four_draw_prefix_card_eq_three
    (first := first) (second := second) (third := third) (fourth := fourth)
    hsecond hthird hfourth
  rw [Bag.card_draw_of_ge_two hfifth (by rw [h4]; omega), h4]

/-- A legal six-draw prefix from a full bag leaves a singleton bag. -/
theorem six_draw_prefix_card_eq_one
    {a b c d e f : Piece}
    (hb : b ∈ Bag.full.draw a)
    (hc : c ∈ (Bag.full.draw a).draw b)
    (hd : d ∈ ((Bag.full.draw a).draw b).draw c)
    (he : e ∈ (((Bag.full.draw a).draw b).draw c).draw d)
    (hf : f ∈ ((((Bag.full.draw a).draw b).draw c).draw d).draw e) :
    ((((((Bag.full.draw a).draw b).draw c).draw d).draw e).draw f).card = 1 := by
  revert a b c d e f
  native_decide

/-- Any admissible opening frontier has exactly two pieces left in the current bag. -/
theorem phaseOK_start5_card_eq_two
    {T : Bag} {first second third fourth fifth : Piece}
    (hPhase : PhaseOK (ReservoirPhase.start5 first second third fourth fifth) T) :
    T.card = 2 := by
  rcases hPhase with ⟨hT, hsecond, hthird, hfourth, hfifth⟩
  rw [hT]
  exact five_draw_prefix_card_eq_two hsecond hthird hfourth hfifth

/-- Any admissible rich frontier has exactly three pieces left in the current bag. -/
theorem phaseOK_rich4_card_eq_three
    {T : Bag} {first second third fourth : Piece}
    (hPhase : PhaseOK (ReservoirPhase.rich4 first second third fourth) T) :
    T.card = 3 := by
  rcases hPhase with ⟨hT, hsecond, hthird, hfourth⟩
  rw [hT]
  exact four_draw_prefix_card_eq_three hsecond hthird hfourth

/-- The delegated graph-completion frontier is now bag-card sharp. -/
theorem phaseOK_frontier_card
    {phase : ReservoirPhase} {T : Bag}
    (hfrontier : FrontierPhase phase)
    (hPhase : PhaseOK phase T) :
    T.card = 2 ∨ T.card = 3 := by
  dsimp [FrontierPhase] at hfrontier
  rcases hfrontier with
    ⟨first, second, third, fourth, fifth, hphase⟩ |
    ⟨first, second, third, fourth, hphase⟩
  · subst phase
    exact Or.inl (phaseOK_start5_card_eq_two hPhase)
  · subst phase
    exact Or.inr (phaseOK_rich4_card_eq_three hPhase)

/-- Drawing once from an admissible opening frontier leaves a singleton bag. -/
theorem phaseOK_start5_draw_card_eq_one
    {T : Bag} {first second third fourth fifth p : Piece}
    (hPhase : PhaseOK (ReservoirPhase.start5 first second third fourth fifth) T)
    (hp : p ∈ T) :
    (T.draw p).card = 1 := by
  have hcard := phaseOK_start5_card_eq_two hPhase
  rw [Bag.card_draw_of_ge_two hp (by omega), hcard]

/-- Drawing once from an admissible rich frontier leaves two pieces in the bag. -/
theorem phaseOK_rich4_draw_card_eq_two
    {T : Bag} {first second third fourth p : Piece}
    (hPhase : PhaseOK (ReservoirPhase.rich4 first second third fourth) T)
    (hp : p ∈ T) :
    (T.draw p).card = 2 := by
  have hcard := phaseOK_rich4_card_eq_three hPhase
  rw [Bag.card_draw_of_ge_two hp (by omega), hcard]

/-- The delegated frontier also has sharp one-step refill timing. -/
theorem phaseOK_frontier_draw_card
    {phase : ReservoirPhase} {T : Bag} {p : Piece}
    (hfrontier : FrontierPhase phase)
    (hPhase : PhaseOK phase T) (hp : p ∈ T) :
    (T.draw p).card = 1 ∨ (T.draw p).card = 2 := by
  dsimp [FrontierPhase] at hfrontier
  rcases hfrontier with
    ⟨first, second, third, fourth, fifth, hphase⟩ |
    ⟨first, second, third, fourth, hphase⟩
  · subst phase
    exact Or.inl (phaseOK_start5_draw_card_eq_one hPhase hp)
  · subst phase
    exact Or.inr (phaseOK_rich4_draw_card_eq_two hPhase hp)

/-- From an opening frontier, two more legal draws force the 7-bag refill. -/
theorem phaseOK_start5_two_step_refill
    {T : Bag} {first second third fourth fifth p q : Piece}
    (hPhase : PhaseOK (ReservoirPhase.start5 first second third fourth fifth) T)
    (hp : p ∈ T) (hq : q ∈ T.draw p) :
    (T.draw p).draw q = Bag.full := by
  have hcard1 : (T.draw p).card = 1 :=
    phaseOK_start5_draw_card_eq_one hPhase hp
  have hcard7 : ((T.draw p).draw q).card = 7 :=
    Bag.card_draw_of_singleton hq hcard1
  exact (Bag.card_eq_seven_iff_eq_full _).mp hcard7

/-- From a rich frontier, two more legal draws leave a singleton bag. -/
theorem phaseOK_rich4_second_draw_card_eq_one
    {T : Bag} {first second third fourth p q : Piece}
    (hPhase : PhaseOK (ReservoirPhase.rich4 first second third fourth) T)
    (hp : p ∈ T) (hq : q ∈ T.draw p) :
    ((T.draw p).draw q).card = 1 := by
  have hcard2 : (T.draw p).card = 2 :=
    phaseOK_rich4_draw_card_eq_two hPhase hp
  rw [Bag.card_draw_of_ge_two hq (by omega), hcard2]

/-- From a rich frontier, three more legal draws force the 7-bag refill. -/
theorem phaseOK_rich4_three_step_refill
    {T : Bag} {first second third fourth p q r : Piece}
    (hPhase : PhaseOK (ReservoirPhase.rich4 first second third fourth) T)
    (hp : p ∈ T) (hq : q ∈ T.draw p) (hr : r ∈ (T.draw p).draw q) :
    ((T.draw p).draw q).draw r = Bag.full := by
  have hcard1 : ((T.draw p).draw q).card = 1 :=
    phaseOK_rich4_second_draw_card_eq_one hPhase hp hq
  have hcard7 : (((T.draw p).draw q).draw r).card = 7 :=
    Bag.card_draw_of_singleton hr hcard1
  exact (Bag.card_eq_seven_iff_eq_full _).mp hcard7

/-- Any delegated frontier refills within two more legal draws after the current draw.

Opening frontiers have one piece left after the current draw, so the next draw
refills.  Rich frontiers have two pieces left after the current draw, so one
more draw leaves a singleton and the following draw refills. -/
theorem phaseOK_frontier_refill_cases
    {phase : ReservoirPhase} {T : Bag} {p : Piece}
    (hfrontier : FrontierPhase phase)
    (hPhase : PhaseOK phase T) (hp : p ∈ T) :
    ((T.draw p).card = 1 ∧
      ∀ q, q ∈ T.draw p → (T.draw p).draw q = Bag.full) ∨
    ((T.draw p).card = 2 ∧
      ∀ q, q ∈ T.draw p →
        ((T.draw p).draw q).card = 1 ∧
        ∀ r, r ∈ (T.draw p).draw q → ((T.draw p).draw q).draw r = Bag.full) := by
  dsimp [FrontierPhase] at hfrontier
  rcases hfrontier with
    ⟨first, second, third, fourth, fifth, hphase⟩ |
    ⟨first, second, third, fourth, hphase⟩
  · subst phase
    refine Or.inl ⟨phaseOK_start5_draw_card_eq_one hPhase hp, ?_⟩
    intro q hq
    exact phaseOK_start5_two_step_refill hPhase hp hq
  · subst phase
    refine Or.inr ⟨phaseOK_rich4_draw_card_eq_two hPhase hp, ?_⟩
    intro q hq
    exact ⟨phaseOK_rich4_second_draw_card_eq_one hPhase hp hq,
      fun r hr => phaseOK_rich4_three_step_refill hPhase hp hq hr⟩

/-- Membership after a non-refilling draw implies membership before the draw. -/
theorem mem_of_mem_draw_of_ge_two {bag : Bag} {p q : Piece}
    (hp : p ∈ bag) (hcard : 2 ≤ bag.card) (hq : q ∈ bag.draw p) :
    q ∈ bag := by
  rw [Bag.draw_eq_erase_of_card_ge_two hp hcard] at hq
  exact (Finset.mem_erase.mp hq).2

/-- Membership after a non-refilling draw proves the member is not the drawn piece. -/
theorem ne_drawn_of_mem_draw_of_ge_two {bag : Bag} {p q : Piece}
    (hp : p ∈ bag) (hcard : 2 ≤ bag.card) (hq : q ∈ bag.draw p) :
    q ≠ p := by
  rw [Bag.draw_eq_erase_of_card_ge_two hp hcard] at hq
  exact (Finset.mem_erase.mp hq).1

/-- A legal four-draw prefix from a full bag has four distinct phase indices. -/
theorem four_draw_prefix_pairwise_ne
    {first second third fourth : Piece}
    (hsecond : second ∈ Bag.full.draw first)
    (hthird : third ∈ (Bag.full.draw first).draw second)
    (hfourth : fourth ∈ ((Bag.full.draw first).draw second).draw third) :
    second ≠ first ∧
    third ≠ first ∧ third ≠ second ∧
    fourth ≠ first ∧ fourth ≠ second ∧ fourth ≠ third := by
  have h1card : (Bag.full.draw first).card = 6 := by simp
  have h2card : ((Bag.full.draw first).draw second).card = 5 := by
    rw [Bag.card_draw_of_ge_two hsecond (by rw [h1card]; omega), h1card]
  have hsecond_ne_first : second ≠ first :=
    (Bag.mem_draw_full_iff_ne first second).mp hsecond
  have hthird_ne_second : third ≠ second :=
    ne_drawn_of_mem_draw_of_ge_two hsecond (by rw [h1card]; omega) hthird
  have hthird_mem_first_draw : third ∈ Bag.full.draw first :=
    mem_of_mem_draw_of_ge_two hsecond (by rw [h1card]; omega) hthird
  have hthird_ne_first : third ≠ first :=
    (Bag.mem_draw_full_iff_ne first third).mp hthird_mem_first_draw
  have hfourth_ne_third : fourth ≠ third :=
    ne_drawn_of_mem_draw_of_ge_two hthird (by rw [h2card]; omega) hfourth
  have hfourth_mem_second_draw :
      fourth ∈ (Bag.full.draw first).draw second :=
    mem_of_mem_draw_of_ge_two hthird (by rw [h2card]; omega) hfourth
  have hfourth_ne_second : fourth ≠ second :=
    ne_drawn_of_mem_draw_of_ge_two hsecond (by rw [h1card]; omega) hfourth_mem_second_draw
  have hfourth_mem_first_draw : fourth ∈ Bag.full.draw first :=
    mem_of_mem_draw_of_ge_two hsecond (by rw [h1card]; omega) hfourth_mem_second_draw
  have hfourth_ne_first : fourth ≠ first :=
    (Bag.mem_draw_full_iff_ne first fourth).mp hfourth_mem_first_draw
  exact ⟨hsecond_ne_first, hthird_ne_first, hthird_ne_second,
    hfourth_ne_first, hfourth_ne_second, hfourth_ne_third⟩

/-- A legal five-draw prefix from a full bag has five distinct phase indices. -/
theorem five_draw_prefix_pairwise_ne
    {first second third fourth fifth : Piece}
    (hsecond : second ∈ Bag.full.draw first)
    (hthird : third ∈ (Bag.full.draw first).draw second)
    (hfourth : fourth ∈ ((Bag.full.draw first).draw second).draw third)
    (hfifth : fifth ∈ (((Bag.full.draw first).draw second).draw third).draw fourth) :
    second ≠ first ∧
    third ≠ first ∧ third ≠ second ∧
    fourth ≠ first ∧ fourth ≠ second ∧ fourth ≠ third ∧
    fifth ≠ first ∧ fifth ≠ second ∧ fifth ≠ third ∧ fifth ≠ fourth := by
  have hfirst4 := four_draw_prefix_pairwise_ne hsecond hthird hfourth
  rcases hfirst4 with
    ⟨hsecond_ne_first, hthird_ne_first, hthird_ne_second,
      hfourth_ne_first, hfourth_ne_second, hfourth_ne_third⟩
  have h4card := four_draw_prefix_card_eq_three hsecond hthird hfourth
  have h3card : (((Bag.full.draw first).draw second).draw third).card = 4 := by
    have h1card : (Bag.full.draw first).card = 6 := by simp
    have h2card : ((Bag.full.draw first).draw second).card = 5 := by
      rw [Bag.card_draw_of_ge_two hsecond (by rw [h1card]; omega), h1card]
    rw [Bag.card_draw_of_ge_two hthird (by rw [h2card]; omega), h2card]
  have hfifth_ne_fourth : fifth ≠ fourth :=
    ne_drawn_of_mem_draw_of_ge_two hfourth (by rw [h3card]; omega) hfifth
  have hfifth_mem_third_draw :
      fifth ∈ ((Bag.full.draw first).draw second).draw third :=
    mem_of_mem_draw_of_ge_two hfourth (by rw [h3card]; omega) hfifth
  have hfifth_ne_third : fifth ≠ third :=
    ne_drawn_of_mem_draw_of_ge_two hthird (by
      have h1card : (Bag.full.draw first).card = 6 := by simp
      have h2card : ((Bag.full.draw first).draw second).card = 5 := by
        rw [Bag.card_draw_of_ge_two hsecond (by rw [h1card]; omega), h1card]
      rw [h2card]; omega) hfifth_mem_third_draw
  have hfifth_mem_second_draw :
      fifth ∈ (Bag.full.draw first).draw second :=
    mem_of_mem_draw_of_ge_two hthird (by
      have h1card : (Bag.full.draw first).card = 6 := by simp
      have h2card : ((Bag.full.draw first).draw second).card = 5 := by
        rw [Bag.card_draw_of_ge_two hsecond (by rw [h1card]; omega), h1card]
      rw [h2card]; omega) hfifth_mem_third_draw
  have hfifth_ne_second : fifth ≠ second :=
    ne_drawn_of_mem_draw_of_ge_two hsecond (by
      have h1card : (Bag.full.draw first).card = 6 := by simp
      rw [h1card]; omega) hfifth_mem_second_draw
  have hfifth_mem_first_draw : fifth ∈ Bag.full.draw first :=
    mem_of_mem_draw_of_ge_two hsecond (by
      have h1card : (Bag.full.draw first).card = 6 := by simp
      rw [h1card]; omega) hfifth_mem_second_draw
  have hfifth_ne_first : fifth ≠ first :=
    (Bag.mem_draw_full_iff_ne first fifth).mp hfifth_mem_first_draw
  exact ⟨hsecond_ne_first,
    hthird_ne_first, hthird_ne_second,
    hfourth_ne_first, hfourth_ne_second, hfourth_ne_third,
    hfifth_ne_first, hfifth_ne_second, hfifth_ne_third, hfifth_ne_fourth⟩

/-- A legal six-draw prefix from a full bag has no repeated pieces. -/
theorem six_draw_prefix_nodup
    {a b c d e f : Piece}
    (hb : b ∈ Bag.full.draw a)
    (hc : c ∈ (Bag.full.draw a).draw b)
    (hd : d ∈ ((Bag.full.draw a).draw b).draw c)
    (he : e ∈ (((Bag.full.draw a).draw b).draw c).draw d)
    (hf : f ∈ ((((Bag.full.draw a).draw b).draw c).draw d).draw e) :
    [a, b, c, d, e, f].Nodup := by
  revert a b c d e f
  native_decide

/-- Admissible rich frontiers have four distinct consumed phase indices. -/
theorem phaseOK_rich4_pairwise_ne
    {T : Bag} {first second third fourth : Piece}
    (hPhase : PhaseOK (ReservoirPhase.rich4 first second third fourth) T) :
    second ≠ first ∧
    third ≠ first ∧ third ≠ second ∧
    fourth ≠ first ∧ fourth ≠ second ∧ fourth ≠ third := by
  rcases hPhase with ⟨_hT, hsecond, hthird, hfourth⟩
  exact four_draw_prefix_pairwise_ne hsecond hthird hfourth

/-- Admissible opening frontiers have five distinct consumed phase indices. -/
theorem phaseOK_start5_pairwise_ne
    {T : Bag} {first second third fourth fifth : Piece}
    (hPhase : PhaseOK (ReservoirPhase.start5 first second third fourth fifth) T) :
    second ≠ first ∧
    third ≠ first ∧ third ≠ second ∧
    fourth ≠ first ∧ fourth ≠ second ∧ fourth ≠ third ∧
    fifth ≠ first ∧ fifth ≠ second ∧ fifth ≠ third ∧ fifth ≠ fourth := by
  rcases hPhase with ⟨_hT, hsecond, hthird, hfourth, hfifth⟩
  exact five_draw_prefix_pairwise_ne hsecond hthird hfourth hfifth

/-- Admissible six-piece opening phases are legal non-repeating prefixes. -/
theorem phaseOK_start6_nodup
    {T : Bag} {a b c d e f : Piece}
    (hPhase : PhaseOK (ReservoirPhase.start6 a b c d e f) T) :
    [a, b, c, d, e, f].Nodup := by
  rcases hPhase with ⟨hT, hb, hc, hd, he, hf⟩
  clear hT T
  revert a b c d e f
  native_decide

/-- A legal remaining piece after four draws is distinct from all four consumed pieces. -/
theorem four_draw_prefix_remaining_ne
    {first second third fourth p : Piece}
    (hsecond : second ∈ Bag.full.draw first)
    (hthird : third ∈ (Bag.full.draw first).draw second)
    (hfourth : fourth ∈ ((Bag.full.draw first).draw second).draw third)
    (hp : p ∈ (((Bag.full.draw first).draw second).draw third).draw fourth) :
    p ≠ first ∧ p ≠ second ∧ p ≠ third ∧ p ≠ fourth := by
  have h1card : (Bag.full.draw first).card = 6 := by simp
  have h2card : ((Bag.full.draw first).draw second).card = 5 := by
    rw [Bag.card_draw_of_ge_two hsecond (by rw [h1card]; omega), h1card]
  have h3card : (((Bag.full.draw first).draw second).draw third).card = 4 := by
    rw [Bag.card_draw_of_ge_two hthird (by rw [h2card]; omega), h2card]
  have hp_ne_fourth : p ≠ fourth :=
    ne_drawn_of_mem_draw_of_ge_two hfourth (by rw [h3card]; omega) hp
  have hp_mem_third_draw :
      p ∈ ((Bag.full.draw first).draw second).draw third :=
    mem_of_mem_draw_of_ge_two hfourth (by rw [h3card]; omega) hp
  have hp_ne_third : p ≠ third :=
    ne_drawn_of_mem_draw_of_ge_two hthird (by rw [h2card]; omega) hp_mem_third_draw
  have hp_mem_second_draw : p ∈ (Bag.full.draw first).draw second :=
    mem_of_mem_draw_of_ge_two hthird (by rw [h2card]; omega) hp_mem_third_draw
  have hp_ne_second : p ≠ second :=
    ne_drawn_of_mem_draw_of_ge_two hsecond (by rw [h1card]; omega) hp_mem_second_draw
  have hp_mem_first_draw : p ∈ Bag.full.draw first :=
    mem_of_mem_draw_of_ge_two hsecond (by rw [h1card]; omega) hp_mem_second_draw
  have hp_ne_first : p ≠ first :=
    (Bag.mem_draw_full_iff_ne first p).mp hp_mem_first_draw
  exact ⟨hp_ne_first, hp_ne_second, hp_ne_third, hp_ne_fourth⟩

/-- Any legal next draw at a rich frontier is not one of the consumed pieces. -/
theorem phaseOK_rich4_next_ne_consumed
    {T : Bag} {first second third fourth p : Piece}
    (hPhase : PhaseOK (ReservoirPhase.rich4 first second third fourth) T)
    (hp : p ∈ T) :
    p ≠ first ∧ p ≠ second ∧ p ≠ third ∧ p ≠ fourth := by
  rcases hPhase with ⟨hT, hsecond, hthird, hfourth⟩
  subst T
  exact four_draw_prefix_remaining_ne hsecond hthird hfourth hp

/-- Rich frontiers plus a legal next draw form a legal five-piece prefix. -/
theorem phaseOK_rich4_next_pairwise_ne
    {T : Bag} {first second third fourth p : Piece}
    (hPhase : PhaseOK (ReservoirPhase.rich4 first second third fourth) T)
    (hp : p ∈ T) :
    second ≠ first ∧
    third ≠ first ∧ third ≠ second ∧
    fourth ≠ first ∧ fourth ≠ second ∧ fourth ≠ third ∧
    p ≠ first ∧ p ≠ second ∧ p ≠ third ∧ p ≠ fourth := by
  rcases phaseOK_rich4_pairwise_ne hPhase with
    ⟨hsecond_ne_first, hthird_ne_first, hthird_ne_second,
      hfourth_ne_first, hfourth_ne_second, hfourth_ne_third⟩
  rcases phaseOK_rich4_next_ne_consumed hPhase hp with
    ⟨hp_ne_first, hp_ne_second, hp_ne_third, hp_ne_fourth⟩
  exact ⟨hsecond_ne_first,
    hthird_ne_first, hthird_ne_second,
    hfourth_ne_first, hfourth_ne_second, hfourth_ne_third,
    hp_ne_first, hp_ne_second, hp_ne_third, hp_ne_fourth⟩

/-- The concrete placement used by the first draft of the phase table.  The
columns match the rich-surface dispatch choices where possible; for opening
states this deliberately records the real post-empty board rather than a
skyline approximation. -/
def phasePlacement : Piece → Placement
  | Piece.O => { piece := Piece.O, rot := 0, col := 1 }
  | Piece.I => { piece := Piece.I, rot := 1, col := 0 }
  | Piece.S => { piece := Piece.S, rot := 0, col := 3 }
  | Piece.Z => { piece := Piece.Z, rot := 0, col := 5 }
  | Piece.T => { piece := Piece.T, rot := 2, col := 1 }
  | Piece.L => { piece := Piece.L, rot := 0, col := 1 }
  | Piece.J => { piece := Piece.J, rot := 0, col := 1 }

theorem phasePlacement_piece (p : Piece) : (phasePlacement p).piece = p := by
  cases p <;> rfl

theorem phasePlacement_valid (p : Piece) :
    (phasePlacement p).Valid GameConfig.standard := by
  cases p <;> dsimp [phasePlacement]
  · exact Board.valid_O (by decide)
  · exact Board.valid_vertI (by decide)
  · exact Board.valid_S (by decide)
  · exact Board.valid_Z (by decide)
  · exact Board.valid_T (by decide)
  · exact Board.valid_L (by decide)
  · exact Board.valid_J (by decide)

def richReadyBoard (base : ℕ) : Board :=
  Board.skyline GameConfig.standard (ReservoirProfile ReservoirPhase.fullReady base)

def richStepBoard (p : Piece) (base : ℕ) : Board :=
  Placement.applyStep GameConfig.standard (richReadyBoard base) (phasePlacement p)

def richPairBoard (first second : Piece) (base : ℕ) : Board :=
  Placement.applyStep GameConfig.standard (richStepBoard first base) (phasePlacement second)

def richTripleBoard (first second third : Piece) (base : ℕ) : Board :=
  Placement.applyStep GameConfig.standard (richPairBoard first second base) (phasePlacement third)

def richQuadBoard (first second third fourth : Piece) (base : ℕ) : Board :=
  Placement.applyStep GameConfig.standard (richTripleBoard first second third base)
    (phasePlacement fourth)

def openingBoard (p : Piece) : Board :=
  Placement.applyStep GameConfig.standard Board.empty (phasePlacement p)

def openingPairBoard (first second : Piece) : Board :=
  Placement.applyStep GameConfig.standard (openingBoard first) (phasePlacement second)

def openingTripleBoard (first second third : Piece) : Board :=
  Placement.applyStep GameConfig.standard (openingPairBoard first second) (phasePlacement third)

def openingQuadBoard (first second third fourth : Piece) : Board :=
  Placement.applyStep GameConfig.standard (openingTripleBoard first second third) (phasePlacement fourth)

def openingPentaBoard (first second third fourth fifth : Piece) : Board :=
  Placement.applyStep GameConfig.standard (openingQuadBoard first second third fourth)
    (phasePlacement fifth)

def openingHexaBoard (first second third fourth fifth sixth : Piece) : Board :=
  Placement.applyStep GameConfig.standard (openingPentaBoard first second third fourth fifth)
    (phasePlacement sixth)

/-- Exact board denotation for each phase.  This is intentionally not just a
profile: opening phases preserve the actual first-piece board. -/
def PhaseBoard (phase : ReservoirPhase) (base : ℕ) : Board :=
  match phase with
  | .emptyStart => Board.empty
  | .startO => openingBoard Piece.O
  | .startT => openingBoard Piece.T
  | .startL => openingBoard Piece.L
  | .startJ => openingBoard Piece.J
  | .startS => openingBoard Piece.S
  | .startZ => openingBoard Piece.Z
  | .startI => openingBoard Piece.I
  | .start2 first second => openingPairBoard first second
  | .start3 first second third => openingTripleBoard first second third
  | .start4 first second third fourth => openingQuadBoard first second third fourth
  | .start5 first second third fourth fifth => openingPentaBoard first second third fourth fifth
  | .start6 first second third fourth fifth sixth =>
      openingHexaBoard first second third fourth fifth sixth
  | .fullReady => richReadyBoard base
  | .noO => richStepBoard Piece.O base
  | .noT => richStepBoard Piece.T base
  | .noL => richStepBoard Piece.L base
  | .noJ => richStepBoard Piece.J base
  | .noS => richStepBoard Piece.S base
  | .noZ => richStepBoard Piece.Z base
  | .noI => richStepBoard Piece.I base
  | .rich2 first second => richPairBoard first second base
  | .rich3 first second third => richTripleBoard first second third base
  | .rich4 first second third fourth => richQuadBoard first second third fourth base

/-- Exact symbolic carrier: the board is the exact denotation of one phase at
one base height, and that phase fits under the standard board ceiling. -/
def PhaseCarrier (T : Bag) (b : Board) : Prop :=
  ∃ (phase : ReservoirPhase) (base : ℕ),
    PhaseOK phase T ∧
    b = PhaseBoard phase base ∧
    base + PhaseMargin phase ≤ GameConfig.standard.rows

theorem reservoirProfile_le_margin (phase : ReservoirPhase) (base j : ℕ) :
    ReservoirProfile phase base j ≤ base + PhaseMargin phase := by
  cases phase <;> dsimp [ReservoirProfile, PhaseMargin] <;> (try split_ifs) <;> omega

theorem phaseCarrier_init :
    PhaseCarrier Bag.full Board.empty := by
  refine ⟨ReservoirPhase.emptyStart, 0, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · decide

def maxColHeightInField (cfg : GameConfig) (b : Board) : ℕ :=
  (Finset.range cfg.cols).sup (fun j => Board.colHeight b j)

theorem colHeight_eq_zero_of_wf_of_cols_le {cfg : GameConfig} {b : Board} {j : ℕ}
    (hWF : Board.WF cfg b) (hj : cfg.cols ≤ j) :
    Board.colHeight b j = 0 := by
  rw [Board.colHeight_eq_zero_iff_forall_not_mem]
  intro r hmem
  have hcol := hWF (j, r) hmem
  omega

theorem colHeight_le_of_maxColHeightInField {cfg : GameConfig} {b : Board} {H j : ℕ}
    (hWF : Board.WF cfg b)
    (hmax : maxColHeightInField cfg b ≤ H) :
    Board.colHeight b j ≤ H := by
  by_cases hj : j < cfg.cols
  · exact le_trans (Finset.le_sup (f := fun k => Board.colHeight b k)
      (Finset.mem_range.mpr hj)) hmax
  · rw [colHeight_eq_zero_of_wf_of_cols_le hWF (Nat.le_of_not_lt hj)]
    exact Nat.zero_le H

theorem reservoirProfile_skyline_height (phase : ReservoirPhase) (base : ℕ)
    (hcap : base + PhaseMargin phase ≤ GameConfig.standard.rows) :
    ∀ j, Board.colHeight
      (Board.skyline GameConfig.standard (ReservoirProfile phase base)) j
        ≤ GameConfig.standard.rows := by
  intro j
  by_cases hj : j < GameConfig.standard.cols
  · rw [Board.colHeight_skyline hj]
    exact le_trans (reservoirProfile_le_margin phase base j) hcap
  · rw [Board.colHeight_skyline_eq_zero (Nat.le_of_not_lt hj)]
    exact Nat.zero_le _

theorem openingBoard_height (p : Piece) :
    ∀ j, Board.colHeight (openingBoard p) j ≤ GameConfig.standard.rows := by
  intro j
  have hnotLost :
      ¬ Board.isLost GameConfig.standard (openingBoard p) := by
    dsimp [openingBoard]
    simpa [GameState.init] using
      standard_init_applyStep_not_lost_of_valid (phasePlacement p) (phasePlacement_valid p)
  exact Board.colHeight_le_rows_of_in_field
    ((Board.not_isLost_iff_forall_row_lt GameConfig.standard (openingBoard p)).mp hnotLost)
    j

theorem openingBoard_wf (p : Piece) :
    Board.WF GameConfig.standard (openingBoard p) := by
  dsimp [openingBoard]
  exact Placement.applyStep_wf (Board.empty_wf GameConfig.standard) (phasePlacement_valid p)

theorem openingBoard_low (p : Piece) :
    ∀ j, Board.colHeight (openingBoard p) j ≤ GameConfig.standard.rows - 4 := by
  intro j
  dsimp [openingBoard]
  have hzero : ∀ j, Board.colHeight (Board.empty) j ≤ 0 := by
    intro j
    simp [Board.empty]
  have hstep := Board.colHeight_applyStep_le_of_colHeight_le
    (cfg := GameConfig.standard) (b := Board.empty) (pl := phasePlacement p) (H := 0) hzero j
  omega

theorem openingBoard_height_le_four (p : Piece) :
    ∀ j, Board.colHeight (openingBoard p) j ≤ 4 := by
  intro j
  dsimp [openingBoard]
  have hzero : ∀ j, Board.colHeight (Board.empty) j ≤ 0 := by
    intro j
    simp [Board.empty]
  simpa using Board.colHeight_applyStep_le_of_colHeight_le
    (cfg := GameConfig.standard) (b := Board.empty) (pl := phasePlacement p) (H := 0) hzero j

theorem openingPairBoard_height (first second : Piece) :
    ∀ j, Board.colHeight (openingPairBoard first second) j ≤ GameConfig.standard.rows := by
  intro j
  have hnotLost :
      ¬ Board.isLost GameConfig.standard (openingPairBoard first second) := by
    dsimp [openingPairBoard]
    exact low_stack_safe (cfg := GameConfig.standard) (by decide)
      (openingBoard_wf first)
      (openingBoard_low first)
      (phasePlacement second)
      (phasePlacement_valid second)
  exact Board.colHeight_le_rows_of_in_field
    ((Board.not_isLost_iff_forall_row_lt GameConfig.standard
      (openingPairBoard first second)).mp hnotLost) j

theorem openingPairBoard_wf (first second : Piece) :
    Board.WF GameConfig.standard (openingPairBoard first second) := by
  dsimp [openingPairBoard]
  exact Placement.applyStep_wf (openingBoard_wf first) (phasePlacement_valid second)

theorem openingPairBoard_height_le_eight (first second : Piece) :
    ∀ j, Board.colHeight (openingPairBoard first second) j ≤ 8 := by
  intro j
  dsimp [openingPairBoard]
  have hstep := Board.colHeight_applyStep_le_of_colHeight_le
    (cfg := GameConfig.standard) (b := openingBoard first) (pl := phasePlacement second)
    (H := 4) (openingBoard_height_le_four first) j
  omega

theorem openingPairBoard_low (first second : Piece) :
    ∀ j, Board.colHeight (openingPairBoard first second) j ≤ GameConfig.standard.rows - 4 := by
  intro j
  dsimp [openingPairBoard]
  have hstep := Board.colHeight_applyStep_le_of_colHeight_le
    (cfg := GameConfig.standard) (b := openingBoard first) (pl := phasePlacement second)
    (H := 4) (openingBoard_height_le_four first) j
  omega

theorem openingTripleBoard_height_le_twelve (first second third : Piece) :
    ∀ j, Board.colHeight (openingTripleBoard first second third) j ≤ 12 := by
  intro j
  dsimp [openingTripleBoard]
  have hstep := Board.colHeight_applyStep_le_of_colHeight_le
    (cfg := GameConfig.standard) (b := openingPairBoard first second) (pl := phasePlacement third)
    (H := 8) (openingPairBoard_height_le_eight first second) j
  omega

theorem openingTripleBoard_wf (first second third : Piece) :
    Board.WF GameConfig.standard (openingTripleBoard first second third) := by
  dsimp [openingTripleBoard]
  exact Placement.applyStep_wf (openingPairBoard_wf first second) (phasePlacement_valid third)

theorem openingTripleBoard_low (first second third : Piece) :
    ∀ j, Board.colHeight (openingTripleBoard first second third) j ≤ GameConfig.standard.rows - 4 := by
  intro j
  have h := openingTripleBoard_height_le_twelve first second third j
  simp [GameConfig.standard_rows] at h ⊢
  omega

theorem openingTripleBoard_height (first second third : Piece) :
    ∀ j, Board.colHeight (openingTripleBoard first second third) j ≤ GameConfig.standard.rows := by
  intro j
  have hnotLost :
      ¬ Board.isLost GameConfig.standard (openingTripleBoard first second third) := by
    dsimp [openingTripleBoard]
    exact low_stack_safe (cfg := GameConfig.standard) (by decide)
      (openingPairBoard_wf first second)
      (openingPairBoard_low first second)
      (phasePlacement third)
      (phasePlacement_valid third)
  exact Board.colHeight_le_rows_of_in_field
    ((Board.not_isLost_iff_forall_row_lt GameConfig.standard
      (openingTripleBoard first second third)).mp hnotLost) j

theorem openingQuadBoard_height (first second third fourth : Piece) :
    ∀ j, Board.colHeight (openingQuadBoard first second third fourth) j ≤ GameConfig.standard.rows := by
  intro j
  have hnotLost :
      ¬ Board.isLost GameConfig.standard (openingQuadBoard first second third fourth) := by
    dsimp [openingQuadBoard]
    exact low_stack_safe (cfg := GameConfig.standard) (by decide)
      (openingTripleBoard_wf first second third)
      (openingTripleBoard_low first second third)
      (phasePlacement fourth)
      (phasePlacement_valid fourth)
  exact Board.colHeight_le_rows_of_in_field
    ((Board.not_isLost_iff_forall_row_lt GameConfig.standard
      (openingQuadBoard first second third fourth)).mp hnotLost) j

theorem openingQuadBoard_height_le_sixteen (first second third fourth : Piece) :
    ∀ j, Board.colHeight (openingQuadBoard first second third fourth) j ≤ 16 := by
  intro j
  dsimp [openingQuadBoard]
  have hstep := Board.colHeight_applyStep_le_of_colHeight_le
    (cfg := GameConfig.standard) (b := openingTripleBoard first second third)
    (pl := phasePlacement fourth)
    (H := 12) (openingTripleBoard_height_le_twelve first second third) j
  omega

theorem openingQuadBoard_wf (first second third fourth : Piece) :
    Board.WF GameConfig.standard (openingQuadBoard first second third fourth) := by
  dsimp [openingQuadBoard]
  exact Placement.applyStep_wf (openingTripleBoard_wf first second third)
    (phasePlacement_valid fourth)

theorem openingQuadBoard_low (first second third fourth : Piece) :
    ∀ j, Board.colHeight (openingQuadBoard first second third fourth) j ≤ GameConfig.standard.rows - 4 := by
  intro j
  have h := openingQuadBoard_height_le_sixteen first second third fourth j
  simp [GameConfig.standard_rows] at h ⊢
  omega

theorem openingPentaBoard_height (first second third fourth fifth : Piece) :
    ∀ j, Board.colHeight (openingPentaBoard first second third fourth fifth) j
      ≤ GameConfig.standard.rows := by
  intro j
  have hnotLost :
      ¬ Board.isLost GameConfig.standard
        (openingPentaBoard first second third fourth fifth) := by
    dsimp [openingPentaBoard]
    exact low_stack_safe (cfg := GameConfig.standard) (by decide)
      (openingQuadBoard_wf first second third fourth)
      (openingQuadBoard_low first second third fourth)
      (phasePlacement fifth)
      (phasePlacement_valid fifth)
  exact Board.colHeight_le_rows_of_in_field
    ((Board.not_isLost_iff_forall_row_lt GameConfig.standard
      (openingPentaBoard first second third fourth fifth)).mp hnotLost) j

theorem openingPentaBoard_wf (first second third fourth fifth : Piece) :
    Board.WF GameConfig.standard (openingPentaBoard first second third fourth fifth) := by
  dsimp [openingPentaBoard]
  exact Placement.applyStep_wf (openingQuadBoard_wf first second third fourth)
    (phasePlacement_valid fifth)

theorem openingPentaBoard_maxHeight_le_sixteen_of_nodup
    (first second third fourth fifth : Piece)
    (hnodup : [first, second, third, fourth, fifth].Nodup) :
    maxColHeightInField GameConfig.standard
      (openingPentaBoard first second third fourth fifth) ≤ 16 := by
  revert first second third fourth fifth
  native_decide

theorem openingPentaBoard_low_of_nodup (first second third fourth fifth : Piece)
    (hnodup : [first, second, third, fourth, fifth].Nodup) :
    ∀ j, Board.colHeight (openingPentaBoard first second third fourth fifth) j
      ≤ GameConfig.standard.rows - 4 := by
  intro j
  have hmax := openingPentaBoard_maxHeight_le_sixteen_of_nodup
    first second third fourth fifth hnodup
  have hle := colHeight_le_of_maxColHeightInField
    (openingPentaBoard_wf first second third fourth fifth) hmax (j := j)
  simp [GameConfig.standard_rows]
  exact hle

theorem openingHexaBoard_wf (first second third fourth fifth sixth : Piece) :
    Board.WF GameConfig.standard
      (openingHexaBoard first second third fourth fifth sixth) := by
  dsimp [openingHexaBoard]
  exact Placement.applyStep_wf (openingPentaBoard_wf first second third fourth fifth)
    (phasePlacement_valid sixth)

theorem openingHexaBoard_maxHeight_le_rows_of_nodup
    (first second third fourth fifth sixth : Piece)
    (hnodup : [first, second, third, fourth, fifth, sixth].Nodup) :
    maxColHeightInField GameConfig.standard
      (openingHexaBoard first second third fourth fifth sixth) ≤ GameConfig.standard.rows := by
  revert first second third fourth fifth sixth
  native_decide

theorem openingHexaBoard_height_of_nodup (first second third fourth fifth sixth : Piece)
    (hnodup : [first, second, third, fourth, fifth, sixth].Nodup) :
    ∀ j, Board.colHeight (openingHexaBoard first second third fourth fifth sixth) j
      ≤ GameConfig.standard.rows := by
  intro j
  exact colHeight_le_of_maxColHeightInField
    (openingHexaBoard_wf first second third fourth fifth sixth)
    (openingHexaBoard_maxHeight_le_rows_of_nodup
      first second third fourth fifth sixth hnodup)
    (j := j)

theorem richReadyBoard_wf (base : ℕ)
    (hcap : base + PhaseMargin ReservoirPhase.fullReady ≤ GameConfig.standard.rows) :
    Board.WF GameConfig.standard (richReadyBoard base) := by
  dsimp [richReadyBoard]
  exact Board.wf_of_isReservedWellSkyline
    (Board.isReservedWellSkyline_richStandard_surface (base := base) (by
      simp [PhaseMargin, GameConfig.standard_rows] at hcap ⊢
      omega))

theorem richReadyBoard_low (base : ℕ)
    (hcap : base + PhaseMargin ReservoirPhase.fullReady ≤ GameConfig.standard.rows) :
    ∀ j, Board.colHeight (richReadyBoard base) j ≤ GameConfig.standard.rows - 4 := by
  intro j
  dsimp [richReadyBoard, ReservoirProfile] at *
  by_cases hj : j < GameConfig.standard.cols
  · rw [Board.colHeight_skyline hj]
    simp [PhaseMargin] at hcap ⊢
    split_ifs <;> omega
  · rw [Board.colHeight_skyline_eq_zero (Nat.le_of_not_lt hj)]
    omega

theorem richStepBoard_height (p : Piece) (base : ℕ)
    (hcap : base + PhaseMargin ReservoirPhase.fullReady ≤ GameConfig.standard.rows) :
    ∀ j, Board.colHeight (richStepBoard p base) j ≤ GameConfig.standard.rows := by
  intro j
  have hnotLost :
      ¬ Board.isLost GameConfig.standard (richStepBoard p base) := by
    dsimp [richStepBoard]
    exact low_stack_safe (cfg := GameConfig.standard) (by decide)
      (richReadyBoard_wf base hcap)
      (richReadyBoard_low base hcap)
      (phasePlacement p)
      (phasePlacement_valid p)
  exact Board.colHeight_le_rows_of_in_field
    ((Board.not_isLost_iff_forall_row_lt GameConfig.standard (richStepBoard p base)).mp hnotLost)
    j

theorem richReadyBoard_colHeight_le_base_add_one (base : ℕ) :
    ∀ j, Board.colHeight (richReadyBoard base) j ≤ base + 1 := by
  intro j
  dsimp [richReadyBoard, ReservoirProfile]
  by_cases hj : j < GameConfig.standard.cols
  · rw [Board.colHeight_skyline hj]
    split_ifs <;> omega
  · rw [Board.colHeight_skyline_eq_zero (Nat.le_of_not_lt hj)]
    omega

theorem richStepBoard_colHeight_le_base_add_five (p : Piece) (base : ℕ) :
    ∀ j, Board.colHeight (richStepBoard p base) j ≤ base + 5 := by
  intro j
  dsimp [richStepBoard]
  have hstep := Board.colHeight_applyStep_le_of_colHeight_le
    (cfg := GameConfig.standard) (b := richReadyBoard base) (pl := phasePlacement p)
    (H := base + 1) (richReadyBoard_colHeight_le_base_add_one base) j
  omega

theorem richStepBoard_wf (p : Piece) (base : ℕ)
    (hcap : base + PhaseMargin ReservoirPhase.fullReady ≤ GameConfig.standard.rows) :
    Board.WF GameConfig.standard (richStepBoard p base) := by
  dsimp [richStepBoard]
  exact Placement.applyStep_wf (richReadyBoard_wf base hcap) (phasePlacement_valid p)

theorem richStepBoard_low (p : Piece) (base : ℕ)
    (hcap : base + PhaseMargin ReservoirPhase.fullReady ≤ GameConfig.standard.rows) :
    ∀ j, Board.colHeight (richStepBoard p base) j ≤ GameConfig.standard.rows - 4 := by
  intro j
  dsimp [richStepBoard]
  have hstep := Board.colHeight_applyStep_le_of_colHeight_le
    (cfg := GameConfig.standard) (b := richReadyBoard base) (pl := phasePlacement p)
    (H := base + 1) (richReadyBoard_colHeight_le_base_add_one base) j
  simp [PhaseMargin, GameConfig.standard_rows] at hcap ⊢
  omega

theorem richPairBoard_height (first second : Piece) (base : ℕ)
    (hcap : base + PhaseMargin ReservoirPhase.fullReady ≤ GameConfig.standard.rows) :
    ∀ j, Board.colHeight (richPairBoard first second base) j ≤ GameConfig.standard.rows := by
  intro j
  have hnotLost :
      ¬ Board.isLost GameConfig.standard (richPairBoard first second base) := by
    dsimp [richPairBoard]
    exact low_stack_safe (cfg := GameConfig.standard) (by decide)
      (richStepBoard_wf first base hcap)
      (richStepBoard_low first base hcap)
      (phasePlacement second)
      (phasePlacement_valid second)
  exact Board.colHeight_le_rows_of_in_field
    ((Board.not_isLost_iff_forall_row_lt GameConfig.standard
      (richPairBoard first second base)).mp hnotLost) j

theorem richPairBoard_wf (first second : Piece) (base : ℕ)
    (hcap : base + PhaseMargin ReservoirPhase.fullReady ≤ GameConfig.standard.rows) :
    Board.WF GameConfig.standard (richPairBoard first second base) := by
  dsimp [richPairBoard]
  exact Placement.applyStep_wf (richStepBoard_wf first base hcap) (phasePlacement_valid second)

theorem richPairBoard_low (first second : Piece) (base : ℕ)
    (hcap : base + PhaseMargin ReservoirPhase.fullReady ≤ GameConfig.standard.rows) :
    ∀ j, Board.colHeight (richPairBoard first second base) j ≤ GameConfig.standard.rows - 4 := by
  intro j
  dsimp [richPairBoard]
  have hstep := Board.colHeight_applyStep_le_of_colHeight_le
    (cfg := GameConfig.standard) (b := richStepBoard first base) (pl := phasePlacement second)
    (H := base + 5) (richStepBoard_colHeight_le_base_add_five first base) j
  simp [PhaseMargin, GameConfig.standard_rows] at hcap ⊢
  omega

theorem richPairBoard_colHeight_le_base_add_nine (first second : Piece) (base : ℕ) :
    ∀ j, Board.colHeight (richPairBoard first second base) j ≤ base + 9 := by
  intro j
  dsimp [richPairBoard]
  have hstep := Board.colHeight_applyStep_le_of_colHeight_le
    (cfg := GameConfig.standard) (b := richStepBoard first base) (pl := phasePlacement second)
    (H := base + 5) (richStepBoard_colHeight_le_base_add_five first base) j
  omega

theorem richTripleBoard_height (first second third : Piece) (base : ℕ)
    (hcap : base + PhaseMargin ReservoirPhase.fullReady ≤ GameConfig.standard.rows) :
    ∀ j, Board.colHeight (richTripleBoard first second third base) j ≤ GameConfig.standard.rows := by
  intro j
  have hnotLost :
      ¬ Board.isLost GameConfig.standard (richTripleBoard first second third base) := by
    dsimp [richTripleBoard]
    exact low_stack_safe (cfg := GameConfig.standard) (by decide)
      (richPairBoard_wf first second base hcap)
      (richPairBoard_low first second base hcap)
      (phasePlacement third)
      (phasePlacement_valid third)
  exact Board.colHeight_le_rows_of_in_field
    ((Board.not_isLost_iff_forall_row_lt GameConfig.standard
      (richTripleBoard first second third base)).mp hnotLost) j

theorem richTripleBoard_colHeight_le_base_add_thirteen
    (first second third : Piece) (base : ℕ) :
    ∀ j, Board.colHeight (richTripleBoard first second third base) j ≤ base + 13 := by
  intro j
  dsimp [richTripleBoard]
  have hstep := Board.colHeight_applyStep_le_of_colHeight_le
    (cfg := GameConfig.standard) (b := richPairBoard first second base) (pl := phasePlacement third)
    (H := base + 9) (richPairBoard_colHeight_le_base_add_nine first second base) j
  omega

theorem richTripleBoard_wf (first second third : Piece) (base : ℕ)
    (hcap : base + PhaseMargin ReservoirPhase.fullReady ≤ GameConfig.standard.rows) :
    Board.WF GameConfig.standard (richTripleBoard first second third base) := by
  dsimp [richTripleBoard]
  exact Placement.applyStep_wf (richPairBoard_wf first second base hcap)
    (phasePlacement_valid third)

theorem richTripleBoard_low (first second third : Piece) (base : ℕ)
    (hcap : base + PhaseMargin ReservoirPhase.fullReady ≤ GameConfig.standard.rows) :
    ∀ j, Board.colHeight (richTripleBoard first second third base) j
      ≤ GameConfig.standard.rows - 4 := by
  intro j
  have h := richTripleBoard_colHeight_le_base_add_thirteen first second third base j
  simp [PhaseMargin, GameConfig.standard_rows] at hcap ⊢
  omega

theorem richQuadBoard_height (first second third fourth : Piece) (base : ℕ)
    (hcap : base + PhaseMargin ReservoirPhase.fullReady ≤ GameConfig.standard.rows) :
    ∀ j, Board.colHeight (richQuadBoard first second third fourth base) j
      ≤ GameConfig.standard.rows := by
  intro j
  have hnotLost :
      ¬ Board.isLost GameConfig.standard (richQuadBoard first second third fourth base) := by
    dsimp [richQuadBoard]
    exact low_stack_safe (cfg := GameConfig.standard) (by decide)
      (richTripleBoard_wf first second third base hcap)
      (richTripleBoard_low first second third base hcap)
      (phasePlacement fourth)
      (phasePlacement_valid fourth)
  exact Board.colHeight_le_rows_of_in_field
    ((Board.not_isLost_iff_forall_row_lt GameConfig.standard
      (richQuadBoard first second third fourth base)).mp hnotLost) j

/-- Sharp per-column ceiling for the exact four-piece rich board: each of the
four fixed online placements adds at most four to any column, so on top of the
`base + 1` ready surface the quad board never exceeds `base + 17`.  This is the
top rung of the `+1/+5/+9/+13` rich ladder and the exact height the `rich4`
frontier step must account for.  Note `base + 17 ≤ rows = 20` already forces
`base ≤ 3`, so a fifth fixed placement (another `+4`) can reach `base + 21 > 20`:
the rich tail cannot stay under the ceiling without a line-clearing drain. -/
theorem richQuadBoard_colHeight_le_base_add_seventeen
    (first second third fourth : Piece) (base : ℕ) :
    ∀ j, Board.colHeight (richQuadBoard first second third fourth base) j ≤ base + 17 := by
  intro j
  dsimp [richQuadBoard]
  have hstep := Board.colHeight_applyStep_le_of_colHeight_le
    (cfg := GameConfig.standard) (b := richTripleBoard first second third base)
    (pl := phasePlacement fourth)
    (H := base + 13) (richTripleBoard_colHeight_le_base_add_thirteen first second third base) j
  omega

/-- Scalar (single-number) field-height ceiling for the four-piece rich board,
the form the drain and clear accounting consumes.  Immediate from the sharp
per-column `base + 17` bound via `Finset.sup_le`. -/
theorem richQuadBoard_maxColHeightInField_le (first second third fourth : Piece) (base : ℕ) :
    maxColHeightInField GameConfig.standard (richQuadBoard first second third fourth base)
      ≤ base + 17 := by
  unfold maxColHeightInField
  exact Finset.sup_le fun j _ =>
    richQuadBoard_colHeight_le_base_add_seventeen first second third fourth base j

/-- The reserved well (column 0) of the rich *reset* surface is empty.  The
`fullReady` profile pins column 0 to height 0 while every other column sits at
`base` (or `base + 1` at the two lips), so a vertical I dropped into column 0
lands on the floor.  This is the base case of the well-emptiness invariant that
lets the once-per-bag I-regulator clear the bottom rows and drain the rich tail
back down — the line-clearing move the fixed-placement ladder (which tops out at
`base + 17`) cannot supply on its own. -/
theorem richReadyBoard_well_empty (base : ℕ) :
    Board.colHeight (richReadyBoard base) 0 = 0 := by
  dsimp [richReadyBoard, ReservoirProfile]
  rw [Board.colHeight_skyline (by decide : (0 : ℕ) < GameConfig.standard.cols)]
  simp

/-- The reserved well (column 0) stays empty after one *non-I* filler lands on the
rich reset surface.  Every non-I online placement sits at column `≥ 1`
(`phasePlacement` puts O/T/L/J at col 1, S at 3, Z at 5; only the I-regulator
targets col 0), so it dodges the well, clears nothing (an empty column blocks
every row), and the exact-column frame lemma holds column 0 at its profile value
`0`.  This is the inductive step of the well-emptiness invariant: only the
once-per-bag I ever fills the well, which is precisely what lets it clear the
bottom rows and drain the rich tail. -/
theorem richStepBoard_well_empty (p : Piece) (hp : p ≠ Piece.I) (base : ℕ) :
    Board.colHeight (richStepBoard p base) 0 = 0 := by
  have hh0 : ReservoirProfile ReservoirPhase.fullReady base 0 = 0 := by
    simp [ReservoirProfile]
  have hcol : 1 ≤ (phasePlacement p).col := by
    cases p with
    | I => exact absurd rfl hp
    | _ => decide
  have havoid :
      ∀ cell ∈ (phasePlacement p).shapeUp, (phasePlacement p).col + cell.1 ≠ 0 := by
    intro cell _
    omega
  calc Board.colHeight (richStepBoard p base) 0
      = ReservoirProfile ReservoirPhase.fullReady base 0 :=
        Board.colHeight_applyStep_skyline_eq_of_avoid_well
          (cfg := GameConfig.standard)
          (h := ReservoirProfile ReservoirPhase.fullReady base)
          (pl := phasePlacement p) (w := 0) (m := 0)
          (by decide) (by decide) hh0 havoid havoid
    _ = 0 := hh0

/-- The reserved well (column 0) stays empty after a *second* non-I filler lands on
the rich surface.  `richStepBoard` is no longer a skyline, so the skyline frame lemma
no longer applies; instead the inductive step rides on the *general* empty-well bricks
(`fullRows_place_eq_empty_of_well` + `colHeight_applyStep_eq_of_not_col_of_no_fullRows`),
which hold on the arbitrary boards a bag passes through between resets.  Both fillers are
non-I, so each sits at column `≥ 1`, dodges the well, clears nothing (an empty column
blocks every row), and column 0 is held at its prior value `0`.  This carries the
well-emptiness invariant one rung up the rich ladder toward the once-per-bag I-drain. -/
theorem richPairBoard_well_empty (first second : Piece)
    (h1 : first ≠ Piece.I) (h2 : second ≠ Piece.I) (base : ℕ) :
    Board.colHeight (richPairBoard first second base) 0 = 0 := by
  have hb0 : Board.colHeight (richStepBoard first base) 0 = 0 :=
    richStepBoard_well_empty first h1 base
  have hcol : 1 ≤ (phasePlacement second).col := by
    cases second with
    | I => exact absurd rfl h2
    | _ => decide
  have havoid :
      ∀ cell ∈ (phasePlacement second).shapeUp, (phasePlacement second).col + cell.1 ≠ 0 := by
    intro cell _
    omega
  calc Board.colHeight (richPairBoard first second base) 0
      = Board.colHeight (richStepBoard first base) 0 :=
        Board.colHeight_applyStep_eq_of_not_col_of_no_fullRows
          (cfg := GameConfig.standard) (b := richStepBoard first base)
          (pl := phasePlacement second) (j := 0)
          havoid
          (Board.fullRows_place_eq_empty_of_well
            (cfg := GameConfig.standard) (b := richStepBoard first base)
            (pl := phasePlacement second) (w := 0)
            (by decide) hb0 havoid)
    _ = 0 := hb0

theorem phaseCarrier_height {T : Bag} {b : Board}
    (h : PhaseCarrier T b) :
    ∀ j, Board.colHeight b j ≤ GameConfig.standard.rows := by
  intro j
  rcases h with ⟨phase, base, hPhase, hb, hcap⟩
  rw [hb]
  cases phase
  · simp [PhaseBoard, Board.empty]
  · exact openingBoard_height Piece.O j
  · exact openingBoard_height Piece.T j
  · exact openingBoard_height Piece.L j
  · exact openingBoard_height Piece.J j
  · exact openingBoard_height Piece.S j
  · exact openingBoard_height Piece.Z j
  · exact openingBoard_height Piece.I j
  · exact openingPairBoard_height _ _ j
  · exact openingTripleBoard_height _ _ _ j
  · exact openingQuadBoard_height _ _ _ _ j
  · exact openingPentaBoard_height _ _ _ _ _ j
  · exact openingHexaBoard_height_of_nodup _ _ _ _ _ _
      (phaseOK_start6_nodup hPhase) j
  · exact reservoirProfile_skyline_height ReservoirPhase.fullReady base hcap j
  · exact richStepBoard_height Piece.O base hcap j
  · exact richStepBoard_height Piece.T base hcap j
  · exact richStepBoard_height Piece.L base hcap j
  · exact richStepBoard_height Piece.J base hcap j
  · exact richStepBoard_height Piece.S base hcap j
  · exact richStepBoard_height Piece.Z base hcap j
  · exact richStepBoard_height Piece.I base hcap j
  · exact richPairBoard_height _ _ base hcap j
  · exact richTripleBoard_height _ _ _ base hcap j
  · exact richQuadBoard_height _ _ _ _ base hcap j

/-- A finite directed edge in the symbolic phase graph. -/
structure PhaseEdge where
  src : ReservoirPhase
  piece : Piece
  dst : ReservoirPhase
  placement : Placement
  baseOut : ℕ → ℕ

/-- First draft of the phase table.  It only pins the rich `fullReady` exits;
`phaseEdge_exists` below records that the table is not yet complete. -/
def phaseEdge? (phase : ReservoirPhase) (p : Piece) : Option PhaseEdge :=
  match phase, p with
  | .emptyStart, Piece.O =>
      some {
        src := .emptyStart
        piece := Piece.O
        dst := .startO
        placement := phasePlacement Piece.O
        baseOut := fun _ => 0
      }
  | .emptyStart, Piece.T =>
      some {
        src := .emptyStart
        piece := Piece.T
        dst := .startT
        placement := phasePlacement Piece.T
        baseOut := fun _ => 0
      }
  | .emptyStart, Piece.L =>
      some {
        src := .emptyStart
        piece := Piece.L
        dst := .startL
        placement := phasePlacement Piece.L
        baseOut := fun _ => 0
      }
  | .emptyStart, Piece.J =>
      some {
        src := .emptyStart
        piece := Piece.J
        dst := .startJ
        placement := phasePlacement Piece.J
        baseOut := fun _ => 0
      }
  | .emptyStart, Piece.S =>
      some {
        src := .emptyStart
        piece := Piece.S
        dst := .startS
        placement := phasePlacement Piece.S
        baseOut := fun _ => 0
      }
  | .emptyStart, Piece.Z =>
      some {
        src := .emptyStart
        piece := Piece.Z
        dst := .startZ
        placement := phasePlacement Piece.Z
        baseOut := fun _ => 0
      }
  | .emptyStart, Piece.I =>
      some {
        src := .emptyStart
        piece := Piece.I
        dst := .startI
        placement := phasePlacement Piece.I
        baseOut := fun _ => 0
      }
  | .startO, p =>
      some {
        src := .startO
        piece := p
        dst := .start2 Piece.O p
        placement := phasePlacement p
        baseOut := fun _ => 0
      }
  | .startT, p =>
      some {
        src := .startT
        piece := p
        dst := .start2 Piece.T p
        placement := phasePlacement p
        baseOut := fun _ => 0
      }
  | .startL, p =>
      some {
        src := .startL
        piece := p
        dst := .start2 Piece.L p
        placement := phasePlacement p
        baseOut := fun _ => 0
      }
  | .startJ, p =>
      some {
        src := .startJ
        piece := p
        dst := .start2 Piece.J p
        placement := phasePlacement p
        baseOut := fun _ => 0
      }
  | .startS, p =>
      some {
        src := .startS
        piece := p
        dst := .start2 Piece.S p
        placement := phasePlacement p
        baseOut := fun _ => 0
      }
  | .startZ, p =>
      some {
        src := .startZ
        piece := p
        dst := .start2 Piece.Z p
        placement := phasePlacement p
        baseOut := fun _ => 0
      }
  | .startI, p =>
      some {
        src := .startI
        piece := p
        dst := .start2 Piece.I p
        placement := phasePlacement p
        baseOut := fun _ => 0
      }
  | .start2 first second, p =>
      some {
        src := .start2 first second
        piece := p
        dst := .start3 first second p
        placement := phasePlacement p
        baseOut := fun _ => 0
      }
  | .start3 first second third, p =>
      some {
        src := .start3 first second third
        piece := p
        dst := .start4 first second third p
        placement := phasePlacement p
        baseOut := fun _ => 0
      }
  | .start4 first second third fourth, p =>
      some {
        src := .start4 first second third fourth
        piece := p
        dst := .start5 first second third fourth p
        placement := phasePlacement p
        baseOut := fun _ => 0
      }
  | .start5 first second third fourth fifth, p =>
      some {
        src := .start5 first second third fourth fifth
        piece := p
        dst := .start6 first second third fourth fifth p
        placement := phasePlacement p
        baseOut := fun _ => 0
      }
  | .fullReady, Piece.O =>
      some {
        src := .fullReady
        piece := Piece.O
        dst := .noO
        placement := phasePlacement Piece.O
        baseOut := fun base => base
      }
  | .fullReady, Piece.T =>
      some {
        src := .fullReady
        piece := Piece.T
        dst := .noT
        placement := phasePlacement Piece.T
        baseOut := fun base => base
      }
  | .fullReady, Piece.L =>
      some {
        src := .fullReady
        piece := Piece.L
        dst := .noL
        placement := phasePlacement Piece.L
        baseOut := fun base => base
      }
  | .fullReady, Piece.J =>
      some {
        src := .fullReady
        piece := Piece.J
        dst := .noJ
        placement := phasePlacement Piece.J
        baseOut := fun base => base
      }
  | .fullReady, Piece.S =>
      some {
        src := .fullReady
        piece := Piece.S
        dst := .noS
        placement := phasePlacement Piece.S
        baseOut := fun base => base
      }
  | .fullReady, Piece.Z =>
      some {
        src := .fullReady
        piece := Piece.Z
        dst := .noZ
        placement := phasePlacement Piece.Z
        baseOut := fun base => base
      }
  | .fullReady, Piece.I =>
      some {
        src := .fullReady
        piece := Piece.I
        dst := .noI
        placement := phasePlacement Piece.I
        baseOut := fun base => base
      }
  | .noO, p =>
      some {
        src := .noO
        piece := p
        dst := .rich2 Piece.O p
        placement := phasePlacement p
        baseOut := fun base => base
      }
  | .noT, p =>
      some {
        src := .noT
        piece := p
        dst := .rich2 Piece.T p
        placement := phasePlacement p
        baseOut := fun base => base
      }
  | .noL, p =>
      some {
        src := .noL
        piece := p
        dst := .rich2 Piece.L p
        placement := phasePlacement p
        baseOut := fun base => base
      }
  | .noJ, p =>
      some {
        src := .noJ
        piece := p
        dst := .rich2 Piece.J p
        placement := phasePlacement p
        baseOut := fun base => base
      }
  | .noS, p =>
      some {
        src := .noS
        piece := p
        dst := .rich2 Piece.S p
        placement := phasePlacement p
        baseOut := fun base => base
      }
  | .noZ, p =>
      some {
        src := .noZ
        piece := p
        dst := .rich2 Piece.Z p
        placement := phasePlacement p
        baseOut := fun base => base
      }
  | .noI, p =>
      some {
        src := .noI
        piece := p
        dst := .rich2 Piece.I p
        placement := phasePlacement p
        baseOut := fun base => base
      }
  | .rich2 first second, p =>
      some {
        src := .rich2 first second
        piece := p
        dst := .rich3 first second p
        placement := phasePlacement p
        baseOut := fun base => base
      }
  | .rich3 first second third, p =>
      some {
        src := .rich3 first second third
        piece := p
        dst := .rich4 first second third p
        placement := phasePlacement p
        baseOut := fun base => base
      }
  | _, _ => none

theorem phaseEdge_exists_emptyStart {T : Bag} {p : Piece}
    (_hPhase : PhaseOK ReservoirPhase.emptyStart T) (_hp : p ∈ T) :
    ∃ edge : PhaseEdge, phaseEdge? ReservoirPhase.emptyStart p = some edge := by
  cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩

theorem phaseEdge_exists_fullReady {T : Bag} {p : Piece}
    (_hPhase : PhaseOK ReservoirPhase.fullReady T) (_hp : p ∈ T) :
    ∃ edge : PhaseEdge, phaseEdge? ReservoirPhase.fullReady p = some edge := by
  cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩

theorem phaseEdge_correct_emptyStart {p : Piece} {edge : PhaseEdge}
    (hedge : phaseEdge? ReservoirPhase.emptyStart p = some edge)
    {base : ℕ}
    (_hcap : base + PhaseMargin ReservoirPhase.emptyStart ≤ GameConfig.standard.rows) :
    edge.placement.piece = p ∧
    edge.placement.Valid GameConfig.standard ∧
    Placement.applyStep GameConfig.standard
      (PhaseBoard ReservoirPhase.emptyStart base)
      edge.placement
    =
    PhaseBoard edge.dst (edge.baseOut base) ∧
    edge.baseOut base + PhaseMargin edge.dst ≤ GameConfig.standard.rows := by
  cases p <;>
    dsimp [phaseEdge?, PhaseBoard, openingBoard, phasePlacement, PhaseMargin] at hedge ⊢
  · cases hedge
    exact ⟨rfl, Board.valid_O (by decide), rfl, by simp⟩
  · cases hedge
    exact ⟨rfl, Board.valid_vertI (by decide), rfl, by simp⟩
  · cases hedge
    exact ⟨rfl, Board.valid_S (by decide), rfl, by simp⟩
  · cases hedge
    exact ⟨rfl, Board.valid_Z (by decide), rfl, by simp⟩
  · cases hedge
    exact ⟨rfl, Board.valid_T (by decide), rfl, by simp⟩
  · cases hedge
    exact ⟨rfl, Board.valid_L (by decide), rfl, by simp⟩
  · cases hedge
    exact ⟨rfl, Board.valid_J (by decide), rfl, by simp⟩

theorem phaseEdge_correct_fullReady {p : Piece} {edge : PhaseEdge}
    (hedge : phaseEdge? ReservoirPhase.fullReady p = some edge)
    {base : ℕ}
    (hcap : base + PhaseMargin ReservoirPhase.fullReady ≤ GameConfig.standard.rows) :
    edge.placement.piece = p ∧
    edge.placement.Valid GameConfig.standard ∧
    Placement.applyStep GameConfig.standard
      (PhaseBoard ReservoirPhase.fullReady base)
      edge.placement
    =
    PhaseBoard edge.dst (edge.baseOut base) ∧
    edge.baseOut base + PhaseMargin edge.dst ≤ GameConfig.standard.rows := by
  cases p <;>
    dsimp [phaseEdge?, PhaseBoard, richStepBoard, PhaseMargin] at hedge ⊢
  · cases hedge
    exact ⟨phasePlacement_piece Piece.O, phasePlacement_valid Piece.O, rfl, by
      simp [PhaseMargin, GameConfig.standard_rows] at hcap ⊢
      omega⟩
  · cases hedge
    exact ⟨phasePlacement_piece Piece.I, phasePlacement_valid Piece.I, rfl, by
      simp [PhaseMargin, GameConfig.standard_rows] at hcap ⊢
      omega⟩
  · cases hedge
    exact ⟨phasePlacement_piece Piece.S, phasePlacement_valid Piece.S, rfl, by
      simp [PhaseMargin, GameConfig.standard_rows] at hcap ⊢
      omega⟩
  · cases hedge
    exact ⟨phasePlacement_piece Piece.Z, phasePlacement_valid Piece.Z, rfl, by
      simp [PhaseMargin, GameConfig.standard_rows] at hcap ⊢
      omega⟩
  · cases hedge
    exact ⟨phasePlacement_piece Piece.T, phasePlacement_valid Piece.T, rfl, by
      simp [PhaseMargin, GameConfig.standard_rows] at hcap ⊢
      omega⟩
  · cases hedge
    exact ⟨phasePlacement_piece Piece.L, phasePlacement_valid Piece.L, rfl, by
      simp [PhaseMargin, GameConfig.standard_rows] at hcap ⊢
      omega⟩
  · cases hedge
    exact ⟨phasePlacement_piece Piece.J, phasePlacement_valid Piece.J, rfl, by
      simp [PhaseMargin, GameConfig.standard_rows] at hcap ⊢
      omega⟩

/-- The remaining completion certificate.  The exact-board table now handles the
opening branch through six consumed pieces.  The rich branch is still delegated
from `rich4`; the next refinement should add rich five/six-piece phases and
move the delegated rich case to the singleton reset point as well. -/
structure PhaseGraphCompletion where
  frontier_step :
    ∀ {phase : ReservoirPhase} {T : Bag} {p : Piece} {base : ℕ},
      CompletionPhase phase →
      PhaseOK phase T → p ∈ T →
      base + PhaseMargin phase ≤ GameConfig.standard.rows →
      ∃ pl : Placement,
        pl.piece = p ∧
        pl.Valid GameConfig.standard ∧
        PhaseCarrier (T.draw p)
          (Placement.applyStep GameConfig.standard
            (PhaseBoard phase base)
            { pl with piece := p })

/-- Graph-completeness obligation: every non-completion phase with a legal draw
has a table edge. -/
theorem phaseEdge_exists_consumed
    {phase : ReservoirPhase} {T : Bag} {p : Piece}
    (hstart : phase ≠ ReservoirPhase.emptyStart)
    (hfull : phase ≠ ReservoirPhase.fullReady)
    (hcompletion : ¬ CompletionPhase phase)
    (hPhase : PhaseOK phase T) (hp : p ∈ T) :
    ∃ edge : PhaseEdge, phaseEdge? phase p = some edge := by
  cases phase
  · exact False.elim (hstart rfl)
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · exact False.elim (hcompletion (Or.inl ⟨_, _, _, _, _, _, rfl⟩))
  · exact False.elim (hfull rfl)
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · cases p <;> dsimp [phaseEdge?] <;> exact ⟨_, rfl⟩
  · exact False.elim (hcompletion (Or.inr ⟨_, _, _, _, rfl⟩))

theorem phaseEdge_exists
    {phase : ReservoirPhase} {T : Bag} {p : Piece}
    (hcompletion : ¬ CompletionPhase phase)
    (hPhase : PhaseOK phase T) (hp : p ∈ T) :
    ∃ edge : PhaseEdge, phaseEdge? phase p = some edge := by
  cases phase
  · exact phaseEdge_exists_emptyStart hPhase hp
  · exact phaseEdge_exists_consumed (by decide) (by decide) hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by decide) (by decide) hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by decide) (by decide) hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by decide) (by decide) hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by decide) (by decide) hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by decide) (by decide) hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by decide) (by decide) hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by intro h; cases h) (by intro h; cases h)
      hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by intro h; cases h) (by intro h; cases h)
      hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by intro h; cases h) (by intro h; cases h)
      hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by intro h; cases h) (by intro h; cases h)
      hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by intro h; cases h) (by intro h; cases h)
      hcompletion hPhase hp
  · exact phaseEdge_exists_fullReady hPhase hp
  · exact phaseEdge_exists_consumed (by decide) (by decide) hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by decide) (by decide) hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by decide) (by decide) hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by decide) (by decide) hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by decide) (by decide) hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by decide) (by decide) hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by decide) (by decide) hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by intro h; cases h) (by intro h; cases h)
      hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by intro h; cases h) (by intro h; cases h)
      hcompletion hPhase hp
  · exact phaseEdge_exists_consumed (by intro h; cases h) (by intro h; cases h)
      hcompletion hPhase hp

/-- Edge-geometry obligation: a table edge really rewrites one exact phase board
into the next exact phase board, with a valid placement and enough headroom. -/
theorem phaseEdge_correct
    {phase : ReservoirPhase} {p : Piece} {edge : PhaseEdge}
    (hedge : phaseEdge? phase p = some edge)
    {base : ℕ}
    (hcap : base + PhaseMargin phase ≤ GameConfig.standard.rows) :
    edge.placement.piece = p ∧
    edge.placement.Valid GameConfig.standard ∧
    Placement.applyStep GameConfig.standard
      (PhaseBoard phase base)
      edge.placement
    =
    PhaseBoard edge.dst (edge.baseOut base) ∧
    edge.baseOut base + PhaseMargin edge.dst ≤ GameConfig.standard.rows := by
  cases phase
  · exact phaseEdge_correct_emptyStart hedge hcap
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, openingPairBoard, PhaseMargin] at hedge ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by simp⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, openingPairBoard, PhaseMargin] at hedge ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by simp⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, openingPairBoard, PhaseMargin] at hedge ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by simp⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, openingPairBoard, PhaseMargin] at hedge ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by simp⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, openingPairBoard, PhaseMargin] at hedge ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by simp⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, openingPairBoard, PhaseMargin] at hedge ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by simp⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, openingPairBoard, PhaseMargin] at hedge ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by simp⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, openingTripleBoard, PhaseMargin] at hedge ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by simp⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, openingQuadBoard, PhaseMargin] at hedge ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by simp⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, openingPentaBoard, PhaseMargin] at hedge ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by simp⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, openingHexaBoard, PhaseMargin] at hedge ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by
        simp⟩
  · cases p <;> dsimp [phaseEdge?] at hedge <;> cases hedge
  · exact phaseEdge_correct_fullReady hedge hcap
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, richPairBoard, PhaseMargin] at hedge hcap ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by omega⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, richPairBoard, PhaseMargin] at hedge hcap ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by omega⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, richPairBoard, PhaseMargin] at hedge hcap ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by omega⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, richPairBoard, PhaseMargin] at hedge hcap ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by omega⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, richPairBoard, PhaseMargin] at hedge hcap ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by omega⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, richPairBoard, PhaseMargin] at hedge hcap ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by omega⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, richPairBoard, PhaseMargin] at hedge hcap ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by omega⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, richTripleBoard, PhaseMargin] at hedge hcap ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by omega⟩
  · cases p <;>
      dsimp [phaseEdge?, PhaseBoard, richQuadBoard, PhaseMargin] at hedge hcap ⊢ <;>
      cases hedge <;>
      exact ⟨phasePlacement_piece _, phasePlacement_valid _, rfl, by omega⟩
  · cases p <;> dsimp [phaseEdge?] at hedge <;> cases hedge

/-- Bag arithmetic obligation: following a graph edge lands in a phase compatible
with the drawn bag. -/
theorem phaseOK_step
    {phase : ReservoirPhase} {p : Piece} {edge : PhaseEdge} {T : Bag}
    (hedge : phaseEdge? phase p = some edge)
    (hPhase : PhaseOK phase T)
    (hp : p ∈ T) :
    PhaseOK edge.dst (T.draw p) := by
  cases phase <;> cases p <;> dsimp [phaseEdge?, PhaseOK] at hedge hPhase ⊢
  all_goals try cases hedge
  all_goals
    try
      rcases hPhase with ⟨hT, hsecond, hthird, hfourth, hfifth⟩
      subst T
      simp_all
    try
      rcases hPhase with ⟨hT, hsecond, hthird, hfourth⟩
      subst T
      simp_all
    try
      rcases hPhase with ⟨hT, hsecond, hthird⟩
      subst T
      simp_all
    try
      rcases hPhase with ⟨hT, hsecond⟩
      subst T
      simp_all
    try
      subst T
      simp_all

/-- The central symbolic closure theorem.  Its proof is pure plumbing once the
finite graph-completeness, edge-geometry, and bag-arithmetic obligations above
are available. -/
theorem phase_step (C : PhaseGraphCompletion)
    {T : Bag} {b : Board}
    (h : PhaseCarrier T b)
    (p : Piece)
    (hp : p ∈ T) :
    ∃ pl : Placement,
      pl.piece = p ∧
      pl.Valid GameConfig.standard ∧
      PhaseCarrier (T.draw p)
        (Placement.applyStep GameConfig.standard b { pl with piece := p }) := by
  rcases h with ⟨phase, base, hPhase, hb, hcap⟩
  classical
  by_cases hcompletion : CompletionPhase phase
  · obtain ⟨pl, hpiece, hvalid, hnext⟩ :=
      C.frontier_step hcompletion hPhase hp hcap
    refine ⟨pl, hpiece, hvalid, ?_⟩
    rw [hb]
    exact hnext
  obtain ⟨edge, hedge⟩ := phaseEdge_exists hcompletion hPhase hp
  obtain ⟨hpiece, hvalid, hstepEq, hcapNext⟩ := phaseEdge_correct hedge hcap
  refine ⟨edge.placement, hpiece, hvalid, ?_⟩
  refine ⟨edge.dst, edge.baseOut base, ?_, ?_, hcapNext⟩
  · exact phaseOK_step hedge hPhase hp
  · rw [hb]
    rw [Placement.applyStep_with_piece_self
      (cfg := GameConfig.standard)
      (b := PhaseBoard phase base)
      edge.placement
      hpiece]
    exact hstepEq

def phaseMove (T : Bag) (b : Board) (p : Piece) (pl : Placement) : Prop :=
  pl.piece = p ∧
  pl.Valid GameConfig.standard ∧
  PhaseCarrier (T.draw p)
    (Placement.applyStep GameConfig.standard b { pl with piece := p })

def phaseGraphCert (C : PhaseGraphCompletion) : RelationalAtlasCert where
  P := PhaseCarrier
  Move := phaseMove
  init := phaseCarrier_init
  height := fun _ _ h => phaseCarrier_height h
  total := by
    intro T b p hP hp
    obtain ⟨pl, hpiece, hvalid, hnext⟩ := phase_step C hP p hp
    exact ⟨pl, hpiece, hvalid, hnext⟩
  piece := by
    intro T b p pl hmove
    exact hmove.1
  valid := by
    intro T b p pl _hP _hp hmove
    exact hmove.2.1
  step := by
    intro T b p pl _hP _hp hmove
    exact hmove.2.2

/-- Conditional end-to-end theorem for the bag-indexed symbolic phase graph. -/
theorem onlinePhaseGraph_solves_tetris (C : PhaseGraphCompletion) : TetrisSolvableValid :=
  (phaseGraphCert C).solved

end PhaseGraph

end

end OnlineReservoir
end Tetris
