import Mathlib
import Proofs.Model.Game
import Proofs.Survival.Survival

/-!
# Lasso certificates: closed cycles from checkable tables

A **lasso certificate** is a finite association table `(state, placement)`
whose induced lookup policy keeps every listed state legal, non-losing, and
inside the table. `checkTable` is the `Bool`-valued (hence `decide`-friendly)
closure check; `ClosedCycle.ofTable` is its soundness theorem, converting a
passing table into a `ClosedCycle` — the M2/M3 infinite-play artifact.

This is the certificate format for *cooperative* (single-successor) survival
witnesses: the policy chooses both the draw order within each bag and the
placement, so closure is required along one trajectory only, not under all
seven piece branches (contrast `Atlas.IsClosedOn`, which is the adversarial,
all-piece closure). The intended inhabitant is a periodic loop through
`GameState.init` — see `Proofs/Experiments/CooperativeLasso.lean`.

Duplicate keys in the table are harmless: `tablePolicy` plays the *first*
matching entry, and every check is stated through `tablePolicy`, so shadowed
entries are simply checked against the placement that is actually played.
No `Nodup` side condition is needed.
-/

namespace Tetris

/-- The states visited when playing a placement list from `g`: the list
`[g, g.step pl₀, (g.step pl₀).step pl₁, …]`, one state per placement (the
final landing state is *not* included — for a loop it equals `g` again). -/
def scanStates (cfg : GameConfig) : GameState → List Placement → List GameState
  | _, [] => []
  | g, pl :: ps => g :: scanStates cfg (g.step cfg pl) ps

/-- `scanStates` visits one state per placement. -/
@[simp] theorem scanStates_length (cfg : GameConfig) (g : GameState)
    (ps : List Placement) : (scanStates cfg g ps).length = ps.length := by
  induction ps generalizing g with
  | nil => rfl
  | cons pl ps ih => simp [scanStates, ih]

/-- The lasso table induced by a placement list: each visited state paired
with the placement played from it. -/
def lassoTable (cfg : GameConfig) (g : GameState) (ps : List Placement) :
    List (GameState × Placement) :=
  (scanStates cfg g ps).zip ps

/-- The lookup policy of a table: play the first matching entry's placement
(an arbitrary default off-table — off-table states are never reached). Since
`Policy cfg` ignores the config, this is a `Policy cfg` for every `cfg`. -/
def tablePolicy (t : List (GameState × Placement)) : GameState → Placement :=
  fun g => ((t.find? (fun e => decide (e.1 = g))).map Prod.snd).getD
    ⟨Piece.O, 0, 0⟩

/-- One entry's closure check, stated through `tablePolicy` (the placement
actually played at this state): the drawn piece is bag-legal, the placement is
in-bounds, the state is not lost, and the successor is again a table key. -/
def checkEntry (cfg : GameConfig) (t : List (GameState × Placement))
    (g : GameState) : Bool :=
  decide (g.bag.canDraw (tablePolicy t g).piece) &&
  decide ((tablePolicy t g).Valid cfg) &&
  decide (¬ g.lost cfg) &&
  t.any (fun e' => decide (e'.1 = g.step cfg (tablePolicy t g)))

/-- The full table check: every listed state passes `checkEntry`. -/
def checkTable (cfg : GameConfig) (t : List (GameState × Placement)) : Bool :=
  t.all (fun e => checkEntry cfg t e.1)

/-- Unpack a passing `checkEntry` into its four propositional obligations. -/
theorem checkEntry_spec {cfg : GameConfig} {t : List (GameState × Placement)}
    {g : GameState} (h : checkEntry cfg t g = true) :
    g.bag.canDraw (tablePolicy t g).piece ∧
    (tablePolicy t g).Valid cfg ∧
    ¬ g.lost cfg ∧
    g.step cfg (tablePolicy t g) ∈ t.map Prod.fst := by
  unfold checkEntry at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.any_eq_true] at h
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, e', he', hstep⟩ := h
  exact ⟨h1, h2, h3, List.mem_map.mpr ⟨e', he', hstep⟩⟩

/-- **Soundness of the table check**: a passing table is a `ClosedCycle` on
its key set, under its lookup policy. The lasso certificate, materialised. -/
def ClosedCycle.ofTable (cfg : GameConfig) (t : List (GameState × Placement))
    (h : checkTable cfg t = true) : ClosedCycle cfg where
  states := (t.map Prod.fst).toFinset
  policy := tablePolicy t
  valid := by
    intro s hs
    obtain ⟨e, he, rfl⟩ := List.mem_map.mp (List.mem_toFinset.mp hs)
    exact (checkEntry_spec (List.all_eq_true.mp h e he)).2.1
  legal_draw := by
    intro s hs
    obtain ⟨e, he, rfl⟩ := List.mem_map.mp (List.mem_toFinset.mp hs)
    exact (checkEntry_spec (List.all_eq_true.mp h e he)).1
  not_lost := by
    intro s hs
    obtain ⟨e, he, rfl⟩ := List.mem_map.mp (List.mem_toFinset.mp hs)
    exact (checkEntry_spec (List.all_eq_true.mp h e he)).2.2.1
  closed := by
    intro s hs
    obtain ⟨e, he, rfl⟩ := List.mem_map.mp (List.mem_toFinset.mp hs)
    exact List.mem_toFinset.mpr
      (checkEntry_spec (List.all_eq_true.mp h e he)).2.2.2

/-- The states of `ofTable` are exactly the table keys. -/
@[simp] theorem ClosedCycle.ofTable_states (cfg : GameConfig)
    (t : List (GameState × Placement)) (h : checkTable cfg t = true) :
    (ClosedCycle.ofTable cfg t h).states = (t.map Prod.fst).toFinset := rfl

/-- The policy of `ofTable` is the table's lookup policy. -/
@[simp] theorem ClosedCycle.ofTable_policy (cfg : GameConfig)
    (t : List (GameState × Placement)) (h : checkTable cfg t = true) :
    (ClosedCycle.ofTable cfg t h).policy = tablePolicy t := rfl

/-- **A passing table's policy survives forever from any table key.** The
end-to-end lasso theorem: `checkTable` + key membership ⇒ infinite play. -/
theorem survivesForever_of_checkTable {cfg : GameConfig}
    {t : List (GameState × Placement)} (h : checkTable cfg t = true)
    {g0 : GameState} (hg0 : g0 ∈ t.map Prod.fst) :
    SurvivesForever cfg (tablePolicy t) g0 :=
  closed_cycle_survives (ClosedCycle.ofTable cfg t h)
    (List.mem_toFinset.mpr hg0)

/-- Existence form at `init`: a passing table keyed at `GameState.init`
yields a policy surviving forever from the initial state. -/
theorem exists_survivesForever_of_checkTable {cfg : GameConfig}
    {t : List (GameState × Placement)} (h : checkTable cfg t = true)
    (hinit : GameState.init ∈ t.map Prod.fst) :
    ∃ π : Policy cfg, SurvivesForever cfg π GameState.init :=
  ⟨tablePolicy t, survivesForever_of_checkTable h hinit⟩

/-- The first visited state heads its own lasso table: `g` is a key of
`lassoTable cfg g (pl :: ps)`. Discharges the `init ∈ keys` obligation for
stemless loops without any computation. -/
theorem lassoTable_head_mem (cfg : GameConfig) (g : GameState)
    (pl : Placement) (ps : List Placement) :
    g ∈ (lassoTable cfg g (pl :: ps)).map Prod.fst := by
  simp [lassoTable, scanStates]

end Tetris
