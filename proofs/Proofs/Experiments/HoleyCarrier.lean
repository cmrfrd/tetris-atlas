import Mathlib
import Proofs.Board
import Proofs.Placement
import Proofs.SafeSet
import Proofs.Experiments.WqoCarrier

/-!
# Hole-aware WQO carrier — where the holes-monotonicity proof has to bite

`WqoCarrier.lean` carries the finite-basis route on the **height-only** domination
order `domLE b β := ∀ j, colHeight b j ≤ colHeight β j`. That order is a genuine WQO
(Dickson) with a small basis, but it is **unsound when holes are forced**: two boards
with identical column heights but different buried empties are *not* equally survivable,
and even cellwise `⊆` fails (removing a cell from under a stack manufactures a hole).

This file enriches the order to `safeLE`, which conjoins height-domination with
**hole-containment** (`holes b ⊆ holes β`), and re-runs the exact `WqoCarrier`
development against it. The height halves are reused verbatim from `WqoCarrier`
(`place_domLE_mono`, `clearLines_domLE`); the **two hole halves are isolated as the
only `sorry`s**, so this file is a precise map of where the hole reasoning bites:

* **Obligation 1 — `place_holes_mono`** (the hole analogue of `place_domLE_mono`):
  the no-clear hard drop preserves hole-containment. *Plausibly true* — it is the
  honest hole keystone.

* **Obligation 2 — `clearLines_holes_le`** (the hole analogue of `clearLines_domLE`):
  line clears do not introduce holes beyond the pre-clear board. **FALSE as stated.**
  Clearing a partially-supporting full row *relocates / creates* buried empties:
  per a single column, `b_j = {0,2,5}` with row `2` board-wide-full becomes `{0,4}`
  after the clear, so the hole set moves `{1,3,4} ↦ {1,2,3}` and `(j,2)` is a *new*
  hole — not `⊆` the old set. This is the real crux: holes cannot ride the
  "no-clear placement is the worst successor" trick that the height route uses,
  because clears help heights (`clearLines b ≼ b`) but can *hurt* holes. The sound
  repair is the **transient-hole budget** (`HoleyBoard` below): a surviving loop only
  ever holds `≤ K` live holes, all cleared by the bag boundary, so the clear step is
  reasoned via a `card ≤ K` invariant at bag boundaries rather than a pointwise `⊆`.

The payoff: `tetrisSolvableValid_of_holey_wqo` has the *same shape* as
`tetrisSolvableValid_of_wqo`, so a hole-aware basis slots into the proven reduction
the instant the two obligations are discharged.
-/

namespace Tetris.HoleyCarrier

open Tetris Tetris.WqoCarrier

/-! ## The hole-aware state and order -/

/-- **Buried empties** ("holes"): in-field cells that are *not* filled yet sit strictly
below their column's stack height — i.e. covered by something above. This is the
feature the height-only order throws away. -/
def holes (cfg : GameConfig) (b : Board) : Finset Coord :=
  (Finset.range cfg.cols ×ˢ Finset.range cfg.rows).filter
    (fun p => p ∉ b ∧ p.2 < b.colHeight p.1)

/-- **Hole-aware domination.** `b` is "no harder to survive than" `β` when it is no
taller in every column *and* has no holes `β` lacks. Both conjuncts point the same way
(`β` is the worst case: tallest *and* holiest), so a basis-closure check at `β` still
covers everything below it. The first conjunct is exactly `WqoCarrier.domLE`. -/
def safeLE (cfg : GameConfig) (b β : Board) : Prop :=
  domLE b β ∧ holes cfg b ⊆ holes cfg β

/-- The height half of `safeLE` is `domLE` (definitionally the first projection). -/
theorem safeLE_domLE {cfg : GameConfig} {b β : Board} (h : safeLE cfg b β) : domLE b β :=
  h.1

@[refl] theorem safeLE_refl (cfg : GameConfig) (b : Board) : safeLE cfg b b :=
  ⟨fun _ => le_refl _, Finset.Subset.refl _⟩

/-- `safeLE` is transitive — the chaining workhorse for the closure step. -/
theorem safeLE_trans {cfg : GameConfig} {a b c : Board}
    (h1 : safeLE cfg a b) (h2 : safeLE cfg b c) : safeLE cfg a c :=
  ⟨domLE_trans h1.1 h2.1, fun _ hx => h2.2 (h1.2 hx)⟩

/-! ## The two hole obligations (the only `sorry`s in this file) -/

/-- **OBLIGATION 1 — holes-monotonicity of the no-clear drop.** The hole analogue of
`WqoCarrier.place_domLE_mono`. On the emptier/shorter board the same piece falls at
least as far (`dropOffset_mono`), so intuitively it cannot manufacture a hole that the
taller board avoids — but proving it requires controlling exactly which cells the drop
buries on each board, which is the genuine work. *Plausibly true; left open.* -/
theorem place_holes_mono {cfg : GameConfig} {b β : Board} (pl : Placement)
    (h : safeLE cfg b β) :
    holes cfg (pl.place b) ⊆ holes cfg (pl.place β) := by
  sorry

/-- **OBLIGATION 2 — line clears do not add holes.** The hole analogue of
`WqoCarrier.clearLines_domLE`. **FALSE as stated** (see the file header counterexample:
clearing a partially-supporting full row creates a new buried empty). It is recorded
here in the exact form the height route's chain *wants*, to pinpoint why holes break
that route: the chain link `applyStep g ≼ place g` holds for heights
(`clearLines b ≼ b`) but not for holes. The sound replacement is a transient-hole
*budget* invariant at bag boundaries (`HoleyBoard.budget`), not this pointwise `⊆`. -/
theorem clearLines_holes_le (cfg : GameConfig) (b : Board) :
    holes cfg (Board.clearLines cfg b) ⊆ holes cfg b := by
  sorry

/-! ## Assembled monotonicity — height halves PROVEN, hole halves from the obligations -/

/-- The no-clear drop preserves `safeLE`: height half is the proven `place_domLE_mono`,
hole half is Obligation 1. -/
theorem place_safeLE_mono {cfg : GameConfig} {b β : Board} (pl : Placement)
    (h : safeLE cfg b β) : safeLE cfg (pl.place b) (pl.place β) :=
  ⟨place_domLE_mono pl h.1, place_holes_mono pl h⟩

/-- A line clear lands `safeLE`-below the pre-clear board: height half is the proven
`clearLines_domLE`, hole half is Obligation 2. -/
theorem clearLines_safeLE (cfg : GameConfig) (b : Board) :
    safeLE cfg (Board.clearLines cfg b) b :=
  ⟨clearLines_domLE cfg b, clearLines_holes_le cfg b⟩

/-! ## S2 — the hole-aware carrier wired to the proven reduction

Mirrors `WqoCarrier` exactly with `safeLE` in place of `domLE`. The loss obligation
(`hheight`) is discharged from the *height half* of `safeLE` alone (holes do not affect
loss); the hole half only sharpens the closure check, exactly as intended. -/

/-- The dominated-by-finite-basis carrier, hole-aware. -/
def Carrier (basis : Bag → Finset Board) : Set GameState :=
  {g | ∃ β ∈ basis g.bag, safeLE GameConfig.standard g.board β}

/-- **The hole-aware finite-basis reduction.** `hheight`/`hinit` discharged here; the
per-piece closure on the whole carrier (`hstep`) is the remaining obligation. -/
theorem tetrisSolvableValid_of_holey_wqo_basis
    (basis : Bag → Finset Board)
    (hbheight : ∀ (T : Bag) (β : Board), β ∈ basis T →
      ∀ j, β.colHeight j ≤ GameConfig.standard.rows)
    (hinit : ∃ β ∈ basis Bag.full, safeLE GameConfig.standard Board.empty β)
    (hstep : ∀ g ∈ Carrier basis, ∀ p, p ∈ g.bag →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        adversarialStep GameConfig.standard g p pl ∈ Carrier basis) :
    TetrisSolvableValid := by
  refine tetrisSolvableValid_of_height_bounded_invariant (Carrier basis) ?_ ?_ hstep
  · -- hinit : GameState.init ∈ Carrier basis
    obtain ⟨β, hβ, hdom⟩ := hinit
    exact ⟨β, by simpa using hβ, by simpa using hdom⟩
  · -- hheight : carrier states are height-bounded (height half of safeLE + basis bound)
    rintro g ⟨β, hβ, hdom⟩ j
    exact le_trans (hdom.1 j) (hbheight g.bag β hβ j)

/-! ## S3 — lift a FINITE basis-closure check to the whole dominated carrier

Identical chain to `WqoCarrier.hstep_of_basis_closure`, but every `≼` is now `safeLE`:

  `applyStep g pl = clearLines (place g pl) ≼ place g pl ≼ place β pl ≼ β'`

via `clearLines_safeLE`, `place_safeLE_mono`, and the basis check. Validity is
board-independent, so it transfers from `β` to `g`. -/
theorem hstep_of_basis_closure_holey
    (basis : Bag → Finset Board)
    (hclosure : ∀ (T : Bag) (β : Board), β ∈ basis T → ∀ p, p ∈ T →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        ∃ β' ∈ basis (T.draw p),
          safeLE GameConfig.standard (Placement.place β { pl with piece := p }) β') :
    ∀ g ∈ Carrier basis, ∀ p, p ∈ g.bag →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        adversarialStep GameConfig.standard g p pl ∈ Carrier basis := by
  rintro g ⟨β, hβ, hgβ⟩ p hp
  obtain ⟨pl, hpiece, hvalid, β', hβ', hpβ'⟩ := hclosure g.bag β hβ p hp
  refine ⟨pl, hpiece, hvalid, β', ?_, ?_⟩
  · simpa [adversarialStep] using hβ'
  · show safeLE GameConfig.standard (adversarialStep GameConfig.standard g p pl).board β'
    have step1 :
        safeLE GameConfig.standard
          (Placement.applyStep GameConfig.standard g.board { pl with piece := p })
          (Placement.place g.board { pl with piece := p }) := by
      rw [Placement.applyStep]; exact clearLines_safeLE _ _
    have step2 :
        safeLE GameConfig.standard
          (Placement.place g.board { pl with piece := p })
          (Placement.place β { pl with piece := p }) :=
      place_safeLE_mono _ hgβ
    have hchain :
        safeLE GameConfig.standard
          (Placement.applyStep GameConfig.standard g.board { pl with piece := p }) β' :=
      safeLE_trans (safeLE_trans step1 step2) hpβ'
    simpa [adversarialStep] using hchain

/-- **S2+S3 composed: `TetrisSolvableValid` from a hole-aware FINITE basis-closure
check.** Same shape as `WqoCarrier.tetrisSolvableValid_of_wqo`: a finite, height-bounded,
init-dominating `basis` whose every element, for every drawable piece, has a valid
placement whose *no-clear* drop is `safeLE`-dominated by another basis element. The only
gaps between here and a sorry-free proof are the two hole obligations above. -/
theorem tetrisSolvableValid_of_holey_wqo
    (basis : Bag → Finset Board)
    (hbheight : ∀ (T : Bag) (β : Board), β ∈ basis T →
      ∀ j, β.colHeight j ≤ GameConfig.standard.rows)
    (hinit : ∃ β ∈ basis Bag.full, safeLE GameConfig.standard Board.empty β)
    (hclosure : ∀ (T : Bag) (β : Board), β ∈ basis T → ∀ p, p ∈ T →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        ∃ β' ∈ basis (T.draw p),
          safeLE GameConfig.standard (Placement.place β { pl with piece := p }) β') :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_holey_wqo_basis basis hbheight hinit
    (hstep_of_basis_closure_holey basis hclosure)

/-! ## The sound carrier: skyline + bounded transient holes

`safeLE` keeps holes faithfully but, as Obligation 2 shows, the `clearLines` link is
false pointwise. The fix factors the board into the part that is a clean WQO (the
skyline) and the part that breaks clean WQOs (the holes), then *bounds the holes by a
budget* `K`. With `holes.card ≤ K` the hole component takes finitely many shapes, so the
carrier stays a small-basis WQO while representing holes exactly. `K` is small in
practice: a surviving loop's steady state is hole-free, with only transient holes
(e.g. the empty-board bootstrap's single S-first hole, cleared within bag 1). -/

/-- A board factored as a skyline plus an explicit, budgeted set of buried empties.
Columns/rows are plain `ℕ` (matching `Board`), with the configured dimensions entering
only through predicates — keeping the factorisation `Fin`-bookkeeping free. -/
structure HoleyBoard (cfg : GameConfig) (K : ℕ) where
  /-- The surface profile (the clean-WQO part), indexed by column. -/
  height : ℕ → ℕ
  /-- The explicit holes (the part that breaks clean WQOs). -/
  buried : Finset Coord
  /-- Every recorded hole really is buried under its column's surface. -/
  covered : ∀ p ∈ buried, p.2 < height p.1
  /-- The transient-hole budget: at most `K` live holes. -/
  budget : buried.card ≤ K

/-- The hole-aware domination on factored boards: shorter skyline and contained holes —
the `HoleyBoard` mirror of `safeLE`. -/
def HoleyBoard.le {cfg : GameConfig} {K : ℕ} (g β : HoleyBoard cfg K) : Prop :=
  (∀ j, g.height j ≤ β.height j) ∧ g.buried ⊆ β.buried

#print axioms tetrisSolvableValid_of_holey_wqo

end Tetris.HoleyCarrier
