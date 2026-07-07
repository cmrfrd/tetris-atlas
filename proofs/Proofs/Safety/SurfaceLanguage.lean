import Mathlib
import Proofs.Model.Placement
import Proofs.Model.Game
import Proofs.Combinatorics.BoardCount
import Proofs.Invariants.ColumnCount
import Proofs.Invariants.Holes
import Proofs.Invariants.SurfaceFiber
import Proofs.Safety.SafeSet

/-!
# The surface language — is the safe set's projection onto skylines regular?

The regular-model-checking route asks whether the safe set, viewed through the
projection `board ↦ (colHeight 0, …, colHeight (cols−1))` — its **surface
word** — admits a small finite-automaton presentation read column by column.
This file makes the question formal and settles its settleable parts.

**The question is quantitative, not qualitative.** At a fixed width every
language of length-`cols` words over the finite height alphabet is recognized
by *some* finite scan automaton (the prefix tree), so "regular" per se is
vacuous; the meaningful measure is the **index** — the minimal state count —
for which `ScanAut.card_le_card_states` (the fooling-set bound) gives lower
bounds: any family of pairwise-distinguishable prefixes injects into the state
space of every recognizer.

**Verdict, negative half (pure height alphabet).**

1. `surfaceWord_applyStep_not_congruent` — the step operator does **not**
   descend through the projection: two well-formed boards with *equal* surface
   words, no pre-existing full rows, and the *same valid placement* produce
   *different* post-step surface words. Placement alone descends
   (`surfaceWord_place_eq`, via `SurfaceFiber`); **clears** read the buried
   holes, and the quotient dynamics is ill-defined exactly there. So no scan
   automaton over pure heights can have its closure obligation even *stated*,
   let alone verified, by surface-level reasoning.
2. The sub-universe where the projection is lossless — hole-free boards, where
   `eq_of_surfaceWord_eq_of_holes_eq_zero` shows the surface word *is* the
   whole state — cannot host any closed invariant containing the initial
   state: `no_holefree_closed_invariant`. The killer is the very first S piece
   (`holes_place_empty_ne_zero_of_S`: **every** valid S placement on the empty
   board buries a hole), so play is ejected from the lossless sub-universe in
   bag one. (The Experiments companion
   `FiveBagReset.no_holes_zero_closed_table_contains_init` is the atlas-table
   form of the same obstruction.)

**Verdict, positive half (what survives).** `scan_certificate_sound` — stated
for an *arbitrary* encoding `GameState → List A` — proves that any scan
automaton whose accepted set contains `init` and is closed under `safeOp` is a
sound solvability certificate. This types the route's real target: the
alphabet must be **richer than heights** (per-column hole data at minimum,
per the two obstructions above), and the open content is exhibiting a
small-index recognizer over such an enriched alphabet whose closure obligation
discharges. The projection language itself is packaged as `safeSurfaceLang`.
-/

namespace Tetris
namespace SurfaceLanguage

/-! ## The projection: boards to surface words -/

/-- The **surface word** of a board: its column heights read left to right
across the configured playfield — the projection whose language this file
studies. -/
def surfaceWord (cfg : GameConfig) (b : Board) : List ℕ :=
  (List.range cfg.cols).map b.colHeight

@[simp] theorem surfaceWord_length (cfg : GameConfig) (b : Board) :
    (surfaceWord cfg b).length = cfg.cols := by
  simp [surfaceWord]

/-- Two boards share a surface word iff their column heights agree across the
playfield. -/
theorem surfaceWord_eq_iff {cfg : GameConfig} {b β : Board} :
    surfaceWord cfg b = surfaceWord cfg β ↔
      ∀ j < cfg.cols, b.colHeight j = β.colHeight j := by
  unfold surfaceWord
  rw [List.map_inj_left]
  simp only [List.mem_range]

/-- The empty board's surface word is the all-zero word. -/
theorem surfaceWord_empty (cfg : GameConfig) :
    surfaceWord cfg (∅ : Board) = List.replicate cfg.cols 0 := by
  have h : surfaceWord cfg (∅ : Board)
      = List.replicate (surfaceWord cfg (∅ : Board)).length 0 :=
    List.eq_replicate_of_mem (by
      intro x hx
      rw [surfaceWord, List.mem_map] at hx
      obtain ⟨j, -, rfl⟩ := hx
      exact Board.colHeight_empty j)
  rwa [surfaceWord_length] at h

/-- Membership in a column's row set is cell membership. -/
theorem mem_colRows_iff {b : Board} {j r : ℕ} :
    r ∈ b.colRows j ↔ (j, r) ∈ b := by
  unfold Board.colRows
  rw [Finset.mem_image]
  constructor
  · rintro ⟨p, hp, rfl⟩
    obtain ⟨hpb, hpj⟩ := Finset.mem_filter.mp hp
    rw [← hpj]
    exact hpb
  · intro h
    exact ⟨(j, r), Finset.mem_filter.mpr ⟨h, rfl⟩, rfl⟩

instance (cfg : GameConfig) (b : Board) : Decidable (Board.WF cfg b) := by
  unfold Board.WF; infer_instance

/-- Well-formed boards are empty beyond the playfield: columns at or past
`cfg.cols` have height 0. -/
theorem colHeight_eq_zero_of_wf {cfg : GameConfig} {b : Board}
    (hwf : Board.WF cfg b) {j : ℕ} (hj : cfg.cols ≤ j) :
    b.colHeight j = 0 := by
  unfold Board.colHeight
  have hempty : b.colRows j = ∅ := by
    rw [Finset.eq_empty_iff_forall_notMem]
    intro r hr
    have := hwf _ (mem_colRows_iff.mp hr)
    omega
  rw [hempty, Finset.sup_empty]
  rfl

/-! ## Placement descends to the surface quotient (the positive leg) -/

/-- **Bare placement descends through the projection**: equal surface words
give equal post-placement surface words. The word-level corollary of
`SurfaceFiber.colHeight_place_eq_of_colHeight_eq` — the hard drop is
hole-blind. -/
theorem surfaceWord_place_eq {cfg : GameConfig} {b β : Board} (pl : Placement)
    (hwb : Board.WF cfg b) (hwβ : Board.WF cfg β)
    (h : surfaceWord cfg b = surfaceWord cfg β) :
    surfaceWord cfg (pl.place b) = surfaceWord cfg (pl.place β) := by
  have hall : ∀ j, b.colHeight j = β.colHeight j := by
    intro j
    by_cases hj : j < cfg.cols
    · exact surfaceWord_eq_iff.mp h j hj
    · rw [colHeight_eq_zero_of_wf hwb (le_of_not_gt hj),
        colHeight_eq_zero_of_wf hwβ (le_of_not_gt hj)]
  exact surfaceWord_eq_iff.mpr
    (fun j _ => SurfaceFiber.colHeight_place_eq_of_colHeight_eq pl hall j)

/-! ## The no-go: the full step (with clears) does NOT descend

The witness pair: row 0 filled across columns 0–8 with a second cell atop
column 0 — once gravity-packed (`flushBoard`), once with the covered cell
missing (`holedBoard`, a buried hole at `(0,0)`). Same skyline. A vertical I
in column 9 completes row 0 on `flushBoard` (the row clears, the skyline
collapses) but not on `holedBoard` (the hole blocks the clear and the skyline
grows). -/

/-- Gravity-packed witness: `[2,1,1,1,1,1,1,1,1,0]` skyline, no holes. -/
def flushBoard : Board :=
  {(0, 0), (0, 1), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0)}

/-- Holed witness: the same `[2,1,1,1,1,1,1,1,1,0]` skyline, but column 0
carries only its row-1 cell — `(0,0)` is a buried hole. -/
def holedBoard : Board :=
  {(0, 1), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0)}

/-- **The step operator does not descend to the surface quotient.** There are
well-formed boards with equal surface words and no pre-existing full rows on
which the *same valid placement* yields different post-step surface words —
line clears read buried holes, which the projection forgets. This is the
formal reason regular model checking cannot run over the pure height
alphabet: the transported dynamics is not well defined on words. -/
theorem surfaceWord_applyStep_not_congruent :
    ¬ ∀ (b β : Board) (pl : Placement),
        Board.WF GameConfig.standard b → Board.WF GameConfig.standard β →
        pl.Valid GameConfig.standard →
        Board.fullRows GameConfig.standard b = ∅ →
        Board.fullRows GameConfig.standard β = ∅ →
        surfaceWord GameConfig.standard b = surfaceWord GameConfig.standard β →
        surfaceWord GameConfig.standard (pl.applyStep GameConfig.standard b) =
          surfaceWord GameConfig.standard (pl.applyStep GameConfig.standard β) := by
  intro H
  have h := H flushBoard holedBoard ⟨Piece.I, 1, 9⟩
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  exact absurd h (by decide)

/-- The projection is not injective, even on well-formed boards: the witness
pair shares a surface word yet differs (by exactly the buried hole). -/
theorem surfaceWord_not_injective :
    ¬ ∀ (b β : Board), Board.WF GameConfig.standard b →
        Board.WF GameConfig.standard β →
        surfaceWord GameConfig.standard b = surfaceWord GameConfig.standard β →
        b = β := by
  intro H
  have h := H flushBoard holedBoard (by decide) (by decide) (by decide)
  exact absurd h (by decide)

/-! ## The lossless sub-universe: hole-free boards ARE their surface words -/

/-- A hole-free column is exactly the initial segment below its height. -/
theorem colRows_eq_range_of_colHoles_eq_zero {b : Board} {j : ℕ}
    (h : Board.colHoles b j = 0) :
    b.colRows j = Finset.range (b.colHeight j) := by
  apply Finset.eq_of_subset_of_card_le
  · intro r hr
    rw [Finset.mem_range]
    exact Board.lt_colHeight (mem_colRows_iff.mp hr)
  · rw [Finset.card_range, (Board.colHoles_eq_zero_iff b j).mp h]

/-- **On the hole-free universe the projection is lossless**: two well-formed
hole-free boards with the same surface word are equal. Surface-level regular
model checking would be *complete* there — every column is gravity-packed, so
the skyline is the entire state. -/
theorem eq_of_surfaceWord_eq_of_holes_eq_zero {cfg : GameConfig} {b β : Board}
    (hwb : Board.WF cfg b) (hwβ : Board.WF cfg β)
    (hb : Board.holes cfg b = 0) (hβ : Board.holes cfg β = 0)
    (h : surfaceWord cfg b = surfaceWord cfg β) : b = β := by
  have hh := surfaceWord_eq_iff.mp h
  have hzb := (Board.holes_eq_zero_iff cfg b).mp hb
  have hzβ := (Board.holes_eq_zero_iff cfg β).mp hβ
  ext p
  obtain ⟨c, r⟩ := p
  by_cases hc : c < cfg.cols
  · rw [← mem_colRows_iff, ← mem_colRows_iff,
      colRows_eq_range_of_colHoles_eq_zero (hzb c (Finset.mem_range.mpr hc)),
      colRows_eq_range_of_colHoles_eq_zero (hzβ c (Finset.mem_range.mpr hc)),
      hh c hc]
  · constructor
    · intro hmem; exact absurd (hwb _ hmem) hc
    · intro hmem; exact absurd (hwβ _ hmem) hc

/-! ## …but play is ejected from it at the very first S piece -/

/-- Kernel-checked base: every valid S placement on the empty board leaves a
buried hole, across all rotations and columns. -/
theorem holes_place_empty_ne_zero_of_S_fin :
    ∀ (r : Rotation) (c : Fin 10),
      (⟨Piece.S, r, (c : ℕ)⟩ : Placement).Valid GameConfig.standard →
      Board.holes GameConfig.standard
        ((⟨Piece.S, r, (c : ℕ)⟩ : Placement).place ∅) ≠ 0 := by
  decide

/-- **Every valid S placement on the empty board buries a hole** — the S
overhang cannot be supported by a flat floor. The general-placement form of
the kernel-checked base. -/
theorem holes_place_empty_ne_zero_of_S {pl : Placement}
    (hp : pl.piece = Piece.S) (hv : pl.Valid GameConfig.standard) :
    Board.holes GameConfig.standard (pl.place ∅) ≠ 0 := by
  obtain ⟨piece, rot, col⟩ := pl
  have hp' : piece = Piece.S := hp
  subst hp'
  have hcol : col < 10 := by
    obtain ⟨cell, hcell⟩ :=
      Placement.shapeUp_nonempty (⟨Piece.S, rot, col⟩ : Placement)
    have hb : col + cell.1 < GameConfig.standard.cols := hv cell hcell
    rw [GameConfig.standard_cols] at hb
    omega
  exact holes_place_empty_ne_zero_of_S_fin rot ⟨col, hcol⟩ hv

/-- **No hole-free closed invariant exists.** Any set of game states that
contains `init`, is closed under the safe-step operator, and consists solely
of hole-free boards is contradictory: closure hands the adversary's first S
piece a valid placement whose successor stays in the set, but every valid S
placement on the empty board buries a hole (and with only 4 cells on the
board no row can complete, so the clear phase cannot rescue it). Surface
words are a complete description exactly on the hole-free universe
(`eq_of_surfaceWord_eq_of_holes_eq_zero`), so this closes the door on
pure-surface regular model checking from the *other* side: the sub-universe
where the projection is faithful supports no invariant at all. -/
theorem no_holefree_closed_invariant (S : Set GameState)
    (hinit : GameState.init ∈ S)
    (hclosed : S ⊆ safeOp GameConfig.standard S)
    (hfree : ∀ g ∈ S, Board.holes GameConfig.standard g.board = 0) : False := by
  obtain ⟨-, hstep⟩ := hclosed hinit
  obtain ⟨pl, hpiece, hvalid, hmem⟩ := hstep Piece.S (Bag.mem_full Piece.S)
  obtain ⟨piece, rot, col⟩ := pl
  have hp' : piece = Piece.S := hpiece
  subst hp'
  have hboard :
      (adversarialStep GameConfig.standard GameState.init Piece.S
        ⟨Piece.S, rot, col⟩).board
      = (⟨Piece.S, rot, col⟩ : Placement).applyStep GameConfig.standard
          (∅ : Board) := rfl
  have hcard : ((⟨Piece.S, rot, col⟩ : Placement).place ∅).card = 4 := by
    have hc := Placement.count_place (∅ : Board) ⟨Piece.S, rot, col⟩
    simpa [Board.count] using hc
  have hnofull : Board.fullRows GameConfig.standard
      ((⟨Piece.S, rot, col⟩ : Placement).place ∅) = ∅ := by
    apply Board.fullRows_eq_empty_of_forall_rowCount_lt
    intro r
    have h1 : Board.rowCount ((⟨Piece.S, rot, col⟩ : Placement).place ∅) r
        ≤ ((⟨Piece.S, rot, col⟩ : Placement).place ∅).card := by
      unfold Board.rowCount
      exact Finset.card_filter_le _ _
    rw [GameConfig.standard_cols]
    omega
  have happ : (⟨Piece.S, rot, col⟩ : Placement).applyStep GameConfig.standard
      (∅ : Board) = (⟨Piece.S, rot, col⟩ : Placement).place ∅ := by
    rw [Placement.applyStep_eq_clearLines_place,
      Board.clearLines_eq_self_of_no_fullRows _ hnofull]
  have hz := hfree _ hmem
  rw [hboard, happ] at hz
  exact holes_place_empty_ne_zero_of_S rfl hvalid hz

/-! ## Scan automata and the index lower-bound machinery

At fixed width, *some* recognizer always exists (the prefix tree on the
finite height alphabet), so the substantive question about any language here
is its **index**. The fooling-set bound below is the standard tool: exhibit
`m` pairwise-distinguishable prefixes and every recognizer needs `m` states. -/

/-- A deterministic scan automaton: reads a word left to right. -/
structure ScanAut (A : Type*) (Q : Type*) where
  /-- The start state. -/
  start : Q
  /-- The transition function. -/
  step : Q → A → Q
  /-- The acceptance predicate on final states. -/
  acc : Q → Bool

namespace ScanAut

variable {A : Type*} {Q : Type*}

/-- The state reached after reading `w`. -/
def run (M : ScanAut A Q) (w : List A) : Q := w.foldl M.step M.start

/-- Acceptance: the reached state satisfies `acc`. -/
def Accepts (M : ScanAut A Q) (w : List A) : Prop := M.acc (M.run w) = true

/-- `M` recognizes the language `L`. -/
def Recognizes (M : ScanAut A Q) (L : List A → Prop) : Prop :=
  ∀ w, L w ↔ M.Accepts w

theorem run_append (M : ScanAut A Q) (u w : List A) :
    M.run (u ++ w) = w.foldl M.step (M.run u) := by
  unfold run
  rw [List.foldl_append]

/-- Prefixes driven to the same state accept exactly the same suffixes — the
Myhill–Nerode collapse. -/
theorem accepts_append_iff_of_run_eq (M : ScanAut A Q) {u v : List A}
    (h : M.run u = M.run v) (w : List A) :
    M.Accepts (u ++ w) ↔ M.Accepts (v ++ w) := by
  unfold Accepts
  rw [run_append, run_append, h]

/-- Two prefixes are distinguished by `L` when some common suffix separates
their membership. -/
def Distinguishes (L : List A → Prop) (u v : List A) : Prop :=
  ∃ w, ¬ (L (u ++ w) ↔ L (v ++ w))

/-- **The fooling-set lower bound.** A family of pairwise-distinguishable
prefixes injects into the state space of every recognizer: `m`
distinguishable prefixes force at least `m` states. This is the yardstick for
"is the language *plausibly* (small-index) regular" — the only non-vacuous
reading of the question at fixed width. -/
theorem card_le_card_states [Fintype Q]
    (M : ScanAut A Q) (L : List A → Prop) (hrec : M.Recognizes L)
    (F : Finset (List A))
    (hdist : ∀ u ∈ F, ∀ v ∈ F, u ≠ v → Distinguishes L u v) :
    F.card ≤ Fintype.card Q := by
  classical
  have hinj : Set.InjOn M.run ↑F := by
    intro u hu v hv heq
    by_contra hne
    obtain ⟨w, hw⟩ := hdist u hu v hv hne
    exact hw (by rw [hrec, hrec]; exact accepts_append_iff_of_run_eq M heq w)
  calc F.card = (F.image M.run).card := (Finset.card_image_of_injOn hinj).symm
    _ ≤ Fintype.card Q := by
        rw [← Finset.card_univ]
        exact Finset.card_le_card (Finset.subset_univ _)

end ScanAut

/-! ## The certificate target: what the route must inhabit -/

/-- **Soundness of a scan-recognized invariant.** Any scan automaton over any
state encoding whose accepted set contains the initial state and is closed
under the safe-step operator certifies solvability. Stated for an arbitrary
`encode` deliberately: the two obstructions above force the alphabet beyond
pure heights, and this theorem is exactly what a successful enriched-alphabet
recognizer would plug into. The open content of the regular route is
inhabiting these hypotheses with a *small* `Q`. -/
theorem scan_certificate_sound {cfg : GameConfig} {A Q : Type*}
    (M : ScanAut A Q) (encode : GameState → List A)
    (hclosed : {g : GameState | M.Accepts (encode g)} ⊆
      safeOp cfg {g : GameState | M.Accepts (encode g)})
    (hinit : M.Accepts (encode GameState.init)) :
    TetrisSolvableFor cfg :=
  safe_extract_for (safe_greatest _ hclosed hinit)

/-- Standard-config headline form of `scan_certificate_sound`. -/
theorem scan_certificate_sound_standard {A Q : Type*}
    (M : ScanAut A Q) (encode : GameState → List A)
    (hclosed : {g : GameState | M.Accepts (encode g)} ⊆
      safeOp GameConfig.standard {g : GameState | M.Accepts (encode g)})
    (hinit : M.Accepts (encode GameState.init)) :
    TetrisSolvable :=
  safe_extract (safe_greatest _ hclosed hinit)

/-! ## The projection language itself -/

/-- The surface language of the safe set — the object of the question: the
set of surface words attained by safe states. -/
def safeSurfaceLang (cfg : GameConfig) : Set (List ℕ) :=
  (fun g : GameState => surfaceWord cfg g.board) '' safe cfg

/-- The projection is empty exactly when the safe set is. -/
theorem safeSurfaceLang_eq_empty_iff (cfg : GameConfig) :
    safeSurfaceLang cfg = ∅ ↔ safe cfg = ∅ :=
  Set.image_eq_empty

/-- If the open conjecture holds, the all-zero word belongs to the safe
surface language (the empty board is a safe surface). -/
theorem replicate_zero_mem_safeSurfaceLang (cfg : GameConfig)
    (h : GameState.init ∈ safe cfg) :
    List.replicate cfg.cols 0 ∈ safeSurfaceLang cfg := by
  refine ⟨GameState.init, h, ?_⟩
  change surfaceWord cfg GameState.init.board = List.replicate cfg.cols 0
  have hb : GameState.init.board = (∅ : Board) := rfl
  rw [hb, surfaceWord_empty]

end SurfaceLanguage
end Tetris
