import Mathlib
import Proofs.Model.Board

/-!
# In-field boards form a finite type

A board confined to the playing field — well-formed (columns < `cols`) with
every cell strictly below row `rows` — is a finite object. There are finitely
many such boards, given by the powerset of the `cols × rows` grid.

This is the foundational fact the Atlas mission rests on: the relevant
state space of Tetris is genuinely finite, so brute-force enumeration is
well-defined in principle.
-/

namespace Tetris

/-- A board confined to the visible playing field: well-formed and with every
cell strictly below row `cfg.rows`. -/
def InFieldBoard (cfg : GameConfig) : Type :=
  {b : Board // Board.WF cfg b ∧ ∀ p ∈ b, p.2 < cfg.rows}

namespace InFieldBoard

variable {cfg : GameConfig}

/-- Equality of in-field boards is decidable (from `DecidableEq` on the
underlying `Finset Coord` carrier). -/
instance : DecidableEq (InFieldBoard cfg) := by
  unfold InFieldBoard; infer_instance

/-- Embed an in-field board into the finite type `Finset (Fin cols × Fin rows)`. -/
def toFin (b : InFieldBoard cfg) : Finset (Fin cfg.cols × Fin cfg.rows) :=
  b.val.attach.image (fun p =>
    (⟨p.val.1, b.prop.1 p.val p.prop⟩, ⟨p.val.2, b.prop.2 p.val p.prop⟩))

/-- Recover an in-field board from a finite set of in-bounds coordinates: project
each `Fin` pair to its underlying `ℕ` pair. The well-formed + in-field obligations
discharge from `Fin.is_lt` on each component. -/
def ofFin (s : Finset (Fin cfg.cols × Fin cfg.rows)) : InFieldBoard cfg :=
  ⟨s.image (fun q : Fin cfg.cols × Fin cfg.rows => (q.1.val, q.2.val)),
   by
     refine ⟨?_, ?_⟩
     · intro p hp
       rw [Finset.mem_image] at hp
       obtain ⟨q, _, rfl⟩ := hp
       exact q.1.is_lt
     · intro p hp
       rw [Finset.mem_image] at hp
       obtain ⟨q, _, rfl⟩ := hp
       exact q.2.is_lt⟩

/-- `ofFin` is a left-inverse of `toFin` on `InFieldBoard cfg`. -/
theorem ofFin_toFin (b : InFieldBoard cfg) : ofFin (toFin b) = b := by
  apply Subtype.ext
  change (toFin b).image (fun q : Fin cfg.cols × Fin cfg.rows => (q.1.val, q.2.val))
    = b.val
  unfold toFin
  rw [Finset.image_image]
  -- The composed function reduces (Fin/Prod eta + beta) to `Subtype.val`.
  exact Finset.attach_image_val

/-- `ofFin` is a right-inverse of `toFin` on `Finset (Fin cols × Fin rows)`. -/
theorem toFin_ofFin (s : Finset (Fin cfg.cols × Fin cfg.rows)) :
    toFin (ofFin s) = s := by
  ext q
  unfold toFin
  rw [Finset.mem_image]
  constructor
  · rintro ⟨⟨p, hp⟩, _, hpe⟩
    have hp' : p ∈ s.image
        (fun (x : Fin cfg.cols × Fin cfg.rows) => (x.1.val, x.2.val)) := hp
    rw [Finset.mem_image] at hp'
    obtain ⟨q', hq's, rfl⟩ := hp'
    -- `hpe : (⟨q'.1.val, _⟩, ⟨q'.2.val, _⟩) = q`
    -- By Fin/Prod structure eta this LHS is defeq `q'`, so `hpe : q' = q`.
    have hqq : q' = q := hpe
    rw [← hqq]; exact hq's
  · intro hq
    refine ⟨⟨(q.1.val, q.2.val), ?_⟩, Finset.mem_attach _ _, ?_⟩
    · change (q.1.val, q.2.val) ∈
        s.image (fun (x : Fin cfg.cols × Fin cfg.rows) => (x.1.val, x.2.val))
      rw [Finset.mem_image]
      exact ⟨q, hq, rfl⟩
    · -- the re-wrapped pair `(⟨q.1.val, _⟩, ⟨q.2.val, _⟩)` is defeq `q`
      rfl

theorem toFin_injective : Function.Injective (toFin : InFieldBoard cfg → _) := by
  intro a b hab
  apply Subtype.ext
  ext q
  constructor
  · intro hq
    have h1a : q.1 < cfg.cols := a.prop.1 q hq
    have h2a : q.2 < cfg.rows := a.prop.2 q hq
    have hqf : (⟨q.1, h1a⟩, ⟨q.2, h2a⟩) ∈ toFin a := by
      unfold toFin
      rw [Finset.mem_image]
      exact ⟨⟨q, hq⟩, Finset.mem_attach _ _, rfl⟩
    rw [hab] at hqf
    unfold toFin at hqf
    rw [Finset.mem_image] at hqf
    obtain ⟨⟨q', hq'b⟩, _, hqe⟩ := hqf
    rw [Prod.mk.injEq] at hqe
    obtain ⟨he1, he2⟩ := hqe
    have h1eq : q'.1 = q.1 := Fin.mk_eq_mk.mp he1
    have h2eq : q'.2 = q.2 := Fin.mk_eq_mk.mp he2
    have hqq : q' = q := Prod.ext h1eq h2eq
    rwa [← hqq]
  · intro hq
    have h1b : q.1 < cfg.cols := b.prop.1 q hq
    have h2b : q.2 < cfg.rows := b.prop.2 q hq
    have hqf : (⟨q.1, h1b⟩, ⟨q.2, h2b⟩) ∈ toFin b := by
      unfold toFin
      rw [Finset.mem_image]
      exact ⟨⟨q, hq⟩, Finset.mem_attach _ _, rfl⟩
    rw [← hab] at hqf
    unfold toFin at hqf
    rw [Finset.mem_image] at hqf
    obtain ⟨⟨q', hq'a⟩, _, hqe⟩ := hqf
    rw [Prod.mk.injEq] at hqe
    obtain ⟨he1, he2⟩ := hqe
    have h1eq : q'.1 = q.1 := Fin.mk_eq_mk.mp he1
    have h2eq : q'.2 = q.2 := Fin.mk_eq_mk.mp he2
    have hqq : q' = q := Prod.ext h1eq h2eq
    rwa [← hqq]

/-- An explicit `Equiv` between `InFieldBoard cfg` and the powerset of the
finite `Fin cols × Fin rows` grid. -/
def equivFin : InFieldBoard cfg ≃ Finset (Fin cfg.cols × Fin cfg.rows) where
  toFun := toFin
  invFun := ofFin
  left_inv := ofFin_toFin
  right_inv := toFin_ofFin

/-- **The state space of in-field boards is finite** (computable Fintype via
`equivFin`). This is the foundational fact behind the Atlas: there are only
finitely many states a legal Tetris position can occupy, and the universe is
algorithmically enumerable. -/
instance : Fintype (InFieldBoard cfg) := Fintype.ofEquiv _ equivFin.symm

/-- `|InFieldBoard cfg| = 2^(cols * rows)` — the cardinality of the
powerset of a `cols × rows` grid. -/
theorem fintype_card : Fintype.card (InFieldBoard cfg) = 2 ^ (cfg.cols * cfg.rows) := by
  rw [Fintype.card_congr equivFin, Fintype.card_finset,
      Fintype.card_prod, Fintype.card_fin, Fintype.card_fin]

/-- Concrete cardinality for the standard config: `|InFieldBoard
GameConfig.standard| = 2^200`. -/
theorem standard_fintype_card :
    Fintype.card (InFieldBoard GameConfig.standard) = 2 ^ 200 := by
  rw [fintype_card, GameConfig.standard_cols, GameConfig.standard_rows]

/-- Concrete cardinality for the tiny config: `|InFieldBoard
GameConfig.tiny| = 2^16 = 65536`. -/
theorem tiny_fintype_card :
    Fintype.card (InFieldBoard GameConfig.tiny) = 65536 := by
  rw [fintype_card, GameConfig.tiny_cols, GameConfig.tiny_rows]; decide

/-- **Cell-count bound** for any in-field board: card ≤ cols * rows. -/
theorem card_le (b : InFieldBoard cfg) : b.val.card ≤ cfg.cols * cfg.rows := by
  have hsub : b.val ⊆ Finset.range cfg.cols ×ˢ Finset.range cfg.rows := by
    intro p hp
    rw [Finset.mem_product, Finset.mem_range, Finset.mem_range]
    exact ⟨b.prop.1 p hp, b.prop.2 p hp⟩
  calc b.val.card
      ≤ (Finset.range cfg.cols ×ˢ Finset.range cfg.rows).card :=
        Finset.card_le_card hsub
    _ = cfg.cols * cfg.rows := by
        rw [Finset.card_product, Finset.card_range, Finset.card_range]

/-- Standard specialization: every standard in-field board has
card `≤ 200`. -/
theorem standard_card_le (b : InFieldBoard GameConfig.standard) :
    b.val.card ≤ 200 := card_le b

/-- Tiny specialization: every tiny in-field board has card `≤ 16`. -/
theorem tiny_card_le (b : InFieldBoard GameConfig.tiny) :
    b.val.card ≤ 16 := card_le b

/-- Strict bound: `b.val.card < cfg.cols * cfg.rows + 1`. -/
theorem card_lt_succ (b : InFieldBoard cfg) :
    b.val.card < cfg.cols * cfg.rows + 1 :=
  Nat.lt_succ_of_le (card_le b)

/-- Standard: `b.val.card < 201`. -/
theorem standard_card_lt (b : InFieldBoard GameConfig.standard) :
    b.val.card < 201 :=
  Nat.lt_succ_of_le (standard_card_le b)

/-- Tiny: `b.val.card < 17`. -/
theorem tiny_card_lt (b : InFieldBoard GameConfig.tiny) :
    b.val.card < 17 :=
  Nat.lt_succ_of_le (tiny_card_le b)

/-- The `InFieldBoard` type's `Fintype.card` is positive (≥ 1, witnessed
by the empty board). -/
theorem fintype_card_pos : 0 < Fintype.card (InFieldBoard cfg) := by
  rw [fintype_card]
  exact Nat.two_pow_pos _

/-- Standard: `Fintype.card (InFieldBoard standard) > 0`. -/
theorem standard_fintype_card_pos : 0 < Fintype.card (InFieldBoard GameConfig.standard) :=
  fintype_card_pos

/-- Tiny: `Fintype.card (InFieldBoard tiny) > 0`. -/
theorem tiny_fintype_card_pos : 0 < Fintype.card (InFieldBoard GameConfig.tiny) :=
  fintype_card_pos

/-- The `InFieldBoard` type's `Fintype.card` is nonzero. -/
theorem fintype_card_ne_zero : Fintype.card (InFieldBoard cfg) ≠ 0 :=
  Nat.ne_of_gt fintype_card_pos

/-- The `InFieldBoard` type's `Fintype.card` is `≥ 1`. -/
theorem one_le_fintype_card : 1 ≤ Fintype.card (InFieldBoard cfg) :=
  fintype_card_pos

/-- The tiny in-field board space is strictly smaller than the
standard. -/
theorem tiny_fintype_card_lt_standard_fintype_card :
    Fintype.card (InFieldBoard GameConfig.tiny) <
      Fintype.card (InFieldBoard GameConfig.standard) := by
  rw [tiny_fintype_card, standard_fintype_card]
  decide

/-- Non-strict form of Tick 1401. -/
theorem tiny_fintype_card_le_standard_fintype_card :
    Fintype.card (InFieldBoard GameConfig.tiny) ≤
      Fintype.card (InFieldBoard GameConfig.standard) :=
  le_of_lt tiny_fintype_card_lt_standard_fintype_card

/-- Distinctness form of Tick 1401. -/
theorem tiny_fintype_card_ne_standard_fintype_card :
    Fintype.card (InFieldBoard GameConfig.tiny) ≠
      Fintype.card (InFieldBoard GameConfig.standard) :=
  Nat.ne_of_lt tiny_fintype_card_lt_standard_fintype_card

/-- The empty board as an `InFieldBoard`. -/
def empty (cfg : GameConfig) : InFieldBoard cfg :=
  ⟨∅, Board.empty_wf cfg, by intro p hp; exact absurd hp (Finset.notMem_empty _)⟩

/-- The empty in-field board has count 0. -/
@[simp] theorem empty_val_card (cfg : GameConfig) : (empty cfg).val.card = 0 := by
  simp [empty]

/-- The `InFieldBoard` type is `Nonempty` (witnessed by `empty`). -/
instance instNonempty : Nonempty (InFieldBoard cfg) := ⟨empty cfg⟩

/-- The underlying carrier of `empty` is `∅`. -/
@[simp] theorem empty_val (cfg : GameConfig) : (empty cfg).val = ∅ := rfl

/-- A cell is never a member of the empty in-field board. -/
@[simp] theorem notMem_empty (cfg : GameConfig) (p : Coord) :
    p ∉ (empty cfg).val := by
  simp [empty]

/-- `toFin` of the empty in-field board is `∅`. -/
@[simp] theorem toFin_empty (cfg : GameConfig) : toFin (empty cfg) = ∅ := by
  unfold toFin
  simp [empty]

/-- `ofFin` of the empty `Fin`-board is the empty in-field board. -/
@[simp] theorem ofFin_empty (cfg : GameConfig) :
    ofFin (∅ : Finset (Fin cfg.cols × Fin cfg.rows)) = empty cfg := by
  apply toFin_injective
  rw [toFin_ofFin, toFin_empty]

/-- An in-field board equals `empty` iff its carrier is `∅`. -/
theorem eq_empty_iff_val_eq_empty (b : InFieldBoard cfg) :
    b = empty cfg ↔ b.val = ∅ := by
  refine ⟨fun h => by rw [h]; rfl, fun h => ?_⟩
  apply Subtype.ext
  rw [h]; rfl

/-- An in-field board equals `empty` iff its `val.card` is `0`. -/
theorem eq_empty_iff_val_card_eq_zero (b : InFieldBoard cfg) :
    b = empty cfg ↔ b.val.card = 0 := by
  rw [eq_empty_iff_val_eq_empty, Finset.card_eq_zero]

/-- Every column of the empty in-field board has height `0`. -/
@[simp] theorem colHeight_empty (cfg : GameConfig) (j : ℕ) :
    (empty cfg).val.colHeight j = 0 := by
  change Board.colHeight ∅ j = 0
  exact Board.colHeight_empty j

/-- The empty in-field board is not lost. -/
@[simp] theorem not_isLost_empty (cfg : GameConfig) :
    ¬ Board.isLost cfg (empty cfg).val := by
  rintro ⟨p, hp, _⟩
  exact absurd hp (Finset.notMem_empty _)

/-- The carrier of any in-field board is a subset of the universe-as-Finset
of the underlying coordinate type — vacuous bound used elsewhere as a
default `Finset` skeleton. -/
theorem empty_subset (b : InFieldBoard cfg) : (empty cfg).val ⊆ b.val := by
  rw [empty_val]; exact Finset.empty_subset _

/-- An in-field board whose carrier is contained in `∅` equals `empty`. -/
theorem eq_empty_of_val_subset_empty (b : InFieldBoard cfg)
    (h : b.val ⊆ ∅) : b = empty cfg :=
  (eq_empty_iff_val_eq_empty b).mpr (Finset.subset_empty.mp h)

/-- A symmetric form of `eq_empty_iff_val_card_eq_zero`: `empty = b
iff val.card = 0`. -/
theorem empty_eq_iff_val_card_eq_zero (b : InFieldBoard cfg) :
    empty cfg = b ↔ b.val.card = 0 :=
  ⟨fun h => h ▸ empty_val_card cfg,
   fun h => ((eq_empty_iff_val_card_eq_zero b).mpr h).symm⟩

/-- Symmetric form of `eq_empty_iff_val_eq_empty`. -/
theorem empty_eq_iff_val_eq_empty (b : InFieldBoard cfg) :
    empty cfg = b ↔ b.val = ∅ :=
  ⟨fun h => h ▸ empty_val cfg,
   fun h => ((eq_empty_iff_val_eq_empty b).mpr h).symm⟩

/-- `b ≠ empty` iff its carrier is nonempty. -/
theorem ne_empty_iff_val_nonempty (b : InFieldBoard cfg) :
    b ≠ empty cfg ↔ b.val.Nonempty := by
  rw [Ne, eq_empty_iff_val_eq_empty]
  exact Finset.nonempty_iff_ne_empty.symm

/-- `b ≠ empty` iff its carrier has positive cardinality. -/
theorem ne_empty_iff_val_card_pos (b : InFieldBoard cfg) :
    b ≠ empty cfg ↔ 0 < b.val.card := by
  rw [ne_empty_iff_val_nonempty, Finset.card_pos]

/-- `b ≠ empty` iff its carrier's cardinality is `≥ 1`. -/
theorem ne_empty_iff_one_le_val_card (b : InFieldBoard cfg) :
    b ≠ empty cfg ↔ 1 ≤ b.val.card :=
  ne_empty_iff_val_card_pos b

/-- `b ≠ empty` iff its carrier's cardinality is nonzero. -/
theorem ne_empty_iff_val_card_ne_zero (b : InFieldBoard cfg) :
    b ≠ empty cfg ↔ b.val.card ≠ 0 := by
  rw [ne_empty_iff_val_card_pos, Nat.pos_iff_ne_zero]

end InFieldBoard
end Tetris
