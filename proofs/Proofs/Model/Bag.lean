import Mathlib
import Proofs.Model.Piece

/-!
# The 7-bag randomizer

The bag holds the pieces not yet drawn from the current bag. Drawing removes a
piece; when the bag empties it refills with all 7 pieces. A piece is drawable
exactly when it is still present in the current bag. This is the standard 7-bag
("1 piece drawn per step, current piece only") randomizer from the reference
engine's `TetrisPieceBagState`.
-/

namespace Tetris

/-- The current bag: the set of pieces not yet drawn this cycle. -/
abbrev Bag := Finset Piece

namespace Bag

/-- A fresh, full bag containing all 7 pieces. -/
def full : Bag := Finset.univ

/-- A piece may be drawn iff it remains in the current bag. -/
def canDraw (bag : Bag) (p : Piece) : Prop := p ∈ bag

instance (bag : Bag) (p : Piece) : Decidable (canDraw bag p) := by
  unfold canDraw; infer_instance

/-- Draw piece `p`: remove it from the bag, refilling to a full bag if that
empties the current one. -/
def draw (bag : Bag) (p : Piece) : Bag :=
  if bag.erase p = ∅ then full else bag.erase p

/-- The full bag contains every piece. -/
theorem mem_full (p : Piece) : p ∈ full := Finset.mem_univ p

/-- Every piece is drawable from the full bag. -/
theorem canDraw_full (p : Piece) : Bag.full.canDraw p := mem_full p

/-- The full bag has 7 pieces. -/
@[simp] theorem full_card : (full : Bag).card = 7 := by decide

/-- The full bag has positive card. -/
theorem full_card_pos : 0 < (full : Bag).card := by rw [full_card]; decide

/-- The full bag has card `< 8`. -/
theorem full_card_lt_eight : (full : Bag).card < 8 := by rw [full_card]; decide

/-- The full bag's card is `≥ 1`. -/
theorem one_le_full_card : 1 ≤ (full : Bag).card := full_card_pos

/-- The full bag's card is nonzero. -/
theorem full_card_ne_zero : (full : Bag).card ≠ 0 := Nat.ne_of_gt full_card_pos

/-- The full bag is not a singleton. -/
theorem full_ne_singleton (p : Piece) : (full : Bag) ≠ {p} := by
  intro h
  have : (full : Bag).card = ({p} : Bag).card := congrArg Finset.card h
  rw [full_card, Finset.card_singleton] at this
  exact absurd this (by decide)

/-- The full bag equals `Finset.univ`. -/
theorem full_eq_univ : (full : Bag) = Finset.univ := rfl

/-- Every bag has at most 7 pieces (since it's a subset of the universe). -/
theorem card_le_seven (bag : Bag) : bag.card ≤ 7 := by
  rw [← full_card]
  exact Finset.card_le_card (Finset.subset_univ _)

/-- A bag has exactly 7 pieces iff it equals the full bag. -/
theorem card_eq_seven_iff_eq_full (bag : Bag) : bag.card = 7 ↔ bag = Bag.full := by
  refine ⟨fun h => ?_, fun h => by rw [h, full_card]⟩
  exact Finset.eq_of_subset_of_card_le (Finset.subset_univ _)
    (by rw [full_card, h])

/-- Symm of `card_eq_seven_iff_eq_full`. -/
theorem eq_full_iff_card_eq_seven (bag : Bag) :
    bag = Bag.full ↔ bag.card = 7 :=
  (card_eq_seven_iff_eq_full bag).symm

/-- The empty bag has card 0. -/
@[simp] theorem empty_card : (∅ : Bag).card = 0 := Finset.card_empty

/-- A bag is empty iff its card is 0. -/
theorem eq_empty_iff_card_eq_zero (bag : Bag) : bag = ∅ ↔ bag.card = 0 :=
  Finset.card_eq_zero.symm

/-- Either the bag is full or its card is ≤ 6. -/
theorem eq_full_or_card_le_six (bag : Bag) :
    bag = Bag.full ∨ bag.card ≤ 6 := by
  rcases (card_le_seven bag).lt_or_eq with h | h
  · right; omega
  · left; exact (card_eq_seven_iff_eq_full bag).mp h

/-- Empty bag is a subset of full bag. -/
theorem empty_subset_full : (∅ : Bag) ⊆ Bag.full := Finset.empty_subset _

/-- Full-bag membership is universal. -/
theorem mem_full_iff (p : Piece) : p ∈ Bag.full ↔ True :=
  ⟨fun _ => trivial, fun _ => mem_full p⟩

/-- Every bag is a subset of `Bag.full`. -/
theorem subset_full (bag : Bag) : bag ⊆ Bag.full := Finset.subset_univ _

/-- `Bag.full` is included in a bag iff the bag is `Bag.full`. -/
theorem full_subset_iff_eq_full (bag : Bag) :
    Bag.full ⊆ bag ↔ bag = Bag.full :=
  ⟨fun h => Finset.Subset.antisymm (subset_full _) h,
   fun h => h.symm ▸ Finset.Subset.refl _⟩

/-- `Bag.full.erase p` has exactly 6 pieces for any piece. -/
@[simp] theorem full_erase_card (p : Piece) : (Bag.full.erase p).card = 6 := by
  rw [Finset.card_erase_of_mem (Bag.mem_full p), Bag.full_card]

/-- Drawing from the full bag never refills (since the remaining bag has 6
pieces, which is nonempty). -/
@[simp] theorem draw_full_eq_erase (p : Piece) :
    Bag.full.draw p = Bag.full.erase p := by
  unfold draw
  have h : Bag.full.erase p ≠ ∅ := by
    intro he
    have : (Bag.full.erase p).card = 0 := by rw [he]; exact Finset.card_empty
    rw [full_erase_card] at this
    exact absurd this (by decide)
  rw [if_neg h]

/-- The bag after drawing from full has exactly 6 pieces. -/
@[simp] theorem draw_full_card (p : Piece) : (Bag.full.draw p).card = 6 := by
  rw [draw_full_eq_erase, full_erase_card]

/-- Drawing the unique piece from a singleton bag refills to full. -/
@[simp] theorem draw_singleton_eq_full (p : Piece) :
    ({p} : Bag).draw p = Bag.full := by
  unfold draw
  rw [Finset.erase_singleton, if_pos rfl]

/-- Drawing from full never gives full back: the drawn piece is now gone. -/
theorem draw_full_ne_full (p : Piece) : Bag.full.draw p ≠ Bag.full := by
  rw [draw_full_eq_erase]
  intro he
  have hp : p ∈ Bag.full.erase p := he.symm ▸ Bag.mem_full p
  exact (Finset.notMem_erase _ _) hp

/-- There are 2^7 = 128 possible bags in total. -/
@[simp] theorem fintype_card : Fintype.card Bag = 128 := by
  change Fintype.card (Finset Piece) = 128
  rw [Fintype.card_finset, Piece.card_eq_seven]
  decide

/-- The full bag is nonempty. -/
theorem full_nonempty : (full : Bag).Nonempty := ⟨Piece.O, mem_full _⟩

/-- A singleton bag has card 1. -/
@[simp] theorem singleton_card (p : Piece) : ({p} : Bag).card = 1 :=
  Finset.card_singleton _

/-- A singleton bag is not the full bag. -/
theorem singleton_ne_full (p : Piece) : ({p} : Bag) ≠ Bag.full := by
  intro h
  have hc : ({p} : Bag).card = (Bag.full : Bag).card := congrArg Finset.card h
  rw [singleton_card, full_card] at hc
  exact absurd hc (by decide)

/-- A singleton bag is nonempty. -/
theorem singleton_nonempty (p : Piece) : ({p} : Bag).Nonempty :=
  Finset.singleton_nonempty p

/-- The full bag is not empty. -/
theorem full_ne_empty : (full : Bag) ≠ ∅ := Finset.Nonempty.ne_empty full_nonempty

/-- Empty bag is a strict subset of full bag. -/
theorem empty_ssubset_full : (∅ : Bag) ⊂ Bag.full := by
  rw [Finset.ssubset_iff_subset_ne]
  exact ⟨empty_subset_full, Ne.symm full_ne_empty⟩


/-- Drawing always leaves a nonempty bag (it refills when emptied). -/
theorem draw_nonempty (bag : Bag) (p : Piece) : (bag.draw p).Nonempty := by
  unfold draw
  split_ifs with h
  · exact full_nonempty
  · exact Finset.nonempty_of_ne_empty h

/-- Drawing is never empty. -/
theorem draw_ne_empty (bag : Bag) (p : Piece) : bag.draw p ≠ ∅ :=
  (draw_nonempty _ _).ne_empty

/-- Drawing from full is never empty. -/
theorem draw_full_ne_empty (p : Piece) : Bag.full.draw p ≠ ∅ :=
  draw_ne_empty _ _

/-- `card = 7 ⇒ draw card = 6`. -/
theorem card_draw_eq_six_of_card_eq_seven {bag : Bag} {p : Piece}
    (h : bag.card = 7) : (bag.draw p).card = 6 := by
  rw [(card_eq_seven_iff_eq_full bag).mp h]; exact draw_full_card p

/-- Empty bag is not a singleton. -/
theorem empty_ne_singleton (p : Piece) : (∅ : Bag) ≠ {p} := by
  intro h
  have : (∅ : Bag).card = ({p} : Bag).card := congrArg _ h
  rw [empty_card, singleton_card] at this
  exact absurd this (by decide)

/-- A singleton bag is not empty. -/
theorem singleton_ne_empty (p : Piece) : ({p} : Bag) ≠ ∅ :=
  (empty_ne_singleton p).symm

/-- The bag card after any draw is positive. -/
theorem draw_card_pos (bag : Bag) (p : Piece) : 0 < (bag.draw p).card :=
  Finset.card_pos.mpr (draw_nonempty bag p)

/-- The bag card after any draw is ≤ 7. -/
theorem draw_card_le_seven (bag : Bag) (p : Piece) : (bag.draw p).card ≤ 7 :=
  card_le_seven _

/-- Distinct pieces in the same bag give distinct draw results. Used to bound
the size of any adversarial closed cycle from below by the bag's cardinality. -/
theorem draw_injOn (bag : Bag) : Set.InjOn bag.draw ↑bag := by
  intro p hp q hq hpq
  by_contra hne
  -- p ≠ q, both in bag ⇒ neither erase is empty (the other piece survives).
  have hpne : bag.erase p ≠ ∅ := by
    intro he
    have : q ∈ bag.erase p := Finset.mem_erase.mpr ⟨Ne.symm hne, hq⟩
    rw [he] at this
    exact absurd this (Finset.notMem_empty _)
  have hqne : bag.erase q ≠ ∅ := by
    intro he
    have : p ∈ bag.erase q := Finset.mem_erase.mpr ⟨hne, hp⟩
    rw [he] at this
    exact absurd this (Finset.notMem_empty _)
  -- Both non-refill: `draw = erase`. `q ∈ bag.erase p` but `q ∉ bag.erase q`.
  unfold Bag.draw at hpq
  rw [if_neg hpne, if_neg hqne] at hpq
  have hq_in : q ∈ bag.erase p := Finset.mem_erase.mpr ⟨Ne.symm hne, hq⟩
  rw [hpq] at hq_in
  exact absurd hq_in (Finset.notMem_erase _ _)

/-- **Bag card after a non-emptying draw.** If the bag has at least 2 pieces
and `p` is one of them, drawing `p` leaves `bag.card - 1` pieces (no refill).
Companion to `card_draw_of_singleton` for the emptying case. -/
theorem card_draw_of_ge_two {bag : Bag} {p : Piece} (hp : p ∈ bag)
    (h : 2 ≤ bag.card) : (bag.draw p).card = bag.card - 1 := by
  have herase_card : (bag.erase p).card = bag.card - 1 := Finset.card_erase_of_mem hp
  have hne : bag.erase p ≠ ∅ := by
    intro he
    rw [he, Finset.card_empty] at herase_card
    omega
  unfold draw
  rw [if_neg hne, herase_card]

/-- **Card strictly decreases on a non-emptying draw**: when card ≥
2, drawing reduces the card by 1 (strictly < original). -/
theorem card_draw_lt_card_of_ge_two {bag : Bag} {p : Piece} (hp : p ∈ bag)
    (h : 2 ≤ bag.card) : (bag.draw p).card < bag.card := by
  rw [card_draw_of_ge_two hp h]; omega


/-- **Non-emptying draw equals erase**: when `card ≥ 2`, a draw of an
in-bag piece equals erase (no refill). -/
theorem draw_eq_erase_of_card_ge_two {bag : Bag} {p : Piece} (hp : p ∈ bag)
    (h : 2 ≤ bag.card) : bag.draw p = bag.erase p := by
  have herase_card : (bag.erase p).card = bag.card - 1 := Finset.card_erase_of_mem hp
  have hne : bag.erase p ≠ ∅ := by
    intro he
    rw [he, Finset.card_empty] at herase_card
    omega
  unfold draw
  rw [if_neg hne]

/-- **Drawn piece not in post-draw bag** (non-refill case): when
`card ≥ 2`, drawing `p` removes `p` from the bag. -/
theorem draw_notMem_of_card_ge_two {bag : Bag} {p : Piece} (hp : p ∈ bag)
    (h : 2 ≤ bag.card) : p ∉ bag.draw p := by
  rw [draw_eq_erase_of_card_ge_two hp h]
  exact Finset.notMem_erase _ _

/-- **Refill characterization**: a draw refills (gives the full bag)
iff the erase result is empty. -/
theorem draw_eq_full_iff_erase_eq_empty (bag : Bag) (p : Piece) :
    bag.draw p = Bag.full ↔ bag.erase p = ∅ := by
  unfold draw
  split_ifs with h
  · exact ⟨fun _ => h, fun _ => rfl⟩
  · refine ⟨fun heq => ?_, fun he => absurd he h⟩
    exact absurd (heq.symm ▸ Bag.mem_full p) (Finset.notMem_erase _ _)

/-- **Refill iff singleton** (for nonempty bag): when the bag is
nonempty, drawing refills iff the bag was a singleton `{p}`. -/
theorem draw_eq_full_iff_eq_singleton {bag : Bag} (hne : bag.Nonempty) (p : Piece) :
    bag.draw p = Bag.full ↔ bag = {p} := by
  rw [draw_eq_full_iff_erase_eq_empty]
  refine ⟨fun h => ?_, fun h => by rw [h, Finset.erase_singleton]⟩
  have hsub : bag ⊆ {p} := by
    intro q hq
    by_cases hqp : q = p
    · rw [hqp]; exact Finset.mem_singleton.mpr rfl
    · have : q ∈ bag.erase p := Finset.mem_erase.mpr ⟨hqp, hq⟩
      rw [h] at this; exact absurd this (Finset.notMem_empty _)
  -- bag ⊆ {p} and bag nonempty ⇒ bag = {p}.
  obtain ⟨q, hq⟩ := hne
  have hqp : q = p := Finset.mem_singleton.mp (hsub hq)
  apply Finset.Subset.antisymm hsub
  intro x hx
  rw [Finset.mem_singleton] at hx
  rw [hx, ← hqp]; exact hq

/-- **Bag card after the emptying draw.** If the bag has exactly one piece
(necessarily `p`), drawing refills to a full bag of 7 pieces. -/
theorem card_draw_of_singleton {bag : Bag} {p : Piece} (hp : p ∈ bag)
    (h : bag.card = 1) : (bag.draw p).card = 7 := by
  have hsing : bag = {p} := by
    have hb := Finset.card_eq_one.mp h
    rcases hb with ⟨q, hq⟩
    rw [hq] at hp
    rw [← Finset.mem_singleton.mp hp] at hq
    exact hq
  have herase_empty : bag.erase p = ∅ := by
    rw [hsing]; exact Finset.erase_singleton p
  unfold draw
  rw [if_pos herase_empty]
  change Bag.full.card = 7
  decide

/-- **After-draw card is either `card - 1` (non-refill) or `7`
(refill)** — when the drawn piece was in the bag. -/
theorem card_draw_cases_of_mem {bag : Bag} {p : Piece} (hp : p ∈ bag) :
    (bag.draw p).card = bag.card - 1 ∨ (bag.draw p).card = 7 := by
  have hpos : 0 < bag.card := Finset.card_pos.mpr ⟨p, hp⟩
  by_cases h : 2 ≤ bag.card
  · left; exact card_draw_of_ge_two hp h
  · right
    have : bag.card = 1 := by omega
    exact card_draw_of_singleton hp this

/-- **Draw of a piece not in the bag is a no-op on nonempty bags**. -/
theorem draw_card_eq_card_of_notMem {bag : Bag} {p : Piece}
    (hp : p ∉ bag) (hne : bag.Nonempty) :
    (bag.draw p).card = bag.card := by
  unfold draw
  have herase : bag.erase p = bag := Finset.erase_eq_of_notMem hp
  rw [herase]
  rw [if_neg hne.ne_empty]

/-- **Identity form**: draw of a piece not in the nonempty bag is the
identity. -/
theorem draw_eq_self_of_notMem {bag : Bag} {p : Piece}
    (hp : p ∉ bag) (hne : bag.Nonempty) :
    bag.draw p = bag := by
  unfold draw
  have herase : bag.erase p = bag := Finset.erase_eq_of_notMem hp
  rw [herase]
  rw [if_neg hne.ne_empty]


/-- **General after-draw card bound**: regardless of in/out, the
after-draw card is either ≤ pre-card or equals 7. -/
theorem draw_card_le_card_or_eq_seven (bag : Bag) (p : Piece) :
    (bag.draw p).card ≤ bag.card ∨ (bag.draw p).card = 7 := by
  by_cases hp : p ∈ bag
  · rcases card_draw_cases_of_mem hp with h | h
    · left; rw [h]; exact Nat.sub_le _ _
    · right; exact h
  · -- p ∉ bag: draw = if (erase p = ∅) then full else erase p = bag
    unfold draw
    split_ifs with h
    · right; rw [full_card]
    · left
      rw [Finset.erase_eq_of_notMem hp]

/-- **After draw, bag is either a subset of pre-bag (non-refill) or
equals `Bag.full` (refill)**. Trichotomy along the refill split. -/
theorem draw_subset_self_or_eq_full (bag : Bag) (p : Piece) :
    bag.draw p ⊆ bag ∨ bag.draw p = Bag.full := by
  unfold draw
  split_ifs with h
  · right; rfl
  · left; exact Finset.erase_subset _ _

/-- Drawing a piece **always** changes the bag (whether the bag is emptied and
refilled, or the piece is simply removed). Key structural fact for closed
cycles: any state with a non-empty bag transitions to a different state. -/
theorem draw_ne_self {bag : Bag} {p : Piece} (hp : p ∈ bag) : bag.draw p ≠ bag := by
  unfold draw
  split_ifs with h
  · -- Refill case: draw = full. Need full ≠ bag.
    -- From h: bag.erase p = ∅, so bag ⊆ {p}. Combined with hp, bag = {p}.
    -- Then bag.card = 1 but full.card = 7.
    intro heq
    rw [← heq] at h
    have hc := Finset.card_erase_of_mem (Bag.mem_full p)
    rw [h, Finset.card_empty] at hc
    have hfull : Bag.full.card = 7 := by decide
    rw [hfull] at hc
    exact absurd hc (by decide)
  · -- Non-refill case: draw = bag.erase p. erase removes p, so card decreases by 1.
    intro heq
    have hpos : 0 < bag.card := Finset.card_pos.mpr ⟨p, hp⟩
    have hcard : (bag.erase p).card = bag.card := by rw [heq]
    rw [Finset.card_erase_of_mem hp] at hcard
    omega

/-- **Draw changes the bag iff the piece was in it** (nonempty bag).
Combines `draw_ne_self` with `draw_eq_self_of_notMem`. -/
theorem draw_ne_self_iff_mem {bag : Bag} {p : Piece}
    (hne : bag.Nonempty) : bag.draw p ≠ bag ↔ p ∈ bag := by
  refine ⟨fun h => ?_, fun h => draw_ne_self h⟩
  by_contra hp
  exact h (draw_eq_self_of_notMem hp hne)

/-- **Draw is identity iff piece not in bag** (nonempty bag).
Contrapositive of `draw_ne_self_iff_mem`. -/
theorem draw_eq_self_iff_notMem {bag : Bag} {p : Piece}
    (hne : bag.Nonempty) : bag.draw p = bag ↔ p ∉ bag :=
  ⟨fun h hp => draw_ne_self hp h,
   fun h => draw_eq_self_of_notMem h hne⟩

/-- Pick an arbitrary piece from a nonempty bag (via `Finset.Nonempty.choose`).
Used to construct a canonical legal piece sequence. -/
noncomputable def someElement (bag : Bag) (h : bag.Nonempty) : Piece := h.choose

theorem someElement_mem (bag : Bag) (h : bag.Nonempty) : bag.someElement h ∈ bag :=
  h.choose_spec

/-- `canDraw` is defeq to set-membership. Convenient `rfl` lemma for
rewriting between `canDraw` and `∈`. -/
@[simp] theorem canDraw_iff_mem (bag : Bag) (p : Piece) :
    bag.canDraw p ↔ p ∈ bag := Iff.rfl

/-- The `someElement` of a nonempty bag is drawable. -/
theorem someElement_canDraw (bag : Bag) (h : bag.Nonempty) :
    bag.canDraw (bag.someElement h) :=
  bag.someElement_mem h

/-- Drawing `someElement` from a nonempty bag changes the bag
unless `bag.card = 1` (in which case the draw refills to full,
which is `≠ bag` when `bag ≠ full`). The general structural fact
is captured by `draw_ne_self` applied to `someElement`. -/
theorem draw_someElement_ne_self (bag : Bag) (h : bag.Nonempty) :
    bag.draw (bag.someElement h) ≠ bag :=
  draw_ne_self (someElement_mem bag h)

/-- Any nonempty bag has a drawable piece. Existence form of
`someElement_canDraw`. -/
theorem exists_canDraw_of_nonempty {bag : Bag} (h : bag.Nonempty) :
    ∃ p : Piece, bag.canDraw p :=
  ⟨bag.someElement h, someElement_canDraw bag h⟩

/-- `Bag.Nonempty` iff `bag ≠ ∅`. Mathlib has `Finset.nonempty_iff_ne_empty`
but the `Bag` namespace alias is convenient. -/
theorem nonempty_iff_ne_empty (bag : Bag) : bag.Nonempty ↔ bag ≠ ∅ :=
  Finset.nonempty_iff_ne_empty

/-- Symmetric form. -/
theorem ne_empty_iff_nonempty (bag : Bag) : bag ≠ ∅ ↔ bag.Nonempty :=
  (nonempty_iff_ne_empty bag).symm

/-- `bag.draw p` has nonzero card. -/
theorem draw_card_ne_zero (bag : Bag) (p : Piece) : (bag.draw p).card ≠ 0 :=
  Nat.ne_of_gt (draw_card_pos bag p)

/-- `bag.draw p ≥ 1`. -/
theorem draw_card_ge_one (bag : Bag) (p : Piece) : 1 ≤ (bag.draw p).card :=
  draw_card_pos bag p

/-- `bag.draw p` has card strictly less than 8. -/
theorem draw_card_lt_eight (bag : Bag) (p : Piece) : (bag.draw p).card < 8 := by
  have := card_le_seven (bag.draw p)
  omega

/-- Bundled card interval `[1, 7]` for `draw`. -/
theorem draw_card_interval (bag : Bag) (p : Piece) :
    1 ≤ (bag.draw p).card ∧ (bag.draw p).card ≤ 7 :=
  ⟨draw_card_ge_one bag p, card_le_seven _⟩

/-- `bag.draw p ⊆ Bag.full`. -/
theorem draw_subset_full (bag : Bag) (p : Piece) : bag.draw p ⊆ Bag.full :=
  Bag.subset_full _

/-- Drawing from the empty bag yields the full bag (since
`empty.erase = empty` triggers the refill branch). -/
@[simp] theorem draw_empty (p : Piece) : (∅ : Bag).draw p = Bag.full := by
  simp [draw]

/-- Drawing from the empty bag yields card 7. -/
@[simp] theorem draw_empty_card (p : Piece) : ((∅ : Bag).draw p).card = 7 := by
  rw [draw_empty]; exact full_card

/-- Drawing from the empty bag yields a nonempty bag. -/
theorem draw_empty_nonempty (p : Piece) : ((∅ : Bag).draw p).Nonempty := by
  rw [draw_empty]; exact full_nonempty

/-- Drawing `p` from a singleton `{p}` yields the full bag (card 7). -/
@[simp] theorem draw_singleton_card_eq_seven (p : Piece) :
    (({p} : Bag).draw p).card = 7 := by
  rw [draw_singleton_eq_full]; exact full_card

/-- A singleton bag is a subset of `Bag.full`. -/
theorem singleton_subset_full (p : Piece) : ({p} : Bag) ⊆ Bag.full :=
  Bag.subset_full _

/-- Drawing from `Bag.full` yields a strict subset (the `erase`). -/
theorem draw_full_ssubset_full (p : Piece) : Bag.full.draw p ⊂ Bag.full := by
  rw [draw_full_eq_erase]
  exact Finset.erase_ssubset (mem_full p)

/-- Drawing from full has positive card (= 6 > 0). -/
theorem draw_full_card_pos (p : Piece) : 0 < (Bag.full.draw p).card := by
  rw [draw_full_card]; decide

/-- Drawing from full has card 6 ≠ 7 (not full). -/
theorem draw_full_card_ne_seven (p : Piece) : (Bag.full.draw p).card ≠ 7 := by
  rw [draw_full_card]; decide

/-- Drawing from full has card < 7. -/
theorem draw_full_card_lt_seven (p : Piece) : (Bag.full.draw p).card < 7 := by
  rw [draw_full_card]; decide

/-- Drawing from full yields a nonempty bag. -/
theorem draw_full_nonempty (p : Piece) : (Bag.full.draw p).Nonempty :=
  draw_nonempty Bag.full p

/-- Drawing from full has card ≥ 1. -/
theorem draw_full_card_ge_one (p : Piece) : 1 ≤ (Bag.full.draw p).card :=
  draw_full_card_pos p

/-- Drawing from full has card ≤ 6. -/
theorem draw_full_card_le_six (p : Piece) : (Bag.full.draw p).card ≤ 6 := by
  rw [draw_full_card]

/-- Drawing from full has card < 8. -/
theorem draw_full_card_lt_eight (p : Piece) : (Bag.full.draw p).card < 8 := by
  have := card_le_seven (Bag.full.draw p)
  omega

/-- Drawing from full has nonzero card. -/
theorem draw_full_card_ne_zero (p : Piece) : (Bag.full.draw p).card ≠ 0 :=
  Nat.ne_of_gt (draw_full_card_pos p)

/-- Bundled `[1, 6]` interval for `Bag.full.draw`. -/
theorem draw_full_card_interval (p : Piece) :
    1 ≤ (Bag.full.draw p).card ∧ (Bag.full.draw p).card ≤ 6 :=
  ⟨draw_full_card_ge_one p, draw_full_card_le_six p⟩

/-- `Bag.full.draw p ⊆ Bag.full`. -/
theorem draw_full_subset_full (p : Piece) : Bag.full.draw p ⊆ Bag.full :=
  Bag.subset_full _

/-- `Bag.full.draw p ≠ Bag.empty`. -/
theorem draw_full_ne_empty_set (p : Piece) : Bag.full.draw p ≠ (∅ : Bag) :=
  draw_full_ne_empty p

/-- Membership in `Bag.full.draw p` iff member is not p. -/
theorem mem_draw_full_iff_ne (p q : Piece) :
    q ∈ Bag.full.draw p ↔ q ≠ p := by
  rw [draw_full_eq_erase, Finset.mem_erase]
  exact ⟨fun h => h.1, fun h => ⟨h, Bag.mem_full q⟩⟩

/-- The drawn piece is not in the resulting bag (when drawn from full). -/
theorem notMem_draw_full (p : Piece) : p ∉ Bag.full.draw p :=
  fun h => ((mem_draw_full_iff_ne p p).mp h) rfl

/-- `Bag.full.draw p = Bag.full ↔ False`. -/
theorem draw_full_eq_full_iff_false (p : Piece) :
    Bag.full.draw p = Bag.full ↔ False :=
  ⟨draw_full_ne_full p, False.elim⟩

/-- If `p ∈ bag` and the after-draw card is *not* 7, then drawing
strictly reduced the cardinality. -/
theorem draw_card_eq_pred_of_mem_of_ne_seven {bag : Bag} {p : Piece}
    (hp : p ∈ bag) (h7 : (bag.draw p).card ≠ 7) :
    (bag.draw p).card = bag.card - 1 :=
  (card_draw_cases_of_mem hp).resolve_right h7

/-- If `p ∈ bag` and after-draw card is not `bag.card - 1`, then drawing
refilled the bag (to 7). -/
theorem draw_card_eq_seven_of_mem_of_ne_pred {bag : Bag} {p : Piece}
    (hp : p ∈ bag) (hne : (bag.draw p).card ≠ bag.card - 1) :
    (bag.draw p).card = 7 :=
  (card_draw_cases_of_mem hp).resolve_left hne

/-- If `p ∈ bag` and `bag.card ≥ 2`, draw strictly decreases card. -/
theorem draw_card_lt_card_of_mem_of_card_ge_two {bag : Bag} {p : Piece}
    (hp : p ∈ bag) (hge : 2 ≤ bag.card) :
    (bag.draw p).card < bag.card := by
  rw [card_draw_of_ge_two hp hge]
  omega

/-- Non-strict companion to `draw_card_lt_card_of_mem_of_card_ge_two`. -/
theorem draw_card_le_card_of_mem_of_card_ge_two {bag : Bag} {p : Piece}
    (hp : p ∈ bag) (hge : 2 ≤ bag.card) :
    (bag.draw p).card ≤ bag.card :=
  Nat.le_of_lt (draw_card_lt_card_of_mem_of_card_ge_two hp hge)

/-- For `p ∈ bag`, the after-draw card is 7 iff the bag had a single piece
(the refill case). -/
theorem draw_card_eq_seven_iff_card_eq_one_of_mem {bag : Bag} {p : Piece}
    (hp : p ∈ bag) :
    (bag.draw p).card = 7 ↔ bag.card = 1 := by
  refine ⟨fun h7 => ?_, fun h1 => card_draw_of_singleton hp h1⟩
  -- Forward: if not card = 1, then card ≥ 2, then erase → card - 1 ≤ 6 < 7.
  have hpos : 0 < bag.card := Finset.card_pos.mpr ⟨p, hp⟩
  have hle7 : bag.card ≤ 7 := bag.card_le_seven
  by_contra hne
  have hge : 2 ≤ bag.card := by omega
  rw [card_draw_of_ge_two hp hge] at h7
  omega

/-- For `p ∈ bag`, the after-draw card is `bag.card - 1` iff `bag.card ≥ 2`
(the strictly-decreasing erase case). -/
theorem draw_card_eq_pred_iff_card_ge_two_of_mem {bag : Bag} {p : Piece}
    (hp : p ∈ bag) :
    (bag.draw p).card = bag.card - 1 ↔ 2 ≤ bag.card := by
  refine ⟨fun heq => ?_, fun hge => card_draw_of_ge_two hp hge⟩
  -- Forward: if not ≥ 2 then card = 1, then post-draw = 7 ≠ 0 = card - 1.
  have hpos : 0 < bag.card := Finset.card_pos.mpr ⟨p, hp⟩
  by_contra hlt
  have h1 : bag.card = 1 := by omega
  have h7 : (bag.draw p).card = 7 := card_draw_of_singleton hp h1
  rw [h7, h1] at heq
  omega

/-- For `p ∈ bag`, the after-draw card is `< 7` iff `bag.card ≥ 2`. -/
theorem draw_card_lt_seven_iff_card_ge_two_of_mem {bag : Bag} {p : Piece}
    (hp : p ∈ bag) :
    (bag.draw p).card < 7 ↔ 2 ≤ bag.card := by
  refine ⟨fun hlt => ?_, fun hge => ?_⟩
  · have hpos : 0 < bag.card := Finset.card_pos.mpr ⟨p, hp⟩
    by_contra hcon
    have h1 : bag.card = 1 := by omega
    have h7 : (bag.draw p).card = 7 := card_draw_of_singleton hp h1
    omega
  · rw [card_draw_of_ge_two hp hge]
    have : bag.card ≤ 7 := bag.card_le_seven
    omega

/-- For `p ∈ bag`, the after-draw card is `≠ 7` iff `bag.card ≥ 2`. -/
theorem draw_card_ne_seven_iff_card_ge_two_of_mem {bag : Bag} {p : Piece}
    (hp : p ∈ bag) :
    (bag.draw p).card ≠ 7 ↔ 2 ≤ bag.card := by
  rw [← draw_card_lt_seven_iff_card_ge_two_of_mem hp]
  have hle : (bag.draw p).card ≤ 7 := draw_card_le_seven bag p
  omega

/-- For `p ∈ bag`, the after-draw card is `≤ 6` iff `bag.card ≥ 2`. -/
theorem draw_card_le_six_iff_card_ge_two_of_mem {bag : Bag} {p : Piece}
    (hp : p ∈ bag) :
    (bag.draw p).card ≤ 6 ↔ 2 ≤ bag.card := by
  rw [← draw_card_lt_seven_iff_card_ge_two_of_mem hp]
  omega

/-- Drawing from any bag yields either the erased bag or the full bag. -/
theorem draw_eq_erase_or_full (bag : Bag) (p : Piece) :
    bag.draw p = bag.erase p ∨ bag.draw p = Bag.full := by
  unfold draw
  split_ifs with h
  · exact Or.inr rfl
  · exact Or.inl rfl

/-- Drawing yields a subset of the pre-draw bag or the full bag. -/
theorem draw_subset_or_eq_full (bag : Bag) (p : Piece) :
    bag.draw p ⊆ bag ∨ bag.draw p = Bag.full := by
  rcases draw_eq_erase_or_full bag p with herase | hfull
  · exact Or.inl (herase ▸ Finset.erase_subset _ _)
  · exact Or.inr hfull

/-- Drawing yields the erased bag exactly when the erased bag is nonempty.
(`Or.inl` branch of `draw_eq_erase_or_full`.) -/
theorem draw_eq_erase_iff (bag : Bag) (p : Piece) :
    bag.draw p = bag.erase p ↔ bag.erase p ≠ ∅ := by
  unfold draw
  split_ifs with h
  · refine ⟨fun heq => ?_, fun hne => absurd h hne⟩
    rw [h] at heq; exact absurd heq.symm (Ne.symm (full_ne_empty))
  · exact ⟨fun _ => h, fun _ => rfl⟩

/-- `bag.draw p = bag.erase p ↔ bag.draw p ≠ Bag.full`. -/
theorem draw_eq_erase_iff_ne_full (bag : Bag) (p : Piece) :
    bag.draw p = bag.erase p ↔ bag.draw p ≠ Bag.full := by
  refine (draw_eq_erase_iff bag p).trans ?_
  refine ⟨fun hne hfull => hne ?_, fun hne heq => hne ?_⟩
  · exact (draw_eq_full_iff_erase_eq_empty bag p).mp hfull
  · exact (draw_eq_full_iff_erase_eq_empty bag p).mpr heq

/-- The post-draw bag has at least two pieces. -/
theorem draw_full_card_ge_two (p : Piece) : 2 ≤ (Bag.full.draw p).card := by
  rw [draw_full_card]; decide

/-- Drawing from `Bag.full` never yields a singleton (it has 6 pieces). -/
theorem draw_full_ne_singleton (p q : Piece) : Bag.full.draw p ≠ ({q} : Bag) := by
  intro h
  have : (Bag.full.draw p).card = ({q} : Bag).card := congrArg Finset.card h
  rw [draw_full_card, Finset.card_singleton] at this
  exact absurd this (by decide)

/-- The post-draw card from a full bag is never 1. -/
theorem draw_full_card_ne_one (p : Piece) : (Bag.full.draw p).card ≠ 1 := by
  rw [draw_full_card]; decide

/-- Drawing from full strictly decreases the cardinality. -/
theorem draw_full_card_lt_full_card (p : Piece) :
    (Bag.full.draw p).card < (Bag.full : Bag).card := by
  rw [draw_full_card, full_card]; decide

/-- Non-strict counterpart of `draw_full_card_lt_full_card`. -/
theorem draw_full_card_le_full_card (p : Piece) :
    (Bag.full.draw p).card ≤ (Bag.full : Bag).card :=
  Nat.le_of_lt (draw_full_card_lt_full_card p)

/-- If `bag ≠ Bag.full` then there is a piece missing from `bag`. -/
theorem exists_notMem_of_ne_full {bag : Bag} (h : bag ≠ Bag.full) :
    ∃ p : Piece, p ∉ bag := by
  by_contra hall
  push Not at hall
  exact h (Finset.eq_univ_iff_forall.mpr hall)

/-- `canDraw` implies the bag is nonempty (the drawn piece witnesses it). -/
theorem nonempty_of_canDraw {bag : Bag} {p : Piece} (h : bag.canDraw p) :
    bag.Nonempty :=
  ⟨p, h⟩

/-- `canDraw` implies the bag has positive cardinality. -/
theorem card_pos_of_canDraw {bag : Bag} {p : Piece} (h : bag.canDraw p) :
    0 < bag.card :=
  Finset.card_pos.mpr (nonempty_of_canDraw h)

/-- `canDraw` implies the bag is not empty. -/
theorem ne_empty_of_canDraw {bag : Bag} {p : Piece} (h : bag.canDraw p) :
    bag ≠ ∅ :=
  (nonempty_of_canDraw h).ne_empty

/-- `canDraw` implies `1 ≤ bag.card`. -/
theorem one_le_card_of_canDraw {bag : Bag} {p : Piece} (h : bag.canDraw p) :
    1 ≤ bag.card :=
  card_pos_of_canDraw h

/-- `canDraw` implies `bag.card ≠ 0`. -/
theorem card_ne_zero_of_canDraw {bag : Bag} {p : Piece} (h : bag.canDraw p) :
    bag.card ≠ 0 :=
  Nat.ne_of_gt (card_pos_of_canDraw h)

/-- `Bag.full.canDraw` is always true. -/
@[simp] theorem canDraw_full_iff (p : Piece) : Bag.full.canDraw p ↔ True :=
  ⟨fun _ => trivial, fun _ => canDraw_full p⟩

/-- No piece can be drawn from the empty bag. -/
@[simp] theorem not_canDraw_empty (p : Piece) : ¬ (∅ : Bag).canDraw p :=
  Finset.notMem_empty p

/-- A bag with positive card always has a drawable piece. -/
theorem exists_canDraw_of_card_pos {bag : Bag} (h : 0 < bag.card) :
    ∃ p : Piece, bag.canDraw p :=
  Finset.card_pos.mp h

/-- A bag with `bag.card ≠ 0` always has a drawable piece. -/
theorem exists_canDraw_of_card_ne_zero {bag : Bag} (h : bag.card ≠ 0) :
    ∃ p : Piece, bag.canDraw p :=
  exists_canDraw_of_card_pos (Nat.pos_of_ne_zero h)

/-- A drawn piece is always drawable from `Bag.full`. -/
theorem canDraw_full_of_canDraw_draw {bag : Bag} {p q : Piece}
    (h : (bag.draw p).canDraw q) : Bag.full.canDraw q :=
  draw_subset_full bag p h

/-- Drawing `q` from `Bag.full.draw p` is allowed iff `q ≠ p`. -/
theorem canDraw_draw_full_iff (p q : Piece) :
    (Bag.full.draw p).canDraw q ↔ q ≠ p := by
  unfold draw canDraw
  have hne : Bag.full.erase p ≠ ∅ := by
    rw [Ne, ← Finset.card_eq_zero, Finset.card_erase_of_mem (mem_full p),
        full_card]
    decide
  simp [hne, Finset.mem_erase, mem_full]

/-- The just-drawn piece is no longer drawable from `Bag.full.draw p`. -/
theorem not_canDraw_draw_full_self (p : Piece) : ¬ (Bag.full.draw p).canDraw p :=
  (canDraw_draw_full_iff p p).not.mpr (by simp)

/-- Any piece distinct from `p` is drawable from `Bag.full.draw p`. -/
theorem canDraw_draw_full_of_ne {p q : Piece} (h : q ≠ p) :
    (Bag.full.draw p).canDraw q :=
  (canDraw_draw_full_iff p q).mpr h

/-- Membership in `Bag.full.draw p`: any piece distinct from `p`. -/
theorem mem_draw_full_iff (p q : Piece) : q ∈ Bag.full.draw p ↔ q ≠ p :=
  canDraw_draw_full_iff p q

/-- Pieces in `Bag.full.draw p` are exactly `≠ p`. -/
theorem mem_draw_full_of_ne {p q : Piece} (h : q ≠ p) : q ∈ Bag.full.draw p :=
  canDraw_draw_full_of_ne h

/-- Drawing the last piece of a singleton bag triggers a refill to `Bag.full`. -/
theorem draw_singleton_self (p : Piece) : ({p} : Bag).draw p = Bag.full := by
  unfold draw
  simp [Finset.erase_singleton]

/-- Drawing from a singleton: the bag accepts only its single piece. -/
theorem canDraw_singleton_iff (p q : Piece) :
    ({p} : Bag).canDraw q ↔ q = p :=
  Finset.mem_singleton

/-- After drawing the singleton's piece, every piece is drawable (refill). -/
theorem canDraw_draw_singleton_self (p q : Piece) :
    (({p} : Bag).draw p).canDraw q := by
  rw [draw_singleton_self]; exact canDraw_full q

/-- The card of `({p}.draw p)` is 7 (refilled). -/
theorem draw_singleton_self_card (p : Piece) :
    (({p} : Bag).draw p).card = 7 := by
  rw [draw_singleton_self, full_card]

/-- After the singleton refill, the bag equals `Bag.full` (alias). -/
theorem draw_singleton_self_eq_full (p : Piece) :
    ({p} : Bag).draw p = Bag.full :=
  draw_singleton_self p

end Bag
end Tetris
