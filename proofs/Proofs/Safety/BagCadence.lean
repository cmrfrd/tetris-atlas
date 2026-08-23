import Mathlib
import Proofs.Safety.Adversarial

/-!
# Bag cadence: every piece appears in every window of 13 draws

The 7-bag randomizer is not merely fair on average — it is **syndetic**: along
any legal piece sequence, *every* piece type appears in *every* window of 13
consecutive draws (`every_piece_within_thirteen`). The bound is tight: a piece
drawn first in one bag and last in the next sits at the ends of a 12-draw gap.

Two auxiliary facts carry the proof, both by induction on the bag's card:

* `exists_draw_within_card` — a piece currently in the bag is drawn before the
  bag empties: legal draws shrink the bag by one until the refill, so its
  entire content is spent within `card` steps.
* `bagAt_add_card_eq_full` — the bag refills to full in exactly `card` steps.

If the target piece is still in the bag it arrives within `card ≤ 7` draws;
if it is missing the bag holds at most 6 pieces, refills within 6 draws, and
the full bag delivers the piece within 7 more: `6 + 7 = 13`.

## The I-drought corollary

Specialised to I (`exists_I_within_thirteen`): **the maximum I-drought is 12
placements**. Combined with `tetris_requires_I` (only I clears four rows) this
is the quantitative constraint on tetris-well architectures: the well must
survive 12 consecutive non-I placements — 48 cells of mass to house elsewhere —
between guaranteed I deliveries, and this is the worst case, not an average.
This is the first cadence (as opposed to count) theorem about the bag: BagBurst
bounds *how much* of each piece a window holds, this bounds *how long* the
adversary can withhold one.
-/

namespace Tetris
namespace BagCadence

/-- **A piece in the bag is drawn before the bag empties.** Legal draws remove
one piece at a time (no refill can occur while a second piece remains), so the
bag's whole content is spent within `card` steps. -/
theorem exists_draw_within_card {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) :
    ∀ c n p, (bagAt initBag s n).card = c → p ∈ bagAt initBag s n →
      ∃ k < c, s (n + k) = p := by
  intro c
  induction c with
  | zero =>
    intro n p hc hp
    exact absurd (Finset.card_pos.mpr ⟨p, hp⟩) (by omega)
  | succ c ih =>
    intro n p hc hp
    by_cases hEq : s n = p
    · exact ⟨0, Nat.succ_pos c, by simpa using hEq⟩
    · have hsn : s n ∈ bagAt initBag s n := hl n
      have hpe : p ∈ (bagAt initBag s n).erase (s n) :=
        Finset.mem_erase.mpr ⟨fun h => hEq h.symm, hp⟩
      have hnext : bagAt initBag s (n + 1) = (bagAt initBag s n).erase (s n) := by
        change (bagAt initBag s n).draw (s n) = _
        unfold Bag.draw
        rw [if_neg (Finset.ne_empty_of_mem hpe)]
      have hcard : (bagAt initBag s (n + 1)).card = c := by
        rw [hnext, Finset.card_erase_of_mem hsn, hc]
        omega
      obtain ⟨k, hk, hks⟩ := ih (n + 1) p hcard (hnext ▸ hpe)
      exact ⟨k + 1, by omega, by rw [show n + (k + 1) = (n + 1) + k by omega]; exact hks⟩

/-- **The bag refills in exactly `card` steps.** Legal draws shrink the bag one
piece at a time; drawing the last piece triggers the refill. -/
theorem bagAt_add_card_eq_full {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) :
    ∀ c n, (bagAt initBag s n).card = c → bagAt initBag s (n + c) = Bag.full := by
  intro c
  induction c with
  | zero =>
    intro n hc
    exact absurd (Finset.card_pos.mpr ⟨s n, hl n⟩) (by omega)
  | succ c ih =>
    intro n hc
    have hsn : s n ∈ bagAt initBag s n := hl n
    by_cases hone : (bagAt initBag s n).card = 1
    · have hc0 : c = 0 := by omega
      have herase : (bagAt initBag s n).erase (s n) = ∅ :=
        Finset.card_eq_zero.mp (by rw [Finset.card_erase_of_mem hsn, hone])
      have hfull : bagAt initBag s (n + 1) = Bag.full := by
        change (bagAt initBag s n).draw (s n) = _
        unfold Bag.draw
        rw [if_pos herase]
      rw [hc0]
      exact hfull
    · have herase_ne : (bagAt initBag s n).erase (s n) ≠ ∅ := by
        intro h
        have := Finset.card_erase_of_mem hsn
        rw [h, Finset.card_empty] at this
        omega
      have hnext : bagAt initBag s (n + 1) = (bagAt initBag s n).erase (s n) := by
        change (bagAt initBag s n).draw (s n) = _
        unfold Bag.draw
        rw [if_neg herase_ne]
      have hcard : (bagAt initBag s (n + 1)).card = c := by
        rw [hnext, Finset.card_erase_of_mem hsn, hc]
        omega
      rw [show n + (c + 1) = (n + 1) + c by omega]
      exact ih (n + 1) hcard

/-- **The 7-bag is 13-syndetic in every piece.** Along any legal sequence,
every window of 13 consecutive draws contains every piece type. Tight: a
12-draw gap occurs when a piece is drawn first in one bag and last in the
next. -/
theorem every_piece_within_thirteen {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) (p : Piece) (n : ℕ) :
    ∃ k < 13, s (n + k) = p := by
  by_cases hp : p ∈ bagAt initBag s n
  · obtain ⟨k, hk, hks⟩ := exists_draw_within_card hl _ n p rfl hp
    have h7 : (bagAt initBag s n).card ≤ 7 := Bag.card_le_seven _
    exact ⟨k, by omega, hks⟩
  · have hc6 : (bagAt initBag s n).card ≤ 6 := by
      by_contra h
      have h7 : (bagAt initBag s n).card ≤ 7 := Bag.card_le_seven _
      have hc : (bagAt initBag s n).card = 7 := by omega
      exact hp (((Bag.card_eq_seven_iff_eq_full _).mp hc) ▸ Bag.mem_full p)
    have hfull := bagAt_add_card_eq_full hl _ n rfl
    have hpfull : p ∈ bagAt initBag s (n + (bagAt initBag s n).card) :=
      hfull ▸ Bag.mem_full p
    obtain ⟨k, hk, hks⟩ :=
      exists_draw_within_card hl _ (n + (bagAt initBag s n).card) p rfl hpfull
    have hcard7 : (bagAt initBag s (n + (bagAt initBag s n).card)).card = 7 := by
      rw [hfull]
      exact Bag.full_card
    rw [hcard7] at hk
    refine ⟨(bagAt initBag s n).card + k, by omega, ?_⟩
    rw [show n + ((bagAt initBag s n).card + k)
        = (n + (bagAt initBag s n).card) + k by omega]
    exact hks

/-- **The maximum I-drought is 12 placements.** Every window of 13 legal draws
contains an I. With `tetris_requires_I`, this is the worst-case cadence a
tetris-well architecture must survive: 12 consecutive non-I pieces — 48 cells
to house without closing the well. -/
theorem exists_I_within_thirteen {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) (n : ℕ) :
    ∃ k < 13, s (n + k) = Piece.I :=
  every_piece_within_thirteen hl Piece.I n

end BagCadence
end Tetris
