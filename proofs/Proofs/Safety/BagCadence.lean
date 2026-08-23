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

/-! ## Refill periodicity and the repetition floor

The refill instants are exactly seven apart (`bagAt_card_of_full`,
`not_full_of_full_close`), and two draws of the *same* piece must straddle a
refill (`exists_refill_between`) — after a piece is drawn it is simply absent
until the bag renews. Consequently three draws of one piece span at least
seven placements (`same_piece_three_apart`): a window of seven consecutive
draws holds **at most two** of any piece type.

With `tetris_requires_I` this caps burst clearing: back-to-back tetrises are
possible (an I last in one bag, first in the next), but a *third* tetris is at
least seven placements after the first — tetris bursts come in pairs, never
triples. -/

/-- From a full bag, the card counts down `7, 6, …` for six legal draws. -/
theorem bagAt_card_of_full {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) {r : ℕ}
    (hfull : bagAt initBag s r = Bag.full) :
    ∀ j ≤ 6, (bagAt initBag s (r + j)).card = 7 - j := by
  intro j
  induction j with
  | zero =>
    intro _
    rw [Nat.add_zero, hfull]
    simp
  | succ j ih =>
    intro hj6
    have hcard := ih (by omega)
    have hsn : s (r + j) ∈ bagAt initBag s (r + j) := hl (r + j)
    have herase_ne : (bagAt initBag s (r + j)).erase (s (r + j)) ≠ ∅ := by
      intro h
      have := Finset.card_erase_of_mem hsn
      rw [h, Finset.card_empty] at this
      omega
    have hnext : bagAt initBag s (r + j + 1)
        = (bagAt initBag s (r + j)).erase (s (r + j)) := by
      change (bagAt initBag s (r + j)).draw (s (r + j)) = _
      unfold Bag.draw
      rw [if_neg herase_ne]
    rw [show r + (j + 1) = r + j + 1 by omega, hnext,
      Finset.card_erase_of_mem hsn, hcard]
    omega

/-- **Refills are at least seven apart.** Between one refill and the next the
bag drains a full seven pieces. -/
theorem not_full_of_full_close {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) {r r' : ℕ}
    (hfull : bagAt initBag s r = Bag.full) (hlt : r < r') (hclose : r' < r + 7) :
    bagAt initBag s r' ≠ Bag.full := by
  intro hfull'
  have hj : r' = r + (r' - r) := by omega
  have hcard := bagAt_card_of_full hl hfull (r' - r) (by omega)
  rw [← hj, hfull'] at hcard
  have : (Bag.full : Bag).card = 7 := Bag.full_card
  omega

/-- **Two draws of the same piece straddle a refill.** Once drawn, a piece is
absent from the bag until the next renewal. -/
theorem exists_refill_between {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) {t t' : ℕ} {p : Piece}
    (h1 : s t = p) (h2 : s t' = p) (hlt : t < t') :
    ∃ r, t < r ∧ r ≤ t' ∧ bagAt initBag s r = Bag.full := by
  by_contra hno
  push Not at hno
  -- with no refill in (t, t'], the piece stays absent after its draw
  have key : ∀ m, t + 1 ≤ m → m ≤ t' → p ∉ bagAt initBag s m := by
    intro m
    induction m with
    | zero => omega
    | succ m ihm =>
      intro h1m hmt
      by_cases hbase : m = t
      · -- the step that drew p
        subst hbase
        have hsn : s m ∈ bagAt initBag s m := hl m
        have herase_ne : (bagAt initBag s m).erase (s m) ≠ ∅ := by
          intro h
          have hfull : bagAt initBag s (m + 1) = Bag.full := by
            change (bagAt initBag s m).draw (s m) = _
            unfold Bag.draw
            rw [if_pos h]
          exact hno (m + 1) (by omega) (by omega) hfull
        have hnext : bagAt initBag s (m + 1)
            = (bagAt initBag s m).erase (s m) := by
          change (bagAt initBag s m).draw (s m) = _
          unfold Bag.draw
          rw [if_neg herase_ne]
        rw [hnext, h1]
        intro hmem
        exact absurd rfl (Finset.mem_erase.mp hmem).1
      · -- a later step: p already absent, draws of other pieces keep it out
        have hprev : p ∉ bagAt initBag s m := ihm (by omega) (by omega)
        have hsn : s m ∈ bagAt initBag s m := hl m
        have herase_ne : (bagAt initBag s m).erase (s m) ≠ ∅ := by
          intro h
          have hfull : bagAt initBag s (m + 1) = Bag.full := by
            change (bagAt initBag s m).draw (s m) = _
            unfold Bag.draw
            rw [if_pos h]
          exact hno (m + 1) (by omega) (by omega) hfull
        have hnext : bagAt initBag s (m + 1)
            = (bagAt initBag s m).erase (s m) := by
          change (bagAt initBag s m).draw (s m) = _
          unfold Bag.draw
          rw [if_neg herase_ne]
        rw [hnext]
        intro hmem
        exact hprev (Finset.mem_erase.mp hmem).2
  have := key t' (by omega) le_rfl
  rw [← h2] at this
  exact this (hl t')

/-- **Three draws of one piece span at least seven placements.** Each
consecutive pair straddles a refill, refills are seven apart, and both refills
fit inside the span — so any window of seven consecutive draws holds at most
two of any piece type. With `tetris_requires_I`: tetris bursts come in pairs,
never triples. -/
theorem same_piece_three_apart {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) {t t' t'' : ℕ} {p : Piece}
    (h1 : s t = p) (h2 : s t' = p) (h3 : s t'' = p)
    (h12 : t < t') (h23 : t' < t'') :
    t + 7 ≤ t'' := by
  obtain ⟨r1, hr1t, hr1t', hr1full⟩ := exists_refill_between hl h1 h2 h12
  obtain ⟨r2, hr2t, hr2t', hr2full⟩ := exists_refill_between hl h2 h3 h23
  by_contra hcon
  exact not_full_of_full_close hl hr1full (by omega) (by omega) hr2full

/-- **Counted form: any seven consecutive draws hold at most two of one
piece.** Three indices in a 7-window would give three draws spanning at most
six placements, contradicting `same_piece_three_apart`. -/
theorem window_same_piece_card_le_two {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) (a : ℕ) (p : Piece) :
    ((Finset.range 7).filter (fun k => s (a + k) = p)).card ≤ 2 := by
  classical
  by_contra h
  push Not at h
  set S := (Finset.range 7).filter (fun k => s (a + k) = p) with hS
  have hprop : ∀ m ∈ S, m < 7 ∧ s (a + m) = p := by
    intro m hm
    obtain ⟨h1, h2⟩ := Finset.mem_filter.mp hm
    exact ⟨Finset.mem_range.mp h1, h2⟩
  have hne1 : S.Nonempty := Finset.card_pos.mp (by omega)
  set m1 := S.min' hne1 with hm1def
  have hm1 : m1 ∈ S := S.min'_mem hne1
  have hne2 : (S.erase m1).Nonempty := by
    refine Finset.card_pos.mp ?_
    rw [Finset.card_erase_of_mem hm1]
    omega
  set m2 := (S.erase m1).min' hne2 with hm2def
  have hm2e : m2 ∈ S.erase m1 := (S.erase m1).min'_mem hne2
  have hm2 : m2 ∈ S := (Finset.mem_erase.mp hm2e).2
  have h12 : m1 < m2 :=
    lt_of_le_of_ne (S.min'_le m2 hm2) (Ne.symm (Finset.mem_erase.mp hm2e).1)
  have hne3 : ((S.erase m1).erase m2).Nonempty := by
    refine Finset.card_pos.mp ?_
    rw [Finset.card_erase_of_mem hm2e, Finset.card_erase_of_mem hm1]
    omega
  set m3 := ((S.erase m1).erase m2).min' hne3 with hm3def
  have hm3e : m3 ∈ (S.erase m1).erase m2 := ((S.erase m1).erase m2).min'_mem hne3
  have hm3e1 : m3 ∈ S.erase m1 := (Finset.mem_erase.mp hm3e).2
  have hm3 : m3 ∈ S := (Finset.mem_erase.mp hm3e1).2
  have h23 : m2 < m3 :=
    lt_of_le_of_ne ((S.erase m1).min'_le m3 hm3e1)
      (Ne.symm (Finset.mem_erase.mp hm3e).1)
  obtain ⟨hlt1, hp1⟩ := hprop m1 hm1
  obtain ⟨hlt3, hp3⟩ := hprop m3 hm3
  have hspan := same_piece_three_apart hl hp1 (hprop m2 hm2).2 hp3
    (by omega) (by omega)
  omega

/-- **Every piece is drawn infinitely often** along any legal sequence — the
ω-form of syndeticity. -/
theorem every_piece_infinitely_often {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) (p : Piece) :
    {n : ℕ | s n = p}.Infinite := by
  refine Set.infinite_of_not_bddAbove ?_
  rintro ⟨N, hN⟩
  obtain ⟨k, -, hk⟩ := every_piece_within_thirteen hl p (N + 1)
  have hmem : (N + 1 + k) ∈ {n : ℕ | s n = p} := hk
  have := hN hmem
  omega

/-- The I piece — the only piece that can clear four rows — arrives infinitely
often, whatever the adversary does with the ordering. -/
theorem exists_I_infinitely_often {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) :
    {n : ℕ | s n = Piece.I}.Infinite :=
  every_piece_infinitely_often hl Piece.I

/-! ## Legal sequences exist -/

/-- The greedy bag stream: from a nonempty bag, always draw some member.
Draws keep the bag nonempty (`Bag.draw_nonempty`), so the stream never
stalls. -/
noncomputable def greedyBag (b0 : {b : Bag // b.Nonempty}) :
    ℕ → {b : Bag // b.Nonempty}
  | 0 => b0
  | n + 1 =>
      ⟨(greedyBag b0 n).1.draw (greedyBag b0 n).2.choose,
       Bag.draw_nonempty _ _⟩

/-- The piece drawn at each step of the greedy stream. -/
noncomputable def greedySeq (b0 : {b : Bag // b.Nonempty}) (n : ℕ) : Piece :=
  (greedyBag b0 n).2.choose

theorem bagAt_greedy (b0 : {b : Bag // b.Nonempty}) :
    ∀ n, bagAt b0.1 (greedySeq b0) n = (greedyBag b0 n).1
  | 0 => rfl
  | n + 1 => by
      change (bagAt b0.1 (greedySeq b0) n).draw (greedySeq b0 n)
        = ((greedyBag b0 (n + 1)) : {b : Bag // b.Nonempty}).1
      rw [bagAt_greedy b0 n]
      rfl

/-- **Legal sequences exist from every nonempty bag.** The 7-bag never
paints itself into a corner: greedy drawing is always legal. This is the
existence primitive that lets trace-based arguments run on arbitrary cycle
certificates. -/
theorem exists_legalSequenceFrom {b : Bag} (hb : b.Nonempty) :
    ∃ s : ℕ → Piece, LegalSequenceFrom b s := by
  refine ⟨greedySeq ⟨b, hb⟩, fun n => ?_⟩
  have h := bagAt_greedy ⟨b, hb⟩ n
  change greedySeq ⟨b, hb⟩ n ∈ bagAt b (greedySeq ⟨b, hb⟩) n
  rw [show bagAt b (greedySeq ⟨b, hb⟩) n
      = bagAt (⟨b, hb⟩ : {b : Bag // b.Nonempty}).1 (greedySeq ⟨b, hb⟩) n from rfl, h]
  exact (greedyBag ⟨b, hb⟩ n).2.choose_spec

/-! ## The balance theorem: five of each piece per cycle window

A 35-draw window whose bag state matches at both ends — the situation at every
closed-cycle period — deals **every piece exactly five times**. The proof needs
no uniqueness bookkeeping: each piece is hit at least once in each of five
disjoint sub-windows (the head that drains the current bag, four full bag
blocks, and the tail that must reproduce the end bag), and seven pieces at
five hits each already exhaust all `35` draws. -/

/-- A piece not drawn survives the step: erasing another piece keeps it, and a
refill restores everything. -/
theorem mem_bagAt_succ_of_ne {initBag : Bag} {s : ℕ → Piece} {m : ℕ} {p : Piece}
    (hp : p ∈ bagAt initBag s m) (hne : s m ≠ p) :
    p ∈ bagAt initBag s (m + 1) := by
  change p ∈ (bagAt initBag s m).draw (s m)
  unfold Bag.draw
  split
  · exact Bag.mem_full p
  · exact Finset.mem_erase.mpr ⟨fun h => hne h.symm, hp⟩

/-- A piece never drawn over an interval survives the whole interval. -/
theorem mem_bagAt_of_not_drawn {initBag : Bag} {s : ℕ → Piece} {p : Piece}
    {a : ℕ} (ha : p ∈ bagAt initBag s a) :
    ∀ b, a ≤ b → (∀ m, a ≤ m → m < b → s m ≠ p) → p ∈ bagAt initBag s b := by
  intro b
  induction b with
  | zero =>
    intro hab _
    exact Nat.le_zero.mp hab ▸ ha
  | succ b ih =>
    intro hab hnd
    rcases Nat.lt_or_ge a (b + 1) with hlt | hge
    · have hab' : a ≤ b := by omega
      exact mem_bagAt_succ_of_ne
        (ih hab' (fun m h1 h2 => hnd m h1 (by omega))) (hnd b hab' (by omega))
    · have : a = b + 1 := by omega
      exact this ▸ ha

/-- Refills iterate: from a full bag, the bag is full again after every
multiple of seven draws. -/
theorem bagAt_full_iterate {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) {r : ℕ}
    (hfull : bagAt initBag s r = Bag.full) :
    ∀ i, bagAt initBag s (r + 7 * i) = Bag.full := by
  intro i
  induction i with
  | zero => simpa using hfull
  | succ j ih =>
    have hcard : (bagAt initBag s (r + 7 * j)).card = 7 := by
      rw [ih]
      exact Bag.full_card
    have h := bagAt_add_card_eq_full hl _ (r + 7 * j) hcard
    rw [show r + 7 * (j + 1) = r + 7 * j + 7 by omega]
    exact h

section BalanceIrred

attribute [local irreducible] bagAt

/-- **The balance theorem.** A 35-draw window with equal bag states at both
ends deals every piece exactly five times. Head + tail jointly cover each
piece once (the head drains the current bag; the tail must carve the end bag
back out of a fresh one), each of the four full blocks covers each piece once,
and `7 × 5 = 35` leaves no slack. -/
theorem window_thirtyfive_balanced {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) {n : ℕ}
    (hbag : bagAt initBag s (n + 35) = bagAt initBag s n) (p : Piece) :
    ((Finset.range 35).filter (fun k => s (n + k) = p)).card = 5 := by
  classical
  obtain ⟨c, hc⟩ : ∃ c, (bagAt initBag s n).card = c := ⟨_, rfl⟩
  have hc1 : 1 ≤ c := hc ▸ Finset.card_pos.mpr ⟨s n, hl n⟩
  have hc7 : c ≤ 7 := hc ▸ Bag.card_le_seven _
  have hfull0 : bagAt initBag s (n + c) = Bag.full :=
    bagAt_add_card_eq_full hl c n hc
  have hfull : ∀ i, bagAt initBag s (n + c + 7 * i) = Bag.full :=
    bagAt_full_iterate hl hfull0
  -- lower bound: every piece is drawn at least five times
  have hlow : ∀ q : Piece,
      5 ≤ ((Finset.range 35).filter (fun k => s (n + k) = q)).card := by
    intro q
    -- one hit in each of the four full blocks
    have hblock : ∀ i, i < 4 → ∃ k, k < c + 7 * i + 7 ∧
        c + 7 * i ≤ k ∧ s (n + k) = q := by
      intro i _
      have hcard : (bagAt initBag s (n + c + 7 * i)).card = 7 := by
        rw [hfull i]
        exact Bag.full_card
      obtain ⟨k, hk7, hks⟩ := exists_draw_within_card hl 7 (n + c + 7 * i) q
        hcard (by rw [hfull i]; exact Bag.mem_full q)
      exact ⟨c + 7 * i + k, by omega, by omega,
        by rw [show n + (c + 7 * i + k) = n + c + 7 * i + k by omega]; exact hks⟩
    obtain ⟨k1, hk1b, hk1a, hs1⟩ := hblock 0 (by omega)
    obtain ⟨k2, hk2b, hk2a, hs2⟩ := hblock 1 (by omega)
    obtain ⟨k3, hk3b, hk3a, hs3⟩ := hblock 2 (by omega)
    obtain ⟨k4, hk4b, hk4a, hs4⟩ := hblock 3 (by omega)
    -- one hit in the head (piece in the current bag) or the tail (otherwise)
    have hhead_or_tail : ∃ k0, k0 < 35 ∧ (k0 < c ∨ c + 28 ≤ k0) ∧ s (n + k0) = q := by
      rcases Classical.em (q ∈ bagAt initBag s n) with hq | hq
      · obtain ⟨k, hk, hks⟩ := exists_draw_within_card hl c n q hc hq
        exact ⟨k, by omega, Or.inl hk, hks⟩
      · -- not in the end bag either; if never drawn in the tail it would be
        by_contra hno
        push Not at hno
        have hmem : q ∈ bagAt initBag s (n + c + 28) := by
          have h4 := hfull 4
          rw [show n + c + 7 * 4 = n + c + 28 by omega] at h4
          rw [h4]
          exact Bag.mem_full q
        have hsurv : q ∈ bagAt initBag s (n + 35) := by
          refine mem_bagAt_of_not_drawn hmem (n + 35) (by omega) ?_
          intro m h1 h2 hsm
          have hm35 : m - n < 35 := by omega
          have hmc : c + 28 ≤ m - n := by omega
          have hcon := hno (m - n) hm35 (Or.inr hmc)
          rw [show n + (m - n) = m by omega] at hcon
          exact hcon hsm
        rw [hbag] at hsurv
        exact hq hsurv
    obtain ⟨k0, hk0lt, hk0r, hs0⟩ := hhead_or_tail
    -- five pairwise-distinct hits
    have hsub : ({k0, k1, k2, k3, k4} : Finset ℕ)
        ⊆ (Finset.range 35).filter (fun k => s (n + k) = q) := by
      intro x hx
      simp only [Finset.mem_insert, Finset.mem_singleton] at hx
      rcases hx with rfl | rfl | rfl | rfl | rfl
      · exact Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (by omega), hs0⟩
      · exact Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (by omega), hs1⟩
      · exact Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (by omega), hs2⟩
      · exact Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (by omega), hs3⟩
      · exact Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (by omega), hs4⟩
    have hcard5 : ({k0, k1, k2, k3, k4} : Finset ℕ).card = 5 := by
      have h1 : k0 ∉ ({k1, k2, k3, k4} : Finset ℕ) := by
        simp only [Finset.mem_insert, Finset.mem_singleton]
        omega
      have h2 : k1 ∉ ({k2, k3, k4} : Finset ℕ) := by
        simp only [Finset.mem_insert, Finset.mem_singleton]
        omega
      have h3 : k2 ∉ ({k3, k4} : Finset ℕ) := by
        simp only [Finset.mem_insert, Finset.mem_singleton]
        omega
      have h4 : k3 ∉ ({k4} : Finset ℕ) := by
        simp only [Finset.mem_singleton]
        omega
      rw [Finset.card_insert_of_notMem h1, Finset.card_insert_of_notMem h2,
        Finset.card_insert_of_notMem h3, Finset.card_insert_of_notMem h4,
        Finset.card_singleton]
    calc 5 = ({k0, k1, k2, k3, k4} : Finset ℕ).card := hcard5.symm
      _ ≤ _ := Finset.card_le_card hsub
  -- total: the 35 draws distribute over the seven pieces
  have htotal : ∑ q ∈ (Finset.univ : Finset Piece),
      ((Finset.range 35).filter (fun k => s (n + k) = q)).card = 35 := by
    rw [← Finset.card_eq_sum_card_fiberwise
      (f := fun k => s (n + k)) (fun x _ => Finset.mem_univ _)]
    exact Finset.card_range 35
  -- equality: seven pieces at five hits each exhaust the window
  by_contra hne
  have h6 : 6 ≤ ((Finset.range 35).filter (fun k => s (n + k) = p)).card := by
    have := hlow p
    omega
  have hbig : (36 : ℕ) ≤ ∑ q ∈ (Finset.univ : Finset Piece),
      ((Finset.range 35).filter (fun k => s (n + k) = q)).card := by
    have hpt : ∀ q ∈ (Finset.univ : Finset Piece),
        (if q = p then 6 else 5)
          ≤ ((Finset.range 35).filter (fun k => s (n + k) = q)).card := by
      intro q _
      split_ifs with h
      · exact h ▸ h6
      · exact hlow q
    have hsum := Finset.sum_le_sum hpt
    have hsplit : ∀ q : Piece,
        (if q = p then 6 else 5) = 5 + (if q = p then 1 else 0) := by
      intro q
      split_ifs <;> rfl
    rw [Finset.sum_congr rfl (fun q _ => hsplit q), Finset.sum_add_distrib,
      Finset.sum_const, Finset.sum_ite_eq' Finset.univ p (fun _ => 1),
      if_pos (Finset.mem_univ p)] at hsum
    have h7 : (Finset.univ : Finset Piece).card = 7 := by decide
    rw [h7] at hsum
    omega
  omega

/-- Every full-bag block hits every piece: from a refill, each piece is drawn
within seven steps. -/
theorem exists_block_hit {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) {r : ℕ}
    (hfull : bagAt initBag s r = Bag.full) (p : Piece) :
    ∃ k, k < 7 ∧ s (r + k) = p := by
  have hcard : (bagAt initBag s r).card = 7 := by
    rw [hfull]
    exact Bag.full_card
  obtain ⟨k, hk, hks⟩ := exists_draw_within_card hl 7 r p hcard
    (by rw [hfull]; exact Bag.mem_full p)
  exact ⟨k, hk, hks⟩

/-- **Any 35 consecutive draws hold at least four of every piece** — no cycle
hypothesis needed: a 35-window always contains four complete bag blocks, and
each block deals every piece. (With matching bag states the count is exactly
five, `window_thirtyfive_balanced`.) -/
theorem window_thirtyfive_ge_four {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) (n : ℕ) (p : Piece) :
    4 ≤ ((Finset.range 35).filter (fun k => s (n + k) = p)).card := by
  classical
  obtain ⟨c, hc⟩ : ∃ c, (bagAt initBag s n).card = c := ⟨_, rfl⟩
  have hc1 : 1 ≤ c := hc ▸ Finset.card_pos.mpr ⟨s n, hl n⟩
  have hc7 : c ≤ 7 := hc ▸ Bag.card_le_seven _
  have hfull0 : bagAt initBag s (n + c) = Bag.full :=
    bagAt_add_card_eq_full hl c n hc
  have hfull : ∀ i, bagAt initBag s (n + c + 7 * i) = Bag.full :=
    bagAt_full_iterate hl hfull0
  have hblock : ∀ i, i < 4 → ∃ k, k < c + 7 * i + 7 ∧
      c + 7 * i ≤ k ∧ s (n + k) = p := by
    intro i _
    obtain ⟨k, hk7, hks⟩ := exists_block_hit hl (hfull i) p
    exact ⟨c + 7 * i + k, by omega, by omega,
      by rw [show n + (c + 7 * i + k) = n + c + 7 * i + k by omega]; exact hks⟩
  obtain ⟨k1, hk1b, hk1a, hs1⟩ := hblock 0 (by omega)
  obtain ⟨k2, hk2b, hk2a, hs2⟩ := hblock 1 (by omega)
  obtain ⟨k3, hk3b, hk3a, hs3⟩ := hblock 2 (by omega)
  obtain ⟨k4, hk4b, hk4a, hs4⟩ := hblock 3 (by omega)
  have hsub : ({k1, k2, k3, k4} : Finset ℕ)
      ⊆ (Finset.range 35).filter (fun k => s (n + k) = p) := by
    intro x hx
    simp only [Finset.mem_insert, Finset.mem_singleton] at hx
    rcases hx with rfl | rfl | rfl | rfl
    · exact Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (by omega), hs1⟩
    · exact Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (by omega), hs2⟩
    · exact Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (by omega), hs3⟩
    · exact Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (by omega), hs4⟩
  have hcard4 : ({k1, k2, k3, k4} : Finset ℕ).card = 4 := by
    have h2 : k1 ∉ ({k2, k3, k4} : Finset ℕ) := by
      simp only [Finset.mem_insert, Finset.mem_singleton]
      omega
    have h3 : k2 ∉ ({k3, k4} : Finset ℕ) := by
      simp only [Finset.mem_insert, Finset.mem_singleton]
      omega
    have h4 : k3 ∉ ({k4} : Finset ℕ) := by
      simp only [Finset.mem_singleton]
      omega
    rw [Finset.card_insert_of_notMem h2, Finset.card_insert_of_notMem h3,
      Finset.card_insert_of_notMem h4, Finset.card_singleton]
  calc 4 = ({k1, k2, k3, k4} : Finset ℕ).card := hcard4.symm
    _ ≤ _ := Finset.card_le_card hsub

end BalanceIrred

end BagCadence
end Tetris
