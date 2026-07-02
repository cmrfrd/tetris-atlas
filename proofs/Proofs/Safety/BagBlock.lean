import Proofs.Safety.Adversarial
import Proofs.Combinatorics.BagBurst

/-!
# Bag-block decomposition of a legal sequence — the 7-bag fairness bridge

This module proves the first **formal statement that the 7-bag randomizer is not
the Burgiel adversary**: every aligned 7-block of *any* legal piece sequence is a
permutation of the full bag (`isBagOrder_of_legal_block`). Consequently each
7-block contains **exactly one `I`, at most one `S`, at most one `Z`** — the
resource cap (via `Tetris.BagBurst`) that makes adversarial survival plausible
where Burgiel's unbounded-`S`/`Z` adversary forces a loss.

Concretely: starting from a full bag, seven legal draws deplete the bag one piece
at a time (`card 7,6,…,1`) and the seventh draw refills it, so the bag returns to
`full` at every multiple of 7 (`bagAt_full_block`), and the seven pieces drawn in
between are distinct (`legal_from_full_distinct`) — hence a bag order. This is the
combinatorial substrate for the bag-block (bounded-depth) reduction
`init_mem_safe_of_bag_invariant`.
-/

namespace Tetris

open Bag

/-- **Bag cardinality along a legal prefix from full.** Starting from a full bag,
`k ≤ 6` legal draws leave a bag of exactly `7 - k` pieces (no refill has happened
yet, since a refill only fires on the emptying draw). -/
theorem card_bagAt_full_le_six {t : ℕ → Piece} (ht : LegalSequenceFrom Bag.full t) :
    ∀ k, k ≤ 6 → (bagAt Bag.full t k).card = 7 - k := by
  intro k
  induction k with
  | zero => intro _; simp
  | succ n ih =>
    intro hle
    have hcard_n : (bagAt Bag.full t n).card = 7 - n := ih (by omega)
    have hge2 : 2 ≤ (bagAt Bag.full t n).card := by rw [hcard_n]; omega
    have hmem : t n ∈ bagAt Bag.full t n := ht n
    rw [bagAt_succ, Bag.card_draw_of_ge_two hmem hge2, hcard_n]
    omega

/-- **A bag of card `c + 1` returns to full after `c + 1` legal draws.** Fully
symbolic (both the starting bag and the sequence are generalized in the
induction) so no literal index ever forces `bagAt` to unfold — the first `c`
draws deplete the bag to a singleton, and the last draw refills it. -/
theorem bag_return_of_card :
    ∀ (c : ℕ) (b : Bag) (u : ℕ → Piece), b.card = c + 1 → LegalSequenceFrom b u →
      bagAt b u (c + 1) = Bag.full := by
  intro c
  induction c with
  | zero =>
    intro b u hcard hu
    have hmem : u 0 ∈ b := hu 0
    have h1 : b.card = 1 := by simpa using hcard
    change b.draw (u 0) = Bag.full
    exact (Bag.card_eq_seven_iff_eq_full _).mp (Bag.card_draw_of_singleton hmem h1)
  | succ e ih =>
    intro b u hcard hu
    have hmem : u 0 ∈ b := hu 0
    have hge2 : 2 ≤ b.card := by omega
    have hb1 : bagAt b u 1 = b.draw (u 0) := rfl
    have hcard' : (b.draw (u 0)).card = e + 1 := by
      rw [Bag.card_draw_of_ge_two hmem hge2]; omega
    have hsuffix : LegalSequenceFrom (b.draw (u 0)) (fun k => u (1 + k)) := by
      have hs := LegalSequenceFrom.suffix hu 1
      rwa [hb1] at hs
    have hstep := ih (b.draw (u 0)) (fun k => u (1 + k)) hcard' hsuffix
    have hidx : e + 1 + 1 = 1 + (e + 1) := by omega
    rw [hidx, bagAt_add, hb1]
    exact hstep

/-- **The bag refills to full after seven legal draws.** Instantiation of
`bag_return_of_card` at the full starting bag (`card 7`). -/
theorem bagAt_full_seven {t : ℕ → Piece} (ht : LegalSequenceFrom Bag.full t) :
    bagAt Bag.full t 7 = Bag.full := by
  have h := bag_return_of_card 6 Bag.full t (by simp) ht
  have h7 : (7 : ℕ) = 6 + 1 := by norm_num
  rw [h7]
  exact h

/-- **Bag is full at every block boundary.** For a legal sequence the bag state
returns to `Bag.full` at every multiple of 7. Backbone of the bag-block
reduction: it lets us treat the game one whole bag at a time. -/
theorem bagAt_full_block {s : ℕ → Piece} (hs : LegalSequence s) (m : ℕ) :
    bagAt Bag.full s (7 * m) = Bag.full := by
  induction m with
  | zero => simp
  | succ n ih =>
    have hsuffix :
        LegalSequenceFrom (bagAt Bag.full s (7 * n)) (fun k => s (7 * n + k)) :=
      LegalSequenceFrom.suffix hs (7 * n)
    rw [ih] at hsuffix
    have hstep :
        bagAt (bagAt Bag.full s (7 * n)) (fun k => s (7 * n + k)) 7 = Bag.full := by
      rw [ih]; exact bagAt_full_seven hsuffix
    have hmul : 7 * (n + 1) = 7 * n + 7 := by ring
    rw [hmul, bagAt_add]
    exact hstep

/-- **The bag shrinks across a non-emptying draw.** For `k ≤ 5` (bag card `≥ 2`),
the post-draw bag is a subset of the pre-draw bag. -/
theorem bagAt_full_subset_step {t : ℕ → Piece} (ht : LegalSequenceFrom Bag.full t)
    {k : ℕ} (hk : k ≤ 5) : bagAt Bag.full t (k + 1) ⊆ bagAt Bag.full t k := by
  have hcard : (bagAt Bag.full t k).card = 7 - k := card_bagAt_full_le_six ht k (by omega)
  have hge2 : 2 ≤ (bagAt Bag.full t k).card := by rw [hcard]; omega
  have hmem : t k ∈ bagAt Bag.full t k := ht k
  rw [bagAt_succ, Bag.draw_eq_erase_of_card_ge_two hmem hge2]
  exact Finset.erase_subset _ _

/-- **The drawn piece leaves the bag** (non-emptying draw, `k ≤ 5`). -/
theorem bagAt_full_notMem_step {t : ℕ → Piece} (ht : LegalSequenceFrom Bag.full t)
    {k : ℕ} (hk : k ≤ 5) : t k ∉ bagAt Bag.full t (k + 1) := by
  have hcard : (bagAt Bag.full t k).card = 7 - k := card_bagAt_full_le_six ht k (by omega)
  have hge2 : 2 ≤ (bagAt Bag.full t k).card := by rw [hcard]; omega
  have hmem : t k ∈ bagAt Bag.full t k := ht k
  rw [bagAt_succ]
  exact Bag.draw_notMem_of_card_ge_two hmem hge2

/-- **The bag is antitone up to the block boundary.** For `a ≤ a + d ≤ 6`, the bag
at step `a + d` is a subset of the bag at step `a` (the whole prefix is a chain of
erases, no refill yet). -/
theorem bagAt_full_subset_of_le {t : ℕ → Piece} (ht : LegalSequenceFrom Bag.full t) :
    ∀ a d, a + d ≤ 6 → bagAt Bag.full t (a + d) ⊆ bagAt Bag.full t a := by
  intro a d
  induction d with
  | zero => intro _; simp
  | succ e ih =>
    intro h
    have he : a + e ≤ 6 := by omega
    have hstep : bagAt Bag.full t (a + e + 1) ⊆ bagAt Bag.full t (a + e) :=
      bagAt_full_subset_step ht (by omega)
    have heq : a + (e + 1) = (a + e) + 1 := by ring
    rw [heq]
    exact Finset.Subset.trans hstep (ih he)

/-- **Distinct legal draws within a bag (ordered form).** If `i < j ≤ 6` then the
`i`-th and `j`-th legal draws from a full bag are different pieces: `t i` was
erased at step `i + 1` and the bag only shrinks up to the boundary, so `t i` is
gone by step `j`, while `t j` is present. -/
theorem legal_from_full_distinct_of_lt {t : ℕ → Piece}
    (ht : LegalSequenceFrom Bag.full t) {i j : ℕ} (hj : j ≤ 6) (hlt : i < j) :
    t i ≠ t j := by
  have hnotmem : t i ∉ bagAt Bag.full t (i + 1) := bagAt_full_notMem_step ht (by omega)
  have hsub : bagAt Bag.full t j ⊆ bagAt Bag.full t (i + 1) := by
    have heq : (i + 1) + (j - (i + 1)) = j := by omega
    have hh := bagAt_full_subset_of_le ht (i + 1) (j - (i + 1)) (by omega)
    rw [heq] at hh; exact hh
  have hmemj : t j ∈ bagAt Bag.full t j := ht j
  intro heq
  apply hnotmem
  apply hsub
  rw [heq]
  exact hmemj

/-- **Distinct legal draws within a bag.** For `i ≠ j` both `≤ 6`, the legal draws
`t i` and `t j` from a full bag are distinct pieces. -/
theorem legal_from_full_distinct {t : ℕ → Piece} (ht : LegalSequenceFrom Bag.full t)
    {i j : ℕ} (hi : i ≤ 6) (hj : j ≤ 6) (hij : i ≠ j) : t i ≠ t j := by
  rcases Nat.lt_or_ge i j with hlt | hge
  · exact legal_from_full_distinct_of_lt ht hj hlt
  · have hji : j < i := by omega
    exact (legal_from_full_distinct_of_lt ht hi hji).symm

/-- A nodup list of the seven pieces (length 7) is a permutation of the full bag.
A length-7 nodup list in the 7-element `Piece` type covers every piece, and two
nodup lists with the same element set are permutations. -/
theorem isBagOrder_of_nodup_length {l : List Piece} (hnd : l.Nodup) (hlen : l.length = 7) :
    Tetris.BagBurst.IsBagOrder l := by
  have hcard : l.toFinset.card = 7 := by rw [List.toFinset_card_of_nodup hnd, hlen]
  have huniv : l.toFinset = Finset.univ :=
    Finset.eq_univ_of_card _ (by rw [hcard, Piece.card_eq_seven])
  have hlmem : ∀ a : Piece, a ∈ l := fun a => by
    have hmem : a ∈ l.toFinset := by rw [huniv]; exact Finset.mem_univ a
    exact List.mem_toFinset.mp hmem
  have hbmem : ∀ a : Piece, a ∈ Tetris.BagBurst.bag := by decide
  have hbnd : Tetris.BagBurst.bag.Nodup := by decide
  unfold Tetris.BagBurst.IsBagOrder
  rw [List.perm_iff_count]
  intro a
  rw [hnd.count, hbnd.count, if_pos (hlmem a), if_pos (hbmem a)]

/-- The aligned `m`-th 7-block of a piece sequence: the seven pieces drawn while
consuming bag `m`. -/
def bagBlock (s : ℕ → Piece) (m : ℕ) : List Piece :=
  (List.range 7).map (fun k => s (7 * m + k))

/-- **The 7-bag fairness bridge.** Every aligned 7-block of any `LegalSequence` is
a permutation of the full bag. Via `Tetris.BagBurst` this immediately caps each
block: exactly one `I`, and at most one each of `S`/`Z` — the first formal
statement that the 7-bag randomizer is *not* Burgiel's unbounded-`S`/`Z`
adversary. -/
theorem isBagOrder_of_legal_block {s : ℕ → Piece} (hs : LegalSequence s) (m : ℕ) :
    Tetris.BagBurst.IsBagOrder (bagBlock s m) := by
  have hsuffix :
      LegalSequenceFrom (bagAt Bag.full s (7 * m)) (fun k => s (7 * m + k)) :=
    LegalSequenceFrom.suffix hs (7 * m)
  rw [bagAt_full_block hs m] at hsuffix
  have hlen : (bagBlock s m).length = 7 := by simp [bagBlock]
  have hnd : (bagBlock s m).Nodup := by
    unfold bagBlock
    refine List.Nodup.map_on ?_ (by decide : (List.range 7).Nodup)
    intro a ha b hb hab
    rw [List.mem_range] at ha hb
    by_contra hne
    exact legal_from_full_distinct hsuffix (by omega) (by omega) hne hab
  exact isBagOrder_of_nodup_length hnd hlen

/-- **Exactly one `I` per legal 7-block.** The guaranteed once-per-bag drain. -/
theorem countP_isI_legal_block {s : ℕ → Piece} (hs : LegalSequence s) (m : ℕ) :
    (bagBlock s m).countP Tetris.BagBurst.isI = 1 :=
  Tetris.BagBurst.countP_isI (isBagOrder_of_legal_block hs m)

/-- **At most two roughness pieces (`S`/`Z`) per legal 7-block.** The bounded
roughness inflow that the once-per-bag `I`-drain must amortize. -/
theorem countP_isSZ_legal_block {s : ℕ → Piece} (hs : LegalSequence s) (m : ℕ) :
    (bagBlock s m).countP Tetris.BagBurst.isSZ = 2 :=
  Tetris.BagBurst.countP_isSZ (isBagOrder_of_legal_block hs m)

end Tetris
