import Mathlib
import Proofs.Survival

/-!
# The adversarial game: solvers vs. legal piece sequences

This module formalizes the project's **central conjecture**:

> There exists a strategy that, starting from the empty board, never tops out
> against *any* legal 7-bag piece sequence.

The pieces in this formalization are:

* `LegalSequenceFrom initBag s` / `LegalSequence s` — infinite piece sequences
  consistent with the 7-bag (starting from `initBag` or from a full bag).
* `Solver cfg` — a strategy that, given the current state and the *announced
  next piece*, chooses a placement.
* `adversarialTrace cfg σ s g0 n` — the state after `n` moves of solver `σ`
  against piece sequence `s`, starting from `g0`.
* `SolvesTetris cfg σ` — `σ` never loses against any legal sequence from `init`.
* `TetrisSolvable` — `∃ σ, SolvesTetris StandardTetris σ`. The conjecture itself.

The conjecture is **stated**, not proved; this file gives the headline question
a precise Lean target, against which all subsequent survivability work aims.
-/

namespace Tetris

/-! ## Bag-legal piece sequences -/

/-- The bag state at step `n` of a piece sequence starting from `initBag`:
draw `s 0`, then `s 1`, etc. -/
def bagAt (initBag : Bag) (s : ℕ → Piece) : ℕ → Bag
  | 0     => initBag
  | n + 1 => (bagAt initBag s n).draw (s n)

@[simp] theorem bagAt_zero (initBag : Bag) (s : ℕ → Piece) :
    bagAt initBag s 0 = initBag := rfl

theorem bagAt_succ (initBag : Bag) (s : ℕ → Piece) (n : ℕ) :
    bagAt initBag s (n + 1) = (bagAt initBag s n).draw (s n) := rfl

/-- **Bag-state concatenation / time-shift identity.** Running `bagAt` for
`n + m` steps equals running for `n` steps to reach intermediate bag state
`bagAt initBag s n`, then running for `m` more steps from there under the
shifted piece sequence `fun k => s (n + k)`. The bag-side companion to
`adversarialTrace_add`, expressing that bag evolution is *associative under
sequence concatenation*. Useful for splicing arguments: legality past the
splice point reduces to legality of the suffix from the splice-point bag. -/
theorem bagAt_add (initBag : Bag) (s : ℕ → Piece) (n m : ℕ) :
    bagAt initBag s (n + m) =
      bagAt (bagAt initBag s n) (fun k => s (n + k)) m := by
  induction m with
  | zero => rfl
  | succ k ih =>
    show bagAt initBag s (n + (k + 1)) = _
    rw [show n + (k + 1) = (n + k) + 1 from rfl,
        bagAt_succ, bagAt_succ, ih]

/-- **`bagAt` prefix-locality.** The bag state at index `n` depends only on
`initBag` and the first `n` entries of the piece sequence. Hence two
sequences agreeing on `[0, n)` produce the same `bagAt … n`. Foundational
companion to `adversarialTrace_congr`: bag evolution is also a
non-anticipative function of the past. -/
theorem bagAt_congr (initBag : Bag) (s s' : ℕ → Piece) (n : ℕ)
    (h : ∀ k < n, s k = s' k) :
    bagAt initBag s n = bagAt initBag s' n := by
  induction n with
  | zero => rfl
  | succ k ih =>
    rw [bagAt_succ, bagAt_succ]
    have ihk : ∀ j < k, s j = s' j := fun j hj => h j (Nat.lt_succ_of_lt hj)
    rw [ih ihk, h k (Nat.lt_succ_self k)]

/-- A piece sequence is **bag-legal** starting from `initBag` if every drawn
piece is in the bag at the moment of the draw. -/
def LegalSequenceFrom (initBag : Bag) (s : ℕ → Piece) : Prop :=
  ∀ n, (bagAt initBag s n).canDraw (s n)

/-- The canonical legal-sequence predicate: bag-legal from a full bag (the
standard 7-bag start). -/
def LegalSequence (s : ℕ → Piece) : Prop := LegalSequenceFrom Bag.full s

/-- **Legality is prefix-local.** If `s` is `LegalSequenceFrom initBag` and
`s'` agrees with `s` on `[0, N)`, then `s'` satisfies the bag-legality
predicate (i.e. `(bagAt initBag s' k).canDraw (s' k)`) at every index `k < N`.
Combines `bagAt_congr` (Tick 1569) — bag state agrees on the prefix — with
pointwise sequence agreement to transfer `s`'s legality witness to `s'`. The
key ingredient for "extend a legal prefix" constructions where one builds a
new sequence agreeing with a known-legal one up to some cutoff and then
diverging. -/
theorem LegalSequenceFrom.prefix_congr
    {initBag : Bag} {s s' : ℕ → Piece} {N : ℕ}
    (heq : ∀ k < N, s k = s' k) (hl : LegalSequenceFrom initBag s) :
    ∀ k < N, (bagAt initBag s' k).canDraw (s' k) := by
  intro k hk
  have hbag : bagAt initBag s' k = bagAt initBag s k :=
    (bagAt_congr initBag s s' k (fun j hj => heq j (lt_trans hj hk))).symm
  rw [hbag, ← heq k hk]
  exact hl k

/-- **Legality splicing.** Given a legal prefix `s` (legal from `initBag`)
and a legal continuation `t` (legal from the prefix's terminal bag
`bagAt initBag s N`), the spliced sequence
`fun k => if k < N then s k else t (k - N)` is legal from `initBag`. This is
the formal construction for "append a known-legal continuation onto a
known-legal prefix at a chosen splice point" — the key tool for
cycle-entering sequence construction. Proof: split on `k < N`; in the
`<`-case use `bagAt_congr` to match the prefix bag and `hs k`; in the `≥`-case
factor via `bagAt_add` at `k = N + (k - N)`, match the splice-point bag via
`bagAt_congr` at `N`, observe that the shifted spliced sequence equals `t`
(by `Nat.add_sub_cancel_left` after the else-branch), and conclude via `ht`. -/
theorem LegalSequenceFrom.splice
    {initBag : Bag} {s t : ℕ → Piece} {N : ℕ}
    (hs : LegalSequenceFrom initBag s)
    (ht : LegalSequenceFrom (bagAt initBag s N) t) :
    LegalSequenceFrom initBag (fun k => if k < N then s k else t (k - N)) := by
  set sp : ℕ → Piece := fun k => if k < N then s k else t (k - N) with hsp_def
  intro k
  by_cases hk : k < N
  · have hspk : sp k = s k := by simp [hsp_def, hk]
    have heq : ∀ j < k, sp j = s j := fun j hj => by
      have : j < N := lt_of_lt_of_le hj (le_of_lt hk)
      simp [hsp_def, this]
    have hbag : bagAt initBag sp k = bagAt initBag s k :=
      bagAt_congr initBag sp s k heq
    rw [hspk, hbag]
    exact hs k
  · have hkN : N ≤ k := Nat.not_lt.mp hk
    set j := k - N with hjdef
    have hkj : k = N + j := by omega
    have hsplice_lt : ∀ m < N, sp m = s m := fun m hm => by simp [hsp_def, hm]
    have hbagN : bagAt initBag sp N = bagAt initBag s N :=
      bagAt_congr initBag sp s N hsplice_lt
    have hshift : ∀ m, sp (N + m) = t m := fun m => by
      have h1 : ¬ N + m < N := Nat.not_lt.mpr (Nat.le_add_right N m)
      change (if N + m < N then s (N + m) else t (N + m - N)) = t m
      rw [if_neg h1, Nat.add_sub_cancel_left]
    have hshift_funext : (fun k' => sp (N + k')) = t := funext hshift
    rw [hkj, bagAt_add, hbagN, hshift_funext, hshift j]
    exact ht j

/-- **Legality suffix-decomposition.** Complement of `LegalSequenceFrom.splice`:
*any* legal sequence's `N`-shifted suffix is itself legal from the
mid-point bag `bagAt initBag sp N`. Proof: at index `k`, rewrite via
`bagAt_add` to recover `bagAt initBag sp (N + k)`, then apply the original
legality witness at `N + k`. Pairs with Tick 1572 to give a clean
"split-then-rejoin" calculus on legal sequences. -/
theorem LegalSequenceFrom.suffix
    {initBag : Bag} {sp : ℕ → Piece} (h : LegalSequenceFrom initBag sp)
    (N : ℕ) :
    LegalSequenceFrom (bagAt initBag sp N) (fun k => sp (N + k)) := by
  intro k
  rw [← bagAt_add]
  exact h (N + k)

/-- **Splice/suffix inverse identity.** Splicing a sequence `s` with the
suffix `fun j => sp (N + j)` at index `N` reconstructs `sp` exactly, provided
`s` and `sp` agree on the prefix `[0, N)`. The function-level identity
witnessing that `splice ∘ suffix` is the identity (on sequences that begin
with the spliced prefix). Together with Tick 1578 this tightens the
"split-then-rejoin" calculus into a full isomorphism. -/
theorem splice_suffix_eq (s sp : ℕ → Piece) (N : ℕ)
    (h : ∀ k < N, s k = sp k) :
    (fun k => if k < N then s k else sp (N + (k - N))) = sp := by
  funext k
  by_cases hk : k < N
  · rw [if_pos hk]; exact h k hk
  · have : N + (k - N) = k := by omega
    rw [if_neg hk, this]


/-- A `LegalSequence` is `LegalSequenceFrom Bag.full`. -/
theorem LegalSequence.canDraw {s : ℕ → Piece} (h : LegalSequence s) (n : ℕ) :
    (bagAt Bag.full s n).canDraw (s n) := h n

/-- Dot accessor for `LegalSequenceFrom`. -/
theorem LegalSequenceFrom.canDraw {initBag : Bag} {s : ℕ → Piece}
    (h : LegalSequenceFrom initBag s) (n : ℕ) :
    (bagAt initBag s n).canDraw (s n) := h n

/-- Zero-step accessor for `LegalSequenceFrom`: the first draw is
legal from `initBag` (the first element of the sequence). -/
theorem LegalSequenceFrom.canDraw_zero {initBag : Bag} {s : ℕ → Piece}
    (h : LegalSequenceFrom initBag s) : initBag.canDraw (s 0) := h 0

/-- `LegalSequenceFrom Bag.full ↔ LegalSequence`. -/
@[simp] theorem legalSequenceFrom_full_iff (s : ℕ → Piece) :
    LegalSequenceFrom Bag.full s ↔ LegalSequence s := Iff.rfl

/-- A `LegalSequenceFrom Bag.full` is exactly a `LegalSequence`. -/
theorem LegalSequenceFrom.toLegalSequence {s : ℕ → Piece}
    (h : LegalSequenceFrom Bag.full s) : LegalSequence s := h

/-- A `LegalSequence` is a `LegalSequenceFrom Bag.full`. -/
theorem LegalSequence.toLegalSequenceFrom_full {s : ℕ → Piece}
    (h : LegalSequence s) : LegalSequenceFrom Bag.full s := h

/-- The zero-th step of a `LegalSequence` can always draw from
the initial full bag (the canDraw_full base case). -/
theorem LegalSequence.canDraw_zero {s : ℕ → Piece} (h : LegalSequence s) :
    (bagAt Bag.full s 0).canDraw (s 0) := h 0

/-- For any `LegalSequence`, the bag at step `0` is `Bag.full`. -/
theorem LegalSequence.bagAt_zero_eq_full (s : ℕ → Piece) :
    bagAt Bag.full s 0 = Bag.full :=
  Tetris.bagAt_zero Bag.full s

/-- `bagAt_succ` specialised to `LegalSequence`'s initial bag. -/
theorem LegalSequence.bagAt_succ_eq (s : ℕ → Piece) (n : ℕ) :
    bagAt Bag.full s (n + 1) = (bagAt Bag.full s n).draw (s n) :=
  Tetris.bagAt_succ Bag.full s n

/-- Parametric `bagAt_succ` dot accessor. -/
theorem LegalSequenceFrom.bagAt_succ_eq (initBag : Bag) (s : ℕ → Piece)
    (n : ℕ) : bagAt initBag s (n + 1) = (bagAt initBag s n).draw (s n) :=
  Tetris.bagAt_succ initBag s n

/-- Parametric `bagAt_zero` dot accessor. -/
theorem LegalSequenceFrom.bagAt_zero_eq (initBag : Bag) (s : ℕ → Piece) :
    bagAt initBag s 0 = initBag :=
  Tetris.bagAt_zero initBag s

/-- Every bag in a `LegalSequenceFrom` is nonempty (since we can
draw from it). -/
theorem LegalSequenceFrom.bagAt_nonempty {initBag : Bag} {s : ℕ → Piece}
    (h : LegalSequenceFrom initBag s) (n : ℕ) :
    (bagAt initBag s n).Nonempty :=
  ⟨s n, h n⟩

/-- `LegalSequence` analog: every bag is nonempty. -/
theorem LegalSequence.bagAt_nonempty {s : ℕ → Piece} (h : LegalSequence s)
    (n : ℕ) : (bagAt Bag.full s n).Nonempty :=
  LegalSequenceFrom.bagAt_nonempty h n

/-- `LegalSequenceFrom.bagAt` is `≠ ∅` at every step. -/
theorem LegalSequenceFrom.bagAt_ne_empty {initBag : Bag} {s : ℕ → Piece}
    (h : LegalSequenceFrom initBag s) (n : ℕ) : bagAt initBag s n ≠ ∅ :=
  (h.bagAt_nonempty n).ne_empty

/-- `LegalSequence` analog. -/
theorem LegalSequence.bagAt_ne_empty {s : ℕ → Piece} (h : LegalSequence s)
    (n : ℕ) : bagAt Bag.full s n ≠ ∅ :=
  LegalSequenceFrom.bagAt_ne_empty h n

/-- The next bag is a subset of the current bag, or it has been refilled to full. -/
theorem bagAt_succ_subset_or_eq_full (initBag : Bag) (s : ℕ → Piece) (n : ℕ) :
    bagAt initBag s (n + 1) ⊆ bagAt initBag s n ∨
      bagAt initBag s (n + 1) = Bag.full := by
  rw [bagAt_succ]; exact Bag.draw_subset_or_eq_full (bagAt initBag s n) (s n)

/-- The next bag equals erase of the current bag, or full. -/
theorem bagAt_succ_eq_erase_or_full (initBag : Bag) (s : ℕ → Piece) (n : ℕ) :
    bagAt initBag s (n + 1) = (bagAt initBag s n).erase (s n) ∨
      bagAt initBag s (n + 1) = Bag.full := by
  rw [bagAt_succ]; exact Bag.draw_eq_erase_or_full (bagAt initBag s n) (s n)

/-- `bagAt initBag s n ⊆ Bag.full` for any `n` (alias of the existing
non-dot accessor). Useful for chained `subset` rewriting. -/
theorem bagAt_subset_full (initBag : Bag) (s : ℕ → Piece) (n : ℕ) :
    bagAt initBag s n ⊆ Bag.full :=
  Bag.subset_full _

/-- The bag card at any step is at most 7 (no `LegalSequenceFrom` premise). -/
theorem bagAt_card_le_seven (initBag : Bag) (s : ℕ → Piece) (n : ℕ) :
    (bagAt initBag s n).card ≤ 7 :=
  Bag.card_le_seven _

/-- The bag card at any step is strictly less than 8. -/
theorem bagAt_card_lt_eight (initBag : Bag) (s : ℕ → Piece) (n : ℕ) :
    (bagAt initBag s n).card < 8 :=
  Nat.lt_succ_of_le (bagAt_card_le_seven initBag s n)

/-- For a `LegalSequenceFrom`, the next bag card is either `predecessor`
of the current or refills to 7. -/
theorem LegalSequenceFrom.bagAt_succ_card_le_or_eq_seven
    {initBag : Bag} {s : ℕ → Piece}
    (_h : LegalSequenceFrom initBag s) (n : ℕ) :
    (bagAt initBag s (n + 1)).card ≤ (bagAt initBag s n).card ∨
      (bagAt initBag s (n + 1)).card = 7 := by
  rw [bagAt_succ]; exact Bag.draw_card_le_card_or_eq_seven _ _

/-- Universal (no `LegalSequenceFrom` premise): the next bag card is
either at most the current or refills to 7. -/
theorem bagAt_succ_card_le_or_eq_seven (initBag : Bag) (s : ℕ → Piece) (n : ℕ) :
    (bagAt initBag s (n + 1)).card ≤ (bagAt initBag s n).card ∨
      (bagAt initBag s (n + 1)).card = 7 := by
  rw [bagAt_succ]; exact Bag.draw_card_le_card_or_eq_seven _ _

/-- Universal: the next bag card is positive (`draw` always yields a
nonempty bag). -/
theorem bagAt_succ_card_pos (initBag : Bag) (s : ℕ → Piece) (n : ℕ) :
    0 < (bagAt initBag s (n + 1)).card := by
  rw [bagAt_succ]; exact Bag.draw_card_pos _ _

/-- Universal: the next bag is nonempty. -/
theorem bagAt_succ_nonempty (initBag : Bag) (s : ℕ → Piece) (n : ℕ) :
    (bagAt initBag s (n + 1)).Nonempty := by
  rw [bagAt_succ]; exact Bag.draw_nonempty _ _

/-- Universal: the next bag is not empty. -/
theorem bagAt_succ_ne_empty (initBag : Bag) (s : ℕ → Piece) (n : ℕ) :
    bagAt initBag s (n + 1) ≠ ∅ :=
  (bagAt_succ_nonempty initBag s n).ne_empty

/-- Universal: the next bag has card `≥ 1`. -/
theorem bagAt_succ_card_ge_one (initBag : Bag) (s : ℕ → Piece) (n : ℕ) :
    1 ≤ (bagAt initBag s (n + 1)).card :=
  bagAt_succ_card_pos initBag s n

/-- Universal: the next bag has nonzero card. -/
theorem bagAt_succ_card_ne_zero (initBag : Bag) (s : ℕ → Piece) (n : ℕ) :
    (bagAt initBag s (n + 1)).card ≠ 0 :=
  Nat.ne_of_gt (bagAt_succ_card_pos initBag s n)

/-- Universal: the next bag has card in `[1, 7]`. -/
theorem bagAt_succ_card_interval (initBag : Bag) (s : ℕ → Piece) (n : ℕ) :
    1 ≤ (bagAt initBag s (n + 1)).card ∧
      (bagAt initBag s (n + 1)).card ≤ 7 :=
  ⟨bagAt_succ_card_ge_one initBag s n, bagAt_card_le_seven initBag s (n + 1)⟩

/-- If `initBag.Nonempty`, then `bagAt initBag s n` has positive card for any `n`. -/
theorem bagAt_card_pos_of_initBag_nonempty {initBag : Bag} (h : initBag.Nonempty)
    (s : ℕ → Piece) (n : ℕ) : 0 < (bagAt initBag s n).card := by
  rcases n with _ | n
  · rw [bagAt_zero]; exact Finset.card_pos.mpr h
  · exact bagAt_succ_card_pos initBag s n

/-- If `initBag.Nonempty`, then `bagAt initBag s n` is nonempty for any `n`. -/
theorem bagAt_nonempty_of_initBag_nonempty {initBag : Bag} (h : initBag.Nonempty)
    (s : ℕ → Piece) (n : ℕ) : (bagAt initBag s n).Nonempty :=
  Finset.card_pos.mp (bagAt_card_pos_of_initBag_nonempty h s n)

/-- If `initBag.Nonempty`, then `bagAt initBag s n ≠ ∅` for any `n`. -/
theorem bagAt_ne_empty_of_initBag_nonempty {initBag : Bag} (h : initBag.Nonempty)
    (s : ℕ → Piece) (n : ℕ) : bagAt initBag s n ≠ ∅ :=
  (bagAt_nonempty_of_initBag_nonempty h s n).ne_empty

/-- For the `Bag.full` start, `bagAt` is always nonempty (specialisation). -/
theorem bagAt_full_nonempty (s : ℕ → Piece) (n : ℕ) :
    (bagAt Bag.full s n).Nonempty :=
  bagAt_nonempty_of_initBag_nonempty Bag.full_nonempty s n

/-- For the `Bag.full` start, `bagAt` has positive card. -/
theorem bagAt_full_card_pos (s : ℕ → Piece) (n : ℕ) :
    0 < (bagAt Bag.full s n).card :=
  bagAt_card_pos_of_initBag_nonempty Bag.full_nonempty s n

/-- For the `Bag.full` start, `bagAt` is never empty. -/
theorem bagAt_full_ne_empty (s : ℕ → Piece) (n : ℕ) :
    bagAt Bag.full s n ≠ ∅ :=
  bagAt_ne_empty_of_initBag_nonempty Bag.full_nonempty s n

/-- For the `Bag.full` start, `bagAt` has card `≥ 1`. -/
theorem bagAt_full_card_ge_one (s : ℕ → Piece) (n : ℕ) :
    1 ≤ (bagAt Bag.full s n).card :=
  bagAt_full_card_pos s n

/-- For the `Bag.full` start, `bagAt` has nonzero card. -/
theorem bagAt_full_card_ne_zero (s : ℕ → Piece) (n : ℕ) :
    (bagAt Bag.full s n).card ≠ 0 :=
  Nat.ne_of_gt (bagAt_full_card_pos s n)

/-- For the `Bag.full` start, `bagAt` has card in `[1, 7]`. -/
theorem bagAt_full_card_interval (s : ℕ → Piece) (n : ℕ) :
    1 ≤ (bagAt Bag.full s n).card ∧ (bagAt Bag.full s n).card ≤ 7 :=
  ⟨bagAt_full_card_ge_one s n, bagAt_card_le_seven Bag.full s n⟩

/-- For the `Bag.full` start, `bagAt` ⊆ `Bag.full`. -/
theorem bagAt_full_subset_full (s : ℕ → Piece) (n : ℕ) :
    bagAt Bag.full s n ⊆ Bag.full :=
  bagAt_subset_full Bag.full s n

/-- For the `Bag.full` start, `bagAt` has card `< 8`. -/
theorem bagAt_full_card_lt_eight (s : ℕ → Piece) (n : ℕ) :
    (bagAt Bag.full s n).card < 8 :=
  bagAt_card_lt_eight Bag.full s n

/-- For the `Bag.full` start at step 0, the bag has card exactly 7. -/
theorem bagAt_full_zero_card (s : ℕ → Piece) :
    (bagAt Bag.full s 0).card = 7 := by
  rw [bagAt_zero]; exact Bag.full_card

/-- For the `Bag.full` start at step 0, the bag equals `Bag.full`. -/
theorem bagAt_full_zero (s : ℕ → Piece) :
    bagAt Bag.full s 0 = Bag.full :=
  bagAt_zero Bag.full s

/-- For `Bag.full` start at step 0, `s 0 ∈ bagAt _ 0` (any piece is drawable). -/
theorem bagAt_full_zero_mem (s : ℕ → Piece) :
    s 0 ∈ bagAt Bag.full s 0 := by
  rw [bagAt_full_zero]; exact Bag.mem_full _

/-- For `Bag.full` start at step 0, `bagAt _ 0` can-draw `s 0`. -/
theorem bagAt_full_zero_canDraw (s : ℕ → Piece) :
    (bagAt Bag.full s 0).canDraw (s 0) :=
  bagAt_full_zero_mem s

/-- For `Bag.full` start, the next bag is subset of current or full. -/
theorem bagAt_full_succ_subset_or_eq_full (s : ℕ → Piece) (n : ℕ) :
    bagAt Bag.full s (n + 1) ⊆ bagAt Bag.full s n ∨
      bagAt Bag.full s (n + 1) = Bag.full :=
  bagAt_succ_subset_or_eq_full Bag.full s n

/-- For `Bag.full` start, the next bag equals erase or full. -/
theorem bagAt_full_succ_eq_erase_or_full (s : ℕ → Piece) (n : ℕ) :
    bagAt Bag.full s (n + 1) = (bagAt Bag.full s n).erase (s n) ∨
      bagAt Bag.full s (n + 1) = Bag.full :=
  bagAt_succ_eq_erase_or_full Bag.full s n

/-- For `Bag.full` start, the next bag card ≤ current or = 7. -/
theorem bagAt_full_succ_card_le_or_eq_seven (s : ℕ → Piece) (n : ℕ) :
    (bagAt Bag.full s (n + 1)).card ≤ (bagAt Bag.full s n).card ∨
      (bagAt Bag.full s (n + 1)).card = 7 :=
  bagAt_succ_card_le_or_eq_seven Bag.full s n

/-- For `Bag.full` start, the next bag has positive card. -/
theorem bagAt_full_succ_card_pos (s : ℕ → Piece) (n : ℕ) :
    0 < (bagAt Bag.full s (n + 1)).card :=
  bagAt_succ_card_pos Bag.full s n

/-- For `Bag.full` start, the next bag is nonempty. -/
theorem bagAt_full_succ_nonempty (s : ℕ → Piece) (n : ℕ) :
    (bagAt Bag.full s (n + 1)).Nonempty :=
  bagAt_succ_nonempty Bag.full s n

/-- For `Bag.full` start, the next bag is not empty. -/
theorem bagAt_full_succ_ne_empty (s : ℕ → Piece) (n : ℕ) :
    bagAt Bag.full s (n + 1) ≠ ∅ :=
  bagAt_succ_ne_empty Bag.full s n

/-- For `Bag.full` start, the next bag has card `≤ 7`. -/
theorem bagAt_full_succ_card_le_seven (s : ℕ → Piece) (n : ℕ) :
    (bagAt Bag.full s (n + 1)).card ≤ 7 :=
  Bag.card_le_seven _

/-- For `Bag.full` start, the next bag has card in `[1, 7]`. -/
theorem bagAt_full_succ_card_interval (s : ℕ → Piece) (n : ℕ) :
    1 ≤ (bagAt Bag.full s (n + 1)).card ∧
      (bagAt Bag.full s (n + 1)).card ≤ 7 :=
  ⟨bagAt_full_succ_card_pos s n, bagAt_full_succ_card_le_seven s n⟩

/-- bag.card ≥ 1 over the lifetime of a `LegalSequenceFrom`. -/
theorem LegalSequenceFrom.bagAt_card_pos {initBag : Bag} {s : ℕ → Piece}
    (h : LegalSequenceFrom initBag s) (n : ℕ) :
    0 < (bagAt initBag s n).card :=
  Finset.card_pos.mpr (h.bagAt_nonempty n)

/-- bag.card ≤ 7 over the lifetime of any (parametric) sequence. -/
theorem LegalSequenceFrom.bagAt_card_le_seven (initBag : Bag) (s : ℕ → Piece)
    (n : ℕ) : (bagAt initBag s n).card ≤ 7 :=
  Bag.card_le_seven _

/-- `LegalSequence` bag.card positivity. -/
theorem LegalSequence.bagAt_card_pos {s : ℕ → Piece} (h : LegalSequence s)
    (n : ℕ) : 0 < (bagAt Bag.full s n).card :=
  LegalSequenceFrom.bagAt_card_pos h n

/-- `LegalSequence` bag.card upper bound. -/
theorem LegalSequence.bagAt_card_le_seven (s : ℕ → Piece) (n : ℕ) :
    (bagAt Bag.full s n).card ≤ 7 :=
  Bag.card_le_seven _

/-- Bundled `[1, 7]` bag-card interval at every step. -/
theorem LegalSequenceFrom.bagAt_card_interval {initBag : Bag} {s : ℕ → Piece}
    (h : LegalSequenceFrom initBag s) (n : ℕ) :
    1 ≤ (bagAt initBag s n).card ∧ (bagAt initBag s n).card ≤ 7 :=
  ⟨h.bagAt_card_pos n, Bag.card_le_seven _⟩

/-- Canonical analog. -/
theorem LegalSequence.bagAt_card_interval {s : ℕ → Piece} (h : LegalSequence s)
    (n : ℕ) :
    1 ≤ (bagAt Bag.full s n).card ∧ (bagAt Bag.full s n).card ≤ 7 :=
  ⟨h.bagAt_card_pos n, Bag.card_le_seven _⟩

/-- Every bag at any step is a subset of `Bag.full`. -/
theorem LegalSequenceFrom.bagAt_subset_full (initBag : Bag) (s : ℕ → Piece)
    (n : ℕ) : bagAt initBag s n ⊆ Bag.full :=
  Bag.subset_full _

/-- Canonical analog. -/
theorem LegalSequence.bagAt_subset_full (s : ℕ → Piece) (n : ℕ) :
    bagAt Bag.full s n ⊆ Bag.full :=
  Bag.subset_full _

/-- The bag at step `k` only depends on the first `k` values of the piece
sequence. -/
theorem bagAt_eq_of_eq_below (initBag : Bag) (s s' : ℕ → Piece) (n : ℕ)
    (h : ∀ i < n, s i = s' i) :
    ∀ k ≤ n, bagAt initBag s k = bagAt initBag s' k := by
  intro k hk
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [bagAt_succ, bagAt_succ, ih (by omega), h k (by omega)]

/-- Auxiliary: the (bag, nonempty-proof) pair at each step of a canonical
descent that always picks `Bag.someElement` of the current bag. -/
noncomputable def legalAux (initBag : Bag) (hne : initBag.Nonempty) :
    ℕ → Σ' (b : Bag), b.Nonempty :=
  Nat.rec ⟨initBag, hne⟩
    (fun _ x => ⟨x.1.draw (Bag.someElement x.1 x.2), Bag.draw_nonempty _ _⟩)

/-- The canonical legal sequence from `initBag`: at each step, picks
`Bag.someElement` of the current bag. -/
noncomputable def someLegalFrom (initBag : Bag) (hne : initBag.Nonempty)
    (n : ℕ) : Piece :=
  Bag.someElement (legalAux initBag hne n).1 (legalAux initBag hne n).2

/-- The bag carried by `legalAux n` agrees with `bagAt … n` for the canonical
sequence. -/
theorem legalAux_bag (initBag : Bag) (hne : initBag.Nonempty) (n : ℕ) :
    (legalAux initBag hne n).1 = bagAt initBag (someLegalFrom initBag hne) n := by
  induction n with
  | zero => rfl
  | succ k ih =>
    change (legalAux initBag hne k).1.draw
         (Bag.someElement (legalAux initBag hne k).1 (legalAux initBag hne k).2) =
         bagAt initBag (someLegalFrom initBag hne) (k+1)
    rw [bagAt_succ, ← ih]
    rfl

/-- The canonical sequence is bag-legal: each chosen piece is in the bag
at that step. -/
theorem someLegalFrom_legal (initBag : Bag) (hne : initBag.Nonempty) :
    LegalSequenceFrom initBag (someLegalFrom initBag hne) := by
  intro n
  change someLegalFrom initBag hne n ∈ bagAt initBag (someLegalFrom initBag hne) n
  rw [← legalAux_bag]
  exact Bag.someElement_mem _ _

/-- **Legal sequences exist** from any nonempty starting bag. -/
theorem exists_legalSequenceFrom (initBag : Bag) (hne : initBag.Nonempty) :
    ∃ s : ℕ → Piece, LegalSequenceFrom initBag s :=
  ⟨someLegalFrom initBag hne, someLegalFrom_legal initBag hne⟩

/-- **Legal sequences exist** from `Bag.full` (the standard 7-bag start). -/
theorem exists_legalSequence : ∃ s : ℕ → Piece, LegalSequence s :=
  exists_legalSequenceFrom Bag.full Bag.full_nonempty

/-- The `Classical.choose` extraction: a chosen legal sequence
from `Bag.full`. -/
noncomputable def someLegalSequence : ℕ → Piece :=
  exists_legalSequence.choose

/-- Spec: `someLegalSequence` is a `LegalSequence`. -/
theorem someLegalSequence_legal : LegalSequence someLegalSequence :=
  exists_legalSequence.choose_spec



/-! ## Solvers and adversarial play -/

/-- A **solver**: given the current state and the announced next piece, choose
a placement (rotation and column) for that piece. -/
abbrev Solver (_cfg : GameConfig) := GameState → Piece → Placement

/-- An **Atlas** is a *partial* function from `(state, piece)` to placement —
i.e. the formal counterpart to the project mission's "complete lookup table"
artifact described in `AGENTS.md`. Unlike a `Solver` (which is total and
always commits to an answer), an `Atlas` may return `none` for states or
pieces outside its certified domain. The Atlas is the proof artifact: a
*closed* atlas (see `Atlas.IsClosedOn`, defined below `adversarialStep`)
certifies infinite play on its domain. -/
abbrev Atlas (_cfg : GameConfig) := GameState → Piece → Option Placement

/-- Lift a total `Solver` to an `Atlas` (always returns `some`). The trivial
embedding `Solver ↪ Atlas` lets every solver-based theorem transfer to the
atlas language by composition with `Solver.toAtlas`. -/
def Solver.toAtlas {cfg : GameConfig} (σ : Solver cfg) : Atlas cfg :=
  fun g p => some (σ g p)

/-- Materialise an `Atlas` into a total `Solver` via `Option.getD`. When the
atlas returns `some pl`, the solver returns `pl`; out-of-domain queries
fall back to an arbitrary placement. Used to thread an M4 atlas through
the `adversarialTrace` machinery. -/
noncomputable def Atlas.toSolver {cfg : GameConfig} (A : Atlas cfg) :
    Solver cfg :=
  fun g p => (A g p).getD ⟨p, ⟨0, by norm_num⟩, 0⟩

/-- **`Atlas.toSolver` agrees with the atlas on `some` values**: when
`A g p = some pl`, the materialised solver returns exactly `pl`. -/
theorem Atlas.toSolver_apply_of_some {cfg : GameConfig} {A : Atlas cfg}
    {g : GameState} {p : Piece} {pl : Placement}
    (h : A g p = some pl) :
    A.toSolver g p = pl := by
  unfold Atlas.toSolver
  rw [h, Option.getD_some]

/-- **Solver → Atlas → Solver round-trip is identity**: for any total
solver `σ`, `σ.toAtlas.toSolver = σ`. The `Atlas.toSolver ∘ Solver.toAtlas`
composition is the identity on `Solver cfg`. Proof is `rfl` after
`funext`: every step (`some (σ g p)).getD _ = σ g p`) is a definitional
reduction. This is the SOLVER half of the Solver↔Atlas correspondence;
the ATLAS half (`A.toSolver.toAtlas = A`) holds only when `A` is
total (i.e., never returns `none`). -/
theorem Solver.toAtlas_toSolver {cfg : GameConfig} (σ : Solver cfg) :
    σ.toAtlas.toSolver = σ := by
  funext _ _
  rfl

/-- **`Solver.toAtlas` is injective**: if two solvers' atlases agree,
the solvers agree. Direct corollary of the round-trip identity
`Solver.toAtlas_toSolver`: apply `Atlas.toSolver` to both sides of
the atlas equality to recover the solvers. -/
theorem Solver.toAtlas_inj {cfg : GameConfig} {σ₁ σ₂ : Solver cfg}
    (h : σ₁.toAtlas = σ₂.toAtlas) : σ₁ = σ₂ := by
  have := congrArg Atlas.toSolver h
  rw [Solver.toAtlas_toSolver, Solver.toAtlas_toSolver] at this
  exact this

/-- **`Solver.toAtlas` equality biconditional**: `σ₁.toAtlas =
σ₂.toAtlas ↔ σ₁ = σ₂`. Forward via Tick 1811's `toAtlas_inj`;
reverse via `congrArg` on `Solver.toAtlas`. Useful as a normal
form for solver-atlas equality reasoning. -/
theorem Solver.toAtlas_eq_iff {cfg : GameConfig} {σ₁ σ₂ : Solver cfg} :
    σ₁.toAtlas = σ₂.toAtlas ↔ σ₁ = σ₂ :=
  ⟨Solver.toAtlas_inj, fun h => h ▸ rfl⟩

/-- **Atlas → Solver → Atlas preserves `some` values**: if
`A g p = some pl`, then `A.toSolver.toAtlas g p = some pl`. The
"partial reverse round-trip": for queries where `A` is defined, the
solver→atlas materialisation recovers the same value. (For queries
where `A g p = none`, the materialisation produces `some <default>`
instead, breaking the full reverse round-trip identity.) -/
theorem Atlas.toSolver_toAtlas_apply_of_some {cfg : GameConfig}
    {A : Atlas cfg} {g : GameState} {p : Piece} {pl : Placement}
    (h : A g p = some pl) :
    A.toSolver.toAtlas g p = some pl := by
  unfold Solver.toAtlas
  rw [Atlas.toSolver_apply_of_some h]

/-- **Reverse round-trip identity for total atlases**: if `A` is
total (every query returns `some`), then `A.toSolver.toAtlas = A`
as full atlas functions. Lifts `toSolver_toAtlas_apply_of_some`
pointwise via `funext` + case analysis on `A g p`; the `none`
branch is impossible under the totality hypothesis. Companion to
the Solver-side `Solver.toAtlas_toSolver` round-trip — together
they establish a **bijective correspondence** between total atlases
and solvers. -/
theorem Atlas.toSolver_toAtlas_of_total {cfg : GameConfig}
    (A : Atlas cfg) (h : ∀ g p, (A g p).isSome) :
    A.toSolver.toAtlas = A := by
  funext g p
  cases hag : A g p with
  | none =>
    have := h g p
    rw [hag] at this
    contradiction
  | some pl =>
    exact Atlas.toSolver_toAtlas_apply_of_some hag

/-- **Reverse round-trip biconditional**: `A.toSolver.toAtlas = A`
iff `A` is total. Forward: if the round-trip holds, then every
query `A g p = A.toSolver.toAtlas g p = some _`, which is `isSome`.
Reverse: Tick 1813's `toSolver_toAtlas_of_total`. Characterizes
exactly which atlases are fixed points of the reverse round-trip. -/
theorem Atlas.toSolver_toAtlas_eq_iff_total {cfg : GameConfig}
    (A : Atlas cfg) :
    A.toSolver.toAtlas = A ↔ ∀ g p, (A g p).isSome := by
  refine ⟨fun h g p => ?_, Atlas.toSolver_toAtlas_of_total A⟩
  rw [← h]
  rfl


/-- Apply one adversarial move: the bag yields piece `p`, the solver chose
placement `pl`; we force the placement's piece to be `p` so the rotation and
column come from the solver but the piece itself comes from the bag. -/
def adversarialStep (cfg : GameConfig) (g : GameState) (p : Piece)
    (pl : Placement) : GameState :=
  let pl' : Placement := { pl with piece := p }
  ⟨Placement.applyStep cfg g.board pl', g.bag.draw p⟩

/-- **Atlas closure on a Finset.** A predicate capturing what it means for an
`Atlas` to *certify infinite play* on a set of states `S`: every state in
`S` is non-lost, the atlas is defined for every bag-drawable piece, the
returned placement is valid and uses the queried piece, AND the post-step
state is back in `S`. This is the **formal definition of the proof artifact**
from `AGENTS.md`'s M4: a complete Atlas on the reachable subgraph. -/
structure Atlas.IsClosedOn (cfg : GameConfig) (A : Atlas cfg)
    (S : Finset GameState) : Prop where
  /-- Every state in `S` is non-lost (the atlas certifies survival). -/
  not_lost : ∀ g ∈ S, ¬ g.lost cfg
  /-- The atlas is defined for every state in `S` and every bag-drawable piece. -/
  total : ∀ g ∈ S, ∀ p ∈ g.bag, (A g p).isSome
  /-- Each returned placement is valid and uses the queried piece. -/
  valid : ∀ g ∈ S, ∀ p ∈ g.bag, ∀ pl, A g p = some pl →
    pl.piece = p ∧ pl.Valid cfg
  /-- Following the atlas's prescription keeps the state in `S`. -/
  closed : ∀ g ∈ S, ∀ p ∈ g.bag, ∀ pl, A g p = some pl →
    adversarialStep cfg g p pl ∈ S

/-- **Closed atlas's materialised solver keeps S closed under steps**:
for any `g ∈ S, p ∈ g.bag`, the adversarial step under `A.toSolver`
stays in `S`. Lifts the structural `closed` field of `IsClosedOn`
(which operates on raw atlas `some`-witnesses) to the materialised
solver level by composing with `Atlas.toSolver_apply_of_some`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialStep_mem {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    {g : GameState} (hg : g ∈ S) {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (A.toSolver g p) ∈ S := by
  obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp (h.total g hg p hp)
  rw [Atlas.toSolver_apply_of_some hpl]
  exact h.closed g hg p hp pl hpl

/-- **Closed atlas's materialised solver step is non-lost**: for any
`g ∈ S, p ∈ g.bag`, the result of one adversarial step under
`A.toSolver` is non-lost. Composes Tick 1818's
`toSolver_adversarialStep_mem` (step stays in S) with `h.not_lost`
(S contains no lost states). -/
theorem Atlas.IsClosedOn.toSolver_adversarialStep_not_lost {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    {g : GameState} (hg : g ∈ S) {p : Piece} (hp : p ∈ g.bag) :
    ¬ (adversarialStep cfg g p (A.toSolver g p)).lost cfg :=
  h.not_lost _ (h.toSolver_adversarialStep_mem hg hp)

/-- **Bundled Atlas step invariants**: 2-conjunct combining mem
(Tick 1818) and not_lost (Tick 1819). One-shot certificate for any
single adversarial step from `g ∈ S, p ∈ g.bag` under the
materialised solver. Atlas-level parallel to M4's bundled step
invariant. -/
theorem Atlas.IsClosedOn.toSolver_adversarialStep_mem_and_not_lost
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g : GameState} (hg : g ∈ S)
    {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (A.toSolver g p) ∈ S ∧
    ¬ (adversarialStep cfg g p (A.toSolver g p)).lost cfg :=
  ⟨h.toSolver_adversarialStep_mem hg hp,
   h.toSolver_adversarialStep_not_lost hg hp⟩

/-- **Atlas init-step closure**: `init`-step under the materialised
solver stays in S (for any piece). Specialisation of Tick 1818 to
`g = init` using `init.bag = Bag.full` (so any piece is in bag). -/
theorem Atlas.IsClosedOn.toSolver_adversarialStep_init_mem
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) (p : Piece) :
    adversarialStep cfg GameState.init p (A.toSolver GameState.init p) ∈ S :=
  h.toSolver_adversarialStep_mem hinit
    (GameState.init_bag.symm ▸ Bag.mem_full p)

/-- **Atlas init-step not-lost**: the result of one adversarial
step from init under the materialised solver is non-lost.
Specialisation of Tick 1819 to `g = init`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialStep_init_not_lost
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) (p : Piece) :
    ¬ (adversarialStep cfg GameState.init p (A.toSolver GameState.init p)).lost
      cfg :=
  h.toSolver_adversarialStep_not_lost hinit
    (GameState.init_bag.symm ▸ Bag.mem_full p)

/-- **Bundled init-step Atlas invariants**: 2-conjunct combining
Tick 1873 (mem) and Tick 1874 (not-lost). One-shot certificate
for any single adversarial step from init under the materialised
solver. -/
theorem Atlas.IsClosedOn.toSolver_adversarialStep_init_mem_and_not_lost
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) (p : Piece) :
    adversarialStep cfg GameState.init p (A.toSolver GameState.init p) ∈ S ∧
    ¬ (adversarialStep cfg GameState.init p
      (A.toSolver GameState.init p)).lost cfg :=
  ⟨h.toSolver_adversarialStep_init_mem hinit p,
   h.toSolver_adversarialStep_init_not_lost hinit p⟩

/-- **Atlas-level materialised solver returns valid placement**:
for any active query `g ∈ S, p ∈ g.bag`, the materialised solver
`A.toSolver g p` returns a placement that plays the right piece
AND is valid for the config. Lifts the `valid` field of
`IsClosedOn` (which operates on raw atlas `some`-witnesses) to the
materialised solver level. Atlas-level parallel of cycle's
`solver_valid_and_piece`. -/
theorem Atlas.IsClosedOn.toSolver_valid_and_piece {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    {g : GameState} (hg : g ∈ S) {p : Piece} (hp : p ∈ g.bag) :
    (A.toSolver g p).piece = p ∧ (A.toSolver g p).Valid cfg := by
  obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp (h.total g hg p hp)
  rw [Atlas.toSolver_apply_of_some hpl]
  exact h.valid g hg p hp pl hpl

/-- **Piece accessor for `toSolver_valid_and_piece`**: extracts
the `piece = p` clause. Useful when only the piece-equality is
needed. -/
theorem Atlas.IsClosedOn.toSolver_piece {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    {g : GameState} (hg : g ∈ S) {p : Piece} (hp : p ∈ g.bag) :
    (A.toSolver g p).piece = p :=
  (h.toSolver_valid_and_piece hg hp).1

/-- **Valid accessor for `toSolver_valid_and_piece`**: extracts
the `Valid cfg` clause. Useful when only the placement-validity
is needed. -/
theorem Atlas.IsClosedOn.toSolver_valid {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    {g : GameState} (hg : g ∈ S) {p : Piece} (hp : p ∈ g.bag) :
    (A.toSolver g p).Valid cfg :=
  (h.toSolver_valid_and_piece hg hp).2

/-- **Init piece accessor**: at `init`, the materialised solver
returns a placement playing the queried piece. Specialisation of
Tick 1877 to `g = init` using `init.bag = Bag.full`. -/
theorem Atlas.IsClosedOn.toSolver_init_piece {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) (p : Piece) :
    (A.toSolver GameState.init p).piece = p :=
  h.toSolver_piece hinit (GameState.init_bag.symm ▸ Bag.mem_full p)

/-- **Init valid accessor**: at `init`, the materialised solver
returns a valid placement. Specialisation of Tick 1878 to
`g = init` using `init.bag = Bag.full`. -/
theorem Atlas.IsClosedOn.toSolver_init_valid {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) (p : Piece) :
    (A.toSolver GameState.init p).Valid cfg :=
  h.toSolver_valid hinit (GameState.init_bag.symm ▸ Bag.mem_full p)

/-- **Full 4-conjunct Atlas step invariants**: combines step-mem
(1818), step-not-lost (1819), solver-piece (1877), solver-valid
(1878). One-shot full step certificate. Atlas-level parallel to
M4's full bundled step invariants. -/
theorem Atlas.IsClosedOn.toSolver_adversarialStep_full
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g : GameState} (hg : g ∈ S)
    {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (A.toSolver g p) ∈ S ∧
    ¬ (adversarialStep cfg g p (A.toSolver g p)).lost cfg ∧
    (A.toSolver g p).piece = p ∧
    (A.toSolver g p).Valid cfg :=
  ⟨h.toSolver_adversarialStep_mem hg hp,
   h.toSolver_adversarialStep_not_lost hg hp,
   h.toSolver_piece hg hp,
   h.toSolver_valid hg hp⟩


/-- **Atlas restriction.** Restrict an atlas to a state set `S`: return the
original atlas's value when the queried state is in `S`, else `none`. The
first operation in the **Atlas algebra** — shrinks the certified domain
without altering the prescriptions inside. -/
def Atlas.restrict {cfg : GameConfig} (A : Atlas cfg) (S : Finset GameState) :
    Atlas cfg :=
  fun g p => if g ∈ S then A g p else none

@[simp] theorem Atlas.restrict_apply_of_mem {cfg : GameConfig}
    (A : Atlas cfg) {S : Finset GameState} {g : GameState} (hg : g ∈ S)
    (p : Piece) :
    A.restrict S g p = A g p := by
  unfold Atlas.restrict
  rw [if_pos hg]

@[simp] theorem Atlas.restrict_apply_of_notMem {cfg : GameConfig}
    (A : Atlas cfg) {S : Finset GameState} {g : GameState} (hg : g ∉ S)
    (p : Piece) :
    A.restrict S g p = none := by
  unfold Atlas.restrict
  rw [if_neg hg]

/-- **`restrict` is compositional**: `(A.restrict S₁).restrict S₂ =
A.restrict (S₁ ∩ S₂)`. Two restrictions in sequence equal a single
restriction to the intersection. Foundational identity for the
`restrict` algebra. Proof: pointwise via `funext` + 4-way case
analysis on `g ∈ S₁, g ∈ S₂`. -/
theorem Atlas.restrict_restrict {cfg : GameConfig} (A : Atlas cfg)
    (S₁ S₂ : Finset GameState) :
    (A.restrict S₁).restrict S₂ = A.restrict (S₁ ∩ S₂) := by
  funext g p
  by_cases h₁ : g ∈ S₁
  · by_cases h₂ : g ∈ S₂
    · rw [Atlas.restrict_apply_of_mem _ h₂,
          Atlas.restrict_apply_of_mem _ h₁,
          Atlas.restrict_apply_of_mem _ (Finset.mem_inter.mpr ⟨h₁, h₂⟩)]
    · have hint : g ∉ S₁ ∩ S₂ :=
        fun h => h₂ (Finset.mem_inter.mp h).2
      rw [Atlas.restrict_apply_of_notMem _ h₂,
          Atlas.restrict_apply_of_notMem _ hint]
  · have hint : g ∉ S₁ ∩ S₂ :=
      fun h => h₁ (Finset.mem_inter.mp h).1
    rw [Atlas.restrict_apply_of_notMem _ hint]
    by_cases h₂ : g ∈ S₂
    · rw [Atlas.restrict_apply_of_mem _ h₂,
          Atlas.restrict_apply_of_notMem _ h₁]
    · rw [Atlas.restrict_apply_of_notMem _ h₂]

/-- **`restrict` is monotone-collapsing**: `(A.restrict S₂).restrict
S₁ = A.restrict S₁` when `S₁ ⊆ S₂`. Restricting to a smaller set
after a larger one collapses to the smaller restriction alone.
Direct corollary of Tick 1733 (`restrict_restrict`) with
`S₂ ∩ S₁ = S₁` from the subset hypothesis. -/
theorem Atlas.restrict_restrict_subset_eq {cfg : GameConfig}
    (A : Atlas cfg) {S₁ S₂ : Finset GameState} (h : S₁ ⊆ S₂) :
    (A.restrict S₂).restrict S₁ = A.restrict S₁ := by
  rw [Atlas.restrict_restrict]
  congr 1
  ext x
  rw [Finset.mem_inter]
  exact ⟨fun ⟨_, h'⟩ => h', fun h' => ⟨h h', h'⟩⟩

/-- **Restriction preserves closure.** If `A` is closed on `S`, then so is
`A.restrict S`. The restriction agrees with `A` on `S`
(`Atlas.restrict_apply_of_mem`), so all four `IsClosedOn` conditions forward
unchanged. The off-`S` portion of the atlas was already irrelevant to the
certificate. -/
theorem Atlas.IsClosedOn.restrict {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (hA : A.IsClosedOn cfg S) :
    (A.restrict S).IsClosedOn cfg S where
  not_lost := hA.not_lost
  total := fun g hg p hp => by
    rw [Atlas.restrict_apply_of_mem A hg p]
    exact hA.total g hg p hp
  valid := fun g hg p hp pl hpl => by
    rw [Atlas.restrict_apply_of_mem A hg p] at hpl
    exact hA.valid g hg p hp pl hpl
  closed := fun g hg p hp pl hpl => by
    rw [Atlas.restrict_apply_of_mem A hg p] at hpl
    exact hA.closed g hg p hp pl hpl

/-- **Atlas union with priority dispatch.** Combine two atlases: prefer `A`'s
prescription if the queried state is in `S₁`, fall back to `B` if in `S₂`,
else `none`. The second operation in the Atlas algebra (after `restrict`).
Priority dispatch avoids needing a disjointness hypothesis — `A` always wins
on the overlap. -/
def Atlas.unionOn {cfg : GameConfig} (A B : Atlas cfg)
    (S₁ S₂ : Finset GameState) : Atlas cfg :=
  fun g p => if g ∈ S₁ then A g p else if g ∈ S₂ then B g p else none

theorem Atlas.unionOn_apply_of_mem_left {cfg : GameConfig}
    (A B : Atlas cfg) {S₁ S₂ : Finset GameState} {g : GameState}
    (hg : g ∈ S₁) (p : Piece) :
    (A.unionOn B S₁ S₂) g p = A g p := by
  unfold Atlas.unionOn
  rw [if_pos hg]

theorem Atlas.unionOn_apply_of_mem_right {cfg : GameConfig}
    (A B : Atlas cfg) {S₁ S₂ : Finset GameState} {g : GameState}
    (h₁ : g ∉ S₁) (h₂ : g ∈ S₂) (p : Piece) :
    (A.unionOn B S₁ S₂) g p = B g p := by
  unfold Atlas.unionOn
  rw [if_neg h₁, if_pos h₂]

theorem Atlas.unionOn_apply_of_notMem {cfg : GameConfig}
    (A B : Atlas cfg) {S₁ S₂ : Finset GameState} {g : GameState}
    (h₁ : g ∉ S₁) (h₂ : g ∉ S₂) (p : Piece) :
    (A.unionOn B S₁ S₂) g p = none := by
  unfold Atlas.unionOn
  rw [if_neg h₁, if_neg h₂]

/-- **Union preserves closure**: if `A.IsClosedOn cfg S₁` and `B.IsClosedOn
cfg S₂`, then `(A.unionOn B S₁ S₂).IsClosedOn cfg (S₁ ∪ S₂)`. Priority
dispatch (`A` wins on `S₁`) keeps the proof clean — each field splits on
`g ∈ S₁` vs `g ∈ S₂ \ S₁` and forwards to the appropriate atlas's certificate.
No disjointness hypothesis required. -/
theorem Atlas.IsClosedOn.unionOn {cfg : GameConfig}
    {A B : Atlas cfg} {S₁ S₂ : Finset GameState}
    (hA : A.IsClosedOn cfg S₁) (hB : B.IsClosedOn cfg S₂) :
    (A.unionOn B S₁ S₂).IsClosedOn cfg (S₁ ∪ S₂) where
  not_lost := fun g hg => by
    rw [Finset.mem_union] at hg
    rcases hg with h₁ | h₂
    · exact hA.not_lost g h₁
    · exact hB.not_lost g h₂
  total := fun g hg p hp => by
    by_cases h₁ : g ∈ S₁
    · rw [Atlas.unionOn_apply_of_mem_left A B h₁ p]
      exact hA.total g h₁ p hp
    · have h₂ : g ∈ S₂ := by
        rw [Finset.mem_union] at hg
        exact hg.resolve_left h₁
      rw [Atlas.unionOn_apply_of_mem_right A B h₁ h₂ p]
      exact hB.total g h₂ p hp
  valid := fun g hg p hp pl hpl => by
    by_cases h₁ : g ∈ S₁
    · rw [Atlas.unionOn_apply_of_mem_left A B h₁ p] at hpl
      exact hA.valid g h₁ p hp pl hpl
    · have h₂ : g ∈ S₂ := by
        rw [Finset.mem_union] at hg
        exact hg.resolve_left h₁
      rw [Atlas.unionOn_apply_of_mem_right A B h₁ h₂ p] at hpl
      exact hB.valid g h₂ p hp pl hpl
  closed := fun g hg p hp pl hpl => by
    by_cases h₁ : g ∈ S₁
    · rw [Atlas.unionOn_apply_of_mem_left A B h₁ p] at hpl
      exact Finset.mem_union_left _ (hA.closed g h₁ p hp pl hpl)
    · have h₂ : g ∈ S₂ := by
        rw [Finset.mem_union] at hg
        exact hg.resolve_left h₁
      rw [Atlas.unionOn_apply_of_mem_right A B h₁ h₂ p] at hpl
      exact Finset.mem_union_right _ (hB.closed g h₂ p hp pl hpl)

/-- **Atlas refinement.** Replace the prescription at a single `(g₀, p₀)`
location with a new placement `pl'`. The third operation in the Atlas
algebra — models "policy improvement" at a specific state without
disturbing the rest of the atlas. -/
def Atlas.refine {cfg : GameConfig} (A : Atlas cfg) (g₀ : GameState)
    (p₀ : Piece) (pl' : Placement) : Atlas cfg :=
  fun g p => if g = g₀ ∧ p = p₀ then some pl' else A g p

theorem Atlas.refine_apply_at {cfg : GameConfig}
    (A : Atlas cfg) (g₀ : GameState) (p₀ : Piece) (pl' : Placement) :
    A.refine g₀ p₀ pl' g₀ p₀ = some pl' := by
  unfold Atlas.refine
  rw [if_pos ⟨rfl, rfl⟩]

theorem Atlas.refine_apply_of_ne {cfg : GameConfig}
    (A : Atlas cfg) {g₀ : GameState} {p₀ : Piece} (pl' : Placement)
    {g : GameState} {p : Piece} (h : ¬(g = g₀ ∧ p = p₀)) :
    A.refine g₀ p₀ pl' g p = A g p := by
  unfold Atlas.refine
  rw [if_neg h]

/-- **Refine overwrites: second `refine` at same `(g₀, p₀)`
overrides the first**: `(A.refine g₀ p₀ pl').refine g₀ p₀ pl'' =
A.refine g₀ p₀ pl''`. The refinement at any given point is fully
determined by the LATEST refine. Proof: pointwise via `funext` +
case analysis on `(g, p) = (g₀, p₀)`. -/
theorem Atlas.refine_refine_same {cfg : GameConfig}
    (A : Atlas cfg) (g₀ : GameState) (p₀ : Piece) (pl' pl'' : Placement) :
    (A.refine g₀ p₀ pl').refine g₀ p₀ pl'' = A.refine g₀ p₀ pl'' := by
  funext g p
  by_cases h : g = g₀ ∧ p = p₀
  · obtain ⟨rfl, rfl⟩ := h
    rw [Atlas.refine_apply_at, Atlas.refine_apply_at]
  · rw [Atlas.refine_apply_of_ne _ _ h, Atlas.refine_apply_of_ne _ _ h,
        Atlas.refine_apply_of_ne _ _ h]

/-- **Refines at different points commute**: `(A.refine g₀ p₀
pl').refine g₁ p₁ pl'' = (A.refine g₁ p₁ pl'').refine g₀ p₀ pl'`
when `(g₀, p₀) ≠ (g₁, p₁)`. Refinement order doesn't matter when
applied to distinct points. Proof: pointwise via `funext` + 3-way
case analysis on `(g, p)`'s relationship to the two refine points. -/
theorem Atlas.refine_refine_ne {cfg : GameConfig}
    (A : Atlas cfg) {g₀ g₁ : GameState} {p₀ p₁ : Piece}
    (pl' pl'' : Placement) (h : ¬(g₀ = g₁ ∧ p₀ = p₁)) :
    (A.refine g₀ p₀ pl').refine g₁ p₁ pl'' =
      (A.refine g₁ p₁ pl'').refine g₀ p₀ pl' := by
  funext g p
  by_cases h0 : g = g₀ ∧ p = p₀
  · obtain ⟨rfl, rfl⟩ := h0
    rw [Atlas.refine_apply_of_ne _ _ h,
        Atlas.refine_apply_at,
        Atlas.refine_apply_at]
  · by_cases h1 : g = g₁ ∧ p = p₁
    · have hne : ¬(g₁ = g₀ ∧ p₁ = p₀) :=
        fun ⟨he1, he2⟩ => h ⟨he1.symm, he2.symm⟩
      obtain ⟨rfl, rfl⟩ := h1
      rw [Atlas.refine_apply_at,
          Atlas.refine_apply_of_ne _ _ hne,
          Atlas.refine_apply_at]
    · rw [Atlas.refine_apply_of_ne _ _ h1,
          Atlas.refine_apply_of_ne _ _ h0,
          Atlas.refine_apply_of_ne _ _ h0,
          Atlas.refine_apply_of_ne _ _ h1]

/-- **`restrict` and `refine` commute when refine target is in `S`**:
`(A.restrict S).refine g₀ p₀ pl' = (A.refine g₀ p₀ pl').restrict S`
when `g₀ ∈ S`. The interaction between restriction and refinement is
clean when the refined point is preserved by the restriction.

When `g₀ ∉ S`, the two differ at `(g₀, p₀)`: LHS keeps the refine
(`some pl'`) while RHS's restrict zeroes it out (`none`). -/
theorem Atlas.restrict_refine_comm_of_mem {cfg : GameConfig}
    (A : Atlas cfg) {S : Finset GameState} {g₀ : GameState}
    (hg₀ : g₀ ∈ S) (p₀ : Piece) (pl' : Placement) :
    (A.restrict S).refine g₀ p₀ pl' =
      (A.refine g₀ p₀ pl').restrict S := by
  funext g p
  by_cases h0 : g = g₀ ∧ p = p₀
  · obtain ⟨rfl, rfl⟩ := h0
    rw [Atlas.refine_apply_at,
        Atlas.restrict_apply_of_mem _ hg₀,
        Atlas.refine_apply_at]
  · by_cases h_S : g ∈ S
    · rw [Atlas.refine_apply_of_ne _ _ h0,
          Atlas.restrict_apply_of_mem _ h_S,
          Atlas.restrict_apply_of_mem _ h_S,
          Atlas.refine_apply_of_ne _ _ h0]
    · rw [Atlas.refine_apply_of_ne _ _ h0,
          Atlas.restrict_apply_of_notMem _ h_S,
          Atlas.restrict_apply_of_notMem _ h_S]

/-- **`refine` on left commutes with `unionOn`** when the refine
target is in `S₁`: `(A.refine g₀ p₀ pl').unionOn B S₁ S₂ = (A.unionOn
B S₁ S₂).refine g₀ p₀ pl'` when `g₀ ∈ S₁`. The refine on the left
atlas can be hoisted outside the union when it's at an active point.

When `g₀ ∉ S₁`, LHS gives `some pl'` at `(g₀, p₀)` (priority left
wins), but RHS depends on whether `g₀ ∈ S₂` and uses B's value. -/
theorem Atlas.refine_unionOn_left {cfg : GameConfig}
    (A B : Atlas cfg) {S₁ S₂ : Finset GameState} {g₀ : GameState}
    (hg₀ : g₀ ∈ S₁) (p₀ : Piece) (pl' : Placement) :
    (A.refine g₀ p₀ pl').unionOn B S₁ S₂ =
      (A.unionOn B S₁ S₂).refine g₀ p₀ pl' := by
  funext g p
  by_cases h0 : g = g₀ ∧ p = p₀
  · rw [h0.1, h0.2,
        Atlas.unionOn_apply_of_mem_left _ B hg₀ p₀,
        Atlas.refine_apply_at,
        Atlas.refine_apply_at]
  · rw [Atlas.refine_apply_of_ne _ _ h0]
    by_cases h₁ : g ∈ S₁
    · rw [Atlas.unionOn_apply_of_mem_left _ _ h₁ p,
          Atlas.unionOn_apply_of_mem_left _ _ h₁ p,
          Atlas.refine_apply_of_ne _ _ h0]
    · by_cases h₂ : g ∈ S₂
      · rw [Atlas.unionOn_apply_of_mem_right _ _ h₁ h₂ p,
            Atlas.unionOn_apply_of_mem_right _ _ h₁ h₂ p]
      · rw [Atlas.unionOn_apply_of_notMem _ _ h₁ h₂ p,
            Atlas.unionOn_apply_of_notMem _ _ h₁ h₂ p]

/-- **`refine` on right commutes with `unionOn`** when the refine
target is in `S₂ \ S₁`: `A.unionOn (B.refine g₀ p₀ pl') S₁ S₂ =
(A.unionOn B S₁ S₂).refine g₀ p₀ pl'` when `g₀ ∉ S₁ ∧ g₀ ∈ S₂`.
Symmetric partner to Tick 1737's `refine_unionOn_left`. The refine
on the right atlas hoists outside the union when the target is
active in S₂ but not shadowed by S₁'s priority. -/
theorem Atlas.refine_unionOn_right {cfg : GameConfig}
    (A B : Atlas cfg) {S₁ S₂ : Finset GameState} {g₀ : GameState}
    (hg₀_notS₁ : g₀ ∉ S₁) (hg₀_S₂ : g₀ ∈ S₂) (p₀ : Piece)
    (pl' : Placement) :
    A.unionOn (B.refine g₀ p₀ pl') S₁ S₂ =
      (A.unionOn B S₁ S₂).refine g₀ p₀ pl' := by
  funext g p
  by_cases h0 : g = g₀ ∧ p = p₀
  · rw [h0.1, h0.2,
        Atlas.unionOn_apply_of_mem_right A _ hg₀_notS₁ hg₀_S₂ p₀,
        Atlas.refine_apply_at,
        Atlas.refine_apply_at]
  · rw [Atlas.refine_apply_of_ne _ _ h0]
    by_cases h₁ : g ∈ S₁
    · rw [Atlas.unionOn_apply_of_mem_left _ _ h₁ p,
          Atlas.unionOn_apply_of_mem_left _ _ h₁ p]
    · by_cases h₂ : g ∈ S₂
      · rw [Atlas.unionOn_apply_of_mem_right _ _ h₁ h₂ p,
            Atlas.unionOn_apply_of_mem_right _ _ h₁ h₂ p,
            Atlas.refine_apply_of_ne _ _ h0]
      · rw [Atlas.unionOn_apply_of_notMem _ _ h₁ h₂ p,
            Atlas.unionOn_apply_of_notMem _ _ h₁ h₂ p]

/-- **`restrict` distributes over `unionOn`**: `(A.unionOn B S₁ S₂).restrict
S₃ = (A.restrict S₃).unionOn (B.restrict S₃) S₁ S₂`. Restricting a
union to S₃ equals unioning the restrictions of each component to S₃.
Holds unconditionally — no constraint on S₃'s relationship to S₁/S₂.
Proof: pointwise via `funext` + case analysis on g's membership in
S₃, S₁, S₂. -/
theorem Atlas.unionOn_restrict {cfg : GameConfig}
    (A B : Atlas cfg) (S₁ S₂ S₃ : Finset GameState) :
    (A.unionOn B S₁ S₂).restrict S₃ =
      (A.restrict S₃).unionOn (B.restrict S₃) S₁ S₂ := by
  funext g p
  by_cases h₃ : g ∈ S₃
  · by_cases h₁ : g ∈ S₁
    · rw [Atlas.restrict_apply_of_mem _ h₃,
          Atlas.unionOn_apply_of_mem_left _ _ h₁ p,
          Atlas.unionOn_apply_of_mem_left _ _ h₁ p,
          Atlas.restrict_apply_of_mem _ h₃]
    · by_cases h₂ : g ∈ S₂
      · rw [Atlas.restrict_apply_of_mem _ h₃,
            Atlas.unionOn_apply_of_mem_right _ _ h₁ h₂ p,
            Atlas.unionOn_apply_of_mem_right _ _ h₁ h₂ p,
            Atlas.restrict_apply_of_mem _ h₃]
      · rw [Atlas.restrict_apply_of_mem _ h₃,
            Atlas.unionOn_apply_of_notMem _ _ h₁ h₂ p,
            Atlas.unionOn_apply_of_notMem _ _ h₁ h₂ p]
  · rw [Atlas.restrict_apply_of_notMem _ h₃]
    by_cases h₁ : g ∈ S₁
    · rw [Atlas.unionOn_apply_of_mem_left _ _ h₁ p,
          Atlas.restrict_apply_of_notMem _ h₃]
    · by_cases h₂ : g ∈ S₂
      · rw [Atlas.unionOn_apply_of_mem_right _ _ h₁ h₂ p,
            Atlas.restrict_apply_of_notMem _ h₃]
      · rw [Atlas.unionOn_apply_of_notMem _ _ h₁ h₂ p]

/-- **Restricting `unionOn` to S₁ collapses to `A.restrict S₁`**:
`(A.unionOn B S₁ S₂).restrict S₁ = A.restrict S₁`. At all points
in S₁, the union prioritizes A (left wins); restricting to S₁ then
gives exactly A's values on S₁, with `none` elsewhere — equivalent
to `A.restrict S₁`. The B atlas's contribution is invisible after
restriction to S₁. -/
theorem Atlas.restrict_unionOn_left {cfg : GameConfig}
    (A B : Atlas cfg) (S₁ S₂ : Finset GameState) :
    (A.unionOn B S₁ S₂).restrict S₁ = A.restrict S₁ := by
  funext g p
  by_cases h₁ : g ∈ S₁
  · rw [Atlas.restrict_apply_of_mem _ h₁,
        Atlas.unionOn_apply_of_mem_left _ _ h₁ p,
        Atlas.restrict_apply_of_mem _ h₁]
  · rw [Atlas.restrict_apply_of_notMem _ h₁,
        Atlas.restrict_apply_of_notMem _ h₁]

/-- **Refinement preserves closure** when the new prescription is admissible:
plays the requested piece, is `Valid cfg`, and lands back in `S`. Note: the
admissibility hypotheses (`hpiece`, `hvalid`, `hland`) need only hold for the
specific `(g₀, p₀)` being refined — they don't constrain the rest of the
atlas's domain. -/
theorem Atlas.IsClosedOn.refine {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (hA : A.IsClosedOn cfg S)
    {g₀ : GameState} {p₀ : Piece} {pl' : Placement}
    (hpiece : pl'.piece = p₀) (hvalid : pl'.Valid cfg)
    (hland : adversarialStep cfg g₀ p₀ pl' ∈ S) :
    (A.refine g₀ p₀ pl').IsClosedOn cfg S where
  not_lost := hA.not_lost
  total := fun g hg p hp => by
    by_cases h : g = g₀ ∧ p = p₀
    · obtain ⟨rfl, rfl⟩ := h
      rw [Atlas.refine_apply_at]
      simp
    · rw [Atlas.refine_apply_of_ne A pl' h]
      exact hA.total g hg p hp
  valid := fun g hg p hp pl hpl => by
    by_cases h : g = g₀ ∧ p = p₀
    · obtain ⟨rfl, rfl⟩ := h
      rw [Atlas.refine_apply_at] at hpl
      have heq : pl = pl' := (Option.some.inj hpl).symm
      subst heq
      exact ⟨hpiece, hvalid⟩
    · rw [Atlas.refine_apply_of_ne A pl' h] at hpl
      exact hA.valid g hg p hp pl hpl
  closed := fun g hg p hp pl hpl => by
    by_cases h : g = g₀ ∧ p = p₀
    · obtain ⟨rfl, rfl⟩ := h
      rw [Atlas.refine_apply_at] at hpl
      have heq : pl = pl' := (Option.some.inj hpl).symm
      subst heq
      exact hland
    · rw [Atlas.refine_apply_of_ne A pl' h] at hpl
      exact hA.closed g hg p hp pl hpl

/-- **Empty-state-set is always closed.** Any atlas is `IsClosedOn` the
empty state set, vacuously. The **identity-state-set** of the Atlas algebra
— the base case for `unionOn`-style inductive constructions. -/
theorem Atlas.IsClosedOn.empty_set {cfg : GameConfig} (A : Atlas cfg) :
    A.IsClosedOn cfg ∅ where
  not_lost := fun g hg => (Finset.notMem_empty g hg).elim
  total := fun g hg _ _ => (Finset.notMem_empty g hg).elim
  valid := fun g hg _ _ _ _ => (Finset.notMem_empty g hg).elim
  closed := fun g hg _ _ _ _ => (Finset.notMem_empty g hg).elim

/-- **Union with empty left state set reduces to restriction**: `unionOn`
generalises `restrict` — when the left state set is `∅`, the result is
exactly `B.restrict S`. -/
theorem Atlas.unionOn_empty_left {cfg : GameConfig} (A B : Atlas cfg)
    (S : Finset GameState) :
    A.unionOn B ∅ S = B.restrict S := by
  funext g p
  unfold Atlas.unionOn Atlas.restrict
  rw [if_neg (Finset.notMem_empty g)]

/-- **Union with empty right state set reduces to restriction**: symmetric
counterpart of `unionOn_empty_left` — `A.unionOn B S ∅ = A.restrict S`. -/
theorem Atlas.unionOn_empty_right {cfg : GameConfig} (A B : Atlas cfg)
    (S : Finset GameState) :
    A.unionOn B S ∅ = A.restrict S := by
  funext g p
  unfold Atlas.unionOn Atlas.restrict
  by_cases h : g ∈ S
  · rw [if_pos h, if_pos h]
  · rw [if_neg h, if_neg h, if_neg (Finset.notMem_empty g)]

/-- **Union commutativity modulo agreement on the overlap.** The priority
dispatch in `Atlas.unionOn` makes it asymmetric: `A.unionOn B S₁ S₂` favors
`A` on `S₁ ∩ S₂`, while `B.unionOn A S₂ S₁` favors `B`. The two are equal
exactly when the atlases agree on the overlap. Captures the precise
condition under which `unionOn` becomes commutative. -/
theorem Atlas.unionOn_comm_of_agree {cfg : GameConfig} (A B : Atlas cfg)
    (S₁ S₂ : Finset GameState)
    (hagree : ∀ g ∈ S₁ ∩ S₂, ∀ p, A g p = B g p) :
    A.unionOn B S₁ S₂ = B.unionOn A S₂ S₁ := by
  funext g p
  unfold Atlas.unionOn
  by_cases h₁ : g ∈ S₁ <;> by_cases h₂ : g ∈ S₂
  · rw [if_pos h₁, if_pos h₂]
    exact hagree g (Finset.mem_inter.mpr ⟨h₁, h₂⟩) p
  · rw [if_pos h₁, if_neg h₂, if_pos h₁]
  · rw [if_neg h₁, if_pos h₂, if_pos h₂]
  · rw [if_neg h₁, if_neg h₂, if_neg h₂, if_neg h₁]

/-- **Disjoint union is commutative**: when `S₁ ∩ S₂ = ∅`, the agreement
hypothesis of `unionOn_comm_of_agree` is vacuous, so `unionOn` is symmetric
unconditionally for disjoint state sets. -/
theorem Atlas.unionOn_comm_of_disjoint {cfg : GameConfig} (A B : Atlas cfg)
    {S₁ S₂ : Finset GameState} (hdisj : S₁ ∩ S₂ = ∅) :
    A.unionOn B S₁ S₂ = B.unionOn A S₂ S₁ := by
  apply Atlas.unionOn_comm_of_agree
  intro g hg p
  rw [hdisj] at hg
  exact (Finset.notMem_empty g hg).elim

/-- **Self-union is restriction to the union**: `A.unionOn A S₁ S₂ =
A.restrict (S₁ ∪ S₂)`. Self-idempotence: combining an atlas with itself on
two state sets is the same as restricting it to their union. Establishes
the natural connection between `unionOn` and `restrict` at the atlas
diagonal. -/
theorem Atlas.unionOn_self {cfg : GameConfig} (A : Atlas cfg)
    (S₁ S₂ : Finset GameState) :
    A.unionOn A S₁ S₂ = A.restrict (S₁ ∪ S₂) := by
  funext g p
  unfold Atlas.unionOn Atlas.restrict
  by_cases h₁ : g ∈ S₁
  · rw [if_pos h₁, if_pos (Finset.mem_union_left S₂ h₁)]
  · by_cases h₂ : g ∈ S₂
    · rw [if_neg h₁, if_pos h₂, if_pos (Finset.mem_union_right S₁ h₂)]
    · have hnotunion : g ∉ S₁ ∪ S₂ := by
        intro h
        rw [Finset.mem_union] at h
        exact h.elim h₁ h₂
      rw [if_neg h₁, if_neg h₂, if_neg hnotunion]

/-- **`unionOn` is associative**: `(A.unionOn B S₁ S₂).unionOn C
(S₁ ∪ S₂) S₃ = A.unionOn (B.unionOn C S₂ S₃) S₁ (S₂ ∪ S₃)`. The
atlas-level associativity for the priority-dispatch union operation.
Lifts to M4 level via Tick 1687's `M4Artifact.union_assoc`, but
here at the raw atlas function level.

Proof: 4-way case analysis on `g`'s membership in `S₁, S₂, S₃`. Each
case reduces both sides to the same `Aᵢ g p` via combinations of
`unionOn_apply_of_mem_left`/`_right`/`_notMem`; the all-negative
case both sides give `none`. -/
theorem Atlas.unionOn_assoc {cfg : GameConfig}
    (A B C : Atlas cfg) (S₁ S₂ S₃ : Finset GameState) :
    (A.unionOn B S₁ S₂).unionOn C (S₁ ∪ S₂) S₃ =
      A.unionOn (B.unionOn C S₂ S₃) S₁ (S₂ ∪ S₃) := by
  funext g p
  by_cases h1 : g ∈ S₁
  · have hg12 : g ∈ S₁ ∪ S₂ := Finset.mem_union_left _ h1
    rw [Atlas.unionOn_apply_of_mem_left _ C hg12 p,
        Atlas.unionOn_apply_of_mem_left A B h1 p,
        Atlas.unionOn_apply_of_mem_left A _ h1 p]
  · by_cases h2 : g ∈ S₂
    · have hg12 : g ∈ S₁ ∪ S₂ := Finset.mem_union_right _ h2
      have hg23 : g ∈ S₂ ∪ S₃ := Finset.mem_union_left _ h2
      rw [Atlas.unionOn_apply_of_mem_left _ C hg12 p,
          Atlas.unionOn_apply_of_mem_right A B h1 h2 p,
          Atlas.unionOn_apply_of_mem_right A _ h1 hg23 p,
          Atlas.unionOn_apply_of_mem_left B C h2 p]
    · by_cases h3 : g ∈ S₃
      · have hg12 : g ∉ S₁ ∪ S₂ := by
          rw [Finset.mem_union]; tauto
        have hg23 : g ∈ S₂ ∪ S₃ := Finset.mem_union_right _ h3
        rw [Atlas.unionOn_apply_of_mem_right _ C hg12 h3 p,
            Atlas.unionOn_apply_of_mem_right A _ h1 hg23 p,
            Atlas.unionOn_apply_of_mem_right B C h2 h3 p]
      · have hg12 : g ∉ S₁ ∪ S₂ := by
          rw [Finset.mem_union]; tauto
        have hg23 : g ∉ S₂ ∪ S₃ := by
          rw [Finset.mem_union]; tauto
        rw [Atlas.unionOn_apply_of_notMem _ C hg12 h3 p,
            Atlas.unionOn_apply_of_notMem A _ h1 hg23 p]

/-- **Total-at predicate.** An atlas `A` is *total at* state `g` if it
returns `some` for every bag-drawable piece. The structural ingredient of
`Atlas.IsClosedOn`'s `total` clause. -/
def Atlas.IsTotalAt {cfg : GameConfig} (A : Atlas cfg) (g : GameState) : Prop :=
  ∀ p ∈ g.bag, (A g p).isSome = true

instance {cfg : GameConfig} (A : Atlas cfg) (g : GameState) :
    Decidable (A.IsTotalAt g) := by
  unfold Atlas.IsTotalAt
  infer_instance

/-- **Canonical domain** of an atlas within a host state-set `S`: the
subset of states `g ∈ S` where the atlas is total at `g`. The largest
sub-set of `S` for which the atlas could possibly satisfy
`IsClosedOn`'s `total` clause. -/
def Atlas.dom {cfg : GameConfig} (A : Atlas cfg) (S : Finset GameState) :
    Finset GameState :=
  S.filter A.IsTotalAt

@[simp] theorem Atlas.mem_dom {cfg : GameConfig} (A : Atlas cfg)
    {S : Finset GameState} {g : GameState} :
    g ∈ A.dom S ↔ g ∈ S ∧ A.IsTotalAt g := by
  unfold Atlas.dom
  exact Finset.mem_filter

theorem Atlas.dom_subset {cfg : GameConfig} (A : Atlas cfg)
    (S : Finset GameState) : A.dom S ⊆ S := by
  unfold Atlas.dom
  exact Finset.filter_subset _ _

/-- **Restriction preserves the canonical domain.** Restricting `A` to `S`
and then taking its domain within `S` gives the same Finset as taking `A`'s
domain within `S` directly. Because `A.restrict S` agrees with `A` on `S`,
the totality predicate is unchanged on the filtered universe. -/
theorem Atlas.restrict_dom_eq {cfg : GameConfig} (A : Atlas cfg)
    (S : Finset GameState) :
    (A.restrict S).dom S = A.dom S := by
  unfold Atlas.dom
  apply Finset.filter_congr
  intro g hg
  unfold Atlas.IsTotalAt
  refine ⟨fun hT p hp => ?_, fun hT p hp => ?_⟩
  · have := hT p hp
    rwa [Atlas.restrict_apply_of_mem A hg p] at this
  · rw [Atlas.restrict_apply_of_mem A hg p]
    exact hT p hp

/-- **Restriction reflects closure.** Converse of `Atlas.IsClosedOn.restrict`
(Tick 1603): if the *restricted* atlas is closed on `S`, then so is the
original — because the restriction agrees with the original on the only
queries the certificate examines (`g ∈ S, p ∈ g.bag`). -/
theorem Atlas.IsClosedOn.of_restrict {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState}
    (hA : (A.restrict S).IsClosedOn cfg S) :
    A.IsClosedOn cfg S where
  not_lost := hA.not_lost
  total := fun g hg p hp => by
    have := hA.total g hg p hp
    rwa [Atlas.restrict_apply_of_mem A hg p] at this
  valid := fun g hg p hp pl hpl => by
    rw [← Atlas.restrict_apply_of_mem A hg p] at hpl
    exact hA.valid g hg p hp pl hpl
  closed := fun g hg p hp pl hpl => by
    rw [← Atlas.restrict_apply_of_mem A hg p] at hpl
    exact hA.closed g hg p hp pl hpl

/-- **Closure is invariant under restriction**: `A.IsClosedOn cfg S` iff
`(A.restrict S).IsClosedOn cfg S`. Bundles `Atlas.IsClosedOn.restrict`
(Tick 1603) with `Atlas.IsClosedOn.of_restrict` (this tick). Shows
restriction is **semantically transparent** for `IsClosedOn cfg S`
certificates. -/
theorem Atlas.IsClosedOn.restrict_iff {cfg : GameConfig}
    (A : Atlas cfg) (S : Finset GameState) :
    (A.restrict S).IsClosedOn cfg S ↔ A.IsClosedOn cfg S :=
  ⟨Atlas.IsClosedOn.of_restrict, Atlas.IsClosedOn.restrict⟩

/-- The `total` clause of `IsClosedOn cfg A S` is precisely `S ⊆ A.dom S`
— totality of the atlas over the bag at each state in `S` is the same
as `S` being contained in `A`'s canonical domain within `S`. -/
theorem Atlas.subset_dom_iff_total {cfg : GameConfig} (A : Atlas cfg)
    (S : Finset GameState) :
    S ⊆ A.dom S ↔ ∀ g ∈ S, A.IsTotalAt g := by
  refine ⟨fun h g hg => ?_, fun h g hg => ?_⟩
  · have := h hg
    rw [Atlas.mem_dom] at this
    exact this.2
  · rw [Atlas.mem_dom]
    exact ⟨hg, h g hg⟩

/-- A closed atlas covers its candidate set: `A.IsClosedOn cfg S → S ⊆
A.dom S`. The `total` clause of the certificate exactly says the
candidate set sits inside the domain. -/
theorem Atlas.IsClosedOn.subset_dom {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (hA : A.IsClosedOn cfg S) :
    S ⊆ A.dom S := by
  rw [Atlas.subset_dom_iff_total]
  intro g hg p hp
  exact hA.total g hg p hp

/-- **Atlas domain equals the candidate set for closed atlases**:
`A.IsClosedOn cfg S → A.dom S = S`. Antisymmetric combination of
`Atlas.dom_subset` (`A.dom S ⊆ S`, always) and
`Atlas.IsClosedOn.subset_dom` (`S ⊆ A.dom S`, from totality). -/
theorem Atlas.IsClosedOn.dom_eq {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S) :
    A.dom S = S :=
  Finset.Subset.antisymm (A.dom_subset S) h.subset_dom

/-- **Atlas dom cardinality equals candidate set cardinality**:
`(A.dom S).card = S.card` for closed atlases. Direct cardinality
corollary of `dom_eq`. Atlas-side parallel of M4's Tick 1959. -/
theorem Atlas.IsClosedOn.dom_card_eq {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S) :
    (A.dom S).card = S.card := by
  rw [h.dom_eq]

/-- **Factor `IsClosedOn` through `dom`.** A 4-clause `IsClosedOn`
certificate is equivalent to a `dom`-containment side condition plus
three local-pointwise clauses (`not_lost` / `valid` / `closed`). Useful
when the totality witness comes from a separate (decidable) computation,
leaving only the gameplay-side clauses to prove. -/
theorem Atlas.IsClosedOn.iff_dom_subset_and {cfg : GameConfig}
    (A : Atlas cfg) (S : Finset GameState) :
    A.IsClosedOn cfg S ↔
      S ⊆ A.dom S ∧
      (∀ g ∈ S, ¬ g.lost cfg) ∧
      (∀ g ∈ S, ∀ p ∈ g.bag, ∀ pl, A g p = some pl →
        pl.piece = p ∧ pl.Valid cfg) ∧
      (∀ g ∈ S, ∀ p ∈ g.bag, ∀ pl, A g p = some pl →
        adversarialStep cfg g p pl ∈ S) := by
  refine ⟨fun hA => ⟨hA.subset_dom, hA.not_lost, hA.valid, hA.closed⟩, ?_⟩
  rintro ⟨hdom, hnl, hv, hc⟩
  rw [Atlas.subset_dom_iff_total] at hdom
  exact ⟨hnl, fun g hg p hp => hdom g hg p hp, hv, hc⟩

/-- **`dom` is monotone in the state set**: a larger candidate set yields
a larger domain. Just `Finset.filter` monotonicity in disguise. -/
theorem Atlas.dom_mono {cfg : GameConfig} (A : Atlas cfg)
    {S₁ S₂ : Finset GameState} (h : S₁ ⊆ S₂) :
    A.dom S₁ ⊆ A.dom S₂ := by
  intro g hg
  rw [Atlas.mem_dom] at hg ⊢
  exact ⟨h hg.1, hg.2⟩

/-- **`dom` distributes over `unionOn`-style state-set union**:
`A.dom (S₁ ∪ S₂) = A.dom S₁ ∪ A.dom S₂`. The `IsTotalAt` predicate is
constant in `g` (independent of the candidate set), so filtering through
either side of the union is equivalent to filtering through the union. -/
theorem Atlas.dom_union {cfg : GameConfig} (A : Atlas cfg)
    (S₁ S₂ : Finset GameState) :
    A.dom (S₁ ∪ S₂) = A.dom S₁ ∪ A.dom S₂ := by
  ext g
  rw [Finset.mem_union, Atlas.mem_dom, Atlas.mem_dom, Atlas.mem_dom,
      Finset.mem_union]
  refine ⟨?_, ?_⟩
  · rintro ⟨h1 | h2, hT⟩
    · exact Or.inl ⟨h1, hT⟩
    · exact Or.inr ⟨h2, hT⟩
  · rintro (⟨h, hT⟩ | ⟨h, hT⟩)
    · exact ⟨Or.inl h, hT⟩
    · exact ⟨Or.inr h, hT⟩

/-- **`A.dom` distributes over intersection**: `A.dom (S₁ ∩ S₂) =
A.dom S₁ ∩ A.dom S₂`. Mirrors `Atlas.dom_union` at the intersection
level. The `IsTotalAt` predicate is constant in `g` (independent of
S), so dom can be pushed inside intersection too. -/
theorem Atlas.dom_inter {cfg : GameConfig} (A : Atlas cfg)
    (S₁ S₂ : Finset GameState) :
    A.dom (S₁ ∩ S₂) = A.dom S₁ ∩ A.dom S₂ := by
  ext g
  rw [Finset.mem_inter, Atlas.mem_dom, Atlas.mem_dom, Atlas.mem_dom,
      Finset.mem_inter]
  refine ⟨?_, ?_⟩
  · rintro ⟨⟨h1, h2⟩, hT⟩
    exact ⟨⟨h1, hT⟩, ⟨h2, hT⟩⟩
  · rintro ⟨⟨h1, hT⟩, ⟨h2, _⟩⟩
    exact ⟨⟨h1, h2⟩, hT⟩

/-- **Refinement only grows the domain.** Replacing a single prescription
at `(g₀, p₀)` with `some pl'` can only add states to the certified domain,
never remove. Proof: for each `g ∈ A.dom S` and `p ∈ g.bag`, the refined
value at `(g, p)` is either `some pl'` (match case, trivially `isSome`)
or `A g p` (non-match case, `isSome` by hypothesis). -/
theorem Atlas.dom_subset_refine_dom {cfg : GameConfig} (A : Atlas cfg)
    (g₀ : GameState) (p₀ : Piece) (pl' : Placement) (S : Finset GameState) :
    A.dom S ⊆ (A.refine g₀ p₀ pl').dom S := by
  intro g hg
  rw [Atlas.mem_dom] at hg ⊢
  refine ⟨hg.1, ?_⟩
  intro p hp
  by_cases h : g = g₀ ∧ p = p₀
  · obtain ⟨rfl, rfl⟩ := h
    rw [Atlas.refine_apply_at]
    rfl
  · rw [Atlas.refine_apply_of_ne A pl' h]
    exact hg.2 p hp

/-- **Refinement outside the candidate set is invisible**: when `g₀ ∉ S`,
refining at `(g₀, p₀)` doesn't change `A.dom S`. Useful for atlas-extension
arguments where the refinement target is outside the currently-certified
set. -/
theorem Atlas.refine_dom_of_notMem {cfg : GameConfig} (A : Atlas cfg)
    {g₀ : GameState} (p₀ : Piece) (pl' : Placement) {S : Finset GameState}
    (h : g₀ ∉ S) :
    (A.refine g₀ p₀ pl').dom S = A.dom S := by
  unfold Atlas.dom
  apply Finset.filter_congr
  intro g hg
  have hne : g ≠ g₀ := fun heq => h (heq ▸ hg)
  unfold Atlas.IsTotalAt
  refine ⟨fun hT p hp => ?_, fun hT p hp => ?_⟩
  · have hpne : ¬(g = g₀ ∧ p = p₀) := fun ⟨hgg₀, _⟩ => hne hgg₀
    have := hT p hp
    rwa [Atlas.refine_apply_of_ne A pl' hpne] at this
  · have hpne : ¬(g = g₀ ∧ p = p₀) := fun ⟨hgg₀, _⟩ => hne hgg₀
    rw [Atlas.refine_apply_of_ne A pl' hpne]
    exact hT p hp

/-- **Atlas certificate transfer under agreement.** If `A g p = B g p` for
every active query (`g ∈ S, p ∈ g.bag`), then `A.IsClosedOn cfg S` transfers
to `B`. The certificate depends only on the atlas's restriction to the
active query domain. -/
theorem Atlas.IsClosedOn.transfer {cfg : GameConfig} {A B : Atlas cfg}
    {S : Finset GameState}
    (h : ∀ g ∈ S, ∀ p ∈ g.bag, A g p = B g p)
    (hA : A.IsClosedOn cfg S) :
    B.IsClosedOn cfg S where
  not_lost := hA.not_lost
  total := fun g hg p hp => by
    rw [← h g hg p hp]
    exact hA.total g hg p hp
  valid := fun g hg p hp pl hpl => by
    rw [← h g hg p hp] at hpl
    exact hA.valid g hg p hp pl hpl
  closed := fun g hg p hp pl hpl => by
    rw [← h g hg p hp] at hpl
    exact hA.closed g hg p hp pl hpl

/-- **Atlas extensionality for certificates.** Bidirectional form of
`Atlas.IsClosedOn.transfer` — two atlases that agree on all active queries
have the same `IsClosedOn cfg S` status. This is the **certificate
extensionality principle**: only the active-query restriction of the atlas
matters for closure. -/
theorem Atlas.IsClosedOn.congr {cfg : GameConfig} {A B : Atlas cfg}
    {S : Finset GameState}
    (h : ∀ g ∈ S, ∀ p ∈ g.bag, A g p = B g p) :
    A.IsClosedOn cfg S ↔ B.IsClosedOn cfg S :=
  ⟨Atlas.IsClosedOn.transfer h,
   Atlas.IsClosedOn.transfer (fun g hg p hp => (h g hg p hp).symm)⟩

/-- **Canonical (minimal) form of an atlas restricted to active queries.**
Returns `A g p` only when `g ∈ S` AND `p ∈ g.bag`; `none` otherwise. The
smallest atlas certificate-equivalent to `A` on `S` — any modification
outside the active query domain is invisible to `IsClosedOn cfg S`. -/
def Atlas.canon {cfg : GameConfig} (A : Atlas cfg) (S : Finset GameState) :
    Atlas cfg :=
  fun g p => if g ∈ S ∧ p ∈ g.bag then A g p else none

theorem Atlas.canon_apply_of_active {cfg : GameConfig} (A : Atlas cfg)
    {S : Finset GameState} {g : GameState} (hg : g ∈ S) {p : Piece}
    (hp : p ∈ g.bag) :
    A.canon S g p = A g p := by
  unfold Atlas.canon
  rw [if_pos ⟨hg, hp⟩]

theorem Atlas.canon_apply_of_inactive {cfg : GameConfig} (A : Atlas cfg)
    {S : Finset GameState} {g : GameState} {p : Piece}
    (h : ¬ (g ∈ S ∧ p ∈ g.bag)) :
    A.canon S g p = none := by
  unfold Atlas.canon
  rw [if_neg h]

/-- **Closure is invariant under canonicalization**: `A.canon S` is
certificate-equivalent to `A` on `S`. Combined with the extensionality
principle (Tick 1614), `canon` realises the "smallest equivalent atlas"
construction. -/
theorem Atlas.IsClosedOn.canon_iff {cfg : GameConfig} (A : Atlas cfg)
    (S : Finset GameState) :
    (A.canon S).IsClosedOn cfg S ↔ A.IsClosedOn cfg S :=
  (Atlas.IsClosedOn.congr
    (fun _ hg _ hp => (Atlas.canon_apply_of_active A hg hp).symm)).symm

/-- **`canon` is idempotent**: `(A.canon S).canon S = A.canon S`.
Canonicalisation reaches a fixed point in one step. -/
theorem Atlas.canon_canon {cfg : GameConfig} (A : Atlas cfg)
    (S : Finset GameState) :
    (A.canon S).canon S = A.canon S := by
  funext g p
  by_cases h : g ∈ S ∧ p ∈ g.bag
  · rw [Atlas.canon_apply_of_active _ h.1 h.2,
        Atlas.canon_apply_of_active _ h.1 h.2]
  · rw [Atlas.canon_apply_of_inactive _ h, Atlas.canon_apply_of_inactive _ h]

/-- **Canon-level reverse round-trip for closed atlases**: if `A`
is closed on `S`, then `A.toSolver.toAtlas.canon S = A.canon S`.
Even though `A` may have arbitrary out-of-`S` values (which the
toSolver materialisation might alter), at the canonical projection
both sides agree: active queries hit the total values that survive
the round-trip; inactive queries collapse to `none`. -/
theorem Atlas.toSolver_toAtlas_canon_eq_canon_of_closed {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S) :
    A.toSolver.toAtlas.canon S = A.canon S := by
  funext g p
  by_cases hga : g ∈ S ∧ p ∈ g.bag
  · obtain ⟨hg, hp⟩ := hga
    rw [Atlas.canon_apply_of_active _ hg hp,
        Atlas.canon_apply_of_active _ hg hp]
    cases hag : A g p with
    | none =>
      have := h.total g hg p hp
      rw [hag] at this
      contradiction
    | some pl =>
      exact Atlas.toSolver_toAtlas_apply_of_some hag
  · rw [Atlas.canon_apply_of_inactive _ hga,
        Atlas.canon_apply_of_inactive _ hga]

/-- **`canon` absorbs `restrict`**: `(A.restrict S).canon S = A.canon S`.
The bag-active filter in `canon` subsumes the state-only filter in
`restrict`, so the composition collapses. -/
theorem Atlas.canon_restrict {cfg : GameConfig} (A : Atlas cfg)
    (S : Finset GameState) :
    (A.restrict S).canon S = A.canon S := by
  funext g p
  by_cases h : g ∈ S ∧ p ∈ g.bag
  · rw [Atlas.canon_apply_of_active _ h.1 h.2,
        Atlas.canon_apply_of_active _ h.1 h.2,
        Atlas.restrict_apply_of_mem _ h.1]
  · rw [Atlas.canon_apply_of_inactive _ h, Atlas.canon_apply_of_inactive _ h]

/-- **`canon` absorbs a subsequent `restrict` on the same set**:
`(A.canon S).restrict S = A.canon S`. Once canonicalised, restricting
to `S` is a no-op (canon's state-filter already enforces `g ∈ S`). -/
theorem Atlas.restrict_canon {cfg : GameConfig} (A : Atlas cfg)
    (S : Finset GameState) :
    (A.canon S).restrict S = A.canon S := by
  funext g p
  by_cases hg : g ∈ S
  · rw [Atlas.restrict_apply_of_mem _ hg]
  · rw [Atlas.restrict_apply_of_notMem _ hg,
        Atlas.canon_apply_of_inactive _ (fun ⟨h, _⟩ => hg h)]

/-- **Generalised canon-restrict factorisation**: `(A.restrict T).canon S
= A.canon (S ∩ T)`. The state-filter `T` and the canon's state-filter `S`
combine via intersection. Strict generalisation of `canon_restrict` (Tick
1616, the `T := S` case). -/
theorem Atlas.canon_restrict_inter {cfg : GameConfig} (A : Atlas cfg)
    (S T : Finset GameState) :
    (A.restrict T).canon S = A.canon (S ∩ T) := by
  funext g p
  by_cases hST : g ∈ S ∧ p ∈ g.bag
  · rw [Atlas.canon_apply_of_active _ hST.1 hST.2]
    by_cases hT : g ∈ T
    · rw [Atlas.restrict_apply_of_mem _ hT,
          Atlas.canon_apply_of_active _ (Finset.mem_inter.mpr ⟨hST.1, hT⟩) hST.2]
    · rw [Atlas.restrict_apply_of_notMem _ hT,
          Atlas.canon_apply_of_inactive _
            (fun ⟨hint, _⟩ => hT (Finset.mem_inter.mp hint).2)]
  · rw [Atlas.canon_apply_of_inactive _ hST,
        Atlas.canon_apply_of_inactive _
          (fun ⟨hint, hp⟩ =>
            hST ⟨(Finset.mem_inter.mp hint).1, hp⟩)]

/-- **`canon` distributes over `unionOn`**: `(A.unionOn B S₁ S₂).canon
(S₁ ∪ S₂) = (A.canon S₁).unionOn (B.canon S₂) S₁ S₂`. Canonicalising the
union of two atlases at the union of their state sets equals unioning the
individually canonicalised atlases. The distribution preserves priority
dispatch: the `g ∈ S₁` priority on the outer `unionOn` translates to
the `S₁ ∪ S₂` (via `Finset.mem_union_left`) on the canonicalised form. -/
theorem Atlas.canon_unionOn {cfg : GameConfig} (A B : Atlas cfg)
    (S₁ S₂ : Finset GameState) :
    (A.unionOn B S₁ S₂).canon (S₁ ∪ S₂) =
      (A.canon S₁).unionOn (B.canon S₂) S₁ S₂ := by
  funext g p
  by_cases h₁ : g ∈ S₁
  · have h_union : g ∈ S₁ ∪ S₂ := Finset.mem_union_left _ h₁
    by_cases hp : p ∈ g.bag
    · rw [Atlas.canon_apply_of_active _ h_union hp,
          Atlas.unionOn_apply_of_mem_left A B h₁ p,
          Atlas.unionOn_apply_of_mem_left _ _ h₁ p,
          Atlas.canon_apply_of_active _ h₁ hp]
    · rw [Atlas.canon_apply_of_inactive _ (fun h => hp h.2),
          Atlas.unionOn_apply_of_mem_left _ _ h₁ p,
          Atlas.canon_apply_of_inactive _ (fun h => hp h.2)]
  · by_cases h₂ : g ∈ S₂
    · have h_union : g ∈ S₁ ∪ S₂ := Finset.mem_union_right _ h₂
      by_cases hp : p ∈ g.bag
      · rw [Atlas.canon_apply_of_active _ h_union hp,
            Atlas.unionOn_apply_of_mem_right A B h₁ h₂ p,
            Atlas.unionOn_apply_of_mem_right _ _ h₁ h₂ p,
            Atlas.canon_apply_of_active _ h₂ hp]
      · rw [Atlas.canon_apply_of_inactive _ (fun h => hp h.2),
            Atlas.unionOn_apply_of_mem_right _ _ h₁ h₂ p,
            Atlas.canon_apply_of_inactive _ (fun h => hp h.2)]
    · have h_not_union : g ∉ S₁ ∪ S₂ := fun h =>
        (Finset.mem_union.mp h).elim h₁ h₂
      rw [Atlas.canon_apply_of_inactive _ (fun h => h_not_union h.1),
          Atlas.unionOn_apply_of_notMem _ _ h₁ h₂]

/-- **Left-projection of canon over unionOn**: canonicalising the
union at `S₁` (the left set) collapses to `A.canon S₁`. The right
operand `B` is filtered out because the `unionOn` priority dispatch
gives the left value on every `g ∈ S₁`, and the `canon S₁` filter
zeroes out everything outside `S₁`. Pure projection — does NOT
require knowing whether `S₁ ∩ S₂` is empty or how `B` behaves on
overlap. -/
theorem Atlas.canon_unionOn_left {cfg : GameConfig} (A B : Atlas cfg)
    (S₁ S₂ : Finset GameState) :
    (A.unionOn B S₁ S₂).canon S₁ = A.canon S₁ := by
  funext g p
  by_cases hg : g ∈ S₁
  · by_cases hp : p ∈ g.bag
    · rw [Atlas.canon_apply_of_active _ hg hp,
          Atlas.unionOn_apply_of_mem_left A B hg p,
          Atlas.canon_apply_of_active _ hg hp]
    · rw [Atlas.canon_apply_of_inactive _ (fun h => hp h.2),
          Atlas.canon_apply_of_inactive _ (fun h => hp h.2)]
  · rw [Atlas.canon_apply_of_inactive _ (fun h => hg h.1),
        Atlas.canon_apply_of_inactive _ (fun h => hg h.1)]

/-- **Right-projection of canon over unionOn, under disjointness**:
when `S₁` and `S₂` are disjoint, canonicalising the union at `S₂`
collapses to `B.canon S₂`. The disjointness hypothesis is essential:
on the overlap `S₁ ∩ S₂`, the priority dispatch would still resolve
to `A`, breaking the projection. Symmetric to `canon_unionOn_left`
but with the disjointness side-condition. -/
theorem Atlas.canon_unionOn_right_of_disjoint {cfg : GameConfig}
    (A B : Atlas cfg) (S₁ S₂ : Finset GameState)
    (hdisj : Disjoint S₁ S₂) :
    (A.unionOn B S₁ S₂).canon S₂ = B.canon S₂ := by
  funext g p
  by_cases hg : g ∈ S₂
  · have hg_not_S1 : g ∉ S₁ := fun h => Finset.disjoint_left.mp hdisj h hg
    by_cases hp : p ∈ g.bag
    · rw [Atlas.canon_apply_of_active _ hg hp,
          Atlas.unionOn_apply_of_mem_right A B hg_not_S1 hg p,
          Atlas.canon_apply_of_active _ hg hp]
    · rw [Atlas.canon_apply_of_inactive _ (fun h => hp h.2),
          Atlas.canon_apply_of_inactive _ (fun h => hp h.2)]
  · rw [Atlas.canon_apply_of_inactive _ (fun h => hg h.1),
        Atlas.canon_apply_of_inactive _ (fun h => hg h.1)]

/-- **`canon` distributes over `refine` at active targets**: when the
refinement target `(g₀, p₀)` is *active* in `S` (both `g₀ ∈ S` and
`p₀ ∈ g₀.bag`), then `(A.refine g₀ p₀ pl').canon S = (A.canon S).refine
g₀ p₀ pl'`. The activeness hypothesis is essential — without it, the LHS
might mask the refined value at `(g₀, p₀)` via the canon's filter, while
the RHS would still inject `some pl'`. -/
theorem Atlas.canon_refine_of_active {cfg : GameConfig} (A : Atlas cfg)
    {g₀ : GameState} {p₀ : Piece} (pl' : Placement) {S : Finset GameState}
    (hg₀ : g₀ ∈ S) (hp₀ : p₀ ∈ g₀.bag) :
    (A.refine g₀ p₀ pl').canon S = (A.canon S).refine g₀ p₀ pl' := by
  funext g p
  by_cases h : g = g₀ ∧ p = p₀
  · obtain ⟨rfl, rfl⟩ := h
    rw [Atlas.canon_apply_of_active _ hg₀ hp₀,
        Atlas.refine_apply_at,
        Atlas.refine_apply_at]
  · rw [Atlas.refine_apply_of_ne (A.canon S) pl' h]
    by_cases h_active : g ∈ S ∧ p ∈ g.bag
    · rw [Atlas.canon_apply_of_active _ h_active.1 h_active.2,
          Atlas.refine_apply_of_ne A pl' h,
          Atlas.canon_apply_of_active _ h_active.1 h_active.2]
    · rw [Atlas.canon_apply_of_inactive _ h_active,
          Atlas.canon_apply_of_inactive _ h_active]

/-- **Canonicalisation preserves the canonical domain.** `(A.canon S).dom S
= A.dom S`. Because `A.canon S` agrees with `A` on every active query
(`g ∈ S, p ∈ g.bag`), the `IsTotalAt` predicate is unchanged on the
filter universe `S`. -/
theorem Atlas.canon_dom {cfg : GameConfig} (A : Atlas cfg)
    (S : Finset GameState) :
    (A.canon S).dom S = A.dom S := by
  unfold Atlas.dom
  apply Finset.filter_congr
  intro g hg
  unfold Atlas.IsTotalAt
  refine ⟨fun hT p hp => ?_, fun hT p hp => ?_⟩
  · have := hT p hp
    rwa [Atlas.canon_apply_of_active A hg hp] at this
  · rw [Atlas.canon_apply_of_active A hg hp]
    exact hT p hp

/-- **Canonicalised closed atlas's domain equals the candidate
set**: `A.IsClosedOn cfg S → (A.canon S).dom S = S`. Composes
`Atlas.canon_dom` (preserves domain) with Tick 1884
(`IsClosedOn.dom_eq`). -/
theorem Atlas.IsClosedOn.canon_dom_eq {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S) :
    (A.canon S).dom S = S := by
  rw [Atlas.canon_dom]
  exact h.dom_eq

/-- **Canonicalised closed atlas's domain cardinality equals
candidate set cardinality**: `A.IsClosedOn cfg S → ((A.canon S).dom
S).card = S.card`. Direct cardinality corollary of `canon_dom_eq`.
Canon-form parallel of Tick 1960's atlas `dom_card_eq`. -/
theorem Atlas.IsClosedOn.canon_dom_card_eq {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S) :
    ((A.canon S).dom S).card = S.card := by
  rw [h.canon_dom_eq]

/-- **Restricted closed atlas's domain equals the candidate set**:
`A.IsClosedOn cfg S → (A.restrict S).dom S = S`. Composes
`Atlas.restrict_dom_eq` (preserves domain) with Tick 1884
(`IsClosedOn.dom_eq`). -/
theorem Atlas.IsClosedOn.restrict_dom_eq {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S) :
    (A.restrict S).dom S = S := by
  rw [Atlas.restrict_dom_eq]
  exact h.dom_eq

/-- **Restricted closed atlas's domain cardinality equals candidate
set cardinality**: `A.IsClosedOn cfg S → ((A.restrict S).dom S).card
= S.card`. Direct cardinality corollary of `restrict_dom_eq`.
Restrict-form parallel of Tick 1960 (`dom_card_eq`) and Tick 1962
(`canon_dom_card_eq`). -/
theorem Atlas.IsClosedOn.restrict_dom_card_eq {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S) :
    ((A.restrict S).dom S).card = S.card := by
  rw [h.restrict_dom_eq]

/-- **Closure is preserved under canonicalisation**:
`A.IsClosedOn cfg S → (A.canon S).IsClosedOn cfg S`. Named
accessor for the `.mpr` direction of `Atlas.IsClosedOn.canon_iff`. -/
theorem Atlas.IsClosedOn.canon {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S) :
    (A.canon S).IsClosedOn cfg S :=
  (Atlas.IsClosedOn.canon_iff A S).mpr h

/-- **Canon-of-closed-atlas: `none` iff inactive**: for a closed
atlas, `(A.canon S) g p = none` iff the query `(g, p)` is inactive
(not `g ∈ S ∧ p ∈ g.bag`). Forward: contrapositive — at active
queries, `A g p` is `some _` (via `h.total`), so canon's active
output is also `some`. Reverse: `canon_apply_of_inactive`
unconditionally. -/
theorem Atlas.IsClosedOn.canon_apply_eq_none_iff_inactive
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g : GameState} {p : Piece} :
    (A.canon S) g p = none ↔ ¬ (g ∈ S ∧ p ∈ g.bag) := by
  refine ⟨fun heq hact => ?_, Atlas.canon_apply_of_inactive _⟩
  obtain ⟨hg, hp⟩ := hact
  rw [Atlas.canon_apply_of_active _ hg hp] at heq
  obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp (h.total g hg p hp)
  rw [hpl] at heq
  exact Option.some_ne_none _ heq

/-- **Canon-of-closed-atlas: `isSome` iff active**: complement of
Tick 1892. For a closed atlas, the canon-projected query is
`isSome` iff the query is active. Direct via De Morgan + Option
isSome/ne-none identity. -/
theorem Atlas.IsClosedOn.canon_apply_isSome_iff_active
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g : GameState} {p : Piece} :
    ((A.canon S) g p).isSome ↔ g ∈ S ∧ p ∈ g.bag := by
  rw [Option.isSome_iff_ne_none, ne_eq,
      h.canon_apply_eq_none_iff_inactive, not_not]

/-- **Closed atlas is total at any state in S**: extracts the
totality predicate at a specific state. Named accessor matching
the `IsTotalAt` convention. -/
theorem Atlas.IsClosedOn.isTotalAt_of_mem {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    {g : GameState} (hg : g ∈ S) :
    A.IsTotalAt g :=
  fun p hp => h.total g hg p hp

/-- **Canonicalisation preserves `toSolver` at active queries**:
for any active `(g, p)` (`g ∈ S, p ∈ g.bag`),
`(A.canon S).toSolver g p = A.toSolver g p`. The canon operation
strips inactive-query values but leaves the materialised solver
unchanged at active queries. Direct from
`Atlas.canon_apply_of_active` + `unfold Atlas.toSolver`. -/
theorem Atlas.canon_toSolver_apply_of_active {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} {g : GameState}
    (hg : g ∈ S) {p : Piece} (hp : p ∈ g.bag) :
    (A.canon S).toSolver g p = A.toSolver g p := by
  unfold Atlas.toSolver
  rw [Atlas.canon_apply_of_active _ hg hp]

/-- The **empty atlas**: returns `none` for every query. The trivial atlas
that certifies nothing — closed only on `∅`, and canonicalises to itself.
The named identity element for `unionOn`-style construction (the
construction's base case). -/
def Atlas.empty (cfg : GameConfig) : Atlas cfg := fun _ _ => none

@[simp] theorem Atlas.empty_apply (cfg : GameConfig) (g : GameState) (p : Piece) :
    (Atlas.empty cfg) g p = none := rfl

/-- **Restricting any atlas to the empty state set yields the empty
atlas**: `A.restrict ∅ = Atlas.empty cfg`. Trivial: at every g,
`g ∉ ∅`, so `restrict_apply_of_notMem` gives `none`, matching
`Atlas.empty`'s constant `none`. -/
theorem Atlas.restrict_empty {cfg : GameConfig} (A : Atlas cfg) :
    A.restrict ∅ = Atlas.empty cfg := by
  funext g p
  rw [Atlas.restrict_apply_of_notMem _ (Finset.notMem_empty g)]
  rfl

/-- **Restricting the empty atlas to any state set yields the empty
atlas**: `(Atlas.empty cfg).restrict S = Atlas.empty cfg`. The empty
atlas is invariant under restriction. Parallel to `Atlas.restrict_empty`
but with the empty-atlas argument instead of empty-set. Proof: case-
split on `g ∈ S` — both branches give `none` (active uses `empty_apply`,
inactive uses `restrict_apply_of_notMem`). -/
theorem Atlas.empty_restrict {cfg : GameConfig} (S : Finset GameState) :
    (Atlas.empty cfg).restrict S = Atlas.empty cfg := by
  funext g p
  by_cases h : g ∈ S
  · rw [Atlas.restrict_apply_of_mem _ h, Atlas.empty_apply]
  · rw [Atlas.restrict_apply_of_notMem _ h]
    rfl

/-- **Empty atlas on left of unionOn reduces to right's restriction
to S₂ \ S₁**: `(Atlas.empty cfg).unionOn B S₁ S₂ = B.restrict (S₂ \
S₁)`. The empty left atlas plus the priority dispatch means only B's
values at points in S₂ but NOT S₁ survive. Generalises Tick (line
1046)'s `unionOn_empty_left` (state-set-empty case) to the atlas-empty
case. Proof: pointwise via `funext` + 3-way case on g ∈ S₁/S₂. -/
theorem Atlas.empty_unionOn {cfg : GameConfig} (B : Atlas cfg)
    (S₁ S₂ : Finset GameState) :
    (Atlas.empty cfg).unionOn B S₁ S₂ = B.restrict (S₂ \ S₁) := by
  funext g p
  by_cases h₁ : g ∈ S₁
  · have h_sdiff : g ∉ S₂ \ S₁ := fun h => (Finset.mem_sdiff.mp h).2 h₁
    rw [Atlas.unionOn_apply_of_mem_left _ _ h₁ p,
        Atlas.empty_apply,
        Atlas.restrict_apply_of_notMem _ h_sdiff]
  · by_cases h₂ : g ∈ S₂
    · have h_sdiff : g ∈ S₂ \ S₁ := Finset.mem_sdiff.mpr ⟨h₂, h₁⟩
      rw [Atlas.unionOn_apply_of_mem_right _ _ h₁ h₂ p,
          Atlas.restrict_apply_of_mem _ h_sdiff]
    · have h_sdiff : g ∉ S₂ \ S₁ := fun h => h₂ (Finset.mem_sdiff.mp h).1
      rw [Atlas.unionOn_apply_of_notMem _ _ h₁ h₂ p,
          Atlas.restrict_apply_of_notMem _ h_sdiff]

/-- **Empty atlas on right of unionOn reduces to left's restriction
to S₁**: `A.unionOn (Atlas.empty cfg) S₁ S₂ = A.restrict S₁`. The
empty right atlas means only A's values at S₁ survive (priority left).
The result doesn't depend on S₂ at all. Symmetric to Tick 1747. -/
theorem Atlas.unionOn_empty {cfg : GameConfig} (A : Atlas cfg)
    (S₁ S₂ : Finset GameState) :
    A.unionOn (Atlas.empty cfg) S₁ S₂ = A.restrict S₁ := by
  funext g p
  by_cases h₁ : g ∈ S₁
  · rw [Atlas.unionOn_apply_of_mem_left _ _ h₁ p,
        Atlas.restrict_apply_of_mem _ h₁]
  · by_cases h₂ : g ∈ S₂
    · rw [Atlas.unionOn_apply_of_mem_right _ _ h₁ h₂ p,
          Atlas.empty_apply,
          Atlas.restrict_apply_of_notMem _ h₁]
    · rw [Atlas.unionOn_apply_of_notMem _ _ h₁ h₂ p,
          Atlas.restrict_apply_of_notMem _ h₁]

/-- **Canon of empty is empty**: canonicalising the all-`none` atlas at any
state set yields the all-`none` atlas. Trivially: every query is `none`,
so the canon's filter doesn't matter — output is `none` either way. -/
theorem Atlas.empty_canon (cfg : GameConfig) (S : Finset GameState) :
    (Atlas.empty cfg).canon S = Atlas.empty cfg := by
  funext g p
  by_cases h : g ∈ S ∧ p ∈ g.bag
  · rw [Atlas.canon_apply_of_active _ h.1 h.2]
  · rw [Atlas.canon_apply_of_inactive _ h]
    rfl

/-- **Canon at empty state set is empty**: for any atlas A,
`A.canon ∅ = Atlas.empty cfg`. The canon filter at `∅` excludes
every query (since `g ∈ ∅` is false), so all outputs are `none`.
Companion to `Atlas.empty_canon` (which fixes the atlas to empty;
this fixes the state set to empty). -/
theorem Atlas.canon_empty_set {cfg : GameConfig} (A : Atlas cfg) :
    A.canon ∅ = Atlas.empty cfg := by
  funext g p
  rw [Atlas.canon_apply_of_inactive _ (fun h => Finset.notMem_empty g h.1),
      Atlas.empty_apply]

/-- **The empty atlas's domain is exactly the empty-bag states**: for any
candidate set `S`, `(Atlas.empty cfg).dom S` consists of those `g ∈ S` with
`g.bag = ∅`. Because `Atlas.empty` returns `none` everywhere, `IsTotalAt g`
holds only when there are no in-bag pieces to fail on (i.e., `g.bag = ∅`).
For reachable states (which always have nonempty bags), the empty atlas's
dom is empty. -/
theorem Atlas.empty_dom (cfg : GameConfig) (S : Finset GameState) :
    (Atlas.empty cfg).dom S = S.filter (fun g => g.bag = ∅) := by
  unfold Atlas.dom
  apply Finset.filter_congr
  intro g _hg
  unfold Atlas.IsTotalAt
  refine ⟨fun hT => ?_, fun hempty p hp => ?_⟩
  · by_contra hne
    obtain ⟨p, hp⟩ := Finset.nonempty_iff_ne_empty.mpr hne
    have := hT p hp
    simp at this
  · rw [hempty] at hp
    exact (Finset.notMem_empty p hp).elim

/-- **Empty atlas certifies nothing on bag-nonempty states**: when every
state in `S` has nonempty bag, `(Atlas.empty cfg).dom S = ∅`. Direct
corollary of `empty_dom`: the filter predicate `g.bag = ∅` fails for every
`g ∈ S`, so the filter yields the empty Finset. -/
theorem Atlas.empty_dom_eq_empty_of_bag_nonempty {cfg : GameConfig}
    {S : Finset GameState} (h : ∀ g ∈ S, g.bag.Nonempty) :
    (Atlas.empty cfg).dom S = ∅ := by
  rw [Atlas.empty_dom]
  apply Finset.filter_eq_empty_iff.mpr
  intro g hg hbag
  exact (h g hg).ne_empty hbag

/-- **Empty atlas certifies nothing on reachable states**: specialisation of
`empty_dom_eq_empty_of_bag_nonempty` to reachable states, using
`Reachable.bag_nonempty`. The concrete operational form for M4 work — if
the candidate set consists of reachable states (the practical case), the
empty atlas's dom is exactly empty. -/
theorem Atlas.empty_dom_eq_empty_of_reachable {cfg : GameConfig}
    {S : Finset GameState} (h : ∀ g ∈ S, Reachable cfg g) :
    (Atlas.empty cfg).dom S = ∅ :=
  Atlas.empty_dom_eq_empty_of_bag_nonempty
    (fun g hg => (h g hg).bag_nonempty)

/-- The bag-projection of an adversarial step: drops the played piece. The
chosen placement only affects the resulting board, not the bag. Building
block for multi-step chain bounds (`AdversarialClosedCycle.states_card_*`). -/
@[simp] theorem adversarialStep_bag (cfg : GameConfig) (g : GameState)
    (p : Piece) (pl : Placement) :
    (adversarialStep cfg g p pl).bag = g.bag.draw p := rfl

@[simp] theorem adversarialStep_board (cfg : GameConfig) (g : GameState)
    (p : Piece) (pl : Placement) :
    (adversarialStep cfg g p pl).board =
      Placement.applyStep cfg g.board ({pl with piece := p}) := rfl

/-- Post-`adversarialStep` bag card is ≤ pre-step or refills to 7. -/
theorem adversarialStep_bag_card_le_or_eq_seven (cfg : GameConfig)
    (g : GameState) (p : Piece) (pl : Placement) :
    (adversarialStep cfg g p pl).bag.card ≤ g.bag.card ∨
      (adversarialStep cfg g p pl).bag.card = 7 := by
  rw [adversarialStep_bag]
  exact Bag.draw_card_le_card_or_eq_seven g.bag p

/-- Post-`adversarialStep` bag card is positive. -/
theorem adversarialStep_bag_card_pos (cfg : GameConfig) (g : GameState)
    (p : Piece) (pl : Placement) :
    0 < (adversarialStep cfg g p pl).bag.card := by
  rw [adversarialStep_bag]; exact Bag.draw_card_pos g.bag p

/-- Post-`adversarialStep` bag card is ≤ 7. -/
theorem adversarialStep_bag_card_le_seven (cfg : GameConfig)
    (g : GameState) (p : Piece) (pl : Placement) :
    (adversarialStep cfg g p pl).bag.card ≤ 7 :=
  Bag.card_le_seven _

/-- Bundled `[1, 7]` interval for adversarial-step bag card. -/
theorem adversarialStep_bag_card_interval (cfg : GameConfig)
    (g : GameState) (p : Piece) (pl : Placement) :
    1 ≤ (adversarialStep cfg g p pl).bag.card ∧
    (adversarialStep cfg g p pl).bag.card ≤ 7 :=
  ⟨adversarialStep_bag_card_pos cfg g p pl,
   adversarialStep_bag_card_le_seven cfg g p pl⟩

/-- Post-adversarial-step bag is never empty. -/
theorem adversarialStep_bag_ne_empty (cfg : GameConfig) (g : GameState)
    (p : Piece) (pl : Placement) :
    (adversarialStep cfg g p pl).bag ≠ ∅ := by
  rw [adversarialStep_bag]; exact Bag.draw_ne_empty g.bag p

/-- `adversarialStep` factors through `GameState.step` (after substituting
the announced piece into the placement). This is the connection between
the adversarial-play API and the single-state-machine `step`. -/
theorem adversarialStep_eq_step (cfg : GameConfig) (g : GameState) (p : Piece)
    (pl : Placement) :
    adversarialStep cfg g p pl = g.step cfg {pl with piece := p} := rfl

/-- **Reachability lifts through `adversarialStep`.** A foundational structural
fact: starting from a reachable, non-lost state `g`, performing an adversarial
step with piece `p ∈ g.bag` and any placement whose piece-overridden form is
`Valid cfg` again yields a reachable state. Combines `adversarialStep_eq_step`
with the standard `Reachable.step` constructor. Lets cycle/atlas arguments
move between the adversarial API and the inductive `Reachable` predicate. -/
theorem Reachable.adversarialStep {cfg : GameConfig} {g : GameState}
    (hr : Reachable cfg g) (hnl : ¬ g.lost cfg)
    {p : Piece} (hp : p ∈ g.bag)
    {pl : Placement} (hvalid : ({ pl with piece := p } : Placement).Valid cfg) :
    Reachable cfg (adversarialStep cfg g p pl) := by
  rw [adversarialStep_eq_step]
  refine Reachable.step hr ?_ hvalid hnl
  rw [Bag.canDraw_iff_mem]
  exact hp

/-- The successor bag is always nonempty (the bag refills when emptied). -/
theorem adversarialStep_bag_nonempty (cfg : GameConfig) (g : GameState)
    (p : Piece) (pl : Placement) :
    (adversarialStep cfg g p pl).bag.Nonempty := by
  rw [adversarialStep_bag]
  exact Bag.draw_nonempty g.bag p

/-- **Solver-reachable states**: states obtained by applying `σ` to `init`
along *any* piece sequence drawn from the bag at each step. Captures the
forward closure of `σ` from `init`, without committing to any particular
legal sequence. Used to define the set whose `safe`-membership we'll prove
via Knaster-Tarski. -/
inductive solverReachable {cfg : GameConfig} (σ : Solver cfg) : GameState → Prop
  | init : solverReachable σ GameState.init
  | step {g : GameState} (p : Piece) :
      solverReachable σ g → p ∈ g.bag →
      solverReachable σ (adversarialStep cfg g p (σ g p))


/-- **Init's successor is never init**: any one-step descendant has bag ≠
Bag.full, hence is distinct from init. -/
theorem init_step_ne_init {cfg : GameConfig} (p : Piece) (pl : Placement) :
    adversarialStep cfg GameState.init p pl ≠ GameState.init := by
  intro he
  have hbag : (adversarialStep cfg GameState.init p pl).bag =
              GameState.init.bag := congrArg _ he
  rw [adversarialStep_bag, GameState.init_bag] at hbag
  exact Bag.draw_full_ne_full p hbag

/-- **Init-successor bag size.** Any one-step descendant of `GameState.init`
has bag card exactly 6 (start with the full 7-bag, remove one, no refill).
Used in depth-stratified cycle-size bounds. -/
theorem init_step_bag_card (cfg : GameConfig) (p : Piece) (pl : Placement) :
    (adversarialStep cfg GameState.init p pl).bag.card = 6 := by
  rw [adversarialStep_bag]
  change (Bag.full.draw p).card = 6
  have h := Bag.card_draw_of_ge_two (bag := Bag.full) (p := p)
    (Bag.mem_full p) (by decide)
  rw [h]
  decide

/-- The state trace of solver `σ` against piece sequence `s` from `g0`. -/
def adversarialTrace (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece)
    (g0 : GameState) : ℕ → GameState
  | 0     => g0
  | n + 1 =>
      let g := adversarialTrace cfg σ s g0 n
      adversarialStep cfg g (s n) (σ g (s n))

@[simp] theorem adversarialTrace_zero (cfg : GameConfig) (σ : Solver cfg)
    (s : ℕ → Piece) (g0 : GameState) :
    adversarialTrace cfg σ s g0 0 = g0 := rfl

theorem adversarialTrace_succ (cfg : GameConfig) (σ : Solver cfg)
    (s : ℕ → Piece) (g0 : GameState) (n : ℕ) :
    adversarialTrace cfg σ s g0 (n + 1) =
      adversarialStep cfg (adversarialTrace cfg σ s g0 n) (s n)
        (σ (adversarialTrace cfg σ s g0 n) (s n)) := rfl

/-- **Trace concatenation / time-shift identity.** Running the trace for
`n + m` steps equals running it for `n` steps to reach `trace s g0 n`, then
running for `m` more steps from that state under the shifted piece sequence
`fun k => s (n + k)`. A foundational compositional law analogous to
`Survival.trace_add` for the single-player `trace`. Lets periodicity /
recurrence arguments reduce to "the same `m`-step walk starting from a
repeated state". -/
theorem adversarialTrace_add (cfg : GameConfig) (σ : Solver cfg)
    (s : ℕ → Piece) (g0 : GameState) (n m : ℕ) :
    adversarialTrace cfg σ s g0 (n + m) =
      adversarialTrace cfg σ (fun k => s (n + k))
        (adversarialTrace cfg σ s g0 n) m := by
  induction m with
  | zero => rfl
  | succ k ih =>
    show adversarialTrace cfg σ s g0 (n + (k + 1)) = _
    rw [show n + (k + 1) = (n + k) + 1 from rfl,
        adversarialTrace_succ, adversarialTrace_succ, ih]

/-- **Periodic-suffix periodicity.** If the trace has a recurrence at base
`b` and offset `d` (i.e. `trace s g0 b = trace s g0 (b + d)`) **and** the
piece sequence is itself `d`-periodic past `b` (i.e.
`s (b + k) = s (b + d + k)` for all `k`), then the entire `[b, ∞)`-suffix of
the trace is `d`-periodic: `trace s g0 (b + k) = trace s g0 (b + d + k)` for
all `k`. Proof: factor both traces via `adversarialTrace_add`, rewrite by the
recurrence hypothesis, and observe the two time-shifted sequences are
pointwise equal. The piece-sequence periodicity is essential — without it,
the same state evolves under different inputs and the periodicity breaks. -/
theorem adversarialTrace_periodic_of_periodic_suffix
    (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece) (g0 : GameState)
    {b d : ℕ}
    (htrace : adversarialTrace cfg σ s g0 b = adversarialTrace cfg σ s g0 (b + d))
    (hs : ∀ k, s (b + k) = s (b + d + k)) (k : ℕ) :
    adversarialTrace cfg σ s g0 (b + k) =
      adversarialTrace cfg σ s g0 (b + d + k) := by
  rw [adversarialTrace_add cfg σ s g0 b k,
      adversarialTrace_add cfg σ s g0 (b + d) k,
      htrace]
  congr 1
  funext j
  exact hs j

/-- **Iterated period.** Under the same hypotheses as
`adversarialTrace_periodic_of_periodic_suffix`, the trace at every base
`b + n * d` (for any `n : ℕ`) equals the trace at the base `b`. Proof:
induction on `n`; the inductive step applies Tick 1565's single-period
shift at offset `k * d`, then closes by IH and `ring` arithmetic on
`b + d + k * d = b + (k + 1) * d`. Establishes the trace is *honestly*
periodic on the lattice `{b + n * d | n : ℕ}` whenever both a trace
recurrence and a sequence period exist. -/
theorem adversarialTrace_iterated_period_of_periodic_suffix
    (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece) (g0 : GameState)
    {b d : ℕ}
    (htrace : adversarialTrace cfg σ s g0 b = adversarialTrace cfg σ s g0 (b + d))
    (hs : ∀ k, s (b + k) = s (b + d + k)) (n : ℕ) :
    adversarialTrace cfg σ s g0 (b + n * d) = adversarialTrace cfg σ s g0 b := by
  induction n with
  | zero => simp
  | succ k ih =>
    have step :=
      adversarialTrace_periodic_of_periodic_suffix cfg σ s g0 htrace hs (k * d)
    have arith : b + d + k * d = b + (k + 1) * d := by ring
    rw [arith] at step
    rw [← step, ih]

/-- **Trace prefix-locality / determinism.** The first `n` values of the
adversarial trace depend only on the first `n` entries of the piece sequence.
Equivalently: rewriting the piece sequence past index `n` does not affect
`trace … n`. Foundational structural property — captures that the trace
genuinely is a deterministic function of `g0`, `σ`, and the *past* of `s`.
Proof: induction on `n`; the `succ` case unfolds both traces with
`adversarialTrace_succ`, applies IH for the inner state, and rewrites the
single pointwise sequence equality at index `k`. -/
theorem adversarialTrace_congr (cfg : GameConfig) (σ : Solver cfg)
    (s s' : ℕ → Piece) (g0 : GameState) (n : ℕ)
    (h : ∀ k < n, s k = s' k) :
    adversarialTrace cfg σ s g0 n = adversarialTrace cfg σ s' g0 n := by
  induction n with
  | zero => rfl
  | succ k ih =>
    rw [adversarialTrace_succ, adversarialTrace_succ]
    have ihk : ∀ j < k, s j = s' j := fun j hj => h j (Nat.lt_succ_of_lt hj)
    have hk : s k = s' k := h k (Nat.lt_succ_self k)
    rw [ih ihk, hk]

/-- **Swap-past-`n` invariance** — strengthened form of
`adversarialTrace_congr`. If two piece sequences agree on `[0, n)`, then
their *entire prefixes up to and including `n`* coincide pointwise: for
every `k ≤ n`, `trace s g0 k = trace s' g0 k`. Proof: apply
`adversarialTrace_congr` at index `k`, weakening `j < k → j < n` via
`hkn`. Lets one freely modify the piece sequence past index `n` without
disturbing any trace value at indices `≤ n`. -/
theorem adversarialTrace_congr_le (cfg : GameConfig) (σ : Solver cfg)
    (s s' : ℕ → Piece) (g0 : GameState) {n k : ℕ} (hkn : k ≤ n)
    (h : ∀ j < n, s j = s' j) :
    adversarialTrace cfg σ s g0 k = adversarialTrace cfg σ s' g0 k :=
  adversarialTrace_congr cfg σ s s' g0 k
    (fun j hj => h j (lt_of_lt_of_le hj hkn))

/-- **Trace at the splice point matches the prefix trace.** The adversarial
trace at index `N` under the spliced sequence
`fun k => if k < N then s k else t (k - N)` equals the trace at `N` under
the prefix sequence `s`. Direct corollary of `adversarialTrace_congr_le`
(Tick 1568): the spliced sequence agrees with `s` on `[0, N)`. This is the
formal "splice point is invisible to the prefix" identity — the cycle-entry
state at index `N` is the same whether you continue with `s` or splice in
`t` past `N`. -/
theorem adversarialTrace_splice_at_N (cfg : GameConfig) (σ : Solver cfg)
    (s t : ℕ → Piece) (g0 : GameState) (N : ℕ) :
    adversarialTrace cfg σ (fun k => if k < N then s k else t (k - N)) g0 N
      = adversarialTrace cfg σ s g0 N := by
  apply adversarialTrace_congr_le cfg σ _ s g0 (le_refl N)
  intro j hj
  simp [hj]

/-- **Trace past the splice point matches the continuation.** For any `k ≥ N`,
the trace at index `k` under the spliced sequence
`fun j => if j < N then s j else t (j - N)` equals the trace at index `k - N`
under the *continuation* `t` started from the cycle-entry state
`trace cfg σ s g0 N`. The completion of the splice trace identity, combining
`adversarialTrace_add` (Tick 1564), `adversarialTrace_splice_at_N` (Tick 1573),
and the shifted-spliced-sequence-equals-continuation observation. -/
theorem adversarialTrace_splice_past_N (cfg : GameConfig) (σ : Solver cfg)
    (s t : ℕ → Piece) (g0 : GameState) {N k : ℕ} (hk : N ≤ k) :
    adversarialTrace cfg σ (fun j => if j < N then s j else t (j - N)) g0 k
      = adversarialTrace cfg σ t (adversarialTrace cfg σ s g0 N) (k - N) := by
  set j := k - N with hjdef
  have hkj : k = N + j := by omega
  set sp : ℕ → Piece := fun m => if m < N then s m else t (m - N) with hsp_def
  have hshift : (fun m => sp (N + m)) = t := by
    funext m
    change (if N + m < N then s (N + m) else t (N + m - N)) = t m
    rw [if_neg (Nat.not_lt.mpr (Nat.le_add_right N m)), Nat.add_sub_cancel_left]
  rw [hkj, adversarialTrace_add, hshift, adversarialTrace_splice_at_N]

/-- **Trace identity under splice/suffix rejoin.** The trace under the
spliced sequence "prefix `s` ++ suffix-of-`sp`-at-`N`" coincides with the
trace under `sp` itself, at every step `k`, provided `s` agrees with `sp` on
`[0, N)`. Direct corollary of `splice_suffix_eq`: the rejoined sequence is
function-equal to `sp`, so all derived quantities (including the entire
trace) match. Seals the splice/suffix isomorphism at the dynamics level. -/
theorem adversarialTrace_splice_suffix_eq
    (cfg : GameConfig) (σ : Solver cfg) (s sp : ℕ → Piece) (g0 : GameState)
    (N : ℕ) (h : ∀ k < N, s k = sp k) (k : ℕ) :
    adversarialTrace cfg σ
      (fun j => if j < N then s j else sp (N + (j - N))) g0 k
      = adversarialTrace cfg σ sp g0 k := by
  rw [splice_suffix_eq s sp N h]

/-- The bag of the adversarial trace from `init` matches the `bagAt`-computed
bag. (Sanity / bookkeeping lemma — confirms the two views of the bag agree.) -/
theorem adversarialTrace_bag (cfg : GameConfig) (σ : Solver cfg)
    (s : ℕ → Piece) (n : ℕ) :
    (adversarialTrace cfg σ s GameState.init n).bag = bagAt Bag.full s n := by
  induction n with
  | zero => rfl
  | succ k ih =>
      rw [adversarialTrace_succ, bagAt_succ]
      simp [adversarialStep, ih]

/-- **Trace bag matches `bagAt` from arbitrary start.** Generalises
`adversarialTrace_bag` from the `init` starting state to any `g0 : GameState`:
the bag of `trace cfg σ s g0 n` equals `bagAt g0.bag s n`. The
solver-independent half of the bag dynamics — bag evolution depends only on
the starting bag and the piece sequence, not on the board or solver. Needed
to argue legality of continuations starting from arbitrary cycle states. -/
theorem adversarialTrace_bag_from (cfg : GameConfig) (σ : Solver cfg)
    (s : ℕ → Piece) (g0 : GameState) (n : ℕ) :
    (adversarialTrace cfg σ s g0 n).bag = bagAt g0.bag s n := by
  induction n with
  | zero => rfl
  | succ k ih =>
      rw [adversarialTrace_succ, bagAt_succ]
      simp [adversarialStep, ih]

/-- **Closed atlas's materialised solver trace stays in S**:
multi-step lift of Tick 1818's single-step `toSolver_adversarialStep_mem`.
For any `g₀ ∈ S` and legal sequence from `g₀.bag`, the trace under
`A.toSolver` stays in `S` for all steps. Inductive proof: zero
step is `hg₀`; successor step applies `toSolver_adversarialStep_mem`
to the inductive hypothesis using `adversarialTrace_bag_from` for
bag-membership of `t k`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_mem {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    {g₀ : GameState} (hg₀ : g₀ ∈ S) {t : ℕ → Piece}
    (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    adversarialTrace cfg A.toSolver t g₀ n ∈ S := by
  induction n with
  | zero => exact hg₀
  | succ k ih =>
    rw [adversarialTrace_succ]
    apply h.toSolver_adversarialStep_mem ih
    rw [adversarialTrace_bag_from]
    exact hl k

/-- **Closed atlas's materialised solver trace is never lost**:
multi-step lift of Tick 1819's single-step `toSolver_adversarialStep_not_lost`.
Composes Tick 1821's `toSolver_adversarialTrace_mem` (trace stays
in S) with `h.not_lost` (S contains no lost states). -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_not_lost {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    {g₀ : GameState} (hg₀ : g₀ ∈ S) {t : ℕ → Piece}
    (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    ¬ (adversarialTrace cfg A.toSolver t g₀ n).lost cfg :=
  h.not_lost _ (h.toSolver_adversarialTrace_mem hg₀ hl n)

/-- **Bundled Atlas trace invariants**: 2-conjunct combining
trace-mem (Tick 1821) and trace-not_lost (Tick 1822). One-shot
certificate for every state along the materialised-solver trace.
Atlas-level parallel to M4's bundled trace invariant. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_mem_and_not_lost
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    adversarialTrace cfg A.toSolver t g₀ n ∈ S ∧
    ¬ (adversarialTrace cfg A.toSolver t g₀ n).lost cfg :=
  ⟨h.toSolver_adversarialTrace_mem hg₀ hl n,
   h.toSolver_adversarialTrace_not_lost hg₀ hl n⟩

/-- **Full 4-conjunct Atlas trace invariants**: combines trace-mem
(1821), trace-not-lost (1822), solver-piece (1877), solver-valid
(1878). Per-step full trace certificate at index `n`. Atlas-level
parallel to M4's full bundled trace invariants. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_full
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    adversarialTrace cfg A.toSolver t g₀ n ∈ S ∧
    ¬ (adversarialTrace cfg A.toSolver t g₀ n).lost cfg ∧
    (A.toSolver (adversarialTrace cfg A.toSolver t g₀ n) (t n)).piece = t n ∧
    (A.toSolver (adversarialTrace cfg A.toSolver t g₀ n) (t n)).Valid cfg := by
  have hmem := h.toSolver_adversarialTrace_mem hg₀ hl n
  have hnl := h.toSolver_adversarialTrace_not_lost hg₀ hl n
  have hbag : t n ∈ (adversarialTrace cfg A.toSolver t g₀ n).bag := by
    rw [adversarialTrace_bag_from]
    exact hl n
  exact ⟨hmem, hnl, h.toSolver_piece hmem hbag, h.toSolver_valid hmem hbag⟩

/-- **Init-specialised full 4-conjunct Atlas trace invariants**:
specialisation of Tick 1882 to `g₀ = init`. Per-step full trace
certificate from init at index `n`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_full_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg A.toSolver s GameState.init n ∈ S ∧
    ¬ (adversarialTrace cfg A.toSolver s GameState.init n).lost cfg ∧
    (A.toSolver (adversarialTrace cfg A.toSolver s GameState.init n)
      (s n)).piece = s n ∧
    (A.toSolver (adversarialTrace cfg A.toSolver s GameState.init n)
      (s n)).Valid cfg :=
  h.toSolver_adversarialTrace_full hinit hl n

/-- **`Set.range` form of Atlas trace invariant**: the full image
of the materialised-solver trace from `g₀ ∈ S` is contained in `S`.
`Set.range_subset_iff` form of Tick 1821. Atlas-level parallel to
cycle's `adversarialTrace_range_subset_states_from_mem`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_range_subset
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    Set.range (adversarialTrace cfg A.toSolver t g₀) ⊆ S :=
  Set.range_subset_iff.mpr (h.toSolver_adversarialTrace_mem hg₀ hl)

/-- **Atlas trace range is finite**: contained in the finite Finset
`S`, hence finite. Direct via `S.finite_toSet.subset` applied to
Tick 1824's range subset. Atlas-level parallel to the cycle's
`range_finite_from_mem`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_range_finite
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    (Set.range (adversarialTrace cfg A.toSolver t g₀)).Finite :=
  S.finite_toSet.subset (h.toSolver_adversarialTrace_range_subset hg₀ hl)

/-- **Pigeonhole on Atlas trace**: indices in `0..S.card` cannot
all be distinct under the materialised-solver trace (since the
trace lives in S of size `S.card`), so two indices share a trace
value. Atlas-level parallel to cycle's `adversarialTrace_pigeonhole_from_mem`
(Tick 1782). -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_pigeonhole
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    ∃ i j : ℕ, i ≤ S.card ∧ j ≤ S.card ∧ i ≠ j ∧
      adversarialTrace cfg A.toSolver t g₀ i =
        adversarialTrace cfg A.toSolver t g₀ j := by
  set N := S.card
  have hmap : ∀ k ∈ Finset.range (N + 1),
      adversarialTrace cfg A.toSolver t g₀ k ∈ S :=
    fun k _ => h.toSolver_adversarialTrace_mem hg₀ hl k
  have hcard : N < (Finset.range (N + 1)).card := by
    rw [Finset.card_range]; omega
  obtain ⟨i, hi, j, hj, hne, heq⟩ :=
    Finset.exists_ne_map_eq_of_card_lt_of_maps_to hcard hmap
  rw [Finset.mem_range] at hi hj
  exact ⟨i, j, by omega, by omega, hne, heq⟩

/-- **Ordered pigeonhole on Atlas trace**: same as Tick 1826 but
with `i < j`. Atlas-level parallel to cycle's
`adversarialTrace_pigeonhole_lt_from_mem` (Tick 1783). -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_pigeonhole_lt
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    ∃ i j : ℕ, i < j ∧ j ≤ S.card ∧
      adversarialTrace cfg A.toSolver t g₀ i =
        adversarialTrace cfg A.toSolver t g₀ j := by
  obtain ⟨i, j, hi, hj, hne, heq⟩ :=
    h.toSolver_adversarialTrace_pigeonhole hg₀ hl
  rcases lt_or_gt_of_ne hne with hlt | hgt
  · exact ⟨i, j, hlt, hj, heq⟩
  · exact ⟨j, i, hgt, hi, heq.symm⟩

/-- **Ordered pigeonhole with state membership on Atlas trace**:
same as Tick 1827 but the common trace state lies in `S`. Atlas-
level parallel to cycle's
`adversarialTrace_pigeonhole_lt_in_states_from_mem` (Tick 1784). -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_pigeonhole_lt_in_S
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    ∃ i j : ℕ, i < j ∧ j ≤ S.card ∧
      adversarialTrace cfg A.toSolver t g₀ i =
        adversarialTrace cfg A.toSolver t g₀ j ∧
      adversarialTrace cfg A.toSolver t g₀ i ∈ S := by
  obtain ⟨i, j, hij, hj, heq⟩ :=
    h.toSolver_adversarialTrace_pigeonhole_lt hg₀ hl
  exact ⟨i, j, hij, hj, heq, h.toSolver_adversarialTrace_mem hg₀ hl i⟩

/-- **Base + period form of Atlas pigeonhole**: from `i < j` with
shared trace value, repackage as
`∃ base, period ≥ 1, trace base = trace (base + period)`.
The period is at most `S.card`. Atlas-level parallel to cycle's
`adversarialTrace_pigeonhole_base_period_from_mem` (Tick 1785). -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_pigeonhole_base_period
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    ∃ base period : ℕ, 1 ≤ period ∧ period ≤ S.card ∧
      adversarialTrace cfg A.toSolver t g₀ base =
        adversarialTrace cfg A.toSolver t g₀ (base + period) := by
  obtain ⟨i, j, hij, hj, heq⟩ :=
    h.toSolver_adversarialTrace_pigeonhole_lt hg₀ hl
  refine ⟨i, j - i, ?_, ?_, ?_⟩
  · omega
  · omega
  · rw [Nat.add_sub_cancel' (le_of_lt hij)]; exact heq

/-- **Bounded base + period form of Atlas pigeonhole**: extends
Tick 1829 with `base < S.card`. Atlas-level parallel to cycle's
`adversarialTrace_pigeonhole_base_period_lt_from_mem` (Tick 1786). -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_pigeonhole_base_period_lt
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    ∃ base period : ℕ, base < S.card ∧
      1 ≤ period ∧ period ≤ S.card ∧
      adversarialTrace cfg A.toSolver t g₀ base =
        adversarialTrace cfg A.toSolver t g₀ (base + period) := by
  obtain ⟨i, j, hij, hj, heq⟩ :=
    h.toSolver_adversarialTrace_pigeonhole_lt hg₀ hl
  refine ⟨i, j - i, ?_, ?_, ?_, ?_⟩
  · omega
  · omega
  · omega
  · rw [Nat.add_sub_cancel' (le_of_lt hij)]; exact heq

/-- **Base + period + base-state-in-S form of Atlas pigeonhole**:
extends Tick 1830 by witnessing that the base state lies in `S`.
Atlas-level parallel to cycle's
`adversarialTrace_pigeonhole_base_period_in_states_from_mem` (Tick
1787). -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_pigeonhole_base_period_in_S
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) {g₀ : GameState} (hg₀ : g₀ ∈ S)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) :
    ∃ base period : ℕ, base < S.card ∧
      1 ≤ period ∧ period ≤ S.card ∧
      adversarialTrace cfg A.toSolver t g₀ base =
        adversarialTrace cfg A.toSolver t g₀ (base + period) ∧
      adversarialTrace cfg A.toSolver t g₀ base ∈ S := by
  obtain ⟨base, period, hbase, hpos, hperiod, heq⟩ :=
    h.toSolver_adversarialTrace_pigeonhole_base_period_lt hg₀ hl
  exact ⟨base, period, hbase, hpos, hperiod, heq,
    h.toSolver_adversarialTrace_mem hg₀ hl base⟩

/-- **Atlas-level init pigeonhole**: specialisation of Tick 1826
to `g₀ = init`. Uses `init.bag = Bag.full` to coerce `LegalSequence`
to `LegalSequenceFrom init.bag` (these are definitionally equal). -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_pigeonhole_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ i j : ℕ, i ≤ S.card ∧ j ≤ S.card ∧ i ≠ j ∧
      adversarialTrace cfg A.toSolver s GameState.init i =
        adversarialTrace cfg A.toSolver s GameState.init j :=
  h.toSolver_adversarialTrace_pigeonhole hinit hl

/-- **Atlas-level ordered init pigeonhole**: `i < j` form,
specialisation of Tick 1827 to `g₀ = init`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_pigeonhole_lt_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ i j : ℕ, i < j ∧ j ≤ S.card ∧
      adversarialTrace cfg A.toSolver s GameState.init i =
        adversarialTrace cfg A.toSolver s GameState.init j :=
  h.toSolver_adversarialTrace_pigeonhole_lt hinit hl

/-- **Atlas-level ordered init pigeonhole with state-mem**:
specialisation of Tick 1828 to `g₀ = init`. The recurrent state
lies in `S`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_pigeonhole_lt_in_S_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ i j : ℕ, i < j ∧ j ≤ S.card ∧
      adversarialTrace cfg A.toSolver s GameState.init i =
        adversarialTrace cfg A.toSolver s GameState.init j ∧
      adversarialTrace cfg A.toSolver s GameState.init i ∈ S :=
  h.toSolver_adversarialTrace_pigeonhole_lt_in_S hinit hl

/-- **Atlas-level base + period init pigeonhole**: explicit cycle-
recurrence form from init. Specialisation of Tick 1829 to
`g₀ = init`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_pigeonhole_base_period_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ base period : ℕ, 1 ≤ period ∧ period ≤ S.card ∧
      adversarialTrace cfg A.toSolver s GameState.init base =
        adversarialTrace cfg A.toSolver s GameState.init (base + period) :=
  h.toSolver_adversarialTrace_pigeonhole_base_period hinit hl

/-- **Atlas-level bounded base + period init pigeonhole**: extends
Tick 1854 with `base < S.card`. Specialisation of Tick 1830 to
`g₀ = init`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_pigeonhole_base_period_lt_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ base period : ℕ, base < S.card ∧
      1 ≤ period ∧ period ≤ S.card ∧
      adversarialTrace cfg A.toSolver s GameState.init base =
        adversarialTrace cfg A.toSolver s GameState.init (base + period) :=
  h.toSolver_adversarialTrace_pigeonhole_base_period_lt hinit hl

/-- **Atlas-level fully-bundled init base + period pigeonhole**:
extends Tick 1855 with `base-state ∈ S`. Specialisation of Tick
1831 to `g₀ = init`. -/
theorem Atlas.IsClosedOn.toSolver_adversarialTrace_pigeonhole_base_period_in_S_of_init
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ base period : ℕ, base < S.card ∧
      1 ≤ period ∧ period ≤ S.card ∧
      adversarialTrace cfg A.toSolver s GameState.init base =
        adversarialTrace cfg A.toSolver s GameState.init (base + period) ∧
      adversarialTrace cfg A.toSolver s GameState.init base ∈ S :=
  h.toSolver_adversarialTrace_pigeonhole_base_period_in_S hinit hl

/-- Extend a legal sequence at position `n` by inserting piece `p`, then
continuing legally from the post-draw bag `postBag` (= `(bagAt initBag s n).draw p`
in practice). The values for `k < n` agree with the original `s`. -/
noncomputable def extendLegalAt (s : ℕ → Piece) (n : ℕ) (p : Piece)
    (postBag : Bag) (hne : postBag.Nonempty) : ℕ → Piece := fun k =>
  if k < n then s k
  else if k = n then p
  else someLegalFrom postBag hne (k - n - 1)

@[simp] theorem extendLegalAt_below {s n p postBag hne k} (hk : k < n) :
    extendLegalAt s n p postBag hne k = s k := by
  unfold extendLegalAt; rw [if_pos hk]

@[simp] theorem extendLegalAt_at {s n p postBag hne} :
    extendLegalAt s n p postBag hne n = p := by
  unfold extendLegalAt; rw [if_neg (lt_irrefl n), if_pos rfl]

theorem extendLegalAt_above {s n p postBag hne k} (hk : n < k) :
    extendLegalAt s n p postBag hne k =
      someLegalFrom postBag hne (k - n - 1) := by
  unfold extendLegalAt
  rw [if_neg (not_lt.mpr (le_of_lt hk)), if_neg (Nat.ne_of_lt hk).symm]

/-- For `k ≤ n`, the bag of the extension matches the bag of the original. -/
theorem bagAt_extendLegalAt_le (initBag : Bag) (s : ℕ → Piece) (n : ℕ)
    (p : Piece) (postBag : Bag) (hne : postBag.Nonempty)
    (k : ℕ) (hk : k ≤ n) :
    bagAt initBag (extendLegalAt s n p postBag hne) k =
      bagAt initBag s k :=
  bagAt_eq_of_eq_below initBag _ s n
    (fun _ hi => extendLegalAt_below hi) k hk

/-- For `k > n`, the bag of the extension matches the bag of the `postBag`
sequence shifted by `n + 1` steps. -/
theorem bagAt_extendLegalAt_above (initBag : Bag) (s : ℕ → Piece) (n : ℕ)
    (p : Piece) (postBag : Bag) (hne : postBag.Nonempty)
    (hpost : postBag = (bagAt initBag s n).draw p) (m : ℕ) :
    bagAt initBag (extendLegalAt s n p postBag hne) (n + 1 + m) =
      bagAt postBag (someLegalFrom postBag hne) m := by
  induction m with
  | zero =>
    change bagAt initBag (extendLegalAt s n p postBag hne) (n + 1) = postBag
    rw [bagAt_succ, bagAt_extendLegalAt_le _ _ _ _ _ _ n (le_refl n),
        extendLegalAt_at, hpost.symm]
  | succ k ih =>
    change bagAt initBag (extendLegalAt s n p postBag hne) (n + 1 + k + 1) =
         bagAt postBag (someLegalFrom postBag hne) (k + 1)
    rw [bagAt_succ, ih, bagAt_succ]
    have hkn : n < n + 1 + k := by omega
    rw [extendLegalAt_above hkn]
    have hred : n + 1 + k - n - 1 = k := by omega
    rw [hred]

/-- **The extension is a legal sequence**: agrees with the original on
`[0..n)`, inserts `p` at `n` (which is in the bag by `hp`), and continues
legally from `postBag = (bagAt initBag s n).draw p`. -/
theorem extendLegalAt_legal {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) {n : ℕ} {p : Piece}
    (hp : p ∈ bagAt initBag s n) {postBag : Bag} {hne : postBag.Nonempty}
    (hpost : postBag = (bagAt initBag s n).draw p) :
    LegalSequenceFrom initBag (extendLegalAt s n p postBag hne) := by
  intro k
  rcases lt_trichotomy k n with hkn | hkn | hkn
  · rw [extendLegalAt_below hkn,
        bagAt_extendLegalAt_le _ _ _ _ _ _ k (le_of_lt hkn)]
    exact hl k
  · subst hkn
    rw [extendLegalAt_at, bagAt_extendLegalAt_le _ _ _ _ _ _ k (le_refl k)]
    exact hp
  · obtain ⟨m, rfl⟩ : ∃ m, k = n + 1 + m := ⟨k - n - 1, by omega⟩
    rw [extendLegalAt_above (by omega),
        bagAt_extendLegalAt_above _ _ _ _ _ _ hpost]
    have hred : n + 1 + m - n - 1 = m := by omega
    rw [hred]
    exact someLegalFrom_legal postBag hne m

/-- The adversarial trace at step `k` only depends on the first `k` values
of the piece sequence — by induction on `k` using `adversarialTrace_succ`. -/
theorem adversarialTrace_eq_of_eq_below {cfg : GameConfig} (σ : Solver cfg)
    (s s' : ℕ → Piece) (g0 : GameState) (n : ℕ)
    (h : ∀ i < n, s i = s' i) :
    ∀ k ≤ n, adversarialTrace cfg σ s g0 k = adversarialTrace cfg σ s' g0 k := by
  intro k hk
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [adversarialTrace_succ, adversarialTrace_succ, ih (by omega), h k (by omega)]

/-- Every state along an adversarial trace from `init` is `solverReachable`. -/
theorem adversarialTrace_solverReachable {cfg : GameConfig} (σ : Solver cfg)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    solverReachable σ (adversarialTrace cfg σ s GameState.init n) := by
  induction n with
  | zero => exact solverReachable.init
  | succ k ih =>
    rw [adversarialTrace_succ]
    have hp : s k ∈ (adversarialTrace cfg σ s GameState.init k).bag := by
      rw [adversarialTrace_bag]; exact hl k
    exact solverReachable.step (s k) ih hp

/-- **From-mem variant of `adversarialTrace_solverReachable`**:
generalizes from `GameState.init` to any solverReachable start
`g₀`, with `LegalSequenceFrom g₀.bag t`. Adversarial trace from
`g₀` retains solverReachability at every step. Solver-agnostic. -/
theorem adversarialTrace_solverReachable_from {cfg : GameConfig}
    {σ : Solver cfg} {g₀ : GameState} (hg₀ : solverReachable σ g₀)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    solverReachable σ (adversarialTrace cfg σ t g₀ n) := by
  induction n with
  | zero => exact hg₀
  | succ k ih =>
    rw [adversarialTrace_succ]
    have hp : t k ∈ (adversarialTrace cfg σ t g₀ k).bag := by
      rw [adversarialTrace_bag_from]
      have := hl k
      rwa [Bag.canDraw_iff_mem] at this
    exact solverReachable.step (t k) ih hp

/-- Every `solverReachable` state has a nonempty bag. -/
theorem solverReachable_bag_nonempty {cfg : GameConfig} {σ : Solver cfg}
    {g : GameState} (h : solverReachable σ g) : g.bag.Nonempty := by
  induction h with
  | init => exact Bag.full_nonempty
  | step p _ _ _ => exact adversarialStep_bag_nonempty _ _ _ _

/-- **`solverReachable` of materialised solver stays in S for
init-containing closed atlases**: if `A.IsClosedOn cfg S ∧ init ∈
S`, every `solverReachable A.toSolver g` lies in S. Direct
induction on `solverReachable` using Tick 1818's step-mem at the
inductive step. -/
theorem Atlas.IsClosedOn.toSolver_solverReachable_mem
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {g : GameState} (hg : solverReachable A.toSolver g) :
    g ∈ S := by
  induction hg with
  | init => exact hinit
  | step p _ hp ih => exact h.toSolver_adversarialStep_mem ih hp

/-- **`solverReachable` set under materialised solver is finite
for init-containing closed atlases**: bounded by the finite
state set S. Direct via Tick 1895 + `S.finite_toSet`. -/
theorem Atlas.IsClosedOn.toSolver_solverReachable_finite
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    {g | solverReachable A.toSolver g}.Finite :=
  S.finite_toSet.subset (fun _ hg => h.toSolver_solverReachable_mem hinit hg)

/-- **`solverReachable` of materialised solver is never lost**:
for init-containing closed atlases, every `solverReachable` state
is non-lost. One-liner via `h.not_lost _` applied to Tick 1895
(solverReachable ⊆ S). -/
theorem Atlas.IsClosedOn.toSolver_solverReachable_not_lost
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S)
    {g : GameState} (hg : solverReachable A.toSolver g) :
    ¬ g.lost cfg :=
  h.not_lost _ (h.toSolver_solverReachable_mem hinit hg)

/-- Every `solverReachable` state has a drawable piece. -/
theorem solverReachable_exists_canDraw {cfg : GameConfig} {σ : Solver cfg}
    {g : GameState} (h : solverReachable σ g) :
    ∃ p : Piece, g.bag.canDraw p :=
  Bag.exists_canDraw_of_nonempty (solverReachable_bag_nonempty h)

/-- Every `solverReachable` state has bag card ≥ 1. -/
theorem solverReachable_bag_card_pos {cfg : GameConfig} {σ : Solver cfg}
    {g : GameState} (h : solverReachable σ g) : 0 < g.bag.card :=
  Finset.card_pos.mpr (solverReachable_bag_nonempty h)

/-- Every `solverReachable` state has bag card ≤ 7. -/
theorem solverReachable_bag_card_le_seven {cfg : GameConfig}
    {σ : Solver cfg} {g : GameState} (_h : solverReachable σ g) :
    g.bag.card ≤ 7 := Bag.card_le_seven _

/-- Bundled `[1, 7]` bag-card interval for `solverReachable`. -/
theorem solverReachable_bag_card_interval {cfg : GameConfig}
    {σ : Solver cfg} {g : GameState} (h : solverReachable σ g) :
    1 ≤ g.bag.card ∧ g.bag.card ≤ 7 :=
  ⟨solverReachable_bag_card_pos h, solverReachable_bag_card_le_seven h⟩

/-- **The empty-bag empty-board state is never `solverReachable`** —
ruled out by `solverReachable_bag_nonempty`. Holds for *any* solver,
without needing `SolvesTetrisValid`. -/
theorem empty_bag_empty_board_not_solverReachable {cfg : GameConfig}
    {σ : Solver cfg} :
    ¬ solverReachable σ (⟨Board.empty, (∅ : Bag)⟩ : GameState) := by
  intro hr
  rcases solverReachable_bag_nonempty hr with ⟨_, hmem⟩
  exact absurd hmem (Finset.notMem_empty _)

/-- Any `solverReachable` state differs from the empty-bag empty-board
state. Contrapositive of `empty_bag_empty_board_not_solverReachable`. -/
theorem solverReachable.ne_empty_bag_empty_board {cfg : GameConfig}
    {σ : Solver cfg} {g : GameState} (h : solverReachable σ g) :
    g ≠ ⟨Board.empty, (∅ : Bag)⟩ := by
  intro heq
  exact empty_bag_empty_board_not_solverReachable (heq ▸ h)

/-- `solverReachable` state's bag is a subset of `Bag.full`. -/
theorem solverReachable.bag_subset_full {cfg : GameConfig}
    {σ : Solver cfg} {g : GameState} (_h : solverReachable σ g) :
    g.bag ⊆ Bag.full :=
  GameState.bag_subset_full g

/-- `solverReachable` state's bag dichotomy: full or card < 7. -/
theorem solverReachable.bag_eq_full_or_card_lt_seven {cfg : GameConfig}
    {σ : Solver cfg} {g : GameState} (_h : solverReachable σ g) :
    g.bag = Bag.full ∨ g.bag.card < 7 :=
  g.bag_eq_full_or_card_lt_seven

/-- `solverReachable` state's bag dichotomy: full or card ≤ 6. -/
theorem solverReachable.bag_eq_full_or_card_le_six {cfg : GameConfig}
    {σ : Solver cfg} {g : GameState} (_h : solverReachable σ g) :
    g.bag = Bag.full ∨ g.bag.card ≤ 6 :=
  g.bag_eq_full_or_card_le_six

/-- Dot-notation: `solverReachable` state has nonempty bag. -/
theorem solverReachable.bag_ne_empty {cfg : GameConfig} {σ : Solver cfg}
    {g : GameState} (h : solverReachable σ g) : g.bag ≠ ∅ :=
  (solverReachable_bag_nonempty h).ne_empty

/-- `solverReachable` state's bag is either ⊂ `full` or = `full`. -/
theorem solverReachable.bag_ssubset_full_or_eq_full {cfg : GameConfig}
    {σ : Solver cfg} {g : GameState} (h : solverReachable σ g) :
    g.bag ⊂ Bag.full ∨ g.bag = Bag.full := by
  rcases h.bag_eq_full_or_card_lt_seven with hf | hlt
  · right; exact hf
  · left
    rw [Finset.ssubset_iff_subset_ne]
    refine ⟨GameState.bag_subset_full _, ?_⟩
    intro heq
    rw [heq, Bag.full_card] at hlt
    omega

/-- Dot-notation: `bag.card = 7 ↔ bag = full` for `solverReachable`. -/
theorem solverReachable.bag_card_eq_seven_iff_bag_eq_full
    {cfg : GameConfig} {σ : Solver cfg} {g : GameState}
    (_h : solverReachable σ g) : g.bag.card = 7 ↔ g.bag = Bag.full :=
  Bag.card_eq_seven_iff_eq_full g.bag


/-- **Every state along an adversarial trace has a nonempty bag.**
Composes `adversarialTrace_solverReachable` with
`solverReachable_bag_nonempty`. -/
theorem adversarialTrace_bag_nonempty {cfg : GameConfig} (σ : Solver cfg)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace cfg σ s GameState.init n).bag.Nonempty :=
  solverReachable_bag_nonempty (adversarialTrace_solverReachable σ hl n)

/-- **Bag card is positive along any adversarial trace.** Numerical
form of `adversarialTrace_bag_nonempty`. -/
theorem adversarialTrace_bag_card_pos {cfg : GameConfig} (σ : Solver cfg)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    0 < (adversarialTrace cfg σ s GameState.init n).bag.card :=
  Finset.card_pos.mpr (adversarialTrace_bag_nonempty σ hl n)

/-- **Bag card is bounded by 7 along any adversarial trace.** -/
theorem adversarialTrace_bag_card_le_seven {cfg : GameConfig}
    (σ : Solver cfg) {s : ℕ → Piece} (_hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace cfg σ s GameState.init n).bag.card ≤ 7 :=
  Bag.card_le_seven _

/-- **Bundled `[1, 7]` bag-card interval along any adversarial trace.** -/
theorem adversarialTrace_bag_card_interval {cfg : GameConfig}
    (σ : Solver cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    1 ≤ (adversarialTrace cfg σ s GameState.init n).bag.card ∧
    (adversarialTrace cfg σ s GameState.init n).bag.card ≤ 7 :=
  ⟨adversarialTrace_bag_card_pos σ hl n,
   adversarialTrace_bag_card_le_seven σ hl n⟩

/-- The first-step state from `init` via any solver is `solverReachable`. -/
theorem solverReachable_init_step (cfg : GameConfig) (σ : Solver cfg)
    (p : Piece) :
    solverReachable σ (adversarialStep cfg GameState.init p
      (σ GameState.init p)) :=
  solverReachable.step p solverReachable.init
    (GameState.init_bag.symm ▸ Bag.mem_full p)

/-- **Every `solverReachable` state appears in some legal trace.** Proved
by induction: init via any legal sequence (Tick 111); step via the
`extendLegalAt` extension (Tick 120). -/
theorem solverReachable_exists_trace {cfg : GameConfig} {σ : Solver cfg}
    {g : GameState} (hr : solverReachable σ g) :
    ∃ s : ℕ → Piece, LegalSequence s ∧
      ∃ n, g = adversarialTrace cfg σ s GameState.init n := by
  induction hr with
  | init =>
    obtain ⟨s, hl⟩ := exists_legalSequence
    exact ⟨s, hl, 0, rfl⟩
  | @step g p _hr hp ih =>
    obtain ⟨s, hl, n, hgeq⟩ := ih
    let postBag := g.bag.draw p
    have hne : postBag.Nonempty := Bag.draw_nonempty _ _
    have hbag_eq : g.bag = bagAt Bag.full s n := by
      rw [hgeq, adversarialTrace_bag]
    have hp' : p ∈ bagAt Bag.full s n := hbag_eq ▸ hp
    have hpost : postBag = (bagAt Bag.full s n).draw p := by
      change g.bag.draw p = _
      rw [hbag_eq]
    refine ⟨extendLegalAt s n p postBag hne, extendLegalAt_legal hl hp' hpost,
            n + 1, ?_⟩
    rw [adversarialTrace_succ]
    have htr : adversarialTrace cfg σ (extendLegalAt s n p postBag hne)
                                  GameState.init n =
               adversarialTrace cfg σ s GameState.init n :=
      (adversarialTrace_eq_of_eq_below σ s
          (extendLegalAt s n p postBag hne) GameState.init n
          (fun i hi => (extendLegalAt_below hi).symm) n (le_refl n)).symm
    rw [htr, ← hgeq, extendLegalAt_at]

/-! ## The central conjecture -/

/-- **Solver `σ` solves Tetris** when, starting from `init`, it never loses
against any bag-legal piece sequence. -/
def SolvesTetris (cfg : GameConfig) (σ : Solver cfg) : Prop :=
  ∀ s : ℕ → Piece, LegalSequence s →
    ∀ n, ¬ (adversarialTrace cfg σ s GameState.init n).lost cfg

/-- **Solver `σ` solves Tetris from `g₀`** — generalises `SolvesTetris` to
an arbitrary starting state. The solver never loses against any
bag-legal piece sequence from `g₀.bag`. Reduces to `SolvesTetris` when
`g₀ = GameState.init`. -/
def SolvesTetrisFrom (cfg : GameConfig) (g₀ : GameState) (σ : Solver cfg) :
    Prop :=
  ∀ t : ℕ → Piece, LegalSequenceFrom g₀.bag t →
    ∀ n, ¬ (adversarialTrace cfg σ t g₀ n).lost cfg

/-- **`SolvesTetrisFrom init = SolvesTetris`**: when the starting state is
`init`, `SolvesTetrisFrom` reduces to `SolvesTetris` definitionally
(`init.bag = Bag.full` by `rfl`, and `LegalSequence := LegalSequenceFrom
Bag.full`). -/
theorem solvesTetrisFrom_init_iff_solvesTetris (cfg : GameConfig)
    (σ : Solver cfg) :
    SolvesTetrisFrom cfg GameState.init σ ↔ SolvesTetris cfg σ :=
  Iff.rfl

/-- **Closed atlas's materialised solver solves Tetris from any `g₀
∈ S`**: applies Tick 1822's trace-not_lost as the SolvesTetrisFrom
witness. Atlas-level parallel to M4's `M4Artifact.toSolver_solvesTetrisFrom`.
Direct from the definition: SolvesTetrisFrom = "∀ legal trace, ∀ n,
not-lost"; supply Tick 1822 pointwise. -/
theorem Atlas.IsClosedOn.toSolver_solvesTetrisFrom {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    {g₀ : GameState} (hg₀ : g₀ ∈ S) :
    SolvesTetrisFrom cfg g₀ A.toSolver :=
  fun _ hl n => h.toSolver_adversarialTrace_not_lost hg₀ hl n

/-- **Closed atlas containing init solves Tetris**: if `A.IsClosedOn
cfg S` and `init ∈ S`, then `A.toSolver` solves Tetris (init-based
no-loss). Specialises Tick 1832's `toSolver_solvesTetrisFrom` to
`g₀ = init` via the existing `solvesTetrisFrom_init_iff_solvesTetris`
biconditional. -/
theorem Atlas.IsClosedOn.toSolver_solvesTetris {cfg : GameConfig}
    {A : Atlas cfg} {S : Finset GameState} (h : A.IsClosedOn cfg S)
    (hinit : GameState.init ∈ S) :
    SolvesTetris cfg A.toSolver :=
  (solvesTetrisFrom_init_iff_solvesTetris cfg A.toSolver).mp
    (h.toSolver_solvesTetrisFrom hinit)

/-- **Existence form: init-containing closed atlas implies
SolvesTetris solver exists**: bridges atlas-existence to solver-
existence directly. Useful for showing TetrisSolvable from atlas-
based constructions without needing to first build a cycle or M4
artifact. -/
theorem exists_solvesTetris_of_exists_init_closed_atlas
    {cfg : GameConfig}
    (h : ∃ A : Atlas cfg, ∃ S : Finset GameState,
         A.IsClosedOn cfg S ∧ GameState.init ∈ S) :
    ∃ σ : Solver cfg, SolvesTetris cfg σ := by
  obtain ⟨A, S, hclosed, hinit⟩ := h
  exact ⟨A.toSolver, hclosed.toSolver_solvesTetris hinit⟩

/-- **The project's central conjecture, formally stated.** There exists a
solver that survives forever against every legal 7-bag piece sequence on the
canonical 10×20 board. -/
def TetrisSolvable : Prop :=
  ∃ σ : Solver GameConfig.standard, SolvesTetris GameConfig.standard σ

/-- **Standard-config TetrisSolvable from init-closed atlas**:
specialisation of Tick 1834 to `GameConfig.standard`. The most
direct Atlas-level bridge to the headline `TetrisSolvable`
proposition. -/
theorem tetrisSolvable_of_exists_init_closed_atlas
    (h : ∃ A : Atlas GameConfig.standard, ∃ S : Finset GameState,
         A.IsClosedOn GameConfig.standard S ∧ GameState.init ∈ S) :
    TetrisSolvable :=
  exists_solvesTetris_of_exists_init_closed_atlas h

/-- A solver is *valid* if its chosen placement is in-bounds (`Placement.Valid`)
and plays the announced piece, for every announced piece that's in the
current bag. This is a stronger structural requirement than just `SolvesTetris`
— the latter only ensures no loss along the trace. -/
def ValidSolver (cfg : GameConfig) (σ : Solver cfg) : Prop :=
  ∀ g p, p ∈ g.bag → (σ g p).piece = p ∧ (σ g p).Valid cfg

/-- A *valid solving* solver: plays valid moves AND never loses. The natural
strengthening of `SolvesTetris` that aligns with the `safe`-set formalism. -/
def SolvesTetrisValid (cfg : GameConfig) (σ : Solver cfg) : Prop :=
  ValidSolver cfg σ ∧ SolvesTetris cfg σ

/-- **A solving solver's reachable states are all non-lost.** Combines
`solverReachable_exists_trace` with the `SolvesTetris` no-loss
promise. -/
theorem solverReachable_not_lost_of_solvesTetris {cfg : GameConfig}
    {σ : Solver cfg} (hs : SolvesTetris cfg σ)
    {g : GameState} (hr : solverReachable σ g) :
    ¬ g.lost cfg := by
  obtain ⟨s, hl, n, rfl⟩ := solverReachable_exists_trace hr
  exact hs s hl n

/-- **A `SolvesTetrisValid` solver's reachable states are all non-lost.**
Specialisation of Tick 296 via the `.toSolvesTetris` projection. -/
theorem solverReachable_not_lost_of_solvesTetrisValid {cfg : GameConfig}
    {σ : Solver cfg} (hs : SolvesTetrisValid cfg σ)
    {g : GameState} (hr : solverReachable σ g) :
    ¬ g.lost cfg :=
  solverReachable_not_lost_of_solvesTetris hs.2 hr

/-- **`solverReachable` ⊆ `Reachable` under a `SolvesTetrisValid` solver.**
The forward-closure under a *valid solving* solver lives inside the
game's reachability set. Induction on `solverReachable`: init case is
`Reachable.init`; step case uses `ValidSolver` for placement validity,
`Placement.eta_of_piece_eq` to convert to a `GameState.step`, and Tick
297 for the no-loss obligation. -/
theorem solverReachable_reachable_of_solvesTetrisValid {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (hr : solverReachable σ g) :
    Reachable cfg g := by
  induction hr with
  | init => exact Reachable.init
  | @step g' p hg' hp ih =>
    obtain ⟨hpiece, hvalid⟩ := hsv.1 g' p hp
    have heta : ({σ g' p with piece := p} : Placement) = σ g' p := by
      rcases hpl : σ g' p with ⟨pc, r, c⟩
      rw [hpl] at hpiece
      cases hpiece
      rfl
    rw [adversarialStep_eq_step, heta]
    refine Reachable.step ih ?_ hvalid ?_
    · rw [hpiece]; exact hp
    · exact solverReachable_not_lost_of_solvesTetrisValid hsv hg'

/-- **Board WF along the `solverReachable` cone of a valid solving
solver.** Corollary of Tick 298 + `Reachable.wf`. -/
theorem solverReachable_wf_of_solvesTetrisValid {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (hr : solverReachable σ g) :
    Board.WF cfg g.board :=
  (solverReachable_reachable_of_solvesTetrisValid hsv hr).wf

/-- **Bundled `solverReachable`-under-`SolvesTetrisValid` invariants.**
Composes Ticks 296–299 into a single four-tuple destructuring source:
`Reachable ∧ Board.WF ∧ bag.Nonempty ∧ ¬ lost`. -/
theorem solverReachable_invariants_of_solvesTetrisValid {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (hr : solverReachable σ g) :
    Reachable cfg g ∧ Board.WF cfg g.board ∧ g.bag.Nonempty ∧
      ¬ g.lost cfg :=
  ⟨solverReachable_reachable_of_solvesTetrisValid hsv hr,
   solverReachable_wf_of_solvesTetrisValid hsv hr,
   solverReachable_bag_nonempty hr,
   solverReachable_not_lost_of_solvesTetrisValid hsv hr⟩

/-- Projection: every `SolvesTetrisValid` is a `ValidSolver`. -/
theorem SolvesTetrisValid.toValidSolver {cfg : GameConfig} {σ : Solver cfg}
    (h : SolvesTetrisValid cfg σ) : ValidSolver cfg σ := h.1

/-- **Any `ValidSolver`'s output lands in `Placement.allValidFor`**.
Generic property: a valid solver's placement for piece `p ∈ g.bag` is
always in the search-enumeration Finset. -/
theorem ValidSolver.mem_allValidFor {cfg : GameConfig} {σ : Solver cfg}
    (h : ValidSolver cfg σ) (g : GameState) {p : Piece} (hp : p ∈ g.bag) :
    σ g p ∈ Placement.allValidFor cfg p :=
  (Placement.mem_allValidFor cfg p _).mpr (h g p hp)


/-- The **trivial solver**: always picks placement at column 0, rotation 0.
Concrete and computable. Useful as a `decide`-friendly probe for
`IsInitCycle`. -/
def trivialSolver (cfg : GameConfig) : Solver cfg := fun _ p => ⟨p, 0, 0⟩

@[simp] theorem trivialSolver_piece (cfg : GameConfig) (g : GameState)
    (p : Piece) : (trivialSolver cfg g p).piece = p := rfl

@[simp] theorem trivialSolver_rot (cfg : GameConfig) (g : GameState)
    (p : Piece) : (trivialSolver cfg g p).rot = 0 := rfl

@[simp] theorem trivialSolver_col (cfg : GameConfig) (g : GameState)
    (p : Piece) : (trivialSolver cfg g p).col = 0 := rfl

/-- The `trivialSolver` is a `ValidSolver` when `cfg.cols ≥ 4`. -/
theorem trivialSolver_validSolver {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) :
    ValidSolver cfg (trivialSolver cfg) := by
  intro g p _hp
  refine ⟨rfl, ?_⟩
  intro cell hcell
  have hc4 := Piece.shapeUp_col_lt_four p 0 cell hcell
  change 0 + cell.1 < cfg.cols
  omega

/-- The `trivialSolver` is `Valid` for `standard`. -/
theorem trivialSolver_validSolver_standard :
    ValidSolver GameConfig.standard (trivialSolver GameConfig.standard) :=
  trivialSolver_validSolver (by decide)

/-- The `trivialSolver` is `Valid` for `tiny`. -/
theorem trivialSolver_validSolver_tiny :
    ValidSolver GameConfig.tiny (trivialSolver GameConfig.tiny) :=
  trivialSolver_validSolver (by decide)

/-- **Existence of a `ValidSolver`** for any cfg with cols ≥ 4. -/
theorem exists_validSolver {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) :
    ∃ σ : Solver cfg, ValidSolver cfg σ :=
  ⟨trivialSolver cfg, trivialSolver_validSolver hcols⟩

/-- Standard specialization. -/
theorem exists_validSolver_standard :
    ∃ σ : Solver GameConfig.standard, ValidSolver GameConfig.standard σ :=
  exists_validSolver (by decide)

/-- Tiny specialization. -/
theorem exists_validSolver_tiny :
    ∃ σ : Solver GameConfig.tiny, ValidSolver GameConfig.tiny σ :=
  exists_validSolver (by decide)

/-- **Solver type is nonempty for any cfg**: trivialSolver
witnesses. -/
theorem exists_solver (cfg : GameConfig) : Nonempty (Solver cfg) :=
  ⟨trivialSolver cfg⟩

/-- `Solver cfg` is inhabited (via `trivialSolver`). -/
instance (cfg : GameConfig) : Inhabited (Solver cfg) :=
  ⟨trivialSolver cfg⟩

/-- The default solver is `trivialSolver`. -/
@[simp] theorem default_solver_eq_trivialSolver (cfg : GameConfig) :
    (default : Solver cfg) = trivialSolver cfg := rfl

/-- The default solver is a `ValidSolver` when `cols ≥ 4`. -/
theorem default_solver_validSolver {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) :
    ValidSolver cfg (default : Solver cfg) :=
  trivialSolver_validSolver hcols

/-- **`Placement.allValidFor` is nonempty for every piece** (cols ≥ 4).
Witnessed by the `trivialSolver`'s output. The search enumeration is
never empty at the supported configs. -/
theorem Placement.allValidFor_nonempty {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (p : Piece) :
    (Placement.allValidFor cfg p).Nonempty :=
  ⟨trivialSolver cfg GameState.init p,
   (trivialSolver_validSolver hcols).mem_allValidFor GameState.init
     (GameState.init_bag.symm ▸ Bag.mem_full p)⟩

/-- Standard specialization: `Placement.allValidFor` is nonempty
for every piece in the standard config. -/
theorem Placement.standard_allValidFor_nonempty (p : Piece) :
    (Placement.allValidFor GameConfig.standard p).Nonempty :=
  Placement.allValidFor_nonempty (by decide) p

/-- Tiny specialization: `Placement.allValidFor` is nonempty
for every piece in the tiny config. -/
theorem Placement.tiny_allValidFor_nonempty (p : Piece) :
    (Placement.allValidFor GameConfig.tiny p).Nonempty :=
  Placement.allValidFor_nonempty (by decide) p

/-- **`Placement.allValidFor` card upper bound**: at most `4 * cfg.cols`
placements per piece (4 rotations × `cfg.cols` columns; validity
filtering only shrinks the set). -/
theorem Placement.allValidFor_card_le (cfg : GameConfig) (p : Piece) :
    (Placement.allValidFor cfg p).card ≤ 4 * cfg.cols := by
  unfold Placement.allValidFor
  refine (Finset.card_filter_le _ _).trans ?_
  refine (Finset.card_image_le).trans ?_
  rw [Finset.card_product, Finset.card_range, Finset.card_univ,
      Fintype.card_fin]
  omega

/-- Standard specialization: `Placement.allValidFor` for any piece on
the standard config has card `≤ 40`. -/
theorem Placement.standard_allValidFor_card_le (p : Piece) :
    (Placement.allValidFor GameConfig.standard p).card ≤ 40 := by
  have h := Placement.allValidFor_card_le GameConfig.standard p
  rw [show (GameConfig.standard.cols : ℕ) = 10 from rfl] at h
  omega

/-- Tiny specialization: `Placement.allValidFor` for any piece on the
tiny config has card `≤ 16`. -/
theorem Placement.tiny_allValidFor_card_le (p : Piece) :
    (Placement.allValidFor GameConfig.tiny p).card ≤ 16 := by
  have h := Placement.allValidFor_card_le GameConfig.tiny p
  rw [show (GameConfig.tiny.cols : ℕ) = 4 from rfl] at h
  omega

/-- **Explicit canonical placement**: `⟨p, 0, 0⟩` is always in
`Placement.allValidFor cfg p` (when `4 ≤ cfg.cols`). The same witness
`trivialSolver` produces, now as a Finset-membership lemma. -/
theorem Placement.mk_zero_mem_allValidFor {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (p : Piece) :
    (⟨p, 0, 0⟩ : Placement) ∈ Placement.allValidFor cfg p :=
  (trivialSolver_validSolver hcols).mem_allValidFor GameState.init
    (GameState.init_bag.symm ▸ Bag.mem_full p)

/-- Standard config: `⟨p, 0, 0⟩ ∈ allValidFor _ p`. -/
theorem Placement.mk_zero_mem_allValidFor_standard (p : Piece) :
    (⟨p, 0, 0⟩ : Placement) ∈ Placement.allValidFor GameConfig.standard p :=
  Placement.mk_zero_mem_allValidFor (by decide) p

/-- Tiny config: `⟨p, 0, 0⟩ ∈ allValidFor _ p`. -/
theorem Placement.mk_zero_mem_allValidFor_tiny (p : Piece) :
    (⟨p, 0, 0⟩ : Placement) ∈ Placement.allValidFor GameConfig.tiny p :=
  Placement.mk_zero_mem_allValidFor (by decide) p

/-- The canonical placement's piece is `p` (definitional). -/
@[simp] theorem Placement.mk_zero_piece (p : Piece) :
    (⟨p, 0, 0⟩ : Placement).piece = p :=
  rfl

/-- The canonical placement's rotation is `0` (definitional). -/
@[simp] theorem Placement.mk_zero_rot (p : Piece) :
    (⟨p, 0, 0⟩ : Placement).rot = 0 :=
  rfl

/-- The canonical placement's column is `0` (definitional). -/
@[simp] theorem Placement.mk_zero_col (p : Piece) :
    (⟨p, 0, 0⟩ : Placement).col = 0 :=
  rfl

/-- The canonical placement equals `⟨p, 0, 0⟩` via component-wise equality. -/
theorem Placement.mk_zero_eq (p : Piece) :
    (⟨p, 0, 0⟩ : Placement) = { piece := p, rot := 0, col := 0 } :=
  rfl

/-- Eta-like rule: rebuilding a `Placement` from its components yields itself. -/
theorem Placement.mk_eta (pl : Placement) :
    ({ piece := pl.piece, rot := pl.rot, col := pl.col } : Placement) = pl := by
  cases pl; rfl

/-- Component-wise equality: two placements are equal iff their piece, rot,
and col are equal. -/
theorem Placement.eq_iff (pl₁ pl₂ : Placement) :
    pl₁ = pl₂ ↔ pl₁.piece = pl₂.piece ∧ pl₁.rot = pl₂.rot ∧ pl₁.col = pl₂.col := by
  cases pl₁; cases pl₂
  simp

/-- A placement equals `⟨p, 0, 0⟩` iff its piece is `p` and rot and col are 0. -/
theorem Placement.eq_mk_zero_iff (pl : Placement) (p : Piece) :
    pl = ⟨p, 0, 0⟩ ↔ pl.piece = p ∧ pl.rot = 0 ∧ pl.col = 0 :=
  Placement.eq_iff pl ⟨p, 0, 0⟩

/-- `solverReachable σ init` for any solver `σ` (constructor alias). -/
theorem solverReachable_init {cfg : GameConfig} (σ : Solver cfg) :
    solverReachable σ GameState.init :=
  solverReachable.init

/-- `solverReachable σ` is closed under adversarial steps from any of its
states. Theorem alias for the `solverReachable.step` constructor. -/
theorem solverReachable_step {cfg : GameConfig} {σ : Solver cfg} {g : GameState}
    (p : Piece) (h : solverReachable σ g) (hp : p ∈ g.bag) :
    solverReachable σ (adversarialStep cfg g p (σ g p)) :=
  solverReachable.step p h hp

/-- Case analysis on `solverReachable`: every solver-reachable state is either
`init` or the adversarial-step image of another. -/
theorem solverReachable.eq_init_or_step {cfg : GameConfig} {σ : Solver cfg}
    {g : GameState} (h : solverReachable σ g) :
    g = GameState.init ∨
    ∃ g' : GameState, ∃ p : Piece, solverReachable σ g' ∧ p ∈ g'.bag ∧
      g = adversarialStep cfg g' p (σ g' p) := by
  cases h with
  | init => exact Or.inl rfl
  | @step g' p hr hp => exact Or.inr ⟨g', p, hr, hp, rfl⟩

/-- A solver-reachable state distinct from `init` arose from an adversarial
step of another solver-reachable state. -/
theorem solverReachable.exists_step_of_ne_init {cfg : GameConfig}
    {σ : Solver cfg} {g : GameState} (h : solverReachable σ g)
    (hne : g ≠ GameState.init) :
    ∃ g' : GameState, ∃ p : Piece, solverReachable σ g' ∧ p ∈ g'.bag ∧
      g = adversarialStep cfg g' p (σ g' p) :=
  h.eq_init_or_step.resolve_left hne

/-- Post-`adversarialStep` bag card is at least 1. -/
theorem adversarialStep_one_le_bag_card (cfg : GameConfig) (g : GameState)
    (p : Piece) (pl : Placement) :
    1 ≤ (adversarialStep cfg g p pl).bag.card :=
  adversarialStep_bag_card_pos cfg g p pl

/-- Post-`adversarialStep` bag card is nonzero. -/
theorem adversarialStep_bag_card_ne_zero (cfg : GameConfig) (g : GameState)
    (p : Piece) (pl : Placement) :
    (adversarialStep cfg g p pl).bag.card ≠ 0 :=
  Nat.ne_of_gt (adversarialStep_bag_card_pos cfg g p pl)

/-- The canonical placement `⟨p, 0, 0⟩` is `Valid` (when `4 ≤ cfg.cols`). -/
theorem Placement.mk_zero_Valid {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (p : Piece) :
    (⟨p, 0, 0⟩ : Placement).Valid cfg :=
  ((trivialSolver_validSolver hcols) GameState.init p
    (GameState.init_bag.symm ▸ Bag.mem_full p)).2

/-- Standard config: `⟨p, 0, 0⟩` is `Valid`. -/
theorem Placement.mk_zero_standard_Valid (p : Piece) :
    (⟨p, 0, 0⟩ : Placement).Valid GameConfig.standard :=
  Placement.mk_zero_Valid (by decide) p

/-- Tiny config: `⟨p, 0, 0⟩` is `Valid`. -/
theorem Placement.mk_zero_tiny_Valid (p : Piece) :
    (⟨p, 0, 0⟩ : Placement).Valid GameConfig.tiny :=
  Placement.mk_zero_Valid (by decide) p

/-- **Concrete one-step reachability**: from init, stepping with
the canonical placement `⟨p, 0, 0⟩` lands in a `Reachable` state. -/
theorem Reachable.init_step_mk_zero {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (p : Piece) :
    Reachable cfg (GameState.init.step cfg (⟨p, 0, 0⟩ : Placement)) :=
  Reachable.init_step (Placement.mk_zero_Valid hcols p)

/-- Standard: from init, stepping with `⟨p, 0, 0⟩` lands in `Reachable`. -/
theorem Reachable.standard_init_step_mk_zero (p : Piece) :
    Reachable GameConfig.standard
      (GameState.init.step GameConfig.standard (⟨p, 0, 0⟩ : Placement)) :=
  Reachable.init_step_mk_zero (by decide) p

/-- Tiny: from init, stepping with `⟨p, 0, 0⟩` lands in `Reachable`. -/
theorem Reachable.tiny_init_step_mk_zero (p : Piece) :
    Reachable GameConfig.tiny
      (GameState.init.step GameConfig.tiny (⟨p, 0, 0⟩ : Placement)) :=
  Reachable.init_step_mk_zero (by decide) p

/-- **Card-interval for `Placement.allValidFor`**: `1 ≤ card ≤ 4 *
cfg.cols`. Bundles `card_pos` (from `Nonempty`) with `card_le`
(combinatorial bound). -/
theorem Placement.allValidFor_card_interval {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (p : Piece) :
    1 ≤ (Placement.allValidFor cfg p).card ∧
      (Placement.allValidFor cfg p).card ≤ 4 * cfg.cols :=
  ⟨Finset.card_pos.mpr (Placement.allValidFor_nonempty hcols p),
   Placement.allValidFor_card_le cfg p⟩

/-- **Sum across pieces**: total candidate placements across all 7
pieces ≤ `28 * cfg.cols`. The full enumeration bound for any solver
that must consider every piece. -/
theorem Placement.sum_allValidFor_card_le (cfg : GameConfig) :
    (∑ p : Piece, (Placement.allValidFor cfg p).card) ≤ 28 * cfg.cols := by
  calc (∑ p : Piece, (Placement.allValidFor cfg p).card)
      ≤ ∑ _p : Piece, 4 * cfg.cols :=
        Finset.sum_le_sum (fun p _ => Placement.allValidFor_card_le cfg p)
    _ = Fintype.card Piece * (4 * cfg.cols) := by
        rw [Finset.sum_const, Finset.card_univ, smul_eq_mul]
    _ = 28 * cfg.cols := by
        rw [Piece.card_eq_seven]; ring

/-- Standard specialization: total candidate placements ≤ 280. -/
theorem Placement.standard_sum_allValidFor_card_le :
    (∑ p : Piece,
        (Placement.allValidFor GameConfig.standard p).card) ≤ 280 := by
  have h := Placement.sum_allValidFor_card_le GameConfig.standard
  rw [show (GameConfig.standard.cols : ℕ) = 10 from rfl] at h
  omega

/-- Tiny specialization: total candidate placements ≤ 112. -/
theorem Placement.tiny_sum_allValidFor_card_le :
    (∑ p : Piece,
        (Placement.allValidFor GameConfig.tiny p).card) ≤ 112 := by
  have h := Placement.sum_allValidFor_card_le GameConfig.tiny
  rw [show (GameConfig.tiny.cols : ℕ) = 4 from rfl] at h
  omega

/-- The same predicate generalised to an arbitrary `GameConfig`. Useful for
attempting concrete-cycle constructions on simpler / degenerate configs first. -/
def TetrisSolvableFor (cfg : GameConfig) : Prop :=
  ∃ σ : Solver cfg, SolvesTetris cfg σ

/-- **Atlas-level TetrisSolvableFor**: any init-containing closed
atlas yields a `TetrisSolvableFor cfg` witness. Direct: `A.toSolver`
is the witness solver, with no-loss certified by Tick 1833's
`toSolver_solvesTetris`. The any-cfg parallel to Tick 1835's
`tetrisSolvable_of_exists_init_closed_atlas` (which uses
`GameConfig.standard`). -/
theorem Atlas.IsClosedOn.tetrisSolvableFor_of_init_mem
    {cfg : GameConfig} {A : Atlas cfg} {S : Finset GameState}
    (h : A.IsClosedOn cfg S) (hinit : GameState.init ∈ S) :
    TetrisSolvableFor cfg :=
  ⟨A.toSolver, h.toSolver_solvesTetris hinit⟩

/-- The strengthened solvability conjecture requiring valid placements.
The version that aligns with the `safe`-set's `.Valid cfg` requirement
and is fully equivalent to `init ∈ safe cfg` (modulo `4 ≤ cfg.cols`). -/
def TetrisSolvableValid : Prop :=
  ∃ σ : Solver GameConfig.standard, SolvesTetrisValid GameConfig.standard σ

/-- Same as `TetrisSolvableValid` but parametric in `cfg`. -/
def TetrisSolvableValidFor (cfg : GameConfig) : Prop :=
  ∃ σ : Solver cfg, SolvesTetrisValid cfg σ

@[simp] theorem tetrisSolvableValid_iff_for_standard :
    TetrisSolvableValid ↔ TetrisSolvableValidFor GameConfig.standard := Iff.rfl

/-- A valid solving solver is a solving solver. -/
theorem SolvesTetrisValid.toSolvesTetris {cfg : GameConfig} {σ : Solver cfg}
    (h : SolvesTetrisValid cfg σ) : SolvesTetris cfg σ := h.2

/-- Strengthened solvability implies the (permissive) standard one. -/
theorem TetrisSolvableValidFor.toSolvable {cfg : GameConfig}
    (h : TetrisSolvableValidFor cfg) : TetrisSolvableFor cfg := by
  obtain ⟨σ, hσ⟩ := h
  exact ⟨σ, hσ.toSolvesTetris⟩

/-- Choose a solver witness from `TetrisSolvableFor`. -/
noncomputable def TetrisSolvableFor.solver {cfg : GameConfig}
    (h : TetrisSolvableFor cfg) : Solver cfg := h.choose

theorem TetrisSolvableFor.solver_solvesTetris {cfg : GameConfig}
    (h : TetrisSolvableFor cfg) : SolvesTetris cfg h.solver :=
  h.choose_spec

/-- Choose a `ValidSolver` witness from `TetrisSolvableValidFor`. -/
noncomputable def TetrisSolvableValidFor.solver {cfg : GameConfig}
    (h : TetrisSolvableValidFor cfg) : Solver cfg := h.choose

/-- The chosen solver is a `SolvesTetrisValid`. -/
theorem TetrisSolvableValidFor.solver_solvesTetrisValid {cfg : GameConfig}
    (h : TetrisSolvableValidFor cfg) : SolvesTetrisValid cfg h.solver :=
  h.choose_spec

/-- The chosen solver is a `ValidSolver`. -/
theorem TetrisSolvableValidFor.solver_validSolver {cfg : GameConfig}
    (h : TetrisSolvableValidFor cfg) : ValidSolver cfg h.solver :=
  h.solver_solvesTetrisValid.toValidSolver

/-- The chosen solver `SolvesTetris`. -/
theorem TetrisSolvableValidFor.solver_solvesTetris {cfg : GameConfig}
    (h : TetrisSolvableValidFor cfg) : SolvesTetris cfg h.solver :=
  h.solver_solvesTetrisValid.toSolvesTetris

/-- Strengthened standard solvability implies the (permissive) standard one. -/
theorem TetrisSolvableValid.toSolvable (h : TetrisSolvableValid) : TetrisSolvable :=
  TetrisSolvableValidFor.toSolvable h

@[simp] theorem tetrisSolvable_iff_for_standard :
    TetrisSolvable ↔ TetrisSolvableFor GameConfig.standard := Iff.rfl

/-- **A solving solver loses on no `solverReachable` state.** Direct
combination of `solverReachable_exists_trace` (Tick 121) with `SolvesTetris`. -/
theorem solverReachable_not_lost {cfg : GameConfig} {σ : Solver cfg}
    (hsolves : SolvesTetris cfg σ) {g : GameState}
    (hr : solverReachable σ g) : ¬ g.lost cfg := by
  obtain ⟨s, hl, n, rfl⟩ := solverReachable_exists_trace hr
  exact hsolves s hl n

end Tetris
