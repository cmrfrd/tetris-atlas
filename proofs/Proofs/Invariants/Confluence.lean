import Mathlib
import Proofs.Combinatorics.BoardCount
import Proofs.Invariants.HoledSkyline

/-!
# Order-confluence: column-disjoint placements commute

The keystone of the confluence route to `TetrisSolvable`. In the adversarial
game the opponent's only power is the ORDER in which the bag's seven pieces
arrive. A strategy that answers each piece inside its own column zone makes
that order irrelevant, because hard-drop placements with disjoint column
footprints COMMUTE as board transformations: neither placement changes the
column heights the other reads (`Board.colHeight_place_of_notMem_cols`), so
each piece hard-drops to the same cells whether it lands before or after the
other, and the final board is the same union either way.

This file proves the pairwise commutation spine:

* `Placement.footCols` — the absolute columns a placement occupies;
* `Placement.ColsDisjoint` — two placements' column footprints do not meet;
* `Placement.dropOffset_place_of_colsDisjoint` — a column-disjoint placement
  never changes a piece's hard-drop offset;
* `Placement.dropped_place_of_colsDisjoint` — nor its landing cells;
* `Placement.place_comm_of_colsDisjoint` — `place` commutes.

Downstream (later files in the route): all `7!` within-bag arrival orders of a
zone-local response collapse to one canonical trajectory, so a cooperative
witness loop certifies the adversarial bag — the reduction that turns the
cooperative lasso into an ingredient of the adversarial proof.

**Measured status of the route (2026-07, `scripts/find_confluent_atlas.py`):**
the theorems here are sound and general, but a FULLY zone-local bag response
does not exist on 10 columns — minimum piece footprints sum to 13 > 10, and
cross-piece two-column box pairs are geometrically impossible (a `2×4` box
tiles only as `L+L`/`J+J`/`O+O`/`I+I`), so every bag forces order-forks
somewhere. Adaptive per-state play reconverges any single bag easily (branch
4 suffices empirically, everywhere), but concentrating every order's
end-of-bag board into a small recurring family fails at every tested design
scale (`≤ 10³` families all die; the loosest workable boundary predicate has
a `5.33M`-board universe). So these lemmas certify zone-local SEGMENTS of
play, not a small closed adversarial table; `TetrisSolvable` is NOT proven
here, and no `sorry` is used.
-/

namespace Tetris
namespace Placement

/-- The absolute columns a placement's cells occupy (its column footprint).
The hard-drop offset and landing cells of a placement depend on the board
only through `colHeight` on `footCols` — so boards agreeing there are
indistinguishable to the placement. -/
def footCols (pl : Placement) : Finset ℕ :=
  pl.shapeUp.image (fun cell => pl.col + cell.1)

/-- **Column-disjointness of two placements**: no absolute column is occupied
by both. The zone discipline of a confluent strategy: each bag piece answers
in its own zone, so any two same-bag placements satisfy this. -/
def ColsDisjoint (pl1 pl2 : Placement) : Prop :=
  ∀ c1 ∈ pl1.shapeUp, ∀ c2 ∈ pl2.shapeUp, pl1.col + c1.1 ≠ pl2.col + c2.1

instance (pl1 pl2 : Placement) : Decidable (ColsDisjoint pl1 pl2) := by
  unfold ColsDisjoint; infer_instance

/-- Column-disjointness is symmetric. -/
theorem ColsDisjoint.symm {pl1 pl2 : Placement} (h : ColsDisjoint pl1 pl2) :
    ColsDisjoint pl2 pl1 :=
  fun c2 hc2 c1 hc1 => (h c1 hc1 c2 hc2).symm

/-- `ColsDisjoint` is exactly disjointness of the column footprints. -/
theorem colsDisjoint_iff_disjoint_footCols (pl1 pl2 : Placement) :
    ColsDisjoint pl1 pl2 ↔ Disjoint pl1.footCols pl2.footCols := by
  constructor
  · intro h
    refine Finset.disjoint_left.2 ?_
    intro j hj1 hj2
    obtain ⟨c1, hc1, rfl⟩ := Finset.mem_image.1 hj1
    obtain ⟨c2, hc2, he⟩ := Finset.mem_image.1 hj2
    exact h c1 hc1 c2 hc2 he.symm
  · intro h c1 hc1 c2 hc2 he
    exact Finset.disjoint_left.1 h
      (Finset.mem_image_of_mem _ hc1)
      (he ▸ Finset.mem_image_of_mem _ hc2)

/-- **A column-disjoint placement never changes a piece's hard-drop offset.**
`dropOffset` reads the board only through `colHeight` on the piece's own
columns, and a disjoint placement preserves those heights (locality). -/
theorem dropOffset_place_of_colsDisjoint (b : Board) {pl1 pl2 : Placement}
    (hd : ColsDisjoint pl1 pl2) :
    pl2.dropOffset (pl1.place b) = pl2.dropOffset b := by
  rw [dropOffset_eq_sup, dropOffset_eq_sup]
  refine Finset.sup_congr rfl fun c2 hc2 => ?_
  simp only [Board.colHeight_place_of_notMem_cols b pl1 (pl2.col + c2.1)
    (fun c1 hc1 => (hd c1 hc1 c2 hc2).symm)]

/-- **A column-disjoint placement never changes a piece's landing cells.** -/
theorem dropped_place_of_colsDisjoint (b : Board) {pl1 pl2 : Placement}
    (hd : ColsDisjoint pl1 pl2) :
    pl2.dropped (pl1.place b) = pl2.dropped b := by
  rw [dropped_eq_cellsAt, dropped_eq_cellsAt, dropOffset_place_of_colsDisjoint b hd]

/-- **Column-disjoint placements commute.** Placing `pl1` then `pl2` produces
the same board as placing `pl2` then `pl1`: each piece drops to the same cells
in either order (landing-cell invariance), and the two unions agree. This is
the pairwise engine of order-confluence — the algebraic fact that deletes the
adversary's order power over a zone-local strategy. -/
theorem place_comm_of_colsDisjoint (b : Board) {pl1 pl2 : Placement}
    (hd : ColsDisjoint pl1 pl2) :
    pl2.place (pl1.place b) = pl1.place (pl2.place b) := by
  have h2 : pl2.dropped (pl1.place b) = pl2.dropped b :=
    dropped_place_of_colsDisjoint b hd
  have h1 : pl1.dropped (pl2.place b) = pl1.dropped b :=
    dropped_place_of_colsDisjoint b hd.symm
  show pl1.place b ∪ pl2.dropped (pl1.place b) = pl2.place b ∪ pl1.dropped (pl2.place b)
  rw [h1, h2]
  show (b ∪ pl1.dropped b) ∪ pl2.dropped b = (b ∪ pl2.dropped b) ∪ pl1.dropped b
  exact Finset.union_right_comm b (pl1.dropped b) (pl2.dropped b)

/-- **A full move without completed rows is a bare placement.** When the merge
completes no row, `clearLines` is the identity (`clearLines_eq_self_of_no_fullRows`)
and `applyStep` collapses to `place`. Mid-bag moves of a confluent design are of
this shape: rows are completed only at designated drain points, so between
drains the game evolves by pure (commuting) placements. -/
theorem applyStep_eq_place_of_no_fullRows (cfg : GameConfig) (b : Board)
    (pl : Placement) (h : Board.fullRows cfg (pl.place b) = ∅) :
    pl.applyStep cfg b = pl.place b := by
  rw [applyStep_eq_clearLines_place, Board.clearLines_eq_self_of_no_fullRows cfg h]

/-- **Column-disjoint full moves commute when neither completes a row on its
own.** The intermediate clears are no-ops (`applyStep = place`), the two bare
placements commute, and the final `clearLines` — which MAY fire — is applied to
the *same* merged board on both sides. So the adversary's choice of order
between two zone-local responses is irrelevant even when the second move
triggers a clear: order-confluence survives row completion at the joint
board, only *intermediate* completions are excluded. -/
theorem applyStep_comm_of_colsDisjoint (cfg : GameConfig) (b : Board)
    {pl1 pl2 : Placement} (hd : ColsDisjoint pl1 pl2)
    (h1 : Board.fullRows cfg (pl1.place b) = ∅)
    (h2 : Board.fullRows cfg (pl2.place b) = ∅) :
    pl2.applyStep cfg (pl1.applyStep cfg b) = pl1.applyStep cfg (pl2.applyStep cfg b) := by
  rw [applyStep_eq_place_of_no_fullRows cfg b pl1 h1,
      applyStep_eq_place_of_no_fullRows cfg b pl2 h2,
      applyStep_eq_clearLines_place, applyStep_eq_clearLines_place,
      place_comm_of_colsDisjoint b hd]

/-- **The n-ary collapse: arrival order is irrelevant for pairwise
column-disjoint placements.** Folding `place` over any two permutations of the
same placement list produces the same board, provided the placements are
pairwise column-disjoint. Induction on the permutation; the `swap` case is the
pairwise engine `place_comm_of_colsDisjoint` (identical heads commute by
`rfl`). This deletes the adversary's `7!` order choices over a zone-local bag
response: every order lands on the canonical-order board. -/
theorem foldl_place_perm {l1 l2 : List Placement} (hp : l1.Perm l2) :
    (∀ x ∈ l1, ∀ y ∈ l1, x ≠ y → ColsDisjoint x y) → ∀ b : Board,
      l1.foldl Placement.place b = l2.foldl Placement.place b := by
  induction hp with
  | nil => intro _ b; rfl
  | cons x p ih =>
    intro hd b
    simp only [List.foldl_cons]
    exact ih
      (fun a ha c hc hac =>
        hd a (List.mem_cons_of_mem x ha) c (List.mem_cons_of_mem x hc) hac)
      (Placement.place b x)
  | swap x y l =>
    intro hd b
    simp only [List.foldl_cons]
    by_cases hxy : x = y
    · subst hxy; rfl
    · have hcd : ColsDisjoint y x :=
        hd y (List.mem_cons_self ..)
          x (List.mem_cons_of_mem _ (List.mem_cons_self ..)) (Ne.symm hxy)
      rw [place_comm_of_colsDisjoint b hcd]
  | trans p1 p2 ih1 ih2 =>
    intro hd b
    exact (ih1 hd b).trans
      (ih2 (fun a ha c hc hac =>
        hd a (p1.symm.subset ha) c (p1.symm.subset hc) hac) b)

/-- `foldl_place_perm`, packaged with a `List.Pairwise` hypothesis (the natural
shape for a bag response list): pairwise disjointness in list order extends to
all unordered pairs by symmetry. -/
theorem foldl_place_perm_of_pairwise {l1 l2 : List Placement} (hp : l1.Perm l2)
    (hd : l1.Pairwise ColsDisjoint) (b : Board) :
    l1.foldl Placement.place b = l2.foldl Placement.place b :=
  foldl_place_perm hp (hd.forall (fun _ _ h => h.symm)) b

/-- **Every piece shape has a bottom cell**: some cell of `shapeUp` sits at
relative height `0`. This is what makes hard drops rest on the stack — and
what anchors the shift-equivariance below (the drop offset is attained at a
bottom cell's column, so a uniform surface shift never truncates away). -/
theorem shapeUp_exists_bottom : ∀ (p : Piece) (r : Rotation),
    ∃ c ∈ p.shapeUp r, c.2 = 0 := by
  decide

/-- **Shift-equivariance of the hard-drop offset.** On a skyline uniformly
lowered by `m` (with every footprint column at least `m` high and in range),
a placement's drop offset drops by exactly `m`. With landing cells
`(col + c.1, dropOffset + c.2)`, this says the piece lands exactly `m`
lower — the algebraic content of "a drain-style uniform clear commutes with
subsequent local placements". -/
theorem dropOffset_skyline_sub (cfg : GameConfig) (h : ℕ → ℕ) (m : ℕ)
    (pl : Placement)
    (hcols : ∀ c ∈ pl.shapeUp, pl.col + c.1 < cfg.cols)
    (hm : ∀ c ∈ pl.shapeUp, m ≤ h (pl.col + c.1)) :
    pl.dropOffset (Board.skyline cfg (fun j => h j - m)) + m
      = pl.dropOffset (Board.skyline cfg h) := by
  have hsup : ∀ (H : ℕ → ℕ), pl.dropOffset (Board.skyline cfg H)
      = pl.shapeUp.sup (fun c => H (pl.col + c.1) - c.2) := by
    intro H
    rw [dropOffset_eq_sup]
    refine Finset.sup_congr rfl fun c hc => ?_
    simp only [Board.colHeight_skyline (hcols c hc)]
  rw [hsup, hsup]
  obtain ⟨c₀, hc₀, hc₀0⟩ := shapeUp_exists_bottom pl.piece pl.rot
  have hc₀' : c₀ ∈ pl.shapeUp := hc₀
  -- the original sup is at least m (bottom-cell column is ≥ m high)
  have hbig : m ≤ pl.shapeUp.sup (fun c => h (pl.col + c.1) - c.2) := by
    refine le_trans ?_ (Finset.le_sup (f := fun c => h (pl.col + c.1) - c.2) hc₀')
    rw [hc₀0]
    exact le_trans (hm c₀ hc₀') (le_of_eq (Nat.sub_zero _).symm)
  -- pointwise: (h j - m) - u = (h j - u) - m
  have hpt : ∀ c ∈ pl.shapeUp,
      (fun c => (h (pl.col + c.1) - m) - c.2) c
        = (fun c => h (pl.col + c.1) - c.2) c - m := by
    intro c _
    simp only [Nat.sub_sub]
    omega
  rw [Finset.sup_congr rfl hpt]
  -- sup of (f - m) = (sup f) - m, then + m cancels since sup f ≥ m
  have hsub : pl.shapeUp.sup (fun c => (fun c => h (pl.col + c.1) - c.2) c - m)
      = pl.shapeUp.sup (fun c => h (pl.col + c.1) - c.2) - m := by
    refine le_antisymm ?_ ?_
    · refine Finset.sup_le fun c hc => ?_
      exact Nat.sub_le_sub_right
        (Finset.le_sup (f := fun c => h (pl.col + c.1) - c.2) hc) m
    · rw [Nat.sub_le_iff_le_add]
      refine Finset.sup_le fun c hc => ?_
      by_cases hcm : m ≤ h (pl.col + c.1) - c.2
      · calc h (pl.col + c.1) - c.2
            = (h (pl.col + c.1) - c.2 - m) + m := (Nat.sub_add_cancel hcm).symm
          _ ≤ _ + m := by
              exact Nat.add_le_add_right
                (Finset.le_sup
                  (f := fun c => (fun c => h (pl.col + c.1) - c.2) c - m) hc) m
      · exact le_trans (le_of_lt (Nat.lt_of_not_le hcm)) (Nat.le_add_left m _)
  rw [hsub, Nat.sub_add_cancel hbig]

/-- Minimum column footprint of each piece across all rotations: the vertical
`I` occupies a single column; every other piece needs at least two columns in
every rotation. -/
def pieceMinWidth : Piece → ℕ
  | Piece.I => 1
  | _ => 2

/-- Every rotation of every piece occupies at least `pieceMinWidth` distinct
piece-columns (checked over all 28 piece/rotation shapes). -/
theorem pieceMinWidth_le_card_image : ∀ (p : Piece) (r : Rotation),
    pieceMinWidth p ≤ ((p.shapeUp r).image Prod.fst).card := by
  decide

/-- A placement's absolute column footprint has exactly as many columns as
the shape's relative footprint (translation by `pl.col` is injective). -/
theorem footCols_card_eq (pl : Placement) :
    pl.footCols.card = (pl.shapeUp.image Prod.fst).card := by
  unfold footCols
  rw [show (fun cell : Coord => pl.col + cell.1)
        = (fun x => pl.col + x) ∘ Prod.fst from rfl,
      ← Finset.image_image]
  exact Finset.card_image_of_injective _ (fun a b hab => by omega)

/-- Any placement of piece `p` occupies at least `pieceMinWidth p` columns. -/
theorem pieceMinWidth_le_footCols_card (pl : Placement) :
    pieceMinWidth pl.piece ≤ pl.footCols.card := by
  rw [footCols_card_eq]
  exact pieceMinWidth_le_card_image pl.piece pl.rot

/-- **The zone-budget obstruction: fully zone-local bag responses do not
exist.** On any board up to 12 columns wide — in particular the canonical 10 —
seven valid placements, one per piece, can never be pairwise column-disjoint:
the minimum footprints (1 column for the vertical `I`, 2 for each of the other
six pieces) already demand `13` columns. Consequently the order-confluence
theorems above can cover zone-local SEGMENTS of a bag but never a whole bag —
every bag response overlaps somewhere, the adversary's order choice forks the
play there, and any closed-table design must manage reconvergence rather than
avoid forks. This is the structural root, verified here once and for all, of
the fork phenomena measured by `scripts/find_confluent_atlas.py`. -/
theorem no_pairwise_colsDisjoint_bag (cfg : GameConfig) (hcols : cfg.cols ≤ 12)
    (f : Piece → Placement)
    (hpiece : ∀ p, (f p).piece = p)
    (hvalid : ∀ p, (f p).Valid cfg) :
    ¬ (∀ p q, p ≠ q → ColsDisjoint (f p) (f q)) := by
  intro hdisj
  have hd : ∀ p ∈ Finset.univ, ∀ q ∈ Finset.univ, p ≠ q →
      Disjoint (f p).footCols (f q).footCols := fun p _ q _ hpq =>
    (colsDisjoint_iff_disjoint_footCols _ _).1 (hdisj p q hpq)
  have hsub : Finset.univ.biUnion (fun p => (f p).footCols)
      ⊆ Finset.range cfg.cols := by
    intro j hj
    rw [Finset.mem_biUnion] at hj
    obtain ⟨p, _, hjp⟩ := hj
    unfold footCols at hjp
    rw [Finset.mem_image] at hjp
    obtain ⟨c, hc, rfl⟩ := hjp
    exact Finset.mem_range.mpr (hvalid p c hc)
  have hcard : (Finset.univ.biUnion (fun p => (f p).footCols)).card
      = ∑ p, (f p).footCols.card := Finset.card_biUnion hd
  have hsum : (13 : ℕ) ≤ ∑ p : Piece, (f p).footCols.card := by
    have hmono : ∑ p : Piece, pieceMinWidth p
        ≤ ∑ p : Piece, (f p).footCols.card := by
      refine Finset.sum_le_sum fun p _ => ?_
      have h := pieceMinWidth_le_footCols_card (f p)
      rwa [hpiece p] at h
    have h13 : ∑ p : Piece, pieceMinWidth p = 13 := by decide
    omega
  have hle : (Finset.univ.biUnion (fun p => (f p).footCols)).card ≤ cfg.cols :=
    le_trans (Finset.card_le_card hsub) (le_of_eq (Finset.card_range _))
  omega

/-- The zone-budget obstruction at the canonical configuration. -/
theorem no_pairwise_colsDisjoint_bag_standard
    (f : Piece → Placement)
    (hpiece : ∀ p, (f p).piece = p)
    (hvalid : ∀ p, (f p).Valid GameConfig.standard) :
    ¬ (∀ p q, p ≠ q → ColsDisjoint (f p) (f q)) :=
  no_pairwise_colsDisjoint_bag GameConfig.standard (by decide) f hpiece hvalid

/-- **Single-order collapse: a clear-free run of moves is a placement fold.**
If no strict prefix of the move list completes a row, then every intermediate
clear is a no-op and the full-move fold (`applyStep`) is the bare placement
fold followed by ONE final `clearLines` — the last move may clear. No
disjointness is needed for a single order; disjointness enters only when
comparing different orders. -/
theorem foldl_applyStep_collapse (cfg : GameConfig) (b : Board)
    {l : List Placement} (hne : l ≠ [])
    (hnf : ∀ k, 0 < k → k < l.length →
      Board.fullRows cfg ((l.take k).foldl Placement.place b) = ∅) :
    l.foldl (Placement.applyStep cfg) b
      = Board.clearLines cfg (l.foldl Placement.place b) := by
  revert hne hnf
  induction l generalizing b with
  | nil => intro hne _; exact absurd rfl hne
  | cons pl t ih =>
    intro _ hnf
    cases t with
    | nil =>
      simp only [List.foldl_cons, List.foldl_nil]
      rfl
    | cons pl2 t2 =>
      have h1 : Board.fullRows cfg (pl.place b) = ∅ := by
        have := hnf 1 (by omega) (by simp)
        simpa using this
      have hstep : Placement.applyStep cfg b pl = Placement.place b pl :=
        applyStep_eq_place_of_no_fullRows cfg b pl h1
      simp only [List.foldl_cons]
      rw [hstep]
      refine ih (Placement.place b pl) (by simp) ?_
      intro k hk hklen
      have := hnf (k + 1) (by omega)
        (by simpa using Nat.succ_lt_succ hklen)
      simpa [List.take_succ_cons, List.foldl_cons] using this

/-- **The bag-confluence assembly.** For pairwise column-disjoint moves whose
sub-multisets never complete a row early (every proper sub-arrangement's
placement board has no full rows), EVERY arrival order produces the SAME
final board: the bare placements of the canonical order, cleared once at the
end. The adversary's order power over such a segment is void — the
adversarial fold equals the canonical cooperative fold. By the zone-budget
obstruction (`no_pairwise_colsDisjoint_bag`) a full 7-piece bag can never
satisfy the disjointness hypothesis, so this covers zone-local SEGMENTS of a
bag (up to 6 pieces, or any partial-bag window) — the largest scope at which
order-confluence is geometrically possible. -/
theorem bag_confluence (cfg : GameConfig) (b : Board) {l l' : List Placement}
    (hne : l ≠ []) (hp : l.Perm l')
    (hd : ∀ x ∈ l, ∀ y ∈ l, x ≠ y → ColsDisjoint x y)
    (hnf : ∀ s : List Placement, s.Subperm l → s.length < l.length →
      Board.fullRows cfg (s.foldl Placement.place b) = ∅) :
    l'.foldl (Placement.applyStep cfg) b
      = Board.clearLines cfg (l.foldl Placement.place b) := by
  have hlen : l'.length = l.length := hp.symm.length_eq
  have hne' : l' ≠ [] := fun h => hne (List.Perm.eq_nil (h ▸ hp))
  have hpre : ∀ k, 0 < k → k < l'.length →
      Board.fullRows cfg ((l'.take k).foldl Placement.place b) = ∅ := by
    intro k hk hklen
    refine hnf _ ((l'.take_sublist k).subperm.trans hp.symm.subperm) ?_
    rw [List.length_take]
    omega
  rw [foldl_applyStep_collapse cfg b hne' hpre]
  congr 1
  exact foldl_place_perm hp.symm
    (fun x hx y hy hxy => hd x (hp.mem_iff.mpr hx) y (hp.mem_iff.mpr hy) hxy) b

end Placement
end Tetris
