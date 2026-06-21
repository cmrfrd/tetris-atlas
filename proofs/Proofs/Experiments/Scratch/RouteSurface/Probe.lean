import Proofs.Experiments.SurfaceInvariant

/-!
# RouteSurface Probe — empirical tractability of the O-piece flat-phase hstep fragment

We probe whether a single O-piece step from a flat-top reserved-well skyline can
land back inside `bagPhaseCarrier`. The intuition is that O makes a +2 bump on two
adjacent columns, so the post-board is neither a flat top (all non-well columns
equal) nor a +1 notch (a single bumped column). We turn that intuition into a built
proof.

All experimental code lives here; `SurfaceInvariant.lean` is imported, never edited.
-/

namespace Tetris

open Board

/-- **Profile read-back from a skyline board equality.** If two skylines agree as
boards on `GameConfig.standard`, their profiles agree on every in-field column.
This is the structural injectivity we use to derive contradictions: it reads each
profile value off the board via `colHeight_skyline`. -/
theorem profile_eq_of_skyline_eq_standard {g g' : ℕ → ℕ}
    (h : skyline GameConfig.standard g = skyline GameConfig.standard g')
    {j : ℕ} (hj : j < GameConfig.standard.cols) : g j = g' j := by
  have hh : Board.colHeight (skyline GameConfig.standard g) j
      = Board.colHeight (skyline GameConfig.standard g') j := by rw [h]
  rwa [colHeight_skyline hj, colHeight_skyline hj] at hh

/-! ## Target 1 — the O-piece flat-phase fragment is FALSE

The concrete O-bumped surface. Start from a flat top of common height `base` with
reserved well at column 0, drop O on columns `2,3` (both real, both `≠ 0`, equal
height). By `applyStep_O_skyline_noclear` the post-board is the skyline of the
profile `oProfile base`: columns `2` and `3` lifted to `base+2`, the well at `0`,
every other column at `base`. -/
def oProfile (base : ℕ) : ℕ → ℕ :=
  Function.update (Function.update (fun j => if j = 0 then 0 else base) 2 (base + 2)) 3 (base + 2)

/-- Read-back values of `oProfile` on the columns we care about. -/
theorem oProfile_zero (base : ℕ) : oProfile base 0 = 0 := by
  simp [oProfile]
theorem oProfile_two (base : ℕ) : oProfile base 2 = base + 2 := by
  simp [oProfile]
theorem oProfile_three (base : ℕ) : oProfile base 3 = base + 2 := by
  simp [oProfile]
theorem oProfile_five (base : ℕ) : oProfile base 5 = base := by
  simp [oProfile]
theorem oProfile_seven (base : ℕ) : oProfile base 7 = base := by
  simp [oProfile]

/-- **The O step from a flat top lands on the bumped skyline.** Specialises
`applyStep_O_skyline_noclear` to the standard config, well at `0`, landing columns
`2,3`. The post-board is exactly `skyline standard (oProfile base)`. -/
theorem applyStep_O_flat_eq (base : ℕ) :
    Placement.applyStep GameConfig.standard
        (skyline GameConfig.standard (fun j => if j = 0 then 0 else base))
        { piece := Piece.O, rot := 0, col := 2 }
      = skyline GameConfig.standard (oProfile base) := by
  have h := applyStep_O_skyline_noclear (cfg := GameConfig.standard)
      (h := fun j => if j = 0 then 0 else base) (c := 2) (w := 0)
      (by decide) (by decide) (by simp) (by decide) (by decide) (by decide)
      (by simp)
  -- normalise the RHS profile of the library lemma to our `oProfile`
  rw [h]
  -- The two profiles are definitionally close: `h 2 = base` reduces the bump value
  -- to `base + 2`, exactly `oProfile`'s value. `rfl` may already close after `rw`.
  rfl

/-- **The O-bumped board is NOT a flat-top reserved-well skyline.** A flat top has
all non-well columns equal; the O-bumped board has columns `2,3` at `base+2` but
column `5` at `base`, two distinct non-well heights, so no `(base', w')` profile can
match. We need `base + 2 ≤ rows = 20` only to keep the surface a real skyline; the
contradiction itself is height-free. -/
theorem oBumped_not_isFlatTop (base : ℕ) :
    ¬ Board.IsFlatTopRWSkyline GameConfig.standard
        (skyline GameConfig.standard (oProfile base)) := by
  rintro ⟨base', w', hw', hbase', heq⟩
  -- profiles agree on all in-field columns
  have key : ∀ j, j < GameConfig.standard.cols →
      oProfile base j = (if j = w' then 0 else base') :=
    fun j hj => profile_eq_of_skyline_eq_standard heq hj
  -- Columns 5 and 7 both sit at `base`; at most one is the (single) well `w'`, so
  -- at least one forces `base = base'`. Column 2 sits at `base + 2`; whether or not
  -- it is the well, that contradicts `base = base'` (or gives `base + 2 = 0`).
  have h2 := key 2 (by decide)
  have h5 := key 5 (by decide)
  have h7 := key 7 (by decide)
  rw [oProfile_two] at h2
  rw [oProfile_five] at h5
  rw [oProfile_seven] at h7
  -- establish `base = base'` from a non-well column among {5,7}
  have hbb : base = base' := by
    by_cases hw5 : (5 : ℕ) = w'
    · -- then 7 ≠ w' (single well), so column 7 is at base'
      have hw7 : (7 : ℕ) ≠ w' := by rw [← hw5]; decide
      rwa [if_neg hw7] at h7
    · rwa [if_neg hw5] at h5
  -- now column 2 cannot be base' (= base) nor 0
  by_cases hw2 : (2 : ℕ) = w'
  · rw [if_pos hw2] at h2; omega
  · rw [if_neg hw2] at h2; omega

theorem oProfile_six (base : ℕ) : oProfile base 6 = base := by
  simp [oProfile]
theorem oProfile_eight (base : ℕ) : oProfile base 8 = base := by
  simp [oProfile]

/-- **The O-bumped board is NOT a notched flat-top reserved-well skyline.** A notch
deviates from its plain level `base'` in at most TWO columns: the well `w'` (height
`0`) and the single bump `m'` (height `base' + 1`). The O-bumped board, however,
has FOUR plain-level columns at `base` (so `base' = base`) yet TWO columns (`2,3`)
at `base + 2`. Since at most the single bump column `m'` can exceed `base'`, both
`2` and `3` would have to be `m'` — impossible, two distinct columns. (The +2 bump
also overshoots the notch's +1 even for a single column.) -/
theorem oBumped_not_isNotched (base : ℕ) :
    ¬ Board.IsNotchedFlatTopRWSkyline GameConfig.standard
        (skyline GameConfig.standard (oProfile base)) := by
  rintro ⟨base', w', m', hw', hm', hwm', hbase', heq⟩
  have key : ∀ j, j < GameConfig.standard.cols →
      oProfile base j
        = (if j = w' then 0 else if j = m' then base' + 1 else base') :=
    fun j hj => profile_eq_of_skyline_eq_standard heq hj
  -- Four plain-height columns; the well + bump can occupy at most two, so at least
  -- two of {5,6,7,8} are plain `base'`. We extract `base = base'`.
  have h5 := key 5 (by decide); rw [oProfile_five] at h5
  have h6 := key 6 (by decide); rw [oProfile_six] at h6
  have h7 := key 7 (by decide); rw [oProfile_seven] at h7
  have h8 := key 8 (by decide); rw [oProfile_eight] at h8
  -- For each plain-level column c, `base = base'` unless c is the well or the bump.
  -- A counting argument: with two special columns among four, one stays plain.
  have hbb : base = base' := by
    -- Suppose not. Then every one of 5,6,7,8 must be `w'` or `m'` (else it pins base=base').
    by_contra hne
    have c5 : (5 : ℕ) = w' ∨ (5 : ℕ) = m' := by
      by_cases a : (5:ℕ) = w'
      · exact Or.inl a
      · by_cases b : (5:ℕ) = m'
        · exact Or.inr b
        · rw [if_neg a, if_neg b] at h5; exact absurd h5 hne
    have c6 : (6 : ℕ) = w' ∨ (6 : ℕ) = m' := by
      by_cases a : (6:ℕ) = w'
      · exact Or.inl a
      · by_cases b : (6:ℕ) = m'
        · exact Or.inr b
        · rw [if_neg a, if_neg b] at h6; exact absurd h6 hne
    have c7 : (7 : ℕ) = w' ∨ (7 : ℕ) = m' := by
      by_cases a : (7:ℕ) = w'
      · exact Or.inl a
      · by_cases b : (7:ℕ) = m'
        · exact Or.inr b
        · rw [if_neg a, if_neg b] at h7; exact absurd h7 hne
    -- Three distinct columns 5,6,7 each ∈ {w', m'} (a 2-element set) → pigeonhole clash.
    rcases c5 with h5w | h5m <;> rcases c6 with h6w | h6m <;> rcases c7 with h7w | h7m <;>
      omega
  -- Now both columns 2 and 3 are at base+2 = base'+2, but the only above-plain
  -- column is the bump m' at base'+1. So 2 and 3 must each be m', a contradiction.
  have h2 := key 2 (by decide); rw [oProfile_two] at h2
  have h3 := key 3 (by decide); rw [oProfile_three] at h3
  have e2 : (2 : ℕ) = m' := by
    by_cases a : (2:ℕ) = w'
    · rw [if_pos a] at h2; omega
    · rw [if_neg a] at h2
      by_cases b : (2:ℕ) = m'
      · exact b
      · rw [if_neg b] at h2; omega
  have e3 : (3 : ℕ) = m' := by
    by_cases a : (3:ℕ) = w'
    · rw [if_pos a] at h3; omega
    · rw [if_neg a] at h3
      by_cases b : (3:ℕ) = m'
      · exact b
      · rw [if_neg b] at h3; omega
  exact absurd (e2.trans e3.symm) (by decide)

/-- **Target 1 verdict, packaged.** From a flat-top reserved-well skyline at any
common height, the O placement on columns `2,3` lands on a board that is NEITHER a
flat-top NOR a notched flat-top reserved-well skyline — i.e. it falls out of the
first conjunct of `bagPhaseCarrier` (which is `IsFlatTop ∨ IsNotched`). Hence the
O step cannot, on this placement, re-enter `bagPhaseCarrier` for ANY drawn bag. -/
theorem oBumped_escapes_carrier_disjunct (base : ℕ) :
    ¬ Board.IsFlatTopRWSkyline GameConfig.standard
        (Placement.applyStep GameConfig.standard
          (skyline GameConfig.standard (fun j => if j = 0 then 0 else base))
          { piece := Piece.O, rot := 0, col := 2 })
    ∧ ¬ Board.IsNotchedFlatTopRWSkyline GameConfig.standard
        (Placement.applyStep GameConfig.standard
          (skyline GameConfig.standard (fun j => if j = 0 then 0 else base))
          { piece := Piece.O, rot := 0, col := 2 }) := by
  rw [applyStep_O_flat_eq]
  exact ⟨oBumped_not_isFlatTop base, oBumped_not_isNotched base⟩

/-! ## Target 2 — the O step DOES close into the spread-bounded / pocket-band carrier

The exact O step that escapes `bagPhaseCarrier` lands cleanly inside the wider
spread-2 carriers, using EXISTING library lemmas with no new math. -/

/-- **The O step lands in `IsSpreadBoundedRWSkyline 2`.** Direct application of the
existing `isSpreadBoundedRWSkyline_O_of_isFlatTop` (line ~8629). The +2 bump that
broke the flat-top/notch carrier is absorbed by the spread-2 band `[base, base+2]`. -/
theorem oStep_lands_spreadBounded2 (base : ℕ) (hslack : base + 2 ≤ GameConfig.standard.rows) :
    Board.IsSpreadBoundedRWSkyline GameConfig.standard 2
      (Placement.applyStep GameConfig.standard
        (skyline GameConfig.standard (fun j => if j = 0 then 0 else base))
        { piece := Piece.O, rot := 0, col := 2 }) :=
  isSpreadBoundedRWSkyline_O_of_isFlatTop (cfg := GameConfig.standard)
    (base := base) (w := 0) (c := 2)
    (by decide) (by decide) (by decide) (by decide) (by decide) hslack

/-- **The O step at column 1 REGENERATES a level pocket band.** Direct application
of the existing `isLevelPocketBand_O_flatTop_standard` (line ~8663): dropping O at
column 1 on the standard flat top lands in `IsLevelPocketBand standard 2`, with a
width-4 flat pocket preserved at columns 3..6. So a non-I step closes into the
pocket-band carrier outright. -/
theorem oStep_col1_lands_levelPocketBand (base : ℕ)
    (hslack : base + 2 ≤ GameConfig.standard.rows) :
    Board.IsLevelPocketBand GameConfig.standard 2
      (Placement.applyStep GameConfig.standard
        (skyline GameConfig.standard (fun j => if j = 0 then 0 else base))
        { piece := Piece.O, rot := 0, col := 1 }) :=
  isLevelPocketBand_O_flatTop_standard hslack

/-! ## Target 3 — the obstruction is fundamentally PER-BAG, not per-piece

The spread-2 pocket-band carrier closes under each single piece in the sense of
`isLevelPocketBandAt_canonical_lands_all7` — BUT that lemma's input surface `s` is
existentially chosen PER PIECE (flat top for O/I/T/L/J, col-3 notch for S, col-1
notch for Z), and the landing FLOOR equals the input floor `base` (no reduction).
The only height regulator is the well-`I` DRAIN, which is a *list* of I-placements
(`isBoundedPocketBandCarrier_of_vertI_drain_bounded`), i.e. inherently multi-piece.

We package the structural fact: the O step strictly RAISES the surface potential by
2 and never clears, so iterating non-I fills is monotone-increasing — closure must
amortise the rise against a periodic clear, which is a per-BAG (composite) event. -/

/-- **A single non-I O fill strictly raises two columns with no clear.** The post-O
board's columns `2` and `3` sit at `base + 2`, strictly above the pre-step common
height `base`. With no line cleared (the well stays empty), the surface potential is
monotone non-decreasing under non-I fills — the engine has no per-piece height
regulator. Only the well-`I` clear (a multi-piece drain) brings the floor back down,
so any height-bounded closure must be amortised PER BAG, not established per piece. -/
theorem oStep_strictly_raises (base : ℕ) :
    Board.colHeight
        (Placement.applyStep GameConfig.standard
          (skyline GameConfig.standard (fun j => if j = 0 then 0 else base))
          { piece := Piece.O, rot := 0, col := 2 }) 2 = base + 2
    ∧ base < Board.colHeight
        (Placement.applyStep GameConfig.standard
          (skyline GameConfig.standard (fun j => if j = 0 then 0 else base))
          { piece := Piece.O, rot := 0, col := 2 }) 2 := by
  rw [applyStep_O_flat_eq, colHeight_skyline (by decide), oProfile_two]
  exact ⟨rfl, by omega⟩

end Tetris
