import Mathlib
import Proofs.Invariants.SlotAlgebra

/-!
# The flush zone game — in-kernel verdicts on schedule design candidates

A zone of `w` columns receives its per-bag piece multiset in adversarial
order and must answer each piece with a FLUSH placement (hole-free seat:
the surface segment matches the rotation's bottom profile exactly) keeping
the normalized spread within a cap. No clears (the drain is global, not
zonal); states are normalized by subtracting the minimum height — the
base-shift quotient the plinth transport justifies.

The adversary here is ADAPTIVE (it picks each next piece after seeing the
responses), unlike `ZoneGame`'s committed-order variant — so `flushDead
… = true` is an unconditional deadness fact (no strategy survives even the
horizon within the cap), and `= false` is genuine adaptive-survival
evidence at the horizon (closure still needs separate certification).
Memoized AND-OR DFS in the accepted `ZoneGame.lean` verdict style; all
instances at zone scale (≤ 6 columns).
-/

namespace Tetris
namespace FlushZone

/-- A rotation as per-column `(bottom, top)` row offsets; width = length. -/
abbrev FShape := List (ℕ × ℕ)

/-- Heights state (relative). -/
abbrev ZS := List ℕ

/-- The drop profiles per piece, listed by rotation value (from
`Piece.shape`; checked against the green calculus by
`shapesOf_consistent`). -/
def shapesOf : Piece → List FShape
  | .O => [[(0,1),(0,1)]]
  | .I => [[(0,0),(0,0),(0,0),(0,0)], [(0,3)]]
  | .S => [[(0,0),(0,1),(1,1)], [(1,2),(0,1)]]
  | .Z => [[(1,1),(0,1),(0,0)], [(0,1),(1,2)]]
  | .T => [[(1,1),(0,1),(1,1)], [(1,1),(0,2)], [(0,0),(0,1),(0,0)], [(0,2),(1,1)]]
  | .L => [[(0,0),(0,0),(0,1)], [(0,2),(0,0)], [(0,1),(1,1),(1,1)], [(2,2),(0,2)]]
  | .J => [[(0,1),(0,0),(0,0)], [(0,2),(2,2)], [(1,1),(1,1),(0,1)], [(0,0),(0,2)]]

def pidx : Piece → ℕ
  | .O => 0 | .I => 1 | .S => 2 | .Z => 3 | .T => 4 | .L => 5 | .J => 6

def insSorted (n : ℕ) : List ℕ → List ℕ
  | [] => [n]
  | m :: ms => if n ≤ m then n :: m :: ms else m :: insSorted n ms

/-- Canonical multiset key for the remaining bag pieces. -/
def remKey (ps : List Piece) : List ℕ :=
  ps.foldl (fun acc p => insSorted (pidx p) acc) []

def minZ (h : ZS) : ℕ := h.foldl Nat.min (h.headD 0)

def normZ (h : ZS) : ZS := let m := minZ h; h.map (· - m)

def spreadZ (h : ZS) : ℕ := h.foldl Nat.max 0 - minZ h

def fitsAux (h : ZS) : ℕ → ℕ → FShape → Bool
  | _, _, [] => true
  | c, off, (b, _) :: rest =>
      h.getD c 0 == off + b && fitsAux h (c + 1) off rest

/-- Flush seat: in-bounds, and the segment matches the bottoms exactly. -/
def fits (h : ZS) (c : ℕ) (s : FShape) : Bool :=
  match s with
  | [] => false
  | (b0, _) :: _ =>
      decide (b0 ≤ h.getD c 0) && decide (c + s.length ≤ h.length) &&
        fitsAux h c (h.getD c 0 - b0) s

def applyAux : ZS → ℕ → ℕ → FShape → ZS
  | h, _, _, [] => h
  | h, c, off, (_, t) :: rest => applyAux (h.set c (off + t + 1)) (c + 1) off rest

def applyAt (h : ZS) (c : ℕ) (s : FShape) : ZS :=
  match s with
  | [] => h
  | (b0, _) :: _ => applyAux h c (h.getD c 0 - b0) s

/-- All flush responses to piece `p` keeping normalized spread ≤ cap. -/
def movesZ (spread : ℕ) (h : ZS) (p : Piece) : List ZS :=
  (shapesOf p).flatMap fun s =>
    (List.range h.length).filterMap fun c =>
      if fits h c s then
        let h' := applyAt h c s
        if spreadZ h' ≤ spread then some (normZ h') else none
      else none

abbrev Memo := Std.HashMap (ℕ × ZS × List ℕ) Bool

mutual

/-- Memoized adaptive AND-OR survival: fuel 0 = horizon reached alive;
empty bag refills (consuming one fuel); otherwise every adversary pick must
admit some flush response surviving on. -/
def surv (spread : ℕ) (bag : List Piece) (memo : Memo) (h : ZS) :
    (fuel : ℕ) → List Piece → Bool × Memo
  | 0, _ => (true, memo)
  | fuel + 1, [] => surv spread bag memo h fuel bag
  | fuel + 1, ps =>
      match memo.get? (fuel + 1, h, remKey ps) with
      | some v => (v, memo)
      | none =>
          let (v, m2) :=
            survAnd spread bag memo h fuel ps (List.range ps.length)
          (v, m2.insert (fuel + 1, h, remKey ps) v)
termination_by fuel ps => (fuel, 2, 0)

/-- AND over the adversary's remaining picks. -/
def survAnd (spread : ℕ) (bag : List Piece) (memo : Memo) (h : ZS)
    (fuel : ℕ) (ps : List Piece) : List ℕ → Bool × Memo
  | [] => (true, memo)
  | i :: is =>
      let (b1, m1) := survOr spread bag memo h fuel (ps.eraseIdx i)
        (movesZ spread h (ps.getD i Piece.O))
      if b1 then survAnd spread bag m1 h fuel ps is else (false, m1)
termination_by is => (fuel + 1, 1, is.length)

/-- OR over the player's flush responses. -/
def survOr (spread : ℕ) (bag : List Piece) (memo : Memo) (h : ZS)
    (fuel : ℕ) (rem : List Piece) : List ZS → Bool × Memo
  | [] => (false, memo)
  | h' :: hs =>
      let (b1, m1) := surv spread bag memo h' fuel rem
      if b1 then (true, m1) else survOr spread bag m1 h fuel rem hs
termination_by hs => (fuel + 1, 0, hs.length + 1)

end

/-- The verdict function: TRUE = no strategy survives `bags` bags under all
adaptive orders within the spread cap (unconditional deadness). FALSE =
adaptive survival at the horizon (evidence, not closure). -/
def flushDead (w spread : ℕ) (bag : List Piece) (bags : ℕ) : Bool :=
  !(surv spread bag (∅ : Memo) (List.replicate w 0)
    (bags * (bag.length + 1)) bag).1

/-! ## Shape-table consistency: `shapesOf` matches the green calculus. -/

/-- Bool check: every `shapesOf` entry `k` agrees columnwise with
`Piece.colBot`/`Piece.colTop` at rotation `k`. -/
def shapesConsistentB : Bool :=
  [Piece.O, Piece.I, Piece.S, Piece.Z, Piece.T, Piece.L, Piece.J].all fun p =>
    (shapesOf p).zipIdx.all fun sk =>
      sk.1.zipIdx.all fun bti =>
        (bti.1.1 == Piece.colBot p ⟨sk.2 % 4, Nat.mod_lt _ (by omega)⟩ bti.2) &&
        (bti.1.2 == Piece.colTop p ⟨sk.2 % 4, Nat.mod_lt _ (by omega)⟩ bti.2)

/-- The transcribed shape table is faithful to the model. -/
theorem shapesOf_consistent : shapesConsistentB = true := by decide

/-! ## Verdicts, batch 1 -/

/-- **The isolated 3-column {O,L,J} zone is DEAD** (spread ≤ 6, horizon 4
bags): finding F2 as an unconditional in-kernel fact — gravity refuses the
tiling T-parity permits. -/
theorem olj3_verdict :
    flushDead 3 6 [Piece.O, Piece.L, Piece.J] 4 = true := by
  native_decide

/-- **The 4-column {O,L,J} zone is DEAD** (spread ≤ 6, horizon 4 bags):
widening by one column does not rescue the O/L/J economy. -/
theorem olj4_verdict :
    flushDead 4 6 [Piece.O, Piece.L, Piece.J] 4 = true := by
  native_decide

/-- **The 4-column {S,Z,T} pool is DEAD** (spread ≤ 6, horizon 4 bags):
two step-windows cannot absorb the roughness pool — T's step-flip strands
S or Z, as the slot algebra predicts. -/
theorem szt4_verdict :
    flushDead 4 6 [Piece.S, Piece.Z, Piece.T] 4 = true := by
  native_decide

/-! ## Verdicts, batch 2 — the ≤ 6-column design space is dead -/

/-- The 5-column {O,L,J} zone survives 3 bags (spread ≤ 6) — transient. -/
theorem olj5_alive3 :
    flushDead 5 6 [Piece.O, Piece.L, Piece.J] 3 = false := by
  native_decide

/-- **…but is DEAD at 6 bags** (same caps): the 3-bag survival is a horizon
artifact; the O/L/J economy does not close at width 5 either. -/
theorem olj5_dead6 :
    flushDead 5 6 [Piece.O, Piece.L, Piece.J] 6 = true := by
  native_decide

/-- **Three step-windows (6 columns) cannot absorb the {S,Z,T} pool.** -/
theorem szt6_verdict :
    flushDead 6 6 [Piece.S, Piece.Z, Piece.T] 3 = true := by
  native_decide

/-- **The entangled {O,L,J,T} candidate is DEAD at width 5.** -/
theorem oljt5_verdict :
    flushDead 5 6 [Piece.O, Piece.L, Piece.J, Piece.T] 3 = true := by
  native_decide

/-- **{O,L,J,T} is DEAD at width 6 too.** -/
theorem oljt6_verdict :
    flushDead 6 6 [Piece.O, Piece.L, Piece.J, Piece.T] 2 = true := by
  native_decide

/-- **The entangled {O,L,J,S,Z} candidate is DEAD at width 6.** -/
theorem oljsz6_verdict :
    flushDead 6 6 [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z] 2 = true := by
  native_decide

/-- **The full non-I pool is DEAD at the 6-column zone-scale line, even at
spread ≤ 8** — no ≤ 6-column window hosts the six-piece flush economy. -/
theorem all6_verdict :
    flushDead 6 8 [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T] 2
      = true := by
  native_decide

/-! ## Drain invisibility

The global drain subtracts 4 from every band column exactly (it fires at
base ≥ 4), and states here are min-normalized — so any drain schedule is a
NO-OP for this game (`normZ_shift`). The batch-1/2 dead verdicts are
therefore DRAIN-ROBUST: they hold under every drain schedule. What the
drain changes in the real band is modeled instead by the band-I pools and
the honest spread caps below. -/

theorem foldl_min_map_sub (k : ℕ) : ∀ (t : List ℕ) (acc : ℕ),
    (∀ x ∈ t, k ≤ x) → k ≤ acc →
    (t.map (· - k)).foldl Nat.min (acc - k) = t.foldl Nat.min acc - k := by
  intro t
  induction t with
  | nil => intro acc _ _; rfl
  | cons a t ih =>
      intro acc hall hacc
      simp only [List.map_cons, List.foldl_cons]
      have ha : k ≤ a := hall a (List.mem_cons_self ..)
      rw [show Nat.min (acc - k) (a - k) = Nat.min acc a - k by
        simp only [Nat.min_def]; split_ifs <;> omega]
      exact ih (Nat.min acc a)
        (fun x hx => hall x (List.mem_cons_of_mem _ hx))
        (by simp only [Nat.min_def]; split_ifs <;> omega)

theorem minZ_map_sub (h : ZS) (k : ℕ) (hall : ∀ x ∈ h, k ≤ x) :
    minZ (h.map (· - k)) = minZ h - k := by
  cases h with
  | nil => simp [minZ]
  | cons a t =>
      have ha : k ≤ a := hall a (List.mem_cons_self ..)
      simp only [minZ, List.map_cons, List.headD_cons, List.foldl_cons]
      rw [show Nat.min (a - k) (a - k) = Nat.min a a - k by
        simp [Nat.min_self]]
      exact foldl_min_map_sub k t (Nat.min a a)
        (fun x hx => hall x (List.mem_cons_of_mem _ hx))
        (by simp only [Nat.min_self]; omega)

theorem le_minZ (h : ZS) (k : ℕ) (hall : ∀ x ∈ h, k ≤ x) : k ≤ minZ h ∨ h = [] := by
  cases h with
  | nil => exact Or.inr rfl
  | cons a t =>
      left
      have : ∀ (t : List ℕ) (acc : ℕ), (∀ x ∈ t, k ≤ x) → k ≤ acc →
          k ≤ t.foldl Nat.min acc := by
        intro t
        induction t with
        | nil => intro acc _ hacc; exact hacc
        | cons b t ih =>
            intro acc hall' hacc
            simp only [List.foldl_cons]
            have hb : k ≤ b := hall' b (List.mem_cons_self ..)
            exact ih (Nat.min acc b)
              (fun x hx => hall' x (List.mem_cons_of_mem _ hx))
              (by simp only [Nat.min_def]; split_ifs <;> omega)
      have ha : k ≤ a := hall a (List.mem_cons_self ..)
      simp only [minZ, List.headD_cons]
      exact this (a :: t) a hall ha

/-- **Drain invisibility**: an exact uniform drop is a no-op after
normalization. -/
theorem normZ_shift (h : ZS) (k : ℕ) (hall : ∀ x ∈ h, k ≤ x) :
    normZ (h.map (· - k)) = normZ h := by
  cases hle : h with
  | nil => rfl
  | cons a t =>
      rw [← hle]
      have hmin : k ≤ minZ h := by
        rcases le_minZ h k hall with h' | h'
        · exact h'
        · rw [hle] at h'; cases h'
      unfold normZ
      rw [minZ_map_sub h k hall, List.map_map]
      apply List.map_congr_left
      intro x hx
      have : k ≤ x := hall x hx
      simp only [Function.comp_apply]
      omega

/-! ## The scheduled game: per-bag piece lists (cycling)

Models schedules like "the band receives the I in 3 bags of 10". The
un-scheduled `surv`/`flushDead` above are kept byte-identical (the
committed verdicts depend on them). -/

abbrev MemoP := Std.HashMap (ℕ × ZS × ℕ × List ℕ) Bool

mutual

def survP (spread : ℕ) (sched : List (List Piece)) (memo : MemoP) (h : ZS) :
    (fuel : ℕ) → ℕ → List Piece → Bool × MemoP
  | 0, _, _ => (true, memo)
  | fuel + 1, ph, [] =>
      let ph' := (ph + 1) % sched.length
      survP spread sched memo h fuel ph' (sched.getD ph' [])
  | fuel + 1, ph, ps =>
      match memo.get? (fuel + 1, h, ph, remKey ps) with
      | some v => (v, memo)
      | none =>
          let (v, m2) :=
            survAndP spread sched memo h fuel ph ps (List.range ps.length)
          (v, m2.insert (fuel + 1, h, ph, remKey ps) v)
termination_by fuel ph ps => (fuel, 2, 0)

def survAndP (spread : ℕ) (sched : List (List Piece)) (memo : MemoP) (h : ZS)
    (fuel : ℕ) (ph : ℕ) (ps : List Piece) : List ℕ → Bool × MemoP
  | [] => (true, memo)
  | i :: is =>
      let (b1, m1) := survOrP spread sched memo h fuel ph (ps.eraseIdx i)
        (movesZ spread h (ps.getD i Piece.O))
      if b1 then survAndP spread sched m1 h fuel ph ps is else (false, m1)
termination_by is => (fuel + 1, 1, is.length)

def survOrP (spread : ℕ) (sched : List (List Piece)) (memo : MemoP) (h : ZS)
    (fuel : ℕ) (ph : ℕ) (rem : List Piece) : List ZS → Bool × MemoP
  | [] => (false, memo)
  | h' :: hs =>
      let (b1, m1) := survP spread sched memo h' fuel ph rem
      if b1 then (true, m1) else survOrP spread sched m1 h fuel ph rem hs
termination_by hs => (fuel + 1, 0, hs.length + 1)

end

/-- Scheduled verdict: TRUE = no strategy survives `bags` bags of the
cycling schedule under all adaptive orders within the spread cap. -/
def flushDeadP (w spread : ℕ) (sched : List (List Piece)) (bags : ℕ) : Bool :=
  let maxLen := sched.foldl (fun a b => Nat.max a b.length) 0
  !(survP spread sched (∅ : MemoP) (List.replicate w 0)
    (bags * (maxLen + 1)) 0 (sched.getD 0 [])).1

/-! ## Verdicts, batch 3 — the I-pools are dead too

The band-I (the piece the earlier pools excluded) does not rescue any
≤ 6-column window, at spread caps up to the honest ceiling budget, nor
under the rate-faithful 3-of-10 schedule. Together with `normZ_shift`
(drain invisibility), the ≤ 6-column zone design space is comprehensively
dead under drain-faithful modeling. -/

/-- **{O,L,J,I} is DEAD at width 4** (spread ≤ 6, horizon 6 bags). -/
theorem olji4_verdict :
    flushDead 4 6 [Piece.O, Piece.L, Piece.J, Piece.I] 6 = true := by
  native_decide

/-- **{O,L,J,I} is DEAD at width 5** (spread ≤ 6, horizon 6 bags). -/
theorem olji5_verdict :
    flushDead 5 6 [Piece.O, Piece.L, Piece.J, Piece.I] 6 = true := by
  native_decide

/-- **{O,L,J,I} is DEAD at width 5 even at spread ≤ 10** — the honest
ceiling budget does not save it. -/
theorem olji5_k10_verdict :
    flushDead 5 10 [Piece.O, Piece.L, Piece.J, Piece.I] 6 = true := by
  native_decide

/-- **{S,Z,T,I} is DEAD at width 4** (spread ≤ 6, horizon 6 bags). -/
theorem szti4_verdict :
    flushDead 4 6 [Piece.S, Piece.Z, Piece.T, Piece.I] 6 = true := by
  native_decide

/-- **{S,Z,T,I} is DEAD at width 6** (spread ≤ 6, horizon 4 bags). -/
theorem szti6_verdict :
    flushDead 6 6 [Piece.S, Piece.Z, Piece.T, Piece.I] 4 = true := by
  native_decide

/-- **{S,Z,T,I} is DEAD at width 6 even at spread ≤ 10.** -/
theorem szti6_k10_verdict :
    flushDead 6 10 [Piece.S, Piece.Z, Piece.T, Piece.I] 4 = true := by
  native_decide

/-- **All seven pieces are DEAD at width 6** (spread ≤ 8, horizon 3 bags). -/
theorem all7_verdict :
    flushDead 6 8 [Piece.O, Piece.I, Piece.S, Piece.Z, Piece.T, Piece.L,
      Piece.J] 3 = true := by
  native_decide

/-- **The rate-faithful slice is DEAD**: all six pieces every bag with the
band-I in 3 bags of 10 (the rate law's split), width 6, spread ≤ 10,
horizon 10 bags. -/
theorem mixed10_verdict :
    flushDeadP 6 10
      [[Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T, Piece.I],
       [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T],
       [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T],
       [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T, Piece.I],
       [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T],
       [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T],
       [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T, Piece.I],
       [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T],
       [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T],
       [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T]] 10 = true := by
  native_decide

end FlushZone
end Tetris
