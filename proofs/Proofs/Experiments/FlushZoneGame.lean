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

end FlushZone
end Tetris
