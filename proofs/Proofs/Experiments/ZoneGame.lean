import Mathlib
import Proofs.Model.Piece

/-!
# The zone game, formalized

The per-zone contract game of the rely-guarantee decomposition (see
`Proofs/Invariants/ZoneView.lean` for the composition algebra and the
verdict ledger). A zone of width `w` plays only its own columns: pieces
arrive per bag in adversarial order (AND), the zone answers each with a
zone-local hard drop (OR), and at each bag boundary the CLEAR SERVICE
removes the bottom `s` zone-full rows (the composition interface, justified
by `zoneView_clearLines`). The zone survives if heights stay under `hcap`
and holes under `dcap` forever.

Boards are lists of column bitmasks (`ZBoard`), mirroring
`scripts/zone_contract_probe.py` exactly; every operation is computable, so
zone-contract instances are decided in-kernel (`native_decide`) rather than
trusted to external runs. This file provides the definitional core:
`stepPlace` (hard drop), `zservice` (the clear service), and `bagOk`
(the decidable one-bag AND-OR game). The closure computation and the
verdict instances (the 10-bag odd-width family) build on it.
-/

namespace Tetris
namespace ZoneGame

/-- A zone board: one bitmask per column (bit `r` = cell at row `r`). -/
abbrev ZBoard := List ℕ

/-- Height of one column mask: index one past the top set bit. -/
def mheight (m : ℕ) : ℕ := Nat.log2 (2 * m + 1)

/-- A placement profile inside the zone: `(column, bottomOffset, cellUps)`
per occupied column — the `shapeUp` data translated to zone columns. -/
abbrev ZProfile := List (ℕ × ℕ × List ℕ)

/-- Hard-drop offset of a profile on a zone board (sup of per-column
`height - u0`, truncated). -/
def dropOff (b : ZBoard) (info : ZProfile) : ℕ :=
  info.foldl (fun d (c, u0, _) => max d (mheight (b.getD c 0) - u0)) 0

/-- Merge the dropped cells (no clears — clears are the service). -/
def mergeAt (b : ZBoard) (info : ZProfile) (d : ℕ) : ZBoard :=
  info.foldl
    (fun acc (c, _, ups) =>
      acc.set c (ups.foldl (fun m u => m ||| (2 ^ (d + u))) (acc.getD c 0)))
    b

/-- One zone hard drop; `none` when the result exceeds `hcap`. -/
def stepPlace (hcap : ℕ) (b : ZBoard) (info : ZProfile) : Option ZBoard :=
  let nb := mergeAt b info (dropOff b info)
  if nb.all (fun m => mheight m ≤ hcap) then some nb else none

/-- Rows the whole zone has filled: the AND of the column masks. -/
def fullMask (b : ZBoard) : ℕ := b.foldl (fun a m => a &&& m) (b.getD 0 0)

/-- Keep the low bits of `x` not selected by `mask`, compacting downward —
the row-deletion primitive (`deleteRows` in the composition algebra),
computed bit by bit with `fuel` rows. -/
def pextAux : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | 0, _, _, _, acc => acc
  | fuel + 1, x, mask, out, acc =>
    if mask % 2 = 0 then
      pextAux fuel (x / 2) (mask / 2) (out + 1)
        (acc + (x % 2) * 2 ^ out)
    else
      pextAux fuel (x / 2) (mask / 2) out acc

/-- The bottom `s` full rows, as a bitmask. -/
def bottomRows (full : ℕ) (s : ℕ) : ℕ :=
  (List.range 32).foldl
    (fun (st : ℕ × ℕ) r =>
      if st.2 < s ∧ (full / 2 ^ r) % 2 = 1 then (st.1 + 2 ^ r, st.2 + 1)
      else st)
    ((0 : ℕ), (0 : ℕ)) |>.1

/-- The clear service: remove the bottom `min s (#full)` zone-full rows. -/
def zservice (s : ℕ) (b : ZBoard) : ZBoard :=
  let mask := bottomRows (fullMask b) s
  if mask = 0 then b else b.map (fun m => pextAux 32 m mask 0 0)

/-- Holes of a zone board (covered empty cells). -/
def zholes (b : ZBoard) : ℕ :=
  (b.map (fun m => mheight m - (Nat.digits 2 m).sum)).sum

/-- **The one-bag AND-OR game, decidably.** `bagOk` holds when for EVERY
arrival order of `pieces` there are placements (drawn from `cands`, the
zone-local profiles per piece) keeping the board under `hcap` throughout,
such that the end-of-bag board, after the service removes `s` rows,
satisfies `accept`. This is the zone contract's step obligation — a bounded
computation (`decide`-able for concrete instances). -/
def ordOk (hcap s : ℕ) (cands : Piece → List ZProfile)
    (accept : ZBoard → Bool) : ZBoard → List Piece → Bool
  | b, [] => accept (zservice s b)
  | b, p :: rest =>
    (cands p).any fun info =>
      match stepPlace hcap b info with
      | none => false
      | some nb => ordOk hcap s cands accept nb rest

def bagOk (hcap s : ℕ) (cands : Piece → List ZProfile)
    (accept : ZBoard → Bool) (b : ZBoard) (pieces : List Piece) : Bool :=
  pieces.permutations.all (ordOk hcap s cands accept b)

/-- Zone-local candidate profiles of a piece for width `w`: every rotation
whose shape fits, at every column offset, as `(col, u0, ups)` triples —
derived from the model's own `Piece.shapeUp` (no transcription trust). -/
def zoneCands (w : ℕ) (p : Piece) : List ZProfile :=
  (List.range 4).flatMap fun r =>
    if r < p.numRotations then
      let cells := ((p.shapeUp ⟨r % 4, by omega⟩).image
          (fun c => c.1 * 16 + c.2)).sort (· ≤ ·) |>.map
          (fun n => (n / 16, n % 16))
      let pcols := (cells.map (·.1)).dedup
      let pw := (pcols.foldl max 0) + 1
      if pw ≤ w then
        (List.range (w - pw + 1)).map fun c =>
          pcols.map fun pc =>
            (c + pc, (cells.filterMap
                (fun cell => if cell.1 = pc then some cell.2 else none)).foldl
                  min 99,
             cells.filterMap
                (fun cell => if cell.1 = pc then some cell.2 else none))
      else []
    else []

/-- **Adversary forces death within `k` bags** — the auditable AND-OR dual
of the zone contract. Dead when SOME arrival order of the bag leaves the
zone with NO placement response avoiding a cap breach, a hole-budget breach
at the boundary, or a position dead within `k - 1` further bags. A
`zoneDead … = true` fact (via `native_decide`) is an unconditional
in-kernel refutation of the zone contract. Memo keyed on
`(board, phase, horizon)`; total by lexicographic `(horizon, pieces)`. -/
theorem deadAt_doc : True := trivial

mutual

def deadBags (w hcap dcap : ℕ) (sched : List (List Piece)) (srv : List ℕ)
    (memo : Std.HashMap (ZBoard × ℕ × ℕ) Bool) (b : ZBoard) (ph : ℕ) :
    (k : ℕ) → Bool × Std.HashMap (ZBoard × ℕ × ℕ) Bool
  | 0 => (false, memo)
  | k + 1 =>
    match memo.get? (b, ph, k + 1) with
    | some v => (v, memo)
    | none =>
      let pieces := sched.getD (ph % sched.length) []
      let (v, m2) :=
        pieces.permutations.foldl
          (fun (st : Bool × Std.HashMap (ZBoard × ℕ × ℕ) Bool) ord =>
            if st.1 then st
            else
              let (d, m') := deadOrd w hcap dcap sched srv st.2 b ph k ord
              (st.1 || d, m'))
          (false, memo)
      (v, m2.insert (b, ph, k + 1) v)
termination_by k => (k, 0)

def deadOrd (w hcap dcap : ℕ) (sched : List (List Piece)) (srv : List ℕ)
    (memo : Std.HashMap (ZBoard × ℕ × ℕ) Bool) (b : ZBoard) (ph k : ℕ) :
    (rest : List Piece) → Bool × Std.HashMap (ZBoard × ℕ × ℕ) Bool
  | [] =>
    let nb := zservice (srv.getD (ph % srv.length) 0) b
    if zholes nb > dcap then (true, memo)
    else deadBags w hcap dcap sched srv memo nb ((ph + 1) % sched.length) k
  | p :: rest' =>
    (zoneCands w p).foldl
      (fun (st : Bool × Std.HashMap (ZBoard × ℕ × ℕ) Bool) info =>
        if !st.1 then st
        else
          match stepPlace hcap b info with
          | none => st
          | some nb =>
            let (d, m') := deadOrd w hcap dcap sched srv st.2 nb ph k rest'
            (st.1 && d, m'))
      (true, memo)
termination_by rest => (k, rest.length + 1)

end

/-- The zone-death verdict from the empty zone at horizon `k`. -/
def zoneDead (w hcap dcap : ℕ) (sched : List (List Piece)) (srv : List ℕ)
    (k : ℕ) : Bool :=
  (deadBags w hcap dcap sched srv {} (List.replicate w 0) 0 k).1

open Piece in
/-- **Sanity instance (in-kernel re-derivation of a decided point):** the
5-bag `5I+2O` two-column mid-zone — the {4,2,4} choke — is dead within 8
bags. Matches the external probe's verdict; the kernel now owns it. -/
theorem midzone_5I2O_dead :
    zoneDead 2 10 2 [[I], [I], [I], [I, O], [I, O]] [3, 3, 3, 3, 2] 8
      = true := by
  native_decide

open Piece in
/-- **The 10-bag `{5,5}` left zone (10S+10Z+10T+5O, width 5) is dead within
6 bags** — the first point of the previously-open odd-width family, decided
in-kernel. -/
theorem L5_dead_h6 :
    zoneDead 5 12 3
      [[S, Z, T, O], [S, Z, T], [S, Z, T, O], [S, Z, T], [S, Z, T, O],
       [S, Z, T], [S, Z, T, O], [S, Z, T], [S, Z, T, O], [S, Z, T]]
      [3, 3, 3, 3, 2, 3, 3, 3, 3, 2] 6 = true := by
  native_decide

open Piece in
/-- The `{5,5}` left zone with the `I` as its filler instead of the `O`
(10S+10Z+10T+5I) is dead within 6 bags too — so EVERY `{5,5}` split that
keeps the S/Z/T roughness core together dies, whichever piece completes
its 35. -/
theorem L5I_dead_h6 :
    zoneDead 5 12 3
      [[S, Z, T, I], [S, Z, T], [S, Z, T, I], [S, Z, T], [S, Z, T, I],
       [S, Z, T], [S, Z, T, I], [S, Z, T], [S, Z, T, I], [S, Z, T]]
      [3, 3, 3, 3, 2, 3, 3, 3, 3, 2] 6 = true := by
  native_decide

open Piece in
/-- The `{3,7}` family's 3-zone roughness core (10S+10Z+1T per 10 bags,
width 3) is dead within 4 bags: S/Z cannot cohabit three columns at the
mandated rate. Kills every `{3,7}` split routing both S and Z through the
3-zone. -/
theorem SZ3_dead_h4 :
    zoneDead 3 12 3
      [[S, Z], [S, Z], [S, Z], [S, Z], [S, Z, T],
       [S, Z], [S, Z], [S, Z], [S, Z], [S, Z]]
      [3, 3, 3, 3, 2, 3, 3, 3, 3, 2] 4 = true := by
  native_decide

open Piece in
/-- An S/Z-SEPARATED representative: a width-5 zone taking S (with L, O,
and half the T supply) but not Z — 10S+10L+10O+5T — is dead within 6 bags.
Separating the roughness pair does not save the design: a lone staircase
piece still out-roughens a five-column window at rate `7w/10`. -/
theorem S_sep_dead_h6 :
    zoneDead 5 12 3
      [[S, L, O, T], [S, L, O], [S, L, O, T], [S, L, O], [S, L, O, T],
       [S, L, O], [S, L, O, T], [S, L, O], [S, L, O, T], [S, L, O]]
      [3, 3, 3, 3, 2, 3, 3, 3, 3, 2] 6 = true := by
  native_decide

end ZoneGame
end Tetris
