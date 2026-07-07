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

end ZoneGame
end Tetris
