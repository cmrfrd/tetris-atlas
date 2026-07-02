import Proofs.Model.Piece

/-!
# Shape-table validation for `Search/CarrierSearch.lean`

Recomputes, from the actual model (`Tetris.Piece.shapeUp`), the per-column
bottom/top row-offsets of every piece orientation, and prints them in the
`Orient` format used by the search harness. Run:

    lake env lean Search/ValidateShapes.lean

and diff the output against the `orients` table in `Search/CarrierSearch.lean`.
Also checks the structural fact the search relies on: every orientation fills
each of its columns contiguously from `bottoms[c]` to `tops[c]` (no internal
vertical gaps — so "flush ⇒ hole-free" and the height update `h' = off + top + 1`
are exact). Uses only decidable membership in `shapeUp` (cells live in the 4×4 box).
-/

open Tetris

def pieceCode : Piece → Nat
  | Piece.O => 0 | Piece.I => 1 | Piece.S => 2 | Piece.Z => 3
  | Piece.T => 4 | Piece.L => 5 | Piece.J => 6

def rotsOf : Piece → List Nat
  | Piece.O => [0]
  | Piece.I | Piece.S | Piece.Z => [0, 1]
  | _ => [0, 1, 2, 3]

def main : IO Unit := do
  let pieces := [Piece.O, Piece.I, Piece.S, Piece.Z, Piece.T, Piece.L, Piece.J]
  let mut allContiguous := true
  for p in pieces do
    for r in rotsOf p do
      let rot : Tetris.Rotation := ⟨r % 4, by omega⟩
      let mem (c ρ : Nat) : Bool := decide ((c, ρ) ∈ p.shapeUp rot)
      let colUsed (c : Nat) : Bool := (List.range 4).any (fun ρ => mem c ρ)
      let width := ((List.range 4).filter colUsed).length
      let mut bots : List Nat := []
      let mut tops : List Nat := []
      for c in List.range width do
        let rows := (List.range 4).filter (fun ρ => mem c ρ)
        let bot := rows.foldl min 1000
        let top := rows.foldl max 0
        bots := bots ++ [bot]
        tops := tops ++ [top]
        for ρ in List.range (top + 1 - bot) do
          if !(rows.contains (bot + ρ)) then
            allContiguous := false
            IO.println s!"NON-CONTIGUOUS: piece {pieceCode p} rot {r} col {c}"
      IO.println s!"⟨{pieceCode p}, {r}, #{bots}, #{tops}⟩"
  IO.println (if allContiguous then "contiguity: OK (flush ⇒ hole-free exact)" else "contiguity: FAILED")

#eval main
