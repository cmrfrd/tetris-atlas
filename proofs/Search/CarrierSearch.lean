import Std.Data.HashMap

/-!
# Carrier search: AND-OR survival game over hole-free skylines (design tool)

Design-time search for a finite closed carrier feeding
`tetrisSolvableValid_of_strategy_bagInvariant`. NOT part of the proof library —
this is an experiment harness whose output (a finite automaton of skyline
profiles) is to be re-proven symbolically in Lean.

Model (mirrors `Proofs/Model` exactly on the hole-free fragment; the shape
tables below are validated against `Piece.shapeUp` by `Search/ValidateShapes.lean`):
- Board = skyline of a 10-column height profile; col 0 is the reserved well
  (height 0 always). Cols 1..9 are the band.
- A placement is *flush* iff every piece column's bottom cell lands exactly on
  the column top (no holes created). Flush placements that avoid the empty well
  never clear lines (the well blocks every row), so the profile just gains the
  piece's column deltas.
- The only clearing move is the drain: vertical I into the well when every band
  column is ≥ 4 — exactly 4 rows clear and the band drops uniformly by 4
  (`applyStep_vertI_well`).

Game: states (band profile, bag). Adversary draws any pending piece; we choose
any flush placement (or drain). Survive = stay within caps forever.
Solve: forward reachability then backward death propagation (AND-OR gfp).
Extract: a deterministic first-alive-choice policy automaton.
-/

namespace CarrierSearch

open Std

/-- Column profile of one piece orientation: `bottoms[c]`/`tops[c]` are the
lowest/highest occupied row-offsets of `shapeUp` in piece-column `c`.
Piece codes 0..6 = O I S Z T L J (matches `Tetris.Piece` constructor order). -/
structure Orient where
  piece : Nat
  rot : Nat
  bottoms : Array Nat
  tops : Array Nat
deriving Repr, Inhabited

/-- All orientations, derived by hand from `Piece.shape`/`shapeUp`
(validated by `Search/ValidateShapes.lean`). -/
def orients : Array Orient := #[
  -- O
  ⟨0, 0, #[0,0], #[1,1]⟩,
  -- I horizontal / vertical
  ⟨1, 0, #[0,0,0,0], #[0,0,0,0]⟩,
  ⟨1, 1, #[0], #[3]⟩,
  -- S horizontal (needs (m,m,m+1)) / vertical (needs (m+1,m))
  ⟨2, 0, #[0,0,1], #[0,1,1]⟩,
  ⟨2, 1, #[1,0], #[2,1]⟩,
  -- Z horizontal (needs (m+1,m,m)) / vertical (needs (m,m+1))
  ⟨3, 0, #[1,0,0], #[1,1,0]⟩,
  ⟨3, 1, #[0,1], #[1,2]⟩,
  -- T: rot0 dip-filler (m+1,m,m+1)→flat; rot1 step (m+1,m); rot2 flat-3 bump; rot3 step (m,m+1)
  ⟨4, 0, #[1,0,1], #[1,1,1]⟩,
  ⟨4, 1, #[1,0], #[1,2]⟩,
  ⟨4, 2, #[0,0,0], #[0,1,0]⟩,
  ⟨4, 3, #[0,1], #[2,1]⟩,
  -- L: rot0 flat-3 (+1,+1,+2); rot1 flat-2→(2,0); rot2 (m,m+1,m+1)→flat; rot3 (m+2,m)→flat
  ⟨5, 0, #[0,0,0], #[0,0,1]⟩,
  ⟨5, 1, #[0,0], #[2,0]⟩,
  ⟨5, 2, #[0,1,1], #[1,1,1]⟩,
  ⟨5, 3, #[2,0], #[2,2]⟩,
  -- J: rot0 flat-3 (+2,+1,+1); rot1 (m,m+2)→flat; rot2 (m+1,m+1,m)→flat; rot3 flat-2→(0,2)
  ⟨6, 0, #[0,0,0], #[1,0,0]⟩,
  ⟨6, 1, #[0,2], #[2,2]⟩,
  ⟨6, 2, #[1,1,0], #[1,1,1]⟩,
  ⟨6, 3, #[0,0], #[0,2]⟩
]

/-! ## Configuration -/

def COLS : Nat := 10

/-- Heights: 10 entries, index 0 = well (always 0 in well mode). -/
abbrev H := Array Nat

def initH : H := Array.replicate 10 0

/-- Runtime-configurable parameters (CLI: `carriersearch hcap spread topk budget rank`). -/
structure Cfg where
  hcap : Nat := 10
  spreadcap : Nat := 4
  topk : Nat := 2
  budget : Nat := 2000000
  /-- 0 = land-low (offset, wmax, col); 1 = leveling (new spread, new max, col) -/
  rank : Nat := 1
  /-- Zone pairing index 0..14 (pairings of {O,S,Z,T,L,J} into 3 zones of
  3 cols: 1-3, 4-6, 7-9), or 99 = no zone restriction. -/
  pairing : Nat := 99
  /-- true = no reserved well: all 10 columns usable, rows clear continuously
  (profile renormalized by subtracting its min after each placement). -/
  wellless : Bool := false
  /-- start profile (min-0 normalized when wellless) -/
  seed : H := Array.replicate 10 0

/-- Encode band heights (each ≤ 15) and bag mask into one UInt64. -/
def encode (h : H) (bag : Nat) : UInt64 := Id.run do
  let mut k : UInt64 := 0
  for j in [0:10] do
    k := (k <<< 5) ||| UInt64.ofNat (h[j]!)
  return (k <<< 7) ||| UInt64.ofNat bag

/-- Flush drop of orientation `o` at leftmost column `c` (band only, avoid well).
Returns the new profile if the placement is flush (hole-free) and inside caps. -/
def tryPlace (cfg : Cfg) (h : H) (o : Orient) (c : Nat) (hiCol : Nat) : Option H := Id.run do
  let w := o.bottoms.size
  let loCol := if cfg.wellless then 0 else 1
  if c < loCol || c + w > COLS || c + w - 1 > hiCol then return none
  -- flush: h[c+i] - bottoms[i] must be constant and ≥ 0
  let need0 : Int := (h[c]! : Int) - (o.bottoms[0]! : Int)
  if need0 < 0 then return none
  for i in [1:w] do
    if (h[c+i]! : Int) - (o.bottoms[i]! : Int) != need0 then return none
  let off := need0.toNat
  let mut h' := h
  for i in [0:w] do
    h' := h'.set! (c+i) (off + o.tops[i]! + 1)
  -- caps
  if cfg.wellless then
    -- continuous clearing: rows 0..min−1 are full on a skyline ⇒ they clear
    let mut mn := h'[0]!
    for j in [1:10] do
      mn := min mn (h'[j]!)
    if mn > 0 then
      h' := h'.map (· - mn)
    let mut mx := 0
    for j in [0:10] do
      mx := max mx (h'[j]!)
    if mx > cfg.spreadcap then return none
    return some h'
  let mut mx := 0
  let mut mn := cfg.hcap + 100
  for j in [1:10] do
    mx := max mx (h'[j]!)
    mn := min mn (h'[j]!)
  if mx > cfg.hcap || mx - mn > cfg.spreadcap then return none
  return some h'

def drainOk (h : H) : Bool := Id.run do
  for j in [1:10] do
    if h[j]! < 4 then return false
  return true

def drain (h : H) : H := Id.run do
  let mut h' := h
  for j in [1:10] do
    h' := h'.set! j (h[j]! - 4)
  return h'

/-- The 15 pairings of the six non-I pieces {O,S,Z,T,L,J} (codes 0,2,3,4,5,6)
into 3 unordered pairs. Pair i of pairing k gets zone i (cols 3i+1..3i+3). -/
def pairings : Array (Array (Nat × Nat)) := #[
  #[(0,2),(3,4),(5,6)], #[(0,2),(3,5),(4,6)], #[(0,2),(3,6),(4,5)],
  #[(0,3),(2,4),(5,6)], #[(0,3),(2,5),(4,6)], #[(0,3),(2,6),(4,5)],
  #[(0,4),(2,3),(5,6)], #[(0,4),(2,5),(3,6)], #[(0,4),(2,6),(3,5)],
  #[(0,5),(2,3),(4,6)], #[(0,5),(2,4),(3,6)], #[(0,5),(2,6),(3,4)],
  #[(0,6),(2,3),(4,5)], #[(0,6),(2,4),(3,5)], #[(0,6),(2,5),(3,4)]
]

/-- Allowed columns for piece `p` under pairing `k`: its zone's columns
(zone i = cols {3i+1, 3i+2, 3i+3}); I (code 1) may use any band column.
Returns (loCol, hiCol) inclusive. -/
def zoneOf (k p : Nat) : Nat × Nat := Id.run do
  if k >= 15 || p == 1 then return (1, 9)
  let prs := pairings[k]!
  for i in [0:3] do
    let (a, b) := prs[i]!
    if p == a || p == b then return (3*i+1, 3*i+3)
  return (1, 9)

/-- All successor profiles for piece `p` (0..6), ranked and truncated to `TOPK`.
Rank: land as low as possible (offset), then lowest resulting window top, then
leftmost. For I, the drain (when legal) always ranks first and does not count
against `TOPK`. -/
def succProfiles (cfg : Cfg) (h : H) (p : Nat) : Array H := Id.run do
  -- I: drain is FORCED when legal (the drain-aware regulator); no well ⇒ no drain
  if p == 1 && !cfg.wellless && drainOk h then
    return #[drain h]
  let (lo, hi) := if cfg.wellless then (0, 9) else zoneOf cfg.pairing p
  let mut cands : Array (Nat × Nat × Nat × H) := #[]
  for o in orients do
    if o.piece == p then
      for c in [lo:hi+1] do
        match tryPlace cfg h o c hi with
        | some h' =>
          let k1k2 : Nat × Nat := Id.run do
            if cfg.rank == 0 then
              let off := h'[c]! - o.tops[0]! - 1
              let mut wmax := 0
              for i in [0:o.bottoms.size] do
                wmax := max wmax (h'[c+i]!)
              return (off, wmax)
            else
              let mut mx := 0
              let mut mn := 1000
              for j in [1:10] do
                mx := max mx (h'[j]!)
                mn := min mn (h'[j]!)
              return (mx - mn, mx)
          cands := cands.push (k1k2.1, k1k2.2, c, h')
        | none => pure ()
  let sorted := cands.qsort (fun a b =>
    a.1 < b.1 || (a.1 == b.1 && (a.2.1 < b.2.1 || (a.2.1 == b.2.1 && a.2.2.1 < b.2.2.1))))
  let mut out : Array H := #[]
  for i in [0:min cfg.topk sorted.size] do
    out := out.push sorted[i]!.2.2.2
  return out

def drawBag (bag p : Nat) : Nat :=
  let b := bag ^^^ (1 <<< p)
  if b == 0 then 127 else b

/-! ## Graph node bookkeeping -/

structure Node where
  h : H
  bag : Nat
deriving Inhabited

partial def solve (cfg : Cfg) : IO Unit := do
  -- forward BFS
  let mut idOf : HashMap UInt64 Nat := {}
  let mut nodes : Array Node := #[]
  let mut queue : Array Nat := #[]
  let initBag := 127
  let k0 := encode cfg.seed initBag
  idOf := idOf.insert k0 0
  nodes := nodes.push ⟨cfg.seed, initBag⟩
  queue := queue.push 0
  -- succs[nid] : Array (piece, Array succ-nid)
  let mut succs : Array (Array (Nat × Array Nat)) := #[]
  let mut qi := 0
  let mut aborted := false
  while qi < queue.size do
    if nodes.size > cfg.budget then
      aborted := true
      break
    let nid := queue[qi]!
    qi := qi + 1
    let n := nodes[nid]!
    let mut perPiece : Array (Nat × Array Nat) := #[]
    for p in [0:7] do
      if n.bag &&& (1 <<< p) != 0 then
        let hs := succProfiles cfg n.h p
        let bag' := drawBag n.bag p
        let mut ids : Array Nat := #[]
        for h' in hs do
          let k := encode h' bag'
          match idOf.get? k with
          | some i => ids := ids.push i
          | none =>
            let i := nodes.size
            idOf := idOf.insert k i
            nodes := nodes.push ⟨h', bag'⟩
            queue := queue.push i
            ids := ids.push i
        perPiece := perPiece.push (p, ids)
    succs := succs.push perPiece
  if aborted then
    IO.println s!"ABORTED: node budget {cfg.budget} exceeded (reachable set too big)"
    return
  IO.println s!"forward reachable nodes: {nodes.size}"
  -- backward death propagation: dead if ∃ pending p with all succs dead (or none)
  let n := succs.size
  let mut dead : Array Bool := Array.replicate n false
  -- (nodes beyond succs.size can't exist; every node got expanded)
  let mut changed := true
  let mut passes := 0
  while changed do
    changed := false
    passes := passes + 1
    for nid in [0:n] do
      if !dead[nid]! then
        let mut isDead := false
        for (_, ids) in succs[nid]! do
          if !isDead then
            let mut allDead := true
            for i in ids do
              if !dead[i]! then allDead := false
            if allDead then isDead := true
        if isDead then
          dead := dead.set! nid true
          changed := true
  IO.println s!"death propagation: {passes} passes"
  let aliveCount := dead.foldl (fun acc d => if d then acc else acc + 1) 0
  IO.println s!"alive nodes: {aliveCount} / {n}"
  if dead[0]! then
    IO.println "INIT IS DEAD — no carrier under this menu/caps"
    -- print one adversarial killing line: from init, adversary picks a piece whose
    -- successors are all dead; follow the "best" dead successor
    let mut cur := 0
    let mut steps := 0
    while steps < 40 do
      let nd := nodes[cur]!
      let hstr := String.intercalate "," ((List.range 10).map (fun j => toString (nd.h[j]!)))
      let mut killer : Option (Nat × Option Nat) := none
      for (p, ids) in succs[cur]! do
        if killer.isNone then
          let mut allDead := true
          for i in ids do
            if !dead[i]! then allDead := false
          if allDead then
            killer := some (p, ids.foldl (fun acc i => acc <|> some i) none)
      match killer with
      | some (p, some nxt) =>
        IO.println s!"  h={hstr} bag={nd.bag} adversary draws {p} -> forced dead"
        cur := nxt
        steps := steps + 1
      | some (p, none) =>
        IO.println s!"  h={hstr} bag={nd.bag} adversary draws {p} -> NO PLACEMENT (stuck)"
        steps := 40
      | none =>
        IO.println s!"  h={hstr} bag={nd.bag} (dead but no immediately-killing piece; deeper)"
        steps := 40
    return
  IO.println "INIT IS ALIVE ✓"
  -- extract deterministic policy: first alive successor per (node, piece)
  -- and compute the σ-reachable sub-automaton
  let mut reach : Array Bool := Array.replicate n false
  let mut rq : Array Nat := #[0]
  reach := reach.set! 0 true
  let mut ri := 0
  let mut edges := 0
  while ri < rq.size do
    let nid := rq[ri]!
    ri := ri + 1
    for (_, ids) in succs[nid]! do
      -- first alive successor
      let mut chosen : Option Nat := none
      for i in ids do
        if chosen.isNone && !dead[i]! then chosen := some i
      match chosen with
      | some i =>
        edges := edges + 1
        if !reach[i]! then
          reach := reach.set! i true
          rq := rq.push i
      | none => pure () -- unreachable for alive nid
  let reachCount := reach.foldl (fun acc r => if r then acc + 1 else acc) 0
  IO.println s!"σ-reachable automaton: {reachCount} states, {edges} edges"
  -- stats: max height over σ-reachable, distinct band profiles
  let mut mxh := 0
  let mut profs : HashMap UInt64 Nat := {}
  for nid in [0:n] do
    if reach[nid]! then
      let nd := nodes[nid]!
      for j in [1:10] do
        mxh := max mxh (nd.h[j]!)
      let pk := encode nd.h 0
      profs := profs.insert pk ((profs.get? pk).getD 0 + 1)
  IO.println s!"max band height: {mxh}; distinct profiles: {profs.size}"
  -- dump automaton
  let mut lines : Array String := #[]
  for nid in [0:n] do
    if reach[nid]! then
      let nd := nodes[nid]!
      let hstr := String.intercalate "," ((List.range 10).map (fun j => toString (nd.h[j]!)))
      let mut pstr := ""
      for (p, ids) in succs[nid]! do
        let mut chosen : Option Nat := none
        for i in ids do
          if chosen.isNone && !dead[i]! then chosen := some i
        match chosen with
        | some i => pstr := pstr ++ s!" {p}->{i}"
        | none => pure ()
      lines := lines.push s!"{nid} bag={nd.bag} h={hstr}{pstr}"
  IO.FS.writeFile "Search/automaton.out" (String.intercalate "\n" lines.toList)
  IO.println "automaton dumped to Search/automaton.out"

/-! ## Pair sub-game: two pieces on a 3-column zone, exhaustive AND-OR gfp.
States: min-normalized rel profile (3 cols, spread ≤ scap) × pending ⊆ {a,b}
(pending = pieces of the pair not yet drawn this bag; empty resets to both).
No drain, no caps other than spread (absolute height is handled at composition
time — a zone that survives with bounded spread rises at exactly 8/3 rows/bag). -/
partial def pairGame (a b : Nat) (scap : Nat) : IO Unit := do
  let w := 3
  let m := scap + 1
  let nStates := m * m * m * 4
  let decodeH (i : Nat) : Array Nat := #[(i / (m*m)) % m, (i / m) % m, i % m]
  let encodeH (h : Array Nat) : Nat := Id.run do
    let mn := min (h[0]!) (min (h[1]!) (h[2]!))
    let h' := h.map (· - mn)
    if h'[0]! >= m || h'[1]! >= m || h'[2]! >= m then return nStates
    return (h'[0]!) * m * m + (h'[1]!) * m + h'[2]!
  -- pending encoding: 0 = {a,b}, 1 = {a}, 2 = {b}  (3 values; use 4 slots)
  let succsOf (hi pend : Nat) : Array (Nat × Array Nat) := Id.run do
    let h := decodeH hi
    let pcs : Array Nat := if pend == 0 then #[a, b] else if pend == 1 then #[a] else #[b]
    let mut out : Array (Nat × Array Nat) := #[]
    for p in pcs do
      let pend' := if pend == 0 then (if p == a then 2 else 1) else 0
      let mut ids : Array Nat := #[]
      for o in orients do
        if o.piece == p then
          let ow := o.bottoms.size
          if ow <= w then
            for c in [0:w-ow+1] do
              -- flush check on 3-col zone
              let need0 : Int := (h[c]! : Int) - (o.bottoms[0]! : Int)
              let mut ok := decide (need0 >= 0)
              for i in [1:ow] do
                if (h[c+i]! : Int) - (o.bottoms[i]! : Int) != need0 then ok := false
              if ok then
                let off := need0.toNat
                let mut h2 := h
                for i in [0:ow] do
                  h2 := h2.set! (c+i) (off + o.tops[i]! + 1)
                let k := encodeH h2
                if k < nStates then
                  ids := ids.push (k * 4 + pend')
      out := out.push (p, ids)
    return out
  -- dead-state iteration over ALL states
  let total := nStates * 4
  let mut dead : Array Bool := Array.replicate total false
  -- mark states with pend == 3 unused (never dead-checked, never referenced)
  let mut changed := true
  while changed do
    changed := false
    for st in [0:total] do
      let pend := st % 4
      if pend != 3 && !dead[st]! then
        let hi := st / 4
        let sc := succsOf hi pend
        let mut isDead := false
        for (_, ids) in sc do
          let mut allDead := true
          for i in ids do
            if !dead[i]! then allDead := false
          if allDead then isDead := true
        if isDead then
          dead := dead.set! st true
          changed := true
  -- report: alive bag-start states (pend = 0)
  let mut aliveStarts : Array (Array Nat) := #[]
  for hi in [0:nStates] do
    if !dead[hi * 4]! then
      aliveStarts := aliveStarts.push (decodeH hi)
  IO.println s!"pair ({a},{b}) scap={scap}: alive bag-start profiles: {aliveStarts.size}"
  for hp in aliveStarts do
    IO.println s!"  {hp[0]!},{hp[1]!},{hp[2]!}"

/-! ## Exact per-piece-closed gfp over ALL min-0 profiles of spread ≤ k.

Computes the maximal set P of 10-column min-0 profiles (heights ≤ k) such that
from every h ∈ P, EVERY piece has a flush placement landing (after subtracting
the new min = continuous line clearing) back in P. No bag, no policy, no
reachability heuristics — the exact greatest fixed point. Nonempty P = a
genuine per-piece carrier for hP_step. -/
partial def gfpMode (k : Nat) : IO Unit := do
  let m := k + 1
  let total := m ^ 10
  let decodeH (i : Nat) : Array Nat := Id.run do
    let mut h := Array.replicate 10 0
    let mut x := i
    for j in [0:10] do
      h := h.set! (9 - j) (x % m)
      x := x / m
    return h
  let encodeH (h : Array Nat) : Option Nat := Id.run do
    let mut mn := h[0]!
    for j in [1:10] do
      mn := min mn (h[j]!)
    let mut acc := 0
    for j in [0:10] do
      let v := h[j]! - mn
      if v >= m then return none
      acc := acc * m + v
    return some acc
  let hasMinZero (i : Nat) : Bool := Id.run do
    let mut x := i
    for _ in [0:10] do
      if x % m == 0 then return true
      x := x / m
    return false
  -- alive bitset
  let mut alive : Array Bool := Array.replicate total false
  let mut aliveCount := 0
  for i in [0:total] do
    if hasMinZero i then
      alive := alive.set! i true
      aliveCount := aliveCount + 1
  IO.println s!"gfp mode k={k}: {aliveCount} min-0 profiles of {total} encodings"
  -- iterate removal
  let mut pass := 0
  let mut changed := true
  while changed do
    changed := false
    pass := pass + 1
    let mut removed := 0
    for i in [0:total] do
      if alive[i]! then
        let h := decodeH i
        let mut ok := true
        for p in [0:7] do
          if ok then
            let mut found := false
            for o in orients do
              if !found && o.piece == p then
                let w := o.bottoms.size
                for c in [0:10-w+1] do
                  if !found then
                    let need0 : Int := (h[c]! : Int) - (o.bottoms[0]! : Int)
                    let mut fl := decide (need0 >= 0)
                    for t in [1:w] do
                      if (h[c+t]! : Int) - (o.bottoms[t]! : Int) != need0 then fl := false
                    if fl then
                      let off := need0.toNat
                      let mut h2 := h
                      for t in [0:w] do
                        h2 := h2.set! (c+t) (off + o.tops[t]! + 1)
                      match encodeH h2 with
                      | some i2 => if alive[i2]! then found := true
                      | none => pure ()
            if !found then ok := false
        if !ok then
          alive := alive.set! i false
          removed := removed + 1
          changed := true
    aliveCount := aliveCount - removed
    IO.println s!"  pass {pass}: removed {removed}, alive {aliveCount}"
  IO.println s!"FIXPOINT: |P| = {aliveCount}"
  if aliveCount > 0 && aliveCount <= 200 then
    IO.println "members:"
    for i in [0:total] do
      if alive[i]! then
        let h := decodeH i
        IO.println s!"  {String.intercalate "," ((List.range 10).map (fun j => toString (h[j]!)))}"
  else if aliveCount > 0 then
    -- print 20 samples
    let mut printed := 0
    for i in [0:total] do
      if printed < 20 && alive[i]! then
        let h := decodeH i
        IO.println s!"  {String.intercalate "," ((List.range 10).map (fun j => toString (h[j]!)))}"
        printed := printed + 1

/-! ## Random-bag rollout: can flush-only play survive at all?

Plays random 7-bag sequences with a greedy flush-only policy (minimize
(resulting spread, resulting max height), tie leftmost; unbounded spread,
heights capped at 20 = real board). Reports survival lengths and the
spread/height the surviving play actually uses. A cheap LCG supplies bags. -/
partial def rollMode (games bagsPerGame seed0 : Nat) : IO Unit := do
  let mut seed : UInt64 := UInt64.ofNat seed0
  let next : UInt64 → UInt64 := fun x => x * 6364136223846793005 + 1442695040888963407
  let mut survived := 0
  let mut totalPieces := 0
  let mut maxSpreadEver := 0
  let mut minLen := 1000000
  let mut deathsByPiece : Array Nat := Array.replicate 7 0
  for _ in [0:games] do
    let mut h : Array Nat := Array.replicate 10 0
    let mut piecesPlaced := 0
    let mut dead := false
    for _ in [0:bagsPerGame] do
      if !dead then
        -- shuffle 0..6 via LCG (Fisher-Yates)
        let mut bag : Array Nat := #[0,1,2,3,4,5,6]
        for t in [0:6] do
          seed := next seed
          let r := t + (seed >>> 33).toNat % (7 - t)
          let tmp := bag[t]!
          bag := bag.set! t (bag[r]!)
          bag := bag.set! r tmp
        for pi in [0:7] do
          if !dead then
            let p := bag[pi]!
            -- greedy flush placement: minimize (spread, max, col)
            let mut best : Option (Nat × Nat × Nat × Array Nat) := none
            for o in orients do
              if o.piece == p then
                let w := o.bottoms.size
                for c in [0:10-w+1] do
                  let need0 : Int := (h[c]! : Int) - (o.bottoms[0]! : Int)
                  let mut fl := decide (need0 >= 0)
                  for t in [1:w] do
                    if (h[c+t]! : Int) - (o.bottoms[t]! : Int) != need0 then fl := false
                  if fl then
                    let off := need0.toNat
                    let mut h2 := h
                    for t in [0:w] do
                      h2 := h2.set! (c+t) (off + o.tops[t]! + 1)
                    -- clear: subtract min
                    let mut mn := h2[0]!
                    for j in [1:10] do
                      mn := min mn (h2[j]!)
                    if mn > 0 then
                      h2 := h2.map (· - mn)
                    let mut mx := 0
                    for j in [0:10] do
                      mx := max mx (h2[j]!)
                    if mx <= 20 then
                      -- resource-aware score: heavily penalize missing window types
                      -- (flat-4 for I, flat-2 for O, left-step for S, right-step for Z),
                      -- then spread, then leftmost.
                      let mut flat4 := 0
                      let mut flat2 := 0
                      let mut lstep := 0
                      let mut rstep := 0
                      for j in [0:9] do
                        if h2[j]! == h2[j+1]! then
                          flat2 := flat2 + 1
                          if j + 3 <= 9 && h2[j+1]! == h2[j+2]! && h2[j+2]! == h2[j+3]! then
                            flat4 := flat4 + 1
                        if h2[j]! == h2[j+1]! + 1 then lstep := lstep + 1
                        if h2[j]! + 1 == h2[j+1]! then rstep := rstep + 1
                      let missing := (if flat4 == 0 then 1 else 0) + (if flat2 == 0 then 1 else 0)
                        + (if lstep == 0 then 1 else 0) + (if rstep == 0 then 1 else 0)
                      let key := (missing * 100 + mx, mx, c)
                      match best with
                      | none => best := some (key.1, key.2.1, key.2.2, h2)
                      | some (k1, k2, k3, _) =>
                        if key.1 < k1 || (key.1 == k1 && (key.2.1 < k2 || (key.2.1 == k2 && key.2.2 < k3))) then
                          best := some (key.1, key.2.1, key.2.2, h2)
            match best with
            | some (mx, _, _, h2) =>
              h := h2
              piecesPlaced := piecesPlaced + 1
              if mx > maxSpreadEver then maxSpreadEver := mx
            | none =>
              dead := true
              deathsByPiece := deathsByPiece.set! p (deathsByPiece[p]! + 1)
    if !dead then survived := survived + 1
    totalPieces := totalPieces + piecesPlaced
    if piecesPlaced < minLen then minLen := piecesPlaced
  IO.println s!"rollouts: {games} games × {bagsPerGame} bags: survived full = {survived}"
  IO.println s!"min pieces before death = {minLen}, avg = {totalPieces / games}, max spread used = {maxSpreadEver}"
  IO.println s!"deaths by piece (O I S Z T L J): {deathsByPiece}"

/-! ## Bag-aware exact gfp over the DEEP-WELL family.

Family: h(0) = 0 (the well, kept lowest), band h(1..9) with spread ≤ s and
band-min ≤ dcap. All flush placements allowed; successors outside the family
are dead. Vertical I into the well when band-min ≥ 4 raises the global min to
4 ⇒ 4 rows clear ⇒ the classic drain, expressed in min-normalized semantics. -/
partial def gfpWellMode (s dcap : Nat) : IO Unit := do
  let mrel := s + 1
  let nrel := mrel ^ 9
  let nprof := (dcap + 1) * nrel  -- idx = bandmin * nrel + relIdx
  let total := nprof * 127
  let decodeH (i : Nat) : Array Nat := Id.run do
    let bandmin := i / nrel
    let mut h := Array.replicate 10 0
    let mut x := i % nrel
    for j in [0:9] do
      h := h.set! (9 - j) (bandmin + x % mrel)
      x := x / mrel
    return h
  -- valid state indices additionally require the rel profile to have min 0
  let relMinZero (i : Nat) : Bool := Id.run do
    let mut x := i % nrel
    for _ in [0:9] do
      if x % mrel == 0 then return true
      x := x / mrel
    return false
  let encodeH (h : Array Nat) : Option Nat := Id.run do
    -- renormalize by global min (continuous clearing), then family-check
    let mut mn := h[0]!
    for j in [1:10] do
      mn := min mn (h[j]!)
    if h[0]! - mn != 0 then return none
    let mut bmn := h[1]! - mn
    let mut bmx := h[1]! - mn
    for j in [2:10] do
      bmn := min bmn (h[j]! - mn)
      bmx := max bmx (h[j]! - mn)
    if bmx - bmn > s || bmn > dcap then return none
    let mut acc := 0
    for j in [1:10] do
      acc := acc * mrel + (h[j]! - mn - bmn)
    return some (bmn * nrel + acc)
  let mut alive : ByteArray := ByteArray.mk (Array.replicate total 0)
  let mut aliveCount := 0
  for i in [0:nprof] do
    if relMinZero i then
      for b in [0:127] do
        alive := alive.set! (i * 127 + b) 1
      aliveCount := aliveCount + 127
  IO.println s!"well-gfp s={s} dcap={dcap}: {aliveCount} states ({nprof} band encodings × 127 bags)"
  let mut lastSurvivors : Array (Nat × Nat × Nat) := #[]
  let mut pass := 0
  let mut changed := true
  while changed do
    changed := false
    pass := pass + 1
    let mut removed := 0
    for i in [0:nprof] do
      let mut anyAlive := false
      for b in [0:127] do
        if alive[i * 127 + b]! == 1 then anyAlive := true
      if anyAlive then
        let h := decodeH i
        let mut succIdx : Array (Array Nat) := Array.replicate 7 #[]
        for o in orients do
          let w := o.bottoms.size
          for c in [0:10-w+1] do
            let need0 : Int := (h[c]! : Int) - (o.bottoms[0]! : Int)
            let mut fl := decide (need0 >= 0)
            for t in [1:w] do
              if (h[c+t]! : Int) - (o.bottoms[t]! : Int) != need0 then fl := false
            if fl then
              let off := need0.toNat
              let mut h2 := h
              for t in [0:w] do
                h2 := h2.set! (c+t) (off + o.tops[t]! + 1)
              match encodeH h2 with
              | some i2 => succIdx := succIdx.set! o.piece (succIdx[o.piece]!.push i2)
              | none => pure ()
        for b in [0:127] do
          let st := i * 127 + b
          if alive[st]! == 1 then
            let bag := b + 1
            let mut ok := true
            let mut killer := 9
            for p in [0:7] do
              if ok && bag &&& (1 <<< p) != 0 then
                let bag' := (bag ^^^ (1 <<< p))
                let b' := (if bag' == 0 then 127 else bag') - 1
                let mut found := false
                for i2 in succIdx[p]! do
                  if !found && alive[i2 * 127 + b']! == 1 then found := true
                if !found then
                  ok := false
                  killer := p
            if !ok then
              alive := alive.set! st 0
              removed := removed + 1
              changed := true
              if aliveCount - removed < 600 then
                lastSurvivors := lastSurvivors.push (i, bag, killer)
    aliveCount := aliveCount - removed
    IO.println s!"  pass {pass}: removed {removed}, alive {aliveCount}"
  IO.println s!"FIXPOINT: |alive| = {aliveCount}"
  if aliveCount == 0 && lastSurvivors.size > 0 then
    IO.println s!"last deaths (band h1..h9 | bag | killer 0=O 1=I 2=S 3=Z 4=T 5=L 6=J):"
    let n := lastSurvivors.size
    for t in [n - min 30 n:n] do
      let (i, bag, killer) := lastSurvivors[t]!
      let h := decodeH i
      IO.println s!"  {String.intercalate "," ((List.range 9).map (fun j => toString (h[j+1]!)))} bag={bag} killer={killer}"
  else if aliveCount > 0 then
    let mut printed := 0
    let mut fullCount := 0
    for i in [0:nprof] do
      if alive[i * 127 + 126]! == 1 then
        fullCount := fullCount + 1
        if printed < 30 then
          let h := decodeH i
          IO.println s!"  bag-full alive: 0|{String.intercalate "," ((List.range 9).map (fun j => toString (h[j+1]!)))}"
          printed := printed + 1
    IO.println s!"bag-full alive band profiles: {fullCount}"

/-! ## Bag-aware exact gfp: states (min-0 profile of spread ≤ k, pending bag).

Adversary draws any pending piece (bag refills when emptied); we need a flush
placement landing in an alive state. Exact AND-OR gfp over the full product
space. Alive bag-full states = candidate bag-boundary carrier; the whole alive
set IS the hP_step-closed family for `tetrisSolvableValid_of_strategy_bagInvariant`. -/
partial def gfpBagMode (k : Nat) : IO Unit := do
  let m := k + 1
  let nprof := m ^ 10
  let total := nprof * 127
  let decodeH (i : Nat) : Array Nat := Id.run do
    let mut h := Array.replicate 10 0
    let mut x := i
    for j in [0:10] do
      h := h.set! (9 - j) (x % m)
      x := x / m
    return h
  let encodeH (h : Array Nat) : Option Nat := Id.run do
    let mut mn := h[0]!
    for j in [1:10] do
      mn := min mn (h[j]!)
    let mut acc := 0
    for j in [0:10] do
      let v := h[j]! - mn
      if v >= m then return none
      acc := acc * m + v
    return some acc
  let hasMinZero (i : Nat) : Bool := Id.run do
    let mut x := i
    for _ in [0:10] do
      if x % m == 0 then return true
      x := x / m
    return false
  let mut alive : ByteArray := ByteArray.mk (Array.replicate total 0)
  let mut aliveCount := 0
  for i in [0:nprof] do
    if hasMinZero i then
      for b in [0:127] do
        alive := alive.set! (i * 127 + b) 1
      aliveCount := aliveCount + 127
  IO.println s!"bag-gfp k={k}: {aliveCount} states ({nprof} profile encodings × 127 bags)"
  let mut lastSurvivors : Array (Nat × Nat × Nat) := #[]  -- (profile idx, bag, killer piece)
  let mut pass := 0
  let mut changed := true
  while changed do
    changed := false
    pass := pass + 1
    let mut removed := 0
    for i in [0:nprof] do
      -- fast skip: all bags dead for this profile?
      let mut anyAlive := false
      for b in [0:127] do
        if alive[i * 127 + b]! == 1 then anyAlive := true
      if anyAlive then
        let h := decodeH i
        -- precompute per piece the list of successor profile indices (flush placements)
        let mut succIdx : Array (Array Nat) := Array.replicate 7 #[]
        for o in orients do
          let w := o.bottoms.size
          for c in [0:10-w+1] do
            let need0 : Int := (h[c]! : Int) - (o.bottoms[0]! : Int)
            let mut fl := decide (need0 >= 0)
            for t in [1:w] do
              if (h[c+t]! : Int) - (o.bottoms[t]! : Int) != need0 then fl := false
            if fl then
              let off := need0.toNat
              let mut h2 := h
              for t in [0:w] do
                h2 := h2.set! (c+t) (off + o.tops[t]! + 1)
              match encodeH h2 with
              | some i2 => succIdx := succIdx.set! o.piece (succIdx[o.piece]!.push i2)
              | none => pure ()
        for b in [0:127] do
          let st := i * 127 + b
          if alive[st]! == 1 then
            let bag := b + 1
            let mut ok := true
            let mut killer := 9
            for p in [0:7] do
              if ok && bag &&& (1 <<< p) != 0 then
                let bag' := (bag ^^^ (1 <<< p))
                let b' := (if bag' == 0 then 127 else bag') - 1
                let mut found := false
                for i2 in succIdx[p]! do
                  if !found && alive[i2 * 127 + b']! == 1 then found := true
                if !found then
                  ok := false
                  killer := p
            if !ok then
              alive := alive.set! st 0
              removed := removed + 1
              changed := true
              if aliveCount - removed < 600 then
                lastSurvivors := lastSurvivors.push (i, bag, killer)
    aliveCount := aliveCount - removed
    IO.println s!"  pass {pass}: removed {removed}, alive {aliveCount}"
  IO.println s!"FIXPOINT: |alive| = {aliveCount}"
  if aliveCount == 0 && lastSurvivors.size > 0 then
    IO.println s!"last {min 30 lastSurvivors.size} deaths (profile | bag mask | killer piece 0=O 1=I 2=S 3=Z 4=T 5=L 6=J):"
    let n := lastSurvivors.size
    for t in [n - min 30 n:n] do
      let (i, bag, killer) := lastSurvivors[t]!
      let h := decodeH i
      IO.println s!"  {String.intercalate "," ((List.range 10).map (fun j => toString (h[j]!)))} bag={bag} killer={killer}"
  -- bag-full alive states
  let mut fullCount := 0
  let mut printed := 0
  let mut best : Option Nat := none
  let mut bestSpread := 1000
  for i in [0:nprof] do
    if alive[i * 127 + 126]! == 1 then
      fullCount := fullCount + 1
      let h := decodeH i
      let mut mx := 0
      for j in [0:10] do
        mx := max mx (h[j]!)
      if mx < bestSpread then
        bestSpread := mx
        best := some i
      if printed < 40 then
        IO.println s!"  bag-full alive: {String.intercalate "," ((List.range 10).map (fun j => toString (h[j]!)))}"
        printed := printed + 1
  IO.println s!"bag-full alive profiles: {fullCount}"
  -- σ-core extraction: from the flattest alive bag-full state, follow the
  -- first-alive-successor policy; dump the closed deterministic automaton.
  match best with
  | none => pure ()
  | some root => do
    IO.println s!"extracting σ-core from flattest bag-full root (spread {bestSpread})"
    let rootSt := root * 127 + 126
    let mut idOf : HashMap Nat Nat := {}
    let mut states : Array Nat := #[]
    let mut lines : Array String := #[]
    idOf := idOf.insert rootSt 0
    states := states.push rootSt
    let mut qi := 0
    while qi < states.size do
      let st := states[qi]!
      qi := qi + 1
      let i := st / 127
      let bag := st % 127 + 1
      let h := decodeH i
      -- successors per piece: pick first alive by (orient index, col) order
      let mut pstr := ""
      for p in [0:7] do
        if bag &&& (1 <<< p) != 0 then
          let bag' := bag ^^^ (1 <<< p)
          let b' := (if bag' == 0 then 127 else bag') - 1
          let mut chosen : Option (Nat × Nat × Nat) := none  -- (orientIdx, col, succState)
          for oi in [0:orients.size] do
            let o := orients[oi]!
            if chosen.isNone && o.piece == p then
              let w := o.bottoms.size
              for c in [0:10-w+1] do
                if chosen.isNone then
                  let need0 : Int := (h[c]! : Int) - (o.bottoms[0]! : Int)
                  let mut fl := decide (need0 >= 0)
                  for t in [1:w] do
                    if (h[c+t]! : Int) - (o.bottoms[t]! : Int) != need0 then fl := false
                  if fl then
                    let off := need0.toNat
                    let mut h2 := h
                    for t in [0:w] do
                      h2 := h2.set! (c+t) (off + o.tops[t]! + 1)
                    match encodeH h2 with
                    | some i2 =>
                      if alive[i2 * 127 + b']! == 1 then
                        chosen := some (oi, c, i2 * 127 + b')
                    | none => pure ()
          match chosen with
          | some (oi, c, st') =>
            let id' := match idOf.get? st' with
              | some x => x
              | none => states.size
            if (idOf.get? st').isNone then
              idOf := idOf.insert st' states.size
              states := states.push st'
            let o := orients[oi]!
            pstr := pstr ++ s!" p{p}:rot{o.rot}@{c}->{id'}"
          | none => pstr := pstr ++ s!" p{p}:DEADBUG"
      let hstr := String.intercalate "," ((List.range 10).map (fun j => toString (h[j]!)))
      lines := lines.push s!"{qi-1} h={hstr} bag={bag}{pstr}"
    IO.println s!"σ-core: {states.size} states"
    IO.FS.writeFile "Search/carrier_core.out" (String.intercalate "\n" lines.toList)
    IO.println "σ-core dumped to Search/carrier_core.out"

end CarrierSearch

def main (args : List String) : IO Unit := do
  -- rollout: `carriersearch roll games bags seed`
  if args.head? == some "roll" then
    let g := (args[1]?.bind (·.toNat?)).getD 100
    let bg := (args[2]?.bind (·.toNat?)).getD 1000
    let sd := (args[3]?.bind (·.toNat?)).getD 12345
    CarrierSearch.rollMode g bg sd
    return
  -- deep-well gfp: `carriersearch gfpwell s dcap`
  if args.head? == some "gfpwell" then
    let sc := (args[1]?.bind (·.toNat?)).getD 3
    let dc := (args[2]?.bind (·.toNat?)).getD 5
    CarrierSearch.gfpWellMode sc dc
    return
  -- bag-aware gfp: `carriersearch gfpbag k`
  if args.head? == some "gfpbag" then
    let k := (args[1]?.bind (·.toNat?)).getD 3
    CarrierSearch.gfpBagMode k
    return
  -- gfp mode: `carriersearch gfp k`
  if args.head? == some "gfp" then
    let k := (args[1]?.bind (·.toNat?)).getD 3
    CarrierSearch.gfpMode k
    return
  -- flat mode: `carriersearch flat scap topk budget rank [seed10]`
  if args.head? == some "flat" then
    let get' (i : Nat) (d : Nat) : Nat :=
      match args[i]? with
      | some s => s.toNat? |>.getD d
      | none => d
    let seedStr := (args[5]?).getD "0000000000"
    let seed : Array Nat := (seedStr.toList.map (fun ch => ch.toNat - '0'.toNat)).toArray
    let cfg : CarrierSearch.Cfg := {
      hcap := 100, spreadcap := get' 1 4, topk := get' 2 2,
      budget := get' 3 2000000, rank := get' 4 1, wellless := true,
      seed := if seed.size == 10 then seed else Array.replicate 10 0 }
    IO.println s!"flat mode: scap={cfg.spreadcap} topk={cfg.topk} budget={cfg.budget} rank={cfg.rank} seed={seedStr}"
    CarrierSearch.solve cfg
    return
  -- pairgame mode: `carriersearch pair A B [scap]`
  if args.head? == some "pair" then
    let a := (args[1]?.bind (·.toNat?)).getD 2
    let b := (args[2]?.bind (·.toNat?)).getD 3
    let scap := (args[3]?.bind (·.toNat?)).getD 6
    CarrierSearch.pairGame a b scap
    return
  let get (i : Nat) (d : Nat) : Nat :=
    match args[i]? with
    | some s => s.toNat? |>.getD d
    | none => d
  let cfg : CarrierSearch.Cfg := {
    hcap := get 0 10, spreadcap := get 1 4, topk := get 2 2,
    budget := get 3 2000000, rank := get 4 1, pairing := get 5 99 }
  IO.println s!"cfg: hcap={cfg.hcap} spread={cfg.spreadcap} topk={cfg.topk} budget={cfg.budget} rank={cfg.rank} pairing={cfg.pairing}"
  CarrierSearch.solve cfg
