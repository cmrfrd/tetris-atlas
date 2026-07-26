# I-Pool Verdicts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The drain-faithful zone verdicts: prove drain-invisibility (`normZ_shift`), add the scheduled game (`survP`/`flushDeadP`), and decide the I-including pools (olji4/olji5/szti4/szti6/all7/mixed10).

**Architecture:** Additive changes to `Proofs/Experiments/FlushZoneGame.lean` only — the existing `surv`/`flushDead` and the ten committed verdicts stay byte-identical. The scheduled family adds a phase index (refill draws `sched.getD (phase % len)`); verdicts follow the observe-then-state protocol. Spec: `docs/superpowers/specs/2026-07-13-ipool-verdicts-design.md`.

**Tech Stack:** Lean 4 + mathlib (pinned), `Std.HashMap`.

## Global Constraints

- `ProofsExperiments` target (native_decide allowed, no `sorry`); green target must remain unaffected (`lake build` still green, hygiene gate clean).
- Foreground builds only; one commit per task staging only `proofs/`; messages `proofs(flush-zone): …`.
- Verdicts committed with ACTUAL observed results; `#eval` observation from the scratchpad only; > 10 min → shrink caps/horizon and note final parameters; no instance above 6 columns.
- Existing identifiers (Task 2 of the 2026-07-12 slot-algebra plan): `ZS`, `FShape`, `shapesOf`, `pidx`, `remKey`, `minZ`, `normZ`, `spreadZ`, `movesZ`, `Memo`, `surv`/`survAnd`/`survOr` (mutual, termination tags `(fuel, 2/1/0, …)`), `flushDead`.

---

### Task 1: engine — invisibility lemma + scheduled game

**Files:**
- Modify: `proofs/Proofs/Experiments/FlushZoneGame.lean` (append before the `/-! ## Verdicts, batch 1 -/` section for the lemmas; append after batch 2 for the scheduled game — or all at the end; placement is free since everything is additive)

**Interfaces:**
- Consumes: `minZ`, `normZ`, `ZS`, `surv`-family shapes.
- Produces: `FlushZone.normZ_shift`, `FlushZone.MemoP`, `FlushZone.survP/survAndP/survOrP`, `FlushZone.flushDeadP (w spread : ℕ) (sched : List (List Piece)) (bags : ℕ) : Bool`.

- [ ] **Step 1: Write the invisibility lemma with its helpers**

```lean
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
      rw [show Nat.min (acc - k) (a - k) = Nat.min acc a - k by omega]
      exact ih (Nat.min acc a)
        (fun x hx => hall x (List.mem_cons_of_mem _ hx)) (by omega)

theorem minZ_map_sub (h : ZS) (k : ℕ) (hall : ∀ x ∈ h, k ≤ x) :
    minZ (h.map (· - k)) = minZ h - k := by
  cases h with
  | nil => rfl
  | cons a t =>
      have ha : k ≤ a := hall a (List.mem_cons_self ..)
      simp only [minZ, List.map_cons, List.headD_cons, List.foldl_cons]
      rw [show Nat.min (a - k) (a - k) = Nat.min a a - k by omega]
      exact foldl_min_map_sub k t (Nat.min a a)
        (fun x hx => hall x (List.mem_cons_of_mem _ hx)) (by omega)

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
              (fun x hx => hall' x (List.mem_cons_of_mem _ hx)) (by omega)
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
```

Fallbacks: if `omega` balks at `Nat.min` (older omega), rewrite via
`Nat.min_def` + `split_ifs` first. If `List.map_congr_left`'s name differs,
try `List.map_congr` or do a final `List.ext_getElem?`-free induction. If
`List.mem_cons_self ..` arity fights, spell `(List.mem_cons_self a t)`.

- [ ] **Step 2: Write the scheduled game (parallel family; existing code untouched)**

```lean
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
```

- [ ] **Step 3: Build** — `lake build ProofsExperiments` then `lake build`,
both expect success.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Experiments/FlushZoneGame.lean
git commit -m "proofs(flush-zone): drain invisibility proven (normZ_shift) — the dead verdicts are drain-robust; the scheduled game added"
```

---

### Task 2: the I-pool verdicts

**Files:**
- Modify: `proofs/Proofs/Experiments/FlushZoneGame.lean` (append verdicts)

**Interfaces:**
- Consumes: `flushDead`, `flushDeadP` (Task 1).

- [ ] **Step 1: Observe from the scratchpad** (one heavier instance at a
time; `time lake env lean <file>`; shrink caps/horizon on > 10 min):

```lean
import Proofs.Experiments.FlushZoneGame
open Tetris FlushZone
#eval flushDead 4 6 [Piece.O, Piece.L, Piece.J, Piece.I] 6            -- olji4
#eval flushDead 5 6 [Piece.O, Piece.L, Piece.J, Piece.I] 6            -- olji5
#eval flushDead 4 6 [Piece.S, Piece.Z, Piece.T, Piece.I] 6            -- szti4
#eval flushDead 6 6 [Piece.S, Piece.Z, Piece.T, Piece.I] 4            -- szti6
```

then separately (bigger):

```lean
#eval flushDead 6 8 [Piece.O, Piece.I, Piece.S, Piece.Z, Piece.T, Piece.L, Piece.J] 3  -- all7
#eval flushDeadP 6 10
  [[Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T, Piece.I],
   [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T],
   [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T],
   [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T, Piece.I],
   [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T],
   [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T],
   [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T, Piece.I],
   [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T],
   [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T],
   [Piece.O, Piece.L, Piece.J, Piece.S, Piece.Z, Piece.T]] 10          -- mixed10
```

If alive verdicts appear, additionally probe one deeper horizon (×2 bags)
before stating, to catch olj5-style horizon artifacts.

- [ ] **Step 2: State the verdict theorems with the actual observed values**
(same format as batches 1–2: named `olji4_verdict` … `mixed10_verdict`,
docstrings carrying the caps/horizon and dead/alive-evidence wording; any
alive result gets its deeper-horizon companion theorem too).

- [ ] **Step 3: Build** — `lake build ProofsExperiments`, expect success;
note per-theorem native_decide wall-time.

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/Proofs/Experiments/FlushZoneGame.lean
git commit -m "proofs(flush-zone): the I-pool verdicts — olji/szti/all7/mixed10 decided in-kernel"
```

---

### Task 3: docs + gates

**Files:**
- Modify: `proofs/PROGRESS.md`, `proofs/LIBRARY.md`

- [ ] **Step 1: Gates** — `scripts/check-green-clean.sh` (clean);
`lake build` + `lake build ProofsExperiments` green.

- [ ] **Step 2: PROGRESS.md** — new tick under `## Last tick`: the
drain-invisibility finding (`normZ_shift`; batch-1/2 verdicts upgraded to
drain-robust), the extended verdict table (instance | params | verdict |
wall), and the schedule implication paragraph derived from the actual
results.

- [ ] **Step 3: LIBRARY.md** — extend the FlushZoneGame keep-active line:
"+ scheduled game (`flushDeadP`), drain-invisibility (`normZ_shift`),
I-pool verdicts".

- [ ] **Step 4: Commit**

```bash
cd /Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas
git add proofs/PROGRESS.md proofs/LIBRARY.md
git commit -m "proofs(flush-zone): drain-robustness note + the I-pool verdict table recorded"
```

---

## Self-review notes (already applied)

- Spec A→Task 1, B→Task 2, C→Task 3. No placeholders; the observe-then-state
  protocol carries the unknown results by design.
- Type consistency: `flushDeadP (w spread : ℕ) (sched : List (List Piece))
  (bags : ℕ) : Bool` matches between Tasks 1–2; `MemoP` key shape
  `(ℕ × ZS × ℕ × List ℕ)` used consistently; termination tags mirror the
  proven `surv` family exactly.
- mixed10's I-bags are 0/3/6 per the spec.
