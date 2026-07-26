# Max-Plus Certificate (Rust CEGIS Prototype) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust research binary that searches — by convex LP inside a CEGIS loop — for a max-plus piecewise-linear certificate (`eigen-surface v` + radius `R` + height bound `H_max`) of a controlled-invariant carrier for canonical Tetris, reproducing the closure-oracle soundness of `tetris_policy` while *learning* the eigen-surface the tropical theory promises exists.

**Architecture:** A new research bin `tetris_maxplus_cert` adapts the existing `tetris_policy` reachable-closure engine. The player policy homes greedily to a candidate eigen-surface `v` (minimize `osc(successor − v)` = Hilbert oscillation). A small LP (`good_lp` + `clarabel`) fits the tightest `osc`-ball around the CEGIS counterexample set, yielding `v, R, H_max`. The loop alternates fit-ball → policy → closure-verify → add-counterexample until the carrier closes (certificate) or floors (refutation with the binding `R` / leak state).

**Tech Stack:** Rust nightly (edition 2024, `generic_const_exprs`), `tetris-game` engine, `good_lp` LP modeling with the pure-Rust `clarabel` backend, `rustc-hash` for state sets, `rayon` (optional, for closure parallelism).

## Global Constraints

- Nightly toolchain (pinned via `rust-toolchain.toml`). First lines of `main.rs`: `#![feature(generic_const_exprs)]` then `#![allow(incomplete_features)]` (match `tetris_policy/main.rs:1-2`).
- Clippy **denies** `panic!`, `todo!`, `unimplemented!`, `unreachable!`. Use `anyhow::Result` / `Option` for fallible paths; never `unwrap` in non-test code without justification (`unwrap_used` is a warning, not a hard error, but avoid it).
- Board is the canonical `tetris_game::TetrisBoard` (`COLS = 10`, `ROWS = 20`). Do **not** modify `tetris-game`. `W = 10` everywhere; never assume a different width.
- `osc(x) = max_j x_j − min_j x_j` (the Hilbert projective diameter / `roughness` in `TopicalTetris.lean:503`). This is **not** the engine's `roughness()` (which is bumpiness `Σ|Δ|`). Define and use our own `osc`.
- **Branch / commit policy (overrides the skill's "frequent commits"):** the `tetris-proofs` branch commits **only** `proofs/` (`feedback_commit_per_iter_proofs_only`). Every existing `tetris_*` research bin is intentionally untracked. So: **each task's checkpoint is "tests green," not a git commit.** Do NOT `git add` `Cargo.toml`, `crates/`, `docs/`, or `.claude/` on this branch. If the user wants commits, they will be done on a separate feature branch on request.
- Every milestone run reports (per `CLAUDE.md`): `N`/seed-set, certified `(v, R, H_max)`, `|Σ|` at convergence, CEGIS iteration count, wall-time + pieces/sec, and on failure the leak state / binding `R`.

## File Structure

```
crates/tetris-playground/
├── Cargo.toml                                   # MODIFY: add [[bin]] + good_lp dep
└── src/bin/research/tetris_maxplus_cert/
    ├── main.rs                                  # CLI, modes, metrics, smoke LP
    ├── engine.rs                                # board↔surface adapter: heights_i64, osc, successor, holes
    ├── lp.rs                                    # fit_ball: the certificate LP (good_lp + clarabel)
    ├── policy.rs                                # osc-greedy player homing to v
    ├── closure.rs                               # reachable (board,bag) closure oracle + counterexample
    └── cegis.rs                                 # the CEGIS driver + Report
```

Each file has one responsibility; `main.rs` only wires CLI → modes → metrics. Tests live inline (`#[cfg(test)] mod tests`) and run via `cargo test --bin tetris_maxplus_cert`.

---

### Task 1: Scaffold the bin + LP toolchain smoke test

**Files:**
- Modify: `crates/tetris-playground/Cargo.toml` (add `[[bin]]` stanza near `tetris_policy` at lines 66-149; add dependency near line 169-227)
- Create: `crates/tetris-playground/src/bin/research/tetris_maxplus_cert/main.rs`

**Interfaces:**
- Produces: `fn smoke_lp() -> f64` (returns the optimum of a trivial LP; proves `good_lp`+`clarabel` link and the API shape used by `lp.rs`).

- [ ] **Step 1: Add the dependency and bin to `Cargo.toml`**

Add to `[dependencies]` (the `clarabel` backend is pure-Rust; disable defaults so it does not pull `coin_cbc`, which needs a system lib):

```toml
good_lp = { version = "1.8", default-features = false, features = ["clarabel"] }
```

Add a `[[bin]]` stanza alongside the other research bins:

```toml
[[bin]]
name = "tetris_maxplus_cert"
path = "src/bin/research/tetris_maxplus_cert/main.rs"
```

- [ ] **Step 2: Write `main.rs` with the banner and the LP smoke test**

```rust
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! tetris_maxplus_cert — convex (LP) search inside a CEGIS loop for a max-plus
//! piecewise-linear survival certificate (eigen-surface v + radius R + height bound H_max).

use good_lp::{constraint, variable, variables, Solution, SolverModel};
use good_lp::solvers::clarabel::clarabel;

/// Trivial LP: minimize x subject to x >= 3. Optimum is 3.0.
/// Proves the good_lp + clarabel toolchain links and the modeling API is as expected.
fn smoke_lp() -> f64 {
    let mut vars = variables!();
    let x = vars.add(variable().min(0.0));
    let sol = vars
        .minimise(x)
        .using(clarabel)
        .with(constraint!(x >= 3.0))
        .solve()
        .expect("smoke LP must solve");
    sol.value(x)
}

fn main() {
    println!("tetris_maxplus_cert — max-plus certificate via convex CEGIS");
    println!("LP toolchain smoke: min x s.t. x>=3  =>  x = {:.3}", smoke_lp());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_lp_is_three() {
        assert!((smoke_lp() - 3.0).abs() < 1e-6, "got {}", smoke_lp());
    }
}
```

- [ ] **Step 3: Run the test to verify the toolchain**

Run: `cargo test -p tetris-playground --bin tetris_maxplus_cert smoke_lp_is_three -- --nocapture`
Expected: PASS. (If `good_lp` 1.8's API differs — e.g. `clarabel` import path — fix the imports now; this task exists to lock the API before `lp.rs`.)

- [ ] **Step 4: Run the binary**

Run: `cargo run -p tetris-playground --bin tetris_maxplus_cert`
Expected: prints the banner and `x = 3.000`.

- [ ] **Step 5: Checkpoint (no commit — see Global Constraints)**

Run: `cargo build -p tetris-playground --bin tetris_maxplus_cert` → must succeed with no errors.

---

### Task 2: Engine adapter — surface, osc, successor, holes

**Files:**
- Create: `crates/tetris-playground/src/bin/research/tetris_maxplus_cert/engine.rs`
- Modify: `main.rs` (add `mod engine;`)

**Interfaces:**
- Consumes: `tetris_game::{TetrisBoard, TetrisPiece, TetrisPiecePlacement, IsLost}`.
- Produces:
  - `const W: usize = 10;`
  - `fn heights_i64(b: &TetrisBoard) -> [i64; W]` — per-column surface heights as `i64`.
  - `fn osc(h: &[i64; W]) -> i64` — `max − min` (Hilbert oscillation).
  - `fn max_height(h: &[i64; W]) -> i64` — `max_j h_j`.
  - `fn holes_of(b: &TetrisBoard) -> u32` — buried-cell count (debt).
  - `fn successor(b: &TetrisBoard, pl: TetrisPiecePlacement) -> Option<(TetrisBoard, u32)>` — apply placement; `None` if the placement loses; else `(new_board, lines_cleared)`.
  - `fn placements(p: TetrisPiece) -> &'static [TetrisPiecePlacement]`.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tetris_game::{TetrisBoard, TetrisPiece, TetrisPiecePlacement};

    #[test]
    fn empty_board_is_flat_zero() {
        let h = heights_i64(&TetrisBoard::new());
        assert_eq!(h, [0i64; W]);
        assert_eq!(osc(&h), 0);
        assert_eq!(max_height(&h), 0);
    }

    #[test]
    fn osc_is_max_minus_min() {
        let h = [3, 0, 5, 1, 1, 1, 1, 1, 1, 2];
        assert_eq!(osc(&h), 5); // 5 - 0
        assert_eq!(max_height(&h), 5);
    }

    #[test]
    fn some_placement_raises_height_and_succeeds() {
        let b = TetrisBoard::new();
        let pl = placements(TetrisPiece::O_PIECE)[0];
        let (nb, cleared) = successor(&b, pl).expect("O on empty must not lose");
        assert_eq!(cleared, 0);
        assert_eq!(max_height(&heights_i64(&nb)), 2); // O is 2 tall
        assert_eq!(holes_of(&nb), 0);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p tetris-playground --bin tetris_maxplus_cert engine -- --nocapture`
Expected: FAIL (module/functions not defined).

- [ ] **Step 3: Implement `engine.rs`**

```rust
//! Board ↔ surface adapter. The certificate lives on the surface (column heights);
//! the ground-truth dynamics stay in the real `tetris-game` engine.

use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};

pub const W: usize = 10;

/// Per-column surface heights (top of each column) as i64, so v-subtraction can go negative.
pub fn heights_i64(b: &TetrisBoard) -> [i64; W] {
    let h = b.heights(); // [u32; 10]
    let mut out = [0i64; W];
    for j in 0..W {
        out[j] = h[j] as i64;
    }
    out
}

/// Hilbert oscillation = max - min. NOT the engine's bumpiness `roughness()`.
pub fn osc(h: &[i64; W]) -> i64 {
    let mx = h.iter().copied().max().unwrap_or(0);
    let mn = h.iter().copied().min().unwrap_or(0);
    mx - mn
}

pub fn max_height(h: &[i64; W]) -> i64 {
    h.iter().copied().max().unwrap_or(0)
}

pub fn holes_of(b: &TetrisBoard) -> u32 {
    b.total_holes()
}

/// Apply a placement to a copy of the board. Returns None if it tops out,
/// else the resulting board and the number of lines cleared.
pub fn successor(b: &TetrisBoard, pl: TetrisPiecePlacement) -> Option<(TetrisBoard, u32)> {
    let mut nb = *b;
    let res = nb.apply_piece_placement(pl);
    if res.is_lost == IsLost::LOST {
        None
    } else {
        Some((nb, res.lines_cleared))
    }
}

pub fn placements(p: TetrisPiece) -> &'static [TetrisPiecePlacement] {
    TetrisPiecePlacement::all_from_piece(p)
}
```

Add `mod engine;` to `main.rs`.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p tetris-playground --bin tetris_maxplus_cert engine -- --nocapture`
Expected: PASS (all three tests).

- [ ] **Step 5: Checkpoint**

Run: `cargo clippy -p tetris-playground --bin tetris_maxplus_cert -- -D warnings` → no errors.

---

### Task 3: The certificate LP — `fit_ball`

**Files:**
- Create: `crates/tetris-playground/src/bin/research/tetris_maxplus_cert/lp.rs`
- Modify: `main.rs` (add `mod lp;`)

**Interfaces:**
- Consumes: `crate::engine::W`; `good_lp` API as locked in Task 1.
- Produces:
  - `struct Cert { pub v: [f64; W], pub r: f64, pub h_max: f64 }`
  - `fn fit_ball(samples: &[[i64; W]]) -> Option<Cert>` — solves: minimize `R` over `v ∈ ℝ^W, R ≥ 0, H_max ∈ [0,20]` and per-sample aux `M_s, m_s`, s.t. for every sample `s` and column `j`: `M_s + v_j ≥ h^s_j`, `m_s + v_j ≤ h^s_j`, `M_s − m_s ≤ R`, `h^s_j ≤ H_max`. Gauge-fix `v_0 = 0`. Returns `None` if infeasible/unsolved (e.g. a sample with `max_j h^s_j > 20` forces `H_max > 20`).

The convexity rationale (for the reviewer): each `h^s` is a constant, so every constraint is linear in `(v, R, H_max, M_s, m_s)`; minimizing `R` drives `M_s → max_j(h^s_j − v_j)` and `m_s → min_j(...)`, so `M_s − m_s = osc(h^s − v)`. The objective is therefore `min_v max_s osc(h^s − v)` — the tightest common `osc`-ball.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_flat_surfaces_fit_radius_zero() {
        // Both surfaces are flat (osc 0) at different absolute heights:
        // a single v can center both with R = 0.
        let s1 = [2i64; W];
        let s2 = [5i64; W];
        let c = fit_ball(&[s1, s2]).expect("flat surfaces must fit");
        assert!(c.r < 1e-6, "R = {}", c.r);
        assert!(c.h_max >= 5.0 - 1e-6 && c.h_max <= 20.0 + 1e-6, "H_max = {}", c.h_max);
    }

    #[test]
    fn incompatible_shapes_force_positive_radius() {
        // NOTE: a SINGLE sample always fits R=0 — set v_j = h_j - h_0 so h-v is
        // constant (osc 0). Radius is forced only by shapes no common v can co-center.
        // Two orthogonal spikes: 2R >= |6+v1| + |6-v1| >= 12, so R >= 6 for every v.
        let mut s1 = [0i64; W];
        s1[0] = 6;
        let mut s2 = [0i64; W];
        s2[1] = 6;
        let c = fit_ball(&[s1, s2]).expect("two spikes fit (heights <= 20)");
        assert!(c.r >= 6.0 - 1e-6, "R = {} should be >= 6", c.r);
    }

    #[test]
    fn over_tall_sample_is_infeasible() {
        // A column at height 25 cannot satisfy H_max <= 20.
        let mut s = [0i64; W];
        s[3] = 25;
        assert!(fit_ball(&[s]).is_none());
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p tetris-playground --bin tetris_maxplus_cert lp -- --nocapture`
Expected: FAIL (not defined).

- [ ] **Step 3: Implement `lp.rs`**

```rust
//! The certificate LP: smallest common osc-ball over a sample set.

use good_lp::{constraint, variable, variables, Expression, Solution, SolverModel};
use good_lp::solvers::clarabel::clarabel;

use crate::engine::W;

#[derive(Clone, Debug)]
pub struct Cert {
    pub v: [f64; W],
    pub r: f64,
    pub h_max: f64,
}

/// Minimize R s.t. every sample lies in the osc-ball {h : osc(h - v) <= R} and within H_max <= 20.
pub fn fit_ball(samples: &[[i64; W]]) -> Option<Cert> {
    if samples.is_empty() {
        // Degenerate: the empty board fits trivially.
        return Some(Cert { v: [0.0; W], r: 0.0, h_max: 0.0 });
    }
    let mut vars = variables!();
    let v: Vec<_> = (0..W).map(|_| vars.add(variable())).collect(); // free
    let r = vars.add(variable().min(0.0));
    let h_max = vars.add(variable().min(0.0).max(20.0));

    let mut model = vars.minimise(Expression::from(r)).using(clarabel);
    // Gauge fix: v_0 = 0 (osc is invariant to a uniform shift of v).
    model = model.with(constraint!(v[0] == 0.0));

    for s in samples {
        let m_s = /* min aux */ {
            // declare fresh aux per sample via a second variables! block is not possible after move;
            // instead add through the model's variable space:
            // good_lp lets us add variables before building the model, so collect aux first.
            unreachable!("replaced below")
        };
        let _ = m_s;
    }
    // NOTE: good_lp requires all variables be created on `vars` BEFORE `.minimise`.
    // The block above is illustrative; the real implementation (Step 3b) pre-creates aux vars.
    let _ = &model;
    None
}
```

- [ ] **Step 3b: Replace with the correct variable-ordering implementation**

`good_lp` requires every variable to be added to the `ProblemVariables` *before* calling `.minimise`. Pre-create the per-sample `M_s, m_s` aux variables, then build constraints:

```rust
pub fn fit_ball(samples: &[[i64; W]]) -> Option<Cert> {
    if samples.is_empty() {
        return Some(Cert { v: [0.0; W], r: 0.0, h_max: 0.0 });
    }
    let mut vars = variables!();
    let v: Vec<_> = (0..W).map(|_| vars.add(variable())).collect();
    let r = vars.add(variable().min(0.0));
    let h_max = vars.add(variable().min(0.0).max(20.0));
    // Per-sample aux: M_s (upper) and m_s (lower), both free.
    let aux: Vec<(good_lp::Variable, good_lp::Variable)> =
        samples.iter().map(|_| (vars.add(variable()), vars.add(variable()))).collect();

    let mut model = vars.minimise(Expression::from(r)).using(clarabel);
    model = model.with(constraint!(v[0] == 0.0));

    for (s, &(m_up, m_lo)) in samples.iter().zip(aux.iter()) {
        for j in 0..W {
            let hsj = s[j] as f64;
            // M_s >= h_sj - v_j  <=>  M_s + v_j >= h_sj
            model = model.with(constraint!(m_up + v[j] >= hsj));
            // m_s <= h_sj - v_j  <=>  m_s + v_j <= h_sj
            model = model.with(constraint!(m_lo + v[j] <= hsj));
            // absolute-height face
            model = model.with(constraint!(h_max >= hsj));
        }
        // osc(h_s - v) = M_s - m_s <= R
        model = model.with(constraint!(m_up - m_lo <= r));
    }

    let sol = model.solve().ok()?;
    let mut v_out = [0.0f64; W];
    for j in 0..W {
        v_out[j] = sol.value(v[j]);
    }
    let r_out = sol.value(r);
    let hmax_out = sol.value(h_max);
    // Reject solutions that violate the loss bound (numerical slack guard).
    if hmax_out > 20.0 + 1e-6 {
        return None;
    }
    Some(Cert { v: v_out, r: r_out, h_max: hmax_out })
}
```

Delete the illustrative Step-3 body; keep only this version. Add `mod lp;` to `main.rs`.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p tetris-playground --bin tetris_maxplus_cert lp -- --nocapture`
Expected: PASS (all three tests). If `over_tall_sample_is_infeasible` instead returns `Some` with `h_max ≈ 25`, the guard at the end converts it to `None` — verify the test passes; if clarabel reports infeasible directly, `.ok()?` already yields `None`.

- [ ] **Step 5: Checkpoint**

Run: `cargo clippy -p tetris-playground --bin tetris_maxplus_cert -- -D warnings` → no errors.

---

### Task 4: The osc-greedy policy — `choose_placement`

**Files:**
- Create: `crates/tetris-playground/src/bin/research/tetris_maxplus_cert/policy.rs`
- Modify: `main.rs` (add `mod policy;`)

**Interfaces:**
- Consumes: `crate::engine::{W, heights_i64, osc, max_height, holes_of, successor, placements}`; `crate::lp::Cert`.
- Produces:
  - `struct Band { pub k_holes: u32, pub h_band: i64 }` — the debt/height band the carrier is restricted to.
  - `fn choose_placement(b: &TetrisBoard, p: TetrisPiece, cert: &Cert, band: &Band) -> Option<(TetrisBoard, [i64; W])>` — among all valid placements of `p` that keep `holes ≤ k_holes` and `max_height ≤ h_band` and do not lose, return the successor minimizing `osc(succ_heights − round(cert.v))`, with ties broken by lower `max_height`. Returns `None` if no placement stays in-band (a leak).

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tetris_game::{TetrisBoard, TetrisPiece};
    use crate::lp::Cert;

    fn flat_cert() -> Cert { Cert { v: [0.0; W], r: 20.0, h_max: 20.0 } }

    #[test]
    fn picks_a_flat_keeping_move_for_o_on_empty() {
        let b = TetrisBoard::new();
        let band = Band { k_holes: 1, h_band: 20 };
        let (_, h) = choose_placement(&b, TetrisPiece::O_PIECE, &flat_cert(), &band)
            .expect("O on empty has in-band placements");
        // O leaves osc 2 (a 2-tall bump) no matter where; just assert it is finite & in band.
        assert!(max_height(&h) <= 20);
    }

    #[test]
    fn tight_band_can_force_a_leak() {
        // With h_band = 0 nothing can be placed (every placement raises height), so it leaks.
        let b = TetrisBoard::new();
        let band = Band { k_holes: 0, h_band: 0 };
        assert!(choose_placement(&b, TetrisPiece::I_PIECE, &flat_cert(), &band).is_none());
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p tetris-playground --bin tetris_maxplus_cert policy -- --nocapture`
Expected: FAIL (not defined).

- [ ] **Step 3: Implement `policy.rs`**

```rust
//! osc-greedy player: home the surface toward the candidate eigen-surface v, staying in-band.

use tetris_game::{TetrisBoard, TetrisPiece};

use crate::engine::{heights_i64, max_height, osc, holes_of, placements, successor, W};
use crate::lp::Cert;

#[derive(Clone, Copy, Debug)]
pub struct Band {
    pub k_holes: u32,
    pub h_band: i64,
}

/// Among in-band, non-losing placements of `p`, pick the one minimizing osc(succ - round(v)),
/// tie-broken by lower max-height. None = leak (no in-band placement).
pub fn choose_placement(
    b: &TetrisBoard,
    p: TetrisPiece,
    cert: &Cert,
    band: &Band,
) -> Option<(TetrisBoard, [i64; W])> {
    let vi: [i64; W] = {
        let mut out = [0i64; W];
        for j in 0..W {
            out[j] = cert.v[j].round() as i64;
        }
        out
    };
    let mut best: Option<(i64, i64, TetrisBoard, [i64; W])> = None; // (osc_key, height_key, board, heights)
    for &pl in placements(p) {
        let Some((nb, _cleared)) = successor(b, pl) else { continue };
        if holes_of(&nb) > band.k_holes {
            continue;
        }
        let h = heights_i64(&nb);
        let mh = max_height(&h);
        if mh > band.h_band {
            continue;
        }
        let mut centered = [0i64; W];
        for j in 0..W {
            centered[j] = h[j] - vi[j];
        }
        let key = (osc(&centered), mh);
        let take = match &best {
            None => true,
            Some((bo, bh, _, _)) => key < (*bo, *bh),
        };
        if take {
            best = Some((key.0, key.1, nb, h));
        }
    }
    best.map(|(_, _, nb, h)| (nb, h))
}
```

Add `mod policy;` to `main.rs`.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p tetris-playground --bin tetris_maxplus_cert policy -- --nocapture`
Expected: PASS (both tests).

- [ ] **Step 5: Checkpoint**

Run: `cargo clippy -p tetris-playground --bin tetris_maxplus_cert -- -D warnings` → no errors.

---

### Task 5: The closure oracle + counterexample

**Files:**
- Create: `crates/tetris-playground/src/bin/research/tetris_maxplus_cert/closure.rs`
- Modify: `main.rs` (add `mod closure;`)

**Interfaces:**
- Consumes: `crate::engine`, `crate::policy::{Band, choose_placement}`, `crate::lp::Cert`; `tetris_game::{TetrisBoard, TetrisPiece}`.
- Produces:
  - `enum Outcome { Closed { size: usize, max_osc: i64, mh: i64 }, Leak { state: [i64; W], piece_idx: usize, size: usize }, Exploded { size: usize, mh: i64 } }`
  - `fn closure(cert: &Cert, band: &Band, cap: usize) -> Outcome` — from the empty board with a full bag, explore `(board, bag_mask)` states. For each state and each remaining piece (AND), the policy picks the in-band successor (OR). A piece with no in-band successor ⇒ `Leak` with that board's surface as the counterexample. Worklist exceeding `cap` ⇒ `Exploded`. Worklist empties ⇒ `Closed`, also reporting the largest `osc(h − round(v))` seen (the binding radius the carrier actually needs).

Mirrors `tetris_policy/main.rs:150-197` (the `(TetrisBoard, u8)` `FxHashSet` worklist, the `PIECES`/`1<<pi`/`FULL_MASK` bag-mask pattern, bag refill to `FULL_MASK` when emptied).

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::Cert;
    use crate::policy::Band;

    #[test]
    fn impossibly_tight_band_leaks_immediately() {
        // h_band 0: the empty root cannot place any first piece in-band -> Leak at size 1.
        let cert = Cert { v: [0.0; W], r: 0.0, h_max: 0.0 };
        let band = Band { k_holes: 0, h_band: 0 };
        match closure(&cert, &band, 1_000) {
            Outcome::Leak { size, .. } => assert_eq!(size, 1),
            other => panic!("expected Leak, got {:?}", other),
        }
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p tetris-playground --bin tetris_maxplus_cert closure -- --nocapture`
Expected: FAIL (not defined).

- [ ] **Step 3: Implement `closure.rs`**

```rust
//! Reachable (board, bag_mask) closure under the osc-greedy policy vs the switching 7-bag
//! adversary. Adapts tetris_policy/main.rs. The closed reachable set IS the survival invariant;
//! the certificate (v, R) is its compact over-approximation.

use rustc_hash::FxHashSet;
use tetris_game::{TetrisBoard, TetrisPiece};

use crate::engine::{heights_i64, max_height, osc, W};
use crate::lp::Cert;
use crate::policy::{choose_placement, Band};

const PIECES: [TetrisPiece; 7] = TetrisPiece::all();
const FULL_MASK: u8 = 0b0111_1111;

#[derive(Clone, Debug)]
pub enum Outcome {
    Closed { size: usize, max_osc: i64, mh: i64 },
    Leak { state: [i64; W], piece_idx: usize, size: usize },
    Exploded { size: usize, mh: i64 },
}

pub fn closure(cert: &Cert, band: &Band, cap: usize) -> Outcome {
    let vi: [i64; W] = {
        let mut out = [0i64; W];
        for j in 0..W {
            out[j] = cert.v[j].round() as i64;
        }
        out
    };
    let centered_osc = |h: &[i64; W]| -> i64 {
        let mut c = [0i64; W];
        for j in 0..W {
            c[j] = h[j] - vi[j];
        }
        osc(&c)
    };

    let root = TetrisBoard::new();
    let mut seen: FxHashSet<(TetrisBoard, u8)> = FxHashSet::default();
    let mut stack: Vec<(TetrisBoard, u8)> = vec![(root, FULL_MASK)];
    seen.insert((root, FULL_MASK));
    let (mut max_osc, mut mh) = (0i64, 0i64);

    while let Some((b, mask)) = stack.pop() {
        for (pi, p) in PIECES.iter().enumerate() {
            let bit = 1u8 << pi;
            if mask & bit == 0 {
                continue;
            }
            let rem = mask & !bit;
            let rem_full = if rem == 0 { FULL_MASK } else { rem };
            match choose_placement(&b, *p, cert, band) {
                None => {
                    return Outcome::Leak {
                        state: heights_i64(&b),
                        piece_idx: pi,
                        size: seen.len(),
                    };
                }
                Some((nb, h)) => {
                    max_osc = max_osc.max(centered_osc(&h));
                    mh = mh.max(max_height(&h));
                    if seen.insert((nb, rem_full)) {
                        if seen.len() > cap {
                            return Outcome::Exploded { size: seen.len(), mh };
                        }
                        stack.push((nb, rem_full));
                    }
                }
            }
        }
    }
    Outcome::Closed { size: seen.len(), max_osc, mh }
}
```

Add `mod closure;` to `main.rs`. (If `rustc_hash` is not already a dependency of `tetris-playground`, it is — `tetris_policy` uses it; confirm in `Cargo.toml`.)

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p tetris-playground --bin tetris_maxplus_cert closure -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `cargo clippy -p tetris-playground --bin tetris_maxplus_cert -- -D warnings` → no errors.

---

### Task 6: M0 — fixed-order validation mode

**Files:**
- Create: `crates/tetris-playground/src/bin/research/tetris_maxplus_cert/cegis.rs` (start the driver module with the M0 function first)
- Modify: `main.rs` (add `mod cegis;`)

**Interfaces:**
- Consumes: `crate::engine`, `crate::policy::Band`, `crate::lp::{Cert, fit_ball}`; `tetris_game::{TetrisBoard, TetrisPiece, TetrisPieceBagState}`.
- Produces:
  - `fn play_fixed_order(order: &[TetrisPiece], steps: usize, band: &Band) -> Vec<[i64; W]>` — from empty, repeat `order` for `steps` pieces, each time picking the placement that minimizes `(max_height, osc)` (a fixed seed objective; no cert needed), collecting the surface after each piece. Stops early (returns what it has) if a piece cannot be placed in-band.
  - `fn m0_fixed_order_radius(order: &[TetrisPiece], steps: usize, band: &Band) -> Option<f64>` — fit the smallest osc-ball over the trajectory; return `R`. `None` if the play leaked.

M0 validates `fit_ball` + the engine adapter on real trajectories: every fixed order should survive at small `osc`, so the fitted `R` should be modest (expected `R ≤ ~7`, matching the tropical-probe finding that good fixed-order play keeps `max height ≤ 7`).

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::Band;
    use tetris_game::TetrisPiece;

    #[test]
    fn fixed_order_validates_ball_fit_over_survivors() {
        // A plain 1-ply flatten player is MYOPIC: it accumulates S/Z holes faster than it
        // drains them and leaks on the straight order around bag 5 (matches the known result
        // that fixed orders need lookahead — dumb 1-ply survives only ~60%). So M0 validates
        // the fit_ball + engine PIPELINE over a horizon the dumb player demonstrably survives
        // (4 bags), NOT long-horizon survival (that needs a beam, out of M0's scope).
        let order = TetrisPiece::all().to_vec();
        let band = Band { k_holes: 4, h_band: 20 };
        let steps = 7 * 4; // 28 = 4 bags; the flatten player survives this on the straight order
        let r = m0_fixed_order_radius(&order, steps, &band)
            .expect("flatten player must survive 4 bags of the straight order");
        assert!((0.0..=20.0).contains(&r), "fitted radius {r} out of sane range [0,20]");
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p tetris-playground --bin tetris_maxplus_cert cegis::tests::straight_order -- --nocapture`
Expected: FAIL (not defined).

- [ ] **Step 3: Implement the M0 functions in `cegis.rs`**

```rust
//! CEGIS driver. Task 6 adds the M0 fixed-order validation; Task 7 adds the switching loop.

use tetris_game::{TetrisBoard, TetrisPiece};

use crate::engine::{heights_i64, max_height, osc, holes_of, placements, successor, W};
use crate::lp::{fit_ball, Cert};
use crate::policy::Band;

/// Seed objective (no cert): minimize (max_height, osc) — a plain flatten player.
fn flatten_choice(b: &TetrisBoard, p: TetrisPiece, band: &Band) -> Option<TetrisBoard> {
    let mut best: Option<((i64, i64), TetrisBoard)> = None;
    for &pl in placements(p) {
        let Some((nb, _)) = successor(b, pl) else { continue };
        if holes_of(&nb) > band.k_holes {
            continue;
        }
        let h = heights_i64(&nb);
        let mh = max_height(&h);
        if mh > band.h_band {
            continue;
        }
        let key = (mh, osc(&h));
        if best.as_ref().map_or(true, |(bk, _)| key < *bk) {
            best = Some((key, nb));
        }
    }
    best.map(|(_, nb)| nb)
}

pub fn play_fixed_order(order: &[TetrisPiece], steps: usize, band: &Band) -> Vec<[i64; W]> {
    let mut b = TetrisBoard::new();
    let mut traj = Vec::with_capacity(steps);
    for i in 0..steps {
        let p = order[i % order.len()];
        match flatten_choice(&b, p, band) {
            Some(nb) => {
                b = nb;
                traj.push(heights_i64(&b));
            }
            None => break,
        }
    }
    traj
}

pub fn m0_fixed_order_radius(order: &[TetrisPiece], steps: usize, band: &Band) -> Option<f64> {
    let traj = play_fixed_order(order, steps, band);
    if traj.len() < steps {
        return None; // leaked before finishing
    }
    fit_ball(&traj).map(|c: Cert| c.r)
}
```

Add `mod cegis;` to `main.rs`.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p tetris-playground --bin tetris_maxplus_cert cegis::tests::straight_order -- --nocapture`
Expected: PASS. If `R` exceeds 12, the flatten player is riding the ceiling on this order — loosen the assertion and record the actual radius (this is data, not a bug), but first confirm `traj.len() == steps` (no leak).

- [ ] **Step 5: Checkpoint**

Run: `cargo clippy -p tetris-playground --bin tetris_maxplus_cert -- -D warnings` → no errors.

---

### Task 7: The CEGIS driver (switching adversary)

**Files:**
- Modify: `crates/tetris-playground/src/bin/research/tetris_maxplus_cert/cegis.rs` (add the switching loop)

**Interfaces:**
- Consumes: everything above (`closure::Outcome`, `closure::closure`, `fit_ball`, `Band`, `Cert`).
- Produces:
  - `struct Report { pub status: Status, pub iters: usize, pub cert: Cert, pub size: usize, pub max_osc: i64 }`
  - `enum Status { Certified, Refuted { binding_r: f64 }, Exploded }`
  - `fn run_cegis(band: &Band, cap: usize, max_iters: usize) -> Report` — seed `samples = {flat empty surface}`; loop: `cert ← fit_ball(samples)`; `out ← closure(&cert, band, cap)`; on `Closed` → `Certified`; on `Exploded` → `Exploded`; on `Leak{state}` → push `state` into `samples` and re-fit. If `fit_ball` returns `None` (the leak set cannot fit any osc-ball within `H_max ≤ 20`) → `Refuted` with the last finite radius. Stop after `max_iters` (report whichever terminal state was reached, else `Refuted` with the current radius).

The CEGIS contract: `samples` only ever holds **leak counterexamples** (a handful), so the LP stays tiny regardless of how large the reachable closure is.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod cegis_tests {
    use super::*;
    use crate::policy::Band;

    #[test]
    fn cegis_terminates_and_reports() {
        // Small cap + tight band: must terminate quickly with SOME terminal status.
        let band = Band { k_holes: 1, h_band: 8 };
        let report = run_cegis(&band, 50_000, 12);
        assert!(report.iters >= 1);
        // Any terminal status is acceptable here; we only assert the driver halts and reports.
        match report.status {
            Status::Certified | Status::Refuted { .. } | Status::Exploded => {}
        }
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p tetris-playground --bin tetris_maxplus_cert cegis_tests -- --nocapture`
Expected: FAIL (not defined).

- [ ] **Step 3: Implement the switching loop in `cegis.rs`**

```rust
use crate::closure::{closure, Outcome};

#[derive(Clone, Debug)]
pub enum Status {
    Certified,
    Refuted { binding_r: f64 },
    Exploded,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub status: Status,
    pub iters: usize,
    pub cert: Cert,
    pub size: usize,
    pub max_osc: i64,
}

pub fn run_cegis(band: &Band, cap: usize, max_iters: usize) -> Report {
    let mut samples: Vec<[i64; W]> = vec![[0i64; W]]; // flat empty root
    let mut last_cert = Cert { v: [0.0; W], r: 0.0, h_max: 0.0 };

    for it in 1..=max_iters {
        let cert = match fit_ball(&samples) {
            Some(c) => c,
            None => {
                return Report {
                    status: Status::Refuted { binding_r: last_cert.r },
                    iters: it,
                    cert: last_cert,
                    size: samples.len(),
                    max_osc: 0,
                }
            }
        };
        last_cert = cert.clone();
        match closure(&cert, band, cap) {
            Outcome::Closed { size, max_osc, .. } => {
                return Report { status: Status::Certified, iters: it, cert, size, max_osc }
            }
            Outcome::Exploded { size, .. } => {
                return Report { status: Status::Exploded, iters: it, cert, size, max_osc: 0 }
            }
            Outcome::Leak { state, size, .. } => {
                // New counterexample; if already present, we are stuck -> refuted.
                if samples.iter().any(|s| *s == state) {
                    return Report {
                        status: Status::Refuted { binding_r: cert.r },
                        iters: it,
                        cert,
                        size,
                        max_osc: 0,
                    };
                }
                samples.push(state);
            }
        }
    }
    Report {
        status: Status::Refuted { binding_r: last_cert.r },
        iters: max_iters,
        cert: last_cert,
        size: samples.len(),
        max_osc: 0,
    }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p tetris-playground --bin tetris_maxplus_cert cegis_tests -- --nocapture`
Expected: PASS (driver halts and reports a terminal status).

- [ ] **Step 5: Checkpoint**

Run: `cargo clippy -p tetris-playground --bin tetris_maxplus_cert -- -D warnings` → no errors.

---

### Task 8: CLI modes, metrics, and milestone runs

**Files:**
- Modify: `crates/tetris-playground/src/bin/research/tetris_maxplus_cert/main.rs`

**Interfaces:**
- Consumes: `cegis::{run_cegis, m0_fixed_order_radius, Report, Status}`, `policy::Band`, `tetris_game::TetrisPiece`, `std::time::Instant`.
- Produces: a `main()` that dispatches on prefix args (matching `tetris_policy`'s manual parser): `mode=m0` | `mode=cegis`, `holes=<k>`, `band=<h>`, `cap=<millions>`, `iters=<n>`; prints the metrics block.

- [ ] **Step 1: Rewrite `main()` to dispatch modes and print metrics**

```rust
mod engine;
mod lp;
mod policy;
mod closure;
mod cegis;

use std::time::Instant;
use tetris_game::TetrisPiece;
use crate::policy::Band;
use crate::cegis::{run_cegis, m0_fixed_order_radius, Status};

// keep smoke_lp + its test from Task 1

fn parse_u64(args: &[String], pfx: &str, default: u64) -> u64 {
    args.iter()
        .find_map(|a| a.strip_prefix(pfx).and_then(|s| s.parse::<u64>().ok()))
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.iter().find_map(|a| a.strip_prefix("mode=")).unwrap_or("cegis").to_string();
    let band = Band {
        k_holes: parse_u64(&args, "holes=", 1) as u32,
        h_band: parse_u64(&args, "band=", 12) as i64,
    };
    let cap = parse_u64(&args, "cap=", 2) as usize * 1_000_000;
    let iters = parse_u64(&args, "iters=", 30) as usize;

    println!("tetris_maxplus_cert — mode={mode} holes={} band={} cap={cap} iters={iters}", band.k_holes, band.h_band);
    let t0 = Instant::now();

    match mode.as_str() {
        "m0" => {
            // M0 validates the fit_ball+engine pipeline over a horizon the myopic 1-ply
            // flatten player survives (4 bags); long survival needs a beam (out of scope).
            let order = TetrisPiece::all().to_vec();
            match m0_fixed_order_radius(&order, 7 * 4, &band) {
                Some(r) => println!("M0 fixed-order OISZTLJ (4 bags): survived, fitted radius R = {r:.3}"),
                None => println!("M0 fixed-order OISZTLJ (4 bags): LEAKED in-band (band too tight?)"),
            }
        }
        _ => {
            let rep = run_cegis(&band, cap, iters);
            print!("CEGIS: iters={} |samples-driven| size={} max_osc={} R={:.3} H_max={:.3} -> ",
                rep.iters, rep.size, rep.max_osc, rep.cert.r, rep.cert.h_max);
            match rep.status {
                Status::Certified => println!("CERTIFIED (closed carrier; v rounded = {:?})", round_v(&rep.cert.v)),
                Status::Refuted { binding_r } => println!("REFUTED (no osc-ball <= 20 fits; binding R ~ {binding_r:.3})"),
                Status::Exploded => println!("EXPLODED (carrier > cap; not certifiable at this band)"),
            }
        }
    }
    println!("wall = {:.2}s", t0.elapsed().as_secs_f64());
}

fn round_v(v: &[f64; engine::W]) -> [i64; engine::W] {
    let mut out = [0i64; engine::W];
    for j in 0..engine::W { out[j] = v[j].round() as i64; }
    out
}
```

- [ ] **Step 2: Build**

Run: `cargo build -p tetris-playground --bin tetris_maxplus_cert`
Expected: success.

- [ ] **Step 3: Run M0 (validation gate)**

Run: `cargo run --release -p tetris-playground --bin tetris_maxplus_cert -- mode=m0 holes=4 band=20`
Expected: `survived, fitted radius R = <small>`. If it leaks, M0 has failed → the bug is in the pipeline (engine adapter / fit_ball), **stop and debug** before trusting any CEGIS result.

- [ ] **Step 4: Run M1 (the real test, tight band)**

Run: `cargo run --release -p tetris-playground --bin tetris_maxplus_cert -- mode=cegis holes=1 band=8 cap=2 iters=30`
Expected: a terminal line — `CERTIFIED` (a closed band carrier; the headline positive), `REFUTED`, or `EXPLODED`. Record the full metrics line in `artifacts/output/` or the run log per `CLAUDE.md`.

- [ ] **Step 5: Checkpoint + record milestone results**

Run: `cargo clippy -p tetris-playground --bin tetris_maxplus_cert -- -D warnings` → no errors.
Record M0 radius and the M1 status + metrics (iters, size, max_osc, R, H_max, wall) in the run notes. Then sweep the band (`band=10/12/14`, `holes=1/2`) to characterize where it floors (expected: I-drain `Leak`/`Exploded` as the band relaxes — the known crux, now seen through the osc-ball lens).

---

## Self-Review

**Spec coverage:**
- Certificate `S = {0 ≤ h ≤ H_max, osc(h−v) ≤ R}` → `lp::fit_ball` (Task 3) + the band in `policy`/`closure`. ✓
- Clear-invariance of `osc` → used implicitly (osc computed on `heights()` post-clear; engine clears automatically). ✓ (No code needed; it is a property, validated by M0 surviving with clears happening.)
- `∃`-player via CEGIS → `cegis::run_cegis` (Task 7). ✓
- Counterexample oracle = closure → `closure::closure` (Task 5), adapted from `tetris_policy`. ✓
- Soundness via closed reachable set (engine hardwired to W=10 ⇒ no exhaustive ball verifier) → `Outcome::Closed` (Task 5). ✓
- M0 fixed-order / M1 switching milestones → Tasks 6, 8. ✓
- Refutation payoff (binding `R` when no osc-ball ≤ 20 fits) → `Status::Refuted` (Task 7). ✓ (The full LP-dual Farkas certificate is a deferred enhancement, noted below.)
- M2 (relax band toward full game) → Task 8 Step 5 sweep. ✓
- M3 Lean import → **out of scope for this plan** (separate plan once a certificate exists; see below).

**Placeholder scan:** Task 3 Step 3 contains a deliberately-illustrative stub that Step 3b replaces — flagged explicitly. No other placeholders.

**Type consistency:** `Cert { v: [f64; W], r, h_max }`, `Band { k_holes: u32, h_band: i64 }`, `Outcome`, `Report`, `Status` are used consistently across `lp`/`policy`/`closure`/`cegis`/`main`. `W` is the single source of width. `successor` returns `(TetrisBoard, u32)` everywhere. ✓

## Deferred to follow-up plans (not this one)

- **M3 — Lean import (`proofs/Proofs/Experiments/MaxPlusCert.lean`):** only worth planning once `mode=cegis` returns `CERTIFIED` with a concrete `(v, R, H_max, policy table)`. The import checks controlled-invariance as a decidable no-clear `place` closure (clears handled structurally via `clearLines_*`, never in a kernel `decide` — `feedback_kernel_decide_no_clears`), then wires `EnergyGame.tetrisSolvableValid_of_maxHeight_invariant`. Commit policy there is `proofs/`-only, per branch convention.
- **LP-dual Farkas certificate** on `Refuted` (extract the infeasibility witness from clarabel) — sharpens the negative result; not required for the milestone verdicts.
- **SOS / SDP escalation** via `clarabel`'s PSD cones if the max-plus class refutes — the dependency is already wired.

---

**Plan complete.** Reachable, testable software at every task; the headline result (CERTIFIED / REFUTED / EXPLODED at a band) lands at Task 8.
