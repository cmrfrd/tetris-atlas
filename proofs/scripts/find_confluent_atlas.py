#!/usr/bin/env python3
"""Witness search for the confluence route to TetrisSolvable.

Searches for a CLOSED BOUNDARY GRAPH of order-robust bag phases: for each
reachable start-of-bag board ("boundary"), a fixed response map
piece -> (rot, col) such that EVERY one of the 7! within-bag arrival orders
survives, mid-bag forking stays within FORK_CAP distinct boards per placed
subset, and the end-of-bag boards (<= FINAL_CAP of them) are themselves
solved boundaries. When the frontier empties, the union of all subset
lattices is a CLOSED ADVERSARIAL TABLE: a finite atlas whose per-state
response keeps every legal 7-bag game inside the table forever. That table
is the object Lean re-certifies (checkTable-style native_decide + the atlas
bridge); this script carries no trust.

Design background (see Proofs/Invariants/Confluence.lean):
  - Placements with disjoint column footprints commute exactly
    (place_comm_of_colsDisjoint / foldl_place_perm), so a zone-local
    response map keeps the subset lattice THIN (few forks).
  - Full per-bag disjointness is impossible (min piece widths sum to
    13 > 10 columns), so some zones are shared and orders there fork; the
    caps bound the damage and end-of-bag clears help reconverge forks
    (divergence confined to rows that die at the boundary).
  - The policy is subset-oblivious per boundary: the same (rot, col) answers
    a piece no matter when it arrives in the bag. Hard-drop semantics adapt
    the landing to the current board.

Semantics mirror proofs/Proofs/Model exactly (same tables as
find_cooperative_lasso.py): dropOffset = sup(colHeight - up), place = union,
clearLines removes full rows and compacts, bag refills at 7.
"""
import argparse
import random
import sys
import time
from itertools import combinations

COLS = 10
ROWS = 20
ALL_PIECES = ("O", "I", "S", "Z", "T", "L", "J")
FULL = frozenset(ALL_PIECES)

# shapeUp per (piece, rotation): {piece_col: sorted list of ups}
# Transcribed from Piece.shape / Piece.shapeUp in Proofs/Model/Piece.lean.
SHAPE_UP = {
    ("O", 0): {0: [0, 1], 1: [0, 1]},
    ("I", 0): {0: [0], 1: [0], 2: [0], 3: [0]},
    ("I", 1): {0: [0, 1, 2, 3]},
    ("S", 0): {0: [0], 1: [0, 1], 2: [1]},
    ("S", 1): {0: [1, 2], 1: [0, 1]},
    ("Z", 0): {0: [1], 1: [0, 1], 2: [0]},
    ("Z", 1): {0: [0, 1], 1: [1, 2]},
    ("T", 0): {0: [1], 1: [0, 1], 2: [1]},
    ("T", 1): {0: [1], 1: [0, 1, 2]},
    ("T", 2): {0: [0], 1: [0, 1], 2: [0]},
    ("T", 3): {0: [0, 1, 2], 1: [1]},
    ("L", 0): {0: [0], 1: [0], 2: [0, 1]},
    ("L", 1): {0: [0, 1, 2], 1: [0]},
    ("L", 2): {0: [0, 1], 1: [1], 2: [1]},
    ("L", 3): {0: [2], 1: [0, 1, 2]},
    ("J", 0): {0: [0, 1], 1: [0], 2: [0]},
    ("J", 1): {0: [0, 1, 2], 1: [2]},
    ("J", 2): {0: [1], 1: [1], 2: [0, 1]},
    ("J", 3): {0: [0], 1: [0, 1, 2]},
}


def candidate_rotations(piece):
    return {
        "O": (0,), "I": (0, 1), "S": (0, 1), "Z": (0, 1),
        "T": (0, 1, 2, 3), "L": (0, 1, 2, 3), "J": (0, 1, 2, 3),
    }[piece]


# (piece) -> list of (rot, col, info); info = tuple of (abs_col, u0, ups)
PLACEMENTS = {p: [] for p in ALL_PIECES}
for _p in ALL_PIECES:
    for _rot in candidate_rotations(_p):
        _prof = SHAPE_UP[(_p, _rot)]
        _cs = sorted(_prof)
        _width = max(_cs) + 1
        _info0 = [(c, _prof[c][0], tuple(_prof[c])) for c in _cs]
        for _col in range(COLS - _width + 1):
            PLACEMENTS[_p].append(
                (_rot, _col, tuple((_col + c, u0, ups) for (c, u0, ups) in _info0)))

EMPTY = (0,) * COLS


def _pext(x, mask):
    """Remove bits of x where mask=1; compact the rest downward (line clear)."""
    res = 0
    outpos = 0
    hi = max(x.bit_length(), mask.bit_length())
    for r in range(hi):
        if not (mask >> r) & 1:
            if (x >> r) & 1:
                res |= (1 << outpos)
            outpos += 1
    return res


_STEP_CACHE = {}


def step(cols, info):
    """One hard-drop placement + line clears; None if lost (matches Lean).
    Memoized: lattice extensions revisit the same (board, placement) pairs
    heavily. info tuples are interned in PLACEMENTS, so id() is a valid key.
    """
    key = (cols, id(info))
    if key in _STEP_CACHE:
        return _STEP_CACHE[key]
    d = 0
    for (ac, u0, _ups) in info:
        t = cols[ac].bit_length() - u0
        if t > d:
            d = t
    newcols = list(cols)
    maxrow = 0
    for (ac, u0, ups) in info:
        cm = newcols[ac]
        for u in ups:
            cm |= (1 << (d + u))
            if d + u > maxrow:
                maxrow = d + u
        newcols[ac] = cm
    if maxrow >= ROWS:
        _STEP_CACHE[key] = None
        return None
    full = newcols[0]
    for c in newcols[1:]:
        full &= c
    if full:
        newcols = [_pext(c, full) for c in newcols]
    res = tuple(newcols)
    _STEP_CACHE[key] = res
    return res


def heights(cols):
    return [c.bit_length() for c in cols]


def cellcount(cols):
    return sum(bin(c).count("1") for c in cols)


def holecount(cols):
    return sum(c.bit_length() - bin(c).count("1") for c in cols)


# ---------------------------------------------------------------------------
# Subset lattice. nodes maps frozenset(placed pieces) -> set of boards
# reachable by placing exactly that subset in SOME order.
# ---------------------------------------------------------------------------

def extend_lattice(nodes, p, assign, fork_cap, hcap, final_cap=1):
    """Extend a lattice over placed set P (not containing p) to P + {p}.

    boards(F + {p}) needs ALL orders: the new piece may arrive at any
    position, so each new subset NF pulls from every predecessor NF - {q}
    (q ranging over NF), not just from placing p last. Iterating old
    subsets in increasing size makes every needed predecessor available.
    Returns the extended lattice, or None if pruned (loss, height cap,
    fork cap, final cap).
    """
    new = dict(nodes)
    for F in sorted(nodes, key=len):
        NF = F | {p}
        acc = set()
        for q in NF:
            prev = new[NF - {q}]
            qinfo = assign[q]
            for b in prev:
                nb = step(b, qinfo)
                if nb is None:
                    return None
                if max(heights(nb)) > hcap:
                    return None
                acc.add(nb)
        cap = final_cap if len(NF) == 7 else fork_cap
        if len(acc) > cap:
            return None
        new[NF] = acc
    return new


def full_lattice(start, assign, fork_cap, hcap, final_cap=1):
    """Lattice over all 7 pieces from scratch; None if pruned."""
    nodes = {frozenset(): {start}}
    for p in ALL_PIECES:
        nodes = extend_lattice(nodes, p, assign, fork_cap, hcap,
                               final_cap=final_cap)
        if nodes is None:
            return None
    return nodes


# ---------------------------------------------------------------------------
# Per-phase DFS: assign placements piece-by-piece, extending the lattice
# incrementally so partial assignments prune early. Troublemakers first.
# ---------------------------------------------------------------------------

PIECE_ORDER = ("S", "Z", "T", "L", "J", "O", "I")


class _Stop(Exception):
    pass


def phase_dfs(start, fork_cap, final_cap, hcap, deadline, max_sols=64,
              rng=None, prefer=None):
    """DFS for response maps at boundary `start`. `final_cap` bounds the
    number of distinct end-of-bag boards (1 = exact reconvergence; >1 lets
    the boundary graph branch). If `prefer` is given, a solution whose
    finals all lie in it short-circuits the search."""
    sols = []
    assign = {}
    meta = {}

    def dfs(idx, nodes):
        if time.time() > deadline or len(sols) >= max_sols:
            return
        if idx == 7:
            finals = frozenset(nodes[FULL])
            sols.append((dict(meta), finals, nodes))
            if prefer is not None and all(f in prefer for f in finals):
                sols[:] = [sols[-1]]
                raise _Stop
            return
        p = PIECE_ORDER[idx]
        cands = PLACEMENTS[p]
        if rng is not None:
            cands = list(cands)
            rng.shuffle(cands)
        for (rot, col, info) in cands:
            assign[p] = info
            meta[p] = (rot, col)
            nn = extend_lattice(nodes, p, assign, fork_cap, hcap,
                                final_cap=final_cap)
            if nn is not None:
                dfs(idx + 1, nn)
            if time.time() > deadline or len(sols) >= max_sols:
                break
        assign.pop(p, None)
        meta.pop(p, None)

    try:
        dfs(0, {frozenset(): {start}})
    except _Stop:
        pass
    return sols


# ---------------------------------------------------------------------------
# Boundary-graph closure. Solve each reachable boundary; new finals join the
# frontier; CLOSED when the frontier empties. Greedy with retry: prefer
# solutions whose finals are already solved or queued; a boundary with no
# solution is marked BAD and its predecessors re-solve avoiding it.
# ---------------------------------------------------------------------------

def score_board(cols):
    hs = heights(cols)
    return (cellcount(cols) * 100 + holecount(cols) * 500 + max(hs) * 50
            + sum(abs(hs[i] - hs[i + 1]) for i in range(COLS - 1)))


def graph_closure(args):
    t0 = time.time()
    rng = random.Random(args.seed) if args.shuffle else None
    solved = {}      # boundary -> (meta, finals, states_count)
    bad = set()
    frontier = [EMPTY]
    solves = 0

    while frontier:
        if time.time() > t0 + args.budget:
            print(f"budget exhausted: {len(solved)} solved, "
                  f"{len(frontier)} open, {len(bad)} bad", flush=True)
            return 1
        frontier.sort(key=score_board)
        b = frontier.pop(0)
        if b in solved or b in bad:
            continue
        deadline = min(time.time() + args.phase_budget, t0 + args.budget)
        prefer = set(solved) | set(frontier) | {b}
        sols = phase_dfs(b, args.fork_cap, args.final_cap, args.hcap,
                         deadline, max_sols=args.phase_sols, rng=rng,
                         prefer=prefer)
        sols = [s for s in sols if not (s[1] & bad)]
        solves += 1
        if not sols:
            bad.add(b)
            if b == EMPTY:
                print(f"EMPTY board unsolvable within caps "
                      f"({time.time() - t0:.0f}s) — relax caps", flush=True)
                return 1
            requeue = [s for s, (_m, fs, _st) in solved.items() if b in fs]
            for s in requeue:
                del solved[s]
                frontier.append(s)
            print(f"[{time.time() - t0:5.0f}s] dead-end boundary "
                  f"(cells={cellcount(b)}); requeued {len(requeue)} "
                  f"predecessors", flush=True)
            continue

        def sol_score(s):
            _meta, finals, nodes = s
            new = [f for f in finals if f not in solved and f != b]
            states = sum(len(v) for v in nodes.values())
            return (len(new), len(finals), states,
                    sum(score_board(f) for f in finals))

        sols.sort(key=sol_score)
        meta, finals, nodes = sols[0]
        states = sum(len(v) for v in nodes.values())
        solved[b] = (meta, finals, states)
        newf = 0
        for f in finals:
            if f not in solved and f not in bad and f != b:
                frontier.append(f)
                newf += 1
        print(f"[{time.time() - t0:5.0f}s] solved #{len(solved)} "
              f"(cells={cellcount(b)} hmax={max(heights(b)) if b != EMPTY else 0})"
              f" -> {len(finals)} finals ({newf} new), lattice={states}, "
              f"frontier={len(frontier)}", flush=True)

    total = sum(st for (_m, _f, st) in solved.values())
    print(f"CLOSED: {len(solved)} boundaries, total lattice states={total}, "
          f"{time.time() - t0:.0f}s, {solves} solves", flush=True)
    emit_closed(solved, args)
    return 0


def emit_closed(solved, args):
    """Re-verify closure from scratch and print the atlas description."""
    for b, (meta, finals, _st) in sorted(
            solved.items(), key=lambda kv: score_board(kv[0])):
        assign = {}
        for p in ALL_PIECES:
            rot, col = meta[p]
            info = next(i for (r, c, i) in PLACEMENTS[p]
                        if r == rot and c == col)
            assign[p] = info
        nodes = full_lattice(b, assign, args.fork_cap, args.hcap,
                             final_cap=args.final_cap)
        assert nodes is not None, "re-check failed"
        assert frozenset(nodes[FULL]) == finals, "finals mismatch"
        assert all(f in solved for f in finals), "not closed"
        hs = heights(b)
        print(f"boundary cells={cellcount(b)} h={hs}: " + ", ".join(
            f"{p}:r{meta[p][0]}c{meta[p][1]}" for p in ALL_PIECES)
            + f" -> {len(finals)} finals")
    print("closure re-verified", flush=True)


# ---------------------------------------------------------------------------
# State-dependent closure with a lasso attractor. The Lean atlas format is a
# per-STATE response table, so the policy may depend on the whole (board,
# bag) — subset-obliviousness was a search simplification, and cross-piece
# box pairs provably don't exist (2x4 boxes tile only as L+L/J+J/O+O/I+I),
# so fixed responses can't tame the forks. Here: BFS the reachable set under
# a policy that (1) steers into states already tabled, (2) pulls toward the
# 35-board orbit of the cooperative lasso, (3) otherwise minimizes a
# holes/height/bumpiness potential. Closes iff the frontier empties.
# ---------------------------------------------------------------------------

LASSO_SEQ = [
    ("T", 2, 3), ("L", 0, 7), ("S", 1, 5), ("Z", 0, 6), ("I", 0, 0),
    ("O", 0, 8), ("J", 0, 0),
    ("J", 0, 2), ("S", 1, 0), ("I", 0, 4), ("T", 3, 3), ("O", 0, 1),
    ("L", 0, 7), ("Z", 0, 4),
    ("T", 2, 6), ("L", 2, 0), ("Z", 0, 7), ("O", 0, 5), ("J", 3, 3),
    ("I", 0, 0), ("S", 0, 2),
    ("J", 2, 7), ("O", 0, 5), ("S", 0, 0), ("L", 0, 7), ("T", 3, 0),
    ("I", 0, 4), ("Z", 0, 1),
    ("Z", 1, 8), ("S", 0, 6), ("J", 0, 3), ("T", 2, 4), ("L", 2, 2),
    ("O", 0, 0), ("I", 0, 6),
]


def lasso_orbit():
    """The 35 states of the cooperative lasso (board before each placement)."""
    boards = set()
    states = set()
    b, bag = EMPTY, FULL
    for (p, rot, col) in LASSO_SEQ:
        boards.add(b)
        states.add((b, bag))
        info = next(i for (r, c, i) in PLACEMENTS[p]
                    if r == rot and c == col)
        b = step(b, info)
        assert b is not None
        bag = bag - {p} or FULL
    assert b == EMPTY and bag == FULL
    return boards, states


def potential(cols):
    hs = heights(cols)
    bump = sum(abs(hs[i] - hs[i + 1]) for i in range(COLS - 1))
    return holecount(cols) * 800 + max(hs) * 60 + bump * 8 + cellcount(cols)


# ---------------------------------------------------------------------------
# Zone discipline (Route 1: the wiki "playing forever" design, adaptivity
# replacing hold). S/T/Z play only in columns 0-3, L/J/O only in columns
# 6-9, I only vertically in columns 4-5 (the alternating drain). Forks from
# awkward orders then stay confined to one 4-column zone, and the I-drain's
# clears are uniform shifts that preserve zone shapes
# (dropOffset_skyline_sub). ZONE_PLACEMENTS[p] is the restricted candidate
# list; zone_potential scores per-zone flatness instead of global bumpiness
# (the three stacks are SUPPOSED to differ in height).
# ---------------------------------------------------------------------------

ZONES = {"S": (0, 3), "T": (0, 3), "Z": (0, 3),
         "L": (6, 9), "J": (6, 9), "O": (6, 9)}


def _zone_ok(p, info):
    if p == "I":
        return len(info) == 1 and info[0][0] in (4, 5)  # vertical, col 4/5
    lo, hi = ZONES[p]
    return all(lo <= ac <= hi for (ac, _u0, _ups) in info)


ZONE_PLACEMENTS = {
    p: [(rot, col, info) for (rot, col, info) in PLACEMENTS[p]
        if _zone_ok(p, info)]
    for p in ALL_PIECES
}


def zone_bump(cols):
    """Bumpiness WITHIN each zone only (the three stacks may differ)."""
    hs = heights(cols)
    return (sum(abs(hs[i] - hs[i + 1]) for i in range(0, 3))
            + sum(abs(hs[i] - hs[i + 1]) for i in range(6, 9))
            + abs(hs[4] - hs[5]))


def stz_service(hs):
    """S/Z-serviceability of the STZ zone (cols 0-3): 0 when the zone offers
    both a unit down-step (S seats flush, self-reproducing) and a unit
    up-step (Z). A FLAT STZ zone is the failure mode — S/Z must hole on flat
    (no_holefree_S_on_flat) and the debt spiral kills the walk."""
    downs = any(hs[i] == hs[i + 1] + 1 for i in range(3))
    ups = any(hs[i + 1] == hs[i] + 1 for i in range(3))
    return (0 if downs else 1) + (0 if ups else 1)


def zone_shape_ok(cols, mid_diff_cap=4):
    """Per-zone boundary shape discipline (the band the wiki shapes live
    in): side-zone adjacent steps of magnitude <= 1 (flat kills S/Z,
    cliffs kill everything), STZ zone S/Z-servable, middle stagger
    bounded. Post-clear debris (deep cliffs) fails this and is rejected as
    a boundary."""
    hs = heights(cols)
    for i in range(0, 3):
        if abs(hs[i] - hs[i + 1]) > 2:
            return False
    for i in range(6, 9):
        if abs(hs[i] - hs[i + 1]) > 2:
            return False
    if abs(hs[4] - hs[5]) > mid_diff_cap:
        return False
    # clear-starvation guard: side pieces can only complete rows BELOW the
    # middle pair's minimum, and the I-tetrises alone (2 rows/bag) cannot
    # match the 2.8 rows/bag the zones add — so past the two bootstrap bags
    # the mid stagger must cycle high (min >= 4), never touching 0.
    if cellcount(cols) > 28 and min(hs[4], hs[5]) < 4:
        return False
    return True


def zone_potential(cols):
    hs = heights(cols)
    ljo_bump = sum(abs(hs[i] - hs[i + 1]) for i in range(6, 9))
    return (holecount(cols) * 800 + max(hs) * 60 + stz_service(hs) * 120
            + ljo_bump * 8 + abs(hs[4] - hs[5]) * 4 + cellcount(cols))


def closure_search(args):
    from collections import deque
    t0 = time.time()
    orbit_boards, orbit_states = lasso_orbit()
    table = {}
    known_boards = {EMPTY}
    queue = deque()
    root = (EMPTY, FULL)
    queue.append(root)
    queued = {root}
    deaths = 0
    while queue:
        if len(table) >= args.state_cap or time.time() > t0 + args.budget:
            print(f"DIVERGED/timeout: table={len(table)} "
                  f"frontier={len(queue)} deaths={deaths} "
                  f"{time.time() - t0:.0f}s", flush=True)
            report_table(table, orbit_states)
            return 1
        (b, bag) = queue.popleft()
        if (b, bag) in table:
            continue
        resp = {}
        dead = False
        succs = []
        for p in sorted(bag):
            best = None
            for (rot, col, info) in PLACEMENTS[p]:
                nb = step(b, info)
                if nb is None:
                    continue
                if max(heights(nb)) > args.hcap:
                    continue
                nbag = bag - {p}
                if not nbag:
                    nbag = FULL
                key = (nb, nbag)
                pri = ((0 if (key in table or key in queued) else 1),
                       (0 if nb in known_boards else 1),
                       (0 if nb in orbit_boards else 1),
                       potential(nb), rot, col)
                if best is None or pri < best[0]:
                    best = (pri, (rot, col), key)
            if best is None:
                deaths += 1
                dead = True
                break
            resp[p] = best[1]
            succs.append(best[2])
        if dead:
            print(f"[{time.time() - t0:5.0f}s] DEATH at state "
                  f"cells={cellcount(b)} h={heights(b)} bag={sorted(bag)}",
                  flush=True)
            continue
        table[(b, bag)] = resp
        for key in succs:
            known_boards.add(key[0])
            if key not in table and key not in queued:
                queue.append(key)
                queued.add(key)
        if len(table) % args.log_every == 0:
            onorbit = sum(1 for s in table if s in orbit_states)
            nb_ratio = len(table) / max(1, len(known_boards))
            print(f"[{time.time() - t0:5.0f}s] table={len(table)} "
                  f"frontier={len(queue)} boards={len(known_boards)} "
                  f"(reuse x{nb_ratio:.2f}) on-orbit={onorbit} "
                  f"deaths={deaths}", flush=True)
    print(f"CLOSED: table={len(table)} states, deaths={deaths}, "
          f"{time.time() - t0:.0f}s", flush=True)
    report_table(table, orbit_states)
    if deaths == 0:
        verify_closure(table)
        return 0
    print("closed but with DEATH states pruned — NOT a valid atlas",
          flush=True)
    return 1


def report_table(table, orbit_states):
    if not table:
        return
    hol = [holecount(b) for (b, _) in table]
    hts = [max(heights(b)) for (b, _) in table]
    cells = [cellcount(b) for (b, _) in table]
    boards = len({b for (b, _) in table})
    print(f"  boards={boards} | holes max={max(hol)} avg={sum(hol)/len(hol):.2f}"
          f" | hmax max={max(hts)} | cells max={max(cells)} "
          f"avg={sum(cells)/len(cells):.1f}", flush=True)


def verify_closure(table):
    """Independent pass: every (state, piece) response stays in the table."""
    for (b, bag), resp in table.items():
        for p in bag:
            rot, col = resp[p]
            info = next(i for (r, c, i) in PLACEMENTS[p]
                        if r == rot and c == col)
            nb = step(b, info)
            assert nb is not None, "death in verify"
            nbag = bag - {p}
            if not nbag:
                nbag = FULL
            assert (nb, nbag) in table, "escape in verify"
    print(f"closure INDEPENDENTLY VERIFIED: {len(table)} states, "
          f"every in-bag piece response stays inside", flush=True)


# ---------------------------------------------------------------------------
# Adaptive per-bag reconvergence (the decisive experiment). Within one bag,
# the policy may answer each (board, remaining-bag, piece) individually — so
# forks opened by the adversary's order can be REFOLDED by compensating
# placements on each branch. solve_bag decides, by memoized AND-OR search,
# whether from a boundary board every arrival order can be answered so that
# every end-of-bag board lands in an acceptable set (existing boundaries or
# `qualify`-good new ones), and extracts the per-state response table.
# ---------------------------------------------------------------------------

_USE_ZONES = False


def bag_candidates(board, p, k, hcap):
    """Top-k placements of p on board by potential (holes first). With
    zone discipline on, candidates are restricted to the piece's zone and
    scored by per-zone flatness."""
    out = []
    plist = ZONE_PLACEMENTS[p] if _USE_ZONES else PLACEMENTS[p]
    pot = zone_potential if _USE_ZONES else potential
    for (rot, col, info) in plist:
        nb = step(board, info)
        if nb is None:
            continue
        if max(heights(nb)) > hcap:
            continue
        out.append((pot(nb), rot, col, info, nb))
    out.sort(key=lambda x: x[0])
    return out[:k]


class _Deadline(Exception):
    pass


def solve_bag(B, accept, args, strict_set=None, deadline=None, memo=None,
              choice=None):
    """AND-OR solve one bag from boundary B. accept(board) -> bool at bag
    end. `strict_set` (if given) reorders last-piece candidates to prefer
    finals inside it. `memo`/`choice` may be shared across solves AGAINST
    THE SAME accept SET (bag DAGs from nearby boundaries overlap heavily).
    Returns (ok, table, finals, memo_size): table maps (board, remaining)
    -> {piece: (rot, col)}; finals = end-of-bag boards actually reached
    under the extracted policy (all orders)."""
    if memo is None:
        memo = {}
    if choice is None:
        choice = {}

    def rec(board, remaining):
        key = (board, remaining)
        if key in memo:
            return memo[key]
        if deadline is not None and time.time() > deadline:
            raise _Deadline
        if not remaining:
            ok = accept(board)
            memo[key] = ok
            return ok
        ok = True
        picks = {}
        for p in remaining:
            found = False
            cands = bag_candidates(board, p, args.branch, args.hcap)
            if strict_set is not None and len(remaining) == 1:
                cands = sorted(
                    cands, key=lambda x: (0 if x[4] in strict_set else 1,
                                          x[0]))
            for (_pot, rot, col, info, nb) in cands:
                if rec(nb, remaining - {p}):
                    picks[p] = (rot, col)
                    found = True
                    break
            if not found:
                ok = False
                break
        memo[key] = ok
        if ok:
            choice[key] = picks
        return ok

    try:
        ok = rec(B, FULL)
    except _Deadline:
        return False, None, None, len(memo)
    if not ok:
        return False, None, None, len(memo)
    # extract reachable sub-DAG under the chosen policy
    table = {}
    finals = set()
    stack = [(B, FULL)]
    seen = {(B, FULL)}
    while stack:
        (board, remaining) = stack.pop()
        if not remaining:
            finals.add(board)
            continue
        picks = choice[(board, remaining)]
        table[(board, remaining)] = picks
        for p in remaining:
            rot, col = picks[p]
            info = next(i for (r, c, i) in PLACEMENTS[p]
                        if r == rot and c == col)
            nb = step(board, info)
            nk = (nb, remaining - {p})
            if nk not in seen:
                seen.add(nk)
                stack.append(nk)
    return True, table, finals, len(memo)


def bump(cols):
    hs = heights(cols)
    return sum(abs(hs[i] - hs[i + 1]) for i in range(COLS - 1))


def adaptive_closure(args):
    t0 = time.time()
    solved = {}       # boundary -> (table, finals)
    bad = set()
    frontier = [EMPTY]
    queued = {EMPTY}
    tier2 = 0
    deadends = 0
    while frontier:
        if time.time() > t0 + args.budget:
            total_states = sum(len(t) for (t, _f) in solved.values())
            print(f"budget exhausted: {len(solved)} boundaries solved, "
                  f"{len(frontier)} open, {len(bad)} bad, "
                  f"states={total_states}, tier2-solves={tier2}", flush=True)
            return 1
        frontier.sort(key=score_board)
        B = frontier.pop(0)
        queued.discard(B)
        if B in solved or B in bad:
            continue
        known = (set(solved) | queued | {B}) - bad

        def accept1(board, _k=known):
            return board in _k

        def accept2(board, _k=known, _bad=bad):
            if board in _k:
                return True
            if board in _bad:
                return False
            if holecount(board) > args.final_holes:
                return False
            if max(heights(board)) > args.final_h:
                return False
            if cellcount(board) > args.final_cells:
                return False
            if _USE_ZONES:
                return True
            return bump(board) <= args.final_bump

        ok, table, finals, memo_n = solve_bag(B, accept1, args,
                                              strict_set=known)
        used2 = False
        if not ok:
            ok, table, finals, memo_n = solve_bag(B, accept2, args,
                                                  strict_set=known)
            used2 = True
            tier2 += 1
        el = time.time() - t0
        if not ok:
            if B == EMPTY:
                print(f"[{el:5.0f}s] EMPTY unsolvable (branch={args.branch}, "
                      f"memo={memo_n}) — raise --branch or relax finals",
                      flush=True)
                return 1
            bad.add(B)
            deadends += 1
            requeue = [s for s, (_t, fs) in solved.items() if B in fs]
            for s in requeue:
                del solved[s]
                if s not in queued:
                    frontier.append(s)
                    queued.add(s)
            print(f"[{el:5.0f}s] dead-end (cells={cellcount(B)} "
                  f"h={heights(B)} holes={holecount(B)}); bad={len(bad)}, "
                  f"requeued {len(requeue)} producers", flush=True)
            continue
        solved[B] = (table, finals)
        newf = 0
        for f in finals:
            if f not in solved and f not in queued and f not in bad and f != B:
                frontier.append(f)
                queued.add(f)
                newf += 1
        if len(solved) % args.log_every == 0 or newf > 20:
            total_states = sum(len(t) for (t, _f) in solved.values())
            print(f"[{el:5.0f}s] solved #{len(solved)} (cells={cellcount(B)}"
                  f"{' T2' if used2 else ' T1'}) memo={memo_n} "
                  f"bag-states={len(table)} finals={len(finals)} (+{newf}) "
                  f"frontier={len(frontier)} states={total_states}",
                  flush=True)
    # closure only counts if every solved boundary's finals are solved
    open_refs = {f for (_t, fs) in solved.values() for f in fs
                 if f not in solved}
    total_states = sum(len(t) for (t, _f) in solved.values())
    if open_refs:
        print(f"frontier empty but {len(open_refs)} finals unsolved "
              f"(bad-referencing) — NOT closed", flush=True)
        return 1
    print(f"CLOSED: {len(solved)} boundaries, total mid-bag states="
          f"{total_states}, tier2={tier2}, deadends={deadends}, "
          f"{time.time() - t0:.0f}s", flush=True)
    verify_adaptive(solved)
    return 0


def verify_adaptive(solved):
    """Independent closure pass over the union of all bag tables."""
    boundaries = set(solved)
    checked = 0
    for B, (table, finals) in solved.items():
        assert all(f in boundaries for f in finals), "finals escape"
        for (board, remaining), picks in table.items():
            for p in remaining:
                rot, col = picks[p]
                info = next(i for (r, c, i) in PLACEMENTS[p]
                            if r == rot and c == col)
                nb = step(board, info)
                assert nb is not None, "death in verify"
                nrem = remaining - {p}
                if nrem:
                    assert (nb, nrem) in table, "mid-bag escape"
                else:
                    assert nb in boundaries, "boundary escape"
                checked += 1
    print(f"adaptive closure INDEPENDENTLY VERIFIED: "
          f"{sum(len(t) for (t, _f) in solved.values())} mid-bag states, "
          f"{checked} transitions, {len(boundaries)} boundaries", flush=True)


# ---------------------------------------------------------------------------
# Designed-family GFP. Cell arithmetic (28 cells/bag, clears in multiples of
# 10) forces boundary cell-counts through residues 0->8->6->4->2->0 mod 10,
# so flat boundaries cannot recur; the canonical family is the LEDGE: a flat
# base at height h with the top row partially filled by k cells (left-
# justified), optionally with one debt hole. GFP: prune every family member
# whose adaptive bag cannot land wholly inside the family; iterate to the
# greatest fixed point; then check EMPTY leads into the surviving family
# within a few bags.
# ---------------------------------------------------------------------------

def ledge_boards(hmax, with_debt):
    """ledge(h, k): cols 0..k-1 at height h+1, cols k..9 at height h.
    with_debt: also one-hole variants (hole at row h-1 under a filled cell
    at row h, in column j >= k)."""
    fam = set()
    for h in range(hmax + 1):
        base = (1 << h) - 1
        for k in range(COLS):
            cols = tuple((base | (1 << h)) if j < k else base
                         for j in range(COLS))
            fam.add(cols)
            if with_debt and h >= 1:
                for j in range(k, COLS):
                    dcols = list(cols)
                    # bury a hole at row h-1 of column j: cell at h, empty h-1
                    dcols[j] = (base & ~(1 << (h - 1))) | (1 << h)
                    fam.add(tuple(dcols))
    return fam


def family_gfp(args):
    t0 = time.time()
    fam = ledge_boards(args.family_h, args.family_debt)
    print(f"designed family: {len(fam)} ledge boards "
          f"(hmax={args.family_h}, debt={args.family_debt})", flush=True)
    it = 0
    while True:
        it += 1
        dead = []
        for B in sorted(fam, key=score_board):
            if time.time() > t0 + args.budget:
                print("budget exhausted mid-GFP", flush=True)
                return 1
            ok, _t, _f, memo_n = solve_bag(
                B, (lambda b: b in fam), args, strict_set=fam)
            if not ok:
                dead.append(B)
        if not dead:
            break
        for B in dead:
            fam.discard(B)
        print(f"[{time.time() - t0:5.0f}s] GFP iter {it}: pruned "
              f"{len(dead)}, family={len(fam)}", flush=True)
        if not fam:
            print("family EMPTY — no ledge GFP at these caps", flush=True)
            return 1
    print(f"[{time.time() - t0:5.0f}s] GFP STABLE: family={len(fam)} "
          f"after {it} iterations", flush=True)
    for B in sorted(fam, key=score_board)[:20]:
        print(f"  member h={heights(B)} cells={cellcount(B)} "
              f"holes={holecount(B)}")
    # lead-in from EMPTY: bags may pass through non-family boundaries
    if EMPTY in fam:
        print("EMPTY is IN the family — closed atlas from init!", flush=True)
        return 0
    lead = {EMPTY}
    for level in range(1, args.leadin + 1):
        nxt = set()
        okall = True
        for B in sorted(lead, key=score_board):
            def acc(b):
                return (b in fam
                        or (holecount(b) <= 1
                            and max(heights(b)) <= args.final_h
                            and bump(b) <= args.final_bump))
            ok, _t, finals, _m = solve_bag(B, acc, args, strict_set=fam)
            if not ok:
                okall = False
                break
            nxt |= {f for f in finals if f not in fam}
        if not okall:
            print(f"lead-in level {level}: some boundary unsolvable",
                  flush=True)
            return 1
        if not nxt:
            print(f"LEAD-IN CLOSED at level {level}: EMPTY -> family. "
                  f"CLOSED ATLAS EXISTS.", flush=True)
            return 0
        print(f"lead-in level {level}: {len(nxt)} boundaries still outside "
              f"family", flush=True)
        lead = nxt
    print("lead-in did not close within levels", flush=True)
    return 1


# ---------------------------------------------------------------------------
# Single-target cycle probe: the sharpest reconvergence question. An edge
# B -> f exists when the adaptive policy can steer EVERY arrival order from
# boundary B onto the single final board f. If such edges exist among the
# ledge/debt shapes, a 5-bag cycle (cells mod 10: 0->8->6->4->2->0) plus an
# EMPTY lead-in is the smallest possible closed atlas.
# ---------------------------------------------------------------------------

def cycle_probe(args):
    t0 = time.time()
    U = sorted(ledge_boards(args.family_h, True), key=score_board)
    print(f"universe: {len(U)} ledge+debt boards (hmax={args.family_h})",
          flush=True)
    edges = {}
    tried = 0
    found = 0
    for B in U:
        if time.time() > t0 + args.budget:
            break
        cb = cellcount(B)
        targets = [f for f in U
                   if (cb + 28 - cellcount(f)) in (0, 10, 20, 30)]
        targets.sort(key=lambda f: (abs(cellcount(f) - cb - 8),
                                    score_board(f)))
        for f in targets[:args.cycle_targets]:
            if time.time() > t0 + args.budget:
                break
            tried += 1
            ok, _t, finals, memo_n = solve_bag(
                B, (lambda b, _f=f: b == _f), args, strict_set={f},
                deadline=time.time() + args.solve_budget)
            if ok:
                found += 1
                edges.setdefault(B, []).append(f)
                print(f"[{time.time() - t0:5.0f}s] EDGE: cells={cb} "
                      f"h={heights(B)} -> cells={cellcount(f)} "
                      f"h={heights(f)} (memo={memo_n})", flush=True)
    print(f"[{time.time() - t0:5.0f}s] probe done: {found} edges / {tried} "
          f"pairs tried, {len(edges)} boundaries with an edge", flush=True)
    if not edges:
        print("NO single-target edges — single-final reconvergence too "
              "rigid at these caps", flush=True)
        return 1
    # cycle search in the edge graph
    import functools
    reach = {B: set(fs) for B, fs in edges.items()}
    for B in edges:
        stack, seen = [B], set()
        while stack:
            x = stack.pop()
            for y in reach.get(x, ()):
                if y == B:
                    print(f"CYCLE through cells={cellcount(B)} "
                          f"h={heights(B)}!", flush=True)
                    return 0
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
    print("edges exist but no cycle yet — enlarge universe/targets",
          flush=True)
    return 1


# ---------------------------------------------------------------------------
# Family v2: residue-stratified contiguous-run family (data-driven — the
# minfinals winners are exactly these shapes). Member = flat base of height
# base_h with a contiguous run of k cells on top (start s). Cells =
# 10*base_h + k covers every residue; runs give S/Z their seating steps at
# the run edges. GFP with a SHARED memo per iteration (bag DAGs from nearby
# boundaries overlap heavily; memo entries depend on the accept set, so the
# share is per-iteration only).
# ---------------------------------------------------------------------------

def run_boards(bases=(0, 1), extra_flats=(2,)):
    fam = set()
    for base_h in bases:
        base = (1 << base_h) - 1
        for k in range(0, COLS + 1):
            for s in range(0, COLS - k + 1):
                cols = tuple(
                    (base | (1 << base_h)) if s <= j < s + k else base
                    for j in range(COLS))
                fam.add(cols)
    for h in extra_flats:
        fam.add(((1 << h) - 1,) * COLS)
    return fam


def family_gfp2(args):
    t0 = time.time()
    fam = run_boards()
    print(f"run-family: {len(fam)} boards (contiguous-run strata)",
          flush=True)
    it = 0
    while True:
        it += 1
        memo = {}
        choice = {}
        dead = []
        for B in sorted(fam, key=score_board):
            if time.time() > t0 + args.budget:
                print("budget exhausted mid-GFP", flush=True)
                return 1
            ok, _t, _f, _m = solve_bag(
                B, (lambda b: b in fam), args, strict_set=fam,
                deadline=time.time() + args.solve_budget,
                memo=memo, choice=choice)
            if not ok:
                dead.append(B)
        print(f"[{time.time() - t0:5.0f}s] GFP iter {it}: pruned "
              f"{len(dead)}/{len(fam)} (shared memo={len(memo)})", flush=True)
        if not dead:
            break
        for B in dead:
            fam.discard(B)
        if not fam:
            print("family EMPTY — run-family does not close at these caps",
                  flush=True)
            return 1
    print(f"[{time.time() - t0:5.0f}s] GFP STABLE: {len(fam)} members "
          f"after {it} iterations:", flush=True)
    for B in sorted(fam, key=score_board):
        print(f"  h={heights(B)} cells={cellcount(B)}")
    # lead-in from EMPTY (EMPTY may be in fam already)
    if EMPTY in fam:
        print("EMPTY IS IN THE STABLE FAMILY — CLOSED ATLAS FROM INIT. "
              "Emit and certify!", flush=True)
        return 0
    def acc(b):
        return (b in fam
                or (holecount(b) <= 1 and max(heights(b)) <= args.final_h
                    and bump(b) <= args.final_bump
                    and cellcount(b) <= args.final_cells))
    lead = {EMPTY}
    for level in range(1, args.leadin + 1):
        nxt = set()
        for B in sorted(lead, key=score_board):
            ok, _t, finals, _m = solve_bag(
                B, acc, args, strict_set=fam,
                deadline=time.time() + 6 * args.solve_budget)
            if not ok:
                print(f"lead-in level {level}: unsolvable boundary "
                      f"h={heights(B)}", flush=True)
                return 1
            nxt |= {f for f in finals if f not in fam}
        if not nxt:
            print(f"LEAD-IN CLOSED at level {level}: EMPTY -> family. "
                  f"CLOSED ATLAS EXISTS.", flush=True)
            return 0
        print(f"lead-in level {level}: {len(nxt)} outside family",
              flush=True)
        lead = nxt
    print("lead-in did not close", flush=True)
    return 1


# ---------------------------------------------------------------------------
# Minimal-finals probe: THE decisive statistic for boundary concentration.
# From a boundary B, solve loosely, then greedily shrink the accept set
# (drop the worst final; re-solve; keep the shrunken extracted finals) until
# no single removal is solvable. Reports the (locally) minimal finals set —
# if it lands near 5-10, a small closed family F is plausible; if it floors
# at hundreds, subset-oblivious-free adaptive concentration is dead too.
# ---------------------------------------------------------------------------

def min_finals(B, args, t0):
    def qual(b):
        return (holecount(b) <= args.final_holes
                and max(heights(b)) <= args.final_h
                and cellcount(b) <= args.final_cells
                and bump(b) <= args.final_bump)

    ok, _t, finals, memo_n = solve_bag(
        B, qual, args, deadline=time.time() + 4 * args.solve_budget)
    if not ok:
        print(f"  loose solve failed (memo={memo_n})", flush=True)
        return None
    S = set(finals)
    print(f"[{time.time() - t0:5.0f}s] loose finals: {len(S)}", flush=True)
    rounds = 0
    while time.time() < t0 + args.budget:
        rounds += 1
        improved = False
        for f in sorted(S, key=score_board, reverse=True):
            T = frozenset(S - {f})
            ok2, _t2, fin2, _m2 = solve_bag(
                B, (lambda b, _T=T: b in _T), args, strict_set=T,
                deadline=time.time() + args.solve_budget)
            if ok2:
                S = set(fin2)
                improved = True
                print(f"[{time.time() - t0:5.0f}s] shrink round {rounds}: "
                      f"finals={len(S)}", flush=True)
                break
        if not improved:
            break
    return S


def minfinals_probe(args):
    t0 = time.time()
    print("boundary: EMPTY", flush=True)
    S = min_finals(EMPTY, args, t0)
    if S is None:
        return 1
    print(f"MINIMAL(ish) finals from EMPTY: {len(S)}", flush=True)
    for f in sorted(S, key=score_board)[:12]:
        print(f"  h={heights(f)} cells={cellcount(f)} holes={holecount(f)}")
    # probe one second-generation boundary too
    nxt = sorted(S, key=score_board)[0]
    print(f"boundary: best final h={heights(nxt)}", flush=True)
    S2 = min_finals(nxt, args, t0)
    if S2 is not None:
        print(f"MINIMAL(ish) finals from it: {len(S2)}", flush=True)
        both = S & S2
        print(f"overlap with first set: {len(both)}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("graph", "closure", "adaptive",
                                       "family", "family2", "cycle",
                                       "minfinals"),
                    default="closure")
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--phase-budget", type=float, default=30.0)
    ap.add_argument("--phase-sols", type=int, default=48)
    ap.add_argument("--fork-cap", type=int, default=4,
                    help="max distinct boards per proper subset")
    ap.add_argument("--final-cap", type=int, default=3,
                    help="max distinct end-of-bag boards")
    ap.add_argument("--hcap", type=int, default=16)
    ap.add_argument("--state-cap", type=int, default=300000)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--branch", type=int, default=4,
                    help="adaptive: OR-branching per (board, piece)")
    ap.add_argument("--final-holes", type=int, default=1,
                    help="adaptive: max holes for a new boundary")
    ap.add_argument("--final-h", type=int, default=8,
                    help="adaptive: max height for a new boundary")
    ap.add_argument("--final-cells", type=int, default=36,
                    help="adaptive: max cells for a new boundary")
    ap.add_argument("--final-bump", type=int, default=6,
                    help="adaptive: max bumpiness for a new boundary")
    ap.add_argument("--family-h", type=int, default=4,
                    help="family: max ledge base height")
    ap.add_argument("--family-debt", action="store_true",
                    help="family: include one-hole debt variants")
    ap.add_argument("--leadin", type=int, default=4,
                    help="family: max lead-in bags from EMPTY")
    ap.add_argument("--cycle-targets", type=int, default=12,
                    help="cycle: candidate targets per boundary")
    ap.add_argument("--solve-budget", type=float, default=8.0,
                    help="cycle: seconds per (B, f) solve")
    ap.add_argument("--zones", action="store_true",
                    help="zone discipline: S/T/Z cols 0-3, L/J/O cols 6-9, "
                         "I vertical cols 4-5 (Route 1)")
    args = ap.parse_args()
    if args.zones:
        global _USE_ZONES
        _USE_ZONES = True
        for p in ALL_PIECES:
            assert ZONE_PLACEMENTS[p], f"no zone placements for {p}"
    if args.mode == "graph":
        return graph_closure(args)
    if args.mode == "adaptive":
        return adaptive_closure(args)
    if args.mode == "family":
        return family_gfp(args)
    if args.mode == "family2":
        return family_gfp2(args)
    if args.mode == "cycle":
        return cycle_probe(args)
    if args.mode == "minfinals":
        return minfinals_probe(args)
    return closure_search(args)


if __name__ == "__main__":
    sys.exit(main())
