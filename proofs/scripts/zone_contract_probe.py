#!/usr/bin/env python3
"""Per-zone contract probe for the rely-guarantee decomposition of Tetris.

The composition idea: split the board into the STZ zone (4 cols), the LJO
zone (4 cols) and the MID drain pair (2 cols). Each zone plays only its own
columns; the coupling is the CLEAR SERVICE — rows the environment removes
(a row clears globally when every zone has it full, so from one zone's view
clears arrive as an external service with some rate and discipline).

This probe answers, EXACTLY (adaptive AND-OR closure over the zone's own
state space, which is tiny), the per-zone contract questions:

  For zone Z with piece set P_Z and width w: does a nonempty closed safe
  band exist under height cap H, hole budget D, and clear service "at the
  end of each zone-bag, the bottom s zone-full rows are removed"?

A zone-bag = one arrival of each piece in P_Z, adversarial order. The
service parameter s is the zone's ASSUMPTION about the environment; the
zone's fill discipline (zone-full rows at the bottom) is its GUARANTEE
(rows can only clear globally where the zone is full). A closed band at
(H, D, s) = the zone contract is satisfiable; the closure set IS the band.

Semantics mirror proofs/Proofs/Model (same SHAPE_UP; hard drop; the
service's row removal mirrors clearLines restricted to the zone).
"""
import argparse
import sys
import time
from itertools import permutations

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
ROTS = {"O": (0,), "I": (0, 1), "S": (0, 1), "Z": (0, 1),
        "T": (0, 1, 2, 3), "L": (0, 1, 2, 3), "J": (0, 1, 2, 3)}


def zone_placements(pieces, width):
    out = {}
    for p in pieces:
        lst = []
        for rot in ROTS[p]:
            prof = SHAPE_UP[(p, rot)]
            w = max(prof) + 1
            if w > width:
                continue
            for col in range(width - w + 1):
                lst.append((rot, col,
                            tuple((col + c, prof[c][0], tuple(prof[c]))
                                  for c in sorted(prof))))
        out[p] = lst
    return out


def _pext(x, mask):
    res, outpos = 0, 0
    hi = max(x.bit_length(), mask.bit_length())
    for r in range(hi):
        if not (mask >> r) & 1:
            if (x >> r) & 1:
                res |= 1 << outpos
            outpos += 1
    return res


def step_place(cols, info, hcap):
    """Hard drop in the zone; NO clears here (clears are the service)."""
    d = 0
    for (ac, u0, _u) in info:
        t = cols[ac].bit_length() - u0
        if t > d:
            d = t
    nc = list(cols)
    for (ac, _u0, ups) in info:
        for u in ups:
            nc[ac] |= 1 << (d + u)
    if max(c.bit_length() for c in nc) > hcap:
        return None
    return tuple(nc)


def service(cols, s):
    """End-of-bag clear service: remove the bottom s' zone-full rows,
    s' = min(s, number of full rows). Mirrors clearLines restricted to
    the zone (only rows the zone has FULL can ever clear globally)."""
    full = cols[0]
    for c in cols[1:]:
        full &= c
    # bottom-most min(s, popcount) full rows
    mask = 0
    cnt = 0
    r = 0
    while cnt < s and (full >> r):
        while not (full >> r) & 1:
            r += 1
            if not (full >> r):
                break
        if (full >> r) & 1:
            mask |= 1 << r
            cnt += 1
            r += 1
    if mask == 0:
        return cols, 0
    return tuple(_pext(c, mask) for c in cols), cnt


def holes(cols):
    return sum(c.bit_length() - bin(c).count("1") for c in cols)


def solve_zone(pieces, width, hcap, dcap, s, branch, budget):
    """Adaptive AND-OR closure of the zone game from the empty zone.
    State = zone board at bag boundary. Per bag: all |P|! orders (AND over
    next piece at each node), placements chosen adaptively (OR), then the
    service fires. Returns (closed?, band size, mid states, deaths)."""
    PL = zone_placements(pieces, width)
    t0 = time.time()

    def bag_solve(B, accept):
        memo = {}

        def rec(board, remaining):
            key = (board, remaining)
            if key in memo:
                return memo[key]
            if not remaining:
                nb, _cleared = service(board, s)
                ok = accept(nb)
                memo[key] = ok
                return ok
            ok = True
            for i, p in enumerate(pieces):
                if not (remaining >> i) & 1:
                    continue
                found = False
                cands = []
                for (rot, col, info) in PL[p]:
                    nb = step_place(board, info, hcap)
                    if nb is None:
                        continue
                    cands.append((holes(nb) * 100
                                  + max(c.bit_length() for c in nb), nb))
                cands.sort(key=lambda x: x[0])
                for (_sc, nb) in cands[:branch]:
                    if rec(nb, remaining & ~(1 << i)):
                        found = True
                        break
                if not found:
                    ok = False
                    break
            memo[key] = ok
            return ok

        full = (1 << len(pieces)) - 1
        okr = rec(B, full)
        if not okr:
            return None
        # extract finals under the found policy
        finals = set()
        stack = [(B, full)]
        seen = {(B, full)}
        choice = {}

        def rec2(board, remaining):
            if not remaining:
                nb, _c = service(board, s)
                finals.add(nb)
                return
            for i, p in enumerate(pieces):
                if not (remaining >> i) & 1:
                    continue
                cands = []
                for (rot, col, info) in PL[p]:
                    nb = step_place(board, info, hcap)
                    if nb is None:
                        continue
                    cands.append((holes(nb) * 100
                                  + max(c.bit_length() for c in nb), nb))
                cands.sort(key=lambda x: x[0])
                for (_sc, nb) in cands[:branch]:
                    if memo.get((nb, remaining & ~(1 << i))):
                        k = (nb, remaining & ~(1 << i))
                        if k not in seen:
                            seen.add(k)
                            rec2(nb, remaining & ~(1 << i))
                        break
            return

        rec2(B, full)
        return finals, len(memo)

    EMPTYZ = (0,) * width
    solved = {}
    bad = set()
    frontier = [EMPTYZ]
    queued = {EMPTYZ}
    mid_states = 0
    while frontier:
        if time.time() - t0 > budget:
            return ("timeout", len(solved), mid_states, len(bad))
        B = frontier.pop(0)
        queued.discard(B)
        if B in solved or B in bad:
            continue
        known = (set(solved) | queued | {B}) - bad

        def acc(nb, _k=known):
            if nb in _k:
                return True
            if nb in bad:
                return False
            return (holes(nb) <= dcap
                    and max(c.bit_length() for c in nb) <= hcap)

        res = bag_solve(B, acc)
        if res is None:
            bad.add(B)
            requeue = [x for x, fs in solved.items() if B in fs]
            for x in requeue:
                del solved[x]
                if x not in queued:
                    frontier.append(x)
                    queued.add(x)
            if B == EMPTYZ:
                return ("empty-dead", len(solved), mid_states, len(bad))
            continue
        finals, memo_n = res
        mid_states += memo_n
        solved[B] = finals
        for f in finals:
            if f not in solved and f not in queued and f not in bad:
                frontier.append(f)
                queued.add(f)
    # verify closure
    for B, fs in solved.items():
        assert all(f in solved for f in fs), "escape"
    return ("CLOSED", len(solved), mid_states, len(bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--branch", type=int, default=6)
    ap.add_argument("--config", type=str, default="base",
                    choices=("base", "redesign"))
    args = ap.parse_args()
    if args.config == "base":
        grid = [
            ("STZ", ("S", "T", "Z"), 4, (6, 8, 10), (0, 1, 2), (3, 4)),
            ("LJO", ("L", "J", "O"), 4, (6, 8, 10), (0, 1, 2), (3, 4)),
        ]
    else:
        grid = [
            # piece reassignment: S/Z alone (valley), T joins the right
            ("SZ", ("S", "Z"), 4, (6, 8, 10), (0, 1, 2), (2, 3)),
            ("TLJO", ("T", "L", "J", "O"), 4, (8, 10, 12), (1, 2, 3), (4, 5)),
            # width reallocation: STZ gets 5, LJO drops to 3
            ("STZ5", ("S", "T", "Z"), 5, (8, 10, 12), (1, 2, 3), (2, 3)),
            ("LJO3", ("L", "J", "O"), 3, (8, 10, 12), (1, 2), (4,)),
            # STZ 4-wide at relaxed caps
            ("STZ+", ("S", "T", "Z"), 4, (12, 14), (3, 4), (3,)),
        ]
    print(f"{'zone':5} {'H':>2} {'D':>2} {'s':>2} -> result (band, mid, bad)")
    for (name, pieces, width, hs, ds, ss) in grid:
        for hcap in hs:
            for dcap in ds:
                for s in ss:
                    r, band, mid, badn = solve_zone(
                        pieces, width, hcap, dcap, s,
                        args.branch, args.budget)
                    print(f"{name:5} {hcap:>2} {dcap:>2} {s:>2} -> "
                          f"{r} (band={band}, mid={mid}, bad={badn})",
                          flush=True)


if __name__ == "__main__":
    sys.exit(main())
