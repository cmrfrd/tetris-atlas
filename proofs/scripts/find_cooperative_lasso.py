#!/usr/bin/env python3
"""Witness search for the cooperative lasso theorem (T1).

Finds a 35-piece placement sequence (5 bags, each a permutation of all 7
pieces) that takes the empty 10x20 board back to the empty board, never
topping out. The output is a Lean `List Placement` literal for
Proofs/Experiments/CooperativeLasso.lean, where it is re-certified from
scratch (`checkTable` + `native_decide`) — this script carries no trust.

Method: a with-HOLES bitboard beam search directed at the empty@35 goal.
Transient holes are ESSENTIAL: a hole-free (flush-only) 5-bag perfect clear
does not exist — a flush-only beam robustly stalls exactly one row (10 cells)
short at depth 35, across all height caps, widths, and seeds, matching the
repo's flush-closure-empty findings. Allowing a piece to rest on the highest
obstruction (leaving holes below, per the model's `dropOffset`) restores the
freedom needed to align empty-board with full-bag at placement 35.

Semantics mirror proofs/Proofs/Model exactly:
  - shapeUp tables transcribed from Proofs/Model/Piece.lean
  - dropOffset = sup over cells of (colHeight - up), colHeight = topmost+1
  - place = union; clearLines removes full rows and compacts down
  - bag: draw erases, refills to all 7 when emptied

A found sequence is re-verified by `cellset_verify`, a literal cell-set
simulation of place/clearLines matching the Lean model, before printing.
"""
import random
import sys
import time

COLS = 10
ROWS = 20
BAGS = 5
PIECES_PER_BAG = 7
TOTAL = BAGS * PIECES_PER_BAG  # 35

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

ALL_PIECES = ("O", "I", "S", "Z", "T", "L", "J")
FULL = frozenset(ALL_PIECES)


def candidate_rotations(piece):
    return {
        "O": (0,), "I": (0, 1), "S": (0, 1), "Z": (0, 1),
        "T": (0, 1, 2, 3), "L": (0, 1, 2, 3), "J": (0, 1, 2, 3),
    }[piece]


# Precomputed placements: (piece, rot, col, info) with
# info = list of (abs_col, u0, ups_tuple).
PLACEMENTS = []
for _p in ALL_PIECES:
    for _rot in candidate_rotations(_p):
        _prof = SHAPE_UP[(_p, _rot)]
        _cs = sorted(_prof)
        _width = max(_cs) + 1
        _info0 = [(c, _prof[c][0], tuple(_prof[c])) for c in _cs]
        for _col in range(COLS - _width + 1):
            PLACEMENTS.append(
                (_p, _rot, _col,
                 [(_col + c, u0, ups) for (c, u0, ups) in _info0]))


# ---------------------------------------------------------------------------
# With-holes bitboard model: board = tuple of 10 column bitmasks (bit r = filled
# at row r). Matches the Lean hard-drop (dropOffset via colHeight = topmost+1).
# ---------------------------------------------------------------------------

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


def _step(cols, info):
    """Apply one hard-drop placement (holes allowed) + line clears; None if lost."""
    hs = [c.bit_length() for c in cols]  # colHeight = topmost filled + 1
    d = 0
    for (ac, u0, _ups) in info:
        t = hs[ac] - u0
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
        return None
    full = newcols[0]
    for c in newcols[1:]:
        full &= c
    if full:
        newcols = [_pext(c, full) for c in newcols]
    return tuple(newcols)


def _cells(cols):
    return sum(bin(c).count("1") for c in cols)


def _heuristic(cols, rng):
    hs = [c.bit_length() for c in cols]
    cc = sum(bin(c).count("1") for c in cols)
    mx = max(hs)
    bump = sum(abs(hs[i] - hs[i + 1]) for i in range(COLS - 1))
    holes = sum(hs[j] - bin(cols[j]).count("1") for j in range(COLS))
    return cc * 1000 + holes * 200 + mx * 100 + bump * 10 + rng.random()


EMPTY = (0,) * COLS


def beam_search(width, seed):
    """Directed beam toward empty@35. Returns a placement list or None."""
    rng = random.Random(seed)
    beam = [(EMPTY, FULL, ())]
    best = 999
    for depth in range(TOTAL):
        last = depth == TOTAL - 1
        nxt = {}
        for cols, bag, path in beam:
            for (p, rot, _col, info) in PLACEMENTS:
                if p not in bag:
                    continue
                res = _step(cols, info)
                if res is None:
                    continue
                nbag = bag - {p}
                if not nbag:
                    nbag = FULL
                if last:
                    cc = _cells(res)
                    if cc == 0:
                        return list(path + ((p, rot, _col),))
                    if cc < best:
                        best = cc
                    continue
                key = (res, nbag)
                sc = _heuristic(res, rng)
                old = nxt.get(key)
                if old is None or sc < old[0]:
                    nxt[key] = (sc, path + ((p, rot, _col),))
        if last:
            return None
        if not nxt:
            return None
        beam = [(k[0], k[1], v[1])
                for k, v in sorted(nxt.items(), key=lambda kv: kv[1][0])[:width]]
    return None


# ---------------------------------------------------------------------------
# Full cell-set re-verification, mirroring Proofs/Model literally (no bitboard
# shortcut): colHeight, dropOffset, place, isFull, clearLines. This is the
# only correctness anchor the script offers; the real anchor is native_decide.
# ---------------------------------------------------------------------------

def cellset_verify(seq):
    board = frozenset()
    bag = set(ALL_PIECES)

    def col_height(b, j):
        rows = [r for (c, r) in b if c == j]
        return max(r + 1 for r in rows) if rows else 0

    for i, (piece, rot, col) in enumerate(seq):
        assert piece in bag, f"step {i}: {piece} not in bag {bag}"
        prof = SHAPE_UP[(piece, rot)]
        cells = [(c, u) for c, ups in prof.items() for u in ups]
        assert all(col + c < COLS for c, _ in cells), f"step {i}: invalid"
        d = max((max(col_height(board, col + c) - u, 0) for c, u in cells),
                default=0)
        dropped = {(col + c, d + u) for c, u in cells}
        assert not (dropped & board), f"step {i}: overlap"
        placed = board | dropped
        assert all(r < ROWS for (_, r) in placed), f"step {i}: lost"
        full = {r for r in {r for (_, r) in placed}
                if all((c, r) in placed for c in range(COLS))}
        board = frozenset(
            (c, r - len([fr for fr in full if fr < r]))
            for (c, r) in placed if r not in full)
        bag.discard(piece)
        if not bag:
            bag = set(ALL_PIECES)
    assert board == frozenset(), f"final board nonempty: {sorted(board)}"
    assert bag == set(ALL_PIECES), "final bag not full"
    return True


def emit_lean(seq):
    names = {"O": ".O", "I": ".I", "S": ".S", "Z": ".Z",
             "T": ".T", "L": ".L", "J": ".J"}
    lines = ["def lassoPlacements : List Placement := ["]
    for i, (piece, rot, col) in enumerate(seq):
        sep = "," if i + 1 < len(seq) else ""
        lines.append(f"  ⟨{names[piece]}, {rot}, {col}⟩{sep}")
    lines.append("]")
    return "\n".join(lines)


def main():
    start = time.time()
    schedule = [(3000, 1), (3000, 2), (8000, 3), (8000, 4),
                (20000, 5), (20000, 6), (60000, 7)]
    for width, seed in schedule:
        seq = beam_search(width, seed)
        el = time.time() - start
        if seq is not None:
            print(f"FOUND loop of {len(seq)} placements "
                  f"(beam {width}, seed {seed}, {el:.0f}s)", flush=True)
            cellset_verify(seq)
            print("cell-set re-verification PASSED (empty -> empty, full bag)",
                  flush=True)
            print(emit_lean(seq))
            return 0
        print(f"beam {width} seed {seed}: no perfect clear, {el:.0f}s",
              flush=True)
    print("no loop found")
    return 1


if __name__ == "__main__":
    sys.exit(main())
