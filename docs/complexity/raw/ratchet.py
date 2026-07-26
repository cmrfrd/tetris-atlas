import itertools
# skyline simulator, width-4 window search for S/T ratchet cycle
PIECES = {
 'O': [[(0,0),(0,1),(1,0),(1,1)]],
 'I': [[(0,0),(0,1),(0,2),(0,3)],[(0,0),(1,0),(2,0),(3,0)]],
 'S': [[(1,0),(2,0),(0,1),(1,1)],[(0,1),(0,2),(1,0),(1,1)]],
 'Z': [[(0,0),(1,0),(1,1),(2,1)],[(0,0),(0,1),(1,1),(1,2)]],
 'T': [[(0,0),(1,0),(2,0),(1,1)],   # up nub
      [(1,0),(0,1),(1,1),(2,1)],   # down nub
      [(0,1),(1,0),(1,1),(1,2)],   # nub left
      [(0,0),(0,1),(0,2),(1,1)]],  # nub right
 'L': [[(0,0),(1,0),(2,0),(0,1)],[(0,0),(0,1),(0,2),(1,2)],[(2,0),(0,1),(1,1),(2,1)],[(0,0),(1,0),(1,1),(1,2)]],
 'J': [[(0,0),(1,0),(2,0),(2,1)],[(0,0),(1,0),(0,1),(0,2)],[(0,1),(1,1),(2,1),(0,0)],[(1,0),(1,1),(1,2),(0,2)]],
}
def flush_placements(h, piece):
    """all (rot,col,newh) placements of piece in window of width len(h) that create no holes"""
    W = len(h); out = []
    for ri,cells in enumerate(PIECES[piece]):
        pw = max(c for c,_ in cells)+1
        for col in range(W-pw+1):
            bot = {}
            for c,r in cells: bot[c] = min(bot.get(c,99), r)
            r0 = max(h[col+c]-bot[c] for c in bot)
            if all(r0+bot[c] == h[col+c] for c in bot):
                nh = list(h)
                for c,r in cells:
                    nh[col+c] = max(nh[col+c], r0+r+1)
                out.append((ri,col,tuple(nh)))
    return out

def norm(h):
    m = min(h); return tuple(x-m for x in h)

# search: profile A (width 4, offsets 0..6, min=0):
#  - S has exactly one flush placement -> B
#  - on A: no flush for T,Z,O,L,J  (I always flush per column, unavoidable)
#  - on B: exactly one flush T -> A+2 uniform;  no flush for S,Z,O,L,J
found = []
for A in itertools.product(range(7),repeat=4):
    if min(A)!=0: continue
    sp = flush_placements(A,'S')
    if len(sp)!=1: continue
    bad=False
    for p in ['T','Z','O','L','J']:
        if flush_placements(A,p): bad=True;break
    if bad: continue
    B = sp[0][2]
    tp = flush_placements(B,'T')
    if len(tp)!=1: continue
    for p in ['S','Z','O','L','J']:
        if flush_placements(B,p): bad=True;break
    if bad: continue
    A2 = tp[0][2]
    if norm(A2)==norm(A) and all(A2[i]-A[i]==2 for i in range(4)):
        found.append((A,sp[0],B,tp[0],A2))
print("strict S->B->T->A+2 cycles:", len(found))
for f in found[:10]: print(f)

# broader: 2-piece cycles A --S(unique)--> B --X(unique)--> A + k uniform, any X
print("\n--- relaxed: any partner X, still unique-flush + hostility ---")
res = {}
for A in itertools.product(range(7),repeat=4):
    if min(A)!=0: continue
    sp = flush_placements(A,'S')
    if len(sp)!=1: continue
    B = sp[0][2]
    for X in ['T','Z','O','L','J']:
        xp = flush_placements(B,X)
        if len(xp)!=1: continue
        A2 = xp[0][2]
        if norm(A2)==norm(A):
            k = A2[0]-A[0]
            if all(A2[i]-A[i]==k for i in range(4)):
                # hostility audit
                hostA = [p for p in ['T','Z','O','L','J'] if flush_placements(A,p)]
                hostB = [p for p in ['S','Z','O','L','J'] if p!=X and flush_placements(B,p)]
                res.setdefault(X,[]).append((A,B,k,hostA,hostB))
for X,v in res.items():
    print(X, len(v))
    for e in v[:5]: print("  ",e)
