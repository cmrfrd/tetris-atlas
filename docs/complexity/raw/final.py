exec(open('/Users/cmrfrd/.claude/jobs/2257e482/tmp/ratchet.py').read().split('# search:')[0])
sols=[]
def dfs(h,seq,depth):
    if len(set(h))==1: sols.append((tuple(seq),h)); return
    if depth==0: return
    for p in PIECES:
        for r,c,nh in flush_placements(h,p):
            if max(nh)<=8: dfs(nh,seq+[(p,r,c)],depth-1)
dfs((3,0,1,0),[],3)
seen=set(); print("C1-family (3,0,1,0) terminal fixes:")
for seq,h in sols:
    k=tuple(sorted(p for p,_,_ in seq))
    if (k,h[0]) not in seen: seen.add((k,h[0])); print("  ",k,"-> flat",h[0],seq)
print("T-down on (5,4,5):", flush_placements((5,4,5),'T'))
# T-well pinwheel MV3: T rot2@c0, rot1@c2, rot3@c0, rot0@c1 from flat; audit foreign flush at each intermediate
# my rotation indexing differs from MV3's; instead exhaustively find a 4-T flush fill of 4x4 and audit its intermediates
tsols=[]
def tdfs(h,seq,depth):
    if depth==0:
        if len(set(h))==1: tsols.append((tuple(seq),h))
        return
    for r,c,nh in flush_placements(h,'T'):
        if max(nh)<=4: tdfs(nh,seq+[(r,c,nh)],depth-1)
tdfs((0,0,0,0),[],4)
print("4-T flush fills of 4x4:",len(tsols))
if tsols:
    seq,_=tsols[0]
    h=(0,0,0,0)
    for (r,c,nh) in seq:
        foreign={p:len(flush_placements(h,p)) for p in PIECES if p!='T' and flush_placements(h,p)}
        print("  profile",h,"T move",(r,c),"foreign flush counts:",foreign)
        h=nh
    print("  final",h)
# budget identity on a concrete instance: s=2,B=8,a=[3,3,2? invalid] use s=2,B=10,a=[3,3,4,4,3,3]: sum=20=sB ok, B/4=2.5<a<5 ok
import math
s=2; B=10; a=[3,3,4,4,3,3]; M=1  # M-scaling optional
ap=[M*x for x in a]; Bp=M*B
U=sum(ap)                     # unit bags
phase=12*s                    # 3 arm + 4*(3s-1) + 1 final close = 12s
fix=s+2                       # bucket fixes (1 bag each: T+? pieces) + 2 anvil bags
# gate: R/4 bags. R from O-well: 2*(N) <= R with N=U+phase+fix+R/4 -> R >= 4*(U+phase+fix)... solve R=4*(U+phase+fix) exactly? then N=U+phase+fix+R/4 = 2*(U+phase+fix); O-well prefill=R-2N=0 ok
core=U+phase+fix
R=4*core; N=core+R//4  # = 2*core
print("s,B,a:",s,B,a,"U,phase,fix,R,N:",U,phase,fix,R,N)
# per-bucket prefill: columns get 12 (ops) + 2*Bp (units) + fixpieces cells; final flat R
# fix from C1-family: use found fix (pieces add f0..f3 per column, total 8 or 12 cells)
# identity check: below-line capacity == prefill + 4*(pieces below line)
# pieces below line: all except gate-bag junk (6 per gate bag)
gate_bags=R//4
below_pieces=7*N - 6*gate_bags - 0  # gate I's are below line (gate column)
print("cells below line needed if identity holds:", 4*below_pieces, " (plus prefill = n*R)")
# widths: gate1 + s*(4+1) + lobby(2+1+1) + zwell(2+1+1) + iwell(1+1) + owell(2+1) + twell(4+1) + lwell(4+1) + jwell(4+1) + outer 2
n=1+1 + 5*s + 4+4+2+3+5+5+5 +1
print("n =",n,"  n*R =",n*R, " prefill = n*R - 4*below_pieces =", n*R-4*below_pieces, "(must be >=0 and shape-feasible)")
