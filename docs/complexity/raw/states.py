exec(open('/Users/cmrfrd/.claude/jobs/2257e482/tmp/ratchet.py').read().split('# search:')[0])
A=(0,1,2,1)
def audit(h,label):
    out={}
    for p in PIECES:
        fp=flush_placements(h,p)
        if fp: out[p]=[(r,c,norm(n)) for r,c,n in fp]
    print(label,h,"flush:",{p:len(v) for p,v in out.items()}, {p:v for p,v in out.items() if p in 'SZ'})
# closed chain: I@c0 then I@c1 then I@c2 then I@c3 -> A+4?
h=list(A)
for c,lab in [(0,'C1'),(1,'C2'),(2,'C3'),(3,'A+4?')]:
    h[c]+=4
    audit(tuple(h),lab)
print("A+4 check:", tuple(h), "== A+4:", tuple(h)==tuple(x+4 for x in A))
# hostility of C1,C2,C3 to S and Z specifically:
for st in [(4,1,2,1),(4,5,2,1),(4,5,6,1)]:
    s=len(flush_placements(st,'S')); z=len(flush_placements(st,'Z'))
    print(st,"S-flush:",s,"Z-flush:",z)
# terminal fix search: from A-family profile (x,x+1,x+2,x+1) reach uniform flat using <=3 pieces
from itertools import product
import sys
start=(0,1,2,1)
def dfs(h,seq,depth):
    if len(set(h))==1:
        sols.append((list(seq),h)); return
    if depth==0: return
    for p in PIECES:
        for r,c,nh in flush_placements(h,p):
            dfs(nh,seq+[(p,r,c)],depth-1)
sols=[]
dfs(start,[],3)
best={}
for seq,h in sols:
    k=tuple(sorted(p for p,_,_ in seq))
    best.setdefault((k,h[0]),seq)
for kk,v in sorted(best.items()): print("terminal fix:",kk,v)
