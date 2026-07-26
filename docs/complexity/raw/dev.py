exec(open('/Users/cmrfrd/.claude/jobs/2257e482/tmp/ratchet.py').read().split('# search:')[0])
from collections import deque
A=(0,1,2,1); B=(0,1,4,3)
targets={norm(A),norm(B)}
def all_flush(h):
    out=[]
    for p in PIECES:
        for (ri,col,nh) in flush_placements(h,p):
            out.append((p,ri,col,norm(nh)))
    return out
# can state s return to targets within depth d via any flush placements?
import functools
@functools.lru_cache(maxsize=None)
def can_return(s,d):
    if s in targets: return True
    if d==0: return False
    for (_,_,_,ns) in all_flush(s):
        if max(ns)<=10 and can_return(ns,d-1): return True
    return False
# enumerate one-step deviations from A and B; intended: A--S(vert? report)-->B, B--Z-->A+2
print("From A:", )
for mv in all_flush(A):
    tag = "INTENDED" if (mv[0]=='S' and mv[3]==norm(B)) else ("RETURNS" if can_return(mv[3],6) else "strands")
    print("  ",mv, tag)
print("From B:")
for mv in all_flush(B):
    tag = "INTENDED" if (mv[0]=='Z' and mv[3]==norm(A)) else ("RETURNS" if can_return(mv[3],6) else "strands")
    print("  ",mv, tag)
