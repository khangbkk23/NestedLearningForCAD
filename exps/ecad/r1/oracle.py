"""Independent CPU scalar specification; no torch or vectorized distance reuse."""
import math


def distance(a, b):
    return math.sqrt(math.fsum((float(x)-float(y))**2 for x,y in zip(a,b)))


def greedy(rows, serials, n, base):
    chosen = []
    while len(chosen) < n:
        candidates = [i for i in range(len(rows)) if i not in chosen]
        anchors = list(base) + [rows[j] for j in chosen]
        def key(i):
            return (-min((distance(rows[i],b) for b in anchors),default=math.inf), serials[i])
        chosen.append(min(candidates,key=key))
    return chosen


def consolidate(rows, serials, k, q):
    if len(rows)<=k:
        return sorted(range(len(rows)),key=serials.__getitem__)
    radii = [min(distance(x,y) for j,y in enumerate(rows) if j!=i) for i,x in enumerate(rows)]
    kept = sorted(range(len(rows)),key=lambda i:(-radii[i],serials[i]))[:math.floor(q*k)]
    rest = [i for i in range(len(rows)) if i not in kept]
    selected = greedy([rows[i] for i in rest],[serials[i] for i in rest],k-len(kept),[rows[i] for i in kept])
    return sorted(kept+[rest[i] for i in selected],key=serials.__getitem__)


def update(bank, serials, incoming, event, k, n, q):
    ids = list(range(event*len(incoming),(event+1)*len(incoming)))
    chosen = greedy(incoming,ids,n,bank)
    union, all_ids = bank+[incoming[i] for i in chosen], serials+[ids[i] for i in chosen]
    kept = consolidate(union,all_ids,k,q)
    return [union[i] for i in kept],[all_ids[i] for i in kept]
