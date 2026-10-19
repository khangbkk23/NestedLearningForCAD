"""Paper-defined per-image project adaptation, NOT released-pipeline parity.

Reference: jungyg/ContCore@0579a727e2320ea93c8a657e02600080837ee2a0.
Distinct row semantics include duplicate-valued rows. No task or category input.
"""
import math
import torch
from .distance import Euclidean, global_nn


def farthest_indices(x, serials, n, base, kernel=None):
    """Select INCOMING indices, conditioned on the complete supplied old base."""
    kernel = kernel or Euclidean()
    if not 0 <= n <= len(x):
        raise ValueError("invalid selection count")
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=x.device)
    order = torch.argsort(serials, stable=True)
    xx = x[order]
    pair = kernel.matrix(xx, xx)
    nearest = kernel.nearest(xx, base)
    used = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    selected = []
    for _ in range(n):
        ix = nearest.masked_fill(used, -torch.inf).argmax()
        selected.append(ix)
        used[ix] = True
        nearest = torch.minimum(nearest, pair[:, ix])
    return order[torch.stack(selected)]


def consolidate_indices(union, serials, k, q=.75, kernel=None):
    kernel = kernel or Euclidean()
    if k < 1 or not 0 <= q <= 1:
        raise ValueError("invalid consolidation parameters")
    order = torch.argsort(serials, stable=True)
    if len(union) <= k:
        return order
    x = union[order]
    pair = kernel.matrix(x, x)  # update-local workspace; released on return
    pair.fill_diagonal_(torch.inf)
    radii = pair.min(1).values
    pair.fill_diagonal_(0)
    a = math.floor(q * k)
    kept = torch.argsort(radii, descending=True, stable=True)[:a]
    used = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    used[kept] = True
    nearest = pair[:, kept].min(1).values if a else torch.full_like(radii, torch.inf)
    selected = list(kept.unbind())
    for _ in range(k-a):
        ix = nearest.masked_fill(used, -torch.inf).argmax()
        selected.append(ix)
        used[ix] = True
        nearest = torch.minimum(nearest, pair[:, ix])
    # Committed storage order is always serial order, independent of geometry.
    return order[torch.stack(selected).sort().values]


class CausalContCore:
    def __init__(self, capacity=2500, dim=768, patches=784, candidates=78, q=.75,
                 device="cpu", chunk=256):
        if capacity < 1 or dim < 1 or patches < 1 or not 0 < candidates <= patches or not 0 <= q <= 1:
            raise ValueError("invalid configuration")
        self.capacity, self.dim, self.patches = capacity, dim, patches
        self.candidates, self.q, self.chunk = candidates, q, chunk
        self.bank = torch.empty((0, dim), dtype=torch.float32, device=device)
        self.serials = torch.empty(0, dtype=torch.int64, device=device)
        self.next_event = 0

    @torch.no_grad()
    def update(self, features, *, event, work=None):
        # Event is an opaque contiguous counter; repeated/skipped/future calls fail
        # before mutation. The caller/arrival service enforces source identity.
        if event != self.next_event:
            raise ValueError("out-of-order or replayed image")
        if features.shape != (self.patches, self.dim) or features.dtype != torch.float32:
            raise ValueError("wrong full-grid representation")
        if features.device != self.bank.device or not bool(torch.isfinite(features).all()):
            raise ValueError("invalid feature device/values")
        if work is not None:
            work.host_scalar_reads += int(features.is_cuda)  # finite validation
        ids = torch.arange(event*self.patches, (event+1)*self.patches,
                           device=features.device, dtype=torch.int64)
        kernel = Euclidean(self.chunk, work=work)
        ix = farthest_indices(features, ids, self.candidates, self.bank, kernel)
        union = torch.cat((self.bank, features[ix]))
        serials = torch.cat((self.serials, ids[ix]))
        keep = consolidate_indices(union, serials, self.capacity, self.q, kernel)
        old_count = len(self.bank)
        new_bank, new_serials = union[keep].clone(), serials[keep].clone()
        old_survivors = (new_serials < event*self.patches).sum()
        self.bank, self.serials = new_bank, new_serials
        self.next_event += 1
        return {"observed": self.patches, "candidates": self.candidates,
                "occupied": len(new_bank), "evicted_old": old_count-old_survivors}

    def score(self, features):
        return global_nn(features, self.bank, Euclidean(self.chunk))

    def state_dict(self):
        return {"schema": "ecad_r1_contcore_v1", "capacity": self.capacity, "dim": self.dim,
                "patches": self.patches, "candidates": self.candidates, "q": self.q,
                "chunk": self.chunk, "next_event": self.next_event,
                "bank": self.bank.clone(), "serials": self.serials.clone()}

    @classmethod
    def from_state_dict(cls, state, device="cpu"):
        if state["schema"] != "ecad_r1_contcore_v1":
            raise ValueError("unsupported checkpoint")
        obj = cls(**{k:state[k] for k in ("capacity", "dim", "patches", "candidates", "q", "chunk")}, device=device)
        bank, ids, event = state["bank"], state["serials"], state["next_event"]
        if (bank.dtype != torch.float32 or bank.shape != (len(ids), obj.dim) or len(ids)>obj.capacity
            or ids.dtype != torch.int64 or ids.ndim!=1 or not isinstance(event,int) or event<0
            or not bool(torch.isfinite(bank).all()) or len(ids)!=min(event*obj.candidates,obj.capacity)
            or bool(((ids<0)|(ids>=event*obj.patches)).any())
            or (len(ids)>1 and not bool((ids[1:]>ids[:-1]).all()))):
            raise ValueError("invalid checkpoint invariants")
        if len(ids) and bool((torch.unique(ids//obj.patches,return_counts=True)[1]>obj.candidates).any()):
            raise ValueError("checkpoint exceeds per-image candidate allowance")
        obj.bank, obj.serials, obj.next_event = bank.to(device).clone(), ids.to(device).clone(), event
        return obj
