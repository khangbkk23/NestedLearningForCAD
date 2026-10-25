# exps/calm01/closest_pair.py
"""Closest-pair (k-center) controls: CADIC rule, scale-normalised rule and equal per-category quota.

``cadic``  : CADIC Eq. (1)-(6) as implemented in ``models/cadic_patch_coreset_v1.py``: fill in input
             order; then repeatedly insert the candidate farthest from the bank in place of the first
             member ``c1`` of the lexicographically first closest pair, while ``d_max > d_pair``.
``scaled`` : the same rule with keys divided by the owner's causal scale ``s_k`` (one fixed change).
``quota``  : one CADIC memory per category with equal quota ``K // C_t``; old categories are shrunk
             by closest-pair removal when a new category starts; the current one may use free slots.
"""
from __future__ import annotations

import torch

from exps.task_p.p0_normal_support import require
from exps.calm01.storage import DISTANCES, Bank, CategoryScales, NNGraph

MODES = ("cadic", "scaled", "quota")


class ClosestPairMemory:
    def __init__(self, capacity, mode="cadic", storage="fp32", dim=768, distance="p0_fp64", cache_device="cpu"):
        require(mode in MODES and distance in DISTANCES, "unknown closest-pair configuration")
        self.mode, self.distance = mode, distance
        self.dist = DISTANCES[distance]
        self.bank = Bank(capacity, dim, storage, kernel=distance, cache_device=cache_device)
        self.graph = NNGraph(self.bank, same_owner=mode == "quota")
        self.scales = CategoryScales(self.dist)
        self.categories, self.current = [], None

    @property
    def capacity(self):
        return self.bank.capacity

    def begin_category(self, c):
        c = int(c)
        if c not in self.categories:
            self.categories.append(c)
            if self.mode == "quota":
                self._shrink(self.capacity // len(self.categories), keep=c)
        self.current = c

    def end_category(self):
        self.current = None

    def count(self, c):
        return int((self.bank.owner == c).sum())

    def _victim(self, among):
        """Return (key, slot) of the row to replace among the boolean slot mask ``among``."""
        if not bool(among.any()):
            return torch.inf, None
        key = self.graph.d.clone()
        if self.mode == "scaled":
            for k in self.categories:
                key[self.bank.owner == k] /= self.scales.s[k]
        key[~among] = torch.inf
        best = float(key.min())
        if best == torch.inf:
            return torch.inf, None
        tied = (key == best).nonzero().flatten()
        if self.mode == "scaled":
            return best, int(tied[0])
        # CADIC tie policy: lexicographically first unordered pair (i<j); replace its i member.
        pairs = sorted((min(int(j), int(self.graph.i[j])), max(int(j), int(self.graph.i[j]))) for j in tied)
        return best, pairs[0][0]

    def _shrink(self, quota, keep):
        for k in self.categories:
            if k == keep:
                continue
            while self.count(k) > quota:
                _, slot = self._victim(self.bank.owner == k)
                require(slot is not None, "quota shrink without a pair")
                self.bank.clear(slot); self.graph.changed(slot)

    def update(self, x, serials):
        require(x.ndim == 2 and len(serials) == len(x), "bad proposal batch")
        c = self.current if self.current is not None else 0
        require(self.mode == "cadic" or self.current is not None, "begin_category required")
        enc = self.bank.encode(x)
        if self.mode == "scaled":
            self.scales.observe(c, x, self.bank)
        stats = {"inserted": 0, "replaced": 0, "rejected": 0}
        pool = list(range(len(x)))
        free = (~self.bank.occupied).nonzero().flatten().tolist()
        filled = free[:len(pool)]                       # fill in input order, as CADIC
        for slot, r in zip(filled, pool):
            self.bank.put(slot, enc, r, c, serials[r])
        self.graph.inserted_many(filled)
        pool = pool[len(filled):]; stats["inserted"] += len(filled)
        if not pool:
            return stats
        cols = (self.bank.owner == c) if self.mode == "quota" else self.bank.occupied
        DI = self.bank.dist_to(x).masked_fill_(~cols[None], torch.inf)
        alive = torch.zeros(len(x), dtype=torch.bool); alive[pool] = True
        while bool(alive.any()):
            novelty = DI.min(1).values.masked_fill(~alive, -torch.inf)
            f = int(novelty.argmax())
            among = (self.bank.owner == c) if self.mode == "quota" else self.bank.occupied
            key, slot = self._victim(among)
            cand = float(novelty[f]) / (self.scales.s[c] if self.mode == "scaled" else 1.0)
            if slot is None or not cand > key:
                stats["rejected"] += int(alive.sum()); break
            self.bank.put(slot, enc, f, c, serials[f]); self.graph.changed(slot)
            DI[:, slot] = self.bank.dist_to(x, [slot])[:, 0]
            alive[f] = False; stats["replaced"] += 1
        return stats

    def export(self):
        return self.bank.export()
