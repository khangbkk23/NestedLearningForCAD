# exps/calm01/storage.py
"""Support-bank storage, exact incremental nearest-other graph and category scale.

Rows live in fixed slots; ``owner < 0`` marks a free slot. All selection distances are
computed on the *stored* (decoded) vectors, so selection sees exactly what scoring sees.
"""
from __future__ import annotations

import torch

from exps.task_p.p0_normal_support import distances as fp64_distances, require

STORAGES = ("fp32", "u8")


def cadic_fp32_distances(x, y):
    """Distance kernel of ``models/cadic_patch_coreset_v1.py`` (FP32, no matmul)."""
    return torch.cdist(x.float(), y.float(), p=2, compute_mode="donot_use_mm_for_euclid_dist")


DISTANCES = {"p0_fp64": fp64_distances, "cadic_fp32": cadic_fp32_distances}


def quantize_rows(x):
    """Per-row affine UINT8, identical to ``exps/csg01/quantize.py`` (ties-to-even rint)."""
    x = x.float()
    xmin = x.min(1).values
    step = ((x.max(1).values.double() - xmin.double()) / 255.0).float()
    q = torch.zeros(x.shape, dtype=torch.uint8)
    live = step > 0
    if bool(live.any()):
        z = (x[live].double() - xmin[live].double()[:, None]) / step[live].double()[:, None]
        q[live] = torch.round(z).clamp_(0, 255).to(torch.uint8)
    decoded = (xmin.double()[:, None] + step.double()[:, None] * q.double()).float()
    return {"q": q, "xmin": xmin, "step": step, "decoded": decoded}


class Bank:
    """Fixed-capacity bank. ``x`` is the decoded working copy; ``q/xmin/step`` the UINT8 state.

    ``x64``/``n2`` cache the FP64 rows and squared norms (derived, not persistent) so that a query
    against the whole bank costs one matmul instead of a bank-wide conversion per call.
    """

    def __init__(self, capacity, dim=768, storage="fp32", kernel="p0_fp64", cache_device="cpu"):
        require(storage in STORAGES and kernel in DISTANCES and capacity >= 2, "bad bank configuration")
        self.capacity, self.dim, self.storage, self.kernel = int(capacity), int(dim), storage, kernel
        self.cache_device = torch.device(cache_device)   # FP64 cache only; all decisions stay on CPU
        self.x = torch.zeros(self.capacity, self.dim, dtype=torch.float32)
        self.x64 = torch.zeros(self.capacity, self.dim, dtype=torch.float64, device=self.cache_device)
        self.n2 = torch.zeros(self.capacity, dtype=torch.float64, device=self.cache_device)
        self.owner = torch.full((self.capacity,), -1, dtype=torch.int64)
        self.ids = torch.full((self.capacity,), -1, dtype=torch.int64)
        if storage == "u8":
            self.q = torch.zeros(self.capacity, self.dim, dtype=torch.uint8)
            self.xmin = torch.zeros(self.capacity, dtype=torch.float32)
            self.step = torch.zeros(self.capacity, dtype=torch.float32)

    def encode(self, x):
        require(x.ndim == 2 and x.shape[1] == self.dim and bool(torch.isfinite(x).all()), "bad rows")
        return quantize_rows(x) if self.storage == "u8" else {"decoded": x.float().clone()}

    def put(self, slot, encoded, index, owner, serial):
        self.x[slot] = encoded["decoded"][index]
        row = self.x[slot].double().to(self.cache_device)
        self.x64[slot] = row; self.n2[slot] = row.square().sum()
        if self.storage == "u8":
            self.q[slot] = encoded["q"][index]
            self.xmin[slot] = encoded["xmin"][index]; self.step[slot] = encoded["step"][index]
        self.owner[slot] = int(owner); self.ids[slot] = int(serial)

    def clear(self, slot):
        self.owner[slot] = -1; self.ids[slot] = -1

    def dist_to(self, q, cols=None):
        """FP64 distances ``[m, capacity]`` (or ``[m, len(cols)]``); free slots are +inf.

        ``p0_fp64`` is the norm/dot formula of ``exps/task_p/p0_normal_support.distances`` with the
        same cancellation repair; ``cadic_fp32`` is the CADIC model's kernel (parity checks only).
        """
        full = cols is None                              # full bank: no gathered copy
        idx = None if full else torch.as_tensor(cols).flatten()
        if self.kernel == "cadic_fp32":
            d = cadic_fp32_distances(q, self.x if full else self.x[idx]).double()
        else:
            a = q.double().to(self.cache_device)
            y, n2 = (self.x64, self.n2) if full else (self.x64[idx.to(self.cache_device)], self.n2[idx.to(self.cache_device)])
            norms = a.square().sum(1)[:, None] + n2[None]
            sq = (norms - 2 * (a @ y.T)).clamp_min_(0)
            r, c = (sq <= 64 * torch.finfo(torch.float64).eps * norms).nonzero(as_tuple=True)
            if len(r):
                sq[r, c] = (a[r] - y[c]).square().sum(1)
            d = sq.sqrt_().cpu()
        d[:, ~(self.occupied if full else self.occupied[idx])] = torch.inf
        return d

    @property
    def occupied(self):
        return self.owner >= 0

    def free_slot(self):
        free = (~self.occupied).nonzero()
        return int(free[0]) if len(free) else None

    def export(self):
        keep = self.occupied
        return {"bank": self.x[keep].clone(), "ids": self.ids[keep].clone(),
                "owners": self.owner[keep].to(torch.int8).clone()}


class NNGraph:
    """Exact nearest-other row per slot; ``same_owner`` restricts neighbours to the row's owner.

    Ties resolve to the lowest slot index, as in ``exps/task_p/p0_normal_support.Support._put``.
    A same-owner graph only ever reads the rows of the owner concerned (exact, and ~C times cheaper).
    Derived cache: recomputable from the bank, therefore not charged as persistent state.
    """

    def __init__(self, bank, same_owner):
        self.bank, self.same_owner = bank, bool(same_owner)
        self.d = torch.full((bank.capacity,), torch.inf, dtype=torch.float64)
        self.i = torch.full((bank.capacity,), -1, dtype=torch.int64)

    def _groups(self, rows):
        """Yield (rows, candidate columns); columns ``None`` means the whole bank."""
        if not self.same_owner:
            yield rows, None; return
        own = self.bank.owner[rows]
        for o in torch.unique(own).tolist():
            yield rows[own == o], (self.bank.occupied & (self.bank.owner == o)).nonzero().flatten()

    def _block(self, rows, cols):
        D = self.bank.dist_to(self.bank.x[rows], cols)       # free slots are already +inf
        ci = torch.arange(self.bank.capacity) if cols is None else cols
        D[rows[:, None] == ci[None]] = torch.inf             # never one's own neighbour
        return D, ci

    def _recompute(self, rows):
        rows = torch.as_tensor(rows, dtype=torch.int64)
        if not len(rows):
            return
        for sub, cols in self._groups(rows):
            D, ci = self._block(sub, cols)
            v, k = D.min(1)                                  # first minimum: lowest slot on ties
            self.d[sub] = v; self.i[sub] = torch.where(torch.isinf(v), torch.full_like(k, -1), ci[k])

    def changed(self, slot):
        """Call after ``slot`` was written (or cleared)."""
        stale = ((self.i == slot) & self.bank.occupied).nonzero().flatten().tolist()
        if not bool(self.bank.occupied[slot]):
            self.d[slot], self.i[slot] = torch.inf, -1
            self._recompute(stale); return
        for sub, cols in self._groups(torch.tensor([slot])):
            D, ci = self._block(sub, cols)
            d = D[0]
            better = torch.isfinite(d) & ((d < self.d[ci]) | ((d == self.d[ci]) & (slot < self.i[ci])))
            self.d[ci[better]] = d[better]; self.i[ci[better]] = slot
            k = int(d.argmin())                              # the slot's own row: reuse d
            self.d[slot], self.i[slot] = (float(d[k]), int(ci[k])) if torch.isfinite(d[k]) else (torch.inf, -1)
        self._recompute(sorted(set(stale) - {slot}))

    def inserted_many(self, slots):
        """Batch form of ``changed`` for slots that were free and are now written.

        The exact graph is a function of the final bank (lowest-index ties), so one batched pass
        equals the sequence of single-slot updates.
        """
        if not len(slots):
            return
        s = torch.as_tensor(sorted(slots), dtype=torch.int64)
        for sub, cols in self._groups(s):
            D, ci = self._block(sub, cols)
            v, k = D.min(0)                                  # best new neighbour for every column
            k = sub[k]
            better = torch.isfinite(v) & ((v < self.d[ci]) | ((v == self.d[ci]) & (k < self.i[ci])))
            self.d[ci[better]] = v[better]; self.i[ci[better]] = k[better]
        self._recompute(s)

    def rebuild(self):
        self._recompute(self.bank.occupied.nonzero().flatten())


def within_image_scale(x, dist=fp64_distances):
    """Median distance to the nearest non-adjacent patch of the same image (Chebyshev > 1).

    A non-square patch count (synthetic fixtures) excludes only the patch itself.
    """
    n = x.shape[0]
    require(n >= 2, "scale needs at least two patches")
    side = int(round(n ** .5))
    if side * side == n and side >= 3:
        r, c = torch.arange(n) // side, torch.arange(n) % side
        near = ((r[:, None] - r[None]).abs() <= 1) & ((c[:, None] - c[None]).abs() <= 1)
    else:
        near = torch.eye(n, dtype=torch.bool)
    d = dist(x, x).double().masked_fill(near, torch.inf)
    return float(d.min(1).values.median())


class CategoryScales:
    """Causal per-category distance scale ``s_k``.

    Image 1 of a category: provisional within-image scale. Image 2: median distance of its patches
    to the category's stored rows, before any update. Frozen afterwards (one float per category).
    """

    def __init__(self, dist=fp64_distances):
        self.dist, self.s, self.provisional, self.images = dist, {}, {}, {}

    def observe(self, c, x, bank):
        self.images[c] = self.images.get(c, 0) + 1
        if c not in self.s:
            self.s[c], self.provisional[c] = within_image_scale(x, self.dist), True
        elif self.provisional[c]:
            rows = (bank.occupied & (bank.owner == c)).nonzero().flatten()
            if len(rows):
                self.s[c] = float(bank.dist_to(x, rows).min(1).values.median())
                self.provisional[c] = False
        require(self.s[c] > 0 and torch.isfinite(torch.tensor(self.s[c])), "degenerate category scale")
        return self.s[c]
