# exps/calm01/calm.py
"""CALM-01: Capacity-Anchored Layered Memory, retention-only (no certificate, raw-NN scoring).

Fast level (event clock, one call per arriving normal image of the current category ``c``):
greedy exact-swap descent on

    L(S) = sum_k w_k K_k(S) + w_c sum_{z in Z} d_c(z, S),     w_k = p_k / (s_k N_k)

where row ``j`` carries observed normal mass ``mu_j`` and transport cost ``kappa_j``; ``K_k`` is the
sum of ``kappa`` over rows owned by ``k``; ``N_k`` the category's observed mass (current image
included for ``c``); ``d_c`` the distance to the nearest row owned by ``c``. Evicting ``j`` moves
``(mu_j, kappa_j)`` to its nearest remaining same-category row ``j'`` and adds ``mu_j d(m_j, m_j')``.

Slow level (boundary clock): ``p_k <- clip(p_k exp(eta (Delta_k - mean Delta)), 1/4, 4)``, mean 1,
with ``Delta_k = J_k - J_k^acq`` and ``J_k = K_k / (N_k s_k)``. A multiplicative-weights feedback
heuristic toward ``min max_k Delta_k``; no convergence or KKT guarantee is claimed.
"""
from __future__ import annotations

import math

import torch

from exps.task_p.p0_normal_support import require
from exps.calm01.storage import DISTANCES, Bank, CategoryScales, NNGraph

CLOCKS = ("boundary", "image")


class CALMMemory:
    def __init__(self, capacity, storage="u8", eta=0.0, clock="boundary", eta_image=0.0,
                 shortlist=16, floor=1, price_bounds=(0.25, 4.0), dim=768, distance="p0_fp64",
                 cache_device="cpu"):
        require(clock in CLOCKS and eta >= 0 and eta_image >= 0 and shortlist >= 1 and floor >= 1,
                "bad CALM configuration")
        self.dist = DISTANCES[distance]
        self.bank = Bank(capacity, dim, storage, kernel=distance, cache_device=cache_device)
        self.graph = NNGraph(self.bank, same_owner=True)
        self.scales = CategoryScales(self.dist)
        self.mu = torch.zeros(capacity, dtype=torch.float64)
        self.kappa = torch.zeros(capacity, dtype=torch.float64)
        self.eta, self.clock, self.eta_image = float(eta), clock, float(eta_image)
        self.shortlist, self.floor, self.bounds = int(shortlist), int(floor), price_bounds
        self.price, self.mass, self.j_acq = {}, {}, {}
        self.categories, self.current = [], None
        self.price_trace = []
        self._pending = []
        self.check_terms = False                     # tests: verify incremental top-2 each swap

    @property
    def capacity(self):
        return self.bank.capacity

    # ----- category bookkeeping (task-aware training; inference never reads owners) -----
    def begin_category(self, c):
        c = int(c)
        if c not in self.categories:
            self.categories.append(c); self.mass[c] = 0.0
            acquired = [self.price[k] for k in self.j_acq]
            self.price[c] = 1.0 if not acquired else float(sum(acquired) / len(acquired))
        self.current = c

    def end_category(self):
        c = self.current
        require(c is not None, "no category in progress")
        self.j_acq[c] = self.cost(c)
        if self.clock == "boundary" and self.eta > 0:
            self._price_step(self.eta)
        self.price_trace.append({"event": "boundary", "category": c, "price": dict(self.price),
                                 "deficit": self.deficits()})
        self.current = None

    def count(self, c):
        return int((self.bank.owner == c).sum())

    def cost(self, k):
        """Normalised mean transport cost ``J_k`` of category k's observed normal mass."""
        if self.mass.get(k, 0.0) <= 0:
            return 0.0
        rows = self.bank.owner == k
        return float(self.kappa[rows].sum()) / (self.mass[k] * self.scales.s[k])

    def deficits(self):
        return {k: self.cost(k) - self.j_acq[k] for k in self.j_acq}

    def _price_step(self, eta):
        delta = self.deficits()
        if len(delta) < 2:
            return
        mean = sum(delta.values()) / len(delta)
        lo, hi = self.bounds
        for k, d in delta.items():
            self.price[k] = self.price[k] * math.exp(eta * (d - mean))
        norm = sum(self.price[k] for k in delta) / len(delta)
        for k in delta:  # renormalise, then box; the box can leave the mean slightly off 1
            self.price[k] = min(hi, max(lo, self.price[k] / norm))

    def _weight(self, k, extra=0.0):
        return self.price[k] / (self.scales.s[k] * (self.mass[k] + extra))

    # ----- fast level -----
    def _row_weights(self, c, extra):
        w = torch.zeros(self.capacity, dtype=torch.float64)
        for k in self.categories:
            if self.mass[k] + (extra if k == c else 0.0) > 0:
                w[self.bank.owner == k] = self._weight(k, extra if k == c else 0.0)
        return w

    def _protected(self):
        """Rows that may not be evicted: their category is at the floor."""
        p = ~self.bank.occupied
        for k in self.categories:
            rows = self.bank.owner == k
            if int(rows.sum()) <= self.floor:
                p |= rows
        return p

    def _current_terms(self, DI, c, rows=None):
        """Per patch: two nearest rows of ``c`` as ``(d1, d2, a1, a2)``; ``a2 = -1`` if absent."""
        cols = ((self.bank.owner == c) & self.bank.occupied).nonzero().flatten()
        if not len(cols):
            return None
        sub = DI[:, cols] if rows is None else DI[rows][:, cols]
        if len(cols) >= 2:
            v, i = sub.topk(2, dim=1, largest=False)
            return v[:, 0], v[:, 1], cols[i[:, 0]], cols[i[:, 1]]
        n = sub.shape[0]
        return (sub[:, 0], torch.full((n,), torch.inf, dtype=torch.float64), cols.expand(n),
                torch.full((n,), -1, dtype=torch.int64))

    def _swap_terms(self, terms, DI, c, j, old_owner):
        """Exact top-2 update after slot ``j`` (previous owner ``old_owner``) became a row of ``c``.

        Patches whose first or second neighbour was the old row ``j`` of ``c`` are recomputed;
        every other patch merges the new column. Distance values equal a full recompute.
        """
        d1, d2, a1, a2 = terms
        g = DI[:, j]
        aff = ((a1 == j) | (a2 == j)) if old_owner == c else torch.zeros_like(a1, dtype=torch.bool)
        lt1 = ~aff & (g < d1)
        lt2 = ~aff & ~lt1 & (g < d2)
        d2, a2 = torch.where(lt1, d1, torch.where(lt2, g, d2)), torch.where(lt1, a1, torch.where(lt2, j, a2))
        d1, a1 = torch.where(lt1, g, d1), torch.where(lt1, j, a1)
        if bool(aff.any()):
            rows = aff.nonzero().flatten()
            r1, r2, b1, b2 = self._current_terms(DI, c, rows)
            d1[rows], d2[rows], a1[rows], a2[rows] = r1, r2, b1, b2
        if self.check_terms:
            full = self._current_terms(DI, c)
            require(torch.equal(full[0], d1) and torch.equal(full[1], d2), "incremental top-2 drifted")
        return d1, d2, a1, a2

    def best_swap(self, DI, G, used, c, wc, Y, terms=None):
        """Global minimum of the exact objective change over (candidate x, evicted row j).

        Returns ``(delta, x, j, target)``; ``target`` is the slot receiving j's mass, or ``-1`` when
        it is the inserted candidate itself. Rows of other categories are exactly separable; rows of
        ``c`` are evaluated jointly on a shortlist (exhaustive when ``shortlist >= count(c)``).
        """
        d1, d2, a1, _ = terms if terms is not None else self._current_terms(DI, c)
        raw = (d1[:, None] - G).clamp_min(0).sum(0)    # = sum_z d1 - sum_z min(d1, G[:, x])
        gain = raw.masked_fill(used, -torch.inf)
        w = self._row_weights(c, extra=DI.shape[0])
        removable = ~self._protected()
        best = (torch.inf, None, None, None)
        old = removable & (self.bank.owner != c)
        if bool(old.any()):
            cost = (w * self.mu * self.graph.d).masked_fill(~old, torch.inf)
            j = int(cost.argmin()); x = int(gain.argmax())
            if math.isfinite(float(cost[j])) and math.isfinite(float(gain[x])):
                best = (float(cost[j]) - wc * float(gain[x]), x, j, int(self.graph.i[j]))
        own = (removable & (self.bank.owner == c)).nonzero().flatten()
        if len(own):
            served = torch.zeros(self.capacity, dtype=torch.float64)
            finite = torch.isfinite(d2)
            served.index_add_(0, a1[finite], (d2 - d1)[finite])
            approx = wc * (self.mu[own] * self.graph.d[own].nan_to_num(posinf=0.0) + served[own])
            R = own[torch.topk(approx, min(self.shortlist, len(own)), largest=False, sorted=True).indices]
            # Removing j changes d_c only for patches whose nearest c-row is j (sparse), so
            # cur[j, x] = -raw[x] + sum_{z: a1=j} [min(d2, G[z,x]) - min(d1, G[z,x])]   (exact).
            r_idx, z_idx = (a1[None] == R[:, None]).nonzero(as_tuple=True)
            cur = (-raw)[None].repeat(len(R), 1)                                   # [m, x]
            if len(z_idx):
                Gz = G[z_idx]
                cur.index_add_(0, r_idx, torch.minimum(d2[z_idx, None], Gz) - torch.minimum(d1[z_idx, None], Gz))
            to_x = self.dist(self.bank.x[R], Y).double()                           # [m, x]
            rho = self.graph.d[R][:, None]
            delta = wc * cur + (w[R] * self.mu[R])[:, None] * torch.minimum(rho.expand_as(to_x), to_x)
            delta = delta.masked_fill(used[None], torch.inf)
            flat = int(delta.argmin()); r, x = divmod(flat, delta.shape[1])  # row-major: shortlist order
            if float(delta[r, x]) < best[0]:
                j = int(R[r])
                target = -1 if float(to_x[r, x]) < float(rho[r, 0]) else int(self.graph.i[j])
                best = (float(delta[r, x]), x, j, target)
        return best

    def _write(self, slot, enc, x, c, serial, DI, G, defer=False):
        """``defer``: slot was free; its graph update is batched by ``_flush`` (exact, see NNGraph)."""
        self.bank.put(slot, enc, x, c, serial)
        self.mu[slot] = 0.0; self.kappa[slot] = 0.0
        DI[:, slot] = G[:, x]
        if defer:
            self._pending.append(slot)
        else:
            self._flush(); self.graph.changed(slot)

    def _flush(self):
        if self._pending:
            self.graph.inserted_many(self._pending); self._pending = []

    def update(self, x, serials):
        c = self.current
        require(c is not None, "begin_category required")
        require(x.ndim == 2 and len(serials) == len(x), "bad proposal batch")
        n = len(x)
        s_c = self.scales.observe(c, x, self.bank)
        enc = self.bank.encode(x)
        Y = enc["decoded"]
        G = self.dist(x, Y).double()                   # [patch z, candidate x]
        DI = self.bank.dist_to(x)
        used = torch.zeros(n, dtype=torch.bool)
        wc = self._weight(c, n)
        stats = {"inserted": 0, "replaced": 0, "forced": 0, "rejected": 0, "evicted_owner": {}}
        self._pending, d1, terms = [], None, None
        while not bool(used.all()):
            if self.count(c) == 0:                      # acquisition floor: forced medoid row
                xm = int(G.sum(0).masked_fill(used, torch.inf).argmin())
                slot = self.bank.free_slot()
                if slot is None:
                    self._flush()
                    w = self._row_weights(c, n)
                    cost = (w * self.mu * self.graph.d).masked_fill(self._protected(), torch.inf)
                    slot = int(cost.argmin())
                    require(math.isfinite(float(cost[slot])), "no evictable row for a new category")
                    self._evict(slot)
                self._write(slot, enc, xm, c, serials[xm], DI, G)
                used[xm] = True; stats["forced"] += 1; d1 = terms = None; continue
            slot = self.bank.free_slot()
            if slot is not None:                        # unfilled bank: insert while it helps
                if d1 is None:
                    d1 = self._current_terms(DI, c)[0]
                gain = (d1[:, None] - G).clamp_min(0).sum(0).masked_fill(used, -torch.inf)
                xb = int(gain.argmax())
                if not float(gain[xb]) > 0:
                    break
                self._write(slot, enc, xb, c, serials[xb], DI, G, defer=True)
                d1 = torch.minimum(d1, G[:, xb])        # insert-only: exact incremental update
                used[xb] = True; stats["inserted"] += 1; continue
            self._flush(); d1 = None
            if terms is None:
                terms = self._current_terms(DI, c)
            delta, xb, j, target = self.best_swap(DI, G, used, c, wc, Y, terms)
            if xb is None or not delta < 0:
                break
            self._evict(j, target_is_new=target == -1, new_vector=Y[xb])
            mu_j, kappa_j = self._carry if target == -1 else (0.0, 0.0)
            self._write(j, enc, xb, c, serials[xb], DI, G)
            self.mu[j], self.kappa[j] = mu_j, kappa_j
            terms = self._swap_terms(terms, DI, c, j, self._last_owner)
            used[xb] = True; stats["replaced"] += 1
            stats["evicted_owner"][self._last_owner] = stats["evicted_owner"].get(self._last_owner, 0) + 1
        self._flush()
        stats["rejected"] = int((~used).sum())
        d1, _, a1, _ = self._current_terms(DI, c)       # assign this image's mass
        self.mu.index_add_(0, a1, torch.ones(n, dtype=torch.float64))
        self.kappa.index_add_(0, a1, d1)
        self.mass[c] += float(n)
        if self.clock == "image" and self.eta_image > 0:
            self._price_step(self.eta_image)
            self.price_trace.append({"event": "image", "category": c, "price": dict(self.price)})
        stats["scale"] = s_c
        return stats

    def _evict(self, j, target_is_new=False, new_vector=None):
        """Move row j's mass to its nearest same-category survivor (or to the incoming row)."""
        self._last_owner = int(self.bank.owner[j])
        mu, kappa = float(self.mu[j]), float(self.kappa[j])
        if target_is_new:
            d = float(self.dist(self.bank.x[j:j + 1], new_vector[None]).double()[0, 0])
            self._carry = (mu, kappa + mu * d)
        else:
            t = int(self.graph.i[j])
            if t >= 0:
                self.kappa[t] += kappa + mu * float(self.graph.d[j]); self.mu[t] += mu
            else:
                require(mu == 0, "mass would be lost: no same-category survivor")
            self._carry = (0.0, 0.0)
        self.mu[j] = 0.0; self.kappa[j] = 0.0

    def objective(self, c, DI):
        """Independent evaluation of L(S) for the current image (used by tests)."""
        n = DI.shape[0]
        total = 0.0
        for k in self.categories:
            rows = self.bank.owner == k
            if self.mass[k] + (n if k == c else 0) > 0:
                total += self._weight(k, n if k == c else 0.0) * float(self.kappa[rows].sum())
        terms = self._current_terms(DI, c)
        if terms is not None:
            total += self._weight(c, n) * float(terms[0].sum())
        return total

    def export(self):
        return self.bank.export()
