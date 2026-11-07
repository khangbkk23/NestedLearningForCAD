"""NL-SGM-01A arm adapters: thin wrappers around the canonical selection rules, plus Stage-B codec wrappers.

Canonical rules re-used unchanged (imported, never copied):
  CADIC closest-pair / equal quota  -> exps.calm01.closest_pair.ClosestPairMemory (p0 FP64 kernel)
  stream greedy                     -> exps.task_p.p0_normal_support.Support("STREAM_GREEDY")
  AD-02 S4 / R4 partitions          -> exps.ad02.hope_cad_ad02_partition (assignment, per_group_quota) with one
                                       ClosestPairMemory(cadic) per group (new implementation of the AD-02
                                       semantics; per-group parity with the canonical AD-02 memory is tested)
  boundary FPS                      -> exps.csg01.select.select_fps
  CALM-L1                           -> exps.calm01.arms.build("L1_FIXED_U8")
  hash-random priority              -> exps.csg01.select.random_priorities
  UINT8 codec                       -> exps.calm01.storage.quantize_rows (= exps.csg01.quantize rule)

Every rule selects on the *decoded* rows it stores. Provenance (global writer id) is never read by a rule's
decision except as the opaque serial the canonical classes already carry; owners are read only by rules that
need them (quota, partition bookkeeping, FPS per-category quotas, CALM, B7 codebook version).
"""
from __future__ import annotations

import numpy as np
import torch

from exps.ad02 import hope_cad_ad02_partition as AP
from exps.calm01 import arms as CA
from exps.calm01.closest_pair import ClosestPairMemory
from exps.calm01.storage import NNGraph
from exps.csg01.select import random_priorities, select_fps
from exps.nl_sgm_01a import codecs as K
from exps.task_p.p0_normal_support import Support, distances, require

FP32_ROW = 768 * 4


def r4_assignment(draw):
    """AD-02 R4 draw ``draw`` exactly as exps/ad02/hope_cad_ad02_run.resolve_partition builds it."""
    a, diag, meta = AP.build_partition("random", grid=4, seed=10_000 + int(draw))
    return a, diag.fingerprint


def s4_assignment():
    a = AP.spatial_assignment(4)
    return a, AP.assignment_fingerprint(a)


def rebuild_graph(graph, chunk=2048):
    """Chunked ``NNGraph.rebuild``: each row's nearest-other is recomputed independently by the canonical
    ``_recompute`` (lowest slot on ties), so chunking gives the identical graph without an [n, n] matrix."""
    occ = graph.bank.occupied.nonzero().flatten()
    for a in range(0, len(occ), chunk):
        graph._recompute(occ[a:a + chunk])


# ====================================================================== inner rules

class ClosestPairRule:
    """CADIC closest-pair (mode 'cadic') or equal-quota (mode 'quota'); optional fixed patch partition."""

    def __init__(self, capacity, mode="cadic", assignment=None, dim=768, distance="p0_fp64"):
        self.mode, self.assignment, self.dim, self.distance = mode, assignment, dim, distance
        mk = lambda q: ClosestPairMemory(q, mode=mode, storage="fp32", dim=dim, distance=distance)
        if assignment is None:
            self.quotas = [int(capacity)]
            self.mems = [mk(int(capacity))]
        else:
            n_groups = int(assignment.max()) + 1
            self.quotas = AP.per_group_quota(int(capacity), n_groups)
            self.mems = [mk(q) for q in self.quotas]
        self.capacity = int(capacity)

    def begin_task(self, t):
        for m in self.mems:
            m.begin_category(t)

    def end_task(self, t):
        for m in self.mems:
            m.end_category()

    def image(self, x, gids, t):
        if self.assignment is None:
            self.mems[0].update(x, gids.tolist())
            return
        for g, m in enumerate(self.mems):
            sel = self.assignment == g
            m.update(x[sel], gids[sel].tolist())

    def rows(self):
        xs, ids, own = [], [], []
        for m in self.mems:
            occ = m.bank.occupied
            xs.append(m.bank.x[occ]); ids.append(m.bank.ids[occ]); own.append(m.bank.owner[occ])
        return torch.cat(xs), torch.cat(ids), torch.cat(own)

    def redecode(self, fn):
        """Replace every stored decoded row by fn(gids) and rebuild the derived NN graph (exact)."""
        for m in self.mems:
            occ = m.bank.occupied.nonzero().flatten()
            if not len(occ):
                continue
            new = fn(m.bank.ids[occ])
            m.bank.x[occ] = new
            m.bank.x64[occ] = new.double(); m.bank.n2[occ] = new.double().square().sum(1)
            rebuild_graph(m.graph)

    def shrink(self, capacity):
        """Evict by the rule's own closest-pair victim until each bank fits, then rebuild smaller banks."""
        quotas = [int(capacity)] if self.assignment is None else AP.per_group_quota(int(capacity), len(self.mems))
        new_mems = []
        for m, q in zip(self.mems, quotas):
            if self.mode == "quota":
                cats = list(m.categories)
                share = q // max(1, len(cats))
                for k in cats:
                    while m.count(k) > share and int(m.bank.occupied.sum()) > q:
                        _, slot = m._victim(m.bank.owner == k)
                        m.bank.clear(slot); m.graph.changed(slot)
            while int(m.bank.occupied.sum()) > q:
                _, slot = m._victim(m.bank.occupied)
                m.bank.clear(slot); m.graph.changed(slot)
            nm = ClosestPairMemory(q, mode=self.mode, storage="fp32", dim=self.dim, distance=self.distance)
            occ = m.bank.occupied.nonzero().flatten()
            for s, slot in enumerate(occ.tolist()):
                nm.bank.put(s, {"decoded": m.bank.x[slot:slot + 1]}, 0, int(m.bank.owner[slot]), int(m.bank.ids[slot]))
            nm.graph = NNGraph(nm.bank, same_owner=self.mode == "quota"); rebuild_graph(nm.graph)
            nm.categories, nm.current = list(m.categories), m.current
            new_mems.append(nm)
        self.mems, self.quotas, self.capacity = new_mems, quotas, int(capacity)


class GreedyRule:
    """P0 STREAM_GREEDY (exact incremental nearest-distinct graph)."""

    def __init__(self, capacity):
        self.s = Support("STREAM_GREEDY", capacity=int(capacity))
        self.capacity = int(capacity)
        self.owner = torch.full((int(capacity),), -1, dtype=torch.int64)    # bookkeeping only (not read)

    def begin_task(self, t):
        self.t = t

    def end_task(self, t):
        pass

    def image(self, x, gids, t):
        before = self.s.ids.clone()
        self.s.update(x, gids)
        self.owner[self.s.ids != before] = t

    def rows(self):
        n = self.s.n
        return self.s.bank[:n].clone(), self.s.ids[:n].clone(), self.owner[:n].clone()

    def _rebuild_graph(self):
        n = self.s.n
        for a in range(0, n, 2048):
            D = distances(self.s.bank[a:a + 2048], self.s.bank[:n])
            D[torch.arange(len(D)), torch.arange(a, a + len(D))] = torch.inf
            v, ix = D.min(1)
            self.s.nn_distance[a:a + len(D)] = v; self.s.nn_index[a:a + len(D)] = ix

    def redecode(self, fn):
        n = self.s.n
        if n:
            self.s.bank[:n] = fn(self.s.ids[:n])
            self._rebuild_graph()

    def shrink(self, capacity):
        while self.s.n > capacity:
            n = self.s.n
            v = int(torch.argsort(self.s.nn_distance[:n], stable=True)[0])     # its own replacement victim
            keep = torch.ones(n, dtype=torch.bool); keep[v] = False
            bank, ids, own = self.s.bank[:n][keep].clone(), self.s.ids[:n][keep].clone(), self.owner[:n][keep].clone()
            self.s.bank[:n - 1] = bank; self.s.ids[:n - 1] = ids; self.owner[:n - 1] = own
            self.s.ids[n - 1] = -1; self.owner[n - 1] = -1; self.s.n = n - 1
            self._rebuild_graph()
        s = Support("STREAM_GREEDY", capacity=int(capacity))
        n = self.s.n
        s.bank[:n] = self.s.bank[:n]; s.ids[:n] = self.s.ids[:n]; s.n = n
        s.seen, s.events = self.s.seen, self.s.events
        own = torch.full((int(capacity),), -1, dtype=torch.int64); own[:n] = self.owner[:n]
        self.s, self.owner, self.capacity = s, own, int(capacity)
        self._rebuild_graph()


class FPSRule:
    """Fixed-rule boundary consolidation: at task end, FPS of the task to its equal quota; old categories are
    FPS-subselected within their own kept rows. Quotas: AD-02 per_group_quota (remainder to earlier tasks)."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.kept = {}                                     # t -> (rows FP32, gids)
        self.buf_x, self.buf_g = [], []
        self.transient_peak = 0

    def begin_task(self, t):
        self.buf_x, self.buf_g = [], []

    def image(self, x, gids, t):
        self.buf_x.append(x.clone()); self.buf_g.append(gids.clone())
        self.transient_peak = max(self.transient_peak, sum(b.numel() * 4 for b in self.buf_x))

    def _requota(self):
        cats = sorted(self.kept)
        quotas = AP.per_group_quota(self.capacity, len(cats)) if len(cats) <= self.capacity else None
        require(quotas is not None, "FPS capacity below category count")
        for k, q in zip(cats, quotas):
            x, g = self.kept[k]
            if len(x) > q:
                idx = torch.as_tensor(select_fps(x, q))
                self.kept[k] = (x[idx], g[idx])

    def end_task(self, t):
        x, g = torch.cat(self.buf_x), torch.cat(self.buf_g)
        self.buf_x, self.buf_g = [], []
        self.transient_peak = max(self.transient_peak, x.numel() * 4 + x.numel() * 8)   # + FP64 copy in FPS
        self.kept[t] = (x, g)
        self._requota()

    def rows(self):
        cats = sorted(self.kept)
        if not cats:
            return torch.zeros(0, 768), torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64)
        return (torch.cat([self.kept[k][0] for k in cats]), torch.cat([self.kept[k][1] for k in cats]),
                torch.cat([torch.full((len(self.kept[k][1]),), k, dtype=torch.int64) for k in cats]))

    def redecode(self, fn):
        for k in list(self.kept):
            self.kept[k] = (fn(self.kept[k][1]), self.kept[k][1])

    def shrink(self, capacity):
        self.capacity = int(capacity)
        self._requota()


class CALMRule:
    """CALM-L1 (fixed prices, UINT8) at B1 exactly as CALM-01/CAD-FINAL build it."""

    def __init__(self, budget=7_680_000, n_categories=15):
        self.m = CA.build("L1_FIXED_U8", budget, n_categories)
        self.capacity = self.m.capacity
        self.n_categories = n_categories

    def begin_task(self, t):
        self.m.begin_category(t)

    def end_task(self, t):
        self.m.end_category()

    def image(self, x, gids, t):
        self.m.update(x, gids.tolist())

    def rows(self):
        b = self.m.bank
        occ = b.occupied
        return b.x[occ].clone(), b.ids[occ].clone(), b.owner[occ].clone()

    def charged_bytes(self):
        return CA.state_bytes(CA.persistent_state(self.m, "L1_FIXED_U8", self.n_categories))


class RandomRule:
    """Causal bottom-k hash-random sample (priority = csg01 random_priorities(seed, gid)); stores what it gets."""

    def __init__(self, capacity, seed=20261106):
        self.capacity, self.seed = int(capacity), int(seed)
        self.x = torch.zeros(0, 768); self.g = torch.zeros(0, dtype=torch.int64)
        self.o = torch.zeros(0, dtype=torch.int64); self.p = np.zeros(0, dtype=np.uint64)

    def begin_task(self, t):
        pass

    def end_task(self, t):
        pass

    def image(self, x, gids, t):
        pr = random_priorities(self.seed, gids.numpy().astype(np.int64))
        p = np.concatenate([self.p, pr]); g = torch.cat([self.g, gids])
        order = np.lexsort((g.numpy(), p))[:self.capacity]
        keep = torch.as_tensor(order)
        self.x = torch.cat([self.x, x])[keep]; self.g = g[keep]
        self.o = torch.cat([self.o, torch.full((len(x),), t, dtype=torch.int64)])[keep]; self.p = p[order]

    def rows(self):
        return self.x.clone(), self.g.clone(), self.o.clone()

    def redecode(self, fn):
        self.x = fn(self.g)

    def shrink(self, capacity):
        self.capacity = int(capacity)
        self.x, self.g, self.o, self.p = self.x[:capacity], self.g[:capacity], self.o[:capacity], self.p[:capacity]


# ====================================================================== Stage-B codec wrapper

class CodecArm:
    """Runs an inner rule on decoded rows under a codec.

    kind: 'fp32' | 'u8' | 'pq_fixed' (B5: C_0 fit at the end of task 0, frozen) | 'pq_refit' (B6: C_0 as B5,
    then at the end of every task t>=1 refit on all *decoded stored rows* plus the hash-first task-t patches up to
    100,000 rows, then transport) | 'pq_append' (B7: one frozen FP16 codebook per task, row version = owner).
    A PQ codebook for task t's rows that does not yet exist (task 0; every task for 'pq_append') defers that task's
    writes to its boundary: the task's FP32 images are held in a transient buffer, the codebook is fit, and the
    images are replayed through the rule in stream order. Historical FP32 source rows are never read.
    """

    FIT_ROWS = 100_000

    def __init__(self, inner, kind, budget=7_680_000, owner_bytes=1, seed=20261106, sidecar=True):
        require(kind in ("fp32", "u8", "pq_fixed", "pq_refit", "pq_append"), "unknown codec kind")
        self.inner, self.kind, self.budget, self.owner_bytes, self.seed = inner, kind, budget, owner_bytes, seed
        self.codebooks = {}                    # version -> FP16 codebook (charged, live)
        self.history = {}                      # every codebook ever fit (diagnostic sidecar, uncharged)
        self.codes = {}                        # gid -> uint8 [96] (pq) ; stored rows only after pruning
        self.version = {}                      # gid -> codebook version
        self.u8 = {}                           # gid -> (q, xmin, step)
        self.write_codes = {} if sidecar else None   # diagnostic sidecar (uncharged): codes/version at write time
        self.buffer, self.sample = [], None
        self.transient_peak = 0
        self.drift = []                        # per boundary transport records (B6)
        self.t = None

    # -------------------------------------------------- helpers
    def _deferred(self, t):
        return self.kind == "pq_append" or (self.kind in ("pq_fixed", "pq_refit") and not self.history)

    def _current_version(self, t):
        if self.kind == "pq_append":
            return t
        return max(self.codebooks) if self.codebooks else None

    def _encode(self, x, gids, t):
        if self.kind == "fp32":
            return x
        if self.kind == "u8":
            pay, dec = K.u8_encode(x)
            for j, g in enumerate(gids.tolist()):
                self.u8[g] = (pay["q"][j], float(pay["xmin"][j]), float(pay["step"][j]))
            return dec
        v = self._current_version(t)
        cb = self.codebooks[v]
        codes = K.encode(x, cb)
        for j, g in enumerate(gids.tolist()):
            self.codes[g] = codes[j]; self.version[g] = v
            if self.write_codes is not None:
                self.write_codes[g] = (codes[j], v)
        return K.decode(codes, cb)

    def _prune(self):
        _, gids, _ = self.inner.rows()
        live = set(gids.tolist())
        for b in getattr(self.inner, "buf_g", []):          # boundary rules buffer the current task: still live
            live.update(b.tolist())                         # (fix r2, 2026-11-06: KeyError in AM02 run r1)
        for store in (self.codes, self.version, self.u8):
            for g in [g for g in store if g not in live]:
                del store[g]

    def _sample_add(self, x, gids):
        pr = random_priorities(self.seed, gids.numpy().astype(np.int64))
        if self.sample is None:
            self.sample = (x.clone(), gids.clone(), pr)
        else:
            sx, sg, sp = self.sample
            p = np.concatenate([sp, pr]); g = torch.cat([sg, gids])
            order = np.lexsort((g.numpy(), p))[:self.FIT_ROWS]
            self.sample = (torch.cat([sx, x])[torch.as_tensor(order)], g[torch.as_tensor(order)], p[order])
        self._note_transient()

    def _note_transient(self):
        b = sum(x.numel() * 4 for x, _ in self.buffer)
        s = 0 if self.sample is None else self.sample[0].numel() * 4
        self.transient_peak = max(self.transient_peak, b + s)

    # -------------------------------------------------- stream interface
    def begin_task(self, t):
        self.t = t
        self.inner.begin_task(t)
        self.buffer, self.sample = [], None

    def image(self, x, gids, t):
        if self.kind.startswith("pq") and (self._deferred(t) or self.kind == "pq_refit"):
            self._sample_add(x, gids)
        if self.kind.startswith("pq") and self._deferred(t):
            self.buffer.append((x.clone(), gids.clone())); self._note_transient()
            return
        self.inner.image(self._encode(x, gids, t), gids, t)
        if len(self.codes) + len(self.u8) > 4 * self.inner.capacity + 10_000:
            self._prune()

    def end_task(self, t):
        if self.kind.startswith("pq"):
            if self._deferred(t):
                sx = self.sample[0]
                v = t if self.kind == "pq_append" else 0
                self.codebooks[v] = K.fit(sx, seed=self.seed); self.history[v] = self.codebooks[v]
                if self.kind == "pq_append":
                    cap = K.pq_capacity(self.budget, len(self.codebooks), self.owner_bytes)
                    require(cap >= 2, "append-only codebooks exhaust the budget")
                    if cap < self.inner.capacity:
                        self.inner.shrink(cap)
                for x, g in self.buffer:
                    self.inner.image(self._encode(x, g, t), g, t)
                self.buffer = []
            elif self.kind == "pq_refit":
                self._refit_transport(t)
        self.inner.end_task(t)
        self._prune()
        self.sample = None

    def _refit_transport(self, t):
        dec, gids, _ = self.inner.rows()
        n_new = max(0, self.FIT_ROWS - len(dec))
        fit_rows = torch.cat([dec, self.sample[0][:n_new]])
        old_v = max(self.codebooks)
        cb_old = self.codebooks[old_v]
        cb_new = K.fit(fit_rows, seed=self.seed)
        old_codes = torch.stack([self.codes[g] for g in gids.tolist()])
        new_codes, rec = K.transport(old_codes, cb_old, cb_new)
        self.codebooks = {t: cb_new}                       # the old codebook is discarded (shared code space)
        self.history[t] = cb_new
        for j, g in enumerate(gids.tolist()):
            self.codes[g] = new_codes[j]; self.version[g] = t
        lookup = {g: j for j, g in enumerate(gids.tolist())}
        newdec = K.decode(new_codes, cb_new)
        self.inner.redecode(lambda ids: newdec[torch.as_tensor([lookup[int(i)] for i in ids])])
        rec = {k: v.clone() for k, v in rec.items()}
        rec.update({"boundary": t, "gids": gids.clone(), "fit_rows": int(len(fit_rows)), "task_rows": n_new})
        self.drift.append(rec)

    # -------------------------------------------------- state
    def support(self):
        return self.inner.rows()

    def persistent_state(self):
        """Charged state in charged dtypes, at the arm's fixed capacity (unused slots are zero rows)."""
        dec, gids, own = self.inner.rows()
        cap = self.inner.capacity
        st = {}
        if self.kind == "fp32":
            x = torch.zeros(cap, 768); x[:len(dec)] = dec; st["x"] = x
        elif self.kind == "u8":
            q = torch.zeros(cap, 768, dtype=torch.uint8); xm = torch.zeros(cap); sp = torch.zeros(cap)
            for j, g in enumerate(gids.tolist()):
                q[j], xm[j], sp[j] = self.u8[g]
            st.update(q=q, xmin=xm, step=sp)
        else:
            c = torch.zeros(cap, K.M, dtype=torch.uint8)
            for j, g in enumerate(gids.tolist()):
                c[j] = self.codes[g]
            st["codes"] = c
            st["codebooks"] = torch.stack([self.codebooks[v] for v in sorted(self.codebooks)])
        if self.owner_bytes:
            o = torch.full((cap,), -1, dtype=torch.int8); o[:len(own)] = own.to(torch.int8); st["owner"] = o
        return st


def state_bytes(state):
    return int(sum(v.numel() * v.element_size() for v in state.values()))
