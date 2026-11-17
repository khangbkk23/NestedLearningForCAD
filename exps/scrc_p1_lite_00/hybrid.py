"""Hybrid certified exact 1-NN (prereg §6.1): device FP32 screen, CPU FP64 batch-invariant final.

The screen is ``exps.nl_cad_pilot_00.nnx.certified_nn`` line for line (same q_chunk / r_chunk loops, same addmm, same
per-query band including the metric term ``nnx._extra``), with every band input (z rows, ||r||^2 rounded to FP32,
R, R_x, z_q, ||z_q||, ||x_q||) computed on the CPU exactly as the CPU path computes it and then moved to the device;
only the GEMM, the running minimum and the band comparison run on the device. The candidate pairs are re-evaluated
on the CPU with ``nnx._ref`` (``exact.reference``-equivalent for metric None; ``scrc.metric_dist_pairs`` otherwise)
and reduced lexicographically over (distance, id). Because the CPU reference is batch-invariant and every candidate
set is a certified superset of the minimisers, the result is bitwise the CPU ``nnx.certified_nn`` result whatever
the device screen rounding (test U4). Leave-image-out is not supported (test queries are not bank members).
"""
from __future__ import annotations

import math
import time

import torch

from exps.nl_cad_pilot_00 import nnx
from exps.nl_cad_pilot_00.scrc import apply_sqrt
from exps.nl_sgm_01a import exact as X
from exps.scrc_p1_lite_00 import protocol as P

U32 = X.U32


class ScreenRows:
    """CPU ``nnx.Rows`` (canonical construction) plus device copies of the screen inputs."""

    def __init__(self, x, ids, metric=None):
        self.metric = metric
        self.cpu = nnx.Rows(x, ids) if metric is None else nnx.Rows.transformed(x, ids, metric, "cpu")
        self.dev = None
        self.device = "cpu"

    def to(self, device):
        device = str(device)
        if device == "cpu":
            self.dev, self.device = None, "cpu"
        elif self.device != device:
            self.dev = {"z": self.cpu.z.to(device), "n2_32": self.cpu.n2_32.to(device)}
            self.device = device
        return self

    def release(self):
        self.dev, self.device = None, "cpu"

    @property
    def z(self):
        return self.cpu.z if self.dev is None else self.dev["z"]

    @property
    def n2_32(self):
        return self.cpu.n2_32 if self.dev is None else self.dev["n2_32"]


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def hybrid_certified_nn(q, srows, q_chunk=1024, r_chunk=32768, stats=None, per_query_counts=False):
    """Exact 1-NN of raw FP32 CPU queries ``q`` [m, d] against ``srows`` under ``srows.metric``.

    Returns CPU (dist float64 [m], id int64 [m]); ``stats`` collects candidates, max per query, timings and (if
    ``per_query_counts``) the per-query candidate counts (int32 [m]).
    """
    P.require(q.ndim == 2 and q.dtype == torch.float32 and q.device.type == "cpu", "queries must be CPU FP32 [m, d]")
    metric, rows, dev = srows.metric, srows.cpu, srows.device
    m, d = q.shape
    out_d = torch.full((m,), math.inf, dtype=torch.float64)
    out_i = torch.full((m,), -1, dtype=torch.int64)
    counts = torch.zeros(m, dtype=torch.int32) if per_query_counts else None
    if len(rows.z) == 0 or m == 0:
        return out_d, out_i
    ident = metric is None or metric.is_identity
    t0 = time.perf_counter()
    zq_all = q if ident else apply_sqrt(q, metric).float()
    t_screen = t_final = 0.0
    t_prep = time.perf_counter() - t0
    for a in range(0, m, q_chunk):
        ta = time.perf_counter()
        Q = zq_all[a:a + q_chunk]
        Qraw = q[a:a + q_chunk]
        qn = Q.double().square().sum(1).sqrt()
        xn = Qraw.double().square().sum(1).sqrt() if not ident else qn
        band = ((X.screen_band(qn, rows.R) + 4 * U32 * (rows.R ** 2 + 2 * qn * rows.R)
                 + nnx._extra(qn, rows.R, xn, rows.Rx, metric, d)) * (1 + 1e-6)).float()
        Qd, bd = (Q, band) if dev == "cpu" else (Q.to(dev), band.to(dev))
        best = torch.full((len(Q),), math.inf, dtype=torch.float32, device=Qd.device)
        cand_q, cand_r, cand_t = [], [], []
        Z, N2 = srows.z, srows.n2_32
        for b in range(0, len(Z), r_chunk):
            Xb = Z[b:b + r_chunk]
            T = torch.addmm(N2[b:b + r_chunk][None].expand(len(Q), -1), Qd, Xb.T, beta=1.0, alpha=-2.0)
            cmin = T.min(1).values
            best = torch.minimum(best, cmin)
            keep = T <= (best + bd)[:, None]
            qi, ri = keep.nonzero(as_tuple=True)
            cand_q.append(qi); cand_r.append(ri + b); cand_t.append(T[qi, ri])
        qi, ri, tv = torch.cat(cand_q), torch.cat(cand_r), torch.cat(cand_t)
        sel = tv <= (best + bd)[qi]
        qi, ri = qi[sel].cpu(), ri[sel].cpu()
        _sync(dev)
        tb = time.perf_counter()
        t_screen += tb - ta
        if stats is not None:
            stats["candidates"] = stats.get("candidates", 0) + int(len(qi))
            stats["queries"] = stats.get("queries", 0) + int(len(Q))
            bc = torch.bincount(qi, minlength=len(Q))
            stats["max_candidates_per_query"] = max(stats.get("max_candidates_per_query", 0),
                                                    int(bc.max()) if len(qi) else 0)
            if counts is not None:
                counts[a:a + len(Q)] = bc.to(torch.int32)
        if len(qi) == 0:
            continue
        dd = torch.empty(len(qi), dtype=torch.float64)
        for s in range(0, len(qi), 65536):
            dd[s:s + 65536] = nnx._ref(Qraw[qi[s:s + 65536]], rows.raw[ri[s:s + 65536]], metric)
        gid = rows.ids[ri]
        order = torch.argsort(gid, stable=True)
        order = order[torch.argsort(dd[order], stable=True)]
        order = order[torch.argsort(qi[order], stable=True)]
        qs = qi[order]
        first = torch.ones(len(qs), dtype=torch.bool)
        first[1:] = qs[1:] != qs[:-1]
        pick = order[first]
        out_d[a + qi[pick]] = dd[pick]
        out_i[a + qi[pick]] = gid[pick]
        t_final += time.perf_counter() - tb
    if stats is not None:
        stats["seconds_prep"] = stats.get("seconds_prep", 0.0) + t_prep
        stats["seconds_screen"] = stats.get("seconds_screen", 0.0) + t_screen
        stats["seconds_final_cpu"] = stats.get("seconds_final_cpu", 0.0) + t_final
        if counts is not None:
            stats["per_query_counts"] = counts
    return out_d, out_i
