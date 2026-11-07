"""Certified exact 1-NN for SC-SN-01 (R1).

Reference distance (the SC-SN-01 per-pair value used by *every* arm, oracle and diagnostic):

    D(q, r) = sqrt( sum_i (float64(q_i) - float64(r_i))^2 )      direct-difference FP64, no cancellation

1-NN = min_r D(q, r); ties -> lowest global row index.

Screen (FP32 GEMM) and completeness argument
--------------------------------------------
For a query q and rows r the screen computes ``t(r) = fl32(||r||^2) - 2 fl32(q . r)`` (the per-query constant
||q||^2 is dropped). With the inputs exactly representable in FP32 and the BLAS inner product evaluated with FP32
multiplications / additions / FMAs in any order (Higham, Thm 3.1 / 3.5: |fl(q.r) - q.r| <= gamma_n sum |q_i r_i|
<= gamma_n ||q|| ||r||, gamma_n = n u / (1 - n u), u = 2^-24, n = 768) and the two FP32 roundings of the final
subtraction, ``|t(r) - t_true(r)| <= E_q``  with

    E_q = 2 gamma_n ||q|| R + u (R^2 + 2 ||q|| R) + u R^2 + u (R^2 + 2||q||R),   R = max_r ||r|| in the row set.

The true squared distance is ||q||^2 + t_true(r). Let r* be any row with minimal *reference* value D. D^2 differs
from the true squared distance by at most eps64 = gamma_n(2^-53) * D^2_true (+ the sqrt rounding, monotone).
Hence every row that can attain min D satisfies ``t(r) <= min_r' t(r') + 2 E_q + 2 eps_q`` with
eps_q = 4 gamma_n(2^-53) (||q|| + R)^2 (an upper bound of eps64 for every row). All rows passing this band are
re-evaluated with D and the minimum (lowest index on ties) is returned. The band is a superset of the minimisers,
so the result equals min over *all* rows of D. Assumption: the BLAS uses ordinary FP32 arithmetic (no Strassen,
no reduced precision); ``verify_screen_bound`` checks the measured screen error against E_q on every call in
tests and on BM1 data.
"""
from __future__ import annotations

import math

import torch

from exps.task_p.p0_normal_support import require

U32 = 2.0 ** -24
U64 = 2.0 ** -53
N = 768


def gamma(n, u):
    return n * u / (1 - n * u)


def reference(q, r):
    """D for aligned pairs: q [k, d], r [k, d] -> [k] float64 (the SC-SN-01 per-pair value)."""
    return (q.double() - r.double()).square_().sum(1).sqrt_()


def screen_band(qnorm, R, n=N):
    """Per-query band 2 E_q + 2 eps_q (float64 tensor)."""
    g32, g64 = gamma(n, U32), gamma(n, U64)
    E = 2 * g32 * qnorm * R + U32 * (R * R + 2 * qnorm * R) + U32 * R * R + U32 * (R * R + 2 * qnorm * R)
    eps = 4 * g64 * (qnorm + R) ** 2
    return 2 * E + 2 * eps + 1e-30


class RowSet:
    """FP32 rows with global ids; ids must be strictly increasing (lowest-index tie-break = lowest id)."""

    def __init__(self, x, ids):
        require(x.ndim == 2 and x.dtype == torch.float32 and len(ids) == len(x), "bad row set")
        ids = torch.as_tensor(ids, dtype=torch.int64)
        require(len(ids) < 2 or bool((ids[1:] > ids[:-1]).all()), "row ids must be strictly increasing")
        self.x, self.ids = x, ids
        n2 = x.double().square().sum(1)
        self.n2_32 = n2.float()
        self.R = float(n2.max().sqrt()) if len(x) else 0.0


def nn(q, rows, exclude=None, q_chunk=1024, r_chunk=32768, stats=None):
    """Exact 1-NN of each query against ``rows`` (RowSet).

    ``exclude``: optional int64 tensor [m] of a row id to skip per query (self-exclusion); -1 = none.
    Returns (dist float64 [m], id int64 [m]); dist=inf, id=-1 if no admissible row.
    """
    require(q.ndim == 2 and q.dtype == torch.float32, "queries must be FP32 [m, d]")
    m = len(q)
    out_d = torch.full((m,), math.inf, dtype=torch.float64)
    out_i = torch.full((m,), -1, dtype=torch.int64)
    if len(rows.x) == 0 or m == 0:
        return out_d, out_i
    for a in range(0, m, q_chunk):
        Q = q[a:a + q_chunk]
        qn = Q.double().square().sum(1).sqrt()
        # band + FP32 rounding of (best + band) comparisons (|t| <= R^2 + 2||q||R)
        band = ((screen_band(qn, rows.R) + 4 * U32 * (rows.R ** 2 + 2 * qn * rows.R)) * (1 + 1e-6)).float()
        ex = None if exclude is None else exclude[a:a + q_chunk]
        best = torch.full((len(Q),), math.inf, dtype=torch.float32)
        cand_q, cand_r, cand_t = [], [], []
        for b in range(0, len(rows.x), r_chunk):
            Xb = rows.x[b:b + r_chunk]
            T = torch.addmm(rows.n2_32[b:b + r_chunk][None].expand(len(Q), -1), Q, Xb.T, beta=1.0, alpha=-2.0)
            if ex is not None:
                rid = rows.ids[b:b + r_chunk]
                hit = (ex[:, None] == rid[None])
                if bool(hit.any()):
                    T.masked_fill_(hit, math.inf)
            cmin = T.min(1).values
            best = torch.minimum(best, cmin)
            keep = T <= (best + band)[:, None]
            qi, ri = keep.nonzero(as_tuple=True)
            cand_q.append(qi); cand_r.append(ri + b); cand_t.append(T[qi, ri])
        qi, ri, tv = torch.cat(cand_q), torch.cat(cand_r), torch.cat(cand_t)
        sel = tv <= (best + band)[qi]                                   # final band against the global min
        qi, ri = qi[sel], ri[sel]
        if stats is not None:
            stats["candidates"] = stats.get("candidates", 0) + int(len(qi))
            stats["queries"] = stats.get("queries", 0) + int(len(Q))
        if len(qi) == 0:
            continue
        d = torch.empty(len(qi), dtype=torch.float64)
        for s in range(0, len(qi), 65536):
            d[s:s + 65536] = reference(Q[qi[s:s + 65536]], rows.x[ri[s:s + 65536]])
        gid = rows.ids[ri]
        # lexicographic min over (d, gid) per query
        order = torch.argsort(gid, stable=True)
        order = order[torch.argsort(d[order], stable=True)]
        order = order[torch.argsort(qi[order], stable=True)]
        qs = qi[order]
        first = torch.ones(len(qs), dtype=torch.bool)
        first[1:] = qs[1:] != qs[:-1]
        pick = order[first]
        out_d[a + qi[pick]] = d[pick]
        out_i[a + qi[pick]] = gid[pick]
    return out_d, out_i


def kth(q, rows, k, exclude=None, q_chunk=512, r_chunk=32768):
    """Exact k-th nearest reference distance (self excluded via ``exclude``); same certified band argument with
    the running k-th smallest screened value as the upper bound."""
    m = len(q)
    out = torch.full((m,), math.inf, dtype=torch.float64)
    for a in range(0, m, q_chunk):
        Q = q[a:a + q_chunk]
        qn = Q.double().square().sum(1).sqrt()
        band = ((screen_band(qn, rows.R) + 4 * U32 * (rows.R ** 2 + 2 * qn * rows.R)) * (1 + 1e-6)).float()
        ex = None if exclude is None else exclude[a:a + q_chunk]
        topk = torch.full((len(Q), k), math.inf)
        cq, cr, ct = [], [], []
        for b in range(0, len(rows.x), r_chunk):
            T = torch.addmm(rows.n2_32[b:b + r_chunk][None].expand(len(Q), -1), Q, rows.x[b:b + r_chunk].T,
                            beta=1.0, alpha=-2.0)
            if ex is not None:
                T.masked_fill_(ex[:, None] == rows.ids[b:b + r_chunk][None], math.inf)
            topk = torch.cat([topk, T], 1).topk(k, dim=1, largest=False).values
            lim = topk[:, -1] + band
            qi, ri = (T <= lim[:, None]).nonzero(as_tuple=True)
            cq.append(qi); cr.append(ri + b); ct.append(T[qi, ri])
        qi, ri, tv = torch.cat(cq), torch.cat(cr), torch.cat(ct)
        sel = tv <= (topk[:, -1] + band)[qi]
        qi, ri = qi[sel], ri[sel]
        d = reference(Q[qi], rows.x[ri])
        for j in range(len(Q)):
            v = d[qi == j]
            if len(v) >= k:
                out[a + j] = v.sort().values[k - 1]
    return out


def merge(d1, i1, d2, i2):
    """Prefix-min of two exact results over disjoint row sets (lowest id on ties)."""
    take2 = (d2 < d1) | ((d2 == d1) & (i2 >= 0) & ((i1 < 0) | (i2 < i1)))
    return torch.where(take2, d2, d1), torch.where(take2, i2, i1)


def brute(q, x, ids=None, exclude=None):
    """Brute-force reference: D on every pair, lowest id on ties. For tests only (small)."""
    ids = torch.arange(len(x)) if ids is None else torch.as_tensor(ids)
    m = len(q)
    D = (q.double()[:, None, :] - x.double()[None]).square().sum(-1).sqrt()
    if exclude is not None:
        D[exclude[:, None] == ids[None]] = math.inf
    out_d, out_i = torch.full((m,), math.inf, dtype=torch.float64), torch.full((m,), -1, dtype=torch.int64)
    for j in range(m):
        v = D[j].min()
        if math.isfinite(float(v)):
            out_d[j] = v
            out_i[j] = ids[D[j] == v].min()
    return out_d, out_i


def verify_screen_bound(q, x):
    """Max |t_screen - t_exact| / E_q over all pairs (must be <= 1)."""
    rows = RowSet(x, torch.arange(len(x)))
    T = torch.addmm(rows.n2_32[None].expand(len(q), -1), q, x.T, beta=1.0, alpha=-2.0).double()
    exact = x.double().square().sum(1)[None] - 2 * (q.double() @ x.double().T)
    qn = q.double().square().sum(1).sqrt()
    band = screen_band(qn, rows.R) / 2                                  # = E_q + eps_q
    return float(((T - exact).abs() / band[:, None]).max())
