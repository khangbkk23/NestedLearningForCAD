"""Certified exact nearest-neighbour search for SCRC (Euclidean, metric-induced, leave-image-out), CPU or CUDA.

The canonical oracle is ``exps.nl_sgm_01a.exact.nn`` (CPU, unchanged). ``certified_nn`` reproduces its loop
structure (q_chunk 1024, r_chunk 32768, FP32 addmm screen, per-query band, FP64 direct-difference reference, lowest-id
tie-break) so that, for the identity metric on CPU, its screen values, candidate sets, distances and ids are bitwise
identical to the oracle (tested). Generalisations:

1. Metric screen. With z = M^{1/2} x (FP64, rounded once to FP32), the screen runs in z-space. The reference distance
   is D_M(q, y) = || (q - y) - U (c * U^T (q - y)) || evaluated in FP64 from the *raw* FP32 vectors. Let D' be the
   exact Euclidean distance of the rounded z vectors. Then |D' - D_M| <= eta_q with
       eta_q = u32 (||z_q|| + R_z)(1 + 1e-6) + 16 gamma_{d+k}(u64) (k + 1)(||x_q|| + R_x)
   (one FP32 rounding per coordinate plus a deliberately loose FP64 transform/reference term). If y* minimises D_M,
   D'(y*) <= min D' + 2 eta_q, so D'(y*)^2 <= min D'^2 + 4 eta_q (||z_q|| + R_z) + 4 eta_q^2 =: min D'^2 + extra_q.
   Adding extra_q to the oracle band keeps every D_M minimiser in the candidate set. For the identity metric
   (all c == 0) z == x bitwise and D' == D_M is the oracle reference, so extra_q = 0 exactly.
2. Exclusion by key (leave-image-out residuals): pairs with q_key == row_key are masked to +inf in the screen and
   never become candidates.

Assumption inherited from the oracle: the GEMM uses ordinary FP32 arithmetic (no TF32 / reduced precision / Strassen).
``screen_error_ratio`` measures the actual screen error against the bound on the device in use.
"""
from __future__ import annotations

import math

import torch

from exps.nl_sgm_01a import exact as X
from exps.nl_cad_pilot_00 import scrc as S
from exps.nl_cad_pilot_00.scrc import Metric, apply_sqrt, require

U32, U64 = X.U32, X.U64


class Rows:
    """Screen-space FP32 rows on a device, plus the raw FP32 rows used by the reference distance."""

    def __init__(self, z, ids, raw=None):
        require(z.ndim == 2 and z.dtype == torch.float32, "rows must be FP32 [n, d]")
        ids = torch.as_tensor(ids, dtype=torch.int64, device=z.device)
        require(len(ids) == len(z), "ids length")
        require(len(ids) < 2 or bool((ids[1:] > ids[:-1]).all()), "row ids must be strictly increasing")
        self.z, self.ids = z, ids
        self.raw = z if raw is None else raw
        n2 = z.double().square().sum(1)
        self.n2_32 = n2.float()
        self.R = float(n2.max().sqrt()) if len(z) else 0.0
        self.Rx = float(self.raw.double().square().sum(1).max().sqrt()) if len(z) else 0.0

    @staticmethod
    def from_rowset(rs, device="cpu"):
        return Rows(rs.x.to(device), rs.ids.to(device))

    @staticmethod
    def transformed(x, ids, metric, device="cpu"):
        """Rows in z-space for ``metric``; raw rows kept for the reference."""
        x = x.to(device)
        z = apply_sqrt(x, metric).float()
        return Rows(z.contiguous(), ids, raw=x)


def _ref(Qraw, Xraw, metric, chunk=256):
    """FP64 reference distance for aligned pairs. Batch-invariant by construction: the projections use elementwise
    products and fixed-length reductions (no GEMM, whose per-row rounding can depend on the batch shape), so the value
    of a pair never depends on which other pairs are evaluated with it (P0 finding: a GEMM-based reference differed
    by 1 ulp between batch sizes)."""
    if metric is None:
        return (Qraw.double() - Xraw.double()).square_().sum(1).sqrt_()           # == exact.reference
    return S.metric_dist_pairs(Qraw, Xraw, metric, chunk)


def _extra(zn, R, xn, Rx, metric, d):
    if metric is None or metric.is_identity:
        return torch.zeros_like(zn)
    k = metric.U.shape[1]
    g = X.gamma(d + k, U64)
    eta = U32 * (zn + R) * (1 + 1e-6) + 16 * g * (k + 1) * (xn + Rx) + 1e-300
    return 4 * eta * (zn + R) + 4 * eta * eta


def certified_nn(q, rows, metric=None, q_key=None, row_key=None, q_chunk=1024, r_chunk=32768, stats=None):
    """Exact 1-NN of raw FP32 queries ``q`` [m, d] against ``rows`` under ``metric`` (None = Euclidean).

    ``rows`` must be in the screen space of ``metric`` (``Rows.transformed``) unless metric is None/identity.
    Returns (dist float64 [m], id int64 [m]) on the rows' device; inf / -1 when no admissible row.
    """
    dev = rows.z.device
    require(q.ndim == 2 and q.dtype == torch.float32, "queries must be FP32 [m, d]")
    q = q.to(dev)
    m, d = q.shape
    out_d = torch.full((m,), math.inf, dtype=torch.float64, device=dev)
    out_i = torch.full((m,), -1, dtype=torch.int64, device=dev)
    if len(rows.z) == 0 or m == 0:
        return out_d, out_i
    zq_all = q if (metric is None or metric.is_identity) else apply_sqrt(q, metric).float()
    if q_key is not None:
        q_key = q_key.to(dev)
        row_key = row_key.to(dev)
    for a in range(0, m, q_chunk):
        Q = zq_all[a:a + q_chunk]
        Qraw = q[a:a + q_chunk]
        qn = Q.double().square().sum(1).sqrt()
        xn = Qraw.double().square().sum(1).sqrt() if metric is not None and not metric.is_identity else qn
        band = ((X.screen_band(qn, rows.R) + 4 * U32 * (rows.R ** 2 + 2 * qn * rows.R)
                 + _extra(qn, rows.R, xn, rows.Rx, metric, d)) * (1 + 1e-6)).float()
        kq = None if q_key is None else q_key[a:a + q_chunk]
        best = torch.full((len(Q),), math.inf, dtype=torch.float32, device=dev)
        cand_q, cand_r, cand_t = [], [], []
        for b in range(0, len(rows.z), r_chunk):
            Xb = rows.z[b:b + r_chunk]
            T = torch.addmm(rows.n2_32[b:b + r_chunk][None].expand(len(Q), -1), Q, Xb.T, beta=1.0, alpha=-2.0)
            if kq is not None:
                hit = kq[:, None] == row_key[b:b + r_chunk][None]
                if bool(hit.any()):
                    T.masked_fill_(hit, math.inf)
            cmin = T.min(1).values
            best = torch.minimum(best, cmin)
            keep = T <= (best + band)[:, None]
            qi, ri = keep.nonzero(as_tuple=True)
            cand_q.append(qi); cand_r.append(ri + b); cand_t.append(T[qi, ri])
        qi, ri, tv = torch.cat(cand_q), torch.cat(cand_r), torch.cat(cand_t)
        sel = tv <= (best + band)[qi]
        qi, ri = qi[sel], ri[sel]
        if kq is not None:                                   # +inf screens can pass an inf band; drop them
            ok = kq[qi] != row_key[ri]
            qi, ri = qi[ok], ri[ok]
        if stats is not None:
            stats["candidates"] = stats.get("candidates", 0) + int(len(qi))
            stats["queries"] = stats.get("queries", 0) + int(len(Q))
            stats["max_candidates_per_query"] = max(stats.get("max_candidates_per_query", 0),
                                                    int(torch.bincount(qi, minlength=len(Q)).max()) if len(qi) else 0)
        if len(qi) == 0:
            continue
        dd = torch.empty(len(qi), dtype=torch.float64, device=dev)
        for s in range(0, len(qi), 65536):
            dd[s:s + 65536] = _ref(Qraw[qi[s:s + 65536]], rows.raw[ri[s:s + 65536]], metric)
        gid = rows.ids[ri]
        order = torch.argsort(gid, stable=True)
        order = order[torch.argsort(dd[order], stable=True)]
        order = order[torch.argsort(qi[order], stable=True)]
        qs = qi[order]
        first = torch.ones(len(qs), dtype=torch.bool, device=dev)
        first[1:] = qs[1:] != qs[:-1]
        pick = order[first]
        out_d[a + qi[pick]] = dd[pick]
        out_i[a + qi[pick]] = gid[pick]
    return out_d, out_i


# ------------------------------------------------------------------ oracles and the P5 candidate route (tests)

def brute_metric(q, x, ids, metric=None, q_key=None, row_key=None):
    """All-pairs FP64 reference (small inputs only): (dist, id) with lowest id on ties."""
    ids = torch.as_tensor(ids)
    out_d = torch.full((len(q),), math.inf, dtype=torch.float64)
    out_i = torch.full((len(q),), -1, dtype=torch.int64)
    for j in range(len(q)):
        dj = _ref(q[j:j + 1].expand(len(x), -1), x, metric)
        if q_key is not None:
            dj = dj.masked_fill(row_key == q_key[j], math.inf)
        v = dj.min()
        if math.isfinite(float(v)):
            out_d[j] = v
            out_i[j] = ids[dj == v].min()
    return out_d, out_i


def candidate_route(q, x, ids, metric, K=64):
    """P5: Euclidean top-K (brute FP64) re-ranked under the metric; certified iff d_E(y_(K)) > d_E(y_(1)) / sqrt(mu_t)
    (FP64 comparison with a 1e-9 relative safety margin); uncertified queries fall back to the full metric search.
    Returns (dist, id, certified mask)."""
    ids = torch.as_tensor(ids)
    De = (q.double()[:, None, :] - x.double()[None]).square().sum(-1).sqrt()            # [m, n]
    K = min(K, x.shape[0])
    # lexicographic (distance, id) order
    order = torch.argsort(ids.expand(len(q), -1), dim=1, stable=True)
    order = torch.gather(order, 1, torch.argsort(torch.gather(De, 1, order), dim=1, stable=True))
    top = order[:, :K]
    dK = torch.gather(De, 1, order[:, K - 1:K])[:, 0]
    d1 = torch.gather(De, 1, order[:, :1])[:, 0]
    cert = dK > d1 / math.sqrt(metric.mu_t) * (1 + 1e-9) if K < x.shape[0] else torch.ones(len(q), dtype=torch.bool)
    out_d = torch.empty(len(q), dtype=torch.float64)
    out_i = torch.empty(len(q), dtype=torch.int64)
    for j in range(len(q)):
        if bool(cert[j]):
            cand = top[j]
        else:
            cand = torch.arange(x.shape[0])
        dm = _ref(q[j:j + 1].expand(len(cand), -1), x[cand], metric)
        v = dm.min()
        out_d[j] = v
        out_i[j] = ids[cand][dm == v].min()
    return out_d, out_i, cert


def screen_error_ratio(q, z):
    """max |screen - exact| / (E_q + eps_q) on the device of q/z (must be <= 1 for the band argument)."""
    dev = z.device
    n2 = z.double().square().sum(1)
    T = torch.addmm(n2.float()[None].expand(len(q), -1), q, z.T, beta=1.0, alpha=-2.0).double()
    exact = n2[None] - 2 * (q.double() @ z.double().T)
    qn = q.double().square().sum(1).sqrt()
    band = X.screen_band(qn, float(n2.max().sqrt())) / 2
    return float(((T - exact).abs() / band[:, None]).max().to("cpu"))
