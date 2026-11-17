"""Farthest-point selection for the C2-DOM sweep: canonical decisions, GPU-accelerated (amendments 01-02).

The canonical rule is ``exps.csg01.select.select_fps`` (CPU, FP64). ``select``:

* small calls (n * k < GPU_MIN_WORK), no CUDA or allow_gpu=False: the canonical function itself;
* large calls: the same algorithm with the FP64 matrix-vector step on the GPU. The first point is computed on the
  CPU with the canonical code (identical mean/argmin). Two FP64 evaluations (any summation order) of
  ``||x||^2 - 2 x.v + v.v`` differ by at most 2 (gamma_768 + 2u)(||x|| + ||v||)^2 <= 8 (gamma_768 + 2u) R^2 and a
  running minimum preserves that bound, so when the largest running min-distance exceeds every other by more than
  BAND = 32 (gamma_768 + u) R^2 the canonical CPU argmax picks the same index (certified step).
* Amendment 02 (ambiguous steps, gap <= BAND; measured ~1 per 2e4-7e4 steps, mostly exact-duplicate rows): the tie set
  (rows within BAND of the top among the top-8) is resolved by recomputing each candidate's running minimum exactly as
  the FPS definition states, min over all selected rows of the direct-difference FP64 squared distance (relative
  error ~1e-13, far below BAND); the largest wins, lowest index on equality (bitwise-identical rows -> lowest index =
  canonical first-occurrence). Here the canonical function's own choice is decided by its rounding; the two can differ
  only at these steps. Every ambiguous step is recorded (count, gap, whether the resolution changed the GPU top-1).
  A tie set filling all 8 slots, or a NaN, falls back to the canonical CPU function.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from exps.csg01.select import _dist_to_vector, select_fps

GPU_MIN_WORK = 2e8
TOPM = 8
U64 = 2.0 ** -53
G768 = 768 * U64 / (1 - 768 * U64)


def band(R2):
    return 32.0 * (G768 + U64) * R2


class _Fallback(Exception):
    pass


def _gpu(D, k):
    n = D.shape[0]
    D = D.to(torch.float64)
    D2 = (D * D).sum(dim=1)
    mean = D.mean(dim=0)
    first = int(_dist_to_vector(D, mean, D2).argmin().item())       # canonical first point (CPU)
    b = band(float(D2.max()))
    G, G2 = D.cuda(), D2.cuda()
    sel = torch.empty(k, dtype=torch.long, device="cuda")
    sel[0] = first
    v = G[first]
    md = (G2 - 2.0 * (G @ v) + (v @ v)).clamp_min_(0.0)
    md[first] = -1.0
    m = min(TOPM, n)
    amb = []
    for s in range(1, k):
        tv, ti = torch.topk(md, m)
        if float(tv[0] - tv[1]) <= b:                                   # ambiguous step (host check)
            tie = (tv[0] - tv) <= b
            if not bool(torch.isfinite(tv).all()):
                raise _Fallback()
            cand = ti[tie]
            if bool(tie[-1]):                                           # amendment 04: full tie set
                cand = torch.nonzero(md >= tv[0] - b).flatten()
            uq, inv = torch.unique(G[cand], dim=0, return_inverse=True)  # identical vectors share one exact value
            S = G[sel[:s]]
            ex = torch.stack([(S - u).square().sum(1).min() for u in uq])  # exact running minima per vector
            if float(ex.max()) <= b:                                     # only copies of selected rows remain:
                raise _Fallback()                                        # canonical order is rounding-defined
            win = int(cand[ex[inv] == ex.max()].min())
            amb.append({"step": s, "gap": float(tv[0] - tv[1]), "tie_size": int(len(cand)),
                        "identical_vectors": len(uq) == 1, "changed_top1": win != int(ti[0])})
            j = torch.tensor(win, device="cuda")
        else:
            j = ti[0]
        sel[s] = j
        v = G[j]
        md = torch.minimum(md, (G2 - 2.0 * (G @ v) + (v @ v)).clamp_min_(0.0))
        md[j] = -1.0
    out = sel.cpu().numpy().astype(np.int64)
    peak = torch.cuda.max_memory_allocated()
    del G, G2, md
    torch.cuda.empty_cache()
    return out, {"ambiguous_steps": len(amb), "ambiguous_distinct": sum(not a["identical_vectors"] for a in amb),
                 "ambiguous_changed_top1": sum(a["changed_top1"] for a in amb), "ambiguous": amb[:50], "band": b,
                 "vram_peak_bytes": int(peak)}


def select(x, k, allow_gpu=True):
    """Indices of select_fps(x, k) (see module doc for ambiguous steps); returns (indices, record)."""
    n = int(x.shape[0])
    t0 = time.time()
    rec = {"n": n, "k": int(k)}
    if k > n:
        raise ValueError(f"cannot select {k} from {n} candidates")
    use_gpu = allow_gpu and torch.cuda.is_available() and n * k >= GPU_MIN_WORK and 1 < k < n
    idx = None
    if use_gpu:
        try:
            idx, info = _gpu(x, k)
            rec.update(info)
            rec["impl"] = "gpu_certified" if info["ambiguous_steps"] == 0 else "gpu_certified_exact_ties"
            if len(np.unique(idx)) != k:
                idx = None
        except _Fallback:
            idx = None
            torch.cuda.empty_cache()
        if idx is None:
            rec["fallback"] = True
    if idx is None:
        idx = select_fps(x, k)
        rec["impl"] = "cpu_canonical_fallback" if use_gpu else "cpu_canonical"
    rec["seconds"] = time.time() - t0
    return np.asarray(idx, dtype=np.int64), rec
