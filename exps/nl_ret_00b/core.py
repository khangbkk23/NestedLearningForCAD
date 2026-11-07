"""NL-RET-00B core: exact, memory-bounded native AP primitives (pure NumPy; no I/O).

AP definition = sklearn.metrics.average_precision_score (non-interpolated, tied scores grouped):
    AP = sum_k (R_k - R_{k-1}) * P_k over descending distinct thresholds.
Only thresholds that contain at least one positive change recall, so with positives P and negatives N
    AP(P, N) = (1/|P|) * sum_{u in uniq(P) desc} cnt_P(u) * TP(>=u) / (TP(>=u) + FP(>=u)),
where FP(>=u) counts negatives with score >= u (ties grouped). Negatives are kept as ascending-sorted arrays
(possibly several disjoint parts, e.g. test/good pixels and anomalous-image background) and counted with
searchsorted, so P and N may come from *different* model states (the factorial cells) without materialising a
merged array.
"""
from __future__ import annotations

import numpy as np


def ap_pn(pos, negs_sorted):
    """Exact AP of positives `pos` (any order) against negatives given as ascending-sorted arrays."""
    pos = np.asarray(pos)
    if pos.size == 0:
        raise ValueError("no positives")
    u, cnt = np.unique(pos, return_counts=True)          # ascending
    u, cnt = u[::-1], cnt[::-1]
    tp = np.cumsum(cnt, dtype=np.float64)
    fp = np.zeros(len(u), dtype=np.float64)
    for n in negs_sorted:
        if n.size:
            fp += n.size - np.searchsorted(n, u.astype(n.dtype), side="left")
    return float(np.sum(cnt * (tp / (tp + fp))) / tp[-1])


def shapley(ap_aa, ap_fa, ap_af, ap_ff):
    """Two-factor symmetric decomposition. ap_xy = AP(P_x, N_y). Returns D, D_P, D_N (D = AA - FF)."""
    d = ap_aa - ap_ff
    dp = 0.5 * ((ap_aa - ap_fa) + (ap_af - ap_ff))
    dn = 0.5 * ((ap_aa - ap_af) + (ap_fa - ap_ff))
    return d, dp, dn


def total(parts):
    return int(sum(p.size for p in parts))


def count_ge(parts, t):
    return int(sum(p.size - np.searchsorted(p, np.asarray(t, dtype=p.dtype), side="left") for p in parts))


def order_stat(parts, k):
    """Exact k-th smallest (0-based) value of the union of ascending-sorted non-negative float32 arrays,
    by bisection on the IEEE bit pattern (monotone for non-negative floats)."""
    n = total(parts)
    if not 0 <= k < n:
        raise ValueError("k out of range")
    lo, hi = 0, 0x7F800000                                   # [0, +inf) as uint32 bits
    while lo < hi:                                           # smallest v with count(<= v) >= k + 1
        mid = (lo + hi) // 2
        v = np.array([mid], dtype=np.uint32).view(np.float32)[0]
        le = sum(int(np.searchsorted(p, v, side="right")) for p in parts)
        if le >= k + 1:
            hi = mid
        else:
            lo = mid + 1
    return float(np.array([lo], dtype=np.uint32).view(np.float32)[0])


def quantiles(parts, qs):
    """Lower-type empirical quantiles (index floor(q*(n-1))) of the union of sorted parts."""
    parts = [p for p in parts if p.size]
    n = total(parts)
    if len(parts) == 1:
        p = parts[0]
        return {q: float(p[int(np.floor(q * (n - 1)))]) for q in qs}
    return {q: order_stat(parts, int(np.floor(q * (n - 1)))) for q in qs}


def mean_parts(parts):
    n = total(parts)
    return float(sum(float(p.sum(dtype=np.float64)) for p in parts) / n)


def pct_rank(img):
    """Per-image monotone transform: empirical CDF F(x) = #{pixels <= x} / n. Preserves within-image order and
    ties; maps every image to (0, 1]."""
    flat = img.ravel()
    s = np.sort(flat)
    return (np.searchsorted(s, flat, side="right") / flat.size).astype(np.float32).reshape(img.shape)
