"""NL-SGM-01A-AM02/v2 — exact support distances, frozen metric definitions (plan v2 §4) and verdict (§6).

Pure functions; the runner (am02_run.py) supplies arrays. Units: one value per whole EVAL image unless stated.
"""
from __future__ import annotations

import numpy as np
import torch

from exps.nl_sgm_01a import exact as X
from exps.nl_sgm_01a import stats as ST
from exps.task_p.p0_normal_support import require

DELTA, EPS = 0.02, 0.05          # operational normal-support margins (q50 units); NOT anomaly-AP thresholds
TOL = 0.1                        # coverage / under-fraction tolerance (q50 units)
DEGENERATE = 1e-12
SEEDS = {"primary": 20261106, "nested": 20261107}
BOOT = 2000


# ------------------------------------------------------------------ exact distances

def rowset(x, gids):
    order = torch.argsort(gids)
    return X.RowSet(x[order].contiguous().float(), gids[order])


def by_owner(x, gids, owner):
    return {int(c): rowset(x[owner == c], gids[owner == c]) for c in torch.unique(owner).tolist()}


def probe_nn(Q, rs):
    """Q [n_img, 784, 768] FP32 -> (dist float64 [n_img, 784], id int64 [n_img, 784]) exact."""
    n, p, d = Q.shape
    dist, ids = X.nn(Q.reshape(n * p, d).contiguous(), rs)
    return dist.reshape(n, p), ids.reshape(n, p)


def spot_check(q, rs, k=24):
    """Exactness audit: certified nn vs direct FP64 reference on every row for k queries (chunk-free brute)."""
    q = q[:k].contiguous()
    d, i = X.nn(q, rs)
    for j in range(len(q)):
        D = X.reference(q[j][None], rs.x)
        v = D.min()
        if not (float(v) == float(d[j]) and int(rs.ids[D == v].min()) == int(i[j])):
            return False
    return True


# ------------------------------------------------------------------ metrics (§4)

def scales(d_sel):
    v = np.asarray(d_sel, float).ravel()
    return float(np.median(v)), float(np.quantile(v, 0.9))


def score(d):
    return np.asarray(d, float).max(axis=-1)                         # max-patch image score


def excess(d_arm, d_ref, q50):
    """Signed per-image excess (s_arm - s_ref) / q50."""
    return (score(d_arm) - score(d_ref)) / q50


def retention(d_acq, d_fin, d_ref, q50):
    """R per image = e(final) - e(acquisition); the frozen reference cancels exactly."""
    return excess(d_fin, d_ref, q50) - excess(d_acq, d_ref, q50)


def coverage(d_arm, d_ref, q50, mask=None):
    ok = np.asarray(d_arm) <= np.asarray(d_ref) + TOL * q50
    if mask is None:
        return ok.mean(axis=-1)
    m = np.asarray(mask)
    return np.array([ok[i][m[i]].mean() if m[i].any() else np.nan for i in range(len(ok))])


def under_fraction(d_arm, d_ref, q50):
    return (np.asarray(d_arm) < np.asarray(d_ref) - TOL * q50).mean(axis=-1)


def false_support(d_arm, d_ref, q50):
    return (score(d_arm) < score(d_ref) - TOL * q50).astype(float)


def survival(ids_t, ids_acq):
    a = set(int(v) for v in ids_acq)
    return len(a & set(int(v) for v in ids_t)) / max(len(a), 1)


def jaccard(a, b):
    a, b = set(int(v) for v in a), set(int(v) for v in b)
    return len(a & b) / max(len(a | b), 1)


def capture(nn_owner, c):
    return (np.asarray(nn_owner) != c).mean(axis=-1)


# ------------------------------------------------------------------ inference (§5)

def ci_both(values, cats):
    """Primary category-cluster and nested bootstrap of the macro mean (paired values per image)."""
    return {"primary": ST.clustered_ci(values, cats, nested=False, b=BOOT, seed=SEEDS["primary"]),
            "nested": ST.clustered_ci(values, cats, nested=True, b=BOOT, seed=SEEDS["nested"])}


def _both(s, cond):
    return all(cond(s[k]) for k in ("primary", "nested"))


def verdict(st, blocked=False, delta=DELTA, eps=EPS):
    """st: {R_C1, A_C1, R_C2, A_C2, dR, dA} each {'primary': {lo, hi, point}, 'nested': {...}}. Plan v2 §6."""
    if blocked:
        return "AM02_BLOCKED"
    if _both(st["R_C1"], lambda s: s["hi"] <= delta) and _both(st["A_C1"], lambda s: s["hi"] <= eps):
        return "NO_RETENTION_PROBLEM"
    if _both(st["dR"], lambda s: s["lo"] > 0) and _both(st["dA"], lambda s: s["lo"] > 0):
        if _both(st["R_C2"], lambda s: s["hi"] <= delta) and _both(st["A_C2"], lambda s: s["hi"] <= eps):
            return "CONVENTIONAL_PQ_SUFFICIENT"
        if _both(st["R_C2"], lambda s: s["lo"] > delta) or _both(st["A_C2"], lambda s: s["lo"] > eps):
            return "RESIDUAL_GAP_CONSOLIDATION_CANDIDATE"
        return "HEADROOM_ESTABLISHED_RESIDUAL_INCONCLUSIVE"
    if _both(st["dR"], lambda s: s["hi"] <= delta) and _both(st["dA"], lambda s: s["hi"] <= eps):
        return "NO_COMPRESSION_HEADROOM"
    return "HEADROOM_NOT_ESTABLISHED"


def codebook_limited(q_abs_final_macro, a_c2_point, rho_ratio_old):
    require(np.isfinite(a_c2_point), "A(C2) not finite")
    return bool(q_abs_final_macro >= 0.5 * a_c2_point or max(rho_ratio_old) >= 2.0)
