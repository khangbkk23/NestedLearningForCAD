"""NL-DRIFT-00 estimators (prereg section 2). Pure functions on arrays; no file access.

Shapes: nc categories, 15 boundaries, 2 coordinates (m, l). ``a`` [nc] acquisition indices. Arrays hold NaN where
undefined (t < a for measurements; t <= a for covariates). Every estimator returns est [nc, 15, 2] with values only at
a < t <= 14 (NaN elsewhere); the acquisition estimate is s_a = (0, 0) by construction of the coordinates.
"""
from __future__ import annotations

import numpy as np
import torch

from exps.nl_drift_00 import common as C

NB = 15


def _blank(nc):
    return np.full((nc, NB, 2), np.nan)


def e0(a):
    est = _blank(len(a))
    for i, ai in enumerate(a):
        est[i, ai + 1:] = 0.0
    return est


def e2(y, a):
    """Direct witness recalibration: s_t = y_t."""
    est = _blank(len(a))
    for i, ai in enumerate(a):
        est[i, ai + 1:] = y[i, ai + 1:]
    return est


def e2a(ya, a):
    """Anchored direct recalibration: s_t = y_t - y_a (ya is already anchored)."""
    return e2(ya, a)


def e3(ya, a, g=C.G_E3):
    """Fixed-gain filter on the anchored measurement, s_a = 0."""
    est = _blank(len(a))
    for i, ai in enumerate(a):
        s = np.zeros(2)
        for t in range(ai + 1, NB):
            s = s + g * (ya[i, t] - s)
            est[i, t] = s
    return est


def shuffle_covariates(u, a):
    """E5: at each boundary t, cyclically shift the non-constant covariates among the old categories (a < t, in
    acquisition order). With a single old category the covariates are unchanged."""
    out = u.copy()
    order = np.argsort(a, kind="stable")
    for t in range(NB):
        old = [i for i in order if a[i] < t]
        k = len(old)
        if k >= 2:
            for j, i in enumerate(old):
                out[i, t, 1:] = u[old[(j + 1) % k], t, 1:]
    return out


# ------------------------------------------------------------------ learned Kalman filter (E4, E4-0, E5)

def n_params(k):
    return 2 * k + 4


_LO, _HI = C.LOG_BOUNDS
_RAW0 = float(np.log((0.0 - _LO) / (_HI - 0.0)))          # raw value giving kappa = rho = 0


def default_phi(k):
    """Raw parameter vector: F = 0 and kappa = rho = 0 (q = r = R_sub)."""
    p = np.zeros(n_params(k))
    p[2 * k:] = _RAW0
    return p


def _unpack(p, k):
    """kappa, rho = lo + (hi - lo) * sigmoid(raw): hard bounds [lo, hi] (prereg section 2), no overflow."""
    F = p[:2 * k].reshape(2, k)
    raw = p[2 * k:2 * k + 4]
    if isinstance(p, torch.Tensor):
        b = _LO + (_HI - _LO) * torch.sigmoid(raw)
    else:
        b = _LO + (_HI - _LO) / (1.0 + np.exp(-raw))
    return F, b[:2], b[2:]


def physical(p, k):
    F, kappa, rho = _unpack(np.asarray(p), k)
    return {"F": F.tolist(), "kappa": kappa.tolist(), "rho": rho.tolist()}


def _nll(p, ya, u, rs, a, t_max, k):
    """Innovation NLL over all transitions a < t <= t_max, recursion restarted from (0, 0) with one phi; torch FP64.
    Inputs are already sliced to boundaries <= t_max by the caller (prefix-only)."""
    F, kappa, rho = _unpack(p, k)
    q = torch.exp(kappa)[None] * rs                        # [nc, 2]
    r = torch.exp(rho)[None] * rs
    nc = ya.shape[0]
    s = torch.zeros(nc, 2, dtype=torch.float64)
    P = torch.zeros(nc, 2, dtype=torch.float64)
    total = torch.zeros((), dtype=torch.float64)
    for t in range(1, t_max + 1):
        act = a < t
        if not bool(act.any()):
            continue
        ut = torch.nan_to_num(u[:, t])                     # inactive rows are masked below
        sm = s + ut @ F.T
        Pm = P + q
        S = Pm + r
        yt = torch.nan_to_num(ya[:, t])
        nu = yt - sm
        total = total + ((nu.square() / S + torch.log(S)) * act[:, None]).sum()
        K = Pm / S
        s = torch.where(act[:, None], sm + K * nu, s)
        P = torch.where(act[:, None], (1 - K) * Pm, P)
    return total + C.RIDGE * F.square().sum() + C.PRIOR_LOGNOISE * (kappa.square().sum() + rho.square().sum())


def fit_phi(p0, ya, u, rs, a, t_max, k):
    """phi_t: argmin of the prefix NLL (boundaries <= t_max). Arrays are sliced here so that nothing after t_max can
    influence the fit."""
    ya_t = torch.from_numpy(np.ascontiguousarray(ya[:, :t_max + 1])).double()
    u_t = torch.from_numpy(np.ascontiguousarray(u[:, :t_max + 1, :k])).double()
    rs_t = torch.from_numpy(rs).double()
    a_t = torch.from_numpy(np.asarray(a))
    p = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([p], lr=1.0, max_iter=200, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        L = _nll(p, ya_t, u_t, rs_t, a_t, t_max, k)
        L.backward()
        return L

    opt.step(closure)
    L = _nll(p, ya_t, u_t, rs_t, a_t, t_max, k)
    g = torch.autograd.grad(L, p)[0]
    out = p.detach().numpy().copy()
    return out, {"nll": float(L.detach()), "grad_norm": float(g.norm()), "n_iter": int(opt.state[p]["n_iter"]),
                 "finite": bool(np.isfinite(out).all() and np.isfinite(float(L.detach())))}


def n_transitions(a, t):
    return int(sum(max(0, t - int(ai)) for ai in a))


def learned(ya, u, rs, a, k, phis=None):
    """Online learned filter. If ``phis`` (dict t -> phi) is given, filtering only (cohort U uses cohort T's phis);
    otherwise phi_t is refitted after every boundary t on the prefix <= t and used at t+1.

    Returns est [nc,15,2], phis {t: phi_t}, diagnostics."""
    a = np.asarray(a)
    nc = len(a)
    est = _blank(nc)
    s = np.zeros((nc, 2))
    P = np.zeros((nc, 2))
    phi = default_phi(k)
    fitted = {}
    diag = {"fits": {}, "min_P": np.inf, "min_K": np.inf, "max_K": -np.inf, "finite": True}
    for t in range(1, NB):
        phi_use = phis[t - 1] if phis is not None else phi
        F, kappa, rho = _unpack(phi_use, k)
        q = np.exp(kappa)[None] * rs
        r = np.exp(rho)[None] * rs
        act = a < t
        if act.any():
            ut = np.nan_to_num(u[:, t, :k])
            sm = s + ut @ F.T
            Pm = P + q
            S = Pm + r
            K = Pm / S
            nu = np.nan_to_num(ya[:, t]) - sm
            sn = sm + K * nu
            Pn = (1 - K) * Pm
            s[act], P[act] = sn[act], Pn[act]
            est[act, t] = sn[act]
            diag["min_P"] = min(diag["min_P"], float(Pn[act].min()))
            diag["min_K"] = min(diag["min_K"], float(K[act].min()))
            diag["max_K"] = max(diag["max_K"], float(K[act].max()))
            diag["finite"] &= bool(np.isfinite(sn[act]).all() and np.isfinite(Pn[act]).all())
        if phis is None:
            if n_transitions(a, t) >= C.MIN_PREFIX:
                phi, info = fit_phi(default_phi(k), ya, u, rs, a, t, k)      # cold start every boundary
                diag["fits"][t] = info
                diag["finite"] &= info["finite"]
            fitted[t] = phi.copy()
    fitted[0] = default_phi(k) if phis is None else phis[0]
    return est, (fitted if phis is None else phis), diag
