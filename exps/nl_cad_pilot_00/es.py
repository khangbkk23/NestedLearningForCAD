"""Slow-level machinery for P0: the fast-level stream driver, antithetic ES with Adam, and a synthetic environment.

The synthetic environment exists only for mechanism verification (T8 and the end-to-end wiring check). Its
"anomalies" are synthetic shifts; no source or target label is read anywhere in P0.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from exps.nl_cad_pilot_00 import nnx
from exps.nl_cad_pilot_00 import scrc as S


# ------------------------------------------------------------------ fast-level stream (theta-dependent part only)

def run_fast_stream(residuals, policy, r=S.RANK, state=None, start_t=1):
    """residuals: list of R_t [m, d] (theta-independent, precomputed). Returns per-boundary records with the persisted
    state after each boundary. ``state``/``start_t`` resume an interrupted stream."""
    st = S.FastState.empty(residuals[0].shape[1], r) if state is None else state
    out = []
    for i, R in enumerate(residuals[start_t - 1:], start=start_t):
        if i >= 2:
            f = S.novelty_features(st, R, i)
            h = f["h"]
        else:
            f, h = {"nu": float("nan"), "a": float("nan")}, None
        rho = policy(i, h)
        st, dg = S.prox_update(st, R, rho, r)
        out.append({"t": i, "rho": rho, "h": h, "nu": f["nu"], "a": f["a"], "state": st, "diag": dg})
    return out


# ------------------------------------------------------------------ ES

class Adam:
    def __init__(self, n, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m, self.v, self.k = np.zeros(n), np.zeros(n), 0
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def ascend(self, theta, g):
        self.k += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.k)
        vh = self.v / (1 - self.b2 ** self.k)
        return theta + self.lr * mh / (np.sqrt(vh) + self.eps)


def es_optimize(J, theta0, iters=60, P=16, sigma=0.1, lr=0.05, seed=20261115):
    """Antithetic Gaussian ES (spec 4.5): g = 1/(2 P sigma) sum_p [J(th + s e_p) - J(th - s e_p)] e_p, Adam ascent.
    Noise of iteration i comes from numpy PCG64(seed + i). Last iterate returned; full trajectory recorded."""
    th = np.asarray(theta0, dtype=np.float64).copy()
    opt = Adam(len(th), lr)
    traj = [{"iter": 0, "theta": th.tolist(), "J": float(J(th)), "evals": 1}]
    evals = 1
    for i in range(1, iters + 1):
        eps = np.random.Generator(np.random.PCG64(seed + i)).standard_normal((P, len(th)))
        g = np.zeros(len(th))
        for p in range(P):
            g += (J(th + sigma * eps[p]) - J(th - sigma * eps[p])) * eps[p]
        evals += 2 * P
        g /= 2 * P * sigma
        th = opt.ascend(th, g)
        traj.append({"iter": i, "theta": th.tolist(), "grad": g.tolist(), "evals": evals})
    traj[-1]["J_last"] = float(J(th))
    return th, traj


# ------------------------------------------------------------------ synthetic environment (mechanism check only)

class SynthWorld:
    """T categories in R^d. Normal patch = mu_c + nuisance (shared low-rank + category low-rank) + isotropic noise;
    anomalous test patch = normal patch + shift along a defect direction drawn mostly orthogonal to the nuisance.
    Bank rows (theta-independent) = all patches of 'bank' images; residual sample = patches of other images."""

    def __init__(self, seed=0, d=S.D_FEAT, T=4, imgs=6, ppi=60, bank_imgs=4, n_test=160, frac_anom=0.25,
                 shared_rank=24, cat_rank=8, nuis=1.5, noise=0.35, defect=1.6):
        g = torch.Generator().manual_seed(seed)
        self.d, self.T = d, T
        Ush = torch.linalg.qr(torch.randn(d, shared_rank, generator=g, dtype=torch.float64))[0]
        self.banks, self.res, self.res_keys, self.tests = [], [], [], []
        gid = 0
        for c in range(T):
            mu = torch.randn(d, generator=g, dtype=torch.float64) * 2.0
            Uc = torch.linalg.qr(torch.randn(d, cat_rank, generator=g, dtype=torch.float64))[0]

            def normals(n):
                a = torch.randn(n, shared_rank, generator=g, dtype=torch.float64) * nuis
                b = torch.randn(n, cat_rank, generator=g, dtype=torch.float64) * nuis
                e = torch.randn(n, d, generator=g, dtype=torch.float64) * noise / math.sqrt(d) * math.sqrt(d / 64)
                return mu + a @ Ush.T + b @ Uc.T + e

            x = normals(imgs * ppi).float()
            keys = torch.arange(imgs).repeat_interleave(ppi) + 1000 * c
            ids = torch.arange(gid, gid + len(x))
            gid += len(x)
            inbank = keys < 1000 * c + bank_imgs
            self.banks.append((x[inbank].contiguous(), ids[inbank], keys[inbank]))
            self.res.append((x.contiguous(), keys))
            xt = normals(n_test)
            lab = torch.rand(n_test, generator=g) < frac_anom
            dirs = torch.randn(int(lab.sum()), d, generator=g, dtype=torch.float64)
            dirs -= (dirs @ Ush) @ Ush.T
            dirs /= dirs.norm(dim=1, keepdim=True)
            xt[lab] += dirs * defect
            self.tests.append((xt.float().contiguous(), lab.numpy().astype(int)))

    def bank_at(self, t):
        """Union bank after boundary t (1-based): the theta-independent Level-0 analogue."""
        x = torch.cat([self.banks[k][0] for k in range(t)])
        ids = torch.cat([self.banks[k][1] for k in range(t)])
        keys = torch.cat([self.banks[k][2] for k in range(t)])
        return x, ids, keys

    def residuals(self, m=None):
        """Leave-image-out Euclidean NN residuals of each task's patches against the union bank after that task."""
        out = []
        for t in range(1, self.T + 1):
            x, ids, keys = self.bank_at(t)
            q, qk = self.res[t - 1]
            if m is not None:
                q, qk = q[:m], qk[:m]
            rows = nnx.Rows(x, ids)
            d, i = nnx.certified_nn(q, rows, None, q_key=qk, row_key=keys)
            pos = torch.searchsorted(ids, i)
            out.append((q.double() - x[pos].double()).float())
        return out


def episode_J(world, residuals, theta, policy_kind="learned"):
    """Mean patch AP over all (t, tau <= t) pairs of one synthetic episode under theta (spec 4.4 form)."""
    th = S.Theta.from_vector(theta)
    pol = {"learned": S.Learned(th.rho), "cumulative": S.Cumulative()}[policy_kind]
    recs = run_fast_stream(residuals, pol)
    aps = []
    for t in range(1, world.T + 1):
        x, ids, _ = world.bank_at(t)
        met = S.make_metric(recs[t - 1]["state"], th.tau, th.gamma)
        rows = nnx.Rows.transformed(x, ids, met)
        for tau in range(1, t + 1):
            xt, lab = world.tests[tau - 1]
            d, _ = nnx.certified_nn(xt, rows, met)
            aps.append(average_precision_score(lab, d.numpy()))
    return float(np.mean(aps))
