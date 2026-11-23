"""NANM-NL-02 Stage 0 mechanics (design-correction phase; no real-data run).

Conventions (task: agents/tasks/task_nanm_nl_02_stage0.md):
- v: unit-norm fp16 ViT token (G3 key convention), promoted to fp64 for all learning/likelihood arithmetic.
- Row-vector maps. G3 whitening z = v W (W symmetric PD, canonical B.whiten). Fast metric acts as z' = z A^T with
  A = I + P Q^T, so the effective linear map is L = W A^T and the v-space precision is M = W A^T A W.
- The canonical G3 oracle (nanm.g3 -> exps.nanm_r0/r2/r2b) is called, never modified.
"""
from __future__ import annotations

import hashlib
import math

import torch

from exps.nanm_r2 import run as R2
from nanm import g3

D = 768
RANK = 16
ALPHA = R2.SHRINK                 # 0.5, G3 shrinkage, reused as the fast objective's prior (see task §C)
U_FLOOR = 1e-3                    # nats / patch, registered numerical floor of the effect scale
LOG2PI = math.log(2 * math.pi)
F64 = torch.float64


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ------------------------------------------------------------------ Correction 1: non-degenerate identity init
def init_phi(seed: int, rank: int = RANK, d: int = D):
    """P ~ N(0, 1/d) (seeded), Q = 0. Then P Q^T = 0 exactly, so A = I exactly, while dL/dQ = G^T P != 0."""
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(d, rank, generator=g, dtype=F64) / math.sqrt(d)
    Q = torch.zeros(d, rank, dtype=F64)
    return P, Q


def A_of(P, Q):
    return torch.eye(P.shape[0], dtype=P.dtype) + P @ Q.T


def logdet_A(P, Q):
    """(sign, log|det A|) via the determinant lemma det(I_d + P Q^T) = det(I_r + Q^T P); O(r^3)."""
    return torch.linalg.slogdet(torch.eye(P.shape[1], dtype=P.dtype) + Q.T @ P)


# ------------------------------------------------------------------ fast objective
def assign(Zq, Zk, img_q, img_k, P, Q):
    """Nearest key under the current metric, leave-own-image-out. Piecewise-constant: no gradient (as in G3)."""
    with torch.no_grad():
        At = A_of(P, Q).T
        d = R2.sqdist(Zq @ At, Zk @ At)
        d.masked_fill_(img_q[:, None] == img_k[None], float('inf'))
        return d.argmin(1)


def fast_loss(P, Q, Vq, Vk, img_q, img_k, W, j=None):
    """Penalised kNN residual quasi-likelihood (task §C.3), up to the constant -log det W:
       L = 1/2 (1-a) mean||r_z A^T||^2 + 1/2 a (mean||r_v||^2 / d) ||W A^T||_F^2 - log|det A|.
    Its minimiser over the full v-space precision M = W A^T A W, for fixed assignment, is shrink(C_hat)^{-1}."""
    Zq, Zk = Vq @ W, Vk @ W
    if j is None:
        j = assign(Zq, Zk, img_q, img_k, P, Q)
    A = A_of(P, Q)
    rz = Zq - Zk[j]
    rv2 = ((Vq - Vk[j]) ** 2).sum(1).mean()
    quad = 0.5 * (1 - ALPHA) * ((rz @ A.T) ** 2).sum(1).mean()
    pen = 0.5 * ALPHA * (rv2 / Vq.shape[1]) * ((W @ A.T) ** 2).sum()
    sign, la = logdet_A(P, Q)
    return quad + pen - la, j, sign


def fast_loss_w(P, Q, vq, zq, img_q, w, Kv, Zk, kimg, W2, trW2):
    """Efficient weighted form of fast_loss (identical value for uniform w; tested):
    ||W A^T||_F^2 = tr(W^2) + 2 sum(P * (W^2 Q)) + sum((Q^T W^2 Q) * (P^T P)); A z = z + (z Q) P^T.
    vq/zq: queries in v / z space (fp64); w: weights summing to 1; Kv/Zk: keys in v / z space; W2 = W @ W."""
    with torch.no_grad():
        d = R2.sqdist(zq + (zq @ Q) @ P.T, Zk + (Zk @ Q) @ P.T)
        d.masked_fill_(img_q[:, None] == kimg[None], float('inf'))
        j = d.argmin(1)
    rz = zq - Zk[j]
    rv2 = ((vq - Kv[j]) ** 2).sum(1)
    rzA = rz + (rz @ Q) @ P.T
    quad = 0.5 * (1 - ALPHA) * (w * (rzA ** 2).sum(1)).sum()
    c = ALPHA * (w * rv2).sum() / vq.shape[1]
    trA = trW2 + 2 * (P * (W2 @ Q)).sum() + ((Q.T @ W2 @ Q) * (P.T @ P)).sum()
    sign, la = logdet_A(P, Q)
    return quad + 0.5 * c * trA - la, j


def Lmap_phi(W, P, Q):
    """Effective v -> z' map W A^T = W + (W Q) P^T and its log|det|."""
    return W + (W @ Q) @ P.T, float(torch.linalg.slogdet(W)[1]) + float(logdet_A(P, Q)[1])


def grads(P, Q, Vq, Vk, img_q, img_k, W, j=None):
    P = P.detach().requires_grad_(True); Q = Q.detach().requires_grad_(True)
    L, j, sign = fast_loss(P, Q, Vq, Vk, img_q, img_k, W, j)
    gP, gQ = torch.autograd.grad(L, (P, Q))
    return float(L.detach()), gP, gQ, j


def sgd_step(P, Q, gP, gQ, lr, max_halvings=10):
    """Plain SGD (no optimizer state). Backtracks if det A would be <= 0 or non-finite. Returns (P, Q, halvings);
    halvings = -1 means the step was rejected and the state is unchanged."""
    for h in range(max_halvings + 1):
        P2, Q2 = P - lr * gP, Q - lr * gQ
        s, la = logdet_A(P2, Q2)
        if float(s) > 0 and math.isfinite(float(la)):
            return P2, Q2, h
        lr /= 2
    return P, Q, -1


# ------------------------------------------------------------------ Correction 3: likelihood under one reference measure
def nll_mixture(Vq, Vk, Lmap, logabsdet_L):
    """Per-query NLL (nats) of the proper equal-weight Gaussian mixture density on R^d (Lebesgue reference measure):
       p(q) = 1/|K| sum_m N(q; m, M^{-1}),  M = Lmap Lmap^T (row convention z = v Lmap).
       -log p = -logsumexp_m(-1/2 ||(q-m) Lmap||^2) + log|K| - log|det Lmap| + d/2 log 2pi.
    log|det Lmap| is the log-Jacobian of the coordinate map; every model is therefore a density on the same v-space."""
    d = R2.sqdist(Vq @ Lmap, Vk @ Lmap)
    return (-torch.logsumexp(-0.5 * d, 1) + math.log(len(Vk)) - logabsdet_L + 0.5 * Vq.shape[1] * LOG2PI)


def nll_maxcomp(Vq, Vk, Lmap, logabsdet_L):
    """Max-component (min-distance) quasi-NLL; NLL_mix <= NLL_max <= NLL_mix + log|K|. Secondary only."""
    d = R2.sqdist(Vq @ Lmap, Vk @ Lmap)
    return 0.5 * d.min(1).values + math.log(len(Vk)) - logabsdet_L + 0.5 * Vq.shape[1] * LOG2PI


def maps_for(W, P=None, Q=None, sigma2=None):
    """(Lmap, log|det Lmap|) for: isotropic (sigma2), G3 (W), adapted (W A^T)."""
    if sigma2 is not None:
        d = W.shape[0] if W is not None else D
        return torch.eye(d, dtype=F64) / math.sqrt(sigma2), -0.5 * d * math.log(sigma2)
    lw = float(torch.linalg.slogdet(W)[1])
    if P is None:
        return W, lw
    return W @ A_of(P, Q).T, lw + float(logdet_A(P, Q)[1])


def iso_sigma2(S, n, d=D):
    """Isotropic MLE from the canonical G3 residual statistic: sigma^2 = tr(S) / (n d)."""
    return float(torch.trace(S)) / (n * d)


def effect_scale(u_values, floor=U_FLOOR):
    """Registered U = max(median(DESIGN-only per-task NLL_iso - NLL_G3), floor). Returns (U, raw, floor_bound)."""
    raw = float(torch.tensor(u_values, dtype=F64).median())
    return max(raw, floor), raw, raw < floor


# ------------------------------------------------------------------ Correction 2: causal G3 state from FIT only
def g3_state_fit_only(fit16, fit_img, seed, k=None):
    """Canonical G3 task state from FIT tokens ONLY (nanm.g3.fit_task -> A.greedy_coreset + B.residual_stats).
    fit16: [N, 768] fp16 unit tokens of this task's FIT images; fit_img: [N] image ids. No other role is an input."""
    k = g3.task_size(len(fit16)) if k is None else k
    st = g3.fit_task(fit16, k=k, seed=seed)
    return st, fit_img[st.kc[:k]]


def stage_W(states):
    W, cond = g3.stage_whitener(states)               # canonical: B.whiten(sum S, sum n)
    return W.double(), cond


def require_disjoint(roles: dict):
    """roles: name -> set of image ids (paths or content hashes). Raises if any image appears in two roles."""
    seen = {}
    for r, ids in roles.items():
        for i in ids:
            if i in seen:
                raise ValueError(f'image {i!r} in both {seen[i]} and {r}')
            seen[i] = r


# ------------------------------------------------------------------ Correction 6: projection on both factors
def apply_H(S, beta, G):
    """H G with H = I - S diag(sigmoid(beta)) S^T (768x768, never materialised); G is a 768 x r gradient of P or Q.
    Rows of P and Q index z-coordinates: A z = z + P (Q^T z) -> Q holds read directions, P write directions.
    Left-multiplying both gradients by H restricts changes to how A reads from and writes into span(S)."""
    return G - S @ (torch.sigmoid(beta)[:, None] * (S.T @ G))


# ------------------------------------------------------------------ Correction 4: interference decomposition
def decompose(nll_before, nll_geo, nll_naive, nll_ctrl: dict, nll_oracle=None):
    """Paired old-task terms on identical EVAL queries (nats/patch, positive = worse):
       geometry  = NLL(G3 state t, phi_{t-1}) - NLL(G3 state t-1, phi_{t-1})     baseline geometry change
       damage    = NLL(G3 state t, phi_t^naive) - NLL(G3 state t, phi_{t-1})      adaptation damage (W, keys fixed)
       recovery  = NLL(..., phi_t^naive) - NLL(..., phi_t^X)                       conventional recovery per control
       oracle    = NLL(..., phi_{t-1}) - NLL(..., phi_j^R2)                        diagnostic only, never in a gate"""
    out = {'geometry': nll_geo - nll_before, 'damage': nll_naive - nll_geo,
           'recovery': {k: nll_naive - v for k, v in nll_ctrl.items()}}
    if nll_oracle is not None:
        out['oracle_advantage'] = nll_geo - nll_oracle
    return out


# ------------------------------------------------------------------ Correction 6: byte ledger
KEY_B, KEY_ID_B, PROBE_B = 1536, 8, 1540             # fp16 key + (task, image/index) id; fp16 probe + int32 id
S_B = g3.S_BYTES                                      # canonical pooled S (fp32) + count (fp64) = 2,359,304
W_CACHE_B = D * D * 4                                 # derived whitener cache (fp32), recomputable from S
PHI_B = 2 * D * RANK * 4                              # P, Q fp32 = 98,304
PSI_B = (D * RANK + RANK + 1) * 4                     # S, beta, rho fp32 = 49,220 (Stage 1 only)
META_B = 128                                          # counters, seeds, per-task key counts, U, hyperparameters
EXTRA = {
    'C0': 0,                       # frozen G3
    'C0q': 0,                      # closed-form refit: replaces canonical S by S' of identical size
    'C1': PHI_B,                   # naive SGD: phi only (no optimizer state)
    'C2': PHI_B,                   # replay: phi (+ probes, charged separately)
    'C3': 3 * PHI_B,               # online EWC: phi + diagonal Fisher (fp32) + anchor phi* (fp32)
    'C4': PHI_B + D * RANK * 4,    # fixed projection: phi + protected basis 768x16 fp32
    'C5': 2 * PHI_B,               # EMA: phi + EMA copy (fp32)
}


def ledger(arm, n_tasks, keys_per_task, n_fb_probes, n_reserve_probes=0):
    keys = sum(keys_per_task) * (KEY_B + KEY_ID_B)
    probes = (n_fb_probes + n_reserve_probes) * PROBE_B
    return {'keys': keys, 'g3_S': S_B, 'w_cache': W_CACHE_B, 'arm_state': EXTRA[arm], 'probes': probes,
            'meta': META_B + 4 * n_tasks,
            'total_charged': keys + S_B + EXTRA[arm] + probes + META_B + 4 * n_tasks}
