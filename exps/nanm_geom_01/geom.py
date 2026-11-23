"""NANM-GEOM-01 pure functions (registration: agents/tasks/task_nanm_geom_01.md).

All likelihoods are densities on R^d in the same v-coordinates (Lebesgue reference measure), built on the NL-02 core
`nll_mixture`; nothing here modifies canonical G3 (nanm/g3.py, exps/nanm_r2*, exps/bench_g3).
"""
from __future__ import annotations

import math

import torch

from exps.nanm_nl_02 import core as C
from exps.nanm_r2 import run as R2

F64 = torch.float64
LOG2PI = C.LOG2PI


# ------------------------------------------------------------------ likelihood parts
def norm_term(n_keys, logabsdet_L, d):
    """Query-independent part of the mixture NLL: log|K| - log|det L| + d/2 log 2pi."""
    return math.log(n_keys) - logabsdet_L + 0.5 * d * LOG2PI


def nll_parts(Vq, Kv, Lmap, ld):
    """Per-query mixture NLL (exactly core.nll_mixture), its constant normalisation N and the fit term F = NLL - N."""
    v = C.nll_mixture(Vq, Kv, Lmap, ld)
    N = norm_term(len(Kv), ld, Vq.shape[1])
    return v, N, v - N


def nll_pertask_mixture(Vq, groups):
    """Task-free equal-weight-per-key mixture with a per-task metric: groups = [(K_i, L_i, log|det L_i|)].
       p(q) = 1/|K| sum_i sum_{m in K_i} N(q; m, (L_i L_i^T)^{-1}) -> per-query NLL (nats)."""
    logits = torch.cat([-0.5 * R2.sqdist(Vq @ L, K @ L) + ld for K, L, ld in groups], 1)
    nk = sum(len(K) for K, _, _ in groups)
    return -torch.logsumexp(logits, 1) + math.log(nk) + 0.5 * Vq.shape[1] * LOG2PI


def energy64(Vq, Kv, W):
    """fp64 reference of the canonical G3 patch energy 0.5 * min_m ||(q - m) W||^2 (M7 check only)."""
    return 0.5 * R2.sqdist(Vq @ W, Kv @ W).min(1).values


def whiten_alpha(S, n, alpha):
    """Pooled-S whitening with shrinkage alpha (canonical alpha = 0.5), same dtype path as nanm.g3.stage_whitener
    (fp64 eigh -> fp32 -> fp64)."""
    W, cond = R2.inv_sqrt_psd(R2.shrink(S / n, alpha))
    return W.float().double(), cond


def logdet(W):
    return float(torch.linalg.slogdet(W)[1])


# ------------------------------------------------------------------ 2 x 2 factorial
def factorial(A, B, C_, D):
    """A: (W_old, K_old), B: (W_old, K_new), C: (W_new, K_old), D: (W_new, K_new); positive = worse.
    Shapley values phi_W + phi_K == D - A exactly (up to fp rounding)."""
    return {'delta': D - A, 'w_only': C_ - A, 'k_only': B - A, 'interaction': D - B - C_ + A,
            'shapley_W': 0.5 * ((C_ - A) + (D - B)), 'shapley_K': 0.5 * ((B - A) + (D - C_))}


def spearman(x, y):
    rx = torch.argsort(torch.argsort(x)).double(); ry = torch.argsort(torch.argsort(y)).double()
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / torch.sqrt((rx * rx).sum() * (ry * ry).sum()))


# ------------------------------------------------------------------ verdict (registration G.5)
def verdict(mech_ok, delta, phi_W, phi_K, inter, cells_w_larger, n_cells, controls, U):
    """controls: {id: {'H': mean9 historical gain, 'H_pos': cells with H > 0, 'Acq': mean9 acquisition cost}} for the
    ELIGIBLE controls only. cells_w_larger: number of cells where phi_W > phi_K."""
    if not mech_ok:
        return 'GEOM01_PROTOCOL_BLOCKED', 'mechanical'
    act = [x for x, c in controls.items() if c['H'] >= 0.05 * U and c['H_pos'] >= 7 and c['Acq'] <= 0.02 * U]
    if delta < 0.02 * U:
        return 'GEOM01_NO_ACTIONABLE_DRIFT', 'drift_below_0.02U'
    if not act:
        return 'GEOM01_NO_ACTIONABLE_DRIFT', 'no_eligible_control_actionable'
    agree = cells_w_larger if phi_W > phi_K else n_cells - cells_w_larger
    if abs(inter) >= 0.5 * abs(delta) or agree < 7:
        return 'GEOM01_INTERACTION_DOMINANT', f'actionable={act}'
    return ('GEOM01_WHITENING_DRIFT_DOMINANT' if phi_W > phi_K else 'GEOM01_KEY_EXPANSION_DOMINANT'), f'actionable={act}'
