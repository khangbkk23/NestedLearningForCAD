"""NL03-MULTIRATE-01 §10 normal-only evaluation (SYNTHETIC_DEV).

Per (arm, stage l, seen task j): energies of 32 synthetic + 32 clean EVAL images and 16 CALIB images (canonical
energy, A1 readout R2.maps_of + A.image_scores), then
  (a) image AUROC (synthetic vs clean), pixel AP at 392x392 over the 64 maps (binned, A.image_hists, 8192 bins);
  (b) FPR = fraction of clean EVAL images with score > theta_j, theta_j = max CALIB score at j's acquisition stage.
Bootstrap material (per base EVAL image i: synth_i + clean_i histograms at 1024 bins, image scores) is kept for the
final stage and for the acquisition stage.
"""
from __future__ import annotations

import numpy as np
import torch

from exps.nl03_multirate_01 import spec as SP

HB = 1024                       # bootstrap histogram bins


def _m():
    from exps.bench_g3 import run as G0
    from exps.nanm_r0 import audit as A
    from exps.nanm_r2 import run as R2
    import nanm.g3 as g3
    return G0, A, R2, g3


@torch.no_grad()
def energies(q16, keys32, W):
    G0, A, R2, g3 = _m()
    flat = q16.reshape(-1, SP.D)
    e = g3.score(flat, keys32, W.cuda()) if W is not None else G0.euclid(flat, keys32)
    return e.reshape(len(q16), 28, 28)


def auroc_mat(pos, neg):
    """M[i, j] = 1[pos_i > neg_j] + 0.5 1[=] (paired by base image index)."""
    return (pos[:, None] > neg[None]).astype(np.float64) + 0.5 * (pos[:, None] == neg[None])


@torch.no_grad()
def eval_task(Q, keys32, W, theta=None, keep=False, cover=False):
    from sklearn.metrics import roc_auc_score
    G0, A, R2, g3 = _m()
    n = len(Q['tokens'])
    q = torch.cat([Q['tokens'], Q['clean'], Q['calib']])
    g = energies(q, keys32, W)
    maps = R2.maps_of(g)
    img = A.image_scores(maps)
    s_syn, s_cln, s_cal = img[:n], img[n:2 * n], img[2 * n:]
    m64 = maps[:2 * n]
    masks = np.concatenate([Q['masks'].numpy(), np.zeros_like(Q['masks'].numpy())])
    hws = [(SP.SYNTH['size'], SP.SYNTH['size'])] * (2 * n)
    hi = float(m64.max()) + 1e-12
    pos, tot = A.image_hists(m64, hws, masks, 0., hi)
    y = np.r_[np.ones(n), np.zeros(n)]
    out = {'auroc': float(roc_auc_score(y, np.r_[s_syn, s_cln])), 'ap': A.binned_ap(pos.sum(0), tot.sum(0)),
           'theta_calib': float(s_cal.max()), 's_syn': s_syn.astype(np.float32), 's_cln': s_cln.astype(np.float32),
           's_cal': s_cal.astype(np.float32)}
    if theta is not None:
        out['fpr'] = float((s_cln > theta).mean())
    if keep:
        f = pos.shape[1] // HB
        hp = pos.reshape(2 * n, HB, f).sum(2); ht = tot.reshape(2 * n, HB, f).sum(2)
        out['hpos'] = (hp[:n] + hp[n:]).astype(np.float32)        # per base image (synth_i + clean_i)
        out['htot'] = (ht[:n] + ht[n:]).astype(np.float32)
        out['ap_hb'] = A.binned_ap(out['hpos'].sum(0), out['htot'].sum(0))
    if cover:
        d = (2 * G0.euclid(Q['clean'].reshape(-1, SP.D), keys32)).clamp_min(0).sqrt().numpy()
        out['cover_median'] = float(np.median(d)); out['cover_p95'] = float(np.quantile(d, .95))
    return out


def exact_check(Q, keys32, W):
    """M-check: binned (8192) vs exact pixel AP on one evaluation."""
    G0, A, R2, g3 = _m()
    n = len(Q['tokens'])
    g = energies(torch.cat([Q['tokens'], Q['clean']]), keys32, W)
    maps = R2.maps_of(g)
    masks = np.concatenate([Q['masks'].numpy(), np.zeros_like(Q['masks'].numpy())])
    hws = [(SP.SYNTH['size'], SP.SYNTH['size'])] * (2 * n)
    pos, tot = A.image_hists(maps, hws, masks, 0., float(maps.max()) + 1e-12)
    return {'binned': A.binned_ap(pos.sum(0), tot.sum(0)), 'exact': float(A.exact_ap(maps, hws, list(masks)))}


# ------------------------------------------------------------------ paired bootstrap (stratified by task)
def boot_w(n, task_index, B=SP.BOOT_B):
    rng = np.random.default_rng([SP.BOOT_SEED, task_index])
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(np.float64)


def boot_metrics(rec, w, theta=None):
    """rec: eval_task output with keep=True. Returns replicate arrays [B] for auroc, ap (1024-bin), fpr."""
    from exps.nanm_r0 import audit as A
    M = auroc_mat(rec['s_syn'].astype(np.float64), rec['s_cln'].astype(np.float64))
    au = np.einsum('bi,ij,bj->b', w, M, w) / w.sum(1) ** 2
    P, T = w @ rec['hpos'], w @ rec['htot']
    ap = np.array([A.binned_ap(P[b], T[b]) for b in range(len(w))])
    out = {'auroc': au, 'ap': ap}
    if theta is not None:
        out['fpr'] = (w * (rec['s_cln'] > theta)).sum(1) / w.sum(1)
    return out
