"""NL-SGM-01A normal-only metrics (no anomaly label, mask or test image).

All distances are the SC-SN-01 per-pair value ``exact.reference`` via the certified ``exact.nn``, so subset
arms satisfy E_sel >= 0 exactly (same per-pair function on a subset of rows).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

from exps.nl_sgm_01a import exact as X
from exps.task_p.p0_normal_support import require


def rowset(rows, gids):
    order = torch.argsort(gids)
    return X.RowSet(rows[order].contiguous(), gids[order])


def decompose(d_W, d_S, d_Q):
    """Per-patch identities. d_S is NaN for arms without a lossless source set (then only E_tot terms)."""
    e_tot = d_Q - d_W
    out = {"E_tot_signed": e_tot, "E_tot_abs": e_tot.abs(), "E_over": e_tot.clamp_min(0),
           "E_under": (-e_tot).clamp_min(0)}
    if d_S is not None:
        out["E_sel"] = d_S - d_W
        out["E_codec_signed"] = d_Q - d_S
    return out


def check_identities(parts, tol=1e-9):
    require(bool(torch.allclose(parts["E_tot_abs"], parts["E_over"] + parts["E_under"], atol=tol, rtol=0)),
            "abs != over + under")
    require(bool(torch.allclose(parts["E_tot_signed"], parts["E_over"] - parts["E_under"], atol=tol, rtol=0)),
            "signed != over - under")
    if "E_sel" in parts:
        require(bool((parts["E_sel"] >= -tol).all()), "E_sel < 0 on a subset arm")
        require(bool(torch.allclose(parts["E_tot_signed"], parts["E_sel"] + parts["E_codec_signed"], atol=tol, rtol=0)),
                "E_tot != E_sel + E_codec")


def image_fidelity(d_ref, d_arm, id_ref=None, id_arm=None):
    """d_* [784] for one image; returns the per-image fidelity record vs the reference (A0)."""
    a, b = d_ref.numpy(), d_arm.numpy()
    rec = {"image_max_abs": float(abs(b.max() - a.max())), "patch_max_abs": float(np.abs(b - a).max()),
           "spearman": float(spearmanr(a, b).statistic), "kendall": float(kendalltau(a, b).statistic)}
    for frac, key in ((0.01, "top1"), (0.05, "top5")):
        k = max(1, int(round(frac * len(a))))
        ta = set(np.argsort(-a, kind="stable")[:k].tolist()); tb = set(np.argsort(-b, kind="stable")[:k].tolist())
        rec[key] = len(ta & tb) / k
    if id_ref is not None and id_arm is not None:
        rec["nn_identity_recall"] = float((id_ref == id_arm).double().mean())
    return rec


def scale_q50(sample_x, sample_ids, rows):
    """q50_c: median of d(x, W_c \\ {x}) over the hash sample; exact duplicates (d = 0) dropped first.

    Returns (q50, n_used, degenerate flag). ``rows`` is the category's writer RowSet.
    """
    d, _ = X.nn(sample_x, rows, exclude=sample_ids)
    d = d[torch.isfinite(d) & (d > 0)]
    if not len(d):
        return float("nan"), 0, True
    q = float(d.median())
    return q, int(len(d)), q <= 1e-12


def false_support(dec_rows, src_ids, writers, q50_by_row):
    """f_incl / f_excl per decoded row; self (source) match excluded for f_excl."""
    d_in, _ = X.nn(dec_rows, writers)
    d_ex, _ = X.nn(dec_rows, writers, exclude=src_ids)
    return {"f_incl": d_in / q50_by_row, "f_excl": d_ex / q50_by_row,
            "frac_excl_gt1": float((d_ex / q50_by_row > 1).double().mean())}


def gap_probes(x_a, x_b):
    """Midpoints of hash-paired writer patches (geometric stress probes, not anomaly evidence)."""
    return 0.5 * (x_a + x_b)


def rare_mode_coverage(sample_x, r16, support):
    """Covered iff the arm's decoded support has a row within the patch's own 16-NN radius in W_c.
    ``r16``: [n] 16-NN radius (self excluded) of the sample patches; only the top-10% radius patches are passed."""
    d, _ = X.nn(sample_x, support)
    return float((d <= r16).double().mean())


def occupancy(positions, side=28, grid=7):
    """7x7 occupancy entropy / min-bin / evenness of stored rows' source lattice positions."""
    r, c = positions // side, positions % side
    b = (r // (side // grid)) * grid + (c // (side // grid))
    cnt = np.bincount(np.asarray(b), minlength=grid * grid).astype(float)
    p = cnt / max(1.0, cnt.sum())
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum())
    return {"entropy": ent, "evenness": ent / math.log(grid * grid), "min_bin": int(cnt.min())}
