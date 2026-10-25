# exps/ntm/review0r.py
"""NTM-00R review: normal-only intervention diagnostics (Stage 0A) and synthetic masking fixtures.

Reads only NORMAL development images (``test/good``) from the Stage-0 label-free arrays; no defect
image, mask or score of a defective image is touched. Fixtures are synthetic.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from exps.calm02 import gate as G
from exps.calm02 import run as R
from exps.calm02.data import ROOT, MVTEC_ORDER
from exps.ntm.stage0 import OUT as STAGE0, combine, context_distance, context_mask

OUT = ROOT / "results/hope_cad/ntm/review0r"
CATS, ACQ, FINAL = G.CATS, G.ACQ, G.FINAL


def _normal_index(rows):
    return np.array([i for i, r in enumerate(rows) if "/test/good/" in r["relative_path"]])


def diagnostics():
    co = G.cohort()
    P = np.load(STAGE0 / "prepared.npz")
    prep = json.loads((STAGE0 / "prepared.json").read_text())
    tau = np.array([prep["tau"][str(k)]["tau"] for k in range(len(prep["tau"]))])
    out = {"tau": {MVTEC_ORDER[k]: float(tau[k]) for k in range(len(tau))}, "cells": {}}
    for c in CATS:
        n = _normal_index(co[c])
        ci = MVTEC_ORDER.index(c)
        dx = P[f"dX_{c}"][n].ravel()
        for k in (ACQ[c], FINAL):
            dm = P[f"dM_{c}_{k:02d}"][n].ravel(); own = P[f"owner_{c}_{k:02d}"][n].ravel()
            t_own, t_c = tau[own], tau[ci]
            s = combine("NTM_FLOOR", dm, dx, t_own)
            red = dm - s
            changed = red > 0
            changed_tc = np.maximum(dx, t_c) < dm
            rec = {"n_normal_patches": int(dm.size),
                   "frac_dX_lt_dM": float((dx < dm).mean()),
                   "frac_dM_gt_tau_owner": float((dm > t_own).mean()),
                   "frac_dM_gt_tau_own_category": float((dm > t_c).mean()),
                   "frac_changed_by_ntm_floor": float(changed.mean()),
                   "mean_reduction_all": float(red.mean()),
                   "mean_reduction_if_changed": float(red[changed].mean()) if changed.any() else 0.0,
                   "max_reduction": float(red.max()),
                   "frac_foreign_nn": float((own != ci).mean()),
                   "mean_abs_tau_owner_minus_tau_c_on_foreign": float(np.abs(t_own - t_c)[own != ci].mean()) if (own != ci).any() else 0.0,
                   "frac_change_decision_differs_tau_owner_vs_tau_c": float((changed != changed_tc).mean()),
                   "tails": {}}
            for q in (0.90, 0.95, 0.99):
                top = dm >= np.quantile(dm, q)
                rec["tails"][f"top{int(round((1 - q) * 100))}pct"] = {
                    "frac_changed": float(changed[top].mean()), "mean_reduction": float(red[top].mean()),
                    "q_before": float(np.quantile(dm, q)), "q_after": float(np.quantile(s, q))}
            out["cells"][f"{c}_{k:02d}"] = rec
    return out


# ---------------- synthetic masking fixtures (patch-level AP; 28x28 grid; dim 8) ----------------
def _fixture(kind, seed=0):
    """Return (features [784,8], defect labels [784], bank [M,8]) for one adversarial case."""
    g = torch.Generator().manual_seed(seed)
    noise = lambda n: 0.05 * torch.randn(n, 8, generator=g)
    e = torch.eye(8)
    a, b, p = 2 * e[0], 2 * e[0] + 3 * e[1], 2 * e[0] - 3.5 * e[2]   # normal mode, defect, unique normal part
    x = a + noise(784); y = torch.zeros(784, dtype=torch.bool)
    grid = torch.arange(784).reshape(28, 28)
    bank_adequate = a + noise(64)                       # covers mode a closely (d_M ~ 0.1-0.2)
    bank_deficient = a + 1.0 * e[3] + noise(64)         # evicted: nearest support ~1.0 away
    if kind == "large_connected_defect":
        idx = grid[5:15, 5:15].ravel(); x[idx] = b + noise(len(idx)); y[idx] = True; bank = bank_deficient
    elif kind == "repeated_distant_defects":
        idx = torch.cat([grid[3:5, 3:5].ravel(), grid[20:22, 20:22].ravel()])
        x[idx] = b + noise(len(idx)); y[idx] = True; bank = bank_deficient
    elif kind == "small_unique_defect_with_evicted_normal_mode":
        rare = torch.cat([grid[r, cc].reshape(1) for r in range(2, 28, 7) for cc in range(2, 28, 7)])
        x[rare] = a + 3.5 * e[4] + noise(len(rare))     # recurring normal mode absent from the bank
        idx = grid[13:15, 13:15].ravel(); x[idx] = b + noise(len(idx)); y[idx] = True; bank = bank_deficient
    elif kind == "regular_texture_adequate_bank":
        idx = grid[13:15, 13:15].ravel(); x[idx] = b + noise(len(idx)); y[idx] = True; bank = bank_adequate
    elif kind == "unique_normal_part":
        part = grid[2:4, 22:24].ravel(); x[part] = p + noise(len(part))   # appears once, evicted
        idx = grid[13:15, 13:15].ravel(); x[idx] = a + 1.6 * e[1] + noise(len(idx)); y[idx] = True
        bank = bank_deficient
    else:
        raise ValueError(kind)
    return x, y, bank


FIXTURES = ("large_connected_defect", "repeated_distant_defects", "small_unique_defect_with_evicted_normal_mode",
            "regular_texture_adequate_bank", "unique_normal_part")


def fixture_aps(kind, tau=0.5, radius=4):
    from sklearn.metrics import average_precision_score
    from exps.task_p.p0_normal_support import distances
    x, y, bank = _fixture(kind)
    d_m = distances(x, bank).min(1).values.numpy()
    d_x = context_distance(x, context_mask(radius)).numpy()
    s = combine("NTM_FLOOR", d_m, d_x, np.full_like(d_m, tau))
    lab = y.numpy()
    return {"bank_ap": float(average_precision_score(lab, d_m)), "ntm_floor_ap": float(average_precision_score(lab, s)),
            "defect_patches_capped_at_tau": int(((s == tau) & lab).sum()),
            "normal_patches_floored_at_tau": int(((s == tau) & ~lab).sum()), "n_defect": int(lab.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("diagnostics", "fixtures"))
    a = ap.parse_args()
    if a.stage == "diagnostics":
        R.write_json(OUT / "stage0a_normal_diagnostics.json", diagnostics())
        print((OUT / "stage0a_normal_diagnostics.json").read_text())
    else:
        res = {k: fixture_aps(k) for k in FIXTURES}
        R.write_json(OUT / "masking_fixtures.json", res)
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
