# exps/nlcad/forensic_coverage_adj.py
"""NLCAD-03 part D: does the buffer-distance association of old-normal loss inflation survive adjustment for the
acquisition-stage loss (and for atypicality, the distance to the nearest training identity)? CPU only; held-out
normals, normal training identities, RH-01b R15 checkpoints; no anomaly labels, no development data.

Outcomes per held-out image: absolute increase (late - acq) and log ratio log(late / acq). Partial Spearman: ranks of
outcome and buffer distance residualised on the ranks of the covariates (OLS), permutation p-value (10,000).

  python -m exps.nlcad.forensic_coverage_adj --seed 2711 --late 15
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.stats import rankdata

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.nlcad.forensic_blocks import checkpoint
from exps.nlcad.forensic_coverage import descriptor, per_image_loss
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB
from exps.rh01b.analysis import ACQ

OUT = D.ROOT / "results/hope_cad/nlcad"


def partial_spearman(y, x, covs, n_perm=10000, seed=0):
    r = lambda v: rankdata(v)
    Z = np.column_stack([np.ones(len(y))] + [r(c) for c in covs]) if covs else np.ones((len(y), 1))
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    ey, ex = res(r(y)), res(r(x))
    rho = float(ey @ ex / np.sqrt((ey @ ey) * (ex @ ex)))
    rng = np.random.default_rng(seed)
    null = np.array([abs(ey @ (p := rng.permutation(ex)) / np.sqrt((ey @ ey) * (p @ p))) for _ in range(n_perm)])
    return rho, float((1 + np.sum(null >= abs(rho))) / (1 + n_perm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2711)
    ap.add_argument("--late", type=int, default=15)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    proto = RB.verify_lock()
    idx = C.buffer_index(a.seed, R1.n_train(proto), list(C.STREAM[:a.late - 1]))
    dec = O.Decoder(768, 12).eval()
    res = {"seed": a.seed, "late": a.late, "categories": {}, "pooled": {}}
    pooled = {k: [] for k in ("inc", "logr", "d_buf", "d_train", "acq", "cat")}
    for c in C.EVAL:
        held, train = DR.load_split("heldout", c), C.load_features(c)
        dh, dt = descriptor(held), descriptor(train)
        loss = {}
        for s in (ACQ[c], a.late):
            dec.load_state_dict(checkpoint(a.seed, s)[0])
            loss[s] = per_image_loss(dec, held)
        acq, late = loss[ACQ[c]], loss[a.late]
        inc, logr = late - acq, np.log(late / acq)
        d_buf = (1 - dh @ dt[idx[c]].T).min(1).values.numpy()
        d_train = (1 - dh @ dt.T).min(1).values.numpy()
        out = {"n": len(held), "members": len(idx[c])}
        for name, y in (("increase", inc), ("log_ratio", logr)):
            out[name] = {"raw": partial_spearman(y, d_buf, []), "adj_acq": partial_spearman(y, d_buf, [acq]),
                         "adj_acq_dtrain": partial_spearman(y, d_buf, [acq, d_train]),
                         "dtrain_adj_acq_dbuf": partial_spearman(y, d_train, [acq, d_buf])}
        out["spearman_acq_vs_dbuf"] = partial_spearman(acq, d_buf, [])
        res["categories"][c] = out
        for k, v in (("inc", inc), ("logr", logr), ("d_buf", d_buf), ("d_train", d_train), ("acq", acq)):
            pooled[k] += v.tolist()
        pooled["cat"] += [C.EVAL.index(c)] * len(held)
        print(c, {n: {k: [round(x, 3) for x in v] for k, v in out[n].items()} for n in ("increase", "log_ratio")}, flush=True)
    P = {k: np.array(v) for k, v in pooled.items()}
    cat = [(P["cat"] == i).astype(float) for i in range(1, len(C.EVAL))]   # category fixed effects
    for name, y in (("increase", P["inc"]), ("log_ratio", P["logr"])):
        res["pooled"][name] = {"adj_acq_category": partial_spearman(y, P["d_buf"], [P["acq"]] + cat),
                               "adj_acq_dtrain_category": partial_spearman(y, P["d_buf"], [P["acq"], P["d_train"]] + cat)}
    print("pooled", res["pooled"], flush=True)
    D.write_json(OUT / f"FORENSIC_COVERAGE_ADJ_R15_{a.seed}_to{a.late}.json", res)


if __name__ == "__main__":
    main()
