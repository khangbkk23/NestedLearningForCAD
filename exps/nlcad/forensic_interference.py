# exps/nlcad/forensic_interference.py
"""Can a compact, data-free memory of old normals predict which updates make Replay-24 forget them? (CPU only;
normal training identities, held-out normals and the RH-01b R15 checkpoints; no anomaly labels, no development data.)

For an old category c, the slow memory is the per-Linear-layer second moment of c's input activations, Sigma_l(c),
computed ONCE at the end of c's own task (stale afterwards, as a deployed method would hold it). For each later stage
s the first-order change of every layer's output on all of c's training identities is
  I_cov(s, c) = sum_l tr(dW_l Sigma_l(c) dW_l^T),   dW = theta_s - theta_{s-1}.
Compared predictors of the held-out normal loss increase of c at stage s:
  I_cov (covers every training identity of c, no images stored), and I_cov_r32 with Sigma truncated to its top-32
  eigenpairs per layer (the deployable size: about 3 MB in FP16 for all 34 layers),
  member loss increase (what replay of c's buffer members sees: the information Variant N's outer objective has),
  ||dW||^2 (update magnitude only).

  python -m exps.nlcad.forensic_interference --seed 2711
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.nlcad.forensic_blocks import checkpoint
from exps.nlcad.forensic_coverage import per_image_loss
from exps.nlcad.forensic_subspace import covariances
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB
from exps.rh01b.analysis import ACQ

OUT = D.ROOT / "results/hope_cad/nlcad"
PRED = ("I_cov", "I_cov_r32", "member_increase", "dW2")


def layer_weights(sd):
    return {k[:-len(".weight")]: v.double() for k, v in sd.items() if k.endswith(".weight") and v.dim() == 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2711)
    ap.add_argument("--images", type=int, default=64)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    proto = RB.verify_lock()
    n = R1.n_train(proto)
    dec = O.Decoder(768, 12).eval()
    sds = {s: checkpoint(a.seed, s)[0] for s in range(1, len(C.STREAM) + 1)}
    held = {s: D.read(C.OUT / f"runs/R15_seed{a.seed}/STAGE{s}.json")["heldout_normal"] for s in sds}
    rows = []
    for c in C.EVAL:
        acq = ACQ[c]
        dec.load_state_dict(sds[acq])
        train = C.load_features(c)
        sigma = covariances(dec, train[:a.images])                     # memory written at the end of c's task
        low = {}
        for l, v in sigma.items():
            w, u = torch.linalg.eigh(v)
            low[l] = u[:, -32:] * w[-32:].clamp(min=0).sqrt()              # Sigma_r = low @ low.T
        for s in range(acq + 1, len(C.STREAM) + 1):
            members = C.buffer_index(a.seed, n, list(C.STREAM[:s - 1]))[c]
            w0, w1 = layer_weights(sds[s - 1]), layer_weights(sds[s])
            icov = sum(float(((w1[l] - w0[l]) @ sigma[l] * (w1[l] - w0[l])).sum()) for l in sigma)
            icov32 = sum(float((((w1[l] - w0[l]) @ low[l]) ** 2).sum()) for l in low)
            mag = sum(float(((w1[l] - w0[l]) ** 2).sum()) for l in sigma)
            mem = []
            for t in (s - 1, s):
                dec.load_state_dict(sds[t])
                mem.append(per_image_loss(dec, train[members]).mean())
            rows.append({"category": c, "stage": s, "I_cov": icov, "I_cov_r32": icov32, "dW2": mag, "members": len(members),
                         "member_increase": float(mem[1] - mem[0]),
                         "heldout_increase": held[s][c]["global_cosine"] - held[s - 1][c]["global_cosine"]})
            print(c, s, f"Icov {icov:.3e} dW2 {mag:.3e} mem {rows[-1]['member_increase']:+.4f} "
                        f"held {rows[-1]['heldout_increase']:+.4f}", flush=True)
    res = {"seed": a.seed, "images_for_sigma": a.images, "rows": rows, "spearman": {}}
    for scope in ["all"] + list(C.EVAL):
        r = [x for x in rows if scope == "all" or x["category"] == scope]
        y = [x["heldout_increase"] for x in r]
        res["spearman"][scope] = {p: list(map(float, spearmanr([x[p] for x in r], y))) for p in PRED}
        res["spearman"][scope]["n"] = len(r)
        print(scope, {p: [round(v, 3) for v in res["spearman"][scope][p]] for p in PRED},
              "n", len(r), flush=True)
    D.write_json(OUT / f"FORENSIC_INTERFERENCE_R15_{a.seed}.json", res)


if __name__ == "__main__":
    main()
