# exps/nlcad/forensic_coverage.py
"""Does Replay-24 forget the old normals that its buffer does not cover? (CPU only; held-out normals, normal training
identities and the frozen encoder features; no anomaly labels, no development data, no GPU.)

Per held-out normal image i of an old category: the increase of its reconstruction loss between the stage the category
was acquired and the final stage (RH-01b R15 identity), against its feature-space distance to the nearest buffer
member (what replay rehearses) and to the nearest training identity (what joint training rehearses). The descriptor
is the token-mean of both encoder groups, cosine distance. A strong positive association with the buffer distance but
not with the training-set distance would say that coverage, not capacity, limits replay. It also tells how much a
diversity-aware (k-center) choice of the same number of members would reduce the buffer distance.

  python -m exps.nlcad.forensic_coverage --seed 2711 --late 15
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.nlcad.forensic_blocks import checkpoint
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB
from exps.rh01b.analysis import ACQ

OUT = D.ROOT / "results/hope_cad/nlcad"


def descriptor(x):
    d = torch.cat([x[:, 0].float().mean(1), x[:, 1].float().mean(1)], 1)
    return torch.nn.functional.normalize(d, dim=1)


@torch.no_grad()
def per_image_loss(dec, x):
    out = []
    for xb in x.split(8):
        en, de = dec(xb[:, 0].float(), xb[:, 1].float())
        out += torch.stack([1 - torch.nn.functional.cosine_similarity(e.flatten(1), d.flatten(1))
                            for e, d in zip(en, de)], 1).mean(1).tolist()
    return np.array(out)


def k_center(desc, k, first):
    chosen = [first]
    dist = 1 - desc @ desc[first]
    for _ in range(k - 1):
        j = int(torch.argmax(dist))
        chosen.append(j)
        dist = torch.minimum(dist, 1 - desc @ desc[j])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2711)
    ap.add_argument("--late", type=int, default=15)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    proto = RB.verify_lock()
    n = R1.n_train(proto)
    idx = C.buffer_index(a.seed, n, list(C.STREAM[:a.late - 1]))
    dec = O.Decoder(768, 12).eval()
    res = {"seed": a.seed, "late": a.late, "categories": {}}
    for c in C.EVAL:
        held, train = DR.load_split("heldout", c), C.load_features(c)
        dh, dt = descriptor(held), descriptor(train)
        members = idx.get(c, [])
        losses = {}
        for s in (ACQ[c], a.late):
            sd, h = checkpoint(a.seed, s)
            dec.load_state_dict(sd)
            losses[s] = per_image_loss(dec, held)
        inc = losses[a.late] - losses[ACQ[c]]
        d_buf = (1 - dh @ dt[members].T).min(1).values.numpy() if members else np.full(len(held), np.nan)
        d_all = (1 - dh @ dt.T).min(1).values.numpy()
        kc = k_center(dt, len(members), members[0]) if members else []
        d_kc = (1 - dh @ dt[kc].T).min(1).values.numpy() if kc else np.full(len(held), np.nan)
        rho_b, p_b = spearmanr(inc, d_buf)
        rho_a, p_a = spearmanr(inc, d_all)
        res["categories"][c] = {
            "n_heldout": len(held), "n_train": len(train), "members": members, "acq_stage": ACQ[c],
            "loss_acq_mean": float(losses[ACQ[c]].mean()), "loss_late_mean": float(losses[a.late].mean()),
            "increase_mean": float(inc.mean()), "increase_relative": float(inc.mean() / losses[ACQ[c]].mean()),
            "spearman_increase_vs_buffer_distance": [float(rho_b), float(p_b)],
            "spearman_increase_vs_trainset_distance": [float(rho_a), float(p_a)],
            "buffer_distance_mean": float(np.mean(d_buf)), "kcenter_distance_mean": float(np.mean(d_kc)),
            "trainset_distance_mean": float(d_all.mean()), "per_image": {"increase": inc.tolist(), "d_buffer": d_buf.tolist(),
                                                                         "d_trainset": d_all.tolist()}}
        r = res["categories"][c]
        print(c, "inc", round(r["increase_mean"], 4), f"({r['increase_relative']:+.0%})", "rho_buf", [round(x, 3) for x in r["spearman_increase_vs_buffer_distance"]],
              "rho_train", [round(x, 3) for x in r["spearman_increase_vs_trainset_distance"]],
              "d_buf", round(r["buffer_distance_mean"], 4), "d_kc", round(r["kcenter_distance_mean"], 4),
              "d_train", round(r["trainset_distance_mean"], 4), flush=True)
    D.write_json(OUT / f"FORENSIC_COVERAGE_R15_{a.seed}_to{a.late}.json", res)


if __name__ == "__main__":
    main()
