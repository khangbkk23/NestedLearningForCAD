# exps/nlcad/horizon_real.py
"""NLCAD-03 part B: the only long window that existing artifacts support is a whole stage (667 steps, one warm restart):
theta_{s-1} and theta_s are stored, intermediate states are not. For the NLCAD-01 Stage C run N (a ~ 1), per stage s:

  endpoint_b = <grad_b J(theta_s), theta_s - theta_{s-1}>     (= d/d alpha_b of the chord objective at alpha = 1)
  chord      J(theta_{s-1} + alpha * (theta_s - theta_{s-1})), alpha in {0.75, 0.9, 1, 1.1}   (forward only)

compared with the one-step slow-event signals recorded in that stage. The chord is NOT the training trajectory: it
answers "would a uniformly shorter displacement of this stage have lowered J", not "would a smaller learning rate".
Objectives: J_q (8 current identities of task s + 8 buffer members at stage s, fixed draw) and J_held (10 held-out
normals per development category). CPU only; normal data; no labels.

  python -m exps.nlcad.horizon_real --run N_per_task_seed2711_from6
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad.controller import BLOCKS, block_of
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

OUT = D.ROOT / "results/hope_cad/nlcad"
ALPHAS = (0.75, 0.9, 1.1)


def state(run, s):
    rec = D.read(OUT / f"runs/{run}/STAGE{s}.json") if s > 6 else D.read(C.OUT / f"runs/R15_seed2711/STAGE{s}.json")
    path = D.ROOT / rec["checkpoint"]
    require(sha(path) == rec["checkpoint_sha256"], f"checkpoint changed: {path}")
    return {k: v.float() for k, v in torch.load(path, weights_only=False)["state"].items()}


def loss_and_grad(dec, x, grad=True, micro=2):
    dec.eval()
    total, n = 0.0, len(x)
    dec.zero_grad(set_to_none=True)
    for xb in x.split(micro):
        with torch.set_grad_enabled(grad):
            en, de = dec(xb[:, 0].float(), xb[:, 1].float())
            loss = O.global_cosine(en, de) * len(xb) / n
            if grad:
                loss.backward()
        total += float(loss)
    g = {k: p.grad.detach().clone() for k, p in dec.named_parameters()} if grad else None
    return total, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="N_per_task_seed2711_from6")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = D.read(OUT / f"runs/{a.run}/CONFIG.json")
    seed, start = cfg["seed"], cfg["start"]
    proto = RB.verify_lock()
    n = R1.n_train(proto)
    held = torch.cat([DR.load_split("heldout", c)[:10] for c in C.EVAL])
    dec = O.Decoder(768, 12)
    res = {"run": a.run, "alphas": ALPHAS, "stages": {}}
    prev = state(a.run, start)
    for s in range(start + 1, len(C.STREAM) + 1):
        cur = state(a.run, s)
        c = C.STREAM[s - 1]
        g = torch.Generator().manual_seed(9000 + s)
        f = C.load_features(c)
        idx = C.buffer_index(seed, n, list(C.STREAM[:s - 1]))
        rep = torch.stack([C.load_features(d)[i] for d in idx for i in idx[d]])
        q = torch.cat([f[torch.randperm(len(f), generator=g)[:8]], rep[torch.randperm(len(rep), generator=g)[:8]]])
        disp = {k: cur[k] - prev[k] for k in cur}
        rec = {}
        for name, x in (("J_q", q), ("J_held", held)):
            dec.load_state_dict(cur)
            J, grad = loss_and_grad(dec, x)
            end = {b: 0.0 for b in BLOCKS}
            for k, v in grad.items():
                end[block_of(k)] += float((v * disp[k]).sum())
            chord = {}
            for al in ALPHAS:
                dec.load_state_dict({k: prev[k] + al * disp[k] for k in cur})
                chord[al] = loss_and_grad(dec, x, grad=False)[0]
            dec.load_state_dict(prev)
            J0 = loss_and_grad(dec, x, grad=False)[0]
            rec[name] = {"J_start": J0, "J_end": J, "chord": chord, "endpoint": end, "endpoint_global": sum(end.values())}
        ev = D.read(OUT / f"runs/{a.run}/EVENTS{s}.json")
        one = np.array([[e["dJ_da"][b] * e["a_used"][b] for b in BLOCKS] for e in ev])
        rec["onestep_events"] = {"global_mean": float(one.sum(1).mean()), "global_pos_frac": float((one.sum(1) > 0).mean()),
                                 "block_mean": dict(zip(BLOCKS, one.mean(0).tolist()))}
        res["stages"][s] = rec
        e = np.array([rec["J_q"]["endpoint"][b] for b in BLOCKS])
        o = one.mean(0)
        cc = lambda u: u - u.mean()
        print(s, c, "J_q", f"{rec['J_q']['J_start']:.4f}->{rec['J_q']['J_end']:.4f}", "chord", {k: round(v, 4) for k, v in rec["J_q"]["chord"].items()},
              "endG", f"{rec['J_q']['endpoint_global']:+.3e}", "| J_held", f"{rec['J_held']['J_start']:.4f}->{rec['J_held']['J_end']:.4f}",
              "chord", {k: round(v, 4) for k, v in rec["J_held"]["chord"].items()}, "endG", f"{rec['J_held']['endpoint_global']:+.3e}",
              "| onestep G>0", round(rec["onestep_events"]["global_pos_frac"], 2),
              "cos_centred(end_q, onestep)", round(float(cc(e) @ cc(o) / (np.linalg.norm(cc(e)) * np.linalg.norm(cc(o)))), 3), flush=True)
        prev = cur
    D.write_json(OUT / f"HORIZON_REAL_{a.run}.json", res)


if __name__ == "__main__":
    main()
