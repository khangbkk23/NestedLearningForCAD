# exps/nlcad/openloop.py
"""NLCAD-01 post-hoc controller diagnosis (CPU, recorded events only; no data, no labels, no GPU).

1. Decomposes every recorded hypergradient into signal (chain rule of dJ/da) and the L2 term 2*lam*phi.
2. Open-loop replay: feeds the recorded dJ/da sequence to a fresh controller with a different lam. The replay with the
   launched lam must reproduce the recorded phi exactly. Other lam values are FIRST-ORDER predictions only: in a
   closed loop the multipliers would change the trajectory and hence dJ/da.

  python -m exps.nlcad.openloop --run N_per_task_seed2711_from6
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly.common import require
from exps.nlcad.controller import BLOCKS, BlockController

OUT = D.ROOT / "results/hope_cad/nlcad"


def events(run, start):
    out = []
    for s in range(start + 1, 16):
        out += [dict(e, stage=s) for e in D.read(OUT / f"runs/{run}/EVENTS{s}.json")]
    return out


def decompose(ev, lam):
    phi = np.array([e["phi"] for e in ev])
    before = np.vstack([np.zeros(len(BLOCKS)), phi[:-1]])            # hypergradient of event t uses phi before step t
    ctrl, sig = BlockController(), []
    for e, p in zip(ev, before):
        with torch.no_grad():
            ctrl.phi.copy_(torch.tensor(p))
        sig.append(ctrl.grad_phi(e["dJ_da"], lam=0.0).numpy())
    sig, reg, hg = np.array(sig), 2 * lam * before, np.array([e["hypergrad"] for e in ev])
    return {"reconstruction_max_abs": float(np.abs(hg - sig - reg).max()),
            "median_abs_signal": float(np.median(np.abs(sig))), "median_abs_l2": float(np.median(np.abs(reg))),
            "frac_block_events_l2_dominates": float(np.mean(np.abs(reg) > np.abs(sig))),
            "signal_sign_agreement_with_pooled_mean": float(np.mean(np.sign(sig) == np.sign(sig.mean(0)))),
            "recorded_sign_agreement_with_pooled_mean": float(np.mean(np.sign(hg) == np.sign(hg.mean(0))))}


def replay(ev, lr, lam):
    ctrl, a = BlockController(lr=lr, weight_decay=lam), []
    for e in ev:
        ctrl.slow_step(e["dJ_da"])
        a.append(ctrl.multipliers().detach().numpy().copy())
    a = np.array(a)
    return {"lam": lam, "max_abs_a_minus_1": float(np.abs(a - 1).max()),
            "a_after_event": {str(t): dict(zip(BLOCKS, a[t - 1].round(4).tolist())) for t in (20, 60, 100, len(ev))},
            "saturated_blocks_end": int((torch.tanh(ctrl.phi.detach()).abs() > 0.95).sum()),
            "phi_end": ctrl.phi.detach().tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="N_per_task_seed2711_from6")
    a = ap.parse_args()
    cfg = D.read(OUT / f"runs/{a.run}/CONFIG.json")
    ev = events(a.run, cfg["start"])
    same = replay(ev, cfg["lr_phi"], cfg["wd_phi"])
    require(np.allclose(same["phi_end"], ev[-1]["phi"], rtol=0, atol=1e-12), "open-loop replay does not reproduce phi")
    res = {"run": a.run, "events": len(ev), "decomposition": decompose(ev, cfg["wd_phi"]), "replay_launched": same,
           "replay_counterfactual": [replay(ev, cfg["lr_phi"], lam) for lam in (0.0, 1e-5)],
           "caveat": "open loop: counterfactual lam values are first-order predictions, not runs"}
    D.write_json(OUT / f"OPENLOOP_{a.run}.json", res)
    print(res["decomposition"])
    for r in res["replay_counterfactual"]:
        print(r["lam"], round(r["max_abs_a_minus_1"], 3), r["saturated_blocks_end"],
              {b: v for b, v in r["a_after_event"][str(len(ev))].items()})


if __name__ == "__main__":
    main()
