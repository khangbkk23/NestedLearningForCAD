# exps/nlcad/forensic_blocks.py
"""Where does Replay-24 forget? Block transplant between two checkpoints of the RH-01b R15 identity (CPU only; held-out
normals and current-task training normals; no anomaly labels, no development data, no GPU).

  revert_b: theta_late with block b restored to theta_early  -> how much of the old-normal damage block b carries
  graft_b:  theta_early with block b taken from theta_late    -> how much damage block b's drift causes on its own
Both are also measured on the latest task's training normals (plasticity side). Proxies: the global-cosine loss and
the Dinomaly image score (R_D, top-1% of the smoothed map) on held-out normals, i.e. the "normal-score inflation" that
the RH-01 forensic analysis identified as the main AP-loss mechanism.

  python -m exps.nlcad.forensic_blocks --seed 2711 --early 6 --late 15
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import readout as RO
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad.controller import BLOCKS
from exps.rh01b import common as C

OUT = D.ROOT / "results/hope_cad/nlcad"


def checkpoint(seed, s):
    rec = D.read(C.OUT / f"runs/R15_seed{seed}/STAGE{s}.json")
    path = D.ROOT / rec["checkpoint"]
    require(sha(path) == rec["checkpoint_sha256"], f"checkpoint changed: {path}")
    return torch.load(path, weights_only=False)["state"], rec["checkpoint_sha256"]


def mix(base, donor, block):
    return {k: (donor[k] if k.startswith(block + ".") else v) for k, v in base.items()}


@torch.no_grad()
def stats(dec, x):
    loss, image = [], []
    for xb in x.split(8):
        en, de = dec(xb[:, 0].float(), xb[:, 1].float())
        loss += torch.stack([1 - torch.nn.functional.cosine_similarity(e.flatten(1), d.flatten(1))
                             for e, d in zip(en, de)], 1).mean(1).tolist()
        image += RO.dinomaly(en, de)[1].tolist()
    return {"global_cosine": float(np.mean(loss)), "image_score_mean": float(np.mean(image)),
            "image_score_p90": float(np.quantile(image, .9))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2711)
    ap.add_argument("--early", type=int, default=6)
    ap.add_argument("--late", type=int, default=15)
    ap.add_argument("--current-n", type=int, default=32)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    early, h_early = checkpoint(a.seed, a.early)
    late, h_late = checkpoint(a.seed, a.late)
    data = {c: DR.load_split("heldout", c) for c in C.EVAL}
    cur = C.STREAM[a.late - 1]
    data[f"train:{cur}"] = C.load_features(cur)[:a.current_n]
    dec = O.Decoder(768, 12).eval()
    configs = {"early": early, "late": late}
    configs |= {f"revert:{b}": mix(late, early, b) for b in BLOCKS}
    configs |= {f"graft:{b}": mix(early, late, b) for b in BLOCKS}
    res = {"seed": a.seed, "early": a.early, "late": a.late, "checkpoints": {"early": h_early, "late": h_late},
           "drift_relative": {b: float(torch.sqrt(sum(((late[k] - early[k]).double() ** 2).sum() for k in late if k.startswith(b + ".")))
                                       / torch.sqrt(sum((early[k].double() ** 2).sum() for k in early if k.startswith(b + "."))))
                              for b in BLOCKS},
           "n": {k: len(v) for k, v in data.items()}, "configs": {}}
    for name, sd in configs.items():
        dec.load_state_dict(sd)
        res["configs"][name] = {k: stats(dec, v) for k, v in data.items()}
        print(name, {k: round(v["image_score_mean"], 4) for k, v in res["configs"][name].items()}, flush=True)
    D.write_json(OUT / f"FORENSIC_BLOCKS_R15_{a.seed}_{a.early}to{a.late}.json", res)


if __name__ == "__main__":
    main()
