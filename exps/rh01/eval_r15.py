# exps/rh01/eval_r15.py
"""AMENDMENT_3 (evaluation only): score and analyse the existing R15 seed-2712 checkpoints with the locked scorer
and metric code (J15 seed 2712 was aborted, so the per-seed-pair scorer cannot run), then report per-seed replay
forgetting for seeds 2711 and 2712. No training, no new checkpoints.

  python -m exps.rh01.eval_r15
"""
from __future__ import annotations

import numpy as np

from exps.dinomaly import analysis as DA
from exps.dinomaly import common as D
from exps.dinomaly.common import require, sha
from exps.rh01 import analysis as RA
from exps.rh01 import common as C
from exps.rh01 import run as RR

SEEDS_SCORED = (2712,)
SEEDS_ALL = (2711, 2712)


def r15_only(seeds):
    out = []
    for seed in seeds:
        rec = D.read(C.OUT / f"runs/R15_seed{seed}/TRAIN.json")
        for st in rec["stages"]:
            path = D.ROOT / st["checkpoint"]
            require(sha(path) == st["checkpoint_sha256"], f"checkpoint changed: {path}")
            admitted = [c for c in C.EVAL if C.STREAM.index(c) < st["index"]]
            if admitted:
                out.append(("R15", seed, st["index"], admitted, path))
    return out


def forgetting():
    target = C.OUT / "R15_FORGETTING.json"
    require(not target.exists(), "one-shot")
    t = RA.Table()
    res = {}
    for s in SEEDS_ALL:
        cats = {}
        for c in C.EVAL:
            a, f = RA.ACQ[c], RA.FINAL
            cats[c] = {"acq": t.point("R15", a, c, (s,)), "final": t.point("R15", f, c, (s,)),
                       "F": t.point("R15", a, c, (s,)) - t.point("R15", f, c, (s,)),
                       "F_ci95_images": DA.diff_ci(t.reps("R15", a, c, (s,), "ap_low"), t.reps("R15", a, c, (s,), "ap_high"),
                                                   t.reps("R15", f, c, (s,), "ap_low"), t.reps("R15", f, c, (s,), "ap_high")),
                       "trajectory": {str(k): t.point("R15", k, c, (s,)) for k in range(a, f + 1)},
                       "image_auroc_final": t.point("R15", f, c, (s,), "image_auroc")}
        res[str(s)] = {"categories": cats, "Fbar": float(np.mean([cats[c]["F"] for c in C.EVAL])),
                  "final_macro": float(np.mean([cats[c]["final"] for c in C.EVAL])),
                  "final_pooled": t.p[f"R15/{s}/{RA.FINAL}/pooled/RD"]["pixel_ap"]}
    res["note"] = ("Per-seed image-bootstrap intervals reflect image sampling only, not training-seed variability. "
                   "No joint reference exists for seed 2712; G15 is not computable for it.")
    D.write_json(target, res)
    for s in SEEDS_ALL:
        r = res[str(s)]
        print(s, "Fbar", round(r["Fbar"], 4), "final macro", round(r["final_macro"], 4),
              {c: (round(v["acq"], 3), round(v["final"], 3), round(v["F"], 3)) for c, v in r["categories"].items()},
              flush=True)


def main():
    amend = D.read(C.OUT / "AMENDMENT_3.json")
    require(sha(D.ROOT / "exps/rh01/eval_r15.py") == amend["new_files"]["exps/rh01/eval_r15.py"], "eval script changed")
    RR.checkpoints = r15_only                       # the locked scorer, restricted to existing R15 checkpoints
    RR.score(list(SEEDS_SCORED))
    RA.metrics(list(SEEDS_SCORED))
    forgetting()


if __name__ == "__main__":
    main()
