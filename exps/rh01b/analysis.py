# exps/rh01b/analysis.py
"""RH-01b metrics, futility and the preregistered decision (identical rules to RH-01 §5-6; RH-01b outputs).

  python -m exps.rh01b.analysis metrics --seeds 2711
  python -m exps.rh01b.analysis futility
  python -m exps.rh01b.analysis metrics --seeds 2712 2713
  python -m exps.rh01b.analysis decide
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from exps.dinomaly import analysis as DA
from exps.dinomaly import common as D
from exps.dinomaly import readout as R
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.rh01b import common as C
from exps.rh01b import run as RR
from exps.task_p.p1_readout import native_map
from exps.task_p.p2.metrics import exact_ranking

OUT = C.OUT
ACQ = {c: C.STREAM.index(c) + 1 for c in C.EVAL}            # bottle 1, carpet 4, hazelnut 6
FINAL = len(C.STREAM)


def metrics(seeds):
    """Exact native metrics and paired bootstrap replicates for one scored seed batch (one-shot)."""
    DR.configure()
    proto = RR.verify_lock()
    tag = "_".join(map(str, seeds))
    target = OUT / f"metrics/METRICS_{tag}.json"
    require(not target.exists(), "metrics are one-shot per seed batch")
    scores = D.read(OUT / f"scores/SCORES_{tag}.json")
    w = DA.draws(proto)
    check = DA.verify_draws(proto, w)
    rows = {r["relative_path"]: r for c in C.EVAL for r in proto["evaluation"][c]}
    masks = {p: torch.from_numpy(DA.native_mask(r)) for p, r in rows.items()}
    points, reps = {}, {}
    for key, rec in scores.items():
        path = D.ROOT / rec["maps"]
        require(sha(path) == rec["sha256"], f"maps changed: {path}")
        d = torch.load(path, weights_only=True)
        pos = {p: i for i, p in enumerate(d["paths"])}
        pooled = {ro: ([], []) for ro in DA.READOUTS}
        for c in rec["admitted"]:
            group = proto["evaluation"][c]
            labels = [r["label"] for r in group]
            for ro in DA.READOUTS:
                maps, image = [], []
                for r in group:
                    i = pos[r["relative_path"]]
                    if ro == "RD":
                        maps.append(R.native(d["s256"][i].cuda(), r["hw"]))
                        image.append(float(d["image_RD"][i]))
                    else:
                        maps.append(torch.from_numpy(native_map(d["grid"][i].flatten(), r["hw"])).cuda())
                        image.append(float(d["grid"][i].max()))
                targets = [masks[r["relative_path"]].cuda() for r in group]
                m, b = DA.category_metrics(maps, targets, labels, image, w[c])
                points[f"{key}/{c}/{ro}"], reps[f"{key}/{c}/{ro}"] = m, b
                pooled[ro][0].extend(x.cpu() for x in maps)
                pooled[ro][1].extend(t.cpu() for t in targets)
                del maps, targets
        for ro in DA.READOUTS:
            points[f"{key}/pooled/{ro}"] = exact_ranking(torch.cat([x.flatten() for x in pooled[ro][0]]),
                                                         torch.cat([t.flatten() for t in pooled[ro][1]]))
        print("metrics", key, {c: round(points[f"{key}/{c}/RD"]["pixel_ap"], 4) for c in rec["admitted"]}, flush=True)
    D.write_json(target, {"points": points, "draw_check_max_abs": check})
    D.write_json(OUT / f"metrics/REPLICATES_{tag}.json", reps)


class Table:
    def __init__(self):
        self.p, self.b = {}, {}
        for f in sorted((OUT / "metrics").glob("METRICS_*.json")):
            self.p.update(D.read(f)["points"])
            self.b.update({k: {m: np.array(v) for m, v in x.items()}
                           for k, x in D.read(f.with_name(f.name.replace("METRICS", "REPLICATES"))).items()})

    def point(self, arm, stage, c, seeds, metric="pixel_ap", ro="RD"):
        return float(np.mean([self.p[f"{arm}/{s}/{stage}/{c}/{ro}"][metric] for s in seeds]))

    def reps(self, arm, stage, c, seeds, metric, ro="RD"):
        return np.mean([self.b[f"{arm}/{s}/{stage}/{c}/{ro}"][metric] for s in seeds], axis=0)

    def macro(self, arm, stage, seeds, metric="pixel_ap", ro="RD"):
        return float(np.mean([self.point(arm, stage, c, seeds, metric, ro) for c in C.EVAL]))

    def macro_lohi(self, arm, stage, seeds):
        return (np.mean([self.reps(arm, stage, c, seeds, "ap_low") for c in C.EVAL], axis=0),
                np.mean([self.reps(arm, stage, c, seeds, "ap_high") for c in C.EVAL], axis=0))


def summarize(seeds):
    t = Table()
    j_lo, j_hi = t.macro_lohi("J15", 1, seeds)
    r_lo, r_hi = t.macro_lohi("R15", FINAL, seeds)
    g = t.macro("J15", 1, seeds) - t.macro("R15", FINAL, seeds)
    f_lo = np.mean([t.reps("R15", ACQ[c], c, seeds, "ap_low") - t.reps("R15", FINAL, c, seeds, "ap_high")
                    for c in C.EVAL], axis=0)
    f_hi = np.mean([t.reps("R15", ACQ[c], c, seeds, "ap_high") - t.reps("R15", FINAL, c, seeds, "ap_low")
                    for c in C.EVAL], axis=0)
    cat = {}
    for c in C.EVAL:
        jl, jh = t.reps("J15", 1, c, seeds, "ap_low"), t.reps("J15", 1, c, seeds, "ap_high")
        rl, rh = t.reps("R15", FINAL, c, seeds, "ap_low"), t.reps("R15", FINAL, c, seeds, "ap_high")
        al, ah = t.reps("R15", ACQ[c], c, seeds, "ap_low"), t.reps("R15", ACQ[c], c, seeds, "ap_high")
        cat[c] = {"J15": t.point("J15", 1, c, seeds), "R15_acq": t.point("R15", ACQ[c], c, seeds),
                  "R15_final": t.point("R15", FINAL, c, seeds),
                  "J_minus_R": t.point("J15", 1, c, seeds) - t.point("R15", FINAL, c, seeds),
                  "J_minus_R_ci95": DA.diff_ci(jl, jh, rl, rh),
                  "F": t.point("R15", ACQ[c], c, seeds) - t.point("R15", FINAL, c, seeds),
                  "F_ci95": DA.diff_ci(al, ah, rl, rh),
                  "J15_ci95": [float(np.quantile(jl, .025)), float(np.quantile(jh, .975))],
                  "R15_final_ci95": [float(np.quantile(rl, .025)), float(np.quantile(rh, .975))],
                  "trajectory": {k: t.point("R15", k, c, seeds) for k in range(ACQ[c], FINAL + 1)},
                  "per_seed": {s: {"J15": t.point("J15", 1, c, (s,)), "R15_acq": t.point("R15", ACQ[c], c, (s,)),
                                   "R15_final": t.point("R15", FINAL, c, (s,))} for s in seeds},
                  "image_auroc": {"J15": t.point("J15", 1, c, seeds, "image_auroc"),
                                  "R15_final": t.point("R15", FINAL, c, seeds, "image_auroc")}}
    return {"seeds": list(seeds), "G15": g, "G15_ci95": DA.diff_ci(j_lo, j_hi, r_lo, r_hi),
            "G15_per_seed": {s: t.macro("J15", 1, (s,)) - t.macro("R15", FINAL, (s,)) for s in seeds},
            "Fbar": float(np.mean([cat[c]["F"] for c in C.EVAL])),
            "Fbar_ci95": [float(np.quantile(f_lo, .025)), float(np.quantile(f_hi, .975))],
            "Fbar_per_seed": {s: float(np.mean([t.point("R15", ACQ[c], c, (s,)) - t.point("R15", FINAL, c, (s,))
                                                for c in C.EVAL])) for s in seeds},
            "macro": {"J15": t.macro("J15", 1, seeds), "R15_final": t.macro("R15", FINAL, seeds),
                      "J15_ci95": [float(np.quantile(j_lo, .025)), float(np.quantile(j_hi, .975))],
                      "R15_final_ci95": [float(np.quantile(r_lo, .025)), float(np.quantile(r_hi, .975))],
                      "J15_image_auroc": t.macro("J15", 1, seeds, "image_auroc"),
                      "R15_final_image_auroc": t.macro("R15", FINAL, seeds, "image_auroc"),
                      "RP1_J15": t.macro("J15", 1, seeds, ro="RP1"), "RP1_R15_final": t.macro("R15", FINAL, seeds, ro="RP1")},
            "pooled": {"J15": float(np.mean([t.p[f"J15/{s}/1/pooled/RD"]["pixel_ap"] for s in seeds])),
                       "R15_final": float(np.mean([t.p[f"R15/{s}/{FINAL}/pooled/RD"]["pixel_ap"] for s in seeds]))},
            "categories": cat}


def futility():
    RR.verify_lock()
    target = OUT / "FUTILITY.json"
    require(not target.exists(), "one-shot")
    res = summarize((2711,))
    res["futile"] = bool(res["G15"] < 0.01)
    res["decision"] = "REPLAY_SUFFICIENT_15" if res["futile"] else "CONTINUE"
    D.write_json(target, res)
    print("futility", res["decision"], "G15(2711)", round(res["G15"], 4), "Fbar", round(res["Fbar"], 4), flush=True)


def decide():
    RR.verify_lock()
    target = OUT / "DECISION.json"
    require(not target.exists(), "one-shot")
    fut = D.read(OUT / "FUTILITY.json")
    if fut["futile"]:
        res = {**fut, "decision": "REPLAY_SUFFICIENT_15", "basis": "futility stop after seed 2711"}
    else:
        res = summarize(RR.SEEDS)
        g, lo, ps = res["G15"], res["G15_ci95"][0], res["G15_per_seed"].values()
        if g >= 0.05 and lo > 0 and all(v >= 0.025 for v in ps):
            res["decision"] = "REPLAY_HEADROOM_15"
        elif g < 0.03 or lo <= 0:
            res["decision"] = "REPLAY_SUFFICIENT_15"
        else:
            res["decision"] = "AMBIGUOUS_15"
        res["flag_forgetting_without_joint_gap"] = bool(res["Fbar"] >= 0.05 and res["Fbar_ci95"][0] > 0
                                                        and res["decision"] != "REPLAY_HEADROOM_15")
    D.write_json(target, res)
    print(res["decision"], "G15", round(res["G15"], 4), res["G15_ci95"], "Fbar", round(res["Fbar"], 4), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("metrics", "futility", "decide"))
    ap.add_argument("--seeds", type=int, nargs="+")
    a = ap.parse_args()
    {"metrics": lambda: metrics(a.seeds), "futility": futility, "decide": decide}[a.stage]()


if __name__ == "__main__":
    main()
