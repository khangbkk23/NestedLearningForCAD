# exps/dinomaly/analysis.py
"""DRF-01 metrics and preregistered gate decisions (task file section 6-7).

  python -m exps.dinomaly.analysis gate0
  python -m exps.dinomaly.analysis gate1
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from exps.dinomaly import common as C
from exps.dinomaly import readout as R
from exps.dinomaly.common import require
from exps.task_p.p1_readout import bootstrap_weights, native_map
from exps.task_p.p2.metrics import bootstrap_bounds, exact_ranking, image_auroc

OUT = C.OUT
READOUTS = ("RD", "RP1")
P1_ARMS = ("FULL784_GREEDY", "FULL784_RESERVOIR")


def native_mask(row):
    h, w = row["hw"]
    if row["label"] == 0:
        return np.zeros((h, w), dtype=bool)
    path = C.DATA / row["mask_path"]
    require(C.sha(path) == row["mask_sha256"], "mask source changed")
    with Image.open(path) as im:
        m = np.array(im)
    require(m.shape == (h, w) and np.isin(m, [0, 1, 255]).all(), "mask geometry/values")
    return m > 0


def draws(proto):
    """Identical to P1/P2: class-stratified whole-image resamples, seed 102701 + category index."""
    out = {}
    for i, c in enumerate(C.CATS):
        gen = np.random.default_rng(102701 + i)
        labels = [r["label"] for r in proto["evaluation"][c]]
        out[c] = np.stack([bootstrap_weights(labels, gen) for _ in range(1000)])
    return out


def verify_draws(proto, w):
    """The draws must reproduce P1's stored image-AUROC replicates exactly (pairing check)."""
    preds = C.read(C.P1 / "predictions_t350.json")
    boot = C.read(C.P1 / "bootstrap.json")
    worst = 0.
    for c in C.CATS:
        score = {r["source"]["relative_path"]: r["image_score"] for r in preds if r["arm"] == "FULL784_GREEDY"}
        rows = proto["evaluation"][c]
        auc = image_auroc([r["label"] for r in rows], [score[r["relative_path"]] for r in rows], w[c])
        worst = max(worst, float(np.abs(auc - np.array(boot[f"350/{c}/FULL784_GREEDY"]["image_auroc"])).max()))
    require(worst < 1e-12, f"bootstrap draws differ from P1 ({worst})")
    return worst


def category_metrics(maps, masks, labels, scores, w):
    m = exact_ranking(torch.cat([x.flatten() for x in maps]), torch.cat([t.flatten() for t in masks]))
    m["image_auroc"] = float(image_auroc(labels, scores, np.ones(len(labels)))[0])
    low, high = bootstrap_bounds(maps, masks, w)
    labels, scores = np.array(labels), np.array(scores)
    pos = sum(float(x[t].double().sum()) for x, t in zip(maps, masks))
    neg = sum(float(x[~t].double().sum()) for x, t in zip(maps, masks))
    per_image_ap = [exact_ranking(x.flatten(), t.flatten())["pixel_ap"] if l == 1 else None
                    for x, t, l in zip(maps, masks, labels)]
    m["separation"] = {"normal_image_median_p95_max": [float(np.median(scores[labels == 0])),
                                                       float(np.quantile(scores[labels == 0], .95)),
                                                       float(scores[labels == 0].max())],
                       "anomaly_image_mean": float(scores[labels == 1].mean()),
                       "positive_pixel_mean": pos / m["positive_pixels"],
                       "negative_pixel_mean": neg / (m["native_pixels"] - m["positive_pixels"]),
                       "per_image_ap_defects": per_image_ap}
    reps = {"ap_low": low.tolist(), "ap_high": high.tolist(), "image_auroc": image_auroc(labels, scores, w).tolist()}
    return m, reps


def compute(gate):
    """Per (arm/seed/stage/category/readout) exact metrics and paired bootstrap replicates; one-shot."""
    from exps.dinomaly.run import configure, verify_lock
    configure()
    proto = verify_lock()
    target = OUT / f"gate{gate}/METRICS.json"
    require(not target.exists(), "gate analysis is one-shot")
    scores = C.read(OUT / f"gate{gate}/SCORES.json")
    w = draws(proto)
    draw_check = verify_draws(proto, w)
    rows = {r["relative_path"]: r for c in C.CATS for r in proto["evaluation"][c]}
    masks = {p: torch.from_numpy(native_mask(r)) for p, r in rows.items()}
    points, reps = {}, {}
    for key, rec in scores.items():
        path = C.ROOT / rec["maps"]
        require(C.sha(path) == rec["sha256"], f"maps changed: {path}")
        d = torch.load(path, weights_only=True)
        pos = {p: i for i, p in enumerate(d["paths"])}
        pooled = {r: ([], []) for r in READOUTS}
        for c in rec["admitted"]:
            group = proto["evaluation"][c]
            labels = [r["label"] for r in group]
            for ro in READOUTS:
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
                m, b = category_metrics(maps, targets, labels, image, w[c])
                points[f"{key}/{c}/{ro}"], reps[f"{key}/{c}/{ro}"] = m, b
                pooled[ro][0].extend(x.cpu() for x in maps)
                pooled[ro][1].extend(t.cpu() for t in targets)
                del maps, targets
        for ro in READOUTS:
            points[f"{key}/pooled/{ro}"] = exact_ranking(torch.cat([x.flatten() for x in pooled[ro][0]]),
                                                         torch.cat([t.flatten() for t in pooled[ro][1]]))
        print("metrics", key, {c: round(points[f"{key}/{c}/RD"]["pixel_ap"], 4) for c in rec["admitted"]}, flush=True)
    C.write_json(target, {"points": points, "draw_check_max_abs": draw_check})
    C.write_json(OUT / f"gate{gate}/REPLICATES.json", reps)


# --- aggregation -------------------------------------------------------------------------------------


class Table:
    def __init__(self, gates):
        self.p, self.b = {}, {}
        for g in gates:
            self.p.update(C.read(OUT / f"gate{g}/METRICS.json")["points"])
            self.b.update({k: {m: np.array(v) for m, v in x.items()} for k, x in
                           C.read(OUT / f"gate{g}/REPLICATES.json").items()})

    def point(self, arm, stage, c, metric="pixel_ap", ro="RD", seeds=None):
        from exps.dinomaly.run import SEEDS
        return float(np.mean([self.p[f"{arm}/{s}/{stage}/{c}/{ro}"][metric] for s in (seeds or SEEDS)]))

    def reps(self, arm, stage, c, metric, ro="RD"):
        from exps.dinomaly.run import SEEDS
        return np.mean([self.b[f"{arm}/{s}/{stage}/{c}/{ro}"][metric] for s in SEEDS], axis=0)

    def macro(self, arm, stage, metric="pixel_ap", ro="RD", seeds=None):
        return float(np.mean([self.point(arm, stage, c, metric, ro, seeds) for c in C.CATS]))

    def macro_ci(self, arm, stage, ro="RD"):
        low = np.mean([self.reps(arm, stage, c, "ap_low", ro) for c in C.CATS], axis=0)
        high = np.mean([self.reps(arm, stage, c, "ap_high", ro) for c in C.CATS], axis=0)
        return [float(np.quantile(low, .025)), float(np.quantile(high, .975))], low, high


def diff_ci(x_low, x_high, y_low, y_high):
    """Conservative paired CI of X - Y from per-replicate AP bounds."""
    return [float(np.quantile(x_low - y_high, .025)), float(np.quantile(x_high - y_low, .975))]


def p1_context(t, low_c, high_c):
    boot = C.read(C.P1 / "bootstrap.json")
    summ = C.read(C.P1 / "summary.json")["per_boundary_category"]
    out = {}
    for arm in P1_ARMS:
        lo = np.mean([boot[f"350/{c}/{arm}"]["ap_lower"] for c in C.CATS], axis=0)
        hi = np.mean([boot[f"350/{c}/{arm}"]["ap_upper"] for c in C.CATS], axis=0)
        macro = float(np.mean([summ[f"350/{c}/{arm}"]["pixel_ap"] for c in C.CATS]))
        out[arm] = {"macro_ap": macro, "per_category": {c: summ[f"350/{c}/{arm}"]["pixel_ap"] for c in C.CATS},
                    "joint_C_minus_this": t.macro("C", 1) - macro, "ci95_conservative": diff_ci(low_c, high_c, lo, hi)}
    return out


def p2_context():
    pm = C.read(C.ROOT / "results/hope_cad/task_p/p2/point_metrics.json")
    out = {}
    for arm in ("A", "B"):
        per = {c: float(np.mean([pm[f"canonical/{s}/350/{c}/{arm}"]["pixel_ap"] for s in (2711, 2712, 2713)]))
               for c in C.CATS}
        out[f"P2_{arm}"] = {"per_category": per, "macro_ap": float(np.mean(list(per.values())))}
    return out


def gate0():
    from exps.dinomaly.run import SEEDS
    target = OUT / "GATE0.json"
    require(not target.exists(), "one-shot")
    t = Table((0,))
    cats = {}
    for c in C.CATS:
        ap, auc, prev = t.point("C", 1, c), t.point("C", 1, c, "image_auroc"), t.point("C", 1, c, "prevalence")
        a_rep = t.reps("C", 1, c, "image_auroc")
        cats[c] = {"pixel_ap": ap, "image_auroc": auc, "pixel_auroc": t.point("C", 1, c, "pixel_auroc"),
                   "prevalence": prev, "useful": bool(auc >= .70 and ap >= max(3 * prev, prev + .05)),
                   "ap_ci95": [float(np.quantile(t.reps("C", 1, c, "ap_low"), .025)),
                               float(np.quantile(t.reps("C", 1, c, "ap_high"), .975))],
                   "image_auroc_ci95": np.quantile(a_rep, [.025, .975]).tolist(),
                   "per_seed_ap": [t.point("C", 1, c, seeds=(s,)) for s in SEEDS],
                   "RP1_pixel_ap": t.point("C", 1, c, ro="RP1"), "RP1_image_auroc": t.point("C", 1, c, "image_auroc", "RP1")}
    macro = t.macro("C", 1)
    ci, low, high = t.macro_ci("C", 1)
    auc_rep = np.mean([t.reps("C", 1, c, "image_auroc") for c in C.CATS], axis=0)
    prev = float(np.mean([cats[c]["prevalence"] for c in C.CATS]))
    per_seed = {s: t.macro("C", 1, seeds=(s,)) for s in SEEDS}
    pooled = float(np.mean([t.p[f"C/{s}/1/pooled/RD"]["pixel_ap"] for s in SEEDS]))
    v1 = macro >= .60
    v2 = bool(sum(cats[c]["useful"] for c in C.CATS) >= 2 and np.quantile(auc_rep, .025) > .5 and ci[0] > prev)
    v3 = all(v >= .50 for v in per_seed.values())
    res = {"categories": cats, "macro_ap": macro, "macro_ap_ci95": ci, "macro_prevalence": prev,
           "macro_image_auroc": t.macro("C", 1, "image_auroc"), "macro_image_auroc_ci95": np.quantile(auc_rep, [.025, .975]).tolist(),
           "macro_pixel_auroc": t.macro("C", 1, "pixel_auroc"), "pooled_ap": pooled, "per_seed_macro_ap": per_seed,
           "per_seed_pooled_ap": {s: t.p[f"C/{s}/1/pooled/RD"]["pixel_ap"] for s in SEEDS},
           "RP1": {"macro_ap": t.macro("C", 1, ro="RP1"), "macro_image_auroc": t.macro("C", 1, "image_auroc", "RP1"),
                   "pooled_ap": float(np.mean([t.p[f"C/{s}/1/pooled/RP1"]["pixel_ap"] for s in SEEDS]))},
           "criteria": {"V1_macro_ap_ge_0.60": bool(v1), "V2_useful_signal": v2, "V3_every_seed_ge_0.50": bool(v3)},
           "viable": bool(v1 and v2 and v3), "context_P1": p1_context(t, low, high), "context_P2": p2_context()}
    res["verdict_gate0"] = "RECONSTRUCTION_VIABLE" if res["viable"] else "RECONSTRUCTION_NOT_VIABLE"
    C.write_json(target, res)
    print(res["verdict_gate0"], "macro", round(macro, 4), ci, "pooled", round(pooled, 4), res["criteria"], flush=True)


def gate1():
    from exps.dinomaly.run import SEEDS
    target = OUT / "GATE1.json"
    require(not target.exists(), "one-shot")
    require(C.read(OUT / "GATE0.json")["viable"], "Gate 0 did not pass")
    t = Table((0, 1))
    # A and B share the bottle acquisition checkpoint (A stage 1).
    def forgetting(arm, seeds=None):
        out = {}
        for c, (stage_acq_arm, s_acq) in (("bottle", ("A", 1)), ("carpet", (arm, 2))):
            a = t.point(stage_acq_arm, s_acq, c, seeds=seeds)
            f = t.point(arm, 3, c, seeds=seeds)
            out[c] = {"acquisition": a, "final": f, "F": a - f}
        out["mean_F"] = float(np.mean([out[c]["F"] for c in ("bottle", "carpet")]))
        return out

    def forgetting_ci(arm):
        lows, highs = [], []
        for c, (aa, sa) in (("bottle", ("A", 1)), ("carpet", (arm, 2))):
            lows.append(t.reps(aa, sa, c, "ap_low") - t.reps(arm, 3, c, "ap_high"))
            highs.append(t.reps(aa, sa, c, "ap_high") - t.reps(arm, 3, c, "ap_low"))
        lo, hi = np.mean(lows, axis=0), np.mean(highs, axis=0)
        return [float(np.quantile(lo, .025)), float(np.quantile(hi, .975))]

    _, c_low, c_high = t.macro_ci("C", 1)
    res = {"arms": {}}
    for arm in ("A", "B"):
        _, a_low, a_high = t.macro_ci(arm, 3)
        per_seed = {s: {"mean_F": forgetting(arm, (s,))["mean_F"],
                        "G": t.macro("C", 1, seeds=(s,)) - t.macro(arm, 3, seeds=(s,))} for s in SEEDS}
        res["arms"][arm] = {
            "forgetting": forgetting(arm), "mean_F_ci95": forgetting_ci(arm),
            "final_macro_ap": t.macro(arm, 3), "final_macro_ap_ci95": t.macro_ci(arm, 3)[0],
            "final_per_category": {c: t.point(arm, 3, c) for c in C.CATS},
            "final_pooled_ap": float(np.mean([t.p[f"{arm}/{s}/3/pooled/RD"]["pixel_ap"] for s in SEEDS])),
            "final_macro_image_auroc": t.macro(arm, 3, "image_auroc"),
            "G": t.macro("C", 1) - t.macro(arm, 3), "G_ci95": diff_ci(c_low, c_high, a_low, a_high),
            "per_seed": per_seed,
            "RP1": {"final_macro_ap": t.macro(arm, 3, ro="RP1"),
                    "mean_F": float(np.mean([t.point("A" if c == "bottle" else arm, 1 if c == "bottle" else 2, c, ro="RP1")
                                             - t.point(arm, 3, c, ro="RP1") for c in ("bottle", "carpet")])),
                    "G": t.macro("C", 1, ro="RP1") - t.macro(arm, 3, ro="RP1")}}
    _, b_low, b_high = t.macro_ci("B", 3)
    _, a_low, a_high = t.macro_ci("A", 3)
    res["B_minus_A_final_macro"] = {"point": t.macro("B", 3) - t.macro("A", 3), "ci95": diff_ci(b_low, b_high, a_low, a_high)}
    res["joint_C"] = {"macro_ap": t.macro("C", 1), "per_category": {c: t.point("C", 1, c) for c in C.CATS},
                      "pooled_ap": float(np.mean([t.p[f"C/{s}/1/pooled/RD"]["pixel_ap"] for s in SEEDS]))}
    A, B = res["arms"]["A"], res["arms"]["B"]
    h1 = bool(A["forgetting"]["mean_F"] >= .05 and A["mean_F_ci95"][0] > 0 and A["G"] >= .05 and A["G_ci95"][0] > 0
              and all(v["mean_F"] >= .025 and v["G"] >= .025 for v in A["per_seed"].values()))
    h2 = bool((B["G"] >= .03 and B["G_ci95"][0] > 0 and all(v["G"] > 0 for v in B["per_seed"].values()))
              or (B["forgetting"]["mean_F"] >= .03 and B["mean_F_ci95"][0] > 0
                  and all(v["mean_F"] > 0 for v in B["per_seed"].values())))
    res["criteria"] = {"H1_A_headroom": h1, "H2_residual_after_replay": h2}
    res["verdict"] = ("RECONSTRUCTION_VIABLE_NO_CONTINUAL_HEADROOM" if not h1 else
                      "RECONSTRUCTION_VIABLE_REPLAY_SUFFICIENT" if not h2 else "VISIONHOPE_NL_DESIGN_JUSTIFIED")
    res["context_P2"] = p2_context()
    C.write_json(target, res)
    print(res["verdict"], res["criteria"], {a: (round(res["arms"][a]["forgetting"]["mean_F"], 4), round(res["arms"][a]["G"], 4))
                                             for a in ("A", "B")}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("metrics0", "metrics1", "gate0", "gate1"))
    a = ap.parse_args()
    {"metrics0": lambda: compute(0), "metrics1": lambda: compute(1), "gate0": gate0, "gate1": gate1}[a.stage]()


if __name__ == "__main__":
    main()
