"""SCRC-P1-LITE-00 Phase L — metrics, fold selection, screening statistics and outcome (prereg §7-§10).

    python -m exps.scrc_p1_lite_00.evaluate --owner-authorized-A5      # real run (owner checklist A5 required)

CPU only (CUDA is hidden before any CUDA call and asserted uninitialised at the end). Refuses to run unless
ENGINEERING_LOCK.json and SCORES_LOCK.json verify (every score-file hash) and no RESULT.json exists. Custody phase L:
the AM03 DATASET_MANIFEST.json and data/mvtec/<c>/ground_truth/** masks are opened only here, after SCORES_LOCK.

Metric definitions are the C2-DOM evaluator's: am03_eval.native_maps (FP32 cast, bilinear 28x28 -> native,
align_corners=False, antialias=False), image score = map max, native Pixel AP = cadfinal.metrics.exact_ap over all
pixels of the category, Image AUROC = sklearn roc_auc_score. Pooled native Pixel AP (descriptive) uses the project's
65,536-bin histogram rule (cadfinal.metrics.hist / ap_from_hist; exact pooled AP does not fit in RAM), with the
per-category binning error reported.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from exps.scrc_p1_lite_00 import custody as CU
from exps.scrc_p1_lite_00 import protocol as P

POOL_BINS = 65_536


# ------------------------------------------------------------------ statistics (prereg §8-§10; test U9)

def select_config(ap, cats):
    """argmax over M01..M12 of macro Pixel AP over ``cats``; means within SEL_TIE_EPS of the best -> lowest ID."""
    means = {j: float(np.mean([ap[j][c] for c in cats])) for j in P.METRIC_ARMS}
    best = max(means.values())
    winners = [j for j in P.METRIC_ARMS if means[j] >= best - P.SEL_TIE_EPS]
    return winners[0], means


def fold_selection(ap):
    out = {}
    for f, held in P.FOLDS.items():
        sel_cats = [c for c in P.ORDER if c not in held]
        j, means = select_config(ap, sel_cats)
        out[f] = {"selected": j, "selection_categories": sel_cats, "selection_means": means, "held_out": list(held)}
    return out


def deltas(ap, auc, sel):
    out = {k: {} for k in ("net", "mech", "auc", "rows")}
    for c in P.ORDER:
        j = sel[P.fold_of(c)]["selected"]
        out["net"][c] = ap[j][c] - ap["R0"][c]
        out["mech"][c] = ap[j][c] - ap["C0"][c]
        out["auc"][c] = auc[j][c] - auc["R0"][c]
        out["rows"][c] = ap["C0"][c] - ap["R0"][c]
    return out


def bootstrap(arrays, B=P.BOOT_B, seed=P.BOOT_SEED):
    """Category-cluster bootstrap: the same B x 15 index draws for every statistic; 95 % percentile interval."""
    rng = np.random.default_rng(seed)
    n = len(P.ORDER)
    idx = rng.integers(0, n, size=(B, n))
    out = {}
    for k, v in arrays.items():
        v = np.asarray(v, dtype=np.float64)
        m = v[idx].mean(1)
        lo, hi = np.percentile(m, [2.5, 97.5])
        out[k] = {"point": float(v.mean()), "lo95": float(lo), "hi95": float(hi), "B": B, "seed": seed,
                  "unit": "category"}
    return out


def sign_flip_p(d):
    """Exact one-sided sign-flip randomisation p for mean(d) over all 2^n sign vectors (row 0 = identity)."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[None]) & 1
    stats = ((1 - 2 * bits) * d[None]).mean(1)
    return float(np.mean(stats >= stats[0])), int(2 ** n)


def decide(invalid, stopped, S_net, S_mech, S_auc, fold_means, lb_net):
    """Outcome in the locked order of prereg §10."""
    if invalid:
        return "SCREEN_INVALID"
    if stopped:
        return "SCREEN_STOPPED_COST"
    if S_net < P.SCREEN_THRESHOLD or S_auc < P.AUROC_NI:
        return "SCREEN_FAIL"
    weak = S_mech < P.MECH_THRESHOLD or sum(1 for v in fold_means.values() if v > 0) < 2 or lb_net <= 0
    return "SCREEN_WEAK" if weak else "SCREEN_PASS"


def screen(ap, auc):
    """Everything §8-§10 needs from per-(arm, category) Pixel AP / AUROC tables."""
    sel = fold_selection(ap)
    dl = deltas(ap, auc, sel)
    arr = {k: [dl[k][c] for c in P.ORDER] for k in dl}
    boot = bootstrap({"S_net": arr["net"], "S_mech": arr["mech"], "S_auc": arr["auc"], "S_rows": arr["rows"]})
    fold_means = {f: float(np.mean([dl["net"][c] for c in held])) for f, held in P.FOLDS.items()}
    p_sf, n_sf = sign_flip_p(arr["net"])
    S_net, S_mech, S_auc = (float(np.mean(arr[k])) for k in ("net", "mech", "auc"))
    out = {"fold_selection": sel, "deltas_per_category": dl, "S_net": S_net, "S_mech": S_mech, "S_auc": S_auc,
           "S_rows_descriptive": float(np.mean(arr["rows"])), "fold_means_net": fold_means,
           "folds_positive": sum(1 for v in fold_means.values() if v > 0), "bootstrap": boot,
           "sign_flip_one_sided_p": p_sf, "sign_flip_vectors": n_sf,
           "categories_net_positive": int(sum(1 for v in arr["net"] if v > 0)),
           "selection_biased_descriptive": {j: {"macro_pixel_ap_all15": float(np.mean([ap[j][c] for c in P.ORDER])),
                                                "macro_image_auroc_all15": float(np.mean([auc[j][c] for c in P.ORDER]))}
                                            for j in P.ARMS},
           "thresholds": {"screen_exploratory_S_net": P.SCREEN_THRESHOLD, "S_mech": P.MECH_THRESHOLD,
                          "S_auc_noninferiority": P.AUROC_NI,
                          "note": "exploratory screen; NOT G-C1 (+0.010, 95% LB>0, VisA-DEV) and NOT Gate P1"}}
    out["outcome"] = decide(False, False, S_net, S_mech, S_auc, fold_means, boot["S_net"]["lo95"])
    return out


# ------------------------------------------------------------------ per-category metrics

def category_metrics(d, labels, masks, hw, hi_pool=None):
    from sklearn.metrics import roc_auc_score
    from exps.cadfinal import metrics as M
    from exps.nl_sgm_01a import am03_eval as A3
    maps = A3.native_maps(torch.as_tensor(d), hw)
    P.require(np.isfinite(maps).all() and float(maps.min()) >= 0.0, "bad native map")
    ap = M.exact_ap(list(maps), masks)["ap"]
    img = maps.reshape(len(maps), -1).max(1)
    auc = float(roc_auc_score(labels, img))
    anom = np.flatnonzero(labels == 1)
    inm = float(np.mean([masks[k].reshape(-1)[int(maps[k].argmax())] for k in anom]))
    rec = {"pixel_ap": float(ap), "image_auroc": auc, "n_images": int(len(labels)), "n_anomalous": int(labels.sum()),
           "positive_pixels": int(sum(int(m.sum()) for m in masks)), "argmax_in_mask": inm}
    hist = None
    if hi_pool is not None:
        pp, pt = np.zeros(POOL_BINS), np.zeros(POOL_BINS)
        for m, y in zip(maps, masks):
            a, b = M.hist(m, y, 0.0, hi_pool, POOL_BINS)
            pp += a; pt += b
        rec["pixel_ap_hist"] = M.ap_from_hist(pp, pt)
        rec["hist_binning_error"] = abs(rec["pixel_ap_hist"] - rec["pixel_ap"])
        hist = (pp, pt)
    return rec, hist


def real_labels(c):
    from exps.nl_sgm_01a import am03_eval as A3
    labels, masks, items = A3.load_masks(c)
    hws = {tuple(x["hw"]) for x in items}
    P.require(len(hws) == 1, f"{c}: mixed native sizes")
    hw = hws.pop()
    ff = CU.guarded_features(P.ROOT / f"results/hope_cad/calm02/features/mvtec/{c}_test.pt")
    rel = [r["relative_path"] for r in ff["rows"]]
    P.require(rel == [x["relative_path"] for x in items], f"{c}: feature order differs from the manifest")
    P.require(all(tuple(m.shape) == hw for m in masks), f"{c}: mask size")
    return labels, masks, hw


def smoke_labels(c, n):
    """Synthetic labels/masks for the engineering smoke: image 0 normal, the others anomalous with a box."""
    H, W = P.MVTEC_HW[c]
    labels = np.array([0] + [1] * (n - 1), dtype=np.int8)
    masks = []
    for k in range(n):
        m = np.zeros((H, W), dtype=bool)
        if labels[k]:
            m[H // 4:H // 2, W // 4:W // 2] = True
        masks.append(m)
    return labels, masks, (H, W)


def run(mode, out_dir, label_provider, log=print):
    from exps.scrc_p1_lite_00 import score as SC
    from exps.cadfinal import metrics as M
    out_dir = Path(out_dir)
    t0 = time.time()
    P.require(not (out_dir / "RESULT.json").exists(), "RESULT.json exists (create-exclusive)")
    sl = SC.verify_scores_lock(out_dir)
    if mode == "real":
        P.verify_engineering_lock()
    hi_pool = {a: sl["diagnostics"]["max_score_fp32"][a] for a in P.ARMS}
    ap, auc, per = {a: {} for a in P.ARMS}, {a: {} for a in P.ARMS}, {a: {} for a in P.ARMS}
    pool = {a: [np.zeros(POOL_BINS), np.zeros(POOL_BINS)] for a in P.ARMS}
    for c in P.ORDER:
        tc = time.time()
        n_c = sl["categories"][c]["n_images"]
        labels, masks, hw = label_provider(c, n_c) if mode == "smoke" else label_provider(c)
        P.require(len(labels) == n_c == len(masks), f"{c}: label count {len(labels)} != scored images {n_c}")
        for a in P.ARMS:
            d = torch.load(out_dir / "scores" / a / f"{c}.pt", weights_only=True)["d"]
            rec, h = category_metrics(d, labels, masks, hw, hi_pool[a])
            per[a][c] = rec
            ap[a][c], auc[a][c] = rec["pixel_ap"], rec["image_auroc"]
            pool[a][0] += h[0]; pool[a][1] += h[1]
        log("metrics", c, round(time.time() - tc, 1), "s")
        wall = time.time() - t0
        P.require(wall <= P.CAPS["phase_L_cpu_wall_h"] * 3600, f"CAP: Phase L wall {wall:.0f} s")
        P.require(wall + sl["timing"]["phase_S_wall_s"] <= P.CAPS["total_wall_h"] * 3600, "CAP: total wall")
        P.require(SC.peak_rss_gb() <= P.CAPS["rss_gb"], "CAP: RSS")
    res = screen(ap, auc)
    res.update(schema="scrc_p1_lite_00_result_v1", mode=mode, label="EXPOSED_SOURCE_EXPLORATORY",
               scores_lock_sha256=P.sha_file(out_dir / "SCORES_LOCK.json"),
               macro={a: {"pixel_ap": float(np.mean([ap[a][c] for c in P.ORDER])),
                          "image_auroc": float(np.mean([auc[a][c] for c in P.ORDER]))} for a in P.ARMS},
               pooled_pixel_ap_hist65536={a: M.ap_from_hist(*pool[a]) for a in P.ARMS},
               pooled_binning_error_max={a: max(per[a][c]["hist_binning_error"] for c in P.ORDER) for a in P.ARMS},
               phase_L_wall_s=time.time() - t0, cuda_initialized=bool(torch.cuda.is_initialized()),
               access_log=CU.access_log())
    P.require(not res["cuda_initialized"], "Phase L initialised CUDA")
    SC.write_json_x(out_dir / "METRICS_L.json", {"per_arm_category": per, "label": "EXPOSED_SOURCE_EXPLORATORY"})
    SC.write_json_x(out_dir / "RESULT.json", res)
    log("outcome", res["outcome"], "S_net", round(res["S_net"], 5))
    return res


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner-authorized-A5", action="store_true", dest="a5")
    args = ap.parse_args()
    P.require(args.a5, "Phase L requires owner checklist item A5 (--owner-authorized-A5)")
    CU.install("L")
    torch.set_num_threads(P.CAPS["threads"])
    run("real", P.OUT, real_labels)


if __name__ == "__main__":
    main()
