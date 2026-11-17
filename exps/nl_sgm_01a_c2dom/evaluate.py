"""NL-SGM-01A-C2-DOM sweep — real-anomaly evaluation with the AM03/F2 scorer (opens MVTec test data).

    python -m exps.nl_sgm_01a_c2dom.evaluate check          # scorer reproduces preserved AM03 cells
    python -m exps.nl_sgm_01a_c2dom.evaluate arm ARM        # all 120 (stage, category) cells of one arm

Scorer (unchanged from AM03/F2): exact FP64 1-NN (exps.nl_sgm_01a.exact.nn) of every test patch to the category's
support at that boundary; AM03 native_maps (bilinear 28x28 -> native HxW); image score = map max; native Pixel AP =
exps.cadfinal.metrics.exact_ap over all pixels of the category; Image AUROC = sklearn roc_auc_score. Masks only for
metrics. A cell whose support (gid set) is identical to the previous boundary's is copied (identical rows ->
identical scores), flagged ``reused_from``.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from exps.cadfinal.metrics import exact_ap
from exps.nl_sgm_01a import am03_eval as A3
from exps.nl_sgm_01a import exact as X
from exps.nl_sgm_01a_c2dom import common as D
from exps.nl_sgm_01a_c2dom import stream as S
from exps.task_p.p0_normal_support import require

EDIR = D.OUT / "eval"


def score(feats, labels, masks, hw, rows):
    t0 = time.time()
    d, _ = X.nn(feats.reshape(-1, 768), rows)
    nn_s = time.time() - t0
    m = A3.native_maps(d.reshape(-1, 784), hw)
    ap = exact_ap(list(m), masks)["ap"]
    img = m.reshape(len(m), -1).max(1)
    anom = np.flatnonzero(labels == 1)
    inm = float(np.mean([masks[k].reshape(-1)[int(m[k].argmax())] for k in anom]))
    return {"pixel_ap": float(ap), "image_auroc": float(roc_auc_score(labels, img)), "n_images": int(len(labels)),
            "n_anomalous": int(labels.sum()), "rows": int(len(rows.ids)), "nn_seconds": nn_s,
            "nn_ms_per_image": 1000 * nn_s / len(labels), "argmax_in_mask": inm,
            "image_scores": [float(v) for v in img], "seconds": time.time() - t0}


def category_data(c, man):
    feats = A3.load_features(c)
    labels, masks, _ = A3.load_masks(c)
    return feats, labels, masks, tuple(man["categories"][c]["images"][0]["hw"])


def do_check():
    D.verify_lock()
    require((D.OUT / "B8_DECISION.json").exists(), "B8 decision must exist before any real-anomaly evaluation")
    man = json.loads((A3.OUT / "DATASET_MANIFEST.json").read_text())
    target = EDIR / "SCORER_CHECK.json"
    require(not target.exists(), "one-shot")
    out = []
    for arm, t, c in (("C1_FPS_FP32", 0, "bottle"), ("C2_FPS_PQ", 4, "grid"), ("C1_FPS_FP32", 14, "zipper")):
        sup, _ = A3.load_support(arm, t)
        f, l, m, hw = category_data(c, man)
        r = score(f, l, m, hw, sup[c])
        ref = json.loads((A3.OUT / "cells" / arm / f"stage{t + 1:02d}_{c}.json").read_text())
        rec = {"arm": arm, "stage": t + 1, "category": c, "ap": r["pixel_ap"], "ap_am03": ref["pixel_ap"],
               "auroc": r["image_auroc"], "auroc_am03": ref["image_auroc"]}
        rec["ok"] = abs(rec["ap"] - rec["ap_am03"]) < 1e-6 and abs(rec["auroc"] - rec["auroc_am03"]) < 1e-9
        out.append(rec)
        print(rec, flush=True)
    EDIR.mkdir(parents=True, exist_ok=True)
    D.write_json(target, {"checks": out, "all_ok": all(r["ok"] for r in out)})
    require(all(r["ok"] for r in out), "SCORER: does not reproduce AM03")


def do_arm(arm):
    D.verify_lock()
    require((D.OUT / "B8_DECISION.json").exists(), "B8 decision must exist before any real-anomaly evaluation")
    require(D.read_json(EDIR / "SCORER_CHECK.json")["all_ok"], "scorer check missing/failed")
    require((D.OUT / "streams" / arm / "STREAM_DONE.json").exists(), f"stream {arm} incomplete")
    man = json.loads((A3.OUT / "DATASET_MANIFEST.json").read_text())
    adir = EDIR / arm
    adir.mkdir(parents=True, exist_ok=True)
    log = D.Log(f"eval_{arm}")
    t_all = time.time()
    bnd = [D.read_json(D.OUT / "streams" / arm / f"boundary_t{t:02d}.json") for t in range(15)]
    for k, c in enumerate(D.ORDER):
        data = None
        prev = None
        for t in range(k, 15):
            p = adir / f"stage{t + 1:02d}_{c}.json"
            gsha = bnd[t]["per_category_gids_sha"][c]
            if p.exists():
                prev = (D.read_json(p), gsha, t); continue
            if prev is not None and prev[1] == gsha:
                cell = dict(prev[0]); cell["reused_from"] = prev[0].get("reused_from", prev[2] + 1)
                cell["seconds"] = 0.0
            else:
                if data is None:
                    data = category_data(c, man)
                rows = S.supports_at(arm, t, only=c)[c]
                require(D.gid_sha(rows.ids) == gsha, "support ids differ from stream ledger")
                cell = score(*data, rows)
            cell.update(arm=arm, stage=t + 1, category=c, gids_sha=gsha)
            D.write_json(p, cell)
            prev = (cell, gsha, t)
            log(arm, "stage", t + 1, c, "AP", round(cell["pixel_ap"], 4), "AUROC", round(cell["image_auroc"], 4),
                "rows", cell["rows"], "reused" if "reused_from" in cell else f"{cell['seconds']:.1f}s")
            D.cap_check(t_all, D.CAPS["eval_wall_h"])
        del data
    D.write_json(adir / "EVAL_DONE.json", {"arm": arm, "seconds_this_process": time.time() - t_all,
                                           "peak_rss_gb": D.peak_rss_gb(), "env": D.env_threads(), "utc": D.now()})


if __name__ == "__main__":
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    if sys.argv[1] == "check":
        do_check()
    elif sys.argv[1] == "arm":
        do_arm(sys.argv[2])
    else:
        raise SystemExit(sys.argv[1])
