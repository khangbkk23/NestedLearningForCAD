"""NL-SGM-01A-F2-DOM: in-domain, support-disjoint, per-category PQ codebook control (same-ID as C1/C4).

Owner-authorized 2026-11-11 after F1 (agents/reports/task_nl_sgm_01a_f1_codec_forensic.md). One control, CPU only.
Single change vs C4: category c's acquisition support rows (C1 identities) are encoded with a codebook fit on c's OWN
writer rows EXCLUDING every gid of c's C1 acquisition support (superset of all later C1 supports of c). Recipe = C0:
M=96, K=256, FP16, k-means++ seed 20261106 (+ subspace), 25 Lloyd iterations, hash-first <= 100k rows
(exps.csg01.select.random_priorities, lexsort by (priority, gid)). Scored with the AM03 path (exact.nn FP64, AM03
native_maps, image = max, exact native AP, sklearn AUROC). No test data in fitting; masks only for evaluation.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES= python -m exps.nl_sgm_01a_f1.f2 lock     # rule + exclusion lists, before any fit
  PYTHONPATH=. CUDA_VISIBLE_DEVICES= python -m exps.nl_sgm_01a_f1.f2 run      # 15 fits + 15 x {C1, F2} scoring
  PYTHONPATH=. CUDA_VISIBLE_DEVICES= python -m exps.nl_sgm_01a_f1.f2 decide   # mechanical verdict
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np
import torch

from exps.cadfinal.metrics import exact_ap
from exps.nl_sgm_01a import am02_run as R2
from exps.nl_sgm_01a import am03_eval as A3
from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a import exact as X
from exps.nl_sgm_01a_f1 import forensic as F1

ROOT = F1.ROOT
OUT = ROOT / "results/hope_cad/nl_sgm_01a_f2"
LOCK = OUT / "F2_LOCK.json"
ORDER = A3.ORDER
FIT_ROWS, FIT_SEED = R2.R.CodecArm.FIT_ROWS, C.SEED
SOURCES = ["exps/nl_sgm_01a_f1/f2.py", "exps/nl_sgm_01a_f1/forensic.py", "exps/nl_sgm_01a/codecs.py",
           "exps/nl_sgm_01a/exact.py", "exps/nl_sgm_01a/am03_eval.py", "exps/nl_sgm_01a/am02_run.py",
           "exps/nl_sgm_01a/common.py", "exps/csg01/select.py", "exps/cadfinal/metrics.py",
           "agents/tasks/task_nl_sgm_01a_f2_dom.md", "results/hope_cad/nl_sgm_01a_f1/F1_FORENSIC.json"]
RULE = {
    "primary": "R_AP = mean over the 14 non-bottle categories of (AP_F2 - AP_C4) / (AP_C1 - AP_C4), acquisition "
               "native Pixel AP; R_AUROC likewise for Image AUROC (AP_C1/AP_C4/AUROC from AM03 cells, verified "
               "against the recomputed C1)",
    "verdicts": {
        "DOMAIN_MISMATCH_PRIMARY": "R_AP >= 0.75 and R_AUROC >= 0.75",
        "DOMAIN_PARTIAL": "0.25 <= R_AP < 0.75 (or R_AP >= 0.75 with R_AUROC < 0.75)",
        "DOMAIN_NOT_PRIMARY": "R_AP < 0.25",
    },
    "flags": {
        "MEMORIZATION": "bottle AP_F2 < AP_C4(bottle) - 0.05 (bottle's C4 immunity partly in-sample fit)",
        "DISTORTION_RANKING_DISSOCIATION": "median non-bottle NMSE reduction (C4 -> F2) >= 0.25 absolute while R_AP < 0.25 "
                                           "(lower reconstruction error without ranking recovery)",
    },
    "lessons_by_outcome": {
        "DOMAIN_MISMATCH_PRIMARY": "96 B/row PQ preserves anomaly ranking when the codebook matches the category "
                                   "distribution; the AM02/AM03 loss is a codebook-scope artifact, not a PQ rate limit. "
                                   "Next: a byte-accounted continual codebook design question (per-category/append-only "
                                   "codebooks cost 393,216 B each = ~4,000 PQ rows), not consolidation or NL.",
        "DOMAIN_PARTIAL": "Domain explains part of the loss; residual is rate (96 B/row) and/or max-patch sensitivity. "
                          "Next: one rate control (higher M) before any memory design; PQ alone is not yet a viable "
                          "support store.",
        "DOMAIN_NOT_PRIMARY": "Even an in-domain codebook at 96 B/row destroys anomaly ranking for out-of-sample rows; "
                              "the bottle result was not evidence of PQ viability. Next: close PQ at this rate for "
                              "NN-distance scoring; retention studies must use high-fidelity storage (FP16/U8).",
        "MEMORIZATION": "C0's bottle immunity came from fitting the very rows stored; in-sample codec results must not "
                        "be used as evidence of codec quality.",
        "DISTORTION_RANKING_DISSOCIATION": "Reconstruction MSE is not a valid proxy for anomaly-ranking damage; future "
                                           "codec selection must be judged on rank metrics.",
    },
    "secondary_descriptive": ["per-category recovery", "NMSE / row error / err-over-normal-NN for F2",
                              "normal-patch d/d1 ratio", "argmax-in-mask", "TPR@1%FPR", "pixel pair inversions vs C1",
                              "Spearman(recovery, NMSE reduction) over 14"],
    "not_claimed": "no memory-budget, retention or deployment claim; 15 codebooks add 5,898,240 B and are not byte-matched",
}


def sha(p):
    return F1.sha(p)


def req(c, m):
    if not c:
        raise RuntimeError(m)


def gid_sha(t):
    return hashlib.sha256(np.ascontiguousarray(t.numpy().astype(np.int64)).tobytes()).hexdigest()


def exclusions(n):
    """{c: sorted int64 gids of c's C1 support at its acquisition boundary}; checks later supports are subsets."""
    ex = {}
    for t, c in enumerate(ORDER):
        ck = torch.load(R2.OUT / f"stages/C1_FPS_FP32/ckpt_t{t:02d}.pt", weights_only=False)
        m = len(ck["gids"])
        ex[c] = torch.sort(ck["gids"][ck["owner"][:m].long() == t]).values
    for t in range(15):
        ck = torch.load(R2.OUT / f"stages/C1_FPS_FP32/ckpt_t{t:02d}.pt", weights_only=False)
        m = len(ck["gids"])
        for a in range(t + 1):
            later = ck["gids"][ck["owner"][:m].long() == a]
            req(bool(torch.isin(later, ex[ORDER[a]]).all()), f"C1 support of {ORDER[a]} at t={t} not in acquisition set")
    return ex


def fit_sample_gids(n, c, excl):
    g = R2.writer_gids(n, c)
    p = R2.random_priorities(FIT_SEED, g.numpy().astype(np.int64))
    order = np.lexsort((g.numpy(), p))
    keep = ~torch.isin(g[torch.as_tensor(order)], excl)
    sel = torch.as_tensor(order)[keep][:FIT_ROWS]
    return sel                                                     # indices into the writer-row matrix


def lock():
    req(not LOCK.exists(), "already locked")
    OUT.mkdir(parents=True, exist_ok=True)
    lk, co, n = R2.setup()
    ex = exclusions(n)
    cats = {}
    for c in ORDER:
        sel = fit_sample_gids(n, c, ex[c])
        g = R2.writer_gids(n, c)[sel]
        req(not bool(torch.isin(g, ex[c]).any()), f"fit sample overlaps support {c}")
        cats[c] = {"excluded_support_rows": len(ex[c]), "excluded_gids_sha256": gid_sha(ex[c]),
                   "writer_rows": int(n[c] * C.PATCHES), "fit_rows": len(sel), "fit_gids_sha256": gid_sha(torch.sort(g).values)}
    rec = {"schema": "nl_sgm_01a_f2_lock_v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rule": RULE, "recipe": {"M": K.M, "K": K.K, "codebook_dtype": "float16", "seed": FIT_SEED, "iters": 25,
                                     "fit_rows_cap": FIT_ROWS, "sample": "hash-first (random_priorities, gid) after "
                                     "excluding the category's C1 acquisition support gids"},
           "categories": cats, "sources": {s: sha(ROOT / s) for s in SOURCES},
           "am03_lock_sha256": sha(F1.AM03 / "AM03_LOCK.json"), "am02_lock_sha256": sha(F1.AM02 / "LOCK_r2.json")}
    with open(LOCK, "x") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    print("locked", sha(LOCK), {c: (v["fit_rows"], v["excluded_support_rows"]) for c, v in cats.items()}, flush=True)


def verify():
    lk = json.loads(LOCK.read_text())
    bad = [s for s, d in lk["sources"].items() if sha(ROOT / s) != d]
    req(not bad and lk["rule"] == RULE, f"post-lock change: {bad}")
    return lk


def score(c, t, sup, man, d1=None):
    feats = A3.load_features(c)
    labels, masks, _ = A3.load_masks(c)
    hw = tuple(man["categories"][c]["images"][0]["hw"])
    d, i = X.nn(feats.reshape(-1, 768), sup)
    m = A3.native_maps(d.reshape(-1, 784), hw)
    ap = exact_ap(list(m), masks)["ap"]
    img = m.reshape(len(m), -1).max(1)
    from sklearn.metrics import roc_auc_score
    return {"d": d.numpy(), "nn": i, "maps": m, "img": img, "ap": ap, "auroc": float(roc_auc_score(labels, img)),
            "labels": labels, "masks": masks}


def diagnostics(r1, r2):
    """F1 diagnostics of arm r2 against C1 (r1)."""
    masks, labels = r1["masks"], r1["labels"]
    y = np.concatenate([mk.reshape(-1) for mk in masks])
    s1, s2 = r1["maps"].reshape(-1), r2["maps"].reshape(-1)
    n1, n2 = np.sort(s1[~y]), np.sort(s2[~y])
    tpr = lambda s, n: float((s[y] > n[int(.99 * len(n))]).mean())
    rng = np.random.default_rng(F1.SEED)
    ai, ni = np.flatnonzero(y), np.flatnonzero(~y)
    pa, pn = ai[rng.integers(0, len(ai), F1.PAIRS)], ni[rng.integers(0, len(ni), F1.PAIRS)]
    c1, c2 = s1[pa] > s1[pn], s2[pa] > s2[pn]
    cov = F1.patch_cov(masks).numpy()
    D1, D2 = r1["d"], r2["d"]
    nrm, ano = cov == 0, cov >= 0.5
    d = labels == 1
    inm = lambda mp: float(np.mean([masks[k].reshape(-1)[int(mp[k].argmax())] for k in np.flatnonzero(d)]))
    from scipy.stats import spearmanr
    out = {"tpr_at_fpr1pct": tpr(s2, n2), "tpr_at_fpr1pct_C1": tpr(s1, n1),
           "pairs_lost_vs_C1": float((c1 & ~c2).mean()), "pairs_gained_vs_C1": float((~c1 & c2).mean()),
           "normal_ratio_median": float(np.median(D2[nrm] / D1[nrm])), "anom_ratio_median": float(np.median(D2[ano] / D1[ano])),
           "spearman_d1_all": float(spearmanr(D1, D2)[0]), "argmax_in_mask": inm(r2["maps"]), "argmax_in_mask_C1": inm(r1["maps"]),
           "contrast_anomq50_over_normq50": float(np.median(s2[y]) / np.median(n2)),
           "contrast_C1": float(np.median(s1[y]) / np.median(n1)),
           "nn_identity_agreement_C1": float((r1["nn"] == r2["nn"]).float().mean())}
    return out


def run():
    verify()
    target = OUT / "F2_RESULTS.json"
    req(not target.exists(), "one-shot: F2_RESULTS.json exists")
    t_all = time.time()
    lk, co, n = R2.setup()                                        # AM02 lock verified, CPU only, 6 threads
    lock_rec = json.loads(LOCK.read_text())
    ex = exclusions(n)
    man = json.loads((F1.AM03 / "DATASET_MANIFEST.json").read_text())
    f1 = json.loads((ROOT / "results/hope_cad/nl_sgm_01a_f1/F1_FORENSIC.json").read_text())["categories"]
    p99, _ = F1.bottle_p99(co, n, torch.load(F1.AM02 / "stages/c4/C0.pt", weights_only=False)["codebook"])
    (OUT / "codebooks").mkdir(exist_ok=True)
    F_ = C.Features()
    res = {"schema": "nl_sgm_01a_f2_results_v1", "lock_sha256": sha(LOCK), "categories": {}}
    for t, c in enumerate(ORDER):
        rec = lock_rec["categories"][c]
        req(gid_sha(ex[c]) == rec["excluded_gids_sha256"], f"exclusion set changed {c}")
        W = F_.images(co[c]["writers"]).reshape(-1, 768)
        sel = fit_sample_gids(n, c, ex[c])
        req(gid_sha(torch.sort(R2.writer_gids(n, c)[sel]).values) == rec["fit_gids_sha256"], f"fit sample changed {c}")
        sample = W[sel].clone().contiguous()
        F_.drop(c); del W
        t0 = time.time()
        cb = K.fit(sample, seed=FIT_SEED)
        fit_sec = time.time() - t0
        del sample
        cbp = OUT / "codebooks" / f"{c}.pt"
        req(not cbp.exists(), f"exists {cbp}")
        torch.save({"codebook": cb, "category": c}, cbp)
        s1, _ = A3.load_support("C1_FPS_FP32", t)
        sup1 = s1[c]
        req(torch.equal(sup1.ids, ex[c]), f"acquisition support ids {c}")
        x2 = K.decode(K.encode(sup1.x, cb), cb)
        sup2 = X.RowSet(x2.contiguous(), sup1.ids)
        r1 = score(c, t, sup1, man)
        req(abs(r1["ap"] - f1[c]["pixel_ap_C1"]) < 1e-9 and abs(r1["auroc"] - f1[c]["image"]["auroc_C1"]) < 1e-9,
            f"C1 recomputation differs from F1 {c}")
        r2 = score(c, t, sup2, man)
        dist = F1.distortion(sup1.x, x2, p99, set(), sup1.ids)
        dist["rel_err_over_normal_nn_scale"] = dist["row_err_median"] / f1[c]["patch"]["d1_median_normal"]
        ap1, ap4 = f1[c]["pixel_ap_C1"], f1[c]["pixel_ap_C4"]
        au1, au4 = f1[c]["image"]["auroc_C1"], f1[c]["image"]["auroc_C4"]
        out = {"stage": t + 1, "fit_seconds": fit_sec, "fit_rows": rec["fit_rows"],
               "codebook_sha256": sha(cbp), "pixel_ap": {"C1": ap1, "C4": ap4, "F2": r2["ap"]},
               "image_auroc": {"C1": au1, "C4": au4, "F2": r2["auroc"]},
               "recovery_ap": (r2["ap"] - ap4) / (ap1 - ap4) if abs(ap1 - ap4) > 1e-9 else None,
               "recovery_auroc": (r2["auroc"] - au4) / (au1 - au4) if abs(au1 - au4) > 1e-9 else None,
               "distortion_F2": dist, "distortion_C4": {k: f1[c]["distortion"][k] for k in
                                                         ("row_err_median", "nmse_total", "rel_err_over_normal_nn_scale",
                                                          "frac_resid_gt_bottle_p99")},
               "diagnostics_F2": diagnostics(r1, r2)}
        res["categories"][c] = out
        print(c, "AP C1/C4/F2", round(ap1, 4), round(ap4, 4), round(r2["ap"], 4), "AUROC", round(au1, 3), round(au4, 3),
              round(r2["auroc"], 3), "NMSE C4/F2", round(f1[c]["distortion"]["nmse_total"], 3), round(dist["nmse_total"], 3),
              "fit", round(fit_sec), "s rss", round(F1.rss_gb(), 2), flush=True)
        del r1, r2, x2, sup2
    res["resources"] = {"seconds": time.time() - t_all, "peak_rss_gb": F1.rss_gb(), "device": "cpu",
                        "threads": torch.get_num_threads(), "cuda_initialized": torch.cuda.is_initialized(),
                        "extra_codebook_bytes": 15 * K.CODEBOOK_BYTES}
    with open(target, "x") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("written", target, sha(target), flush=True)


def decide():
    verify()
    target = OUT / "F2_VERDICT.json"
    req(not target.exists(), "one-shot")
    r = json.loads((OUT / "F2_RESULTS.json").read_text())
    cs = r["categories"]
    nb = [c for c in ORDER if c != "bottle"]
    R_ap = float(np.mean([cs[c]["recovery_ap"] for c in nb]))
    R_au = float(np.mean([cs[c]["recovery_auroc"] for c in nb]))
    if R_ap >= 0.75 and R_au >= 0.75:
        v = "DOMAIN_MISMATCH_PRIMARY"
    elif R_ap >= 0.25:
        v = "DOMAIN_PARTIAL"
    else:
        v = "DOMAIN_NOT_PRIMARY"
    b = cs["bottle"]
    flags = []
    if b["pixel_ap"]["F2"] < b["pixel_ap"]["C4"] - 0.05:
        flags.append("MEMORIZATION")
    nmse_red = float(np.median([cs[c]["distortion_C4"]["nmse_total"] - cs[c]["distortion_F2"]["nmse_total"] for c in nb]))
    if nmse_red >= 0.25 and R_ap < 0.25:
        flags.append("DISTORTION_RANKING_DISSOCIATION")
    from scipy.stats import spearmanr
    rec = [cs[c]["recovery_ap"] for c in nb]
    red = [cs[c]["distortion_C4"]["nmse_total"] - cs[c]["distortion_F2"]["nmse_total"] for c in nb]
    out = {"verdict": v, "flags": flags, "R_AP": R_ap, "R_AUROC": R_au,
           "R_AP_median": float(np.median(rec)), "categories_recovering_ge_0_75": int(sum(x >= .75 for x in rec)),
           "macro_ap": {a: float(np.mean([cs[c]["pixel_ap"][a] for c in ORDER])) for a in ("C1", "C4", "F2")},
           "macro_auroc": {a: float(np.mean([cs[c]["image_auroc"][a] for c in ORDER])) for a in ("C1", "C4", "F2")},
           "median_nmse_reduction_nonbottle": nmse_red,
           "spearman_recovery_vs_nmse_reduction": float(spearmanr(rec, red)[0]),
           "bottle": {"C4": b["pixel_ap"]["C4"], "F2": b["pixel_ap"]["F2"], "C1": b["pixel_ap"]["C1"]},
           "lessons": [RULE["lessons_by_outcome"][v]] + [RULE["lessons_by_outcome"][f] for f in flags],
           "results_sha256": sha(OUT / "F2_RESULTS.json"), "lock_sha256": sha(LOCK)}
    with open(target, "x") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    {"lock": lock, "run": run, "decide": decide}[sys.argv[1]]()
