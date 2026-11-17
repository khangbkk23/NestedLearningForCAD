"""NL-SGM-01A-F1: causal score-distortion forensic of the AM03 C1 (FP32) vs C4 (same-ID PQ, frozen C0) acquisition gap.

Analysis only (task agents/tasks/task_nl_sgm_01a_f1_codec_forensic.md). Reads frozen AM02 checkpoints, the frozen C0
codebook, the CALM-02 ViT-B/8 caches and the AM03 dataset manifest; recomputes ONLY the 30 matched acquisition cells
(15 categories x {C1, C4}) with the AM03 scorer path (exact FP64 1-NN via exps.nl_sgm_01a.exact.nn, AM03 native_maps,
image score = max). CPU only, 6 threads, TF32 off, fixed seeds. Writes results/hope_cad/nl_sgm_01a_f1/ (create-exclusive).

  PYTHONPATH=. python -m exps.nl_sgm_01a_f1.forensic
"""
from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from exps.cadfinal.metrics import exact_ap
from exps.nl_sgm_01a import am02_run as R2
from exps.nl_sgm_01a import am03_eval as A3
from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a import exact as X

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hope_cad/nl_sgm_01a_f1"
AM02 = ROOT / "results/hope_cad/nl_sgm_01a_am02"
AM03 = ROOT / "results/hope_cad/nl_sgm_01a_am03"
ORDER = A3.ORDER
PAIRS, SEED = 2_000_000, 20261109
EXPECT = {"AM02_LOCK_r2": "66c92f5ae2e5c20f07ce99d2c47c1336a434bab77e6153d9d9a78598f7e500bb",
          "AM03_LOCK": "21f9809242e2169fa6615a1d2c3e7ee282377e742358ccb72f46f93b8350f481",
          "am03_eval.py": "28346641191255f0e866947f4c31e8525c0c396d56ed822c61c0eb65c9da7176"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def req(c, m):
    if not c:
        raise RuntimeError(m)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


# --- 1. integrity --------------------------------------------------------------------------------------------------


def integrity(co, n, C0):
    out = {"hashes": {"AM02_LOCK_r2": sha(AM02 / "LOCK_r2.json"), "AM03_LOCK": sha(AM03 / "AM03_LOCK.json"),
                      "am03_eval.py": sha(ROOT / "exps/nl_sgm_01a/am03_eval.py"),
                      "C0.pt": sha(AM02 / "stages/c4/C0.pt")}, "boundaries": {}}
    for k, v in EXPECT.items():
        req(out["hashes"][k] == v, f"hash changed: {k}")
    A3.lock()
    F_ = C.Features()
    for t in range(15):
        c1 = torch.load(AM02 / f"stages/C1_FPS_FP32/ckpt_t{t:02d}.pt", weights_only=False)
        c4 = torch.load(AM02 / f"stages/c4/ckpt_t{t:02d}.pt", weights_only=False)
        m = len(c1["gids"])
        rec = {"rows": m, "gids_equal": bool(torch.equal(c1["gids"], c4["gids"])),
               "owner_equal": bool(torch.equal(c1["owner"][:m].long(), c4["owner"].long())),
               "codes_equal_encode_C1_rows": bool(torch.equal(K.encode(c1["x"][:m], C0), c4["codes"]))}
        # C1 rows of the acquired category equal the CALM-02 train feature rows of their gids (bitwise)
        c = ORDER[t]
        sel = c1["owner"][:m].long() == t
        g = c1["gids"][sel] - R2.gid_base(n, c)
        req(bool((g >= 0).all()) and bool((g < n[c] * C.PATCHES).all()), f"gid outside writer range {c}")
        W = F_.images(co[c]["writers"]).reshape(-1, 768)
        rec["c1_rows_equal_cached_train_features"] = bool(torch.equal(W[g], c1["x"][:m][sel]))
        F_.drop(c)
        out["boundaries"][t] = rec
        req(all(v for k, v in rec.items() if k != "rows"), f"integrity failure at t={t}: {rec}")
    return out


# --- 2. codec distortion ---------------------------------------------------------------------------------------------


def bottle_p99(co, n, C0):
    F_ = C.Features()
    c = ORDER[0]
    x = F_.images(co[c]["writers"]).reshape(-1, 768).clone()
    sample = R2.c0_sample(x, R2.writer_gids(n, c)); del x
    nfit = len(sample)
    r = (sample - K.decode(K.encode(sample, C0), C0)).reshape(len(sample), K.M, K.SUB).square().sum(-1)
    p99 = torch.quantile(r[::4].double(), 0.99, dim=0); del sample, r; F_.drop(c)
    fit_ids = set(R2.writer_gids(n, c)[torch.as_tensor(np.lexsort(
        (R2.writer_gids(n, c).numpy(), R2.random_priorities(C.SEED, R2.writer_gids(n, c).numpy().astype(np.int64)))
    )[:nfit])].tolist())
    return p99, fit_ids


def distortion(x1, x4, p99, fit_ids, gids):
    r = (x1 - x4).reshape(len(x1), K.M, K.SUB).square().sum(-1).double()        # [rows, M] squared residual
    var = x1.reshape(len(x1), K.M, K.SUB).double().var(0).sum(-1)              # per-subspace variance of the rows
    e = r.sum(1).sqrt()
    return {"rows": len(x1), "row_err_median": float(e.median()), "row_err_q90": float(e.quantile(.9)),
            "row_norm_median": float(x1.double().norm(dim=1).median()),
            "rel_row_err_median": float((e / x1.double().norm(dim=1)).median()),
            "subspace_mse": r.mean(0).tolist(), "subspace_nmse": (r.mean(0) / var).tolist(),
            "nmse_total": float(r.mean(0).sum() / var.sum()),
            "frac_resid_gt_bottle_p99": float((r > p99[None]).double().mean()),
            "frac_rows_in_C0_fit_sample": float(np.mean([int(g) in fit_ids for g in gids.tolist()]))}


# --- 3/4. matched score analysis -----------------------------------------------------------------------------------


def patch_cov(masks):
    return torch.stack([F.adaptive_avg_pool2d(torch.from_numpy(m.astype(np.float32))[None, None], 28)[0, 0]
                        for m in masks]).reshape(-1)


def ranks(norm_sorted, x):
    return np.searchsorted(norm_sorted, x, side="left") / len(norm_sorted)


def side_stats(s, y, good_pix, norm):
    anom = s[y]
    q = lambda a, p: float(np.quantile(a, p))
    u = ranks(norm, anom)
    return {"normal_q50": q(norm, .5), "normal_q90": q(norm, .9), "normal_q99": q(norm, .99), "normal_q999": q(norm, .999),
            "good_img_normal_q99": q(s[good_pix], .99), "defect_img_background_q99": q(s[(~y) & (~good_pix)], .99),
            "anom_q10": q(anom, .1), "anom_q50": q(anom, .5), "anom_q90": q(anom, .9),
            "pixel_auroc_approx": float(u.mean()), "tpr_at_fpr1pct": float((anom > norm[int(.99 * len(norm))]).mean()),
            "tpr_at_fpr01pct": float((anom > norm[int(.999 * len(norm))]).mean()),
            "fpr_at_anom_median": float(1 - ranks(norm, np.array([np.median(anom)]))[0]),
            "contrast_anomq50_over_normq50": q(anom, .5) / q(norm, .5),
            "normal_tail_q99_over_q50": q(norm, .99) / q(norm, .5),
            "anom_exceedance_q10_q50": [float(np.quantile(u, .1)), float(np.quantile(u, .5))]}


def analyse_category(t, c, sup1, sup4, e_row, man):
    feats = A3.load_features(c)
    labels, masks, items = A3.load_masks(c)
    hw = tuple(man["categories"][c]["images"][0]["hw"])
    q = feats.reshape(-1, 768)
    t0 = time.time()
    d1, i1 = X.nn(q, sup1)
    d4, i4 = X.nn(q, sup4)
    nn_sec = time.time() - t0
    req(bool(torch.isfinite(d1).all()) and bool(torch.isfinite(d4).all()), "nonfinite distance")
    # native maps exactly as AM03 (float32), image score = max
    m1 = A3.native_maps(d1.reshape(-1, 784), hw)
    m4 = A3.native_maps(d4.reshape(-1, 784), hw)
    ap1, ap4 = exact_ap(list(m1), masks)["ap"], exact_ap(list(m4), masks)["ap"]
    img1, img4 = m1.reshape(len(m1), -1).max(1), m4.reshape(len(m4), -1).max(1)
    from sklearn.metrics import roc_auc_score
    au1, au4 = float(roc_auc_score(labels, img1)), float(roc_auc_score(labels, img4))
    # pixel arrays
    y = np.concatenate([mk.reshape(-1) for mk in masks])
    good_pix = np.concatenate([np.full(mk.size, lab == 0) for mk, lab in zip(masks, labels)])
    s1, s4 = m1.reshape(-1), m4.reshape(-1)
    n1s = np.sort(s1[~y]); n4s = np.sort(s4[~y])
    st1, st4 = side_stats(s1, y, good_pix, n1s), side_stats(s4, y, good_pix, n4s)
    # rank inversions on identical random (anomaly, normal) pixel pairs
    rng = np.random.default_rng(SEED + t)
    ai, ni = np.flatnonzero(y), np.flatnonzero(~y)
    pa, pn = ai[rng.integers(0, len(ai), PAIRS)], ni[rng.integers(0, len(ni), PAIRS)]
    c1, c4 = s1[pa] > s1[pn], s4[pa] > s4[pn]
    inv = {"correct_C1": float(c1.mean()), "correct_C4": float(c4.mean()), "lost": float((c1 & ~c4).mean()),
           "gained": float((~c1 & c4).mean())}
    # per-pair attribution of lost pairs: which side moved more in rank (normal-pixel rank under each arm)
    lost = c1 & ~c4
    if lost.any():
        ra1, ra4 = ranks(n1s, s1[pa[lost]]), ranks(n4s, s4[pa[lost]])        # anomaly pixel normal-CDF rank
        rn1, rn4 = ranks(n1s, s1[pn[lost]]), ranks(n4s, s4[pn[lost]])        # normal pixel normal-CDF rank
        inv["lost_pairs_anom_rank_drop_mean"] = float((ra1 - ra4).mean())
        inv["lost_pairs_normal_rank_rise_mean"] = float((rn4 - rn1).mean())
        inv["lost_pairs_dominated_by_anomaly_suppression"] = float(((ra1 - ra4) > (rn4 - rn1)).mean())
    del c1, c4, pa, pn, good_pix
    # patch-level: decomposition of the distance change
    cov = patch_cov(masks).numpy()
    D1, D4 = d1.numpy(), d4.numpy()
    nrm, ano = cov == 0, cov >= 0.5
    eq = (i1 == i4).numpy()
    e_at_nn4 = e_row[torch.searchsorted(sup4.ids, i4)].numpy()
    e_at_nn1 = e_row[torch.searchsorted(sup1.ids, i1)].numpy()
    pred = np.sqrt(D1 ** 2 + e_at_nn1 ** 2)
    from scipy.stats import spearmanr, pearsonr
    pr = lambda a, b: [float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0])]
    per_img_rho = [float(spearmanr(D1[k * 784:(k + 1) * 784], D4[k * 784:(k + 1) * 784])[0]) for k in range(len(labels))]
    patch = {"n_normal": int(nrm.sum()), "n_anom": int(ano.sum()),
             "corr_d1_d4_all": pr(D1, D4), "corr_d1_d4_normal": pr(D1[nrm], D4[nrm]),
             "corr_d1_d4_anom": pr(D1[ano], D4[ano]) if ano.sum() > 2 else None,
             "per_image_spearman_mean_good": float(np.nanmean([r for r, l in zip(per_img_rho, labels) if l == 0])),
             "per_image_spearman_mean_defect": float(np.nanmean([r for r, l in zip(per_img_rho, labels) if l == 1])),
             "nn_identity_agreement": float(eq.mean()),
             "d1_median_normal": float(np.median(D1[nrm])), "d4_median_normal": float(np.median(D4[nrm])),
             "d1_median_anom": float(np.median(D1[ano])), "d4_median_anom": float(np.median(D4[ano])),
             "delta_median_normal": float(np.median(D4[nrm] - D1[nrm])),
             "delta_median_anom": float(np.median(D4[ano] - D1[ano])),
             "ratio_median_normal": float(np.median(D4[nrm] / D1[nrm])),
             "ratio_median_anom": float(np.median(D4[ano] / D1[ano])),
             "codec_err_at_C4_nn_median_normal": float(np.median(e_at_nn4[nrm])),
             "codec_err_over_d1_normal_median": float(np.median(e_at_nn1[nrm] / D1[nrm])),
             "noise_floor_model_corr_normal": pr(pred[nrm], D4[nrm]),
             "noise_floor_model_median_abs_rel_err_normal": float(np.median(np.abs(pred[nrm] - D4[nrm]) / D4[nrm])),
             "d4_below_d1_frac": float((D4 < D1).mean())}
    # image level
    g, d = labels == 0, labels == 1
    lost_img = sum(1 for a in np.flatnonzero(d) for b in np.flatnonzero(g) if img1[a] > img1[b] and not img4[a] > img4[b])
    gain_img = sum(1 for a in np.flatnonzero(d) for b in np.flatnonzero(g) if not img1[a] > img1[b] and img4[a] > img4[b])
    npair = int(d.sum() * g.sum())
    argmax_in_mask = lambda m: float(np.mean([masks[k].reshape(-1)[int(m[k].argmax())] for k in np.flatnonzero(d)]))
    image = {"auroc_C1": au1, "auroc_C4": au4, "pairs": npair, "lost_frac": lost_img / npair, "gained_frac": gain_img / npair,
             "good_max_ratio_median": float(np.median(img4[g] / img1[g])),
             "defect_max_ratio_median": float(np.median(img4[d] / img1[d])),
             "argmax_in_mask_C1": argmax_in_mask(m1), "argmax_in_mask_C4": argmax_in_mask(m4),
             "good_max_rank_in_C4_normal_pixels_median": float(np.median(ranks(n4s, img4[g]))),
             "defect_max_exceeds_all_good_max_C1": float(np.mean(img1[d] > img1[g].max())),
             "defect_max_exceeds_all_good_max_C4": float(np.mean(img4[d] > img4[g].max()))}
    del n1s, n4s, s1, s4, y, m1, m4
    import gc; gc.collect()
    am03 = {a: json.loads((AM03 / f"cells/{a}/stage{t + 1:02d}_{c}.json").read_text())
            for a in ("C1_FPS_FP32", "C4_SAMEID_PQ")}
    repro = {"pixel_ap_abs_diff_C1": abs(ap1 - am03["C1_FPS_FP32"]["pixel_ap"]),
             "pixel_ap_abs_diff_C4": abs(ap4 - am03["C4_SAMEID_PQ"]["pixel_ap"]),
             "image_auroc_abs_diff_C1": abs(au1 - am03["C1_FPS_FP32"]["image_auroc"]),
             "image_auroc_abs_diff_C4": abs(au4 - am03["C4_SAMEID_PQ"]["image_auroc"])}
    req(max(repro.values()) < 1e-6, f"AM03 reproduction failed {c}: {repro}")
    return {"stage": t + 1, "category": c, "n_images": len(labels), "n_anomalous": int(labels.sum()), "hw": hw,
            "pixel_ap_C1": ap1, "pixel_ap_C4": ap4, "delta_ap": ap1 - ap4, "rel_delta_ap": (ap1 - ap4) / ap1,
            "pixels_C1": st1, "pixels_C4": st4, "pixel_pair_inversions": inv, "patch": patch, "image": image,
            "am03_reproduction": repro, "nn_seconds": nn_sec}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "F1_FORENSIC.json"
    req(not target.exists(), "one-shot: F1_FORENSIC.json exists")
    t_all = time.time()
    lock, co, n = R2.setup()                                     # verifies AM02 LOCK_r2, CPU only, 6 threads
    C0 = torch.load(AM02 / "stages/c4/C0.pt", weights_only=False)["codebook"]
    res = {"schema": "nl_sgm_01a_f1_forensic_v1", "seed": SEED, "pairs": PAIRS, "threads": torch.get_num_threads(),
           "device": "cpu", "torch": torch.__version__}
    res["integrity"] = integrity(co, n, C0)
    print("integrity OK", flush=True)
    p99, fit_ids = bottle_p99(co, n, C0)
    man = json.loads((AM03 / "DATASET_MANIFEST.json").read_text())
    res["categories"] = {}
    for t, c in enumerate(ORDER):
        s1, _ = A3.load_support("C1_FPS_FP32", t)
        s4, _ = A3.load_support("C4_SAMEID_PQ", t)
        sup1, sup4 = s1[c], s4[c]
        req(torch.equal(sup1.ids, sup4.ids), f"support ids differ {c}")
        e_row = (sup1.x.double() - sup4.x.double()).norm(dim=1)
        rec = analyse_category(t, c, sup1, sup4, e_row, man)
        rec["distortion"] = distortion(sup1.x, sup4.x, p99, fit_ids, sup1.ids)
        rec["distortion"]["rel_err_over_normal_nn_scale"] = rec["distortion"]["row_err_median"] / rec["patch"]["d1_median_normal"]
        res["categories"][c] = rec
        print(c, "AP C1/C4", round(rec["pixel_ap_C1"], 4), round(rec["pixel_ap_C4"], 4), "rel err",
              round(rec["distortion"]["rel_err_over_normal_nn_scale"], 3), "nmse", round(rec["distortion"]["nmse_total"], 3),
              "inv lost", round(rec["pixel_pair_inversions"]["lost"], 4), "rss", round(rss_gb(), 2), flush=True)
    res["resources"] = {"seconds": time.time() - t_all, "peak_rss_gb": rss_gb(), "gpu": "not initialized",
                        "cuda_initialized": torch.cuda.is_initialized()}
    with open(target, "x") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("written", target, sha(target), flush=True)


if __name__ == "__main__":
    main()
