"""VITB8-NL-00 Phase 1 forensics (read-only on existing artifacts; label-free recomputation from existing checkpoints).

  python -m exps.vitb8_nl00.forensics

Writes results/hope_cad/vitb8_nl00/forensics/FORENSICS.json. No training, no selection. Test labels/masks are read
only to describe already-consumed evaluation outcomes.
  A. Q2 effective dose per stage (LOG files): rho*||g_A||/(||g_A||+eps), cos(g_N,g_A), saturation.
  B. Retention decomposition from EVAL_STAGE*.json: signed/positive/peak-based forgetting, backward improvement.
  C. Screw and capsule: decoder AUROC/AP at the checkpoints around acquisition and at the end, normal vs anomalous
     image-score distributions, argmax location of good-image maps, and a raw-feature 1-NN representation ceiling
     (frozen ViT-B/8 layer-9 train features of the category, same 28x28 -> 256 -> Gaussian scorer chain).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from exps.cadfinal import metrics as M
from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly.common import require
from exps.vitb8_q2 import common as V
from exps.vitb8_q2 import evaluate as QE

OUT = V.ROOT / "results/hope_cad/vitb8_nl00/forensics"
ON = V.OUT


def dose():
    out = {}
    for s in range(1, 16):
        L = torch.load(ON / f"runs/seed{V.SEED}/LOG{s:02d}.pt", weights_only=False)
        a = {k: np.asarray(v, np.float64) for k, v in L.items()}
        d = a["scale"] * a["nA"] / a["nN"]
        sat = a["aux_loss"] < 1e-3
        n = len(d)
        out[s] = {"dose_mean": float(np.mean(d)), "dose_median": float(np.median(d)),
                  "dose_first_third": float(np.mean(d[:n // 3])), "dose_last_third": float(np.mean(d[-n // 3:])),
                  "frac_dose_below_half_nominal": float(np.mean(d < 0.125)), "saturated_frac": float(np.mean(sat)),
                  "nA_median": float(np.median(a["nA"])), "cos_mean_saturated": float(np.mean(a["cos"][sat])),
                  "cos_mean_unsaturated": float(np.mean(a["cos"][~sat])) if (~sat).any() else None,
                  "clip_active_frac": float(np.mean(a["norm_pre_clip"] > 0.1))}
    return out


def retention():
    ev = {s: D.read(ON / f"EVAL_STAGE{s:02d}.json")["per_category"] for s in range(1, 16)}
    out = {}
    for j, c in enumerate(V.STREAM[:-1]):
        tr = [ev[s][c]["pixel_ap"] for s in range(j + 1, 16)]
        tra = [ev[s][c]["image_auroc"] for s in range(j + 1, 16)]
        out[c] = {"ap_trajectory": tr, "auroc_trajectory": tra, "acq": tr[0], "final": tr[-1], "peak": max(tr),
                  "min": min(tr), "signed_F": tr[0] - tr[-1], "peak_F": max(tr) - tr[-1],
                  "max_drawdown_from_acq": tr[0] - min(tr),
                  "good_mean": [ev[s][c]["good_image_score_mean"] for s in range(j + 1, 16)],
                  "good_q99": [ev[s][c]["good_image_score_q99"] for s in range(j + 1, 16)],
                  "anom_mean": [ev[s][c]["anomalous_image_score_mean"] for s in range(j + 1, 16)]}
    F_ = np.array([v["signed_F"] for v in out.values()])
    summary = {"signed_macro": float(F_.mean()), "positive_macro": float(np.maximum(F_, 0).mean()),
               "backward_improvement_macro": float(np.maximum(-F_, 0).mean()),
               "n_forgot": int((F_ > 0.01).sum()), "n_improved": int((F_ < -0.01).sum()),
               "peak_based_macro": float(np.mean([v["peak_F"] for v in out.values()])),
               "max_drawdown_macro": float(np.mean([v["max_drawdown_from_acq"] for v in out.values()]))}
    return out, summary


def _chain(grid28):
    """Same map chain as exps.vitb8_q2.common.score, from a [B,28,28] distance grid."""
    a = F.interpolate(grid28[:, None].float(), size=V.IMG, mode="bilinear", align_corners=True)
    a = F.interpolate(a, size=V.EVAL_SIZE, mode="bilinear", align_corners=False)
    s = V.gauss(a.device)(a)
    flat = s.flatten(1)
    img = torch.sort(flat, dim=1, descending=True)[0][:, :int(flat.shape[1] * V.TOPK_RATIO)].mean(1)
    return s[:, 0], img


@torch.no_grad()
def knn_scores(c, chunk=4096):
    bank = V.train_features(c).reshape(-1, V.DIM).cuda()             # [n*784, 768] FP32
    bn = (bank * bank).sum(1)
    f, _ = V.test_features(c)
    maps, imgs = [], []
    for x in f.split(2):
        q = x.reshape(-1, V.DIM).cuda()
        qn = (q * q).sum(1, keepdim=True)
        best = torch.full((len(q),), float("inf"), device="cuda")
        for i in range(0, len(bank), chunk):
            b = bank[i:i + chunk]
            d = qn + bn[i:i + chunk][None] - 2 * q @ b.T
            best = torch.minimum(best, d.min(1).values)
        g = best.clamp_min(0).sqrt().reshape(len(x), V.SIDE, V.SIDE)
        s, im = _chain(g)
        maps.append(s.cpu()); imgs.append(im.cpu())
    del bank
    torch.cuda.empty_cache()
    return torch.cat(maps), torch.cat(imgs)


@torch.no_grad()
def decoder_scores(stage, c):
    rec = D.read(ON / f"runs/seed{V.SEED}/STAGE{stage:02d}.json")
    dec = V.SingleStreamDecoder(checkpoint=False).cuda()
    dec.load_state_dict(torch.load(V.ROOT / rec["checkpoint"], weights_only=True)["state"])
    s256, img, _ = QE.predict(dec.eval(), c)
    del dec
    torch.cuda.empty_cache()
    return s256, img


def describe(c, s256, img, rows):
    labels = np.array([r["label"] for r in rows])
    hws, masks = QE.masks_for(c, rows)
    maps = [M.to_native("dinomaly", x, hw) for x, hw in zip(s256.numpy(), hws)]
    ap = M.exact_ap(maps, masks)["ap"]
    im = img.numpy()
    good = s256.numpy()[labels == 0]
    loc = np.array([divmod(int(m.argmax()), 256) for m in good])
    border = float(np.mean((loc < 32).any(1) | (loc > 223).any(1)))   # argmax within 1/8 of the frame edge
    types = sorted({r["relative_path"].split("/")[2] for r in rows})
    per_type = {t: float(M.auroc(labels[[i for i, r in enumerate(rows) if r["relative_path"].split("/")[2] in (t, "good")]],
                                 im[[i for i, r in enumerate(rows) if r["relative_path"].split("/")[2] in (t, "good")]]))
                for t in types if t != "good"}
    return {"pixel_ap": ap, "image_auroc": M.auroc(labels, im), "auroc_per_defect_type": per_type,
            "good_score_mean_q50_q99": [float(im[labels == 0].mean()), float(np.quantile(im[labels == 0], .5)),
                                        float(np.quantile(im[labels == 0], .99))],
            "anom_score_mean_q50": [float(im[labels == 1].mean()), float(np.quantile(im[labels == 1], .5))],
            "good_argmax_border_frac": border}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "FORENSICS.json"
    require(not target.exists(), "exists")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(2.0 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    ids = V.identities()
    res = {"dose": dose()}
    res["retention"], res["retention_summary"] = retention()
    cases = {"screw": (9, 10, 15), "capsule": (2, 3, 6, 10, 15), "cable": (2, 15)}
    for c, stages in cases.items():
        rows = ids["test"][c]
        r = {}
        for st in stages:
            r[f"decoder_stage{st}"] = describe(c, *decoder_scores(st, c), rows)
            print(c, st, round(r[f"decoder_stage{st}"]["image_auroc"], 4), round(r[f"decoder_stage{st}"]["pixel_ap"], 4),
                  flush=True)
        r["knn1_raw_feature_ceiling"] = describe(c, *knn_scores(c), rows)
        print(c, "knn", round(r["knn1_raw_feature_ceiling"]["image_auroc"], 4),
              round(r["knn1_raw_feature_ceiling"]["pixel_ap"], 4), flush=True)
        res[c] = r
    D.write_json(target, res)
    print("written", target, flush=True)


if __name__ == "__main__":
    main()
