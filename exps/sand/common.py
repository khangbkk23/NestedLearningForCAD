"""SAND-FINAL (task agents/tasks/task_sand_final.md): constants, custody, fusion scorer, readouts, lock.

Protocol: agents/plans/sand_final_protocol.md (hashed into LOCK.json). Canonical modules are imported, never edited:
decoder + canonical decoder scorer = exps.vitb8_q2.common; support memory = exps.nl_sgm_01a_c2dom (DOM_PQ_B2);
exact 1-NN = exps.nl_sgm_01a.exact.nn; native maps = exps.nl_sgm_01a.am03_eval.native_maps;
Pixel AP = exps.cadfinal.metrics.exact_ap.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from exps.cadfinal import metrics as MX
from exps.nl_sgm_01a import am03_eval as A3
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a_c2dom import common as D2
from exps.task_p.p0_normal_support import require

ROOT = C.ROOT
OUT = ROOT / "results/hope_cad/sand_final"
LOCK, PROTOCOL = OUT / "LOCK.json", OUT / "PROTOCOL.json"
PLAN = "agents/plans/sand_final_protocol.md"
TASK = "agents/tasks/task_sand_final.md"
ORDER = C.ORDER
SEEDS = (2711, 2712, 2713)
SIDE, TOKENS, DIM = 28, 784, 768

MEM_ARM = "DOM_PQ_B2"
MEM_DIR = D2.OUT / "streams" / MEM_ARM
MEM_EVAL = D2.OUT / "eval" / MEM_ARM
A1_OUT = ROOT / "results/hope_cad/vitb8_nl00"
A1_RUN = A1_OUT / "runs/q2off_seed2711"

BOOT = {"B": 2000, "seed": 20261108, "unit": "category (paired, resample categories with replacement)"}
GATE0 = {"min_final_pixel_ap_gain": 0.010, "max_final_image_auroc_loss": 0.010}
GATE1 = {"min_final_pixel_ap_gain": 0.005, "max_acq_pixel_ap_loss": 0.010, "max_final_image_auroc_loss": 0.010}
RHO = 0.5
TOL = {"a1_image_score": 1e-5, "a1_pixel_ap": 1e-6, "a2_image_score": 1e-9}
CAPS = {"rss_gb": 8.5, "vram_gib_train": 3.5}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


sha, text_sha, write_json = C.sha, C.text_sha, C.write_json


def read_json(path):
    return json.loads(Path(path).read_text())


class Log:
    def __init__(self, name):
        (OUT / "logs").mkdir(parents=True, exist_ok=True)
        self.f = (OUT / "logs" / f"{name}.log").open("a")

    def __call__(self, *a):
        s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def verify_lock():
    """LOCK source hashes, overridden only by create-exclusive amendments (AMENDMENT_NN.json, applied in order). An
    amendment never changes a rule; one written after test data was opened may only touch analysis/reporting code."""
    lk = read_json(LOCK)
    src = dict(lk["sources"])
    for a in sorted(OUT.glob("AMENDMENT_*.json")):
        am = read_json(a)
        require(am["protocol_change"] is False, f"invalid amendment {a}")
        if am["test_data_opened_before"] is not False:
            require(am.get("post_eval_reporting_only") is True and all(
                s.startswith("exps/sand/analyze") for s in am["sources"]), f"invalid post-eval amendment {a}")
        src.update(am["sources"])
    bad = [s for s, h in src.items() if sha(ROOT / s) != h]
    require(not bad, f"LOCK: source changed after lock: {bad}")
    require(sha(PROTOCOL) == lk["protocol_sha256"], "LOCK: protocol changed after lock")
    return lk


# ------------------------------------------------------------------ fusion scorer (protocol section 2)

def calibrate(a_probe, d_probe):
    """Population mean/std (FP64) of the decoder residual and of the support distance over normal probe patches."""
    a = np.maximum(np.asarray(a_probe, dtype=np.float64).ravel(), 0.0)
    d = np.asarray(d_probe, dtype=np.float64).ravel()
    require(a.size == d.size and a.size > 1, "calibration probes misaligned")
    cal = {"mu_theta": float(a.mean()), "sd_theta": float(a.std()), "mu_m": float(d.mean()), "sd_m": float(d.std()),
           "n_patches": int(a.size)}
    require(all(np.isfinite(v) for v in cal.values()) and cal["sd_theta"] > 0 and cal["sd_m"] > 0
            and cal["mu_theta"] >= 0 and cal["mu_m"] >= 0, f"degenerate calibrators {cal}")
    return cal


def zscores(a, d, cal):
    """a: decoder residual [n,784], d: support distance [n,784] -> (z_theta, z_M) FP64."""
    a = np.maximum(np.asarray(a, dtype=np.float64), 0.0)
    d = np.asarray(d, dtype=np.float64)
    require(a.shape == d.shape and a.ndim == 2 and a.shape[1] == TOKENS and float(d.min()) >= 0, "bad channel maps")
    return (a - cal["mu_theta"]) / cal["sd_theta"], (d - cal["mu_m"]) / cal["sd_m"]


def shift(cal):
    """Each z >= -mu/sd because a, d >= 0; adding this constant makes the fused map non-negative."""
    return max(cal["mu_theta"] / cal["sd_theta"], cal["mu_m"] / cal["sd_m"])


def fuse(a, d, cal):
    """g = min(z_theta, z_M) + shift, FP32 [n,784], >= 0. Returns (g, fraction of patches where theta is the min)."""
    zt, zm = zscores(a, d, cal)
    g = np.minimum(zt, zm) + shift(cal)
    require(np.isfinite(g).all() and float(g.min()) >= -1e-9, "fused map negative / non-finite")
    return np.maximum(g, 0.0).astype(np.float32), float((zt < zm).mean())


def readout_native(g, hw):
    """Primary readout P: 28x28 -> native bilinear (AM03 chain); image score = map max."""
    m = A3.native_maps(torch.from_numpy(np.ascontiguousarray(g)).reshape(-1, TOKENS), hw)
    return m, m.reshape(len(m), -1).max(1)


_GAUSS = []


def readout_dinomaly(g, hw):
    """S1: the official Dinomaly chain applied to a 28x28 score grid (CPU): 224 (align_corners) -> 256 -> Gaussian
    5x5/sigma 4 -> native bilinear; image score = mean of the top 1 % of the 256x256 map."""
    from exps.dinomaly import official as O
    from exps.vitb8_q2 import common as V
    if not _GAUSS:
        _GAUSS.append(O.get_gaussian_kernel(*O.RECIPE["gaussian"]))
    x = torch.from_numpy(np.ascontiguousarray(g)).float().reshape(-1, 1, SIDE, SIDE)
    with torch.no_grad():
        x = F.interpolate(x, size=(V.IMG, V.IMG), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=V.EVAL_SIZE, mode="bilinear", align_corners=False)
        s = _GAUSS[0](x)
    flat = s.flatten(1)
    image = torch.sort(flat, dim=1, descending=True)[0][:, :int(flat.shape[1] * V.TOPK_RATIO)].mean(dim=1)
    maps = np.stack([MX.to_native("dinomaly", m, hw) for m in s[:, 0].numpy()])
    return maps, image.numpy()


def metrics(maps, image, labels, masks):
    return {"pixel_ap": float(MX.exact_ap(list(maps), masks)["ap"]),
            "image_auroc": float(roc_auc_score(labels, np.asarray(image, dtype=np.float64)))}


def zstats(z):
    z = np.asarray(z, dtype=np.float64).ravel()
    return {"mean": float(z.mean()), "std": float(z.std()), "q99": float(np.quantile(z, 0.99)), "max": float(z.max())}


# ------------------------------------------------------------------ gates (protocol sections 4 and 5)

def gate0(a1, a2, a3):
    """a1, a2, a3: {'final_pixel_ap', 'final_image_auroc'} macro values."""
    ap_gain = a3["final_pixel_ap"] - max(a1["final_pixel_ap"], a2["final_pixel_ap"])
    au_diff = a3["final_image_auroc"] - max(a1["final_image_auroc"], a2["final_image_auroc"])
    ok = ap_gain >= GATE0["min_final_pixel_ap_gain"] and au_diff >= -GATE0["max_final_image_auroc_loss"]
    return {"final_pixel_ap_gain_over_best_single": ap_gain, "final_image_auroc_diff_to_best_single": au_diff,
            "thresholds": GATE0, "pass": bool(ok), "verdict": "FUSION_SUPPORTED" if ok else "FUSION_NOT_SUPPORTED"}


def gate1(a3, a4):
    d = {"final_pixel_ap": a4["final_pixel_ap"] - a3["final_pixel_ap"],
         "acq_pixel_ap": a4["acq_pixel_ap"] - a3["acq_pixel_ap"],
         "final_image_auroc": a4["final_image_auroc"] - a3["final_image_auroc"]}
    ok = (d["final_pixel_ap"] >= GATE1["min_final_pixel_ap_gain"] and d["acq_pixel_ap"] >= -GATE1["max_acq_pixel_ap_loss"]
          and d["final_image_auroc"] >= -GATE1["max_final_image_auroc_loss"])
    return {"A4_minus_A3": d, "thresholds": GATE1, "pass": bool(ok),
            "verdict": "SLOW_TO_FAST_SUPPORTED" if ok else "SLOW_TO_FAST_NOT_SUPPORTED"}
