"""NORM-00: single-factor normalisation control on the exact 1-NN ceiling (prepared; execution needs owner
authorisation: GPU feature extraction ~0.3 GPU-h and reads of the historically exposed MVTec test masks).

    python -m exps.norm_00.run lock
    python -m exps.norm_00.run evaluate

Prereg: agents/plans/norm_00_prereg.md. Arms differ ONLY in the input normalisation:
  IMN  = the existing CALM-02 caches (ImageNet mean/std; the current project front end), read unchanged;
  INC  = the same images, same resize (full image, 224, PIL bilinear), same ViT-B/8 block 9, re-extracted with
         mean = std = (0.5, 0.5, 0.5) (the checkpoint's own configuration). Nothing is cached to disk.
Per category (single task, no continual learning): bank = all training patches of the category; scorer = exact
Euclidean 1-NN (certified, exps.nl_cad_pilot_00.nnx); maps 28x28 -> native bilinear and 224 x 224; image score = map
max. Historical code and artifacts are imported read-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hope_cad/norm_00"
LOCK, RESULT = OUT / "LOCK.json", OUT / "RESULT.json"
FEAT = ROOT / "results/hope_cad/calm02/features/mvtec"
MVTEC = ROOT / "data/mvtec"
ORDER = ("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw",
         "tile", "toothbrush", "transistor", "wood", "zipper")
WEAK = ("screw", "cable", "capsule", "transistor", "zipper")
NORMS = {"IMN": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), "INC": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))}
RULE = {"go_macro_ap": 0.010, "go_weak_auroc": 0.05, "go_weak_n": 3, "stop_macro_ap": 0.005, "stop_weak_auroc": 0.02}
BOOT = {"B": 10000, "seed": 20261119}
CAPS = {"gpu_wall_h": 1.0, "rss_gb": 8.5, "vram_gb": 5.0}
SOURCES = ("exps/norm_00/__init__.py", "exps/norm_00/run.py", "exps/norm_00/test_norm_00.py",
           "dataset/benchmark_protocol_v1.py", "models/feature_extractors/cadic_vit_v1.py", "exps/calm02/data.py",
           "exps/nl_cad_pilot_00/nnx.py", "exps/cadfinal/metrics.py", "exps/bench_aud_00/run.py",
           "agents/plans/norm_00_prereg.md")
_LOG = []


def require(c, msg):
    if not c:
        raise RuntimeError(msg)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ custody

def decide(path):
    p = Path(os.fsdecode(path)).resolve()
    s = str(p)
    gov = any(s == str(r) or s.startswith(str(r) + "/") for r in (ROOT / "results", ROOT / "data", ROOT / "checkpoints"))
    if not gov:
        return True
    if s.startswith(str(OUT)):
        return True
    if p.parent == FEAT.resolve() and p.name.endswith(("_train.pt", "_test.pt")):
        return True
    if p.name == "vit_base_patch8_224_augreg_in21k_state_dict.pth":
        return True
    if s.startswith(str(MVTEC.resolve()) + "/"):
        parts = p.relative_to(MVTEC.resolve()).parts
        return len(parts) >= 3 and parts[0] in ORDER and parts[1] in ("train", "test", "ground_truth")
    return False


def _audit(ev, a):
    if ev == "open" and a and isinstance(a[0], (str, bytes, os.PathLike)):
        if not decide(a[0]):
            _LOG.append({"denied": os.fsdecode(a[0])})
            raise PermissionError(f"NORM-00 custody: refused {os.fsdecode(a[0])}")


# ------------------------------------------------------------------ rules

def verdict(d_macro_ap, d_weak_auroc):
    """d_weak_auroc: {category: INC - IMN Image AUROC} for the WEAK categories."""
    n_go = sum(v >= RULE["go_weak_auroc"] for v in d_weak_auroc.values())
    if d_macro_ap >= RULE["go_macro_ap"] or n_go >= RULE["go_weak_n"]:
        return "NORMALIZATION_MATTERS_GO"
    if abs(d_macro_ap) < RULE["stop_macro_ap"] and all(abs(v) < RULE["stop_weak_auroc"] for v in d_weak_auroc.values()):
        return "NORMALIZATION_EXONERATED"
    if d_macro_ap <= -RULE["go_macro_ap"]:
        return "NORMALIZATION_HARMFUL"
    return "NORMALIZATION_INCONCLUSIVE"


def load_mask(row, hw):
    from PIL import Image
    if not row["label"]:
        return np.zeros(hw, dtype=bool)
    with Image.open(MVTEC / row["mask_path"]) as im:
        m = np.asarray(im.convert("L")) > 0
    require(m.shape == tuple(hw), f"mask size {row['mask_path']}")
    return m


def extract(model, rows, norm, device, batch=16):
    import torch
    from dataset.benchmark_protocol_v1 import _ImageDataset
    ds = _ImageDataset(MVTEC, rows, 224, *NORMS[norm], training=True)
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            x = torch.stack([ds[j]["images"] for j in range(i, min(i + batch, len(rows)))]).to(device)
            out.append(model.extract_patch_features(x).float().cpu())
    f = torch.cat(out)
    require(f.shape == (len(rows), 784, 768) and bool(f.isfinite().all()), "feature shape / finiteness")
    return f


def score(train, test, device):
    """Exact certified Euclidean 1-NN of every test patch against all training patches. Returns [n_test, 784] FP64."""
    import torch
    from exps.nl_cad_pilot_00 import nnx
    bank = train.reshape(-1, 768).contiguous().to(device)
    rows = nnx.Rows(bank, torch.arange(len(bank), device=device))
    d = [nnx.certified_nn(test[i:i + 8].reshape(-1, 768).contiguous(), rows)[0].cpu() for i in range(0, len(test), 8)]
    return torch.cat(d).reshape(len(test), 784).double().numpy()


def metrics(dist, labels, masks_native, hw):
    import torch
    from sklearn.metrics import roc_auc_score
    from exps.bench_aud_00 import run as B
    from exps.cadfinal import metrics as MX
    import torch.nn.functional as F
    x = torch.from_numpy(dist).float().reshape(-1, 1, 28, 28)
    nat = F.interpolate(x, size=hw, mode="bilinear", align_corners=False)[:, 0].numpy()
    m224 = B.maps224(dist)
    return {"pixel_ap_native": float(MX.exact_ap(list(nat), masks_native)["ap"]),
            "pixel_ap_224": float(MX.exact_ap(list(m224), [B.mask224(m) for m in masks_native])["ap"]),
            "image_auroc": float(roc_auc_score(labels, nat.reshape(len(nat), -1).max(1)))}


def write_lock():
    require(not LOCK.exists(), "LOCK exists")
    OUT.mkdir(parents=True, exist_ok=True)
    lk = {"schema": "norm_00_lock_v1", "sources": {s: sha(ROOT / s) for s in SOURCES}, "rule": RULE, "boot": BOOT,
          "caps": CAPS, "norms": NORMS, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with LOCK.open("x") as f:
        json.dump(lk, f, indent=1)
    print("LOCK", sha(LOCK))


def evaluate():
    import torch
    sys.addaudithook(_audit)
    lk = json.loads(LOCK.read_text())
    require(all(sha(ROOT / s) == h for s, h in lk["sources"].items()), "LOCK: sources changed")
    require(not RESULT.exists(), "RESULT exists (one-shot)")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from exps.calm02.data import CHECKPOINT, CHECKPOINT_SHA
    from exps.nl_sgm_01a import am03_eval as A3
    from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor
    dev = "cuda"
    model = CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=str(CHECKPOINT),
                                                    checkpoint_identity="sha256:" + CHECKPOINT_SHA), dev)
    t0, per, parity = time.time(), {}, {}
    for c in ORDER:
        tr = torch.load(FEAT / f"{c}_train.pt", map_location="cpu", weights_only=False, mmap=True)
        te = torch.load(FEAT / f"{c}_test.pt", map_location="cpu", weights_only=False, mmap=True)
        rows_tr, rows_te = tr["rows"], te["rows"]
        from PIL import Image
        with Image.open(MVTEC / rows_te[0]["relative_path"]) as im:
            hw = (im.size[1], im.size[0])
        labels = np.array([int(r["label"]) for r in rows_te])
        masks = [load_mask(r, hw) for r in rows_te]
        # parity: re-extracting a few cached images with IMN reproduces the cache (same code path as INC)
        chk = extract(model, rows_te[:4], "IMN", dev)
        parity[c] = float((chk - te["features"][:4].float()).norm() / te["features"][:4].float().norm())
        require(parity[c] < 1e-4, f"IMN re-extraction does not reproduce the cache {c}: {parity[c]}")
        res = {}
        for arm in ("IMN", "INC"):
            ftr = tr["features"].float() if arm == "IMN" else extract(model, rows_tr, "INC", dev)
            fte = te["features"].float() if arm == "IMN" else extract(model, rows_te, "INC", dev)
            res[arm] = metrics(score(ftr, fte, dev), labels, masks, hw)
            del ftr, fte
            torch.cuda.empty_cache()
        per[c] = res
        wall = time.time() - t0
        require(wall <= CAPS["gpu_wall_h"] * 3600, "CAP: wall")
        print(c, json.dumps(res), f"{wall:.0f}s", flush=True)
    from exps.nl_sgm_01a import am03_eval as A3b
    out = {"per_category": per, "parity_rel": parity}
    for m in ("pixel_ap_native", "pixel_ap_224", "image_auroc"):
        a = [per[c]["INC"][m] for c in ORDER]
        b = [per[c]["IMN"][m] for c in ORDER]
        out[f"INC_minus_IMN_{m}"] = A3b.bootstrap_delta(a, b, seed=BOOT["seed"], B=BOOT["B"])
        out[f"macro_{m}"] = {"IMN": float(np.mean(b)), "INC": float(np.mean(a))}
    dw = {c: per[c]["INC"]["image_auroc"] - per[c]["IMN"]["image_auroc"] for c in WEAK}
    out["weak_delta_image_auroc"] = dw
    out["verdict"] = verdict(out["INC_minus_IMN_pixel_ap_native"]["point"], dw)
    out["custody_denied"] = [x for x in _LOG if "denied" in x]
    out["wall_s"] = time.time() - t0
    out["exposure"] = "MVTec AD official test, historically exposed; diagnostic only; adoption needs VisA-DEV confirmation"
    with RESULT.open("x") as f:
        json.dump(out, f, indent=1)
    print("VERDICT", out["verdict"], json.dumps({k: v for k, v in out.items() if k.startswith("macro_")}))


if __name__ == "__main__":
    {"lock": write_lock, "evaluate": evaluate}[sys.argv[1]]()
