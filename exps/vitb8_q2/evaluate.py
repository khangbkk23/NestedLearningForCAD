"""VITB8-Q2 evaluation with the frozen scorer (lock-verified). Masks are opened only here, after the lock.

  python -m exps.vitb8_q2.evaluate follow     # evaluate stage checkpoints in order as they appear (resumable)
  python -m exps.vitb8_q2.evaluate summary    # FINAL.json: final macro AP / AUROC, forgetting, trajectory
  python -m exps.vitb8_q2.evaluate latency    # image -> ViT-B/8 -> decoder -> map -> native, batch 1

Every stage s: all acquired categories (stream positions 1..s), full MVTec test cohort, native Pixel AP (exact),
Image AUROC, normal image-score tails. EVAL_STAGE{s}.json is create-exclusive (one-shot per stage).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from exps.cadfinal import metrics as M
from exps.dinomaly import common as D
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.vitb8_q2 import common as V
from exps.vitb8_q2 import train as T

OUT = V.OUT
DATA = V.ROOT / "data/mvtec"
_MASKS = {}


def eval_path(s):
    return OUT / f"EVAL_STAGE{s:02d}.json"


def masks_for(c, rows):
    """Native masks (bit-packed, cached in-process) and native sizes."""
    if c not in _MASKS:
        hws = [M.native_hw(DATA / r["relative_path"]) for r in rows]
        packed = [np.packbits(M.native_mask(DATA / r["mask_path"] if r["mask_path"] else None, hw))
                  for r, hw in zip(rows, hws)]
        _MASKS[c] = (hws, packed)
    hws, packed = _MASKS[c]
    return hws, [np.unpackbits(p, count=hw[0] * hw[1]).reshape(hw).astype(bool) for p, hw in zip(packed, hws)]


@torch.no_grad()
def predict(dec, c):
    f, paths = V.test_features(c)
    s256, img = [], []
    for xb in f.split(16):
        en, de = dec(xb.cuda())
        s, i = V.score(en, de)
        s256.append(s.cpu()); img.append(i.cpu())
    return torch.cat(s256), torch.cat(img), paths


def eval_stage(s, ids, keep_maps=False):
    target = eval_path(s)
    require(not target.exists(), f"one-shot: {target}")
    rec = D.read(T.RUN / f"STAGE{s:02d}.json")
    ck = V.ROOT / rec["checkpoint"]
    require(sha(ck) == rec["checkpoint_sha256"], f"checkpoint changed: {ck}")
    dec = V.SingleStreamDecoder(checkpoint=False).cuda()
    dec.load_state_dict(torch.load(ck, weights_only=True)["state"])
    dec.eval()
    out, t0 = {}, time.perf_counter()
    saved = {}
    for c in V.STREAM[:s]:
        rows = ids["test"][c]
        s256, img, paths = predict(dec, c)
        require(paths == [r["relative_path"] for r in rows], f"identity order {c}")
        require(bool(torch.isfinite(s256).all()) and bool(torch.isfinite(img).all()), f"non-finite scores {c}")
        labels = np.array([r["label"] for r in rows])
        hws, masks = masks_for(c, rows)
        maps = [M.to_native("dinomaly", x, hw) for x, hw in zip(s256.numpy(), hws)]
        ap = M.exact_ap(maps, masks)["ap"]
        del maps, masks
        im = img.numpy()
        good, bad = im[labels == 0], im[labels == 1]
        out[c] = {"pixel_ap": ap, "image_auroc": M.auroc(labels, im), "n": len(rows), "n_anomalous": int(labels.sum()),
                  "good_image_score_mean": float(good.mean()), "good_image_score_q99": float(np.quantile(good, 0.99)),
                  "anomalous_image_score_mean": float(bad.mean()),
                  "map_min": float(s256.min()), "map_max": float(s256.max()), "map_std": float(s256.std())}
        if keep_maps:
            saved[c] = {"s256": s256, "image": img, "paths": paths}
        print("eval stage", s, c, "AP", round(ap, 4), "AUROC", round(out[c]["image_auroc"], 4), flush=True)
    res = {"stage": s, "checkpoint_sha256": rec["checkpoint_sha256"], "per_category": out,
           "macro_pixel_ap": float(np.mean([v["pixel_ap"] for v in out.values()])),
           "macro_image_auroc": float(np.mean([v["image_auroc"] for v in out.values()])),
           "seconds": time.perf_counter() - t0, "lock_sha256": sha(T.LOCK), "utc": DR.now()}
    if keep_maps:
        p = V.CACHE / f"predictions_stage{s:02d}.pt"
        require(not p.exists(), f"exists: {p}")
        torch.save(saved, p)
        res["predictions"] = {"path": str(p), "sha256": sha(p), "note": "label-free 256x256 maps + image scores"}
    D.write_json(target, res)
    print("stage", s, "macro AP", round(res["macro_pixel_ap"], 4), "macro AUROC", round(res["macro_image_auroc"], 4),
          flush=True)
    del dec
    torch.cuda.empty_cache()


def follow(poll=60):
    DR.configure()
    T.cap("1.2")
    T.verify_lock()                  # evaluation only under the frozen contract
    ids = V.identities()
    for s in range(1, len(V.STREAM) + 1):
        if eval_path(s).exists():
            continue
        while not (T.RUN / f"STAGE{s:02d}.json").exists():
            time.sleep(poll)
        eval_stage(s, ids, keep_maps=(s == len(V.STREAM)))
    print("evaluation complete", flush=True)


def summary():
    T.verify_lock()
    target = OUT / "FINAL.json"
    require(not target.exists(), "exists")
    ev = {s: D.read(eval_path(s)) for s in range(1, 16)}
    fin = ev[15]["per_category"]
    acq = {c: ev[j + 1]["per_category"][c]["pixel_ap"] for j, c in enumerate(V.STREAM)}
    acq_auroc = {c: ev[j + 1]["per_category"][c]["image_auroc"] for j, c in enumerate(V.STREAM)}
    forg = {c: acq[c] - fin[c]["pixel_ap"] for c in V.STREAM[:-1]}
    tr = D.read(T.RUN / "TRAIN.json")
    st = [D.read(T.RUN / f"STAGE{s:02d}.json") for s in range(1, 16)]
    res = {"final_macro_pixel_ap": ev[15]["macro_pixel_ap"], "final_macro_image_auroc": ev[15]["macro_image_auroc"],
           "final_per_category": {c: {"pixel_ap": fin[c]["pixel_ap"], "image_auroc": fin[c]["image_auroc"]}
                                  for c in V.STREAM},
           "acquisition_pixel_ap": acq, "acquisition_image_auroc": acq_auroc,
           "forgetting_acq_to_final": {"per_category": forg, "macro_signed": float(np.mean(list(forg.values()))),
                                       "macro_positive_part": float(np.mean(np.maximum(list(forg.values()), 0))),
                                       "worst": [max(forg, key=forg.get), max(forg.values())],
                                       "image_auroc_macro_signed": float(np.mean(
                                           [acq_auroc[c] - fin[c]["image_auroc"] for c in V.STREAM[:-1]]))},
           "trajectory": {s: {"macro_pixel_ap": ev[s]["macro_pixel_ap"], "macro_image_auroc": ev[s]["macro_image_auroc"]}
                          for s in ev},
           "stage_average_macro_pixel_ap": float(np.mean([ev[s]["macro_pixel_ap"] for s in ev])),
           "training": {"seconds_total": tr["train_seconds_total"], "peak_allocated_mib": tr["peak_allocated_mib_max"],
                        "persistent_decoder_bytes": tr["persistent_decoder_bytes"],
                        "nonfinite_skips_total": tr["nonfinite_skips_total"],
                        "clip_saturation_per_stage": [r["clip_saturation"] for r in st],
                        "aux_loss_last50_per_stage": [r["aux"]["aux_loss_last50"] for r in st],
                        "synthetic_diag_macro_per_stage": [r["synthetic_q2_diag_ap"]["macro"] for r in st]},
           "eval_files": {s: sha(eval_path(s)) for s in ev}, "lock_sha256": sha(T.LOCK), "utc": DR.now()}
    D.write_json(target, res)
    print({k: res[k] for k in ("final_macro_pixel_ap", "final_macro_image_auroc", "stage_average_macro_pixel_ap")},
          res["forgetting_acq_to_final"]["macro_signed"], flush=True)


@torch.no_grad()
def latency():
    from exps.calm02.extract import extractor
    from exps.ptag.synth import preprocess
    from PIL import Image
    DR.configure()
    T.verify_lock()
    target = OUT / "LATENCY.json"
    require(not target.exists(), "exists")
    rec = D.read(T.RUN / "STAGE15.json")
    enc = extractor("cuda")
    dec = V.SingleStreamDecoder(checkpoint=False).cuda().eval()
    dec.load_state_dict(torch.load(V.ROOT / rec["checkpoint"], weights_only=True)["state"])
    row = V.identities()["test"]["bottle"][0]
    with Image.open(DATA / row["relative_path"]) as im:
        a = np.array(im.convert("RGB"), dtype=np.uint8)
    hw = a.shape[:2]
    x = preprocess([a]).cuda()
    times = []
    for _ in range(25):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        f = enc.extract_patch_features(x)
        en, de = dec(f)
        s, img = V.score(en, de)
        from exps.dinomaly.readout import native
        native(s[0], hw)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    enc_bytes = sum(p.numel() * p.element_size() for p in enc.model.parameters())
    D.write_json(target, {"seconds_per_image_batch1_fp32_median": float(np.median(times[5:])), "repeats": 20,
                          "pipeline": "preprocess excluded; ViT-B/8 + decoder + R_D-style map + native upsample "
                                      f"{hw[0]}x{hw[1]}", "gpu": torch.cuda.get_device_name(0),
                          "encoder_bytes_fp32": enc_bytes, "decoder_bytes_fp32": V.state_bytes(dec),
                          "replay_buffer_feature_bytes": 24 * V.TOKENS * V.DIM * 4})
    print(D.read(target), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("follow", "summary", "latency"))
    a = ap.parse_args()
    {"follow": follow, "summary": summary, "latency": latency}[a.stage]()


if __name__ == "__main__":
    main()
