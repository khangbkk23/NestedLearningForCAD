"""VITB8-Q2 preparation (no outcome data is read here: test features are label-free, no mask is opened).

  python -m exps.vitb8_q2.prepare cache    # SHA-verified subsets of the frozen CALM-02 ViT-B/8 cache -> ext4
  python -m exps.vitb8_q2.prepare bank     # Q2 (DRAEM Perlin + DTD) bank on lawful normal hosts, ViT-B/8 features
"""
from __future__ import annotations

import argparse
import io
import time

import numpy as np
import torch
from PIL import Image

from exps.dinomaly import common as D
from exps.dinomaly.common import require, sha
from exps.vitb8_q2 import common as V


def _calm02(name):
    man = D.read(V.CALM02 / "MANIFEST.json")
    require(man["checkpoint_sha256"] == "755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415"
            and man["tf32"] is False, "CALM-02 manifest")
    task, split = name.rsplit("_", 1)
    rec = man["tasks"][task][split]
    raw = (V.CALM02 / rec["file"]).read_bytes()
    import hashlib
    require(hashlib.sha256(raw).hexdigest() == rec["file_sha256"], f"CALM-02 cache changed: {rec['file']}")
    d = torch.load(io.BytesIO(raw), weights_only=True)
    require(tuple(d["features"].shape[1:]) == (V.TOKENS, V.DIM) and d["features"].dtype == torch.float32, "shape")
    return d, rec["file_sha256"]


def _save(name, feats, paths, files, source):
    path = V.CACHE / f"{name}.pt"
    require(not path.exists(), f"exists: {path}")
    torch.save({"features": feats.contiguous(), "paths": paths}, path)
    files[name] = {"sha256": sha(path), "n": len(paths), "source": source}
    print("cache", name, len(paths), flush=True)


def cache():
    target = V.OUT / "CACHE.json"
    require(not target.exists(), "cache already built")
    V.CACHE.mkdir(parents=True, exist_ok=True)
    ids = V.identities()
    files = {}
    for c in V.STREAM:
        d, fsha = _calm02(f"{c}_train")
        index = {r["relative_path"]: (j, r["source_sha256"]) for j, r in enumerate(d["rows"])}
        for split, rows in (("train", ids["train"][c]), ("heldout", ids["heldout"].get(c, []))):
            if not rows:
                continue
            for r in rows:
                require(index[r["relative_path"]][1] == r["sha256"], f"source sha mismatch {r['relative_path']}")
            sel = torch.tensor([index[r["relative_path"]][0] for r in rows])
            _save(f"{split}_{c}", d["features"][sel].clone(), [r["relative_path"] for r in rows], files,
                  f"calm02/{c}_train.pt sha256:{fsha}")
        del d
        d, fsha = _calm02(f"{c}_test")
        paths = [r["relative_path"] for r in d["rows"]]
        require(paths == [r["relative_path"] for r in ids["test"][c]], f"test identity order {c}")
        _save(f"test_{c}", d["features"], paths, files, f"calm02/{c}_test.pt sha256:{fsha}")
        del d
    D.write_json(target, {"files": files, "utc": __import__("exps.dinomaly.run", fromlist=["now"]).now(),
                          "note": "exact FP32 row subsets of the frozen CALM-02 ViT-B/8 layer-9 cache"})


def coverage(mask):
    """Native boolean mask -> fraction of positive pixels per 8x8 patch of the 224x224 input (nearest resize)."""
    m = Image.fromarray(mask.astype(np.uint8) * 255).resize((V.IMG, V.IMG), Image.NEAREST)
    t = torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.)
    return torch.nn.functional.avg_pool2d(t[None, None], V.PATCH)[0, 0]


def bank():
    from exps.calm02.extract import extractor
    from exps.nlcad import bank as BK
    from exps.ptag.synth import NativePerlin, preprocess, q2
    target = V.OUT / "BANK.json"
    require(not target.exists(), "bank already built")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids = V.identities()
    hosts = BK.host_sets(V.SEED, ids["n_train"])          # FS_B host rule: 16 current + every future buffer member
    model = extractor("cuda")
    gen = NativePerlin()
    man = {"seed": V.SEED, "generator": "Q2 exps.ptag.synth.q2 (NativePerlin DRAEM Perlin + DTD)", "variants": 2,
           "hosts": hosts, "files": {}, "extractor": "CADIC ViT-B/8 layer 9 (calm02.extract.extractor)"}
    for ci, c in enumerate(V.STREAM):
        path = V.CACHE / f"bank_Q2_{c}.pt"
        require(not path.exists(), f"exists: {path}")
        idx = sorted(set(hosts[c]["current"]) | set(hosts[c]["buffer"]))
        feats, covs, keys = [], [], []
        t0 = time.perf_counter()
        for i in idx:
            row = ids["train"][c][i]
            with Image.open(D.source(row)) as im:
                img = np.array(im.convert("RGB"), dtype=np.uint8)
            outs = []
            for v in range(2):
                o, mask, _ = q2(gen, img, c, ci, i, v)
                outs.append(o)
                covs.append(coverage(mask))
                keys.append((c, i, v))
            with torch.no_grad():
                f = model.extract_patch_features(preprocess(outs).cuda()).float().cpu()
            require(bool(torch.isfinite(f).all()), "non-finite bank features")
            feats.append(f)
        torch.save({"features": torch.cat(feats), "coverage": torch.stack(covs), "keys": keys}, path)
        man["files"][c] = {"n": len(keys), "sha256": sha(path), "seconds": time.perf_counter() - t0}
        print("bank", c, len(keys), f"{time.perf_counter() - t0:.0f}s", flush=True)
    D.write_json(target, man)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("cache", "bank"))
    a = ap.parse_args()
    cache() if a.stage == "cache" else bank()


if __name__ == "__main__":
    main()
