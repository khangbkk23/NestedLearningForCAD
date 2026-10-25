# exps/calm02/extract.py
"""CALM-02 frozen feature extraction (CUDA, TF32 off) into a create-exclusive cache.

Train and test images go through the image-only path of the canonical ``_ImageDataset``
(``training=True`` returns no mask), so no test mask is opened here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.benchmark_protocol_v1 import _ImageDataset
from exps.calm02.data import CHECKPOINT, CHECKPOINT_SHA, MEAN, STD, ROOT, identities
from exps.task_p.p0_normal_support import require
from exps.task_p.p1_data import sha

OUT = ROOT / "results/hope_cad/calm02/features"


def extractor(device):
    from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor
    require(sha(CHECKPOINT) == CHECKPOINT_SHA, "checkpoint mismatch")
    return CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=str(CHECKPOINT),
                                                   checkpoint_identity="sha256:" + CHECKPOINT_SHA), device)


@torch.no_grad()
def extract_split(model, root, rows, device, batch_size):
    data = _ImageDataset(root, rows, 224, MEAN, STD, training=True)   # image-only path
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4)
    out = torch.empty(len(rows), 784, 768, dtype=torch.float32)
    i = 0
    for batch in loader:
        f = model.extract_patch_features(batch["images"].to(device)).float().cpu()
        require(bool(torch.isfinite(f).all()), "non-finite features")
        out[i:i + len(f)] = f; i += len(f)
    require(i == len(rows), "row count mismatch")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("benchmark", choices=("mvtec", "visa"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids = identities(args.benchmark)
    root = Path(ids["root"])
    out = OUT / args.benchmark
    out.mkdir(parents=True, exist_ok=True)
    model = extractor(args.device)
    manifest = {"benchmark": args.benchmark, "checkpoint_sha256": CHECKPOINT_SHA, "device": args.device,
                "torch": torch.__version__, "tf32": False, "preprocessing": "dataset/benchmark_protocol_v1._ImageDataset",
                "tasks": {}}
    for t in ids["tasks"]:
        rec = {}
        for split in ("train", "test"):
            target = out / f"{t['task']}_{split}.pt"
            require(not target.exists(), f"cache exists: {target}")
            start = time.perf_counter()
            feats = extract_split(model, root, t[split], args.device, args.batch_size)
            for r in t[split]:
                r["source_sha256"] = sha(root / r["relative_path"])
            part = target.with_suffix(".part")
            torch.save({"features": feats, "rows": t[split]}, part)
            part.replace(target)
            rec[split] = {"file": target.name, "file_sha256": sha(target), "n": len(t[split]),
                          "feature_sha256": hashlib.sha256(feats.numpy().tobytes()).hexdigest(),
                          "seconds": time.perf_counter() - start}
            print(args.benchmark, t["task"], split, len(t[split]), f"{rec[split]['seconds']:.0f}s", flush=True)
        manifest["tasks"][t["task"]] = rec
    with (out / "MANIFEST.json").open("x") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
