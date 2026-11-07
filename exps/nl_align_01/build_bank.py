"""Build the stage-1..9 Dinomaly synthetic bank for NL-ALIGN-01.

The historical NLCAD bank manifest is not reused as a storage location: its
feature payloads are absent in this checkout and this task needs a future-task
exclusion. This builder extracts only categories through stage 9 and only the
current/replay identities that can be legal by stage 9.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from exps.dinomaly import common as D
from exps.dinomaly import encoder as E
from exps.dinomaly.common import require, sha
from exps.nlcad import bank as BK
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

ROOT = D.ROOT
OUT = ROOT / "results/hope_cad/nl_align_01"
LAST_STAGE = 9
GENERATORS = ("Q2", "Q3")
VARIANTS = 2


def host_sets(seed, n_train):
    perms, _ = __import__("exps.rh01.common", fromlist=["buffer_plan"]).buffer_plan(seed, n_train)
    quota = {c: 0 for c in C.STREAM[:LAST_STAGE]}
    for k in range(1, LAST_STAGE + 1):
        for c, idx in C.buffer_index(seed, n_train, list(C.STREAM[:k])).items():
            if c in quota:
                quota[c] = max(quota[c], len(idx))
    out = {}
    for ci, c in enumerate(C.STREAM[:LAST_STAGE]):
        fixed = torch.randperm(n_train[c], generator=torch.Generator().manual_seed(31337 + ci)).tolist()
        out[c] = {"current": fixed[:BK.CURRENT_HOSTS], "buffer": perms[c][:quota[c]]}
    return out


def build(seed, device="cuda"):
    require(torch.cuda.is_available(), "CUDA is required for the bank build")
    __import__("exps.dinomaly.run", fromlist=["configure"]).configure()
    proto = RB.verify_lock()
    n_train = R1.n_train(proto)
    hosts = host_sets(seed, n_train)
    out_dir = OUT / f"bank_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = E.load(D.CKPT["base"][0], "base", device)
    gen = __import__("exps.ptag.synth", fromlist=["NativePerlin"]).NativePerlin()
    manifest = {"seed": seed, "stage_max": LAST_STAGE, "generators": list(GENERATORS),
                "variants": VARIANTS, "hosts": hosts, "files": {}, "encoder_sha256": sha(D.CKPT["base"][0])}
    for ci, c in enumerate(C.STREAM[:LAST_STAGE]):
        idx = sorted(set(hosts[c]["current"]) | set(hosts[c]["buffer"]))
        for g in GENERATORS:
            target = out_dir / f"{g}_{c}.pt"
            require(not target.exists(), f"exists: {target}")
            feats, covs, keys = [], [], []
            t0 = time.perf_counter()
            for i in idx:
                row = proto["train"][c][i]
                with Image.open(D.source(row)) as im:
                    img = np.array(im.convert("RGB"), dtype=np.uint8)
                for v in range(VARIANTS):
                    out, mask, _ = (BK.q2(gen, img, c, ci, i, v) if g == "Q2" else BK.q3(img, ci, i, v)) \
                        if hasattr(BK, "q2") else (None, None, None)
                    if out is None:
                        from exps.ptag.synth import q2, q3
                        out, mask, _ = q2(gen, img, c, ci, i, v) if g == "Q2" else q3(img, ci, i, v)
                    x = BK.preprocess_array(out)[None].to(device)
                    with torch.no_grad():
                        feats.append(enc.groups(x).float().cpu()[0])
                    covs.append(BK.patch_coverage(mask))
                    keys.append((c, i, v))
            torch.save({"features": torch.stack(feats), "coverage": torch.stack(covs), "keys": keys}, target)
            manifest["files"][f"{g}_{c}"] = {"n": len(keys), "sha256": sha(target),
                                               "seconds": time.perf_counter() - t0}
            print("bank", seed, g, c, len(keys), flush=True)
    D.write_json(OUT / f"BANK_seed{seed}.json", manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(2711, 2712, 2713))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    build(args.seed, args.device)


if __name__ == "__main__":
    main()
