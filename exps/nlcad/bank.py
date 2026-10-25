# exps/nlcad/bank.py
"""Variant S synthetic bank: PTAG Q2 (DRAEM Perlin + DTD) and Q3 (CutPaste) anomalies generated from allowed
normal TRAINING hosts only, encoded once by the frozen DINOv2-reg encoder (deterministic, so caching is equivalent),
with masks mapped to the 28x28 patch grid of the 392x392 model input. No real defect image or label is used.

Hosts per (seed, category): the category's Replay-24 members over the whole stream (nested prefix of its fixed
buffer permutation, so every buffer member that will ever be replayed has synthetic versions) plus the first
CURRENT_HOSTS identities of a fixed seed-independent permutation (used while the category is the current task).
Access at stage s is enforced by `allowed(...)`: current-task hosts of the current category and the buffer members
of completed categories only.

  python -m exps.nlcad.bank build --seed 2711      (GPU; after RH-01b frees it)
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from exps.dinomaly import common as D
from exps.dinomaly import encoder as E
from exps.dinomaly.common import require, sha
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

CURRENT_HOSTS, VARIANTS = 16, 2
GENERATORS = ("Q2", "Q3")
BANK = C.CACHE.parent / "nlcad_bank"


def preprocess_array(a):
    """exps.dinomaly.encoder.preprocess on an in-memory uint8 RGB array."""
    im = Image.fromarray(a).convert("RGB").resize((E.SIZE, E.SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1)
    return (x - torch.tensor(E.MEAN).view(3, 1, 1)) / torch.tensor(E.STD).view(3, 1, 1)


def patch_coverage(mask):
    """Native boolean mask -> fraction of positive pixels in each of the 28x28 patches of the 392x392 input
    (mask resized like the image's geometry, nearest neighbour)."""
    m = Image.fromarray(mask.astype(np.uint8) * 255).resize((E.SIZE, E.SIZE), Image.NEAREST)
    t = torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.)
    return torch.nn.functional.avg_pool2d(t[None, None], 14)[0, 0]


def host_sets(seed, n_train):
    """{category: {"current": [...], "buffer": [...]}} training-identity indices."""
    perms, priority = __import__("exps.rh01.common", fromlist=["buffer_plan"]).buffer_plan(seed, n_train)
    quota = {c: 0 for c in C.STREAM}
    for k in range(1, len(C.STREAM)):
        for c, idx in C.buffer_index(seed, n_train, list(C.STREAM[:k])).items():
            quota[c] = max(quota[c], len(idx))
    out = {}
    for ci, c in enumerate(C.STREAM):
        fixed = torch.randperm(n_train[c], generator=torch.Generator().manual_seed(31337 + ci)).tolist()
        out[c] = {"current": fixed[:CURRENT_HOSTS], "buffer": perms[c][:quota[c]]}
    return out


def allowed(hosts, seed, n_train, stage):
    """Allowed host keys (category, index) at a 1-based stage: current task's current-hosts + completed buffers."""
    c = C.STREAM[stage - 1]
    keys = {(c, i) for i in hosts[c]["current"]}
    done = list(C.STREAM[:stage - 1])
    if done:
        for d, idx in C.buffer_index(seed, n_train, done).items():
            keys |= {(d, i) for i in idx}
    return keys


def build(seed, device="cuda"):
    from exps.ptag.synth import NativePerlin, q2, q3
    DR = __import__("exps.dinomaly.run", fromlist=["configure"])
    DR.configure()
    from exps.nlcad.run import cap
    cap()
    proto = RB.verify_lock()
    n = R1.n_train(proto)
    hosts = host_sets(seed, n)
    out_dir = BANK / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = E.load(D.CKPT["base"][0], "base", device)
    gen = NativePerlin()
    manifest = {}
    for ci, c in enumerate(C.STREAM):
        idx = sorted(set(hosts[c]["current"]) | set(hosts[c]["buffer"]))
        for g in GENERATORS:
            target = out_dir / f"{g}_{c}.pt"
            require(not target.exists(), f"exists: {target}")
            feats, covs, keys = [], [], []
            for i in idx:
                row = proto["train"][c][i]
                with Image.open(D.source(row)) as im:
                    img = np.array(im.convert("RGB"), dtype=np.uint8)
                for v in range(VARIANTS):
                    out, mask, _ = q2(gen, img, c, ci, i, v) if g == "Q2" else q3(img, ci, i, v)
                    x = preprocess_array(out)[None].to(device)
                    with torch.no_grad():
                        feats.append(enc.groups(x).float().cpu()[0])
                    covs.append(patch_coverage(mask))
                    keys.append((c, i, v))
            torch.save({"features": torch.stack(feats), "coverage": torch.stack(covs), "keys": keys}, target)
            manifest[f"{g}_{c}"] = {"n": len(keys), "sha256": sha(target)}
            print("bank", seed, g, c, len(keys), flush=True)
    D.write_json(D.ROOT / f"results/hope_cad/nlcad/BANK_seed{seed}.json",
                 {"hosts": {c: v for c, v in hosts.items()}, "files": manifest, "variants": VARIANTS,
                  "generators": GENERATORS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("build",))
    ap.add_argument("--seed", type=int, default=2711)
    a = ap.parse_args()
    build(a.seed)


if __name__ == "__main__":
    main()


def load_bank(seed):
    """{generator: [{"host": (category, index), "variant": v, "features": mmap view, "coverage": [28, 28]}]},
    SHA-verified against the bank manifest."""
    man = D.read(D.ROOT / f"results/hope_cad/nlcad/BANK_seed{seed}.json")
    out = {g: [] for g in GENERATORS}
    for g in GENERATORS:
        for c in C.STREAM:
            path = BANK / f"seed{seed}" / f"{g}_{c}.pt"
            require(sha(path) == man["files"][f"{g}_{c}"]["sha256"], f"bank file changed: {path}")
            d = torch.load(path, weights_only=False, mmap=True)
            for j, (cat, i, v) in enumerate(d["keys"]):
                out[g].append({"host": (cat, i), "variant": v, "features": d["features"][j],
                               "coverage": d["coverage"][j]})
    return out, {c: {k: list(v) for k, v in h.items()} for c, h in man["hosts"].items()}
