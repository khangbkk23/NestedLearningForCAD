"""VITB8-Q2: ViT-B/8 single-stream Dinomaly + Replay-24 + Fixed Q2 (task agents/tasks/task_vitb8_q2_single_stream.md).

Shared definitions: paths, identities, the single-stream decoder, the readout and cache/bank loaders.
A new single-stream baseline; not a Nested Learning contribution. One seed (2711).
"""
from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly.common import require, sha

ROOT = D.ROOT
OUT = ROOT / "results/hope_cad/vitb8_q2"
CACHE = Path.home() / "hope_cad_cache/vitb8_q2"
CALM02 = ROOT / "results/hope_cad/calm02/features/mvtec"
TASK = "agents/tasks/task_vitb8_q2_single_stream.md"
SEED = 2711
ITERS = 667
STREAM = ("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw",
          "tile", "toothbrush", "transistor", "wood", "zipper")
TOKENS, DIM, SIDE, PATCH, IMG = 784, 768, 28, 8, 224
EVAL_SIZE = O.RECIPE["eval_resize_mask"]          # 256
TOPK_RATIO = O.RECIPE["max_ratio"]                # 0.01 -> image score = mean of the top 1% of the 256x256 map


# --- decoder -----------------------------------------------------------------------------------------


class SingleStreamDecoder(nn.Module):
    """Dinomaly ViTill decoder half on ONE frozen feature map.

    x [B,784,768] (ViT-B/8 layer 9, CLS removed, no register tokens -> zero prefix tokens)
      -> noisy bMlp bottleneck (dropout 0.2) -> 8 linear-attention blocks.
    Readout (explicit, single group): mean of the 8 decoder block outputs (Dinomaly's loose group fusion with one
    group spanning all blocks) -> [B,768,28,28]. Target: the same frozen map (stop-gradient in the loss).
    Module names/creation order equal exps.dinomaly.official.Decoder, so init_trainable(seed) and the per-block
    optimizer groups (exps.nlcad.fast.block_optimizer) apply unchanged.
    """

    num_prefix = 0

    def __init__(self, embed_dim=DIM, num_heads=12, checkpoint=False):
        super().__init__()
        self.bottleneck = nn.ModuleList([O.bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
        self.decoder = nn.ModuleList([
            O.Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=O.LinearAttention2) for _ in range(8)])
        self.checkpoint = checkpoint

    def forward(self, f):
        require(f.ndim == 3 and tuple(f.shape[1:]) == (TOKENS, DIM), f"expected [B,{TOKENS},{DIM}], got {tuple(f.shape)}")
        x = f
        for blk in self.bottleneck:
            x = blk(x)
        de_list = []
        for blk in self.decoder:
            x = checkpoint(blk, x, use_reentrant=False) if self.checkpoint and self.training else blk(x)
            de_list.append(x)
        de = torch.stack(de_list, dim=1).mean(dim=1)
        en, de = f[:, self.num_prefix:, :], de[:, self.num_prefix:, :]
        side = int(math.sqrt(en.shape[1]))
        require(side * side == en.shape[1] == TOKENS and side == SIDE, "token grid is not 28x28")
        en = en.permute(0, 2, 1).reshape(en.shape[0], -1, side, side).contiguous()
        de = de.permute(0, 2, 1).reshape(de.shape[0], -1, side, side).contiguous()
        return [en], [de]


def new_decoder(seed=SEED, device="cuda", ckpt=True):
    torch.manual_seed(seed)
    dec = SingleStreamDecoder(checkpoint=ckpt).to(device)
    O.init_trainable(dec)
    return dec


def state_bytes(dec):
    return sum(t.numel() * t.element_size() for t in dec.state_dict().values())


# --- readout -----------------------------------------------------------------------------------------

_GAUSS = {}


def gauss(device):
    key = str(device)
    if key not in _GAUSS:
        _GAUSS[key] = O.get_gaussian_kernel(*O.RECIPE["gaussian"]).to(device)
    return _GAUSS[key]


def grid(en, de):
    """[B,28,28] 1 - cos (FP32)."""
    return 1 - F.cosine_similarity(en[0].float(), de[0].float())


@torch.no_grad()
def score(en, de):
    """Dinomaly evaluation chain at this model's input size: 28x28 1-cos -> 224 (bilinear, align_corners=True; the
    official cal_anomaly_maps upsamples to the input size) -> 256 -> Gaussian 5x5/sigma 4 -> top-1% mean image score.
    Returns ([B,256,256] map, [B] image score). Native upsampling: exps.dinomaly.readout.native."""
    a, _ = O.cal_anomaly_maps([e.float() for e in en], [d.float() for d in de], IMG)
    a = F.interpolate(a, size=EVAL_SIZE, mode="bilinear", align_corners=False)
    s = gauss(a.device)(a)
    flat = s.flatten(1)
    image = torch.sort(flat, dim=1, descending=True)[0][:, :int(flat.shape[1] * TOPK_RATIO)].mean(dim=1)
    return s[:, 0], image


# --- identities --------------------------------------------------------------------------------------


def identities():
    """R15 training identities (RH-01b PROTOCOL, lock-verified), PTAG held-out normals, full MVTec test cohort."""
    from exps.rh01b import run as RB
    from exps.cadfinal import common as K
    proto = RB.verify_lock()
    require(tuple(proto["stream"]) == STREAM, "stream order")
    train = {c: proto["train"][c] for c in STREAM}
    require(sum(len(v) for v in train.values()) == 3099, "R15 identity count")
    ids = K.benchmark_identities("mvtec")
    require(tuple(t["task"] for t in ids["tasks"]) == STREAM, "test stream order")
    test = {t["task"]: t["test"] for t in ids["tasks"]}
    tr = {r["relative_path"] for v in train.values() for r in v}
    held = {r["relative_path"] for v in proto["heldout"].values() for r in v}
    te = {r["relative_path"] for v in test.values() for r in v}
    require(all("/train/good/" in p for p in tr | held), "training/held-out identity outside train/good")
    require(not (tr & held) and not (tr & te) and not (held & te), "identity overlap")
    return {"train": train, "heldout": proto["heldout"], "test": test, "n_train": {c: len(train[c]) for c in STREAM}}


# --- caches ------------------------------------------------------------------------------------------

_VERIFIED = set()


def _load(name, mmap=True):
    path = CACHE / f"{name}.pt"
    if name not in _VERIFIED:
        rec = D.read(OUT / "CACHE.json")["files"][name]
        require(sha(path) == rec["sha256"], f"cache changed: {path}")
        _VERIFIED.add(name)
    return torch.load(path, weights_only=True, mmap=mmap)


def train_features(c):
    return _load(f"train_{c}")["features"]


def heldout_features(c):
    return _load(f"heldout_{c}")["features"]


def test_features(c):
    d = _load(f"test_{c}")
    return d["features"], d["paths"]


def load_bank():
    """{(category, index, variant): (features [784,768], coverage [28,28])}, hosts; SHA-verified."""
    man = D.read(OUT / "BANK.json")
    items = {}
    for c in STREAM:
        path = CACHE / f"bank_Q2_{c}.pt"
        require(sha(path) == man["files"][c]["sha256"], f"bank changed: {path}")
        d = torch.load(path, weights_only=False, mmap=True)
        for j, (cat, i, v) in enumerate(d["keys"]):
            items[(cat, int(i), int(v))] = (d["features"][j], d["coverage"][j])
    hosts = {c: {k: [int(x) for x in v] for k, v in h.items()} for c, h in man["hosts"].items()}
    return items, hosts
