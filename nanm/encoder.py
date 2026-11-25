"""Frozen Patch Encoder: DINOv2-reg ViT-B/14, mean of raw blocks 6-9, prefix dropped, fp32.

Lines below the marker are a VERBATIM copy of exps/dinomaly/encoder.py (sha 46b03801…, BENCH-G3 locked source),
so the forward pass is the same code; parity is tested bitwise against the oracle. Nothing is learned.
"""
# ---- verbatim copy of exps/dinomaly/encoder.py (from its first import line) ----
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ARCH = {"base": (768, 12), "small": (384, 6)}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SIZE = 392
PATCH = 14
GROUPS = ((2, 3, 4, 5), (6, 7, 8, 9))


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.attn, self.ls1 = nn.LayerNorm(dim, eps=1e-6), Attention(dim, heads), LayerScale(dim)
        self.norm2, self.mlp, self.ls2 = nn.LayerNorm(dim, eps=1e-6), Mlp(dim), LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, PATCH, PATCH)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class DinoV2Reg(nn.Module):
    num_prefix = 5

    def __init__(self, arch="base"):
        super().__init__()
        dim, heads = ARCH[arch]
        self.patch_embed = PatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, 4, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 37 * 37, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))               # unused; kept for strict loading
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(12))
        self.norm = nn.LayerNorm(dim, eps=1e-6)                           # unused (raw block outputs)

    def interpolate_pos_encoding(self, w, h):
        pos = self.pos_embed.float()
        cls_pos, patch_pos = pos[:, 0], pos[:, 1:]
        dim, n = pos.shape[-1], patch_pos.shape[1]
        w0, h0 = w // PATCH + 0.1, h // PATCH + 0.1                     # interpolate_offset=0.1
        m = math.sqrt(n)
        patch_pos = F.interpolate(patch_pos.reshape(1, int(m), int(m), dim).permute(0, 3, 1, 2),
                                  scale_factor=(float(w0) / m, float(h0) / m), mode="bicubic", antialias=False)
        assert int(w0) == patch_pos.shape[-2] and int(h0) == patch_pos.shape[-1]
        patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((cls_pos.unsqueeze(0), patch_pos), dim=1)

    def prepare_tokens(self, x):
        _, _, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(w, h).to(x.dtype)
        return torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)

    @torch.no_grad()
    def groups(self, x):
        """Group means of raw block outputs over target layers 2-5 and 6-9, all 789 tokens kept."""
        x, out = self.prepare_tokens(x), {}
        for i, blk in enumerate(self.blocks[:GROUPS[1][-1] + 1]):
            x = blk(x)
            out[i] = x
        return torch.stack([torch.stack([out[i] for i in g], dim=1).mean(dim=1) for g in GROUPS], dim=1)


def load(path, arch="base", device="cuda"):
    model = DinoV2Reg(arch)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.eval().requires_grad_(False).to(device)


def preprocess(path):
    """RGB, PIL bilinear resize to 392x392 (full field of view), ToTensor, ImageNet normalisation."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1)
    return (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
# ---- end of verbatim copy ----

import hashlib
from pathlib import Path

GROUP = 1                      # GROUPS[1] = blocks (6, 7, 8, 9)


def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


class FrozenPatchEncoder:
    """Images -> [N, 784, 768] fp32 tokens. Batch composition is part of the numerical contract (oracle: 8, in order)."""

    def __init__(self, model, sha256, device):
        self.model, self.sha256, self.device = model, sha256, device

    @classmethod
    def load(cls, path, sha256, device='cuda', arch='base'):
        if _sha256(path) != sha256:
            raise ValueError(f'encoder checkpoint sha256 mismatch: {path}')
        return cls(load(path, arch, device), sha256, device)

    @torch.no_grad()
    def tokens_from_batch(self, x):
        g = self.model.groups(x)
        if tuple(g.shape[1:]) != (2, 789, 768):
            raise RuntimeError('DINOv2 group shape')
        return g[:, GROUP, self.model.num_prefix:, :].float().contiguous()

    @torch.no_grad()
    def tokens(self, paths, batch=8):
        paths = [Path(p) for p in paths]
        feats = torch.empty(len(paths), 784, 768, dtype=torch.float32)
        for i in range(0, len(paths), batch):
            x = torch.stack([preprocess(p) for p in paths[i:i + batch]]).to(self.device)
            feats[i:i + len(x)] = self.tokens_from_batch(x).cpu()
        if not bool(torch.isfinite(feats).all()):
            raise RuntimeError('feature finiteness')
        return feats
