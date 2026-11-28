"""Frozen RIGEL-CAD encoders (no training, weights loaded strictly from local checkpoints).

* RigelDinoEncoder (default): DINOv2-reg ViT-B/14 (4 registers), 392 x 392 full-field bilinear input, ImageNet
  normalisation; token = mean of the raw outputs of blocks 6-9 (0-based), CLS and registers removed -> 28 x 28 x 768.
  Re-implements the official DINOv2 forward with Dinomaly's construction arguments (interpolate_antialias=False,
  interpolate_offset=0.1), identical to exps/dinomaly/encoder.py + exps/nar_b14_01/run.py::tokens.
* RigelViTB8Encoder (matched-backbone ablation): timm vit_base_patch8_224 (ImageNet-21k augreg weights), 224 x 224
  bilinear input, ImageNet normalisation, output of block 9 (1-based), CLS removed -> 28 x 28 x 768.

Both return unit-norm fp16 tokens [B, 784, 768] (RigelEncoder.tokens).
"""
from __future__ import annotations

import hashlib
import math
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from rigel_cad import ops
from rigel_cad.config import INPUT_SIZE, TOKEN_DIM, TOKENS_PER_IMAGE

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
REPO = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    'dinov2_vitb14_reg4': (REPO / 'checkpoints/dinov2/dinov2_vitb14_reg4_pretrain.pth',
                           '73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71'),
    'vit_b8_in21k': (REPO / 'checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth',
                     '755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415'),
}
DINO_PATCH = 14
DECODE_WORKERS = min(8, os.cpu_count() or 1)   # image decode threads (engineering only; tokens are unchanged)
DECODE_AHEAD = 3                               # batches decoded ahead of the backbone
DINO_TOKEN_BLOCKS = (6, 7, 8, 9)       # 0-based blocks averaged into the token


def file_sha256(path, buf=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(buf):
            h.update(b)
    return h.hexdigest()


def preprocess(path, size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """PIL RGB, direct square bilinear resize (full field of view), ToTensor, normalisation. -> [3, size, size]."""
    with Image.open(path) as im:
        im = im.convert('RGB').resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1)
    return (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)


# ====================================================================== DINOv2-reg ViT-B/14
class _Attention(nn.Module):
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


class _LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class _Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.attn, self.ls1 = nn.LayerNorm(dim, eps=1e-6), _Attention(dim, heads), _LayerScale(dim)
        self.norm2, self.mlp, self.ls2 = nn.LayerNorm(dim, eps=1e-6), _Mlp(dim), _LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class _PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, DINO_PATCH, DINO_PATCH)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _DinoV2Reg(nn.Module):
    """DINOv2 ViT-B/14 with 4 registers; parameter names match the official checkpoint (strict loading)."""
    num_prefix = 5

    def __init__(self, dim=768, heads=12):
        super().__init__()
        self.patch_embed = _PatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, 4, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 37 * 37, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))          # unused; kept for strict loading
        self.blocks = nn.ModuleList(_Block(dim, heads) for _ in range(12))
        self.norm = nn.LayerNorm(dim, eps=1e-6)                      # unused (raw block outputs are read)

    def interpolate_pos_encoding(self, w, h):
        pos = self.pos_embed.float()
        cls_pos, patch_pos = pos[:, 0], pos[:, 1:]
        dim, n = pos.shape[-1], patch_pos.shape[1]
        w0, h0 = w // DINO_PATCH + 0.1, h // DINO_PATCH + 0.1     # interpolate_offset = 0.1
        m = math.sqrt(n)
        patch_pos = F.interpolate(patch_pos.reshape(1, int(m), int(m), dim).permute(0, 3, 1, 2),
                                  scale_factor=(float(w0) / m, float(h0) / m), mode='bicubic', antialias=False)
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
    def block_mean(self, x, blocks=DINO_TOKEN_BLOCKS):
        """Mean of the raw outputs of `blocks` (all 789 tokens)."""
        x, out = self.prepare_tokens(x), {}
        for i, blk in enumerate(self.blocks[:blocks[-1] + 1]):
            x = blk(x)
            out[i] = x
        return torch.stack([out[i] for i in blocks], dim=1).mean(dim=1)


class RigelEncoder:
    """Common interface: .size (input side), .preprocess(path) -> [3, s, s], .tokens(batch) -> [B, 784, 768] fp16 unit."""
    name = 'base'
    size = INPUT_SIZE

    def preprocess(self, path) -> torch.Tensor:
        return preprocess(path, self.size)

    @torch.no_grad()
    def raw_tokens(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        f = self.raw_tokens(x)
        return ops.unit(f.reshape(-1, TOKEN_DIM)).half().reshape(len(x), TOKENS_PER_IMAGE, TOKEN_DIM)

    def preprocess_many(self, paths, workers: int = None) -> torch.Tensor:
        """torch.stack([preprocess(p) for p in paths]) with the decoding spread over `workers` threads (same values)."""
        paths = list(paths)
        workers = DECODE_WORKERS if workers is None else workers
        if workers <= 1 or len(paths) <= 1:
            return torch.stack([self.preprocess(p) for p in paths])
        with ThreadPoolExecutor(min(workers, len(paths))) as ex:
            return torch.stack(list(ex.map(self.preprocess, paths)))

    @torch.no_grad()
    def encode_paths(self, paths, batch: int = 8, workers: int = None) -> torch.Tensor:
        """Unit fp16 tokens [n, 784, 768] (CPU) of image files, in the given order.

        Images are decoded by `workers` threads (default DECODE_WORKERS), DECODE_AHEAD batches ahead of the backbone;
        every image goes through the same preprocess(), so the tokens are identical to a sequential loop
        (workers=1)."""
        paths = list(paths)
        workers = DECODE_WORKERS if workers is None else workers
        out = []
        if workers <= 1:
            for i in range(0, len(paths), batch):
                x = torch.stack([self.preprocess(p) for p in paths[i:i + batch]]).to(self.device)
                out.append(self.tokens(x).cpu())
        else:
            chunks = [paths[i:i + batch] for i in range(0, len(paths), batch)]
            with ThreadPoolExecutor(workers) as ex:
                pending = deque()
                nxt = 0
                for _ in range(len(chunks)):
                    while nxt < len(chunks) and len(pending) < DECODE_AHEAD:
                        pending.append([ex.submit(self.preprocess, p) for p in chunks[nxt]])
                        nxt += 1
                    x = torch.stack([f.result() for f in pending.popleft()]).to(self.device)
                    out.append(self.tokens(x).cpu())
        return torch.cat(out) if out else torch.empty(0, TOKENS_PER_IMAGE, TOKEN_DIM, dtype=torch.float16)


class RigelDinoEncoder(RigelEncoder):
    name = 'dinov2_vitb14_reg4'
    size = INPUT_SIZE

    def __init__(self, checkpoint=None, device='cuda', verify_hash=True):
        path, sha = CHECKPOINTS[self.name]
        path = Path(checkpoint) if checkpoint else path
        if verify_hash and file_sha256(path) != sha:
            raise RuntimeError(f'DINOv2 checkpoint hash mismatch: {path}')
        self.device = device
        self.model = _DinoV2Reg()
        self.model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        self.model = self.model.eval().requires_grad_(False).to(device)

    @torch.no_grad()
    def raw_tokens(self, x):
        g = self.model.block_mean(x)
        if tuple(g.shape[1:]) != (789, TOKEN_DIM):
            raise RuntimeError(f'DINOv2 token shape {tuple(g.shape)}')
        return g[:, _DinoV2Reg.num_prefix:, :].float().contiguous()


class RigelViTB8Encoder(RigelEncoder):
    name = 'vit_b8_in21k'
    size = 224
    layer = 9                                   # 1-based block index

    def __init__(self, checkpoint=None, device='cuda', verify_hash=True):
        import timm
        path, sha = CHECKPOINTS[self.name]
        path = Path(checkpoint) if checkpoint else path
        if verify_hash and file_sha256(path) != sha:
            raise RuntimeError(f'ViT-B/8 checkpoint hash mismatch: {path}')
        self.device = device
        self.model = timm.create_model('vit_base_patch8_224', pretrained=False, num_classes=0, img_size=224)
        sd = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(sd, dict):
            for key in ('state_dict', 'model', 'model_state_dict'):
                if key in sd and isinstance(sd[key], dict):
                    sd = sd[key]
                    break
        self.model.load_state_dict(sd, strict=True)
        self.model = self.model.to(device).eval().requires_grad_(False)
        self._cap = None
        self.model.blocks[self.layer - 1].register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        self._cap = out[0] if isinstance(out, (tuple, list)) else out

    @torch.no_grad()
    def raw_tokens(self, x):
        self._cap = None
        self.model(x.to(self.device))
        t, self._cap = self._cap, None
        if t is None or tuple(t.shape[1:]) != (785, TOKEN_DIM):
            raise RuntimeError('ViT-B/8 block output missing or wrong shape')
        return t[:, 1:, :].float()


def build_encoder(name: str = 'dinov2_vitb14_reg4', device: str = 'cuda', checkpoint=None) -> RigelEncoder:
    if name == 'dinov2_vitb14_reg4':
        return RigelDinoEncoder(checkpoint, device)
    if name == 'vit_b8_in21k':
        return RigelViTB8Encoder(checkpoint, device)
    raise KeyError(f'unknown backbone {name!r}')
