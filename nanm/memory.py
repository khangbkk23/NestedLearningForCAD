"""Boundary Exemplar Writer (fast level): greedy farthest-first k-centre on a seeded Gaussian projection.

Oracle: exps/nanm_r0/audit.py::unit, ::greedy_coreset; exps/nanm_r1/run.py::n_of; exps/bench_g3/run.py::fit_tokens.
Runs once per task boundary over the whole buffered FIT pool (no per-image update).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def unit(x):
    return F.normalize(x.float(), dim=-1)


def quota(ratio, n):
    return math.ceil(ratio * n)


@torch.no_grad()
def greedy_kcenter(x, n, seed, dim):
    """x [N, D] fp16 on device. Generator re-seeded per call (same P and first-centre draw for every task)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    P = (torch.randn(x.shape[1], dim, generator=g) / math.sqrt(dim)).to(x.device, torch.float32)
    z = x.float() @ P
    idx = torch.empty(n, dtype=torch.long)
    cur = int(torch.randint(len(z), (1,), generator=g))
    mind = torch.full((len(z),), float('inf'), device=x.device)
    for i in range(n):
        idx[i] = cur
        mind = torch.minimum(mind, ((z - z[cur]) ** 2).sum(1))
        cur = int(mind.argmax())
    return idx


class BoundaryExemplarWriter:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device

    @torch.no_grad()
    def prepare(self, tokens):
        """[n_img, 784, D] fp32 (FIT order) -> unit fp16 rows on device, normalised in chunks of 64 images on device."""
        c, D = self.cfg.fit_chunk_images, self.cfg.dim
        return torch.cat([unit(tokens[i:i + c].reshape(-1, D).to(self.device)).half()
                          for i in range(0, len(tokens), c)])

    def write(self, fit16, k):
        return greedy_kcenter(fit16, k, self.cfg.seed, self.cfg.proj_dim)
