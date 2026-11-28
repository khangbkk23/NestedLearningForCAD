"""Numerical primitives of RIGEL-CAD.

Each function reproduces, operation for operation, the frozen research implementation it replaces (named in the
docstring), so that RIGEL-CAD results are bitwise identical on the same hardware (checked by
rigel_cad/tests/test_equivalence.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from rigel_cad.config import CORESET_RATIO, EIG_FLOOR, PROJ_DIM, QUERY_CHUNK, RESIDUAL_CHUNK, SHRINK, TOKEN_DIM


def unit(x: torch.Tensor) -> torch.Tensor:
    """Per-token L2 normalisation in fp32 (callers cast to fp16 for storage)."""
    return F.normalize(x.float(), dim=-1)


def coreset_size(n_tokens: int, ratio: float = CORESET_RATIO) -> int:
    """ceil(ratio * n): number of keys a task contributes before consolidation."""
    return math.ceil(ratio * n_tokens)


KCENTER_CHUNK = 8192            # rows per distance block in the CUDA path (engineering only; results are unchanged)


@torch.no_grad()
def greedy_kcenter(x: torch.Tensor, n: int, seed: int, dim: int = PROJ_DIM) -> torch.Tensor:
    """Greedy k-centre (farthest-point) selection on a seeded Gaussian projection.

    x: [N, D] tokens on any device (fp16 or fp32). Returns n row indices (CPU long) in selection order, so every
    prefix is itself a valid smaller selection. Mirrors exps/nanm_r0/audit.py::greedy_coreset.

    On CUDA the loop runs as a captured CUDA graph over row blocks (no host synchronisation per step, block
    temporaries stay in L2). Every distance is the same elementwise computation and the same row reduction as in
    greedy_kcenter_reference, so the selected indices are identical (rigel_cad/tests/test_unit.py,
    exps/rigel_eng_01/check.py).
    """
    if x.device.type != 'cuda' or n < 1:
        return greedy_kcenter_reference(x, n, seed, dim)
    g = torch.Generator(device='cpu').manual_seed(seed)
    P = (torch.randn(x.shape[1], dim, generator=g) / math.sqrt(dim)).to(x.device, torch.float32)
    z = x.float() @ P
    N = len(z)
    cur = torch.randint(N, (1,), generator=g).to(x.device)
    idx = torch.zeros(n, dtype=torch.long, device=x.device)
    pos = torch.arange(n, device=x.device)
    it = torch.zeros(1, dtype=torch.long, device=x.device)
    mind = torch.full((N,), float('inf'), device=x.device)
    blocks = [(a, min(a + KCENTER_CHUNK, N)) for a in range(0, N, KCENTER_CHUNK)]

    def step():
        idx.add_((pos == it).long() * cur)            # idx[it] = cur without an index-put kernel
        zc = z.index_select(0, cur)
        for a, b in blocks:
            torch.minimum(mind[a:b], ((z[a:b] - zc) ** 2).sum(1), out=mind[a:b])
        it.add_(1)
        cur.copy_(mind.argmax().view(1))

    # warm-up on a side stream (required before capture), then restore the loop state exactly
    keep = (cur.clone(), mind.clone(), it.clone())
    side = torch.cuda.Stream(device=x.device)
    side.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream(x.device).wait_stream(side)
    cur.copy_(keep[0])
    mind.copy_(keep[1])
    it.copy_(keep[2])
    idx.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(n):
        graph.replay()
    out = idx.cpu()
    del graph
    return out


@torch.no_grad()
def greedy_kcenter_reference(x: torch.Tensor, n: int, seed: int, dim: int = PROJ_DIM) -> torch.Tensor:
    """Plain loop of greedy_kcenter (RIGEL-CAD 1.0/1.1 code; used on CPU and as the test oracle)."""
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


def sqdist(q: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """fp32 squared Euclidean distances [len(q), len(m)], clamped at 0 (exps/nanm_r2/run.py::sqdist)."""
    return (q.pow(2).sum(1, keepdim=True) + m.pow(2).sum(1)[None] - 2 * q @ m.T).clamp_min(0)


@torch.no_grad()
def residual_moment(fit32: torch.Tensor, member_idx: torch.Tensor, chunk: int = RESIDUAL_CHUNK):
    """Uncentred second moment of the residuals r = x - nearest key, over the task tokens that are not keys.

    fit32: [N, D] fp32 task tokens; member_idx: indices of the task's keys in fit32 (same device).
    Returns (S [D, D] fp64 on CPU, count). Mirrors exps/nanm_r2b/run.py::residual_stats (mem_idx branch).
    """
    n = len(fit32)
    S = torch.zeros(TOKEN_DIM, TOKEN_DIM, dtype=torch.float64, device=fit32.device)
    cnt = 0
    member = torch.zeros(n, dtype=torch.bool, device=fit32.device)
    member[member_idx] = True
    mem = fit32[member_idx]
    for i in range(0, n, chunk):
        x = fit32[i:i + chunk]
        x = x[~member[i:i + chunk]]
        j = sqdist(x, mem).argmin(1)
        r = x - mem[j]
        S += (r.T @ r).double()
        cnt += len(r)
    return S.cpu(), cnt


def shrink(C: torch.Tensor, alpha: float = SHRINK) -> torch.Tensor:
    """Linear shrinkage towards the scaled identity (exps/nanm_r2/run.py::shrink)."""
    return (1 - alpha) * C + alpha * (torch.trace(C) / C.shape[0]) * torch.eye(C.shape[0], dtype=C.dtype)


def inv_sqrt_psd(S: torch.Tensor, floor: float = EIG_FLOOR):
    """S^(-1/2) via fp64 eigendecomposition. Returns (W fp64, condition number) (exps/nanm_r2/run.py)."""
    w, V = torch.linalg.eigh(S.double())
    if not float(w.min()) > floor:
        raise RuntimeError('metric not positive definite')
    return (V * w.rsqrt()) @ V.T, float(w.max() / w.min())


def whitener(S_sum: torch.Tensor, n: int):
    """Slow-level read-out transform W = shrink(S / n)^(-1/2) (exps/nanm_r2b/run.py::whiten). Returns (W fp64, cond)."""
    return inv_sqrt_psd(shrink(S_sum / n, SHRINK))


@torch.no_grad()
def energy_whitened(q16: torch.Tensor, keys: torch.Tensor, W: torch.Tensor, chunk: int = QUERY_CHUNK,
                    mw: torch.Tensor = None) -> torch.Tensor:
    """Patch energy 0.5 * min_m ||q W - m W||^2 (fp32), returned on CPU.

    q16: [Nq, D] fp16 unit tokens (any device); keys: [M, D] fp32; W: [D, D] fp32 on the keys' device.
    mw: optional precomputed keys @ W (same device), e.g. cached by RigelCAD.prepare; identical values.
    Mirrors nanm/g3.py::score and exps/bench_g3/run.py::whitened.
    """
    mw = keys @ W if mw is None else mw
    out = torch.empty(len(q16))
    for i in range(0, len(q16), chunk):
        out[i:i + chunk] = (0.5 * sqdist(q16[i:i + chunk].to(keys.device).float() @ W, mw).min(1).values).cpu()
    return out


@torch.no_grad()
def energy_euclidean(q16: torch.Tensor, keys: torch.Tensor, chunk: int = QUERY_CHUNK) -> torch.Tensor:
    """0.5 * min_m ||q - m||^2 (fp32) for the fast-keys-only ablation (exps/bench_g3/run.py::euclid)."""
    out = torch.empty(len(q16))
    for i in range(0, len(q16), chunk):
        out[i:i + chunk] = (0.5 * sqdist(q16[i:i + chunk].to(keys.device).float(), keys).min(1).values).cpu()
    return out
