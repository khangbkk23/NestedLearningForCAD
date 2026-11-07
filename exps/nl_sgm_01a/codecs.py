"""Support codecs for NL-SGM-01A Stage B: FP32, UINT8 (canonical CSG-01 / CALM-01 rule) and PQ (FP16 codebook).

PQ: M = 96 sub-spaces of 8 dims, K = 256 centroids, FP16 codebook (393,216 B, Amendment 01 item 2).
Fit: k-means++ initialisation (seed 20261106 + sub-space), 25 Lloyd iterations in FP32, empty cluster -> the
point farthest from its assigned centroid (lowest index on ties). Encoding uses the stored FP16 codebook (as FP32
values) and an exact direct-difference argmin with lowest-index ties, so codes are a deterministic function of
(row, codebook).
"""
from __future__ import annotations

import torch

from exps.calm01.storage import quantize_rows
from exps.task_p.p0_normal_support import require

M, K, SUB = 96, 256, 8
CODEBOOK_BYTES = M * K * SUB * 2
U8_ROW_BYTES = 768 + 8
PQ_ROW_BYTES = M


# ------------------------------------------------------------------ UINT8 (canonical rule, re-used)

def u8_encode(x):
    e = quantize_rows(x)
    return {"q": e["q"], "xmin": e["xmin"], "step": e["step"]}, e["decoded"]


def u8_decode(state):
    return (state["xmin"].double()[:, None] + state["step"].double()[:, None] * state["q"].double()).float()


# ------------------------------------------------------------------ PQ

def _assign_gemm(Xs, C):
    """Lloyd assignment for all sub-spaces: Xs [M, n, 8], C [M, K, 8] -> [M, n] (FP32 GEMM form)."""
    d = (C.square().sum(-1))[:, None, :] - 2 * torch.bmm(Xs, C.transpose(1, 2))
    return d.argmin(-1)


def _kmeanspp(Xs, seed):
    """Deterministic k-means++ per sub-space. Xs [M, n, 8] -> [M, K, 8]."""
    Mm, n, _ = Xs.shape
    out = torch.empty(Mm, K, SUB)
    for m in range(Mm):
        g = torch.Generator().manual_seed(seed + m)
        X = Xs[m]
        first = int(torch.randint(n, (1,), generator=g))
        C = [X[first]]
        d2 = (X - X[first]).square().sum(1)
        for _ in range(1, K):
            tot = float(d2.sum())
            if tot <= 0:
                j = int(torch.randint(n, (1,), generator=g))
            else:
                j = int(torch.multinomial(d2.double() / tot, 1, generator=g))
            C.append(X[j])
            d2 = torch.minimum(d2, (X - X[j]).square().sum(1))
        out[m] = torch.stack(C)
    return out


def fit(sample, seed=20261106, iters=25, chunk=8192):
    """Fit a PQ codebook on FP32 rows [n, 768]; returns the FP16 codebook [M, K, 8]."""
    require(sample.ndim == 2 and sample.shape[1] == M * SUB and len(sample) >= K, "bad PQ fit sample")
    Xs = sample.float().reshape(len(sample), M, SUB).transpose(0, 1).contiguous()     # [M, n, 8]
    C = _kmeanspp(Xs, seed)
    n = Xs.shape[1]
    for _ in range(iters):
        a = torch.cat([_assign_gemm(Xs[:, s:s + chunk], C) for s in range(0, n, chunk)], 1)   # [M, n]
        S = torch.zeros(M, K, SUB); cnt = torch.zeros(M, K)
        S.scatter_add_(1, a[..., None].expand(-1, -1, SUB), Xs)
        cnt.scatter_add_(1, a, torch.ones(M, n))
        newC = S / cnt.clamp_min(1)[..., None]
        empty = cnt == 0
        if bool(empty.any()):
            for m in empty.any(1).nonzero().flatten().tolist():
                resid = (Xs[m] - C[m][a[m]]).square().sum(1)
                for k in empty[m].nonzero().flatten().tolist():
                    j = int(resid.argmax())                      # first maximum = lowest index
                    newC[m, k] = Xs[m, j]; resid[j] = -1.0
        C = newC
    return C.half()


def encode(x, codebook, chunk=4096):
    """Exact direct-difference argmin per sub-space, lowest index on ties. x [n, 768] -> uint8 [n, M]."""
    C = codebook.float()                                          # [M, K, 8]
    out = torch.empty(len(x), M, dtype=torch.uint8)
    for s in range(0, len(x), chunk):
        X = x[s:s + chunk].float().reshape(-1, M, SUB)            # [b, M, 8]
        d = (X[:, :, None, :] - C[None]).square().sum(-1)        # [b, M, K]
        out[s:s + chunk] = d.argmin(-1).to(torch.uint8)           # torch argmin returns the first minimum
    return out


def decode(codes, codebook):
    C = codebook.float()
    idx = codes.long()                                            # [n, M]
    return C[torch.arange(M)[None], idx].reshape(len(codes), M * SUB)


def pq_capacity(budget, n_codebooks, owner_bytes=1, other_bytes=0):
    return int((budget - n_codebooks * CODEBOOK_BYTES - other_bytes) // (PQ_ROW_BYTES + owner_bytes))


def u8_capacity(budget, owner_bytes=1, other_bytes=0):
    return int((budget - other_bytes) // (U8_ROW_BYTES + owner_bytes))


# ------------------------------------------------------------------ transport (B6)

def transport(codes_old, cb_old, cb_new):
    """Re-encode stored rows under a new codebook from their *decoded* values (no FP32 source access).

    Returns new codes and the per-row decomposition of the squared step drift
    ||dec_new(code_new) - dec_old(code_old)||^2 = reassign + move, where, per sub-space, ``move`` collects the
    sub-spaces whose code index is unchanged (pure centroid movement C_new[k] - C_old[k]) and ``reassign`` those
    whose index changed. Also the frozen-code counterfactual dec_new(code_old) (pure codebook drift).
    """
    old = decode(codes_old, cb_old)
    new_codes = encode(old, cb_new)
    new = decode(new_codes, cb_new)
    frozen = decode(codes_old, cb_new)
    diff = (new - old).reshape(-1, M, SUB).double().square().sum(-1)        # [n, M]
    same = new_codes == codes_old
    return new_codes, {
        "step_sq": diff.sum(1),
        "move_sq": (diff * same).sum(1),
        "reassign_sq": (diff * ~same).sum(1),
        "frozen_code_drift": (frozen - old).double().norm(dim=1),
        "reassigned_subspaces": (~same).sum(1),
    }
