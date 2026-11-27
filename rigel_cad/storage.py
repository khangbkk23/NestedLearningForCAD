"""Storage formats of the RIGEL-CAD persistent state.

Two formats are supported (``RigelConfig.storage``):

``'fp16'`` (default, RIGEL-CAD v1)
    keys stored as fp16 unit vectors (1,536 B per key); residual moment S accumulated in fp64 and charged as a full
    fp32 [768, 768] matrix plus an fp64 count (2,359,304 B).

``'int4_triangular'`` (compact memory)
    keys stored as signed int4 with per-key absmax scaling: scale = fp16(max|x| / 7), q = clamp(round(x / scale), -7, 7),
    two codes per byte (384 B) + the fp16 scale (2 B) = 386 B per key. The symmetric residual moment is stored as its
    fp32 upper triangle (295,296 values, row-major, diagonal included) plus an fp64 count (1,181,192 B). The keys are
    quantised when written, consolidated and scored in their dequantised form, and the triangle is re-rounded to fp32
    after every task, so the format holds throughout the continual stream (not only in a final checkpoint).

The functions below are a verbatim port of the implementation that produced the benchmark results of the compact
format (see rigel_cad/README.md, "Provenance"); rigel_cad/tests/test_unit.py checks their invariants and
rigel_cad/tests/test_equivalence.py checks bitwise agreement with the research implementation.
"""
from __future__ import annotations

import torch

from rigel_cad.config import TOKEN_DIM

D = TOKEN_DIM
N_TRI = D * (D + 1) // 2                                # 295,296 upper-triangle entries


# ---------------------------------------------------------------------------------------------- int4 keys
@torch.no_grad()
def quantize4(x: torch.Tensor):
    """Symmetric per-row absmax int4. Returns (q [N, D] int8 with values in [-7, 7], scale [N] fp16)."""
    x = x.float()
    scale = (x.abs().amax(1) / 7.0).clamp_min(1e-12).half()
    q = torch.round(x / scale.float()[:, None]).clamp_(-7, 7).to(torch.int8)
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 keys as scored: q * scale (row-wise)."""
    return q.float() * scale.float()[:, None]


def pack4(q: torch.Tensor) -> torch.Tensor:
    """[N, D] int8 codes in [-7, 7] -> [N, D / 2] uint8 (two nibbles per byte, offset 8; even column in the low
    nibble)."""
    u = (q.to(torch.int16) + 8).to(torch.uint8)
    return u[:, 0::2] | (u[:, 1::2] << 4)


def unpack4(p: torch.Tensor) -> torch.Tensor:
    """Inverse of pack4: [N, D / 2] uint8 -> [N, D] int8."""
    lo = (p & 0x0F).to(torch.int16) - 8
    hi = (p >> 4).to(torch.int16) - 8
    return torch.stack([lo, hi], dim=2).reshape(len(p), -1).to(torch.int8)


# ---------------------------------------------------------------------------------------------- triangular S
_IU = {}


def _iu(device) -> torch.Tensor:
    dev = torch.device(device)
    if dev not in _IU:
        _IU[dev] = torch.triu_indices(D, D, device=dev)
    return _IU[dev]


def tri_pack(S: torch.Tensor) -> torch.Tensor:
    """fp32 upper triangle (row-major, diagonal included) of a symmetric [D, D] matrix."""
    i, j = _iu(S.device)
    return S[i, j].float()


def tri_unpack(t: torch.Tensor) -> torch.Tensor:
    """fp64 symmetric [D, D] matrix from an fp32 upper triangle."""
    i, j = _iu(t.device)
    S = torch.zeros(D, D, dtype=torch.float64, device=t.device)
    S[i, j] = t.double()
    S[j, i] = t.double()
    return S
