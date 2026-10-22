"""Per-vector affine UINT8 quantization for CSG-01 Arm C/D.

Registered quantizer:

    xmin = min_j x_j
    xmax = max_j x_j
    step = (xmax - xmin) / 255
    q_j  = clamp(rint((x_j - xmin) / step), 0, 255)      # rint = ties-to-even
    xhat_j = xmin + step * q_j

``xmin`` and ``step`` are stored as float32 and used in decode; constant vectors
use step=0, q=0, xhat=xmin.  The decoded buffer is temporary working memory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from . import contract as C1


def encode_support(support: Sequence[torch.Tensor]) -> Dict[str, Any]:
    """Encode concatenated FP32 support vectors; returns q, xmin, step, decoded."""
    X = torch.cat([x.to(torch.float32) for x in support], dim=0).numpy().astype(np.float32)
    n = X.shape[0]
    if X.shape[1] != C1.DIM:
        raise ValueError(f"expected d={C1.DIM}, got {X.shape[1]}")
    xmin = X.min(axis=1).astype(np.float32)
    xmax = X.max(axis=1).astype(np.float32)
    step64 = (xmax.astype(np.float64) - xmin.astype(np.float64)) / 255.0
    step = step64.astype(np.float32)
    q = np.zeros((n, C1.DIM), dtype=np.uint8)
    nonconstant = step > 0.0
    if np.any(nonconstant):
        z = (X[nonconstant].astype(np.float64) - xmin[nonconstant].astype(np.float64)[:, None]) \
            / step[nonconstant].astype(np.float64)[:, None]
        q[nonconstant] = np.clip(np.rint(z), 0.0, 255.0).astype(np.uint8)
    decoded64 = xmin.astype(np.float64)[:, None] + \
        step.astype(np.float64)[:, None] * q.astype(np.float64)
    decoded = decoded64.astype(np.float32)
    err = np.linalg.norm(X.astype(np.float64) - decoded.astype(np.float64), axis=1)
    bound = np.sqrt(C1.DIM) * step.astype(np.float64) / 2.0
    return {
        "q": q,
        "xmin": xmin,
        "step": step,
        "decoded": decoded,
        "error": err,
        "bound": bound,
        "constant_vectors": int((~nonconstant).sum()),
    }


def decode(q: np.ndarray, xmin: np.ndarray, step: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.uint8)
    xmin = np.asarray(xmin, dtype=np.float32)
    step = np.asarray(step, dtype=np.float32)
    out = xmin.astype(np.float64)[:, None] + step.astype(np.float64)[:, None] * q.astype(np.float64)
    return out.astype(np.float32)


def quantization_summary(encoded: Dict[str, Any], counts: np.ndarray) -> Dict[str, Any]:
    err = np.asarray(encoded["error"], dtype=np.float64)
    bound = np.asarray(encoded["bound"], dtype=np.float64)
    step = np.asarray(encoded["step"], dtype=np.float64)
    per_bin = []
    offset = 0
    for b in range(C1.BINS):
        k = int(counts[b])
        seg = err[offset:offset + k]
        seg_bound = bound[offset:offset + k]
        per_bin.append({
            "bin": int(b),
            "n": k,
            "delta_b": float(seg.max()) if k else 0.0,
            "delta_mean": float(seg.mean()) if k else 0.0,
            "delta_max_bound": float(seg_bound.max()) if k else 0.0,
            "violations": int(np.sum(seg > seg_bound + C1.QUANT_TOL)),
            "constant_vectors": int(np.sum(np.asarray(encoded["step"])[offset:offset + k] == 0.0)),
        })
        offset += k
    violations = int(np.sum(err > bound + C1.QUANT_TOL))
    return {
        "n_vectors": int(err.size),
        "error_mean": float(err.mean()),
        "error_median": float(np.median(err)),
        "error_p95": float(np.quantile(err, 0.95)),
        "error_max": float(err.max()),
        "bound_mean": float(bound.mean()),
        "bound_max": float(bound.max()),
        "step_mean": float(step.mean()),
        "step_max": float(step.max()),
        "constant_vectors": int(encoded["constant_vectors"]),
        "bound_violations": violations,
        "uint8_l2_bound_sqrtD_h_over_2": True,
        "tolerance": C1.QUANT_TOL,
        "per_bin": per_bin,
    }


def pack_quantized(encoded: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {
        "q": np.asarray(encoded["q"], dtype=np.uint8),
        "xmin": np.asarray(encoded["xmin"], dtype=np.float32),
        "step": np.asarray(encoded["step"], dtype=np.float32),
    }


def quantized_bin_support(state: Dict[str, np.ndarray]) -> List[torch.Tensor]:
    counts = np.asarray(state["counts"], dtype=np.int64)
    offsets = np.asarray(state["offsets"], dtype=np.int64)
    q = np.asarray(state["q"], dtype=np.uint8)
    xmin = np.asarray(state["xmin"], dtype=np.float32)
    step = np.asarray(state["step"], dtype=np.float32)
    out = []
    for b in range(C1.BINS):
        a, z = int(offsets[b]), int(offsets[b] + counts[b])
        decoded = decode(q[a:z], xmin[a:z], step[a:z])
        out.append(torch.as_tensor(decoded, dtype=torch.float32))
    return out


def delta_by_bin(encoded: Dict[str, Any], counts: np.ndarray) -> np.ndarray:
    err = np.asarray(encoded["error"], dtype=np.float64)
    out = np.zeros((C1.BINS,), dtype=np.float64)
    offset = 0
    for b in range(C1.BINS):
        k = int(counts[b])
        if k:
            out[b] = float(err[offset:offset + k].max())
        offset += k
    return out
