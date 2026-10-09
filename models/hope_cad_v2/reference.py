"""Slow, explicit per-head oracle for the v2 visual recurrence.

This deliberately does NOT use the optimized shared-gain/rank-one update or its
step-size helper. Each memory is updated with M_previous @ A + M_boundary @ B.
Use tiny FP64 fixtures; this is a correctness oracle, not a benchmark backend.
"""

import math

import torch
from torch.nn import functional as F

from .scan_geometry import chunk_spans
from .visual_memory import ScanResult, VisualMemoryState, validate_scan_inputs


def scan_reference(z, q, initial: VisualMemoryState, chunk_size: int) -> ScanResult:
    validate_scan_inputs(z, q, initial, chunk_size)
    if z.device.type != "cpu" or z.dtype != torch.float64:
        raise ValueError("the independent oracle requires CPU float64")
    n, d = z.shape[-2:]
    leading = z.shape[:-2]
    zs, qs = z.reshape(-1, n, d), q.reshape(-1, n, d)
    memories = [matrix.reshape(-1, matrix.shape[-2], d) for matrix in initial.tensors()]
    all_outputs, all_states = [], []
    identity = torch.eye(d, dtype=z.dtype, device=z.device)
    for batch in range(zs.shape[0]):
        current = [matrix[batch] for matrix in memories]
        outputs = []
        for start, end in chunk_spans(n, chunk_size):
            boundary = current
            for t in range(start, end):
                token = zs[batch, t]
                key_raw = boundary[1] @ token
                key = key_raw / torch.sqrt(torch.sum(key_raw * key_raw) + 1e-6)
                value = boundary[2] @ token
                raw_eta = 0.025 * F.softplus((boundary[3] @ token).squeeze(0))
                alpha = torch.sigmoid((boundary[4] @ token).squeeze(0) + math.log(9.0))
                delta = key - value
                radius = torch.sqrt(torch.dot(delta, delta) + 1e-12) + 1e-6
                injection = (1 - alpha) * (1 - 1e-3) / radius
                safe = torch.where(injection > 0, injection, torch.ones_like(injection))
                soft = -injection * torch.expm1(-raw_eta / safe)
                spectral = 2 * alpha * (1 - 1e-6) / torch.dot(key, key).clamp_min(1e-20)
                rounded = torch.nextafter(spectral.detach(), torch.zeros_like(spectral))
                spectral = spectral + (rounded - spectral).detach()
                eta = torch.minimum(soft, spectral)
                a = alpha * identity - eta * torch.outer(key, key)
                b = -eta * torch.outer(delta, key)
                outputs.append(boundary[0] @ qs[batch, t])
                current = [previous @ a + reset @ b for previous, reset in zip(current, boundary)]
        all_outputs.append(torch.stack(outputs))
        all_states.append(current)
    output = torch.stack(all_outputs).reshape(*leading, n, d)
    final = VisualMemoryState(*(
        torch.stack([state[i] for state in all_states]).reshape(*leading, matrix.shape[-2], d)
        for i, matrix in enumerate(initial.tensors())
    ))
    return ScanResult(output, final)
