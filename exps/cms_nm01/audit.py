# exps/cms_nm01/audit.py
"""Normal-only projection, storage and isolated CPU resource diagnostics."""

from __future__ import annotations

import json
import math
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .cms_nd import CMSNormalDensityMemory, FixedProjection


def projection_audit(probes: dict[str, torch.Tensor]) -> dict:
    projection = FixedProjection()
    result = {"digest": projection.digest(), "orthonormal_error": projection.orthonormal_error(), "categories": {}}
    generator = torch.Generator().manual_seed(2402)
    for category, images in probes.items():
        flat = images.reshape(-1, 768)
        indices = torch.randperm(len(flat), generator=generator)[:2048]
        x = flat[indices]
        z = projection(x)
        xc = x - x.mean(0)
        zc = z - z.mean(0)
        eigenvalues = torch.linalg.eigvalsh(xc.T @ xc).clamp_min(0)
        normalized = eigenvalues / eigenvalues.sum()
        full_effective_rank = torch.exp(-(normalized * normalized.clamp_min(1e-30).log()).sum())
        projected_energy = zc.square().sum() / xc.square().sum()
        raw_energy = z.square().sum() / x.square().sum()
        pairs = torch.randint(0, len(x), (1024, 2), generator=generator)
        valid = pairs[:, 0] != pairs[:, 1]
        pairs = pairs[valid]
        dx = torch.linalg.vector_norm(x[pairs[:, 0]] - x[pairs[:, 1]], dim=-1)
        dz = math.sqrt(12) * torch.linalg.vector_norm(z[pairs[:, 0]] - z[pairs[:, 1]], dim=-1)
        relative = (dz / dx.clamp_min(1e-12) - 1).abs()
        full_distance = torch.cdist(x[:256], x[:256]).fill_diagonal_(float("inf"))
        projected_distance = torch.cdist(z[:256], z[:256]).fill_diagonal_(float("inf"))
        recall = (full_distance.argmin(1) == projected_distance.argmin(1)).float().mean()
        result["categories"][category] = {
            "sampled_normal_patches": len(x), "original_effective_rank": float(full_effective_rank),
            "centered_energy_retained_fraction": float(projected_energy),
            "raw_energy_retained_fraction": float(raw_energy),
            "distance_relative_error_q50": float(torch.quantile(relative, 0.5)),
            "distance_relative_error_q95": float(torch.quantile(relative, 0.95)),
            "distance_pearson": float(torch.corrcoef(torch.stack((dx, dz)))[0, 1]),
            "nearest_neighbor_recall_at_1_of_256": float(recall),
        }
    return result


def storage_breakdown(model: torch.nn.Module) -> dict:
    groups: dict[str, int] = {}
    for name, value in model.state_dict().items():
        if name.startswith("projection."):
            group = "projection"
        elif ".flow." in name:
            group = "flow_parameters"
        elif ".anchor_" in name:
            group = "anchors"
        elif ".grad_accum_" in name:
            group = "accumulated_gradients"
        else:
            group = "counters_or_gaussian_statistics"
        groups[group] = groups.get(group, 0) + value.numel() * value.element_size()
    return {"groups": groups, "tensor_total_bytes": sum(groups.values())}


def isolated_resource_audit(model: torch.nn.Module, image: torch.Tensor) -> dict:
    """Use a clone for update profiling; no experimental checkpoint is changed.

    Memory is the high-water net live tensor-allocation increase recorded by
    PyTorch CPU allocation events within the operation, excluding preexisting
    model/input tensors. It is not the Python allocator or process RSS.
    """
    def operation(update: bool):
        if update:
            if isinstance(model, CMSNormalDensityMemory):
                model.commit_image(image)
            else:
                model.update_image(image)
        else:
            return model.score_patches(image)

    latencies = []
    operation(False)
    for _ in range(30):
        started = time.perf_counter()
        operation(False)
        latencies.append(time.perf_counter() - started)
    peaks = {}
    for update in (False, True):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profiler:
            returned = operation(update)
        with tempfile.TemporaryDirectory(prefix="cms_nm01_profile_") as temporary:
            path = Path(temporary) / "trace.json"
            profiler.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
        events = [event for event in trace["traceEvents"] if event.get("name") == "[memory]" and "Bytes" in event.get("args", {})]
        events.sort(key=lambda event: event["ts"])
        live = peak = 0
        for event in events:
            live += int(event["args"]["Bytes"])
            peak = max(peak, live)
        peaks["update" if update else "inference"] = {"peak_extra_live_tensor_bytes": peak, "net_bytes_at_end": live, "allocation_events": len(events)}
        del returned
    return {
        "cpu": platform.processor(), "torch_threads": torch.get_num_threads(),
        "single_image_inference_ms_median": float(np.median(latencies) * 1000),
        "single_image_inference_ms_p95": float(np.quantile(latencies, 0.95) * 1000),
        "extra_tensor_memory": peaks,
        "persistent": storage_breakdown(model),
    }
