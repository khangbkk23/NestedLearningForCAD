"""Opt-in elapsed-time and peak-memory profiler for pipeline stages."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import time
from typing import Dict, Iterator

import torch

try:
    import psutil
except ImportError:  # Optional: CUDA profiling remains available without psutil.
    psutil = None


class RuntimeProfiler:
    def __init__(self, enabled: bool = False, device: str | torch.device = "cpu") -> None:
        self.enabled = bool(enabled)
        self.device = torch.device(device)
        self._stats: Dict[str, dict] = defaultdict(
            lambda: {
                "calls": 0,
                "elapsed_seconds": 0.0,
                "max_cuda_allocated_bytes": 0,
                "max_cuda_incremental_peak_bytes": 0,
                "max_cuda_reserved_bytes": 0,
                "max_observed_host_rss_bytes": 0,
                "max_host_rss_increase_bytes": 0,
            }
        )
        self._process = psutil.Process() if psutil is not None else None

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        allocated_before = 0
        host_rss_before = 0
        if self._process is not None:
            host_rss_before = int(self._process.memory_info().rss)
        if use_cuda:
            torch.cuda.synchronize(self.device)
            allocated_before = int(torch.cuda.memory_allocated(self.device))
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            if use_cuda:
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
            stats = self._stats[str(stage)]
            stats["calls"] += 1
            stats["elapsed_seconds"] += elapsed
            if self._process is not None:
                host_rss_after = int(self._process.memory_info().rss)
                stats["max_observed_host_rss_bytes"] = max(
                    stats["max_observed_host_rss_bytes"], host_rss_before, host_rss_after
                )
                stats["max_host_rss_increase_bytes"] = max(
                    stats["max_host_rss_increase_bytes"], host_rss_after - host_rss_before
                )
            if use_cuda:
                peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
                peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
                stats["max_cuda_allocated_bytes"] = max(
                    stats["max_cuda_allocated_bytes"], peak_allocated
                )
                stats["max_cuda_incremental_peak_bytes"] = max(
                    stats["max_cuda_incremental_peak_bytes"],
                    max(0, peak_allocated - allocated_before),
                )
                stats["max_cuda_reserved_bytes"] = max(
                    stats["max_cuda_reserved_bytes"], peak_reserved
                )

    def summary(self) -> dict:
        return {
            "host_rss_available": self._process is not None,
            "stages": {stage: dict(values) for stage, values in sorted(self._stats.items())},
        }
