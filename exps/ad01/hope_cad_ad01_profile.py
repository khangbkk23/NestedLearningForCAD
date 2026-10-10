# exps/ad01/hope_cad_ad01_profile.py
"""AD-01 profiling: locate the real bottleneck before optimizing anything.

Measures, on the actual AD-01 work units:

1. memory update per image (global and spatial allocation)
2. scoring per development image (global and local scoring)
3. host-to-device transfer for a development batch
4. CUDA synchronization overhead, i.e. how much measured time is launch
   asynchrony rather than computation
5. nearest-neighbour search cost in isolation, to judge whether the distance
   kernel is worth touching

No scientific result is produced here. The point is to decide what to optimize
from measurement rather than intuition.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from exps.ad01.hope_cad_ad01_arms import PRIMARY_GRID, TOTAL_BUDGET
from exps.ad01.hope_cad_ad01_normal_support import (
    ALLOCATION_GLOBAL,
    ALLOCATION_SPATIAL,
    PATCHES,
    SCORING_GLOBAL,
    SCORING_LOCAL,
    NormalSupportMemory,
)
from exps.ad01.hope_cad_ad01_run import CATEGORIES, load_cached

ROOT = Path(__file__).resolve().parents[2]


def sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device: str, repeat: int = 1) -> float:
    """Wall seconds with proper synchronisation around the call."""
    sync(device)
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync(device)
    return (time.perf_counter() - start) / repeat


def main() -> None:
    parser = argparse.ArgumentParser(description="AD-01 profiling")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results/hope_cad/ad01_phase0/profile.json")
    args = parser.parse_args()
    device = args.device

    payload = load_cached("bottle", "train", device)
    patches = payload["patches"].to(device)
    dev = load_cached("bottle", "dev", device)
    dev_patches = dev["patches"].to(device)

    report: dict[str, object] = {"device": device, "budget": TOTAL_BUDGET, "grid": PRIMARY_GRID}

    for allocation in (ALLOCATION_GLOBAL, ALLOCATION_SPATIAL):
        memory = NormalSupportMemory(
            budget=TOTAL_BUDGET, grid=PRIMARY_GRID, allocation=allocation, device=device
        )
        # Warm up so the first-call kernel compilation is not attributed to the
        # steady-state per-image cost.
        for index in range(min(8, patches.shape[0])):
            memory.update(patches[index])

        single_update = timed(lambda: memory.update(patches[10]), device, repeat=10)
        report[f"update_seconds_per_image_{allocation}"] = single_update

        # Fill the bank so scoring runs against a realistic, saturated memory.
        for index in range(patches.shape[0]):
            memory.update(patches[index])
        report[f"bank_count_{allocation}"] = int(memory.count)

        for scoring in (SCORING_GLOBAL, SCORING_LOCAL):
            batch = dev_patches[:8]
            report[f"score_seconds_per_image_{allocation}_{scoring}"] = timed(
                lambda b=batch: memory.score(b, scoring=scoring), device, repeat=3
            ) / batch.shape[0]

        # A second, independent run measures the *unsynchronised* time so the
        # difference from the synchronised number exposes launch asynchrony.
        sync(device)
        start = time.perf_counter()
        memory.score(dev_patches[:8], scoring=SCORING_GLOBAL)
        unsynced = (time.perf_counter() - start) / 8
        report[f"score_seconds_per_image_{allocation}_global_unsynced"] = unsynced

    # Transfer cost for one development batch, which is pure overhead if the
    # features are moved on every evaluation instead of staying resident.
    host = torch.randn(8, PATCHES, 768)
    report["transfer_seconds_per_batch_8"] = timed(
        lambda: host.to(device), device, repeat=5
    )

    # Isolated nearest-neighbour search at the saturated bank size.
    bank = torch.randn(TOTAL_BUDGET, 768, device=device)
    query = torch.randn(PATCHES, 768, device=device)
    reference = memory._global
    report["nearest_seconds_784x2500"] = timed(
        lambda: reference._nearest(query, bank), device, repeat=3
    )
    report["nearest_seconds_784x10000"] = timed(
        lambda: reference._nearest(query, torch.randn(10000, 768, device=device)),
        device,
        repeat=3,
    )

    output = Path(args.out)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
