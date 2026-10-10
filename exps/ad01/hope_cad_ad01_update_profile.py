# exps/ad01/hope_cad_ad01_update_profile.py
"""Break down `NormalSupportMemory.update()` cost once the bank is full.

The v3 sweep measured 32-40 s per image after the 2500-vector bank saturated,
which makes the approved experiment infeasible. This script attributes that time
to specific operations rather than guessing:

* fill cost (bank not yet full)
* per-update cost once full, split into the nearest-neighbour pass, the
  closest-pair pass and the replacement bookkeeping
* how many candidate iterations the CADIC rule actually performs per image
* how many distinct kernel launches that implies

It also times the matrix-multiply distance variant on the identical inputs, so
the optimisation is judged on measured numbers. Numerical agreement between the
two variants is checked by the companion test, not here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from exps.ad01.hope_cad_ad01_fast_coreset import FastCADICPatchCoresetV1
from exps.ad01.hope_cad_ad01_normal_support import (
    ALLOCATION_GLOBAL,
    NormalSupportMemory,
)
from exps.ad01.hope_cad_ad01_run import load_cached

ROOT = Path(__file__).resolve().parents[2]
BUDGET = 2500
GRID = 4


def sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, device: str) -> float:
    sync(device)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return time.perf_counter() - start, result


def instrument(core, counters: dict) -> None:
    """Wrap the two distance passes so their cost is attributed separately."""
    original_nearest = core._nearest
    original_pair = core._closest_pair

    def spy_nearest(*args, **kwargs):
        started = time.perf_counter()
        out = original_nearest(*args, **kwargs)
        counters["nearest_seconds"] += time.perf_counter() - started
        counters["nearest_calls"] += 1
        return out

    def spy_pair(*args, **kwargs):
        started = time.perf_counter()
        out = original_pair(*args, **kwargs)
        counters["pair_seconds"] += time.perf_counter() - started
        counters["pair_calls"] += 1
        return out

    core._nearest = spy_nearest
    core._closest_pair = spy_pair


def measure(fast: bool, device: str, images: int = 5) -> dict:
    # Device placement is part of the measurement: the cached patches must sit
    # where the bank lives, or every distance pass copies the bank over PCIe.
    payload = load_cached("bottle", "train", device)
    patches = payload["patches"].to(device)
    memory = NormalSupportMemory(
        budget=BUDGET, grid=GRID, allocation=ALLOCATION_GLOBAL,
        device=device, fast=fast,
    )
    counters = {"nearest_seconds": 0.0, "pair_seconds": 0.0, "nearest_calls": 0, "pair_calls": 0}
    instrument(memory._global, counters)

    fill: list[float] = []
    full: list[float] = []
    for index in range(images):
        seconds, _ = timed(lambda i=index: memory.update(patches[i]), device)
        was_full = memory._global.is_full
        (full if was_full and index >= 3 else fill).append(seconds)

    return {
        "fast": fast,
        "device": device,
        "coreset_class": type(memory._global).__name__,
        "bank_count": int(memory.count),
        "replaced_features": int(memory._global.replaced_features),
        "fill_seconds": fill,
        "full_bank_seconds_per_image": full,
        "full_bank_ms_per_image": [round(value * 1000, 1) for value in full],
        "mean_full_ms_per_image": (
            round(1000 * sum(full) / len(full), 1) if full else None
        ),
        "seconds_in_nearest": round(counters["nearest_seconds"], 3),
        "seconds_in_closest_pair": round(counters["pair_seconds"], 3),
        "nearest_calls": counters["nearest_calls"],
        "closest_pair_calls": counters["pair_calls"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AD-01 update cost breakdown")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--images", type=int, default=5)
    parser.add_argument("--out", default="results/hope_cad/ad01_phase0/update_profile.json")
    args = parser.parse_args()

    report = {
        "device": args.device,
        "budget": BUDGET,
        "note": "full_bank_* covers updates performed while the bank was already full",
    }
    for fast in (False, True):
        report["reference" if not fast else "fast"] = measure(
            fast, args.device, args.images
        )

    reference = report["reference"]["mean_full_ms_per_image"]
    fast = report["fast"]["mean_full_ms_per_image"]
    if reference and fast:
        report["speedup"] = round(reference / fast, 1)

    output = Path(args.out)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
