# exps/ad01/gpu_smoke_worker.py
"""Minimal per-GPU smoke worker: touch one small memory, then exit.

Used only by the lightweight two-GPU dispatch test. It builds a small normal
support memory on the visible device, performs exactly one update and one
scoring call, writes a tiny JSON report and exits. It never reads the dataset,
the real feature cache or the development manifest, and it cannot affect any
scientific result.

The point is to prove three things cheaply: the dispatched command line is
accepted, the subprocess starts and exits cleanly, and the process really sees
the GPU it was pinned to.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# A subprocess does not inherit pytest's `pythonpath` plugin, so the repository
# root must be put on `sys.path` explicitly before importing project modules.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from exps.ad01.hope_cad_ad01_normal_support import (  # noqa: E402
    ALLOCATION_SPATIAL,
    PATCHES,
    SCORING_LOCAL,
    NormalSupportMemory,
)

DIM = 768
BUDGET = 64          # tiny on purpose
GRID = 2
PATCHES_PER_IMAGE = PATCHES


def main() -> None:
    parser = argparse.ArgumentParser(description="AD-01 per-GPU smoke worker")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    available = torch.cuda.is_available()
    report = {
        "device_requested": args.device,
        "cuda_available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
    }

    if available:
        torch.cuda.synchronize()
        report["device_name"] = torch.cuda.get_device_name(0)

    memory = NormalSupportMemory(
        budget=BUDGET, grid=GRID, allocation=ALLOCATION_SPATIAL,
        dim=DIM, device=args.device if available else "cpu",
    )
    generator = torch.Generator().manual_seed(11)
    image = torch.randn(PATCHES_PER_IMAGE, DIM, generator=generator)
    memory.update(image)
    if available:
        torch.cuda.synchronize()
    images, pixels = memory.score(
        image.unsqueeze(0), scoring=SCORING_LOCAL
    )
    if available:
        torch.cuda.synchronize()

    report.update(
        {
            "count": int(memory.count),
            "bank_features_device": str(memory._banks[0].features.device),
            "image_score_finite": bool(torch.isfinite(images).all()),
            "pixel_scores": int(pixels.shape[1]),
            "seconds": time.perf_counter() - start,
            "status": "ok",
        }
    )
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
