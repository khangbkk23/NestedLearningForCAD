"""B322 analysis A1: headroom and teacher quality from stored boundary matrices.

Post-hoc and exploratory. Standard library only; reads the committed B322
metrics.json and changes no closed receipt or verdict.
"""
import argparse
import json
from pathlib import Path
from statistics import mean

ANALYSIS = "B322-A1-headroom-v1"


def analyse(report):
    readouts = report["readouts"]
    tasks = readouts["fast"]["matrix"]["tasks"]
    ap = {name: value["matrix"]["values"]["pixel_AP"] for name, value in readouts.items()}
    last = len(tasks) - 1
    fast = ap["fast"]
    final = [fast[last][j] for j in range(len(tasks))]
    acquisition = [fast[j][j] for j in range(len(tasks))]
    best = [max(fast[b][j] for b in range(j, last + 1)) for j in range(len(tasks))]
    loss = {t: acquisition[j] - final[j] for j, t in enumerate(tasks)}
    positive = {t: v for t, v in loss.items() if v > 0}
    top = sorted(positive, key=positive.get, reverse=True)[:3]
    out = {
        "analysis": ANALYSIS,
        "status": "exploratory post-hoc, one seed, exposed MVTec test",
        "fast": {
            "final_macro_pixel_AP": mean(final),
            "acquisition_macro_pixel_AP": mean(acquisition),
            "acquisition_minus_final_macro_15": mean(acquisition) - mean(final),
            "best_boundary_macro_pixel_AP": mean(best),
            "best_boundary_minus_final": mean(best) - mean(final),
            "note": "best-boundary is an optimistic ceiling (per-category max on one seed)",
        },
        "forgetting_concentration": {
            "acquisition_minus_final": loss,
            "top3": top,
            "top3_share_of_positive_mass": sum(positive[t] for t in top) / sum(positive.values()),
        },
        "backward_transfer": {t: v for t, v in loss.items() if v < -0.01},
        "oracle_readout_choice_at_final": {
            "macro_pixel_AP": mean(max(ap[n][last][j] for n in ap) for j in range(len(tasks))),
            "note": "per-category choice among the three readouts; needs the task identity",
        },
        "shadow_minus_fast_on_old_tasks": {},
    }
    for name in ap:
        if name == "fast":
            continue
        by_age = {}
        for age in range(0, last + 1):
            cells = [ap[name][b][j] - fast[b][j] for b in range(last + 1) for j in range(b + 1) if b - j == age]
            by_age[str(age)] = {"mean": mean(cells), "cells": len(cells)}
        older = {}
        for minimum in (1, 2, 3):
            cells = [ap[name][b][j] - fast[b][j] for b in range(last + 1) for j in range(b + 1) if b - j >= minimum]
            older[f"age>={minimum}"] = {"mean": mean(cells), "share_positive": sum(c > 0 for c in cells) / len(cells),
                                         "cells": len(cells)}
        out["shadow_minus_fast_on_old_tasks"][name] = {"by_age_in_boundaries": by_age, "pooled": older}
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="B322 A1 headroom analysis (stdlib, post-hoc)")
    parser.add_argument("--metrics", type=Path,
                        default=root / "experiments/B322/runs/b322-s0-20261115-01/metrics.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; analyses are write-once")
    result = analyse(json.loads(args.metrics.read_text(encoding="utf8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
    print(json.dumps({k: result[k] for k in ("fast", "oracle_readout_choice_at_final")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
