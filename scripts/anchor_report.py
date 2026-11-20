"""B323 report: ``anchor`` versus the byte-matched ``er_bytes`` control. Standard library only.

All deltas are ``anchor - er_bytes`` from the same job. Metric definitions are
those of B319/B322 (see shadow_report.summarize). The 32-sample ER arm of B319
is a reported reference, not part of the gate.
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRIC_VERSION = "B323-anchor-v1"


def _shadow_report():
    spec = importlib.util.spec_from_file_location("shadow_report", ROOT / "scripts/shadow_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stages_from_matrix(matrix):
    tasks, values = matrix["tasks"], matrix["values"]
    return [
        {c: {"pixel": {"AP": values["pixel_AP"][b][j], "AUROC": values["pixel_AUROC"][b][j]},
             "image": {"AP": values["image_AP"][b][j], "AUROC": values["image_AUROC"][b][j]}}
         for j, c in enumerate(tasks[: b + 1])}
        for b in range(len(tasks))
    ]


def compare(control, candidate, gate):
    cats = list(control["per_category"])
    delta = {
        c: {key: candidate["per_category"][c][key] - control["per_category"][c][key]
            for key in ("final_pixel_AP", "final_image_AUROC", "acquisition_pixel_AP")}
        for c in cats
    }
    values = {
        "final_pixel_AP_gain": candidate["final_macro_pixel_AP"] - control["final_macro_pixel_AP"],
        "worst_category_final_pixel_AP_delta": min(d["final_pixel_AP"] for d in delta.values()),
        "acquisition_macro_delta": candidate["macro_acquisition_pixel_AP"] - control["macro_acquisition_pixel_AP"],
        "final_image_AUROC_delta": candidate["final_macro_image_AUROC"] - control["final_macro_image_AUROC"],
        "Fmax_reduction": control["macro_Fmax"] - candidate["macro_Fmax"],
    }
    conditions = {
        "a_final_pixel_AP": values["final_pixel_AP_gain"] >= gate["final_pixel_AP_gain_min"],
        "b_category_floor": values["worst_category_final_pixel_AP_delta"] >= gate["category_pixel_AP_delta_min"],
        "c_plasticity": values["acquisition_macro_delta"] >= -gate["acquisition_macro_drop_max"],
        "d_image_AUROC": values["final_image_AUROC_delta"] >= -gate["final_image_AUROC_drop_max"],
    }
    if all(conditions.values()):
        label = "SIGNAL"
    elif not conditions["a_final_pixel_AP"]:
        label = "NO_SIGNAL"
    else:
        label = "MIXED"
    floor = sorted(c for c, d in delta.items() if d["final_pixel_AP"] < gate["category_pixel_AP_delta_min"])
    return {"label": label, "values": values, "conditions": conditions,
            "category_delta": delta, "floor_violations": floor}


def age_profile(control_study, candidate_study):
    """Descriptive, post-hoc: anchor - er_bytes Pixel AP by task age (boundaries since acquisition)."""
    control = [s["metrics"] for s in control_study["snapshots"]]
    candidate = [s["metrics"] for s in candidate_study["snapshots"]]
    last = len(control) - 1
    by_age = {}
    for b in range(last + 1):
        for task in candidate[b]:
            if task not in control[b]:
                continue
            age = b - list(candidate[b]).index(task)
            by_age.setdefault(age, []).append(candidate[b][task]["pixel"]["AP"] - control[b][task]["pixel"]["AP"])
    return {str(age): {"mean": sum(v) / len(v), "cells": len(v)} for age, v in sorted(by_age.items())}


def complete(study, arm):
    if (study.get("arm") != arm or study.get("status") != "COMPLETE_REVIEW_PENDING"
            or study.get("committed") != 3629 or len(study.get("snapshots", [])) != 15):
        raise ValueError("Incomplete B323 arm: " + arm)
    return [s["metrics"] for s in study["snapshots"]]


def report(control_study, candidate_study, pilot, reference_matrix=None):
    summarize = _shadow_report().summarize
    control = summarize(complete(control_study, pilot["comparison"]["control"]))
    candidate = summarize(complete(candidate_study, pilot["comparison"]["candidate"]))
    prefix = pilot["prefix_reference"]
    first = candidate_study["snapshots"][0]
    result = {
        "metric_version": METRIC_VERSION,
        "experiment": "B323",
        "scientific_acceptance": False,
        "delta_sign": "anchor minus er_bytes",
        "arms": {"er_bytes": control, "anchor": candidate},
        "primary": compare(control, candidate, pilot["gate"]),
        "prefix_check": {
            "expected_model_sha256": prefix["model_sha256"],
            "observed_model_sha256": first["model_sha256"],
            "status": "PREFIX_REPRODUCED" if first["cursor"] == prefix["boundary"]
            and first["model_sha256"] == prefix["model_sha256"] else "PREFIX_MISMATCH",
            "note": "before the first boundary the anchor arm must equal B319 ER; er_bytes has no such reference",
        },
        "anchor_state": candidate_study.get("anchor"),
        "anchor_forwards": candidate_study.get("counters", {}).get("anchor_forwards"),
        "reservoir_capacity": {"er_bytes": control_study.get("reservoir_capacity"),
                               "anchor": candidate_study.get("reservoir_capacity")},
        "age_profile": age_profile(control_study, candidate_study),
        "compute": {"matched": False,
                    "arm_seconds": {"er_bytes": control_study.get("seconds"), "anchor": candidate_study.get("seconds")}},
    }
    result["verdict"] = result["primary"]["label"]
    if reference_matrix is not None:
        reference = summarize(stages_from_matrix(reference_matrix))
        result["reference_er32"] = {
            "source": "B319 er_uniform (32 samples), recorded",
            "summary": {k: reference[k] for k in ("final_macro_pixel_AP", "final_macro_image_AUROC",
                                                   "macro_acquisition_pixel_AP", "macro_Fmax", "macro_Facq")},
            "anchor_minus_er32": compare(reference, candidate, pilot["gate"])["values"],
            "er_bytes_minus_er32": compare(reference, control, pilot["gate"])["values"],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="B323 anchor versus byte-matched ER report")
    parser.add_argument("--control", type=Path, required=True, help="er_bytes study.json")
    parser.add_argument("--candidate", type=Path, required=True, help="anchor study.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/b323.json")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "experiments/B319/runs/b319-s0-20261110-kaggle-v5/metrics.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; reports are write-once")
    reference = None
    if args.reference.is_file():
        reference = json.loads(args.reference.read_text(encoding="utf8"))["arms"]["er_uniform"]["matrix"]
    result = report(json.loads(args.control.read_text(encoding="utf8")),
                    json.loads(args.candidate.read_text(encoding="utf8")),
                    json.loads(args.config.read_text(encoding="utf8")), reference)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
    print(result["verdict"], result["prefix_check"]["status"], json.dumps(result["primary"]["values"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
