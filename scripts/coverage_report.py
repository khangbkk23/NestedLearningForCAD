"""B324 reports (card section 6). Standard library only.

    python scripts/coverage_report.py q0 --q0 <study> --nested <study> --output <json>
    python scripts/coverage_report.py s1 --raw1 <study> --iid <study> --output <json>
    python scripts/coverage_report.py s2 --pair <flat study> <nested study> [--pair ...] --output <json>

Metric definitions are B322's (shadow_report.summarize). Deltas are ``nested - flat``.
"""
import argparse
import importlib.util
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
METRIC_VERSION = "B324-coverage-v1"
REFERENCE = ROOT / "experiments/B319/runs/b319-s0-20261110-kaggle-v5/metrics.json"


def _summarize():
    spec = importlib.util.spec_from_file_location("shadow_report", ROOT / "scripts/shadow_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize


def _stages_from_matrix(matrix):
    tasks, values = matrix["tasks"], matrix["values"]
    return [{c: {"pixel": {"AP": values["pixel_AP"][b][j], "AUROC": values["pixel_AUROC"][b][j]},
                 "image": {"AP": values["image_AP"][b][j], "AUROC": values["image_AUROC"][b][j]}}
             for j, c in enumerate(tasks[: b + 1])} for b in range(len(tasks))]


def _complete(study, arm, committed=3629, boundaries=15):
    if (study.get("arm") != arm or study.get("status") != "COMPLETE_REVIEW_PENDING"
            or study.get("committed") != committed or len(study.get("snapshots", [])) != boundaries):
        raise ValueError(f"Incomplete or unexpected B324 arm: {arm}")
    return study


def _memory_trajectory(events_path):
    return [json.loads(line)["memory_sha256"] for line in Path(events_path).read_text(encoding="utf8").splitlines()]


def _summary_keys(summary):
    return {k: summary[k] for k in ("final_macro_pixel_AP", "final_macro_image_AUROC",
                                     "macro_acquisition_pixel_AP", "macro_Fmax", "macro_Facq")}


def q0_report(q0, nested, pilot):
    _complete(q0, "q0", committed=433, boundaries=2)
    _complete(nested, "nested", committed=433, boundaries=2)
    reference = pilot["identity_reference"]["model_sha256"]
    observed = {str(s["cursor"]): s["model_sha256"] for s in q0["snapshots"]}
    identity = observed == reference
    expected_nested = 433 + (433 // 64) * 128
    checks = {
        "q0_model_sha256_matches_B319_ER": identity,
        "q0_optimizer_steps": q0["counters"]["optimizer"] == 433,
        "nested_optimizer_steps": nested["counters"]["optimizer"] == expected_nested,
        "nested_extra_updates": nested["counters"]["extra_updates"] == (433 // 64) * 128,
        "replay_within_budget": all(s.get("replay_memory_bytes", 0) <= pilot["replay"]["budget_bytes"]
                                    for s in (q0, nested)),
    }
    return {"metric_version": METRIC_VERSION, "experiment": "B324", "job": "q0",
            "status": "IDENTITY_REPRODUCED" if all(checks.values()) else "IDENTITY_MISMATCH",
            "checks": checks, "observed_model_sha256": observed, "expected_model_sha256": reference,
            "timing": {"q0": q0.get("timing"), "nested": nested.get("timing")},
            "counters": {"q0": q0.get("counters"), "nested": nested.get("counters")},
            "note": "AD numbers of this prefix are not used for any decision"}


def s1_report(raw1, iid, pilot, reference_matrix):
    summarize = _summarize()
    raw = summarize([s["metrics"] for s in _complete(raw1, "raw1")["snapshots"]])
    ref = summarize(_stages_from_matrix(reference_matrix))
    prediction = pilot["s1_prediction"]
    met = raw["macro_Facq"] <= prediction["macro_Facq_max"] and \
        raw["final_macro_pixel_AP"] >= prediction["final_macro_pixel_AP_min"]
    result = {"metric_version": METRIC_VERSION, "experiment": "B324", "job": "s1",
              "label": "COVERAGE_SUPPORTED" if met else "COVERAGE_NOT_SUPPORTED",
              "raw1": _summary_keys(raw), "er32_reference": _summary_keys(ref),
              "raw1_minus_er32": {k: raw[k] - ref[k] for k in _summary_keys(raw)},
              "per_category": raw["per_category"], "matrix": raw["matrix"],
              "note": "raw1 versus the recorded ER 32 is a descriptive, cross-job comparison"}
    if iid is not None:
        _complete(iid, "iid", boundaries=1)
        final = iid["snapshots"][0]["metrics"]
        result["iid_reference"] = {
            "final_macro_pixel_AP": mean(v["pixel"]["AP"] for v in final.values()),
            "final_macro_image_AUROC": mean(v["image"]["AUROC"] for v in final.values()),
            "categories": len(final),
            "note": "same learner and budget as flat on a shuffled stream; not Dinomaly joint training",
        }
    return result


def s2_compare(flat, nested, gate):
    cats = list(flat["per_category"])
    delta = {c: nested["per_category"][c]["final_pixel_AP"] - flat["per_category"][c]["final_pixel_AP"] for c in cats}
    values = {
        "final_pixel_AP_gain": nested["final_macro_pixel_AP"] - flat["final_macro_pixel_AP"],
        "worst_category_delta": min(delta.values()),
        "final_image_AUROC_delta": nested["final_macro_image_AUROC"] - flat["final_macro_image_AUROC"],
        "acquisition_macro_delta": nested["macro_acquisition_pixel_AP"] - flat["macro_acquisition_pixel_AP"],
        "Fmax_reduction": flat["macro_Fmax"] - nested["macro_Fmax"],
    }
    passed = (values["final_pixel_AP_gain"] >= gate["final_pixel_AP_gain_min"]
              and values["worst_category_delta"] >= gate["category_pixel_AP_delta_min"]
              and values["final_image_AUROC_delta"] >= -gate["final_image_AUROC_drop_max"])
    return {"values": values, "passes_signal_rule": passed, "category_delta": delta,
            "floor_violations": sorted(c for c, d in delta.items() if d < gate["category_pixel_AP_delta_min"])}


def s2_report(pairs, pilot):
    summarize = _summarize()
    gate = pilot["s2_gate"]
    seeds = []
    for flat_study, nested_study, memory_equal in pairs:
        if flat_study.get("seed") != nested_study.get("seed"):
            raise ValueError("A pair must share one seed")
        flat = summarize([s["metrics"] for s in _complete(flat_study, "flat")["snapshots"]])
        nested = summarize([s["metrics"] for s in _complete(nested_study, "nested")["snapshots"]])
        seeds.append({"seed": flat_study["seed"], "memory_trajectory_equal": memory_equal,
                      "flat": _summary_keys(flat), "nested": _summary_keys(nested),
                      "compare": s2_compare(flat, nested, gate),
                      "seconds": {"flat": flat_study.get("seconds"), "nested": nested_study.get("seconds")}})
    gains = [s["compare"]["values"]["final_pixel_AP_gain"] for s in seeds]
    if len(seeds) == 2 and all(s["compare"]["passes_signal_rule"] for s in seeds):
        label = "STRUCTURE_SIGNAL"
    elif len(seeds) == 2 and abs(mean(gains)) < gate["adds_nothing_abs_mean_gain_below"]:
        label = "STRUCTURE_ADDS_NOTHING"
    elif len(seeds) == 2:
        label = "MIXED"
    else:
        label = "INCOMPLETE_ONE_SEED"
    return {"metric_version": METRIC_VERSION, "experiment": "B324", "job": "s2", "label": label,
            "delta_sign": "nested minus flat", "seeds": seeds, "mean_final_pixel_AP_gain": mean(gains),
            "all_memory_trajectories_equal": all(s["memory_trajectory_equal"] for s in seeds),
            "compute_note": "flat and nested make the same number of optimizer steps"}


def r1_report(q0, raw1, pilot):
    """Same-job replication on seed 1: raw1 (336 uint8 samples) minus q0 (ER 32 in uint8 form)."""
    summarize = _summarize()
    if q0.get("seed") != 1 or raw1.get("seed") != 1:
        raise ValueError("The replication job runs seed 1")
    base = summarize([s["metrics"] for s in _complete(q0, "q0")["snapshots"]])
    raw = summarize([s["metrics"] for s in _complete(raw1, "raw1")["snapshots"]])
    rule = pilot["r1_rule"]
    gain = raw["final_macro_pixel_AP"] - base["final_macro_pixel_AP"]
    category = {c: raw["per_category"][c]["final_pixel_AP"] - base["per_category"][c]["final_pixel_AP"]
                for c in base["per_category"]}
    image = {c: raw["per_category"][c]["final_image_AUROC"] - base["per_category"][c]["final_image_AUROC"]
             for c in base["per_category"]}
    replicated = gain >= rule["final_pixel_AP_gain_min"] and raw["macro_Facq"] <= rule["raw1_macro_Facq_max"]
    return {"metric_version": METRIC_VERSION, "experiment": "B324", "job": "r1", "seed": 1,
            "label": "REPLICATED" if replicated else "NOT_REPLICATED",
            "delta_sign": "raw1 minus q0 (ER 32 in uint8 form), same job",
            "q0": _summary_keys(base), "raw1": _summary_keys(raw),
            "raw1_minus_q0": {k: raw[k] - base[k] for k in _summary_keys(raw)},
            "category_pixel_AP_delta": category, "category_image_AUROC_delta": image,
            "historical_screen": {
                "pixel_AP_gain_met": gain >= 0.005,
                "pixel_AP_floor_violations": sorted(c for c, d in category.items() if d < -0.01),
                "image_AUROC_floor_violations": sorted(c for c, d in image.items() if d < -0.005),
            },
            "note": "replication of the S1 comparison; seed 0 compared raw1 with the recorded B319 ER across jobs"}


def main():
    parser = argparse.ArgumentParser(description="B324 reports")
    parser.add_argument("kind", choices=("q0", "s1", "s2", "r1"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/b324.json")
    parser.add_argument("--q0", type=Path)
    parser.add_argument("--nested", type=Path)
    parser.add_argument("--raw1", type=Path)
    parser.add_argument("--iid", type=Path)
    parser.add_argument("--pair", nargs=2, type=Path, action="append", metavar=("FLAT", "NESTED"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; reports are write-once")
    pilot = json.loads(args.config.read_text(encoding="utf8"))
    load = lambda path: json.loads(path.read_text(encoding="utf8"))
    if args.kind == "q0":
        result = q0_report(load(args.q0), load(args.nested), pilot)
    elif args.kind == "r1":
        result = r1_report(load(args.q0), load(args.raw1), pilot)
    elif args.kind == "s1":
        reference = load(REFERENCE)["arms"]["er_uniform"]["matrix"]
        result = s1_report(load(args.raw1), load(args.iid) if args.iid else None, pilot, reference)
    else:
        pairs = []
        for flat_path, nested_path in args.pair or []:
            equal = _memory_trajectory(flat_path.with_name("events.jsonl")) == \
                _memory_trajectory(nested_path.with_name("events.jsonl"))
            pairs.append((load(flat_path), load(nested_path), equal))
        result = s2_report(pairs, pilot)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
    print(result.get("label", result.get("status")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
