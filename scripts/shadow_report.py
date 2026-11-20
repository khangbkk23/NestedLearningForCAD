"""B322 report: fast and shadow readouts from one study.json. Standard library only.

All deltas are ``shadow - fast``. Forgetting follows the B319 definition:
Fmax = max(Pixel AP before final) - final, Facq = Pixel AP at acquisition - final,
both signed and averaged over the 14 categories seen before the final boundary.
Acquisition Pixel AP is the value at the first boundary where the category is
evaluated; its macro is the mean over all 15 categories.
"""
import argparse
import json
from pathlib import Path
from statistics import mean

METRIC_VERSION = "B322-shadow-v1"


def summarize(stages):
    """``stages``: per-boundary ``{category: {pixel: {AP, AUROC}, image: {AP, AUROC}}}``."""
    categories = list(stages[-1])
    if len(stages) != len(categories) or any(
        list(stage) != categories[: index + 1] for index, stage in enumerate(stages)
    ):
        raise ValueError("Expected one new category per boundary in task order")
    per = {}
    for category in categories:
        history = [s[category]["pixel"]["AP"] for s in stages if category in s]
        prior = len(history) > 1
        per[category] = {
            "final_pixel_AP": history[-1],
            "final_image_AUROC": stages[-1][category]["image"]["AUROC"],
            "acquisition_pixel_AP": history[0],
            "Fmax": max(history[:-1]) - history[-1] if prior else None,
            "Facq": history[0] - history[-1] if prior else None,
        }
    matrix = {
        f"{scope}_{metric}": [
            [stage.get(c, {}).get(scope, {}).get(metric) for c in categories]
            for stage in stages
        ]
        for scope in ("pixel", "image")
        for metric in ("AP", "AUROC")
    }
    return {
        "final_macro_pixel_AP": mean(x["final_pixel_AP"] for x in per.values()),
        "final_macro_image_AUROC": mean(x["final_image_AUROC"] for x in per.values()),
        "macro_acquisition_pixel_AP": mean(x["acquisition_pixel_AP"] for x in per.values()),
        "macro_Fmax": mean(x["Fmax"] for x in per.values() if x["Fmax"] is not None),
        "macro_Facq": mean(x["Facq"] for x in per.values() if x["Facq"] is not None),
        "per_category": per,
        "matrix": {"tasks": categories, "values": matrix},
    }


def compare(fast, shadow, gate, screen):
    """Registered pilot label and the unchanged historical screen, shadow versus fast."""
    cats = list(fast["per_category"])
    delta = {
        c: {
            "final_pixel_AP": shadow["per_category"][c]["final_pixel_AP"] - fast["per_category"][c]["final_pixel_AP"],
            "final_image_AUROC": shadow["per_category"][c]["final_image_AUROC"] - fast["per_category"][c]["final_image_AUROC"],
            "acquisition_pixel_AP": shadow["per_category"][c]["acquisition_pixel_AP"] - fast["per_category"][c]["acquisition_pixel_AP"],
        }
        for c in cats
    }
    values = {
        "final_pixel_AP_gain": shadow["final_macro_pixel_AP"] - fast["final_macro_pixel_AP"],
        "Fmax_reduction": fast["macro_Fmax"] - shadow["macro_Fmax"],
        "acquisition_macro_delta": shadow["macro_acquisition_pixel_AP"] - fast["macro_acquisition_pixel_AP"],
        "acquisition_worst_category_delta": min(d["acquisition_pixel_AP"] for d in delta.values()),
        "final_image_AUROC_delta": shadow["final_macro_image_AUROC"] - fast["final_macro_image_AUROC"],
    }
    conditions = {
        "a_final_pixel_AP": values["final_pixel_AP_gain"] >= gate["final_pixel_AP_gain_min"],
        "b_retention": values["Fmax_reduction"] >= gate["Fmax_reduction_min"],
        "c_plasticity": values["acquisition_macro_delta"] >= -gate["acquisition_macro_drop_max"]
        and values["acquisition_worst_category_delta"] >= -gate["acquisition_category_drop_max"],
        "d_image_AUROC": values["final_image_AUROC_delta"] >= -gate["final_image_AUROC_drop_max"],
    }
    if all(conditions.values()):
        label = "SIGNAL"
    elif not conditions["a_final_pixel_AP"] or not conditions["b_retention"]:
        label = "NO_SIGNAL"
    else:
        label = "MIXED"
    violations = {
        c: d
        for c, d in delta.items()
        if d["final_pixel_AP"] < screen["category_pixel_AP_delta_min"]
        or d["final_image_AUROC"] < screen["category_image_AUROC_delta_min"]
    }
    met = values["final_pixel_AP_gain"] >= screen["pixel_AP_gain_min"] and not violations
    return {
        "label": label,
        "values": values,
        "conditions": conditions,
        "category_delta": delta,
        "historical_screen": {
            "status": "SCREEN_MET" if met else "SCREEN_NOT_MET",
            "floor_violations": sorted(violations),
        },
    }


def report(study, pilot):
    snapshots = study["snapshots"]
    if study["status"] != "COMPLETE_REVIEW_PENDING" or study["committed"] != 3629 or len(snapshots) != 15:
        raise ValueError("Incomplete B322 arm")
    taus = [str(t) for t in pilot["shadow"]["taus"]]
    readouts = {"fast": summarize([s["metrics"] for s in snapshots])}
    for tau in taus:
        readouts["ema" + tau] = summarize([s["shadow"][tau]["metrics"] for s in snapshots])
    primary, secondary = str(pilot["shadow"]["primary"]), str(pilot["shadow"]["secondary"])
    result = {
        "metric_version": METRIC_VERSION,
        "experiment": "B322",
        "scientific_acceptance": False,
        "delta_sign": "shadow minus fast",
        "readouts": readouts,
        "primary": compare(readouts["fast"], readouts["ema" + primary], pilot["gate"], pilot["historical_screen"]),
        "secondary_exploratory": compare(readouts["fast"], readouts["ema" + secondary], pilot["gate"], pilot["historical_screen"]),
        "fast_model_sha256": [s["model_sha256"] for s in snapshots],
    }
    result["verdict"] = result["primary"]["label"]
    result["secondary_note"] = (
        "secondary has a signal that needs confirmation; the primary verdict is unchanged"
        if result["verdict"] != "SIGNAL" and result["secondary_exploratory"]["label"] == "SIGNAL"
        else None
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="B322 fast/shadow report from a completed arm")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/b322.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; reports are write-once")
    result = report(json.loads(args.study.read_text(encoding="utf8")), json.loads(args.config.read_text(encoding="utf8")))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
    print(result["verdict"], json.dumps(result["primary"]["values"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
