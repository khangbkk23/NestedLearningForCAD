import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _metric_delta(before: Dict[str, Any], after: Dict[str, Any], metric: str) -> Dict[str, Any]:
    before_value = float(before.get(metric, 0.0))
    after_value = float(after.get(metric, 0.0))
    return {
        "metric": metric,
        "before": before_value,
        "after": after_value,
        "delta": after_value - before_value,
    }


def evaluate_acceptance(
    before: Dict[str, Any],
    after: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    acceptance_cfg = config.get("phase3", {}).get("acceptance", {})
    enabled = bool(acceptance_cfg.get("enabled", True))
    primary_metric = str(acceptance_cfg.get("primary_metric", "final_cumulative_pixel_aupr"))
    min_primary_delta = float(acceptance_cfg.get("min_primary_delta", -0.005))
    image_metric = str(acceptance_cfg.get("image_metric", "final_cumulative_image_auroc"))
    min_image_delta = float(acceptance_cfg.get("min_image_delta", -0.002))

    primary = _metric_delta(before, after, primary_metric)
    image = _metric_delta(before, after, image_metric)

    checks = [
        {
            **primary,
            "min_delta": min_primary_delta,
            "passed": primary["delta"] >= min_primary_delta,
            "role": "primary",
        },
        {
            **image,
            "min_delta": min_image_delta,
            "passed": image["delta"] >= min_image_delta,
            "role": "guardrail",
        },
    ]

    accepted = bool(enabled and all(check["passed"] for check in checks))
    return {
        "enabled": enabled,
        "accepted": accepted,
        "decision": "accepted" if accepted else "rejected",
        "checks": checks,
        "before_summary": before,
        "after_summary": after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Phase 3 metric-gated acceptance.")
    parser.add_argument("--config", type=Path, default=Path("conf/full_demo.yaml"))
    parser.add_argument("--before", type=Path, required=True, help="Before checkpoint_eval_summary.json")
    parser.add_argument("--after", type=Path, required=True, help="After checkpoint_eval_summary.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = evaluate_acceptance(
        before=_load_json(args.before),
        after=_load_json(args.after),
        config=_load_config(args.config),
    )

    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
