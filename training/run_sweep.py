import argparse
import copy
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conf.config import load_config
from training.run_experiment import run_experiment


def _as_int_list(values: List[int]) -> List[int]:
    return [int(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameter sweeps for continual anomaly experiments")
    parser.add_argument("--config", default="conf/config.yaml", help="Path to YAML config")
    parser.add_argument("--profile", default=None, choices=["tiny", "quick", "default", "full"], help="Optional profile")
    parser.add_argument("--run_prefix", default="sweep", help="Prefix for generated run names")

    parser.add_argument("--max_tasks", type=int, default=None, help="Limit tasks per run")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per task")
    parser.add_argument("--device", default=None, help="Override device")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B for all runs")
    parser.add_argument("--quiet", action="store_true", help="Disable per-task verbose output")

    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Seed list")
    parser.add_argument("--generators", nargs="*", default=None, help="Anomaly generator list")
    parser.add_argument("--cms_levels", type=int, nargs="*", default=None, help="CMS level list")
    parser.add_argument("--k_values", type=int, nargs="*", default=None, help="CMS k list")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config: Dict[str, Any] = load_config(args.config)

    dataset_cfg = base_config.setdefault("dataset", {})
    model_cfg = base_config.setdefault("model", {})
    training_cfg = base_config.setdefault("training", {})
    logging_cfg = base_config.setdefault("logging", {})

    seeds = _as_int_list(args.seeds) if args.seeds else [int(training_cfg.get("seed", 42))]
    generators = args.generators if args.generators else [str(dataset_cfg.get("anomaly_generator", "superpixel"))]
    cms_levels = _as_int_list(args.cms_levels) if args.cms_levels else [int(model_cfg.get("cms_levels", 3))]
    k_values = _as_int_list(args.k_values) if args.k_values else [int(model_cfg.get("k", 2))]

    combinations = list(itertools.product(seeds, generators, cms_levels, k_values))
    total_runs = len(combinations)
    if total_runs == 0:
        raise RuntimeError("Sweep produced no run combinations.")

    results_dir = Path(logging_cfg.get("results_dir", "results"))
    sweep_root = results_dir / "sweeps"
    sweep_root.mkdir(parents=True, exist_ok=True)

    sweep_name = f"{args.run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir = sweep_root / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=False)

    with open(sweep_dir / "sweep_plan.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_runs": total_runs,
                "seeds": seeds,
                "generators": generators,
                "cms_levels": cms_levels,
                "k_values": k_values,
            },
            f,
            indent=2,
        )

    summaries = []

    for idx, (seed, generator, cms_level, k_value) in enumerate(combinations, start=1):
        run_config = copy.deepcopy(base_config)
        run_config.setdefault("dataset", {})["anomaly_generator"] = generator
        run_config.setdefault("model", {})["cms_levels"] = cms_level
        run_config.setdefault("model", {})["k"] = k_value
        run_config.setdefault("training", {})["seed"] = seed

        run_name = f"{args.run_prefix}_s{seed}_g{generator}_l{cms_level}_k{k_value}"
        print(f"\n[{idx}/{total_runs}] Running {run_name}")

        summary = run_experiment(
            config=run_config,
            config_path=args.config,
            profile=args.profile,
            run_name=run_name,
            max_tasks=args.max_tasks,
            epochs_override=args.epochs,
            device_override=args.device,
            seed_override=seed,
            disable_wandb=args.disable_wandb,
            verbose=not args.quiet,
        )

        summary_record = {
            "run_name": run_name,
            "seed": seed,
            "generator": generator,
            "cms_levels": cms_level,
            "k": k_value,
            "run_dir": summary.get("run_dir"),
            "num_tasks": summary.get("num_tasks", 0),
            "aggregate": summary.get("aggregate", {}),
        }
        summaries.append(summary_record)

        with open(sweep_dir / "sweep_results.json", "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)

    def _best(metric_name: str, higher_is_better: bool = True):
        metric_runs = [
            r for r in summaries
            if metric_name in r.get("aggregate", {})
        ]
        if not metric_runs:
            return None
        return sorted(
            metric_runs,
            key=lambda r: r["aggregate"][metric_name],
            reverse=higher_is_better,
        )[0]

    final_summary = {
        "sweep_name": sweep_name,
        "sweep_dir": sweep_dir.as_posix(),
        "num_runs": len(summaries),
        "best_by_image_auroc": _best("avg_image_auroc", higher_is_better=True),
        "best_by_pixel_ap": _best("avg_pixel_ap", higher_is_better=True),
    }

    with open(sweep_dir / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    with open(sweep_dir / "sweep_config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(base_config, f, sort_keys=False)

    print("\nSweep complete")
    print(f"Sweep directory: {sweep_dir.as_posix()}")


if __name__ == "__main__":
    main()
