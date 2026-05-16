import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _task_rows(records: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Task | Category | Image AUROC | Pixel AUPR | Pixel AUROC | Image AP | Approval | Updates | Coreset | ACC min/p50/p90/max |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rec in records:
        train = rec.get("train", {})
        eval_metrics = rec.get("eval", {})
        acc = (
            f"{_fmt(train.get('acc_min'))}/"
            f"{_fmt(train.get('acc_p50'))}/"
            f"{_fmt(train.get('acc_p90'))}/"
            f"{_fmt(train.get('acc_max'))}"
        )
        lines.append(
            "| {task} | {cat} | {img} | {paupr} | {pauc} | {ap} | {approval} | {updates} | {coreset} | {acc} |".format(
                task=rec.get("task_id", "-"),
                cat=rec.get("category", "-"),
                img=_fmt(eval_metrics.get("image_auroc")),
                paupr=_fmt(eval_metrics.get("pixel_aupr")),
                pauc=_fmt(eval_metrics.get("pixel_auroc")),
                ap=_fmt(eval_metrics.get("image_ap")),
                approval=_fmt(train.get("acc_approval_rate")),
                updates=train.get("coreset_update_count", "-"),
                coreset=train.get("coreset_size", "-"),
                acc=acc,
            )
        )
    return lines


def _cumulative_rows(records: List[Dict[str, Any]], final_metrics: Dict[str, Any] | None) -> List[str]:
    rows = []
    for rec in records:
        if "cumulative_eval" not in rec:
            continue
        metrics = rec["cumulative_eval"]
        rows.append(
            "| {task} | {cat} | {img} | {paupr} | {pauc} | {n} | {sec} |".format(
                task=rec.get("task_id", "-"),
                cat=rec.get("category", "-"),
                img=_fmt(metrics.get("image_auroc")),
                paupr=_fmt(metrics.get("pixel_aupr")),
                pauc=_fmt(metrics.get("pixel_auroc")),
                n=metrics.get("eval_num_images", "-"),
                sec=_fmt(metrics.get("eval_seconds"), digits=1),
            )
        )
    if final_metrics and not rows:
        rows.append(
            "| final | final | {img} | {paupr} | {pauc} | {n} | {sec} |".format(
                img=_fmt(final_metrics.get("image_auroc")),
                paupr=_fmt(final_metrics.get("pixel_aupr")),
                pauc=_fmt(final_metrics.get("pixel_auroc")),
                n=final_metrics.get("eval_num_images", "-"),
                sec=_fmt(final_metrics.get("eval_seconds"), digits=1),
            )
        )
    if not rows:
        return []
    return [
        "| After Task | Category | Cumulative Image AUROC | Cumulative Pixel AUPR | Cumulative Pixel AUROC | Images | Seconds |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        *rows,
    ]


def summarize_run(run_dir: Path) -> str:
    summary = _load_json(run_dir / "run_summary.json")
    records_path = run_dir / "task_records.json"
    if records_path.exists():
        records = _load_json(records_path)
    else:
        records = [
            _load_json(path)
            for path in sorted(run_dir.glob("task_*_metrics.json"))
        ]

    final_path = run_dir / "final_cumulative_metrics.json"
    final_metrics = _load_json(final_path) if final_path.exists() else None
    resolved_config_path = run_dir / "resolved_config.yaml"
    resolved_config = None
    if resolved_config_path.exists():
        with resolved_config_path.open("r", encoding="utf-8") as f:
            resolved_config = yaml.safe_load(f)

    lines = [
        f"## {summary.get('run_name', run_dir.name)}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Run dir | `{run_dir.as_posix()}` |",
        f"| Tasks completed | {summary.get('tasks_completed', '-')} |",
        f"| Backbone | `{_config_value(resolved_config, 'model.backbone', '-')}` |",
        f"| tau_acc | {_config_value(resolved_config, 'model.tau_acc', '-')} |",
        f"| nearest_neighbors | {summary.get('nearest_neighbors', '-')} |",
        f"| max_coreset_size | {_config_value(resolved_config, 'model.max_coreset_size', '-')} |",
        f"| pixel_score_norm | `{summary.get('pixel_score_norm', '-')}` |",
        f"| gaussian_smoothing_sigma | {summary.get('gaussian_smoothing_sigma', '-')} |",
        f"| patch_grid / n_patch | {summary.get('patch_grid', '-')} / {summary.get('n_patch', '-')} |",
        f"| avg current image AUROC | {_fmt(summary.get('avg_eval_image_auroc'))} |",
        f"| avg current pixel AUROC | {_fmt(summary.get('avg_eval_pixel_auroc'))} |",
        f"| avg current pixel AUPR | {_fmt(summary.get('avg_eval_pixel_aupr'))} |",
        f"| final cumulative image AUROC | {_fmt(summary.get('final_cumulative_image_auroc'))} |",
        f"| final cumulative pixel AUPR | {_fmt(summary.get('final_cumulative_pixel_aupr'))} |",
        "",
        "### Per-Task Breakdown",
        "",
        *_task_rows(records),
    ]

    cumulative = _cumulative_rows(records, final_metrics)
    if cumulative:
        lines.extend(["", "### Cumulative Eval", "", *cumulative])

    if records:
        counts = records[-1].get("train", {}).get("coreset_task_counts", {})
        if counts:
            lines.extend(["", "### Final Coreset Task Counts", ""])
            for task_id, count in sorted(counts.items(), key=lambda item: int(item[0])):
                lines.append(f"- task {task_id}: {count}")

    return "\n".join(lines).rstrip() + "\n"


def _config_value(config: Dict[str, Any] | None, dotted_key: str, default: Any) -> Any:
    if config is None:
        return default
    node: Any = config
    for key in dotted_key.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a Meta-NATH run directory.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown output path.")
    args = parser.parse_args()

    markdown = summarize_run(args.run_dir)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
