# exps/cms_nm01/reporting.py
"""Explicitly keyed scientific tables; never transpose category/level vectors."""

from __future__ import annotations

import math


def level_rows(record: dict) -> list[dict]:
    rows = []
    for boundary in record["stages"]:
        for category, metrics in boundary["metrics"]["per_category"].items():
            means = metrics["level_mean"]
            mixture = metrics["mixture_mean"]
            if mixture > min(means) + math.log(len(means)) + 2e-4:
                raise ValueError("mixture/level mean bound violated")
            rows.append({"seed": record["order_seed"], "arm": record["arm"],
                         "boundary": boundary["stage"], "category": category,
                         "mixture": mixture, **{name: means[index] for index, name in enumerate(("fast", "medium", "slow")[:len(means)])}})
    return rows


def markdown_level_table(records: list[dict], *, final_only: bool = False) -> str:
    lines = ["| Seed | Boundary | Category | Mixture | Fast | Medium | Slow |",
             "|---:|---:|---|---:|---:|---:|---:|"]
    for record in records:
        for row in level_rows(record):
            if final_only and row["boundary"] != 3:
                continue
            values = [f"{row[name]:.6f}" if name in row else "—" for name in ("mixture", "fast", "medium", "slow")]
            lines.append(f"| {row['seed']} | {row['boundary']} | {row['category']} | " + " | ".join(values) + " |")
    return "\n".join(lines)
