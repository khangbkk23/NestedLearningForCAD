# exps/ad02/hope_cad_ad02_analysis.py
"""AD-02 mechanism analysis.

Reads the stored per-run records and answers the three competing explanations
with paired, per-seed contrasts rather than significance claims:

    H1 spatial locality       S4 - R4 : identical groups, sizes, quotas and
                              per-image patch populations; only spatial
                              adjacency differs
    H2 reduced competition    the fineness family G(1) < S2(4) < S4(16) < S7(49)
                              groups, and the fact that R4 has S4's group size
    H3 old-support mediation  treatment -> mediator (retained early-task slots and
                              origin-conditional coverage) -> outcome, with the
                              correlational and the manipulated parts separated

It also aggregates the mandated secondary measurements: false-positive
distributions, update cost split by fill and steady state, inference latency,
persistent bytes and transient working memory.

Writes `results/hope_cad/ad02/analysis.json` and prints the tables used in the
scientific report.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/hope_cad/ad02"
CATEGORIES = ("bottle", "carpet", "hazelnut")
ARMS = ("G", "S2", "S4", "S7", "R4")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}][analysis] {message}", flush=True)


# ------------------------------------------------------------------- loading


def load_runs(base: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted((base / "runs").glob("*.json")):
        payload = json.loads(path.read_text())
        payload["_path"] = str(path.relative_to(ROOT))
        runs.append(payload)
    return runs


def macro(record: dict[str, Any], metric: str) -> float:
    return float(np.mean([record["final"][c][metric] for c in CATEGORIES]))


def per_category(record: dict[str, Any], metric: str) -> dict[str, float]:
    return {c: float(record["final"][c][metric]) for c in CATEGORIES}


def group_by_unit(
    runs: Iterable[dict[str, Any]]
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """`(arm, order_seed) -> [records]`; R4 contributes one record per draw."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in runs:
        grouped.setdefault((record["arm"], record["order_seed"]), []).append(record)
    for records in grouped.values():
        records.sort(key=lambda r: (r.get("draw") is not None, r.get("draw") or 0))
    return grouped


def mean_over_draws(records: Sequence[dict[str, Any]], metric: str) -> float:
    return float(np.mean([macro(record, metric) for record in records]))


# ------------------------------------------------------------------ contrasts


def paired_contrast(
    grouped: dict[tuple[str, int], list[dict[str, Any]]],
    arm_a: str,
    arm_b: str,
    metric: str,
) -> dict[str, Any]:
    """Per-seed paired difference `arm_a - arm_b` on a macro metric."""
    per_seed = {}
    for seed in (0, 1, 2):
        first = grouped.get((arm_a, seed))
        second = grouped.get((arm_b, seed))
        if not first or not second:
            continue
        per_seed[seed] = mean_over_draws(first, metric) - mean_over_draws(
            second, metric
        )
    values = np.array(list(per_seed.values()), dtype=np.float64)
    return {
        "contrast": f"{arm_a}-{arm_b}",
        "metric": metric,
        "per_seed": per_seed,
        "mean": float(values.mean()) if values.size else None,
        "sd": float(values.std(ddof=1)) if values.size > 1 else None,
        "n_seeds": int(values.size),
        "sign_consistent": bool(
            values.size and (np.all(values > 0) or np.all(values < 0))
        ),
        "note": "paired by order seed; no significance claim is made from three seeds",
    }


def category_contrast(
    grouped: dict[tuple[str, int], list[dict[str, Any]]],
    arm_a: str,
    arm_b: str,
    metric: str,
) -> dict[str, Any]:
    per_category_out: dict[str, Any] = {}
    wins = 0
    cells = 0
    for category in CATEGORIES:
        values = []
        for seed in (0, 1, 2):
            first = grouped.get((arm_a, seed))
            second = grouped.get((arm_b, seed))
            if not first or not second:
                continue
            a = float(np.mean([r["final"][category][metric] for r in first]))
            b = float(np.mean([r["final"][category][metric] for r in second]))
            values.append(a - b)
            wins += int(a > b)
            cells += 1
        per_category_out[category] = {
            "per_seed": values,
            "mean": float(np.mean(values)) if values else None,
        }
    return {
        "contrast": f"{arm_a}-{arm_b}",
        "metric": metric,
        "per_category": per_category_out,
        "cells_favouring_first": wins,
        "cells": cells,
    }


# -------------------------------------------------------------------- trends


def fineness_trend(
    grouped: dict[tuple[str, int], list[dict[str, Any]]], metric: str
) -> dict[str, Any]:
    """Metric against log2(group count) over the spatial family G, S2, S4, S7."""
    ladder = (("G", 1), ("S2", 4), ("S4", 16), ("S7", 49))
    points = []
    for arm, groups in ladder:
        for seed in (0, 1, 2):
            records = grouped.get((arm, seed))
            if not records:
                continue
            points.append(
                {
                    "arm": arm,
                    "groups": groups,
                    "seed": seed,
                    "value": mean_over_draws(records, metric),
                }
            )
    if len(points) < 3:
        return {"metric": metric, "points": points, "slope": None}
    x = np.log2([p["groups"] for p in points])
    y = np.array([p["value"] for p in points])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    by_arm = {}
    for arm, _ in ladder:
        values = [p["value"] for p in points if p["arm"] == arm]
        if values:
            by_arm[arm] = float(np.mean(values))
    monotone = None
    ordered = [by_arm.get(arm) for arm, _ in ladder]
    if all(value is not None for value in ordered):
        monotone = bool(
            all(b >= a for a, b in zip(ordered, ordered[1:]))
            or all(b <= a for a, b in zip(ordered, ordered[1:]))
        )
    return {
        "metric": metric,
        "points": points,
        "mean_by_arm": by_arm,
        "slope_per_doubling_of_groups": float(slope),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
        "monotone_in_fineness": monotone,
    }


# ----------------------------------------------------------------- mediation


def _final_boundary(record: dict[str, Any]) -> dict[str, Any]:
    return record["boundaries"][-1]


def first_task_of(record: dict[str, Any]) -> str:
    return record["order"][0]


def mediators(record: dict[str, Any]) -> dict[str, float]:
    """Candidate mediators measured at the final task boundary.

    `retained_first_share`   share of the budget still holding first-task exemplars
    `retained_middle_share`  the same for the task learned second
    `retained_last_share`    the same for the last task; complement of recency bias
    `retention_evenness`     1 - normalised spread of the three shares, so 1 means
                             the budget is split evenly between tasks and 0 means
                             one task holds everything
    `d_own_first_final`      ABSOLUTE mean distance from the first task's normal
                             patches to the nearest surviving exemplar of that same
                             task. This is the coverage quantity; lower is better.
    `d_own_first_increase`   the same minus its value at the boundary where the
                             first task was learned, i.e. how much own-support
                             distance later learning added.

    The per-patch ratio `d_own(final)/d_own(learned)` is deliberately NOT used as a
    mediator. Each arm starts from a different baseline (a spatially dense bank has
    a much smaller baseline distance), so the ratio is not comparable across arms:
    it can rise while the absolute distance falls. Both quantities are reported;
    only the absolute one is used for inference.
    """
    boundary = _final_boundary(record)
    order = record["order"]
    task_ids = {category: index for index, category in enumerate(order)}
    slots = boundary["slots_by_origin"]
    shares = [float(slots.get(str(task_ids[c]), 0)) / record["budget"] for c in order]
    if len(shares) > 1 and np.mean(shares) > 0:
        evenness = 1.0 - float(np.std(shares) / np.mean(shares)) / np.sqrt(len(shares))
    else:
        evenness = float("nan")
    learned = record["boundaries"][0]["coverage"][order[0]]["d_own"]["mean"]
    final = boundary["coverage"][order[0]]["d_own"]["mean"]
    return {
        "retained_first_share": shares[0],
        "retained_middle_share": shares[1] if len(shares) > 1 else float("nan"),
        "retained_last_share": shares[-1],
        "retention_evenness": evenness,
        "d_own_first_final": float(final),
        "d_own_first_learned": float(learned),
        "d_own_first_increase": float(final - learned),
    }


def outcomes(record: dict[str, Any]) -> dict[str, float]:
    first = record["order"][0]
    return {
        "macro_i_auroc": macro(record, "i_auroc"),
        "macro_p_aupr_native": macro(record, "p_aupr_native"),
        "forgetting_i_auroc": float(record["forgetting_i_auroc"]["fm"]),
        "forgetting_p_aupr": float(record["forgetting_p_aupr"]["fm"]),
        "first_task_i_auroc": float(record["final"][first]["i_auroc"]),
        "first_task_p_aupr": float(record["final"][first]["p_aupr_native"]),
    }


def within_arm_slope(
    rows: Sequence[dict[str, Any]], mediator: str, outcome: str, arm_key: str = "arm"
) -> dict[str, Any]:
    """Fixed-effects slope of outcome on mediator, arm means removed.

    Pooling after removing each arm's mean removes exactly the between-arm
    variation that the treatment created, so the residual slope is not simply the
    treatment effect re-labelled as mediation.
    """
    x = np.array([row[mediator] for row in rows], dtype=np.float64)
    y = np.array([row[outcome] for row in rows], dtype=np.float64)
    arms = np.array([row[arm_key] for row in rows])
    if x.size < 4:
        return {"slope": None, "n": int(x.size)}
    x_res = x.copy()
    y_res = y.copy()
    for arm in np.unique(arms):
        mask = arms == arm
        x_res[mask] -= x[mask].mean()
        y_res[mask] -= y[mask].mean()
    if x_res.std() == 0 or y_res.std() == 0:
        return {"slope": None, "n": int(x.size), "reason": "no within-arm variation"}
    slope = float(np.polyfit(x_res, y_res, 1)[0])
    correlation = float(np.corrcoef(x_res, y_res)[0, 1])
    return {
        "slope": slope,
        "within_arm_correlation": correlation,
        "n": int(x.size),
        "n_arms": int(np.unique(arms).size),
    }


def mediation_report(
    grouped: dict[tuple[str, int], list[dict[str, Any]]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for (arm, seed), records in sorted(grouped.items()):
        for record in records:
            row = {
                "arm": arm,
                "order_seed": seed,
                "draw": record.get("draw"),
                "first_task": first_task_of(record),
            }
            row.update(mediators(record))
            row.update(outcomes(record))
            rows.append(row)

    report: dict[str, Any] = {"units": rows, "paths": {}}
    mediator_names = (
        "retained_first_share",
        "retained_middle_share",
        "retention_evenness",
        "d_own_first_final",
        "d_own_first_increase",
    )
    outcome_names = (
        "forgetting_i_auroc",
        "macro_i_auroc",
        "first_task_i_auroc",
        "first_task_p_aupr",
    )
    for mediator in mediator_names:
        for outcome in outcome_names:
            # a-path: effect of the treatment on the mediator itself
            per_seed = {}
            for seed in (0, 1, 2):
                first = grouped.get(("S4", seed))
                second = grouped.get(("G", seed))
                if not first or not second:
                    continue
                per_seed[seed] = float(
                    np.mean([mediators(r)[mediator] for r in first])
                    - np.mean([mediators(r)[mediator] for r in second])
                )
            values = np.array(list(per_seed.values()))
            b_path = within_arm_slope(rows, mediator, outcome)
            report["paths"][f"{mediator}->{outcome}"] = {
                "a_path_S4_minus_G_per_seed": per_seed,
                "a_path_mean": float(values.mean()) if values.size else None,
                "b_path": b_path,
                "product_of_coefficients": (
                    float(values.mean()) * b_path["slope"]
                    if values.size and b_path.get("slope") is not None
                    else None
                ),
                "caveat": (
                    "descriptive only: three order seeds, and the mediator is not "
                    "randomised independently of the arms"
                ),
            }
    return report


# ------------------------------------------------------------------ reporting


def aggregate_arm(grouped, arm: str) -> dict[str, Any]:
    records = [r for (a, _), rs in grouped.items() if a == arm for r in rs]
    if not records:
        return {}
    per_seed_i, per_seed_p = {}, {}
    for seed in (0, 1, 2):
        rs = grouped.get((arm, seed))
        if rs:
            per_seed_i[seed] = mean_over_draws(rs, "i_auroc")
            per_seed_p[seed] = mean_over_draws(rs, "p_aupr_native")
    categories = {}
    for category in CATEGORIES:
        categories[category] = {
            "i_auroc_mean": float(
                np.mean([r["final"][category]["i_auroc"] for r in records])
            ),
            "p_aupr_native_mean": float(
                np.mean([r["final"][category]["p_aupr_native"] for r in records])
            ),
        }
    forgetting = [
        r["forgetting_i_auroc"]["fm"]
        for r in records
        if r["forgetting_i_auroc"]["fm"] is not None
    ]
    forgetting_p = [
        r["forgetting_p_aupr"]["fm"]
        for r in records
        if r["forgetting_p_aupr"]["fm"] is not None
    ]
    normal_scores = [
        value
        for r in records
        for category in CATEGORIES
        for value in r["final"][category]["normal_image_scores"]
    ]
    defect_scores = [
        value
        for r in records
        for category in CATEGORIES
        for value in r["final"][category]["defect_image_scores"]
    ]
    margins = [
        r["final"][category]["separation_margin"]
        for r in records
        for category in CATEGORIES
    ]
    return {
        "arm": arm,
        "n_records": len(records),
        "records": [r["run_id"] for r in records],
        "macro_i_auroc_mean": float(np.mean([macro(r, "i_auroc") for r in records])),
        "macro_p_aupr_native_mean": float(
            np.mean([macro(r, "p_aupr_native") for r in records])
        ),
        "macro_i_auroc_per_seed": per_seed_i,
        "macro_p_aupr_native_per_seed": per_seed_p,
        "per_category": categories,
        "forgetting_i_auroc_mean": float(np.mean(forgetting)) if forgetting else None,
        "forgetting_i_auroc_per_seed": {
            seed: float(
                np.mean(
                    [
                        r["forgetting_i_auroc"]["fm"]
                        for r in grouped.get((arm, seed), [])
                        if r["forgetting_i_auroc"]["fm"] is not None
                    ]
                )
            )
            if grouped.get((arm, seed))
            else None
            for seed in (0, 1, 2)
        },
        "forgetting_p_aupr_mean": float(np.mean(forgetting_p)) if forgetting_p else None,
        "timing": {
            "update_seconds_mean_full_bank": float(
                np.mean(
                    [
                        r["timing"]["update_seconds_mean_full_bank"]
                        for r in records
                        if r["timing"]["update_seconds_mean_full_bank"] is not None
                    ]
                )
            ),
            "update_seconds_mean_fill": float(
                np.mean(
                    [
                        r["timing"]["update_seconds_mean_fill"]
                        for r in records
                        if r["timing"]["update_seconds_mean_fill"] is not None
                    ]
                )
            ),
            "updates_full_bank": int(
                np.sum([r["timing"]["updates_full_bank"] for r in records])
            ),
            "inference_seconds_per_image": float(
                np.mean([r["timing"]["inference_seconds_per_image"] for r in records])
            ),
        },
        "storage": {
            "persistent_exemplar_bytes": int(
                records[0]["storage"]["persistent_exemplar_bytes"]
            ),
            "persistent_partition_metadata_bytes": int(
                np.mean(
                    [
                        r["storage"]["persistent_partition_metadata_bytes"]
                        for r in records
                    ]
                )
            ),
            "exemplar_vectors": int(records[0]["storage"]["exemplar_vectors"]),
        },
        "transient": {
            "closest_pair_matrix_bytes_total": int(
                records[0]["transient"]["closest_pair_matrix_bytes_total"]
            ),
            "closest_pair_matrix_bytes_max_bank": int(
                records[0]["transient"]["closest_pair_matrix_bytes_max_bank"]
            ),
            "peak_allocated_bytes_mean": (
                float(
                    np.mean(
                        [
                            r["boundaries"][-1]["peak_allocated_bytes"]
                            for r in records
                            if r["boundaries"][-1]["peak_allocated_bytes"]
                        ]
                    )
                )
                if any(
                    r["boundaries"][-1]["peak_allocated_bytes"] for r in records
                )
                else None
            ),
        },
        "false_positives": {
            "normal_image_scores": normal_scores,
            "defect_image_scores": defect_scores,
            "normal_image_mean": float(np.mean(normal_scores)),
            "normal_image_max": float(np.max(normal_scores)),
            "defect_image_min": float(np.min(defect_scores)),
            "separation_margin_mean": float(np.mean(margins)),
            "separation_margin_per_seed": {
                seed: float(
                    np.mean(
                        [
                            r["final"][c]["separation_margin"]
                            for r in grouped.get((arm, seed), [])
                            for c in CATEGORIES
                        ]
                    )
                )
                if grouped.get((arm, seed))
                else None
                for seed in (0, 1, 2)
            },
            "normal_patch_q": {
                key: float(
                    np.mean(
                        [
                            r["final"][c]["normal_patch_summary"][key]
                            for r in records
                            for c in CATEGORIES
                        ]
                    )
                )
                for key in ("q50", "q95", "q100", "mean")
            },
        },
    }


def mechanism_tables(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Origin composition per boundary and eviction flow per stage.

    This is the direct measurement of Condition A: how much of each earlier
    task's normal support is still in memory at every later boundary, and which
    task's exemplars each stage's replacements actually removed.
    """
    composition: list[dict[str, Any]] = []
    eviction_flow: list[dict[str, Any]] = []
    coverage_absolute: list[dict[str, Any]] = []
    for record in sorted(
        runs, key=lambda r: (r["arm"], r["order_seed"], r.get("draw") or 0)
    ):
        order = record["order"]
        task_ids = {category: index for index, category in enumerate(order)}
        run_key = record["run_id"]
        for boundary in record["boundaries"]:
            composition.append(
                {
                    "run": run_key,
                    "arm": record["arm"],
                    "order_seed": record["order_seed"],
                    "draw": record.get("draw"),
                    "stage": boundary["stage"],
                    "state_after": boundary["state_after"],
                    "shares": {
                        category: boundary["slots_by_origin"].get(
                            str(task_ids[category]), 0
                        )
                        / record["budget"]
                        for category in order
                    },
                }
            )
            coverage_absolute.append(
                {
                    "run": run_key,
                    "arm": record["arm"],
                    "order_seed": record["order_seed"],
                    "draw": record.get("draw"),
                    "stage": boundary["stage"],
                    "state_after": boundary["state_after"],
                    "d_own": {
                        category: boundary["coverage"][category]["d_own"]["mean"]
                        for category in order[: boundary["stage"] + 1]
                    },
                    "d_any": {
                        category: boundary["coverage"][category]["d_any"]["mean"]
                        for category in order[: boundary["stage"] + 1]
                    },
                }
            )
            eviction_flow.append(
                {
                    "run": run_key,
                    "arm": record["arm"],
                    "order_seed": record["order_seed"],
                    "draw": record.get("draw"),
                    "stage": boundary["stage"],
                    "inserted": boundary["events"]["inserted_by_origin"],
                    "evicted": boundary["events"]["evicted_by_origin"],
                }
            )
    # Category-resolved composition at the final boundary. Retention turns out to
    # be a property of the category and the arm, not of the category's position in
    # the stream, so this view is the interpretable one.
    by_category: dict[str, dict[str, list[float]]] = {}
    for record in runs:
        order = record["order"]
        task_ids = {category: index for index, category in enumerate(order)}
        boundary = record["boundaries"][-1]
        for category in order:
            shares = by_category.setdefault(record["arm"], {}).setdefault(category, [])
            shares.append(
                boundary["slots_by_origin"].get(str(task_ids[category]), 0)
                / record["budget"]
            )

    return {
        "composition_by_boundary": composition,
        "composition_by_category_final": by_category,
        "coverage_absolute_by_boundary": coverage_absolute,
        "eviction_flow_by_stage": eviction_flow,
    }


def occupancy_report(grouped) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ARMS:
        records = [r for (a, _), rs in grouped.items() if a == arm for r in rs]
        if not records:
            continue
        record = records[0]
        out[arm] = {
            "n_groups": record["occupancy"]["n_groups"],
            "partitioned": record["occupancy"]["partitioned"],
            "bins_are_disjoint": record["occupancy"]["bins_are_disjoint"],
            "sum_bins_is_total": record["occupancy"]["sum_bins_is_total"],
            "sum_bin_counts": int(
                sum(row["count"] for row in record["occupancy"]["bins"])
            ),
            "total_count": int(record["occupancy"]["total_count"]),
            "capacity_total": int(record["occupancy"]["capacity_total"]),
            "total_replacements": int(
                sum(row["replaced"] for row in record["occupancy"]["bins"])
            ),
            "quotas": record["quotas"],
        }
    return out


def retention_report(grouped) -> dict[str, Any]:
    """Retained share of each earlier task's exemplars at each later boundary."""
    out: dict[str, Any] = {}
    for arm in ARMS:
        per_seed = {}
        for seed in (0, 1, 2):
            records = grouped.get((arm, seed))
            if not records:
                continue
            per_seed[seed] = []
            for record in records:
                order = record["order"]
                task_ids = {c: i for i, c in enumerate(order)}
                rows = []
                for boundary in record["boundaries"]:
                    row = {"stage": boundary["stage"], "state_after": boundary["state_after"]}
                    for position, category in enumerate(order):
                        if position > boundary["stage"]:
                            continue
                        slots = boundary["slots_by_origin"].get(
                            str(task_ids[category]), 0
                        )
                        row[category] = slots / record["budget"]
                    rows.append(row)
                per_seed[seed].append(rows)
        out[arm] = per_seed
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AD-02 mechanism analysis")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    base = Path(args.out) if args.out else BASE
    if not base.is_absolute():
        base = (ROOT / base).resolve()

    runs = load_runs(base)
    if not runs:
        raise SystemExit("no AD-02 run records found")
    grouped = group_by_unit(runs)
    log(f"loaded {len(runs)} run records covering {len(grouped)} (arm, seed) units")

    replay_files = sorted(base.glob("replay_check_shard*.json"))
    replay: dict[str, Any] = {}
    for path in replay_files:
        replay.update(json.loads(path.read_text()))

    analysis: dict[str, Any] = {
        "runs": len(runs),
        "units": sorted(f"{arm}_seed{seed}" for arm, seed in grouped),
        "arms": {arm: aggregate_arm(grouped, arm) for arm in ARMS if any(
            a == arm for a, _ in grouped
        )},
        "occupancy": occupancy_report(grouped),
        "retention": retention_report(grouped),
        "contrasts": {},
        "category_contrasts": {},
        "fineness": {},
        "mediation": mediation_report(grouped),
        "mechanism": mechanism_tables(runs),
        "replay_check": replay,
    }

    for metric in ("i_auroc", "p_aupr_native"):
        for arm_a, arm_b in (
            ("S4", "G"),
            ("S2", "G"),
            ("S7", "G"),
            ("R4", "G"),
            ("S4", "R4"),
            ("S2", "S4"),
            ("S4", "S7"),
        ):
            analysis["contrasts"][f"{arm_a}-{arm_b}:{metric}"] = paired_contrast(
                grouped, arm_a, arm_b, metric
            )
            analysis["category_contrasts"][f"{arm_a}-{arm_b}:{metric}"] = (
                category_contrast(grouped, arm_a, arm_b, metric)
            )
        analysis["fineness"][metric] = fineness_trend(grouped, metric)

    # forgetting contrasts
    for arm_a, arm_b in (("S4", "G"), ("R4", "S4"), ("S4", "S2"), ("S4", "S7")):
        first = grouped.get((arm_a, 0))
        second = grouped.get((arm_b, 0))
        if not first or not second:
            continue
        per_seed = {}
        for seed in (0, 1, 2):
            a_records = grouped.get((arm_a, seed))
            b_records = grouped.get((arm_b, seed))
            if not a_records or not b_records:
                continue
            per_seed[seed] = float(
                np.mean([r["forgetting_i_auroc"]["fm"] for r in a_records])
                - np.mean([r["forgetting_i_auroc"]["fm"] for r in b_records])
            )
        analysis["contrasts"][f"{arm_a}-{arm_b}:forgetting_i_auroc"] = {
            "per_seed": per_seed,
            "mean": float(np.mean(list(per_seed.values()))) if per_seed else None,
        }

    oracle_files = sorted(base.glob("oracle_seed*.json"))
    if oracle_files:
        oracle = []
        for path in oracle_files:
            payload = json.loads(path.read_text())
            oracle.append(
                {
                    "order_seed": payload["order_seed"],
                    "per_state": payload["per_state"],
                    "forgetting_i_auroc": payload["forgetting_i_auroc"]["fm"],
                    "forgetting_p_aupr": payload["forgetting_p_aupr"]["fm"],
                    "macro_i_auroc": float(
                        np.mean(
                            [
                                row["i_auroc"]
                                for row in payload["per_state"]
                                if row["stage"] == 2
                            ]
                        )
                    ),
                    "macro_p_aupr_native": float(
                        np.mean(
                            [
                                row["p_aupr_native"]
                                for row in payload["per_state"]
                                if row["stage"] == 2
                            ]
                        )
                    ),
                }
            )
        analysis["budget_free_oracle"] = oracle

    target = base / "analysis.json"
    target.write_text(json.dumps(analysis, indent=2) + "\n")
    log(f"wrote {target.relative_to(ROOT)}")

    # ------------------------------------------------------------- console
    print("\n== arm summary ==")
    header = f"{'arm':>4} {'n':>3} {'I-AUROC':>9} {'P-AUPR':>9} {'fm_i':>8} {'fm_p':>8}"
    print(header)
    for arm, payload in analysis["arms"].items():
        print(
            f"{arm:>4} {payload['n_records']:>3} "
            f"{payload['macro_i_auroc_mean']:>9.4f} "
            f"{payload['macro_p_aupr_native_mean']:>9.4f} "
            f"{(payload['forgetting_i_auroc_mean'] or float('nan')):>8.4f} "
            f"{(payload['forgetting_p_aupr_mean'] or float('nan')):>8.4f}"
        )

    print("\n== paired contrasts ==")
    for key, payload in analysis["contrasts"].items():
        if payload.get("mean") is None:
            continue
        sd = payload.get("sd")
        print(
            f"{key:>34}  mean={payload['mean']:+.4f} "
            f"sd={(sd if sd is not None else float('nan')):.4f} "
            f"per_seed={ {k: round(v, 4) for k, v in payload['per_seed'].items()} }"
        )

    print("\n== fineness ==")
    for metric, payload in analysis["fineness"].items():
        slope = payload.get("slope_per_doubling_of_groups")
        print(
            f"{metric}: slope/doubling="
            f"{(slope if slope is not None else float('nan')):+.5f} "
            f"r2={(payload['r2'] if payload.get('r2') is not None else float('nan')):.3f} "
            f"monotone={payload['monotone_in_fineness']} "
            f"means={ {k: round(v, 4) for k, v in payload['mean_by_arm'].items()} }"
        )

    print("\n== retention at the final boundary (share of budget by origin) ==")
    retention = analysis["retention"]
    for arm, per_seed in retention.items():
        for seed, records in per_seed.items():
            for index, rows in enumerate(records):
                final = rows[-1]
                shares = {k: round(v, 3) for k, v in final.items() if k not in ("stage", "state_after")}
                print(f"{arm:>3} seed{seed} run{index} {final['state_after']:>9} after: {shares}")

    print("\n== origin composition at each boundary (share of budget) ==")
    for arm in ARMS:
        rows = [r for r in analysis["mechanism"]["composition_by_boundary"] if r["arm"] == arm]
        if not rows:
            continue
        for order_seed in (0, 1, 2):
            subset = [r for r in rows if r["order_seed"] == order_seed]
            if not subset:
                continue
            for stage in sorted({r["stage"] for r in subset}):
                stage_rows = [r for r in subset if r["stage"] == stage]
                mean_shares = {
                    key: float(np.mean([r["shares"][key] for r in stage_rows]))
                    for key in stage_rows[0]["shares"]
                }
                print(
                    f"{arm:>3} seed{order_seed} after {stage + 1}: "
                    + " ".join(f"{k}={v:.3f}" for k, v in mean_shares.items())
                )

    print("\n== final-boundary budget share by category (mean over seeds/draws) ==")
    for arm, payload in analysis["mechanism"]["composition_by_category_final"].items():
        print(
            f"{arm:>3} " + "  ".join(
                f"{category}={float(np.mean(values)):.3f} (n={len(values)})"
                for category, values in payload.items()
            )
        )

    print("\n== first-task absolute own-support distance d_own (mean) ==")
    for arm in ARMS:
        rows = [
            r
            for r in analysis["mechanism"]["coverage_absolute_by_boundary"]
            if r["arm"] == arm
        ]
        if not rows:
            continue
        for order_seed in (0, 1, 2):
            subset = [r for r in rows if r["order_seed"] == order_seed]
            if not subset:
                continue
            first = subset[0]["state_after"]
            series = []
            for stage in sorted({r["stage"] for r in subset}):
                stage_rows = [
                    r for r in subset if r["stage"] == stage and first in r["d_own"]
                ]
                if stage_rows:
                    series.append(
                        float(np.mean([r["d_own"][first] for r in stage_rows]))
                    )
            print(f"{arm:>3} seed{order_seed} {first:>9}: " + " -> ".join(f"{v:.3f}" for v in series))

    print("\n== mediation paths (S4 vs G) ==")
    for key, payload in analysis["mediation"]["paths"].items():
        print(
            f"{key:>46} a={payload['a_path_mean']:+.5f} "
            f"b={payload['b_path'].get('slope')} "
            f"product={payload['product_of_coefficients']}"
        )

    print("\n== replay check vs archived AD-01 ==")
    for key, payload in analysis["replay_check"].items():
        print(
            f"{key}: exact={payload['exact']} "
            f"final_mismatch={len(payload['final_mismatches'])} "
            f"per_state_mismatch={len(payload['per_state_mismatches'])}"
        )


if __name__ == "__main__":
    main()
