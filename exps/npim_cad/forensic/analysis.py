# exps/npim_cad/forensic/analysis.py
"""Read-only forensic analyses for EXP-NPIM-02.

Consumes the sealed NPIM-01 predictions, the frozen per-level reconstruction and
the previously exposed native DEV masks.  Produces per-level/counterfactual
metrics, cross-term/cancellation evidence, defect/background statistics,
capacity/effective-rank summaries, an NPIM-01 metric reproduction check and a
spatial audit.  No model or state is updated.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
import torch

from exps.npim_cad.forensic import spatial
from exps.npim_cad.forensic.decompose import PAIR_INDEX, mix_from_statistics
from exps.npim_cad.npim01_contract import (
    ARMS,
    BOUNDARIES,
    CATEGORIES,
    MVTEC,
    admitted_categories,
    load_contract,
    load_state_detectors,
    sha256_file,
)
from exps.npim_cad.npim01_metrics import (
    group_point_metrics,
    load_native_mask,
    native_scores,
)
from exps.task_p.p1_readout import count_metrics, image_metrics

NPIM01_RUN = Path(__file__).resolve().parents[3] / "results/hope_cad/npim_01/npim01_dev_20261021_cuda"
RESULTS = Path(__file__).resolve().parents[3] / "results/hope_cad/npim_02"
SEALED_SHA = "7c535937b2ff6e2ed5f6f6197a1f84650ab2a046063e5596d584f5cd4f6daa99"
METRICS_SHA = "713530f739516131abd0e5fa577782c610c418cda1e108956a18c715f9b27380"
QUANTILE = [0.5, 0.95, 0.99]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    return value


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_sealed():
    payload = torch.load(NPIM01_RUN / "predictions.pt", map_location="cpu", weights_only=False)
    if sha256_file(NPIM01_RUN / "predictions.pt") != SEALED_SHA:
        raise RuntimeError("NPIM-01 sealed predictions changed")
    metrics = json.loads((NPIM01_RUN / "metrics.json").read_text())
    if sha256_file(NPIM01_RUN / "metrics.json") != METRICS_SHA:
        raise RuntimeError("NPIM-01 metrics changed")
    return payload, metrics


def load_reconstruction(run_dir: Path):
    parity = json.loads((run_dir / "reconstruction" / "parity.json").read_text())
    if not parity.get("all_pass"):
        raise RuntimeError("reconstruction parity is not sealed as passing")
    per_level = torch.load(run_dir / "reconstruction" / "per_level_scores.pt", map_location="cpu", weights_only=False)
    cross = torch.load(run_dir / "reconstruction" / "cross_terms.pt", map_location="cpu", weights_only=False)
    capacity_inputs = torch.load(run_dir / "reconstruction" / "capacity_inputs.pt", map_location="cpu", weights_only=False)
    return parity, per_level, cross, capacity_inputs


def entry_lookup(per_level: Mapping[str, Any]) -> Dict[tuple[str, int, int], int]:
    arms = list(per_level["arms"])
    lookup: Dict[tuple[str, int, int], int] = {}
    for index in range(len(per_level["fused_scores"])):
        key = (arms[int(per_level["entry_arm"][index])], int(per_level["entry_boundary"][index]), int(per_level["entry_record"][index]))
        if key in lookup:
            raise RuntimeError(f"duplicate reconstructed entry {key}")
        lookup[key] = index
    if len(lookup) != 540:
        raise RuntimeError("reconstruction lookup is not 540 entries")
    return lookup


def diagnostic_kinds(arm: str, ready_row: torch.Tensor) -> List[tuple[str, int | None]]:
    if arm == "C1":
        return [("single", None)]
    kinds: List[tuple[str, int | None]] = [("fused", None)]
    for index, name in enumerate(("fast", "mid", "slow")):
        if bool(ready_row[index]):
            kinds.append((name, index))
    if bool(ready_row[0]) and bool(ready_row[1]):
        kinds.append(("fast_mid", None))
    return kinds


def patch_for_kind(per_level: Mapping[str, Any], cross: Mapping[str, Any], entry: int, kind: str, index: int | None) -> torch.Tensor:
    if kind in ("fused", "single"):
        return per_level["fused_scores"][entry]
    if kind in ("fast", "mid", "slow"):
        return per_level["level_scores"][entry, int(index)]
    if kind == "fast_mid":
        return mix_from_statistics(
            per_level["level_scores"][entry],
            cross["dot_products"][entry],
            torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32),
            d=768,
        )
    raise ValueError(f"unknown diagnostic kind: {kind}")


def fast_group_metrics(
    maps: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    labels: Sequence[int],
    image_scores: Sequence[float],
) -> Dict[str, Any]:
    """Exact tie-aware native metrics with one float32 unique/sort.

    Algebraically identical to the NPIM-01 point-metric routine, but avoids a
    float64 conversion of the full native maps.  Used only for forensic
    counterfactuals; the registered NPIM-01 fused metrics are reproduced with
    the original routine in ``reproduction_analysis``.
    """

    scores = np.concatenate([np.asarray(m, dtype=np.float32).ravel() for m in maps])
    mask = np.concatenate([np.asarray(k, dtype=bool).ravel() for k in masks])
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_mask = mask[order].astype(np.int64)
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    positive = np.add.reduceat(sorted_mask, starts)
    total = np.diff(np.r_[starts, scores.size])
    pixel = count_metrics(positive[::-1], total[::-1])
    image = image_metrics(np.asarray(labels), np.asarray(image_scores))
    normal = np.asarray(image_scores)[np.asarray(labels) == 0]
    anomaly = np.asarray(image_scores)[np.asarray(labels) == 1]
    return {
        "images": int(len(labels)),
        "normal_images": int((np.asarray(labels) == 0).sum()),
        "anomaly_images": int((np.asarray(labels) == 1).sum()),
        "positive_pixels": int(mask.sum()),
        "total_pixels": int(mask.size),
        "prevalence": float(mask.mean()),
        "pixel_ap": pixel["ap"],
        "pixel_auroc": pixel["auroc"],
        "pixel_degenerate": pixel["degenerate"],
        "image_ap": image["ap"],
        "image_auroc": image["auroc"],
        "image_degenerate": image["degenerate"],
        "normal_image_score_mean": float(normal.mean()) if len(normal) else None,
        "normal_image_score_p95": float(np.quantile(normal, 0.95)) if len(normal) else None,
        "normal_image_score_max": float(normal.max()) if len(normal) else None,
        "anomaly_image_score_min": float(anomaly.min()) if len(anomaly) else None,
        "anomaly_image_score_max": float(anomaly.max()) if len(anomaly) else None,
        "normal_minus_anomaly_gap": (float(normal.mean() - anomaly.mean()) if len(normal) and len(anomaly) else None),
    }


def _stats(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return {"n": 0}
    quantiles = np.quantile(values, [0.5, 0.95, 0.99])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(quantiles[0]),
        "p95": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "max": float(values.max()),
        "min": float(values.min()),
    }


def defect_statistics(
    maps: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    labels: Sequence[int],
    rows: Sequence[Any],
    image_scores: Sequence[float],
) -> Dict[str, Any]:
    labels = list(labels)
    defect_values, background_values, normal_values = [], [], []
    image_max = np.asarray(image_scores, dtype=np.float64)
    normal_mask = np.asarray(labels) == 0
    anomaly_mask = ~normal_mask
    for index, (native, mask) in enumerate(zip(maps, masks)):
        native = np.asarray(native, dtype=np.float64)
        if labels[index] == 0:
            normal_values.append(native.ravel())
        else:
            defect_values.append(native[mask])
            background_values.append(native[~mask])
    defect_values = np.concatenate(defect_values) if defect_values else np.zeros(0)
    background_values = np.concatenate(background_values) if background_values else np.zeros(0)
    normal_values = np.concatenate(normal_values) if normal_values else np.zeros(0)
    normal_median = float(np.median(normal_values)) if normal_values.size else None
    normal_p95 = float(np.quantile(normal_values, 0.95)) if normal_values.size else None
    anomaly_indices = np.flatnonzero(anomaly_mask)
    miss_median = float(np.mean(image_max[anomaly_indices] < normal_median)) if normal_median is not None and anomaly_indices.size else None
    miss_p95 = float(np.mean(image_max[anomaly_indices] < normal_p95)) if normal_p95 is not None and anomaly_indices.size else None
    top_fraction, top_mass = [], []
    for index in anomaly_indices:
        native = np.asarray(maps[index])
        mask = np.asarray(masks[index], dtype=bool)
        flat = native.ravel()
        k = max(1, int(0.01 * flat.size))
        top = np.argpartition(flat, -k)[-k:]
        top_fraction.append(float(mask.ravel()[top].mean()))
        total = float(flat.sum())
        top_mass.append(float(flat[top].sum() / total) if total > 0 else None)
    per_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for index in anomaly_indices:
        row = rows[index]
        defect_type = Path(row.mask_path).parent.name if row.mask_path else "unknown"
        native = np.asarray(maps[index], dtype=np.float64)
        mask = np.asarray(masks[index], dtype=bool)
        per_type[defect_type].append({
            "defect_mean": float(native[mask].mean()),
            "background_mean": float(native[~mask].mean()),
            "contrast": float(native[mask].mean() - native[~mask].mean()),
            "image_max": float(native.max()),
        })
    type_summary = {}
    for defect_type, entries in sorted(per_type.items()):
        type_summary[defect_type] = {
            "count": len(entries),
            "defect_mean_mean": float(np.mean([entry["defect_mean"] for entry in entries])),
            "background_mean_mean": float(np.mean([entry["background_mean"] for entry in entries])),
            "contrast_mean": float(np.mean([entry["contrast"] for entry in entries])),
            "image_max_mean": float(np.mean([entry["image_max"] for entry in entries])),
        }
    normal_index = int(np.argmax(np.where(normal_mask, image_max, -np.inf))) if normal_mask.any() else None
    anomaly_failure_index = int(np.argmin(np.where(anomaly_mask, image_max, np.inf))) if anomaly_mask.any() else None
    anomaly_easy_index = int(np.argmax(np.where(anomaly_mask, image_max, -np.inf))) if anomaly_mask.any() else None
    return {
        "defect": _stats(defect_values),
        "background_in_anomalous": _stats(background_values),
        "normal": _stats(normal_values),
        "contrast_defect_minus_background": float(defect_values.mean() - background_values.mean())
        if defect_values.size and background_values.size else None,
        "ratio_defect_over_background": float(defect_values.mean() / max(background_values.mean(), 1e-12))
        if defect_values.size and background_values.size else None,
        "normal_image_score_median": normal_median,
        "normal_image_score_p95": normal_p95,
        "image_miss_below_normal_median": miss_median,
        "image_miss_below_normal_p95": miss_p95,
        "anomalous_images": int(anomaly_indices.size),
        "top1pct_in_mask_mean": float(np.mean(top_fraction)) if top_fraction else None,
        "top1pct_score_mass_mean": float(np.mean([x for x in top_mass if x is not None])) if top_mass else None,
        "defect_area_mean": float(np.mean([np.asarray(masks[i]).sum() for i in anomaly_indices])) if anomaly_indices.size else None,
        "by_defect_type": type_summary,
        "hard_normal": {
            "relative_path": rows[normal_index].relative_path,
            "image_score": float(image_max[normal_index]),
        } if normal_index is not None else None,
        "hard_anomaly_failure": {
            "relative_path": rows[anomaly_failure_index].relative_path,
            "image_score": float(image_max[anomaly_failure_index]),
        } if anomaly_failure_index is not None else None,
        "easy_anomaly": {
            "relative_path": rows[anomaly_easy_index].relative_path,
            "image_score": float(image_max[anomaly_easy_index]),
        } if anomaly_easy_index is not None else None,
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).ravel()[::8]
    b = np.asarray(b, dtype=np.float64).ravel()[::8]
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    value = float(spearmanr(a, b).statistic)
    return value if math.isfinite(value) else None


def levels_analysis(run_dir: Path) -> tuple[List[Dict[str, Any]], Dict[tuple[int, str, str], Dict[str, Any]]]:
    contract = load_contract()
    sealed, metrics = load_sealed()
    parity, per_level, cross, capacity_inputs = load_reconstruction(run_dir)
    lookup = entry_lookup(per_level)
    masks = [load_native_mask(MVTEC, record) for record in contract.records]
    by_category = {category: [record for record in contract.records if record.category == category] for category in CATEGORIES}
    levels: List[Dict[str, Any]] = []
    fused_groups: Dict[tuple[int, str, str], Dict[str, Any]] = {}
    for boundary in BOUNDARIES:
        for category in admitted_categories(boundary):
            rows = by_category[category]
            labels = [record.label for record in rows]
            group_masks = [masks[record.index] for record in rows]
            for arm in ARMS:
                entries = [lookup[(arm, boundary, record.index)] for record in rows]
                ready_row = per_level["ready"][entries[0]]
                kinds = diagnostic_kinds(arm, ready_row)
                maps_by_kind: Dict[str, List[np.ndarray]] = {}
                for kind, level_index in kinds:
                    patches = [patch_for_kind(per_level, cross, entry, kind, level_index) for entry in entries]
                    maps = [native_scores(patch.numpy(), record.hw) for patch, record in zip(patches, rows)]
                    maps_by_kind[kind] = maps
                    image_scores = [float(patch.max()) for patch in patches]
                    group = fast_group_metrics(maps, group_masks, labels, image_scores)
                    stats = defect_statistics(maps, group_masks, labels, rows, image_scores)
                    record = {
                        "boundary": boundary,
                        "arm": arm,
                        "category": category,
                        "kind": kind,
                        **group,
                        "defect_stats": stats,
                    }
                    if kind == "fused":
                        record["spearman_fused_vs_levels"] = {
                            level_kind: _spearman(
                                np.concatenate(maps_by_kind["fused"]),
                                np.concatenate(maps_by_kind[level_kind]),
                            )
                            for _kind, _index in kinds
                            if _kind in ("fast", "mid", "slow") and _kind in maps_by_kind
                        }
                    levels.append(record)
                    if kind == "fused":
                        fused_groups[(boundary, arm, category)] = group
            print(f"NPIM-02 levels {boundary}/{category}", flush=True)
    return levels, fused_groups


def cross_analysis(run_dir: Path) -> List[Dict[str, Any]]:
    contract = load_contract()
    parity, per_level, cross, capacity_inputs = load_reconstruction(run_dir)
    lookup = entry_lookup(per_level)
    masks = [load_native_mask(MVTEC, record) for record in contract.records]
    by_category = {category: [record for record in contract.records if record.category == category] for category in CATEGORIES}
    output: List[Dict[str, Any]] = []
    for boundary in BOUNDARIES:
        for category in admitted_categories(boundary):
            rows = by_category[category]
            labels = [record.label for record in rows]
            group_masks = [masks[record.index] for record in rows]
            for arm in ("C2", "C3"):
                entries = [lookup[(arm, boundary, record.index)] for record in rows]
                ready = per_level["ready"][entries]
                weights = per_level["weights"][entries]
                level_scores = per_level["level_scores"][entries].to(torch.float64)
                dots = cross["dot_products"][entries].to(torch.float64)
                fused = per_level["fused_scores"][entries].to(torch.float64)
                self_terms = torch.zeros_like(fused)
                for index in range(3):
                    active = ready[:, index]
                    self_terms[active] += weights[active, index].unsqueeze(1).square() * level_scores[active, index]
                cross_total = torch.zeros_like(fused)
                pair_stats = {}
                for pair_offset, (left, right) in enumerate(PAIR_INDEX):
                    active = ready[:, left] & ready[:, right]
                    if not active.any():
                        continue
                    pair_cross = (2.0 / 768.0) * weights[:, left].unsqueeze(1) * weights[:, right].unsqueeze(1) * dots[:, pair_offset]
                    cross_total[active] += pair_cross[active]
                    denominator = 768.0 * torch.sqrt(level_scores[:, left].clamp_min(0) * level_scores[:, right].clamp_min(0))
                    cosine = dots[:, pair_offset] / denominator.clamp_min(1e-12)
                    pair_stats[f"{left}-{right}"] = {
                        "dot": _stats(dots[active].numpy()),
                        "cosine": _stats(cosine[active].numpy()),
                        "fraction_negative_dot": float((dots[active] < 0).float().mean()),
                    }
                identity_error = float((self_terms + cross_total - fused).abs().max())
                fused_maps = [native_scores(patch.numpy(), record.hw) for patch, record in zip(fused, rows)]
                self_maps = [native_scores(patch.numpy(), record.hw) for patch, record in zip(self_terms, rows)]
                cross_maps = [native_scores(patch.numpy(), record.hw) for patch, record in zip(cross_total, rows)]
                fused_metrics = fast_group_metrics(
                    fused_maps, group_masks, labels, [float(p.max()) for p in fused]
                )
                self_metrics = fast_group_metrics(
                    self_maps, group_masks, labels, [float(p.max()) for p in self_terms]
                )
                defect_values, background_values, normal_values = [], [], []
                for index, (fusion, self_map, cross_map) in enumerate(zip(fused_maps, self_maps, cross_maps)):
                    if labels[index] == 0:
                        normal_values.append((float(cross_map.mean()), float(fusion.mean()), float(self_map.mean())))
                    else:
                        defect_values.append((float(cross_map[group_masks[index]].mean()), float(fusion[group_masks[index]].mean()), float(self_map[group_masks[index]].mean())))
                        background_values.append((float(cross_map[~group_masks[index]].mean()), float(fusion[~group_masks[index]].mean()), float(self_map[~group_masks[index]].mean())))
                cross_negative_fraction = float((cross_total < 0).float().mean())
                output.append({
                    "boundary": boundary,
                    "arm": arm,
                    "category": category,
                    "weights": weights[0].tolist(),
                    "identity_max_abs": identity_error,
                    "fused_pixel_ap": fused_metrics["pixel_ap"],
                    "self_pixel_ap": self_metrics["pixel_ap"],
                    "self_minus_fused_pixel_ap": (self_metrics["pixel_ap"] - fused_metrics["pixel_ap"])
                    if self_metrics["pixel_ap"] is not None and fused_metrics["pixel_ap"] is not None else None,
                    "fused_image_auroc": fused_metrics["image_auroc"],
                    "self_image_auroc": self_metrics["image_auroc"],
                    "cross_fraction_negative_all": cross_negative_fraction,
                    "self_mean_all": float(self_terms.mean()),
                    "cross_mean_all": float(cross_total.mean()),
                    "fused_mean_all": float(fused.mean()),
                    "cross_to_fused_ratio": float(cross_total.mean() / fused.mean()) if abs(float(fused.mean())) > 1e-12 else None,
                    "defect_cross_mean": float(np.mean([row[0] for row in defect_values])) if defect_values else None,
                    "defect_fused_mean": float(np.mean([row[1] for row in defect_values])) if defect_values else None,
                    "defect_self_mean": float(np.mean([row[2] for row in defect_values])) if defect_values else None,
                    "background_cross_mean": float(np.mean([row[0] for row in background_values])) if background_values else None,
                    "background_fused_mean": float(np.mean([row[1] for row in background_values])) if background_values else None,
                    "background_self_mean": float(np.mean([row[2] for row in background_values])) if background_values else None,
                    "normal_cross_mean": float(np.mean([row[0] for row in normal_values])) if normal_values else None,
                    "normal_fused_mean": float(np.mean([row[1] for row in normal_values])) if normal_values else None,
                    "normal_self_mean": float(np.mean([row[2] for row in normal_values])) if normal_values else None,
                    "pair_stats": pair_stats,
                })
    return output


def reproduction_analysis(run_dir: Path) -> Dict[str, Any]:
    contract = load_contract()
    sealed, metrics = load_sealed()
    lookup = entry_lookup(torch.load(run_dir / "reconstruction" / "per_level_scores.pt", map_location="cpu", weights_only=False))
    masks = [load_native_mask(MVTEC, record) for record in contract.records]
    by_category = {category: [record for record in contract.records if record.category == category] for category in CATEGORIES}
    rows_out = []
    for stored in metrics["groups"]:
        boundary, arm, category = stored["boundary"], stored["arm"], stored["category"]
        rows = by_category[category]
        entries = [lookup[(arm, boundary, record.index)] for record in rows]
        maps = [native_scores(sealed["patch_scores"][entry].numpy(), record.hw) for entry, record in zip(entries, rows)]
        group = group_point_metrics(
            maps,
            [masks[record.index] for record in rows],
            [record.label for record in rows],
            [float(sealed["image_score"][entry]) for entry in entries],
        )
        fields = ("pixel_ap", "pixel_auroc", "image_ap", "image_auroc", "normal_image_score_mean", "normal_image_score_p95", "positive_pixels", "prevalence")
        deltas = {field: abs(float(group[field]) - float(stored[field])) for field in fields if group[field] is not None and stored[field] is not None}
        rows_out.append({
            "boundary": boundary,
            "arm": arm,
            "category": category,
            "max_abs_delta": max(deltas.values()) if deltas else None,
            "deltas": deltas,
            "reproduced": all(value <= 1e-12 for value in deltas.values()),
        })
    return {
        "schema": "npim02_metric_reproduction_v1",
        "source_metrics_sha256": METRICS_SHA,
        "rows": rows_out,
        "max_abs_delta": max(row["max_abs_delta"] for row in rows_out if row["max_abs_delta"] is not None),
        "all_reproduced": all(row["reproduced"] for row in rows_out),
    }


def _rank(values: np.ndarray, tolerance: float = 1e-8) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    effective = float(total * total / max(float((values * values).sum()), 1e-30))
    return {
        "rank_tolerance": int((values > tolerance * max(float(values.max()), 1e-30)).sum()),
        "nonzero": int((values > 0).sum()),
        "effective_rank_participation": effective,
        "trace": total,
        "top1_fraction": float(values[0] / total) if total > 0 else None,
        "singular_values": values[:min(20, len(values))].tolist(),
    }


def capacity_analysis(run_dir: Path, parity: Mapping[str, Any], capacity_inputs: Mapping[str, Any]) -> Dict[str, Any]:
    per_level = torch.load(run_dir / "reconstruction" / "per_level_scores.pt", map_location="cpu", weights_only=False)
    detectors = load_state_detectors("cpu")
    weight_summary = {}
    for (arm, boundary), detector in detectors.items():
        rows = []
        for index, level in enumerate(detector.levels):
            matrix = level.W.detach().to(torch.float64).numpy()
            singular = np.linalg.svd(matrix, compute_uv=False)
            rows.append({"level": ("fast", "mid", "slow")[index], **_rank(singular)})
        indices, weights = None, None
        from exps.npim_cad.forensic.decompose import ready_weights, effective_W
        indices, weights = ready_weights(detector)
        W_eff = effective_W([detector.levels[i].W for i in indices], weights).detach().to(torch.float64).numpy()
        singular_eff = np.linalg.svd(W_eff, compute_uv=False)
        weight_summary[f"{arm}@{boundary}"] = {
            "levels": rows,
            "effective_W": _rank(singular_eff),
            "ready_levels": indices,
            "weights": weights.tolist(),
        }
    # Normal-fit and per-level residual energy from stored scores/energies.
    y_energy = capacity_inputs["y_energy"]
    contract = load_contract()
    lookup = entry_lookup(per_level)
    fit = {}
    for arm in ARMS:
        fit[arm] = {}
        for boundary in BOUNDARIES:
            normal_records = [record for record in contract.records if record.label == 0 and record.category in admitted_categories(boundary)]
            entries = [lookup[(arm, boundary, record.index)] for record in normal_records]
            residual_energy = 768.0 * sum(float(per_level["fused_scores"][entry].sum()) for entry in entries)
            target_energy = sum(float(y_energy[record.index]) for record in normal_records)
            per_level_energy = {}
            for level_index, level_name in enumerate(("fast", "mid", "slow")):
                ready = [entry for entry in entries if bool(per_level["ready"][entry, level_index])]
                per_level_energy[level_name] = (
                    768.0 * sum(float(per_level["level_scores"][entry, level_index].sum()) for entry in ready)
                    if ready else None
                )
            fit[arm][str(boundary)] = {
                "normal_images": len(normal_records),
                "target_energy": target_energy,
                "fused_residual_energy": residual_energy,
                "normal_r2": 1.0 - residual_energy / max(target_energy, 1e-30),
                "per_level_residual_energy": per_level_energy,
                "per_level_r2": {
                    name: (1.0 - energy / max(target_energy, 1e-30) if energy is not None else None)
                    for name, energy in per_level_energy.items()
                },
            }
    key_summary = {}
    for arm, stats in capacity_inputs["key_stats"].items():
        count = int(stats["count"])
        ktk = stats["ktk"].double().numpy()
        ksum = stats["ksum"].double().numpy()
        covariance = ktk / max(count, 1) - np.outer(ksum / max(count, 1), ksum / max(count, 1))
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
        key_summary[arm] = {
            "rank_dimension": int(covariance.shape[0]),
            "normal_images": count,
            **_rank(np.maximum(eigenvalues, 0.0)),
        }
    yty = capacity_inputs["yty"].double().numpy()
    ysum = capacity_inputs["ysum"].double().numpy()
    y_count = int(capacity_inputs["normal_count"])
    y_covariance = yty / max(y_count, 1) - np.outer(ysum / max(y_count, 1), ysum / max(y_count, 1))
    y_eigenvalues = np.linalg.eigvalsh(y_covariance)[::-1]
    drift = {}
    for arm in ARMS:
        matrices = {boundary: detectors[(arm, boundary)].projection.detach().to(torch.float64).numpy() for boundary in BOUNDARIES}
        drift[arm] = {
            "projection_equal_across_boundaries": all(
                np.array_equal(matrices[100], matrices[boundary]) for boundary in BOUNDARIES
            ),
        }
    return {
        "schema": "npim02_capacity_v1",
        "weights": weight_summary,
        "normal_fit": fit,
        "keys": key_summary,
        "target_Y": {
            "normal_images": y_count,
            "dimension": 768,
            **_rank(np.maximum(y_eigenvalues, 0.0)),
        },
        "projection_drift": drift,
        "note": "rank/spectrum diagnostics are descriptive; no causal capacity claim",
    }


def analyze(run_id: str) -> Dict[str, Any]:
    run_dir = RESULTS / run_id
    started = time.perf_counter()
    parity, per_level, cross, capacity_inputs = load_reconstruction(run_dir)
    levels, fused_groups = levels_analysis(run_dir)
    cross_rows = cross_analysis(run_dir)
    reproduction = reproduction_analysis(run_dir)
    capacity = capacity_analysis(run_dir, parity, capacity_inputs)
    sealed, metrics = load_sealed()
    geometric = spatial.geometric_audit()
    qualitative = spatial.qualitative_audit(NPIM01_RUN / "qualitative", sealed)
    analysis_dir = run_dir / "analysis"
    write_json_exclusive(analysis_dir / "levels.json", {"schema": "npim02_levels_v1", "rows": levels})
    write_json_exclusive(analysis_dir / "cross_terms.json", {"schema": "npim02_cross_terms_analysis_v1", "rows": cross_rows})
    write_json_exclusive(analysis_dir / "capacity.json", capacity)
    write_json_exclusive(analysis_dir / "reproduction.json", reproduction)
    write_json_exclusive(analysis_dir / "spatial_audit.json", {
        "schema": "npim02_spatial_audit_v1",
        "geometric": geometric,
        "qualitative": qualitative,
    })
    summary = {
        "schema": "npim02_analysis_summary_v1",
        "run_id": run_id,
        "created_utc": utc_now(),
        "levels": len(levels),
        "cross_groups": len(cross_rows),
        "reproduction": reproduction,
        "spatial": {"geometric_all_ok": geometric["all_cases_ok"], "qualitative": {
            "case_count": qualitative["case_count"],
            "sealed_map_parity_max_abs": qualitative["sealed_map_parity_max_abs"],
            "sealed_map_parity_ok": qualitative["sealed_map_parity_ok"],
            "identity_best_count": qualitative["identity_best_count"],
        }},
        "resource": {"analysis_seconds": time.perf_counter() - started},
    }
    write_json_exclusive(analysis_dir / "summary.json", summary)
    write_json_exclusive(analysis_dir / "E_ANALYSIS_COMPLETE.json", {
        "schema": "npim02_analysis_complete_v1",
        "created_utc": utc_now(),
        "levels_sha256": sha256_file(analysis_dir / "levels.json"),
        "cross_sha256": sha256_file(analysis_dir / "cross_terms.json"),
        "capacity_sha256": sha256_file(analysis_dir / "capacity.json"),
        "reproduction_sha256": sha256_file(analysis_dir / "reproduction.json"),
        "spatial_sha256": sha256_file(analysis_dir / "spatial_audit.json"),
    })
    return summary
