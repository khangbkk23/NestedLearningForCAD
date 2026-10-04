# exps/hope_gate3.py
"""Experiment-only long-horizon normal-stream diagnostics."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from exps.hope_retention_stabilization import (
    _state_metrics,
    _stream_state_summary,
    clone_cms_from_state,
    clone_smt_from_state,
    probe_objective,
)
from exps.hope_update_stabilization import EPS, UpdateMapping, run_update_smt

CLASS_ORDER = ("bottle", "carpet", "grid", "toothbrush", "transistor")
EVENTS_PER_CLASS = 40
RANK_EPS = 1e-12
CHECKPOINTS = (40, 80, 120, 160, 200)


def load_200_stream(feature_root: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(CLASS_ORDER):
        payload = torch.load(feature_root / f"class_{class_name}.pt", map_location="cpu", weights_only=False)
        patches = payload["patches"]
        paths = payload["relative_paths"]
        if payload.get("class_name") != class_name or patches.shape != (40, 784, 768) or patches.dtype != torch.float32:
            raise ValueError(f"invalid feature shard for {class_name}")
        if len(paths) != 40 or paths != sorted(paths) or not all(path.startswith(f"{class_name}/train/good/") for path in paths):
            raise ValueError(f"invalid normal path order for {class_name}")
        if not torch.isfinite(patches).all().item():
            raise ValueError(f"non-finite features for {class_name}")
        for index, path in enumerate(paths):
            rows.append({"patches": patches[index:index + 1].contiguous(), "class_name": class_name, "relative_path": path, "class_index": index})
    if len(rows) != 200:
        raise ValueError(f"expected 200 records, got {len(rows)}")
    return rows


def full_geometry(value: torch.Tensor) -> dict[str, float | bool]:
    matrix = value.detach().to(device="cpu", dtype=torch.float64)
    if matrix.ndim == 3:
        if matrix.shape[0] != 1:
            raise ValueError("one image required")
        matrix = matrix[0]
    if matrix.ndim != 2 or matrix.shape[0] != 784:
        raise ValueError(f"expected [784,D], got {tuple(matrix.shape)}")
    finite = bool(torch.isfinite(matrix).all().item())
    if not finite:
        return {key: float("nan") for key in ("norm", "rms", "centered_variance", "effective_rank", "top1_energy_fraction", "pairwise_cosine_mean", "pairwise_cosine_std")} | {"finite": False}
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    probabilities = singular / (singular.sum() + RANK_EPS)
    effective_rank = torch.exp(-(probabilities * torch.log(probabilities + RANK_EPS)).sum())
    energy = singular.square()
    indices = torch.arange(min(256, matrix.shape[0]))
    pairwise = F.cosine_similarity(matrix[indices], matrix[torch.roll(indices, -1)], dim=-1, eps=RANK_EPS)
    return {
        "norm": float(matrix.norm().item()),
        "rms": float(matrix.square().mean().sqrt().item()),
        "centered_variance": float(centered.square().mean().item()),
        "effective_rank": float(effective_rank.item()),
        "top1_energy_fraction": float((energy[0] / (energy.sum() + RANK_EPS)).item()),
        "pairwise_cosine_mean": float(pairwise.mean().item()),
        "pairwise_cosine_std": float(pairwise.std(unbiased=False).item()),
        "finite": True,
    }


def cosine_flat(left: torch.Tensor, right: torch.Tensor) -> float:
    l = left.detach().double().reshape(-1)
    r = right.detach().double().reshape(-1)
    denom = float(l.norm().item() * r.norm().item())
    return float(torch.dot(l, r).item() / denom) if denom > RANK_EPS else float("nan")


def relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    ref = right.detach().double()
    return float((left.detach().double() - ref).norm().item() / (ref.norm().item() + RANK_EPS))


def anchor_stream(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    anchors: dict[str, list[dict[str, Any]]] = {}
    for class_index, class_name in enumerate(CLASS_ORDER):
        start = class_index * EVENTS_PER_CLASS
        anchors[class_name] = [records[start + local - 1] for local in (1, 10, 20, 40) if start + local - 1 < len(records)]
    return anchors


def aggregate_terms(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    selected = [row for row in rows if row.get("memory") == "memory"]
    if not selected:
        return {key: float("nan") for key in ("rank_update_norm", "surprise_update_norm", "rank_update_state_ratio", "surprise_update_state_ratio", "max_rank_state_ratio", "max_surprise_state_ratio", "surprise_rank_norm_ratio")}
    rank = torch.tensor([float(row["rank_update_norm"]) for row in selected], dtype=torch.float64)
    surprise = torch.tensor([float(row["surprise_update_norm"]) for row in selected], dtype=torch.float64)
    state = torch.tensor([float(row["state_norm"]) for row in selected], dtype=torch.float64)
    rank_norm = float(rank.square().sum().sqrt().item())
    surprise_norm = float(surprise.square().sum().sqrt().item())
    state_norm = float(state.square().sum().sqrt().item())
    return {
        "rank_update_norm": rank_norm,
        "surprise_update_norm": surprise_norm,
        "rank_update_state_ratio": rank_norm / (state_norm + EPS),
        "surprise_update_state_ratio": surprise_norm / (state_norm + EPS),
        "max_rank_state_ratio": max(float(row["max_rank_state_ratio"]) for row in selected),
        "max_surprise_state_ratio": max(float(row["max_surprise_state_ratio"]) for row in selected),
        "surprise_rank_norm_ratio": surprise_norm / (rank_norm + EPS),
    }


def snapshot_module(module: Any) -> dict[str, Any]:
    return {key: value.detach().clone() if isinstance(value, torch.Tensor) else value for key, value in module.state_dict().items()}


def tensor_state_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.keys() == right.keys() and all((torch.equal(left[key], right[key]) if isinstance(left[key], torch.Tensor) else left[key] == right[key]) for key in left)


def evaluate_anchor(smt: Any, cms: Any, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    before = {"smt": snapshot_module(smt), "cms": snapshot_module(cms)}
    with torch.no_grad():
        smt_out = smt.forward(image, update=False).memory_prediction.detach().clone()
        hope_out = cms.forward(smt_out).detach().clone()
    after = {"smt": snapshot_module(smt), "cms": snapshot_module(cms)}
    if not tensor_state_equal(before["smt"], after["smt"]) or not tensor_state_equal(before["cms"], after["cms"]):
        raise AssertionError("read-only anchor evaluation mutated state")
    return smt_out, hope_out


def compare_anchor(current: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    current_geometry = full_geometry(current)
    reference_geometry = full_geometry(reference)
    return {
        "cosine": cosine_flat(current, reference),
        "relative_l2": relative_l2(current, reference),
        "rms_ratio": float(current_geometry["rms"] / (float(reference_geometry["rms"]) + RANK_EPS)),
        "centered_variance_ratio": float(current_geometry["centered_variance"] / (float(reference_geometry["centered_variance"]) + RANK_EPS)),
        "effective_rank_ratio": float(current_geometry["effective_rank"] / (float(reference_geometry["effective_rank"]) + RANK_EPS)),
    }


def run_long_candidate(records: Sequence[Mapping[str, Any]], mapping: UpdateMapping, smt_state: Mapping[str, Any], cms_state: Mapping[str, Any], device: str, *, max_events: int = 200, progress: bool = False) -> dict[str, Any]:
    dim = int(records[0]["patches"].shape[-1])
    smt = clone_smt_from_state(smt_state, dim, device)
    cms = clone_cms_from_state(cms_state, dim, device)
    initial = {name: memory.weight.detach().clone() for name, memory in smt.memories.items()}
    rows: list[dict[str, Any]] = []
    checkpoints: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    previous_class: str | None = None
    for event_id, record in enumerate(records[:max_events], start=1):
        image = record["patches"].to(device)
        with torch.no_grad():
            pre_smt = smt.forward(image, update=False).memory_prediction.detach()
            pre_hope = cms.forward(pre_smt).detach()
        term_summaries: list[dict[str, Any]] = []
        causal_smt, trace, _ = run_update_smt(
            smt, image, mapping, event_id=event_id, capture_trace=True,
            trace_points=frozenset({784}), term_summary_sink=term_summaries,
            capture_terms=False, capture_term_names=frozenset({"memory"}),
        )
        trace_final = trace[-1] if trace else {}
        failed_token = trace_final.get("point") if trace_final.get("kind") == "non-finite-failure" else None
        complete = causal_smt.shape == image.shape and bool(torch.isfinite(causal_smt).all().item()) and failed_token is None
        causal_hope = torch.empty(0, device=image.device, dtype=image.dtype)
        cms_result = None
        error = None
        if complete:
            try:
                cms_result = cms.commit_image(causal_smt, [probe_objective] * cms.K)
                causal_hope = cms_result.output.detach()
                complete = causal_hope.shape == image.shape and bool(torch.isfinite(causal_hope).all().item())
            except Exception as exc:
                complete = False
                error = f"CMS event {event_id}: {type(exc).__name__}: {exc}"
        else:
            error = f"SMT event {event_id} became non-finite or incomplete"
        smt_geometry = full_geometry(causal_smt) if causal_smt.numel() else {key: float("nan") for key in ("norm", "rms", "centered_variance", "effective_rank", "top1_energy_fraction", "pairwise_cosine_mean", "pairwise_cosine_std")} | {"finite": False}
        hope_geometry = full_geometry(causal_hope) if complete else {key: float("nan") for key in ("norm", "rms", "centered_variance", "effective_rank", "top1_energy_fraction", "pairwise_cosine_mean", "pairwise_cosine_std")} | {"finite": False}
        pre_smt_geometry = full_geometry(pre_smt)
        pre_hope_geometry = full_geometry(pre_hope)
        state = _stream_state_summary(smt, cms)
        memory_metrics = _state_metrics(smt, initial)
        term_metrics = aggregate_terms(term_summaries)
        row: dict[str, Any] = {
            "candidate": mapping.name, "event_id": event_id, "class_name": record["class_name"], "relative_path": record["relative_path"], "class_index": record["class_index"], "previous_class": previous_class,
            "is_class_boundary": previous_class is not None and previous_class != record["class_name"], "complete_event": complete, "finite": bool(complete and state["finite_state"]), "error": error, "smt_failure_token": failed_token,
            "due_cms_levels": tuple(cms_result.due_levels) if cms_result is not None else tuple(),
            "alpha_product_observed": trace_final.get("alpha_product_float64"), "alpha_log_product_observed": trace_final.get("alpha_log_product_float64"),
            "raw_eta_min": trace_final.get("raw_eta_min"), "raw_eta_q25": trace_final.get("raw_eta_q25"), "raw_eta_median": trace_final.get("raw_eta_median"), "raw_eta_mean": trace_final.get("raw_eta_mean"), "raw_eta_q75": trace_final.get("raw_eta_q75"), "raw_eta_max": trace_final.get("raw_eta_max"),
            "eta_min": trace_final.get("eta_min"), "eta_q25": trace_final.get("eta_q25"), "eta_median": trace_final.get("eta_median"), "eta_mean": trace_final.get("eta_mean"), "eta_q75": trace_final.get("eta_q75"), "eta_max": trace_final.get("eta_max"),
            "raw_alpha_min": trace_final.get("raw_alpha_min"), "raw_alpha_q25": trace_final.get("raw_alpha_q25"), "raw_alpha_median": trace_final.get("raw_alpha_median"), "raw_alpha_mean": trace_final.get("raw_alpha_mean"), "raw_alpha_q75": trace_final.get("raw_alpha_q75"), "raw_alpha_max": trace_final.get("raw_alpha_max"),
            "alpha_min": trace_final.get("alpha_min"), "alpha_q25": trace_final.get("alpha_q25"), "alpha_median": trace_final.get("alpha_median"), "alpha_mean": trace_final.get("alpha_mean"), "alpha_q75": trace_final.get("alpha_q75"), "alpha_max": trace_final.get("alpha_max"),
            "smt_rms": smt_geometry["rms"], "smt_centered_variance": smt_geometry["centered_variance"], "smt_effective_rank": smt_geometry["effective_rank"], "smt_top1_energy_fraction": smt_geometry["top1_energy_fraction"], "smt_pairwise_cosine_mean": smt_geometry["pairwise_cosine_mean"], "smt_pairwise_cosine_std": smt_geometry["pairwise_cosine_std"], "smt_finite": smt_geometry["finite"],
            "hope_rms": hope_geometry["rms"], "hope_centered_variance": hope_geometry["centered_variance"], "hope_effective_rank": hope_geometry["effective_rank"], "hope_top1_energy_fraction": hope_geometry["top1_energy_fraction"], "hope_pairwise_cosine_mean": hope_geometry["pairwise_cosine_mean"], "hope_pairwise_cosine_std": hope_geometry["pairwise_cosine_std"], "hope_finite": hope_geometry["finite"],
            "pre_smt_rms": pre_smt_geometry["rms"], "pre_hope_rms": pre_hope_geometry["rms"], "smt_rms_fresh_read_only": pre_smt_geometry["rms"], "hope_rms_fresh_read_only": pre_hope_geometry["rms"],
            "current_image_smt_relative_l2": relative_l2(causal_smt, pre_smt), "current_image_smt_cosine": cosine_flat(causal_smt, pre_smt), "current_image_smt_rms_ratio": float(smt_geometry["rms"] / (pre_smt_geometry["rms"] + RANK_EPS)),
            "current_image_hope_relative_l2": relative_l2(causal_hope, pre_hope) if complete else float("nan"), "current_image_hope_cosine": cosine_flat(causal_hope, pre_hope) if complete else float("nan"), "current_image_hope_rms_ratio": float(hope_geometry["rms"] / (pre_hope_geometry["rms"] + RANK_EPS)) if complete else float("nan"),
            **term_metrics, **{f"{name}_{key}": value for name, metrics in memory_metrics.items() for key, value in metrics.items()}, **state,
        }
        rows.append(row)
        if event_id in CHECKPOINTS and complete:
            checkpoints[event_id] = (snapshot_module(smt), snapshot_module(cms))
        if progress and (event_id in (1, 10, 20, 40, 41, 80, 81, 120, 121, 160, 161, 200) or not complete):
            print(f"{mapping.name}: event={event_id} complete={complete} SMT_RMS={row['smt_rms']:.6g} M={row['memory_norm']:.6g} current_rel_l2={row['current_image_smt_relative_l2']:.5g}", flush=True)
        previous_class = record["class_name"]
        if not complete:
            break
    if rows:
        first = rows[0]
        cumulative_log = 0.0
        cumulative_product = 1.0
        for row in rows:
            if row["alpha_product_observed"] is not None:
                cumulative_product *= float(row["alpha_product_observed"])
                cumulative_log += float(row["alpha_log_product_observed"])
            row["alpha_product_cumulative"] = cumulative_product
            row["alpha_log_product_cumulative"] = cumulative_log
            for name in ("memory", "k", "v", "eta", "alpha"):
                row[f"{name}_norm_rel_event1"] = row[f"{name}_norm"] / (first[f"{name}_norm"] + EPS)
            for prefix in ("smt", "hope"):
                row[f"{prefix}_rms_rel_event1"] = row[f"{prefix}_rms"] / (first[f"{prefix}_rms"] + EPS)
                row[f"{prefix}_centered_variance_rel_event1"] = row[f"{prefix}_centered_variance"] / (first[f"{prefix}_centered_variance"] + EPS)
                row[f"{prefix}_effective_rank_rel_event1"] = row[f"{prefix}_effective_rank"] / (first[f"{prefix}_effective_rank"] + EPS)
    anchors = anchor_stream(records)
    anchor_rows: list[dict[str, Any]] = []
    references: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for checkpoint in CHECKPOINTS:
        if checkpoint not in checkpoints:
            continue
        smt_checkpoint = clone_smt_from_state(checkpoints[checkpoint][0], dim, device)
        cms_checkpoint = clone_cms_from_state(checkpoints[checkpoint][1], dim, device)
        seen = CLASS_ORDER[: checkpoint // EVENTS_PER_CLASS]
        for class_name in seen:
            for anchor in anchors[class_name]:
                current_smt, current_hope = evaluate_anchor(smt_checkpoint, cms_checkpoint, anchor["patches"].to(device))
                if class_name not in references:
                    references[class_name] = (current_smt.detach().clone(), current_hope.detach().clone())
                reference_smt, reference_hope = references[class_name]
                for space, current, reference in (("SMT", current_smt, reference_smt), ("HOPE", current_hope, reference_hope)):
                    metrics = compare_anchor(current, reference)
                    anchor_rows.append({"candidate": mapping.name, "checkpoint_event": checkpoint, "class_name": class_name, "anchor_relative_path": anchor["relative_path"], "anchor_class_index": anchor["class_index"], "space": space, **metrics})
    classification = classify_long(rows, anchor_rows)
    return {"candidate": mapping.name, "mapping": mapping, "rows": rows, "anchor_rows": anchor_rows, "classification": classification, "complete_event_count": sum(int(row["complete_event"]) for row in rows), "event_count": len(rows), "failure": next((row["error"] for row in rows if row["error"]), None), "smt": smt, "cms": cms}


def classify_long(rows: Sequence[Mapping[str, Any]], anchors: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "INCONCLUSIVE"
    if not all(bool(row["finite"] and row["smt_finite"] and row["hope_finite"]) for row in rows):
        return "EXPLODING"
    if any(int(row["fast_nonzero_count"]) == 0 for row in rows):
        return "COLLAPSED"
    if len({row["state_bytes"] for row in rows}) != 1 or len({row["state_schema_signature"] for row in rows}) != 1:
        return "INCONCLUSIVE"
    if any(bool(row["persistent_grad_fn"] or row["online_requires_grad"]) for row in rows):
        return "INCONCLUSIVE"
    last = rows[-1]
    if max(float(row["smt_rms_rel_event1"]) for row in rows) > 10.0 or max(float(row["hope_rms_rel_event1"]) for row in rows) > 10.0:
        return "EXPANDING"
    if float(last["smt_rms_rel_event1"]) < 0.5 or float(last["hope_rms_rel_event1"]) < 0.5:
        return "DECAYING"
    prior = [row for row in anchors if row["space"] == "SMT" and int(row["checkpoint_event"]) > (CLASS_ORDER.index(row["class_name"]) + 1) * EVENTS_PER_CLASS]
    finite_prior_cosines = [float(row["cosine"]) for row in prior if math.isfinite(float(row["cosine"]))]
    if finite_prior_cosines and min(finite_prior_cosines) < 0.80:
        return "SLOW_DRIFT"
    if float(last["smt_centered_variance_rel_event1"]) < 0.25 or float(last["hope_centered_variance_rel_event1"]) < 0.25:
        return "DECAYING"
    return "STABLE"


def fit_slopes(rows: Sequence[Mapping[str, Any]], windows: Sequence[tuple[int, int]] = ((1, 40), (41, 80), (81, 120), (121, 160), (161, 200), (1, 200))) -> list[dict[str, Any]]:
    getters = {
        "log_memory_norm": lambda row: math.log(float(row["memory_norm"]) + RANK_EPS),
        "log_smt_rms": lambda row: math.log(float(row["smt_rms"]) + RANK_EPS),
        "log_hope_rms": lambda row: math.log(float(row["hope_rms"]) + RANK_EPS),
        "smt_effective_rank": lambda row: float(row["smt_effective_rank"]),
        "hope_effective_rank": lambda row: float(row["hope_effective_rank"]),
        "smt_top1_energy_fraction": lambda row: float(row["smt_top1_energy_fraction"]),
        "hope_top1_energy_fraction": lambda row: float(row["hope_top1_energy_fraction"]),
    }
    output: list[dict[str, Any]] = []
    for start, end in windows:
        selected = [row for row in rows if start <= int(row["event_id"]) <= end and row["complete_event"]]
        if len(selected) < 2:
            continue
        x = torch.tensor([float(row["event_id"]) for row in selected], dtype=torch.float64)
        xc = x - x.mean()
        denominator = float((xc * xc).sum().item()) + RANK_EPS
        for name, getter in getters.items():
            y = torch.tensor([getter(row) for row in selected], dtype=torch.float64)
            yc = y - y.mean()
            slope = float((xc * yc).sum().item() / denominator)
            output.append({"window_start": start, "window_end": end, "metric": name, "slope": slope, "count": len(selected)})
    return output


def retention_matrix(anchor_rows: Sequence[Mapping[str, Any]], metric: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in anchor_rows:
        if row["space"] == "SMT" and math.isfinite(float(row[metric])):
            grouped[(int(row["checkpoint_event"]), str(row["class_name"]))].append(float(row[metric]))
    return [{"checkpoint_event": checkpoint, "class_name": class_name, f"mean_{metric}": sum(values) / len(values), "count": len(values)} for (checkpoint, class_name), values in sorted(grouped.items())]


# Descriptive compatibility alias for focused tests and downstream readers.
build_retention_matrix = retention_matrix


def boundary_rows(rows: Sequence[Mapping[str, Any]], anchor_rows: Sequence[Mapping[str, Any]], candidate: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for before_id, after_id in ((40, 41), (80, 81), (120, 121), (160, 161)):
        if len(rows) < after_id:
            continue
        before, after = rows[before_id - 1], rows[after_id - 1]
        anchors = [row for row in anchor_rows if int(row["checkpoint_event"]) == before_id and row["space"] == "SMT"]
        result.append({"candidate": candidate, "boundary": f"{before_id}->{after_id}", "from_class": before["class_name"], "to_class": after["class_name"], "memory_relative_change": after["memory_norm"] / (before["memory_norm"] + EPS) - 1.0, "smt_rms_before": before["smt_rms"], "smt_rms_after": after["smt_rms"], "hope_rms_before": before["hope_rms"], "hope_rms_after": after["hope_rms"], "smt_rank_before": before["smt_effective_rank"], "smt_rank_after": after["smt_effective_rank"], "hope_rank_before": before["hope_effective_rank"], "hope_rank_after": after["hope_effective_rank"], "rank_update_ratio_after": after["rank_update_state_ratio"], "surprise_update_ratio_after": after["surprise_update_state_ratio"], "prior_anchor_cosine_mean": sum(float(row["cosine"]) for row in anchors) / len(anchors) if anchors else float("nan"), "prior_anchor_relative_l2_mean": sum(float(row["relative_l2"]) for row in anchors) / len(anchors) if anchors else float("nan")})
    return result
