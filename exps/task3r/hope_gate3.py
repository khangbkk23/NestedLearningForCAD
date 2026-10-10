# exps/hope_gate3.py
"""Experiment-only long-horizon normal-stream diagnostics."""
from __future__ import annotations

import math
import copy
import hashlib
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from exps.hope_retention_stabilization import (
    _state_metrics,
    _stream_state_summary,
    clone_cms_from_state,
    clone_smt_from_state,
    build_read_only_rms_reference,
    probe_objective,
)
from exps.hope_update_stabilization import EPS, UpdateMapping, run_update_smt

CLASS_ORDER = ("bottle", "carpet", "grid", "toothbrush", "transistor")
EVENTS_PER_CLASS = 40
RANK_EPS = 1e-12
CHECKPOINTS = (40, 80, 120, 160, 200)
ANCHOR_POSITIONS = (1, 10, 20, 40)
SELF_REFERENCE_TOLERANCE = 3e-5


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
        anchors[class_name] = []
        for local in ANCHOR_POSITIONS:
            if start + local - 1 >= len(records):
                continue
            record = records[start + local - 1]
            if record["class_name"] != class_name:
                raise ValueError("anchor stream is not in the prescribed class order")
            anchors[class_name].append({**record, "anchor_position": local})
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
    return {key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value) for key, value in module.state_dict().items()}


def tensor_state_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.keys() == right.keys() and all((torch.equal(left[key], right[key]) if isinstance(left[key], torch.Tensor) else left[key] == right[key]) for key in left)


def evaluate_anchor(smt: Any, cms: Any, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    before = {"smt": snapshot_module(smt), "cms": snapshot_module(cms)}
    smt_out, hope_out = read_only_representations(smt, cms, image)
    after = {"smt": snapshot_module(smt), "cms": snapshot_module(cms)}
    if not tensor_state_equal(before["smt"], after["smt"]) or not tensor_state_equal(before["cms"], after["cms"]):
        raise AssertionError("read-only anchor evaluation mutated state")
    return smt_out, hope_out


def read_only_representations(smt: Any, cms: Any, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve from frozen state through the public projection API.

    With no writes, every chunk uses the same memory. This is the read-only
    representation equation of forward(update=False), without preparing
    discarded update candidates. Causal update passes never use this helper.
    Batched linear algebra can differ at FP32 rounding precision.
    """
    with torch.no_grad():
        projections = smt.project(image)
        smt_out = F.linear(projections.q, smt.memories["memory"].weight.detach()).detach().clone()
        hope_out = cms.forward(smt_out).detach().clone()
    return smt_out, hope_out


def fresh_initial_references(records: Sequence[Mapping[str, Any]], smt_state: Mapping[str, Any], cms_state: Mapping[str, Any], device: str) -> list[dict[str, float]]:
    dim = int(records[0]["patches"].shape[-1])
    smt, cms = clone_smt_from_state(smt_state, dim, device), clone_cms_from_state(cms_state, dim, device)
    before_smt, before_cms = snapshot_module(smt), snapshot_module(cms)
    rows = []
    for event, record in enumerate(records, 1):
        smt_out, hope_out = read_only_representations(smt, cms, record["patches"].to(device))
        rows.append({"smt_rms": tensor_rms(smt_out), "hope_rms": tensor_rms(hope_out)})
        if event % 40 == 0:
            print(f"FRESH_INITIAL_READ_ONLY {event}/{len(records)}", flush=True)
    if not tensor_state_equal(before_smt, snapshot_module(smt)) or not tensor_state_equal(before_cms, snapshot_module(cms)):
        raise AssertionError("fresh initial reference state mutated")
    return rows


def read_only_oracle(smt: Any, cms: Any, image: torch.Tensor) -> dict[str, Any]:
    before_smt, before_cms = snapshot_module(smt), snapshot_module(cms)
    with torch.no_grad():
        expected_smt = smt.forward(image, update=False).memory_prediction
        expected_hope = cms.forward(expected_smt)
    actual_smt, actual_hope = read_only_representations(smt, cms, image)
    differences = {"smt_max_abs": float((expected_smt - actual_smt).abs().max().item()), "hope_max_abs": float((expected_hope - actual_hope).abs().max().item())}
    unchanged = tensor_state_equal(before_smt, snapshot_module(smt)) and tensor_state_equal(before_cms, snapshot_module(cms))
    return {**differences, "tolerance": 3e-5, "state_unchanged": unchanged, "passed": unchanged and max(differences.values()) <= 3e-5}


def compare_anchor(current: torch.Tensor, reference: torch.Tensor, reference_geometry: Mapping[str, Any] | None = None) -> dict[str, float]:
    current_geometry = full_geometry(current)
    if reference_geometry is None:
        reference_geometry = full_geometry(reference)
    return {
        "cosine": cosine_flat(current, reference),
        "relative_l2": relative_l2(current, reference),
        "rms_ratio": float(current_geometry["rms"] / (float(reference_geometry["rms"]) + RANK_EPS)),
        "centered_variance_ratio": float(current_geometry["centered_variance"] / (float(reference_geometry["centered_variance"]) + RANK_EPS)),
        "effective_rank_ratio": float(current_geometry["effective_rank"] / (float(reference_geometry["effective_rank"]) + RANK_EPS)),
    }


def tensor_rms(value: torch.Tensor) -> float:
    return float(value.detach().double().square().mean().sqrt().item())


def self_reference_check(row: Mapping[str, Any]) -> None:
    """Reject identity errors before any scientific drift interpretation."""
    if row["evaluation_checkpoint"] != row["reference_checkpoint"]:
        return
    for metric, expected in (("cosine", 1.0), ("relative_l2", 0.0), ("rms_ratio", 1.0)):
        value = float(row[metric])
        if not math.isfinite(value) or abs(value - expected) > SELF_REFERENCE_TOLERANCE:
            identity = {key: row[key] for key in ("candidate", "class_name", "anchor_position", "evaluation_checkpoint", "space")}
            raise AssertionError(f"anchor self-reference failure: {identity}, {metric}={value!r}")


def reference_outputs(smt: Any, cms: Any, anchors: Sequence[Mapping[str, Any]], device: str) -> dict[tuple[str, int], dict[str, Any]]:
    references: dict[tuple[str, int], dict[str, Any]] = {}
    for anchor in anchors:
        identity = (str(anchor["class_name"]), int(anchor["anchor_position"]))
        if identity in references:
            raise ValueError(f"duplicate anchor identity: {identity}")
        smt_out, hope_out = evaluate_anchor(smt, cms, anchor["patches"].to(device))
        references[identity] = {
            "relative_path": anchor["relative_path"],
            "SMT": smt_out.detach().cpu().clone(), "HOPE": hope_out.detach().cpu().clone(),
            "SMT_geometry": full_geometry(smt_out), "HOPE_geometry": full_geometry(hope_out),
        }
    return references


def anchor_rows_at_checkpoint(
    candidate: str, checkpoint: int, smt: Any, cms: Any,
    anchors: Mapping[str, Sequence[Mapping[str, Any]]],
    references: dict[tuple[str, int], dict[str, Any]], device: str,
    fresh_rms: Mapping[tuple[str, int], Mapping[str, float]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for class_index, class_name in enumerate(CLASS_ORDER):
        reference_checkpoint = (class_index + 1) * EVENTS_PER_CLASS
        if reference_checkpoint > checkpoint:
            continue
        if checkpoint == reference_checkpoint:
            new = reference_outputs(smt, cms, anchors[class_name], device)
            if any(key in references for key in new):
                raise ValueError(f"reference checkpoint repeated for {candidate}/{class_name}")
            references.update(new)
        for anchor in anchors[class_name]:
            identity = (class_name, int(anchor["anchor_position"]))
            reference = references[identity]
            if reference["relative_path"] != anchor["relative_path"]:
                raise AssertionError(f"anchor path mismatch: {candidate}/{identity}")
            current_smt, current_hope = evaluate_anchor(smt, cms, anchor["patches"].to(device))
            for space, current in (("SMT", current_smt), ("HOPE", current_hope)):
                metrics = compare_anchor(current.cpu(), reference[space], reference[f"{space}_geometry"])
                row = {
                    "candidate": candidate, "checkpoint_event": checkpoint,
                    "evaluation_checkpoint": checkpoint, "reference_checkpoint": reference_checkpoint,
                    "class_name": class_name, "anchor_position": identity[1],
                    "anchor_image_id": anchor["relative_path"], "anchor_relative_path": anchor["relative_path"],
                    "anchor_cache_key": f"class_{class_name}.pt:patches[{identity[1] - 1}]",
                    "space": space, **metrics,
                    "reference_rms": reference[f"{space}_geometry"]["rms"], "evaluation_rms": tensor_rms(current),
                    "fresh_initial_read_only_rms": fresh_rms[identity][space] if fresh_rms is not None else None,
                }
                self_reference_check(row)
                rows.append(row)
    return rows


def check_event_counters(smt: Any, cms: Any, event: int, tokens: int) -> None:
    expected_memory = event * math.ceil(tokens / smt.memory_chunk_size)
    expected_aux = event * math.ceil(tokens / smt.auxiliary_memory_chunk_size)
    state = _stream_state_summary(smt, cms)
    expected = {
        "smt_memory_updates": expected_memory, "smt_auxiliary_updates": expected_aux,
        "smt_online_updates": expected_memory + expected_aux,
        "cms_completed_events": event, "cms_pending_counts": [event % p for p in cms.update_periods],
        "cms_update_counts": [event // p for p in cms.update_periods],
    }
    for key, value in expected.items():
        actual = list(state[key]) if isinstance(value, list) else state[key]
        if actual != value:
            raise AssertionError(f"continuation counter {key}: {state[key]!r} != {value!r}")


def _cpu_state(module: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value) for key, value in module.state_dict().items()}


def save_checkpoint(path: Path, smt: Any, cms: Any, mapping: UpdateMapping, event: int, tokens: int) -> dict[str, Any]:
    check_event_counters(smt, cms, event, tokens)
    state = _stream_state_summary(smt, cms)
    metadata = {"schema_version": 1, "candidate": mapping.name, "mapping": asdict(mapping), "event_count": event, "tokens_per_image": tokens, "dim": smt.dim, **state}
    payload = {"metadata": metadata, "smt_state": _cpu_state(smt), "cms_state": _cpu_state(cms)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".writing")
    torch.save(payload, temporary)
    temporary.replace(path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if loaded["metadata"] != metadata or not tensor_state_equal(loaded["smt_state"], payload["smt_state"]) or not tensor_state_equal(loaded["cms_state"], payload["cms_state"]):
        raise AssertionError(f"checkpoint round trip failed: {path}")
    return {**metadata, "path": str(path), "reopened_exactly": True}


def load_checkpoint(path: Path, mapping: UpdateMapping, device: str) -> tuple[Any, Any, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata["schema_version"] != 1 or metadata["mapping"] != asdict(mapping) or metadata["candidate"] != mapping.name:
        raise ValueError(f"incompatible checkpoint mapping: {path}")
    smt = clone_smt_from_state(payload["smt_state"], metadata["dim"], device)
    cms = clone_cms_from_state(payload["cms_state"], metadata["dim"], device)
    check_event_counters(smt, cms, metadata["event_count"], metadata["tokens_per_image"])
    actual = _stream_state_summary(smt, cms)
    for key in ("state_bytes", "state_key_count", "state_schema_signature"):
        if metadata[key] != actual[key]:
            raise AssertionError(f"checkpoint state schema mismatch: {path}/{key}")
    return smt, cms, metadata


def artifact_fingerprints(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            result[str(path.relative_to(root))] = digest.hexdigest()
    return result


def replay_candidate(
    records: Sequence[Mapping[str, Any]], mapping: UpdateMapping,
    smt_state: Mapping[str, Any], cms_state: Mapping[str, Any], device: str,
    checkpoint_root: Path, original_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconstruct continuation states without recomputing per-event geometry."""
    dim = int(records[0]["patches"].shape[-1])
    smt = clone_smt_from_state(smt_state, dim, device)
    cms = clone_cms_from_state(cms_state, dim, device)
    original = {int(row["event_id"]): row for row in original_rows if row["candidate"] == mapping.name}
    if set(original) != set(range(1, len(records) + 1)):
        raise ValueError(f"original rows missing or duplicated: {mapping.name}")
    checks = []
    manifests = []
    comparison_events = {1, 40, 80, 120, 160, 200}
    checkpoints = set(CHECKPOINTS) | {50}
    schema = _stream_state_summary(smt, cms)
    # Cross-device FP32 comparisons are scalar trajectory checks, not a claim
    # of bitwise CUDA/CPU state equivalence. TF32 is disabled by the runner.
    rtol, atol = (3e-5, 1e-7) if device.startswith("cuda") else (2e-7, 1e-9)
    for event, record in enumerate(records, 1):
        image = record["patches"].to(device)
        smt_out, _, _ = run_update_smt(smt, image, mapping, event_id=event, capture_trace=False)
        hope_out = cms.commit_image(smt_out, [probe_objective] * cms.K).output.detach()
        state = _stream_state_summary(smt, cms)
        check_event_counters(smt, cms, event, image.shape[1])
        if not state["finite_state"] or state["persistent_grad_fn"] or state["online_requires_grad"]:
            raise AssertionError(f"replay state failure: {mapping.name}/{event}")
        for key in ("state_bytes", "state_key_count", "state_schema_signature"):
            if state[key] != schema[key]:
                raise AssertionError(f"replay schema changed: {mapping.name}/{event}/{key}")
        if not torch.isfinite(smt_out).all().item() or not torch.isfinite(hope_out).all().item():
            raise AssertionError(f"replay output failure: {mapping.name}/{event}")
        if event in comparison_events:
            values = {"memory_norm": float(smt.memories["memory"].weight.detach().double().norm().item()), "smt_rms": tensor_rms(smt_out), "hope_rms": tensor_rms(hope_out)}
            for key, value in values.items():
                expected = float(original[event][key])
                passed = abs(value - expected) <= atol + rtol * abs(expected)
                checks.append({"candidate": mapping.name, "event_id": event, "metric": key, "original": expected, "replay": value, "absolute_error": abs(value - expected), "relative_error": abs(value - expected) / (abs(expected) + RANK_EPS), "rtol": rtol, "atol": atol, "passed": passed})
                if not passed:
                    raise AssertionError(f"replay trajectory mismatch: {checks[-1]}")
            for key in ("smt_memory_updates", "smt_auxiliary_updates", "smt_online_updates", "cms_completed_events", "cms_update_counts", "cms_pending_counts"):
                expected_counter = tuple(original[event][key]) if isinstance(state[key], tuple) else original[event][key]
                if state[key] != expected_counter:
                    raise AssertionError(f"original/replay counter mismatch: {mapping.name}/{event}/{key}")
        if event in checkpoints:
            path = checkpoint_root / f"{mapping.name}__event_{event:03d}.pt"
            manifests.append(save_checkpoint(path, smt, cms, mapping, event, image.shape[1]))
        if event == 1 or event % 10 == 0:
            print(f"REPLAY {mapping.name}: {event}/{len(records)} finite=True counters=OK checkpoints={len(manifests)}", flush=True)
    return {"candidate": mapping.name, "device": device, "checks": checks, "checkpoints": manifests, "passed": True}


def evaluate_checkpoint_anchors(
    records: Sequence[Mapping[str, Any]], mapping: UpdateMapping,
    checkpoint_paths: Mapping[int, Path], device: str,
    fresh_rms: Mapping[tuple[str, int], Mapping[str, float]],
    reference_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anchors = anchor_stream(records)
    references: dict[tuple[str, int], dict[str, Any]] = {}
    rows = []
    manifest = []
    for checkpoint, path in sorted(checkpoint_paths.items()):
        smt, cms, _ = load_checkpoint(path, mapping, device)
        current_rows = anchor_rows_at_checkpoint(mapping.name, checkpoint, smt, cms, anchors, references, device, fresh_rms)
        rows.extend(current_rows)
        for row in current_rows:
            if row["evaluation_checkpoint"] == row["reference_checkpoint"]:
                manifest.append({key: row[key] for key in ("candidate", "class_name", "anchor_position", "anchor_image_id", "anchor_cache_key", "reference_checkpoint", "space", "reference_rms")})
        print(f"ANCHORS {mapping.name}: checkpoint={checkpoint} rows={len(current_rows)} self_reference=PASS", flush=True)
    if reference_path is not None:
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"candidate": mapping.name, "references": references}, reference_path)
        restored = torch.load(reference_path, weights_only=False, map_location="cpu")
        if restored["candidate"] != mapping.name or restored["references"].keys() != references.keys():
            raise AssertionError("reference artifact identity did not round trip")
        for identity, reference in references.items():
            for space in ("SMT", "HOPE"):
                if not torch.equal(restored["references"][identity][space], reference[space]):
                    raise AssertionError(f"reference output did not round trip: {identity}/{space}")
    return rows, manifest


def neutral_retention(mapping: UpdateMapping, events: int) -> float:
    if mapping.alpha_kind == "one":
        return 1.0
    if mapping.alpha_kind != "horizon_near_one":
        raise ValueError("neutral horizon retention requires alpha=1 or HNR")
    return math.exp(mapping.horizon * events * math.log1p(-mapping.lambda_h / (2.0 * mapping.horizon)))


def run_long_candidate(records: Sequence[Mapping[str, Any]], mapping: UpdateMapping, smt_state: Mapping[str, Any], cms_state: Mapping[str, Any], device: str, *, max_events: int = 200, progress: bool = False, checkpoint_root: Path | None = None, fresh_references: Sequence[Mapping[str, float]] | None = None) -> dict[str, Any]:
    dim = int(records[0]["patches"].shape[-1])
    smt = clone_smt_from_state(smt_state, dim, device)
    cms = clone_cms_from_state(cms_state, dim, device)
    initial = {name: memory.weight.detach().clone() for name, memory in smt.memories.items()}
    rows: list[dict[str, Any]] = []
    checkpoints: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    checkpoint_events = set(CHECKPOINTS) | {min(max_events, len(records))}
    if fresh_references is None:
        fresh_references = fresh_initial_references(records[:max_events], smt_state, cms_state, device)
    previous_class: str | None = None
    for event_id, record in enumerate(records[:max_events], start=1):
        image = record["patches"].to(device)
        pre_smt, pre_hope = read_only_representations(smt, cms, image)
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
        pre_smt_rms = tensor_rms(pre_smt)
        pre_hope_rms = tensor_rms(pre_hope)
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
            "pre_event_read_only_smt_rms": pre_smt_rms, "pre_event_read_only_hope_rms": pre_hope_rms,
            "post_event_smt_rms": smt_geometry["rms"], "post_event_hope_rms": hope_geometry["rms"],
            "fresh_initial_read_only_smt_rms": fresh_references[event_id - 1]["smt_rms"],
            "fresh_initial_read_only_hope_rms": fresh_references[event_id - 1]["hope_rms"],
            "smt_rms_to_fresh_initial": smt_geometry["rms"] / (fresh_references[event_id - 1]["smt_rms"] + RANK_EPS),
            "hope_rms_to_fresh_initial": hope_geometry["rms"] / (fresh_references[event_id - 1]["hope_rms"] + RANK_EPS),
            "current_image_smt_relative_l2": relative_l2(causal_smt, pre_smt), "current_image_smt_cosine": cosine_flat(causal_smt, pre_smt), "current_image_smt_rms_ratio": float(smt_geometry["rms"] / (pre_smt_rms + RANK_EPS)),
            "current_image_hope_relative_l2": relative_l2(causal_hope, pre_hope) if complete else float("nan"), "current_image_hope_cosine": cosine_flat(causal_hope, pre_hope) if complete else float("nan"), "current_image_hope_rms_ratio": float(hope_geometry["rms"] / (pre_hope_rms + RANK_EPS)) if complete else float("nan"),
            **term_metrics, **{f"{name}_{key}": value for name, metrics in memory_metrics.items() for key, value in metrics.items()}, **state,
        }
        rows.append(row)
        if complete:
            check_event_counters(smt, cms, event_id, image.shape[1])
        if event_id in checkpoint_events and event_id >= EVENTS_PER_CLASS and complete:
            checkpoints[event_id] = (_cpu_state(smt), _cpu_state(cms))
            if checkpoint_root is not None:
                save_checkpoint(checkpoint_root / f"{mapping.name}__event_{event_id:03d}.pt", smt, cms, mapping, event_id, image.shape[1])
        if progress and (event_id == 1 or event_id % 10 == 0 or row["is_class_boundary"] or not complete):
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
    references: dict[tuple[str, int], dict[str, Any]] = {}
    fresh_anchor_rms = {
        (class_name, int(anchor["anchor_position"])): {
            "SMT": fresh_references[index * EVENTS_PER_CLASS + int(anchor["anchor_position"]) - 1]["smt_rms"],
            "HOPE": fresh_references[index * EVENTS_PER_CLASS + int(anchor["anchor_position"]) - 1]["hope_rms"],
        }
        for index, class_name in enumerate(CLASS_ORDER)
        for anchor in anchors[class_name]
        if index * EVENTS_PER_CLASS + int(anchor["anchor_position"]) <= len(fresh_references)
    }
    for checkpoint in sorted(checkpoints):
        smt_checkpoint = clone_smt_from_state(checkpoints[checkpoint][0], dim, device)
        cms_checkpoint = clone_cms_from_state(checkpoints[checkpoint][1], dim, device)
        anchor_rows.extend(anchor_rows_at_checkpoint(mapping.name, checkpoint, smt_checkpoint, cms_checkpoint, anchors, references, device, fresh_anchor_rms))
    classification = classify_long(rows, anchor_rows, mapping)
    return {"candidate": mapping.name, "mapping": mapping, "rows": rows, "anchor_rows": anchor_rows, "classification": classification, "complete_event_count": sum(int(row["complete_event"]) for row in rows), "event_count": len(rows), "failure": next((row["error"] for row in rows if row["error"]), None), "smt": smt, "cms": cms}


def classify_long(rows: Sequence[Mapping[str, Any]], anchors: Sequence[Mapping[str, Any]], mapping: UpdateMapping | None = None) -> str:
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
    for anchor in anchors:
        self_reference_check(anchor)
        if not all(math.isfinite(float(anchor[key])) for key in ("cosine", "relative_l2", "rms_ratio")):
            return "INCONCLUSIVE"
    resolution = torch.finfo(torch.float32).eps
    for row in rows:
        for space in ("smt", "hope"):
            rms, variance = float(row[f"{space}_rms"]), float(row[f"{space}_centered_variance"])
            if rms == 0 or variance <= resolution ** 2 * rms ** 2:
                return "COLLAPSED"
            if float(row[f"{space}_effective_rank"]) <= 1 + math.sqrt(resolution) or float(row[f"{space}_top1_energy_fraction"]) >= 1 - resolution:
                return "COLLAPSED"
    # Amplitude is interpreted relative to measured retention, not an anchor
    # cosine cutoff. The broad range check must agree across both output
    # spaces and state before it flags uncontrolled expansion/contraction.
    last = rows[-1]
    retention = float(last.get("alpha_product_cumulative", 1.0))
    adjusted = [float(last[f"{space}_rms_rel_event1"]) / max(retention, RANK_EPS) for space in ("smt", "hope")]
    memory_change = float(last["memory_norm"]) / (float(rows[0]["memory_norm"]) + RANK_EPS)
    if min(adjusted) > 10 and memory_change > 10:
        return "EXPANDING"
    if max(adjusted) < 0.1 and memory_change / max(retention, RANK_EPS) < 0.1:
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
    grouped: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    identities = set()
    for row in anchor_rows:
        checkpoint = int(row["evaluation_checkpoint"])
        key = (str(row["candidate"]), checkpoint, str(row["class_name"]), str(row["space"]))
        identity = (*key, int(row["anchor_position"]))
        if identity in identities:
            raise ValueError(f"duplicate matrix source anchor: {identity}")
        identities.add(identity)
        if math.isfinite(float(row[metric])):
            grouped[key].append(float(row[metric]))
    return [{"candidate": candidate, "checkpoint": checkpoint, "evaluation_checkpoint": checkpoint, "class_name": class_name, "space": space, "metric": metric, "value": sum(values) / len(values), f"mean_{metric}": sum(values) / len(values), "anchor_count": len(values)} for (candidate, checkpoint, class_name, space), values in sorted(grouped.items())]


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
