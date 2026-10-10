# exps/hope_outer_learning_h50_v1.py
"""Contextual-signal screen for a bounded outer-learning study."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hope_cad/outer_learning_ol04e"
SOURCE = ROOT / "results/hope_cad/outer_learning_pilot_v1/shared_gpu_attempt"
FINAL = SOURCE / "evaluation_completion"
CACHE = ROOT / "results/hope_cad/memory_learning_gate/features/class_bottle.pt"
MASKED = SOURCE / "masked_features.pt"
CENTERS = tuple((r, c) for r in (3, 10, 17, 24) for c in (3, 10, 17, 24))
CENTER_IDS = tuple(r * 28 + c for r, c in CENTERS)
TRAIN_QUERY = tuple(range(60, 80))
VAL_QUERY = tuple(range(92, 100))
ONLINE = tuple(range(100, 150))
PROBES = tuple(range(150, 170))
EXPECTED_CACHE_SHA = "25c17dd008468a3746564cd75b01f0cc1507ff518014c59dc73be2491a39d44b"
EXPECTED_CHECKPOINT_SHA = "ae3012808a9b406a19b799381bd26b253634ad26125937c26d02dbcbbc85dd92"
FP32_ATOL = 1e-6
FP32_RTOL = 1e-5
BOOTSTRAP_SEED = 4404
BOOTSTRAP_REPEATS = 2000


class HardGate(RuntimeError):
    pass


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    plain = value.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(plain.shape), plain.dtype)).encode() + plain.numpy().tobytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)
    json.loads(path.read_text())


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    reopened = pd.read_parquet(path)
    if list(reopened.columns) != list(frame.columns) or len(reopened) != len(frame):
        raise HardGate(f"table reopen differs: {path}")


def clean_targets(clean: torch.Tensor, identities: tuple[int, ...]) -> dict[int, torch.Tensor]:
    return {i: F.normalize(clean[i], dim=-1, eps=1e-8) for i in identities}


def source_warning() -> dict[str, Any]:
    training = json.loads((SOURCE / "execution_source_sha256.json").read_text())
    evaluation = json.loads((SOURCE / "evaluation_completion_source_sha256.json").read_text())
    paths = sorted(set(training) | set(evaluation))
    rows = []
    for path in paths:
        current = file_sha(ROOT / path)
        rows.append({"path": path, "training_sha256": training.get(path),
                     "evaluation_sha256": evaluation.get(path), "current_sha256": current,
                     "matches_training": current == training.get(path),
                     "matches_evaluation": current == evaluation.get(path)})
    return {"warning_only": True, "rows": rows}


def preflight() -> dict[str, Any]:
    pointer = json.loads((ROOT / "results/hope_cad/outer_learning_pilot_v1/authoritative_result.json").read_text())
    if pointer.get("status") != "COMPLETE":
        raise HardGate("authoritative OL-04 pointer is not complete")
    summary = json.loads((FINAL / "summary.json").read_text())
    if summary.get("status") != "COMPLETE" or summary.get("new_optimizer_steps") != 0:
        raise HardGate("authoritative OL-04 completion is invalid")
    parity = json.loads((ROOT / "results/hope_cad/outer_learning_ol04d/p1_event50_parity.json").read_text())
    if not parity.get("state_parity") or parity.get("authoritative_readout", {}).get("status") != "PASS":
        raise HardGate("OL-04D event-50 parity is not verified")
    for path, expected in {
        "models/hope_cad/self_modifying_titans.py": "ffcd8ca7810ade758954effcb708903eb36b182835dac58e0e577cb497e1932c",
        "models/hope_cad/continuum_memory.py": "920135550843a33d2cc064030bd5155af349dbab0d99cddae213ddc589055581",
        "models/hope_cad/hope_block.py": "16e9b00cf9bba7e66ceaebe94d2c7f6cf23f95c907bbe59d47d879b16b179437",
        "exps/hope_image_synchronous_memory.py": "f5e022384da6195ae0429a6c172cfd5d0c0b5690f8e7439f7fb9dd3fcbee668b",
    }.items():
        if file_sha(ROOT / path) != expected:
            raise HardGate(f"protected source changed: {path}")
    if file_sha(CACHE) != EXPECTED_CACHE_SHA:
        raise HardGate("clean feature cache identity differs")
    payload = torch.load(CACHE, map_location="cpu", weights_only=False, mmap=True)
    if tuple(payload["patches"].shape) != (170, 784, 768) or payload["patches"].dtype != torch.float32:
        raise HardGate("clean feature cache geometry differs")
    masked = torch.load(MASKED, map_location="cpu", weights_only=True)
    if masked.get("provenance", {}).get("source_cache_sha256") != EXPECTED_CACHE_SHA:
        raise HardGate("masked feature provenance differs")
    required = set(TRAIN_QUERY) | set(VAL_QUERY) | set(PROBES)
    if set(masked.get("features", {})) != required:
        raise HardGate("masked feature identities differ")
    for identity in required:
        value = masked["features"][identity]
        if tuple(value.shape) != (784, 768) or value.dtype != torch.float32:
            raise HardGate(f"masked feature geometry differs: {identity}")
        if tensor_sha(value) != masked["feature_hashes"][str(identity)]:
            raise HardGate(f"masked feature hash differs: {identity}")
    return {
        "status": "PASS", "authoritative_pointer": pointer,
        "manifest_sha256": file_sha(SOURCE / "stream_manifest.parquet"),
        "cache_sha256": file_sha(CACHE), "masked_payload_sha256": file_sha(MASKED),
        "masked_identities": sorted(required), "train_query": list(TRAIN_QUERY),
        "val_query": list(VAL_QUERY), "online": list(ONLINE), "probes": list(PROBES),
        "event50_parity": parity, "source_hash_warning": source_warning(),
        "forbidden_data_accessed": False, "new_vit_extractions": 0,
        "optimizer_steps": 0, "production_core_unchanged": True,
    }


def ridge_fit(features: torch.Tensor, residual: torch.Tensor, *, seed: int | None = None) -> dict[str, Any]:
    """Fit one shared B with row convention X @ B = R."""
    n_images, n_centers, dim = features.shape
    mean = features.mean(0)
    centered = features - mean
    if seed is not None:
        generator = np.random.default_rng(seed)
        shuffled = torch.empty_like(features)
        for center in range(n_centers):
            order = generator.permutation(n_images)
            shuffled[:, center] = features[torch.as_tensor(order), center]
        centered = shuffled - mean
    X = centered.reshape(n_images * n_centers, dim).double()
    Y = residual.reshape(n_images * n_centers, dim).double()
    gram = X.T @ X
    lam = max(1e-12, 0.01 * float(torch.trace(gram)) / (n_images * n_centers))
    system = gram + lam * torch.eye(dim, dtype=torch.float64)
    B = torch.linalg.solve(system, X.T @ Y)
    singular = torch.linalg.svdvals(X)
    return {"mean": mean, "B": B.float(), "lambda": lam,
            "effective_rank": float(torch.linalg.matrix_rank(X)),
            "condition_number": float(torch.linalg.cond(system)),
            "singular_min": float(singular[-1]), "singular_max": float(singular[0]),
            "shuffled": seed is not None}


def predict(model: dict[str, Any], features: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    centered = features - model["mean"]
    return mu + centered @ model["B"]


def image_losses(predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    return (0.5 * (predictions - targets).square().sum(-1).mean(-1)).detach().cpu().numpy()


def bootstrap(values: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci_low": float(np.quantile(sample, .025)),
            "ci_high": float(np.quantile(sample, .975)), "repeats": BOOTSTRAP_REPEATS}


def gate_a(payload: dict[str, Any], masked_payload: dict[str, Any]) -> dict[str, Any]:
    clean = payload["patches"]
    targets = clean_targets(clean, TRAIN_QUERY + VAL_QUERY)
    train_u = torch.stack([targets[i][list(CENTER_IDS)] for i in TRAIN_QUERY])
    val_u = torch.stack([targets[i][list(CENTER_IDS)] for i in VAL_QUERY])
    mu = train_u.mean(0)
    train_h = torch.stack([masked_payload["features"][i][list(CENTER_IDS)] for i in TRAIN_QUERY])
    val_h = torch.stack([masked_payload["features"][i][list(CENTER_IDS)] for i in VAL_QUERY])
    residual = train_u - mu
    contextual = ridge_fit(train_h, residual)
    shuffled = ridge_fit(train_h, residual, seed=BOOTSTRAP_SEED)
    mean_pred = mu.unsqueeze(0).expand_as(val_u)
    contextual_pred = predict(contextual, val_h, mu)
    shuffled_pred = predict(shuffled, val_h, mu)
    mean_loss = image_losses(mean_pred, val_u)
    contextual_loss = image_losses(contextual_pred, val_u)
    shuffled_loss = image_losses(shuffled_pred, val_u)
    rows = []
    for pos, identity in enumerate(VAL_QUERY):
        rows.append({"identity": identity, "mean_error": float(mean_loss[pos]),
                     "contextual_error": float(contextual_loss[pos]),
                     "shuffled_error": float(shuffled_loss[pos]),
                     "contextual_beats_mean": bool(contextual_loss[pos] < mean_loss[pos])})
    aggregate_mean = float(mean_loss.mean())
    aggregate_contextual = float(contextual_loss.mean())
    aggregate_shuffled = float(shuffled_loss.mean())
    contextual_improvement = 1.0 - aggregate_contextual / aggregate_mean
    shuffle_gap = (aggregate_shuffled - aggregate_contextual) / aggregate_mean
    count_better = int((contextual_loss < mean_loss).sum())
    passed = bool(contextual_improvement >= 0.02 and count_better >= 6 and shuffle_gap >= 0.02)
    return {"status": "PASS" if passed else "FAIL", "gate": "A",
            "mean_error": aggregate_mean, "contextual_error": aggregate_contextual,
            "shuffled_error": aggregate_shuffled, "contextual_improvement_vs_mean": contextual_improvement,
            "contextual_gap_vs_shuffled": shuffle_gap, "contextual_better_images": count_better,
            "required_improvement": 0.02, "required_better_images": 6,
            "rows": rows, "bootstrap_mean": bootstrap(mean_loss, BOOTSTRAP_SEED),
            "bootstrap_contextual": bootstrap(contextual_loss, BOOTSTRAP_SEED + 1),
            "bootstrap_shuffled": bootstrap(shuffled_loss, BOOTSTRAP_SEED + 2),
            "fit": {"contextual": {k: v for k, v in contextual.items() if k not in ("mean", "B")},
                    "shuffled": {k: v for k, v in shuffled.items() if k not in ("mean", "B")},
                    "train_observations": int(train_h.shape[0] * train_h.shape[1]),
                    "dimension": int(train_h.shape[-1]), "centers": list(CENTERS)},
            "teacher_normalization": "unit_clean_layer9_features",
            "loss": "0.5 * mean_image_center(sum_channel((prediction-target)^2))",
            "shuffle_seed": BOOTSTRAP_SEED}


def run(device: str = "cpu") -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "config_resolved.yaml", {
        "task": "OL-04E", "device": device, "gate_a_only_until_pass": True,
        "train_query": list(TRAIN_QUERY), "val_query": list(VAL_QUERY),
        "contextual_ridge": "shared_B_from_masked_features_to_teacher_residual",
        "ridge_lambda": "0.01 * trace(X.T @ X) / n_train with 1e-12 floor",
        "shuffle_seed": BOOTSTRAP_SEED, "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "outer_optimizer_steps": 0, "forbidden_data_accessed": False})
    pre = preflight()
    atomic_json(OUT / "preflight.json", pre)
    payload = torch.load(CACHE, map_location="cpu", weights_only=False, mmap=True)
    masked = torch.load(MASKED, map_location="cpu", weights_only=True)
    result = gate_a(payload, masked)
    atomic_json(OUT / "gate_a_fit.json", result["fit"])
    write_table(OUT / "gate_a_per_image.parquet", result["rows"])
    atomic_json(OUT / "gate_a_summary.json", {k: v for k, v in result.items() if k != "rows"})
    summary = {"status": "GATE_A_PASS" if result["status"] == "PASS" else "OBJECTIVE_NOT_DEMONSTRATED",
               "gate_a": result, "meta50_steps": 0, "static50_steps": 0,
               "training_authorized_by_gate": result["status"] == "PASS",
               "forbidden_data_accessed": False, "optimizer_steps": 0,
               "production_core_unchanged": True,
               "scientific_decision": "H50_NOT_RUN_GATE_A_FAIL" if result["status"] != "PASS" else "GATE_B_C_NOT_IMPLEMENTED_IN_THIS_RUN",
               "report_path": "agents/reports/hope_ol04e_horizon_matched_intervention.md"}
    atomic_json(OUT / "summary.json", summary)
    return summary

