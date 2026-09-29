# scripts/hope_cad/probe_real_features.py
"""Real frozen-feature viability diagnostics for the HOPE representation core."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import yaml

from dataset.benchmark_manifest_v1 import build_training_manifest
from dataset.benchmark_protocol_v1 import MVTecContinualProtocol
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor
from models.hope_cad import HopeBlock
from models.hope_cad import state as state_utils


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_RELATIVE = Path("checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth")
EXPECTED_CHECKPOINT_SHA = "ae3012808a9b406a19b799381bd26b253634ad26125937c26d02dbcbbc85dd92"
CLASSES = ("bottle", "carpet", "grid", "toothbrush", "transistor")
IMAGES_PER_CLASS = 40
SENSITIVITY_PER_CLASS = 10
FEATURE_SHAPE = (784, 768)
FLOAT_EPS = float(torch.finfo(torch.float32).eps)
RELATIVE_EPS = 1e-12


@dataclass(frozen=True)
class ProbeConfig:
    seed: int
    device: str
    feature_batch_size: int
    run_id: str
    max_classes: int = len(CLASSES)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def load_protocol() -> dict[str, Any]:
    protocol = yaml.safe_load(
        (REPO_ROOT / "conf/benchmarks/protocols/mvtec_1x15_v1.yaml").read_text()
    )
    protocol["dataset"]["root"] = str((REPO_ROOT / "data/mvtec").resolve())
    return protocol


def selected_rows(protocol: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    manifest = build_training_manifest(protocol, seed, "hope-real-feature-probe")
    rows: list[dict[str, Any]] = []
    for class_name in CLASSES:
        class_rows = [row for row in manifest["entries"] if row["task_name"] == class_name]
        rows.extend(class_rows[:IMAGES_PER_CLASS])
    if len(rows) != len(CLASSES) * IMAGES_PER_CLASS:
        raise RuntimeError("selected normal stream does not contain 200 images")
    return rows


def make_protocol_loader(protocol: dict[str, Any], rows: list[dict[str, Any]], batch_size: int):
    manifest = build_training_manifest(protocol, 0, "hope-real-feature-probe")
    # The protocol constructor validates the complete manifest. The selected
    # rows are then passed through its existing exact preprocessing loader.
    protocol_instance = MVTecContinualProtocol(protocol, manifest, {
        "runtime": {"batch_size": batch_size},
        "preprocessing": {
            "image_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }, 0, 0)
    return protocol_instance._loader(rows, training=True)


def make_extractor(device: torch.device, checkpoint: Path) -> CADICViTFeatureExtractor:
    config = CADICViTConfig(
        checkpoint_path=str(checkpoint),
        checkpoint_identity=f"sha256:{EXPECTED_CHECKPOINT_SHA}",
    )
    return CADICViTFeatureExtractor(config, device)


def feature_metadata(checkpoint: Path, sha: str) -> dict[str, Any]:
    return {
        "model_name": "vit_base_patch8_224",
        "pretraining": "ImageNet-21k compatibility metadata",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha,
        "image_size": 224,
        "patch_size": 8,
        "layer_number": 9,
        "layer_indexing": "one_based",
        "block_index": 8,
        "cls_removed": True,
        "feature_shape_per_image": list(FEATURE_SHAPE),
        "feature_normalization": "none",
        "preprocessing": {
            "resize": [224, 224],
            "resize_rule": "PIL direct square bilinear",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }


def cache_paths(cache_dir: Path, class_name: str) -> Path:
    return cache_dir / f"class_{class_name}.pt"


def cache_payload(
    class_name: str,
    patches: torch.Tensor,
    paths: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "patches": patches.detach().cpu().to(torch.float32).contiguous(),
        "relative_paths": list(paths),
        "class_name": class_name,
        "metadata": copy.deepcopy(metadata),
        "selection_rule": "first 40 lexicographic train/good images",
        "dtype": "float32",
        "shape": list(patches.shape),
    }


def validate_cache_payload(payload: dict[str, Any], class_name: str, metadata: dict[str, Any]) -> None:
    required = {"patches", "relative_paths", "class_name", "metadata", "selection_rule", "dtype", "shape"}
    if not required.issubset(payload):
        raise ValueError(f"cache {class_name} is missing required metadata")
    patches = payload["patches"]
    if payload["class_name"] != class_name or payload["selection_rule"] != "first 40 lexicographic train/good images":
        raise ValueError(f"cache selection metadata mismatch for {class_name}")
    if not isinstance(patches, torch.Tensor) or patches.dtype != torch.float32 or tuple(patches.shape) != (IMAGES_PER_CLASS, *FEATURE_SHAPE):
        raise ValueError(f"cache tensor shape/dtype mismatch for {class_name}")
    if payload["dtype"] != "float32" or payload["shape"] != list(patches.shape):
        raise ValueError(f"cache tensor metadata mismatch for {class_name}")
    if len(payload["relative_paths"]) != IMAGES_PER_CLASS or list(payload["relative_paths"]) != sorted(payload["relative_paths"]):
        raise ValueError(f"cache path ordering mismatch for {class_name}")
    if payload["metadata"] != metadata:
        raise ValueError(f"cache feature identity/preprocessing mismatch for {class_name}")
    if not torch.isfinite(patches).all().item():
        raise ValueError(f"cache contains non-finite features for {class_name}")


def build_or_load_cache(
    protocol: dict[str, Any],
    rows: list[dict[str, Any]],
    extractor: CADICViTFeatureExtractor,
    cache_dir: Path,
    metadata: dict[str, Any],
    batch_size: int,
    force: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    by_class = {name: [row for row in rows if row["task_name"] == name] for name in CLASSES}
    loaders = {
        name: make_protocol_loader(protocol, class_rows, batch_size)
        for name, class_rows in by_class.items()
    }
    feature_seconds = 0.0
    records: list[dict[str, Any]] = []
    for class_name in CLASSES:
        path = cache_paths(cache_dir, class_name)
        if path.is_file() and not force:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            validate_cache_payload(payload, class_name, metadata)
        else:
            parts: list[torch.Tensor] = []
            relative_paths: list[str] = []
            for batch in loaders[class_name]:
                started = time.perf_counter()
                images = batch["images"].to(extractor.device)
                with torch.no_grad():
                    features = extractor.extract_patch_features(images)
                feature_seconds += time.perf_counter() - started
                if tuple(features.shape[1:]) != FEATURE_SHAPE or features.dtype != torch.float32:
                    raise RuntimeError(f"unexpected extracted shape for {class_name}: {tuple(features.shape)}")
                parts.append(features.detach().cpu())
                relative_paths.extend(list(batch["relative_path"]))
            patches = torch.cat(parts, dim=0)
            payload = cache_payload(class_name, patches, relative_paths, metadata)
            validate_cache_payload(payload, class_name, metadata)
            torch.save(payload, path)
        for index, relative_path in enumerate(payload["relative_paths"]):
            records.append({
                "class_name": class_name,
                "relative_path": relative_path,
                "patches": payload["patches"][index],
            })
    return records, {
        "cache_feature_seconds": feature_seconds,
        "cache_bytes": sum(path.stat().st_size for path in cache_dir.glob("class_*.pt")),
        "cache_files": [str(cache_paths(cache_dir, name).name) for name in CLASSES],
    }


def verify_cache_equivalence(
    protocol: dict[str, Any],
    records: list[dict[str, Any]],
    extractor: CADICViTFeatureExtractor,
) -> dict[str, Any]:
    max_abs = 0.0
    checked: list[str] = []
    for class_name in CLASSES:
        record = next(row for row in records if row["class_name"] == class_name)
        row = {
            "task_id": 0,
            "task_name": class_name,
            "split": "train",
            "relative_path": record["relative_path"],
        }
        loader = make_protocol_loader(protocol, [row], 1)
        image = next(iter(loader))["images"].to(extractor.device)
        with torch.no_grad():
            direct = extractor.extract_patch_features(image).detach().cpu()[0]
        difference = float((direct - record["patches"]).abs().max().item())
        max_abs = max(max_abs, difference)
        checked.append(record["relative_path"])
    if max_abs > 1e-6:
        raise RuntimeError(f"direct/cache feature mismatch: max_abs={max_abs}")
    return {"checked_paths": checked, "max_abs_difference": max_abs, "tolerance": 1e-6}


def stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    rows = value.reshape(-1, value.shape[-1]) if value.ndim >= 2 else value.reshape(-1, 1)
    patch_norms = rows.norm(dim=-1)
    centered = rows - rows.mean(dim=0, keepdim=True)
    centered_variance = float(centered.square().mean().item())
    return {
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
        "average_patch_norm": float(patch_norms.mean().item()),
        "patch_norm_p01": float(torch.quantile(patch_norms, 0.01).item()),
        "patch_norm_median": float(torch.quantile(patch_norms, 0.50).item()),
        "patch_norm_p99": float(torch.quantile(patch_norms, 0.99).item()),
        "centered_patch_variance": centered_variance,
        "finite": bool(torch.isfinite(value).all().item()),
    }


def six_stats(tensor: torch.Tensor) -> dict[str, float]:
    flat = tensor.detach().float().reshape(-1).cpu()
    quantiles = torch.quantile(flat, torch.tensor([0.01, 0.50, 0.99]))
    return {
        "min": float(flat.min().item()),
        "p01": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "mean": float(flat.mean().item()),
        "p99": float(quantiles[2].item()),
        "max": float(flat.max().item()),
    }


def norm_delta(before: torch.Tensor, after: torch.Tensor) -> tuple[float, float]:
    delta = float((after.detach() - before.detach()).float().norm().item())
    denominator = float(before.detach().float().norm().item())
    return delta, delta / (denominator + RELATIVE_EPS)


def state_norms(model: HopeBlock) -> dict[str, float]:
    return {
        name: float(value.detach().float().norm().item())
        for name, value in model.smt.memory_state().items()
    }


def cms_level_stats(model: HopeBlock) -> list[dict[str, Any]]:
    output = []
    for level in model.cms.levels:
        current_norm = math.sqrt(sum(float(value.detach().float().square().sum().item()) for value in level.current_tensors()))
        accumulator_norm = math.sqrt(sum(float(value.detach().float().square().sum().item()) for value in level.gradient_accumulators()))
        output.append({
            "state_norm": current_norm,
            "accumulator_norm": accumulator_norm,
            "pending_count": int(level.pending_count.item()),
            "update_count": int(level.update_count.item()),
        })
    return output


def deterministic_samples(seed: int, count: int = 128, pairs: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(FEATURE_SHAPE[0], generator=generator)[:count]
    pair_indices = torch.randint(FEATURE_SHAPE[0], (pairs, 2), generator=generator)
    return indices, pair_indices


def cosine_summary(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    values = F.cosine_similarity(left.float(), right.float(), dim=-1, eps=1e-12)
    return {f"cosine_{key}": value for key, value in six_stats(values).items()}


def heavy_geometry(
    values: dict[str, torch.Tensor],
    sample_indices: torch.Tensor,
    pair_indices: torch.Tensor,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    sampled: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        rows = value.detach().float().reshape(-1, value.shape[-1])
        sampled[name] = rows[sample_indices]
        centered = sampled[name] - sampled[name].mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(centered)
        probabilities = singular / (singular.sum() + RELATIVE_EPS)
        effective_rank = float(torch.exp(-(probabilities * (probabilities + RELATIVE_EPS).log()).sum()).item())
        result[f"{name}_effective_rank"] = effective_rank
        result[f"{name}_top_singular_ratio"] = float((singular[0] / (singular.sum() + RELATIVE_EPS)).item())
        pair_left = rows[pair_indices[:, 0]]
        pair_right = rows[pair_indices[:, 1]]
        result.update({f"{name}_{key}": val for key, val in cosine_summary(pair_left, pair_right).items()})
        result[f"{name}_covariance_trace"] = float(centered.square().sum().item())
    result.update({f"vit_smt_{key}": value for key, value in cosine_summary(sampled["vit"], sampled["smt"]).items()})
    result.update({f"smt_hope_{key}": value for key, value in cosine_summary(sampled["smt"], sampled["hope"]).items()})
    return result


def objectives(count: int, preserve: bool):
    def objective(_index, level_input, level_output, _metadata):
        if preserve:
            return 0.5 * (level_output - level_input.detach()).square().mean()
        return 0.5 * level_output.square().mean()
    return [objective] * count


def make_hope(device: torch.device) -> HopeBlock:
    model = HopeBlock(
        768,
        adaptive_q=False,
        memory_chunk_size=16,
        auxiliary_memory_chunk_size=16,
        cms_update_periods=(1, 8),
        cms_learning_rates=(1e-3, 1e-3),
        cms_hidden_dim=768,
    )
    return model.to(device)


def heavy_events(event_id: int) -> bool:
    return event_id == 1 or event_id % 10 == 0 or event_id in {41, 81, 121, 161, 200}


def event_row(
    model: HopeBlock,
    record: dict[str, Any],
    result: Any,
    event_id: int,
    previous_class: str | None,
    before_memory: dict[str, torch.Tensor],
    before_bytes: int,
    before_keys: int,
    heavy: dict[str, Any] | None,
    commit_seconds: float,
    diagnostic_seconds: float,
) -> dict[str, Any]:
    vit = record["patches"]
    smt = result.smt_representation[0].detach().cpu()
    hope = result.output[0].detach().cpu()
    current_memory = {name: value.detach().cpu() for name, value in model.smt.memory_state().items()}
    row: dict[str, Any] = {
        "event_id": event_id,
        "class_name": record["class_name"],
        "relative_path": record["relative_path"],
        "is_class_boundary": previous_class is not None and previous_class != record["class_name"],
        "previous_class": previous_class,
        "completed_events": int(model.cms.completed_events.item()),
        "due_levels": list(result.due_levels),
        "finite": bool(torch.isfinite(result.output).all().item()),
        "state_bytes": model.memory_stats()["hope_full_tensor_bytes"],
        "state_key_count": len(model.state_dict()),
        "persistent_grad_fn": any(value.grad_fn is not None for value in model.buffers()),
        "commit_runtime": commit_seconds,
        "diagnostic_runtime": diagnostic_seconds,
    }
    for prefix, value in (("vit", vit), ("smt", smt), ("hope", hope)):
        for key, item in stats(value).items():
            row[f"{prefix}_{key}"] = item
    for level_index, level_output in enumerate(result.cms_level_outputs):
        level_stats = stats(level_output[0].detach().cpu())
        for key, value in level_stats.items():
            row[f"cms_level_{level_index}_output_{key}"] = value
    controls = {"eta": result.eta[0].detach().cpu(), "alpha": result.alpha[0].detach().cpu()}
    for name, value in controls.items():
        for key, item in six_stats(value).items():
            row[f"{name}_{key}"] = item
    for name in ("memory", "k", "v", "eta", "alpha"):
        before = before_memory[name]
        after = current_memory[name]
        absolute, relative = norm_delta(before, after)
        row[f"smt_{name}_norm_before"] = float(before.float().norm().item())
        row[f"smt_{name}_norm_after"] = float(after.float().norm().item())
        row[f"smt_{name}_delta_norm"] = absolute
        row[f"smt_{name}_relative_delta"] = relative
    levels = cms_level_stats(model)
    for index, level in enumerate(levels):
        for key, value in level.items():
            row[f"cms_level_{index}_{key}"] = value
        row[f"cms_level_{index}_due"] = index in result.due_levels
    row["heavy_geometry_sampled"] = heavy is not None
    if heavy is not None:
        row.update(heavy)
    else:
        for key in ("vit_effective_rank", "smt_effective_rank", "hope_effective_rank"):
            row[key] = None
    row["state_bytes_before"] = before_bytes
    row["state_key_count_before"] = before_keys
    return row


def write_event_table(path: Path, rows: list[dict[str, Any]]) -> str:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        pq.write_table(pa.Table.from_pylist(rows), path)
        return "parquet"
    except ImportError:
        # Keep the required artifact name and provide a dependency-free,
        # reopenable fallback in environments without an Arrow backend.
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=True) + "\n")
        path.with_suffix(path.suffix + ".format.json").write_text(
            json.dumps({"format": "jsonl-fallback", "reason": "pyarrow unavailable"}, indent=2)
        )
        return "jsonl-fallback"


def read_event_table(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
        return pq.read_table(path).to_pylist()
    except ImportError:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_read_only(records: list[dict[str, Any]], device: torch.device, seed: int) -> dict[str, Any]:
    seed_everything(seed)
    model = make_hope(device)
    before = state_utils.snapshot_persistent_state(model.state_dict())
    started = time.perf_counter()
    outputs = []
    # Read-only SMT/CMS paths are batch-safe; larger transport batches keep
    # this reference view practical while preserving exact state semantics.
    for start in range(0, len(records), 16):
        features = torch.stack([row["patches"] for row in records[start:start + 16]]).to(device)
        outputs.append(model(features).detach().cpu())
    elapsed = time.perf_counter() - started
    after = model.state_dict()
    return {
        "seconds": elapsed,
        "seconds_per_image": elapsed / len(records),
        "state_equal": state_utils.persistent_states_equal(before, after),
        "output_finite": bool(torch.isfinite(torch.cat(outputs)).all().item()),
        "output_shape": list(torch.cat(outputs).shape),
    }


def run_continual(
    records: list[dict[str, Any]],
    device: torch.device,
    preserve_objective: bool,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    seed_everything(seed)
    model = make_hope(device)
    objective_set = objectives(model.cms.K, preserve_objective)
    rows: list[dict[str, Any]] = []
    started_total = time.perf_counter()
    previous_class: str | None = None
    baseline_bytes = model.memory_stats()["hope_full_tensor_bytes"]
    baseline_keys = len(model.state_dict())
    all_finite = True
    graph_free = True
    schedule_valid = True
    memory_trajectory: list[dict[str, float]] = []
    output_trajectory: list[dict[str, float]] = []
    for event_id, record in enumerate(records, start=1):
        features = record["patches"].unsqueeze(0).to(device)
        before_memory = {name: value.detach().cpu().clone() for name, value in model.smt.memory_state().items()}
        before_bytes = model.memory_stats()["hope_full_tensor_bytes"]
        before_keys = len(model.state_dict())
        event_started = time.perf_counter()
        result = model.commit_image(
            features,
            objective_set,
            {"class_name": record["class_name"], "relative_path": record["relative_path"], "event_id": event_id},
        )
        commit_seconds = time.perf_counter() - event_started
        heavy = None
        diagnostic_started = time.perf_counter()
        if heavy_events(event_id):
            indices, pairs = deterministic_samples(seed + event_id)
            values = {
                "vit": record["patches"],
                "smt": result.smt_representation[0].detach().cpu(),
                "cms_level_0": result.cms_level_outputs[0][0].detach().cpu(),
                "hope": result.output[0].detach().cpu(),
            }
            heavy = heavy_geometry(values, indices, pairs)
        diagnostic_seconds = time.perf_counter() - diagnostic_started
        row = event_row(
            model, record, result, event_id, previous_class, before_memory,
            before_bytes, before_keys, heavy, commit_seconds, diagnostic_seconds,
        )
        rows.append(row)
        all_finite = all_finite and bool(row["finite"]) and all(value is not None and math.isfinite(float(value)) for value in row.values() if isinstance(value, float))
        graph_free = graph_free and not bool(row["persistent_grad_fn"])
        schedule_valid = schedule_valid and row["completed_events"] == event_id and row["cms_level_0_update_count"] == event_id and row["cms_level_1_update_count"] == event_id // 8
        memory_trajectory.append({name: row[f"smt_{name}_norm_after"] for name in ("memory", "k", "v", "eta", "alpha")})
        output_trajectory.append({"smt": row["smt_average_patch_norm"], "hope": row["hope_average_patch_norm"], "hope_variance": row["hope_centered_patch_variance"]})
        previous_class = record["class_name"]
    total_seconds = time.perf_counter() - started_total
    event_path = output_dir / "per_event.parquet"
    table_format = write_event_table(event_path, rows)
    return {
        "model": model,
        "rows": rows,
        "event_table_format": table_format,
        "seconds": total_seconds,
        "seconds_per_image": total_seconds / len(records),
        "state_bytes": baseline_bytes,
        "state_bytes_constant": all(row["state_bytes"] == baseline_bytes for row in rows),
        "state_keys_constant": all(row["state_key_count"] == baseline_keys for row in rows),
        "mechanical_finite": all_finite,
        "persistent_graph_free": graph_free,
        "schedule_valid": schedule_valid,
        "memory_trajectory": memory_trajectory,
        "output_trajectory": output_trajectory,
        "first_row": rows[0],
        "last_row": rows[-1],
        "boundary_rows": [row for row in rows if row["is_class_boundary"]],
        "event_count": len(rows),
    }


def run_sensitivity(records: list[dict[str, Any]], device: torch.device, seed: int, output_dir: Path) -> dict[str, Any]:
    subset: list[dict[str, Any]] = []
    for class_name in CLASSES:
        subset.extend(
            [row for row in records if row["class_name"] == class_name][:SENSITIVITY_PER_CLASS]
        )
    if len(subset) != len(CLASSES) * SENSITIVITY_PER_CLASS:
        raise RuntimeError("sensitivity subset does not contain 50 images")
    seed_everything(seed)
    base = make_hope(device)
    model_b = copy.deepcopy(base)
    model_a = copy.deepcopy(base)
    initial_equal = state_utils.persistent_states_equal(model_a.state_dict(), model_b.state_dict())
    output: dict[str, Any] = {"initial_state_equal": initial_equal, "runs": {}}
    for label, model, preserve in (("identity50", model_b, True), ("zero50", model_a, False)):
        objective_set = objectives(model.cms.K, preserve)
        rows = []
        started = time.perf_counter()
        for event_id, record in enumerate(subset, start=1):
            result = model.commit_image(
                record["patches"].unsqueeze(0).to(device),
                objective_set,
                {"class_name": record["class_name"], "relative_path": record["relative_path"], "event_id": event_id},
            )
            rows.append({
                "event_id": event_id,
                "class_name": record["class_name"],
                "output_norm": float(result.output.float().norm().item()),
                "smt_memory_norm": float(model.smt.memories["memory"].weight.float().norm().item()),
                "output_stats": stats(result.output[0].detach().cpu()),
                "eta": six_stats(result.eta),
                "alpha": six_stats(result.alpha),
                "finite": bool(torch.isfinite(result.output).all().item()),
            })
        table_path = output_dir / f"sensitivity_{label}.parquet"
        table_format = write_event_table(table_path, rows)
        output["runs"][label] = {
            "seconds": time.perf_counter() - started,
            "event_count": len(rows),
            "table_format": table_format,
            "first": rows[0],
            "last": rows[-1],
            "finite": all(row["finite"] for row in rows),
        }
    return output


def run_preflight(records: list[dict[str, Any]], device: torch.device, seed: int) -> dict[str, Any]:
    seed_everything(seed)
    model = make_hope(device)
    before = model.memory_stats()["hope_full_tensor_bytes"]
    results = []
    for event_id, record in enumerate(records[:8], start=1):
        result = model.commit_image(record["patches"].unsqueeze(0).to(device), objectives(2, True))
        results.append({
            "event_id": result.event_id,
            "due_levels": list(result.due_levels),
            "pending_counts": list(result.pending_counts),
            "update_counts": list(result.update_counts),
            "finite": bool(torch.isfinite(result.output).all().item()),
        })
    return {
        "events": results,
        "exact_event_count": int(model.cms.completed_events.item()) == 8,
        "level0_updates": int(model.cms.levels[0].update_count.item()) == 8,
        "level1_updates": int(model.cms.levels[1].update_count.item()) == 1,
        "state_bytes_constant": model.memory_stats()["hope_full_tensor_bytes"] == before,
        "persistent_graph_free": not any(value.grad_fn is not None for value in model.buffers()),
        "finite": all(item["finite"] for item in results),
    }


def classify_health(continual: dict[str, Any], read_only: dict[str, Any], sensitivity: dict[str, Any]) -> dict[str, Any]:
    rows = continual["rows"]
    final = continual["last_row"]
    mechanical_flags = {
        "finite": continual["mechanical_finite"],
        "bounded_state_bytes": continual["state_bytes_constant"],
        "stable_state_keys": continual["state_keys_constant"],
        "persistent_graph_free": continual["persistent_graph_free"],
        "scheduler_valid": continual["schedule_valid"],
        "read_only_state_equal": read_only["state_equal"],
        "read_only_finite": read_only["output_finite"],
    }
    final_zero = final["hope_average_patch_norm"] <= FLOAT_EPS
    final_variance_collapse = final["hope_centered_patch_variance"] <= FLOAT_EPS
    final_rank = final.get("hope_effective_rank")
    rank_collapse = final_rank is not None and final_rank <= 1.0 + 100.0 * FLOAT_EPS
    final_cosine_mean = final.get("hope_cosine_mean")
    final_cosine_std = final.get("hope_cosine_std")
    collinear = (
        final_cosine_mean is not None and final_cosine_std is not None
        and abs(final_cosine_mean) >= 1.0 - 100.0 * FLOAT_EPS
        and final_cosine_std <= 100.0 * FLOAT_EPS
    )
    initial_memory = rows[0]["smt_memory_norm_before"]
    final_memory = final["smt_memory_norm_after"]
    memory_machine_collapse = final_memory <= FLOAT_EPS * max(initial_memory, 1.0)
    numerical_flags = {
        "final_representation_zero": final_zero,
        "final_variance_machine_collapse": final_variance_collapse,
        "effective_rank_single_direction": rank_collapse,
        "sampled_cosines_collinear": collinear,
        "smt_memory_machine_collapse": memory_machine_collapse,
    }
    hard_failure = not all(mechanical_flags.values())
    degenerate = hard_failure or final_zero or final_variance_collapse or rank_collapse or collinear
    warning = memory_machine_collapse and not degenerate
    if degenerate:
        outcome = "DEGENERATE"
    elif warning:
        outcome = "HEALTHY_WITH_RETENTION_WARNING"
    else:
        outcome = "HEALTHY"
    return {
        "outcome": outcome,
        "mechanical_validity": all(mechanical_flags.values()),
        "numerical_health": not degenerate,
        "retention_warning": warning,
        "mechanical_flags": mechanical_flags,
        "numerical_flags": numerical_flags,
        "candidate_a_b_initial_state_equal": sensitivity["initial_state_equal"],
    }


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("real_%Y%m%dT%H%M%SZ_seed%d" % args.seed)
    output_dir = REPO_ROOT / "results/hope_cad/real_feature_probe" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "features"
    checkpoint = (REPO_ROOT / CHECKPOINT_RELATIVE).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = checkpoint_sha256(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError(f"checkpoint SHA mismatch: {checkpoint_sha}")
    protocol = load_protocol()
    rows = selected_rows(protocol, args.seed)
    extractor_started = time.perf_counter()
    extractor = make_extractor(device, checkpoint)
    checkpoint_load_seconds = time.perf_counter() - extractor_started
    metadata = feature_metadata(checkpoint, checkpoint_sha)
    records, cache_info = build_or_load_cache(
        protocol, rows, extractor, cache_dir, metadata, args.feature_batch_size, args.force_cache
    )
    cache_check = verify_cache_equivalence(protocol, records, extractor)
    smoke = run_preflight(records, device, args.seed)
    if not all(smoke.values()):
        raise RuntimeError(f"real-feature preflight failed: {smoke}")
    read_only = run_read_only(records, device, args.seed)
    continual = run_continual(records, device, True, output_dir, args.seed)
    sensitivity = run_sensitivity(records, device, args.seed, output_dir)
    health = classify_health(continual, read_only, sensitivity)
    summary = {
        "run_id": run_id,
        "status": "COMPLETED",
        "device": str(device),
        "dtype": "float32",
        "seed": args.seed,
        "feature_extractor": metadata,
        "class_order": list(CLASSES),
        "images_per_class": IMAGES_PER_CLASS,
        "total_events": len(records),
        "hope_config": {
            "adaptive_q": False,
            "memory_chunk_size": 16,
            "auxiliary_memory_chunk_size": 16,
            "cms_update_periods": [1, 8],
            "cms_hidden_dim": 768,
        },
        "probe_objective": "PROBE-ONLY PROJECT MAPPING: 0.5 * mean((y - stop_gradient(h))^2)",
        "sensitivity_objective": "PROBE-ONLY PROJECT MAPPING: 0.5 * mean(y^2)",
        "cache": {**cache_info, **cache_check},
        "state_bytes": continual["state_bytes"],
        "timings": {
            "checkpoint_load_seconds": checkpoint_load_seconds,
            "feature_extraction_seconds": cache_info["cache_feature_seconds"],
            "read_only_seconds": read_only["seconds"],
            "continual_commit_seconds": continual["seconds"],
            "sensitivity_identity_seconds": sensitivity["runs"]["identity50"]["seconds"],
            "sensitivity_zero_seconds": sensitivity["runs"]["zero50"]["seconds"],
        },
        "preflight": smoke,
        "read_only": read_only,
        "continual": {
            "event_count": continual["event_count"],
            "event_table_format": continual["event_table_format"],
            "seconds": continual["seconds"],
            "seconds_per_image": continual["seconds_per_image"],
            "mechanical_finite": continual["mechanical_finite"],
            "persistent_graph_free": continual["persistent_graph_free"],
            "state_bytes_constant": continual["state_bytes_constant"],
            "state_keys_constant": continual["state_keys_constant"],
            "schedule_valid": continual["schedule_valid"],
            "first_event": continual["first_row"],
            "last_event": continual["last_row"],
            "class_boundaries": continual["boundary_rows"],
        },
        "sensitivity": sensitivity,
        "health": health,
        "limitations": [
            "No anomaly images or anomaly metrics were used.",
            "The local checkpoint provenance is recorded as a compatibility assumption.",
            "per_event.parquet uses a JSONL fallback because pyarrow is unavailable in this environment.",
        ],
    }
    write_config(output_dir / "config_resolved.yaml", {
        "run_id": run_id,
        "seed": args.seed,
        "device": str(device),
        "feature_extractor": metadata,
        "classes": list(CLASSES),
        "images_per_class": IMAGES_PER_CLASS,
        "selection_rule": "first 40 lexicographic train/good images per class",
        "hope_config": summary["hope_config"],
        "objectives": {
            "candidate_b": summary["probe_objective"],
            "candidate_a": summary["sensitivity_objective"],
        },
        "heavy_diagnostics": {
            "events": "1, every 10th, every class boundary, and final 200",
            "patch_indices": "torch.randperm(784, generator=manual_seed(seed + event_id))[:128]",
            "cosine_pairs": "torch.randint(784, (256,2), generator=manual_seed(seed + event_id))",
            "effective_rank": "exp(-sum(p_i*log(p_i+1e-12))), p_i=s_i/(sum(s)+1e-12)",
            "relative_delta_epsilon": RELATIVE_EPS,
            "machine_precision": FLOAT_EPS,
        },
    })
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    # Reopen the primary table before reporting completion.
    reopened = read_event_table(output_dir / "per_event.parquet")
    if len(reopened) != len(records):
        raise RuntimeError("per-event artifact could not be reopened with the recorded event count")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-feature HOPE viability probe")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    if args.feature_batch_size < 1:
        raise ValueError("feature-batch-size must be positive")
    torch.set_num_threads(1)
    summary = run(args)
    print(json.dumps({
        "run_id": summary["run_id"],
        "status": summary["status"],
        "health": summary["health"],
        "cache": summary["cache"],
        "timings": summary["timings"],
    }, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
