# scripts/exps/hope_anomaly_signal.py
"""Read-only associative-residual evaluation with paired normal memories."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
import yaml

from dataset.benchmark_protocol_v1 import _ImageDataset
from exps.anomaly.hope_anomaly_signal import (
    CATEGORIES, CHECKPOINTS, METHOD_NAMES, PooledCovariance, bootstrap_image_counts,
    development_manifests, evaluate_metrics, native_mask, pixel_map, residual_scores,
    score_distribution, sha256,
)
from exps.hope_image_synchronous_memory import ImageSynchronousMemory, fingerprint, synchronized_time
from models.cadic_patch_coreset_v1 import CADICPatchCoresetConfig, CADICPatchCoresetV1
from scripts.hope_cad.probe_real_features import EXPECTED_CHECKPOINT_SHA, feature_metadata, make_extractor


ROOT = Path(__file__).resolve().parents[3]
MEMORY_ROOT = ROOT / "results/hope_cad/memory_learning_gate"
DEFAULT_OUTPUT = ROOT / "results/hope_cad/anomaly_signal_gate"
DATA_ROOT = ROOT / "data/mvtec"
CHECKPOINT = ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"
CORE_HASHES = {
    "self_modifying_titans.py": "ffcd8ca7810ade758954effcb708903eb36b182835dac58e0e577cb497e1932c",
    "continuum_memory.py": "920135550843a33d2cc064030bd5155af349dbab0d99cddae213ddc589055581",
    "hope_block.py": "16e9b00cf9bba7e66ceaebe94d2c7f6cf23f95c907bbe59d47d879b16b179437",
}
CADIC_CONFIG = CADICPatchCoresetConfig(budget=2500, dim=768, chunk_size=256,
                                     query_chunk_size=256, pair_chunk_size=256, image_neighbors=9)
SCHEMA_VERSION = 1


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_tensor(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


def table_records(path):
    frame = pd.read_parquet(path).astype(object)
    return frame.where(pd.notna(frame), None).to_dict("records")


def progress(output, phase, **fields):
    payload = {"phase": phase, "time_utc": pd.Timestamp.now(tz="UTC").isoformat(), **fields}
    write_json(output / "progress.json", payload)
    print(json.dumps(payload), flush=True)


def validate_core():
    actual = {name: sha256(ROOT / "models/hope_cad" / name) for name in CORE_HASHES}
    if actual != CORE_HASHES:
        raise ValueError("locked production source identity changed")
    return actual


def load_fixture():
    payload = torch.load(MEMORY_ROOT / "p0_initial_fixture.pt", map_location="cpu", weights_only=False)
    if fingerprint(payload["smt"]) != payload["initialization_hash"]:
        raise ValueError("initial state identity differs")
    return payload


def manifest_identity(rows):
    import hashlib
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def prepare(output, device):
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifests").mkdir(exist_ok=True)
    validate_core()
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("local ViT checkpoint identity differs")
    dev, confirmation = development_manifests(DATA_ROOT)
    for name, rows in (("anomaly_dev_manifest", dev), ("anomaly_confirmation_manifest", confirmation)):
        path = output / "manifests" / f"{name}.parquet"
        if path.exists() and table_records(path) != rows:
            raise ValueError("previously fixed anomaly manifest differs")
        if not path.exists():
            save_table(path, rows)
        if not (output / f"{name}.parquet").exists():
            save_table(output / f"{name}.parquet", rows)
    feature_identity = feature_metadata(CHECKPOINT, EXPECTED_CHECKPOINT_SHA)
    fixture = load_fixture()
    cache_manifest = read_json(MEMORY_ROOT / "feature_cache_manifest.json")
    if cache_manifest["metadata"] != feature_identity:
        raise ValueError("training feature contract differs")
    for category, shard in cache_manifest["shards"].items():
        path = Path(shard["path"])
        if sha256(path) != shard["sha256"]:
            raise ValueError(f"training cache identity differs: {category}")
    state_rows = []
    for seed in range(3):
        training = pd.read_parquet(MEMORY_ROOT / f"seed{seed}/stream_manifest.parquet")
        training = training[(training.role == "update") & (training.event_id <= 300)]
        if len(training) != 300 or training.class_name.tolist() != [c for c in CATEGORIES for _ in range(100)]:
            raise ValueError("normal training stream differs")
        if not training.relative_path.str.contains("/train/good/", regex=False).all():
            raise ValueError("training source is not train/good")
        save_table(output / f"seed{seed}/stream_manifest.parquet", training.to_dict("records"))
        for raw, method in METHOD_NAMES.items():
            for event in CHECKPOINTS:
                source = MEMORY_ROOT / (f"seed{seed}/checkpoints/{raw}_event{event}.pt" if event else "p0_initial_fixture.pt")
                if not source.exists():
                    raise FileNotFoundError(source)
                state_rows.append({"seed": seed, "method": method, "checkpoint": event,
                                   "source_path": str(source), "source_sha256": sha256(source)})
    save_table(output / "states/smt_state_manifest.parquet", state_rows)
    # Example identities are selected by mask extent before any detector scores.
    examples = []
    for category in CATEGORIES:
        rows = [row for row in dev if row["category"] == category]
        positives = [row for row in rows if row["label"]]
        positives.sort(key=lambda row: (float(native_mask(DATA_ROOT, row).mean()), row["relative_path"]))
        for reason, row in (("smallest_mask_extent", positives[0]), ("largest_mask_extent", positives[-1]),
                            ("first_normal", next(row for row in rows if not row["label"]))):
            examples.append({"category": category, "reason": reason, "image_id": row["image_id"]})
    write_json(output / "manifests/visualization_selection.json", examples)
    config = {
        "schema_version": SCHEMA_VERSION, "device": str(device), "order_seeds": [0, 1, 2],
        "initialization_seed": 0, "initialization_hash": fixture["initialization_hash"],
        "feature_contract": feature_identity, "development_manifest_identity": manifest_identity(dev),
        "confirmation_manifest_identity": manifest_identity(confirmation), "confirmation_evaluated": False,
        "classes": list(CATEGORIES), "checkpoints": list(CHECKPOINTS), "core_hashes": CORE_HASHES,
        "training": "Reuse completed 100 bottle + 100 carpet + 100 hazelnut states; no bottle return",
        "methods": {"P0_BASE": "alpha=1; eta=(0.02/784)*sigmoid(raw_eta)",
                    "P1_SYNC": "mean-weighted image-snapshot SR-DGD h=0.02; alpha_image=1",
                    "P2_PROJ": "P1 full-transition dense spectral projection at singular value 1",
                    "FROZEN": "identical original initialization; no mutable writes"},
        "CMS": "DISABLED", "scoring": "sum_d (M_memory k - v)^2; current snapshot k/v; no self-target transform",
        "image_score": "maximum patch score", "pixel_map": "28x28 bilinear to native GT size; align_corners=False",
        "pixel_AUPR": "sklearn average_precision_score, un-interpolated weighted step integral",
        "bootstrap": {"unit": "IMAGE", "paired": True, "stratified_by_label": True, "repetitions": 400,
                      "seed_rule": "category_index + 1000*order_seed", "interval": "2.5/97.5 percentile",
                      "pixel_intervals": "native per-image threshold histograms; exact point AP retained",
                      "max_histogram_point_AP_error": 0.0002},
        "covariance": {"basis": "fixed normalized q", "precision": "float64",
                       "covariance": "unbiased pooled scatter/(count-1)",
                       "shrinkage": "0.9*Sigma + 0.1*trace(Sigma)/d*I + 1e-6*I"},
        "CADIC": {"route": "ordinary CADIC-compatible, not paperfaithful", "config": CADIC_CONFIG.__dict__,
                  "training_batch_images": 8, "module": "models.cadic_patch_coreset_v1.CADICPatchCoresetV1",
                  "adapter_reference": "models.cadic_benchmark_adapter_v1.CADICBenchmarkAdapterV1",
                  "source_sha256": sha256(ROOT / "models/cadic_patch_coreset_v1.py"),
                  "config_path": "conf/benchmarks/methods/cadic_compatible_v1.yaml",
                  "config_sha256": sha256(ROOT / "conf/benchmarks/methods/cadic_compatible_v1.yaml"),
                  "score": "native weighted support-neighbor image score; native nearest Euclidean pixel score",
                  "event0": "NOT_FITTED"},
    }
    config_path = output / "config_resolved.yaml"
    if config_path.exists():
        previous = yaml.safe_load(config_path.read_text())
        if previous != config:
            raise ValueError("resolved configuration differs; refusing silent reuse")
    else:
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    cache_path = output / "features/development.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=False, map_location="cpu", mmap=True)
        validate_dev_cache(cache, dev, feature_identity)
        progress(output, "development_cache_reused", images=len(dev), bytes=cache_path.stat().st_size)
        return
    cache_path.parent.mkdir(exist_ok=True)
    start = synchronized_time(device)
    extractor = make_extractor(device, CHECKPOINT)
    load_seconds = synchronized_time(device) - start
    dataset = _ImageDataset(DATA_ROOT, dev, 224, [0.485, .456, .406], [.229, .224, .225], training=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(0))
    chunks = []
    start = synchronized_time(device)
    for batch in loader:
        patches = extractor.extract_patch_features(batch["images"].to(device)).detach().cpu()
        if patches.shape[1:] != (784, 768) or patches.dtype != torch.float32 or not torch.isfinite(patches).all():
            raise ValueError("development feature contract failed")
        chunks.append(patches)
        progress(output, "development_features", images=sum(len(x) for x in chunks), total=len(dev))
    extraction_seconds = synchronized_time(device) - start
    cache = {"patches": torch.cat(chunks), "relative_paths": [row["relative_path"] for row in dev],
             "metadata": feature_identity, "manifest_identity": manifest_identity(dev)}
    validate_dev_cache(cache, dev, feature_identity)
    # Direct extraction of one real development image validates the new cache.
    direct = extractor.extract_patch_features(dataset[0]["images"].unsqueeze(0).to(device)).cpu()[0]
    reference = cache["patches"][0]
    difference = float((direct.double() - reference.double()).abs().max())
    if not torch.allclose(direct, reference, atol=2e-4, rtol=3.1e-5):
        raise ValueError("direct development extraction/cache mismatch")
    save_tensor(cache_path, cache)
    write_json(output / "feature_cache_validation.json", {"passed": True, "max_abs_difference": difference,
               "checkpoint_load_seconds": load_seconds, "extraction_seconds": extraction_seconds,
               "images": len(dev), "bytes": cache_path.stat().st_size, "sha256": sha256(cache_path)})


def validate_dev_cache(cache, dev, metadata):
    patches = cache["patches"]
    if cache["metadata"] != metadata or cache["manifest_identity"] != manifest_identity(dev):
        raise ValueError("development cache compatibility identity differs")
    if cache["relative_paths"] != [row["relative_path"] for row in dev]:
        raise ValueError("development cache image ordering differs")
    if patches.shape != (len(dev), 784, 768) or patches.dtype != torch.float32 or not torch.isfinite(patches).all():
        raise ValueError("development cache tensor differs")


def evaluation_data(output):
    rows = table_records(output / "manifests/anomaly_dev_manifest.parquet")
    cache = torch.load(output / "features/development.pt", weights_only=False, map_location="cpu", mmap=True)
    validate_dev_cache(cache, rows, yaml.safe_load((output / "config_resolved.yaml").read_text())["feature_contract"])
    return rows, cache["patches"]


def smt_model(raw, seed, event, fixture, device):
    if not event or raw == "FROZEN":
        return ImageSynchronousMemory(fixture["smt"], "FROZEN", device=device)
    payload = torch.load(MEMORY_ROOT / f"seed{seed}/checkpoints/{raw}_event{event}.pt", weights_only=False, map_location="cpu")
    if payload["method"] != raw or payload["completed_events"] != event:
        raise ValueError("normal checkpoint method/event differs")
    model = ImageSynchronousMemory.deserialize_state(payload, device=device)
    updates = event * (49 if raw == "P0" else 1)
    if int(model.smt.memory_update_count) != updates or int(model.smt.auxiliary_update_count) != updates:
        raise ValueError("normal checkpoint counters differ")
    return model


def evaluate_unit(output, seed, method, event, category, rows, patches, scorer, storage, update_ms):
    folder = output / f"seed{seed}/units"
    folder.mkdir(parents=True, exist_ok=True)
    unit = folder / f"{method}_event{event}_{category}"
    context = {"seed": seed, "method": method, "checkpoint": event, "category": category}
    if unit.with_suffix(".json").exists():
        previous = read_json(unit.with_suffix(".json"))
        if previous["context"] != context or not unit.with_suffix(".npz").exists():
            raise ValueError("evaluation unit identity differs")
        return previous
    ids = [i for i, row in enumerate(rows) if row["category"] == category]
    subset = [rows[i] for i in ids]
    scores, patch_scores, diagnostics = [], [], []
    score_start = time.perf_counter()
    for index in ids:
        result = scorer(patches[index:index + 1])
        patch_scores.append(result.pop("patch_scores").reshape(784).numpy())
        scores.append(result.pop("image_score"))
        diagnostics.append(result)
    scoring_seconds = time.perf_counter() - score_start
    masks = [native_mask(DATA_ROOT, row) for row in subset]
    maps = [pixel_map(torch.from_numpy(array), mask.shape) for array, mask in zip(patch_scores, masks)]
    labels = np.asarray([row["label"] for row in subset])
    counts = bootstrap_image_counts(labels, seed=1000 * seed + CATEGORIES.index(category))
    metric_start = time.perf_counter()
    metrics = evaluate_metrics(scores, labels, maps, masks, counts)
    distributions = score_distribution(scores, labels, maps, masks)
    metric_seconds = time.perf_counter() - metric_start
    images = []
    for row, score, array, diagnostic, values, mask in zip(subset, scores, patch_scores, diagnostics, maps, masks):
        pixel_stats = {"background_score_mean": float(values[~mask].mean()),
                       "defect_score_mean": float(values[mask].mean()) if mask.any() else None}
        if mask.any():
            pixel_stats["defect_background_ratio"] = pixel_stats["defect_score_mean"] / max(pixel_stats["background_score_mean"], np.finfo(float).tiny)
        images.append({**context, **row, "image_score": score, "mean_patch_score": float(array.mean()), **diagnostic, **pixel_stats})
    payload = {"context": context, "metrics": {**context, **metrics.metrics, "n_normal": int((labels == 0).sum()),
               "n_anomaly": int((labels == 1).sum()), "status": "EVALUATED",
               "persistent_bytes": storage["persistent_bytes"], "update_ms_per_image": update_ms},
               "distribution": {**context, **distributions}, "images": images, "storage": storage,
               "scoring_seconds": scoring_seconds, "metrics_seconds": metric_seconds}
    np.savez_compressed(unit.with_suffix(".npz"), patch_scores=np.asarray(patch_scores),
                        image_ids=np.asarray([row["image_id"] for row in subset]), image_scores=np.asarray(scores),
                        bootstrap_image_AUROC=metrics.bootstrap_image_auroc,
                        bootstrap_pixel_AUPR=metrics.bootstrap_pixel_aupr)
    write_json(unit.with_suffix(".json"), payload)
    progress(output, "evaluated", **context, image_AUROC=metrics.metrics["image_AUROC"],
             pixel_AUPR=metrics.metrics["pixel_AUPR"], metrics_seconds=round(metric_seconds, 3))
    collect_tables(output, seed)
    return payload


def copy_evaluation(output, source, seed, method, event, category, storage, update_ms):
    target = output / f"seed{seed}/units/{method}_event{event}_{category}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.with_suffix(".json").exists():
        return
    payload = deepcopy(read_json(source.with_suffix(".json")))
    context = {"seed": seed, "method": method, "checkpoint": event, "category": category}
    # Scores are identical, but paired bootstrap resampling is seed-specific.
    arrays = dict(np.load(source.with_suffix(".npz"), allow_pickle=False))
    if payload["context"]["seed"] != seed:
        rows = payload["images"]
        masks = [native_mask(DATA_ROOT, row) for row in rows]
        maps = [pixel_map(torch.from_numpy(array), mask.shape) for array, mask in zip(arrays["patch_scores"], masks)]
        counts = bootstrap_image_counts(np.array([row["label"] for row in rows]), seed=1000 * seed + CATEGORIES.index(category))
        metrics = evaluate_metrics(arrays["image_scores"], [row["label"] for row in rows], maps, masks, counts)
        payload["metrics"].update(metrics.metrics)
        arrays["bootstrap_image_AUROC"] = metrics.bootstrap_image_auroc
        arrays["bootstrap_pixel_AUPR"] = metrics.bootstrap_pixel_aupr
    payload["context"] = context
    for name in ("metrics", "distribution"):
        payload[name].update(context)
    for row in payload["images"]:
        row.update(context)
    payload["metrics"].update({"persistent_bytes": storage["persistent_bytes"], "update_ms_per_image": update_ms})
    payload["storage"] = storage
    payload["scoring_seconds"] = 0.0
    payload["metrics_seconds"] = 0.0
    payload["reused_exact_scores_from"] = str(source)
    np.savez_compressed(target.with_suffix(".npz"), **arrays)
    write_json(target.with_suffix(".json"), payload)


def collect_tables(output, seed):
    directory = output / f"seed{seed}"
    units = [read_json(path) for path in sorted((directory / "units").glob("*.json"))]
    if not units:
        return
    metrics = [unit["metrics"] for unit in units]
    for method in ("COVARIANCE", "CADIC"):
        for category in CATEGORIES:
            metrics.append({"seed": seed, "method": method, "checkpoint": 0, "category": category,
                            "status": "NOT_FITTED", "n_normal": 10, "n_anomaly": 10})
    distributions = [unit["distribution"] for unit in units]
    save_table(directory / "metrics_by_checkpoint.parquet", metrics)
    save_table(directory / "image_scores.parquet", [row for unit in units for row in unit["images"]])
    save_table(directory / "pixel_metrics.parquet", metrics)
    save_table(directory / "score_distributions.parquet", distributions)
    save_table(directory / "defect_background.parquet", distributions)
    write_json(directory / "timing.json", {"evaluation": [{**unit["context"], "scoring_seconds": unit["scoring_seconds"],
                   "metrics_seconds": unit["metrics_seconds"]} for unit in units]})
    write_json(directory / "storage.json", [{**unit["context"], **unit["storage"]} for unit in units])


def residual_phase(output, device):
    rows, patches = evaluation_data(output)
    fixture = load_fixture()
    for seed in range(3):
        original = pd.read_parquet(MEMORY_ROOT / f"seed{seed}/per_event.parquet")
        for raw, method in METHOD_NAMES.items():
            for event in CHECKPOINTS:
                model = smt_model(raw, seed, event, fixture, device)
                stats = model.memory_stats()
                storage = {"persistent_bytes": stats["full_tensor_bytes"], "mutable_bytes": stats["mutable_bytes"],
                           "static_bytes": stats["full_tensor_bytes"] - stats["mutable_bytes"],
                           "model_probe_bytes": 0, "CMS_bytes": 0}
                timings = original[(original.method == raw) & (original.event_id <= event)]
                update_ms = 1000 * float(timings.update_seconds.mean()) if len(timings) and raw != "FROZEN" else 0.0
                state_before = model.state_fingerprint()
                for category in CATEGORIES:
                    frozen_unit = output / f"seed{seed}/units/FROZEN_event0_{category}"
                    if frozen_unit.with_suffix(".json").exists() and (raw == "FROZEN" or event == 0):
                        copy_evaluation(output, frozen_unit, seed, method, event, category, storage, update_ms)
                    else:
                        evaluate_unit(output, seed, method, event, category, rows, patches,
                                      lambda image: residual_scores(model, image.to(device)), storage, update_ms)
                if model.state_fingerprint() != state_before or model.memory_stats()["persistent_graph"]:
                    raise ValueError("read-only residual phase changed source state")
                collect_tables(output, seed)
                del model


def training_cache():
    return {category: torch.load(MEMORY_ROOT / f"features/class_{category}.pt", weights_only=False,
                               mmap=True, map_location="cpu") for category in CATEGORIES}


def training_features(row, cache):
    shard = cache[row["class_name"]]
    index = row["cache_index"]
    if shard["relative_paths"][index] != row["relative_path"] or "/train/good/" not in row["relative_path"]:
        raise ValueError("normal training cache/manifest pairing differs")
    return shard["patches"][index:index + 1]


def covariance_phase(output, device):
    rows, patches = evaluation_data(output)
    cache, fixture = training_cache(), load_fixture()
    basis = ImageSynchronousMemory(fixture["smt"], "FROZEN", device=device)
    basis_identity = basis.state_fingerprint()
    basis_implementation_bytes = basis.memory_stats()["full_tensor_bytes"]
    basis_minimum_bytes = sum(parameter.numel() * parameter.element_size()
                              for module in (basis.smt.local_conv, basis.smt.base_q) for parameter in module.parameters())
    for seed in range(3):
        model = PooledCovariance(768, device)
        stream = pd.read_parquet(output / f"seed{seed}/stream_manifest.parquet").to_dict("records")
        training_hash = manifest_identity(stream)
        times = []
        for event in (100, 200, 300):
            state_path = output / f"states/seed{seed}/COVARIANCE_event{event}.pt"
            if state_path.exists():
                saved = torch.load(state_path, weights_only=False, map_location="cpu")
                if saved["stream_identity"] != training_hash or saved["event"] != event:
                    raise ValueError("covariance continuation identity differs")
                model.load_state_dict(saved["state"])
                times = saved["update_seconds"]
            else:
                for row in stream[event - 100:event]:
                    start = synchronized_time(device)
                    quantities = basis.generate_update_quantities(training_features(row, cache).to(device), basis.snapshot_state())
                    model.update(quantities.queries)
                    times.append(synchronized_time(device) - start)
                start = synchronized_time(device)
                model.factorize()
                factor_seconds = synchronized_time(device) - start
                save_tensor(state_path, {"state": model.state_dict(), "stream_identity": training_hash,
                            "event": event, "factor_seconds": factor_seconds, "update_seconds": times})
                progress(output, "covariance_fitted", seed=seed, event=event, count=int(model.count))
            if int(model.completed_events) != event or int(model.count) != event * 784:
                raise ValueError("covariance clock/count differs")
            before = fingerprint(model.state_dict())
            def score(image):
                quantities = basis.generate_update_quantities(image.to(device), basis.snapshot_state())
                scores = model.score(quantities.queries)
                return {"patch_scores": scores, "image_score": float(scores.max())}
            for category in CATEGORIES:
                storage = model.storage()
                storage.update({"statistical_tensor_bytes": storage["persistent_bytes"],
                                "fixed_basis_implementation_bytes": basis_implementation_bytes,
                                "fixed_basis_minimum_required_bytes": basis_minimum_bytes,
                                "persistent_bytes": storage["persistent_bytes"] + basis_implementation_bytes})
                evaluate_unit(output, seed, "COVARIANCE", event, category, rows, patches, score,
                              storage, 1000 * float(np.mean(times)))
            if before != fingerprint(model.state_dict()) or basis.state_fingerprint() != basis_identity:
                raise ValueError("covariance evaluation mutated source state")


def cadic_phase(output, device):
    rows, patches = evaluation_data(output)
    cache = training_cache()
    for seed in range(3):
        model = CADICPatchCoresetV1(CADIC_CONFIG, device=device)
        stream = pd.read_parquet(output / f"seed{seed}/stream_manifest.parquet").to_dict("records")
        training_hash = manifest_identity(stream)
        times, completed = [], 0
        continuation = output / f"states/seed{seed}/CADIC_continuation.pt"
        for event in (100, 200, 300):
            state_path = output / f"states/seed{seed}/CADIC_event{event}.pt"
            if state_path.exists():
                saved = torch.load(state_path, weights_only=False, map_location="cpu")
                if saved["stream_identity"] != training_hash or saved["event"] != event:
                    raise ValueError("CADIC checkpoint identity differs")
                model.load_state_dict(saved["state"])
                times, completed = saved["update_seconds"], event
            else:
                if continuation.exists():
                    saved = torch.load(continuation, weights_only=False, map_location="cpu")
                    if saved["stream_identity"] != training_hash or saved["event"] > event:
                        raise ValueError("CADIC continuation identity differs")
                    if saved["event"] > completed:
                        model.load_state_dict(saved["state"])
                        times, completed = saved["update_seconds"], saved["event"]
                while completed < event:
                    end = min(event, completed + 8)
                    batch = torch.cat([training_features(row, cache) for row in stream[completed:end]]).to(device)
                    progress(output, "CADIC_update_started", seed=seed, events=[completed + 1, end])
                    start = synchronized_time(device)
                    model.update(batch)
                    elapsed = synchronized_time(device) - start
                    times.extend([elapsed / (end - completed)] * (end - completed))
                    completed = end
                    save_tensor(continuation, {"state": model.state_dict(), "event": completed,
                                "stream_identity": training_hash, "update_seconds": times})
                    progress(output, "CADIC_update_finished", seed=seed, event=completed,
                             seconds=round(elapsed, 3), memory_count=model.count)
                save_tensor(state_path, {"state": model.state_dict(), "stream_identity": training_hash,
                            "event": event, "update_seconds": times})
            if model.seen_features != event * 784:
                raise ValueError("CADIC normal feature count differs")
            before = fingerprint(model.state_dict())
            def score(image):
                image_scores, pixels = model.score(image.to(device))
                return {"patch_scores": pixels.detach().cpu(), "image_score": float(image_scores[0])}
            for category in CATEGORIES:
                evaluate_unit(output, seed, "CADIC", event, category, rows, patches, score,
                              {"persistent_bytes": model.memory_bytes, "feature_count": model.count,
                               "budget": CADIC_CONFIG.budget, "factorization_cache_bytes": 0},
                              1000 * float(np.mean(times)))
            if before != fingerprint(model.state_dict()):
                raise ValueError("CADIC scoring changed coreset or counters")


def paired_deltas(output, seed):
    units = {path.stem: read_json(path) for path in (output / f"seed{seed}/units").glob("*.json")}
    rows = []
    for name, payload in units.items():
        context = payload["context"]
        category, event, method = context["category"], context["checkpoint"], context["method"]
        own = np.load(output / f"seed{seed}/units/{name}.npz", allow_pickle=False)
        frozen_name = f"FROZEN_event{event}_{category}"
        if frozen_name not in units:
            continue
        frozen = np.load(output / f"seed{seed}/units/{frozen_name}.npz", allow_pickle=False)
        if not np.array_equal(own["image_ids"], frozen["image_ids"]):
            raise ValueError("paired bootstrap image identities differ")
        row = {**context}
        for metric in ("image_AUROC", "pixel_AUPR"):
            difference = own[f"bootstrap_{metric}"] - frozen[f"bootstrap_{metric}"]
            low, high = np.quantile(difference, [.025, .975])
            row[f"delta_{metric}_vs_frozen"] = payload["metrics"][metric] - units[frozen_name]["metrics"][metric]
            row[f"delta_{metric}_ci_low"] = float(low)
            row[f"delta_{metric}_ci_high"] = float(high)
        rows.append(row)
    save_table(output / f"seed{seed}/method_deltas.parquet", rows)
    return rows


def comparison_tables(metrics):
    rows = []
    for left, right in (("P1_SYNC", "P0_BASE"), ("P2_PROJ", "P1_SYNC")):
        a = metrics[metrics.method == left]
        b = metrics[metrics.method == right]
        for _, pair in a.merge(b, on=["seed", "checkpoint", "category"], suffixes=("_left", "_right")).iterrows():
            rows.append({"seed": pair.seed, "checkpoint": pair.checkpoint, "category": pair.category,
                         "comparison": f"{left}_minus_{right}",
                         "image_AUROC_difference": pair.image_AUROC_left - pair.image_AUROC_right,
                         "pixel_AUPR_difference": pair.pixel_AUPR_left - pair.pixel_AUPR_right})
    return rows


def visualizations(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selection = read_json(output / "manifests/visualization_selection.json")
    rows = table_records(output / "manifests/anomaly_dev_manifest.parquet")
    lookup = {row["image_id"]: row for row in rows}
    folder = output / "visualizations"
    folder.mkdir(exist_ok=True)
    scales = []
    for category in CATEGORIES:
        examples = [example for example in selection if example["category"] == category]
        scored = {}
        maximum = 0.0
        for method in METHOD_NAMES.values():
            unit = np.load(output / f"seed0/units/{method}_event300_{category}.npz", allow_pickle=False)
            for example in examples:
                index = unit["image_ids"].tolist().index(example["image_id"])
                row = lookup[example["image_id"]]
                array = pixel_map(torch.from_numpy(unit["patch_scores"][index]), (row["height"], row["width"]))
                scored[(method, example["image_id"])] = array
                maximum = max(maximum, float(array.max()))
        fig, axes = plt.subplots(3, 6, figsize=(17, 9), squeeze=False)
        for line, example in enumerate(examples):
            row = lookup[example["image_id"]]
            with Image.open(DATA_ROOT / row["relative_path"]) as image:
                axes[line, 0].imshow(image.convert("RGB"))
            axes[line, 0].set_title(example["reason"])
            axes[line, 1].imshow(native_mask(DATA_ROOT, row), cmap="gray", vmin=0, vmax=1)
            axes[line, 1].set_title("GT")
            for col, method in enumerate(METHOD_NAMES.values(), start=2):
                handle = axes[line, col].imshow(scored[(method, example["image_id"])], cmap="magma", vmin=0, vmax=maximum)
                axes[line, col].set_title(method)
            for axis in axes[line]:
                axis.axis("off")
        fig.colorbar(handle, ax=axes[:, 2:].ravel().tolist(), shrink=.6, label="Unscaled squared K/V residual")
        fig.suptitle(f"{category}: preselected development examples, seed 0, event 300")
        fig.savefig(folder / f"{category}_residual.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        scales.append({"category": category, "min": 0.0, "max": maximum, "checkpoint": 300, "seed": 0})
    write_json(folder / "numeric_scales.json", scales)


def finalize(output):
    validate_core()
    metrics_frames, distribution_frames, delta_frames, comparison_rows = [], [], [], []
    for seed in range(3):
        collect_tables(output, seed)
        metrics = pd.read_parquet(output / f"seed{seed}/metrics_by_checkpoint.parquet")
        expected = {(method, event, category) for method in (*METHOD_NAMES.values(), "COVARIANCE", "CADIC")
                    for event in CHECKPOINTS for category in CATEGORIES}
        actual = set(zip(metrics.method, metrics.checkpoint, metrics.category))
        if actual != expected or len(metrics) != len(expected):
            raise ValueError(f"incomplete or duplicate method/checkpoint grid in seed {seed}")
        numeric = metrics[metrics.status == "EVALUATED"]
        if not np.isfinite(numeric[["image_AUROC", "pixel_AUPR", "pixel_AUROC", "image_AP"]]).all().all():
            raise ValueError("non-finite metric artifacts")
        delta_frames.append(pd.DataFrame(paired_deltas(output, seed)))
        comparison_rows.extend(comparison_tables(metrics))
        metrics_frames.append(metrics)
        distribution_frames.append(pd.read_parquet(output / f"seed{seed}/score_distributions.parquet"))
    metrics, distributions, deltas = pd.concat(metrics_frames), pd.concat(distribution_frames), pd.concat(delta_frames)
    for name in ("metrics_by_checkpoint", "pixel_metrics"):
        save_table(output / f"{name}.parquet", metrics.to_dict("records"))
    for name in ("score_distributions", "defect_background"):
        save_table(output / f"{name}.parquet", distributions.to_dict("records"))
    save_table(output / "method_deltas.parquet", deltas.to_dict("records"))
    save_table(output / "method_comparisons.parquet", comparison_rows)
    images = pd.concat([pd.read_parquet(output / f"seed{s}/image_scores.parquet") for s in range(3)])
    save_table(output / "image_scores.parquet", images.to_dict("records"))
    visualizations(output)
    summary = interpret_results(metrics, distributions, deltas, pd.DataFrame(comparison_rows))
    summary.update({"status": "COMPLETED", "core_unchanged": True, "confirmation_evaluated": False,
                    "evaluated_metric_rows": int((metrics.status == "EVALUATED").sum()),
                    "development_images": 60, "order_seeds": 3,
                    "config": yaml.safe_load((output / "config_resolved.yaml").read_text())})
    write_json(output / "summary.json", summary)
    timing = {"feature_cache": read_json(output / "feature_cache_validation.json"),
              "seeds": [read_json(output / f"seed{s}/timing.json") for s in range(3)],
              "training": {}}
    for seed in range(3):
        timing["training"][f"seed{seed}"] = {}
        for method in ("COVARIANCE", "CADIC"):
            saved = torch.load(output / f"states/seed{seed}/{method}_event300.pt", weights_only=False, map_location="cpu")
            timing["training"][f"seed{seed}"][method] = {"seconds": sum(saved["update_seconds"]),
                                                         "images": 300, "device": summary["config"]["device"]}
    write_json(output / "timing.json", timing)
    write_json(output / "storage.json", {"seeds": [read_json(output / f"seed{s}/storage.json") for s in range(3)],
               "evaluator_only_feature_cache_bytes": (output / "features/development.pt").stat().st_size,
               "SMT_state_storage": "Native completed source checkpoints referenced without duplicate tensor storage"})
    write_report(output, summary, metrics, distributions, deltas, pd.DataFrame(comparison_rows))
    progress(output, "completed", **summary["completion_fields"])


def interpret_results(metrics, distributions, deltas, comparisons):
    """Keep primary metrics and residual selectivity separate from mechanics."""
    frozen = metrics[(metrics.method == "FROZEN") & (metrics.checkpoint == 0)]
    frozen_by_class = frozen.groupby("category")[["image_AUROC", "pixel_AUPR", "pixel_prevalence"]].mean()
    # Chance image ranking and pixel prevalence are declared descriptive nulls,
    # not tuned thresholds or a combined method-selection score.
    useful = (frozen_by_class.image_AUROC > .5) | (frozen_by_class.pixel_AUPR > frozen_by_class.pixel_prevalence)
    frozen_signal = "USEFUL" if useful.all() else ("WEAK" if useful.any() else "FAILED")
    immediate, contractions = [], []
    for category, event in zip(CATEGORIES, (100, 200, 300)):
        for method in ("P0_BASE", "P1_SYNC", "P2_PROJ"):
            subset = deltas[(deltas.category == category) & (deltas.checkpoint == event) & (deltas.method == method)]
            immediate.extend(subset.to_dict("records"))
            for seed in range(3):
                before = distributions[(distributions.seed == seed) & (distributions.category == category) &
                                       (distributions.checkpoint == event - 100) & (distributions.method == method)].iloc[0]
                after = distributions[(distributions.seed == seed) & (distributions.category == category) &
                                      (distributions.checkpoint == event) & (distributions.method == method)].iloc[0]
                contractions.append({"seed": seed, "category": category, "method": method,
                                     "before_checkpoint": event - 100, "after_checkpoint": event,
                                     "normal_ratio": after.normal_pixel_mean / before.normal_pixel_mean,
                                     "defect_ratio": after.defect_score_mean / before.defect_score_mean,
                                     "background_ratio": after.anomaly_background_score_mean / before.anomaly_background_score_mean})
    immediate = pd.DataFrame(immediate)
    acquisition = immediate.groupby("method")[["delta_image_AUROC_vs_frozen", "delta_pixel_AUPR_vs_frozen"]].mean()
    contractions_df = pd.DataFrame(contractions)
    selective = contractions_df.groupby("method")[["normal_ratio", "defect_ratio"]].mean()
    eligible = []
    for method in acquisition.index:
        rows = immediate[immediate.method == method]
        by_category = rows.groupby("category")[["delta_image_AUROC_vs_frozen", "delta_pixel_AUPR_vs_frozen"]].mean()
        positive_categories = ((by_category.delta_image_AUROC_vs_frozen > 0) |
                               (by_category.delta_pixel_AUPR_vs_frozen > 0)).all()
        paired_positive = ((rows.delta_image_AUROC_ci_low > 0) | (rows.delta_pixel_AUPR_ci_low > 0)).any()
        no_systematic_loss = (by_category.delta_image_AUROC_vs_frozen.mean() >= 0 and
                              by_category.delta_pixel_AUPR_vs_frozen.mean() >= 0)
        residual_selective = selective.loc[method, "normal_ratio"] < selective.loc[method, "defect_ratio"]
        if positive_categories and paired_positive and no_systematic_loss and residual_selective:
            eligible.append(method)
    if acquisition.lt(0).all().all():
        effect = "NEGATIVE"
    elif acquisition.ge(0).all().all() and acquisition.gt(0).any().any():
        effect = "POSITIVE"
    elif acquisition.eq(0).all().all():
        effect = "NEUTRAL"
    else:
        effect = "MIXED"
    # A method must be non-dominated in the two primary metrics.  Efficiency
    # chooses between empirically equal P0/P1, never an arbitrary fused score.
    nondominated = [m for m in eligible if not any(
        (acquisition.loc[n] >= acquisition.loc[m]).all() and (acquisition.loc[n] > acquisition.loc[m]).any()
        for n in eligible if n != m)]
    selected = nondominated[0] if len(nondominated) == 1 else ("P1_SYNC" if "P1_SYNC" in nondominated else "NONE")
    p01 = comparisons[comparisons.comparison == "P1_SYNC_minus_P0_BASE"]
    equal_p01 = bool(p01.image_AUROC_difference.eq(0).all() and p01.pixel_AUPR_difference.abs().max() < 0.0002)
    p2_delta = acquisition.loc["P2_PROJ"] - acquisition.loc["P1_SYNC"]
    p2_relation = "POSITIVE" if (p2_delta >= 0).all() and (p2_delta > 0).any() else (
        "NEGATIVE" if (p2_delta <= 0).all() and (p2_delta < 0).any() else
        ("NEUTRAL" if (p2_delta == 0).all() else "INCONCLUSIVE"))
    contextual = metrics[(metrics.method.isin(["COVARIANCE", "CADIC"])) & (metrics.checkpoint == 300)].groupby("method")[["image_AUROC", "pixel_AUPR"]].mean()
    mutable_final = metrics[(metrics.method.isin(["P0_BASE", "P1_SYNC", "P2_PROJ"])) & (metrics.checkpoint == 300)].groupby("method")[["image_AUROC", "pixel_AUPR"]].mean()
    covariance_differences = mutable_final.rsub(contextual.loc["COVARIANCE"])
    covariance_relation = "BETTER" if covariance_differences.ge(0).all().all() else (
        "WORSE" if covariance_differences.le(0).all().all() else "MIXED")
    ready = selected != "NONE" and frozen_signal != "FAILED"
    if covariance_relation == "BETTER":
        ready = False
    blocker = "NONE" if ready else (
        "Pooled fixed-query covariance dominates the mutable residual; HOPE complexity lacks demonstrated value"
        if covariance_relation == "BETTER" else
        "Normal self-target adaptation has not shown reproducible selective anomaly separation across categories")
    fields = {"ANOMALY_SIGNAL_GATE_COMPLETE": "YES", "FROZEN_RESIDUAL_SIGNAL": frozen_signal,
              "ONLINE_MEMORY_EFFECT": effect, "SELECTED_MEMORY": selected,
              "P1_OVER_P0": "ENGINEERING_ONLY" if equal_p01 else "INCONCLUSIVE", "P2_OVER_P1": p2_relation,
              "COVARIANCE_CONTROL": covariance_relation, "READY_FOR_ANOMALY_METHOD_RESEARCH": "YES" if ready else "NO",
              "PRIMARY_ANOMALY_BLOCKER": blocker}
    return {"completion_fields": fields, "immediate_deltas": acquisition.reset_index().to_dict("records"),
            "contractions": contractions, "frozen_by_category": frozen_by_class.reset_index().to_dict("records"),
            "eligible_methods": eligible, "contextual_final": contextual.reset_index().to_dict("records")}


def markdown_table(frame, digits=5):
    if frame.empty:
        return "No rows."
    frame = frame.copy()
    for column in frame:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "—")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in frame.fillna("—").itertuples(index=False, name=None))
    return "\n".join(lines)


def write_report(output, summary, metrics, distributions, deltas, comparisons):
    fields = summary["completion_fields"]
    group = ["method", "checkpoint", "category"]
    table = metrics[metrics.status == "EVALUATED"].groupby(group).mean(numeric_only=True).reset_index()
    delta = deltas.groupby(group).mean(numeric_only=True).reset_index()
    dist = distributions.groupby(group).mean(numeric_only=True).reset_index()
    table = table.merge(delta, on=group).merge(dist, on=group)
    columns = group + ["image_AUROC", "pixel_AUPR", "delta_image_AUROC_vs_frozen", "delta_pixel_AUPR_vs_frozen",
                      "normal_score_median", "anomaly_score_median", "defect_background_ratio",
                      "persistent_bytes", "update_ms_per_image"]
    p01 = comparisons[comparisons.comparison == "P1_SYNC_minus_P0_BASE"].groupby(["checkpoint", "category"]).mean(numeric_only=True).reset_index()
    p12 = comparisons[comparisons.comparison == "P2_PROJ_minus_P1_SYNC"].groupby(["checkpoint", "category"]).mean(numeric_only=True).reset_index()
    residual = pd.DataFrame(summary["contractions"]).groupby(["method", "category"]).mean(numeric_only=True).reset_index()
    final_context = metrics[(metrics.checkpoint == 300) & (metrics.status == "EVALUATED")].groupby("method")[["image_AUROC", "pixel_AUPR", "persistent_bytes", "update_ms_per_image"]].mean().reset_index()
    frozen = pd.DataFrame(summary["frozen_by_category"])
    text = "# HOPE Anomaly Signal Gate\n\n"
    text += ("This is a small DEVELOPMENT study. Bottle, carpet and hazelnut each use ten normal and ten "
             "anomalous test images; anomaly images were selected round-robin across defect folders before "
             "scoring. A second disjoint 60-image CONFIRMATION manifest is fixed but unevaluated. These "
             "development images are no longer untouched test evidence. No new loss/head or production "
             "SMT/CMS/HOPE change was made.\n\n")
    text += "## Scoring and evidence contracts\n\n"
    text += ("All SMT methods use the exact current-snapshot squared residual `||M_memory k-v||²`, "
             "with normalized current keys and unnormalized current values. The self-generated "
             "training target `M_memory^- v` is not substituted for `v` in scoring. Q-space "
             "`M_memory q` RMS is logged separately. Image scoring takes the maximum patch residual; "
             "the 28×28 patch map is bilinearly interpolated to the native GT resolution with "
             "`align_corners=False`, with no smoothing or normalization. Evaluations are strictly "
             "read-only and fingerprint checked. CMS is disabled.\n\n"
             "P0/P1/P2 states at events 100/200/300 are reused from the completed normal-only memory "
             "gate. All methods share the seed-0 initialization and three paired within-task orders. "
             "Covariance uses fixed normalized q, float64 pooled moments and the specified 0.9/0.1 "
             "shrinkage plus 1e-6I. CADIC is the existing ordinary compatible route, budget 2500, "
             "native Euclidean/support-neighbor score and native eight-image update batches; it is "
             "contextual evidence, not a clock ablation. Both statistical controls are NOT_FITTED "
             "at event 0. Source/config/checkpoint hashes are in config_resolved.yaml.\n\n"
             "Pixel AUPR means sklearn average precision (uninterpolated step integral). Point "
             "metrics use every native-resolution pixel exactly. Intervals use 400 paired stratified "
             "IMAGE bootstrap samples, keeping all pixels of each image together. Only pixel "
             "interval thresholds use histograms; recorded agreement with exact point AP is within "
             "0.0002. No patch/pixel independence or tiny-sample significance claim is made. "
             "Tables below average the three order seeds; per-seed paired intervals remain in parquet.\n\n")
    text += "## Main category/checkpoint table\n\n" + markdown_table(table[columns]) + "\n\n"
    text += "## Frozen residual and learning deltas\n\n" + markdown_table(frozen) + "\n\n"
    text += markdown_table(pd.DataFrame(summary["immediate_deltas"])) + "\n\n"
    text += "## Normal versus defect/background residual contraction\n\n"
    text += "Ratios compare each category's immediately preceding checkpoint with its post-learning checkpoint, on the SAME development inputs. Values below one are residual contraction.\n\n"
    text += markdown_table(residual[["method", "category", "normal_ratio", "defect_ratio", "background_ratio"]]) + "\n\n"
    text += "## P0–P1 and P1–P2\n\n" + markdown_table(p01.drop(columns="seed")) + "\n\n" + markdown_table(p12.drop(columns="seed")) + "\n\n"
    text += "P0/P1 score-distribution differences and all image-level raw scores are retained; nearly identical ranking is an engineering comparison, not an accuracy breakthrough. P2's Q-space retention result was not used to predict K/V anomaly quality.\n\n"
    text += "## Retention and contextual controls\n\n"
    text += "The main table retains all checkpoints for bottle and carpet, so later-task effects are visible independently of immediate acquisition. There is no test-time write.\n\n"
    text += markdown_table(final_context) + "\n\n"
    text += "## Cost and visualization\n\n"
    text += ("The SMT persistent payload counts static and mutable tensors without CMS. Update "
             "latencies are the measured original normal-memory-gate timings; device provenance "
             "is mixed (recorded in its artifacts), so they are not a common-device speed benchmark. "
             "Covariance/CADIC timings are measured in this run, including fixed-basis construction "
             "for covariance. Covariance storage includes its full Cholesky cache. Full timings/storage "
             "are in timing.json/storage.json. Native-map visualizations use seed 0 event 300 with one "
             "shared raw numeric scale per category and examples fixed by GT extent before scores. "
             "Largest extent is not claimed to be easiest for the detector.\n\n")
    text += "## Scientific decision\n\n"
    text += f"Frozen residual: **{fields['FROZEN_RESIDUAL_SIGNAL']}**. Online effect: **{fields['ONLINE_MEMORY_EFFECT']}**. Selected memory: **{fields['SELECTED_MEMORY']}**.\n\n"
    text += ("The decision uses both primary metrics, paired uncertainty, normal-versus-defect "
             "selectivity and later-checkpoint retention. No anomaly-performance threshold, "
             "score fusion or loss tuning was introduced. Covariance dominance is a reason to "
             "question added self-modifying complexity, not to silently replace HOPE. A descriptive "
             "above-chance point estimate alone does not establish a robust anomaly detector.\n\n")
    text += f"Primary blocker: {fields['PRIMARY_ANOMALY_BLOCKER']}.\n\n"
    text += ("If ready, the next step is a single selected formulation with the same locked residual "
             "on a full three-category development benchmark. If not ready, inspect read/write "
             "and self-target semantics or outer initialization before inventing a loss. "
             "The confirmation manifest remains sealed in either case.\n\n")
    text += "```text\n" + "\n".join(f"{name} = {value}" for name, value in fields.items()) + "\nCANONICAL_CORE_UNCHANGED = YES\n```\n"
    (ROOT / "agents/reports/hope_anomaly_signal_gate.md").write_text(text)
    task = "# Current Task — HOPE Anomaly Signal Gate\n\n"
    task += "Completed read-only K/V-residual development evaluation using the completed normal-learning checkpoints and three paired order seeds. No new loss/head, confirmation metrics or production changes.\n\n"
    task += f"Report: `agents/reports/hope_anomaly_signal_gate.md`. Artifacts: `{output.relative_to(ROOT)}`.\n\n"
    task += "```text\n" + "\n".join(f"{name} = {value}" for name, value in fields.items()) + "\nCANONICAL_CORE_UNCHANGED = YES\n```\n"
    (ROOT / "agents/tasks/current_task_HOPE_AnomalySignal.md").write_text(task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "prepare", "residual", "covariance", "cadic", "finalize"], default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    progress(args.output, "started", stage=args.stage, device=str(device))
    validate_core()
    try:
        if args.stage in ("all", "prepare"):
            prepare(args.output, device)
        if args.stage in ("all", "residual"):
            residual_phase(args.output, device)
        if args.stage in ("all", "covariance"):
            covariance_phase(args.output, device)
        if args.stage in ("all", "cadic"):
            cadic_phase(args.output, device)
        if args.stage in ("all", "finalize"):
            finalize(args.output)
    except BaseException as exc:
        write_json(args.output / "failure.json", {"status": "FAILED", "error": repr(exc),
                   "progress": read_json(args.output / "progress.json"), "core_unchanged": validate_core() == CORE_HASHES})
        raise


if __name__ == "__main__":
    main()
