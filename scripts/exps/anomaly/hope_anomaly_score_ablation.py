# scripts/exps/hope_anomaly_score_ablation.py
"""Verify a completed residual study before read-only score ablations."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from exps.anomaly.hope_anomaly_signal import CATEGORIES, CHECKPOINTS, METHOD_NAMES
from exps.anomaly.hope_anomaly_score_ablation import (
    COSINE_EPS, affinity_scores, angular_scores, assert_files_unchanged, file_manifest,
    original_process_status, verify_original, verify_unit_arrays,
)
from scripts.exps.anomaly import hope_anomaly_signal as original


DEFAULT_SOURCE = original.DEFAULT_OUTPUT
DEFAULT_OUTPUT = original.ROOT / "results/hope_cad/anomaly_score_ablation"


def prepare_ablation(source, output, score, verification, workers=1):
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((source / "config_resolved.yaml").read_text())
    formula = ("1 - cosine_similarity(M_content @ k_i, v_i, eps=1e-8)" if score == "ANGULAR" else
               "mean_j [cos(x_i,x_j)-cos(M_content q_i,M_content q_j)]^2; valid 8-connected neighbors")
    config.update({"source_results": str(source), "score": score, "COSINE_EPS": COSINE_EPS,
                   "source_raw_scoring": config["scoring"], "scoring": formula,
                   "score_evaluation_workers": workers,
                   "memory_updates": "NONE; reuse exact original completed states",
                   "fresh_reference": "original p0_initial_fixture.pt, never an evolving state",
                   "pixel_map": "original 28x28 bilinear to native GT resolution, align_corners=False",
                   "near_zero_rule": "norm < 1e-8; numerical diagnostic, not a learned threshold"})
    path = output / "config_resolved.yaml"
    if path.exists() and yaml.safe_load(path.read_text()) != config:
        raise ValueError("score ablation configuration differs")
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = original.table_records(source / "manifests/anomaly_dev_manifest.parquet")
    original.write_json(output / "source_dev_manifest.json", {
        "source": str(source / "manifests/anomaly_dev_manifest.parquet"),
        "identity": verification["development_manifest_identity"], "images": manifest,
        "confirmation_evaluated": False,
        "confirmation_identity": verification["sealed_manifest_identity"]})
    states = pd.read_parquet(source / "states/smt_state_manifest.parquet").to_dict("records")
    original.write_json(output / "source_state_manifest.json", {
        "states": states,
        "frozen_evaluation": "original initial fixture for every evaluation checkpoint label",
        "state_verification": verification["checkpoint_checks"]})
    original.write_json(output / "score_definition.json", {
        "score": score, "formula": formula,
        "image": "maximum patch score", "pixel": "28x28 bilinear; align_corners=False",
        "direction": "higher means more abnormal", "score_fusion": False,
        "smoothing": False, "normalization": False, "trainable_parameters": 0})


def collect_ablation(output):
    units = []
    for seed in range(3):
        own = [original.read_json(path) for path in sorted((output / f"seed{seed}/units").glob("*.json"))]
        if len(own) != 48:
            raise ValueError("score checkpoint grid is incomplete")
        units.extend(own)
        original.paired_deltas(output, seed)
        for name, rows in (("metrics_by_checkpoint", [u["metrics"] for u in own]),
                           ("pixel_metrics", [u["metrics"] for u in own]),
                           ("image_scores", [row for u in own for row in u["images"]]),
                           ("score_distributions", [u["distribution"] for u in own]),
                           ("defect_background", [u["distribution"] for u in own])):
            original.save_table(output / f"seed{seed}/{name}.parquet", rows)
    metrics = [u["metrics"] for u in units]
    images = [row for u in units for row in u["images"]]
    distributions = [u["distribution"] for u in units]
    deltas = pd.concat([pd.read_parquet(output / f"seed{s}/method_deltas.parquet") for s in range(3)])
    for name, rows in (("metrics_by_checkpoint", metrics), ("pixel_metrics", metrics),
                       ("image_scores", images), ("score_distributions", distributions),
                       ("defect_background", distributions), ("method_deltas", deltas.to_dict("records"))):
        original.save_table(output / f"{name}.parquet", rows)
    decomposition = [{k: v for k, v in row.items() if k in (
        "seed", "method", "checkpoint", "category", "image_id", "label", "snapshot_hash",
        "near_zero_prediction_fraction", "near_zero_value_fraction", "prediction_norm_median",
        "value_norm_median", "raw_component_mean", "radial_component_mean", "angular_component_mean",
        "radial_fraction", "decomposition_max_abs_error", "memory_readout_rms",
        "reset_score_relative_l2", "history_angle_mean_delta", "history_angle_max_abs_delta", "history_angle_delta_std",
        "near_zero_readout_fraction", "near_zero_feature_fraction", "query_readout_norm_median", "feature_norm_median",
        "memory_reset_relative_l2", "memory_reset_cosine", "history_affinity_relative_l2",
        "history_affinity_mean_delta", "history_affinity_max_abs_delta", "history_affinity_delta_std")}
        for row in images]
    original.save_table(output / "score_decomposition_summary.parquet", decomposition)
    temporary = output / "score_decomposition.tmp.parquet"
    writer = None
    try:
        for unit in units:
            context = unit["context"]
            stem = output / f"seed{context['seed']}/units/{context['method']}_event{context['checkpoint']}_{context['category']}"
            with np.load(Path(str(stem) + "_components.npz"), allow_pickle=False) as archive:
                ids = archive["image_ids"].tolist()
                columns = {key: archive[key].reshape(-1) for key in archive.files if key != "image_ids"}
            rows = len(ids) * 784
            columns.update({key: np.repeat(value, rows) for key, value in context.items()})
            columns.update({"image_id": np.repeat(ids, 784), "patch_index": np.tile(np.arange(784), len(ids))})
            table = pa.Table.from_pandas(pd.DataFrame(columns), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output / "score_decomposition.parquet")
    original.save_table(output / "retention_matrix.parquet", [r for r in metrics if r["checkpoint"] >=
                       (100, 200, 300)[CATEGORIES.index(r["category"])]])
    return units


def verify_ablation(output, source, score="ANGULAR"):
    source_rows = original.table_records(source / "manifests/anomaly_dev_manifest.parquet")
    metrics = pd.read_parquet(output / "metrics_by_checkpoint.parquet")
    images = pd.read_parquet(output / "image_scores.parquet")
    keys = ["seed", "method", "checkpoint", "category"]
    expected = {(s, m, e, c) for s in range(3) for m in METHOD_NAMES.values()
                for e in CHECKPOINTS for c in CATEGORIES}
    if len(metrics) != 144 or metrics.duplicated(keys).any() or set(map(tuple, metrics[keys].values)) != expected:
        raise ValueError("ablation metric identities differ")
    if len(images) != 2880 or images.duplicated(keys + ["image_id"]).any():
        raise ValueError("ablation image identities differ")
    for path in sorted(output.glob("seed*/units/*.json")):
        payload = original.read_json(path)
        category = payload["context"]["category"]
        rows = [r for r in source_rows if r["category"] == category]
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        verify_unit_arrays(payload, arrays, rows)
        raw_path = source / path.relative_to(output)
        raw_payload = original.read_json(raw_path)
        if [r["snapshot_hash"] for r in payload["images"]] != [r["snapshot_hash"] for r in raw_payload["images"]]:
            raise ValueError("raw and angular evaluations used different persistent states")
        forensic_path = path.with_name(path.stem + "_components.npz")
        with np.load(forensic_path, allow_pickle=False) as components, np.load(raw_path.with_suffix(".npz"), allow_pickle=False) as raw:
            np.testing.assert_array_equal(components["image_ids"], arrays["image_ids"])
            np.testing.assert_array_equal(components["angle" if score == "ANGULAR" else "affinity"], arrays["patch_scores"])
            if score == "ANGULAR":
                np.testing.assert_allclose(components["raw"], raw["patch_scores"], rtol=3.1e-5, atol=2e-4)
    for name in ("pixel_metrics", "score_distributions", "score_decomposition_summary", "method_deltas", "retention_matrix"):
        pd.read_parquet(output / f"{name}.parquet")
    patch_rows = pq.ParquetFile(output / "score_decomposition.parquet").metadata.num_rows
    if patch_rows != 2880 * 784:
        raise ValueError("patch decomposition coverage differs")
    checks = {"passed": True, "metric_rows": 144, "image_rows": 2880,
            "finite": bool(np.isfinite(images.image_score).all()), "decomposition_patch_rows": patch_rows,
            "confirmation_evaluated": False, "all_states_unchanged": True}
    if score == "ANGULAR":
        checks.update({"max_near_zero_prediction_fraction": float(images.near_zero_prediction_fraction.max()),
                       "max_near_zero_value_fraction": float(images.near_zero_value_fraction.max()),
                       "max_decomposition_abs_error": float(images.decomposition_max_abs_error.max())})
    else:
        checks.update({"max_near_zero_readout_fraction": float(images.near_zero_readout_fraction.max()),
                       "max_near_zero_feature_fraction": float(images.near_zero_feature_fraction.max())})
    return checks


def analyze_readout(source, output):
    """Keep readout effects and within-readout memory effects as separate tables."""
    metrics = pd.read_parquet(output / "metrics_by_checkpoint.parquet")
    old = pd.read_parquet(source / "metrics_by_checkpoint.parquet")
    keys = ["seed", "method", "checkpoint", "category"]
    comparisons = []
    for row in metrics.to_dict("records"):
        identity = {key: row[key] for key in keys}
        seed, method, event, category = (row[key] for key in keys)
        stem = f"seed{seed}/units/{method}_event{event}_{category}"
        base_stem = f"seed{seed}/units/FROZEN_event{event}_{category}"
        reference = old[(old.seed == seed) & (old.method == method) &
                        (old.checkpoint == event) & (old.category == category)].iloc[0]
        frozen_raw = old[(old.seed == seed) & (old.method == "FROZEN") &
                         (old.checkpoint == event) & (old.category == category)].iloc[0]
        frozen_new = metrics[(metrics.seed == seed) & (metrics.method == "FROZEN") &
                             (metrics.checkpoint == event) & (metrics.category == category)].iloc[0]
        with np.load(output / (stem + ".npz")) as own, np.load(source / (stem + ".npz")) as raw, \
                np.load(output / (base_stem + ".npz")) as base_new, np.load(source / (base_stem + ".npz")) as base_raw:
            np.testing.assert_array_equal(own["image_ids"], raw["image_ids"])
            for metric in ("image_AUROC", "pixel_AUPR"):
                bootstrap = "bootstrap_" + metric
                for effect, point, samples in (
                    ("READOUT_AT_FIXED_STATE", row[metric] - reference[metric], own[bootstrap] - raw[bootstrap]),
                    ("MEMORY_RAW", reference[metric] - frozen_raw[metric], raw[bootstrap] - base_raw[bootstrap]),
                    ("MEMORY_NEW_SCORE", row[metric] - frozen_new[metric], own[bootstrap] - base_new[bootstrap])):
                    low, high = np.quantile(samples, [.025, .975])
                    comparisons.append({**identity, "metric": metric, "effect": effect,
                                        "delta": point, "paired_ci_low": low, "paired_ci_high": high})
    original.save_table(output / "readout_comparison.parquet", comparisons)
    method_pairs = original.comparison_tables(metrics)
    for row in method_pairs:
        left, right = row["comparison"].split("_minus_")
        stem = output / f"seed{row['seed']}/units"
        with np.load(stem / f"{left}_event{row['checkpoint']}_{row['category']}.npz") as a, \
                np.load(stem / f"{right}_event{row['checkpoint']}_{row['category']}.npz") as b:
            for metric in ("image_AUROC", "pixel_AUPR"):
                low, high = np.quantile(a["bootstrap_" + metric] - b["bootstrap_" + metric], [.025, .975])
                row[metric + "_paired_ci_low"], row[metric + "_paired_ci_high"] = low, high
    original.save_table(output / "method_comparisons.parquet", method_pairs)
    distributions = pd.read_parquet(output / "score_distributions.parquet")
    contractions = []
    for seed in range(3):
        for method in ("P0_BASE", "P1_SYNC", "P2_PROJ"):
            for category, event in zip(CATEGORIES, (100, 200, 300)):
                selected = distributions[(distributions.seed == seed) & (distributions.method == method) &
                                         (distributions.category == category)]
                before, after = (selected[selected.checkpoint == e].iloc[0] for e in (event - 100, event))
                row = {"seed": seed, "method": method, "category": category,
                       "before_checkpoint": event - 100, "after_checkpoint": event}
                for name in ("normal_pixel_mean", "anomaly_score_mean", "defect_score_mean", "anomaly_background_score_mean"):
                    row[name + "_before"], row[name + "_after"] = before[name], after[name]
                    row[name + "_ratio"] = after[name] / before[name]
                contractions.append(row)
    original.save_table(output / "acquisition_selectivity.parquet", contractions)
    deltas = pd.read_parquet(output / "method_deltas.parquet")
    immediate = deltas[deltas.apply(lambda r: r.checkpoint == (100, 200, 300)[CATEGORIES.index(r.category)], axis=1)]
    original.save_table(output / "order_seed_summary.parquet", immediate.groupby(["seed", "method"])[
        ["delta_image_AUROC_vs_frozen", "delta_pixel_AUPR_vs_frozen"]].mean().reset_index().to_dict("records"))
    units = [original.read_json(path) for path in output.glob("seed*/units/*.json")]
    original.write_json(output / "timing.json", {
        "scoring_seconds": sum(u["scoring_seconds"] for u in units),
        "metrics_seconds": sum(u["metrics_seconds"] for u in units),
        "memory_training_seconds": 0, "reused_original_states": True})
    original.write_json(output / "storage.json", {
        "score_parameter_bytes": 0, "new_memory_state_bytes": 0,
        "source_full_state_bytes": sorted(set(int(r["persistent_bytes"]) for r in metrics.to_dict("records"))),
        "diagnostic_patch_table_bytes": (output / "score_decomposition.parquet").stat().st_size,
        "additional_raw_feature_cache_bytes": 0})
    original.write_json(output / "score_analysis.json", {
        "final_by_category": metrics[metrics.checkpoint == 300].groupby(["method", "category"])[
            ["image_AUROC", "pixel_AUPR", "image_AP", "pixel_AUROC"]].mean().reset_index().to_dict("records"),
        "immediate_by_category": immediate.groupby(["method", "category"])[
            ["delta_image_AUROC_vs_frozen", "delta_pixel_AUPR_vs_frozen"]].mean().reset_index().to_dict("records"),
        "effect_interpretation": "readout effect at fixed state is distinct from learned-minus-FROZEN within one score"})


def score_visualizations(source, output, score="ANGULAR"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from exps.anomaly.hope_anomaly_signal import native_mask, pixel_map
    selection = original.read_json(source / "manifests/visualization_selection.json")
    lookup = {r["image_id"]: r for r in original.table_records(source / "manifests/anomaly_dev_manifest.parquet")}
    folder = output / "visualizations"
    folder.mkdir(exist_ok=True)
    for category in CATEGORIES:
        examples = [e for e in selection if e["category"] == category]
        fig, axes = plt.subplots(3, 6, figsize=(17, 9), squeeze=False)
        for line, example in enumerate(examples):
            row = lookup[example["image_id"]]
            with Image.open(original.DATA_ROOT / row["relative_path"]) as image:
                axes[line, 0].imshow(image.convert("RGB"))
            axes[line, 0].set_title(example["reason"])
            axes[line, 1].imshow(native_mask(original.DATA_ROOT, row), cmap="gray", vmin=0, vmax=1)
            axes[line, 1].set_title("GT")
            for col, method in enumerate(METHOD_NAMES.values(), 2):
                with np.load(output / f"seed0/units/{method}_event300_{category}.npz") as unit:
                    index = unit["image_ids"].tolist().index(row["image_id"])
                    values = pixel_map(torch.from_numpy(unit["patch_scores"][index]), (row["height"], row["width"]))
                handle = axes[line, col].imshow(values, cmap="magma", vmin=0, vmax=2 if score == "ANGULAR" else 4)
                axes[line, col].set_title(method)
            for axis in axes[line]:
                axis.axis("off")
        fig.colorbar(handle, ax=axes[:, 2:].ravel().tolist(), shrink=.6,
                     label="1 - cosine; fixed [0,2] display scale" if score == "ANGULAR" else "Squared affinity discrepancy; fixed [0,4] display scale")
        fig.suptitle(f"{category}: original preselected development examples, seed 0, event 300")
        fig.savefig(folder / f"{category}_{score.lower()}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
    original.write_json(folder / "selection.json", selection)


def score_seed(source, output, device, score, seed):
    rows, patches = original.evaluation_data(source)
    fixture = original.load_fixture()
    initial = original.smt_model("FROZEN", 0, 0, fixture, device)
    score_function = angular_scores if score == "ANGULAR" else affinity_scores
    for seed in (seed,):
        for raw, method in METHOD_NAMES.items():
            for event in CHECKPOINTS:
                model = original.smt_model(raw, seed, event, fixture, device)
                before = model.state_fingerprint()
                stats = model.memory_stats()
                storage = {"persistent_bytes": stats["full_tensor_bytes"], "CMS_bytes": 0,
                           "score_parameters_bytes": 0, "mutable_bytes": stats["mutable_bytes"]}
                for category in CATEGORIES:
                    frozen = output / f"seed{seed}/units/FROZEN_event0_{category}"
                    stem = output / f"seed{seed}/units/{method}_event{event}_{category}"
                    forensic_path = Path(str(stem) + "_components.npz")
                    if frozen.with_suffix(".json").exists() and (raw == "FROZEN" or event == 0):
                        original.copy_evaluation(output, frozen, seed, method, event, category, storage, 0.0)
                        if not forensic_path.exists():
                            forensic_path.write_bytes(Path(str(frozen) + "_components.npz").read_bytes())
                    else:
                        captured = []
                        def scorer(image):
                            result = score_function(model, image.to(device), initial_model=initial, capture_components=True)
                            captured.append(result.pop("patch_components"))
                            return result
                        original.evaluate_unit(output, seed, method, event, category, rows, patches,
                                               scorer, storage, 0.0)
                        if not forensic_path.exists():
                            selected = [i for i, row in enumerate(rows) if row["category"] == category]
                            if not captured:
                                # An interrupted process may already have durable metrics.
                                # Recover only the detached decomposition, never replay writes.
                                for i in selected:
                                    scorer(patches[i:i + 1])
                            np.savez_compressed(forensic_path,
                                image_ids=np.array([rows[i]["image_id"] for i in selected]),
                                **{key: np.stack([part[key] for part in captured]) for key in captured[0]})
                if model.state_fingerprint() != before or model.memory_stats()["persistent_graph"]:
                    raise ValueError("score-only evaluation changed source state")
                del model
    return seed


def _score_worker(job):
    source, output, device_name, score, seed = job
    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    prior = original.progress
    original.progress = lambda target, phase, **fields: prior(target / f"seed{seed}", phase, **fields)
    try:
        return score_seed(source, output, device, score, seed)
    finally:
        original.progress = prior


def score_phase(source, output, device, verification, score, workers=1):
    prepare_ablation(source, output, score, verification, workers)
    start = time.perf_counter()
    if workers > 1:
        jobs = [(source, output, str(device), score, seed) for seed in range(3)]
        with ProcessPoolExecutor(max_workers=min(workers, 3), mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_score_worker, job) for job in jobs]
            for future in as_completed(futures):
                original.progress(output, "seed_completed", seed=future.result(), score=score)
    else:
        for seed in range(3):
            score_seed(source, output, device, score, seed)
    collect_ablation(output)
    checks = verify_ablation(output, source, score)
    analyze_readout(source, output)
    score_visualizations(source, output, score)
    original.write_json(output / "score_checks.json", checks)
    original.write_json(output / "summary.json", {"status": "COMPLETED", "score": score,
        "execution_valid": True, "seconds": time.perf_counter() - start, "checks": checks,
        "source_execution_valid": True, "confirmation_evaluated": False,
        "interpretation": "pending paired causal analysis; highest development metric is not automatic selection"})
    original.progress(output, "completed", score=score, metric_rows=144, image_rows=2880)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validate", "angular", "affinity"), default="validate")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if original_process_status(original.ROOT / "logs/hope_cad/anomaly_signal_gate.pid")["alive"]:
        raise RuntimeError("original experiment is still running; no competing workload allowed")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    check_path = args.output / "raw_verification.json"
    try:
        if args.stage == "validate":
            result = verify_original(args.source, device, workers=args.workers,
                                     notify=lambda payload: print(json.dumps(payload), flush=True))
            original.write_json(check_path, result)
            print(json.dumps({"phase": "original_validated", "execution_valid": True}), flush=True)
        else:
            verification = original.read_json(check_path)
            if not verification["execution_valid"]:
                raise ValueError("original execution has not passed verification")
            assert_files_unchanged(args.source, verification["original_files"])
            if args.stage == "affinity":
                eligibility = original.read_json(args.output / "affinity_eligibility.json")
                if not eligibility["eligible"] or not all(eligibility["conditions"].values()):
                    raise ValueError("spatial fallback eligibility has not passed")
                if original.read_json(args.output / "angular/summary.json")["status"] != "COMPLETED":
                    raise ValueError("angular follow-up has not completed")
            score_phase(args.source, args.output / args.stage, device, verification,
                        "ANGULAR" if args.stage == "angular" else "AFFINITY",
                        args.workers if args.stage == "affinity" else 1)
            assert_files_unchanged(args.source, verification["original_files"])
        original.validate_core()
    except BaseException as exc:
        original.write_json(args.output / f"{args.stage}_failure.json", {
            "status": "FAILED", "stage": args.stage, "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
