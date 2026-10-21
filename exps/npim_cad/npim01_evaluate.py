# exps/npim_cad/npim01_evaluate.py
"""EXP-NPIM-01: sealed development evaluation of the NPIM-CAD predictive memories.

Two deliberately separate subcommands:

* ``predict`` (E1): verifies the frozen contract, encodes each of the 60 admitted
  DEV images at most once to obtain pre-attention ``Z`` and block-8 ``Y``,
  scores only categories admitted at each boundary, checks read-only state/RNG,
  then atomically seals ``predictions.pt`` + ``predictions_seal.json``.
  **It never reads a numeric mask.**
* ``metrics`` (E2): refuses unless the E1 seal exists and its SHA-256 matches,
  then opens native masks for the same 60 identities and computes native Pixel
  AP, Image AUROC, paired bootstrap, retention and diagnostics.

Both subcommands are default-deny and require the explicit
``--allow-consumed-dev`` flag.  No category ID is passed to the scorer; no model
is updated; no retraining, C4, coreset or retrieval method is used.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exps.npim_cad import npim01_metrics as metrics_lib  # noqa: E402
from exps.npim_cad.npim01_contract import (  # noqa: E402
    ARMS,
    BOUNDARIES,
    CATEGORIES,
    CHECKPOINT,
    CP_SHA,
    MVTEC,
    ContractError,
    admitted_categories,
    canonical_digest,
    load_contract,
    load_state_detectors,
    sha256_file,
    source_hashes,
)
from models.hope_cad.npim_context import NPIMContextConfig, NPIMFeatureExtractor  # noqa: E402
from models.hope_cad.npim_state import rng_fingerprint  # noqa: E402

RESULTS = ROOT / "results/hope_cad/npim_01"
PREDICTION_SCHEMA = "npim01_predictions_v1"
METRICS_SCHEMA = "npim01_metrics_v1"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(text)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def environment(device: str) -> Dict[str, Any]:
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    try:
        import timm
        info["timm"] = timm.__version__
    except Exception:  # pragma: no cover
        info["timm"] = None
    return info


def require_authorization(args: argparse.Namespace) -> bool:
    if not getattr(args, "allow_consumed_dev", False):
        print(json.dumps({
            "decision": "BLOCKED_CONSUMED_DEV_NOT_AUTHORIZED",
            "reason": "explicit --allow-consumed-dev is required; no DEV image or mask was opened",
            "opened_anomaly_data": False,
        }, indent=2))
        return False
    return True


def validate_run_id(run_id: str) -> None:
    if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in run_id):
        raise ValueError("run_id must be a safe identifier")


def _state_hash_map(detectors: Mapping[tuple[str, int], Any]) -> Dict[str, str]:
    return {f"{arm}@{boundary}": detector.state_hash() for (arm, boundary), detector in detectors.items()}


def _state_file_map() -> Dict[str, str]:
    from exps.npim_cad.npim01_contract import STATE_SHA256

    return {f"{arm}@{boundary}": digest for (arm, boundary), digest in STATE_SHA256.items()}


# ---------------------------------------------------------------------------
# E1: prediction-only
# ---------------------------------------------------------------------------


def run_predict(args: argparse.Namespace) -> int:
    if not require_authorization(args):
        return 2
    validate_run_id(args.run_id)
    run_dir = RESULTS / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing NPIM-01 run directory: {run_dir}")

    started = time.perf_counter()
    contract = load_contract()
    if not contract.source_sha256:
        raise ContractError("no pinned source hashes")
    run_dir.mkdir(parents=True, exist_ok=False)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    detectors = load_state_detectors(device=args.device)
    extractor = NPIMFeatureExtractor(
        NPIMContextConfig(checkpoint_path=str(CHECKPOINT), checkpoint_sha256=CP_SHA),
        device=args.device,
    )

    state_before = _state_hash_map(detectors)
    rng_before = rng_fingerprint()
    entries: List[Dict[str, Any]] = []
    encode_seconds: List[float] = []
    key_seconds: List[float] = []
    score_seconds: List[float] = []
    per_image_seconds: List[Dict[str, Any]] = []

    for record in contract.records:
        image = metrics_lib.load_raw_image(MVTEC, record)
        t0 = time.perf_counter()
        z, y = extractor.encode(image)
        encode_seconds.append(time.perf_counter() - t0)
        z0, y0 = z[0], y[0]
        keys: Dict[str, torch.Tensor] = {}
        for arm in ARMS:
            t_key = time.perf_counter()
            keys[arm] = detectors[(arm, BOUNDARIES[0])].build_keys(z0)
            key_seconds.append(time.perf_counter() - t_key)
        image_scores: Dict[str, float] = {}
        for boundary in BOUNDARIES:
            if record.category not in admitted_categories(boundary):
                continue
            for arm in ARMS:
                t_score = time.perf_counter()
                output = detectors[(arm, boundary)].score_keys(keys[arm], y0)
                score_seconds.append(time.perf_counter() - t_score)
                patch_scores = output["patch_scores"].detach().to("cpu", copy=True).numpy().astype(np.float32)
                if patch_scores.shape != (784,) or not np.isfinite(patch_scores).all():
                    raise RuntimeError(f"invalid patch scores for {record.relative_path} {arm}@{boundary}")
                entries.append({
                    "boundary": int(boundary),
                    "arm": arm,
                    "record_index": int(record.index),
                    "patch_scores": patch_scores,
                    "image_score": float(output["image_score"]),
                })
                image_scores[f"{arm}@{boundary}"] = float(output["image_score"])
        per_image_seconds.append({
            "record_index": int(record.index),
            "category": record.category,
            "encode_seconds": encode_seconds[-1],
            "image_scores": image_scores,
        })
        if record.index % 10 == 0:
            print(f"E1 encoded/scored {record.index + 1}/{len(contract.records)}", flush=True)

    expected_pairs = sum(
        sum(1 for record in contract.records if record.category in admitted_categories(boundary))
        for boundary in BOUNDARIES
    )
    expected_entries = expected_pairs * len(ARMS)
    if expected_pairs != 180 or len(entries) != expected_entries:
        raise RuntimeError(f"expected 180 pairs / {expected_entries} entries, got {len(entries)}")
    counts = Counter(
        (entry["boundary"], entry["arm"], contract.records[entry["record_index"]].category)
        for entry in entries
    )
    for boundary in BOUNDARIES:
        for arm in ARMS:
            for category in admitted_categories(boundary):
                if counts[(boundary, arm, category)] != 20:
                    raise RuntimeError(f"unexpected score count for {boundary}/{arm}/{category}: {counts[(boundary, arm, category)]}")

    if _state_hash_map(detectors) != state_before:
        raise RuntimeError("scoring mutated one or more NPIM states")
    if rng_fingerprint() != rng_before:
        raise RuntimeError("scoring mutated the RNG state")

    patch_scores = np.stack([entry["patch_scores"] for entry in entries], axis=0).astype(np.float32)
    if not np.isfinite(patch_scores).all():
        raise RuntimeError("non-finite sealed patch scores")
    payload = {
        "schema": PREDICTION_SCHEMA,
        "contract": {
            "cohort_digest": contract.cohort_digest,
            "writer_digest": contract.writer_digest,
            "protocol_sha256": contract.protocol_sha256,
            "p1_sha256": contract.p1_sha256,
            "p0c_sha256": contract.p0c_sha256,
            "checkpoint_sha256": contract.checkpoint_sha256,
            "state_file_sha256": _state_file_map(),
            "state_hash": state_before,
            "source_sha256": dict(contract.source_sha256),
        },
        "records": [
            {
                "index": int(record.index),
                "relative_path": record.relative_path,
                "sha256": record.sha256,
                "category": record.category,
                "hw": list(record.hw),
            }
            for record in contract.records
        ],
        "boundaries": [int(value) for value in BOUNDARIES],
        "arms": list(ARMS),
        "record_index": torch.tensor([entry["record_index"] for entry in entries], dtype=torch.int64),
        "boundary": torch.tensor([entry["boundary"] for entry in entries], dtype=torch.int64),
        "arm_index": torch.tensor([list(ARMS).index(entry["arm"]) for entry in entries], dtype=torch.int64),
        "patch_scores": torch.from_numpy(patch_scores),
        "image_score": torch.tensor([entry["image_score"] for entry in entries], dtype=torch.float32),
        "created_utc": utc_now(),
    }
    predictions_path = run_dir / "predictions.pt"
    temporary = predictions_path.with_suffix(".pt.part")
    torch.save(payload, temporary)
    os.replace(temporary, predictions_path)
    prediction_sha = sha256_file(predictions_path)

    resource_info = {
        "elapsed_seconds": time.perf_counter() - started,
        "encode_seconds_total": float(sum(encode_seconds)),
        "encode_seconds_per_image": float(np.mean(encode_seconds)),
        "key_seconds_total": float(sum(key_seconds)),
        "score_seconds_total": float(sum(score_seconds)),
        "score_seconds_per_entry": float(np.mean(score_seconds)),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        resource_info["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        resource_info["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())

    seal = {
        "schema": PREDICTION_SCHEMA,
        "sha256": prediction_sha,
        "bytes": predictions_path.stat().st_size,
        "entries": len(entries),
        "pairs": expected_pairs,
        "records": len(contract.records),
        "boundaries": [int(value) for value in BOUNDARIES],
        "arms": list(ARMS),
        "cohort_digest": contract.cohort_digest,
        "writer_digest": contract.writer_digest,
        "protocol_sha256": contract.protocol_sha256,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "state_file_sha256": _state_file_map(),
        "state_hash_before": state_before,
        "state_hash_after": _state_hash_map(detectors),
        "rng_before": rng_before,
        "rng_after": rng_fingerprint(),
        "mask_read": False,
        "created_utc": utc_now(),
        "command": " ".join(sys.argv),
        "device": args.device,
        "environment": environment(args.device),
    }
    write_json_exclusive(run_dir / "predictions_seal.json", seal)
    write_text_exclusive(run_dir / "E1_SEALED", prediction_sha + "\n")
    manifest = {
        "schema": "npim01_e1_manifest_v1",
        "run_id": args.run_id,
        "stage": "E1 prediction-only",
        "created_utc": utc_now(),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "git_status_short": git("status", "--short"),
        "protocol_sha256": contract.protocol_sha256,
        "contract_digest": canonical_digest({
            "cohort": contract.cohort_digest,
            "writer": contract.writer_digest,
            "protocol": contract.protocol_sha256,
            "checkpoint": contract.checkpoint_sha256,
        }),
        "cohort_digest": contract.cohort_digest,
        "writer_digest": contract.writer_digest,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "source_sha256": dict(contract.source_sha256),
        "state_file_sha256": _state_file_map(),
        "state_hash": state_before,
        "environment": environment(args.device),
        "resource": resource_info,
        "mask_read": False,
        "command": " ".join(sys.argv),
        "seal": seal,
        "per_image": per_image_seconds,
    }
    write_json_exclusive(run_dir / "manifest.json", manifest)
    extractor.close()
    print(json.dumps({
        "decision": "E1_SEALED",
        "run_id": args.run_id,
        "entries": len(entries),
        "pairs": expected_pairs,
        "prediction_sha256": prediction_sha,
        "state_unchanged": True,
        "rng_unchanged": True,
        "mask_read": False,
        "runtime_seconds": resource_info["elapsed_seconds"],
        "run_dir": str(run_dir),
    }, indent=2))
    return 0


# ---------------------------------------------------------------------------
# E2: metrics after seal
# ---------------------------------------------------------------------------


def _load_sealed_predictions(run_dir: Path, contract) -> tuple[Dict[str, Any], Dict[str, Any]]:
    seal_path = run_dir / "predictions_seal.json"
    predictions_path = run_dir / "predictions.pt"
    if not seal_path.is_file() or not predictions_path.is_file():
        raise RuntimeError("E1 seal is missing; refusing to open masks")
    seal = json.loads(seal_path.read_text())
    if seal.get("schema") != PREDICTION_SCHEMA:
        raise RuntimeError("unexpected E1 seal schema")
    actual_sha = sha256_file(predictions_path)
    if actual_sha != seal["sha256"]:
        raise RuntimeError(f"E1 prediction seal mismatch: {actual_sha} != {seal['sha256']}")
    if seal.get("cohort_digest") != contract.cohort_digest:
        raise RuntimeError("E1 seal cohort digest mismatch")
    if seal.get("protocol_sha256") != contract.protocol_sha256:
        raise RuntimeError("E1 seal protocol SHA mismatch")
    if seal.get("checkpoint_sha256") != contract.checkpoint_sha256:
        raise RuntimeError("E1 seal checkpoint SHA mismatch")
    if seal.get("mask_read") is not False:
        raise RuntimeError("E1 seal does not assert mask_read=false")
    payload = torch.load(predictions_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != PREDICTION_SCHEMA:
        raise RuntimeError("predictions payload schema mismatch")
    if payload["contract"]["cohort_digest"] != contract.cohort_digest:
        raise RuntimeError("predictions payload cohort digest mismatch")
    for key in ("patch_scores", "record_index", "boundary", "arm_index", "image_score"):
        if key not in payload:
            raise RuntimeError(f"predictions payload missing {key}")
    if tuple(payload["patch_scores"].shape) != (540, 784):
        raise RuntimeError(f"predictions patch_scores shape is {tuple(payload['patch_scores'].shape)}")
    if not torch.isfinite(payload["patch_scores"]).all():
        raise RuntimeError("non-finite sealed patch scores")
    if payload["records"] != [
        {
            "index": int(record.index),
            "relative_path": record.relative_path,
            "sha256": record.sha256,
            "category": record.category,
            "hw": list(record.hw),
        }
        for record in contract.records
    ]:
        raise RuntimeError("sealed prediction records do not match the admitted cohort")
    return seal, payload


def run_metrics(args: argparse.Namespace) -> int:
    if not require_authorization(args):
        return 2
    validate_run_id(args.run_id)
    run_dir = RESULTS / args.run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"missing E1 run directory: {run_dir}")
    contract = load_contract()
    seal, payload = _load_sealed_predictions(run_dir, contract)

    # First mask access happens only after the seal above has been verified.
    mask_read_started = utc_now()
    masks = [metrics_lib.load_native_mask(MVTEC, record) for record in contract.records]
    if len(masks) != 60:
        raise RuntimeError("mask count mismatch")

    started = time.perf_counter()
    patch_scores = payload["patch_scores"].numpy()
    image_scores = payload["image_score"].numpy()
    record_index = payload["record_index"].numpy()
    boundary_values = payload["boundary"].numpy()
    arm_values = payload["arm_index"].numpy()
    arm_names = list(ARMS)
    index_map: Dict[tuple[int, str, int], int] = {}
    for i in range(len(patch_scores)):
        index_map[(int(boundary_values[i]), arm_names[int(arm_values[i])], int(record_index[i]))] = i

    groups: Dict[tuple[int, str, str], Dict[str, Any]] = {}
    bootstrap_image: Dict[str, Any] = {}
    bootstrap_pixel: Dict[str, Any] = {}
    qualitative: List[Dict[str, Any]] = []
    qualitative_dir = run_dir / "qualitative"
    selected_pixel = {
        (350, "bottle"), (350, "carpet"), (350, "hazelnut"),
        (100, "bottle"), (200, "carpet"), (300, "hazelnut"),
    }

    for boundary in BOUNDARIES:
        for category in admitted_categories(boundary):
            rows = [record for record in contract.records if record.category == category]
            if len(rows) != 20:
                raise RuntimeError(f"expected 20 records for {category}, got {len(rows)}")
            labels = [record.label for record in rows]
            masks_group = [masks[record.index] for record in rows]
            maps_by_arm: Dict[str, List[np.ndarray]] = {}
            scores_by_arm: Dict[str, List[float]] = {}
            for arm in ARMS:
                indices = [index_map[(boundary, arm, record.index)] for record in rows]
                maps = [
                    metrics_lib.native_scores(patch_scores[index], record.hw)
                    for index, record in zip(indices, rows)
                ]
                maps_by_arm[arm] = maps
                scores_by_arm[arm] = [float(image_scores[index]) for index in indices]
                metrics = metrics_lib.group_point_metrics(
                    maps, masks_group, labels, scores_by_arm[arm]
                )
                normal_positions = [i for i, label in enumerate(labels) if label == 0]
                anomaly_positions = [i for i, label in enumerate(labels) if label == 1]
                metrics["hardest_normal"] = {
                    "relative_path": rows[max(normal_positions, key=lambda i: scores_by_arm[arm][i])].relative_path,
                    "image_score": float(max(scores_by_arm[arm][i] for i in normal_positions)),
                }
                metrics["easiest_anomaly"] = {
                    "relative_path": rows[max(anomaly_positions, key=lambda i: scores_by_arm[arm][i])].relative_path,
                    "image_score": float(max(scores_by_arm[arm][i] for i in anomaly_positions)),
                }
                metrics["hardest_anomaly"] = {
                    "relative_path": rows[min(anomaly_positions, key=lambda i: scores_by_arm[arm][i])].relative_path,
                    "image_score": float(min(scores_by_arm[arm][i] for i in anomaly_positions)),
                }
                if metrics["pixel_ap"] is None or metrics["image_auroc"] is None:
                    metrics["degenerate"] = True
                groups[(boundary, arm, category)] = metrics

            bootstrap_image[f"{boundary}:{category}"] = metrics_lib.paired_image_auroc_bootstrap(
                labels,
                scores_by_arm,
                reps=args.bootstrap_reps,
                seed=20261021 + boundary + CATEGORIES.index(category),
            )
            if (boundary, category) in selected_pixel:
                bootstrap_pixel[f"{boundary}:{category}"] = metrics_lib.paired_pixel_ap_bootstrap(
                    {
                        arm: {
                            "maps": maps_by_arm[arm],
                            "masks": masks_group,
                            "labels": labels,
                            "point_ap": groups[(boundary, arm, category)]["pixel_ap"],
                        }
                        for arm in ARMS
                    },
                    reps=args.bootstrap_pixel_reps,
                    bins=args.bootstrap_bins,
                    seed=20261021 + boundary + CATEGORIES.index(category),
                )
            if boundary == 350:
                for arm in ARMS:
                    qualitative.extend(
                        metrics_lib.save_qualitative(
                            qualitative_dir,
                            groups,
                            rows,
                            {(350, arm, category): maps_by_arm[arm]},
                            {record.index: masks[record.index] for record in rows},
                        )
                    )
            print(
                f"E2 {boundary}/{category}: C1 AP={groups[(boundary,'C1',category)]['pixel_ap']}, "
                f"C2 AP={groups[(boundary,'C2',category)]['pixel_ap']}, "
                f"C3 AP={groups[(boundary,'C3',category)]['pixel_ap']}",
                flush=True,
            )
            del maps_by_arm

    macro = [
        {"boundary": boundary, "arm": arm, **metrics_lib.macro_over_categories(groups, boundary, arm)}
        for boundary in BOUNDARIES
        for arm in ARMS
    ]
    contrasts = metrics_lib.contrast_table(groups, boundaries=BOUNDARIES, categories=CATEGORIES)
    retention = metrics_lib.retention_table(
        groups,
        arms=list(ARMS),
        acquisition={"bottle": 100, "carpet": 200, "hazelnut": 300},
        boundaries=BOUNDARIES,
    )
    decision = metrics_lib.decide(contrasts, groups)
    if any(row.get("degenerate") for row in groups.values()):
        decision = {
            "verdict": "NPIM01_INCONCLUSIVE_DEV",
            "reason": "at least one category/boundary group was degenerate; no silently averaged result",
        }

    resource_info = {
        "metrics_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    metrics_payload = {
        "schema": METRICS_SCHEMA,
        "run_id": args.run_id,
        "created_utc": utc_now(),
        "mask_read_started_utc": mask_read_started,
        "e1_seal_sha256": seal["sha256"],
        "e1_prediction_sha256": seal["sha256"],
        "cohort_digest": contract.cohort_digest,
        "protocol_sha256": contract.protocol_sha256,
        "groups": [
            {"boundary": boundary, "arm": arm, "category": category, **row}
            for (boundary, arm, category), row in sorted(groups.items())
        ],
        "macro": macro,
        "contrasts": contrasts,
        "retention": retention,
        "decision": decision,
        "qualitative": qualitative,
        "resource": resource_info,
        "source_sha256": source_hashes(),
        "bootstrap": {
            "image_auroc": bootstrap_image,
            "pixel_ap_selected": bootstrap_pixel,
            "reps": args.bootstrap_reps,
            "pixel_reps": args.bootstrap_pixel_reps,
            "bins": args.bootstrap_bins,
        },
        "notes": [
            "native Pixel AP is non-interpolated tie-aware AP; image score is the max patch score",
            "no category ID was available to the scorer; no model update occurred",
            "the 60 identities are historical exposed P1 development images; not independent confirmation",
            "C3 slow level is not ready at boundary 100; C3 vs C2 at 100 is not a clean schedule comparison",
        ],
    }
    write_json_exclusive(run_dir / "metrics.json", metrics_payload)
    write_json_exclusive(run_dir / "E2_COMPLETE.json", {
        "schema": "npim01_e2_complete_v1",
        "created_utc": utc_now(),
        "e1_seal_sha256": seal["sha256"],
        "decision": decision["verdict"],
        "mask_read_started_utc": mask_read_started,
    })
    print(json.dumps({
        "decision": decision["verdict"],
        "macro_pixel_ap_C3_minus_C2_at_350": decision.get("macro_pixel_ap_C3_minus_C2_at_350"),
        "worst_category_pixel_ap_C3_minus_C2_at_350": decision.get("worst_category_pixel_ap_C3_minus_C2_at_350"),
        "e1_seal_sha256": seal["sha256"],
        "metrics_path": str(run_dir / "metrics.json"),
    }, indent=2))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXP-NPIM-01 sealed development evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("predict", "E1 prediction-only stage"), ("metrics", "E2 sealed metrics stage")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--run-id", required=True)
        command.add_argument("--allow-consumed-dev", action="store_true")
        command.add_argument("--device", default="cpu")
        command.add_argument("--bootstrap-reps", type=int, default=500)
        command.add_argument("--bootstrap-pixel-reps", type=int, default=200)
        command.add_argument("--bootstrap-bins", type=int, default=65536)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "predict":
        return run_predict(args)
    if args.command == "metrics":
        return run_metrics(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
