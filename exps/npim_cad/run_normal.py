# exps/npim_cad/run_normal.py
"""EXP-NPIM-00 Stage D: one bounded, normal-only writer pilot.

Consumes exactly the pre-existing resolved P0c writer order
(bottle 100 -> carpet 100 -> hazelnut 100 -> bottle 50 = 350 normal
``train/good`` images), fits C1/C2/C3 on the SAME events, saves immutable
states at 100/200/300/350, and evaluates ONLY normal witnesses (59
content-disjoint normal probe rows).

It never opens an anomaly image, label, mask, sealed payload or official-test
path.  It does not produce Image AUROC / Pixel AP; Stage E requires separate
authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.benchmark_protocol_v1 import _ImageDataset  # noqa: E402
from exps.task_p.p0_normal_support import require  # noqa: E402
from models.hope_cad.npim_cad import NPIMConfig, NPIMDetector, UncalibratedDetectorError  # noqa: E402
from models.hope_cad.npim_context import (  # noqa: E402
    NPIMContextConfig,
    NPIMFeatureExtractor,
    make_projection,
    sha256_file,
    sha256_tensor,
)
from models.hope_cad.npim_state import rng_fingerprint  # noqa: E402

CP = ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"
CP_SHA = "755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415"
P0C = ROOT / "results/hope_cad/task_p/p0c/protocol_resolved.json"
MVTEC = ROOT / "data/mvtec"
PROTOCOL = ROOT / "exps/npim_cad/protocol.md"
RESULTS = ROOT / "results/hope_cad/npim_00"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# C1 raises the key rank so its total persistent detector bytes approximately
# match C3 (projection + W + pending statistics + fused calibration).  C2 is the
# same-frequency control; C3 is the registered multi-timescale hypothesis.
ARMS: Dict[str, Dict[str, Any]] = {
    "C1": {
        "periods": (1,),
        "lambdas": (0.1,),
        "weights": (1.0,),
        "r": 145,
        "projection_seed": 20261021,
        "role": "single learned predictor, byte-matched to C3, same image events",
    },
    "C2": {
        "periods": (1, 1, 1),
        "lambdas": (0.1, 1.0, 10.0),
        "weights": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        "r": 64,
        "projection_seed": 20261021,
        "role": "three predictors, identical 1-image period, same total bytes as C3",
    },
    "C3": {
        "periods": (1, 16, 128),
        "lambdas": (0.1, 1.0, 10.0),
        "weights": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        "r": 64,
        "projection_seed": 20261021,
        "role": "three predictors at 1/16/128 image clocks (the hypothesis arm)",
    },
}


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def key_tag(arm: str) -> str:
    spec = ARMS[arm]
    return f"r{spec['r']}_s{spec['projection_seed']}"


def load_normal_row(row: Mapping[str, Any]) -> torch.Tensor:
    path = MVTEC / row["relative_path"]
    require(path.is_file(), f"missing normal image: {row['relative_path']}")
    require(sha256_file(path) == row["raw_png_sha256"], f"normal image content changed: {row['relative_path']}")
    dataset = _ImageDataset(MVTEC.resolve(), [dict(row, label=0)], 224, MEAN, STD, training=True)
    return dataset[0]["images"][None]


def validate_stream(protocol: Mapping[str, Any], max_events: int, boundaries: List[int]) -> List[Mapping[str, Any]]:
    updates = protocol["updates"]
    require(len(updates) == 350, "writer stream must contain 350 normal images")
    require(1 <= max_events <= 350, "max_events must be in 1..350")
    require(all(b > 0 and b <= max_events for b in boundaries), "boundary outside the tested stream")
    if max_events == 350:
        require(all(b in protocol["arrival_boundaries"] for b in boundaries),
                "full pilot boundaries must be the registered 100/200/300/350")
    else:
        require(all(b == max_events or b in protocol["arrival_boundaries"] for b in boundaries),
                "smoke boundary must be the tested stream end or a registered boundary")
    require(all(row["category"] in {"bottle", "carpet", "hazelnut"} for row in updates),
            "unexpected writer category")
    require(all(row["relative_path"].split("/")[1:3] == ["train", "good"] for row in updates),
            "writer stream escaped train/good")
    return updates[:max_events]


def aggregate(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"images": 0}
    image_scores = np.array([row["image_score"] for row in rows], dtype=np.float64)
    residual = np.array([row["residual_l2"] for row in rows], dtype=np.float64)
    p95 = np.array([row["patch_p95"] for row in rows], dtype=np.float64)
    p99 = np.array([row["patch_p99"] for row in rows], dtype=np.float64)
    top = sorted(rows, key=lambda row: row["image_score"], reverse=True)[:5]
    return {
        "images": len(rows),
        "image_score_mean": float(image_scores.mean()),
        "image_score_median": float(np.median(image_scores)),
        "image_score_max": float(image_scores.max()),
        "residual_l2_mean": float(residual.mean()),
        "residual_l2_max": float(residual.max()),
        "patch_p95_mean": float(p95.mean()),
        "patch_p99_mean": float(p99.mean()),
        "p99_max": float(p99.max()),
        "nonfinite_count": int(sum(1 for row in rows if not row["finite"])),
        "hardest_normal_witnesses": [
            {"relative_path": row["relative_path"], "image_score": row["image_score"]} for row in top
        ],
    }


def score_probes(
    detector: NPIMDetector,
    probes: List[Mapping[str, Any]],
    tag: str,
) -> Dict[str, Any]:
    """Read-only normal-witness telemetry; verifies state and RNG are untouched."""

    before_state = detector.state_hash()
    before_rng = rng_fingerprint()
    rows: List[Dict[str, Any]] = []
    for probe in probes:
        K = probe["keys"][tag].to(detector.device)
        Y = probe["Y"].to(detector.device)
        try:
            output = detector.score_keys(K, Y)
        except UncalibratedDetectorError:
            continue
        patch = output["patch_scores"].detach().to("cpu", copy=True)
        rows.append(
            {
                "relative_path": probe["relative_path"],
                "category": probe["category"],
                "image_score": float(output["image_score"]),
                "residual_l2": float(output["residual_l2"]),
                "patch_mean": float(patch.mean().item()),
                "patch_p95": float(torch.quantile(patch, 0.95).item()),
                "patch_p99": float(torch.quantile(patch, 0.99).item()),
                "patch_max": float(patch.max().item()),
                "finite": bool(torch.isfinite(patch).all().item()),
            }
        )
    if detector.state_hash() != before_state:
        raise RuntimeError("probe scoring mutated persistent state")
    if rng_fingerprint() != before_rng:
        raise RuntimeError("probe scoring mutated RNG state")
    by_category: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)
    return {
        "overall": aggregate(rows),
        "by_category": {category: aggregate(group) for category, group in sorted(by_category.items())},
        "rows": rows,
        "state_hash_before_after_equal": True,
    }


def drift(previous: Mapping[str, Any] | None, current: Mapping[str, Any]) -> Dict[str, Any] | None:
    if previous is None:
        return None
    prior = {row["relative_path"]: row["image_score"] for row in previous["rows"]}
    now = {row["relative_path"]: row["image_score"] for row in current["rows"]}
    common = sorted(set(prior) & set(now))
    if not common:
        return None
    deltas = np.array([abs(now[path] - prior[path]) for path in common], dtype=np.float64)
    denom = np.array([max(abs(prior[path]), 1e-12) for path in common], dtype=np.float64)
    relative = deltas / denom
    return {
        "paired_normal_witnesses": len(common),
        "mean_abs_image_score_delta": float(deltas.mean()),
        "max_abs_image_score_delta": float(deltas.max()),
        "mean_relative_image_score_delta": float(relative.mean()),
        "p95_relative_image_score_delta": float(np.quantile(relative, 0.95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP-NPIM-00 normal-only writer pilot")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-events", type=int, default=350)
    parser.add_argument("--boundaries", default="100,200,300,350")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arms", default="C1,C2,C3")
    parser.add_argument("--probe-limit", type=int, default=None)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    if not args.run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in args.run_id):
        raise ValueError("run_id must be a safe identifier")
    out = RESULTS / args.run_id
    if out.exists():
        raise FileExistsError(f"refusing to overwrite an existing run directory: {out}")
    out.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    arm_names = [name.strip() for name in args.arms.split(",") if name.strip()]
    require(all(name in ARMS for name in arm_names), f"unknown arm: {arm_names}")
    boundaries = sorted({int(value) for value in args.boundaries.split(",") if value.strip()})
    require(all(value > 0 for value in boundaries), "invalid boundaries")

    protocol = json.loads(P0C.read_text())
    updates = validate_stream(protocol, args.max_events, boundaries)
    probes = protocol["probes"]
    if args.probe_limit:
        probes = probes[: args.probe_limit]
    require(set(row["relative_path"] for row in updates).isdisjoint(row["relative_path"] for row in probes),
            "writer/probe overlap")
    require(set(row["raw_png_sha256"] for row in updates).isdisjoint(row["raw_png_sha256"] for row in probes),
            "writer/probe content overlap")

    checkpoint_sha = sha256_file(CP)
    require(checkpoint_sha == CP_SHA, "checkpoint identity mismatch")
    context_cfg = NPIMContextConfig(checkpoint_path=str(CP), checkpoint_sha256=CP_SHA)
    extractor = NPIMFeatureExtractor(context_cfg, device=args.device)

    detectors: Dict[str, NPIMDetector] = {}
    for arm in arm_names:
        spec = ARMS[arm]
        detectors[arm] = NPIMDetector(
            NPIMConfig(
                r=int(spec["r"]),
                periods=tuple(int(p) for p in spec["periods"]),
                lambdas=tuple(float(v) for v in spec["lambdas"]),
                weights=tuple(float(v) for v in spec["weights"]),
                projection_seed=int(spec["projection_seed"]),
            ),
            device=args.device,
        )

    # One fixed projection per distinct (rank, seed) tag.  C2 and C3 share it.
    tag_to_arm = {}
    projections = {}
    for arm in arm_names:
        tag = key_tag(arm)
        if tag not in projections:
            spec = ARMS[arm]
            projections[tag] = make_projection(int(spec["r"]), 768, int(spec["projection_seed"]))
            tag_to_arm[tag] = arm
        require(
            sha256_tensor(detectors[arm].projection.cpu()) == sha256_tensor(projections[tag]),
            "detector projection differs from the frozen shared projection",
        )

    manifest = {
        "experiment": "EXP-NPIM-00",
        "stage": "D normal-only writer pilot",
        "run_id": args.run_id,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_path": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "p0c_protocol_sha256": sha256_file(P0C),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": CP.stat().st_size,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short"),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": args.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_threads": torch.get_num_threads(),
        },
        "writer": {
            "images": len(updates),
            "order": "bottle 100 -> carpet 100 -> hazelnut 100 -> bottle 50",
            "category_counts": {
                category: sum(1 for row in updates if row["category"] == category)
                for category in sorted({row["category"] for row in updates})
            },
            "first_event": updates[0]["relative_path"],
            "last_event": updates[-1]["relative_path"],
            "roles": "normal train/good only; no anomaly label or mask opened",
        },
        "probes": {
            "images": len(probes),
            "source": "P0c content-disjoint normal probe rows (train/good only)",
            "categories": {
                category: sum(1 for row in probes if row["category"] == category)
                for category in sorted({row["category"] for row in probes})
            },
        },
        "arms": {
            arm: {
                "role": ARMS[arm]["role"],
                "r": int(ARMS[arm]["r"]),
                "periods": list(ARMS[arm]["periods"]),
                "lambdas": list(ARMS[arm]["lambdas"]),
                "weights": list(ARMS[arm]["weights"]),
                "projection_seed": int(ARMS[arm]["projection_seed"]),
                "projection_sha256": sha256_tensor(detectors[arm].projection.cpu()),
            }
            for arm in arm_names
        },
        "scoring": {
            "type": "l2 innovation (mean squared residual over d=768)",
            "diagonal_normalization_enabled": False,
            "mixture": "fixed weights renormalised over ready levels only",
            "no_anomaly_metrics": True,
        },
    }
    write_json_exclusive(out / "manifest.json", manifest)

    events_path = out / "events.jsonl"
    boundary_summaries: Dict[str, Dict[str, Any]] = {arm: {} for arm in arm_names}
    previous_probes: Dict[str, Mapping[str, Any] | None] = {arm: None for arm in arm_names}
    state_records: Dict[str, Dict[str, Any]] = {arm: {} for arm in arm_names}

    # Pre-compute normal witness keys/targets once; they are reused read-only.
    started = time.perf_counter()
    probe_cache: List[Dict[str, Any]] = []
    for probe in probes:
        image = load_normal_row(probe)
        z, y = extractor.encode(image)
        probe_cache.append(
            {
                "relative_path": probe["relative_path"],
                "category": probe["category"],
                "Y": y[0].detach().to("cpu", copy=True),
                "keys": {
                    tag: detectors[tag_to_arm[tag]].build_keys(z[0]).detach().to("cpu", copy=True)
                    for tag in projections
                },
            }
        )
    probe_encode_seconds = time.perf_counter() - started

    encode_seconds: List[float] = []
    arm_seconds: Dict[str, List[float]] = {arm: [] for arm in arm_names}
    with events_path.open("x") as events:
        for event, row in enumerate(updates, start=1):
            t0 = time.perf_counter()
            image = load_normal_row(row)
            z, y = extractor.encode(image)
            encode_seconds.append(time.perf_counter() - t0)

            z0 = z[0]
            for arm in arm_names:
                detector = detectors[arm]
                t_arm = time.perf_counter()
                keys = detector.build_keys(z0)
                record = detector.commit_keys(
                    keys,
                    y[0],
                    writer_identity={
                        "event": event,
                        "relative_path": row["relative_path"],
                        "raw_sha256": row["raw_png_sha256"],
                    },
                )
                arm_seconds[arm].append(time.perf_counter() - t_arm)
                events.write(
                    json.dumps(
                        {
                            "event": event,
                            "category": row["category"],
                            "relative_path": row["relative_path"],
                            "arm": arm,
                            "due_levels": record["due_levels"],
                            "ready": record["ready"],
                            "update_counts": record["update_counts"],
                            "pending_images": record["pending_images"],
                            "state_hash": record["state_hash"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            events.flush()

            if event in boundaries:
                for arm in arm_names:
                    detector = detectors[arm]
                    state_path = out / "states" / f"{arm}_{event:03d}.pt"
                    saved = detector.save(str(state_path))
                    probe_report = score_probes(detector, probe_cache, key_tag(arm))
                    # Keep only stats in the durable report; raw keys/targets stay out.
                    probe_payload = {
                        "arm": arm,
                        "boundary": event,
                        "completed_events": int(detector.completed_events),
                        "ready": [bool(level.ready) for level in detector.levels],
                        "update_counts": [int(level.update_count) for level in detector.levels],
                        "pending_images": [int(level.pending_images) for level in detector.levels],
                        "pair_counts": [int(level.pair_count) for level in detector.levels],
                        "state_hash": detector.state_hash(),
                        "state_file": saved,
                        "overall": probe_report["overall"],
                        "by_category": probe_report["by_category"],
                        "rows": probe_report["rows"],
                        "drift_vs_previous_boundary": drift(previous_probes[arm], probe_report),
                        "state_hash_before_after_equal": probe_report["state_hash_before_after_equal"],
                    }
                    write_json_exclusive(out / "probes" / f"{arm}_{event:03d}.json", probe_payload)
                    state_records[arm][event] = saved
                    boundary_summaries[arm][event] = probe_payload
                    previous_probes[arm] = probe_payload

    # -- close-out: ledger, deterministic reload, resource accounting ---------
    final_reports: Dict[str, Any] = {}
    for arm in arm_names:
        detector = detectors[arm]
        ledger = detector.ledger()
        reload_check = None
        if boundaries:
            final_state_path = out / "states" / f"{arm}_{boundaries[-1]:03d}.pt"
            if final_state_path.is_file():
                from models.hope_cad.npim_state import load_state_file

                fresh = NPIMDetector(detector.config, device=args.device)
                fresh.load_state_dict(load_state_file(final_state_path))
                reload_check = {
                    "state_hash_original": detector.state_hash(),
                    "state_hash_reloaded": fresh.state_hash(),
                    "equal": detector.state_hash() == fresh.state_hash(),
                }
        final_reports[arm] = {
            "completed_events": int(detector.completed_events),
            "ready": [bool(level.ready) for level in detector.levels],
            "update_counts": [int(level.update_count) for level in detector.levels],
            "pending_images": [int(level.pending_images) for level in detector.levels],
            "pair_counts": [int(level.pair_count) for level in detector.levels],
            "state_hash": detector.state_hash(),
            "ledger": ledger,
            "reload_check": reload_check,
        }

    elapsed = time.perf_counter() - started
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gpu = None
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        gpu = {
            "device": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }

    total_encode = float(sum(encode_seconds))
    summary = {
        "experiment": "EXP-NPIM-00",
        "stage": "D normal-only writer pilot",
        "run_id": args.run_id,
        "decision": (
            "NPIM_NORMAL_ONLY_PILOT_PASS" if len(updates) == 350 else "NPIM_NORMAL_ONLY_SMOKE_COMPLETE"
        ),
        "scope": "normal-only mechanics and telemetry; no anomaly efficacy claim",
        "writer_images": len(updates),
        "probe_images": len(probes),
        "boundaries": boundaries,
        "probe_encode_seconds": probe_encode_seconds,
        "encoder": {
            "total_seconds": total_encode,
            "seconds_per_image": total_encode / max(len(encode_seconds), 1),
            "images": len(encode_seconds),
        },
        "memory_update_seconds": {
            arm: {
                "total_seconds": float(sum(arm_seconds[arm])),
                "seconds_per_image": float(sum(arm_seconds[arm])) / max(len(arm_seconds[arm]), 1),
            }
            for arm in arm_names
        },
        "elapsed_seconds": elapsed,
        "peak_rss_kib": int(peak_rss_kib),
        "gpu": gpu,
        "arms": final_reports,
        "boundary_summaries": boundary_summaries,
        "rng_fingerprint": rng_fingerprint(),
        "notes": [
            "No category ID, label, mask or task boundary was an inference input.",
            "No anomaly image, anomaly label, mask, sealed payload or official-test path was opened.",
            "Probe scores are normal-only telemetry, not anomaly metrics.",
        ],
    }
    write_json_exclusive(out / "summary.json", summary)
    extractor.close()
    print(json.dumps(
        {
            "decision": summary["decision"],
            "run_id": args.run_id,
            "events": len(updates),
            "elapsed_seconds": elapsed,
            "encoder_seconds_per_image": total_encode / max(len(encode_seconds), 1),
            "bytes": {arm: final_reports[arm]["ledger"]["total_persistent_detector_bytes"] for arm in arm_names},
            "reload_equal": {arm: final_reports[arm]["reload_check"]["equal"] if final_reports[arm]["reload_check"] else None for arm in arm_names},
            "out": str(out),
        },
        indent=2,
    ))
    return 0


def _git(*args: str) -> str:
    import subprocess

    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
