"""G0 integrity verification and protocol lock for CSG-01."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from . import contract as C1
from exps.csg00 import contract as C0
from exps.csg00 import features as F0
from exps.csg00 import g0 as G0


def git_state() -> Dict[str, Any]:
    return G0.git_state()


def environment_info(probe_gpu: bool = False) -> Dict[str, Any]:
    return G0.environment_info(probe_gpu=probe_gpu)


def verify_all() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    agents_sha = C0.sha256_file(C1.AGENTS_MD)
    checks["agents_md"] = {"sha256": agents_sha, "pin": C1.AGENTS_MD_PIN,
                           "match": agents_sha == C1.AGENTS_MD_PIN}
    if agents_sha != C1.AGENTS_MD_PIN:
        raise RuntimeError("AGENTS.md audit pin mismatch")

    cp = C1.CHECKPOINT
    cp_info = {"path": str(cp.relative_to(C1.ROOT)), "bytes": cp.stat().st_size,
               "expected_bytes": C1.CHECKPOINT_BYTES,
               "sha256": C0.sha256_file(cp),
               "expected_sha256": C1.CHECKPOINT_SHA256}
    cp_info["match"] = cp_info["sha256"] == C1.CHECKPOINT_SHA256
    if not cp_info["match"]:
        raise RuntimeError("frozen checkpoint digest mismatch")
    checks["checkpoint"] = cp_info

    p0c_sha = C0.sha256_file(C1.P0C_PROTOCOL)
    checks["p0c_protocol"] = {"sha256": p0c_sha, "expected": C1.P0C_PROTOCOL_SHA256,
                              "match": p0c_sha == C1.P0C_PROTOCOL_SHA256}
    if p0c_sha != C1.P0C_PROTOCOL_SHA256:
        raise RuntimeError("P0c stream protocol digest mismatch")

    stream = C0.load_stream()
    checks["stream"] = {
        "writer_events": len(stream["updates"]),
        "witnesses": len(stream["probes"]),
        "schedule": stream["schedule"],
        "stream_digest": stream["stream_digest"],
        "witness_digest": stream["witness_digest"],
        "distinct_writer_paths": len(set(stream["writer_paths"])),
        "distinct_writer_contents": len(set(r["raw_png_sha256"] for r in stream["updates"])),
        "writer_witness_path_disjoint": True,
        "writer_witness_content_disjoint": True,
    }

    guard = G0._forbidden_guard_selftest(set(stream["writer_paths"]) | set(stream["probe_paths"]))
    checks["normal_only_guard"] = guard
    if not guard["passed"]:
        raise RuntimeError("normal-only guard self-test failed")

    cache = F0.load_feature_cache(verify_hashes=True)
    checks["feature_cache"] = {
        category: {k: v for k, v in info.items() if k in ("path", "bytes", "sha256", "rows")}
        for category, info in cache.files.items()
    }
    checks["cache_rows"] = F0.verify_cache_rows(cache, stream["updates"], stream["probes"])
    checks["feature_geometry"] = {
        "patch_grid": [C1.GRID, C1.GRID],
        "dim": C1.DIM,
        "dtype": "float32",
        "capture": "block 8 output / layer 9 / CLS removed / no normalization / no projection",
        "cache_metadata": {cat: info.get("metadata", {}) for cat, info in cache.files.items()},
    }

    git = git_state()
    checks["git"] = git
    if git["branch"] != "dikhang_space":
        raise RuntimeError(f"unexpected branch: {git['branch']}")

    checks["environment"] = environment_info(probe_gpu=False)
    checks["source_hashes"] = C1.source_hashes()
    checks["historical_hashes"] = C1.historical_hashes()

    byte_plan = C1.choose_size_matched_count()
    checks["byte_budget_dry_run"] = byte_plan
    if byte_plan["chosen_count"] != C1.SIZE_MATCHED_COUNT:
        raise RuntimeError("byte-matched E count differs from the locked constant")
    if abs(byte_plan["ratio"] - 1.0) > 0.05:
        raise RuntimeError("C/E dry-run byte ratio outside ±5%")
    checks["quota_vectors"] = {
        "B": C1.quota_vector(C1.SELECTION_COUNT).tolist(),
        "E": C1.quota_vector(C1.SIZE_MATCHED_COUNT).tolist(),
    }
    if int(C1.quota_vector(C1.SELECTION_COUNT).sum()) != C1.SELECTION_COUNT:
        raise RuntimeError("B quota sum mismatch")
    if int(C1.quota_vector(C1.SIZE_MATCHED_COUNT).sum()) != C1.SIZE_MATCHED_COUNT:
        raise RuntimeError("E quota sum mismatch")
    return checks


def write_lock() -> Dict[str, Any]:
    if not C1.TASK_RECORD.is_file():
        raise RuntimeError(f"task record must exist before locking: {C1.TASK_RECORD}")
    C1.RUN_DIR.mkdir(parents=True, exist_ok=True)
    if C1.G0_LOCK_PATH.exists():
        raise FileExistsError(f"G0 lock already exists: {C1.G0_LOCK_PATH}")
    verification = verify_all()
    lock = {
        "schema": "csg01_g0_lock_v1",
        "run_id": C1.RUN_ID,
        "locked_utc": C0.utc_now(),
        "task_record": {"path": str(C1.TASK_RECORD.relative_to(C1.ROOT)),
                        "sha256": C0.sha256_file(C1.TASK_RECORD)},
        "agents_md": verification["agents_md"],
        "git": verification["git"],
        "checkpoint": verification["checkpoint"],
        "feature_cache": verification["feature_cache"],
        "cache_rows": verification["cache_rows"],
        "feature_geometry": verification["feature_geometry"],
        "p0c_protocol": verification["p0c_protocol"],
        "stream": verification["stream"],
        "normal_only_guard": verification["normal_only_guard"],
        "environment": verification["environment"],
        "byte_budget_dry_run": verification["byte_budget_dry_run"],
        "quota_vectors": verification["quota_vectors"],
        "source_hashes": verification["source_hashes"],
        "historical_hashes": verification["historical_hashes"],
        "protocol": C1.protocol_manifest(),
        "no_anomaly_attestation": (
            "G0 read only AGENTS.md, task/report metadata, the P0c normal stream "
            "protocol, normal train/good identities and the frozen normal feature "
            "cache. No MVTec test image, ground-truth mask, anomaly label, consumed "
            "DEV anomaly or sealed confirmation payload was opened, hashed or decoded."
        ),
    }
    with open(C1.G0_LOCK_PATH, "x", encoding="utf-8") as fh:
        json.dump(lock, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return {"lock_path": str(C1.G0_LOCK_PATH),
            "sha256": C0.sha256_file(C1.G0_LOCK_PATH),
            "bytes": C1.G0_LOCK_PATH.stat().st_size}


def verify_lock() -> Dict[str, Any]:
    if not C1.G0_LOCK_PATH.is_file():
        raise RuntimeError(f"missing G0 lock: {C1.G0_LOCK_PATH}")
    lock = json.loads(C1.G0_LOCK_PATH.read_text())
    problems = []
    if lock["schema"] != "csg01_g0_lock_v1" or lock["run_id"] != C1.RUN_ID:
        problems.append("schema/run_id")
    current = C1.source_hashes()
    if current != lock["source_hashes"]:
        changed = sorted(set(current) | set(lock["source_hashes"]))
        problems.append("source_hashes: " + ", ".join(
            k for k in changed if current.get(k) != lock["source_hashes"].get(k)))
    current_h = C1.historical_hashes()
    if current_h != lock["historical_hashes"]:
        changed = sorted(set(current_h) | set(lock["historical_hashes"]))
        problems.append("historical_hashes: " + ", ".join(
            k for k in changed if current_h.get(k) != lock["historical_hashes"].get(k)))
    if C0.sha256_file(C1.TASK_RECORD) != lock["task_record"]["sha256"]:
        problems.append("task_record")
    if C0.sha256_file(C1.CHECKPOINT) != lock["checkpoint"]["sha256"]:
        problems.append("checkpoint")
    if problems:
        raise RuntimeError("G0 lock re-verification failed: " + "; ".join(problems))
    return {"lock_sha256": C0.sha256_file(C1.G0_LOCK_PATH),
            "source_hash_count": len(current),
            "historical_hash_count": len(current_h),
            "verified": True}
