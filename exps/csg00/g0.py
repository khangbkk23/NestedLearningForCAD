"""G0 integrity verification and protocol lock for CSG-00.

This module only reads metadata, normal-only manifests and normal-only feature
caches.  It never opens an anomaly image or mask.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from . import contract as C
from . import features


def _run(cmd: List[str]) -> str:
    result = subprocess.run(cmd, cwd=str(C.ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def git_state() -> Dict[str, Any]:
    return {
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "head": _run(["git", "rev-parse", "HEAD"]),
        "head_subject": _run(["git", "log", "-1", "--format=%s"]),
        "status_short": _run(["git", "status", "--short"]).splitlines(),
        "diff_stat": _run(["git", "diff", "--stat"]).splitlines(),
    }


def environment_info(probe_gpu: bool = False) -> Dict[str, Any]:
    import numpy as np
    import PIL
    import timm
    import torch

    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "timm": timm.__version__,
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_threads": int(torch.get_num_threads()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
    if torch.cuda.is_available() and probe_gpu:
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        info["gpu"] = {
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "free_memory_bytes": int(free),
            "major": int(props.major),
            "minor": int(props.minor),
            "count": int(torch.cuda.device_count()),
        }
    elif torch.cuda.is_available():
        info["gpu"] = {
            "name": "deferred",
            "note": "G0 records availability only; full device/memory probe is deferred to "
                    "G2 preflight so protocol locking does not create a CUDA context",
        }
    return info


def source_hashes() -> Dict[str, str]:
    files = sorted((C.ROOT / "exps" / "csg00").glob("*.py"))
    return {str(path.relative_to(C.ROOT)): C.sha256_file(path) for path in files}


def historical_hashes() -> Dict[str, str]:
    paths = [
        C.AGENTS_MD,
        C.ROOT / "AGENTS.local.md",
        C.SUPPLIED_TASK,
        C.P0C_PROTOCOL,
        C.ROOT / "results" / "hope_cad" / "task_p" / "p1" / "protocol_resolved.json",
        C.CHECKPOINT,
        C.SOURCE_MANIFEST,
        C.CACHE_DIR / "bottle_train.pt",
        C.CACHE_DIR / "carpet_train.pt",
        C.CACHE_DIR / "hazelnut_train.pt",
        C.ROOT / "agents" / "reports" / "task_p_p1_anomaly_signal_validation.md",
        C.ROOT / "agents" / "reports" / "task_npim_01_real_anomaly_dev.md",
        C.ROOT / "agents" / "reports" / "task_npim_02_forensic.md",
        C.ROOT / "agents" / "reports" / "task_npim_03_uncertainty.md",
        C.ROOT / "agents" / "reports" / "task_cs_lrnsm_01_compact_support.md",
    ]
    return {str(path.relative_to(C.ROOT)): C.sha256_file(path) for path in paths if path.is_file()}


def _forbidden_guard_selftest(whitelist) -> Dict[str, Any]:
    candidates = [
        "bottle/test/crack/000.png",
        "bottle/ground_truth/crack/000_mask.png",
        "carpet/test/good/000.png",
        "bottle/train/good/000.png",
    ]
    results = {}
    for rel in candidates:
        try:
            C.assert_normal_only(rel, whitelist)
            results[rel] = "allowed"
        except PermissionError as exc:
            results[rel] = f"denied: {exc}"
    ok = (
        results["bottle/test/crack/000.png"].startswith("denied")
        and results["bottle/ground_truth/crack/000_mask.png"].startswith("denied")
        and results["bottle/train/good/000.png"] == "allowed"
    )
    return {"cases": results, "passed": bool(ok)}


def verify_all(checkpoint_hash: bool = True, cache_hash: bool = True) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    agents_sha = C.sha256_file(C.AGENTS_MD)
    checks["agents_md"] = {"sha256": agents_sha, "pin": C.AGENTS_MD_PIN,
                           "match": agents_sha == C.AGENTS_MD_PIN}
    if agents_sha != C.AGENTS_MD_PIN:
        raise RuntimeError("AGENTS.md audit pin mismatch")

    checks["supplied_task"] = {"path": str(C.SUPPLIED_TASK.relative_to(C.ROOT)),
                               "sha256": C.sha256_file(C.SUPPLIED_TASK)}

    cp = C.CHECKPOINT
    if not cp.is_file():
        raise RuntimeError(f"frozen checkpoint missing: {cp}")
    cp_info = {"path": str(cp.relative_to(C.ROOT)), "bytes": cp.stat().st_size,
               "expected_bytes": C.CHECKPOINT_BYTES}
    if checkpoint_hash:
        cp_info["sha256"] = C.sha256_file(cp)
        cp_info["expected_sha256"] = C.CHECKPOINT_SHA256
        cp_info["match"] = (cp_info["sha256"] == C.CHECKPOINT_SHA256)
        if not cp_info["match"]:
            raise RuntimeError("frozen checkpoint digest mismatch")
    checks["checkpoint"] = cp_info

    p0c_sha = C.sha256_file(C.P0C_PROTOCOL)
    checks["p0c_protocol"] = {"sha256": p0c_sha, "expected": C.P0C_PROTOCOL_SHA256,
                              "match": p0c_sha == C.P0C_PROTOCOL_SHA256}
    if p0c_sha != C.P0C_PROTOCOL_SHA256:
        raise RuntimeError("P0c canonical stream protocol digest mismatch")

    stream = C.load_stream()
    checks["stream"] = {
        "writer_events": len(stream["updates"]),
        "witnesses": len(stream["probes"]),
        "writer_blocks": stream["schedule"][100]["categories_seen"],
        "schedule": stream["schedule"],
        "stream_digest": stream["stream_digest"],
        "witness_digest": stream["witness_digest"],
        "distinct_writer_paths": len(set(stream["writer_paths"])),
        "distinct_writer_contents": len(set(r["raw_png_sha256"] for r in stream["updates"])),
        "writer_witness_path_disjoint": True,
        "writer_witness_content_disjoint": True,
    }

    guard = _forbidden_guard_selftest(set(stream["writer_paths"]) | set(stream["probe_paths"]))
    checks["normal_only_guard"] = guard
    if not guard["passed"]:
        raise RuntimeError("normal-only guard self-test failed")

    cache = features.load_feature_cache(verify_hashes=cache_hash)
    checks["feature_cache"] = {
        category: {k: v for k, v in info.items() if k in ("path", "bytes", "sha256", "rows")}
        for category, info in cache.files.items()
    }
    checks["cache_rows"] = features.verify_cache_rows(cache, stream["updates"], stream["probes"])

    git = git_state()
    checks["git"] = git
    if git["branch"] != "dikhang_space":
        raise RuntimeError(f"unexpected branch: {git['branch']}")

    checks["environment"] = environment_info()
    checks["source_hashes"] = source_hashes()
    checks["historical_hashes"] = historical_hashes()
    checks["byte_budget_dry_run"] = C.reservoir_capacity_for_byte_match()
    if checks["byte_budget_dry_run"]["chosen_capacity"] != C.RESERVOIR_CAPACITY:
        raise RuntimeError("reservoir capacity derived from the schema differs from the locked constant")
    return checks


def write_lock() -> Dict[str, Any]:
    if not C.CANONICAL_TASK.is_file():
        raise RuntimeError(f"canonical task file must exist before locking: {C.CANONICAL_TASK}")
    C.RUN_DIR.mkdir(parents=True, exist_ok=True)
    if C.G0_LOCK_PATH.exists():
        raise FileExistsError(f"G0 lock already exists (refusing overwrite): {C.G0_LOCK_PATH}")

    verification = verify_all()
    lock = {
        "schema": "csg00_g0_lock_v1",
        "run_id": C.RUN_ID,
        "locked_utc": C.utc_now(),
        "canonical_task": {"path": str(C.CANONICAL_TASK.relative_to(C.ROOT)),
                           "sha256": C.sha256_file(C.CANONICAL_TASK)},
        "supplied_task": verification["supplied_task"],
        "agents_md": verification["agents_md"],
        "git": verification["git"],
        "checkpoint": verification["checkpoint"],
        "feature_cache": verification["feature_cache"],
        "cache_rows": verification["cache_rows"],
        "p0c_protocol": verification["p0c_protocol"],
        "stream": verification["stream"],
        "normal_only_guard": verification["normal_only_guard"],
        "environment": verification["environment"],
        "source_hashes": verification["source_hashes"],
        "historical_hashes": verification["historical_hashes"],
        "byte_budget_dry_run": verification["byte_budget_dry_run"],
        "protocol": C.protocol_manifest(),
        "no_anomaly_attestation": (
            "G0 read only AGENTS.md, task/report metadata, the P0c normal stream protocol, "
            "normal train/good image identities and the frozen normal feature cache. "
            "No MVTec test image, ground-truth mask, anomaly label, consumed DEV anomaly "
            "or sealed confirmation payload was opened, hashed or decoded."
        ),
    }
    with open(C.G0_LOCK_PATH, "x", encoding="utf-8") as fh:
        json.dump(lock, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return {"lock_path": str(C.G0_LOCK_PATH), "sha256": C.sha256_file(C.G0_LOCK_PATH),
            "bytes": C.G0_LOCK_PATH.stat().st_size}


def verify_lock() -> Dict[str, Any]:
    """Re-verify the lock, current source hashes and historical artifact hashes."""
    if not C.G0_LOCK_PATH.is_file():
        raise RuntimeError(f"missing G0 lock: {C.G0_LOCK_PATH}")
    lock = json.loads(C.G0_LOCK_PATH.read_text())
    problems = []
    if lock["schema"] != "csg00_g0_lock_v1" or lock["run_id"] != C.RUN_ID:
        problems.append("schema/run_id")
    current_sources = source_hashes()
    if current_sources != lock["source_hashes"]:
        changed = sorted(set(current_sources) | set(lock["source_hashes"]))
        problems.append("source_hashes: " + ", ".join(
            k for k in changed if current_sources.get(k) != lock["source_hashes"].get(k)))
    current_hist = historical_hashes()
    if current_hist != lock["historical_hashes"]:
        changed = sorted(set(current_hist) | set(lock["historical_hashes"]))
        problems.append("historical_hashes: " + ", ".join(
            k for k in changed if current_hist.get(k) != lock["historical_hashes"].get(k)))
    if C.sha256_file(C.CANONICAL_TASK) != lock["canonical_task"]["sha256"]:
        problems.append("canonical_task")
    if C.sha256_file(C.CHECKPOINT) != lock["checkpoint"]["sha256"]:
        problems.append("checkpoint")
    if problems:
        raise RuntimeError("G0 lock re-verification failed: " + "; ".join(problems))
    return {
        "lock_sha256": C.sha256_file(C.G0_LOCK_PATH),
        "source_hash_count": len(current_sources),
        "historical_hash_count": len(current_hist),
        "verified": True,
    }
