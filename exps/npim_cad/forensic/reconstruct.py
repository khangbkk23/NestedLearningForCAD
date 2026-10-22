# exps/npim_cad/forensic/reconstruct.py
"""Frozen per-level reconstruction for EXP-NPIM-02.

Encodes the EXACT 60 admitted P1 DEV identities once with the pinned ViT-B/8,
recomputes per-level Fast/Mid/Slow predictions and the exact fused score, and
compares all 540 fused maps to the sealed NPIM-01 payload.  Raw Z/Y tensors are
temporary and are not persisted; only derived scores, cross terms, parity
evidence and capacity sufficient statistics are written.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exps.npim_cad.forensic.decompose import (  # noqa: E402
    PAIR_INDEX,
    decompose_mixture,
    ready_weights,
)
from exps.npim_cad.npim01_contract import (  # noqa: E402
    ARMS,
    BOUNDARIES,
    CHECKPOINT,
    CP_SHA,
    MVTEC,
    admitted_categories,
    load_contract,
    load_state_detectors,
    sha256_file,
    source_hashes,
)
from exps.npim_cad.npim01_metrics import load_raw_image  # noqa: E402
from models.hope_cad.npim_context import NPIMContextConfig, NPIMFeatureExtractor  # noqa: E402
from models.hope_cad.npim_state import rng_fingerprint  # noqa: E402

RESULTS = ROOT / "results/hope_cad/npim_02"
NPIM01_RUN = ROOT / "results/hope_cad/npim_01/npim01_dev_20261021_cuda"
PREDICTION_SHA = "7c535937b2ff6e2ed5f6f6197a1f84650ab2a046063e5596d584f5cd4f6daa99"
MAX_ABS_TOLERANCE = 1.0e-4
REL_L2_TOLERANCE = 1.0e-5


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


def environment(device: str) -> Dict[str, Any]:
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def load_sealed(run_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    seal = json.loads((run_dir / "predictions_seal.json").read_text())
    predictions_path = run_dir / "predictions.pt"
    actual = sha256_file(predictions_path)
    if actual != PREDICTION_SHA or actual != seal["sha256"]:
        raise RuntimeError(f"NPIM-01 sealed prediction identity mismatch: {actual}")
    payload = torch.load(predictions_path, map_location="cpu", weights_only=False)
    if tuple(payload["patch_scores"].shape) != (540, 784):
        raise RuntimeError("NPIM-01 sealed patch-score geometry mismatch")
    return seal, payload


def sealed_lookup(payload: Mapping[str, Any]) -> Dict[tuple[str, int, int], int]:
    arms = list(payload["arms"])
    lookup: Dict[tuple[str, int, int], int] = {}
    for entry in range(len(payload["patch_scores"])):
        key = (arms[int(payload["arm_index"][entry])], int(payload["boundary"][entry]), int(payload["record_index"][entry]))
        if key in lookup:
            raise RuntimeError(f"duplicate sealed entry {key}")
        lookup[key] = entry
    if len(lookup) != 540:
        raise RuntimeError("sealed lookup is not 540 entries")
    return lookup


def _state_hashes(detectors: Mapping[tuple[str, int], Any]) -> Dict[str, str]:
    return {f"{arm}@{boundary}": detector.state_hash() for (arm, boundary), detector in detectors.items()}


def reconstruct(run_id: str, device: str) -> Dict[str, Any]:
    run_dir = RESULTS / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite forensic run directory: {run_dir}")
    started = time.perf_counter()
    contract = load_contract()
    seal, sealed = load_sealed(NPIM01_RUN)
    lookup = sealed_lookup(sealed)
    detectors = load_state_detectors(device=device)
    extractor = NPIMFeatureExtractor(
        NPIMContextConfig(checkpoint_path=str(CHECKPOINT), checkpoint_sha256=CP_SHA),
        device=device,
    )
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    state_before = _state_hashes(detectors)
    rng_before = rng_fingerprint()

    level_scores = torch.full((540, 3, 784), float("nan"), dtype=torch.float32)
    dot_products = torch.full((540, 3, 784), float("nan"), dtype=torch.float32)
    weights = torch.full((540, 3), float("nan"), dtype=torch.float32)
    ready = torch.zeros((540, 3), dtype=torch.bool)
    fused_scores = torch.full((540, 784), float("nan"), dtype=torch.float32)
    fused_image_score = torch.full((540,), float("nan"), dtype=torch.float32)
    entry_arm = torch.zeros(540, dtype=torch.int64)
    entry_boundary = torch.zeros(540, dtype=torch.int64)
    entry_record = torch.zeros(540, dtype=torch.int64)
    entry_sealed = torch.zeros(540, dtype=torch.int64)

    yty = torch.zeros(768, 768, dtype=torch.float64)
    ysum = torch.zeros(768, dtype=torch.float64)
    normal_count = 0
    y_energy = torch.full((60,), float("nan"), dtype=torch.float64)
    key_stats: Dict[str, Dict[str, torch.Tensor | int]] = {}
    for arm in ARMS:
        rank = int(ARMS[arm]["r"])
        key_stats[arm] = {
            "ktk": torch.zeros(rank, rank, dtype=torch.float64),
            "ksum": torch.zeros(rank, dtype=torch.float64),
            "count": 0,
        }

    parity_rows: List[Dict[str, Any]] = []
    for record in contract.records:
        image = load_raw_image(MVTEC, record)
        z, y = extractor.encode(image)
        z0, y0 = z[0], y[0]
        keys = {arm: detectors[(arm, 100)].build_keys(z0) for arm in ARMS}
        if record.label == 0:
            normal_count += 1
            y64 = y0.detach().to("cpu", dtype=torch.float64)
            yty += y64.T @ y64
            ysum += y64.sum(dim=0)
            y_energy[record.index] = (y64 * y64).sum()
            for arm, K in keys.items():
                k64 = K.detach().to("cpu", dtype=torch.float64)
                key_stats[arm]["ktk"] += k64.T @ k64
                key_stats[arm]["ksum"] += k64.sum(dim=0)
                key_stats[arm]["count"] += 1

        for boundary in BOUNDARIES:
            if record.category not in admitted_categories(boundary):
                continue
            for arm in ARMS:
                entry = lookup[(arm, boundary, record.index)]
                detector = detectors[(arm, boundary)]
                indices, mixture_weights = ready_weights(detector)
                Ws = [detector.levels[index].W for index in indices]
                decomposition = decompose_mixture(y0, keys[arm], Ws, mixture_weights)
                level_scores[entry, indices] = decomposition.level_scores.detach().to("cpu", copy=True)
                fused_scores[entry] = decomposition.fused_scores.detach().to("cpu", copy=True)
                fused_image_score[entry] = decomposition.fused_scores.max().to("cpu")
                ready[entry, indices] = True
                weights[entry, indices] = mixture_weights.to("cpu")
                local_pairs = [pair for pair in PAIR_INDEX if pair[0] < len(indices) and pair[1] < len(indices)]
                for local_index, (left, right) in enumerate(local_pairs):
                    global_left, global_right = indices[left], indices[right]
                    slot = PAIR_INDEX.index((global_left, global_right))
                    dot_products[entry, slot] = decomposition.dot_products[local_index].detach().to("cpu", copy=True)
                entry_arm[entry] = list(ARMS).index(arm)
                entry_boundary[entry] = boundary
                entry_record[entry] = record.index
                entry_sealed[entry] = entry

                sealed_patch = sealed["patch_scores"][entry].numpy()
                reconstructed = decomposition.fused_scores.detach().to("cpu", copy=True).numpy()
                difference = np.abs(reconstructed - sealed_patch)
                max_abs = float(difference.max())
                relative_l2 = float(
                    np.linalg.norm(reconstructed - sealed_patch) / max(np.linalg.norm(sealed_patch), 1e-12)
                )
                image_delta = float(abs(float(decomposition.fused_scores.max().item()) - float(sealed["image_score"][entry])))
                parity_rows.append(
                    {
                        "entry": entry,
                        "arm": arm,
                        "boundary": boundary,
                        "record_index": record.index,
                        "max_abs": max_abs,
                        "relative_l2": relative_l2,
                        "image_score_max_abs": image_delta,
                        "identity_max_abs": float(decomposition.identity_max_abs),
                        "effective_weight_max_abs": float(decomposition.effective_weight_error),
                        "pass": bool(max_abs <= MAX_ABS_TOLERANCE and relative_l2 <= REL_L2_TOLERANCE),
                    }
                )
        if record.index % 10 == 0:
            print(f"NPIM-02 reconstructed {record.index + 1}/{len(contract.records)}", flush=True)
        del z, y, z0, y0, keys

    if not torch.isfinite(fused_scores).all():
        raise RuntimeError("non-finite reconstructed fused scores")
    counts = Counter((int(entry_boundary[i]), list(ARMS)[int(entry_arm[i])]) for i in range(540))
    expected_counts = {
        (boundary, arm): sum(1 for record in contract.records if record.category in admitted_categories(boundary))
        for boundary in BOUNDARIES
        for arm in ARMS
    }
    if dict(counts) != expected_counts:
        raise RuntimeError(f"unexpected reconstructed entry counts: {dict(counts)} != {expected_counts}")
    if _state_hashes(detectors) != state_before:
        raise RuntimeError("reconstruction mutated an NPIM state")
    if rng_fingerprint() != rng_before:
        raise RuntimeError("reconstruction mutated the RNG state")

    failed = [row for row in parity_rows if not row["pass"]]
    parity = {
        "schema": "npim02_reconstruction_parity_v1",
        "tolerance": {"max_abs": MAX_ABS_TOLERANCE, "relative_l2": REL_L2_TOLERANCE},
        "entries": len(parity_rows),
        "failed_entries": len(failed),
        "all_pass": len(failed) == 0,
        "max_abs_global": float(max(row["max_abs"] for row in parity_rows)),
        "max_relative_l2_global": float(max(row["relative_l2"] for row in parity_rows)),
        "max_image_score_abs_global": float(max(row["image_score_max_abs"] for row in parity_rows)),
        "max_identity_error_global": float(max(row["identity_max_abs"] for row in parity_rows)),
        "max_effective_weight_error_global": float(max(row["effective_weight_max_abs"] for row in parity_rows)),
        "state_hashes_unchanged": True,
        "rng_unchanged": True,
        "rows": parity_rows,
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(run_dir / "reconstruction" / "parity.json", parity)
    if failed:
        write_json_exclusive(run_dir / "BLOCKED_PARITY.json", {
            "decision": "NPIM02_BLOCKED_RECONSTRUCTION_PARITY",
            "failed_entries": len(failed),
            "first_failures": failed[:5],
            "tolerance": parity["tolerance"],
        })
        raise RuntimeError(f"reconstruction parity failed for {len(failed)} entries; STOP")

    per_level = {
        "schema": "npim02_per_level_scores_v1",
        "level_names": ["fast", "mid", "slow"],
        "arms": list(ARMS),
        "boundaries": list(BOUNDARIES),
        "records": [
            {"index": int(record.index), "relative_path": record.relative_path,
             "sha256": record.sha256, "category": record.category, "hw": list(record.hw)}
            for record in contract.records
        ],
        "entry_arm": entry_arm,
        "entry_boundary": entry_boundary,
        "entry_record": entry_record,
        "level_scores": level_scores,
        "fused_scores": fused_scores,
        "weights": weights,
        "ready": ready,
        "image_score": fused_image_score,
        "sealed_patch_scores_sha256": PREDICTION_SHA,
    }
    cross = {
        "schema": "npim02_cross_terms_v1",
        "pair_index": [list(pair) for pair in PAIR_INDEX],
        "dot_products": dot_products,
        "weights": weights,
        "ready": ready,
        "level_scores": level_scores,
    }
    capacity = {
        "schema": "npim02_capacity_inputs_v1",
        "normal_count": normal_count,
        "yty": yty,
        "ysum": ysum,
        "y_energy": y_energy,
        "key_stats": key_stats,
        "record_indices": [int(record.index) for record in contract.records if record.label == 0],
    }
    torch.save(per_level, run_dir / "reconstruction" / "per_level_scores.pt")
    torch.save(cross, run_dir / "reconstruction" / "cross_terms.pt")
    torch.save(capacity, run_dir / "reconstruction" / "capacity_inputs.pt")
    reconstruction_seal = {
        "schema": "npim02_reconstruction_seal_v1",
        "created_utc": utc_now(),
        "run_id": run_id,
        "sealed_prediction_sha256": PREDICTION_SHA,
        "npim01_metrics_sha256": "713530f739516131abd0e5fa577782c610c418cda1e108956a18c715f9b27380",
        "protocol_sha256": contract.protocol_sha256,
        "cohort_digest": contract.cohort_digest,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "state_file_sha256": {f"{arm}@{boundary}": digest for (arm, boundary), digest in contract.state_sha256.items()},
        "source_sha256": dict(contract.source_sha256),
        "parity_tolerance": parity["tolerance"],
        "parity_all_pass": True,
        "per_level_scores_sha256": sha256_file(run_dir / "reconstruction" / "per_level_scores.pt"),
        "cross_terms_sha256": sha256_file(run_dir / "reconstruction" / "cross_terms.pt"),
        "capacity_inputs_sha256": sha256_file(run_dir / "reconstruction" / "capacity_inputs.pt"),
    }
    write_json_exclusive(run_dir / "reconstruction" / "reconstruction_seal.json", reconstruction_seal)
    write_text_exclusive(run_dir / "reconstruction" / "E_RECON_SEALED", reconstruction_seal["per_level_scores_sha256"] + "\n")
    resource_info = {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    if torch.cuda.is_available() and device.startswith("cuda"):
        resource_info["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        resource_info["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    manifest = {
        "schema": "npim02_reconstruction_manifest_v1",
        "run_id": run_id,
        "created_utc": utc_now(),
        "device": device,
        "environment": environment(device),
        "checkpoint_sha256": contract.checkpoint_sha256,
        "protocol_sha256": contract.protocol_sha256,
        "cohort_digest": contract.cohort_digest,
        "sealed_prediction_sha256": PREDICTION_SHA,
        "state_hash": state_before,
        "parity": parity,
        "resource": resource_info,
        "source_sha256": dict(contract.source_sha256),
        "note": "raw Z/Y tensors were temporary and are not persisted; only derived scores/statistics are stored",
    }
    write_json_exclusive(run_dir / "manifest.json", manifest)
    extractor.close()
    return {"run_dir": str(run_dir), "parity": parity, "resource": resource_info}

def finalize(run_id: str) -> Dict[str, Any]:
    """Seal an already-completed one-pass reconstruction; never re-encodes."""

    run_dir = RESULTS / run_id
    parity = json.loads((run_dir / "reconstruction" / "parity.json").read_text())
    if not parity.get("all_pass"):
        raise RuntimeError("cannot finalize a reconstruction whose parity did not pass")
    per_level = torch.load(run_dir / "reconstruction" / "per_level_scores.pt", map_location="cpu", weights_only=False)
    cross = torch.load(run_dir / "reconstruction" / "cross_terms.pt", map_location="cpu", weights_only=False)
    capacity = torch.load(run_dir / "reconstruction" / "capacity_inputs.pt", map_location="cpu", weights_only=False)
    if tuple(per_level["fused_scores"].shape) != (540, 784):
        raise RuntimeError("per-level fused-score geometry mismatch")
    contract = load_contract()
    state_file_sha = {f"{arm}@{boundary}": digest for (arm, boundary), digest in contract.state_sha256.items()}
    forensic_sources = {}
    for relative in (
        "exps/npim_cad/forensic/decompose.py",
        "exps/npim_cad/forensic/reconstruct.py",
        "exps/npim_cad/forensic/analysis.py",
        "exps/npim_cad/forensic/spatial.py",
        "exps/npim_cad/forensic/run.py",
        "agents/tasks/task_npim_02_forensic.md",
    ):
        path = ROOT / relative
        if path.is_file():
            forensic_sources[relative] = sha256_file(path)
    seal = {
        "schema": "npim02_reconstruction_seal_v1",
        "created_utc": utc_now(),
        "run_id": run_id,
        "sealed_prediction_sha256": PREDICTION_SHA,
        "npim01_metrics_sha256": "713530f739516131abd0e5fa577782c610c418cda1e108956a18c715f9b27380",
        "protocol_sha256": contract.protocol_sha256,
        "cohort_digest": contract.cohort_digest,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "state_file_sha256": state_file_sha,
        "source_sha256": dict(contract.source_sha256),
        "forensic_source_sha256": forensic_sources,
        "parity_tolerance": parity["tolerance"],
        "parity_all_pass": True,
        "per_level_scores_sha256": sha256_file(run_dir / "reconstruction" / "per_level_scores.pt"),
        "cross_terms_sha256": sha256_file(run_dir / "reconstruction" / "cross_terms.pt"),
        "capacity_inputs_sha256": sha256_file(run_dir / "reconstruction" / "capacity_inputs.pt"),
        "reconstruction_note": "one frozen encode per admitted image; raw Z/Y were temporary and are not persisted",
    }
    write_json_exclusive(run_dir / "reconstruction" / "reconstruction_seal.json", seal)
    write_text_exclusive(run_dir / "reconstruction" / "E_RECON_SEALED", seal["per_level_scores_sha256"] + "\n")
    manifest = {
        "schema": "npim02_reconstruction_manifest_v1",
        "run_id": run_id,
        "created_utc": utc_now(),
        "finalized_from_existing_one_pass_artifacts": True,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "protocol_sha256": contract.protocol_sha256,
        "cohort_digest": contract.cohort_digest,
        "sealed_prediction_sha256": PREDICTION_SHA,
        "state_file_sha256": state_file_sha,
        "parity": parity,
        "source_sha256": dict(contract.source_sha256),
        "forensic_source_sha256": forensic_sources,
        "resource": {
            "reconstruction_seconds": None,
            "note": "the one-pass numerical reconstruction completed and wrote all derived artifacts; the process then failed while serializing tuple-keyed metadata, so exact wall-clock/peak VRAM were not captured. No second encode was performed.",
        },
    }
    write_json_exclusive(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "parity": parity, "seal": seal}

