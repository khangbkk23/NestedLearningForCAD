# exps/npim_cad/uncertainty/calibrate.py
"""EXP-NPIM-03 G2/G3: bounded normal-only residual calibration.

Encodes the 59 eligible P0c normal probes exactly once with the pinned ViT-B/8,
reuses the cached Y and per-arm keys for all 12 frozen NPIM states, fits global
per-channel normal residual moments on the fit split, and evaluates only the
normal-only witness split.  No anomaly image, mask, DEV record or sealed payload
is opened.  No NPIM weight/state is updated.
"""

from __future__ import annotations

import json
import math
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy.stats import kurtosis, ks_2samp, skew, spearmanr
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exps.npim_cad.npim01_contract import (  # noqa: E402
    ARMS,
    BOUNDARIES,
    CHECKPOINT,
    CP_SHA,
    PILOT,
    STATE_HASH,
    STATE_SHA256,
    load_state_detectors,
    sha256_file,
)
from exps.npim_cad.uncertainty.cohort import (  # noqa: E402
    ProbeRecord,
    audit as cohort_audit,
    eligible,
    fit_records,
    load_normal_image,
    load_probes,
    witness_records,
)
from exps.npim_cad.uncertainty.moments import (  # noqa: E402
    anisotropy,
    image_bootstrap_anisotropy,
    regularization,
    PerChannelMoments,
)
from exps.npim_cad.uncertainty.readout import (  # noqa: E402
    a0_parity,
    image_max,
    score_patch_residual,
    score_keys,
)
from models.hope_cad.npim_context import NPIMContextConfig, NPIMFeatureExtractor  # noqa: E402
from models.hope_cad.npim_state import rng_fingerprint  # noqa: E402

RESULTS = ROOT / "results/hope_cad/npim_03"
PROTOCOL = ROOT / "exps/npim_cad/protocol_npim01.md"
PROTOCOL_SHA = "75f68db4ddbda68bedb836e8485bc16056fed6b2e70f20dd009115303c54d72d"
RHO = 0.10
TAU = 0.01
EPS_ABS = 1e-8
ANISOTROPY_NULL = 1.10
ANISOTROPY_LOWER = 1.05
BOOTSTRAP_CV_MAX = 0.30
OFFDIAGONAL_MAX = 0.50
KS_MAX = 0.10
DEVICE_BUDGET = int(4.5 * 1024**3)
READOUT_KEY_PRIMARY = "C3@350"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def verify_lineage() -> Dict[str, Any]:
    if not CHECKPOINT.is_file() or sha256_file(CHECKPOINT) != CP_SHA:
        raise RuntimeError("frozen checkpoint identity mismatch")
    if not PROTOCOL.is_file() or sha256_file(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("NPIM-01 protocol identity mismatch")
    state_files = {}
    for (arm, boundary), expected in STATE_SHA256.items():
        path = PILOT / "states" / f"{arm}_{boundary}.pt"
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"state file identity mismatch: {arm}@{boundary}")
        state_files[f"{arm}@{boundary}"] = expected
    return {
        "checkpoint_sha256": CP_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "state_file_sha256": state_files,
    }


def encode_probes(records: Sequence[ProbeRecord], device: str) -> tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
    extractor = NPIMFeatureExtractor(
        NPIMContextConfig(checkpoint_path=str(CHECKPOINT), checkpoint_sha256=CP_SHA),
        device=device,
    )
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    detectors = load_state_detectors("cpu")
    targets: List[torch.Tensor] = []
    keys: Dict[str, List[torch.Tensor]] = {arm: [] for arm in ARMS}
    started = time.perf_counter()
    try:
        for position, record in enumerate(records):
            image = load_normal_image(record)
            z, y = extractor.encode(image.unsqueeze(0))
            z0, y0 = z[0], y[0]
            targets.append(y0.detach().to("cpu", copy=True))
            for arm in ARMS:
                keys[arm].append(detectors[(arm, 100)].build_keys(z0).detach().to("cpu", copy=True))
            if position % 10 == 0:
                print(f"NPIM-03 encoded {position + 1}/{len(records)}", flush=True)
    finally:
        extractor.close()
    resource_info = {
        "encoded_images": len(records),
        "encode_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    if torch.cuda.is_available() and device.startswith("cuda"):
        resource_info["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        resource_info["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    return torch.stack(targets, dim=0), {arm: torch.stack(values, dim=0) for arm, values in keys.items()}, resource_info


def _score_stats(patch_scores: Sequence[np.ndarray], image_scores: Sequence[float]) -> Dict[str, Any]:
    patch = np.concatenate([np.asarray(value, dtype=np.float64).ravel() for value in patch_scores])
    images = np.asarray(image_scores, dtype=np.float64)
    return {
        "patch_mean": float(patch.mean()),
        "patch_median": float(np.median(patch)),
        "patch_p95": float(np.quantile(patch, 0.95)),
        "patch_p99": float(np.quantile(patch, 0.99)),
        "image_max_mean": float(images.mean()),
        "image_max_p95": float(np.quantile(images, 0.95)),
        "images": int(images.size),
        "patch_count": int(patch.size),
    }


def _rank_correlation(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> float | None:
    left = np.concatenate([np.asarray(value).ravel()[::8] for value in a])
    right = np.concatenate([np.asarray(value).ravel()[::8] for value in b])
    if left.size < 3 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def decide_primary(primary: Mapping[str, Any]) -> tuple[str, str]:
    """Apply the frozen normal-only gates to the C3@350 readout."""

    geometry = primary["geometry_raw"]
    bootstrap = primary["bootstrap"]
    lower = float(bootstrap["p95_over_p5_interval"][0])
    boot_cv = float(bootstrap["bootstrap_channel_cv_mean"])
    offdiag_energy = float(primary["covariance"]["offdiag_energy_fraction"])
    ks_a2 = float(primary["fit_witness"]["A2"]["ks"])
    var = primary["var_raw"]
    values = var.detach().cpu().numpy() if hasattr(var, "detach") else np.asarray(var)
    if not np.isfinite(values).all():
        return "NPIM03_NORMAL_READOUT_STATISTICALLY_UNRELIABLE", "non-finite variance"
    if float(geometry["p95_over_p5"]) < ANISOTROPY_NULL:
        return (
            "NPIM03_NULL_ISOTROPIC_RESIDUAL",
            "estimated variances are nearly identical; A2 is only a rescaling of A1",
        )
    reasons = []
    if lower <= ANISOTROPY_LOWER:
        reasons.append(f"bootstrap R lower bound {lower:.3f} <= {ANISOTROPY_LOWER}")
    if boot_cv > BOOTSTRAP_CV_MAX:
        reasons.append(f"bootstrap channel-variance CV {boot_cv:.3f} > {BOOTSTRAP_CV_MAX}")
    if offdiag_energy > OFFDIAGONAL_MAX:
        reasons.append(f"off-diagonal covariance energy {offdiag_energy:.3f} > {OFFDIAGONAL_MAX}")
    if ks_a2 > KS_MAX:
        reasons.append(f"fit/witness A2 KS {ks_a2:.3f} > {KS_MAX}")
    if reasons:
        return "NPIM03_NORMAL_READOUT_STATISTICALLY_UNRELIABLE", "; ".join(reasons)
    return (
        "NPIM03_NORMAL_READOUT_READY_FOR_EXTERNAL_EVAL",
        "C3@350 variance geometry is anisotropic and stable under whole-image bootstrap; "
        "only a future authorized G4 may test AP",
    )


def calibrate(run_id: str, device: str) -> Dict[str, Any]:
    started = time.perf_counter()
    run_dir = RESULTS / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite NPIM-03 run directory: {run_dir}")
    lineage = verify_lineage()
    probes = load_probes()
    probe_audit = cohort_audit(probes)
    if probe_audit["probe_count"] != 59 or not probe_audit["all_train_good"] or not probe_audit["writer_disjoint"]:
        raise RuntimeError("normal calibration source is not eligible")
    targets, keys, resource_info = encode_probes(probes, device)
    if resource_info.get("gpu_peak_reserved_bytes", 0) > DEVICE_BUDGET:
        raise RuntimeError("NPIM03_BLOCKED_RESOURCE: CUDA reserved memory above budget")

    position = {record.relative_path: index for index, record in enumerate(probes)}
    detectors = load_state_detectors("cpu")
    state_hashes_before = {f"{arm}@{boundary}": detector.state_hash() for (arm, boundary), detector in detectors.items()}
    rng_before = rng_fingerprint()
    readouts: Dict[str, Any] = {}
    analysis_readouts: Dict[str, Any] = {}
    run_dir.mkdir(parents=True, exist_ok=False)

    for boundary in BOUNDARIES:
        fit = fit_records(probes, boundary)
        witness = witness_records(probes, boundary)
        if not fit or not witness:
            raise RuntimeError(f"empty fit/witness set at boundary {boundary}")
        for arm in ARMS:
            key = f"{arm}@{boundary}"
            detector = detectors[(arm, boundary)]
            fit_indices = [position[record.relative_path] for record in fit]
            witness_indices = [position[record.relative_path] for record in witness]
            moments = PerChannelMoments()
            fit_scores = {"A0": [], "A1": [], "A2": []}
            fit_image_scores = {"A0": [], "A1": [], "A2": []}
            subsample: List[np.ndarray] = []
            parity = None
            for index in fit_indices:
                K = keys[arm][index]
                Y = targets[index]
                scored = score_keys(detector, K, Y)
                residual = scored["residual"]
                if parity is None:
                    parity = a0_parity(detector, K, Y)
                    if not parity["pass"]:
                        raise RuntimeError(f"A0 parity failed for {key}: {parity}")
                moments.add_image_group(residual.numpy())
                subsample.append(residual.numpy()[::4].astype(np.float64))
            var_raw = moments.variance()
            regularization_info = regularization(var_raw, RHO, TAU, EPS_ABS)
            mu = moments.mu.astype(np.float32)
            var_safe = regularization_info["var_safe"]
            # Fit scores after moments are available (A1/A2 use the fitted stats).
            for index in fit_indices:
                scored = score_keys(detector, keys[arm][index], targets[index])
                residual = scored["residual"]
                for kind in ("A0", "A1", "A2"):
                    patches = score_patch_residual(residual, kind=kind, mu=mu, var_safe=var_safe).numpy()
                    fit_scores[kind].append(patches)
                    fit_image_scores[kind].append(image_max(torch.from_numpy(patches)))
            # Witness scores.
            witness_scores = {"A0": [], "A1": [], "A2": []}
            witness_image_scores = {"A0": [], "A1": [], "A2": []}
            for index in witness_indices:
                scored = score_keys(detector, keys[arm][index], targets[index])
                residual = scored["residual"]
                for kind in ("A0", "A1", "A2"):
                    patches = score_patch_residual(residual, kind=kind, mu=mu, var_safe=var_safe).numpy()
                    witness_scores[kind].append(patches)
                    witness_image_scores[kind].append(image_max(torch.from_numpy(patches)))
            geometry = anisotropy(var_raw)
            bootstrap = image_bootstrap_anisotropy(
                moments.group_means, moments.group_m2, moments.group_n, reps=200, seed=20261021
            )
            covariance = np.cov(np.concatenate(subsample, axis=0), rowvar=False, bias=False)
            diagonal = np.diag(covariance)
            offdiag = covariance - np.diag(diagonal)
            offdiag_energy = float(np.sum(offdiag * offdiag) / max(np.sum(covariance * covariance), 1e-30))
            eigenvalues = np.linalg.eigvalsh(covariance)
            positive = np.maximum(eigenvalues[::-1], 0.0)
            effective_rank = float(positive.sum() ** 2 / max((positive * positive).sum(), 1e-30))
            sub = np.concatenate(subsample, axis=0)
            channel_skew = np.abs(skew(sub, axis=0, bias=False))
            channel_kurt = kurtosis(sub, axis=0, bias=False, fisher=True)
            marginal = {
                "median_abs_skew": float(np.nanmedian(channel_skew)),
                "max_abs_skew": float(np.nanmax(channel_skew)),
                "median_excess_kurtosis": float(np.nanmedian(channel_kurt)),
                "max_abs_excess_kurtosis": float(np.nanmax(np.abs(channel_kurt))),
            }
            fit_witness = {}
            for kind in ("A0", "A1", "A2"):
                fit_patch = np.concatenate(fit_scores[kind])
                witness_patch = np.concatenate(witness_scores[kind])
                fit_witness[kind] = {
                    "ks": float(ks_2samp(fit_patch, witness_patch).statistic),
                    "fit": _score_stats(fit_scores[kind], fit_image_scores[kind]),
                    "witness": _score_stats(witness_scores[kind], witness_image_scores[kind]),
                }
            rank_correlation = {
                "A0_vs_A1": _rank_correlation(witness_scores["A0"], witness_scores["A1"]),
                "A0_vs_A2": _rank_correlation(witness_scores["A0"], witness_scores["A2"]),
                "A1_vs_A2": _rank_correlation(witness_scores["A1"], witness_scores["A2"]),
            }
            readout = {
                "arm": arm,
                "boundary": boundary,
                "fit_images": len(fit),
                "witness_images": len(witness),
                "fit_categories": dict(Counter(record.category for record in fit)),
                "witness_categories": dict(Counter(record.category for record in witness)),
                "mu": torch.from_numpy(mu),
                "var_raw": torch.from_numpy(var_raw.astype(np.float32)),
                "var_safe": torch.from_numpy(var_safe.astype(np.float32)),
                "n_patches": int(moments.n_patches),
                "n_images": int(moments.n_images),
                "fit_paths": [record.relative_path for record in fit],
                "witness_paths": [record.relative_path for record in witness],
                "fit_sha256": [record.sha256 for record in fit],
                "witness_sha256": [record.sha256 for record in witness],
                "a0_parity": parity,
                "regularization": {
                    "rho": RHO, "tau": TAU, "eps_abs": EPS_ABS,
                    "mean_variance": float(regularization_info["mean_variance"]),
                },
                "geometry_raw": geometry,
                "geometry_safe": anisotropy(var_safe),
                "bootstrap": bootstrap,
                "covariance": {
                    "offdiag_energy_fraction": offdiag_energy,
                    "effective_rank_participation": effective_rank,
                    "diagonal_fraction": float(np.sum(diagonal * diagonal) / max(np.sum(covariance * covariance), 1e-30)),
                    "patches_used": int(sub.shape[0]),
                    "patch_subsample": "every 4th patch",
                },
                "marginal": marginal,
                "fit_witness": fit_witness,
                "rank_correlation": rank_correlation,
            }
            readouts[key] = readout
            analysis_readouts[key] = {
                field: value for field, value in readout.items()
                if field not in ("mu", "var_raw", "var_safe")
            }
            print(
                f"NPIM-03 {key}: R={geometry['p95_over_p5']:.3f} "
                f"boot_low={bootstrap['p95_over_p5_interval'][0]:.3f} "
                f"bootCV={bootstrap['bootstrap_channel_cv_mean']:.3f} O={offdiag_energy:.3f}",
                flush=True,
            )

    state_hashes_after = {f"{arm}@{boundary}": detector.state_hash() for (arm, boundary), detector in detectors.items()}
    if state_hashes_after != state_hashes_before:
        raise RuntimeError("calibration mutated an NPIM state")
    if rng_fingerprint() != rng_before:
        raise RuntimeError("calibration mutated the RNG state")

    primary = readouts[READOUT_KEY_PRIMARY]
    primary_decision, reason = decide_primary(primary)

    stats_payload = {
        "schema": "npim03_stats_v1",
        "run_id": run_id,
        "created_utc": utc_now(),
        "constants": {"rho": RHO, "tau": TAU, "eps_abs": EPS_ABS, "d": 768},
        "probe_audit": probe_audit,
        "lineage": lineage,
        "readouts": readouts,
    }
    torch.save(stats_payload, run_dir / "stats.pt")
    stats_sha = sha256_file(run_dir / "stats.pt")
    analysis = {
        "schema": "npim03_analysis_v1",
        "run_id": run_id,
        "created_utc": utc_now(),
        "constants": {"rho": RHO, "tau": TAU, "eps_abs": EPS_ABS},
        "primary_readout": READOUT_KEY_PRIMARY,
        "primary_decision": primary_decision,
        "primary_reason": reason,
        "thresholds": {
            "anisotropy_null": ANISOTROPY_NULL,
            "bootstrap_lower": ANISOTROPY_LOWER,
            "bootstrap_channel_cv_max": BOOTSTRAP_CV_MAX,
            "offdiagonal_max": OFFDIAGONAL_MAX,
            "ks_max": KS_MAX,
        },
        "readouts": analysis_readouts,
        "note": "normal-only diagnostics; no anomaly image, mask, DEV record or AP/AUROC was used",
    }
    write_json_exclusive(run_dir / "analysis.json", analysis)
    inference_bytes = sum(int(value["mu"].numel() * 4 + value["var_safe"].numel() * 4) for value in readouts.values())
    audit_bytes = sum(
        int(value["mu"].numel() * 4 + value["var_raw"].numel() * 4 + value["var_safe"].numel() * 4)
        for value in readouts.values()
    )
    ledger = {
        "schema": "npim03_ledger_v1",
        "readouts": len(readouts),
        "d": 768,
        "bytes_per_inference_readout": 768 * 4 * 2,
        "inference_only_bytes": inference_bytes,
        "bytes_per_audit_readout": 768 * 4 * 3,
        "audit_stats_bytes": audit_bytes,
        "stats_file_bytes": (run_dir / "stats.pt").stat().st_size,
        "npim_state_bytes_C1_C2_C3": [1423963, 1429500, 1429513],
        "disabled_npim_variance_buffer_bytes": 3072,
        "encoder_checkpoint_bytes": 343282995,
        "raw_features_persisted": False,
        "calibration_images_retained_in_state": False,
        "resumable_online_note": "mu/variance counters are sufficient to resume, but re-accumulation would require re-accessing historical normal images; source paths/hashes are recorded in stats.pt",
    }
    write_json_exclusive(run_dir / "ledger.json", ledger)
    resource_info.update({
        "calibration_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "device": device,
    })
    manifest = {
        "schema": "npim03_manifest_v1",
        "run_id": run_id,
        "created_utc": utc_now(),
        "branch": __import__("subprocess").run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "head": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "lineage": lineage,
        "probe_audit": probe_audit,
        "constants": {"rho": RHO, "tau": TAU, "eps_abs": EPS_ABS},
        "state_hashes_before": state_hashes_before,
        "state_hashes_after": state_hashes_after,
        "rng_before": rng_before,
        "rng_after": rng_fingerprint(),
        "stats_sha256": stats_sha,
        "resource": resource_info,
        "data_access": {
            "normal_probes_only": True,
            "dev_records": False,
            "anomaly_images": False,
            "masks": False,
            "sealed": False,
            "writer_stream": False,
        },
        "source_sha256": {
            "exps/npim_cad/uncertainty/math_contract.py": sha256_file(ROOT / "exps/npim_cad/uncertainty/math_contract.py"),
            "exps/npim_cad/uncertainty/moments.py": sha256_file(ROOT / "exps/npim_cad/uncertainty/moments.py"),
            "exps/npim_cad/uncertainty/cohort.py": sha256_file(ROOT / "exps/npim_cad/uncertainty/cohort.py"),
            "exps/npim_cad/uncertainty/readout.py": sha256_file(ROOT / "exps/npim_cad/uncertainty/readout.py"),
            "exps/npim_cad/uncertainty/calibrate.py": sha256_file(ROOT / "exps/npim_cad/uncertainty/calibrate.py"),
        },
    }
    write_json_exclusive(run_dir / "manifest.json", manifest)
    write_json_exclusive(run_dir / "E_CALIB_SEALED.json", {
        "schema": "npim03_calib_seal_v1",
        "created_utc": utc_now(),
        "stats_sha256": stats_sha,
        "analysis_sha256": sha256_file(run_dir / "analysis.json"),
        "ledger_sha256": sha256_file(run_dir / "ledger.json"),
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "primary_decision": primary_decision,
    })
    return {
        "run_dir": str(run_dir),
        "primary_decision": primary_decision,
        "primary_reason": reason,
        "stats_sha256": stats_sha,
        "resource": resource_info,
        "ledger": ledger,
    }
