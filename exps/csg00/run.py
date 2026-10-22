"""CSG-00 G2 execution: bounded normal-only geometry feasibility.

Usage (after G0 lock and G1 tests):

    python -m exps.csg00.run verify
    python -m exps.csg00.run lock
    python -m exps.csg00.run parity --device cuda
    python -m exps.csg00.run experiment --device cpu --tag cpu
    python -m exps.csg00.run experiment --device cuda --tag cuda
    python -m exps.csg00.run seal

No anomaly, test, mask, consumed-DEV or sealed identity is opened.  The only
data reads are the frozen normal feature cache and normal train/good identities.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from . import contract as C
from . import features, fit_modes, g0, geometry_diagnostics as D, reservoir, scoring


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def configure_torch(device: str) -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


def write_json(path: Path, obj: Any, exclusive: bool = True) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean_json(obj), indent=2, sort_keys=True, allow_nan=False) + "\n"
    mode = "x" if exclusive else "w"
    with open(path, mode, encoding="utf-8") as fh:
        fh.write(text)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": C.sha256_file(path)}


def cuda_memory() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    return {
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "free_bytes": int(torch.cuda.mem_get_info()[0]),
        "total_bytes": int(torch.cuda.mem_get_info()[1]),
    }


class AccessLog:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def record(self, phase: str, row: Dict[str, Any]) -> None:
        self.rows.append({
            "phase": phase,
            "category": row["category"],
            "cache_index": int(row["cache_index"]),
            "relative_path": row["relative_path"],
            "raw_png_sha256": row["raw_png_sha256"],
        })

    def write(self, path: Path) -> Dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x", encoding="utf-8") as fh:
            for row in self.rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        illegal = [r for r in self.rows
                   if r["relative_path"].split("/")[1:3] != ["train", "good"]]
        return {
            "path": str(path),
            "entries": len(self.rows),
            "illegal_entries": len(illegal),
            "sha256": C.sha256_file(path),
            "distinct_identities": len({r["relative_path"] for r in self.rows}),
        }


def _category_groups(eligible: int) -> Dict[str, np.ndarray]:
    groups: Dict[str, np.ndarray] = {}
    start = 0
    for cat, count in C.PROBE_BLOCKS:
        if start >= eligible:
            break
        end = min(start + count, eligible)
        groups[cat] = np.arange(start, end, dtype=np.int64)
        start = end
    return groups


def _state_reload_match(written: Dict[str, np.ndarray], reloaded: Dict[str, np.ndarray]) -> bool:
    if set(written) != set(reloaded):
        return False
    return all(np.array_equal(written[key], reloaded[key]) for key in written)


def _state_hash_arrays(state: Dict[str, np.ndarray]) -> str:
    import hashlib

    h = hashlib.sha256()
    for key in sorted(state):
        arr = np.ascontiguousarray(state[key])
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Feature parity (GPU preflight)
# ---------------------------------------------------------------------------


def cmd_parity(args: argparse.Namespace) -> int:
    g0.verify_lock()
    device = args.device
    configure_torch(device)
    stream = C.load_stream()
    cache = features.load_feature_cache(verify_hashes=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = features.feature_parity_check(cache, stream["updates"], device=device)
    elapsed = time.perf_counter() - start
    result["elapsed_seconds"] = elapsed
    result["cuda_memory"] = cuda_memory()
    result["checkpoint_bytes"] = C.CHECKPOINT.stat().st_size
    result["checkpoint_sha256"] = C.sha256_file(C.CHECKPOINT)
    result["no_anomaly_attestation"] = "four normal train/good writer images only"
    info = write_json(C.RUN_DIR / "feature_parity.json", result, exclusive=True)
    print(json.dumps({"all_ok": result["all_ok"], "bytes": info["bytes"],
                      "sha256": info["sha256"], "elapsed_seconds": elapsed}, indent=2))
    return 0 if result["all_ok"] else 3


# ---------------------------------------------------------------------------
# One boundary
# ---------------------------------------------------------------------------


def run_boundary(t: int, cache, stream, positions, out_dir: Path, device: str,
                 skip_global: bool, access: AccessLog,
                 determinism_check: bool = False) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    boundary_start = time.perf_counter()
    updates = stream["updates"]
    eligible = C.ELIGIBLE_PREFIX[t]
    category_of_event = [row["category"] for row in updates[:t]]

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    phase = time.perf_counter()
    writer = features.writer_tensor(cache, updates, t)
    writer_bins = features.descriptor_bins(writer, positions)
    del writer  # bin views are copies; release the full writer tensor
    timings["load_writer"] = time.perf_counter() - phase

    phase = time.perf_counter()
    fit = fit_modes.fit_all_bins(writer_bins, t, category_of_event)
    b_state = fit["state"]
    timings["fit_B"] = time.perf_counter() - phase

    determinism = None
    if determinism_check:
        phase = time.perf_counter()
        fit2 = fit_modes.fit_all_bins(writer_bins, t, category_of_event)
        determinism = {
            "refit_state_equal": _state_hash_arrays(fit["state"]) == _state_hash_arrays(fit2["state"]),
            "refit_diagnostics_equal": fit["bins"] == fit2["bins"],
            "refit_seconds": time.perf_counter() - phase,
        }
        if not determinism["refit_state_equal"]:
            raise RuntimeError("deterministic refit failed")

    b_path = out_dir / "states" / f"b_t{t}.csg00"
    b_info = C.write_state(b_path, "B", t, C.protocol_manifest()["arms"]["B"],
                           b_state, C.B_STATE_LAYOUT, exclusive=True)
    _, b_reloaded = C.read_state(b_path)
    reload_b = _state_reload_match(b_state, b_reloaded)
    if not reload_b:
        raise RuntimeError("arm B save/reload parity failed")

    phase = time.perf_counter()
    c_state = reservoir.build_reservoir(writer_bins, t, C.RESERVOIR_CAPACITY, C.RESERVOIR_SEED)
    timings["reservoir_C"] = time.perf_counter() - phase
    c_path = out_dir / "states" / f"c_t{t}.csg00"
    c_info = C.write_state(c_path, "C", t,
                           {"capacity": C.RESERVOIR_CAPACITY, "seed": C.RESERVOIR_SEED,
                            "rng": C.RESERVOIR_RULE},
                           c_state, C.C_STATE_LAYOUT, exclusive=True)
    _, c_reloaded = C.read_state(c_path)
    reload_c = _state_reload_match(c_state, c_reloaded)
    if not reload_c:
        raise RuntimeError("arm C save/reload parity failed")
    byte_ratio = c_info["bytes"] / b_info["bytes"]

    for row in updates[:t]:
        access.record("G2_writer", row)
    for row in stream["probes"][:eligible]:
        access.record("G2_witness", row)

    phase = time.perf_counter()
    witness = features.witness_tensor(cache, stream["probes"])[:eligible]
    queries = scoring.per_bin_queries(witness, positions)
    timings["load_witness_queries"] = time.perf_counter() - phase

    phase = time.perf_counter()
    A = scoring.assemble(scoring.score_binned(queries, writer_bins, device=device),
                         eligible, positions)
    timings["score_A"] = time.perf_counter() - phase
    phase = time.perf_counter()
    B = scoring.assemble(scoring.score_modes(queries, b_reloaded, device=device),
                         eligible, positions)
    timings["score_B"] = time.perf_counter() - phase
    phase = time.perf_counter()
    Cscore = scoring.assemble(scoring.score_reservoir(queries, c_reloaded, device=device),
                              eligible, positions)
    timings["score_C"] = time.perf_counter() - phase

    global_scores = None
    if not skip_global:
        phase = time.perf_counter()
        global_scores = scoring.assemble(scoring.score_global(queries, writer_bins, device=device),
                                         eligible, positions)
        timings["score_A_global_diagnostic"] = time.perf_counter() - phase

    scores = {"A": A.numpy(), "B": B.numpy(), "C": Cscore.numpy()}
    if global_scores is not None:
        scores["A_global"] = global_scores.numpy()
    groups = _category_groups(eligible)
    scores_path = out_dir / "scores" / f"scores_t{t}.npz"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    if scores_path.exists():
        raise FileExistsError(f"refusing overwrite: {scores_path}")
    # Store float32 score maps (analysis/parity artifacts).  The canonical metrics
    # above are computed from the in-memory float64 scores before this cast.
    np.savez(
        scores_path,
        A=scores["A"].astype(np.float32), B=scores["B"].astype(np.float32),
        C=scores["C"].astype(np.float32),
        imgmax_A=scores["A"].max(axis=1), imgmax_B=scores["B"].max(axis=1),
        imgmax_C=scores["C"].max(axis=1),
        imgmax_A_global=(scores["A_global"].max(axis=1) if "A_global" in scores
                         else np.zeros((0,))),
        categories=np.asarray([cat for cat, _ in C.PROBE_BLOCKS if cat in groups]),
    )

    phase = time.perf_counter()
    patch = {arm: D.patch_summary(scores[arm]) for arm in ("A", "B", "C")}
    patch_by_category = {
        arm: {cat: D.patch_summary(scores[arm][idx]) for cat, idx in groups.items()}
        for arm in ("A", "B", "C")
    }
    image = {arm: D._summarize(scores[arm].max(axis=1)) for arm in ("A", "B", "C")}
    image_by_category = {
        arm: {cat: D._summarize(scores[arm][idx].max(axis=1)) for cat, idx in groups.items()}
        for arm in ("A", "B", "C")
    }
    coverage = {
        "B": {cat: float(np.mean(scores["B"][idx] <= scores["A"][idx])) for cat, idx in groups.items()},
        "C": {cat: float(np.mean(scores["C"][idx] <= scores["A"][idx])) for cat, idx in groups.items()},
    }
    timings["summaries"] = time.perf_counter() - phase

    phase = time.perf_counter()
    bootstrap = D.bootstrap_selection(scores, groups, C.BOOTSTRAP_REPLICATES, t)
    per_category = {
        cat: D.bootstrap_single_category(scores, idx, C.BOOTSTRAP_REPLICATES, t, cat)
        for cat, idx in groups.items()
    }
    timings["bootstrap"] = time.perf_counter() - phase

    phase = time.perf_counter()
    occupancy = D.mode_occupancy_summary(fit["bins"])
    utilization = D.mode_utilization(queries, b_reloaded)
    timings["mode_diagnostics"] = time.perf_counter() - phase

    phase = time.perf_counter()
    gap = D.gap_probe_summary(writer_bins, fit["probe_index"], b_reloaded, c_reloaded,
                              scores["A"], positions, t)
    support = D.support_sample_summary(writer_bins, b_reloaded, scores["A"], positions, t)
    timings["gap_and_overreach"] = time.perf_counter() - phase

    global_diag = None
    if global_scores is not None:
        g_img = global_scores.numpy().max(axis=1)
        a_img = scores["A"].max(axis=1)
        global_diag = {
            "definition": "report-only bin-unrestricted NN; never a primary arm",
            "image_max_spearman_vs_A_all": D.spearman(a_img, g_img),
            "image_max_spearman_vs_A_by_category": {
                cat: D.spearman(a_img[idx], g_img[idx]) for cat, idx in groups.items()
            },
            "patch_mean_A": float(scores["A"].mean()),
            "patch_mean_global": float(global_scores.numpy().mean()),
            "patch_median_ratio_global_over_A": float(np.median(
                global_scores.numpy().reshape(-1) / np.maximum(scores["A"].reshape(-1), 1e-12))),
            "fraction_global_strictly_below_A": float(
                np.mean(global_scores.numpy() < scores["A"] - 1e-12)),
            "image_max": D._summarize(g_img),
        }

    if determinism_check:
        phase = time.perf_counter()
        B2 = scoring.assemble(scoring.score_modes(queries, b_reloaded, device=device),
                              eligible, positions).numpy()
        determinism["rescore_equal"] = bool(np.array_equal(B, B2))
        determinism["rescore_seconds"] = time.perf_counter() - phase
        if not determinism["rescore_equal"]:
            raise RuntimeError("deterministic rescoring failed")

    timings["boundary_total"] = time.perf_counter() - boundary_start
    result = {
        "boundary": int(t),
        "n_writer_events": int(t),
        "n_images": int(eligible),
        "categories": list(groups.keys()),
        "category_groups": {cat: [int(idx[0]), int(idx[-1]) + 1] for cat, idx in groups.items()},
        "patch": patch,
        "patch_by_category": patch_by_category,
        "image_scores": image,
        "image_scores_by_category": image_by_category,
        "coverage_vs_A": coverage,
        "bootstrap_pooled": bootstrap,
        "bootstrap_by_category": per_category,
        "mode": {"occupancy": occupancy, "utilization": utilization, "bins": fit["bins"]},
        "gap": gap,
        "support_overreach": support,
        "global_diagnostic": global_diag,
        "state": {
            "B": b_info,
            "C": c_info,
            "reload_match_B": reload_b,
            "reload_match_C": reload_c,
            "byte_ratio_C_over_B": float(byte_ratio),
            "byte_ratio_within_5pct": bool(0.95 <= byte_ratio <= 1.05),
        },
        "determinism": determinism,
        "timings_seconds": timings,
        "cuda_memory": cuda_memory(),
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    info = write_json(out_dir / f"boundary_t{t}.json", result, exclusive=True)
    result["artifact"] = info

    # free large structures before the next boundary
    del writer_bins, queries, A, B, Cscore, global_scores
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Drift and parity
# ---------------------------------------------------------------------------


def drift_summary(imgmax: Dict[str, Dict[int, np.ndarray]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for arm in ("A", "B", "C"):
        per_arm: Dict[str, Any] = {}
        boundaries = sorted(imgmax[arm].keys())
        for i in range(len(boundaries) - 1):
            t1, t2 = boundaries[i], boundaries[i + 1]
            left, right = imgmax[arm][t1], imgmax[arm][t2]
            n = min(left.size, right.size)
            common = np.arange(n)
            per_arm[f"{t1}->{t2}"] = {
                "n_images": int(n),
                "spearman": D.spearman(left[common], right[common]),
                "mean_abs_change": float(np.mean(np.abs(left[common] - right[common]))),
                "relative_mean_abs_change": float(
                    np.mean(np.abs(left[common] - right[common])) / max(np.mean(right[common]), 1e-12)),
            }
        out[arm] = per_arm
    return out


def parity_report(cpu_dir: Path, cuda_dir: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"boundaries": {}}
    for t in C.BOUNDARIES:
        cpu_npz = np.load(cpu_dir / "scores" / f"scores_t{t}.npz")
        cuda_npz = np.load(cuda_dir / "scores" / f"scores_t{t}.npz")
        entry: Dict[str, Any] = {}
        for arm in ("A", "B", "C"):
            a = cpu_npz[arm]
            b = cuda_npz[arm]
            if a.shape != b.shape:
                entry[arm] = {"shape_match": False, "cpu_shape": a.shape, "cuda_shape": b.shape}
                continue
            diff = np.abs(a - b)
            entry[arm] = {
                "shape_match": True,
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "p99_abs_diff": float(np.quantile(diff, 0.99)),
                "image_max_spearman": D.spearman(a.max(axis=1), b.max(axis=1)),
                "bitwise_equal": bool(np.array_equal(a, b)),
            }
        payload["boundaries"][str(t)] = entry
    # state files must be byte-identical (fitting and reservoir are CPU deterministic)
    state_parity = {}
    for t in C.BOUNDARIES:
        for arm, prefix in (("B", "b"), ("C", "c")):
            p1 = cpu_dir / "states" / f"{prefix}_t{t}.csg00"
            p2 = cuda_dir / "states" / f"{prefix}_t{t}.csg00"
            state_parity[f"{arm}_t{t}"] = {
                "sha256_cpu": C.sha256_file(p1),
                "sha256_cuda": C.sha256_file(p2),
                "identical": C.sha256_file(p1) == C.sha256_file(p2),
            }
    payload["state_files"] = state_parity
    worst = max(
        entry[arm]["max_abs_diff"]
        for entry in payload["boundaries"].values() for arm in ("A", "B", "C")
        if entry[arm].get("shape_match")
    )
    payload["worst_arm_max_abs_diff"] = float(worst)
    payload["tolerance"] = 1.0e-3
    payload["passed"] = bool(worst <= 1.0e-3 and all(
        v["identical"] for v in state_parity.values()))
    return payload


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def decide(metrics: Dict[str, Any], parity: Dict[str, Any]) -> Dict[str, Any]:
    t = 350
    raw_boundaries = metrics["boundaries"]
    if isinstance(raw_boundaries, list):
        boundaries = {str(int(entry["boundary"])): entry for entry in raw_boundaries}
    else:
        boundaries = {str(k): v for k, v in raw_boundaries.items()}
    metrics = dict(metrics)
    metrics["boundaries"] = boundaries
    boundary = boundaries[str(t)]
    boot = boundary["bootstrap_pooled"]["bootstrap"]
    delta_cov = boot["delta_cov"]
    delta_rho = boot["delta_rho"]
    ovl_b = float(boundary["bootstrap_pooled"]["macro_point_estimate"]["zero_fraction_B"])

    gfa = {}
    n_accepted = 0
    n_high = 0
    for key, entry in metrics["boundaries"].items():
        gfa[key] = entry["gap"]["gfa_B"]
        n_accepted += int(entry["gap"]["n_B_accepted"])
        n_high += int(entry["gap"]["n_high_A"])
    gfa_pooled = (n_accepted / n_high) if n_high else None
    gfa_max = max((v for v in gfa.values() if v is not None), default=None)

    degenerate = {}
    for cat in ("bottle", "carpet", "hazelnut"):
        vals = boundary["bootstrap_by_category"].get(cat, {}).get("point_estimate", {}).get("image_score_B", {})
        if not vals:
            continue
        degenerate[cat] = bool(abs(float(vals.get("max", 0.0)) - float(vals.get("min", 0.0))) < 1e-15)

    integrity_ok = bool(
        parity.get("passed", False)
        and all(boundary["state"]["byte_ratio_within_5pct"] for boundary in metrics["boundaries"].values())
        and all(entry["state"]["reload_match_B"] and entry["state"]["reload_match_C"]
                for entry in metrics["boundaries"].values())
        and not any(v is None for v in gfa.values())
    )

    ci_cov_lo, ci_cov_hi = delta_cov["ci"]
    ci_rho_lo, ci_rho_hi = delta_rho["ci"]
    undefined_primary = any(v is None for v in (ci_cov_lo, ci_cov_hi, ci_rho_lo, ci_rho_hi))

    checks = {
        "integrity_ok": integrity_ok,
        "delta_cov": delta_cov,
        "delta_rho": delta_rho,
        "gfa_by_boundary": gfa,
        "gfa_pooled": gfa_pooled,
        "gfa_max_boundary": gfa_max,
        "zero_fraction_B_t350": ovl_b,
        "degenerate_arm_B_image_scores": degenerate,
        "parity_worst_abs_diff": parity.get("worst_arm_max_abs_diff"),
    }

    if not integrity_ok:
        return {"decision": "CSG00_BLOCKED_INTEGRITY", "checks": checks,
                "reasons": ["integrity, parity, serialization or gap-diagnostic failure"]}
    if undefined_primary:
        return {"decision": "CSG00_BLOCKED_UNDEFINED_PRIMARY", "checks": checks,
                "reasons": ["primary bootstrap endpoint undefined (constant rankings or no high-A gap probes)"]}

    go = (
        ci_cov_lo > 0.05
        and ci_rho_lo > -0.05
        and gfa_pooled is not None and gfa_pooled <= 0.25
        and gfa_max is not None and gfa_max <= 0.40
        and ovl_b <= 0.95
        and not any(degenerate.values())
    )
    negative = (
        (gfa_pooled is not None and gfa_pooled > 0.25)
        or (gfa_max is not None and gfa_max > 0.40)
        or ovl_b > 0.95
        or any(degenerate.values())
        or ci_cov_hi < 0.05
        or ci_rho_hi < -0.10
    )
    if go:
        decision = "CSG00_GO_FRESH_ANOMALY_GATE_REQUESTED"
        reasons = ["coverage advantage over C with CI lower > 0.05",
                   "rank fidelity not materially worse than C",
                   "gap overreach below pre-registered caps"]
    elif negative:
        decision = "CSG00_NEGATIVE_GEOMETRY"
        reasons = []
        if gfa_pooled is not None and gfa_pooled > 0.25:
            reasons.append(f"pooled gap false-acceptance {gfa_pooled:.3f} > 0.25")
        if gfa_max is not None and gfa_max > 0.40:
            reasons.append(f"boundary gap false-acceptance {gfa_max:.3f} > 0.40")
        if ovl_b > 0.95:
            reasons.append(f"arm-B zero-score saturation {ovl_b:.3f} > 0.95")
        if any(degenerate.values()):
            reasons.append("arm-B image scores constant in at least one category")
        if ci_cov_hi < 0.05:
            reasons.append(f"coverage advantage CI upper {ci_cov_hi:.3f} < 0.05")
        if ci_rho_hi < -0.10:
            reasons.append(f"rank-fidelity delta CI upper {ci_rho_hi:.3f} < -0.10")
    else:
        decision = "CSG00_INCONCLUSIVE"
        reasons = ["neither GO nor NEGATIVE bands were met"]
    return {"decision": decision, "checks": checks, "reasons": reasons}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    result = g0.verify_all()
    print(json.dumps(clean_json({
        "agents_md": result["agents_md"],
        "checkpoint": result["checkpoint"],
        "p0c_protocol": result["p0c_protocol"],
        "stream": {k: result["stream"][k] for k in
                   ("writer_events", "witnesses", "distinct_writer_paths",
                    "distinct_writer_contents", "stream_digest", "witness_digest")},
        "normal_only_guard": result["normal_only_guard"],
        "cache_rows": result["cache_rows"]["checked_rows"],
        "byte_budget_dry_run": result["byte_budget_dry_run"],
        "environment": result["environment"],
    }), indent=2))
    return 0


def cmd_lock(args: argparse.Namespace) -> int:
    info = g0.write_lock()
    print(json.dumps(info, indent=2))
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    lock_info = g0.verify_lock()
    configure_torch(args.device)
    out_dir = C.RUN_DIR / args.tag
    if out_dir.exists():
        raise FileExistsError(f"run tag already exists (refusing overwrite): {out_dir}")
    out_dir.mkdir(parents=True)
    stream = C.load_stream()
    cache = features.load_feature_cache(verify_hashes=True)
    positions = C.bin_positions()
    access = AccessLog()
    started = time.perf_counter()
    boundary_results = []
    imgmax: Dict[str, Dict[int, np.ndarray]] = {arm: {} for arm in ("A", "B", "C")}
    for t in C.BOUNDARIES:
        print(f"[csg00] boundary {t} on {args.device}", flush=True)
        result = run_boundary(t, cache, stream, positions, out_dir, args.device,
                              args.skip_global, access, determinism_check=(t == 350))
        boundary_results.append(result)
        npz = np.load(out_dir / "scores" / f"scores_t{t}.npz")
        for arm in ("A", "B", "C"):
            imgmax[arm][t] = npz[f"imgmax_{arm}"]
    wall = time.perf_counter() - started
    access_info = access.write(out_dir / "access_log.jsonl")
    drift = drift_summary(imgmax)
    resource_ledger = {
        "device": args.device,
        "wall_seconds": wall,
        "boundary_timings": {str(r["boundary"]): r["timings_seconds"] for r in boundary_results},
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cuda_memory": cuda_memory(),
        "environment": g0.environment_info(probe_gpu=True),
        "checkpoint_bytes": C.CHECKPOINT.stat().st_size,
        "analysis_only_feature_cache_bytes": int(sum(
            (C.CACHE_DIR / f"{c}_train.pt").stat().st_size for c in C.CATEGORIES)),
        "scores_artifact_bytes": int(sum(
            p.stat().st_size for p in (out_dir / "scores").glob("*.npz"))),
        "state_bytes": {str(r["boundary"]): {"B": r["state"]["B"]["bytes"],
                                             "C": r["state"]["C"]["bytes"]}
                        for r in boundary_results},
        "optimizer_bytes": 0,
        "access_log": access_info,
        "no_anomaly_attestation": (
            "only MVTec AD v1 train/good normal identities and the frozen normal "
            "feature cache were read; no test/mask/anomaly/consumed-DEV/sealed payload"),
    }
    byte_ledger = {
        "dry_run_schema": C.reservoir_capacity_for_byte_match(),
        "b_payload_bytes_declared": C.b_payload_bytes(),
        "c_payload_bytes_declared": C.c_payload_bytes(),
        "actual": {str(r["boundary"]): {
            "B_bytes": r["state"]["B"]["bytes"],
            "B_payload_bytes": r["state"]["B"]["payload_bytes"],
            "B_header_bytes": r["state"]["B"]["header_bytes"],
            "C_bytes": r["state"]["C"]["bytes"],
            "C_payload_bytes": r["state"]["C"]["payload_bytes"],
            "C_header_bytes": r["state"]["C"]["header_bytes"],
            "ratio_C_over_B": r["state"]["byte_ratio_C_over_B"],
        } for r in boundary_results},
        "executable_inference_bytes": {
            "checkpoint_bytes": C.CHECKPOINT.stat().st_size,
            "state_B_t350_bytes": boundary_results[-1]["state"]["B"]["bytes"],
            "state_C_t350_bytes": boundary_results[-1]["state"]["C"]["bytes"],
            "csg00_code_bytes": int(sum(p.stat().st_size for p in (C.ROOT / "exps" / "csg00").glob("*.py"))),
            "optimizer_bytes": 0,
        },
        "analysis_only_not_part_of_detector": {
            "feature_cache_bytes": int(sum(
                (C.CACHE_DIR / f"{c}_train.pt").stat().st_size for c in C.CATEGORIES)),
            "note": "the full-support reference A retains normal descriptors in RAM by design; "
                    "the compact arm B and control C states are the byte-matched artifacts",
        },
    }
    summary = {
        "run_id": C.RUN_ID,
        "tag": args.tag,
        "device": args.device,
        "lock": lock_info,
        "boundaries": boundary_results,
        "drift": drift,
        "resource_ledger": resource_ledger,
        "byte_ledger": byte_ledger,
    }
    info = write_json(out_dir / "boundary_metrics.json", summary, exclusive=True)
    print(json.dumps({"tag": args.tag, "device": args.device,
                      "artifact": info, "wall_seconds": wall}, indent=2))
    return 0


def cmd_seal(args: argparse.Namespace) -> int:
    g0.verify_lock()
    if C.G2_SEAL_PATH.exists():
        raise FileExistsError(f"G2 seal already exists: {C.G2_SEAL_PATH}")
    cpu_dir = C.RUN_DIR / args.cpu
    cuda_dir = C.RUN_DIR / args.cuda
    cpu_metrics = json.loads((cpu_dir / "boundary_metrics.json").read_text())
    cuda_metrics = json.loads((cuda_dir / "boundary_metrics.json").read_text())
    parity = parity_report(cpu_dir, cuda_dir)
    decision = decide(cpu_metrics, parity)
    combined = {
        "run_id": C.RUN_ID,
        "canonical_run_tag": args.cpu,
        "execution_run_tag": args.cuda,
        "protocol": C.protocol_manifest(),
        "decision": decision,
        "parity": parity,
        "cpu_metrics": cpu_metrics,
        "cuda_metrics": cuda_metrics,
        "no_anomaly_attestation": (
            "G2 read only MVTec AD v1 train/good normal identities and the frozen normal "
            "feature cache. No test image, ground-truth mask, anomaly label, consumed DEV "
            "cohort or sealed confirmation identity was opened, hashed or decoded."
        ),
        "limitations": [
            "normal-support geometry only; no Pixel AP, AUROC or anomaly efficacy is measured",
            "arm A is an empirical geometry reference, not a Pixel-AP upper bound",
            "arm B is fitted offline from the causal prefix; this is not online continual learning",
            "79 normal-only witnesses were historically exposed in P0c/P1; they are not a fresh confirmation set",
            "one stream, one order, one seed; no multi-seed uncertainty",
            "finite gap probes give descriptive evidence, not a Hausdorff certificate",
        ],
    }
    info = write_json(C.RUN_DIR / "G2_METRICS.json", combined, exclusive=True)
    # hash every artifact in the run directory except the seal itself
    artifacts = {}
    for path in sorted(C.RUN_DIR.rglob("*")):
        if path.is_file() and path != C.G2_SEAL_PATH:
            artifacts[str(path.relative_to(C.RUN_DIR))] = {
                "bytes": path.stat().st_size, "sha256": C.sha256_file(path)
            }
    seal = {
        "schema": "csg00_g2_seal_v1",
        "run_id": C.RUN_ID,
        "sealed_utc": C.utc_now(),
        "decision": decision["decision"],
        "decision_detail": decision,
        "git": g0.git_state(),
        "environment": g0.environment_info(probe_gpu=True),
        "artifacts": artifacts,
        "metrics_artifact": info,
        "no_anomaly_attestation": combined["no_anomaly_attestation"],
        "historical_hashes_verified": True,
    }
    with open(C.G2_SEAL_PATH, "x", encoding="utf-8") as fh:
        json.dump(clean_json(seal), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    print(json.dumps({"seal": str(C.G2_SEAL_PATH), "sha256": C.sha256_file(C.G2_SEAL_PATH),
                      "decision": decision["decision"], "reasons": decision["reasons"]}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSG-00 bounded normal-only geometry feasibility")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    sub.add_parser("lock")
    p_parity = sub.add_parser("parity")
    p_parity.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    p_run = sub.add_parser("experiment")
    p_run.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p_run.add_argument("--tag", required=True)
    p_run.add_argument("--skip-global", action="store_true")
    p_seal = sub.add_parser("seal")
    p_seal.add_argument("--cpu", default="cpu")
    p_seal.add_argument("--cuda", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "verify":
        return cmd_verify(args)
    if args.command == "lock":
        return cmd_lock(args)
    if args.command == "parity":
        return cmd_parity(args)
    if args.command == "experiment":
        return cmd_experiment(args)
    if args.command == "seal":
        return cmd_seal(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    sys.exit(main())
