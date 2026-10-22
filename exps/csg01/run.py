"""CSG-01 G2 execution: distance-preserving quantized normal-support memory.

Commands (after G0 lock and G1 tests):

    python -m exps.csg01.run verify
    python -m exps.csg01.run lock
    python -m exps.csg01.run parity --device cuda
    python -m exps.csg01.run experiment --device cpu --tag cpu
    python -m exps.csg01.run experiment --device cuda --tag cuda
    python -m exps.csg01.run seal

Only normal MVTec AD v1 train/good identities and the frozen normal feature
cache are read.  G3 is not executed.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from exps.csg00 import contract as C0
from exps.csg00 import features as F0
from exps.csg00 import scoring as S0

from . import contract as C1
from . import diagnostics, g0, quantize, scoring, select


def configure_torch(device: str) -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
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
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def write_json(path: Path, obj: Any, exclusive: bool = True) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean_json(obj), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with open(path, "x" if exclusive else "w", encoding="utf-8") as fh:
        fh.write(text)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": C0.sha256_file(path)}


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
        return {"path": str(path), "entries": len(self.rows),
                "illegal_entries": len(illegal),
                "distinct_identities": len({r["relative_path"] for r in self.rows}),
                "sha256": C0.sha256_file(path)}


def _category_groups(eligible: int) -> Dict[str, np.ndarray]:
    groups: Dict[str, np.ndarray] = {}
    start = 0
    for cat, count in C1.PROBE_BLOCKS:
        if start >= eligible:
            break
        end = min(start + count, eligible)
        groups[cat] = np.arange(start, end, dtype=np.int64)
        start = end
    return groups


def _state_reload_match(written: Dict[str, np.ndarray], reloaded: Dict[str, np.ndarray]) -> bool:
    return set(written) == set(reloaded) and all(
        np.array_equal(np.asarray(written[k]), np.asarray(reloaded[k])) for k in written)


def _selection_digest(selection: Dict[str, Any]) -> str:
    return C0.canonical_digest([
        {"bin": r["bin"], "global_id": r["global_id"], "local_index": r["local_index"]}
        for r in selection["provenance"]])


def _write_provenance(path: Path, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return {"path": str(path), "rows": len(rows), "sha256": C0.sha256_file(path),
            "bytes": path.stat().st_size}


def _selection_summary(selection: Dict[str, Any], other: Dict[str, Any] | None = None) -> Dict[str, Any]:
    gids = select.selected_global_ids(selection)
    out = {
        "policy": selection["policy"],
        "n_selected": int(selection["n_selected"]),
        "n_unique_global_ids": int(len(set(gids.tolist()))),
        "quotas": selection["quotas"].tolist(),
        "all_events_causal": bool(selection["all_events_causal"]),
        "max_event": int(max(r["event"] for r in selection["provenance"])),
        "elapsed_seconds": float(selection["elapsed_seconds"]),
        "digest": _selection_digest(selection),
    }
    if other is not None:
        og = select.selected_global_ids(other)
        out["overlap_with_other"] = int(len(set(gids.tolist()) & set(og.tolist())))
        out["identical_to_other"] = bool(np.array_equal(gids, og))
    return out


def run_boundary(t: int, cache, stream, positions, out_dir: Path, device: str,
                 access: AccessLog, determinism_check: bool = False) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    started = time.perf_counter()
    updates = stream["updates"]
    eligible = C1.ELIGIBLE_PREFIX[t]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    phase = time.perf_counter()
    writer = F0.writer_tensor(cache, updates, t)
    writer_bins = F0.descriptor_bins(writer, positions)
    del writer
    timings["load_writer"] = time.perf_counter() - phase

    quotas_b = C1.quota_vector(C1.SELECTION_COUNT)
    quotas_e = C1.quota_vector(C1.SIZE_MATCHED_COUNT)

    phase = time.perf_counter()
    sel_b = select.build_selection(writer_bins, t, quotas_b, "fps",
                                   updates=updates, positions=positions)
    timings["fps_B"] = time.perf_counter() - phase
    phase = time.perf_counter()
    sel_d = select.build_selection(writer_bins, t, quotas_b, "random",
                                   seed=C1.RANDOM_SEED, updates=updates, positions=positions)
    timings["random_D"] = time.perf_counter() - phase
    phase = time.perf_counter()
    sel_e = select.build_selection(writer_bins, t, quotas_e, "fps",
                                   updates=updates, positions=positions)
    timings["fps_E"] = time.perf_counter() - phase

    support_b = select.take_selected(writer_bins, sel_b)
    support_d = select.take_selected(writer_bins, sel_d)
    support_e = select.take_selected(writer_bins, sel_e)
    state_b = select.pack_support(support_b)
    state_e = select.pack_support(support_e)

    phase = time.perf_counter()
    encoded_c = quantize.encode_support(support_b)
    encoded_d = quantize.encode_support(support_d)
    quant_c = quantize.quantization_summary(encoded_c, state_b["counts"])
    quant_d = quantize.quantization_summary(encoded_d, select.pack_support(support_d)["counts"])
    delta_c = quantize.delta_by_bin(encoded_c, state_b["counts"])
    delta_d = quantize.delta_by_bin(encoded_d, select.pack_support(support_d)["counts"])
    timings["quantize_C_D"] = time.perf_counter() - phase

    counts_c = np.asarray(state_b["counts"], dtype=np.int32)
    offsets_c = np.asarray(state_b["offsets"], dtype=np.int32)
    state_c = {"counts": counts_c, "offsets": offsets_c, **quantize.pack_quantized(encoded_c)}
    state_d = {"counts": counts_c, "offsets": offsets_c, **quantize.pack_quantized(encoded_d)}

    phase = time.perf_counter()
    b_path = out_dir / "states" / f"b_t{t}.csg00"
    c_path = out_dir / "states" / f"c_t{t}.csg00"
    d_path = out_dir / "states" / f"d_t{t}.csg00"
    e_path = out_dir / "states" / f"e_t{t}.csg00"
    b_info = C1.write_state(b_path, "B", t, C1.arms_params()["B"], state_b, exclusive=True)
    c_info = C1.write_state(c_path, "C", t, C1.arms_params()["C"], state_c, exclusive=True)
    d_info = C1.write_state(d_path, "D", t, C1.arms_params()["D"], state_d, exclusive=True)
    e_info = C1.write_state(e_path, "E", t, C1.arms_params()["E"], state_e, exclusive=True)
    _, b_reload = C1.read_state(b_path)
    _, c_reload = C1.read_state(c_path)
    _, d_reload = C1.read_state(d_path)
    _, e_reload = C1.read_state(e_path)
    reload = {
        "B": _state_reload_match(state_b, b_reload),
        "C": _state_reload_match(state_c, c_reload),
        "D": _state_reload_match(state_d, d_reload),
        "E": _state_reload_match(state_e, e_reload),
    }
    if not all(reload.values()):
        raise RuntimeError(f"state reload parity failed: {reload}")
    timings["write_reload_states"] = time.perf_counter() - phase

    support_b_r = scoring.support_bins_from_state(b_reload, "B")
    support_c_r = scoring.support_bins_from_state(c_reload, "C")
    support_d_r = scoring.support_bins_from_state(d_reload, "D")
    support_e_r = scoring.support_bins_from_state(e_reload, "E")

    phase = time.perf_counter()
    witness = F0.witness_tensor(cache, stream["probes"])[:eligible]
    queries = S0.per_bin_queries(witness, positions)
    timings["load_witness_queries"] = time.perf_counter() - phase

    phase = time.perf_counter()
    scores = {"A": scoring.score_arm(queries, writer_bins, eligible, positions, device=device)}
    timings["score_A"] = time.perf_counter() - phase
    for arm, support in (("B", support_b_r), ("C", support_c_r),
                         ("D", support_d_r), ("E", support_e_r)):
        phase = time.perf_counter()
        scores[arm] = scoring.score_arm(queries, support, eligible, positions, device=device)
        timings[f"score_{arm}"] = time.perf_counter() - phase
    scores_np = {arm: scores[arm].numpy() for arm in scores}

    groups = _category_groups(eligible)
    phase = time.perf_counter()
    eps = {
        "A": np.zeros((C1.BINS,), dtype=np.float64),
        "B": scoring.covering_radius(writer_bins, support_b_r),
        "C": scoring.covering_radius(writer_bins, support_c_r),
        "D": scoring.covering_radius(writer_bins, support_d_r),
        "E": scoring.covering_radius(writer_bins, support_e_r),
    }
    timings["covering_radii"] = time.perf_counter() - phase

    phase = time.perf_counter()
    point = diagnostics.point_estimates(scores_np, groups)
    reps = C1.BOOTSTRAP_REPLICATES_T350 if t == 350 else C1.BOOTSTRAP_REPLICATES_OTHER
    boot = diagnostics.bootstrap_metrics(scores_np, groups, reps, t)
    timings["bootstrap"] = time.perf_counter() - phase

    phase = time.perf_counter()
    cert = diagnostics.certificate_checks(
        scores_np, {"B": eps["B"], "C": eps["C"], "D": eps["D"], "E": eps["E"]},
        {"C": delta_c, "D": delta_d})
    image_cert = diagnostics.image_max_certificates(
        scores_np, {"B": eps["B"], "C": eps["C"], "D": eps["D"], "E": eps["E"]},
        {"C": delta_c, "D": delta_d})
    tail = diagnostics.quantization_tail_error(scores_np)
    timings["certificates"] = time.perf_counter() - phase

    phase = time.perf_counter()
    scores_path = out_dir / "scores" / f"scores_t{t}.npz"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    if scores_path.exists():
        raise FileExistsError(scores_path)
    np.savez(
        scores_path,
        **{arm: scores_np[arm].astype(np.float32) for arm in "ABCDE"},
        **{f"imgmax_{arm}": scores_np[arm].max(axis=1) for arm in "ABCDE"},
        categories=np.asarray(list(groups.keys())),
    )
    timings["save_scores"] = time.perf_counter() - phase

    phase = time.perf_counter()
    manifests = {}
    for arm, sel in (("B", sel_b), ("C", sel_b), ("D", sel_d), ("E", sel_e)):
        manifests[arm] = _write_provenance(out_dir / "selections" / f"{arm.lower()}_t{t}.jsonl",
                                           sel["provenance"])
    timings["write_manifests"] = time.perf_counter() - phase

    bytes_ratio = {
        "C_B": c_info["bytes"] / b_info["bytes"],
        "C_D": c_info["bytes"] / d_info["bytes"],
        "C_E": c_info["bytes"] / e_info["bytes"],
        "B_E": b_info["bytes"] / e_info["bytes"],
    }
    selection_summary = {
        "B": _selection_summary(sel_b, sel_d),
        "C": _selection_summary(sel_b, sel_b),
        "D": _selection_summary(sel_d, sel_b),
        "E": _selection_summary(sel_e, sel_b),
    }

    determinism = None
    if determinism_check:
        phase = time.perf_counter()
        sel_b2 = select.build_selection(writer_bins, t, quotas_b, "fps",
                                        updates=updates, positions=positions)
        sel_d2 = select.build_selection(writer_bins, t, quotas_b, "random",
                                        seed=C1.RANDOM_SEED, updates=updates,
                                        positions=positions)
        enc_c2 = quantize.encode_support(select.take_selected(writer_bins, sel_b2))
        determinism = {
            "fps_state_equal": _selection_digest(sel_b) == _selection_digest(sel_b2),
            "random_state_equal": _selection_digest(sel_d) == _selection_digest(sel_d2),
            "quantization_equal": bool(
                np.array_equal(encoded_c["q"], enc_c2["q"]) and
                np.array_equal(encoded_c["xmin"], enc_c2["xmin"]) and
                np.array_equal(encoded_c["step"], enc_c2["step"])),
            "refit_seconds": time.perf_counter() - phase,
        }
        phase = time.perf_counter()
        b2 = scoring.score_arm(queries, support_b_r, eligible, positions, device=device).numpy()
        determinism["rescore_equal"] = bool(np.array_equal(scores_np["B"], b2))
        determinism["rescore_seconds"] = time.perf_counter() - phase
        if not all(v is True for k, v in determinism.items() if k.endswith("_equal")):
            raise RuntimeError(f"determinism failure: {determinism}")

    timings["boundary_total"] = time.perf_counter() - started
    radius_summary = {arm: scoring.radius_summary(eps[arm]) for arm in "ABCDE"}
    result = {
        "boundary": int(t),
        "n_writer_events": int(t),
        "n_images": int(eligible),
        "categories": list(groups.keys()),
        "category_groups": {cat: [int(idx[0]), int(idx[-1]) + 1] for cat, idx in groups.items()},
        "scores_path": str(scores_path),
        "image_max": {arm: {cat: diagnostics._summarize(scores_np[arm][idx].max(axis=1))
                            for cat, idx in groups.items()} for arm in "ABCDE"},
        "patch": {arm: {cat: diagnostics.patch_summary(scores_np[arm][idx])
                        for cat, idx in groups.items()} for arm in "ABCDE"},
        "point": point,
        "bootstrap": boot,
        "covering_radius": {arm: {"per_bin": eps[arm].tolist(), "summary": radius_summary[arm]}
                            for arm in "ABCDE"},
        "quantization": {"C": quant_c, "D": quant_d},
        "delta_by_bin": {"C": delta_c.tolist(), "D": delta_d.tolist()},
        "certificates": cert,
        "image_max_certificates": image_cert,
        "quantization_tail_error": tail,
        "selection": selection_summary,
        "manifests": manifests,
        "state": {
            "B": b_info, "C": c_info, "D": d_info, "E": e_info,
            "reload_match": reload,
            "byte_ratio": bytes_ratio,
            "C_D_within_5pct": bool(0.95 <= bytes_ratio["C_D"] <= 1.05),
            "C_E_within_5pct": bool(0.95 <= bytes_ratio["C_E"] <= 1.05),
        },
        "determinism": determinism,
        "timings_seconds": timings,
        "cuda_memory": cuda_memory(),
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    # gate evaluation for this boundary
    gates = diagnostics.gate_evaluation(
        point, boot, cert, quant_c, tail,
        {"C_D": bytes_ratio["C_D"], "C_E": bytes_ratio["C_E"]},
        list(groups.keys()))
    result["gates"] = gates
    result["artifact"] = write_json(out_dir / f"boundary_t{t}.json", result, exclusive=True)

    del writer_bins, queries, scores, scores_np
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def parity_report(cpu_dir: Path, cuda_dir: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"boundaries": {}}
    for t in C1.BOUNDARIES:
        a = np.load(cpu_dir / "scores" / f"scores_t{t}.npz")
        b = np.load(cuda_dir / "scores" / f"scores_t{t}.npz")
        entry = {}
        for arm in "ABCDE":
            x, y = a[arm], b[arm]
            diff = np.abs(x - y)
            entry[arm] = {
                "shape_match": x.shape == y.shape,
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "bitwise_equal": bool(np.array_equal(x, y)),
                "image_max_spearman": diagnostics.spearman(x.max(axis=1), y.max(axis=1)),
            }
        payload["boundaries"][str(t)] = entry
    state_parity = {}
    for t in C1.BOUNDARIES:
        for arm in ("B", "C", "D", "E"):
            p1 = cpu_dir / "states" / f"{arm.lower()}_t{t}.csg00"
            p2 = cuda_dir / "states" / f"{arm.lower()}_t{t}.csg00"
            state_parity[f"{arm}_t{t}"] = {
                "sha256_cpu": C0.sha256_file(p1),
                "sha256_cuda": C0.sha256_file(p2),
                "identical": C0.sha256_file(p1) == C0.sha256_file(p2),
            }
    worst = max(e[arm]["max_abs_diff"] for e in payload["boundaries"].values()
                for arm in "ABCDE")
    payload["state_files"] = state_parity
    payload["worst_arm_max_abs_diff"] = float(worst)
    payload["tolerance"] = 1e-3
    payload["passed"] = bool(worst <= 1e-3 and all(v["identical"] for v in state_parity.values()))
    return payload


def decide(cpu_metrics: Dict[str, Any], parity: Dict[str, Any]) -> Dict[str, Any]:
    raw = cpu_metrics["boundaries"]
    boundaries = ({str(int(e["boundary"])): e for e in raw}
                  if isinstance(raw, list) else {str(k): v for k, v in raw.items()})
    final = boundaries["350"]
    per_boundary_gates = {t: boundaries[t]["gates"] for t in boundaries}
    quant_fail = [t for t, g in per_boundary_gates.items() if not g["quantization"]["passed"]]
    quant_undef = [t for t, g in per_boundary_gates.items()
                   if g["quantization"]["undefined"]]
    sel_fail = [t for t, g in per_boundary_gates.items() if not g["selection"]["passed"]]
    sel_undef = [t for t, g in per_boundary_gates.items() if g["selection"]["undefined"]]
    mem = final["gates"]["memory_efficiency"]
    integrity = bool(
        parity.get("passed", False)
        and all(e["state"]["reload_match"][a] for e in boundaries.values() for a in "BCDE")
        and all(e["state"]["C_D_within_5pct"] and e["state"]["C_E_within_5pct"]
                for e in boundaries.values())
        and final["certificates"]["all_hold"]
        and not any(e["selection"][a]["n_unique_global_ids"] != e["selection"][a]["n_selected"]
                    for e in boundaries.values() for a in "BCDE")
        and not any(not e["selection"][a]["all_events_causal"]
                    for e in boundaries.values() for a in "BCDE"))
    checks = {
        "integrity": integrity,
        "quantization_failures": quant_fail,
        "quantization_undefined": quant_undef,
        "selection_failures": sel_fail,
        "selection_undefined": sel_undef,
        "memory": mem,
        "certificates_hold": final["certificates"]["all_hold"],
        "parity_worst_abs_diff": parity.get("worst_arm_max_abs_diff"),
        "final_quantization": final["gates"]["quantization"],
        "final_selection": final["gates"]["selection"],
    }
    if not integrity:
        return {"decision": "CSG01_BLOCKED", "checks": checks,
                "reasons": ["integrity, certificate, byte or parity failure"]}
    if quant_fail:
        return {"decision": "CSG01_NEGATIVE_QUANTIZATION", "checks": checks,
                "reasons": [f"quantization gate failed at boundaries {quant_fail}"]}
    if quant_undef:
        return {"decision": "CSG01_INCONCLUSIVE", "checks": checks,
                "reasons": [f"quantization endpoint undefined at {quant_undef}"]}
    if sel_fail:
        return {"decision": "CSG01_NEGATIVE_SELECTION_OR_BUDGET", "checks": checks,
                "reasons": [f"selection gate failed at boundaries {sel_fail}"]}
    if sel_undef:
        return {"decision": "CSG01_INCONCLUSIVE", "checks": checks,
                "reasons": [f"selection endpoint undefined at {sel_undef}"]}
    if not mem["passed"]:
        return {"decision": "CSG01_NEGATIVE_SELECTION_OR_BUDGET", "checks": checks,
                "reasons": ["C has no pre-registered byte-efficiency advantage over D/E",
                            f"bytes_ok={mem['bytes_within_5pct']}",
                            f"advantage_hits={mem['advantage_hits']}",
                            f"severe_category_loss={mem['severe_category_loss']}"]}
    return {"decision": "CSG01_GO_REQUEST_FRESH_REAL_ANOMALY_GATE", "checks": checks,
            "reasons": ["quantization, selection and byte-efficiency gates all passed"]}


def cmd_verify(args: argparse.Namespace) -> int:
    result = g0.verify_all()
    print(json.dumps(clean_json({
        "agents_md": result["agents_md"],
        "checkpoint": result["checkpoint"],
        "p0c_protocol": result["p0c_protocol"],
        "stream": {k: result["stream"][k] for k in
                   ("writer_events", "witnesses", "stream_digest", "witness_digest")},
        "normal_only_guard": result["normal_only_guard"],
        "cache_rows": result["cache_rows"]["checked_rows"],
        "feature_geometry": {k: result["feature_geometry"][k] for k in
                             ("patch_grid", "dim", "dtype", "capture")},
        "byte_budget_dry_run": result["byte_budget_dry_run"],
        "quota_sum_B": int(sum(result["quota_vectors"]["B"])),
        "quota_sum_E": int(sum(result["quota_vectors"]["E"])),
        "environment": result["environment"],
    }), indent=2))
    return 0


def cmd_lock(args: argparse.Namespace) -> int:
    info = g0.write_lock()
    print(json.dumps(info, indent=2))
    return 0


def cmd_parity(args: argparse.Namespace) -> int:
    g0.verify_lock()
    configure_torch(args.device)
    stream = C0.load_stream()
    cache = F0.load_feature_cache(verify_hashes=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = F0.feature_parity_check(cache, stream["updates"], device=args.device)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["cuda_memory"] = cuda_memory()
    result["checkpoint_bytes"] = C1.CHECKPOINT.stat().st_size
    result["no_anomaly_attestation"] = "four normal train/good writer images only"
    info = write_json(C1.RUN_DIR / "feature_parity.json", result, exclusive=True)
    print(json.dumps({"all_ok": result["all_ok"], "sha256": info["sha256"],
                      "elapsed_seconds": result["elapsed_seconds"],
                      "cuda_memory": result["cuda_memory"]}, indent=2))
    return 0 if result["all_ok"] else 3


def cmd_experiment(args: argparse.Namespace) -> int:
    g0.verify_lock()
    configure_torch(args.device)
    out_dir = C1.RUN_DIR / args.tag
    if out_dir.exists():
        raise FileExistsError(f"run tag exists: {out_dir}")
    out_dir.mkdir(parents=True)
    stream = C0.load_stream()
    cache = F0.load_feature_cache(verify_hashes=True)
    positions = C0.bin_positions()
    access = AccessLog()
    started = time.perf_counter()
    boundary_results = []
    imgmax: Dict[str, Dict[int, np.ndarray]] = {arm: {} for arm in "ABCDE"}
    for t in C1.BOUNDARIES:
        print(f"[csg01] boundary {t} on {args.device}", flush=True)
        result = run_boundary(t, cache, stream, positions, out_dir, args.device,
                              access, determinism_check=(t == 350))
        boundary_results.append(result)
        npz = np.load(out_dir / "scores" / f"scores_t{t}.npz")
        for arm in "ABCDE":
            imgmax[arm][t] = npz[f"imgmax_{arm}"]
        for row in stream["updates"][:t]:
            access.record("G2_writer", row)
        for row in stream["probes"][: C1.ELIGIBLE_PREFIX[t]]:
            access.record("G2_witness", row)
    wall = time.perf_counter() - started
    drift = diagnostics.drift_summary(imgmax)
    access_info = access.write(out_dir / "access_log.jsonl")
    state_bytes = {str(r["boundary"]): {a: r["state"][a]["bytes"] for a in "BCDE"}
                   for r in boundary_results}
    a_full_bytes = {str(r["boundary"]): r["n_writer_events"] * C1.PATCHES * C1.DIM * 4
                    for r in boundary_results}
    resource_ledger = {
        "device": args.device,
        "wall_seconds": wall,
        "boundary_timings": {str(r["boundary"]): r["timings_seconds"] for r in boundary_results},
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cuda_memory": cuda_memory(),
        "boundary_cuda_memory": {str(r["boundary"]): r["cuda_memory"] for r in boundary_results},
        "checkpoint_bytes": C1.CHECKPOINT.stat().st_size,
        "analysis_only_feature_cache_bytes": int(sum(
            (C1.CACHE_DIR / f"{c}_train.pt").stat().st_size for c in C1.CATEGORIES)),
        "scores_artifact_bytes": int(sum(p.stat().st_size
                                         for p in (out_dir / "scores").glob("*.npz"))),
        "state_bytes": state_bytes,
        "full_support_A_ram_bytes": a_full_bytes,
        "decoded_temporary_bytes": {"C_D": C1.SELECTION_COUNT * C1.DIM * 4,
                                    "E": C1.SIZE_MATCHED_COUNT * C1.DIM * 4},
        "optimizer_bytes": 0,
        "access_log": access_info,
        "no_anomaly_attestation": (
            "only MVTec AD v1 train/good normal identities and the frozen normal "
            "feature cache were read; no test/mask/anomaly/consumed-DEV/sealed payload"),
    }
    byte_ledger = {
        "dry_run": C1.choose_size_matched_count(),
        "actual": {str(r["boundary"]): {a: {
            "bytes": r["state"][a]["bytes"],
            "payload_bytes": r["state"][a]["payload_bytes"],
            "header_bytes": r["state"][a]["header_bytes"],
            "sha256": r["state"][a]["sha256"],
        } for a in "BCDE"} for r in boundary_results},
        "ratios": {str(r["boundary"]): r["state"]["byte_ratio"] for r in boundary_results},
        "A_full_support_bytes": a_full_bytes,
        "executable_inference_bytes": {
            "checkpoint_bytes": C1.CHECKPOINT.stat().st_size,
            "state_C_t350_bytes": boundary_results[-1]["state"]["C"]["bytes"],
            "state_B_t350_bytes": boundary_results[-1]["state"]["B"]["bytes"],
            "state_D_t350_bytes": boundary_results[-1]["state"]["D"]["bytes"],
            "state_E_t350_bytes": boundary_results[-1]["state"]["E"]["bytes"],
            "code_bytes": int(sum(p.stat().st_size for p in (C1.ROOT / "exps" / "csg01").glob("*.py"))),
            "optimizer_bytes": 0,
        },
        "analysis_only_not_part_of_detector": {
            "feature_cache_bytes": int(sum(
                (C1.CACHE_DIR / f"{c}_train.pt").stat().st_size for c in C1.CATEGORIES)),
            "full_support_A_note": "Arm A retains all normal descriptors by design and is not a "
                                   "compact deployable detector",
            "selection_provenance_note": "selection ID manifests are audit artifacts, not "
                                         "persistent detector state",
        },
    }
    summary = {
        "run_id": C1.RUN_ID,
        "tag": args.tag,
        "device": args.device,
        "lock": g0.verify_lock(),
        "boundaries": boundary_results,
        "drift": drift,
        "resource_ledger": resource_ledger,
        "byte_ledger": byte_ledger,
    }
    info = write_json(out_dir / "boundary_metrics.json", summary, exclusive=True)
    print(json.dumps({"tag": args.tag, "device": args.device, "artifact": info,
                      "wall_seconds": wall}, indent=2))
    return 0


def cmd_seal(args: argparse.Namespace) -> int:
    g0.verify_lock()
    if C1.G2_SEAL_PATH.exists():
        raise FileExistsError(f"seal exists: {C1.G2_SEAL_PATH}")
    cpu_dir = C1.RUN_DIR / args.cpu
    cuda_dir = C1.RUN_DIR / args.cuda
    cpu_metrics = json.loads((cpu_dir / "boundary_metrics.json").read_text())
    cuda_metrics = json.loads((cuda_dir / "boundary_metrics.json").read_text())
    parity = parity_report(cpu_dir, cuda_dir)
    decision = decide(cpu_metrics, parity)
    combined = {
        "run_id": C1.RUN_ID,
        "canonical_run_tag": args.cpu,
        "execution_run_tag": args.cuda,
        "protocol": C1.protocol_manifest(),
        "decision": decision,
        "parity": parity,
        "cpu_metrics": cpu_metrics,
        "cuda_metrics": cuda_metrics,
        "no_anomaly_attestation": (
            "G2 read only MVTec AD v1 train/good normal identities and the frozen "
            "normal feature cache. No test image, ground-truth mask, anomaly label, "
            "consumed DEV cohort or sealed confirmation identity was opened."),
        "limitations": [
            "normal-support geometry only; no Pixel AP, AUROC or anomaly efficacy",
            "Arm A is a geometry reference, not a Pixel-AP upper bound",
            "offline refitting is not online continual learning",
            "the 59 normal witnesses are historically exposed, not fresh confirmation",
            "one stream/order/seed; no multi-seed uncertainty",
            "the mathematical bounds are about the distance function, not detection",
        ],
    }
    info = write_json(C1.RUN_DIR / "G2_METRICS.json", combined, exclusive=True)
    artifacts = {}
    for path in sorted(C1.RUN_DIR.rglob("*")):
        if path.is_file() and path != C1.G2_SEAL_PATH:
            artifacts[str(path.relative_to(C1.RUN_DIR))] = {
                "bytes": path.stat().st_size, "sha256": C0.sha256_file(path)}
    seal = {
        "schema": "csg01_g2_seal_v1",
        "run_id": C1.RUN_ID,
        "sealed_utc": C0.utc_now(),
        "decision": decision["decision"],
        "decision_detail": decision,
        "git": g0.git_state(),
        "environment": g0.environment_info(probe_gpu=True),
        "artifacts": artifacts,
        "metrics_artifact": info,
        "no_anomaly_attestation": combined["no_anomaly_attestation"],
        "historical_hashes_verified": True,
    }
    with open(C1.G2_SEAL_PATH, "x", encoding="utf-8") as fh:
        json.dump(clean_json(seal), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    print(json.dumps({"seal": str(C1.G2_SEAL_PATH),
                      "sha256": C0.sha256_file(C1.G2_SEAL_PATH),
                      "decision": decision["decision"], "reasons": decision["reasons"]},
                     indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSG-01 quantized normal-support memory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    sub.add_parser("lock")
    p = sub.add_parser("parity")
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    p = sub.add_parser("experiment")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--tag", required=True)
    p = sub.add_parser("seal")
    p.add_argument("--cpu", default="cpu")
    p.add_argument("--cuda", default="cuda")
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
