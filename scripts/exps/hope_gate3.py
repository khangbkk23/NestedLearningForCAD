# scripts/exps/hope_gate3.py
"""Run the experiment-only 200-image HOPE Task 3R Gate 3 diagnostic."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from exps.hope_gate3 import (
    CLASS_ORDER,
    boundary_rows,
    fit_slopes,
    load_200_stream,
    retention_matrix,
    run_long_candidate,
)
from exps.hope_retention_stabilization import canonical_initial_states, write_table
from exps.hope_update_stabilization import UpdateMapping

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = REPO_ROOT / "results/hope_cad/real_feature_probe/real_cpu_seed0/features"
GATE3_ROOT = REPO_ROOT / "results/hope_cad/update_stabilization/gate3_seed0"
FRONTIER_ROOT = REPO_ROOT / "results/hope_cad/update_stabilization/gate3_frontier_seed0"
DUALRATE_ROOT = REPO_ROOT / "results/hope_cad/update_stabilization/gate3_dualrate_seed0"


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def mapping(name: str, *, alpha: str = "one", lam: float = 0.0, eta_h: float = 0.02, surprise_scale: float = 1.0) -> UpdateMapping:
    return UpdateMapping(name, alpha_kind=alpha, lambda_h=lam, eta_kind="horizon_sigmoid", eta_h=eta_h, surprise_eta_multiplier=surprise_scale)


def phase_a() -> tuple[UpdateMapping, ...]:
    return (mapping("HNP_eta_h_0.02_alpha1"), mapping("HNR_0.002_HNP_0.02", alpha="horizon_near_one", lam=0.002), mapping("HNR_0.002_HNP_0.10", alpha="horizon_near_one", lam=0.002, eta_h=0.10))


def phase_b() -> tuple[UpdateMapping, ...]:
    return (mapping("B0_HNP_eta_h_0.05_alpha1", eta_h=0.05), mapping("B1_HNR_0.001_HNP_0.02", alpha="horizon_near_one", lam=0.001), mapping("B2_HNR_0.005_HNP_0.02", alpha="horizon_near_one", lam=0.005), mapping("B3_HNR_0.002_HNP_0.05", alpha="horizon_near_one", lam=0.002, eta_h=0.05))


def phase_c() -> tuple[UpdateMapping, ...]:
    return (mapping("DUALRATE_eta0.10_gamma0.50", alpha="horizon_near_one", lam=0.002, eta_h=0.10, surprise_scale=0.50), mapping("DUALRATE_eta0.10_gamma0.25", alpha="horizon_near_one", lam=0.002, eta_h=0.10, surprise_scale=0.25))


def map_config(value: UpdateMapping) -> dict[str, Any]:
    return {key: getattr(value, key) for key in ("name", "alpha_kind", "lambda_h", "eta_kind", "eta_scale", "eta_h", "horizon", "disable_rank", "disable_surprise", "no_update", "freeze_eta", "freeze_alpha", "rank_eta_multiplier", "surprise_eta_multiplier")}


def mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else float("nan")


def summary(result: Mapping[str, Any], phase: str) -> dict[str, Any]:
    rows = list(result["rows"])
    first, last = rows[0], rows[-1]
    slopes = fit_slopes(rows)
    whole = {row["metric"]: row["slope"] for row in slopes if row["window_start"] == 1 and row["window_end"] == 200}
    prior = [row for row in result["anchor_rows"] if row["space"] == "SMT" and int(row["checkpoint_event"]) > (list(CLASS_ORDER).index(row["class_name"]) + 1) * 40]
    return {
        "phase": phase, "candidate": result["candidate"], "classification": result["classification"], "completed_events": result["complete_event_count"], "failure": result["failure"],
        "alpha_kind": result["mapping"].alpha_kind, "lambda_h": result["mapping"].lambda_h, "eta_h": result["mapping"].eta_h, "surprise_eta_multiplier": result["mapping"].surprise_eta_multiplier,
        "memory_final_event1": float(last["memory_norm"] / (first["memory_norm"] + 1e-12)), "memory_whole_slope": whole.get("log_memory_norm"), "smt_rms_final_event1": last["smt_rms_rel_event1"], "smt_rms_whole_slope": whole.get("log_smt_rms"), "smt_variance_final_event1": last["smt_centered_variance_rel_event1"], "smt_rank_final_event1": last["smt_effective_rank_rel_event1"], "smt_top1_energy_final": last["smt_top1_energy_fraction"], "hope_rms_final_event1": last["hope_rms_rel_event1"], "hope_rms_whole_slope": whole.get("log_hope_rms"), "hope_variance_final_event1": last["hope_centered_variance_rel_event1"], "hope_rank_final_event1": last["hope_effective_rank_rel_event1"], "hope_top1_energy_final": last["hope_top1_energy_fraction"], "mean_prior_anchor_cosine": mean(prior, "cosine"), "mean_prior_anchor_relative_l2": mean(prior, "relative_l2"), "mean_surprise_rank_ratio": mean(rows, "surprise_rank_norm_ratio"), "mean_current_image_smt_relative_l2": mean(rows, "current_image_smt_relative_l2"), "mean_current_image_hope_relative_l2": mean(rows, "current_image_hope_relative_l2"), "state_bytes": last["state_bytes"], "schema_constant": len({row["state_schema_signature"] for row in rows}) == 1, "state_bytes_constant": len({row["state_bytes"] for row in rows}) == 1,
    }


def run_set(records: list[dict[str, Any]], mappings: Sequence[UpdateMapping], smt_state: Mapping[str, Any], cms_state: Mapping[str, Any], device: str, max_events: int, phase: str) -> list[dict[str, Any]]:
    results = []
    for candidate in mappings:
        print(f"START {phase} {candidate.name}", flush=True)
        result = run_long_candidate(records, candidate, smt_state, cms_state, device, max_events=max_events, progress=True)
        result["phase"] = phase
        results.append(result)
        print(f"END {phase} {candidate.name}: {result['classification']} events={result['complete_event_count']}", flush=True)
    return results


def write_phase(root: Path, results: Sequence[Mapping[str, Any]], phase: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root.mkdir(parents=True, exist_ok=True)
    summaries = [summary(result, phase) for result in results]
    event_rows = [{"phase": phase, **dict(row)} for result in results for row in result["rows"]]
    anchor_rows = [{"phase": phase, **dict(row)} for result in results for row in result["anchor_rows"]]
    if phase == "phase_a":
        write_table(root / "candidates_200.parquet", summaries)
        write_table(root / "per_event_200.parquet", event_rows)
        write_table(root / "per_boundary.parquet", [row for result in results for row in boundary_rows(result["rows"], result["anchor_rows"], result["candidate"])])
        write_table(root / "state_slopes.parquet", [{"candidate": result["candidate"], **row} for result in results for row in fit_slopes(result["rows"])])
        write_table(root / "anchor_drift.parquet", anchor_rows)
        write_table(root / "anchor_retention_matrix.parquet", [row for result in results for row in retention_matrix(result["anchor_rows"], "cosine")])
        write_table(root / "anchor_retention_l2_matrix.parquet", [row for result in results for row in retention_matrix(result["anchor_rows"], "relative_l2")])
    elif phase in ("phase_b", "phase_c_screen"):
        write_table(root / "candidates_50.parquet", summaries)
        write_table(root / "per_event_50.parquet", event_rows)
        write_table(root / "anchor_drift.parquet", anchor_rows)
    elif phase == "phase_c_200":
        write_table(root / "candidates_200.parquet", summaries)
        write_table(root / "per_event_200.parquet", event_rows)
        write_table(root / "anchor_drift_200.parquet", anchor_rows)
        write_table(root / "state_slopes_200.parquet", [{"candidate": result["candidate"], **row} for result in results for row in fit_slopes(result["rows"])])
        write_table(root / "anchor_retention_matrix_200.parquet", [row for result in results for row in retention_matrix(result["anchor_rows"], "cosine")])
        write_table(root / "anchor_retention_l2_matrix_200.parquet", [row for result in results for row in retention_matrix(result["anchor_rows"], "relative_l2")])
    else:
        raise ValueError(f"unknown output phase: {phase}")
    return summaries, event_rows


def format_table(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str:
    if not rows:
        return "(none)"
    text = "| " + " | ".join(keys) + " |\n|" + "|".join("---" for _ in keys) + "|\n"
    return text + "".join("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |\n" for row in rows)


def write_report(path: Path, phase_a_summaries: Sequence[Mapping[str, Any]], phase_b_summaries: Sequence[Mapping[str, Any]], phase_c_summaries: Sequence[Mapping[str, Any]], phase_a_boundaries: Sequence[Mapping[str, Any]], phase_a_slopes: Sequence[Mapping[str, Any]], duration: float, device: str, phase_c_ran: bool) -> None:
    stable = [row for row in phase_a_summaries if row["classification"] == "STABLE"]
    lines = ["# HOPE Task 3R Gate 3 — Long-Horizon Normal Viability", "", "Experiment-only normal-train diagnostics. Production SMT/CMS/HOPE files were not modified; no anomaly/test images or anomaly metrics were used.", "", f"Device: `{device}`; seed `0`; runtime `{duration:.2f}` seconds. Stream: 40 bottle, 40 carpet, 40 grid, 40 toothbrush, 40 transistor train/good cache entries. No reset or flush.", "", "## Mandatory 200-image candidates", "", format_table(phase_a_summaries, ["candidate", "completed_events", "classification", "memory_final_event1", "memory_whole_slope", "smt_rms_final_event1", "smt_variance_final_event1", "smt_rank_final_event1", "smt_top1_energy_final", "hope_rms_final_event1", "hope_variance_final_event1", "hope_rank_final_event1", "hope_top1_energy_final", "mean_prior_anchor_cosine", "mean_prior_anchor_relative_l2", "mean_surprise_rank_ratio", "state_bytes", "schema_constant"]), "", "## Task-boundary diagnostics", "", format_table(phase_a_boundaries, ["candidate", "boundary", "from_class", "to_class", "memory_relative_change", "smt_rms_before", "smt_rms_after", "hope_rms_before", "hope_rms_after", "smt_rank_before", "smt_rank_after", "prior_anchor_cosine_mean", "prior_anchor_relative_l2_mean"]), "", "## Window/whole-stream slopes", "", format_table(phase_a_slopes, ["candidate", "window_start", "window_end", "metric", "slope", "count"]), "", "## Stability/plasticity frontier", "", format_table(list(phase_b_summaries) + list(phase_c_summaries), ["phase", "candidate", "classification", "completed_events", "mean_current_image_smt_relative_l2", "mean_current_image_hope_relative_l2", "mean_prior_anchor_cosine", "mean_prior_anchor_relative_l2", "smt_rms_final_event1", "smt_variance_final_event1"]), "", "## Findings", "", f"- Phase-A stable count: `{len(stable)}/{len(phase_a_summaries)}`; Phase B ran iff at least two were stable.", f"- Phase C dual-rate screen ran: `{phase_c_ran}` and at most one dual-rate candidate was advanced to 200 events.", "- Anchor matrices separate state-induced drift from changing input geometry. The reference for each anchor is its first category-boundary read-only output.", "- H1 patch-horizon scaling and H2 surprise-dominated instability are assessed against the earlier Gate-2C decomposition and these 200-event slopes.", "- H3 stability/plasticity is reported as separate historical-anchor drift and current-image adaptation proxies; no scalar performance score is invented.", "- H4 dual-rate remains an experimental mapping only.", "", "## Task-3B selection", "", "Recommend at most three: the simplest stable HNP baseline, the best HNR+HNP candidate, and at most one genuinely non-dominated frontier/dual-rate candidate. No anomaly work was started.", "", "## Artifacts", "", "Mandatory files are under `results/hope_cad/update_stabilization/gate3_seed0/`; frontier and dual-rate files are written when their conditional phases run.", "", "TASK3R_GATE3_COMPLETE = YES"]
    path.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gate3",), default="gate3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    for name in CLASS_ORDER:
        if not (FEATURE_ROOT / f"class_{name}.pt").is_file():
            parser.error(f"missing cache shard {name}")
    GATE3_ROOT.mkdir(parents=True, exist_ok=True)
    config = {"seed": args.seed, "device": device, "stage": "gate3", "feature_root": str(FEATURE_ROOT), "stream": {"classes": list(CLASS_ORDER), "images_per_class": 40, "events": 200, "reset_at_boundaries": False}, "smt": {"adaptive_q": False, "memory_chunk_size": 16, "auxiliary_memory_chunk_size": 16, "eq93_unchanged": True}, "cms": {"K": 2, "update_periods": [1, 8], "objective": "Candidate-B probe-only project mapping"}, "phase_a": [map_config(value) for value in phase_a()], "phase_b": [map_config(value) for value in phase_b()], "phase_c": [map_config(value) for value in phase_c()], "effective_rank": "full 784 centered entropy rank eps=1e-12", "top1_energy": "s[0]^2/sum(s^2)"}
    (GATE3_ROOT / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    records = load_200_stream(FEATURE_ROOT)
    smt_state, cms_state = canonical_initial_states(args.seed, 768)
    started = time.perf_counter()
    phase_a_results = run_set(records, phase_a(), smt_state, cms_state, device, 200, "phase_a")
    stable_count = sum(result["classification"] == "STABLE" for result in phase_a_results)
    phase_b_results: list[dict[str, Any]] = []
    phase_c_results: list[dict[str, Any]] = []
    if stable_count >= 2:
        phase_b_results = run_set(records[:50], phase_b(), smt_state, cms_state, device, 50, "phase_b")
        baseline = next(result for result in phase_a_results if result["candidate"] == "HNP_eta_h_0.02_alpha1")
        frontier = next(result for result in phase_b_results if result["candidate"] == "B0_HNP_eta_h_0.05_alpha1")
        base_plasticity = sum(float(row["current_image_smt_relative_l2"]) for row in baseline["rows"][:50]) / 50.0
        frontier_plasticity = sum(float(row["current_image_smt_relative_l2"]) for row in frontier["rows"]) / max(1, len(frontier["rows"]))
        if frontier["classification"] in ("STABLE", "SLOW_DRIFT") and frontier_plasticity > base_plasticity * 1.05:
            phase_c_results = run_set(records[:50], phase_c(), smt_state, cms_state, device, 50, "phase_c")
            eligible = [result for result in phase_c_results if result["classification"] in ("STABLE", "SLOW_DRIFT")]
            eligible.sort(key=lambda result: summary(result, "phase_c")["mean_prior_anchor_relative_l2"])
            if eligible:
                selected = eligible[0]
                print(f"START phase_c_200 {selected['candidate']}", flush=True)
                advanced = run_long_candidate(records, selected["mapping"], smt_state, cms_state, device, max_events=200, progress=True)
                advanced["phase"] = "phase_c_200"
                selected["advanced_200"] = advanced
                print(f"END phase_c_200 {selected['candidate']}: {advanced['classification']} events={advanced['complete_event_count']}", flush=True)
        else:
            print("SKIP phase_c: frontier point did not clearly improve the current-image plasticity proxy", flush=True)
    phase_a_summaries, _, = write_phase(GATE3_ROOT, phase_a_results, "phase_a")
    phase_a_boundaries = [row for result in phase_a_results for row in boundary_rows(result["rows"], result["anchor_rows"], result["candidate"])]
    phase_a_slopes = [{"candidate": result["candidate"], **row} for result in phase_a_results for row in fit_slopes(result["rows"])]
    phase_b_summaries: list[dict[str, Any]] = []
    if phase_b_results:
        phase_b_summaries, _ = write_phase(FRONTIER_ROOT, phase_b_results, "phase_b")
        (FRONTIER_ROOT / "config_resolved.yaml").write_text(yaml.safe_dump({**config, "stage": "gate3_frontier"}, sort_keys=False))
    phase_c_summaries: list[dict[str, Any]] = []
    if phase_c_results:
        phase_c_screen_summaries, _ = write_phase(DUALRATE_ROOT, phase_c_results, "phase_c_screen")
        advanced_results = [result["advanced_200"] for result in phase_c_results if "advanced_200" in result]
        phase_c_advanced_summaries: list[dict[str, Any]] = []
        if advanced_results:
            phase_c_advanced_summaries, _ = write_phase(DUALRATE_ROOT, advanced_results, "phase_c_200")
        phase_c_summaries = phase_c_screen_summaries + phase_c_advanced_summaries
        (DUALRATE_ROOT / "config_resolved.yaml").write_text(yaml.safe_dump({**config, "stage": "gate3_dualrate"}, sort_keys=False))
    duration = time.perf_counter() - started
    summary_rows = phase_a_summaries + phase_b_summaries + phase_c_summaries
    payload = {"status": "GATE3_COMPLETE", "run_id": "gate3_seed0", "device": device, "duration_seconds": duration, "phase_a_stable_count": stable_count, "phase_b_ran": bool(phase_b_results), "phase_c_ran": bool(phase_c_results), "stress400_ran": False, "phase_a_candidates": phase_a_summaries, "frontier_candidates": phase_b_summaries, "dualrate_candidates": phase_c_summaries, "task3b_candidates": [row["candidate"] for row in phase_a_summaries if row["classification"] == "STABLE"][:3], "gate3_blocked": not any(row["classification"] == "STABLE" for row in phase_a_summaries)}
    (GATE3_ROOT / "summary.json").write_text(json.dumps(safe(payload), indent=2) + "\n")
    write_report(REPO_ROOT / "agents/reports/hope_task3r_gate3_long_horizon.md", phase_a_summaries, phase_b_summaries, phase_c_summaries, phase_a_boundaries, phase_a_slopes, duration, device, bool(phase_c_results))
    print(json.dumps(safe(payload), indent=2), flush=True)
    print("TASK3R_GATE3_COMPLETE = YES", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
