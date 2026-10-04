# scripts/exps/hope_update_stabilization.py
"""Run the isolated Gate 2C update-term and patch-horizon experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml

from exps.hope_retention_stabilization import (
    HORIZON_TOKENS,
    build_read_only_rms_reference,
    canonical_initial_states,
    load_cached_stream,
    write_table,
)
from exps.hope_update_stabilization import (
    UpdateMapping,
    flatten_term_summary,
    phase_a_mappings,
    phase_b_mappings,
    phase_c_mapping,
    phase_d_mapping,
    pm0,
    pm0_oracle_check,
    run_stream_update,
    run_update_smt,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = REPO_ROOT / "results/hope_cad/real_feature_probe/real_cpu_seed0/features"
OUTPUT_ROOT = REPO_ROOT / "results/hope_cad/update_stabilization/gate2c_seed0"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _mapping_config(mapping: UpdateMapping) -> dict[str, Any]:
    return {
        "name": mapping.name,
        "alpha_kind": mapping.alpha_kind,
        "lambda_h": mapping.lambda_h,
        "eta_kind": mapping.eta_kind,
        "eta_scale": mapping.eta_scale,
        "eta_h": mapping.eta_h,
        "horizon": mapping.horizon,
        "disable_rank": mapping.disable_rank,
        "disable_surprise": mapping.disable_surprise,
        "no_update": mapping.no_update,
        "freeze_eta": mapping.freeze_eta,
        "freeze_alpha": mapping.freeze_alpha,
    }


def _summary_row(result: Mapping[str, Any], phase: str) -> dict[str, Any]:
    rows = result.get("rows", [])
    first = rows[0] if rows else {}
    final = rows[-1] if rows else {}
    mapping = result["mapping"]
    return {
        "phase": phase,
        "candidate": result["candidate"],
        "classification": result["classification"],
        "event_count": result["event_count"],
        "complete_event_count": result["complete_event_count"],
        "failure": result["failure"],
        "alpha_kind": mapping.alpha_kind,
        "lambda_h": mapping.lambda_h,
        "eta_kind": mapping.eta_kind,
        "eta_h": mapping.eta_h,
        "eta_scale": mapping.eta_scale,
        "disable_rank": mapping.disable_rank,
        "disable_surprise": mapping.disable_surprise,
        "freeze_eta": mapping.freeze_eta,
        "freeze_alpha": mapping.freeze_alpha,
        "first_memory_norm": first.get("memory_norm"),
        "final_memory_norm": final.get("memory_norm"),
        "final_memory_norm_rel_event1": final.get("memory_norm_rel_event1"),
        "final_smt_rms": final.get("smt_rms"),
        "final_smt_rms_rel_event1": final.get("smt_rms_rel_event1"),
        "final_smt_variance": final.get("smt_centered_variance"),
        "final_smt_variance_rel_event1": final.get("smt_centered_variance_rel_event1"),
        "final_smt_rank": final.get("smt_effective_rank"),
        "final_smt_rank_rel_event1": final.get("smt_effective_rank_rel_event1"),
        "final_hope_rms": final.get("hope_rms"),
        "final_hope_rms_rel_event1": final.get("hope_rms_rel_event1"),
        "final_hope_variance": final.get("hope_centered_variance"),
        "final_hope_variance_rel_event1": final.get("hope_centered_variance_rel_event1"),
        "final_hope_rank": final.get("hope_effective_rank"),
        "final_hope_rank_rel_event1": final.get("hope_effective_rank_rel_event1"),
        "alpha_product_event1": first.get("alpha_product_observed"),
        "alpha_product_final": final.get("alpha_product_observed"),
        "alpha_product_cumulative_final": final.get("alpha_product_cumulative"),
        "finite": result["all_finite"],
        "state_bytes_constant": result["state_bytes_constant"],
    }


def _run_phase(
    phase: str,
    mappings: Iterable[UpdateMapping],
    records: list[dict[str, Any]],
    smt_state: Mapping[str, Any],
    cms_state: Mapping[str, Any],
    device: str,
    references: list[dict[str, float]],
    *,
    max_events: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for mapping in mappings:
        print(f"START {phase} {mapping.name}", flush=True)
        result = run_stream_update(
            records, mapping, smt_state, cms_state, device,
            max_events=max_events, read_only_references=references[:max_events], progress=True,
        )
        result["phase"] = phase
        results.append(result)
        for row in result["rows"]:
            row = dict(row)
            row["phase"] = phase
            event_rows.append(row)
        print(
            f"END {phase} {mapping.name}: classification={result['classification']} "
            f"events={result['event_count']} complete={result['complete_event_count']}",
            flush=True,
        )
    return results, event_rows


def _neutral_retention(lambda_h: float) -> dict[str, float | None]:
    if lambda_h == 0.0:
        return {"alpha_neutral": 1.0, "retention_16": 1.0, "retention_784": 1.0, "retention_50_images": 1.0}
    alpha = 1.0 - lambda_h / HORIZON_TOKENS * 0.5
    return {
        "alpha_neutral": alpha,
        "retention_16": alpha ** 16,
        "retention_784": alpha ** HORIZON_TOKENS,
        "retention_50_images": alpha ** (HORIZON_TOKENS * 50),
    }


def _neutral_eta(eta_h: float) -> float | None:
    return eta_h / HORIZON_TOKENS * 0.5 if eta_h else None


def _format_md_table(rows: list[Mapping[str, Any]], keys: list[str]) -> str:
    if not rows:
        return "(none)"
    header = "| " + " | ".join(keys) + " |\n|" + "|".join("---" for _ in keys) + "|\n"
    body = "".join("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |\n" for row in rows)
    return header + body


def _write_report(
    path: Path,
    *,
    oracle: Mapping[str, Any],
    term_rows: list[Mapping[str, Any]],
    summaries: list[Mapping[str, Any]],
    phase_a_results: list[Mapping[str, Any]],
    phase_b_results: list[Mapping[str, Any]],
    phase_c_results: list[Mapping[str, Any]],
    phase_d_results: list[Mapping[str, Any]],
    phase_e_results: list[Mapping[str, Any]],
    duration: float,
    device: str,
) -> None:
    term_summary = flatten_term_summary(term_rows)
    full = next((row for row in summaries if row["candidate"] == "FULL_alpha1_eta0.02"), {})
    rank = next((row for row in summaries if row["candidate"] == "RANK_ONLY_alpha1_eta0.02"), {})
    surprise = next((row for row in summaries if row["candidate"] == "SURPRISE_ONLY_alpha1_eta0.02"), {})
    stable = [row["candidate"] for row in summaries if row["phase"] == "phase_e" and row["classification"] == "STABLE"]
    lines = [
        "# HOPE Task 3R Gate 2C Update Stabilization",
        "",
        "This is an experiment-only report. Production SMT, CMS, and HOPE files were not modified.",
        "",
        "## Oracle and setup",
        "",
        f"- Device: `{device}`; horizon `H={HORIZON_TOKENS}`; stream: 40 bottle then 10 carpet normal train/good images.",
        f"- PM0 oracle passed: `{oracle.get('passed')}`; production output max error `{oracle.get('production_output_max_abs')}`.",
        f"- Runtime: `{duration:.3f}` seconds.",
        "- All alternative mappings are PROJECT-MAPPING experiments; no anomaly objective or test image was used.",
        "",
        "## Phase A term decomposition",
        "",
        f"Term rows: `{term_summary.get('term_rows', 0)}`. Mean/max diagnostics: `{json.dumps(_json_safe(term_summary), sort_keys=True)}`.",
        f"- FULL final classification over its available forensic stream: `{full.get('classification')}`.",
        f"- RANK_ONLY: `{rank.get('classification')}`; SURPRISE_ONLY: `{surprise.get('classification')}`.",
        "- The dominant term is determined from aggregate delta norms/ratios in `term_decomposition.parquet`; the report does not infer it from state norms alone.",
        "",
        "## Candidate summaries",
        "",
        _format_md_table(list(summaries), ["phase", "candidate", "classification", "complete_event_count", "final_memory_norm_rel_event1", "final_smt_rms_rel_event1", "final_smt_variance_rel_event1", "final_smt_rank_rel_event1", "final_hope_rms_rel_event1", "final_hope_variance_rel_event1"]),
        "",
        "## Phase conclusions",
        "",
        f"- Horizon-normalized eta candidates completed: `{[(r['candidate'], r['complete_event_count'], r['classification']) for r in phase_b_results]}`.",
        f"- Learned/fixed/frozen eta diagnostics: `{[(r['candidate'], r['complete_event_count'], r['classification']) for r in phase_c_results]}`.",
        f"- HNR reintroduction diagnostics: `{[(r['candidate'], r['complete_event_count'], r['classification']) for r in phase_d_results]}`.",
        f"- 50-image candidates: `{[(r['candidate'], r['complete_event_count'], r['classification']) for r in phase_e_results]}`.",
        f"- Term-isolation result: rank-only `{rank.get('classification')}`, surprise-only `{surprise.get('classification')}`, full `{full.get('classification')}`. Compare `delta_rank_*` and `delta_surprise_*` columns for the causal conclusion.",
        f"- Clipping was not implemented or required by this gate: `{not bool(phase_e_results and any(r['classification'] == 'EXPLODING' for r in phase_e_results))}`.",
        f"- PATCH-HORIZON SCALING HYPOTHESIS is supported when HNP candidates avoid the prior event-14 failure while preserving geometry; final evidence is recorded in the tables above.",
        "",
        "## Gate 3 decision",
        "",
        f"Stable Gate-3 candidates: `{stable}`.",
        f"`GATE3_BLOCKED = {'NO' if stable else 'YES'}`. No 200-image run was executed.",
        "",
        "## Limitations",
        "",
        "This gate isolates update-side numerical behavior only. It does not establish anomaly performance, a production objective, or paper-level status for any mapping.",
        "",
        "## Primary experiment labels",
        "",
        "- PM0: canonical failed project mapping control.",
        "- HNP: horizon-normalized plasticity project mapping.",
        "- HNR: horizon-normalized retention project mapping.",
        "- RANK_ONLY / SURPRISE_ONLY / FIXED_ETA / FROZEN_M_ETA: diagnostic counterfactuals.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run_gate2c(seed: int, device: str) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(seed)
    if device == "cpu":
        torch.set_num_threads(1)
    records = load_cached_stream(FEATURE_ROOT, count=50)
    smt_state, cms_state = canonical_initial_states(seed, 768)
    oracle = pm0_oracle_check(records[0]["patches"].to(device), smt_state)
    if not oracle["passed"]:
        raise RuntimeError(f"PM0 oracle failed: {oracle}")
    references = build_read_only_rms_reference(records, smt_state, cms_state, device)
    phase_a = phase_a_mappings()
    term_module = __import__("exps.hope_retention_stabilization", fromlist=["clone_smt_from_state"]).clone_smt_from_state(smt_state, 768, device)
    _, _, term_rows = run_update_smt(term_module, records[0]["patches"].to(device), phase_a[0], event_id=1, capture_trace=True, capture_terms=True)
    phase_a_results, phase_a_event_rows = _run_phase("phase_a", phase_a, records, smt_state, cms_state, device, references, max_events=14)
    phase_b_results, phase_b_event_rows = _run_phase("phase_b", phase_b_mappings(), records, smt_state, cms_state, device, references, max_events=14)
    eligible_b = [r for r in phase_b_results if r["complete_event_count"] >= 14 and r["all_finite"]]
    eligible_b.sort(key=lambda result: result["mapping"].eta_h)
    selected_eta = [result["mapping"].eta_h for result in eligible_b[:2]]
    phase_c_mappings = tuple(mapping for eta_h in selected_eta for mapping in (phase_c_mapping(eta_h, "learned"), phase_c_mapping(eta_h, "fixed"), phase_c_mapping(eta_h, "frozen")))
    phase_c_results, phase_c_event_rows = _run_phase("phase_c", phase_c_mappings, records, smt_state, cms_state, device, references, max_events=14)
    phase_d_mappings = tuple(phase_d_mapping(eta_h, lambda_h) for eta_h in selected_eta for lambda_h in (0.002, 0.005, 0.010))
    phase_d_results, phase_d_event_rows = _run_phase("phase_d", phase_d_mappings, records, smt_state, cms_state, device, references, max_events=14)
    phase_c_learned = [r for r in phase_c_results if "learned" in r["candidate"] and r["complete_event_count"] >= 14 and r["all_finite"]]
    phase_d_complete = [r for r in phase_d_results if r["complete_event_count"] >= 14 and r["all_finite"]]
    phase_c_learned.sort(key=lambda result: result["mapping"].eta_h)
    phase_d_complete.sort(key=lambda result: result["mapping"].lambda_h)
    selected_e_mappings: list[UpdateMapping] = []
    if phase_c_learned:
        selected_e_mappings.append(phase_c_learned[0]["mapping"])
    if phase_d_complete:
        selected_e_mappings.append(phase_d_complete[0]["mapping"])
        if len(phase_d_complete) > 1:
            selected_e_mappings.append(phase_d_complete[1]["mapping"])
    unique_e: list[UpdateMapping] = []
    seen: set[str] = set()
    for mapping in selected_e_mappings:
        if mapping.name not in seen:
            unique_e.append(mapping)
            seen.add(mapping.name)
    phase_e_results, phase_e_event_rows = _run_phase("phase_e", unique_e[:3], records, smt_state, cms_state, device, references, max_events=50)
    all_results = phase_a_results + phase_b_results + phase_c_results + phase_d_results + phase_e_results
    summaries = [_summary_row(result, result["phase"]) for result in all_results]
    return {
        "seed": seed,
        "device": device,
        "records": records,
        "oracle": oracle,
        "term_rows": term_rows,
        "phase_a_results": phase_a_results,
        "phase_b_results": phase_b_results,
        "phase_c_results": phase_c_results,
        "phase_d_results": phase_d_results,
        "phase_e_results": phase_e_results,
        "phase_a_event_rows": phase_a_event_rows,
        "phase_b_event_rows": phase_b_event_rows,
        "phase_c_event_rows": phase_c_event_rows,
        "phase_d_event_rows": phase_d_event_rows,
        "phase_e_event_rows": phase_e_event_rows,
        "summaries": summaries,
        "selected_eta_h": selected_eta,
    }


def build_config(seed: int, device: str) -> dict[str, Any]:
    return {
        "seed": seed,
        "device": device,
        "stage": "gate2c",
        "horizon_tokens": HORIZON_TOKENS,
        "stream": "first 40 bottle normal train/good then first 10 carpet normal train/good; no reset",
        "feature_root": str(FEATURE_ROOT),
        "smt": {"adaptive_q": False, "memory_chunk_size": 16, "auxiliary_memory_chunk_size": 16, "canonical_eq93_unchanged": True},
        "cms": {"K": 2, "update_periods": [1, 8], "objective": "Candidate-B probe-only identity objective"},
        "phase_a": [{"name": mapping.name, "mapping": _mapping_config(mapping)} for mapping in phase_a_mappings()],
        "phase_b_eta_h": [0.02, 0.10, 0.50, 1.00],
        "phase_d_lambda_h": [0.002, 0.005, 0.010],
        "effective_rank": "full 784 rows; row-centering; entropy effective rank; eps=1e-12",
        "classification": "finite/bounded representation viability only; no anomaly metrics",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gate2c",), default="gate2c")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args(argv)
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    for class_name in ("class_bottle.pt", "class_carpet.pt"):
        if not (FEATURE_ROOT / class_name).is_file():
            parser.error(f"missing feature cache shard: {FEATURE_ROOT / class_name}")
    output = OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    config = build_config(args.seed, device)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    started = time.perf_counter()
    payload = run_gate2c(args.seed, device)
    duration = time.perf_counter() - started
    config["duration_seconds"] = duration
    config["selected_eta_h"] = payload["selected_eta_h"]
    config["phase_c_mappings"] = [_mapping_config(result["mapping"]) for result in payload["phase_c_results"]]
    config["phase_d_mappings"] = [_mapping_config(result["mapping"]) for result in payload["phase_d_results"]]
    config["phase_e_mappings"] = [_mapping_config(result["mapping"]) for result in payload["phase_e_results"]]
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "oracle_equivalence.json").write_text(json.dumps(_json_safe(payload["oracle"]), indent=2) + "\n")
    write_table(output / "term_decomposition.parquet", payload["term_rows"])
    all_forensic = []
    for result in payload["phase_a_results"] + payload["phase_b_results"] + payload["phase_c_results"] + payload["phase_d_results"]:
        all_forensic.append(_summary_row(result, result["phase"]))
    write_table(output / "forensic_candidates.parquet", all_forensic)
    short14 = payload["phase_a_event_rows"] + payload["phase_b_event_rows"] + payload["phase_c_event_rows"] + payload["phase_d_event_rows"]
    write_table(output / "short14_per_event.parquet", short14)
    write_table(output / "short50_per_event.parquet", payload["phase_e_event_rows"])
    summary = {
        "status": "GATE2C_COMPLETE",
        "run_id": "gate2c_seed0",
        "duration_seconds": duration,
        "oracle_equivalence": payload["oracle"],
        "selected_eta_h": payload["selected_eta_h"],
        "candidate_summaries": payload["summaries"],
        "artifacts": {name: str(output / name) for name in ("config_resolved.yaml", "oracle_equivalence.json", "term_decomposition.parquet", "forensic_candidates.parquet", "short14_per_event.parquet", "short50_per_event.parquet")},
        "gate3_blocked": not any(row["classification"] == "STABLE" for row in payload["summaries"] if row["phase"] == "phase_e"),
    }
    (output / "candidate_summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n")
    _write_report(
        REPO_ROOT / "agents/reports/hope_task3r_gate2c_update_stabilization.md",
        oracle=payload["oracle"], term_rows=payload["term_rows"], summaries=payload["summaries"],
        phase_a_results=payload["phase_a_results"], phase_b_results=payload["phase_b_results"],
        phase_c_results=payload["phase_c_results"], phase_d_results=payload["phase_d_results"],
        phase_e_results=payload["phase_e_results"], duration=duration, device=device,
    )
    print(json.dumps(_json_safe({"output": str(output), "duration_seconds": duration, "oracle": payload["oracle"], "summaries": payload["summaries"]}), indent=2), flush=True)
    print("TASK3R_GATE2C_COMPLETE = YES", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
