# scripts/exps/hope_retention_stabilization.py
"""CLI for isolated HOPE retention/control viability experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from exps.hope_retention_stabilization import (
    RetentionMapping,
    canonical_initial_states,
    canonical_oracle,
    candidate_mappings,
    clone_smt_from_state,
    GATE2_CANDIDATE_NAMES,
    GATE2B_CANDIDATE_NAMES,
    HORIZON_TOKENS,
    gate2_mappings,
    gate2b_mappings,
    build_read_only_rms_reference,
    load_cached_stream,
    load_bottle_event,
    run_candidate,
    run_experimental_smt,
    run_stream_candidate,
    STREAM_CHECKPOINTS,
    tensor_geometry,
    write_table,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "results/hope_cad/real_feature_probe/real_cpu_seed0/features/class_bottle.pt"
OUTPUT_ROOT = REPO_ROOT / "results/hope_cad/retention_stabilization/gate1_seed0"
FEATURE_ROOT = REPO_ROOT / "results/hope_cad/real_feature_probe/real_cpu_seed0/features"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _trace_rows(result: dict) -> list[dict]:
    rows = []
    for row in result["trace"]:
        rows.append({key: value for key, value in row.items() if not isinstance(value, torch.Tensor)})
    return rows


def run_one_image(seed: int, device_name: str) -> dict:
    del device_name
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    x, relative_path = load_bottle_event(CACHE_PATH)
    smt_state, cms_state = canonical_initial_states(seed, x.shape[-1])
    # The experiment is intentionally CPU for this cached forensic Gate 1.
    x = x.cpu()
    canonical = clone_smt_from_state(smt_state, x.shape[-1])
    read_only = canonical.forward(x, update=False).memory_prediction
    read_only_geometry = tensor_geometry(read_only)
    base_geometry = tensor_geometry(x)
    results = []
    traces = []
    oracle_equivalence = None
    started = time.perf_counter()
    for index, mapping in enumerate(candidate_mappings()):
        candidate_started = time.perf_counter()
        result = run_candidate(x, mapping, smt_state, cms_state)
        result["runtime_seconds"] = time.perf_counter() - candidate_started
        results.append(result)
        traces.extend(_trace_rows(result))
        if index == 0:
            oracle_equivalence = canonical_oracle(x, smt_state, result)
            if not (oracle_equivalence["output_equal"] and oracle_equivalence["state_equal"] and oracle_equivalence["counters_equal"]):
                raise RuntimeError(f"PM0 oracle equivalence failed: {oracle_equivalence}")
    duration = time.perf_counter() - started
    candidate_rows = []
    for result in results:
        row = {
            "relative_path": relative_path,
            "input_geometry": base_geometry,
            "read_only_smt_geometry": read_only_geometry,
            "runtime_seconds": result["runtime_seconds"],
        }
        from exps.hope_retention_stabilization import flatten_candidate_row
        row.update(flatten_candidate_row(result, x))
        candidate_rows.append(row)
    return {
        "seed": seed,
        "input_shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "relative_path": relative_path,
        "candidate_count": len(results),
        "duration_seconds": duration,
        "read_only_geometry": read_only_geometry,
        "input_geometry": base_geometry,
        "oracle_equivalence": oracle_equivalence,
        "candidate_rows": candidate_rows,
        "trace_rows": traces,
        "results": results,
    }


def build_config(seed: int, device: str) -> dict:
    return {
        "seed": seed,
        "device": device,
        "stage": "one-image",
        "input": {
            "cache": str(CACHE_PATH),
            "class": "bottle",
            "event": 1,
            "shape": [1, 784, 768],
            "selection": "first lexicographic cached bottle train/good image",
        },
        "smt": {
            "adaptive_q": False,
            "memory_chunk_size": 16,
            "auxiliary_memory_chunk_size": 16,
            "control_recurrence": "canonical Eq.93; only eta/alpha mapping varies",
        },
        "cms": {"K": 2, "update_periods": [1, 8], "hidden_dim": 768, "objective": "Candidate-B probe-only project mapping: 0.5 * mean((y-stop_gradient(h))^2)"},
        "candidates": [mapping.__dict__ for mapping in candidate_mappings()],
        "r4_cleanly_identifiable": False,
        "r4_reason": "Bias-free M_alpha cannot be shifted to a target alpha operating point by a state-only, data-independent initialization edit without defining a new mapping; no R4 candidate was run.",
        "effective_rank": "all 784 rows, centered, entropy effective rank, eps=1e-12",
        "classification": "numerical/representation viability only; no anomaly metrics",
    }


def build_gate2_config(seed: int, device: str) -> dict:
    return {
        "seed": seed,
        "device": device,
        "stage": "short50",
        "input": {
            "feature_root": str(FEATURE_ROOT),
            "classes": ["bottle", "carpet"],
            "selection": "first 40 bottle events then first 10 carpet events from deterministic Task-3A cache",
            "events": 50,
            "no_class_boundary_reset": True,
        },
        "smt": {
            "adaptive_q": False,
            "memory_chunk_size": 16,
            "auxiliary_memory_chunk_size": 16,
            "control_recurrence": "canonical Eq.93; only locked Gate-2 eta/alpha mappings vary",
        },
        "cms": {
            "K": 2,
            "update_periods": [1, 8],
            "hidden_dim": 768,
            "objective": "Candidate-B probe-only project mapping: 0.5 * mean((y-stop_gradient(h))^2)",
        },
        "candidates": list(GATE2_CANDIDATE_NAMES),
        "relative_baseline": "event 1 post-commit state and geometry",
        "classification_rules": {
            "exploding": "non-finite/incomplete event or non-finite persistent state",
            "collapsed": "zero fast state",
            "decaying": "finite non-collapsed but final geometry rank < 0.5 or variance < 0.25 of event 1",
            "stable": "finite, non-collapsed, and geometry above decay criteria",
        },
        "effective_rank": "all 784 rows, centered, entropy effective rank, eps=1e-12",
        "classification": "normal-stream numerical/representation viability only; no anomaly metrics",
    }


def build_gate2b_config(seed: int, device: str) -> dict:
    return {
        "seed": seed,
        "device": device,
        "stage": "horizon50",
        "mapping_label": "HORIZON-NORMALIZED PROJECT-MAPPING EXPERIMENT",
        "horizon_tokens": HORIZON_TOKENS,
        "input": {
            "feature_root": str(FEATURE_ROOT),
            "classes": ["bottle", "carpet"],
            "selection": "first 40 bottle events then first 10 carpet events from deterministic Task-3A cache",
            "events": 50,
            "no_class_boundary_reset": True,
        },
        "smt": {
            "adaptive_q": False,
            "memory_chunk_size": 16,
            "auxiliary_memory_chunk_size": 16,
            "control_recurrence": "canonical Eq.93; only Gate-2B eta/alpha mappings vary",
        },
        "cms": {
            "K": 2,
            "update_periods": [1, 8],
            "hidden_dim": 768,
            "objective": "Candidate-B probe-only project mapping: 0.5 * mean((y-stop_gradient(h))^2)",
        },
        "candidates": list(GATE2B_CANDIDATE_NAMES),
        "candidate_mappings": [mapping.__dict__ for mapping in gate2b_mappings()],
        "optional_eta01_sensitivity": "not run; five mandatory candidates only",
        "neutral_retention": "alpha=(1-lambda_h/H*sigmoid(raw_alpha)); raw_alpha=0 gives sigmoid=0.5",
        "relative_baseline": "event 1 post-commit state and geometry",
        "classification_rules": {
            "exploding": "non-finite/incomplete event or non-finite persistent state",
            "collapsed": "zero fast state",
            "decaying": "finite nonzero state but final RMS ratio <0.5, variance ratio <0.25, or rank ratio <0.5 for SMT or HOPE",
            "inconclusive": "schema/graph violation, >10x RMS growth, or sampled cosine mean >0.999 without a hard numerical failure",
            "stable": "complete finite stream, nonzero fixed-schema state, no history, and geometry passes the stated viability screens",
        },
        "effective_rank": "all 784 rows, centered, entropy effective rank, eps=1e-12",
        "diagnostic_precision": "float64 detached reductions; exact all-row SVD on CPU to avoid CUDA Jacobi fallback",
        "scientific_dtype": "torch.float32; unchanged recurrence",
        "cosine_pairs": "first 256 patches (i,(i+1) modulo 256); mean and population std; eps=1e-12",
        "reference": "same event input through identical fresh SMT(update=False) and CMS.forward; computed once and shared across candidates",
        "checkpoint_events": list(STREAM_CHECKPOINTS),
        "nonfinite_policy": "stop each candidate on non-finite state; retain its partial failed event; never substitute it for event 50",
        "retention_reporting": "float64 products and natural logs; cumulative products include the recorded prefix of a failed image",
        "classification": "normal-stream numerical/representation viability only; no anomaly metrics",
    }


def run_gate2(seed: int, device_name: str) -> dict:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("Gate 2 requested CUDA but it is unavailable")
    if device_name == "cpu":
        torch.set_num_threads(1)
    torch.manual_seed(seed)
    records = load_cached_stream(FEATURE_ROOT, count=50)
    smt_state, cms_state = canonical_initial_states(seed, 768)
    started = time.perf_counter()
    candidate_results = []
    event_rows = []
    for mapping in gate2_mappings():
        result = run_stream_candidate(records, mapping, smt_state, cms_state, device_name)
        candidate_results.append(result)
        event_rows.extend(result["rows"])
    candidates = []
    for result in candidate_results:
        rows = result["rows"]
        candidates.append({
            "candidate": result["candidate"],
            "classification": result["classification"],
            "event_count": result["event_count"],
            "failure": result["failure"],
            "state_bytes_constant": result["state_bytes_constant"],
            "state_schema_constant": result["state_schema_constant"],
            "all_finite": result["all_finite"],
            "first_event": rows[0] if rows else {},
            "last_event": rows[-1] if rows else {},
            "boundary_events": [row for row in rows if row["is_class_boundary"]],
        })
    return {
        "seed": seed,
        "device": device_name,
        "duration_seconds": time.perf_counter() - started,
        "record_count": len(records),
        "class_counts": {"bottle": 40, "carpet": 10},
        "candidate_results": candidate_results,
        "candidate_summaries": candidates,
        "event_rows": event_rows,
    }


def _mapping_retention_summary(mapping: RetentionMapping) -> dict[str, float | int | None]:
    if mapping.alpha_kind == "horizon_near_one":
        neutral_alpha = 1.0 - (mapping.alpha_scale / mapping.horizon) * 0.5
        return {
            "horizon_tokens": mapping.horizon,
            "lambda_h": mapping.alpha_scale,
            "neutral_alpha": neutral_alpha,
            "neutral_retention_16": neutral_alpha ** 16,
            "neutral_retention_784": neutral_alpha ** mapping.horizon,
            "neutral_retention_50_images": neutral_alpha ** (mapping.horizon * 50),
        }
    if mapping.alpha_kind == "fixed_one":
        return {
            "horizon_tokens": mapping.horizon,
            "lambda_h": 0.0,
            "neutral_alpha": 1.0,
            "neutral_retention_16": 1.0,
            "neutral_retention_784": 1.0,
            "neutral_retention_50_images": 1.0,
        }
    return {
        "horizon_tokens": mapping.horizon,
        "lambda_h": None,
        "neutral_alpha": None,
        "neutral_retention_16": None,
        "neutral_retention_784": None,
        "neutral_retention_50_images": None,
    }


def run_gate2b(seed: int, device_name: str) -> dict:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("Gate 2B requested CUDA but it is unavailable")
    torch.set_num_threads(1)
    started = time.perf_counter()
    torch.manual_seed(seed)
    records = load_cached_stream(FEATURE_ROOT, count=50)
    smt_state, cms_state = canonical_initial_states(seed, 768)
    references = build_read_only_rms_reference(records, smt_state, cms_state, device_name)
    oracle_image = records[0]["patches"].to(device_name)
    oracle_smt = clone_smt_from_state(smt_state, 768, device_name)
    oracle_output, _ = run_experimental_smt(oracle_smt, oracle_image, RetentionMapping("PM0"), capture_trace=False)
    oracle_equivalence = canonical_oracle(oracle_image, smt_state, {"representation": oracle_output, "smt_state": oracle_smt})
    if not all(oracle_equivalence[key] for key in ("output_equal", "state_equal", "counters_equal", "chunk_boundaries_equal")):
        raise RuntimeError(f"PM0 oracle equivalence failed: {oracle_equivalence}")
    print(f"PM0 oracle passed on {device_name}: max output difference={oracle_equivalence['output_max_abs']}", flush=True)
    candidate_results = []
    event_rows = []
    for mapping in gate2b_mappings():
        result = run_stream_candidate(
            records, mapping, smt_state, cms_state, device_name, references, progress=True,
        )
        candidate_results.append(result)
        event_rows.extend(result["rows"])
    candidates = []
    for result in candidate_results:
        rows = result["rows"]
        first = rows[0] if rows else {}
        last = rows[-1] if rows else {}
        event50 = next((row for row in rows if row["event_id"] == 50 and row["complete_event"]), {})
        last_completed = next((row for row in reversed(rows) if row["complete_event"]), {})
        retention = _mapping_retention_summary(result["mapping"])
        candidates.append({
            "candidate": result["candidate"],
            "alpha_kind": result["mapping"].alpha_kind,
            "alpha_scale": result["mapping"].alpha_scale,
            "eta_kind": result["mapping"].eta_kind,
            "eta_scale": result["mapping"].eta_scale,
            **retention,
            "classification": result["classification"],
            "event_count": result["event_count"],
            "completed_event_count": sum(row["complete_event"] for row in rows),
            "last_recorded_event": last.get("event_id"),
            "failure_event": last.get("event_id") if result["failure"] else None,
            "failure_token": last.get("smt_failure_token"),
            "failure": result["failure"],
            "state_bytes_constant": result["state_bytes_constant"],
            "state_schema_constant": result["state_schema_constant"],
            "all_finite": result["all_finite"],
            "alpha_product_event1": first.get("alpha_product_observed"),
            "alpha_product_event50": event50.get("alpha_product_observed"),
            "alpha_product_cumulative_event50": event50.get("alpha_product_cumulative"),
            "alpha_product_cumulative_last_recorded": last.get("alpha_product_cumulative"),
            "alpha_log_product_cumulative_last_recorded": last.get("alpha_log_product_cumulative"),
            "memory_norm_final": event50.get("memory_norm"),
            "memory_norm_rel_event1": event50.get("memory_norm_rel_event1"),
            "smt_rms_final": event50.get("smt_rms"),
            "smt_rms_rel_event1": event50.get("smt_rms_rel_event1"),
            "smt_variance_final": event50.get("smt_centered_variance"),
            "smt_variance_rel_event1": event50.get("smt_centered_variance_rel_event1"),
            "smt_rank_final": event50.get("smt_effective_rank"),
            "smt_rank_rel_event1": event50.get("smt_effective_rank_rel_event1"),
            "hope_rms_final": event50.get("hope_rms"),
            "hope_rms_rel_event1": event50.get("hope_rms_rel_event1"),
            "hope_variance_final": event50.get("hope_centered_variance"),
            "hope_variance_rel_event1": event50.get("hope_centered_variance_rel_event1"),
            "hope_rank_final": event50.get("hope_effective_rank"),
            "hope_rank_rel_event1": event50.get("hope_effective_rank_rel_event1"),
            "first_event": first,
            "last_event": last,
            "last_completed_event": last_completed,
            "checkpoints": [row for row in rows if row["event_id"] in STREAM_CHECKPOINTS],
            "unreached_checkpoint_events": [event for event in STREAM_CHECKPOINTS if not any(row["event_id"] == event for row in rows)],
            "boundary_events": [row for row in rows if row["is_class_boundary"]],
        })
    return {
        "seed": seed,
        "device": device_name,
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device_name == "cuda" else None,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "duration_seconds": time.perf_counter() - started,
        "record_count": len(records),
        "class_counts": {"bottle": 40, "carpet": 10},
        "oracle_equivalence": oracle_equivalence,
        "candidate_results": candidate_results,
        "candidate_summaries": candidates,
        "event_rows": event_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("one-image", "short50", "horizon50", "full200"), default="one-image")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args(argv)
    if args.stage == "full200":
        parser.error("Gate 3/full200 is reserved and must not run in Gate 2")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.stage == "short50":
        if not (FEATURE_ROOT / "class_bottle.pt").is_file() or not (FEATURE_ROOT / "class_carpet.pt").is_file():
            parser.error(f"bottle/carpet feature shards not found under {FEATURE_ROOT}")
        output_root = REPO_ROOT / "results/hope_cad/retention_stabilization/gate2_seed0"
        device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
        config = build_gate2_config(args.seed, device)
        payload = run_gate2(args.seed, device)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        write_table(output_root / "per_event.parquet", payload["event_rows"])
        write_table(output_root / "candidates.parquet", payload["candidate_summaries"])
        summary = {
            "status": "GATE2_COMPLETE",
            "run_id": "gate2_seed0",
            "config": config,
            "duration_seconds": payload["duration_seconds"],
            "record_count": payload["record_count"],
            "class_counts": payload["class_counts"],
            "candidates": payload["candidate_summaries"],
            "artifacts": {
                "per_event": str(output_root / "per_event.parquet"),
                "candidates": str(output_root / "candidates.parquet"),
            },
        }
        (output_root / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n")
        print(json.dumps(_json_safe({"output": str(output_root), "candidates": payload["candidate_summaries"]}), indent=2))
        return 0

    if args.stage == "horizon50":
        if not (FEATURE_ROOT / "class_bottle.pt").is_file() or not (FEATURE_ROOT / "class_carpet.pt").is_file():
            parser.error(f"bottle/carpet feature shards not found under {FEATURE_ROOT}")
        output_root = REPO_ROOT / "results/hope_cad/retention_stabilization/gate2b_seed0"
        device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
        config = build_gate2b_config(args.seed, device)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        payload = run_gate2b(args.seed, device)
        write_table(output_root / "per_event.parquet", payload["event_rows"])
        write_table(output_root / "candidates.parquet", payload["candidate_summaries"])
        summary = {
            "status": "GATE2B_COMPLETE",
            "run_id": "gate2b_seed0",
            "config": config,
            "duration_seconds": payload["duration_seconds"],
            "record_count": payload["record_count"],
            "class_counts": payload["class_counts"],
            "oracle_equivalence": payload["oracle_equivalence"],
            "environment": payload["environment"],
            "candidate_failures_observed": any(candidate["failure"] for candidate in payload["candidate_summaries"]),
            "candidates": payload["candidate_summaries"],
            "artifacts": {
                "per_event": str(output_root / "per_event.parquet"),
                "candidates": str(output_root / "candidates.parquet"),
            },
        }
        (output_root / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n")
        print(json.dumps(_json_safe({"output": str(output_root), "oracle": payload["oracle_equivalence"], "candidates": [{key: row[key] for key in ("candidate", "classification", "completed_event_count", "failure_event", "failure_token")} for row in payload["candidate_summaries"]]}), indent=2))
        return 0

    if not CACHE_PATH.is_file():
        parser.error(f"cached bottle feature not found: {CACHE_PATH}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config = build_config(args.seed, "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    (OUTPUT_ROOT / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    payload = run_one_image(args.seed, config["device"])
    (OUTPUT_ROOT / "candidates.parquet").unlink(missing_ok=True)
    (OUTPUT_ROOT / "trace.parquet").unlink(missing_ok=True)
    write_table(OUTPUT_ROOT / "candidates.parquet", payload["candidate_rows"])
    write_table(OUTPUT_ROOT / "trace.parquet", payload["trace_rows"])
    summary = {
        "status": "PASS_ORACLE_AND_RUN",
        "run_id": "gate1_seed0",
        "config": config,
        "oracle_equivalence": payload["oracle_equivalence"],
        "candidate_count": payload["candidate_count"],
        "duration_seconds": payload["duration_seconds"],
        "input_geometry": payload["input_geometry"],
        "read_only_smt_geometry": payload["read_only_geometry"],
        "candidates": payload["candidate_rows"],
        "r4_cleanly_identifiable": False,
        "r4_reason": config["r4_reason"],
        "artifacts": {"candidates": str(OUTPUT_ROOT / "candidates.parquet"), "trace": str(OUTPUT_ROOT / "trace.parquet")},
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n")
    print(json.dumps(_json_safe({"output": str(OUTPUT_ROOT), "oracle": payload["oracle_equivalence"], "candidates": payload["candidate_rows"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
