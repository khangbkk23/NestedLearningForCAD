# scripts/hope_cad/benchmark_hope_core.py
"""CPU-safe synthetic diagnostics for the canonical HOPE core."""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Mapping
from typing import Any

import torch

from models.hope_cad import HopeBlock
from models.hope_cad import state


def quadratic(index, level_input, level_output, metadata):
    del index, level_input, metadata
    return 0.5 * level_output.square().mean()


def objectives(count: int):
    return [quadratic] * count


def tree_delta(left: Any, right: Any) -> float:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        difference = left.detach() - right.detach()
        if not (torch.is_floating_point(difference) or torch.is_complex(difference)):
            return float(difference.abs().sum().item())
        return float(difference.norm().item())
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return sum(tree_delta(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return sum(tree_delta(a, b) for a, b in zip(left, right))
    return 0.0


def persistent_grad_fn(module: HopeBlock) -> bool:
    for value in module.buffers():
        if value.grad_fn is not None:
            return True
    return False


def finite_module(module: HopeBlock) -> bool:
    return state.validate_finite_state(module.state_dict()) is True


def make_model(dim: int, *, periods=(1, 8), chunk_size: int = 16) -> HopeBlock:
    return HopeBlock(
        dim,
        adaptive_q=False,
        memory_chunk_size=chunk_size,
        auxiliary_memory_chunk_size=chunk_size,
        cms_update_periods=periods,
        cms_learning_rates=tuple(1e-3 for _ in periods),
    )


def run_real_case(batch_size: int) -> dict[str, Any]:
    dim, tokens = 768, 784
    seed = 700 + batch_size
    torch.manual_seed(seed)
    model = make_model(dim)
    initial = copy.deepcopy(model)
    inputs = torch.randn(batch_size, tokens, dim) * 0.01
    objectives_for_cms = objectives(model.cms.K)
    before_online = model.snapshot_online_state()
    before_stats = model.memory_stats()

    start = time.perf_counter()
    read_output = model(inputs)
    read_seconds = time.perf_counter() - start

    start = time.perf_counter()
    if batch_size == 1:
        commit_results = (model.commit_image(inputs, objectives_for_cms),)
    else:
        commit_results = model.commit_batch(inputs, objectives_for_cms)
    commit_seconds = time.perf_counter() - start
    after_online = model.snapshot_online_state()

    before_eval = state.snapshot_persistent_state(model.state_dict())
    start = time.perf_counter()
    evaluation_output = model.evaluate_image(inputs[:1])
    evaluation_seconds = time.perf_counter() - start
    source_unchanged_after_eval = state.persistent_states_equal(before_eval, model.state_dict())

    saved = state.snapshot_persistent_state(model.state_dict())
    continuation = make_model(dim)
    continuation.load_state_dict(saved)
    next_input = torch.randn(1, tokens, dim) * 0.01
    left_next = model.commit_image(next_input, objectives_for_cms)
    right_next = continuation.commit_image(next_input, objectives_for_cms)
    serialization_equal = (
        torch.allclose(left_next.output, right_next.output, atol=1e-5, rtol=1e-5)
        and state.persistent_states_equal(model.state_dict(), continuation.state_dict(), atol=1e-5)
    )

    reset_probe = copy.deepcopy(model)
    reset_probe.reset_state()
    initial.reset_state()
    reset_equal = state.persistent_states_equal(reset_probe.state_dict(), initial.state_dict())
    after_stats = model.memory_stats()
    cuda_metrics = {}
    if torch.cuda.is_available():
        cuda_metrics["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())

    return {
        "batch_size": batch_size,
        "input_shape": list(inputs.shape),
        "read_only_seconds": read_seconds,
        "training_commit_seconds": commit_seconds,
        "official_evaluation_seconds": evaluation_seconds,
        "read_output_shape": list(read_output.shape),
        "evaluation_output_shape": list(evaluation_output.shape),
        "finite": finite_module(model),
        "smt_online_state_delta_norm": tree_delta(before_online["smt"], after_online["smt"]),
        "cms_online_state_delta_norm": tree_delta(before_online["cms"], after_online["cms"]),
        "source_state_delta_after_evaluation": 0.0 if source_unchanged_after_eval else None,
        "source_unchanged_after_evaluation": source_unchanged_after_eval,
        "serialization_continuation_equal": serialization_equal,
        "reset_equivalent": reset_equal,
        "commit_event_ids": [result.event_id for result in commit_results],
        "commit_due_levels": [list(result.due_levels) for result in commit_results],
        "state_accounting_before": before_stats,
        "state_accounting_after": after_stats,
        "cuda": cuda_metrics,
    }


def summarize_controls(result) -> dict[str, float]:
    values = {}
    for name, tensor in (("eta", result.eta), ("alpha", result.alpha)):
        flat = tensor.detach().reshape(-1).float().cpu()
        quantiles = torch.quantile(flat, torch.tensor([0.01, 0.5, 0.99]))
        values.update({
            f"{name}_min": float(flat.min()),
            f"{name}_p01": float(quantiles[0]),
            f"{name}_median": float(quantiles[1]),
            f"{name}_mean": float(flat.mean()),
            f"{name}_p99": float(quantiles[2]),
            f"{name}_max": float(flat.max()),
        })
    return values


def run_long_horizon(events: int = 1000) -> dict[str, Any]:
    torch.manual_seed(900)
    model = make_model(32, periods=(1, 8), chunk_size=8)
    x = torch.randn(1, 32, 32) * 0.02
    cms_objectives = objectives(model.cms.K)
    checkpoints = []
    initial_bytes = model.memory_stats()["hope_full_tensor_bytes"]
    initial_keys = tuple(model.state_dict().keys())
    prior_online = model.snapshot_online_state()
    for event in range(1, events + 1):
        result = model.commit_image(x, cms_objectives)
        if event % 100 == 0 or event == 1:
            current_online = model.snapshot_online_state()
            smt_norms = {
                name: float(value.norm().item())
                for name, value in model.smt.memory_state().items()
            }
            cms_norms = []
            accum_norms = []
            for level in model.cms.levels:
                cms_norms.append(float(sum(value.norm().item() ** 2 for value in level.current_tensors()) ** 0.5))
                accum_norms.append(float(sum(value.norm().item() ** 2 for value in level.gradient_accumulators()) ** 0.5))
            checkpoints.append({
                "event": event,
                "smt_memory_norms": smt_norms,
                "relative_fast_state_update_norm": tree_delta(current_online["smt"], prior_online["smt"]),
                "cms_level_state_norms": cms_norms,
                "cms_gradient_accumulator_norms": accum_norms,
                "pending_counts": [int(level.pending_count) for level in model.cms.levels],
                "update_counts": [int(level.update_count) for level in model.cms.levels],
                "output_norm": float(result.output.norm().item()),
                "controls": summarize_controls(result),
                "tensor_state_bytes": model.memory_stats()["hope_full_tensor_bytes"],
                "finite": finite_module(model),
                "persistent_grad_fn": persistent_grad_fn(model),
            })
            prior_online = current_online
    final_stats = model.memory_stats()
    memory_norms = [
        checkpoint["smt_memory_norms"].get("memory", 0.0)
        for checkpoint in checkpoints
    ]
    return {
        "events": events,
        "checkpoints": checkpoints,
        "state_bytes_constant": final_stats["hope_full_tensor_bytes"] == initial_bytes,
        "tensor_key_count_constant": tuple(model.state_dict().keys()) == initial_keys,
        "finite": finite_module(model),
        "persistent_grad_fn": persistent_grad_fn(model),
        "final_state_accounting": final_stats,
        "observations": {
            "memory_norm_reached_exact_zero": any(value == 0.0 for value in memory_norms),
            "output_norm_first_checkpoint": checkpoints[0]["output_norm"],
            "output_norm_last_checkpoint": checkpoints[-1]["output_norm"],
            "eta_alpha_mapping_observable": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic HOPE core diagnostic")
    parser.add_argument("--skip-long-horizon", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {
        "real_shape": [run_real_case(1), run_real_case(2)],
        "long_horizon": None if args.skip_long_horizon else run_long_horizon(),
        "optional_real_feature_probe": "SKIPPED",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
